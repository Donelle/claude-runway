"""
Shared, dependency-light logic behind `tools/memory_bank_mcp_server.py`
(issue #175): `remember`/`recall`/`forget` persist durable, cross-session
lessons into ONE shared Qdrant collection across every project (not each
project's own code-index collection), tagged by writing project so lookups
stay precise by default while still allowing a deliberate cross-project
lookup.

Uses a raw `QdrantClient` + embedding provider directly, not
`QdrantConnector` (the class `find_in_collection`/`index_repo`/`sync_repo`
build on): the embedded field (`summary`) and the field returned on a hit
(`description`) are NOT the same field here, unlike `find_in_collection`'s
generic `content`-in/`content`-out pattern -- see `remember_point`'s payload
shape below.

`metadata` also carries `description` (verbatim stored text), `created_at`,
and `pending` (see `remember_point`/`wipe_memory_bank`) -- the four
CONTROL/classification fields below are the ones that gate access
(source/repo) or describe provenance (embedding_model) or open-ended
categorization (kind), only one of which (`kind`) is ever left to the
calling model's own judgment:
  - `source` -- always "memory-bank", fixed by this module, never
    model-settable. The structural "is this actually a memory-bank record"
    signal `index_repo(reset=true)`/`sync_repo` key their guard exclusions
    off of (see tools/ingest_mcp_server.py).
  - `repo` -- the writing project's own MEMORY_BANK_ID (a plain identifier,
    not necessarily a real Qdrant collection name), or the reserved
    GENERAL_REPO sentinel when the caller explicitly asked for
    general/cross-project scope. Also tool-resolved (see resolve_repo), not
    directly model-settable -- `general` can only ever steer this to the one
    fixed sentinel, never to spoof a specific other project's identity.
  - `embedding_model` -- diagnostic only. An earlier draft of this module
    gated `recall` results on this field per-point; that check was cut after
    review found it structurally almost unreachable (a stored point can only
    exist if its vector already matched the collection's fixed schema at
    write time) and, on the one path that WAS reachable, compared raw model-
    name strings rather than resolved vector name/dimension -- a real bug
    that made it more likely to raise a false positive than catch a genuine
    mismatch. The collection-level schema check (qdrant_model_check.py) is
    the actual defense against a real cross-model mismatch; this field is
    kept purely so a human can later see what model wrote a given point.
  - `kind` -- fully open, model-decided (no fixed enum).
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Optional

from qdrant_client import QdrantClient, models

from qdrant_retry import call_with_retry
from qdrant_model_check import check_embedding_model_mismatch

# Passed as check_embedding_model_mismatch's recovery_hint by every memory-bank
# call site (PR #178 review) -- the shared check's DEFAULT text tells a caller
# to "use index_repo(reset=true)"/mentions "find_in_collection", both wrong
# here: index_repo refuses to ever touch the shared memory-bank collection
# (see is_memory_bank_collection_name's callers in tools/ingest_mcp_server.py),
# and memory-bank isn't a cross-repo lookup.
MISMATCH_RECOVERY_HINT = (
    "This shared memory-bank collection was created with a different embedding "
    "model than this project's EMBEDDING_MODEL. Every project sharing it must "
    "agree on one model -- either change this project's EMBEDDING_MODEL to match "
    "whichever model created the collection, or (if you control every project "
    "using it) delete the collection directly in Qdrant and let the next "
    "remember() call recreate it under the new model."
)

SOURCE_FIELD = "metadata.source"
KIND_FIELD = "metadata.kind"
REPO_FIELD = "metadata.repo"
EMBEDDING_MODEL_FIELD = "metadata.embedding_model"
CREATED_AT_FIELD = "metadata.created_at"
PENDING_FIELD = "metadata.pending"
WEIGHT_FIELD = "metadata.weight"

MEMORY_BANK_SOURCE = "memory-bank"

# metadata.source value compress_mcp_server.compact_store stamps on every
# conversation-compact point (issue #282). Lives here beside
# MEMORY_BANK_SOURCE so the writer and the indexer's delete guard
# (memory_bank_exclusion_filter) share one constant and can't drift apart.
COMPACT_SOURCE = "conversation-compact"

# Default weight for a point with no metadata.weight at all (issue #177) --
# either a point written before this field existed, or one explicitly
# remembered with weight=1.0 (the default). Multiplicative against
# FLOORED similarity (see _normalize_similarity below -- effective_score =
# max(0, raw) * weight, issue #272), so this value is specifically chosen to
# be a true no-op: 1.0 preserves plain-similarity ORDERING exactly (a
# constant scale factor across every hit, with negative-similarity ties
# broken by raw score in recall_points).
DEFAULT_WEIGHT = 1.0

# recall_points over-fetches this many times `limit` from query_points BEFORE
# applying the weight multiplier and truncating (issue #177) -- re-ranking by
# effective_score = max(0, raw_score) * weight (see _normalize_similarity's
# docstring for the full semantics and why) has to happen against a wider raw-similarity pool than
# the final `limit`, or a genuinely high-weight but lower-raw-similarity
# match could get cut by Qdrant's own similarity-only ordering before
# re-ranking ever sees it. 3x is a fixed implementation choice (the ticket
# calls out the two-stage SHAPE -- fetch pool, re-rank, truncate -- as the
# spec decision, not this exact multiplier).
_RECALL_OVERFETCH_MULTIPLIER = 3


def _normalize_similarity(raw_score: float) -> float:
    """
    Floors a raw cosine similarity at 0 (`max(0.0, raw_score)`, no offset, no
    rescale) BEFORE it's multiplied by `weight`. FINAL WEIGHT SEMANTICS
    (issue #272; the contract a future `reweight_point`, #341, builds on):

        effective_score = max(0, raw_similarity) * weight

    i.e. `weight` is a PROPORTIONAL relevance multiplier. A hit's
    effective_score is always proportional to its own relevance, so:
      * a zero-or-negative-similarity hit scores exactly 0 at ANY weight --
        weight can never rescue an irrelevant memory (nor, for a finite
        weight, turn it into a top hit);
      * `weight=1.0` (the default, and any point with no `metadata.weight`)
        is a true no-op for ORDERING: effective_score equals max(0, raw), a
        monotonic function of raw similarity;
      * `weight=0` floors to exactly 0, the bottom of the ranking;
      * `weight=w>1` lets a hit beat a default-weight competitor iff
        `w * raw_w > raw_d`, e.g. weight=2.0 makes a hit with half the
        similarity TIE a default-weight one and beat any equally-similar
        one -- a boost among comparably-relevant hits, never a takeover.
    Hits that tie at effective_score 0 (all non-positive similarities, or
    weight=0) are ordered weight>0 hits first, then by raw similarity
    descending (see recall_points), so default-weight ordering is exactly
    plain-similarity ordering even in the negative range, and weight=0 hits
    always sit at the very bottom.

    History: issue #177 first multiplied the RAW score, which inverted for
    NEGATIVE similarities (weight=0 on score=-0.8 gave 0, ranking ABOVE a
    weight=1 hit at -0.1; weight>1 demoted). PR #199 fixed the sign with a
    `(raw+1)/2` rescale into [0, 1], but that put every realistic hit
    (raw>=0) in [0.5, 1.0]: weight=2.0 on raw=0.0 scored 1.0 and beat a
    default-weight raw=0.99 (0.995) -- an unconditional takeover, issue #272.
    Flooring at 0 with no offset keeps both sign fixes and removes the 0.5
    floor.

    This is an internal ranking mechanism only -- the `score` field in a
    recall_points hit dict stays the RAW, un-floored similarity Qdrant
    returned (unchanged meaning, see recall_points' own docstring for why).
    """
    return max(0.0, raw_score)

# Reserved metadata.repo value for knowledge that isn't tied to any one
# project -- set via remember(general=True). Collides, in principle, with a
# real project whose own MEMORY_BANK_ID happens to resolve to this exact
# string (e.g. setup_project.py defaults it from default_collection_name(),
# which slugifies a directory name, so a repo literally named "general"
# would produce this) -- resolve_repo() below refuses that case explicitly
# rather than silently aliasing a real project's memories onto the general
# bucket.
GENERAL_REPO = "general"

# Must match templates/mcp.json.template's literal MEMORY_BANK_ID
# placeholder text exactly -- an unedited project would otherwise silently
# tag every memory with this same string, indistinguishable from a real,
# unique project identifier and shared across every un-configured project.
_PLACEHOLDER_MEMORY_BANK_ID = "ANY-NAME-YOU-WANT"


def resolve_repo(memory_bank_id: Optional[str], general: bool) -> tuple:
    """
    Resolves the `metadata.repo` value a `remember`/`recall`/`forget`
    (`wipe_all`) call should use. Returns `(repo, None)` on success or
    `(None, error_message)` on failure -- mirrors this repo's
    `validate_chunk_params`-style "return an error string, let the caller
    decide how to surface it" convention (see qdrant_ingest_lib.py) rather
    than raising.

    general=True always resolves to the fixed GENERAL_REPO sentinel
    regardless of memory_bank_id -- the whole point of `general` is that
    it doesn't depend on which project is calling, and it can never resolve
    to any value other than that one fixed sentinel (so it can't be used to
    spoof a specific OTHER project's identity).

    general=False errors (rather than silently degrading) when:
    - memory_bank_id is falsy: no MEMORY_BANK_ID configured for this
      project's .mcp.json, mirroring index_repo/sync_repo's own "no
      collection specified" error.
    - memory_bank_id is still the unedited mcp.json.template placeholder:
      every project left unconfigured this way would otherwise silently
      share one `repo` tag, leaking memories across projects that never
      intended to share anything.
    - memory_bank_id == GENERAL_REPO: a real project's own identifier
      colliding with the reserved sentinel would make its memories
      indistinguishable from genuinely general ones.
    """
    if general:
        return GENERAL_REPO, None
    if not memory_bank_id:
        return None, (
            "Error: no collection specified and no MEMORY_BANK_ID env var "
            "configured for this project's .mcp.json -- memory-bank needs this "
            "to tag/scope entries by project. Set MEMORY_BANK_ID (any identifier "
            "unique to this project) for this project, or pass general=True to "
            "store this as general, cross-project knowledge instead."
        )
    if memory_bank_id == _PLACEHOLDER_MEMORY_BANK_ID:
        return None, (
            f"Error: MEMORY_BANK_ID is still the unedited template placeholder "
            f"('{_PLACEHOLDER_MEMORY_BANK_ID}') -- every project left unconfigured "
            f"this way would silently share the same memory-bank 'repo' tag. "
            f"Set a real, unique MEMORY_BANK_ID for this project first."
        )
    if memory_bank_id == GENERAL_REPO:
        return None, (
            f"Error: this project's MEMORY_BANK_ID is '{GENERAL_REPO}', which "
            f"collides with memory-bank's reserved general-knowledge tag. Rename "
            f"this project's MEMORY_BANK_ID to something else."
        )
    return memory_bank_id, None


def is_memory_bank_collection_name(collection: str, memory_bank_collection: str) -> bool:
    """
    True if `collection` (an index_repo/sync_repo target) matches the
    configured memory-bank collection name -- used by ingest_mcp_server.py
    to refuse indexing/syncing directly into the shared memory bank,
    regardless of reset/force, rather than only guarding the reset path.
    """
    return collection == memory_bank_collection


def memory_bank_exclusion_filter() -> models.Filter:
    """
    The reusable `must_not` fragment index_repo/sync_repo fold into their own
    delete filters, so a collection that ever ends up holding both code
    chunks and protected points (it shouldn't, given
    is_memory_bank_collection_name's guard, but this stays as
    defense-in-depth per issue #175) never has them caught by an unrelated
    delete. Protects memory-bank points (metadata.source == "memory-bank")
    and, since issue #282, conversation compacts (metadata.source ==
    "conversation-compact"). Compacts stored before #282 carry no
    metadata.source and stay unprotected until backfilled.
    """
    return models.Filter(
        must_not=[
            models.FieldCondition(key=SOURCE_FIELD, match=models.MatchValue(value=MEMORY_BANK_SOURCE)),
            models.FieldCondition(key=SOURCE_FIELD, match=models.MatchValue(value=COMPACT_SOURCE)),
        ]
    )


def _scope_filter(
    caller_repo: str,
    repo: Optional[str],
    all_repos: bool,
    kind: Optional[str] = None,
) -> models.Filter:
    """
    Builds the `metadata.source == "memory-bank"` (+ repo/kind) filter shared
    by recall/count. Precedence: an explicit `repo` override narrows to
    exactly that repo; `all_repos=True` (with no explicit `repo`) applies no
    repo constraint at all; otherwise (the default) scope is this project's
    own repo OR the general-knowledge sentinel -- `should` conditions in a
    Qdrant Filter are OR'd together and required (min_should_match defaults
    to 1) alongside `must`, giving `source == memory-bank AND (repo ==
    caller_repo OR repo == "general")`.

    Also excludes `pending == True` points (PR #178 review, seventh pass):
    `remember_point`'s point is visible to a search the instant its initial
    upsert returns, before its follow-up call clears `pending`/stamps
    `created_at` -- without this, `recall` could surface a record whose
    `description` (the verbatim text `remember` was given) is fully written,
    but which `wipe_memory_bank` cannot yet remove (pending points are always
    wipe-ineligible -- see `_own_repo_filter`). Excluding it from recall too
    means a point that is visibly searchable is *also* always wipeable --
    the two states move together instead of a window where one is true and
    not the other.
    """
    # Bare `list` (not `list[FieldCondition]`) so the type checker (mypy
    # originally, Pyright since issue #297 -- both treat this the same way)
    # sees these as list[Any] -- Filter's must/should accept a broader
    # Condition union, and list's invariance would otherwise reject a
    # narrower list[FieldCondition].
    must: list = [models.FieldCondition(key=SOURCE_FIELD, match=models.MatchValue(value=MEMORY_BANK_SOURCE))]
    must_not: list = [models.FieldCondition(key=PENDING_FIELD, match=models.MatchValue(value=True))]
    should: Optional[list] = None
    if repo:
        must.append(models.FieldCondition(key=REPO_FIELD, match=models.MatchValue(value=repo)))
    elif not all_repos:
        should = [
            models.FieldCondition(key=REPO_FIELD, match=models.MatchValue(value=caller_repo)),
            models.FieldCondition(key=REPO_FIELD, match=models.MatchValue(value=GENERAL_REPO)),
        ]
    if kind:
        must.append(models.FieldCondition(key=KIND_FIELD, match=models.MatchValue(value=kind)))
    return models.Filter(must=must, must_not=must_not, should=should)


def ensure_collection(client: "QdrantClient", collection: str, embedding_provider) -> Optional[str]:
    """
    Creates `collection` with a named-vector schema matching
    `embedding_provider` if it doesn't already exist. Returns an error string
    if the collection that ends up existing (whether already there, or won by
    a concurrent creator) doesn't actually match `embedding_provider`'s
    schema; None on success.

    Idempotent against a concurrent creator: multiple projects' sessions can
    plausibly call `remember` against the shared memory-bank collection for
    the first time at close to the same moment (e.g. right after a Qdrant
    wipe), so a bare create_collection here would race. A concurrent creator
    winning that race surfaces as an exception from Qdrant (typically a 409
    Conflict) -- re-checking existence before re-raising treats that as
    "the collection now exists" rather than a real failure.

    Found in PR #178 review: the previous version stopped there, silently
    treating "it exists now" as full success -- but the WINNING creator could
    be a different project configured with a DIFFERENT EMBEDDING_MODEL, in
    which case this process's next upsert would fail with an opaque Qdrant
    error (wrong/missing vector name) instead of the clear, actionable
    mismatch message this repo already has a function for. Re-validating the
    schema here -- on BOTH the immediate already-exists path and the
    race-recovery path -- closes that gap; any other, genuine
    create_collection error still propagates.
    """
    if call_with_retry(client.collection_exists, collection):
        return check_embedding_model_mismatch(client, collection, embedding_provider, recovery_hint=MISMATCH_RECOVERY_HINT)
    vector_name = embedding_provider.get_vector_name()
    vector_size = embedding_provider.get_vector_size()
    try:
        call_with_retry(
            client.create_collection,
            collection_name=collection,
            vectors_config={
                vector_name: models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            },
        )
    except Exception:
        if not call_with_retry(client.collection_exists, collection):
            raise
        return check_embedding_model_mismatch(client, collection, embedding_provider, recovery_hint=MISMATCH_RECOVERY_HINT)
    return None


def ensure_memory_bank_indexes(client: "QdrantClient", collection: str) -> None:
    """
    Idempotently backfills KEYWORD payload indexes on `source`/`repo` --
    mirrors qdrant_batch_store.ensure_file_path_index's "check current
    payload_schema before creating" pattern so re-running this doesn't
    reissue a full-collection-scanning create_payload_index call every time.
    These two cover the `source` match clause that every recall/count/wipe
    filter has, and the `repo` match clause wherever one is applied (not
    always: `_scope_filter` omits it for `all_repos=True`, and
    `count_memory_bank_points` omits it when `repo` is None), so those
    clauses no longer full-scan a collection shared across every project as
    it grows.

    They do NOT cover the other clauses: `metadata.pending` (the
    `must_not` exclusion in `_scope_filter`, `count_memory_bank_points` and
    `_own_repo_filter`), `metadata.kind` (`_scope_filter`'s optional `kind`
    match) and `metadata.created_at` (`_own_repo_filter`'s range /
    is-empty `created_before` clause) have no payload index, so Qdrant
    evaluates them per point. Those filters are therefore only partly
    index-assisted, not fully so; adding indexes for them is a separate,
    unmade decision (issue #310).
    """
    info = call_with_retry(client.get_collection, collection)
    existing = info.payload_schema or {}
    for field in (SOURCE_FIELD, REPO_FIELD):
        if field not in existing:
            call_with_retry(
                client.create_payload_index,
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )


async def remember_point(
    client: "QdrantClient",
    embedding_provider,
    collection: str,
    summary: str,
    description: str,
    kind: str,
    repo: str,
    embedding_model: str,
    weight: float = DEFAULT_WEIGHT,
    point_id: str | None = None,
) -> tuple:
    """
    Embeds `summary` (the only text actually searched) and stores it
    alongside `description` (verbatim, payload-only, never embedded) and the
    tool-set `source`/`repo`/`embedding_model` metadata.

    `weight` (issue #177) is a static, caller-supplied quality multiplier --
    `recall_points` re-ranks by `effective_score = max(0, raw_similarity) *
    weight` (see `_normalize_similarity`'s docstring for the exact semantics
    and why) instead of raw similarity alone, so a stale or superseded memory
    doesn't outrank a more trustworthy one purely by having more similar
    wording. weight is proportional: it never lifts an irrelevant memory
    (raw similarity <= 0 scores 0 at any weight), and weight=2.0 only lets
    a hit with half the similarity tie a default-weight one.
    Stored verbatim in `metadata.weight`; a point with no such field at all
    (written before this change) is treated as `weight=1.0` at read time by
    `recall_points` -- a true no-op, not a behavior change for existing data.

    `point_id` (issue #334, part of #327) is optional. When omitted (the
    default, and what `remember()`'s MCP tool always does) a fresh random
    `uuid.uuid4().hex` is minted exactly as before -- behavior is unchanged
    for every existing caller. When provided, that exact ID is used for the
    upsert instead, so a caller that needs deterministic per-source IDs (the
    legacy-collection transfer tool's idempotency, see #327) can get them.
    This function itself does NO check-before-write: upserting an existing ID
    overwrites it and re-stamps `created_at`/re-cycles `pending`, so a caller
    that wants a re-run to leave unchanged points alone must check first.

    Returns `(point_id, created_at, None)` on success, or
    `(None, None, error_message)` if `ensure_collection` finds this
    project's embedding model doesn't actually match the collection's schema
    (see that function's docstring for why this can happen even when the
    collection already existed) -- same `(value, error)` two-outcome
    convention `resolve_repo` uses, just with an extra success-only field.

    `created_at` (issue #179, PR #197 review) is the EXACT `time.time()`
    value this function itself stamped into `metadata.created_at` below --
    returned so a caller logging a memory-events row for this write (see
    `tools/memory_bank_mcp_server.py`'s `remember()`) can record the real
    creation timestamp instead of re-sampling `time.time()` after this
    function returns. Re-sampling would drift from the real stamped value by
    however long the upsert + set_payload round trips (plus any
    `call_with_retry` backoff) took -- not always negligible, and this field
    exists specifically to support a time-to-first-reuse metric that a
    drifted value would corrupt.

    Known residual gap, accepted as-is rather than built out further (PR #178
    review, seventh pass): if the initial upsert below succeeds but the
    follow-up `set_payload` call that clears `pending` fails even after
    `call_with_retry`'s retries are exhausted, that exception propagates out
    of this function uncaught -- the caller never receives `point_id`, so it
    has no way to `forget_point` the record directly, and the point is stuck
    `pending=True` forever: excluded from both `recall_points` (see
    `_scope_filter`) and `wipe_memory_bank` (see `_own_repo_filter`) with no
    tool-level path back. This requires the upsert to durably succeed and the
    immediately-following set_payload to fail all 4 attempts -- rare, and no
    worse than the alternative of NOT having a pending marker at all (a
    silent wipe-race data loss on every write, not just this narrow failure
    case) -- so a stale-pending reconciliation/TTL mechanism was deliberately
    not built for this. A stuck record is inert (unsearchable) rather than
    actively harmful; if cleanup is ever needed, it's a plain Qdrant payload
    filter (`metadata.pending == true`) any Qdrant client can query and
    delete directly, no different from an operator poking at any other
    stuck row in a database.
    """
    mismatch = ensure_collection(client, collection, embedding_provider)
    if mismatch:
        return None, None, mismatch
    ensure_memory_bank_indexes(client, collection)
    vector_name = embedding_provider.get_vector_name()
    embeddings = await embedding_provider.embed_documents([summary])
    if point_id is None:
        point_id = uuid.uuid4().hex
    # pending=True is written in this SAME atomic upsert that creates the
    # point -- there is no window where the point exists without it, unlike
    # created_at (stamped in a separate follow-up call below). wipe_memory_bank
    # unconditionally excludes any point with pending=True regardless of its
    # created_at state, so a point can never be swept up while it's between
    # "exists" and "fully stamped," no matter how that window's timing lines
    # up against a concurrent wipe's cutoff/scroll. Found in PR #178 review
    # (sixth pass): a two-step write (create, THEN stamp created_at
    # separately) closes the "timestamp read before upsert completes" race
    # from the previous round, but during the gap between those two calls
    # the point has no created_at at all -- and the existing IsEmptyCondition
    # fallback (added for points written before created_at existed at all)
    # treated that missing state as "definitely safe to wipe," which is
    # exactly backwards for a point that's mid-write right now. pending is a
    # POSITIVE marker for that specific in-between state, so it can be
    # excluded without disturbing the legacy-point fallback (which stays
    # keyed on created_at being absent -- a legacy point never has pending
    # set at all, so it's untouched by this check either way).
    call_with_retry(
        client.upsert,
        collection_name=collection,
        points=[
            models.PointStruct(
                id=point_id,
                vector={vector_name: embeddings[0]},
                payload={
                    "document": summary,
                    "metadata": {
                        "source": MEMORY_BANK_SOURCE,
                        "kind": kind,
                        "description": description,
                        "repo": repo,
                        "embedding_model": embedding_model,
                        "weight": weight,
                        "pending": True,
                    },
                },
            )
        ],
    )
    # created_at is stamped -- and pending cleared -- in a SEPARATE call,
    # made only AFTER the upsert above has returned (Qdrant's default
    # wait=True means that return already confirms durability). A value read
    # and embedded in the SAME payload as the initial write is necessarily
    # captured BEFORE that write's own round-trip completes, which is exactly
    # why this is a second call rather than one -- see the pending comment
    # above for why the gap between these two calls is still safe regardless.
    #
    # This does NOT fully close the created_at race, only shrink it further
    # (PR #178 review, ninth pass): time.time() here is STILL a client-side
    # read taken before THIS call's own RPC is sent, same pattern as the
    # original upsert-embedded version, just one level narrower -- if this
    # specific set_payload round-trip is delayed and a concurrent wipe's
    # cutoff lands in that exact gap, the point can still be swept up once
    # the delayed write finally applies, since its own created_at is
    # genuinely <= that cutoff by wall-clock time. Qdrant exposes no
    # server-assigned write-order primitive (a monotonic revision, a
    # server-stamped timestamp) this could tie to instead, so "the RPC's
    # own network round-trip" is the tightest bound achievable with a
    # client-side timestamp -- not a perfect guarantee, and not meant to be
    # read as one. In practice this is a millisecond-scale window between
    # two operations from the same caller (a remember() finishing at almost
    # the exact instant that same project's own wipe_all begins), not an
    # adversarial multi-tenant race.
    created_at = time.time()
    call_with_retry(
        client.set_payload,
        collection_name=collection,
        payload={"created_at": created_at, "pending": False},
        points=[point_id],
        key="metadata",
    )
    return point_id, created_at, None


async def recall_points(
    client: "QdrantClient",
    embedding_provider,
    collection: str,
    query: str,
    caller_repo: str,
    repo: Optional[str] = None,
    all_repos: bool = False,
    kind: Optional[str] = None,
    limit: int = 5,
) -> list:
    """
    Searches `summary` vectors, always scoped to `metadata.source ==
    "memory-bank"`, plus the repo/kind narrowing `_scope_filter` builds.
    Returns a list of dicts: id/summary/description/kind/repo/
    embedding_model/score/weight/effective_score/created_at, sorted by
    `effective_score` descending and truncated to `limit`. Returns [] (not an
    error) when the collection doesn't exist yet -- an empty memory bank is a
    normal, expected state, not a failure.

    `created_at` (issue #179) is the point's own `metadata.created_at`
    (a float `time.time()` value stamped by `remember_point`, or None for a
    legacy point written before that field existed) -- included so a caller
    logging a memory-events row for this hit can denormalize the memory's
    real creation time without a second Qdrant round trip.

    `weight`/`effective_score` (issue #177): `score` stays the RAW cosine
    similarity Qdrant returned (unchanged meaning, for backward compat with
    any existing caller reading it) -- `effective_score` is what re-ranking
    and truncation actually use, so raw similarity alone can't tell "most
    trustworthy" from "most similar wording." A point with no `metadata.weight`
    (written before this field existed) is treated as `weight=1.0`, a true
    no-op for ORDERING purposes among other default-weight hits:
    `effective_score = max(0, score) * weight` (issue #272; see
    `_normalize_similarity` for the full semantics), so at weight=1.0 it
    equals max(0, score) -- same ordering as raw similarity, though a
    negative `score` reports effective_score 0. `query_points` itself is
    asked for `limit * _RECALL_OVERFETCH_MULTIPLIER` candidates, not just
    `limit` -- re-ranking against only `limit` raw-similarity hits could
    never let a high-weight-but-lower-raw-similarity match rise above one
    Qdrant's own similarity-only ordering already cut before re-ranking saw
    it.

    Ties on `effective_score` (all non-positive-similarity hits score 0, as
    do weight=0 hits) are broken by weight>0 first (weight=0 is always last),
    then raw `score` descending, so default-weight ordering stays exactly
    plain-similarity ordering in the negative range.
    """
    if not call_with_retry(client.collection_exists, collection):
        return []
    vector_name = embedding_provider.get_vector_name()
    query_vector = await embedding_provider.embed_query(query)
    query_filter = _scope_filter(caller_repo, repo, all_repos, kind)
    fetch_limit = limit * _RECALL_OVERFETCH_MULTIPLIER
    response = call_with_retry(
        client.query_points,
        collection_name=collection,
        query=query_vector,
        using=vector_name,
        query_filter=query_filter,
        limit=fetch_limit,
        with_payload=True,
    )
    results = []
    for point in response.points:
        payload = point.payload or {}
        meta = payload.get("metadata") or {}
        weight = meta.get("weight")
        if weight is None:
            weight = DEFAULT_WEIGHT
        score = point.score
        results.append(
            {
                "id": point.id,
                "summary": payload.get("document", ""),
                "description": meta.get("description", ""),
                "kind": meta.get("kind"),
                "repo": meta.get("repo"),
                "embedding_model": meta.get("embedding_model"),
                "score": score,
                "weight": weight,
                "effective_score": _normalize_similarity(score) * weight,
                "created_at": meta.get("created_at"),
            }
        )
    # Sort key: effective_score, then weight>0, then raw score (all
    # descending). The raw-score tie-break keeps default-weight ordering equal
    # to plain-similarity ordering among all-negative hits (which all floor to
    # effective 0). The `weight > 0` middle term keeps weight=0 a true bottom:
    # without it a weight=0 hit with raw 0.95 would out-tie a default-weight
    # hit at raw -0.1 (both effective 0) on raw score alone (PR #359 review).
    results.sort(key=lambda r: (r["effective_score"], r["weight"] > 0, r["score"]), reverse=True)
    return results[:limit]


# Page size for scroll_collection (issue #335). Qdrant's own scroll default is
# 10; 1000 matches wipe_memory_bank's page size, keeping round trips low on a
# large legacy collection without one enormous response.
_SCROLL_BATCH_SIZE = 1000


def scroll_collection(client: "QdrantClient", source: str, batch_size: int = _SCROLL_BATCH_SIZE) -> tuple:
    """
    Read-only, COMPLETE-for-an-unchanged-collection enumeration of every point
    in `source` (issue #335, part of #327) -- for reading a foreign/legacy
    collection (e.g. one that predates the shared memory-bank collection) so
    its points can be audited or migrated. Loops `client.scroll` following
    `next_offset` until Qdrant returns None, so unlike `recall_points` it is
    a full walk rather than semantic top-k search, never silently capped at
    a top-k: no query text and no embedding provider are involved at all.
    Vectors are never fetched.

    Completeness caveat (PR #382 review): `scroll()` has no cross-page
    consistency guarantee (see wipe_memory_bank's docstring), so if points
    are inserted or deleted WHILE this runs, the result is not a snapshot --
    a concurrent insert can be missed or a concurrent delete still seen. The
    full-enumeration guarantee therefore holds only for a collection nothing
    is writing to (the normal case for a legacy/foreign source being
    audited or migrated); callers needing more must quiesce writers first.

    Returns `(points, errors)`:
      * `points`: one dict per well-shaped point -- id, summary (the
        `document` field), description, kind, repo, embedding_model,
        created_at, weight (DEFAULT_WEIGHT when absent) -- the same field
        names `recall_points` returns, minus the similarity scores.
      * `errors`: one `{"id": ..., "error": "..."}` per point that is NOT
        memory-bank-shaped (payload/metadata not a dict, or `metadata.repo`
        missing). Such a point is reported and skipped rather than crashing
        the whole scroll, since `source` may be any collection at all. A
        missing collection yields `([], [{"id": None, "error": ...}])`.

    Deliberately applies NO source/pending filter: the point is to see
    everything in `source`, and callers decide what to do with each point.
    """
    if not call_with_retry(client.collection_exists, source):
        return [], [{"id": None, "error": f"collection '{source}' does not exist."}]
    points: list = []
    errors: list = []
    offset = None
    while True:
        records, offset = call_with_retry(
            client.scroll,
            collection_name=source,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload
            if not isinstance(payload, dict):
                errors.append({"id": record.id, "error": "payload is missing or not an object."})
                continue
            meta = payload.get("metadata")
            if not isinstance(meta, dict):
                errors.append({"id": record.id, "error": "payload has no 'metadata' object."})
                continue
            repo = meta.get("repo")
            if not repo:
                errors.append({"id": record.id, "error": "metadata.repo is missing."})
                continue
            weight = meta.get("weight")
            points.append(
                {
                    "id": record.id,
                    "summary": payload.get("document", ""),
                    "description": meta.get("description", ""),
                    "kind": meta.get("kind"),
                    "repo": repo,
                    "embedding_model": meta.get("embedding_model"),
                    "weight": DEFAULT_WEIGHT if weight is None else weight,
                    "created_at": meta.get("created_at"),
                }
            )
        if offset is None:
            break
    return points, errors


# Fixed namespace for transfer_point_id (issue #336). Never change it: every
# id a past transfer_points run wrote was derived from this exact value, so a
# new namespace would make a re-run see none of them as already present and
# write every point a second time under a fresh id.
TRANSFER_ID_NAMESPACE = uuid.UUID("6f1c2b7e-4d3a-5e8f-9a0b-1c2d3e4f5a6b")


def transfer_point_id(source: str, source_point_id) -> str:
    """
    Deterministic target id for one source point (issue #336, see #327):
    `uuid5(TRANSFER_ID_NAMESPACE, "<source>:<source_point_id>")`. The same
    source point always maps to the same target id, so a re-run of
    `transfer_points` can tell "already migrated" apart from "new" with a
    plain `retrieve` instead of duplicating every point under a fresh random
    id. Qdrant ids are either unsigned ints or UUIDs, so an int id `5` and a
    UUID string can never format to the same `<source>:<id>` text.
    """
    return str(uuid.uuid5(TRANSFER_ID_NAMESPACE, f"{source}:{source_point_id}"))


def _transfer_field_error(point: dict) -> Optional[str]:
    """
    Checks the fields `remember_point` will write for one scrolled point, so a
    legacy point that is memory-bank-shaped enough to have a `repo` (which is
    all `scroll_collection` checks) but not enough to be written lands in the
    report's `errors` bucket, not in the shared collection. Same weight rule
    `remember()` enforces at its own tool boundary (finite, >= 0): an `inf`/
    `nan`/negative weight copied verbatim would break `recall_points`'
    ranking for every project sharing the collection.
    """
    summary = point.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return "summary (payload 'document') is missing or empty -- nothing to embed."
    if not isinstance(point.get("kind"), str) or not point["kind"]:
        return "metadata.kind is missing or not a string."
    if not isinstance(point.get("description"), str):
        return "metadata.description is not a string."
    weight = point.get("weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        return f"metadata.weight is not a number (got {weight!r})."
    if not math.isfinite(weight) or weight < 0:
        return f"metadata.weight must be a finite number 0 or greater (got {weight})."
    return None


async def transfer_points(
    client: "QdrantClient",
    embedding_provider,
    source: str,
    target: str,
    caller_repo: str,
    embedding_model: str,
    dry_run: bool = True,
) -> dict:
    """
    Migrates the in-scope points of a legacy/foreign collection `source` into
    the shared memory-bank collection `target` (issue #336, part 3 of #327's
    design). Returns a report dict; never raises for a per-point problem.

    Repo-tag mapping, per point -- the same boundary `resolve_repo` enforces
    for a single `remember()` call, applied point by point, with no bypass:
      * `metadata.repo == caller_repo` -> written project-scoped (repo=caller_repo).
      * `metadata.repo == GENERAL_REPO` -> written as general.
      * anything else -> NOT written, counted in `skipped_foreign_repo` by tag,
        so the user can run this same tool from a session in that repo.
    A point is never written under a tag other than the one it already had.

    Idempotency is check-before-write, not blind re-upsert: each point's
    target id comes from `transfer_point_id`, the target is `retrieve`d for
    those ids first, and only ids not already there are written. Re-upserting
    an existing id would re-stamp `created_at` and re-cycle `pending` on a
    point that hasn't changed (see `remember_point`'s docstring), so already
    present points are counted in `already_present` and left alone. Each new
    point gets a fresh `created_at` (the time of migration) and is re-embedded
    with this project's `embedding_provider`; the source's own vectors and
    timestamps are not copied. Dedup only recognizes ids this function
    derived: a memory copied over earlier by hand via `remember()` has a
    random id and is NOT detected (no content matching is attempted), so a
    real run would write a second copy of it -- the dry-run report is how a
    caller checks for that before writing.

    `dry_run=True` (the default) does everything except the write and reports
    `would_migrate` instead of `migrated`; it never embeds, so
    `embedding_provider` may be None there. A dry run cannot detect an
    embedding-model mismatch on the target, which only shows up when the
    first real write calls `ensure_collection`; that mismatch is
    collection-wide, so it stops the run and is returned as `error` (with the
    counts reached so far), rather than being repeated once per point.

    Read-only against `source`: nothing here ever writes to or deletes from
    it. Deleting a legacy collection stays a separate, manual action.

    Report keys: source_collection, target_collection, dry_run,
    `would_migrate` or `migrated` ({"general": n, "project": n}),
    skipped_foreign_repo ({tag: n}), already_present (n), errors (list of
    {"id", "error"}), and `error` (str, only when the run stopped early).
    """
    bucket = "would_migrate" if dry_run else "migrated"
    report: dict = {
        "source_collection": source,
        "target_collection": target,
        "dry_run": dry_run,
        bucket: {"general": 0, "project": 0},
        "skipped_foreign_repo": {},
        "already_present": 0,
        "errors": [],
    }
    # Compared by BACKING collection, not by name (PR #422 review): Qdrant
    # accepts an alias anywhere a collection name goes (collection_exists and
    # scroll both resolve it -- confirmed live), so an alias of the memory-bank
    # collection would pass a plain name check, scroll memory-bank itself, and
    # write copies of its points back into it under new ids; each later run
    # would then copy those copies again. Fails closed: if the alias list
    # can't be read, nothing is scrolled or written.
    #
    # Every Qdrant call below then uses these RESOLVED backing names, never
    # the requested ones (PR #422 review, second pass): an alias can be
    # repointed atomically at any moment, so checking resolved names but then
    # scrolling/writing through the alias would leave a window where the
    # check passes and the alias is switched to memory-bank (copying it into
    # itself) or the target alias is switched to the source (writing into the
    # supposedly read-only source). The requested `source` name is still
    # what transfer_point_id hashes, so ids stay stable across runs however
    # the source is reached.
    try:
        aliases = {
            a.alias_name: a.collection_name
            for a in call_with_retry(client.get_aliases).aliases
        }
    except Exception as e:
        report["error"] = f"could not read Qdrant collection aliases to check source vs. target: {e}"
        return report
    source_backing = aliases.get(source, source)
    target_backing = aliases.get(target, target)
    if source_backing == target_backing:
        report["error"] = (
            f"source_collection '{source}' is the memory-bank collection itself "
            f"(directly or through an alias) -- name the legacy collection to transfer from."
        )
        return report
    if not call_with_retry(client.collection_exists, source_backing):
        report["error"] = (
            f"collection '{source}' does not exist. Name it exactly "
            f"(codebase-indexer's list_collections shows what exists)."
        )
        return report

    points, scroll_errors = scroll_collection(client, source_backing)
    report["errors"].extend(scroll_errors)

    # (target_id, scope, point) for every point that passes the repo boundary
    # and field checks -- foreign-tagged points never get this far, so they
    # are never even looked up in the target.
    candidates: list = []
    for point in points:
        repo = point["repo"]
        if repo == GENERAL_REPO:
            scope = "general"
        elif repo == caller_repo:
            scope = "project"
        else:
            tag = str(repo)
            report["skipped_foreign_repo"][tag] = report["skipped_foreign_repo"].get(tag, 0) + 1
            continue
        field_error = _transfer_field_error(point)
        if field_error:
            report["errors"].append({"id": point["id"], "error": field_error})
            continue
        candidates.append((transfer_point_id(source, point["id"]), scope, point))

    # One retrieve per batch instead of one per point. A target that doesn't
    # exist yet holds nothing, so there is nothing to look up.
    #
    # Only a FINISHED write counts as present (PR #422 review, second pass):
    # remember_point upserts with pending=True and clears it in a second
    # call, so a run whose second call failed leaves the id in the target
    # with pending=True -- hidden from recall and wipe (see _scope_filter /
    # _own_repo_filter). Counting that as already_present would make it
    # permanent; leaving it out lets the next run rewrite the same id and
    # finish it. Hence payload is fetched (the pending flag only).
    present: set = set()
    if candidates and call_with_retry(client.collection_exists, target_backing):
        ids = [c[0] for c in candidates]
        for start in range(0, len(ids), _SCROLL_BATCH_SIZE):
            found = call_with_retry(
                client.retrieve,
                collection_name=target_backing,
                ids=ids[start:start + _SCROLL_BATCH_SIZE],
                with_payload=[PENDING_FIELD],
                with_vectors=False,
            )
            for record in found:
                meta = (record.payload or {}).get("metadata")
                if isinstance(meta, dict) and meta.get("pending") is True:
                    continue
                present.add(str(record.id))

    for target_id, scope, point in candidates:
        if target_id in present:
            report["already_present"] += 1
            continue
        if dry_run:
            report[bucket][scope] += 1
            continue
        try:
            _, _, mismatch = await remember_point(
                client, embedding_provider, target_backing,
                summary=point["summary"],
                description=point["description"],
                kind=point["kind"],
                repo=GENERAL_REPO if scope == "general" else caller_repo,
                embedding_model=embedding_model,
                weight=point["weight"],
                point_id=target_id,
            )
        except Exception as e:  # one bad point must not abort the whole run
            report["errors"].append({"id": point["id"], "error": f"write failed: {e}"})
            continue
        if mismatch:
            report["error"] = mismatch
            return report
        report[bucket][scope] += 1
    return report


def count_memory_bank_points(client: "QdrantClient", collection: str, repo: Optional[str] = None) -> int:
    """
    Counts memory-bank points in `collection`, optionally scoped to `repo`.
    Returns 0 if the collection doesn't exist. Used by `wipe_memory_bank`'s
    pre-confirm count (repo=caller's own) -- NOT by `index_repo(reset=true)`
    anymore (PR #178 review): that guard used to call this with repo=None
    first to decide between a full `delete_collection` and a filtered one,
    but the count-then-act sequence was racy against a concurrent write, so
    `index_repo(reset=true)` now always does the filtered delete unconditionally
    instead of counting at all.

    Excludes `pending == True` points (PR #178 review, seventh pass): without
    this, a dry-run (`confirm=False`) call could report a count that includes
    a still-pending point, while the immediately following `confirm=True`
    call excludes that same point via `_own_repo_filter` -- the dry-run number
    and the confirmed delete could then disagree even with no time passing
    and no concurrent writer at all, purely because the two paths applied
    different eligibility rules to the exact same snapshot.
    """
    if not call_with_retry(client.collection_exists, collection):
        return 0
    must: list = [models.FieldCondition(key=SOURCE_FIELD, match=models.MatchValue(value=MEMORY_BANK_SOURCE))]
    if repo:
        must.append(models.FieldCondition(key=REPO_FIELD, match=models.MatchValue(value=repo)))
    must_not: list = [models.FieldCondition(key=PENDING_FIELD, match=models.MatchValue(value=True))]
    result = call_with_retry(
        client.count,
        collection_name=collection,
        count_filter=models.Filter(must=must, must_not=must_not),
        exact=True,
    )
    return result.count


def forget_point(client: "QdrantClient", collection: str, point_id: str, caller_repo: str, confirm: bool) -> str:
    """
    Deletes exactly one point by id. Refuses (error string, no delete) if the
    point doesn't exist or isn't actually a memory-bank point -- `forget`
    must never touch a code-index chunk even if handed its id directly.

    If the point's own `repo` differs from `caller_repo` (this includes a
    "general"-tagged point -- general knowledge is never "this project's
    own"), deleting it requires confirm=True: a cross-repo id can only reach
    here via an explicit recall(repo=...)/recall(all_repos=True) call, so
    this is the one place a copy-pasted or hallucinated id from a DIFFERENT
    project could otherwise be deleted in a single unconfirmed call. A
    same-repo delete (the normal case -- the id came from this project's own
    recall) needs no confirm, unchanged from the original spec.
    """
    if not call_with_retry(client.collection_exists, collection):
        return f"Error: collection '{collection}' does not exist."
    try:
        records = call_with_retry(client.retrieve, collection_name=collection, ids=[point_id], with_payload=True)
    except Exception:
        # A malformed/non-UUID point_id (e.g. a hallucinated string) raises a
        # raw Qdrant 400 from retrieve() rather than returning an empty list --
        # treat any retrieve failure the same as "not found" instead of letting
        # a raw exception escape this tool call.
        records = []
    if not records:
        return f"Error: no point with id '{point_id}' found in '{collection}'."
    payload = records[0].payload or {}
    meta = payload.get("metadata") or {}
    if meta.get("source") != MEMORY_BANK_SOURCE:
        return f"Error: point '{point_id}' is not a memory-bank point -- refusing to delete it."
    point_repo = meta.get("repo")
    if point_repo != caller_repo and not confirm:
        return (
            f"This point belongs to repo '{point_repo}', not your own ('{caller_repo}'). "
            f"Pass confirm=True to delete a memory belonging to a different repo."
        )
    call_with_retry(
        client.delete,
        collection_name=collection,
        points_selector=models.PointIdsList(points=[point_id]),
    )
    return f"Deleted point '{point_id}' (repo='{point_repo}')."


def _own_repo_filter(caller_repo: str, created_before: Optional[float] = None) -> models.Filter:
    """
    Always excludes `pending == True` points (PR #178 review, sixth pass):
    that marker covers the brief window between remember_point's initial
    upsert (which creates the point, including pending=True, atomically)
    and its follow-up call that clears pending while stamping `created_at`
    -- a point can never be swept up while it's in that window, regardless
    of how its timing lines up against a wipe's cutoff/scroll. A legacy
    point (written before `pending` existed) never has this field set at
    all, so it's unaffected by this exclusion either way.

    `created_before`, if given, additionally requires `created_at <= created_before`
    OR the field is missing entirely (via `should`, alongside the `must` conditions --
    see wipe_memory_bank's docstring for why the cutoff exists, and why a missing
    field must still match: a point written before this field existed necessarily
    predates any cutoff computed after this fix shipped, so excluding it would be a
    pure regression with no safety benefit).
    """
    must: list = [
        models.FieldCondition(key=SOURCE_FIELD, match=models.MatchValue(value=MEMORY_BANK_SOURCE)),
        models.FieldCondition(key=REPO_FIELD, match=models.MatchValue(value=caller_repo)),
    ]
    must_not: list = [models.FieldCondition(key=PENDING_FIELD, match=models.MatchValue(value=True))]
    if created_before is None:
        return models.Filter(must=must, must_not=must_not)
    return models.Filter(
        must=must,
        must_not=must_not,
        should=[
            models.FieldCondition(key=CREATED_AT_FIELD, range=models.Range(lte=created_before)),
            models.IsEmptyCondition(is_empty=models.PayloadField(key=CREATED_AT_FIELD)),
        ],
    )


def wipe_memory_bank(client: "QdrantClient", collection: str, caller_repo: str, confirm: bool) -> tuple:
    """
    Bulk-clears this project's OWN memory-bank points -- filter is
    `source == "memory-bank" AND repo == caller_repo`, which deliberately
    EXCLUDES "general"-tagged points even though they're visible from this
    project's own recall: clearing one project's own bank shouldn't take
    shared/global knowledge down with it. A general memory can only be
    removed individually via forget_point (which requires confirm=True for
    it, since its repo is "general", never a caller's own).

    Returns (count, deleted): `deleted` is only True when confirm=True
    actually triggered a delete -- a call without confirm=True only counts,
    never mutates anything.

    `confirm=True` snapshots the matching point ids first (via `scroll`),
    then deletes precisely those ids, and `count` is `len(ids)` -- not a
    separate `count()` call taken before the delete. Found in PR #178
    review: the previous version called `count_memory_bank_points` (one
    Qdrant request) and then deleted by the same live FILTER (a second,
    separate request) -- if a new same-repo point arrived in between, the
    filter-based delete would remove it too, but the EARLIER count wouldn't
    reflect it, so the number this function reported could understate what
    it actually just deleted. Deleting the exact snapshotted ids instead
    means `count` reflects the SNAPSHOT this call took, not a live filter
    evaluated separately at delete time.

    `count` is this snapshot's size, not a guaranteed-exact count of points
    physically removed (PR #178 review, sixth-and-a-half pass): Qdrant's
    delete-by-id tolerates an already-missing id without error, so if some
    OTHER concurrent actor (a `forget_point` call, another `wipe_memory_bank`
    call) removes one of these exact snapshotted ids in the narrow gap
    between this call's `scroll` and its `delete`, that id is silently a
    no-op here -- `len(ids)` still reports the snapshot size, slightly
    overstating what this specific call physically deleted. The guarantee
    that actually matters is preserved regardless: nothing outside this
    snapshot (in particular, nothing created after it was taken) is ever
    touched -- see the `created_at`/`pending` discussion below for that.

    This does NOT make the count shown by an EARLIER, separate dry-run call
    (`confirm=False`) a binding contract -- a new point can still arrive
    between that call and this one, since they're two independent tool
    calls with no state carried between them. That gap is inherent to any
    two-step confirm UX spanning separate calls (visible in real time via
    each call's own honestly-reported count, never silently hidden) rather
    than something this function alone can close -- see PR #178's review
    thread for the full reasoning on why a heavier cross-call snapshot
    mechanism wasn't added on top of this.

    WITHIN this one call, though, the snapshot ids are additionally bounded
    by a `created_at` cutoff captured before scrolling starts (PR #178
    review, second wipe-related pass): `scroll()` has no cross-page
    consistency guarantee, and point ids are random UUIDs (not
    insertion-ordered), so once a repo has enough memories to need more
    than one page (>1000), a `remember()` landing mid-scroll could sort
    into a not-yet-visited page and get swept up despite being created
    after this wipe began. The cutoff closes that regardless of paging.

    A `remember()` call whose point exists but is still `pending` (between
    its initial upsert and the follow-up call that stamps `created_at`) is
    excluded unconditionally, regardless of the cutoff comparison (PR #178
    review, sixth pass) -- see `remember_point`'s and `_own_repo_filter`'s
    own docstrings for why a missing `created_at` alone isn't enough to
    tell "legacy point, definitely safe" apart from "brand new point,
    mid-write, definitely NOT safe."
    """
    if not confirm:
        return count_memory_bank_points(client, collection, repo=caller_repo), False
    if not call_with_retry(client.collection_exists, collection):
        return 0, True
    ids: list = []
    offset = None
    cutoff = time.time()
    scroll_filter = _own_repo_filter(caller_repo, created_before=cutoff)
    while True:
        records, offset = call_with_retry(
            client.scroll,
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=1000,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        ids.extend(r.id for r in records)
        if offset is None:
            break
    if ids:
        call_with_retry(
            client.delete,
            collection_name=collection,
            points_selector=models.PointIdsList(points=ids),
        )
    return len(ids), True
