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

MEMORY_BANK_SOURCE = "memory-bank"

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
    The reusable `must_not: metadata.source == "memory-bank"` fragment
    index_repo/sync_repo fold into their own delete filters, so a collection
    that ever ends up holding both code chunks and memory-bank points (it
    shouldn't, given is_memory_bank_collection_name's guard, but this stays
    as defense-in-depth per issue #175) never has its memory-bank points
    caught by an unrelated delete.
    """
    return models.Filter(
        must_not=[models.FieldCondition(key=SOURCE_FIELD, match=models.MatchValue(value=MEMORY_BANK_SOURCE))]
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
    # Bare `list` (not `list[FieldCondition]`) so mypy treats these as
    # list[Any] -- Filter's must/should accept a broader Condition union,
    # and list's invariance would otherwise reject a narrower list[FieldCondition].
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
    Without these, every recall/count/wipe filter on a collection shared
    across every project full-scans it as it grows.
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
) -> tuple:
    """
    Embeds `summary` (the only text actually searched) and stores it
    alongside `description` (verbatim, payload-only, never embedded) and the
    tool-set `source`/`repo`/`embedding_model` metadata.

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
    embedding_model/score/created_at. Returns [] (not an error) when the
    collection doesn't exist yet -- an empty memory bank is a normal,
    expected state, not a failure.

    `created_at` (issue #179) is the point's own `metadata.created_at`
    (a float `time.time()` value stamped by `remember_point`, or None for a
    legacy point written before that field existed) -- included so a caller
    logging a memory-events row for this hit can denormalize the memory's
    real creation time without a second Qdrant round trip.
    """
    if not call_with_retry(client.collection_exists, collection):
        return []
    vector_name = embedding_provider.get_vector_name()
    query_vector = await embedding_provider.embed_query(query)
    query_filter = _scope_filter(caller_repo, repo, all_repos, kind)
    response = call_with_retry(
        client.query_points,
        collection_name=collection,
        query=query_vector,
        using=vector_name,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    results = []
    for point in response.points:
        payload = point.payload or {}
        meta = payload.get("metadata") or {}
        results.append(
            {
                "id": point.id,
                "summary": payload.get("document", ""),
                "description": meta.get("description", ""),
                "kind": meta.get("kind"),
                "repo": meta.get("repo"),
                "embedding_model": meta.get("embedding_model"),
                "score": point.score,
                "created_at": meta.get("created_at"),
            }
        )
    return results


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
