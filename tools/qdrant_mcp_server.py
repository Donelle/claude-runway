#!/usr/bin/env python3
"""
The `qdrant` MCP server: this project's own semantic memory plus the
conversation-compact store, in one process.

Five tools:

  - `qdrant-find` / `qdrant-store`: search and save notes in THIS project's own
    collection (`COLLECTION_NAME`). These keep the names the third-party
    Qdrant MCP server this one replaced served, so `mcp__qdrant__qdrant-find` and every permission rule written against it
    keep working. Their descriptions live in the docstrings below instead of
    in `TOOL_FIND_DESCRIPTION`/`TOOL_STORE_DESCRIPTION` env entries.
  - `compact_store` / `compact_find` / `compact_prune`: the conversation-compact
    store `/my-compact` and `/my-resume` use, moved here from the
    `local-compress` server. They share this server's Qdrant connection and
    embedding code. `compact_store` also compresses the summary it is given
    through `libs/local_compress_lib.py` before storing it (see below).

On-disk format is unchanged. Code-index points are written by
`libs/qdrant_connector.py` (one named cosine vector, payload
`{document, metadata}`), and compacts keep their own payload fields and the
model-named vector `fast-<model basename>` they were always stored under, so
collections written before this server existed stay readable.

Two embedding models, on purpose. `EMBEDDING_MODEL` is the code-index model and
must agree with the `codebase-indexer` block. `COMPACT_EMBEDDING_MODEL` is the
conversation-compact model: compacts are stored under a model-named vector, so
changing it for an install that already has compacts orphans them. When it is
unset it falls back to `BAAI/bge-small-en`, which every compact stored before
this variable existed was embedded with.

LM Studio is optional here. `compact_store` asks the local model to compress
the summary and, if LM Studio is unavailable (or `openai` is not installed, as
on a Qdrant-only install), stores it uncompressed and says so in its reply.
The model client comes from `libs/lmstudio_client.py`, imported only when a
compact is actually stored.

Embedding models are loaded on first use, not at startup, so the server
answers `initialize` immediately whatever the network or model cache looks like.
"""

import json
import os
import sys
import uuid
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from mcp.server.mcpserver import MCPServer
from qdrant_client import QdrantClient, models as qdrant_models

from local_compress_lib import (
    COMPACT_FOCUS,
    compress as _compress_impl,
    derive_compact_label,
    strip_total_outage_notice,
)
from memory_bank_lib import COMPACT_SOURCE
from qdrant_compact_marker import mark_compact_collection
from qdrant_connector import Entry, QdrantConnector
from qdrant_embeddings import FastEmbedProvider
from qdrant_ingest_lib import ensure_persistent_fastembed_cache
from qdrant_model_check import check_embedding_model_mismatch
from qdrant_retry import call_with_retry

# Must run before any FastEmbedProvider(...) is constructed: fastembed reads
# FASTEMBED_CACHE_PATH at construction time, and its unset default is a
# wipeable OS temp dir. A value the user already exported wins (setdefault).
ensure_persistent_fastembed_cache()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# An empty string in .mcp.json means "no key"; normalize it so no auth header is sent.
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION_NAME = os.environ.get("COLLECTION_NAME") or None
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
# How many hits qdrant-find returns. The package this replaces defaulted to 10.
SEARCH_LIMIT = 10

# The names the compact tools and their helpers below are written against; the
# compact code was moved here unchanged, so these keep its original spelling.
COMPACT_QDRANT_URL = QDRANT_URL
COMPACT_QDRANT_API_KEY = QDRANT_API_KEY
COMPACT_COLLECTION = os.environ.get("COMPACT_COLLECTION", "conversation-compacts")

# Fallback when COMPACT_EMBEDDING_MODEL is unset: the historical default from
# qdrant-client's QdrantFastembedMixin (the QdrantClient.add()/.query()
# convenience methods compact_store/compact_find used to call directly,
# deprecated since ~1.7 and removed in 1.19.0). Neither tool ever called
# set_model(), so every compact stored before this setting existed was embedded
# with exactly this model, under the vector name
# FastEmbedProvider.get_vector_name() derives from it ("fast-bge-small-en").
# Keeping it as the unset-fallback means nobody who sets nothing sees any
# change; `upgrade` writes it explicitly for existing installs.
LEGACY_COMPACT_EMBEDDING_MODEL = "BAAI/bge-small-en"

# CHANGING it for an existing install orphans that install's compacts: points
# live under a model-named vector, so a collection created under one model is
# unreadable under another. compact_store/compact_find run
# check_embedding_model_mismatch so that fails loudly instead of silently
# creating a second vector or missing existing points.
COMPACT_EMBEDDING_MODEL = os.environ.get("COMPACT_EMBEDDING_MODEL") or LEGACY_COMPACT_EMBEDDING_MODEL

# Replaces the shared check's default find_in_collection-oriented advice
# ("check the other repo's .mcp.json"), which is wrong for compacts.
COMPACT_MODEL_MISMATCH_HINT = (
    "Compacts are stored under the embedding model that created their collection, so this "
    "server's COMPACT_EMBEDDING_MODEL (qdrant env block in .mcp.json) must match it. Set "
    "COMPACT_EMBEDDING_MODEL back to the model the collection was created with (BAAI/bge-small-en "
    "for installs that predate this setting). Changing the model for an existing collection is "
    "not supported: it orphans the stored compacts."
)

mcp = MCPServer("qdrant")

# The connector (and with it the embedding model) is built on the first
# qdrant-find/qdrant-store call, not at import: loading the ONNX model takes
# seconds and, on a cold cache, a download, and none of that should delay the
# server answering `initialize`.
_connector: Optional[QdrantConnector] = None


def _get_connector() -> QdrantConnector:
    global _connector
    if _connector is None:
        _connector = QdrantConnector(
            QDRANT_URL,
            QDRANT_API_KEY,
            COLLECTION_NAME,
            FastEmbedProvider(EMBEDDING_MODEL),
        )
    return _connector


def _format_entry(entry: Entry) -> str:
    """One search hit, in the shape the package this server replaces returned,
    so a caller or prompt that learned to read it keeps working."""
    entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
    return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"


_NO_COLLECTION_ERROR = (
    "Error: COLLECTION_NAME is not set in this server's env block (.mcp.json), so there is no "
    "project collection to use. Run `claude-runway-setup init` or set it by hand."
)


@mcp.tool(name="qdrant-find")
async def qdrant_find(query: str) -> list[str]:
    """
    Semantically search previously stored project knowledge (code locations,
    design decisions, past investigation findings) for THIS project. Use this
    BEFORE grepping or reading files for conceptual questions like 'where is X
    handled' or 'why was Y built this way'. Do NOT use for exact string, symbol,
    or error-message matches -- use Grep for those. Returns stored entries as
    text, including any metadata (e.g. file_path, line_range) attached when
    stored -- use that metadata to read only the relevant lines instead of the
    whole file. Input: query (a natural-language question, not a code snippet).
    """
    if not COLLECTION_NAME:
        return [_NO_COLLECTION_ERROR]
    entries = await _get_connector().search(query, collection_name=COLLECTION_NAME, limit=SEARCH_LIMIT)
    if not entries:
        return [f"No information found for the query '{query}'"]
    return [f"Results for the query '{query}'"] + [_format_entry(e) for e in entries]


@mcp.tool(name="qdrant-store")
async def qdrant_store(information: str, metadata: Optional[dict[str, Any]] = None) -> str:
    """
    Persist a distilled finding for this project's future sessions -- e.g. a
    bug's root cause, a design decision and its rationale, or the answer to a
    search that took real exploration to find. Store a short summary, not raw
    file dumps. ALWAYS include file_path and line_range (or source/url for
    non-code) in metadata. Do not store speculative or in-progress reasoning.
    Input: information (concise summary), metadata (JSON, must include
    file_path/line_range or source when applicable).
    """
    if not COLLECTION_NAME:
        return _NO_COLLECTION_ERROR
    await _get_connector().store(Entry(content=information, metadata=metadata), collection_name=COLLECTION_NAME)
    return f"Remembered: {information} in collection {COLLECTION_NAME}"


def _normalize_project(project: str) -> str:
    """
    Case-fold a project string (issue #37), so "MyProject" and "myproject"
    derive the identical CANONICAL collection name via _sanitize_project
    (its only caller). casefold() (not .lower()) is used deliberately --
    it's the Unicode-aware form intended for caseless comparison (e.g. it
    correctly equates German "ß" with "ss", which .lower() does not).

    IMPORTANT (round 2 of issue #37, flagged in PR review on #123 as a
    docstring/implementation mismatch that a prior version of this
    function's docs left in place): this is NOT called anywhere else in
    compact_store/compact_find, and cross-casing identity for a project
    with EXISTING history is deliberately NOT handled by normalizing a
    stored value at all -- compact_store writes the RAW `project` argument
    into both the payload's `project` field and _compact_point_id, and an
    explicit `collection` override on compact_find filters on the RAW
    value too. Case-insensitive matching against existing data instead
    happens entirely via case-insensitive COMPARISON at read/resolution
    time (see _collections_for_project, and the per-entry `.casefold()`
    filtering in compact_find's default path) -- there is no single
    "normalized form" that every stored entry is expected to already be
    in, since entries written under different historical casings
    legitimately coexist and are matched by comparison, not by rewriting
    history to agree with each other.
    """
    return project.casefold()


def _sanitize_project(project: str) -> str:
    """
    Replace characters not valid in Qdrant collection names with hyphens.
    No hash suffix is appended: this helper derives the same, readable
    canonical name for the same project string. It only DERIVES that name;
    compact_store still reuses an existing collection that already holds the
    project's history (see _collections_for_project, which can be a legacy-named
    one) and falls back to this canonical name only for a brand-new project.

    Case-folds `project` FIRST (issue #37), before the hyphen substitution,
    so "MyProject" and "myproject" collapse onto the identical sanitized
    string. Without this, a casing drift between /my-compact and
    /my-resume (different terminal profile, a renamed directory recreated
    with different casing, etc.) pointed at an entirely different,
    nonexistent collection, producing a misleading "no collection found"
    message instead of just finding the right one.

    Deterministic (same input -> same output every call) is required here:
    compact_store and compact_find each derive the canonical name
    independently and must agree on it for the same project, so a brand-new
    project's /my-resume finds what /my-compact just stored.
    """
    import re as _re
    return _re.sub(r"[^a-zA-Z0-9_-]", "-", _normalize_project(project))


def _collections_for_project(client: QdrantClient, project: str) -> list:
    """
    Every existing '<COMPACT_COLLECTION>-*' collection that already holds
    THIS logical project's history, case-insensitively.

    Round 2 of issue #37 (flagged in PR review on #123): an earlier
    version of this fix picked exactly ONE collection per call (a
    "legacy vs. canonical" choice), which still silently fragmented
    history across a genuine casing drift -- reproduced directly: a
    pre-existing "MyProject" collection plus a later /my-compact call
    passing "myproject" landed on the NEW case-folded canonical
    collection (since a lowercase input's legacy-style and canonical
    names coincide, so the old resolver's "does the legacy name exist"
    check never even looked for "MyProject"), after which lowercase
    reads only ever saw the new collection and uppercase reads only ever
    saw the old one -- two disjoint views of the same project, exactly
    the "does not resolve to the same entries as promised" failure the
    review called out.

    This replaces that single-collection choice with a case-insensitive
    SCAN across every collection that currently exists: the canonical
    (case-folded) name is checked first, then every OTHER
    '<COMPACT_COLLECTION>-*' collection is checked via
    _collection_contains_project, which scans for ANY stored point whose
    `project` payload field case-folds to match -- not just its most
    recent point (an earlier version of this scan did exactly that, which
    is its own bug; see _collection_contains_project's docstring, round
    4). Comparing against REAL stored values (rather than re-deriving a
    name algorithmically, as the round-1 fix did) is what lets this find
    a match under ANY historical casing, not just the one non-case-folded
    algorithm this file used before issue #37 -- there could in principle
    be more than two differently-cased collections for the same project
    by now.

    Deliberately uncapped (unlike _candidate_hint, which caps the number
    of DISPLAYED "did you mean" candidates at a handful, purely for
    AskUserQuestion's option-count limit) -- this is used for actual data
    resolution, not just a suggestion, so truncating it could silently
    hide a real match instead of merely shortening a hint. Compact-history
    collections are per-project and few in practice (this is a personal
    productivity tool called a handful of times per session, not a hot
    path), so the extra Qdrant round-trips this costs are an acceptable,
    deliberate trade of a little latency for actually-correct results,
    not an oversight.

    Deliberately does NOT swallow errors the way _sibling_compact_collections
    (a pure best-effort DISPLAY hint) does (flagged in PR review on #123):
    this function's result feeds an AUTHORITATIVE decision -- which
    collection compact_store writes into, and what compact_find reports
    existing at all -- so silently treating a failed get_collections()
    call (all retries exhausted) as "no other collections exist" would let
    compact_store create a second, fragmenting canonical collection right
    next to a real one it just couldn't see, or let compact_find falsely
    report a project as never-compacted. Letting the exception propagate
    means the caller sees a real error instead of confidently wrong data.
    _collection_contains_project (called below) is the same story: it
    doesn't swallow scroll failures either, so a failure probing one
    candidate collection here also propagates rather than being silently
    treated as "this collection doesn't match."

    Returns collection names only. The canonical name is deterministic for a
    given project string, but NOT guaranteed project-exclusive: names that
    differ only in punctuation/whitespace sanitize to the same collection, so
    callers must still filter entries by their `project` payload. Every OTHER returned name is only PROBABLY project-exclusive --
    proven only by containing AT LEAST ONE matching point (one match
    proves containment, not exclusivity: the rest of that collection's
    points could still belong to other projects entirely, e.g. an
    explicitly shared collection someone deliberately created via the
    `collection` override), never by any persisted metadata that actually
    distinguishes a dedicated collection from a shared one -- so
    compact_find still applies a per-entry case-insensitive filter on top
    of whatever this returns, rather than trusting collection membership
    alone as isolation (see compact_find's own comment at its filtering
    step). The explicit `collection` override on compact_store/compact_find
    bypasses this function entirely and keeps its own exact-match filter,
    since that path is a deliberately shared, caller-controlled collection
    from the start.
    """
    canonical_col = f"{COMPACT_COLLECTION}-{_sanitize_project(project)}"
    matches = []
    if call_with_retry(client.collection_exists, canonical_col):
        matches.append(canonical_col)
    target = project.casefold()
    response = call_with_retry(client.get_collections)
    all_names = [c.name for c in response.collections if c.name.startswith(f"{COMPACT_COLLECTION}-")]
    for name in all_names:
        if name == canonical_col:
            continue
        if _collection_contains_project(client, name, target):
            matches.append(name)
    return matches


def _collection_contains_project(client: QdrantClient, collection_name: str, target_casefolded: str) -> bool:
    """
    True if ANY point in `collection_name` has a `project` payload field
    matching `target_casefolded` case-insensitively.

    _collections_for_project used to decide a candidate collection's
    identity purely from its MOST RECENT point (via
    _most_recent_sibling_entry) -- wrong for a genuinely shared collection
    where a DIFFERENT project's entry happens to carry a LATER date than
    this project's own (flagged in PR review on #123, round 4): that
    older, real history would be silently excluded from the match
    entirely, so compact_find would omit it and compact_store could
    create a second, fragmenting canonical collection alongside a shared
    one that actually already holds this project's data. Scanning for ANY
    matching point, not just the newest one, is what actually answers
    "does this collection hold this project's history" -- using recency
    as a stand-in for identity was the bug.

    Scans the WHOLE collection (paginated, same pattern as
    _most_recent_sibling_entry -- and returns as soon as a match is found,
    so this is often cheaper in practice, not more expensive, despite
    being more thorough in the worst case). Deliberately does NOT catch
    exceptions (no try/except) for the same reason _collections_for_project
    itself doesn't: this feeds an authoritative decision, not a
    best-effort hint, so a scroll failure must propagate rather than be
    silently treated as "no match here."
    """
    offset = None
    while True:
        batch, offset = call_with_retry(
            client.scroll, collection_name=collection_name, limit=100, offset=offset, with_payload=True,
        )
        for p in batch:
            if (p.payload or {}).get("project", "").casefold() == target_casefolded:
                return True
        if offset is None:
            break
    return False


def _collection_project_values(client: QdrantClient, collection_name: str, target_casefolded: str) -> list:
    """
    Every DISTINCT literal `project` payload value stored in
    `collection_name` that case-insensitively matches `target_casefolded`
    -- there can be more than one (e.g. "MyProject" and "myproject" both
    landed in the same collection over time, since compact_store
    converges future writes onto whichever existing collection already
    matches, regardless of the exact casing each individual call passed).

    Used to build an EXACT, Qdrant-SIDE MatchAny filter for a non-canonical
    collection in compact_find's semantic query path (issue #37 round 5,
    flagged in PR review on #123): an earlier version instead over-fetched
    a larger, but still fundamentally CAPPED, number of results and
    filtered client-side afterward -- which the review correctly pointed
    out just moves the truncation-before-filter bug to a bigger threshold
    rather than eliminating it (a long-lived shared collection with more
    higher-scoring OTHER-project points than the over-fetch cap would
    still silently omit real matches). Filtering AT the Qdrant query
    itself, before its own internal ranking/limit ever truncates
    anything, is what actually eliminates the class of bug -- not a
    bigger buffer.

    Unlike _collection_contains_project (which only needs to prove
    non-emptiness and can short-circuit on the first match), this needs
    EVERY distinct value to build a correct filter, so it scans the whole
    collection unconditionally. Deliberately does NOT catch exceptions,
    for the same reason the rest of this call chain doesn't: this feeds
    an authoritative decision.
    """
    values = set()
    offset = None
    while True:
        batch, offset = call_with_retry(
            client.scroll, collection_name=collection_name, limit=100, offset=offset, with_payload=True,
        )
        for p in batch:
            v = (p.payload or {}).get("project", "")
            if v.casefold() == target_casefolded:
                values.add(v)
        if offset is None:
            break
    return list(values)


# Fixed namespace for _compact_point_id's uuid5 derivation -- any constant
# UUID works here (uuid5's own spec just requires SOME namespace), this one
# is simply namespaced under this project's own URL so it doesn't collide
# with some other subsystem's unrelated uuid5 usage that happened to pick
# the same namespace by coincidence.
_COMPACT_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "compact_store:01a0a23a-3633-7309-96ee-244a4002cd96"
)


def _compact_point_id(project: str, label: str, date: str) -> str:
    """
    Deterministic Qdrant point id for a compact, derived from
    project+label+date (issue #36).

    compact_store used to assign every stored compact a fresh uuid.uuid4(),
    which is why a retried /my-compact call (an MCP transport hiccup, or a
    user/model re-running /my-compact before confirming the first call
    succeeded -- /my-compact's own SKILL.md calls compact_store exactly
    once per run, so the retry has to come from outside that skill) landed
    on a second, indistinguishable point next to the first: two random ids
    never collide, so Qdrant just inserted both. Deriving the id instead of
    randomizing it means a retry with the SAME project/label/date always
    computes the SAME id, so client.upsert() overwrites the earlier point
    in place rather than inserting a sibling -- upsert-by-deterministic-id,
    the approach the issue's own suggested fix names first.

    uuid5 (not uuid4) is required here specifically because it has to be a
    pure function of its inputs -- same project/label/date must always
    produce the same id on every call, on every process, forever, or this
    wouldn't dedupe anything. The null-byte separator between fields (\\x00)
    is deliberately not something project/label/date names are expected to
    contain, so "a" + "b|c" can't collide with "a|b" + "c" the way a plain
    "|" join could.

    This does NOT dedupe two intentionally different compacts that happen
    to share the same label and date (e.g. saved twice in one day under
    the same label on purpose) -- that's an upsert-overwrite by design, not
    a bug: project+label+date IS the identity key the issue asks for.
    """
    key = f"{project}\x00{label}\x00{date}"
    return str(uuid.uuid5(_COMPACT_ID_NAMESPACE, key))


async def _compress_for_storage(information: str) -> tuple[str, str]:
    """
    Compresses a conversation summary through libs/local_compress_lib.py and
    returns (text to store, a short note for compact_store's reply). Never
    raises: any failure returns `information` unchanged with a note saying why,
    because losing a compact to a compression problem would be worse than
    storing it uncompressed. A successful compression returns the compressed
    text, which is deliberately much shorter than `information`.

    The model client is built here, from the CLAUDE_RUNWAY_LMSTUDIO_* values in
    this server's own env block (local_compress_lib reads them), and
    libs/lmstudio_client.py is imported only now: `openai` is not installed on
    a Qdrant-only setup, which must still be able to store compacts.

    compress() returns text under its own size threshold as is (apart from
    credential redaction), and that is reported as stored uncompressed rather
    than as a compression.
    """
    try:
        from lmstudio_client import build_client
    except Exception as e:  # ImportError, or a broken `openai` install
        return information, f"stored uncompressed: the LM Studio client is unavailable ({type(e).__name__})"
    try:
        result = await _compress_impl(
            information,
            COMPACT_FOCUS,
            preserve_identifiers=True,
            preserve_sections=True,
            client=build_client(None),
        )
    except Exception as e:
        return information, f"stored uncompressed: compression failed ({type(e).__name__}: {e})"
    if result.startswith("Error:"):
        return information, f"stored uncompressed: {result[len('Error:'):].strip()}"
    # Every request failed: the result is the original content (credentials
    # redacted) behind a notice. Keep the content, drop the notice.
    fallback = strip_total_outage_notice(result)
    if fallback is not None:
        return fallback, "stored uncompressed: LM Studio did not respond"
    if result.startswith("[compressed "):
        header = result.split("]", 1)[0].lstrip("[")
        return result, header
    return result, "stored uncompressed: the summary was under the compression threshold"


@mcp.tool()
async def compact_store(
    information: str,
    project: str,
    date: str,
    label: str = "",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection: Optional[str] = None,
) -> str:
    """
    Store a conversation compact in a dedicated Qdrant collection, isolated
    from the codebase index. The collection defaults to
    '<COMPACT_COLLECTION>-<sanitized-project>' (e.g.
    'conversation-compacts-Acme-Support-TicketsApi'), giving each
    project its own collection by default; isolation between projects whose
    names sanitize identically comes from compact_find's `project` payload
    filter, not from the collection name. Non-alphanumeric characters in the project name (dots, spaces,
    etc.) are replaced with hyphens to satisfy Qdrant naming rules, so two
    names differing only in punctuation/whitespace (e.g. "my.project" vs
    "my-project") map to the SAME collection; there is no hash suffix.
    `project` is also case-folded before any of the above (issue #37), so
    "MyProject" and "myproject" are treated as the SAME project for any
    BRAND-NEW project. If a collection already exists for this project
    under ANY casing (checked by reading a stored entry's own `project`
    payload field case-insensitively, not by re-deriving a name), writes
    continue landing in that existing collection instead of fragmenting
    into a new one -- see _collections_for_project. Pass collection
    explicitly to override.

    Called by /my-compact instead of qdrant-store. project, label, and date
    are stored as payload fields and used by compact_find for native filtering
    and sorting. The collection is created automatically on first use.

    `information` is compressed by the local LM Studio model before it is
    stored (with preserve_identifiers and preserve_sections, so file paths,
    identifiers and every section heading survive), so the caller passes the
    full structured summary once instead of compressing it in a separate call
    first. When LM Studio is unavailable the summary is stored uncompressed and
    the reply says so; the compact is never lost to a compression failure.

    `label` is optional (issue #72): when omitted or blank, the label is
    derived server-side from the "## What we were working on" section of
    `information` (the first sentence, up to 80 chars) using the same
    section-splitting logic that already parses the compact template.  This
    removes label-quality variance that came from asking each calling model to
    extract the first sentence on its own.  Pass an explicit `label` to
    override the derived value (e.g. from a /my-compact argument).

    Idempotent (issue #36): the stored point's id is deterministically
    derived from project+label+date (see _compact_point_id), not a random
    uuid4. A retried call with the same three values upserts onto the same
    id instead of inserting a second, indistinguishable point, so an
    external retry (an MCP transport hiccup, or a re-run before confirming
    the first call succeeded) can't silently duplicate a compact in the
    /my-resume picker.

    Requires fastembed and qdrant-client (both already installed via
    requirements.txt's Core section). QDRANT_URL / QDRANT_API_KEY /
    COMPACT_COLLECTION / COMPACT_EMBEDDING_MODEL and the CLAUDE_RUNWAY_LMSTUDIO_*
    values are read from this server's .mcp.json env block.
    """
    import re as _re
    if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return f"Error: date must be ISO format YYYY-MM-DD (got {date!r}). compact_find sorts lexicographically, so non-ISO dates produce incorrect ordering."

    # Derive a label server-side when the caller didn't supply one (issue #72).
    # Uses the same split_sections logic that parse_compact already applies,
    # so this is deterministic and consistent across every calling model.
    if not label.strip():
        label = derive_compact_label(information) or "(unlabeled)"

    # Compress BEFORE touching Qdrant, and from the label's own source text, so
    # the label above is derived from what the caller wrote, not from a model's
    # rewrite of it. Never raises: a failure here stores the summary as given.
    information, compression_note = await _compress_for_storage(information)

    client = QdrantClient(
        url=qdrant_url or COMPACT_QDRANT_URL,
        api_key=(qdrant_api_key or COMPACT_QDRANT_API_KEY) or None,
    )
    canonical_col = f"{COMPACT_COLLECTION}-{_sanitize_project(project)}"
    if collection:
        # Caller-controlled override -- use exactly as given, no
        # case-insensitive resolution (the caller already fully controls
        # both sides of this lookup).
        col = collection
    else:
        # Reuse whichever existing collection (under ANY casing) already
        # holds this project's history, case-insensitively (issue #37
        # round 2 -- see _collections_for_project's docstring for why a
        # single legacy-vs-canonical choice wasn't enough: it could still
        # fragment history across a genuine casing drift). Prefer the
        # canonical (case-folded) collection if it's among the matches,
        # so writes keep converging on ONE collection going forward;
        # otherwise reuse whichever existing match was found. Falls back
        # to creating the canonical collection fresh if nothing matches
        # yet (a brand-new project).
        matches = _collections_for_project(client, project)
        col = canonical_col if canonical_col in matches else (matches[0] if matches else canonical_col)
    embedding_provider = FastEmbedProvider(COMPACT_EMBEDDING_MODEL)
    vector_name = embedding_provider.get_vector_name()
    # Every direct QdrantClient network call below is wrapped in
    # call_with_retry -- constructing FastEmbedProvider (loads the ONNX
    # model) and embed_documents() are both CPU-bound gaps during which
    # Docker Desktop's vpnkit can reap an idle pooled connection, so the
    # very next Qdrant call can raise a ResponseHandlingException on a
    # connection the client believes is still healthy. See issue #75/#76 /
    # libs/qdrant_retry.py, and ingest_mcp_server.py's identical pattern.
    created_here = False
    if not call_with_retry(client.collection_exists, col):
        try:
            call_with_retry(
                client.create_collection,
                collection_name=col,
                vectors_config={
                    vector_name: qdrant_models.VectorParams(
                        size=embedding_provider.get_vector_size(),
                        distance=qdrant_models.Distance.COSINE,
                    )
                },
            )
            created_here = True
        except Exception:
            # Two server processes saving a project's FIRST compact at the same
            # time can both see the collection absent; the loser's create then
            # conflicts. If the collection exists now, the other process won:
            # carry on and store into it (its model is checked below, like any
            # existing collection). If it is still absent, the failure was
            # something else, so re-raise it. Same recovery as
            # QdrantConnector.ensure_collection_exists.
            if not call_with_retry(client.collection_exists, col):
                raise
    if not created_here:
        # Issue #397: an existing collection created under a different model
        # would otherwise get a SECOND vector name added by the upsert below
        # (or fail opaquely), leaving the old points unsearchable by this
        # model. Fails open on an inconclusive check, like find_in_collection.
        mismatch = check_embedding_model_mismatch(
            client, col, embedding_provider, recovery_hint=COMPACT_MODEL_MISMATCH_HINT
        )
        if mismatch:
            return mismatch
    # Issue #397: mark the collection internal on EVERY write (idempotent), so
    # find_in_collection/list_collections can tell it isn't a search target.
    # Done on every write rather than only at creation so a legacy collection
    # gets marked on its next /my-compact. Fails open: a failed mark must not
    # lose the user's compact.
    mark_compact_collection(client, col)
    [embedding] = await embedding_provider.embed_documents([information])
    # Deterministic, not random (issue #36) -- see _compact_point_id's
    # docstring for why this is what makes a retry an overwrite instead of
    # a duplicate insert. Uses the raw `project` argument, same as before
    # issue #37 -- point-id dedup only ever needs to matter within a
    # single call's own retries, which always pass identical arguments,
    # so there's no cross-casing dedup need here the way there is for
    # collection/payload resolution above.
    point_id = _compact_point_id(project, label, date)
    call_with_retry(
        client.upsert,
        collection_name=col,
        points=[
            qdrant_models.PointStruct(
                id=point_id,
                vector={vector_name: embedding},
                # "document" mirrors the payload shape the now-removed
                # QdrantClient.add() used to write -- compact_find's
                # scroll-path fallback (.get("information", .get("document",
                # ""))) still checks for it on entries stored before this fix.
                # payload's "project" is the RAW argument (issue #37 round
                # 2) -- nothing downstream expects it to be normalized:
                # _collections_for_project reads this exact raw value back
                # to decide whether a collection matches a project
                # (case-insensitively), and compact_find's default
                # (non-explicit-collection) read path applies its OWN
                # case-insensitive PER-ENTRY filter against it. The exact
                # literal value is also a human-readable record of what this
                # specific call actually passed.
                # metadata.source (issue #282) is what index_repo/sync_repo's
                # delete filters exclude on (mb.memory_bank_exclusion_filter),
                # so a reset pointed at a compact collection can't wipe these.
                payload={
                    "document": information, "project": project, "label": label, "date": date,
                    "information": information, "metadata": {"source": COMPACT_SOURCE},
                },
            )
        ],
    )
    return (
        f"Stored compact in '{col}': project={project!r}, date={date!r}, label={label!r}, id={point_id!r} "
        f"({compression_note})"
    )


def _sibling_compact_collections(client: QdrantClient, exclude: list) -> list:
    """
    List other '<COMPACT_COLLECTION>-*' collections besides everything in
    `exclude` (issue #37's recovery path). Used when NOTHING was found for
    a given project under any casing -- either this project was genuinely
    never compacted, or the project name has drifted entirely (a renamed
    directory, a different worktree checkout) from whatever it was when
    /my-compact ran (a pure casing drift never reaches here at all -- see
    _collections_for_project, which already finds any casing variant
    directly, so this is purely for a genuinely different name). `exclude`
    is always a list, never a bare string -- iterating a bare string would
    silently exclude by individual CHARACTER instead of by name. Returns
    an empty list (never raises) if the get_collections() call itself
    fails, so this stays purely additive -- a Qdrant hiccup here degrades
    back to the plain "not found" message instead of blowing up the whole
    call.

    Returns ALL matching sibling names, uncapped. An earlier version
    capped this list to `max_collections` (3) raw names before
    _candidate_hint ever got a chance to probe and skip the ones that
    turn out empty or unprobeable -- `compact_store` creates a collection
    before it has any points, so an empty sibling is a realistic state,
    and 3 empty siblings could hide a 4th with real saved compacts from
    ever being considered (flagged in PR review on #123, round 6). The
    actual display cap belongs in _candidate_hint below, applied AFTER
    filtering, not here.
    """
    try:
        response = call_with_retry(client.get_collections)
    except Exception:
        return []
    prefix = f"{COMPACT_COLLECTION}-"
    excluded = set(exclude)
    return [c.name for c in response.collections if c.name.startswith(prefix) and c.name not in excluded]


def _most_recent_sibling_entry(client: QdrantClient, collection_name: str):
    """
    Fetch one collection's most-recently-dated compact, keyed by ISO date
    string (lexicographic sort, same assumption compact_find's own
    date-sort already relies on). Used ONLY by _candidate_hint, a
    best-effort DISPLAY hint -- authoritative project resolution
    (_collections_for_project) now uses _collection_contains_project
    instead, precisely because "most recent point" is the wrong question
    to ask when deciding whether a collection belongs to a project at all
    (round 4, see that function's docstring); it's still exactly the
    right question for "what should we SHOW the user as this candidate's
    most recent activity," which is all _candidate_hint needs. This
    function itself does NOT swallow failures -- a scroll error
    propagates to the caller, and _candidate_hint wraps its own call in a
    try/except (a failed hint probe should degrade gracefully, never
    block the underlying result) rather than this shared helper deciding
    that for every caller.

    Pages through the WHOLE collection (mirroring compact_find's own
    scroll loop for its primary path) rather than reading a single capped
    page -- an earlier version of this function read only the first
    100-point page, which is ID-ordered, not date-ordered, so a
    collection with more than 100 points could have its actual
    most-recently-dated entry sitting on a later page and never get
    considered (flagged in PR review on #123, round 1). Returns None (not
    an exception) only for the genuinely different case of a collection
    that scrolled successfully but turned out to be empty -- that's a
    real, valid result, not a failure.
    """
    all_points = []
    offset = None
    while True:
        batch, offset = call_with_retry(
            client.scroll, collection_name=collection_name, limit=100, offset=offset, with_payload=True,
        )
        all_points.extend(batch)
        if offset is None:
            break
    if not all_points:
        return None
    best = max(all_points, key=lambda p: (p.payload or {}).get("date", ""))
    payload = best.payload or {}
    return payload.get("date", "?"), payload.get("label", "(no label)"), payload.get("project", "?")


def _candidate_hint(client: QdrantClient, exclude: list) -> str:
    """
    Builds the "did you mean one of these other projects" suffix shared by
    both of compact_find's fallback paths (collection-not-found and
    zero-results-within-an-existing-collection). `exclude` is every
    collection name already considered for this call (so it's never
    re-offered as if it were a NEW candidate) -- a list, since the
    default (non-explicit-collection) path can have considered more than
    one collection (issue #37 round 2's aggregation; see
    _collections_for_project). Returns "" when no sibling collections
    exist or none of them have any entries, so callers can just
    concatenate this onto their base message unconditionally.

    This function treats a failed _most_recent_sibling_entry probe as
    "skip this candidate" rather than letting it propagate (issue #37
    round 3, flagged in PR review on #123) -- deliberately: this function
    is a pure best-effort DISPLAY hint layered on top of an
    already-decided result (compact_find has already determined nothing
    matched before this ever runs), so one bad sibling collection
    degrading the suggestion list is the right failure mode. Contrast
    _collections_for_project's own resolution helper,
    _collection_contains_project, which deliberately does NOT catch its
    exceptions, for exactly the opposite reason (that path is
    authoritative, not a suggestion).

    The 3-candidate cap is applied HERE, to `hints` (candidates that
    actually turned out to have content), not to the raw sibling-name
    list _sibling_compact_collections returns -- capping the raw list
    first (an earlier version's bug, round 6) meant 3 empty or unprobeable
    siblings could hide a 4th with real saved compacts from ever being
    probed at all, so the recovery path could wrongly report zero
    candidates even when one genuinely existed further down the list.
    Stopping the loop itself once 3 real hints are collected (rather than
    probing every sibling and slicing afterward) also avoids scrolling
    entire collections that will just be discarded once the cap is hit.

    3, NOT an arbitrary round number: Claude Code's built-in
    AskUserQuestion tool hard-caps every question at 2-4 options total
    (confirmed directly against the CLI's own bundled schema, not just
    the reviewer's claim on PR #123 -- verified via `strings` against the
    installed `claude` binary, which contains the literal schema
    description "2-4 options" and "do not exceed them even if the user
    requests more"). my-resume/SKILL.md's "If no match is found" flow
    presents these candidates as AskUserQuestion options PLUS one extra
    "None of these" option, so up to 3 real candidates + 1 fallback stays
    within that hard 4-option ceiling.
    """
    max_candidates = 3
    hints: list = []
    for name in _sibling_compact_collections(client, exclude):
        if len(hints) >= max_candidates:
            break
        try:
            entry = _most_recent_sibling_entry(client, name)
        except Exception:
            continue
        if entry is None:
            continue
        date, label, stored_project = entry
        hints.append(f"  - project '{stored_project}' (most recent: {date} — {label})")
    if not hints:
        return ""
    return (
        "\n\nDid you mean one of these other projects, which DO have saved compacts "
        "(this can happen if the working directory was renamed, or you're in a "
        "different checkout/worktree)?\n" + "\n".join(hints)
    )


@mcp.tool()
async def compact_find(
    project: str,
    query: Optional[str] = None,
    limit: int = 10,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection: Optional[str] = None,
) -> str:
    """
    Retrieve conversation compacts for a project from its dedicated Qdrant
    collection ('<COMPACT_COLLECTION>-<sanitized-project>' by default --
    see compact_store / _sanitize_project). The collection name is a shared base
    prefix plus the sanitized project, which can collide for names differing
    only in punctuation/whitespace; the payload filter on 'project' is what
    isolates one project's entries from another's, including when a shared
    collection is passed explicitly via the collection param.

    Called by /my-resume instead of qdrant-find. Returns compacts sorted by
    date descending (most recent first), each with its date, label, and full
    content, ready for /my-resume to present as a selection list.

    When query is omitted, uses scroll (payload filter only, no vector
    ranking) so date-sort is reliable and recent entries are never buried by
    semantic distance. When query is provided, uses semantic search to narrow
    results to the closest match — suited for keyword-based selection like
    '/my-resume auth refactor'. limit caps results in both cases.

    Each entry's header also includes a short suffix of its own Qdrant
    point id (issue #36). Going forward, compact_store derives that id
    deterministically from project+label+date, so a retried call with the
    SAME three values now overwrites the existing point instead of
    inserting a second one -- there is nothing left to disambiguate for
    compacts stored under this scheme. The suffix exists for the LEGACY
    case: any point stored before this fix still carries its old random
    uuid4, and because compact_store used to accept unlimited calls with
    identical project+label+date as unlimited separate inserts, two such
    pre-fix points can legitimately have identical date/label already
    sitting in the collection today, which would otherwise render as
    indistinguishable entries in /my-resume's selection UI. Surfacing the
    id suffix here is what lets that UI (or a human reading raw output)
    tell those specific pre-existing entries apart.

    `project` is case-folded before deriving the canonical collection name
    (issue #37), so "MyProject" and "myproject" always land on the same
    NEW-style collection for a brand-new project. For a project with
    EXISTING history, this AGGREGATES entries across every
    '<COMPACT_COLLECTION>-*' collection that case-insensitively matches
    `project` -- not just one -- so a casing drift between /my-compact and
    /my-resume, or even multiple different historical casings accumulated
    over time, still surfaces everything instead of silently showing only
    whichever single collection happened to match (round 2 of issue #37,
    flagged in PR review on #123: picking exactly one collection could
    still fragment history across a genuine casing drift). See
    _collections_for_project. For a project with NO existing history under
    any casing, both the collection-not-found path and the
    zero-matching-entries path list any sibling '<COMPACT_COLLECTION>-*'
    collections' most recent entries as candidates, instead of just
    failing flatly -- these are necessarily a genuinely different project
    name at that point (a renamed working directory, a different worktree
    checkout), since any same-project casing variant would already have
    been picked up by the aggregation above.
    """
    client = QdrantClient(
        url=qdrant_url or COMPACT_QDRANT_URL,
        api_key=(qdrant_api_key or COMPACT_QDRANT_API_KEY) or None,
    )
    canonical_col = f"{COMPACT_COLLECTION}-{_sanitize_project(project)}"

    if collection:
        # Caller-controlled override -- a single, explicit, possibly
        # SHARED collection, so (unlike the default path below) isolation
        # isn't already guaranteed by construction and the exact-match
        # payload filter is the only thing keeping it project-scoped.
        col_list = [collection]
        project_filter = qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(key="project", match=qdrant_models.MatchValue(value=project))]
        )
        # See compact_store's identical comment -- every direct
        # QdrantClient call here is wrapped in call_with_retry for the
        # same CPU-bound-gap-then-dropped-connection reason (issue
        # #75/#76 / libs/qdrant_retry.py).
        if not call_with_retry(client.collection_exists, collection):
            return (
                f"No compacts collection found ('{collection}') for project '{project}'. "
                "Run /my-compact first to create it."
            ) + _candidate_hint(client, [collection])
    else:
        # Every collection (under ANY casing) that already holds this
        # project's history -- see _collections_for_project's docstring
        # for why this scans rather than picks one. No Qdrant-level
        # project_filter here (each collection this returns is EXPECTED
        # to already be scoped to this one project) -- but see the
        # case-insensitive PER-ENTRY filter applied below, after fetching,
        # which is the actual isolation guarantee for this path (issue
        # #37, PR #123 review round 3). Every match _collections_for_project returns --
        # INCLUDING the canonical one -- is only PROVEN project-exclusive
        # by inspecting stored data, not by any persisted metadata that
        # distinguishes a dedicated collection from a shared one someone
        # deliberately (or accidentally) wrote into via compact_store's own
        # `collection` override: the canonical collection is "trusted" for
        # RESOLUTION (deciding a collection belongs to this project) but
        # not for ISOLATION (nothing stops another project's points from
        # landing inside it). Without a per-entry check here too, such a
        # collection could leak another project's entries into this
        # result.
        col_list = _collections_for_project(client, project)
        project_filter = None
        if not col_list:
            return (
                f"No compacts collection found ('{canonical_col}') for project '{project}'. "
                "Run /my-compact first to create it."
            ) + _candidate_hint(client, [canonical_col])

    # Case-insensitive per-entry isolation for the default (non-explicit-
    # collection) path -- see the comment above where col_list was built.
    # None when `collection` was explicit (that path already has its own
    # exact-match Qdrant-level project_filter and doesn't need this).
    case_insensitive_target = None if collection else project.casefold()

    if query:
        # Semantic search — preserve relevance order, do not date-sort.
        embedding_provider = FastEmbedProvider(COMPACT_EMBEDDING_MODEL)
        # Issue #397: a collection stored under another model has no vector
        # of this name, so the query below would fail opaquely or silently
        # miss everything. Report it clearly instead (fails open).
        for col in col_list:
            mismatch = check_embedding_model_mismatch(
                client, col, embedding_provider, recovery_hint=COMPACT_MODEL_MISMATCH_HINT
            )
            if mismatch:
                return mismatch
        query_vector = await embedding_provider.embed_query(query)
        raw = []
        for col in col_list:
            # For the default (non-explicit-collection) path, Qdrant's
            # OWN `limit` truncates results BEFORE any client-side filter
            # could ever see them -- so filtering has to happen AT the
            # query itself (server-side), not after, or a genuinely
            # SHARED collection with more than `limit` higher-ranked
            # points from a DIFFERENT project could fill the entire
            # returned page with points a client-side filter would then
            # remove, silently reporting no matches despite this
            # project's own (lower-ranked) real entries still sitting in
            # that same collection (a bigger over-fetch would only move
            # that bug to a larger threshold, not remove it). This filter
            # is built and applied for EVERY collection in col_list,
            # including the canonical one, because `collection` is a
            # caller-controlled override on compact_store's WRITE side
            # too, so nothing prevents another project's points from being
            # written under this exact canonical name; without a filter
            # here, those foreign points could rank ahead of the real ones
            # and reproduce the same truncation-before-filter bug (issue
            # #37, PR #123 review rounds 5-6). Every collection gets an EXACT MatchAny
            # filter built from its own actual stored project values
            # (there can be more than one casing variant within a single
            # collection -- see _collection_project_values), so Qdrant
            # itself only ever ranks and truncates already-isolated
            # points. A collection with zero matching literal values
            # (should not happen for anything _collections_for_project
            # returned, but defensively possible if data changed between
            # resolution and query) is skipped outright rather than
            # queried with an empty MatchAny, which Qdrant would treat as
            # "match nothing" anyway -- skipping just avoids the wasted
            # round trip.
            if case_insensitive_target is not None:
                values = _collection_project_values(client, col, case_insensitive_target)
                if not values:
                    continue
                col_filter: Optional[qdrant_models.Filter] = qdrant_models.Filter(
                    must=[qdrant_models.FieldCondition(key="project", match=qdrant_models.MatchAny(any=values))]
                )
            else:
                col_filter = project_filter
            raw.extend(call_with_retry(
                client.query_points,
                collection_name=col,
                query=query_vector,
                using=embedding_provider.get_vector_name(),
                query_filter=col_filter,
                limit=limit,
                with_payload=True,
            ).points)
        if case_insensitive_target is not None:
            # Defensive double-check, not the primary isolation mechanism
            # anymore (that's the per-collection MatchAny filter above) --
            # cheap, and matches this file's existing defense-in-depth
            # style (e.g. the retry-wrapped calls throughout).
            raw = [r for r in raw if (r.payload or {}).get("project", "").casefold() == case_insensitive_target]
        # Only matters when col_list has more than one entry (a genuine
        # historical casing-drift case) -- each collection's own results
        # already arrive relevance-sorted, so re-sort the COMBINED set by
        # score before capping to `limit`, rather than just concatenating
        # and cutting off partway through the second collection's results.
        # Real Qdrant ScoredPoint objects always carry `.score`; getattr's
        # default only matters for this file's own test fakes.
        raw.sort(key=lambda r: getattr(r, "score", 0), reverse=True)
        raw = raw[:limit]
        entries = [
            (r.id, (r.payload or {}).get("date", ""), (r.payload or {}).get("label", ""),
             (r.payload or {}).get("information", (r.payload or {}).get("document", "")))
            for r in raw
        ]
    else:
        # No query — page through ALL matching points in EVERY resolved
        # collection so date-sort covers the full combined set, not just
        # the first scroll page of the first collection (which is
        # ID-ordered, not date-ordered, so a cap-then-sort would miss
        # newer entries stored with higher IDs).
        all_points = []
        for col in col_list:
            offset = None
            while True:
                batch, offset = call_with_retry(
                    client.scroll,
                    collection_name=col,
                    scroll_filter=project_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )
                all_points.extend(batch)
                if offset is None:
                    break
        if case_insensitive_target is not None:
            all_points = [p for p in all_points if (p.payload or {}).get("project", "").casefold() == case_insensitive_target]
        entries = [
            (
                p.id, (p.payload or {}).get("date", ""), (p.payload or {}).get("label", ""),
                (p.payload or {}).get("information", (p.payload or {}).get("document", "")),
            )
            # qdrant-client types payload as Optional[dict] -- a point written
            # by this codebase always has one, but the client's own stubs
            # don't guarantee it, so this guards against a genuinely payload-
            # less point crashing the whole call instead of just losing that
            # one entry's fields.
            for p in all_points
        ]
        # Date is now index 1 (id is 0) -- sort key updated to match.
        entries.sort(key=lambda e: e[1], reverse=True)
        entries = entries[:limit]

    if not entries:
        return f"No compacts found for project '{project}'." + _candidate_hint(client, col_list)

    lines = [f"Found {len(entries)} compact(s) for project '{project}':\n"]
    for i, (point_id, date, label, information) in enumerate(entries, 1):
        # Short disambiguator, not the full id. Strip
        # dashes first so the 8 chars taken are hex digits, not separators.
        id_suffix = str(point_id).replace("-", "")[:8]
        lines.append(f"--- {i}. {date or '?'} — {label or '(no label)'} (id: {id_suffix}) ---")
        lines.append(information or "(no content)")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def compact_prune(
    project: str,
    keep_last_n: Optional[int] = None,
    older_than_days: Optional[int] = None,
    dry_run: bool = True,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection: Optional[str] = None,
) -> str:
    """
    Delete stored conversation compacts for a project, either by keeping
    only the N most recent entries or by removing entries older than a
    given number of days.

    Exactly one of `keep_last_n` or `older_than_days` must be provided;
    passing both, or neither, is an error.

    `dry_run` defaults to True -- reports what WOULD be deleted without
    removing anything. Pass dry_run=False explicitly to actually delete.
    This is the most important safety default: a prune call that accidentally
    matched the wrong project, or a keep_last_n that was set too aggressively,
    can be caught and aborted before any data is lost.

    `keep_last_n=N` retains the N most recent compacts by parsed ISO date
    and deletes the rest. Compacts whose stored date is missing, malformed,
    or calendar-invalid (e.g. "2026-99-99") cannot be date-ranked and are
    always preserved outside the N quota -- `keep_last_n=0` does NOT
    delete such entries. `keep_last_n=1` keeps only the most recently
    dated parseable compact (plus any unparseable ones).

    `older_than_days=N` deletes any compact whose parsed date is strictly
    before (today − N days). Entries whose date is today, in the future,
    or unparseable are never deleted by this mode. `older_than_days=0`
    deletes any entry dated before today (today-dated and future-dated
    entries survive). `older_than_days=1` deletes entries dated before
    yesterday (today's and yesterday's survive), etc.

    Uses the same project collection resolution (case-insensitive, multi-
    collection aggregation) as compact_find -- see _collections_for_project's
    docstring. The `collection` override bypasses that resolution in favour
    of a caller-controlled exact-match filter, same as compact_store and
    compact_find.

    Only individual POINTS are deleted, never the collection itself.
    """
    import datetime as _dt

    if (keep_last_n is None) == (older_than_days is None):
        return (
            "Error: provide exactly one of keep_last_n or older_than_days, not both or neither. "
            "Example: compact_prune(project='myproject', keep_last_n=5) "
            "or compact_prune(project='myproject', older_than_days=30)."
        )
    if keep_last_n is not None and keep_last_n < 0:
        return f"Error: keep_last_n must be a non-negative integer (got {keep_last_n!r})."
    if older_than_days is not None and older_than_days < 0:
        return f"Error: older_than_days must be a non-negative integer (got {older_than_days!r})."

    client = QdrantClient(
        url=qdrant_url or COMPACT_QDRANT_URL,
        api_key=(qdrant_api_key or COMPACT_QDRANT_API_KEY) or None,
    )
    canonical_col = f"{COMPACT_COLLECTION}-{_sanitize_project(project)}"

    if collection:
        col_list = [collection]
        project_filter = qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(key="project", match=qdrant_models.MatchValue(value=project))]
        )
        if not call_with_retry(client.collection_exists, collection):
            return (
                f"No compacts collection found ('{collection}') for project '{project}'. "
                "Run /my-compact first to create it."
            )
        case_insensitive_target: Optional[str] = None
    else:
        col_list = _collections_for_project(client, project)
        project_filter = None
        if not col_list:
            return (
                f"No compacts collection found ('{canonical_col}') for project '{project}'. "
                "Run /my-compact first to create it."
            )
        case_insensitive_target = project.casefold()

    # Gather every matching point across all resolved collections, with
    # collection membership so we know WHERE each point lives for deletion.
    # qdrant_client stubs type scroll() results as list[Record] (which is not
    # an importable public type in every version we support) -- use a plain
    # untyped list and suppress the two attribute accesses a type checker
    # (mypy originally, Pyright since issue #297) would otherwise flag below,
    # the same pattern call_with_retry's own callers use throughout this file.
    all_points: list = []  # list of (collection_name, Record)
    for col in col_list:
        offset = None
        while True:
            batch, offset = call_with_retry(
                client.scroll,
                collection_name=col,
                scroll_filter=project_filter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for p in batch:
                if case_insensitive_target is not None:
                    if (p.payload or {}).get("project", "").casefold() != case_insensitive_target:
                        continue
                all_points.append((col, p))
            if offset is None:
                break

    if not all_points:
        return f"No compacts found for project '{project}' — nothing to prune."

    # `# type: ignore[no-untyped-def]` (mypy's complaint about `p`'s missing
    # parameter annotation) is gone as of issue #297 -- confirmed Pyright
    # doesn't flag this at all, so there's nothing left to suppress here.
    def _parse_date(p) -> Optional[_dt.date]:
        """Parse the payload date field; return None for unparseable values."""
        try:
            return _dt.date.fromisoformat((p.payload or {}).get("date", ""))
        except (ValueError, TypeError):
            return None

    # Separate entries with valid parseable dates from those without.
    # Unparseable dates (malformed, missing, or calendar-invalid values like
    # "2026-99-99") must NOT participate in date-based ranking:
    # - For keep_last_n: a malformed date string sorts ahead of every valid
    #   2026 date lexicographically, so including it in the sort would
    #   silently treat it as "newest" and discard the genuinely newest real
    #   compact instead (Copilot review, round 1).
    # - For older_than_days: these are already excluded in that branch, but
    #   keeping the separation uniform makes the intent explicit in both paths.
    # Entries with unparseable dates are always preserved outside the quota.
    parseable: list = []
    unparseable_kept = 0
    for col, p in all_points:
        if _parse_date(p) is not None:
            parseable.append((col, p))
        else:
            unparseable_kept += 1

    # Sort parseable entries by date descending (most recent first).
    # All entries in `parseable` have a non-None _parse_date -- the list was
    # built by filtering out None results above, so the cast is safe.
    parseable.sort(key=lambda cp: _parse_date(cp[1]) or _dt.date.min, reverse=True)

    # Determine which points to delete.
    if keep_last_n is not None:
        # Keep the first keep_last_n parseable entries, delete the rest.
        to_delete = parseable[keep_last_n:]
        kept_count = min(keep_last_n, len(parseable)) + unparseable_kept
    else:
        # older_than_days: delete any parseable point whose date is strictly
        # more than older_than_days days before today. The XOR check above
        # (`(keep_last_n is None) == (older_than_days is None)`) guarantees
        # older_than_days is not None here, but a type checker can't narrow
        # through that separate boolean comparison (issue #297 -- same
        # limitation the chars_limited/max_chars comment in
        # libs/local_compress_lib.py's compress() describes), hence the
        # explicit assert.
        assert older_than_days is not None
        today = _dt.date.today()
        try:
            cutoff = today - _dt.timedelta(days=older_than_days)
        except OverflowError:
            # older_than_days is astronomically large — nothing is that old.
            return (
                f"Nothing to prune for project '{project}': "
                f"older_than_days={older_than_days} exceeds the representable date range; "
                "no compact could be that old."
            )
        to_delete = []
        kept_count = unparseable_kept
        for col, p in parseable:
            entry_date = _parse_date(p)
            assert entry_date is not None  # guaranteed non-None (parseable list)
            if entry_date < cutoff:
                to_delete.append((col, p))
            else:
                kept_count += 1

    if not to_delete:
        return (
            f"Nothing to prune for project '{project}': "
            f"all {len(all_points)} compact(s) satisfy the retention criteria."
        )

    # Build a human-readable summary of what will be (or was) deleted.
    # Same as _parse_date above -- no longer needs a `# type: ignore` under Pyright.
    def _entry_summary(col: str, p) -> str:
        payload = (p.payload or {})
        date = payload.get("date", "?")
        label = payload.get("label", "(no label)")
        id_suffix = str(p.id).replace("-", "")[:8]
        return f"  {date} — {label} (id: {id_suffix}) [{col}]"

    summary_lines = [_entry_summary(col, p) for col, p in to_delete]
    summary_text = "\n".join(summary_lines)

    mode_desc = (
        f"keep_last_n={keep_last_n}" if keep_last_n is not None
        else f"older_than_days={older_than_days}"
    )

    if dry_run:
        return (
            f"[DRY RUN] Would delete {len(to_delete)} compact(s) for project '{project}' "
            f"({mode_desc}), retaining {kept_count}:\n"
            f"{summary_text}\n\n"
            "Pass dry_run=False to actually delete."
        )

    # Group deletes by collection to minimise round trips.
    by_col: dict[str, list] = {}
    for col, p in to_delete:
        # `# type: ignore[attr-defined]` (mypy's complaint about `p.id`) is
        # gone as of issue #297 -- confirmed Pyright doesn't flag this at all.
        by_col.setdefault(col, []).append(p.id)

    for col, ids in by_col.items():
        call_with_retry(
            client.delete,
            collection_name=col,
            points_selector=qdrant_models.PointIdsList(points=ids),
        )

    return (
        f"Deleted {len(to_delete)} compact(s) for project '{project}' "
        f"({mode_desc}), retained {kept_count}:\n"
        f"{summary_text}"
    )


if __name__ == "__main__":
    mcp.run()
