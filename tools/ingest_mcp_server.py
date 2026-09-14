#!/usr/bin/env python3
"""
MCP server exposing repo indexing as callable tools, so Claude (in Claude
Code) can trigger re-indexing on demand -- e.g. right after cloning a repo,
or when it notices qdrant-find is coming back empty -- instead of you
running ingest_to_qdrant.py by hand every time.

Project minimum: Python 3.12 (policy floor; the dependency chain supports >=3.10).

Setup:
    pip install "mcp[cli]" mcp-server-qdrant qdrant-client --break-system-packages

Register in .mcp.json (adjust the path to wherever you save this file). Set
QDRANT_URL / COLLECTION_NAME / EMBEDDING_MODEL here -- these are the SAME env
var names mcp-server-qdrant itself reads, so one project config drives both
servers and they can't drift apart:
{
  "mcpServers": {
    "codebase-indexer": {
      "command": "python",
      "args": ["/absolute/path/to/tools/ingest_mcp_server.py"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "<this-project's-collection-name>",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
      }
    }
  }
}

With COLLECTION_NAME set per-project this way, every tool call below defaults
to that project's collection automatically -- you don't pass collection on
every call, and opening a different project (different .mcp.json) uses a
different collection with no manual switching.

IMPORTANT: this server uses stdio transport, meaning stdout is reserved for
MCP protocol messages. Nothing in this file should ever call print() --
progress and results must be returned as tool output, not printed. (This is
also why this server is a separate file from ingest_to_qdrant.py, which
prints freely as a normal CLI script.)
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from mcp.server.mcpserver import MCPServer, Context
from mcp_server_qdrant.qdrant import QdrantConnector, Entry
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from qdrant_client import QdrantClient, models

from qdrant_ingest_lib import (
    batched,
    build_entries,
    chunk_file,
    compute_file_hashes,
    ensure_persistent_fastembed_cache,
    files_removed_since_last_sync,
    iter_entries,
    validate_chunk_params,
)
from qdrant_retry import call_with_retry, async_call_with_retry
from qdrant_batch_store import store_batch, ensure_file_path_index, FIELD_INDEXES, UPSERT_BATCH_SIZE
from qdrant_collection_hints import get_cached_descriptions, set_cached_description
from mcp_tool_introspect import tool_count

# Must run before any FastEmbedProvider(...) construction below (index_repo,
# sync_repo, find_in_collection each make their own) -- see issue #77 and the
# function's own docstring for why this can't just be a .mcp.json env value.
ensure_persistent_fastembed_cache()

MANIFEST_FILENAME = ".qdrant_index_manifest.json"

# sync_repo batches its per-file delete filter into chunks of this size
# instead of one client.delete() call per changed/removed file (issue #75/
# #76 follow-up): a live repro against this repo's own 10,532-file monorepo
# confirmed the real trigger wasn't an idle-connection gap at all, but this
# loop itself -- thousands of rapid individual HTTP round-trips to Qdrant
# exhausting local ephemeral ports / Docker Desktop's connection-tracking
# table, so the very next connection attempt (for the embedding/upsert
# phase) failed outright. Confirmed empirically: a single Filter(should=...)
# covering 12,000 conditions times out, but chunks of this size complete
# reliably in ~1-2s each regardless of chunk size 250-1000 tried -- 500 is
# the conservative middle ground (few enough round-trips to avoid exhaustion,
# small enough to comfortably clear typical client timeouts).
DELETE_BATCH_SIZE = 500


def _load_manifest(repo: Path) -> dict:
    manifest_path = repo / MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            # Explicit encoding on both halves so a manifest written on one
            # platform reads back on another. UnicodeDecodeError is caught
            # alongside the others because it is NOT a json.JSONDecodeError --
            # uncaught, it would propagate out of a sync and fail the tool call
            # rather than degrading to "no manifest, treat everything as new."
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return {}
    return {}


def _save_manifest(repo: Path, manifest: dict) -> None:
    (repo / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

# Same env var names mcp-server-qdrant reads, so a single project .mcp.json
# config keeps this server and the qdrant-find/qdrant-store server aligned.
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
DEFAULT_COLLECTION = os.environ.get("COLLECTION_NAME")
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Optional, this repo's own -- not read by mcp-server-qdrant. Declared once
# alongside COLLECTION_NAME so a project's description doesn't require a
# one-off set_collection_description call (issue #83); applied by
# _sync_static_collection_description() below.
DEFAULT_COLLECTION_DESCRIPTION = os.environ.get("COLLECTION_DESCRIPTION")


def _parse_csv_set(value: Optional[str]) -> Optional[set]:
    return {v.strip() for v in value.split(",") if v.strip()} if value else None


def _skipped_summary(skipped: list) -> str:
    """
    Formats a (rel_path, error) skip list from build_entries/
    compute_file_hashes into a trailing summary sentence -- empty string if
    nothing was skipped, so callers can just append this to their return
    message unconditionally (issue #27).
    """
    if not skipped:
        return ""
    lines = "\n".join(f"  - {rel}: {reason}" for rel, reason in skipped)
    return f" Skipped {len(skipped)} unreadable file(s):\n{lines}"


# Project-specific file filtering, settable once via .mcp.json so every tool
# call picks it up without repeating it per call.
DEFAULT_INCLUDE_EXTENSIONS = _parse_csv_set(os.environ.get("INDEX_INCLUDE_EXTENSIONS"))
DEFAULT_EXTRA_EXCLUDE_DIRS = _parse_csv_set(os.environ.get("INDEX_EXCLUDE_DIRS"))


def _sync_static_collection_description(collection: Optional[str], qdrant_url: str, qdrant_api_key: Optional[str]) -> None:
    """
    Applies COLLECTION_DESCRIPTION (this project's own .mcp.json env var,
    not read by mcp-server-qdrant) to the collection's native Qdrant
    metadata + the local hint cache from libs/qdrant_collection_hints.py,
    if it's configured and doesn't already match (issue #83). This is what
    lets a description be declared once in .mcp.json instead of requiring
    an explicit set_collection_description call every time a project is
    freshly wired up.

    Best-effort: catches everything, logs to stderr, never raises -- same
    discipline as compress_mcp_server.py's _log_fixed_overhead(). A broad
    except Exception (not the narrower sqlite3.Error/OSError caught inside
    qdrant_collection_hints.py itself) is deliberate here: this function
    also makes QdrantClient calls, which can raise assorted client/network
    exceptions unrelated to the local cache.

    Called from three places: server startup (against this project's own
    DEFAULT_COLLECTION/DEFAULT_QDRANT_URL), and after index_repo/sync_repo
    finish writing (guarded there to only fire when that call's resolved
    collection/qdrant_url actually match this project's own defaults --
    an explicit override to index a DIFFERENT collection or instance must
    not get this project's description stamped onto it).

    No-ops (silently, this is a normal/expected case, not a failure) when:
    COLLECTION_DESCRIPTION isn't set; collection is falsy (no COLLECTION_NAME
    configured); or the collection doesn't exist yet (nothing to attach
    metadata to). The two post-store_batch call sites cover the
    fresh-collection case themselves in the COMMON case -- but NOT
    guaranteed: store_batch itself returns immediately without creating the
    collection when entries is empty (libs/qdrant_batch_store.py's own
    `if not entries: return 0`, before _ensure_collection_exists is ever
    called) -- reachable from index_repo on an empty/no-matching-files repo,
    or from sync_repo when only files were REMOVED (changed is empty, so
    _chunk_changed_files() produces no entries, but the "nothing to do"
    early-return only fires when BOTH changed and removed are empty). This
    function handles that fine either way -- collection_exists below is
    what actually decides, and simply no-ops if it's still missing.

    When the live description already matches, this skips the (expensive,
    network) Qdrant write but still refreshes the LOCAL cache from the
    verified live value before returning (cheap, local-only) -- otherwise a
    stale or missing cache row (e.g. from a prior cache write that was
    silently swallowed per libs/qdrant_collection_hints.py's fail-open
    design, or a description changed by some other means entirely outside
    this tool) could keep list_collections showing an outdated hint
    indefinitely, since list_collections trusts an existing cache row
    without re-checking it against live Qdrant.
    """
    if not DEFAULT_COLLECTION_DESCRIPTION or not collection:
        return
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        if not call_with_retry(client.collection_exists, collection):
            return
        info = call_with_retry(client.get_collection, collection)
        current = (info.config.metadata or {}).get("description", "")
        if not isinstance(current, str):
            current = ""
        if current == DEFAULT_COLLECTION_DESCRIPTION:
            # Already correct in Qdrant -- skip the network write, but
            # still refresh the cache (cheap, local-only) in case it's
            # stale or missing. See this function's docstring for why.
            set_cached_description(qdrant_url, collection, DEFAULT_COLLECTION_DESCRIPTION)
            return
        call_with_retry(
            client.update_collection, collection_name=collection, metadata={"description": DEFAULT_COLLECTION_DESCRIPTION}
        )
        set_cached_description(qdrant_url, collection, DEFAULT_COLLECTION_DESCRIPTION)
        print(f"[claude-runway] synced COLLECTION_DESCRIPTION for '{collection}'", file=sys.stderr)
    except Exception as e:
        print(f"[claude-runway] could not sync COLLECTION_DESCRIPTION for '{collection}': {e}", file=sys.stderr)


mcp = MCPServer("codebase-indexer")


def _log_registered_tool_count():
    """
    Best-effort startup log (stderr only -- see this file's own module
    docstring: stdio transport reserves stdout for MCP protocol messages,
    so nothing here may ever call print() without file=sys.stderr) of how
    many tools this server currently registers.

    This is Track A's counterpart to compress_mcp_server.py's
    _log_fixed_overhead() -- EVALUATION.md's Track A section used to have
    no live self-reporting mechanism at all for this server's half of the
    fixed per-turn overhead (unlike Track B, which issue #22 already fixed
    this way), so its "as of this writing" tool count drifted stale twice
    in a row (issue #23 corrected 6->8 total across both servers; by the
    time of issue #52's investigation this server alone had already moved
    on to 7, since set_collection_description shipped after #23 and was
    never added to the doc). Uses libs/mcp_tool_introspect.py (issue #52/
    GROW-05) so this can't drift relative to compress_mcp_server.py's own
    count -- both go through the identical enumeration logic.

    Deliberately doesn't write to savings_ledger the way
    _log_fixed_overhead() does: the Qdrant memory piece is intentionally
    never credited toward the savings tracker (see README's "Savings
    tracker" section -- qdrant-find's counterfactual isn't observable the
    way local-compress's is), so there's no meta key for this server to
    populate. This is purely an operator-visible log line plus the live
    count `tools/report_tool_counts.py` and EVALUATION.md's Track A prose
    both point at instead of a hardcoded number.
    """
    try:
        n = tool_count(mcp)
        print(f"[claude-runway] codebase-indexer registered {n} tools", file=sys.stderr)
    except Exception as e:
        print(f"[claude-runway] could not count codebase-indexer tools: {e}", file=sys.stderr)


@mcp.tool()
async def index_repo(
    repo_path: str,
    collection: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    scope: str = "both",
    include_extensions: Optional[str] = None,
    exclude_dirs: Optional[str] = None,
    respect_gitignore: bool = True,
    chunk_lines: int = 50,
    overlap: int = 10,
    reset: bool = False,
    force: bool = False,
    # MCPServer injects context by matching the `Context` annotation on a
    # parameter name. `ctx: Context = None` keeps the framework-visible type
    # annotation while suppressing mypy's "None isn't a valid Context default"
    # complaint per-line, avoiding the need to widen to Optional[Context].
    # Same reasoning at sync_repo's ctx param below and throughout
    # compress_mcp_server.py.
    ctx: Context = None,  # type: ignore[assignment]
) -> str:
    """
    Bulk-index a repo's code and docs into a Qdrant collection so qdrant-find
    can retrieve them later. Use this when: a collection is missing or empty
    and qdrant-find is returning nothing useful, a repo was just cloned and
    hasn't been indexed yet, or the code has changed enough that the
    existing index is stale.

    collection/qdrant_url/embedding_model default to this project's
    COLLECTION_NAME/QDRANT_URL/EMBEDDING_MODEL env vars if not given -- only
    pass these explicitly to target a different project's collection.

    include_extensions/exclude_dirs are comma-separated strings (e.g.
    "'.py,.md'" / "'fixtures,generated'") and default to this project's
    INDEX_INCLUDE_EXTENSIONS/INDEX_EXCLUDE_DIRS env vars if not given.
    include_extensions overrides scope entirely when set. exclude_dirs adds
    to (not replaces) the built-in defaults (.git, node_modules, dist, etc).
    respect_gitignore additionally skips whatever the repo's own .gitignore
    excludes (silently ignored if there's no .gitignore).

    embedding_model MUST match the EMBEDDING_MODEL configured on the
    qdrant-find/qdrant-store MCP server, or search relevance will silently
    break (different models produce incompatible vector spaces).

    Set reset=true to delete all existing entries in the collection first --
    otherwise re-running this on an already-indexed repo creates duplicate
    chunks. Do NOT set reset=true if the collection is shared with other
    data (e.g. conversation-memory summaries) you don't want wiped.

    When reset=False and the collection already contains data, this tool
    returns a warning before starting the expensive embedding work (issue
    #58) so you can cancel/reset/use sync_repo rather than discovering
    duplicate risk after the full embed-and-store loop has already run.
    Pass force=True to bypass this pre-flight check and add content to
    the existing index deliberately. force has no effect when reset=True.

    WARNING -- reset=false never detects or removes chunks for files that
    were DELETED from the repo since the last run (issue #31): this tool
    only ever adds/updates chunks for whatever build_entries() finds right
    now, with no memory of what a previous call indexed. A deleted file's
    old chunks are never touched and remain in the collection indefinitely
    as stale, orphaned search results. Deliberately not fixed by having
    this tool detect and delete them itself -- sync_repo already exists
    specifically for that. But sync_repo's removal-detection only covers
    files IT has itself previously recorded in its own manifest (PR #96
    review, Copilot) -- since index_repo never writes that manifest, a
    file already deleted before sync_repo's first-ever run on this repo is
    absent from both its (empty) manifest and the current scan, so that
    first sync_repo call has nothing to compare against and can't discover
    or delete it either. sync_repo only prevents FUTURE orphans once it
    has established its own baseline; reset=true here is what's actually
    required to clean up chunks ALREADY orphaned by index_repo-only usage.
    If your repo's files can be deleted between runs, switch to sync_repo
    for ongoing re-indexing -- just don't expect its first run to also
    retroactively clean up whatever index_repo already orphaned.

    This can take a while on large repos since every chunk is embedded
    locally. Reports progress via MCP progress notifications as it runs (not
    all clients render these visibly, but it keeps the call from timing out
    on large repos). Returns a summary once indexing completes. For a big
    first-time index where you want to watch progress in a terminal, prefer
    running ingest_to_qdrant.py directly instead of this tool.
    """
    # Checked first, before even the filesystem -- cheapest possible input to
    # validate, and issue #26's whole point is catching this before any real
    # work (chunking every file in the repo) happens on a bad value.
    chunk_err = validate_chunk_params(chunk_lines, overlap)
    if chunk_err:
        return f"Error: {chunk_err}"

    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        return f"Error: {repo} is not a directory."

    collection = collection or DEFAULT_COLLECTION
    if not collection:
        return (
            "Error: no collection specified and no COLLECTION_NAME env var "
            "configured for this project's .mcp.json. Pass collection explicitly "
            "or set COLLECTION_NAME in this server's env block."
        )
    qdrant_url = qdrant_url or DEFAULT_QDRANT_URL
    qdrant_api_key = qdrant_api_key or DEFAULT_QDRANT_API_KEY
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    # Retry once on a transient dropped connection -- see issue #75/#76 /
    # libs/qdrant_retry.py. Every QdrantClient/QdrantConnector call in this
    # server is wrapped, not just the ones directly after a known CPU-bound
    # gap -- this is a long-running MCP server process, so any call can be
    # the first one after an arbitrarily long idle period since the
    # previous tool invocation. Constructed unconditionally now (not just
    # inside `if reset:`) since the non-reset branch below also needs it
    # for the file_path payload-index backfill (issue #54).
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    if reset:
        if call_with_retry(client.collection_exists, collection):
            call_with_retry(client.delete_collection, collection)
    elif call_with_retry(client.collection_exists, collection):
        # Check existing point count BEFORE starting the expensive embedding
        # work (issue #58) -- surface the duplicate-risk warning up front so
        # the user can cancel/reset/use sync_repo instead of discovering it
        # after the full embed-and-store loop has already run.
        _existing_info = call_with_retry(client.get_collection, collection)
        _existing_count = _existing_info.points_count or 0
        if _existing_count > 0 and not force:
            return (
                f"Warning: collection '{collection}' already contains "
                f"{_existing_count} points. Re-running index_repo without "
                f"reset=True may add duplicate chunks on top of those. "
                f"Options: (1) use sync_repo instead for an incremental update "
                f"that avoids duplicates (preferred for most cases), "
                f"(2) re-run with reset=True to wipe ALL collection data and "
                f"re-index cleanly (caution: wipes the entire collection, "
                f"including any conversation-memory or other non-code data "
                f"stored there), or (3) re-run with force=True to add content "
                f"to the existing index deliberately."
            )
        # FIELD_INDEXES below only takes effect when QdrantConnector's own
        # _ensure_collection_exists() creates a BRAND NEW collection -- an
        # already-existing collection (any repo indexed before this fix, or
        # simply re-run with reset=false) never hits that branch again, so
        # it needs this explicit backfill instead (issue #54).
        call_with_retry(ensure_file_path_index, client, collection)

    embedding_provider = FastEmbedProvider(embedding_model)
    connector = QdrantConnector(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection,
        embedding_provider=embedding_provider,
        field_indexes=FIELD_INDEXES,
    )

    # Collect chunks via iter_entries() with concurrent progress reporting
    # (issue #57): the original build_entries() approach held every chunk in
    # memory before a single embed/store call went out, and -- critically --
    # never reported any progress during the chunking phase, making the call
    # look completely frozen on large repos until embed/store finally started.
    #
    # The producer/consumer pattern here:
    #   - A background thread runs iter_entries() synchronously (CPU-bound
    #     file I/O stays off the event loop) and blocks on
    #     asyncio.run_coroutine_threadsafe(queue.put(...), loop) for each
    #     chunk. The bounded queue (maxsize=_CHUNK_QUEUE_MAXSIZE) provides
    #     backpressure: the producer blocks as soon as the queue is full and
    #     only proceeds when the event loop has consumed a slot -- bounding
    #     how many chunks can accumulate in the queue at any one time regardless
    #     of how fast the chunking runs. Without this the producer would call
    #     put_nowait() at generator speed while the consumer awaits an MCP
    #     round-trip per chunk, letting the queue grow to repository-size and
    #     effectively doubling peak memory usage on large repos.
    #   - The event loop awaits queue.get() in a drain loop alongside the
    #     producer thread, calling ctx.report_progress() per chunk in real
    #     time rather than in bulk at the end.
    #
    # Progress uses a single monotonically increasing counter across BOTH the
    # chunking and embed/store phases, with the combined total only known once
    # chunking is done (total_chunks * 2 -- chunking counted as N steps, then
    # embed/store counts another N steps against the same total). This avoids
    # the progress-goes-backward problem that occurs when the embed/store phase
    # reports from 0 while the prior chunking phase already reported up to N.
    #
    # A sentinel value (_DONE) signals that the producer has finished.
    # Errors from iter_entries() arrive as (None, (rel, err)) tuples (the
    # same sentinel contract as the generator itself) and are collected into
    # `skipped` without interrupting the drain.
    resolved_include_extensions = _parse_csv_set(include_extensions) or DEFAULT_INCLUDE_EXTENSIONS
    resolved_exclude_dirs = _parse_csv_set(exclude_dirs) or DEFAULT_EXTRA_EXCLUDE_DIRS

    _DONE = object()  # sentinel -- never equals any real chunk tuple
    # Caps how many items can sit in the queue at once. Note: a chunk is a
    # configurable number of lines with unbounded line length (minified files
    # can produce very large chunks), so this is a cap on item COUNT, not a
    # guaranteed memory bound -- its purpose is backpressure (slowing the
    # producer) rather than a hard memory ceiling.
    _CHUNK_QUEUE_MAXSIZE = 256

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=_CHUNK_QUEUE_MAXSIZE)

    # Threading.Event used to signal the producer to stop early (e.g. on
    # consumer cancellation). Checked between puts so the producer can exit
    # cleanly when the consumer is gone rather than blocking forever.
    import threading as _threading
    _stop_event = _threading.Event()

    def _produce_chunks():
        """Run iter_entries() in a background thread; block on a bounded queue.

        Uses asyncio.run_coroutine_threadsafe(queue.put(...), loop) rather than
        call_soon_threadsafe(queue.put_nowait, ...) so the producer actually
        BLOCKS when the queue is full -- providing real backpressure. Checks
        _stop_event between items so the consumer's finally block can signal
        early termination (e.g. cancellation), preventing a permanent block if
        the event loop stops draining the queue.
        """
        try:
            for item in iter_entries(
                repo, scope, chunk_lines, overlap,
                include_extensions=resolved_include_extensions,
                extra_exclude_dirs=resolved_exclude_dirs,
                respect_gitignore=respect_gitignore,
                # The manifest file is a caller-owned bookkeeping file, not source --
                # always exclude it by name so it can never index itself regardless of
                # extension filter, scope, or gitignore state (issue #91).
                extra_exclude_files={MANIFEST_FILENAME},
            ):
                if _stop_event.is_set():
                    break
                # .result() blocks this thread until the event loop has accepted
                # the item (i.e. a slot opened up in the bounded queue) or times
                # out -- the timeout provides a final escape hatch if the event
                # loop is gone/cancelled and can no longer drain the queue.
                fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                while not _stop_event.is_set():
                    try:
                        fut.result(timeout=1.0)
                        break
                    except TimeoutError:
                        continue  # keep waiting, checking _stop_event each second
        finally:
            # Post the sentinel so the consumer's drain loop can exit cleanly.
            # Use the same timeout-loop pattern as normal items: the check before
            # the loop guards against a stop that was already set (no point even
            # trying), and the loop itself wakes every second to re-check in case
            # _stop_event is set WHILE we are waiting for a slot in a full queue --
            # the race that the previous unbounded `.result()` couldn't handle.
            if not _stop_event.is_set():
                fut = asyncio.run_coroutine_threadsafe(queue.put(_DONE), loop)
                while not _stop_event.is_set():
                    try:
                        fut.result(timeout=1.0)
                        break
                    except TimeoutError:
                        continue  # keep waiting, checking _stop_event each second

    # pending_batch collects chunks to feed into store_batch in UPSERT_BATCH_SIZE-
    # sized groups as they arrive -- the whole-repo accumulation that issue #57
    # specifically targets (holding every chunk before storage begins) is gone.
    pending_batch: list = []
    skipped: list = []
    chunk_count = 0
    code_count = 0
    doc_count = 0
    stored_count = 0
    # Cumulative base for within-batch stored_in_batch values: store_batch()
    # reports per-batch progress starting from 1, so to get an overall
    # cumulative count we add the prior batches' total here.
    _stored_offset = 0

    async def _emit_stored_progress(stored_in_batch: int, _total_in_batch: int) -> None:
        """Report embed/store progress as chunks flow through, not after all are done.

        `stored_in_batch` counts stored entries within the CURRENT store_batch()
        call (resets to 1 on each new call). We accumulate across calls into the
        nonlocal stored_count so the progress value is monotonically increasing.
        chunk_count is at its final value by the time embed/store begins (we only
        flush once a full UPSERT_BATCH_SIZE batch has been collected, after the
        corresponding chunks were already counted in the drain loop above). Using
        chunk_count as the total makes the embed/store progress determinate.
        """
        nonlocal stored_count
        # stored_in_batch is within-batch progress (1..batch_size); to make
        # stored_count cumulative, we must not simply assign stored_in_batch.
        # Store the previous batch's final value and add the within-batch delta.
        stored_count = _stored_offset + stored_in_batch
        if ctx is not None:
            await ctx.report_progress(
                progress=stored_count, total=chunk_count,
                message=f"Stored {stored_count}/{chunk_count} chunks",
            )

    producer = loop.run_in_executor(None, _produce_chunks)
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            content, metadata_or_err = item
            if content is None:
                # Sentinel from iter_entries: (None, (rel, err)) means a file that
                # raised OSError while being read or chunked.
                skipped.append(metadata_or_err)
            else:
                pending_batch.append(Entry(content=content, metadata=metadata_or_err))
                chunk_count += 1
                # Track type counts for the final summary without retaining entries.
                if metadata_or_err["type"] == "code":
                    code_count += 1
                else:
                    doc_count += 1
                if ctx is not None:
                    await ctx.report_progress(
                        progress=chunk_count, total=None,
                        message=f"Chunked {chunk_count} chunk(s) so far...",
                    )
                # Flush a full batch into storage as chunks arrive -- avoids
                # holding the whole repo in memory before storage begins
                # (the full-accumulation case issue #57 specifically targets).
                if len(pending_batch) >= UPSERT_BATCH_SIZE:
                    _stored_offset = stored_count  # base for next batch's within-batch count
                    await store_batch(
                        connector, pending_batch,
                        collection_name=collection,
                        progress_callback=_emit_stored_progress,
                    )
                    pending_batch.clear()
    except BaseException:
        # Consumer cancelled or ctx.report_progress() raised -- signal the
        # producer to stop so it doesn't block forever waiting for a drain that
        # will never come, then wait for it to exit before re-raising.
        _stop_event.set()
        raise
    finally:
        await producer  # propagate any exception the thread raised; also ensures
                        # the executor worker is released before we proceed.

    # Flush any remaining entries that didn't fill a complete batch.
    if pending_batch:
        _stored_offset = stored_count  # base for the final partial batch
        await store_batch(
            connector, pending_batch,
            collection_name=collection,
            progress_callback=_emit_stored_progress,
        )
        pending_batch.clear()

    total = chunk_count

    # Only for THIS project's own default collection/instance -- an
    # explicit collection/qdrant_url override means the caller is indexing
    # something else, which must not get this project's own
    # COLLECTION_DESCRIPTION stamped onto it. See issue #83.
    if collection == DEFAULT_COLLECTION and qdrant_url == DEFAULT_QDRANT_URL:
        _sync_static_collection_description(collection, qdrant_url, qdrant_api_key)

    return (
        f"Indexed {total} chunks from {repo} into collection '{collection}' "
        f"({code_count} code chunks, {doc_count} doc chunks). "
        f"{'Collection was reset before indexing.' if reset else 'Existing entries were kept -- rerun with reset=true if you suspect duplicates.'}"
        f"{_skipped_summary(skipped)}"
    )


@mcp.tool()
async def sync_repo(
    repo_path: str,
    collection: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    scope: str = "both",
    include_extensions: Optional[str] = None,
    exclude_dirs: Optional[str] = None,
    respect_gitignore: bool = True,
    chunk_lines: int = 50,
    overlap: int = 10,
    ctx: Context = None,  # type: ignore[assignment]  # see index_repo's ctx comment above
) -> str:
    """
    Incrementally update a Qdrant collection to match the repo's current
    state -- only re-embeds files that changed since the last sync, deletes
    entries for files that were removed, and skips unchanged files entirely.

    include_extensions/exclude_dirs/respect_gitignore work the same as in
    index_repo and default to the same INDEX_INCLUDE_EXTENSIONS/
    INDEX_EXCLUDE_DIRS env vars. IMPORTANT: keep these consistent between
    calls for the same repo -- changing them changes which files "exist" as
    far as the manifest is concerned, which can make previously-indexed
    files look removed or vice versa.

    Use this for routine re-indexing (e.g. after editing a handful of files,
    or as a habit at the start of a session) -- it's far cheaper than
    index_repo since it doesn't touch or re-embed anything unchanged. Use
    index_repo with reset=true instead when: this is the first time indexing
    this repo, chunk_lines/overlap/embedding_model changed and everything
    needs re-chunking consistently, or the index and manifest seem to have
    drifted (e.g. after manually editing points in Qdrant).

    Tracks a small manifest file (.qdrant_index_manifest.json) in the repo
    root to know what's already indexed -- safe to gitignore, don't hand-edit it.

    collection/qdrant_url/embedding_model default to this project's
    COLLECTION_NAME/QDRANT_URL/EMBEDDING_MODEL env vars if not given.

    Reports progress via MCP progress notifications per changed file.
    """
    # See index_repo's identical check above -- issue #26.
    chunk_err = validate_chunk_params(chunk_lines, overlap)
    if chunk_err:
        return f"Error: {chunk_err}"

    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        return f"Error: {repo} is not a directory."

    collection = collection or DEFAULT_COLLECTION
    if not collection:
        return (
            "Error: no collection specified and no COLLECTION_NAME env var "
            "configured for this project's .mcp.json."
        )
    qdrant_url = qdrant_url or DEFAULT_QDRANT_URL
    qdrant_api_key = qdrant_api_key or DEFAULT_QDRANT_API_KEY
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    resolved_include_extensions = _parse_csv_set(include_extensions) or DEFAULT_INCLUDE_EXTENSIONS
    resolved_exclude_dirs = _parse_csv_set(exclude_dirs) or DEFAULT_EXTRA_EXCLUDE_DIRS

    manifest = _load_manifest(repo)
    # Run off the event loop -- this is a synchronous scan that can take
    # several seconds on a large repo (thousands of files hashed one by
    # one), and stalling the event loop for that long is the actual root
    # cause behind the Qdrant connection drop tracked in issue #75/#76:
    # Docker Desktop's vpnkit reaps an idle pooled connection during exactly
    # this kind of CPU-bound gap. asyncio.to_thread keeps the loop free
    # (and any pooled connection's keep-alive traffic flowing) while this
    # runs, instead of just retrying after the fact.
    current_hashes, hash_skipped = await asyncio.to_thread(
        compute_file_hashes,
        repo, scope,
        include_extensions=resolved_include_extensions,
        extra_exclude_dirs=resolved_exclude_dirs,
        respect_gitignore=respect_gitignore,
        # Same as index_repo -- the manifest must never hash itself into the
        # change-detection baseline (issue #91).
        extra_exclude_files={MANIFEST_FILENAME},
    )

    changed = [f for f, h in current_hashes.items() if manifest.get(f) != h]
    # Excludes hash_skipped's paths -- a file that failed to hash is neither
    # "removed" nor "changed," it's unknown, and must not be treated as
    # removed (which would delete its existing good embeddings over what may
    # be a transient read error). See files_removed_since_last_sync (issue #27).
    removed = files_removed_since_last_sync(manifest, current_hashes, [rel for rel, _ in hash_skipped])
    unchanged_count = len(current_hashes) - len(changed)

    if not changed and not removed:
        return (
            f"No changes since last sync. {unchanged_count} files already up to date in '{collection}'."
            f"{_skipped_summary(hash_skipped)}"
        )

    def _chunk_changed_files():
        entries = []
        skipped = []
        for rel in changed:
            try:
                for content, metadata in chunk_file(repo / rel, rel, chunk_lines, overlap):
                    entries.append(Entry(content=content, metadata=metadata))
            except OSError as e:
                skipped.append((rel, str(e)))
        return entries, skipped

    # Chunk BEFORE issuing any deletes (PR #90 review, Copilot) -- doing this
    # the other way around, delete-then-chunk, meant a file that hashed fine
    # but then failed to chunk (the same hash/chunk race compute_file_hashes
    # is exposed to) had its existing, still-good embeddings deleted with
    # nothing to replace them -- leaving it with ZERO searchable content
    # until a later successful sync, unlike the preservation behavior
    # hash-skipped files already got. Chunking first means we know exactly
    # which changed files actually produced replacement content before any
    # delete call ever goes out, so a chunk failure now behaves the same as
    # a hash failure: leave the existing embeddings alone. Run off the event
    # loop -- same reasoning as compute_file_hashes above: chunking every
    # changed file (file I/O + markdown heading detection) is a synchronous,
    # potentially multi-second scan.
    entries, chunk_skipped = await asyncio.to_thread(_chunk_changed_files)
    new_chunk_count = len(entries)
    chunk_skipped_paths = {rel for rel, _ in chunk_skipped}
    # Only files that actually produced replacement chunks get re-indexed --
    # chunk_skipped files stay OUT of the delete set (see above) and out of
    # this count, so the returned summary never claims a file was both
    # "re-indexed" and "skipped" (PR #90 review, Copilot).
    successfully_changed = [f for f in changed if f not in chunk_skipped_paths]

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Retry once on a transient dropped connection -- see issue #75/#76 /
    # libs/qdrant_retry.py. This specific call is the one confirmed to
    # actually trigger the bug in practice: it's the first Qdrant request
    # after compute_file_hashes (above) has just spent several seconds
    # synchronously hashing every file in a large repo, which is long enough
    # for Docker Desktop's vpnkit to reap an idle connection.
    if call_with_retry(client.collection_exists, collection):
        # Backfill the metadata.file_path payload index before the
        # delete-by-filter loop below -- this is the exact hotspot issue
        # #54 exists to fix: without an index on this field, each of the
        # filtered client.delete() calls below full-scans every point in
        # the collection instead of using the index. FIELD_INDEXES (passed
        # to QdrantConnector further down) only applies to a brand new
        # collection, which this already-existing one isn't -- see
        # ensure_file_path_index()'s docstring.
        call_with_retry(ensure_file_path_index, client, collection)

        to_delete = successfully_changed + removed
        # Batched into DELETE_BATCH_SIZE-sized OR filters instead of one
        # client.delete() per file -- see DELETE_BATCH_SIZE's comment above
        # for why. `should` (OR, not `must`/AND) matches a point if ANY of
        # the batch's file_paths match, deleting every changed/removed
        # file's points in each chunked call.
        for batch in batched(to_delete, DELETE_BATCH_SIZE):
            # Retry once on a transient dropped connection -- see issue #75 /
            # libs/qdrant_retry.py.
            call_with_retry(
                client.delete,
                collection_name=collection,
                points_selector=models.Filter(
                    should=[
                        models.FieldCondition(key="metadata.file_path", match=models.MatchValue(value=rel))
                        for rel in batch
                    ]
                ),
            )

    embedding_provider = FastEmbedProvider(embedding_model)
    connector = QdrantConnector(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection,
        embedding_provider=embedding_provider,
        field_indexes=FIELD_INDEXES,
    )

    async def _report_progress(stored, total_entries):
        if ctx is not None:
            await ctx.report_progress(progress=stored, total=total_entries, message=f"Re-indexed {stored}/{total_entries} chunks")

    # Batched embed+upsert instead of one connector.store() call per chunk --
    # see libs/qdrant_batch_store.py's module docstring for why: real-world
    # testing found thousands of individual round-trips in this loop hits
    # the same connection/request ceiling the delete loop hit before it was
    # batched, at a deterministic failure point that a bigger retry count
    # didn't move -- confirming it's a hard ceiling, not a probabilistic
    # transient drop that retrying more could fix.
    await store_batch(connector, entries, collection_name=collection, progress_callback=_report_progress)

    # Only for THIS project's own default collection/instance -- see the
    # matching guard in index_repo above and issue #83 for why.
    if collection == DEFAULT_COLLECTION and qdrant_url == DEFAULT_QDRANT_URL:
        _sync_static_collection_description(collection, qdrant_url, qdrant_api_key)

    # A file that failed to hash OR failed to chunk this round must not lose
    # its manifest entry outright (PR #90 review, Copilot): _save_manifest
    # below overwrites the whole manifest with current_hashes, so simply
    # never adding/popping its entry means a GENUINE deletion of that file
    # from disk on some later sync would go undetected forever -- it would
    # no longer appear in the manifest at all, so files_removed_since_last_sync
    # could never flag it as removed, and its now-actually-stale embeddings
    # would never get cleaned up. Restoring the file's PRIOR manifest hash
    # (its last known-good state, since we still don't know its current one)
    # keeps that removal path alive; a brand-new file with no prior entry
    # correctly stays absent -- there's nothing to preserve or clean up for
    # something that was never indexed.
    for rel in chunk_skipped_paths | {rel for rel, _ in hash_skipped}:
        if rel in manifest:
            current_hashes[rel] = manifest[rel]
        else:
            current_hashes.pop(rel, None)

    all_skipped = hash_skipped + chunk_skipped
    _save_manifest(repo, current_hashes)

    return (
        f"Synced '{collection}': {len(successfully_changed)} file(s) re-indexed ({new_chunk_count} chunks), "
        f"{len(removed)} file(s) removed, {unchanged_count} file(s) unchanged and skipped."
        f"{_skipped_summary(all_skipped)}"
    )


@mcp.tool()
def preview_index(
    repo_path: str,
    scope: str = "both",
    include_extensions: Optional[str] = None,
    exclude_dirs: Optional[str] = None,
    respect_gitignore: bool = True,
    chunk_lines: int = 50,
    overlap: int = 10,
    limit: int = 10,
) -> str:
    """
    Preview how a repo would be chunked WITHOUT writing anything to Qdrant.
    Use this before index_repo on a repo you haven't indexed before, to
    sanity-check chunk boundaries and metadata (especially file_path and
    line_range) -- and to sanity-check include_extensions/exclude_dirs are
    actually filtering the way you expect -- before committing to a real
    index run.
    """
    # See index_repo's identical check above -- issue #26.
    chunk_err = validate_chunk_params(chunk_lines, overlap)
    if chunk_err:
        return f"Error: {chunk_err}"

    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        return f"Error: {repo} is not a directory."

    entries, skipped = build_entries(
        repo, scope, chunk_lines, overlap,
        include_extensions=_parse_csv_set(include_extensions) or DEFAULT_INCLUDE_EXTENSIONS,
        extra_exclude_dirs=_parse_csv_set(exclude_dirs) or DEFAULT_EXTRA_EXCLUDE_DIRS,
        respect_gitignore=respect_gitignore,
        # Same as index_repo -- the manifest must never appear in preview
        # results (issue #91).
        extra_exclude_files={MANIFEST_FILENAME},
    )
    lines = [f"{len(entries)} total chunks would be created. Showing first {min(limit, len(entries))}:\n"]
    for content, metadata in entries[:limit]:
        preview = content[:200].replace("\n", " ")
        lines.append(f"[{metadata['type']}] {metadata['file_path']} ({metadata['line_range']}): {preview}...")
    return "\n".join(lines) + _skipped_summary(skipped)


@mcp.tool()
def get_collection_info(
    collection: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> str:
    """
    Check whether a Qdrant collection exists and how many points it holds.
    Use this to confirm a previous index_repo call actually wrote data, or
    to check whether a collection is empty before relying on qdrant-find
    against it.

    collection/qdrant_url default to this project's COLLECTION_NAME/QDRANT_URL
    env vars if not given.
    """
    collection = collection or DEFAULT_COLLECTION
    if not collection:
        return "Error: no collection specified and no COLLECTION_NAME env var configured for this project."
    qdrant_url = qdrant_url or DEFAULT_QDRANT_URL
    qdrant_api_key = qdrant_api_key or DEFAULT_QDRANT_API_KEY

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    # Retry once on a transient dropped connection -- see issue #75/#76 /
    # libs/qdrant_retry.py.
    if not call_with_retry(client.collection_exists, collection):
        return f"Collection '{collection}' does not exist yet."
    info = call_with_retry(client.get_collection, collection)
    return f"Collection '{collection}': {info.points_count} points, status={info.status}."


def _check_embedding_model_mismatch(
    client: "QdrantClient",
    collection: str,
    provider: "FastEmbedProvider",
) -> Optional[str]:
    """
    Compare the embedding provider's expected vector name and dimension against
    what the collection actually stores, returning an error string on a
    definitive mismatch or None when the check passes (or is inconclusive).

    Fails open on any exception -- a network error or unexpected config shape
    must never block a valid query; the caller already verified the collection
    exists. Only returns an error string when the mismatch is unambiguous.

    Covers three Qdrant vector-config cases:
    - None -- sparse-only collection with no dense-vector config at all; definitive
      incompatibility (not inconclusive), same as empty dict.
    - Dict[str, VectorParams] -- named vectors (mcp-server-qdrant's own format,
      using a "fast-<model-slug>" key produced by FastEmbedProvider.get_vector_name()).
      Errors on missing name (including empty dict -- sparse-only) or wrong dimension.
    - VectorParams -- unnamed/default single vector (older or third-party ingestion).
      QdrantConnector.search always passes using=get_vector_name(), which Qdrant
      cannot resolve for an unnamed/default-vector collection. Errors with a note to
      use index_repo(reset=true) -- NOT sync_repo, which does not recreate the schema.
    """
    try:
        info = call_with_retry(client.get_collection, collection)
        vectors_config = info.config.params.vectors

        expected_name = provider.get_vector_name()
        expected_size = provider.get_vector_size()

        from qdrant_client.http.models import models as _m
        if vectors_config is None or (isinstance(vectors_config, dict) and expected_name not in vectors_config):
            # vectors_config is None: sparse-only collection with no dense-vector
            # config at all (a definitive incompatibility, not inconclusive).
            # Empty dict {}: sparse-only collection -- same incompatibility.
            # Non-empty dict without the expected key: wrong model was used.
            # All three cases: QdrantConnector.search would fail trying to resolve
            # using=expected_name against a collection that doesn't have it.
            present = sorted(vectors_config.keys()) if isinstance(vectors_config, dict) else []
            hint = (
                f", but the collection has: {present}" if present
                else " (collection has no dense vectors)"
            )
            return (
                f"Error: embedding model mismatch for collection '{collection}'. "
                f"The model '{provider.model_name}' produces a vector named "
                f"'{expected_name}'{hint}. "
                f"Pass the embedding_model that matches the one used when this "
                f"collection was indexed (check the other repo's .mcp.json)."
            )
        elif isinstance(vectors_config, dict):
            stored_size = vectors_config[expected_name].size
            if stored_size != expected_size:
                return (
                    f"Error: embedding dimension mismatch for collection '{collection}'. "
                    f"Model '{provider.model_name}' produces {expected_size}-dim vectors, "
                    f"but the collection's '{expected_name}' vector has {stored_size} dims. "
                    f"Pass the embedding_model that matches the one used when indexing."
                )
        elif isinstance(vectors_config, _m.VectorParams):
            # Single unnamed VectorParams: QdrantConnector.search always passes
            # using=get_vector_name(), which Qdrant cannot resolve for an unnamed/
            # default-vector collection. Return an error immediately rather than
            # letting it fail downstream with an opaque Qdrant error.
            # Recovery: index_repo(reset=true) to wipe and recreate with named-vector
            # schema. sync_repo does NOT recreate the schema, so pointing users there
            # would silently fail or (worse) delete old points before the upsert fails.
            return (
                f"Error: collection '{collection}' uses an unnamed/default vector "
                f"format (not compatible with mcp-server-qdrant's named-vector search). "
                f"Re-index it using index_repo(reset=true) so it stores vectors under "
                f"the '{expected_name}' name that find_in_collection expects."
            )
    except Exception:
        # Network error, unexpected config shape, model-description lookup
        # failure -- don't block a potentially valid query.
        return None
    return None


@mcp.tool()
async def find_in_collection(
    query: str,
    collection: str,
    limit: int = 10,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> str:
    """
    Semantically search a DIFFERENT project's Qdrant collection than this
    project's own -- e.g. from a frontend repo, search an already-indexed
    backend repo's collection to understand how an API it calls is
    implemented on the other side. Use this instead of qdrant-find whenever
    the question requires another repo's codebase, not this one's.

    Unlike qdrant-find (which is locked to this project's own collection),
    collection is REQUIRED here. If the user names an exact collection
    directly in their prompt, pass it through as given -- don't guess or
    substitute a different name. If you don't know the other repo's exact
    collection name, call list_collections first rather than guessing.

    qdrant_url/qdrant_api_key default to this project's QDRANT_URL/
    QDRANT_API_KEY env vars -- only pass these explicitly if the other
    repo's collection lives on a different Qdrant instance. embedding_model
    defaults to this project's EMBEDDING_MODEL env var and MUST match
    whatever model the OTHER collection was actually indexed with -- check
    the other repo's own .mcp.json if unsure. A mismatch is detected before
    querying and returns an actionable error rather than silently returning
    irrelevant results; the check fails open on network errors so a transient
    Qdrant hiccup never blocks a valid query.
    """
    if not collection:
        return "Error: collection is required -- call list_collections to see what's available."
    if limit < 1:
        return f"Error: limit must be 1 or greater (got {limit})."

    qdrant_url = qdrant_url or DEFAULT_QDRANT_URL
    qdrant_api_key = qdrant_api_key or DEFAULT_QDRANT_API_KEY
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    # Retry once on a transient dropped connection -- see issue #75/#76 /
    # libs/qdrant_retry.py.
    if not call_with_retry(client.collection_exists, collection):
        return (
            f"Error: collection '{collection}' does not exist at {qdrant_url}. "
            "Call list_collections to see what's available."
        )

    embedding_provider = FastEmbedProvider(embedding_model)

    # Guard against silently-meaningless results from an embedding model
    # mismatch -- different models produce incompatible vector spaces, so a
    # wrong model returns plausible-looking garbage with no error (issue #56).
    # Checked AFTER constructing the provider (needed for get_vector_name/size)
    # and BEFORE creating the connector or running the query. Fails open: any
    # exception inside the check is swallowed and the query proceeds normally.
    mismatch_error = _check_embedding_model_mismatch(client, collection, embedding_provider)
    if mismatch_error:
        return mismatch_error

    connector = QdrantConnector(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection,
        embedding_provider=embedding_provider,
    )
    entries = await async_call_with_retry(connector.search, query, collection_name=collection, limit=limit)
    if not entries:
        return f"No results found in collection '{collection}' for query '{query}'."

    lines = [f"Results for '{query}' in collection '{collection}':"]
    for entry in entries:
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        lines.append(f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>")
    return "\n".join(lines)


@mcp.tool()
def set_collection_description(
    collection: str,
    description: str,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> str:
    """
    Set or update a short, one-line description hint for a collection, so a
    future list_collections call can surface it and let a model pick the
    right collection for find_in_collection on its own -- instead of the
    user always having to name it explicitly (issue #81). Works on ANY
    already-indexed collection, including this project's own -- other
    sessions benefit from this project's collection having a hint too.

    collection is REQUIRED and must already exist (call list_collections
    first if unsure of the exact name). description should be a short
    sentence describing what the collection actually contains -- this is
    what gets shown to a model deciding which collection to search, not
    read by a human browsing a UI.

    Stored as native Qdrant collection metadata (update_collection's
    metadata field), which Qdrant itself merges rather than replaces --
    confirmed directly against a live instance -- so this never clobbers
    any other metadata a collection might carry. Also written straight into
    the local hint cache so the very next list_collections call (in this
    session or any other) sees the new value with no stale window.

    qdrant_url/qdrant_api_key default to this project's QDRANT_URL/
    QDRANT_API_KEY env vars -- only pass these explicitly to target a
    different Qdrant instance.
    """
    if not collection:
        return "Error: collection is required -- call list_collections to see what's available."
    qdrant_url = qdrant_url or DEFAULT_QDRANT_URL
    qdrant_api_key = qdrant_api_key or DEFAULT_QDRANT_API_KEY

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    # Retry once on a transient dropped connection -- see issue #75/#76 /
    # libs/qdrant_retry.py.
    if not call_with_retry(client.collection_exists, collection):
        return (
            f"Error: collection '{collection}' does not exist at {qdrant_url}. "
            "Call list_collections to see what's available."
        )

    call_with_retry(client.update_collection, collection_name=collection, metadata={"description": description})
    set_cached_description(qdrant_url, collection, description)
    return f"Description set for collection '{collection}'."


@mcp.tool()
def list_collections(
    include_counts: bool = True,
    include_descriptions: bool = True,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> str:
    """
    List every collection on the connected Qdrant instance -- use this to
    discover another repo's exact collection name before calling
    find_in_collection, or to sanity-check this project's own collection
    exists.

    include_counts (default true) additionally fetches each collection's
    point count, at the cost of one extra Qdrant request per collection --
    set false for a fast names-only listing if the instance has many
    collections and you don't need counts.

    include_descriptions (default true) surfaces each collection's stored
    hint (set via set_collection_description), so you can often pick the
    right collection for find_in_collection from this listing alone instead
    of asking the user to name it. A REAL hint is cached locally after the
    first fetch (persists across sessions/restarts) -- when include_counts
    is also true, a cache miss reuses that SAME per-collection request
    rather than making a second one, so the two flags together cost no more
    than include_counts alone once a collection's hint has been fetched
    anywhere. Collections with no hint set are deliberately NOT cached as
    "no description" -- they're re-checked live on every call (free when
    include_counts is also true, since that request already happens; one
    extra request per undescribed collection when include_counts=false)
    so a hint added outside this tool (e.g. directly via the Qdrant API)
    shows up on the very next call instead of staying invisible.

    qdrant_url/qdrant_api_key default to this project's QDRANT_URL/
    QDRANT_API_KEY env vars -- only pass these explicitly to look at a
    different Qdrant instance.
    """
    qdrant_url = qdrant_url or DEFAULT_QDRANT_URL
    qdrant_api_key = qdrant_api_key or DEFAULT_QDRANT_API_KEY
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    # Retry once on a transient dropped connection -- see issue #75/#76 /
    # libs/qdrant_retry.py.
    names = sorted(c.name for c in call_with_retry(client.get_collections).collections)
    if not names:
        return f"No collections found at {qdrant_url}."

    # Cache holds only genuine hits (see get_cached_descriptions's own
    # docstring) -- a name absent here still needs a live fetch below.
    descriptions = get_cached_descriptions(qdrant_url, names) if include_descriptions else {}

    lines = [f"Collections at {qdrant_url}:"]
    for name in names:
        marker = " (this project's own)" if name == DEFAULT_COLLECTION else ""
        needs_description_fetch = include_descriptions and name not in descriptions
        info = None
        if include_counts or needs_description_fetch:
            info = call_with_retry(client.get_collection, name)
        description = descriptions.get(name, "")
        if needs_description_fetch:
            description = (info.config.metadata or {}).get("description", "") if info else ""
            if not isinstance(description, str):
                description = ""
            # Only cache a REAL hint. An empty result is used for this one
            # call's display but deliberately not written to the cache --
            # keeps the cache limited to collections someone actually
            # described (not a growing pile of "nothing here" rows), and
            # means a description added outside set_collection_description
            # (e.g. directly via the Qdrant API/UI) is picked up on the very
            # next call instead of staying invisible until this tool is
            # called explicitly. Costs an extra per-call fetch only when
            # include_counts=False -- with the default include_counts=True,
            # this same get_collection() call already happens for the point
            # count regardless.
            if description:
                set_cached_description(qdrant_url, name, description)

        hint = f" -- {description}" if include_descriptions and description else ""
        if include_counts and info:
            lines.append(f"- {name}: {info.points_count} points{marker}{hint}")
        else:
            lines.append(f"- {name}{marker}{hint}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Covers the common case: the collection already exists from a
    # previous index. index_repo/sync_repo's own call sites (above) cover
    # a brand-new collection that doesn't exist yet at this point. See
    # issue #83.
    _sync_static_collection_description(DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, DEFAULT_QDRANT_API_KEY)
    _log_registered_tool_count()
    mcp.run()
