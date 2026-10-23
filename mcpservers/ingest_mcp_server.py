#!/usr/bin/env python3
"""
MCP server exposing repo indexing as callable tools, so Claude (in Claude
Code) can trigger re-indexing on demand -- e.g. right after cloning a repo,
or when it notices qdrant-find is coming back empty -- instead of you
running ingest_to_qdrant.py by hand every time.

Project minimum: Python 3.12 (policy floor; the dependency chain supports >=3.10).

Setup:
    pip install "mcp[cli]" qdrant-client fastembed pydantic_settings --break-system-packages

Register in .mcp.json (adjust the path to wherever you save this file). Set
QDRANT_URL / COLLECTION_NAME / EMBEDDING_MODEL here -- these are the SAME env
var names mcpservers/qdrant_mcp_server.py reads, so one project config drives both
servers and they can't drift apart:
{
  "mcpServers": {
    "codebase-indexer": {
      "command": "python",
      "args": ["/absolute/path/to/mcpservers/ingest_mcp_server.py"],
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
import concurrent.futures
import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from mcp.server.mcpserver import MCPServer, Context
from qdrant_connector import QdrantConnector, Entry
from qdrant_embeddings import FastEmbedProvider
from qdrant_client import QdrantClient, models

from qdrant_ingest_lib import (
    MANIFEST_FILENAME,
    TokenBudgetError,
    batched,
    build_entries,
    chunk_file,
    chunk_position,
    compute_file_hashes,
    ensure_persistent_fastembed_cache,
    file_has_indexable_content,
    files_removed_since_last_sync,
    format_largest_files,
    ingest_env_defaults,
    iter_entries,
    parse_csv_set,
    validate_token_overlap,
)
from sync_manifest_lib import (
    UNVERIFIED_HASH,
    ManifestLock,
    discard_manifest,
    index_baseline,
    invalidation_error,
    load_manifest,
    save_manifest,
    unindexed_hashed_files,
)
from qdrant_retry import call_with_retry, async_call_with_retry
from qdrant_batch_store import store_batch, ensure_file_path_index, FIELD_INDEXES, UPSERT_BATCH_SIZE
from qdrant_index_prep import prepare_collection_for_index
from qdrant_collection_hints import (
    compute_search_limit,
    get_cached_descriptions,
    get_cached_search_limit,
    set_cached_description,
    set_cached_search_limit,
)
from qdrant_compact_marker import collection_is_marked_compact, is_marked_compact
from qdrant_model_check import check_embedding_model_mismatch
from qdrant_credentials import resolve_qdrant_credentials
from mcp_tool_introspect import tool_count
import memory_bank_lib as mb

# Must run before any FastEmbedProvider(...) construction below (index_repo,
# sync_repo, find_in_collection each make their own) -- see the
# function's own docstring for why this can't just be a .mcp.json env value.
ensure_persistent_fastembed_cache()

# MANIFEST_FILENAME is imported from qdrant_ingest_lib so
# ingest_to_qdrant.py excludes the same file -- see its comment there.

# sync_repo batches its per-file delete filter into chunks of this size
# instead of one client.delete() call per changed/removed file:
# a live repro against this repo's own 10,532-file monorepo
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


# The manifest helpers live in libs/sync_manifest_lib.py so the CLI's
# --write-manifest uses the same code. They keep their underscore
# names here: this module's own callers and tests refer to them (and patch them)
# as _ims.<name>, and index_repo/sync_repo always call them through these module
# globals.
_load_manifest = load_manifest
_save_manifest = save_manifest
_discard_manifest = discard_manifest
_index_baseline = index_baseline


def _unindexed_hashed_files(repo: Path, current_hashes: dict, chunk_skipped: list, chunked_files) -> set:
    # file_has_indexable_content is looked up here, at call time, so it stays
    # patchable as this module's own name.
    return unindexed_hashed_files(
        repo, current_hashes, chunk_skipped, chunked_files, has_indexable_content=file_has_indexable_content
    )


# How many non-chunk points the pre-rebuild warning quotes, and how long each
# quote is. Display only: the count it reports is exact.
_NON_CHUNK_SAMPLE_SIZE = 3
_NON_CHUNK_SAMPLE_CHARS = 120

# Points per scroll page when listing the indexed files.
_SCROLL_PAGE_SIZE = 1000

_CHUNK_TYPES = ["code", "doc"]


def _non_chunk_points_warning(client: QdrantClient, collection: str) -> str:
    """
    Before a rebuild: describe the points in `collection` that the
    indexer did not write and that the rebuild may delete. Returns "" when
    there are none.

    sync_repo deletes a file's old points by metadata.file_path alone, excluding
    only memory-bank and conversation-compact points. qdrant-store and
    qdrant-find accept any content, so a project collection can hold a note with
    a matching file_path, and it would go with the file's chunks. A warning, not
    a block: the rebuild still has to run, and most collections hold none.

    "Not a chunk" means metadata.type is not code or doc (including having no
    type at all). Protected points are left out of the count because the
    delete already skips them.
    """
    non_chunk = models.Filter(
        # The shared exclusion is nested as a condition of its own: points that
        # are not memory-bank or compact points AND are not code/doc chunks.
        must=[mb.memory_bank_exclusion_filter()],
        must_not=[models.FieldCondition(key="metadata.type", match=models.MatchAny(any=_CHUNK_TYPES))],
    )
    count = call_with_retry(client.count, collection_name=collection, count_filter=non_chunk, exact=True).count
    if not count:
        return ""
    points, _ = call_with_retry(
        client.scroll,
        collection_name=collection,
        scroll_filter=non_chunk,
        limit=_NON_CHUNK_SAMPLE_SIZE,
        with_payload=True,
        with_vectors=False,
    )
    samples = []
    for point in points:
        payload = point.payload or {}
        samples.append(json.dumps(payload.get("metadata"))[:_NON_CHUNK_SAMPLE_CHARS] or "no metadata")
    sample_text = "; ".join(samples)
    return (
        f" Warning: '{collection}' holds {count} point(s) that are not indexer chunks (metadata.type is not "
        f"code or doc), such as {sample_text}. Any of them whose metadata.file_path names a re-indexed file was "
        f"deleted with that file's old chunks."
    )


def _confirmed_absent(path: Path) -> bool:
    """
    True only if `path` is positively known not to exist: lstat
    says no such file, or a parent component is not a directory. Any other
    failure (permissions on a parent directory, an I/O error) is "cannot tell"
    and returns False, so the caller keeps whatever it was about to delete.
    A dangling symlink is a directory entry, so it is not absent either.
    """
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def _delete_orphaned_chunks(
    client: QdrantClient, collection: str, repo: Path, scanned=(), preserved_out: Optional[set] = None
) -> list:
    """
    After a rebuild: delete the chunks of files that no longer
    exist on disk, and return their paths (sorted).

    A rebuild starts from an empty manifest, so a file deleted from the repo
    before it is in neither the manifest nor the scan, and its old chunks are
    never removed by the per-file delete (the same gap index_repo documents).
    Existence is tested on disk, NOT by membership in this sync's scan, so a
    sync_repo(scope=...), an extension filter or a gitignore difference can
    never remove the chunks of a file that is still there. A file counts as
    gone only when os.lstat positively reports a missing path (_confirmed_absent):
    lstat does not follow symlinks, so a symlink whose target has disappeared is
    still a directory entry that hashing skips as unreadable, and its chunks
    must be left alone like any other unreadable file's (Path.exists() follows
    the link and would say it is gone); and any other failure -- a permission or
    I/O error on a directory above the file -- means "cannot tell", which must
    keep the chunks, where os.path.lexists would report False and delete them. Only indexer chunks
    (metadata.type of code or doc) are listed, and the delete keeps the
    memory-bank/compact exclusion the per-file delete uses.

    A file that is indexed and not confirmed gone, yet not in `scanned` (this
    rebuild's own scan), is one whose old chunks were kept although the new
    manifest has no entry for it. When `preserved_out` is given
    those paths are added to it, so the caller can record them and a later sync
    can notice when one is deleted. Without that record, deleting such a file
    would leave its chunks searchable until the next rebuild. The return value
    is unchanged: the orphans that were deleted.

    Listing is a scroll over the chunk points' metadata.file_path (a keyword
    payload index exists on it) rather than a facet, which is not available on
    every Qdrant version this runs against.
    """
    chunk_filter = models.Filter(
        must=[models.FieldCondition(key="metadata.type", match=models.MatchAny(any=_CHUNK_TYPES))]
    )
    indexed_files: set = set()
    offset = None
    while True:
        points, offset = call_with_retry(
            client.scroll,
            collection_name=collection,
            scroll_filter=chunk_filter,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=models.PayloadSelectorInclude(include=["metadata.file_path"]),
            with_vectors=False,
        )
        for point in points:
            file_path = ((point.payload or {}).get("metadata") or {}).get("file_path")
            if isinstance(file_path, str) and file_path:
                indexed_files.add(file_path)
        if offset is None:
            break
    orphans = sorted(rel for rel in indexed_files if _confirmed_absent(repo / rel))
    if preserved_out is not None:
        scanned_set = set(scanned)
        gone = set(orphans)
        preserved_out.update(rel for rel in indexed_files if rel not in gone and rel not in scanned_set)
    for batch in batched(orphans, DELETE_BATCH_SIZE):
        call_with_retry(
            client.delete,
            collection_name=collection,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="metadata.type", match=models.MatchAny(any=_CHUNK_TYPES))],
                should=[
                    models.FieldCondition(key="metadata.file_path", match=models.MatchValue(value=rel))
                    for rel in batch
                ],
                must_not=mb.memory_bank_exclusion_filter().must_not,
            ),
        )
    return orphans


async def _announce_warning(ctx, message: str) -> None:
    """
    Deliver a warning NOW, ahead of the tool's final result: the
    pre-rebuild non-chunk-points warning has to reach the caller before the
    per-file deletes it is about, not only in a result string returned after
    they ran (and, if storing then fails, never). Sent as an MCP log
    notification and to stderr; the same text is also kept in the result.

    Fail-open: a notification that cannot be sent must never fail or block the
    sync. Some clients do not render MCP notifications at all (see the README's
    Known limitations), which is why the result repeats it.
    """
    print(f"[claude-runway] {message}", file=sys.stderr)
    if ctx is None:
        return
    try:
        await ctx.warning(message)
    except Exception as e:
        print(f"[claude-runway] could not send the warning as an MCP notification: {e}", file=sys.stderr)


class _HashScanStopped(Exception):
    """Raised inside the hash worker thread when its call was cancelled, to end
    the scan at its next report instead of reading the rest of the repo."""


# How long the hash worker waits on one pending report before checking whether
# its call was cancelled. Short enough that a cancelled call frees its thread
# promptly, long enough not to spin.
_REPORT_WAIT_SLICE_SECONDS = 0.2


async def _hash_files_with_progress(ctx, repo: Path, scope: str, **hash_kwargs) -> tuple:
    """
    compute_file_hashes off the event loop, reporting progress while it runs.
    Returns its (hashes, skipped).

    Hashing is the first phase of sync_repo (and of index_repo when it writes
    the manifest baseline) and reads every matching file in the repo; on a
    first sync that is thousands of files with nothing else to report yet, so
    without this the call looks frozen for the whole stretch -- the thing MCP
    progress notifications exist to prevent. The scan is a lazy directory
    walk, so no total is known: messages carry a running count only.

    The worker thread cannot await, so each report is handed to the event loop
    and waited for. That keeps at most one notification in flight, and a report
    that raises (the client went away) stops the scan at its next report
    instead of letting it read the rest of the repo, the same meaning a failing
    report has in index_repo's chunk loop.

    Cancelling the call must free the worker too. Cancelling the await below
    does not stop its thread, and the report it is waiting on is a separate
    task that cancellation never touches, so a report stuck behind a stalled
    transport would hold the thread (and, repeated, the executor) for good.
    The `stop` event is set however the await ends; the worker waits on a
    report in short slices, and on seeing it set cancels that report and ends
    the scan. With ctx None nothing is reported and the scan runs exactly as it
    did before.
    """
    if ctx is None:
        return await asyncio.to_thread(compute_file_hashes, repo, scope, **hash_kwargs)

    loop = asyncio.get_running_loop()
    stop = threading.Event()

    def _on_progress(visited: int) -> None:
        if stop.is_set():
            raise _HashScanStopped()
        pending = asyncio.run_coroutine_threadsafe(
            ctx.report_progress(progress=visited, total=None, message=f"Hashing files: {visited} checked so far"),
            loop,
        )
        # wait() rather than result(timeout=...): result() raises TimeoutError
        # for an expired slice AND re-raises a TimeoutError the report itself
        # raised, and the two cannot be told apart, so a report that timed out
        # would be retried forever. wait() never raises for a slice; once the
        # future is done, result() gives the report's own value or exception.
        while True:
            done, _ = concurrent.futures.wait([pending], timeout=_REPORT_WAIT_SLICE_SECONDS)
            if done:
                pending.result()
                return
            if stop.is_set():
                pending.cancel()
                raise _HashScanStopped()

    # Before the walk starts, so even a repo with fewer files than one report
    # interval announces the phase it is in.
    await ctx.report_progress(progress=0, total=None, message="Hashing files to find what changed")
    try:
        hashes, skipped = await asyncio.to_thread(
            compute_file_hashes, repo, scope, on_progress=_on_progress, **hash_kwargs
        )
    finally:
        # Also on success, where it is a no-op: the worker has already returned.
        stop.set()
    await ctx.report_progress(
        progress=len(hashes) + len(skipped), total=None,
        message=f"Hashed {len(hashes) + len(skipped)} file(s)",
    )
    return hashes, skipped


def _preserved_paths_gone(repo: Path, paths) -> list:
    """
    Which of the preserved-but-untracked `paths` are confirmed gone
    from disk, sorted. Uses _confirmed_absent, the lstat-based check the rebuild
    sweep uses, so a permission or I/O error means "cannot tell" and the path is
    kept for the next sync rather than having its chunks deleted.
    """
    return sorted(rel for rel in paths if _confirmed_absent(repo / rel))


def _preserved_note(preserved: set) -> str:
    """The sentence a rebuild adds to its result about files it kept but did not scan."""
    if not preserved:
        return ""
    shown = ", ".join(sorted(preserved)[:5]) + (f", and {len(preserved) - 5} more" if len(preserved) > 5 else "")
    return (
        f" Rebuild kept the chunks of {len(preserved)} file(s) still on disk but outside this scan and now watches "
        f"them, so deleting one removes its chunks on the next sync: {shown}."
    )


def _preserved_gone_note(gone: list) -> str:
    """The sentence a sync adds to its result after removing a watched file's chunks."""
    if not gone:
        return ""
    shown = ", ".join(gone[:5]) + (f", and {len(gone) - 5} more" if len(gone) > 5 else "")
    return f" Also removed the chunks of {len(gone)} previously kept file(s) since deleted from disk: {shown}."


def _orphans_note(orphans: list) -> str:
    """The sentence a rebuild adds to its result after removing orphaned chunks."""
    if not orphans:
        return ""
    shown = ", ".join(orphans[:5]) + (f", and {len(orphans) - 5} more" if len(orphans) > 5 else "")
    return f" Rebuild also removed the chunks of {len(orphans)} file(s) no longer on disk: {shown}."

# The chunk_lines default every indexing tool declares. Compared against so a
# caller who passes something else is told it is ignored.
_DEFAULT_CHUNK_LINES = 50


def _chunk_lines_note(chunk_lines: int) -> str:
    """chunk_lines no longer sets the chunk size, but the tools
    still accept it so existing calls do not break. A non-default value is a
    caller expecting the old behavior: say so instead of ignoring it silently."""
    if chunk_lines == _DEFAULT_CHUNK_LINES:
        return ""
    return (
        f" Note: chunk_lines={chunk_lines} was ignored -- chunks are cut by the embedding model's token window, "
        f"not by line count."
    )

# Same env var names mcpservers/qdrant_mcp_server.py reads, so a single project .mcp.json
# config keeps this server and the qdrant-find/qdrant-store server aligned.
# Resolved through qdrant_ingest_lib.ingest_env_defaults -- the
# same helper ingest_to_qdrant.py's argparse defaults come from, so the two
# entry points can't drift apart on these again.
_ENV_DEFAULTS = ingest_env_defaults()
DEFAULT_QDRANT_URL = _ENV_DEFAULTS["qdrant_url"]
DEFAULT_QDRANT_API_KEY = _ENV_DEFAULTS["qdrant_api_key"]
DEFAULT_COLLECTION = os.environ.get("COLLECTION_NAME")
DEFAULT_EMBEDDING_MODEL = _ENV_DEFAULTS["embedding_model"]

# Optional, this repo's own -- not read by the qdrant server. Declared once
# alongside COLLECTION_NAME so a project's description doesn't require a
# one-off set_collection_description call; applied by
# _sync_static_collection_description() below.
DEFAULT_COLLECTION_DESCRIPTION = os.environ.get("COLLECTION_DESCRIPTION")

# Same env var name memory_bank_mcp_server.py reads -- used
# here ONLY to refuse index_repo/sync_repo ever targeting the shared
# memory-bank collection, not to write into it. Defaults to the same
# "memory-bank" default so the guard still applies out of the box even on a
# project that hasn't explicitly configured this var.
DEFAULT_MEMORY_BANK_COLLECTION = _ENV_DEFAULTS["memory_bank_collection"]


def _resolve_credentials(
    qdrant_url: Optional[str], qdrant_api_key: Optional[str]
) -> tuple[str, Optional[str]]:
    """
    Resolves a tool call's qdrant_url/qdrant_api_key against this project's
    QDRANT_URL/QDRANT_API_KEY. Every tool that takes those two inputs goes
    through here, never an inline `x or DEFAULT_...` pair: qdrant_url is tool
    input (text the model picks, so a prompt-injected URL is a real path), and
    falling back to the configured key for ANY url would send that credential
    to whatever host a call names. The configured key is only used with the
    configured URL; see libs/qdrant_credentials.py.

    The defaults are read from the module globals at call time, not bound at
    import, so a test (or a future reload) that changes them is honoured.
    """
    return resolve_qdrant_credentials(
        qdrant_url,
        qdrant_api_key,
        default_url=DEFAULT_QDRANT_URL,
        default_api_key=DEFAULT_QDRANT_API_KEY,
    )


def _skipped_summary(skipped: list) -> str:
    """
    Formats a (rel_path, error) skip list from build_entries/
    compute_file_hashes into a trailing summary sentence -- empty string if
    nothing was skipped, so callers can just append this to their return
    message unconditionally.
    """
    if not skipped:
        return ""
    lines = "\n".join(f"  - {rel}: {reason}" for rel, reason in skipped)
    return f" Skipped {len(skipped)} unreadable file(s):\n{lines}"


# Project-specific file filtering, settable once via .mcp.json so every tool
# call picks it up without repeating it per call.
DEFAULT_INCLUDE_EXTENSIONS = parse_csv_set(_ENV_DEFAULTS["include_extensions"])
DEFAULT_EXTRA_EXCLUDE_DIRS = parse_csv_set(_ENV_DEFAULTS["exclude_dirs"])


def _sync_static_collection_description(collection: Optional[str], qdrant_url: str, qdrant_api_key: Optional[str]) -> None:
    """
    Applies COLLECTION_DESCRIPTION (this project's own .mcp.json env var,
    not read by the qdrant server) to the collection's native Qdrant
    metadata + the local hint cache from libs/qdrant_collection_hints.py,
    if it's configured and doesn't already match. This is what
    lets a description be declared once in .mcp.json instead of requiring
    an explicit set_collection_description call every time a project is
    freshly wired up.

    Best-effort: catches everything, logs to stderr, never raises -- same
    discipline as this file's startup tool-count log. A broad
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
    `if not entries: return 0`, before ensure_collection_exists is ever
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


def _recompute_search_limit(collection: Optional[str], qdrant_url: str, qdrant_api_key: Optional[str]) -> None:
    """
    Recomputes the collection's default search limit from its current point
    count and writes it through to BOTH Qdrant collection metadata
    (`search_limit`, the source of truth) and the local hint cache -- but
    each only if it actually differs. Shared by
    index_repo and sync_repo: a fresh index is often the single biggest
    point-count jump a collection sees (0 -> N), and sync_repo covers
    incremental drift.

    Unlike _sync_static_collection_description, this is NOT guarded to this
    project's own default collection: any collection these tools index has a
    point count that should drive its limit.

    Fails open: catches everything, logs to stderr, never raises, so a
    search_limit hiccup can never turn an otherwise-successful sync/index
    into a reported failure. The metadata write sends ONLY the search_limit
    key: Qdrant merges metadata patches (same contract as
    set_collection_description), and resending a read-back snapshot would
    risk overwriting a concurrent change to another key.

    The two stores are compared independently: Qdrant metadata is compared
    against the new value on its own, and the cache against it on its own, so
    a stale or missing cache row self-heals without a needless network write
    (and vice versa). No-ops if the collection doesn't exist (e.g.
    store_batch skipped creating it for an empty entry list).
    """
    if not collection:
        return
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        if not call_with_retry(client.collection_exists, collection):
            return
        info = call_with_retry(client.get_collection, collection)
        new_limit = compute_search_limit(info.points_count or 0)
        if (info.config.metadata or {}).get("search_limit") != new_limit:
            call_with_retry(
                client.update_collection, collection_name=collection, metadata={"search_limit": new_limit}
            )
            print(f"[claude-runway] search_limit for '{collection}' set to {new_limit}", file=sys.stderr)
        if get_cached_search_limit(qdrant_url, collection) != new_limit:
            set_cached_search_limit(qdrant_url, collection, new_limit)
    except Exception as e:
        print(f"[claude-runway] could not recompute search_limit for '{collection}': {e}", file=sys.stderr)


# Hardcoded last-resort search limit for find_in_collection when the caller
# passes none and no per-collection value is stored anywhere.
DEFAULT_SEARCH_LIMIT = 10


def _valid_search_limit(value: object) -> Optional[int]:
    """A stored limit is usable only if it's a real positive int (bool is an
    int subclass, so it's excluded explicitly)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _resolve_search_limit(client: QdrantClient, collection: str, qdrant_url: str) -> int:
    """
    Resolves find_in_collection's limit when the caller passed none:
    local cache first, then live Qdrant collection
    metadata (`search_limit`, the source of truth written by
    _recompute_search_limit), then DEFAULT_SEARCH_LIMIT. A live hit
    backfills the cache; a miss deliberately does not (same "cache only
    genuine values" rule as descriptions), so a limit written later is picked
    up on the next call.

    Fails open to the default on any error: a limit-lookup hiccup must never
    block an otherwise-valid query.
    """
    try:
        cached = _valid_search_limit(get_cached_search_limit(qdrant_url, collection))
        if cached is not None:
            return cached
        info = call_with_retry(client.get_collection, collection)
        live = _valid_search_limit((info.config.metadata or {}).get("search_limit"))
        if live is not None:
            set_cached_search_limit(qdrant_url, collection, live)
            return live
    except Exception as e:
        print(f"[claude-runway] could not resolve search_limit for '{collection}': {e}", file=sys.stderr)
    return DEFAULT_SEARCH_LIMIT


mcp = MCPServer("codebase-indexer")


def _log_registered_tool_count():
    """
    Best-effort startup log (stderr only -- see this file's own module
    docstring: stdio transport reserves stdout for MCP protocol messages,
    so nothing here may ever call print() without file=sys.stderr) of how
    many tools this server currently registers.

    Gives EVALUATION.md's tool count a live source instead of a hand-typed
    number that drifted stale. Uses
    libs/mcp_tool_introspect.py (GROW-05), the same enumeration
    `tools/report_tool_counts.py` uses, so the two can't drift.

    This is purely an operator-visible log line plus the live count
    `tools/report_tool_counts.py` and EVALUATION.md's Track A prose point at
    instead of a hardcoded number. (compress_mcp_server.py used to log a
    schema-size estimate and store it in the savings ledger; that was
    removed -- see tools/measure_startup_context.py for the measured
    replacement.)
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
    # annotation while suppressing a "None isn't a valid Context default"
    # complaint per-line, avoiding the need to widen to Optional[Context].
    # Confirmed still needed under Pyright (which replaced mypy):
    # removing the ignore surfaces Pyright's own equivalent
    # reportArgumentType error here, just phrased differently than mypy's.
    # Same reasoning at sync_repo's ctx param below and throughout
    # compress_mcp_server.py/memory_bank_mcp_server.py.
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
    pass these explicitly to target a different project's collection. The
    configured QDRANT_API_KEY is only sent to the configured QDRANT_URL; a
    different qdrant_url needs its own qdrant_api_key if it requires one.

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

    Set reset=true to delete all existing UNPROTECTED entries (everything
    except the protected sources listed below) in the collection first -- otherwise re-running this on an already-indexed repo
    creates duplicate chunks. memory-bank points (metadata.source ==
    "memory-bank") and conversation compacts stored
    (metadata.source == "conversation-compact") are always preserved
    regardless -- but do NOT set reset=true if the collection is shared with
    OTHER non-code data (e.g. qdrant-store notes, or conversation-compacts
    stored before the source marker existed, which carry none) you don't want wiped;
    those still get deleted. Note this means reset=true no longer recreates the
    collection's underlying vector SCHEMA either (an
    earlier version did, via a full delete_collection, whenever no
    memory-bank points existed at check-time, but that check-then-act was
    itself racy against a concurrent write): changing EMBEDDING_MODEL on an
    existing collection now requires manually dropping it first, outside
    this tool, before re-running index_repo.

    When reset=False and the collection already contains data, this tool
    returns a warning before starting the expensive embedding work
    so you can cancel/reset/use sync_repo rather than discovering
    duplicate risk after the full embed-and-store loop has already run.
    Pass force=True to bypass this pre-flight check and add content to
    the existing index deliberately. force has no effect when reset=True.

    WARNING -- reset=false never detects or removes chunks for files that
    were DELETED from the repo since the last run: this tool
    only ever adds/updates chunks for whatever build_entries() finds right
    now, with no memory of what a previous call indexed. A deleted file's
    old chunks are never touched and remain in the collection indefinitely
    as stale, orphaned search results. Deliberately not fixed by having
    this tool detect and delete them itself -- sync_repo already exists
    specifically for that. But sync_repo's removal-detection only covers
    files recorded in its manifest -- a file
    already deleted before the manifest was first written is absent from
    both the manifest and the current scan, so a sync_repo has nothing to
    compare against and can't discover or delete it either. sync_repo only
    prevents FUTURE orphans once a baseline exists; reset=true here is
    what's actually required to clean up chunks ALREADY orphaned by an
    index_repo run that did not write one (see the last paragraph for when
    that is). If your repo's files can be deleted between runs, switch to
    sync_repo for ongoing re-indexing -- just don't expect it to also
    retroactively clean up whatever index_repo already orphaned.

    Chunking is by the embedding model's own token window, not by line count:
    the model reads only the first N tokens of any text it
    embeds (N is read from the loaded model, never assumed), so each chunk
    is cut to fit that window whatever the line structure, wherever possible at
    a markdown heading, a code boundary, a newline or whitespace. `overlap` is
    therefore a number of TOKENS repeated between consecutive chunks, and
    `chunk_lines` no longer sets any size: it is accepted so existing calls do
    not break and is ignored. Every file is indexed, with no per-file cap: a
    minified bundle or a data dump becomes as many chunks as it needs, which
    can be thousands, so progress names the file being chunked and the result
    names the files that produced the most chunks. If the model's window cannot
    be read the call fails before chunking or deleting anything. Chunk metadata
    carries file_path, line_number (the 1-based line where the chunk's text
    begins) and char_range ("start-end", character offsets into the decoded
    file); points indexed before this carry line_range instead until the
    collection is re-indexed.

    This can take a while on large repos since every chunk is embedded
    locally. Reports progress via MCP progress notifications as it runs (not
    all clients render these visibly, but it keeps the call from timing out
    on large repos). Returns a summary once indexing completes. For a big
    first-time index where you want to watch progress in a terminal, prefer
    running ingest_to_qdrant.py directly instead of this tool.

    When this run leaves the collection holding exactly what it indexed
    (reset=true, or a collection that was absent or empty before) it also
    writes sync_repo's manifest (.qdrant_index_manifest.json, the same
    versioned format, through the same writer), so the next sync_repo picks
    up from here instead of re-chunking and re-embedding every file. Each
    file's content hash is taken BEFORE chunking (a first step that reports a
    running count of files hashed), so a file edited while the
    run was going is simply re-indexed by that next sync; a file that failed
    to chunk is left out of the manifest and retried by it. Any existing
    manifest is removed before anything is stored (for reset=true, before the
    wipe, and the reset does not start if it cannot be removed), so a run that
    fails or is cancelled partway leaves no manifest claiming files are
    indexed. Use the same scope, include_extensions, exclude_dirs and
    respect_gitignore for later sync_repo calls: the manifest records the
    files this run's filters selected. A force=true run into a collection
    that already held points leaves the manifest untouched and says so: that
    run only adds chunks (duplicates, for files already indexed), so no
    manifest written from it could vouch for what is in the collection. Use
    reset=true when you want a baseline. The manifest is one file per
    checkout that does not record which collection it describes, so it is
    only written (or removed) when this run targets this project's own
    collection (the configured COLLECTION_NAME on the configured QDRANT_URL);
    indexing the checkout into any other collection leaves it alone and says
    so.

    A run into this project's own collection holds the checkout's manifest
    lock (.qdrant_index_manifest.json.lock) from before it prepares and counts
    the collection to the end, whether or not it ends up writing the
    manifest: whether it may write one depends on that count, which no other
    manifest user may change in between. If another index_repo, sync_repo or
    CLI --write-manifest run on the same checkout holds it, this returns an
    error at once, before anything is deleted or stored; run it again once
    that one has finished. A run into any other collection never takes the
    lock.
    """
    # _index_repo takes the lock (see ManifestLock in
    # libs/sync_manifest_lib.py) only for this project's own collection, the
    # one whose manifest it may write. Released here however the run ends --
    # returned, raised or cancelled.
    manifest_lock = ManifestLock(Path(repo_path).resolve())
    try:
        return await _index_repo(
            repo_path,
            collection=collection,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            embedding_model=embedding_model,
            scope=scope,
            include_extensions=include_extensions,
            exclude_dirs=exclude_dirs,
            respect_gitignore=respect_gitignore,
            chunk_lines=chunk_lines,
            overlap=overlap,
            reset=reset,
            force=force,
            ctx=ctx,
            manifest_lock=manifest_lock,
        )
    finally:
        manifest_lock.release()


async def _index_repo(
    repo_path: str,
    *,
    collection: Optional[str],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    embedding_model: Optional[str],
    scope: str,
    include_extensions: Optional[str],
    exclude_dirs: Optional[str],
    respect_gitignore: bool,
    chunk_lines: int,
    overlap: int,
    reset: bool,
    force: bool,
    ctx: Context,
    manifest_lock: ManifestLock,
) -> str:
    """index_repo's body (see its docstring), run with the manifest lock its
    caller releases."""
    # Checked first, before even the filesystem -- cheapest possible input to
    # validate, and the whole point is catching this before any real
    # work (chunking every file in the repo) happens on a bad value. The upper
    # bound on overlap (the model's window) is checked when the budget is built.
    chunk_err = validate_token_overlap(overlap)
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
    # Refuse outright, regardless of reset/force, if this would
    # index code chunks into (or, via reset, wipe) the shared memory-bank
    # collection -- e.g. a code repo accidentally configured with the same
    # COLLECTION_NAME as MEMORY_BANK_COLLECTION.
    if mb.is_memory_bank_collection_name(collection, DEFAULT_MEMORY_BANK_COLLECTION):
        return (
            f"Error: '{collection}' is configured as the shared memory-bank "
            f"collection (MEMORY_BANK_COLLECTION). Refusing to index code into it "
            f"or reset it via index_repo -- use a different COLLECTION_NAME for "
            f"this project's code index."
        )
    qdrant_url, qdrant_api_key = _resolve_credentials(qdrant_url, qdrant_api_key)
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    # Retry on a transient dropped connection -- see
    # libs/qdrant_retry.py. Every QdrantClient/QdrantConnector call in this
    # server is wrapped, not just the ones directly after a known CPU-bound
    # gap -- this is a long-running MCP server process, so any call can be
    # the first one after an arbitrarily long idle period since the
    # previous tool invocation.
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # How many points the collection held before this run added to it. Stays
    # 0 for a collection that does not exist yet; reset=True never
    # consults it (the wipe makes the run's own chunks the whole index). Only
    # a run that started from nothing may write the sync manifest at the end.
    points_before = {"count": 0}

    # The sync manifest is ONE file in the repo root: it records hashes and
    # nothing about which collection or Qdrant instance they were indexed into.
    # So index_repo only ever touches it for this project's own collection (the
    # same test the collection description uses further down); indexing the
    # checkout into some other collection must not replace the baseline of the
    # one sync_repo keeps up to date from the same checkout.
    owns_manifest = collection == DEFAULT_COLLECTION and qdrant_url == DEFAULT_QDRANT_URL
    if owns_manifest:
        # Taken BEFORE the collection is prepared and counted, not when the
        # manifest is first touched: whether this run may write a baseline
        # rests on the collection being absent or empty (points_before below),
        # and a sync_repo finishing in between could fill it and save its own
        # manifest, after which this run would append duplicates and record a
        # baseline that did not describe them. Holding the lock from here
        # means no other manifest user can change the collection between the
        # count and the save. A run into another collection never locks.
        busy = manifest_lock.acquire()
        if busy:
            return busy

    def _invalidate_manifest() -> Optional[str]:
        # An error string, in the shape prepare_collection_for_index aborts on
        # (the wording is shared with the CLI, libs/sync_manifest_lib.py).
        # _discard_manifest is looked up here, at call time, so it stays
        # patchable as this module's own name. The lock is already held (taken
        # above for every run into this project's own collection, the only
        # runs that get here); passing it keeps "never removed unlocked" true
        # whatever the call site.
        return invalidation_error(repo, _discard_manifest, manifest_lock)

    def _duplicate_risk_warning(existing_count: int) -> Optional[str]:
        # Checked BEFORE the expensive embedding work -- surface
        # the duplicate-risk warning up front so the user can cancel/reset/
        # use sync_repo instead of discovering it after the full
        # embed-and-store loop has already run.
        points_before["count"] = existing_count
        if existing_count == 0 and owns_manifest:
            # prepare_collection_for_index passes `points_count or 0`, and
            # Qdrant can report no count at all (None), which is not "empty".
            # An apparent zero is what lets a run replace the manifest, so it is
            # confirmed with an exact count first: a populated collection
            # misread as empty would lose the manifest that tracks its deleted
            # files. Only the manifest decision uses it: the warning below keeps
            # judging by the reported count, as before.
            points_before["count"] = call_with_retry(client.count, collection_name=collection, exact=True).count
        if existing_count == 0 or force:
            return None
        return (
            f"Warning: collection '{collection}' already contains "
            f"{existing_count} points. Re-running index_repo without "
            f"reset=True may add duplicate chunks on top of those. "
            f"Options: (1) use sync_repo instead for an incremental update "
            f"that avoids duplicates (preferred for most cases), "
            f"(2) re-run with reset=True to wipe unprotected collection "
            f"data and re-index cleanly (caution: wipes any other non-code "
            f"data stored there, e.g. qdrant-store notes or conversation-"
            f"compacts stored before the source marker existed -- memory-bank points and "
            f"newer conversation compacts are always preserved), "
            f"or (3) re-run with force=True to add content "
            f"to the existing index deliberately."
        )

    # Shared with tools/ingest_to_qdrant.py -- see
    # libs/qdrant_index_prep.py for the full reset (pre-delete model check,
    # then an always-filtered, memory-bank-preserving delete)
    # and non-reset (duplicate-risk guard above, then the
    # file_path payload-index backfill) sequence. This module's own
    # FastEmbedProvider/call_with_retry/check_embedding_model_mismatch/
    # ensure_file_path_index are passed in so they stay the patch point
    # this server's tests use. The provider is only built inside the reset
    # branch, so the guard's early return never pays for a model load; it's
    # reused (not reconstructed) below.
    #
    # The token budget is built in the same step as the provider,
    # so on reset=True an unreadable window fails BEFORE the filtered delete
    # prep runs, not after the collection has been emptied.
    budgets: dict = {}

    def _provider_with_budget(model_name: str):
        provider = FastEmbedProvider(model_name)
        budgets["budget"] = provider.build_token_budget(overlap)
        return provider

    try:
        prep = prepare_collection_for_index(
            client,
            collection,
            embedding_model,
            reset=reset,
            provider_factory=_provider_with_budget,
            existing_points_guard=_duplicate_risk_warning,
            retry=call_with_retry,
            model_check=check_embedding_model_mismatch,
            backfill=ensure_file_path_index,
            # After the model check, before the destructive delete:
            # a reset that cannot invalidate the manifest does not start.
            before_reset_delete=_invalidate_manifest if reset and owns_manifest else None,
        )
        if prep.error:
            return prep.error
        embedding_provider = prep.embedding_provider
        if embedding_provider is None:
            embedding_provider = _provider_with_budget(embedding_model)
    except TokenBudgetError as e:
        return f"Error: {e}"
    # Whether this run will leave the collection holding exactly what it
    # indexes, so that a manifest written from it is true: after a
    # reset, or into a collection that was absent or empty. A force=True run
    # into a populated one only adds chunks, so it writes none. Another
    # collection's run never touches the manifest either (see owns_manifest).
    writes_baseline = owns_manifest and (reset or points_before["count"] == 0)
    if writes_baseline:
        # From the first stored chunk the old manifest describes a collection
        # that no longer exists, so a run that fails or is cancelled partway
        # must not leave it behind: removing it HERE, not only replacing it at
        # the end, is what guarantees that. A reset on an existing collection
        # already did this before the wipe (the hook above), which is the
        # stricter point; this covers the runs with nothing to wipe, an absent
        # or empty collection, and costs nothing when it is already gone. The
        # old manifest is still true when preparation failed above, so this is
        # not reached then.
        manifest_error = _invalidate_manifest()
        if manifest_error:
            return manifest_error
    budget = budgets["budget"]
    connector = QdrantConnector(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection,
        embedding_provider=embedding_provider,
        field_indexes=FIELD_INDEXES,
    )

    # Collect chunks via iter_entries() with concurrent progress reporting:
    # the original build_entries() approach held every chunk in
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
    # Progress is reported as two interleaved streams: chunking reports a
    # running chunk count with no known total, and each flushed batch reports
    # stored/chunk_count-so-far (batches are embedded and stored while
    # chunking is still going, so that total keeps growing until chunking ends).
    #
    # A sentinel value (_DONE) signals that the producer has finished.
    # Errors from iter_entries() arrive as (None, (rel, err)) tuples (the
    # same sentinel contract as the generator itself) and are collected into
    # `skipped` without interrupting the drain.
    resolved_include_extensions = parse_csv_set(include_extensions) or DEFAULT_INCLUDE_EXTENSIONS
    resolved_exclude_dirs = parse_csv_set(exclude_dirs) or DEFAULT_EXTRA_EXCLUDE_DIRS

    current_hashes: dict = {}
    if writes_baseline:
        # BEFORE any file is chunked, never after (see _index_baseline): a file
        # edited between the two must end up with a hash older than its chunks,
        # so the next sync re-indexes it. Same scan parameters as sync_repo's
        # own hashing, and off the event loop for the same reason (a large
        # repo is thousands of files read one by one).
        current_hashes, _hash_skipped = await _hash_files_with_progress(
            ctx, repo, scope,
            include_extensions=resolved_include_extensions,
            extra_exclude_dirs=resolved_exclude_dirs,
            respect_gitignore=respect_gitignore,
            extra_exclude_files={MANIFEST_FILENAME},
        )

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
                budget=budget,
                include_extensions=resolved_include_extensions,
                extra_exclude_dirs=resolved_exclude_dirs,
                respect_gitignore=respect_gitignore,
                # The manifest file is a caller-owned bookkeeping file, not source --
                # always exclude it by name so it can never index itself regardless of
                # extension filter, scope, or gitignore state.
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
    # sized groups as they arrive -- the whole-repo accumulation
    # (holding every chunk before storage begins) is gone.
    pending_batch: list = []
    skipped: list = []
    chunk_count = 0
    code_count = 0
    doc_count = 0
    stored_count = 0
    # Chunks per file: no per-file cap exists, so this is how a
    # file that expanded into thousands of chunks is noticed, both in the
    # progress message while it is happening and in the final summary.
    file_chunk_counts: dict = {}
    # Cumulative base for within-batch stored_in_batch values: store_batch()
    # reports per-batch progress starting from 1, so to get an overall
    # cumulative count we add the prior batches' total here.
    _stored_offset = 0

    async def _emit_stored_progress(stored_in_batch: int, _total_in_batch: int) -> None:
        """Report embed/store progress as chunks flow through, not after all are done.

        `stored_in_batch` counts stored entries within the CURRENT store_batch()
        call (starts over on each new call). We accumulate across calls into the
        nonlocal stored_count so the progress value is monotonically increasing.
        chunk_count is only the number of chunks seen so far, not a final total:
        batches are flushed while chunking is still running (a flush happens
        once a full UPSERT_BATCH_SIZE batch has been collected), so it can still
        grow after this is reported.
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
                current_file = metadata_or_err.get("file_path", "?")
                file_chunk_counts[current_file] = file_chunk_counts.get(current_file, 0) + 1
                if ctx is not None:
                    await ctx.report_progress(
                        progress=chunk_count, total=None,
                        message=(
                            f"Chunked {chunk_count} chunk(s) so far "
                            f"(now in {current_file}: {file_chunk_counts[current_file]})..."
                        ),
                    )
                # Flush a full batch into storage as chunks arrive -- avoids
                # holding the whole repo in memory before storage begins
                # (the full-accumulation case this avoids).
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

    # The sync baseline: written through _save_manifest, the one
    # writer sync_repo uses, so it is always the current versioned format. After
    # the last chunk is stored (a run that failed earlier writes nothing, and
    # a reset run has already removed the old one) and before the fail-open
    # steps below, so nothing there can come between the index and its baseline.
    if not owns_manifest:
        manifest_note = (
            " The sync manifest was not written: it is one file per checkout and belongs to this project's own "
            "collection, which this run did not target."
        )
    elif not writes_baseline:
        manifest_note = (
            f" The sync manifest was not written: the collection already held {points_before['count']} point(s) "
            f"this run added to, so no manifest from it could say what is indexed. Run with reset=true to "
            f"index from a clean baseline."
        )
    else:
        missed = await asyncio.to_thread(_unindexed_hashed_files, repo, current_hashes, skipped, file_chunk_counts)
        baseline = _index_baseline(current_hashes, skipped, file_chunk_counts, missed)
        try:
            _save_manifest(repo, baseline)
            manifest_note = f" Wrote the sync manifest ({len(baseline)} file(s)) so sync_repo can pick up from here."
        except OSError as e:
            # The chunks are already stored and searchable, so a manifest that
            # cannot be written (read-only checkout) is reported, not raised:
            # raising would hide the summary of an index that succeeded. The
            # next sync_repo finds no manifest and rebuilds, as before.
            manifest_note = (
                f" Warning: could not write the sync manifest ({e}); the next sync_repo will rebuild the index."
            )

    # Only for THIS project's own default collection/instance -- an
    # explicit collection/qdrant_url override means the caller is indexing
    # something else, which must not get this project's own
    # COLLECTION_DESCRIPTION stamped onto it.
    if collection == DEFAULT_COLLECTION and qdrant_url == DEFAULT_QDRANT_URL:
        _sync_static_collection_description(collection, qdrant_url, qdrant_api_key)

    # Fail-open (never raises) -- see _recompute_search_limit.
    _recompute_search_limit(collection, qdrant_url, qdrant_api_key)

    largest = format_largest_files(file_chunk_counts)
    return (
        f"Indexed {total} chunks from {repo} into collection '{collection}' "
        f"({code_count} code chunks, {doc_count} doc chunks, each cut to the model's "
        f"{budget.max_tokens}-token window). "
        f"{'Collection was reset before indexing.' if reset else 'Existing entries were kept -- rerun with reset=true if you suspect duplicates.'}"
        f"{' ' + largest if largest else ''}"
        f"{manifest_note}"
        f"{_chunk_lines_note(chunk_lines)}"
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
    this repo, overlap changed and everything needs re-chunking
    consistently, or the index and manifest seem to have drifted (e.g. after
    manually editing points in Qdrant). If EMBEDDING_MODEL changed, reset=true
    can no longer rebuild the schema for you -- manually drop the
    collection first, outside this tool, then run index_repo fresh.

    Tracks a small manifest file (.qdrant_index_manifest.json) in the repo
    root to know what's already indexed -- safe to gitignore, don't hand-edit it.
    The manifest is versioned: one in the old flat format is
    ignored, which makes the next sync a REBUILD -- every file is re-chunked and
    re-embedded, which on a large repo is one long call. A rebuild first
    reports any points in the collection that are not indexer chunks (a
    metadata.type other than code or doc), because a point carrying the
    file_path of a re-indexed file is deleted with that file's old chunks; and
    afterwards deletes the chunks of files that no longer exist on disk, which
    an empty manifest cannot know about. A file still on disk but outside the
    rebuild's scope keeps its old chunks and is recorded in the manifest as a
    kept file, apart from the hashes: every later sync checks whether it has
    been deleted and removes its chunks if so, or re-indexes it once it falls
    inside the scope. A file whose existence cannot be checked is left alone.

    Chunking is by the embedding model's own token window, not by line count,
    exactly as described under index_repo: `overlap` is in tokens and
    `chunk_lines` is accepted but ignored. Every changed file is indexed with
    no per-file cap; progress names each file as it is chunked and the result
    names the files that produced the most chunks. Fails before chunking or
    deleting anything if the model's window cannot be read.

    collection/qdrant_url/embedding_model default to this project's
    COLLECTION_NAME/QDRANT_URL/EMBEDDING_MODEL env vars if not given. The
    configured QDRANT_API_KEY is only sent to the configured QDRANT_URL; a
    different qdrant_url needs its own qdrant_api_key if it requires one.

    Reports progress via MCP progress notifications in each phase: a running
    count while the repo's files are hashed (no total, the scan is lazy), the
    file being chunked, "i/N" over the changed files, and chunks embedded and
    stored. The short Qdrant-side steps between them (consistency checks, the
    batched delete of replaced chunks, orphan cleanup on a rebuild) send none.

    Holds the checkout's manifest lock (.qdrant_index_manifest.json.lock, safe
    to gitignore too) from reading the manifest to saving it, so two
    overlapping syncs on one checkout -- two sessions each syncing at start,
    say -- cannot leave a manifest that disagrees with the collection. If
    another sync_repo, an index_repo into this project's own collection or a
    CLI --write-manifest run on the same checkout holds it, this returns an
    error at once, before
    anything is read, deleted or stored; run it again once that one has
    finished.
    """
    # Released here however the run ends -- returned, raised or cancelled;
    # _sync_repo takes it right before it reads the manifest.
    manifest_lock = ManifestLock(Path(repo_path).resolve())
    try:
        return await _sync_repo(
            repo_path,
            collection=collection,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            embedding_model=embedding_model,
            scope=scope,
            include_extensions=include_extensions,
            exclude_dirs=exclude_dirs,
            respect_gitignore=respect_gitignore,
            chunk_lines=chunk_lines,
            overlap=overlap,
            ctx=ctx,
            manifest_lock=manifest_lock,
        )
    finally:
        manifest_lock.release()


async def _sync_repo(
    repo_path: str,
    *,
    collection: Optional[str],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    embedding_model: Optional[str],
    scope: str,
    include_extensions: Optional[str],
    exclude_dirs: Optional[str],
    respect_gitignore: bool,
    chunk_lines: int,
    overlap: int,
    ctx: Context,
    manifest_lock: ManifestLock,
) -> str:
    """sync_repo's body (see its docstring), run with the manifest lock its
    caller releases."""
    # See index_repo's identical check above.
    chunk_err = validate_token_overlap(overlap)
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
    # Same reserved-collection-name refusal as index_repo -- see
    # its comment above for why.
    if mb.is_memory_bank_collection_name(collection, DEFAULT_MEMORY_BANK_COLLECTION):
        return (
            f"Error: '{collection}' is configured as the shared memory-bank "
            f"collection (MEMORY_BANK_COLLECTION). Refusing to sync code into it -- "
            f"use a different COLLECTION_NAME for this project's code index."
        )
    qdrant_url, qdrant_api_key = _resolve_credentials(qdrant_url, qdrant_api_key)
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    resolved_include_extensions = parse_csv_set(include_extensions) or DEFAULT_INCLUDE_EXTENSIONS
    resolved_exclude_dirs = parse_csv_set(exclude_dirs) or DEFAULT_EXTRA_EXCLUDE_DIRS

    # From here to the final _save_manifest this run owns the manifest: a
    # second run reading it in between could save over this one's result with
    # hashes for chunks it then replaced (see libs/sync_manifest_lib.py's
    # module docstring for the exact sequence), so it is refused instead.
    busy = manifest_lock.acquire()
    if busy:
        return busy
    preserved_before: set = set()
    manifest = _load_manifest(repo, preserved_before)
    # Run off the event loop -- this is a synchronous scan that can take
    # several seconds on a large repo (thousands of files hashed one by
    # one), and stalling the event loop for that long is the actual root
    # cause behind the Qdrant connection drop libs/qdrant_retry.py handles:
    # Docker Desktop's vpnkit reaps an idle pooled connection during exactly
    # this kind of CPU-bound gap. asyncio.to_thread keeps the loop free
    # (and any pooled connection's keep-alive traffic flowing) while this
    # runs, instead of just retrying after the fact.
    # Reports a running count as it goes; see
    # _hash_files_with_progress.
    current_hashes, hash_skipped = await _hash_files_with_progress(
        ctx, repo, scope,
        include_extensions=resolved_include_extensions,
        extra_exclude_dirs=resolved_exclude_dirs,
        respect_gitignore=respect_gitignore,
        # Same as index_repo -- the manifest must never hash itself into the
        # change-detection baseline.
        extra_exclude_files={MANIFEST_FILENAME},
    )

    changed = [f for f, h in current_hashes.items() if manifest.get(f) != h]
    # Excludes hash_skipped's paths -- a file that failed to hash is neither
    # "removed" nor "changed," it's unknown, and must not be treated as
    # removed (which would delete its existing good embeddings over what may
    # be a transient read error). See files_removed_since_last_sync.
    removed = files_removed_since_last_sync(manifest, current_hashes, [rel for rel, _ in hash_skipped])
    unchanged_count = len(current_hashes) - len(changed)

    # Files an earlier rebuild kept but did not scan, read with the
    # hash map above. With no usable manifest this sync is a rebuild, which
    # works the set out again from the collection. Anything that is also a
    # manifest key is tracked normally and is not "kept". A kept file that is
    # in this scan needs nothing here: it is not in the manifest, so it counts
    # as changed and is re-indexed, which moves it to normal tracking when the
    # manifest is saved below. A kept file the scan did not yield is checked on
    # disk with the same lstat-based test the rebuild sweep uses, NOT by
    # absence from the scan (a narrower scope must never delete a file that is
    # still there); only a confirmed-gone one has its chunks deleted below.
    hash_skipped_paths = {rel for rel, _ in hash_skipped}
    preserved_before = preserved_before - set(manifest)
    preserved_gone = await asyncio.to_thread(
        _preserved_paths_gone,
        repo,
        [rel for rel in preserved_before if rel not in current_hashes and rel not in hash_skipped_paths],
    )

    # Constructed before the "no changes" early return below so
    # the index-consistency check can run ahead of it.
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    async def _check_index_invariant(will_be_indexed):
        """ONE invariant: "every manifest file that
        would produce chunks is either (a) present in the Qdrant collection
        or (b) proven to be (re)indexed during THIS sync". Returns
        (error_or_None, collection_present). Run at EVERY collection-existence
        check in sync_repo (the early one and the one before the delete/store
        phase), and the caller must use the returned collection_present
        rather than asking Qdrant again, so there is no gap between check and
        use.

        The manifest is a local file that knows nothing about Qdrant. The
        documented EMBEDDING_MODEL remedy (drop the collection by hand)
        leaves it claiming everything is indexed, so without this the "no
        changes" return, or a changed-files-only re-index into a brand new
        collection, silently masks the mismatch.

        `will_be_indexed` is the set of paths proven indexed by this sync:
        the early call passes every changed path (nothing has been chunked
        yet); the later call passes only files whose chunking
        completed without raising (a clean zero-chunk result counts: it
        proves there is nothing to index), so a changed file whose chunking
        failed is NOT exempt. Presence is verified at collection granularity (the
        collection holds at least one file chunk), NOT per file: a per-file
        check would cost a Qdrant query per manifest file on every sync,
        defeating the incremental design, and no check made before
        store_batch can be atomic with it anyway -- a collection dropped
        mid-write by a concurrent actor is outside what this guards. It
        targets the documented drop-then-sync scenario. Everything else in
        the manifest is checked, failing closed
        on any unknown state: a missing collection, an empty one (no point
        with metadata.type in {code, doc} -- preserved memory-bank /
        qdrant-store points, which can also carry metadata.file_path,
        must not mask that), a not-found raised mid-check by a
        concurrent drop, or a manifest file that is unreadable (hash-skipped
        or fails to chunk). The only files ignored are ones proven to be
        nothing to index: removed from disk, or chunking cleanly to zero
        chunks (empty/whitespace-only), which never create a point.

        An empty manifest is a legitimate first sync: never an error.
        """
        collection_present = call_with_retry(client.collection_exists, collection)
        if not manifest:
            return None, collection_present
        indexed_chunk_count = 0
        if collection_present:
            try:
                indexed_chunk_count = call_with_retry(
                    client.count,
                    collection_name=collection,
                    count_filter=models.Filter(
                        must=[models.FieldCondition(key="metadata.type", match=models.MatchAny(any=["code", "doc"]))]
                    ),
                    exact=True,
                ).count
            except Exception:
                # exists and count are separate requests: a concurrent drop
                # between them surfaces as not-found from count. Treat it as
                # missing only if a follow-up check confirms; re-raise
                # anything else (a real failure must not be swallowed).
                if call_with_retry(client.collection_exists, collection):
                    raise
                collection_present = False
        if indexed_chunk_count > 0:
            return None, collection_present

        proven_indexed = set(will_be_indexed)
        hash_skipped_paths = {rel for rel, _ in hash_skipped}

        def _some_manifest_file_should_be_indexed():
            for rel in manifest:
                if rel in hash_skipped_paths:
                    return True  # unreadable = unknown, fail closed
                if rel not in current_hashes or rel in proven_indexed:
                    continue
                try:
                    # Whether the file would produce ANY chunk, not what the
                    # chunks are, so no token budget (and no model load) is
                    # needed; the "no changes" path never loads the model.
                    if file_has_indexable_content(repo / rel):
                        return True
                except OSError:
                    return True  # unreadable = unknown, fail closed
            return False

        if await asyncio.to_thread(_some_manifest_file_should_be_indexed):
            state = "does not exist" if not collection_present else "has no indexed file chunks"
            return (
                f"Error: the sync manifest says {len(manifest)} files are already indexed, but collection "
                f"'{collection}' {state} in Qdrant, so an incremental sync would silently leave those files "
                f"unindexed. Run index_repo(reset=True) to rebuild it from scratch (which also rewrites the manifest).",
                collection_present,
            )
        return None, collection_present

    if manifest:
        early_error, _ = await _check_index_invariant(changed)
        if early_error:
            return early_error

    if not changed and not removed and not preserved_gone:
        # The one "no changes" case that still has work to do: a
        # REBUILD (no usable manifest, e.g. an old-format one) of a repo that
        # now has nothing to index -- every file the old manifest tracked is
        # gone. With no changed and no removed file the code below never runs,
        # so the deleted files' chunks would stay searchable on every later
        # sync. The normal "no changes" path stays model-free; this rare one
        # builds a provider for the same two pre-delete checks the full path
        # below makes: the token window must be readable (nothing here is
        # chunked, but the contract is that an unreadable window fails the run
        # before anything is deleted or written, whatever the run would have
        # done), then the vector schema must match.
        rebuild_notes = ""
        if not manifest and call_with_retry(client.collection_exists, collection):
            sweep_provider = FastEmbedProvider(embedding_model)
            try:
                sweep_provider.build_token_budget(overlap)
            except TokenBudgetError as e:
                return f"Error: {e}"
            mismatch = check_embedding_model_mismatch(client, collection, sweep_provider, fail_closed=True)
            if mismatch:
                return mismatch
            # Hash-skipped files are tracked below (UNVERIFIED_HASH), so the scan
            # for this purpose is everything the hash phase visited.
            scanned_paths = set(current_hashes) | hash_skipped_paths
            sweep_preserved: set = set()
            rebuild_notes = _orphans_note(
                await asyncio.to_thread(
                    _delete_orphaned_chunks, client, collection, repo, scanned_paths, sweep_preserved
                )
            )
            rebuild_notes += _preserved_note(sweep_preserved)
            # Replace an old-format manifest with a versioned one, or `upgrade`
            # keeps offering to delete it and every sync repeats this sweep. A
            # file skipped as unreadable keeps a retryable entry (see
            # UNVERIFIED_HASH), since its old chunks were left alone.
            for rel, _ in hash_skipped:
                current_hashes[rel] = UNVERIFIED_HASH
            _save_manifest(repo, current_hashes, sweep_preserved)
        return (
            f"No changes since last sync. {unchanged_count} files already up to date in '{collection}'."
            f"{rebuild_notes}"
            f"{_skipped_summary(hash_skipped)}"
        )

    # The embedding provider -- and with it the token budget -- is built BEFORE
    # anything is chunked or deleted: an unreadable window must
    # fail the sync here, not after chunks were cut with a guessed size. It was
    # constructed after chunking before; chunking now needs it. The "no
    # changes" early return above stays model-free.
    embedding_provider = FastEmbedProvider(embedding_model)
    try:
        budget = embedding_provider.build_token_budget(overlap)
    except TokenBudgetError as e:
        return f"Error: {e}"

    def _chunk_one_file(rel: str) -> list:
        return [
            Entry(content=content, metadata=metadata)
            for content, metadata in chunk_file(repo / rel, rel, chunk_lines, overlap, budget)
        ]

    async def _chunk_changed_files():
        entries = []
        skipped = []
        file_chunk_counts: dict = {}
        # One file at a time, off the event loop, so progress can be reported
        # between files. Every file is indexed with no cap on its chunks (an owner
        # decision), so a minified bundle can run for a long time;
        # the message before each file names it, and the one after gives its
        # chunk count, which is how that run gets noticed.
        for position, rel in enumerate(changed, start=1):
            if ctx is not None:
                await ctx.report_progress(
                    progress=position - 1, total=len(changed),
                    message=f"Chunking {rel} ({position}/{len(changed)}, {len(entries)} chunks so far)",
                )
            try:
                file_entries = await asyncio.to_thread(_chunk_one_file, rel)
            except OSError as e:
                skipped.append((rel, str(e)))
                continue
            entries.extend(file_entries)
            file_chunk_counts[rel] = len(file_entries)
            if ctx is not None:
                await ctx.report_progress(
                    progress=position, total=len(changed),
                    message=f"Chunked {rel}: {len(file_entries)} chunk(s) ({len(entries)} so far)",
                )
        return entries, skipped, file_chunk_counts

    # Chunk BEFORE issuing any deletes -- doing this
    # the other way around, delete-then-chunk, meant a file that hashed fine
    # but then failed to chunk (the same hash/chunk race compute_file_hashes
    # is exposed to) had its existing, still-good embeddings deleted with
    # nothing to replace them -- leaving it with ZERO searchable content
    # until a later successful sync, unlike the preservation behavior
    # hash-skipped files already got. Chunking first means we know exactly
    # which changed files actually produced replacement content before any
    # delete call ever goes out, so a chunk failure now behaves the same as
    # a hash failure: leave the existing embeddings alone. Each file is chunked
    # off the event loop -- same reasoning as compute_file_hashes above:
    # chunking a file (file I/O + boundary detection + tokenizing) is a
    # synchronous, potentially multi-second scan.
    entries, chunk_skipped, file_chunk_counts = await _chunk_changed_files()
    new_chunk_count = len(entries)
    chunk_skipped_paths = {rel for rel, _ in chunk_skipped}
    # Only files that actually produced replacement chunks get re-indexed --
    # chunk_skipped files stay OUT of the delete set (see above) and out of
    # this count, so the returned summary never claims a file was both
    # "re-indexed" and "skipped".
    successfully_changed = [f for f in changed if f not in chunk_skipped_paths]

    # embedding_provider was constructed before chunking (see above), so the
    # schema check below can validate BEFORE the delete loop runs --
    # same reasoning as index_repo's own pre-delete check, and the exact
    # same "validate/chunk before deleting" principle the review
    # already established for this function's chunk-before-delete ordering.

    # Qdrant calls here go through call_with_retry (transient dropped
    # connections -- see libs/qdrant_retry.py for the attempt
    # count).
    # Re-run the same invariant now that chunking is done (the collection can
    # be dropped in the seconds since the early check, and store_batch would
    # recreate it holding only the changed files), with the exemption narrowed
    # to files whose chunking completed without raising. Its collection_present
    # answer drives the branch below -- no separate existence request.
    late_error, collection_present = await _check_index_invariant(successfully_changed)
    if late_error:
        return late_error
    # A REBUILD: no usable manifest (none yet, or the old format),
    # against a collection that already exists. Every file counts as changed,
    # and the manifest cannot say which indexed files are gone, so this sync
    # also cleans those up afterwards. Normal syncs already handle removals
    # through the manifest and never take this path.
    rebuild = not manifest and bool(collection_present)
    rebuild_notes = ""
    if collection_present:
        # Found in review: without this, a changed
        # EMBEDDING_MODEL would still delete each changed/removed file's old
        # chunks below and only discover the mismatch later, once the
        # embed/store loop's upsert failed against the incompatible schema --
        # leaving those specific files with NO searchable content at all
        # (deleted, never replaced), unlike every other skip path in this
        # function, which deliberately preserves existing embeddings on
        # failure. fail_closed=True (not the default fail-open) for the same
        # reason as index_repo's own pre-delete check: this is about to run a
        # destructive delete, so an inconclusive result must block it, not
        # silently permit it.
        mismatch = check_embedding_model_mismatch(client, collection, embedding_provider, fail_closed=True)
        if mismatch:
            return mismatch
        if rebuild:
            # After the model check (nothing is deleted on a mismatch) and
            # before the per-file deletes below, which are what the warning is
            # about. Reported, never blocking.
            non_chunk_warning = await asyncio.to_thread(_non_chunk_points_warning, client, collection)
            if non_chunk_warning:
                await _announce_warning(ctx, non_chunk_warning.strip())
                rebuild_notes += non_chunk_warning
        # Backfill the metadata.file_path payload index before the
        # delete-by-filter loop below -- this is the exact hotspot the
        # payload index exists to fix: without an index on this field, each of the
        # filtered client.delete() calls below full-scans every point in
        # the collection instead of using the index. FIELD_INDEXES (passed
        # to QdrantConnector further down) only applies to a brand new
        # collection, which this already-existing one isn't -- see
        # ensure_file_path_index()'s docstring.
        call_with_retry(ensure_file_path_index, client, collection)

        # A kept file confirmed gone is deleted exactly as a tracked removed
        # file is: same filter, same memory-bank/compact exclusion.
        to_delete = successfully_changed + removed + preserved_gone
        # Batched into DELETE_BATCH_SIZE-sized OR filters instead of one
        # client.delete() per file -- see DELETE_BATCH_SIZE's comment above
        # for why. `should` (OR, not `must`/AND) matches a point if ANY of
        # the batch's file_paths match, deleting every changed/removed
        # file's points in each chunked call.
        for batch in batched(to_delete, DELETE_BATCH_SIZE):
            # Retry on a transient dropped connection -- see
            # libs/qdrant_retry.py.
            call_with_retry(
                client.delete,
                collection_name=collection,
                points_selector=models.Filter(
                    should=[
                        models.FieldCondition(key="metadata.file_path", match=models.MatchValue(value=rel))
                        for rel in batch
                    ],
                    # Defense-in-depth: today's schema never sets
                    # metadata.file_path on a memory-bank point, so this isn't
                    # fixing a live bug -- just closing the gap structurally in
                    # case a future schema change reintroduces overlap.
                    # Same shared fragment as the reset path, so
                    # conversation compacts are protected here too.
                    must_not=mb.memory_bank_exclusion_filter().must_not,
                ),
            )

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

    # The kept-but-untracked set to record. A rebuild works it out
    # from the sweep below. Otherwise it is last time's set minus what is gone
    # and what moved to normal tracking (in this scan and re-indexed); a kept
    # file whose re-index failed to chunk stays kept, since its old chunks were
    # not replaced and it is about to lose its hash entry again.
    preserved_after: set = set()
    if rebuild:
        # After the new chunks are stored, so a failure above
        # never leaves a collection both emptied of orphans and missing chunks.
        scanned_paths = set(current_hashes) | hash_skipped_paths
        rebuild_notes += _orphans_note(
            await asyncio.to_thread(_delete_orphaned_chunks, client, collection, repo, scanned_paths, preserved_after)
        )
        rebuild_notes += _preserved_note(preserved_after)
    else:
        gone_set = set(preserved_gone)
        preserved_after = {
            rel
            for rel in preserved_before
            if rel not in gone_set and (rel not in current_hashes or rel in chunk_skipped_paths)
        }
        rebuild_notes += _preserved_gone_note(preserved_gone)

    # Only for THIS project's own default collection/instance -- see the
    # matching guard in index_repo above for why.
    if collection == DEFAULT_COLLECTION and qdrant_url == DEFAULT_QDRANT_URL:
        _sync_static_collection_description(collection, qdrant_url, qdrant_api_key)

    # A file that failed to hash OR failed to chunk this round must not lose
    # its manifest entry outright: _save_manifest
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
        elif rebuild:
            # No prior entry because the old manifest was ignored,
            # not because the file was never indexed: its old chunks may exist
            # and were preserved. Keep it tracked and retryable.
            current_hashes[rel] = UNVERIFIED_HASH
        else:
            current_hashes.pop(rel, None)

    all_skipped = hash_skipped + chunk_skipped
    _save_manifest(repo, current_hashes, preserved_after)

    # After the manifest save, so a (fail-open) recompute can never precede or
    # interfere with it. Not reached on the "no changes" early return.
    _recompute_search_limit(collection, qdrant_url, qdrant_api_key)

    largest = format_largest_files(file_chunk_counts)
    return (
        f"Synced '{collection}': {len(successfully_changed)} file(s) re-indexed ({new_chunk_count} chunks, each cut "
        f"to the model's {budget.max_tokens}-token window), "
        f"{len(removed)} file(s) removed, {unchanged_count} file(s) unchanged and skipped."
        f"{' ' + largest if largest else ''}"
        f"{rebuild_notes}"
        f"{_chunk_lines_note(chunk_lines)}"
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
    embedding_model: Optional[str] = None,
) -> str:
    """
    Preview how a repo would be chunked WITHOUT writing anything to Qdrant.
    Use this before index_repo on a repo you haven't indexed before, to
    sanity-check chunk boundaries and metadata (especially file_path,
    line_number and char_range) -- and to sanity-check
    include_extensions/exclude_dirs are actually filtering the way you expect
    -- before committing to a real index run.

    Chunking is by the embedding model's token window, so this
    loads the embedding model (embedding_model, default this project's
    EMBEDDING_MODEL) to read that window, and fails with an explicit error if
    it cannot. overlap is in tokens; chunk_lines is accepted but ignored (see
    index_repo). The result names the files that produced the most chunks, to
    show how large a minified or generated file would become before indexing it.
    """
    # See index_repo's identical check above.
    chunk_err = validate_token_overlap(overlap)
    if chunk_err:
        return f"Error: {chunk_err}"

    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        return f"Error: {repo} is not a directory."

    try:
        budget = FastEmbedProvider(embedding_model or DEFAULT_EMBEDDING_MODEL).build_token_budget(overlap)
    except TokenBudgetError as e:
        return f"Error: {e}"

    file_chunk_counts: dict = {}
    entries, skipped = build_entries(
        repo, scope, chunk_lines, overlap,
        budget=budget,
        on_file=file_chunk_counts.__setitem__,
        include_extensions=parse_csv_set(include_extensions) or DEFAULT_INCLUDE_EXTENSIONS,
        extra_exclude_dirs=parse_csv_set(exclude_dirs) or DEFAULT_EXTRA_EXCLUDE_DIRS,
        respect_gitignore=respect_gitignore,
        # Same as index_repo -- the manifest must never appear in preview
        # results.
        extra_exclude_files={MANIFEST_FILENAME},
    )
    lines = [
        f"{len(entries)} total chunks from {len(file_chunk_counts)} file(s) would be created, each cut to the "
        f"model's {budget.max_tokens}-token window. Showing first {min(limit, len(entries))}:\n"
    ]
    for content, metadata in entries[:limit]:
        preview = content[:200].replace("\n", " ")
        # chunk_position reads either metadata shape, never a bare key.
        lines.append(f"[{metadata['type']}] {metadata['file_path']} ({chunk_position(metadata)}): {preview}...")
    largest = format_largest_files(file_chunk_counts)
    if largest:
        lines.append(f"\n{largest}")
    return "\n".join(lines) + _chunk_lines_note(chunk_lines) + _skipped_summary(skipped)


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
    env vars if not given. The configured QDRANT_API_KEY is only sent to the
    configured QDRANT_URL; a different qdrant_url needs its own qdrant_api_key
    if it requires one.
    """
    collection = collection or DEFAULT_COLLECTION
    if not collection:
        return "Error: no collection specified and no COLLECTION_NAME env var configured for this project."
    qdrant_url, qdrant_api_key = _resolve_credentials(qdrant_url, qdrant_api_key)

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    # Retry on a transient dropped connection -- see
    # libs/qdrant_retry.py.
    if not call_with_retry(client.collection_exists, collection):
        return f"Collection '{collection}' does not exist yet."
    info = call_with_retry(client.get_collection, collection)
    return f"Collection '{collection}': {info.points_count} points, status={info.status}."


@mcp.tool()
async def find_in_collection(
    query: str,
    collection: str,
    limit: Optional[int] = None,
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
    repo's collection lives on a different Qdrant instance. The configured
    QDRANT_API_KEY is only sent to the configured QDRANT_URL, so a different
    qdrant_url needs its own qdrant_api_key if it requires one. embedding_model
    defaults to this project's EMBEDDING_MODEL env var and MUST match
    whatever model the OTHER collection was actually indexed with -- check
    the other repo's own .mcp.json if unsure. A mismatch is detected before
    querying and returns an actionable error rather than silently returning
    irrelevant results; the check fails open on network errors so a transient
    Qdrant hiccup never blocks a valid query.

    limit is how many results to return. Leave it unset to use the
    collection's own stored search_limit (scaled to its size by
    index_repo/sync_repo), falling back to 10 if none is stored;
    an explicit limit always wins outright.

    Each hit's metadata locates the chunk with file_path, line_number and
    char_range, or with file_path and line_range ("start-end") when that
    collection has not been re-indexed since chunks were cut by token window
    -- such a collection can miss code a re-indexed one finds.
    """
    if not collection:
        return "Error: collection is required -- call list_collections to see what's available."
    if limit is not None and limit < 1:
        return f"Error: limit must be 1 or greater (got {limit})."

    qdrant_url, qdrant_api_key = _resolve_credentials(qdrant_url, qdrant_api_key)
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    # Retry on a transient dropped connection -- see
    # libs/qdrant_retry.py.
    if not call_with_retry(client.collection_exists, collection):
        return (
            f"Error: collection '{collection}' does not exist at {qdrant_url}. "
            "Call list_collections to see what's available."
        )

    # conversation-compact collections hold saved session summaries
    # read back only through compact_find, never by semantic search. Answer
    # exactly as a genuinely empty search would -- an error or special message
    # would prompt the calling model to react and go off track. Checked BEFORE
    # the model-mismatch guard below, whose misleading error (compacts use a
    # different embedding model than the project default) was the only thing
    # keeping callers out of these collections until now. Fails open.
    if collection_is_marked_compact(client, collection):
        return f"No results found in collection '{collection}' for query '{query}'."

    embedding_provider = FastEmbedProvider(embedding_model)

    # Guard against silently-meaningless results from an embedding model
    # mismatch -- different models produce incompatible vector spaces, so a
    # wrong model returns plausible-looking garbage with no error.
    # Checked AFTER constructing the provider (needed for get_vector_name/size)
    # and BEFORE creating the connector or running the query. Fails open: any
    # exception inside the check is swallowed and the query proceeds normally.
    mismatch_error = check_embedding_model_mismatch(client, collection, embedding_provider)
    if mismatch_error:
        return mismatch_error

    if limit is None:
        limit = _resolve_search_limit(client, collection, qdrant_url)

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
    user always having to name it explicitly. Works on ANY
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
    different Qdrant instance. The configured QDRANT_API_KEY is only sent to
    the configured QDRANT_URL; a different qdrant_url needs its own
    qdrant_api_key if it requires one.
    """
    if not collection:
        return "Error: collection is required -- call list_collections to see what's available."
    qdrant_url, qdrant_api_key = _resolve_credentials(qdrant_url, qdrant_api_key)

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    # Retry on a transient dropped connection -- see
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

    include_counts (default true) additionally shows each collection's
    point count. Every collection's metadata is fetched (one Qdrant request
    per collection) regardless, to skip internal conversation-compact
    collections, which are never search targets and are not
    listed.

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
    different Qdrant instance. The configured QDRANT_API_KEY is only sent to
    the configured QDRANT_URL; a different qdrant_url needs its own
    qdrant_api_key if it requires one.
    """
    qdrant_url, qdrant_api_key = _resolve_credentials(qdrant_url, qdrant_api_key)
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    # Retry on a transient dropped connection -- see
    # libs/qdrant_retry.py.
    names = sorted(c.name for c in call_with_retry(client.get_collections).collections)
    if not names:
        return f"No collections found at {qdrant_url}."

    # Cache holds only genuine hits (see get_cached_descriptions's own
    # docstring) -- a name absent here still needs a live fetch below.
    descriptions = get_cached_descriptions(qdrant_url, names) if include_descriptions else {}

    lines = [f"Collections at {qdrant_url}:"]
    listed = 0
    for name in names:
        marker = " (this project's own)" if name == DEFAULT_COLLECTION else ""
        needs_description_fetch = include_descriptions and name not in descriptions
        # Every collection's metadata is read so conversation-
        # compact collections (marked by compact_store) can be skipped
        # entirely -- they're internal, not search targets. This is the same
        # request include_counts/the description fetch already made, so it
        # only adds requests when neither of those needed one.
        info = call_with_retry(client.get_collection, name)
        if is_marked_compact(info.config.metadata):
            continue
        listed += 1
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
    if not listed:
        return f"No collections found at {qdrant_url}."
    return "\n".join(lines)


if __name__ == "__main__":
    # Covers the common case: the collection already exists from a
    # previous index. index_repo/sync_repo's own call sites (above) cover
    # a brand-new collection that doesn't exist yet at this point.
    _sync_static_collection_description(DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, DEFAULT_QDRANT_API_KEY)
    _log_registered_tool_count()
    mcp.run()
