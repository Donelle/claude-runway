"""
The sync manifest's read, write, discard and baseline logic, shared by
mcpservers/ingest_mcp_server.py (sync_repo and index_repo) and tools/ingest_to_qdrant.py
(the CLI's opt-in --write-manifest), so the two cannot drift apart on a file
whose format every project's next sync_repo depends on.

Moved here unchanged from the server module, which imports MCP and therefore
cannot be imported by the CLI. The manifest format and
qdrant_ingest_lib.MANIFEST_FORMAT_VERSION are NOT touched: a version bump forces
a full re-embed on every project that has already been indexed.

Dependency-light on purpose (standard library plus qdrant_ingest_lib, which is
itself dependency-free): no Qdrant client, no MCP, no embedding model.
"""

import json
from pathlib import Path
from typing import Callable, Optional

from qdrant_ingest_lib import (
    MANIFEST_FILENAME,
    file_has_indexable_content,
    manifest_to_json,
    parse_manifest,
    parse_manifest_preserved,
)

# Manifest value recorded for a file that could not be hashed or chunked during
# a rebuild and has no prior manifest entry. Not a hash: it never
# equals one, so the file counts as changed on the next sync and is retried.
# Without an entry the file would vanish from the new, non-empty manifest: its
# preserved old chunks would then never be removed if the file is deleted
# before a retry succeeds, because later syncs are no longer rebuilds and only
# files the manifest knows about are ever treated as removed.
UNVERIFIED_HASH = "unverified"


def load_manifest(repo: Path, preserved_out: Optional[set] = None) -> dict:
    """
    The {file: content hash} map from the repo's sync manifest, or {} when
    there is no usable manifest -- in which case every file counts as changed
    and the whole index is rebuilt.

    When `preserved_out` is given, the paths a rebuild kept but did not scan
    are added to it, from the SAME read: another process saving the
    manifest can leave it truncated for a moment, and two reads could then
    return a good hash map and an empty kept set, which a later save would
    write back without the watched paths. One read makes a bad read lose both,
    and no usable manifest already means a rebuild, which works the set out
    again. The set is left empty whenever the map is.

    "Usable" means the versioned format (see
    qdrant_ingest_lib.MANIFEST_FORMAT_VERSION). A manifest in the old flat
    {file: hash} shape is ignored on purpose: it records file content but not
    how files were chunked, so trusting it after the chunker changed would make
    sync_repo report "No changes" and keep serving the old chunks indefinitely.
    """
    manifest_path = repo / MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            # Explicit encoding on both halves so a manifest written on one
            # platform reads back on another. UnicodeDecodeError is caught
            # alongside the others because it is NOT a json.JSONDecodeError --
            # uncaught, it would propagate out of a sync and fail the tool call
            # rather than degrading to "no manifest, treat everything as new."
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return {}
        files = parse_manifest(data)
        if not files:
            return {}
        if preserved_out is not None:
            # NOT in the {file: hash} map, so files_removed_since_last_sync
            # never sees them.
            preserved_out.update(parse_manifest_preserved(data))
        return files
    return {}


def save_manifest(repo: Path, manifest: dict, preserved=()) -> None:
    """
    Write the versioned manifest: the {file: hash} map plus, when there are any,
    the preserved-but-untracked paths. index_repo's baseline (and
    the CLI's) passes none: a run that starts from a clean slate has nothing
    preserved.
    """
    (repo / MANIFEST_FILENAME).write_text(manifest_to_json(manifest, preserved), encoding="utf-8")


def discard_manifest(repo: Path) -> None:
    """
    Remove the repo's sync manifest, if any. index_repo (and the
    CLI with --write-manifest) calls it before it stores anything into a
    collection it will write a baseline for (and, on reset, before the wipe):
    from the moment the collection changes the old manifest no longer describes
    it, and if the run then fails or is cancelled halfway, nothing would replace
    it, so the next sync_repo would trust it and report "No changes" over a
    partly filled collection. With no manifest that sync is a rebuild instead.

    Raises OSError when an existing manifest cannot be removed. That is NOT
    fail-open: carrying on would leave exactly the stale manifest this exists
    to prevent, so the caller refuses to start (nothing deleted or stored yet).
    """
    (repo / MANIFEST_FILENAME).unlink(missing_ok=True)


def invalidation_error(repo: Path, discard: Callable[[Path], None] = discard_manifest) -> Optional[str]:
    """
    Remove the old manifest and return None, or return the error message to
    abort a run with when it cannot be removed. The one wording index_repo and
    the CLI both stop on, in the shape prepare_collection_for_index's
    before_reset_delete hook expects. `discard` is injectable so each caller
    passes its own module-level name, which is the patch point its tests use.
    """
    try:
        discard(repo)
    except OSError as e:
        return (
            f"Error: could not remove the old sync manifest {MANIFEST_FILENAME} ({e}). Nothing was deleted or "
            f"indexed: a manifest left behind would claim files are indexed that this run is about to replace, "
            f"and sync_repo would trust it if the run failed partway."
        )
    return None


def unindexed_hashed_files(
    repo: Path,
    current_hashes: dict,
    chunk_skipped: list,
    chunked_files,
    has_indexable_content: Callable[[Path], bool] = file_has_indexable_content,
) -> set:
    """
    The hashed files a full index has no evidence it indexed: no
    chunks, no chunking failure, yet something to index.

    iter_entries yields chunks and failures only, so a file that is chunked into
    nothing and a file the second tree walk never reached look the same. They
    differ in the content: an empty or whitespace-only file legitimately has no
    chunks, but one with text should have had some, so it was missed (deleted
    when the walk got there, say, and restored since). Recording its hash would
    make the next sync call it up to date although it was never indexed, and
    another file's chunks are enough to pass sync's collection check. A file
    that cannot be read now counts as missed too. Only files without chunks are
    read, which is few.

    `chunked_files` is every file that produced at least one chunk (a mapping
    keyed by path or a plain set). `has_indexable_content` is injectable for the
    same reason as invalidation_error's `discard`.
    """
    skipped_paths = {rel for rel, _ in chunk_skipped}
    missed = set()
    for rel in current_hashes:
        if rel in chunked_files or rel in skipped_paths:
            continue
        try:
            if has_indexable_content(repo / rel):
                missed.add(rel)
        except OSError:
            missed.add(rel)
    return missed


def index_baseline(current_hashes: dict, chunk_skipped: list, chunked_files, missed=()) -> dict:
    """
    The {file: hash} manifest a full index records when it finishes, in the
    same shape sync_repo's own manifest has so the next sync_repo diffs
    against it.

    `current_hashes` must have been taken BEFORE the files were chunked: a file
    edited in between then has a recorded hash older than the content that was
    indexed, so the next sync re-indexes it. Hashing afterwards would record the
    new content's hash against old chunks and sync would call it up to date for
    good.

    - A file that failed to chunk has no chunks (the collection was empty or
      reset), so it is left out and the next sync treats it as new and retries.
    - A file that produced chunks but has no hash is recorded as
      UNVERIFIED_HASH: retried by the next sync, and still known to the
      manifest so deleting it later removes its chunks. That is a file that
      could not be hashed (it became unreadable between the scan and the
      chunking) and also one created after the hash scan, since chunking walks
      the tree again and can find a file the hashing never saw.
    - A file with a hash that produced no chunks keeps its hash when there was
      nothing to index (empty or whitespace-only); when it has text but was
      never chunked, `missed` (see unindexed_hashed_files) records it as
      UNVERIFIED_HASH too.
    """
    skipped_paths = {rel for rel, _ in chunk_skipped}
    baseline = {rel: h for rel, h in current_hashes.items() if rel not in skipped_paths}
    for rel in chunked_files:
        if rel not in baseline and rel not in skipped_paths:
            baseline[rel] = UNVERIFIED_HASH
    for rel in missed:
        if rel in baseline:
            baseline[rel] = UNVERIFIED_HASH
    return baseline
