"""
The sync manifest's read, write, discard and baseline logic, shared by
mcpservers/ingest_mcp_server.py (sync_repo and index_repo) and tools/ingest_to_qdrant.py
(the CLI's opt-in --write-manifest), so the two cannot drift apart on a file
whose format every project's next sync_repo depends on.

Moved here from the server module, which imports MCP and therefore cannot be
imported by the CLI. The manifest format and
qdrant_ingest_lib.MANIFEST_FORMAT_VERSION are NOT touched: a version bump forces
a full re-embed on every project that has already been indexed.

Dependency-light on purpose (standard library plus qdrant_ingest_lib, which is
itself dependency-free): no Qdrant client, no MCP, no embedding model.

Concurrency. The manifest is one file per checkout, and two Claude Code
sessions on the same checkout (or a session and the CLI) can each run a sync at
once. Two separate guards cover that:

- save_manifest writes atomically (temp file, fsync, os.replace), so a reader
  never sees a half-written manifest. Before, a reader landing between the
  truncate and the write got a JSON error, which load_manifest turns into a
  full rebuild of the index for no reason.
- ManifestLock serializes the runs themselves. Atomic writes alone do not stop
  a lost update: run A scans a file at one version, B scans it after an edit,
  B stores the new chunks, then slower A replaces them with the old ones and
  saves, and B saves last. The collection then holds the old chunks while the
  manifest records the new hash, and every later sync skips the file as
  unchanged. Every run that reads or replaces the manifest holds the lock from
  its first manifest access (the load, or the removal of the old one) to its
  final save, and a second run is refused at once with a clear message instead
  of queueing behind a long embed.
"""

import errno
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from qdrant_ingest_lib import (
    BOOKKEEPING_TEMP_SUFFIX,
    MANIFEST_FILENAME,
    MANIFEST_LOCK_FILENAME,
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
    are added to it, from the SAME read: two reads could see two different
    files (a save landing in between, or, from a release before saves were
    atomic, a manifest truncated for a moment) and return a good hash map with
    an empty kept set, which a later save would write back without the watched
    paths. One read makes a bad read lose both, and no usable manifest already
    means a rebuild, which works the set out again. The set is left empty
    whenever the map is.

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

    Atomic (see _atomic_write_text): a concurrent load_manifest sees either the
    old manifest or the new one, never a truncated file. Raises OSError when the
    write fails, with the manifest left as it was and no temp file behind.
    """
    _atomic_write_text(repo / MANIFEST_FILENAME, manifest_to_json(manifest, preserved))


# os.replace onto the manifest can fail with PermissionError on Windows while
# another process has it open (a load_manifest that is reading it, which takes
# no lock). Those reads are short, so a few short retries ride them out.
_REPLACE_ATTEMPTS = 5
_REPLACE_RETRY_SECONDS = 0.05


def _atomic_write_text(
    path: Path,
    text: str,
    replace: Optional[Callable[[Path, Path], None]] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> None:
    """
    Write `text` (UTF-8) to `path` through a temp file in the same directory,
    fsynced and then renamed over `path` with os.replace, so a reader sees the
    old content or the new content and nothing in between.

    The temp file is named `<name>.<random>.tmp`, which iter_files skips for any
    excluded name (qdrant_ingest_lib.is_excluded_file_name), so one left behind
    by a crash is never indexed. It is created with os.open and mode 0o666
    rather than tempfile.mkstemp, whose 0o600 would make the manifest
    owner-only after the rename: this way it gets the permissions write_text
    gave it before (0o666 less the umask).

    `replace` and `sleep` are injectable for tests, and default to os.replace
    and time.sleep looked up at call time (so patching this module's os or
    time reaches them too). replace is tried _REPLACE_ATTEMPTS times in all
    while it raises PermissionError; anything else, or the last
    PermissionError, removes the temp file and is raised.
    """
    replace = replace if replace is not None else os.replace
    sleep = sleep if sleep is not None else time.sleep
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(8)}{BOOKKEEPING_TEMP_SUFFIX}")
    # O_BINARY exists only on Windows, where without it "\n" would be written
    # as "\r\n"; the manifest's bytes are then the same on every platform.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(1, _REPLACE_ATTEMPTS + 1):
            try:
                replace(tmp, path)
                return
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS:
                    raise
                sleep(_REPLACE_RETRY_SECONDS)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class ManifestLockBusy(Exception):
    """The manifest lock is held by another run (another process, or another
    ManifestLock in this one)."""


if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> None:
        # Locks the file's first byte; locking past the end of a file is
        # allowed on Windows, so the lock file can stay empty. LK_NBLCK fails
        # at once with EACCES (or EDEADLOCK) when another handle holds it.
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as e:
            if e.errno in (errno.EACCES, getattr(errno, "EDEADLOCK", errno.EACCES)):
                raise ManifestLockBusy() from e
            raise

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> None:
        # flock locks belong to the open file description, so a second open()
        # of the same file in the SAME process is refused too: two sync_repo
        # calls in one MCP server are serialized like two servers are. The
        # kernel drops the lock when the process dies, so a crashed run never
        # leaves the checkout locked.
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise ManifestLockBusy() from e
            raise

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def manifest_lock_busy_message(repo: Path) -> str:
    """What a run refused by ManifestLock.acquire returns or prints. Every
    caller acquires before deleting or storing anything, so "nothing was
    deleted or indexed" holds wherever it is shown."""
    return (
        f"Error: another index or sync run is in progress for this checkout ({repo}): it holds the sync "
        f"manifest's lock ({MANIFEST_LOCK_FILENAME}). Nothing was deleted or indexed. Run this again once that "
        f"run has finished."
    )


class ManifestLock:
    """
    Cross-process, non-blocking exclusive lock on one checkout's sync manifest,
    held through the sibling file MANIFEST_LOCK_FILENAME in the repo root (an
    OS file lock: fcntl.flock on POSIX, msvcrt.locking on Windows).

    acquire() returns None once the lock is held (and again on a repeat call
    while held), or manifest_lock_busy_message when another run holds it; it
    never waits. release() is idempotent and must run however the run ends
    (success, error, cancellation): callers put it in a `finally`.

    The lock file is created on first use and left in place afterwards, empty.
    Removing it on release would let a run that opened the old file and a run
    that created a new one each hold "the" lock at once. It is safe to
    gitignore alongside the manifest, and iter_files never indexes it.

    Fails open, as before the lock existed, in two cases, where `held` stays
    False and acquire() returns None: the lock file cannot be opened (a
    read-only checkout, where the manifest cannot be written either), and a
    filesystem that does not support the OS lock (some network mounts). Only a
    lock that another run demonstrably holds refuses the run.
    """

    def __init__(self, repo: Path):
        self.repo = repo
        self._fd: Optional[int] = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> Optional[str]:
        if self._fd is not None:
            return None
        path = self.repo / MANIFEST_LOCK_FILENAME
        fd = _open_lock_file(path)
        if fd is None:
            return None
        try:
            _try_lock(fd)
        except ManifestLockBusy:
            os.close(fd)
            return manifest_lock_busy_message(self.repo)
        except OSError:
            os.close(fd)
            return None
        self._fd = fd
        return None

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        except OSError:
            # Closing the descriptor releases the lock anyway.
            pass
        finally:
            os.close(fd)


def _open_lock_file(path: Path) -> Optional[int]:
    """A descriptor for the lock file, creating it if needed, or None when it
    cannot be opened. Falls back to read-only for a lock file another user
    created without write permission for this one: both OS locks work on a
    read-only descriptor, so the two users are still serialized."""
    binary = getattr(os, "O_BINARY", 0)
    try:
        return os.open(path, os.O_RDWR | os.O_CREAT | binary, 0o666)
    except OSError:
        pass
    try:
        return os.open(path, os.O_RDONLY | binary)
    except OSError:
        return None


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


def invalidation_error(
    repo: Path,
    discard: Callable[[Path], None] = discard_manifest,
    lock: Optional[ManifestLock] = None,
) -> Optional[str]:
    """
    Remove the old manifest and return None, or return the error message to
    abort a run with when it cannot be removed. The one wording index_repo and
    the CLI both stop on, in the shape prepare_collection_for_index's
    before_reset_delete hook expects. `discard` is injectable so each caller
    passes its own module-level name, which is the patch point its tests use.

    With `lock`, the manifest lock is taken first (a no-op when this run
    already holds it): removing the old manifest is a run's first manifest
    access, so this is where it starts to hold the lock, through its final
    save. A lock held by another run returns manifest_lock_busy_message, with
    the manifest untouched. The caller releases the lock however the run ends.
    """
    if lock is not None:
        busy = lock.acquire()
        if busy:
            return busy
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
