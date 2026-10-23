#!/usr/bin/env python3
"""Tests for serializing the sync manifest's users per checkout
(libs/sync_manifest_lib.py's ManifestLock and atomic save_manifest) and for
how sync_repo, index_repo and the ingest CLI hold that lock.

Two overlapping runs on one checkout could leave the collection holding one
version of a file's chunks while the manifest records another version's hash,
and every later sync_repo would then skip the file as unchanged. The lock makes
the second run fail fast instead; the atomic write keeps a reader from ever
seeing a half-written manifest.

Stdlib-only and model-free: Qdrant, the embedding provider and the store are
mocked, as in the test_ingest_mcp_server_* and test_ingest_to_qdrant files.
The cross-process tests start a real second Python process.

    .venv/bin/python -m unittest discover -s tests
"""

import argparse
import asyncio
import errno
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "mcpservers"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # fake_token_budget

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "test-collection")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

import ingest_mcp_server as _ims  # noqa: E402
import ingest_to_qdrant as _cli  # noqa: E402
import sync_manifest_lib as lib  # noqa: E402
from fake_token_budget import word_budget  # noqa: E402
from qdrant_ingest_lib import (  # noqa: E402
    MANIFEST_FILENAME,
    MANIFEST_LOCK_FILENAME,
    is_excluded_file_name,
    iter_files,
)

BUSY_PREFIX = "Error: another index or sync run is in progress for this checkout"


def _provider_cls():
    cls = mock.MagicMock(name="FastEmbedProvider")
    cls.return_value.build_token_budget.return_value = word_budget(20)
    return cls


class RepoTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name).resolve()
        self.manifest_path = self.repo / MANIFEST_FILENAME

    def _hold(self) -> lib.ManifestLock:
        """A lock held by 'another run' for the rest of the test."""
        other = lib.ManifestLock(self.repo)
        self.assertIsNone(other.acquire())
        self.assertTrue(other.held)
        self.addCleanup(other.release)
        return other

    def assert_lock_is_free(self):
        probe = lib.ManifestLock(self.repo)
        self.assertIsNone(probe.acquire(), "the lock must be free again once the run has ended")
        self.assertTrue(probe.held)
        probe.release()

    def _temp_files(self) -> list:
        return [p.name for p in self.repo.iterdir() if p.name.endswith(".tmp")]


# ---------------------------------------------------------------------------
# ManifestLock itself
# ---------------------------------------------------------------------------


class ManifestLockInProcess(RepoTestCase):
    def test_a_second_lock_is_refused_while_the_first_is_held_and_granted_after(self):
        first = lib.ManifestLock(self.repo)
        second = lib.ManifestLock(self.repo)
        self.assertIsNone(first.acquire())
        busy = second.acquire()
        self.assertTrue(busy.startswith(BUSY_PREFIX), busy)
        self.assertIn(MANIFEST_LOCK_FILENAME, busy)
        self.assertIn("Nothing was deleted or indexed", busy)
        self.assertFalse(second.held)
        first.release()
        self.assertIsNone(second.acquire())
        self.assertTrue(second.held)
        second.release()

    def test_acquire_is_idempotent_for_the_holder_and_release_is_idempotent(self):
        lock = lib.ManifestLock(self.repo)
        self.assertIsNone(lock.acquire())
        self.assertIsNone(lock.acquire())
        lock.release()
        lock.release()
        self.assertFalse(lock.held)
        self.assert_lock_is_free()

    def test_the_lock_file_is_created_and_left_in_place_empty(self):
        lock = lib.ManifestLock(self.repo)
        lock.acquire()
        lock.release()
        lock_path = self.repo / MANIFEST_LOCK_FILENAME
        self.assertTrue(lock_path.exists())
        self.assertEqual(lock_path.read_bytes(), b"")

    def test_a_read_only_checkout_fails_open(self):
        # The manifest cannot be written there either, so there is nothing
        # for an overlapping run to get out of step.
        lock = lib.ManifestLock(self.repo)
        with mock.patch.object(lib, "_open_lock_file", side_effect=OSError(errno.EROFS, "read-only")):
            self.assertIsNone(lock.acquire())
        self.assertFalse(lock.held)
        lock.release()

    def test_a_repo_root_this_user_cannot_write_fails_open(self):
        lock = lib.ManifestLock(self.repo)
        with mock.patch.object(lib, "_open_lock_file", side_effect=OSError(errno.EACCES, "denied")), \
             mock.patch.object(lib.os, "access", return_value=False):
            self.assertIsNone(lock.acquire())
        self.assertFalse(lock.held)

    def test_any_other_failure_to_open_the_lock_file_refuses_the_run(self):
        # Including a permission error on the lock file in a WRITABLE repo
        # root: the manifest could still be written there, unlocked.
        for error in (OSError(errno.EIO, "I/O error"), OSError(errno.EACCES, "denied")):
            with self.subTest(errno=error.errno):
                lock = lib.ManifestLock(self.repo)
                with mock.patch.object(lib, "_open_lock_file", side_effect=error):
                    message = lock.acquire()
                self.assertTrue(message.startswith("Error: could not lock this checkout's sync manifest"), message)
                self.assertIn("Nothing was deleted or indexed", message)
                self.assertFalse(lock.held)

    def test_a_filesystem_without_lock_support_fails_open_and_closes_the_descriptor(self):
        real_close = os.close
        for code in sorted(lib._LOCKING_UNSUPPORTED_ERRNOS):
            with self.subTest(errno=code):
                lock = lib.ManifestLock(self.repo)
                closed: list = []
                with mock.patch.object(lib, "_try_lock", side_effect=OSError(code, "no locks")), \
                     mock.patch.object(lib.os, "close", side_effect=lambda fd, c=closed: c.append(fd) or real_close(fd)):
                    self.assertIsNone(lock.acquire())
                self.assertFalse(lock.held)
                self.assertEqual(len(closed), 1)

    def test_any_other_lock_error_refuses_the_run_and_closes_the_descriptor(self):
        # Running unlocked on, say, an I/O error is exactly what lets two runs
        # leave the manifest out of step with the collection.
        lock = lib.ManifestLock(self.repo)
        closed = []
        real_close = os.close
        with mock.patch.object(lib, "_try_lock", side_effect=OSError(errno.EIO, "I/O error")), \
             mock.patch.object(lib.os, "close", side_effect=lambda fd: closed.append(fd) or real_close(fd)):
            message = lock.acquire()
        self.assertTrue(message.startswith("Error: could not lock this checkout's sync manifest"), message)
        self.assertIn("I/O error", message)
        self.assertFalse(lock.held)
        self.assertEqual(len(closed), 1)

    def test_the_real_open_raises_when_neither_mode_works(self):
        with mock.patch.object(lib.os, "open", side_effect=OSError(errno.EIO, "I/O error")):
            with self.assertRaises(OSError) as caught:
                lib._open_lock_file(self.repo / MANIFEST_LOCK_FILENAME)
        self.assertEqual(caught.exception.errno, errno.EIO)

    @unittest.skipIf(sys.platform == "win32", "POSIX permission bits")
    def test_a_lock_file_this_user_cannot_write_still_locks(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root ignores permission bits")
        lock_path = self.repo / MANIFEST_LOCK_FILENAME
        lock_path.touch()
        lock_path.chmod(0o444)
        self.addCleanup(lock_path.chmod, 0o644)
        first = lib.ManifestLock(self.repo)
        self.assertIsNone(first.acquire())
        self.assertTrue(first.held)
        self.assertTrue(lib.ManifestLock(self.repo).acquire().startswith(BUSY_PREFIX))
        first.release()


_HOLDER_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import sync_manifest_lib as lib
lock = lib.ManifestLock(Path(sys.argv[2]))
print("busy" if lock.acquire() else ("held" if lock.held else "open"), flush=True)
sys.stdin.readline()
if sys.argv[3] == "release":
    lock.release()
    print("released", flush=True)
    sys.stdin.readline()
"""


class ManifestLockAcrossProcesses(RepoTestCase):
    """The actual case: two processes (two Claude Code sessions' servers, or a
    server and the CLI) on one checkout."""

    def _start_holder(self, then: str) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, "-c", _HOLDER_SCRIPT, os.path.join(REPO_ROOT, "libs"), str(self.repo), then],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        self.addCleanup(self._stop, proc)
        self.assertEqual(proc.stdout.readline().strip(), "held")
        return proc

    @staticmethod
    def _stop(proc):
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        for stream in (proc.stdin, proc.stdout):
            if stream:
                stream.close()

    def test_another_process_holding_the_lock_refuses_this_one_until_it_releases(self):
        proc = self._start_holder("release")
        self.assertTrue(lib.ManifestLock(self.repo).acquire().startswith(BUSY_PREFIX))
        proc.stdin.write("\n")
        proc.stdin.flush()
        self.assertEqual(proc.stdout.readline().strip(), "released")
        self.assert_lock_is_free()

    @unittest.skipIf(
        sys.platform == "win32",
        "Windows releases a dead process's byte-range locks too, but asynchronously, so not by the next line",
    )
    def test_a_process_that_dies_holding_the_lock_does_not_leave_the_checkout_locked(self):
        proc = self._start_holder("exit")
        self.assertTrue(lib.ManifestLock(self.repo).acquire().startswith(BUSY_PREFIX))
        proc.kill()
        proc.wait(timeout=10)
        self.assert_lock_is_free()


class InvalidationTakesTheLock(RepoTestCase):
    def test_a_lock_held_elsewhere_stops_it_before_the_manifest_is_removed(self):
        lib.save_manifest(self.repo, {"a.py": "h"})
        self._hold()
        mine = lib.ManifestLock(self.repo)
        message = lib.invalidation_error(self.repo, lock=mine)
        self.assertTrue(message.startswith(BUSY_PREFIX), message)
        self.assertTrue(self.manifest_path.exists(), "a refused run must not remove the manifest")
        self.assertFalse(mine.held)

    def test_it_takes_the_lock_then_removes_the_manifest(self):
        lib.save_manifest(self.repo, {"a.py": "h"})
        mine = lib.ManifestLock(self.repo)
        self.addCleanup(mine.release)
        self.assertIsNone(lib.invalidation_error(self.repo, lock=mine))
        self.assertTrue(mine.held)
        self.assertFalse(self.manifest_path.exists())
        # Called again in the same run (index_repo's reset hook, then its
        # baseline step): still this run's lock, no refusal.
        self.assertIsNone(lib.invalidation_error(self.repo, lock=mine))


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------


class AtomicSave(RepoTestCase):
    def test_overlapping_saves_never_show_a_reader_a_truncated_manifest(self):
        # Before the atomic write, save_manifest truncated the file and then
        # wrote it, so a reader landing in between got a JSON error and
        # load_manifest's {} -- a needless full rebuild.
        big = {f"dir/file_{i:05d}.py": "h" * 64 for i in range(3000)}
        lib.save_manifest(self.repo, big)
        stop = threading.Event()
        errors: list = []

        def writer(tag):
            n = 0
            while not stop.is_set():
                try:
                    lib.save_manifest(self.repo, {**big, "tag": f"{tag}-{n}"})
                except PermissionError:
                    pass  # Windows: replace refused while a reader holds the file; retried, then raised
                except Exception as e:  # pragma: no cover - the assertion below reports it
                    errors.append(e)
                n += 1

        threads = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b")]
        for t in threads:
            t.start()
        try:
            empty_reads = sum(1 for _ in range(300) if not lib.load_manifest(self.repo))
        finally:
            stop.set()
            for t in threads:
                t.join()
        self.assertEqual(errors, [])
        self.assertEqual(empty_reads, 0, "a reader saw a partly written manifest")
        self.assertEqual(len(lib.load_manifest(self.repo)), len(big) + 1)
        self.assertEqual(self._temp_files(), [])

    def test_a_failed_replace_leaves_the_old_manifest_and_no_temp_file(self):
        lib.save_manifest(self.repo, {"old.py": "h"})
        with mock.patch.object(lib.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                lib.save_manifest(self.repo, {"new.py": "h"})
        self.assertEqual(lib.load_manifest(self.repo), {"old.py": "h"})
        self.assertEqual(self._temp_files(), [])

    def test_a_failed_write_leaves_no_temp_file(self):
        with mock.patch.object(lib.os, "fsync", side_effect=OSError("io error")):
            with self.assertRaises(OSError):
                lib.save_manifest(self.repo, {"a.py": "h"})
        self.assertFalse(self.manifest_path.exists())
        self.assertEqual(self._temp_files(), [])

    def test_a_permission_error_on_replace_is_retried(self):
        real_replace = os.replace
        calls = []

        def flaky(src, dst):
            calls.append(src)
            if len(calls) < 3:
                raise PermissionError(errno.EACCES, "in use")
            real_replace(src, dst)

        sleeps = []
        lib._atomic_write_text(self.manifest_path, "{}", replace=flaky, sleep=sleeps.append)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [lib._REPLACE_RETRY_SECONDS] * 2)
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"), "{}")
        self.assertEqual(self._temp_files(), [])

    def test_a_lasting_permission_error_is_raised_after_the_retries(self):
        replace = mock.MagicMock(side_effect=PermissionError(errno.EACCES, "in use"))
        with self.assertRaises(PermissionError):
            lib._atomic_write_text(self.manifest_path, "{}", replace=replace, sleep=lambda s: None)
        self.assertEqual(replace.call_count, lib._REPLACE_ATTEMPTS)
        self.assertEqual(self._temp_files(), [])

    def test_the_temp_file_is_a_side_file_the_scanner_skips(self):
        seen = []
        real_replace = os.replace

        def spy(src, dst):
            seen.append(Path(src).name)
            real_replace(src, dst)

        lib._atomic_write_text(self.manifest_path, "{}", replace=spy)
        self.assertEqual(len(seen), 1)
        self.assertTrue(is_excluded_file_name(seen[0], {MANIFEST_FILENAME}), seen[0])

    @unittest.skipIf(sys.platform == "win32", "POSIX permission bits")
    def test_the_manifest_keeps_the_permissions_a_plain_write_gives_it(self):
        # tempfile.mkstemp would create it 0o600 and the rename would keep that.
        umask = os.umask(0o022)
        os.umask(umask)
        lib.save_manifest(self.repo, {"a.py": "h"})
        self.assertEqual(stat.S_IMODE(self.manifest_path.stat().st_mode), 0o666 & ~umask)

    def test_the_bytes_are_the_same_on_every_platform(self):
        lib.save_manifest(self.repo, {"a.py": "h"})
        self.assertNotIn(b"\r\n", self.manifest_path.read_bytes())


# ---------------------------------------------------------------------------
# The scanner never indexes the lock or a leftover temp file
# ---------------------------------------------------------------------------


class ScannerSkipsSideFiles(RepoTestCase):
    def test_side_files_of_an_excluded_name_are_skipped(self):
        names = [
            MANIFEST_FILENAME,
            MANIFEST_LOCK_FILENAME,
            f"{MANIFEST_FILENAME}.0123abcd.tmp",
            "keep.json",
            "other.lock",
            "other.tmp",
            f"{MANIFEST_FILENAME}.backup",
            f"x{MANIFEST_FILENAME}.1.tmp",
        ]
        for name in names:
            (self.repo / name).write_text("{}")
        found = {
            p.name for p in iter_files(self.repo, {".json", ".lock", ".tmp", ".backup"},
                                       extra_exclude_files={MANIFEST_FILENAME})
        }
        self.assertEqual(found, {"keep.json", "other.lock", "other.tmp", f"{MANIFEST_FILENAME}.backup",
                                 f"x{MANIFEST_FILENAME}.1.tmp"})

    def test_without_an_exclusion_nothing_is_skipped(self):
        self.assertFalse(is_excluded_file_name(MANIFEST_LOCK_FILENAME, set()))


# ---------------------------------------------------------------------------
# sync_repo
# ---------------------------------------------------------------------------


class SyncRepoHoldsTheLock(RepoTestCase):
    def setUp(self):
        super().setUp()
        (self.repo / "a.txt").write_text("alpha beta gamma")
        (self.repo / "b.txt").write_text("delta epsilon")
        self.client = mock.MagicMock()
        self.client.collection_exists.return_value = True
        self.client.count.return_value = mock.MagicMock(count=5)
        self.lock_seen_during_store: list = []

        async def _store(connector, entries, **kw):
            # Another run trying its luck while this one is embedding.
            self.lock_seen_during_store.append(lib.ManifestLock(self.repo).acquire())
            return len(entries)

        self.store = mock.AsyncMock(side_effect=_store)
        for target, kwargs in (
            ("store_batch", {"new": self.store}),
            ("QdrantConnector", {}),
            ("ensure_file_path_index", {}),
            ("check_embedding_model_mismatch", {"return_value": None}),
            ("_recompute_search_limit", {}),
            ("_non_chunk_points_warning", {"return_value": ""}),
            ("_delete_orphaned_chunks", {"return_value": []}),
        ):
            patcher = mock.patch.object(_ims, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _sync_coro(self):
        return _ims.sync_repo(repo_path=str(self.repo), collection="c", scope="docs", include_extensions=".txt")

    def _sync(self):
        with mock.patch.object(_ims, "QdrantClient", return_value=self.client), \
             mock.patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            return asyncio.run(self._sync_coro())

    def test_a_held_lock_refuses_the_sync_before_the_manifest_is_read(self):
        lib.save_manifest(self.repo, {"a.txt": "old"})
        before = self.manifest_path.read_bytes()
        self._hold()
        with mock.patch.object(_ims, "_load_manifest", wraps=_ims._load_manifest) as load:
            result = self._sync()
        self.assertTrue(result.startswith(BUSY_PREFIX), result)
        load.assert_not_called()
        self.store.assert_not_called()
        self.client.delete.assert_not_called()
        self.assertEqual(self.manifest_path.read_bytes(), before)

    def test_the_lock_is_held_through_the_store_and_released_on_success(self):
        result = self._sync()
        self.assertIn("Synced 'c'", result)
        self.assertEqual(len(self.lock_seen_during_store), 1)
        self.assertTrue(self.lock_seen_during_store[0].startswith(BUSY_PREFIX))
        self.assertTrue(self.manifest_path.exists())
        self.assert_lock_is_free()

    def test_the_lock_is_released_when_the_sync_fails(self):
        self.store.side_effect = RuntimeError("qdrant went away")
        with self.assertRaises(RuntimeError):
            self._sync()
        self.assert_lock_is_free()

    def test_the_lock_is_released_on_the_no_changes_return(self):
        self._sync()
        self.assertIn("No changes since last sync", self._sync())
        self.assert_lock_is_free()

    def test_the_lock_is_released_when_the_sync_is_cancelled(self):
        entered = asyncio.Event()

        async def _hang(connector, entries, **kw):
            entered.set()
            await asyncio.Event().wait()

        self.store.side_effect = _hang

        async def drive():
            task = asyncio.ensure_future(self._sync_coro())
            await entered.wait()
            self.assertTrue(lib.ManifestLock(self.repo).acquire().startswith(BUSY_PREFIX))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with mock.patch.object(_ims, "QdrantClient", return_value=self.client), \
             mock.patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            asyncio.run(drive())
        self.assert_lock_is_free()

    def test_a_second_sync_in_the_same_server_is_refused_while_the_first_runs(self):
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def _wait(connector, entries, **kw):
            entered.set()
            await finish.wait()
            return len(entries)

        self.store.side_effect = _wait

        async def drive():
            first = asyncio.ensure_future(self._sync_coro())
            await entered.wait()
            second = await self._sync_coro()
            finish.set()
            return await first, second

        with mock.patch.object(_ims, "QdrantClient", return_value=self.client), \
             mock.patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            first, second = asyncio.run(drive())
        self.assertIn("Synced 'c'", first)
        self.assertTrue(second.startswith(BUSY_PREFIX), second)
        self.assert_lock_is_free()

    def test_the_docstring_says_it_holds_the_lock(self):
        doc = " ".join((_ims.sync_repo.__doc__ or "").split())
        self.assertIn(MANIFEST_LOCK_FILENAME, doc)
        self.assertIn("returns an error at once", doc)


# ---------------------------------------------------------------------------
# index_repo
# ---------------------------------------------------------------------------


class IndexRepoHoldsTheLockOnlyWhenItWritesTheManifest(RepoTestCase):
    def setUp(self):
        super().setUp()
        (self.repo / "a.txt").write_text("alpha beta gamma")
        self.client = mock.MagicMock()
        self.client.collection_exists.return_value = False
        self.lock_seen_during_store: list = []
        self.stored: list = []

        async def _store(connector, entries, **kw):
            self.lock_seen_during_store.append(lib.ManifestLock(self.repo).acquire())
            self.stored.extend(entries)
            return len(entries)

        self.store = mock.AsyncMock(side_effect=_store)
        for target, kwargs in (
            ("store_batch", {"new": self.store}),
            ("QdrantConnector", {}),
            ("ensure_file_path_index", {}),
            ("check_embedding_model_mismatch", {"return_value": None}),
            ("_recompute_search_limit", {}),
            ("_sync_static_collection_description", {}),
            ("DEFAULT_COLLECTION", {"new": "c"}),
        ):
            patcher = mock.patch.object(_ims, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _index(self, **kwargs):
        with mock.patch.object(_ims, "QdrantClient", return_value=self.client), \
             mock.patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            return asyncio.run(_ims.index_repo(
                repo_path=str(self.repo), scope="docs", include_extensions=".txt", **{"collection": "c", **kwargs}
            ))

    def test_a_baseline_run_is_refused_while_the_lock_is_held(self):
        lib.save_manifest(self.repo, {"a.txt": "old"})
        self._hold()
        result = self._index()
        self.assertTrue(result.startswith(BUSY_PREFIX), result)
        self.store.assert_not_called()
        self.assertEqual(lib.load_manifest(self.repo), {"a.txt": "old"})

    def test_a_reset_is_refused_before_its_delete(self):
        self.client.collection_exists.return_value = True
        self._hold()
        result = self._index(reset=True)
        self.assertTrue(result.startswith(BUSY_PREFIX), result)
        self.client.delete.assert_not_called()
        self.store.assert_not_called()

    def test_a_baseline_run_holds_the_lock_through_the_store_and_releases_it(self):
        result = self._index()
        self.assertIn("Wrote the sync manifest", result)
        self.assertTrue(self.lock_seen_during_store[0].startswith(BUSY_PREFIX))
        self.assert_lock_is_free()

    def test_the_lock_is_released_when_a_baseline_run_fails(self):
        self.store.side_effect = RuntimeError("qdrant went away")
        with self.assertRaises(RuntimeError):
            self._index()
        self.assert_lock_is_free()

    def test_a_run_into_another_collection_never_takes_the_lock(self):
        self._hold()
        result = self._index(collection="elsewhere")
        self.assertIn("Indexed", result)
        self.assertIn("The sync manifest was not written", result)
        self.assertEqual(len(self.stored), 1)

    def test_the_lock_is_already_held_while_the_collection_is_prepared_and_counted(self):
        # Whether this run may write a baseline rests on the collection being
        # absent or empty when prepared. Taking the lock only afterwards let a
        # sync_repo fill the collection and save its manifest in between, and
        # this run then appended duplicates and wrote a baseline that did not
        # describe them.
        seen = []
        real_prep = _ims.prepare_collection_for_index

        def prep(*a, **kw):
            seen.append(lib.ManifestLock(self.repo).acquire())
            return real_prep(*a, **kw)

        with mock.patch.object(_ims, "prepare_collection_for_index", side_effect=prep):
            self.assertIn("Wrote the sync manifest", self._index())
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].startswith(BUSY_PREFIX), seen[0])

    def test_every_run_into_its_own_collection_is_refused_while_the_lock_is_held(self):
        # Including one that would end up not writing the manifest (force into
        # a populated collection): that is only known after the count, which
        # has to happen under the lock.
        self.client.collection_exists.return_value = True
        self.client.get_collection.return_value = mock.MagicMock(points_count=7)
        self.client.count.return_value = mock.MagicMock(count=7)
        self._hold()
        result = self._index(force=True)
        self.assertTrue(result.startswith(BUSY_PREFIX), result)
        self.store.assert_not_called()
        self.client.get_collection.assert_not_called()

    def test_the_docstring_says_when_it_holds_the_lock(self):
        doc = " ".join((_ims.index_repo.__doc__ or "").split())
        self.assertIn(MANIFEST_LOCK_FILENAME, doc)
        self.assertIn("before it prepares and counts the collection", doc)
        self.assertIn("A run into any other collection never takes the lock", doc)


class ToolsKeepTheirSignatures(unittest.TestCase):
    """The tool wrappers must expose exactly the parameters the bodies take,
    minus the lock: MCP builds each tool's schema from its signature."""

    def test_the_wrappers_expose_no_lock_parameter(self):
        import inspect

        for tool, body in ((_ims.index_repo, _ims._index_repo), (_ims.sync_repo, _ims._sync_repo)):
            with self.subTest(tool=tool.__name__):
                tool_params = list(inspect.signature(tool).parameters)
                body_params = list(inspect.signature(body).parameters)
                self.assertNotIn("manifest_lock", tool_params)
                self.assertEqual(tool_params, [p for p in body_params if p != "manifest_lock"])


# ---------------------------------------------------------------------------
# The CLI's --write-manifest
# ---------------------------------------------------------------------------


class CliHoldsTheLockOnlyWithWriteManifest(RepoTestCase):
    def setUp(self):
        super().setUp()
        (self.repo / "a.txt").write_text("alpha beta gamma\n")
        self.client = mock.MagicMock()
        self.client.collection_exists.return_value = False
        self.lock_seen_during_store: list = []

        async def _store(connector, entries, **kw):
            self.lock_seen_during_store.append(lib.ManifestLock(self.repo).acquire())
            return len(entries)

        self.store = mock.AsyncMock(side_effect=_store)
        patcher = mock.patch("qdrant_retry.RETRY_BACKOFF_SECONDS", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _ingest(self, **overrides):
        args = argparse.Namespace(**{
            **dict(
                repo_path=str(self.repo), collection="c", qdrant_url="http://localhost:6333", qdrant_api_key=None,
                embedding_model="m", scope="both", include_ext=".txt", exclude_dirs=None, no_gitignore=True,
                chunk_lines=50, overlap=10, dry_run=False, reset=False, write_manifest=True,
                memory_bank_collection=None,
            ),
            **overrides,
        })
        self.out, self.err = io.StringIO(), io.StringIO()
        with mock.patch.object(_cli, "QdrantClient", return_value=self.client), \
             mock.patch.object(_cli, "FastEmbedProvider", new=_provider_cls()), \
             mock.patch.object(_cli, "QdrantConnector"), \
             mock.patch.object(_cli, "ensure_file_path_index", return_value=False), \
             mock.patch.object(_cli, "check_embedding_model_mismatch", return_value=None), \
             mock.patch.object(_cli, "store_batch", new=self.store), \
             redirect_stdout(self.out), redirect_stderr(self.err):
            asyncio.run(_cli.ingest(args))

    def test_write_manifest_is_refused_while_the_lock_is_held(self):
        lib.save_manifest(self.repo, {"sentinel.txt": "h"})
        self._hold()
        with self.assertRaises(SystemExit) as caught:
            self._ingest()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn(BUSY_PREFIX, self.err.getvalue())
        self.store.assert_not_called()
        self.assertEqual(lib.load_manifest(self.repo), {"sentinel.txt": "h"})

    def test_a_reset_with_write_manifest_is_refused_before_its_delete(self):
        self.client.collection_exists.return_value = True
        self._hold()
        with self.assertRaises(SystemExit):
            self._ingest(reset=True)
        self.client.delete.assert_not_called()
        self.store.assert_not_called()

    def test_the_lock_is_already_held_while_the_collection_is_prepared_and_counted(self):
        # Same race as index_repo's: the clean-slate decision must be made
        # under the lock, not before it.
        seen = []
        real_prep = _cli.prepare_collection_for_index

        def prep(*a, **kw):
            seen.append(lib.ManifestLock(self.repo).acquire())
            return real_prep(*a, **kw)

        with mock.patch.object(_cli, "prepare_collection_for_index", side_effect=prep):
            self._ingest()
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].startswith(BUSY_PREFIX), seen[0])
        self.assertIn("Wrote the sync manifest", self.out.getvalue())

    def test_a_refused_run_never_reaches_qdrant(self):
        self._hold()
        with mock.patch.object(_cli, "prepare_collection_for_index") as prep, self.assertRaises(SystemExit):
            self._ingest()
        prep.assert_not_called()
        self.client.collection_exists.assert_not_called()

    def test_a_dry_run_with_write_manifest_never_takes_the_lock(self):
        self._hold()
        self._ingest(dry_run=True)
        self.assertIn("dry run, not storing", self.out.getvalue())
        self.assertNotIn(BUSY_PREFIX, self.err.getvalue())

    def test_write_manifest_holds_the_lock_through_the_store_and_releases_it(self):
        self._ingest()
        self.assertIn("Wrote the sync manifest", self.out.getvalue())
        self.assertTrue(self.lock_seen_during_store[0].startswith(BUSY_PREFIX))
        self.assert_lock_is_free()

    def test_the_lock_is_released_when_the_run_fails(self):
        self.store.side_effect = RuntimeError("qdrant went away")
        with self.assertRaises(RuntimeError):
            self._ingest()
        self.assert_lock_is_free()

    def test_without_write_manifest_the_lock_is_never_taken(self):
        # Another run holds the lock throughout, yet this run indexes normally:
        # without the flag it never touches the manifest, so it never asks.
        self._hold()
        self._ingest(write_manifest=False)
        self.store.assert_called_once()
        self.assertNotIn(BUSY_PREFIX, self.err.getvalue())

    def test_help_mentions_the_lock(self):
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["ingest_to_qdrant.py", "--help"]), \
             self.assertRaises(SystemExit), redirect_stdout(out):
            _cli.parse_args()
        self.assertIn(MANIFEST_LOCK_FILENAME, " ".join(out.getvalue().split()))


class SavedManifestIsStillPlainJson(RepoTestCase):
    def test_round_trip_through_json(self):
        lib.save_manifest(self.repo, {"a.py": "h"}, preserved={"k.py"})
        doc = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["files"], {"a.py": "h"})


if __name__ == "__main__":
    unittest.main()
