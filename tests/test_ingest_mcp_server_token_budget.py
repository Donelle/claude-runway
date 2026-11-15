#!/usr/bin/env python3
"""Tests for the token-budget chunking wiring in mcpservers/ingest_mcp_server.py:
index_repo, sync_repo and preview_index chunk by the embedding
model's window, fail before touching anything when it cannot be read, report
per-file progress and the largest files, read both metadata shapes, and
rebuild safely from a missing or old-format manifest.

Stdlib-only and model-free: FastEmbedProvider is replaced by a stand-in whose
budget counts words (tests/fake_token_budget.py). The two rebuild helpers that
talk to Qdrant run against qdrant-client's in-process `:memory:` mode, so their
filters are exercised for real rather than asserted against a mock.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "mcpservers"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "test-collection")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

import ingest_mcp_server as _ims  # noqa: E402
import qdrant_ingest_lib as _lib  # noqa: E402
from fake_token_budget import word_budget  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402
from qdrant_ingest_lib import MANIFEST_FILENAME, TokenBudgetError  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def _progress_messages(ctx) -> list:
    return [c.kwargs["message"] for c in ctx.report_progress.call_args_list if "message" in c.kwargs]


def _hashing_reports_every_file():
    """Patch compute_file_hashes so it reports after EVERY file (the real
    interval is HASH_PROGRESS_EVERY, far above these two-file repos)."""
    real = _ims.compute_file_hashes
    return patch.object(
        _ims, "compute_file_hashes", side_effect=lambda *a, **kw: real(*a, progress_every=1, **kw)
    )


def _provider_cls(max_tokens=20, window_error=None):
    cls = MagicMock(name="FastEmbedProvider")
    if window_error:
        cls.return_value.build_token_budget.side_effect = TokenBudgetError(window_error)
    else:
        cls.return_value.build_token_budget.return_value = word_budget(max_tokens)
    return cls


# ---------------------------------------------------------------------------
# The two rebuild helpers, against a real (in-memory) Qdrant client.
# ---------------------------------------------------------------------------


def _local_collection(points):
    client = QdrantClient(":memory:")
    client.create_collection("col", vectors_config={"v": models.VectorParams(size=2, distance=models.Distance.COSINE)})
    client.upsert(
        "col",
        [
            models.PointStruct(id=i, vector={"v": [1.0, 0.0]}, payload={"document": "text", "metadata": md})
            for i, md in points.items()
        ],
    )
    return client


def _ids(client):
    return sorted(p.id for p in client.scroll("col", limit=1000)[0])


class NonChunkPointsWarning(unittest.TestCase):
    def test_reports_the_count_and_a_sample_of_points_the_indexer_did_not_write(self):
        client = _local_collection(
            {
                1: {"file_path": "a.py", "type": "code"},
                2: {"file_path": "a.py", "note": "stored with qdrant-store"},
                3: {"source": "https://example.com/page"},
                4: {"file_path": "b.md", "type": "doc"},
            }
        )
        warning = _ims._non_chunk_points_warning(client, "col")
        self.assertIn("holds 2 point(s) that are not indexer chunks", warning)
        self.assertIn("stored with qdrant-store", warning)
        self.assertIn("deleted with that file's old chunks", warning)

    def test_no_non_chunk_points_means_no_warning(self):
        client = _local_collection({1: {"file_path": "a.py", "type": "code"}, 2: {"file_path": "b.md", "type": "doc"}})
        self.assertEqual(_ims._non_chunk_points_warning(client, "col"), "")

    def test_protected_points_are_not_counted_because_the_delete_skips_them(self):
        client = _local_collection(
            {
                1: {"file_path": "a.py", "type": "code"},
                2: {"source": "memory-bank", "file_path": "a.py"},
                3: {"source": "conversation-compact"},
            }
        )
        self.assertEqual(_ims._non_chunk_points_warning(client, "col"), "")

    def test_sample_is_bounded_but_the_count_is_exact(self):
        client = _local_collection({i: {"note": f"n{i}"} for i in range(1, 11)})
        warning = _ims._non_chunk_points_warning(client, "col")
        self.assertIn("holds 10 point(s)", warning)
        self.assertLessEqual(warning.count("{"), _ims._NON_CHUNK_SAMPLE_SIZE)


class DeleteOrphanedChunks(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)

    def test_removes_chunks_of_files_no_longer_on_disk_and_nothing_else(self):
        (self.repo / "keep.py").write_text("x")
        client = _local_collection(
            {
                1: {"file_path": "keep.py", "type": "code"},
                2: {"file_path": "gone.py", "type": "code"},
                3: {"file_path": "gone.md", "type": "doc"},
                4: {"file_path": "gone.py", "note": "a stored note, not a chunk"},
                5: {"source": "memory-bank", "file_path": "gone.py"},
                6: {"source": "conversation-compact"},
                7: {"source": "https://example.com"},
            }
        )
        removed = _ims._delete_orphaned_chunks(client, "col", self.repo)
        self.assertEqual(removed, ["gone.md", "gone.py"])
        self.assertEqual(_ids(client), [1, 4, 5, 6, 7])

    def test_existence_is_tested_on_disk_not_against_a_scan(self):
        """A file that is still there but outside this sync's scope or extension
        filter must keep its chunks: a sync_repo(scope="code") must not remove
        the doc chunks of files that exist."""
        (self.repo / "README.md").write_text("# hi")
        (self.repo / "dir").mkdir()
        (self.repo / "dir" / "gitignored.py").write_text("x")
        client = _local_collection(
            {
                1: {"file_path": "README.md", "type": "doc"},
                2: {"file_path": os.path.join("dir", "gitignored.py"), "type": "code"},
            }
        )
        self.assertEqual(_ims._delete_orphaned_chunks(client, "col", self.repo), [])
        self.assertEqual(_ids(client), [1, 2])

    def test_a_dangling_symlink_is_still_a_directory_entry_so_its_chunks_stay(self):
        """Hashing a symlink whose target has disappeared fails, so sync skips
        the file as unreadable and leaves its embeddings alone. Path.exists()
        follows the link and says False, which would have deleted them here."""
        link = self.repo / "dangling.py"
        try:
            link.symlink_to(self.repo / "no-such-target.py")
        except (OSError, NotImplementedError):
            self.skipTest("this platform cannot create symlinks")
        client = _local_collection(
            {
                1: {"file_path": "dangling.py", "type": "code"},
                2: {"file_path": "really-gone.py", "type": "code"},
            }
        )
        self.assertFalse(link.exists())
        self.assertEqual(_ims._delete_orphaned_chunks(client, "col", self.repo), ["really-gone.py"])
        self.assertEqual(_ids(client), [1])

    def test_a_file_whose_existence_cannot_be_checked_keeps_its_chunks(self):
        """os.path.lexists returns False for a permission error as well as for a
        missing path, so a file under a directory the process cannot search was
        classified as gone and its embeddings deleted."""
        if os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0):
            self.skipTest("needs POSIX permissions and a non-root user")
        locked = self.repo / "locked"
        locked.mkdir()
        (locked / "inside.py").write_text("x")
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o700)
        if os.path.lexists(locked / "inside.py"):
            self.skipTest("this filesystem lets the locked directory be searched")
        client = _local_collection(
            {
                1: {"file_path": os.path.join("locked", "inside.py"), "type": "code"},
                2: {"file_path": "really-gone.py", "type": "code"},
            }
        )
        self.assertEqual(_ims._delete_orphaned_chunks(client, "col", self.repo), ["really-gone.py"])
        self.assertEqual(_ids(client), [1])

    def test_a_path_through_a_file_is_confirmed_absent(self):
        (self.repo / "plain.py").write_text("x")
        self.assertTrue(_ims._confirmed_absent(self.repo / "plain.py" / "child.py"))
        self.assertTrue(_ims._confirmed_absent(self.repo / "nope.py"))
        self.assertFalse(_ims._confirmed_absent(self.repo / "plain.py"))

    def test_pages_through_a_listing_larger_than_one_page(self):
        points = {i: {"file_path": f"gone{i}.py", "type": "code"} for i in range(1, 8)}
        client = _local_collection(points)
        with patch.object(_ims, "_SCROLL_PAGE_SIZE", 2):
            removed = _ims._delete_orphaned_chunks(client, "col", self.repo)
        self.assertEqual(len(removed), 7)
        self.assertEqual(_ids(client), [])

    def test_deletes_in_batches(self):
        points = {i: {"file_path": f"gone{i}.py", "type": "code"} for i in range(1, 8)}
        client = _local_collection(points)
        with patch.object(_ims, "DELETE_BATCH_SIZE", 3), patch.object(client, "delete", wraps=client.delete) as spy:
            _ims._delete_orphaned_chunks(client, "col", self.repo)
        self.assertEqual(spy.call_count, 3)
        self.assertEqual(_ids(client), [])

    def test_a_collection_with_nothing_orphaned_deletes_nothing(self):
        (self.repo / "a.py").write_text("x")
        client = _local_collection({1: {"file_path": "a.py", "type": "code"}})
        with patch.object(client, "delete", wraps=client.delete) as spy:
            self.assertEqual(_ims._delete_orphaned_chunks(client, "col", self.repo), [])
        spy.assert_not_called()


# ---------------------------------------------------------------------------
# The versioned manifest on disk.
# ---------------------------------------------------------------------------


class ManifestOnDisk(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        self.path = self.repo / MANIFEST_FILENAME

    def test_save_writes_the_versioned_wrapper_and_load_reads_it_back(self):
        _ims._save_manifest(self.repo, {"a.py": "h1"})
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(doc["files"], {"a.py": "h1"})
        self.assertIn("manifest_format", doc)
        self.assertEqual(_ims._load_manifest(self.repo), {"a.py": "h1"})

    def test_an_old_flat_manifest_is_ignored_so_every_file_counts_as_changed(self):
        self.path.write_text(json.dumps({"a.py": "h1", "b.py": "h2"}), encoding="utf-8")
        self.assertEqual(_ims._load_manifest(self.repo), {})

    def test_absent_corrupt_and_foreign_manifests_load_as_empty(self):
        self.assertEqual(_ims._load_manifest(self.repo), {})
        for text in ("not json", "[1]", json.dumps({"manifest_format": 99, "files": {"a.py": "h"}})):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                self.assertEqual(_ims._load_manifest(self.repo), {})
        self.path.write_bytes(b"\xff\xfe\x00")
        self.assertEqual(_ims._load_manifest(self.repo), {})


# ---------------------------------------------------------------------------
# sync_repo
# ---------------------------------------------------------------------------


class SyncRepoTokenBudget(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        (self.repo / "big.txt").write_text(_words(100))
        (self.repo / "small.txt").write_text("a few words here")
        self.events: list = []

        self.client = MagicMock()
        self.client.collection_exists.return_value = True
        self.client.count.return_value = MagicMock(count=5)
        self.client.delete.side_effect = lambda **kw: self.events.append("delete")

        def _store(connector, entries, **kw):
            self.events.append("store")
            self.stored = list(entries)
            return len(entries)

        self.store = AsyncMock(side_effect=_store)
        self.warn = MagicMock(side_effect=lambda *a: self.events.append("warn") or " WARN-TEXT")
        self.orphans = MagicMock(side_effect=lambda *a: self.events.append("orphans") or ["gone.py"])
        for target, kwargs in (
            ("store_batch", {"new": self.store}),
            ("QdrantConnector", {}),
            ("ensure_file_path_index", {}),
            ("check_embedding_model_mismatch", {"return_value": None}),
            ("_recompute_search_limit", {}),
            ("_non_chunk_points_warning", {"new": self.warn}),
            ("_delete_orphaned_chunks", {"new": self.orphans}),
        ):
            patcher = patch.object(_ims, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _sync(self, provider_cls=None, ctx=None, client=None, **kwargs):
        provider_cls = provider_cls or _provider_cls()
        with patch.object(_ims, "QdrantClient", return_value=client or self.client), \
             patch.object(_ims, "FastEmbedProvider", new=provider_cls):
            return _run(_ims.sync_repo(
                repo_path=str(self.repo), collection="c", scope="docs", include_extensions=".txt", ctx=ctx, **kwargs,
            ))

    def test_chunks_carry_the_new_metadata_and_are_cut_to_the_window(self):
        result = self._sync()
        self.assertIn("Synced 'c': 2 file(s) re-indexed (6 chunks, each cut to the model's 20-token window)", result)
        by_file = {}
        for entry in self.stored:
            by_file.setdefault(entry.metadata["file_path"], []).append(entry)
            self.assertEqual(set(entry.metadata), {"file_path", "line_number", "char_range", "type"})
            self.assertLessEqual(len(entry.content.split()), 20)
        self.assertEqual({k: len(v) for k, v in by_file.items()}, {"big.txt": 5, "small.txt": 1})

    def test_the_result_names_the_largest_files(self):
        self.assertIn("Largest files by chunk count: big.txt (5), small.txt (1).", self._sync())

    def test_progress_names_each_file_and_its_chunk_count(self):
        # The order files are chunked in is the order the repo scan yields them
        # (a directory walk), which differs between filesystems: CI chunked
        # small.txt first, a laptop big.txt first. So the cumulative "so far"
        # figures are asserted as a running total over whatever order was used,
        # never as fixed numbers per file. The per-file counts (5 and 1) do not
        # depend on the order or on the model: the fixed 20-word budget in
        # setUp's _provider_cls() cuts big.txt's 100 words into 5 chunks.
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        self._sync(ctx=ctx)
        messages = [c.kwargs["message"] for c in ctx.report_progress.call_args_list if "message" in c.kwargs]
        done = [re.fullmatch(r"Chunked (\S+): (\d+) chunk\(s\) \((\d+) so far\)", m) for m in messages]
        done = [m for m in done if m]
        self.assertEqual(sorted(m.group(1) for m in done), ["big.txt", "small.txt"], messages)
        expected_counts = {"big.txt": 5, "small.txt": 1}
        running = 0
        for m in done:
            running += int(m.group(2))
            self.assertEqual(int(m.group(2)), expected_counts[m.group(1)], messages)
            self.assertEqual(int(m.group(3)), running, messages)
        self.assertEqual(running, 6)
        starting = [m for m in messages if m.startswith("Chunking ")]
        self.assertEqual(len(starting), 2, messages)
        self.assertTrue(starting[0].endswith(", 0 chunks so far)"), messages)
        self.assertTrue(any(m.startswith("Chunking big.txt (") for m in messages), messages)

    # -- the hash phase reports too ------------------------------

    def test_hashing_reports_a_running_count_before_anything_is_chunked(self):
        # Only counts are asserted, never which file was hashed when: the scan
        # order is the filesystem's (it differs between macOS and the CI runner).
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        with _hashing_reports_every_file():
            self._sync(ctx=ctx)
        messages = _progress_messages(ctx)
        counts = sorted(int(m.split(":")[1].split()[0]) for m in messages if m.startswith("Hashing files:"))
        self.assertEqual(counts, [1, 2], messages)
        self.assertEqual(messages[0], "Hashing files to find what changed")
        self.assertIn("Hashed 2 file(s)", messages)
        last_hash_message = max(i for i, m in enumerate(messages) if m.startswith("Hash"))
        first_chunk_message = min(i for i, m in enumerate(messages) if m.startswith("Chunking "))
        self.assertLess(last_hash_message, first_chunk_message, messages)
        for call in ctx.report_progress.call_args_list:
            if call.kwargs["message"].startswith("Hash"):
                self.assertIsNone(call.kwargs["total"], "the scan is lazy, so no total is known")

    def test_hashing_without_a_context_reports_nothing_and_passes_no_callback(self):
        with patch.object(_ims, "compute_file_hashes", wraps=_ims.compute_file_hashes) as hashes:
            result = self._sync(ctx=None)
        self.assertIn("Synced 'c'", result)
        self.assertNotIn("on_progress", hashes.call_args.kwargs)

    def test_a_failing_report_stops_the_hash_scan_and_nothing_is_chunked(self):
        ctx = MagicMock()

        async def report(**kw):
            if kw["message"].startswith("Hashing files:"):
                raise RuntimeError("client went away")

        ctx.report_progress = AsyncMock(side_effect=report)
        with _hashing_reports_every_file(), patch.object(_ims, "chunk_file") as chunk_file:
            with self.assertRaises(RuntimeError):
                self._sync(ctx=ctx)
        chunk_file.assert_not_called()
        self.assertEqual(len([m for m in _progress_messages(ctx) if m.startswith("Hashing files:")]), 1)
        self.assertFalse((self.repo / MANIFEST_FILENAME).exists())

    def test_an_unreadable_window_fails_before_anything_is_chunked_deleted_or_written(self):
        with patch.object(_ims, "chunk_file") as chunk_file:
            result = self._sync(provider_cls=_provider_cls(window_error="Cannot read the token window of 'x/y'"))
        self.assertTrue(result.startswith("Error: Cannot read the token window"), result)
        chunk_file.assert_not_called()
        self.client.delete.assert_not_called()
        self.store.assert_not_called()
        self.assertFalse((self.repo / MANIFEST_FILENAME).exists(), "no manifest may be written for a failed sync")

    def test_the_no_changes_path_never_loads_the_model(self):
        _ims._save_manifest(self.repo, {
            "big.txt": _sha(self.repo / "big.txt"), "small.txt": _sha(self.repo / "small.txt"),
        })
        provider_cls = _provider_cls()
        result = self._sync(provider_cls=provider_cls)
        self.assertIn("No changes since last sync", result)
        provider_cls.assert_not_called()

    def test_non_default_chunk_lines_is_reported_as_ignored(self):
        self.assertIn("chunk_lines=7 was ignored", self._sync(chunk_lines=7))
        self.assertNotIn("was ignored", self._sync())

    def test_overlap_is_passed_to_the_budget_as_tokens(self):
        provider_cls = _provider_cls()
        self._sync(provider_cls=provider_cls, overlap=60)
        provider_cls.return_value.build_token_budget.assert_called_once_with(60)

    def test_the_saved_manifest_is_versioned(self):
        self._sync()
        doc = json.loads((self.repo / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(set(doc["files"]), {"big.txt", "small.txt"})
        self.assertIn("manifest_format", doc)

    # -- rebuild ------------------------------------------------------------

    def test_a_missing_manifest_against_an_existing_collection_is_a_rebuild(self):
        result = self._sync()
        self.assertIn(" WARN-TEXT", result)
        self.assertIn("Rebuild also removed the chunks of 1 file(s) no longer on disk: gone.py.", result)
        # Warn before the per-file deletes; sweep orphans after the new chunks are stored.
        self.assertEqual(self.events, ["warn", "delete", "store", "orphans"])

    def test_an_old_format_manifest_is_a_rebuild_too_and_is_replaced(self):
        (self.repo / MANIFEST_FILENAME).write_text(json.dumps({"big.txt": "x", "small.txt": "y"}), encoding="utf-8")
        result = self._sync()
        self.assertIn("2 file(s) re-indexed", result)
        self.assertEqual(self.events, ["warn", "delete", "store", "orphans"])
        self.assertIn("manifest_format", json.loads((self.repo / MANIFEST_FILENAME).read_text(encoding="utf-8")))

    def test_a_normal_sync_neither_warns_nor_sweeps(self):
        _ims._save_manifest(self.repo, {"big.txt": "stale", "small.txt": _sha(self.repo / "small.txt")})
        result = self._sync()
        self.assertIn("1 file(s) re-indexed", result)
        self.warn.assert_not_called()
        self.orphans.assert_not_called()
        self.assertNotIn("Rebuild", result)

    def test_a_first_sync_into_a_new_collection_is_not_a_rebuild(self):
        self.client.collection_exists.return_value = False
        result = self._sync()
        self.assertIn("Synced", result)
        self.warn.assert_not_called()
        self.orphans.assert_not_called()

    def test_a_schema_mismatch_stops_a_rebuild_before_the_warning_and_the_delete(self):
        with patch.object(_ims, "check_embedding_model_mismatch", return_value="Error: embedding model mismatch"):
            result = self._sync()
        self.assertIn("Error", result)
        self.assertEqual(self.events, [])

    # -- the warning is delivered BEFORE the deletes it is about ---------------

    def _ctx(self):
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        ctx.warning = AsyncMock(side_effect=lambda message: self.events.append(("notified", message)))
        return ctx

    def test_the_non_chunk_warning_is_sent_as_a_notification_before_the_delete(self):
        ctx = self._ctx()
        result = self._sync(ctx=ctx)
        ctx.warning.assert_awaited_once_with("WARN-TEXT")
        notified = self.events.index(("notified", "WARN-TEXT"))
        self.assertLess(notified, self.events.index("delete"))
        self.assertLess(notified, self.events.index("store"))
        # ...and the same text is kept in the final result for clients that
        # never render notifications.
        self.assertIn(" WARN-TEXT", result)

    def test_a_failure_to_send_the_notification_does_not_fail_the_sync(self):
        ctx = self._ctx()
        ctx.warning = AsyncMock(side_effect=RuntimeError("client went away"))
        result = self._sync(ctx=ctx)
        self.assertIn("Synced", result)
        self.assertEqual(self.events, ["warn", "delete", "store", "orphans"])

    def test_no_notification_when_there_is_nothing_to_warn_about(self):
        self.warn.side_effect = lambda *a: ""
        ctx = self._ctx()
        self._sync(ctx=ctx)
        ctx.warning.assert_not_awaited()

    def test_a_non_rebuild_never_notifies(self):
        _ims._save_manifest(self.repo, {"big.txt": "stale", "small.txt": _sha(self.repo / "small.txt")})
        ctx = self._ctx()
        self._sync(ctx=ctx)
        ctx.warning.assert_not_awaited()

    # -- files skipped as unreadable during a rebuild ---------------------------

    def _flaky_hash(self, name):
        real_read_bytes = Path.read_bytes

        def read_bytes(path, *a, **kw):
            if path.name == name:
                raise OSError("simulated read failure")
            return real_read_bytes(path, *a, **kw)

        return patch.object(Path, "read_bytes", read_bytes)

    def _saved_files(self):
        return json.loads((self.repo / MANIFEST_FILENAME).read_text(encoding="utf-8"))["files"]

    def test_a_file_skipped_during_a_rebuild_stays_tracked_and_retryable(self):
        """The old manifest was ignored, so a skipped file has no prior entry to
        restore. Dropping it from the new, non-empty manifest meant later syncs
        were no longer rebuilds, so deleting the file before a retry succeeded
        left its preserved chunks searchable forever."""
        (self.repo / "flaky.txt").write_text("some words that were indexed before")
        with self._flaky_hash("flaky.txt"):
            result = self._sync()
        self.assertIn("Skipped 1 unreadable file(s)", result)
        files = self._saved_files()
        self.assertEqual(files["flaky.txt"], _ims.UNVERIFIED_HASH)
        self.assertEqual(set(files), {"big.txt", "small.txt", "flaky.txt"})

    def test_a_chunk_failure_during_a_rebuild_is_tracked_the_same_way(self):
        real_chunk_file = _ims.chunk_file

        def chunk_file(path, rel, *a, **kw):
            if rel == "small.txt":
                raise OSError("simulated chunk failure")
            return real_chunk_file(path, rel, *a, **kw)

        with patch.object(_ims, "chunk_file", side_effect=chunk_file):
            self._sync()
        self.assertEqual(self._saved_files()["small.txt"], _ims.UNVERIFIED_HASH)

    def test_deleting_such_a_file_before_a_retry_still_removes_its_chunks(self):
        (self.repo / "flaky.txt").write_text("some words that were indexed before")
        with self._flaky_hash("flaky.txt"):
            self._sync()
        (self.repo / "flaky.txt").unlink()
        self.client.delete.reset_mock()
        result = self._sync()
        self.assertIn("1 file(s) removed", result)
        deleted = {
            cond.match.value
            for call in self.client.delete.call_args_list
            for cond in call.kwargs["points_selector"].should
        }
        self.assertIn("flaky.txt", deleted)

    def test_a_retry_that_can_read_the_file_reindexes_it(self):
        (self.repo / "flaky.txt").write_text("some words that were indexed before")
        with self._flaky_hash("flaky.txt"):
            self._sync()
        result = self._sync()
        self.assertIn("1 file(s) re-indexed", result)
        self.assertEqual(self._saved_files()["flaky.txt"], _sha(self.repo / "flaky.txt"))

    def test_an_ordinary_sync_still_forgets_a_never_indexed_file_that_fails(self):
        """Outside a rebuild a skipped file with no prior entry was never
        indexed: nothing to preserve or clean up, so nothing to track."""
        _ims._save_manifest(self.repo, {"big.txt": "stale", "small.txt": _sha(self.repo / "small.txt")})
        (self.repo / "fresh.txt").write_text("brand new words")
        with self._flaky_hash("fresh.txt"):
            self._sync()
        self.assertNotIn("fresh.txt", self._saved_files())

    def test_an_unreadable_file_in_an_empty_rebuild_stays_tracked_too(self):
        self._empty_repo()
        (self.repo / "flaky.txt").write_text("words")
        with self._flaky_hash("flaky.txt"):
            result = self._sync()
        self.assertIn("No changes since last sync", result)
        self.assertEqual(self._saved_files(), {"flaky.txt": _ims.UNVERIFIED_HASH})

    # -- a rebuild of a repo with nothing left to index ------------------------

    def _empty_repo(self):
        (self.repo / "big.txt").unlink()
        (self.repo / "small.txt").unlink()

    def test_an_old_format_manifest_whose_files_are_all_gone_still_sweeps_orphans(self):
        """Every file the old manifest tracked was deleted: no file is changed
        and none is "removed" (the old manifest is ignored), so the early
        no-changes return used to skip the sweep, on this sync and every later
        one, leaving the deleted files searchable forever."""
        self._empty_repo()
        (self.repo / MANIFEST_FILENAME).write_text(json.dumps({"big.txt": "x", "small.txt": "y"}), encoding="utf-8")
        result = self._sync()
        self.assertIn("No changes since last sync", result)
        self.assertIn("Rebuild also removed the chunks of 1 file(s) no longer on disk: gone.py.", result)
        self.orphans.assert_called_once()
        self.store.assert_not_called()
        # Replaced by a versioned manifest, so this does not repeat and `upgrade`
        # has nothing left to offer.
        doc = json.loads((self.repo / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(doc["files"], {})
        self.assertIn("manifest_format", doc)

    def test_that_sweep_keeps_the_pre_delete_schema_check(self):
        self._empty_repo()
        with patch.object(_ims, "check_embedding_model_mismatch", return_value="Error: embedding model mismatch"):
            result = self._sync()
        self.assertIn("Error", result)
        self.orphans.assert_not_called()

    def test_that_sweep_needs_a_readable_token_window_before_it_deletes_anything(self):
        """The documented contract: an unreadable window fails the run before
        anything is deleted or written, even when nothing would be chunked."""
        self._empty_repo()
        (self.repo / MANIFEST_FILENAME).write_text(json.dumps({"big.txt": "x"}), encoding="utf-8")
        result = self._sync(provider_cls=_provider_cls(window_error="Cannot read the token window of 'x/y'"))
        self.assertTrue(result.startswith("Error: Cannot read the token window"), result)
        self.orphans.assert_not_called()
        self.client.delete.assert_not_called()
        # The old-format manifest is untouched: no new one was written either.
        self.assertEqual(json.loads((self.repo / MANIFEST_FILENAME).read_text(encoding="utf-8")), {"big.txt": "x"})

    def test_that_sweep_checks_the_schema_fail_closed(self):
        self._empty_repo()
        with patch.object(_ims, "check_embedding_model_mismatch", return_value=None) as check:
            self._sync()
        self.assertTrue(check.call_args.kwargs["fail_closed"])

    def test_an_empty_repo_against_a_missing_collection_has_nothing_to_sweep(self):
        self._empty_repo()
        self.client.collection_exists.return_value = False
        result = self._sync()
        self.assertIn("No changes since last sync", result)
        self.orphans.assert_not_called()

    def test_an_ordinary_no_changes_sync_never_sweeps_or_loads_the_model(self):
        _ims._save_manifest(self.repo, {
            "big.txt": _sha(self.repo / "big.txt"), "small.txt": _sha(self.repo / "small.txt"),
        })
        provider_cls = _provider_cls()
        result = self._sync(provider_cls=provider_cls)
        self.assertIn("No changes since last sync", result)
        self.orphans.assert_not_called()
        provider_cls.assert_not_called()

    def test_a_rebuild_with_nothing_to_report_adds_no_notes(self):
        self.warn.side_effect = lambda *a: ""
        self.orphans.side_effect = lambda *a: []
        result = self._sync()
        self.assertNotIn("Rebuild also removed", result)
        self.assertNotIn("Warning", result)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# index_repo
# ---------------------------------------------------------------------------


class IndexBaseline(unittest.TestCase):
    """_index_baseline: which {file: hash} entries index_repo records."""

    def test_every_hashed_file_keeps_its_hash(self):
        self.assertEqual(_ims._index_baseline({"a": "h1", "b": "h2"}, [], {"a": 1, "b": 2}), {"a": "h1", "b": "h2"})

    def test_a_chunk_failure_is_left_out(self):
        self.assertEqual(_ims._index_baseline({"a": "h1", "b": "h2"}, [("b", "boom")], {"a": 1}), {"a": "h1"})

    def test_a_chunked_file_without_a_hash_is_unverified_whatever_the_reason(self):
        baseline = _ims._index_baseline({"a": "h1"}, [], {"a": 1, "unhashable": 2, "late": 1})
        self.assertEqual(baseline, {"a": "h1", "unhashable": _ims.UNVERIFIED_HASH, "late": _ims.UNVERIFIED_HASH})

    def test_a_hashed_file_without_chunks_keeps_its_hash(self):
        self.assertEqual(_ims._index_baseline({"empty": "h"}, [], {}), {"empty": "h"})

    def test_a_hashed_file_that_should_have_had_chunks_is_unverified(self):
        baseline = _ims._index_baseline({"a": "h1", "missed": "h2"}, [], {"a": 1}, missed={"missed"})
        self.assertEqual(baseline, {"a": "h1", "missed": _ims.UNVERIFIED_HASH})

    def test_missed_never_adds_a_file_that_was_not_hashed(self):
        self.assertEqual(_ims._index_baseline({"a": "h1"}, [], {"a": 1}, missed={"ghost"}), {"a": "h1"})

    def test_it_does_not_modify_the_hashes_it_was_given(self):
        hashes = {"a": "h1"}
        _ims._index_baseline(hashes, [], {"a": 1, "b": 1})
        self.assertEqual(hashes, {"a": "h1"})


class HashProgressCancellation(unittest.TestCase):
    """Cancelling a call while a hash-phase report is
    blocked must free the worker thread, not leave it holding the executor."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        for i in range(10):
            (self.repo / f"f{i}.txt").write_text(f"words {i}")
        patcher = patch.object(_ims, "_REPORT_WAIT_SLICE_SECONDS", 0.01)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cancelling_while_a_report_is_blocked_stops_the_worker_and_the_report(self):
        import threading

        worker_done = threading.Event()
        real = _ims.compute_file_hashes

        def hashes(*a, **kw):
            try:
                return real(*a, progress_every=1, **kw)
            finally:
                worker_done.set()

        reports: list = []
        report_cancelled = []

        async def scenario():
            blocked = asyncio.Event()
            never = asyncio.Event()

            async def report(**kw):
                if not kw["message"].startswith("Hashing files:"):
                    return
                reports.append(kw["message"])
                blocked.set()
                try:
                    await never.wait()  # a stalled transport: this report never completes
                except asyncio.CancelledError:
                    report_cancelled.append(True)
                    raise

            ctx = MagicMock()
            ctx.report_progress = AsyncMock(side_effect=report)
            task = asyncio.create_task(
                _ims._hash_files_with_progress(ctx, self.repo, "docs", include_extensions={".txt"})
            )
            await asyncio.wait_for(blocked.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            try:
                for _ in range(500):
                    if worker_done.is_set():
                        break
                    await asyncio.sleep(0.01)
            finally:
                never.set()  # lets a (buggy) stuck worker end so the test cannot hang the run
            return worker_done.is_set()

        with patch.object(_ims, "compute_file_hashes", side_effect=hashes):
            freed = _run(scenario())
        self.assertTrue(freed, "the hash worker is still blocked on a report after its call was cancelled")
        self.assertEqual(report_cancelled, [True], "the pending report was not cancelled")
        self.assertEqual(len(reports), 1, "the scan carried on reporting after the cancel")

    def test_a_report_that_raises_is_propagated_whatever_the_exception_type(self):
        """A report raising TimeoutError (or any other error) ends the scan with
        that error. It is not an expired wait slice to retry: waiting on the
        pending report used to mistake it for one and spin forever."""
        real = _ims.compute_file_hashes
        for exc in (TimeoutError("report timed out"), RuntimeError("client went away")):
            with self.subTest(exc=type(exc).__name__):
                reports: list = []

                async def report(exc=exc, reports=reports, **kw):
                    if kw["message"].startswith("Hashing files:"):
                        reports.append(kw["message"])
                        raise exc

                async def scenario(report=report):
                    ctx = MagicMock()
                    ctx.report_progress = AsyncMock(side_effect=report)
                    # A hang here is the bug: bound it so it fails instead of stalling the run.
                    return await asyncio.wait_for(
                        _ims._hash_files_with_progress(ctx, self.repo, "docs", include_extensions={".txt"}), 5
                    )

                with patch.object(_ims, "compute_file_hashes", side_effect=lambda *a, **kw: real(*a, progress_every=1, **kw)):
                    with self.assertRaises(type(exc)) as caught:
                        _run(scenario())
                # Compared by value: asyncio re-creates some exception types when it hands them to a thread.
                self.assertEqual(caught.exception.args, exc.args)
                self.assertEqual(len(reports), 1, "the scan carried on after a failed report")

    def test_a_completed_scan_is_unaffected_by_the_stop_signal(self):
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        hashes, skipped = _run(_ims._hash_files_with_progress(ctx, self.repo, "docs", include_extensions={".txt"}))
        self.assertEqual((len(hashes), skipped), (10, []))


class IndexRepoTokenBudget(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        (self.repo / "big.txt").write_text(_words(100))
        (self.repo / "small.txt").write_text("a few words here")
        self.stored: list = []

        async def _store(connector, entries, **kw):
            self.stored.extend(entries)
            return len(entries)

        for target, kwargs in (
            ("store_batch", {"new": AsyncMock(side_effect=_store)}),
            ("QdrantConnector", {}),
            ("ensure_file_path_index", {}),
            ("check_embedding_model_mismatch", {"return_value": None}),
            ("_recompute_search_limit", {}),
        ):
            patcher = patch.object(_ims, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = MagicMock()
        self.client.collection_exists.return_value = False
        # "c", which every test here indexes into, is this project's own
        # collection: the sync manifest belongs to it and nothing else.
        patcher = patch.object(_ims, "DEFAULT_COLLECTION", "c")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _index(self, provider_cls=None, client=None, ctx=None, **kwargs):
        provider_cls = provider_cls or _provider_cls()
        with patch.object(_ims, "QdrantClient", return_value=client or self.client), \
             patch.object(_ims, "FastEmbedProvider", new=provider_cls):
            return _run(_ims.index_repo(
                repo_path=str(self.repo), collection="c", scope="docs", include_extensions=".txt", ctx=ctx, **kwargs,
            ))

    def test_indexes_every_file_cut_to_the_window_and_names_the_largest(self):
        result = self._index()
        self.assertIn("Indexed 6 chunks", result)
        self.assertIn("each cut to the model's 20-token window", result)
        self.assertIn("Largest files by chunk count: big.txt (5), small.txt (1).", result)
        self.assertEqual(len(self.stored), 6)
        for entry in self.stored:
            self.assertIn("line_number", entry.metadata)
            self.assertIn("char_range", entry.metadata)

    def test_there_is_no_per_file_cap_a_huge_file_is_indexed_in_full(self):
        (self.repo / "huge.txt").write_text(_words(20_000))
        result = self._index()
        self.assertIn("huge.txt (1000)", result)
        self.assertEqual(sum(1 for e in self.stored if e.metadata["file_path"] == "huge.txt"), 1000)

    def test_progress_names_the_file_being_chunked_and_its_running_count(self):
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        self._index(ctx=ctx)
        messages = [c.kwargs["message"] for c in ctx.report_progress.call_args_list if "message" in c.kwargs]
        self.assertTrue(any("now in big.txt: 5" in m for m in messages), messages)

    def test_the_baseline_hash_reports_a_running_count(self):
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        with _hashing_reports_every_file():
            self._index(ctx=ctx)
        messages = _progress_messages(ctx)
        counts = sorted(int(m.split(":")[1].split()[0]) for m in messages if m.startswith("Hashing files:"))
        self.assertEqual(counts, [1, 2], messages)
        self.assertIn("Hashed 2 file(s)", messages)

    def test_no_hash_progress_when_no_baseline_is_hashed(self):
        self.client.collection_exists.return_value = True
        self.client.count.return_value = MagicMock(count=7)
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        with _hashing_reports_every_file():
            self._index(ctx=ctx, force=True)
        self.assertFalse([m for m in _progress_messages(ctx) if m.startswith("Hash")])

    def test_an_unreadable_window_on_reset_fails_before_the_delete(self):
        self.client.collection_exists.return_value = True
        result = self._index(provider_cls=_provider_cls(window_error="Cannot read the token window of 'x/y'"), reset=True)
        self.assertTrue(result.startswith("Error: Cannot read the token window"), result)
        self.client.delete.assert_not_called()
        self.assertEqual(self.stored, [])

    def test_an_unreadable_window_without_reset_fails_before_chunking(self):
        with patch.object(_ims, "iter_entries") as iter_entries:
            result = self._index(provider_cls=_provider_cls(window_error="Cannot read the token window of 'x/y'"))
        self.assertTrue(result.startswith("Error:"), result)
        iter_entries.assert_not_called()

    def test_overlap_is_passed_to_the_budget_as_tokens(self):
        provider_cls = _provider_cls()
        self._index(provider_cls=provider_cls, overlap=60, chunk_lines=50)
        provider_cls.return_value.build_token_budget.assert_called_once_with(60)

    def test_non_default_chunk_lines_is_reported_as_ignored(self):
        self.assertIn("chunk_lines=7 was ignored", self._index(chunk_lines=7))
        self.assertNotIn("was ignored", self._index())

    def test_the_budget_reaches_iter_entries(self):
        with patch.object(_ims, "iter_entries", return_value=iter([])) as iter_entries:
            self._index()
        self.assertIsNotNone(iter_entries.call_args.kwargs["budget"])

    def test_docstring_says_what_changed_for_chunk_lines_and_overlap(self):
        doc = _ims.index_repo.__doc__ or ""
        self.assertIn("token window", doc)
        self.assertIn("overlap", doc)
        self.assertIn("ignored", doc)
        self.assertIn("no per-file cap", doc)

    def test_docstring_says_when_the_sync_manifest_is_written(self):
        doc = " ".join((_ims.index_repo.__doc__ or "").split())
        self.assertIn("writes sync_repo's manifest", doc)
        self.assertIn("BEFORE chunking", doc)
        self.assertIn("force=true run into a collection that already held points leaves the manifest untouched", doc)
        self.assertIn("only written (or removed) when this run targets this project's own collection", doc)
        self.assertIn("the reset does not start if it cannot be removed", doc)
        self.assertNotIn("does not write sync_repo's manifest", doc)
        self.assertNotIn("a second time", doc)

    # -- the sync manifest baseline -----------------------------------------------

    def _saved_files(self):
        return json.loads((self.repo / MANIFEST_FILENAME).read_text(encoding="utf-8"))["files"]

    def _populated(self, points=7, reported=...):
        """An existing collection holding `points` points. `reported` is the
        count get_collection gives (default: the same); None is what Qdrant
        returns when it does not report one."""
        self.client.collection_exists.return_value = True
        self.client.get_collection.return_value = MagicMock(points_count=points if reported is ... else reported)
        self.client.count.return_value = MagicMock(count=points)

    def test_a_fresh_index_writes_a_versioned_baseline_of_every_file_hash(self):
        result = self._index()
        self.assertIn("Wrote the sync manifest (2 file(s))", result)
        doc = json.loads((self.repo / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        self.assertIn("manifest_format", doc)
        self.assertEqual(
            doc["files"], {"big.txt": _sha(self.repo / "big.txt"), "small.txt": _sha(self.repo / "small.txt")}
        )

    def test_the_baseline_goes_through_the_one_manifest_writer_sync_repo_uses(self):
        with patch.object(_ims, "_save_manifest", wraps=_ims._save_manifest) as save:
            self._index()
        save.assert_called_once()
        self.assertEqual(set(save.call_args.args[1]), {"big.txt", "small.txt"})

    def test_a_sync_right_after_an_index_has_nothing_to_do_and_loads_no_model(self):
        self._index()
        self.client.collection_exists.return_value = True
        self.client.count.return_value = MagicMock(count=6)
        provider_cls = _provider_cls()
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=provider_cls):
            result = _run(_ims.sync_repo(
                repo_path=str(self.repo), collection="c", scope="docs", include_extensions=".txt",
            ))
        self.assertIn("No changes since last sync. 2 files already up to date", result)
        self.assertNotIn("Rebuild", result)
        provider_cls.assert_not_called()

    def test_a_sync_after_an_index_reembeds_only_what_changed_since(self):
        self._index()
        (self.repo / "small.txt").write_text("different words entirely now")
        self.client.collection_exists.return_value = True
        self.client.count.return_value = MagicMock(count=6)
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            result = _run(_ims.sync_repo(
                repo_path=str(self.repo), collection="c", scope="docs", include_extensions=".txt",
            ))
        self.assertIn("1 file(s) re-indexed", result)
        self.assertIn("1 file(s) unchanged", result)

    def test_a_later_deletion_is_noticed_by_the_next_sync(self):
        """The point of the baseline: files recorded by index_repo are the ones
        sync_repo can later see removed."""
        self._index()
        (self.repo / "small.txt").unlink()
        self.client.collection_exists.return_value = True
        self.client.count.return_value = MagicMock(count=6)
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            result = _run(_ims.sync_repo(
                repo_path=str(self.repo), collection="c", scope="docs", include_extensions=".txt",
            ))
        self.assertIn("1 file(s) removed", result)
        deleted = {
            cond.match.value
            for call in self.client.delete.call_args_list
            for cond in call.kwargs["points_selector"].should
        }
        self.assertEqual(deleted, {"small.txt"})

    def test_hashes_are_taken_before_any_file_is_chunked(self):
        """A file edited while the run is going must end up with a hash OLDER
        than its indexed chunks (so the next sync re-indexes it), never newer."""
        original_hash = _sha(self.repo / "small.txt")
        # index_repo chunks through iter_entries, which calls the library's
        # chunk_file, so that is the one to intercept (sync_repo calls its own
        # imported name instead).
        real_chunk_file = _lib.chunk_file

        def edit_then_chunk(path, rel, *a, **kw):
            if rel == "small.txt":
                path.write_text("edited while the index was running")
            return real_chunk_file(path, rel, *a, **kw)

        with patch.object(_lib, "chunk_file", side_effect=edit_then_chunk):
            self._index()
        self.assertEqual(self._saved_files()["small.txt"], original_hash)
        self.assertNotEqual(self._saved_files()["small.txt"], _sha(self.repo / "small.txt"))

    def test_reset_removes_the_old_manifest_as_soon_as_the_collection_is_wiped(self):
        """A run that dies after the wipe must not leave a manifest claiming
        files are indexed."""
        _ims._save_manifest(self.repo, {"big.txt": _sha(self.repo / "big.txt"), "old.txt": "h"})
        self.client.collection_exists.return_value = True
        self.client.get_collection.return_value.config.params.vectors = {}

        class _Boom(Exception):
            pass

        with patch.object(_ims, "iter_entries", side_effect=_Boom), \
             patch.object(_ims, "check_embedding_model_mismatch", return_value=None):
            with self.assertRaises(_Boom):
                self._index(reset=True)
        self.client.delete.assert_called_once()
        self.assertFalse((self.repo / MANIFEST_FILENAME).exists())

    def test_a_failed_reset_that_deleted_nothing_keeps_the_old_manifest(self):
        _ims._save_manifest(self.repo, {"big.txt": "h"})
        self.client.collection_exists.return_value = True
        with patch.object(_ims, "check_embedding_model_mismatch", return_value="Error: embedding model mismatch"):
            result = self._index(reset=True)
        self.assertIn("Error", result)
        self.client.delete.assert_not_called()
        self.assertEqual(self._saved_files(), {"big.txt": "h"})

    def test_reset_replaces_the_old_manifest_with_the_new_baseline(self):
        _ims._save_manifest(self.repo, {"old.txt": "h"})
        self.client.collection_exists.return_value = True
        result = self._index(reset=True)
        self.assertIn("Wrote the sync manifest (2 file(s))", result)
        self.assertEqual(set(self._saved_files()), {"big.txt", "small.txt"})

    def test_an_interrupted_run_without_reset_writes_no_manifest(self):
        class _Boom(Exception):
            pass

        with patch.object(_ims, "iter_entries", side_effect=_Boom):
            with self.assertRaises(_Boom):
                self._index()
        self.assertFalse((self.repo / MANIFEST_FILENAME).exists())

    def test_a_forced_run_into_a_populated_collection_leaves_the_manifest_alone_and_says_so(self):
        self._populated(points=7)
        _ims._save_manifest(self.repo, {"big.txt": "stale", "old.txt": "h"})
        result = self._index(force=True)
        self.assertIn("Indexed 6 chunks", result)
        self.assertIn("The sync manifest was not written", result)
        self.assertIn("7 point(s)", result)
        self.assertIn("reset=true", result)
        self.assertEqual(self._saved_files(), {"big.txt": "stale", "old.txt": "h"})

    def test_a_forced_run_into_a_populated_collection_writes_none_when_there_was_none(self):
        self._populated(points=7)
        self._index(force=True)
        self.assertFalse((self.repo / MANIFEST_FILENAME).exists())

    def test_an_existing_but_empty_collection_counts_as_a_clean_start(self):
        self._populated(points=0)
        result = self._index()
        self.assertIn("Wrote the sync manifest", result)
        self.assertEqual(set(self._saved_files()), {"big.txt", "small.txt"})
        # An apparent zero is confirmed with an exact count before it is trusted.
        self.client.count.assert_called_once()
        self.assertTrue(self.client.count.call_args.kwargs["exact"])

    def test_an_unreported_point_count_is_not_taken_for_an_empty_collection(self):
        """Qdrant can report points_count as None, which the preparation step
        turns into 0. A populated collection read as empty would lose the
        manifest that tracks its deleted files."""
        self._populated(points=5, reported=None)
        _ims._save_manifest(self.repo, {"deleted.txt": "h", "big.txt": "stale"})
        result = self._index(force=True)
        self.assertIn("The sync manifest was not written", result)
        self.assertIn("5 point(s)", result)
        self.assertEqual(self._saved_files(), {"deleted.txt": "h", "big.txt": "stale"})

    def test_an_unreported_point_count_on_an_empty_collection_still_writes_the_baseline(self):
        self._populated(points=0, reported=None)
        result = self._index()
        self.assertIn("Wrote the sync manifest", result)

    def test_the_exact_count_is_not_asked_for_when_the_manifest_is_not_in_play(self):
        self._populated(points=0)
        self._index_elsewhere()
        self.client.count.assert_not_called()

    def test_a_file_that_fails_to_chunk_is_left_out_so_the_next_sync_retries_it(self):
        real_chunk_file = _lib.chunk_file

        def chunk_file(path, rel, *a, **kw):
            if rel == "small.txt":
                raise OSError("simulated chunk failure")
            return real_chunk_file(path, rel, *a, **kw)

        with patch.object(_lib, "chunk_file", side_effect=chunk_file):
            result = self._index()
        self.assertIn("Skipped", result)
        self.assertEqual(set(self._saved_files()), {"big.txt"})

    def test_a_file_that_cannot_be_hashed_but_was_chunked_stays_tracked_and_retryable(self):
        real_read_bytes = Path.read_bytes

        def read_bytes(path, *a, **kw):
            if path.name == "small.txt":
                raise OSError("simulated read failure")
            return real_read_bytes(path, *a, **kw)

        with patch.object(Path, "read_bytes", read_bytes):
            self._index()
        files = self._saved_files()
        self.assertEqual(files["small.txt"], _ims.UNVERIFIED_HASH)
        self.assertEqual(files["big.txt"], _sha(self.repo / "big.txt"))

    def test_a_file_that_cannot_be_hashed_and_produced_no_chunks_is_left_out(self):
        real_read_bytes = Path.read_bytes

        def read_bytes(path, *a, **kw):
            if path.name == "small.txt":
                raise OSError("simulated read failure")
            return real_read_bytes(path, *a, **kw)

        real_chunk_file = _lib.chunk_file

        def chunk_file(path, rel, *a, **kw):
            if rel == "small.txt":
                raise OSError("simulated chunk failure")
            return real_chunk_file(path, rel, *a, **kw)

        with patch.object(Path, "read_bytes", read_bytes), patch.object(_lib, "chunk_file", side_effect=chunk_file):
            self._index()
        self.assertEqual(set(self._saved_files()), {"big.txt"})

    def test_the_manifest_file_never_indexes_or_hashes_itself(self):
        # An extension filter that would match the manifest's own .json suffix.
        _ims._save_manifest(self.repo, {"x.json": "h"})
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            result = _run(_ims.index_repo(
                repo_path=str(self.repo), collection="c", include_extensions=".txt,.json",
            ))
        self.assertNotIn(MANIFEST_FILENAME, self._saved_files())
        self.assertIn("Indexed 6 chunks", result)

    def test_an_unwritable_manifest_is_reported_not_raised(self):
        with patch.object(_ims, "_save_manifest", side_effect=OSError("read-only file system")):
            result = self._index()
        self.assertIn("Indexed 6 chunks", result)
        self.assertIn("could not write the sync manifest (read-only file system)", result)

    def test_a_reset_that_cannot_remove_the_old_manifest_does_not_start(self):
        """Carrying on would leave a manifest claiming files are indexed that the
        wipe removed, and a failure after it would never replace it: the next
        sync then reported "No changes" over a partly filled collection."""
        _ims._save_manifest(self.repo, {"big.txt": _sha(self.repo / "big.txt"), "old.txt": "h"})
        self.client.collection_exists.return_value = True
        with patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            result = self._index(reset=True)
        self.assertTrue(result.startswith("Error: could not remove the old sync manifest"), result)
        self.assertIn("permission denied", result)
        self.assertIn("Nothing was deleted or indexed", result)
        self.client.delete.assert_not_called()
        self.assertEqual(self.stored, [])
        self.assertEqual(self._saved_files()["old.txt"], "h")

    def test_the_manifest_is_invalidated_between_the_model_check_and_the_delete(self):
        events: list = []
        self.client.collection_exists.return_value = True
        self.client.delete.side_effect = lambda **kw: events.append("delete")
        _ims._save_manifest(self.repo, {"old.txt": "h"})
        real_discard = _ims._discard_manifest

        def discard(repo):
            events.append("discard")
            real_discard(repo)

        with patch.object(_ims, "_discard_manifest", side_effect=discard), \
             patch.object(_ims, "check_embedding_model_mismatch", side_effect=lambda *a, **kw: events.append("check")):
            self._index(reset=True)
        self.assertEqual(events[:3], ["check", "discard", "delete"])

    def test_a_clean_start_that_cannot_remove_the_old_manifest_does_not_start_either(self):
        _ims._save_manifest(self.repo, {"old.txt": "h"})
        with patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            result = self._index()
        self.assertTrue(result.startswith("Error: could not remove the old sync manifest"), result)
        self.assertEqual(self.stored, [])

    def test_a_failed_run_into_an_empty_collection_leaves_no_stale_manifest(self):
        """Nothing to wipe, but the old manifest is no longer true once chunks
        land: a run that dies partway must not leave it for sync_repo to trust."""
        _ims._save_manifest(self.repo, {
            "big.txt": _sha(self.repo / "big.txt"), "small.txt": _sha(self.repo / "small.txt"),
        })

        class _Boom(Exception):
            pass

        with patch.object(_ims, "iter_entries", side_effect=_Boom):
            with self.assertRaises(_Boom):
                self._index()
        self.assertFalse((self.repo / MANIFEST_FILENAME).exists())

    def test_a_missing_manifest_is_not_an_error_for_the_invalidation(self):
        result = self._index()
        self.assertIn("Indexed 6 chunks", result)

    # -- only this project's own collection owns the manifest ------------------------

    def _index_elsewhere(self, **kwargs):
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            return _run(_ims.index_repo(
                repo_path=str(self.repo), collection="another", scope="docs", include_extensions=".txt", **kwargs,
            ))

    def test_indexing_another_collection_never_replaces_this_projects_baseline(self):
        """The manifest is one file per checkout and records no collection:
        indexing the checkout into collection B must not overwrite what
        sync_repo keeps for collection A, or A's next sync reports no changes
        for files edited since."""
        self._index()
        before = self._saved_files()
        (self.repo / "small.txt").write_text("edited after the baseline was written")
        result = self._index_elsewhere()
        self.assertIn("Indexed 6 chunks", result)
        self.assertIn("The sync manifest was not written", result)
        self.assertIn("belongs to this project's own collection", result)
        self.assertEqual(self._saved_files(), before)

    def test_indexing_another_collection_writes_no_manifest_when_there_is_none(self):
        self._index_elsewhere()
        self.assertFalse((self.repo / MANIFEST_FILENAME).exists())

    def test_a_reset_of_another_collection_does_not_remove_this_projects_manifest(self):
        self._index()
        before = self._saved_files()
        self.client.collection_exists.return_value = True
        self._index_elsewhere(reset=True)
        self.assertEqual(self._saved_files(), before)

    def test_another_qdrant_instance_does_not_own_the_manifest_either(self):
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            _run(_ims.index_repo(
                repo_path=str(self.repo), collection="c", qdrant_url="http://other-host:6333",
                scope="docs", include_extensions=".txt",
            ))
        self.assertFalse((self.repo / MANIFEST_FILENAME).exists())

    def _without_chunks_for(self, name):
        """Run index_repo as if the second tree walk never reached `name`, which
        is how a file deleted at that moment and restored afterwards looks."""
        real_iter_entries = _ims.iter_entries

        def entries(*a, **kw):
            for content, md in real_iter_entries(*a, **kw):
                if not (isinstance(md, dict) and md.get("file_path") == name):
                    yield content, md

        return patch.object(_ims, "iter_entries", side_effect=entries)

    def test_a_file_the_chunking_walk_never_reached_is_unverified_not_up_to_date(self):
        """It was hashed, so its hash would make the next sync call it unchanged
        although it was never indexed, and the other files' chunks satisfy
        sync's check that the collection holds something."""
        with self._without_chunks_for("small.txt"):
            self._index()
        files = self._saved_files()
        self.assertEqual(files["small.txt"], _ims.UNVERIFIED_HASH)
        self.assertEqual(files["big.txt"], _sha(self.repo / "big.txt"))

    def test_the_next_sync_indexes_that_missed_file(self):
        with self._without_chunks_for("small.txt"):
            self._index()
        self.client.collection_exists.return_value = True
        self.client.count.return_value = MagicMock(count=5)
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            result = _run(_ims.sync_repo(
                repo_path=str(self.repo), collection="c", scope="docs", include_extensions=".txt",
            ))
        self.assertIn("1 file(s) re-indexed", result)

    def test_a_file_with_nothing_to_index_keeps_its_hash_and_is_not_retried_forever(self):
        (self.repo / "empty.txt").write_text("")
        (self.repo / "blank.txt").write_text("  \n\n ")
        self._index()
        files = self._saved_files()
        self.assertEqual(files["empty.txt"], _sha(self.repo / "empty.txt"))
        self.assertEqual(files["blank.txt"], _sha(self.repo / "blank.txt"))
        self.client.collection_exists.return_value = True
        self.client.count.return_value = MagicMock(count=6)
        provider_cls = _provider_cls()
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=provider_cls):
            result = _run(_ims.sync_repo(
                repo_path=str(self.repo), collection="c", scope="docs", include_extensions=".txt",
            ))
        self.assertIn("No changes since last sync", result)
        provider_cls.assert_not_called()

    def test_a_hashed_file_that_cannot_be_read_afterwards_and_has_no_chunks_is_unverified(self):
        real_has_content = _ims.file_has_indexable_content

        def has_content(path):
            if path.name == "small.txt":
                raise OSError("gone")
            return real_has_content(path)

        with self._without_chunks_for("small.txt"), \
             patch.object(_ims, "file_has_indexable_content", side_effect=has_content):
            self._index()
        self.assertEqual(self._saved_files()["small.txt"], _ims.UNVERIFIED_HASH)

    def test_a_file_created_after_the_hash_scan_is_tracked_as_unverified(self):
        """Chunking walks the tree again, so it can store chunks for a file the
        hashing never saw. Leaving it out of a non-empty manifest meant deleting
        it before the next sync left its chunks searchable forever."""
        real_hashes = _ims.compute_file_hashes

        def hash_then_create(*a, **kw):
            result = real_hashes(*a, **kw)
            (self.repo / "late.txt").write_text("created after the hash scan")
            return result

        with patch.object(_ims, "compute_file_hashes", side_effect=hash_then_create):
            self._index()
        files = self._saved_files()
        self.assertEqual(files["late.txt"], _ims.UNVERIFIED_HASH)
        self.assertEqual(files["big.txt"], _sha(self.repo / "big.txt"))

    def test_deleting_that_late_file_before_the_next_sync_removes_its_chunks(self):
        real_hashes = _ims.compute_file_hashes

        def hash_then_create(*a, **kw):
            result = real_hashes(*a, **kw)
            (self.repo / "late.txt").write_text("created after the hash scan")
            return result

        with patch.object(_ims, "compute_file_hashes", side_effect=hash_then_create):
            self._index()
        (self.repo / "late.txt").unlink()
        self.client.collection_exists.return_value = True
        self.client.count.return_value = MagicMock(count=7)
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            result = _run(_ims.sync_repo(
                repo_path=str(self.repo), collection="c", scope="docs", include_extensions=".txt",
            ))
        self.assertIn("1 file(s) removed", result)
        deleted = {
            cond.match.value
            for call in self.client.delete.call_args_list
            for cond in call.kwargs["points_selector"].should
        }
        self.assertEqual(deleted, {"late.txt"})

    def test_nothing_is_hashed_when_no_baseline_will_be_written(self):
        self._populated(points=7)
        with patch.object(_ims, "compute_file_hashes", wraps=_ims.compute_file_hashes) as hashes:
            self._index(force=True)
        hashes.assert_not_called()

    def test_the_baseline_is_scoped_by_the_same_filters_as_the_run(self):
        (self.repo / "skipme").mkdir()
        (self.repo / "skipme" / "inner.txt").write_text("words in an excluded directory")
        self._index(exclude_dirs="skipme")
        self.assertEqual(set(self._saved_files()), {"big.txt", "small.txt"})


# ---------------------------------------------------------------------------
# preview_index
# ---------------------------------------------------------------------------


class PreviewIndexTokenBudget(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        (self.repo / "big.txt").write_text(_words(100))
        (self.repo / "small.txt").write_text("a few words here")

    def _preview(self, provider_cls=None, **kwargs):
        with patch.object(_ims, "FastEmbedProvider", new=provider_cls or _provider_cls()):
            return _ims.preview_index(repo_path=str(self.repo), scope="docs", include_extensions=".txt", **kwargs)

    def test_shows_line_number_and_char_range_and_the_largest_files(self):
        text = self._preview(limit=2)
        self.assertIn("6 total chunks from 2 file(s)", text)
        self.assertIn("20-token window", text)
        self.assertIn("(line 1, chars 0-", text)
        self.assertIn("Largest files by chunk count: big.txt (5), small.txt (1).", text)

    def test_loads_the_model_for_its_window_and_reports_an_unreadable_one(self):
        result = self._preview(provider_cls=_provider_cls(window_error="Cannot read the token window of 'x/y'"))
        self.assertTrue(result.startswith("Error: Cannot read the token window"), result)

    def test_uses_the_given_embedding_model(self):
        provider_cls = _provider_cls()
        self._preview(provider_cls=provider_cls, embedding_model="other/model")
        provider_cls.assert_called_once_with("other/model")

    def test_reads_points_that_still_carry_line_range_without_a_key_error(self):
        old_shape = [("some text", {"type": "code", "file_path": "old.py", "line_range": "10-40"})]
        with patch.object(_ims, "build_entries", return_value=(old_shape, [])):
            text = self._preview()
        self.assertIn("[code] old.py (lines 10-40): some text", text)

    def test_a_point_with_neither_position_shape_does_not_crash(self):
        bare = [("some text", {"type": "doc", "file_path": "x.md"})]
        with patch.object(_ims, "build_entries", return_value=(bare, [])):
            self.assertIn("[doc] x.md (): some text", self._preview())


class IndexingToolsKeepTheirSignatures(unittest.TestCase):
    """chunk_lines stays on every tool so existing calls keep working."""

    def test_chunk_lines_and_overlap_are_still_parameters(self):
        import inspect

        for tool in (_ims.index_repo, _ims.sync_repo, _ims.preview_index):
            params = inspect.signature(tool).parameters
            self.assertEqual(params["chunk_lines"].default, 50)
            self.assertEqual(params["overlap"].default, 10)
        self.assertEqual(inspect.signature(_ims.preview_index).parameters["limit"].default, 10)


if __name__ == "__main__":
    unittest.main()
