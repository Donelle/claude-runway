#!/usr/bin/env python3
"""Tests for the memory-bank guards added to tools/ingest_mcp_server.py
(issue #175):

- index_repo/sync_repo refuse outright when their target collection matches
  the configured shared memory-bank collection name.
- index_repo(reset=True) ALWAYS preserves memory-bank points via a filtered
  delete, and never calls delete_collection or counts anything first (PR
  #178 review: an earlier version branched on a memory-bank point count, but
  that count and the delete were separate Qdrant requests with no
  transaction between them -- a concurrent remember() landing in the gap
  could still be destroyed by a "zero-count" full delete. Always using the
  filtered delete removes the race instead of narrowing it).
- sync_repo's per-batch delete filter carries a must_not exclusion for
  memory-bank points.

Stdlib-only (unittest, no pytest), no network -- mirrors
test_ingest_mcp_server_preflight.py's mocking style.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "test-collection")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

import ingest_mcp_server as _ims  # noqa: E402
import memory_bank_lib as mb  # noqa: E402
from qdrant_client.http.models import models as _m  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class ReservedCollectionNameGuardTest(unittest.TestCase):
    """No client/network mocking needed -- this guard fires before any
    QdrantClient is even constructed."""

    def test_index_repo_refuses_memory_bank_collection(self):
        result = _run(_ims.index_repo(repo_path=REPO_ROOT, collection=_ims.DEFAULT_MEMORY_BANK_COLLECTION))
        self.assertIn("Error", result)
        self.assertIn("memory-bank", result)

    def test_index_repo_refuses_regardless_of_reset_and_force(self):
        result = _run(
            _ims.index_repo(
                repo_path=REPO_ROOT, collection=_ims.DEFAULT_MEMORY_BANK_COLLECTION, reset=True, force=True
            )
        )
        self.assertIn("Error", result)

    def test_sync_repo_refuses_memory_bank_collection(self):
        result = _run(_ims.sync_repo(repo_path=REPO_ROOT, collection=_ims.DEFAULT_MEMORY_BANK_COLLECTION))
        self.assertIn("Error", result)
        self.assertIn("memory-bank", result)


class IndexRepoResetGuardTest(unittest.TestCase):
    def setUp(self):
        self._retry_patcher = patch.object(_ims, "call_with_retry", side_effect=lambda fn, *a, **kw: fn(*a, **kw))
        self._retry_patcher.start()
        self.addCleanup(self._retry_patcher.stop)

        self._store_batch_patcher = patch.object(_ims, "store_batch", new_callable=AsyncMock)
        self._store_batch_patcher.start()
        self.addCleanup(self._store_batch_patcher.stop)

        self._iter_entries_patcher = patch.object(_ims, "iter_entries", return_value=iter([]))
        self._iter_entries_patcher.start()
        self.addCleanup(self._iter_entries_patcher.stop)

        self._embed_patcher = patch("ingest_mcp_server.FastEmbedProvider")
        self._embed_patcher.start()
        self.addCleanup(self._embed_patcher.stop)

        self._connector_patcher = patch("ingest_mcp_server.QdrantConnector")
        self._connector_patcher.start()
        self.addCleanup(self._connector_patcher.stop)

        self._index_patcher = patch.object(_ims, "ensure_file_path_index")
        self._index_patcher.start()
        self.addCleanup(self._index_patcher.stop)

    def _make_client(self, exists=True):
        client = MagicMock()
        client.collection_exists.return_value = exists
        return client

    def test_reset_always_takes_the_filtered_delete_path(self):
        # PR #178 review: an earlier version counted memory-bank points first
        # and only used the filtered (preserving) delete when that count was
        # nonzero -- but count-then-act is racy against a concurrent
        # remember() call landing in between. Fixed by ALWAYS using the
        # filtered delete regardless of count, which removes the race
        # entirely instead of narrowing it. This test pins that: no count
        # check happens at all, and delete_collection is never called.
        client = self._make_client()
        # get_collection must return a vectors_config the mismatch check
        # actually recognizes as compatible (PR #178 review, seventh pass):
        # an unconfigured MagicMock is neither None, dict, nor VectorParams,
        # which now correctly fails closed rather than silently passing.
        info = MagicMock()
        info.config.params.vectors = {"fast-some-model": _m.VectorParams(size=384, distance=_m.Distance.COSINE)}
        client.get_collection.return_value = info
        provider_instance = MagicMock()
        provider_instance.get_vector_name.return_value = "fast-some-model"
        provider_instance.get_vector_size.return_value = 384
        with patch("ingest_mcp_server.QdrantClient", return_value=client), \
             patch("ingest_mcp_server.FastEmbedProvider", return_value=provider_instance), \
             patch.object(mb, "count_memory_bank_points") as mock_count:
            _run(_ims.index_repo(repo_path=REPO_ROOT, collection="test-collection", reset=True))
        mock_count.assert_not_called()
        client.delete_collection.assert_not_called()
        client.delete.assert_called_once()
        _, kwargs = client.delete.call_args
        self.assertEqual(kwargs["collection_name"], "test-collection")
        must_not = kwargs["points_selector"].must_not
        self.assertEqual(len(must_not), 1)
        self.assertEqual(must_not[0].match.value, mb.MEMORY_BANK_SOURCE)

    def test_collection_not_existing_skips_both_delete_paths(self):
        client = self._make_client(exists=False)
        with patch("ingest_mcp_server.QdrantClient", return_value=client):
            _run(_ims.index_repo(repo_path=REPO_ROOT, collection="test-collection", reset=True))
        client.delete_collection.assert_not_called()
        client.delete.assert_not_called()

    def test_schema_mismatch_blocks_the_destructive_delete(self):
        # Regression for PR #178 review (third pass): without a schema check,
        # reset=True would run the filtered delete FIRST and only discover an
        # incompatible embedding model later, once the embed/store loop's
        # upsert failed -- destroying the existing non-memory-bank index with
        # nothing successfully re-indexed to replace it. Must now check
        # BEFORE deleting anything.
        client = self._make_client()
        info = MagicMock()
        info.config.params.vectors = {}  # empty dict: definitive mismatch (sparse-only)
        client.get_collection.return_value = info
        provider_instance = MagicMock()
        provider_instance.get_vector_name.return_value = "fast-some-model"
        provider_instance.get_vector_size.return_value = 384
        # Nested patch of the same target setUp already patched -- this
        # shadows it for the duration of the `with` block only, and restores
        # setUp's bare MagicMock class automatically on exit (no manual
        # stop/restart of setUp's own patcher needed).
        with patch("ingest_mcp_server.QdrantClient", return_value=client), \
             patch("ingest_mcp_server.FastEmbedProvider", return_value=provider_instance):
            result = _run(_ims.index_repo(repo_path=REPO_ROOT, collection="test-collection", reset=True))
        self.assertIn("Error", result)
        client.delete.assert_not_called()
        client.delete_collection.assert_not_called()

    def test_inconclusive_schema_check_blocks_the_destructive_delete(self):
        # Regression for PR #178 review (fourth pass): the pre-delete check
        # must fail CLOSED, not open -- a transient error verifying schema
        # compatibility (e.g. get_collection failing) must block the delete
        # the same way a confirmed mismatch does, not be silently treated as
        # "compatible" just because no definitive mismatch was ever found.
        client = self._make_client()
        client.get_collection.side_effect = RuntimeError("connection refused")
        with patch("ingest_mcp_server.QdrantClient", return_value=client):
            result = _run(_ims.index_repo(repo_path=REPO_ROOT, collection="test-collection", reset=True))
        self.assertIn("Error", result)
        client.delete.assert_not_called()
        client.delete_collection.assert_not_called()


class SyncRepoDeleteFilterTest(unittest.TestCase):
    def setUp(self):
        self._retry_patcher = patch.object(_ims, "call_with_retry", side_effect=lambda fn, *a, **kw: fn(*a, **kw))
        self._retry_patcher.start()
        self.addCleanup(self._retry_patcher.stop)

        # One changed file, nothing removed -- enough to enter the delete loop.
        self._hashes_patcher = patch.object(
            _ims, "compute_file_hashes", return_value=({"changed_file.py": "newhash"}, [])
        )
        self._hashes_patcher.start()
        self.addCleanup(self._hashes_patcher.stop)

        self._manifest_load_patcher = patch.object(_ims, "_load_manifest", return_value={})
        self._manifest_load_patcher.start()
        self.addCleanup(self._manifest_load_patcher.stop)

        self._manifest_save_patcher = patch.object(_ims, "_save_manifest")
        self._manifest_save_patcher.start()
        self.addCleanup(self._manifest_save_patcher.stop)

        self._chunk_file_patcher = patch.object(_ims, "chunk_file", return_value=iter([]))
        self._chunk_file_patcher.start()
        self.addCleanup(self._chunk_file_patcher.stop)

        self._store_batch_patcher = patch.object(_ims, "store_batch", new_callable=AsyncMock)
        self._store_batch_patcher.start()
        self.addCleanup(self._store_batch_patcher.stop)

        self._embed_patcher = patch("ingest_mcp_server.FastEmbedProvider")
        self._embed_patcher.start()
        self.addCleanup(self._embed_patcher.stop)

        self._connector_patcher = patch("ingest_mcp_server.QdrantConnector")
        self._connector_patcher.start()
        self.addCleanup(self._connector_patcher.stop)

        self._index_patcher = patch.object(_ims, "ensure_file_path_index")
        self._index_patcher.start()
        self.addCleanup(self._index_patcher.stop)

    def test_delete_filter_excludes_memory_bank_points(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        # get_collection must return a vectors_config the mismatch check
        # actually recognizes as compatible (PR #178 review, seventh pass):
        # an unconfigured MagicMock is neither None, dict, nor VectorParams,
        # which now correctly fails closed rather than silently passing.
        info = MagicMock()
        info.config.params.vectors = {"fast-some-model": _m.VectorParams(size=384, distance=_m.Distance.COSINE)}
        client.get_collection.return_value = info
        provider_instance = MagicMock()
        provider_instance.get_vector_name.return_value = "fast-some-model"
        provider_instance.get_vector_size.return_value = 384
        with patch("ingest_mcp_server.QdrantClient", return_value=client), \
             patch("ingest_mcp_server.FastEmbedProvider", return_value=provider_instance):
            _run(_ims.sync_repo(repo_path=REPO_ROOT, collection="test-collection"))
        client.delete.assert_called_once()
        _, kwargs = client.delete.call_args
        must_not = kwargs["points_selector"].must_not
        self.assertEqual(len(must_not), 1)
        self.assertEqual(must_not[0].key, mb.SOURCE_FIELD)
        self.assertEqual(must_not[0].match.value, mb.MEMORY_BANK_SOURCE)
        # The existing should=[file_path...] behavior must be unaffected.
        should = kwargs["points_selector"].should
        self.assertEqual({c.match.value for c in should}, {"changed_file.py"})

    def test_schema_mismatch_blocks_the_delete_loop(self):
        # Regression for PR #178 review (fifth pass): without this check,
        # sync_repo would delete each changed/removed file's old chunks
        # FIRST and only discover an incompatible EMBEDDING_MODEL later,
        # once the embed/store loop's upsert failed -- leaving those files
        # with no searchable content at all, unlike every other skip path
        # in this function (which deliberately preserves existing
        # embeddings on failure).
        client = MagicMock()
        client.collection_exists.return_value = True
        info = MagicMock()
        info.config.params.vectors = {}  # empty dict: definitive mismatch
        client.get_collection.return_value = info
        provider_instance = MagicMock()
        provider_instance.get_vector_name.return_value = "fast-some-model"
        provider_instance.get_vector_size.return_value = 384
        with patch("ingest_mcp_server.QdrantClient", return_value=client), \
             patch("ingest_mcp_server.FastEmbedProvider", return_value=provider_instance):
            result = _run(_ims.sync_repo(repo_path=REPO_ROOT, collection="test-collection"))
        self.assertIn("Error", result)
        client.delete.assert_not_called()

    def test_inconclusive_schema_check_blocks_the_delete_loop(self):
        # fail_closed=True: an inconclusive check (e.g. a transient error)
        # must block the delete the same way a confirmed mismatch does.
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.side_effect = RuntimeError("connection refused")
        with patch("ingest_mcp_server.QdrantClient", return_value=client):
            result = _run(_ims.sync_repo(repo_path=REPO_ROOT, collection="test-collection"))
        self.assertIn("Error", result)
        client.delete.assert_not_called()


class SyncRepoMissingCollectionTest(unittest.TestCase):
    """Issue #303 (bug 2): sync_repo's manifest-diff "no changes" early return
    used to run before any collection_exists check, so a manifest claiming
    every file was indexed masked a dropped/empty collection -- which is
    exactly the state the documented EMBEDDING_MODEL-change remediation
    (manually drop the collection) leaves behind."""

    def setUp(self):
        for target, kwargs in (
            ("call_with_retry", {"side_effect": lambda fn, *a, **kw: fn(*a, **kw)}),
            ("compute_file_hashes", {"return_value": ({"a.py": "h1", "b.py": "h2"}, [])}),
            ("_save_manifest", {}),
            # Every file yields one chunk by default; individual tests override.
            ("chunk_file", {"side_effect": lambda *a, **kw: iter([("content", {"file_path": "x"})])}),
            ("ensure_file_path_index", {}),
        ):
            patcher = patch.object(_ims, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, kwargs in (
            ("ingest_mcp_server.FastEmbedProvider", {}),
            ("ingest_mcp_server.QdrantConnector", {}),
            ("ingest_mcp_server.store_batch", {"new_callable": AsyncMock}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _sync(self, client, manifest):
        with patch.object(_ims, "_load_manifest", return_value=manifest), \
             patch("ingest_mcp_server.QdrantClient", return_value=client):
            return _run(_ims.sync_repo(repo_path=REPO_ROOT, collection="test-collection"))

    def _client(self, exists: bool, points: int = 5) -> MagicMock:
        client = MagicMock()
        client.collection_exists.return_value = exists
        client.count.return_value = MagicMock(count=points)
        return client

    def test_up_to_date_manifest_against_missing_collection_is_reported(self):
        # The exact repro from the issue: manifest fully up to date, collection gone.
        result = self._sync(self._client(exists=False), {"a.py": "h1", "b.py": "h2"})
        self.assertNotIn("No changes since last sync", result)
        self.assertIn("Error", result)
        self.assertIn("test-collection", result)
        self.assertIn("index_repo", result)

    def test_up_to_date_manifest_against_empty_collection_is_reported(self):
        result = self._sync(self._client(exists=True, points=0), {"a.py": "h1", "b.py": "h2"})
        self.assertNotIn("No changes since last sync", result)
        self.assertIn("Error", result)

    def test_up_to_date_manifest_against_populated_collection_still_reports_no_changes(self):
        result = self._sync(self._client(exists=True, points=5), {"a.py": "h1", "b.py": "h2"})
        self.assertIn("No changes since last sync", result)

    def test_changed_files_with_stale_manifest_and_missing_collection_is_reported(self):
        # Without the check this would silently index only the changed file
        # into a brand new collection, leaving the "unchanged" file absent.
        result = self._sync(self._client(exists=False), {"a.py": "h1", "b.py": "old"})
        self.assertIn("Error", result)
        _ims.store_batch.assert_not_called()

    def test_count_ignores_points_without_a_file_path(self):
        # PR #350 review: preserved memory-bank / qdrant-store points must not
        # make an otherwise chunk-less collection look populated. The fake
        # count honors the filter the way Qdrant would.
        client = MagicMock()
        client.collection_exists.return_value = True

        def fake_count(collection_name, count_filter=None, exact=True):
            return MagicMock(count=1 if count_filter is None else 0)

        client.count.side_effect = fake_count
        result = self._sync(client, {"a.py": "h1", "b.py": "h2"})
        self.assertIn("Error", result)
        self.assertIn("no indexed file chunks", result)

    def test_manifest_of_only_zero_chunk_files_is_not_an_error(self):
        # PR #350 review: a file that hashes but chunks to nothing (empty /
        # whitespace-only) is in the manifest yet never creates a point.
        with patch.object(_ims, "chunk_file", side_effect=lambda *a, **kw: iter([])):
            result = self._sync(self._client(exists=False), {"a.py": "h1", "b.py": "h2"})
        self.assertIn("No changes since last sync", result)

    def test_full_rebuild_where_every_tracked_file_changed_is_not_rejected(self):
        # PR #350 review: every manifest entry is changed, so this sync will
        # index all of them itself -- nothing unchanged is missing.
        result = self._sync(self._client(exists=False), {"a.py": "old1", "b.py": "old2"})
        self.assertNotIn("Error", result)
        _ims.store_batch.assert_called_once()

    def test_collection_dropped_between_early_and_later_check_is_reported(self):
        # PR #350 review: exists at the early check, dropped before the
        # later one -- store_batch must not recreate a partial index.
        client = self._client(exists=True, points=5)
        client.collection_exists.side_effect = [True, False]
        result = self._sync(client, {"a.py": "h1", "b.py": "old"})
        self.assertIn("Error", result)
        _ims.store_batch.assert_not_called()

    def test_chunk_failed_changed_file_is_not_exempt_from_the_guard(self):
        # PR #350 review: every tracked file changed and the collection is
        # missing, but chunking one of them just failed -- store_batch must
        # not build a partial collection while the manifest keeps its old hash.
        def flaky_chunk_file(path, rel, *a, **kw):
            if rel == "b.py":
                raise OSError("transient")
            return iter([("content", {"file_path": rel})])

        with patch.object(_ims, "chunk_file", side_effect=flaky_chunk_file):
            result = self._sync(self._client(exists=False), {"a.py": "old1", "b.py": "old2"})
        self.assertIn("Error", result)
        _ims.store_batch.assert_not_called()

    def test_existing_empty_collection_plus_chunk_failure_is_not_a_partial_index(self):
        # Round 6 (a): collection exists but holds no file chunks; every
        # tracked file changed (early check exempts them all), then one fails
        # chunking. The late check must still run and fail closed.
        def flaky_chunk_file(path, rel, *a, **kw):
            if rel == "b.py":
                raise OSError("transient")
            return iter([("content", {"file_path": rel})])

        with patch.object(_ims, "chunk_file", side_effect=flaky_chunk_file):
            result = self._sync(self._client(exists=True, points=0), {"a.py": "old1", "b.py": "old2"})
        self.assertIn("Error", result)
        _ims.store_batch.assert_not_called()

    def test_count_not_found_after_concurrent_drop_is_reported_as_missing(self):
        # Round 6 (b): exists() says yes, the collection is dropped, count()
        # raises not-found; a follow-up exists() confirms the drop.
        client = self._client(exists=True)
        client.collection_exists.side_effect = [True, False]
        client.count.side_effect = RuntimeError("Not found: Collection doesn't exist")
        result = self._sync(client, {"a.py": "h1", "b.py": "h2"})
        self.assertIn("Error", result)
        self.assertIn("does not exist", result)

    def test_count_failure_with_collection_still_present_is_reraised(self):
        client = self._client(exists=True)
        client.count.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self._sync(client, {"a.py": "h1", "b.py": "h2"})

    def test_unreadable_manifest_files_fail_closed(self):
        # PR #350 review: a hash-skipped or re-chunk-unreadable entry is
        # unknown, not a proven zero-chunk file, so it must not permit the
        # zero-chunk exception.
        with patch.object(_ims, "compute_file_hashes", return_value=({"a.py": "h1"}, [("b.py", "boom")])), \
             patch.object(_ims, "chunk_file", side_effect=lambda *a, **kw: iter([])):
            result = self._sync(self._client(exists=False), {"a.py": "h1", "b.py": "h2"})
        self.assertIn("Error", result)
        with patch.object(_ims, "chunk_file", side_effect=OSError("transient")):
            result = self._sync(self._client(exists=False), {"a.py": "h1", "b.py": "h2"})
        self.assertIn("Error", result)

    def test_empty_manifest_with_missing_collection_is_a_normal_first_sync(self):
        result = self._sync(self._client(exists=False), {})
        self.assertNotIn("Error", result)
        _ims.store_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
