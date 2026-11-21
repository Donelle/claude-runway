#!/usr/bin/env python3
"""Tests for the protected-collection guards in mcpservers/codebase_indexer_mcp_server.py:


- cb_indexer_index_repo/cb_indexer_sync_repo refuse outright an existing collection
  the codebase-indexer did not create (libs/qdrant_code_index_marker.py): the
  memory bank under any name, a compact collection, an empty or unrelated
  one. This replaced a check against the indexer's own copy of
  MEMORY_BANK_COLLECTION, which protected one name only; no env var is set
  here, so a refusal below can only come from the mark.
- cb_indexer_index_repo(reset=True) ALWAYS preserves memory-bank points via a filtered
  delete, and never calls delete_collection or counts anything first (PR
  An earlier version branched on a memory-bank point count, but
  that count and the delete were separate Qdrant requests with no
  transaction between them -- a concurrent remember() landing in the gap
  could still be destroyed by a "zero-count" full delete. Always using the
  filtered delete removes the race instead of narrowing it).
- cb_indexer_sync_repo's per-batch delete filter carries a must_not exclusion for
  memory-bank points.

Stdlib-only (unittest, no pytest), no network -- mirrors
test_codebase_indexer_mcp_server_preflight.py's mocking style.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "mcpservers"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # fake_ownership_client

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "test-collection")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

import codebase_indexer_mcp_server as _ims  # noqa: E402
import memory_bank_lib as mb  # noqa: E402
import qdrant_code_index_marker as marker  # noqa: E402
from fake_ownership_client import (  # noqa: E402
    FakeOwnershipClient,
    code_chunk,
    memory_point,
    unrelated_point,
)
from qdrant_client.http.models import models as _m  # noqa: E402
from qdrant_compact_marker import MARKER_KEY as COMPACT_KEY, MARKER_VALUE as COMPACT_VALUE  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _accept_ownership(test: unittest.TestCase) -> None:
    """For tests about what happens AFTER the ownership guard: their MagicMock
    clients carry no real collection metadata, so the guard (which fails
    closed) would refuse them. The guard has its own tests below."""
    patcher = patch.object(_ims, "check_code_index_ownership", return_value=marker.OwnershipCheck())
    patcher.start()
    test.addCleanup(patcher.stop)


def _foreign_collections():
    """(label, client) for every existing collection the indexer must refuse,
    whatever it is called -- the issue's acceptance list."""
    return (
        ("memory bank under a non-default name", FakeOwnershipClient(payloads=[memory_point()])),
        ("marked compact collection", FakeOwnershipClient(metadata={COMPACT_KEY: COMPACT_VALUE})),
        ("empty unmarked collection", FakeOwnershipClient(payloads=[])),
        ("unmarked collection of unrelated points", FakeOwnershipClient(payloads=[unrelated_point()])),
        ("unreadable metadata", FakeOwnershipClient(fail_on="get_collection")),
    )


class OwnershipGuardTest(unittest.TestCase):
    """The FakeOwnershipClient has no delete/upsert/count methods at all, so a
    run that got past a refusal would raise instead of returning the error."""

    def setUp(self):
        for target, kwargs in (
            ("call_with_retry", {"side_effect": lambda fn, *a, **kw: fn(*a, **kw)}),
            ("_save_manifest", {}),
            ("_discard_manifest", {}),
            ("_load_manifest", {}),
            # Neither may be reached on a refusal: no model load, no store.
            ("FastEmbedProvider", {}),
            ("store_batch", {"new_callable": AsyncMock}),
        ):
            patcher = patch.object(_ims, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _assert_refused(self, client, call):
        with patch("codebase_indexer_mcp_server.QdrantClient", return_value=client):
            result = _run(call())
        self.assertTrue(result.startswith("Error"), result)
        self.assertIn("'shared-thing'", result)
        self.assertEqual(client.updates, [])
        _ims.FastEmbedProvider.assert_not_called()
        _ims.store_batch.assert_not_called()
        _ims._discard_manifest.assert_not_called()
        _ims._load_manifest.assert_not_called()

    def test_index_repo_refuses_with_and_without_reset_and_force(self):
        for reset, force in ((False, False), (True, False), (False, True), (True, True)):
            for label, client in _foreign_collections():
                with self.subTest(label=label, reset=reset, force=force):
                    # Bound as defaults: the lambda runs inside this iteration.
                    self._assert_refused(client, lambda reset=reset, force=force: _ims.index_repo(
                        repo_path=REPO_ROOT, collection="shared-thing", reset=reset, force=force
                    ))

    def test_sync_repo_refuses(self):
        for label, client in _foreign_collections():
            with self.subTest(label=label):
                self._assert_refused(client, lambda: _ims.sync_repo(repo_path=REPO_ROOT, collection="shared-thing"))

    def test_no_memory_bank_name_is_kept_by_the_indexer(self):
        # The indexer no longer has its own copy of the memory-bank name.
        self.assertFalse(hasattr(_ims, "DEFAULT_MEMORY_BANK_COLLECTION"))
        self.assertFalse(hasattr(mb, "is_memory_bank_collection_name"))

    def test_unmarked_code_collection_is_adopted_and_indexed(self):
        # The issue's step-3 decision: a collection indexed before the mark
        # existed is marked and used, not refused.
        client = FakeOwnershipClient(payloads=[code_chunk()])
        with patch("codebase_indexer_mcp_server.QdrantClient", return_value=client), \
             patch.object(_ims, "prepare_collection_for_index", return_value=MagicMock(error="stop-here")):
            result = _run(_ims.index_repo(repo_path=REPO_ROOT, collection="shared-thing", force=True))
        self.assertEqual(result, "stop-here")  # got past the guard to the preparation step
        self.assertEqual(client.updates, [marker.code_index_metadata()])


class NewCollectionsCarryTheMarkTest(unittest.TestCase):
    """The connector every write path builds passes the mark, so a collection
    it creates is marked in the create request itself."""

    def setUp(self):
        _accept_ownership(self)
        for target, kwargs in (
            ("call_with_retry", {"side_effect": lambda fn, *a, **kw: fn(*a, **kw)}),
            ("store_batch", {"new_callable": AsyncMock}),
            ("iter_entries", {"return_value": iter([])}),
            ("ensure_file_path_index", {}),
            ("_save_manifest", {}),
            ("_discard_manifest", {}),
            ("_load_manifest", {"return_value": {}}),
            ("compute_file_hashes", {"return_value": ({"a.py": "h"}, [])}),
            ("chunk_file", {"return_value": iter([])}),
            ("FastEmbedProvider", {}),
        ):
            patcher = patch.object(_ims, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _connector_kwargs(self, call):
        client = MagicMock()
        client.collection_exists.return_value = False
        with patch("codebase_indexer_mcp_server.QdrantClient", return_value=client), \
             patch("codebase_indexer_mcp_server.QdrantConnector") as connector_cls:
            _run(call())
        connector_cls.assert_called_once()
        return connector_cls.call_args.kwargs

    def _assert_marks_and_checks(self, kwargs):
        self.assertEqual(kwargs["collection_metadata"], marker.code_index_metadata())
        # An existing collection is re-checked before every batch.
        self.assertIs(kwargs["existing_collection_check"], marker.metadata_ownership_error)

    def test_index_repo(self):
        self._assert_marks_and_checks(
            self._connector_kwargs(lambda: _ims.index_repo(repo_path=REPO_ROOT, collection="new-coll"))
        )

    def test_sync_repo(self):
        self._assert_marks_and_checks(
            self._connector_kwargs(lambda: _ims.sync_repo(repo_path=REPO_ROOT, collection="new-coll"))
        )

    def test_store_repo_connector(self):
        with patch.object(_ims, "_find_store_connector", None), \
             patch("codebase_indexer_mcp_server.QdrantConnector") as connector_cls:
            _ims._get_find_store_connector()
        self._assert_marks_and_checks(connector_cls.call_args.kwargs)


class IndexRepoResetGuardTest(unittest.TestCase):
    def setUp(self):
        _accept_ownership(self)
        self._retry_patcher = patch.object(_ims, "call_with_retry", side_effect=lambda fn, *a, **kw: fn(*a, **kw))
        self._retry_patcher.start()
        self.addCleanup(self._retry_patcher.stop)

        self._store_batch_patcher = patch.object(_ims, "store_batch", new_callable=AsyncMock)
        self._store_batch_patcher.start()
        self.addCleanup(self._store_batch_patcher.stop)

        self._iter_entries_patcher = patch.object(_ims, "iter_entries", return_value=iter([]))
        self._iter_entries_patcher.start()
        self.addCleanup(self._iter_entries_patcher.stop)

        self._embed_patcher = patch("codebase_indexer_mcp_server.FastEmbedProvider")
        self._embed_patcher.start()
        self.addCleanup(self._embed_patcher.stop)

        self._connector_patcher = patch("codebase_indexer_mcp_server.QdrantConnector")
        self._connector_patcher.start()
        self.addCleanup(self._connector_patcher.stop)

        self._index_patcher = patch.object(_ims, "ensure_file_path_index")
        self._index_patcher.start()
        self.addCleanup(self._index_patcher.stop)

        # These tests index REPO_ROOT, the real checkout: cb_indexer_index_repo writes (and
        # on reset removes) the repo's sync manifest, which must not touch it.
        for name in ("_save_manifest", "_discard_manifest"):
            patcher = patch.object(_ims, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _make_client(self, exists=True):
        client = MagicMock()
        client.collection_exists.return_value = exists
        return client

    def test_reset_always_takes_the_filtered_delete_path(self):
        # An earlier version counted memory-bank points first
        # and only used the filtered (preserving) delete when that count was
        # nonzero -- but count-then-act is racy against a concurrent
        # remember() call landing in between. Fixed by ALWAYS using the
        # filtered delete regardless of count, which removes the race
        # entirely instead of narrowing it. This test pins that: no count
        # check happens at all, and delete_collection is never called.
        client = self._make_client()
        # get_collection must return a vectors_config the mismatch check
        # actually recognizes as compatible:
        # an unconfigured MagicMock is neither None, dict, nor VectorParams,
        # which now correctly fails closed rather than silently passing.
        info = MagicMock()
        info.config.params.vectors = {"fast-some-model": _m.VectorParams(size=384, distance=_m.Distance.COSINE)}
        client.get_collection.return_value = info
        provider_instance = MagicMock()
        provider_instance.get_vector_name.return_value = "fast-some-model"
        provider_instance.get_vector_size.return_value = 384
        with patch("codebase_indexer_mcp_server.QdrantClient", return_value=client), \
             patch("codebase_indexer_mcp_server.FastEmbedProvider", return_value=provider_instance), \
             patch.object(mb, "count_memory_bank_points") as mock_count:
            _run(_ims.index_repo(repo_path=REPO_ROOT, collection="test-collection", reset=True))
        mock_count.assert_not_called()
        client.delete_collection.assert_not_called()
        client.delete.assert_called_once()
        _, kwargs = client.delete.call_args
        self.assertEqual(kwargs["collection_name"], "test-collection")
        must_not = kwargs["points_selector"].must_not
        # Memory-bank points AND conversation compacts.
        self.assertEqual({c.match.value for c in must_not}, {mb.MEMORY_BANK_SOURCE, mb.COMPACT_SOURCE})

    def test_collection_not_existing_skips_both_delete_paths(self):
        client = self._make_client(exists=False)
        with patch("codebase_indexer_mcp_server.QdrantClient", return_value=client):
            _run(_ims.index_repo(repo_path=REPO_ROOT, collection="test-collection", reset=True))
        client.delete_collection.assert_not_called()
        client.delete.assert_not_called()

    def test_schema_mismatch_blocks_the_destructive_delete(self):
        # Regression for a review finding (third pass): without a schema check,
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
        with patch("codebase_indexer_mcp_server.QdrantClient", return_value=client), \
             patch("codebase_indexer_mcp_server.FastEmbedProvider", return_value=provider_instance):
            result = _run(_ims.index_repo(repo_path=REPO_ROOT, collection="test-collection", reset=True))
        self.assertIn("Error", result)
        client.delete.assert_not_called()
        client.delete_collection.assert_not_called()

    def test_inconclusive_schema_check_blocks_the_destructive_delete(self):
        # Regression for a review finding (fourth pass): the pre-delete check
        # must fail CLOSED, not open -- a transient error verifying schema
        # compatibility (e.g. get_collection failing) must block the delete
        # the same way a confirmed mismatch does, not be silently treated as
        # "compatible" just because no definitive mismatch was ever found.
        client = self._make_client()
        client.get_collection.side_effect = RuntimeError("connection refused")
        with patch("codebase_indexer_mcp_server.QdrantClient", return_value=client):
            result = _run(_ims.index_repo(repo_path=REPO_ROOT, collection="test-collection", reset=True))
        self.assertIn("Error", result)
        client.delete.assert_not_called()
        client.delete_collection.assert_not_called()


class SyncRepoDeleteFilterTest(unittest.TestCase):
    def setUp(self):
        _accept_ownership(self)
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

        # An empty manifest against an existing collection is a rebuild,
        # which adds a non-chunk-points count before the per-file delete
        # and an orphan sweep after the store. Both are tested on their own in
        # test_codebase_indexer_mcp_server_rebuild.py; here they are stubbed so
        # `client.delete` still sees only the per-file delete these tests pin.
        for helper, value in (("_non_chunk_points_warning", ""), ("_delete_orphaned_chunks", [])):
            patcher = patch.object(_ims, helper, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self._store_batch_patcher = patch.object(_ims, "store_batch", new_callable=AsyncMock)
        self._store_batch_patcher.start()
        self.addCleanup(self._store_batch_patcher.stop)

        self._embed_patcher = patch("codebase_indexer_mcp_server.FastEmbedProvider")
        self._embed_patcher.start()
        self.addCleanup(self._embed_patcher.stop)

        self._connector_patcher = patch("codebase_indexer_mcp_server.QdrantConnector")
        self._connector_patcher.start()
        self.addCleanup(self._connector_patcher.stop)

        self._index_patcher = patch.object(_ims, "ensure_file_path_index")
        self._index_patcher.start()
        self.addCleanup(self._index_patcher.stop)

    def test_delete_filter_excludes_memory_bank_points(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        # get_collection must return a vectors_config the mismatch check
        # actually recognizes as compatible:
        # an unconfigured MagicMock is neither None, dict, nor VectorParams,
        # which now correctly fails closed rather than silently passing.
        info = MagicMock()
        info.config.params.vectors = {"fast-some-model": _m.VectorParams(size=384, distance=_m.Distance.COSINE)}
        client.get_collection.return_value = info
        provider_instance = MagicMock()
        provider_instance.get_vector_name.return_value = "fast-some-model"
        provider_instance.get_vector_size.return_value = 384
        with patch("codebase_indexer_mcp_server.QdrantClient", return_value=client), \
             patch("codebase_indexer_mcp_server.FastEmbedProvider", return_value=provider_instance):
            _run(_ims.sync_repo(repo_path=REPO_ROOT, collection="test-collection"))
        client.delete.assert_called_once()
        _, kwargs = client.delete.call_args
        must_not = kwargs["points_selector"].must_not
        self.assertEqual({c.key for c in must_not}, {mb.SOURCE_FIELD})
        self.assertEqual({c.match.value for c in must_not}, {mb.MEMORY_BANK_SOURCE, mb.COMPACT_SOURCE})
        # The existing should=[file_path...] behavior must be unaffected.
        should = kwargs["points_selector"].should
        self.assertEqual({c.match.value for c in should}, {"changed_file.py"})

    def test_schema_mismatch_blocks_the_delete_loop(self):
        # Regression for a review finding (fifth pass): without this check,
        # cb_indexer_sync_repo would delete each changed/removed file's old chunks
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
        with patch("codebase_indexer_mcp_server.QdrantClient", return_value=client), \
             patch("codebase_indexer_mcp_server.FastEmbedProvider", return_value=provider_instance):
            result = _run(_ims.sync_repo(repo_path=REPO_ROOT, collection="test-collection"))
        self.assertIn("Error", result)
        client.delete.assert_not_called()

    def test_inconclusive_schema_check_blocks_the_delete_loop(self):
        # fail_closed=True: an inconclusive check (e.g. a transient error)
        # must block the delete the same way a confirmed mismatch does.
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.side_effect = RuntimeError("connection refused")
        with patch("codebase_indexer_mcp_server.QdrantClient", return_value=client):
            result = _run(_ims.sync_repo(repo_path=REPO_ROOT, collection="test-collection"))
        self.assertIn("Error", result)
        client.delete.assert_not_called()


class SyncRepoMissingCollectionTest(unittest.TestCase):
    """cb_indexer_sync_repo's manifest-diff "no changes" early return
    used to run before any collection_exists check, so a manifest claiming
    every file was indexed masked a dropped/empty collection -- which is
    exactly the state the documented EMBEDDING_MODEL-change remediation
    (manually drop the collection) leaves behind."""

    def setUp(self):
        _accept_ownership(self)
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
            ("codebase_indexer_mcp_server.FastEmbedProvider", {}),
            ("codebase_indexer_mcp_server.QdrantConnector", {}),
            ("codebase_indexer_mcp_server.store_batch", {"new_callable": AsyncMock}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _sync(self, client, manifest):
        with patch.object(_ims, "_load_manifest", return_value=manifest), \
             patch("codebase_indexer_mcp_server.QdrantClient", return_value=client):
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
        self.assertIn("cb_indexer_index_repo", result)

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
        # Preserved memory-bank / cb_indexer_store_repo points must not
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
        # A file that hashes but chunks to nothing (empty /
        # whitespace-only) is in the manifest yet never creates a point. The
        # probe asks file_has_indexable_content (no token budget is
        # available yet at that point), not chunk_file.
        with patch.object(_ims, "file_has_indexable_content", return_value=False):
            result = self._sync(self._client(exists=False), {"a.py": "h1", "b.py": "h2"})
        self.assertIn("No changes since last sync", result)

    def test_full_rebuild_where_every_tracked_file_changed_is_not_rejected(self):
        # Every manifest entry is changed, so this sync will
        # index all of them itself -- nothing unchanged is missing.
        result = self._sync(self._client(exists=False), {"a.py": "old1", "b.py": "old2"})
        self.assertNotIn("Error", result)
        _ims.store_batch.assert_called_once()

    def test_collection_dropped_between_early_and_later_check_is_reported(self):
        # Exists at the early check, dropped before the
        # later one -- store_batch must not recreate a partial index.
        client = self._client(exists=True, points=5)
        client.collection_exists.side_effect = [True, False]
        result = self._sync(client, {"a.py": "h1", "b.py": "old"})
        self.assertIn("Error", result)
        _ims.store_batch.assert_not_called()

    def test_chunk_failed_changed_file_is_not_exempt_from_the_guard(self):
        # Every tracked file changed and the collection is
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
        # a hash-skipped or re-chunk-unreadable entry is
        # unknown, not a proven zero-chunk file, so it must not permit the
        # zero-chunk exception.
        with patch.object(_ims, "compute_file_hashes", return_value=({"a.py": "h1"}, [("b.py", "boom")])), \
             patch.object(_ims, "file_has_indexable_content", return_value=False):
            result = self._sync(self._client(exists=False), {"a.py": "h1", "b.py": "h2"})
        self.assertIn("Error", result)
        with patch.object(_ims, "file_has_indexable_content", side_effect=OSError("transient")):
            result = self._sync(self._client(exists=False), {"a.py": "h1", "b.py": "h2"})
        self.assertIn("Error", result)

    def test_empty_manifest_with_missing_collection_is_a_normal_first_sync(self):
        result = self._sync(self._client(exists=False), {})
        self.assertNotIn("Error", result)
        _ims.store_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
