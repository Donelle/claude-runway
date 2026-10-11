#!/usr/bin/env python3
"""Tests for the search_limit recompute wired into sync_repo/index_repo
(issue #331, part of #326). Stdlib-only, no network: QdrantClient, the cache
and the embed/store pipeline are all mocked.

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
from qdrant_client import models as _m  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _client(points: int, metadata=None, exists: bool = True) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = exists
    info = MagicMock()
    info.points_count = points
    info.config.metadata = metadata
    client.get_collection.return_value = info
    return client


class RecomputeSearchLimitTest(unittest.TestCase):
    def setUp(self):
        for target, kwargs in (
            ("call_with_retry", {"side_effect": lambda fn, *a, **kw: fn(*a, **kw)}),
            ("get_cached_search_limit", {"return_value": None}),
            ("set_cached_search_limit", {}),
        ):
            patcher = patch.object(_ims, target, **kwargs)
            setattr(self, "m_" + target, patcher.start())
            self.addCleanup(patcher.stop)

    def _call(self, client):
        with patch("ingest_mcp_server.QdrantClient", return_value=client):
            _ims._recompute_search_limit("c", "http://q", None)

    def test_writes_metadata_and_cache_when_changed(self):
        client = _client(points=2_000, metadata={"description": "keep me"})
        self._call(client)
        # 2000 points -> second tier (15); only the search_limit key is sent
        # (Qdrant merges the patch), never a read-back snapshot of the rest.
        client.update_collection.assert_called_once_with(collection_name="c", metadata={"search_limit": 15})
        self.m_set_cached_search_limit.assert_called_once_with("http://q", "c", 15)

    def test_none_metadata_is_handled(self):
        client = _client(points=0, metadata=None)
        self._call(client)
        client.update_collection.assert_called_once_with(collection_name="c", metadata={"search_limit": 10})

    def test_no_writes_when_both_already_current(self):
        client = _client(points=2_000, metadata={"search_limit": 15})
        self.m_get_cached_search_limit.return_value = 15
        self._call(client)
        client.update_collection.assert_not_called()
        self.m_set_cached_search_limit.assert_not_called()

    def test_stale_cache_alone_is_refreshed_without_a_qdrant_write(self):
        client = _client(points=2_000, metadata={"search_limit": 15})
        self.m_get_cached_search_limit.return_value = 10
        self._call(client)
        client.update_collection.assert_not_called()
        self.m_set_cached_search_limit.assert_called_once_with("http://q", "c", 15)

    def test_missing_collection_is_a_noop(self):
        client = _client(points=0, exists=False)
        self._call(client)
        client.get_collection.assert_not_called()
        client.update_collection.assert_not_called()

    def test_fails_open_on_any_error(self):
        client = _client(points=5)
        client.update_collection.side_effect = RuntimeError("boom")
        self._call(client)  # must not raise
        self.m_set_cached_search_limit.assert_not_called()

    def test_fails_open_when_client_construction_fails(self):
        with patch("ingest_mcp_server.QdrantClient", side_effect=RuntimeError("no qdrant")):
            _ims._recompute_search_limit("c", "http://q", None)  # must not raise

    def test_no_collection_is_a_noop(self):
        with patch("ingest_mcp_server.QdrantClient") as cls:
            _ims._recompute_search_limit(None, "http://q", None)
        cls.assert_not_called()


class WiringTest(unittest.TestCase):
    """Both tools call the shared helper; sync_repo's no-change path doesn't."""

    def setUp(self):
        for target, kwargs in (
            ("call_with_retry", {"side_effect": lambda fn, *a, **kw: fn(*a, **kw)}),
            ("_save_manifest", {}),
            ("chunk_file", {"side_effect": lambda *a, **kw: iter([("content", {"file_path": "x"})])}),
            ("ensure_file_path_index", {}),
            ("iter_entries", {"return_value": iter([])}),
            ("_recompute_search_limit", {}),
        ):
            patcher = patch.object(_ims, target, **kwargs)
            setattr(self, "m_" + target, patcher.start())
            self.addCleanup(patcher.stop)
        for target, kwargs in (
            ("ingest_mcp_server.FastEmbedProvider", {}),
            ("ingest_mcp_server.QdrantConnector", {}),
            ("ingest_mcp_server.store_batch", {"new_callable": AsyncMock}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _sync_client(self) -> MagicMock:
        client = _client(points=1)
        client.count.return_value = MagicMock(count=5)
        client.get_collection.return_value.config.params.vectors = {
            "fast-some-model": _m.VectorParams(size=384, distance=_m.Distance.COSINE)
        }
        return client

    def _sync(self, hashes, manifest):
        provider = MagicMock()
        provider.get_vector_name.return_value = "fast-some-model"
        provider.get_vector_size.return_value = 384
        with patch.object(_ims, "compute_file_hashes", return_value=(hashes, [])), \
             patch.object(_ims, "_load_manifest", return_value=manifest), \
             patch("ingest_mcp_server.QdrantClient", return_value=self._sync_client()), \
             patch("ingest_mcp_server.FastEmbedProvider", return_value=provider):
            return _run(_ims.sync_repo(repo_path=REPO_ROOT, collection="test-collection"))

    def test_sync_repo_recomputes_after_a_real_sync(self):
        result = self._sync({"a.py": "h1"}, {})
        self.assertIn("Synced", result)
        self.m__recompute_search_limit.assert_called_once()
        self.assertEqual(self.m__recompute_search_limit.call_args.args[0], "test-collection")

    def test_sync_repo_skips_recompute_on_no_changes(self):
        result = self._sync({"a.py": "h1"}, {"a.py": "h1"})
        self.assertIn("No changes", result)
        self.m__recompute_search_limit.assert_not_called()

    def test_index_repo_recomputes(self):
        client = _client(points=0, exists=False)
        with patch("ingest_mcp_server.QdrantClient", return_value=client):
            result = _run(_ims.index_repo(repo_path=REPO_ROOT, collection="test-collection"))
        self.assertIn("Indexed", result)
        self.m__recompute_search_limit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
