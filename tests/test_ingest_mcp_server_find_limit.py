#!/usr/bin/env python3
"""Tests for find_in_collection's limit resolution (issue #332, part of
#326). Stdlib-only, no network: QdrantClient, the cache, the embedding
provider and the connector are all mocked.

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


def _client(metadata=None) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value.config.metadata = metadata
    return client


class FindInCollectionLimitTest(unittest.TestCase):
    def setUp(self):
        self.cached = None
        patches = {
            "call_with_retry": patch.object(_ims, "call_with_retry", side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            "async_call_with_retry": patch.object(
                _ims, "async_call_with_retry", new_callable=AsyncMock, return_value=[]
            ),
            "get_cached": patch.object(_ims, "get_cached_search_limit", side_effect=lambda *_: self.cached),
            "set_cached": patch.object(_ims, "set_cached_search_limit"),
            "provider": patch.object(_ims, "FastEmbedProvider"),
            "connector": patch.object(_ims, "QdrantConnector"),
            "mismatch": patch.object(_ims, "check_embedding_model_mismatch", return_value=None),
        }
        self.m = {}
        for name, p in patches.items():
            self.m[name] = p.start()
            self.addCleanup(p.stop)

    def _find(self, client, **kwargs):
        with patch.object(_ims, "QdrantClient", return_value=client):
            asyncio.run(_ims.find_in_collection("q", "c", **kwargs))
        return self.m["async_call_with_retry"].call_args.kwargs["limit"]

    def test_explicit_limit_wins_without_any_lookup(self):
        self.cached = 25
        client = _client({"search_limit": 40})
        self.assertEqual(self._find(client, limit=3), 3)
        client.get_collection.assert_not_called()

    def test_cache_hit_used_without_qdrant_fetch(self):
        self.cached = 25
        client = _client({"search_limit": 40})
        self.assertEqual(self._find(client), 25)
        client.get_collection.assert_not_called()

    def test_cache_miss_reads_live_metadata_and_backfills(self):
        client = _client({"search_limit": 15})
        self.assertEqual(self._find(client), 15)
        self.m["set_cached"].assert_called_once_with(_ims.DEFAULT_QDRANT_URL, "c", 15)

    def test_no_stored_value_falls_back_to_default_without_caching(self):
        for metadata in (None, {}, {"search_limit": "x"}, {"search_limit": True}, {"search_limit": 0}):
            with self.subTest(metadata=metadata):
                self.m["set_cached"].reset_mock()
                self.assertEqual(self._find(_client(metadata)), 10)
                self.m["set_cached"].assert_not_called()

    def test_lookup_failure_fails_open_to_default(self):
        client = _client()
        client.get_collection.side_effect = RuntimeError("boom")
        self.assertEqual(self._find(client), 10)

    def test_invalid_explicit_limit_rejected(self):
        client = _client()
        with patch.object(_ims, "QdrantClient", return_value=client):
            out = asyncio.run(_ims.find_in_collection("q", "c", limit=0))
        self.assertTrue(out.startswith("Error: limit must be 1 or greater"))


if __name__ == "__main__":
    unittest.main()
