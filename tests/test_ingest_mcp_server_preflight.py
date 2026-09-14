#!/usr/bin/env python3
"""Tests for index_repo's pre-flight duplicate-point warning (issue #58).

Stdlib-only (unittest, no pytest) and no network -- all QdrantClient,
FastEmbedProvider, store_batch, and build_entries calls are mocked so these
run without a live Qdrant instance or model download.

    .venv/bin/python -m unittest discover -s tests

Why this file exists: the warning fires early (before the expensive embedding
loop) and must NOT fire when reset=True, force=True, collection is empty, or
collection doesn't exist -- all four "don't warn" cases are pinned here because
a regression in any direction is silent: a false positive stops legitimate
re-indexing; a false negative lets duplicate chunks accumulate unnoticed.
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call as mock_call

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

# Minimal env so module-level DEFAULT_* constants don't explode on import --
# same pattern established in test_qdrant_ingest_lib.py and
# test_ingest_mcp_server_tool_count.py.
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "test-collection")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

import ingest_mcp_server as _ims  # noqa: E402


def _make_collection_info(points_count: int) -> MagicMock:
    """Build a mock CollectionInfo-like object with the given points_count."""
    info = MagicMock()
    info.points_count = points_count
    return info


def _run(coro):
    """Run an async coroutine in the test runner's sync context."""
    return asyncio.run(coro)


class IndexRepoPreflightWarningTest(unittest.TestCase):
    """
    Pre-flight check: when reset=False, force=False, and the target collection
    already contains points, index_repo must return a warning string before
    starting the expensive embedding loop.
    """

    def setUp(self):
        # call_with_retry: pass through to fn(*args, **kwargs) by default.
        # Individual tests override return values on the mock client below.
        self._retry_patcher = patch.object(
            _ims, "call_with_retry", side_effect=lambda fn, *a, **kw: fn(*a, **kw)
        )
        self._retry_patcher.start()

        # store_batch is an async function that does the actual embed+upsert.
        # Any test that expects an early return must assert it was NOT called.
        self._store_batch_patcher = patch.object(_ims, "store_batch", new_callable=AsyncMock)
        self.mock_store_batch = self._store_batch_patcher.start()

        # iter_entries is the CPU-bound scan -- also must not run on early return.
        # Returns an empty generator (no chunks, no skipped files) to keep tests
        # fast and network-free. index_repo now calls iter_entries(), not build_entries().
        self._iter_entries_patcher = patch.object(_ims, "iter_entries", return_value=iter([]))
        self.mock_iter_entries = self._iter_entries_patcher.start()

        # FastEmbedProvider and QdrantConnector: never need to do anything real.
        self._embed_patcher = patch("ingest_mcp_server.FastEmbedProvider")
        self.mock_embed_cls = self._embed_patcher.start()

        self._connector_patcher = patch("ingest_mcp_server.QdrantConnector")
        self.mock_connector_cls = self._connector_patcher.start()

        # ensure_file_path_index: no-op for all tests.
        self._index_patcher = patch.object(_ims, "ensure_file_path_index")
        self.mock_ensure_index = self._index_patcher.start()

    def tearDown(self):
        self._retry_patcher.stop()
        self._store_batch_patcher.stop()
        self._iter_entries_patcher.stop()
        self._embed_patcher.stop()
        self._connector_patcher.stop()
        self._index_patcher.stop()

    def _make_client(self, exists: bool, points_count: int = 0) -> MagicMock:
        """Return a mock QdrantClient with controlled collection_exists / get_collection."""
        client = MagicMock()
        client.collection_exists.return_value = exists
        client.get_collection.return_value = _make_collection_info(points_count)
        return client

    def _patch_client(self, client: MagicMock):
        return patch("ingest_mcp_server.QdrantClient", return_value=client)

    # ------------------------------------------------------------------ #
    #  Cases that SHOULD return a warning (no embedding performed)        #
    # ------------------------------------------------------------------ #

    def test_non_empty_collection_returns_warning_before_embedding(self):
        """reset=False, force=False, 500 existing points → warning, no embed."""
        client = self._make_client(exists=True, points_count=500)
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="test-collection",
                reset=False,
                force=False,
            ))
        self.assertIn("Warning", result)
        self.assertIn("500", result)
        self.mock_store_batch.assert_not_called()
        self.mock_iter_entries.assert_not_called()

    def test_warning_mentions_reset_option(self):
        """Warning must mention reset=True and caution that it wipes ALL collection data."""
        client = self._make_client(exists=True, points_count=1)
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="test-collection",
                reset=False,
                force=False,
            ))
        self.assertIn("reset=True", result)
        # Must warn that reset wipes ALL data, not just this repo's chunks --
        # collections can hold unrelated conversation-memory data too.
        self.assertIn("wipes", result.lower())

    def test_warning_uses_may_not_will_for_duplicate_risk(self):
        """Warning must say 'may' (not 'will') add duplicates -- existing points may not
        be from a prior index_repo run at all (they could be conversation-memory data)."""
        client = self._make_client(exists=True, points_count=1)
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="test-collection",
                reset=False,
                force=False,
            ))
        self.assertIn("may add", result)
        self.assertNotIn("will add", result)

    def test_warning_mentions_sync_repo_alternative(self):
        """Warning must mention sync_repo as the preferred duplicate-safe alternative."""
        client = self._make_client(exists=True, points_count=1)
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="test-collection",
                reset=False,
                force=False,
            ))
        self.assertIn("sync_repo", result)

    def test_warning_mentions_force_option(self):
        """Warning must tell the user force=True bypasses the check."""
        client = self._make_client(exists=True, points_count=1)
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="test-collection",
                reset=False,
                force=False,
            ))
        self.assertIn("force=True", result)

    def test_warning_includes_collection_name(self):
        """Warning message should name the specific collection."""
        client = self._make_client(exists=True, points_count=42)
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="my-specific-repo",
                reset=False,
                force=False,
            ))
        self.assertIn("my-specific-repo", result)
        self.assertIn("42", result)

    # ------------------------------------------------------------------ #
    #  Cases that MUST NOT warn (embedding proceeds normally)             #
    # ------------------------------------------------------------------ #

    def test_empty_collection_does_not_warn(self):
        """Collection exists but has 0 points: no warning, embedding runs."""
        client = self._make_client(exists=True, points_count=0)
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="test-collection",
                reset=False,
                force=False,
            ))
        self.assertNotIn("Warning", result)
        # iter_entries was called (the chunking path was not blocked by a warning).
        # With an empty iterator (no chunks), store_batch is never reached since
        # the pending batch stays empty -- the important thing is no warning fired.
        self.mock_iter_entries.assert_called()

    def test_nonexistent_collection_does_not_warn(self):
        """Collection doesn't exist: no warning (no check needed), embedding runs."""
        client = self._make_client(exists=False)
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="test-collection",
                reset=False,
                force=False,
            ))
        self.assertNotIn("Warning", result)
        # iter_entries was reached (the path wasn't blocked by a warning).
        # With an empty iterator (no chunks), store_batch is not called.
        self.mock_iter_entries.assert_called()

    def test_reset_true_skips_preflight_check(self):
        """reset=True: the pre-flight check is never reached, embedding runs."""
        client = self._make_client(exists=True, points_count=9999)
        client.delete_collection.return_value = True
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="test-collection",
                reset=True,
                force=False,
            ))
        self.assertNotIn("Warning", result)
        # iter_entries was reached (the path wasn't blocked by a warning).
        # With an empty iterator (no chunks), store_batch is not called.
        self.mock_iter_entries.assert_called()

    def test_force_true_bypasses_warning_for_non_empty_collection(self):
        """reset=False, force=True, non-empty collection: no warning, embedding runs."""
        client = self._make_client(exists=True, points_count=500)
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="test-collection",
                reset=False,
                force=True,
            ))
        self.assertNotIn("Warning", result)
        # iter_entries was reached (the path wasn't blocked by a warning).
        # With an empty iterator (no chunks), store_batch is not called.
        self.mock_iter_entries.assert_called()

    def test_none_points_count_does_not_warn(self):
        """points_count=None (freshly created or schema variation): no warning."""
        client = self._make_client(exists=True, points_count=0)
        client.get_collection.return_value.points_count = None
        with self._patch_client(client):
            result = _run(_ims.index_repo(
                repo_path=REPO_ROOT,
                collection="test-collection",
                reset=False,
                force=False,
            ))
        self.assertNotIn("Warning", result)
        # iter_entries was reached (the path wasn't blocked by a warning).
        # With an empty iterator (no chunks), store_batch is not called.
        self.mock_iter_entries.assert_called()


if __name__ == "__main__":
    unittest.main()
