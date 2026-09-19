#!/usr/bin/env python3
"""Tests for tools/memory_bank_mcp_server.py's tool-layer argument
validation and tool registration (issue #175).

Stdlib-only (unittest, no pytest), no network -- these only cover
validation that happens before any QdrantClient/FastEmbedProvider is ever
constructed. Full remember/recall/forget round-trip behavior is covered by
tests/test_memory_bank_lib.py (the underlying logic) and was additionally
verified manually against a real Qdrant instance during development.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import io
import os
import sys
import unittest
from contextlib import redirect_stderr
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

# Minimal env so module-level DEFAULT_* constants don't explode on import --
# same pattern established in test_ingest_mcp_server_tool_count.py.
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MEMORY_BANK_ID", "test-collection")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("MEMORY_BANK_COLLECTION", "memory-bank")

import memory_bank_mcp_server as _mbs  # noqa: E402
import memory_bank_lib as _mb  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class ForgetArgumentValidationTest(unittest.TestCase):
    def test_neither_point_id_nor_wipe_all_is_an_error(self):
        result = _mbs.forget()
        self.assertIn("Error", result)

    def test_both_point_id_and_wipe_all_is_an_error(self):
        result = _mbs.forget(point_id="some-id", wipe_all=True)
        self.assertIn("Error", result)

    def test_wipe_all_false_with_no_point_id_is_an_error(self):
        result = _mbs.forget(point_id=None, wipe_all=False)
        self.assertIn("Error", result)


class RecallNonexistentCollectionTest(unittest.TestCase):
    def test_returns_immediately_without_calling_recall_points(self):
        # Regression for PR #178 review (previously-suppressed finding): when
        # the collection doesn't exist yet, recall must return directly
        # rather than falling through to mb.recall_points, which does its
        # OWN separate collection_exists check -- a concurrent remember()
        # could create the collection under a different embedding model in
        # that gap, making recall_points proceed past its own check and hit
        # an opaque query_points failure instead of a clean result.
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = False
        with patch("memory_bank_mcp_server.QdrantClient", return_value=fake_client), \
             patch.object(_mb, "recall_points") as mock_recall_points:
            result = _run(_mbs.recall(query="anything"))
        self.assertIn("No memories found", result)
        mock_recall_points.assert_not_called()

    def test_does_not_construct_embedding_provider_when_collection_is_missing(self):
        # Regression for PR #178 review (second pass, previously-suppressed
        # finding): FastEmbedProvider's constructor eagerly loads/downloads
        # the ONNX model (TextEmbedding's default lazy_load=False) -- a
        # recall against a not-yet-existing collection must never pay for
        # that just to return "no results."
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = False
        with patch("memory_bank_mcp_server.QdrantClient", return_value=fake_client), \
             patch("memory_bank_mcp_server.FastEmbedProvider") as mock_provider_cls:
            _run(_mbs.recall(query="anything"))
        mock_provider_cls.assert_not_called()


class ToolRegistrationTest(unittest.TestCase):
    def test_registers_exactly_three_tools(self):
        from mcp_tool_introspect import tool_count
        self.assertEqual(tool_count(_mbs.mcp), 3)

    def test_logs_tool_count_to_stderr_not_stdout(self):
        captured_err = io.StringIO()
        captured_out = io.StringIO()
        import contextlib
        with redirect_stderr(captured_err), contextlib.redirect_stdout(captured_out):
            _mbs._log_registered_tool_count()
        self.assertIn("memory-bank", captured_err.getvalue())
        self.assertIn("3", captured_err.getvalue())
        self.assertEqual(captured_out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
