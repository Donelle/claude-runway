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
from unittest.mock import AsyncMock, MagicMock, patch

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


class CollectionOverrideRemovedTest(unittest.TestCase):
    """Regression for issue #188: remember/recall/forget used to accept a
    caller-controlled `collection` kwarg with no validation, letting a call
    write/read/delete memory-bank points in a collection other than the one
    configured via MEMORY_BANK_COLLECTION -- bypassing index_repo/sync_repo's
    reserved-collection guard, which only ever checks the configured name.
    The fix removes the parameter entirely rather than validating it, so a
    caller-supplied `collection` must now be rejected at the call boundary."""

    def test_remember_rejects_collection_kwarg(self):
        with self.assertRaises(TypeError):
            _mbs.remember(summary="s", description="d", kind="k", collection="other")

    def test_recall_rejects_collection_kwarg(self):
        with self.assertRaises(TypeError):
            _mbs.recall(query="anything", collection="other")

    def test_forget_rejects_collection_kwarg(self):
        with self.assertRaises(TypeError):
            _mbs.forget(point_id="some-id", collection="other")


class MemoryEventLoggingTest(unittest.TestCase):
    """Issue #179: remember()/recall() log a memory-events row via
    memory_events_lib, guarded by its own tracking_enabled() switch --
    entirely separate from the point_id/description content itself."""

    def test_remember_logs_one_event_when_tracking_enabled(self):
        with patch("memory_bank_mcp_server.QdrantClient"), \
             patch("memory_bank_mcp_server.FastEmbedProvider"), \
             patch.object(_mb, "remember_point", new=AsyncMock(return_value=("new-id", 1700000000.0, None))), \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True), \
             patch.object(_mbs.mev, "record_memory_event") as mock_record:
            result = _run(_mbs.remember(summary="s", description="d", kind="lesson"))
        self.assertIn("Remembered", result)
        mock_record.assert_called_once()
        _, kwargs = mock_record.call_args
        self.assertEqual(kwargs["event_type"], "remember")
        self.assertEqual(kwargs["point_id"], "new-id")
        self.assertEqual(kwargs["kind"], "lesson")
        # Issue #179, PR #197 review: summary_created_at must be the EXACT
        # value remember_point stamped/returned, not a re-sampled time.time().
        self.assertEqual(kwargs["summary_created_at"], 1700000000.0)

    def test_remember_does_not_log_when_tracking_disabled(self):
        with patch("memory_bank_mcp_server.QdrantClient"), \
             patch("memory_bank_mcp_server.FastEmbedProvider"), \
             patch.object(_mb, "remember_point", new=AsyncMock(return_value=("new-id", 1700000000.0, None))), \
             patch.object(_mbs.mev, "tracking_enabled", return_value=False), \
             patch.object(_mbs.mev, "record_memory_event") as mock_record:
            _run(_mbs.remember(summary="s", description="d", kind="lesson"))
        mock_record.assert_not_called()

    def test_remember_does_not_log_on_embedding_mismatch(self):
        # A mismatch means nothing was actually written -- there is no real
        # memory event to log.
        with patch("memory_bank_mcp_server.QdrantClient"), \
             patch("memory_bank_mcp_server.FastEmbedProvider"), \
             patch.object(_mb, "remember_point", new=AsyncMock(return_value=(None, None, "Error: mismatch"))), \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True), \
             patch.object(_mbs.mev, "record_memory_event") as mock_record:
            result = _run(_mbs.remember(summary="s", description="d", kind="lesson"))
        self.assertIn("Error", result)
        mock_record.assert_not_called()

    def test_remember_advances_turn_even_on_embedding_mismatch(self):
        # Regression for PR #197 review: turn must be consumed for every
        # invocation, including ones that end up not logging anything --
        # otherwise a later successful call's turn number understates how
        # many memory-bank calls actually happened this session.
        with patch("memory_bank_mcp_server.QdrantClient"), \
             patch("memory_bank_mcp_server.FastEmbedProvider"), \
             patch.object(_mb, "remember_point", new=AsyncMock(return_value=(None, None, "Error: mismatch"))), \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True):
            turn_before = _mbs._next_turn()
            _run(_mbs.remember(summary="s", description="d", kind="lesson"))
            turn_after = _mbs._next_turn()
        # Two _next_turn() calls plus one remember() call in between should
        # advance the counter by 3 total steps from turn_before to turn_after.
        self.assertEqual(turn_after, turn_before + 2)

    def _recall_setup(self, results):
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = True
        return patch("memory_bank_mcp_server.QdrantClient", return_value=fake_client), \
            patch("memory_bank_mcp_server.check_embedding_model_mismatch", return_value=None), \
            patch("memory_bank_mcp_server.FastEmbedProvider"), \
            patch.object(_mb, "recall_points", new=AsyncMock(return_value=results))

    def test_recall_logs_one_event_per_hit_sharing_a_turn(self):
        results = [
            {"id": "p1", "summary": "s1", "description": "d1", "kind": "lesson", "repo": "proj-a",
             "embedding_model": "m", "score": 0.9, "created_at": 111.0},
            {"id": "p2", "summary": "s2", "description": "d2", "kind": "idea", "repo": "proj-a",
             "embedding_model": "m", "score": 0.8, "created_at": 222.0},
        ]
        p1, p2, p3, p4 = self._recall_setup(results)
        with p1, p2, p3, p4, \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True), \
             patch.object(_mbs.mev, "record_memory_event") as mock_record:
            _run(_mbs.recall(query="anything"))
        self.assertEqual(mock_record.call_count, 2)
        calls = mock_record.call_args_list
        turns = {c.kwargs["turn"] for c in calls}
        self.assertEqual(len(turns), 1)  # both hits share the same turn
        self.assertEqual(calls[0].kwargs["event_type"], "recall")
        self.assertEqual(calls[0].kwargs["point_id"], "p1")
        self.assertEqual(calls[0].kwargs["summary_created_at"], 111.0)
        self.assertEqual(calls[1].kwargs["point_id"], "p2")

    def test_recall_does_not_log_when_tracking_disabled(self):
        results = [{"id": "p1", "summary": "s1", "description": "d1", "kind": "lesson",
                    "repo": "proj-a", "embedding_model": "m", "score": 0.9, "created_at": 111.0}]
        p1, p2, p3, p4 = self._recall_setup(results)
        with p1, p2, p3, p4, \
             patch.object(_mbs.mev, "tracking_enabled", return_value=False), \
             patch.object(_mbs.mev, "record_memory_event") as mock_record:
            _run(_mbs.recall(query="anything"))
        mock_record.assert_not_called()

    def test_recall_does_not_log_when_no_results(self):
        p1, p2, p3, p4 = self._recall_setup([])
        with p1, p2, p3, p4, \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True), \
             patch.object(_mbs.mev, "record_memory_event") as mock_record:
            _run(_mbs.recall(query="anything"))
        mock_record.assert_not_called()

    def test_recall_advances_turn_even_when_no_results(self):
        # Regression for PR #197 review: an empty recall still consumed a
        # turn number -- it was a real memory-bank call, even though nothing
        # gets logged for it.
        p1, p2, p3, p4 = self._recall_setup([])
        with p1, p2, p3, p4, \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True):
            turn_before = _mbs._next_turn()
            _run(_mbs.recall(query="anything"))
            turn_after = _mbs._next_turn()
        self.assertEqual(turn_after, turn_before + 2)


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
