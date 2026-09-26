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
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock
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


# Issue #211 (Copilot review): most tests in this module call remember()/
# recall()/forget() with tracking left at its real default (True) and don't
# mock mev.record_memory_metric -- only MemoryMetricCounterTest below does.
# Without this module-wide redirect, every one of those calls falls through
# to the REAL metrics_lib.MetricsStore(), which resolves CLAUDE_RUNWAY_METRICS_DB
# via os.environ and, if unset, writes into the DEVELOPER'S OWN real
# ~/.claude/claude-runway/metrics.db -- confirmed live: running this suite
# once appended 45 real "memory-bank" rows to that file. setUpModule/
# tearDownModule redirect the env var to a throwaway temp path for the
# entire module's test run (not just one test class), so every remember/
# recall/forget call in this file -- mocked or not -- writes (if it writes
# at all) into a file nothing else reads, then discards it wholesale on
# teardown. Same tempfile-per-test-run approach test_memory_events_lib.py
# already uses for CLAUDE_RUNWAY_MEMORY_EVENTS_DB, just scoped to the whole
# module here rather than per-test-class, since the risk here is "any test
# that doesn't explicitly mock record_memory_metric," not one specific class.
_metrics_db_tmpdir: tempfile.TemporaryDirectory
_metrics_db_env_patch: "mock._patch_dict"


def setUpModule():
    global _metrics_db_tmpdir, _metrics_db_env_patch
    _metrics_db_tmpdir = tempfile.TemporaryDirectory()
    db_path = Path(_metrics_db_tmpdir.name) / "metrics.db"
    _metrics_db_env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": str(db_path)})
    _metrics_db_env_patch.start()


def tearDownModule():
    _metrics_db_env_patch.stop()
    _metrics_db_tmpdir.cleanup()


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


class MemoryMetricCounterTest(unittest.TestCase):
    """Issue #211: remember()/recall()/forget() all call
    mev.record_memory_metric(...) -- a plain per-call-attempt tally into
    metrics.db, entirely separate from mev.record_memory_event's own rich
    per-point memory-events.db log covered by MemoryEventLoggingTest above.
    Counts every call ATTEMPT, including a validation error / embedding
    mismatch / empty result, not just a successful remember/recall/forget."""

    def test_remember_counts_even_on_invalid_weight(self):
        # Regression-shaped: an invalid weight returns an Error before any
        # Qdrant/embedding-provider work, but the call still happened.
        with patch("memory_bank_mcp_server.QdrantClient") as mock_client_cls, \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True), \
             patch.object(_mbs.mev, "record_memory_metric") as mock_metric:
            result = _run(_mbs.remember(summary="s", description="d", kind="lesson", weight=-1.0))
        self.assertIn("Error", result)
        mock_metric.assert_called_once_with("remember", session_id=_mbs._SESSION_ID)
        mock_client_cls.assert_not_called()

    def test_remember_counts_on_embedding_mismatch(self):
        with patch("memory_bank_mcp_server.QdrantClient"), \
             patch("memory_bank_mcp_server.FastEmbedProvider"), \
             patch.object(_mb, "remember_point", new=AsyncMock(return_value=(None, None, "Error: mismatch"))), \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True), \
             patch.object(_mbs.mev, "record_memory_metric") as mock_metric:
            _run(_mbs.remember(summary="s", description="d", kind="lesson"))
        mock_metric.assert_called_once_with("remember", session_id=_mbs._SESSION_ID)

    def test_remember_does_not_count_when_tracking_disabled(self):
        with patch("memory_bank_mcp_server.QdrantClient"), \
             patch("memory_bank_mcp_server.FastEmbedProvider"), \
             patch.object(_mb, "remember_point", new=AsyncMock(return_value=("new-id", 1700000000.0, None))), \
             patch.object(_mbs.mev, "tracking_enabled", return_value=False), \
             patch.object(_mbs.mev, "record_memory_metric") as mock_metric:
            _run(_mbs.remember(summary="s", description="d", kind="lesson"))
        mock_metric.assert_not_called()

    def test_recall_counts_on_missing_collection(self):
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = False
        with patch("memory_bank_mcp_server.QdrantClient", return_value=fake_client), \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True), \
             patch.object(_mbs.mev, "record_memory_metric") as mock_metric:
            result = _run(_mbs.recall(query="anything"))
        self.assertIn("No memories found", result)
        mock_metric.assert_called_once_with("recall", session_id=_mbs._SESSION_ID)

    def test_recall_counts_once_even_with_multiple_hits(self):
        # Unlike record_memory_event (one row per hit), record_memory_metric
        # is a per-CALL tally -- exactly one call regardless of hit count.
        results = [
            {"id": "p1", "summary": "s1", "description": "d1", "kind": "lesson", "repo": "proj-a",
             "embedding_model": "m", "score": 0.9, "created_at": 111.0},
            {"id": "p2", "summary": "s2", "description": "d2", "kind": "idea", "repo": "proj-a",
             "embedding_model": "m", "score": 0.8, "created_at": 222.0},
        ]
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = True
        with patch("memory_bank_mcp_server.QdrantClient", return_value=fake_client), \
             patch("memory_bank_mcp_server.check_embedding_model_mismatch", return_value=None), \
             patch("memory_bank_mcp_server.FastEmbedProvider"), \
             patch.object(_mb, "recall_points", new=AsyncMock(return_value=results)), \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True), \
             patch.object(_mbs.mev, "record_memory_metric") as mock_metric:
            _run(_mbs.recall(query="anything"))
        mock_metric.assert_called_once_with("recall", session_id=_mbs._SESSION_ID)

    def test_recall_does_not_count_when_tracking_disabled(self):
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = False
        with patch("memory_bank_mcp_server.QdrantClient", return_value=fake_client), \
             patch.object(_mbs.mev, "tracking_enabled", return_value=False), \
             patch.object(_mbs.mev, "record_memory_metric") as mock_metric:
            _run(_mbs.recall(query="anything"))
        mock_metric.assert_not_called()

    def test_forget_counts_even_on_validation_error(self):
        # Neither point_id nor wipe_all -- a validation error, but the call
        # still happened and must still count.
        with patch.object(_mbs.mev, "tracking_enabled", return_value=True), \
             patch.object(_mbs.mev, "record_memory_metric") as mock_metric:
            result = _mbs.forget()
        self.assertIn("Error", result)
        mock_metric.assert_called_once_with("forget", session_id=_mbs._SESSION_ID)

    def test_forget_counts_on_ordinary_call(self):
        with patch("memory_bank_mcp_server.QdrantClient"), \
             patch.object(_mb, "forget_point", return_value="Deleted."), \
             patch.object(_mbs.mev, "tracking_enabled", return_value=True), \
             patch.object(_mbs.mev, "record_memory_metric") as mock_metric:
            _mbs.forget(point_id="some-id")
        mock_metric.assert_called_once_with("forget", session_id=_mbs._SESSION_ID)

    def test_forget_does_not_count_when_tracking_disabled(self):
        with patch.object(_mbs.mev, "tracking_enabled", return_value=False), \
             patch.object(_mbs.mev, "record_memory_metric") as mock_metric:
            _mbs.forget()
        mock_metric.assert_not_called()


class RememberWeightValidationTest(unittest.TestCase):
    """PR #199 review: a negative weight actively breaks recall_points'
    ranking invariant (weight=0 must be a true floor, weight>1 must only
    ever boost -- both assume weight>=0), so it's rejected at the tool
    boundary before any Qdrant/embedding-provider work happens."""

    def test_negative_weight_rejected_without_constructing_client(self):
        with patch("memory_bank_mcp_server.QdrantClient") as mock_client_cls, \
             patch("memory_bank_mcp_server.FastEmbedProvider") as mock_provider_cls, \
             patch.object(_mb, "remember_point") as mock_remember:
            result = _run(_mbs.remember(summary="s", description="d", kind="lesson", weight=-1.0))
        self.assertIn("Error", result)
        mock_client_cls.assert_not_called()
        mock_provider_cls.assert_not_called()
        mock_remember.assert_not_called()

    def test_positive_infinity_weight_rejected(self):
        # Regression for PR #199 review (second pass): float("inf") < 0 is
        # False in Python, so `inf` (reachable from ordinary JSON-RPC input,
        # e.g. the JSON literal 1e999 parses to inf) silently bypassed a
        # `weight < 0`-only check -- it must be caught by a finiteness check
        # too, not just negativity.
        with patch("memory_bank_mcp_server.QdrantClient") as mock_client_cls, \
             patch.object(_mb, "remember_point") as mock_remember:
            result = _run(_mbs.remember(summary="s", description="d", kind="lesson", weight=float("inf")))
        self.assertIn("Error", result)
        mock_client_cls.assert_not_called()
        mock_remember.assert_not_called()

    def test_nan_weight_rejected(self):
        # Regression for PR #199 review (second pass): float("nan") < 0 is
        # ALSO False in Python (every comparison with NaN is False), so NaN
        # silently bypassed a `weight < 0`-only check too, and would have
        # made sort() ordering in recall_points undefined.
        with patch("memory_bank_mcp_server.QdrantClient") as mock_client_cls, \
             patch.object(_mb, "remember_point") as mock_remember:
            result = _run(_mbs.remember(summary="s", description="d", kind="lesson", weight=float("nan")))
        self.assertIn("Error", result)
        mock_client_cls.assert_not_called()
        mock_remember.assert_not_called()

    def test_zero_weight_is_allowed(self):
        with patch("memory_bank_mcp_server.QdrantClient"), \
             patch("memory_bank_mcp_server.FastEmbedProvider"), \
             patch.object(_mb, "remember_point", new=AsyncMock(return_value=("new-id", 1700000000.0, None))):
            result = _run(_mbs.remember(summary="s", description="d", kind="lesson", weight=0.0))
        self.assertIn("Remembered", result)


class RememberWeightPassthroughTest(unittest.TestCase):
    """Issue #177: remember()'s weight param must reach mb.remember_point
    unchanged, defaulting to mb.DEFAULT_WEIGHT (1.0) when not given."""

    def test_default_weight_passed_through(self):
        with patch("memory_bank_mcp_server.QdrantClient"), \
             patch("memory_bank_mcp_server.FastEmbedProvider"), \
             patch.object(_mb, "remember_point", new=AsyncMock(return_value=("new-id", 1700000000.0, None))) as mock_remember, \
             patch.object(_mbs.mev, "tracking_enabled", return_value=False):
            result = _run(_mbs.remember(summary="s", description="d", kind="lesson"))
        _, kwargs = mock_remember.call_args
        self.assertEqual(kwargs["weight"], _mb.DEFAULT_WEIGHT)
        self.assertIn(f"weight={_mb.DEFAULT_WEIGHT}", result)

    def test_explicit_weight_passed_through(self):
        with patch("memory_bank_mcp_server.QdrantClient"), \
             patch("memory_bank_mcp_server.FastEmbedProvider"), \
             patch.object(_mb, "remember_point", new=AsyncMock(return_value=("new-id", 1700000000.0, None))) as mock_remember, \
             patch.object(_mbs.mev, "tracking_enabled", return_value=False):
            result = _run(_mbs.remember(summary="s", description="d", kind="lesson", weight=2.0))
        _, kwargs = mock_remember.call_args
        self.assertEqual(kwargs["weight"], 2.0)
        self.assertIn("weight=2.0", result)


class SessionIdDelegatesToSessionIdLibTest(unittest.TestCase):
    """Issue #214: _SESSION_ID must come from libs/session_id_lib.py's
    SessionIdStrategy.PROXY (issue #198) rather than this module minting its
    own uuid.uuid4().hex -- pure dedup, no behavior change. Asserting on the
    exact cached value (not just "looks like a uuid hex string") is what
    actually proves delegation happened: session_id_lib's PROXY strategy
    caches one uuid.uuid4().hex per process (see its own module docstring),
    so a second call to session_id(PROXY) within this same test process must
    return the SAME value _mbs._SESSION_ID was set to at import time -- a
    module that still generated its own independent uuid would fail this
    even though both values are equally uuid-hex-shaped."""

    def test_session_id_matches_shared_proxy_strategy(self):
        import session_id_lib as sid
        self.assertEqual(_mbs._SESSION_ID, sid.session_id(sid.SessionIdStrategy.PROXY))

    def test_session_id_is_a_uuid4_hex_string(self):
        # Same shape the old uuid.uuid4().hex call produced -- confirms the
        # switch didn't change what downstream memory_events_lib rows store.
        self.assertRegex(_mbs._SESSION_ID, r"^[0-9a-f]{32}$")


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
