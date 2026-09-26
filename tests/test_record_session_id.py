#!/usr/bin/env python3
"""Tests for hooks/record_session_id.py (issue #198).

Stdlib-only (unittest, no pytest) and no disk beyond a temp dir -- redirects
session_id_lib's own directory resolution the same way
tests/test_session_id_lib.py does, following the same mock.patch.object
pattern test_redirect_webfetch_to_fetch_url.py/test_session_end_savings.py
use for hook scripts.

    .venv/bin/python -m unittest discover -s tests
"""

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import record_session_id as hook  # noqa: E402
import session_id_lib  # noqa: E402


def _run_main(stdin_text):
    buf = io.StringIO()
    with mock.patch("sys.stdin", io.StringIO(stdin_text)):
        with redirect_stdout(buf):
            try:
                hook.main()
            except SystemExit:
                pass
    return buf.getvalue()


_POST_TOOL_USE_PAYLOAD = json.dumps({
    "hook_event_name": "PostToolUse",
    "session_id": "sess-1",
    "cwd": "/repos/my-project",
    "tool_name": "Bash",
    "tool_response": {"stdout": "ok"},
})

_SESSION_END_PAYLOAD = json.dumps({
    "hook_event_name": "SessionEnd",
    "session_id": "sess-1",
    "cwd": "/repos/my-project",
    "reason": "exit",
})


class RecordSessionIdTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.sessions_dir = Path(self._tmpdir.name) / "sessions"
        self._sessions_patch = mock.patch.object(session_id_lib, "_sessions_dir", return_value=self.sessions_dir)
        self._sessions_patch.start()
        self.addCleanup(self._sessions_patch.stop)


class PostToolUseBehavior(RecordSessionIdTestCase):
    def test_writes_no_stdout(self):
        # This hook never touches tool output -- no hookSpecificOutput at all.
        output = _run_main(_POST_TOOL_USE_PAYLOAD)
        self.assertEqual(output, "")

    def test_refreshes_own_marker(self):
        _run_main(_POST_TOOL_USE_PAYLOAD)
        marker = session_id_lib._shadow_marker_path("sess-1")
        self.assertTrue(marker.exists())
        data = json.loads(marker.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(data["project"], "/repos/my-project")

    def test_overwrite_behavior_updates_mtime_and_content(self):
        _run_main(json.dumps({
            "hook_event_name": "PostToolUse", "session_id": "sess-1",
            "cwd": "/repos/proj-a", "tool_name": "Bash",
        }))
        marker = session_id_lib._shadow_marker_path("sess-1")
        first_mtime = marker.stat().st_mtime
        time.sleep(0.01)
        _run_main(json.dumps({
            "hook_event_name": "PostToolUse", "session_id": "sess-1",
            "cwd": "/repos/proj-b", "tool_name": "Bash",
        }))
        self.assertGreaterEqual(marker.stat().st_mtime, first_mtime)
        lines = marker.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)  # overwritten, not appended
        self.assertEqual(json.loads(lines[0])["project"], "/repos/proj-b")

    def test_missing_session_id_is_a_silent_no_op(self):
        payload = json.dumps({"hook_event_name": "PostToolUse", "cwd": "/repos/my-project", "tool_name": "Bash"})
        output = _run_main(payload)
        self.assertEqual(output, "")
        self.assertFalse(self.sessions_dir.exists() and any(self.sessions_dir.iterdir()))

    def test_missing_cwd_still_exits_cleanly_without_writing_a_marker(self):
        payload = json.dumps({"hook_event_name": "PostToolUse", "session_id": "sess-1", "tool_name": "Bash"})
        _run_main(payload)
        self.assertFalse(session_id_lib._shadow_marker_path("sess-1").exists())

    def test_invalid_json_stdin_no_output(self):
        self.assertEqual(_run_main("not valid json {{"), "")

    def test_no_tool_name_but_no_event_name_treated_as_post_tool_use(self):
        # Fallback discriminator: absent hook_event_name + tool_name present
        # -> treated as PostToolUse (refresh, not delete).
        payload = json.dumps({"session_id": "sess-1", "cwd": "/repos/my-project", "tool_name": "Bash"})
        _run_main(payload)
        self.assertTrue(session_id_lib._shadow_marker_path("sess-1").exists())


class SessionEndBehavior(RecordSessionIdTestCase):
    def test_writes_no_stdout(self):
        output = _run_main(_SESSION_END_PAYLOAD)
        self.assertEqual(output, "")

    def test_deletes_own_marker_on_session_end(self):
        session_id_lib.record_shadow_marker("sess-1", project="/repos/my-project")
        self.assertTrue(session_id_lib._shadow_marker_path("sess-1").exists())
        _run_main(_SESSION_END_PAYLOAD)
        self.assertFalse(session_id_lib._shadow_marker_path("sess-1").exists())

    def test_no_marker_to_delete_is_a_silent_no_op(self):
        output = _run_main(_SESSION_END_PAYLOAD)
        self.assertEqual(output, "")

    def test_missing_session_id_is_a_silent_no_op(self):
        payload = json.dumps({"hook_event_name": "SessionEnd", "cwd": "/repos/my-project", "reason": "exit"})
        output = _run_main(payload)
        self.assertEqual(output, "")

    def test_fallback_discriminator_no_event_name_no_tool_name(self):
        # Absent hook_event_name + no tool_name -> treated as SessionEnd.
        session_id_lib.record_shadow_marker("sess-1", project="/repos/my-project")
        payload = json.dumps({"session_id": "sess-1", "cwd": "/repos/my-project", "reason": "exit"})
        _run_main(payload)
        self.assertFalse(session_id_lib._shadow_marker_path("sess-1").exists())


class SelfRefreshBeforeSweepOrdering(RecordSessionIdTestCase):
    """The core safety contract (issue #198's proposal, item 4): a live
    session's own PostToolUse call must refresh its marker BEFORE any sweep
    runs, so it can never observe (and delete) itself as stale."""

    def test_own_marker_survives_a_sweep_triggered_by_the_same_call(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS": "0.0001"}):
            with mock.patch.object(session_id_lib, "should_sweep", return_value=True), \
                 mock.patch.object(session_id_lib, "mark_swept"):
                _run_main(_POST_TOOL_USE_PAYLOAD)
        # Even with an aggressively short TTL and a forced sweep this same
        # call, the marker this same call just wrote must still be present --
        # record_shadow_marker() runs before sweep_stale_shadow_markers().
        self.assertTrue(session_id_lib._shadow_marker_path("sess-1").exists())

    def test_sweep_is_rate_limited_via_should_sweep(self):
        with mock.patch.object(session_id_lib, "should_sweep", return_value=False) as mock_should_sweep, \
             mock.patch.object(session_id_lib, "sweep_stale_shadow_markers") as mock_sweep:
            _run_main(_POST_TOOL_USE_PAYLOAD)
        mock_should_sweep.assert_called_once()
        mock_sweep.assert_not_called()


class UnexpectedErrorsFailOpen(unittest.TestCase):
    """Any exception from session_id_lib (a disk-full/permissions error on
    the sessions directory, etc.) must never surface as a broken tool call
    or a broken session shutdown -- main()'s top-level guard swallows it and
    still exits 0 with no stdout, the same fail-open philosophy
    compress_bash_output.py's own top-level guard uses."""

    def test_record_shadow_marker_raising_still_exits_silently(self):
        with mock.patch.object(hook.session_id_lib, "record_shadow_marker", side_effect=OSError("disk full")):
            output = _run_main(_POST_TOOL_USE_PAYLOAD)
        self.assertEqual(output, "")

    def test_delete_shadow_marker_raising_still_exits_silently(self):
        with mock.patch.object(hook.session_id_lib, "delete_shadow_marker", side_effect=OSError("disk full")):
            output = _run_main(_SESSION_END_PAYLOAD)
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
