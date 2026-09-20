#!/usr/bin/env python3
"""Tests for hooks/session_end_savings.py.

Stdlib-only (unittest, no pytest) and no disk/DB/network — all savings_ledger
calls are stubbed at the hook's module-level reference, following the same
mock.patch.object pattern as test_redirect_webfetch_to_fetch_url.py.

Issue #162: the hook wrapped systemMessage inside hookSpecificOutput, which is
not a valid key for SessionEnd — Claude Code rejects the entire payload with a
schema validation error. The fix emits {"systemMessage": summary} at the JSON
root. These tests pin that output shape and verify the three silent-exit paths
produce no stdout.

    .venv/bin/python -m unittest discover -s tests
"""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))

import session_end_savings as hook  # noqa: E402


_PAYLOAD = json.dumps({"session_id": "abc123", "cwd": "/some/project"})
_SUMMARY = "ClaudeRunway · Token Savings\n\nThis session · my-project\n  Context tokens avoided  ~1.8k"


def _run_main(stdin_text=_PAYLOAD):
    buf = io.StringIO()
    with mock.patch("sys.stdin", io.StringIO(stdin_text)):
        with redirect_stdout(buf):
            try:
                hook.main()
            except SystemExit:
                pass
    return buf.getvalue()


class OutputShape(unittest.TestCase):
    """systemMessage must appear at the JSON root — not inside hookSpecificOutput."""

    def setUp(self):
        patchers = [
            mock.patch.object(hook.savings_ledger, "tracking_enabled", return_value=True),
            mock.patch.object(hook.savings_ledger, "read_session_events", return_value=[object()]),
            mock.patch.object(hook.savings_ledger, "project_name_from_cwd", return_value="my-project"),
            mock.patch.object(hook.savings_ledger, "get_schema_overhead_tokens", return_value=0),
            mock.patch.object(hook.savings_ledger, "finalize_session", return_value={}),
            mock.patch.object(hook.savings_ledger, "query_project_summary", return_value={}),
            mock.patch.object(hook.savings_ledger, "format_simple_view", return_value=_SUMMARY),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_systemmessage_at_root(self):
        data = json.loads(_run_main())
        self.assertIn("systemMessage", data)
        self.assertEqual(data["systemMessage"], _SUMMARY)

    def test_no_hookspecificoutput(self):
        # Regression for #162: hookSpecificOutput is not valid for SessionEnd;
        # Claude Code rejects any payload that contains it.
        data = json.loads(_run_main())
        self.assertNotIn("hookSpecificOutput", data)

    def test_only_systemmessage_key(self):
        # The root object must contain exactly one key — no extras smuggled in.
        data = json.loads(_run_main())
        self.assertEqual(list(data.keys()), ["systemMessage"])


class SilentExitPaths(unittest.TestCase):
    """Hook must produce no stdout when it has nothing to report."""

    def test_tracking_disabled_no_output(self):
        with mock.patch.object(hook.savings_ledger, "tracking_enabled", return_value=False):
            self.assertEqual(_run_main(), "")

    def test_missing_session_id_no_output(self):
        payload = json.dumps({"cwd": "/some/project"})  # no session_id key
        with mock.patch.object(hook.savings_ledger, "tracking_enabled", return_value=True):
            self.assertEqual(_run_main(payload), "")

    def test_empty_events_no_output(self):
        with (
            mock.patch.object(hook.savings_ledger, "tracking_enabled", return_value=True),
            mock.patch.object(hook.savings_ledger, "read_session_events", return_value=[]),
        ):
            self.assertEqual(_run_main(), "")

    def test_invalid_json_stdin_no_output(self):
        # Malformed stdin must not crash the hook — it exits silently.
        self.assertEqual(_run_main("not valid json {{"), "")


_PAYLOAD_WITH_TRANSCRIPT = json.dumps({
    "session_id": "abc123",
    "cwd": "/some/project",
    "transcript_path": "/fake/transcript.jsonl",
})


class TranscriptParsing(unittest.TestCase):
    """Hook calls parse_transcript_token_counts and passes result to
    finalize_session when CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS is enabled."""

    def _run_with_patched_ledger(self, env, finalize_kwargs_capture, transcript_result):
        """Run main() with all ledger calls stubbed; capture finalize_session kwargs."""
        captured = {}

        def _fake_finalize(session_id, project, overhead_tokens=0, delete_jsonl=True, actual_tokens=None):
            captured["actual_tokens"] = actual_tokens
            return {}

        # Start from the real environment minus the one var under test, so a
        # developer who exports CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS in their
        # shell (the documented way to enable the feature) doesn't leak it
        # into the "env var not set" case. Each test then supplies exactly the
        # value it means to via `env`.
        patched_env = {k: v for k, v in os.environ.items() if k != "CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS"}
        patched_env.update(env)

        patchers = [
            mock.patch.dict(os.environ, patched_env, clear=True),
            mock.patch.object(hook.savings_ledger, "tracking_enabled", return_value=True),
            mock.patch.object(hook.savings_ledger, "read_session_events", return_value=[object()]),
            mock.patch.object(hook.savings_ledger, "project_name_from_cwd", return_value="proj"),
            mock.patch.object(hook.savings_ledger, "get_schema_overhead_tokens", return_value=0),
            mock.patch.object(hook.savings_ledger, "finalize_session", side_effect=_fake_finalize),
            mock.patch.object(hook.savings_ledger, "query_project_summary", return_value={}),
            mock.patch.object(hook.savings_ledger, "format_simple_view", return_value=_SUMMARY),
            mock.patch.object(
                hook.savings_ledger, "parse_transcript_token_counts",
                return_value=transcript_result
            ),
        ]
        for p in patchers:
            p.start()
        try:
            _run_main(_PAYLOAD_WITH_TRANSCRIPT)
        finally:
            for p in patchers:
                p.stop()
        finalize_kwargs_capture.update(captured)

    def test_actual_tokens_passed_when_env_var_set(self):
        fake_counts = {"input": 10, "output": 5, "cache_read": 200, "cache_write": 30}
        captured = {}
        self._run_with_patched_ledger(
            {"CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS": "1"},
            captured,
            fake_counts,
        )
        self.assertEqual(captured["actual_tokens"], fake_counts)

    def test_actual_tokens_none_when_env_var_not_set(self):
        captured = {}
        self._run_with_patched_ledger(
            {},  # env var absent
            captured,
            {"input": 10, "output": 5, "cache_read": 200, "cache_write": 30},
        )
        self.assertIsNone(captured["actual_tokens"])

    def test_actual_tokens_none_when_transcript_parse_returns_none(self):
        # parse_transcript_token_counts returned None (e.g. file unreadable) --
        # finalize_session must still receive None, not an exception.
        captured = {}
        self._run_with_patched_ledger(
            {"CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS": "1"},
            captured,
            None,  # transcript parse failed
        )
        self.assertIsNone(captured["actual_tokens"])

    def test_parse_transcript_enabled_accepts_true_yes(self):
        for val in ("1", "true", "True", "TRUE", "yes", "YES"):
            with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS": val}):
                self.assertTrue(hook._parse_transcript_enabled(), f"Failed for value {val!r}")

    def test_parse_transcript_disabled_for_other_values(self):
        for val in ("0", "false", "no", "", "off"):
            with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS": val}):
                self.assertFalse(hook._parse_transcript_enabled(), f"Should be disabled for {val!r}")


if __name__ == "__main__":
    unittest.main()
