#!/usr/bin/env python3
"""Tests for libs/session_id_lib.py (issue #198) -- all four
SessionIdStrategy variants (HOOK_PAYLOAD, SHADOW_FILE, TRANSCRIPT_SCAN,
PROXY) behind the unified session_id() entry point.

Stdlib-only (unittest, no pytest) and no network -- everything here uses
temp directories (via mock.patch.object on the module's own private
directory-resolution functions, the same class of indirection
tests/test_savings_ledger.py achieves via env-var redirection) and synthetic
hook payloads, so it needs neither Qdrant nor LM Studio nor a real Claude
Code session:

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import session_id_lib as L  # noqa: E402


class SessionIdLibTestCase(unittest.TestCase):
    """Common temp-dir setup: redirects _sessions_dir()/_transcript_projects_dir()
    (and resets the PROXY module-level cache) so nothing here ever touches the
    real ~/.claude/claude-runway/sessions or ~/.claude/projects directories."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.sessions_dir = Path(self._tmpdir.name) / "sessions"
        self.projects_dir = Path(self._tmpdir.name) / "projects"

        self._sessions_patch = mock.patch.object(L, "_sessions_dir", return_value=self.sessions_dir)
        self._projects_patch = mock.patch.object(L, "_transcript_projects_dir", return_value=self.projects_dir)
        self._sessions_patch.start()
        self._projects_patch.start()
        self.addCleanup(self._sessions_patch.stop)
        self.addCleanup(self._projects_patch.stop)

        # Reset the process-lifetime proxy cache between tests -- it's a
        # module-level global by design (see _proxy_session_id's docstring),
        # so a prior test's cached UUID must not leak into this one.
        L._process_proxy_id = None

        self._env_patch = mock.patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)
        os.environ.pop("CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS", None)


class HookPayloadStrategy(SessionIdLibTestCase):
    def test_extracts_session_id_from_payload(self):
        result = L.session_id(L.SessionIdStrategy.HOOK_PAYLOAD, hook_payload={"session_id": "abc-123"})
        self.assertEqual(result, "abc-123")

    def test_none_payload_returns_none(self):
        self.assertIsNone(L.session_id(L.SessionIdStrategy.HOOK_PAYLOAD, hook_payload=None))

    def test_empty_payload_returns_none(self):
        self.assertIsNone(L.session_id(L.SessionIdStrategy.HOOK_PAYLOAD, hook_payload={}))

    def test_missing_key_returns_none(self):
        self.assertIsNone(L.session_id(L.SessionIdStrategy.HOOK_PAYLOAD, hook_payload={"cwd": "/x"}))

    def test_non_string_value_returns_none(self):
        self.assertIsNone(L.session_id(L.SessionIdStrategy.HOOK_PAYLOAD, hook_payload={"session_id": None}))


class ShadowFileStrategy(SessionIdLibTestCase):
    def test_no_directory_returns_none(self):
        self.assertIsNone(L.session_id(L.SessionIdStrategy.SHADOW_FILE))

    def test_no_markers_returns_none(self):
        self.sessions_dir.mkdir(parents=True)
        self.assertIsNone(L.session_id(L.SessionIdStrategy.SHADOW_FILE))

    def test_omitted_project_never_matches(self):
        # SHADOW_FILE never evaluates `project` as a path and never falls
        # back to os.getcwd() -- omitting it means there is nothing to
        # compare a marker's basename against, so it always returns None,
        # even when a marker exists that an old cwd-fallback would have
        # matched. (Contrast TRANSCRIPT_SCAN, which still falls back to
        # os.getcwd() via `_resolve_project` -- see that strategy's own
        # `test_no_project_falls_back_to_cwd`.)
        L.record_shadow_marker("sess-a", project="/repos/proj-a")
        result = L.session_id(L.SessionIdStrategy.SHADOW_FILE)
        self.assertIsNone(result)

    def test_never_calls_getcwd(self):
        # Regression guard: SHADOW_FILE must never resolve `project` as a
        # path at all, so os.getcwd() should never be invoked for it, even
        # when `project` is omitted -- unlike TRANSCRIPT_SCAN.
        L.record_shadow_marker("sess-a", project="/repos/proj-a")
        with mock.patch.object(os, "getcwd", side_effect=AssertionError("must not be called")):
            result = L.session_id(L.SessionIdStrategy.SHADOW_FILE)
        self.assertIsNone(result)  # project omitted -> never matches, per above

    def test_project_filter_matches_by_bare_name(self):
        # `project` is compared AS GIVEN (a bare name) against a marker's
        # own recorded project reduced to its basename -- record_shadow_marker
        # happens to store an absolute path, but the caller never needs one.
        L.record_shadow_marker("sess-a", project="/repos/proj-a")
        L.record_shadow_marker("sess-b", project="/repos/proj-b")
        # sess-b is more recently modified, but the caller wants proj-a's session.
        result = L.session_id(L.SessionIdStrategy.SHADOW_FILE, project="proj-a")
        self.assertEqual(result, "sess-a")

    def test_project_filter_with_no_match_returns_none(self):
        L.record_shadow_marker("sess-a", project="/repos/proj-a")
        result = L.session_id(L.SessionIdStrategy.SHADOW_FILE, project="proj-nonexistent")
        self.assertIsNone(result)

    def test_full_path_project_no_longer_matches(self):
        # Deliberate behavior change: since `project` is now compared AS
        # GIVEN (never reduced), passing a full path instead of the bare
        # name no longer matches, even though the stored marker's own path
        # shares that basename -- `project` is a name for this strategy,
        # not a path, full stop.
        L.record_shadow_marker("sess-a", project="/repos/proj-a")
        result = L.session_id(L.SessionIdStrategy.SHADOW_FILE, project="/somewhere/proj-a")
        self.assertIsNone(result)

    def test_invalid_utf8_marker_content_fails_open_instead_of_raising(self):
        # readline() raises UnicodeDecodeError (a ValueError subclass, NOT
        # an OSError) on invalid UTF-8 bytes -- confirmed this used to
        # escape _read_shadow_marker_project's except clause entirely
        # instead of being treated as just another malformed-marker case.
        self.sessions_dir.mkdir(parents=True)
        marker = self.sessions_dir / "session_bad-utf8.jsonl"
        with open(marker, "wb") as f:
            f.write(b"\xff\xfe\x00invalid utf8 bytes\n")

        result = L.session_id(L.SessionIdStrategy.SHADOW_FILE, project="/repos/proj-a")

        self.assertIsNone(result)

    def test_marker_write_is_atomic_no_leftover_temp_file(self):
        L.record_shadow_marker("sess-atomic", project="/repos/proj-a")
        marker = L._shadow_marker_path("sess-atomic")
        self.assertTrue(marker.exists())
        leftover = [p for p in self.sessions_dir.iterdir() if p.name != marker.name]
        self.assertEqual(leftover, [])
        content = json.loads(marker.read_text(encoding="utf-8").strip())
        self.assertEqual(content, {"project": "/repos/proj-a"})

    def test_marker_write_failure_cleans_up_temp_file_and_propagates(self):
        # record_shadow_marker() is a hook-side writer, not a best-effort
        # lookup -- hooks/record_session_id.py's own top-level guard is
        # what fails open here, so this is allowed to raise. What it must
        # NOT do is leave a stray .tmp file behind, or (per the atomic-write
        # fix) ever have replaced the real marker with partial content.
        with mock.patch.object(L.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                L.record_shadow_marker("sess-fail", project="/repos/proj-a")

        self.assertEqual(list(self.sessions_dir.glob("*.tmp")), [])
        self.assertFalse(L._shadow_marker_path("sess-fail").exists())

    def test_non_dict_marker_content_fails_open_instead_of_raising(self):
        # A marker can be syntactically valid JSON without being the object
        # this module writes (corrupted/truncated write, or something else
        # landing at this path) -- confirmed live this used to raise
        # AttributeError from `data.get(...)` on a non-dict `data` instead of
        # failing toward None like every other malformed-marker case.
        self.sessions_dir.mkdir(parents=True)
        marker = self.sessions_dir / "session_sess-bad.jsonl"
        marker.write_text("null\n", encoding="utf-8")

        result = L.session_id(L.SessionIdStrategy.SHADOW_FILE, project="/repos/proj-a")

        self.assertIsNone(result)

    def test_marker_overwrites_not_appends(self):
        L.record_shadow_marker("sess-1", project="/repos/proj-a")
        L.record_shadow_marker("sess-1", project="/repos/proj-b")
        path = L._shadow_marker_path("sess-1")
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["project"], "/repos/proj-b")

    def test_session_id_with_path_unsafe_characters_is_sanitized(self):
        # Same defensive sanitization as savings_ledger._sanitize_session_id --
        # a malformed/adversarial session_id must not escape the sessions dir.
        L.record_shadow_marker("../../../etc/evil", project="/repos/proj-a")
        marker_path = L._shadow_marker_path("../../../etc/evil")
        self.assertTrue(str(marker_path.resolve()).startswith(str(self.sessions_dir.resolve())))

    def test_delete_shadow_marker_removes_file(self):
        L.record_shadow_marker("sess-1", project="/repos/proj-a")
        self.assertTrue(L._shadow_marker_path("sess-1").exists())
        L.delete_shadow_marker("sess-1")
        self.assertFalse(L._shadow_marker_path("sess-1").exists())

    def test_delete_shadow_marker_missing_file_is_a_no_op(self):
        L.delete_shadow_marker("never-existed")  # must not raise

    def test_a_savings_ledger_style_bare_jsonl_is_never_touched(self):
        # SHADOW_FILE only ever globs session_*.jsonl -- a bare
        # <session_id>.jsonl (savings_ledger's own event log format) living
        # in the same directory must never be picked up or deleted.
        self.sessions_dir.mkdir(parents=True)
        bare = self.sessions_dir / "some-session-id.jsonl"
        bare.write_text('{"tool": "hook:Bash"}\n', encoding="utf-8")
        self.assertIsNone(L.session_id(L.SessionIdStrategy.SHADOW_FILE))
        deleted = L.sweep_stale_shadow_markers(now=time.time() + 10_000_000)
        self.assertEqual(deleted, 0)
        self.assertTrue(bare.exists())


class ShadowFileSweep(SessionIdLibTestCase):
    def test_sweep_deletes_markers_older_than_ttl(self):
        L.record_shadow_marker("sess-old", project="/repos/proj-a")
        old_path = L._shadow_marker_path("sess-old")
        # Derived from the real constant (not hand-typed) so this doesn't
        # silently go stale the next time the default TTL changes -- same
        # class of drift EVALUATION.md already flags for hardcoded numbers
        # (issues #22/#23).
        very_old = time.time() - ((L._DEFAULT_SESSION_MARKER_TTL_HOURS + 1) * 3600)
        os.utime(old_path, (very_old, very_old))
        L.record_shadow_marker("sess-fresh", project="/repos/proj-a")

        deleted = L.sweep_stale_shadow_markers()
        self.assertEqual(deleted, 1)
        self.assertFalse(old_path.exists())
        self.assertTrue(L._shadow_marker_path("sess-fresh").exists())

    def test_sweep_skips_marker_refreshed_between_the_two_stat_calls(self):
        # The staleness check and the unlink are two separate syscalls --
        # if a DIFFERENT process refreshes this exact marker in the gap
        # (simulated here via a second stat() call returning a fresh
        # mtime), the re-check immediately before unlinking must catch it
        # and skip deletion, rather than deleting a file that was just
        # replaced with fresh content underneath it.
        L.record_shadow_marker("sess-a", project="/repos/proj-a")
        marker_path = L._shadow_marker_path("sess-a")
        old_time = time.time() - 1_000_000
        os.utime(marker_path, (old_time, old_time))

        real_stat = Path.stat
        calls_for_marker = {"n": 0}

        class _FakeStatResult:
            def __init__(self, st_mtime):
                self.st_mtime = st_mtime

        def fake_stat(self_path, *args, **kwargs):
            if self_path == marker_path:
                calls_for_marker["n"] += 1
                if calls_for_marker["n"] == 1:
                    return _FakeStatResult(old_time)  # first check: looks stale
                return _FakeStatResult(time.time())  # re-check: was just refreshed
            return real_stat(self_path, *args, **kwargs)

        with mock.patch.object(Path, "stat", fake_stat):
            deleted = L.sweep_stale_shadow_markers()

        self.assertEqual(deleted, 0)
        self.assertTrue(marker_path.exists())

    def test_sweep_respects_custom_ttl_env_var(self):
        os.environ["CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS"] = "1"
        L.record_shadow_marker("sess-a", project="/repos/proj-a")
        two_hours_ago = time.time() - (2 * 3600)
        os.utime(L._shadow_marker_path("sess-a"), (two_hours_ago, two_hours_ago))

        deleted = L.sweep_stale_shadow_markers()
        self.assertEqual(deleted, 1)

    def test_invalid_ttl_env_var_falls_back_to_default(self):
        os.environ["CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS"] = "not-a-number"
        L.record_shadow_marker("sess-a", project="/repos/proj-a")
        # Fresh marker, well under the default TTL -- must survive.
        deleted = L.sweep_stale_shadow_markers()
        self.assertEqual(deleted, 0)

    def test_default_ttl_is_168_hours(self):
        # Pins issue #233's fix directly: SessionStart-based refresh (#231)
        # only resets a marker's mtime on resume/clear/compact, not on
        # every tool call, so the default TTL was raised from 48h to 168h
        # (1 week) to shrink the window where a long-running session's
        # marker can be swept out from under it. A future accidental
        # revert of the constant fails HERE, not just as a side effect of
        # a timing test elsewhere.
        self.assertEqual(L._ttl_hours(), 168.0)
        self.assertEqual(L._DEFAULT_SESSION_MARKER_TTL_HOURS, 168.0)

    def test_non_positive_ttl_env_var_falls_back_to_default(self):
        # "0" and negative values parse fine via float() without raising,
        # but a non-positive TTL makes the very next sweep treat even the
        # marker THIS SAME hook invocation just refreshed as stale --
        # confirmed live this defeated the self-refresh-before-sweep safety
        # guarantee entirely. Must fall back to the default like any other
        # invalid value.
        for bad_ttl in ("0", "-1", "-48"):
            with self.subTest(bad_ttl=bad_ttl):
                os.environ["CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS"] = bad_ttl
                self.assertEqual(L._ttl_hours(), L._DEFAULT_SESSION_MARKER_TTL_HOURS)

    def test_nan_ttl_env_var_falls_back_to_default(self):
        # float("nan") parses without raising too -- reject it the same way.
        os.environ["CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS"] = "nan"
        self.assertEqual(L._ttl_hours(), L._DEFAULT_SESSION_MARKER_TTL_HOURS)

    def test_sweep_on_missing_directory_is_a_no_op(self):
        self.assertEqual(L.sweep_stale_shadow_markers(), 0)

    def test_should_sweep_true_when_no_sentinel_yet(self):
        self.assertTrue(L.should_sweep())

    def test_should_sweep_false_right_after_mark_swept(self):
        L.mark_swept()
        self.assertFalse(L.should_sweep())

    def test_should_sweep_true_once_interval_elapsed(self):
        L.mark_swept()
        far_future = time.time() + L._SWEEP_INTERVAL_SECONDS + 1
        self.assertTrue(L.should_sweep(now=far_future))

    def test_self_refresh_before_sweep_survives_even_if_technically_stale(self):
        # The ordering contract hooks/record_session_id.py relies on: a
        # session refreshing its OWN marker before the sweep runs can never
        # observe itself as stale, no matter how long it's been since this
        # session's own last tool call relative to the TTL.
        os.environ["CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS"] = "1"
        L.record_shadow_marker("sess-self", project="/repos/proj-a")
        # Refresh happens at "now" (mtime just set) -- sweep immediately after
        # must not delete it even though the TTL is aggressively short.
        deleted = L.sweep_stale_shadow_markers()
        self.assertEqual(deleted, 0)
        self.assertTrue(L._shadow_marker_path("sess-self").exists())


class TranscriptScanStrategy(SessionIdLibTestCase):
    def test_no_directory_returns_none(self):
        self.assertIsNone(L.session_id(L.SessionIdStrategy.TRANSCRIPT_SCAN, project="/repos/my-project"))

    def test_empty_directory_returns_none(self):
        slug = L._project_slug("/repos/my-project")
        (self.projects_dir / slug).mkdir(parents=True)
        self.assertIsNone(L.session_id(L.SessionIdStrategy.TRANSCRIPT_SCAN, project="/repos/my-project"))

    def test_returns_most_recently_modified_transcript_stem(self):
        slug = L._project_slug("/repos/my-project")
        project_dir = self.projects_dir / slug
        project_dir.mkdir(parents=True)
        old = project_dir / "old-session-id.jsonl"
        old.write_text("{}\n", encoding="utf-8")
        os.utime(old, (time.time() - 100, time.time() - 100))
        new = project_dir / "new-session-id.jsonl"
        new.write_text("{}\n", encoding="utf-8")

        result = L.session_id(L.SessionIdStrategy.TRANSCRIPT_SCAN, project="/repos/my-project")
        self.assertEqual(result, "new-session-id")

    def test_slug_replaces_every_slash_with_hyphen(self):
        self.assertEqual(L._project_slug("/Users/dev/repos/my-project"), "-Users-dev-repos-my-project")

    def test_no_project_falls_back_to_cwd(self):
        with mock.patch.object(os, "getcwd", return_value="/repos/cwd-project"):
            slug = L._project_slug(None)
        self.assertEqual(slug, "-repos-cwd-project")

    def test_getcwd_oserror_returns_none_instead_of_raising(self):
        # Same fail-open concern as SHADOW_FILE's identical test -- the
        # os.getcwd() fallback used when `project` is omitted can itself
        # raise OSError, which must not propagate out of a call this module
        # documents as never raising for a recognized strategy.
        with mock.patch.object(os, "getcwd", side_effect=OSError("no such directory")):
            result = L.session_id(L.SessionIdStrategy.TRANSCRIPT_SCAN)
        self.assertIsNone(result)

    def test_relative_project_is_rejected_instead_of_computing_wrong_slug(self):
        # Same contract-enforcement gap as SHADOW_FILE's identical test --
        # a relative `project` used to compute a bogus <slug> instead of
        # being rejected.
        result = L.session_id(L.SessionIdStrategy.TRANSCRIPT_SCAN, project="my-project")
        self.assertIsNone(result)


class ProxyStrategy(SessionIdLibTestCase):
    def test_returns_a_string(self):
        result = L.session_id(L.SessionIdStrategy.PROXY)
        self.assertIsInstance(result, str)
        self.assertTrue(result)

    def test_stable_across_repeated_calls_same_process(self):
        first = L.session_id(L.SessionIdStrategy.PROXY)
        second = L.session_id(L.SessionIdStrategy.PROXY)
        self.assertEqual(first, second)

    def test_concurrent_first_calls_agree_on_one_id(self):
        # A bare "if cached is None: generate" is not atomic -- confirmed
        # live that concurrent first callers could each observe None and
        # mint their own UUID before the fix added a lock. Widen the race
        # window deterministically (slow down uuid4() itself) rather than
        # relying on timing luck to make this flaky.
        num_threads = 8
        barrier = threading.Barrier(num_threads)
        results: list[str] = []
        results_lock = threading.Lock()

        real_uuid4 = uuid.uuid4

        def slow_uuid4(*args, **kwargs):
            time.sleep(0.01)
            return real_uuid4()

        def worker():
            barrier.wait()
            result = L.session_id(L.SessionIdStrategy.PROXY)
            with results_lock:
                results.append(result)

        with mock.patch.object(L.uuid, "uuid4", side_effect=slow_uuid4):
            threads = [threading.Thread(target=worker) for _ in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(set(results)), 1)


class ShadowFileStaleMarkerFallback(SessionIdLibTestCase):
    """Issue #236: when the best-matching SHADOW_FILE marker is stale (older
    than CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS), fall back to TRANSCRIPT_SCAN
    before returning None -- a live session that has outlived the TTL without
    triggering a hook lifecycle event should still be recoverable."""

    def _write_stale_marker(self, session_id: str, project_abs: str, ttl_hours: float = 168.0) -> float:
        """Write a marker and backdate its mtime so it's clearly past the TTL."""
        L.record_shadow_marker(session_id, project=project_abs)
        marker = L._shadow_marker_path(session_id)
        stale_time = time.time() - (ttl_hours * 3600 + 1)
        os.utime(marker, (stale_time, stale_time))
        return stale_time

    def test_fresh_marker_returns_session_id_without_fallback(self):
        # A non-stale marker is returned directly -- no TRANSCRIPT_SCAN call.
        L.record_shadow_marker("fresh-sess", project="/repos/myapp")
        result = L._session_id_from_shadow_file("myapp")
        self.assertEqual(result, "fresh-sess")

    def test_stale_marker_falls_back_to_transcript_scan_result(self):
        # The matching marker is past the TTL; TRANSCRIPT_SCAN finds a live one.
        # Copilot review on PR #252: TRANSCRIPT_SCAN must receive the marker's
        # own recorded absolute path, not None/os.getcwd(), so it scans the
        # RIGHT project's transcript directory even when the caller's cwd
        # differs from the marker's project.
        self._write_stale_marker("stale-sess", "/repos/myapp")
        with mock.patch.object(L, "_session_id_from_transcript_scan", return_value="live-from-transcript") as mock_scan:
            result = L._session_id_from_shadow_file("myapp")
        self.assertEqual(result, "live-from-transcript")
        mock_scan.assert_called_once_with("/repos/myapp")  # marker's own absolute path

    def test_stale_marker_with_transcript_scan_returning_none_returns_none(self):
        # Neither strategy can resolve -- return None rather than fabricating.
        self._write_stale_marker("stale-sess", "/repos/myapp")
        with mock.patch.object(L, "_session_id_from_transcript_scan", return_value=None) as mock_scan:
            result = L._session_id_from_shadow_file("myapp")
        self.assertIsNone(result)
        mock_scan.assert_called_once_with("/repos/myapp")

    def test_no_matching_marker_at_all_falls_back_to_transcript_scan(self):
        # Issue #253: when the sessions dir has markers but none match the
        # requested project (all swept by sweep_stale_shadow_markers()), fall
        # back to TRANSCRIPT_SCAN with the MCP server's cwd (not None) when the
        # cwd's basename matches the requested project.
        # Contrast with the stale-marker case (issue #236), where the marker's
        # own recorded absolute path is used so TRANSCRIPT_SCAN always scans the
        # right project root regardless of cwd.
        L.record_shadow_marker("sess-other", project="/repos/other-project")
        with mock.patch.object(os, "getcwd", return_value="/repos/myapp"):
            with mock.patch.object(L, "_session_id_from_transcript_scan", return_value="live-from-ts") as mock_scan:
                result = L._session_id_from_shadow_file("myapp")  # "myapp" has no marker
        self.assertEqual(result, "live-from-ts")
        mock_scan.assert_called_once_with("/repos/myapp")

    def test_no_matching_marker_transcript_scan_returns_none(self):
        # Issue #253: TRANSCRIPT_SCAN is called but also finds nothing -- None returned.
        L.record_shadow_marker("sess-other", project="/repos/other-project")
        with mock.patch.object(os, "getcwd", return_value="/repos/myapp"):
            with mock.patch.object(L, "_session_id_from_transcript_scan", return_value=None) as mock_scan:
                result = L._session_id_from_shadow_file("myapp")
        self.assertIsNone(result)
        mock_scan.assert_called_once_with("/repos/myapp")

    def test_cwd_mismatch_skips_swept_marker_fallback(self):
        # Cross-project guard (issue #35): if the MCP server's cwd basename
        # does NOT match the requested project, skip the fallback -- returning
        # the wrong project's session_id would be worse than returning None.
        L.record_shadow_marker("sess-other", project="/repos/other-project")
        with mock.patch.object(os, "getcwd", return_value="/repos/projectA"):
            with mock.patch.object(L, "_session_id_from_transcript_scan") as mock_scan:
                result = L._session_id_from_shadow_file("projectB")
        self.assertIsNone(result)
        mock_scan.assert_not_called()

    def test_empty_sessions_dir_falls_back_to_transcript_scan_on_cwd_match(self):
        # Issue #253 (Copilot review fix): when the project's last marker was
        # swept AND it was the only marker (dated is empty), the fallback must
        # still fire -- the early `if not dated: return None` was removed to
        # cover this case.
        self.sessions_dir.mkdir(parents=True)  # dir exists but empty
        with mock.patch.object(os, "getcwd", return_value="/repos/myapp"):
            with mock.patch.object(L, "_session_id_from_transcript_scan", return_value="ts-result") as mock_scan:
                result = L._session_id_from_shadow_file("myapp")
        self.assertEqual(result, "ts-result")
        mock_scan.assert_called_once_with("/repos/myapp")

    def test_project_none_with_swept_markers_skips_transcript_fallback(self):
        # project=None means no marker's basename can ever equal None --
        # there is nothing to scope the fallback to, so TRANSCRIPT_SCAN
        # must NOT be called when project is None, even when markers exist.
        L.record_shadow_marker("sess-other", project="/repos/other-project")
        with mock.patch.object(L, "_session_id_from_transcript_scan") as mock_scan:
            result = L._session_id_from_shadow_file(None)
        self.assertIsNone(result)
        mock_scan.assert_not_called()

    def test_project_none_returns_none_even_with_stale_markers(self):
        # Passing project=None never matches any marker, so no fallback fires.
        self._write_stale_marker("stale-sess", "/repos/myapp")
        with mock.patch.object(L, "_session_id_from_transcript_scan") as mock_scan:
            result = L._session_id_from_shadow_file(None)
        self.assertIsNone(result)
        mock_scan.assert_not_called()

    def test_stale_threshold_reads_ttl_env_var(self):
        # A custom TTL of 1h: marker 2h old is stale; marker 0.5h old is fresh.
        os.environ["CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS"] = "1"
        self._write_stale_marker("stale-sess", "/repos/myapp", ttl_hours=1.0)
        with mock.patch.object(L, "_session_id_from_transcript_scan", return_value="ts-result") as mock_scan:
            result = L._session_id_from_shadow_file("myapp")
        self.assertEqual(result, "ts-result")
        mock_scan.assert_called_once_with("/repos/myapp")

    def test_fresh_under_custom_ttl_not_fallen_back(self):
        # Marker is 2h old but TTL is 3h -- should NOT fall back.
        os.environ["CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS"] = "3"
        L.record_shadow_marker("recent-sess", project="/repos/myapp")
        marker = L._shadow_marker_path("recent-sess")
        two_hours_ago = time.time() - (2 * 3600)
        os.utime(marker, (two_hours_ago, two_hours_ago))
        with mock.patch.object(L, "_session_id_from_transcript_scan") as mock_scan:
            result = L._session_id_from_shadow_file("myapp")
        self.assertEqual(result, "recent-sess")
        mock_scan.assert_not_called()


class UnknownStrategy(SessionIdLibTestCase):
    def test_raises_value_error(self):
        with self.assertRaises(ValueError):
            L.session_id(99)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
