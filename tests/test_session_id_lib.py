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

    def test_omitted_project_falls_back_to_cwd_and_still_filters(self):
        # Per this module's `project` contract (issue #198), omitting
        # `project` must fall back to os.getcwd() the same way every other
        # strategy does -- it must NOT disable filtering. Confirmed live
        # this used to return the most-recently-modified marker across ALL
        # projects unfiltered when `project` was omitted.
        L.record_shadow_marker("sess-old", project="/repos/proj-a")
        old_path = L._shadow_marker_path("sess-old")
        os.utime(old_path, (time.time() - 100, time.time() - 100))
        L.record_shadow_marker("sess-new", project="/repos/proj-b")

        with mock.patch.object(os, "getcwd", return_value="/somewhere/proj-a"):
            result = L.session_id(L.SessionIdStrategy.SHADOW_FILE)

        self.assertEqual(result, "sess-old")

    def test_omitted_project_with_no_cwd_match_returns_none(self):
        L.record_shadow_marker("sess-a", project="/repos/proj-a")

        with mock.patch.object(os, "getcwd", return_value="/somewhere/unrelated"):
            result = L.session_id(L.SessionIdStrategy.SHADOW_FILE)

        self.assertIsNone(result)

    def test_project_filter_matches_by_basename(self):
        L.record_shadow_marker("sess-a", project="/repos/proj-a")
        L.record_shadow_marker("sess-b", project="/repos/proj-b")
        # sess-b is more recently modified, but the caller wants proj-a's session.
        result = L.session_id(L.SessionIdStrategy.SHADOW_FILE, project="/somewhere/proj-a")
        self.assertEqual(result, "sess-a")

    def test_project_filter_with_no_match_returns_none(self):
        L.record_shadow_marker("sess-a", project="/repos/proj-a")
        result = L.session_id(L.SessionIdStrategy.SHADOW_FILE, project="/repos/proj-nonexistent")
        self.assertIsNone(result)

    def test_getcwd_oserror_returns_none_instead_of_raising(self):
        # This module documents that a recognized strategy never raises --
        # but the os.getcwd() fallback used when `project` is omitted can
        # itself raise OSError (e.g. the process's cwd was deleted/renamed
        # out from under it). Confirmed this used to propagate straight out
        # instead of failing toward None like every other unresolvable case.
        with mock.patch.object(os, "getcwd", side_effect=OSError("no such directory")):
            result = L.session_id(L.SessionIdStrategy.SHADOW_FILE)
        self.assertIsNone(result)

    def test_relative_project_is_rejected_instead_of_used_unchanged(self):
        # This module's documented `project` contract requires an absolute
        # path, never a bare/relative one -- but the contract wasn't
        # actually enforced: a relative value like "app" used to pass
        # through unchanged, silently matching an unrelated project that
        # happens to share that basename. Must fail toward None instead.
        L.record_shadow_marker("sess-a", project="/repos/app")
        result = L.session_id(L.SessionIdStrategy.SHADOW_FILE, project="app")
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
        very_old = time.time() - (49 * 3600)  # older than the 48h default
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
        # Fresh marker, well under the 48h default -- must survive.
        deleted = L.sweep_stale_shadow_markers()
        self.assertEqual(deleted, 0)

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


class UnknownStrategy(SessionIdLibTestCase):
    def test_raises_value_error(self):
        with self.assertRaises(ValueError):
            L.session_id(99)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
