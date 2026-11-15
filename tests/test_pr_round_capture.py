"""Tests for the SessionEnd capture of PR review rounds: libs/pr_round_capture_lib.py,
tools/capture_pr_rounds.py (the detached worker), hooks/session_end_pr_rounds.py, and how
`setup_project` init/upgrade install the hook and `doctor` reports it. Fixture transcripts (the
builders of test_pr_round_cost_lib.py), a stubbed `gh`, temporary metrics databases and logs:
no network, no real ~/.claude, no live model."""

import builtins
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "libs"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "hooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import capture_pr_rounds as CLI  # noqa: E402
import doctor as doctor_cli  # noqa: E402
import doctor_lib  # noqa: E402
import metrics_lib  # noqa: E402
import pr_round_capture_lib as C  # noqa: E402
import pr_round_record_lib as P  # noqa: E402
import session_end_pr_rounds as hook  # noqa: E402
import setup_project_lib  # noqa: E402
import upgrade_lib  # noqa: E402
from test_pr_round_backfill_lib import BackfillGh  # noqa: E402
from test_pr_round_cost_lib import PR, Transcript, iso, usage  # noqa: E402
from test_pr_round_record_lib import three_round_transcript  # noqa: E402

MERGED = {PR: {"state": "MERGED", "mergedAt": iso("04:20:28"), "closedAt": iso("04:20:28")}}
OPEN = {PR: {"state": "OPEN", "mergedAt": None, "closedAt": None}}
REPO_NAME = "Donelle/claude-runway"


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.db = self.root / "metrics.db"
        self.log = self.root / "pr-round-capture.log"
        self.store = metrics_lib.MetricsStore(self.db)

    def transcript(self, t=None, name="sess-1.jsonl"):
        return (t or three_round_transcript()).write(self.root, name)

    def capture(self, path, lookups=None, gh=None, **kw):
        gh = gh if gh is not None else BackfillGh(lookups=lookups if lookups is not None else MERGED)
        return C.capture(path, repo=REPO_NAME, store=self.store, gh=gh, **kw)

    def count(self):
        return self.store.summary(P.METRIC_ID)["event_count"]

    def rows(self):
        return self.store.detail(P.METRIC_ID, limit=100)


class Records(_Base):
    def test_the_hook_s_worker_records_the_rounds_in_a_transcript(self):
        report = self.capture(self.transcript())
        self.assertTrue(report["ok"])
        self.assertEqual((report["written"], report["existing"], report["failed"]), (6, 0, 0))
        self.assertEqual(report["session_id"], "sess-1")
        self.assertEqual([(e["pr"], e["state"], e["rounds"], e["fetch_events"]) for e in report["prs"]],
                         [(PR, "MERGED", 3, 3)])
        self.assertEqual(sorted(r["event_type"] for r in self.rows()), [P.EVENT_FETCH] * 3 + [P.EVENT_ROUND] * 3)

    def test_what_it_writes_is_what_the_recorder_writes_for_the_same_transcript_and_end(self):
        path = self.transcript()
        self.capture(path)
        direct = metrics_lib.MetricsStore(self.root / "direct.db")
        P.record_pr_rounds(path, PR, store=direct, gh=BackfillGh(), merge_time=iso("04:20:28"))
        strip = lambda s: sorted(json.dumps([r["event_type"], r["metadata"]], sort_keys=True)
                                 for r in s.detail(P.METRIC_ID, limit=100))
        self.assertEqual(strip(self.store), strip(direct))

    def test_running_it_twice_writes_nothing_twice(self):
        path = self.transcript()
        self.capture(path)
        again = self.capture(path)
        self.assertEqual((again["written"], again["existing"], again["failed"]), (0, 6, 0))
        self.assertEqual(self.count(), 6)

    def test_a_backfill_style_run_after_the_session_end_capture_finds_the_rows_there(self):
        path = self.transcript()
        self.capture(path)
        P.record_pr_rounds(path, PR, store=self.store, gh=BackfillGh(), merge_time=iso("04:20:28"))
        self.assertEqual(self.count(), 6)

    def test_a_merge_after_the_session_ended_does_not_stretch_the_last_window(self):
        later = {PR: {"state": "MERGED", "mergedAt": "2026-11-09T00:00:00Z", "closedAt": "2026-11-09T00:00:00Z"}}
        self.capture(self.transcript(), lookups=later)
        ends = {(r["metadata"]["window_end"], r["metadata"]["window_open"])
                for r in self.rows() if r["event_type"] == P.EVENT_ROUND}
        self.assertIn((iso("04:20:28"), False), ends)
        self.assertFalse(any(is_open for _, is_open in ends))

    def test_a_closed_pr_ends_at_its_closing_time(self):
        closed = {PR: {"state": "CLOSED", "mergedAt": None, "closedAt": iso("04:20:28")}}
        report = self.capture(self.transcript(), lookups=closed)
        self.assertEqual(report["written"], 6)
        self.assertEqual(report["prs"][0]["state"], "CLOSED")

    def test_the_marker_alone_is_enough_when_the_commands_are_not_a_known_shape(self):
        t = Transcript()
        t.bash("04:01:00", "bash ./fetch-everything.sh 603",
               f"out\npr-feedback-fetch: pr={PR} copilot-review=11\n", u=usage(1, 1, 1, 7))
        t.assistant("04:02:00", u=usage(1, 1, 1, 9))
        report = self.capture(self.transcript(t))
        self.assertEqual([(e["pr"], e["rounds"]) for e in report["prs"]], [(PR, 1)])
        self.assertGreater(report["written"], 0)

    def test_a_session_with_neither_marker_nor_known_command_records_nothing_and_says_so(self):
        # What the real backfill met most often: a compound the reader's whitelist does not know.
        t = Transcript()
        t.bash("04:01:00", f"cd /x/repo && gh api repos/o/r/pulls/{PR}/reviews --paginate", "[]")
        t.assistant("04:02:00")
        report = self.capture(self.transcript(t))
        self.assertEqual((report["written"], report["prs"]), (0, []))
        self.assertEqual(report["unrecognised_fetches"], 1)
        self.assertEqual(self.count(), 0)
        self.assertIn("unrecognised=1", C.summary_line(report))
        self.assertTrue(C.has_something_to_log(report))

    def test_a_local_review_call_that_names_the_pr_records_a_local_round(self):
        t = Transcript()
        t.assistant("04:00:00")
        t.review_call("04:01:00", {"pr": PR})
        t.assistant("04:02:00")
        report = self.capture(self.transcript(t))
        self.assertEqual([(e["pr"], e["rounds"]) for e in report["prs"]], [(PR, 1)])
        self.assertEqual([r["metadata"]["kind"] for r in self.rows() if r["event_type"] == P.EVENT_ROUND], ["local"])

    def test_a_pr_number_that_merely_appears_in_a_command_records_nothing(self):
        t = Transcript()
        t.bash("04:01:00", f"gh pr view {PR} --json state", '{"state": "MERGED"}')
        t.assistant("04:02:00")
        report = self.capture(self.transcript(t))
        self.assertEqual((report["written"], report["skipped"], report["candidates"]), (0, [], 1))
        self.assertFalse(C.has_something_to_log(report))


class RepoAndReviewerFlag(_Base):
    """What the capture adds to each row through the recorder: the repository (or project) and,
    per Copilot round, whether the reviewer was used."""

    def copilot_rows(self):
        return [r["metadata"] for r in self.rows() if r["event_type"] == P.EVENT_ROUND
                and r["metadata"]["kind"] == "copilot"][::-1]

    def test_every_row_carries_the_repo_the_capture_was_given(self):
        self.capture(self.transcript())
        for row in self.rows():
            self.assertEqual((row["metadata"]["repo"], row["metadata"]["repo_source"]), (REPO_NAME, "remote"))

    def test_a_session_without_a_repo_gets_the_project_name_of_its_cwd(self):
        t = three_round_transcript()
        for line in t.lines:
            line["cwd"] = "/Users/me/repos/other-project"
        gh = BackfillGh(lookups=MERGED, remote=None)
        report = C.capture(self.transcript(t), store=self.store, gh=gh)
        self.assertEqual(report["written"], 6)
        for row in self.rows():
            self.assertEqual((row["metadata"]["repo"], row["metadata"]["repo_source"]), ("other-project", "project"))

    def test_the_flag_follows_a_review_call_between_two_copilot_rounds(self):
        t = three_round_transcript()
        t.review_call("04:05:00", {"pr": PR})
        self.capture(self.transcript(t))
        self.assertEqual([m["local_reviewer_used"] for m in self.copilot_rows()], [True, False, False])

    def test_a_second_capture_after_the_flag_was_recorded_writes_nothing(self):
        t = three_round_transcript()
        t.review_call("04:05:00", {"pr": PR})
        path = self.transcript(t)
        self.capture(path)
        before = self.count()
        again = self.capture(path)
        self.assertEqual((again["written"], again["existing"]), (0, before))


class OpenPullRequests(_Base):
    def test_an_open_pr_is_skipped_so_no_permanently_short_last_window_is_stored(self):
        report = self.capture(self.transcript(), lookups=OPEN)
        self.assertEqual((report["written"], report["prs"]), (0, []))
        self.assertIn("still open", report["skipped"][0]["reason"])
        self.assertEqual(self.count(), 0)

    def test_the_skipped_pr_is_recorded_by_a_later_capture_once_it_is_merged(self):
        path = self.transcript()
        self.capture(path, lookups=OPEN)
        later = self.capture(path, lookups=MERGED)
        self.assertEqual(later["written"], 6)
        self.assertFalse(any(r["metadata"]["window_open"] for r in self.rows() if r["event_type"] == P.EVENT_ROUND))

    def test_include_open_records_it_as_it_stands_with_window_open_true(self):
        report = self.capture(self.transcript(), lookups=OPEN, include_open=True)
        self.assertEqual(report["written"], 6)
        self.assertTrue(any(r["metadata"]["window_open"] for r in self.rows() if r["event_type"] == P.EVENT_ROUND))


class NothingToRecord(_Base):
    def test_a_session_with_no_feedback_fetch_records_nothing_and_logs_nothing(self):
        t = Transcript()
        t.assistant("04:00:00")
        t.bash("04:01:00", "ls -la", "total 0")
        t.assistant("04:02:00")
        gh = BackfillGh()
        report = self.capture(self.transcript(t), gh=gh)
        self.assertTrue(report["ok"])
        self.assertEqual((report["candidates"], report["written"], report["prs"], report["skipped"]), (0, 0, [], []))
        self.assertEqual(gh.calls, [], "a session with no feedback must not even ask GitHub")
        self.assertFalse(C.has_something_to_log(report))
        self.assertEqual(self.count(), 0)

    def test_an_empty_transcript_is_noted(self):
        path = self.root / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        report = self.capture(path)
        self.assertTrue(report["ok"])
        self.assertEqual(report["notes"], ["the transcript is empty"])
        self.assertTrue(C.has_something_to_log(report))


class FailsOpen(_Base):
    def test_a_malformed_transcript_is_a_logged_note_not_an_error(self):
        path = self.root / "bad.jsonl"
        path.write_text("not json at all\n{ broken\n\x00\x01 garbage\n", encoding="utf-8")
        report = self.capture(path)
        self.assertTrue(report["ok"])
        self.assertEqual((report["written"], report["error"]), (0, None))
        self.assertEqual(len(report["notes"]), 1)
        self.assertIn("malformed", report["notes"][0])
        self.assertIn("note=", C.summary_line(report))

    def test_binary_garbage_does_not_raise(self):
        path = self.root / "bin.jsonl"
        path.write_bytes(bytes(range(256)) * 50)
        report = self.capture(path)
        self.assertIsNone(report["error"])
        self.assertEqual(report["written"], 0)

    def test_a_missing_transcript_is_an_error_in_the_report_not_an_exception(self):
        report = self.capture(self.root / "nope.jsonl")
        self.assertFalse(report["ok"])
        self.assertIn("could not be opened", report["error"])

    def test_a_gh_failure_for_the_lookup_skips_the_pr_and_says_why(self):
        gh = BackfillGh(lookups={PR: "gh exited 4: not signed in"})
        report = self.capture(self.transcript(), gh=gh)
        self.assertEqual((report["written"], report["error"]), (0, None))
        self.assertIn("could not be looked up", report["skipped"][0]["reason"])
        self.assertIn("not signed in", C.summary_line(report))
        self.assertEqual(self.count(), 0)

    def test_a_gh_failure_while_enriching_still_records_the_rows_with_null_github_fields(self):
        gh = BackfillGh(lookups=MERGED, fail=("reviews", "size", "suites", "comments"))
        report = self.capture(self.transcript(), gh=gh)
        self.assertEqual(report["written"], 6)
        self.assertTrue(report["github_errors"])
        self.assertEqual({r["metadata"]["github"] for r in self.rows() if r["event_type"] == P.EVENT_ROUND},
                         {"unavailable"})

    def test_gh_missing_entirely_is_a_skip(self):
        def no_gh(args):
            raise P.GhError("gh could not run: FileNotFoundError")
        report = self.capture(self.transcript(), gh=no_gh)
        self.assertEqual((report["written"], report["error"]), (0, None))
        self.assertEqual(len(report["skipped"]), 1)

    def test_a_failing_store_stops_the_run_and_is_reported(self):
        failing = mock.Mock()
        failing.record_if_absent.return_value = "failed"
        report = C.capture(self.transcript(), repo=REPO_NAME, store=failing, gh=BackfillGh())
        self.assertIsNotNone(report["store_error"])
        self.assertEqual(failing.record_if_absent.call_count, 1)
        self.assertIn("store_error=", C.summary_line(report))
        # A PR whose write failed is not a recorded PR: not in `prs`, not in the log's `recorded=`.
        self.assertEqual(report["prs"], [])
        self.assertIn("the store failed", report["skipped"][0]["reason"])
        self.assertNotIn("recorded=", C.summary_line(report))

    def test_the_cli_does_not_call_a_failed_pr_recorded(self):
        failing = mock.Mock()
        failing.record_if_absent.return_value = "failed"
        report = C.capture(self.transcript(), repo=REPO_NAME, store=failing, gh=BackfillGh())
        self.assertIn("0 recorded", CLI.format_text(report, dry_run=False))

    def test_an_unexpected_exception_is_contained(self):
        with mock.patch.object(C.B, "scan_transcript", side_effect=RuntimeError("boom")):
            report = self.capture(self.transcript())
        self.assertEqual(report["error"], "unexpected RuntimeError")
        self.assertNotIn("boom", C.summary_line(report))

    def test_a_bad_repo_is_refused_before_anything_is_read(self):
        report = C.capture(self.transcript(), repo="not a repo", store=self.store, gh=BackfillGh())
        self.assertIn("repo must look like", report["error"])

    def test_an_exhausted_time_budget_skips_the_remaining_prs(self):
        report = self.capture(self.transcript(), deadline=0.0, clock=lambda: 1.0)
        self.assertEqual(report["written"], 0)
        self.assertIn("time budget", report["skipped"][0]["reason"])

    def test_a_dry_run_builds_everything_and_never_opens_the_store(self):
        with mock.patch.object(metrics_lib, "MetricsStore", side_effect=AssertionError("must not open the store")):
            report = C.capture(self.transcript(), repo=REPO_NAME, gh=BackfillGh(), dry_run=True)
        self.assertEqual(len(report["prs"]), 1)
        self.assertEqual(report["written"], 0)
        self.assertEqual(self.count(), 0)


class TheLog(_Base):
    def test_transcript_controlled_text_cannot_inject_a_line_or_a_control_sequence(self):
        report = {"transcript": "a\nb.jsonl", "session_id": "s\x1b[31m\r\nFAKE 2026 error=x", "prs": [], "skipped": [
            {"pr": 5, "reason": "gh said:\nline two \x07 bell"}], "unrecognised_fetches": 0, "notes": [],
            "github_errors": [], "store_error": None, "error": None, "written": 0, "existing": 0, "failed": 0}
        line = C.summary_line(report)
        self.assertNotIn("\n", line)
        self.assertTrue(all(32 <= ord(ch) < 127 for ch in line), line)
        C.log_note(self.log, line)
        self.assertEqual(len(self.log.read_text(encoding="utf-8").splitlines()), 1)

    def test_the_log_is_bounded_by_rotation(self):
        self.log.write_text("x" * (C.MAX_LOG_BYTES + 10), encoding="utf-8")
        C.log_note(self.log, "fresh")
        self.assertEqual(len(self.log.read_text(encoding="utf-8").splitlines()), 1)
        self.assertEqual(self.log.with_name(self.log.name + ".1").stat().st_size, C.MAX_LOG_BYTES + 10)

    def test_a_really_oversized_log_replaces_the_archive(self):
        archive = self.log.with_name(self.log.name + ".1")
        archive.write_text("older", encoding="utf-8")
        self.log.write_text("y" * (C.MAX_LOG_BYTES + 1), encoding="utf-8")
        C.log_note(self.log, "fresh")
        self.assertEqual(archive.stat().st_size, C.MAX_LOG_BYTES + 1)
        self.assertEqual(len(self.log.read_text(encoding="utf-8").splitlines()), 1)

    def test_concurrent_writers_lose_no_line(self):
        # Many processes' worth of appends at once, through the lock: every line is there.
        import threading
        errors = []

        def write(n):
            try:
                for i in range(25):
                    C.log_note(self.log, f"writer {n} line {i}")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=write, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(self.log.read_text(encoding="utf-8").splitlines()), 150)

    def test_the_lock_serializes_rotation_with_the_append_of_a_writer_that_arrived_first(self):
        # A holds the lock; B (oversized view) must wait, and when A releases B sees A's fresh
        # log (small) and does NOT rotate it over the archive.
        archive = self.log.with_name(self.log.name + ".1")
        archive.write_text("OLD HISTORY\n", encoding="utf-8")
        self.log.write_text("y" * (C.MAX_LOG_BYTES + 1), encoding="utf-8")
        import threading
        with C._log_lock(self.log) as held:
            self.assertTrue(held)
            writer = threading.Thread(target=C.log_note, args=(self.log, "B"))
            writer.start()
            time.sleep(0.2)
            self.assertTrue(writer.is_alive(), "B must wait for the lock")
            # A rotates and appends its own fresh line before releasing.
            os.replace(self.log, archive)
            self.log.write_text("A fresh\n", encoding="utf-8")
        writer.join(timeout=10)
        self.assertEqual(self.log.read_text(encoding="utf-8").splitlines()[0], "A fresh")
        self.assertEqual(archive.stat().st_size, C.MAX_LOG_BYTES + 1)
        self.assertEqual(len(self.log.read_text(encoding="utf-8").splitlines()), 2)

    def test_a_lock_that_cannot_be_taken_in_time_still_gets_the_line_without_rotating(self):
        self.log.write_text("y" * (C.MAX_LOG_BYTES + 1), encoding="utf-8")
        with C._log_lock(self.log) as mine:  # another holder (a separate open file, as another process would be)
            self.assertTrue(mine)
            with mock.patch.object(C, "LOG_LOCK_WAIT_SECONDS", 0.1):
                C.log_note(self.log, "late")
        self.assertIn("late", self.log.read_text(encoding="utf-8").splitlines()[-1])
        self.assertFalse(self.log.with_name(self.log.name + ".1").exists(), "no rotation without the lock")

    def test_a_second_holder_cannot_take_the_lock_until_the_first_lets_go(self):
        with C._log_lock(self.log) as first:
            self.assertTrue(first)
            with C._log_lock(self.log, wait=0.05) as second:
                self.assertFalse(second)
        with C._log_lock(self.log, wait=0.05) as again:
            self.assertTrue(again)

    @unittest.skipIf(os.name == "nt", "uses os._exit in a child to skip any release")
    def test_a_lock_held_by_a_process_that_died_is_free_with_no_stale_lock_to_break(self):
        # The OS releases the lock when its holder dies, so nothing has to judge a lock stale
        # (and no two waiters can delete each other's fresh lock).
        code = ("import os, sys; sys.path.insert(0, sys.argv[1]); import pr_round_capture_lib as C; "
                "from pathlib import Path; cm = C._log_lock(Path(sys.argv[2])); assert cm.__enter__(); os._exit(0)")
        done = subprocess.run([sys.executable, "-I", "-c", code, str(REPO / "libs"), str(self.log)],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(self.log.with_name(self.log.name + ".lock").exists())
        with C._log_lock(self.log, wait=0.05) as held:
            self.assertTrue(held)

    def test_the_lock_file_is_never_deleted_so_a_waiter_cannot_lock_a_file_nobody_else_sees(self):
        with C._log_lock(self.log):
            pass
        self.assertTrue(self.log.with_name(self.log.name + ".lock").exists())

    def test_an_unwritable_log_is_no_error(self):
        blocker = self.root / "file"
        blocker.write_text("x", encoding="utf-8")
        C.log_note(blocker / "sub" / "log", "hello")  # the parent is a file: cannot be created

    def test_the_log_sits_beside_the_database_it_describes(self):
        self.assertEqual(C.default_log_path(self.db), self.root / C.LOG_FILE_NAME)
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": str(self.db)}):
            self.assertEqual(C.default_log_path(), self.root / C.LOG_FILE_NAME)


class OptIn(unittest.TestCase):
    def test_only_one_two_or_three_truthy_spellings_turn_it_on(self):
        for value, expected in (("1", True), ("true", True), (" YES ", True), ("0", False), ("", False),
                                ("on", False), ("no", False)):
            self.assertEqual(C.capture_enabled({C.ENABLE_ENV_VAR: value}), expected, value)
        self.assertFalse(C.capture_enabled({}))

    def test_doctor_names_the_same_variable_as_the_hook(self):
        self.assertEqual(doctor_lib.PR_ROUND_CAPTURE_ENV_VAR, C.ENABLE_ENV_VAR)


class Spawning(unittest.TestCase):
    def test_the_worker_runs_isolated_and_names_only_the_one_transcript(self):
        command = C.worker_command("/v/bin/python", "/r/tools", "/p/sess.jsonl")
        self.assertEqual(command, ["/v/bin/python", "-I", str(Path("/r/tools") / "capture_pr_rounds.py"),
                                   "--transcript", "/p/sess.jsonl"])

    def test_it_is_started_detached_with_no_inherited_streams_in_the_sessions_folder(self):
        popen = mock.Mock()
        C.spawn_worker(["x"], "/work", popen=popen)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["x"])
        self.assertEqual((kwargs["stdin"], kwargs["stdout"], kwargs["stderr"]),
                         (subprocess.DEVNULL, subprocess.DEVNULL, subprocess.DEVNULL))
        self.assertEqual(kwargs["cwd"], "/work")
        if sys.platform == "win32":
            self.assertTrue(kwargs["creationflags"] & 0x00000008)
        else:
            self.assertTrue(kwargs["start_new_session"])
        # It must not wait: nothing in the call reads or waits on the process.
        self.assertEqual(popen.return_value.method_calls, [])


class WorkerCli(_Base):
    def run_cli(self, *argv, expect=0, gh=None):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(P, "run_gh", gh if gh is not None else BackfillGh()), \
                redirect_stdout(out), redirect_stderr(err):
            code = CLI.main(list(argv))
        self.assertEqual(code, expect, err.getvalue() + out.getvalue())
        return out.getvalue()

    def test_a_run_records_logs_and_prints(self):
        path = self.transcript()
        out = self.run_cli("--transcript", str(path), "--repo", REPO_NAME, "--db", str(self.db))
        self.assertIn("PR 603: MERGED, 3 round(s), 3 fetch event(s)", out)
        self.assertEqual(self.count(), 6)
        log = (self.root / C.LOG_FILE_NAME).read_text(encoding="utf-8")
        self.assertIn("recorded=603(3r/3f)", log)
        self.assertNotIn("\n\n", log)

    def test_a_second_run_writes_nothing(self):
        path = self.transcript()
        self.run_cli("--transcript", str(path), "--repo", REPO_NAME, "--db", str(self.db))
        out = self.run_cli("--transcript", str(path), "--repo", REPO_NAME, "--db", str(self.db))
        self.assertIn("written 0, already recorded 6", out)
        self.assertEqual(self.count(), 6)

    def test_a_dry_run_writes_nothing_logs_nothing_and_does_not_open_the_database(self):
        path = self.transcript()
        fresh = self.root / "fresh.db"
        self.run_cli("--transcript", str(path), "--repo", REPO_NAME, "--dry-run")
        self.assertFalse(fresh.exists())
        self.assertFalse((self.root / C.LOG_FILE_NAME).exists())

    def test_no_log_writes_no_line(self):
        self.run_cli("--transcript", str(self.transcript()), "--repo", REPO_NAME, "--db", str(self.db), "--no-log")
        self.assertFalse((self.root / C.LOG_FILE_NAME).exists())

    def test_an_explicit_log_path_is_used(self):
        other = self.root / "elsewhere" / "my.log"
        self.run_cli("--transcript", str(self.transcript()), "--repo", REPO_NAME, "--db", str(self.db),
                     "--log", str(other))
        self.assertTrue(other.exists())

    def test_a_session_with_no_feedback_adds_no_log_line(self):
        t = Transcript()
        t.assistant("04:00:00")
        self.run_cli("--transcript", str(self.transcript(t)), "--repo", REPO_NAME, "--db", str(self.db))
        self.assertFalse((self.root / C.LOG_FILE_NAME).exists())

    def test_a_missing_transcript_exits_1_and_is_logged(self):
        self.run_cli("--transcript", str(self.root / "gone.jsonl"), "--db", str(self.db), expect=1)
        self.assertIn("could not be opened", (self.root / C.LOG_FILE_NAME).read_text(encoding="utf-8"))

    def test_a_gh_failure_exits_0_and_is_logged(self):
        gh = BackfillGh(lookups={PR: "gh exited 1: boom"})
        self.run_cli("--transcript", str(self.transcript()), "--repo", REPO_NAME, "--db", str(self.db), gh=gh)
        self.assertIn("could not be looked up", (self.root / C.LOG_FILE_NAME).read_text(encoding="utf-8"))

    def test_bad_arguments_exit_2(self):
        t = str(self.transcript())
        self.run_cli("--transcript", t, "--repo", "bad repo", expect=2)
        self.run_cli("--transcript", t, "--dry-run", "--db", str(self.db), expect=2)

    def test_json_output_is_the_report(self):
        out = self.run_cli("--transcript", str(self.transcript()), "--repo", REPO_NAME, "--dry-run",
                           "--format", "json")
        data = json.loads(out)
        self.assertEqual(data["prs"][0]["pr"], PR)


def _payload(root: Path, **over) -> dict:
    transcript = root / "sess-1.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    base = {"session_id": "sess-1", "transcript_path": str(transcript), "cwd": str(root),
            "hook_event_name": "SessionEnd", "reason": "other"}
    base.update(over)
    return base


class Hook(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.log = self.root / "hook.log"

    def log_text(self):
        return self.log.read_text(encoding="utf-8") if self.log.exists() else ""

    def run_main(self, payload, env=None, spawn=None):
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        out = io.StringIO()
        spawner = spawn if spawn is not None else mock.Mock()
        env = {"CLAUDE_RUNWAY_METRICS_DB": str(self.root / "metrics.db")} | ({} if env is None else env)
        with mock.patch.dict(os.environ, env), mock.patch.object(sys, "stdin", io.StringIO(raw)), \
                mock.patch.object(C, "spawn_worker", spawner), redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            hook.main()
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(out.getvalue(), "", "the hook must never print anything")
        return spawner

    def test_it_is_a_silent_no_op_unless_the_opt_in_is_on(self):
        spawner = self.run_main(_payload(self.root), env={C.ENABLE_ENV_VAR: ""})
        spawner.assert_not_called()
        self.assertFalse((self.root / C.LOG_FILE_NAME).exists())

    def test_with_the_opt_in_on_it_starts_the_worker_for_the_sessions_own_transcript(self):
        payload = _payload(self.root)
        spawner = self.run_main(payload, env={C.ENABLE_ENV_VAR: "1"})
        spawner.assert_called_once()
        command, cwd = spawner.call_args.args
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "-I")
        self.assertEqual(Path(command[2]), REPO / "tools" / "capture_pr_rounds.py")
        self.assertEqual(command[3:], ["--transcript", payload["transcript_path"]])
        self.assertEqual(cwd, payload["cwd"])

    def test_the_hook_never_opens_the_transcript_itself(self):
        payload = _payload(self.root)
        opened = []
        real_open = builtins.open

        def spy(file, *a, **k):
            opened.append(str(file))
            return real_open(file, *a, **k)

        with mock.patch.object(builtins, "open", spy):
            hook.handle(payload, spawn=mock.Mock(), log_path=self.log)
        self.assertNotIn(payload["transcript_path"], opened)

    def test_a_payload_that_cannot_be_captured_is_a_logged_note_and_no_process(self):
        rel = {"transcript_path": "rel/sess.jsonl"}
        cases = {
            "no session_id": {"session_id": ""},
            "no absolute transcript_path": rel,
            "not a .jsonl file": {"transcript_path": str(self.root / "t.txt")},
            "the transcript file does not exist": {"transcript_path": str(self.root / "missing.jsonl")},
            "cwd is missing": {"cwd": str(self.root / "gone")},
            "cwd is missing (wrong type)": {"cwd": 7},
        }
        (self.root / "t.txt").write_text("x", encoding="utf-8")
        for why, over in cases.items():
            self.log.unlink(missing_ok=True)
            spawn = mock.Mock()
            hook.handle(_payload(self.root, **over), spawn=spawn, log_path=self.log)
            spawn.assert_not_called()
            self.assertIn("nothing captured", self.log_text(), why)

    def test_a_payload_that_is_not_an_object_is_logged_and_ignored(self):
        spawn = mock.Mock()
        hook.handle(["not", "an", "object"], spawn=spawn, log_path=self.log)
        spawn.assert_not_called()
        self.assertIn("not a JSON object", self.log_text())

    def test_a_worker_that_will_not_start_is_logged_and_the_hook_still_exits_0(self):
        spawn = mock.Mock(side_effect=OSError("no such interpreter"))
        hook.handle(_payload(self.root), spawn=spawn, log_path=self.log)
        self.assertIn("could not start the worker: OSError", self.log_text())
        self.assertNotIn("no such interpreter", self.log_text())

    def test_unreadable_stdin_ends_cleanly(self):
        spawner = self.run_main("{not json", env={C.ENABLE_ENV_VAR: "1"})
        spawner.assert_not_called()
        self.assertIn("not valid JSON", (self.root / C.LOG_FILE_NAME).read_text(encoding="utf-8"))

    def test_an_unexpected_error_inside_the_hook_still_exits_0_without_output(self):
        with mock.patch.object(hook, "handle", side_effect=RuntimeError("boom")):
            self.run_main(_payload(self.root), env={C.ENABLE_ENV_VAR: "1"})

    def test_a_missing_worker_script_is_a_logged_note(self):
        with mock.patch.object(hook, "TOOLS_DIR", str(self.root / "no-tools")):
            spawn = mock.Mock()
            hook.handle(_payload(self.root), spawn=spawn, log_path=self.log)
        spawn.assert_not_called()
        self.assertIn("capture_pr_rounds.py is missing", self.log_text())


@unittest.skipIf(os.name == "nt", "the stand-in gh is a shell script")
class EndToEnd(unittest.TestCase):
    """The real hook, the real detached worker (python -I), a stand-in `gh` on PATH: the hook
    returns with nothing on stdout and the rounds appear in a scratch database afterwards."""

    def test_the_hook_returns_at_once_and_the_worker_records_the_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "session"
            session.mkdir()
            transcript = three_round_transcript().write(session, "sess-1.jsonl")
            bindir = root / "bin"
            bindir.mkdir()
            fake_gh = bindir / "gh"
            fake_gh.write_text(
                "#!/bin/sh\n"
                'case "$*" in\n'
                f'  *state,mergedAt,closedAt*) echo \'{json.dumps(MERGED[PR])}\';;\n'
                "  *) exit 1;;\n"
                "esac\n", encoding="utf-8")
            fake_gh.chmod(fake_gh.stat().st_mode | stat.S_IXUSR)
            db = root / "scratch-metrics.db"
            env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
                   "CLAUDE_RUNWAY_METRICS_DB": str(db), C.ENABLE_ENV_VAR: "1"}
            payload = {"session_id": "sess-1", "transcript_path": str(transcript), "cwd": str(session)}
            done = subprocess.run([sys.executable, str(REPO / "hooks" / "session_end_pr_rounds.py")],
                                  input=json.dumps(payload), capture_output=True, text=True, env=env, timeout=60)
            self.assertEqual((done.returncode, done.stdout), (0, ""), done.stderr)
            store = metrics_lib.MetricsStore(db)
            log = root / C.LOG_FILE_NAME
            # The worker writes its six events in separate transactions and the log line only
            # after capture ends, so wait for BOTH the full count and the finished log line.
            deadline = time.monotonic() + 60
            count, text = 0, ""
            while time.monotonic() < deadline:
                if db.exists():
                    count = store.summary(P.METRIC_ID)["event_count"]
                if log.exists():
                    text = log.read_text(encoding="utf-8")
                if count == 6 and "recorded=603" in text:
                    break
                time.sleep(0.25)
            self.assertEqual(count, 6, "the detached worker did not record the rounds")
            self.assertIn("recorded=603", text)


class Installation(unittest.TestCase):
    """`setup_project` init and upgrade install the hook; `doctor` reports it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.target = Path(self._tmp.name) / "proj"
        self.target.mkdir()
        self.venv = Path("/fake/venv/bin/python")

    def init(self):
        result = setup_project_lib.run_setup(self.target, REPO, home_dir=Path("/home/user"))
        (self.target / ".claude").mkdir(exist_ok=True)
        (self.target / ".mcp.json").write_text(json.dumps(result.mcp_json), encoding="utf-8")
        (self.target / ".claude" / "settings.json").write_text(json.dumps(result.settings_json), encoding="utf-8")
        return result

    def settings(self):
        return json.loads((self.target / ".claude" / "settings.json").read_text(encoding="utf-8"))

    @staticmethod
    def scripts(settings, event):
        return [Path(h["args"][-1]).name for b in settings["hooks"][event] for h in b["hooks"]]

    def test_init_installs_the_capture_hook_under_session_end(self):
        result = self.init()
        self.assertEqual(self.scripts(result.settings_json, "SessionEnd").count("session_end_pr_rounds.py"), 1)
        args = [h["args"] for b in result.settings_json["hooks"]["SessionEnd"] for h in b["hooks"]
                if Path(h["args"][-1]).name == "session_end_pr_rounds.py"][0]
        self.assertEqual(args, [f"{REPO.as_posix()}/hooks/session_end_pr_rounds.py"])
        self.assertFalse(any("REPLACE-WITH" in json.dumps(b) or "/absolute/path/to" in json.dumps(b)
                             for b in result.settings_json["hooks"]["SessionEnd"]))

    def test_the_hook_script_init_points_at_exists(self):
        self.assertTrue((REPO / "hooks" / setup_project_lib.PR_ROUND_CAPTURE_HOOK_SCRIPT).is_file())

    def test_upgrade_adds_the_missing_hook_once_and_leaves_every_other_hook_alone(self):
        self.init()
        settings = self.settings()
        before = {event: [b for b in blocks if "session_end_pr_rounds.py" not in json.dumps(b)]
                  for event, blocks in settings["hooks"].items()}
        settings["hooks"]["SessionEnd"] = before["SessionEnd"]
        (self.target / ".claude" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        applied = upgrade_lib.run_upgrade(self.target, REPO, self.venv, auto_yes=True)
        self.assertEqual(applied, ["pr-round-capture-hook-missing"])
        after = self.settings()
        self.assertEqual(self.scripts(after, "SessionEnd").count("session_end_pr_rounds.py"), 1)
        for event, blocks in before.items():
            self.assertEqual([b for b in after["hooks"][event] if "session_end_pr_rounds.py" not in json.dumps(b)],
                             blocks, event)
        self.assertEqual(upgrade_lib.run_upgrade(self.target, REPO, self.venv, auto_yes=True), [])

    def test_upgrade_dry_run_reports_it_and_writes_nothing(self):
        self.init()
        settings = self.settings()
        settings["hooks"]["SessionEnd"] = [b for b in settings["hooks"]["SessionEnd"]
                                           if "session_end_pr_rounds.py" not in json.dumps(b)]
        path = self.target / ".claude" / "settings.json"
        path.write_text(json.dumps(settings), encoding="utf-8")
        original = path.read_text(encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            upgrade_lib.run_upgrade(self.target, REPO, self.venv, dry_run=True)
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_the_session_id_migration_does_not_duplicate_the_capture_block(self):
        # Both are SessionEnd core blocks of one template: the older migration must add only its own.
        self.init()
        settings = self.settings()
        settings["hooks"]["SessionEnd"] = [b for b in settings["hooks"]["SessionEnd"]
                                           if "record_session_id.py" not in json.dumps(b)]
        migration = next(m for m in upgrade_lib.MIGRATIONS if m.id == "record-session-id-hooks-missing")
        ctx = upgrade_lib.MigrationContext(tools_repo_dir=REPO, venv_python=self.venv)
        _, new_settings = migration.apply({}, settings, ctx)
        scripts = self.scripts(new_settings, "SessionEnd")
        self.assertEqual(scripts.count("record_session_id.py"), 1)
        self.assertEqual(scripts.count("session_end_pr_rounds.py"), 1)

    def test_doctor_reports_present_missing_and_unreadable(self):
        self.init()
        self.assertEqual(doctor_lib.find_pr_round_capture_hook_state(self.target), "present")
        settings = self.settings()
        settings["hooks"]["SessionEnd"] = []
        (self.target / ".claude" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        self.assertEqual(doctor_lib.find_pr_round_capture_hook_state(self.target), "missing")
        (self.target / ".claude" / "settings.json").write_text("{ nope", encoding="utf-8")
        self.assertEqual(doctor_lib.find_pr_round_capture_hook_state(self.target), "missing")
        (self.target / ".claude" / "settings.json").write_text(json.dumps({"hooks": {"SessionEnd": "x"}}), encoding="utf-8")
        self.assertEqual(doctor_lib.find_pr_round_capture_hook_state(self.target), "missing")
        (self.target / ".claude" / "settings.json").unlink()
        self.assertEqual(doctor_lib.find_pr_round_capture_hook_state(self.target), "missing")

    def doctor(self, hook_registered: bool, enabled: bool):
        (self.target / ".claude").mkdir(exist_ok=True)
        (self.target / ".mcp.json").write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
        blocks = [{"hooks": [{"type": "command", "command": "py", "args": ["/x/hooks/session_end_pr_rounds.py"]}]}] \
            if hook_registered else []
        (self.target / ".claude" / "settings.json").write_text(json.dumps({"hooks": {"SessionEnd": blocks}}),
                                                               encoding="utf-8")
        # `clear=True` empties the environment so nothing from the developer's shell leaks in, but
        # Path.home() then has nothing to resolve: Windows reads USERPROFILE, POSIX reads HOME (and
        # falls back to the password database only there), and without it doctor_lib raises
        # "Could not determine home directory". Give both an empty directory of its own, which is
        # also the cleaner isolation: the doctor sees no user-level settings of the real machine.
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        env = {"HOME": home.name, "USERPROFILE": home.name}
        if enabled:
            env[C.ENABLE_ENV_VAR] = "1"
        out = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), redirect_stdout(out):
            code = doctor_cli.main([str(self.target)])
        return code, out.getvalue()

    def test_doctor_present_and_on_is_clean(self):
        code, out = self.doctor(True, True)
        self.assertEqual(code, 0)
        self.assertIn("PR round capture hook: present, and CLAUDE_RUNWAY_CAPTURE_PR_ROUNDS is on", out)

    def test_doctor_present_but_off_says_it_records_nothing_and_is_not_a_failure(self):
        code, out = self.doctor(True, False)
        self.assertEqual(code, 0)
        self.assertIn("records nothing", out)

    def test_doctor_missing_and_off_is_only_a_note(self):
        code, out = self.doctor(False, False)
        self.assertEqual(code, 0)
        self.assertIn("PR round capture hook: missing (optional", out)

    def test_doctor_missing_but_opted_in_is_a_problem(self):
        code, out = self.doctor(False, True)
        self.assertEqual(code, 1)
        self.assertIn("Problem: CLAUDE_RUNWAY_CAPTURE_PR_ROUNDS is on", out)
        self.assertIn("pr-round-capture-hook-missing", out)


if __name__ == "__main__":
    unittest.main()
