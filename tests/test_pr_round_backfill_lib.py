"""Tests for libs/pr_round_backfill_lib.py and tools/backfill_pr_rounds.py: walking a project's
main-session transcripts, finding the PRs each fetched feedback for, looking each up, and
recording through libs/pr_round_record_lib.py. Fixture transcripts (the builders of
test_pr_round_cost_lib.py) in a temporary projects folder, a stubbed `gh` and a temporary
metrics.db: no network, no real ~/.claude, no live model."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "libs"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backfill_pr_rounds as CLI  # noqa: E402
import metrics_lib  # noqa: E402
import pr_round_backfill_lib as B  # noqa: E402
import pr_round_record_lib as P  # noqa: E402
from test_pr_round_cost_lib import PR, Transcript, iso, usage  # noqa: E402
from test_pr_round_record_lib import FakeGh, three_round_transcript  # noqa: E402

LOOKUP_FIELDS = "state,mergedAt,closedAt"


class BackfillGh(FakeGh):
    """FakeGh plus the PR lookup the backfill makes: `lookups` maps a PR number to the JSON it
    answers, or to an exception message that makes the call fail."""

    def __init__(self, lookups=None, **kw):
        super().__init__(**kw)
        self.lookups = lookups if lookups is not None else {PR: {"state": "MERGED", "mergedAt": iso("04:20:28"), "closedAt": iso("04:20:28")}}

    def __call__(self, args):
        if args[:2] == ["pr", "view"] and args[-1] == LOOKUP_FIELDS:
            self.calls.append(list(args))
            answer = self.lookups.get(int(args[2]))
            if answer is None or isinstance(answer, str):
                raise P.GhError(answer or "gh exited 1: no pull request found")
            return json.dumps(answer)
        return super().__call__(args)


def other_pr_transcript(pr):
    """One Copilot review of another PR, in a session of its own."""
    from test_pr_round_cost_lib import COPILOT, rest_review
    t = Transcript()
    t.reviews_api("06:01:00", [rest_review(21, COPILOT, "05:50:00")], pr=pr, u=usage(1, 10, 100, 5))
    t.assistant("06:05:00", u=usage(2, 20, 200, 10))
    for line in t.lines:
        line["sessionId"] = f"sess-{pr}"
    return t


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.project = self.root / "proj"
        self.project.mkdir()
        self.db = self.root / "metrics.db"

    def add(self, transcript, name="a.jsonl", **kw):
        return transcript.write(self.project, name, **kw)

    def store(self):
        return metrics_lib.MetricsStore(self.db)

    def run_backfill(self, gh=None, store=None, **kw):
        return B.backfill(self.project, repo="Donelle/claude-runway", store=store,
                          gh=gh if gh is not None else BackfillGh(), **kw)

    def count(self, store):
        return store.summary(P.METRIC_ID)["event_count"]


class DryRun(_Base):
    def test_a_dry_run_reports_the_plan_and_writes_nothing(self):
        self.add(three_round_transcript())
        store = self.store()
        with mock.patch.object(metrics_lib, "MetricsStore", side_effect=AssertionError("a dry run must not open the store")):
            report = self.run_backfill()
        self.assertFalse(report["write"])
        self.assertEqual((report["transcripts_found"], report["transcripts_read"]), (1, 1))
        self.assertEqual((report["transcripts_with_feedback"], report["transcripts_without_feedback"]), (1, 0))
        self.assertEqual((report["prs_found"], report["prs_recorded"]), (1, 1))
        self.assertEqual((report["rounds"], report["fetch_events"], report["events_built"]), (3, 3, 6))
        self.assertEqual((report["written"], report["existing"], report["failed"]), (0, 0, 0))
        self.assertEqual(report["skipped"], [])
        self.assertEqual(report["prs"][PR]["state"], "MERGED")
        self.assertEqual(report["prs"][PR]["end"], iso("04:20:28"))
        self.assertEqual(self.count(store), 0)

    def test_the_dry_run_is_the_default_of_the_cli_and_names_what_it_would_do(self):
        self.add(three_round_transcript())
        with mock.patch.object(P, "run_gh", BackfillGh()):
            out = self.cli("--project-dir", str(self.project), "--repo", "Donelle/claude-runway")
        self.assertIn("DRY RUN", out)
        self.assertIn("PR 603: MERGED", out)
        self.assertIn("3 round(s), 3 fetch event(s)", out)
        self.assertFalse(self.db.exists())

    def cli(self, *argv, expect=0):
        buf, err = StringIO(), StringIO()
        with redirect_stdout(buf), redirect_stderr(err):
            code = CLI.main(list(argv))
        self.assertEqual(code, expect, err.getvalue() + buf.getvalue())
        return buf.getvalue()


class Write(_Base):
    def test_a_real_run_writes_the_events_a_single_transcript_run_would(self):
        self.add(three_round_transcript())
        store = self.store()
        report = self.run_backfill(store=store, write=True)
        self.assertEqual((report["written"], report["existing"], report["failed"]), (6, 0, 0))
        self.assertEqual(self.count(store), 6)
        rows = store.detail(P.METRIC_ID, limit=100)
        self.assertEqual(sorted(r["event_type"] for r in rows), [P.EVENT_FETCH] * 3 + [P.EVENT_ROUND] * 3)
        # What the recorder writes for the same transcript and merge time, byte for byte.
        direct = metrics_lib.MetricsStore(self.root / "direct.db")
        P.record_pr_rounds(self.project / "a.jsonl", PR, store=direct, gh=BackfillGh(), merge_time=iso("04:20:28"))
        strip = lambda s: sorted(json.dumps([r["event_type"], r["metadata"]], sort_keys=True) for r in s.detail(P.METRIC_ID, limit=100))
        self.assertEqual(strip(store), strip(direct))

    def test_rows_carry_the_repo_and_a_per_round_reviewer_flag(self):
        t = three_round_transcript()
        t.review_call("04:15:00", {"pr": PR})   # between the second and third Copilot feedback
        self.add(t)
        store = self.store()
        self.run_backfill(store=store, write=True)
        rows = store.detail(P.METRIC_ID, limit=100)
        for r in rows:
            self.assertEqual((r["metadata"]["repo"], r["metadata"]["repo_source"]), ("Donelle/claude-runway", "remote"))
        copilot = [r["metadata"] for r in rows if r["event_type"] == P.EVENT_ROUND and r["metadata"]["kind"] == "copilot"][::-1]
        self.assertEqual([m["local_reviewer_used"] for m in copilot], [False, True, False])

    def test_without_a_repo_the_backfill_falls_back_to_the_checkout_remote_then_the_project(self):
        self.add(three_round_transcript())
        store = self.store()
        B.backfill(self.project, store=store, gh=BackfillGh(remote="Acme/widgets"), write=True)
        self.assertEqual({r["metadata"]["repo"] for r in store.detail(P.METRIC_ID, limit=100)}, {"Acme/widgets"})
        t = three_round_transcript()
        for line in t.lines:
            line["cwd"] = "/home/me/proj-x"
        other = self.root / "other"
        other.mkdir()
        t.write(other, "a.jsonl")
        second = metrics_lib.MetricsStore(self.root / "second.db")
        B.backfill(other, store=second, gh=BackfillGh(remote=None), write=True)
        self.assertEqual({(r["metadata"]["repo"], r["metadata"]["repo_source"]) for r in second.detail(P.METRIC_ID, limit=100)},
                         {("proj-x", "project")})

    def test_a_second_run_writes_nothing(self):
        self.add(three_round_transcript())
        store = self.store()
        self.run_backfill(store=store, write=True)
        again = self.run_backfill(store=store, write=True)
        self.assertEqual((again["written"], again["existing"], again["failed"]), (0, 6, 0))
        self.assertEqual(self.count(store), 6)

    def test_a_run_after_the_live_capture_wrote_the_same_rows_writes_nothing(self):
        path = self.add(three_round_transcript())
        store = self.store()
        P.record_pr_rounds(path, PR, store=store, gh=BackfillGh(), merge_time=iso("04:20:28"))
        again = self.run_backfill(store=store, write=True)
        self.assertEqual((again["written"], again["existing"]), (0, 6))

    def test_a_store_that_fails_stops_the_run_and_says_so(self):
        self.add(three_round_transcript(), "a.jsonl")
        self.add(other_pr_transcript(604), "b.jsonl")
        failing = mock.Mock()
        failing.record_if_absent.return_value = "failed"
        gh = BackfillGh(lookups={PR: {"state": "MERGED", "mergedAt": iso("04:20:28")},
                                 604: {"state": "MERGED", "mergedAt": iso("06:10:00")}})
        report = self.run_backfill(gh=gh, store=failing, write=True)
        self.assertIsNotNone(report["store_error"])
        self.assertEqual(failing.record_if_absent.call_count, 1)
        self.assertEqual(report["written"], 0)
        self.assertTrue(any("store failed earlier" in s["reason"] for s in report["skipped"]))


class EndOfTheWindow(_Base):
    def window_ends(self, store):
        return [(r["metadata"]["window_end"], r["metadata"]["window_open"])
                for r in store.detail(P.METRIC_ID, limit=100) if r["event_type"] == P.EVENT_ROUND]

    def test_a_merge_after_the_transcript_ends_is_clamped_to_its_last_line(self):
        self.add(three_round_transcript())
        store = self.store()
        gh = BackfillGh(lookups={PR: {"state": "MERGED", "mergedAt": "2026-11-09T00:00:00Z"}})
        report = self.run_backfill(gh=gh, store=store, write=True)
        # Not stretched across the three days, and a closed window (the session is over).
        self.assertEqual(report["prs"][PR]["end"], "2026-11-09T00:00:00Z")
        self.assertEqual(sorted(self.window_ends(store)),
                         [(iso("04:11:05"), False), (iso("04:19:59"), False), (iso("04:20:28"), False)])

    def test_the_clamp_keeps_the_fractional_seconds_of_the_last_line(self):
        # Real transcript timestamps carry milliseconds; a clamp to the whole second would cut
        # the last message (and any call) of that second out of the window for good.
        from test_pr_round_cost_lib import COPILOT, rest_review
        t = Transcript()
        t.reviews_api("04:01:15.100", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 100, 1000, 10))
        t.assistant("04:02:00.250", u=usage(2, 200, 2000, 77))
        self.add(t)
        store = self.store()
        gh = BackfillGh(lookups={PR: {"state": "MERGED", "mergedAt": "2026-11-09T00:00:00Z"}})
        self.run_backfill(gh=gh, store=store, write=True)
        round_row = [r for r in store.detail(P.METRIC_ID, limit=100) if r["event_type"] == P.EVENT_ROUND][0]
        self.assertEqual(round_row["metadata"]["messages"], 2)
        self.assertEqual(round_row["metadata"]["output_tokens"], 87)
        self.assertFalse(round_row["metadata"]["window_open"])

    def test_a_merge_inside_the_transcript_cuts_it_there(self):
        self.add(three_round_transcript())
        store = self.store()
        gh = BackfillGh(lookups={PR: {"state": "MERGED", "mergedAt": iso("04:12:00")}})
        self.run_backfill(gh=gh, store=store, write=True)
        ends = sorted(e for e, _ in self.window_ends(store))
        self.assertEqual(ends[-1], iso("04:12:00"))

    def test_an_open_pr_is_skipped_and_the_reason_says_why(self):
        self.add(three_round_transcript())
        store = self.store()
        gh = BackfillGh(lookups={PR: {"state": "OPEN", "mergedAt": None, "closedAt": None}})
        report = self.run_backfill(gh=gh, store=store, write=True)
        self.assertEqual(self.count(store), 0)
        self.assertEqual(report["prs_recorded"], 0)
        self.assertEqual(report["prs_found"], 1)
        self.assertEqual(len(report["skipped"]), 1)
        self.assertIn("still open", report["skipped"][0]["reason"])
        self.assertEqual(report["skipped"][0]["pr"], PR)

    def test_a_pr_merged_before_the_session_fetched_its_feedback_has_nothing_to_record(self):
        self.add(three_round_transcript())
        store = self.store()
        gh = BackfillGh(lookups={PR: {"state": "MERGED", "mergedAt": iso("03:00:00")}})
        report = self.run_backfill(gh=gh, store=store, write=True)
        self.assertEqual(self.count(store), 0)
        self.assertEqual((report["prs_found"], report["prs_recorded"], report["rounds"]), (1, 0, 0))
        self.assertIn("before the PR's merge", report["skipped"][0]["reason"])

    def test_the_unrecognised_fetches_are_counted_whatever_becomes_of_the_pair(self):
        t = three_round_transcript()
        t.bash("04:30:00", f"timeout 30 gh api repos/Donelle/claude-runway/pulls/{PR}/reviews", "[]")
        t.assistant("04:31:00")
        self.add(t)
        gh = BackfillGh(lookups={PR: {"state": "OPEN", "mergedAt": None, "closedAt": None}})
        report = self.run_backfill(gh=gh)
        self.assertEqual(report["unrecognised_fetches"], 1)
        self.assertIn("still open", report["skipped"][0]["reason"])

    def test_include_open_records_it_as_an_open_window(self):
        self.add(three_round_transcript())
        store = self.store()
        gh = BackfillGh(lookups={PR: {"state": "OPEN", "mergedAt": None, "closedAt": None}})
        report = self.run_backfill(gh=gh, store=store, write=True, include_open=True)
        self.assertEqual(report["written"], 6)
        # Only the last round's window ends at the transcript's last line, and only it can still grow.
        self.assertEqual(sorted(self.window_ends(store)),
                         [(iso("04:11:05"), False), (iso("04:19:59"), False), (iso("04:20:28"), True)])

    def test_a_pr_closed_without_merging_ends_at_its_closing(self):
        self.add(three_round_transcript())
        store = self.store()
        gh = BackfillGh(lookups={PR: {"state": "CLOSED", "mergedAt": None, "closedAt": iso("04:12:00")}})
        report = self.run_backfill(gh=gh, store=store, write=True)
        self.assertEqual(report["prs"][PR]["state"], "CLOSED")
        self.assertEqual(report["prs"][PR]["end"], iso("04:12:00"))
        self.assertEqual(sorted(e for e, _ in self.window_ends(store))[-1], iso("04:12:00"))


class UnusableInput(_Base):
    def test_a_pr_that_cannot_be_looked_up_is_reported_and_skipped(self):
        self.add(three_round_transcript(), "a.jsonl")
        self.add(other_pr_transcript(604), "b.jsonl")
        store = self.store()
        gh = BackfillGh(lookups={PR: {"state": "MERGED", "mergedAt": iso("04:20:28")},
                                 604: "gh exited 1: Could not resolve to a PullRequest with the number of 604"})
        report = self.run_backfill(gh=gh, store=store, write=True)
        self.assertEqual(report["prs_found"], 2)
        self.assertEqual(report["prs_recorded"], 1)
        skipped = [s for s in report["skipped"] if s["pr"] == 604]
        self.assertEqual(len(skipped), 1)
        self.assertIn("could not be looked up", skipped[0]["reason"])
        self.assertIn("Could not resolve", skipped[0]["reason"])
        # Nothing of the unlookable PR is written, the other one is.
        self.assertEqual({r["metadata"]["pr"] for r in store.detail(P.METRIC_ID, limit=100)}, {PR})

    def test_a_gh_that_prints_nonsense_is_a_lookup_failure_not_a_crash(self):
        gh = mock.Mock(return_value="not json")
        self.assertFalse(B.lookup_pr(PR, gh)["ok"])
        gh = mock.Mock(return_value=json.dumps({"state": "WEIRD"}))
        self.assertIn("unexpected state", B.lookup_pr(PR, gh)["error"])
        gh = mock.Mock(return_value=json.dumps({"state": "MERGED", "mergedAt": None}))
        self.assertIn("no usable time", B.lookup_pr(PR, gh)["error"])
        gh = mock.Mock(side_effect=P.GhError("gh could not run: FileNotFoundError"))
        self.assertIn("gh could not run", B.lookup_pr(PR, gh)["error"])

    def test_a_malformed_transcript_is_skipped_and_the_rest_still_run(self):
        self.add(three_round_transcript(), "good.jsonl")
        (self.project / "bad.jsonl").write_bytes(b'{"type": "assistant", "message": {"content": [\n\xff\xfe not json at all\n')
        (self.project / "empty.jsonl").write_bytes(b"")
        store = self.store()
        report = self.run_backfill(store=store, write=True)
        self.assertEqual(report["transcripts_found"], 3)
        self.assertEqual(report["transcripts_read"], 3)
        self.assertEqual((report["transcripts_with_feedback"], report["transcripts_without_feedback"]), (1, 2))
        self.assertEqual(report["written"], 6)

    def test_a_garbled_line_inside_a_good_transcript_is_skipped_not_the_transcript(self):
        self.add(three_round_transcript(), "a.jsonl", extra_raw=['{"broken": ', "\x00\x01"])
        report = self.run_backfill()
        self.assertEqual((report["rounds"], report["fetch_events"]), (3, 3))

    def test_a_transcript_that_cannot_be_opened_is_listed_as_unreadable(self):
        self.add(three_round_transcript(), "a.jsonl")
        self.add(three_round_transcript(), "b.jsonl")
        real = B.scan_transcript
        with mock.patch.object(B, "scan_transcript", side_effect=lambda p: (
                {"ok": False, "prs": set(), "last_ts": None, "bash_ids": set()} if str(p).endswith("b.jsonl") else real(p))):
            report = self.run_backfill()
        self.assertEqual(report["transcripts_unreadable"], ["b.jsonl"])
        self.assertEqual((report["transcripts_found"], report["transcripts_read"]), (2, 1))
        self.assertTrue(any("could not be opened" in s["reason"] for s in report["skipped"]))
        self.assertFalse(B.scan_transcript(self.project / "missing.jsonl")["ok"])

    def test_a_number_in_a_command_that_is_no_fetch_costs_no_gh_call(self):
        t = Transcript()
        t.bash("04:00:00", "gh pr view 77 -R Donelle/claude-runway --json state", '{"state":"OPEN"}')
        t.assistant("04:01:00")
        self.add(t)
        gh = BackfillGh()
        report = self.run_backfill(gh=gh)
        self.assertEqual(gh.calls, [])
        self.assertEqual((report["prs_found"], report["candidates_without_fetch"]), (0, 1))
        self.assertEqual(report["transcripts_without_feedback"], 1)

    def test_an_unqualified_local_review_is_not_attributed_to_an_incidental_pr_number(self):
        # `find_rounds` reads a `review` call that names no PR for EVERY PR it is asked about, so a
        # number in some other command must not be enough to record that review against it.
        t = Transcript()
        t.bash("04:00:00", "gh pr view 77 -R Donelle/claude-runway --json state", '{"state":"MERGED"}')
        t.review_call("04:05:00")
        t.assistant("04:06:00")
        self.add(t)
        store = self.store()
        gh = BackfillGh(lookups={77: {"state": "MERGED", "mergedAt": iso("04:30:00")}})
        report = self.run_backfill(gh=gh, store=store, write=True)
        self.assertEqual(self.count(store), 0)
        self.assertEqual((report["prs_found"], report["candidates_without_fetch"]), (0, 1))
        self.assertEqual(report["transcripts_without_feedback"], 1)
        self.assertEqual(gh.calls, [])

    def test_a_local_review_that_names_the_pr_is_enough_by_itself(self):
        t = Transcript()
        t.review_call("04:05:00", {"pr": PR})
        t.assistant("04:06:00")
        self.add(t)
        store = self.store()
        report = self.run_backfill(store=store, write=True)
        self.assertEqual((report["prs_found"], report["rounds"]), (1, 1))
        row = store.detail(P.METRIC_ID, limit=10)[0]
        self.assertEqual((row["event_type"], row["metadata"]["kind"]), (P.EVENT_ROUND, "local"))

    def test_a_local_review_that_names_another_pr_does_not_count_for_this_one(self):
        t = Transcript()
        t.bash("04:00:00", "gh pr view 77 -R Donelle/claude-runway --json state", "{}")
        t.review_call("04:05:00", {"pr": 78})
        t.assistant("04:06:00")
        self.add(t)
        gh = BackfillGh(lookups={77: {"state": "MERGED", "mergedAt": iso("04:30:00")}, 78: {"state": "MERGED", "mergedAt": iso("04:30:00")}})
        report = self.run_backfill(gh=gh)
        self.assertEqual(sorted(report["prs"]), [78])

    def test_a_fetch_the_reader_does_not_recognise_is_counted_and_not_guessed_at(self):
        t = Transcript()
        t.bash("04:00:00", f"timeout 30 gh api repos/Donelle/claude-runway/pulls/{PR}/reviews", "[]")
        t.assistant("04:01:00")
        self.add(t)
        gh = BackfillGh()
        report = self.run_backfill(gh=gh)
        self.assertEqual(gh.calls, [])
        self.assertEqual(report["unrecognised_fetches"], 1)
        self.assertEqual(report["transcripts_without_feedback"], 1)
        self.assertIn("does not recognise", report["skipped"][0]["reason"])

    def test_subagent_transcripts_and_other_files_are_not_read(self):
        self.add(three_round_transcript(), "a.jsonl")
        sub = self.project / "a" / "subagents"
        sub.mkdir(parents=True)
        three_round_transcript().write(sub, "agent-1.jsonl")
        (self.project / "notes.txt").write_text("not a transcript")
        (self.project / "dir.jsonl").mkdir()
        report = self.run_backfill()
        self.assertEqual(report["transcripts_found"], 1)

    def test_a_missing_project_folder_has_no_transcripts_and_a_bad_repo_is_refused(self):
        report = B.backfill(self.root / "nowhere", repo="Donelle/claude-runway", gh=BackfillGh())
        self.assertEqual(report["transcripts_found"], 0)
        refused = B.backfill(self.project, repo="not a repo", gh=BackfillGh())
        self.assertIn("owner/name", refused["store_error"])


class ReadsOnce(_Base):
    def test_one_pr_in_two_transcripts_is_asked_of_github_once(self):
        self.add(three_round_transcript(), "a.jsonl")
        self.add(three_round_transcript(), "b.jsonl")
        gh = BackfillGh()
        self.run_backfill(gh=gh)
        self.assertEqual(len(gh.calls_to("/reviews")), 1)
        self.assertEqual(len([c for c in gh.calls if c[-1] == LOOKUP_FIELDS]), 1)

    def test_a_gh_failure_is_one_attempt_not_one_per_transcript(self):
        self.add(three_round_transcript(), "a.jsonl")
        self.add(three_round_transcript(), "b.jsonl")
        calls = []

        def gh(args):
            calls.append(list(args))
            raise P.GhError("gh exited 4: offline")
        report = self.run_backfill(gh=gh)
        self.assertEqual(len([c for c in calls if c[-1] == LOOKUP_FIELDS]), 1)
        self.assertEqual([s["pr"] for s in report["skipped"]], [PR, PR])


class SharedHistory(_Base):
    def test_two_transcripts_sharing_tool_use_ids_are_reported(self):
        self.add(three_round_transcript(), "a.jsonl")
        self.add(three_round_transcript(), "b.jsonl")
        report = self.run_backfill()
        self.assertEqual(len(report["shared_tool_use_ids"]), 1)
        pair = report["shared_tool_use_ids"][0]
        self.assertEqual((pair["a"], pair["b"]), ("a.jsonl", "b.jsonl"))
        self.assertEqual(pair["ids"], 3)
        self.assertIn("WARNING: a.jsonl and b.jsonl share 3", CLI.format_text(report))

    def test_transcripts_with_their_own_history_report_nothing(self):
        t1 = three_round_transcript()
        t2 = other_pr_transcript(604)
        for line in t2.lines:  # ids from another numbering, as two real sessions have
            for block in (line.get("message") or {}).get("content", []):
                if isinstance(block, dict) and "id" in block:
                    block["id"] = "other_" + block["id"]
                if isinstance(block, dict) and "tool_use_id" in block:
                    block["tool_use_id"] = "other_" + block["tool_use_id"]
        self.add(t1, "a.jsonl")
        self.add(t2, "b.jsonl")
        self.assertEqual(self.run_backfill()["shared_tool_use_ids"], [])


class Discovery(_Base):
    def scan(self, t):
        return B.scan_transcript(self.add(t, "s.jsonl"))

    def test_candidates_come_from_commands_markers_and_review_calls(self):
        t = Transcript()
        t.bash("04:00:00", "gh api repos/o/r/pulls/11/reviews --paginate", "[]")
        t.bash("04:00:01", "gh api repos/o/r/issues/12/comments", "[]")
        t.bash("04:00:02", "gh pr view 13 -R o/r --json reviews", "{}")
        t.bash("04:00:03", "echo pr-feedback-fetch: pr=14 copilot-review=none", "pr-feedback-fetch: pr=14 copilot-review=none")
        t.review_call("04:00:04", {"pr": "#15"})
        t.review_call("04:00:05", {"pr": 16})
        t.review_call("04:00:06", {"base_ref": "main"})
        t.assistant("04:00:07", blocks=[{"type": "text", "text": "pr-feedback-fetch: pr=99 copilot-review=none"}])
        t.call("04:00:08", "Read", {"file_path": "/x"}, "pr-feedback-fetch: pr=98 copilot-review=none")
        t.bash("04:00:09", "git log --oneline", "pulls/97/comments is just text here")
        scanned = self.scan(t)
        self.assertTrue(scanned["ok"])
        self.assertEqual(scanned["prs"], {11, 12, 13, 14, 15, 16})
        self.assertEqual(len(scanned["bash_ids"]), 5)
        self.assertEqual(scanned["last_ts"], B._epoch(iso("04:00:09")))

    def test_an_absurd_number_in_a_transcript_is_no_candidate_and_no_crash(self):
        t = Transcript()
        t.bash("04:00:00", f"gh api repos/o/r/pulls/{'9' * 5000}/reviews", "[]")
        t.bash("04:00:01", "echo x", f"pr-feedback-fetch: pr={'9' * 5000} copilot-review=none")
        t.review_call("04:00:02", {"pr": "²"})
        t.review_call("04:00:03", {"pr": 10 ** 30})
        t.review_call("04:00:04", {"pr": True})
        t.assistant("04:00:05")
        self.assertEqual(self.scan(t)["prs"], set())

    def test_a_result_with_an_odd_tool_use_id_does_not_stop_the_scan(self):
        t = Transcript()
        t.bash("04:00:00", "gh api repos/o/r/pulls/11/reviews", "[]")
        t.assistant("04:00:01")
        path = self.add(t, "odd.jsonl", extra_raw=[
            json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": ["x"], "content": "y"}]}}),
            json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": {"a": 1}, "content": 7}]}}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": 5, "name": "Bash", "input": []}]}}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "z", "name": "Bash", "input": {"command": 7}}]}}),
        ])
        self.assertEqual(B.scan_transcript(path)["prs"], {11})

    def test_a_transcript_that_names_nothing_has_no_candidates(self):
        t = Transcript()
        t.assistant("04:00:00")
        scanned = self.scan(t)
        self.assertEqual((scanned["ok"], scanned["prs"]), (True, set()))


class ProjectFolders(unittest.TestCase):
    def test_a_project_folder_name_is_the_path_with_every_other_character_a_hyphen(self):
        if os.name == "nt":
            # os.path.abspath keeps the drive on Windows, and the colon becomes a hyphen like any other character.
            self.assertEqual(B.project_folder_name(r"C:\Users\me\repos\My_Repo.x"), "C--Users-me-repos-My-Repo-x")
        else:
            self.assertEqual(B.project_folder_name("/Users/me/repos/My_Repo.x"), "-Users-me-repos-My-Repo-x")

    def test_the_default_folder_is_the_checkouts_under_the_projects_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "work" / "a_repo"
            checkout.mkdir(parents=True)
            got = B.default_project_dir(checkout, projects_root="/root/projects")
            self.assertEqual(got, os.path.join("/root/projects", B.project_folder_name(checkout)))

    def test_from_a_linked_worktree_the_default_is_the_main_checkouts_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            main = Path(tmp) / "main"
            main.mkdir()
            git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-C", str(main)]
            try:
                subprocess.run([*git, "init", "-q"], check=True, capture_output=True)
                subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True, capture_output=True)
                tree = Path(tmp) / "tree"
                subprocess.run([*git, "worktree", "add", "-q", str(tree), "-b", "side"], check=True, capture_output=True)
            except (OSError, subprocess.CalledProcessError):
                self.skipTest("git is not available here")
            got = B.default_project_dir(tree, projects_root="/root/projects")
            self.assertEqual(got, os.path.join("/root/projects", B.project_folder_name(Path(os.path.realpath(main)))))


class Cli(_Base):
    def cli(self, *argv, expect=0):
        buf, err = StringIO(), StringIO()
        with redirect_stdout(buf), redirect_stderr(err):
            code = CLI.main(list(argv))
        self.assertEqual(code, expect, err.getvalue() + buf.getvalue())
        return buf.getvalue(), err.getvalue()

    def test_write_records_and_a_rerun_reports_it_already_recorded(self):
        self.add(three_round_transcript())
        args = ("--project-dir", str(self.project), "--repo", "Donelle/claude-runway", "--write", "--db", str(self.db))
        with mock.patch.object(P, "run_gh", BackfillGh()):
            first, _ = self.cli(*args)
            second, _ = self.cli(*args)
        self.assertIn("written 6, already recorded 0", first)
        self.assertIn("written 0, already recorded 6", second)
        self.assertEqual(self.count(self.store()), 6)

    def test_json_output_is_the_report(self):
        self.add(three_round_transcript())
        with mock.patch.object(P, "run_gh", BackfillGh()):
            out, _ = self.cli("--project-dir", str(self.project), "--format", "json")
        report = json.loads(out)
        self.assertEqual((report["transcripts_found"], report["rounds"], report["write"]), (1, 3, False))

    def test_bad_arguments_and_a_missing_folder(self):
        self.cli("--project-dir", str(self.project), "--repo", "nope", expect=2)
        self.cli("--project-dir", str(self.project), "--db", str(self.db), expect=2)
        _, err = self.cli("--project-dir", str(self.root / "gone"), expect=1)
        self.assertIn("no transcripts folder", err)

    def test_a_store_failure_is_a_nonzero_exit(self):
        self.add(three_round_transcript())
        with mock.patch.object(P, "run_gh", BackfillGh()), \
                mock.patch.object(metrics_lib.MetricsStore, "record_if_absent", return_value="failed"):
            out, _ = self.cli("--project-dir", str(self.project), "--write", "--db", str(self.db), expect=1)
        self.assertIn("Store problem", out)

    def test_text_report_lists_skips_and_counts(self):
        self.add(three_round_transcript(), "a.jsonl")
        self.add(other_pr_transcript(604), "b.jsonl")
        gh = BackfillGh(lookups={PR: {"state": "MERGED", "mergedAt": iso("04:20:28")}, 604: "gh exited 1: no pull request found"})
        with mock.patch.object(P, "run_gh", gh):
            out, _ = self.cli("--project-dir", str(self.project))
        self.assertIn("Skipped:", out)
        self.assertIn("b.jsonl PR 604: the PR could not be looked up", out)
        self.assertIn("Transcripts: 2 found, 2 read, 0 unreadable, 2 with a feedback fetch, 0 without", out)
        self.assertIn("PRs: 2 found, 1 that would be recorded", out)


if __name__ == "__main__":
    unittest.main()
