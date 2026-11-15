"""Tests for libs/pr_round_record_lib.py and tools/record_pr_rounds.py: the review rounds of a
PR, found in a session transcript by libs/pr_round_cost_lib.py, recorded to the shared metrics
store and enriched from GitHub. Synthetic transcripts (the builders of
test_pr_round_cost_lib.py), a stubbed `gh` and a temporary metrics.db: no network, no real
~/.claude, no live model."""

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

import metrics_lib  # noqa: E402
import pr_round_record_lib as P  # noqa: E402
import record_pr_rounds as CLI  # noqa: E402
from test_pr_round_cost_lib import (  # noqa: E402
    COPILOT, ME, PR, Transcript, epoch, iso, rest_review, usage, view_review)

SHA1 = "1" * 40
SHA2 = "2" * 40
SHA3 = "3" * 40


def gh_review(rid, node, login, ts, sha, body="", effort=None, state="COMMENTED"):
    return {"id": rid, "node_id": node, "user": {"login": login}, "state": state, "submitted_at": iso(ts),
            "commit_id": sha, "body": body, "copilot_code_review_effort": effort,
            "is_copilot_code_review": login == COPILOT}


BODY_TWO = ("### Changes recommended\n<details open>\n<summary><strong>2 open issues</strong></summary>\n</details>\n"
            "<sub>Review effort: Balanced</sub>")
BODY_NONE = "### Approval recommended\n**No open issues**\n<sub>Review effort: Balanced</sub>"


class FakeGh:
    """A stand-in for `gh`: answers the four questions the recorder asks, from fixtures."""

    def __init__(self, reviews=None, comments=None, size=None, suites=None, fail=(),
                 remote="Donelle/claude-runway"):
        self.reviews = reviews if reviews is not None else [
            gh_review(11, "PRR_a", COPILOT, "03:27:21", SHA1, BODY_TWO, "balanced"),
            gh_review(12, "PRR_b", ME, "04:07:21", SHA2),
            gh_review(13, "PRR_c", COPILOT, "04:10:15", SHA2, BODY_TWO, "balanced"),
            gh_review(15, "PRR_d", COPILOT, "04:19:20", SHA3, BODY_NONE, "balanced", state="APPROVED"),
        ]
        # Two inline comments on review 11, one on review 13, none on 15; one reply by the user.
        self.comments = comments if comments is not None else [11, 11, 13, 12]
        self.size = size if size is not None else {"additions": 954, "deletions": 113, "changedFiles": 9}
        self.suites = suites if suites is not None else {
            SHA1: [iso("03:20:00"), iso("03:20:05")], SHA2: [iso("04:06:53"), iso("04:07:24")], SHA3: [iso("04:15:00")]}
        self.fail = set(fail)   # which sources fail: reviews, comments, size, suites, remote
        self.remote = remote    # what `gh repo view` names; None makes it fail
        self.calls: list = []

    def __call__(self, args):
        self.calls.append(list(args))
        if args[0] == "repo":
            if self.remote is None or "remote" in self.fail:
                raise P.GhError("gh exited 1: no remote repository found")
            return self.remote + "\n"
        if args[0] == "pr":
            name, out = "size", json.dumps(self.size)
        elif args[1].endswith("/reviews"):
            name, out = "reviews", "\n".join(json.dumps(r) for r in self.reviews)
        elif args[1].endswith("/comments"):
            name, out = "comments", "\n".join(json.dumps({"r": r}) for r in self.comments)
        else:
            sha = args[1].split("/commits/")[1].split("/")[0]
            name, out = "suites", "\n".join(self.suites.get(sha, []))
        if name in self.fail:
            raise P.GhError(f"gh exited 1: {name} unavailable")
        return out

    def calls_to(self, fragment):
        return [c for c in self.calls if fragment in " ".join(c)]


def three_round_transcript():
    t = Transcript()
    t.assistant("03:59:00")
    t.reviews_api("04:01:15", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 100, 1000, 10))
    t.assistant("04:02:00", u=usage(2, 200, 2000, 20))
    t.reviews_api("04:11:05", [rest_review(11, COPILOT, "03:27:21"), rest_review(12, ME, "04:07:21"),
                               rest_review(13, COPILOT, "04:10:15")], u=usage(3, 300, 3000, 30))
    t.assistant("04:12:00", u=usage(4, 400, 4000, 40))
    t.reviews_view("04:19:59", [view_review("PRR_a", COPILOT, "03:27:21"), view_review("PRR_b", ME, "04:17:28"),
                                view_review("PRR_c", COPILOT, "04:10:15"),
                                view_review("PRR_d", COPILOT, "04:19:20", state="APPROVED")],
                   latest=[view_review("", COPILOT, "04:19:20")], u=usage(5, 500, 5000, 50))
    t.assistant("04:20:28", u=usage(6, 600, 6000, 60))
    return t


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.db = Path(self.dir) / "metrics.db"
        self.store = metrics_lib.MetricsStore(self.db)

    def write(self, t, name="t.jsonl"):
        return t.write(self.dir, name)

    def record(self, path, gh=None, **kw):
        kw.setdefault("merge_time", iso("04:20:28"))
        return P.record_pr_rounds(path, PR, store=self.store, gh=gh or FakeGh(), **kw)

    def rows(self, event_type):
        return [r for r in self.store.detail(P.METRIC_ID, limit=1000) if r["event_type"] == event_type][::-1]


class ThreeRounds(_Base):
    def test_one_round_event_per_round_with_every_field(self):
        out = self.record(self.write(three_round_transcript()))
        self.assertTrue(out["ok"])
        self.assertEqual((out["rounds"], out["fetch_calls"], out["fetch_events"]), (3, 3, 3))
        self.assertEqual((out["written"], out["existing"], out["failed"]), (6, 0, 0))
        self.assertEqual(out["github_errors"], [])
        rounds = self.rows(P.EVENT_ROUND)
        self.assertEqual(len(rounds), 3)
        for r in rounds:
            self.assertEqual(r["value"], 1.0)
            self.assertEqual(r["session_id"], "sess-1")
        m1, m2, m3 = (r["metadata"] for r in rounds)
        self.assertEqual([m["round"] for m in (m1, m2, m3)], [1, 2, 3])
        self.assertEqual([m["copilot_review_number"] for m in (m1, m2, m3)], [1, 2, 3])
        # `review_id` is what the transcript showed (round 3 only by its node id); the GitHub id joins.
        self.assertEqual([m["review_id"] for m in (m1, m2, m3)], ["11", "13", "PRR_d"])
        self.assertEqual([m["github_review_id"] for m in (m1, m2, m3)], ["11", "13", "15"])
        self.assertEqual([m["kind"] for m in (m1, m2, m3)], ["copilot"] * 3)
        self.assertEqual([m["effort"] for m in (m1, m2, m3)], ["balanced"] * 3)
        self.assertEqual([m["commit_id"] for m in (m1, m2, m3)], [SHA1, SHA2, SHA3])
        self.assertEqual(m3["review_state"], "APPROVED")
        self.assertEqual(m1["submitted_at"], iso("03:27:21"))
        # Findings: inline comments counted by review id (the user's reply is review 12's, not Copilot's).
        self.assertEqual([m["inline_comments"] for m in (m1, m2, m3)], [2, 1, 0])
        self.assertEqual([m["body_open_issues"] for m in (m1, m2, m3)], [2, 2, 0])
        self.assertEqual([m["had_findings"] for m in (m1, m2, m3)], [True, True, False])
        # Push time is the earliest check suite for the reviewed commit; the wait runs to the review.
        self.assertEqual(m2["push_time"], iso("04:06:53"))
        self.assertEqual(m2["wait_from_push_s"], 202)
        self.assertEqual((m1["pr_additions"], m1["pr_deletions"], m1["pr_changed_files"]), (954, 113, 9))
        self.assertEqual(m1["github"], "ok")
        self.assertEqual(m1["schema"], P.SCHEMA_VERSION)
        # The window figures are the reader's, the four token fields kept apart.
        self.assertEqual((m1["messages"], m1["tool_calls"], m1["wall_clock_s"]), (2, 1, 590))
        self.assertEqual((m1["input_tokens"], m1["cache_creation_tokens"], m1["cache_read_tokens"], m1["output_tokens"]),
                         (3, 300, 3000, 30))
        self.assertEqual((m3["input_tokens"], m3["output_tokens"]), (11, 110))
        self.assertEqual(m1["model"], "claude-sonnet-5-5")
        self.assertEqual(m1["window_end"], iso("04:11:05"))
        self.assertEqual(m3["window_end"], iso("04:20:28"))
        self.assertEqual(m1["call_time"], iso("04:01:15"))
        # Copilot-only fields are filled; the local-only ones are empty.
        self.assertIsNone(m1["review_type"])
        self.assertEqual(rounds[0]["event_timestamp"], iso("04:01:15"))

    def test_one_fetch_event_per_fetch_call_inside_its_round(self):
        self.record(self.write(three_round_transcript()))
        fetches = self.rows(P.EVENT_FETCH)
        self.assertEqual(len(fetches), 3)
        self.assertEqual([f["metadata"]["round"] for f in fetches], [1, 2, 3])
        self.assertEqual([f["metadata"]["round_kind"] for f in fetches], ["copilot"] * 3)
        self.assertEqual([f["metadata"]["review_id"] for f in fetches], ["11", "13", "PRR_d"])
        self.assertTrue(all(f["metadata"]["recognised"] for f in fetches))
        self.assertEqual(fetches[0]["metadata"]["sources"], ["api_reviews"])
        self.assertTrue(all(f["metadata"]["tool_use_id"].startswith("toolu_") for f in fetches))
        # The first fetch is the whole first round here, so the window figures agree.
        round1 = self.rows(P.EVENT_ROUND)[0]["metadata"]
        self.assertEqual(fetches[0]["metadata"]["output_tokens"], round1["output_tokens"])

    def test_a_node_id_only_round_is_matched_to_its_github_review(self):
        # Round 3 was seen as PRR_d only; GitHub knows it as 15 / PRR_d.
        self.record(self.write(three_round_transcript()))
        self.assertEqual(self.rows(P.EVENT_ROUND)[2]["metadata"]["copilot_review_number"], 3)

    def test_a_rerun_over_the_same_transcript_writes_nothing(self):
        path = self.write(three_round_transcript())
        self.record(path)
        before = self.store.summary(P.METRIC_ID)["event_count"]
        again = self.record(path)
        self.assertEqual((again["written"], again["existing"], again["failed"]), (0, 6, 0))
        self.assertEqual(self.store.summary(P.METRIC_ID)["event_count"], before)
        self.assertEqual(before, 6)

    def test_a_rerun_that_cannot_reach_github_still_finds_what_was_written(self):
        path = self.write(three_round_transcript())
        self.record(path)
        again = self.record(path, gh=FakeGh(fail={"reviews", "comments", "size", "suites"}))
        self.assertEqual((again["written"], again["existing"]), (0, 6))

    def test_dry_run_builds_the_events_and_writes_nothing(self):
        out = self.record(self.write(three_round_transcript()), dry_run=True)
        self.assertEqual(len(out["events"]), 6)
        self.assertEqual(out["written"], 0)
        self.assertEqual(self.store.summary(P.METRIC_ID)["event_count"], 0)


class RoundsSpanningTwoTranscripts(_Base):
    def two_sessions(self):
        first = Transcript()
        first.reviews_api("04:01:15", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 100, 1000, 10))
        first.reviews_api("04:11:05", [rest_review(11, COPILOT, "03:27:21"), rest_review(13, COPILOT, "04:10:15")],
                          u=usage(3, 300, 3000, 30))
        first.assistant("04:12:00", u=usage(4, 400, 4000, 40))
        second = Transcript()
        second.reviews_view("04:19:59", [view_review("PRR_d", COPILOT, "04:19:20", state="APPROVED")],
                            u=usage(5, 500, 5000, 50))
        second.assistant("04:20:28", u=usage(6, 600, 6000, 60))
        for line in second.lines:
            line["sessionId"] = "sess-2"
        return self.write(first, "a.jsonl"), self.write(second, "b.jsonl")

    def test_each_round_is_recorded_once_and_a_rerun_writes_nothing(self):
        a, b = self.two_sessions()
        # First transcript ends before the third review, so it is cut where the second round's window ends.
        r1 = self.record(a, merge_time=iso("04:12:00"))
        r2 = self.record(b)
        self.assertEqual((r1["rounds"], r2["rounds"]), (2, 1))
        rounds = self.rows(P.EVENT_ROUND)
        self.assertEqual([r["metadata"]["review_id"] for r in rounds], ["11", "13", "PRR_d"])
        self.assertEqual([r["metadata"]["copilot_review_number"] for r in rounds], [1, 2, 3])
        self.assertEqual([r["session_id"] for r in rounds], ["sess-1", "sess-1", "sess-2"])
        # The second session's own numbering starts again at 1; the GitHub ordinal does not.
        self.assertEqual([r["metadata"]["round"] for r in rounds], [1, 2, 1])
        before = self.store.summary(P.METRIC_ID)["event_count"]
        self.record(a, merge_time=iso("04:12:00"))
        self.record(b)
        self.assertEqual(self.store.summary(P.METRIC_ID)["event_count"], before)

    def test_a_review_both_sessions_fetched_keeps_each_sessions_own_spend(self):
        a, _ = self.two_sessions()
        other = Transcript()
        other.reviews_api("05:00:00", [rest_review(13, COPILOT, "04:10:15")], u=usage(7, 700, 7000, 70))
        other.assistant("05:01:00", u=usage(8, 800, 8000, 80))
        for line in other.lines:
            line["sessionId"] = "sess-3"
        c = self.write(other, "c.jsonl")
        self.record(a, merge_time=iso("04:12:00"))
        self.record(c, merge_time=iso("05:02:00"))
        same_review = [r for r in self.rows(P.EVENT_ROUND) if r["metadata"]["review_id"] == "13"]
        self.assertEqual([r["session_id"] for r in same_review], ["sess-1", "sess-3"])
        self.assertEqual([r["metadata"]["copilot_review_number"] for r in same_review], [2, 2])
        self.assertNotEqual(same_review[0]["metadata"]["output_tokens"], same_review[1]["metadata"]["output_tokens"])


class GitHubFailures(_Base):
    def test_every_gh_call_failing_leaves_nulls_and_still_writes(self):
        gh = FakeGh(fail={"reviews", "comments", "size", "suites"})
        out = self.record(self.write(three_round_transcript()), gh=gh)
        self.assertTrue(out["ok"])
        self.assertEqual((out["written"], out["failed"]), (6, 0))
        self.assertTrue(out["github_errors"])
        m = self.rows(P.EVENT_ROUND)[0]["metadata"]
        for field in ("copilot_review_number", "effort", "inline_comments", "body_open_issues", "had_findings",
                      "push_time", "wait_from_push_s", "pr_additions", "pr_deletions", "pr_changed_files"):
            self.assertIsNone(m[field], field)
        self.assertEqual(m["github"], "unavailable")
        # What the transcript knows is still there.
        self.assertEqual(m["review_id"], "11")
        self.assertEqual(m["commit_id"], "abc1234")
        self.assertEqual(m["output_tokens"], 30)

    def test_one_failure_is_listed_once_not_once_per_round(self):
        gh = FakeGh(fail={"reviews"})
        out = self.record(self.write(three_round_transcript()), gh=gh)
        self.assertEqual(len([e for e in out["github_errors"] if e.startswith("reviews")]), 1)
        self.assertEqual(len(gh.calls_to("/reviews")), 1)

    def test_only_the_comments_failing_is_partial_and_keeps_the_rest(self):
        out = self.record(self.write(three_round_transcript()), gh=FakeGh(fail={"comments"}))
        m = self.rows(P.EVENT_ROUND)[1]["metadata"]
        self.assertEqual(m["github"], "partial")
        self.assertIsNone(m["inline_comments"])
        self.assertEqual(m["body_open_issues"], 2)
        self.assertTrue(m["had_findings"])  # the body alone says so
        self.assertEqual(m["effort"], "balanced")
        self.assertEqual(len(out["github_errors"]), 1)

    def test_no_check_suite_is_a_null_push_time_not_a_guess(self):
        out = self.record(self.write(three_round_transcript()), gh=FakeGh(suites={}))
        m = self.rows(P.EVENT_ROUND)[1]["metadata"]
        self.assertIsNone(m["push_time"])
        self.assertIsNone(m["wait_from_push_s"])
        self.assertEqual(out["github_errors"], [])

    def test_a_review_github_does_not_list_has_null_review_fields(self):
        out = self.record(self.write(three_round_transcript()), gh=FakeGh(reviews=[]))
        m = self.rows(P.EVENT_ROUND)[0]["metadata"]
        self.assertIsNone(m["effort"])
        self.assertIsNone(m["copilot_review_number"])
        self.assertEqual(out["written"], 6)

    def test_the_effort_falls_back_to_the_bodys_own_line(self):
        reviews = [gh_review(11, "PRR_a", COPILOT, "03:27:21", SHA1, BODY_TWO, None)]
        self.record(self.write(three_round_transcript()), gh=FakeGh(reviews=reviews))
        self.assertEqual(self.rows(P.EVENT_ROUND)[0]["metadata"]["effort"], "balanced")

    def test_an_older_body_without_an_issue_count_uses_the_inline_comments(self):
        reviews = [gh_review(11, "PRR_a", COPILOT, "03:27:21", SHA1, "Copilot reviewed 3 files.", None)]
        self.record(self.write(three_round_transcript()), gh=FakeGh(reviews=reviews, comments=[11]))
        m = self.rows(P.EVENT_ROUND)[0]["metadata"]
        self.assertIsNone(m["body_open_issues"])
        self.assertEqual((m["inline_comments"], m["had_findings"]), (1, True))
        self.record(self.write(three_round_transcript(), "u.jsonl"), gh=FakeGh(reviews=reviews, comments=[]), dry_run=True)

    def test_no_findings_anywhere_is_false_not_null(self):
        reviews = [gh_review(11, "PRR_a", COPILOT, "03:27:21", SHA1, BODY_NONE, "balanced")]
        self.record(self.write(three_round_transcript()), gh=FakeGh(reviews=reviews, comments=[]))
        self.assertIs(self.rows(P.EVENT_ROUND)[0]["metadata"]["had_findings"], False)

    def test_a_gh_binary_that_is_missing_is_a_gh_error(self):
        with mock.patch.object(P.subprocess, "run", side_effect=FileNotFoundError("gh")):
            with self.assertRaises(P.GhError):
                P.run_gh(["api", "x"])
        failed = subprocess.CompletedProcess(["gh"], 1, "", "not logged in")
        with mock.patch.object(P.subprocess, "run", return_value=failed):
            with self.assertRaises(P.GhError) as ctx:
                P.run_gh(["api", "x"])
        self.assertIn("not logged in", str(ctx.exception))

    def test_garbage_from_gh_is_an_error_entry_not_an_exception(self):
        class Garbage(FakeGh):
            def __call__(self, args):
                return "this is not json"
        out = self.record(self.write(three_round_transcript()), gh=Garbage())
        self.assertEqual(out["written"], 6)
        self.assertTrue(out["github_errors"])

    def test_a_malformed_repo_is_refused_before_any_call(self):
        with self.assertRaises(ValueError):
            P.GitHubFacts(1, FakeGh(), "not a repo; rm -rf")
        gh = FakeGh()
        out = self.record(self.write(three_round_transcript()), gh=gh, repo="o/r; rm -rf")
        self.assertEqual((out["written"], gh.calls), (0, []))
        self.assertIn("owner/name", out["store_error"])

    def test_a_repo_is_passed_to_gh_and_the_placeholder_otherwise(self):
        gh = FakeGh()
        self.record(self.write(three_round_transcript()), gh=gh, repo="o/r")
        self.assertTrue(all("repos/o/r/" in " ".join(c) or c[:2] == ["pr", "view"] for c in gh.calls))
        self.assertIn(["-R", "o/r"], [c[3:5] for c in gh.calls if c[0] == "pr"])
        gh2 = FakeGh()
        self.record(self.write(three_round_transcript(), "x.jsonl"), gh=gh2)
        self.assertTrue(any("{owner}/{repo}" in " ".join(c) for c in gh2.calls))

    def test_a_non_hex_commit_is_never_sent_to_gh(self):
        reviews = [gh_review(11, "PRR_a", COPILOT, "03:27:21", "--evil", BODY_TWO, "balanced")]
        gh = FakeGh(reviews=reviews)
        self.record(self.write(three_round_transcript()), gh=gh)
        self.assertEqual(gh.calls_to("check-suites"), [])


class StoreFailures(_Base):
    def test_an_unreadable_store_is_reported_not_raised_and_stops_early(self):
        blocking = Path(self.dir) / "not-a-directory"
        blocking.write_text("x")
        bad = metrics_lib.MetricsStore(blocking / "sub" / "metrics.db")
        err = StringIO()
        with redirect_stderr(err):
            out = P.record_pr_rounds(self.write(three_round_transcript()), PR, store=bad, gh=FakeGh(),
                                     merge_time=iso("04:20:28"))
        self.assertTrue(out["ok"])
        self.assertEqual((out["written"], out["failed"], out["not_attempted"]), (0, 1, 5))
        self.assertIn("could not be written", out["store_error"])
        self.assertEqual(err.getvalue().count("could not record metric"), 1)

    def test_an_unexpected_error_is_swallowed_into_the_summary(self):
        with mock.patch.object(P, "build_events", side_effect=RuntimeError("boom")), redirect_stderr(StringIO()):
            out = self.record(self.write(three_round_transcript()))
        self.assertEqual(out["written"], 0)
        self.assertIn("RuntimeError", out["store_error"])

    def test_an_unreadable_transcript_is_ok_false_and_writes_nothing(self):
        out = self.record(Path(self.dir) / "missing.jsonl")
        self.assertFalse(out["ok"])
        self.assertEqual(out["written"], 0)
        self.assertEqual(self.store.summary(P.METRIC_ID)["event_count"], 0)

    def test_a_transcript_with_no_rounds_writes_nothing_and_asks_github_nothing(self):
        gh = FakeGh()
        t = Transcript()
        t.assistant("04:00:00")
        out = self.record(self.write(t), gh=gh)
        self.assertTrue(out["ok"])
        self.assertEqual((out["rounds"], out["written"]), (0, 0))


class LocalRoundsAndOrphans(_Base):
    def test_a_local_round_carries_its_review_fields_and_no_copilot_ones(self):
        t = Transcript()
        t.review_call("04:00:00", params={"pr": PR}, result="x" * 40, rtype="reasoning")
        t.assistant("04:00:30", u=usage(2, 20, 200, 20))
        t.reviews_api("04:05:00", [rest_review(11, COPILOT, "04:04:00")], u=usage(3, 30, 300, 30))
        gh = FakeGh()
        self.record(self.write(t), gh=gh, merge_time=iso("04:10:00"))
        local, copilot = (r["metadata"] for r in self.rows(P.EVENT_ROUND))
        self.assertEqual(local["kind"], "local")
        self.assertEqual((local["review_type"], local["review_pr"], local["result_chars"]), ("reasoning", PR, 40))
        self.assertIsNone(local["review_id"])
        for field in ("effort", "inline_comments", "had_findings", "push_time", "copilot_review_number", "submitted_at"):
            self.assertIsNone(local[field], field)
        self.assertEqual((local["round"], local["kind_number"]), (1, 1))
        self.assertEqual(copilot["kind"], "copilot")
        self.assertEqual((copilot["round"], copilot["kind_number"]), (2, 1))
        self.assertEqual(local["output_tokens"], 5 + 20)
        # A local round asks GitHub for the PR's size only.
        self.assertEqual(local["pr_additions"], 954)

    def test_a_local_round_is_keyed_by_its_call_time(self):
        t = Transcript()
        t.review_call("04:00:00", params={"pr": PR})
        t.review_call("04:03:00", params={"pr": PR})
        path = self.write(t)
        self.record(path, merge_time=iso("04:10:00"))
        self.assertEqual(len(self.rows(P.EVENT_ROUND)), 2)
        self.record(path, merge_time=iso("04:10:00"))
        self.assertEqual(len(self.rows(P.EVENT_ROUND)), 2)

    def test_two_local_calls_in_one_response_are_two_events_and_a_rerun_writes_nothing(self):
        # Parallel `review` calls share a timestamp and have no review id: they must not collapse.
        t = Transcript()
        t.review_call("04:00:00", params={"pr": PR}, rtype="reasoning")
        t.review_call("04:00:00", params={"pr": PR}, rtype="verifier")
        t.assistant("04:00:30", u=usage(2, 20, 200, 20))
        path = self.write(t)
        out = self.record(path, merge_time=iso("04:10:00"))
        self.assertEqual((out["rounds"], out["written"], out["existing"]), (2, 2, 0))
        rows = self.rows(P.EVENT_ROUND)
        self.assertEqual([r["metadata"]["kind_number"] for r in rows], [1, 2])
        # The window with the tokens is not lost.
        self.assertEqual(sum(r["metadata"]["output_tokens"] for r in rows), 5 + 5 + 20)
        again = self.record(path, merge_time=iso("04:10:00"))
        self.assertEqual((again["written"], again["existing"]), (0, 2))

    def test_the_same_local_call_time_in_two_sessions_is_two_events(self):
        a, b = Transcript(), Transcript()
        a.review_call("04:00:00", params={"pr": PR})
        b.review_call("04:00:00", params={"pr": PR})
        for line in b.lines:
            line["sessionId"] = "sess-2"
        self.record(self.write(a, "a.jsonl"), merge_time=iso("04:10:00"))
        self.record(self.write(b, "b.jsonl"), merge_time=iso("04:10:00"))
        self.assertEqual([r["session_id"] for r in self.rows(P.EVENT_ROUND)], ["sess-1", "sess-2"])

    def test_the_fetch_total_is_the_readers_and_counts_an_unread_call(self):
        t = Transcript()
        t.bash("04:01:00", f"timeout 30 gh api repos/o/r/pulls/{PR}/reviews", "x")  # counted, not read
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21")])
        out = self.record(self.write(t), merge_time=iso("04:10:00"))
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"], out["fetch_events"]), (2, 1, 1))
        wrapped_only = Transcript()
        wrapped_only.bash("04:01:00", f"timeout 30 gh api repos/o/r/pulls/{PR}/reviews", "x")
        out = self.record(self.write(wrapped_only, "w.jsonl"), merge_time=iso("04:10:00"))
        self.assertEqual((out["fetch_calls"], out["skipped_fetches"], out["fetch_events"]), (1, 1, 0))

    def test_a_marker_only_fetch_takes_the_review_time_and_state_from_github(self):
        # The skill's echo call prints the marker line and its output shows nothing else.
        t = Transcript()
        t.bash("04:01:15", "echo 'pr-feedback-fetch: pr=603 copilot-review=11'", f"pr-feedback-fetch: pr={PR} copilot-review=11")
        t.assistant("04:02:00", u=usage(2, 20, 200, 20))
        self.record(self.write(t), merge_time=iso("04:10:00"))
        rnd = self.rows(P.EVENT_ROUND)[0]["metadata"]
        fetch = self.rows(P.EVENT_FETCH)[0]["metadata"]
        self.assertEqual(rnd["submitted_at"], iso("03:27:21"))
        self.assertEqual(rnd["review_state"], "COMMENTED")
        self.assertEqual(fetch["submitted_at"], iso("03:27:21"))
        self.assertEqual(rnd["review_id"], "11")  # the key field is untouched
        # With GitHub unreachable the same call keeps its nulls, and a rerun is the same event.
        other = metrics_lib.MetricsStore(Path(self.dir) / "offline.db")
        out = P.record_pr_rounds(self.write(t, "o.jsonl"), PR, store=other, gh=FakeGh(fail={"reviews"}),
                                 merge_time=iso("04:10:00"))
        self.assertEqual(out["written"], 2)
        self.assertIsNone(other.detail(P.METRIC_ID)[0]["metadata"]["submitted_at"])

    def test_the_transcripts_own_review_time_wins_over_githubs(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21", state="APPROVED")])
        gh = FakeGh(reviews=[gh_review(11, "PRR_a", COPILOT, "03:27:22", SHA1, BODY_TWO, "balanced", state="DISMISSED")])
        self.record(self.write(t), gh=gh, merge_time=iso("04:10:00"))
        rnd = self.rows(P.EVENT_ROUND)[0]["metadata"]
        self.assertEqual((rnd["submitted_at"], rnd["review_state"]), (iso("03:27:21"), "APPROVED"))

    def test_the_wait_uses_the_submission_time_the_row_stores(self):
        # Transcript and GitHub disagree by a second: the stored wait must follow the stored time.
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        gh = FakeGh(reviews=[gh_review(11, "PRR_a", COPILOT, "03:27:22", SHA1, BODY_TWO, "balanced")],
                    suites={SHA1: [iso("03:20:00")]})
        self.record(self.write(t), gh=gh, merge_time=iso("04:10:00"))
        rnd = self.rows(P.EVENT_ROUND)[0]["metadata"]
        self.assertEqual(rnd["submitted_at"], iso("03:27:21"))
        self.assertEqual(rnd["wait_from_push_s"], 441)
        # And a review the transcript places before the push is not given a negative wait by GitHub's later time.
        early = Transcript()
        early.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:19:59")])
        other = metrics_lib.MetricsStore(Path(self.dir) / "early.db")
        P.record_pr_rounds(self.write(early, "e.jsonl"), PR, store=other, gh=gh, merge_time=iso("04:10:00"))
        early_round = [r for r in other.detail(P.METRIC_ID) if r["event_type"] == P.EVENT_ROUND][0]
        self.assertIsNone(early_round["metadata"]["wait_from_push_s"])

    def test_the_last_window_is_open_without_a_merge_time_and_every_other_is_closed(self):
        t = three_round_transcript()
        self.record(self.write(t), merge_time=None)
        rounds = [r["metadata"] for r in self.rows(P.EVENT_ROUND)]
        self.assertEqual([m["window_open"] for m in rounds], [False, False, True])
        fetches = [r["metadata"] for r in self.rows(P.EVENT_FETCH)]
        self.assertEqual([m["window_open"] for m in fetches], [False, False, True])

    def test_a_merge_time_closes_every_window(self):
        self.record(self.write(three_round_transcript()), merge_time=iso("04:20:28"))
        self.assertFalse(any(r["metadata"]["window_open"] for r in self.rows(P.EVENT_ROUND) + self.rows(P.EVENT_FETCH)))

    def test_an_unusable_merge_time_leaves_the_last_window_open(self):
        # The reader ignores a merge time it cannot parse; the row must not claim it was closed.
        self.record(self.write(three_round_transcript()), merge_time="not a time")
        self.assertTrue(self.rows(P.EVENT_ROUND)[-1]["metadata"]["window_open"])

    def test_a_resumed_session_does_not_change_an_open_row_and_the_row_says_it_may_be_short(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 10, 100, 5))
        t.assistant("04:02:00", u=usage(2, 20, 200, 20))
        path = self.write(t)
        self.record(path, merge_time=None)
        t.assistant("04:30:00", u=usage(9, 90, 900, 90))  # the session is resumed and works on
        again = self.record(self.write(t), merge_time=None)
        self.assertEqual((again["written"], again["existing"]), (0, 2))
        row = self.rows(P.EVENT_ROUND)[0]["metadata"]
        self.assertEqual(row["output_tokens"], 25)       # the figure at the first recording ...
        self.assertTrue(row["window_open"])              # ... which is marked as possibly short
        # Recorded with the merge time instead, the final figure is what is stored.
        closed = metrics_lib.MetricsStore(Path(self.dir) / "closed.db")
        P.record_pr_rounds(self.write(t, "c.jsonl"), PR, store=closed, gh=FakeGh(), merge_time=iso("04:30:00"))
        final = closed.detail(P.METRIC_ID)
        rnd = [r for r in final if r["event_type"] == P.EVENT_ROUND][0]["metadata"]
        self.assertEqual((rnd["output_tokens"], rnd["window_open"]), (115, False))

    def test_a_fetch_before_any_round_is_recorded_with_no_round(self):
        t = Transcript()
        t.bash("04:00:00", f"gh api repos/o/r/issues/{PR}/comments --paginate", json.dumps([{"id": 1}]))
        t.review_call("04:05:00", params={"pr": PR})
        self.record(self.write(t), merge_time=iso("04:10:00"))
        fetch = self.rows(P.EVENT_FETCH)[0]["metadata"]
        self.assertIsNone(fetch["round"])
        self.assertIsNone(fetch["round_kind"])
        self.assertIsNone(fetch["review_id"])
        self.assertEqual(fetch["window_end"], iso("04:05:00"))

    def test_two_fetches_showing_one_review_are_two_events(self):
        t = Transcript()
        t.reviews_api("04:01:00", [rest_review(11, COPILOT, "03:27:21")])
        t.reviews_api("04:02:00", [rest_review(11, COPILOT, "03:27:21")])
        self.record(self.write(t), merge_time=iso("04:10:00"))
        self.assertEqual(len(self.rows(P.EVENT_FETCH)), 2)
        self.assertEqual(len(self.rows(P.EVENT_ROUND)), 1)
        self.assertEqual(self.rows(P.EVENT_ROUND)[0]["metadata"]["fetch_calls"], 2)

    def test_the_event_time_is_when_the_work_happened_so_a_backfill_buckets_by_it(self):
        self.record(self.write(three_round_transcript()))
        trend = self.store.trend(P.METRIC_ID, bucket="day")
        self.assertEqual([b["bucket"] for b in trend], ["2026-11-06"])
        self.assertEqual(trend[0]["event_count"], 6)
        self.assertEqual(epoch("04:01:15"), epoch("04:01:15"))


def reviewer_transcript(local_times=(), cwd=None):
    """Three Copilot rounds (04:01:15, 04:11:05, 04:19:59, the last one clean) with a `review`
    call at each time in `local_times`; `cwd` is stamped on every entry when given."""
    t = Transcript()
    t.assistant("03:59:00")
    for when in local_times:
        t.review_call(when, params={"pr": PR})
    t.reviews_api("04:01:15", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 100, 1000, 10))
    t.reviews_api("04:11:05", [rest_review(11, COPILOT, "03:27:21"), rest_review(13, COPILOT, "04:10:15")],
                  u=usage(3, 300, 3000, 30))
    t.reviews_view("04:19:59", [view_review("PRR_a", COPILOT, "03:27:21"), view_review("PRR_c", COPILOT, "04:10:15"),
                                view_review("PRR_d", COPILOT, "04:19:20", state="APPROVED")],
                   latest=[view_review("", COPILOT, "04:19:20")], u=usage(5, 500, 5000, 50))
    t.assistant("04:20:28", u=usage(6, 600, 6000, 60))
    if cwd is not None:
        for line in t.lines:
            line["cwd"] = cwd
    return t


class RepoAndReviewerFlag(_Base):
    def flags(self, local_times, **kw):
        # A fresh store per call: a second call over the same store would find the rows of the first.
        self.store = metrics_lib.MetricsStore(Path(self.dir) / f"flags-{len(os.listdir(self.dir))}.db")
        self.record(self.write(reviewer_transcript(local_times), f"flags-{len(os.listdir(self.dir))}.jsonl"), **kw)
        return [r["metadata"].get("local_reviewer_used") for r in self.rows(P.EVENT_ROUND) if r["metadata"]["kind"] == "copilot"]

    # --- repo and repo_source ---

    def test_a_repo_the_caller_names_is_the_repo_and_comes_from_a_remote(self):
        out = self.record(self.write(three_round_transcript()), repo="o/r")
        self.assertEqual((out["repo"], out["repo_source"]), ("o/r", "remote"))
        for row in self.rows(P.EVENT_ROUND) + self.rows(P.EVENT_FETCH):
            self.assertEqual((row["metadata"]["repo"], row["metadata"]["repo_source"]), ("o/r", "remote"))

    def test_without_a_repo_the_checkouts_remote_is_asked_once_and_is_a_remote(self):
        gh = FakeGh(remote="Acme/widgets")
        self.record(self.write(three_round_transcript()), gh=gh)
        self.assertEqual(len(gh.calls_to("repo view")), 1)
        for row in self.rows(P.EVENT_ROUND) + self.rows(P.EVENT_FETCH):
            self.assertEqual((row["metadata"]["repo"], row["metadata"]["repo_source"]), ("Acme/widgets", "remote"))

    def test_no_remote_falls_back_to_the_project_name_of_the_transcripts_cwd(self):
        t = reviewer_transcript(cwd="/Users/me/repos/my-project")
        out = self.record(self.write(t), gh=FakeGh(remote=None))
        self.assertEqual((out["repo"], out["repo_source"]), ("my-project", "project"))
        for row in self.rows(P.EVENT_ROUND) + self.rows(P.EVENT_FETCH):
            self.assertEqual((row["metadata"]["repo"], row["metadata"]["repo_source"]), ("my-project", "project"))
        # The failed lookup is the fallback's business, not a GitHub field left null.
        self.assertFalse([e for e in out["github_errors"] if "repo" in e.lower()])

    def test_a_remote_that_is_not_owner_slash_name_is_not_used(self):
        t = reviewer_transcript(cwd="/work/proj")
        self.record(self.write(t), gh=FakeGh(remote="not a repo; rm -rf"))
        self.assertEqual(self.rows(P.EVENT_ROUND)[0]["metadata"]["repo_source"], "project")

    def test_neither_a_remote_nor_a_cwd_is_null_never_a_guess(self):
        self.record(self.write(reviewer_transcript()), gh=FakeGh(remote=None))
        for row in self.rows(P.EVENT_ROUND) + self.rows(P.EVENT_FETCH):
            self.assertEqual((row["metadata"]["repo"], row["metadata"]["repo_source"]), (None, None))

    def test_the_first_cwd_wins_over_a_later_one(self):
        t = reviewer_transcript()
        t.lines[0]["cwd"] = "/a/first"
        t.lines[-1]["cwd"] = "/a/later"
        self.assertEqual(P.transcript_cwd(self.write(t)), "/a/first")
        self.assertIsNone(P.transcript_cwd(Path(self.dir) / "missing.jsonl"))

    def test_a_transcript_with_nothing_to_record_does_not_ask_for_the_repo(self):
        gh = FakeGh()
        t = Transcript()
        t.assistant("04:00:00")
        out = self.record(self.write(t), gh=gh)
        self.assertEqual(gh.calls_to("repo view"), [])
        self.assertEqual((out["repo"], out["repo_source"]), (None, None))

    # --- local_reviewer_used ---

    def test_no_review_call_makes_every_copilot_round_false(self):
        self.assertEqual(self.flags(()), [False, False, False])

    def test_a_review_between_two_copilot_rounds_flags_the_round_it_follows(self):
        # 04:05 follows round 1's feedback (04:01) and precedes round 2's (04:11).
        self.assertEqual(self.flags(("04:05:00",)), [True, False, False])

    def test_the_reviewer_used_for_the_first_rounds_and_then_not_is_true_then_false(self):
        self.assertEqual(self.flags(("04:05:00", "04:15:00")), [True, True, False])
        self.assertEqual(self.flags(("04:05:00",)), [True, False, False])

    def test_several_review_calls_in_one_round_are_still_one_true(self):
        self.assertEqual(self.flags(("04:05:00", "04:06:00", "04:07:00")), [True, False, False])

    def test_a_review_before_the_first_copilot_feedback_belongs_to_no_round(self):
        flags = self.flags(("04:00:30",))
        self.assertEqual(flags, [False, False, False])
        locals_ = [r["metadata"] for r in self.rows(P.EVENT_ROUND) if r["metadata"]["kind"] == "local"]
        self.assertEqual(len(locals_), 1, "it remains its own local row")

    def test_the_final_clean_round_is_false_and_a_review_after_it_is_what_would_flag_it(self):
        self.assertFalse(self.flags(())[-1])
        self.assertEqual(self.flags(("04:20:00",)), [False, False, True])

    def tie_flags(self, local_first, local_time="04:11:05"):
        """Round 1 at 04:01:15; a `review` call and round 2's feedback fetch issued in one
        response (the same timestamp), in either order in the transcript."""
        t = Transcript()
        t.assistant("03:59:00")
        t.reviews_api("04:01:15", [rest_review(11, COPILOT, "03:27:21")], u=usage(1, 100, 1000, 10))
        second = [rest_review(11, COPILOT, "03:27:21"), rest_review(13, COPILOT, "04:10:15")]
        if local_first:
            t.review_call(local_time, params={"pr": PR})
            t.reviews_api("04:11:05", second, u=usage(3, 300, 3000, 30))
        else:
            t.reviews_api("04:11:05", second, u=usage(3, 300, 3000, 30))
            t.review_call(local_time, params={"pr": PR})
        t.assistant("04:12:00", u=usage(4, 400, 4000, 40))
        # A fresh store and file per call: rows of an earlier call would be found as already recorded.
        n = len(os.listdir(self.dir))
        self.store = metrics_lib.MetricsStore(Path(self.dir) / f"tie-{n}.db")
        self.record(self.write(t, f"tie-{n}.jsonl"), merge_time=iso("04:20:28"))
        return [r["metadata"]["local_reviewer_used"] for r in self.rows(P.EVENT_ROUND) if r["metadata"]["kind"] == "copilot"]

    def test_a_review_call_tied_with_a_copilot_feedback_fetch_sets_no_flag_whatever_the_transcript_order(self):
        # Parallel tool calls share a timestamp, so their order in the transcript says nothing
        # about which came first. A tie is not strictly after one round's start and strictly
        # before the next one's, so it belongs to neither round, in either order.
        self.assertEqual(self.tie_flags(local_first=True), [False, False])
        self.assertEqual(self.tie_flags(local_first=False), [False, False])

    def test_a_tie_with_the_first_round_start_is_also_left_out_and_a_strict_neighbour_still_counts(self):
        self.assertEqual(self.tie_flags(local_first=True, local_time="04:01:15"), [False, False])
        self.assertEqual(self.tie_flags(local_first=True, local_time="04:11:04"), [True, False])

    def test_local_and_fetch_rows_do_not_carry_the_flag(self):
        self.record(self.write(reviewer_transcript(("04:05:00",))))
        for row in self.rows(P.EVENT_FETCH):
            self.assertNotIn("local_reviewer_used", row["metadata"])
        local = [r["metadata"] for r in self.rows(P.EVENT_ROUND) if r["metadata"]["kind"] == "local"][0]
        self.assertNotIn("local_reviewer_used", local)
        self.assertEqual(local["repo"], "Donelle/claude-runway")

    def test_the_flag_is_a_real_boolean(self):
        self.record(self.write(reviewer_transcript(("04:05:00",))))
        copilot = [r["metadata"] for r in self.rows(P.EVENT_ROUND) if r["metadata"]["kind"] == "copilot"]
        self.assertIs(copilot[0]["local_reviewer_used"], True)
        self.assertIs(copilot[1]["local_reviewer_used"], False)

    # --- keys and schema ---

    def test_the_new_fields_are_not_in_any_idempotency_key(self):
        gh = FakeGh()
        found = P.pr_round_cost_lib.find_rounds(self.write(reviewer_transcript(("04:05:00",))), PR,
                                                merge_time=iso("04:20:28"))
        events = P.build_events(found, P.GitHubFacts(PR, gh), True, {"repo": "o/r", "repo_source": "remote"})
        for ev in events:
            for field in ("repo", "repo_source", "local_reviewer_used"):
                self.assertNotIn(field, ev["key"])
        round_keys = {tuple(ev["key"]) for ev in events if ev["event_type"] == P.EVENT_ROUND}
        self.assertEqual(round_keys, {("pr", "kind", "review_id", "call_time"),
                                      ("pr", "kind", "review_id", "call_time", "kind_number", "session_id")})
        fetch_keys = {tuple(ev["key"]) for ev in events if ev["event_type"] == P.EVENT_FETCH}
        self.assertEqual(fetch_keys, {("pr", "review_id", "call_time", "tool_use_id")})

    def test_a_rerun_that_resolves_the_repo_differently_still_writes_nothing(self):
        path = self.write(reviewer_transcript(("04:05:00",), cwd="/w/proj"))
        first = self.record(path, gh=FakeGh(remote=None))
        again = self.record(path, gh=FakeGh(remote="o/r"))
        self.assertEqual((again["written"], again["existing"]), (0, first["written"]))
        # The first row stays as written: the store is append-only.
        self.assertEqual(self.rows(P.EVENT_ROUND)[0]["metadata"]["repo"], "proj")

    def test_the_schema_is_two_because_the_fields_are_additive(self):
        self.assertEqual(P.SCHEMA_VERSION, 2)
        self.record(self.write(reviewer_transcript()))
        self.assertEqual({r["metadata"]["schema"] for r in self.rows(P.EVENT_ROUND) + self.rows(P.EVENT_FETCH)}, {2})

    def test_the_cli_prints_the_repo(self):
        out = StringIO()
        with redirect_stdout(out):
            CLI.main(["--transcript", str(self.write(reviewer_transcript())), "--pr", str(PR), "--db",
                      str(Path(self.dir) / "cli.db"), "--dry-run", "--repo", "o/r"])
        self.assertIn("repo: o/r (remote)", out.getvalue())


class Cli(_Base):
    def run_cli(self, *argv):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = CLI.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def fake_run(self):
        gh = FakeGh()
        return lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, gh(cmd[1:]), "")

    def test_records_and_reports_and_a_rerun_says_already_recorded(self):
        path = self.write(three_round_transcript())
        with mock.patch.object(P.subprocess, "run", self.fake_run()):
            code, out, _ = self.run_cli("--transcript", str(path), "--pr", str(PR), "--db", str(self.db),
                                        "--merge-time", iso("04:20:28"))
            code2, out2, _ = self.run_cli("--transcript", str(path), "--pr", str(PR), "--db", str(self.db),
                                          "--merge-time", iso("04:20:28"))
        self.assertEqual(code, 0)
        self.assertIn("written 6, already recorded 0", out)
        self.assertIn("3 fetch call(s) found, 3 with an event", out)
        self.assertIn("written 0, already recorded 6", out2)

    def test_dry_run_json_lists_events_and_writes_nothing(self):
        path = self.write(three_round_transcript())
        with mock.patch.object(P.subprocess, "run", self.fake_run()):
            code, out, _ = self.run_cli("--transcript", str(path), "--pr", str(PR), "--db", str(self.db),
                                        "--merge-time", iso("04:20:28"), "--dry-run", "--format", "json")
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(out)["events"]), 6)
        self.assertFalse(self.db.exists())

    def test_gh_missing_still_exits_zero_and_says_which_field_is_null(self):
        path = self.write(three_round_transcript())
        with mock.patch.object(P.subprocess, "run", side_effect=FileNotFoundError("gh")):
            code, out, _ = self.run_cli("--transcript", str(path), "--pr", str(PR), "--db", str(self.db),
                                        "--merge-time", iso("04:20:28"))
        self.assertEqual(code, 0)
        self.assertIn("GitHub field left null", out)
        self.assertIn("written 6", out)

    def test_exit_codes_for_a_bad_argument_and_an_unreadable_transcript(self):
        self.assertEqual(self.run_cli("--transcript", "x", "--pr", "0")[0], 2)
        self.assertEqual(self.run_cli("--transcript", "x", "--pr", "5", "--repo", "bad repo")[0], 2)
        code, out, _ = self.run_cli("--transcript", str(Path(self.dir) / "none.jsonl"), "--pr", "5",
                                    "--db", str(self.db))
        self.assertEqual(code, 1)
        self.assertIn("could not be read", out)

    def test_the_default_database_follows_the_env_override(self):
        path = self.write(three_round_transcript())
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": str(self.db)}), \
                mock.patch.object(P.subprocess, "run", self.fake_run()):
            self.run_cli("--transcript", str(path), "--pr", str(PR), "--merge-time", iso("04:20:28"))
        self.assertEqual(self.store.summary(P.METRIC_ID)["event_count"], 6)


if __name__ == "__main__":
    unittest.main()
