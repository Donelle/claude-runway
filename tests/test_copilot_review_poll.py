"""Tests for libs/copilot_review_poll_lib.py and tools/poll_copilot_review.py.
The `gh` call is always a fake: no network, no real sleeping."""

import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "libs"))
sys.path.insert(0, str(REPO / "tools"))

import copilot_review_poll_lib as P  # noqa: E402
import poll_copilot_review as CLI  # noqa: E402

SHA = "a" * 40
OTHER = "b" * 40


def review(rid, sha=SHA, login=P.COPILOT_LOGIN):
    return {"id": rid, "commit_id": sha, "user": {"login": login}}


def run_cli(args, fetch=None):
    out = io.StringIO()
    sleeps = []
    with contextlib.redirect_stdout(out):
        code = CLI.main(args, fetch=fetch, sleep=sleeps.append)
    return code, out.getvalue().strip(), sleeps


class Matching(unittest.TestCase):
    def test_only_copilot_review_of_exact_head_that_is_unseen(self):
        reviews = [review(1), review(2, sha=OTHER), review(3, login="someone"), review(4, login="me"), review(5)]
        self.assertEqual(P.new_review_ids(reviews, SHA, {"1"}, "me"), ["5"])

    def test_seen_ids_accept_prefixed_and_comma_forms(self):
        self.assertEqual(P.normalize_seen(["review:1,review:2", "3 comment:9"]), {"1", "2", "3", "comment:9"})

    def test_parse_paginated_concatenated_pages(self):
        text = json.dumps([review(1)]) + "\n" + json.dumps([review(2)]) + "\n"
        self.assertEqual([r["id"] for r in P.parse_paginated(text)], [1, 2])

    def test_stalled_or_missing_gh_is_a_fetch_error_not_a_crash(self):
        def stalled(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))

        def missing(cmd, **kw):
            raise FileNotFoundError("gh")

        for run in (stalled, missing):
            with self.assertRaises(P.FetchError):
                P.gh_fetch_reviews("o/r", 7, run=run)
        ids, errors, looks = P.poll(lambda: P.gh_fetch_reviews("o/r", 7, run=stalled), SHA, [], None,
                                    iterations=2, interval_s=0, sleep=lambda s: None)
        self.assertEqual((ids, errors, looks), ([], 2, 2))

    def test_gh_fetch_uses_paginate_and_reports_failure(self):
        calls = []

        def ok(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([review(1)]), stderr="")

        self.assertEqual(P.gh_fetch_reviews("o/r", 7, run=ok)[0]["id"], 1)
        self.assertIn("--paginate", calls[0])
        self.assertIn("repos/o/r/pulls/7/reviews", calls[0])

        def bad(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

        with self.assertRaises(P.FetchError):
            P.gh_fetch_reviews("o/r", 7, run=bad)


class Poll(unittest.TestCase):
    def test_arrives_on_a_later_iteration(self):
        seq = iter([[], [review(1, sha=OTHER)], [review(9)]])
        sleeps = []
        ids, errors, looks = P.poll(lambda: next(seq), SHA, [], "me", iterations=5, interval_s=15, sleep=sleeps.append)
        self.assertEqual((ids, errors, looks), (["9"], 0, 3))
        self.assertEqual(sleeps, [15, 15])

    def test_not_yet_is_bounded_by_iterations(self):
        sleeps = []
        ids, _, looks = P.poll(lambda: [], SHA, [], None, iterations=12, interval_s=15, sleep=sleeps.append)
        self.assertEqual((ids, looks), ([], 12))
        self.assertEqual(len(sleeps), 12)  # a full NOT_YET call covers iterations x interval

    def test_fetch_errors_are_retried_and_counted(self):
        state = {"n": 0}

        def fetch():
            state["n"] += 1
            if state["n"] < 3:
                raise P.FetchError("blip")
            return [review(4)]

        ids, errors, looks = P.poll(fetch, SHA, [], None, iterations=5, interval_s=0, sleep=lambda s: None)
        self.assertEqual((ids, errors, looks), (["4"], 2, 3))

    def test_deadline_counts_fetch_time_not_just_sleeps(self):
        # Each fetch "takes" 60 s on a fake clock; 12 iterations must not all run.
        now = {"t": 0.0}

        def fetch():
            now["t"] += 60
            raise P.FetchError("slow")

        ids, errors, looks = P.poll(fetch, SHA, [], None, iterations=12, interval_s=15,
                                    sleep=lambda s: now.__setitem__("t", now["t"] + s), clock=lambda: now["t"])
        self.assertEqual(ids, [])
        self.assertLess(looks, 12)
        self.assertLessEqual(now["t"], P.MAX_CALL_SECONDS)
        self.assertEqual(errors, looks)


class Cli(unittest.TestCase):
    base = ["--pr", "5", "--head-sha", SHA, "--me", "me"]

    def test_prints_review_ids(self):
        code, out, _ = run_cli(self.base + ["--seen", "review:1"], fetch=lambda: [review(1), review(2)])
        self.assertEqual((code, out), (0, "review:2"))

    def test_prints_not_yet_without_sleeping_past_the_bound(self):
        code, out, sleeps = run_cli(self.base + ["--iterations", "4", "--interval", "15"], fetch=lambda: [])
        self.assertEqual((code, out), (0, P.NOT_YET))
        self.assertEqual(len(sleeps), 4)

    def test_short_sha_is_rejected_up_front(self):
        code, out, _ = run_cli(["--pr", "5", "--head-sha", "abc1234"], fetch=lambda: [])
        self.assertEqual(code, 2)
        self.assertIn("40-character", out)

    def test_one_call_cannot_wait_past_the_cache_lifetime(self):
        self.assertLessEqual(P.DEFAULT_ITERATIONS * P.DEFAULT_INTERVAL_S, P.MAX_CALL_SECONDS)
        self.assertLess(P.MAX_CALL_SECONDS, 300)
        code, out, _ = run_cli(self.base + ["--iterations", "36", "--interval", "15"], fetch=lambda: [])
        self.assertEqual(code, 2)
        self.assertIn("several calls", out)

    def test_all_fetches_failing_is_an_error_not_not_yet(self):
        def boom():
            raise P.FetchError("down")

        code, out, _ = run_cli(self.base + ["--iterations", "3", "--interval", "1"], fetch=boom)
        self.assertEqual(code, 1)
        self.assertTrue(out.startswith("ERROR"))

    def test_total_window_is_about_nine_minutes(self):
        self.assertEqual(P.MAX_CALLS * P.DEFAULT_ITERATIONS * P.DEFAULT_INTERVAL_S, 540)


if __name__ == "__main__":
    unittest.main()
