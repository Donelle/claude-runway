#!/usr/bin/env python3
"""Tests for choosing which Copilot reviews to record: `history_lib.choose_balanced` and harness/collect_balanced_history.py.

Stdlib-only (unittest), no network: `gh` is a stub and every output folder is a temporary one.

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
for _directory in (REPO_ROOT / "harness", REPO_ROOT / "libs"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import collect_balanced_history as balanced  # noqa: E402
import history_lib  # noqa: E402


def prof(number, kind="python", size="medium", rounds="some rounds", findings="some findings", n_rounds=3, inline=3):
    return {"pr": number, "title": f"pr {number}", "rounds": n_rounds, "inline": inline, "traits": (kind, size, rounds, findings)}


class Traits(unittest.TestCase):
    def test_a_change_of_prose_only_is_docs(self):
        self.assertEqual(history_lib.kind_of(["README.md", "docs/a.md"]), "docs")

    def test_a_shell_script_makes_it_shell_even_beside_python(self):
        self.assertEqual(history_lib.kind_of(["bootstrap.sh", "libs/a.py"]), "shell")

    def test_python_beside_prose_is_python_and_anything_else_is_other(self):
        self.assertEqual(history_lib.kind_of(["libs/a.py", "README.md"]), "python")
        self.assertEqual(history_lib.kind_of(["config.json"]), "other")
        self.assertEqual(history_lib.kind_of([]), "other")

    def test_the_buckets_have_their_edges(self):
        self.assertEqual([history_lib.size_of(n) for n in (0, 149, 150, 999, 1000)], ["small", "small", "medium", "medium", "large"])
        self.assertEqual([history_lib.rounds_of(n) for n in (1, 2, 3, 5, 6)],
                         ["few rounds", "few rounds", "some rounds", "some rounds", "many rounds"])
        self.assertEqual([history_lib.findings_of(n) for n in (1, 2, 3, 5, 6)],
                         ["few findings", "few findings", "some findings", "some findings", "many findings"])

    def test_a_profile_reads_a_gh_pr_list_item(self):
        pr = {"number": 7, "title": "t", "additions": 100, "deletions": 60, "files": [{"path": "a.md"}]}
        self.assertEqual(history_lib.profile(pr, 4, 2)["traits"], ("docs", "medium", "some rounds", "few findings"))


class ChooseBalanced(unittest.TestCase):
    def test_every_value_of_every_trait_is_covered_before_one_is_repeated(self):
        profiles = [
            prof(1, "python", "medium", "many rounds", "many findings", 9, 9),
            prof(2, "python", "medium", "many rounds", "many findings", 9, 8),   # the same again
            prof(3, "docs", "small", "few rounds", "few findings", 1, 1),
            prof(4, "shell", "large", "some rounds", "some findings", 3, 3),
        ]
        self.assertEqual(set(history_lib.choose_balanced(profiles, 3)), {1, 3, 4})

    def test_the_result_does_not_depend_on_the_order_the_prs_arrive_in(self):
        profiles = [prof(n, kind, size) for n, (kind, size) in enumerate(
            [("python", "small"), ("docs", "medium"), ("shell", "large"), ("python", "large"), ("other", "small")], start=1)]
        self.assertEqual(history_lib.choose_balanced(profiles, 4), history_lib.choose_balanced(list(reversed(profiles)), 4))

    def test_a_tie_goes_to_the_pr_with_more_findings_then_the_lower_number(self):
        self.assertEqual(history_lib.choose_balanced([prof(5, inline=3), prof(6, inline=4)], 1), [6])
        self.assertEqual(history_lib.choose_balanced([prof(6), prof(5)], 1), [5])

    def test_a_pr_with_nothing_to_score_is_never_chosen(self):
        profiles = [prof(1, n_rounds=0), prof(2, inline=0), prof(3)]
        self.assertEqual(history_lib.choose_balanced(profiles, 5), [3])

    def test_an_excluded_pr_is_left_out_and_fewer_than_count_is_fine(self):
        self.assertEqual(history_lib.choose_balanced([prof(1), prof(2, "docs")], 5, exclude={1}), [2])

    def test_after_every_value_is_covered_the_rest_are_new_combinations_first(self):
        profiles = [prof(1, "python", "medium"), prof(2, "python", "medium", inline=9), prof(3, "python", "small", inline=1)]
        # 2 first (most findings), then 3 adds a value, then 1: the same combination as 2, so last whatever its findings
        self.assertEqual(history_lib.choose_balanced(profiles, 3), [2, 3, 1])


def stub_gh(prs, reviews, comments):
    def gh_json(*args):
        if args[0] == "pr":
            return prs
        path = args[1]
        number = int(path.split("/")[4])
        return [reviews.get(number, [])] if path.endswith("/reviews") else [comments.get(number, [])]
    return gh_json


def review(rid, login="copilot-pull-request-reviewer[bot]"):
    return {"id": rid, "user": {"login": login}}


def comment(rid, path="a.py", reply=None):
    return {"pull_request_review_id": rid, "path": path, "in_reply_to_id": reply}


class CollectBalanced(unittest.TestCase):
    PRS = [
        {"number": 10, "title": "docs", "additions": 5, "deletions": 1, "files": [{"path": "README.md"}]},
        {"number": 11, "title": "code", "additions": 900, "deletions": 100, "files": [{"path": "libs/a.py"}]},
        {"number": 12, "title": "no copilot", "additions": 1, "deletions": 1, "files": [{"path": "a.py"}]},
    ]
    REVIEWS = {10: [review(1), review(2, "someone")], 11: [review(3), review(4), review(5)], 12: [review(6, "someone")]}
    COMMENTS = {10: [comment(1), comment(1, reply=99), comment(2)], 11: [comment(3), comment(4), comment(4), comment(5, path="")]}

    def profiles(self):
        with mock.patch.object(balanced.collector, "gh_json", stub_gh(self.PRS, self.REVIEWS, self.COMMENTS)):
            return balanced.profiles_of(150)

    def test_a_pr_is_profiled_by_its_copilot_reviews_and_its_inline_findings_only(self):
        by_number = {p["pr"]: p for p in self.profiles()}
        self.assertEqual(sorted(by_number), [10, 11])  # 12 had no Copilot review
        self.assertEqual((by_number[10]["rounds"], by_number[10]["inline"]), (1, 1))  # a reply and another reviewer's comment do not count
        self.assertEqual((by_number[11]["rounds"], by_number[11]["inline"]), (3, 3))  # a comment with no path is not a finding
        self.assertEqual(by_number[10]["traits"][0], "docs")

    def run_main(self, *argv, retargeted=()):
        profiles = self.profiles()
        written = []

        def collect(number):
            return {"pr": number, "rounds": [{}], "base_changes": 1 if number in retargeted else 0}

        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as out, mock.patch.object(balanced, "profiles_of", return_value=profiles), \
                mock.patch.object(balanced.collector, "collect", side_effect=collect), \
                mock.patch.object(balanced.collector, "fetch_reviewed_commits", return_value=[]), \
                contextlib.redirect_stdout(stdout):
            status = balanced.main([*argv, "--out", out])
            written = sorted(p.name for p in Path(out).iterdir())
        return status, written, stdout.getvalue()

    def test_dry_run_shows_the_pick_and_writes_nothing(self):
        status, written, text = self.run_main("--dry-run")
        self.assertEqual((status, written), (0, []))
        self.assertIn("#11: python / large", text)

    def test_it_records_each_pick_as_a_snapshot(self):
        status, written, _ = self.run_main("--count", "2")
        self.assertEqual((status, written), (0, ["pr-10.json", "pr-11.json"]))

    def test_a_retargeted_pr_is_dropped_and_not_written(self):
        status, written, text = self.run_main("--count", "2", retargeted={11})
        self.assertEqual((status, written), (0, ["pr-10.json"]))
        self.assertIn("PR #11: retargeted 1 time(s)", text)

    def test_an_excluded_pr_is_never_picked(self):
        _, written, _ = self.run_main("--count", "2", "--exclude", "11")
        self.assertEqual(written, ["pr-10.json"])

    def test_nothing_to_score_is_reported(self):
        with mock.patch.object(balanced, "profiles_of", return_value=[]), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(balanced.main(["--dry-run"]), 1)

    def test_the_snapshot_written_is_the_collected_one(self):
        with tempfile.TemporaryDirectory() as out, mock.patch.object(balanced, "profiles_of", return_value=self.profiles()), \
                mock.patch.object(balanced.collector, "collect", return_value={"pr": 11, "rounds": [{}], "base_changes": 0, "marker": "x"}), \
                mock.patch.object(balanced.collector, "fetch_reviewed_commits", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            balanced.main(["--count", "1", "--out", out])
            (path,) = Path(out).iterdir()
            self.assertEqual(json.loads(path.read_text())["marker"], "x")


if __name__ == "__main__":
    unittest.main()
