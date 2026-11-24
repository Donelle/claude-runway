#!/usr/bin/env python3
"""Tests for harness/pi/score_run.py: scoring a pi_driver.py summary against the seeded-three case.

No Docker, no LM Studio, no network: the review text is canned in the format `format_review` prints, and the scoring is
the harness's own `score()` on the real case. Stdlib-only (unittest, no pytest).

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _directory in (REPO_ROOT / "harness" / "pi", REPO_ROOT / "libs", REPO_ROOT / "harness"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import score_run as s  # noqa: E402

# seeded-three's head file: the shell=True call is line 6, the bad slice line 11, the division line 15.
ALL_THREE = """Reasoning review: 3 finding(s) in 3 class(es) from 1 reviewer(s) (general), up to 1 diff chunk(s) each.
These findings are not verified.

## shell-injection (1)
- backup.py:6 [general] name reaches a shell
    failing input: name = 'x; rm -rf ~'

## off-by-one (1)
- backup.py:10-12 [general] the slice drops an item

## division-by-zero (1)
- backup.py:15 [general] empty list divides by zero
"""

TWO_OF_THREE = """Reasoning review: 2 finding(s) in 2 class(es) from 1 reviewer(s) (general), up to 1 diff chunk(s) each.
These findings are not verified.

## shell-injection (1)
- backup.py:6-8 [general] shell injection

## off-by-one (1)
- backup.py:10-12 [general] off by one
"""


class ParseTests(unittest.TestCase):
    def test_finding_lines_are_read_with_their_class(self):
        findings = s.parse_review_text(ALL_THREE)
        self.assertEqual([(f.file, f.line_start, f.line_end, f.cls) for f in findings],
                         [("backup.py", 6, 6, "shell-injection"), ("backup.py", 10, 12, "off-by-one"),
                          ("backup.py", 15, 15, "division-by-zero")])

    def test_an_alternate_is_read_as_its_own_finding_in_the_same_file(self):
        text = "## off-by-one (1)\n- backup.py:10 [general] first\n    also, at 11-12 [general]: second\n"
        findings = s.parse_review_text(text)
        self.assertEqual([(f.line_start, f.line_end, f.claim) for f in findings], [(10, 10, "first"), (11, 12, "second")])
        self.assertEqual({f.file for f in findings}, {"backup.py"})

    def test_a_finding_with_no_line_gets_line_zero_and_matches_nothing(self):
        findings = s.parse_review_text("## other (1)\n- backup.py [general] vague\n")
        self.assertEqual(findings[0].line_start, 0)
        score = s.score_summary({"review_text": "## other (1)\n- backup.py [general] vague\n"})
        self.assertEqual(score.found, [])
        self.assertEqual(len(score.other), 1)

    def test_prose_and_blank_lines_are_not_findings(self):
        self.assertEqual(s.parse_review_text("Reasoning review: 0 finding(s)\n\nProblems (1):\n- something broke\n"), [])


class ScoreTests(unittest.TestCase):
    def test_all_three_found(self):
        score = s.score_summary({"review_text": ALL_THREE, "final_text": "backup.py:6 and backup.py:15"})
        self.assertTrue(score.readable)
        self.assertEqual(sorted(score.found), ["division-by-zero", "off-by-one", "shell-injection"])
        self.assertEqual(score.missed, [])
        self.assertEqual(score.findings, 3)
        self.assertEqual(score.final_repeats, 2)  # the caller repeated two of the three places

    def test_two_of_three_names_the_missed_bug(self):
        score = s.score_summary({"review_text": TWO_OF_THREE})
        self.assertEqual(score.missed, ["division-by-zero"])
        self.assertEqual(len(score.found), 2)

    def test_empty_text_is_unreadable_and_misses_everything(self):
        score = s.score_summary({"review_text": ""})
        self.assertFalse(score.readable)
        self.assertEqual(len(score.missed), 3)

    def test_a_refusal_is_not_scored_as_a_review(self):
        score = s.score_summary({"review_text": "Error: the reviewers file is invalid"})
        self.assertFalse(score.readable)
        self.assertIn("refused", score.note)

    def test_an_incomplete_review_is_flagged_even_when_it_has_findings(self):
        text = "INCOMPLETE REVIEW: 1 problem(s)\n" + TWO_OF_THREE + "\nProblems (1):\n- reviewer 'general': unreadable reply\n"
        score = s.score_summary({"review_text": text})
        self.assertFalse(score.readable)
        self.assertEqual(len(score.found), 2)
        self.assertIn("INCOMPLETE REVIEW", score.note)

    def test_unknown_case_raises(self):
        with self.assertRaises(KeyError):
            s.score_summary({"review_text": ALL_THREE}, "no-such-case")


class MainTests(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = s.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def write(self, directory: str, payload) -> str:
        path = Path(directory) / "run.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_one_line_row_carries_the_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, {"review_text": TWO_OF_THREE, "final_text": "", "stopped_by": "completed", "turns": 2,
                                    "tool_calls": 1, "elapsed_seconds": 9.5})
            code, out, _err = self.run_main(path)
        self.assertEqual(code, 0)
        for part in ("stopped_by=completed", "readable=yes", "found=2/3", "findings=2", "turns=2", "seconds=9.5", "caller_repeated=0/2"):
            self.assertIn(part, out)

    def test_json_flag_prints_the_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _err = self.run_main(self.write(tmp, {"review_text": ALL_THREE}), "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["missed"], [])

    def test_a_file_that_is_not_json_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _out, err = self.run_main(self.write(tmp, "not json"))
        self.assertEqual(code, 2)
        self.assertIn("not a JSON object", err)

    def test_a_missing_file_exits_2(self):
        code, _out, err = self.run_main("/no/such/run.json")
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)

    def test_an_unknown_case_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _out, err = self.run_main(self.write(tmp, {"review_text": ALL_THREE}), "--case", "nope")
        self.assertEqual(code, 2)
        self.assertIn("unknown case", err)


if __name__ == "__main__":
    unittest.main()
