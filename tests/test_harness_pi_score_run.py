#!/usr/bin/env python3
"""Tests for harness/pi/score_run.py: scoring a pi_driver.py summary against the answer key's entry for the change it names.

No Docker, no LM Studio, no network: the review text is canned in the format `format_review` prints, and the scoring is
`live_scoring_lib.score_tag` against the real answer key (`harness/live-tests/answer_key.json`). The real change used throughout
is `change-6` (one bug, B7 at `libs/version_lib.py:35`, and one decoy, D4 at `:30`) and, for the clean control, `change-5`.
Stdlib-only (unittest, no pytest).

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
PI_DIR = REPO_ROOT / "harness" / "pi"
for _directory in (PI_DIR, REPO_ROOT / "libs", REPO_ROOT / "harness"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import live_scoring_lib  # noqa: E402
import score_run as s  # noqa: E402

BUG_ONLY = """Reasoning review: 1 finding(s) in 1 class(es) from 1 reviewer(s) (general), up to 1 diff chunk(s) each.
These findings are not verified.

## logic (1)
- libs/version_lib.py:34-35 [general] the success test is inverted
    failing input: git describe exits 0 and prints a tag
"""

# B7 at line 35, the decoy D4 at line 30, and a remark at line 12 that sits at neither.
BUG_DECOY_AND_A_REMARK = """Reasoning review: 3 finding(s) in 2 class(es) from 1 reviewer(s) (general), up to 1 diff chunk(s) each.
These findings are not verified.

## logic (2)
- libs/version_lib.py:35 [general] the success test is inverted
- libs/version_lib.py:30 [general] the timeout was doubled

## style (1)
- libs/version_lib.py:12 [general] an unrelated remark
"""

NOTHING = "Reasoning review: 0 finding(s) in 0 class(es) from 1 reviewer(s) (general), up to 1 diff chunk(s) each.\nThese findings are not verified.\n"

# The clean control changes no behaviour, so each of these is a false alarm.
ON_THE_CLEAN_CHANGE = """Reasoning review: 2 finding(s) in 1 class(es) from 1 reviewer(s) (general), up to 1 diff chunk(s) each.
These findings are not verified.

## logic (2)
- libs/qdrant_retry.py:40 [general] the exception is no longer caught
- libs/cache_db.py:12 [general] a comment changed
"""


def summary(text: str, change: str = "change-6", **more) -> dict:
    return {"change": change, "review_text": text, **more}


class ParseTests(unittest.TestCase):
    def test_finding_lines_are_read_with_their_class(self):
        findings = s.parse_review_text(BUG_DECOY_AND_A_REMARK)
        self.assertEqual([(f.file, f.line_start, f.line_end, f.cls) for f in findings],
                         [("libs/version_lib.py", 35, 35, "logic"), ("libs/version_lib.py", 30, 30, "logic"),
                          ("libs/version_lib.py", 12, 12, "style")])

    def test_a_range_is_read_with_both_ends(self):
        (found,) = s.parse_review_text(BUG_ONLY)
        self.assertEqual((found.line_start, found.line_end), (34, 35))

    def test_an_alternate_is_read_as_its_own_finding_in_the_same_file(self):
        text = "## off-by-one (1)\n- backup.py:10 [general] first\n    also, at 11-12 [general]: second\n"
        findings = s.parse_review_text(text)
        self.assertEqual([(f.line_start, f.line_end, f.claim) for f in findings], [(10, 10, "first"), (11, 12, "second")])
        self.assertEqual({f.file for f in findings}, {"backup.py"})

    def test_a_finding_with_no_line_gets_line_zero_and_matches_nothing(self):
        text = "Reasoning review: 1 finding(s)\n\n## other (1)\n- libs/version_lib.py [general] vague\n"
        findings = s.parse_review_text(text)
        self.assertEqual(findings[0].line_start, 0)
        score = s.score_summary(summary(text))
        self.assertEqual((score.found, score.on_decoys, len(score.unscored)), ([], [], 1))

    def test_prose_and_blank_lines_are_not_findings(self):
        self.assertEqual(s.parse_review_text("Reasoning review: 0 finding(s)\n\nProblems (1):\n- something broke\n"), [])


class ScoreTests(unittest.TestCase):
    def test_the_bug_is_found_and_the_callers_repeat_of_its_place_is_counted(self):
        score = s.score_summary(summary(BUG_ONLY, final_text="the bug is at libs/version_lib.py:34"))
        self.assertTrue(score.readable)
        self.assertEqual((score.change, score.found, score.missed), ("change-6", ["B7"], []))
        self.assertEqual((score.on_decoys, score.clean_false_alarms, score.unscored), ([], [], []))
        self.assertEqual((score.findings, score.final_repeats), (1, 1))

    def test_a_repeat_names_the_findings_start_line_and_no_other(self):
        # The review's finding is 34-35: the start line alone is enough, and the end line is not another place.
        self.assertEqual(s.score_summary(summary(BUG_ONLY, final_text="libs/version_lib.py:35")).final_repeats, 0)
        self.assertEqual(s.score_summary(summary(BUG_ONLY, final_text="libs/version_lib.py:34 and again libs/version_lib.py:34")).final_repeats, 1)

    def test_a_finding_at_a_decoy_and_one_at_neither_are_told_apart_from_the_bug(self):
        score = s.score_summary(summary(BUG_DECOY_AND_A_REMARK, final_text="libs/version_lib.py:35 and libs/version_lib.py:12"))
        self.assertEqual((score.found, score.missed, score.on_decoys), (["B7"], [], ["D4"]))
        self.assertEqual(len(score.unscored), 1)
        self.assertIn("libs/version_lib.py:12-12", score.unscored[0])
        self.assertEqual((score.findings, score.final_repeats), (3, 2))

    def test_a_review_that_names_nothing_misses_the_bug_and_is_still_readable(self):
        score = s.score_summary(summary(NOTHING))
        self.assertTrue(score.readable)
        self.assertEqual((score.found, score.missed, score.findings), ([], ["B7"], 0))

    def test_every_finding_on_the_clean_change_is_a_false_alarm_and_none_is_unscored(self):
        score = s.score_summary(summary(ON_THE_CLEAN_CHANGE, change="change-5"))
        self.assertEqual((score.change, len(score.clean_false_alarms), score.unscored, score.on_decoys), ("change-5", 2, [], []))
        self.assertEqual((score.found, score.missed), ([], []))  # it has no bug to find

    def test_the_answers_are_the_named_changes_not_a_default(self):
        # The same review text is a hit on change-6 and an unscored remark on change-1, whose bugs are in another file.
        self.assertEqual(s.score_summary(summary(BUG_ONLY, change="change-6")).found, ["B7"])
        other = s.score_summary(summary(BUG_ONLY, change="change-1"))
        self.assertEqual((other.found, other.missed, len(other.unscored)), ([], ["B1", "B2"], 1))

    def test_a_key_that_is_passed_in_is_used_instead_of_the_file(self):
        tags = (live_scoring_lib.LiveTag("change-9", (live_scoring_lib.LiveBug("X1", "libs/version_lib.py", 35, "logic", "c", "diff"),)),)
        score = s.score_summary(summary(BUG_ONLY, change="change-9"), tags)
        self.assertEqual(score.found, ["X1"])

    def test_empty_text_is_unreadable_and_misses_every_bug_of_the_change(self):
        score = s.score_summary(summary(""))
        self.assertFalse(score.readable)
        self.assertEqual(score.missed, ["B7"])
        self.assertEqual(s.score_summary(summary("", change="change-1")).missed, ["B1", "B2"])
        self.assertEqual(s.score_summary(summary("", change="change-5")).missed, [])

    def test_text_that_is_not_text_reads_as_no_review_not_a_crash(self):
        for review_text in (None, 5, ["a"]):
            with self.subTest(review_text=review_text):
                score = s.score_summary({"change": "change-6", "review_text": review_text, "final_text": 7})
                self.assertFalse(score.readable)
                self.assertIn("no review text", score.note)

    def test_a_caller_error_or_a_timeout_is_not_a_review(self):
        # What Pi puts in the tool result when the caller's arguments were wrong, and when the reviewer was too slow.
        for text in ('Validation failed for tool "mcp__reviewer__review":\n  - type: missing', "MCP request timed out after 60000ms"):
            with self.subTest(text=text[:30]):
                score = s.score_summary(summary(text))
                self.assertFalse(score.readable)
                self.assertEqual(score.missed, ["B7"])
                self.assertIn("no review ran", score.note)

    def test_a_refusal_is_not_scored_as_a_review(self):
        score = s.score_summary(summary("Error: the reviewers file is invalid"))
        self.assertFalse(score.readable)
        self.assertIn("refused", score.note)

    def test_an_incomplete_review_is_flagged_even_when_it_has_findings(self):
        text = "INCOMPLETE REVIEW: 1 problem(s)\n" + BUG_ONLY + "\nProblems (1):\n- reviewer 'general': unreadable reply\n"
        score = s.score_summary(summary(text))
        self.assertFalse(score.readable)
        self.assertEqual(score.found, ["B7"])
        self.assertIn("INCOMPLETE REVIEW", score.note)

    def test_a_run_whose_review_call_was_the_wrong_one_is_scored_but_not_readable(self):
        # With the refs swapped, say, the text still looks like a review, so only the driver's list tells the two apart.
        score = s.score_summary(summary(BUG_ONLY, call_errors=["a review call that ran was not this session's: base_ref"], failures=[]))
        self.assertEqual((score.readable, score.found, score.findings), (False, ["B7"], 1))
        self.assertEqual(score.note, "FAILED SESSION: a review call that ran was not this session's: base_ref")

    def test_a_run_with_failures_is_not_readable_and_says_how_many_more_reasons_there_are(self):
        score = s.score_summary(summary(BUG_ONLY, call_errors=["wrong call"], failures=["no results file", "container left"]))
        self.assertFalse(score.readable)
        self.assertEqual(score.note, "FAILED SESSION: wrong call (+2 more)")
        self.assertEqual(s.score_summary(summary(NOTHING, failures=["no results file"])).note, "FAILED SESSION: no results file")

    def test_an_incomplete_review_in_a_failed_session_says_both(self):
        text = "INCOMPLETE REVIEW: 1 problem(s)\n" + BUG_ONLY
        score = s.score_summary(summary(text, failures=["no results file"]))
        self.assertIn("INCOMPLETE REVIEW", score.note)
        self.assertIn("FAILED SESSION: no results file", score.note)

    def test_empty_lists_and_values_that_are_not_lists_do_not_fail_a_row(self):
        for more in ({"call_errors": [], "failures": []}, {"call_errors": None, "failures": "text"}, {}):
            with self.subTest(more=more):
                score = s.score_summary(summary(BUG_ONLY, **more))
                self.assertEqual((score.readable, score.note), (True, ""))

    def test_a_summary_that_names_no_change_is_refused_saying_so(self):
        for named in ({}, {"change": ""}, {"change": None}, {"change": 6}):
            with self.subTest(named=named), self.assertRaises(ValueError) as caught:
                s.score_summary({**named, "review_text": BUG_ONLY})
            self.assertIn("names no change", str(caught.exception))

    def test_a_change_the_key_does_not_list_is_refused_naming_the_ones_it_does(self):
        with self.assertRaises(ValueError) as caught:
            s.score_summary(summary(BUG_ONLY, change="change-9"))
        message = str(caught.exception)
        self.assertIn("'change-9'", message)
        for known in ("change-1", "change-6"):
            self.assertIn(known, message)


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

    def test_one_line_row_carries_the_change_and_the_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, summary(BUG_DECOY_AND_A_REMARK, final_text="", stopped_by="completed", turns=2, tool_calls=1,
                                           review_calls=1, review_errors=0, elapsed_seconds=9.5))
            code, out, _err = self.run_main(path)
        self.assertEqual(code, 0)
        for part in ("change=change-6", "stopped_by=completed", "readable=yes", "found=1/1", "decoy_alarms=1", "clean_false_alarms=0",
                     "unscored=1", "findings=3", "review_calls=1", "review_errors=0", "turns=2", "tool_calls=1", "seconds=9.5",
                     "caller_repeated=0/3"):
            self.assertIn(part, out)
        self.assertEqual(out.count("\n"), 1)  # one row

    def test_the_row_of_the_clean_change_counts_its_false_alarms(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _err = self.run_main(self.write(tmp, summary(ON_THE_CLEAN_CHANGE, change="change-5")))
        self.assertEqual(code, 0)
        for part in ("change=change-5", "found=0/0", "clean_false_alarms=2", "unscored=0"):
            self.assertIn(part, out)

    def test_an_unreadable_review_says_why_in_the_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            _code, out, _err = self.run_main(self.write(tmp, summary("MCP request timed out after 60000ms")))
        self.assertIn("readable=NO", out)
        self.assertIn("found=0/1", out)
        self.assertIn("(no review ran: MCP request timed out after 60000ms)", out)

    def test_json_flag_prints_the_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _err = self.run_main(self.write(tmp, summary(BUG_DECOY_AND_A_REMARK)), "--json")
        self.assertEqual(code, 0)
        printed = json.loads(out)
        self.assertEqual((printed["change"], printed["found"], printed["on_decoys"], printed["missed"]), ("change-6", ["B7"], ["D4"], []))
        self.assertEqual(len(printed["unscored"]), 1)
        self.assertNotIn("other", printed)  # renamed: `unscored`, as the key's own report calls it

    def test_a_file_that_is_not_json_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _out, err = self.run_main(self.write(tmp, "not json"))
        self.assertEqual(code, 2)
        self.assertIn("not a JSON object", err)

    def test_a_file_with_a_byte_order_mark_is_read(self):
        # Windows PowerShell 5.1's `>` writes UTF-16 with a mark; some editors put a mark on UTF-8.
        text = json.dumps(summary(BUG_ONLY))
        for encoding, mark in (("utf-16-le", b"\xff\xfe"), ("utf-16-be", b"\xfe\xff"), ("utf-8", b"\xef\xbb\xbf")):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "run.json"
                path.write_bytes(mark + text.encode(encoding))
                code, out, err = self.run_main(str(path), "--json")
                self.assertEqual(code, 0, err)
                self.assertEqual(json.loads(out)["found"], ["B7"])

    def test_a_file_that_is_not_text_in_either_encoding_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_bytes(b'{"review_text": "\x80"}')
            code, _out, err = self.run_main(str(path))
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)

    def test_a_missing_file_exits_2(self):
        code, _out, err = self.run_main("/no/such/run.json")
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)

    def test_a_summary_that_names_no_change_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self.run_main(self.write(tmp, {"review_text": BUG_ONLY, "stopped_by": "completed"}))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("names no change", err)

    def test_an_unknown_change_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self.run_main(self.write(tmp, summary(BUG_ONLY, change="change-9")))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("'change-9'", err)

    def test_the_old_case_flag_is_gone_because_the_change_comes_from_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                s.main([self.write(tmp, summary(BUG_ONLY)), "--case", "seeded-three"])
        self.assertEqual(caught.exception.code, 2)

    def test_a_key_that_cannot_be_read_exits_2_saying_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.write(tmp, summary(BUG_ONLY))
            with mock.patch.object(live_scoring_lib, "ANSWER_KEY_PATH", Path(tmp) / "no-such-key.json"):
                code, out, err = self.run_main(run)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("answer key", err)
        self.assertIn("no-such-key.json", err)


class RunAsAScript(unittest.TestCase):
    """The way it is used: `python harness/pi/score_run.py run.json`, in a new interpreter with nothing on the path."""

    def run_script(self, path: Path, *extra):
        return subprocess.run([sys.executable, "-I", str(PI_DIR / "score_run.py"), str(path), *extra], capture_output=True, text=True, timeout=60)

    def test_it_scores_a_summary_without_help_from_the_caller_s_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run.json"
            run.write_text(json.dumps(summary(BUG_ONLY, stopped_by="completed")), encoding="utf-8")
            completed = self.run_script(run)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(completed.stdout.startswith("change=change-6 stopped_by=completed readable=yes found=1/1"), completed.stdout)

    def test_scoring_loads_neither_cases_nor_harness_lib(self):
        program = (
            "import sys\n"
            f"sys.path.insert(0, {str(PI_DIR)!r})\n"
            "import score_run\n"
            "score_run.score_summary({'change': 'change-6', 'review_text': 'Reasoning review: 0 finding(s)\\n'})\n"
            "print(sorted(m for m in ('cases', 'harness_lib', 'live_branch_lib') if m in sys.modules))\n"
        )
        completed = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[]")

    def test_a_summary_with_no_change_exits_2_in_the_script_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run.json"
            run.write_text(json.dumps({"review_text": BUG_ONLY}), encoding="utf-8")
            completed = self.run_script(run)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("names no change", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
