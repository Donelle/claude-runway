#!/usr/bin/env python3
"""Tests for libs/verifier_reviewer_lib.py.

Stdlib-only (unittest), no network: the model completion is a stub and LM Studio's
REST API is `tests/fake_lmstudio.py`.

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import verifier_reviewer_lib  # noqa: E402
from codebase_reviewer_config_lib import Reviewer, ReviewersConfig  # noqa: E402
from fake_lmstudio import FakeLmStudio  # noqa: E402
from lmstudio_model_loader_lib import ModelLoader, ModelNotLoadedError  # noqa: E402
from reasoning_reviewer_lib import Alternate, Finding, ReasoningReview, ReviewError, collect_findings  # noqa: E402
from verifier_reviewer_lib import (  # noqa: E402
    CONFIRMED,
    DISPUTED,
    REJECTED,
    UNCHECKED,
    Decision,
    VerificationReview,
    VerifiedReview,
    build_verifier_message,
    build_verifier_prompt,
    code_window,
    combine,
    flatten,
    format_verified_review,
    git_file_reader,
    ground_findings,
    locate_quote,
    parse_verdict,
    run_verified_review,
    verify_findings,
)

BASE = "http://localhost:1234/v1"

SOURCE = "\n".join(
    [
        "import subprocess",  # 1
        "",  # 2
        "def run(cmd):",  # 3
        "    return subprocess.run(cmd, shell=True)",  # 4
        "",  # 5
        "def clean(items=[]):",  # 6
        "    items.append(1)",  # 7
        "    return items",  # 8
    ]
) + "\n"


def rv(name, model="model-a", roles=("verifier",), *, prompt="check it", active=True, on_demand=False, context=None) -> Reviewer:
    return Reviewer(
        name=name,
        model=model,
        prompt=prompt,
        active=active,
        roles=tuple(r for r in ("reasoning", "verifier", "classifier") if r in roles),
        load_on_demand=on_demand,
        context_length=context,
        prompt_file="",
        # The tests' reasoning stub replies are JSON; the library's default for a reasoning reviewer is lines.
        reply_format="json",
    )


def cfg(*reviewers) -> ReviewersConfig:
    return ReviewersConfig(rubric=(), reviewers=tuple(reviewers))


def fnd(quote="subprocess.run(cmd, shell=True)", file="a.py", start=4, end=4, cls="injection", claim="runs the input in a shell",
        failing="cmd='x; rm -rf /'", reviewers=("r1",), alternates=()) -> Finding:
    return Finding(file, start, end, cls, claim, failing, reviewers, alternates, quote)


def reader(files):
    return lambda path: files.get(path)


def verdict(kind="confirmed", failing="cmd='a; b'", wrong="runs b as well", reasoning="traced it"):
    return json.dumps({"verdict": kind, "failing_input": failing, "wrong_result": wrong, "reasoning": reasoning})


class StubComplete:
    def __init__(self, replies):
        self.replies = replies  # callable(model, system, content) -> str | None
        self.calls: list = []
        self._lock = threading.Lock()

    def __call__(self, client, model, system, content):
        with self._lock:
            self.calls.append((model, system, content))
        return self.replies(model, system, content)


class LocateQuote(unittest.TestCase):
    def test_a_quote_at_the_claimed_line_is_found_there(self):
        self.assertEqual(locate_quote("subprocess.run(cmd, shell=True)", SOURCE, 4, 4), (4, 4))

    def test_indentation_tabs_and_wrapping_do_not_matter(self):
        self.assertEqual(locate_quote("return   subprocess.run(cmd,\tshell=True)", SOURCE, 4, 4), (4, 4))
        self.assertEqual(locate_quote("def run(cmd):\n    return subprocess.run(cmd, shell=True)", SOURCE, 3, 4), (3, 4))

    def test_the_occurrence_nearest_the_claim_wins(self):
        text = "\n".join(["x = compute()"] + ["pad"] * 18 + ["x = compute()"]) + "\n"  # lines 1 and 20
        self.assertEqual(locate_quote("x = compute()", text, 18, 19), (20, 20))
        self.assertEqual(locate_quote("x = compute()", text, 3, 4), (1, 1))

    def test_a_quote_on_thousands_of_lines_is_found_at_the_claimed_line(self):
        # A cap on the occurrences looked at stops short of the one at the claim and moves the finding to unrelated code.
        text = "total += 1\n" * 12000
        for claimed in (1, 4999, 5000, 5001, 11000, 12000):
            with self.subTest(claimed=claimed):
                self.assertEqual(locate_quote("total += 1", text, claimed, claimed), (claimed, claimed))

    def test_a_claim_before_the_first_occurrence_or_past_the_last_takes_the_nearest(self):
        text = "\n".join(["pad"] * 4 + ["x = compute()"] + ["pad"] * 10 + ["x = compute()"] + ["pad"] * 3) + "\n"  # lines 5 and 16
        self.assertEqual(locate_quote("x = compute()", text, 1, 1), (5, 5))
        self.assertEqual(locate_quote("x = compute()", text, 99, 99), (16, 16))
        self.assertEqual(locate_quote("x = compute()", text, 10, 10), (5, 5))  # 5 lines from line 5, 6 from line 16
        self.assertEqual(locate_quote("x = compute()", text, 11, 11), (16, 16))  # 6 lines from line 5, 5 from line 16
        self.assertEqual(locate_quote("x = compute()", text, 10, 11), (5, 5))  # 5 lines from each: the first of equals wins

    def test_a_claim_with_no_line_takes_the_first_occurrence(self):
        text = "x = compute()\npad\nx = compute()\n"
        self.assertEqual(locate_quote("x = compute()", text, 0, 0), (1, 1))

    def test_a_quote_that_is_not_there_or_is_too_short_matches_nothing(self):
        self.assertIsNone(locate_quote("except:", SOURCE, 4, 4))
        self.assertIsNone(locate_quote("ab", SOURCE, 4, 4))
        self.assertIsNone(locate_quote("   \n ", SOURCE, 4, 4))

    def test_a_quote_spanning_lines_skips_blank_lines_between_them(self):
        # Lines 1 and 3 with a blank line 2 between: the squeezed text has one space where the break was.
        self.assertEqual(locate_quote("import subprocess def run(cmd):", SOURCE, 1, 3), (1, 3))


class GroundFindings(unittest.TestCase):
    FILES = {"a.py": SOURCE}

    def ground(self, *findings, files=None):
        return ground_findings(findings, reader(self.FILES if files is None else files))

    def test_a_quote_found_at_the_claimed_lines_is_kept(self):
        grounded, ungrounded = self.ground(fnd(start=4, end=4))
        self.assertEqual((len(grounded), ungrounded), (1, []))
        self.assertEqual((grounded[0].line_start, grounded[0].line_end), (4, 4))

    def test_a_quote_found_elsewhere_in_the_file_is_kept_and_its_lines_corrected(self):
        (kept,), ungrounded = self.ground(fnd(start=7, end=7))  # the model said line 7; the code is on line 4
        self.assertEqual(ungrounded, [])
        self.assertEqual((kept.line_start, kept.line_end), (4, 4))

    def test_the_corrected_lines_are_those_of_the_match_nearest_the_claim(self):
        files = {"a.py": "x = compute()\n" + "pad\n" * 18 + "x = compute()\n"}
        (kept,), _ = self.ground(fnd(quote="x = compute()", start=19, end=19), files=files)
        self.assertEqual((kept.line_start, kept.line_end), (20, 20))

    def test_a_quote_that_is_not_in_the_file_is_dropped_as_ungrounded(self):
        grounded, ungrounded = self.ground(fnd(quote="except:", claim="bare except"))
        self.assertEqual(grounded, [])
        self.assertEqual(len(ungrounded), 1)
        self.assertEqual(ungrounded[0].finding.claim, "bare except")
        self.assertIn("not in a.py", ungrounded[0].reason)

    def test_a_quote_differing_only_in_whitespace_matches(self):
        for quote in ("   subprocess.run(cmd,   shell=True)  ", "subprocess.run(cmd,\n\tshell=True)", "return\tsubprocess.run(cmd, shell=True)"):
            with self.subTest(quote=quote):
                grounded, ungrounded = self.ground(fnd(quote=quote))
                self.assertEqual((len(grounded), ungrounded), (1, []))

    def test_a_finding_with_no_quote_is_ungrounded(self):
        for quote in ("", "   \n"):
            with self.subTest(quote=quote):
                grounded, ungrounded = self.ground(fnd(quote=quote))
                self.assertEqual(grounded, [])
                self.assertIn("no quote", ungrounded[0].reason)

    def test_a_quote_too_short_to_place_is_ungrounded(self):
        _, ungrounded = self.ground(fnd(quote="x"))
        self.assertIn("too short", ungrounded[0].reason)

    def test_two_reports_of_one_claim_stay_grounded_when_one_quote_is_too_short(self):
        # `[]` sits inside the longer quote, so the reports merge; keeping the shorter one would throw away a
        # finding that grounds on its own.
        files = {"a.py": "def clean(items=[]):\n    items.append(1)\n"}
        for first, second in (("def clean(items=[]):", "[]"), ("[]", "def clean(items=[]):")):
            with self.subTest(first=first):
                reports = [fnd(quote=first, claim="mutable default", start=1, end=1), fnd(quote=second, claim="mutable default", start=1, end=1)]
                merged = [f for group in collect_findings(reports) for f in group.findings]
                grounded, ungrounded = self.ground(*merged, files=files)
                self.assertEqual((len(grounded), ungrounded), (1, []))
                self.assertEqual(grounded[0].quote, "def clean(items=[]):")

    def test_a_file_that_is_not_in_the_change_is_ungrounded(self):
        for name in ("hooks/*.py", "nope.py", ""):
            with self.subTest(file=name):
                grounded, ungrounded = self.ground(fnd(file=name))
                self.assertEqual(grounded, [])
                self.assertIn("not a file in this change", ungrounded[0].reason)

    def test_a_dot_slash_path_is_read_as_the_file_it_names_and_reported_cleanly(self):
        (kept,), _ = self.ground(fnd(file="./a.py"))
        self.assertEqual(kept.file, "a.py")

    def test_a_windows_style_path_is_read_as_the_changed_file_with_slashes_and_reported_that_way(self):
        # A model that writes `src\a.py` for the changed `src/a.py`: tried as written first, then with `/`.
        (kept,), ungrounded = self.ground(fnd(file="src\\a.py"), files={"src/a.py": SOURCE})
        self.assertEqual((kept.file, ungrounded), ("src/a.py", []))

    def test_each_claim_of_a_merged_finding_is_grounded_on_its_own_quote(self):
        merged = fnd(
            alternates=(
                Alternate("shared default list", "clean() twice", ("r2",), 6, 7, "def clean(items=[]):"),
                Alternate("made up claim", "", ("r3",), 7, 7, "this is not in the file"),
            ),
        )
        grounded, ungrounded = self.ground(merged)
        self.assertEqual([(g.claim, g.reviewers, g.line_start) for g in grounded],
                         [("runs the input in a shell", ("r1",), 4), ("shared default list", ("r2",), 6)])
        self.assertEqual([u.finding.claim for u in ungrounded], ["made up claim"])

    def test_flatten_keeps_the_class_and_gives_each_claim_its_own_lines(self):
        merged = fnd(start=1, end=9, alternates=(Alternate("second", "in", ("r2",), 6, 7, "q"),))
        first, second = flatten(merged)
        self.assertEqual((first.alternates, first.line_start, first.line_end), ((), 1, 9))
        self.assertEqual((second.claim, second.cls, second.line_start, second.line_end, second.quote, second.reviewers),
                         ("second", "injection", 6, 7, "q", ("r2",)))


class ParseVerdict(unittest.TestCase):
    def test_a_confirmation_with_its_failing_input_and_wrong_result(self):
        decision, problem = parse_verdict(verdict(), "v")
        self.assertIsNone(problem)
        self.assertEqual(decision, Decision("v", CONFIRMED, "cmd='a; b'", "runs b as well", "traced it"))

    def test_a_rejection_needs_nothing_else(self):
        decision, problem = parse_verdict('{"verdict": "rejected", "reasoning": "the command is a constant"}', "v")
        self.assertEqual((problem, decision), (None, Decision("v", REJECTED, "", "", "the command is a constant")))

    def test_fences_prose_think_envelopes_and_case_are_tolerated(self):
        body = verdict("Confirmed")
        for text in (f"```json\n{body}\n```", f"Here you go: {body} done", f"<think>maybe {{x}}</think>\n{body}"):
            with self.subTest(text=text[:20]):
                self.assertEqual(parse_verdict(text, "v")[0].verdict, CONFIRMED)

    def test_a_confirmation_without_a_failing_input_or_a_wrong_result_is_not_a_confirmation(self):
        for failing, wrong in (("", "wrong"), ("input", ""), ("", ""), (None, "x")):
            with self.subTest(failing=failing, wrong=wrong):
                decision, problem = parse_verdict(verdict(failing=failing, wrong=wrong), "v")
                self.assertIsNone(decision)
                self.assertIn("without a failing input", problem)

    def test_junk_is_a_problem_not_an_exception(self):
        for text in ("I think it is fine.", "[1, 2]", '{"verdict": "maybe"}', '{"verdict": 3}', "{}"):
            with self.subTest(text=text):
                decision, problem = parse_verdict(text, "v")
                self.assertEqual((decision, isinstance(problem, str)), (None, True))

    def test_an_invalid_verdict_is_a_fixed_sentence_without_what_the_model_wrote(self):
        # Problems are recorded (session log, request row); the value the model wrote can be any text.
        _, problem = parse_verdict('{"verdict": "SECRET_FROM_THE_REPLY", "reasoning": "x"}', "v")
        self.assertEqual(problem, "reviewer 'v': the verdict is neither 'confirmed' nor 'rejected'")


class Combine(unittest.TestCase):
    def test_outcomes(self):
        c, r = Decision("a", CONFIRMED), Decision("b", REJECTED)
        self.assertEqual(combine([c, Decision("c", CONFIRMED)]), CONFIRMED)
        self.assertEqual(combine([r, Decision("c", REJECTED)]), REJECTED)
        self.assertEqual(combine([c, r]), DISPUTED)
        self.assertEqual(combine([c]), CONFIRMED)
        self.assertEqual(combine([]), UNCHECKED)


class CodeShown(unittest.TestCase):
    def test_the_window_numbers_the_lines_and_marks_the_cited_ones(self):
        window = code_window(SOURCE, 4, 4, 10_000)
        self.assertIn("    4 >     return subprocess.run(cmd, shell=True)", window)
        self.assertIn("    3   def run(cmd):", window)
        self.assertEqual(window.count(">"), 1)

    def test_a_small_budget_shrinks_the_context_before_the_cited_lines(self):
        window = code_window(SOURCE, 4, 4, 60)
        self.assertIn("return subprocess.run(cmd, shell=True)", window)
        self.assertNotIn("import subprocess", window)
        self.assertLessEqual(len(window), 60)

    def test_cited_lines_longer_than_the_budget_are_cut_and_say_so(self):
        window = code_window("x = " + "9" * 500 + "\n", 1, 1, 100)
        self.assertIn("[cut:", window)

    def test_a_window_on_an_empty_file_is_empty(self):
        self.assertEqual(code_window("", 1, 1, 100), "")

    def test_the_message_carries_the_claim_the_reporters_input_the_quote_and_the_code(self):
        message = build_verifier_message(fnd(), "    4 > code")
        for part in ("File: a.py:4", "Kind of defect: injection", "Claim: runs the input in a shell",
                     "cmd='x; rm -rf /'", "subprocess.run(cmd, shell=True)", "    4 > code"):
            self.assertIn(part, message)
        self.assertIn("(none)", build_verifier_message(fnd(failing=""), ""))

    def test_the_prompt_is_the_lens_then_how_to_verify_then_the_reply_format(self):
        prompt = build_verifier_prompt(rv("v", prompt="LENS-V"))
        order = [prompt.index(x) for x in ("LENS-V", "skeptical", '"verdict"')]
        self.assertEqual(order, sorted(order))
        self.assertIn("would have to be true", prompt)
        self.assertIn("concrete input", prompt)


class GitFileReader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

        def git(*args):
            subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True, env=env)

        git("init", "-q", "-b", "main")
        # newline="\n": on Windows write_text would commit CRLF blobs, and these tests compare what
        # `git show` returns with the text written here.
        (self.repo / "a.py").write_text("old = 1\n", encoding="utf-8", newline="\n")
        (self.repo / "other.py").write_text("other = 2\n", encoding="utf-8", newline="\n")
        git("add", ".")
        git("commit", "-q", "-m", "base")
        (self.repo / "a.py").write_text("new = 2\n", encoding="utf-8", newline="\n")
        git("commit", "-q", "-am", "change")
        (self.repo / "a.py").write_text("uncommitted = 3\n", encoding="utf-8")

    def test_the_file_is_read_as_it_is_at_the_head_ref_not_as_it_is_on_disk(self):
        read = git_file_reader(self.repo, "HEAD", ["a.py"])
        self.assertEqual(read("a.py"), "new = 2\n")
        self.assertEqual(git_file_reader(self.repo, "HEAD~1", ["a.py"])("a.py"), "old = 1\n")

    def test_only_files_of_the_change_can_be_read(self):
        read = git_file_reader(self.repo, "HEAD", ["a.py"])
        self.assertIsNone(read("other.py"))
        self.assertIsNone(read("../etc/passwd"))
        self.assertEqual(read("./a.py"), "new = 2\n")

    def test_a_file_missing_at_the_head_is_none(self):
        self.assertIsNone(git_file_reader(self.repo, "HEAD", ["gone.py"])("gone.py"))

    def test_a_head_ref_that_could_be_an_option_is_refused(self):
        for ref in ("--output=/tmp/x", "-p", "a b"):
            with self.subTest(ref=ref), self.assertRaises(ReviewError):
                git_file_reader(self.repo, ref, ["a.py"])

    def test_a_directory_that_is_not_a_repository_reads_nothing(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(git_file_reader(empty, "HEAD", ["a.py"])("a.py"))

    def _commit_two_files_whose_names_differ_only_by_a_backslash(self):
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        (self.repo / "x").mkdir()
        (self.repo / "x" / "b.py").write_text("slash = 1\n", encoding="utf-8")
        (self.repo / "x\\b.py").write_text("danger = os.system(cmd)\n", encoding="utf-8")
        for args in (["add", "x"], ["add", "--", "x\\b.py"], ["commit", "-q", "-m", "names"]):
            subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True, env=env)

    @unittest.skipIf(os.name == "nt", "a file name cannot contain a backslash on Windows")
    def test_x_backslash_b_and_x_slash_b_are_two_files_and_a_name_reads_only_its_own(self):
        self._commit_two_files_whose_names_differ_only_by_a_backslash()
        only_backslash = git_file_reader(self.repo, "HEAD", ["x\\b.py"])
        self.assertEqual(only_backslash("x\\b.py"), "danger = os.system(cmd)\n")
        self.assertIsNone(only_backslash("x/b.py"))  # not a changed file: the other spelling is not a guess at it
        both = git_file_reader(self.repo, "HEAD", ["x\\b.py", "x/b.py"])
        self.assertEqual(both("x\\b.py"), "danger = os.system(cmd)\n")
        self.assertEqual(both("x/b.py"), "slash = 1\n")
        self.assertEqual(both("./x/b.py"), "slash = 1\n")

    @unittest.skipIf(os.name == "nt", "a file name cannot contain a backslash on Windows")
    def test_grounding_keeps_each_finding_on_the_file_it_named_when_two_names_differ_only_by_a_backslash(self):
        self._commit_two_files_whose_names_differ_only_by_a_backslash()
        both = git_file_reader(self.repo, "HEAD", ["x\\b.py", "x/b.py"])
        (backslash, slash), ungrounded = ground_findings(
            [fnd(file="x\\b.py", quote="os.system(cmd)", start=1, end=1), fnd(file="x/b.py", quote="slash = 1", start=1, end=1)], both)
        self.assertEqual((ungrounded, backslash.file, slash.file), ([], "x\\b.py", "x/b.py"))
        # A finding on a file that is not in the change is not attributed to the other spelling.
        grounded, ungrounded = ground_findings([fnd(file="x/b.py", quote="os.system(cmd)", start=1, end=1)],
                                               git_file_reader(self.repo, "HEAD", ["x\\b.py"]))
        self.assertEqual(grounded, [])
        self.assertIn("not a file in this change", ungrounded[0].reason)

    def test_a_quote_at_the_end_of_a_large_file_is_found(self):
        # A small edit near the end of a file of several megabytes reaches the reasoning stage; cutting the
        # file short would make its quote look absent and drop a genuine finding as ungrounded.
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        lines = 400_000
        (self.repo / "big.py").write_text("# padding\n" * lines + "os.system(cmd)\n", encoding="utf-8")
        for args in (["add", "big.py"], ["commit", "-q", "-m", "big"]):
            subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True, env=env)
        text = git_file_reader(self.repo, "HEAD", ["big.py"])("big.py")
        self.assertGreater(len(text or ""), 3_000_000)
        grounded, ungrounded = ground_findings([fnd(quote="os.system(cmd)", file="big.py", start=lines + 1, end=lines + 1)],
                                               git_file_reader(self.repo, "HEAD", ["big.py"]))
        self.assertEqual((len(grounded), ungrounded), (1, []))
        self.assertEqual(grounded[0].line_start, lines + 1)


class VerifyFindings(unittest.TestCase):
    FILES = {"a.py": SOURCE}

    def _verify(self, config, replies, findings, *, loaded=("model-a", "model-b"), files=None, max_reviewers=3):
        fake = FakeLmStudio(loaded=loaded)
        stub = StubComplete(replies)
        result = verify_findings(config, ModelLoader(BASE, http=fake), object(), findings, reader(self.FILES if files is None else files),
                                 max_reviewers, complete=stub)
        return result, stub, fake

    def test_when_every_verifier_confirms_the_finding_is_confirmed(self):
        config = cfg(rv("v1", "model-a"), rv("v2", "model-b"))
        result, stub, _ = self._verify(config, lambda *a: verdict("confirmed"), [fnd()])
        self.assertEqual([v.status for v in result.verified], [CONFIRMED])
        self.assertEqual(sorted(d.reviewer for d in result.confirmed[0].decisions), ["v1", "v2"])
        self.assertEqual(sorted(result.verifiers_run), ["v1", "v2"])
        self.assertEqual(result.errors, [])
        self.assertEqual(len(stub.calls), 2)

    def test_when_every_verifier_rejects_the_finding_is_dropped_and_recorded(self):
        config = cfg(rv("v1", "model-a"), rv("v2", "model-b"))
        result, _, _ = self._verify(config, lambda *a: verdict("rejected", "", "", "the command is constant"), [fnd()])
        self.assertEqual([v.status for v in result.verified], [REJECTED])
        self.assertEqual(result.kept, [])
        self.assertEqual(result.rejected[0].decisions[0].reasoning, "the command is constant")

    def test_when_the_verifiers_disagree_the_finding_is_a_dispute_with_each_decision(self):
        config = cfg(rv("v1", "model-a"), rv("v2", "model-b"))

        def replies(model, system, content):
            return verdict("confirmed", reasoning="it is exploitable") if model == "model-a" else verdict("rejected", "", "", "a guard exists")

        result, _, _ = self._verify(config, replies, [fnd()])
        (disputed,) = result.disputed
        self.assertEqual({d.reviewer: (d.verdict, d.reasoning) for d in disputed.decisions},
                         {"v1": (CONFIRMED, "it is exploitable"), "v2": (REJECTED, "a guard exists")})
        self.assertEqual(len(result.kept), 1)  # a dispute is returned, not dropped
        text = format_verified_review(VerifiedReview(ReasoningReview(reviewers_run=["r1"]), result))
        self.assertIn("1 disputed", text)
        self.assertIn("v1 confirmed: it is exploitable", text)
        self.assertIn("v2 rejected: a guard exists", text)

    def test_with_no_verifier_active_the_grounded_findings_pass_through_unchecked_and_nothing_is_loaded(self):
        config = cfg(rv("r", roles=("reasoning",)), rv("off", roles=("verifier",), active=False))
        result, stub, fake = self._verify(config, lambda *a: verdict(), [fnd(), fnd(quote="not in the file", claim="made up")])
        self.assertEqual([(v.status, v.note) for v in result.verified], [(UNCHECKED, "no verifier is active")])
        self.assertEqual(len(result.ungrounded), 1)  # grounding still runs
        self.assertEqual((stub.calls, result.verifiers_run), ([], []))
        self.assertEqual(fake.posts("/models/load"), [])
        text = format_verified_review(VerifiedReview(ReasoningReview(reviewers_run=["r"]), result))
        self.assertIn("GROUNDED", text)
        self.assertIn("NOT VERIFIED", text)
        self.assertIn("1 unchecked", text)

    def test_ungrounded_findings_never_reach_a_verifier_and_are_counted(self):
        config = cfg(rv("v1", "model-a"))
        findings = [fnd(), fnd(quote="except:", claim="bare except"), fnd(quote="", claim="no quote")]
        result, stub, _ = self._verify(config, lambda *a: verdict(), findings)
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(sorted(u.finding.claim for u in result.ungrounded), ["bare except", "no quote"])
        text = format_verified_review(VerifiedReview(ReasoningReview(reviewers_run=["r"]), result))
        self.assertIn("dropped 0 rejected and 2 ungrounded", text)
        self.assertIn("## Dropped: ungrounded (2)", text)
        self.assertIn("ungrounded: its quote is not in a.py", text)

    def test_nothing_grounded_means_no_model_is_loaded_or_asked(self):
        config = cfg(rv("v1", "model-a", on_demand=True))
        result, stub, fake = self._verify(config, lambda *a: verdict(), [fnd(quote="nope nope")], loaded=())
        self.assertEqual((stub.calls, result.verified), ([], []))
        self.assertEqual(len(result.ungrounded), 1)
        self.assertEqual(fake.posts("/models/load"), [])

    def test_the_verifier_is_shown_the_claim_and_the_numbered_code_with_the_corrected_lines(self):
        config = cfg(rv("v1", "model-a", prompt="LENS-V1"))
        _, stub, _ = self._verify(config, lambda *a: verdict(), [fnd(start=8, end=8)])  # the code is on line 4, not 8
        (model, system, content), = stub.calls
        self.assertIn("LENS-V1", system)
        self.assertIn("File: a.py:4", content)
        self.assertIn("    4 >     return subprocess.run(cmd, shell=True)", content)
        self.assertIn("runs the input in a shell", content)

    def test_a_confirmation_without_a_failing_input_does_not_count_and_the_other_verifiers_vote_decides(self):
        config = cfg(rv("v1", "model-a"), rv("v2", "model-b"))

        def replies(model, system, content):
            return verdict("confirmed", failing="") if model == "model-a" else verdict("rejected", "", "", "fine")

        result, _, _ = self._verify(config, replies, [fnd()])
        self.assertEqual([v.status for v in result.verified], [REJECTED])
        self.assertTrue(any("without a failing input" in e for e in result.errors))

    def test_when_no_verifier_gives_a_usable_answer_the_finding_is_unchecked_not_dropped(self):
        config = cfg(rv("v1", "model-a"))
        result, _, _ = self._verify(config, lambda *a: "I am not sure.", [fnd()])
        self.assertEqual([(v.status, v.note) for v in result.verified], [(UNCHECKED, "no verifier gave a usable answer")])
        self.assertEqual(len(result.kept), 1)
        self.assertTrue(any("not JSON" in e for e in result.errors))

    def test_a_verifier_whose_requests_all_fail_is_reported_and_the_other_still_decides(self):
        config = cfg(rv("v1", "model-a"), rv("v2", "model-b"))

        def replies(model, system, content):
            return None if model == "model-a" else verdict("confirmed")

        result, _, _ = self._verify(config, replies, [fnd()])
        self.assertEqual([v.status for v in result.verified], [CONFIRMED])
        self.assertEqual(result.verifiers_run, ["v2"])
        self.assertTrue(any("v1" in e and "every request failed" in e for e in result.errors))

    def test_each_verifier_decides_each_finding_and_findings_keep_their_own_decisions(self):
        config = cfg(rv("v1", "model-a"))
        findings = [fnd(), fnd(quote="def clean(items=[]):", start=6, end=6, cls="shared-state", claim="mutable default")]

        def replies(model, system, content):
            return verdict("confirmed") if "mutable default" in content else verdict("rejected", "", "", "constant")

        result, _, _ = self._verify(config, replies, findings)
        self.assertEqual({v.finding.claim: v.status for v in result.verified},
                         {"runs the input in a shell": REJECTED, "mutable default": CONFIRMED})

    def test_findings_over_the_limit_are_not_verified_and_come_back_unchecked_with_a_warning(self):
        config = cfg(rv("v1", "model-a"))
        findings = [fnd(claim=f"claim {n}") for n in range(3)]
        with mock.patch.object(verifier_reviewer_lib, "MAX_VERIFIED_FINDINGS", 2):
            result, stub, _ = self._verify(config, lambda *a: verdict(), findings)
        self.assertEqual(len(stub.calls), 2)
        self.assertEqual([v.status for v in result.verified], [CONFIRMED, CONFIRMED, UNCHECKED])
        self.assertTrue(any("beyond the first 2" in w for w in result.warnings))

    def test_a_resident_model_that_is_not_loaded_refuses_before_anything_runs(self):
        config = cfg(rv("v1", "model-a"))
        stub = StubComplete(lambda *a: verdict())
        with self.assertRaises(ModelNotLoadedError):
            verify_findings(config, ModelLoader(BASE, http=FakeLmStudio(loaded=())), object(), [fnd()], reader(self.FILES), 3, complete=stub)
        self.assertEqual(stub.calls, [])


class RunVerifiedReview(unittest.TestCase):
    DIFF = (
        "diff --git a/a.py b/a.py\nindex 1..2 100644\n--- a/a.py\n+++ b/a.py\n"
        "@@ -1,0 +1,8 @@\n" + "".join(f"+{line}\n" for line in SOURCE.splitlines())
    )

    def _run(self, config, replies, *, loaded=("model-a", "model-b")):
        fake = FakeLmStudio(loaded=loaded)
        stub = StubComplete(replies)
        result = run_verified_review(config, ModelLoader(BASE, http=fake), object(), self.DIFF, "/unused", "HEAD", 3,
                                     complete=stub, read_file=reader({"a.py": SOURCE}))
        return result, stub

    @staticmethod
    def _replies(verifier_says):
        def replies(model, system, content):
            if "checking ONE reported defect" in system:
                return verifier_says
            return json.dumps({"findings": [
                {"file": "a.py", "line_start": 4, "line_end": 4, "class": "injection", "claim": "runs the input in a shell",
                 "failing_input": "cmd='a; b'", "quote": "subprocess.run(cmd, shell=True)"},
                {"file": "a.py", "line_start": 7, "line_end": 7, "class": "bare-except", "claim": "a bare except hides errors",
                 "failing_input": "any error", "quote": "except:"},
            ]})
        return replies

    def test_reasoning_then_grounding_then_the_verifiers(self):
        config = cfg(rv("finder", "model-a", roles=("reasoning",)), rv("checker", "model-b", roles=("verifier",)))
        result, stub = self._run(config, self._replies(verdict("confirmed")))
        self.assertEqual(result.reasoning.finding_count, 2)
        review = result.verification
        self.assertEqual([v.finding.claim for v in review.confirmed], ["runs the input in a shell"])
        self.assertEqual([u.finding.claim for u in review.ungrounded], ["a bare except hides errors"])
        # The decoy never reached the verifier: one reasoning request, one verifier request.
        self.assertEqual([m for m, _, _ in stub.calls], ["model-a", "model-b"])
        text = format_verified_review(result)
        self.assertIn("1 confirmed", text)
        self.assertIn("1 ungrounded", text)
        self.assertIn("finder", text)

    def test_a_verifier_that_rejects_removes_the_finding(self):
        config = cfg(rv("finder", "model-a", roles=("reasoning",)), rv("checker", "model-b", roles=("verifier",)))
        result, _ = self._run(config, self._replies(verdict("rejected", "", "", "constant")))
        self.assertEqual(result.verification.kept, [])

    def test_a_verifier_model_that_is_not_loaded_is_reported_before_the_reasoning_stage_starts(self):
        config = cfg(rv("finder", "model-a", roles=("reasoning",)), rv("checker", "model-b", roles=("verifier",)))
        fake = FakeLmStudio(loaded=("model-a",))
        stub = StubComplete(self._replies(verdict()))
        with self.assertRaises(ModelNotLoadedError):
            run_verified_review(config, ModelLoader(BASE, http=fake), object(), self.DIFF, "/unused", "HEAD", 3,
                                complete=stub, read_file=reader({"a.py": SOURCE}))
        self.assertEqual(stub.calls, [])

    def test_with_no_verifier_the_result_says_the_findings_are_grounded_but_not_verified(self):
        config = cfg(rv("finder", "model-a", roles=("reasoning",)))
        result, _ = self._run(config, self._replies(verdict()))
        self.assertEqual([v.status for v in result.verification.verified], [UNCHECKED])
        self.assertIn("NOT VERIFIED", format_verified_review(result))

    def test_no_active_reasoning_reviewer_is_the_reasoning_stages_error(self):
        config = cfg(rv("checker", "model-b", roles=("verifier",)))
        with self.assertRaises(ReviewError):
            self._run(config, self._replies(verdict()))


class FormatVerifiedReview(unittest.TestCase):
    def test_a_confirmed_finding_shows_the_failing_input_and_the_wrong_result_the_verifier_gave(self):
        finding = fnd()
        review = VerificationReview(verified=[verifier_reviewer_lib.Verified(
            finding, CONFIRMED, (Decision("v1", CONFIRMED, "cmd='a; b'", "runs b too", "x"),))], verifiers_run=["v1"])
        text = format_verified_review(VerifiedReview(ReasoningReview(reviewers_run=["r1"]), review))
        self.assertIn("1 confirmed, 0 disputed, 0 unchecked; dropped 0 rejected and 0 ungrounded", text)
        self.assertIn("- a.py:4 [injection] [r1] runs the input in a shell", text)
        self.assertIn("v1: failing input: cmd='a; b' -> runs b too", text)

    def test_problems_and_warnings_from_both_stages_are_listed(self):
        reasoning = ReasoningReview(errors=["reasoning problem"], warnings=["reasoning warning"])
        review = VerificationReview(errors=["verifier problem"], warnings=["verifier warning"])
        text = format_verified_review(VerifiedReview(reasoning, review))
        for part in ("reasoning problem", "verifier problem", "reasoning warning", "verifier warning", "Problems (2)"):
            self.assertIn(part, text)
        self.assertTrue(text.startswith("INCOMPLETE REVIEW: 2 problem(s)"), text)

    def test_a_problem_in_either_stage_alone_opens_the_text_before_the_counts(self):
        for reasoning, review in ((ReasoningReview(errors=["reasoning problem"]), VerificationReview()),
                                  (ReasoningReview(), VerificationReview(errors=["verifier problem"]))):
            lines = format_verified_review(VerifiedReview(reasoning, review)).splitlines()
            self.assertTrue(lines[0].startswith("INCOMPLETE REVIEW: 1 problem(s)"), lines[0])
            self.assertTrue(lines[1].startswith("Verified review:"), lines[1])

    def test_a_clean_or_warnings_only_review_has_no_failure_header(self):
        for reasoning in (ReasoningReview(reviewers_run=["r"]), ReasoningReview(reviewers_run=["r"], warnings=["a note"])):
            text = format_verified_review(VerifiedReview(reasoning, VerificationReview(warnings=["another"])))
            self.assertTrue(text.startswith("Verified review:"), text)
            self.assertNotIn("INCOMPLETE", text)


if __name__ == "__main__":
    unittest.main()
