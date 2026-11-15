#!/usr/bin/env python3
"""Tests for harness/live_routes.py: the routes, how replies are counted, and the 95% verdict.

Stdlib-only (unittest), no network, no LM Studio: the model call is a stub. Where a test needs the text of a
problem a reply reader reports, it gets it from the real reader, so a reworded message fails here instead of
quietly counting an unreadable reply as readable.

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
for _directory in (REPO_ROOT / "harness", REPO_ROOT / "libs"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import harness_lib as h  # noqa: E402
import live_routes as lr  # noqa: E402
from cases import ALL_CASES  # noqa: E402
from codebase_reviewer_config_lib import Reviewer  # noqa: E402
from reasoning_reviewer_lib import _review_one, parse_findings  # noqa: E402


def _problem_from_reader(reply: str, reply_format: str) -> str:
    return parse_findings(reply, "r", "", reply_format)[1][0]


def _real_translator_problem(directory: Path) -> str:
    script = directory / "fails.py"
    script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    reviewer = Reviewer("r", "m", "p", True, ("reasoning",), False, translator=str(script))
    result = _review_one(reviewer, lambda _reviewer, _system: ["File: a.py\n    1 + x\n"], None,
                         lambda *_args: "any reply", [], [])
    return result.problems[0]


def _real_request_problem() -> str:
    replies = iter([None, "NONE"])
    reviewer = Reviewer("r", "m", "p", True, ("reasoning",), False)
    result = _review_one(reviewer, lambda _reviewer, _system: ["File: a.py\n    1 + x\n", "File: b.py\n    1 + y\n"], None,
                         lambda *_args: next(replies), [], [])
    return result.problems[0]


class Routes(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name).resolve()
        (self.dir / "p.md").write_text("a prompt\n", encoding="utf-8")
        (self.dir / "t.py").write_text("print('x')\n", encoding="utf-8")

    def test_the_built_in_routes_come_first_in_order_then_the_custom_ones(self):
        routes = lr.build_routes([f"native=json,promptFile={self.dir / 'p.md'},translator={self.dir / 't.py'}"])
        self.assertEqual([r.name for r in routes], ["lines", "json", "native"])
        self.assertEqual((routes[0].reply_format, routes[1].reply_format), ("lines", "json"))
        self.assertEqual(routes[2], lr.Route("native", "json", str(self.dir / "p.md"), str(self.dir / "t.py")))

    def test_no_builtin_leaves_only_the_given_routes(self):
        routes = lr.build_routes([f"only=lines,promptFile={self.dir / 'p.md'}"], builtin=False)
        self.assertEqual([r.name for r in routes], ["only"])

    def test_every_problem_in_the_routes_is_reported_together(self):
        with self.assertRaises(SystemExit) as caught:
            lr.build_routes([
                "bad=xml",
                f"worse=lines,promptFile={self.dir / 'missing.md'}",
                f"odd=lines,colour=red,translator={self.dir / 't.py'}",
                "justaname",
                f"lines=lines,promptFile={self.dir / 'p.md'}",
            ])
        text = str(caught.exception)
        for expected in ("the format must be one of lines, json (got 'xml')", "is not a file", "unknown setting 'colour=red'",
                         "must start with NAME=FORMAT", "route name 'lines' is used twice"):
            self.assertIn(expected, text)

    def test_a_setting_given_twice_or_without_a_path_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            lr.build_routes([f"a=lines,promptFile={self.dir / 'p.md'},promptFile={self.dir / 'p.md'}", "b=lines,translator="])
        self.assertIn("promptFile is given twice", str(caught.exception))
        self.assertIn("translator needs a path", str(caught.exception))

    def test_a_route_the_servers_validator_would_refuse_is_rejected_before_anything_runs(self):
        (self.dir / "empty.md").write_text("  \n", encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            lr.build_routes([f"blank=lines,promptFile={self.dir / 'empty.md'}"])
        self.assertIn("route 'blank'", str(caught.exception))
        self.assertIn("is empty", str(caught.exception))

    @unittest.skipIf(os.name == "nt", "the executable check means little on Windows")
    def test_a_translator_that_is_not_executable_is_rejected_up_front(self):
        (self.dir / "t.sh").write_text("#!/bin/sh\necho x\n", encoding="utf-8")  # no execute bit
        with self.assertRaises(SystemExit) as caught:
            lr.build_routes([f"sh=lines,translator={self.dir / 't.sh'}"])
        self.assertIn("is not executable", str(caught.exception))

    def test_the_validator_runs_once_every_route_has_parsed(self):
        (self.dir / "empty.md").write_text("", encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            lr.build_routes([f"blank=lines,promptFile={self.dir / 'empty.md'}", "bad=xml"])
        text = str(caught.exception)
        self.assertIn("the format must be one of lines, json", text)
        # A route that did not parse has no entry to validate; the validator's verdict shows once the parse problem is fixed.
        with self.assertRaises(SystemExit) as again:
            lr.build_routes([f"blank=lines,promptFile={self.dir / 'empty.md'}"])
        self.assertIn("is empty", str(again.exception))

    def test_a_usable_route_passes_the_servers_validator(self):
        routes = lr.build_routes([f"ok=json,promptFile={self.dir / 'p.md'},translator={self.dir / 't.py'}"])
        self.assertEqual(routes[-1].name, "ok")

    def test_no_routes_at_all_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            lr.build_routes([], builtin=False)
        self.assertIn("no routes to run", str(caught.exception))


class Classifying(unittest.TestCase):
    def test_a_reply_the_readers_could_not_read_is_a_format_problem(self):
        for reply, reply_format, reason in (
            ("not json at all", "json", "not JSON"),
            ('{"findings": 3}', "json", "no 'findings' list"),
            ("some prose and no blocks", "lines", "no FINDING blocks"),
        ):
            kind, found = lr.classify_problem(_problem_from_reader(reply, reply_format))
            self.assertEqual((kind, found), (lr.FORMAT_PROBLEM, reason), reply)

    def test_a_failed_translator_is_a_translator_problem_and_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            problem = _real_translator_problem(Path(tmp))
        kind, reason = lr.classify_problem(problem)
        self.assertEqual(kind, lr.TRANSLATOR_PROBLEM)
        self.assertIn("exited with status 1", reason)

    def test_a_request_lm_studio_never_answered_is_not_a_format_problem(self):
        self.assertEqual(lr.classify_problem(_real_request_problem())[0], lr.REQUEST_PROBLEM)
        self.assertEqual(lr.classify_problem("reviewer 'general': every request failed; the last one: x")[0], lr.REQUEST_PROBLEM)

    def test_an_item_skipped_from_a_reply_that_was_read_is_not_unreadable(self):
        reply = "FINDING\nfile: a.py\nlines: 1-1\nclass: c\nEND"  # a block with no claim: read, then skipped
        problems = parse_findings(reply, "r", "", "lines")[1]
        self.assertTrue(problems)
        self.assertEqual(lr.classify_problem(problems[0])[0], lr.SKIPPED_PROBLEM)

    def test_a_setup_failure_is_neither_unreadable_nor_a_skipped_item_and_keeps_its_text(self):
        # What the review reports for a reviewer whose model could not be loaded (no request was ever sent).
        line = "reviewer 'general': could not load model 'm': HTTP 500"
        self.assertEqual(lr.classify_problem(line), (lr.OTHER_PROBLEM, line))


def _result(sent: int, failed: int = 0, unreadable: int = 0, name: str = "r") -> "lr.RouteResult":
    result = lr.RouteResult(lr.Route(name, "lines"), sent=sent, failed=failed)
    if unreadable:
        result.unreadable["no FINDING blocks"] = unreadable
    return result


class Verdict(unittest.TestCase):
    def test_nineteen_of_twenty_is_reliable_and_eighteen_of_twenty_is_not(self):
        self.assertTrue(_result(20, unreadable=1).reliable())
        self.assertFalse(_result(20, unreadable=2).reliable())

    def test_the_boundary_is_exact_at_other_sizes(self):
        self.assertTrue(_result(100, unreadable=5).reliable())
        self.assertFalse(_result(100, unreadable=6).reliable())
        self.assertTrue(_result(1).reliable())
        self.assertFalse(_result(1, unreadable=1).reliable())

    def test_failed_requests_are_left_out_of_the_denominator(self):
        # 20 sent, 10 got no reply, 1 of the 10 replies unreadable: 9/10 = 90%, not 19/20 and not 9/20.
        result = _result(20, failed=10, unreadable=1)
        self.assertEqual((result.received, result.readable), (10, 9))
        self.assertFalse(result.reliable())
        # The same 10 replies all readable is 100% even though half the requests failed.
        self.assertTrue(_result(20, failed=10).reliable())

    def test_a_route_that_got_no_reply_is_no_answer_not_reliable(self):
        self.assertIsNone(_result(5, failed=5).reliable())
        self.assertIsNone(_result(0).reliable())
        self.assertIn("no reply came back to judge", lr.verdict([_result(5, failed=5, name="lines")]))

    def test_the_first_reliable_route_wins_in_order(self):
        results = [_result(20, unreadable=10, name="lines"), _result(20, name="json"), _result(20, name="native")]
        self.assertEqual(lr.verdict(results), "json is reliable (100.0%)")

    def test_none_reliable_says_so(self):
        text = lr.verdict([_result(20, unreadable=10, name="lines"), _result(20, unreadable=20, name="json")])
        self.assertIn("no route was reliable", text)
        self.assertIn("95%", text)


def _stub(script: list):
    """A run function that sends and fails the requests, and reports the problems, `script` gives for each call."""
    calls = iter(script)

    def run(_repo, _base, _head, reviewers, _args, recorder):
        sent, failed, problems, claims = next(calls)
        for _ in range(sent):
            recorder.log("req", "stub")
        for _ in range(failed):
            recorder.log("fail", "stub")
        return claims, SimpleNamespace(errors=list(problems), reviewers=reviewers)

    return run


def _args(**change):
    values = dict(resident=False, context=8192)
    values.update(change)
    return SimpleNamespace(**values)


class Measuring(unittest.TestCase):
    def setUp(self):
        self.case = ALL_CASES["seeded-three"]
        self.repos = {self.case.name: (Path("."), "main", "feature")}
        self.format_problem = _problem_from_reader("prose", "lines")

    def test_replies_are_counted_per_route_over_every_run(self):
        script = [(4, 0, [], []), (4, 0, [self.format_problem], []), (4, 1, [], [])]
        result = lr.measure_route(lr.Route("lines", "lines"), "m", [self.case], 3, _args(), h.Recorder(), self.repos, _stub(script))
        self.assertEqual((result.sent, result.failed), (12, 1))
        self.assertEqual(result.unreadable, {"no FINDING blocks": 1})
        self.assertEqual((result.received, result.readable), (11, 10))
        self.assertEqual(result.runs, 3)

    def test_a_found_bug_and_a_false_alarm_are_counted_but_never_decide_reliability(self):
        bug = self.case.bugs[0]
        hit = h.Claim(bug.file, h.bug_line(self.case, bug), h.bug_line(self.case, bug), "c", "claim", ("r",))
        far = h.Claim(bug.file, 1, 1, "c", "elsewhere", ("r",))
        result = lr.measure_route(lr.Route("lines", "lines"), "m", [self.case], 1, _args(), h.Recorder(), self.repos,
                                  _stub([(2, 0, [], [hit, far])]))
        self.assertEqual((result.found, result.possible, result.false_alarms), (1, 3, 1))
        self.assertTrue(result.reliable())

    def test_a_setup_failure_is_kept_with_its_cause_and_not_counted_as_a_skipped_item(self):
        failure = "reviewer 'general': could not load model 'm': HTTP 500"
        skipped = parse_findings("FINDING\nfile: a.py\nlines: 1-1\nEND", "r", "", "lines")[1][0]
        result = lr.measure_route(lr.Route("lines", "lines"), "m", [self.case], 2, _args(), h.Recorder(), self.repos,
                                  _stub([(0, 0, [failure], []), (2, 0, [skipped], [])]))
        self.assertEqual((result.skipped_items, dict(result.other_problems)), (1, {failure: 1}))
        self.assertEqual(result.unreadable_count, 0)
        text = lr.format_model("m", [self.case], 2, [result])
        self.assertIn("1 x " + failure, text)
        self.assertIn("1 item(s) skipped from replies that were read", text)
        self.assertNotIn("other problem(s) in replies that were read", text)

    def test_the_reviewer_carries_the_routes_format_prompt_file_and_translator(self):
        seen = []

        def run(_repo, _base, _head, reviewers, _args, _recorder):
            seen.extend(reviewers)
            return [], SimpleNamespace(errors=[])

        with tempfile.TemporaryDirectory() as tmp:
            prompt, translator = Path(tmp, "p.md"), Path(tmp, "t.py")
            prompt.write_text("p", encoding="utf-8")
            translator.write_text("x", encoding="utf-8")
            route = lr.Route("native", "json", str(prompt), str(translator))
            lr.measure_route(route, "model-x", [self.case], 1, _args(resident=True), h.Recorder(), self.repos, run)
        entry = seen[0]
        self.assertEqual((entry["model"], entry["replyFormat"], entry["loadOnDemand"]), ("model-x", "json", False))
        self.assertEqual((entry["promptFile"], entry["translator"]), (str(prompt), str(translator)))
        self.assertNotIn("prompt", entry)

    def test_a_built_in_route_sends_no_prompt_file_and_no_translator(self):
        seen = []

        def run(_repo, _base, _head, reviewers, _args, _recorder):
            seen.extend(reviewers)
            return [], SimpleNamespace(errors=[])

        lr.measure_route(lr.Route("lines", "lines"), "m", [self.case], 1, _args(), h.Recorder(), self.repos, run)
        self.assertNotIn("promptFile", seen[0])
        self.assertNotIn("translator", seen[0])
        self.assertEqual(seen[0]["prompt"], h.DEFAULT_LENS)


class Output(unittest.TestCase):
    def test_a_models_table_shows_each_route_the_reasons_and_the_verdict(self):
        case = ALL_CASES["seeded-three"]
        lines = _result(80, unreadable=2, name="lines")
        lines.runs, lines.possible, lines.found = 20, 60, 20
        json_route = _result(80, failed=4, name="json")
        json_route.runs, json_route.possible = 20, 60
        text = lr.format_model("some-model", [case], 10, [lines, json_route])
        self.assertIn("some-model   1 case(s) x 10 run(s): seeded-three", text)
        self.assertIn("78/80  97.5%", text)
        self.assertIn("2 no FINDING blocks", text)
        self.assertIn("76/76 100.0%", text)
        self.assertIn("verdict: lines is reliable (97.5%)", text)
        self.assertIn("json: 4 request(s) got no reply from LM Studio", text)
        self.assertIn("for information only (does not decide the verdict)", text)
        self.assertIn("lines found 2/6 bug(s)", text)

    def test_show_config_lists_models_routes_cases_and_the_least_requests(self):
        text = lr.describe_run(["a", "b"], list(lr.BUILTIN_ROUTES), ["seeded-three", "clean"], 10)
        self.assertIn("models: a, b", text)
        self.assertIn("lines", text)
        self.assertIn("json", text)
        self.assertIn("requests: at least 80", text)  # 2 models x 2 routes x 2 cases x 10 runs
        self.assertIn("95%", text)

    def test_show_config_runs_with_no_lm_studio(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = lr.main(["--model", "m", "--runs", "3", "--show-config"])
        self.assertEqual(code, 0)
        self.assertIn("seeded-three, clean", out.getvalue())
        self.assertIn("at least 12", out.getvalue())

    def test_a_case_named_twice_is_listed_and_run_once(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lr.main(["--model", "m", "--case", "clean", "--case", "clean", "--runs", "1", "--show-config"])
        text = out.getvalue()
        self.assertIn("cases: clean   runs: 1", text)
        self.assertIn("requests: at least 2", text)  # 1 model x 2 routes x 1 case x 1 run

    def test_a_bad_route_stops_the_run_before_anything_else(self):
        with self.assertRaises(SystemExit) as caught:
            lr.main(["--model", "m", "--route", "x=xml", "--show-config"])
        self.assertIn("the format must be one of lines, json", str(caught.exception))


class RequestFailuresAreLogged(unittest.TestCase):
    def test_a_request_with_no_reply_is_logged_as_a_failure_as_well_as_a_request(self):
        recorder = h.Recorder()
        replies = iter(["ok", None])
        call = h.timed_complete(recorder, lambda *_args: next(replies))
        call(None, "m", "system", "content")
        call(None, "m", "system", "content")
        self.assertEqual((recorder.count("req"), recorder.count("fail")), (2, 1))


if __name__ == "__main__":
    unittest.main()
