#!/usr/bin/env python3
"""Tests for harness/live_classifier.py, `harness_lib.run_classifier` and the classifier step of
harness/live_all_reviewers.py.

Stdlib-only (unittest), no network, no LM Studio: the model's probability of yes is a stub and LM Studio's REST API is
`tests/fake_lmstudio.py`. Every database, metrics store and log folder is a temporary one.

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
for _directory in (REPO_ROOT / "harness", REPO_ROOT / "libs", REPO_ROOT / "tests"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import harness_lib as h  # noqa: E402
import live_all_reviewers  # noqa: E402
import live_classifier as live  # noqa: E402
import openai_lib  # noqa: E402
from cases import RUBRIC, classifier_cases  # noqa: E402
from classifier_reviewer_lib import Classification, Classified, ClassificationReview, EntryScore, Rating  # noqa: E402
from fake_lmstudio import FakeLmStudio  # noqa: E402
from reasoning_reviewer_lib import Finding  # noqa: E402

BASE = "http://localhost:1234/v1"


def entry_in(content: str) -> str:
    """The rubric entry a classifier request is about: the name on its `Category:` line."""
    return re.search(r"^Category: (\S+) \(level", content, re.MULTILINE).group(1)


def probability_for(content: str, **by_entry) -> float:
    """What a stub answers for the request in `content`: the probability of yes for the entry it is about (keys are the entry
    names with `_` for `-`), 0.05 for any entry not given."""
    return by_entry.get(entry_in(content).replace("-", "_"), 0.05)


def perfect_probability(content: str) -> float:
    """What a classifier that gets every fixture right says for the request in `content`: yes (0.95) for the entry the finding
    belongs to, and for both entries of the one that fits two; no (0.05) for the others and for a finding that fits nothing."""
    for case in classifier_cases():
        if f"Claim: {case.claim}\n" in content + "\n":
            fits = {"injection-or-traversal", "crash-on-input"} if case.overlap else ({case.expected} if case.expected else set())
            return 0.95 if entry_in(content) in fits else 0.05
    raise AssertionError(f"no fixture for {content!r}")


N = len(classifier_cases())


class Isolated(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name).resolve()
        self.db_path = self.dir / "r.db"
        patcher = mock.patch.dict(os.environ, {
            "CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB": str(self.db_path),
            "CLAUDE_RUNWAY_METRICS_DB": str(self.dir / "metrics.db"),
            "CLAUDE_RUNWAY_SAVINGS_DB": str(self.dir / "savings.db"),
            "CLAUDE_RUNWAY_TRACK_SAVINGS": "",
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def config(self, *, stages=None, reviewers=None, rubric=True) -> str:
        reviewers = reviewers if reviewers is not None else [
            {"name": "sorter", "model": "model-s", "prompt": "sort it", "active": True, "type": ["classifier"], "loadOnDemand": False}]
        data: dict = {"reviewers": reviewers}
        if rubric:
            data["rubric"] = [dict(entry) for entry in RUBRIC]
        if stages is not None:
            data["stages"] = stages
        path = self.dir / "cfg.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    @staticmethod
    def run_main(*argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = live.main(list(argv))
        return code, out.getvalue()

    def lmstudio(self, probability, complete=None):
        """Patches the things a run touches: the server module, the REST API, the probability of yes (and, when given, the
        completion the earlier stages use)."""
        server = SimpleNamespace(lmstudio_base_url=lambda: BASE)
        fake = FakeLmStudio(loaded=["model-s", "model-t"])
        for patcher in (
            mock.patch.object(h, "load_server", return_value=server),
            mock.patch("lmstudio_client.build_client", return_value=object()),
            mock.patch("lmstudio_model_loader_lib._urllib_http", fake),
            mock.patch.object(openai_lib, "yes_probability", probability),
            mock.patch.object(openai_lib, "complete", complete or mock.Mock(side_effect=AssertionError("not a classifier call"))),
            mock.patch.object(h, "preflight_reviewers"),
            mock.patch.object(h, "loaded_models", return_value=["model-s"]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)


class CheckAndConfig(Isolated):
    def test_check_validates_the_fixtures_and_needs_no_lm_studio(self):
        with mock.patch.object(h, "preflight_reviewers", side_effect=AssertionError("must not touch LM Studio")):
            code, out = self.run_main("--check")
        self.assertEqual(code, 0)
        self.assertIn(f"{N} classifier fixture(s)", out)
        self.assertIn("expect None", out)

    def test_show_config_resolves_the_classifiers_and_the_rubric_without_lm_studio(self):
        with mock.patch.object(h, "preflight_reviewers", side_effect=AssertionError("must not touch LM Studio")):
            code, out = self.run_main("--config", self.config(), "--show-config")
        self.assertEqual(code, 0)
        self.assertIn("sorter", out)
        self.assertIn("roles=classifier", out)
        self.assertIn("model-s", out)
        self.assertIn("rubric: injection-or-traversal (high, score 0.5)", out)

    def test_a_config_whose_stages_leave_out_the_classifier_exits_0_without_loading_a_model(self):
        path = self.config(stages=["reasoning"], reviewers=[
            {"name": "r", "model": "model-r", "prompt": "look", "active": True, "type": ["reasoning"], "loadOnDemand": True}], rubric=False)
        with mock.patch.object(h, "preflight_reviewers", side_effect=AssertionError("loaded a model")), \
                mock.patch.object(h, "load_server", side_effect=AssertionError("loaded a model")):
            code, out = self.run_main("--config", path)
        self.assertEqual(code, 0)
        self.assertIn("classifier is not one of them", out)

    def test_a_config_that_selects_the_classifier_with_no_active_classifier_reviewer_is_refused_naming_the_problem(self):
        path = self.config(stages=["classifier"], reviewers=[
            {"name": "sorter", "model": "model-s", "prompt": "sort it", "active": False, "type": ["classifier"], "loadOnDemand": False}])
        with self.assertRaises(SystemExit) as caught:
            self.run_main("--config", path)
        self.assertIn("classifier", str(caught.exception))

    def test_an_active_classifier_with_no_rubric_is_refused_by_the_validator(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_main("--config", self.config(rubric=False), "--show-config")
        self.assertIn("rubric", str(caught.exception))

    def test_a_flag_next_to_a_config_that_names_reviewers_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_main("--config", self.config(), "--model", "other")
        self.assertIn("--model", str(caught.exception))

    def test_the_rubric_is_the_configs_when_it_has_one_and_the_fixtures_otherwise(self):
        custom = {"name": "only", "level": "low", "definition": "d", "score": 0.4}
        path = self.dir / "cfg.json"
        path.write_text(json.dumps({"rubric": [custom], "reviewers": [
            {"name": "sorter", "model": "model-s", "prompt": "sort", "active": True, "type": ["classifier"], "loadOnDemand": False}]}),
            encoding="utf-8")
        self.assertEqual([e["name"] for e in live.rubric_for(live.parse_args(["--config", str(path)]))], ["only"])
        self.assertEqual([e["name"] for e in live.rubric_for(live.parse_args([]))], [e["name"] for e in RUBRIC])

    def test_with_no_config_one_classifier_is_built_from_the_flags_with_the_default_lens(self):
        args = live.parse_args(["--model", "model-s"])
        (entry,) = h.stage_reviewers(args, "classifier")
        self.assertEqual((entry["model"], entry["type"]), ("model-s", ["classifier"]))
        self.assertEqual(entry["prompt"], h.DEFAULT_CLASSIFIER_LENS)


class Scoring(unittest.TestCase):
    @staticmethod
    def classified(case, entry, mean=0.95, means=None):
        """The fixture as the stage classified it: `entry` taken at `mean`, every other entry averaging 0.05 (or what `means` says)."""
        values = {e["name"]: 0.05 for e in RUBRIC}
        if entry:
            values[entry] = float(mean)
        values.update(means or {})
        scores = tuple(EntryScore(e["name"], values[e["name"]], values[e["name"]] >= e["score"]) for e in RUBRIC)
        answers = (Classification("c", tuple(Rating(name, value) for name, value in values.items())),)
        finding = live.fixture_finding(case)
        return Classified(finding, "classified" if entry else "unclassified", "", entry, "high" if entry else None,
                          float(mean) if entry else None, scores=scores, classifications=answers)

    def review_of(self, picks):
        cases = classifier_cases()
        return cases, ClassificationReview(classified=[self.classified(c, picks.get(c.id, c.expected)) for c in cases])

    def text(self, review, cases):
        return live.score_choices(live.choices_from(review, cases), live.entry_averages(review), live.rubric_for(live.parse_args([])))

    def test_choices_follow_the_fixtures_in_order(self):
        cases, review = self.review_of({})
        choices = live.choices_from(review, cases)
        self.assertEqual({k: v[0] for k, v in choices.items()}, {c.id: c.expected for c in cases})

    def test_an_unclassified_finding_and_an_unanswered_one_come_back_as_no_entry(self):
        cases = classifier_cases()[:2]
        review = ClassificationReview(classified=[
            Classified(live.fixture_finding(cases[0]), "unclassified"),
            Classified(live.fixture_finding(cases[1]), "unchecked", note="x")])
        self.assertEqual(set(live.choices_from(review, cases).values()), {(None, None, "unclassified"), (None, None, "unchecked")})

    def test_every_fixture_right_reads_as_such(self):
        cases, review = self.review_of({})
        text = self.text(review, cases)
        self.assertIn(f"on the expected entry: {N}/{N}", text)
        self.assertIn("fits no entry, came back unclassified: 3/3", text)
        self.assertIn("fits two entries, took the higher level (injection-or-traversal): 1/1", text)

    def test_a_wrong_entry_the_missed_drop_and_the_lower_level_overlap_are_named(self):
        cases, review = self.review_of({"none/readme": "style-only", "overlap/path-traversal-crash": "crash-on-input",
                                        "style/short-name": "wrong-result"})
        text = self.text(review, cases)
        self.assertIn(f"on the expected entry: {N - 3}/{N}", text)
        self.assertIn("none/readme: expected None, got style-only", text)
        self.assertIn("style/short-name: expected style-only, got wrong-result", text)
        self.assertIn("fits no entry, came back unclassified: 2/3", text)
        self.assertIn("fits two entries, took the higher level (injection-or-traversal): 0/1", text)

    def test_the_average_for_each_entry_is_shown_against_its_score(self):
        cases = classifier_cases()
        review = ClassificationReview(classified=[
            self.classified(cases[0], "injection-or-traversal", 0.9),
            self.classified(cases[1], None, means={"injection-or-traversal": 0.3, "wrong-result": 0.4})])
        text = live.score_choices({}, live.entry_averages(review), live.rubric_for(live.parse_args([])))
        self.assertRegex(text, r"injection-or-traversal\s+score 0.5: averages \[0.3, 0.9\], 1 at or above the score, 1 below")
        self.assertRegex(text, r"wrong-result\s+score 0.5: averages \[0.05, 0.4\], 0 at or above the score, 2 below")
        self.assertNotIn("the scores dropped nothing", text)

    def test_an_average_the_library_rounds_up_to_its_score_is_counted_as_reaching_it(self):
        # 0.5999996 is 0.6 to the six places the classifier compares at, so the finding was classified and the count must say so
        rubric = [{"name": "wrong-result", "score": 0.6}]
        text = live.score_choices({}, {"wrong-result": [0.5999996, 0.5999]}, rubric)
        self.assertRegex(text, r"wrong-result\s+score 0.6: averages \[0.6, 0.6\], 1 at or above the score, 1 below")

    def test_float_noise_below_the_score_does_not_make_the_scores_drop_something(self):
        rubric = [{"name": "wrong-result", "score": 0.3}]
        text = live.score_choices({}, {"wrong-result": [0.7 - 0.4]}, rubric)  # 0.3 minus a hair
        self.assertIn("the scores dropped nothing", text)

    def test_scores_that_dropped_nothing_are_called_out(self):
        cases = classifier_cases()
        every = {e["name"]: 0.95 for e in RUBRIC}
        review = ClassificationReview(classified=[self.classified(c, c.expected or "style-only", 0.95, means=every) for c in cases])
        self.assertIn("no average fell below its entry's score: the scores dropped nothing", self.text(review, cases))

    def test_no_answers_at_all_say_so(self):
        cases = classifier_cases()
        review = ClassificationReview(classified=[Classified(live.fixture_finding(c), "unchecked") for c in cases])
        text = self.text(review, cases)
        self.assertIn("no probabilities returned", text)
        # A finding nobody answered is not a correct drop, even one that fits nothing.
        self.assertIn(f"on the expected entry: 0/{N}", text)
        self.assertIn("fits no entry, came back unclassified: 0/3", text)
        self.assertIn("none/readme: expected None, got not answered", text)

    def test_a_finding_that_fits_nothing_is_right_only_when_it_came_back_unclassified(self):
        cases = classifier_cases()
        nothing = next(c for c in cases if c.expected is None)
        for status, credited in (("unclassified", 1), ("unchecked", 0)):
            with self.subTest(status=status):
                choices = {c.id: (c.expected, 0.9, "classified") for c in cases if c.expected is not None}
                choices[nothing.id] = (None, None, status)
                text = live.score_choices(choices, {}, live.rubric_for(live.parse_args([])))
                self.assertIn(f"fits no entry, came back unclassified: {credited}/3", text)

    def test_a_rubric_without_an_entry_the_fixtures_expect_leaves_those_fixtures_out_of_the_score(self):
        cases = classifier_cases()
        rubric = [dict(e) for e in RUBRIC if e["name"] != "wrong-result"]
        choices = {c.id: (c.expected, 0.9, "classified") if c.expected else (None, None, "unclassified") for c in cases}
        expecting = [c for c in cases if c.expected == "wrong-result"]
        for c in expecting:
            choices[c.id] = (None, None, "unclassified")  # the model cannot name an entry that is not in the rubric
        text = live.score_choices(choices, {}, rubric)
        self.assertIn(f"on the expected entry: {N - len(expecting)}/{N - len(expecting)}", text)
        self.assertIn(f"left out of the score: {len(expecting)} fixture(s) expect an entry the rubric does not have (wrong-result)", text)
        self.assertNotIn("expected wrong-result", text)

    def test_the_shipped_example_config_has_every_entry_the_fixtures_expect(self):
        example = json.loads((REPO_ROOT / "harness" / "config.example.json").read_text(encoding="utf-8"))
        names = {entry["name"] for entry in example["rubric"]}
        self.assertLessEqual({c.expected for c in classifier_cases() if c.expected}, names)

    def test_a_fixture_is_shown_as_a_finding_with_one_class_and_no_code(self):
        case = classifier_cases()[0]
        finding = live.fixture_finding(case)
        self.assertEqual((finding.cls, finding.claim, finding.failing_input, finding.quote), ("uncategorized", case.claim, case.failing_input, ""))


class Repeat(unittest.TestCase):
    def test_averages_that_came_out_the_same_in_every_run_repeat(self):
        runs = [{"f": {"wrong-result": 0.8123}}, {"f": {"wrong-result": 0.8123}}, {"f": {"wrong-result": 0.9}}]
        head, *differ = live.repeat_lines(runs)
        self.assertEqual(differ, [])
        self.assertIn("0 differ of 1 averages compared (tolerance 0.1)", head)

    def test_averages_that_differ_are_reported_by_fixture_and_entry(self):
        runs = [{"f": {"wrong-result": 0.80, "style-only": 0.1}}, {"f": {"wrong-result": 0.95, "style-only": 0.1}}]
        head, *differ = live.repeat_lines(runs)
        self.assertIn("1 differ of 2 averages compared", head)
        self.assertEqual(differ, ["  differs: f / wrong-result: 0.8000, 0.9500"])

    def test_one_run_has_nothing_to_compare(self):
        self.assertEqual(live.repeat_lines([{"f": {"x": 0.1}}]), [])


class YesNoProbe(unittest.TestCase):
    """`probe_answers_yes_no` and its place in `preflight_reviewers`: a classifier model that cannot answer yes or no is
    refused before the run loads anything else, with the reason."""

    def probe(self, value, failure=(None, None)):
        loader = mock.MagicMock()
        loader.release_all.return_value = []
        with mock.patch("lmstudio_model_loader_lib.ModelLoader", return_value=loader), \
                mock.patch("lmstudio_client.build_client", return_value=object()), \
                mock.patch.object(h, "load_server", return_value=SimpleNamespace(lmstudio_base_url=lambda: BASE)), \
                mock.patch.object(openai_lib, "yes_probability", return_value=value), \
                mock.patch.object(openai_lib, "take_failure", return_value=failure):
            return h.probe_answers_yes_no("m", on_demand=True, context=8192), loader

    def test_a_model_that_answers_passes_and_is_unloaded_again(self):
        problem, loader = self.probe(0.97)
        self.assertIsNone(problem)
        loader.release_all.assert_called_once_with()

    def test_a_model_whose_reply_does_not_begin_with_yes_or_no_is_refused_with_what_it_began_with(self):
        problem, loader = self.probe(None, ("no_answer", "'' 56%, 'The' 4%"))
        self.assertIn("cannot be a classifier", problem)
        self.assertIn("'' 56%, 'The' 4%", problem)
        self.assertIn("chat template", problem)
        loader.release_all.assert_called_once_with()

    def test_a_model_with_no_token_probabilities_is_refused(self):
        problem, _ = self.probe(None, ("no_probabilities", None))
        self.assertIn("no token probabilities", problem)

    def test_any_other_failure_is_reported_as_one(self):
        problem, _ = self.probe(None, ("timeout", None))
        self.assertIn("timeout", problem)

    def entries(self, *types):
        return [{"name": "r", "model": "m", "type": list(types)}]

    def preflight(self, entries, **args):
        namespace = SimpleNamespace(context=8192, **args)
        with mock.patch.object(h, "preflight_models"), mock.patch.object(h, "require_chat_model"), \
                mock.patch.object(h, "probe_answers_yes_no", return_value=args.get("problem")) as probe:
            h.preflight_reviewers(entries, namespace)
        return probe

    def test_only_a_model_with_a_classifier_reviewer_is_asked(self):
        self.preflight(self.entries("reasoning")).assert_not_called()
        self.preflight(self.entries("verifier")).assert_not_called()
        self.preflight(self.entries("reasoning", "classifier")).assert_called_once()

    def test_no_probe_skips_it(self):
        self.preflight(self.entries("classifier"), no_probe=True).assert_not_called()

    def test_a_problem_stops_the_run_with_its_message(self):
        with self.assertRaises(SystemExit) as raised:
            self.preflight(self.entries("classifier"), problem="m cannot be a classifier")
        self.assertEqual(str(raised.exception), "m cannot be a classifier")


class RunsAgainstTheStage(Isolated):
    def test_a_run_scores_every_fixture_and_records_it_in_the_environments_database(self):
        seen = []

        def probability(client, model, system, content):
            seen.append((model, system, content))
            return perfect_probability(content)

        self.lmstudio(probability)
        code, out = self.run_main("--config", self.config(), "--runs", "1")
        self.assertEqual(code, 0)
        self.assertIn(f"on the expected entry: {N}/{N}", out)
        self.assertIn("fits no entry, came back unclassified: 3/3", out)
        self.assertIn("fits two entries, took the higher level (injection-or-traversal): 1/1", out)
        self.assertEqual(len(seen), N * len(RUBRIC))  # one request per fixture per rubric entry
        self.assertTrue(all(model == "model-s" for model, _, _ in seen))
        self.assertIn("Category: injection-or-traversal (level: high)\nDefinition: Untrusted input reaches a shell", seen[0][2])
        self.assertNotIn("score", seen[0][1])
        self.assertNotIn("score", seen[0][2])
        # The rubric text and the findings reached the model; the fixtures' expected answers did not.
        self.assertNotIn("expect", seen[0][2])
        conn = sqlite3.connect(str(self.db_path))
        try:
            self.assertEqual(conn.execute("SELECT review_type, source, outcome FROM runs").fetchall(), [("classifier", "harness", "finished")])
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0], N)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM requests WHERE role = 'classifier'").fetchone()[0], N * len(RUBRIC))
            self.assertEqual(conn.execute("SELECT status, level, confidence FROM classifications WHERE finding_no = 12").fetchone(),
                             ("classified", "high", 0.95))
        finally:
            conn.close()

    def test_a_model_that_says_yes_to_every_entry_is_reported_as_dropping_nothing(self):
        self.lmstudio(lambda client, model, system, content: 0.95)
        _, out = self.run_main("--config", self.config())
        self.assertIn("the scores dropped nothing", out)

    def test_a_classifier_that_returns_no_probability_is_reported_not_scored_as_right(self):
        self.lmstudio(lambda client, model, system, content: None)
        code, out = self.run_main("--config", self.config())
        self.assertEqual(code, 0)
        self.assertIn("classifier problem:", out)
        self.assertIn("not answered:", out)
        self.assertIn("no probabilities returned", out)

    def test_several_classifiers_are_averaged_into_one_probability_per_entry(self):
        reviewers = [
            {"name": "s1", "model": "model-s", "prompt": "sort", "active": True, "type": ["classifier"], "loadOnDemand": False},
            {"name": "s2", "model": "model-t", "prompt": "sort", "active": True, "type": ["classifier"], "loadOnDemand": False}]

        def probability(client, model, system, content):
            if model == "model-t":
                return probability_for(content, wrong_result=0.95)
            return perfect_probability(content)

        self.lmstudio(probability)
        _, out = self.run_main("--config", self.config(reviewers=reviewers))
        self.assertNotIn("disputed", out)
        self.assertIn("average probability per entry against its score:", out)

    def test_the_probability_requests_go_through_the_openai_compatible_url(self):
        # The native /api/v0 that the reviewers' client uses returns no token probabilities.
        self.lmstudio(lambda client, model, system, content: perfect_probability(content))
        with mock.patch("lmstudio_client.build_client", return_value=object()) as build:
            self.run_main("--config", self.config())
        urls = [call.args[0] for call in build.call_args_list]
        self.assertIn(BASE, urls)

    def test_the_sampling_flags_are_gone(self):
        for flag in (("--samples", "3"), ("--temperature", "0.4")):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                live.parse_args(["--model", "model-s", *flag])

    def test_a_classifier_built_from_the_flags_has_no_sampling_fields(self):
        (entry,) = h.stage_reviewers(live.parse_args(["--model", "model-s"]), "classifier")
        self.assertNotIn("samples", entry)
        self.assertNotIn("temperature", entry)

    def test_two_runs_report_that_the_averages_repeated(self):
        self.lmstudio(lambda client, model, system, content: perfect_probability(content))
        _, out = self.run_main("--config", self.config(), "--runs", "2")
        self.assertRegex(out, r"score repeat over 2 runs: 0 differ of \d+ averages compared")

    def test_runs_repeat_the_stage(self):
        self.lmstudio(lambda client, model, system, content: perfect_probability(content))
        _, out = self.run_main("--config", self.config(), "--runs", "2")
        self.assertEqual(out.count("--- run "), 2)


class AllReviewersRunsTheClassifier(Isolated):
    """`live_all_reviewers.py` runs the classifier after the verifier, over what the verifiers left standing."""

    def test_the_classifier_step_runs_after_the_verifier_and_shows_each_finding_with_its_level(self):
        source = "import os\nos.system(cmd)\n"
        finding_json = json.dumps({"findings": [{"file": "a.py", "line_start": 2, "line_end": 2, "class": "injection",
                                                 "claim": "shell injection", "failing_input": "cmd='a; b'", "quote": "os.system(cmd)"}]})
        verdict = json.dumps({"verdict": "confirmed", "failing_input": "cmd='a; b'", "wrong_result": "runs b", "reasoning": "traced"})
        sorted_calls = []

        def complete(client, model, system, content, **options):
            if "checking ONE reported defect" in system:
                return verdict
            return finding_json

        def probability(client, model, system, content):
            sorted_calls.append(content)
            return probability_for(content, injection_or_traversal=0.95)

        reviewers = [
            {"name": "finder", "model": "model-s", "prompt": "look", "active": True, "type": ["reasoning"], "loadOnDemand": False, "replyFormat": "json"},
            {"name": "checker", "model": "model-s", "prompt": "check", "active": True, "type": ["verifier"], "loadOnDemand": False},
            {"name": "sorter", "model": "model-s", "prompt": "sort", "active": True, "type": ["classifier"], "loadOnDemand": False}]
        path = self.config(reviewers=reviewers)
        repo = self.dir / "repo"
        repo.mkdir()
        server = SimpleNamespace(lmstudio_base_url=lambda: BASE, _git_diff=lambda *args: (
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -0,0 +1,2 @@\n+import os\n+os.system(cmd)\n"))
        fake = FakeLmStudio(loaded=["model-s"])
        out = io.StringIO()
        patches = (
            mock.patch.object(h, "load_server", return_value=server),
            mock.patch("lmstudio_client.build_client", return_value=object()),
            mock.patch("lmstudio_model_loader_lib._urllib_http", fake),
            mock.patch.object(openai_lib, "complete", complete),
            mock.patch.object(openai_lib, "yes_probability", probability),
            mock.patch.object(h, "preflight_reviewers"),
            mock.patch.object(h, "loaded_models", return_value=["model-s"]),
            mock.patch.object(h, "build_repo", return_value=(repo, "main", "HEAD")),
            mock.patch("verifier_reviewer_lib.git_file_reader", return_value=lambda p: source),
        )
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            with contextlib.redirect_stdout(out):
                code = live_all_reviewers.main(["--config", path])
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("--- run 1 classified: 1 classified (1 high, 0 moderate, 0 low), 0 unclassified, 0 without a rubric level", text)
        self.assertIn("a.py:2 [injection] shell injection -> injection-or-traversal [high], average probability 0.95", text)
        self.assertNotIn("not built", text)
        self.assertEqual(len(sorted_calls), len(RUBRIC))  # the one finding, once per rubric entry
        conn = sqlite3.connect(str(self.db_path))
        try:  # one run, all three stages
            self.assertEqual(conn.execute("SELECT review_type, combination FROM runs").fetchall(),
                             [("classifier", "reasoning:model-s|verifier:model-s|classifier:model-s")])
            self.assertEqual(conn.execute("SELECT verification, status FROM classifications").fetchall(), [("confirmed", "classified")])
        finally:
            conn.close()


    def test_without_a_verifier_every_claim_of_a_merged_finding_is_classified(self):
        # Two different claims about the same line are merged by the reasoning stage into one `Finding` whose second claim is an
        # `alternate`. The verifier stage normally flattens those; with no verifier the harness has to, or the second claim is
        # never classified and never recorded.
        finding_json = json.dumps({"findings": [
            {"file": "a.py", "line_start": 2, "line_end": 2, "class": "injection", "claim": "shell injection",
             "failing_input": "cmd='a; b'", "quote": "os.system(cmd)"},
            {"file": "a.py", "line_start": 2, "line_end": 2, "class": "injection", "claim": "the return code is ignored",
             "failing_input": "cmd='false'", "quote": "os.system(cmd)"}]})
        sorted_calls = []

        def complete(client, model, system, content, **options):
            return finding_json

        def probability(client, model, system, content):
            sorted_calls.append(content)
            return probability_for(content, injection_or_traversal=0.95)

        reviewers = [
            {"name": "finder", "model": "model-s", "prompt": "look", "active": True, "type": ["reasoning"], "loadOnDemand": False, "replyFormat": "json"},
            {"name": "sorter", "model": "model-s", "prompt": "sort", "active": True, "type": ["classifier"], "loadOnDemand": False}]
        repo = self.dir / "repo"
        repo.mkdir()
        server = SimpleNamespace(lmstudio_base_url=lambda: BASE, _git_diff=lambda *args: (
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -0,0 +1,2 @@\n+import os\n+os.system(cmd)\n"))
        out = io.StringIO()
        patches = (
            mock.patch.object(h, "load_server", return_value=server),
            mock.patch("lmstudio_client.build_client", return_value=object()),
            mock.patch("lmstudio_model_loader_lib._urllib_http", FakeLmStudio(loaded=["model-s"])),
            mock.patch.object(openai_lib, "complete", complete),
            mock.patch.object(openai_lib, "yes_probability", probability),
            mock.patch.object(h, "preflight_reviewers"),
            mock.patch.object(h, "loaded_models", return_value=["model-s"]),
            mock.patch.object(h, "build_repo", return_value=(repo, "main", "HEAD")),
        )
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            with contextlib.redirect_stdout(out):
                code = live_all_reviewers.main(["--config", self.config(reviewers=reviewers)])
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("--- run 1 classified: 2 classified", text)
        self.assertEqual(len(sorted_calls), 2 * len(RUBRIC))  # both claims, once per rubric entry
        self.assertIn("Claim: the return code is ignored", "\n".join(sorted_calls))


class FindingShape(unittest.TestCase):
    def test_the_fixture_finding_is_a_finding(self):
        self.assertIsInstance(live.fixture_finding(classifier_cases()[0]), Finding)


if __name__ == "__main__":
    unittest.main()
