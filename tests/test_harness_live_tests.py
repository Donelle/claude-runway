#!/usr/bin/env python3
"""Tests for harness/live-tests: a config in `configs/` is a live test.

Covers `live_configs.py` (finding the configs, choosing the script from `stages`, reading the baselines
sidecar), the runner's choosing and checking (`run_live_tests.py`, fed canned script output so no model
runs), and the pre-flight's per-config checks (`preflight/run_checks.py`).

Stdlib-only (unittest, no pytest), no network, no LM Studio.

    .venv/bin/python -m unittest discover -s tests
"""

import argparse
import contextlib
import functools
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_TESTS = REPO_ROOT / "harness" / "live-tests"
for _directory in (LIVE_TESTS, LIVE_TESTS / "preflight", REPO_ROOT / "libs", REPO_ROOT / "harness"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import harness_lib as h  # noqa: E402
import live_all_reviewers  # noqa: E402
import live_classifier  # noqa: E402
import live_configs as lc  # noqa: E402
import live_verifier  # noqa: E402
import run_checks  # noqa: E402
import run_live_tests as rlt  # noqa: E402
from cases import ALL_CASES, RUBRIC, classifier_cases, fixture_cases  # noqa: E402
from classifier_reviewer_lib import Classification, Classified, ClassificationReview, Rating  # noqa: E402
from codebase_reviewer_config_lib import Reviewer  # noqa: E402
from reasoning_reviewer_lib import Finding, _review_one  # noqa: E402


def reviewer(name: str, role: str, model: str = "some-model") -> dict:
    return {"name": name, "model": model, "prompt": "p", "active": True, "type": [role], "loadOnDemand": True, "contextLength": 8192}


def write_config(folder: Path, name: str, stages, roles=("verifier",), model: str = "some-model") -> Path:
    data: dict = {"stages": list(stages), "reviewers": [reviewer(f"{r}-reviewer", r, model) for r in roles]}
    if "classifier" in roles:  # a classifier reviewer must have a rubric to classify against
        data["rubric"] = [{"name": "crash", "level": "high", "score": 0.6, "definition": "an input makes the code raise"}]
    path = folder / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_baselines(config: Path, expect: dict, measured="2026-01-02", model="some-model", run_type=lc.SYNTHETIC, metadata=None, omit=()) -> None:
    """A sidecar. By default a synthetic one counted on `orders`; `omit` drops a key to test a sidecar that lacks it."""
    data = {"measured": measured, "model": model, "runType": run_type, "metadata": {"cases": ["orders"]} if metadata is None else metadata, "expect": expect}
    lc.baselines_path(config).write_text(json.dumps({k: v for k, v in data.items() if k not in omit}), encoding="utf-8")


def orphan_sidecars(folder: Path) -> list:
    """Every sidecar in `folder` with no config of the same stem beside it.

    A sidecar is found only by name (`lc.baselines_path`), and `lc.load_baselines` treats a missing one as "no bars",
    so a config renamed without its sidecar loses its quality bars silently: no error and no failure. This is the
    check that turns that into a failure."""
    return sorted(p.name for p in folder.glob("*" + lc.BASELINES_SUFFIX)
                  if not (folder / (p.name[: -len(lc.BASELINES_SUFFIX)] + ".json")).is_file())


class Folder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)


class Discover(Folder):
    def test_lists_configs_sorted_and_leaves_out_the_baselines_files(self):
        b = write_config(self.dir, "b-config", ["verifier"])
        a = write_config(self.dir, "a-config", ["verifier"])
        write_baselines(a, {"min_real_confirmed": 1})
        (self.dir / "notes.txt").write_text("not a config")
        self.assertEqual(lc.discover(self.dir), [a, b])

    def test_empty_folder(self):
        self.assertEqual(lc.discover(self.dir), [])

    def test_a_sidecar_whose_config_was_renamed_without_it_is_reported(self):
        kept = write_config(self.dir, "reasoning-kept", ["reasoning"], roles=("reasoning",))
        write_baselines(kept, {"min_bugs_after": 1})
        renamed = write_config(self.dir, "reasoning-old-name", ["reasoning"], roles=("reasoning",))
        write_baselines(renamed, {"min_bugs_after": 1})
        self.assertEqual(orphan_sidecars(self.dir), [])
        renamed.rename(self.dir / "reasoning-new-name.json")  # the config moved, its sidecar did not
        self.assertEqual(orphan_sidecars(self.dir), ["reasoning-old-name.baselines.json"])


class KindFor(unittest.TestCase):
    def test_the_stages_choose_the_script(self):
        self.assertEqual(lc.kind_for(("verifier",)).script, "live_verifier.py")
        self.assertEqual(lc.kind_for(("reasoning",)).script, "live_reasoning.py")
        self.assertEqual(lc.kind_for(("classifier",)).script, "live_classifier.py")
        self.assertEqual(lc.kind_for(("reasoning", "verifier")).script, "live_all_reviewers.py")

    def test_a_pipeline_is_reasoning_with_the_verifier_the_classifier_or_both(self):
        for stages in (("reasoning", "verifier"), ("reasoning", "classifier"), ("reasoning", "verifier", "classifier")):
            with self.subTest(stages=stages):
                kind = lc.kind_for(stages)
                self.assertEqual((kind.name, kind.script, kind.args), ("pipeline", "live_all_reviewers.py", ("--case", "orders")))

    def test_the_order_a_config_lists_the_stages_in_does_not_matter(self):
        self.assertEqual(lc.kind_for(("verifier", "reasoning")), lc.kind_for(("reasoning", "verifier")))

    def test_a_stage_set_no_script_runs_has_no_kind(self):
        # The stage scripts each run one stage on fixtures, and live_all_reviewers.py starts from reasoning.
        self.assertIsNone(lc.kind_for(("verifier", "classifier")))
        self.assertIsNone(lc.kind_for(()))

    def test_every_kind_has_a_checking_function(self):
        self.assertEqual({k.name for k in lc.KINDS.values()}, set(rlt.CHECKERS))

    def test_every_script_exists(self):
        for kind in lc.KINDS.values():
            self.assertTrue((REPO_ROOT / "harness" / kind.script).is_file(), kind.script)


class LoadBaselines(Folder):
    def setUp(self):
        super().setUp()
        self.config = write_config(self.dir, "c", ["verifier"])

    def test_none_without_a_sidecar(self):
        self.assertIsNone(lc.load_baselines(self.config))

    def test_reads_the_numbers_and_what_they_were_measured_with(self):
        write_baselines(self.config, {"min_real_confirmed": 4}, measured="2026-10-27", model="m1")
        found = lc.load_baselines(self.config)
        self.assertEqual(found.expect, {"min_real_confirmed": 4})
        self.assertEqual(found.note, "baseline measured 2026-10-27 with m1")

    def test_unusable_sidecars_are_errors_not_silently_no_baselines(self):
        path = lc.baselines_path(self.config)
        for text, expected in (("not json", "cannot be read as JSON"), ("[]", "must be a JSON object"),
                               ('{"expect": {}}', "lacks"), ('{"measured": "d", "model": "m", "expect": {"a": "x"}}', "numbers"),
                               ('{"measured": "d", "model": "m", "expect": {"a": true}}', "numbers")):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(lc.BaselinesError) as caught:
                    lc.load_baselines(self.config)
                self.assertIn(expected, str(caught.exception))

    def refused(self, **write) -> str:
        """The message `load_baselines` refuses a sidecar written with these `write_baselines` arguments with."""
        write_baselines(self.config, {"min_real_confirmed": 1}, **write)
        with self.assertRaises(lc.BaselinesError) as caught:
            lc.load_baselines(self.config)
        return str(caught.exception)

    def test_reads_the_run_type_and_the_cases(self):
        write_baselines(self.config, {}, metadata={"cases": ["orders", "leaks"]})
        found = lc.load_baselines(self.config)
        self.assertEqual((found.run_type, found.metadata, found.cases), ("synthetic", {"cases": ["orders", "leaks"]}, ("orders", "leaks")))

    def test_a_live_sidecar_loads_with_its_tags_and_names_no_cases(self):
        # A live answer key is checked when it loads (tests/test_harness_live_branch.py covers what is refused).
        meta = {"tags": [{"tagId": "change-1", "bugs": [{"id": "B1", "file": "f.py", "line": 1, "class": "c", "claim": "x", "evidence": "diff"}],
                          "decoys": []}]}
        write_baselines(self.config, {}, run_type="live", metadata=meta)
        found = lc.load_baselines(self.config)
        self.assertEqual((found.run_type, found.metadata, found.cases, lc.expected_tags(found)), ("live", meta, (), ["change-1"]))

    def test_a_sidecar_with_no_run_type_is_refused_and_the_message_names_the_file(self):
        message = self.refused(omit=("runType",))
        self.assertTrue(message.startswith(lc.baselines_path(self.config).name), message)
        self.assertIn("no 'runType'", message)

    def test_a_run_type_that_is_neither_synthetic_nor_live_is_refused_and_the_message_names_the_file(self):
        for value in ("other", "", "Synthetic", None, 1, ["synthetic"]):
            with self.subTest(run_type=value):
                message = self.refused(run_type=value)
                self.assertTrue(message.startswith(lc.baselines_path(self.config).name), message)
                self.assertIn("neither", message)

    def test_a_sidecar_with_no_metadata_object_is_refused(self):
        for metadata in ("x", ["orders"], None):
            with self.subTest(metadata=metadata):
                data = {"measured": "d", "model": "m", "runType": "synthetic", "expect": {}, **({} if metadata is None else {"metadata": metadata})}
                lc.baselines_path(self.config).write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(lc.BaselinesError) as caught:
                    lc.load_baselines(self.config)
                self.assertIn("'metadata' object", str(caught.exception))

    def test_cases_that_cannot_be_used_are_refused(self):
        for cases, expected in (([], "non-empty"), ("orders", "non-empty"), ([1], "non-empty"), (["nope"], "does not have"),
                                (["all", "orders"], "mixes"), (["orders", "orders"], "twice")):
            with self.subTest(cases=cases):
                message = self.refused(metadata={"cases": cases})
                self.assertIn(expected, message)
                self.assertTrue(message.startswith(lc.baselines_path(self.config).name), message)

    def test_all_stands_alone_and_every_name_in_cases_py_is_accepted(self):
        for cases in (["all"], list(ALL_CASES)):
            with self.subTest(cases=cases):
                write_baselines(self.config, {}, metadata={"cases": cases})
                self.assertEqual(lc.load_baselines(self.config).cases, tuple(cases))


class DescribeBaselines(Folder):
    """What `describe` makes of a sidecar once it knows the script the config runs."""

    def described(self, stages, metadata=None, run_type="synthetic", roles=None) -> lc.LiveConfig:
        config = write_config(self.dir, "t", stages, roles=roles or tuple(stages))
        write_baselines(config, {}, metadata=metadata, run_type=run_type)
        return lc.describe(config)

    def test_a_config_with_no_sidecar_is_still_runnable_with_its_own_arguments(self):
        live_config = lc.describe(write_config(self.dir, "t", ["reasoning", "verifier"], roles=("reasoning", "verifier")))
        self.assertEqual((live_config.problem, live_config.baselines), ("", None))
        self.assertEqual(lc.script_args(live_config), ("--case", "orders"))

    def test_a_synthetic_sidecar_without_cases_is_invalid_for_a_script_that_takes_cases(self):
        for stages, roles in ((["verifier"], None), (["reasoning"], None), (["reasoning", "verifier"], ("reasoning", "verifier"))):
            with self.subTest(stages=stages):
                live_config = self.described(stages, metadata={}, roles=roles)
                self.assertIn("t.baselines.json", live_config.problem)
                self.assertIn("needs 'cases'", live_config.problem)
                self.assertFalse(live_config.unrunnable)

    def test_the_classifier_takes_no_cases_so_its_sidecar_needs_none_and_may_not_name_any(self):
        self.assertEqual(self.described(["classifier"], metadata={}).problem, "")
        named = self.described(["classifier"], metadata={"cases": ["orders"]})
        self.assertIn("classifier script takes no --case", named.problem)
        self.assertFalse(named.unrunnable)

    def test_the_pipeline_script_runs_one_case_so_a_sidecar_names_exactly_one(self):
        both = ("reasoning", "verifier")
        self.assertEqual(self.described(["reasoning", "verifier"], {"cases": ["orders"]}, roles=both).problem, "")
        for cases in (["orders", "leaks"], ["all"]):
            with self.subTest(cases=cases):
                live_config = self.described(["reasoning", "verifier"], {"cases": cases}, roles=both)
                self.assertIn("one case at a time", live_config.problem)
                self.assertFalse(live_config.unrunnable)

    def test_the_verifier_script_takes_cases_that_have_fixtures(self):
        self.assertEqual(self.described(["verifier"], {"cases": ["orders", "leaks"]}).problem, "")
        self.assertIn("has none", self.described(["verifier"], {"cases": ["clean"]}).problem)

    def test_a_live_sidecar_is_not_available_for_the_fixture_scripts_and_says_so(self):
        # The verifier and classifier scripts run fixtures from cases.py; a live bar is never run as a synthetic one.
        meta = {"tags": [{"tagId": "change-5", "clean": True, "bugs": [], "decoys": []}]}
        live_config = self.described(["verifier"], metadata=meta, run_type="live")
        self.assertTrue(live_config.unrunnable)
        self.assertIn("not available for the verifier script", live_config.problem)
        self.assertIn("t.baselines.json", live_config.problem)


class ScriptArgs(Folder):
    def args_for(self, stages, cases, roles=None) -> tuple:
        config = write_config(self.dir, "t", stages, roles=roles or tuple(stages))
        write_baselines(config, {}, metadata={"cases": cases})
        return lc.script_args(lc.describe(config))

    def test_the_cases_a_sidecar_names_become_one_case_argument_each_in_order(self):
        self.assertEqual(self.args_for(["verifier"], ["leaks", "orders"]), ("--case", "leaks", "--case", "orders"))
        self.assertEqual(self.args_for(["reasoning"], ["orders"]), ("--case", "orders"))
        self.assertEqual(self.args_for(["reasoning", "verifier"], ["leaks"], roles=("reasoning", "verifier")), ("--case", "leaks"))

    def test_all_is_expanded_here_to_what_the_kinds_script_can_run(self):
        names = lambda args: args[1::2]  # noqa: E731
        self.assertEqual(names(self.args_for(["reasoning"], ["all"])), tuple(ALL_CASES))
        self.assertEqual(names(self.args_for(["verifier"], ["all"])), tuple(fixture_cases()))

    def test_a_config_with_no_sidecar_keeps_its_kinds_own_arguments(self):
        for stages, roles, expected in ((["verifier"], None, ()), (["reasoning"], None, ()), (["classifier"], None, ()),
                                        (["reasoning", "verifier"], ("reasoning", "verifier"), ("--case", "orders"))):
            with self.subTest(stages=stages):
                self.assertEqual(lc.script_args(lc.describe(write_config(self.dir, "-".join(stages), stages, roles=roles or tuple(stages)))), expected)

    def test_the_runner_runs_the_script_with_those_arguments(self):
        config = write_config(self.dir, "t", ["verifier"])
        write_baselines(config, {"min_real_confirmed": 1}, metadata={"cases": ["orders", "leaks"]})
        commands: list = []

        class FakeProcess:
            stdout = io.StringIO("")

            def __init__(self, command, **_kwargs):
                commands.append(command)

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def wait(self):
                return 0

        with unittest.mock.patch.object(rlt.subprocess, "Popen", FakeProcess), contextlib.redirect_stdout(io.StringIO()):
            rlt.run_script(lc.describe(config), 3, 60, self.dir / "log.txt", {})
        self.assertEqual(commands[0][-4:], ["--case", "orders", "--case", "leaks"])
        self.assertIn("--runs", commands[0])

    def test_list_shows_the_arguments_the_runner_will_pass(self):
        config = write_config(self.dir, "t", ["verifier"])
        write_baselines(config, {"min_real_confirmed": 1}, metadata={"cases": ["orders", "leaks"]})
        self.assertIn("--case orders --case leaks", rlt.describe_config(lc.describe(config)))

        config = write_config(self.dir, "v", ["verifier"], model="m1")
        live = lc.describe(config)
        self.assertEqual((live.name, live.stages, live.models, live.problem), ("v", ("verifier",), ("m1",), ""))
        self.assertEqual(live.kind.script, "live_verifier.py")
        self.assertIsNone(live.baselines)

    def test_inactive_reviewers_name_no_model(self):
        data = {"stages": ["verifier"], "reviewers": [reviewer("on", "verifier", "m1"), {**reviewer("off", "verifier", "m2"), "active": False}]}
        path = self.dir / "x.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(lc.describe(path).models, ("m1",))

    def test_models_are_only_those_of_reviewers_a_selected_stage_runs(self):
        # A pipeline config copied to test reasoning alone keeps its verifier entry; live_reasoning.py never loads it.
        data = {"stages": ["reasoning"], "reviewers": [reviewer("r", "reasoning", "reason-model"), reviewer("v", "verifier", "verify-model")]}
        path = self.dir / "reasoning-only.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(lc.describe(path).models, ("reason-model",))
        data["stages"] = ["reasoning", "verifier"]
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(lc.describe(path).models, ("reason-model", "verify-model"))

    def test_a_reviewer_with_several_roles_counts_when_any_of_its_roles_is_selected(self):
        both = {**reviewer("b", "reasoning", "both-model"), "type": ["reasoning", "verifier"]}
        path = self.dir / "both.json"
        path.write_text(json.dumps({"stages": ["verifier"], "reviewers": [both]}), encoding="utf-8")
        self.assertEqual(lc.describe(path).models, ("both-model",))

    def test_a_verifier_that_shares_its_entry_with_another_type_and_has_a_prompt_file_is_refused_by_the_harness(self):
        # The server appends the built-in verify steps after such a file; the harness runs one stage at a time and would not.
        (self.dir / "lens.md").write_text("Reply with finding blocks, or NONE.", encoding="utf-8")
        both = {**reviewer("b", "reasoning", "both-model"), "type": ["reasoning", "verifier"], "promptFile": "lens.md"}
        path = self.dir / "both-file.json"
        path.write_text(json.dumps({"stages": ["verifier"], "reviewers": [both]}), encoding="utf-8")
        live = lc.describe(path)
        self.assertIn("shares its entry", live.problem)
        self.assertIn("its own entry", live.problem)

    def test_several_types_with_a_prompt_file_are_fine_when_the_verifier_is_not_one_of_them(self):
        (self.dir / "lens.md").write_text("Reply with finding blocks, or NONE.", encoding="utf-8")
        both = {**reviewer("b", "reasoning", "both-model"), "type": ["reasoning", "classifier"], "promptFile": "lens.md"}
        path = self.dir / "rc.json"
        path.write_text(json.dumps({"stages": ["reasoning"], "rubric": [{"name": "r", "level": "low", "definition": "d", "score": 0.5}],
                                    "reviewers": [both]}), encoding="utf-8")
        self.assertFalse(lc.describe(path).problem)

    def test_a_config_that_does_not_load_is_invalid_and_says_why(self):
        path = self.dir / "bad.json"
        path.write_text('{"stages": ["verifier"], "liveTest": 1, "reviewers": []}', encoding="utf-8")
        live = lc.describe(path)
        self.assertTrue(live.problem)
        self.assertFalse(live.unrunnable)
        self.assertIsNone(live.kind)

    def test_stages_no_script_runs_are_unrunnable_not_invalid(self):
        config = write_config(self.dir, "vc", ["verifier", "classifier"], roles=("verifier", "classifier"))
        live = lc.describe(config)
        self.assertIsNone(live.kind)
        self.assertTrue(live.unrunnable)
        self.assertIn("no live test script runs the stages ['verifier', 'classifier']", live.problem)

    def test_a_broken_baselines_file_makes_the_config_invalid(self):
        config = write_config(self.dir, "v", ["verifier"])
        lc.baselines_path(config).write_text("nope", encoding="utf-8")
        live = lc.describe(config)
        self.assertIn("cannot be read as JSON", live.problem)
        self.assertFalse(live.unrunnable)


# What the scripts print: the lines the checkers read (see run_live_tests.py).
MODELS = "loaded before: ['x']\nloaded after:  ['x']\n"


def verifier_output(confirmed=(5, 5, 5), wrongly_confirmed=((), (), ()), cases=("orders",)) -> str:
    out = f"cases: {', '.join(cases)}\n" if cases else ""
    for n, (c, w) in enumerate(zip(confirmed, wrongly_confirmed, strict=True), 1):
        out += f"--- run {n}\nreal findings confirmed: {c}/6  wrongly rejected: []\nfalse findings rejected: {3 - len(w)}/3  wrongly confirmed: [{', '.join(w)}]\n"
    return out + MODELS


def score_blocks(cases) -> str:
    """The per-case blocks `harness_lib.format_score` prints, from the real function, for each case name."""
    return "".join(h.format_score(ALL_CASES[name], [h.score([], ALL_CASES[name])]) + "\n" for name in cases)


def pipeline_output(runs=1, unchecked=0, bugs_after=2, alarms_after=0, cases=("orders",)) -> str:
    out = ""
    for n in range(1, runs + 1):
        out += f"--- run {n} verified: 2 confirmed, 1 disputed, {unchecked} unchecked; dropped 1 rejected, 0 ungrounded\n"
    # The scores are printed twice, before and after verification, each time with the per-case blocks first.
    return (out + "===== before verification\n" + score_blocks(cases) + "3/6 bug(s) found; 2/3 decoy alarm\n"
            "===== after verification (everything not rejected)\n" + score_blocks(cases) + f"{bugs_after}/6 bug(s) found; {alarms_after}/3 decoy alarm\n" + MODELS)


def _classification(statuses: dict) -> ClassificationReview:
    """A real `ClassificationReview` with the given number of findings in each outcome, for the real summary line."""
    items = []
    for _ in range(statuses.get("high", 0)):
        items.append(Classified(Finding("a.py", 1, 1, "c", "x", "y", ("r",)), "classified", "confirmed", "e", "high", 0.9,
                                classifications=(Classification("k", (Rating("e", 0.9),)),)))
    for status in ("unclassified", "unchecked"):
        items.extend(Classified(Finding("a.py", 1, 1, "c", "x", "y", ("r",)), status, "confirmed", note="no classifier gave a usable answer")
                     for _ in range(statuses.get(status, 0)))
    return ClassificationReview(classified=items, classifiers_run=["k"])


def classifier_fixture_output(runs=1, on_expected=10, unanswered=0, drift=0.0) -> str:
    """What `live_classifier.py` prints, built by its own functions."""
    cases = classifier_cases()
    # The first `on_expected` fixtures land on their entry; the rest land on one that is wrong for them (a finding that fits
    # nothing is "wrong" when it is given an entry), so the printed count is exactly `on_expected`.
    def landed(entry):
        return (entry, 0.9, "classified") if entry else (None, None, "unclassified")

    right = {c.id: landed(c.expected) if n < on_expected else landed("style-only" if c.expected != "style-only" else "crash-on-input")
             for n, c in enumerate(cases)}
    out = ""
    for n in range(1, runs + 1):
        review = _classification({"unchecked": unanswered})
        out += f"--- run {n}\n" + live_classifier.score_choices(right, {}, list(RUBRIC)) + "\n" + "\n".join(live_classifier.unanswered_lines(review)) + "\n"
    # The repeat report the script prints after its runs: here every run gave the same averages (or, with `drift`, one differed).
    means = [{"f": {"wrong-result": 0.6 + (drift if n == runs else 0)}} for n in range(1, runs + 1)]
    return out + "\n".join(live_classifier.repeat_lines(means)) + ("\n" if runs > 1 else "") + MODELS


def classified_output(runs=1, unsorted=0, with_verifier=True) -> str:
    """The classifier part of what `live_all_reviewers.py` prints: one line per run, from its own function."""
    out = ""
    for n in range(1, runs + 1):
        out += live_all_reviewers.classified_summary(n, _classification({"high": 2, "unchecked": unsorted})) + "\n"
    return out if with_verifier else out + "=====\n"


def live(folder: Path, stages, baselines=None, name="t", metadata=None) -> lc.LiveConfig:
    roles = tuple(stages)
    config = write_config(folder, name, stages, roles=roles)
    if baselines is not None:
        # The classifier script takes no --case, so a classifier's sidecar names none.
        write_baselines(config, baselines, metadata={} if metadata is None and tuple(stages) == ("classifier",) else metadata)
    return lc.describe(config)


def kinds_of(checks) -> set:
    return {c.kind for c in checks}


@functools.lru_cache(maxsize=None)
def real_reviewer_errors() -> dict:
    """What the harness prints for each kind of reviewer problem, with the text produced by the real library.

    Each reply is run through the real reviewer code (`_review_one`, with a stub in place of the model), and each problem
    it records is formatted the way `harness/live_reasoning.py` and `live_all_reviewers.py` print it. Nothing here types
    the library's wording, so a reworded message fails the checks' tests instead of the checks going blind."""
    with tempfile.TemporaryDirectory() as tmp:
        failing_translator = Path(tmp) / "fails.py"
        failing_translator.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")

        def printed(*replies, **reviewer_args) -> str:
            # One chunk per reply. A request that fails raises only when every one did (the reviewer then has no
            # result at all), so a failed request needs a second chunk that is answered.
            reviewer = Reviewer("r", "m", "p", True, ("reasoning",), False, **reviewer_args)
            answers = iter(replies)
            result = _review_one(reviewer, lambda _reviewer, _system: [f"File: {n}.py\n    1 + x\n" for n in range(len(replies))],
                                 None, lambda *_args: next(answers), [], [])
            assert result.problems, "the real library reported no problem for a reply meant to have one"
            return "".join(f"  run 1: reviewer error: {problem}\n" for problem in result.problems)

        return {
            "translator": printed("any reply", translator=str(failing_translator)),
            "no_finding_block": printed("I could not find a defect worth reporting."),
            "not_json": printed("not json at all", reply_format="json"),
            "no_findings_list": printed('{"issues": []}', reply_format="json"),
            "failed_request": printed(None, "NONE"),
            "skipped_finding": printed("FINDING\nfile: a.py\nlines: 1-1\nclass: x\nEND"),
        }


class VerifierChecker(Folder):
    def test_without_baselines_only_mechanics_are_checked(self):
        config = live(self.dir, ["verifier"])
        checks = rlt.check_verifier_fixtures(verifier_output(confirmed=(0, 0, 0), wrongly_confirmed=(("a", "b"),) * 3), 0, config, 3)
        self.assertEqual(kinds_of(checks), {"mechanics"})
        self.assertTrue(all(c.ok for c in checks), [c for c in checks if not c.ok])

    def test_with_baselines_a_worse_run_fails_only_the_baseline_checks(self):
        config = live(self.dir, ["verifier"], {"min_real_confirmed": 4, "max_false_confirmed": 0})
        checks = rlt.check_verifier_fixtures(verifier_output(confirmed=(5, 3, 5), wrongly_confirmed=((), ("a",), ())), 0, config, 3)
        failed = [c for c in checks if not c.ok]
        self.assertEqual({c.kind for c in failed}, {"baseline"})
        self.assertEqual(len(failed), 2)
        self.assertIn("baseline measured 2026-01-02 with some-model", "\n".join(c.name for c in checks))

    def test_with_baselines_a_good_run_passes(self):
        config = live(self.dir, ["verifier"], {"min_real_confirmed": 4, "max_false_confirmed": 0})
        checks = rlt.check_verifier_fixtures(verifier_output(), 0, config, 3)
        self.assertTrue(all(c.ok for c in checks), [c for c in checks if not c.ok])
        self.assertEqual(kinds_of(checks), {"mechanics", "baseline"})

    def test_a_key_the_sidecar_leaves_out_is_not_checked(self):
        config = live(self.dir, ["verifier"], {"min_real_confirmed": 4})
        checks = rlt.check_verifier_fixtures(verifier_output(wrongly_confirmed=(("a",),) * 3), 0, config, 3)
        self.assertEqual([c.name for c in checks if c.kind == "baseline" and c.name != rlt.CHECK_CASES][0][:34], "real findings confirmed, worst run")
        self.assertEqual(len([c for c in checks if c.kind == "baseline" and c.name != rlt.CHECK_CASES]), 1)
        self.assertTrue(all(c.ok for c in checks))

    def test_mechanics_fail_whatever_the_baselines_say(self):
        config = live(self.dir, ["verifier"])
        checks = rlt.check_verifier_fixtures(verifier_output() + "verifier problem: boom\n", 1, config, 3)
        failed = {c.name for c in checks if not c.ok}
        self.assertIn("the script exits 0", failed)
        self.assertIn("no verifier problem (an unusable reply, a failed request)", failed)

    def test_a_run_that_did_not_complete_fails(self):
        config = live(self.dir, ["verifier"])
        checks = rlt.check_verifier_fixtures(verifier_output(confirmed=(5, 5), wrongly_confirmed=((), ())), 0, config, 3)
        self.assertFalse(next(c for c in checks if "completed and was scored" in c.name).ok)


class PipelineChecker(Folder):
    BARS = {"max_decoy_alarms_after": 0, "min_bugs_after": 1}

    def test_without_baselines_only_mechanics_are_checked(self):
        config = live(self.dir, ["reasoning", "verifier"])
        checks = rlt.check_pipeline(pipeline_output(bugs_after=0, alarms_after=3), 0, config, 1)
        self.assertEqual(kinds_of(checks), {"mechanics"})
        self.assertTrue(all(c.ok for c in checks), [c for c in checks if not c.ok])

    def test_with_baselines_worse_quality_fails_them(self):
        config = live(self.dir, ["reasoning", "verifier"], self.BARS)
        checks = rlt.check_pipeline(pipeline_output(bugs_after=0, alarms_after=1), 0, config, 1)
        self.assertEqual({c.kind for c in checks if not c.ok}, {"baseline"})
        self.assertEqual(len([c for c in checks if not c.ok]), 2)

    def test_with_baselines_good_quality_passes(self):
        config = live(self.dir, ["reasoning", "verifier"], self.BARS)
        self.assertTrue(all(c.ok for c in rlt.check_pipeline(pipeline_output(), 0, config, 1)))

    def test_a_verifier_that_left_findings_unchecked_fails_mechanics(self):
        config = live(self.dir, ["reasoning", "verifier"])
        checks = rlt.check_pipeline(pipeline_output(unchecked=2), 0, config, 1)
        self.assertFalse(next(c for c in checks if "none unchecked" in c.name).ok)

    def test_no_summary_after_the_marker_fails_even_without_baselines(self):
        config = live(self.dir, ["reasoning", "verifier"])
        checks = rlt.check_pipeline(pipeline_output().replace("bug(s) found", "bugs"), 0, config, 1)
        self.assertFalse(next(c for c in checks if "summary line follows" in c.name).ok)

    def test_a_reasoning_reviewer_problem_fails_the_pipeline_check_with_the_text(self):
        # The pipeline runs the reasoning stage too, and its script prints the same `reviewer error:` lines.
        config = live(self.dir, ["reasoning", "verifier"])
        for key, name in (("translator", rlt.CHECK_TRANSLATOR), ("no_finding_block", rlt.CHECK_READABLE), ("failed_request", rlt.CHECK_REVIEWER_ERROR)):
            with self.subTest(problem=key):
                errors = real_reviewer_errors()[key]
                checks = rlt.check_pipeline(errors + pipeline_output(), 0, config, 1)
                self.assertEqual([c.name for c in checks if not c.ok], [name])
                self.assertIn(errors.split("reviewer error: ", 1)[1].strip(), next(c.detail for c in checks if c.name == name))

    def test_the_pipeline_still_fails_on_a_verifier_problem(self):
        config = live(self.dir, ["reasoning", "verifier"])
        checks = rlt.check_pipeline(pipeline_output() + "  verifier problem: boom\n", 0, config, 1)
        self.assertEqual([c.name for c in checks if not c.ok], [rlt.CHECK_VERIFIER_PROBLEM])


class ClassifierChecker(Folder):
    def test_clean_output_passes_with_mechanics_only(self):
        config = live(self.dir, ["classifier"])
        checks = rlt.check_classifier_fixtures(classifier_fixture_output(runs=2), 0, config, 2)
        self.assertEqual(kinds_of(checks), {"mechanics"})
        self.assertTrue(all(c.ok for c in checks), [c for c in checks if not c.ok])

    def test_a_run_that_did_not_complete_fails(self):
        config = live(self.dir, ["classifier"])
        checks = rlt.check_classifier_fixtures(classifier_fixture_output(runs=1), 0, config, 2)
        # The second run never printed, so there is no repeat report to read either.
        self.assertEqual([c.name for c in checks if not c.ok], ["every one of the 2 run(s) completed and was scored",
                                                                "the entry averages came out the same from run to run"])

    def test_a_finding_nobody_answered_fails_and_shows_it(self):
        config = live(self.dir, ["classifier"])
        checks = rlt.check_classifier_fixtures(classifier_fixture_output(unanswered=1), 0, config, 1)
        failed = [c for c in checks if not c.ok]
        self.assertEqual([c.name for c in failed], ["every fixture was answered: none left without a usable answer"])
        self.assertIn("not answered:", failed[0].detail)

    def test_a_classifier_problem_fails_its_own_check(self):
        config = live(self.dir, ["classifier"])
        output = "  classifier problem: finding 2/12: reviewer 'c': no probability for crash, so this classifier gives no answer for it\n" + classifier_fixture_output()
        checks = rlt.check_classifier_fixtures(output, 0, config, 1)
        self.assertEqual([c.name for c in checks if not c.ok], [rlt.CHECK_CLASSIFIER_PROBLEM])

    def test_with_a_baseline_a_worse_run_fails_it_and_a_better_one_passes(self):
        config = live(self.dir, ["classifier"], {"min_on_expected": 9})
        better = rlt.check_classifier_fixtures(classifier_fixture_output(on_expected=10), 0, config, 1)
        self.assertTrue(all(c.ok for c in better))
        self.assertIn("baseline", kinds_of(better))
        worse = rlt.check_classifier_fixtures(classifier_fixture_output(on_expected=8), 0, config, 1)
        self.assertEqual([(c.kind, c.ok) for c in worse if not c.ok], [("baseline", False)])

    def test_the_score_line_the_checker_reads_is_the_one_the_script_prints(self):
        self.assertEqual(rlt.ON_EXPECTED_LINE.findall(classifier_fixture_output(on_expected=10)), [("10", str(len(classifier_cases())))])

    def test_averages_that_came_out_the_same_between_runs_pass_the_repeat_check(self):
        config = live(self.dir, ["classifier"])
        checks = rlt.check_classifier_fixtures(classifier_fixture_output(runs=2), 0, config, 2)
        same = [c for c in checks if "came out the same" in c.name]
        self.assertEqual([(c.ok, c.kind) for c in same], [(True, "mechanics")])

    def test_an_average_that_differed_fails_it_and_names_the_fixture_and_entry(self):
        config = live(self.dir, ["classifier"])
        checks = rlt.check_classifier_fixtures(classifier_fixture_output(runs=2, drift=0.2), 0, config, 2)
        (failed,) = [c for c in checks if not c.ok]
        self.assertIn("came out the same", failed.name)
        self.assertIn("differs: f / wrong-result: 0.6000, 0.8000", failed.detail)

    def test_one_run_has_no_repeat_check(self):
        config = live(self.dir, ["classifier"])
        checks = rlt.check_classifier_fixtures(classifier_fixture_output(runs=1), 0, config, 1)
        self.assertFalse([c for c in checks if "came out the same" in c.name])

    def test_two_runs_that_printed_no_repeat_report_fail_rather_than_pass_silently(self):
        config = live(self.dir, ["classifier"])
        output = classifier_fixture_output(runs=2).split("score repeat")[0] + MODELS
        failed = [c for c in rlt.check_classifier_fixtures(output, 0, config, 2) if not c.ok]
        self.assertIn("came out the same", failed[0].name)


class PipelineWithTheClassifier(Folder):
    def test_all_three_stages_pass_when_each_ran_and_sorted_every_finding(self):
        config = live(self.dir, ["reasoning", "verifier", "classifier"])
        checks = rlt.check_pipeline(pipeline_output(runs=2) + classified_output(runs=2), 0, config, 2)
        self.assertTrue(all(c.ok for c in checks), [c for c in checks if not c.ok])
        names = [c.name for c in checks]
        self.assertIn("the verifier ran after reasoning in every one of the 2 run(s)", names)
        self.assertIn("the classifier ran after verification in every one of the 2 run(s)", names)

    def test_a_classifier_that_did_not_run_in_every_run_fails(self):
        config = live(self.dir, ["reasoning", "verifier", "classifier"])
        checks = rlt.check_pipeline(pipeline_output(runs=2) + classified_output(runs=1), 0, config, 2)
        self.assertEqual([c.name for c in checks if not c.ok], ["the classifier ran after verification in every one of the 2 run(s)"])

    def test_findings_left_without_a_rubric_level_fail(self):
        config = live(self.dir, ["reasoning", "verifier", "classifier"])
        checks = rlt.check_pipeline(pipeline_output() + classified_output(unsorted=1), 0, config, 1)
        self.assertEqual([c.name for c in checks if not c.ok], ["every finding that stood was sorted: none without a rubric level in any run"])

    def test_a_classifier_problem_fails_the_pipeline_check_with_its_own_name(self):
        config = live(self.dir, ["reasoning", "verifier", "classifier"])
        output = pipeline_output() + classified_output() + "  classifier problem: boom\n"
        self.assertEqual([c.name for c in rlt.check_pipeline(output, 0, config, 1) if not c.ok], [rlt.CHECK_CLASSIFIER_PROBLEM])

    def test_reasoning_then_the_classifier_has_no_verification_checks_and_says_what_the_classifier_followed(self):
        config = live(self.dir, ["reasoning", "classifier"])
        checks = rlt.check_pipeline(classified_output(with_verifier=False) + MODELS, 0, config, 1)
        self.assertTrue(all(c.ok for c in checks), [c for c in checks if not c.ok])
        names = [c.name for c in checks]
        self.assertIn("the classifier ran after reasoning in every one of the 1 run(s)", names)
        self.assertFalse([n for n in names if "verifier" in n.lower() and n != rlt.CHECK_VERIFIER_PROBLEM])
        self.assertNotIn(rlt.CHECK_VERIFIER_PROBLEM, names)

    def test_the_verification_baselines_apply_only_when_the_verifier_is_selected(self):
        config = live(self.dir, ["reasoning", "classifier"], {"max_decoy_alarms_after": 0, "min_bugs_after": 1})
        checks = rlt.check_pipeline(score_blocks(["orders"]) + classified_output(with_verifier=False) + MODELS, 0, config, 1)
        # The only baseline-kind check is the one about which cases ran; neither verification bar is there.
        self.assertEqual([c.name for c in checks if c.kind == "baseline"], [rlt.CHECK_CASES])
        self.assertTrue(all(c.ok for c in checks), [c for c in checks if not c.ok])

    def test_a_pipeline_without_the_classifier_is_unchanged(self):
        config = live(self.dir, ["reasoning", "verifier"])
        names = [c.name for c in rlt.check_pipeline(pipeline_output(), 0, config, 1)]
        self.assertFalse([n for n in names if "classifier" in n])


class CommonMechanics(Folder):
    # The checks each stage calls for. A stage a kind of test runs that is not in this table is a gap: see the last test.
    FOR_STAGE = {"reasoning": (rlt.CHECK_TRANSLATOR, rlt.CHECK_READABLE, rlt.CHECK_REVIEWER_ERROR), "verifier": (rlt.CHECK_VERIFIER_PROBLEM,),
                 "classifier": (rlt.CHECK_CLASSIFIER_PROBLEM,)}

    def test_every_kind_of_test_runs_the_common_checks_its_stages_call_for(self):
        for stages, kind in lc.KINDS.items():
            with self.subTest(kind=kind.name):
                config = live(self.dir, sorted(stages), name=kind.name)
                names = {c.name for c in rlt.CHECKERS[kind.name]("", 0, config, 1)}
                wanted = {rlt.CHECK_EXITS, rlt.CHECK_MODELS} | {n for stage in stages for n in self.FOR_STAGE[stage]}
                self.assertLessEqual(wanted, names)

    def test_every_stage_a_kind_of_test_runs_has_its_common_checks_listed(self):
        # Adding a kind of test for a new stage must add its checks here and in `common_mechanics`.
        self.assertLessEqual(set().union(*lc.KINDS), set(self.FOR_STAGE))

    def test_the_verifier_test_has_no_reviewer_checks(self):
        config = live(self.dir, ["verifier"])
        names = [c.name for c in rlt.check_verifier_fixtures(verifier_output(), 0, config, 3)]
        for name in self.FOR_STAGE["reasoning"]:
            self.assertNotIn(name, names)

    def test_the_checks_come_in_a_fixed_order_with_the_exit_code_first(self):
        names = [c.name for c in rlt.common_mechanics(MODELS, 0, ("reasoning", "verifier"))]
        self.assertEqual(names, [rlt.CHECK_EXITS, rlt.CHECK_TRANSLATOR, rlt.CHECK_READABLE, rlt.CHECK_REVIEWER_ERROR,
                                 rlt.CHECK_VERIFIER_PROBLEM, rlt.CHECK_MODELS])
        names = [c.name for c in rlt.common_mechanics(MODELS, 0, ("reasoning", "verifier", "classifier"))]
        self.assertEqual(names, [rlt.CHECK_EXITS, rlt.CHECK_TRANSLATOR, rlt.CHECK_READABLE, rlt.CHECK_REVIEWER_ERROR,
                                 rlt.CHECK_VERIFIER_PROBLEM, rlt.CHECK_CLASSIFIER_PROBLEM, rlt.CHECK_MODELS])

    def test_a_classifier_problem_fails_its_own_check_alone_and_shows_the_line_wherever_it_is_printed(self):
        line = "  classifier problem: finding 1/12: reviewer 'general': it rated an entry that is not in the rubric, so the answer is not counted\n"
        for stages in (("classifier",), ("reasoning", "classifier"), ("reasoning", "verifier", "classifier")):
            with self.subTest(stages=stages):
                checks = rlt.common_mechanics(line + MODELS, 0, stages)
                self.assertEqual([c.name for c in checks if not c.ok], [rlt.CHECK_CLASSIFIER_PROBLEM])
                self.assertIn("not in the rubric", next(c.detail for c in checks if c.name == rlt.CHECK_CLASSIFIER_PROBLEM))
        self.assertTrue(all(c.ok for c in rlt.common_mechanics(line + MODELS, 0, ("reasoning", "verifier"))))  # not a stage here


class ReasoningChecker(Folder):
    def setUp(self):
        super().setUp()
        self.config = live(self.dir, ["reasoning"])

    def test_clean_output_passes_with_mechanics_only(self):
        checks = rlt.check_reasoning("===== seeded-three\n" + MODELS, 0, self.config, 2)
        self.assertEqual(kinds_of(checks), {"mechanics"})
        self.assertTrue(all(c.ok for c in checks))

    def failed_names(self, output: str) -> list:
        return [c.name for c in rlt.check_reasoning(output + MODELS, 0, self.config, 2) if not c.ok]

    def detail_of(self, output: str, name: str) -> str:
        return next(c.detail for c in rlt.check_reasoning(output + MODELS, 0, self.config, 2) if c.name == name)

    def test_a_reviewer_error_that_is_neither_fails_the_catch_all_check_alone(self):
        self.assertEqual(self.failed_names("run 1: reviewer error: boom\n"), [rlt.CHECK_REVIEWER_ERROR])

    def test_a_translator_failure_fails_its_own_check_and_shows_the_text(self):
        errors = real_reviewer_errors()["translator"]
        self.assertEqual(self.failed_names(errors), [rlt.CHECK_TRANSLATOR])
        self.assertIn(errors.split("reviewer error: ", 1)[1].strip(), self.detail_of(errors, rlt.CHECK_TRANSLATOR))

    def test_every_kind_of_unreadable_reply_fails_the_readable_check_and_shows_the_text(self):
        for key in ("no_finding_block", "not_json", "no_findings_list"):
            with self.subTest(reply=key):
                errors = real_reviewer_errors()[key]
                self.assertEqual(self.failed_names(errors), [rlt.CHECK_READABLE])
                self.assertIn(errors.split("reviewer error: ", 1)[1].strip(), self.detail_of(errors, rlt.CHECK_READABLE))

    def test_a_failed_request_and_a_skipped_finding_fail_the_catch_all_check(self):
        for key in ("failed_request", "skipped_finding"):
            with self.subTest(problem=key):
                self.assertEqual(self.failed_names(real_reviewer_errors()[key]), [rlt.CHECK_REVIEWER_ERROR])

    def test_a_line_is_in_one_check_only(self):
        # A translator's own message can echo the model's reply, which may contain the reader's wording.
        line = "  run 1: reviewer error: reviewer 'r': chunk 1/1: the translator exited with status 1: the reply was not JSON\n"
        self.assertEqual(self.failed_names(line), [rlt.CHECK_TRANSLATOR])

    def test_a_reasoning_test_has_no_verifier_check(self):
        names = [c.name for c in rlt.check_reasoning(MODELS, 0, self.config, 2)]
        self.assertNotIn(rlt.CHECK_VERIFIER_PROBLEM, names)

    def test_models_not_restored_fails(self):
        checks = rlt.check_reasoning("loaded before: ['x']\nloaded after:  ['x', 'y']\n", 0, self.config, 2)
        self.assertEqual(len([c for c in checks if not c.ok]), 1)

    def test_a_nonzero_exit_fails(self):
        self.assertFalse(rlt.check_reasoning(MODELS, 2, self.config, 2)[0].ok)


class CasesCheck(Folder):
    """The set of cases a run reports must be the set its baselines were counted on."""

    def test_the_set_the_baselines_name_passes_and_a_different_one_fails_naming_both(self):
        config = live(self.dir, ["verifier"], {"min_real_confirmed": 4}, metadata={"cases": ["orders"]})
        good = rlt.check_verifier_fixtures(verifier_output(), 0, config, 3)
        self.assertTrue(all(c.ok for c in good), [c for c in good if not c.ok])
        bad = rlt.check_verifier_fixtures(verifier_output(cases=("leaks",)), 0, config, 3)
        failed = [c for c in bad if not c.ok]
        self.assertEqual([c.name for c in failed], [rlt.CHECK_CASES])
        self.assertEqual(failed[0].kind, "baseline")
        self.assertIn("['orders']", failed[0].detail)
        self.assertIn("['leaks']", failed[0].detail)

    def test_a_larger_set_than_the_baselines_name_fails_too(self):
        config = live(self.dir, ["verifier"], {"min_real_confirmed": 4}, metadata={"cases": ["orders"]})
        checks = rlt.check_verifier_fixtures(verifier_output(cases=("orders", "leaks")), 0, config, 3)
        self.assertEqual([c.name for c in checks if not c.ok], [rlt.CHECK_CASES])

    def test_a_run_that_reports_no_cases_fails_rather_than_passing_for_want_of_a_comparison(self):
        config = live(self.dir, ["verifier"], {"min_real_confirmed": 4}, metadata={"cases": ["orders"]})
        checks = rlt.check_verifier_fixtures(verifier_output(cases=()), 0, config, 3)
        failed = [c for c in checks if not c.ok]
        self.assertEqual([c.name for c in failed], [rlt.CHECK_CASES])
        self.assertIn("no cases", failed[0].detail)

    def test_the_order_the_cases_are_reported_in_does_not_matter(self):
        config = live(self.dir, ["verifier"], {"min_real_confirmed": 4}, metadata={"cases": ["orders", "leaks"]})
        checks = rlt.check_verifier_fixtures(verifier_output(cases=("leaks", "orders")), 0, config, 3)
        self.assertTrue(all(c.ok for c in checks), [c for c in checks if not c.ok])

    def test_a_config_with_no_sidecar_has_no_cases_check(self):
        config = live(self.dir, ["verifier"])
        names = [c.name for c in rlt.check_verifier_fixtures(verifier_output(cases=("leaks",)), 0, config, 3)]
        self.assertNotIn(rlt.CHECK_CASES, names)

    def test_a_live_sidecar_is_not_compared_with_cases(self):
        config = write_config(self.dir, "t", ["verifier"])
        write_baselines(config, {}, run_type="live", metadata={})
        self.assertEqual(rlt.cases_checks(verifier_output(), lc.describe(config)), [])

    def test_the_bars_say_which_cases_they_are_counted_over(self):
        config = live(self.dir, ["verifier"], {"min_real_confirmed": 4, "max_false_confirmed": 0})
        names = [c.name for c in rlt.check_verifier_fixtures(verifier_output(), 0, config, 3)]
        self.assertIn("real findings confirmed, worst run >= 4 of 6 on `orders` (baseline measured 2026-01-02 with some-model)", names)
        self.assertIn("false findings wrongly confirmed, worst run <= 0 of 3 on `orders`", names)

    def test_a_long_set_is_counted_not_listed_in_a_bar_name(self):
        config = live(self.dir, ["verifier"], {"min_real_confirmed": 4}, metadata={"cases": ["all"]})
        self.assertEqual(rlt.on_cases(config), f" on {len(fixture_cases())} cases")

    def test_the_verifier_script_prints_the_line_the_checker_reads(self):
        for names in (["orders"], ["orders", "leaks"], ["all"]):
            with self.subTest(cases=names):
                line = live_verifier.cases_line(argparse.Namespace(case=names))
                expected = list(fixture_cases()) if names == ["all"] else names
                self.assertEqual(rlt.reported_cases(line + "\n" + MODELS, "verifier"), expected)

    def test_the_pipeline_reports_its_case_in_the_blocks_the_scripts_print_once_or_twice(self):
        config = live(self.dir, ["reasoning", "verifier"], {"max_decoy_alarms_after": 0, "min_bugs_after": 1})
        good = rlt.check_pipeline(pipeline_output(), 0, config, 1)
        self.assertTrue(all(c.ok for c in good), [c for c in good if not c.ok])
        self.assertIn(rlt.CHECK_CASES, [c.name for c in good])
        bad = rlt.check_pipeline(pipeline_output(cases=("leaks",)), 0, config, 1)
        self.assertEqual([c.name for c in bad if not c.ok], [rlt.CHECK_CASES])
        self.assertIn("bugs found after verification >= 1 of 6 on `orders`", [c.name for c in good])

    def test_reasoning_numbers_are_a_record_but_the_cases_they_were_counted_on_are_still_checked(self):
        names = list(ALL_CASES)
        config = live(self.dir, ["reasoning"], {"bugs_found_of_38": 13}, metadata={"cases": names})
        good = rlt.check_reasoning(score_blocks(names) + MODELS, 0, config, 2)
        self.assertTrue(all(c.ok for c in good), [c for c in good if not c.ok])
        self.assertIn(rlt.CHECK_CASES, [c.name for c in good])
        short = rlt.check_reasoning(score_blocks(names[:-1]) + MODELS, 0, config, 2)
        self.assertEqual([c.name for c in short if not c.ok], [rlt.CHECK_CASES])

    def test_all_in_a_reasoning_sidecar_means_every_case_in_cases_py(self):
        config = live(self.dir, ["reasoning"], {}, metadata={"cases": ["all"]})
        self.assertTrue(all(c.ok for c in rlt.check_reasoning(score_blocks(ALL_CASES) + MODELS, 0, config, 2)))
        self.assertFalse(all(c.ok for c in rlt.check_reasoning(score_blocks(["orders"]) + MODELS, 0, config, 2)))

    def test_a_classifier_has_no_cases_check(self):
        config = live(self.dir, ["classifier"], {"min_on_expected": 9})
        self.assertNotIn(rlt.CHECK_CASES, [c.name for c in rlt.check_classifier_fixtures(classifier_fixture_output(on_expected=10), 0, config, 1)])


class Choose(Folder):
    def setUp(self):
        super().setUp()
        self.configs = [lc.describe(write_config(self.dir, n, ["verifier"])) for n in ("alpha-verifier", "beta-verifier", "gamma")]

    def names(self, chosen):
        return [c.name for c in chosen]

    def test_nothing_chosen_is_everything(self):
        self.assertEqual(self.names(rlt.choose(self.configs, [], None)[0]), ["alpha-verifier", "beta-verifier", "gamma"])

    def test_by_name_with_or_without_json(self):
        self.assertEqual(self.names(rlt.choose(self.configs, ["gamma.json", "alpha-verifier"], None)[0]), ["gamma", "alpha-verifier"])

    def test_a_name_given_twice_runs_once(self):
        self.assertEqual(self.names(rlt.choose(self.configs, ["gamma", "gamma"], None)[0]), ["gamma"])

    def test_an_unknown_name_lists_the_configs(self):
        chosen, error = rlt.choose(self.configs, ["nope"], None)
        self.assertEqual(chosen, [])
        self.assertIn("alpha-verifier, beta-verifier, gamma", error)

    def test_only_is_a_substring_filter(self):
        self.assertEqual(self.names(rlt.choose(self.configs, [], "VERIFIER")[0]), ["alpha-verifier", "beta-verifier"])
        self.assertIn("no config name contains 'zzz'", rlt.choose(self.configs, [], "zzz")[1])


class Main(Folder):
    """`main` with `configs/` pointed at a temp folder. Nothing here reaches LM Studio or runs a script."""

    def run_main(self, *argv):
        out = io.StringIO()
        original = rlt.CONFIGS
        rlt.CONFIGS = self.dir
        try:
            # `main` would make a throwaway reviewer-database folder and set the variables for the scripts it starts; neither belongs
            # to these tests (tests/test_harness_reviewer_db.py covers it), so the helper is a no-op: nothing is left in the temp folder.
            with contextlib.redirect_stdout(out), unittest.mock.patch.object(h, "isolate_reviewer_db"):
                code = rlt.main(list(argv))
        finally:
            rlt.CONFIGS = original
        return code, out.getvalue()

    def test_list_shows_stages_models_script_and_baselines(self):
        write_baselines(write_config(self.dir, "with-bars", ["verifier"], model="m1"), {"min_real_confirmed": 4}, measured="2026-10-27", model="m1")
        write_config(self.dir, "no-bars", ["reasoning", "verifier"], roles=("reasoning", "verifier"))
        code, text = self.run_main("--list")
        self.assertEqual(code, 0)
        self.assertIn("with-bars: harness/live_verifier.py", text)
        self.assertIn("baseline measured 2026-10-27 with m1", text)
        self.assertIn("no-bars: harness/live_all_reviewers.py", text)
        self.assertIn("--case orders", text)
        self.assertIn("none (mechanics checks only)", text)

    def test_list_marks_invalid_and_not_runnable_configs_with_the_reason(self):
        (self.dir / "broken.json").write_text("{", encoding="utf-8")
        write_config(self.dir, "later", ["verifier", "classifier"], roles=("verifier", "classifier"))
        code, text = self.run_main("--list")
        self.assertEqual(code, 0)
        self.assertIn("broken: INVALID", text)
        self.assertIn("later: NOT RUNNABLE", text)

    def test_list_selects_by_name(self):
        write_config(self.dir, "one", ["verifier"])
        write_config(self.dir, "two", ["verifier"])
        _, text = self.run_main("--list", "two")
        self.assertIn("two:", text)
        self.assertNotIn("one:", text)

    def test_an_unknown_name_exits_1(self):
        write_config(self.dir, "one", ["verifier"])
        code, text = self.run_main("nope")
        self.assertEqual(code, 1)
        self.assertIn("no config named ['nope']", text)

    def test_an_empty_folder_runs_nothing_and_exits_1(self):
        code, text = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("a config there is a live test", text)

    def test_naming_a_config_that_cannot_run_exits_1_before_any_check_or_model(self):
        write_config(self.dir, "later", ["verifier", "classifier"], roles=("verifier", "classifier"))
        code, text = self.run_main("later")
        self.assertEqual(code, 1)
        self.assertIn("NOT RUN later: no live test script runs the stages", text)
        self.assertNotIn("preflight", text)

    def test_an_unused_verifier_model_that_is_not_downloaded_does_not_stop_a_reasoning_run(self):
        data = {"stages": ["reasoning"], "reviewers": [reviewer("r", "reasoning", "present-model"), reviewer("v", "verifier", "not-downloaded")]}
        (self.dir / "reasoning-only.json").write_text(json.dumps(data), encoding="utf-8")
        listed, ran = {"present-model"}, []
        originals = (rlt.lmstudio_models, rlt.run_script)
        rlt.lmstudio_models = lambda url: listed
        rlt.run_script = lambda config, runs, timeout, log, env: (ran.append(config.name) or (0, "loaded before: ['x']\nloaded after:  ['x']\n"))
        try:
            code, text = self.run_main("--skip-checks")
        finally:
            rlt.lmstudio_models, rlt.run_script = originals
        self.assertEqual(ran, ["reasoning-only"], text)
        self.assertNotIn("does not list", text)
        self.assertEqual(code, 0, text)

    def test_unreachable_lmstudio_exits_3_and_loads_nothing(self):
        write_config(self.dir, "one", ["verifier"])
        previous = os.environ.get("CLAUDE_RUNWAY_LMSTUDIO_URL")
        os.environ["CLAUDE_RUNWAY_LMSTUDIO_URL"] = "http://127.0.0.1:9/v1"
        try:
            code, text = self.run_main("--skip-checks")
        finally:
            if previous is None:
                del os.environ["CLAUDE_RUNWAY_LMSTUDIO_URL"]
            else:
                os.environ["CLAUDE_RUNWAY_LMSTUDIO_URL"] = previous
        self.assertEqual(code, rlt.EXIT_NOT_RUN)
        self.assertIn("LM Studio is not reachable", text)


class ShippedConfigs(unittest.TestCase):
    """The real `configs/` folder: what a person looking at it would pick from."""

    def setUp(self):
        self.configs = {p.stem: lc.describe(p) for p in lc.discover(lc.CONFIGS)}

    def test_every_shipped_sidecar_has_its_config_beside_it(self):
        # Config names carry the build format (`reasoning-mlx-...`, `reasoning-gguf-...`), so renaming a config is a normal
        # change, and a rename that forgets the sidecar would otherwise drop that config's bars without a word.
        self.assertEqual(orphan_sidecars(lc.CONFIGS), [])

    def test_every_shipped_config_is_runnable(self):
        self.assertTrue(self.configs)
        for name, live in self.configs.items():
            with self.subTest(config=name):
                self.assertEqual(live.problem, "")
                self.assertIsNotNone(live.kind)

    def test_the_measured_baselines_are_the_ones_the_tests_were_written_with(self):
        verifier = self.configs["verifier-mlx-deepseek-7b"]
        self.assertEqual(verifier.kind.script, "live_verifier.py")
        self.assertEqual(verifier.baselines.expect, {"min_real_confirmed": 4, "max_false_confirmed": 0})
        self.assertEqual(verifier.baselines.measured, "2026-10-27")
        pipeline = self.configs["pipeline-mlx-qwen-7b-deepseek-7b"]
        self.assertEqual(pipeline.kind.script, "live_all_reviewers.py")
        self.assertEqual(pipeline.kind.args, ("--case", "orders"))
        self.assertEqual(pipeline.baselines.expect, {"max_decoy_alarms_after": 0, "min_bugs_after": 1})

    def test_the_classifier_configs_run_the_classifier_script_and_the_three_stage_pipeline_and_hold_only_mechanics(self):
        alone = self.configs["classifier-mlx-qwen-7b-4bit"]
        self.assertEqual((alone.stages, alone.kind.script), (("classifier",), "live_classifier.py"))
        pipeline = self.configs["pipeline-mlx-qwen-7b-deepseek-7b-classifier-qwen-7b"]
        self.assertEqual((pipeline.stages, pipeline.kind.name, pipeline.kind.script),
                         (("reasoning", "verifier", "classifier"), "pipeline", "live_all_reviewers.py"))
        for live in (alone, pipeline):
            self.assertIsNone(live.baselines)  # one measurement is not a bar: see the README

    def test_every_shipped_sidecar_has_a_run_type_and_metadata_and_the_runner_passes_its_cases(self):
        sidecars = sorted(lc.CONFIGS.glob("*" + lc.BASELINES_SUFFIX))
        self.assertTrue(sidecars)
        for path in sidecars:
            with self.subTest(sidecar=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn(data.get("runType"), ("synthetic", "live"))
                self.assertIsInstance(data.get("metadata"), dict)
                live = self.configs[path.name[: -len(lc.BASELINES_SUFFIX)]]
                if data["runType"] == "live":
                    # A live sidecar's metadata is the answer key (tags with bugs and decoys), not a case list: the harness's
                    # own parser must accept it, and the config must load. There are no `--case` arguments to compare.
                    import live_branch_lib

                    self.assertEqual(live.problem, "")
                    self.assertEqual(live.baselines.run_type, "live")
                    self.assertTrue(live_branch_lib.parse_live_metadata(data["metadata"]))
                    continue
                self.assertEqual(live.problem, "")  # loads, validates, and its script can take its cases
                self.assertEqual(live.baselines.run_type, "synthetic")
                passed = lc.script_args(live)
                self.assertEqual(passed[0::2], ("--case",) * (len(passed) // 2))
                self.assertEqual(list(passed[1::2]), lc.expected_cases(live.kind, live.baselines))
                self.assertEqual(list(passed[1::2]), data["metadata"]["cases"])

    def test_the_verifier_and_pipeline_bars_were_counted_on_orders_and_the_reasoning_numbers_on_every_case(self):
        for name in ("verifier-mlx-deepseek-7b", "pipeline-mlx-qwen-7b-deepseek-7b"):
            self.assertEqual(self.configs[name].baselines.cases, ("orders",), name)
            self.assertEqual(lc.script_args(self.configs[name]), ("--case", "orders"), name)
        reasoning = [live for name, live in self.configs.items() if name.startswith("reasoning-") and live.baselines]
        self.assertEqual(len(reasoning), 7)
        for live in reasoning:
            # Spelled out, not ["all"]: a case added to cases.py later must not change what these numbers claim.
            self.assertEqual(live.baselines.cases, tuple(ALL_CASES), live.name)


class PreflightChecks(Folder):
    def test_one_check_per_config_run_with_the_script_it_will_use(self):
        write_config(self.dir, "v", ["verifier"])
        write_config(self.dir, "p", ["reasoning", "verifier"], roles=("reasoning", "verifier"))
        write_baselines(self.dir / "v.json", {"min_real_confirmed": 1})
        checks = run_checks.live_config_checks(self.dir)
        self.assertEqual([c.script for c in checks], ["live_all_reviewers.py", "live_verifier.py"])
        for check in checks:
            self.assertIn("--show-config", check.args)
            self.assertEqual(check.exit_code, 0)

    def test_a_config_no_script_runs_and_a_broken_one_are_still_checked(self):
        write_config(self.dir, "later", ["verifier", "classifier"], roles=("verifier", "classifier"))
        (self.dir / "broken.json").write_text("{", encoding="utf-8")
        checks = run_checks.live_config_checks(self.dir)
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(c.script == "live_all_reviewers.py" for c in checks))

    def test_the_shipped_checks_cover_every_shipped_config(self):
        covered = {a.rsplit("/", 1)[-1] for c in run_checks.CHECKS for a in c.args if a.startswith("{live_configs}")}
        self.assertEqual(covered, {p.name for p in lc.discover(lc.CONFIGS)})

    def test_a_config_added_to_the_folder_is_checked_with_no_edit_to_the_preflight(self):
        self.assertEqual(len(run_checks.live_config_checks(self.dir)), 0)
        write_config(self.dir, "new", ["verifier"])
        self.assertEqual(len(run_checks.live_config_checks(self.dir)), 1)


if __name__ == "__main__":
    unittest.main()
