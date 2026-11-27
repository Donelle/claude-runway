#!/usr/bin/env python3
"""Tests for `libs/reviewer_orchestrator_lib.py` and the `review` tool running through it.

What they pin down:
  * the orchestrator with the rules policy returns, for each review type, exactly the text (and the recorded notes and
    counts) the old `run_*_review` + `format_*` path returns on the same stubbed inputs;
  * a review pins `head_ref` to one commit for EVERY type, reasoning included;
  * the loop's invariants hold against a stub policy: `verify` before `ground` and `classify` before `verify` are refused
    and recorded, targets outside what a step may take are refused, the step cap ends the run with a note, and the
    recorder closes after a step raises.

Stdlib-only (unittest), no network, no LM Studio: the model calls are stubs, LM Studio's REST API is `tests/fake_lmstudio.py`.

    .venv/bin/python -m unittest discover -s tests
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import reviewer_orchestrator_lib as orch  # noqa: E402
from classifier_reviewer_lib import format_classified_review, run_classified_review  # noqa: E402
from codebase_reviewer_config_lib import Reviewer, ReviewersConfig, RubricEntry  # noqa: E402
from fake_lmstudio import FakeLmStudio  # noqa: E402
from lmstudio_model_loader_lib import ModelLoader, ModelLoadError  # noqa: E402
from reasoning_reviewer_lib import Finding, ReviewError, format_review, run_reasoning_review  # noqa: E402
from reviewer_db_lib import NullRecorder  # noqa: E402
from reviewer_files_lib import FindingsInput  # noqa: E402
from test_reviewer_recording import FakeClient, IsolatedEnv  # noqa: E402
from verifier_reviewer_lib import format_verified_review, run_verified_review  # noqa: E402

BASE = "http://localhost:1234/v1"
SERVER_PATH = REPO_ROOT / "mcpservers" / "codebase_reviewer_mcp_server.py"
HEAD = "0" * 40

DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n+x = 1\n+y = 2\n"
FILE_TEXT = "x = 1\ny = 2\n"


def finding(claim, quote, cls="output", line=1):
    return {"file": "a.py", "line_start": line, "line_end": line, "class": cls, "claim": claim, "failing_input": "any", "quote": quote}


TWO_FINDINGS = json.dumps({"findings": [finding("first constant", "x = 1"), finding("second constant", "y = 2", line=2)]})
VERDICT = json.dumps({"verdict": "confirmed", "failing_input": "any", "wrong_result": "1", "reasoning": "yes"})
RUBRIC = (RubricEntry("injection", "high", "input reaches a shell", 0.6),)


def complete(client, model, system, text):
    """The stub model call the reasoning and verifier stages share: the verifier's instructions tell the two apart."""
    return VERDICT if "checking ONE reported defect" in system else TWO_FINDINGS


def probability(client, model, system, content):
    return 0.9


def reviewer(name, model="model-a", roles=("reasoning",), on_demand=False):
    return Reviewer(name=name, model=model, prompt="look at it", active=True, roles=tuple(roles), load_on_demand=on_demand,
                    context_length=None, reply_format="json")


def one_model_config():
    return ReviewersConfig(rubric=RUBRIC, reviewers=(reviewer("all", roles=("reasoning", "verifier", "classifier")),))


class SpyRecorder(NullRecorder):
    """What a run told its recorder: the notes, the counts, the sizes and how it was closed."""

    def __init__(self):
        self.notes: list = []
        self.counts: dict = {}
        self.sizes: tuple = ()
        self.result_text = ""
        self.closed: list = []

    def note(self, stage, text, *, reviewer="", detail=()):
        self.notes.append((stage, text, tuple(detail)))

    def set_counts(self, **counts):
        self.counts.update(counts)

    def set_sizes(self, diff_chars=None, result_chars=None):
        self.sizes = (diff_chars, result_chars)

    def set_result_text(self, text):
        self.result_text = text

    def close(self, outcome="finished", *, error=None):
        self.closed.append((outcome, error))


def loader(*loaded):
    return ModelLoader(BASE, http=FakeLmStudio(loaded=loaded or ["model-a"]))


def state_for(config=None, recorder=None, lmstudio=None, **overrides):
    values = dict(
        config=config or one_model_config(), loader=lmstudio or loader(), client=object(), repo_root=Path("."), head=HEAD,
        diff=DIFF, max_reviewers=1, recorder=recorder or SpyRecorder(), complete=complete, probability=probability,
        read_file=lambda path: FILE_TEXT,
    )
    values.update(overrides)
    return orch.RunState(**values)


class ScriptedPolicy:
    """A stub policy: each item is an Action, or a function of the state returning one; `finish` once they run out."""

    def __init__(self, *script, preflight_roles=()):
        self._script = list(script)
        self.preflight_roles = preflight_roles
        self.asked = 0

    def next_action(self, state):
        self.asked += 1
        if not self._script:
            return orch.Action(orch.FINISH)
        item = self._script.pop(0)
        return item(state) if callable(item) else item


def A(name, targets=None):
    return orch.Action(name, targets)


class SameAsTheOldPath(unittest.TestCase):
    """The same stubbed inputs through the old `run_*` path and through the orchestrator give byte-identical text, and the
    run's record gets the same notes."""

    def old_path(self, review_type):
        rec = SpyRecorder()
        common = dict(complete=complete, recorder=rec)
        config = one_model_config()
        if review_type == "reasoning":
            text = format_review(run_reasoning_review(config, loader(), object(), DIFF, Path("."), 1, **common))
        elif review_type == "verifier":
            text = format_verified_review(
                run_verified_review(config, loader(), object(), DIFF, Path("."), HEAD, 1, read_file=lambda p: FILE_TEXT, **common))
        else:
            text = format_classified_review(run_classified_review(
                config, loader(), object(), DIFF, Path("."), HEAD, 1, read_file=lambda p: FILE_TEXT, probability=probability, **common))
        return text, rec

    def new_path(self, review_type):
        rec = SpyRecorder()
        text = orch.run_orchestrated(state_for(recorder=rec), orch.RulesPolicy(review_type))
        return text, rec

    def check(self, review_type):
        old_text, old_rec = self.old_path(review_type)
        new_text, new_rec = self.new_path(review_type)
        self.assertEqual(new_text, old_text)
        self.assertEqual(new_rec.notes, old_rec.notes)
        self.assertEqual(new_rec.result_text, new_text)
        self.assertEqual(new_rec.closed, [("finished", None)])
        return new_text, new_rec

    def test_reasoning(self):
        text, rec = self.check("reasoning")
        self.assertIn("2 finding(s)", text)
        self.assertEqual((rec.counts["findings"], "confirmed" in rec.counts), (2, False))
        self.assertEqual(rec.sizes, (len(DIFF), len(text)))

    def test_verifier(self):
        text, rec = self.check("verifier")
        self.assertIn("2 confirmed", text)
        self.assertEqual(rec.counts["confirmed"], 2)
        self.assertTrue(any(stage == "grounding" for stage, _, _ in rec.notes))

    def test_classifier(self):
        text, rec = self.check("classifier")
        self.assertIn("Classified review: 2 classified", text)
        self.assertEqual(rec.counts["confirmed"], 2)

    def test_the_rules_policy_runs_each_type_in_its_old_order(self):
        for review_type, steps in (("reasoning", ["reason"]), ("verifier", ["reason", "ground", "verify"]),
                                   ("classifier", ["reason", "ground", "verify", "classify"])):
            with self.subTest(review_type):
                state = state_for()
                orch.run_orchestrated(state, orch.RulesPolicy(review_type))
                self.assertEqual(state.done, steps)

    def test_an_unknown_review_type_is_not_a_policy(self):
        with self.assertRaises(ValueError):
            orch.RulesPolicy("guess")


class ARunNeedsNoRecorder(unittest.TestCase):
    """`RunState`'s default recorder is the null one, and the loop closes the recorder at the end of every run. Every other
    test passes a spy, which defines its own `close`, so these are the only ones that use the default."""

    def default_state(self, config=None):
        return orch.RunState(
            config=config or one_model_config(), loader=loader(), client=object(), repo_root=Path("."), head=HEAD, diff=DIFF,
            max_reviewers=1, complete=complete, probability=probability, read_file=lambda path: FILE_TEXT,
        )

    def test_a_successful_review_returns_its_text(self):
        text = orch.run_orchestrated(self.default_state(), orch.RulesPolicy("reasoning"))
        self.assertIn("2 finding(s)", text)

    def test_an_unreadable_reply_opens_every_review_type_with_the_failure_line(self):
        # The orchestrated path (what every `type`, pipeline included, goes through) puts the failure first.
        for review_type, opening in (("reasoning", "Reasoning review:"), ("verifier", "Verified review:"),
                                     ("classifier", "Classified review:")):
            state = self.default_state()
            state.complete = lambda client, model, system, text: "this is not a review"
            lines = orch.run_orchestrated(state, orch.RulesPolicy(review_type)).splitlines()
            self.assertTrue(lines[0].startswith("INCOMPLETE REVIEW: "), (review_type, lines[0]))
            self.assertTrue(lines[1].startswith(opening), (review_type, lines[1]))

    def test_a_failing_review_raises_its_own_error_not_an_attribute_error(self):
        config = ReviewersConfig(rubric=RUBRIC, reviewers=(reviewer("v", roles=("verifier",)),))  # no reasoning reviewer
        with self.assertRaises(ReviewError) as raised:
            orch.run_orchestrated(self.default_state(config), orch.RulesPolicy("reasoning"))
        self.assertIn("no active reasoning reviewer", str(raised.exception))

    def test_a_crash_inside_a_step_raises_the_crash(self):
        def broken(path):
            raise RuntimeError("disk gone")
        state = self.default_state()
        state.read_file = broken
        with self.assertRaises(RuntimeError) as raised:
            orch.run_orchestrated(state, orch.RulesPolicy("verifier"))
        self.assertEqual(str(raised.exception), "disk gone")


class ModelsAreCheckedBeforeAnyStep(unittest.TestCase):
    """The resident models are checked up front, classifier's first, so a model that is not loaded is reported at once."""

    config = ReviewersConfig(rubric=RUBRIC, reviewers=(
        reviewer("r"), reviewer("v", model="model-v", roles=("verifier",)), reviewer("c", model="model-c", roles=("classifier",))))

    def run_type(self, review_type):
        state = state_for(config=self.config, lmstudio=loader("model-a"))
        calls = []
        state.complete = lambda *a: calls.append(a) or TWO_FINDINGS
        with self.assertRaises(ModelLoadError) as raised:
            orch.run_orchestrated(state, orch.RulesPolicy(review_type))
        self.assertEqual(calls, [], "no reasoning request may be sent before the models are known to be loaded")
        return str(raised.exception), state

    def test_the_classifier_type_names_the_classifier_model_first(self):
        message, state = self.run_type("classifier")
        self.assertIn("'model-c'", message)
        self.assertNotIn("'model-v'", message)
        self.assertEqual(state.recorder.closed[0][0], "refused")

    def test_the_verifier_type_names_the_verifier_model(self):
        message, _ = self.run_type("verifier")
        self.assertIn("'model-v'", message)

    def test_the_reasoning_type_checks_nothing_extra(self):
        state = state_for(config=self.config, lmstudio=loader("model-a"))
        orch.run_orchestrated(state, orch.RulesPolicy("reasoning"))
        self.assertEqual(state.done, ["reason"])


class TheLoopEnforcesTheInvariants(unittest.TestCase):
    def run_script(self, *script, **kwargs):
        state = state_for()
        policy = ScriptedPolicy(*script)
        text = orch.run_orchestrated(state, policy, **kwargs)
        return state, text

    def refused_notes(self, state):
        return [text for stage, text, _ in state.recorder.notes if stage == orch.NOTE_STAGE]

    def test_verify_before_ground_is_refused_and_recorded(self):
        state, text = self.run_script(A("reason"), A("verify"))
        self.assertEqual(state.done, ["reason"])
        self.assertIsNone(state.verification)
        self.assertTrue(any("verify before ground is not allowed" in n for n in self.refused_notes(state)))
        self.assertIn("refused an action", text)  # the caller sees it, as a warning
        self.assertEqual(state.recorder.closed, [("finished", None)])

    def test_classify_before_verify_is_refused_and_recorded(self):
        state, text = self.run_script(A("reason"), A("ground"), A("classify"))
        self.assertEqual(state.done, ["reason", "ground"])
        self.assertIsNone(state.classification)
        self.assertTrue(any("classify before verify is not allowed" in n for n in self.refused_notes(state)))

    def test_ground_before_reason_is_refused_and_a_run_that_did_nothing_is_an_error(self):
        state = state_for()
        with self.assertRaises(ReviewError) as raised:
            orch.run_orchestrated(state, ScriptedPolicy(A("ground")))
        self.assertIn("ended before any stage ran", str(raised.exception))
        self.assertEqual(state.recorder.closed[0][0], "refused")

    def test_a_step_cannot_run_twice(self):
        state, _ = self.run_script(A("reason"), A("reason"))
        self.assertEqual(state.done, ["reason"])
        self.assertTrue(any("already ran" in n for n in self.refused_notes(state)))

    def test_an_unknown_action_and_a_non_action_are_refused(self):
        for action in (A("think"), "reason"):
            with self.subTest(action=action):
                state, _ = self.run_script(A("reason"), action)
                self.assertEqual(state.done, ["reason"])
                self.assertEqual(len(self.refused_notes(state)), 1)

    def test_targets_that_are_not_eligible_are_refused(self):
        from reasoning_reviewer_lib import Finding
        stranger = Finding(file="z.py", line_start=1, line_end=1, cls="output", claim="elsewhere", quote="z", failing_input="", reviewers=("all",))
        state, _ = self.run_script(A("reason"), lambda s: A("ground", (stranger,)))
        self.assertEqual(state.done, ["reason"])
        self.assertTrue(any("not findings it may take" in n for n in self.refused_notes(state)))

    def test_every_eligible_finding_as_targets_is_allowed_in_any_order(self):
        state, _ = self.run_script(A("reason"), lambda s: A("ground", tuple(reversed(s.findings))))
        self.assertEqual(state.done, ["reason", "ground"])
        self.assertEqual(len(state.grounded), 2)

    def test_a_subset_of_the_eligible_findings_is_refused_for_now(self):
        state, _ = self.run_script(A("reason"), A("ground"), lambda s: A("verify", tuple(s.grounded[:1])))
        self.assertEqual(state.done, ["reason", "ground"])
        self.assertTrue(any("a subset is not supported yet" in n for n in self.refused_notes(state)))

    def test_verify_may_only_take_grounded_findings(self):
        # The second finding quotes code that is not in the file, so it is dropped at `ground` and is no target for `verify`.
        def reply(client, model, system, text):
            if "checking ONE reported defect" in system:
                return VERDICT
            return json.dumps({"findings": [finding("real", "x = 1"), finding("invented", "z = 9", line=2)]})
        state = state_for(complete=reply)
        orch.run_orchestrated(state, ScriptedPolicy(A("reason"), A("ground"), lambda s: A("verify", tuple(s.findings))))
        self.assertEqual(state.done, ["reason", "ground"])
        self.assertEqual(len(state.ungrounded), 1)

    def grounded_only_state(self, *script, **kwargs):
        """A run over one real finding (its claim names line 1, its quote is on line 2) and one invented finding."""
        def reply(client, model, system, text):
            return json.dumps({"findings": [finding("real constant", "y = 2", line=1), finding("invented", "z = 9", line=2)]})
        state = state_for(complete=reply)
        return state, orch.run_orchestrated(state, ScriptedPolicy(*script), **kwargs)

    def check_grounding_result_survives(self, state, text):
        self.assertEqual(state.done, ["reason", "ground"])
        self.assertTrue(text.startswith("Verified review:"), text)
        self.assertIn("No verifier ran, so these findings are GROUNDED", text)
        # The finding grounding dropped is reported as dropped, with its reason, and not as an ordinary finding...
        self.assertIn("## Dropped: ungrounded (1)", text)
        self.assertIn("ungrounded: its quote is not in a.py", text)
        self.assertNotIn("## output", text)
        # ...and the surviving one is unchecked, at the line grounding corrected it to (the quote is on line 2).
        self.assertIn("## Unchecked (1)", text)
        self.assertIn("- a.py:2 [output] [all] real constant", text)
        self.assertNotIn("a.py:1", text)
        self.assertIn("not checked: the run ended before the verifiers were asked", text)

    def test_a_run_that_ends_after_ground_by_the_step_cap_keeps_the_grounding_result(self):
        state, text = self.grounded_only_state(A("reason"), A("ground"), A("verify"), max_steps=2)
        self.check_grounding_result_survives(state, text)
        self.assertIn("step cap (2)", text)

    def test_a_run_that_ends_after_ground_by_a_refused_action_keeps_the_grounding_result(self):
        state, text = self.grounded_only_state(A("reason"), A("ground"), lambda s: A("verify", tuple(s.grounded[:0])))
        self.check_grounding_result_survives(state, text)
        self.assertIn("refused an action", text)

    def test_a_run_that_ends_after_ground_with_nothing_grounded_says_what_was_dropped(self):
        def reply(client, model, system, text):
            return json.dumps({"findings": [finding("invented", "z = 9")]})
        state = state_for(complete=reply)
        text = orch.run_orchestrated(state, ScriptedPolicy(A("reason"), A("ground")))
        self.assertIn("dropped 0 rejected and 1 ungrounded", text)
        self.assertIn("No verifier ran.", text)
        self.assertEqual(state.recorder.counts["confirmed"], 0)

    def test_the_step_cap_ends_the_run_with_a_note(self):
        state, text = self.run_script(A("reason"), A("ground"), A("verify"), A("classify"), max_steps=2)
        self.assertEqual(state.done, ["reason", "ground"])
        self.assertIn("step cap (2)", text)
        self.assertTrue(any("step cap" in n for n in self.refused_notes(state)))
        self.assertEqual(state.recorder.closed, [("finished", None)])

    def test_a_policy_that_finishes_exactly_at_the_cap_is_not_cut_short(self):
        state, text = self.run_script(A("reason"), A("ground"), max_steps=2)
        self.assertEqual(state.done, ["reason", "ground"])
        self.assertNotIn("step cap", text)

    def test_the_recorder_closes_as_errored_after_a_step_raises(self):
        def broken(path):
            raise RuntimeError("disk gone")
        state = state_for(read_file=broken)
        with self.assertRaises(RuntimeError):
            orch.run_orchestrated(state, orch.RulesPolicy("verifier"))
        self.assertEqual(state.recorder.closed, [("errored", "RuntimeError: disk gone")])

    def test_the_recorder_closes_as_refused_after_a_review_error(self):
        config = ReviewersConfig(rubric=RUBRIC, reviewers=(reviewer("v", roles=("verifier",)),))  # no reasoning reviewer
        state = state_for(config=config)
        with self.assertRaises(ReviewError):
            orch.run_orchestrated(state, orch.RulesPolicy("reasoning"))
        self.assertEqual(state.recorder.closed[0][0], "refused")
        self.assertIn("no active reasoning reviewer", state.recorder.closed[0][1])


def supplied_finding(claim, quote, line, cls="output") -> Finding:
    """A finding as a findings file would hold it: raised by `finder`, with the code it accuses."""
    return Finding(file="a.py", line_start=line, line_end=line, cls=cls, claim=claim, failing_input="any", reviewers=("finder",), quote=quote)


class StageRunsAloneOnSuppliedFindings(unittest.TestCase):
    """The verifier and the classifier start from findings the caller supplied: the stages before them do not run."""

    def asked(self, seen):
        def reply(client, model, system, text):
            kind = "verifier" if "checking ONE reported defect" in system else "reasoning"
            seen.append(kind)
            return VERDICT if kind == "verifier" else TWO_FINDINGS
        return reply

    def verifier_config(self):
        return ReviewersConfig(rubric=RUBRIC, reviewers=(reviewer("finder"), reviewer("v", roles=("verifier",))))

    def test_the_verifier_runs_grounding_and_the_verifiers_and_never_the_reasoning_stage(self):
        seen: list = []
        state = state_for(config=self.verifier_config(), complete=self.asked(seen))
        orch.supply_findings(state, FindingsInput((supplied_finding("first constant", "x = 1", 1), supplied_finding("invented", "z = 9", 2))),
                             "verifier")
        text = orch.run_orchestrated(state, orch.RulesPolicy("verifier"))
        self.assertEqual(state.supplied, ("reason",))
        self.assertEqual(state.done, ["ground", "verify"])  # `done` lists only what ran
        self.assertEqual(seen, ["verifier"])  # one grounded finding, one verifier: no reasoning request at all
        self.assertTrue(text.startswith("Verified review: 1 confirmed, 0 disputed, 0 unchecked; dropped 0 rejected and 1 ungrounded"), text)
        self.assertIn("(of 2 claim(s) from 1 reasoning reviewer(s): finder)", text)
        self.assertIn("Warnings:\n- The findings were read from a findings file: the reasoning stage did not run.", text)
        self.assertIn("ungrounded: its quote is not in a.py", text)

    def test_a_run_from_supplied_findings_records_the_findings_read_and_claims_no_diff_saving(self):
        state = state_for(config=self.verifier_config(), complete=self.asked([]))
        orch.supply_findings(state, FindingsInput((supplied_finding("first constant", "x = 1", 1),)), "verifier")
        text = orch.run_orchestrated(state, orch.RulesPolicy("verifier"))
        rec = state.recorder
        self.assertEqual(rec.counts, {"chunks": 0, "findings": 1, "confirmed": 1})
        # No model read the diff, so the savings record (diff characters against result characters) is left out: None, not len(DIFF).
        self.assertEqual(rec.sizes, (None, len(text)))
        self.assertEqual(rec.closed, [("finished", None)])
        self.assertTrue(any(stage == "grounding" for stage, _, _ in rec.notes))

    def test_a_full_run_still_records_the_diff_size(self):
        state = state_for(complete=self.asked([]))
        text = orch.run_orchestrated(state, orch.RulesPolicy("verifier"))
        self.assertEqual(state.recorder.sizes, (len(DIFF), len(text)))
        self.assertEqual(state.supplied, ())

    def test_the_findings_are_grounded_as_given_nothing_is_merged_first(self):
        # Two findings on the same line of one class would merge in the reasoning stage; supplied, they are two findings.
        state = state_for(config=self.verifier_config(), complete=self.asked([]))
        orch.supply_findings(state, FindingsInput((supplied_finding("one claim", "x = 1", 1), supplied_finding("another claim", "x = 1", 1))),
                             "verifier")
        orch.run_orchestrated(state, orch.RulesPolicy("verifier"))
        self.assertEqual([f.claim for f in state.grounded], ["one claim", "another claim"])

    def test_the_classifier_runs_only_the_classifiers_on_the_findings_with_the_status_each_arrived_with(self):
        def refuse(client, model, system, text):
            raise AssertionError("no reasoning or verifier request may be sent")
        calls: list = []

        def asked_probability(client, model, system, content):
            calls.append(content)
            return 0.9

        state = state_for(complete=refuse, probability=asked_probability)
        orch.supply_findings(state, FindingsInput((supplied_finding("first constant", "x = 1", 1), supplied_finding("second", "y = 2", 2)),
                                                  ("confirmed", "disputed")), "classifier")
        text = orch.run_orchestrated(state, orch.RulesPolicy("classifier"))
        self.assertEqual((state.supplied, state.done), (("reason", "ground", "verify"), ["classify"]))
        self.assertEqual(len(calls), 2)  # two findings, one rubric entry
        self.assertTrue(text.startswith("Classified review: 2 classified"), text)
        self.assertIn("first constant (verifier: confirmed)", text)
        self.assertIn("second (verifier: disputed)", text)
        self.assertIn("Each finding's verification status is the one its findings file gave: no verifier ran in this review.", text)
        self.assertNotIn("No verifier ran, so the findings are grounded", text)
        self.assertIn("the reasoning and verifier stages did not run", text)
        self.assertTrue(state.verification.supplied)
        self.assertEqual(state.recorder.sizes, (None, len(text)))

    def test_the_classifier_needs_no_reasoning_or_verifier_model_to_be_loaded(self):
        # A full classifier review checks the verifier's models first (and refuses when one is missing); a run that supplies the
        # findings never uses them, so only the classifier's own model has to be there.
        config = ReviewersConfig(rubric=RUBRIC, reviewers=(
            reviewer("r"), reviewer("v", model="model-v", roles=("verifier",)), reviewer("c", model="model-c", roles=("classifier",))))
        state = state_for(config=config, lmstudio=loader("model-c"))
        orch.supply_findings(state, FindingsInput((supplied_finding("first constant", "x = 1", 1),), ("confirmed",)), "classifier")
        text = orch.run_orchestrated(state, orch.RulesPolicy("classifier"))
        self.assertTrue(text.startswith("Classified review: 1 classified"), text)

    def test_the_verifier_whose_model_is_not_loaded_is_still_refused_before_any_request(self):
        seen: list = []
        config = ReviewersConfig(rubric=RUBRIC, reviewers=(reviewer("v", model="model-v", roles=("verifier",)),))  # no reasoning reviewer
        state = state_for(config=config, lmstudio=loader("model-a"), complete=self.asked(seen))
        orch.supply_findings(state, FindingsInput((supplied_finding("first constant", "x = 1", 1),)), "verifier")
        with self.assertRaises(ModelLoadError) as raised:
            orch.run_orchestrated(state, orch.RulesPolicy("verifier"))
        self.assertIn("'model-v'", str(raised.exception))
        self.assertEqual(seen, [])
        self.assertEqual(state.recorder.closed[0][0], "refused")

    def test_no_reasoning_reviewer_is_needed_when_the_findings_are_supplied(self):
        config = ReviewersConfig(rubric=RUBRIC, reviewers=(reviewer("v", roles=("verifier",)),))
        state = state_for(config=config)
        orch.supply_findings(state, FindingsInput((supplied_finding("first constant", "x = 1", 1),)), "verifier")
        self.assertIn("1 confirmed", orch.run_orchestrated(state, orch.RulesPolicy("verifier")))
        # ...and without them the same configuration is still refused, as before.
        with self.assertRaises(ReviewError):
            orch.run_orchestrated(state_for(config=config), orch.RulesPolicy("verifier"))

    def test_a_step_whose_result_was_supplied_is_refused_if_a_policy_asks_for_it(self):
        state = state_for(config=self.verifier_config())
        orch.supply_findings(state, FindingsInput((supplied_finding("first constant", "x = 1", 1),)), "verifier")
        text = orch.run_orchestrated(state, ScriptedPolicy(A("reason")))
        self.assertEqual(state.done, [])
        self.assertIn("reason is not run: its result came with the findings the caller supplied", text)
        state = state_for(config=self.verifier_config())
        orch.supply_findings(state, FindingsInput((supplied_finding("first constant", "x = 1", 1),)), "verifier")
        orch.run_orchestrated(state, ScriptedPolicy(A("verify")))  # still not before ground
        self.assertTrue(any("verify before ground is not allowed" in text for stage, text, _ in state.recorder.notes))

    def test_a_supplied_step_counts_as_done_for_the_step_that_needs_it(self):
        state = state_for(config=self.verifier_config())
        orch.supply_findings(state, FindingsInput((supplied_finding("first constant", "x = 1", 1),)), "verifier")
        self.assertIsNone(orch.refusal(state, A("ground")))
        classifier_state = state_for()
        orch.supply_findings(classifier_state, FindingsInput((supplied_finding("first constant", "x = 1", 1),), ("confirmed",)), "classifier")
        self.assertIsNone(orch.refusal(classifier_state, A("classify")))
        self.assertIn("is not run", orch.refusal(classifier_state, A("verify")) or "")

    def test_reasoning_has_no_earlier_stage_to_replace(self):
        with self.assertRaises(ValueError):
            orch.supply_findings(state_for(), FindingsInput(()), "reasoning")

    def test_with_nothing_supplied_a_run_is_what_it_was(self):
        state = state_for()
        orch.run_orchestrated(state, orch.RulesPolicy("classifier"))
        self.assertEqual((state.supplied, state.done), ((), ["reason", "ground", "verify", "classify"]))
        self.assertFalse(state.verification.supplied)


class StructuredResultsOfARun(unittest.TestCase):
    """`results_of`: one section for each stage that really ran, built from the run's own state and JSON all the way down."""

    def run_full(self, review_type, **over):
        state = state_for(**over)
        orch.run_orchestrated(state, orch.RulesPolicy(review_type))
        return state, orch.results_of(state, review_type=review_type, run_id="run-1", review_id="review-9")

    def test_a_full_classifier_run_has_a_section_for_each_stage_and_says_which_review_it_was(self):
        state, results = self.run_full("classifier")
        self.assertEqual(json.loads(json.dumps(results)), results)  # JSON all the way down
        self.assertEqual({k: results[k] for k in ("version", "type", "run_id", "review_id", "head", "steps")},
                         {"version": 1, "type": "classifier", "run_id": "run-1", "review_id": "review-9", "head": HEAD,
                          "steps": ["reason", "ground", "verify", "classify"]})
        self.assertEqual([k for k in results if k in ("reasoning", "verifier", "classifier")], ["reasoning", "verifier", "classifier"])
        (reasoning,) = results["reasoning"]["reviewers"]
        self.assertEqual((reasoning["name"], reasoning["chunks"], len(reasoning["findings"])), ("all", 1, 2))
        self.assertEqual({k: reasoning[k] for k in ("requests", "replies", "readable")}, {"requests": 1, "replies": 1, "readable": 1})
        (verifier,) = results["verifier"]["reviewers"]
        self.assertEqual((verifier["requests"], verifier["readable"], len(verifier["decisions"]), verifier["gave_up"]), (2, 2, 2, []))
        (classifier,) = results["classifier"]["reviewers"]
        self.assertEqual((classifier["requests"], classifier["readable"], len(classifier["answers"])), (2, 2, 2))
        self.assertEqual([f["verification"] for f in results["classifier"]["findings"]], ["confirmed", "confirmed"])
        self.assertIsNotNone(state.result)

    def test_a_type_that_stops_early_has_no_later_section_and_a_review_id_only_when_one_is_set(self):
        state = state_for()
        orch.run_orchestrated(state, orch.RulesPolicy("reasoning"))
        results = orch.results_of(state, review_type="reasoning", run_id="run-2")
        self.assertEqual([k for k in results if k in ("reasoning", "verifier", "classifier")], ["reasoning"])
        self.assertNotIn("review_id", results)
        self.assertEqual(results["steps"], ["reason"])

    def test_a_run_from_supplied_findings_has_no_section_for_the_stage_it_did_not_run(self):
        state = state_for(config=ReviewersConfig(rubric=RUBRIC, reviewers=(reviewer("v", roles=("verifier",)),)))
        orch.supply_findings(state, FindingsInput((supplied_finding("first constant", "x = 1", 1), supplied_finding("invented", "z = 9", 2))),
                             "verifier")
        orch.run_orchestrated(state, orch.RulesPolicy("verifier"))
        results = orch.results_of(state, review_type="verifier", run_id="run-3")
        self.assertEqual(results["steps"], ["ground", "verify"])
        self.assertEqual([k for k in results if k in ("reasoning", "verifier", "classifier")], ["verifier"])
        self.assertEqual([f["claim"] for f in results["verifier"]["findings"]], ["first constant"])
        self.assertEqual([(u["finding"]["claim"], u["reason"]) for u in results["verifier"]["ungrounded"]],
                         [("invented", "its quote is not in a.py")])
        # The warning that says where the findings came from belongs to the text; the section holds the verifier stage's own warnings.
        self.assertEqual(results["verifier"]["warnings"], [])

        classifier_state = state_for()
        orch.supply_findings(classifier_state, FindingsInput((supplied_finding("first constant", "x = 1", 1),), ("disputed",)), "classifier")
        orch.run_orchestrated(classifier_state, orch.RulesPolicy("classifier"))
        results = orch.results_of(classifier_state, review_type="classifier", run_id="run-4")
        self.assertEqual(results["steps"], ["classify"])
        self.assertEqual([k for k in results if k in ("reasoning", "verifier", "classifier")], ["classifier"])
        self.assertEqual(results["classifier"]["findings"][0]["verification"], "disputed")

    def test_a_run_that_ended_after_grounding_still_has_what_grounding_decided(self):
        state = state_for()
        orch.run_orchestrated(state, ScriptedPolicy(A("reason"), A("ground"), A("verify")), max_steps=2)
        results = orch.results_of(state, review_type="verifier", run_id="run-5")
        self.assertEqual(results["steps"], ["reason", "ground"])
        self.assertEqual(len(results["verifier"]["findings"]), 2)
        self.assertEqual(results["verifier"]["reviewers"], [])
        self.assertTrue(any("step cap" in w for w in results["verifier"]["warnings"]))

    def test_a_reviewer_that_failed_has_an_error_and_no_counts(self):
        def failing(client, model, system, text):
            raise RuntimeError("model blew up")
        state, results = self.run_full("reasoning", complete=failing)
        (entry,) = results["reasoning"]["reviewers"]
        self.assertEqual(entry["name"], "all")
        self.assertIn("RuntimeError: model blew up", entry["error"])
        self.assertNotIn("requests", entry)
        self.assertTrue(any("model blew up" in e for e in results["reasoning"]["errors"]))


@unittest.skipUnless(shutil.which("git"), "needs git")
class TheHeadIsPinnedForEveryType(IsolatedEnv):
    """`HEAD` moves part-way through a review (a commit in another session); the review still reads one commit."""

    def git(self, *args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-C", str(self.repo), *args],
                       check=True, capture_output=True)

    def commit(self, text, message):
        (self.repo / "a.py").write_text(text, encoding="utf-8")
        self.git("add", "a.py")
        self.git("commit", "-m", message)
        return subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()

    def setUp(self):
        super().setUp()
        self.repo = self.dir / "repo"
        self.repo.mkdir()
        self.git("init", "-b", "main")
        self.commit("x = 0\n", "base")
        self.git("checkout", "-b", "feature")
        self.pinned = self.commit("x = 1\n", "the change under review")
        spec = importlib.util.spec_from_file_location("codebase_reviewer_mcp_server_orchestrated", SERVER_PATH)
        assert spec is not None and spec.loader is not None
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        data = {"rubric": [], "reviewers": [{"name": "all", "model": "model-a", "prompt": "look", "active": True,
                                             "type": ["reasoning", "verifier"], "loadOnDemand": False, "replyFormat": "json"}]}
        self.reviewers = self.dir / "reviewers.json"
        self.reviewers.write_text(json.dumps(data), encoding="utf-8")

    def review(self, review_type):
        one_finding = json.dumps({"findings": [finding("constant", "x = 1")]})
        moved = []

        class MovingClient(FakeClient):
            def _create(inner, **kwargs):
                if not moved:  # the first model request: another session commits, so HEAD is no longer the reviewed commit
                    moved.append(self.commit("x = 2\n", "someone else's commit"))
                return super()._create(**kwargs)

        client = MovingClient([one_finding, VERDICT])
        env = {"REVIEWERS_FILE": str(self.reviewers), "MAX_REVIEWERS": "1"}
        with mock.patch.dict(os.environ, env), mock.patch.object(
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=FakeLmStudio(loaded=["model-a"]), **kw)
        ), mock.patch.object(self.mod, "new_client", return_value=client), mock.patch.object(
            self.mod, "_resolve_commit", wraps=self.mod._resolve_commit
        ) as resolve, mock.patch.object(self.mod, "_git_diff", wraps=self.mod._git_diff) as diff:
            out = self.mod.review(review_type, {"repo_path": str(self.repo), "base_ref": "main"})
        self.assertTrue(moved, "the test must have moved HEAD during the run")
        self.assertNotEqual(moved[0], self.pinned)
        return out, resolve, diff

    def test_a_reasoning_only_review_pins_head_ref_once(self):
        out, resolve, diff = self.review("reasoning")
        self.assertFalse(out.startswith("Error:"), out)
        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(diff.call_args.args[2], self.pinned)  # the diff is cut from the commit, not from the name `HEAD`

    def test_a_verifier_review_reads_the_cited_file_at_the_same_commit(self):
        # Had the file been read at the moved `HEAD` ("x = 2"), the finding's quote would be gone and it would be ungrounded.
        out, resolve, diff = self.review("verifier")
        self.assertIn("1 confirmed", out)
        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(diff.call_args.args[2], self.pinned)


if __name__ == "__main__":
    unittest.main()
