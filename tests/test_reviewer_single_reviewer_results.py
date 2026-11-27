#!/usr/bin/env python3
"""The results of single-reviewer runs, combined, give the outcome of one run with those reviewers.

A harness that runs each reviewer in its own session gets one results file per reviewer (`libs/reviewer_files_lib.py`) and
combines them with the functions the stages already use: `collect_findings` for the reasoning stage, `combine` for the verifier
stage and `classify_answers` for the classifier stage. These tests run each stage once with every reviewer, and once per reviewer
alone (the verifier and the classifier from a findings file, as the `review` tool runs them), write every run's results to a
file, read the files back, combine them, and require the same outcome, finding by finding.

Stdlib-only (unittest), no network, no LM Studio: the models are stubs, so what is compared is the code's own bookkeeping.

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import reviewer_files_lib as files  # noqa: E402
import reviewer_orchestrator_lib as orch  # noqa: E402
from classifier_reviewer_lib import classify_answers  # noqa: E402
from codebase_reviewer_config_lib import Reviewer, ReviewersConfig, RubricEntry  # noqa: E402
from fake_lmstudio import FakeLmStudio  # noqa: E402
from lmstudio_model_loader_lib import ModelLoader  # noqa: E402
from reasoning_reviewer_lib import collect_findings  # noqa: E402
from reviewer_db_lib import NullRecorder  # noqa: E402
from verifier_reviewer_lib import CONFIRMED, DISPUTED, REJECTED, UNCHECKED, combine  # noqa: E402

BASE = "http://localhost:1234/v1"
HEAD = "0" * 40
FILE_TEXT = "x = 1\ny = 2\nz = 3\nw = 4\n"
DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,4 +1,4 @@\n+x = 1\n+y = 2\n+z = 3\n+w = 4\n"
RUBRIC = (RubricEntry("injection", "high", "input reaches a shell", 0.6), RubricEntry("crash", "moderate", "valid input raises", 0.5))


def reviewer(name, model, role) -> Reviewer:
    return Reviewer(name=name, model=model, prompt="look at it", active=True, roles=(role,), load_on_demand=False, context_length=None,
                    reply_format="json")


def raised(claim, quote, line, cls="output") -> dict:
    return {"file": "a.py", "line_start": line, "line_end": line, "class": cls, "claim": claim, "failing_input": "any", "quote": quote}


# What each reasoning model raises. They overlap on purpose: the same claim at line 1 (one finding, two reviewers), and a different
# claim at line 3 (the second becomes an alternate of the first), so merging has real work to do. `finder` raises all of them alone
# for the stages that run after it.
REASONING = {
    "model-a": [raised("first constant", "x = 1", 1), raised("third constant", "z = 3", 3)],
    "model-b": [raised("first constant", "x = 1", 1), raised("another claim about the third", "z = 3", 3), raised("second constant", "y = 2", 2)],
    "model-f": [raised("first constant", "x = 1", 1), raised("third constant", "z = 3", 3), raised("another claim about the third", "z = 3", 3),
                raised("second constant", "y = 2", 2), raised("invented", "q = 9", 4)],
}
# What each verifier decides about a claim ("garbled" is a reply that cannot be read).
VERIFIERS = {
    "model-c": {"first constant": "confirmed", "second constant": "rejected", "third constant": "confirmed",
                "another claim about the third": "confirmed"},
    "model-d": {"first constant": "confirmed", "second constant": "rejected", "third constant": "rejected",
                "another claim about the third": "garbled"},
}
# Each classifier's probability of yes, by claim and rubric entry; `None` is a request that failed.
CLASSIFIERS = {
    "model-e": {"first constant": {"injection": 0.9, "crash": 0.2}, "third constant": {"injection": 0.3, "crash": 0.9},
                "another claim about the third": {"injection": 0.1, "crash": 0.1}},
    "model-g": {"first constant": {"injection": 0.5, "crash": 0.4}, "third constant": {"injection": 0.4, "crash": 0.5},
                "another claim about the third": None},
}


def claim_of(content) -> str:
    return content.split("Claim: ", 1)[1].split("\n", 1)[0]


def complete(client, model, system, content):
    """The stub for the reasoning and verifier stages."""
    if "checking ONE reported defect" in system:
        verdict = VERIFIERS[model][claim_of(content)]
        if verdict == "garbled":
            return "I think so, but I cannot say"
        if verdict == "confirmed":
            return json.dumps({"verdict": "confirmed", "failing_input": f"input for {claim_of(content)}", "wrong_result": "wrong", "reasoning": "traced"})
        return json.dumps({"verdict": "rejected", "failing_input": "", "wrong_result": "", "reasoning": "a guard exists"})
    return json.dumps({"findings": REASONING[model]})


def probability(client, model, system, content):
    by_entry = CLASSIFIERS[model][claim_of(content)]
    if by_entry is None:
        return None
    return by_entry[content.split("Category: ", 1)[1].split(" (", 1)[0]]


FINDER = reviewer("finder", "model-f", "reasoning")
V1, V2 = reviewer("v1", "model-c", "verifier"), reviewer("v2", "model-d", "verifier")
C1, C2 = reviewer("c1", "model-e", "classifier"), reviewer("c2", "model-g", "classifier")
MODELS = ["model-a", "model-b", "model-c", "model-d", "model-e", "model-f", "model-g"]


def run(review_type, *reviewers, supplied=None, folder=None, name="run") -> tuple:
    """One run of `review_type` with `reviewers` active (from `supplied` findings when given), its results written to a file in
    `folder` and read back: `(state, the results as the file holds them)`."""
    state = orch.RunState(
        config=ReviewersConfig(rubric=RUBRIC, reviewers=tuple(reviewers)), loader=ModelLoader(BASE, http=FakeLmStudio(loaded=MODELS)),
        client=object(), repo_root=Path("."), head=HEAD, diff=DIFF, max_reviewers=1, recorder=NullRecorder(), complete=complete,
        probability=probability, read_file=lambda path: FILE_TEXT if path == "a.py" else None,
    )
    if supplied is not None:
        orch.supply_findings(state, supplied, review_type)
    orch.run_orchestrated(state, orch.RulesPolicy(review_type))
    path = files.write_results(folder, orch.results_of(state, review_type=review_type, run_id=name))
    return state, json.loads(path.read_text(encoding="utf-8"))


class InTempDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def findings_file(self, findings, statuses=None) -> files.FindingsInput:
        """The findings as a harness would hand them on: written to a file, and read by the same reader `review` uses."""
        path = self.dir / "findings.json"
        path.write_text(json.dumps(files.findings_document(findings, statuses)), encoding="utf-8")
        return files.read_findings_file(str(path), "classifier" if statuses is not None else "verifier")


class ReasoningResultsCombine(InTempDir):
    def test_each_reviewers_own_findings_merged_with_collect_findings_are_what_one_run_with_both_reports(self):
        both_state, both = run("reasoning", reviewer("r1", "model-a", "reasoning"), reviewer("r2", "model-b", "reasoning"),
                               folder=self.dir, name="both")
        _, first = run("reasoning", reviewer("r1", "model-a", "reasoning"), folder=self.dir, name="r1")
        _, second = run("reasoning", reviewer("r2", "model-b", "reasoning"), folder=self.dir, name="r2")
        one_run = both_state.reasoning.groups
        raw = []
        for results in (first, second):
            for entry in results["reasoning"]["reviewers"]:
                raw.extend(files.reasoning_findings(entry))
        self.assertEqual(collect_findings(raw), one_run)
        # There is real merging to agree on: three findings from five reports, one carrying the other claim as an alternate.
        self.assertEqual(sum(len(g.findings) for g in one_run), 3)
        self.assertEqual(sum(len(f.alternates) for g in one_run for f in g.findings), 1)
        # The file of the run with both reviewers combines to the same thing.
        both_raw = [f for entry in both["reasoning"]["reviewers"] for f in files.reasoning_findings(entry)]
        self.assertEqual(collect_findings(both_raw), one_run)

    def test_each_reviewers_findings_are_kept_as_that_reviewer_raised_them(self):
        _, results = run("reasoning", reviewer("r2", "model-b", "reasoning"), folder=self.dir)
        (entry,) = results["reasoning"]["reviewers"]
        self.assertEqual([(f.reviewers, f.claim) for f in files.reasoning_findings(entry)],
                         [(("r2",), "first constant"), (("r2",), "another claim about the third"), (("r2",), "second constant")])


class VerifierResultsCombine(InTempDir):
    def test_each_verifiers_decisions_combined_with_combine_are_what_one_run_with_both_verifiers_decides(self):
        one_state, _ = run("verifier", FINDER, V1, V2, folder=self.dir, name="all")
        # The findings a harness hands every verifier: what the reasoning reviewers found, written down once.
        reasoning_state, _ = run("reasoning", FINDER, folder=self.dir, name="reasoning")
        findings = [f for g in reasoning_state.reasoning.groups for f in g.findings]
        supplied = self.findings_file(findings)
        first_state, first = run("verifier", V1, supplied=supplied, folder=self.dir, name="v1")
        second_state, second = run("verifier", V2, supplied=supplied, folder=self.dir, name="v2")
        self.assertEqual((first_state.done, second_state.done), (["ground", "verify"], ["ground", "verify"]))  # no reasoning stage

        sections = [first["verifier"], second["verifier"]]
        # Every verifier saw the same findings in the same positions: that is what lets their decisions be matched up.
        grounded = [files.finding_from_dict(item) for item in sections[0]["findings"]]
        self.assertEqual([files.finding_from_dict(item) for item in sections[1]["findings"]], grounded)
        self.assertEqual(sections[0]["ungrounded"], sections[1]["ungrounded"])
        by_position: dict = {n: [] for n in range(len(grounded))}
        for section in sections:
            for entry in section["reviewers"]:
                for position, decision in files.verifier_decisions(entry).items():
                    by_position[position].append(decision)
        combined = [(finding, combine(by_position[n]), tuple(by_position[n])) for n, finding in enumerate(grounded)]

        one_run = [(v.finding, v.status, v.decisions) for v in one_state.verification.verified]
        self.assertEqual(combined, one_run)
        self.assertEqual([(u["finding"]["claim"], u["reason"]) for u in sections[0]["ungrounded"]],
                         [(u.finding.claim, u.reason) for u in one_state.verification.ungrounded])
        # And the combination is not trivial: every status a verifier stage can reach, except unchecked, is in it.
        self.assertEqual(sorted(status for _, status, _ in combined), sorted([CONFIRMED, CONFIRMED, REJECTED, DISPUTED]))
        self.assertEqual(len(sections[0]["ungrounded"]), 1)

    def test_a_reply_that_could_not_be_read_is_counted_and_leaves_that_verifier_out_of_that_findings_vote(self):
        reasoning_state, _ = run("reasoning", FINDER, folder=self.dir, name="reasoning")
        supplied = self.findings_file([f for g in reasoning_state.reasoning.groups for f in g.findings])
        _, results = run("verifier", V2, supplied=supplied, folder=self.dir, name="v2")
        (entry,) = results["verifier"]["reviewers"]
        # Four grounded claims were put to it: it answered four times and could read three of them.
        self.assertEqual((entry["requests"], entry["replies"], entry["readable"]), (4, 4, 3))
        self.assertEqual(len(entry["decisions"]), 3)


class ClassifierResultsCombine(InTempDir):
    def test_each_classifiers_probabilities_combined_with_classify_answers_are_what_one_run_with_both_decides(self):
        one_state, _ = run("classifier", FINDER, V1, V2, C1, C2, folder=self.dir, name="all")
        verified = one_state.verification
        standing = [v for v in verified.verified if v.status != REJECTED]
        supplied = self.findings_file([v.finding for v in standing], [v.status for v in standing])
        first_state, first = run("classifier", C1, supplied=supplied, folder=self.dir, name="c1")
        second_state, second = run("classifier", C2, supplied=supplied, folder=self.dir, name="c2")
        self.assertEqual((first_state.done, second_state.done), (["classify"], ["classify"]))  # nothing before it

        sections = [first["classifier"], second["classifier"]]
        findings = [files.finding_from_dict(item) for item in sections[0]["findings"]]
        self.assertEqual([files.finding_from_dict(item) for item in sections[1]["findings"]], findings)
        self.assertEqual([item["verification"] for item in sections[0]["findings"]], [v.status for v in standing])
        by_position: dict = {n: [] for n in range(len(findings))}
        for section in sections:
            for entry in section["reviewers"]:
                for position, answer in files.classifier_answers(entry).items():
                    by_position[position].append(answer)
        combined = [classify_answers(by_position[n], RUBRIC) for n in range(len(findings))]

        one_run = one_state.classification.classified
        self.assertEqual([c.finding for c in one_run], findings)
        self.assertEqual([(o.status, o.entry, o.level, o.confidence, o.overlap, o.overlapping, o.scores) for o in combined],
                         [(c.status, c.entry, c.level, c.confidence, c.overlap, c.overlapping, c.scores) for c in one_run])
        # Not trivial: one finding classified high, one moderate, one dropped as matching no entry, and one classifier failed a request.
        self.assertEqual(sorted(o.status for o in combined), sorted(["classified", "classified", "unclassified"]))
        self.assertEqual(sorted(str(o.entry) for o in combined), sorted(["injection", "crash", "None"]))
        self.assertTrue(any(len(by_position[n]) == 1 for n in by_position))  # a finding only one classifier could answer

    def test_a_request_that_failed_is_counted_and_that_classifier_gives_no_answer_for_that_finding(self):
        verified_state, _ = run("verifier", FINDER, V1, folder=self.dir, name="verifier")
        standing = [v for v in verified_state.verification.verified if v.status != REJECTED]
        supplied = self.findings_file([v.finding for v in standing], [v.status for v in standing])
        _, results = run("classifier", C2, supplied=supplied, folder=self.dir, name="c2")
        (entry,) = results["classifier"]["reviewers"]
        # Three findings, two rubric entries each: six requests. The alternate claim's two failed, so this classifier has no answer for it.
        self.assertEqual((entry["requests"], entry["replies"], entry["readable"]), (6, 4, 4))
        self.assertEqual(sorted(a["finding"] for a in entry["answers"]), [0, 1])
        self.assertEqual(files.classifier_answers(entry).keys(), {0, 1})


class UncheckedStaysUnchecked(InTempDir):
    def test_a_finding_no_verifier_could_answer_is_unchecked_whichever_way_it_got_there(self):
        # Verifier v2 cannot be read on the alternate claim: one run leaves that finding unchecked, and so does the combined result.
        full_state, _ = run("verifier", FINDER, V2, folder=self.dir, name="full")
        (alternate,) = [v for v in full_state.verification.verified if v.finding.claim == "another claim about the third"]
        self.assertEqual((alternate.status, alternate.decisions), (UNCHECKED, ()))
        _, results = run("verifier", V2, supplied=self.findings_file([alternate.finding]), folder=self.dir, name="v2")
        (entry,) = results["verifier"]["reviewers"]
        self.assertEqual((entry["requests"], entry["replies"], entry["readable"]), (1, 1, 0))
        self.assertEqual(combine(list(files.verifier_decisions(entry).values())), UNCHECKED)


if __name__ == "__main__":
    unittest.main()
