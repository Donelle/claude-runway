#!/usr/bin/env python3
"""Tests for libs/classifier_reviewer_lib.py.

Stdlib-only (unittest), no network: the model's probability of "yes" is a stub and LM Studio's
REST API is `tests/fake_lmstudio.py`.

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import openai_lib  # noqa: E402
from classifier_reviewer_lib import (  # noqa: E402
    CLASSIFIED,
    MAX_CLASSIFIED_FINDINGS,
    UNCHECKED,
    UNCLASSIFIED,
    Classification,
    ClassificationReview,
    ClassifiedReview,
    Rating,
    build_classifier_message,
    build_classifier_prompt,
    build_entry_text,
    classify_answers,
    classify_findings,
    format_classified_review,
    run_classified_review,
    score_entries,
)
from codebase_reviewer_config_lib import ConfigError, Reviewer, ReviewersConfig, RubricEntry, load_definition  # noqa: E402
from fake_lmstudio import FakeLmStudio  # noqa: E402
from lmstudio_model_loader_lib import ModelLoader, ModelNotLoadedError  # noqa: E402
from reasoning_reviewer_lib import Finding, ReasoningReview  # noqa: E402
from verifier_reviewer_lib import VerificationReview, VerifiedReview  # noqa: E402
from reviewer_db_lib import NullRecorder  # noqa: E402

BASE = "http://localhost:1234/v1"

# Order matters: the tie-break for equal level and average is rubric order.
RUBRIC = (
    RubricEntry("injection", "high", "Untrusted input reaches a shell or a path.", 0.6),
    RubricEntry("crash", "moderate", "Valid input makes the code raise.", 0.6),
    RubricEntry("wrong-result", "moderate", "Valid input gives the wrong data.", 0.4),
    RubricEntry("style", "low", "Naming and layout only.", 0.8),
)
# What a stub answers for an entry it is told nothing about: the model is sure it does not fit.
NOT_A_FIT = 0.05


def rv(name, model="model-a", roles=("classifier",), *, prompt="check it", active=True, on_demand=False, context=None) -> Reviewer:
    return Reviewer(
        name=name, model=model, prompt=prompt, active=active,
        roles=tuple(r for r in ("reasoning", "verifier", "classifier") if r in roles),
        load_on_demand=on_demand, context_length=context, prompt_file="", reply_format="json",
    )


def cfg(*reviewers, rubric=RUBRIC) -> ReviewersConfig:
    return ReviewersConfig(rubric=tuple(rubric), reviewers=tuple(reviewers))


def fnd(claim="runs the input in a shell", cls="injection", file="a.py", start=4, end=4, quote="subprocess.run(cmd, shell=True)") -> Finding:
    return Finding(file, start, end, cls, claim, "cmd='a; b'", ("r1",), (), quote)


def entry_of(content: str) -> str:
    """The rubric entry a request is about: the name on its `Category:` line."""
    return re.search(r"^Category: (\S+) \(level", content, re.MULTILINE).group(1)


def probs(**by_entry):
    """A stub answer: the probability of yes for each named entry (`wrong_result` for `wrong-result`), `NOT_A_FIT` for the rest."""
    return lambda model, system, content: by_entry.get(entry_of(content).replace("-", "_"), NOT_A_FIT)


def rated(**by_entry) -> tuple:
    """A full set of probabilities in rubric order, the entries not given being `NOT_A_FIT`."""
    return tuple(Rating(e.name, by_entry.get(e.name.replace("-", "_"), NOT_A_FIT)) for e in RUBRIC)


def given(*sets) -> list:
    return [Classification(f"c{n}", ratings) for n, ratings in enumerate(sets, start=1)]


class StubProbability:
    def __init__(self, values):
        self.values = values  # callable(model, system, content) -> float | None
        self.calls: list = []
        self._lock = threading.Lock()

    def __call__(self, client, model, system, content):
        with self._lock:
            self.calls.append((model, system, content))
        return self.values(model, system, content)


class StubComplete:
    """The reasoning and verifier stages' completion, for the review tests."""

    def __init__(self, replies):
        self.replies = replies
        self.calls: list = []
        self._lock = threading.Lock()

    def __call__(self, client, model, system, content, **options):
        with self._lock:
            self.calls.append((model, system, content))
        return self.replies(model, system, content)


class RecordingRecorder(NullRecorder):
    """A recorder that keeps what the classifier stage hands it."""

    def __init__(self):
        self.classifications: list = []
        self.notes: list = []

    def add_classification(self, **record) -> None:
        self.classifications.append(record)

    def note(self, stage, text, *, reviewer="", detail=()) -> None:
        self.notes.append((stage, text))


class ScoreEntries(unittest.TestCase):
    def test_an_entrys_score_is_the_plain_average_of_every_classifiers_probability_for_it(self):
        scores = score_entries(given(rated(injection=0.9), rated(injection=0.6), rated(injection=0.75)), RUBRIC)
        self.assertEqual(scores[0].entry, "injection")
        self.assertAlmostEqual(scores[0].mean, 0.75)
        self.assertAlmostEqual(scores[1].mean, NOT_A_FIT)

    def test_an_average_exactly_at_the_score_reaches_it_and_just_below_does_not(self):
        self.assertTrue(score_entries(given(rated(injection=0.5), rated(injection=0.7)), RUBRIC)[0].reached)  # 0.6 against 0.6
        self.assertFalse(score_entries(given(rated(injection=0.59)), RUBRIC)[0].reached)
        self.assertTrue(score_entries(given(rated(injection=0.61)), RUBRIC)[0].reached)

    def test_float_noise_cannot_keep_an_average_that_is_the_score_from_reaching_it(self):
        entry = (RubricEntry("e", "low", "d", 0.7),)
        answers = [Classification("c", (Rating("e", 0.1 + 0.6),))]  # 0.7 as a float is 0.7 minus a hair
        self.assertTrue(score_entries(answers, entry)[0].reached)

    def test_the_score_is_per_entry(self):
        scores = score_entries(given(rated(crash=0.5, wrong_result=0.5)), RUBRIC)  # score 0.6 for crash, 0.4 for wrong-result
        self.assertEqual([s.reached for s in scores], [False, False, True, False])

    def test_a_score_of_one_is_reached_only_by_certainty(self):
        entry = (RubricEntry("e", "low", "d", 1.0),)
        self.assertTrue(score_entries([Classification("c", (Rating("e", 1.0),))], entry)[0].reached)
        self.assertFalse(score_entries([Classification("c", (Rating("e", 0.999),))], entry)[0].reached)


class ClassifyAnswers(unittest.TestCase):
    def test_no_entry_reaching_its_score_leaves_the_finding_unclassified_with_its_averages(self):
        got = classify_answers(given(rated(injection=0.3)), RUBRIC)
        self.assertEqual((got.status, got.entry, got.level), (UNCLASSIFIED, None, None))
        self.assertEqual(got.scores[0].mean, 0.3)

    def test_an_entry_at_or_above_its_score_classifies_the_finding_with_its_average(self):
        got = classify_answers(given(rated(style=0.9)), RUBRIC)  # score 0.8
        self.assertEqual((got.status, got.entry, got.level, got.confidence, got.overlap), (CLASSIFIED, "style", "low", 0.9, False))

    def test_a_finding_that_fits_several_entries_takes_the_highest_level_and_records_the_overlap(self):
        got = classify_answers(given(rated(crash=0.95, injection=0.65, style=0.95)), RUBRIC)
        self.assertEqual((got.entry, got.level, got.confidence), ("injection", "high", 0.65))
        self.assertTrue(got.overlap)
        self.assertEqual(got.overlapping, ("crash", "style"))  # rubric order

    def test_an_entry_below_its_score_is_not_part_of_the_overlap(self):
        got = classify_answers(given(rated(crash=0.9, injection=0.4)), RUBRIC)
        self.assertEqual((got.entry, got.overlap, got.overlapping), ("crash", False, ()))

    def test_equal_levels_go_to_the_higher_average_then_to_the_first_in_the_rubric(self):
        self.assertEqual(classify_answers(given(rated(crash=0.7, wrong_result=0.9)), RUBRIC).entry, "wrong-result")
        self.assertEqual(classify_answers(given(rated(crash=0.9, wrong_result=0.9)), RUBRIC).entry, "crash")

    def test_no_answer_is_unchecked(self):
        self.assertEqual(classify_answers([], RUBRIC).status, UNCHECKED)

    def test_the_answers_of_several_classifiers_are_averaged(self):
        got = classify_answers(given(rated(injection=0.9), rated(injection=0.8), rated(injection=0.7)), RUBRIC)
        self.assertEqual((got.status, got.entry), (CLASSIFIED, "injection"))
        self.assertAlmostEqual(got.confidence, 0.8)


class WhatTheClassifierIsShown(unittest.TestCase):
    def test_the_prompt_is_the_lens_then_the_question(self):
        prompt = build_classifier_prompt(rv("c1", prompt="MY LENS"))
        positions = [prompt.index(part) for part in ("MY LENS", "checking ONE reported defect against ONE category", "Reply with ONLY Yes or No.")]
        self.assertEqual(positions, sorted(positions))

    def test_the_prompt_asks_whether_the_claim_falls_under_the_definition_and_does_not_use_a_scale(self):
        prompt = build_classifier_prompt(rv("c1"))
        self.assertIn("the kind of defect the category's definition describes", prompt)
        self.assertNotIn("1 to 5", prompt)
        self.assertNotIn("JSON", prompt)
        self.assertNotIn("confident", prompt)

    def test_the_prompt_is_the_same_for_every_entry_so_it_names_none(self):
        prompt = build_classifier_prompt(rv("c1"))
        for entry in RUBRIC:
            self.assertNotIn(entry.name, prompt)

    def test_the_entry_text_shows_its_name_level_and_definition_but_never_its_score(self):
        for entry in RUBRIC:
            text = build_entry_text(entry)
            self.assertEqual(text, f"Category: {entry.name} (level: {entry.level})\nDefinition: {entry.definition}")
            self.assertNotIn("score", text)
            self.assertNotIn(str(entry.score), text)

    def test_a_definition_file_adds_to_the_inline_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long.md"
            path.write_text("A longer account of what counts.\n", encoding="utf-8")
            entry = RubricEntry("injection", "high", "Short form.", 0.6, definition_file=str(path))
            self.assertEqual(load_definition(entry), "Short form.\n\nA longer account of what counts.")
            self.assertIn("A longer account of what counts.", build_entry_text(entry))
            path.write_text("Edited since.\n", encoding="utf-8")  # read at run time, not cached
            self.assertIn("Edited since.", load_definition(entry))
            path.unlink()
            with self.assertRaises(ConfigError) as caught:
                load_definition(entry)
            self.assertIn("'injection'", str(caught.exception))

    def test_the_message_carries_the_finding_then_the_entry_it_is_checked_against(self):
        message = build_classifier_message(fnd(start=4, end=6), RUBRIC[1])
        for part in ("File: a.py:4-6", "(not a rubric entry): injection", "Claim: runs the input in a shell",
                     "Failing input the reporter gave: cmd='a; b'", "The code the claim accuses: subprocess.run(cmd, shell=True)"):
            self.assertIn(part, message)
        self.assertTrue(message.endswith("Category: crash (level: moderate)\nDefinition: Valid input makes the code raise."))
        self.assertNotIn("Code:", message)


class ClassifyFindings(unittest.TestCase):
    def _run(self, config, values, findings, *, loaded=("model-a", "model-b"), statuses=None, recorder=None):
        fake = FakeLmStudio(loaded=loaded)
        stub = StubProbability(values)
        review = classify_findings(config, ModelLoader(BASE, http=fake), object(), findings, 3, statuses=statuses, probability=stub,
                                   recorder=recorder or RecordingRecorder())
        return review, stub, fake

    def test_an_average_at_or_above_the_score_classifies_the_finding(self):
        review, _, _ = self._run(cfg(rv("c1")), probs(injection=0.9), [fnd()])
        (item,) = review.classified
        self.assertEqual((item.status, item.entry, item.level, item.confidence), (CLASSIFIED, "injection", "high", 0.9))
        self.assertEqual(review.classifiers_run, ["c1"])

    def test_every_entry_is_asked_about_separately_for_every_finding(self):
        _, stub, _ = self._run(cfg(rv("c1")), probs(), [fnd(), fnd(claim="second")])
        self.assertEqual(len(stub.calls), 2 * len(RUBRIC))
        asked = [entry_of(content) for _, _, content in stub.calls]
        self.assertEqual(asked, [e.name for e in RUBRIC] * 2)

    def test_an_average_below_the_score_drops_the_finding_as_unclassified_and_records_it(self):
        recorder = RecordingRecorder()
        review, _, _ = self._run(cfg(rv("c1")), probs(injection=0.3), [fnd()], recorder=recorder)
        self.assertEqual([c.status for c in review.classified], [UNCLASSIFIED])
        self.assertEqual(review.unclassified[0].classifications[0].ratings[0], Rating("injection", 0.3))
        self.assertEqual([r["status"] for r in recorder.classifications], [UNCLASSIFIED])

    def test_a_finding_that_fits_two_entries_takes_the_higher_level_and_the_overlap_is_recorded(self):
        recorder = RecordingRecorder()
        review, _, _ = self._run(cfg(rv("c1")), probs(crash=0.95, injection=0.7), [fnd()], recorder=recorder)
        (item,) = review.classified
        self.assertEqual((item.entry, item.level, item.overlap, item.overlapping), ("injection", "high", True, ("crash",)))
        self.assertEqual(len(review.overlaps), 1)
        record = recorder.classifications[0]
        self.assertEqual((record["entry"], record["level"], record["overlap"], record["overlapping"]), ("injection", "high", True, ["crash"]))

    def test_several_models_asked_once_each_are_averaged(self):
        values = lambda model, s, c: {"model-a": 0.9, "model-b": 0.7}[model] if entry_of(c) == "injection" else NOT_A_FIT  # noqa: E731
        review, stub, _ = self._run(cfg(rv("c1", "model-a"), rv("c2", "model-b")), values, [fnd()])
        (item,) = review.classified
        self.assertEqual(len(stub.calls), 2 * len(RUBRIC))
        self.assertEqual((item.status, item.entry), (CLASSIFIED, "injection"))
        self.assertAlmostEqual(item.confidence, 0.8)
        self.assertEqual({c.reviewer for c in item.classifications}, {"c1", "c2"})

    def test_classifiers_that_split_between_entries_do_not_reach_the_score_and_the_finding_is_unclassified(self):
        # There is no dispute: a disagreement is a lower average. (0.9 + 0.05) / 2 = 0.475 is under both scores of 0.6.
        values = lambda model, s, c: (0.9 if entry_of(c) == ("injection" if model == "model-a" else "crash") else NOT_A_FIT)  # noqa: E731
        review, _, _ = self._run(cfg(rv("c1", "model-a"), rv("c2", "model-b")), values, [fnd()])
        (item,) = review.classified
        self.assertEqual(item.status, UNCLASSIFIED)
        self.assertEqual([(s.entry, round(s.mean, 3)) for s in item.scores[:2]], [("injection", 0.475), ("crash", 0.475)])

    def test_the_score_is_never_sent_to_the_model(self):
        _, stub, _ = self._run(cfg(rv("c1")), probs(), [fnd()])
        for _, system, content in stub.calls:
            self.assertNotIn("score", system)
            self.assertNotIn("score", content)
            self.assertNotIn("0.6", content)

    def test_the_prompt_and_message_a_classifier_receives(self):
        _, stub, _ = self._run(cfg(rv("c1")), probs(), [fnd()])
        model, system, content = stub.calls[0]
        self.assertEqual(model, "model-a")
        self.assertEqual(system, build_classifier_prompt(rv("c1")))
        self.assertIn("Claim: runs the input in a shell", content)
        self.assertIn("Category: injection (level: high)\nDefinition: Untrusted input reaches a shell or a path.", content)

    def test_the_probability_requests_use_the_probability_client_when_one_is_given(self):
        seen = []

        def values(client, model, system, content):
            seen.append(client)
            return NOT_A_FIT

        stub = lambda client, model, system, content: values(client, model, system, content)  # noqa: E731
        reviewers_client, probability_client = object(), object()
        classify_findings(cfg(rv("c1")), ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])), reviewers_client, [fnd()], 3,
                          probability=stub, probability_client=probability_client)
        self.assertEqual(seen, [probability_client] * len(RUBRIC))
        seen.clear()
        classify_findings(cfg(rv("c1")), ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])), reviewers_client, [fnd()], 3, probability=stub)
        self.assertEqual(seen, [reviewers_client] * len(RUBRIC))

    def test_a_request_that_gets_no_probability_leaves_that_classifier_without_an_answer_for_the_finding(self):
        # An answer needs every entry: a missing one would read as "does not fit".
        values = lambda model, s, c: None if entry_of(c) == "crash" else 0.9  # noqa: E731
        review, _, _ = self._run(cfg(rv("c1")), values, [fnd()])
        (item,) = review.classified
        self.assertEqual((item.status, item.note), (UNCHECKED, "no classifier gave a usable answer"))
        self.assertTrue(any("finding 1/1" in e and "no probability for crash" in e for e in review.errors))

    def test_with_no_classifier_active_the_findings_carry_no_level_and_nothing_is_loaded_or_asked(self):
        config = cfg(rv("c1", active=False), rv("other", roles=("verifier",)))
        review, stub, _ = self._run(config, probs(), [fnd(), fnd(claim="second")], statuses=["confirmed", "unchecked"])
        self.assertEqual([c.status for c in review.classified], [UNCHECKED, UNCHECKED])
        self.assertEqual([c.verification for c in review.classified], ["confirmed", "unchecked"])
        self.assertTrue(all(c.level is None and c.entry is None for c in review.classified))
        self.assertEqual("no classifier is active", review.classified[0].note)
        self.assertEqual(stub.calls, [])
        self.assertEqual(review.classifiers_run, [])

    def test_a_classifier_whose_requests_all_fail_is_reported_and_the_other_still_decides(self):
        values = lambda model, s, c: None if model == "model-a" else (0.9 if entry_of(c) == "crash" else NOT_A_FIT)  # noqa: E731
        review, _, _ = self._run(cfg(rv("c1", "model-a"), rv("c2", "model-b")), values, [fnd()])
        self.assertEqual(review.classified[0].status, CLASSIFIED)
        self.assertEqual(review.classifiers_run, ["c2"])
        self.assertTrue(any("'c1'" in e and "every request failed" in e for e in review.errors))

    def test_a_model_that_returns_no_token_probabilities_is_an_error_naming_the_reviewer_and_the_model(self):
        def values(model, system, content):
            openai_lib._failure_state.kind = "no_probabilities"
            return None

        review, _, _ = self._run(cfg(rv("c1")), values, [fnd()])
        self.assertEqual(review.classified[0].status, UNCHECKED)
        (error,) = [e for e in review.errors if "every request failed" in e]
        self.assertIn("'c1'", error)
        self.assertIn("model 'model-a' returned no token probabilities", error)

    def test_a_model_that_starts_its_reply_with_something_other_than_yes_or_no_is_named_too(self):
        def values(model, system, content):
            openai_lib._failure_state.kind = "no_answer"
            return None

        review, _, _ = self._run(cfg(rv("c1")), values, [fnd()])
        self.assertTrue(any("began its reply with something other than Yes or No" in e for e in review.errors))

    def test_the_error_for_it_says_what_the_model_began_with_and_what_to_check(self):
        def values(model, system, content):
            openai_lib._failure_state.kind = "no_answer"
            openai_lib._failure_state.detail = "'' 56%, 'The' 4%"
            return None

        review, _, _ = self._run(cfg(rv("c1")), values, [fnd()])
        (error,) = [e for e in review.errors if "every request failed" in e]
        self.assertIn("its likeliest first tokens: '' 56%, 'The' 4%", error)
        self.assertIn("check the template in LM Studio or use another model", error)

    def test_findings_over_the_limit_are_not_classified_and_come_back_unchecked_with_a_warning(self):
        findings = [fnd(claim=f"claim {n}") for n in range(MAX_CLASSIFIED_FINDINGS + 2)]
        review, stub, _ = self._run(cfg(rv("c1")), probs(injection=0.9), findings)
        self.assertEqual(len(stub.calls), MAX_CLASSIFIED_FINDINGS * len(RUBRIC))
        self.assertEqual(len(review.classified), MAX_CLASSIFIED_FINDINGS + 2)
        self.assertEqual([c.status for c in review.classified[-2:]], [UNCHECKED, UNCHECKED])
        self.assertTrue(any("not sent to the classifiers" in w for w in review.warnings))

    def test_a_resident_model_that_is_not_loaded_refuses_before_anything_runs(self):
        stub = StubProbability(probs())
        with self.assertRaises(ModelNotLoadedError):
            classify_findings(cfg(rv("c1", "model-z")), ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])), object(), [fnd()], 3,
                              probability=stub)
        self.assertEqual(stub.calls, [])

    def test_statuses_must_match_the_findings(self):
        with self.assertRaises(ValueError):
            self._run(cfg(rv("c1")), probs(), [fnd()], statuses=["confirmed", "confirmed"])

    def test_every_classification_is_handed_to_the_recorder_with_every_probability_and_the_averages(self):
        recorder = RecordingRecorder()
        findings = [fnd(claim="first"), fnd(claim="second", file="b.py", start=9, end=11, cls="style")]
        values = lambda m, s, c: (0.9 if entry_of(c) == ("injection" if "first" in c else "style") else NOT_A_FIT)  # noqa: E731
        self._run(cfg(rv("c1"), rv("c2")), values, findings, statuses=["confirmed", "disputed"], recorder=recorder)
        first, second = recorder.classifications
        self.assertEqual((first["finding_no"], first["file"], first["line_start"], first["finding_class"], first["verification"]),
                         (1, "a.py", 4, "injection", "confirmed"))
        self.assertEqual((second["finding_no"], second["file"], second["line_end"], second["status"], second["entry"], second["level"]),
                         (2, "b.py", 11, CLASSIFIED, "style", "low"))
        self.assertEqual(first["confidence"], 0.9)
        self.assertEqual([a["reviewer"] for a in first["answers"]], ["c1", "c2"])
        self.assertEqual(first["answers"][0]["probabilities"][0], {"entry": "injection", "probability": 0.9})
        self.assertEqual(len(first["answers"][0]["probabilities"]), len(RUBRIC))  # every entry
        self.assertEqual(first["scores"][0], {"entry": "injection", "mean": 0.9, "reached": True})
        self.assertNotIn("claim", first)  # the claim text is for the session log, not the table
        self.assertEqual([stage for stage, _ in recorder.notes if stage == "classifier"], ["classifier", "classifier"])


class RunClassifiedReview(unittest.TestCase):
    SOURCE = "\n".join([
        "import subprocess",  # 1
        "",  # 2
        "def run(cmd):",  # 3
        "    return subprocess.run(cmd, shell=True)",  # 4
        "",  # 5
        "def clean(items=[]):",  # 6
        "    try:",  # 7
        "        items.append(1)",  # 8
        "    except:",  # 9
        "        pass",  # 10
        "    return items",  # 11
    ])
    DIFF = "diff --git a/a.py b/a.py\nindex 1..2 100644\n--- a/a.py\n+++ b/a.py\n@@ -1,0 +1,11 @@\n" + "".join(
        f"+{line}\n" for line in SOURCE.splitlines()
    )

    def _run(self, config, values, *, replies=None, loaded=("model-a", "model-b", "model-c"), recorder=None):
        fake = FakeLmStudio(loaded=loaded)
        stub = StubComplete(replies or self._replies())
        probability = StubProbability(values)
        result = run_classified_review(config, ModelLoader(BASE, http=fake), object(), self.DIFF, "/unused", "HEAD", 3, complete=stub,
                                       probability=probability, read_file=lambda path: self.SOURCE if path == "a.py" else None,
                                       recorder=recorder or RecordingRecorder())
        return result, stub, probability

    @staticmethod
    def _replies(verdict_kind="confirmed"):
        def replies(model, system, content):
            if "checking ONE reported defect" in system:
                if "bare except" in content:
                    return json.dumps({"verdict": "rejected", "reasoning": "it is handled"})
                return json.dumps({"verdict": verdict_kind, "failing_input": "cmd='a; b'", "wrong_result": "runs b", "reasoning": "traced"})
            return json.dumps({"findings": [
                {"file": "a.py", "line_start": 4, "line_end": 4, "class": "injection", "claim": "runs the input in a shell",
                 "failing_input": "cmd='a; b'", "quote": "subprocess.run(cmd, shell=True)"},
                {"file": "a.py", "line_start": 9, "line_end": 9, "class": "bare-except", "claim": "a bare except hides errors",
                 "failing_input": "any error", "quote": "except:"},
                {"file": "a.py", "line_start": 1, "line_end": 1, "class": "style", "claim": "made up", "failing_input": "x",
                 "quote": "this is not in the file"},
            ]})
        return replies

    def _config(self):
        return cfg(rv("finder", "model-a", roles=("reasoning",)), rv("checker", "model-b", roles=("verifier",)),
                   rv("sorter", "model-c", roles=("classifier",)))

    def test_reasoning_then_grounding_then_verification_then_only_what_stands_is_classified(self):
        result, stub, probability = self._run(self._config(), probs(injection=0.9))
        self.assertEqual([m for m, _, _ in stub.calls], ["model-a", "model-b", "model-b"])  # one finder, two grounded findings checked
        self.assertEqual({m for m, _, _ in probability.calls}, {"model-c"})
        self.assertEqual(len(probability.calls), len(RUBRIC))  # the one finding that stood, once per entry
        classified = result.classification.classified
        self.assertEqual([(c.finding.claim, c.status, c.verification) for c in classified],
                         [("runs the input in a shell", CLASSIFIED, "confirmed")])
        self.assertEqual(len(result.verified.verification.rejected), 1)  # the bare except never reached the classifier
        self.assertEqual(len(result.verified.verification.ungrounded), 1)

    def test_a_disputed_finding_is_classified_too_and_keeps_its_verification_status(self):
        config = cfg(rv("finder", "model-a", roles=("reasoning",)), rv("checker", "model-b", roles=("verifier",)),
                     rv("skeptic", "model-d", roles=("verifier",)), rv("sorter", "model-c", roles=("classifier",)))

        def replies(model, system, content):
            if "checking ONE reported defect" in system and "Claim: runs the input in a shell" in content:
                if model == "model-d":
                    return json.dumps({"verdict": "rejected", "reasoning": "constant command"})
                return json.dumps({"verdict": "confirmed", "failing_input": "cmd='a; b'", "wrong_result": "runs b", "reasoning": "traced"})
            return self._replies()(model, system, content)

        result, _, _ = self._run(config, probs(injection=0.9), replies=replies, loaded=("model-a", "model-b", "model-c", "model-d"))
        self.assertEqual(len(result.verified.verification.disputed), 1)
        self.assertEqual([(c.finding.claim, c.status, c.verification) for c in result.classification.classified],
                         [("runs the input in a shell", CLASSIFIED, "disputed")])

    def test_with_no_classifier_active_the_review_still_returns_the_verified_findings_and_says_so(self):
        config = cfg(rv("finder", "model-a", roles=("reasoning",)), rv("checker", "model-b", roles=("verifier",)))
        result, stub, probability = self._run(config, probs())
        self.assertEqual(probability.calls, [])
        self.assertEqual([c.status for c in result.classification.classified], [UNCHECKED])
        text = format_classified_review(result)
        self.assertIn("No classifier ran, so these findings carry NO rubric level.", text)
        self.assertIn("runs the input in a shell", text)

    def test_a_classifier_model_that_is_not_loaded_refuses_before_the_reasoning_stage(self):
        stub = StubComplete(self._replies())
        with self.assertRaises(ModelNotLoadedError):
            run_classified_review(self._config(), ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a", "model-b"])), object(), self.DIFF,
                                  "/unused", "HEAD", 3, complete=stub, probability=StubProbability(probs()),
                                  read_file=lambda path: self.SOURCE)
        self.assertEqual(stub.calls, [])

    def test_the_formatted_result_groups_by_class_and_shows_entry_level_and_average_probability(self):
        result, _, _ = self._run(self._config(), probs(injection=0.95))
        text = format_classified_review(result)
        self.assertIn("1 classified, 0 without a rubric level; dropped 0 unclassified, 1 rejected and 1 ungrounded", text)
        self.assertIn("## injection", text)
        self.assertIn("- a.py:4 [injection] [finder] runs the input in a shell (verifier: confirmed)", text)
        self.assertIn("injection [high], average probability 0.95 over 1 classifier(s)", text)
        self.assertIn("## Dropped: rejected by the verifiers (1)", text)
        self.assertIn("## Dropped: ungrounded (1)", text)

    def test_a_confirmed_finding_keeps_the_failing_input_its_verifier_reproduced(self):
        result, _, _ = self._run(self._config(), probs(injection=0.95))
        text = format_classified_review(result)
        self.assertIn("    checker: failing input: cmd='a; b' -> runs b", text)

    def test_a_disputed_finding_keeps_what_each_verifier_said(self):
        config = cfg(rv("finder", "model-a", roles=("reasoning",)), rv("checker", "model-b", roles=("verifier",)),
                     rv("skeptic", "model-d", roles=("verifier",)), rv("sorter", "model-c", roles=("classifier",)))

        def replies(model, system, content):
            if "checking ONE reported defect" in system and "Claim: runs the input in a shell" in content:
                if model == "model-d":
                    return json.dumps({"verdict": "rejected", "reasoning": "constant command"})
                return json.dumps({"verdict": "confirmed", "failing_input": "cmd='a; b'", "wrong_result": "runs b", "reasoning": "traced the call"})
            return self._replies()(model, system, content)

        result, _, _ = self._run(config, probs(injection=0.9), replies=replies, loaded=("model-a", "model-b", "model-c", "model-d"))
        text = format_classified_review(result)
        self.assertIn("(verifier: disputed)", text)
        self.assertIn("    reported failing input: cmd='a; b'", text)
        self.assertIn("    checker confirmed: traced the call", text)
        self.assertIn("    skeptic rejected: constant command", text)

    def test_a_finding_without_a_rubric_level_keeps_its_verifier_evidence_too(self):
        config = cfg(rv("finder", "model-a", roles=("reasoning",)), rv("checker", "model-b", roles=("verifier",)))
        result, _, _ = self._run(config, probs())
        text = format_classified_review(result)
        self.assertIn("## No rubric level (1)", text)
        self.assertIn("    checker: failing input: cmd='a; b' -> runs b", text)

    def test_the_formatted_result_lists_an_unclassified_finding_as_dropped_with_its_averages(self):
        result, _, _ = self._run(self._config(), probs(injection=0.3))  # below the score of 0.6
        text = format_classified_review(result)
        self.assertIn("dropped 1 unclassified", text)
        self.assertIn("## Dropped: matched no rubric entry (1)", text)
        self.assertIn("average probability per entry: injection 0.30, crash 0.05, wrong-result 0.05, style 0.05", text)

    def test_the_formatted_result_explains_the_average_rule_and_never_mentions_a_rating_or_a_dispute(self):
        result, _, _ = self._run(self._config(), probs(injection=0.9))
        text = format_classified_review(result)
        self.assertIn("the average of every classifier's probability for that entry is at or above the entry's score", text)
        self.assertNotIn("from 1 to 5", text)
        self.assertNotIn("isputed (", text)

    def test_the_formatted_result_notes_an_overlap(self):
        result, _, _ = self._run(self._config(), probs(crash=0.95, injection=0.7))
        text = format_classified_review(result)
        self.assertIn("injection [high], average probability 0.70 over 1 classifier(s); also fits crash (took the higher level)", text)

    def test_two_classifiers_that_split_leave_the_finding_unclassified_and_the_averages_show_why(self):
        config = cfg(rv("finder", "model-a", roles=("reasoning",)), rv("checker", "model-b", roles=("verifier",)),
                     rv("sorter", "model-c", roles=("classifier",)), rv("sorter2", "model-b", roles=("classifier",)))
        values = lambda model, s, c: (0.9 if entry_of(c) == ("injection" if model == "model-c" else "crash") else NOT_A_FIT)  # noqa: E731
        result, _, _ = self._run(config, values)
        text = format_classified_review(result)
        self.assertIn("## Dropped: matched no rubric entry (1)", text)
        self.assertIn("average probability per entry: injection 0.48, crash 0.48, wrong-result 0.05, style 0.05", text)

    def test_a_review_error_from_the_classifier_stage_is_listed_under_problems(self):
        result, _, _ = self._run(self._config(), lambda *a: None)
        text = format_classified_review(result)
        self.assertIn("Problems (1):", text)
        self.assertIn("'sorter'", text)
        self.assertIn("every request failed", text)
        # The classifier stage's failure opens the text, before the counts.
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("INCOMPLETE REVIEW: 1 problem(s)"), lines[0])
        self.assertTrue(lines[1].startswith("Classified review:"), lines[1])

    def test_a_problem_in_any_stage_opens_the_text_and_a_clean_review_has_no_header(self):
        def text_for(reasoning_errors=(), verifier_errors=(), classifier_errors=(), warnings=()):
            verified = VerifiedReview(ReasoningReview(errors=list(reasoning_errors), warnings=list(warnings)),
                                      VerificationReview(errors=list(verifier_errors)))
            return format_classified_review(ClassifiedReview(verified, ClassificationReview(errors=list(classifier_errors))))
        for kwargs in ({"reasoning_errors": ["r"]}, {"verifier_errors": ["v"]}, {"classifier_errors": ["c"]}):
            self.assertTrue(text_for(**kwargs).startswith("INCOMPLETE REVIEW: 1 problem(s)"), kwargs)
        self.assertTrue(text_for(["r"], ["v"], ["c"]).startswith("INCOMPLETE REVIEW: 3 problem(s)"))
        for clean in (text_for(), text_for(warnings=["a note"])):
            self.assertTrue(clean.startswith("Classified review:"), clean)
            self.assertNotIn("INCOMPLETE", clean)


class EveryRequestFails(unittest.TestCase):
    def test_a_classifier_whose_every_request_fails_is_an_error_naming_it_not_a_crash(self):
        review = classify_findings(cfg(rv("c1")), ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])), object(), [fnd()], 3,
                                   probability=StubProbability(lambda *a: None))
        self.assertEqual(review.classified[0].status, UNCHECKED)
        self.assertTrue(any("'c1'" in e and "every request failed" in e for e in review.errors))


class RepliesCounted(unittest.TestCase):
    """How many requests a classifier sent, how many got a reply and how many of those it could read: a request that got no
    reply at all (a timeout, a refusal) is told apart from a reply that was no yes or no (a model without token probabilities)."""

    def counts(self, values, findings=None, reviewers=None):
        review = classify_findings(cfg(*(reviewers or [rv("c1")])), ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a", "model-b"])), object(),
                                   findings or [fnd()], 1, probability=StubProbability(values))
        return review, {o.reviewer.name: (o.value.counts.requests, o.value.counts.replies, o.value.counts.readable)
                        for o in review.outcomes if o.error is None}

    def test_every_request_that_got_a_probability_is_a_readable_reply(self):
        review, counts = self.counts(probs(injection=0.9))
        self.assertEqual(counts, {"c1": (len(RUBRIC), len(RUBRIC), len(RUBRIC))})
        self.assertEqual(sorted(review.outcomes[0].value.answers), [0])

    def test_a_request_with_no_reply_is_not_a_reply_and_a_reply_that_is_no_yes_or_no_is_not_readable(self):
        def values(model, system, content):
            entry = entry_of(content)
            if entry == "crash":
                return None  # no failure kind recorded: the request got no reply at all
            if entry == "wrong-result":
                openai_lib._failure_state.kind = "no_probabilities"  # a reply came back, with nothing the reader can use
            if entry == "style":
                openai_lib._failure_state.kind = "no_answer"
            return None if entry in ("wrong-result", "style") else 0.9

        review, counts = self.counts(values)
        # Four rubric entries: one answered, one with no reply, two replies that could not be read.
        self.assertEqual(counts, {"c1": (4, 3, 1)})
        self.assertEqual(review.outcomes[0].value.answers, {})  # an answer needs a probability for every entry

    def test_a_classifier_that_failed_has_an_error_and_every_asked_classifier_is_kept(self):
        review, counts = self.counts(lambda model, system, content: None if model == "model-a" else 0.9,
                                     reviewers=[rv("c1", "model-a"), rv("c2", "model-b")])
        self.assertEqual(counts, {"c2": (4, 4, 4)})
        self.assertEqual([(o.reviewer.name, o.error is None) for o in review.outcomes], [("c1", False), ("c2", True)])
        self.assertIn("every request failed", review.outcomes[0].error)


if __name__ == "__main__":
    unittest.main()
