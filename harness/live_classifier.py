#!/usr/bin/env python3
"""Harness 2 of 4: the classifier stage against a real LM Studio model.

    .venv/bin/python harness/live_classifier.py --check                 # validate and summarise the fixtures; needs no LM Studio
    .venv/bin/python harness/live_classifier.py --model <lm-studio-id>  # one classifier built from the flags
    .venv/bin/python harness/live_classifier.py --config my-config.json --runs 3   # the config's active classifier reviewers
    .venv/bin/python harness/live_classifier.py --config my-config.json --show-config   # what a config resolves to, no LM Studio

The classifier is given a verified finding and, one request per entry, that rubric
entry's name, level and definition (never its `score`): does the finding fall under
this definition? The answer is the model's probability of "yes", read from its token
probabilities at temperature 0, so asking again gives the same number. Several
classifiers are averaged: the average probability for an entry is what is compared
with that entry's `score` (0 to 1). The finding takes the level of an entry whose
average reaches its `score`; a finding that reaches none is dropped as unclassified,
and one that reaches several takes the highest level. A model that returns no token
probabilities cannot be a classifier.

The fixtures (`cases.RUBRIC`, `cases.classifier_cases`) cover all four outcomes:
a clear match for each rubric entry, a style-only remark, a finding that fits no
entry (it must come back unclassified), and one that fits two (the higher level
must win). Each is shown to the classifier as a finding with the same class, so
the class does not give the answer away, and with no code: a classifier reads
the finding, not the file.

A run reports:
- the share of findings landing on the expected entry (and each one that did not),
- whether the finding that fits nothing came back unclassified,
- whether the finding that fits two entries took the higher level,
- the average probability each entry got, over the fixtures, against that entry's `score` (when no
  average falls below its `score`, the scores dropped nothing),
- the findings that none answered,
- with --runs above 1, whether an entry's average came out the same from run to run on the same
  fixture (within `REPEAT_TOLERANCE`): the numbers are not sampled, so they should.

The rubric is the config's `rubric` when it has one, and `cases.RUBRIC` otherwise.
With --config, every active reviewer in the file that has the `classifier` role runs, each
with its own model and prompt, instead of the single reviewer --model/--prompt build; see
README.md for the file. A config whose `stages` leave out `classifier` makes this script
exit 0 without loading a model.

Needs LM Studio running with its server enabled and the model downloaded; the model is
loaded and unloaded by the scheduler unless --resident is given.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_lib as h  # noqa: E402
from cases import RUBRIC, classifier_cases  # noqa: E402
from classifier_reviewer_lib import CLASSIFIED, UNCHECKED, UNCLASSIFIED, reaches  # noqa: E402
from reasoning_reviewer_lib import Finding  # noqa: E402

# One class and one file for every fixture: a bug's own class would hand the classifier the answer.
FIXTURE_CLASS = "uncategorized"
FIXTURE_FILE = "app.py"
# The same fixture's average for the same entry from two runs differs by no more than this, or the run is reported as not repeating.
# One tenth is the old drift rule (an average within 1.0 of another on the 1-5 scale was the same result) in tenths of a probability.
# The numbers are read from token probabilities at temperature 0 and in practice repeat to about a thousandth (0.9242 and 0.9252
# were measured: the GPU's arithmetic is not bit-for-bit the same from one run to the next).
REPEAT_TOLERANCE = 0.1


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    h.add_model_options(parser, config=True)
    h.add_run_type_option(parser, live=False)
    parser.add_argument("--check", action="store_true", help="validate and summarise the fixtures; needs no LM Studio")
    return h.finish_args(parser.parse_args(argv))


def rubric_for(args: argparse.Namespace) -> list:
    """The rubric the stage runs against: the config's, when it has one, else `cases.RUBRIC`."""
    if args.config_data is not None and args.config_data.rubric:
        return list(args.config_data.rubric)
    return [dict(entry) for entry in RUBRIC]


def summarise(args: argparse.Namespace) -> str:
    cases = classifier_cases()
    entries = ", ".join("{} ({}, score {:g})".format(e["name"], e["level"], e["score"]) for e in rubric_for(args))
    lines = [f"rubric: {entries}", f"{len(cases)} classifier fixture(s)"]
    for c in cases:
        lines.append(f"  expect {str(c.expected):<24} {c.id:<34} {c.claim[:60]}")
    return "\n".join(lines)


def fixture_finding(case) -> Finding:
    """A fixture as the finding the earlier stages would hand the classifier stage."""
    return Finding(FIXTURE_FILE, 1, 1, FIXTURE_CLASS, case.claim, case.failing_input, ("fixture",))


def choices_from(review, cases: list) -> dict:
    """{fixture id: (entry name or None, probability or None, status)} from a `ClassificationReview`.

    The findings come back in the order they went in, so the nth classified item is the nth fixture. An entry is given
    only for a finding that was classified, the probability being the average of the entry it took. The status is what the
    stage made of the finding (`classified`, `unclassified`, or `unchecked` for one nobody answered): a finding that fits
    nothing is only right when it came back unclassified, never when it simply got no answer."""
    choices: dict = {c.id: (None, None, UNCHECKED) for c in cases}
    for case, item in zip(cases, review.classified, strict=True):
        choices[case.id] = (item.entry, item.confidence, item.status) if item.status == CLASSIFIED else (None, None, item.status)
    return choices


def entry_averages(review) -> dict:
    """{entry name: [the average probability of every finding that got an answer]}, including averages below the entry's
    score (which is what shows whether the scores are doing anything)."""
    averages: dict = {}
    for item in review.classified:
        for score in item.scores:
            averages.setdefault(score.entry, []).append(score.mean)
    return averages


def fixture_means(review, cases: list) -> dict:
    """{fixture id: {entry name: average probability}} for the fixtures that got an answer."""
    return {case.id: {s.entry: s.mean for s in item.scores} for case, item in zip(cases, review.classified, strict=True) if item.scores}


def repeat_lines(runs: list) -> list:
    """The report on whether the averages repeated over `runs` (a list of `fixture_means` results, one per run): the entry
    averages that differ by more than `REPEAT_TOLERANCE` between the runs, each named. Nothing for fewer than two runs."""
    if len(runs) < 2:
        return []
    differ = []
    compared = 0
    for case_id in sorted({c for run in runs for c in run}):
        for entry in sorted({e for run in runs for e in run.get(case_id, {})}):
            values = [run[case_id][entry] for run in runs if entry in run.get(case_id, {})]
            if len(values) < 2:
                continue
            compared += 1
            if max(values) - min(values) > REPEAT_TOLERANCE:
                differ.append(f"  differs: {case_id} / {entry}: " + ", ".join(f"{v:.4f}" for v in values))
    head = f"score repeat over {len(runs)} runs: {len(differ)} differ of {compared} averages compared (tolerance {REPEAT_TOLERANCE:g})"
    return [head, *differ]


def score_choices(choices: dict, averages: dict, rubric: list) -> str:
    """`choices` maps fixture id -> (entry name or None, average probability or None, status); `averages` is `entry_averages`'s result.

    A fixture is on its expected entry when it was classified as that entry; one that fits no entry is right only when its
    status is `unclassified`, so a finding nobody answered (unchecked) is not credited as a correct drop."""
    all_cases = classifier_cases()
    names = {entry["name"] for entry in rubric}
    # A fixture that expects an entry the rubric does not have can never pass, so it is left out of the score (and said so)
    # rather than counted as a miss: a config with its own rubric would otherwise be graded against entries it never defined.
    cases = [c for c in all_cases if c.expected is None or c.expected in names]
    unscorable = [c for c in all_cases if c not in cases]
    unanswered = (None, None, UNCHECKED)

    def right(case) -> bool:
        entry, _, status = choices.get(case.id, unanswered)
        return status == UNCLASSIFIED if case.expected is None else (status == CLASSIFIED and entry == case.expected)

    def got(case) -> str:
        entry, _, status = choices.get(case.id, unanswered)
        return str(entry) if status == CLASSIFIED else ("not answered" if status == UNCHECKED else "unclassified")

    hits = [c.id for c in cases if right(c)]
    wrong = [f"{c.id}: expected {c.expected}, got {got(c)}" for c in cases if c.id not in hits]
    lines = [f"on the expected entry: {len(hits)}/{len(cases)}", *(f"  {w}" for w in wrong)]
    if unscorable:
        missing = sorted({c.expected for c in unscorable})
        lines.append(f"left out of the score: {len(unscorable)} fixture(s) expect an entry the rubric does not have ({', '.join(missing)})")
    nothing = [c for c in cases if c.expected is None]
    if nothing:
        dropped = sum(1 for c in nothing if choices.get(c.id, unanswered)[2] == UNCLASSIFIED)
        lines.append(f"fits no entry, came back unclassified: {dropped}/{len(nothing)}")
    overlapping = [c for c in cases if c.overlap]
    if overlapping:
        took = sum(1 for c in overlapping if right(c))
        lines.append(f"fits two entries, took the higher level ({overlapping[0].expected}): {took}/{len(overlapping)}")
    lines.append("average probability per entry against its score:")
    below_any = False
    for entry in rubric:
        given = sorted(averages.get(entry["name"], []))
        if given:
            reached = sum(1 for value in given if reaches(value, entry["score"]))
            below_any = below_any or reached < len(given)
            lines.append(f"  {entry['name']:<26} score {entry['score']:g}: averages {[round(v, 3) for v in given]}, "
                         f"{reached} at or above the score, {len(given) - reached} below")
        else:
            lines.append(f"  {entry['name']:<26} score {entry['score']:g}: no answers")
    if not averages:
        lines.append("no probabilities returned")
    elif not below_any:
        lines.append("no average fell below its entry's score: the scores dropped nothing")
    return "\n".join(lines)


def unanswered_lines(review) -> list:
    """One line per finding no classifier gave a usable answer for, which `live-tests/run_live_tests.py` reads."""
    return [f"  not answered: {item.finding.claim[:70]} ({item.note})" for item in review.unchecked]


def run_stage(args: argparse.Namespace, classifiers: list, rubric: list, recorder: h.Recorder) -> tuple:
    """One run: every fixture as a finding. Returns (choices, review)."""
    cases = classifier_cases()
    review = h.run_classifier([fixture_finding(c) for c in cases], classifiers, rubric, args, recorder)
    return choices_from(review, cases), review


def main(argv=None) -> int:
    args = parse_args(argv)
    h.refuse_live(args, "classifier", "this script runs the classifier's own fixtures, and there are no live fixtures yet "
                                      "(run live_all_reviewers.py --run-type live to have the classifier sort a live review's findings)")
    if args.show_config:
        print(h.describe_config(args))
        return 0
    if args.check:
        print(summarise(args))
        return 0
    if not h.stage_selected(args, "classifier"):
        print(f"--config selects the stage(s) {list(h.selected_stages(args))}; classifier is not one of them, so there is nothing for this script to run.")
        return 0
    classifiers = h.stage_reviewers(args, "classifier")
    if not classifiers:
        raise SystemExit("no active reviewer with the `classifier` type to run: add one to the config, or leave --config out and pass --model.")
    rubric = rubric_for(args)
    print(summarise(args))
    h.preflight_reviewers(classifiers, args)
    h.isolate_reviewer_db()  # past every exit that runs no review, the preflight included: from here the run records
    base_url = h.load_server().lmstudio_base_url()
    before = h.loaded_models(base_url)
    recorder = h.Recorder()
    print(f"classifiers: {h.describe_reviewers(classifiers)}  runs={args.runs}")
    means: list = []
    for run in range(1, args.runs + 1):
        choices, review = run_stage(args, classifiers, rubric, recorder)
        means.append(fixture_means(review, classifier_cases()))
        print(f"--- run {run}")
        print(score_choices(choices, entry_averages(review), rubric))
        for line in unanswered_lines(review):
            print(line)
        for error in review.errors:
            print(f"  classifier problem: {error}")
        for warning in review.warnings:
            print(f"  warning: {warning}")
    for line in repeat_lines(means):
        print(line)
    print(h.report_run(args, recorder, base_url, before))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
