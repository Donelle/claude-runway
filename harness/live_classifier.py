#!/usr/bin/env python3
"""Harness 2 of 4: the classifier stage against a real LM Studio model.

    .venv/bin/python harness/live_classifier.py --check    # validate and summarise the fixtures
    .venv/bin/python harness/live_classifier.py            # run the stage (once it exists)

The classifier is given a verified finding and the rubric, and returns the
rubric entry that applies and a confidence from 1 to 5. The finding takes that
entry's level only when the confidence reaches the entry's `score`; a finding
that matches nothing is dropped as unclassified, and one that fits several
entries takes the highest level.

The fixtures (`cases.RUBRIC`, `cases.classifier_cases`) cover all four outcomes:
a clear match for each rubric entry, a style-only remark, a finding that fits no
entry (it must come back unclassified), and one that fits two (the higher level
must win).

What a run will report, once the stage exists: the share of findings landing on
the expected entry, the unclassified one being dropped, and how confidence
compares with each entry's `score` threshold (a model that is always "5" makes
the threshold meaningless).

The stage is not built, so a run exits with status 3 and says so. When it
lands, fill in `run_stage` with the new `review` type's parameters.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_lib as h  # noqa: E402
from cases import RUBRIC, classifier_cases  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    h.add_model_options(parser)
    parser.add_argument("--check", action="store_true", help="validate and summarise the fixtures; needs no LM Studio")
    return parser.parse_args(argv)


def summarise() -> str:
    cases = classifier_cases()
    entries = ", ".join("{} ({}, score {})".format(e["name"], e["level"], e["score"]) for e in RUBRIC)
    lines = [f"rubric: {entries}", f"{len(cases)} classifier fixture(s)"]
    for c in cases:
        lines.append(f"  expect {str(c.expected):<24} {c.id:<34} {c.claim[:60]}")
    return "\n".join(lines)


def score_choices(choices: dict) -> str:
    """`choices` maps fixture id -> (entry name or None, confidence 1-5 or None)."""
    cases = classifier_cases()
    right = [c.id for c in cases if choices.get(c.id, (None, None))[0] == c.expected]
    wrong = [f"{c.id}: expected {c.expected}, got {choices.get(c.id, (None, None))[0]}" for c in cases if c.id not in right]
    confidences = [conf for _, conf in choices.values() if conf is not None]
    spread = f"confidence values seen: {sorted(set(confidences))}" if confidences else "no confidences returned"
    return "\n".join([f"on the expected entry: {len(right)}/{len(cases)}", *(f"  {w}" for w in wrong), spread])


def run_stage(args: argparse.Namespace) -> dict:
    h.refuse_unbuilt("classifier", "call the new type with each `classifier_cases()` claim and `RUBRIC` "
                     "and return {fixture id: (entry name or None, confidence)}, then print `score_choices(...)`.")
    raise AssertionError("unreachable: refuse_unbuilt always exits")


def main(argv=None) -> int:
    args = parse_args(argv)
    print(summarise())
    if args.check:
        return 0
    h.preflight(args.model)
    print(score_choices(run_stage(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
