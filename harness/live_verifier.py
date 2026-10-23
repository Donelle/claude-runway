#!/usr/bin/env python3
"""Harness 3 of 4: the verifier stage against a real LM Studio model.

    .venv/bin/python harness/live_verifier.py --check    # validate and summarise the fixtures
    .venv/bin/python harness/live_verifier.py            # run the stage (once it exists)

The verifier is shown a finding and the code it cites, and must confirm it with
a concrete failing input or reject it. The fixtures (`cases.verifier_cases`) are
that exact pair of things with the answer attached: every seeded bug as a true
finding, every decoy (code that looks wrong and is not) as a false one.

What a run will report, once the stage exists:
- true findings confirmed / wrongly rejected (the verifier must not lose real bugs),
- decoy findings rejected / wrongly confirmed (the whole point of the stage),
- with several verifiers, how often they disagree (a dispute).

The stage is not built, so a run exits with status 3 and says so. When it
lands, `review` gains a verifier type; fill in `run_stage` below with that
type's parameters and the scoring in `score_verdicts`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_lib as h  # noqa: E402
from cases import ORDERS, verifier_cases  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    h.add_model_options(parser)
    parser.add_argument("--check", action="store_true", help="validate and summarise the fixtures; needs no LM Studio")
    return parser.parse_args(argv)


def summarise() -> str:
    cases = verifier_cases(ORDERS)
    real = [c for c in cases if c.real]
    false = [c for c in cases if not c.real]
    lines = [f"{len(cases)} verifier fixture(s): {len(real)} real finding(s) to confirm, {len(false)} false finding(s) to reject"]
    for c in cases:
        lines.append(f"  {'CONFIRM' if c.real else 'REJECT '} {c.id:<34} {c.file}:{c.line}  {c.claim[:70]}")
    return "\n".join(lines)


def score_verdicts(verdicts: dict) -> str:
    """`verdicts` maps fixture id -> True (confirmed) / False (rejected) / None (disputed or no answer)."""
    cases = verifier_cases(ORDERS)
    kept = sum(1 for c in cases if c.real and verdicts.get(c.id) is True)
    lost = [c.id for c in cases if c.real and verdicts.get(c.id) is False]
    caught = sum(1 for c in cases if not c.real and verdicts.get(c.id) is False)
    leaked = [c.id for c in cases if not c.real and verdicts.get(c.id) is True]
    disputed = [c.id for c in cases if verdicts.get(c.id) is None]
    real_total = sum(1 for c in cases if c.real)
    false_total = len(cases) - real_total
    return "\n".join([
        f"real findings confirmed: {kept}/{real_total}   wrongly rejected: {lost}",
        f"false findings rejected: {caught}/{false_total}   wrongly confirmed: {leaked}",
        f"disputed or unanswered: {disputed}",
    ])


def run_stage(args: argparse.Namespace) -> dict:
    h.refuse_unbuilt("verifier", "call the new type with the findings and code of each `cases.verifier_cases()` entry "
                     "and return {fixture id: True | False | None}, then print `score_verdicts(...)`.")
    raise AssertionError("unreachable: refuse_unbuilt always exits")


def main(argv=None) -> int:
    args = parse_args(argv)
    print(summarise())
    if args.check:
        return 0
    h.preflight(args.model)
    print(score_verdicts(run_stage(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
