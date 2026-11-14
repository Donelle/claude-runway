#!/usr/bin/env python3
"""Harness 3 of 4: the verifier stage against a real LM Studio model.

    .venv/bin/python harness/live_verifier.py --check                 # validate and summarise the fixtures; needs no LM Studio
    .venv/bin/python harness/live_verifier.py --model <lm-studio-id>  # one verifier built from the flags
    .venv/bin/python harness/live_verifier.py --config my-config.json --runs 3   # the config's active verifier reviewers
    .venv/bin/python harness/live_verifier.py --config my-config.json --show-config   # what a config resolves to, no LM Studio

The verifier is shown a finding and the code it cites, and must confirm it with
a concrete failing input or reject it. The fixtures (`cases.verifier_cases`) are
that exact pair of things with the answer attached: every seeded bug as a true
finding, every decoy (code that looks wrong and is not) as a false one. Every
fixture is reported with the same class, and a decoy carries a plausible failing
input like a real finding does, so neither tells the verifier which kind it is.

A run reports:
- true findings confirmed / wrongly rejected (the verifier must not lose real bugs),
- decoy findings rejected / wrongly confirmed (the whole point of the stage),
- with several verifiers, the findings they disagree on (a dispute) or that none answered.

With --config, every active reviewer in the file that has the `verifier` role runs, each with its
own model and prompt, instead of the single reviewer --model/--prompt build; see README.md for the
file. A config whose `stages` leave out `verifier` makes this script exit 0 without loading a model.

Needs LM Studio running with its server enabled and the model downloaded; the model is loaded and
unloaded by the scheduler unless --resident is given. A verifier is a reasoning-distilled model that
thinks before it answers, so the up-front turn-ending check gives it a 2000-token cap instead of 40.
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_lib as h  # noqa: E402
from cases import fixture_cases, merged_case, verifier_cases_for  # noqa: E402
from reasoning_reviewer_lib import Finding  # noqa: E402
from verifier_reviewer_lib import CONFIRMED, REJECTED  # noqa: E402

# One class for every fixture: a bug's own class, or "decoy", would hand the verifier the answer.
FIXTURE_CLASS = "uncategorized"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    h.add_model_options(parser, config=True)
    h.add_run_type_option(parser, live=False)
    parser.add_argument("--check", action="store_true", help="validate and summarise the fixtures; needs no LM Studio")
    parser.add_argument("--case", action="append", choices=[*fixture_cases(), "all"], default=None,
                        help="case whose bugs and decoys become fixtures (repeatable, or `all`); default orders, "
                             "the set the verifier baselines were measured on")
    return h.finish_args(parser.parse_args(argv))


def chosen_cases(args: argparse.Namespace) -> list:
    """The cases the fixtures come from: `--case` (once or several times, or `all`), else `orders`."""
    names = args.case or ["orders"]
    available = fixture_cases()
    return list(available.values()) if "all" in names else [available[name] for name in dict.fromkeys(names)]


def cases_line(args: argparse.Namespace) -> str:
    """The cases this run uses, by name: what `live-tests/run_live_tests.py` compares with the cases a baseline was counted on."""
    return "cases: " + ", ".join(case.name for case in chosen_cases(args))


def summarise(args: argparse.Namespace) -> str:
    cases = verifier_cases_for(chosen_cases(args))
    real = [c for c in cases if c.real]
    false = [c for c in cases if not c.real]
    lines = [f"{len(cases)} verifier fixture(s): {len(real)} real finding(s) to confirm, {len(false)} false finding(s) to reject"]
    for c in cases:
        lines.append(f"  {'CONFIRM' if c.real else 'REJECT '} {c.id:<34} {c.file}:{c.line}  {c.claim[:70]}")
    return "\n".join(lines)


def score_verdicts(verdicts: dict, cases: list) -> str:
    """`verdicts` maps fixture id -> True (confirmed) / False (rejected) / None (disputed or no answer)."""
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


def fixture_finding(case) -> Finding:
    """A fixture as the finding the reasoning stage would hand the verifier stage."""
    return Finding(case.file, case.line, case.line, FIXTURE_CLASS, case.claim, case.failing_input, ("fixture",), quote=case.quote)


def verdicts_from(review, cases: list) -> dict:
    """{fixture id: True | False | None} from a `VerificationReview`.

    Confirmed is True and rejected False. A dispute, an unchecked finding and an ungrounded one (a fixture whose quote the
    grounding step could not find, which is a defect in the fixture) are None: nobody gave an answer to score."""
    by_key = {(c.file, c.claim): c.id for c in cases}
    verdicts: dict = {c.id: None for c in cases}
    for v in review.verified:
        key = (v.finding.file, v.finding.claim)
        if key in by_key and v.status in (CONFIRMED, REJECTED):
            verdicts[by_key[key]] = v.status == CONFIRMED
    return verdicts


def describe_misses(review, cases: list, verdicts: dict) -> list:
    """What each verifier said about every fixture that was not answered correctly (so a wrong verdict can be read)."""
    by_key = {(c.file, c.claim): c for c in cases}
    lines = []
    for v in review.verified:
        case = by_key.get((v.finding.file, v.finding.claim))
        if case is None or verdicts[case.id] is case.real:
            continue
        lines.append(f"  {case.id}: {v.status}" + (f" ({v.note})" if v.note else ""))
        for d in v.decisions:
            lines.append(f"    {d.reviewer} {d.verdict}: {d.reasoning or d.wrong_result or '(no reason given)'}")
    for u in review.ungrounded:
        case = by_key.get((u.finding.file, u.finding.claim))
        if case is not None:
            lines.append(f"  {case.id}: ungrounded ({u.reason}): the fixture's quote is wrong")
    return lines


def run_stage(args: argparse.Namespace, verifiers: list, recorder: h.Recorder, scratch: Path) -> tuple:
    """One run: every fixture as a finding, over a throwaway repository holding the chosen cases' code. Returns (verdicts, review)."""
    cases = verifier_cases_for(chosen_cases(args))
    repo, _base, head = h.build_repo(scratch, merged_case(chosen_cases(args)))
    review = h.run_verifier(repo, head, [fixture_finding(c) for c in cases], verifiers, args, recorder)
    return verdicts_from(review, cases), review


def main(argv=None) -> int:
    args = parse_args(argv)
    h.refuse_live(args, "verifier", "this script runs the fixtures built from cases.py, and there are no live fixtures yet "
                                    "(run live_all_reviewers.py --run-type live to have the verifier check a live review's findings)")
    if args.show_config:
        print(h.describe_config(args))
        return 0
    if args.check:
        print(summarise(args))
        return 0
    if not h.stage_selected(args, "verifier"):
        print(f"--config selects the stage(s) {list(h.selected_stages(args))}; verifier is not one of them, so there is nothing for this script to run.")
        return 0
    verifiers = h.stage_reviewers(args, "verifier")
    if not verifiers:
        raise SystemExit("no active reviewer with the `verifier` type to run: add one to the config, or leave --config out and pass --model.")
    print(summarise(args))
    h.preflight_reviewers(verifiers, args)
    base_url = h.load_server().lmstudio_base_url()
    before = h.loaded_models(base_url)
    recorder = h.Recorder()
    cases = verifier_cases_for(chosen_cases(args))
    print(cases_line(args))
    print(f"verifiers: {h.describe_reviewers(verifiers)}  runs={args.runs}")
    for run in range(1, args.runs + 1):
        with tempfile.TemporaryDirectory(prefix="live-verifier-") as tmp:
            verdicts, review = run_stage(args, verifiers, recorder, Path(tmp))
        print(f"--- run {run}")
        print(score_verdicts(verdicts, cases))
        for line in describe_misses(review, cases, verdicts):
            print(line)
        for error in review.errors:
            print(f"  verifier problem: {error}")
        for warning in review.warnings:
            print(f"  warning: {warning}")
    print(h.report_run(args, recorder, base_url, before))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
