#!/usr/bin/env python3
"""Harness 4 of 4: every configured reviewer, against test code with known answers.

    .venv/bin/python harness/live_all_reviewers.py                       # orders case, three lenses on one model
    .venv/bin/python harness/live_all_reviewers.py --config my-reviewers.json --case orders --runs 3   # a reviewers file as it stands
    .venv/bin/python harness/live_all_reviewers.py --config my-config.json --runs 3   # reviewers, rubric and stages from a config

The other three harnesses isolate one stage. This one runs the whole reviewers
file the way the `review` tool would (several reviewers, models shared or
swapped, resident and on-demand mixed) over the four-file `orders` case, and
scores the result: which of its six defects were found, which of its three
decoys drew a false alarm.

Without --config it builds three reasoning reviewers on --model, each
with a narrower lens (injection, bounds and state, general), which also
exercises the scheduler's grouping of reviewers that share a model.

With --config it uses that file's reviewers and rubric as written (a reviewers file is a valid config). Every
stage the `review` tool has runs; for a stage it does not have yet, the output
says so. Today that is reasoning only. The verifier and classifier
stages join here once they exist, and this is the script that should show them
working together: the decoy false alarms disappearing after verification, and
each surviving finding carrying a rubric level.

--config also takes a `stages` list: it picks which of reasoning, verifier and classifier to test
(see README.md), and a stage it leaves out is not run even when reviewers are configured for it.
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_lib as h  # noqa: E402
from cases import ALL_CASES, RUBRIC  # noqa: E402
from codebase_reviewer_config_lib import load_reviewers_file  # noqa: E402
from reasoning_reviewer_lib import format_review  # noqa: E402
from reviewer_scheduler_lib import describe_plan, plan_stages  # noqa: E402

LENSES = (
    ("injection", "You review Python changes ONLY for injection: shell, SQL, path or argument injection from untrusted input."),
    ("bounds-and-state", "You review Python changes ONLY for off-by-one errors, bad slicing, empty-input crashes, wrong formulas, "
                         "and state shared between calls such as mutable default arguments or unsynchronised counters."),
    ("general", h.DEFAULT_LENS),
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    h.add_model_options(parser, config=True)
    parser.add_argument("--case", default="orders", choices=list(ALL_CASES), help="the code under review (default orders)")
    parser.add_argument("--history", default=None, metavar="SNAPSHOT",
                        help="replay a Copilot review from a snapshot (collect_copilot_history.py) instead of --case, and score parity with it")
    parser.add_argument("--round", type=int, default=1, help="which Copilot review of the snapshot to replay (default 1)")
    return h.finish_args(parser.parse_args(argv))


def build_config(args: argparse.Namespace) -> tuple:
    """(reviewer entries, rubric) from --config's reviewers, else three lenses on --model."""
    if args.config_data is not None and args.config_data.reviewers:
        return list(args.config_data.reviewers), list(args.config_data.rubric)
    entries = [h.reviewer_entry(name, args.model, prompt=lens, prompt_file=None, on_demand=not args.resident, context=args.context)
               for name, lens in LENSES]
    return entries, list(RUBRIC)


def stage_notes(entries: list, stages: tuple) -> list:
    """What the selected stages ask for beyond reasoning, and whether this checkout can do it yet."""
    have = h.available_review_types()
    notes = []
    for stage in ("verifier", "classifier"):
        if stage not in stages:
            continue
        wanted = [e["name"] for e in entries if e.get("active", True) and stage in e.get("type", [])]
        if not wanted:
            continue
        if stage not in have:
            notes.append(f"{stage} reviewer(s) {wanted} configured but the {stage} stage is not built yet; not run.")
        else:
            notes.append(f"{stage} reviewer(s) {wanted} configured and `review` has a {stage!r} type, but this script has no runner "
                         f"for it yet: add one in `main` after the reasoning run.")
    return notes


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.show_config:
        print(h.describe_config(args))
        return 0
    entries, rubric = build_config(args)
    stages = h.selected_stages(args)
    if "reasoning" not in stages:
        print(f"stages selected: {list(stages)}; reasoning is not one of them, so no review runs here.")
        for note in stage_notes(entries, stages):
            print(note)
        return 0
    # Only the reviewers `run_reasoning` will actually run: a disabled reviewer's model, or one that only
    # takes a stage this script does not run, must not stop the others.
    reasoning = [e for e in entries if e.get("active", True) and "reasoning" in e.get("type", [])]
    h.preflight_reviewers(reasoning, args)
    if args.history:
        case, hist_base, hist_head = h.history_case(args.history, args.round - 1)
    else:
        case, hist_base, hist_head = ALL_CASES[args.case], None, None
    base_url = h.load_server().lmstudio_base_url()
    before = h.loaded_models(base_url)
    recorder = h.Recorder()
    active = len(reasoning)
    with tempfile.TemporaryDirectory(prefix="live-all-") as tmp:
        scratch = Path(tmp)
        plan_file = h.write_reviewers_file(scratch, entries, rubric)
        print(describe_plan(plan_stages(load_reviewers_file(str(plan_file)))))
        repo, base, head = (h.REPO_ROOT, hist_base, hist_head) if args.history else h.build_repo(scratch, case)
        scores = []
        all_findings = []
        for run in range(1, args.runs + 1):
            findings, review = h.run_reasoning(repo, base, head, entries, args, recorder, max_reviewers=max(active, 1), rubric=rubric)
            scores.append(h.score(findings, case))
            all_findings.append(findings)
            print(f"--- run {run}: {len(findings)} finding(s) from {review.reviewers_run}, {review.chunks} chunk(s)")
            for error in review.errors:
                print(f"  reviewer error: {error}")
            if run == args.runs:
                print(format_review(review))
    summarize = h.parity_line if args.history else h.seeded_line  # parity speaks of Copilot; a seeded case has bugs and decoys
    print(h.format_score(case, scores))
    if args.history:
        print(summarize(case, scores))
    if len(entries) > 1:
        print("per reviewer (findings each one raised, merged duplicates counted for both):")
        for entry in entries:
            alone = [h.score([f for f in run_findings if entry["name"] in f.reviewers], case) for run_findings in all_findings]
            print(f"  {entry['name']:<20} {entry['model']:<34} {summarize(case, alone)}")
    for note in stage_notes(entries, stages):
        print(note)
    print(h.report_run(args, recorder, base_url, before))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
