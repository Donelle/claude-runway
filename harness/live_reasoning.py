#!/usr/bin/env python3
"""Harness 1 of 4: the reasoning stage against a real LM Studio model.

    .venv/bin/python harness/live_reasoning.py                       # seeded-three + clean, default model
    .venv/bin/python harness/live_reasoning.py --case orders --runs 3
    .venv/bin/python harness/live_reasoning.py --prompt-file my-lens.md --model qwen/qwen2.5-coder-14b
    .venv/bin/python harness/live_reasoning.py --case own-diff        # this checkout's branch vs origin/main, unscored
    .venv/bin/python harness/live_reasoning.py --case own-planted     # the same, with one shell-injection planted in a real file
    .venv/bin/python harness/live_reasoning.py --config my-config.json --show-config   # what a config resolves to, no LM Studio

With --config, every active reviewer in the file that has the `reasoning` role runs, each with its own model and
prompt, instead of the single reviewer --model/--prompt build; see README.md for the file.

What it answers: does this model, with this prompt and chunk size, find the bugs
that are known to be in the diff, and how many false alarms does it raise on
code that is fine? Cases and their answers are in `cases.py`.

`own-diff` and `own-planted` use this repository's real diff (many chunks, a
real large file). `own-planted` needs a throwaway clone so the planted defect
never touches the working tree.

Needs LM Studio running with its server enabled and the model downloaded; the
model is loaded and unloaded by the scheduler unless --resident is given.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_lib as h  # noqa: E402
from cases import ALL_CASES  # noqa: E402
from reasoning_reviewer_lib import format_review  # noqa: E402

PLANT_FILE = "mcpservers/codebase_reviewer_mcp_server.py"
PLANT_OLD = "done = subprocess.run(command, capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False)"
PLANT_NEW = ('done = subprocess.run("git -C " + repo_path + " diff " + base_ref + "..." + head_ref, '
             "shell=True, capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False)")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    h.add_model_options(parser, config=True)
    parser.add_argument("--case", action="append", choices=[*ALL_CASES, "own-diff", "own-planted"],
                        help="case to run (repeatable); default seeded-three and clean")
    parser.add_argument("--base", default="origin/main", help="base ref for own-diff / own-planted")
    parser.add_argument("--history", action="append", default=[], metavar="SNAPSHOT",
                        help="a Copilot history snapshot (see collect_copilot_history.py): review the diff Copilot reviewed "
                             "and score parity with its findings (repeatable)")
    parser.add_argument("--round", type=int, default=1, help="which Copilot review of the snapshot to replay (default 1, the first)")
    return h.finish_args(parser.parse_args(argv))


def plant_defect(clone: Path) -> None:
    target = clone / PLANT_FILE
    text = target.read_text(encoding="utf-8")
    if PLANT_OLD not in text:
        raise SystemExit(f"cannot plant: {PLANT_OLD!r} is no longer in {PLANT_FILE}. Update PLANT_OLD in this script.")
    target.write_text(text.replace(PLANT_OLD, PLANT_NEW), encoding="utf-8")
    h.git(clone, "commit", "-q", "-am", "plant a shell injection")


def run_known_case(name: str, args: argparse.Namespace, scratch: Path, recorder: h.Recorder, reviewers: list) -> None:
    case = ALL_CASES[name]
    repo, base, head = h.build_repo(scratch, case)
    scores = []
    for run in range(1, args.runs + 1):
        findings, review = h.run_reasoning(repo, base, head, reviewers, args, recorder)
        scores.append(h.score(findings, case))
        for error in review.errors:
            print(f"  run {run}: reviewer error: {error}")
    print(h.format_score(case, scores))


def run_own(name: str, args: argparse.Namespace, scratch: Path, recorder: h.Recorder, reviewers: list) -> None:
    # Resolve the base here, in this checkout. A clone's `origin` is this repository, so its own `origin/main`
    # would be this checkout's local `main`, which may be missing or stale; a commit id means the same
    # thing in both.
    found = subprocess.run(["git", "-C", str(h.REPO_ROOT), "rev-parse", "--verify", "--quiet", f"{args.base}^{{commit}}"],
                           capture_output=True, text=True, check=False)
    if found.returncode != 0:
        raise SystemExit(f"--base {args.base!r} does not name a commit in this checkout; fetch it or pass another ref.")
    repo, base, head = h.REPO_ROOT, found.stdout.strip(), "HEAD"
    if name == "own-planted":
        clone = scratch / "clone"
        subprocess.run(["git", "clone", "-q", "--local", str(h.REPO_ROOT), str(clone)], check=True, capture_output=True)
        plant_defect(clone)
        repo = clone
    for run in range(1, args.runs + 1):
        _findings, review = h.run_reasoning(repo, base, head, reviewers, args, recorder)
        print(f"--- {name} run {run}: {review.chunks} chunk(s)")
        print(format_review(review))


def run_history(path: str, args: argparse.Namespace, recorder: h.Recorder, reviewers: list) -> None:
    """Review the diff Copilot reviewed, from recorded history; Copilot itself is never contacted."""
    case, base, head = h.history_case(path, args.round - 1)
    print(case.description)
    scores = []
    for run in range(1, args.runs + 1):
        findings, review = h.run_reasoning(h.REPO_ROOT, base, head, reviewers, args, recorder)
        scores.append(h.score(findings, case))
        print(f"  run {run}: {len(findings)} finding(s), {review.chunks} chunk(s), {len(review.errors)} reviewer error(s), {len(review.warnings)} warning(s)")
    print(h.format_score(case, scores))
    print(h.parity_line(case, scores))


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.show_config:
        print(h.describe_config(args))
        return 0
    if not h.stage_selected(args, "reasoning"):
        print(f"--config selects the stage(s) {list(h.selected_stages(args))}; reasoning is not one of them, so there is nothing for this script to run.")
        return 0
    reviewers = h.stage_reviewers(args, "reasoning")
    h.preflight_reviewers(reviewers, args)
    base_url = h.load_server().lmstudio_base_url()
    before = h.loaded_models(base_url)
    recorder = h.Recorder()
    with tempfile.TemporaryDirectory(prefix="live-reasoning-") as tmp:
        scratch = Path(tmp)
        for path in args.history:
            print(f"===== history {path}  reviewers: {h.describe_reviewers(reviewers)}  runs={args.runs}")
            run_history(path, args, recorder, reviewers)
        for name in args.case or ([] if args.history else ["seeded-three", "clean"]):
            print(f"===== {name}  reviewers: {h.describe_reviewers(reviewers)}  runs={args.runs}")
            if name in ALL_CASES:
                run_known_case(name, args, scratch, recorder, reviewers)
            else:
                run_own(name, args, scratch, recorder, reviewers)
    print(h.report_run(args, recorder, base_url, before))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
