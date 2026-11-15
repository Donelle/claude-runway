#!/usr/bin/env python3
"""Harness checks that need no LM Studio: exit codes and messages only.

    .venv/bin/python harness/live-tests/preflight/run_checks.py           # run every check
    .venv/bin/python harness/live-tests/preflight/run_checks.py --list    # what the checks are
    .venv/bin/python harness/live-tests/preflight/run_checks.py --only stages   # checks whose name contains "stages"

Each check runs one harness script the way a person would, and compares its exit code and
output with what it should be. They cover what the scripts do BEFORE they touch a model: reading
a `--config`, choosing the stages, refusing a bad combination, and printing what a run would use.

Every check runs with `CLAUDE_RUNWAY_LMSTUDIO_URL` pointing at an address nothing listens on, so a
check that wrongly reaches for LM Studio fails instead of quietly using the real one. The real
runs are in the folder above this one, and its `run_live_tests.py` runs these checks first and stops when one fails.

To add a check, append a `Check` to `STATIC_CHECKS` (and, if it needs a config file, put one in
`configs/`). The live-tests configs need no line here: there is one `--show-config` check for every config in
`harness/live-tests/configs/`, found by listing the folder, so a new config is checked as soon as it exists. `{configs}`, `{live_configs}` and `{example}` in `args` stand for
`harness/live-tests/preflight/configs`, `harness/live-tests/configs` and `harness/config.example.json`.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE_TESTS = HERE.parent
if str(LIVE_TESTS) not in sys.path:
    sys.path.insert(0, str(LIVE_TESTS))

import live_configs as lc  # noqa: E402

HARNESS = LIVE_TESTS.parent
REPO_ROOT = HARNESS.parent
TIMEOUT_SECONDS = 60
# Nothing listens here: a check that reaches for LM Studio gets a connection error, not a model.
NO_LMSTUDIO_URL = "http://127.0.0.1:9/v1"


@dataclass(frozen=True)
class Check:
    name: str
    script: str  # a script in harness/
    args: tuple
    exit_code: int
    contains: tuple = ()  # each must appear in the output (stdout and stderr together)
    absent: tuple = ()  # none may appear


STATIC_CHECKS = (
    Check("verifier fixtures validate (--check)", "live_verifier.py", ("--check",), 0,
          contains=("9 verifier fixture(s): 6 real finding(s) to confirm, 3 false finding(s) to reject",)),
    Check("verifier fixtures of every case validate (--case all --check)", "live_verifier.py", ("--case", "all", "--check"), 0,
          contains=("51 verifier fixture(s): 38 real finding(s) to confirm, 13 false finding(s) to reject",)),
    Check("example config resolves for the verifier (--show-config)", "live_verifier.py",
          ("--config", "{example}", "--show-config"), 0,
          contains=("stages: reasoning, verifier", "deepseek-r1-distill-qwen-7b", "roles=verifier")),
    Check("example config resolves for the reasoning script (--show-config)", "live_reasoning.py",
          ("--config", "{example}", "--show-config"), 0, contains=("stages: reasoning, verifier", "roles=reasoning")),
    Check("example config resolves for the all-reviewers script (--show-config)", "live_all_reviewers.py",
          ("--config", "{example}", "--show-config"), 0,
          contains=("stages: reasoning, verifier, classifier", "roles=classifier")),
    Check("stages leave out the verifier: exit 0, no model loaded", "live_verifier.py",
          ("--config", "{configs}/no-verifier-stage.json"), 0,
          contains=("verifier is not one of them, so there is nothing for this script to run",),
          absent=("Traceback", "LM Studio is not reachable")),
    Check("verifier selected with no verifier reviewer: refused, naming the problem", "live_verifier.py",
          ("--config", "{configs}/selects-verifier-none.json"), 1,
          contains=("is invalid (1 problem)", "stage 'verifier' is selected but no active reviewer has the type 'verifier'"),
          absent=("Traceback",)),
    Check("only an inactive verifier and no stages key: exit 0, nothing to run", "live_verifier.py",
          ("--config", "{configs}/inactive-verifier.json"), 0,
          contains=("verifier is not one of them",), absent=("Traceback",)),
    Check("--model alongside a config that names reviewers is refused", "live_verifier.py",
          ("--config", "{example}", "--model", "x"), 1,
          contains=("--config already names the reviewers", "--model would be ignored"), absent=("Traceback",)),
    Check("classifier fixtures validate (--check)", "live_classifier.py", ("--check",), 0,
          contains=("16 classifier fixture(s)", "expect None")),
    Check("example config resolves for the classifier (--show-config)", "live_classifier.py",
          ("--config", "{example}", "--show-config"), 0,
          contains=("stages: reasoning, verifier, classifier", "roles=classifier", "rubric: injection-or-traversal")),
    Check("stages leave out the classifier: exit 0, no model loaded", "live_classifier.py",
          ("--config", "{configs}/no-classifier-stage.json"), 0,
          contains=("classifier is not one of them, so there is nothing for this script to run",),
          absent=("Traceback", "LM Studio is not reachable")),
    Check("classifier selected with no classifier reviewer: refused, naming the problem", "live_classifier.py",
          ("--config", "{configs}/selects-classifier-none.json"), 1,
          contains=("is invalid (1 problem)", "stage 'classifier' is selected but no active reviewer has the type 'classifier'"),
          absent=("Traceback",)),
    Check("only an inactive classifier and no stages key: exit 0, nothing to run", "live_classifier.py",
          ("--config", "{configs}/inactive-classifier.json"), 0,
          contains=("classifier is not one of them",), absent=("Traceback",)),
    Check("--model alongside a config that names reviewers is refused (classifier)", "live_classifier.py",
          ("--config", "{example}", "--model", "x"), 1,
          contains=("--config already names the reviewers", "--model would be ignored"), absent=("Traceback",)),
)


def live_config_checks(configs_dir: Path = LIVE_TESTS / "configs") -> tuple:
    """One `--show-config` check per config the live tests will run: it must resolve before a model is loaded for it.

    Each runs with the script its config will use. A config that does not load, or that no script runs, is still
    checked with `live_all_reviewers.py` (all three read the same config format): a broken one fails here with the
    loader's own message, which is the point of having a pre-flight."""
    checks = []
    for path in lc.discover(configs_dir):
        live = lc.describe(path)
        script = live.kind.script if live.kind else "live_all_reviewers.py"
        shown = ("stages: " + ", ".join(live.stages),) if live.stages else ()
        checks.append(Check(f"live-tests config {path.stem} resolves (--show-config, {script})", script,
                            ("--config", f"{{live_configs}}/{path.name}", "--show-config"), 0,
                            contains=shown, absent=("Traceback", "is invalid")))
    return tuple(checks)


CHECKS = (*STATIC_CHECKS, *live_config_checks())


def run(check: Check) -> tuple:
    """`(passed, problems, output)` for one check."""
    env = {**os.environ, "CLAUDE_RUNWAY_LMSTUDIO_URL": NO_LMSTUDIO_URL}
    paths = {"{configs}": HERE / "configs", "{live_configs}": LIVE_TESTS / "configs", "{example}": HARNESS / "config.example.json"}
    args = list(check.args)
    for placeholder, path in paths.items():
        args = [a.replace(placeholder, str(path)) for a in args]
    try:
        done = subprocess.run([sys.executable, str(HARNESS / check.script), *args], capture_output=True, text=True,
                              timeout=TIMEOUT_SECONDS, env=env, cwd=REPO_ROOT, check=False)
    except subprocess.TimeoutExpired:
        return False, [f"did not finish in {TIMEOUT_SECONDS}s (is it waiting on LM Studio?)"], ""
    output = done.stdout + done.stderr
    problems = []
    if done.returncode != check.exit_code:
        problems.append(f"exit code {done.returncode}, expected {check.exit_code}")
    problems += [f"output lacks {text!r}" for text in check.contains if text not in output]
    problems += [f"output has {text!r}" for text in check.absent if text in output]
    return not problems, problems, output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="list the checks and exit")
    parser.add_argument("--only", default=None, metavar="TEXT", help="run only the checks whose name contains TEXT")
    parser.add_argument("--show-output", action="store_true", help="print the output of every check, not only failing ones")
    args = parser.parse_args(argv)
    chosen = [c for c in CHECKS if not args.only or args.only.lower() in c.name.lower()]
    if args.list:
        for c in chosen:
            print(f"{c.name}\n    harness/{c.script} {' '.join(c.args)}  -> exit {c.exit_code}")
        return 0
    if not chosen:
        print(f"no check name contains {args.only!r}")
        return 1
    failed = 0
    for check in chosen:
        passed, problems, output = run(check)
        print(f"{'PASS' if passed else 'FAIL'}  {check.name}")
        for problem in problems:
            print(f"        {problem}")
        if args.show_output or not passed:
            print("".join(f"        | {line}\n" for line in output.splitlines()))
        failed += 0 if passed else 1
    print(f"\n{len(chosen) - failed}/{len(chosen)} check(s) passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
