#!/usr/bin/env python3
"""Live tests: the harness scripts run for real against LM Studio models, with explicit pass/fail checks.

    .venv/bin/python harness/live-tests/run_live_tests.py                       # every config in configs/
    .venv/bin/python harness/live-tests/run_live_tests.py --list                # the configs, and what each would run
    .venv/bin/python harness/live-tests/run_live_tests.py verifier-mlx-deepseek-7b  # one config, by its file name without .json
    .venv/bin/python harness/live-tests/run_live_tests.py --only verifier --runs 1   # the configs whose name contains "verifier", one run (quick)

A config in `configs/` IS a live test; adding a test is adding a file there (see `live_configs.py` and the README):

- the script comes from the config's `stages`: verifier -> live_verifier.py, classifier -> live_classifier.py,
  reasoning -> live_reasoning.py, reasoning with the verifier and/or the classifier -> live_all_reviewers.py;
- the quality bars, when there are any, come from `<config>.baselines.json` next to it. A config with none runs
  the mechanics checks only, which hold for any model.

Needs LM Studio running with its server enabled and the models named in the configs downloaded. The
scripts load and unload the models themselves. A test takes minutes (a 7B model, several
requests), so this is run by hand, not by CI.

It starts with the pre-flight in `preflight/` (seconds, no model): if a script or one of these
configs does not resolve, nothing is loaded. `--skip-checks` leaves that out.

Exit status: 0 every check passed, 1 a check failed (a pre-flight failure included, and naming a config that
cannot be run), 3 nothing was run because LM Studio is not reachable or a model the configs name is not listed
(a skipped run is not a pass).

Each test runs one harness script with one config from `configs/` and then applies two kinds of check:

- MECHANICS must always hold, whatever the model: the script exits 0, completes every run, reports no
  reviewer problem (a translator failure, a reply the reader could not parse, any other), verifier problem or
  classifier problem, grounds every fixture, and leaves LM Studio with the models it started with. The ones every kind of test
  shares are built in one place from the config's stages (`common_mechanics`).
- BASELINES are the quality measured when the sidecar was written (it says when, and with which model). They fail
  when quality gets WORSE than that, so they catch a regression in the prompt, the parser or the grounding
  step. They are numbers to raise as the verifier improves, not a claim that they are good enough.

The full output of every run is kept in `logs/` (not committed).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import live_configs as lc  # noqa: E402

HARNESS_DIR = str(HERE.parent)
if HARNESS_DIR not in sys.path:
    sys.path.insert(0, HARNESS_DIR)
import harness_lib as h  # noqa: E402
import live_branch_lib as lb  # noqa: E402

HARNESS = HERE.parent
REPO_ROOT = HARNESS.parent
CONFIGS = HERE / "configs"
LOGS = HERE / "logs"
EXIT_NOT_RUN = 3
EXIT_PREFLIGHT_FAILED = 1
DEFAULT_URL = "http://localhost:1234/v1"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    kind: str = "mechanics"  # or "baseline"


# ---------------------------------------------------------------------------
# Reading what a script printed
# ---------------------------------------------------------------------------

REAL_LINE = re.compile(r"real findings confirmed: (\d+)/(\d+)\s+wrongly rejected: \[(.*)\]")
FALSE_LINE = re.compile(r"false findings rejected: (\d+)/(\d+)\s+wrongly confirmed: \[(.*)\]")
RUN_LINE = re.compile(r"^--- run (\d+)$", re.MULTILINE)
ON_EXPECTED_LINE = re.compile(r"^on the expected entry: (\d+)/(\d+)$", re.MULTILINE)
REPEAT_LINE = re.compile(r"^score repeat over (\d+) runs: (\d+) differ of (\d+) averages compared \(tolerance ([\d.]+)\)", re.MULTILINE)
CLASSIFIED_LINE = re.compile(
    r"--- run (\d+) classified: (\d+) classified \([^)]*\), (\d+) unclassified, (\d+) without a rubric level; (\d+) overlap"
)
VERIFIED_LINE = re.compile(r"--- run (\d+) verified: (\d+) confirmed, (\d+) disputed, (\d+) unchecked; dropped (\d+) rejected, (\d+) ungrounded")
SEEDED_LINE = re.compile(r"([\d.]+)/(\d+) bug\(s\) found; ([\d.]+)/(\d+) decoy alarm")
# The cases a script says it ran. The verifier prints one `cases:` line (live_verifier.py); the reasoning and pipeline
# scripts print one `case NAME: N bug(s), ...` block per case (`harness_lib.format_score`), once or twice over.
CASES_LINE = re.compile(r"^cases: (.*)$", re.MULTILINE)
CASE_BLOCK = re.compile(r"^case (\S+): \d+ bug\(s\), \d+ decoy\(s\), \d+ run\(s\)$", re.MULTILINE)
LOADED_BEFORE = re.compile(r"^loaded before: (.*)$", re.MULTILINE)
LOADED_AFTER = re.compile(r"^loaded after:\s+(.*)$", re.MULTILINE)
BEFORE_MARK = "===== before verification"
AFTER_MARK = "===== after verification"
# A live run (`--run-type live`, live_branch_lib.py) prints one `live tag NAME: ...` block per tag, then one line per evidence kind
# and one for the clean control. The bars are read from those lines (after `lb.LIVE_AFTER_MARK` when a verifier ran).
LIVE_TAG_BLOCK = re.compile(r"^live tag (\S+): \d+ bug\(s\), \d+ decoy\(s\)", re.MULTILINE)
LIVE_KIND_LINE = re.compile(r"^live (diff|same-file|other-file): ([\d.]+) of (\d+) bug\(s\) found; ([\d.]+) of (\d+) decoy alarm", re.MULTILINE)
LIVE_CLEAN_LINE = re.compile(r"^live clean: ([\d.]+) false positive", re.MULTILINE)


# The names of the mechanics checks every kind of test shares (`common_mechanics`). Constants, so a test names the check
# it expects instead of copying its wording.
CHECK_EXITS = "the script exits 0"
CHECK_TRANSLATOR = "no translator failure"
CHECK_READABLE = "every reply was readable (the reader could parse it as lines or JSON)"
CHECK_REVIEWER_ERROR = "no other reviewer error (a failed request, a skipped finding)"
CHECK_VERIFIER_PROBLEM = "no verifier problem (an unusable reply, a failed request)"
CHECK_CLASSIFIER_PROBLEM = "no classifier problem (an unusable reply, a failed request)"
CHECK_MODELS = "LM Studio is left with the models it started with"
CHECK_CASES = "the run covered the cases the baselines were measured on"
CHECK_TAGS = "the run covered the tags the baselines were measured on"

# What the library writes for a reviewer's problem (`reasoning_reviewer_lib._review_one` and `parse_findings`), which the
# harness scripts print after `reviewer error:`. The tests build their sample output from the real library, so a reworded
# message fails there instead of this reading going blind.
TRANSLATOR_FAILURE = ": the translator "
UNREADABLE_REPLY = ("the reply had no FINDING blocks", "the reply was not JSON", "the reply has no 'findings' list")
DETAIL_CHARS = 600


def models_restored(output: str) -> Check:
    before, after = LOADED_BEFORE.search(output), LOADED_AFTER.search(output)
    if not before or not after:
        return Check(CHECK_MODELS, False, "the script printed no loaded-models lines")
    return Check(CHECK_MODELS, before.group(1) == after.group(1), f"before {before.group(1)}, after {after.group(1)}")


def reviewer_errors(output: str) -> tuple:
    """The `reviewer error:` lines of a script's output, as `(translator failures, unreadable replies, the rest)`.

    A line is in exactly one list: a translator failure is named first, because the translator's own message can echo
    text from the model's reply."""
    translator: list = []
    unreadable: list = []
    rest: list = []
    for raw in output.splitlines():
        if "reviewer error:" not in raw:
            continue
        line = raw.strip()
        if TRANSLATOR_FAILURE in line:
            translator.append(line)
        elif any(phrase in line for phrase in UNREADABLE_REPLY):
            unreadable.append(line)
        else:
            rest.append(line)
    return translator, unreadable, rest


def common_mechanics(output: str, code: int, stages) -> list:
    """The mechanics checks every kind of test shares, chosen by the stages the config runs, so that no kind of test has
    to remember to copy one: the exit code; for a `reasoning` stage, the reviewer problems (a translator failure, a reply
    that could not be read, any other); for a `verifier` stage, verifier problems; for a `classifier` stage, classifier
    problems; and the models LM Studio is left with.
    Each kind's checker calls this and adds only what is specific to it."""
    checks = [Check(CHECK_EXITS, code == 0, f"exit code {code}")]
    if "reasoning" in stages:
        translator, unreadable, rest = reviewer_errors(output)
        for name, lines in ((CHECK_TRANSLATOR, translator), (CHECK_READABLE, unreadable), (CHECK_REVIEWER_ERROR, rest)):
            checks.append(Check(name, not lines, "\n".join(lines)[:DETAIL_CHARS]))
    if "verifier" in stages:
        checks.append(Check(CHECK_VERIFIER_PROBLEM, "verifier problem:" not in output,
                            "\n".join(line.strip() for line in output.splitlines() if "verifier problem:" in line)[:DETAIL_CHARS]))
    if "classifier" in stages:
        checks.append(Check(CHECK_CLASSIFIER_PROBLEM, "classifier problem:" not in output,
                            "\n".join(line.strip() for line in output.splitlines() if "classifier problem:" in line)[:DETAIL_CHARS]))
    checks.append(models_restored(output))
    return checks


def reported_cases(output: str, kind_name: str) -> list:
    """The case names a script's output says it ran, in order, each once."""
    if kind_name == "verifier":
        found = CASES_LINE.search(output)
        names = found.group(1).split(",") if found else []
    else:
        names = CASE_BLOCK.findall(output)
    return list(dict.fromkeys(n.strip() for n in names if n.strip()))


def cases_checks(output: str, config: lc.LiveConfig) -> list:
    """For a synthetic sidecar that names cases: the set the output reports must be the set its bars were counted on.

    Without this, a bar (an absolute count, like "4 of 6") could be applied to a different set of cases and still read as
    measured. A run that reports no cases fails too, rather than passing for want of anything to compare."""
    baselines = config.baselines
    if baselines is not None and baselines.run_type == lc.LIVE:
        # The same guard for a live sidecar, over its tags: a bar counted on some tags must not be applied to others.
        expected_tags = lc.expected_tags(baselines)
        reported_tags = list(dict.fromkeys(LIVE_TAG_BLOCK.findall(output)))
        ok = bool(reported_tags) and set(reported_tags) == set(expected_tags)
        detail = ", ".join(expected_tags) if ok else f"the baselines were counted on {expected_tags}; the run reported {reported_tags or 'no tags'}"
        return [Check(CHECK_TAGS, ok, detail, "baseline")]
    if baselines is None or baselines.run_type != lc.SYNTHETIC or not baselines.cases:
        return []
    assert config.kind is not None
    expected = lc.expected_cases(config.kind, baselines)
    reported = reported_cases(output, config.kind.name)
    ok = bool(reported) and set(reported) == set(expected)
    detail = ", ".join(expected) if ok else f"the baselines were counted on {expected}; the run reported {reported or 'no cases'}"
    return [Check(CHECK_CASES, ok, detail, "baseline")]


def on_cases(config: lc.LiveConfig) -> str:
    """` on `orders`` for a bar's name, so it says the set it is counted over; empty when the sidecar names none."""
    if config.baselines is not None and config.baselines.run_type == lc.LIVE:
        tags = lc.expected_tags(config.baselines)
        return f" on `{', '.join(tags)}`" if len(tags) <= 3 else f" on {len(tags)} live tags"
    if config.baselines is None or config.kind is None or not config.baselines.cases:
        return ""
    names = lc.expected_cases(config.kind, config.baselines)
    return f" on `{', '.join(names)}`" if len(names) <= 3 else f" on {len(names)} cases"


def seeded_after(output: str, mark: str) -> Optional[tuple]:
    """`(bugs found, bugs, decoy alarms, decoys)` from the first summary line after `mark`."""
    at = output.find(mark)
    match = SEEDED_LINE.search(output, at) if at != -1 else None
    return None if match is None else (float(match.group(1)), int(match.group(2)), float(match.group(3)), int(match.group(4)))


def live_bar_checks(output: str, config: lc.LiveConfig) -> list:
    """The bars of a live sidecar, per evidence kind, read from the lines a live run prints (`live_branch_lib.summary_lines`).

    Nothing for a synthetic sidecar or none. With a verifier the bars are on what is left standing after it, as the synthetic
    pipeline's are. A bar whose line is missing fails (it is not skipped), so a run that printed nothing cannot pass."""
    baselines = config.baselines
    if baselines is None or baselines.run_type != lc.LIVE or not baselines.expect:
        return []
    text = output
    if "verifier" in config.stages:
        at = output.find(lb.LIVE_AFTER_MARK)
        if at == -1:
            return [Check("a live report follows the 'live after verification' marker", False, "not found", "baseline")]
        text = output[at:]
    kinds = {m.group(1): (float(m.group(2)), int(m.group(3)), float(m.group(4)), int(m.group(5))) for m in LIVE_KIND_LINE.finditer(text)}
    clean = LIVE_CLEAN_LINE.search(text)
    expect, note, on = baselines.expect, baselines.note, on_cases(config)
    checks = []
    for kind in lb.EVIDENCE_KINDS:
        key = kind.replace("-", "_")
        got = kinds.get(kind)
        missing = f"no 'live {kind}:' line printed"
        if f"min_{key}_found" in expect:
            bar = expect[f"min_{key}_found"]
            checks.append(Check(f"`{kind}` bugs found >= {bar:g} of {got[1] if got else '?'}{on} ({note})",
                                got is not None and got[0] >= bar, f"{got[0]:g}" if got else missing, "baseline"))
        if f"max_{key}_decoy_alarms" in expect:
            bar = expect[f"max_{key}_decoy_alarms"]
            checks.append(Check(f"`{kind}` decoy alarms <= {bar:g} of {got[3] if got else '?'}{on}",
                                got is not None and got[2] <= bar, f"{got[2]:g}" if got else missing, "baseline"))
    if "max_clean_false_positives" in expect:
        bar = expect["max_clean_false_positives"]
        value = float(clean.group(1)) if clean else None
        checks.append(Check(f"false positives on the clean control <= {bar:g}{on}", value is not None and value <= bar,
                            f"{value:g}" if value is not None else "no 'live clean:' line printed", "baseline"))
    return checks


# ---------------------------------------------------------------------------
# One checker per kind of test (see live_configs.KINDS): (output, exit code, config, runs) -> checks
# ---------------------------------------------------------------------------


def check_verifier_fixtures(output: str, code: int, config: lc.LiveConfig, runs: int) -> list:
    checks = common_mechanics(output, code, config.stages)
    real, false = REAL_LINE.findall(output), FALSE_LINE.findall(output)
    checks.append(Check(f"every one of the {runs} run(s) completed and was scored", len(RUN_LINE.findall(output)) == runs == len(real) == len(false),
                        f"{len(RUN_LINE.findall(output))} run line(s), {len(real)} real-finding score(s), {len(false)} false-finding score(s)"))
    checks.append(Check("every fixture's quote is grounded in its file", "the fixture's quote is wrong" not in output))
    checks.extend(cases_checks(output, config))
    on = on_cases(config)
    # The bars are the sidecar's, for the model it was measured with; without one only the mechanics above apply.
    expect, note = (config.baselines.expect, config.baselines.note) if config.baselines else ({}, "")
    confirmed = [int(r[0]) for r in real]
    wrongly_confirmed = [len([x for x in f[2].split(",") if x.strip()]) for f in false]
    if confirmed and "min_real_confirmed" in expect:
        checks.append(Check(f"real findings confirmed, worst run >= {expect['min_real_confirmed']:g} of {real[0][1]}{on} ({note})",
                            min(confirmed) >= expect["min_real_confirmed"], f"per run: {confirmed}", "baseline"))
    if wrongly_confirmed and "max_false_confirmed" in expect:
        checks.append(Check(f"false findings wrongly confirmed, worst run <= {expect['max_false_confirmed']:g} of {false[0][1]}{on}",
                            max(wrongly_confirmed) <= expect["max_false_confirmed"], f"per run: {wrongly_confirmed}", "baseline"))
    return checks


def check_classifier_fixtures(output: str, code: int, config: lc.LiveConfig, runs: int) -> list:
    checks = common_mechanics(output, code, config.stages)
    scores = ON_EXPECTED_LINE.findall(output)
    checks.append(Check(f"every one of the {runs} run(s) completed and was scored", len(RUN_LINE.findall(output)) == runs == len(scores),
                        f"{len(RUN_LINE.findall(output))} run line(s), {len(scores)} score(s)"))
    unanswered = [line.strip() for line in output.splitlines() if "not answered:" in line]
    checks.append(Check("every fixture was answered: none left without a usable answer", not unanswered, "\n".join(unanswered)[:DETAIL_CHARS]))
    if runs > 1:
        # The probabilities are read at temperature 0, not sampled, so the same fixture gives the same average for the same
        # entry in every run; the script prints the ones that differ by more than its tolerance (live_classifier.REPEAT_TOLERANCE).
        repeat = REPEAT_LINE.findall(output)
        differ = [line.strip() for line in output.splitlines() if line.strip().startswith("differs:")]
        checks.append(Check("the entry averages came out the same from run to run", bool(repeat) and repeat[0][1] == "0",
                            "\n".join(differ)[:DETAIL_CHARS] if differ else ("no repeat line printed" if not repeat else "")))
    # The bar is the sidecar's, for the model it was measured with; without one only the mechanics above apply.
    expect, note = (config.baselines.expect, config.baselines.note) if config.baselines else ({}, "")
    on_expected = [int(r[0]) for r in scores]
    if on_expected and "min_on_expected" in expect:
        checks.append(Check(f"fixtures on the expected entry, worst run >= {expect['min_on_expected']:g} of {scores[0][1]} ({note})",
                            min(on_expected) >= expect["min_on_expected"], f"per run: {on_expected}", "baseline"))
    return checks


def check_pipeline(output: str, code: int, config: lc.LiveConfig, runs: int) -> list:
    """Reasoning, then the verifier and/or the classifier, whichever the config selects. A stage the config leaves out has
    no checks here, and the baselines (which are about verification) apply only when the verifier is selected."""
    stages = config.stages
    checks = common_mechanics(output, code, stages)
    checks.extend(cases_checks(output, config))
    # A live run reviews every tag, `runs` times each, and a stage prints its summary line once per review.
    rounds = runs * (len(config.baselines.tags) if config.baselines is not None and config.baselines.run_type == lc.LIVE else 1)
    if "verifier" in stages:
        checks.extend(_verifier_pipeline_checks(output, config, rounds))
    if "classifier" in stages:
        classified = CLASSIFIED_LINE.findall(output)
        after = "verification" if "verifier" in stages else "reasoning"
        checks.append(Check(f"the classifier ran after {after} in every one of the {runs} run(s)", len(classified) == rounds,
                            f"{len(classified)} classified line(s)"))
        # A summary line is printed even when no classifier answered; a finding nobody answered is kept with no rubric level,
        # so this (and the common classifier-problem check) is what shows that the classifier actually did its job.
        without = [int(c[3]) for c in classified]
        checks.append(Check("every finding that stood was sorted: none without a rubric level in any run", bool(classified) and not any(without),
                            f"without a rubric level per run: {without}"))
    checks.extend(live_bar_checks(output, config))
    return checks


def _verifier_pipeline_checks(output: str, config: lc.LiveConfig, rounds: int) -> list:
    """`rounds` is how many reviews ran: the runs, times the tags of a live sidecar."""
    checks = []
    verified = VERIFIED_LINE.findall(output)
    checks.append(Check(f"the verifier ran after reasoning in every one of the {rounds} run(s)", len(verified) == rounds, f"{len(verified)} verified line(s)"))
    # A summary line is printed even when no verifier answered, and unchecked findings count as standing in the
    # after-verification score, so these two (and the common verifier-problem check) are what show that the verifier
    # actually did its job.
    unchecked = [int(v[3]) for v in verified]
    checks.append(Check("every finding was checked: none unchecked in any run", bool(verified) and not any(unchecked), f"unchecked per run: {unchecked}"))
    answered = sum(int(v[1]) + int(v[2]) + int(v[4]) for v in verified)
    checks.append(Check("the verifier answered at least one finding", answered > 0, f"{answered} confirmed, disputed or rejected over {rounds} run(s)"))
    if config.baselines is not None and config.baselines.run_type == lc.LIVE:
        # The live bars are applied by `live_bar_checks`, from the report after the verifier.
        checks.append(Check("the before and after reports are both printed", lb.LIVE_BEFORE_MARK in output and lb.LIVE_AFTER_MARK in output))
        return checks
    checks.append(Check("the before and after scores are both printed", BEFORE_MARK in output and AFTER_MARK in output))
    checks.append(Check("no 'not built' note for the verifier", "verifier stage is not built" not in output))
    before, after = seeded_after(output, BEFORE_MARK), seeded_after(output, AFTER_MARK)
    if after is None:
        checks.append(Check("a summary line follows the 'after verification' marker", False, "not found"))
        return checks
    expect, note = (config.baselines.expect, config.baselines.note) if config.baselines else ({}, "")
    found, bugs, alarms, decoys = after
    on = on_cases(config)
    if "max_decoy_alarms_after" in expect:
        checks.append(Check(f"decoy alarms after verification <= {expect['max_decoy_alarms_after']:g} of {decoys}{on} ({note})",
                            alarms <= expect["max_decoy_alarms_after"], f"{alarms:g} (before verification: {before[2]:g})" if before else f"{alarms:g}", "baseline"))
    if "min_bugs_after" in expect:
        checks.append(Check(f"bugs found after verification >= {expect['min_bugs_after']:g} of {bugs}{on}",
                            found >= expect["min_bugs_after"], f"{found:g} (before verification: {before[0]:g})" if before else f"{found:g}", "baseline"))
    return checks


def check_reasoning(output: str, code: int, config: lc.LiveConfig, runs: int) -> list:
    """Mechanics, plus the cases or tags a sidecar's numbers were counted on, which are checked against the run. A synthetic
    sidecar's numbers are a record, not a bar: no reasoning bars have been measured yet. A live sidecar's per-kind bars ARE
    applied (`live_bar_checks`), and those checks are of kind `baseline`."""
    return [*common_mechanics(output, code, config.stages), *cases_checks(output, config), *live_bar_checks(output, config)]


CHECKERS: dict = {"verifier": check_verifier_fixtures, "classifier": check_classifier_fixtures, "reasoning": check_reasoning,
                  "pipeline": check_pipeline}


# ---------------------------------------------------------------------------
# LM Studio, and running a script
# ---------------------------------------------------------------------------


def lmstudio_models(base_url: str) -> Optional[set]:
    """The ids LM Studio lists (loaded or not), or None when it cannot be reached."""
    root = base_url.rstrip("/")
    root = root[: -len("/v1")] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(f"{root}/api/v0/models", timeout=10) as response:
            return {m["id"] for m in json.load(response)["data"]}
    except (OSError, ValueError, KeyError):
        return None


def run_script(config: lc.LiveConfig, runs: int, timeout: int, log: Path, env: dict) -> tuple:
    """Run the config's script, echoing its output as it arrives and keeping it all in `log`. Returns (exit code, output)."""
    assert config.kind is not None
    command = [sys.executable, str(HARNESS / config.kind.script), "--config", str(config.path), "--runs", str(runs), *lc.script_args(config)]
    print(f"$ {' '.join(command)}", flush=True)
    lines: list = []
    timed_out = threading.Event()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO_ROOT, env=env) as process:
        assert process.stdout is not None

        def stop() -> None:
            # A timer, not a check between lines: a script waiting on a model prints nothing, and reading
            # its stdout would block past the deadline. Killing it closes the pipe and ends the loop below.
            timed_out.set()
            process.kill()

        timer = threading.Timer(timeout, stop)
        timer.start()
        try:
            for line in process.stdout:
                lines.append(line)
                print(f"    | {line}", end="", flush=True)
        finally:
            timer.cancel()
            code = process.wait()
    if timed_out.is_set():
        lines.append(f"[killed: still running after {timeout}s]\n")
    output = "".join(lines)
    log.write_text(f"$ {' '.join(command)}\n{output}", encoding="utf-8")
    return code, output


def describe_config(config: lc.LiveConfig) -> str:
    """What `--list` prints for one config: what it would run, its models, and which bars it is held to."""
    if config.problem:
        return f"{config.name}: " + ("NOT RUNNABLE" if config.unrunnable else "INVALID") + f"\n    {config.problem}"
    assert config.kind is not None
    bars = f"{config.baselines.note}: {config.baselines.expect}" if config.baselines else "none (mechanics checks only)"
    return (f"{config.name}: harness/{config.kind.script} --config live-tests/configs/{config.path.name} {' '.join(('--runs', str(config.kind.runs), *lc.script_args(config)))}\n"
            f"    stages: {', '.join(config.stages)}   models: {', '.join(config.models) or 'none'}\n"
            f"    baselines: {bars}")


def choose(configs: list, names: list, only: Optional[str]) -> tuple:
    """`(chosen, error)`: the configs named (by file name, with or without `.json`) and containing `only`, or the problem."""
    by_name = {c.name: c for c in configs}
    chosen = list(configs)
    if names:
        wanted = [n[: -len(".json")] if n.endswith(".json") else n for n in names]
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            return [], f"no config named {unknown}. The configs are: {', '.join(by_name) or '(none)'}"
        chosen = [by_name[n] for n in dict.fromkeys(wanted)]
    if only:
        chosen = [c for c in chosen if only.lower() in c.name.lower()]
        if not chosen:
            return [], f"no config name contains {only!r}. The configs are: {', '.join(by_name) or '(none)'}"
    return chosen, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", metavar="CONFIG", help="the configs to run, by file name without .json (default: every runnable config in configs/)")
    parser.add_argument("--list", action="store_true", help="list the configs and exit")
    parser.add_argument("--only", default=None, metavar="TEXT", help="run only the configs whose name contains TEXT")
    parser.add_argument("--runs", type=int, default=None, help="runs per config, instead of each kind of test's own count")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds one test may run before it is stopped (default 1800)")
    parser.add_argument("--skip-checks", action="store_true", help="do not run the preflight/ checks first")
    args = parser.parse_args(argv)
    configs = [lc.describe(path) for path in lc.discover(CONFIGS)]
    chosen, error = choose(configs, args.names, args.only)
    if error:
        print(error)
        return 1
    if args.list:
        print("\n".join(describe_config(c) for c in chosen) if chosen else f"no configs in {CONFIGS}")
        return 0
    if not chosen:
        print(f"no configs in {CONFIGS}: a config there is a live test")
        return 1
    # A config that cannot run is named, never skipped quietly. One that is BROKEN (it does not load, or its baselines
    # file is unusable) fails the run, though the others still run. One that is fine but has no script yet is a note,
    # unless it was asked for by name: then there is nothing to run for what was asked.
    unusable = [c for c in chosen if c.problem]
    for config in unusable:
        print(f"NOT RUN {config.name}: {config.problem}")
    broken = [c for c in unusable if not c.unrunnable]
    chosen = [c for c in chosen if not c.problem]
    if not chosen or (args.names and unusable):
        return 1
    if not args.skip_checks:
        # The quick checks need no model and take seconds; a broken config or script is better found here than minutes into a run.
        print("===== preflight: harness/live-tests/preflight (no LM Studio needed)", flush=True)
        checked = subprocess.run([sys.executable, str(HERE / "preflight" / "run_checks.py")], cwd=REPO_ROOT, check=False)
        if checked.returncode != 0:
            print("NOT RUN: a preflight check failed, so no model was loaded (--skip-checks runs the tests anyway).")
            return EXIT_PREFLIGHT_FAILED
        print()
    url = os.environ.get("CLAUDE_RUNWAY_LMSTUDIO_URL", DEFAULT_URL)
    listed = lmstudio_models(url)
    if listed is None:
        print(f"NOT RUN: LM Studio is not reachable at {url}. Start it and enable its server.")
        return EXIT_NOT_RUN
    missing = sorted({m for c in chosen for m in c.models} - listed)
    if missing:
        print(f"NOT RUN: LM Studio does not list {missing}. Download them, or point the configs in {CONFIGS} at models `lms ls` shows.")
        return EXIT_NOT_RUN
    LOGS.mkdir(exist_ok=True)
    # One throwaway reviewer database for every config this invocation runs, unless the caller set one: the scripts below
    # record their runs, and a test run must not write into the real database under ~/.claude. It is set in os.environ,
    # before `env` is built, so each child inherits it and finds the variable already set. Placed after the checks that can
    # end the run without a model (preflight, LM Studio not reachable) so those never announce a database they do not use.
    h.isolate_reviewer_db()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    results: list = []
    for config in chosen:
        assert config.kind is not None
        runs = args.runs or config.kind.runs
        print(f"\n===== {config.name}  ({runs} run(s), models: {', '.join(config.models)})", flush=True)
        log = LOGS / f"{time.strftime('%Y%m%d-%H%M%S')}-{config.name}.log"
        code, output = run_script(config, runs, args.timeout, log, env)
        results.append((config, CHECKERS[config.kind.name](output, code, config, runs), log))
    failed = 0
    print("\n===== results")
    for config, checks, log in results:
        print(f"{config.name}   (full output: {log.relative_to(REPO_ROOT)})")
        for check in checks:
            print(f"  {'PASS' if check.ok else 'FAIL'}  [{check.kind}] {check.name}" + (f"\n        {check.detail}" if check.detail and (not check.ok or check.kind == 'baseline') else ""))
            failed += 0 if check.ok else 1
    total = sum(len(c) for _, c, _ in results)
    print(f"\n{'FAILED' if failed or broken else 'PASSED'}: {total - failed} of {total} check(s) passed"
          + (f"; {len(broken)} config(s) could not be run" if broken else ""))
    return 1 if failed or broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
