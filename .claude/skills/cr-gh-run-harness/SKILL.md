---
name: cr-gh-run-harness
description: "Run one of the codebase-reviewer harness live tests against real LM Studio models: lists the configs in harness/live-tests/configs/, asks which to run and how many runs, runs it, and reports the results and the log path."
---

# Skill: cr-gh-run-harness

Run a harness live test for this repo (claude-runway). A live test is a config file in `harness/live-tests/configs/`: the config's `stages` decide which harness script runs it, its reviewers name the models, and an optional `<config>.baselines.json` next to it holds the quality bars. This skill is how to pick one and run it without remembering the command line.

A run takes minutes (a 7B model, several requests) and loads real models into LM Studio, so this skill never starts one the person has not chosen.

## Usage
```
/cr-gh-run-harness
/cr-gh-run-harness <config-name>
/cr-gh-run-harness <config-name> <runs>
```
`<config-name>` is a config's file name without `.json`. `<runs>` overrides how many runs each test does.

## Steps

1. **Resolve the venv's Python interpreter in every command.** Windows `uv venv` creates `.venv/Scripts/`, Linux/macOS creates `.venv/bin/`, so a hardcoded `.venv/bin/python` fails with command-not-found on a Windows checkout. Shell variables do not persist from one Bash call to the next, so each command below starts with the resolver line; run it in the same call as the command that uses `$PYTHON`, never in a call of its own.

2. **List the configs.** This needs no LM Studio and loads no model:
   ```bash
   PYTHON=$(if [ -f .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe; else echo .venv/bin/python; fi)
   $PYTHON harness/live-tests/run_live_tests.py --list
   ```
   Each entry shows the script the config runs, its default run count, its stages and models, and its baselines (the date and model they were measured with, or `none (mechanics checks only)`). An entry marked `INVALID` or `NOT RUNNABLE` carries the reason; tell the person which configs those are and why, and do not offer them.

3. **Decide what to run.** If the person named a config (and a run count) in the arguments, use that. Otherwise ask with AskUserQuestion which config to run, and how many runs: the config's default, or one run for a quick check. Offer "every runnable config" as a choice when there is more than one. Do not guess: the choice decides which models get loaded and how long it takes.

4. **Hand the run to the `harness-runner` agent.** The run takes minutes and the scripts print no progress of their own while a model works (only one raw HTTP log line per request), so run in this session it looks frozen and its output floods the conversation. The agent (`.claude/agents/harness-runner.md`) starts the run in the background, watches it with a monitor, says in a line where it is as it goes, and returns the result. Launch it with the Agent tool by the exact name `harness-runner`, with `run_in_background`, and put in the prompt only the config name and, when the person gave one, the run count. The agent file sets its model and effort: do not pass `model` or `effort`. Wait for its completion notice rather than polling it.

   One config per agent. For "every runnable config", start the next agent only after the previous one has reported: two runs at once would fight over LM Studio's models and make both untrustworthy. Do not run the test yourself. Do not add `--skip-checks` anywhere: the pre-flight is the guard that keeps a model from loading for a config that does not resolve.

5. **Report the result.** Relay the agent's report, in the person's terms:
   - the exit code: `0` every check passed, `1` a check failed, `3` nothing ran because LM Studio was not reachable or a model the config names is not downloaded. A `3` is not a pass: say that nothing was tested.
   - the `FAIL` lines with their detail, and the `[baseline]` lines with their per-run numbers (a baseline check is the quality bar, a `[mechanics]` one is whether the run worked at all). Say which kind failed, because a mechanics failure means the run is not trustworthy and a baseline failure means quality got worse than what was measured.
   - the log path (`harness/live-tests/logs/<time>-<config>.log`) for each config, which holds the script's full output.
   If the agent replied `BLOCKED: ...`, say what it needs and do not run anything yourself.

6. **Stop there.** Do not edit configs, baselines or code to make a failing run pass. Baselines are numbers measured with a specific model; raising or lowering one is a decision for the person, made after reading the log. If a run failed, say what the log shows and offer to look into it.
