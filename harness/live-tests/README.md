# Live tests: the harness against real LM Studio models

These run the harness scripts for real, so a change to the prompts, the parser or the grounding step is
judged by what an actual model does with it, not by a stub. They need LM Studio running with its server
enabled and the models in `configs/` downloaded. Each test takes minutes, so run them by hand.

**A config in `configs/` is a test.** Look in that folder, pick a file, run it. Nothing about a test is written in Python.

```
.venv/bin/python harness/live-tests/run_live_tests.py --list                       # each config: script, models, runs, baselines
.venv/bin/python harness/live-tests/run_live_tests.py verifier-deepseek-7b         # one config, by file name without .json
.venv/bin/python harness/live-tests/run_live_tests.py                              # every runnable config
.venv/bin/python harness/live-tests/run_live_tests.py --only verifier --runs 1     # configs whose name contains "verifier", one run (quick)
```

In Claude Code, `/cr-gh-run-harness` lists the configs, asks which to run and how many runs, and runs it.

The runner starts with the checks in `preflight/` (a few seconds, no model). If a script or one of the configs here
does not resolve, it stops before loading anything. `--skip-checks` leaves them out. They can be run alone with
`.venv/bin/python harness/live-tests/preflight/run_checks.py` (`--list` shows what each runs and expects, `--only TEXT`
picks some). See [Preflight](#preflight).

Exit status: `0` every check passed, `1` a check failed (a preflight failure included, and a config that is broken or
was named but cannot be run), `3` nothing was run (LM Studio not reachable, or a model the configs name is not listed).
A skipped run is not a pass. The full output of every run is kept in `logs/` (not committed).

The runs the scripts make are recorded in a **throwaway** reviewer database under the system temp folder, not in `~/.claude`: the runner picks one for the whole invocation, shared by every config it runs, and prints `reviewer database: <path>` (the reviewer's own logs are beside it). Set `CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB` first to record into a database you keep. See [Where a run is recorded](../README.md#where-a-run-is-recorded).

## How a config becomes a test

The script comes from the config's `stages`, so the config format needs no extra key (all three scripts read it):

| `stages` | Script | Default runs |
| :--- | :--- | :--- |
| `["verifier"]` | `live_verifier.py` | 3 |
| `["classifier"]` | `live_classifier.py` | 2 |
| `["reasoning"]` | `live_reasoning.py` | 2 |
| `["reasoning", "verifier"]`, `["reasoning", "classifier"]` or all three | `live_all_reviewers.py --case orders` (the case comes from a sidecar's `metadata.cases` when there is one) | 2 |

A pipeline is reasoning followed by the verifier, the classifier or both: the classifier rates what the verifiers left
standing, or every finding when there is no verifier. Any other set of stages (the verifier and the classifier with no
reasoning) is listed as `NOT RUNNABLE`: the stage scripts each run one stage on fixtures, and `live_all_reviewers.py` starts
from reasoning. A config that does not load is listed as `INVALID`, with the loader's message, and fails the run.

The configs that ship:

| Config | Stage | Models | Baselines | Runs on |
| :--- | :--- | :--- | :--- | :--- |
| `verifier-deepseek-7b.json` | verification | `deepseek-r1-distill-qwen-7b@4bit` | yes | format not recorded |
| `verifier-deepseek-7b-8bit.json` | verification | `deepseek-r1-distill-qwen-7b@8bit` | mechanics only | format not recorded |
| `verifier-deepseek-7b-need-loop.json` | verification | `deepseek-r1-distill-qwen-7b@4bit` loaded with a 32768-token context, with `"tools": ["read_file", "grep"]` and `"promptFile": "../prompts/deepseek-loop-prompt.md"`: it may ask for more code in a JSON `need` request, and the whole verifier prompt (how to verify, both reply shapes, the lookup menu) comes from that file instead of the server's built-in text; compare it with `verifier-deepseek-7b.json`, which has the built-in prompt and no tools | mechanics only | format not recorded |
| `verifier-deepseek-llama-8b.json` | verification | `deepseek-r1-distill-llama-8b` | mechanics only | format not recorded |
| `classifier-qwen-7b-4bit.json` | classification | `qwen2.5-coder-7b-instruct-mlx` (4-bit MLX), with the shipped rubric of `harness/cases.py` (scores are probabilities), one yes/no question per rubric entry | mechanics only | Apple Silicon only (MLX) |
| `reasoning-qwen-7b-4bit.json` | reasoning | `qwen2.5-coder-7b-instruct-mlx` (4-bit MLX) | measured reply route, no pass/fail bar | Apple Silicon only (MLX) |
| `reasoning-qwen-7b-4bit-promptfile.json` | reasoning | `qwen2.5-coder-7b-instruct-mlx` (4-bit MLX, the model of `reasoning-qwen-7b-4bit`), with `prompts/reasoning-lines.md` as the whole prompt | mechanics only | Apple Silicon only (MLX) |
| `reasoning-qwen-7b-4bit-promptfile-example.json` | reasoning | the same model, with `prompts/reasoning-lines-example.md` as the whole prompt | mechanics only | Apple Silicon only (MLX) |
| `reasoning-qwen-7b-8bit.json` | reasoning | `qwen2.5-coder-7b-instruct` (8-bit MLX) | measured reply route, no pass/fail bar | Apple Silicon only (MLX) |
| `reasoning-seed-coder-8b.json` | reasoning | `seed-coder-8b-instruct` (Q4_K_S GGUF) | measured reply route, no pass/fail bar | Any platform (GGUF) |
| `reasoning-llama-3.1-8b-4bit.json` | reasoning | `meta-llama-3.1-8b-instruct` (4-bit MLX, `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`) | measured reply route, no pass/fail bar | Apple Silicon only (MLX) |
| `reasoning-llama-3.2-3b-4bit.json` | reasoning | `llama-3.2-3b-instruct@4bit` (4-bit MLX) | measured reply route, no pass/fail bar | Apple Silicon only (MLX) |
| `reasoning-llama-3.2-3b-8bit.json` | reasoning | `llama-3.2-3b-instruct@8bit` (8-bit MLX) | measured reply route, no pass/fail bar | Apple Silicon only (MLX) |
| `reasoning-llama-3.2-3b-gguf.json` | reasoning | `bartowski/llama-3.2-3b-instruct` (Q4_K_S GGUF) | measured reply route, no pass/fail bar | Any platform (GGUF) |
| `pipeline-qwen-7b-deepseek-7b.json` | both: reasoning, then verification | `qwen2.5-coder-7b-instruct-mlx` (reasoning), `deepseek-r1-distill-qwen-7b@4bit` (verifier) | yes | Apple Silicon only (the reasoning model is MLX) |
| `pipeline-qwen-7b-deepseek-llama-8b.json` | both | `qwen2.5-coder-7b-instruct-mlx` (reasoning), `deepseek-r1-distill-llama-8b` (verifier) | mechanics only | Apple Silicon only (the reasoning model is MLX) |
| `pipeline-qwen-7b-deepseek-7b-8bit.json` | both | `qwen2.5-coder-7b-instruct-mlx` (reasoning), `deepseek-r1-distill-qwen-7b@8bit` (verifier) | mechanics only | Apple Silicon only (the reasoning model is MLX) |
| `pipeline-qwen-7b-8bit-deepseek-7b.json` | both | `qwen2.5-coder-7b-instruct` (reasoning), `deepseek-r1-distill-qwen-7b@4bit` (verifier) | mechanics only | Apple Silicon only (the reasoning model is MLX) |
| `pipeline-qwen-7b-8bit-deepseek-llama-8b.json` | both | `qwen2.5-coder-7b-instruct` (reasoning), `deepseek-r1-distill-llama-8b` (verifier) | mechanics only | Apple Silicon only (the reasoning model is MLX) |
| `pipeline-qwen-7b-deepseek-7b-classifier-qwen-7b.json` | all three: reasoning, verification, then classification | `qwen2.5-coder-7b-instruct-mlx` (reasoning and classifier, the classifier asked 3 times per finding), `deepseek-r1-distill-qwen-7b@4bit` (verifier) | mechanics only | Apple Silicon only (the reasoning model is MLX) |

**Runs on.** MLX builds run on Apple Silicon only; GGUF builds run on any platform LM Studio runs on, Apple Silicon included. On a machine without Apple Silicon only the two GGUF reasoning configs can be reproduced as measured. A `pipeline-` config needs Apple Silicon because its reasoning model is an MLX build; its verifier's format is not recorded. `verifier-` configs name an id (`deepseek-r1-distill-qwen-7b@4bit`, `deepseek-r1-distill-llama-8b`) and the docs record no format for it, so on another machine check what the id loads (`lms ls --json`) before trusting a baseline. Even a config marked MLX can name an id that another machine resolves to a different build (for example `meta-llama-3.1-8b-instruct`, which a Windows machine can serve as a Q4_K_M GGUF build); a baseline was counted on the build in `prompts/README.md`, not on the id.

LM Studio's id for a build can change when a second build of the same model is installed, so name the quant in a config's file name and say which build a model id is; `prompts/README.md` maps each config to its id, source and quant. The reasoning configs set `replyFormat` to `lines`. The ones that measure a model's reply route each have a `.baselines.json` that records what `live_routes.py` measured for that model (see [the harness README](../README.md#reply-routes-which-way-a-models-reply-reaches-the-server)); the two `-promptfile` configs have none, since they check that a prompt file is read by a real model, not how well it finds bugs. A synthetic `live_reasoning.py` run reads no key from `expect`, so those numbers are a record, not a bar: nothing fails when a model scores lower. (A live sidecar is the exception: its per-kind bars are applied, see below.)

Two other Llama-3.1-8B builds (an 8-bit MLX one, and a different 4-bit MLX conversion) were measured and have no config here: both failed the reviewer-error mechanics checks, one on timeouts and one on findings with no `claim`. `prompts/README.md` has their numbers.

The name's first word says the stage: `reasoning-` finds defects in a diff, `verifier-` confirms or rejects given findings, `classifier-` rates given findings against a rubric, `pipeline-` runs reasoning and then the verifier and/or the classifier. `--list` shows the stages of each too.

The configs without baselines cover the reviewer ("reasoning") and verifier models recommended in `../README.md`, combined
so they can be compared with the first two. Only one classifier model has a config: `qwen2.5-coder-7b-instruct-mlx`, one yes/no question per rubric entry (13 of 16 fixtures on the expected entry in both of two runs, every entry average the same in both). The other installed models that were run through `live_classifier.py` are in `../README.md` ("Code Classifier Models") with their results; none of them has a config here.
A config that names a model LM Studio does not list stops
the whole run with exit 3, so add a config for a model only once it is downloaded.

Two kinds of check apply to each:

- **Mechanics** must hold whatever the model does, so a config with nothing else gets only these. The ones every kind
  of test shares come from one function, `common_mechanics`, which chooses them by the config's `stages` (so a pipeline
  config gets the reasoning checks because it has a reasoning stage); each kind of test adds only what is its own:

  | Check | Applies when | Fails when | The failure shows |
  | :--- | :--- | :--- | :--- |
  | the script exits 0 | always | the exit code is not 0 | the exit code |
  | no translator failure | `reasoning` is a stage | a reviewer's `translator` script failed (the line has `: the translator `) | the `reviewer error:` lines |
  | every reply was readable | `reasoning` is a stage | a reply could not be read as the reviewer's `replyFormat` (no `FINDING` blocks and not `NONE`, not JSON, or no `findings` list) | the `reviewer error:` lines |
  | no other reviewer error | `reasoning` is a stage | any other `reviewer error:` line, such as a failed request or a finding skipped for having no `claim` | the `reviewer error:` lines |
  | no verifier problem | `verifier` is a stage | a `verifier problem:` line | the `verifier problem:` lines |
  | no classifier problem | `classifier` is a stage | a `classifier problem:` line (a failed request, a model that returns no token probabilities, a reply that does not start with Yes or No) | the `classifier problem:` lines |
  | the entry averages came out the same from run to run | a classifier test with more than one run | an entry's average on a fixture differed by more than 0.1 between runs (`live_classifier.REPEAT_TOLERANCE`), or no repeat report was printed | the `differs:` lines |
  | LM Studio is left with the models it started with | always | the loaded models differ before and after | both lists |

  Each line a check shows is the library's own text, so the check reads that wording; the tests build their sample
  output from the real library, so a reworded message fails a test instead of silently blinding a check. On top of
  these, a verifier test needs every run to complete and be scored and every fixture's quote to be grounded in its
  file. For a config that runs the verifier after reasoning, the verifier must also have answered (nothing left
  unchecked, at least one finding confirmed, disputed or rejected), because unchecked findings count as standing in
  the score. A classifier test needs every run to complete and be scored and every fixture to have been answered. For a config that runs the classifier after reasoning, the classifier must also have run in every run and left no finding without a rubric level. A reasoning-only config with a synthetic sidecar (or none) has no bars yet and is held to the common checks alone; with a live sidecar its per-kind bars are applied as well. No classifier config has a baselines file: one measurement on one model is not a bar. The classifier answers one yes/no question per rubric entry and its answer is read from the model's token probabilities at temperature 0, so there is no reply to be unreadable and the numbers do not move from run to run: a multi-run classifier test also checks that every entry average came out the same in every run (`score repeat`, within 0.1). A model that returns no token probabilities makes every request fail, which the "no classifier problem" check reports.
- **Baselines** are the quality measured when they were written. They are specific to the model they were measured
  with, so they live next to the config in `<config name>.baselines.json`, which says when and with what:

  ```json
  {"measured": "2026-10-27", "model": "deepseek-r1-distill-qwen-7b",
   "runType": "synthetic", "metadata": {"cases": ["orders"]},
   "expect": {"min_real_confirmed": 4, "max_false_confirmed": 0}}
  ```

  They fail when quality gets worse than that, which catches a regression in a prompt or a parser. They are not a claim
  that the numbers are good enough; raise them in the file as the verifier improves. `expect` holds the keys the kind
  of test reads; a key left out is not checked.

  A bar is an absolute count ("4 of 6"), so the file also says what it was counted on, and a sidecar without it is
  refused (the message names the file), because a bar that does not say could be applied to something it was not
  measured on and still read as measured:
  - `runType` is `synthetic` (mock scenarios from `harness/cases.py`) or `live` (the `harness-live-tests` branch). There
    is no default.
  - `metadata` holds what the run type needs to choose its data. A synthetic run's `cases` names the cases the numbers were
    counted on (`["all"]` is every case the kind's script can run: every case in `cases.py` for reasoning, every case with
    fixtures for the verifier). The runner passes exactly those to the script as `--case` (`--list` shows them), states
    them beside each bar ("of 6 on `orders`"), and fails the run when the cases the output reports are not the ones the
    sidecar names, so a baseline is never applied to another case set. When a model is re-measured on a larger set,
    write the new `cases` and the new numbers in the same edit.
  - The pipeline script runs one case at a time, so a pipeline sidecar names exactly one; the classifier script takes no
    `--case`, so its sidecar names none (`"metadata": {}`). A sidecar that asks for more is refused.
  - The seven `reasoning-*` sidecars list all 14 cases of `cases.py` by name, because their numbers were counted over
    all of them (2 runs, 38 bugs); spelled out, a case added to `cases.py` later does not change what they claim.
  - **A live sidecar** (`"runType": "live"`) is the answer key for a run of the `harness-live-tests` branch, so its `metadata`
    names the `tags` the numbers were counted on and, for each, the bugs and decoys a run is scored against
    (`harness/live_branch_lib.py`; the scoring rules and the table of the real branch are in [../README.md](../README.md)):

    ```json
    {"measured": "2026-11-08", "model": "some-model", "runType": "live",
     "metadata": {"tags": [
       {"tagId": "change-1",
        "bugs":   [{"id": "B1", "file": "libs/qdrant_batch_store.py", "line": 99, "class": "wrong-argument-order",
                    "claim": "the first two arguments are swapped", "evidence": "other-file"}],
        "decoys": [{"id": "D1", "file": "libs/qdrant_batch_store.py", "line": 91,
                    "claim": "two processes can both create the collection", "evidence": "other-file"}]},
       {"tagId": "change-5", "clean": true, "bugs": [], "decoys": []}]},
     "expect": {"min_other_file_found": 1, "max_clean_false_positives": 0}}
    ```

    `evidence` is `diff`, `same-file` or `other-file`. A tag that lists neither bugs nor decoys must say `"clean": true`
    (any finding on it is a false positive), and a `clean` tag cannot list any, so an empty tag is never an accident. The
    file is checked when it loads: a misspelt key, a duplicate id, a line below 1, or an `expect` key a live run does not
    read is refused, naming the file. A live sidecar runs with `--run-type live` on the reasoning and the pipeline scripts
    (each reviewer stage after reasoning runs, in order, once per tag); the verifier and classifier scripts have no live
    mode, so a live sidecar on a verifier-only or classifier-only config is listed as not runnable. The runner fails the run
    when the tags the output reports are not the ones the sidecar names, and, with a verifier, applies the bars to what is
    left standing after it.

| Kind (script) | `expect` key | Meaning |
| :--- | :--- | :--- |
| verifier | `min_real_confirmed` | real findings confirmed (of 6), worst run, at least |
| verifier | `max_false_confirmed` | decoy findings wrongly confirmed (of 3), worst run, at most |
| reasoning + verifier | `max_decoy_alarms_after` | decoy alarms after verification (of 3), at most |
| reasoning + verifier | `min_bugs_after` | bugs found after verification (of 6), at least |
| classifier | `min_on_expected` | fixtures on the expected entry (of 16), worst run, at least |
| reasoning or pipeline, live | `min_diff_found`, `min_same_file_found`, `min_other_file_found` | bugs of that evidence kind found, averaged per run, at least |
| reasoning or pipeline, live | `max_diff_decoy_alarms`, `max_same_file_decoy_alarms`, `max_other_file_decoy_alarms` | decoys of that evidence kind alarmed, averaged per run, at most |
| reasoning or pipeline, live | `max_clean_false_positives` | findings on a `clean` tag, averaged per run, at most |

**To add a test**, put a config in `configs/`. That is all: it shows in `--list`, the preflight checks it, and it runs.
Add a `<name>.baselines.json` once you have measured numbers worth holding it to. **To try other models**, copy a
config, change the model ids to ones `lms ls` shows, and run it; with no baselines file it is judged on mechanics alone,
which is the honest comparison until you have measured numbers for that model.

## Preflight

Exit-code and message checks for the harness scripts: what they do **before** they touch a model. Each one runs a
harness script the way a person would and compares its exit code and output with what they should be. The runner points
`CLAUDE_RUNWAY_LMSTUDIO_URL` at an address nothing listens on, so a check that wrongly reaches for LM Studio fails rather
than quietly using the real one. They say nothing about how good a model is; the live tests above do that.

| Check | Proves |
| :--- | :--- |
| `--check` | the verifier and the classifier fixtures are valid |
| `--show-config` on `config.example.json`, for `live_verifier.py`, `live_classifier.py`, `live_reasoning.py` and `live_all_reviewers.py` | the example config resolves for every script that takes `--config` |
| `preflight/configs/no-verifier-stage.json` | `stages` leaves out `verifier`: exit 0, no model loaded |
| `preflight/configs/selects-verifier-none.json` | `verifier` selected with no verifier reviewer: refused, with the problem named |
| `preflight/configs/inactive-verifier.json` | the only verifier is inactive and `stages` is omitted: exit 0, nothing to run |
| `preflight/configs/no-classifier-stage.json` | `stages` leaves out `classifier`: exit 0, no model loaded |
| `preflight/configs/selects-classifier-none.json` | `classifier` selected with no classifier reviewer: refused, with the problem named |
| `preflight/configs/inactive-classifier.json` | the only classifier is inactive and `stages` is omitted: exit 0, nothing to run |
| `--config` with `--model` | refused, because the config already names the reviewers |
| `--show-config` on each config in `configs/` | the configs the live run is about to use resolve, with the script each one runs. The folder is listed, so a new config is checked without adding a line |

**To add a check**, append a `Check` to `STATIC_CHECKS` in `preflight/run_checks.py` (a script, its arguments, the expected exit
code, and text that must or must not appear). Put any config file it needs in `preflight/configs/`; `{configs}`,
`{live_configs}` and `{example}` in the arguments stand for that folder, `harness/live-tests/configs` and
`harness/config.example.json`.

The library and server unit tests (`tests/`) are separate: `.venv/bin/python -m unittest discover -s tests`.
