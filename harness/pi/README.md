# Testing the codebase-reviewer through Pi, in a container

`pi_driver.py` runs [Pi](https://pi.dev) (a coding agent) in a throwaway Docker container, gives it the reviewer's MCP server, and asks it to call the `review` tool on a change of the `harness-live-tests` branch: real code with a seeded bug and a decoy (see [the harness README](../README.md#a-real-repository-with-seeded-bugs-the-harness-live-tests-branch)). It tests the path a real caller takes, finding the tool over MCP and calling it, which the `live_*.py` scripts never do: they call the reviewer's libraries directly. `score_run.py` scores a reasoning run's output against the branch's answer key.

One run is one **session**: one container, one review of one type (`reasoning`, `verifier` or `classifier`). The driver does only that. Reading a config for a whole run, choosing reviewers, loading models, combining results and scoring them belong to whoever calls it (the planned orchestrator).

A development tool, not product code. CI does not run it (it needs Docker, LM Studio and real models); the driver's logic is covered by `tests/test_harness_pi_driver.py` with a fake `docker`, the scorer's by `tests/test_harness_pi_score_run.py`, and the shared-database check's by `tests/test_harness_shared_db_check.py`.

## Run it

```
git fetch origin harness-live-tests 'refs/tags/harness-live-tests/*:refs/tags/harness-live-tests/*'   # once: the branch's tags (a normal clone lacks them)
docker build -t claude-runway-pi:dev harness/pi          # once; or add --build below
.venv/bin/python harness/pi/pi_driver.py \
    --model <caller model> \
    --config harness/live-tests/configs/reasoning-gguf-seed-coder-8b.json
```

- `--model` is the model **Pi** calls the tool with (the caller). It must be one LM Studio lists (`lms ls`) and it must make tool calls reliably. Load it first, with the context you pass to `--context-length` (default 16384); see "Before and after" below.
- `--config` is a `harness/live-tests/configs/*.json` file; its reviewers become the reviewer's `REVIEWERS_FILE`, so the reviewer models and contexts come from where the harness already records them. It needs an active reviewer of the session's `--type`; its other reviewers are passed on as they are (a session meant to run one reviewer is given a config with one). Without `--config`, one reasoning reviewer is built from `--model` (`--reviewer-model` changes it). A `promptFile` is copied into the container (see below), and `replyFormat` (`lines` or `json`) stays each reasoning reviewer's own choice. A config that uses `translator` or a rubric `definitionFile` is refused: the driver does not copy those.
- `--type` is the review the session runs (default `reasoning`), and the driver writes Pi's prompt for it. `verifier` and `classifier` run **only** that stage, on `--findings-file PATH`, and need a `--config`. The file's format is described in `libs/reviewer_files_lib.py`; a classifier's findings must each carry a `verification` status. The driver checks the file with the review tool's own reader and gives the container a read-only **copy**, so the file may sit anywhere on the host.
- `--review-id` is stored in the run's record, to group the sessions of one review. It goes into the review tool's own settings, so the Pi model never has to copy it.
- `--change` is the change of the `harness-live-tests` branch the reviewer is asked to review, named as in the answer key (default `change-6`: two lines added and two removed in one file, one bug and one decoy, both judgeable from the diff). `--repo PATH` is a checkout that has the branch's tags (default: this one). The driver fetches just the base and that change from it into an empty scratch repository and checks the change out, so the repository holds those two commits and nothing else, and it is that repository the container is given, not the checkout. A checkout without the tags stops at once, before a model is asked or an image built, with the `git fetch` above.
- The reviewer model is loaded by the reviewer server when its reviewer has `loadOnDemand: true` (the config says), and unloaded again afterwards. With `loadOnDemand: false` the server only checks that the model is loaded, and loads and unloads nothing. The driver loads nothing.
- Other options: `--prompt` (what Pi is asked; `{repo}`, `{base}` and `{head}` are filled in with the repository's path in the container and the two tags it holds, and `{findings}` with the findings file's path, which is empty in a reasoning session; the call check below applies whatever the prompt says), `--log FILE` (every event Pi emitted, written to a new file: an existing file, or any path under `~/.claude` or `~/.pi`, is refused), `--dry-run` (print the `docker run` command and stop), `--build` and `--image` (build or name the image), and the container's limits `--memory` (default `4g`), `--cpus` (`2`) and `--pids-limit` (`256`).
- The caps are `--max-turns` (default 10), `--max-tool-calls` (30) and `--timeout` (600 seconds for the whole run).
- `--mcp-timeout` (Pi's limit for one `review` call) and `--lmstudio-timeout` (the reviewer's limit for one LM Studio request), in seconds, both default to `--timeout`. Each is 60 seconds by itself, and Pi would then give up on a longer review and call again while it was still running.

### Before and after: loading and restoring models

The driver loads and unloads nothing in LM Studio, so a run starts and ends with whatever you left loaded. To keep your own setup:

```
lms ps                                              # note what is loaded now (and its context length)
lms load <caller model> --context-length 16384      # the same context you pass to --context-length
.venv/bin/python harness/pi/pi_driver.py --model <caller model> --config ...
lms unload <caller model>                           # afterwards
lms load <each model that lms ps showed>            # restore what was loaded before, with its context
```

The reviewer's model is not part of this: a reviewer with `loadOnDemand: true` is loaded by the reviewer server when it runs and unloaded again afterwards, except when the run is stopped during the review (Ctrl+C, or a cap or timeout that kills the container): the server dies before it can unload, so check `lms ps` and unload the reviewer model yourself. If the caller and the reviewer are the same model, load it once. Both models need room at the same time, so check `lms ps` against your GPU memory first (the sizes are in the tier table below).

Output: one progress line per turn and tool call on stderr, then a JSON summary on stdout. It names the **`change`** that was reviewed, so `score_run.py` needs no flag to find its answers, and the session's `review_type` and `review_id`. **`results`** holds the review tool's structured results (complete findings, each verifier's verdicts, each classifier's probabilities, how many replies were readable, errors), as `libs/reviewer_files_lib.py` writes them. **`review_text`** is the reviewer tool's own text, from the last `review` call that ran (when none ran, what Pi said about the last one it refused); `final_text` is the caller's summary of it, which varies from run to run and can leave most of it out. `call_errors` and `failures` say why a run that finished did not pass.

| Exit code | Meaning |
| :-- | :-- |
| 0 | Pi finished, its `review` call was this session's, and the results came back |
| 1 | it finished without that: no `review` call ran, every call errored (an error result, or the tool's own `Error: ...` refusal, which it returns as plain text), a call that ran was not this session's (`call_errors`), or no results file of the session's type arrived (`failures`). Also: Pi aborted (`stopped_by: aborted`), or the container failed, including a nonzero exit after Pi had settled, or Pi's output ended without it settling, whatever the exit code (`stopped_by: error`) |
| 2 | a bad argument (an unknown `--change` among them), a findings file or `promptFile` the container must not be given, a model the preflight could not find, a checkout without the branch's tags, or `docker` could not be started (nothing was run) |
| 3 | a cap, the time limit or a stop request stopped it (`stopped_by` says `turns`, `tool_calls`, `timeout` or `stopped`) |

### When a session passes

The Pi model chooses what it sends, so the driver checks every `review` call that **ran**: its `type`, `repo_path`, `base_ref`, `head_ref` and `findings_file` must be the session's (`head_ref` may be left out or be `HEAD`, since the change is checked out). One wrong call fails the session, even if a right one follows: swapped refs, say, would review another diff and still look like a review. A call Pi **refused before running it** (arguments that fail its check, say) ran nothing and is not judged, so a refused call followed by a good retry passes. Pi sends `tool_execution_start` before it checks the arguments, so the start record cannot tell the two apart; only the `tool_execution_end` of a call that ran carries `durationMs` (Pi 1.6.0). A session also needs at least one right call that succeeded and a results file of its own type, and no results file of another. So if a later Pi stopped sending `durationMs`, sessions would fail rather than pass unchecked.

### From Python

`run_session(SessionSpec(...), stop=None, on_event=...)` is the same session, returning the same fields. `SessionSpec` takes the reviewers file's content as a dict (`reviewers_from_config` builds one from a config, with each `promptFile` made an absolute path) and LM Studio's address as the container sees it (`container_url`). `stop` is a `threading.Event`: setting it ends the session within about half a second, removes its container, and returns `stopped_by: stopped`. It raises `SessionError` when the session cannot be set up, with nothing run. The model preflight is the command line's; a Python caller checks its own models.

## The harness database and results

Every session records its run in one database, `harness/live-tests/logs/db/codebase-reviewer.db` (ignored by Git, one per machine, kept between runs, separate from the real one under `~/.claude`), with the reviewer's text logs beside it. The `db/` **folder** is mounted writable, never the single file and never `logs/` above it. The driver creates the database when it is missing, before a container starts, and otherwise never opens it, and it does not read the record back. These rules come from a test of sharing one database between containers: containers that created it together lost records, and writes from the host while containers wrote lost records and damaged it. Reading it from the host while a container writes has not been tested, so read it when no session is running.

`harness/orchestrator/shared_db_check.py` is that test as a script, for each machine whose containers will share a database. It uses a throwaway database under `logs/`, never the harness one:

```
.venv/bin/python harness/orchestrator/shared_db_check.py     # .venv/Scripts/python.exe on Windows
```

6 containers record 25 synthetic runs each with the reviewer's real recording code, all at once, 5 times; a trial passes when all 150 records are there, SQLite's integrity check says `ok` and every container exited 0. A failed database write only prints a warning, so the count of records is what shows a loss.

| Machine | Result |
| :-- | :-- |
| Windows desktop, Docker Desktop | 5 of 5 trials passed, 150 of 150 records each (2026-11-27) |
| MacBook, Docker Desktop | **not yet run**: run it there before relying on a shared database |

## What keeps it contained

Pi has no turn or tool-call limit of its own, so the driver reads Pi's JSON event stream (split on LF only, as Pi's docs require) and stops the run when a cap is reached. A turn cap alone would not stop a flood of tool calls inside one turn, hence the three caps. A run that does not end cleanly (a cap, a stop request, Ctrl+C, an error, or output that ends without Pi having settled, as a `docker` client that died leaves its container running) is cleaned up the same way: the `docker` client is stopped first, so a container that is not created yet never is; then `docker kill`; then `docker rm --force`, which also removes a container that was created and never started, and a check that it is gone (`failures` says so when that cannot be confirmed). Each `docker` command gives up after 30 seconds. Stopping uses no operating system signal. As a backstop for a driver that is killed outright, which then cleans up nothing, the container runs Pi under the image's `timeout` command, set 15 seconds past `--timeout`, so it ends by itself; the models it was using stay loaded. It runs with `--init`, so signals reach the processes inside and none are left unreaped.

The container sees only: the reviewer's `libs/` and `mcpservers/` (read-only), a scratch repository holding the branch's base and the chosen change (read-only), the reviewers file with any copied prompt files (read-only), a copy of the session's findings file (read-only; a verifier or classifier session only), two scratch folders (Pi's agent directory and the review tool's results), and the harness database folder. It never sees the rest of the checkout, `~/.claude`, `~/.pi`, Qdrant or the memory bank, and **never the answer key**: the key is `harness/live-tests/answer_key.json` in the checkout, which is not mounted, older logs under `logs/` outside `db/` (which can print it) are not either, and the scratch repository is fetched from the branch's two tags only, which have no `harness/` and no history but the base and the change. Files copied in are checked first. A findings file with any key the format does not have is refused, so the answer key cannot be passed off as one. A `promptFile` must be a `.md` file outside `~/.claude` and `~/.pi`, and one inside the checkout must be under `harness/prompts/` or `harness/live-tests/prompts/`. `tests/test_harness_pi_driver.py` checks that the repository holds nothing else and that no mount reaches the key. The driver refuses to start if a mount would overlap the checkout, `~/.claude` or `~/.pi`; the exceptions are `libs/` and `mcpservers/` read-only, and `logs/db/` as the database mount. Each must be the checkout's own folder (a symlink in its place, or above `db/`, is refused, wherever it points). `~/.pi/agent` is Pi's own folder, where it keeps credentials, settings, extensions and sessions; the container gets a scratch one through `PI_CODING_AGENT_DIR` instead. Pi runs with its read-only tools plus the reviewer's, never `bash`, `edit` or `write`. Memory, CPU and process limits are set, and capabilities are dropped.

The limits that remain: the container runs as root on Windows (as your user on Linux and macOS) and has open outbound network access, so it is a sandbox for files and not an egress barrier.

## LM Studio and the models

LM Studio's address comes from `libs/runway_defaults.py` (`CLAUDE_RUNWAY_LMSTUDIO_URL`, else `http://localhost:1234/v1`). The container cannot use `localhost`, so the driver swaps that name for `host.docker.internal` and keeps the port; `--lmstudio-url` and `--host-lmstudio-url` override either side. Windows, macOS and Linux (WSL2) with Docker Desktop have been run (Docker Desktop resolves the name on all three): on plain Linux Docker Engine, without Docker Desktop, the name comes from `--add-host host.docker.internal:host-gateway`, which reaches LM Studio only if it listens beyond `127.0.0.1`, and that has not been tried.

Which reviewer model to use is not a question for this kit: `harness/README.md` records the measured reviewer builds per role (reasoning, verifier, classifier) and which build runs where. What Pi adds is the **caller**: the model that finds the `review` tool over MCP and calls it with the right arguments. This section records which models can do that, per hardware tier. Sizes are the weights only. MLX builds run on Apple Silicon only, so the Mac rows (MLX) and the Windows rows (GGUF) are different builds and are not comparable.

**Machines the comparison is run on** (recorded 2026-11-23; the caller and the reviewer are loaded at the same time, so both must fit in what the GPU can use, which on a Mac is less than the unified memory):

| Machine | CPU / GPU | Memory | OS | Builds it runs |
| :-- | :-- | :-- | :-- | :-- |
| MacBook Pro (Mac17,6) | Apple M5 Max, 18 cores (6 Super, 12 Performance), 40-core GPU | 48 GB unified | macOS 27.2.1 | MLX (`safetensors` in `lms ls`); GGUF also runs under llama.cpp |
| Windows desktop | Intel Core Ultra 9 285K, NVIDIA RTX 5090 | 191 GB RAM, 32 GB VRAM | Windows | GGUF |

### Callers, measured on the Mac (Apple M5 Max, 48 GB)

**Historical.** Measured on `seeded-three`, a made-up case from `harness/cases.py` that the driver no longer reviews: these runs used other test code and are not comparable with runs on `change-6`. 4 runs per pair, so this is a first measurement, not a ranking. Every model was loaded beforehand at 16384 context (no on-demand load), with the default prompt, caps of 10 turns, 30 tool calls and 240 seconds, and Pi's own sampling (the driver sets no caller temperature). A run is **clean** when the driver exited 0 and a review actually ran. Seconds include the reviewer's time. The reviewer is held to a model that answers well inside Pi's 60-second MCP timeout (`qwen3-coder-30b-a3b-instruct-mlx`, 6 to 7 seconds when it calls itself), so that a failure is the caller's; a few runs use another reviewer.

| Tier | Weights | Caller | Clean | Failed calls before the review | Seconds |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Small | 3 to 9 GB | `google/gemma-4-e4b` (6.9 GB) | 7 of 8 | the first call failed Pi's argument check in 6 of the 7 clean runs and the retry worked | 8 to 40 |
| Moderate | 15 to 17 GB | `mistralai/devstral-small-2-2512` (15.1 GB) | 8 of 8 | none | 8 to 16 |
| Moderate | | `qwen3-coder-30b-a3b-instruct-mlx` (17.2 GB) | 15 of 16 | none; the one failure never called `review` (it called `list_reviewers`, then hit the turn cap) | 6 to 26 |
| Moderate | | `qwen/qwen3.8-27b` (16.1 GB) | 4 of 4 | none | 14 to 23 |
| Large | 40 GB and up | none measured | | | no model of this size fits this Mac |


### Callers, measured on Windows (NVIDIA RTX 5090, 32 GB VRAM)

**Historical.** Measured on `seeded-three`, a made-up case from `harness/cases.py` that the driver no longer reviews: these runs used other test code and are not comparable with runs on `change-6`. 4 runs per caller, so this is a first measurement, not a ranking. Every model is a GGUF build, loaded beforehand at 16384 context with full GPU offload, with the default prompt, caps of 10 turns, 30 tool calls and 240 seconds, and Pi's own sampling. The reviewer is `seed-coder-8b-instruct` (`reasoning-gguf-seed-coder-8b.json`). A run is **clean** when the driver exited 0 and a review actually ran. Seconds include the reviewer's time.

| Tier | Weights | Caller | Clean | Failed calls before the review | Seconds |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Small | 2 to 9 GB | `llama-3.2-3b-instruct` (1.9 GB) | 4 of 4 | none | 10 to 17 |
| Small | | `meta-llama-3.1-8b-instruct` (4.9 GB) | 4 of 4 | none | 10 to 26 |
| Small | | `google/gemma-4-12b` (7.6 GB) | 4 of 4 | none | 10 to 14 |
| Moderate | 12 to 20 GB | `openai/gpt-oss-20b` (12.1 GB) | 4 of 4 | none | 9 to 19 |
| Moderate | | `qwen/qwen3.8-27b` (17.7 GB) | 4 of 4 | none | 10 to 13 |
| Moderate | | `meta/muse-glimmer` (18.2 GB) | 4 of 4 | none | 18 to 26 |
| Moderate | | `qwen2.5-32b-instruct` (19.9 GB) | 4 of 4 | none | 12 to 13 |
| Large | 40 GB and up | none measured | | | no model of this size fits this GPU |

### Scoring more callers

To compare calling models, run the driver on **`change-6`** (the default: two lines added and two removed in `libs/version_lib.py`; one bug, `B7`, and one decoy, `D4`, both judgeable from the diff) with the **same reviewer settings** for every caller: the same `--config`, or the same `--reviewer-model`, and the same `--context-length`. `score_run.py` turns one run's JSON summary into one row, so every caller is scored the same way. It reads the change from the summary and scores the reviewer's text against that change's entry in the answer key (`harness/live-tests/answer_key.json`, the key the harness shares), by file and nearby line only:

- a finding near a bug's line is **found**;
- a finding at a decoy's line is a **decoy alarm**, a false alarm;
- any finding on `change-5`, the clean control that changes no behaviour, is a **clean false alarm**;
- anything else is **unscored**: shown, and never counted against the model, because the code is real and may have real bugs nobody has listed.

A result that is not a review at all (Pi's `Validation failed for tool ...`, `MCP request timed out`) is marked `no review ran` instead of being scored as an empty review. A run the driver says did not pass (`call_errors` or `failures`) is still scored, but its row says `readable=NO` and `FAILED SESSION:` with the first reason:

```
.venv/bin/python harness/pi/pi_driver.py --model <caller> --config <config> > run.json
.venv/bin/python harness/pi/score_run.py run.json
```

The row reads `change=change-6 stopped_by=completed readable=yes found=1/1 decoy_alarms=0 clean_false_alarms=0 unscored=0 findings=2 review_calls=1 review_errors=0 turns=2 tool_calls=1 seconds=10.8 caller_repeated=0/2`. That is a real run on the Windows machine above: the caller `google/gemma-4-12b`, already loaded (262144 context), and the reviewer `seed-coder-8b-instruct` (`reasoning-gguf-seed-coder-8b.json`), which the reviewer server loaded on demand at 8192 and unloaded afterwards. For a caller, the columns that matter are `readable`, `review_calls` against `review_errors` (failed calls before the review) and `turns`. `found`, `decoy_alarms` and `unscored` say how the reviewer did, not the caller, and `caller_repeated` is how many of the findings the caller's final message repeats. `score_run.py` exits 2 when the summary names no change (the output of an older driver, which ran a case from `harness/cases.py`) or one the key does not list. Run each caller 4 times, with the same reviewer, and record the rows in a new table of runs on `change-6`, beside the historical ones above.

## One reviewer of each type, run for real

On 2026-11-27, on the Windows desktop above (Docker Desktop 29.9.3, Pi 1.6.0), with `google/gemma-4-12b` as the caller on `change-6`. Every model was loaded beforehand and every reviewer had `loadOnDemand: false`. This shows that a session works, not how good any model is: one run each.

| Session | Reviewer | Result |
| :-- | :-- | :-- |
| reasoning | `seed-coder-8b-instruct` | passed in 9 s; 2 findings |
| reasoning, the caller as its own reviewer | `google/gemma-4-12b`, one loaded copy | passed in 50 s (the `review` call took 39 s); 1 finding |
| verifier alone, on the first session's findings | `deepseek-r1-distill-llama-8b` | passed in 13 s; no reasoning request; one finding dropped as ungrounded (its quote had `...`), the other confirmed |
| classifier alone, on the confirmed finding | `qwen2.5-coder-14b-instruct` | passed in 6 s; 4 requests, one per rubric entry, all readable (`wrong-result` 0.63) |

In each, Pi's event log showed the `review` call's arguments and its `durationMs`; the results file came back; the database held a finished run with the same run ID and review ID and **no model load**; no container was left; and `lms ps` was the same before and after. Also checked live:

- **Stop:** set 8 s into a review, with the container running: the session returned 0.6 s later and `docker ps -a` showed no container.
- **Backstop:** with Pi waiting on a stand-in model server that never answers, the driver and its child processes were killed outright (`--timeout 20`). The container ran on until `timeout` ended it at its 35-second limit (exit code 124), and was then removed.
- **What a container can see:** a shell run with a verifier session's own `docker run` command hashed all 36,906 files visible to it. None had the content of the answer key, a baselines file or any log outside `db/`, and no path was named like one.

## What this showed that the orchestrator will need

- **Read the tool's result, not the caller's summary.** The caller's final text differed between two identical runs, and dropped the descriptions and failing inputs; the reviewer's result was the same both times.
- **A review has to finish within Pi's MCP timeout.** The default is 60 seconds per MCP request (`MCP request timed out after 60000ms`). The `timeout` setting (in seconds) on the server's `mcp.json` entry changes it, and Pi's documentation says progress notifications reset it. The reviewer sends no progress, so the whole `review` call has to fit in that time. Each model request also has its own limit, 60 seconds by default (`CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS`). The driver sets both to the session's `--timeout` (`--mcp-timeout` and `--lmstudio-timeout` override them), and `--timeout` still bounds the whole run. The caller tables above were measured with Pi's own 60 seconds.
- **A caller needs Pi's MCP naming.** Pi names the tool `mcp__<server>__<tool>`, and `--tools` must list `mcp__reviewer__*` or the tool is not offered.
- **A run needs a models file for LM Studio.** Pi starts with no models; the driver writes a `models.json` that points an OpenAI-compatible provider at LM Studio.
- **The reviewer is registered by a file, not a command.** The driver writes Pi's `mcp.json` (the file `pi mcp add` would write) into the agent scratch folder, so the LM Studio URL, which can carry a user name and password, never appears on a command line (host `ps`, `docker inspect`, `--dry-run`). Printed URLs show `***` in place of a user name and password. Pi itself cannot use such a URL as its own provider address: with `http://user:pass@...` the caller made no tool calls at all.
- **An unreadable or empty reviewer reply** shows up in the tool result's text, not in Pi's exit status: a run can exit 0 and still carry a failed review. That is the subject of "codebase-reviewer: put a review's failure state at the top of the result so a caller cannot report 'no bugs' when a reviewer's reply was unreadable".

On Windows with Git Bash, if you run the `docker` command it prints by hand, set `MSYS_NO_PATHCONV=1` or Git Bash rewrites the mount paths. The driver itself passes paths as a list and is not affected.
