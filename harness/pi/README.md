# Testing the codebase-reviewer through Pi, in a container

`pi_driver.py` runs [Pi](https://pi.dev) (a coding agent) in a throwaway Docker container, gives it the reviewer's MCP server, and asks it to call the `review` tool on a small scratch repository. It tests the path a real caller takes, finding the tool over MCP and calling it, which the `live_*.py` scripts never do: they call the reviewer's libraries directly.

A development tool, not product code. CI does not run it (it needs Docker, LM Studio and real models); the driver's logic is covered by `tests/test_harness_pi_driver.py` with a fake `docker`.

## Run it

```
docker build -t claude-runway-pi:dev harness/pi          # once; or add --build below
.venv/bin/python harness/pi/pi_driver.py \
    --model <caller model> \
    --config harness/live-tests/configs/reasoning-gguf-seed-coder-8b.json
```

- `--model` is the model **Pi** calls the tool with (the caller). It must be one LM Studio lists (`lms ls`) and it must make tool calls reliably. Load it first, with the context you pass to `--context-length` (default 16384); see "Before and after" below.
- `--config` is a `harness/live-tests/configs/*.json` file; its reviewers become the reviewer's `REVIEWERS_FILE`, so the reviewer models and contexts come from where the harness already records them. Without it, one reasoning reviewer is built from `--model` (`--reviewer-model` changes it). A config that uses `promptFile` or `translator` is refused: the container does not have those files.
- The reviewer model is loaded by the reviewer server when its reviewer has `loadOnDemand: true` (the config says), and unloaded again afterwards. The driver loads nothing.
- Other options: `--case` (a case from `harness/cases.py`, default `seeded-three`), `--log FILE` (every event Pi emitted, written to a new file: an existing file, or any path under `~/.claude` or `~/.pi`, is refused), `--dry-run` (print the `docker run` command and stop), `--build` and `--image` (build or name the image), and the container's limits `--memory` (default `4g`), `--cpus` (`2`) and `--pids-limit` (`256`).
- The caps are `--max-turns` (default 10), `--max-tool-calls` (30) and `--timeout` (600 seconds for the whole run).

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

Output: one progress line per turn and tool call on stderr, then a JSON summary on stdout. Read **`review_text`**, the reviewer tool's own result; `final_text` is the caller's summary of it, which varies from run to run and can leave most of it out.

| Exit code | Meaning |
| :-- | :-- |
| 0 | Pi finished and the reviewer returned a result |
| 1 | it finished without calling `review`, every call to it errored (an error result, or the tool's own `Error: ...` refusal, which it returns as plain text), Pi aborted (`stopped_by: aborted`), or the container failed, including a nonzero exit after Pi had settled, or Pi's output ended without it settling, whatever the exit code (`stopped_by: error`) |
| 2 | a bad argument, a model the preflight could not find, or `docker` could not be started (nothing was run) |
| 3 | a cap stopped it (`stopped_by` says `turns`, `tool_calls` or `timeout`) |

## What keeps it contained

Pi has no turn or tool-call limit of its own, so the driver reads Pi's JSON event stream (split on LF only, as Pi's docs require) and runs `docker kill` when a cap is reached. A turn cap alone would not stop a flood of tool calls inside one turn, hence the three caps. The container is also killed if the driver itself stops for any other reason (Ctrl+C, an error; the `docker` client is stopped first, so a container that is not created yet never is) and if the output ends without Pi having settled (a `docker` client that died leaves its container running), and `docker kill` itself gives up after 30 seconds. It runs with `--init` so Pi, as the container's first process, receives stop signals.

The container sees only: the reviewer's `libs/` and `mcpservers/` (read-only), a scratch case repository (read-only), the reviewers file (read-only), and two scratch folders (Pi's agent directory and the reviewer's database and logs). It never sees the checkout, `~/.claude`, `~/.pi`, Qdrant or the memory bank, and the driver refuses to start if a mount would overlap one, or if `libs/` or `mcpservers/` is not the checkout's own folder (a symlink in its place is refused, wherever it points). `~/.pi/agent` is Pi's own folder, where it keeps credentials, settings, extensions and sessions; the container gets a scratch one through `PI_CODING_AGENT_DIR` instead. Pi runs with its read-only tools plus the reviewer's, never `bash`, `edit` or `write`. Memory, CPU and process limits are set, and capabilities are dropped.

The limits that remain: the container runs as root on Windows (as your user on Linux and macOS) and has open outbound network access, so it is a sandbox for files and not an egress barrier.

## LM Studio and the models

LM Studio's address comes from `libs/runway_defaults.py` (`CLAUDE_RUNWAY_LMSTUDIO_URL`, else `http://localhost:1234/v1`). The container cannot use `localhost`, so the driver swaps that name for `host.docker.internal` and keeps the port; `--lmstudio-url` and `--host-lmstudio-url` override either side. Windows, macOS and Linux (WSL2) with Docker Desktop have been run (Docker Desktop resolves the name on all three): on plain Linux Docker Engine, without Docker Desktop, the name comes from `--add-host host.docker.internal:host-gateway`, which reaches LM Studio only if it listens beyond `127.0.0.1`, and that has not been tried.

Which reviewer model to use is not a question for this kit: `harness/README.md` records the measured reviewer builds per role (reasoning, verifier, classifier) and which build runs where. What Pi adds is the **caller**: the model that finds the `review` tool over MCP and calls it with the right arguments. This section records which models can do that, per hardware tier. Sizes are the weights only. MLX builds run on Apple Silicon only, so the Mac rows (MLX) and the Windows rows (GGUF) are different builds and are not comparable.

**Machines the comparison is run on** (recorded 2026-11-23; the caller and the reviewer are loaded at the same time, so both must fit in what the GPU can use, which on a Mac is less than the unified memory):

| Machine | CPU / GPU | Memory | OS | Builds it runs |
| :-- | :-- | :-- | :-- | :-- |
| MacBook Pro (Mac17,6) | Apple M5 Max, 18 cores (6 Super, 12 Performance), 40-core GPU | 48 GB unified | macOS 27.2.1 | MLX (`safetensors` in `lms ls`); GGUF also runs under llama.cpp |
| Windows desktop (the maintainer's) | Intel Core Ultra 9 285K, NVIDIA RTX 5090 | 191 GB RAM, 32 GB VRAM | Windows | GGUF |

### Callers, measured on the Mac (Apple M5 Max, 48 GB)

`seeded-three`, 4 runs per pair, so this is a first measurement, not a ranking. Every model was loaded beforehand at 16384 context (no on-demand load), with the default prompt, caps of 10 turns, 30 tool calls and 240 seconds, and Pi's own sampling (the driver sets no caller temperature). A run is **clean** when the driver exited 0 and a review actually ran. Seconds include the reviewer's time. The reviewer is held to a model that answers well inside Pi's 60-second MCP timeout (`qwen3-coder-30b-a3b-instruct-mlx`, 6 to 7 seconds when it calls itself), so that a failure is the caller's; a few runs use another reviewer and are marked.

| Tier | Weights | Caller | Clean | Failed calls before the review | Seconds |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Small | 3 to 9 GB | `google/gemma-4-e4b` (6.9 GB) | 7 of 8 | the first call failed Pi's argument check in 6 of the 7 clean runs and the retry worked | 8 to 40 |
| Small | | `meta-llama-3.1-8b-instruct` (8.5 GB) | **0 of 8** | every call: `params` sent as a string (`expected object, got string`) | 4 to 13 |
| Moderate | 15 to 17 GB | `mistralai/devstral-small-2-2512` (15.1 GB) | 8 of 8 | none | 8 to 16 |
| Moderate | | `qwen3-coder-30b-a3b-instruct-mlx` (17.2 GB) | 15 of 16 | none; the one failure never called `review` (it called `list_reviewers`, then hit the turn cap) | 6 to 26 |
| Moderate | | `qwen/qwen3.8-27b` (16.1 GB) | 4 of 4 | none | 14 to 23 |
| Large | 40 GB and up | none measured | | | no model of this size fits this Mac |

- **Default caller on this Mac:** `devstral-small-2` for the moderate tier (8 of 8, no failed calls); `gemma-4-e4b` for the small tier, with the caveat that it usually needs a retry. `meta-llama-3.1-8b-instruct` is not usable as a caller: this is the failure the earlier spike saw with the GGUF build, so it is not specific to GGUF.
- **Excluded runs:** `qwen3.8-27b` as the caller of itself (4 runs) is left out of the table, because its reviewer decided those runs, not the caller. The 27B never finishes a review inside Pi's 60-second MCP timeout (`MCP request timed out after 60000ms`), which is why it appears only with a fast reviewer above.
- **Where the reviewer varied:** `qwen3-coder-30b` as the caller ran under four reviewers (itself, `gemma-4-e4b`, `meta-llama-3.1-8b-instruct`, `devstral-small-2`); `gemma-4-e4b` ran under itself and under `qwen3-coder-30b`.
- **Failure state:** not tested on `INCOMPLETE REVIEW`, because no run produced that header. Where no review ran (a validation failure or a timeout), no caller reported "no bugs": `gemma-4-e4b` said it could not list findings because of a validation error, `meta-llama-3.1-8b-instruct` said no output file was generated (wrong, but not a clean bill), and `qwen3.8-27b`'s final message was blank.

### Scoring more callers

`score_run.py` turns one run's JSON summary into one row, so every caller is scored the same way. The reviewer's text is matched to the case's seeded bugs by file and line, and a result that is not a review at all (Pi's `Validation failed for tool ...`, `MCP request timed out`) is marked `no review ran` instead of being scored as an empty review:

```
.venv/bin/python harness/pi/pi_driver.py --model <caller> --reviewer-model <reviewer> > run.json
.venv/bin/python harness/pi/score_run.py run.json
```

The row reads `stopped_by=completed readable=yes found=3/3 findings=3 review_calls=1 review_errors=0 turns=2 tool_calls=1 seconds=9.5 caller_repeated=3/3`. For a caller, the columns that matter are `readable`, `review_calls` against `review_errors` (failed calls before the review) and `turns`. `found` says the reviewer worked, not the caller, and `caller_repeated` is how many of the findings the caller's final message repeats. Run each caller 4 times, with the same reviewer, and record it as a row of the table above.

### Windows (GGUF) callers still to run

Not run yet. The same settings and `score_run.py`, with the reviewer held fixed (the earlier Windows runs used `reasoning-gguf-seed-coder-8b.json`). Candidates installed on the maintainer's machine:

| Tier | Reference machine | Weights that fit | Caller candidate | Status |
| :-- | :-- | :-- | :-- | :-- |
| Small | 24 GB RAM, no large GPU | 3 to 9 GB | `meta-llama-3.1-8b-instruct` 4.9 GB (tool-trained) | a poor caller in an earlier spike; not run through `score_run.py` |
| Small | | | `bartowski/llama-3.2-3b-instruct` 1.9 GB (tool-trained) | not run |
| Moderate | 32 GB VRAM | up to about 20 GB | `openai/gpt-oss-20b` 12.1 GB (tool-trained) | run once as the caller (below), not scored over 4 runs |
| Moderate | | | `meta/muse-glimmer` 18.2 GB (tool-trained) | not run |
| Large | more than 32 GB VRAM | 40 GB and up | `meta/llama-3.3-70b` 42.5 GB (tool-trained) | not run; would spill into system memory here |

`seed-coder-8b-instruct` is not tool-trained, so it is not a caller candidate: it appears below as the reviewer of the earlier runs.

**The Windows run (2026-11-22, Windows, Docker 29.9.3, Pi 1.6.0).** Caller `openai/gpt-oss-20b` at 16384 context; reviewer `seed-coder-8b-instruct` (`reasoning-gguf-seed-coder-8b.json`); case `seeded-three`. Pi called `mcp__reviewer__review` once, with the parameters the prompt gave, in 2 turns and about 9 to 11 seconds, and the driver exited 0. The reviewer reported a shell injection and an off-by-one, two of the case's three seeded kinds of defect (not the division by zero), unverified, as its own result says; the lines were not scored against the case's answers. A second run with `--max-turns 1` was stopped at the start of turn 2: exit 3, `stopped_by: turns`, and no container left behind. After the runs `~/.pi` did not exist, there was no reviewer database under `~/.claude` (the newest file in `~/.claude/claude-runway/` predates the runs), and no container or scratch folder remained. One case and one model pair is a trial that the kit works, not a measurement of either model.

**The macOS run (2026-11-22, Apple Silicon with 48 GB, Docker Desktop 29.9.3 arm64, Pi 1.6.0).** The models installed there are not the ones the config names, so the config's reviewer was replaced by one built from the arguments. Caller `qwen/qwen3.8-27b` at 16384 context; case `seeded-three`. With `--reviewer-model google/gemma-4-e4b --load-on-demand`, Pi called `mcp__reviewer__review` once, in 2 turns and about 32 to 34 seconds, and the driver exited 0. The reviewer reported all three seeded kinds of defect (shell injection, off-by-one, division by zero), unverified. The reviewer server loaded gemma at 16384 for the review and unloaded it afterwards. The container ran as the host user, `~/.pi` was never created, and no container or scratch folder remained. With the 27B caller also used as the reviewer, the review call hit Pi's 60-second MCP timeout (below): `review_text` was `MCP request timed out after 60000ms`, the caller wandered until the turn cap stopped it (exit 3), and no container was left behind.

**The Linux run (2026-11-22, Ubuntu on WSL2 with mirrored networking, Docker Desktop 29.9.3 through its WSL integration, Pi 1.6.0).** LM Studio ran on the Windows host. The same models and config as the Windows run: caller `openai/gpt-oss-20b` at 16384 context, `reasoning-gguf-seed-coder-8b.json`, case `seeded-three`. Pi called `mcp__reviewer__review` once, in 2 turns and about 25 seconds, and the driver exited 0, with the same two findings as on Windows (shell injection at `backup.py:6-8`, off-by-one at `backup.py:10-12`). The container ran as the host user (`--user 1000:1000`) and the read-only bind mounts worked. `--max-turns 1` stopped the run at the start of turn 2 (exit 3, `stopped_by: turns`). A repeat of the plain run hit Pi's 60-second MCP timeout while another large model was loaded beside the reviewer: `review_errors: 1` and exit 1, so it did not read as a pass. A SIGINT to the driver's process group (what Ctrl+C sends) during the review call stopped it in 0.2 seconds (exit 130) and killed the container. After all four runs no container or scratch folder remained, `~/.pi` did not exist, and nothing under `~/.claude/claude-runway/` had changed. The interrupted run left the reviewer model loaded (see "Before and after"). This is Docker Desktop's engine reached from Linux, so plain Docker Engine's `host-gateway` route is still untried.

## What this showed that the orchestrator will need

- **Read the tool's result, not the caller's summary.** The caller's final text differed between two identical runs, and dropped the descriptions and failing inputs; the reviewer's result was the same both times.
- **A review has to finish within Pi's MCP timeout.** Pi gives up on an MCP call after 60 seconds (`MCP request timed out after 60000ms`), and the driver does not change that. A slow reviewer, such as a 27B model on a laptop, needs a smaller reviewer model or a shorter case.
- **A caller needs Pi's MCP naming.** Pi names the tool `mcp__<server>__<tool>`, and `--tools` must list `mcp__reviewer__*` or the tool is not offered.
- **A run needs a models file for LM Studio.** Pi starts with no models; the driver writes a `models.json` that points an OpenAI-compatible provider at LM Studio.
- **The reviewer is registered by a file, not a command.** The driver writes Pi's `mcp.json` (the file `pi mcp add` would write) into the agent scratch folder, so the LM Studio URL, which can carry a user name and password, never appears on a command line (host `ps`, `docker inspect`, `--dry-run`). Printed URLs show `***` in place of a user name and password. Pi itself cannot use such a URL as its own provider address: with `http://user:pass@...` the caller made no tool calls at all.
- **An unreadable or empty reviewer reply** shows up in the tool result's text, not in Pi's exit status: a run can exit 0 and still carry a failed review. That is the subject of "codebase-reviewer: put a review's failure state at the top of the result so a caller cannot report 'no bugs' when a reviewer's reply was unreadable".

On Windows with Git Bash, if you run the `docker` command it prints by hand, set `MSYS_NO_PATHCONV=1` or Git Bash rewrites the mount paths. The driver itself passes paths as a list and is not affected.
