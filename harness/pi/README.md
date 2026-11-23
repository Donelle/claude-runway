# Testing the codebase-reviewer through Pi, in a container

`pi_driver.py` runs [Pi](https://pi.dev) (a coding agent) in a throwaway Docker container, gives it the reviewer's MCP server, and asks it to call the `review` tool on a small scratch repository. It tests the path a real caller takes, finding the tool over MCP and calling it, which the `live_*.py` scripts never do: they call the reviewer's libraries directly.

A development tool, not product code. CI does not run it (it needs Docker, LM Studio and real models); the driver's logic is covered by `tests/test_harness_pi_driver.py` with a fake `docker`.

## Run it

```
docker build -t claude-runway-pi:dev harness/pi          # once; or add --build below
.venv/bin/python harness/pi/pi_driver.py \
    --model <caller model> \
    --config harness/live-tests/configs/reasoning-seed-coder-8b.json
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
| 1 | it finished without calling `review`, every call to it errored (an error result, or the tool's own `Error: ...` refusal, which it returns as plain text), Pi aborted (`stopped_by: aborted`), or the container failed, including a nonzero exit after Pi had settled (`stopped_by: error`) |
| 2 | a bad argument, a model the preflight could not find, or `docker` could not be started (nothing was run) |
| 3 | a cap stopped it (`stopped_by` says `turns`, `tool_calls` or `timeout`) |

## What keeps it contained

Pi has no turn or tool-call limit of its own, so the driver reads Pi's JSON event stream (split on LF only, as Pi's docs require) and runs `docker kill` when a cap is reached. A turn cap alone would not stop a flood of tool calls inside one turn, hence the three caps. The container is also killed if the driver itself stops for any other reason (Ctrl+C, an error; the `docker` client is stopped first, so a container that is not created yet never is) and if the output ends without Pi having settled (a `docker` client that died leaves its container running), and `docker kill` itself gives up after 30 seconds. It runs with `--init` so Pi, as the container's first process, receives stop signals.

The container sees only: the reviewer's `libs/` and `mcpservers/` (read-only), a scratch case repository (read-only), the reviewers file (read-only), and two scratch folders (Pi's agent directory and the reviewer's database and logs). It never sees the checkout, `~/.claude`, `~/.pi`, Qdrant or the memory bank, and the driver refuses to start if a mount would overlap one. `~/.pi/agent` is Pi's own folder, where it keeps credentials, settings, extensions and sessions; the container gets a scratch one through `PI_CODING_AGENT_DIR` instead. Pi runs with its read-only tools plus the reviewer's, never `bash`, `edit` or `write`. Memory, CPU and process limits are set, and capabilities are dropped.

The limits that remain: the container runs as root on Windows (as your user on Linux and macOS) and has open outbound network access, so it is a sandbox for files and not an egress barrier.

## LM Studio and the models

LM Studio's address comes from `libs/runway_defaults.py` (`CLAUDE_RUNWAY_LMSTUDIO_URL`, else `http://localhost:1234/v1`). The container cannot use `localhost`, so the driver swaps that name for `host.docker.internal` and keeps the port; `--lmstudio-url` and `--host-lmstudio-url` override either side. Windows, macOS and Linux (WSL2) with Docker Desktop have been run (Docker Desktop resolves the name on all three): on plain Linux Docker Engine, without Docker Desktop, the name comes from `--add-host host.docker.internal:host-gateway`, which reaches LM Studio only if it listens beyond `127.0.0.1`, and that has not been tried.

Which model to use per hardware tier is not settled here: see "codebase-reviewer harness: compare local models per hardware tier through Pi and name a default caller and reviewer per tier in the kit's doc". The sizes below are the weights only. Only the row marked "run" has been tried; the rest are candidates. MLX builds run on Apple Silicon only (see the harness README's "Which build runs where"), so the candidates here are the builds that run on any platform.

| Tier | Reference machine | Weights that fit | Candidates installed on the maintainer's machine | Status |
| :-- | :-- | :-- | :-- | :-- |
| Small | 24 GB RAM, no large GPU | 3 to 9 GB | `seed-coder-8b-instruct` 4.8 GB | run as the **reviewer** (below) |
| Small | | | `meta-llama-3.1-8b-instruct` 4.9 GB | not run; a poor caller in an earlier spike |
| Moderate | 32 GB VRAM | up to about 20 GB | `openai/gpt-oss-20b` 12.1 GB (tool-trained) | run as the **caller** (below) |
| Moderate | | | `meta/muse-glimmer` 18.2 GB (tool-trained) | not run |
| Large | more than 32 GB VRAM | 40 GB and up | `meta/llama-3.3-70b` 42.5 GB (tool-trained) | not run; would spill into system memory here |

**The Windows run (2026-11-22, Windows, Docker 29.9.3, Pi 1.6.0).** Caller `openai/gpt-oss-20b` at 16384 context; reviewer `seed-coder-8b-instruct` (`reasoning-seed-coder-8b.json`); case `seeded-three`. Pi called `mcp__reviewer__review` once, with the parameters the prompt gave, in 2 turns and about 9 to 11 seconds, and the driver exited 0. The reviewer reported a shell injection and an off-by-one, two of the case's three seeded kinds of defect (not the division by zero), unverified, as its own result says; the lines were not scored against the case's answers. A second run with `--max-turns 1` was stopped at the start of turn 2: exit 3, `stopped_by: turns`, and no container left behind. After the runs `~/.pi` did not exist, there was no reviewer database under `~/.claude` (the newest file in `~/.claude/claude-runway/` predates the runs), and no container or scratch folder remained. One case and one model pair is a trial that the kit works, not a measurement of either model.

**The macOS run (2026-11-22, Apple Silicon with 48 GB, Docker Desktop 29.9.3 arm64, Pi 1.6.0).** The models installed there are not in the table above, so the config's reviewer was replaced by one built from the arguments. Caller `qwen/qwen3.8-27b` at 16384 context; case `seeded-three`. With `--reviewer-model google/gemma-4-e4b --load-on-demand`, Pi called `mcp__reviewer__review` once, in 2 turns and about 32 to 34 seconds, and the driver exited 0. The reviewer reported all three seeded kinds of defect (shell injection, off-by-one, division by zero), unverified. The reviewer server loaded gemma at 16384 for the review and unloaded it afterwards. The container ran as the host user, `~/.pi` was never created, and no container or scratch folder remained. With the 27B caller also used as the reviewer, the review call hit Pi's 60-second MCP timeout (below): `review_text` was `MCP request timed out after 60000ms`, the caller wandered until the turn cap stopped it (exit 3), and no container was left behind.

**The Linux run (2026-11-22, Ubuntu on WSL2 with mirrored networking, Docker Desktop 29.9.3 through its WSL integration, Pi 1.6.0).** LM Studio ran on the Windows host. The same models and config as the Windows run: caller `openai/gpt-oss-20b` at 16384 context, `reasoning-seed-coder-8b.json`, case `seeded-three`. Pi called `mcp__reviewer__review` once, in 2 turns and about 25 seconds, and the driver exited 0, with the same two findings as on Windows (shell injection at `backup.py:6-8`, off-by-one at `backup.py:10-12`). The container ran as the host user (`--user 1000:1000`) and the read-only bind mounts worked. `--max-turns 1` stopped the run at the start of turn 2 (exit 3, `stopped_by: turns`). A repeat of the plain run hit Pi's 60-second MCP timeout while another large model was loaded beside the reviewer: `review_errors: 1` and exit 1, so it did not read as a pass. A SIGINT to the driver's process group (what Ctrl+C sends) during the review call stopped it in 0.2 seconds (exit 130) and killed the container. After all four runs no container or scratch folder remained, `~/.pi` did not exist, and nothing under `~/.claude/claude-runway/` had changed. The interrupted run left the reviewer model loaded (see "Before and after"). This is Docker Desktop's engine reached from Linux, so plain Docker Engine's `host-gateway` route is still untried.

## What this showed that the orchestrator will need

- **Read the tool's result, not the caller's summary.** The caller's final text differed between two identical runs, and dropped the descriptions and failing inputs; the reviewer's result was the same both times.
- **A review has to finish within Pi's MCP timeout.** Pi gives up on an MCP call after 60 seconds (`MCP request timed out after 60000ms`), and the driver does not change that. A slow reviewer, such as a 27B model on a laptop, needs a smaller reviewer model or a shorter case.
- **A caller needs Pi's MCP naming.** Pi names the tool `mcp__<server>__<tool>`, and `--tools` must list `mcp__reviewer__*` or the tool is not offered.
- **A run needs a models file for LM Studio.** Pi starts with no models; the driver writes a `models.json` that points an OpenAI-compatible provider at LM Studio.
- **An unreadable or empty reviewer reply** shows up in the tool result's text, not in Pi's exit status: a run can exit 0 and still carry a failed review. That is the subject of "codebase-reviewer: put a review's failure state at the top of the result so a caller cannot report 'no bugs' when a reviewer's reply was unreadable".

On Windows with Git Bash, if you run the `docker` command it prints by hand, set `MSYS_NO_PATHCONV=1` or Git Bash rewrites the mount paths. The driver itself passes paths as a list and is not affected.
