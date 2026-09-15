# ClaudeRunway

Local-compute token savings for Claude Code.

Two independent pieces, both built on the same idea: push work onto local compute (a local vector DB, a local LLM) instead of Claude's own context, so Claude only pays tokens for a distilled result rather than raw data.

1. **Qdrant codebase memory** — per-project semantic memory over code/docs, so Claude can retrieve relevant chunks instead of grepping and reading whole files, and persist distilled findings across sessions.
2. **Local compression (LM Studio)** — a local model summarizes large, mechanical content (logs, build output, big diffs) server-side before anything reaches Claude's context.

Each piece works independently — you don't need LM Studio to use the Qdrant memory, or vice versa.

## Table of contents

- [Files (24)](#files-24)
- [Prerequisites](#prerequisites)
  - [Windows GPU setup for LM Studio](docs/windows-setup.md)
- [Installation](#installation)
- [Environment variables](#environment-variables) — **must be set in two places and kept in sync**
- [Verifying it's working](#verifying-its-working)
- [Cross-repo lookups](#cross-repo-lookups)
- [Skills and hooks](#skills-and-hooks)
  - [Exactness-critical commands (Bash only)](#exactness-critical-commands-bash-only)
  - [fetch_url vs. WebFetch vs. the hooks](#fetch_url-vs-webfetch-vs-the-hooks)
- [Session continuity skills](#session-continuity-skills)
- [Savings tracker](#savings-tracker)
- [Development workflow for this repo](#development-workflow-for-this-repo)
- [Known limitations](#known-limitations)

## Files (24)

| File                              | Purpose                                                                                                                                                                                                                   |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`                  | Console-script packaging for `tools/ingest_to_qdrant.py`/`tools/setup_project.py`/`tools/doctor.py` (`claude-runway-ingest`/`claude-runway-setup`/`claude-runway-doctor`), so `pipx install claude-runway`/`uv tool install claude-runway` don't require cloning this repo just to run those 3 tools. See "Installation"'s "Alternative: pipx/`uv tool install`" callout below for what this does and doesn't replace (and why a bare `uvx` isn't a substitute for `claude-runway-setup init` specifically). |
| `requirements.txt`                | `pip install -r requirements.txt` for everything both pieces need. Grouped by piece with comments — see the file itself if you only want a subset installed.                                                             |
| `libs/setup_project_lib.py`       | Shared, dependency-free logic behind `tools/setup_project.py` — computes defaults (collection name, venv Python path, the `mcp-server-qdrant` console-script path) and patches `templates/mcp.json.template`/`templates/settings.json.template` into a real config, merging with (not clobbering) whatever a target project's `.mcp.json`/`.claude/settings.json` already has. See "Installation" step 4 below. |
| `tools/setup_project.py`          | CLI: `python tools/setup_project.py init /path/to/target-repo` writes that project's `.mcp.json`/`.claude/settings.json` with every `REPLACE-WITH-*`/`/absolute/path/to/...` placeholder filled in automatically, instead of the manual per-project hand-editing in Installation steps 4/4b/5. |
| `libs/doctor_lib.py`              | Shared, dependency-free logic behind `tools/doctor.py` — diffs a target project's `.mcp.json` `local-compress` env block against the live shell environment for each dual-config `CLAUDE_RUNWAY_*` var, applying the exact same default-resolution rules `local_compress_lib.py`/`savings_ledger.py` use (without importing either, to avoid an `openai` dependency for a `--qdrant-only` setup that has no need of it). See "Keeping them in sync" below. |
| `tools/doctor.py`                 | CLI: `python tools/doctor.py /path/to/target-repo` reports any mismatch between `.mcp.json`'s `local-compress` env block and the live shell for `CLAUDE_RUNWAY_LMSTUDIO_URL`/`_MODEL`/`_TRACK_SAVINGS`/`_SAVINGS_DB` instead of leaving that silently split. Exit code 1 on a mismatch, so it doubles as a CI/pre-flight gate. |
| `libs/qdrant_ingest_lib.py`       | Shared, dependency-free chunking logic. `.md`/`.mdx` chunk by heading. Curly-brace languages (`.cs`, `.ts`/`.tsx`, `.js`/`.jsx`, `.java`, `.go`, `.c`/`.h`/`.cpp`/`.hpp`, `.php`, `.swift`, `.kt`, `.scala`, `.rs`) and `.py` chunk boundary-aware — a lightweight brace/bracket-depth heuristic nudges each chunk's cut to the nearest safe function/class boundary instead of an arbitrary fixed line count, falling back to a fixed-line cut when no boundary exists nearby. Every other code extension (`.rb`, `.sql`, `.sh`, `.yaml`/`.yml`, `.json`) and non-markdown docs (`.txt`/`.rst`, since `#` there means "comment", not "heading") still chunk by fixed line windows. Used by both files below — keeps them from drifting apart. |
| `tools/ingest_to_qdrant.py`       | Standalone CLI script for one-off/manual full indexing. Prints progress to the terminal; use this for the first big index of a repo.                                                                                      |
| `tools/ingest_mcp_server.py`      | MCP server exposing `index_repo`, `sync_repo`, `preview_index`, `get_collection_info`, `find_in_collection`, and `list_collections` as tools Claude can call directly. `find_in_collection`/`list_collections` let a session in one project search a DIFFERENT already-indexed repo's collection by exact name — e.g. searching a backend repo's collection from a frontend repo's session — since `qdrant-find` itself is locked to this project's own collection (see "Cross-repo lookups" below). |
| `libs/local_compress_lib.py`      | Shared, MCP-independent map-reduce compression logic (chunking, model resolution, LM Studio calls). Used by both `compress_mcp_server.py` and `hooks/compress_bash_output.py` — keeps them from drifting apart.          |
| `tools/compress_mcp_server.py`    | MCP server exposing `compress_file`, `compress_command_output`, `fetch_url`, `compress_text`, `list_local_models`, and (for the opt-in savings tracker) `savings_summary`/`savings_detail`, backed by a local LM Studio model. Still a draft — design notes and open questions are in the file's docstring. |
| `libs/savings_ledger.py`          | Shared, dependency-light storage + formatting for the opt-in savings tracker (see "Savings tracker" below). Two-tier storage (transient per-session JSONL, perpetual per-project SQLite) behind a small function-based interface — swapping SQLite for another backend later only touches this file. |
| `templates/mcp.json.template`     | Per-project Claude Code config wiring up the `qdrant-find`/`qdrant-store` server, the `codebase-indexer` server, and (optionally) `local-compress` to the same project.                                                   |
| `templates/CLAUDE.md.template`    | Usage rules for Claude covering when to reach for each tool (and when NOT to) — copy the relevant section(s) into a project's `CLAUDE.md`. Note: CLAUDE.md guidance only applies to open-ended prompts, not skills — see "Skills and hooks" below. |
| `templates/settings.json.template` | `PostToolUse` hook config that compresses tool output over a size threshold — by size, not by a hardcoded list of commands, apart from the byte-exact Bash exemptions in "Exactness-critical commands" below. See "Skills and hooks" below.                                                                  |
| `hooks/compress_bash_output.py`   | The hook script referenced by `templates/settings.json.template` at its path in the tools repo (nothing copied into the target project). Filename is legacy -- fires on Bash, Grep, Glob, WebFetch, and WebSearch (matcher covers all five, deliberately excluding Read/Write/Edit), compresses via `libs/local_compress_lib.py` (found automatically in `libs/` under the repo root) if output exceeds a threshold, fails open (leaves output untouched) if LM Studio is unreachable or the import path is somehow wrong. For Bash specifically, commands whose output has to be byte-exact (`git`, `wc`, digests, `jq`, `--json`/`-o json`, `grep -c`, …) are skipped regardless of size — see "Exactness-critical commands" below. |
| `hooks/redirect_webfetch_to_fetch_url.py` | `PreToolUse` hook on `WebFetch` — hard-denies the call and tells Claude to use `fetch_url` instead, but only when LM Studio is reachable right now (checked live); fails open (lets WebFetch through) otherwise. |
| `hooks/session_end_savings.py`    | `SessionEnd` hook for the opt-in savings tracker — rolls the session's transient ledger into the perpetual SQLite store and surfaces a short summary via `systemMessage`. Silent no-op unless `CLAUDE_RUNWAY_TRACK_SAVINGS` is on and the session logged at least one credited event. See "Savings tracker" below. |
| `skills/my-compact/SKILL.md`      | Slash command that summarizes the current conversation via LM Studio and stores it in Qdrant — a local alternative to native `/compact`. See "Session continuity skills" below. |
| `skills/my-resume/SKILL.md`       | Slash command that retrieves a stored session compact from Qdrant by project and label, and restores it into a fresh session. See "Session continuity skills" below. |
| `skills/my-savings/SKILL.md`      | Slash command that shows the savings tracker's simple or detail view (`/my-savings` / `/my-savings detail`). See "Savings tracker" below. |
| `skills/my-setup-clauderunway/SKILL.md` | Slash command that interactively configures a target project for ClaudeRunway — runs `setup_project.py` with guided options and also handles the `CLAUDE.md` update step that the CLI doesn't do. Requires `CLAUDE_RUNWAY_DIR` to be exported. Install to `~/.claude/skills/` and run `/my-setup-clauderunway` from any project you want to configure. |
| `tests/test_compress_bash_output.py` | Table tests for the Bash exactness-critical exempt list (see "Exactness-critical commands" below). Stdlib `unittest`, no network — run with `.venv/bin/python -m unittest discover -s tests`. The exempt list is the one place here where a regex mistake is silent in *both* directions (a false positive forfeits savings unnoticed; a false negative reintroduces the lossy-`git` bug), so the cases are pinned rather than reasoned about. |
| `EVALUATION.md`                   | A plan for measuring whether this actually reduces token usage, rather than assuming it does — covers both the Qdrant memory piece and the local-compress piece as separate tracks, plus Track C for the session continuity skills, plus hard-won lessons on measuring `/usage` cleanly. |

## Prerequisites

- A running Qdrant instance (local or remote), reachable at a URL — required for piece 1
- LM Studio (https://lmstudio.ai) running locally with a model loaded — required for piece 2, optional otherwise. On a Windows laptop with both an Intel integrated GPU and a discrete NVIDIA GPU (a common corporate hardware split), LM Studio may fail to use the discrete GPU by default — see [docs/windows-setup.md](docs/windows-setup.md) for the fix.
- **Network access to `huggingface.co` (or its `storage.googleapis.com` mirror) the first time you index a repo** — piece 1's embedding step (`FastEmbedProvider`, from the `mcp_server_qdrant`/`fastembed` packages) downloads the ONNX embedding model weights on first use, then caches them locally. This is a one-time bootstrap on a machine, not a per-run dependency: verified directly that constructing the provider against an already-cached model makes zero network calls. The cache lives at `FASTEMBED_CACHE_PATH` (default `~/.claude/claude-runway/fastembed-cache`), a persistent location this toolkit controls rather than `fastembed`'s own OS-temp-dir default, so a reboot or periodic temp cleanup no longer reintroduces this network dependency. This is set automatically, no configuration needed, for the `codebase-indexer` MCP server and `ingest_to_qdrant.py`'s CLI usage — but the separate `qdrant` MCP server (the standalone `mcp-server-qdrant` process that `qdrant-find`/`qdrant-store` run through) is third-party code this toolkit doesn't control the startup of, so **its `.mcp.json` env block requires the explicit `FASTEMBED_CACHE_PATH` substitution** in Installation step 4 below — skipping it leaves that one process still using the ephemeral OS-temp-dir default.
- Python 3.12+ for the Qdrant codebase memory piece (`tools/ingest_mcp_server.py`, `tools/ingest_to_qdrant.py`) — project policy minimum; the dependency chain supports >=3.10
- Python 3.11+ for the local compression piece (`tools/compress_mcp_server.py` and hooks) — project policy minimum; the dependency chain supports >=3.10
- [uv](https://docs.astral.sh/uv/) — creates the virtual environment and installs dependencies; required on Linux, recommended on macOS
- Claude Code

## Installation

### Start Qdrant with Docker
If you do not already have a Qdrant instance, install and start
[Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or
Docker Engine (Linux), then save this Compose configuration as `compose.yml`:
```yml
services:
  qdrant:
    image: qdrant/qdrant:v1.10.0  # Replace with your target version
    container_name: qdrant
    ports:
      # Bound to 127.0.0.1 deliberately. Qdrant runs unauthenticated here and
      # holds your indexed code and conversation compacts, so a bare
      # "6333:6333" would publish it on every host interface -- reachable by
      # anything on your network. Same reasoning as the docker run form below.
      - "127.0.0.1:6333:6333"
      - "127.0.0.1:6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
    restart: unless-stopped
volumes:
  qdrant_storage:
```
```bash
docker compose up -d
```
Alternatively, start the same setup directly with Docker:
```bash
docker volume create qdrant_storage
docker run -d --name qdrant --restart unless-stopped -p 127.0.0.1:6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```
The named volume keeps your indexes when the container is stopped or recreated.
Binding the port to `127.0.0.1` keeps this unauthenticated development instance
accessible only from your machine. Confirm that Qdrant is running:
```bash
curl http://localhost:6333
```
You should receive JSON containing Qdrant's title and version. You can also open
the dashboard at <http://localhost:6333/dashboard>. The repository's
`templates/mcp.json.template` already uses the matching
`QDRANT_URL=http://localhost:6333`; leave `QDRANT_API_KEY` empty for this local
instance.
Useful container commands:
```bash
docker compose stop      # stop it
docker compose start     # start it again without losing indexed data
docker compose logs      # inspect startup or runtime errors
```
Run Compose commands from the directory containing `compose.yml`.
If you used the direct `docker run` command instead, use `docker stop qdrant`,
`docker start qdrant`, and `docker logs qdrant`.
For remote or production deployments, configure authentication and TLS rather
than exposing this default unauthenticated container. See Qdrant's
[local quickstart](https://qdrant.tech/documentation/quick-start/) and
[security guide](https://qdrant.tech/documentation/security/).

**1. Install uv** (if not already installed)

[uv](https://docs.astral.sh/uv/) is a cross-platform Python tool manager used to create the virtual environment and install dependencies in isolation — which is required on Linux (system Python is off-limits for third-party packages) and avoids the same class of conflicts on macOS.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS (Homebrew)
brew install uv

# windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your shell (or run `source ~/.bashrc` / `source ~/.zshrc`) to pick up the `uv` and `uvx` commands before continuing.

**2. Place the files somewhere stable** (not inside a project repo — one copy serves all projects), e.g.:

```bash
git clone <this-repo> ~/tools/claude-runway
```

**3. Create a virtual environment and install dependencies**

All systems requires packages to be installed in an isolated Python environment rather than system-wide; a `uv`-managed virtual environment works identically on MacOS and Linux and Windows.

```bash
cd ~/tools/claude-runway
uv venv --python 3.12 
source .venv/bin/activate        # macOS / Linux
uv pip install -r requirements.txt --index-url https://pypi.org/simple
```

PowerShell is the default command-line environment for Windows 10/11 and Windows Terminal.

```powershell
# Navigate to the project directory
cd ~\tools\claude-runway

# Create a virtual environment using Python 3.12
uv venv --python 3.12 

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements from PyPI
uv pip install -r requirements.txt --index-url https://pypi.org/simple
```

> **Note on Execution Policy:**  
> If PowerShell returns an error stating that *script execution is disabled on this system*, run the following command once to allow local scripts to execute:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

If you are using standard Windows Command Prompt (`cmd.exe`):

```cmd
:: Navigate to the project directory
cd %USERPROFILE%\tools\claude-runway

:: Create a virtual environment using Python 3.12
uv venv --python 3.12 

:: Activate the virtual environment
.\.venv\Scripts\activate.bat

:: Install requirements from PyPI
uv pip install -r requirements.txt --index-url https://pypi.org/simple
```

---

> **Corporate PyPI proxy:** if your machine routes pip/uv through an internal package mirror (e.g. a Nexus or Artifactory instance), the `--index-url` flag above bypasses it and installs directly from the public PyPI. All packages in `requirements.txt` are public — no internal packages are required.

Note the absolute path to the venv's Python — you'll use it in place of `REPLACE-WITH-VENV-PYTHON` in the next steps:

```bash
echo "$(pwd)/.venv/bin/python"
# example: /home/youruser/tools/claude-runway/.venv/bin/python
```

If you only want the Qdrant memory piece (no local-compress), install just:

```bash
uv pip install mcp mcp-server-qdrant qdrant-client --index-url https://pypi.org/simple
# optional — enables respecting a project's .gitignore during indexing:
uv pip install pathspec --index-url https://pypi.org/simple
```

> **Alternative: pipx/`uv tool install` instead of steps 1–3**. Steps 1–3 above exist to give you a stable, remembered absolute path plus a dedicated venv — this repo's `pyproject.toml` lets you skip that entirely for the 3 CLI tools it packages as console scripts:
>
> ```bash
> pipx install git+https://github.com/Donelle/claude-runway.git
> # or, uv's own equivalent -- both install PERSISTENTLY, unlike bare `uvx` below:
> uv tool install git+https://github.com/Donelle/claude-runway.git
> ```
>
> This installs `claude-runway-ingest`/`claude-runway-setup`/`claude-runway-doctor` on your `PATH`, backed by pipx's/`uv tool`'s own managed, persistent environment instead of a `.venv` you create and remember yourself. Once installed, `claude-runway-setup init` (step 4 below) correctly detects there's no local clone and points the generated `.mcp.json`/`.claude/settings.json` at that same environment's own interpreter (`sys.executable`, resolved at the moment it runs) instead of a `REPLACE-WITH-VENV-PYTHON`-style path that would never exist for this workflow.
>
> **Use `pipx install`/`uv tool install` here, NOT bare `uvx`.** `uvx` (`uv tool run`) executes a single command in a cache-backed, effectively ephemeral environment — it does not durably add all 3 console scripts to your `PATH` the way the persistent installs above do, and `uv cache clean` (or uv's own cache eviction) can remove that environment later. `claude-runway-setup init` specifically writes config that hardcodes absolute paths into whichever environment ran it (the interpreter path above, plus the MCP-server/hook paths below) — pointing those at a `uvx`-backed environment risks them going stale once that cache entry is gone, and the "next step" reminder it prints (`claude-runway-ingest ...`) would need that same environment to still exist. A one-off `uvx --from git+https://github.com/Donelle/claude-runway.git claude-runway-doctor /path/to/project` for a quick, non-persistent check is fine (the explicit `--from` is required here, not optional decoration — bare `uvx claude-runway-doctor` treats `claude-runway-doctor` itself as the distribution name to resolve, which doesn't exist as a published package; confirmed live, it fails before the entry point ever runs); `claude-runway-setup init` is not that kind of one-off.
>
> **What none of this changes:** `tools/ingest_mcp_server.py`/`tools/compress_mcp_server.py` (the MCP servers `.mcp.json` actually launches) and the hook scripts under `hooks/` still get referenced by an absolute file path in the generated config — same as the clone workflow, just pointing into pipx's/`uv tool`'s own managed install location instead of a repo you cloned yourself. This isn't because Claude Code's `.mcp.json`/`.claude/settings.json` formats can't resolve a bare command via `PATH` — they can (this repo's own doc examples elsewhere use `"command": "python"`/`"python3"` exactly that way) — it's that this package declares no console-script entry points for those specific files, so there's nothing bare on `PATH` for them to resolve to regardless of install method, and pipx/`uv tool install` don't expose a dependency's own console script (like `mcp-server-qdrant`) globally either. An absolute path sidesteps both gaps at once.

**4. Per project you want memory for:**

**Recommended: run the setup script instead of hand-editing the steps below.** `tools/setup_project.py` reads `templates/mcp.json.template` (and, unless `--qdrant-only` is passed, `templates/settings.json.template`) and writes a target project's `.mcp.json`/`.claude/settings.json` with every placeholder below filled in automatically — venv Python path, collection name (defaulted from the target repo's directory name), absolute script paths, home directory — instead of hand-typing them, some of which are required to match exactly across files. Safe to re-run: it merges into (rather than clobbers) any existing `.mcp.json`/`.claude/settings.json` content this toolkit doesn't own, and won't duplicate a hook block it already added.

**Alternatively, use the `/my-setup-clauderunway` Claude Code skill** (see `skills/my-setup-clauderunway/SKILL.md`): install it once to `~/.claude/skills/`, export `CLAUDE_RUNWAY_DIR` in your shell profile pointing at this tools repo, then run `/my-setup-clauderunway` from any project you want to configure. It wraps `setup_project.py` with a guided question flow and also handles the `CLAUDE.md` update step that the CLI doesn't do.

**If you installed via pipx/`uv tool install`** (the "Alternative" callout
above) instead of cloning: replace `python tools/setup_project.py init`
with `claude-runway-setup init` in every command below — same flags, same
behavior, just invoked as the installed console script instead of a path
into a clone that doesn't exist for that install method:

```bash
# from this tools repo, venv activated (or use .venv/bin/python directly):
python tools/setup_project.py init /path/to/target-repo --dry-run   # preview first
python tools/setup_project.py init /path/to/target-repo             # then write for real

# Qdrant memory only, no local-compress: also removes this toolkit's own
# hooks from .claude/settings.json if a prior run had added them, so
# switching to qdrant-only doesn't leave stale compress-dependent hooks
# still firing (leaves the file alone if it never had any):
python tools/setup_project.py init /path/to/target-repo --qdrant-only

# common options: --collection-name, --collection-description, --lmstudio-model,
# --lmstudio-url, --qdrant-url, --qdrant-api-key, --include-extensions, --exclude-dirs, --track-savings,
# --compact-collection
python tools/setup_project.py init --help
```

**If you pass `--qdrant-api-key`**, the real key is written in plaintext into `.mcp.json` (never echoed back in `--dry-run`'s preview or the printed follow-up command, which are both redacted/placeholdered instead) — the script prints a warning reminding you NOT to commit `.mcp.json` as-is in that case, replacing the usual "commit it" instruction. There's no built-in mechanism here for keeping the key out of a committed `.mcp.json`; either gitignore `.mcp.json` for that project or manage the key through your own separate process.

The manual steps below still apply if you'd rather hand-edit (or need to understand exactly what the script does / verify its output):

- Copy `templates/mcp.json.template` to `.mcp.json` at that project's repo root.
- Replace `REPLACE-WITH-THIS-PROJECTS-NAME` (appears twice) with a unique collection name for that project.
- Optionally set `COLLECTION_DESCRIPTION` to a short one-line description of what this collection contains — applied automatically to the collection (no manual `set_collection_description` call needed) at server startup and after `index_repo`/`sync_repo` write, so `list_collections` from a *different* project's session can surface it. Leave blank (`""`) for no static hint.
- Replace `/absolute/path/to/tools/ingest_mcp_server.py` with the real path from step 2.
- Replace every `REPLACE-WITH-VENV-PYTHON` with the absolute venv Python path from step 3. All three MCP servers use the same venv, so it's the same path in each block — **except** the `qdrant` server's `command`: that field is the literal text `REPLACE-WITH-VENV-PYTHON/bin/mcp-server-qdrant`, which is POSIX-only and wrong on either platform if you substitute only the placeholder token and leave the rest as-is. `mcp-server-qdrant` (hyphenated — this is pip's registered console-script name; `mcp_server_qdrant`, underscored, is only the Python *import* package name and never exists as a file on disk) is installed as a *sibling* of `python` in the same directory as the interpreter — **on macOS/Linux**, replace the ENTIRE `REPLACE-WITH-VENV-PYTHON/bin/mcp-server-qdrant` segment with `<venv-root>/bin/mcp-server-qdrant` (e.g. `/home/youruser/tools/claude-runway/.venv/bin/mcp-server-qdrant`); **on Windows**, the template's hardcoded `/bin/` suffix doesn't apply at all — replace that same segment with `<venv-root>/Scripts/mcp-server-qdrant.exe` instead (e.g. `C:/Users/youruser/tools/claude-runway/.venv/Scripts/mcp-server-qdrant.exe` — use forward slashes here even on Windows, since this value goes into a JSON string: a single backslash like `C:\Users\...` is not a valid JSON escape sequence and will fail to parse, and forward slashes work identically to backslashes in Windows file paths). (`tools/setup_project.py` gets both the correct name and both platforms right automatically — see above — this gotcha only bites the hand-edited path.)
- Replace `REPLACE-WITH-YOUR-HOME-DIR` (in the `qdrant` server's `FASTEMBED_CACHE_PATH`) with your actual absolute home directory (e.g. `/home/youruser` or `/Users/youruser`) — **not** a literal `~`, which `fastembed`'s cache-dir resolution doesn't expand (see [Environment variables](#environment-variables)). Same value on every project's `.mcp.json` on a given machine.
- Leave `QDRANT_API_KEY` (present in all three servers' `env` blocks) blank for an unauthenticated local Qdrant instance, or set it to your API key for an authenticated remote one — same variable name `mcp-server-qdrant`/`ingest_mcp_server.py`/`compress_mcp_server.py` all read, so one value keeps every Qdrant-talking server in this file aligned. **If you set a real key, do not commit `.mcp.json` with it inlined** — there's no built-in mechanism here for loading it from an untracked source instead, so either keep `.mcp.json` itself out of version control for this project (add it to `.gitignore`) or manage the key through your own separate, untracked process.
- Commit `.mcp.json` to the repo — **unless** you just set a real `QDRANT_API_KEY` above, in which case committing it publishes that credential in plaintext to anyone with repo access (permanently, since git history retains it even after a later edit); see the caveat on that bullet.

This is what makes memory automatically project-scoped: opening Claude Code in a project loads that project's `.mcp.json`, which points at that project's collection. Opening a different project uses a different collection automatically — no manual switching.

**Controlling what gets ingested:** by default, `index_repo`/`sync_repo`/`ingest_to_qdrant.py` include a built-in list of common code/doc extensions, skip common noise folders (`.git`, `node_modules`, `venv`, `dist`, `build`, etc.), and additionally respect the project's own `.gitignore` if the optional `pathspec` package is installed (add it with `uv pip install pathspec` in the tools repo venv from step 3). To narrow this per project, set in `.mcp.json`'s `codebase-indexer` env block:

- `INDEX_INCLUDE_EXTENSIONS`: comma-separated extensions (e.g. `".py,.md"`) — overrides the built-in list entirely, so only these get ingested.
- `INDEX_EXCLUDE_DIRS`: comma-separated folder names (e.g. `"fixtures,generated"`) — adds to (not replaces) the built-in excludes.

Same options exist as `--include-ext`/`--exclude-dirs`/`--no-gitignore` flags on `ingest_to_qdrant.py`, and as `include_extensions`/`exclude_dirs`/`respect_gitignore` parameters on the `index_repo`/`sync_repo`/`preview_index` tools if you want to override the env defaults for a one-off run. Use `preview_index` first to confirm the filtering is doing what you expect before running a real index.

**4b. If also using local-compress**, add this server to the same `.mcp.json`:

```json
"local-compress": {
  "command": "REPLACE-WITH-VENV-PYTHON",
  "type": "stdio",
  "args": ["/absolute/path/to/tools/compress_mcp_server.py"],
  "env": {
    "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1",
    "CLAUDE_RUNWAY_LMSTUDIO_MODEL": "<exact model id loaded in LM Studio, or omit to auto-detect if only one model is loaded>"
  }
}
```

Replace `REPLACE-WITH-VENV-PYTHON` with the absolute venv Python path from step 3. If `compress_mcp_server.py` fails to start in Claude Code with `MCP error -32000: Connection closed`, it's almost always a missing dependency (`requests`/`trafilatura` were added later for `fetch_url` and are easy to miss). Run it directly using the venv Python to see the real `ModuleNotFoundError`:

```bash
/absolute/path/to/.venv/bin/python /absolute/path/to/tools/compress_mcp_server.py
```

If also using the PostToolUse / PreToolUse hooks from `templates/settings.json.template`, merge that hooks block into the project's `.claude/settings.json` and replace `REPLACE-WITH-VENV-PYTHON` there too — the hook scripts need the same venv Python to find their installed packages.

**5. Add the relevant section(s) from `templates/CLAUDE.md.template` to each project's `CLAUDE.md`** — only include a section for a tool that's actually configured in that project's `.mcp.json`.

**6. Initial index** (first time only, per project):

```bash
# Run from inside the tools repo with the venv activated, or use the venv Python directly:
source ~/tools/claude-runway/.venv/bin/activate
python tools/ingest_to_qdrant.py --repo-path /path/to/project --collection <project-collection-name> --dry-run
# check the preview, then run for real:
python tools/ingest_to_qdrant.py --repo-path /path/to/project --collection <project-collection-name>
```

Or ask Claude to run it via the `index_repo` tool once `.mcp.json` is set up.

**7. Ongoing use:** ask Claude to run `sync_repo` at the start of a session, or add it as a standing instruction in `CLAUDE.md` (see `templates/CLAUDE.md.template`). It only re-embeds changed files, so it's cheap to run routinely.

## Environment variables

**Any variable read by _both_ halves of the toolkit must be set in _two_ places, with _identical_ values: `.mcp.json`'s `env` block *and* an export at the OS/shell level.** Setting only one side does not raise an error, and won't even reliably fail silently either — an MCP server subprocess *does* inherit a shell export when `claude` happens to be launched from a shell that already ran it (see "Why two places" below), but ONLY when that variable's key is genuinely *absent* from the server's own `env` block; a key that's present there, even set to `""`, always overrides/masks the inherited value rather than falling through to it. This toolkit's own shipped templates (`templates/mcp.json.template`, `tools/setup_project.py`) always declare the specific `CLAUDE_RUNWAY_*` variables that actually need coordinating across both halves (`LMSTUDIO_URL`/`LMSTUDIO_MODEL`/`TRACK_SAVINGS`/`SAVINGS_DB`) explicitly in the MCP server's `env` block (blank if disabled) — so for those variables specifically, that inheritance path never actually gets a chance to apply. This doesn't extend to every `CLAUDE_RUNWAY_*` name: `CLAUDE_RUNWAY_CACHE_DB` (read by both the `codebase-indexer` MCP server AND the `redirect_webfetch_to_fetch_url.py` hook — shell export is the override channel since hooks can't read `.mcp.json`) and `CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS`/`CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL` (hook-only) aren't part of the two-places coordination rule — see the `Set it in` column below for which variables actually need both places. This is the single most common misconfiguration here. The `Set it in` column below is authoritative; don't assume every variable needs both.

| Variable | Default | Read by | Set it in |
| --- | --- | --- | --- |
| `CLAUDE_RUNWAY_LMSTUDIO_URL` | `http://localhost:1234/v1` | MCP server + both hooks | **both** |
| `CLAUDE_RUNWAY_LMSTUDIO_MODEL` | none — auto-detect only if exactly one model is loaded | MCP server + `compress_bash_output.py` | **both** |
| `CLAUDE_RUNWAY_TRACK_SAVINGS` | off | MCP server + `compress_bash_output.py` + `session_end_savings.py` | **both** |
| `CLAUDE_RUNWAY_SAVINGS_DB` | `~/.claude/claude-runway/savings.db` | same as above | **both** if overridden — the default needs no coordination, since every script derives it identically |
| `CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS` | `2000` | `compress_bash_output.py` only | **shell only** — the hook is the sole reader, so an `.mcp.json` copy is inert |
| `CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS` | off | `session_end_savings.py` only | **shell only** — hook-only, no MCP counterpart. When set to `1`/`true`/`yes`, the `SessionEnd` hook parses the session transcript to extract real Anthropic token counts. See "Savings tracker" below. |
| `FASTEMBED_CACHE_PATH` | `~/.claude/claude-runway/fastembed-cache` | `fastembed` (third-party), inside all three of: `codebase-indexer` MCP server, `ingest_to_qdrant.py` CLI, and the standalone `qdrant` MCP server | **`codebase-indexer`/CLI: optional** — `ensure_persistent_fastembed_cache()` sets this default automatically if unset, only override to use a different location. **`qdrant` server: required** — it's a third-party process this toolkit doesn't control the startup of, so there's no code hook to default it from; must be set explicitly in `.mcp.json`'s `qdrant` env block (an *absolute* path — no `~`, since `fastembed`'s cache-dir resolution doesn't expand it) |
| `CLAUDE_RUNWAY_CACHE_DB` | `~/.claude/claude-runway/cache.db` | `codebase-indexer` MCP server (`list_collections`/`set_collection_description` hint cache) and `redirect_webfetch_to_fetch_url.py` hook (denied-URL TTL cache, issue #64) | **shell only, if overridden** — since the hook is now a reader and hooks can only inherit from the shell (not from `.mcp.json`), override via shell export rather than `.mcp.json`. The default needs no configuration at all. Safe to delete this file anytime — everything in it is a disposable, always-rebuildable cache, deliberately kept separate from `CLAUDE_RUNWAY_SAVINGS_DB`'s file, which the user wants kept forever |
| `CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL` | `3600` (seconds) | `redirect_webfetch_to_fetch_url.py` hook only | **shell only** — hook-only variable, no MCP counterpart |

### Why two places

The two halves of this toolkit configure environment variables through different guarantees, even though both actually inherit the same base channel (the full parent process environment): MCP servers additionally get a per-server `env` block in `.mcp.json` layered on top of that inheritance; hooks have no equivalent override channel at all.

- **MCP servers** (`local-compress`, `codebase-indexer`) run as stdio subprocesses. Verified directly against live processes (`ps eww -p <pid>`, issue #117): a stdio subprocess actually inherits the **entire parent shell environment** — not the small MCP-spec allowlist this section used to claim — but only for a key that's genuinely *absent* from `.mcp.json`'s own `env` block; a key that's present there, even set to `""`, always overrides/masks whatever the shell would have supplied, rather than merely adding to it. In practice this means relying on inheritance is fragile for a different reason than the old claim implied: `claude` isn't always launched from a shell that ran the export (a GUI launcher, a different profile, a non-interactive shell, another machine), AND this toolkit's own shipped setup (`templates/mcp.json.template`, `tools/setup_project.py`) always pre-declares the specific `CLAUDE_RUNWAY_*` variables that need coordinating across both halves (`LMSTUDIO_URL`/`LMSTUDIO_MODEL`/`TRACK_SAVINGS`/`SAVINGS_DB`) explicitly (blank if disabled) — so for those variables specifically, inheritance never actually gets a chance to apply. (Not every `CLAUDE_RUNWAY_*` name is covered by this — `CLAUDE_RUNWAY_CACHE_DB` genuinely isn't declared in the shipped templates, so it could in principle be supplied by inheritance; since `redirect_webfetch_to_fetch_url.py` also reads it now, an override must be a shell export, not a `.mcp.json` entry — see the env var table above.) Set the coordinated ones explicitly in that server's `env` block in `.mcp.json` regardless.
- **Hooks** (`compress_bash_output.py`, `redirect_webfetch_to_fetch_url.py`, `session_end_savings.py`) are plain command hooks. A hook entry in `.claude/settings.json` has **no `env` field at all**, so it inherits the entire parent environment unfiltered. A shell export is the *only* channel that exists for it, and it cannot read `.mcp.json`.

**Security-relevant consequence of the above:** since a stdio MCP server inherits the entire launching shell's environment, this applies to *every stdio-transport* MCP server configured in a project's `.mcp.json` — not just this toolkit's `local-compress`/`codebase-indexer`. (An HTTP/SSE-transport MCP server is a remote service, not a local subprocess Claude Code spawns, so it can't inherit shell environment this way at all.) An unrelated secret merely exported in your shell (an API key, an auth token) is visible to any stdio server in that file; there's no per-server isolation boundary to rely on here.

So the requirements point in opposite directions and neither substitutes for the other. For a variable both halves read: set it only in `.mcp.json` and the hooks never see it — that failure is absolute, since a hook entry has no `env` field at all to read it from. Set it only in the shell and the MCP server *can* still see it — but only if that key is genuinely absent (not merely blank) from the server's own `env` block, AND `claude` happened to be launched from that exact shell. This toolkit's own shipped templates always declare the specific `CLAUDE_RUNWAY_*` variables that need coordinating across both halves, so in practice that inheritance path never actually applies to those — not to every `CLAUDE_RUNWAY_*` name (see the env var table above). A variable only one half reads (see the `Set it in` column above) only needs that half's channel.

### Keeping them in sync

For the variables that do need both copies, nothing enforces that they agree automatically — divergent values are not an error, they just produce split behavior that's confusing to debug:

- Different `CLAUDE_RUNWAY_LMSTUDIO_MODEL` on each side → your hooks compress with a different model than `compress_file`/`fetch_url` do, so quality differs by code path.
- Different `CLAUDE_RUNWAY_SAVINGS_DB` → the savings tracker writes to two separate databases and `/my-savings` reports from only one of them.
- Different `CLAUDE_RUNWAY_TRACK_SAVINGS` → whichever half is set to off just never logs, so the totals silently under-report rather than erroring.

Run `python tools/doctor.py /path/to/target-repo` to check this instead of relying on noticing split behavior by hand  — it opens that repo's `.mcp.json`, extracts the `local-compress` server's env block, and diffs each of the four vars above against the live shell environment (so run it from the same shell, or an equivalent one with the same exports, that launches `claude`). Exits 1 and prints the specific divergence if anything disagrees, 0 (with a short "OK" line) otherwise — including when `local-compress` isn't configured at all (e.g. a `--qdrant-only` project), which is a valid state, not a misconfiguration. See `libs/doctor_lib.py`'s docstring for exactly how each variable's blank-vs-absent/default distinction is resolved, since that differs per variable (mirroring `local_compress_lib.py`/`savings_ledger.py`'s own real resolution rules).

Shell side (`~/.zshrc` / `~/.bashrc`, wherever `claude` is launched from):

```bash
export CLAUDE_RUNWAY_LMSTUDIO_URL="http://localhost:1234/v1"   # must match .mcp.json
export CLAUDE_RUNWAY_LMSTUDIO_MODEL="google/gemma-3-4b"        # must match .mcp.json
export CLAUDE_RUNWAY_TRACK_SAVINGS="1"                         # must match .mcp.json; only if using the savings tracker
export CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS="2000"           # shell only -- no .mcp.json counterpart
```

Changing either side requires a full restart of Claude Code, not just a new shell: MCP servers read their environment once at startup, and hooks inherit from the already-running `claude` process.

### Renamed in this version

`LMSTUDIO_BASE_URL` → `CLAUDE_RUNWAY_LMSTUDIO_URL`, `LMSTUDIO_MODEL` → `CLAUDE_RUNWAY_LMSTUDIO_MODEL`, `HOOK_COMPRESS_THRESHOLD_CHARS` → `CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS`. **The old names are no longer read at all** — there's deliberately no fallback, so two live name sets can't drift.

The prefix exists precisely *because* these have to live in a global shell profile: an unnamespaced `LMSTUDIO_MODEL` sitting in `~/.zshrc` is easy for another tool to collide with and hard to attribute back to this toolkit.

To make an unmigrated config loud rather than silent, `libs/local_compress_lib.py`'s `stale_env_warning()` detects "old name set, new name missing" and callers surface it — a hard error from the MCP server (checked *before* model auto-detect, so you get the real cause instead of a misleading "multiple models are loaded" message), and stderr plus a note on the fail-open path from the hooks. It stays quiet when both names are set, since the new one wins and the leftover is harmless.

## Verifying it's working

- Call `get_collection_info` to confirm the collection has a nonzero point count.
- Ask a conceptual question about the codebase and check the Claude Code transcript (`~/.claude/projects/<project-hash>/<session-id>.jsonl`) for whether it called `qdrant-find` before `Grep`.
- Compare `/cost` on a conceptual question before and after indexing.

## Cross-repo lookups

`qdrant-find` is locked to whichever collection is configured in the CURRENT project's `.mcp.json` — there's no way to pass it a different collection name, since `mcp-server-qdrant` bakes `COLLECTION_NAME` into the tool at server startup when that env var is set. That's fine for searching your own repo, but doesn't help when a question in one repo genuinely needs another already-indexed repo's codebase to answer — e.g. working in a frontend repo and needing to understand how the backend it calls implements an endpoint.

For that, `ingest_mcp_server.py` (the `codebase-indexer` server) also exposes:

- **`find_in_collection(query, collection, ...)`** — semantically searches ANY named Qdrant collection, not just this project's own. `collection` is required (unlike `qdrant-find`, which never asks for one) — pass the exact collection name. Returns results in the same `<entry><content>...</content><metadata>...</metadata></entry>` format as `qdrant-find`.
- **`list_collections()`** — lists every collection on the connected Qdrant instance with point counts and (by default) each collection's stored description hint, so you can look up another repo's exact collection name if you don't already know it — or let a model pick the right one from the hints alone.
- **`set_collection_description(collection, description, ...)`** — sets or updates a short, one-line hint for any already-indexed collection (including this project's own), so `list_collections` can surface it later. Stored as native Qdrant collection metadata; Qdrant merges rather than replaces it, so this is always safe to call again to update a hint. For this project's *own* collection specifically, setting `COLLECTION_DESCRIPTION` in `.mcp.json` (Installation step 4) does this automatically — no manual call needed.

All three work from any project's session using just the existing `codebase-indexer` server already in `.mcp.json` — no extra config needed. Naming the collection directly in your prompt still works exactly as before (e.g. "search the `acme-support-ticketsapi` collection for how auth works") — an exact name given directly in the prompt is still the highest-signal way to point Claude at the right tool call. Hints are a fallback for when you *don't* already know which collection to name, not a replacement, and not a reintroduction of the standing alias table this section previously argued against: a hint is discovered live via a `list_collections` tool call each time it's needed, not a table sitting in `.mcp.json`/`CLAUDE.md` that Claude would have to recall on its own without ever checking whether it's still accurate. Hints are cached locally after the first fetch (see [Environment variables](#environment-variables)'s `CLAUDE_RUNWAY_CACHE_DB`) so repeated `list_collections` calls — across sessions, across repos — don't re-pay a per-collection Qdrant round-trip once a hint has been fetched anywhere. The one thing to get right for `find_in_collection`: `embedding_model` must match whatever model the OTHER collection was actually indexed with (check that repo's own `.mcp.json`) — a mismatch is now detected before querying and returns an actionable error rather than silently returning irrelevant results; the check fails open on network errors so a transient Qdrant hiccup never blocks a valid query.

## Skills and hooks

`CLAUDE.md` guidance only helps for open-ended prompts. If a project uses custom skills (slash commands like `/load-context` or `/rr-plan`) that have their own step-by-step instructions, those dominate — a skill's explicit steps are followed over CLAUDE.md's general guidance whenever they compete, because CLAUDE.md content stays in context throughout the session but isn't more directive than a skill's own prescribed steps. To get `qdrant-find`/`compress_command_output` used inside a specific skill's workflow, edit that skill's own `SKILL.md` to reference the tool explicitly — there's no shared instructions folder that skills automatically inherit beyond CLAUDE.md itself.

For deterministic enforcement instead of prose guidance, use a hook. `settings.json.template` + `hooks/compress_bash_output.py` register a `PostToolUse` hook on `Bash`, `Grep`, `Glob`, `WebFetch`, `WebSearch`, a vetted allowlist of GitHub MCP tools, and all Splunk MCP tools. These all compress their output once it exceeds a size threshold (`CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS`, default 2000 chars) — driven by measured size, not by a hardcoded list of commands. (The one command-aware carve-out is the byte-exact Bash exemption list below, which subtracts from this rather than defining it.) This works because `PostToolUse` fires after the tool has already run, when the real output size is known, and can rewrite what Claude actually sees via `updatedToolOutput` before it ever reaches Claude's context. Unlike CLAUDE.md, this applies regardless of what skill (if any) is active, since hooks fire on tool events rather than being tied to a particular skill. To install: nothing needs to be copied into the target repo — merge `templates/settings.json.template`'s hooks block into that repo's `.claude/settings.json`, pointing the `args` path at `hooks/compress_bash_output.py`'s real location in the tools repo you cloned in step 2 (same pattern as `.mcp.json` referencing `ingest_mcp_server.py` by absolute path). The script finds `local_compress_lib.py` automatically since it lives in `libs/` under the tools repo root; set `TOOLS_REPO_DIR` only if you've moved the files to a different location.

The dividing line for what's in the matcher isn't "runs locally" (WebFetch/WebSearch actually run through Anthropic's own infrastructure, not local compute) — it's whether Claude ever uses that tool's raw output as the exact, literal basis for a following `Edit`. Bash/Grep/Glob/WebFetch/WebSearch output is read for gist or lookup, never edited from directly, so lossy compression is safe there. The GitHub allowlist applies the same test individually to each tool: search/list/get-by-id operations return text read for gist or lookup (issue bodies, PR metadata, review comments, CI status, commit logs) and are included; `get_file_contents` and `get_pull_request_files` are explicitly excluded because their output (raw file bytes and per-file patch hunks respectively) IS the exact source Claude uses for a following `Edit`. Two write calls (`add_issue_comment`, `create_pull_request_review`) are also included — their responses are created-object JSON (id, url, body, timestamps) read for confirmation only, not edited from. The dividing line is edit-basis, not read-vs-write. Splunk tools are all included since their search results are log/event text read for gist with no file-edit paths.

Bash's `tool_response` shape (`{stdout, stderr}`) is documented and tested directly. The others are not — Claude Code types `tool_response` as `Any`/`unknown` in its hooks docs, so the script doesn't guess field names. Instead it recursively finds every string value in the response, and if the large ones (≥200 chars, to skip metadata like a URL or file count) sum past the threshold, it compresses their concatenation and writes the result back into the single largest field, blanking the other large fields and leaving small ones untouched. This keeps the original JSON shape intact regardless of what the real schema turns out to be. Set `HOOK_DEBUG_LOG` to a file path to see the raw payload for any tool that passes through the hook, useful if you add another tool to the matcher and want to confirm what its `tool_response` actually looks like.

In practice, WebFetch and WebSearch are usually no-ops: both already run their own extraction server-side on Anthropic's infrastructure before Claude sees the result (confirmed empirically — a 2MB fetched Wikipedia page came back as a ~1300 char summary), so there's rarely anything left worth compressing further. Glob is also usually a no-op for a different reason: a large file listing is many *short* strings (individual paths), and the 200-char per-field floor means the aggregate size never triggers compression — deliberately, since file paths are exact identifiers a follow-up `Read`/`Edit`/`Grep` needs verbatim, so summarizing a file listing away would risk the same class of problem `Read`'s exclusion avoids.

`Read` (and `Write`/`Edit`/`NotebookEdit`) are deliberately excluded from the matcher: their output is often the exact basis for a following `Edit`, and a silent lossy rewrite there risks Claude editing from compressed content — a worse failure than a compressed log.

### Exactness-critical commands (Bash only)

The same "is this output the exact basis for what happens next" question gets asked a second time *within* Bash, because — unlike the other matched tools — Bash output isn't one kind of thing. `dotnet build` is a log to skim; `git rev-parse HEAD` is a 40-char identifier where one wrong character is a silent, confident lie. So before compressing a Bash call, the hook checks the command string and skips compression entirely for commands whose value depends on being byte-exact:

| Matched at the **start of a segment** | Matched **anywhere in a segment** |
|---|---|
| `git` (except `clone`/`fetch`/`pull`/`push` — progress noise, not state) | `--porcelain`, `--json`, `--version`, `--query`, `--format=` |
| `wc`, `cksum`, `md5*`, `sha*sum` | `-o json` / `-o tsv` / `-o yaml` |
| `pwd`, `realpath`, `readlink`, `basename`, `dirname`, `which`, `hostname`, `whoami`, `id` | `\| jq` |
| `env`, `printenv`, `jq`, `yq`, `pip freeze`, `npm ls`, `base64`, `openssl`, `xxd`, `od` | `grep -c` (and bundled forms like `grep -rc`) |
| | `--dry-run` (a preview's output *is* the exact planned action) |
| | `--help`, standalone `-h` (flag spellings get copied into the next command; bundled `-lh`/`-sh` don't match) |

Start-of-segment matching happens after leading `VAR=val` assignments and `sudo`, so `FOO=1 git log` and `sudo git status` are caught while `--message "regit"` is not. *Every* segment of a compound command is checked (split on `||`, `&&`, `;`, `|`, newline), because a pipeline's outputs interleave into one stdout — there's no way to compress only part of it, so one exactness-critical segment protects the whole command.

Why this isn't just a higher threshold: a threshold sees only bytes, and the dangerous outputs are frequently the **small** ones. The bug that motivated this — `git show HEAD:f.json | grep -c pattern; git log` compressed 1217 → 924 chars — saved ~300 chars and dropped the `grep -c` count, the entire point of the call. The surrounding `git log` prose survived, so the summary *read* as complete, which is the worst property a lossy summary can have: there's nothing to notice. Raising the threshold would have caught that one incident by accident while still compressing a 3000-char `git diff` and still refusing to compress a 1500-char build log.

Skipping is silent (no `additionalContext` note), matching the under-threshold path — raw output is the correct result here, not something to announce. If a genuinely large exempt output *should* be compressed (a 5MB `git diff`), append `# compress-ok` to the command to opt it back in.

Covered by [tests/test_compress_bash_output.py](tests/test_compress_bash_output.py) — stdlib `unittest`, no network or LM Studio needed, since the check is pure regex over the command string:

```bash
.venv/bin/python -m unittest discover -s tests
```

This hook fails OPEN, not closed: if LM Studio is unreachable or the request otherwise fails, the original output is left completely unchanged and a short note is attached via `additionalContext` so it doesn't silently degrade forever unnoticed. This is different from the harder-line "fail loud" stance used elsewhere in this project (e.g. refusing to guess an ambiguous model) — `PostToolUse` can't block the tool call anyway since the command already ran, so discarding real output on a compression failure would be strictly worse than leaving it alone.

Hooks only work well here because "how large was this output" is measurable after the fact. "Should this have been a `qdrant-find` instead of a `Grep`" has no equivalent measurable signal to hook on — that gap still has to be closed at the skill or CLAUDE.md level, not via hooks. Not every "which tool should this have been" question is like that, though — see the WebFetch/`fetch_url` case below, which DOES have a usable signal (the tool name plus whether local infra is reachable), unlike qdrant-find/Grep which has no equivalent way to know in advance which one is "correct" for a given question.

### fetch_url vs. WebFetch vs. the hooks

There are now three overlapping ways a URL's content can end up compressed, worth being explicit about:

- **`fetch_url`** (in `compress_mcp_server.py`) fetches the page and runs the whole extraction/summarization step on your LOCAL LM Studio model, steerable via `focus`. This is the one that actually shifts the "review the page" work off Anthropic's infrastructure onto local compute — it's the closest to the local-compute cost-savings goal this whole project is built around.
- **WebFetch** (built-in) fetches the page and summarizes it using an Anthropic-hosted model, before Claude ever sees the result. Whether that summarization step itself counts against your token budget is NOT documented anywhere (checked `code.claude.com/docs/en/costs` directly), and real testing suggests it may be unanswerable from `/usage` alone regardless (see `EVALUATION.md`'s "Measuring cleanly" section) — not load-bearing for evaluating `fetch_url` itself, since `EVALUATION.md`'s Track B compares total session cost directly instead. Either way, the step runs on Anthropic's infrastructure, not local compute.
- **The `PostToolUse` hook** on WebFetch is a safety net on top of WebFetch, not a replacement for it — it only fires if WebFetch's own (Anthropic-side) summary somehow still comes back over the threshold, which empirically is rare.

Prefer `fetch_url` when you want the local model to do the reviewing (matches this project's actual goal). Prefer WebFetch directly for anything needing auth, JS rendering, or session/cookie handling — `fetch_url` is a plain unauthenticated GET and won't work for those.

Since CLAUDE.md's "prefer fetch_url" guidance is prose Claude can deprioritize (the same qdrant-find-vs-Grep problem noted above), `templates/settings.json.template` also registers a `PreToolUse` hook (`hooks/redirect_webfetch_to_fetch_url.py`) that enforces it deterministically: it hard-denies any `WebFetch` call and tells Claude to retry with `fetch_url`, but only when LM Studio is reachable right now (checked live, not cached) — if it's down, the hook fails open and lets WebFetch through normally, so a stopped local model never makes WebFetch unusable. This works as a hard `deny` (unlike the size-based `PostToolUse` hook above) because, unlike output size, "should this go to fetch_url instead" is knowable *before* the call — it only depends on the tool name and whether the local model is currently up, not on anything only known after the fact.

The deny reason's exact wording matters more than it might seem: an earlier version phrased it as a suggestion ("use fetch_url instead... if that captures what you're looking for"), and in real testing Claude treated that denial cautiously — it stopped and asked the user for permission to switch tools rather than just proceeding. That's the same "block enforced, retry not enforced" gap as prose guidance in general: the hook deterministically stopped WebFetch, but what Claude did *next* was still just its own judgment call. Rewording the reason to be explicitly imperative ("call fetch_url now — do NOT ask the user for confirmation first") fixed this in testing; Claude now proceeds automatically and only stops to ask if `fetch_url` itself then errors.

The known gap: this can't tell in advance whether a URL needs authentication, JavaScript rendering, or session/cookie handling — cases where `fetch_url`'s plain GET will fail and WebFetch is genuinely required. The deny reason tells Claude to fall back and inform you if that happens, but the hook will keep denying further WebFetch retries to that same URL rather than learning from the failure. If this matters for your usage, either drop `WebFetch` from that hook's matcher, or change its `permissionDecision` from `deny` to `ask` so you can approve WebFetch per-call instead of it being fully automatic.

## Session continuity skills

The `skills/` directory ships two Claude Code slash commands that use both toolkit pieces together to replace the native `/compact` command with a fully local alternative.

| Skill | Invocation | What it does |
|---|---|---|
| `my-compact` | `/my-compact` or `/my-compact "label"` | Summarizes the current conversation via LM Studio (`compress_text`), stores the result in Qdrant with project/date/label metadata, then prompts you to `/clear` and `/my-resume` |
| `my-resume` | `/my-resume` or `/my-resume <keyword>` | Retrieves compacted sessions from Qdrant for the current project, shows a numbered list with labels and dates, and restores the one you pick |

The optional label argument to `/my-compact` (e.g. `/my-compact "auth refactor"`) is how you identify the session later — without it, the label is auto-derived from the summary content. `/my-resume` accepts an optional keyword to narrow the search (e.g. `/my-resume auth refactor`) and skips straight to restore if there's one clear match; when multiple sessions exist it presents a selection UI (`AskUserQuestion`) rather than a plain numbered list.

**Why this matters:** native `/compact` sends your conversation to Anthropic for summarization. These skills route the same work through LM Studio locally — zero Anthropic tokens spent on compaction — and persist the result in Qdrant so it's retrievable across sessions by project and label. See `EVALUATION.md` Track C for how to measure whether this actually saves tokens vs. the alternatives.

**Prerequisites:** both the Qdrant and local-compress pieces must be configured (steps 4 and 4b of Installation above). These skills call `compress_text`, `compact_store`, and `compact_find` — all three live in the `local-compress` server, so only that server needs to be in `.mcp.json` (the `qdrant` server is not required for the session continuity skills).

**Installation:**

```bash
mkdir -p ~/.claude/skills/my-compact ~/.claude/skills/my-resume
cp skills/my-compact/SKILL.md ~/.claude/skills/my-compact/SKILL.md
cp skills/my-resume/SKILL.md ~/.claude/skills/my-resume/SKILL.md
```

Skills install globally (under `~/.claude/skills/`) rather than per-project, so you only need to do this once. The `/my-resume` command retrieves the right session by matching the `project` field in metadata against the current working directory name — switching projects automatically scopes the results.

## Savings tracker

An opt-in, off-by-default feature that estimates context tokens avoided by local compression (`compress_file`, `compress_command_output`, `fetch_url`) and surfaces the estimate automatically at session end plus on demand via `/my-savings`.

**Important framing — this is an estimate, not a benchmark.** `EVALUATION.md` measures savings rigorously: run a task with the tools, run it again without, diff the two (a controlled A/B). This tracker can't do that — there's no baseline run in a live session. Instead it estimates the counterfactual online: `saved ≈ tokens(raw content that was compressed) − tokens(the compressed result actually used)`. Only local-compression events are credited this way, because the tool holds both sides of that comparison directly — it read the raw content and produced the compressed result, so there's nothing to guess. `qdrant-find` activity is intentionally **never** credited as "savings," because its counterfactual (what Grep+Read would have cost instead) isn't observable — the same gap noted under "Skills and hooks" above for qdrant-find-vs-Grep. Labeling a guess as a measurement would undermine trust in the one number here that IS directly observed. See `EVALUATION.md`'s new "Track D" section for the full distinction.

Token counts throughout are estimates (chars ÷ 3.5, the same crude ratio applied to both sides of every comparison so its error mostly cancels in the resulting percentage) — not Claude's real tokenizer, which isn't public. And because tool schemas ride in the request's cached prefix, the fixed per-turn overhead these 5 tools add is reported once, separately, as an annotation — never subtracted from the headline savings number, since doing that would overstate the real tax by roughly 10x on any session past its first turn (cache reads cost far less than the initial cache write).

**Enabling it:**

1. In `.mcp.json`'s `local-compress` server env block, set `CLAUDE_RUNWAY_TRACK_SAVINGS` to `1` (accepted values: `1`, `true`, or `yes`, case-insensitive and whitespace-trimmed — anything else, including unset, is treated as off).
2. **Also export `CLAUDE_RUNWAY_TRACK_SAVINGS=1` at the shell level** (e.g. in `~/.zshrc`/`~/.bashrc`, wherever `claude` gets launched from). Both steps are genuinely required, for two different reasons: an MCP server subprocess does inherit your full shell environment (see [Environment variables](#environment-variables)'s "Why two places" section — not the narrow allowlist this doc used to claim), but only for a key genuinely *absent* from `.mcp.json`'s own `env` block — and `templates/mcp.json.template`/`tools/setup_project.py` always declare `CLAUDE_RUNWAY_TRACK_SAVINGS` explicitly (blank if disabled), so for this toolkit's own generated configs, step 1's explicit entry is unconditionally what the MCP server's own credited tools (`compress_file`/`compress_command_output`/`fetch_url`) actually read — there's no inheritance fallback in practice here. Hook entries in `.claude/settings.json`, by contrast, have no `env` field of their own at all — so they inherit the *entire* parent environment unfiltered, which is exactly what step 2 supplies. Skip step 1 and the MCP server's own credited tools don't see the setting (this toolkit's generated config leaves the key blank rather than omitting it); skip step 2 and `hooks/compress_bash_output.py`/`hooks/session_end_savings.py` don't either — either gap produces silently incomplete tracking, not an error. This is the same rule that applies to `CLAUDE_RUNWAY_LMSTUDIO_URL`/`CLAUDE_RUNWAY_LMSTUDIO_MODEL` for these same hook scripts. `CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS` is the exception, not a fourth example of it — `compress_bash_output.py` is its only reader, so it's shell-only; an `.mcp.json` copy is inert (see the env var table above). See [Environment variables](#environment-variables) for the full list and the reason they share a `CLAUDE_RUNWAY_` prefix.
3. Merge the extended `PostToolUse` matcher (now includes `mcp__local-compress__.*`, a vetted GitHub allowlist, and `mcp__claude_ai_Splunk__.*`) and the new `SessionEnd` hook block from `templates/settings.json.template` into `.claude/settings.json`.
4. Install the skill: `mkdir -p ~/.claude/skills/my-savings && cp skills/my-savings/SKILL.md ~/.claude/skills/my-savings/SKILL.md`.

**Where the data lives:** a single, central SQLite database — `~/.claude/claude-runway/savings.db` by default, one row per session, tagged by project. It's central (not per-repo) on purpose: the cross-project comparison in `/my-savings detail` needs to query across all your projects at once. Override the location with `CLAUDE_RUNWAY_SAVINGS_DB` (an absolute path) in **both** `.mcp.json` and your shell export if you ever change it from the default — same coordination requirement as `QDRANT_URL` elsewhere in this doc. The default needs no such coordination since every script derives it identically. Storage goes through a small function-based interface in `libs/savings_ledger.py`, so SQLite (the current backend) can be swapped later without touching any of its callers.

**Using it:** `/my-savings` shows a simple view (this session + this project's all-time totals); `/my-savings detail` adds a per-tool breakdown, a sparkline of recent sessions, and the cross-project comparison. `fetch_url` calls are always logged but never credited toward the headline number — its honest counterfactual is WebFetch's own already-compressed summary, which isn't something this toolkit observes.

**Real token counts (opt-in):** export `CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS=1` in your shell (shell-only — no `.mcp.json` counterpart needed, since only `hooks/session_end_savings.py` reads it) to have the `SessionEnd` hook parse the session's transcript JSONL and extract the actual tokens Anthropic processed per turn. When enabled, the savings summary adds a token breakdown table and cache efficiency ratio at the end of each session:

```
Tokens Anthropic processed · claude-runway
  Token type       Count     % of total   Relative cost
  ────────────────────────────────────────────────────────
  Fresh input      1,523          0.1%    1.00×  (full price)
  Cache writes     ~5.0M          4.3%    1.25×  (storage overhead)
  Cache reads    ~110.6M         95.6%    0.10×  (10× cheaper than fresh)
  Output           ~660K          0.6%    5.00×  (most expensive per token)
  ────────────────────────────────────────────────────────
  Total          ~115.6M        100.0%

  Local compression avoided   ~320K est.  (0.3% of total processed)
  Cache efficiency            72×  (reads vs. fresh input — higher is better)
```

The **Relative cost** column shows approximate cost ratios vs. full input price (not exact Anthropic rates — those aren't exposed to hooks). The **cache efficiency ratio** (`cache_read_input_tokens ÷ input_tokens`) is a health indicator for how well skills and `CLAUDE.md` are structured for Anthropic's prompt caching layer: a high ratio (e.g. 72×) means context has stabilised into a cacheable shape and Claude is reusing existing knowledge rather than re-ingesting fresh tokens every turn — a sign of well-crafted skills. A low ratio signals a lot of novel uncached content entering each turn.

**Note:** this feature parses Claude Code's internal transcript JSONL format, which is undocumented and may change without notice. It is a stopgap until [anthropics/claude-code#52089](https://github.com/anthropics/claude-code/issues/52089) ships real token counts in the `Stop` hook payload directly (tracked in [#164](https://github.com/Donelle/claude-runway/issues/164)).

## Development workflow for this repo

Separately from the `skills/` directory above (which ships product features that get installed into *other* projects), this repo also carries its own contributor-facing skills in `.claude/skills/` — project-scoped, so they only work inside claude-runway itself, and hardcode this repo's own coordinates (`Donelle/claude-runway`) rather than trying to be generic. They're the recommended way to work an issue end to end, and are the ideal shape for this project's own development cycle.

| Skill | Invocation | What it does |
|---|---|---|
| `my-gh-create-issue` | `/my-gh-create-issue type=bug theme=theme-savings-tracker priority=priority-p1 <description>` | Investigates a described bug/idea first (reproduces it or confirms it by reading the actual code — never files a claim that wasn't checked), then files a GitHub issue in the exact format the existing backlog uses: a **Location**/**Suggested fix** (bug) or **Sanity-checked**/**Proposal** (enhancement) body, labeled by type (`bug`/`enhancement`/`documentation`), `priority-p{0-3}`, and `theme-*`. Missing `type`/`theme`/`priority` get inferred and confirmed rather than guessed silently. |
| `my-gh-code-it` | `/my-gh-code-it 21` | Fetches the GitHub issue, sets its type (`Bug`/`Feature`/`Task`, from labels) and assignee, creates a **native GitHub linked branch** (`gh issue develop`) following this repo's `fix/`/`feature/` naming convention, then plans the fix (seeded from the issue's own **Location**/**Suggested fix** sections) and waits for approval before writing any code. |
| `my-gh-pr` | `/my-gh-pr 21` | Opens the PR (`Fixes #21` in the body, so merging auto-closes the issue) and posts research/plan/validate artifacts as **comments on the issue** — the closest GitHub-native equivalent to attaching RPIV artifacts to a ticket, since GitHub has no generic file-attachment API for issues. |
| `my-gh-pr-feedback` | `/my-gh-pr-feedback <pr-number>` | Fetches and categorizes PR review feedback (blocking/suggestion/question/nitpick/praise), separates human from bot/Copilot comments, and enforces verifying each finding — reproduce it or explain specifically why not — before fixing or replying. Replies never tag `@copilot` — doing so reliably triggers a broken GitHub coding-agent invocation on this repo, confirmed three times (PR #73, PR #84, PR #107); Copilot's automated review re-runs on its own after every push regardless. |
| `my-gh-autowork` | `/my-gh-autowork [issue-number\|all]` | Fully autonomous version of the cycle below: delegates each ticket to a subagent that runs plan → code → PR → verified review-feedback loop → merge with no approval gates, stopping and reporting instead of guessing on anything genuinely ambiguous. For use once a human has explicitly authorized working the backlog without per-step check-ins — it packages the other four skills' procedures (plus the merge step none of them do) into one delegated pass per ticket. |

**Recommended cycle:** `/my-gh-create-issue` to file new work in the same verified, labeled format as the existing backlog → pick an open issue → `/my-gh-code-it {N}` to implement it → `/my-gh-pr {N}` to open the PR and document it against the issue → `/my-gh-pr-feedback` to work through review comments as they come in. This is exactly how the code-review backlog filed against this repo (see `.plans/` locally, or the repo's GitHub issues) is meant to be worked one item at a time — and how new items should keep getting added to it. `/my-gh-autowork` runs this same cycle without stopping between steps, once told to.

**A note on tracking:** `.claude/skills/` is committed to this repo, but a personal `~/.gitignore_global` on some machines (including the one these skills were authored on) blanket-ignores `.claude/` — which is otherwise the right default, since `.claude/settings.json`/`.claude/settings.local.json` hold machine-specific absolute paths that should never be committed (see the "Dogfood config" comments in those files). If a `git status` here ever looks unexpectedly quiet after editing a file under `.claude/skills/`, check for that global rule shadowing it — the fix is `git add -f` for files under this specific path, not disabling the global ignore rule itself.

**Type checking:** `mypy.ini` configures type checking for `libs`/`tools`/`hooks`, but mypy itself is a dev-only tool for CONTRIBUTING to claude-runway, not something an end user installing the toolkit into their own project needs — so it lives in `requirements-dev.txt`, a separate file from the user-facing `requirements.txt`, installed the same way (`uv pip install -r requirements-dev.txt --index-url https://pypi.org/simple`) into the same venv. Run it with `.venv/bin/mypy libs tools hooks` — `/my-gh-code-it`'s test-running step runs this alongside the unit test suite, so both must be clean before a PR.

## Known limitations

- Claude Code doesn't currently render MCP progress notifications visibly in the UI (open issue upstream), though `index_repo`/`sync_repo` still report progress internally to avoid call timeouts on large repos.
- `codebase-indexer` and the `qdrant-find`/`qdrant-store` server are independent MCP servers that both need to agree on `QDRANT_URL`/`COLLECTION_NAME`/`EMBEDDING_MODEL` — the `.mcp.json` template keeps them in sync via shared env vars; don't edit one without the other.
- Changing `EMBEDDING_MODEL` after a collection has data requires a full `index_repo --reset` rebuild — old and new vectors aren't compatible.
- `compress_file`/`compress_command_output`/`fetch_url` have been validated against real sources across multiple rounds of testing; `compress_text` and `list_local_models` are simpler and less exercised. `compress_text` only saves tokens for content you already legitimately hold (e.g. your own draft) — if you had to Read or run something into context first just to pass it to `compress_text`, that defeats the point, since the raw content already cost tokens to get there. Use `compress_file`/`compress_command_output` instead for files/commands, since those read the source server-side and never round-trip the raw content through Claude at all.
- Compression is lossy by nature — don't use either compress tool on source code you intend to edit, or on output you need verbatim (something to parse, a diff to apply, an exact string to grep for next).
- **Unverified assumption underlying `fetch_url`/the WebFetch redirect hook**: whether Claude Code's built-in `WebFetch` tool's own internal extraction/summarization step counts against your token budget is not documented anywhere (checked `code.claude.com/docs/en/costs` directly — no mention), and real testing suggests this specific question may be unanswerable from `/usage` alone regardless (only one model ever appeared in the "Usage by model" breakdown across testing, suggesting any internal step isn't billed against your visible account usage — see `EVALUATION.md`'s "Measuring cleanly" section). This doesn't block evaluating `fetch_url` itself, though: `EVALUATION.md`'s Track B compares total session cost for `fetch_url` vs. `WebFetch` directly on equivalent tasks, which answers "which costs less in practice" without needing to know why.
- **The savings tracker (`/my-savings`) is a heuristic estimate, not a benchmark.** It uses a chars÷3.5 approximation of token counts and estimates a counterfactual online rather than measuring one via a controlled A/B run (that's what `EVALUATION.md`'s tracks are for — see its new Track D for the explicit distinction). Treat its numbers as directional, not exact.
- **`CLAUDE_RUNWAY_TRACK_SAVINGS` needs setting in two places, not one.** A stdio MCP server actually inherits the full parent shell environment (not the narrow allowlist this doc used to claim — see [Environment variables](#environment-variables)'s "Why two places" section), but only for a key genuinely absent from its `env` block — and this toolkit's own generated `.mcp.json` always declares this key explicitly (blank if disabled), so in practice the explicit `.mcp.json` entry is unconditionally what the MCP server reads. Hook entries have no `env` field at all, so they inherit the full parent environment unconditionally — a shell export is the *only* way they see it. See "Savings tracker" above.
- **Relevance classification only runs for a `focus` that names a target.** A non-selective focus ("summarize this," or the default focus used when you pass none) skips the classifier entirely, because there is no subset to filter down to and the classifier is deliberately biased toward NO. This was a real bug, not a precaution: the default focus's "preserve anything that looks like an error, failure, stack trace" clause read to the classifier as its *selection criterion*, so a clean build log and a README chunk were both classified not-relevant 8/8 at temperature=0, while appending a single `error CS0246:` line flipped the same log to relevant. The practical effect was that the `PostToolUse` hook compressed only **failing** output — successful builds, passing test runs, docs and diffs paid for an LM Studio round-trip and then came back raw. Selective focuses are unaffected and still filter normally (verified both ways: a focus whose target is absent is still rejected, one whose target is present still extracts it).
- `focus` on `compress_file`/`compress_command_output`/`fetch_url` handles positional asks ("the lead section," "the first N lines," "the abstract") automatically — recognized phrasings get truncated to the document's actual beginning (via structural section-heading detection where possible, a fixed-size fallback otherwise) before any classification runs, verified end-to-end against a real Wikipedia article and a real local model with no retries needed. This took several rounds of real-world testing to get right — see `libs/local_compress_lib.py`'s docstrings (`_find_first_heading_boundary`, `classify_relevant`, `compress`) for the specific failures found and fixed along the way, useful reading if you're extending this logic. Two residual limits worth knowing: the heading-detection heuristic only helps for content with clear heading-like structure (falls back to a fixed ~4000-char guess for plain text with no heading structure, e.g. a log file); and `_POSITIONAL_FOCUS_HINTS` is a fixed keyword list — a positional phrasing outside that list falls through to the position-aware classifier instead, which real testing showed isn't fully reliable on its own.
- **The pipx/`uv tool install` alternative only supports those two install methods, not a bare `pip install`/`pip install --user`.** `claude-runway-setup init` resolves the interpreter it writes into the generated `.mcp.json` as `sys.executable`, then derives `mcp-server-qdrant`'s own path as a SIBLING of that interpreter in the same directory — true for pipx and `uv tool install` (each creates a dedicated venv where a dependency's console script really does live next to the interpreter; confirmed directly for `uv tool install`), but not guaranteed for a bare `pip install --user`, where the running interpreter can be a shared system Python while installed console scripts land somewhere else entirely (e.g. `~/.local/bin`). Use pipx or `uv tool install`, not a bare `pip install`, for this workflow.
- **`sync_repo`/`index_repo` retry once on a dropped Qdrant connection** (`libs/qdrant_retry.py`) — on macOS, Docker Desktop's userspace networking proxy (`vpnkit`) can silently close an idle pooled connection during the CPU-bound gap while the local embedding model loads, surfacing as `httpx.RemoteProtocolError`/`ConnectError`; this is a known, long-standing Docker Desktop for Mac limitation ([moby/vpnkit#414](https://github.com/moby/vpnkit/issues/414), [docker/for-mac#5857](https://github.com/docker/for-mac/issues/5857)), not a Qdrant-side issue — the failing request never reaches the Qdrant container at all, so there's no Qdrant config that helps. If the retry path still gets exercised more often than you'd like, Docker Desktop's `vpnKitMaxPortIdleTime` setting (in `settings-store.json`, default 300s; set to `0` to disable culling entirely) can raise the idle threshold, though that only reduces how often it happens rather than eliminating the race, and only applies on macOS.
