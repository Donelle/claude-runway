# Installation

## Start Qdrant with Docker
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
> **Which version/commit is installed, and upgrading** (issue upstream #348). The package version is derived from the repo's commit history at build time (`setuptools-scm`), not hand-edited, so every commit on `main` installs as a distinct, increasing version (e.g. `0.1.1.dev5+gabc1234` — the number counts commits since the repo's `v0.1.0` baseline tag and the `g<sha>` part is the exact commit; a build with uncommitted changes adds a `.d<date>` suffix). Check it with `claude-runway-doctor --version` (also `claude-runway-setup --version` / `claude-runway-ingest --version`), or `uv tool list` / `pipx list`. To pick up newer commits, reinstall from the repo URL: `uv tool install --reinstall git+https://github.com/Donelle/claude-runway.git` (or `pipx install --force git+https://github.com/Donelle/claude-runway.git`), then re-run `claude-runway-setup init` for projects configured under an older install (see the upgrade note above). A build with no repo metadata at all (e.g. a source tarball) reports `0.0.0+unknown`.
>
> **Use `pipx install`/`uv tool install` here, NOT bare `uvx`.** `uvx` (`uv tool run`) executes a single command in a cache-backed, effectively ephemeral environment — it does not durably add all 3 console scripts to your `PATH` the way the persistent installs above do, and `uv cache clean` (or uv's own cache eviction) can remove that environment later. `claude-runway-setup init` specifically writes config that hardcodes absolute paths into whichever environment ran it (the interpreter path above, plus the MCP-server/hook paths below) — pointing those at a `uvx`-backed environment risks them going stale once that cache entry is gone, and the "next step" reminder it prints (`claude-runway-ingest ...`) would need that same environment to still exist. A one-off `uvx --from git+https://github.com/Donelle/claude-runway.git claude-runway-doctor /path/to/project` for a quick, non-persistent check is fine (the explicit `--from` is required here, not optional decoration — bare `uvx claude-runway-doctor` treats `claude-runway-doctor` itself as the distribution name to resolve, which doesn't exist as a published package; confirmed live, it fails before the entry point ever runs); `claude-runway-setup init` is not that kind of one-off.
>
> **What none of this changes:** `tools/ingest_mcp_server.py`/`tools/compress_mcp_server.py` (the MCP servers `.mcp.json` actually launches) and the hook scripts under `hooks/` still get referenced by an absolute file path in the generated config — same as the clone workflow, just pointing into pipx's/`uv tool`'s own managed install location instead of a repo you cloned yourself. This isn't because Claude Code's `.mcp.json`/`.claude/settings.json` formats can't resolve a bare command via `PATH` — they can (this repo's own doc examples elsewhere use `"command": "python"`/`"python3"` exactly that way) — it's that this package declares no console-script entry points for those specific files, so there's nothing bare on `PATH` for them to resolve to regardless of install method, and pipx/`uv tool install` don't expose a dependency's own console script (like `mcp-server-qdrant`) globally either. An absolute path sidesteps both gaps at once.
>
> **The product skills (`skills/`) ship in this same install** (issue [#14](https://github.com/Donelle/claude-runway/issues/14)) — `claude-runway-setup init --install-skills` reads them from wherever this package landed (a clone or a pipx/`uv tool install` environment, transparently) and installs/updates them under `~/.claude/skills/`, so a tool-install user doesn't need a clone just for this step either. See [Session continuity skills](session-continuity.md#installation)/[Savings tracker](savings-tracker.md)/[Shared metrics store](metrics.md) for the per-skill details. One caveat: `my-setup-clauderunway` itself still needs `CLAUDE_RUNWAY_DIR` pointing at a full clone to actually run, regardless of how it was installed — the command prints a reminder about this whenever that skill is among the ones covered by the run (including `--dry-run` previews and a run where it was already up to date, not only when it was actually installed/updated).

**4. Per project you want memory for:**

**Recommended: run the setup script instead of hand-editing the steps below.** `tools/setup_project.py` reads `templates/mcp.json.template` (and, unless `--skip-hooks` is passed, `templates/settings.json.template`) and writes a target project's `.mcp.json`/`.claude/settings.json` with every placeholder below filled in automatically — venv Python path, collection name (defaulted from the target repo's directory name), absolute script paths, home directory — instead of hand-typing them, some of which are required to match exactly across files. Safe to re-run: it merges into (rather than clobbers) any existing `.mcp.json`/`.claude/settings.json` content this toolkit doesn't own, and won't duplicate a hook block it already added. **`--qdrant-only` no longer skips `templates/settings.json.template` entirely** (issue #198): it only omits the local-compress-dependent hook entries (`compress_bash_output.py`, `redirect_webfetch_to_fetch_url.py`, `session_end_savings.py`) — the CORE `record_session_id.py` hook is written either way, since it's base install now, not local-compress-gated. Only `--skip-hooks` (a full opt-out of touching `.claude/settings.json` at all) omits everything, including the core hook.

> **If you received or copied an existing `.mcp.json` from somewhere else (another project, a teammate, a doc example, or even this repo's own `templates/mcp.json.template`) instead of generating it via this script — this isn't only about convenience.** Skipping `init`, or never having run it at all against this specific project ON THIS MACHINE, means the project silently never gets issue upstream #221's fastembed-cache/`HF_HUB_OFFLINE` fix confirmed correct for this machine's own cache. `HF_HUB_OFFLINE` can be **absent** or **present but blank** (`""` — the template's own default, and also what `upgrade` writes) — both behave identically, keeping every launch on the live huggingface.co round-trip indefinitely — or **present as `"1"`** copied from a project that ran `init` successfully on a DIFFERENT machine, which is worse, not better: the fastembed cache path is local per-machine, so startup fails outright instead of merely risking the timeout (see README's Known limitations for the full mechanism and exact code references). Running `init` once against the project on THIS machine, with whatever custom flags it was originally configured with re-passed (collection name, `QDRANT_URL`/API key, memory-bank IDs, local-compress options, …), is the complete fix for all three states — but only once its live warm-up actually succeeds: it can fail (offline, a flaky connection, `fastembed` failing to import), in which case `HF_HUB_OFFLINE` is deliberately left blank and the CLI reports the setup is safe to re-run later to retry (the warm-up-outcome `changes` entry in `run_setup`, `libs/setup_project_lib.py`) rather than silently claiming success — check the CLI's own output (or the resulting `.mcp.json` value) and re-run if it couldn't confirm the cache is warm. `/my-setup-clauderunway` wraps this same path but refuses outright and points you to the raw CLI instead whenever a real `QDRANT_API_KEY` is already configured — it can't safely read/re-pass a credential (see `skills/my-setup-clauderunway/SKILL.md`'s API-key check); use `setup_project.py init`/`claude-runway-setup init` directly with `--qdrant-api-key` re-passed in that case. `merge_mcp_json` overwrites the toolkit-owned server blocks (`qdrant` among them) even against a `.mcp.json` this toolkit never generated. A BARE rerun with no flags instead regenerates every OTHER value in those blocks back to defaults, so re-passing your existing options matters here. **`EMBEDDING_MODEL` is preserved across a re-run** (upstream #279): `init` reuses the model already in the project's existing `.mcp.json` (written to all three blocks, and the cache warm-up targets that same model), so a hand-customized model survives. Pass `--embedding-model <model>` only to change it deliberately — `init` then prints a WARNING, because vectors already indexed under the old model are incompatible (drop the collection in Qdrant and re-index; see README's `EMBEDDING_MODEL` bullets). `upgrade` (below) only detects and fixes the fully-absent-key case (not blank, not copied-`"1"`) by adding the placeholder, and still needs a follow-up `init` to actually warm the cache and set the value.

> **If you installed via pipx/`uv tool install` and just upgraded claude-runway itself** (`pipx upgrade claude-runway` / `uv tool upgrade claude-runway`), re-run `init` (NOT `upgrade` — see below) for every project you'd previously configured — don't just assume the "safe to re-run" merge behavior above means an old config still works untouched. A packaging change moved `libs`/`tools`/`hooks`/`templates` out of site-packages into `<env>/src/{libs,tools,hooks,templates}`: any `.mcp.json`/`.claude/settings.json` an OLDER install's `init` generated hardcodes absolute paths into the old site-packages location, which no longer exists after upgrading past this change — those entries would point at files that were never written there in the new layout, not files that merely moved. `upgrade` (a few paragraphs below) is NOT a substitute here (Copilot review, PR #228): it only applies its own fixed, named list of config gaps (currently: the `memory-bank` server, `record_session_id.py`'s hooks, `HF_HUB_OFFLINE`) — none of them rewrites an existing MCP-server/hook path, so it would leave every stale site-packages path untouched. This is a one-time gotcha for the specific upgrade that crosses issue #206 landing, not a general re-run risk.

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
# COMPRESS-DEPENDENT hooks from .claude/settings.json if a prior run had
# added them, so switching to qdrant-only doesn't leave those still firing.
# The CORE hooks/record_session_id.py hook (issue #198) is written either
# way -- it's base install, not local-compress-dependent, and only
# --skip-hooks (a full opt-out of touching settings.json at all) omits it:
python tools/setup_project.py init /path/to/target-repo --qdrant-only

# common options: --collection-name, --collection-description, --lmstudio-model,
# --lmstudio-url, --qdrant-url, --qdrant-api-key, --include-extensions, --exclude-dirs, --track-savings,
# --compact-collection
python tools/setup_project.py init --help

# Install/update the product skills (my-compact/my-resume/my-savings/my-metrics/
# my-setup-clauderunway) into ~/.claude/skills/ -- combine with a target_repo to do
# both in one run, or pass alone (no target_repo) for skills-only mode (issue #205):
python tools/setup_project.py init /path/to/target-repo --install-skills
python tools/setup_project.py init --install-skills   # skills only
```

**Already configured this project before and just want to pick up a new feature a recent release added, without re-running `init` and re-specifying every option you originally set?** Use `upgrade` instead (issue #225): it compares this project's current `.mcp.json`/`.claude/settings.json` against a fixed list of specific, named config gaps this toolkit knows how to fill (e.g. the `memory-bank` server, the `codebase-indexer` block's `MEMORY_BANK_COLLECTION` key, `record_session_id.py`'s hooks, `HF_HUB_OFFLINE`), tells you about each one it finds still missing, and applies only the ones you approve — leaving everything else (including anything `init` would otherwise reset back to its own defaults/flags) untouched. Safe to run repeatedly: once a migration is applied, it no longer shows up as pending.

```bash
python tools/setup_project.py upgrade /path/to/target-repo             # interactive: prompts per pending migration
python tools/setup_project.py upgrade /path/to/target-repo --dry-run   # preview only, writes nothing
python tools/setup_project.py upgrade /path/to/target-repo --auto-yes  # apply every pending migration, no prompts (scripting/CI)
```

**If you pass `--qdrant-api-key`**, the real key is written in plaintext into `.mcp.json` (never echoed back in `--dry-run`'s preview or the printed follow-up command, which are both redacted/placeholdered instead) — the script prints a warning reminding you NOT to commit `.mcp.json` as-is in that case, replacing the usual "commit it" instruction. There's no built-in mechanism here for keeping the key out of a committed `.mcp.json`; either gitignore `.mcp.json` for that project or manage the key through your own separate process.

The manual steps below still apply if you'd rather hand-edit (or need to understand exactly what the script does / verify its output):

- Copy `templates/mcp.json.template` to `.mcp.json` at that project's repo root.
- Replace `REPLACE-WITH-THIS-PROJECTS-NAME` (appears twice) with a unique collection name for that project.
- Keep `QDRANT_URL`, `COLLECTION_NAME` and `EMBEDDING_MODEL` identical everywhere they appear: `claude-runway-setup init` writes them aligned, but a hand edit must change every block that carries the value (`COLLECTION_NAME` is in the `qdrant` and `codebase-indexer` blocks; `EMBEDDING_MODEL` is also in `memory-bank`). A mismatched `EMBEDDING_MODEL` is usually caught and returned as an actionable error (different models produce different vector names or dimensions), but the check fails open when it can't reach Qdrant or the result is inconclusive, and then searches quietly return irrelevant results instead.
- Optionally set `COLLECTION_DESCRIPTION` to a short one-line description of what this collection contains — applied automatically to the collection (no manual `set_collection_description` call needed) at server startup and after `index_repo`/`sync_repo` write, so `list_collections` from a *different* project's session can surface it. Leave blank (`""`) for no static hint.
- Replace `/absolute/path/to/tools/ingest_mcp_server.py` with the real path from step 2.
- Replace every `REPLACE-WITH-VENV-PYTHON` with the absolute venv Python path from step 3. All three MCP servers use the same venv, so it's the same path in each block — **except** the `qdrant` server's `command`: that field is the literal text `REPLACE-WITH-VENV-PYTHON/bin/mcp-server-qdrant`, which is POSIX-only and wrong on either platform if you substitute only the placeholder token and leave the rest as-is. `mcp-server-qdrant` (hyphenated — this is pip's registered console-script name; `mcp_server_qdrant`, underscored, is only the Python *import* package name and never exists as a file on disk) is installed as a *sibling* of `python` in the same directory as the interpreter — **on macOS/Linux**, replace the ENTIRE `REPLACE-WITH-VENV-PYTHON/bin/mcp-server-qdrant` segment with `<venv-root>/bin/mcp-server-qdrant` (e.g. `/home/youruser/tools/claude-runway/.venv/bin/mcp-server-qdrant`); **on Windows**, the template's hardcoded `/bin/` suffix doesn't apply at all — replace that same segment with `<venv-root>/Scripts/mcp-server-qdrant.exe` instead (e.g. `C:/Users/youruser/tools/claude-runway/.venv/Scripts/mcp-server-qdrant.exe` — use forward slashes here even on Windows, since this value goes into a JSON string: a single backslash like `C:\Users\...` is not a valid JSON escape sequence and will fail to parse, and forward slashes work identically to backslashes in Windows file paths). (`tools/setup_project.py` gets both the correct name and both platforms right automatically — see above — this gotcha only bites the hand-edited path.)
- Replace `REPLACE-WITH-YOUR-HOME-DIR` (in the `qdrant` server's `FASTEMBED_CACHE_PATH`) with your actual absolute home directory (e.g. `/home/youruser` or `/Users/youruser`) — **not** a literal `~`, which `fastembed`'s cache-dir resolution doesn't expand (see [Environment variables](environment-variables.md)). Same value on every project's `.mcp.json` on a given machine.
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

**4c. If also using memory-bank** (`remember`/`recall`/`forget` — durable, cross-session lessons, separate from the per-project Qdrant memory above): `templates/mcp.json.template` already wires up its `memory-bank` block by default, and `tools/setup_project.py init` (step 4 above) already writes it for you — no separate opt-in flag is needed the way `--qdrant-only` opts *out* of local-compress. See [Memory bank](memory-bank.md#setup) for the recommended `--memory-bank-collection`/`--memory-bank-id` flags and the manual `.mcp.json`-editing fallback if you're hand-editing instead of using the script. Unlike local-compress, this needs no dependencies beyond what step 3 (or the "Qdrant memory only" reduced install above) already installs — `tools/memory_bank_mcp_server.py` only imports `mcp`, `mcp-server-qdrant`'s `FastEmbedProvider`, and `qdrant-client`, all already required for the Qdrant memory piece.

**If you plan to run `/my-gh-autowork` (or any skill that autonomously drives `git`/`gh` from a subagent) on this checkout**, see `templates/settings.json.template`'s `_permissions_note` first — a fresh checkout with no `git`/`gh` Bash allow rule at all in either `.claude/settings.json` or `~/.claude/settings.json`'s `permissions.allow` has been observed hitting an opaque server-side denial on the orchestrator's very first `Agent` call. Add the SPECIFIC, narrow rules the issue actually confirmed (`Bash(git fetch *)`/`Bash(git push *)`/`Bash(gh pr *)`/`Bash(gh api *)`, extended per-subcommand as needed) — do NOT reach for a blanket `Bash(git:*)`/`Bash(gh:*)`/`Bash(uv:*)` wildcard, which adds entire additional command families at once (via git aliases and `uv run`, respectively) rather than just broader convenience, and prefer project-scoped `.claude/settings.json` over `~/.claude/settings.json` where possible so the rule doesn't extend to unrelated checkouts on the same machine. **Narrow is not the same as safe, though**: autonomous development inherently needs code-execution authority, and even these narrow rules still permit it through legitimate flags (`git fetch --upload-pack=<cmd>`, `.venv/bin/python -c '<code>'`, etc. — see `templates/settings.json.template`'s `_permissions_note` for the full account); the actual containment is only running this in a checkout you trust, not the permission syntax. This is unrelated to the hooks/MCP config above and worth setting up before your first autonomous run, not discovering it after one fails.

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

## Updating

Updating has two halves: pull newer claude-runway code, then decide whether each already-configured project needs `init` or `upgrade` run against it.

**1. Update claude-runway itself.** Use whichever matches how you installed it:

```bash
# Clone workflow (steps 1-3 above)
cd ~/tools/claude-runway
git pull
uv pip install -r requirements.txt --index-url https://pypi.org/simple   # picks up any new/changed dependencies

# pipx workflow -- reinstall from the repo URL so you get the newest commit
pipx install --force git+https://github.com/Donelle/claude-runway.git

# uv tool workflow
uv tool install --reinstall git+https://github.com/Donelle/claude-runway.git
```

The package version is derived from commit history (see the "Which version/commit is installed" note in the pipx/`uv tool` callout above), so `claude-runway-doctor --version` confirms what you now have. `pipx upgrade claude-runway` / `uv tool upgrade claude-runway` also work, but reinstalling from the URL is the form that always re-resolves the newest `main` commit.

**2. Pick `init` or `upgrade` per configured project.** Neither runs automatically; the generated `.mcp.json`/`.claude/settings.json` keep whatever the older version wrote until you re-run one of them:

- **`upgrade`** (step 4 above) is the default after a normal update. It applies only the specific, named config gaps a newer release added (for example the `memory-bank` server or the `record_session_id.py` hooks), asks before each one, and leaves every option you originally set alone. Preview with `--dry-run` first.
- **`init`** (re-passing the flags you originally used) is required when existing paths in the generated config are no longer valid, or when you need the fastembed-cache/`HF_HUB_OFFLINE` fix confirmed for this machine. In particular, after a pipx/`uv tool` upgrade that crosses issue upstream #206 (the move of `libs`/`tools`/`hooks`/`templates` out of site-packages into `<env>/src/`), `upgrade` will NOT repair the stale absolute paths, so re-run `init`. See the callout under step 4 and README's Known limitations for the full explanation.
- If you installed the product skills, refresh them too: `claude-runway-setup init --install-skills` (or `python tools/setup_project.py init --install-skills` from a clone).

Restart Claude Code in each project afterward so it reloads the MCP servers and hooks.

## Uninstalling

There is no single uninstall command. A full removal touches the places below; each is independent, so you can stop after any of them.

**1. Remove the toolkit from each configured project.** In every project you ran `init`/`upgrade` against:

- **Note any custom data locations before deleting any of the config below.** `CLAUDE_RUNWAY_SAVINGS_DB`, `CLAUDE_RUNWAY_METRICS_DB`, `CLAUDE_RUNWAY_MEMORY_EVENTS_DB`, `CLAUDE_RUNWAY_CACHE_DB`, and `FASTEMBED_CACHE_PATH` can each move a file out of `~/.claude/claude-runway/`. If you set any of them (in `.mcp.json`, `.claude/settings.json`, or your shell profile), write down the paths first so step 4 can remove them too.
- In `.mcp.json`, delete the server blocks this toolkit added: `qdrant`, `codebase-indexer`, `memory-bank`, and (if you used it) `local-compress`. Leave any other servers alone. If the file then has no servers left, delete it.
- In `.claude/settings.json`, delete the hook entries whose commands point at this toolkit's scripts: `record_session_id.py`, `compress_bash_output.py`, `redirect_webfetch_to_fetch_url.py`, and `session_end_savings.py`.
- Remove the claude-runway sections you added to the project's `CLAUDE.md` (from `templates/CLAUDE.md.template`).
- Optionally delete the project's Qdrant collections (via the Qdrant dashboard at <http://localhost:6333/dashboard> or its API) if you want the stored data gone too. These are separate collections and each needs its own deletion: the code index (`COLLECTION_NAME` in the `codebase-indexer` block), the project's conversation compacts (named `<COMPACT_COLLECTION>-<project>-<hash>`, `conversation-compacts-...` by default; find it with `list_collections`). The `memory-bank` collection is different: it is shared by every project on this Qdrant instance (see [Memory bank](memory-bank.md)), so deleting it erases other projects' durable memories too. Leave it alone unless you are removing claude-runway from every project.

**2. Remove the installed product skills** from `~/.claude/skills/`:

```bash
rm -rf ~/.claude/skills/{my-compact,my-resume,my-savings,my-metrics,my-setup-clauderunway}
```

```powershell
# Windows PowerShell
"my-compact","my-resume","my-savings","my-metrics","my-setup-clauderunway" | ForEach-Object { Remove-Item -Recurse -Force "$HOME\.claude\skills\$_" }
```

Also remove the `CLAUDE_RUNWAY_DIR` export (and `CLAUDE_RUNWAY_TRACK_SAVINGS`, if set) from your shell profile.

**3. Remove the claude-runway environment itself:**

```bash
# Clone workflow: delete the clone (this includes its .venv)
rm -rf ~/tools/claude-runway

# pipx workflow
pipx uninstall claude-runway

# uv tool workflow
uv tool uninstall claude-runway
```

```powershell
# Windows PowerShell, clone workflow
Remove-Item -Recurse -Force "$HOME\tools\claude-runway"
```

Do this after step 1: the hook and server entries in your projects point at absolute paths inside this environment, so deleting it first leaves them failing on every session until you clean them up.

**4. Optionally, delete the data this toolkit accumulated outside any project.** By default this is everything under `~/.claude/claude-runway/` (plus any custom locations you noted in step 1):

- `savings.db`, `metrics.db`, `memory-events.db`, `cache.db`: savings, shared metrics, and memory-event history, plus the shared cache. Deleting them discards that history permanently.
- `fastembed-cache/`: the downloaded embedding model. Deleting it just means a re-download if you ever reinstall.

```bash
rm -rf ~/.claude/claude-runway
```

```powershell
# Windows PowerShell
Remove-Item -Recurse -Force "$HOME\.claude\claude-runway"
```

Skip this step if you might reinstall and want to keep your history. The Qdrant container and its `qdrant_storage` volume are separate (see "Start Qdrant with Docker" above); remove them with `docker compose down -v` or `docker rm -f qdrant && docker volume rm qdrant_storage` only if nothing else uses that instance.

