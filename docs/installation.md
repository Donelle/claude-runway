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
> **Use `pipx install`/`uv tool install` here, NOT bare `uvx`.** `uvx` (`uv tool run`) executes a single command in a cache-backed, effectively ephemeral environment — it does not durably add all 3 console scripts to your `PATH` the way the persistent installs above do, and `uv cache clean` (or uv's own cache eviction) can remove that environment later. `claude-runway-setup init` specifically writes config that hardcodes absolute paths into whichever environment ran it (the interpreter path above, plus the MCP-server/hook paths below) — pointing those at a `uvx`-backed environment risks them going stale once that cache entry is gone, and the "next step" reminder it prints (`claude-runway-ingest ...`) would need that same environment to still exist. A one-off `uvx --from git+https://github.com/Donelle/claude-runway.git claude-runway-doctor /path/to/project` for a quick, non-persistent check is fine (the explicit `--from` is required here, not optional decoration — bare `uvx claude-runway-doctor` treats `claude-runway-doctor` itself as the distribution name to resolve, which doesn't exist as a published package; confirmed live, it fails before the entry point ever runs); `claude-runway-setup init` is not that kind of one-off.
>
> **What none of this changes:** `tools/ingest_mcp_server.py`/`tools/compress_mcp_server.py` (the MCP servers `.mcp.json` actually launches) and the hook scripts under `hooks/` still get referenced by an absolute file path in the generated config — same as the clone workflow, just pointing into pipx's/`uv tool`'s own managed install location instead of a repo you cloned yourself. This isn't because Claude Code's `.mcp.json`/`.claude/settings.json` formats can't resolve a bare command via `PATH` — they can (this repo's own doc examples elsewhere use `"command": "python"`/`"python3"` exactly that way) — it's that this package declares no console-script entry points for those specific files, so there's nothing bare on `PATH` for them to resolve to regardless of install method, and pipx/`uv tool install` don't expose a dependency's own console script (like `mcp-server-qdrant`) globally either. An absolute path sidesteps both gaps at once.
>
> **What this doesn't install: the skills.** `claude-runway-setup init` only writes a project's `.mcp.json`/`.claude/settings.json`, and the package itself contains just `libs/`, `tools/`, `hooks/` and `templates/` — `skills/` isn't part of the wheel. `/my-compact`, `/my-resume`, `/my-savings` and `/my-setup-clauderunway` have to be copied into `~/.claude/skills/` separately (see [Updating](#updating) for a way to do that without keeping your own clone, and [Session continuity skills](session-continuity.md)/[Savings tracker](savings-tracker.md) for what each one needs). `/my-setup-clauderunway` additionally expects `CLAUDE_RUNWAY_DIR` to point at a clone of this repo, so with a pipx/`uv tool install` setup you'll usually just run `claude-runway-setup init` directly instead.

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

**If you plan to run `/my-gh-autowork` (or any skill that autonomously drives `git`/`gh` from a subagent) on this checkout**, see `templates/settings.json.template`'s `_permissions_note` and README's Known limitations bullet for issue #195 first — a fresh checkout with no `git`/`gh` Bash allow rule at all in either `.claude/settings.json` or `~/.claude/settings.json`'s `permissions.allow` has been observed hitting an opaque server-side denial on the orchestrator's very first `Agent` call. Add the SPECIFIC, narrow rules the issue actually confirmed (`Bash(git fetch *)`/`Bash(git push *)`/`Bash(gh pr *)`/`Bash(gh api *)`, extended per-subcommand as needed) — do NOT reach for a blanket `Bash(git:*)`/`Bash(gh:*)`/`Bash(uv:*)` wildcard, which adds entire additional command families at once (via git aliases and `uv run`, respectively) rather than just broader convenience, and prefer project-scoped `.claude/settings.json` over `~/.claude/settings.json` where possible so the rule doesn't extend to unrelated checkouts on the same machine. **Narrow is not the same as safe, though**: autonomous development inherently needs code-execution authority, and even these narrow rules still permit it through legitimate flags (`git fetch --upload-pack=<cmd>`, `.venv/bin/python -c '<code>'`, etc. — see `templates/settings.json.template`'s `_permissions_note` for the full account); the actual containment is only running this in a checkout you trust, not the permission syntax. This is unrelated to the hooks/MCP config above and worth setting up before your first autonomous run, not discovering it after one fails.

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

Nothing here updates automatically. Whichever way you installed, you keep the snapshot you installed until you refresh it yourself.

**1. Pull in the new code**

- **Cloned repo** (steps 1–3): `git pull` in the clone, then re-run `uv pip install -r requirements.txt --index-url https://pypi.org/simple` in its venv if `requirements.txt` changed. The absolute paths already written into your projects' `.mcp.json`/`.claude/settings.json` keep pointing at the same files, so they pick up the new code as-is.
- **pipx / `uv tool install`**: `uv tool install` builds from whatever commit was on the default branch at that moment and pins to it. To move to the latest commit:

  ```bash
  uv tool install --reinstall git+https://github.com/Donelle/claude-runway.git
  # pipx equivalent:
  pipx install --force git+https://github.com/Donelle/claude-runway.git
  ```

  `uv tool upgrade claude-runway` is the lighter-weight alternative. If it reports nothing to upgrade when you expect changes, use the `--reinstall` form above. To stay on a fixed version instead of the moving default branch, install from `git+https://github.com/Donelle/claude-runway.git@<tag-or-commit>`. `uv tool list` shows what's installed, but not how far behind it is.

  The interpreter, MCP-server and hook paths that `claude-runway-setup init` wrote into each project live inside the tool's own environment, so a reinstall replaces the files at those same paths and existing configs keep working.

**2. Restart Claude Code.** MCP servers read their code and environment once at startup, so a running session keeps the old server code until you quit and relaunch `claude`.

**3. Re-run setup if the templates changed.** Generated configs are a snapshot of `templates/mcp.json.template`/`templates/settings.json.template` at the time you ran `setup_project.py init`, so a release that adds a server block or changes a hook matcher doesn't reach an existing project on its own. Re-run `init` (or `claude-runway-setup init`) for each project — it merges rather than clobbers, and `--dry-run` previews the result first. If a project already has memory-bank memories, pass the **same** `--memory-bank-collection`/`--memory-bank-id` values you used originally: neither default is sticky across a rerun (see [Memory bank](memory-bank.md#setup)). Run `python tools/doctor.py /path/to/target-repo` afterwards to confirm `.mcp.json` and your shell environment still agree (see [Environment variables](environment-variables.md#keeping-them-in-sync)).

**4. Refresh the skills.** The copies in `~/.claude/skills/` don't change when the package does, and `uv tool install` never installs them in the first place (see the callout under step 3). With a clone, re-run the `cp` commands from [Session continuity skills](session-continuity.md) and [Savings tracker](savings-tracker.md). Without one, take a throwaway shallow clone and copy the whole `skills/` directory:

```bash
git clone --depth 1 https://github.com/Donelle/claude-runway.git /tmp/claude-runway-skills
mkdir -p ~/.claude/skills
cp -r /tmp/claude-runway-skills/skills/* ~/.claude/skills/
```

```powershell
git clone --depth 1 https://github.com/Donelle/claude-runway.git $env:TEMP\claude-runway-skills
New-Item -ItemType Directory -Force ~\.claude\skills | Out-Null
Copy-Item -Recurse -Force $env:TEMP\claude-runway-skills\skills\* ~\.claude\skills\
```

Only copy from the repo's top-level `skills/` directory. The `.claude/skills/` directory holds contributor-only skills for working on this repo itself (see [Development workflow](development-workflow.md)) and doesn't belong in your own `~/.claude/skills/`.

**Dependencies can also drift between updates.** `mcp-server-qdrant` is unpinned in `requirements.txt`, so a fresh install or reinstall may pull a newer upstream release than the one you tested with.

## Uninstalling

`claude-runway-setup` has no uninstall or cleanup command (`init` is its only subcommand), and removing the tool doesn't remove what it wrote elsewhere — a project's config keeps pointing at the deleted environment, and its MCP servers and hooks then fail until you clean that up. So do the steps in this order.

**1. Clean up each configured project (before removing the tool).**

- In the project's `.mcp.json`, delete the `qdrant`, `codebase-indexer`, `memory-bank` and (if present) `local-compress` server blocks, or delete the file if this toolkit was all it held.
- In the project's `.claude/settings.json`, delete the hook blocks this toolkit added (`compress_bash_output.py`, `redirect_webfetch_to_fetch_url.py`, `session_end_savings.py`). `claude-runway-setup init /path/to/project --qdrant-only` strips those hooks for you, but it has to run while the tool is still installed.
- Remove the sections you copied from `templates/CLAUDE.md.template` into the project's `CLAUDE.md` (step 5 above).
- Delete the `.qdrant_index_manifest.json` that `sync_repo`/`index_repo` left in the repo root, if it isn't already gitignored.

**2. Remove the tool.**

```bash
uv tool uninstall claude-runway
```

This deletes the tool's own environment and the `claude-runway-setup`, `claude-runway-ingest` and `claude-runway-doctor` commands.

**3. Remove what's left outside the package** (each is optional — skip anything you want to keep):

- **Skills:** the copies in `~/.claude/skills/` — `my-compact`, `my-resume`, `my-savings` and `my-setup-clauderunway`.
- **Local data:** `~/.claude/claude-runway/` (on Windows, `%USERPROFILE%\.claude\claude-runway`), which holds `savings.db`, `cache.db`, `memory-events.db` and the `fastembed-cache`. Deleting it discards your savings history and the memory-bank usage log; the embedding model is simply downloaded again if you reinstall.
- **Qdrant collections:** they stay in Qdrant until you drop them — each project's own collection, the shared `memory-bank` collection, and the conversation-compact collections (`conversation-compacts-<project>-<hash8>`). Use the dashboard at <http://localhost:6333/dashboard>, or `curl -X DELETE http://localhost:6333/collections/<name>`. Dropping `memory-bank` permanently deletes every project's `remember` entries. If the Qdrant container was set up only for this toolkit, `docker compose down -v` removes it along with its volume.
- **Shell profile:** any `CLAUDE_RUNWAY_*` exports (see [Environment variables](environment-variables.md)), `CLAUDE_RUNWAY_DIR` if you used `/my-setup-clauderunway`, and `TOOLS_REPO_DIR` if you set it. Restart Claude Code afterwards so the change takes effect.
