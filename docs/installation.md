# Installation

## Start Qdrant with Docker
If you do not already have a Qdrant instance, install and start
[Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or
Docker Engine (Linux), then save this Compose configuration as `compose.yml`:
```yml
services:
  qdrant:
    image: qdrant/qdrant:v1.19.2  # Pinned on purpose; change it deliberately to upgrade (see "Upgrading Qdrant")
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
docker run -d --name qdrant --restart unless-stopped -p 127.0.0.1:6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.19.2
```
Pin the tag in `docker run` too: an unpinned `qdrant/qdrant` means `latest`, so
after a `docker pull` the next time the container is recreated it can silently
come up on a newer, unreviewed version.

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
Run Compose commands from the directory containing `compose.yml` so Compose
finds it and keeps the same project name (see "Upgrading Qdrant" below).
If you used the direct `docker run` command instead, use `docker stop qdrant`,
`docker start qdrant`, and `docker logs qdrant`.

For remote or production deployments, configure authentication and TLS rather
than exposing this default unauthenticated container. See Qdrant's
[local quickstart](https://qdrant.tech/documentation/quick-start/) and
[security guide](https://qdrant.tech/documentation/security/).

### Upgrading Qdrant
Your data lives in the named volume, so these points decide whether it survives:
- **Which volume Compose uses depends on the project name.** Compose prefixes
  the volume name with the project name, which defaults to the name of the
  directory containing the Compose file (a `compose.yml` in `~/Docker` creates
  `docker_qdrant_storage`). Copying the file to a differently named directory, or using
  a different `-p`, `COMPOSE_PROJECT_NAME`, or top-level `name:`, creates a new,
  empty volume. Your collections look gone, but the old volume is intact; switch
  back to the original project name to see it again. List volumes with
  `docker volume ls`. See Docker's
  [project name docs](https://docs.docker.com/compose/how-tos/project-name/).
- **What deletes data:** `docker compose down -v` (removes the named volumes
  declared in the file) and `docker volume rm`. Plain `docker compose down`,
  `stop`, and recreating the container keep the volume.
- **Back up first.** Take a [snapshot](https://qdrant.tech/documentation/snapshots/)
  or, with Qdrant stopped (`docker compose stop`; a raw copy of a running
  instance's files may be inconsistent), copy the volume before upgrading. Qdrant's upgrade guide does not describe
  downgrading, so treat it as unsupported: your backup is the way back. Code
  collections can be rebuilt with `index_repo(reset=True)` (`sync_repo` refuses
  when its local manifest still lists files the empty collection lacks), but conversation
  compacts (`compact_store`) and the shared memory-bank entries (`remember`)
  cannot, so they are what the backup protects most.
- **Upgrade:** edit the `image:` tag, then run `docker compose pull` and
  `docker compose up -d`. No `down` is needed; Compose recreates the container on
  the same volume. Qdrant says to upgrade to the latest patch of each
  intermediate minor version first (for example 1.17 to 1.19 goes via 1.18.x),
  and read the release notes for breaking changes. For the direct `docker run`
  form, `docker stop qdrant && docker rm qdrant`, then re-run the command with the
  new tag and the same `-v qdrant_storage:/qdrant/storage`.
- **Verify:** `curl http://localhost:6333` shows the new version, and
  `curl http://localhost:6333/collections` lists your collections.

The authoritative rules are in Qdrant's [upgrade guide](https://qdrant.tech/documentation/upgrades/).

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
uv pip install mcp qdrant-client fastembed pydantic_settings --index-url https://pypi.org/simple
# optional — enables respecting a project's .gitignore during indexing:
uv pip install pathspec --index-url https://pypi.org/simple
```

> **Alternative: pipx/`uv tool install` instead of steps 1–3**. Steps 1–3 above exist to give you a stable, remembered absolute path plus a dedicated venv — this repo's `pyproject.toml` lets you skip that entirely for the 3 CLI tools it packages as console scripts:
>
> ```bash
> pipx install --python python3.12 git+https://github.com/Donelle/claude-runway.git
> # or, uv's own equivalent -- both install PERSISTENTLY, unlike bare `uvx` below:
> uv tool install --python 3.12 git+https://github.com/Donelle/claude-runway.git
> ```
>
> **The savings tracker is off unless you turn it on, and `/my-savings` and `/my-metrics` need it.** A bare `claude-runway-setup init` writes no `analytics` MCP server, so those two skills fail with an unknown-tool error. On a terminal, `init` (and `upgrade`) asks whether to enable the tracker; a piped or CI run, `--dry-run` and `upgrade --auto-yes` never ask and leave it off, and `init` prints a note saying so. To enable it without the question, run `claude-runway-setup init <project> --track-savings` and `export CLAUDE_RUNWAY_TRACK_SAVINGS=1` in your shell (see [Savings tracker](savings-tracker.md)). An older version already installed does not ask: reinstall it with `--reinstall` (see "Updating" below) to pick up new config behavior.
>
> **The `codebase-reviewer` server is opt-in too, and needs a file you write.** `claude-runway-setup init <project> --codebase-reviewer --reviewers-file /path/you/choose/reviewers.json` writes its `.mcp.json` block, and an interactive `upgrade` offers it. `REVIEWERS_FILE` is required and has no default: neither command ever writes a default path or a reviewers file, and both warn when it is not set. See [Codebase reviewer](codebase-reviewer.md) for the file's format.
>
> **Pin the Python (`--python`).** Without `--python`, `uv tool install` uses the newest Python it finds, while `pipx` uses the interpreter pipx itself runs on (or `PIPX_DEFAULT_PYTHON`, if set). 3.12 is the version this project tests (CI, [prerequisites](prerequisites.md)). If the interpreter that gets selected is newer than `onnxruntime` (required by `fastembed`) has wheels for, the install fails with a long "your requirements are unsatisfiable" explanation that reads like a dependency conflict but is really a Python-version problem. Pinning 3.12 avoids it with either tool. If the same output also shows a `401 Unauthorized` for a corporate index (e.g. Nexus), that is unrelated; add `--index-url https://pypi.org/simple` to the install command to bypass it, as the clone workflow does.
>
> This installs `claude-runway-ingest`/`claude-runway-setup`/`claude-runway-doctor` on your `PATH`, backed by pipx's/`uv tool`'s own managed, persistent environment instead of a `.venv` you create and remember yourself. Once installed, `claude-runway-setup init` (step 4 below) correctly detects there's no local clone and points the generated `.mcp.json`/`.claude/settings.json` at that same environment's own interpreter (`sys.executable`, resolved at the moment it runs) instead of a `REPLACE-WITH-VENV-PYTHON`-style path that would never exist for this workflow.
>
> **Which version/commit is installed, and upgrading**. The package version is derived from the repo's commit history at build time (`setuptools-scm`), not hand-edited, so every commit on `main` installs as a distinct, increasing version (e.g. `0.1.1.dev5+gabc1234` — the number counts commits since the repo's `v0.1.0` baseline tag and the `g<sha>` part is the exact commit; a build with uncommitted changes adds a `.d<date>` suffix). Check it with `claude-runway-doctor --version` (also `claude-runway-setup --version` / `claude-runway-ingest --version`), or `uv tool list` / `pipx list`. To pick up newer commits, reinstall from the repo URL: `uv tool install --reinstall --python 3.12 git+https://github.com/Donelle/claude-runway.git` (or `pipx install --force --python python3.12 git+https://github.com/Donelle/claude-runway.git`), then re-run `claude-runway-setup init` for projects configured under an older install (see the upgrade note above). A build with no repo metadata at all (e.g. a source tarball) reports `0.0.0+unknown`.
>
> **Use `pipx install`/`uv tool install` here, NOT bare `uvx`.** `uvx` (`uv tool run`) executes a single command in a cache-backed, effectively ephemeral environment — it does not durably add all 3 console scripts to your `PATH` the way the persistent installs above do, and `uv cache clean` (or uv's own cache eviction) can remove that environment later. `claude-runway-setup init` specifically writes config that hardcodes absolute paths into whichever environment ran it (the interpreter path above, plus the MCP-server/hook paths below) — pointing those at a `uvx`-backed environment risks them going stale once that cache entry is gone, and the "next step" reminder it prints (`claude-runway-ingest ...`) would need that same environment to still exist. A one-off `uvx --from git+https://github.com/Donelle/claude-runway.git claude-runway-doctor /path/to/project` for a quick, non-persistent check is fine (the explicit `--from` is required here, not optional decoration — bare `uvx claude-runway-doctor` treats `claude-runway-doctor` itself as the distribution name to resolve, which doesn't exist as a published package; confirmed live, it fails before the entry point ever runs); `claude-runway-setup init` is not that kind of one-off.
>
> **What none of this changes:** `mcpservers/qdrant_mcp_server.py`/`mcpservers/ingest_mcp_server.py`/`mcpservers/memory_bank_mcp_server.py`/`mcpservers/compress_mcp_server.py` (the MCP servers `.mcp.json` actually launches) and the hook scripts under `hooks/` still get referenced by an absolute file path in the generated config — same as the clone workflow, just pointing into pipx's/`uv tool`'s own managed install location instead of a repo you cloned yourself. This isn't because Claude Code's `.mcp.json`/`.claude/settings.json` formats can't resolve a bare command via `PATH` — they can (this repo's own doc examples elsewhere use `"command": "python"`/`"python3"` exactly that way) — it's that this package declares no console-script entry points for those specific files, so there's nothing bare on `PATH` for them to resolve to regardless of install method. An absolute path to the one interpreter every server runs under sidesteps that.
>
> **The product skills (`skills/`) ship in this same install** — `claude-runway-setup init --install-skills` reads them from wherever this package landed (a clone or a pipx/`uv tool install` environment, transparently) and installs/updates them under `~/.claude/skills/`, so a tool-install user doesn't need a clone just for this step either. See [Session continuity skills](session-continuity.md#installation)/[Savings tracker](savings-tracker.md)/[Shared metrics store](metrics.md) for the per-skill details. One caveat: `my-setup-clauderunway` itself still needs `CLAUDE_RUNWAY_DIR` pointing at a full clone to actually run, regardless of how it was installed — the command prints a reminder about this whenever that skill is among the ones covered by the run (including `--dry-run` previews and a run where it was already up to date, not only when it was actually installed/updated).

**4. Per project you want memory for:**

**Recommended: run the setup script instead of hand-editing the steps below.** `tools/setup_project.py` reads `templates/mcp.json.template` (and, unless `--skip-hooks` is passed, `templates/settings.json.template`) and writes a target project's `.mcp.json`/`.claude/settings.json` with every placeholder below filled in automatically (and does step 5's `CLAUDE.md` update too) — venv Python path, collection name (defaulted from the target repo's directory name), absolute script paths, home directory — instead of hand-typing them, some of which are required to match exactly across files. Safe to re-run: it merges into (rather than clobbers) any existing `.mcp.json`/`.claude/settings.json` content this toolkit doesn't own, and won't duplicate a hook block it already added. If an existing `.mcp.json`/`.claude/settings.json` can't be merged into (not valid JSON, a top level that isn't an object, or an `mcpServers`/`hooks` member of the wrong shape), `init` and `upgrade` print a one-line `error:` naming the file and the problem and exit 1 without writing anything, the same condition `claude-runway-doctor` reports; fix or remove the file and re-run. **`--qdrant-only` no longer skips `templates/settings.json.template` entirely**: it only omits the local-compress-dependent hook entries (`compress_output.py`, `redirect_webfetch_to_fetch_url.py`, `session_end_savings.py`) — the CORE `record_session_id.py` hook is written either way, since it's base install now, not local-compress-gated. Only `--skip-hooks` (a full opt-out of touching `.claude/settings.json` at all) omits everything, including the core hook.

> **If you received or copied an existing `.mcp.json` from somewhere else (another project, a teammate, a doc example, or even this repo's own `templates/mcp.json.template`) instead of generating it via this script — this isn't only about convenience.** The paths in it (the venv interpreter, the tools repo) are local to whichever machine generated it, and a config written by an older release may still launch the third-party Qdrant MCP server's console script for the `qdrant` server, which lacks the conversation-compact tools, instead of this repo's own `mcpservers/qdrant_mcp_server.py`, which provides them. Running `init` once against the project on THIS machine, with whatever custom flags it was originally configured with re-passed (collection name, `QDRANT_URL`/API key, memory-bank IDs, local-compress options, …), regenerates every path for this machine; on a project that was already generated here, `upgrade` (below) moves an old `qdrant` block to the new server without resetting anything else. `/my-setup-clauderunway` wraps this same path but refuses outright and points you to the raw CLI instead whenever a real `QDRANT_API_KEY` is already configured — it can't safely read/re-pass a credential (see `skills/my-setup-clauderunway/SKILL.md`'s API-key check); use `setup_project.py init`/`claude-runway-setup init` directly with `--qdrant-api-key` re-passed in that case. `merge_mcp_json` overwrites the toolkit-owned server blocks (`qdrant` among them) even against a `.mcp.json` this toolkit never generated. A BARE rerun with no flags instead regenerates every OTHER value in those blocks back to defaults, so re-passing your existing options matters here. **`EMBEDDING_MODEL` is preserved across a re-run**: `init` reuses the model already in the project's existing `.mcp.json` (written to the `qdrant` and `codebase-indexer` blocks), so a hand-customized model survives. Pass `--embedding-model <model>` only to change it deliberately — `init` then prints a WARNING, because vectors already indexed under the old model are incompatible (drop the collection in Qdrant and re-index; see README's `EMBEDDING_MODEL` bullets). **The `memory-bank` block's `EMBEDDING_MODEL` is separate**: `--embedding-model` never writes to it. A re-run keeps its existing value (template default if the block is absent), and `--memory-bank-embedding-model <model>` changes it, with a WARNING that the shared collection is locked to the model that created it, so `remember`/`recall` fail for every project until it is rebuilt. **The `qdrant` block's `COMPACT_EMBEDDING_MODEL` is separate too**, and also sticky: a re-run keeps it (carrying it over from an older `local-compress` block when there is one), because changing it orphans the compacts already stored.

> **If you installed via pipx/`uv tool install` and just upgraded claude-runway itself** (`pipx upgrade claude-runway` / `uv tool upgrade claude-runway`), re-run `init` (NOT `upgrade` — see below) for every project you'd previously configured — don't just assume the "safe to re-run" merge behavior above means an old config still works untouched. An earlier release moved `libs`/`tools`/`hooks`/`templates` out of site-packages into `<env>/src/{libs,tools,hooks,templates}` (and a later one added `mcpservers`, below it, for the five MCP server scripts): any `.mcp.json`/`.claude/settings.json` an OLDER install's `init` generated hardcodes absolute paths into the old site-packages location, which no longer exists after upgrading past this change — those entries would point at files that were never written there in the new layout, not files that merely moved. `upgrade` (a few paragraphs below) is NOT a substitute here: it only applies its own fixed, named list of config gaps (see `upgrade --dry-run` or `MIGRATIONS` in `libs/upgrade_lib.py` for the live list) — the only one that rewrites an MCP-server script path (`mcp-server-scripts-moved`, below) swaps `tools` for `mcpservers` in it and nothing else, so it would leave every stale site-packages path just as stale. This is a one-time gotcha for the specific upgrade that crosses that layout change, not a general re-run risk.

**Alternatively, use the `/my-setup-clauderunway` Claude Code skill** (see `skills/my-setup-clauderunway/SKILL.md`): install it once to `~/.claude/skills/`, export `CLAUDE_RUNWAY_DIR` in your shell profile pointing at this tools repo, then run `/my-setup-clauderunway` from any project you want to configure. It wraps `setup_project.py` with a guided question flow (the `CLAUDE.md` update in step 5 is done by `init` itself, so the skill and the CLI do it the same way). It is also the supported **init-and-upgrade** route for anyone who cannot install the `claude-runway-*` console scripts: on an already-configured project it offers `upgrade` (a `--dry-run` preview, then all pending migrations applied at once, since a skill cannot answer the CLI's per-migration prompts; run the CLI directly for per-migration control), and it works even when a real `QDRANT_API_KEY` is configured, because `upgrade` never takes or prints the key. It runs the checkout's own `tools/setup_project.py`, so it sees every migration in that checkout, unlike a stale `claude-runway-setup` install, which silently skips newer ones. Restart Claude Code afterwards and confirm with `doctor`.

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
# The CORE hooks/record_session_id.py hook is written either
# way -- it's base install, not local-compress-dependent, and only
# --skip-hooks (a full opt-out of touching settings.json at all) omits it:
python tools/setup_project.py init /path/to/target-repo --qdrant-only

# common options: --collection-name, --collection-description, --lmstudio-model,
# --lmstudio-url, --qdrant-url, --qdrant-api-key, --include-extensions, --exclude-dirs, --track-savings,
# --compact-collection, --skip-claude-md (leave the project's CLAUDE.md alone, see step 5)
python tools/setup_project.py init --help

# Install/update the product skills (my-compact/my-resume/my-savings/my-metrics/
# my-setup-clauderunway) into ~/.claude/skills/ -- combine with a target_repo to do
# both in one run, or pass alone (no target_repo) for skills-only mode:
python tools/setup_project.py init /path/to/target-repo --install-skills
python tools/setup_project.py init --install-skills   # skills only
```

**Already configured this project before and just want to pick up a new feature a recent release added, without re-running `init` and re-specifying every option you originally set?** Use `upgrade` instead: it compares this project's current `.mcp.json`/`.claude/settings.json` against a fixed list of specific, named config gaps this toolkit knows how to fill (e.g. the `memory-bank` server, the `codebase-indexer` block's `MEMORY_BANK_COLLECTION` key, `record_session_id.py`'s hooks, the `qdrant` server's move from the third-party console script to this repo's own), tells you about each one it finds still missing, and applies only the ones you approve — leaving everything else (including anything `init` would otherwise reset back to its own defaults/flags) untouched. Safe to run repeatedly: once a migration is applied, it no longer shows up as pending.

```bash
python tools/setup_project.py upgrade /path/to/target-repo             # interactive: prompts per pending migration
python tools/setup_project.py upgrade /path/to/target-repo --dry-run   # preview only, writes nothing
python tools/setup_project.py upgrade /path/to/target-repo --auto-yes  # apply every pending migration, no prompts (scripting/CI)
```

**`upgrade` also refreshes installed skills**. After the config migrations it compares each product skill you have ALREADY installed under `~/.claude/skills/` (`--skills-dir <path>` to override) with the shipped one, lists the ones that differ, and asks before rewriting each (`--auto-yes` approves all; `--dry-run` previews and writes nothing). This runs even when no config migration is pending. It never installs a skill you do not have (that stays `init --install-skills`), so a user with no skills installed is not forced to get any. Frontmatter rule: shipped skills have no `---` frontmatter block but installed ones usually do, so an installed copy's frontmatter is kept as-is and only the body is compared and replaced. A copy that differs from the shipped file only by its frontmatter is therefore reported as up to date, `init --install-skills` and `upgrade` agree on this, and a refresh never deletes your frontmatter. Comparison is byte-for-byte on the body, so a hand edit to the body counts as a difference and is overwritten when you approve. **"Shipped" means the skills of whichever copy of `setup_project.py` runs.** `claude-runway-setup upgrade` (a `uv tool`/`pipx` install) compares against the skills that install shipped when it was installed, not against a checkout's `skills/`, and says so with a line naming that location and the package version; reinstall to pick up skill changes made since. `/my-setup-clauderunway upgrade` runs the checkout's own copy, so it always compares against that checkout. With both present, the two can offer different refreshes, and the console script writes its own (possibly older) copy by design. When `CLAUDE_RUNWAY_DIR` points at a checkout, `claude-runway-setup upgrade` warns on each stale skill whose copy in that checkout differs, since it cannot tell which one is newer.

**`upgrade` does not touch the tools virtual environment.** It only rewrites `.mcp.json`/`.claude/settings.json` (and refreshes installed skills); it reads the venv's interpreter path to write into the config, but never installs or uninstalls a package. So an environment built before the move to this repo's own `qdrant` server may still have the third-party Qdrant MCP server package and its pinned `fastmcp` installed. Nothing launches or imports them any more, and leaving them does no harm. To drop them, run `uv pip uninstall --python <venv python> mcp-server-qdrant fastmcp` (or recreate the venv from `requirements.txt`), and `uv pip install --python <venv python> -r requirements.txt` afterwards to be sure everything this repo still needs is present.

**If you pass `--qdrant-api-key`**, the real key is written in plaintext into `.mcp.json` (never echoed back in `--dry-run`'s preview or the printed follow-up command, which are both redacted/placeholdered instead) — the script prints a warning reminding you NOT to commit `.mcp.json` as-is in that case, replacing the usual "commit it" instruction. There's no built-in mechanism here for keeping the key out of a committed `.mcp.json`; either gitignore `.mcp.json` for that project or manage the key through your own separate process.

The manual steps below still apply if you'd rather hand-edit (or need to understand exactly what the script does / verify its output):

- Copy `templates/mcp.json.template` to `.mcp.json` at that project's repo root.
- Replace `REPLACE-WITH-THIS-PROJECTS-NAME` (appears twice) with a unique collection name for that project.
- Keep `QDRANT_URL`, `COLLECTION_NAME` and `EMBEDDING_MODEL` identical everywhere they appear: `claude-runway-setup init` writes them aligned, but a hand edit must change every block that carries the value (`COLLECTION_NAME` is in the `qdrant` and `codebase-indexer` blocks; `EMBEDDING_MODEL` is also in `memory-bank`). A mismatched `EMBEDDING_MODEL` is usually caught and returned as an actionable error (different models produce different vector names or dimensions), but the check fails open when it can't reach Qdrant or the result is inconclusive, and then searches quietly return irrelevant results instead.
- Optionally set `COLLECTION_DESCRIPTION` to a short one-line description of what this collection contains — applied automatically to the collection (no manual `set_collection_description` call needed) at server startup and after `index_repo`/`sync_repo` write, so `list_collections` from a *different* project's session can surface it. Leave blank (`""`) for no static hint.
- Replace `/absolute/path/to/mcpservers/qdrant_mcp_server.py` and `/absolute/path/to/mcpservers/ingest_mcp_server.py` with the real paths from step 2. In the `qdrant` block, `CLAUDE_RUNWAY_LMSTUDIO_MODEL` (replace `REPLACE-WITH-MODEL` with the exact model id loaded in LM Studio, or `""` to auto-detect when only one is loaded) is used only by `compact_store`, which compresses the summary through the local model and stores it uncompressed when LM Studio is unavailable.
- Replace every `REPLACE-WITH-VENV-PYTHON` with the absolute venv Python path from step 3. All the MCP servers use the same venv, so it's the same path in each block, and each runs one of this repo's own scripts (the `qdrant` server is no longer a separate console script).
- Leave `QDRANT_API_KEY` (present in the `qdrant`, `codebase-indexer` and `memory-bank` `env` blocks) blank for an unauthenticated local Qdrant instance, or set it to your API key for an authenticated remote one — same variable name `qdrant_mcp_server.py`/`ingest_mcp_server.py`/`memory_bank_mcp_server.py` all read, so one value keeps every Qdrant-talking server in this file aligned. **If you set a real key, do not commit `.mcp.json` with it inlined** — there's no built-in mechanism here for loading it from an untracked source instead, so either keep `.mcp.json` itself out of version control for this project (add it to `.gitignore`) or manage the key through your own separate, untracked process.
- Commit `.mcp.json` to the repo — **unless** you just set a real `QDRANT_API_KEY` above, in which case committing it publishes that credential in plaintext to anyone with repo access (permanently, since git history retains it even after a later edit); see the caveat on that bullet.

This is what makes memory automatically project-scoped: opening Claude Code in a project loads that project's `.mcp.json`, which points at that project's collection. Opening a different project uses a different collection automatically — no manual switching.

**Controlling what gets ingested:** by default, `index_repo`/`sync_repo`/`ingest_to_qdrant.py` include a built-in list of common code/doc extensions, skip common noise folders (`.git`, `node_modules`, `venv`, `dist`, `build`, etc.), and additionally respect the project's own `.gitignore` if the optional `pathspec` package is installed (add it with `uv pip install pathspec` in the tools repo venv from step 3). To narrow this per project, set in `.mcp.json`'s `codebase-indexer` env block:

- `INDEX_INCLUDE_EXTENSIONS`: comma-separated extensions (e.g. `".py,.md"`) — overrides the built-in list entirely, so only these get ingested.
- `INDEX_EXCLUDE_DIRS`: comma-separated folder names (e.g. `"fixtures,generated"`) — adds to (not replaces) the built-in excludes.

Same options exist as `--include-ext`/`--exclude-dirs`/`--no-gitignore` flags on `ingest_to_qdrant.py` (which falls back to `INDEX_INCLUDE_EXTENSIONS`/`INDEX_EXCLUDE_DIRS` too, but only from the shell it runs in, never from `.mcp.json` — see step 6), and as `include_extensions`/`exclude_dirs`/`respect_gitignore` parameters on the `index_repo`/`sync_repo`/`preview_index` tools if you want to override the env defaults for a one-off run. Use `preview_index` first to confirm the filtering is doing what you expect before running a real index.

**How files are chunked:** by the embedding model's own token window, not by line count. The model reads only the first N tokens of any text it embeds (N comes from the model's tokenizer configuration; for the default `sentence-transformers/all-MiniLM-L6-v2` it is 128 or 256 depending on the FastEmbed version and model snapshot installed) and silently ignores the rest, so the indexer reads N from the loaded model, subtracts the special tokens the tokenizer adds, and cuts every chunk to fit. It walks the file as a stream of characters, so a minified file that is one line is chunked like any other, and ends each chunk where it can at a markdown heading, a safe code boundary, a newline or whitespace, in that order, and at the exact token budget when none exists. Changing `EMBEDDING_MODEL` therefore changes the chunk size too (and already requires a re-index). If the window cannot be read from the model, `index_repo`, `sync_repo`, `preview_index` and `ingest_to_qdrant.py` stop with an error naming the model before chunking, deleting or writing anything; they do not fall back to line-based chunking.

What the old knobs mean now: `overlap` (`--overlap`) is a number of **tokens** repeated between consecutive chunks (default 10), and must be smaller than the model's usable window. `chunk_lines` (`--chunk-lines`) no longer sets any size: it is still accepted so existing calls and command lines keep working, and it is ignored (a non-default value is reported as ignored in the result). Each chunk's metadata is `file_path`, `line_number` (the 1-based line of the file where the chunk's text begins) and `char_range` (`start-end`, character offsets into the file as decoded, end exclusive); points indexed before this change carry `line_range` instead until their collection is re-indexed, and everything that reads chunk metadata accepts both.

**Cost:** every file is indexed, with no per-file cap and no skip threshold, so the chunk count is several times what line-based chunking produced, and a minified bundle or a large data file can become thousands of chunks, each of which has to be embedded. The tools make that visible: `index_repo`/`sync_repo` report a running count while hashing, then progress per file, and name the files that produced the most chunks, `ingest_to_qdrant.py` prints each file's chunk count, and `preview_index` shows the same before you commit to a run. If a generated or data file you do not need searched is the cause, exclude it with `INDEX_EXCLUDE_DIRS` or `INDEX_INCLUDE_EXTENSIONS`.

**4b. If also using local-compress**, add this server to the same `.mcp.json`:

```json
"local-compress": {
  "command": "REPLACE-WITH-VENV-PYTHON",
  "type": "stdio",
  "args": ["/absolute/path/to/mcpservers/compress_mcp_server.py"],
  "env": {
    "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1",
    "CLAUDE_RUNWAY_LMSTUDIO_MODEL": "<exact model id loaded in LM Studio, or omit to auto-detect if only one model is loaded>"
  }
}
```

Replace `REPLACE-WITH-VENV-PYTHON` with the absolute venv Python path from step 3. If `compress_mcp_server.py` fails to start in Claude Code with `MCP error -32000: Connection closed`, it's almost always a missing dependency (`requests`/`trafilatura` were added later for `fetch_url` and are easy to miss). Run it directly using the venv Python to see the real `ModuleNotFoundError`:

```bash
/absolute/path/to/.venv/bin/python /absolute/path/to/mcpservers/compress_mcp_server.py
```

**4b-2. If also tracking savings** (the opt-in tracker, see [Savings tracker](savings-tracker.md)), add the `analytics` server too. It serves the savings and metrics tools (`savings_summary`, `savings_detail`, `savings_trend`, `get_metrics`, `record_metric`) and needs only the `mcp` package. `init --track-savings` writes it for you; by hand, copy the `analytics` block from `templates/mcp.json.template` and replace its path and `REPLACE-WITH-VENV-PYTHON` the same way. Skip it if you do not track savings: nothing else depends on it.

If also using the PostToolUse / PreToolUse hooks from `templates/settings.json.template`, merge that hooks block into the project's `.claude/settings.json` and replace `REPLACE-WITH-VENV-PYTHON` there too — the hook scripts need the same venv Python to find their installed packages.

**4c. If also using memory-bank** (`remember`/`recall`/`forget` — durable, cross-session lessons, separate from the per-project Qdrant memory above): `templates/mcp.json.template` already wires up its `memory-bank` block by default, and `tools/setup_project.py init` (step 4 above) already writes it for you — no separate opt-in flag is needed the way `--qdrant-only` opts *out* of local-compress. See [Memory bank](memory-bank.md#setup) for the recommended `--memory-bank-collection`/`--memory-bank-id` flags and the manual `.mcp.json`-editing fallback if you're hand-editing instead of using the script. Unlike local-compress, this needs no dependencies beyond what step 3 (or the "Qdrant memory only" reduced install above) already installs — `mcpservers/memory_bank_mcp_server.py` only imports `mcp`, `fastembed` and `qdrant-client` (through this repo's own `libs/qdrant_embeddings.py`), all already required for the Qdrant memory piece.

**If you plan to run `/my-gh-autowork` (or any skill that autonomously drives `git`/`gh` from a subagent) on this checkout**, set up two separate layers first, because an opaque denial of the orchestrator's very first `Agent` call is the usual way this goes wrong.

1. **The auto mode classifier** (the one that denies the `Agent` call). The denial seen in practice is its built-in "Merge Without Review" rule: the subagent merges with `gh pr merge --squash`, and Copilot's automated review is not human approval to the classifier. It runs after the permission rules, so no `Bash(...)` rule can clear it. Add an `autoMode.allow` entry to your **user-level** `~/.claude/settings.json` (Claude Code ignores `autoMode` in `.claude/settings.json` and `.claude/settings.local.json`), with the literal `"$defaults"` first so the built-in rules stay (an `allow` list without it replaces them):

   ```json
   {
     "autoMode": {
       "allow": [
         "$defaults",
         "The user has authorized /my-gh-autowork to spawn autowork-ticket subagents that work and squash-merge pull requests on Donelle/claude-runway with gh pr merge. Merging is gated by the repository's Copilot review_on_push ruleset and required status checks, which stand in for human approval; this is deliberate, not a review bypass."
       ]
     }
   }
   ```

   Run `claude auto-mode config` to confirm what is in effect (`/auto-mode-setup` drafts `environment` entries). For a one-off, stating the intent in a message and retrying works. Claude cannot make this edit itself: the built-in Self-Modification block stops an agent from adding or widening permissions in its own config, so you make the change.
2. **The `Bash(...)` permission rules** in `permissions.allow` (see `templates/settings.json.template`'s `_permissions_note` and README's Known limitations bullet on the auto mode classifier). These decide whether the subagent's own `git`/`gh`/`.venv` commands run without a prompt; they are not what clears the classifier. Add SPECIFIC, narrow, per-subcommand rules (`Bash(git fetch *)`/`Bash(git push *)`/`Bash(gh pr *)`/`Bash(gh api *)`, extended as needed) to `.claude/settings.json` to keep the rule to this checkout, or to `~/.claude/settings.json` to cover every checkout on the machine. Do NOT reach for a blanket `Bash(git:*)`/`Bash(gh:*)`/`Bash(uv:*)` wildcard, which adds entire additional command families at once (via git aliases and `uv run`, respectively) rather than just broader convenience. **Narrow is not the same as safe, though**: autonomous development inherently needs code-execution authority, and even these narrow rules still permit it through legitimate flags (`git fetch --upload-pack=<cmd>`, `.venv/bin/python -c '<code>'`, etc. — see `templates/settings.json.template`'s `_permissions_note` for the full account); the actual containment is only running this in a checkout you trust, not the permission syntax.

This is unrelated to the hooks/MCP config above and worth setting up before your first autonomous run, not discovering it after one fails.

**5. Add the relevant section(s) from `templates/CLAUDE.md.template` to each project's `CLAUDE.md`** — only include a section for a tool that's actually configured in that project's `.mcp.json`.

`setup_project.py init`/`claude-runway-setup init` (step 4) does this for you, reading the template from its own install, so it works the same from a clone and from a pipx/`uv tool install`. It creates `CLAUDE.md` if the project has none. Otherwise it appends, after a blank line, the section for each server it writes that the file has no heading for (Vector memory and Durable memory always; Local compression for a full setup; Analytics with the savings tracker; Local review with a `codebase-reviewer` block), and removes the section of a toolkit server that is not configured (Local compression after `--qdrant-only`, Analytics after a run without the savings tracker), from its heading to the next level-1 or level-2 heading. Nothing else in the file changes: a section that is already there is left as it is, even if you edited it, and a run with nothing to add or remove does not rewrite the file. `--dry-run` names the sections it would add or remove; `--skip-claude-md` leaves `CLAUDE.md` alone. The template's "Setup and diagnostics" section is not managed (it depends on which skills you installed); copy it yourself if you want it. A `CLAUDE.md` that is a symlink is written through to the file it points at.

**6. Initial index** (first time only, per project):

```bash
# Run from inside the tools repo with the venv activated, or use the venv Python directly:
source ~/tools/claude-runway/.venv/bin/activate
python tools/ingest_to_qdrant.py --repo-path /path/to/project --collection <project-collection-name> --embedding-model <same EMBEDDING_MODEL as .mcp.json> --dry-run
# check the preview, then run for real:
python tools/ingest_to_qdrant.py --repo-path /path/to/project --collection <project-collection-name> --embedding-model <same EMBEDDING_MODEL as .mcp.json>
# optional: add --write-manifest to the real run (see below)
```

**`--write-manifest` (optional, opt-in).** Without it the CLI writes no sync manifest, so the first `sync_repo` after the index finds none and re-chunks and re-embeds every file a second time, which is the expensive pass on a big repo. Add the flag when `--collection` (on `--qdrant-url`) is the collection `sync_repo` will use for this checkout, meaning the `codebase-indexer` block's `COLLECTION_NAME` on its `QDRANT_URL` in `.mcp.json`. It is not automatic because the manifest is one file per checkout that does not record which collection it describes, and the CLI cannot read `.mcp.json`: a manifest written after indexing some other collection would make a later `sync_repo` skip files it should index. The CLI writes the baseline only when the run leaves the collection holding exactly what it indexed (`--reset`, or a collection that was absent or empty), recording each file's hash from before it was chunked, so a file edited during the run is simply re-indexed by the next `sync_repo`. It writes nothing on `--dry-run` or after a failed run. It removes any old manifest before anything is deleted or stored (with `--reset`, before the wipe, and the reset does not start if the manifest cannot be removed), so a run that fails partway leaves none behind. Into a populated collection without `--reset` it leaves the manifest alone and prints why. With the flag it holds the checkout's manifest lock for the whole run (see the next paragraph), so it stops with an error, before deleting or storing anything, if a `sync_repo` or `index_repo` on the same checkout is running. Run later `sync_repo` calls with the same scope, extensions, excluded directories and gitignore setting as the CLI run.

**One manifest user at a time per checkout.** `sync_repo`, an `index_repo` into this project's own collection (the only one whose manifest it may write) and the CLI with `--write-manifest` each hold a lock on `.qdrant_index_manifest.json.lock` (next to the manifest) from before they read the manifest or count the collection to their final save. The count matters as much as the manifest: whether an index run may write a baseline depends on the collection being absent or empty, which no other run may change between that check and the save. Without it, two overlapping runs (two Claude Code sessions on one checkout each syncing at start, say) could leave the collection holding one version of a file's chunks while the manifest records another version's hash, and every later `sync_repo` would skip that file as unchanged. A second run is refused at once with "another index or sync run is in progress for this checkout" rather than waiting behind a long embed; run it again once the first has finished. The lock is an OS file lock, so a crashed run never leaves the checkout locked, and the lock file stays in place, empty, between runs: gitignore it along with the manifest. Separate worktrees have separate checkouts, manifests and locks. The manifest itself is written atomically (temp file, then rename), so a reader never sees it half-written.

The CLI can't read `.mcp.json`. For `--qdrant-url`, `--qdrant-api-key`, `--embedding-model`, `--include-ext` and `--exclude-dirs`, it falls back to the `QDRANT_URL`/`QDRANT_API_KEY`/`EMBEDDING_MODEL`/`INDEX_INCLUDE_EXTENSIONS`/`INDEX_EXCLUDE_DIRS` exported in your shell. If none is exported, it uses the built-in default (`http://localhost:6333`, no key, `sentence-transformers/all-MiniLM-L6-v2`, the built-in extension list). A flag always wins. The exported `QDRANT_API_KEY` is sent only to the exported `QDRANT_URL` (or `http://localhost:6333` when that is unset): a `--qdrant-url` naming any other host gets no key unless you also pass `--qdrant-api-key`. If the project's `.mcp.json` sets any of these to something else, pass the flag or export the variable, or the initial index won't match what `sync_repo` later expects. A wrong embedding model surfaces later as a model-mismatch error. A different include/exclude filter leaves chunks for files `sync_repo` doesn't track, so `sync_repo` can never remove them.

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
pipx install --force --python python3.12 git+https://github.com/Donelle/claude-runway.git

# uv tool workflow
uv tool install --reinstall --python 3.12 git+https://github.com/Donelle/claude-runway.git
```

The package version is derived from commit history (see the "Which version/commit is installed" note in the pipx/`uv tool` callout above), so `claude-runway-doctor --version` confirms what you now have. `pipx upgrade claude-runway` / `uv tool upgrade claude-runway` also work, but reinstalling from the URL is the form that always re-resolves the newest `main` commit.

**2. Pick `init` or `upgrade` per configured project.** Neither runs automatically; the generated `.mcp.json`/`.claude/settings.json` keep whatever the older version wrote until you re-run one of them:

- **`upgrade`** (step 4 above) is the default after a normal update. It applies only the specific, named config gaps a newer release added (for example the `memory-bank` server, the `record_session_id.py` hooks, the `qdrant` server's move to this repo's own `mcpservers/qdrant_mcp_server.py`, which also brings the conversation-compact tools with it, or `mcp-server-scripts-moved`, which rewrites each server's script path from `tools/<server>.py` to `mcpservers/<server>.py`), asks before each one, and leaves every option you originally set alone. It then also refreshes any already-installed skill that went stale (asking per skill). Preview with `--dry-run` first.
- **`init`** (re-passing the flags you originally used) is required when existing paths in the generated config are no longer valid, or when you want every path regenerated for this machine. In particular, after a pipx/`uv tool` upgrade that crosses the layout change (the move of `libs`/`tools`/`hooks`/`templates` out of site-packages into `<env>/src/`), `upgrade` will NOT repair the stale absolute paths, so re-run `init`. See the callout under step 4 and README's Known limitations for the full explanation.
- If you installed the product skills, `upgrade` (above) now refreshes the stale ones for you. To also install skills you do not have yet, run `claude-runway-setup init --install-skills` (or `python tools/setup_project.py init --install-skills` from a clone).

- **A project already indexed with an older release has to be re-indexed once.** Chunks are cut differently now and carry different position metadata, and the sync manifest (`.qdrant_index_manifest.json`) only records file content, so it cannot tell. The manifest is versioned: `sync_repo` ignores one in the old format, so the next `sync_repo` rebuilds the whole index with no action from you, and `upgrade` offers to delete the stale old-format manifest and says so (it changes no config and embeds nothing). That first `sync_repo` is one long call that re-embeds every file. Before it deletes anything it reports any points in the collection that are not indexer chunks (for example notes stored with `qdrant-store` that carry a `file_path` of a re-indexed file, which are deleted with that file's old chunks), and afterwards it removes the chunks of files that no longer exist on disk. `index_repo` writes the manifest as well when it starts from a clean slate (`reset=true`, or a collection that was absent or empty), recording each file's hash from before it was chunked, so a `sync_repo` after it re-embeds only what changed since; a `force=true` run into a collection that already holds points leaves the manifest alone and says so, because it only adds chunks. The manifest is one file per checkout and does not record which collection it describes, so `index_repo` only writes or removes it for this project's own collection (the configured `COLLECTION_NAME` on the configured `QDRANT_URL`); indexing the checkout into any other collection leaves it alone. `ingest_to_qdrant.py` does the same under its opt-in `--write-manifest` flag, with the same clean-start rules (see step 6 of the setup above), and writes none without it. Before it stores anything, `index_repo` removes the old manifest (for `reset=true`, before the wipe, and the reset does not start if the manifest cannot be removed), so a run that fails partway leaves none behind. The scope of that first rebuild (`scope`, `include_extensions`, `exclude_dirs`, `respect_gitignore`) does not have to match how you normally index. A file that is still on disk but outside it keeps its old chunks and is recorded in the manifest as kept, in a separate list from the file hashes (the manifest format is unchanged, so no further rebuild is forced). Every later `sync_repo` checks that list: if a kept file has been deleted, its chunks are removed (memory-bank memories and conversation compacts are left alone, as with any removed file, but a note stored with `qdrant-store` that carries the deleted file's `file_path` is deleted with it); if a scan now includes it, it is re-indexed and tracked like any other file; if its existence cannot be checked (for example a permission error on a parent directory), it stays on the list for the next sync. A sync that is not a rebuild never scans the whole collection for such files.

Restart Claude Code in each project afterward so it reloads the MCP servers and hooks.

## Uninstalling

There is no single uninstall command. A full removal touches the places below; each is independent, so you can stop after any of them.

**1. Remove the toolkit from each configured project.** In every project you ran `init`/`upgrade` against:

- **Note any custom data locations before deleting any of the config below.** `CLAUDE_RUNWAY_SAVINGS_DB`, `CLAUDE_RUNWAY_METRICS_DB`, `CLAUDE_RUNWAY_MEMORY_EVENTS_DB`, `CLAUDE_RUNWAY_CACHE_DB`, and `FASTEMBED_CACHE_PATH` can each move a file out of `~/.claude/claude-runway/`. If you set any of them (in `.mcp.json`, `.claude/settings.json`, or your shell profile), write down the paths first so step 4 can remove them too.
- In `.mcp.json`, delete the server blocks this toolkit added: `qdrant`, `codebase-indexer`, `memory-bank`, and (if you used them) `local-compress` and `analytics`. Leave any other servers alone. If the file then has no servers left, delete it.
- In `.claude/settings.json`, delete the hook entries whose commands point at this toolkit's scripts: `record_session_id.py`, `compress_output.py` (or `compress_bash_output.py`, its earlier name, if you never ran `upgrade`), `redirect_webfetch_to_fetch_url.py`, and `session_end_savings.py`.
- Remove the claude-runway sections `init` (or you) added to the project's `CLAUDE.md` (from `templates/CLAUDE.md.template`).
- Optionally delete the project's Qdrant collections (via the Qdrant dashboard at <http://localhost:6333/dashboard> or its API) if you want the stored data gone too. These are separate collections and each needs its own deletion: the code index (`COLLECTION_NAME` in the `codebase-indexer` block), the project's conversation compacts (named `<COMPACT_COLLECTION>-<project>`, `conversation-compacts-...` by default; find it with `list_collections`). Before deleting a compact collection, check that it holds only this project's compacts: project names that differ only in punctuation or whitespace (e.g. `my.project` and `my-project`) share one collection, so deleting it wholesale can erase another project's history. If it contains other projects' points, delete only the points whose `project` payload matches this project instead. The `memory-bank` collection is different: it is shared by every project on this Qdrant instance (see [Memory bank](memory-bank.md)), so deleting it erases other projects' durable memories too. Leave it alone unless you are removing claude-runway from every project.

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

