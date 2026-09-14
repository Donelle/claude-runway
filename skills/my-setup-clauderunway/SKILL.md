# Skill: my-setup-clauderunway

Interactively configure any project for ClaudeRunway. Generates `.mcp.json`,
updates `.claude/settings.json` hooks, and adds the correct usage guidance to
`CLAUDE.md` — all in one pass.

## Prerequisites

`CLAUDE_RUNWAY_DIR` must be exported in your shell profile, pointing to the
claude-runway tools repo:

```bash
# macOS / Linux (~/.bashrc or ~/.zshrc)
export CLAUDE_RUNWAY_DIR="$HOME/tools/claude-runway"

# Windows (PowerShell profile)
$env:CLAUDE_RUNWAY_DIR = "C:\tools\claude-runway"
```

Restart Claude Code after setting it so the new export is visible.

## Installation

```bash
cp -r skills/my-setup-clauderunway ~/.claude/skills/
```

## Usage

Run `/my-setup-clauderunway` from the root of the project you want to configure. It
configures that project (the current working directory).

## Steps

### 1. Validate the tools repo

Read the `CLAUDE_RUNWAY_DIR` environment variable from the shell. If it is not
set or is empty, stop immediately and tell the user:

> `CLAUDE_RUNWAY_DIR` is not set. Add `export CLAUDE_RUNWAY_DIR="/path/to/claude-runway"` to your shell profile (the same place you launch `claude` from), then restart Claude Code and try again.

Check that `$CLAUDE_RUNWAY_DIR/tools/setup_project.py` exists. If the file is
missing the path is wrong — stop and tell the user.

Determine the venv Python path (exact platform rules — check both, use whichever exists):
- Windows: `$CLAUDE_RUNWAY_DIR\.venv\Scripts\python.exe`
- macOS/Linux: `$CLAUDE_RUNWAY_DIR/.venv/bin/python`

If neither exists, stop and tell the user:

> The claude-runway virtual environment is missing. Open a terminal in the tools repo and run:
> ```
> uv venv --python 3.12
> uv pip install -r requirements.txt --index-url https://pypi.org/simple
> ```
> Then try `/my-setup-clauderunway` again.

### 2. Detect current project state

Note the **current working directory** — this is the target project path.

Derive the **slug default**: take the directory's leaf name, lowercase it, replace
any run of non-alphanumeric characters with a single hyphen, strip leading/trailing
hyphens, and fall back to `"project"` if the result is empty (e.g. `My.Cool.App` →
`my-cool-app`; `___` → `project`). This matches `setup_project_lib.default_collection_name()`.

Check whether `.mcp.json` already exists in the current directory. If it does,
read it to see which claude-runway servers are currently configured (`qdrant`,
`codebase-indexer`, `local-compress`). Note this for the summary at the end.

Also extract the following values from the existing config — these would silently
revert to defaults on re-run without preservation:
- `COLLECTION_NAME` from `qdrant.env` — use as the collection name default in Q2 when
  present, falling back to the slug default only for unconfigured projects
- `QDRANT_URL` from `qdrant.env` (default: `http://localhost:6333`)
- `QDRANT_API_KEY` — check only whether it is non-empty in **any** of the three owned
  server env blocks (`qdrant.env`, `codebase-indexer.env`, `local-compress.env`); do
  NOT read the value itself into context. All three blocks are checked because
  `build_mcp_servers()` sets the same key in all of them, and legacy or hand-edited
  configs may only have it in one.

**If `QDRANT_API_KEY` is non-empty in any owned block, stop immediately** and tell the user:

> This project has `QDRANT_API_KEY` configured. The skill cannot carry credentials
> through safely — reading the key and passing it in a shell command would expose it
> in context and the tool-call transcript. Run `setup_project.py` directly instead
> (from inside the ClaudeRunway tools repo):
> ```
> <venv-python> <CLAUDE_RUNWAY_DIR>/tools/setup_project.py init <cwd> --qdrant-api-key <your-key> [other options]
> ```
> See `<CLAUDE_RUNWAY_DIR>/tools/setup_project.py init --help` for the full option list.
>
> (Use the absolute paths validated in step 1 — `<venv-python>` and `<CLAUDE_RUNWAY_DIR>` — when running this command.)

- `COLLECTION_DESCRIPTION` from `codebase-indexer.env` (default: empty)
- `INDEX_INCLUDE_EXTENSIONS` from `codebase-indexer.env` (default: empty)
- `INDEX_EXCLUDE_DIRS` from `codebase-indexer.env` (default: empty)
- `CLAUDE_RUNWAY_LMSTUDIO_URL` from `local-compress.env` (default: `http://localhost:1234/v1`)
- `CLAUDE_RUNWAY_LMSTUDIO_MODEL` from `local-compress.env` (default: empty — auto-detect)
- `CLAUDE_RUNWAY_TRACK_SAVINGS` from `local-compress.env` (default: off)
- `CLAUDE_RUNWAY_SAVINGS_DB` from `local-compress.env` (default: empty)
- `COMPACT_COLLECTION` from `local-compress.env` — use as the "Keep current" value in Q5 when
  present. Note this is a DIFFERENT env var than `COLLECTION_NAME` above: it names the
  collection the `my-compact`/`my-resume` skills store conversation compacts under, not the
  codebase-memory collection.

**Also detect a prior qdrant-only round-trip**: if `.mcp.json` already exists and has a
`qdrant` and/or `codebase-indexer` server configured, but has **no** `local-compress` server
at all (not just an empty one — the key itself absent), note this as
`previously-qdrant-only = true`. `--qdrant-only` deliberately deletes the entire
`local-compress` block (see `setup_project_lib.build_mcp_servers`'s `include_compress=False`
path) — including any `COMPACT_COLLECTION` it held — so this state is indistinguishable from
a genuinely brand-new project by `COMPACT_COLLECTION`'s absence alone. This matters for Q5 in
step 4 below: without this signal, re-enabling full setup on a project that was full →
qdrant-only at some point silently computes a fresh `<collection-name>-conversation` default
as if no prior compact history could possibly exist, when it might (found in Copilot review
on PR #130). Only relevant if Q1 (step 3) ends up "Full setup" — a project going TO
qdrant-only, or staying qdrant-only, never reaches Q5 at all (step 4 is skipped entirely).

Check whether `CLAUDE.md` exists in the current directory and, if so, read it.

### 3. Ask configuration questions — scope and collection name

Use `AskUserQuestion` with two questions:

**Q1 — Setup scope**
Header: "Setup scope"
- "Full setup (Recommended)" — Qdrant memory + LM Studio compression + PostToolUse/PreToolUse/SessionEnd hooks
- "Qdrant-only" — no LM Studio server, no hooks (good for projects where LM Studio isn't in use or you want to add it later)

**Q2 — Collection name**
Header: "Collection"
- If `.mcp.json` already has a `COLLECTION_NAME`: show `"Keep current"` as the first option (description: the existing collection name)
- Otherwise: show `"Use default"` as the first option (description: the slug default you derived)
- "Specify" — always present as the second option (description: "Enter a custom collection name"), triggers a follow-up

If the user picks "Specify", ask them to type the exact collection name they
want (use `AskUserQuestion` with a single text-entry option or just read their
next message). Use whatever they provide for the remainder of the skill.

### 4. Ask configuration questions — LM Studio options (full setup only)

Skip this entire step if "Qdrant-only" was selected in step 3.

Use `AskUserQuestion` with three questions:

**Q3 — LM Studio model**
Header: "Model"
- "Auto-detect" — (description: "Only works if exactly one model is loaded in LM Studio") always present; omits `--lmstudio-model` in step 5
- If `CLAUDE_RUNWAY_LMSTUDIO_MODEL` is set: also show `"Keep current"` (description: the configured model name) as the **first/recommended** option, before "Auto-detect"
- "Specify model" — (description: "Enter the exact model ID from LM Studio") triggers a follow-up

If the user picks "Keep current", use the existing model value as `--lmstudio-model` in step 5.
If the user picks "Auto-detect", omit `--lmstudio-model` entirely.
If the user picks "Specify model", ask for the model ID and use it as `--lmstudio-model`.

**Q4 — Savings tracker**
Header: "Savings"
- If `CLAUDE_RUNWAY_TRACK_SAVINGS` is currently on — trim whitespace and fold to lowercase before checking; on values are `"1"`, `"true"`, `"yes"`: show `"Enabled"` first (description: "Keep savings tracking on"); show `"Disabled"` second
- Otherwise: show `"Disabled"` first (description: "No token savings tracking"); show `"Enabled"` second (description: "Set CLAUDE_RUNWAY_TRACK_SAVINGS=1; also requires a shell export, see step 7")

**If `previously-qdrant-only` is true** (detected in step 2), print this note **before** asking
Q5, so the user sees it while the choice is still live rather than after already picking
"Use default":

> This project previously had LM Studio compression removed (`--qdrant-only`), which also
> deletes any compact-collection name it had. If it was ever fully configured before that
> with a custom or auto-generated compact-collection name, this skill can't recover it —
> Q5's "Use default" below will start a fresh `<collection-name>-conversation`, which won't
> match any conversation history stored under the old name. If you remember the old name,
> pick "Specify" and enter it instead.

**Q5 — Compact collection name**
Header: "Compacts"
The default here is derived from Q2's answer (the resolved collection name), so ask this
after Q2 has been answered — e.g. show it in the same `AskUserQuestion` call as Q3/Q4,
never before step 3 completes.
- If `COMPACT_COLLECTION` already has a value: show `"Keep current"` as the first option (description: the existing value)
- Otherwise: show `"Use default"` as the first option (description: `<collection-name>-conversation`, using Q2's resolved collection name — if `previously-qdrant-only` is true, append " (can't recover a prior name — see note above)" to this description)
- "Specify" — always present as the second option (description: "Enter a custom compact-collection name"), triggers a follow-up

If the user picks "Specify", ask them to type the exact name they want. Use whatever they
provide (or the resolved default/current value) as `--compact-collection` in step 5.

### 5. Build and run setup_project.py

Construct the command from all collected inputs. Start with:

```
<venv-python> <CLAUDE_RUNWAY_DIR>/tools/setup_project.py init <cwd>
  --collection-name <name>
  [--qdrant-only]                    # only if qdrant-only was selected
  [--lmstudio-model <model>]         # only if the user gave a specific model name (omit for auto-detect)
  [--track-savings]                  # only if savings tracker was enabled
  [--compact-collection <name>]      # only for full setup -- Q5's resolved value
```

Then, for each value extracted from the existing `.mcp.json` in step 2 that is
non-default, add the corresponding flag so it is not silently reset:
- `QDRANT_URL` ≠ `http://localhost:6333` → add `--qdrant-url <value>`
- `COLLECTION_DESCRIPTION` ≠ empty → add `--collection-description <value>`
- `INDEX_INCLUDE_EXTENSIONS` ≠ empty → add `--include-extensions <value>`
- `INDEX_EXCLUDE_DIRS` ≠ empty → add `--exclude-dirs <value>`
- `CLAUDE_RUNWAY_LMSTUDIO_URL` ≠ `http://localhost:1234/v1` → add `--lmstudio-url <value>`
- `CLAUDE_RUNWAY_SAVINGS_DB` ≠ empty → add `--savings-db <value>`

(Q3/Q4's answers already handle `--lmstudio-model` and `--track-savings`; skip those here.)

`--compact-collection` is Q5's resolved value ("Keep current"'s existing value, "Use
default"'s `<collection-name>-conversation`, or "Specify"'s typed name) — always pass it
explicitly for a full setup, the same way `--collection-name` is always passed rather than
only added when non-default. Omit this flag entirely for `--qdrant-only` (Q5 was skipped
in step 4, since `local-compress` — and therefore `COMPACT_COLLECTION` — isn't configured
at all in that mode).

Use **single quotes** around every dynamic argument on **both platforms** — PowerShell
single-quotes and POSIX single-quotes both prevent expansion of `$`, `$()`, and
backticks, unlike double quotes. This covers: `<venv-python>`, the script path,
`<cwd>`, every `--flag <value>` argument (collection name, model name, and all
values carried through from `.mcp.json`).

If any value contains a literal `'`, escape it before quoting:
- macOS/Linux: end the outer quote, escape the apostrophe, reopen — e.g.
  `'O'\''Brien'` to represent `O'Brien`
- PowerShell: double the apostrophe inside single quotes — e.g. `'O''Brien'`

On **PowerShell**, prefix the entire command with `&` (the call operator):
```
& '<venv-python>' '<script-path>' init '<cwd>' --collection-name '<name>' ...
```
Without `&`, a single-quoted executable path is parsed as a string expression
rather than a command and the invocation silently does nothing.

Show the full command to the user before running it.

Run it using Bash (macOS/Linux) or PowerShell (Windows). Capture both stdout and
stderr.

If the command exits non-zero, show the full output and stop — do not proceed to
CLAUDE.md.

If it succeeds, extract the "Run the initial index" command line that
`setup_project.py` prints at the end (the `ingest_to_qdrant.py ...` line). You
will include this in the final summary.

**Note on Windows:** `setup_project.py` formats this command with cmd.exe quoting
(`subprocess.list2cmdline`). Run it in **Command Prompt** (`cmd.exe`), not
PowerShell — cmd.exe quoting uses double quotes which PowerShell would expand.

### 6. Create or update CLAUDE.md

Read the template at `$CLAUDE_RUNWAY_DIR/templates/CLAUDE.md.template`.

From that file, extract two sections as verbatim text:
- **Qdrant section**: everything from the line `## Vector memory (Qdrant MCP)` up to (but not including) the blank line that immediately precedes the next `##` heading.
- **LM Studio section**: everything from the line `## Local compression (LM Studio MCP)` to the end of the file (or the blank line before the next `##` heading, if any).

Do **not** include the HTML comment block at the top of the template — that is a
template note, not guidance for Claude.

**If `CLAUDE.md` does not exist in the target project:**

Create it with:
```
<Qdrant section>

<LM Studio section>   # only if full setup (not qdrant-only)
```

**If `CLAUDE.md` already exists:**

Check whether it already contains `## Vector memory (Qdrant MCP)`.
- If missing: append the Qdrant section (preceded by a blank line).

Check whether it already contains `## Local compression (LM Studio MCP)`.
- If this is a **full setup** and the section is missing: append it (preceded by a blank line).
- If this is **qdrant-only** and the section IS present: remove it. The section spans
  from the `## Local compression (LM Studio MCP)` heading line through to (but not
  including) the next `##` heading line, or end of file if there is none. Remove the
  blank line immediately preceding the heading as well. This keeps CLAUDE.md consistent
  with `.mcp.json` — leaving LM Studio guidance in place when `local-compress` is gone
  would instruct Claude to call tools that are no longer connected.

If both sections were already in the desired state (present for full setup, absent for
qdrant-only), note "CLAUDE.md already up to date — no changes made."

Never modify or remove any content already in CLAUDE.md other than the LM Studio section
removal described above.

### 7. Report results and next steps

Print a concise summary:

```
Done. Configured <target-project-path>:

  ✓ .mcp.json              — qdrant, codebase-indexer[, local-compress]
  ✓ .claude/settings.json  — PostToolUse, PreToolUse, SessionEnd hooks   [only if full setup]
  ✓ CLAUDE.md              — <what was added or "already up to date">

Next: run the initial index (from inside the tools repo, venv activated):
  <exact ingest_to_qdrant.py command from setup_project.py output>     # with --dry-run to preview
  <same command without --dry-run>                                     # to actually index
```

**Shell export reminders** — skip this block entirely if qdrant-only was selected
(hooks are not installed, so there is nothing to export). This mirrors
`setup_project.py`'s own behavior (`reminders = [] if qdrant_only`).

Otherwise, print this block whenever ANY of the following apply (these settings must
also be in the shell profile that launches `claude`, since hook scripts inherit the
shell environment and have no env block of their own).

Use the platform detected in step 1 to render the correct syntax:

- macOS/Linux (`export` syntax):
  ```bash
  export CLAUDE_RUNWAY_TRACK_SAVINGS=1           # if savings tracker was enabled
  export CLAUDE_RUNWAY_LMSTUDIO_MODEL='<model>'  # if a specific model was given
  export CLAUDE_RUNWAY_LMSTUDIO_URL='<url>'      # if a non-default LM Studio URL was used
  export CLAUDE_RUNWAY_SAVINGS_DB='<path>'       # if a non-default savings DB path was used
  ```

- Windows (PowerShell `$env:` syntax):
  ```powershell
  $env:CLAUDE_RUNWAY_TRACK_SAVINGS = '1'
  $env:CLAUDE_RUNWAY_LMSTUDIO_MODEL = '<model>'
  $env:CLAUDE_RUNWAY_LMSTUDIO_URL = '<url>'
  $env:CLAUDE_RUNWAY_SAVINGS_DB = '<path>'
  ```

Only include the lines that actually apply. If any value contains a literal `'`,
escape it (bash: `'\''`; PowerShell: `''`) before embedding it in the block.
If none apply, skip the block entirely.

**Shell unset reminders** — the dual-place requirement works in both directions. When
a setting is being REMOVED from `.mcp.json`, the matching shell-profile export must
also be removed or the hooks will keep using it regardless. Print an unset reminder
whenever ANY of the following transitions apply:

- User switched from a **pinned model to Auto-detect**: tell the user to remove
  `CLAUDE_RUNWAY_LMSTUDIO_MODEL` from their shell profile (the hooks will otherwise
  keep using the old model, and may fail if it is no longer loaded in LM Studio):
  - macOS/Linux: `unset CLAUDE_RUNWAY_LMSTUDIO_MODEL` (and remove the `export` line from the profile)
  - Windows: `Remove-Item Env:CLAUDE_RUNWAY_LMSTUDIO_MODEL` (and remove the `$env:` line)

- User switched savings tracker from **Enabled to Disabled**: tell the user to remove
  `CLAUDE_RUNWAY_TRACK_SAVINGS` from their shell profile (otherwise the hooks still
  see it as enabled even though the MCP server now has it cleared):
  - macOS/Linux: `unset CLAUDE_RUNWAY_TRACK_SAVINGS`
  - Windows: `Remove-Item Env:CLAUDE_RUNWAY_TRACK_SAVINGS`
