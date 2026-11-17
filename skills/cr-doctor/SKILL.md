---
name: cr-doctor
description: "Run claude-runway's doctor check against this project (dual-config env values, stale hook, reviewers file) and say plainly whether it passed."
---

# Skill: cr-doctor

Run claude-runway's `doctor` check against the current project and say plainly whether it passed. `doctor` diffs the project's `.mcp.json` dual-config `CLAUDE_RUNWAY_*` values against the live environment, flags a stale `compress_bash_output.py` hook, an unusable `codebase-reviewer` reviewers file, and the PR-round capture opt-in set without its hook. It exits 0 when nothing is wrong and 1 on any problem.

Running it from inside the session is more faithful than a separate terminal: `doctor` reads `os.environ`, and this skill's Bash call inherits Claude Code's own environment, which is exactly what the hooks see.

## Prerequisites

`CLAUDE_RUNWAY_DIR` must be exported in your shell profile (the same place you launch `claude` from), pointing to the claude-runway tools repo. See `/cr-setup`.

## Installation

```bash
cp -r skills/cr-doctor ~/.claude/skills/
```

Or `python tools/setup_project.py init --install-skills` from the tools repo, which installs every product skill.

## Usage

Run `/cr-doctor` from the root of a configured project. It checks that project (the current working directory). It takes no arguments.

## Steps

### 1. Validate the tools repo

Read the `CLAUDE_RUNWAY_DIR` environment variable from the shell. If it is not set or is empty, stop immediately and tell the user:

> `CLAUDE_RUNWAY_DIR` is not set. Add `export CLAUDE_RUNWAY_DIR="/path/to/claude-runway"` to your shell profile (the same place you launch `claude` from), then restart Claude Code and try again.

Check that `$CLAUDE_RUNWAY_DIR/tools/doctor.py` exists. If the file is missing the path is wrong: stop and tell the user.

Determine the venv Python path (check both, use whichever exists):
- Windows: `$CLAUDE_RUNWAY_DIR\.venv\Scripts\python.exe`
- macOS/Linux: `$CLAUDE_RUNWAY_DIR/.venv/bin/python`

If neither exists, stop and tell the user:

> The claude-runway virtual environment is missing. Open a terminal in the tools repo and run:
> ```
> uv venv --python 3.12
> uv pip install -r requirements.txt --index-url https://pypi.org/simple
> ```
> Then try `/cr-doctor` again.

### 2. Run doctor

The target is the **current working directory**. Use **single quotes** around every dynamic argument on both platforms, so `$`, `$()` and backticks are not expanded. Escape a literal `'` as `'\''` on macOS/Linux and as `''` on PowerShell.

macOS/Linux (Bash):

```
'<venv-python>' '<CLAUDE_RUNWAY_DIR>/tools/doctor.py' '<cwd>'
```

Windows (PowerShell): prefix the command with `&` (the call operator), because a single-quoted executable path is otherwise parsed as a string expression:

```
& '<venv-python>' '<CLAUDE_RUNWAY_DIR>\tools\doctor.py' '<cwd>'
```

Do not read `.mcp.json` yourself, and do not `cat` it: an inline `QDRANT_API_KEY` would land in the model context. `doctor` does the reading.

Capture stdout, stderr and the exit code. Show the output to the user.

### 3. Report the result

- **Exit 0:** say plainly that doctor passed and nothing needs fixing.
- **Exit 1:** say plainly that doctor found problems, then list each finding with its fix. Doctor's own text already names the fix; repeat it rather than inventing one.
- **Any other exit code, or a traceback:** show the output and say doctor itself failed to run (for example a malformed `.mcp.json`), not that the project passed.

If a finding says `upgrade` fixes it, offer `/cr-setup update`. Do not run it unasked.
