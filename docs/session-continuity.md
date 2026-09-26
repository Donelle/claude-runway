# Session continuity skills

The `skills/` directory ships two Claude Code slash commands that use both toolkit pieces together to replace the native `/compact` command with a fully local alternative.

| Skill | Invocation | What it does |
|---|---|---|
| `my-compact` | `/my-compact` or `/my-compact "label"` | Summarizes the current conversation via LM Studio (`compress_text`), stores the result in Qdrant with project/date/label metadata, then prompts you to `/clear` and `/my-resume` |
| `my-resume` | `/my-resume` or `/my-resume <keyword>` | Retrieves compacted sessions from Qdrant for the current project, shows a numbered list with labels and dates, and restores the one you pick |

The optional label argument to `/my-compact` (e.g. `/my-compact "auth refactor"`) is how you identify the session later — without it, the label is auto-derived from the summary content. `/my-resume` accepts an optional keyword to narrow the search (e.g. `/my-resume auth refactor`) and skips straight to restore if there's one clear match; when multiple sessions exist it presents a selection UI (`AskUserQuestion`) rather than a plain numbered list.

**Why this matters:** native `/compact` sends your conversation to Anthropic for summarization. These skills route the same work through LM Studio locally — zero Anthropic tokens spent on compaction — and persist the result in Qdrant so it's retrievable across sessions by project and label. See `EVALUATION.md` Track C for how to measure whether this actually saves tokens vs. the alternatives.

**Prerequisites:** both the Qdrant and local-compress pieces must be configured (steps 4 and 4b of [Installation](installation.md)). These skills call `compress_text`, `compact_store`, and `compact_find` — all three live in the `local-compress` server, so only that server needs to be in `.mcp.json` (the `qdrant` server is not required for the session continuity skills).

**Installation:**

```bash
claude-runway-setup init --install-skills   # installs/updates every product skill, including these two
```

(Or, from a clone: `python tools/setup_project.py init --install-skills`.) This installs/updates ALL of this repo's product skills at once (not just these two) — missing ones are installed, changed ones are updated, identical ones are left alone (issue [#14](https://github.com/Donelle/claude-runway/issues/14)). Works the same way whether you cloned this repo or installed it via pipx/`uv tool install` (see [Installation](installation.md)'s "Alternative" callout) — `skills/` ships in the package either way. Equivalent manual copy, if you'd rather not run the CLI:

```bash
mkdir -p ~/.claude/skills/my-compact ~/.claude/skills/my-resume
cp skills/my-compact/SKILL.md ~/.claude/skills/my-compact/SKILL.md
cp skills/my-resume/SKILL.md ~/.claude/skills/my-resume/SKILL.md
```

Skills install globally (under `~/.claude/skills/`) rather than per-project, so you only need to do this once — pass `--skills-dir <path>` to `--install-skills` if you want a per-project install instead (e.g. `<repo>/.claude/skills`). The `/my-resume` command retrieves the right session by matching the `project` field in metadata against the current working directory name — switching projects automatically scopes the results.

