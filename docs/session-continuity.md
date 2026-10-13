# Session continuity skills

The `skills/` directory ships two Claude Code slash commands that use both toolkit pieces together to replace the native `/compact` command with a fully local alternative.

| Skill | Invocation | What it does |
|---|---|---|
| `my-compact` | `/my-compact` or `/my-compact "label"` | Fills in a structured summary of the current conversation and stores it in Qdrant with project/date/label metadata through one `compact_store` call, which compresses it on LM Studio first (or stores it as given when LM Studio is unavailable), then prompts you to `/clear` and `/my-resume` |
| `my-resume` | `/my-resume` or `/my-resume <keyword>` | Retrieves compacted sessions from Qdrant for the current project, shows a numbered list with labels and dates, and restores the one you pick |

The optional label argument to `/my-compact` (e.g. `/my-compact "auth refactor"`) is how you identify the session later — without it, the label is auto-derived from the summary content. `/my-resume` accepts an optional keyword to narrow the search (e.g. `/my-resume auth refactor`) and skips straight to restore if there's one clear match; when multiple sessions exist it presents a selection UI (`AskUserQuestion`) rather than a plain numbered list.

**Why this matters:** native `/compact` sends your conversation to Anthropic for summarization. These skills route the same work through LM Studio locally — zero Anthropic tokens spent on compaction — and persist the result in Qdrant so it's retrievable across sessions by project and label. See `EVALUATION.md` Track C for how to measure whether this actually saves tokens vs. the alternatives.

**Prerequisites:** the `qdrant` server must be configured (step 4 of [Installation](installation.md)). These skills call `compact_store` and `compact_find`, which live in that server alongside `qdrant-find`/`qdrant-store`; the `local-compress` server is not needed. LM Studio is optional: `compact_store` compresses the summary on the local model when it can, and otherwise stores it uncompressed (its reply says which), so the compact is never lost to a compression problem.

**Which embedding model compacts use:** the `qdrant` block's `COMPACT_EMBEDDING_MODEL`, deliberately separate from `EMBEDDING_MODEL` (the code-index model). Compacts are stored under a model-named vector, so changing it for a project that already has compacts orphans them; the server reports a mismatch instead of silently missing them. Compacts saved before the tools moved to the `qdrant` server keep working unchanged: `claude-runway-setup upgrade` carries the model over (pinning `BAAI/bge-small-en` for installs that never set one).

**Installation:**

```bash
claude-runway-setup init --install-skills   # installs/updates every product skill, including these two
```

(Or, from a clone: `python tools/setup_project.py init --install-skills`.) This installs/updates ALL of this repo's product skills at once (not just these two) — missing ones are installed, changed ones are updated, identical ones are left alone. Works the same way whether you cloned this repo or installed it via pipx/`uv tool install` (see [Installation](installation.md)'s "Alternative" callout) — `skills/` ships in the package either way. Equivalent manual copy, if you'd rather not run the CLI:

```bash
mkdir -p ~/.claude/skills/my-compact ~/.claude/skills/my-resume
cp skills/my-compact/SKILL.md ~/.claude/skills/my-compact/SKILL.md
cp skills/my-resume/SKILL.md ~/.claude/skills/my-resume/SKILL.md
```

Skills install globally (under `~/.claude/skills/`) rather than per-project, so you only need to do this once — pass `--skills-dir <path>` to `--install-skills` if you want a per-project install instead (e.g. `<repo>/.claude/skills`). The `/my-resume` command retrieves the right session by matching the `project` field in metadata against the current working directory name — switching projects automatically scopes the results.

