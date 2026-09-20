---
name: my-gh-load-context
description: "Load complete project context for claude-runway by reading CLAUDE.md, README.md, docs/, EVALUATION.md, and .plans/, then checking that understanding against the repo's current state before resuming or asking what's next."
---

# Skill: my-gh-load-context

Load complete project context for claude-runway. `CLAUDE.md` and `.plans/` are both gitignored (local-only, not present in a fresh checkout), so `README.md` is the actual always-present documentation entry point — this skill reads it plus `docs/`, `EVALUATION.md`, and, when present, `CLAUDE.md` and `.plans/`. It also checks whether what it just read is still accurate to the repo's current state, and produces a TODO for anything it's behind on.

## Usage
```
/my-gh-load-context
```
No arguments.

## Steps

1. **Read `CLAUDE.md`** in the repo root, if present — it's gitignored, so a fresh clone won't have it locally even though it exists in this working copy. If present, note it's scoped to MCP tool usage (Qdrant memory, memory-bank, LM Studio compression) — there's no doc-links or "First Steps" table here. If absent, report that plainly and continue with `README.md` as the documentation entry point instead of claiming its rules were loaded.

2. **Read `README.md`** end to end to understand the file structure and design philosophy: the two independent components (Qdrant codebase memory, local compression), the key files under `tools/`, `libs/`, `hooks/`, `skills/`, `.claude/skills/`, and the "Known Limitations & Caveats" section.

3. **Read every file in `docs/`** (currently: `cross-repo-lookups.md`, `development-workflow.md`, `environment-variables.md`, `installation.md`, `memory-bank.md`, `prerequisites.md`, `savings-tracker.md`, `session-continuity.md`, `skills-and-hooks.md`, `verifying-its-working.md`, `windows-setup.md` — re-glob `docs/*.md` since this list will drift). These cover setup, env vars, hooks/skills precedence, and the design rationale behind specific mechanisms (exactness-critical Bash exemptions, fetch_url vs WebFetch, etc.).

4. **Read `EVALUATION.md`** to understand the project's actual goal: the core thesis is "Qdrant memory + local compression reduces token usage without hurting answer quality," tested per-track (A: Qdrant memory, B: local compression, C: session continuity, D: the savings tracker as an estimate only, E: memory-bank reuse). Note which tracks have a completed measurement vs. are still just a plan.

5. **Read `.plans/`, if present**, to understand what's done, in progress, or next — like `CLAUDE.md`, it's gitignored, so a fresh clone won't have it. If absent, report that local plan artifacts aren't available and rely on GitHub issue/PR state instead (step 9 covers that).
   - List all files. Most follow an `{issue-number}-plan.md` / `{issue-number}-research.md` / `{issue-number}-validate.md` triad; the rest are standalone lesson/observation docs (e.g. `autowork-observations-log.md`).
   - A `-plan.md` with no matching `-validate.md` doesn't by itself mean the work is unfinished — `gh issue list --json state` only reports `OPEN`/`CLOSED` and can't distinguish an issue closed by a merged PR from one closed manually. Use `gh issue view {number} --json state,closedByPullRequestsReferences` (or the GraphQL `closedByPullRequestsReferences` field) to confirm via the actual closing PR before calling something stalled.
   - Read `autowork-observations-log.md` if present — it's the running log of autowork sessions and is the most direct signal for "what happened most recently."

6. **Sync the Qdrant index** by calling `sync_repo` from the codebase-indexer MCP server, passing the current repo's absolute root path explicitly as `repo_path` (it's a required positional argument — there's no cwd default). This re-embeds only changed files *once a manifest exists* — but on a fresh checkout (no `.qdrant_index_manifest.json` yet), there's nothing to diff against, so the first sync re-embeds the entire repo, not a cheap incremental update. It also isn't side-effect-free either way: it writes `.qdrant_index_manifest.json` into the repo root and updates the remote Qdrant collection. Skip silently if the MCP server isn't available.

7. **Summarize back** what was loaded, concisely, so it's in context for the rest of the session:
   - What the two components do and how they're independent
   - Which EVALUATION.md tracks are measured vs. still theoretical
   - What `.plans/` shows as the most recent completed work and the most recent in-progress/open item, if `.plans/` was present — otherwise say so and rely on step 9's GitHub-based check instead

8. **Accuracy check.** Compare what the docs claim against the repo's actual current state, and list any mismatch as a TODO rather than silently trusting the docs:
   - Does `README.md`'s skill listing match what's actually in `skills/`? Separately, does `docs/development-workflow.md`'s skill listing match what's actually in `.claude/skills/`? These are two different inventories for two different audiences — `skills/` holds the globally-synced product skills documented in README (`my-compact`/`my-resume`/`my-savings`/`my-setup-clauderunway`); `.claude/skills/` holds the contributor-facing `my-gh-*` skills documented in `docs/development-workflow.md`. Compare each directory against its own doc, not both against README — otherwise every `my-gh-*` skill will falsely show up as undocumented on every run.
   - Does `EVALUATION.md` claim (or imply) a track is measured based only on its implementation PR being merged? A merged PR proves the feature exists, not that the A/B measurement was actually run — Track D is explicitly called out as an estimate rather than a measured track for exactly this reason. Check for a recorded benchmark result or explicit evidence of a completed run, and track implementation status separately from measurement status.
   - Does `.plans/` contain a `-plan.md` for an issue confirmed closed-via-merged-PR (per step 5's `closedByPullRequestsReferences` check), with no matching `-validate.md`? That's either stale or genuinely incomplete — flag it, don't assume either way.
   - Report findings as a short TODO list: "docs claim X, repo shows Y" — one line each. If nothing is out of sync, say so explicitly rather than omitting the check.

9. **Pick up where it left off, or ask what's next.** Check `gh issue list --state open --json number,title,labels,assignees` and `gh pr list --state open`. If there's a clear single open issue/PR already in progress (e.g. matching the current branch, or the most recent `.plans/` entry with no `-validate.md`), propose resuming it and wait for confirmation. Otherwise, ask the user what's next.

This skill doesn't create or edit any files itself, and the accuracy-check TODO is reported back rather than auto-fixed — but it isn't fully read-only end to end: step 6's `sync_repo` call writes `.qdrant_index_manifest.json` into the repo and mutates the remote Qdrant collection as a disclosed side effect of keeping the index current, not something this skill's own logic does.
