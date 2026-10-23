---
name: my-gh-load-context
description: "Load complete project context for claude-runway by reading CLAUDE.md, README.md, docs/, EVALUATION.md, .plans/ and .ai-plans/, then checking that understanding against the repo's current state before resuming or asking what's next."
---

# Skill: my-gh-load-context

Load complete project context for claude-runway. `CLAUDE.md`, `.plans/` and `.ai-plans/` are all gitignored (local-only, not present in a fresh checkout), so `README.md` is the actual always-present documentation entry point — this skill reads it plus `docs/`, `EVALUATION.md`, and, when present, `CLAUDE.md`, `.plans/` and `.ai-plans/`. It also checks whether what it just read is still accurate to the repo's current state, and produces a TODO for anything it's behind on.

## Usage
```
/my-gh-load-context
```
No arguments.

## Steps

1. **Read `CLAUDE.md`** in the repo root, if present — it's gitignored, so a fresh clone won't have it locally even though it exists in this working copy. If present, note it's scoped to MCP tool usage (Qdrant memory, memory-bank, LM Studio compression) — there's no doc-links or "First Steps" table here. If absent, report that plainly and continue with `README.md` as the documentation entry point instead of claiming its rules were loaded.

2. **Read `README.md`** end to end to understand the file structure and design philosophy: the two independent components (Qdrant codebase memory, local compression), the key files under `tools/`, `libs/`, `hooks/`, `skills/`, `.claude/skills/`, and the "Known limitations" section.

3. **Read every file in `docs/`** — glob `docs/*.md` for the current set rather than relying on a list written down here, since any such list drifts as docs are added. These cover setup, env vars, hooks/skills precedence, and the design rationale behind specific mechanisms (exactness-critical Bash exemptions, fetch_url vs WebFetch, etc.).

4. **Read `EVALUATION.md`** to understand the project's actual goal: the core thesis is "Qdrant memory + local compression reduces token usage without hurting answer quality," tested per-track (A: Qdrant memory, B: local compression, C: session continuity, D: the savings tracker as an estimate only, E: memory-bank reuse). Note which tracks have a completed measurement vs. are still just a plan.

5. **Read the two local plan folders, if present.** Like `CLAUDE.md`, both are gitignored, so a fresh clone won't have them. If either is absent, say so for that folder and rely on GitHub issue/PR state instead (step 9 covers that). They hold different things:
   - **`.ai-plans/`** — the backlog view. Read `ticket-work-order.md` first when present: it summarizes what is open, ready, blocked or waiting on a decision, the suggested order, and what was recently done. Then glob the folder for anything else (issue drafts from `/my-gh-create-issue`, standalone notes) rather than relying on a list written down here.
   - **`.plans/`** — only the artifacts of tickets actually worked: an `{issue-number}-plan.md` / `{issue-number}-research.md` / `{issue-number}-validate.md` triad per ticket. List all files; anything that isn't part of a triad is misplaced (step 8 flags it).
   - A `-plan.md` with no matching `-validate.md` doesn't by itself mean the work is unfinished — `gh issue list --limit 300 --json state` only reports `OPEN`/`CLOSED` and can't distinguish an issue closed by a merged PR from one closed manually. Use `gh issue view {number} --json state,closedByPullRequestsReferences` (or the GraphQL `closedByPullRequestsReferences` field) to confirm via the actual closing PR before calling something stalled.

6. **Sync the Qdrant index** by calling `sync_repo` from the codebase-indexer MCP server, passing the current repo's absolute root path explicitly as `repo_path` (it's a required positional argument — there's no cwd default). This re-embeds only changed files *once a manifest exists* — but on a fresh checkout (no `.qdrant_index_manifest.json` yet), there's nothing to diff against, so the first sync re-embeds the entire repo, not a cheap incremental update. It also isn't side-effect-free either way: it writes `.qdrant_index_manifest.json` into the repo root and updates the remote Qdrant collection. Skip silently if the MCP server isn't available.

7. **Summarize back** what was loaded, concisely, so it's in context for the rest of the session:
   - What the two components do and how they're independent
   - Which EVALUATION.md tracks are measured vs. still theoretical
   - What `.plans/` shows as the most recent completed work and the most recent in-progress/open item, and what `.ai-plans/ticket-work-order.md` lists as ready next — for whichever folder was present; otherwise say so and rely on step 9's GitHub-based check instead

8. **Accuracy check.** Compare what the docs claim against the repo's actual current state, and list any mismatch as a TODO rather than silently trusting the docs:
   - Does `README.md`'s skill listing match what's actually in `skills/`? Separately, does `docs/development-workflow.md`'s skill listing match what's actually in `.claude/skills/`? These are two different inventories for two different audiences — `skills/` holds the globally-synced product skills documented in README; `.claude/skills/` holds the contributor-facing `my-gh-*` skills documented in `docs/development-workflow.md`. Compare each directory against its own doc, not both against README — otherwise every `my-gh-*` skill will falsely show up as undocumented on every run.
   - Does `EVALUATION.md` claim (or imply) a track is measured based only on its implementation PR being merged? A merged PR proves the feature exists, not that the A/B measurement was actually run — Track D is explicitly called out as an estimate rather than a measured track for exactly this reason. Check for a recorded benchmark result or explicit evidence of a completed run, and track implementation status separately from measurement status.
   - Does `.plans/` contain a `-plan.md` for an issue confirmed closed-via-merged-PR (per step 5's `closedByPullRequestsReferences` check), with no matching `-validate.md`? That's either stale or genuinely incomplete — flag it, don't assume either way.
   - Does `.plans/` contain anything that isn't part of an `{issue-number}-*` triad? Flag it as misplaced: drafts and standalone notes belong in `.ai-plans/`.
   - Is `.ai-plans/ticket-work-order.md`'s snapshot stale? Compare the open-issue count and `main` SHA it states against `gh issue list --state open --limit 300` and `git rev-parse --short main`, and flag any difference.
   - Does `.ai-plans/` hold an issue draft that has already been filed? A title search (`gh issue list --search "<title>" --state all --limit 300`) only finds candidates: it is full-text and ranks loosely, so it returns unrelated issues even for a draft that was never filed. For each candidate, fetch its body (`gh issue view <n> --json title,body`) and check that every substantive line of the draft appears in it. Flag the draft as safe to delete only when they all do; otherwise report it as a possible match to review and keep it, since a draft edited after filing, or one matched to a merely related issue, would be the only copy of that text.
   - Report findings as a short TODO list: "docs claim X, repo shows Y" — one line each. If nothing is out of sync, say so explicitly rather than omitting the check.

9. **Pick up where it left off, or ask what's next.** Check `gh issue list --state open --limit 300 --json number,title,labels,assignees` and `gh pr list --state open --limit 300` (always pass `--limit`: both commands silently truncate to the 30 newest results without it, which once hid two P0s and ten P1s out of 73 open issues). If there's a clear single open issue/PR already in progress (e.g. matching the current branch, or the most recent `.plans/` entry with no `-validate.md`), propose resuming it and wait for confirmation. Otherwise, ask the user what's next, using the work order's suggested order (if `.ai-plans/ticket-work-order.md` was present) to recommend a starting point.

This skill doesn't create or edit any files itself, and the accuracy-check TODO is reported back rather than auto-fixed — but it isn't fully read-only end to end: step 6's `sync_repo` call writes `.qdrant_index_manifest.json` into the repo and mutates the remote Qdrant collection as a disclosed side effect of keeping the index current, not something this skill's own logic does.
