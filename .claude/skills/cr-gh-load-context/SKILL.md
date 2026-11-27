---
name: cr-gh-load-context
description: "Load project context for claude-runway: gather the repo, GitHub and plan-folder facts in one script, read the README, the start of EVALUATION.md and the latest work-order snapshot, then check that understanding against the repo's current state before resuming or asking what's next."
---

# Skill: cr-gh-load-context

Load project context for claude-runway, quickly. `CLAUDE.md`, `.plans/` and `.ai-plans/` are all gitignored (local-only, not present in a fresh checkout), so `README.md` is the always-present documentation entry point. The skill also checks whether the docs are still accurate to the repo's current state, and produces a TODO for anything that is behind.

It is built to finish in three rounds of tool calls: (1) facts, sync and the two entry docs in parallel, (2) the plan folders, (3) the report. It used to read every doc in full and took about 11 minutes; most of that was reading what the session never used, and Bash output being summarized by LM Studio.

## Usage
```
/cr-gh-load-context
```
No arguments.

## Rules that keep it fast

- **Read file content with Read (`offset`/`limit`), never with `cat`, `sed`, `head` or `grep` through Bash, and never with `compress_file`.** Bash and Grep output longer than the compress hook's threshold (`CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS`, default 2000) is sent to LM Studio before you see it. That is slow (LM Studio serves about 4 requests at a time and queues the rest, each up to the 60s timeout) and lossy: it paraphrases, so a doc read that way is not what the file says. Read is not hooked.
- **Issue independent calls in the same message.** Every step below that says "in parallel" means one message.
- **Don't re-read what is already in context.** When `CLAUDE.md` exists, Claude Code has already loaded it as project instructions; the facts script only reports whether it is present.
- **Don't read `docs/` by default.** The facts list each doc with its title (first heading) and length, and README's table of contents links them by name; together those say what each covers. Read one when the session's task needs it, not to load context.

## Steps

1. **Round 1, in parallel (one message):**
   - **The facts script:** `bash .claude/skills/cr-gh-load-context/facts.sh --porcelain` (read-only; about 5 seconds; on Windows the `facts.ps1` twin below). Keep `--porcelain`: the script ignores it, but it marks the command's output as exactness-critical, so the compress hook never summarizes it whatever its length or the threshold. It prints the branch, `main` and how far the branch is ahead, uncommitted paths, open issues by priority, issues assigned to the user, open PRs, which gitignored files exist, every `.plans/` triad with its issue's state and closing PRs plus anything misplaced, each `.ai-plans/` file's title, the work order's line ranges and whether its stated issue count and `main` SHA are stale, both skill-inventory checks, EVALUATION.md's track and status lines, and the `docs/` list (each page's title and line count). On macOS and Linux run it as above. **On Windows run its PowerShell twin instead:** `powershell -NoProfile -ExecutionPolicy Bypass -File .claude/skills/cr-gh-load-context/facts.ps1 --porcelain` (`pwsh` works too; Windows PowerShell 5.1 is always there). Keep the forward slashes: the Bash tool on Windows is Git Bash, which silently strips the backslashes from an unquoted path (`.claude\skills\...` arrives as `.claudeskills...` and PowerShell reports the file missing), while PowerShell accepts `/` as well. It prints the same sections in the same words. **Do not assume bash exists on Windows, and never use WSL** (`wsl`, or the System32 `bash.exe`): its filesystem view and its own `git`/`gh` with separate sign-ins describe a different environment. If a script cannot run at all, run the same commands by hand; the script is the list of what to check.
   - **`cb_indexer_sync_repo`** from the codebase-indexer MCP server, with this repo's absolute root as `repo_path` (required; there is no cwd default). Sent in the same message, it runs while the reads do, so nothing waits on it. It re-embeds only changed files once a manifest exists; on a fresh checkout (no `.qdrant_index_manifest.json`) it re-embeds the whole repo, which is long. It writes `.qdrant_index_manifest.json` into the repo root and updates the remote Qdrant collection. Skip silently if the server isn't available.
   - **Read `README.md` in full** (no `limit`; it is short). It holds the list of shipped pieces and the table of contents; the limitations that decide whether the toolkit fits are all on `docs/known-limitations.md`, which is not read by default. The per-file table lives in `docs/files.md`, which is not read by default; look a file up there when it matters.
   - **Read `EVALUATION.md` lines 1–60**: the thesis (Qdrant memory + local compression reduce token usage without hurting answer quality), the kinds of track (controlled A, B, C, E; the estimate D; the quality check F; the observational G) and the table of contents. The facts give the status lines; read a track's section only if its status lines leave its measurement state unclear.

2. **Round 2, in parallel, using the facts' line numbers.** Skip any folder the facts report ABSENT, and say so; step 5 then relies on GitHub state.
   - **`.ai-plans/ticket-work-order.md`:** read only the latest snapshot (the facts give its range, ending before the first `**Earlier snapshot`) and the first ~12 lines of "1. What's left" (the counts per priority). The earlier snapshots are history; read further only when choosing what to work next needs it.
   - **Other `.ai-plans/` files:** the facts give each title. A file whose title names a filed issue (an epic tracker, or design notes headed with the issue number) is a tracker or design notes, not a draft; read it only if the session's work touches it. A file that reads as an issue draft (from `/cr-gh-create-issue`) gets the filed-draft check in step 4.
   - **`.plans/`:** for a plan whose issue is OPEN, especially one assigned to the user or matching the current branch, read its "Open questions" section (the facts give the line) and its Goal. Don't read closed tickets' triads.

3. **Summarize back** what was loaded, concisely:
   - What the shipped pieces the README lists do, and which stand alone and which build on another.
   - Which EVALUATION.md tracks are measured vs. still a plan. Track implementation separately from measurement: a merged PR proves a feature exists, not that its A/B run happened (Track D is an estimate by design).
   - The most recent completed work and the in-progress item (from `.plans/` and the work order's latest snapshot), and what the work order lists as ready next.

4. **Accuracy check.** Report each mismatch as one TODO line, "docs claim X, repo shows Y"; if nothing is out of sync, say so explicitly.
   - **Skill inventories:** the facts compare `skills/` against `docs/files.md` (every skill, and every file shipped beside a `SKILL.md`) and `docs/skills-and-hooks.md`, and `.claude/skills/` against `docs/development-workflow.md`: two audiences, product skills vs. the contributor-facing `cr-gh-*` skills. Report what they print.
   - **EVALUATION.md:** does any track claim or imply a measurement on the strength of its implementation PR alone? Check its status lines for a recorded result.
   - **`.plans/`:** a `-plan.md` whose issue is CLOSED with a closing PR and has no `-validate.md` is either stale or incomplete: flag it, don't assume. A plan whose issue is CLOSED with no closing PR was closed by hand, not by a merge. Anything the facts mark MISPLACED belongs in `.ai-plans/`.
   - **Work order:** report the facts' `count`/`sha` verdict; STALE means its snapshot predates the live state.
   - **Filed drafts (only for files that read as issue drafts):** a title search (`gh issue list --search "<title>" --state all --limit 300 --json number,title`) only finds candidates: it is full-text and ranks loosely. For each candidate, fetch its body (`gh issue view <n> --json title,body`) and check every substantive line of the draft appears in it. Call the draft safe to delete only when they all do; otherwise report a possible match to review and keep it, since it may be the only copy of that text.

5. **Pick up where it left off, or ask what's next.** If there is a clear single item in progress (an issue assigned to the user, one matching the current branch, or a `.plans/` entry with no `-validate.md` whose issue is open), propose resuming it, name any open questions in its plan, and wait for confirmation. Otherwise ask what's next, recommending a starting point from the work order's suggested order. For the full open list, run `gh issue list --state open --limit 300 --json number,title,labels,assignees` (always pass `--limit`: without it `gh` returns only the 30 newest, which once hid two P0s and ten P1s).

This skill creates and edits no files, and the accuracy-check TODO is reported, not fixed. It isn't fully read-only end to end: `cb_indexer_sync_repo` writes `.qdrant_index_manifest.json` and updates the remote Qdrant collection, a disclosed side effect of keeping the index current.
