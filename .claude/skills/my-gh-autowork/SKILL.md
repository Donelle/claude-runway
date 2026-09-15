---
name: my-gh-autowork
description: "Autonomously work claude-runway's open bug/enhancement backlog end-to-end (plan, code, PR, verified multi-round review-feedback loop, merge) by delegating each ticket to an isolated subagent — zero approval gates, one ticket at a time, stops and reports rather than guessing on anything genuinely ambiguous. Refuses to run at all outside a verified dogfood checkout."
---

# Skill: my-gh-autowork

Fully autonomous version of the `/my-gh-code-it` → `/my-gh-pr` → `/my-gh-pr-feedback` → merge cycle already used by hand on this repo. Where those three skills pause for human approval (plan review, branch-name confirmation), this one doesn't — it's for the case where the user has explicitly authorized working through tickets without per-step check-ins. It doesn't replace those three skills' *content*; it packages their combined, battle-tested procedure (plus the merge step, which none of them do) into one subagent prompt run via the Agent tool.

**Zero required human touchpoints, confirmed live.** This repo has a repository ruleset ("PR Re-review", `gh api repos/Donelle/claude-runway/rulesets/23376382`) with `copilot_code_review: {review_on_push: true}` active — Copilot automatically re-reviews on EVERY push to a PR targeting the default branch, no manual re-request and no `@copilot` mention needed. Confirmed live (2026-08-25, working issue #45/PR #110): a fix push triggered a fresh Copilot review within ~6 minutes with zero action from anyone. This means the entire review-feedback loop — including every round after the first — can run inside a single subagent call: push, wait, check, fix-and-push-again if needed, repeat, merge. An earlier version of this skill split this into orchestrator-driven `start`/`continue` modes with a mandatory human ping between rounds, because at the time a re-review genuinely required a manual action that risked the `@copilot`-mention failure mode if done wrong. That's no longer true on this repo and the split has been removed — don't reintroduce it without first re-confirming `review_on_push` is still active (`gh api repos/Donelle/claude-runway/rulesets` should list a ruleset with that rule; if it's gone, the old multi-mode design in this file's git history is the fallback).

**Do not skip the verification discipline just because there's no human in the loop.** The two safety practices that matter most when nobody's watching in real time are: (1) verify every review-feedback claim by actual reproduction before touching code, never by taking a reviewer's (bot or human) word for it, and (2) stop and report rather than push forward on anything that isn't a well-defined "make this specific defect go away" fix — a design decision, an ambiguous requirement, or a review disagreement that survives a second explanation round is a *reason to stop*, not something to resolve by guessing or looping indefinitely (see the round cap in Step 4 below).

**Every subagent call below runs as a genuinely fresh, isolated context, in its own git worktree.** These are two separate settings doing two separate jobs — don't conflate them, a future edit that "simplifies" by dropping one thinking the other covers it would reintroduce a real problem: `subagent_type: "general-purpose"` is what makes each Agent call spawn a brand-new agent with no memory of the orchestrator's conversation or any other ticket. `isolation: "worktree"` is what keeps that subagent's own `git checkout`/`commit`/`push` from ever touching whatever the orchestrator or a human happens to be doing in the primary checkout at the same time — confirmed necessary live: an earlier test run without it shared the primary working directory with a concurrent uncommitted edit and only avoided a collision because the subagent happened to scope its `git add` narrowly, not because anything structurally prevented one. Together they're what keeps the *orchestrating* conversation's context flat no matter how many tickets get processed in one run, since every file read, diff, and review-comment thread lives inside a subagent's own throwaway worktree and only a short report ever comes back.

Project-scoped to this repo (hardcodes `Donelle/claude-runway`, `.venv`-based tests, this repo's branch/commit conventions) for the same reason `my-gh-code-it`/`my-gh-pr`/`my-gh-pr-feedback` are.

## Usage
```
/my-gh-autowork              # pick the single highest-priority open bug/enhancement and work it fully
/my-gh-autowork 45           # work that specific issue fully
/my-gh-autowork all          # work the entire open bug/enhancement backlog, one ticket at a time
```

## Steps (orchestrator — runs in the main conversation, kept deliberately thin)

0. **Dogfood preflight — STOP before doing anything else if this checkout isn't configured for it.** `.claude/skills/` is committed to this repo, so this skill ships to every clone — but it grants full autonomous commit/push/merge authority with zero human approval gate, which should only run somewhere deliberately, verifiably set up for it, never assumed just because the skill file is present. This check has TWO parts, and the first one is not optional even though it seems obvious: **claude-runway is itself a toolkit meant to be installed into OTHER repos** ("one copy serves all projects," per its own README) — any OTHER project that installed it would have a `.mcp.json` with the exact same `qdrant`/`codebase-indexer` server names and a `.claude/settings.json` with the exact same `PostToolUse` hook shape, since those come from THIS project's own `templates/`. A file-shape check alone can't tell "a genuine claude-runway dogfood checkout" apart from "some unrelated repo that happens to have installed claude-runway's tools" — or a fork. Confirming the repo identity FIRST, before spawning anything, is what actually gates this — checking it only inside the subagent (too late: it would have already read files and run `sync_repo` against whatever repo it's actually in by the time it notices) doesn't count:
   ```bash
   ORIGIN=$(git remote get-url origin 2>/dev/null)
   case "$ORIGIN" in
     *Donelle/claude-runway*) REPO_OK=yes ;;
     *) REPO_OK=no ;;
   esac
   test -f .mcp.json && test -f .claude/settings.json \
     && jq -e '.mcpServers | has("qdrant") and has("codebase-indexer")' .mcp.json >/dev/null 2>&1 \
     && jq -e '.hooks | has("PostToolUse")' .claude/settings.json >/dev/null 2>&1 \
     && [ "$REPO_OK" = "yes" ] \
     && echo "DOGFOODING: yes" || echo "DOGFOODING: no"
   ```
   If this prints anything other than `DOGFOODING: yes`, **STOP immediately** — report to the user that this checkout isn't a verified claude-runway dogfood checkout (wrong repo, or missing/structurally-incomplete `.mcp.json`/`.claude/settings.json` — point them at README's Installation section and `templates/mcp.json.template`/`templates/settings.json.template`) and do not spawn any subagent, do not touch any ticket.

1. **Track only a minimal `attempted` list of issue numbers** across this run — nothing else about a completed ticket needs to live in this conversation's context. Starts empty.

2. **Loop, one subagent call per ticket:**
   a. Before calling, note which model this Agent call will run as (an explicit `model`
      param on this call, or the harness default this conversation is itself running as,
      if none is set) and the current time to the second (e.g. `date +%H%M%S`) — both
      needed for the metrics logging in 2b below, and cheapest to capture right here since
      the orchestrator already knows/can get them at call time. Call the **Agent** tool,
      `subagent_type: "general-purpose"`, `isolation: "worktree"`, with the template below
      — substitute `{ISSUE_NUMBER}` (explicit) or `{ATTEMPTED_LIST}` (auto-pick mode).
      **After backgrounding the call, just wait for the Agent tool's own task-completion
      notification — do NOT call `ScheduleWakeup`, do NOT start a manual polling loop.
      There is nothing to schedule and nothing to poll: the harness re-invokes this
      conversation automatically when the backgrounded task finishes. `ScheduleWakeup` is
      a `/loop`-only tool and errors here.** Confirmed observed live (issue #132, 2026-09-09):
      the orchestrator spontaneously called `ScheduleWakeup` at this exact moment, then
      correctly self-corrected in the same turn — but a failed tool call is wasted latency.
   b. Read its final report — exactly one of:
      - `OUTCOME: MERGED (issue #<n>, PR #<n>)` → append `<n>` to `attempted`, log this
        ticket to the metrics store (see below), continue to the next ticket.
      - `OUTCOME: DONE` → no open, unattempted ticket was left to pick. Stop the whole run
        cleanly — not a failure, just an empty backlog. Nothing to log here; no ticket was
        actually worked.
      - `OUTCOME: BLOCKED (issue #<n>) — <reason>` / `OUTCOME: FAILED (issue #<n>) —
        <reason>` → append `<n>` to `attempted` anyway (so re-running this same invocation
        doesn't immediately re-pick the same stuck ticket), log this ticket to the metrics
        store too (a blocked/failed outcome is itself real signal, not noise to skip),
        **stop the whole run immediately**, surface the reason verbatim.

      **Logging a ticket to the metrics store** (orchestrator-only bookkeeping — the
      subagent's prompt and report format are NOT changed for this, so every subagent
      stays focused purely on its own ticket and has no logging responsibility at all):
      derive `rounds` and a short categorized `findings` list (a category tag plus
      reproduced/fixed vs. declined, per finding) yourself from the report's own existing
      prose summary — every subagent report already includes this per the Final report
      section below, this just means reading it rather than asking for a new format.
      Combine that with the model and timestamp noted in 2a and this Agent call's own
      `usage` block (`duration_ms` / 1000 → `wall_clock_s`, `subagent_tokens`, `tool_uses`
      → `tool_calls`), then call:
      ```
      compact_store(
        project="claude-runway-autowork-metrics",
        label=f"issue-{n}-{HHMMSS}",
        date=<today, ISO format YYYY-MM-DD>,
        information=<JSON string: {"issue": n, "pr": pr_or_null, "outcome": "...",
                       "model": "...", "rounds": N, "manual_interventions": N,
                       "wall_clock_s": N, "subagent_tokens": N, "tool_calls": N,
                       "findings": [{"category": "...", "reproduced": bool, "fixed": bool}]}>
      )
      ```
      **`{HHMMSS}` in the label is not decorative — it's what keeps two attempts at the
      SAME issue on the SAME calendar day from colliding.** `compact_store` derives its
      point id from `(project, label, date)` and upserts on a match (issue #36); a
      fixed `f"issue-{n}"` label would let a same-day retry silently overwrite an earlier
      attempt's entry — e.g. a timeout/`FAILED` run followed by a same-day successful
      retry would erase the very failure signal this whole feature exists to retain
      (flagged in PR review on #127). The per-attempt timestamp makes every attempt its
      own point regardless of how many times the same issue is worked in one day.
      `manual_interventions` counts any `SendMessage` resume THIS orchestrator had to send
      this same subagent to get a compliant final report (0 for a clean single-call
      ticket — see ticket #1's stalling incident in this skill's own commit history for
      why that field exists).

      **This logging step is best-effort and additive — it must never change what the
      outer 2b/2c control flow does next, only whether a metrics entry got recorded.**
      `compact_store` can fail two different ways, and both need checking (a raised
      exception is NOT the only failure shape): it can also return an ordinary string
      starting with `Error:` instead of raising (see `/my-compact`'s own Step 6 discipline
      for the same check) — treat either shape as "logging failed," note it, and move on
      to whatever 2b/2c already say to do for this ticket's actual outcome. Concretely:
      for a `MERGED` result in an `all` run, still continue to the next ticket even if
      logging failed; for a `BLOCKED`/`FAILED` result, still **stop the whole run
      immediately** and surface the reason, even if logging failed; for a single-ticket
      invocation (no argument, or an explicit issue number), the run still ends after this
      one call regardless. A metrics-logging failure is never itself a reason to stop a
      run that would otherwise continue, or continue a run that would otherwise stop.
   c. If this invocation was `all`, go back to 2a for the next ticket (unless the outcome
      was `BLOCKED`/`FAILED`/`DONE`, per 2b above). **For every OTHER invocation — an
      explicit issue number, or no argument at all — the run ends after this one call,
      regardless of outcome.** No-argument mode picks and works exactly one ticket, per
      Usage above; it must not fall through to looping like `all` just because it also
      goes through Step 0's auto-pick logic to choose that one ticket.

3. **After the run ends**, give the user a summary table: ticket #, outcome, PR # if any, and the one-line reason for any non-merge outcome. Don't start a new run without being asked again, even if this invocation was `all`.

## Subagent Prompt Template

Pass this whole block as the `prompt` for one `Agent` call per ticket, substituting `{ISSUE_NUMBER}` or `{ATTEMPTED_LIST}` as directed above. Don't paraphrase or shorten it — several of these lines exist because skipping them caused a real, confirmed failure the first time this cycle was worked (issue #38, 2026-08-25 by hand; issues #44/#45, 2026-08-25 during this skill's own live testing); the citations are there so a future edit doesn't silently drop the fix.

````
You are autonomously working a GitHub issue on Donelle/claude-runway, end to end:
pick-a-ticket (if not given one) → load context → plan → implement → test → PR → verified
review-feedback loop (may span several rounds, all within this one call — see Step 5) →
merge. You have full autonomy — do not ask the user anything and do not wait for approval
at any step. If you hit something genuinely ambiguous or that needs a human decision, STOP
and report it as BLOCKED rather than guessing. You are a fresh, isolated context (running
in your own git worktree) with no memory of any other ticket worked in this run.

**YOU are the one actually doing the work, end to end, in your own tool calls — there is
no separate background process or orchestrator waiting to run steps for you.** Whenever
this prompt says "wait" or "poll" (Step 19's wait for a Copilot review is the main case),
that means YOU must run a blocking command yourself (a shell loop with `sleep`, with an
explicit long-enough timeout on the Bash call itself, up to 600000ms) and keep making tool
calls until you reach a terminal state. Confirmed live and repeated identically on the
first ticket this skill worked without this paragraph (issue #49, 2026-08-27): the
subagent completed real work correctly, then twice ended its turn with a bare
"I'll wait for the notification" / "I'll stop here and wait" instead of running Step 19's
poll itself — it had correctly identified that *something* needed to wait, then misapplied
the orchestrator's own "spawn a subagent and wait for its notification" mental model to
itself, even though nothing else was ever going to pick the work back up. Confirmed fixed
by this exact paragraph across 6/6 tickets since (two separate `/my-gh-autowork all`
invocations, 2026-08-27), zero recurrence. Never end a turn with "I'll wait for..." or
similar language — there is nothing else that will pick this up; if you stop, the work
simply stops. Keep issuing tool calls until you hit Step 5 (merge) or a genuine terminal
BLOCKED/FAILED condition — the one exception is Step 0's own empty-backlog case, where
`OUTCOME: DONE` as your entire response IS the correct immediate stop, per that step's own
instructions below. Whichever of these applies, your literal last line of output must
always be one of the four OUTCOME line shapes from the Final report section at the bottom
(three from that section directly, plus the `DONE` case Step 0 and that section both
call out separately).

## Step 0 — determine the target ticket
{ISSUE_NUMBER}                                   <-- orchestrator fills in ONE of these two
--- OR ---
Pick the highest-priority open issue on Donelle/claude-runway labeled `bug` OR
`enhancement` that is NOT already assigned to someone other than you, and whose number is
NOT in this already-attempted list this run: {ATTEMPTED_LIST}. IMPORTANT: `gh issue list`
silently truncates to 30 results with no `--limit` flag — always pass one. Also note two
labels do NOT OR together via repeated `--label` flags (that ANDs, requiring both labels on
the same issue); use `--search` for OR:
  gh issue list -R Donelle/claude-runway --state open --search "label:bug,enhancement" \
    --limit 200 --json number,title,labels,assignees
Sort by priority label (priority-p1 > p2 > p3; unlabeled sorts last), then by issue number
ascending as a tiebreaker; bug and enhancement are not otherwise prioritized relative to
each other. If nothing qualifies, output exactly `OUTCOME: DONE` as your entire response
and stop — do not proceed to any step below.

## Step 1 — load project context (this repo's own /my-load-context, inlined)
This is inlined rather than invoked as `/my-load-context` because that's a personal skill
under `~/.claude/skills/`, not guaranteed to exist on whichever machine is running this.
1. **Fetch and reset to `origin/main` FIRST, before anything else in this worktree** —
   confirmed necessary live: a freshly created worktree's initial files reflect whatever
   commit the orchestrator's OWN primary checkout happened to be on at the moment this
   worktree was created, not necessarily current `main` (e.g. if the orchestrator is
   itself mid-edit on some other branch). Without this, the `sync_repo` call in the next
   item could index stale or wrong-branch content while believing it reflects current
   `main`. This does NOT hit the "main already checked out elsewhere" conflict Step 2
   describes below — it resets whatever branch this worktree already has checked out to
   point at the same commit as `origin/main`, without ever checking out the `main` ref
   itself:
   ```bash
   git fetch origin main && git reset --hard origin/main
   ```

2. Read `README.md` in full — this repo has no `CLAUDE.md`, so README is the doc entry
   point (its Files/Environment variables/Skills-and-hooks tables cover the domain
   knowledge a `CLAUDE.md`'s "Technical Documentation" table would otherwise point at).
3. Note the top-level solution structure (`libs/`, `tools/`, `hooks/`, `tests/`, `skills/`,
   `templates/`) and this repo's entry points if the ticket's Location field touches them.
4. Call the `codebase-indexer` MCP server's `sync_repo` tool once, now, so the Qdrant index
   reflects the current `main` before you investigate anything (normally cheap — it only
   re-embeds changed files, using a manifest file to know what's already indexed):
   `sync_repo(repo_path="<absolute path to YOUR OWN current working directory — run pwd to
   get it, since you're in your own isolated git worktree, not the primary checkout>")`.
   Note: since your worktree is fresh, the gitignored sync manifest won't be present here
   even though it exists in the primary checkout, so this one call may re-embed more than
   the usual "only what changed" — a modest, known cost, not a sign anything's wrong.

## Step 2 — setup
5. `git remote get-url origin` must show Donelle/claude-runway — if not, report
   FAILED immediately, something is wrong with the working directory.
6. `git fetch origin main` — do NOT `git checkout main` here. Confirmed live: inside your
   isolated worktree, checking out `main` fails outright (`fatal: 'main' is already used by
   worktree at '<primary-checkout-path>'`) whenever the primary checkout has `main` checked
   out, which it normally does. You don't need it checked out locally anyway — Step 2's
   `gh issue develop --checkout` step below creates the new branch server-side, off
   whatever GitHub's actual default branch currently is, independent of your local HEAD;
   the `git fetch` here just makes sure your local knowledge of `origin/main`'s tip is
   current for anything you diff against it later (e.g. Step 11's `git diff origin/main...HEAD`).
7. Fetch the issue: `gh issue view <N> -R Donelle/claude-runway --json
   number,title,body,labels,assignees,issueType,state,url`. If state is CLOSED, report
   BLOCKED — don't reopen work on a closed ticket autonomously. Read the body's
   **Location**/**Verdict**/**Suggested fix** sections as a head start, not something to
   re-derive from zero.
8. Set issueType if null (bug label → Bug, enhancement label → Feature, neither → Task):
   `gh issue edit <N> -R Donelle/claude-runway --type <type>`
9. Set assignee if empty (always the current authenticated user — this is what `@me`
   resolves to): `gh issue edit <N> -R Donelle/claude-runway --add-assignee @me`.
   If someone else is already assigned, report BLOCKED — don't take over someone else's
   issue.
10. **Create and check out the branch BEFORE touching any files** — this order matters: in
   this same cycle worked by hand, the fix got implemented directly on `main` before the
   branch existed, and had to be corrected after the fact. Check for an existing linked
   branch first (`gh issue develop <N> -R Donelle/claude-runway --list`); if one
   exists, `git fetch` and check it out instead of creating a new one. Otherwise pick a
   short, specific name yourself (no need to ask — `fix/<slug>` for Bug, `feature/<slug>`
   for Feature/Task, ≤6 words, matching this repo's real branch history style) and run:
   `gh issue develop <N> -R Donelle/claude-runway --name <name> --checkout`
   **If this (or `gh pr checkout`, when picking up an already-open PR for this ticket)
   fails because the branch is already checked out in another worktree** — confirmed live:
   a prior round's worktree for this same ticket can still be holding the branch if it
   wasn't cleaned up. Find it with `git worktree list` (works from ANY worktree — they all
   share the same `.git` metadata, so you don't need the primary checkout's path). **Only
   remove an entry you can positively confirm is this same ticket's own stale artifact —
   the branch being locked elsewhere does not by itself prove that worktree is stale; it
   could belong to an active human or an unrelated concurrent process, and force-removing
   someone else's in-progress work would defeat the entire point of worktree isolation.**
   Treat an entry as safe to remove only if BOTH hold:
   - its path or branch name is clearly this ticket's own (this repo's own primary/branch
     naming, or the harness's `agent-<id>`-style worktree path for a PRIOR attempt at THIS
     issue number — never a worktree whose branch/path you can't tie to this specific
     ticket), AND
   - `git -C <path> status --short` is empty (nothing uncommitted) and any commits it has
     are already reachable from that SAME branch's remote copy — run this WITH `-C <path>`
     (a filesystem path is not a git revision, and the comparison direction matters: it's
     `origin/<branch>..HEAD`, not the reverse), confirmed live:
     `git -C <path> log origin/<branch>..HEAD --oneline` empty means nothing unpushed sits
     there — i.e. its real work, if any, already made it to the remote.
   If either check fails, don't force anything — report BLOCKED with what you found, so a
   human can look at what's actually in that worktree before it's touched. Only once
   confirmed safe: `git worktree remove <stale-path>` (plain, not `--force` — if it still
   complains, that's itself a sign the "nothing uncommitted" check above was wrong, so stop
   and report BLOCKED rather than forcing past it).

11. **Check whether this ticket is already mid-flight before doing anything else** —
    confirmed necessary live: a prior attempt (this run or an earlier one) may have failed
    or stopped partway through, and a fresh subagent has no memory of that. Resuming
    correctly here avoids both a duplicate-PR error and redundant/conflicting
    reimplementation:
    ```bash
    EXISTING_PR=$(gh pr list -R Donelle/claude-runway --head <branch> --state all --json number,state --jq '.[0] // empty')
    ```
    - **No PR, and no commits ahead of `main`** (`git log origin/main..HEAD --oneline` is
      empty — using `origin/main`, not local `main`, since Step 2 deliberately never checks
      local `main` out or updates it in this worktree)
      → nothing was done yet on this branch. Proceed normally to Step 3 (plan and
      implement) below.
    - **No PR, but commits already exist ahead of `main`** → a prior attempt implemented
      (and maybe pushed) before failing. Do NOT blindly redo the implementation. Read
      what's already there (`git diff origin/main...HEAD`), bootstrap `.venv/` per Step 3
      below if it isn't already there in this worktree, then re-run
      `.venv/bin/python -m unittest discover -s tests` and `.venv/bin/mypy libs tools
      hooks` against the current branch state: if both are clean and the diff looks like
      it genuinely addresses the issue,
      treat implementation as done — push if not already pushed, write/complete the
      `.plans/<N>-*.md` artifacts if missing, and go straight to PR creation (within Step
      3, the `gh pr create` step). If tests fail or the diff looks incomplete/wrong,
      continue/fix the existing work in place rather than starting over from a blank slate.
    - **An OPEN PR already exists for this branch** → the ticket is already past
      implementation and into (or done with) the review cycle. Capture that PR's number
      and skip straight to Step 4 (the review-feedback loop) using it — do not touch the
      rest of Step 3 (implement/PR-create) at all; calling `gh pr create` again on a
      branch that already has an open PR errors. **Before starting Step 4's round loop,
      pre-seed `seen_ids` with EVERYTHING that already exists right now, MINUS the ids
      belonging to any still-unresolved review thread** — not just the inline-comment
      domain. A GraphQL `reviewThreads` query only returns inline comment `databaseId`s,
      not review-envelope ids (from `pulls/.../reviews`) or general issue-comment ids
      (from `issues/.../comments`) — pre-seeding from that query alone leaves those other
      two domains unseeded, so an already-fully-handled review envelope or a general
      "suppressed comment" reply from a prior attempt would look "new" again on resume and
      get redundantly reprocessed. Self-authored ids don't need separate handling here —
      Step 4's poll filters those out by login on every round regardless of resume state,
      including this pre-seed. **Every id gathered anywhere in this whole flow — here and
      in Step 4's own poll — is prefixed with its domain (`comment:`, `review:`,
      `issue:`).** These three REST resource types are backed by separate database
      sequences, not one shared id space (confirmed live: comment ids and review ids on
      this very PR sit in completely different numeric ranges) — bare numeric ids could
      coincidentally collide across domains as both counters grow over time, and a
      collision would silently make one resource look "already seen" because an unrelated
      resource in a different domain happened to reuse its number. The prefix costs
      nothing and removes the possibility entirely.
      ```bash
      ALL_CURRENT_IDS=$( { gh api repos/Donelle/claude-runway/pulls/<PR>/comments --paginate --jq '.[] | "comment:" + (.id|tostring)'
                            gh api repos/Donelle/claude-runway/pulls/<PR>/reviews  --paginate --jq '.[] | "review:" + (.id|tostring)'
                            gh api repos/Donelle/claude-runway/issues/<PR>/comments --paginate --jq '.[] | "issue:" + (.id|tostring)'
                          ; } 2>/dev/null | sort -u )
      UNRESOLVED_IDS=$(gh api graphql -f query='query { repository(owner: "Donelle", name: "claude-runway") { pullRequest(number: <PR>) { reviewThreads(first: 100) { nodes { isResolved comments(first: 10) { nodes { databaseId } } } } } } }' \
        --jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | .comments.nodes[] | "comment:" + (.databaseId|tostring)' | sort -u)
      SEEN_IDS=$(comm -23 <(echo "$ALL_CURRENT_IDS") <(echo "$UNRESOLVED_IDS"))
      ```
      This treats everything that already exists as handled UNLESS it's still sitting in
      an open thread. **But an empty `UNRESOLVED_IDS` does NOT by itself prove the current
      commit was actually reviewed** — two real gaps: resuming right after a prior
      `FAILED`-on-timeout attempt (Step 19) can mean NO review has landed for this commit
      at all yet, and "Suppressed comments" (a review body's own text, not a separate
      GraphQL thread) never gets a thread id in the first place, so it can't show up as
      "unresolved" here even when genuinely unaddressed. Resolve this before trusting
      `UNRESOLVED_IDS`:
      ```bash
      HEAD_SHA=$(git rev-parse HEAD)
      EXISTING_REVIEW=$(gh api repos/Donelle/claude-runway/pulls/<PR>/reviews --paginate \
        --jq ".[] | select(.user.login == \"copilot-pull-request-reviewer[bot]\" and .commit_id == \"$HEAD_SHA\") | .id" | head -1)
      ```
      - `EXISTING_REVIEW` is empty → no review exists yet for the current commit at all
        (this is exactly the case a resume-after-timeout produces). Do NOT skip to merge.
        Run Step 19's poll (same HEAD-SHA-matched form) to wait for one, then continue
        through Steps 20–25 exactly like any other round — resuming doesn't change this
        part, only the pre-seeding above does.
      - `EXISTING_REVIEW` is non-empty → a review of the current commit does exist. Read
        ITS body specifically for a "Suppressed comments" section (the same content-check
        Step 21 already does for any new review) — if present and you can't positively
        confirm from your own prior reply/comment history that it was already addressed,
        treat it as a new finding and jump into Step 21 with it. Only if there's genuinely
        nothing outstanding (empty `UNRESOLVED_IDS` AND no unaddressed suppressed content
        in `EXISTING_REVIEW`) → skip Step 4's poll, go straight to Step 5 (merge).
      When there IS something to process either way, treat it as this round's `NEW_IDS`
      and jump directly into Step 21's content-reading/verification logic (fix
        or decline each, per Step 22, then reply/resolve per Steps 23–24, then Step 25
        decides the next round exactly as it would for any other round).
    - **A MERGED or CLOSED PR already exists for this branch, but the issue is still
      OPEN** → this shouldn't happen if `Fixes #N` worked correctly, but if it does, don't
      guess at why — report BLOCKED with the PR number and its state, so a human can sort
      out whether it needs reopening, a new fix, or just closing the issue manually.

## Step 3 — plan and implement
12. Read whatever specific file(s) the issue's Location field names (README was already
    read in Step 1). Write a plan to `.plans/<N>-plan.md` (goal, what's already known from
    the issue body, approach, files to touch, test strategy) — do NOT wait for approval,
    just proceed once it's written. `.plans/` is globally gitignored on this machine; never
    `git add` anything under it.
13. Implement the fix, matching the style of the surrounding code (explicit comments
    explaining *why*, not just *what* — this codebase's established density).
14. Write or update unit tests in the matching `tests/test_*.py`, following that file's
    existing patterns (e.g. the `hook.compress = _fake_compress` monkeypatch style already
    used in `tests/test_compress_bash_output.py` for anything touching LM Studio, so tests
    need no live model). If sizing test fixtures relative to a threshold constant, read the
    constant from the module dynamically (e.g. `hook.THRESHOLD`) rather than hardcoding its
    documented default — this repo's own dogfood shell overrides
    `CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS` to 4000, not the code's 2000 default, and a
    hardcoded assumption will fail confusingly in this exact environment.
15. **`.venv/` won't exist yet in your worktree — check and bootstrap it before running
    anything.** Confirmed live: `.venv/` is gitignored, and a linked git worktree only
    ever contains tracked files, so every fresh worktree starts with no venv at all, not
    a stale-but-present one:
    ```bash
    test -d .venv || uv venv --python 3.12
    uv pip install -r requirements.txt --index-url https://pypi.org/simple
    uv pip install -r requirements-dev.txt --index-url https://pypi.org/simple
    ```
    (The second/third lines are cheap no-ops if already satisfied, so it's fine to always
    run them rather than trying to detect exactly what's missing.) Then run and require
    both clean before proceeding:
    ```
    .venv/bin/python -m unittest discover -s tests
    .venv/bin/mypy libs tools hooks
    ```
    If you cannot get both clean after reasonable effort, report FAILED with what's
    failing and why, rather than committing broken code.
16. Commit (no `Fixes #N` in the commit message — that goes in the PR body only, or
    GitHub's squash-merge won't attribute it right) and push:
    `git add <files> && git commit -m "..." && git push -u origin <branch>`
17. `gh pr create --base main --head <branch> --title "<tightened summary>" --body
    "<bulleted summary of the change>\n\nFixes #<N>\n\nSee
    https://github.com/Donelle/claude-runway/issues/<N>\n\n---\n🤖 Opened and driven
    autonomously via the \`/my-gh-autowork\` skill (plan → implement → test →
    review-feedback loop → merge) — no human wrote or reviewed this before it was pushed;
    verify accordingly."`. **Capture the PR number right here, reliably** — either parse it
    from the URL `gh pr create` prints to stdout (`.../pull/<PR>`), or immediately run
    `gh pr view <branch> -R Donelle/claude-runway --json number --jq .number`. Do
    NOT try to (re)discover the PR later via a text/body search (e.g. `gh pr list --search
    "<N> in:body"`) — confirmed live during this skill's own testing to false-positive: a
    search for issue #44 matched an unrelated, already-merged PR from weeks earlier whose
    body just happened to contain the substring "44".
18. Write `.plans/<N>-research.md` (what you actually read/confirmed, with specific
    file:line references) and `.plans/<N>-validate.md` (real `git diff --stat origin/main`
    — NOT bare `main`, which this worktree deliberately never checks out or updates, so it
    can be arbitrarily stale and report unrelated commits in the diff — plus real
    test output with pass counts, plan-vs-implementation comparison) — reuse the plan file
    from step 12, update it if implementation diverged. Post all three as separate issue
    comments:
    ```
    gh issue comment <N> -R Donelle/claude-runway --body-file .plans/<N>-research.md
    gh issue comment <N> -R Donelle/claude-runway --body-file .plans/<N>-plan.md
    gh issue comment <N> -R Donelle/claude-runway --body-file .plans/<N>-validate.md
    ```

## Step 4 — review-feedback loop (verify before acting, capped at 5 rounds total)
This repo's "PR Re-review" ruleset (`copilot_code_review: {review_on_push: true}`,
confirmed active) means EVERY push you make to this PR — the one that just opened it, or
any later fix — automatically triggers a fresh Copilot review with zero manual action. So
this whole loop runs inside this one call; there's no orchestrator hand-off and no human
touchpoint needed.

Maintain a `seen_ids` set locally (starts empty on a fresh ticket; see Step 11 above for
what to pre-seed it with when resuming an already-open PR) across the rounds below — every
inline comment id, review-envelope id, and general-issue-comment id you've already looked
at goes in it, so "new" always means "not in this set yet."

**Every poll below excludes YOUR OWN comments/reviews up front** — confirmed necessary
live: without this, a later round's poll sees your own prior reply (or the empty
self-review artifact `gh` can leave behind from posting one) as "new feedback" and has to
waste a round figuring out it's not a real finding. Get your own login once and filter on
it, not on `seen_ids` membership (a self-authored id would still need filtering even if,
say, `seen_ids` bookkeeping had a gap somewhere):
```bash
ME=$(gh api user --jq .login)
```

**Round loop (repeat from here; hard cap 5 rounds — if you're still going after 5, report
BLOCKED with the round history rather than continuing indefinitely):**

Initialize `ROUND=0` and `ROUND_HISTORY=""` (empty) right here, once, before the first
iteration. **This counter is the actual enforcement mechanism for the cap — the prose above
alone is not.** Confirmed live on issue #48/PR #113: a version of this loop with only prose
("hard cap 5 rounds") and Step 25's "same concern resurfaces twice" check as its sole stop
condition ran to ~11 real review rounds before a human had to intervene manually, because
each round's finding was genuinely new and narrower than the last (Windows-path quoting in a
printed reminder, secret redaction in a `--dry-run` preview, write-order sequencing on a rare
partial-failure path) — never the literal same concern twice, so the only check that existed
never tripped. `ROUND` is incremented once per round that has at least one actionable finding
to fix (i.e. right at the top of Step 22, before touching any code) — a round where Step 21
shortcuts straight to merge (nothing actionable) does NOT increment it, since no fix cycle
happens there.

19. **Wait for a NEW COPILOT review OF THE CURRENT HEAD specifically — author alone isn't
    enough.** Pulling comments too early just gets you last round's state again, and the
    arrival condition has to be Copilot's own review of THIS round's actual commit, not
    just any review from that author: if this round pushed more than one fix commit
    (possible — Step 22 can push once per finding, and each push independently triggers
    `review_on_push`), an earlier, already-queued review for a PRIOR commit could arrive
    first and get mistaken for coverage of the latest one, wrongly authorizing a merge
    while the true latest push goes unreviewed. Capture the current HEAD SHA right before
    polling, and require the arriving review's own `commit_id` to match it — a stray human
    comment or unrelated bot can't satisfy this either, since it isn't a review at all:
    ```bash
    HEAD_SHA=$(git rev-parse HEAD)
    for i in $(seq 1 36); do
      NEW_COPILOT_IDS=$(gh api repos/Donelle/claude-runway/pulls/<PR>/reviews --paginate \
          --jq ".[] | select(.user.login == \"copilot-pull-request-reviewer[bot]\" and .commit_id == \"$HEAD_SHA\") | \"review:\" + (.id|tostring)" 2>/dev/null \
        | sort -u | comm -23 - <(printf '%s\n' $SEEN_IDS | sort -u))
      if [ -n "$NEW_COPILOT_IDS" ]; then
        echo "Copilot review of $HEAD_SHA arrived: $NEW_COPILOT_IDS"
        break
      fi
      sleep 15
    done
    ```
    **36 iterations, not 40 — the loop's own sleep budget must leave real headroom under the
    wrapping Bash tool call's hard 600000ms timeout, not just equal it.** 40×15s of pure
    `sleep` alone already totals exactly 600000ms with zero margin left for the 40 `gh api`
    calls interleaved between those sleeps, each of which takes real, nonzero wall-clock
    time; the Bash tool would then kill the command mid-loop on any run that genuinely needs
    the full timeout path, before the shell script ever reaches its own `done` and before the
    subagent could print the required `FAILED` outcome line — silently recreating the exact
    stalled/no-terminal-output problem this whole section exists to prevent (flagged in PR
    review on #126, verified independently: `40 * 15 == 600` exactly). 36×15s = 540s (9
    minutes) of guaranteed sleep, leaving a full 60 real seconds of headroom under the cap for
    the loop's own `gh api` calls plus general shell/tool overhead — comfortably safe for
    calls that normally take well under a second each, and the loop can still always finish
    (whether by finding a match or exhausting its iterations) before the Bash tool's own
    timeout would ever fire.
    (Same `--jq` constraint as elsewhere in this file — it takes exactly one argument, so
    the shell variable is interpolated directly into the expression string, not passed via
    jq's own `--arg` flag, which `gh api` doesn't support regardless of position;
    re-verified live for this exact form before writing it down here.) ~9 minutes, not 5 —
    confirmed live that a real re-review can take longer than 5 minutes (observed ~6
    minutes working issue #45/PR #110), so a 5-minute window risked timing out and merging
    before Copilot's actual review arrived.
    (`gh api --jq` takes exactly one argument — it does NOT accept jq's own `--arg` flag for
    passing variables in; confirmed live, `gh api ... --jq --arg me "$ME" '...'` errors with
    "accepts 1 arg(s), received 4". Interpolate the shell variable directly into the
    expression string instead, as above.)
    **If the loop times out with nothing new, that is NOT the same as "clean" — do not
    proceed to merge.** Absence of a new id after ~9 minutes doesn't prove the review is
    clean; it may just mean the review hasn't arrived yet, and merging in that ambiguous
    state risks shipping code no review actually covered. Report
    `OUTCOME: FAILED (issue #<N>) — Copilot's review did not arrive within 9 minutes on
    PR #<PR>; can't confirm the PR is clean` and stop — a human (or a later re-run of this
    same skill, which Step 11's resume logic will pick straight back up from this PR) can
    check whether the review is just slow or something's actually stuck. (A `gh api
    ... --paginate` call's output may look
    summarized/compressed rather than raw JSON — this repo's own `PostToolUse` compression
    hook did that. This pattern already reads via `--jq`/`jq`, which is on that hook's
    byte-exact exemption list, so it reaches you unmodified either way.)
20. The poll above only reaches here when a new Copilot review specifically arrived — a
    genuine timeout is Step 19's `FAILED` path above, not this one; there's no "nothing
    new, not a timeout" case, so this step doesn't need one either. NOW gather the full
    picture across all three endpoints, non-self. **Check each endpoint's own success —
    don't blanket-suppress errors across all three.** A silently swallowed failure on just
    one endpoint (a transient network blip, a rate limit) would make a partial fetch look
    complete, and `NEW_IDS` could then omit a real finding that a merge decision below
    would never know it was missing:
    ```bash
    FETCH_FAILED=0
    COMMENT_IDS=$(gh api repos/Donelle/claude-runway/pulls/<PR>/comments --paginate --jq ".[] | select(.user.login != \"$ME\") | \"comment:\" + (.id|tostring)") || FETCH_FAILED=1
    REVIEW_IDS=$(gh api repos/Donelle/claude-runway/pulls/<PR>/reviews --paginate --jq ".[] | select(.user.login != \"$ME\") | \"review:\" + (.id|tostring)") || FETCH_FAILED=1
    ISSUE_COMMENT_IDS=$(gh api repos/Donelle/claude-runway/issues/<PR>/comments --paginate --jq ".[] | select(.user.login != \"$ME\") | \"issue:\" + (.id|tostring)") || FETCH_FAILED=1
    if [ "$FETCH_FAILED" -eq 1 ]; then
      echo "One or more feedback endpoints failed to fetch — do not treat this as a complete picture."
    fi
    ALL_IDS=$(printf '%s\n%s\n%s\n' "$COMMENT_IDS" "$REVIEW_IDS" "$ISSUE_COMMENT_IDS" | sort -u)
    NEW_IDS=$(comm -23 <(echo "$ALL_IDS") <(printf '%s\n' $SEEN_IDS | sort -u))
    ```
    If `FETCH_FAILED` is 1, report `OUTCOME: FAILED (issue #<N>) — could not reliably fetch
    PR #<PR>'s feedback (one or more of the comments/reviews/issue-comments endpoints
    failed)` rather than proceeding on a picture you know is incomplete. **The
    `comment:`/`review:`/`issue:` prefix on each entry in `NEW_IDS` tells you which of the
    three endpoints it came from** — strip the prefix to get the real numeric id when you
    need to act on one directly (e.g. replying to a specific inline comment needs its bare
    id in the URL path; the prefix itself is only for `seen_ids` set membership, never part
    of an actual API call). This can include
    more than just the Copilot review itself (e.g. a human comment that
    happened to land around the same time) — fetch full detail for everything in
    `NEW_IDS` and add it all to `seen_ids`.
21. **"Something new" can still mean a review with zero actionable comments** — every
    Copilot review, even a fully clean one, is its own new review-envelope id (this is
    expected: the "PR Re-review" ruleset fires on every push regardless of what the diff
    contains), so a genuinely clean re-review still lands here as "something new," not as
    Step 19's timeout case. Read the actual content: if this round's new id(s) contain no
    concrete finding at all (no inline comments, no "Suppressed comments" section, just a
    summary like "reviewed N files, generated no comments") → that's clean too. Go to Step
    5 (merge) directly — do NOT loop back to Step 19 for another poll-wait just because an
    id was technically new; doing so burns a round of the 5-round cap for nothing, which
    matters on a ticket that genuinely needs several real rounds. Only proceed to Step 22
    below when there's an actual finding to verify.
22. **First, initialize defensively if needed, then increment `ROUND` and check the cap —
    before touching any code this round.** If `ROUND`/`ROUND_HISTORY` aren't already set,
    set them now (`ROUND=0`, `ROUND_HISTORY=""`) before proceeding — Step 11's resume-an-
    open-PR shortcut jumps directly into this step from Step 21 without ever passing
    through the round loop's own preamble where these are normally initialized, so relying
    solely on that preamble leaves them unset on a resumed ticket and the cap silently
    unenforced. `ROUND=$((ROUND + 1))`. If `ROUND` is now greater than 5, STOP: do not fix, reply to, or
    reproduce anything from this round's findings — but DO first append a one-line summary
    of what this round's new finding(s) actually ARE (straight from `NEW_IDS`'s content, not
    from investigating them) to `ROUND_HISTORY`. Skipping this would leave the human reading
    the BLOCKED report with only the five already-handled rounds and no idea what the
    unresolved round-6 concern actually is — the exact thing they need to assess. Print
    `ROUND_HISTORY` as part of your PRECEDING summary (one line per round is fine there —
    that text isn't constrained to a single line, per the Final report section below), then
    end with a single-line `OUTCOME: BLOCKED (issue #<N>) — hit the 5-round review-feedback
    cap on PR #<PR>, see round history above` as your literal last line. **Do NOT
    interpolate `ROUND_HISTORY` directly into the outcome line itself** — it accumulates one
    entry per round, so by round 6 it can span multiple lines, which would break the Final
    report section's "literal last line" one-line contract the orchestrator relies on to
    recognize the outcome. A finding surviving to round 6 is itself a signal this ticket
    needs human judgment — either the fix is generating new surface area as fast as it
    closes it, or the review is holding it to a standard beyond the ticket's actual scope;
    both are reasons to stop and hand it back, not to keep going. If `ROUND` is still ≤ 5,
    proceed below — once each finding this round is actually handled (reproduced/fixed or
    declined, per the rest of this step), append a one-line summary of it to
    `ROUND_HISTORY` (for use in either this report or the final one):

    **For every NEW finding that makes a concrete, checkable claim (a bug, an edge case, a
    security issue) — actually reproduce it before touching anything.** Write a tiny
    script or direct function call exercising the actual mechanism the reviewer describes,
    not just re-reading the code and agreeing it sounds plausible (this repo's own live
    testing of this skill caught a real bug this way — an 8-hex-char/32-bit hash suffix
    genuinely can collide, confirmed by brute-forcing to a real collision, not just
    theorizing about the birthday bound). Two outcomes only:
    - **Reproduces** → fix it, add a regression test, re-run the full test suite + mypy,
      commit, push (this auto-triggers the next review). **Do NOT jump back to the poll
      immediately after this push** — a reproduced finding still needs the same reply and
      thread-resolution as a declined one (Steps 23–24 below apply to every finding this
      round, reproduced or declined, not just declined ones); skipping straight back to
      polling here would leave it replied-to-never and its thread unresolved. Continue to
      the next finding (if any), then to Steps 23–25 once every finding this round has
      been handled — Step 25 is the one place that decides whether to loop back to Step
      19 (the wait/poll item — NOT Step 18, the one-time PR/artifact-posting step; looping
      back there would repost the three `.plans/<N>-*.md` issue comments every round) or
      go to merge.
    - **Doesn't reproduce** → don't change the code. Reply with the specific reasoning or
      test evidence for why — never silently ignore a finding, never defer to a reviewer
      by default just because it's a reviewer (bot or human).
    Style/best-practice suggestions with no checkable failure mode don't need a repro —
    judgment call directly.
23. **Reply to every NEW finding, confirmed or declined — but NEVER mention `@copilot` in
    any reply, for any reason, on this repo.** Not for pushback, not for a fix
    confirmation, not to ask for re-review (that's automatic now — see the ruleset note
    above — so there is never a reason to ask anyone, bot or human, for one). This was a
    conditional rule until it was tried and failed identically on three separate occasions
    — PR #73 (four tagged pushback replies, ~20 auto-retried error comments), PR #84 (two
    tagged re-review requests, 12 error comments), and PR #107 (2026-08-25: a tagged reply
    triggered a self-perpetuating bot error loop needing two cleanup passes, ~24 comments
    total). Mentioning `@copilot` anywhere invokes GitHub's full coding agent, not a
    lightweight chat reply, and that invocation itself errors out on this repo.
    - Reply to an inline comment (note the `<PR>` segment — omitting it 404s, a real
      mistake made working this cycle by hand):
      `gh api repos/Donelle/claude-runway/pulls/<PR>/comments/<comment_id>/replies --method POST -f body="..."`
    - A finding buried inside a review body's "Suppressed comments" section (no standalone
      comment id) gets a general PR comment instead:
      `gh issue comment <PR> -R Donelle/claude-runway --body "..."`
    - If an error-comment spam loop happens anyway (from any source), stop immediately —
      it's a GitHub-side backend failure, not fixable from this side — and clean up:
      `gh api repos/Donelle/claude-runway/issues/comments/<id> --method DELETE`
      per comment, then continue with the underlying finding (already resolved either way).
24. Resolve addressed review threads (GraphQL only — REST can't do this):
    ```
    gh api graphql -f query='query { repository(owner: "Donelle", name: "claude-runway") { pullRequest(number: <PR>) { reviewThreads(first: 100) { nodes { id isResolved comments(first: 1) { nodes { databaseId path body } } } } } } }'
    ```
    then for each unresolved thread whose finding you addressed this round:
    ```
    gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "<node-id>"}) { thread { isResolved } } }'
    ```
    Some threads may already show `isResolved: true` on their own (Copilot has been
    observed to self-resolve after seeing a fix land) — skip those, don't error on them.
25. **Only loop back to Step 19 if this round actually pushed a fix.** `review_on_push` is
    the ONLY re-review trigger this repo has — if every finding this round was declined
    (no code change, no push), nothing will ever trigger a fresh review to check against,
    so looping back to poll would just burn ~9 minutes and land on Step 19's
    timeout-`FAILED` path for no reason. In that case, go straight to Step 5 (merge)
    instead — there's nothing further a review-loop round could discover once nothing
    changed. If at least one finding WAS fixed and pushed this round, loop back to Step 19
    to wait for the review that push triggers. **If the SAME concern resurfaces on that
    next review after you've already replied to it with reasoning or a fix once, that's a
    stop condition — report BLOCKED with both replies quoted, rather than looping a third
    time on the same point.**

## Step 5 — merge
26. `gh pr view <PR> -R Donelle/claude-runway --json mergeable,mergeStateStatus,statusCheckRollup`.
    All required checks must be SUCCESS. The `prjiralink` check may show `ACTION_REQUIRED`
    — that's known non-blocking on this repo, ignore it. If any other check is failing and
    you can't fix it (not the code you touched, e.g. an unrelated flake), report FAILED
    with the check name and its log URL rather than force-merging. If the merge itself is
    rejected by a repository rule, report FAILED with the exact error — don't try to work
    around a repo rule autonomously.
27. `gh pr merge <PR> -R Donelle/claude-runway --squash --delete-branch`
28. Confirm: `gh pr view <PR> --json state,mergedAt,mergeCommit` shows MERGED, and
    `gh issue view <N> --json state` shows CLOSED (auto-closed via `Fixes #N`).
29. Do NOT `git checkout main` here — same reason as Step 2: it fails outright
    (`fatal: 'main' is already used by worktree...`) whenever the primary checkout has
    `main` checked out, which it normally does; git only allows one worktree per branch at
    a time. Your worktree gets discarded once this call ends anyway, so there's nothing to
    "leave clean" locally — just confirm the merge actually reached the remote:
    `git fetch origin main && git log origin/main -1` should show your merge commit.

## Final report (your literal last line of output)
Before the outcome line, give a short summary: files changed, tests added, findings
verified/declined per round, how many review rounds it took, any deviation from the plan
and why. Then end with EXACTLY one of these three line shapes:
```
OUTCOME: MERGED (issue #<n>, PR #<pr>)
OUTCOME: BLOCKED (issue #<n>) — <one-line reason>
OUTCOME: FAILED (issue #<n>) — <one-line reason>
```
(A Step 0 call that found no qualifying ticket at all reports `OUTCOME: DONE` instead — no
issue number.)
````
