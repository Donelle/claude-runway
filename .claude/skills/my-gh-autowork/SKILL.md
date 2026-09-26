---
name: my-gh-autowork
description: "Autonomously work claude-runway's open bug/enhancement backlog end-to-end (plan, code, PR, verified multi-round review-feedback loop, merge) by delegating each ticket to an isolated subagent — zero approval gates, one ticket at a time, stops and reports rather than guessing on anything genuinely ambiguous. Refuses to run at all outside a verified dogfood checkout."
---

# Skill: my-gh-autowork

Fully autonomous version of the `/my-gh-code-it` → `/my-gh-pr` → `/my-gh-pr-feedback` → merge cycle already used by hand on this repo. Where those three skills pause for human approval (plan review, branch-name confirmation), this one doesn't — it's for the case where the user has explicitly authorized working through tickets without per-step check-ins. It doesn't replace those three skills' *content*; it packages their combined, battle-tested procedure (plus the merge step, which none of them do) into one subagent prompt run via the Agent tool.

**Zero required human touchpoints, confirmed live.** This repo has a repository ruleset ("PR Re-review", `gh api repos/Donelle/claude-runway/rulesets/23376382`) with `copilot_code_review: {review_on_push: true}` active — Copilot automatically re-reviews on EVERY push to a PR targeting the default branch, no manual re-request and no `@copilot` mention needed. Confirmed live (2026-08-25, working issue #45/PR #110): a fix push triggered a fresh Copilot review within ~6 minutes with zero action from anyone. This means the entire review-feedback loop — including every round after the first — can run inside a single subagent call: push, wait, check, fix-and-push-again if needed, repeat, merge. An earlier version of this skill split this into orchestrator-driven `start`/`continue` modes with a mandatory human ping between rounds, because at the time a re-review genuinely required a manual action that risked the `@copilot`-mention failure mode if done wrong. That's no longer true on this repo and the split has been removed — don't reintroduce it without first re-confirming `review_on_push` is still active (`gh api repos/Donelle/claude-runway/rulesets` should list a ruleset with that rule; if it's gone, the old multi-mode design in this file's git history is the fallback).

**Caveat: this claim is about the review-feedback loop specifically, not an absolute guarantee against every touchpoint.** A separate, server-side "auto mode classifier" — independent of anything in `.claude/settings.json` and outside this repo's control — rejected the orchestrator's own `Agent` call outright on 2026-09-19, before the subagent it would have spawned ran a single tool call, with no reasoning given beyond "judged this action dangerous." A follow-up attempt to self-correct by editing `.claude/settings.json` to add the permission rule the denial message itself suggested was separately blocked as `[Self-Modification]`. The confirmed (but not confirmed-*causal* — see Step 0's new preflight check below and `templates/settings.json.template`'s `_permissions_note`) workaround was a human manually adding the SPECIFIC narrow rules `Bash(git fetch *)`/`Bash(git push *)`/`Bash(gh pr *)`/`Bash(gh api *)` to `.claude/settings.json`; the identical `Agent` call then succeeded on retry. **Do not broaden this to a wildcard like `Bash(git:*)`/`Bash(gh:*)`/`Bash(uv:*)`** — none of those were actually confirmed, and a blanket `git`/`uv` wildcard is meaningfully worse: it adds entire additional command families at once, not just extra flags within an already-allowed subcommand (`git -c alias.x='!<any shell command>' x` runs arbitrary shell through git's own alias mechanism, verified live; `uv run <script>` executes arbitrary code by design). Extend the narrow, per-subcommand style to cover whatever additional `git`/`gh` subcommands the rest of this skill needs (see `templates/settings.json.template`'s `_permissions_note` for the fuller suggested list) rather than reaching for a wildcard. **This is a reduction in exposed surface, not an elimination of code-execution risk** — autonomous development inherently requires code-execution authority, and even the narrow, per-subcommand rules above still permit it through legitimate flags (`git fetch --upload-pack=<cmd>`/`git push --receive-pack=<cmd>` run an arbitrary local command regardless of subcommand scoping — verified live; `.venv/bin/python:*` trivially allows `python -c '<any code>'`). The real containment is running this only in a checkout/environment you already trust, not the permission syntax — see the template note for the full account.

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

   **Baseline Bash permission preflight — run this ONLY after `DOGFOODING: yes`, before the first `Agent` call.** This does not, and cannot, guarantee the server-side auto mode classifier documented above won't reject the `Agent` call anyway (its reasoning is opaque and the one observed data point doesn't confirm causation) — but a missing baseline permission rule is a known, checkable precondition of the one workaround that has actually worked, so ruling it out first turns a possible opaque classifier denial into an actionable, specific message instead. Claude Code merges `permissions.allow` from BOTH the project's own `.claude/settings.json` and the user's `~/.claude/settings.json` — check both, since the working example that was observed lived only in the user-level file, not a project one:
   ```bash
   jq -e '(.permissions.allow // []) | any(test("^Bash[(](git|gh|uv)[ :)]|^Bash[(][.]venv/(bin|Scripts)/"))' \
     .claude/settings.json >/dev/null 2>&1 && echo "PROJECT: yes" || echo "PROJECT: no"
   jq -e '(.permissions.allow // []) | any(test("^Bash[(](git|gh|uv)[ :)]|^Bash[(][.]venv/(bin|Scripts)/"))' \
     ~/.claude/settings.json >/dev/null 2>&1 && echo "USER: yes" || echo "USER: no"
   ```
   The `[ :)]` right after `git`/`gh`/`uv` is load-bearing, not decorative — without it, `test("^Bash\\((git|gh|uv|\\.venv/bin/)")` (an earlier version of this check) matched unrelated tool names that merely start with the same letters, e.g. `Bash(github-cli:*)`, `Bash(ghastly:*)`, `Bash(uvicorn:*)` all false-positived as `yes` (confirmed live against exactly these three), letting the preflight silently proceed with no real `git`/`gh`/`uv` access at all — the opposite of what this check exists to catch. Requiring a space/colon/close-paren immediately after the command name rules those out while still matching `Bash(git fetch:*)`/`Bash(gh pr:*)`/`Bash(uv venv:*)`/etc. The venv pattern matches both `bin/` (Linux/macOS) and `Scripts/` (Windows) — a Windows-only setup using `"Bash(.venv/Scripts/python *)"` would have incorrectly printed `no` with the earlier `.venv/bin/`-only pattern (issue #230).
   If BOTH print `no`, **STOP** — report to the user that no `git`/`gh`/`uv`/`.venv` Bash allow rule was found in either settings file, point them at `templates/settings.json.template`'s `_permissions_note` for the suggested NARROW, per-subcommand patterns to add (`Bash(git fetch:*)`, `Bash(gh pr:*)`, etc. — never a blanket `Bash(git:*)`/`Bash(gh:*)`/`Bash(uv:*)` wildcard, which is an arbitrary-command-execution bypass via git aliases/`uv run`, not just a broader convenience), and do not spawn any subagent yet. This detection regex intentionally accepts either narrow or broad existing rules (it only checks "is there *something* here already," not "is it appropriately scoped") — it's a precondition check, not an endorsement of whatever pattern happens to already be present, and not proof the classifier will allow the call either way.

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
      → `tool_calls`), then call the `record_metric` tool (from the `local-compress` MCP
      server — the same server that hosts `get_metrics`/`savings_summary`):
      ```
      record_metric(
        metric_id="autowork",
        event_type=<"ticket_" + outcome lowercased: "ticket_merged", "ticket_blocked", or "ticket_failed">,
        value=1.0,
        metadata=<JSON string: {"issue": n, "pr": pr_or_null, "outcome": "...",
                    "model": "...", "rounds": N, "manual_interventions": N,
                    "wall_clock_s": N, "subagent_tokens": N, "tool_calls": N,
                    "findings": [{"category": "...", "reproduced": bool, "fixed": bool}]}>,
        session_id=f"issue-{n}-{HHMMSS}"
      )
      ```
      **`{HHMMSS}` in the `session_id` is not decorative — it keeps two attempts at the
      SAME issue on the SAME calendar day distinguishable as separate rows** (issue #210:
      the shared metrics store is append-only, so two calls never collide the way
      `compact_store`'s upsert-on-`(project, label, date)` did — every call creates a new
      row regardless; the per-attempt timestamp is stored as `session_id` in the raw table
      for future per-attempt filtering, though current read tools (`get_metrics`/
      `/my-metrics`) aggregate by `metric_id` only and do not yet expose a `session_id`
      filter — that's a separate follow-up).
      `manual_interventions` counts any `SendMessage` resume THIS orchestrator had to send
      this same subagent to get a compliant final report (0 for a clean single-call
      ticket — see ticket #1's stalling incident in this skill's own commit history for
      why that field exists).

      **This logging step is best-effort and additive — it must never change what the
      outer 2b/2c control flow does next, only whether a metrics entry got recorded.**
      `record_metric` returns either `"OK"` or a string starting with `"Error:"` — treat
      either a raised exception OR an `"Error:"` return as "logging failed," note it, and
      move on to whatever 2b/2c already say to do for this ticket's actual outcome.
      Concretely: for a `MERGED` result in an `all` run, still continue to the next ticket
      even if logging failed; for a `BLOCKED`/`FAILED` result, still **stop the whole run
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
`enhancement` that has no assignee at all, is NOT labeled `blocked` or `theme-design`, and
whose number is NOT in this already-attempted list this run: {ATTEMPTED_LIST}. The
`blocked`/`theme-design` exclusion is deliberate: a `blocked` issue is waiting on something
outside this skill's control (an upstream fix, a human decision), and a `theme-design`
issue is a pre-implementation design/brainstorm doc, not a ticket with a concrete fix to
implement — auto-picking either would either stall the run or produce the wrong kind of
output. IMPORTANT: `gh issue list` silently truncates to 30 results with no `--limit` flag
— always pass one. Also note two labels do NOT OR together via repeated `--label` flags
(that ANDs, requiring both labels on the same issue); use `--search` for OR, and GitHub's
search syntax negates a qualifier with a leading `-` (`-label:X` excludes issues carrying
that label). `no:assignee` (issue #240) matches only issues with ZERO assignees — not "not
assigned to me"; any assignee at all, including yourself from an earlier attempt, excludes
an issue from auto-pick now:
  gh issue list -R Donelle/claude-runway --state open \
    --search "label:bug,enhancement -label:blocked -label:theme-design no:assignee" \
    --limit 200 --json number,title,labels,assignees
Sort by priority label (priority-p1 > p2 > p3; unlabeled sorts last), then by issue number
ascending as a tiebreaker; bug and enhancement are not otherwise prioritized relative to
each other.

**Then, walking that sorted list in order, skip any candidate that already has a linked
branch (issue #240)** — an empty `assignees` field alone doesn't prove nobody has started
on it; a branch can exist without an assignee (e.g. someone ran `gh issue develop` without
ever running `gh issue edit --add-assignee`). For each candidate, in order:
  gh issue develop <candidate> -R Donelle/claude-runway --list
A printed branch name means this candidate is already spoken for — move to the next
candidate WITHOUT adding this number to `attempted` (it was never picked, so it isn't a
worked-and-failed ticket the way a BLOCKED/FAILED outcome's ticket would be). Nothing
printed means this candidate has no assignee and no branch — pick it and proceed to Step 1
below. (This walk only checks for a branch, not a PR, per candidate, to avoid an extra API
call per candidate that gets skipped anyway — Step 8 below re-checks both branch AND PR for
whichever ticket actually gets picked, as the authoritative gate.)

If nothing qualifies at all, or every remaining candidate after this walk turns out to
already be spoken for, output exactly `OUTCOME: DONE` as your entire response and stop — do
not proceed to any step below. **The label-based exclusions above (`blocked`/
`theme-design`) apply to auto-pick only — if the orchestrator instead passes an explicit
{ISSUE_NUMBER}, work it regardless of its labels; a human naming a specific ticket is a
deliberate override of that default. The assignee/branch/PR check is NOT overridden by an
explicit issue number, though** — Step 8 below runs this same check again for every
invocation, auto-picked or explicit, and reports BLOCKED if it's already spoken for and you
aren't already on its own branch. The difference an explicit invocation makes is only that
there's no "next candidate" to fall back to, so a BLOCKED report there ends the whole run
(per the Final report section's existing single-invocation behavior) rather than silently
trying another ticket.

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
   current for anything you diff against it later (e.g. Step 18's `git diff --stat origin/main`
   when writing the validate artifact).
7. Fetch the issue: `gh issue view <N> -R Donelle/claude-runway --json
   number,title,body,labels,assignees,issueType,state,url`. If state is CLOSED, report
   BLOCKED — don't reopen work on a closed ticket autonomously. Read the body's
   **Location**/**Verdict**/**Suggested fix** sections as a head start, not something to
   re-derive from zero.
8. **Check whether this ticket is already being worked on — report BLOCKED immediately
   unless you're already on its own branch (issue #240; moved to run before the issue-type
   edit below per Copilot review on this PR — the gate must sit before ANY mutation, not
   just before branch/assignee changes, or an explicit invocation of an already-in-progress
   ticket would still flip its issue type before reporting BLOCKED).** This is an absolute
   check: it doesn't matter WHO the existing assignee is, including yourself from an earlier
   attempt — any pre-existing signal below means this ticket is spoken for, full stop. (For
   an auto-picked ticket this is mostly a defense-in-depth backstop — Step 0 above already
   filtered out anything with an assignee or a linked branch before ever reaching here —
   but for an explicit `{ISSUE_NUMBER}` invocation, which bypasses Step 0's filtering
   entirely, this is the ONLY point that catches it.)
   - Determine the ticket's own linked branch, if any:
     `gh issue develop <N> -R Donelle/claude-runway --list`
   - If a linked branch was found, check for a PR on it (`--state all`, since a merged/
     closed PR on a still-open issue is itself a signal worth surfacing):
     `gh pr list -R Donelle/claude-runway --head <branch> --state all --json number,state,url`
   - **The one exception:** if the linked branch found above is already the branch checked
     out in THIS worktree right now (`git branch --show-current`), you're actively
     continuing a session already on it — proceed normally to Step 9 below. In this
     subagent's fresh-worktree model (Step 1 above always resets to `origin/main` before
     anything else runs) this exception cannot actually fire in practice — nothing before
     this point ever checks out a ticket-specific branch — it's stated here only for
     consistency with `/my-gh-code-it`'s identical check, in case the isolation model ever
     changes.
   - **Otherwise**, if the issue's `assignees` (from Step 7 above) is non-empty, OR a
     linked branch was found, OR a PR was found: report
     `OUTCOME: BLOCKED (issue #<N>) — already in progress: <assignee login(s) if any>,
     <branch name if any>, <PR #<n> if any>` and stop — do not set issue type, do not set
     assignee, do not create a branch, do not touch any files.
   - If none of the above signals are present, set assignee (always the current
     authenticated user — this is what `@me` resolves to):
     `gh issue edit <N> -R Donelle/claude-runway --add-assignee @me`
9. Set issueType if null (bug label → Bug, enhancement label → Feature, neither → Task):
   `gh issue edit <N> -R Donelle/claude-runway --type <type>`
10. **Create and check out the branch BEFORE touching any files** — this order matters: in
   this same cycle worked by hand, the fix got implemented directly on `main` before the
   branch existed, and had to be corrected after the fact. Step 8 above already guarantees
   no linked branch exists yet for this ticket (otherwise you'd have reported BLOCKED there
   instead of reaching here) — issue #240 removed the old "check for an existing branch,
   reuse it" path here for exactly that reason: reuse is no longer a case this step can
   encounter. Pick a short, specific name yourself (no need to ask — `fix/<slug>` for Bug,
   `feature/<slug>` for Feature/Task, ≤6 words, matching this repo's real branch history
   style) and run:
   `gh issue develop <N> -R Donelle/claude-runway --name <name> --checkout`
   If this fails for any reason — including a stale worktree from an unrelated process
   still holding a same-named branch — report BLOCKED with the exact error rather than
   investigating or force-removing anything yourself. Step 8's guarantee means a genuine
   failure here is unexpected and worth a human's eyes, not autonomous cleanup (an earlier
   version of this step contained ~15 lines of worktree-cleanup logic for exactly this
   failure mode, back when it was a normal, expected case under the old resume-on-reattempt
   design; it isn't anymore, so that investigation no longer earns its complexity here).

11. **Mid-flight resume is intentionally not supported (issue #240).** Step 8 above already
    reports BLOCKED on any pre-existing assignee, linked branch, or PR for this ticket — the
    only tickets that reach this point have none of those, so there is nothing to resume.
    (An earlier version of this step contained ~90 lines of logic for continuing a ticket a
    prior attempt had gotten partway through — detecting an already-open PR, or commits
    already ahead of `main` with no PR yet, and picking up from wherever that left off,
    including pre-seeding the review-feedback loop's `seen_ids` for an already-open PR.
    That capability was deliberately removed: the new preflight in Step 8 now treats any
    pre-existing signal as another actor's or a stalled attempt's territory and stops there
    instead of continuing it — including the "MERGED/CLOSED PR but issue still OPEN"
    anomaly this step used to handle separately, which Step 8's `--state all` PR check now
    also catches as an in-progress signal on its own. A ticket a prior attempt stalled on
    now requires a human to clear its assignee/branch/PR before it can be auto-picked or
    explicitly retried again.)

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
    run them rather than trying to detect exactly what's missing.) After bootstrapping,
    resolve the interpreter path once — Windows venv uses `Scripts/`, Linux/macOS uses
    `bin/` (issue #230). Mypy is invoked via `$PYTHON -m mypy` rather than as a direct
    `$MYPY` binary, so only one allow rule (`.venv/Scripts/python` or `.venv/bin/python`)
    is needed instead of two:
    ```bash
    PYTHON=$(if [ -f .venv/Scripts/python ]; then echo .venv/Scripts/python; else echo .venv/bin/python; fi)
    ```
    Then run and require both clean before proceeding:
    ```
    $PYTHON -m unittest discover -s tests
    $PYTHON -m mypy libs tools hooks
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

Maintain a `seen_ids` set locally (starts empty — issue #240 removed the old resume-an-
already-open-PR path, so there is no pre-seeding case anymore; Step 8 already confirmed this
ticket had no pre-existing PR before you ever got here) across the rounds below — every
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
    _seen_tmp=$(mktemp)
    printf '%s\n' $SEEN_IDS | sort -u > "$_seen_tmp"
    for i in {1..36}; do
      NEW_COPILOT_IDS=$(gh api repos/Donelle/claude-runway/pulls/<PR>/reviews --paginate \
          --jq ".[] | select(.user.login == \"copilot-pull-request-reviewer[bot]\" and .commit_id == \"$HEAD_SHA\") | \"review:\" + (.id|tostring)" 2>/dev/null \
        | sort -u | grep -vxFf "$_seen_tmp" || true)
      if [ -n "$NEW_COPILOT_IDS" ]; then
        echo "Copilot review of $HEAD_SHA arrived: $NEW_COPILOT_IDS"
        rm -f "$_seen_tmp"
        break
      fi
      sleep 15
    done
    rm -f "$_seen_tmp"
    ```
    (`{1..36}` brace expansion, not `$(seq 1 36)` — `seq` is absent from Git Bash on
    Windows; brace expansion is a bash built-in and works cross-platform. `grep -vxFf` with
    a temp file, not `comm -23 - <(...)` — process substitution `<(...)` is unreliable in
    Git Bash on Windows; `grep -vxFf tmpfile` achieves the same set-difference without it.
    `mktemp` IS available in Git Bash. Issue #230.)
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
    PR #<PR>; can't confirm the PR is clean` and stop — a human can check whether the review
    is just slow or something's actually stuck. (A later re-run of this skill will NOT pick
    this back up automatically — issue #240 removed mid-flight resume, so Step 8 will now
    report this ticket BLOCKED as already-in-progress on any re-attempt; clearing its
    assignee/branch/PR first is required before it can be retried.) (A `gh api
    ... --paginate` call's output used to be able to come back summarized/compressed rather
    than raw JSON — this repo's own `PostToolUse` compression hook did that once for real
    during PR #187's review-feedback pass (a fabricated phrase in a summarized review body,
    documented in memory-bank), because the hook's exemption list never actually recognized
    `gh`'s own `--jq` flag or `gh api graphql`, despite an earlier version of this note
    claiming otherwise. Fixed for issue #190: the hook now matches `gh api` (REST or
    GraphQL, any flags) as exactness-critical, so this pattern reaches you unmodified.)
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
    _seen_tmp=$(mktemp)
    printf '%s\n' $SEEN_IDS | sort -u > "$_seen_tmp"
    NEW_IDS=$(printf '%s\n' "$ALL_IDS" | grep -vxFf "$_seen_tmp" || true)
    rm -f "$_seen_tmp"
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
22. **Increment `ROUND` and check the cap — before touching any code this round.**
    (An earlier version of this step also defensively re-initialized `ROUND`/
    `ROUND_HISTORY` here, in case Step 11's now-removed resume-an-open-PR shortcut jumped
    directly into this step from Step 21 without ever passing through the round loop's own
    preamble above. Issue #240 removed that shortcut, so every path reaching here has
    already gone through the preamble, and that defensive re-init no longer earns its
    keep.) `ROUND=$((ROUND + 1))`. If `ROUND` is now greater than 5, STOP: do not fix, reply to, or
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
and why. If, anywhere during this run (Step 3's implementation, or Step 4's review-feedback
loop), you hit a genuinely non-obvious root cause or correction worth remembering beyond
this one ticket, list it here as a suggested memory-bank candidate (one line each: what to
remember, why) — do NOT call `remember` yourself during this run. This skill runs with zero
approval gates and can't ask mid-run the way an interactive session would, and storing a
memory is a real side effect that always needs the user's go-ahead first (see
templates/CLAUDE.md.template's memory-bank section) — so batch candidates into this final
report instead of skipping the checkpoint entirely. Then end with EXACTLY one of these three
line shapes:
```
OUTCOME: MERGED (issue #<n>, PR #<pr>)
OUTCOME: BLOCKED (issue #<n>) — <one-line reason>
OUTCOME: FAILED (issue #<n>) — <one-line reason>
```
(A Step 0 call that found no qualifying ticket at all reports `OUTCOME: DONE` instead — no
issue number.)
````
