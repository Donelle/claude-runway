---
name: cr-gh-pr-feedback
description: "Handle the review feedback on this repo's GitHub pull requests: a read-only agent fetches, sorts and reproduces every finding, then this session fixes what reproduces and answers what does not, untagged."
---

# Skill: cr-gh-pr-feedback

> A shareable, repo-agnostic version of this skill ships as `/cr-pr-feedback` (`skills/cr-pr-feedback/SKILL.md`, installed by `init --install-skills`). This one stays as this repo's own.

Work the review feedback on a claude-runway pull request. This is a project-scoped extraction of `/my-pr-review-feedback`: GitHub only, no `CLAUDE.md` read (this repo has none), hardcoded `Donelle/claude-runway`. The `@copilot` tagging rule was *not* kept as-is — see [Replying to Feedback](#replying-to-feedback): never tag `@copilot` in a reply, full stop, based on confirmed repeat failures on this repo's own PRs (see the incident log below).

The work is split in two. The **read-and-verify half** (fetch, categorize, separate human from bot, and reproduce every claim in a throwaway copy) runs in the `pr-feedback` agent (`.claude/agents/pr-feedback.md`): read-only, Opus 5.5, high effort, so the reproductions do not fill this conversation. The **acting half** stays here, in the main session, where the user can be asked: fixes, commits, pushes, replies and resolving threads.

## Usage
```
/cr-gh-pr-feedback 12345
```
If no PR number is provided, find the active PR for the current branch.

## Steps

1. **Confirm the repo** via `git remote get-url origin` — expect `Donelle/claude-runway` on `github.com`. Stop and ask if it doesn't match rather than guessing.

2. **Find the PR number** if not provided:
   ```bash
   gh pr list -R Donelle/claude-runway --head $(git branch --show-current) --json number,url
   ```

3. **Fetch the review list here, not in the agent**, as one plain command (no pipe, no wrapper, no `--jq`):
   ```bash
   gh api repos/Donelle/claude-runway/pulls/{number}/reviews --paginate
   ```
   The PR-round capture reads this session's transcript, not a subagent's, and recognises a feedback round from exactly this command; a round fetched only inside the agent would not be recorded. Read it for who reviewed, on which commit and how (`copilot-pull-request-reviewer[bot]` is Copilot's automated review). Then get the PR's head branch and commit, and make sure that commit is here, as separate commands (never added to the reviews fetch above, which the capture recognises by its exact shape):
   ```bash
   gh pr view {number} -R Donelle/claude-runway --json headRefName,headRefOid
   git fetch origin pull/{number}/head
   git cat-file -e {headRefOid}^{commit}
   ```
   The head lookup does not download the commit, and the agent cannot fetch (it is read-only), so its worktree of the head fails when the commit is absent: another contributor's PR, or a push not yet fetched. Stop if the fetch or the check fails.

   Then check the checkout this session will edit: `git branch --show-current` must equal `headRefName`. Stop and ask the user if it does not, rather than fixing and committing on the wrong branch. `git rev-parse HEAD` is the head commit, or behind it: if behind, ask before pulling; if it has commits the PR lacks, or uncommitted changes, list them (`git log {headRefOid}..HEAD`, `git status --short`) and say so before going on. A finding that one of those local-only changes already fixes stays pending: no "does not reproduce" reply and no resolving until that change is pushed and `headRefOid` shows it; then it is a "Fixed in" reply.

4. **Verify with the `pr-feedback` agent.** Launch it with the `Agent` tool: `subagent_type: "pr-feedback"`, in the foreground (Step 5 depends on its result). The model and the effort come from the agent file, so do not pass them. Give it the PR number, the head commit, and which reviews or comments this conversation has already answered. It fetches the rest itself and returns a verdict for each finding.

   Agent files are loaded when a session starts, so an agent created after that is unknown to the `Agent` tool until the next session. Only for an unknown-agent-type error, launch `subagent_type: "general-purpose"`, `model: "opus"`, `effort: "high"` for this run instead, and make its prompt the full body of the agent file followed by the inputs above. Never fall back on any other error.

5. **Present the result** as an actionable summary, not the agent's raw report:
   - Count of blocking vs. non-blocking findings, and the verdicts: reproduces, does not reproduce, already fixed, judgment call.
   - Common themes, and the repo conventions the feedback refers to.
   - A prioritized action plan.
   - **A review's headline count is not the test — its body is.** The agent treats any concrete, checkable claim in a review body as a finding even when the headline says there are none; if its report lists none for a review whose body names some, ask it again rather than accept that.
   - **Reproduce every `REPRODUCES` yourself before fixing it** (its evidence holds the command), against the head commit you gave the agent: in a throwaway worktree of `{headRefOid}` (the commands are in `.claude/agents/pr-feedback.md`), removed afterwards, not in this checkout, which may be ahead of the PR and pass only because of a local fix that is not on it. A verdict that does not hold up is reported as declined, with the evidence, and is not carried forward.
   - Put every `JUDGMENT CALL` to the user, with the options the agent named. Do not choose for them.

6. **Act on the findings.**
   - **Reproduces:** fix it. Combine the fixes of one pass into one commit. Do not push until the user has said to.
   - **Does not reproduce:** do not change the code. Reply with the specific reason and the evidence. Do not silently ignore it and do not defer to a reviewer, human or bot, just because of who they are.
   - **Already fixed:** reply naming the commit.
   - Every reply follows [Replying to Feedback](#replying-to-feedback). Reply first, then resolve the thread as a separate step: GitHub's REST API cannot resolve a review thread, so use the thread's GraphQL node ID from the agent's report with the `resolveReviewThread` mutation through `gh api graphql`; there is no single call that does both.
   - **A reply that says "Fixed in <commit>", and resolving that thread, wait until the fix is on the PR.** Post them only after the push, and after `gh pr view {number} -R Donelle/claude-runway --json headRefOid` shows the fixing commit (or a later one on the same branch). Until then the PR still has the defect, and a resolved thread would say it does not. The "does not reproduce" and "already fixed" replies need no push and can go at once, provided the check ran against the PR head commit (never a checkout with local-only changes) and no local-only change touches the cited code.

7. **Offer interactive follow-up and memory:**
   - "Create todo list for these action items?" / "Show diff for file X?" / "Draft a reply to comment Y?"
   - If a finding reproduced and the fix reflects a non-obvious root cause or a correction worth remembering beyond this one PR, offer to `remember` it via the memory-bank MCP server (if configured for this repo) — "Is this something you'd like to remember for next time?" Don't store automatically.

## Replying to Feedback

**Never mention `@copilot` in any reply on this repo, for any reason.** Not for pushback, not for a fix confirmation, not even to explicitly ask for a re-review. Every reply is untagged, always.

This was a conditional rule (tag it only when asking Copilot to act) until it was tried three times, across three separate PRs, and failed **every** time the same way: mentioning `@copilot` anywhere in a PR comment thread invokes GitHub's full coding agent (a sandboxed environment trying to act on the mention), not a lightweight chat reply — and on this repo, that invocation itself errors out.
- **First PR:** four `@copilot`-tagged pushback replies ("checked, doesn't reproduce, leaving as-is" — no action requested) each independently triggered the coding agent, which failed with a generic internal error on all four, spamming ~20 auto-retried error comments onto the PR.
- **Second PR:** two `@copilot`-tagged replies explicitly requesting re-review (real fixes had landed, this was the "genuine request to act" case the old rule was written to allow) failed the exact same way — 12 error comments (`"Unfortunately I hit an unexpected error while processing your comment"`) in under a minute.
- **Third PR (2026-08-25):** happened a third time via the *generic, non-project-scoped* `/my-pr-review-feedback` skill, which still carries the old "always tag @copilot" rule — its blanket instruction was followed instead of this file's corrected one, triggering a self-perpetuating retry loop across two bursts (~24 error comments total, requiring two cleanup passes since more kept appearing minutes after the first delete). Lesson generalized beyond this one rule: a repo's own documented, empirically-confirmed exception overrides a generic skill's blanket default for that repo, even when the generic skill was invoked directly by name.

Three-for-three, including the case the conditional rule was specifically designed to permit, is enough to stop distinguishing tagged-vs-untagged entirely rather than narrow the condition further.

**This costs nothing.** Copilot's automated PR *review* (the bot that leaves inline comments — a completely different thing from the coding agent a mention invokes) already re-runs on its own after every push to the PR, with no mention required — confirmed repeatedly across this session's PRs (new review entries appeared automatically after each fix commit, unprompted). So "ask for re-review" was never actually gated on tagging in the first place; it happens regardless.

Untagged reply (works for every case — pushback, confirmation, or a fix that landed):

```
Checked this and it doesn't reproduce. <specific reasoning>. Leaving the code as-is.
```
```
Confirmed this reproduces -- <what you did to verify>. Fixed in <commit>: <what changed>.
```

Reply to an inline comment:
```bash
gh api repos/Donelle/claude-runway/pulls/{pr}/comments/{comment_id}/replies --method POST -f body="..."
```
Note the `{pr}` segment — a real mistake made in this session: the endpoint is `pulls/{pr}/comments/{id}/replies`, not `pulls/comments/{id}/replies`; omitting the PR number 404s.

**If an error-comment spam loop happens anyway** (from any source — not just a `@copilot` mention this skill might have missed catching, since a human reviewer or another automation could still trigger the same GitHub-side failure), stop immediately rather than retrying or reposting — it's a GitHub-side backend failure (see [GitHub community discussion of this failure](https://github.com/orgs/community/discussions/188531)), not something fixable from this side. Clean up the error comments:
```bash
gh api repos/Donelle/claude-runway/issues/comments/{id} --method DELETE
```
per comment, then move on — the underlying finding is already resolved either way, whether that meant a code fix or an explanatory reply.
