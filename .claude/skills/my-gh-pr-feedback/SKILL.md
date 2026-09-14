---
name: my-gh-pr-feedback
description: "Fetch and organize pull request feedback for this repo's GitHub PRs — a GitHub-only, project-scoped extraction of my-pr-review-feedback."
---

# Skill: my-gh-pr-feedback

Fetch and organize pull request feedback for claude-runway. This is a project-scoped extraction of `/my-pr-review-feedback` — that skill already fully supports GitHub, so this mirror mainly drops the Azure DevOps branching and the `CLAUDE.md` read (this repo has none), hardcodes `YOUR_GITHUB_USERNAME/claude-runway`, and keeps the categorization scheme and verify-before-fixing discipline as-is. The `@copilot` tagging rule was *not* kept as-is — see [Replying to Feedback](#replying-to-feedback): never tag `@copilot` in a reply, full stop, based on confirmed repeat failures on this repo's PR #73 and PR #84.

## Usage
```
/my-gh-pr-feedback 12345
```
If no PR number is provided, find the active PR for the current branch.

## Steps

1. **Confirm the repo** via `git remote get-url origin` — expect `YOUR_GITHUB_USERNAME/claude-runway` on `github.com`. Stop and ask if it doesn't match rather than guessing.

2. **Find the PR number** if not provided:
   ```bash
   gh pr list -R YOUR_GITHUB_USERNAME/claude-runway --head $(git branch --show-current) --json number,url
   ```

3. **Fetch PR details and feedback**:
   ```bash
   gh pr view {number} -R YOUR_GITHUB_USERNAME/claude-runway --json title,body,state,mergeable,reviewDecision
   gh api repos/YOUR_GITHUB_USERNAME/claude-runway/pulls/{number}/comments --paginate     # inline (diff) review comments
   gh api repos/YOUR_GITHUB_USERNAME/claude-runway/pulls/{number}/reviews --paginate      # review summaries
   gh api repos/YOUR_GITHUB_USERNAME/claude-runway/issues/{number}/comments --paginate    # general/bot issue comments
   ```

4. **Categorize feedback**:

   | Category | Criteria |
   |----------|----------|
   | Blocking (🚨) | Must fix before merge |
   | Suggestions (💡) | Should strongly consider |
   | Questions (❓) | Needs clarification or response |
   | Nitpicks (🔧) | Optional / style preference |
   | Praise (✅) | Positive feedback |

   The category name is the primary signal; the emoji is a scannable accent alongside it, not a standalone encoding — a screen reader or a renderer that drops emoji still gets the full meaning from the text.

5. **Separate human vs. automated feedback**:
   - Human reviewer comments → categorize by severity
   - Bot / Copilot comments → group separately
   - Required-status-check failures / branch-protection blocks → note as blockers

6. **Organize by file and line number** for easy IDE navigation.

7. **Provide an actionable summary**:
   - Count of blocking vs. non-blocking issues
   - Common themes
   - Patterns from this repo's actual conventions (README.md's documented design philosophy — fail-open hooks, the exactness-critical exemption discipline, etc.) that feedback references
   - Prioritized action plan

8. **Verify each finding before fixing or replying** — don't implement a fix just because a reviewer (bot or human) flagged something. For any finding that claims a specific failure mode (a bug, a bad edge case, a security issue), actually try to reproduce it first: a quick test script, or rigorous step-by-step reasoning through the real mechanism — not just re-reading the code and nodding along. Two outcomes:
   - **Reproduces** → fix it. The fix is stronger for having a confirmed repro, worth mentioning in the reply.
   - **Doesn't reproduce** → don't change the code. Reply explaining specifically why, with the reasoning or test evidence — don't silently ignore it, and don't defer to the reviewer by default just because they're a reviewer (or a bot with a confident-sounding tone).
   - This applies to any finding with a concrete, checkable claim — not just correctness/security. Style or "best practice" suggestions without a checkable failure mode don't need a repro; use judgment on those directly.
   - Never tag `@copilot` in the reply — see [Replying to Feedback](#replying-to-feedback) for why: it reliably triggers a broken coding-agent invocation on this repo, confirmed three times (PR #73, PR #84, PR #107), and costs nothing to skip since Copilot's automated review already re-runs on every push with no mention needed.

9. **Offer interactive follow-up**:
   - "Create todo list for these action items?"
   - "Show diff for file X?"
   - "Draft a reply to comment Y?"
   - "Resolve thread Y?" — GitHub's REST API cannot resolve a review thread; fetch the thread's GraphQL node ID via a `reviewThreads` query, then call the `resolveReviewThread` mutation through `gh api graphql`. Reply first (via the REST endpoint below), then resolve as a separate step — there's no single call that does both.
   - Any reply drafted or posted here follows [Replying to Feedback](#replying-to-feedback).

## Replying to Feedback

**Never mention `@copilot` in any reply on this repo, for any reason.** Not for pushback, not for a fix confirmation, not even to explicitly ask for a re-review. Every reply is untagged, always.

This was a conditional rule (tag it only when asking Copilot to act) until it was tried three times, across three separate PRs, and failed **every** time the same way: mentioning `@copilot` anywhere in a PR comment thread invokes GitHub's full coding agent (a sandboxed environment trying to act on the mention), not a lightweight chat reply — and on this repo, that invocation itself errors out.
- **PR #73:** four `@copilot`-tagged pushback replies ("checked, doesn't reproduce, leaving as-is" — no action requested) each independently triggered the coding agent, which failed with a generic internal error on all four, spamming ~20 auto-retried error comments onto the PR.
- **PR #84:** two `@copilot`-tagged replies explicitly requesting re-review (real fixes had landed, this was the "genuine request to act" case the old rule was written to allow) failed the exact same way — 12 error comments (`"Unfortunately I hit an unexpected error while processing your comment"`) in under a minute.
- **PR #107 (2026-08-25):** happened a third time via the *generic, non-project-scoped* `/my-pr-review-feedback` skill, which still carries the old "always tag @copilot" rule — its blanket instruction was followed instead of this file's corrected one, triggering a self-perpetuating retry loop across two bursts (~24 error comments total, requiring two cleanup passes since more kept appearing minutes after the first delete). Lesson generalized beyond this one rule: a repo's own documented, empirically-confirmed exception overrides a generic skill's blanket default for that repo, even when the generic skill was invoked directly by name.

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
gh api repos/YOUR_GITHUB_USERNAME/claude-runway/pulls/{pr}/comments/{comment_id}/replies --method POST -f body="..."
```
Note the `{pr}` segment — a real mistake made in this session: the endpoint is `pulls/{pr}/comments/{id}/replies`, not `pulls/comments/{id}/replies`; omitting the PR number 404s.

**If an error-comment spam loop happens anyway** (from any source — not just a `@copilot` mention this skill might have missed catching, since a human reviewer or another automation could still trigger the same GitHub-side failure), stop immediately rather than retrying or reposting — it's a GitHub-side backend failure (see [community discussion #188531](https://github.com/orgs/community/discussions/188531)), not something fixable from this side. Clean up the error comments:
```bash
gh api repos/YOUR_GITHUB_USERNAME/claude-runway/issues/comments/{id} --method DELETE
```
per comment, then move on — the underlying finding is already resolved either way, whether that meant a code fix or an explanatory reply.
