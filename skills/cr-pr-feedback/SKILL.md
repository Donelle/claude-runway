---
name: cr-pr-feedback
description: "Fetch, sort and answer the review feedback on a GitHub pull request: find the PR for the current branch, categorize every comment, verify each finding before fixing it or replying, then reply and resolve threads. GitHub only; the repository always comes from the current checkout."
---

# Skill: cr-pr-feedback

Fetch and organize the review feedback on a GitHub pull request, then work through it: sort it, check each claim, fix it or answer it, reply, resolve the thread. It works for any repository on GitHub, human or automated reviewers alike.

**The repository is never written into this skill.** Every command below takes it from the checkout you run it in: `gh pr view` and `gh pr list` without `-R`, `gh api "repos/{owner}/{repo}/..."` (the `gh` CLI fills in `{owner}` and `{repo}` from the current repository), and `gh repo view --json owner,name` where GraphQL needs the two names. Do not paste an organisation or repository name into a command, and do not parse one out of the remote URL.

## Usage

```
/cr-pr-feedback 12345
```

If no PR number is given, find the open PR for the current branch.

## Steps

1. **Confirm this is a GitHub repository.** Run `gh repo view --json owner,name`. If it fails, or `origin` is not a GitHub remote, stop and say so: this skill supports GitHub only (see [Other hosts](#other-hosts)). Do not guess a different host's commands.

2. **Find the PR number** if you were not given one:
   ```bash
   gh pr list --head "$(git branch --show-current)" --json number,url
   ```
   If that returns no PR, say so and stop.

3. **Fetch the feedback. This is one named step, the fetch: run exactly these four commands, in this order, once per round.** Run them again only to re-check after something changed (a push, a new reply), never to re-read what you already have. Substitute the PR number for `<N>`:
   ```bash
   gh pr view <N> --json number,title,body,state,mergeable,reviewDecision,reviews
   gh api "repos/{owner}/{repo}/pulls/<N>/reviews" --paginate
   gh api "repos/{owner}/{repo}/pulls/<N>/comments" --paginate
   gh api "repos/{owner}/{repo}/issues/<N>/comments" --paginate
   ```
   What each one gives you: the PR's details and its `reviews` (the review summaries, each with its `id`, `author`, `submittedAt` and the commit it reviewed); the same reviews with their numeric ids and bodies; the inline (diff) comments; and the general PR comments, where bots often leave their output. Keep the exact command shapes: they are what a later reader of the session recognises as one feedback fetch.

   **Then run one more Bash command, the marker**, as its own fifth call right after the four fetches (never appended to one of them: their shapes are what a reader recognises, so they stay exactly as written above):
   ```bash
   echo "pr-feedback-fetch: pr=<N> copilot-review=<id>"
   ```
   The line must come out of a tool call, not out of your reply: a later reader of the session reads it from that call's result and does not look at the text you write. So do not print it in your reply as well. `<id>` is the numeric `id` field of the newest entry in the output of the second command (the REST `reviews` list, a plain array, so there is no `latestReviews` to avoid here), among the entries whose `user.login` is the automated Copilot reviewer (`copilot-pull-request-reviewer[bot]`); newest means the latest `submitted_at`. Use that numeric id, not the `PRR_...` node id the first command shows for the same review: the numeric one is what the REST fetch shows, so the two name one review without any matching. If that reviewer has not reviewed this PR yet, write `copilot-review=none`. The call costs nothing and changes nothing; its only purpose is to tie this round's cost to the PR and the review it answers.

   **This skill records nothing else.** It does not call the analytics tool's `record_metric` and its output never depends on that tool: it is not required, and if it is absent nothing changes. (A future step that records a round's facts would be conditional on that tool being available and would skip silently when it is not; none exists today.)

4. **Categorize the feedback**:

   | Category | Criteria |
   |----------|----------|
   | Blocking (🚨) | Must fix before merge |
   | Suggestions (💡) | Should strongly consider |
   | Questions (❓) | Needs clarification or response |
   | Nitpicks (🔧) | Optional / style preference |
   | Praise (✅) | Positive feedback |

   The category name is the primary signal; the emoji is a scannable accent alongside it, not a standalone encoding: a screen reader or a renderer that drops emoji still gets the full meaning from the text.

5. **Separate human from automated feedback**:
   - Human reviewer comments: categorize by severity.
   - Bot comments (code-review bots, linters, CI summaries): group separately.
   - Required-status-check failures and branch-protection blocks: note as blockers.

6. **Organize by file and line number** for easy navigation in an editor.

7. **Give an actionable summary**:
   - Count of blocking vs. non-blocking items
   - Common themes
   - Any project convention the feedback refers to. Read the project's own conventions file (`CLAUDE.md`, `README.md` or `CONTRIBUTING.md`, whichever exists) if there is one, so you judge a finding against what this project actually asks for
   - A prioritized action plan

8. **Verify each finding before fixing it or replying.** Do not implement a fix just because a reviewer, bot or human, flagged something. For a finding that claims a specific failure (a bug, a bad edge case, a security issue), try to reproduce it first: a small script or a direct call that exercises the mechanism the reviewer describes, or step-by-step reasoning through the real code path. Re-reading the code and agreeing it sounds plausible is not a reproduction. Two outcomes:
   - **It reproduces**: fix it, add a regression test where that is cheap, and say in the reply how you confirmed it.
   - **It does not reproduce**: leave the code alone and reply with the specific reasoning or test evidence. Never ignore a finding silently, and never defer to a reviewer just because it is a reviewer or sounds confident. For a "this value is misclassified" or "this edge case is mishandled" claim, check whether any real caller or writer can produce that value at all (trace every one, not only the function the reviewer quoted). A value no real path produces is not a bug, and a deliberate, documented contract (for example "0 means unknown") is not a bug either.
   - **A review's headline count is not the test, its body is.** A review that headlines "0 open findings" can still list concrete "previously missed" or suppressed items in its body. Treat every concrete, checkable claim as a finding whatever the headline says.
   - Style and best-practice suggestions with no checkable failure behind them need no reproduction; use judgment.

9. **Offer interactive follow-up**:
   - "Create a todo list for these action items?"
   - "Show the diff for file X?"
   - "Draft a reply to comment Y?"
   - "Resolve thread Y?" (see below)
   - Do not post any reply or resolve any thread without the person seeing it first.

## Replying to feedback

**Do not @-mention the automated reviewer in a reply.** Not to push back, not to confirm a fix, not to ask for a re-review. A mention can start a separate coding-agent run instead of a chat reply, and that run can fail and flood the PR with error comments; automated reviews also re-run on every push without being asked, so a mention gains nothing. Every reply is plain text with no handle.

Reply templates (both untagged):

```
Checked this and it does not reproduce. <specific reasoning or test evidence>. Leaving the code as it is.
```
```
Confirmed this reproduces: <how you verified it>. Fixed in <commit>: <what changed>.
```

Reply to an inline comment:
```bash
gh api "repos/{owner}/{repo}/pulls/<N>/comments/<comment_id>/replies" --method POST -f body="..."
```
The `<N>` segment is required: the endpoint is `pulls/<N>/comments/<id>/replies`, not `pulls/comments/<id>/replies`, and leaving the PR number out returns 404.

A finding that sits only in a review's body (no standalone inline comment) gets a general PR comment instead:
```bash
gh pr comment <N> --body "..."
```

## Resolving a thread

GitHub's REST API cannot resolve a review thread. Reply first (above), then resolve in a separate GraphQL call. Get the two names with `gh repo view --json owner,name`, list the threads, and resolve each unresolved thread whose finding you addressed:

```bash
gh api graphql --paginate -f owner="<owner>" -f name="<name>" -F pr=<N> -f query='query($owner: String!, $name: String!, $pr: Int!, $endCursor: String) { repository(owner: $owner, name: $name) { pullRequest(number: $pr) { reviewThreads(first: 100, after: $endCursor) { pageInfo { hasNextPage endCursor } nodes { id isResolved comments(first: 1) { nodes { databaseId path body } } } } } } }'
gh api graphql -f threadId="<thread node id>" -f query='mutation($threadId: ID!) { resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }'
```

`--paginate` with the `$endCursor` variable and the `pageInfo` block follows the cursor through every page, so a thread beyond the first 100 is listed too; without them only the first 100 threads (resolved ones included) come back and a later thread's id is never seen. Skip threads that already show `isResolved: true`; some reviewers resolve their own once a fix lands.

## If error comments start piling up

If a burst of near-identical comments from the coding-agent or automation account appears right after your reply, each only saying it hit an unexpected error while processing a comment, and none of them carrying a finding, a test result or any other substance, treat it as a failure on the host's side: stop replying, because retrying or reposting makes it worse.

Delete only comments you have read and confirmed are that kind of invocation-error noise. Leave everything else alone: CI output, a linter's or another reviewer's diagnostics, and any comment that says what actually failed are evidence, even when several arrive together. Delete one at a time:

```bash
gh api "repos/{owner}/{repo}/issues/comments/<id>" --method DELETE
```

Deleting the noise does not close the finding it interrupted: that finding stays open until it has been fixed or answered with a reply that actually posted.

## Other hosts

Not supported yet. Azure DevOps and other hosts would need their own fetch, reply and resolve commands; until they exist this skill stops at step 1 on a non-GitHub remote instead of improvising.
