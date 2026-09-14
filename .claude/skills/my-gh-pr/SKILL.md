---
name: my-gh-pr
description: "Create a pull request for this repo and post research/plan/validate artifacts as comments on the linked GitHub issue — the GitHub-issue equivalent of my-post-pr's JIRA-attachment workflow."
---

# Skill: my-gh-pr

Create a pull request for claude-runway and document the work against its GitHub issue — the GitHub-issue equivalent of `/my-post-pr`. GitHub has no direct analog to "attach a file to a ticket," so the RPIV-style artifacts (research/plan/validate) get posted as comments on the issue instead of attached as files, which is the closest native equivalent and keeps the trail visible on the issue itself.

Project-scoped to this repo: hardcodes `YOUR_GITHUB_USERNAME/claude-runway`, README.md as the doc entry point (no `CLAUDE.md` here), and the `.venv`-based test command — no Azure DevOps branching, since this repo is GitHub-only.

## Usage
```
/my-gh-pr 21
```
The issue number is required — unlike `/my-pr-review-feedback`, there's no reliable way to reverse-lookup "which issue is this branch for" without asking, so don't guess it from the branch name.

## Steps

1. **Confirm the repo** via `git remote get-url origin` — expect `YOUR_GITHUB_USERNAME/claude-runway` on `github.com`. Stop and ask if it doesn't match rather than guessing.

2. **Ensure all changes are committed and pushed**:
   ```bash
   git status
   git push -u origin HEAD
   ```

3. **Fetch the issue** for context (title, labels, original **Location**/**Suggested fix** if this repo's code-review-filed issues):
   ```bash
   gh issue view {N} -R YOUR_GITHUB_USERNAME/claude-runway --json number,title,body,url,labels
   ```

4. **Gather PR details**:
   - Title: a short, descriptive summary of the change itself (not the raw issue title verbatim — tighten it the way a human-written PR title would read).
   - Body: a bulleted summary of the actual changes, plus `Fixes #{N}` on its own line so merging auto-closes the issue. Link back to the issue explicitly too, in case the auto-close phrase is ever edited out during review.

5. **Create the PR**:
   ```bash
   gh pr create --base main --head {branch} --title "..." --body "..."
   ```
   Check `gh auth status` first; if `gh` isn't authenticated or the repo isn't visible to it (this has happened before in this repo's session history — the `mcp__github` MCP server's token couldn't see this org even though `gh` could), fall back to `mcp__github__create_pull_request` only if `gh` itself is unusable, not just because the MCP tool is available.

6. **Generate three artifacts in `.plans/`** (reminder: `.plans/` is globally gitignored on this machine, so these are a working/documentation trail, not something that ends up in the diff):
   - `{N}-research.md` — what was actually investigated to understand the bug/feature: specific files read, the mechanism confirmed (or ruled out), any repro performed. If `/my-gh-code-it` already ran, most of this is the issue body's own **Location**/**Verdict** sections plus whatever else came up during implementation — summarize and extend, don't restate verbatim.
   - `{N}-plan.md` — should already exist from `/my-gh-code-it`; update it if the implementation diverged from the original plan, rather than leaving it stale.
   - `{N}-validate.md` — actual verification evidence: the real `.venv/bin/python -m unittest discover -s tests` output with pass/fail counts (not "tests pass"), `git diff --stat` against `main` for the files-changed list, and an explicit plan-vs-implementation comparison (what matched, what changed and why).

7. **Quality bar for artifacts** (same standard as `/my-post-pr`):
   - Research names specific files and line numbers, not just modules
   - Plan lists specific files with the type of change (add/modify/remove) and maps back to the issue's acceptance criteria
   - Validate includes actual command output with counts — a claim of "all tests pass" with no pasted output doesn't meet the bar

8. **Post the artifacts as comments on the issue**, one per artifact so each stays readable rather than one giant comment:
   ```bash
   gh issue comment {N} -R YOUR_GITHUB_USERNAME/claude-runway --body-file .plans/{N}-research.md
   gh issue comment {N} -R YOUR_GITHUB_USERNAME/claude-runway --body-file .plans/{N}-plan.md
   gh issue comment {N} -R YOUR_GITHUB_USERNAME/claude-runway --body-file .plans/{N}-validate.md
   ```

9. **Confirm with**:
   - PR URL
   - Issue URL
   - Confirmation that `Fixes #{N}` is in the PR body (so merge will auto-close)
   - Summary of which artifacts were posted as comments
