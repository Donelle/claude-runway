---
name: my-gh-code-it
description: "Implement code changes for a GitHub issue in this repository — creates a linked branch, sets assignee/issue type, and works the fix like my-code-it does for JIRA tickets."
---

# Skill: my-gh-code-it

Implement code changes for a GitHub issue in the current repository (claude-runway). This is the GitHub-issue equivalent of `/my-code-it` — same plan-then-implement discipline, but driven by a GitHub issue number instead of a JIRA ticket key, and using GitHub's native linked-branch mechanism instead of a plain feature branch.

Project-scoped to this repo on purpose: it hardcodes claude-runway's actual conventions (README.md as the doc entry point — this repo has no CLAUDE.md, its `.venv`-based test command, its `fix/`/`feature/` branch prefixes, and its `.plans/` directory) rather than trying to be generic across repos.

## Usage
```
/my-gh-code-it 21
/my-gh-code-it #21
/my-gh-code-it https://github.com/Donelle/claude-runway/issues/21
```

## Steps

1. **Normalize the input** to a bare issue number (strip a leading `#`, or extract the trailing number from a full URL).

2. **Confirm the repo** via `git remote get-url origin` — expect `github.com` and owner/repo `Donelle/claude-runway`. If the remote doesn't match, stop and ask the user rather than guessing which repo the issue number refers to.

3. **Fetch the issue**:
   ```bash
   gh issue view {N} -R Donelle/claude-runway --json number,title,body,labels,assignees,issueType,state,url
   ```
   - If `state` is `CLOSED`, tell the user and ask whether to proceed anyway (e.g. reopening work on a regression) rather than silently continuing.
   - Read the full `body` — issues filed via the code-review pass already contain **Location**, **Verdict/Confirmed**, and **Suggested fix** sections; treat these as a head start on the plan below, not something to re-derive from scratch.

4. **Check whether this ticket is already being worked on — stop immediately unless you're already on its own branch (issue #240).** This is an absolute check: it doesn't matter WHO the existing assignee is, including yourself from an earlier run — any pre-existing signal below means don't proceed.
   - Determine the ticket's own linked branch, if any:
     ```bash
     gh issue develop {N} -R Donelle/claude-runway --list
     ```
   - If a linked branch was found, check for a PR on it (`--state all`, since a merged/closed PR on a still-open issue is itself a signal worth surfacing, not just an open one):
     ```bash
     gh pr list -R Donelle/claude-runway --head {branch} --state all --json number,state,url
     ```
   - **The one exception:** if the linked branch found above is already the branch you currently have checked out (`git branch --show-current`) — i.e. you're actively continuing a session you already started on it — proceed normally to Step 5 below; its own "only if not already set"/"only if empty" guards will no-op whatever's already done.
   - **Otherwise**, if `assignees` (from Step 3) is non-empty, OR a linked branch was found, OR a PR was found: **STOP HERE.** Tell the user this issue is already in progress — report the assignee login(s) if any, the branch name if any, and the PR number + URL if any — and do not proceed to any step below (no issue-type change, no assignee change, no branch creation).
   - If none of the above signals are present, proceed normally.

5. **Set issue type**, only if not already set (`issueType` is `null`):
   - `bug` label present → `Task` is wrong, use `Bug`
   - `enhancement` label present → `Feature`
   - neither → `Task`
   ```bash
   gh issue edit {N} -R Donelle/claude-runway --type {Bug|Feature|Task}
   ```
   If `issueType` is already set, leave it alone and note the existing value instead.

6. **Set assignee**, only if `assignees` is empty:
   ```bash
   gh issue edit {N} -R Donelle/claude-runway --add-assignee @me
   ```
   (Step 4 above already stopped the workflow if anyone was assigned, so this only ever fires on a genuinely fresh ticket — the residual "someone else" case here is only reachable via Step 4's own branch-match exception, e.g. you're on the ticket's branch yourself but someone else happens to be the assignee; if so, tell the user and ask before reassigning rather than silently taking over.)

7. **GitHub Project board — not configured yet.** This repo doesn't use a Projects v2 board as of this skill's creation, so there is no project-assignment step. If that changes later, add a step here using `gh issue edit {N} --add-project "<title>"` — note this requires `gh auth refresh -s project,read:project` once (the current token lacks both scopes), and confirm the exact project title/number with the user before hardcoding it.

8. **Create and link the branch** using GitHub's native linked-branch feature (shows up on the issue itself, not just a plain `git checkout -b`). Step 4 above already confirmed no linked branch exists yet (otherwise you'd have stopped there, or already be on it) — issue #240 removed the old "check for an existing branch, reuse it" bullet here for exactly that reason: reuse is no longer a case this step can encounter.
   - Derive a branch name from the issue type and title — `fix/<short-kebab-slug>` for Bug, `feature/<short-kebab-slug>` for Feature/Task — matching this repo's actual history (`fix/chunker-defects-found-by-dogfooding`, `feature/session-continuity-skills`, etc.), not a ticket-ID-based name. Keep the slug short (aim for ≤ 6 words) and specific to the actual defect/feature, not a verbatim slugification of the full issue title.
   - **Show the proposed branch name to the user and wait for confirmation** before creating it — same "don't act until approved" discipline as the plan gate below, just lighter-weight since it's one name, not a whole plan.
   - Create, link, and check out in one step:
     ```bash
     gh issue develop {N} -R Donelle/claude-runway --name {branch-name} --checkout
     ```

9. **Load project context** if not already loaded this session — read `README.md` (this repo has no `CLAUDE.md`) plus whichever specific files the issue's **Location** field names. Don't invoke `/my-load-context` blindly; it's written to expect a `CLAUDE.md` this repo doesn't have.

10. **Create an implementation plan** in `.plans/{N}-plan.md`, seeded from the issue body:
    - Goal (one sentence, from the issue title)
    - What's already known (paste/summarize the issue's **Location** and **Verdict/Confirmed** sections — this was already verified when the issue was filed, don't re-litigate it from zero)
    - Approach (start from the issue's **Suggested fix**; note any deviation and why)
    - Files to modify/create with specific changes
    - Test strategy — this repo uses stdlib `unittest`, no network; new/updated tests belong in `tests/test_*.py` matching the module under fix

    ⛔ **STOP HERE. Present the plan to the user and wait for explicit approval before continuing.** Do not write any code until the user confirms the plan is correct.

11. **Implement the changes**, matching the style of the surrounding code (this codebase favors explicit comments explaining *why*, not just *what* — see any existing file for the density expected).

12. **Write or update unit tests** in the relevant `tests/test_*.py` file, following existing patterns in that file (e.g. `test_local_compress_lib.py`'s stub-injection technique for anything touching LM Studio, so tests don't require a live model).

13. **Run the test suite, then type-check**:
    ```bash
    .venv/bin/python -m unittest discover -s tests
    .venv/bin/mypy libs tools hooks
    ```
    All tests must pass, including any new ones, and mypy must report no
    issues. mypy is installed via `requirements-dev.txt` (a separate,
    dev-only file from `requirements.txt` — see that file's own comment); if
    it's missing from this repo's venv, install it with
    `uv pip install -r requirements-dev.txt --index-url https://pypi.org/simple`
    (not `.venv/bin/pip install` — `uv venv` doesn't seed a `pip` executable
    by default, so that would fail in exactly the recovery case this is
    for) rather than skipping the check.

14. **Fix any failures** before proceeding.

15. **Commit and push**:
    ```bash
    git add <files>
    git commit -m "<description of the fix/feature>"
    git push -u origin {branch-name}
    ```
    Don't put "Fixes #{N}" in the commit message itself — this repo's history shows the issue/PR number gets attached automatically by GitHub's squash-merge, not authored by hand per commit. Save the explicit "Fixes #{N}" for the PR body instead, so merging actually auto-closes the issue.

16. **Provide a summary**:
    - Files created/modified
    - Tests added/updated
    - Confirmation that the branch is linked to issue #{N} (visible on the issue's GitHub page)
    - Any deferred items or known limitations
    - Next step: open a PR with `Fixes #{N}` in the body (e.g. `gh pr create --fill` and then edit the body to add that line, or use whichever PR-creation skill/workflow you'd normally reach for) — this skill doesn't create the PR itself.
