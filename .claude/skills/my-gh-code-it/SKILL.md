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
   - Read the full `body` — issues filed via the code-review pass already contain **Location**, **Verdict/Confirmed**, and **Suggested fix** sections; treat these as a head start on the plan in step 7, not something to re-derive from scratch.

4. **Set issue type**, only if not already set (`issueType` is `null`):
   - `bug` label present → `Task` is wrong, use `Bug`
   - `enhancement` label present → `Feature`
   - neither → `Task`
   ```bash
   gh issue edit {N} -R Donelle/claude-runway --type {Bug|Feature|Task}
   ```
   If `issueType` is already set, leave it alone and note the existing value instead.

5. **Set assignee**, only if `assignees` is empty:
   ```bash
   gh issue edit {N} -R Donelle/claude-runway --add-assignee @me
   ```
   If someone is already assigned, tell the user and ask before reassigning — don't silently take over someone else's issue.

6. **GitHub Project board — not configured yet.** This repo doesn't use a Projects v2 board as of this skill's creation, so there is no project-assignment step. If that changes later, add a step here using `gh issue edit {N} --add-project "<title>"` — note this requires `gh auth refresh -s project,read:project` once (the current token lacks both scopes), and confirm the exact project title/number with the user before hardcoding it.

7. **Create and link the branch** using GitHub's native linked-branch feature (shows up on the issue itself, not just a plain `git checkout -b`):
   - First check for an existing linked branch so re-running this skill on an in-progress issue doesn't create a duplicate:
     ```bash
     gh issue develop {N} -R Donelle/claude-runway --list
     ```
   - If one already exists, `git fetch` and check it out instead of creating a new one.
   - Otherwise, derive a branch name from the issue type and title — `fix/<short-kebab-slug>` for Bug, `feature/<short-kebab-slug>` for Feature/Task — matching this repo's actual history (`fix/chunker-defects-found-by-dogfooding`, `feature/session-continuity-skills`, etc.), not a ticket-ID-based name. Keep the slug short (aim for ≤ 6 words) and specific to the actual defect/feature, not a verbatim slugification of the full issue title.
   - **Show the proposed branch name to the user and wait for confirmation** before creating it — same "don't act until approved" discipline as the plan gate below, just lighter-weight since it's one name, not a whole plan.
   - Create, link, and check out in one step:
     ```bash
     gh issue develop {N} -R Donelle/claude-runway --name {branch-name} --checkout
     ```

8. **Load project context** if not already loaded this session — read `README.md` (this repo has no `CLAUDE.md`) plus whichever specific files the issue's **Location** field names. Don't invoke `/my-load-context` blindly; it's written to expect a `CLAUDE.md` this repo doesn't have.

9. **Create an implementation plan** in `.plans/{N}-plan.md`, seeded from the issue body:
   - Goal (one sentence, from the issue title)
   - What's already known (paste/summarize the issue's **Location** and **Verdict/Confirmed** sections — this was already verified when the issue was filed, don't re-litigate it from zero)
   - Approach (start from the issue's **Suggested fix**; note any deviation and why)
   - Files to modify/create with specific changes
   - Test strategy — this repo uses stdlib `unittest`, no network; new/updated tests belong in `tests/test_*.py` matching the module under fix

   ⛔ **STOP HERE. Present the plan to the user and wait for explicit approval before continuing.** Do not write any code until the user confirms the plan is correct.

10. **Implement the changes**, matching the style of the surrounding code (this codebase favors explicit comments explaining *why*, not just *what* — see any existing file for the density expected).

11. **Write or update unit tests** in the relevant `tests/test_*.py` file, following existing patterns in that file (e.g. `test_local_compress_lib.py`'s stub-injection technique for anything touching LM Studio, so tests don't require a live model).

12. **Run the test suite, then type-check**:
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

13. **Fix any failures** before proceeding.

14. **Commit and push**:
    ```bash
    git add <files>
    git commit -m "<description of the fix/feature>"
    git push -u origin {branch-name}
    ```
    Don't put "Fixes #{N}" in the commit message itself — this repo's history shows the issue/PR number gets attached automatically by GitHub's squash-merge, not authored by hand per commit. Save the explicit "Fixes #{N}" for the PR body instead, so merging actually auto-closes the issue.

15. **Provide a summary**:
    - Files created/modified
    - Tests added/updated
    - Confirmation that the branch is linked to issue #{N} (visible on the issue's GitHub page)
    - Any deferred items or known limitations
    - Next step: open a PR with `Fixes #{N}` in the body (e.g. `gh pr create --fill` and then edit the body to add that line, or use whichever PR-creation skill/workflow you'd normally reach for) — this skill doesn't create the PR itself.
