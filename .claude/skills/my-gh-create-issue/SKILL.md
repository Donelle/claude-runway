---
name: my-gh-create-issue
description: "Interactively create a well-structured, verified GitHub issue for this repo — labeled by type/theme/priority in the same format as the existing backlog, so my-gh-code-it/my-gh-pr keep working correctly on it."
---

# Skill: my-gh-create-issue

Interactively create a GitHub issue for claude-runway — the GitHub-issue equivalent of `/my-create-task`. The output must match the format already established across this repo's existing issues (#21–#72), because `/my-gh-code-it` and `/my-gh-pr` both read specific fields out of an issue's body (**Location**, **Suggested fix**/**Proposal**) and its labels (type, `priority-p{0-3}`, `theme-*`) — a differently-shaped issue breaks that chain, not just this skill.

**The format is not just cosmetic.** Every existing issue's specifics (line numbers, "Suggested fix") came from actually reading the code or reproducing the bug first, not from a plausible-sounding guess. Skipping the investigation step and writing a generic-sounding issue defeats the entire point of this skill — see step 3.

## Usage
```
/my-gh-create-issue type=bug theme=theme-savings-tracker priority=priority-p1 Description of the suspected issue...
/my-gh-create-issue A rough description of a bug or feature idea, no fields given
```
Any of `type=`/`theme=`/`priority=` may be omitted — missing ones get inferred from investigation (steps 4–6) and confirmed with the user, the same way `/my-create-task` asks clarifying questions rather than guessing silently.

## Steps

1. **Confirm the repo** via `git remote get-url origin` — expect `Donelle/claude-runway` on `github.com`. Stop and ask if it doesn't match.

2. **Understand the request.** Parse any `type=`/`theme=`/`priority=` fields out of the input; treat the rest as the free-text description of the bug or idea. If the description alone is too vague to investigate (no file, symptom, or area named), ask one clarifying question before proceeding — don't guess a plausible-sounding subsystem.

3. **Investigate before writing anything.** This is the step that makes the output trustworthy, and it branches by what kind of claim this is:
   - **Suspected bug** — actually find the relevant code (`Grep`/`Read`, or `qdrant-find` if available) and either reproduce the failure (a small script, a live call, or rigorous step-by-step tracing through the real mechanism — same discipline as `/my-gh-pr-feedback`'s verification step) or confirm the defect by direct code reading with exact file:line citations. If it doesn't reproduce and doesn't hold up on reading, say so and ask the user whether to still file it as an unverified/plausible risk (lower priority, said so explicitly in the body) or drop it — don't silently file an unconfirmed claim as if it were confirmed.
   - **Feature/enhancement idea** — sanity-check that its premise still holds against current code (e.g. "no caching exists" → actually grep for one first) the same way the growth-opportunity pass in this repo's history did before filing any of #48–#72.
   - Record what was actually checked (files read, commands run, repro result) — this becomes the **Location**/**Sanity-checked** line in the body, not a to-do.

4. **Determine the type label** if not given by `type=`:
   - A confirmed or plausible defect → `bug`
   - A new capability/idea → `enhancement`
   - Pure documentation fix → `documentation`
   This is the *label*, not GitHub's native Issue Type field (`Task`/`Bug`/`Feature`) — that field gets set later by `/my-gh-code-it` when work actually starts, derived from this same label, so don't set it here.

5. **Determine the theme label** if not given by `theme=`. Run `gh label list -R Donelle/claude-runway` to see the live set (don't hardcode a list here — it will drift) and match against whichever `theme-*` label fits the subsystem step 3 actually touched. As of this skill's creation the themes are:
   - `theme-install-config` — installation/configuration friction
   - `theme-ingestion` — the Qdrant codebase ingestion pipeline
   - `theme-compression-core` — local compression core & hooks
   - `theme-savings-tracker` — the opt-in savings tracker
   - `theme-session-continuity` — `my-compact`/`my-resume`
   If nothing fits well, propose a new `theme-<name>` label to the user (with a one-line description, matching the existing labels' style) rather than forcing a poor fit or silently omitting a theme.

6. **Determine the priority label** if not given by `priority=`, using the same rubric this backlog was triaged with — always confirm the assignment with the user rather than deciding alone, since it's a judgment call:
   - `priority-p0` — Critical: silently produces wrong results, or undermines the project's core value proposition
   - `priority-p1` — High: confirmed, real impact, but a narrower blast radius or specific trigger
   - `priority-p2` — Medium: real but narrower-impact, OR a plausible risk that wasn't independently confirmed in step 3
   - `priority-p3` — Low: polish, nice-to-have, or low-urgency coverage/documentation

7. **Write the issue body**, matching whichever shape fits the type:

   **Bug-shaped** (`bug`/`documentation`):
   ```
   **Priority:** {priority-pN} — {one-line severity rationale}
   **Location:** `path/to/file.py:123-145` (function/class name)

   {What's wrong, stated plainly. Cite exactly what was checked in step 3 —
   "Verified directly," "Reproduced live," or "Confirmed via code trace" —
   and quote the relevant code/output if it helps. Include the concrete
   failure scenario: what input/state triggers it, what happens instead of
   the correct behavior.}

   **Suggested fix:** {a concrete direction, not just "fix this"}
   ```

   **Enhancement-shaped** (`enhancement`):
   ```
   **Priority:** {priority-pN} — {leverage/effort rationale}
   **Sanity-checked:** {what was confirmed still true in step 3, with evidence}

   {Why this is worth doing, grounded in something actually observed —
   a real gap, a real friction point — not generic best-practice advice.}

   **Proposal:** {the concrete shape of the fix/feature}
   ```

   Only add a `Tracked in ...` footer line if this issue genuinely originates from an existing plan doc in `.plans/` — don't fabricate a reference to one that doesn't exist for this issue.

8. **Check for duplicates** before filing:
   ```bash
   gh issue list -R Donelle/claude-runway --search "{key terms from the title}" --state all
   ```
   If a close match exists, show it to the user and ask whether to link/comment on that one instead of filing a new one.

9. **Save a draft** to `.plans/{slug}.md` (kebab-case slug from the title) — same "draft before publish" step as `/my-create-task`, and gives the user something concrete to review before it becomes a public issue.

   ⛔ **STOP HERE. Show the drafted title, body, and labels to the user and wait for explicit approval before creating the issue.**

10. **Ensure every label used actually exists** (`gh label list`), creating any new theme label approved in step 5 first:
    ```bash
    gh label create "theme-<name>" -R Donelle/claude-runway --description "<one line>" --color "<hex>"
    ```

11. **Create the issue**:
    ```bash
    gh issue create -R Donelle/claude-runway \
      --title "<title>" \
      --body-file .plans/{slug}.md \
      --label "<type>,<priority>,<theme if any>"
    ```

12. **Confirm with**:
    - Issue URL and number
    - Labels applied
    - Next step: `/my-gh-code-it {N}` whenever ready to work it
