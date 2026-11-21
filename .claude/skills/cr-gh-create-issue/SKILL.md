---
name: cr-gh-create-issue
description: "Interactively create a well-structured, verified GitHub issue for this repo — labeled by type/theme/priority in the same format as the existing backlog, so cr-gh-code-it/cr-gh-pr keep working correctly on it."
---

# Skill: cr-gh-create-issue

Interactively create a GitHub issue for claude-runway — the GitHub-issue equivalent of `/my-create-task`. The output must match the format already established across this repo's existing issues, because `/cr-gh-code-it` and `/cr-gh-pr` both read specific fields out of an issue's body (**Location**, **Suggested fix**/**Proposal**, now inside its `## AI Instructions` section; the body opens with a short `## Summary` for people) and its labels (type, `priority-p{0-3}`, `theme-*`) — a differently-shaped issue breaks that chain, not just this skill.

**The format is not just cosmetic.** Every existing issue's specifics (line numbers, "Suggested fix") came from actually reading the code or reproducing the bug first, not from a plausible-sounding guess. Skipping the investigation step and writing a generic-sounding issue defeats the entire point of this skill — see step 3.

## Usage
```
/cr-gh-create-issue type=bug theme=theme-savings-tracker priority=priority-p1 Description of the suspected issue...
/cr-gh-create-issue A rough description of a bug or feature idea, no fields given
/cr-gh-create-issue type=spike theme=theme-metrics priority=priority-p3 The question to be answered...
```
`type=spike` files a spike ticket, a research question whose outcome is findings and a go/no-go rather than code; `/cr-gh-spike` works it. Any of `type=`/`theme=`/`priority=` may be omitted — missing ones get inferred from investigation (steps 4–6) and confirmed with the user, the same way `/my-create-task` asks clarifying questions rather than guessing silently.

## Steps

1. **Confirm the repo** via `git remote get-url origin` — expect `Donelle/claude-runway` on `github.com`. Stop and ask if it doesn't match.

2. **Understand the request.** Parse any `type=`/`theme=`/`priority=` fields out of the input; treat the rest as the free-text description of the bug or idea. If the description alone is too vague to investigate (no file, symptom, or area named), ask one clarifying question before proceeding — don't guess a plausible-sounding subsystem.

3. **Investigate before writing anything.** This is the step that makes the output trustworthy, and it branches by what kind of claim this is:
   - **Suspected bug** — actually find the relevant code (`Grep`/`Read`, or `cb_indexer_find_repo` if available) and either reproduce the failure (a small script, a live call, or rigorous step-by-step tracing through the real mechanism — same discipline as `/cr-gh-pr-feedback`'s verification step) or confirm the defect by direct code reading with exact file:line citations. If it doesn't reproduce and doesn't hold up on reading, say so and ask the user whether to still file it as an unverified/plausible risk (lower priority, said so explicitly in the body) or drop it — don't silently file an unconfirmed claim as if it were confirmed.
   - **Feature/enhancement idea** — sanity-check that its premise still holds against current code (e.g. "no caching exists" → actually grep for one first) the same way the growth-opportunity pass in this repo's history did before filing that backlog batch.
   - **Spike (a question to answer, not a change to make)** — do not investigate the answer: that is the spike's own job, and a half-answered spike ticket hides what is still open. Check only that the premise holds (the code, tool or limit the question is about exists as described) and that the question is not already answered by the code, `docs/` or a stored memory (`recall`). If it already is, say so and ask whether to still file it.
   - Record what was actually checked (files read, commands run, repro result) — this becomes the **Location**/**Sanity-checked** line in the body, not a to-do.

4. **Determine the type** if not given by `type=`:
   - A confirmed or plausible defect → `bug`
   - A new capability/idea → `enhancement`
   - Pure documentation fix → `documentation`
   - A question to be answered before deciding on a change ("can we", "is it feasible", "which of these") → `spike`
   For `bug`, `enhancement` and `documentation` this is the *label*, not GitHub's native Issue Type field (`Task`/`Bug`/`Feature`) — that field gets set later by `/cr-gh-code-it` when work actually starts, derived from this same label, so don't set it here.

   `spike` is the exception, and it is not a label: a spike ticket gets no `bug`/`enhancement`/`documentation` label and no `spike` label, and its native Issue Type is `Spike`. `/cr-gh-code-it` never works a spike, so nothing later would set that type: step 12 sets it right after the issue is created. A spike ticket's title starts with `Spike:`.

5. **Determine the theme label** if not given by `theme=`. Run `gh label list -R Donelle/claude-runway` to see the live set (don't hardcode a list here — it will drift) and match against whichever `theme-*` label fits the subsystem step 3 actually touched. As of this skill's creation the themes are:
   - `theme-install-config` — installation/configuration friction
   - `theme-ingestion` — the Qdrant codebase ingestion pipeline
   - `theme-compression-core` — local compression core & hooks
   - `theme-savings-tracker` — the opt-in savings tracker
   - `theme-session-continuity` — `cr-compact`/`cr-resume`
   If nothing fits well, propose a new `theme-<name>` label to the user (with a one-line description, matching the existing labels' style) rather than forcing a poor fit or silently omitting a theme.

6. **Determine the priority label** if not given by `priority=`, using the same rubric this backlog was triaged with — always confirm the assignment with the user rather than deciding alone, since it's a judgment call:
   - `priority-p0` — Critical: silently produces wrong results, or undermines the project's core value proposition
   - `priority-p1` — High: confirmed, real impact, but a narrower blast radius or specific trigger
   - `priority-p2` — Medium: real but narrower-impact, OR a plausible risk that wasn't independently confirmed in step 3
   - `priority-p3` — Low: polish, nice-to-have, or low-urgency coverage/documentation

7. **Write the issue body in two sections: `## Summary` for the person, then `## AI Instructions` for the implementing agent.** The person skims tickets and stops reading long ones, so everything they need to decide or react goes in the Summary, and everything an implementer needs goes below it, where length is fine.

   **Summary** (top of the body, for a person; this is what they see first):
   - At most about six lines, in plain words. If it does not fit, the ticket is too big or the thinking is not done.
   - Always: **Problem** (what is wrong or missing, one or two sentences), **Ask** (what changes, one sentence), **Done when** (one line a person can check), and the **Priority** with its reason in a few words.
   - Add **Blocked by** only when something real blocks it, and **Decision needed** only when the person has a choice to make.
   - No file:line citations, no code, no evidence dump, no history. Those go in the AI Instructions.
   - Never a bare issue or PR number: put its title next to it.

   **AI Instructions** (everything below the Summary; as verbose as it helps): the sections for the type, below. It keeps the exact field names `/cr-gh-code-it`, `/cr-gh-pr` and `/cr-gh-code-review` look for (**Location**, **Suggested fix** or **Proposal**, **Acceptance**, **Depends on**). Write it for an agent starting cold: what was checked and how, exact locations, the failure scenario or evidence, the reasoning, edge cases, decisions already settled, and open questions.

   **Line breaks:** GitHub Markdown joins consecutive lines into one paragraph, so every labelled line (the Summary fields, and the **Location**/**Depends on**-style lines in the AI Instructions) is its own bullet (`- **Field:** ...`) or is separated by a blank line. Never stack bare `**Field:**` lines.

   Body skeleton:
   ```
   ## Summary
   - **Problem:** ...
   - **Ask:** ...
   - **Done when:** ...
   - **Priority:** {priority-pN} — {a few words}

   ## AI Instructions
   {the type-specific shape below}
   ```

   **Bug-shaped** (`bug`/`documentation`), under `## AI Instructions`:
   ```
   - **Severity rationale:** {why this priority: the full reasoning behind the Summary's few words}
   - **Location:** `path/to/file.py:123-145` (function/class name)

   {What's wrong, stated plainly. Cite exactly what was checked in step 3 —
   "Verified directly," "Reproduced live," or "Confirmed via code trace" —
   and quote the relevant code/output if it helps. Include the concrete
   failure scenario: what input/state triggers it, what happens instead of
   the correct behavior.}

   **Suggested fix:** {a concrete direction, not just "fix this"}
   ```

   **Enhancement-shaped** (`enhancement`), under `## AI Instructions`:
   ```
   - **Leverage/effort rationale:** {why this priority: the full reasoning behind the Summary's few words}
   - **Sanity-checked:** {what was confirmed still true in step 3, with evidence}

   {Why this is worth doing, grounded in something actually observed —
   a real gap, a real friction point — not generic best-practice advice.}

   **Proposal:** {the concrete shape of the fix/feature}
   ```

   **Spike-shaped** (`spike`), under `## AI Instructions`. Its Summary uses the same four fields, with **Ask** phrased as the question and **Done when** as "each question below is answered with evidence (or marked not answerable), a verdict and recommendation are posted on this ticket, and follow-up tickets are filed if worth it":
   ```
   - **Part of:** {the parent ticket's title, if any}
   - **Depends on:** {what must happen first, or "nothing"}
   - **Leverage/effort rationale:** {why this priority: the full reasoning behind the Summary's few words}
   - **Sanity-checked:** {what was confirmed about the premise in step 3, with evidence}

   **Questions.**
   1. {a question with an answer}
   2. ...

   **Done looks like:** {the evidence that settles each question: a measurement, a working call, a file:line fact, a documented limit}

   **Out of scope:** {what the spike will not look into}
   ```
   Do not put a verdict or a suggested fix in it: the findings are posted later as a comment by `/cr-gh-spike`.

   Only add a `Tracked in ...` footer line if this issue genuinely originates from an existing note in `.ai-plans/` — don't fabricate a reference to one that doesn't exist for this issue.

8. **Check for duplicates** before filing:
   ```bash
   gh issue list -R Donelle/claude-runway --search "{key terms from the title}" --state all --limit 300
   ```
   Always pass `--limit`: `gh issue list` silently truncates to the 30 newest results without it, so an older matching issue can be dropped and a duplicate filed (this has happened before).
   If a close match exists, show it to the user and ask whether to link/comment on that one instead of filing a new one.

9. **Save a draft** to `.ai-plans/{slug}.md` (kebab-case slug from the title) — same "draft before publish" step as `/my-create-task`, and gives the user something concrete to review before it becomes a public issue. Not `.plans/`: that folder holds only the `{N}-plan.md`/`{N}-research.md`/`{N}-validate.md` artifacts of tickets being worked, while drafts, the backlog work order and standalone notes live in `.ai-plans/`. Both are gitignored.

   ⛔ **STOP HERE. Show the drafted title, the Summary, and the labels to the user and wait for explicit approval before creating the issue.** Show the Summary in the message itself and name the draft file for the AI Instructions; do not paste the whole AI Instructions into the chat, since that is the wall of text the Summary exists to avoid. Show the AI Instructions in full only if the user asks.

10. **Ensure every label used actually exists** (`gh label list`), creating any new theme label approved in step 5 first:
    ```bash
    gh label create "theme-<name>" -R Donelle/claude-runway --description "<one line>" --color "<hex>"
    ```

11. **Create the issue**:
    ```bash
    gh issue create -R Donelle/claude-runway \
      --title "<title>" \
      --body-file .ai-plans/{slug}.md \
      --label "<type>,<priority>,<theme if any>"
    ```
    For a spike, leave the type out of `--label` (the labels are just `<priority>,<theme if any>`).

12. **Set the Issue Type of a spike**, right after creating it; skip this step for every other type. `gh issue edit` has no `--type` flag, so use the REST API, with the number `gh issue create` printed at the end of its URL:
    ```bash
    gh api repos/Donelle/claude-runway/issues/{N} --method PATCH -f type=Spike --jq '.type.name'
    ```
    It must print `Spike`. If it fails, say so in the confirmation rather than leaving the ticket untyped silently.

13. **Delete the draft** (`.ai-plans/{slug}.md`) once the issue exists. The issue body now holds its full text, and a leftover copy only drifts from it.

14. **Confirm with**:
    - Issue URL and number
    - Labels applied (and, for a spike, the Issue Type)
    - Next step: `/cr-gh-code-it {N}` whenever ready to work it, or for a spike `/cr-gh-spike {N}`
