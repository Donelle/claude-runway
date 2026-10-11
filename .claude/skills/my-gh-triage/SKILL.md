---
name: "my-gh-triage"
description: "Triage a reported problem or question in claude-runway: investigate first, state a conclusion, file a ticket only if one is needed, then decide whether working it now is right."
---

# Skill: my-gh-triage

Use this when the user reports something odd (an error message, surprising behavior, "I see this a lot", "why does X happen") rather than handing over a ready ticket. It runs four gates in order. Each gate ends with a written outcome, and you never skip ahead: no ticket before a conclusion, no code before a ticket.

This skill orchestrates the repo's existing skills; it does not replace them:
- `/my-gh-create-issue` files the ticket (it keeps its own approval stop).
- `/my-gh-code-it` works the ticket (it keeps its own already-being-worked check).

## Usage
```
/my-gh-triage <paste the error, symptom, or question>
```

## Gate 1: Investigate (read-only)

1. Confirm the repo with `git remote get-url origin` (expect `Donelle/claude-runway`). Stop and ask if it doesn't match.
2. Find where the behavior comes from. Grep for distinctive text from the report (an error string, a tool name) to locate the source file, then read the whole code path, not just the matching line.
3. Establish the facts, each with a `file:line` citation:
   - What rule or condition triggers the behavior.
   - Whether it is intended. Check the file's docstring, `docs/`, `templates/`, and `git log` on the file for the design reasoning.
   - Whether it actually works as intended. Run the relevant tests, or reproduce with a small script or by tracing the real mechanism.
4. Check parity: if the user keeps more than one copy of the repo (for example an open-source and a work checkout), diff the relevant file across them and note any drift.
5. Make no edits, branches, or issues during this gate.

## Gate 2: Conclude

Write a short conclusion for the user before doing anything else. Lead with the answer:

- **Verdict**, exactly one of:
  - `working as intended`: the behavior is by design and works.
  - `intended but confusing`: by design, but the wording, docs, or UX mislead.
  - `bug`: confirmed defect.
  - `plausible risk`: could not confirm; say what was tried.
  - `config / environment`: caused by the user's setup, not the code.
- **Why**: the rule in plain words, with `file:line` evidence.
- **What's wrong** (if anything): numbered, one line each.
- **Recommended fix**: one recommendation, not a menu.

Say which claims were verified (tests run, repro output) and which were inferred from reading.

## Gate 3: Ticket (only if needed)

A ticket is needed when the verdict is `bug`, `intended but confusing`, or `plausible risk` AND there is a concrete change to make.

No ticket for `working as intended` or `config / environment`. Answer the user and stop there, unless a doc change would prevent the confusion. That is an `intended but confusing` verdict, so file it.

Before filing, search open issues for a duplicate (`gh issue list -R Donelle/claude-runway --search "<key terms>" --state open`). If one exists, add your findings as a comment on it instead of filing a new one.

Otherwise run `/my-gh-create-issue` and feed it the Gate 1 findings directly. They become its Location and Sanity-checked lines, so it should not need to re-investigate. Respect its approval stop.

## Gate 4: Decide whether to work it now

Work it now only if ALL of these hold:
- Priority is `priority-p0` or `priority-p1`, OR the fix is small (roughly one file plus its tests) and low risk.
- It is not already assigned or linked to a branch or PR (`/my-gh-code-it` step 4 checks this too).
- It doesn't need a design decision only the user can make.
- It doesn't conflict with work in progress on the same files.

If all hold, run `/my-gh-code-it <issue>`, then `/my-gh-pr`.

If any fails, leave the ticket in the backlog and tell the user in one line which condition failed and what would unblock it.

If the user explicitly said "just investigate", stop after Gate 2 or Gate 3, whatever this gate would decide.

## Final report

One short message:
1. The verdict and the one-line reason.
2. The ticket link, or why no ticket was filed.
3. Worked now (with the PR link) or deferred (with the reason).
4. Anything the user needs to do.