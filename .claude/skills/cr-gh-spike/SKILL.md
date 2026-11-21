---
name: "cr-gh-spike"
description: "Run a spike in claude-runway: frame one question on a ticket of issue type Spike, investigate it read-only (experiments only in a throwaway copy), reach a go/no-go verdict, post the findings on the ticket as a comment, and offer follow-up tickets."
---

# Skill: cr-gh-spike

Use this when the user wants to find something out before committing to a change: "can we do X", "is Y feasible", "what would it take to Z", "which of these two approaches". A spike answers one question and produces a decision, not a feature. It runs four gates in order and never skips ahead: no investigation before the question is framed, no findings before a verdict, no follow-up ticket before the findings are posted.

Every spike has a ticket of GitHub issue type `Spike` (research whose outcome is findings and a go/no-go, not shipped code), and the findings are posted on that ticket as a comment, the same way `/cr-gh-pr` posts its research, plan and validate artifacts on the issue it works. The ticket is the trail: anyone who finds it later sees the question, the verdict and the evidence in one place.

How it differs from the neighbouring skills:
- `/cr-gh-triage` starts from a reported symptom ("why does X happen") and decides whether a bug needs a ticket. A spike starts from a question about a change that does not exist yet.
- `/cr-gh-code-it` implements a decided change. A spike never leaves code behind: it ends with a recommendation, and the implementation is a separate ticket.

This skill orchestrates the repo's existing skills; it does not replace them:
- `/cr-gh-create-issue` files the spike ticket when there is none, and any follow-up ticket (it keeps its own approval stop).

## Usage
```
/cr-gh-spike <issue-number>
/cr-gh-spike <the question, and the decision it feeds, if you know it>
```

## Gate 1: Frame

1. Confirm the repo with `git remote get-url origin` (expect `Donelle/claude-runway`). Stop and ask if it doesn't match.
2. Find the spike ticket.
   - Given an issue number: read the whole ticket with `gh issue view {N} -R Donelle/claude-runway --json number,title,state,labels,body,comments`, body and comments. Its **Summary** and **AI Instructions** usually already say the ask, what done looks like and what it is part of or depends on. Take the framing from there and do not ask the user what the ticket already answers. If it is closed, say so and ask whether to continue. Read its issue type with `gh api repos/Donelle/claude-runway/issues/{N} --jq '.type.name // empty'` (`gh issue view --json` has no type field; empty output means none is set). If the type is not `Spike`, say so and offer to set it (the PATCH command in the next bullet); continue either way.
   - Given only a question: search for an existing spike first, by issue type, with `gh issue list -R Donelle/claude-runway --state all --search "type:Spike <key terms>" --limit 50`, then the same search without the `type:Spike` qualifier over open issues for a ticket that already covers the question under another type. If one covers it, use it and say so. If none does, offer to file one with `/cr-gh-create-issue type=spike <the question>`, so the findings have somewhere to go. That skill writes the `Spike:` title and the spike-shaped body, applies the priority and theme labels (no `spike` label and no type label), keeps its own approval stop, and sets the Issue Type to `Spike` once the ticket exists. Check that it did, with the type read above; if it is empty, set it with `gh api repos/Donelle/claude-runway/issues/{N} --method PATCH -f type=Spike --jq '.type.name'`, which must print `Spike`. If the user declines a ticket, run the spike anyway and keep the write-up local (Gate 4 says how).
3. Write down the framing in a few lines, taken from the ticket when there is one:
   - **Question**: one question that has an answer, in a sentence the user can correct.
   - **Decision it informs**: what the user will do differently depending on the answer.
   - **Done looks like**: the evidence that would settle it (a measurement, a working call, a `file:line` fact, a documented limit). If the ticket has a "Done when" line, use it.
   - **Out of scope**: what you will not look into, even if it turns up.
4. Check what already exists before starting: `recall` (memory-bank) for past decisions on the topic, `cb_indexer_find_repo` for existing code or docs on it, and `.plans/` for an earlier spike on the same question.
5. Ask a clarifying question only for a point the request and ticket genuinely leave open or where their parts disagree, and ask all such points at once.

Show the framing to the user in a few lines and continue unless they object. There is no fixed time budget; the stop rule in Gate 2 is what keeps a spike from sprawling.

## Gate 2: Investigate

Read-only on the repository. Make no edits, commits, branches or issues in the user's checkout during this gate.

1. Start with the cheapest evidence: `cb_indexer_find_repo` for "where is X handled" questions, Grep for exact identifiers, Read for the spans the hits point at (the whole code path, not only the matching line), `git log` on the files for the design reasoning, and `docs/` and `templates/` for stated intent. Check any claim the ticket makes against the code instead of trusting it.
2. For an external dependency, API or limit, read the primary source (the library's own docs or source, the vendor's documentation) rather than answering from memory. Say which version or date the evidence is from.
3. When reading is not enough, run an experiment, but only in a throwaway place: a scratch directory in the session's scratchpad, or `git worktree add` on a temporary branch that you remove afterwards. Never run an experiment that writes to the user's working tree, `.mcp.json`, `~/.claude`, the Qdrant collections or the shared memory bank. If the question needs one of those, stop and ask. Delete what you created before the gate ends and say you did.
4. Keep a running list of facts. Each one carries its source (`file:line`, a command and its output, a URL) and a mark: **verified** (you ran it or read the exact code) or **inferred** (reasoned from adjacent evidence). Do not blur the two.
5. **Stop rule.** Stop and re-scope with the user, instead of continuing, as soon as any of these happens:
   - the question has changed from the one you framed, or splits into two;
   - you are on a third line of inquiry that was not in the framing;
   - you have the evidence "done looks like" asked for, or the evidence shows it cannot be had (then say what is missing and what would produce it).
   Say in two or three lines what you know, what you do not, and what you propose next. The user decides whether to extend.

## Gate 3: Conclude

Write the conclusion for the user before doing anything else. Lead with the answer:

- **Verdict**, exactly one of:
  - `go`: feasible and worth doing as described.
  - `go with changes`: feasible, but only with a different shape than the question assumed; say what changes.
  - `no-go`: not feasible, or the cost clearly outweighs the benefit; say which.
  - `inconclusive`: the evidence could not settle it; say what was tried and what would.
- **Why**: the deciding facts in plain words, each with its source and its verified/inferred mark.
- **Risks and unknowns**: numbered, one line each. Include anything you could not check.
- **Recommendation**: one recommended next step, not a menu. If the verdict is `go` or `go with changes`, include a rough size (one file plus tests, a few files, a cross-cutting change) and anything that must happen first.

## Gate 4: Post the findings and follow up

1. Write the findings to `.plans/{N}-spike.md`, where `{N}` is the spike ticket's number (the same naming `/cr-gh-pr` uses for `.plans/{N}-research.md`), or `.plans/spike-<short-slug>.md` when the user declined a ticket. `.plans/` is gitignored, so the file is a local working copy: never commit it, and never add a spike file anywhere tracked. Use the Write tool, not a shell heredoc. Write it for a reader of the ticket who was not in this session:
   - the verdict and the recommendation first;
   - the framed question, the decision it informs, and what was out of scope;
   - the facts, each with its source and its verified/inferred mark;
   - what was tried and discarded (a dead end written down is a re-investigation avoided);
   - risks and unknowns;
   - the date and the commit the evidence was taken from (`git rev-parse --short HEAD`).
2. Check the file before it goes anywhere. The repo is public, so the comment is too:
   - no credential, token or API key, and nothing copied out of `.mcp.json` or an environment file;
   - no absolute path that contains a user's home folder or name (use repo-relative paths), and no email address;
   - no name of a private project or employer; say "another project" instead;
   - command output is trimmed to the lines that carry the evidence, with counts kept exact.
3. Post it on the spike ticket, one comment:
   ```bash
   gh issue comment {N} -R Donelle/claude-runway --body-file .plans/{N}-spike.md
   ```
   GitHub rejects a comment over 65,536 characters. If the findings are longer, post the verdict, recommendation, facts and risks as the first comment and the dead ends and raw command output as a second, saying in the first that the second exists. Report the comment URL(s) from the command's output. With no ticket, skip this step and say the findings are local only.
4. Offer, do not do, the rest:
   - For a `go` or `go with changes` with a concrete change to make: offer `/cr-gh-create-issue`, feeding it the findings so they become its **Sanity-checked** and **Proposal** lines and naming the spike ticket as what it is **Part of** or **Depends on**. Search open issues for a duplicate first, and if one exists offer to comment on it instead.
   - For a verdict that answers the ticket (`go`, `go with changes`, `no-go`): offer to close the spike ticket as completed, with a one-line comment pointing at the findings. Leave an `inconclusive` spike open.
   - When the ticket's "Done when" names a change to a tracked file (a doc that should match what was found, for instance): point it out as a follow-up for `/cr-gh-code-it`. Do not make the edit here.
   - For a durable, non-obvious finding or design decision: ask first ("Is this something you'd like to remember for next time?"), then `remember` it with a summary phrased the way a future query would be.
5. Never start the implementation. If the user says to proceed, point them at the follow-up ticket and `/cr-gh-code-it`.

## Final report

One short message:
1. The verdict and the one-line reason.
2. The recommendation, with its rough size.
3. The URL of the comment on the spike ticket (or that the findings are local only, and the path of `.plans/spike-<slug>.md`), and which facts are inferred rather than verified.
4. Which follow-ups you are offering, and anything the user needs to do.
