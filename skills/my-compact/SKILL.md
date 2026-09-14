# Skill: my-compact

Locally summarize and persist the current conversation to Qdrant so it can be restored in a fresh session with `/my-resume`. Everything stays local — the summarization runs on LM Studio, and storage goes to Qdrant. No Anthropic tokens are spent on the summarization step.

Optional argument: a short label to identify this compact (e.g., `/my-compact "auth middleware refactor"`). If omitted, `compact_store` derives the label server-side from the first sentence of "## What we were working on" in the structured summary — you do not need to extract it yourself.

## When to use

Run this when the conversation is growing long and you want to continue in a fresh context without losing what was accomplished.

## Steps

1. **Identify the project**: Note the current working directory name (e.g., `Acme.Support.TicketsApi`). This becomes the `project` tag in Qdrant so `/my-resume` can find the right entry later.

2. **Determine the label**:
   - If an argument was provided: use it as-is (e.g., `"auth middleware refactor"`)
   - If no argument: omit `label` from the `compact_store` call (or pass `""`). The server derives it automatically from the first sentence of "## What we were working on" (issue #72), so you do not need to extract it.

3. **Build a structured conversation summary** by filling in this exact template from what you can see in context:

```
PROJECT: <working directory name>
DATE: <today's date in ISO format YYYY-MM-DD, e.g. 2026-07-22>
LABEL: <label from step 2, or omit/leave blank if using server-side derivation>

## What we were working on
<1-3 sentences describing the task, feature, or bug>

## Key decisions made
<bullet list — include rationale when notable>

## Current state / progress
<what was completed, what is in-flight, any blockers>

## Open tasks / next steps
<what remains to be done, in order>

## Important files and locations
<file paths, line numbers, class or function names that are directly relevant>

## Unresolved questions
<anything left open, unclear, or deferred>
```

4. **Compress the summary** by calling `compress_text` with:
   - `text`: the structured summary from step 3
   - `preserve_identifiers`: `true` — required here, not optional. The focus below asks for verbatim identifiers, but asking is not enforcing: a real 9,876-char handoff came back with the `Options` segment silently dropped from a colon-delimited config key and a stray triple-quote. This flag re-appends, verbatim, any identifier the model dropped.
   - `preserve_sections`: `true` — also required. The same handoff lost three *entire* sections (all of its unresolved questions, all of its environment notes, and a user instruction) while still reading as complete. A single call over the whole summary can only get shorter by discarding content, and whole sections are the cheapest thing to discard. This compresses one section at a time, passes the exactness-critical sections through untouched, and re-emits every heading from the source so none can vanish.
   - `focus`: >
       Create a high-density handoff summary for resuming this session in a new agent instance.

       CRITICAL EXTRACTION REQUIREMENTS:
       1. Verbatim Technical Identifiers: Retain exact file paths, class/function/variable names, CLI commands, and database schema keys without modification or generic substitution (e.g., do NOT replace `src/utils/auth.ts` with "the auth file").
       2. Session Goal & Intent: Core objective the user wants to achieve.
       3. Decisions & Rationales: Agreed architectural choices, solutions implemented, and explicitly rejected alternatives.
       4. State & Next Steps: Current execution state, open errors/blockers, and immediate actions needed to resume work immediately.
       5. User Constraints & Rules: Specific guidelines, preferences, or boundaries established by the user.

       STYLE: Maximum signal-to-noise ratio. Eliminate conversational filler; keep the technical specificity needed to continue the work. Tighten prose, do not drop facts.

   If LM Studio is not available and `compress_text` fails, skip this step and use the uncompressed summary from step 3 directly.

5. **Store in the dedicated compacts collection** by calling `compact_store` with:
   - `information`: the output from step 4 (or step 3 if step 4 was skipped)
   - `project`: the working directory name from step 1
   - `label`: the label from step 2 — omit this parameter (or pass `""`) if no argument was given; the server derives it automatically from the summary (issue #72)
   - `date`: today's date

   Note: `compact_store` uses a dedicated per-project Qdrant collection (e.g. `conversation-compacts-ProjectX`), isolated from the codebase index. Do NOT use `qdrant-store` here — that writes to the shared codebase collection where compacts get buried under code chunks.

6. **Check the result before confirming anything.** `compact_store` returns exactly one of
   two shapes: a success string starting with `Stored compact in '...'`, or an error string
   starting with `Error:` (currently only a non-ISO `date`, but treat any `Error:`-prefixed
   return the same way — future error paths shouldn't need this check updated). Do NOT
   assume success and confirm regardless — issue #115 was exactly this: a silent
   `date`-validation failure looked identical to a real save, so the user saw a normal
   "stored successfully" confirmation and the compact was never actually stored.
   - **If it starts with `Error:`** — stop here. Show the user the exact error text and do
     NOT print step 7's confirmation. Tell them the compact was NOT saved and they'll need
     to re-run `/my-compact` after the underlying issue is fixed (e.g. correct the date).
   - **If it starts with `Stored compact in`** — proceed to step 7.

7. **Confirm** to the user:

   > Conversation compacted and stored.
   > Project: `<name>` | Date: `<date>` | Label: `<label>`
   > Run `/clear` then `/my-resume` in the new session to restore context.
