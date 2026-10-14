# Skill: my-compact

Locally summarize and persist the current conversation to Qdrant so it can be restored in a fresh session with `/my-resume`. Everything stays local — `compact_store` (on the `qdrant` MCP server) compresses the summary on LM Studio and stores it in Qdrant, in one call. No Anthropic tokens are spent on the compression step, and the summary passes through your output only once.

Optional argument: a short label to identify this compact (e.g., `/my-compact "auth middleware refactor"`). If omitted, `compact_store` derives the label server-side from the first sentence of "## What we were working on" in the structured summary — you do not need to extract it yourself.

## When to use

Run this when the conversation is growing long and you want to continue in a fresh context without losing what was accomplished.

## Steps

1. **Identify the project**: Note the current working directory name (e.g., `Acme.Support.TicketsApi`). This becomes the `project` tag in Qdrant so `/my-resume` can find the right entry later.

2. **Determine the label**:
   - If an argument was provided: use it as-is (e.g., `"auth middleware refactor"`)
   - If no argument: omit `label` from the `compact_store` call (or pass `""`). The server derives it automatically from the first sentence of "## What we were working on", so you do not need to extract it.

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

4. **Store it by calling `compact_store`** (on the `qdrant` MCP server) with:
   - `information`: the full structured summary from step 3, exactly as you filled in the template. Do NOT compress it yourself first: `compact_store` compresses it on the local model before storing it, and it asks for what a handoff needs (every file path, identifier and command kept verbatim, every `##` section kept). A second compression pass here would only throw away what that one preserves.
   - `project`: the working directory name from step 1
   - `label`: the label from step 2 — omit this parameter (or pass `""`) if no argument was given; the server derives it automatically from the summary
   - `date`: today's date

   If LM Studio is not available, `compact_store` stores the summary uncompressed instead of failing, and its reply says so; nothing for you to do. Note: `compact_store` uses a dedicated per-project Qdrant collection (e.g. `conversation-compacts-ProjectX`), isolated from the codebase index. Do NOT use `qdrant-store` here — that writes to the shared codebase collection where compacts get buried under code chunks.

5. **Check the result before confirming anything.** `compact_store` returns exactly one of
   two shapes: a success string starting with `Stored compact in '...'` (ending with a
   parenthesized note saying whether the summary was compressed or stored uncompressed, and why), or an error string
   starting with `Error:` (currently only a non-ISO `date`, but treat any `Error:`-prefixed
   return the same way — future error paths shouldn't need this check updated). Do NOT
   assume success and confirm regardless — this has happened before: a silent
   `date`-validation failure looked identical to a real save, so the user saw a normal
   "stored successfully" confirmation and the compact was never actually stored.
   - **If it starts with `Error:`** — stop here. Show the user the exact error text and do
     NOT print step 6's confirmation. Tell them the compact was NOT saved and they'll need
     to re-run `/my-compact` after the underlying issue is fixed (e.g. correct the date).
   - **If it starts with `Stored compact in`** — proceed to step 6.

6. **Confirm** to the user:

   > Conversation compacted and stored.
   > Project: `<name>` | Date: `<date>` | Label: `<label>`
   > Run `/clear` then `/my-resume` in the new session to restore context.

   If the success note in step 5 said the summary was stored uncompressed, add one line saying so, with the reason it gave (for example, LM Studio was not reachable). The compact is complete either way; it is just larger.
