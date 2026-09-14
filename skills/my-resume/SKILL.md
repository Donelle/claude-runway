# Skill: my-resume

Restore a compacted conversation that was previously saved by `/my-compact`. Run this at the start of a fresh session (after `/clear`) to pick up where you left off.

Optional argument: a keyword or phrase to narrow which compact to find (e.g., `/my-resume auth refactor`). If omitted, the most recent compacts for this project are listed (up to the tool's default limit) and you pick one.

## Steps

1. **Identify the project**: Note the current working directory name.

2. **Call `compact_find`** with:
   - `project`: the working directory name from step 1
   - `query` (optional): the argument if one was provided (e.g., `"auth refactor"`)

   Note: `compact_find` filters by `project` using Qdrant's native payload filter — it only returns compacts for this project. When no `query` is provided, results are sorted by date descending. When a `query` keyword is provided, results are relevance-ordered (most semantically similar first). Do NOT use `qdrant-find` here — that searches the shared codebase collection where compacts are buried under code chunks.

3. **Handle results by count**:

   **0 results** → go to "If no match is found" below.

   **Exactly 1 result** → skip to step 4 immediately. Do not ask the user to confirm — just restore it.

   **2 or more results** → use `AskUserQuestion` to present a selection UI. `AskUserQuestion` hard-caps every question at 4 options total (verified directly against the Claude Code CLI's own bundled schema) — if `compact_find` returned more than 4 results, present only the top 4 (already sorted most-recent-first, or most-relevant-first when a `query` was given, so the top 4 are the most useful ones anyway) and mention in the question text that N additional results exist beyond those shown (NOT "older" -- when a `query` was given, the cut results are lower-ranked by relevance, not necessarily older, so calling them "older" would misinform the user). Each option should be:
   - `label`: `<date> — <label from metadata>`
   - `description`: the "What we were working on" excerpt from the stored summary

   Note: each entry's header from `compact_find` includes a short `(id: ...)`
   suffix — this exists so that two entries which otherwise render
   identically (same date and label; possible for compacts stored before
   issue #36's idempotency fix) can still be told apart. If two or more
   options in the selection UI would otherwise have an identical `label`,
   append that suffix to each one's `label` (e.g. `<date> — <label> (id:
   <suffix>)`) so the user can distinguish them; omit it when every
   option's `label` is already unique.

   Wait for the user's selection, then proceed to step 4 with the chosen entry.

4. **Present the restored context** in this format:

   ```
   ## Restored from previous session (<date>)
   Project: <project name> | Label: <label>

   <full content of the selected entry>
   ```

5. **Pre-load key files from the compact** — do this now, before inviting continuation:

   a. Locate the `## Important files and locations` section in the restored compact. If it
      is absent or empty, skip steps 5b–5d and go directly to step 6.

   b. Extract file paths from that section. A file path is any token that:
      - Starts with `/`, `~/`, `./`, a bare filename component followed by `/` (e.g.
        `libs/`, `tools/`, `skills/`), or looks like `<word>/<word>` with an extension
      - Contains a recognizable file extension (`.py`, `.md`, `.ts`, `.tsx`, `.js`,
        `.json`, `.yaml`, `.yml`, `.toml`, `.txt`, `.sh`, `.cs`, `.go`, `.rb`, `.sql`)
      - Is NOT just a symbol name, class name, or function name without a `/`

      Take up to **5** unique paths in the order they appear.

   c. **Scope check — skip any path whose canonical resolution falls outside the current
      project.** A textual prefix check is not sufficient: a relative token like
      `dir/../../private/secrets.json` would pass a "no leading /" test yet escape the
      working tree, and an in-tree symlink can resolve to a target outside it. Before calling
      `Read`, canonically resolve the path and verify the result stays inside the current
      working directory:

      1. Resolve the path to its real, absolute form (follow symlinks, resolve `..`
         components) — e.g. `os.path.realpath(os.path.join(cwd, path))` in Python.
      2. Confirm the resolved path starts with the resolved current working directory
         (ensure a trailing separator is included in the prefix check to prevent
         `/project-data/...` from matching a project rooted at `/project`).
      3. If the resolved path escapes the project root for any reason — traversal, symlink,
         or absolute/home-relative prefix — skip it silently.

      For each path that passes the canonical scope check, call `Read` immediately — do NOT
      ask for permission or wait for user input. If a path does not exist or is not
      accessible, skip it silently without reporting an error.

   d. After reading, append a compact summary under the restored context:

      ```
      **Pre-loaded from Important files and locations:**
      - `<path>` ✓
      - `<path>` ✓
      ```

      List only the paths that were successfully read. If none were readable, omit this
      block entirely.

6. **Invite continuation**: Ask the user: "I've restored the context from your previous session. Where would you like to pick up?"

## If no match is found

If `compact_find` returns no results for this project (a "No compacts
collection found" or "No compacts found" message), check whether that
message also includes a "Did you mean one of these other projects..."
section (issue #37) — `compact_find` looks for sibling projects with saved
compacts whenever the current project comes up empty, since an empty
result usually means either "never compacted" or the project name has
drifted from whatever it was when /my-compact ran in a way that ISN'T just
casing (a renamed working directory, a different checkout/worktree — a
pure casing difference is already resolved transparently upstream and
never produces this candidate list at all, since `compact_find` finds
matching history under any casing on its own).

- **Candidates listed** → present them to the user via `AskUserQuestion`
  (one option per candidate project, `description` showing its most recent
  date/label) plus an explicit "None of these — start fresh" option.
  `compact_find` caps this list at 3 candidates specifically so that 3
  candidates + 1 "None of these" option stays within `AskUserQuestion`'s
  hard 4-option-per-question ceiling (verified directly against the Claude
  Code CLI's own bundled schema) — never add more options than what
  `compact_find` actually returned. If the user picks a candidate, re-call
  `compact_find` with that candidate's project name and proceed to step 3
  above with the new results.
- **No candidates listed** (the message has no such section) → say:

  > No compacted conversations found for project `<name>`. Either `/my-compact` hasn't been run yet, or the working directory name changed between sessions.
  > Run `/my-compact` at the end of this session to start the cycle.

## Notes
- This is read-only — do not write, modify, or re-index anything.
- Do not confuse this with `/my-load-context`, which loads codebase documentation, not a prior conversation.
