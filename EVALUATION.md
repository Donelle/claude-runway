# Measuring Whether This Toolkit Actually Reduces Token Usage

The thesis behind this whole repo: **using Qdrant memory + local compression saves Claude token usage compared to not having them.** That's a testable claim, not something to assume — this doc is the plan for actually testing it, for both pieces this repo ships, without quietly hurting answer quality.

Two things need to be true, not just one, for either piece:

1. Tasks that benefit from the tool use fewer tokens with it than without.
2. That's not offset by the tool's own fixed cost — every connected MCP server's tool schemas ride along in context on **every single turn**, whether or not they're used. If most of your work doesn't actually benefit, that fixed tax could outweigh the wins on the tasks that do.

This repo has two independent pieces, so there are two independent tracks below. Read "Measuring cleanly" first regardless of which track you're running — it covers real mistakes made while first trying to test this.

## Measuring cleanly (read this first)

Lessons from actually trying to run `/usage`-based comparisons, not theoretical caveats:

- **`/clear` does not reset what `/usage` reports.** Running `/clear` resets the visible conversation, but `/usage`'s Session block appears to track the CLI process's lifetime, not the logical conversation. Confirmed by testing: three sequential `/clear`'d "sessions" showed input tokens climbing in an exact additive pattern (each exactly the sum of the ones before), which only makes sense if the counter wasn't resetting. **To get an independent reading, fully quit the `claude` process and relaunch it** — don't rely on `/clear` alone between measurements.
- **Tool-calling turns cost more than plain turns, structurally, regardless of the tool.** A prompt that triggers a tool call is at minimum two API round-trips (decide to call the tool, then process its result), each resending accumulated context. A prompt with no tool call is one round-trip. Comparing a tool-using baseline task against a non-tool-using treatment task (or vice versa) will mostly measure "tool calls cost more turns," not the thing you're trying to isolate. **Match turn structure between what you're comparing** — e.g. compare two different tools that both take exactly one call-and-result round-trip, not a tool call against a plain question.
- **Some costs may not be visible to `/usage` at all.** Claude Code's built-in tools (e.g. `WebFetch`) may run internal processing on Anthropic's own infrastructure using a different model. Check the "Usage by model" breakdown in `/usage` — if only one model ever appears there across your testing, that's evidence (not proof) that whatever a built-in tool does internally isn't being billed against your visible account usage, and no `/usage`-based test will be able to detect it either way. Don't chase this indefinitely if the breakdown never shows a second model — measure the outcome (does the session cost more or less) instead of the internal mechanism (why).
- **Repeat every measurement 2-3x.** Prompt caching, background token usage, and non-deterministic exploration paths all add noise to a single run.

## Track A: Qdrant codebase memory (`qdrant-find`/`qdrant-store` + `index_repo`/`sync_repo`)

### Step A1: Build a benchmark task set

Pick 8-12 real questions/tasks against the actual project repo, covering:

- **Conceptual questions** (best case for `qdrant-find`) — "where is retry logic handled," "why is the auth module structured this way"
- **Exact-match lookups** (Grep should still win) — "find all callers of `functionName`," "where does this exact error string come from"
- **A multi-step task** requiring real exploration — "fix bug X" or "add feature Y" — to see the effect on a realistic task, not just a single lookup

Keep the same list for every run so comparisons are apples-to-apples.

### Step A2: Baseline run (setup disabled)

1. Temporarily comment out or remove the `qdrant` and `codebase-indexer` entries from `.mcp.json` (or rename the file) so Claude falls back to pure Grep/Read.
2. Fully quit and relaunch `claude` per task (see "Measuring cleanly" above — don't rely on `/clear` alone).
3. After each task, run `/usage` and record the input/output/cache-read/cache-write breakdown.
4. Note whether the task was actually completed correctly — a cheaper wrong answer isn't a win.

### Step A3: Treatment run (setup enabled)

1. Restore `.mcp.json`. Run `sync_repo` first so the collection reflects the current repo state.
2. Run the exact same tasks, one per freshly-relaunched process, same repo state (ideally same day, so the codebase hasn't drifted between baseline and treatment).
3. Record `/usage` per task, and note which tools got called and in what order (visible inline, or in the saved transcript — see Step A5).
4. Same correctness check as Step A2.

Repeat each task 2-3x per condition per "Measuring cleanly" above.

### Step A4: Measure the fixed overhead

Separately, estimate the token cost of just having the two extra servers connected: token-count the six tool descriptions/schemas (`qdrant-find`, `qdrant-store`, `index_repo`, `sync_repo`, `preview_index`, `get_collection_info`) since that's paid every turn regardless of use. This is the number that has to be beaten by the savings on conceptual tasks.

### Step A5: Extract transcript data (optional but recommended)

Claude Code session transcripts live at `~/.claude/projects/<project-hash>/<session-id>.jsonl`. Each line includes per-turn token usage and tool calls, so instead of manually reading `/usage` output you can parse these to get exact per-task totals and confirm `qdrant-find` was actually chosen over `Grep` for the conceptual tasks. Ask if you want a script that does this.

## Track B: Local compression (`compress_file`, `compress_command_output`, `fetch_url`)

This track didn't exist in an earlier version of this doc, which only covered Track A despite the repo's core claim being about both pieces together. Same discipline applies: benchmark tasks, baseline vs. treatment, matched turn structure, multiple runs.

### Step B1: Build a benchmark task set

Pick 3 tasks, one per local-compress tool, each with a real large source:

- **`compress_command_output` vs. `Bash`**: a command with genuinely large output (a verbose test suite run, a build, `find` across a big directory).
- **`compress_file` vs. `Read`**: a large file already on disk (a saved log, a big generated diff).
- **`fetch_url` vs. `WebFetch`**: a real URL with substantial page content (same one used throughout this repo's testing: `https://en.wikipedia.org/wiki/Artificial_intelligence`), with the SAME focus/prompt across both conditions so the comparison is fair.

### Step B2: Baseline run (built-in tool)

1. Fully quit/relaunch `claude` per task.
2. Send a prompt that forces the built-in tool specifically (e.g. "Use Bash to run X," "Use Read to open Y," "Use WebFetch to fetch Z") — don't leave it to Claude's own judgment, since CLAUDE.md guidance may or may not be followed (see this repo's own README for that exact problem with `qdrant-find` vs. `Grep`).
3. Record `/usage` immediately after.

### Step B3: Treatment run (local-compress tool)

1. Fully quit/relaunch `claude` per task.
2. Send the equivalent prompt forcing the local-compress tool specifically (e.g. "Use compress_command_output to run X," "Use fetch_url to fetch Z with focus...").
3. Record `/usage` immediately after.
4. Note whether LM Studio was actually running and reachable — a treatment run where the tool silently fails open (falls back to returning the original, uncompressed content) isn't a real treatment sample.

Repeat 2-3x per condition per "Measuring cleanly" above.

### Step B4: Measure the fixed overhead

Token-count the five `local-compress` tool schemas (`compress_file`, `compress_command_output`, `fetch_url`, `compress_text`, `list_local_models`) — paid every turn regardless of use, same as Track A's servers. Hooks (`compress_bash_output.py`, `redirect_webfetch_to_fetch_url.py`) do NOT add this kind of fixed cost — they're not tools Claude sees in its schema list, they intercept transparently, so they should be near-zero fixed overhead by design. Worth spot-checking this assumption once rather than just asserting it.

### Step B5: What this track can't tell you

Whether a built-in tool's own internal processing (e.g. WebFetch's server-side extraction) is itself billed to you is a separate, narrower question from "which approach costs less for an equivalent task" — Track B answers the second question directly by comparing total session cost, without needing to know the answer to the first. Testing attempted the narrower question directly and it turned out to likely be unanswerable from `/usage` alone (see "Measuring cleanly" above) — don't get stuck trying to resolve it; it isn't load-bearing for this repo's actual thesis.

## Compute the result (both tracks)

- Per-task reduction = `(baseline_tokens − treatment_tokens) / baseline_tokens`
- Net result = per-task savings across the benchmark, minus the fixed overhead (Step A4 / B4) for that track
- Only count it as a real win if treatment correctness ≥ baseline correctness on the spot-checked tasks
- Report the two tracks separately — nothing requires both pieces to individually pay for themselves at the same rate, since a project might only use one piece

## Caveats

- Prompt caching (cache reads vs. fresh input) changes what's actually billed — a raw token count comparison can overstate or understate real cost savings. Look at the cache read/write breakdown in `/usage`, not just total cost.
- This benchmark reflects one repo's structure and question mix — results won't automatically generalize to a very different codebase, question style, or local model.
- Local-compress results (Track B) also depend heavily on which local model is loaded in LM Studio — a result with one model doesn't necessarily generalize to another, especially given real testing in this repo already found meaningful quality differences between models on the same task (see `fetch_url`'s positional-focus handling in `libs/local_compress_lib.py`'s docstrings for a concrete example).
- Re-run this periodically as the repo grows; the case for semantic search gets stronger as a codebase gets too large to explore cheaply with Grep alone.
