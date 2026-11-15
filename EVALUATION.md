# Measuring Whether This Toolkit Actually Reduces Token Usage

The thesis behind this whole repo: **using Qdrant memory + local compression saves Claude token usage compared to not having them.** That's a testable claim, not something to assume — this doc is the plan for actually testing it, for both pieces this repo ships, without quietly hurting answer quality.

Two things need to be true, not just one, for either piece:

1. Tasks that benefit from the tool use fewer tokens with it than without.
2. That's not offset by the tool's own fixed cost — whatever having the extra MCP servers connected adds to a session's starting context, whether or not the tools get used. An earlier version of this doc assumed every connected server's tool schemas ride along on every turn; session transcripts on the author's machine show Claude Code listing this toolkit's tools as deferred (loaded on demand) in most sessions, so the real fixed cost has to be measured, not assumed (Step A4). If most of your work doesn't actually benefit, whatever fixed cost remains could still outweigh the wins on the tasks that do.

This repo has two independent core pieces (Qdrant memory, local compression) plus a third durable-memory layer (memory-bank) that shares the Qdrant piece's infrastructure but is independent of local compression, and a local code reviewer whose purpose is to take review rounds off the pull-request loop. There are seven tracks below (A-G), of four kinds:

- **Four controlled protocols (A, B, C, E)**, each testing a different mechanism on its own: run a task, run it again under the other condition, compare `/usage`.
- **One live diagnostic estimate (D)**, which is explicitly NOT a measured protocol — see its own intro.
- **One quality check (F)**, which is explicitly not a token measurement: it asks whether the local reviewers find what matters.
- **One observational measurement (G)**, the track that measures the review loop's cost: how many Copilot review rounds a pull request takes and what Claude spends answering each, read from real pull requests and session transcripts instead of from a controlled run.

Read "Measuring cleanly" first when you run A, B, C or E — it covers real mistakes made while first trying to test this. Track G is not run that way, and says what replaces those rules.

## Table of contents

- [Measuring cleanly (read this first)](#measuring-cleanly-read-this-first)
- [Track A: Qdrant codebase memory (`qdrant-find`/`qdrant-store` + `index_repo`/`sync_repo`)](#track-a-qdrant-codebase-memory-qdrant-findqdrant-store--index_reposync_repo)
  - [Step A1: Build a benchmark task set](#step-a1-build-a-benchmark-task-set)
  - [Step A2: Baseline run (setup disabled)](#step-a2-baseline-run-setup-disabled)
  - [Step A3: Treatment run (setup enabled)](#step-a3-treatment-run-setup-enabled)
  - [Step A4: Measure the fixed overhead](#step-a4-measure-the-fixed-overhead)
  - [Step A5: Extract transcript data (optional but recommended)](#step-a5-extract-transcript-data-optional-but-recommended)
- [Track B: Local compression (`compress_file`, `compress_command_output`, `fetch_url`)](#track-b-local-compression-compress_file-compress_command_output-fetch_url)
  - [Step B1: Build a benchmark task set](#step-b1-build-a-benchmark-task-set)
  - [Step B2: Baseline run (built-in tool)](#step-b2-baseline-run-built-in-tool)
  - [Step B3: Treatment run (local-compress tool)](#step-b3-treatment-run-local-compress-tool)
  - [Step B4: Measure the fixed overhead](#step-b4-measure-the-fixed-overhead)
  - [Step B5: What this track can't tell you](#step-b5-what-this-track-cant-tell-you)
- [Track C: Session continuity (`/cr-compact` + `/cr-resume`)](#track-c-session-continuity-cr-compact--cr-resume)
  - [The two comparisons worth making](#the-two-comparisons-worth-making)
  - [Step C1: Build a benchmark scenario](#step-c1-build-a-benchmark-scenario)
  - [Step C2: Comparison C1 run (native `/compact` vs. `/cr-compact`)](#step-c2-comparison-c1-run-native-compact-vs-cr-compact)
  - [Step C3: Comparison C2 run (`/cr-resume` vs. starting fresh)](#step-c3-comparison-c2-run-cr-resume-vs-starting-fresh)
  - [Step C4: What this track can't tell you](#step-c4-what-this-track-cant-tell-you)
- [Track D: the live per-session savings tracker (`/cr-savings`) — an estimate, NOT a fourth measured track](#track-d-the-live-per-session-savings-tracker-cr-savings--an-estimate-not-a-fourth-measured-track)
- [Track E: Durable memory reuse (`remember`/`recall`/`forget`)](#track-e-durable-memory-reuse-rememberrecallforget)
  - [Step E1: Build a benchmark scenario set (real, previously-researched findings)](#step-e1-build-a-benchmark-scenario-set-real-previously-researched-findings)
  - [Step E2: Baseline run (memory-bank absent/disabled)](#step-e2-baseline-run-memory-bank-absentdisabled)
  - [Step E3: Treatment run (memory-bank enabled)](#step-e3-treatment-run-memory-bank-enabled)
  - [Step E4: Measure the fixed overhead](#step-e4-measure-the-fixed-overhead)
  - [Step E5: remember:recall ratio (diagnostic only)](#step-e5-rememberrecall-ratio-diagnostic-only)
  - [Step E6: Why this track is harder to run than A-C](#step-e6-why-this-track-is-harder-to-run-than-a-c)
- [Track F: Local code review quality (not a token measurement)](#track-f-local-code-review-quality-not-a-token-measurement)
- [Track G: The cost of the review loop (Claude's tokens per review round, observed)](#track-g-the-cost-of-the-review-loop-claudes-tokens-per-review-round-observed)
  - [What it asks](#what-it-asks)
  - [What is measured](#what-is-measured)
  - [How to run it](#how-to-run-it)
  - [Where the numbers are](#where-the-numbers-are)
  - [What to compute](#what-to-compute)
  - [Like for like](#like-for-like)
  - [How it differs from Tracks A to E](#how-it-differs-from-tracks-a-to-e)
  - [What it cannot tell you](#what-it-cannot-tell-you)
  - [Baseline entry (not recorded yet)](#baseline-entry-not-recorded-yet)
- [Compute the result (all measured tracks)](#compute-the-result-all-measured-tracks)
- [Caveats](#caveats)

## Measuring cleanly (read this first)

Lessons from actually trying to run `/usage`-based comparisons, not theoretical caveats. They apply to the controlled tracks (A, B, C, E). Track G reads real pull requests and transcripts instead of running a task twice, so restarting `claude` and repeating a run do not apply to it; see "How it differs from Tracks A to E" there for what replaces them.

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

Separately, find out what just having the extra servers connected adds to a session's **starting context** -- the fixed prefix (system prompt, tool definitions that are sent up front, CLAUDE.md) the API processes before your first prompt. **Measure it from session transcripts; don't estimate it from tool schemas**. This doc and the tracker used to estimate it as "tool schema characters / 3.5, paid every turn regardless of use". Real transcripts don't support that premise: Claude Code can list MCP tools as *deferred* (names only, loaded on demand) instead of sending their schemas, and a deferred tool is not paid for every turn.

A transcript records two real things (see `libs/transcript_context_lib.py`):

- the first assistant turn's `usage` (`input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens`), the token count the API reported for the whole starting prefix; and
- `deferred_tools_delta` entries naming the tools held back from that prefix, plus `deferred_tools_record` entries carrying the full schema of tools that were later loaded.

One session gives one number for the whole prefix, so it cannot attribute a share to this toolkit. Two ways to get an answer, with different strength:

- **Controlled (the one to cite):** run two otherwise-identical sessions per model, same repo, same CLAUDE.md, same other MCP servers, same first prompt -- one with this toolkit's servers connected, one with them removed (`claude mcp remove` / a project `.mcp.json` without them) -- then run `python tools/measure_startup_context.py --with <with.jsonl> --without <without.jsonl>`. It reports, per model (models are never mixed), the mean starting context of each side and the difference. Repeat 2-3x per `Measuring cleanly` above.
- **Observational (context, not a measurement):** `python tools/measure_startup_context.py --survey` reads every transcript under `~/.claude/projects` and reports, per server, whether the toolkit's tools were listed as deferred at each session's first turn (the four always-on servers feed the per-server counts and the "all 4 servers" line; the opt-in `analytics` server's tools count as toolkit tools everywhere else, and it is listed on its own line), how large the schemas recorded as loaded were, and compares sessions that listed the toolkit's tools with sessions that listed none (which does not prove the toolkit was absent). Those are different sessions (different repos, CLAUDE.md files, MCP servers, conversation lengths), so the difference reflects all of that, not the toolkit's cost.

**What the transcripts on the author's machine showed (survey run 2026-10-12, 115 transcripts under `~/.claude/projects`; 7 had no assistant turn and were left out):**

- **The toolkit's tools were listed as deferred.** In 99 of the 115 transcripts at least one of the toolkit's tools was named in the first-turn deferred-tools list (not sent in the starting context). Per server: `qdrant` 98, `codebase-indexer` 97, `local-compress` 97, `memory-bank` 82, and all four core servers in 81 of those 99; what explains the sessions missing a server was not investigated. So "paid every turn" is not what these transcripts show for the servers that were listed. Two sessions called a toolkit tool that was not in the first-turn deferred list: either that schema was sent up front, or its server connected after the first turn; the transcript does not say which, so those are reported as unclassified, not as evidence either way.
- **On-demand loads are the part of the cost that is visible.** In 72 of the 108 sessions that parsed (whatever their first-turn classification) the transcript recorded at least one toolkit tool's schema as loaded on demand. Across all sessions that is 24 distinct tools, 53,051 characters of recorded schema in total, the largest (`index_repo`) 3,483 characters. Characters, not tokens: a token count depends on the model's vocabulary, which is not public. This is only the observed on-demand subset, not the complete cost: a schema sent up front leaves no such record, and its cost sits inside the first turn's usage, where one transcript cannot size it separately.
- **The controlled with/without measurement was NOT made.** No pair of otherwise-identical sessions with the toolkit connected vs. removed exists in the local transcripts, so there is no measured number for "what the toolkit adds to the starting context" beyond the deferral and loaded-schema facts above. The observational comparison is not a substitute, and its "none listed" group (8 sessions) is not verified to be toolkit-free: a connected toolkit whose schemas were sent up front and never called leaves no trace. It showed *lower* starting context for sessions that listed the toolkit's tools (about 61k vs. 67k tokens on `claude-sonnet-5-5`, 63k vs. 77k on `claude-sonnet-5`, with only 4 sessions in the other group on each model), which reflects the sessions differing, not the toolkit saving anything. This also covers only the Claude Code builds that wrote those transcripts; re-run the survey after an upgrade rather than assuming it still holds.

Keep the live tool count separate from this, because it is still useful and still drifts if hand-typed (a hardcoded count has gone stale before): run `python tools/report_tool_counts.py` for the current live count AND the size in characters of the tool descriptions (name + signature + docstring, a proxy for the JSON schema, not the schema itself) of `qdrant` (`qdrant-find`, `qdrant-store` and the three compact tools), `codebase-indexer`, `local-compress`, the opt-in `analytics` (the savings and metrics tools) and `memory-bank`, or read `codebase_indexer_mcp_server.py`'s stderr startup log (`[claude-runway] codebase-indexer registered N tools`). All five servers are this repo's own scripts, imported without a live Qdrant or embedding model (the `qdrant` server builds its model on first use). All of these go through `libs/mcp_tool_introspect.py`, so they can't drift relative to each other. Those are sizes of the schemas as written, not a per-turn cost. Where A4's measurement shows the toolkit adds a starting-context cost at all, that cost is what the savings on conceptual tasks have to beat.

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

Same question as Step A4, for the `local-compress` tools: what do they add to the starting context? Use A4's measurement (`tools/measure_startup_context.py`, controlled `--with`/`--without` for the number to cite, `--survey` for what real sessions show), not an estimate from the schema text. Per A4's survey, `local-compress`'s tools were listed as deferred, so they are not a fixed per-turn cost in that data. `local-compress` now holds five tools (`compress_file`, `compress_command_output`, `fetch_url`, `compress_text`, `list_local_models`): the savings and metrics tools it used to carry live on the opt-in `analytics` server, which a project that does not track savings never connects, so measure `local-compress` and `analytics` as separate servers (`python tools/report_tool_counts.py` shows both). A survey taken before that split counted those five tools under `local-compress`.

For the live tool count and the size in characters of the tool descriptions (name + signature + docstring, a proxy for the JSON schema; no token estimate involved), run `python tools/report_tool_counts.py`. **Don't hardcode a number in this paragraph** -- a hardcoded tool count here went stale once already, and a hardcoded doc count went stale a second time the same way, before both were replaced by the live introspection above.
The savings tracker no longer computes or shows a tool-overhead figure: its `schema_overhead_tokens` estimate and the "Tool overhead .../turn" line were removed because the estimate described a cost the transcripts show is not paid.

Hooks (`compress_output.py`, `redirect_webfetch_to_fetch_url.py`) do NOT add this kind of fixed cost — they're not tools Claude sees in its schema list, they intercept transparently, so they should be near-zero fixed overhead by design. Worth spot-checking this assumption once rather than just asserting it.

### Step B5: What this track can't tell you

Whether a built-in tool's own internal processing (e.g. WebFetch's server-side extraction) is itself billed to you is a separate, narrower question from "which approach costs less for an equivalent task" — Track B answers the second question directly by comparing total session cost, without needing to know the answer to the first. Testing attempted the narrower question directly and it turned out to likely be unanswerable from `/usage` alone (see "Measuring cleanly" above) — don't get stuck trying to resolve it; it isn't load-bearing for this repo's actual thesis.

## Track C: Session continuity (`/cr-compact` + `/cr-resume`)

This track tests a different claim than A and B: not "does the tool save tokens during a task" but "does resuming via Qdrant cost less than the alternatives across session boundaries." The `/cr-compact` skill fills in a structured summary of the current conversation and hands it to `compact_store`, which compresses it locally (LM Studio) and stores the result in Qdrant in the same call, and `/cr-resume` retrieves it at the start of the next session.

One structural advantage over Tracks A and B: if the `qdrant` server is already connected (which Track A requires), this workflow adds no tool schemas of its own -- the compact tools live on that same server, so there is no additional fixed cost.

### The two comparisons worth making

**Comparison C1: `/cr-compact` + `/cr-resume` vs. native `/compact`**

The native `/compact` command sends the conversation to Anthropic for summarization. `/cr-compact` routes the same job through LM Studio locally. Both produce a summary that seeds the next session — but the token cost profile differs:

- Native `/compact`: one Anthropic API call (input = full conversation, output = summary) billed to your account
- `/cr-compact`: one `compact_store` call whose compression runs on LM Studio (zero Anthropic tokens), then a later `compact_find` costs one tool round-trip; the summary is written once, not passed through your output a second time as a separate compression step would have required

Whether C1 shows a win depends on conversation length — short conversations may not generate enough summary-output tokens to offset the tool call overhead of `compact_store`.

**Comparison C2: `/cr-compact` + `/cr-resume` vs. starting fresh**

Without any compaction workflow, the alternative to a long session is abandoning context entirely and paying to re-establish it — re-reading key files, re-exploring the codebase, re-asking questions already answered. `/cr-resume` amortizes that re-establishment cost.

This comparison is harder to measure precisely because "re-establishing context from scratch" isn't a repeatable, fixed-cost operation — it depends on the task and how much the developer remembers. A practical proxy: measure the token cost of the first 3-5 turns of a fresh session (with `/cr-gh-load-context` as a stand-in for natural re-exploration) vs. the cost of `/cr-resume` alone.

### Step C1: Build a benchmark scenario

Pick one realistic multi-session task — something you'd naturally pause mid-way and resume later (a feature implementation, a debugging investigation). Split it at a natural stopping point. This gives you a real "session 1 → compact → session 2" sequence to measure.

### Step C2: Comparison C1 run (native `/compact` vs. `/cr-compact`)

1. Run session 1 of the task until the natural stopping point. Record `/usage`.
2. Run native `/compact`. Record the `/usage` delta (input + output tokens added by the compaction call).
3. Quit and relaunch. Paste the compact summary manually to seed session 2. Record the `/usage` cost of that seeding turn.
4. Repeat with `/cr-compact` instead: record the `/usage` delta for the `compact_store` round-trip, then in session 2 the `compact_find` round-trip cost.
5. Compare: total cross-session overhead of native path vs. `/cr-compact` path.

### Step C3: Comparison C2 run (`/cr-resume` vs. starting fresh)

1. Use the same session 1 transcript. Run `/cr-compact` at the end.
2. Quit and relaunch. Run `/cr-resume` and record its `/usage` cost (just the `compact_find` round-trip).
3. Separately, quit and relaunch without resuming. Run the natural re-establishment sequence (load CLAUDE.md, re-read key files, re-ask the first substantive question). Record that total `/usage`.
4. Compare: does `/cr-resume` undercut fresh re-establishment?

Repeat both comparisons 2-3x per "Measuring cleanly" above.

### Step C4: What this track can't tell you

- **Quality of the local summary vs. Anthropic's**: LM Studio's compression quality varies by model. A cheaper summary that loses key file paths or decisions is not a win — spot-check the restored context for correctness before counting a run as valid.
- **Multi-compact scenarios**: if you compact three times and resume the first, you pay `compact_find` retrieval overhead for a list + selection interaction. That's not captured in the single-compact measurement above. Worth a separate note if you use the workflow heavily.
- **Privacy benefit isn't measurable by `/usage`**: the primary motivation for `/cr-compact` may be keeping conversation content off Anthropic's infrastructure entirely, not just cost. That's real value that won't show up in any token comparison — account for it qualitatively when reporting results.

## Track D: the live per-session savings tracker (`/cr-savings`) — an estimate, NOT a fourth measured track

Unlike Tracks A–C, this isn't a protocol for you to run — it's a note about a *different kind of number* the repo now also produces, and why it must never be confused with the measured deltas above.

Tracks A–C all follow the same rigorous shape: run a task, run it again under the other condition, diff the two `/usage` readings. That's a controlled A/B measurement. The savings tracker (`libs/savings_ledger.py`, surfaced via `/cr-savings` and a `SessionEnd` hook) can't do that — there's no baseline run inside a single live session. Instead it reports an **online observation of a counterfactual**, in exact characters:

> `characters before (the raw content that was compressed)` vs. `characters after (the compressed result actually used)`, plus the percent reduction between them

This is directly observable for local-compression events only (`compress_file`/`compress_command_output`/`fetch_url`'s server-side compression, plus the `compress_output.py` hook), because the tool holds both sides of the comparison in hand — it read the raw content and produced the compressed result itself, so nothing about that specific delta is guessed. That's exactly why Track A's Qdrant piece is NOT part of this tracker: the counterfactual for `qdrant-find` (what Grep+Read would have cost instead) isn't observable the same way, so the tracker deliberately never labels Qdrant activity as "savings" — see "Measuring cleanly" above and the README's own qdrant-find-vs-Grep discussion for why that gap can't be closed by a heuristic.

Other differences from Tracks A–C worth being explicit about:

- **Characters, not tokens.** The tracker used to show estimated tokens (chars÷3.5), but Claude's real tokenizer isn't public and that ratio measured about 29% low, so it now shows exact character counts and the percent reduction, which is identical in either unit. What the removed characters would have cost in tokens is not measured — unlike `/usage`'s real, billed figures, which remain the only token numbers to trust. The one token figure the tracker does show is the opt-in per-model one (next bullet), and it is derived, not measured.
- **A fixed tool-schema cost is not reported by the tracker, and is not netted out of the headline.** The tracker used to show a "Tool overhead .../turn" line, a chars÷3.5 estimate of the toolkit's tool-schema size. It was removed: real session transcripts list these tools as deferred, loaded on demand instead of sent every turn, so the line described a cost that mostly isn't paid. What the toolkit adds to a session's starting context is measured from transcripts (Step A4, `tools/measure_startup_context.py`), not estimated. An earlier draft of this design also considered `net = gross_saved − (schema_tokens × turns)`; that was always wrong in the other direction too, since a cached prefix is read at roughly a tenth of full price, so subtracting at the full rate would have overstated the tax.
- **Per-model figures: one exact, one derived.** With `CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS=1`, `SessionEnd` joins each compression event to its transcript turn by `tool_use_id`, so the tracker can show characters before and after **per model**. Those counts are **exact**: the tool made them, and the join only chooses the row. It can also show a token figure per model, which is **derived**: the characters removed divided by that session's own observed chars-per-token for that model (growth in real context tokens against the characters of uncompressed tool results, median over at least 5 samples), counted once for every request that carried the result until the session ended or compacted, so it is comparable with the real "tokens processed" total. It is labelled as derived, shown with its ratio and sample size, and omitted when there are too few samples. It is still **not a measurement**: the uncompressed result never entered context and was never counted, the ratio is an observed median that absorbs whatever else grew the context, and the per-request counting assumes the result stayed in context. Never report it as a saving the way a Track A–C delta is reported; if a defensible token figure is needed, run the A/B protocol.
- **`fetch_url` is logged but never credited.** Its honest counterfactual is WebFetch's own already-compressed summary (see Track B above), not the raw page — crediting it against raw bytes would overstate savings for a comparison this repo already established isn't apples-to-apples.
- **Explicitly NOT a substitute for Tracks A–C.** If you want a rigorous, defensible savings number for a report or a decision about whether to keep using this toolkit, run the actual A/B protocol above. `/cr-savings` is for a quick, continuous, directional sense of "is this doing anything," not for a claim you'd want to defend under scrutiny.

## Track E: Durable memory reuse (`remember`/`recall`/`forget`)

This track tests the same core claim as every other track — does using the tool save tokens on a task — applied to memory-bank specifically. Two distinct mechanisms drive the savings, and either one alone is enough to produce a measurable win:

1. **Avoiding a repeated research cycle.** A past investigation that took real exploration (multiple tool calls, file reads, cross-referencing, back-and-forth) to reach a conclusion the first time can be retrieved with a single `recall` call in a later session, instead of that exploration happening all over again.
2. **A shorter answer than reconstructing it from scratch.** Even on a task where some exploration is unavoidable regardless, the stored `summary`/`description` a `remember` call captured is shorter than re-deriving and re-explaining the same reasoning and supporting evidence live.

Both mechanisms produce the same baseline-vs-treatment shape Tracks A–C use: fewer tokens for the treatment condition (memory-bank enabled, prior `remember` entry available) than the baseline (memory-bank absent, forced to re-derive from zero) on a task a fresh session would otherwise have to solve from scratch. (Track D, by contrast, has no baseline/treatment protocol at all — see its own intro above — so it's excluded from this comparison, not implicitly included.)

**Scope: this track only measures reuse, never the original encounter.** The very first time a finding is produced — the research cycle (or the mistake-then-correction cycle) that eventually becomes a `remember` entry — has to happen once regardless of whether memory-bank exists at all; that one-time cost is identical in both conditions, so it's outside the comparison. What's measured below is entirely about occasions *after* that `remember` entry already exists: does a later occasion cost less with it available (treatment) than without it (baseline)? If a scenario can't recur — if the finding only ever mattered once — there's nothing for this track to measure on it at all.

### Step E1: Build a benchmark scenario set (real, previously-researched findings)

Unlike Track A's 8-12 arbitrary questions, these have to be real findings pulled from actual session history (4-6 of them) — a fabricated "finding" tells you nothing about whether `recall` is actually shortcutting real work. Candidates include a root-cause investigation, a design decision with its rationale, or a correction — anything that took genuine multi-step exploration to reach the first time, was (or would be) `remember`-able, and is plausible to come up again in a later, unrelated session. For each scenario, record:

- The task/question that requires the finding, phrased so it's repeatable in a fresh session.
- The research path taken the first time (roughly how many tool calls/turns it took), so the baseline reproduction is realistic rather than trivially cheap.
- The corresponding `remember` entry (`summary`/`description`) as it would actually be stored.

### Step E2: Baseline run (memory-bank absent/disabled)

Fully quit/relaunch per scenario (see "Measuring cleanly" above), with the `memory-bank` server disconnected. Run the task as a later occasion would present it — Claude has no access to the earlier finding and must derive it fresh, the same exploration (or, for a correction-flavor scenario, the same class of mistake) a genuinely new encounter would require. Record `/usage` for that full derivation.

For a genuinely clean baseline, also temporarily remove (or comment out) the "Durable memory" section from the test project's `CLAUDE.md` for this run — same idea as Step A2's `.mcp.json` toggle for Track A. Otherwise Claude may still attempt a `recall` call despite the server being disconnected, which fails outright and adds an extra failed-tool-call turn the treatment condition doesn't have, skewing the comparison in memory-bank's favor for reasons unrelated to `recall`'s actual usefulness. Restore the section before running Step E3.

Note: "Measuring cleanly"'s "match turn structure" rule does NOT apply here the way it does in Track B, and this is deliberate, not an oversight — there, a turn-count mismatch between conditions is a confound to eliminate before it contaminates the comparison; here, baseline taking more turns than treatment (the full re-derivation vs. a single `recall` call) *is* the effect this track exists to measure. Padding treatment with extra turns or truncating baseline's would remove the exact thing being tested.

### Step E3: Treatment run (memory-bank enabled)

Same task, fresh session, with the corresponding `remember` entry already stored from a prior session. Record `/usage`. Judge the outcome from the transcript (see Step A5), not from the final answer's correctness alone — a right answer with no `recall` call in the transcript is not evidence of a win, it's unfalsifiable: there's no way to tell whether the model reasoned to the same conclusion independently on this particular run, coincidentally, or would have on every run. Classify as one of:

- Recalled and correctly applied (re-derivation avoided) — `recall` was called AND its result is what the answer is actually built on
- Recalled but ignored (Claude re-derived anyway — no savings on this run, but not the same failure mode as a retrieval miss)
- Not recalled (retrieval miss — falls back to the same cost as baseline)
- Wrong memory recalled

Only "recalled and correctly applied" is eligible to count as a win — same discipline Tracks A/B use to discard a cheaper wrong answer, sharpened here to also discard a cheaper answer that merely looks right without a verifiable `recall` behind it.

Report two numbers per scenario set, not one — averaging only over successes overstates memory-bank's real-world value, the same way reporting a cache's hit-rate-conditioned latency instead of its overall latency would:

- **Intention-to-treat result** (primary): `(baseline_tokens − treatment_tokens) / baseline_tokens`, computed from the REAL recorded `/usage` cost of every scenario actually run — never a substituted or assumed value for any outcome. A "not recalled" (retrieval miss) run's real cost will typically land close to baseline's since nothing else changed, but that has to fall out of the actual numbers, not be assumed. A "recalled but ignored" run costs MORE than baseline, not the same — treatment paid for the `recall` round-trip AND the full re-derivation, so its real recorded cost stays in the average as-is; zeroing it out would hide a real regression. "Wrong memory recalled" runs' real token cost also stays in this average — but since that outcome fails the correctness bar entirely, also report its rate/count separately as a distinct answer-quality risk, which the token number alone can't capture.
- **Success-conditioned result** (secondary, clearly labeled as such): the same formula, computed only over "recalled and correctly applied" scenarios — answers "how much does memory-bank save when retrieval actually works," a real but different question from the intention-to-treat number above.

### Step E4: Measure the fixed overhead

Same as Step A4/B4: measure the starting-context cost from transcripts with `tools/measure_startup_context.py` rather than estimating it from schema text. `memory-bank`'s tools (`remember`/`recall`/`forget`/`transfer_memories`/`copy_memories`/`list_memory_banks`) are among the servers the survey checks for deferral. `python tools/report_tool_counts.py`'s server loop includes `memory-bank` (`memory_bank_mcp_server`) alongside `local-compress` and `codebase-indexer` and reports its live tool count and tool-description size in characters (a proxy for the JSON schema), through the same shared `libs/mcp_tool_introspect.py` utility -- run it rather than hand-typing a number here, since a hand-typed count has gone stale before.

### Step E5: remember:recall ratio (diagnostic only)

Same role as Track D's savings estimate — a cheap diagnostic number, not a pass/fail result. It no longer needs deriving by hand from session transcripts: the `memory-bank` server now records its own usage. The correct source for a call-frequency ratio is `/cr-metrics memory-bank by_event_type` — a plain per-call-attempt tally in `metrics.db` (`libs/metrics_lib.py`), fired on every `recall`/`remember` invocation (including validation errors and empty results) whenever the shared, default-on `CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS` tracking flag is enabled. Treat it as a best-effort 1:1 call count, not a hard guarantee: that flag can be turned off, and the underlying `MetricsStore.record()` write deliberately fails open (an unwritable/locked/corrupt `metrics.db`, or a non-finite value, logs to stderr and returns `False` rather than raising or being retried), so a failed write silently leaves that one call uncounted. **`~/.claude/claude-runway/memory-events.db`'s `memory_events` table (`libs/memory_events_lib.py`, gated by the same flag) is NOT the right source for this ratio**: `recall()` logs one row per returned hit (zero rows for an empty or failed call, up to `limit` rows for one call with several hits), while `remember()` logs one row only after a successful write — so a naive `GROUP BY event_type` over that table counts point-level events, not tool invocations. `memory-events.db` is still the right source for the diagnostics this ratio doesn't cover — which specific memories get recalled, how often, how soon after creation (see [Memory bank](docs/memory-bank.md)'s "Usage event logging" section). A ratio heavily skewed toward `remember` suggests memories aren't being retrieved when relevant, but doesn't distinguish "nothing to recall yet" from "retrieval isn't working," so it shouldn't be reported as if it were Step E1-E3's result.

### Step E6: Why this track is harder to run than A-C

- **A correct treatment answer only counts as a win if `recall` is verifiably load-bearing in the transcript** — outcome correctness alone can't distinguish "recall saved the re-derivation" from "the model got there anyway on this particular run." This is why Step E3 classifies from the transcript rather than the final answer.
- **The re-derivation cost in baseline is inherently variable** (same caveat Track C2 raises about re-establishing context not being a fixed-cost operation) — a scenario that happens to be quick to re-derive understates the savings a harder scenario would show, so scenario selection matters more here than in Track A/B's more uniform lookups.
- **Small N is structural** — credible scenarios have to come from real, previously-researched findings, so this track won't reach Track A's 8-12-task sample size. Treat results as directional.
- **Reuse value compounds over a memory's lifetime in a way one baseline/treatment pairing can't capture** — a stored finding reused 5 times pays for itself 5 times over, but this protocol only measures a single reuse instance per scenario.

## Track F: Local code review quality (not a token measurement)

This track asks a different question from A to E: not "does it use fewer tokens" but "do the local-model reviewers find what matters, without crying wolf". It is measured by the scripts in `harness/` (see `harness/README.md`), which run the codebase reviewer against a real LM Studio. Nothing in `harness/` is run by CI.

- **Seeded cases** (`harness/cases.py`): small repositories with defects planted at known lines, correct code that merely looks suspicious (decoys), and a clean case. Each run reports which defects were found, which decoys drew a false alarm, and how many findings landed nowhere.
- **Parity with Copilot** (`harness/collect_copilot_history.py`, then `--history`): the existing Copilot reviews of a past PR are recorded read-only, the diff is reviewed again exactly as it stood at the commit Copilot reviewed, and the report says how many of Copilot's findings the local reviewers also raised, and how many further findings Copilot did not. The harness never requests or starts a Copilot review.
- **Model choice**: `harness/README.md` lists the recommended models for each role. Use an `-Instruct` build: a base model never ends its turn, and the harness's up-front probe reports that in seconds.

How to run it so the result means something:

- Compare changes on identical input: the same case, the same model, with `--runs N` greater than 1 once a reviewer's temperature is above 0.
- A finding Copilot did not raise is not automatically wrong. Treat the count as "to be judged", and judge a sample by hand.
- One model, one repository and a handful of cases: treat results as directional, as with Track E.

## Track G: The cost of the review loop (Claude's tokens per review round, observed)

**Status: not done until a run is recorded.** The protocol below is written and the data it needs is being recorded, but no baseline has been recorded yet (see "Baseline entry (not recorded yet)"), and nothing here is a result. Report this track as not done until that entry holds a real run.

This track asks a different question from A to F: not "does a tool shorten a task" but "what does a pull request cost in review rounds, and what does Claude spend answering each round". Every pull request in this repo is reviewed by Copilot after each push, Claude reads that review and answers it, and each fix is a push that starts another round. The local code reviewer (the opt-in `codebase-reviewer` server, Track F) exists to take rounds off that loop by catching findings before the push. Track F says whether it finds the right things; this track says what a round costs, so that the saving from fewer rounds can be stated instead of assumed.

### What it asks

What a pull request costs in Copilot review rounds and in **Claude's** tokens answering them, before and after the local reviewer is in use.

- **Copilot's own cost is not part of it.** It is not observable per review: a review carries no cost field, and the recorder captures none. No figure in this track is Copilot's cost or Copilot's tokens, and none should be reported as one. Wherever this track speaks of tokens spent on review rounds, they are Claude's.
- **The saving is asserted from the rounds that go away, and stated as a net.** The Copilot rounds that go away, times what a round costs Claude, minus what the local review rounds cost Claude. The local rounds are on the cost side: a local reviewer that removes two Copilot rounds but takes four local rounds to do it is not a saving until the arithmetic says so.
- **Local compute time is not a Claude token.** What a local review costs in time and electricity is reported separately, from the reviewer's own database (`python tools/codebase_reviewer_report.py review_times`, which splits each review's time into model loading and each stage; the `response_times` and `under_target` views and `--this-machine` are in its `--help`; see [Codebase reviewer](docs/codebase-reviewer.md#what-every-review-records)), and is never added to the token figures here.

### What is measured

- **Rounds per pull request**, of two kinds: a `copilot` round (one Copilot review, identified by the newest Copilot review a feedback fetch shows) and a `local` round (one call to the codebase-reviewer's `review` tool). A round runs from its first marker to the start of the next round of either kind, or to the merge. What counts as a feedback fetch, and why a fetch the reader cannot recognise starts no round, is in the docstring of `libs/pr_round_cost_lib.py` and in [Shared metrics store](docs/metrics.md#the-pr-review-rounds-events).
- **Per round, Claude's tokens in four fields**: `input_tokens`, `cache_creation_tokens`, `cache_read_tokens` and `output_tokens`. They are kept apart and never summed into one figure. This is the cache warning from "Caveats" applied: cache reads dominate by orders of magnitude, so a single blended total would mostly report how long the session's context was. Read each field with the cache split in view. (`output_tokens` in a transcript is a streaming-start snapshot, so treat it as a floor; see [Shared metrics store](docs/metrics.md#what-the-transcript-records-that-the-earlier-estimates-missed).)
- **Per round, alongside the tokens**: messages (distinct API responses), tool calls and wall clock, and the model with the most output tokens in the round.
- **Per Copilot round, whether the local reviewer was used** (`local_reviewer_used`, true or false). A round is: Copilot feedback arrives, changes are made (with the codebase-reviewer tool or without it), the changes are pushed. The flag is true when a `local` round (a call to the reviewer's `review` tool) starts strictly after that Copilot round began and strictly before the next Copilot round begins, and false when there is none (a `review` call issued in the same response as a Copilot round's first fetch shares its timestamp, so it is neither after nor before it and sets no flag). It belongs to the round, never to the pull request: a pull request can use the reviewer for its first rounds and then stop, and has true rounds and false rounds. A `review` call before the first Copilot feedback belongs to no Copilot round and sets no flag (it stays its own `local` row), and the final clean round, with no changes after it, is false unless a `review` call follows it. **False means "no local review captured for this round", not proof the tool was not used:** the flag is computed from the transcript that was read, so a review made in a session whose transcript is not read (subagent transcripts are never read, and the session-end capture is opt-in) shows as false, and the row is never corrected afterwards. A `review` call that names no pull request is attached to every pull request the session fetched feedback for, so it can set the flag on a round of another one.
- **Per row, which repository it is from** (`repo` and `repo_source`). `repo` is `owner/name` with `repo_source` `remote` when the recorder could name the repository (the `--repo` it was given, else the checkout's remote through `gh`), and otherwise the project folder name of the session's working directory with `repo_source` `project`, so a folder name is never read as a repository. With neither, both are null and the row is grouped on its own, never merged into a repository. A session started in a subfolder or a worktree gives that folder's name.
- **Per pull request, for like-for-like comparison**: additions, deletions and files changed, and per Copilot round the review effort, the inline-comment count and the open-issue count.

All of it is in `metrics.db` under `metric_id` `pr-review-rounds`, event types `pr_round` (one per round) and `pr_feedback_fetch` (one per fetch call, a finer grain of the same spend: use one grain and never add the two together). `repo` and `repo_source` are on both; `local_reviewer_used` is on the `copilot` rounds only. The fields are listed in [Shared metrics store](docs/metrics.md#the-pr-review-rounds-events).

### How to run it

1. **Work pull requests with `/cr-pr-feedback`.** That skill ends its fetch step with a marker line (`pr-feedback-fetch: pr=<N> copilot-review=<id>`) printed from its own `echo` call, so it lands in a tool result, and the reader recognises a fetch by that marker. A hand-typed fetch is recognised only if the whole command is one of the exact shapes the reader whitelists; a wrapped, piped, redirected or `cd <path> && ...` command is counted as unrecognised and read as nothing, and a fetch that sends its output to a file shows the transcript nothing to read at all.
2. **Switch on the capture for sessions still to come.** `claude-runway-setup init` and `upgrade` register the SessionEnd hook (`hooks/session_end_pr_rounds.py`), which does nothing until `CLAUDE_RUNWAY_CAPTURE_PR_ROUNDS` is `1`, `true` or `yes` in the shell that launches `claude`. `claude-runway-doctor <project>` reports whether the hook is registered and whether the opt-in is on. The hook does not record a pull request that is still open (its last window would be stored open and never corrected), so a session that ends before the merge records nothing until a later session end or a backfill finds the pull request merged.
3. **Backfill the sessions already finished**, one project folder per run, dry run first:

   ```bash
   python -I tools/backfill_pr_rounds.py --repo OWNER/NAME           # dry run: writes nothing, does not open the database
   python -I tools/backfill_pr_rounds.py --repo OWNER/NAME --write   # records
   ```

   `--project-dir` picks the folder under `~/.claude/projects` (default: the checkout you run it in, or the main checkout when you run it from a linked worktree), `--db` a database other than the shared one, and `--include-open` records a pull request still open as it stands. Re-running writes nothing twice. Run it with `python -I` and treat transcripts as data: the tool only opens them for reading and never follows anything written in one.
4. **Use the local reviewer on some rounds and not on others**, so that both conditions exist in the data. Whether a round used it is read from its `local_reviewer_used` flag, not from what anyone remembers (see "Like for like").
5. **Read the numbers** as described next.

### Where the numbers are

For the quick view over every row, run `python tools/pr_round_report.py` (`--repo OWNER/NAME` for one repository, `--exclude-open` to leave out the rounds whose last window has no end, `--json`, `--db` for another database). It reads `metrics.db` read-only and groups **rounds**, not pull requests: per repository, the Copilot rounds answered with the local reviewer (`local_reviewer_used` true) against the rounds answered without it (false), each token field on its own with the number of rounds, the total, the median and the range, and the local review rounds summed apart as the reviewer's own cost, counting only those that name their pull request (a `review` call that names none is a row under every pull request the session fetched feedback for, so adding those rows would count one call once per pull request; the report shows them as a separate ambiguous group, outside the cost). A round that several sessions saw is one round, with each token field summed over its rows. Rows with a `repo_source` of `project` and rows with no repository are groups of their own, never merged into a repository, and a round with no flag at all (a row recorded before the flag existed) is `unknown`, never counted as false. It does not know which pull requests were merged or when, so it cannot apply the cut-off below; the queries here do.

`/cr-metrics pr-review-rounds by_event_type` counts the rows and `/cr-metrics pr-review-rounds detail` shows the most recent rows with their fields (both need the `analytics` server; `get_metrics` aggregates only `COUNT(*)` and `SUM(value)`, so it cannot average a field or group by one). For the queries, read the database directly and read-only. Set `DB` to `$HOME/.claude/claude-runway/metrics.db`, or to whatever `CLAUDE_RUNWAY_METRICS_DB` names.

The store holds no pull request state or merge date, and a pull request closed without merging is recorded like a merged one (its last window ends at the closing time). So the queries take the pull requests to use from GitHub: the ones merged after the cut-off (see "Like for like"), as a JSON list in `MERGED`, and the repository they are from in `REPO` (`OWNER/NAME`). The queries read one repository at a time, because a pull request number is not unique across repositories; they leave out the rows of a project folder name and the rows with no repository, which the report above shows as their own groups. Run the first command from a checkout of the repository, or keep `--repo`:

```bash
REPO=OWNER/NAME
MERGED=$(gh pr list --repo "$REPO" --state merged --search "merged:>2026-09-28" --limit 1000 --json number --jq '[.[].number]')
```

Rounds per pull request, with its size, whether the transcripts saw every Copilot review, and how many of its Copilot rounds used the reviewer (a review that two sessions both fetched is two rows, so the query first folds the rows into rounds: one round per review, `used` when any of its rows says so):

```bash
sqlite3 -readonly -header -cmd ".parameter set @merged '$MERGED'" -cmd ".parameter set @repo '$REPO'" "$DB" <<'SQL'
WITH r AS (
  SELECT json_extract(metadata, '$.pr') AS pr,
         json_extract(metadata, '$.kind') AS kind,
         CASE WHEN json_extract(metadata, '$.kind') = 'copilot'
              THEN COALESCE(json_extract(metadata, '$.github_review_id'),
                            json_extract(metadata, '$.review_id'))
              ELSE session_id || ':' || json_extract(metadata, '$.kind_number') END AS round_key,
         json_extract(metadata, '$.copilot_review_number') AS review_number,
         json_extract(metadata, '$.local_reviewer_used') AS used,
         json_extract(metadata, '$.review_pr') AS review_pr,
         json_extract(metadata, '$.pr_additions') + json_extract(metadata, '$.pr_deletions') AS lines_changed,
         json_extract(metadata, '$.pr_changed_files') AS files_changed
  FROM metrics
  WHERE metric_id = 'pr-review-rounds' AND event_type = 'pr_round'
    AND json_extract(metadata, '$.repo') = @repo
    AND json_extract(metadata, '$.pr') IN (SELECT value FROM json_each(@merged))),
rounds AS (
  SELECT pr, kind, round_key, MAX(review_number) AS review_number, MAX(used) AS used,
         MAX(review_pr) AS review_pr, MAX(lines_changed) AS lines_changed, MAX(files_changed) AS files_changed
  FROM r GROUP BY pr, kind, round_key)
SELECT pr,
       COUNT(CASE WHEN kind = 'copilot' THEN 1 END) AS copilot_rounds_seen,
       MAX(review_number) AS highest_copilot_review_number,
       CASE WHEN COUNT(CASE WHEN kind = 'copilot' THEN 1 END) = 0 THEN 'no copilot rounds'
            WHEN COUNT(CASE WHEN kind = 'copilot' AND review_number IS NULL THEN 1 END) > 0 THEN 'gaps'
            WHEN COUNT(CASE WHEN kind = 'copilot' THEN 1 END) = MAX(review_number) THEN 'no gaps'
            ELSE 'gaps' END AS coverage,
       COUNT(CASE WHEN kind = 'copilot' AND used = 1 THEN 1 END) AS rounds_reviewer_used,
       COUNT(CASE WHEN kind = 'copilot' AND used = 0 THEN 1 END) AS rounds_reviewer_not_used,
       COUNT(CASE WHEN kind = 'copilot' AND used IS NULL THEN 1 END) AS rounds_flag_unknown,
       COUNT(CASE WHEN kind = 'local' AND review_pr = pr THEN 1 END) AS local_rounds_named,
       COUNT(CASE WHEN kind = 'local' AND review_pr IS NULL THEN 1 END) AS local_rounds_unnamed,
       MAX(lines_changed) AS lines_changed,
       MAX(files_changed) AS files_changed
FROM rounds GROUP BY pr ORDER BY pr;
SQL
```

One line per round, with each token field on its own (a round that several sessions saw is summed field by field, never across fields), for the same pull requests; `reviewer` is the round's flag (`used`, `not used`, or `unknown` when the row has none) and is empty on a `local` round, which is not given one:

```bash
sqlite3 -readonly -header -cmd ".parameter set @merged '$MERGED'" -cmd ".parameter set @repo '$REPO'" "$DB" <<'SQL'
WITH r AS (
  SELECT json_extract(metadata, '$.pr') AS pr,
         json_extract(metadata, '$.kind') AS kind,
         CASE WHEN json_extract(metadata, '$.kind') = 'copilot'
              THEN COALESCE(json_extract(metadata, '$.github_review_id'),
                            json_extract(metadata, '$.review_id'))
              ELSE session_id || ':' || json_extract(metadata, '$.kind_number') END AS round_key,
         json_extract(metadata, '$.copilot_review_number') AS review_number,
         json_extract(metadata, '$.local_reviewer_used') AS used,
         json_extract(metadata, '$.review_pr') AS review_pr,
         json_extract(metadata, '$.model') AS model,
         json_extract(metadata, '$.effort') AS effort,
         json_extract(metadata, '$.window_open') AS window_open,
         json_extract(metadata, '$.messages') AS messages,
         json_extract(metadata, '$.tool_calls') AS tool_calls,
         json_extract(metadata, '$.wall_clock_s') AS wall_clock_s,
         json_extract(metadata, '$.input_tokens') AS input_tokens,
         json_extract(metadata, '$.cache_creation_tokens') AS cache_creation_tokens,
         json_extract(metadata, '$.cache_read_tokens') AS cache_read_tokens,
         json_extract(metadata, '$.output_tokens') AS output_tokens
  FROM metrics
  WHERE metric_id = 'pr-review-rounds' AND event_type = 'pr_round'
    AND json_extract(metadata, '$.repo') = @repo
    AND json_extract(metadata, '$.pr') IN (SELECT value FROM json_each(@merged)))
SELECT pr, kind, MAX(review_number) AS copilot_review_number,
       CASE WHEN kind = 'copilot' THEN
         CASE MAX(used) WHEN 1 THEN 'used' WHEN 0 THEN 'not used' ELSE 'unknown' END
       END AS reviewer,
       CASE WHEN kind = 'local' THEN
         CASE WHEN MAX(review_pr) = pr THEN 'named' WHEN MAX(review_pr) IS NULL THEN 'unnamed' ELSE 'other' END
       END AS local_call,
       GROUP_CONCAT(DISTINCT model) AS models, MAX(effort) AS effort, MAX(window_open) AS window_open,
       SUM(messages) AS messages, SUM(tool_calls) AS tool_calls, SUM(wall_clock_s) AS wall_clock_s,
       SUM(input_tokens) AS input_tokens, SUM(cache_creation_tokens) AS cache_creation_tokens,
       SUM(cache_read_tokens) AS cache_read_tokens, SUM(output_tokens) AS output_tokens
FROM r GROUP BY pr, kind, round_key ORDER BY pr, kind, round_key;
SQL
```

SQLite has no median, so take the median and range of each column in whatever you analyse with (`sqlite3 -json` gives a form to load). Four things to apply when you do:

- **Coverage.** A recorded round is one the transcripts saw through a recognised fetch, so a missed fetch looks like a round that never happened, and fewer rounds would then look like a saving. `copilot_review_number` is the review's place among the pull request's Copilot reviews on GitHub, so a pull request whose `coverage` is `gaps` has seen fewer Copilot reviews than GitHub numbers up to its highest one (the check is conservative: a Copilot round with no review number, because GitHub was unreachable when it was recorded, makes the whole pull request read as `gaps`, since the rounds that remain can no longer be checked against each other). Take rounds per pull request only from pull requests with `no gaps`, and report the others separately (a `no copilot rounds` pull request is one of them: nothing was recorded for its Copilot reviews, which is not the same as having had none). Even `no gaps` is a lower bound: a missing review after the highest one seen leaves no trace in these rows, so where the count matters, check it against the pull request's reviews on GitHub (`gh api repos/OWNER/NAME/pulls/<N>/reviews`).
- **Reviewer used or not, per round.** Compare the Copilot rounds with `reviewer` `used` against those with `not used`; leave `unknown` rounds out of both groups and report them. Remember what false means (no local review captured for the round) and that a `review` call naming no pull request can set the flag on a round of another one: `local_rounds_unnamed` above is non-zero for a pull request exposed to that, so report such pull requests separately or leave them out. The local review cost is the `local` rounds whose `local_call` is `named`, for the pull requests that have `used` rounds; `unnamed` local rows are never added to it, because one such call is a row under every pull request the session fetched feedback for. A `local` round is never counted as a Copilot round.
- **Mixed models.** `models` lists every distinct model recorded for that round. A round with more than one was answered by sessions on different models: report it on its own or leave it out of a like-for-like comparison. Within one recorded row, `model` is the model with the most output tokens in the window, and `by_model` (shown by `/cr-metrics pr-review-rounds detail`) holds every model in it.
- **Open windows.** A row with `window_open` set is a floor, because its last window was recorded without an end time and is never corrected: leave it out of the medians, or report it on its own.

A `github_review_id` that GitHub could not supply is null, and the queries then fall back to the review id the transcript showed; a review shown under both its numeric and its `PRR_` form by different sessions could be counted twice, which shows as `copilot_rounds_seen` above the highest review number.

### What to compute

The same reading as "Compute the result (all measured tracks)", with these terms, at like pull-request size, each token field on its own. The unit of the tokens comparison is the **round**, grouped by its own `local_reviewer_used` flag: baseline is the Copilot rounds with the flag false, treatment the Copilot rounds with it true, so a pull request that used the reviewer for some rounds contributes to both. The unit of the rounds comparison is the **pull request**, grouped from those flags: baseline is a pull request with no round `used`, treatment one with at least one round `used` (the first query in "Where the numbers are" gives `rounds_reviewer_used` for each).

- **Rounds per pull request**: `(baseline − treatment) / baseline` over the median Copilot rounds per pull request of each group, reported with the range and the number of pull requests in each group. Use only pull requests whose `coverage` is `no gaps` (see "Where the numbers are"), leave out those whose rounds are all `unknown`, and say how many were left out.
- **Tokens per round**: the same formula for each of the four fields separately, over the Copilot rounds of each group (`tools/pr_round_report.py` prints the median and range per group); local rounds are reported on their own.
- **Net saving per pull request, per token field**: `(baseline Copilot rounds − treatment Copilot rounds) × the baseline's tokens per Copilot round − the treatment's tokens in the named local rounds of the pull request`. Every term is a field the recorder writes: the round kind and review id, the reviewer flag, the four token fields, and the pull request size that sets the comparison group. A positive figure in one field and a negative one in another is a result, not a contradiction to be averaged away.
- **Fewer rounds is a saving only if no more defects reach `main`.** This track cannot show that: it counts rounds and Claude's tokens, not what a round caught. Read it next to Track F, which measures detection and not cost. A round that goes away because the local reviewer raised the same finding earlier is a saving; a round that goes away because a defect was never found is not.

### Like for like

- **Compare at like pull-request size.** The data records additions, deletions and files changed. Group pull requests into size bands chosen before looking at the results and compare within a band: a pull request with ten times the changed lines will take more rounds whatever reviewed it.
- **The same cut-off, and only merged pull requests.** Only pull requests merged after 2026-09-28 are used, the day GitHub made Balanced the default Copilot review effort, so a comparison with any other counts-only baseline of Copilot rounds per pull request describes the same pull requests. The merge date comes from GitHub (the `MERGED` list in "Where the numbers are"), because the store records neither a merge date nor whether a pull request was merged or only closed; all of a selected pull request's recorded rounds are kept, including any that started before the cut-off.
- **Hold the Claude model and the Copilot review effort constant, or report them.** Each round carries `model` and, for a Copilot round, `effort`; a comparison across a model change or an effort change is two comparisons.
- **Whether the local reviewer was in use comes from the data, per round.** A Copilot round used it when its `local_reviewer_used` is true; the baseline is the rounds where it is false, which includes every round from the period before the reviewer was in use. A pull request is never put in a group from memory or intent, and is not put in one as a whole: it can have rounds in both. A pull request with a local round that names no pull request (`local_rounds_unnamed` above zero) may have had that call made for another one, so its cost cannot be placed: leave it out of both groups and report it separately.

### How it differs from Tracks A to E

It is observational: real pull requests, read from session transcripts and from GitHub, not a controlled run measured with `/usage`. So the "Measuring cleanly" rules about `/usage`, restarting `claude` between measurements and repeating each run 2-3 times do not apply. What replaces them:

- **Many pull requests, not repeated runs.** Noise is handled by the number of pull requests in each group, compared at like size.
- **A median and a range with the number of pull requests, never one average.** A few large rounds would dominate a mean, and a figure reported without its n cannot be judged.
- **Nothing is held fixed by whoever runs it.** The model, effort and size controls above are applied after the fact, from the data, not by arranging the run.

### What it cannot tell you

- **Copilot's own cost.** Not observable per review and not recorded; the saving is asserted from the rounds that go away, as a net with the local rounds on the cost side.
- **The local reviewer's compute time.** A cost in time and electricity, kept in the reviewer's own database, not a Claude token.
- **What else happened in a window.** A round's window holds anything else Claude did between two fetches (another pull request, an unrelated question) and leaves out a Copilot round's first few messages, before its first fetch call.
- **Subagent and autowork work.** Only main-session transcripts are read, so only manual sessions are in scope: a ticket worked by an autowork subagent is not counted.
- **Old pull requests.** Claude Code deletes old transcripts after about a month, so only recent pull requests have tokens. A pull request still open is not recorded until it is merged or closed.
- **Every Copilot review of a pull request.** A round is recorded only when a surviving transcript saw it through a recognised fetch, so the rounds per pull request counted here are a lower bound on what GitHub lists; the `coverage` column flags the gaps it can see and nothing flags the rest.
- **Which repository a row is from, when the recorder could not tell.** `metrics.db` is one database shared by every project, so each row carries `repo` and `repo_source`. A row whose `repo_source` is `project` holds a folder name, which names a working directory and not a repository (two checkouts of one repository, or a worktree, are different names); a row with no repository at all has a null `repo`. Both are reported in groups of their own and are never merged into a repository. The fields are written when the row is, and the store is append-only: a row recorded when the repository could not be named keeps its project name even if a later run could name it. A pull request number alone is not unique across repositories, so always group by `repo`.
- **A reviewer that was used where no transcript was read.** `local_reviewer_used` false means no local review was captured for the round, not that none was made (see "What is measured"), so the "not used" group can hold rounds that did use it.
- **More than one project folder per backfill.** The backfill covers one project folder per run.
- **Anything beyond one repository.** One repository's numbers are directional.

### Baseline entry (not recorded yet)

**What the protocol measures:** the rounds a pull request takes and, per round, Claude's four token fields, with and without the local reviewer in use. This entry is where the first real figures go.

**The real run's numbers (to be filled in once `python -I tools/backfill_pr_rounds.py --write` has been run on the real `metrics.db`; not run yet):**

| Item | Value |
| --- | --- |
| Date of the run | not run yet |
| Repository (`repo`) | not run yet |
| Pull requests recorded | not run yet |
| Copilot rounds per pull request (median, range, n) | not run yet |
| Tokens per Copilot round with `local_reviewer_used` false, each field on its own (median, range, n) | not run yet |
| Tokens per Copilot round with `local_reviewer_used` true, each field on its own (median, range, n) | not run yet |
| Local review rounds recorded and their tokens | not run yet |

**From here on:** new sessions are captured through the `/cr-pr-feedback` marker and the opt-in SessionEnd hook (`CLAUDE_RUNWAY_CAPTURE_PR_ROUNDS=1`), so the baseline grows from the point the hook is switched on, not backwards. Until the table above is filled in, and pull requests with local rounds exist beside it, this track has no result to report.

## Compute the result (all measured tracks)

- Per-task reduction = `(baseline_tokens − treatment_tokens) / baseline_tokens`
- Fixed overhead (Step A4 / B4 / E4, whichever applies to the track) is the measured starting-context difference from `tools/measure_startup_context.py`, reported alongside this reduction as its own separate figure, not subtracted from it into a single blended "net" number -- and where that measurement could not be made (A4 states which parts were not), say so rather than substituting an estimate
- Only count it as a real win if treatment correctness ≥ baseline correctness on the spot-checked tasks
- Report each measured track separately — nothing requires every piece to individually pay for itself at the same rate, since a project might only use a subset of them
- Track G is read differently, as set out in its own "What to compute" section: the reduction in rounds per pull request and in tokens per round (each of the four token fields on its own, never blended), at like pull-request size and as a median and range with the number of pull requests, and the net saving per pull request with the local review rounds counted as a cost. Its tokens are Claude's only, and its "correctness does not drop" condition is read next to Track F.

## Caveats

- Prompt caching (cache reads vs. fresh input) changes what's actually billed — a raw token count comparison can overstate or understate real cost savings. Look at the cache read/write breakdown in `/usage`, not just total cost.
- This benchmark reflects one repo's structure and question mix — results won't automatically generalize to a very different codebase, question style, or local model.
- Local-compress results (Track B) also depend heavily on which local model is loaded in LM Studio — a result with one model doesn't necessarily generalize to another, especially given real testing in this repo already found meaningful quality differences between models on the same task (see `fetch_url`'s positional-focus handling in `libs/local_compress_lib.py`'s docstrings for a concrete example).
- Track G's figures come from real pull requests, not a controlled run, so they carry whatever else was happening in each round's window, and they are Claude's tokens only: Copilot's own cost is not measured by anything here.
- Re-run this periodically as the repo grows; the case for semantic search gets stronger as a codebase gets too large to explore cheaply with Grep alone.
