# Measuring Whether This Toolkit Actually Reduces Token Usage

The thesis behind this whole repo: **using Qdrant memory + local compression saves Claude token usage compared to not having them.** That's a testable claim, not something to assume — this doc is the plan for actually testing it, for both pieces this repo ships, without quietly hurting answer quality.

Two things need to be true, not just one, for either piece:

1. Tasks that benefit from the tool use fewer tokens with it than without.
2. That's not offset by the tool's own fixed cost — whatever having the extra MCP servers connected adds to a session's starting context, whether or not the tools get used. An earlier version of this doc assumed every connected server's tool schemas ride along on every turn; session transcripts on the author's machine show Claude Code listing this toolkit's tools as deferred (loaded on demand) in most sessions, so the real fixed cost has to be measured, not assumed (Step A4). If most of your work doesn't actually benefit, whatever fixed cost remains could still outweigh the wins on the tasks that do.

This repo has two independent core pieces (Qdrant memory, local compression) plus a third durable-memory layer (memory-bank) that shares the Qdrant piece's infrastructure but is independent of local compression — five tracks below (A-E): four measured protocols (A, B, C, E), each testing a different mechanism on its own, plus one live diagnostic estimate (D, which is explicitly NOT a measured protocol — see its own intro). Read "Measuring cleanly" first regardless of which track you're running — it covers real mistakes made while first trying to test this.

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
- [Track C: Session continuity (`/my-compact` + `/my-resume`)](#track-c-session-continuity-my-compact--my-resume)
  - [The two comparisons worth making](#the-two-comparisons-worth-making)
  - [Step C1: Build a benchmark scenario](#step-c1-build-a-benchmark-scenario)
  - [Step C2: Comparison C1 run (native `/compact` vs. `/my-compact`)](#step-c2-comparison-c1-run-native-compact-vs-my-compact)
  - [Step C3: Comparison C2 run (`/my-resume` vs. starting fresh)](#step-c3-comparison-c2-run-my-resume-vs-starting-fresh)
  - [Step C4: What this track can't tell you](#step-c4-what-this-track-cant-tell-you)
- [Track D: the live per-session savings tracker (`/my-savings`) — an estimate, NOT a fourth measured track](#track-d-the-live-per-session-savings-tracker-my-savings--an-estimate-not-a-fourth-measured-track)
- [Track E: Durable memory reuse (`remember`/`recall`/`forget`)](#track-e-durable-memory-reuse-rememberrecallforget)
  - [Step E1: Build a benchmark scenario set (real, previously-researched findings)](#step-e1-build-a-benchmark-scenario-set-real-previously-researched-findings)
  - [Step E2: Baseline run (memory-bank absent/disabled)](#step-e2-baseline-run-memory-bank-absentdisabled)
  - [Step E3: Treatment run (memory-bank enabled)](#step-e3-treatment-run-memory-bank-enabled)
  - [Step E4: Measure the fixed overhead](#step-e4-measure-the-fixed-overhead)
  - [Step E5: remember:recall ratio (diagnostic only)](#step-e5-rememberrecall-ratio-diagnostic-only)
  - [Step E6: Why this track is harder to run than A-C](#step-e6-why-this-track-is-harder-to-run-than-a-c)
- [Compute the result (all measured tracks)](#compute-the-result-all-measured-tracks)
- [Caveats](#caveats)

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

Keep the live tool count separate from this, because it is still useful and still drifts if hand-typed (a hardcoded count has gone stale before): run `python tools/report_tool_counts.py` for the current live count AND the size in characters of the tool descriptions (name + signature + docstring, a proxy for the JSON schema, not the schema itself) of `qdrant` (`qdrant-find`, `qdrant-store` and the three compact tools), `codebase-indexer`, `local-compress`, the opt-in `analytics` (the savings and metrics tools) and `memory-bank`, or read `ingest_mcp_server.py`'s stderr startup log (`[claude-runway] codebase-indexer registered N tools`). All five servers are this repo's own scripts, imported without a live Qdrant or embedding model (the `qdrant` server builds its model on first use). All of these go through `libs/mcp_tool_introspect.py`, so they can't drift relative to each other. Those are sizes of the schemas as written, not a per-turn cost. Where A4's measurement shows the toolkit adds a starting-context cost at all, that cost is what the savings on conceptual tasks have to beat.

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

## Track C: Session continuity (`/my-compact` + `/my-resume`)

This track tests a different claim than A and B: not "does the tool save tokens during a task" but "does resuming via Qdrant cost less than the alternatives across session boundaries." The `/my-compact` skill fills in a structured summary of the current conversation and hands it to `compact_store`, which compresses it locally (LM Studio) and stores the result in Qdrant in the same call, and `/my-resume` retrieves it at the start of the next session.

One structural advantage over Tracks A and B: if the `qdrant` server is already connected (which Track A requires), this workflow adds no tool schemas of its own -- the compact tools live on that same server, so there is no additional fixed cost.

### The two comparisons worth making

**Comparison C1: `/my-compact` + `/my-resume` vs. native `/compact`**

The native `/compact` command sends the conversation to Anthropic for summarization. `/my-compact` routes the same job through LM Studio locally. Both produce a summary that seeds the next session — but the token cost profile differs:

- Native `/compact`: one Anthropic API call (input = full conversation, output = summary) billed to your account
- `/my-compact`: one `compact_store` call whose compression runs on LM Studio (zero Anthropic tokens), then a later `compact_find` costs one tool round-trip; the summary is written once, not passed through your output a second time as a separate compression step would have required

Whether C1 shows a win depends on conversation length — short conversations may not generate enough summary-output tokens to offset the tool call overhead of `compact_store`.

**Comparison C2: `/my-compact` + `/my-resume` vs. starting fresh**

Without any compaction workflow, the alternative to a long session is abandoning context entirely and paying to re-establish it — re-reading key files, re-exploring the codebase, re-asking questions already answered. `/my-resume` amortizes that re-establishment cost.

This comparison is harder to measure precisely because "re-establishing context from scratch" isn't a repeatable, fixed-cost operation — it depends on the task and how much the developer remembers. A practical proxy: measure the token cost of the first 3-5 turns of a fresh session (with `/my-gh-load-context` as a stand-in for natural re-exploration) vs. the cost of `/my-resume` alone.

### Step C1: Build a benchmark scenario

Pick one realistic multi-session task — something you'd naturally pause mid-way and resume later (a feature implementation, a debugging investigation). Split it at a natural stopping point. This gives you a real "session 1 → compact → session 2" sequence to measure.

### Step C2: Comparison C1 run (native `/compact` vs. `/my-compact`)

1. Run session 1 of the task until the natural stopping point. Record `/usage`.
2. Run native `/compact`. Record the `/usage` delta (input + output tokens added by the compaction call).
3. Quit and relaunch. Paste the compact summary manually to seed session 2. Record the `/usage` cost of that seeding turn.
4. Repeat with `/my-compact` instead: record the `/usage` delta for the `compact_store` round-trip, then in session 2 the `compact_find` round-trip cost.
5. Compare: total cross-session overhead of native path vs. `/my-compact` path.

### Step C3: Comparison C2 run (`/my-resume` vs. starting fresh)

1. Use the same session 1 transcript. Run `/my-compact` at the end.
2. Quit and relaunch. Run `/my-resume` and record its `/usage` cost (just the `compact_find` round-trip).
3. Separately, quit and relaunch without resuming. Run the natural re-establishment sequence (load CLAUDE.md, re-read key files, re-ask the first substantive question). Record that total `/usage`.
4. Compare: does `/my-resume` undercut fresh re-establishment?

Repeat both comparisons 2-3x per "Measuring cleanly" above.

### Step C4: What this track can't tell you

- **Quality of the local summary vs. Anthropic's**: LM Studio's compression quality varies by model. A cheaper summary that loses key file paths or decisions is not a win — spot-check the restored context for correctness before counting a run as valid.
- **Multi-compact scenarios**: if you compact three times and resume the first, you pay `compact_find` retrieval overhead for a list + selection interaction. That's not captured in the single-compact measurement above. Worth a separate note if you use the workflow heavily.
- **Privacy benefit isn't measurable by `/usage`**: the primary motivation for `/my-compact` may be keeping conversation content off Anthropic's infrastructure entirely, not just cost. That's real value that won't show up in any token comparison — account for it qualitatively when reporting results.

## Track D: the live per-session savings tracker (`/my-savings`) — an estimate, NOT a fourth measured track

Unlike Tracks A–C, this isn't a protocol for you to run — it's a note about a *different kind of number* the repo now also produces, and why it must never be confused with the measured deltas above.

Tracks A–C all follow the same rigorous shape: run a task, run it again under the other condition, diff the two `/usage` readings. That's a controlled A/B measurement. The savings tracker (`libs/savings_ledger.py`, surfaced via `/my-savings` and a `SessionEnd` hook) can't do that — there's no baseline run inside a single live session. Instead it reports an **online observation of a counterfactual**, in exact characters:

> `characters before (the raw content that was compressed)` vs. `characters after (the compressed result actually used)`, plus the percent reduction between them

This is directly observable for local-compression events only (`compress_file`/`compress_command_output`/`fetch_url`'s server-side compression, plus the `compress_output.py` hook), because the tool holds both sides of the comparison in hand — it read the raw content and produced the compressed result itself, so nothing about that specific delta is guessed. That's exactly why Track A's Qdrant piece is NOT part of this tracker: the counterfactual for `qdrant-find` (what Grep+Read would have cost instead) isn't observable the same way, so the tracker deliberately never labels Qdrant activity as "savings" — see "Measuring cleanly" above and the README's own qdrant-find-vs-Grep discussion for why that gap can't be closed by a heuristic.

Other differences from Tracks A–C worth being explicit about:

- **Characters, not tokens.** The tracker used to show estimated tokens (chars÷3.5), but Claude's real tokenizer isn't public and that ratio measured about 29% low, so it now shows exact character counts and the percent reduction, which is identical in either unit. What the removed characters would have cost in tokens is not measured — unlike `/usage`'s real, billed figures, which remain the only token numbers to trust. The one token figure the tracker does show is the opt-in per-model one (next bullet), and it is derived, not measured.
- **A fixed tool-schema cost is not reported by the tracker, and is not netted out of the headline.** The tracker used to show a "Tool overhead .../turn" line, a chars÷3.5 estimate of the toolkit's tool-schema size. It was removed: real session transcripts list these tools as deferred, loaded on demand instead of sent every turn, so the line described a cost that mostly isn't paid. What the toolkit adds to a session's starting context is measured from transcripts (Step A4, `tools/measure_startup_context.py`), not estimated. An earlier draft of this design also considered `net = gross_saved − (schema_tokens × turns)`; that was always wrong in the other direction too, since a cached prefix is read at roughly a tenth of full price, so subtracting at the full rate would have overstated the tax.
- **Per-model figures: one exact, one derived.** With `CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS=1`, `SessionEnd` joins each compression event to its transcript turn by `tool_use_id`, so the tracker can show characters before and after **per model**. Those counts are **exact**: the tool made them, and the join only chooses the row. It can also show a token figure per model, which is **derived**: the characters removed divided by that session's own observed chars-per-token for that model (growth in real context tokens against the characters of uncompressed tool results, median over at least 5 samples), counted once for every request that carried the result until the session ended or compacted, so it is comparable with the real "tokens processed" total. It is labelled as derived, shown with its ratio and sample size, and omitted when there are too few samples. It is still **not a measurement**: the uncompressed result never entered context and was never counted, the ratio is an observed median that absorbs whatever else grew the context, and the per-request counting assumes the result stayed in context. Never report it as a saving the way a Track A–C delta is reported; if a defensible token figure is needed, run the A/B protocol.
- **`fetch_url` is logged but never credited.** Its honest counterfactual is WebFetch's own already-compressed summary (see Track B above), not the raw page — crediting it against raw bytes would overstate savings for a comparison this repo already established isn't apples-to-apples.
- **Explicitly NOT a substitute for Tracks A–C.** If you want a rigorous, defensible savings number for a report or a decision about whether to keep using this toolkit, run the actual A/B protocol above. `/my-savings` is for a quick, continuous, directional sense of "is this doing anything," not for a claim you'd want to defend under scrutiny.

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

Same as Step A4/B4: measure the starting-context cost from transcripts with `tools/measure_startup_context.py` rather than estimating it from schema text. `memory-bank`'s tools (`remember`/`recall`/`forget`/`transfer_memories`) are among the servers the survey checks for deferral. `python tools/report_tool_counts.py`'s server loop includes `memory-bank` (`memory_bank_mcp_server`) alongside `local-compress` and `codebase-indexer` and reports its live tool count and tool-description size in characters (a proxy for the JSON schema), through the same shared `libs/mcp_tool_introspect.py` utility -- run it rather than hand-typing a number here, since a hand-typed count has gone stale before.

### Step E5: remember:recall ratio (diagnostic only)

Same role as Track D's savings estimate — a cheap diagnostic number, not a pass/fail result. It no longer needs deriving by hand from session transcripts: the `memory-bank` server now records its own usage. The correct source for a call-frequency ratio is `/my-metrics memory-bank by_event_type` — a plain per-call-attempt tally in `metrics.db` (`libs/metrics_lib.py`), fired on every `recall`/`remember` invocation (including validation errors and empty results) whenever the shared, default-on `CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS` tracking flag is enabled. Treat it as a best-effort 1:1 call count, not a hard guarantee: that flag can be turned off, and the underlying `MetricsStore.record()` write deliberately fails open (an unwritable/locked/corrupt `metrics.db`, or a non-finite value, logs to stderr and returns `False` rather than raising or being retried), so a failed write silently leaves that one call uncounted. **`~/.claude/claude-runway/memory-events.db`'s `memory_events` table (`libs/memory_events_lib.py`, gated by the same flag) is NOT the right source for this ratio**: `recall()` logs one row per returned hit (zero rows for an empty or failed call, up to `limit` rows for one call with several hits), while `remember()` logs one row only after a successful write — so a naive `GROUP BY event_type` over that table counts point-level events, not tool invocations. `memory-events.db` is still the right source for the diagnostics this ratio doesn't cover — which specific memories get recalled, how often, how soon after creation (see [Memory bank](docs/memory-bank.md)'s "Usage event logging" section). A ratio heavily skewed toward `remember` suggests memories aren't being retrieved when relevant, but doesn't distinguish "nothing to recall yet" from "retrieval isn't working," so it shouldn't be reported as if it were Step E1-E3's result.

### Step E6: Why this track is harder to run than A-C

- **A correct treatment answer only counts as a win if `recall` is verifiably load-bearing in the transcript** — outcome correctness alone can't distinguish "recall saved the re-derivation" from "the model got there anyway on this particular run." This is why Step E3 classifies from the transcript rather than the final answer.
- **The re-derivation cost in baseline is inherently variable** (same caveat Track C2 raises about re-establishing context not being a fixed-cost operation) — a scenario that happens to be quick to re-derive understates the savings a harder scenario would show, so scenario selection matters more here than in Track A/B's more uniform lookups.
- **Small N is structural** — credible scenarios have to come from real, previously-researched findings, so this track won't reach Track A's 8-12-task sample size. Treat results as directional.
- **Reuse value compounds over a memory's lifetime in a way one baseline/treatment pairing can't capture** — a stored finding reused 5 times pays for itself 5 times over, but this protocol only measures a single reuse instance per scenario.

## Compute the result (all measured tracks)

- Per-task reduction = `(baseline_tokens − treatment_tokens) / baseline_tokens`
- Fixed overhead (Step A4 / B4 / E4, whichever applies to the track) is the measured starting-context difference from `tools/measure_startup_context.py`, reported alongside this reduction as its own separate figure, not subtracted from it into a single blended "net" number -- and where that measurement could not be made (A4 states which parts were not), say so rather than substituting an estimate
- Only count it as a real win if treatment correctness ≥ baseline correctness on the spot-checked tasks
- Report each measured track separately — nothing requires every piece to individually pay for itself at the same rate, since a project might only use a subset of them

## Caveats

- Prompt caching (cache reads vs. fresh input) changes what's actually billed — a raw token count comparison can overstate or understate real cost savings. Look at the cache read/write breakdown in `/usage`, not just total cost.
- This benchmark reflects one repo's structure and question mix — results won't automatically generalize to a very different codebase, question style, or local model.
- Local-compress results (Track B) also depend heavily on which local model is loaded in LM Studio — a result with one model doesn't necessarily generalize to another, especially given real testing in this repo already found meaningful quality differences between models on the same task (see `fetch_url`'s positional-focus handling in `libs/local_compress_lib.py`'s docstrings for a concrete example).
- Re-run this periodically as the repo grows; the case for semantic search gets stronger as a codebase gets too large to explore cheaply with Grep alone.
