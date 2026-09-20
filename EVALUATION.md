# Measuring Whether This Toolkit Actually Reduces Token Usage

The thesis behind this whole repo: **using Qdrant memory + local compression saves Claude token usage compared to not having them.** That's a testable claim, not something to assume — this doc is the plan for actually testing it, for both pieces this repo ships, without quietly hurting answer quality.

Two things need to be true, not just one, for either piece:

1. Tasks that benefit from the tool use fewer tokens with it than without.
2. That's not offset by the tool's own fixed cost — every connected MCP server's tool schemas ride along in context on **every single turn**, whether or not they're used. If most of your work doesn't actually benefit, that fixed tax could outweigh the wins on the tasks that do.

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

Separately, estimate the token cost of just having the two extra servers connected: token-count all tool descriptions/schemas across BOTH servers, since that's paid every turn regardless of use. This has **two independently-drifting halves**, so don't trust a number written into this doc's prose for either one -- both #22 and #23 already caught this exact class of drift once (a hardcoded tuple in `compress_mcp_server.py`, and this very paragraph's own hardcoded count), and by the time issue #52 built the fix below, `codebase-indexer`'s count had already drifted a THIRD time past #23's correction (`set_collection_description` shipped after that fix landed and was never added here) -- proof that re-typing the right number by hand doesn't hold, only a live count does:

- **`codebase-indexer`'s half** (`index_repo`, `sync_repo`, `preview_index`, `get_collection_info`, `find_in_collection`, `set_collection_description`, `list_collections`, and any tool added since): run `python tools/report_tool_counts.py` for the current live count AND an estimated schema-token cost, or read `ingest_mcp_server.py`'s own stderr startup log (`[claude-runway] codebase-indexer registered N tools`) for the count alone the next time it starts.
- **The standalone `qdrant` server's half** (`qdrant-find`, `qdrant-store` -- currently 2): the same `tools/report_tool_counts.py` command also introspects this one, by constructing the third-party `mcp_server_qdrant` package's own server directly (no live Qdrant/embedding model needed). This matters because `mcp-server-qdrant` is **unpinned** in `requirements.txt` -- unlike `codebase-indexer`'s count, this one isn't fully under this repo's control, so a future upstream release could add a tool without this repo's code changing at all. Introspecting the installed package directly, rather than assuming "always exactly 2," is what actually closes that gap (PR #120 review finding).

Both halves go through the same shared, self-updating utility (`libs/mcp_tool_introspect.py`, issue #52/GROW-05), so none of these three numbers can drift relative to each other. This is the number that has to be beaten by the savings on conceptual tasks.

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

Token-count all `local-compress` tool schemas — paid every turn regardless of use, same as Track A's servers. **Don't trust a hardcoded number in this paragraph** -- a hardcoded 5 here went stale once already (issue #22's root cause) and a hardcoded doc count went stale a second time the same way (issue #23) before both got fixed for good (issue #52/GROW-05): `_log_fixed_overhead()` and this doc now go through the same shared, self-updating utility (`libs/mcp_tool_introspect.py`), so neither can drift relative to the other again. Two ways to get the current live count, depending on whether the savings tracker is on:

- **`CLAUDE_RUNWAY_TRACK_SAVINGS` on**: the server computes this figure itself at every startup and stores it as `schema_overhead_tokens`, surfaced via the plain `/my-savings` simple view's "Tool overhead" line (`format_detail_view()` doesn't currently repeat this annotation -- `format_simple_view()` is the one that emits it; see that function's own definition in `libs/savings_ledger.py`, not a line number pinned here, since line numbers shift and go stale silently) -- read that number instead of re-deriving it here.
- **Either way, or if the tracker is off**: run `python tools/report_tool_counts.py`, which imports the server module directly and reports both its live tool count AND an estimated schema-token cost (the same `schema_overhead_tokens` figure, computed the same way) without needing `CLAUDE_RUNWAY_TRACK_SAVINGS` on at all -- a tool count alone isn't enough here, since Step B4/the "Compute the result" section below need an actual token number, not just "how many tools."

Hooks (`compress_bash_output.py`, `redirect_webfetch_to_fetch_url.py`) do NOT add this kind of fixed cost — they're not tools Claude sees in its schema list, they intercept transparently, so they should be near-zero fixed overhead by design. Worth spot-checking this assumption once rather than just asserting it.

### Step B5: What this track can't tell you

Whether a built-in tool's own internal processing (e.g. WebFetch's server-side extraction) is itself billed to you is a separate, narrower question from "which approach costs less for an equivalent task" — Track B answers the second question directly by comparing total session cost, without needing to know the answer to the first. Testing attempted the narrower question directly and it turned out to likely be unanswerable from `/usage` alone (see "Measuring cleanly" above) — don't get stuck trying to resolve it; it isn't load-bearing for this repo's actual thesis.

## Track C: Session continuity (`/my-compact` + `/my-resume`)

This track tests a different claim than A and B: not "does the tool save tokens during a task" but "does resuming via Qdrant cost less than the alternatives across session boundaries." The `/my-compact` skill summarizes the current conversation locally (LM Studio via `compress_text`), stores the result in Qdrant, and `/my-resume` retrieves it at the start of the next session.

One structural advantage over Tracks A and B: if the Qdrant and local-compress servers are already connected (which Tracks A and B require), the tool schema overhead for this workflow is already paid — zero additional fixed cost per turn.

### The two comparisons worth making

**Comparison C1: `/my-compact` + `/my-resume` vs. native `/compact`**

The native `/compact` command sends the conversation to Anthropic for summarization. `/my-compact` routes the same job through LM Studio locally. Both produce a summary that seeds the next session — but the token cost profile differs:

- Native `/compact`: one Anthropic API call (input = full conversation, output = summary) billed to your account
- `/my-compact`: `compress_text` runs on LM Studio (zero Anthropic tokens), then `compact_store` + later `compact_find` each cost one tool round-trip

Whether C1 shows a win depends on conversation length — short conversations may not generate enough summary-output tokens to offset the tool call overhead of `compress_text` + `compact_store`.

**Comparison C2: `/my-compact` + `/my-resume` vs. starting fresh**

Without any compaction workflow, the alternative to a long session is abandoning context entirely and paying to re-establish it — re-reading key files, re-exploring the codebase, re-asking questions already answered. `/my-resume` amortizes that re-establishment cost.

This comparison is harder to measure precisely because "re-establishing context from scratch" isn't a repeatable, fixed-cost operation — it depends on the task and how much the developer remembers. A practical proxy: measure the token cost of the first 3-5 turns of a fresh session (with `/my-load-context` as a stand-in for natural re-exploration) vs. the cost of `/my-resume` alone.

### Step C1: Build a benchmark scenario

Pick one realistic multi-session task — something you'd naturally pause mid-way and resume later (a feature implementation, a debugging investigation). Split it at a natural stopping point. This gives you a real "session 1 → compact → session 2" sequence to measure.

### Step C2: Comparison C1 run (native `/compact` vs. `/my-compact`)

1. Run session 1 of the task until the natural stopping point. Record `/usage`.
2. Run native `/compact`. Record the `/usage` delta (input + output tokens added by the compaction call).
3. Quit and relaunch. Paste the compact summary manually to seed session 2. Record the `/usage` cost of that seeding turn.
4. Repeat with `/my-compact` instead: record the `/usage` delta for the `compress_text` + `compact_store` round-trips, then in session 2 the `compact_find` round-trip cost.
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

Tracks A–C all follow the same rigorous shape: run a task, run it again under the other condition, diff the two `/usage` readings. That's a controlled A/B measurement. The savings tracker (`libs/savings_ledger.py`, surfaced via `/my-savings` and a `SessionEnd` hook) can't do that — there's no baseline run inside a single live session. Instead it computes an **online estimate of a counterfactual**:

> `saved ≈ tokens(raw content that was compressed) − tokens(the compressed result actually used)`

This is directly observable for local-compression events only (`compress_file`/`compress_command_output`/`fetch_url`'s server-side compression, plus the `compress_bash_output.py` hook), because the tool holds both sides of the comparison in hand — it read the raw content and produced the compressed result itself, so nothing about that specific delta is guessed. That's exactly why Track A's Qdrant piece is NOT part of this tracker: the counterfactual for `qdrant-find` (what Grep+Read would have cost instead) isn't observable the same way, so the tracker deliberately never labels Qdrant activity as "savings" — see "Measuring cleanly" above and the README's own qdrant-find-vs-Grep discussion for why that gap can't be closed by a heuristic.

Other differences from Tracks A–C worth being explicit about:

- **Token counts are a chars÷3.5 approximation**, not Claude's real tokenizer (which isn't public). The same crude ratio is applied to both sides of every comparison, so its error mostly cancels out of the resulting percentage, but absolute numbers should be read as directional, not exact — unlike `/usage`'s real, billed figures.
- **The fixed per-turn schema overhead these tools add is reported separately, never netted out of the headline.** An earlier draft of this design considered `net = gross_saved − (schema_tokens × turns)`, but tool schemas sit in the request's cached prefix — only the first turn pays full price, every turn after is a cache read at roughly a tenth the cost. Subtracting at the full rate would overstate the real tax by nearly 10x on any session past its first turn, manufacturing a "cost" that mostly isn't actually paid. The tracker reports both numbers side by side instead and lets the reader judge.
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

Same pattern as Step A4/B4 — and already wired up rather than something this track still needs to add: `tools/report_tool_counts.py`'s server loop already includes `memory-bank` (`memory_bank_mcp_server`) alongside `local-compress` and `codebase-indexer`, so its 3 tools' (`remember`/`recall`/`forget`) schema-token cost goes through the same shared `libs/mcp_tool_introspect.py` utility as the other two servers. Run `python tools/report_tool_counts.py` for the current live number rather than hand-typing one here — same discipline Step A4/B4 already established, for the same reason: issues #22/#23 are exactly what happens when a schema-token count gets hand-typed into this doc instead of read live.

### Step E5: remember:recall ratio (diagnostic only)

Same role as Track D's savings estimate — a cheap diagnostic number, not a pass/fail result. Define it as the ratio of `remember` rows to `recall` rows in `~/.claude/claude-runway/memory-events.db`, which the `memory-bank` server writes automatically (on by default — see [Memory bank](docs/memory-bank.md)), so it can be read off the database instead of counted by hand from transcripts. Two limits to state alongside any number: `recall` logs one row per *hit* returned, not one per call, so a single call that returns five memories adds five rows (the `turn` column groups rows that came from the same call, if you need per-call counts); and calls that record nothing — a `recall` with no matches, a failed call, or an embedding-model mismatch — leave no row at all, so "retrieval isn't finding anything" is undercounted rather than shown. If `CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS` is turned off there are no rows, and the ratio has to be derived from session transcripts instead (same approach as Step A5). A ratio heavily skewed toward `remember` suggests memories aren't being retrieved when relevant, but doesn't distinguish "nothing to recall yet" from "retrieval isn't working," so it shouldn't be reported as if it were Step E1-E3's result.

### Step E6: Why this track is harder to run than A-C

- **A correct treatment answer only counts as a win if `recall` is verifiably load-bearing in the transcript** — outcome correctness alone can't distinguish "recall saved the re-derivation" from "the model got there anyway on this particular run." This is why Step E3 classifies from the transcript rather than the final answer.
- **The re-derivation cost in baseline is inherently variable** (same caveat Track C2 raises about re-establishing context not being a fixed-cost operation) — a scenario that happens to be quick to re-derive understates the savings a harder scenario would show, so scenario selection matters more here than in Track A/B's more uniform lookups.
- **Small N is structural** — credible scenarios have to come from real, previously-researched findings, so this track won't reach Track A's 8-12-task sample size. Treat results as directional.
- **Reuse value compounds over a memory's lifetime in a way one baseline/treatment pairing can't capture** — a stored finding reused 5 times pays for itself 5 times over, but this protocol only measures a single reuse instance per scenario.

## Compute the result (all measured tracks)

- Per-task reduction = `(baseline_tokens − treatment_tokens) / baseline_tokens`
- Fixed overhead (Step A4 / B4 / E4, whichever applies to the track) is reported alongside this reduction as its own separate figure, not subtracted from it into a single blended "net" number — tool schemas sit in the request's cached prefix, so only the first turn pays full price for that overhead, same rationale as Track D's discussion above
- Only count it as a real win if treatment correctness ≥ baseline correctness on the spot-checked tasks
- Report each measured track separately — nothing requires every piece to individually pay for itself at the same rate, since a project might only use a subset of them

## Caveats

- Prompt caching (cache reads vs. fresh input) changes what's actually billed — a raw token count comparison can overstate or understate real cost savings. Look at the cache read/write breakdown in `/usage`, not just total cost.
- This benchmark reflects one repo's structure and question mix — results won't automatically generalize to a very different codebase, question style, or local model.
- Local-compress results (Track B) also depend heavily on which local model is loaded in LM Studio — a result with one model doesn't necessarily generalize to another, especially given real testing in this repo already found meaningful quality differences between models on the same task (see `fetch_url`'s positional-focus handling in `libs/local_compress_lib.py`'s docstrings for a concrete example).
- Re-run this periodically as the repo grows; the case for semantic search gets stronger as a codebase gets too large to explore cheaply with Grep alone.
