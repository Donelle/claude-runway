# Shared metrics store

A generic, cross-tool metrics-recording primitive: ONE shared SQLite database any domain in this toolkit can write append-only events into, behind ONE small class and ONE MCP tool — instead of each new metric domain (memory-bank's `recall`/`remember` events, `my-gh-autowork`'s per-ticket run metrics, whatever comes next) inventing its own schema and its own dedicated MCP tool.

**Why one shared tool, not one per domain:** this repo ships a live tool count across its connected MCP servers (`python tools/report_tool_counts.py`; see `EVALUATION.md`'s Track A4/B4/E4) rather than a number written into prose, which goes stale as tools are added. An earlier version of this paragraph said those ride along in context on every turn; session transcripts measured on the author's machine show Claude Code listing them as deferred (loaded on demand), so what they cost is a schema loaded when a session first needs one, not a per-turn charge (see Step A4 and `tools/measure_startup_context.py`). A metrics-reporting design that adds a new MCP tool per future metric domain would still add a schema to load and one more tool to choose among each time. A single generic dispatch tool (`get_metrics`) adds a flat, bounded cost no matter how many domains eventually write into this store.

**Status:** two domains write into this store today:

- `metric_id="autowork"` — `my-gh-autowork`'s per-run outcome logging. Use `/my-metrics autowork`; a fresh install has no data until at least one run completes.
- `metric_id="memory-bank"` — a plain per-call-attempt counter for `recall`/`remember`/`forget` (event types of the same names), written by `libs/memory_events_lib.py`'s `record_memory_metric()`. Use `/my-metrics memory-bank`. It shares the `CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS` opt-out with memory-bank's detailed per-point `memory-events.db` log (see [Memory bank](memory-bank.md)) — there's no separate switch — so a session with tracking disabled writes no rows here either.

That memory-bank counter is in addition to, not a replacement for, the detailed `memory-events.db` log. Moving that log onto this store was considered and deliberately declined: its indexed `point_id`/`repo` columns support rollups ("most-recalled memories") that this generic, schema-free store can't do without an unindexed `json_extract` scan, so the two stores stay side by side on purpose. Migrating `libs/savings_ledger.py`'s `savings.db` onto this store remains an open, undecided follow-up.

## Where the data lives

A single, central SQLite database — `~/.claude/claude-runway/metrics.db` by default, same location convention as `savings.db`/`memory-events.db`. Override with `CLAUDE_RUNWAY_METRICS_DB` (an absolute path; a relative override is anchored to the home directory, not the caller's cwd).

## Schema

One append-only `metrics` table:

| Column            | Type    | Notes                                                        |
| ----------------- | ------- | ------------------------------------------------------------- |
| `id`               | INTEGER | Autoincrement primary key.                                    |
| `metric_id`        | TEXT    | The domain (e.g. `"autowork"`, `"memory-bank"`) — caller-defined, no fixed enum. |
| `event_type`       | TEXT    | Domain-defined event kind (e.g. `"ticket_merged"`, `"recall"`). |
| `value`            | REAL    | A signed numeric delta — see "Never a mutable counter" below. |
| `metadata`         | TEXT    | Optional JSON-encoded dict of domain-specific extra fields.    |
| `event_timestamp`  | TEXT    | ISO 8601 UTC, e.g. `"2026-09-24T12:00:00Z"`.                   |
| `session_id`       | TEXT    | Optional caller-supplied session identifier.                   |

Generic enough that a brand-new `metric_id` never requires a schema migration — only a genuine change to the table's own shape would.

**Never a mutable in-place counter.** "Current value" for any `metric_id`/`event_type` pair is always `SUM(value)` computed at read time, the same append-only-log-then-aggregate design `savings_ledger.py`/`memory_events_lib.py` already use — deliberate, since multiple separate stdio MCP server processes (`local-compress`, `codebase-indexer`, `memory-bank`) could in principle write concurrently, and a mutable counter's read-modify-write cycle isn't safe against that without locking a log-then-aggregate design doesn't need.

## The `MetricsStore` class (`libs/metrics_lib.py`)

```python
from metrics_lib import MetricsStore

store = MetricsStore()
store.record("autowork", "ticket_merged", value=1.0, metadata={"issue": 208}, session_id="sess-1")
store.increment("autowork", "review_round")   # sugar over record(value=+by)
store.decrement("autowork", "open_tickets")   # sugar over record(value=-by)

store.summary("autowork")          # {"metric_id", "event_count", "total_value", "first_event_at", "last_event_at"}
store.by_event_type("autowork")    # [{"event_type", "event_count", "total_value"}, ...] sorted by total_value desc
store.trend("autowork", bucket="week", n=12)  # [{"bucket", "event_count", "total_value"}, ...] oldest first
store.detail("autowork", limit=12) # [{"event_type", "value", "metadata", "event_timestamp", "session_id"}, ...] most recent first

store.summary("autowork", session_id="issue-248-143022")        # same shape, narrowed to one session
store.by_event_type("autowork", session_id="issue-248-143022")  # same shape, narrowed to one session
store.detail("autowork", session_id="issue-248-143022")         # same shape, narrowed to one session
```

`summary()`/`by_event_type()`/`detail()` also take an optional `session_id` to narrow the result down to one specific session's rows (e.g. one `my-gh-autowork` attempt's own `issue-<n>-<HHMMSS>` marker) instead of the metric_id's entire history. Omitting it keeps the metric_id-only result unchanged. Not available on `trend()` — per-session filtering doesn't compose with time-bucketed grouping (deliberately out of scope). The text formatters (`format_summary_view`/`format_by_event_type_view`/`format_detail_view`) are session-aware too: a `session_id`-filtered query with zero matching rows reports "No events recorded yet for this metric_id with session_id=...", not the unqualified "no events for this metric_id" message — that message would be misleading when other sessions under the same metric_id do have data.

`detail()` is the raw, per-event counterpart to `summary()`/`by_event_type()`'s aggregates: it returns individual rows, most-recent first, with each row's `metadata` deserialized back into a dict — the richer per-run detail (e.g. autowork's `model`/`rounds`/`wall_clock_s`/`subagent_tokens`/`tool_calls`/`findings`) that the two aggregate views never surface, since those only ever sum `value`. Deliberately generic: it doesn't hardcode any domain's metadata field names, so any future `metric_id` that stores structured metadata gets this view for free. `limit` caps the number of rows returned (default 12, same default as `trend()`'s bucket count) via SQL `LIMIT`, not a Python-side slice.

`record()` also takes an optional `event_timestamp` override — full ISO 8601 UTC (`"YYYY-MM-DDTHH:MM:SSZ"`), used verbatim instead of the current time. Ordinary callers should never pass this; it exists for backfilling historical events with their own original dates — an invalid format fails open (logged to stderr, nothing written), same as the non-finite `value` guard.

Writes (`record`/`increment`/`decrement`) **fail open** — a broken/locked/corrupt metrics database is logged to stderr and swallowed, never raised, since the actual domain operation an event describes has already happened by the time it's logged. Reads (`summary`/`by_event_type`/`trend`) do **not** fail open — they raise, and it's the caller's job to decide whether to surface a friendlier error (the `get_metrics` MCP tool below does exactly that, matching `savings_summary`/`savings_detail`/`savings_trend`'s own try/except pattern).

## MCP tools

Both live in the opt-in `analytics` server (`mcpservers/analytics_mcp_server.py`), beside the savings tracker's `savings_summary`/`savings_detail`/`savings_trend`. That server needs only the `mcp` package and these two SQLite stores, and it is separate from `local-compress` so a project that never looks at metrics or savings does not carry these five tools' schemas. `setup_project.py init --track-savings` writes its `.mcp.json` block (`--metrics-db` carries a non-default database path into it, and into the `memory-bank` block too, which writes into the same store); `init` and an interactive `upgrade` ask whether to enable the tracker, and `upgrade` also offers the block to a project that already tracks savings. Without the block the two tools do not exist, and calling one is an unknown-tool error, not "no events recorded yet". Writers do not need the server: `memory-bank` records through `libs/memory_events_lib.py` directly and `tools/autowork_cost.py record` writes `autowork` rows itself.

**Which `metric_id`s have a writer today:** `autowork` (the per-ticket outcome rows, including the `ticket_cost` rows `tools/autowork_cost.py record` writes) and `memory-bank` (the call tally above). The savings tracker is a separate store (`savings.db`), read with `/my-savings`; asking `get_metrics` about it reports nothing.

### `record_metric` (write)

The write counterpart to `get_metrics`.

```
record_metric(metric_id, event_type, value=1.0, metadata=None, session_id=None)
```

- `metric_id`: the domain (e.g. `"autowork"`) — caller-defined free-text string.
- `event_type`: domain-defined event kind (e.g. `"ticket_merged"`, `"ticket_blocked"`).
- `value`: a signed numeric delta (default `1.0`). Must be finite.
- `metadata`: optional domain-specific extra fields, as either a JSON string or an already-parsed dict — both shapes are accepted directly (a client that coerces a JSON-shaped string argument into a native object before the tool sees it, as Claude Code itself does, still succeeds).
- `session_id`: optional caller-supplied identifier stored in the raw row — `get_metrics`/`/my-metrics` can filter `view="summary"`/`"by_event_type"` down to one exact `session_id`.

Returns `"OK"` on success or an `"Error:..."` string on any failure — including DB write failures (when `MetricsStore.record()` returns `False`). DB write errors are also logged to stderr but not raised, preserving fail-open behavior; the `"Error:"` return is the only surface the caller sees.

### `get_metrics` (read)

```
get_metrics(metric_id, view="summary", bucket="week", n=12, format="text", session_id=None)
```

- `view`: `"summary"` (default, total event count + total value), `"by_event_type"` (per-event_type breakdown), `"trend"` (day/week-bucketed history), or `"detail"` (raw per-event rows with deserialized `metadata`, most recent first).
- `bucket`: only applies to `view="trend"`.
- `n`: "max buckets" for `view="trend"`, or "max rows" for `view="detail"` — ignored for `"summary"`/`"by_event_type"`. Reused across both rather than a second row-count parameter, since only one meaning ever applies for a given call.
- `session_id`: applies to `view="summary"`/`"by_event_type"`/`"detail"` — narrows the result to one specific session (e.g. one `my-gh-autowork` attempt's `issue-<n>-<HHMMSS>` marker). Ignored for `view="trend"` (not supported there — see the `MetricsStore` section above).
- `format`: `"text"` (default, human-readable) or `"json"`.

## Using it

**Install the skill first**: `claude-runway-setup init --install-skills` (or `python tools/setup_project.py init --install-skills` from a clone) installs/updates `my-metrics` along with every other product skill — equivalent manual copy: `mkdir -p ~/.claude/skills/my-metrics && cp skills/my-metrics/SKILL.md ~/.claude/skills/my-metrics/SKILL.md`. Product skills are only discovered once installed under `~/.claude/skills/` — the rest of the setup flow (`.mcp.json`/`.claude/settings.json`) doesn't install them for you (same step `docs/savings-tracker.md`'s "Enabling it" section documents for `/my-savings`).

`/my-metrics <metricId> [view] [bucket|sessionId]` — thin wrapper matching `/my-savings`'s shape, e.g.:

- `/my-metrics autowork` — summary view (total tickets worked, all outcomes).
- `/my-metrics autowork by_event_type` — breakdown by outcome type (`ticket_merged`, `ticket_blocked`, `ticket_failed`).
- `/my-metrics autowork trend day` — daily trend.
- `/my-metrics autowork detail` — the last 12 autowork runs, each with its full `metadata` (model, rounds, wall_clock_s, subagent_tokens, tool_calls, findings) shown.
- `/my-metrics autowork summary issue-<n>-143022` — summary narrowed to one specific autowork attempt's `session_id`. Only supported for `summary`/`by_event_type`/`detail`, not `trend`.

Live data appears immediately with no migration needed.

## Autowork cost from transcripts (`tools/autowork_cost.py`)

`/my-gh-autowork`'s per-ticket cost is computed from the subagent's own transcript (`<session>/subagents/agent-<id>.jsonl`) instead of being copied by hand from the Agent call's `usage` block, which nothing enforced. Stdlib only; `libs/autowork_cost_lib.py` holds the logic.

- **`record --issue N --outcome merged|blocked|failed --agent-id ID`** (or `--tool-use-id`) writes the usual `ticket_*` row with `wall_clock_s`, `subagent_tokens`, `tool_calls` and `model` filled in, plus a `cost` object: every counter (input, cache creation, cache reads, output, requests) per model, the cache-read share, and the agent id. A transcript that can't be found still records the row, marked `cost.available=false`, never as a zero-cost run.
- **Linking rule.** One subagent file is one attempt. A re-attempt or a follow-up subagent has its own agent id and its own row. An attempt is identified exactly by agent id or by the meta file's `toolUseId`. Its issue number (needed only to match old rows, and only taken when its prompt names `Donelle/claude-runway`, since discovery scans every Claude project and issue numbers are per repository) comes from the bare number the skill's template substitutes on the line after its `## Step 0` heading, else a stray `The issue to work is #N` sentence, else `#N`/`issue N` in the description; auto-pick attempts have none of these and link by id only. `record` refuses an attempt whose own issue differs from `--issue` (it records the row marked unavailable, with the reason), so an id from another ticket can't file its usage here.
- **Counters.** Usage is taken once per `message.id` (each content block repeats it), field-wise max. `output_tokens` is meant to include thinking and the API gives no separate thinking count, but the value stored in a transcript is a streaming-start snapshot, so it is a floor, not the response's real output (see [What a subagent starts with](#what-a-subagent-starts-with-and-why-its-context-grows-toolscontext_breakdownpy)). `subagent_tokens` is input + cache creation + output, **excluding cache reads**, because that is the scale the existing rows use (see `compare`); cache reads are in `cost` and the report.
- **`backfill [--apply]`** adds an append-only `ticket_cost` event (value 0, same `session_id` and timestamp as its row) for each row with no figures, from the transcript that matches its issue and finished nearest before the row was logged; rows whose transcripts are gone get `cost.available=false`. Rows that already have figures are never touched. Dry run unless `--apply`.
- **`compare`** puts the existing hand-copied figures beside the transcript totals. On the 2026-10-13 data (43 rows matched), tool-call counts agreed on 42 of 43, and the old token figure was a median 101% of input + cache creation + output (range 23% to 138%) but only a median 3.6% of all four counters: the old figures leave out cache reads, which are the bulk of the volume. Trust them as "fresh tokens", not as total volume.
- **`report [--format json]`** shows per-ticket and per-model totals and the cache-read share. Rows with only the old single total are listed as legacy and left out of the per-model table. Each ticket line also shows three cache figures: cache-creation tokens, the longest gap between requests, and the number of cold-cache rebuilds (below). The gap and rebuild count are computed from the transcript when the row is recorded (the report reads rows, not transcripts), so rows recorded before they existed show `n/a`; `backfill` fills them in for rows it recovers.
- **Longest gap.** The longest time between the *start* of one request and the start of the next, over one subagent's requests in transcript order. A request streamed as several transcript lines counts once, at its earliest timestamp.
- **Cold-cache rebuild.** A request that writes a large cache block and reads nothing, after a gap of more than 300 seconds since the previous request started. "Large" is at least 10,000 cache-creation tokens (`REBUILD_MIN_CREATION`; a subagent's context starts near 56,000 and an ordinary turn adds a few hundred), and the gap threshold is `REBUILD_GAP_S`. The first request of a run is never a rebuild: it has no previous request and always writes. The idea behind it is that the prompt cache appears to lapse after about five minutes idle, so a long single wait makes the next request re-write the whole context at the dearer cache-write rate. That five-minute figure is an inference from one run, so a rebuild that still occurs with every gap under about 240 seconds means the inference is wrong. Compare cache-creation tokens and rebuild count per ticket, not total tokens, because ticket size varies a lot.
- Subagent cost only: the orchestrator's own session (the main transcript) is not included.

## What a subagent starts with, and why its context grows (`tools/context_breakdown.py`)

An autowork subagent's first request is about 56,000 tokens, and its context at the end is two to four times that. `tools/context_breakdown.py` (logic in `libs/context_breakdown_lib.py`) itemizes both from the subagent transcripts, reading them only and printing counts, sizes and token figures, never text. Measured 2026-10-13 on a general-purpose subagent in a worktree, Sonnet 5.5, with this repository's dogfood configuration. The figures are tied to that machine's skills, plugins, MCP servers and CLAUDE.md files; re-run the tool after any of those change.

### What the transcript records that the earlier estimates missed

- Besides the API's token counts, a transcript stores the attachments Claude Code injects: the skill list, the CLAUDE.md files, the agent list, MCP server instructions, deferred tool names, and a `prompt_snapshot` whose `tools` list is the full tool definitions. Sizes in characters are therefore known exactly; only the characters-to-tokens ratio has to be measured.
- `usage.output_tokens` in a transcript is a streaming-start snapshot, not the final count: a 109-request run sums to about 1,600 although every response carries a tool call. Output cost is under-reported by `tools/autowork_cost.py`; treat its output column as a floor.
- Extended-thinking blocks are stored with empty text and a long `signature`. The thinking is in the context (and is paid for), but its text is not in the transcript; the signature length is the only recorded proxy.
- `hook_success` attachments (the hook's own stdout) are not part of the context; a hook result only enters it as `hook_additional_context`.

### Request 0 in tokens (56,254 for a full-size ticket)

The no-op baseline is the same subagent type given a one-line prompt: 40,176 tokens (40,106 without worktree isolation, so isolation costs about 70). The difference to a real run is the prompt template alone: 16,078 tokens for 43,712 characters, exact by subtraction.

| Item | Characters | Tokens | How measured |
| --- | ---: | ---: | --- |
| Prompt template (the user turn) | 43,712 | 16,173 | 0.37 tokens per character; the exact difference to the no-op baseline (16,078) confirms it within 1 percent |
| Tool definitions, 9 tools | 80,723 | 19,461 | Remainder (see below); of which the Artifact tool is about 13,350 and the Agent tool about 1,940 by share of recorded characters |
| Skill list (167 skills, capped at 30,000 characters) | 30,000 | 10,980 | Probe: the same text read into a no-op subagent |
| CLAUDE.md files (global, project, memory index) | 19,778 | 6,428 | Probe, same method |
| System prompt | 2,714 | 1,004 | 0.37 tokens per character (not probed) |
| Deferred tool names | 2,366 | 875 | Same rate |
| Environment, session context, model, other attachments, reminder | about 3,600 | about 1,330 | Same rate |
| **Total** | | **56,254** | |

Two caveats on that table:

- **The tool-definition line is a remainder, not a measurement.** The API does not tokenize tool definitions as the JSON text in the transcript: the same JSON read through a user turn cost about 1.4 times what the remainder allows. So that line absorbs the estimation error of every other line. A controlled difference bounds it from the other side: a read-only subagent type (no Agent, Artifact, Edit or Write tool, and no CLAUDE.md) starts 20,381 tokens lower; subtracting the measured CLAUDE.md share leaves about 13,900 for those four tools, against 15,900 when the remainder is split by characters. Treat the tool-definition line as 17,500 to 19,500 tokens, and the other lines as correct to roughly 10 percent.
- The skill list and CLAUDE.md probes include the Read tool's line-number prefixes, which add roughly 0.5k tokens to the skill list.

### What arrives after request 0

Two more fixed items enter the context at request 1 and are then carried by every later request: the agent list (about 15,800 characters, 5,570 tokens, exact by difference to a subagent type that has no Agent tool and gets none) and the MCP servers' instructions (about 4,100 characters, about 1,500 tokens). The fixed overhead of a subagent is therefore about 63,300 tokens, not 56,000. Across the 65 Sonnet autowork transcripts with at least five requests, that fixed part is a median 61 percent of a ticket's token volume (pooled 36.5 percent, because the large tickets dominate the pool); for the two tickets first measured it was 34.5 percent (109 requests) and 47.9 percent (50 requests). The prompt template alone is a median 11.4 percent.

### Where the growth goes

The context after request 0 grows by what the previous response added (tool inputs, thinking, text) plus whatever arrived on the user side (tool results, attachments). `calibrate` fits tokens per character for each class by least squares over every request of every Sonnet subagent transcript (about 7,000 requests, 287 tokens root-mean-square error per request); `growth` applies it to one transcript. Holding a transcript out of the fit and predicting its growth missed by at most 2.6 percent on five transcripts, so the split below is not an artefact of fitting the same data.

Measured ratios (tokens per character, characters per token in brackets): Read results 0.397 (2.5), Bash results 0.415 (2.4), tool-call inputs 0.376 (2.7), thinking signature 0.261 (3.8), agent list 0.401 (2.5), the schema of a tool loaded on demand 0.294 (3.4); plus a constant 112 tokens per request that no recorded text explains (the per-call wrapper). The earlier estimate of 4 characters per token was therefore about 55 percent too generous for code and tool output.

For the 109-request ticket (56,254 to 238,565 tokens, growth 182,311):

| Source of growth | Tokens |
| --- | ---: |
| Read results (130,760 characters, 18 reads) | 51,877 |
| Tool-call inputs the model wrote (136,504 characters, 162 calls) | 51,384 |
| Bash results (70,194 characters, 82 calls) | 29,107 |
| Thinking (stored as signatures only; the text is not in the transcript) | 25,384 |
| Per-request constant (112 tokens times 108 requests) | 12,087 |
| Edit and Write results | 8,868 |
| Agent list and MCP instructions arriving at request 1 | 6,439 |
| Assistant text and edited-file notices | 404 |
| **Explained** | **185,551** |
| Remainder | -3,240 |

The roughly 120,000 tokens that looked unexplained are explained. Three things were missing from the earlier count: the characters-to-tokens ratio (about 2.5, not 4, which alone moves the visible text from about 98,000 tokens to about 150,000), the thinking that the transcript stores only as signatures (about 25,400 tokens, 14 percent of the growth), and the per-request wrapper plus the attachments that arrive after request 0 (about 18,500 tokens). No single request grows by much more than its recorded content predicts. The 50-request ticket reconciles the same way (growth 105,749, remainder 1,523).

### Big reads

Over the 66 autowork transcripts on disk (200 whole-file reads, 568 ranged reads), 74 whole-file reads returned at least 12,000 characters. The tokens they were carried for (result size times the requests after the read) are 10.1 percent of those transcripts' token volume; the README's mandatory full read alone is 5.9 percent, and it was the largest single Read result in 36 of the 41 tickets that read it. About 17,800 of the README's 33,300 characters are its Files table, which is placed last on purpose: reading only up to that heading would cut about 3.1 percent of volume. Among the six most recent transcripts there are 10 big whole-file reads: five are the README and two are the same large library file, which a ranged read would have served.

### Candidate reductions, ranked

Per-ticket figures are token volume (input, cache creation and cache reads counted alike) for the median 44-request ticket unless stated. Cache reads cost far less per token than fresh input, so volume overstates the money saved, but the ordering holds.

| Reduction | Saving per request | Per median ticket | Risk |
| --- | ---: | ---: | --- |
| A subagent type without the Agent and Artifact tools (autowork uses neither) | about 19,000 (13,300 to 15,300 tools, 5,570 agent list) | about 836,000 (11.8 percent of a mean ticket) | Low: the mechanism is a tool allow-list, shown to work by the read-only built-in types; the autowork-specific definition itself is not measured yet |
| Read the README only up to its Files table | about 7,100, once read | about 500,000 per ticket that reads it (3.1 percent pooled) | Low to medium: Step 1 says "in full" for a reason |
| Remove the duplicated CLAUDE.md block (the project file repeats the global one line for line, 7,714 characters) | about 2,500 | about 110,000 | Low; the fix is local configuration, so the repo can only detect it |
| Drop skills and plugin MCP instructions the subagent never uses | up to 13,400 (skill list 10,980, MCP instructions 1,500, deferred names 875) | up to about 590,000 | Unmeasured: needs configuration outside this repository, so the number is an upper bound |
| Trim the prompt template | 370 per 1,000 characters removed | about 16,000 per 1,000 characters | High: each line exists because skipping it caused a real failure |
| Ranged reads of large non-README files | not separable | about 4 percent of volume is whole-file reads of other large files; a half saving is about 2 percent | Medium |

### Commands

- `python tools/context_breakdown.py start TRANSCRIPT` prints request 0 itemized as above for any transcript.
- `calibrate` prints the fitted ratios; `growth TRANSCRIPT` prints the split for one transcript.
- `reads [--since-agent ID] [--autowork-only]` lists whole-file reads and their carried cost.

Autowork's subagents use their own agent type, `autowork-ticket` (`.claude/agents/autowork-ticket.md`), whose tool allow-list leaves out the Agent and Artifact tools that the procedure never uses; that is expected to remove roughly 13,300 to 15,300 tokens of tool definitions from request 0 and, from request 1 on, about 19,000 tokens per request once the 5,570-token agent list is also gone (figures inferred from the built-in subagent types that lack those tools, not yet measured on this definition). To check it after a restart (agent types are found when a session starts), run one ticket and run `start` on the subagent's transcript: request 0 is expected near 41,000 to 43,000 tokens and the transcript should contain no `agent_listing_delta` attachment. Until that check has been run, treat the saving as a prediction.

To re-measure the start without a ticket, launch a no-op subagent (`general-purpose`, a one-line prompt) and run `start` on its transcript; compare subagent types to see what each tool set costs. Reading a text through a no-op subagent's Read tool and subtracting a control read gives that text's exact token count.
