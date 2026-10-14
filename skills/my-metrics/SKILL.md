# Skill: my-metrics

Show a view of ClaudeRunway's shared cross-tool metrics store: a generic SQLite table (`~/.claude/claude-runway/metrics.db`) any domain in this toolkit can write into via `libs/metrics_lib.py`'s `MetricsStore` class, read here through the `get_metrics` MCP tool from the opt-in `analytics` server. That server exists only when the project's `.mcp.json` has an `analytics` block (`setup_project.py init --track-savings` writes it; `upgrade` offers it to a project that already tracks savings). Without that block `get_metrics` does not exist, and the call fails as an unknown tool, which is not the same as "no data".

**Important — this is generic infrastructure, not a specific report.** Which `metricId`s have a writer today:

- `autowork`: `/my-gh-autowork`'s per-ticket outcome rows, including the `ticket_cost` rows that `tools/autowork_cost.py record` writes. `/my-metrics autowork` shows outcome totals; `/my-metrics autowork detail` shows each ticket's row.
- `memory-bank`: a plain tally of `remember`/`recall`/`forget` calls (`/my-metrics memory-bank`). `memory_events_lib` also keeps its own richer table in `memory-events.db`, deliberately separate (that table's per-memory and per-repo rollups do not fit this store's schema-free design); `tools/memory_bank_report.py` reads it.

**Savings are not in this store.** The savings tracker keeps its own database (`savings.db`); read it with `/my-savings`, not with `/my-metrics`. Asking `get_metrics` for a savings metric id returns "no events recorded yet", which would be misleading.

Any other `metricId` has no writer and honestly reports "no events recorded yet": that is expected, not a bug.

**For the views `get_metrics` cannot give**, run the reporting CLIs from the claude-runway checkout instead (they read files and print text; none calls a model):

- Per-ticket cost and cache-read share of `/my-gh-autowork` runs: `python tools/autowork_cost.py report`.
- Memory-bank usage (dead-memory ratio, recall payload trend, per project): `python tools/memory_bank_report.py`.
- What a subagent starts with and why its context grows: `python tools/context_breakdown.py` (`start`, `calibrate`, `growth`, `reads`).
- How many tools each server exposes and how big their descriptions are: `python tools/report_tool_counts.py`.

Required argument: `metricId` — the domain to query (`autowork` or `memory-bank` today), the string a writer used when calling `record()`/`increment()`/`decrement()`.

Optional argument: `view` — `summary` (default), `by_event_type`, `trend` (optionally followed by `day` or `week` for the bucket size, e.g. `/my-metrics autowork trend day`), or `detail` — raw per-event rows, most-recent first, with each row's `metadata` fields shown (e.g. autowork's model/rounds/wall_clock_s/findings) instead of just an aggregate count/total.

Optional argument: `sessionId` — narrows `summary`/`by_event_type`/`detail` down to one specific session's rows (e.g. one `my-gh-autowork` attempt's own `issue-<n>-<HHMMSS>` marker), instead of that `metricId`'s entire history. Not supported for `view=trend` (per-session filtering doesn't compose with time-bucketed grouping) — pass it after `view` for `summary`/`by_event_type`/`detail` only, e.g. `/my-metrics autowork summary issue-<n>-143022`.

## When to use

Run this any time you want to inspect a metric domain's recorded events — a total, a per-event-type breakdown, a day/week trend, or the raw rows — optionally narrowed to one session. You need to know the `metricId` you're asking about; this skill doesn't enumerate which metric_ids currently have data (see the list above for the ones with a writer).

## Steps

1. **Parse the arguments**: the first word is `metricId` (required — if missing, ask the user which metric_id to query rather than guessing one). The second word, if present, is `view` (`summary`/`by_event_type`/`trend`/`detail`); default to `summary` if omitted.
   - If `view` is `trend`: a third word equal to exactly `day` or `week` sets the bucket size; default to `week` if omitted. At most 3 words are valid for `trend` — reject BOTH (a) any third word that isn't exactly `day` or `week` (e.g. something that looks like a `sessionId`), AND (b) any word beyond the third (e.g. `/my-metrics autowork trend day issue-<n>-143022`, where `day` is a valid bucket but the trailing `issue-<n>-143022` is not). Either case is an unsupported argument for `trend` — treat it as an error and tell the user per-session trend filtering isn't available rather than silently dropping it.
   - If `view` is `summary`, `by_event_type`, or `detail`, an optional third word is `sessionId` — pass it through as-is. At most 3 words are valid here too — reject a 4th word the same way.

2. **Call the `get_metrics` MCP tool** (from the `analytics` server) with `metric_id=<metricId>` and `view=<view>` (plus `bucket=<bucket>` when `view="trend"`, or `session_id=<sessionId>` when given for `summary`/`by_event_type`/`detail`). Print its return value to the user verbatim — it's pre-formatted text, not something to summarize or reformat.

3. **If the tool reports "no events recorded yet"**: this is normal — relay it plainly, don't treat it as an error, and don't fabricate example data. If the user asked about savings, point them at `/my-savings` instead.

4. **If `get_metrics` is not available** (the call fails because no such tool is connected): the project has no `analytics` server. Say so, and tell the user `setup_project.py init --track-savings` writes it (or `setup_project.py upgrade` adds it to a project that already tracks savings), and that Claude Code must be restarted afterwards. Don't guess at numbers.

5. **If the tool reports it's unavailable** (`libs/metrics_lib.py` missing from this install): relay that message plainly.

## Notes

- Never writes, modifies, or deletes any metric event data. It CAN, however, initialize the metrics database itself the first time it's queried — `get_metrics` (via `MetricsStore`) creates `~/.claude/claude-runway/metrics.db`'s parent directory, table, and indexes if they don't already exist yet, the same lazy-init behavior `savings_ledger.py`/`memory_events_lib.py` already use. That's schema/file-existence setup, not a mutation of any metric's actual event history.
- `metric_id`/`event_type` are domain-defined free-text strings, not a fixed enum this skill validates — a typo'd `metricId` simply returns "no events recorded yet" rather than an error, since the store itself has no way to distinguish "wrong name" from "no data yet for a valid name."
- See `docs/metrics.md` for the full schema and `MetricsStore` API this store is built on.
