# Skill: my-metrics

Show a view of ClaudeRunway's shared cross-tool metrics store (issue #208): a generic SQLite table (`~/.claude/claude-runway/metrics.db`) any domain in this toolkit can write into via `libs/metrics_lib.py`'s `MetricsStore` class, read here through the `get_metrics` MCP tool (from the `local-compress` server).

**Important — this is generic infrastructure, not a specific report.** No domain writes into this store yet as of this ticket landing (migrating `savings_ledger`/`memory_events_lib`, or `my-gh-autowork`'s own run metrics, onto it are separate follow-up tickets — see issue #208's "Explicitly out of scope" section). Until a future domain writes real data, every `metric_id` you try here will honestly report "no events recorded yet" — that's expected, not a bug.

Required argument: `metricId` — the domain to query (e.g. `autowork`, `memory-bank`), whatever string a writer used when calling `record()`/`increment()`/`decrement()`.

Optional argument: `view` — `summary` (default), `by_event_type`, or `trend` (optionally followed by `day` or `week` for the bucket size, e.g. `/my-metrics autowork trend day`).

## When to use

Run this any time you want to inspect a specific metric domain's recorded events — a total, a per-event-type breakdown, or a day/week trend. You need to know the `metricId` you're asking about; this skill doesn't enumerate which metric_ids currently have data.

## Steps

1. **Parse the arguments**: the first word is `metricId` (required — if missing, ask the user which metric_id to query rather than guessing one). The second word, if present, is `view` (`summary`/`by_event_type`/`trend`); default to `summary` if omitted. If `view` is `trend`, an optional third word (`day` or `week`) sets the bucket size; default to `week`.

2. **Call the `get_metrics` MCP tool** (from the `local-compress` server) with `metric_id=<metricId>` and `view=<view>` (plus `bucket=<bucket>` when `view="trend"`). Print its return value to the user verbatim — it's pre-formatted text, not something to summarize or reformat.

3. **If the tool reports "no events recorded yet"**: this is normal — relay it plainly, don't treat it as an error, and don't fabricate example data.

4. **If the tool reports it's unavailable** (`libs/metrics_lib.py` missing from this install): relay that message plainly.

## Notes

- Never writes, modifies, or deletes any metric event data. It CAN, however, initialize the metrics database itself the first time it's queried — `get_metrics` (via `MetricsStore`) creates `~/.claude/claude-runway/metrics.db`'s parent directory, table, and indexes if they don't already exist yet, the same lazy-init behavior `savings_ledger.py`/`memory_events_lib.py` already use. That's schema/file-existence setup, not a mutation of any metric's actual event history.
- `metric_id`/`event_type` are domain-defined free-text strings, not a fixed enum this skill validates — a typo'd `metricId` simply returns "no events recorded yet" rather than an error, since the store itself has no way to distinguish "wrong name" from "no data yet for a valid name."
- See `docs/metrics.md` for the full schema and `MetricsStore` API this store is built on.
