# Shared metrics store

A generic, cross-tool metrics-recording primitive (issue #208): ONE shared SQLite database any domain in this toolkit can write append-only events into, behind ONE small class and ONE MCP tool — instead of each new metric domain (memory-bank's `recall`/`remember` events, `my-gh-autowork`'s per-ticket run metrics, whatever comes next) inventing its own schema and its own dedicated MCP tool.

**Why one shared tool, not one per domain:** this repo's own tool-count sanity check found 23 tools already riding along in context on every single turn across the four connected MCP servers, regardless of whether they're used that turn (see `EVALUATION.md`'s Track A4/B4/E4). A metrics-reporting design that adds a new MCP tool per future metric domain would directly work against that cost model. A single generic dispatch tool (`get_metrics`) adds a flat, bounded cost no matter how many domains eventually write into this store.

**Status:** `my-gh-autowork`'s per-run outcome logging now writes into this store — the first real domain writing into `metrics.db`, using `metric_id="autowork"`. Use `/my-metrics autowork` to see a summary. A fresh install has no autowork data until at least one run completes. Migrating `libs/savings_ledger.py`'s `savings.db` or `libs/memory_events_lib.py`'s `memory-events.db` onto this store remains an explicit, separate follow-up.

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

`summary()`/`by_event_type()`/`detail()` also take an optional `session_id` (issue #248, extended to `detail()` by issue #249) to narrow the result down to one specific session's rows (e.g. one `my-gh-autowork` attempt's own `issue-<n>-<HHMMSS>` marker from issue #210) instead of the metric_id's entire history. Omitting it keeps the metric_id-only result unchanged. Not available on `trend()` — per-session filtering doesn't compose with time-bucketed grouping (out of scope for #248). The text formatters (`format_summary_view`/`format_by_event_type_view`/`format_detail_view`) are session-aware too: a `session_id`-filtered query with zero matching rows reports "No events recorded yet for this metric_id with session_id=...", not the unqualified "no events for this metric_id" message — that message would be misleading when other sessions under the same metric_id do have data.

`detail()` (issue #249) is the raw, per-event counterpart to `summary()`/`by_event_type()`'s aggregates: it returns individual rows, most-recent first, with each row's `metadata` deserialized back into a dict — the richer per-run detail (e.g. autowork's `model`/`rounds`/`wall_clock_s`/`subagent_tokens`/`tool_calls`/`findings`) that the two aggregate views never surface, since those only ever sum `value`. Deliberately generic: it doesn't hardcode any domain's metadata field names, so any future `metric_id` that stores structured metadata gets this view for free. `limit` caps the number of rows returned (default 12, same default as `trend()`'s bucket count) via SQL `LIMIT`, not a Python-side slice.

`record()` also takes an optional `event_timestamp` override — full ISO 8601 UTC (`"YYYY-MM-DDTHH:MM:SSZ"`), used verbatim instead of the current time. Ordinary callers should never pass this; it exists for backfilling historical events with their own original dates — an invalid format fails open (logged to stderr, nothing written), same as the non-finite `value` guard.

Writes (`record`/`increment`/`decrement`) **fail open** — a broken/locked/corrupt metrics database is logged to stderr and swallowed, never raised, since the actual domain operation an event describes has already happened by the time it's logged. Reads (`summary`/`by_event_type`/`trend`) do **not** fail open — they raise, and it's the caller's job to decide whether to surface a friendlier error (the `get_metrics` MCP tool below does exactly that, matching `savings_summary`/`savings_detail`/`savings_trend`'s own try/except pattern).

## MCP tools

Both live in the `local-compress` server (already the natural home — it hosts `savings_summary`/`savings_detail`/`savings_trend` and is where `/my-savings` already looks).

### `record_metric` (write)

Added by issue #210 — the write counterpart to `get_metrics`.

```
record_metric(metric_id, event_type, value=1.0, metadata=None, session_id=None)
```

- `metric_id`: the domain (e.g. `"autowork"`) — caller-defined free-text string.
- `event_type`: domain-defined event kind (e.g. `"ticket_merged"`, `"ticket_blocked"`).
- `value`: a signed numeric delta (default `1.0`). Must be finite.
- `metadata`: optional domain-specific extra fields, as either a JSON string or an already-parsed dict — both shapes are accepted directly (a client that coerces a JSON-shaped string argument into a native object before the tool sees it, as Claude Code itself does, still succeeds).
- `session_id`: optional caller-supplied identifier stored in the raw row — `get_metrics`/`/my-metrics` can filter `view="summary"`/`"by_event_type"` down to one exact `session_id` (issue #248).

Returns `"OK"` on success or an `"Error:..."` string on any failure — including DB write failures (when `MetricsStore.record()` returns `False`). DB write errors are also logged to stderr but not raised, preserving fail-open behavior; the `"Error:"` return is the only surface the caller sees.

### `get_metrics` (read)

```
get_metrics(metric_id, view="summary", bucket="week", n=12, format="text", session_id=None)
```

- `view`: `"summary"` (default, total event count + total value), `"by_event_type"` (per-event_type breakdown), `"trend"` (day/week-bucketed history), or `"detail"` (issue #249 — raw per-event rows with deserialized `metadata`, most recent first).
- `bucket`: only applies to `view="trend"`.
- `n`: "max buckets" for `view="trend"`, or "max rows" for `view="detail"` — ignored for `"summary"`/`"by_event_type"`. Reused across both rather than a second row-count parameter, since only one meaning ever applies for a given call.
- `session_id` (issue #248, extended to `"detail"` by issue #249): applies to `view="summary"`/`"by_event_type"`/`"detail"` — narrows the result to one specific session (e.g. one `my-gh-autowork` attempt's `issue-<n>-<HHMMSS>` marker). Ignored for `view="trend"` (not supported there — see the `MetricsStore` section above).
- `format`: `"text"` (default, human-readable) or `"json"`.

## Using it

**Install the skill first**: `claude-runway-setup init --install-skills` (or `python tools/setup_project.py init --install-skills` from a clone) installs/updates `my-metrics` along with every other product skill — equivalent manual copy: `mkdir -p ~/.claude/skills/my-metrics && cp skills/my-metrics/SKILL.md ~/.claude/skills/my-metrics/SKILL.md`. Product skills are only discovered once installed under `~/.claude/skills/` — the rest of the setup flow (`.mcp.json`/`.claude/settings.json`) doesn't install them for you (same step `docs/savings-tracker.md`'s "Enabling it" section documents for `/my-savings`).

`/my-metrics <metricId> [view] [bucket|sessionId]` — thin wrapper matching `/my-savings`'s shape, e.g.:

- `/my-metrics autowork` — summary view (total tickets worked, all outcomes).
- `/my-metrics autowork by_event_type` — breakdown by outcome type (`ticket_merged`, `ticket_blocked`, `ticket_failed`).
- `/my-metrics autowork trend day` — daily trend.
- `/my-metrics autowork detail` — the last 12 autowork runs, each with its full `metadata` (model, rounds, wall_clock_s, subagent_tokens, tool_calls, findings) shown (issue #249).
- `/my-metrics autowork summary issue-248-143022` — summary narrowed to one specific autowork attempt's `session_id` (issue #248). Only supported for `summary`/`by_event_type`/`detail`, not `trend`.

Live data appears immediately with no migration needed.
