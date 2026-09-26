# Shared metrics store

A generic, cross-tool metrics-recording primitive (issue #208): ONE shared SQLite database any domain in this toolkit can write append-only events into, behind ONE small class and ONE MCP tool — instead of each new metric domain (memory-bank's `recall`/`remember` events, `my-gh-autowork`'s per-ticket run metrics, whatever comes next) inventing its own schema and its own dedicated MCP tool.

**Why one shared tool, not one per domain:** this repo's own tool-count sanity check found 23 tools already riding along in context on every single turn across the four connected MCP servers, regardless of whether they're used that turn (see `EVALUATION.md`'s Track A4/B4/E4). A metrics-reporting design that adds a new MCP tool per future metric domain would directly work against that cost model. A single generic dispatch tool (`get_metrics`) adds a flat, bounded cost no matter how many domains eventually write into this store.

**Status as of this ticket:** infrastructure only. No domain writes into this store yet — migrating `libs/savings_ledger.py`'s `savings.db` or `libs/memory_events_lib.py`'s `memory-events.db` onto it, and migrating `my-gh-autowork`'s own `compact_store`-based run metrics onto it, are both explicit, separate follow-up tickets (see issue #208's "Explicitly out of scope" section). Both existing databases are genuinely tailored schemas that work fine today; whether they ever adopt this shared store is a call to make once a real second domain exists to compare against.

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
```

Writes (`record`/`increment`/`decrement`) **fail open** — a broken/locked/corrupt metrics database is logged to stderr and swallowed, never raised, since the actual domain operation an event describes has already happened by the time it's logged. Reads (`summary`/`by_event_type`/`trend`) do **not** fail open — they raise, and it's the caller's job to decide whether to surface a friendlier error (the `get_metrics` MCP tool below does exactly that, matching `savings_summary`/`savings_detail`/`savings_trend`'s own try/except pattern).

## The `get_metrics` MCP tool

Added to the `local-compress` server (already the natural home — it hosts `savings_summary`/`savings_detail`/`savings_trend` and is where `/my-savings` already looks).

```
get_metrics(metric_id, view="summary", bucket="week", n=12, format="text")
```

- `view`: `"summary"` (default, total event count + total value), `"by_event_type"` (per-event_type breakdown), or `"trend"` (day/week-bucketed history).
- `bucket`/`n`: only apply to `view="trend"`.
- `format`: `"text"` (default, human-readable) or `"json"`.

## Using it

**Install the skill first**: `claude-runway-setup init --install-skills` (or `python tools/setup_project.py init --install-skills` from a clone) installs/updates `my-metrics` along with every other product skill — equivalent manual copy: `mkdir -p ~/.claude/skills/my-metrics && cp skills/my-metrics/SKILL.md ~/.claude/skills/my-metrics/SKILL.md`. Product skills are only discovered once installed under `~/.claude/skills/` — the rest of the setup flow (`.mcp.json`/`.claude/settings.json`) doesn't install them for you (same step `docs/savings-tracker.md`'s "Enabling it" section documents for `/my-savings`).

`/my-metrics <metricId> [view] [bucket]` — thin wrapper matching `/my-savings`'s shape, e.g.:

- `/my-metrics autowork` — summary view.
- `/my-metrics autowork by_event_type` — per-event-type breakdown.
- `/my-metrics autowork trend day` — daily trend.

Since no domain writes into this store yet, every query currently reports "no events recorded yet" honestly rather than fabricating data — that's expected, not a bug, until a future ticket migrates a real domain onto it.
