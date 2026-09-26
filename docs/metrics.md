# Shared metrics store

A generic, cross-tool metrics-recording primitive (issue #208): ONE shared SQLite database any domain in this toolkit can write append-only events into, behind ONE small class and ONE MCP tool — instead of each new metric domain (memory-bank's `recall`/`remember` events, `my-gh-autowork`'s per-ticket run metrics, whatever comes next) inventing its own schema and its own dedicated MCP tool.

**Why one shared tool, not one per domain:** this repo's own tool-count sanity check found 23 tools already riding along in context on every single turn across the four connected MCP servers, regardless of whether they're used that turn (see `EVALUATION.md`'s Track A4/B4/E4). A metrics-reporting design that adds a new MCP tool per future metric domain would directly work against that cost model. A single generic dispatch tool (`get_metrics`) adds a flat, bounded cost no matter how many domains eventually write into this store.

**Status as of issue #209:** a one-time migration script (`tools/migrate_autowork_metrics.py`) now exists that reads every point previously stored via `compact_store(project="claude-runway-autowork-metrics", ...)` and inserts an equivalent row here under `metric_id="autowork"`, preserving each entry's own original date rather than stamping the migration's run-time. **This only populates YOUR OWN local `metrics.db` after you actually run that script** against your own Qdrant instance — a fresh install, or one where the migration hasn't been run yet, has an empty `metric_id="autowork"` exactly like any other not-yet-written domain (see "Using it" below). `my-gh-autowork`'s LIVE logging still goes through `compact_store` regardless — cutting Step 2b over to write directly into this store is issue #210, a separate follow-up (the migration had to land and be verified complete first, precisely so no ticket worked in the gap between the two would have its history silently skipped). Migrating `libs/savings_ledger.py`'s `savings.db` or `libs/memory_events_lib.py`'s `memory-events.db` onto this store remains a separate, not-yet-started follow-up (see issue #208's "Explicitly out of scope" section) — both are genuinely tailored schemas that work fine today.

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

`record()` also takes an optional `event_timestamp` override (issue #209) — full ISO 8601 UTC (`"YYYY-MM-DDTHH:MM:SSZ"`), used verbatim instead of the current time. Ordinary callers should never pass this; it exists for backfilling historical events with their own original date (see `tools/migrate_autowork_metrics.py` below) — an invalid format fails open (logged to stderr, nothing written), same as the non-finite `value` guard.

Writes (`record`/`increment`/`decrement`) **fail open** — a broken/locked/corrupt metrics database is logged to stderr and swallowed, never raised, since the actual domain operation an event describes has already happened by the time it's logged. Reads (`summary`/`by_event_type`/`trend`) do **not** fail open — they raise, and it's the caller's job to decide whether to surface a friendlier error (the `get_metrics` MCP tool below does exactly that, matching `savings_summary`/`savings_detail`/`savings_trend`'s own try/except pattern).

## The `get_metrics` MCP tool

Added to the `local-compress` server (already the natural home — it hosts `savings_summary`/`savings_detail`/`savings_trend` and is where `/my-savings` already looks).

```
get_metrics(metric_id, view="summary", bucket="week", n=12, format="text")
```

- `view`: `"summary"` (default, total event count + total value), `"by_event_type"` (per-event_type breakdown), or `"trend"` (day/week-bucketed history).
- `bucket`/`n`: only apply to `view="trend"`.
- `format`: `"text"` (default, human-readable) or `"json"`.

## Historical migration (`tools/migrate_autowork_metrics.py`, issue #209)

A one-time, throwaway script — not meant to be run more than once in normal
operation, though it's idempotent (safe to re-run; already-migrated points
are detected via a `qdrant_point_id` tag stashed in `metadata` and skipped)
in case of a partial failure or to pick up points logged after an earlier
run. It:

1. Reads every point under Qdrant project `claude-runway-autowork-metrics`
   (resolved via `tools/compress_mcp_server.py`'s own `_collections_for_project`,
   so it can never drift from what `compact_store`/`compact_find` themselves
   resolve for that project).
2. Maps each point's `information` JSON payload into one `metric_id="autowork"`
   row: `event_type` derived from `outcome` (`"ticket_merged"`/
   `"ticket_blocked"`/`"ticket_failed"`, or `"ticket_unknown"` for anything
   else), `value=1.0`, `metadata` set to the full original JSON plus
   `qdrant_point_id`/`qdrant_collection`, and `event_timestamp` taken from
   the point's own `date` field (not migration run-time).
3. Verifies the resulting metrics.db row count matches the Qdrant point
   count before reporting PASS/FAIL (non-zero exit on a mismatch).

```
.venv/bin/python tools/migrate_autowork_metrics.py [--dry-run]
```

`--dry-run` previews the mapping without writing anything. Does NOT touch
`.claude/skills/my-gh-autowork/SKILL.md` or delete anything from Qdrant —
live logging still goes through `compact_store` until issue #210's cutover.

## Using it

**Install the skill first**: `claude-runway-setup init --install-skills` (or `python tools/setup_project.py init --install-skills` from a clone) installs/updates `my-metrics` along with every other product skill — equivalent manual copy: `mkdir -p ~/.claude/skills/my-metrics && cp skills/my-metrics/SKILL.md ~/.claude/skills/my-metrics/SKILL.md`. Product skills are only discovered once installed under `~/.claude/skills/` — the rest of the setup flow (`.mcp.json`/`.claude/settings.json`) doesn't install them for you (same step `docs/savings-tracker.md`'s "Enabling it" section documents for `/my-savings`).

`/my-metrics <metricId> [view] [bucket]` — thin wrapper matching `/my-savings`'s shape, e.g.:

- `/my-metrics autowork` — summary view.
- `/my-metrics autowork by_event_type` — per-event-type breakdown.
- `/my-metrics autowork trend day` — daily trend.

- `/my-metrics autowork` reflects `my-gh-autowork`'s full prior history ONLY after you've run `tools/migrate_autowork_metrics.py` against your own Qdrant instance (issue #209) — it is not automatic, and not real-time even after running it, since live logging is still Qdrant-only until issue #210 lands. Before running the migration (or for any OTHER metric_id nothing writes into), it honestly reports "no events recorded yet" rather than fabricating data.
