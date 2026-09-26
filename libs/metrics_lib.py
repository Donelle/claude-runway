"""
Shared, cross-tool metrics store (issue #208) -- ONE generic SQLite table
any domain in this toolkit can write into (memory-bank's recall/remember
events, my-gh-autowork's per-ticket run metrics, whatever comes next),
behind ONE small `MetricsStore` class and ONE new MCP tool
(`get_metrics`, added to tools/compress_mcp_server.py -- see that file).

Why this exists instead of "add a new MCP tool per domain": this repo's own
tool-count sanity check (issue #208's own investigation) found 23 tools
already riding along in context on every single turn across the four
connected MCP servers, regardless of whether they're used that turn -- see
EVALUATION.md's Track A4/B4/E4. A metrics-reporting design that adds one new
tool per future metric domain would directly work against that cost model;
a single generic dispatch tool (`get_metrics`) adds a flat, bounded cost
regardless of how many domains eventually write into this store.

Deliberately NOT the home for savings_ledger.py's `savings.db` or
memory_events_lib.py's `memory-events.db` -- both are genuinely tailored
schemas that already work well for their one specific domain each,
migrating them onto this generic store is an explicit follow-up, not part
of this ticket (see issue #208's "Explicitly out of scope" section).
As of issue #210, my-gh-autowork's per-ticket outcome logging is the first
real domain writing into this store (`metric_id="autowork"`).

Storage shape: single append-only `metrics` table --
`(id, metric_id, event_type, value, metadata, event_timestamp, session_id)`.
`metric_id` is the domain (e.g. "autowork", "memory-bank"); `event_type` is
domain-defined (e.g. "ticket_merged", "recall"). Never a mutable in-place
counter -- "current value" for any metric_id/event_type pair is always
`SUM(value)` computed at READ time (via summary()/by_event_type()/trend()
below), the same append-only-log-then-aggregate design
`libs/savings_ledger.py`/`libs/memory_events_lib.py` already establish and
this module's own docstring inherits the reasoning for: multiple separate
stdio MCP server processes (local-compress, codebase-indexer, memory-bank)
could in principle write concurrently, and a mutable counter's
read-modify-write cycle isn't safe against that without locking a
log-then-aggregate design doesn't need.

Schema is intentionally generic (no per-domain columns) so a brand-new
metric_id NEVER requires a schema migration -- only a genuine change to the
`metrics` table's own shape would. Unlike `savings_ledger.py`, this is a
brand-new v1 table with no prior shape to preserve, so there's no migration
scaffold yet; add one (SCHEMA_VERSION + a migrations map, following that
file's `_run_migrations` pattern) the first time this schema actually needs
to change.

Follows this repo's small, function/class-based, dependency-light interface
over SQLite (same shape as `libs/memory_events_lib.py`/`libs/cache_db.py`)
rather than an ORM -- swapping the backend later only touches this file.

Writes fail OPEN (a broken/locked/corrupt metrics db must never break the
actual domain operation being measured), matching
`memory_events_lib.record_memory_event`'s philosophy exactly. Reads
(summary/by_event_type/trend) do NOT fail open -- they raise, the same way
`savings_ledger.py`'s query_* functions do; it's the CALLER's job to decide
whether a read failure should produce a friendly error string (see
`get_metrics` in tools/compress_mcp_server.py, which wraps these in a
try/except exactly like `savings_summary`/`savings_detail`/`savings_trend`
already do) or propagate further.
"""

import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


def resolve_db_path() -> Path:
    """
    Same convention as `savings_ledger.resolve_db_path()`/
    `memory_events_lib.resolve_db_path()`: `CLAUDE_RUNWAY_METRICS_DB` env var
    (absolute path) if set, else `~/.claude/claude-runway/metrics.db`. A
    relative override is anchored to the home directory rather than left
    ambiguous against whichever process's cwd happened to be current -- same
    reasoning as those two functions.
    """
    override = os.environ.get("CLAUDE_RUNWAY_METRICS_DB")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.home() / path
        return path
    return Path.home() / ".claude" / "claude-runway" / "metrics.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    """
    Opens the db file and ensures its schema exists. Mirrors
    `memory_events_lib._connect()`'s exact failure handling: if schema/index
    creation fails partway through (a locked/corrupt db file -- realistic
    here since every call re-runs these `CREATE TABLE IF NOT EXISTS`/
    `CREATE INDEX IF NOT EXISTS` statements), the connection this function
    already opened is explicitly closed before the error propagates --
    otherwise a caller whose `try` only wraps the call to `_connect()`
    itself would never get a handle back to close, leaking the open
    connection/file descriptor on every failed call.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                value REAL NOT NULL,
                metadata TEXT,
                event_timestamp TEXT NOT NULL,
                session_id TEXT
            )
            """
        )
        # Supports summary()/by_event_type()/trend() below -- created eagerly
        # rather than waiting for a later feature to add them, since ALTER-ing
        # indexes onto a table that may already hold rows is strictly more
        # disruptive than creating them alongside the table from day one.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_metric_id ON metrics (metric_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_event_type ON metrics (metric_id, event_type)")
        # PR #222 review: trend()'s `ORDER BY event_timestamp DESC WHERE
        # metric_id = ?` had no index covering that access pattern, forcing
        # SQLite to scan the whole table for one metric_id and then sort the
        # result. DESC in the index definition itself (supported since
        # SQLite 3.7) lets a `... WHERE metric_id = ? ORDER BY
        # event_timestamp DESC` query walk the index directly in the
        # already-correct order, with no separate sort step -- combined with
        # trend()'s own early cursor break (see that method's docstring),
        # this bounds actual index pages read to roughly what's needed to
        # fill `n` buckets, not the metric_id's entire history.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_trend ON metrics (metric_id, event_timestamp DESC)")
    except sqlite3.Error:
        conn.close()
        raise
    return conn


class MetricsStore:
    """
    Thin wrapper over the shared `metrics` SQLite table. `db_path` defaults
    to `None`, meaning "resolve `resolve_db_path()` fresh on every call" --
    deliberately NOT cached at construction time, so a test (or a caller)
    that changes `CLAUDE_RUNWAY_METRICS_DB` between calls, or constructs a
    long-lived module-level instance across a process lifetime, always sees
    the current value rather than a stale one captured at `__init__` time.
    Pass an explicit `db_path` to pin one path regardless of the env var
    (e.g. for a test using a temp file without needing to patch the
    environment).
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._explicit_db_path = db_path

    def _resolve_path(self) -> Path:
        return self._explicit_db_path if self._explicit_db_path is not None else resolve_db_path()

    # -------------------------------------------------------------------
    # Writes -- fail open (see module docstring)
    # -------------------------------------------------------------------

    def record(
        self,
        metric_id: str,
        event_type: str,
        value: float = 1.0,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
        event_timestamp: Optional[str] = None,
    ) -> bool:
        """
        Append one row. Returns True on success, False on any failure.

        `metric_id`/`event_type` are domain-defined strings
        (no fixed enum here, unlike `memory_events_lib`'s `event_type` --
        this table is meant to serve domains not yet known, so validating
        against a hardcoded set would defeat the point). `metadata`, if
        given, must be JSON-serializable; it's stored as TEXT and decoded
        back to a dict by summary()/by_event_type()/trend() callers that
        need it (none of the three below actually need it today -- it's
        carried through for a future domain-specific reporting view to read
        directly from the raw table if needed).

        `event_timestamp`, if given, overrides the default "stamp with the
        current time" behavior below -- added for issue #209's one-time
        migration of `my-gh-autowork`'s pre-existing Qdrant-stored history
        into this store (`tools/migrate_autowork_metrics.py`), which needs
        to preserve each migrated event's OWN original date rather than
        having every historical entry collapse onto the migration's own
        run-time: `trend()`'s day/week bucketing would otherwise put years
        of prior history into a single "today" bucket. Must already be full
        ISO 8601 UTC (`"YYYY-MM-DDTHH:MM:SSZ"`, matching this column's own
        documented format) -- validated via `datetime.strptime` (REAL
        calendar/time validation, e.g. rejects `"2026-99-99T99:99:99Z"`;
        PR #246 review caught an earlier version that only checked
        character-class SHAPE via regex, which would have accepted that
        exact string and persisted it, only for `trend()`'s own
        `datetime.fromisoformat` parsing to raise `ValueError` on it later
        for every future `trend()` call against this metric_id, not just
        this one row). Validated with the same fail-open philosophy as the
        non-finite `value` guard above: an invalid override is refused
        before the DB is even opened (logged to stderr, nothing written).
        Ordinary callers should never pass this -- omitting it (the
        default) keeps today's "stamp with now" behavior for every existing
        call site.

        Fails OPEN on any write failure (unwritable/locked/corrupt db file,
        non-JSON-serializable metadata, or a non-finite `value`): caught and
        logged to stderr, never raised -- returns False instead. By the time
        this runs, the actual operation this event describes has already
        happened -- a broken
        passive metrics log must never turn that into a reported failure
        for the caller. Same fail-open philosophy as
        `memory_events_lib.record_memory_event`.

        `value` must be finite (not `inf`/`-inf`/`nan`) -- PR #222 review:
        a `float` in Python/SQLite can be non-finite, and persisting one
        poisons EVERY future aggregate for this metric_id: `SUM(value)`
        propagates the non-finite value into `summary()`'s `total_value`
        forever (confirmed live: one `inf` row makes `total_value` `inf`
        even after later, ordinary rows are added), `_fmt_value()`'s
        `int(v)` check then raises `OverflowError` on that aggregate
        (confirmed live), and `format_json()` emits a bare `Infinity`/`NaN`
        token, which is not valid JSON per RFC 8259 even though Python's
        `json.dumps` emits it by default. Rejected here, before the DB is
        even opened, rather than validated at read time -- once a bad value
        is committed there's no way to "un-poison" a SUM() without deleting
        the offending row, which read-only aggregate methods have no
        business doing.
        """
        if not math.isfinite(value):
            print(f"[claude-runway] refusing to record non-finite metric value for {metric_id}/{event_type}: {value!r}", file=sys.stderr)
            return False
        if event_timestamp is not None:
            try:
                # PR #246 review (Copilot, round 1): a regex only checks
                # character SHAPE, not that the value is a real calendar
                # date/time -- it would accept "2026-99-99T99:99:99Z" and
                # persist it, which trend()'s datetime.fromisoformat parsing
                # then raises ValueError on for every future call against
                # this metric_id, not just this one row (confirmed by
                # reproduction). datetime.strptime performs real
                # calendar/time validation, not just character-class
                # matching.
                #
                # PR #246 review (Copilot, round 2): strptime alone still
                # isn't sufficient -- it's lenient about zero-padding, so it
                # happily parses "2026-8-5T01:02:03Z" (non-zero-padded month/
                # day) and would have persisted that non-canonical string
                # verbatim, only for trend()'s datetime.fromisoformat (which
                # requires the STRICT zero-padded form) to raise on it later
                # (confirmed by reproduction: strptime parses it, fromisoformat
                # rejects it). Reformatting the parsed value back through
                # strftime and requiring an EXACT round-trip match to the
                # original string is what actually enforces the canonical
                # "YYYY-MM-DDTHH:MM:SSZ" shape AND real calendar validity
                # together -- a value that round-trips has to be both.
                parsed = datetime.strptime(event_timestamp, "%Y-%m-%dT%H:%M:%SZ")
                if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != event_timestamp:
                    raise ValueError("not canonical (fails strict round-trip)")
            except (TypeError, ValueError):
                print(
                    f"[claude-runway] refusing to record metric {metric_id}/{event_type} with invalid "
                    f"event_timestamp override (expected canonical ISO 8601 UTC 'YYYY-MM-DDTHH:MM:SSZ'): {event_timestamp!r}",
                    file=sys.stderr,
                )
                return False
        try:
            metadata_json = json.dumps(metadata) if metadata is not None else None
        except (TypeError, ValueError) as e:
            print(f"[claude-runway] could not serialize metrics metadata for {metric_id}/{event_type}: {e}", file=sys.stderr)
            return False
        # `conn` is closed in a `finally`, not just after a successful
        # `with conn:` block -- `with conn:` only manages the transaction
        # (commit/rollback), it does NOT close the connection itself, and a
        # failure INSIDE that block (e.g. "database is locked" mid-insert)
        # would re-raise past a bare post-block conn.close() line, leaking
        # the connection exactly like an unclosed _connect() failure would
        # (same reasoning as memory_events_lib.record_memory_event).
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = _connect(self._resolve_path())
            with conn:
                conn.execute(
                    "INSERT INTO metrics (metric_id, event_type, value, metadata, event_timestamp, session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        metric_id,
                        event_type,
                        float(value),
                        metadata_json,
                        event_timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        session_id,
                    ),
                )
            return True
        except (sqlite3.Error, OSError) as e:
            print(f"[claude-runway] could not record metric {metric_id}/{event_type}: {e}", file=sys.stderr)
            return False
        finally:
            if conn is not None:
                conn.close()

    def increment(
        self,
        metric_id: str,
        event_type: str,
        by: float = 1.0,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Thin sugar over record() storing a positive signed delta -- NOT a
        mutable counter, see module docstring for why. Returns record()'s
        success bool."""
        return self.record(metric_id, event_type, value=by, metadata=metadata, session_id=session_id)

    def decrement(
        self,
        metric_id: str,
        event_type: str,
        by: float = 1.0,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Thin sugar over record() storing a negative signed delta. Returns
        record()'s success bool."""
        return self.record(metric_id, event_type, value=-by, metadata=metadata, session_id=session_id)

    # -------------------------------------------------------------------
    # Reads -- raise on failure (see module docstring); callers wrap these
    # in their own try/except when a friendlier error string is wanted
    # (e.g. tools/compress_mcp_server.py's get_metrics tool).
    # -------------------------------------------------------------------

    def summary(self, metric_id: str, session_id: Optional[str] = None) -> dict:
        """
        Aggregate totals for one metric_id across ALL event types:
        `{"metric_id", "event_count", "total_value", "first_event_at",
        "last_event_at"}`. `first_event_at`/`last_event_at` are `None` when
        the metric_id has no rows yet (never a fabricated timestamp).

        `session_id` (issue #248), when given, narrows the aggregate to rows
        matching BOTH `metric_id` AND this exact `session_id` -- e.g. one
        specific `my-gh-autowork` attempt's `f"issue-{n}-{HHMMSS}"` marker
        (see issue #210) rather than that metric_id's entire history.
        Omitting it (the default) keeps today's metric_id-only aggregate
        unchanged -- existing callers see no behavior change.
        """
        query = "SELECT COUNT(*), COALESCE(SUM(value), 0.0), MIN(event_timestamp), MAX(event_timestamp) FROM metrics WHERE metric_id = ?"
        params: tuple = (metric_id,)
        if session_id is not None:
            query += " AND session_id = ?"
            params = (metric_id, session_id)
        conn = _connect(self._resolve_path())
        try:
            row = conn.execute(query, params).fetchone()
        finally:
            conn.close()
        event_count, total_value, first_at, last_at = row
        return {
            "metric_id": metric_id,
            "event_count": event_count,
            "total_value": total_value,
            "first_event_at": first_at,
            "last_event_at": last_at,
        }

    def by_event_type(self, metric_id: str, session_id: Optional[str] = None) -> list:
        """
        Per-event_type breakdown for one metric_id, ordered by total_value
        descending (largest contributor first) -- mirrors
        `savings_ledger.query_session_tool_breakdown`'s ordering convention.
        Each row: `{"event_type", "event_count", "total_value"}`.

        `session_id` (issue #248): same optional narrowing as `summary()`
        above -- when given, only rows matching BOTH `metric_id` and this
        exact `session_id` are aggregated. Omitting it keeps today's
        metric_id-only breakdown unchanged.
        """
        query = "SELECT event_type, COUNT(*), COALESCE(SUM(value), 0.0) FROM metrics WHERE metric_id = ?"
        params: tuple = (metric_id,)
        if session_id is not None:
            query += " AND session_id = ?"
            params = (metric_id, session_id)
        query += " GROUP BY event_type ORDER BY 3 DESC"
        conn = _connect(self._resolve_path())
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
        return [
            {"event_type": r[0], "event_count": r[1], "total_value": r[2]}
            for r in rows
        ]

    def trend(self, metric_id: str, bucket: str = "week", n: int = 12) -> list:
        """
        Groups a metric_id's history by day or week, same bucketing
        semantics as `savings_ledger.query_trend` (see that function's
        docstring for why ISO week keys are computed in Python via
        `datetime.isocalendar()` rather than SQLite's own strftime
        modifiers, which can't produce correct ISO week keys portably
        across calendar-year boundaries and SQLite versions).

        Returns at most `n` buckets ordered ascending (oldest first,
        most-recent last), each `{"bucket", "event_count", "total_value"}`.
        Empty when the metric_id has no rows.

        Raises ValueError for an unrecognised `bucket` value, same as
        `savings_ledger.query_trend`.

        PR #222 review: this iterates the cursor directly and breaks out of
        the aggregation loop as soon as `n` distinct buckets are filled,
        rather than calling `.fetchall()` up front -- `savings_ledger.
        query_trend` can get away with `fetchall()` because its `sessions`
        table has at most one row per session (a naturally small
        cardinality); this store's `metrics` table is explicitly meant to
        hold many append-only rows per metric_id (e.g. every `recall`
        event), so materializing the ENTIRE history into a Python list
        before the bucket loop even runs would scale with total row count,
        not with `n`. Combined with the `idx_metrics_trend` index created
        in `_connect()` (covers `WHERE metric_id = ? ORDER BY
        event_timestamp DESC` directly, no separate sort step), this bounds
        both the SQL work AND the Python-side memory to roughly what's
        needed to fill `n` buckets, not the metric_id's entire history.
        """
        if bucket == "day":
            def _key(event_timestamp: str) -> str:
                # event_timestamp is ISO 8601 UTC, e.g.
                # "2026-09-24T12:00:00Z" -- the first 10 characters are
                # always YYYY-MM-DD.
                return event_timestamp[:10]
        elif bucket == "week":
            def _key(event_timestamp: str) -> str:  # type: ignore[no-redef]
                dt = datetime.fromisoformat(event_timestamp.replace("Z", "+00:00"))
                iso = dt.isocalendar()
                return f"{iso.year}-W{iso.week:02d}"
        else:
            raise ValueError(
                f"Unknown bucket {bucket!r}. Valid values: 'day', 'week'."
            )

        conn = _connect(self._resolve_path())
        try:
            cursor = conn.execute(
                "SELECT event_timestamp, value FROM metrics "
                "WHERE metric_id = ? ORDER BY event_timestamp DESC",
                (metric_id,),
            )
            agg: dict = {}
            # Iterating the cursor directly (not .fetchall()) so this loop's
            # own `break` actually stops further rows from being pulled off
            # the index -- see this method's docstring for why that matters
            # here specifically (unbounded row count per metric_id).
            for event_timestamp, value in cursor:
                key = _key(event_timestamp)
                if key not in agg:
                    if len(agg) >= n:
                        break
                    agg[key] = {"event_count": 0, "total_value": 0.0}
                agg[key]["event_count"] += 1
                agg[key]["total_value"] += value
        finally:
            conn.close()

        return [
            {"bucket": k, "event_count": v["event_count"], "total_value": v["total_value"]}
            for k, v in reversed(list(agg.items()))
        ]


# ---------------------------------------------------------------------------
# Formatting -- shared by the get_metrics MCP tool (tools/compress_mcp_server.py)
# and skills/my-metrics/SKILL.md, so both surfaces render identically. Kept
# deliberately simpler than savings_ledger.py's formatting (no bars/sparklines) --
# this store has no established human-facing view of its own yet (that's the
# "future /my-metrics reporting skill" the issue itself describes as a
# downstream consumer, not something this ticket needs to design in full).
# ---------------------------------------------------------------------------

def _fmt_value(v: float) -> str:
    # Renders as an int when the value is a whole number (the common case --
    # increment()/decrement() default to whole-number deltas), otherwise
    # keeps up to 2 decimal places without trailing zeros.
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _no_events_message(session_id: Optional[str] = None) -> str:
    # Issue #248 (Copilot review, round 4): a session_id-filtered query that
    # comes back empty must NOT say "No events recorded yet for this
    # metric_id" -- that's true of the empty FILTERED result, but false (and
    # misleading for per-attempt forensics) about the metric_id as a whole,
    # which may well have plenty of history under other session_ids.
    if session_id is not None:
        return f"  No events recorded yet for this metric_id with session_id={session_id!r}."
    return "  No events recorded yet for this metric_id."


def format_summary_view(summary: dict, session_id: Optional[str] = None) -> str:
    metric_id = summary.get("metric_id", "")
    event_count = summary.get("event_count", 0)
    total_value = summary.get("total_value", 0.0)
    first_at = summary.get("first_event_at")
    last_at = summary.get("last_event_at")
    lines = [
        f"ClaudeRunway · Metrics · {metric_id}",
        "",
        f"  Events        {event_count}",
        f"  Total value   {_fmt_value(total_value)}",
    ]
    if first_at:
        lines.append(f"  First event   {first_at}")
    if last_at:
        lines.append(f"  Last event    {last_at}")
    if not event_count:
        lines.append("")
        lines.append(_no_events_message(session_id))
    return "\n".join(lines)


def format_by_event_type_view(metric_id: str, rows: list, session_id: Optional[str] = None) -> str:
    header = f"ClaudeRunway · Metrics · {metric_id} · By event type"
    if not rows:
        return f"{header}\n\n{_no_events_message(session_id)}"
    lines = [
        header,
        "",
        f"  {'Event type':<24} {'Events':>8}  {'Total value':>12}",
        f"  {'-' * 24} {'-' * 8}  {'-' * 12}",
    ]
    for r in rows:
        lines.append(f"  {r['event_type']:<24} {r['event_count']:>8}  {_fmt_value(r['total_value']):>12}")
    return "\n".join(lines)


def format_trend_view(metric_id: str, trend_rows: list, bucket: str = "week") -> str:
    bucket_label = "Weekly" if bucket == "week" else "Daily"
    header = f"ClaudeRunway · Metrics · {metric_id} · {bucket_label} trend"
    if not trend_rows:
        return f"{header}\n\n  No event history yet for this metric_id."
    lines = [
        header,
        "",
        f"  {'Bucket':<16} {'Events':>8}  {'Total value':>12}",
        f"  {'-' * 16} {'-' * 8}  {'-' * 12}",
    ]
    for r in trend_rows:
        lines.append(f"  {r['bucket']:<16} {r['event_count']:>8}  {_fmt_value(r['total_value']):>12}")
    total_events = sum(r["event_count"] for r in trend_rows)
    total_value = sum(r["total_value"] for r in trend_rows)
    lines += [
        f"  {'-' * 16} {'-' * 8}  {'-' * 12}",
        f"  {'Total':<16} {total_events:>8}  {_fmt_value(total_value):>12}",
    ]
    return "\n".join(lines)


def format_json(view: str, data) -> str:
    """Machine-readable JSON export for any of the three views above --
    `data` is whatever summary()/by_event_type()/trend() returned."""
    return json.dumps({"view": view, "data": data}, indent=2)
