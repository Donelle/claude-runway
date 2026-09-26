"""
Passive, append-only usage log for memory-bank's `recall`/`remember` calls
(issue #179) -- a dedicated `~/.claude/claude-runway/memory-events.db`,
deliberately SEPARATE from `libs/savings_ledger.py`'s `savings.db`. That
file's schema is built around session-level TOKEN aggregates (a different
concern); this one is per-POINT access tracking (which memories get
recalled, how often, how soon after creation) -- mixing the two would couple
unrelated metric domains into one file/schema.

`forget` is deliberately NOT tracked here (see the issue) -- tracking it
would only matter alongside an autonomous-removal feature that doesn't exist
yet; today `forget` always requires explicit human confirm=True, so there's
nothing passive to observe.

Why append-only INSERT, not an in-place counter on the Qdrant point itself:
mutating a point's payload (read-current-count, increment, write-back) is a
read-then-write race under concurrent access -- multiple projects/sessions
on the same machine can hit the shared memory-bank collection at close to
the same moment, and the read-modify-write cycle can lose updates. An
INSERT-per-event log has no such race: every write is independent, and any
rollup (most-recalled memories, memories never recalled, retrieval trend
over time) is computed later via GROUP BY, not maintained in place.

Session identity -- a real constraint, not a shortcut: this file is written
from `tools/memory_bank_mcp_server.py`, an MCP server, which (unlike a hook)
has NO access to Claude Code's actual session_id -- see
`tools/compress_mcp_server.py`'s `_append_savings_footer` docstring, which
documents the identical limitation for the local-compress server. A stdio
MCP server subprocess is spawned fresh per Claude Code session (one process
per session -- see `libs/qdrant_collection_hints.py`'s docstring for the
same observation), so the caller resolves one process-lifetime UUID at
import time via `libs/session_id_lib.py`'s `SessionIdStrategy.PROXY`
(issue #198/#214, rather than minting its own `uuid.uuid4().hex` inline)
and passes it in as `session_id` on every call here -- a documented proxy
for "this session," not Claude Code's own internal id.

`turn` is likewise caller-maintained (a plain in-process counter), not
computed in this module -- see the caller (`tools/memory_bank_mcp_server.py`)
for why it increments once per tool CALL, not once per event row.

Follows this repo's small, function-based, dependency-light interface over
SQLite (same shape as `libs/cache_db.py`/`libs/qdrant_collection_hints.py`)
rather than an ORM -- swapping the backend later only touches this file.
This is a brand-new v1 table, so (unlike `savings_ledger.py`) there's no
migration scaffold yet; add one (SCHEMA_VERSION + a migrations map, following
that file's `_run_migrations` pattern) the first time this schema actually
needs to change.
"""

import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional


def tracking_enabled() -> bool:
    """
    Defaults to ON, unlike `savings_ledger.tracking_enabled()` (default OFF)
    -- deliberately different defaults for two different tradeoffs. The
    savings tracker's opt-in-only sibling variable
    (`CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS`) parses full session
    transcripts, a real privacy/cost consideration; this log only ever
    stores ids/timestamps/kind/repo -- never a memory's `summary` or
    `description` text -- and costs one local SQLite INSERT, no network, no
    model call, no token cost. Still exposes an explicit opt-out for anyone
    who wants zero local telemetry from this server: set
    `CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS` to a falsy value to disable.
    """
    raw = os.environ.get("CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no")


def resolve_db_path() -> Path:
    """
    Same convention as `savings_ledger.resolve_db_path()`/
    `cache_db.resolve_cache_db_path()`: `CLAUDE_RUNWAY_MEMORY_EVENTS_DB` env
    var (absolute path) if set, else `~/.claude/claude-runway/memory-events.db`.
    A relative override is anchored to the home directory rather than left
    ambiguous against whichever process's cwd happened to be current --
    same reasoning as those two functions.
    """
    override = os.environ.get("CLAUDE_RUNWAY_MEMORY_EVENTS_DB")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.home() / path
        return path
    return Path.home() / ".claude" / "claude-runway" / "memory-events.db"


_VALID_EVENT_TYPES = ("recall", "remember")


def _connect() -> sqlite3.Connection:
    """
    Opens the db file and ensures its schema exists. If schema/index
    creation fails partway through (a locked/corrupt db file -- realistic
    here since every single event call re-runs these `CREATE TABLE IF NOT
    EXISTS`/`CREATE INDEX IF NOT EXISTS` statements), the connection this
    function already opened is explicitly closed before the error
    propagates (PR #197 review) -- otherwise a caller whose `try` only
    wraps the call to `_connect()` itself (as `record_memory_event` below
    does) would never get a handle back to close, leaking the open
    connection/file descriptor on every failed call.
    """
    db_path = resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                point_id TEXT,
                repo TEXT,
                kind TEXT,
                summary_created_at REAL,
                event_timestamp TEXT NOT NULL,
                turn INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                project TEXT
            )
            """
        )
        # Supports the "most-recalled memories"/"memories never recalled"
        # rollups the issue describes (GROUP BY point_id) and per-project/repo
        # aggregation -- created eagerly rather than waiting for a later
        # surfacing feature to add them, since ALTER-ing indexes onto a table
        # that may already hold rows is strictly more disruptive than creating
        # them alongside the table from day one.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_events_point_id ON memory_events (point_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_events_repo ON memory_events (repo)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_events_event_type ON memory_events (event_type)")
    except sqlite3.Error:
        conn.close()
        raise
    return conn


def record_memory_event(
    event_type: str,
    point_id: Optional[str],
    repo: Optional[str],
    kind: Optional[str],
    summary_created_at: Optional[float],
    session_id: str,
    project: Optional[str],
    turn: int,
) -> None:
    """
    Append one row. Raises ValueError for an event_type outside
    ("recall", "remember") -- that's always a caller bug (this module is
    only ever invoked by this repo's own two call sites), not a runtime
    condition worth failing open over.

    Everything else fails OPEN: a write failure here (unwritable/locked/
    corrupt db file) is caught and logged to stderr, never raised -- by the
    time this runs, the actual `remember`/`recall` operation this event
    describes has already succeeded, and a broken passive usage log must
    never turn that into a reported tool failure. Same fail-open philosophy
    as `libs/qdrant_collection_hints.py`'s `set_cached_description`.
    """
    if event_type not in _VALID_EVENT_TYPES:
        raise ValueError(f"Unknown event_type {event_type!r}. Valid values: {_VALID_EVENT_TYPES}.")
    # `conn` is closed in a `finally` (PR #197 review), not just after a
    # successful `with conn:` block -- `with conn:` only manages the
    # transaction (commit/rollback), it does NOT close the connection
    # itself, and a failure INSIDE that block (e.g. "database is locked"
    # mid-insert) re-raises past a bare post-block conn.close() line,
    # leaking the connection exactly like an unclosed _connect() failure
    # would.
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        with conn:
            conn.execute(
                "INSERT INTO memory_events "
                "(event_type, point_id, repo, kind, summary_created_at, event_timestamp, turn, session_id, project) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_type,
                    str(point_id) if point_id is not None else None,
                    repo,
                    kind,
                    summary_created_at,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    turn,
                    session_id,
                    project,
                ),
            )
    except (sqlite3.Error, OSError) as e:
        print(f"[claude-runway] could not log memory event ({event_type}): {e}", file=sys.stderr)
    finally:
        if conn is not None:
            conn.close()
