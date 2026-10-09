"""
Storage + formatting for ClaudeRunway's opt-in savings tracker. Used by:
  - hooks/compress_output.py   (sole ledger WRITER -- see below for why)
  - hooks/session_end_savings.py    (rolls a session's ledger into SQLite, emits a summary)
  - compress_mcp_server.py          (writes the fixed schema-overhead estimate at startup;
                                      the 3 credited tools themselves do NOT import this module --
                                      they only need estimate_tokens, see their footer helper)
  - skills/my-savings/SKILL.md      (reads, via the savings_summary/savings_detail MCP tools)

This tracker estimates a COUNTERFACTUAL online (what compression avoided vs. what
raw content would have cost), which is categorically different from EVALUATION.md's
measured A/B delta -- the two numbers must never be conflated. See EVALUATION.md's
"Track D" section.

Two-tier storage:
  Tier 1 (transient): one JSONL file per session, appended to DURING the session
    by the PostToolUse hook every time it observes a credited compression.
  Tier 2 (perpetual): one SQLite row per session, written at SessionEnd by rolling
    up that session's JSONL. This is the single, central, cross-project store --
    required because the "across projects" comparison view can't work against
    separate per-repo databases.

All storage access goes through the functions in this module -- callers never
touch sqlite3 or file paths directly. SQLite is the current backend "for now"
(explicit user requirement for future flexibility): swapping to Postgres/CSV/a
team API later only requires changing the internals of this one module.

DB location resolution (see resolve_db_path()): CLAUDE_RUNWAY_SAVINGS_DB env var
(absolute path) if set, else ~/.claude/claude-runway/savings.db. The default lives
outside any git repo (no accidental-commit risk) and is stable across reinstalls/
moves of the tools repo itself. If overridden, the SAME value must be set in both
mcp.json (server env) and exported in the shell environment that launches `claude`
(hook entries in settings.json have no env field of their own), or producers
disagree about where the DB lives -- same coordination class as this repo's
QDRANT_URL note.
"""

import csv
import io
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

# Bare-name import, same sibling-libs-module convention
# libs/memory_bank_lib.py already uses for libs/qdrant_model_check.py --
# both files live in libs/, which is already on sys.path by the time either
# is imported (every caller inserts libs/ before importing anything from
# it). Used by current_session_id() below (issue #213) -- see that
# function's docstring for exactly what is and isn't delegated.
import session_id_lib


def tracking_enabled() -> bool:
    return os.environ.get("CLAUDE_RUNWAY_TRACK_SAVINGS", "").strip().lower() in ("1", "true", "yes")


def resolve_db_path() -> Path:
    override = os.environ.get("CLAUDE_RUNWAY_SAVINGS_DB")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            # The documented contract is an absolute path, but nothing enforced
            # that -- a relative override would resolve against whichever
            # process's cwd happened to be current (MCP server, hook, and skill
            # invocations don't share one), silently pointing different
            # processes at different files and breaking the "one central DB"
            # guarantee this tracker depends on. Anchor to a fixed location
            # (home directory, NOT cwd) instead of leaving it ambiguous, and
            # instead of raising: this function is called from deep inside
            # hook logic on ordinary Bash/Grep/etc. calls, where an exception
            # would break normal tool use over a savings-tracker misconfiguration.
            path = Path.home() / path
        return path
    return Path.home() / ".claude" / "claude-runway" / "savings.db"


def _sessions_dir() -> Path:
    # Deliberately NOT delegated to session_id_lib._sessions_dir() (issue
    # #213) even though the two resolve to the identical default path
    # (Path.home()/".claude"/"claude-runway"/"sessions") -- session_id_lib's
    # own module docstring frames that overlap as "a directory-CONVENTION
    # match, not a code dependency" precisely so each module can keep
    # resolving it independently. This one MUST stay derived from
    # resolve_db_path(), since record_event()/read_session_events()/
    # finalize_session() all rely on it moving together with a
    # CLAUDE_RUNWAY_SAVINGS_DB override (see that env var's docs) -- session_id_lib's
    # own sessions dir is intentionally fixed and never honors that override.
    return resolve_db_path().parent / "sessions"


def _sanitize_session_id(session_id: str) -> str:
    """
    session_id ends up directly in a filename below -- if it ever contained a
    path separator or ".." (a malformed hook payload, a manual/adversarial
    invocation), the resulting path could escape _sessions_dir() entirely
    (confirmed directly: a session_id of "../../../../tmp/evil-ledger" wrote
    outside the sessions directory). Replacing anything that isn't a-z/A-Z/
    0-9/hyphen/underscore keeps this confined regardless of what session_id
    turns out to be, without changing behavior for the UUID-like IDs Claude
    Code actually sends.
    """
    import re as _re
    return _re.sub(r"[^A-Za-z0-9_-]", "_", session_id)


def _session_jsonl_path(session_id: str) -> Path:
    return _sessions_dir() / f"{_sanitize_session_id(session_id)}.jsonl"


def project_name_from_cwd(cwd: str) -> str:
    return Path(cwd).name if cwd else "unknown"


def current_session_id(project: Optional[str] = None) -> Optional[str]:
    """
    Best-effort guess at the currently-active session, for on-demand /my-savings
    calls made mid-session. Unlike a hook, an MCP tool call has no direct access
    to Claude Code's session_id.

    Issue #213: when `project` is given, this delegates entirely to
    `session_id_lib.session_id(SessionIdStrategy.SHADOW_FILE, project=project)`
    -- scanning the CORE `hooks/record_session_id.py` hook's own
    always-populated marker files, rather than this module's own transient
    event-log JSONLs (which only exist once CLAUDE_RUNWAY_TRACK_SAVINGS is on
    AND at least one compression has actually happened this session). This is
    a strict improvement over the old project-filtered mechanism (finds a
    live session even when nothing has been compressed yet, or when the
    savings tracker is off entirely) while preserving the exact behavior
    issue #35 fixed: a live session in a DIFFERENT project is never
    substituted just because its marker happens to be the most recently
    touched file -- no match still means None, not someone else's session.

    When `project` is omitted, this KEEPS the old, independent mechanism
    (scanning this module's own `_sessions_dir()` for the most-recently-
    modified `*.jsonl`, returning its stem) rather than also delegating --
    session_id_lib's own docstring is explicit that SHADOW_FILE's `project=None`
    means "no marker can match" (a marker's basename is never `None`), NOT
    "most recent regardless of project," and that a caller wanting that
    different, unfiltered meaning "must implement that itself." Implementing
    it here, unchanged, is exactly that.

    Known limitation introduced by this split: if `CLAUDE_RUNWAY_SAVINGS_DB`
    is overridden away from its default location, the project-FILTERED path
    above targets session_id_lib's own fixed marker directory
    (~/.claude/claude-runway/sessions, which never honors that override) while
    the UNFILTERED path below still targets this module's own (overridden)
    `_sessions_dir()` -- the two can only diverge in that non-default
    configuration. In the common, unset-override case both directories are
    identical, so this is a documented edge case, not a real-world regression
    for the default install.

    Still only a best-effort guess when multiple live sessions share the SAME
    project (e.g. two windows open on the same repo) -- picks whichever of
    those was modified most recently. That ambiguity is out of scope here;
    only the cross-project mislabeling is what issue #35's filter fixes.
    """
    if project is not None:
        return session_id_lib.session_id(
            session_id_lib.SessionIdStrategy.SHADOW_FILE, project=project
        )

    d = _sessions_dir()
    if not d.is_dir():
        return None
    # Read each file's mtime individually rather than inside the sort key --
    # a concurrent SessionEnd hook (a different session, possibly a different
    # project, sharing this same central sessions dir) can delete a JSONL
    # between glob()'s enumeration and a later stat() call. Skipping files
    # that vanish/become unreadable mid-scan keeps this best-effort lookup
    # from raising over a race that has nothing to do with the caller's own
    # session.
    dated = []
    for p in d.glob("*.jsonl"):
        try:
            dated.append((p.stat().st_mtime, p))
        except OSError:
            continue
    if not dated:
        return None
    dated.sort(key=lambda pair: pair[0], reverse=True)
    return dated[0][1].stem


# ---------------------------------------------------------------------------
# Tier 1: transient per-session JSONL ledger
# ---------------------------------------------------------------------------

def record_event(session_id: str, tool: str, raw_tokens: int, out_tokens: int, credited: bool, source: str = "", project: Optional[str] = None,
                agent_id: Optional[str] = None, agent_type: Optional[str] = None) -> None:
    """
    Append one compression event to the session's transient JSONL ledger.
    saved_tokens is forced to 0 for uncredited events (e.g. fetch_url, whose
    honest counterfactual -- WebFetch's own server-side summary -- we don't
    observe) so an uncredited event can never contribute to the headline sum.

    `project` was originally what current_session_id()'s optional project
    filter matched against (issue #35); as of issue #213, that filtered
    lookup delegates to session_id_lib's shadow markers instead and no
    longer reads this per-event field at all -- it's retained here purely
    as informational/debugging metadata on each transient event, not
    because anything still resolves a session_id from it. The sole writer,
    hooks/compress_output.py, derives it from the hook payload's own
    `cwd` field (the same source session_end_savings.py already uses via
    project_name_from_cwd).

    `agent_id`/`agent_type` (issue #365) identify WHICH agent earned this
    saving. A tool call made inside a subagent reports the PARENT's
    session_id, so without these an event is indistinguishable from a
    main-session one. They're written only when non-empty, so every
    main-session event -- and every ledger written before this existed -- keeps
    its exact previous shape. Note `agent_type` alone does NOT mean "subagent":
    the hooks docs say it's also present on the main thread when the session
    was started with `claude --agent <name>`; only `agent_id` marks a subagent
    (see _aggregate_by_agent). `agent_id` lives in this transient ledger only --
    the SQLite rollup keeps just the subagent flag and `agent_type`.
    """
    saved_tokens = max(0, raw_tokens - out_tokens) if credited else 0
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": tool,
        "raw_tokens": raw_tokens,
        "out_tokens": out_tokens,
        "saved_tokens": saved_tokens,
        "credited": credited,
        "source": source[:200] if source else "",
        "project": project,
    }
    if agent_id:
        entry["agent_id"] = str(agent_id)[:200]
    if agent_type:
        entry["agent_type"] = str(agent_type)[:200]
    path = _session_jsonl_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit UTF-8 rather than the platform default -- json.dumps() already
    # escapes non-ASCII to \uXXXX by default, so this file's actual bytes are
    # always pure ASCII regardless of encoding today, but declaring it
    # explicitly is still correct practice and future-proofs against ever
    # writing raw non-ASCII (e.g. if ensure_ascii=False were added later).
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_session_events(session_id: str) -> list:
    # No separate exists() check -- that's a check-then-act race: a concurrent
    # SessionEnd hook can finalize and delete this exact session's JSONL
    # between the check and the open (confirmed directly). Opening straight
    # away and catching OSError closes that window instead of narrowing it.
    path = _session_jsonl_path(session_id)
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return []
    with f:
        return _parse_event_lines(f)


def _parse_event_lines(lines) -> list:
    """Shared by read_session_events() and recover_orphaned_sessions() (issue
    #306), which parses a claimed ledger under a different filename."""
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a corrupted line shouldn't sink the whole session's tally
    return events


# ---------------------------------------------------------------------------
# Tier 2: perpetual SQLite store
# ---------------------------------------------------------------------------

# Bumped whenever the shape of ANY persistent table changes: `sessions`,
# `session_tools`, `session_agents` (added in v5, #365) or `meta`. That includes
# adding a new table, even though _connect creates it for fresh DBs via CREATE
# TABLE IF NOT EXISTS -- the bump plus a registered migration is what keeps
# existing databases on the same versioned path as new ones.
# _MIGRATIONS below must gain a matching entry for every bump -- see
# _run_migrations' docstring for how the two stay in sync (issue #30/#69).
SCHEMA_VERSION = 5

# Issue #365: per-session savings split by which agent earned them. One DDL
# string shared by _connect (fresh DBs) and _migrate_to_v5 (existing DBs) so the
# two can't drift. `is_subagent` is part of the key so a `claude --agent X` main
# session and a subagent of type X can never collide into one row; '' in
# agent_type means "no agent_type reported". The raw agent_id is deliberately NOT
# stored here -- one row per spawned subagent would grow without bound for little
# analytical value; type + flag answers "main or subagent, and which kind".
_SESSION_AGENTS_DDL = """
    CREATE TABLE IF NOT EXISTS session_agents (
        session_id TEXT NOT NULL,
        is_subagent INTEGER NOT NULL,
        agent_type TEXT NOT NULL DEFAULT '',
        event_count INTEGER NOT NULL,
        saved_tokens INTEGER NOT NULL,
        raw_tokens_sum INTEGER NOT NULL DEFAULT 0,
        out_tokens_sum INTEGER NOT NULL DEFAULT 0,
        credited_event_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (session_id, is_subagent, agent_type)
    )
"""


def _migrate_to_v1(conn) -> None:
    """
    Backfills raw_tokens_sum/out_tokens_sum/overhead_tokens onto a `sessions`
    table that predates them (issue #30) -- these three shipped with
    DEFAULT 0 (unlike this table's other NOT NULL columns, which have none)
    specifically so ALTER TABLE ADD COLUMN can backfill existing rows with a
    real, honest value (0 -- "unknown," not a fabricated historical number)
    instead of requiring a destructive rebuild.

    Checks actual column presence via PRAGMA table_info rather than assuming
    "not yet migrated" means "column is missing": a `sessions` table created
    by this file's OWN CREATE TABLE statement already has all three columns
    (they're baked into it for a brand-new DB), so it can reach here with
    user_version still 0 (never previously versioned, e.g. a DB created by
    an already-shipped build of this file from before this migration
    scaffold existed) yet nothing actually missing to add. Re-running
    ALTER TABLE ADD COLUMN on an existing column is a hard sqlite error, not
    a no-op, so this check is required for correctness, not just caution.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    for column in ("raw_tokens_sum", "out_tokens_sum", "overhead_tokens"):
        if column not in existing:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")


def _migrate_to_v2(conn) -> None:
    """
    Adds raw_tokens_sum to `session_tools` so per-tool average reduction %
    can be computed (issue #65). DEFAULT 0 lets existing rows get backfilled
    with an honest "unknown" rather than requiring a destructive rebuild --
    the same pattern as _migrate_to_v1's columns on `sessions`.

    Checks actual column presence first so a DB created by THIS file's own
    CREATE TABLE (which already has raw_tokens_sum baked in) doesn't raise
    "duplicate column name" if it somehow reaches here with user_version < 2.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(session_tools)").fetchall()}
    if "raw_tokens_sum" not in existing:
        conn.execute("ALTER TABLE session_tools ADD COLUMN raw_tokens_sum INTEGER NOT NULL DEFAULT 0")


def _migrate_to_v3(conn) -> None:
    """
    Adds out_tokens_sum and credited_event_count to `session_tools` (issue #65
    follow-up after Copilot review).

    out_tokens_sum: actual compressed-output token total for credited events --
      necessary because record_event clamps saved_tokens to max(0, raw-out),
      so computing out as raw - saved_tokens reconstructs the wrong value when
      a summary expands (raw=100, out=150 -> saved=0, reconstructed out=100,
      actual out=150). Storing the real out makes the per-tool avg reduction %
      consistent with the session-level pct that already uses the real out_sum.

    credited_event_count: number of credited events for this tool -- lets
      format_detail_view distinguish a credited tool whose compression saved
      zero tokens (show "0% reduction") from a genuinely uncredited tool like
      fetch_url (show "not credited"), without relying on saved_tokens > 0.

    Both default to 0 so existing rows are backfilled with a safe, honest value.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(session_tools)").fetchall()}
    if "out_tokens_sum" not in existing:
        conn.execute("ALTER TABLE session_tools ADD COLUMN out_tokens_sum INTEGER NOT NULL DEFAULT 0")
    if "credited_event_count" not in existing:
        conn.execute("ALTER TABLE session_tools ADD COLUMN credited_event_count INTEGER NOT NULL DEFAULT 0")


def _migrate_to_v4(conn) -> None:
    """
    Adds actual per-turn token counts from the transcript_path stopgap
    (issue #165) to `sessions`: the real tokens Anthropic processed this
    session, broken out by type. Distinct from the compression heuristic
    estimates in raw_tokens_sum/out_tokens_sum. DEFAULT 0 so existing rows
    backfill safely and the prior rows never report fabricated non-zero counts.
    # STOPGAP: remove when #164 is resolved (Stop hook will expose these directly).
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    for column in (
        "actual_input_tokens",
        "actual_output_tokens",
        "actual_cache_read_tokens",
        "actual_cache_write_tokens",
    ):
        if column not in existing:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")


def _migrate_to_v5(conn) -> None:
    """
    Adds the `session_agents` table (issue #365): savings split by main session
    vs. subagent, and by agent_type. Purely additive and idempotent
    (CREATE TABLE IF NOT EXISTS) -- _connect already creates it before migrations
    run, so this mostly exists so SCHEMA_VERSION and _MIGRATIONS stay in lockstep
    (the guard in _run_migrations requires an entry for every version). Sessions
    finalized before this version have no rows here: the agent information was
    never recorded, so nothing is backfilled.
    """
    conn.execute(_SESSION_AGENTS_DDL)


# Maps target schema version -> the function that migrates INTO it from the
# version immediately before. The next schema change adds one more entry here
# and bumps SCHEMA_VERSION -- _run_migrations doesn't need to change at all,
# which is the actual "scaffold" issue #69 asked for: a repeatable shape for
# schema changes, not a one-off patch for this specific set of columns.
_MIGRATIONS = {1: _migrate_to_v1, 2: _migrate_to_v2, 3: _migrate_to_v3, 4: _migrate_to_v4, 5: _migrate_to_v5}


def _table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone() is not None


def _run_migrations(conn, is_new_db: bool) -> None:
    """
    Brings the DB's PRAGMA user_version up to SCHEMA_VERSION. Two real bugs
    found in PR #94 review (Copilot), both fixed here rather than left as
    "works today, breaks on the next migration":

    1. A brand-new DB's CREATE TABLE (run by _connect just before this)
       already produces the LATEST schema -- there's nothing to migrate.
       The original version instead always started at user_version 0 and
       replayed every historical migration, relying on EACH migration
       independently noticing its target columns already exist (the way
       _migrate_to_v1 happens to). That's not a real scaffold -- it silently
       requires every future migration author to remember the same
       defensive check, and a plain `ALTER TABLE ADD COLUMN` (the normal,
       expected shape for a future migration) would collide with the column
       CREATE TABLE already added for a fresh install, breaking every new
       user's first run. `is_new_db` (computed by _connect from whether
       `sessions` existed BEFORE its CREATE TABLE ran) lets a fresh DB skip
       straight to SCHEMA_VERSION instead.

    2. The version check, the migration(s), and recording the new version
       were three separate auto-committing statements with no lock held
       across them. Two processes calling _connect() concurrently against
       the SAME pre-existing v0 savings.db (a real scenario: this is a
       single central DB and the SessionEnd hook can fire from multiple
       Claude Code sessions/projects at once) could both read "column
       missing" before either had committed its own ALTER TABLE, then both
       attempt to add it -- the second raising "duplicate column name",
       silently swallowed by the SessionEnd hook's blanket except, losing
       that session's roll-up entirely. BEGIN IMMEDIATE acquires SQLite's
       write lock up front (verified directly: a second connection's own
       BEGIN IMMEDIATE against the same file correctly blocks/errors rather
       than interleaving), serializing the whole check-migrate-record
       sequence against any concurrent caller. SQLite's DDL is itself
       transactional (verified directly: ALTER TABLE inside an open
       transaction cleanly reverts on ROLLBACK), so a failure partway
       through -- including case 3 below -- rolls back completely instead
       of leaving the schema half-migrated.

    3. `_MIGRATIONS.get(version)` let a version with no registered migration
       silently advance user_version anyway, permanently recording a schema
       that was never actually migrated. Missing entries now raise instead.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        if is_new_db:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        else:
            current_version = conn.execute("PRAGMA user_version").fetchone()[0]
            for version in range(current_version + 1, SCHEMA_VERSION + 1):
                if version not in _MIGRATIONS:
                    raise RuntimeError(
                        f"savings_ledger: SCHEMA_VERSION is {SCHEMA_VERSION} but no migration "
                        f"is registered for version {version} in _MIGRATIONS -- add one."
                    )
                _MIGRATIONS[version](conn)
                # f-string, not a `?` placeholder -- PRAGMA doesn't accept
                # bound parameters for its value. Safe here regardless:
                # `version` is this loop's own int, never external input.
                conn.execute(f"PRAGMA user_version = {version}")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _connect():
    import sqlite3
    db_path = resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_schema(conn)
    except Exception:
        # Close before re-raising: a failed schema/migration step (a version
        # with no registered migration, or a migration that raises partway
        # through) would otherwise leave this connection open, kept alive
        # only by the propagating exception's traceback, so the db file
        # stays locked until that traceback is released. Harmless on POSIX,
        # but on Windows an open handle makes the file undeletable, which is
        # how this was found (a test's TemporaryDirectory cleanup raised
        # WinError 32). memory_events_lib._connect already does the same for
        # the same reason.
        conn.close()
        raise
    return conn


def _ensure_schema(conn) -> None:
    """
    Creates any missing tables on an already-open connection and brings its
    schema up to SCHEMA_VERSION. Split out of _connect() so that function
    can close the connection if any step here raises.
    """
    # Checked BEFORE the CREATE TABLE calls below, specifically so
    # _run_migrations can tell a genuinely fresh DB (nothing to migrate --
    # see its docstring) apart from a pre-existing one that needs its
    # history replayed. `sessions` specifically, not the other three tables
    # (`session_tools`/`session_agents`/`meta`), since it's the one this
    # migration scaffold actually versions.
    sessions_existed = _table_exists(conn, "sessions")
    # Always the LATEST schema -- a brand-new DB gets every current column
    # in one step and never needs a migration to add anything for it.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            credited_saved_tokens INTEGER NOT NULL,
            raw_tokens_sum INTEGER NOT NULL DEFAULT 0,
            out_tokens_sum INTEGER NOT NULL DEFAULT 0,
            event_count INTEGER NOT NULL,
            fetch_url_count INTEGER NOT NULL,
            overhead_tokens INTEGER NOT NULL DEFAULT 0,
            actual_input_tokens INTEGER NOT NULL DEFAULT 0,
            actual_output_tokens INTEGER NOT NULL DEFAULT 0,
            actual_cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            actual_cache_write_tokens INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_tools (
            session_id TEXT NOT NULL,
            tool TEXT NOT NULL,
            event_count INTEGER NOT NULL,
            saved_tokens INTEGER NOT NULL,
            raw_tokens_sum INTEGER NOT NULL DEFAULT 0,
            out_tokens_sum INTEGER NOT NULL DEFAULT 0,
            credited_event_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (session_id, tool)
        )
        """
    )
    conn.execute(_SESSION_AGENTS_DDL)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    # Only after all four tables are guaranteed to exist -- a migration may
    # assume `sessions`/`session_tools`/`session_agents`/`meta` are all already there.
    _run_migrations(conn, is_new_db=not sessions_existed)


def set_meta(key: str, value: str) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
    conn.close()


def get_meta(key: str, default: Optional[str] = None) -> Optional[str]:
    conn = _connect()
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else default


def get_schema_overhead_tokens() -> int:
    """
    The fixed per-turn schema-overhead estimate, computed live at
    compress_mcp_server.py startup (see its _log_fixed_overhead) and stashed
    here so hooks/skills in a different process can read the same number
    without recomputing it or importing the MCP server module.
    """
    return int(get_meta("schema_overhead_tokens", "0") or "0")


def _aggregate_by_agent(events: list) -> list:
    """
    Group a session's events by which agent earned them (issue #365). Returns a
    list of dicts, one per (is_subagent, agent_type), sorted by saved_tokens
    descending then deterministically, with the same counters as by_tool and
    the same credited-only rules (saved/raw/out sums and credited_event_count
    only count credited events; event_count counts all of them).

    `is_subagent` is decided by `agent_id` alone. `agent_type` is just a label:
    the hooks docs say it's also present on the MAIN thread when the session
    was started with `claude --agent <name>`, so keying "subagent" off
    agent_type would misfile that whole session's savings as subagent work. An
    event with neither field (every event recorded before #365, and every
    plain main-session event) lands in the (0, '') row.
    """
    groups: dict[tuple[int, str], dict[str, int]] = {}
    for e in events:
        key = (1 if e.get("agent_id") else 0, str(e.get("agent_type") or ""))
        agg = groups.setdefault(key, {
            "event_count": 0,
            "saved_tokens": 0,
            "raw_tokens_sum": 0,
            "out_tokens_sum": 0,
            "credited_event_count": 0,
        })
        agg["event_count"] += 1
        if e.get("credited"):
            agg["saved_tokens"] += e.get("saved_tokens", 0)
            agg["raw_tokens_sum"] += e.get("raw_tokens", 0)
            agg["out_tokens_sum"] += e.get("out_tokens", 0)
            agg["credited_event_count"] += 1
    rows = [{"is_subagent": k[0], "agent_type": k[1], **v} for k, v in groups.items()]
    rows.sort(key=lambda r: (-r["saved_tokens"], r["is_subagent"], r["agent_type"]))
    return rows


def _write_session_rows(conn, session_id: str, project: str, events: list, overhead_tokens: int,
                        actual_tokens: Optional[dict], ended_at: str, accumulate: bool = False) -> dict:
    """
    Aggregate `events` and write them as `session_id`'s rows on `conn`, inside
    whatever transaction the caller holds. Shared by finalize_session() and
    finalize_owned_session() so the two can't drift. Returns the aggregate of
    `events` alone (what the SessionEnd summary reports as "this session").

    accumulate=False (finalize_session): one `sessions` row INSERT OR REPLACE,
    `session_tools`/`session_agents` delete-then-insert -- the row is exactly
    these events.

    accumulate=True (finalize_owned_session, issue #306 / PR #412 round 5):
    these events are ADDED to whatever the session already has. A session's
    events can reach savings.db in more than one roll-up -- recovered as an
    orphan, then resumed and ended cleanly; or ended, resumed, ended again --
    and replacing threw away every earlier part. With no existing row this
    writes exactly what the replace path writes, so a session that ends once
    is unchanged. Counters add; project/ended_at/overhead take the newest
    roll-up's value; actual_* take the new transcript counts when there are
    any (a resumed session reuses its transcript, so a later parse covers the
    whole session) and otherwise keep the counts already recorded. Adding is
    only correct if no event is ever rolled up twice -- finalize_owned_session
    guarantees that, see its docstring.
    """
    credited = [e for e in events if e.get("credited")]
    credited_saved = sum(e.get("saved_tokens", 0) for e in credited)
    raw_sum = sum(e.get("raw_tokens", 0) for e in credited)
    out_sum = sum(e.get("out_tokens", 0) for e in credited)
    # Credited-only, not len(events): the reduction % (and raw/out sums above)
    # are computed from credited events only, so the "N events" figure shown
    # alongside them must match that same subset or the two numbers disagree
    # (e.g. "3 events, 90% reduction" where one of the 3 didn't count toward
    # the 90% at all). Uncredited activity (fetch_url) is still surfaced,
    # just via fetch_url_count separately, not folded into this count.
    event_count = len(credited)
    fetch_url_count = sum(1 for e in events if e.get("tool") == "fetch_url")

    by_tool: dict[str, dict[str, int]] = {}
    for e in events:
        tool = e.get("tool", "unknown")
        agg = by_tool.setdefault(tool, {
            "event_count": 0,
            "saved_tokens": 0,
            "raw_tokens_sum": 0,
            "out_tokens_sum": 0,
            "credited_event_count": 0,
        })
        agg["event_count"] += 1
        if e.get("credited"):
            agg["saved_tokens"] += e.get("saved_tokens", 0)
            agg["raw_tokens_sum"] += e.get("raw_tokens", 0)
            # Actual out_tokens (pre-clamping) for accurate avg % computation --
            # see _migrate_to_v3's docstring for why raw - saved_tokens is wrong.
            agg["out_tokens_sum"] += e.get("out_tokens", 0)
            agg["credited_event_count"] += 1

    by_agent = _aggregate_by_agent(events)

    _at = actual_tokens or {}
    actual_inp  = int(_at.get("input",      0) or 0)
    actual_out  = int(_at.get("output",     0) or 0)
    actual_cr   = int(_at.get("cache_read", 0) or 0)
    actual_cw   = int(_at.get("cache_write",0) or 0)

    session_values = (session_id, project, ended_at, credited_saved, raw_sum, out_sum, event_count,
                      fetch_url_count, overhead_tokens, actual_inp, actual_out, actual_cr, actual_cw)
    session_columns = (
        "(session_id, project, ended_at, credited_saved_tokens, raw_tokens_sum, out_tokens_sum, "
        " event_count, fetch_url_count, overhead_tokens, "
        " actual_input_tokens, actual_output_tokens, actual_cache_read_tokens, actual_cache_write_tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    # Per-key counter columns shared by session_tools and session_agents.
    counters = ("event_count", "saved_tokens", "raw_tokens_sum", "out_tokens_sum", "credited_event_count")
    add_counters = ", ".join(f"{c} = {c} + excluded.{c}" for c in counters)
    if accumulate:
        # SQLite UPSERT (3.24+): with no existing row this is a plain INSERT
        # of the same values the replace path writes.
        keep_actual = "excluded.{0}" if actual_tokens else "{0}"
        actual_cols = ("actual_input_tokens", "actual_output_tokens",
                       "actual_cache_read_tokens", "actual_cache_write_tokens")
        conn.execute(
            "INSERT INTO sessions " + session_columns + " ON CONFLICT(session_id) DO UPDATE SET "
            "project = excluded.project, ended_at = excluded.ended_at, "
            "credited_saved_tokens = credited_saved_tokens + excluded.credited_saved_tokens, "
            "raw_tokens_sum = raw_tokens_sum + excluded.raw_tokens_sum, "
            "out_tokens_sum = out_tokens_sum + excluded.out_tokens_sum, "
            "event_count = event_count + excluded.event_count, "
            "fetch_url_count = fetch_url_count + excluded.fetch_url_count, "
            "overhead_tokens = excluded.overhead_tokens, "
            + ", ".join(f"{c} = {keep_actual.format(c)}" for c in actual_cols),
            session_values,
        )
        tools_conflict = f" ON CONFLICT(session_id, tool) DO UPDATE SET {add_counters}"
        agents_conflict = f" ON CONFLICT(session_id, is_subagent, agent_type) DO UPDATE SET {add_counters}"
    else:
        conn.execute("INSERT OR REPLACE INTO sessions " + session_columns, session_values)
        conn.execute("DELETE FROM session_tools WHERE session_id = ?", (session_id,))
        # Same replace-then-insert shape as session_tools (issue #365), so
        # re-finalizing a session can never leave stale per-agent rows behind.
        conn.execute("DELETE FROM session_agents WHERE session_id = ?", (session_id,))
        tools_conflict = agents_conflict = ""
    for tool, agg in by_tool.items():
        conn.execute(
            "INSERT INTO session_tools (session_id, tool, event_count, saved_tokens, "
            "raw_tokens_sum, out_tokens_sum, credited_event_count) VALUES (?, ?, ?, ?, ?, ?, ?)"
            + tools_conflict,
            (session_id, tool, agg["event_count"], agg["saved_tokens"],
             agg["raw_tokens_sum"], agg["out_tokens_sum"], agg["credited_event_count"]),
        )
    for row in by_agent:
        conn.execute(
            "INSERT INTO session_agents (session_id, is_subagent, agent_type, event_count, saved_tokens, "
            "raw_tokens_sum, out_tokens_sum, credited_event_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            + agents_conflict,
            (session_id, row["is_subagent"], row["agent_type"], row["event_count"], row["saved_tokens"],
             row["raw_tokens_sum"], row["out_tokens_sum"], row["credited_event_count"]),
        )

    return {
        "session_id": session_id,
        "project": project,
        "credited_saved_tokens": credited_saved,
        "raw_tokens_sum": raw_sum,
        "out_tokens_sum": out_sum,
        "event_count": event_count,
        "fetch_url_count": fetch_url_count,
        "overhead_tokens": overhead_tokens,
        "by_tool": by_tool,
        "by_agent": by_agent,
        "actual_input_tokens":      actual_inp,
        "actual_output_tokens":     actual_out,
        "actual_cache_read_tokens": actual_cr,
        "actual_cache_write_tokens": actual_cw,
    }


def finalize_session(session_id: str, project: str, overhead_tokens: int = 0, delete_jsonl: bool = True, actual_tokens: Optional[dict] = None,
                     ended_at: Optional[str] = None, events: Optional[list] = None) -> dict:
    """
    Aggregate a session's transient JSONL into one SQLite row (+ per-tool
    rows), then delete the JSONL (its job is done once rolled up). Returns
    the aggregate dict, used both for the DB write and to format the
    SessionEnd summary directly without a second read.

    The SessionEnd hook and orphan recovery use finalize_owned_session()
    instead (issue #306), which adds the cross-process ownership this simple
    read-write-delete doesn't have; this stays the plain primitive.

    actual_tokens: optional dict from parse_transcript_token_counts() with
    keys input/output/cache_read/cache_write -- the real per-turn token counts
    Anthropic processed this session. When None (opt-in feature disabled or
    transcript parse failed), the four actual_* columns stay 0.

    ended_at: optional "%Y-%m-%dT%H:%M:%SZ" timestamp to record instead of
    "now". events: optional, already-read events to aggregate instead of
    re-reading the ledger.
    """
    if events is None:
        events = read_session_events(session_id)
    if ended_at is None:
        ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = _connect()
    with conn:
        result = _write_session_rows(conn, session_id, project, events, overhead_tokens, actual_tokens, ended_at)
    conn.close()

    if delete_jsonl:
        try:
            _session_jsonl_path(session_id).unlink(missing_ok=True)
        except OSError:
            pass

    return result


def _orphan_project(events: list) -> str:
    """
    The project to file a recovered orphan under (issue #306). An orphan has
    no SessionEnd payload, so there's no `cwd` to derive it from the way
    hooks/session_end_savings.py does -- but every event the sole writer
    (hooks/compress_output.py) records carries `project`, derived from that
    same `cwd` via project_name_from_cwd(), so it's the same value a graceful
    SessionEnd would have used. The most recent non-empty one wins; "unknown"
    (project_name_from_cwd's own fallback for a missing cwd) when none has it,
    e.g. a ledger written before the field existed (issue #35).

    "Most recent" is by each event's own `ts`, not list position (PR #412
    review): `events` can combine several artifacts of one session (a
    leftover claim plus the ledger written after it), and a resumed session
    can have moved to another project in between. `ts` is a fixed-width
    UTC "%Y-%m-%dT%H:%M:%SZ" string, so it sorts chronologically as text; an
    event without one sorts first. Ties (same second) fall back to list
    position, which finalize_owned_session orders oldest artifact first.
    """
    best: Optional[tuple] = None
    for i, e in enumerate(events):
        project = e.get("project") if isinstance(e, dict) else None
        if not (isinstance(project, str) and project):
            continue
        ts = e.get("ts")
        key = (ts if isinstance(ts, str) else "", i)
        if best is None or key > best[0]:
            best = (key, project)
    return best[1] if best else "unknown"


# Issue #306: while a session's events are being rolled up, its ledger is
# renamed to "<session_id>.jsonl.claimed-<hex>". That name never matches the
# "*.jsonl" glob current_session_id() scans, and a record_event() call after
# the rename starts a fresh ledger instead of writing into the file about to
# be deleted. A claim normally lives only inside finalize_owned_session(); one
# found later was left by a roll-up that failed or died, and is simply rolled
# up again together with the session's other artifacts.
_CLAIM_MARKER = ".claimed-"
_CLAIM_RE = re.compile(r"\A(.+)\.jsonl\.claimed-[0-9a-f]+\Z")


def _artifact_session_id(name: str) -> Optional[str]:
    """The session a sessions-dir file belongs to if it's a ledger or a claim
    of one; None for anything else, including session_id_lib's shadow markers
    (`session_*.jsonl`, same directory by default -- never a ledger)."""
    if name.startswith("session_"):
        return None
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")]
    m = _CLAIM_RE.match(name)
    return m.group(1) if m else None


def _session_artifacts(session_id: str) -> list:
    """Every file currently holding events for `session_id`: its ledger (if
    present) and any claims of it."""
    sid = _sanitize_session_id(session_id)
    d = _sessions_dir()
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob(f"{sid}.jsonl*") if _artifact_session_id(p.name) == sid)


def _mtime_or_zero(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _consumed_claim_key(claim: Path) -> str:
    """`meta` key recording that a claim's events are already in savings.db
    (see finalize_owned_session). Keyed by the claim's own unique filename."""
    return "consumed_claim:" + claim.name


def has_session_artifacts(session_id: str) -> bool:
    """True if `session_id` has anything left to roll up: its ledger, or a
    claim of it left by a roll-up that failed or died (issue #306). The
    SessionEnd hook's cheap "anything logged?" check -- checking the ledger
    alone would skip a session whose only artifact is such a claim."""
    return bool(_session_artifacts(session_id))


def finalize_owned_session(session_id: str, project: Optional[str] = None, overhead_tokens: int = 0,
                           actual_tokens: Optional[dict] = None,
                           stale_before: Optional[float] = None) -> Optional[dict]:
    """
    Roll ALL of a session's event artifacts (its ledger plus any leftover
    claims) into savings.db as ONE snapshot, then delete them -- the single
    ownership protocol both the SessionEnd hook and orphan recovery go
    through (issue #306). Returns the aggregate dict, {} if the artifacts held
    no readable events (they're still deleted; no row is written, matching
    the hook's "nothing logged -> nothing finalized" rule), or None if there
    was nothing to take or `stale_before` said the session is live after all.

    Why one protocol, and why this shape (each piece fixes a race found and
    reproduced in PR #412's review):
      - Ownership is SQLite's own write lock: everything below runs inside
        BEGIN IMMEDIATE on savings.db, so any two finalizers -- two
        recoveries, or a recovery and a session's own SessionEnd -- run one
        after the other, never interleaved. The second one finds the files
        already consumed and returns None. (Per-file rename claims alone let
        two workers each own part of one session and overwrite each other's
        subset; a re-read inside a plain finalize let a vanished file write
        an all-zero row.) It's the same lock _run_migrations already relies
        on, portable to Windows, and released by SQLite itself if the process
        dies -- no lock file that can be orphaned.
      - Events are ADDED to the session's existing row, never replace it
        (round 5): a session's events can arrive in several roll-ups --
        recovered as an orphan, then resumed and ended; or ended, resumed,
        ended again -- and replacing kept only the last one. A session that
        ends once writes exactly the row it always did (see
        _write_session_rows).
      - Adding requires exactly-once: every claim's unique name is recorded
        in `meta` in the same transaction that adds its events. A roll-up
        that committed but died before deleting its claims leaves them, and
        their keys, in place; the next owner sees the key, skips those
        events, and deletes the file. Keys are dropped once their files are
        verifiably gone, so `meta` stays bounded. No schema change: `meta` is
        the existing key/value table.
      - The ledger is renamed to a claim before it's read, so an event
        appended after that point lands in a fresh ledger rather than being
        deleted unread. Residual, documented rather than locked away: a
        record_event() that opened the ledger just before the rename and
        writes just after the read (microseconds). Closing that would need a
        lock in record_event(), which runs on every compressed tool call.

    project: the caller's project (the hook derives it from `cwd`); None means
    "an orphan -- take it from the events" (see _orphan_project).
    actual_tokens: the transcript counts for this session. Given, they
    replace the row's counts (a resumed session reuses its transcript, so the
    latest parse covers all of it); None keeps whatever counts the row
    already has (0 for a new row), since the row now spans every roll-up of
    the session.
    stale_before: recovery only -- an epoch time; if any artifact was written
    after it, the session is live again, so nothing is taken. Checked under
    the lock, closing the gap between recovery's directory scan and here.
    ended_at is "now" for a graceful end, and the newest artifact's last
    write for a recovery, so /my-savings trend buckets a session that
    crashed weeks ago into the week it actually ran.
    """
    sid = _sanitize_session_id(session_id)
    conn = _connect()
    claimed = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            artifacts = _session_artifacts(sid)
            mtimes = []
            for p in artifacts:
                try:
                    mtimes.append(p.stat().st_mtime)
                except OSError:
                    continue
            if not mtimes or (stale_before is not None and max(mtimes) > stale_before):
                conn.execute("ROLLBACK")
                return None
            for p in artifacts:
                if p.name.endswith(".jsonl"):
                    target = p.with_name(f"{p.name}{_CLAIM_MARKER}{uuid.uuid4().hex}")
                    try:
                        os.rename(p, target)
                    except FileNotFoundError:
                        continue
                    # Any other OSError (e.g. Windows refusing to rename a file
                    # another process holds open) propagates: nothing is
                    # written, and a later SessionEnd retries.
                    claimed.append(target)
                else:
                    claimed.append(p)
            # Oldest artifact first (rename keeps mtime), so events read below
            # are in roughly chronological order -- _orphan_project's tie-break
            # for events sharing a one-second `ts`.
            claimed.sort(key=_mtime_or_zero)
            # Exactly-once guard for accumulating: each claim's id is recorded
            # in `meta` in the SAME transaction that adds its events. A claim
            # whose id is already there was added by a roll-up that committed
            # but died before deleting it -- skip its events (they're in the
            # row), just delete the file below.
            events = []
            for p in claimed:
                key = _consumed_claim_key(p)
                if conn.execute("SELECT 1 FROM meta WHERE key = ?", (key,)).fetchone():
                    continue
                with open(p, encoding="utf-8", errors="replace") as f:
                    events.extend(_parse_event_lines(f))
                conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (key, sid))
            result: dict = {}
            if events:
                if stale_before is not None:
                    ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(mtimes)))
                else:
                    ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                result = _write_session_rows(
                    conn, sid, project or _orphan_project(events), events,
                    overhead_tokens, actual_tokens, ended_at, accumulate=True,
                )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    # Only after COMMIT: a crash before this point leaves the claims (and
    # their consumed-claim keys) for the next owner, who deletes them without
    # counting them again. Each key is dropped once its file is gone, so the
    # bookkeeping in `meta` stays bounded; a failure here only leaves a stale
    # key behind, which is harmless (claim names are random, never reused).
    gone = []
    for p in claimed:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
        # Drop the key only once the file is verifiably gone -- a key without
        # its file is harmless, a file without its key would count twice.
        if not p.exists():
            gone.append(_consumed_claim_key(p))
    if gone:
        try:
            conn = _connect()
            with conn:
                conn.executemany("DELETE FROM meta WHERE key = ?", [(k,) for k in gone])
            conn.close()
        except Exception:
            pass
    return result


def recover_orphaned_sessions(exclude_session_id: Optional[str] = None, now: Optional[float] = None) -> int:
    """
    Roll up, then delete, the event artifacts of every session that never
    reached a graceful SessionEnd -- a crash, a force-kill, a closed terminal
    (issue #306). The SessionEnd hook is otherwise the ONLY path from a ledger
    into savings.db, so without this an orphan's credited savings never reach
    /my-savings and its JSONL is never cleaned up. Returns how many sessions
    were consumed.

    "Orphaned" means none of the session's artifacts has been written for
    longer than the SAME TTL session_id_lib.sweep_stale_shadow_markers()
    applies to shadow markers (CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS, default
    168h), read through that module's own _ttl_hours() so one knob governs
    both sweeps and both get the same positive/NaN validation. A ledger is
    appended on every credited compression, so its mtime is "last activity":
    a week with none is this toolkit's existing convention for "treat as
    dead". Still a heuristic -- a live session idle longer than the TTL looks
    the same as a dead one, exactly as for shadow markers (#233); its
    recovered events are in savings.db either way, and a later finalize of
    the same id adds to that row (see finalize_owned_session()).

    This lives here as a SIBLING of session_id_lib's marker sweep rather than
    as part of it: session_id_lib deliberately never imports this module, and
    this module's sessions dir follows a CLAUDE_RUNWAY_SAVINGS_DB override
    while session_id_lib's does not (issue #408), so only this module knows
    where its own ledgers actually are.

    `exclude_session_id` is the caller's own session (the SessionEnd hook
    passes the session that is ending), which that hook finalizes itself --
    through the same finalize_owned_session(), so its own leftover claims are
    included there rather than here. All the concurrency handling lives in
    finalize_owned_session(); this is only the scan. Per-session failures are
    skipped, never raised, so one bad ledger can't block the rest.
    """
    now = time.time() if now is None else now
    stale_before = now - session_id_lib._ttl_hours() * 3600
    d = _sessions_dir()
    if not d.is_dir():
        return 0
    excluded = _sanitize_session_id(exclude_session_id) if exclude_session_id else None
    newest: dict[str, float] = {}
    for p in list(d.iterdir()):
        sid = _artifact_session_id(p.name)
        if sid is None or sid == excluded:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue  # vanished mid-scan
        newest[sid] = max(mtime, newest.get(sid, mtime))

    overhead: Optional[int] = None
    recovered = 0
    for sid, mtime in newest.items():
        if mtime > stale_before:
            continue  # live, or written to recently
        try:
            if overhead is None:
                overhead = get_schema_overhead_tokens()
            if finalize_owned_session(sid, overhead_tokens=overhead, stale_before=stale_before) is not None:
                recovered += 1
        except Exception:
            continue
    return recovered


def get_live_session_aggregate(session_id: str) -> dict:
    """
    Same shape as finalize_session's return, but read-only -- for /my-savings
    mid-session checks where the session hasn't ended yet, so nothing should
    be written to SQLite or have its JSONL deleted.
    """
    events = read_session_events(session_id)
    credited = [e for e in events if e.get("credited")]
    by_tool: dict[str, dict[str, int]] = {}
    for e in events:
        tool = e.get("tool", "unknown")
        agg = by_tool.setdefault(tool, {
            "event_count": 0,
            "saved_tokens": 0,
            "raw_tokens_sum": 0,
            "out_tokens_sum": 0,
            "credited_event_count": 0,
        })
        agg["event_count"] += 1
        if e.get("credited"):
            agg["saved_tokens"] += e.get("saved_tokens", 0)
            agg["raw_tokens_sum"] += e.get("raw_tokens", 0)
            agg["out_tokens_sum"] += e.get("out_tokens", 0)
            agg["credited_event_count"] += 1
    return {
        "session_id": session_id,
        "credited_saved_tokens": sum(e.get("saved_tokens", 0) for e in credited),
        "raw_tokens_sum": sum(e.get("raw_tokens", 0) for e in credited),
        "out_tokens_sum": sum(e.get("out_tokens", 0) for e in credited),
        "event_count": len(credited),  # credited-only -- see finalize_session's comment
        "fetch_url_count": sum(1 for e in events if e.get("tool") == "fetch_url"),
        "overhead_tokens": get_schema_overhead_tokens(),
        "by_tool": by_tool,
        "by_agent": _aggregate_by_agent(events),
        # Transcript parsing only happens at SessionEnd -- always 0 mid-session.
        "actual_input_tokens":      0,
        "actual_output_tokens":     0,
        "actual_cache_read_tokens": 0,
        "actual_cache_write_tokens": 0,
    }


def query_project_summary(project: str) -> dict:
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(credited_saved_tokens), 0), COALESCE(MAX(credited_saved_tokens), 0) "
        "FROM sessions WHERE project = ?",
        (project,),
    ).fetchone()
    conn.close()
    sessions, total, best = row
    return {"project": project, "sessions": sessions, "total_saved_tokens": total, "best_session_tokens": best}


def query_last_n_sessions(project: str, n: int = 10) -> list:
    conn = _connect()
    rows = conn.execute(
        "SELECT ended_at, credited_saved_tokens, "
        "actual_input_tokens, actual_output_tokens, actual_cache_read_tokens, actual_cache_write_tokens "
        "FROM sessions WHERE project = ? "
        "ORDER BY ended_at DESC LIMIT ?",
        (project, n),
    ).fetchall()
    conn.close()
    return [
        {
            "ended_at": r[0],
            "saved_tokens": r[1],
            "actual_input_tokens": r[2],
            "actual_output_tokens": r[3],
            "actual_cache_read_tokens": r[4],
            "actual_cache_write_tokens": r[5],
        }
        for r in reversed(rows)
    ]


def query_trend(project: str, bucket: str = "week", n: int = 12) -> list:
    """
    Groups a project's session history by day or week so a trend view can
    show whether usage is increasing or decreasing without the per-session
    noise that `query_last_n_sessions` exposes.

    bucket: "day"  -> groups by YYYY-MM-DD calendar date.
            "week" -> groups by ISO 8601 year-week key YYYY-Www, derived
                      via Python's datetime.isocalendar() so the grouping
                      is correct across calendar-year boundaries.

                      SQLite's strftime modifiers cannot produce correct ISO
                      week keys portably: %W/%Y splits year-boundary weeks
                      (2024-12-30 and 2025-01-01 fall in the same ISO week
                      but %Y/%W gives "2024-W53" and "2025-W00"), and
                      %G/%V (the ISO variants) were only added in SQLite
                      3.46.0 so they return NULL on Python 3.11 installs
                      that bundle SQLite 3.45.1. Python's isocalendar() is
                      available in all supported Python versions, depends on
                      no SQLite feature, and is correct by the stdlib.

    Returns at most `n` buckets ordered ascending (oldest first, most-recent
    last) so the sparkline in format_trend_view reads left-to-right in time.
    Empty when the project has no sessions.

    Raises ValueError for an unrecognised `bucket` value so callers can
    surface a clear error rather than silently querying with a broken format
    string.
    """
    from datetime import datetime as _dt

    if bucket == "day":
        def _key(ended_at: str) -> str:
            # ended_at is ISO 8601 UTC, e.g. "2026-09-07T12:00:00Z". The
            # first 10 characters are always YYYY-MM-DD.
            return ended_at[:10]
    elif bucket == "week":
        def _key(ended_at: str) -> str:  # type: ignore[no-redef]
            # Parse the timestamp and use isocalendar() for the ISO week --
            # correct across calendar-year boundaries regardless of SQLite
            # version (see docstring for why the SQL route is avoided here).
            dt = _dt.fromisoformat(ended_at.replace("Z", "+00:00"))
            iso = dt.isocalendar()
            return f"{iso.year}-W{iso.week:02d}"
    else:
        raise ValueError(
            f"Unknown bucket {bucket!r}. Valid values: 'day', 'week'."
        )

    conn = _connect()
    # Fetch all sessions for this project ordered descending so we can stop
    # early once we have `n` distinct buckets without reading the full table.
    raw = conn.execute(
        "SELECT ended_at, credited_saved_tokens FROM sessions "
        "WHERE project = ? ORDER BY ended_at DESC",
        (project,),
    ).fetchall()
    conn.close()

    # Aggregate into buckets from most-recent to oldest, stopping once we have
    # `n` distinct buckets -- equivalent to the SQL LIMIT but applied after the
    # Python-side bucketing so partial calendar-year weeks are counted correctly.
    agg: dict[str, dict] = {}
    for ended_at, saved in raw:
        key = _key(ended_at)
        if key not in agg:
            if len(agg) >= n:
                # We have enough buckets; older rows are outside the window.
                break
            agg[key] = {"sessions": 0, "total_saved_tokens": 0}
        agg[key]["sessions"] += 1
        agg[key]["total_saved_tokens"] += saved

    # Return ascending (oldest-first) so the sparkline reads left-to-right.
    return [
        {"bucket": k, "sessions": v["sessions"], "total_saved_tokens": v["total_saved_tokens"]}
        for k, v in reversed(list(agg.items()))
    ]


def query_all_projects() -> list:
    """
    Aggregates cross-project stats for the detail view's 'Across projects'
    block. Computes avg_tokens_per_session and best_tool per project via the
    existing sessions/session_tools data -- no new storage needed (issue #68).

    avg_tokens_per_session is computed in Python using round() to match the
    same rounding rule the "Last N sessions" average uses (Python's
    ties-to-even, not SQLite's half-away-from-zero ROUND(), which would give
    inconsistent results for the same data across the two blocks).

    best_tool is the tool with the highest total saved_tokens across all sessions
    for a given project, derived from session_tools. Returns None when no
    session_tools rows exist for that project (e.g. all its sessions predate the
    per-tool breakdown schema), so callers can suppress the field rather than
    fabricating a value.
    """
    conn = _connect()
    rows = conn.execute(
        """
        SELECT
            s.project,
            COUNT(*)                                                              AS sessions,
            COALESCE(SUM(s.credited_saved_tokens), 0)                            AS total_saved_tokens,
            (SELECT st.tool
             FROM session_tools st
             WHERE st.session_id IN (
                 SELECT session_id FROM sessions WHERE project = s.project
             )
             GROUP BY st.tool
             ORDER BY SUM(st.saved_tokens) DESC
             LIMIT 1)                                                             AS best_tool
        FROM sessions s
        GROUP BY s.project
        ORDER BY 3 DESC
        """
    ).fetchall()
    conn.close()
    return [
        {
            "project": r[0],
            "sessions": r[1],
            "total_saved_tokens": r[2],
            # Computed in Python (not SQL ROUND()) so both displayed averages use
            # the same ties-to-even rounding rule as format_detail_view's "Last N
            # sessions" block -- SQLite's ROUND() is half-away-from-zero and would
            # produce different results for the same inputs (e.g. 2.5 → 3 in SQL,
            # 2 in Python), making the two blocks inconsistent for identical data.
            "avg_tokens_per_session": round(r[2] / r[1]) if r[1] else 0,
            "best_tool": r[3],  # None when no session_tools rows exist for the project
        }
        for r in rows
    ]


def query_session_tool_breakdown(session_id: str) -> list:
    conn = _connect()
    rows = conn.execute(
        "SELECT tool, event_count, saved_tokens, raw_tokens_sum, out_tokens_sum, credited_event_count "
        "FROM session_tools WHERE session_id = ? ORDER BY saved_tokens DESC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "tool": r[0],
            "event_count": r[1],
            "saved_tokens": r[2],
            "raw_tokens_sum": r[3],
            "out_tokens_sum": r[4],
            "credited_event_count": r[5],
        }
        for r in rows
    ]


def query_session_agent_breakdown(session_id: str) -> list:
    """Per-agent rows for one finalized session (issue #365), same shape as
    finalize_session's `by_agent`. Empty for a session finalized before schema v5
    (nothing was recorded to backfill from)."""
    conn = _connect()
    rows = conn.execute(
        "SELECT is_subagent, agent_type, event_count, saved_tokens, raw_tokens_sum, out_tokens_sum, "
        "credited_event_count FROM session_agents WHERE session_id = ? "
        "ORDER BY saved_tokens DESC, is_subagent, agent_type",
        (session_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "is_subagent": r[0],
            "agent_type": r[1],
            "event_count": r[2],
            "saved_tokens": r[3],
            "raw_tokens_sum": r[4],
            "out_tokens_sum": r[5],
            "credited_event_count": r[6],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Transcript token parsing (opt-in stopgap -- see issue #165)
# ---------------------------------------------------------------------------

def parse_transcript_token_counts(transcript_path) -> Optional[dict]:
    """
    Parse a Claude Code session transcript JSONL and sum the actual token
    counts from each assistant turn's `message.usage` block (once per distinct
    `message.id` -- see the dedup note in the loop). Returns a dict
    with keys input/output/cache_read/cache_write, or None when the path is
    absent, the file is unreadable, or no assistant entries with usage data
    were found. Always fails open -- never raises.

    The transcript format is internal to Claude Code and undocumented; it may
    change with any Claude Code release. Callers must treat a None return as
    "no data available" rather than an error.

    # STOPGAP: remove when #164 is resolved (Stop hook will expose these directly).
    """
    if not transcript_path:
        return None
    totals: dict = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    found_any = False
    # Claude Code writes one JSONL line per content block (text, tool_use, ...)
    # of an assistant turn. Every line of the same turn shares one message.id
    # and repeats the SAME whole-turn usage object (not a per-block delta), so
    # summing per line multiplies each turn's usage by its block count (~1.9x
    # on real transcripts, #298). Invariant: each distinct message.id is
    # counted exactly once (first occurrence wins). Lines with no id cannot be
    # deduplicated, so they are always counted -- never silently dropped.
    seen_message_ids: set = set()
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "assistant":
                    continue
                message = obj.get("message", {})
                usage = message.get("usage", {})
                if not usage:
                    continue
                message_id = message.get("id")
                if message_id:
                    if message_id in seen_message_ids:
                        continue
                    seen_message_ids.add(message_id)
                totals["input"]      += int(usage.get("input_tokens",                   0) or 0)
                totals["output"]     += int(usage.get("output_tokens",                  0) or 0)
                totals["cache_read"] += int(usage.get("cache_read_input_tokens",        0) or 0)
                totals["cache_write"]+= int(usage.get("cache_creation_input_tokens",    0) or 0)
                found_any = True
    except Exception:
        return None
    return totals if found_any else None


def parse_session_token_counts(transcript_path) -> Optional[dict]:
    """
    Like parse_transcript_token_counts, but for a whole SESSION: the main
    transcript plus every subagent transcript that belongs to it (#364).

    Why a separate function: the payload's `transcript_path` is only the MAIN
    session's transcript. Claude Code writes each subagent's turns to its own
    file, `<projdir>/<session_id>/subagents/agent-<agent_id>.jsonl`, next to the
    main `<projdir>/<session_id>.jsonl` -- confirmed live: the main transcript
    had zero `isSidechain` lines, so its totals never included subagent usage
    and "tokens processed" was understated for any session that delegates work
    (autowork most of all). parse_transcript_token_counts is left untouched so
    its contract, and the #298 dedup it implements, stay scoped to one file.

    The subagents directory is derived from the transcript path itself (strip
    the `.jsonl` suffix, append `/subagents`) rather than from the hook
    payload's session_id, so it follows whatever layout Claude Code actually
    used for this transcript. Each file is parsed separately: a `message.id`
    repeated across two DIFFERENT files is counted in each, because the dedup
    exists to collapse one turn's per-content-block lines inside one file, not
    to merge separate files.

    Returns the summed input/output/cache_read/cache_write dict, or None when
    neither the main transcript nor any subagent file yielded usage data (same
    contract as parse_transcript_token_counts). Always fails open -- a missing
    directory or an unreadable/malformed subagent file is skipped, never raised,
    so a layout change degrades to the previous main-transcript-only behavior.

    Known gap, by design: a subagent still running when the session ends, or a
    resumed session, may be only partially counted.

    # STOPGAP: remove when #164 is resolved (Stop hook will expose these directly).
    """
    if not transcript_path:
        return None
    parts = []
    main = parse_transcript_token_counts(transcript_path)
    if main:
        parts.append(main)
    try:
        subagents_dir = Path(transcript_path).with_suffix("") / "subagents"
        for agent_file in sorted(subagents_dir.glob("agent-*.jsonl")):
            counts = parse_transcript_token_counts(agent_file)
            if counts:
                parts.append(counts)
    except Exception:
        pass  # keep whatever was gathered before the failure
    if not parts:
        return None
    return {key: sum(p[key] for p in parts) for key in ("input", "output", "cache_read", "cache_write")}


# ---------------------------------------------------------------------------
# Formatting -- shared by hooks/session_end_savings.py and the
# savings_summary/savings_detail MCP tools, so both surfaces render
# identically.
# ---------------------------------------------------------------------------

def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"~{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1000:
        return f"~{n / 1000:.1f}k".replace(".0k", "k")
    return f"~{n}" if n else "0"


# Relative cost labels shown in the actual-token breakdown table.
# These are approximate ratios vs. full input price, not exact Anthropic rates
# (Anthropic doesn't expose live pricing to hooks). Rendered as-is so users
# understand the cost *structure*, not as a calculated dollar figure.
_COST_ROWS = [
    ("Fresh input",  "actual_input_tokens",       "1.00×", "full price"),
    ("Cache writes", "actual_cache_write_tokens",  "1.25×", "storage overhead"),
    ("Cache reads",  "actual_cache_read_tokens",   "0.10×", "10× cheaper than fresh"),
    ("Output",       "actual_output_tokens",       "5.00×", "most expensive per token"),
]


def _format_actual_token_block(session_agg: dict, project: str = "") -> list:
    """Return lines for the 'Tokens Anthropic processed' block, or [] when no
    actual token data is present (feature disabled or transcript unavailable)."""
    inp  = session_agg.get("actual_input_tokens",       0) or 0
    out  = session_agg.get("actual_output_tokens",      0) or 0
    cr   = session_agg.get("actual_cache_read_tokens",  0) or 0
    cw   = session_agg.get("actual_cache_write_tokens", 0) or 0
    total = inp + out + cr + cw
    if not total:
        return []

    def _pct(val: int) -> str:
        return f"{100 * val / total:.1f}%"

    sep = "─" * 56
    header = f"Tokens Anthropic processed · {project}" if project else "Tokens Anthropic processed"
    lines = [
        "",
        header,
        f"  {'Token type':<14}  {'Count':>9}  {'% of total':>10}   Relative cost",
        f"  {sep}",
    ]
    for label, key, cost, note in _COST_ROWS:
        val = session_agg.get(key, 0) or 0
        lines.append(f"  {label:<14}  {_fmt_tokens(val):>9}  {_pct(val):>10}   {cost}  ({note})")
    lines += [
        f"  {sep}",
        f"  {'Total':<14}  {_fmt_tokens(total):>9}  {'100.0%':>10}",
        "",
    ]

    saved = session_agg.get("credited_saved_tokens", 0) or 0
    if saved:
        rate = f"{100 * saved / total:.1f}%"
        lines.append(f"  Local compression avoided  {_fmt_tokens(saved):>9} est.  ({rate} of total processed)")

    # Cache efficiency ratio: how many times more cache-read tokens than fresh
    # input. A high ratio means skills/CLAUDE.md are producing stable, cacheable
    # context -- Claude reuses existing knowledge rather than re-ingesting fresh
    # tokens every turn. A low ratio signals a lot of novel uncached content.
    if inp > 0:
        ratio = round(cr / inp)
        lines.append(f"  Cache efficiency           {ratio:>9}×  (reads vs. fresh input — higher is better)")

    return lines


def _bar(fraction: float, width: int = 12) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    return "▇" * filled + "·" * (width - filled)


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return _SPARK_CHARS[0] * len(values)
    span = hi - lo
    return "".join(_SPARK_CHARS[min(7, int((v - lo) / span * 7))] for v in values)


def format_simple_view(session_agg: dict, project_summary: dict) -> str:
    saved = session_agg.get("credited_saved_tokens", 0)
    events = session_agg.get("event_count", 0)
    raw_sum = session_agg.get("raw_tokens_sum", 0)
    out_sum = session_agg.get("out_tokens_sum", 0)
    # Clamped like saved_tokens already is (record_event's max(0, raw-out)) --
    # an LLM summary CAN end up longer than a small raw chunk, which would
    # otherwise display a negative "reduction" alongside a 0-token saved
    # figure, an inconsistent and confusing pairing.
    pct = max(0, round(100 * (1 - out_sum / raw_sum))) if raw_sum else 0
    overhead = session_agg.get("overhead_tokens", 0)
    fetch_count = session_agg.get("fetch_url_count", 0)
    project = session_agg.get("project") or project_summary.get("project", "")

    lines = [
        "ClaudeRunway · Token Savings",
        "",
        f"This session · {project}",
        f"  Context tokens avoided   {_fmt_tokens(saved):>8}   ({events} events{f', {pct}% reduction' if raw_sum else ''})",
    ]
    if overhead:
        lines.append(f"  Tool overhead            {_fmt_tokens(overhead):>8}/turn  (mostly cached after turn 1)")
    if fetch_count:
        lines.append(f"  fetch_url                {fetch_count:>8} calls  (not credited — WebFetch already compresses server-side)")

    lines += _format_actual_token_block(session_agg, project)

    lines += [
        "",
        "This project · all time",
        f"  Sessions tracked         {project_summary.get('sessions', 0):>8}",
        f"  Total tokens avoided     {_fmt_tokens(project_summary.get('total_saved_tokens', 0)):>8}",
        f"  Best session             {_fmt_tokens(project_summary.get('best_session_tokens', 0)):>8}",
    ]
    return "\n".join(lines)


def format_detail_view(session_agg: dict, project_summary: dict, tool_breakdown: list, last_n: list, all_projects: list) -> str:
    saved = session_agg.get("credited_saved_tokens", 0)
    events = session_agg.get("event_count", 0)
    raw_sum = session_agg.get("raw_tokens_sum", 0)
    out_sum = session_agg.get("out_tokens_sum", 0)
    pct = max(0, round(100 * (1 - out_sum / raw_sum))) if raw_sum else 0
    project = session_agg.get("project") or project_summary.get("project", "")

    lines = [
        "ClaudeRunway · Token Savings — Detail",
        "",
        f"This session · {project}",
        f"  Context tokens avoided  {_fmt_tokens(saved)}  ({events} events{f', {pct}% reduction' if raw_sum else ''})",
    ]

    lines += _format_actual_token_block(session_agg, project)

    if tool_breakdown:
        max_saved = max((t["saved_tokens"] for t in tool_breakdown), default=0) or 1
        lines += ["", "By tool"]
        for t in tool_breakdown:
            credited_event_count = t.get("credited_event_count", 0)
            # credited_event_count > 0: row has real v3+ per-tool aggregates, so
            # both the credited/uncredited distinction AND the avg % are reliable.
            # saved_tokens > 0: heuristic fallback for v2-era rows migrated to v3
            # (credited_event_count backfilled to 0 but saved_tokens is real) --
            # these can be rendered as credited, but the avg % is suppressed
            # because out_tokens_sum is also 0 (backfilled), so any percentage
            # derived from it would be fabricated (raw=X, out=0 → 100%, wrong).
            credited = credited_event_count > 0 or t["saved_tokens"] > 0
            if credited:
                bar = _bar(t["saved_tokens"] / max_saved)
                raw = t.get("raw_tokens_sum", 0)
                out = t.get("out_tokens_sum", 0)
                # Only compute avg % when credited_event_count is available --
                # if it's 0 (backfilled v2 row), out_tokens_sum is also 0 and
                # computing 1 - 0/raw would yield a fabricated 100%, not a real
                # measurement. Omitting the metric is more honest than a wrong %.
                if raw and credited_event_count > 0:
                    avg_pct: Optional[int] = max(0, round(100 * (1 - out / raw)))
                else:
                    avg_pct = None
                pct_str = f"  {avg_pct}% avg reduction" if avg_pct is not None else ""
                lines.append(f"  {t['tool']:<16} {t['event_count']:>3} events  {_fmt_tokens(t['saved_tokens']):>8}  {bar}{pct_str}")
            else:
                lines.append(f"  {t['tool']:<16} {t['event_count']:>3} calls  {'—':>8}  (not credited)")

    # Issue #365: shown only when at least one subagent earned a POSITIVE saving
    # this session, so a session with no delegation renders exactly as before.
    # Gated on saved_tokens > 0, not merely "a subagent row exists": a subagent
    # whose only activity was uncredited (fetch_url) or whose credited events
    # saved nothing still gets a row in by_agent, and listing it would print a
    # misleading "0 events / 0 tokens" line (Copilot review on PR #367). The same
    # filter applies to the per-type lines below. Counts are credited events,
    # matching the "N events" headline above (not total calls, which include
    # uncredited fetch_url activity).
    by_agent = session_agg.get("by_agent") or []
    subagent_rows = [a for a in by_agent if a.get("is_subagent") and a.get("saved_tokens", 0) > 0]
    if subagent_rows:
        main_rows = [a for a in by_agent if not a.get("is_subagent")]
        main_saved = sum(a.get("saved_tokens", 0) for a in main_rows)
        main_events = sum(a.get("credited_event_count", 0) for a in main_rows)
        lines += ["", "Main session vs subagents",
                  f"  {'main session':<24} {main_events:>3} events  {_fmt_tokens(main_saved):>8}"]
        for a in subagent_rows:
            label = f"subagent · {a['agent_type']}" if a.get("agent_type") else "subagent"
            lines.append(
                f"  {label:<24} {a.get('credited_event_count', 0):>3} events  {_fmt_tokens(a.get('saved_tokens', 0)):>8}"
            )

    if last_n:
        values = [s["saved_tokens"] for s in last_n]
        avg = round(sum(values) / len(values)) if values else 0
        best = max(values) if values else 0
        lines += [
            "",
            f"Last {len(last_n)} sessions · {project}",
            f"  tokens avoided  {_sparkline(values)}  avg {_fmt_tokens(avg)} · best {_fmt_tokens(best)}",
        ]

        # Show the actual Anthropic token breakdown for the most recent past
        # session that has data (last_n is ascending, so last entry = most
        # recent). These values are only populated when
        # CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS=1 was set at session end;
        # sessions without it have all-zero actual_* columns in the DB and
        # are skipped silently.
        for recent in reversed(last_n):
            actual_total = (
                recent.get("actual_input_tokens", 0)
                + recent.get("actual_output_tokens", 0)
                + recent.get("actual_cache_read_tokens", 0)
                + recent.get("actual_cache_write_tokens", 0)
            )
            if actual_total:
                date = recent["ended_at"][:10]
                past_agg = {
                    "actual_input_tokens":      recent["actual_input_tokens"],
                    "actual_output_tokens":     recent["actual_output_tokens"],
                    "actual_cache_read_tokens": recent["actual_cache_read_tokens"],
                    "actual_cache_write_tokens": recent["actual_cache_write_tokens"],
                    "credited_saved_tokens":    recent["saved_tokens"],
                }
                lines += _format_actual_token_block(past_agg, f"{project} · {date}")
                break  # only the most recent session with data

    if all_projects:
        lines += ["", "Across projects · all time"]
        # Column width fits the longest project name (minimum 5 for "Total").
        col_w = max(max(len(p["project"]) for p in all_projects), 5)
        for p in all_projects:
            avg = p.get("avg_tokens_per_session")
            best = p.get("best_tool")
            # Append avg and best-tool when present -- both are optional so
            # old-shape dicts (e.g. passed by tests that predate issue #68)
            # and projects whose session_tools rows are absent render safely.
            suffix = ""
            if avg is not None:
                suffix += f"  avg {_fmt_tokens(avg)}/session"
            if best:
                suffix += f"  best: {best}"
            lines.append(f"  {p['project']:<{col_w}} {p['sessions']:>3} sessions  {_fmt_tokens(p['total_saved_tokens']):>8}{suffix}")
        # Grand total (#95) -- answers "how much has this toolkit saved me
        # overall" directly instead of making the reader add up every row by
        # hand. Deliberately inside the `if all_projects:` guard so an empty
        # list still means no total row, same as the per-project loop above.
        total_sessions = sum(p["sessions"] for p in all_projects)
        total_saved = sum(p["total_saved_tokens"] for p in all_projects)
        lines.append(f"  {'—' * col_w} {'—' * 3} {'—' * 9} {'—' * 8}")
        lines.append(f"  {'Total':<{col_w}} {total_sessions:>3} sessions  {_fmt_tokens(total_saved):>8}")

    return "\n".join(lines)


def format_trend_view(trend_rows: list, project: str, bucket: str = "week") -> str:
    """
    Renders a day/week-bucketed trend view for one project.

    trend_rows is the list returned by query_trend (ascending, oldest first).
    Each row: {bucket, sessions, total_saved_tokens}.

    Shows:
      - a sparkline across all buckets (reuses _sparkline)
      - a table of recent buckets with session count and token totals
      - a grand-total line at the bottom

    When trend_rows is empty, returns a short "no history yet" message rather
    than an empty table so callers can relay it directly.
    """
    bucket_label = "Weekly" if bucket == "week" else "Daily"
    header = f"ClaudeRunway · {bucket_label} Trend · {project}"

    if not trend_rows:
        return f"{header}\n\n  No session history yet for this project."

    values = [r["total_saved_tokens"] for r in trend_rows]
    spark = _sparkline(values)

    total_sessions = sum(r["sessions"] for r in trend_rows)
    total_saved = sum(r["total_saved_tokens"] for r in trend_rows)

    # Column widths: bucket strings are at most 10 chars (YYYY-MM-DD) or
    # 8 chars (YYYY-Www); 16 chars gives comfortable padding for both.
    lines = [
        header,
        "",
        f"  {spark}  ({len(trend_rows)} {bucket}s · {total_sessions} sessions)",
        "",
        f"  {'Bucket':<16} {'Sessions':>8}  {'Tokens avoided':>14}",
        f"  {'-' * 16} {'-' * 8}  {'-' * 14}",
    ]
    for r in trend_rows:
        lines.append(
            f"  {r['bucket']:<16} {r['sessions']:>8}  {_fmt_tokens(r['total_saved_tokens']):>14}"
        )
    lines += [
        f"  {'-' * 16} {'-' * 8}  {'-' * 14}",
        f"  {'Total':<16} {total_sessions:>8}  {_fmt_tokens(total_saved):>14}",
    ]
    return "\n".join(lines)


def format_json(session_agg: dict, project_summary: dict, tool_breakdown: list, last_n: list, all_projects: list) -> str:
    """
    Machine-readable JSON export of all savings data.  Mirrors the same five
    data blocks that format_detail_view renders as text, serialised as a
    single JSON object so the caller can pipe it into jq, a spreadsheet
    importer, or a dashboard without hand-parsing the terminal output.

    The ``session`` key carries the same aggregate dict that
    format_simple_view/format_detail_view already receive, with one
    normalisation: numeric values are always ints (never None) so downstream
    consumers don't have to guard against null.
    """
    def _int(v) -> int:  # type: ignore[no-untyped-def]
        return int(v) if v is not None else 0

    session_raw = session_agg or {}
    doc = {
        "session": {
            "project": session_raw.get("project") or project_summary.get("project", ""),
            "credited_saved_tokens": _int(session_raw.get("credited_saved_tokens")),
            "raw_tokens_sum": _int(session_raw.get("raw_tokens_sum")),
            "out_tokens_sum": _int(session_raw.get("out_tokens_sum")),
            "event_count": _int(session_raw.get("event_count")),
            "fetch_url_count": _int(session_raw.get("fetch_url_count")),
            "overhead_tokens": _int(session_raw.get("overhead_tokens")),
            # actual_* are always 0 for a live session (transcript is only
            # parsed at SessionEnd); included here so JSON consumers don't
            # need to special-case their absence.
            "actual_input_tokens":      _int(session_raw.get("actual_input_tokens")),
            "actual_output_tokens":     _int(session_raw.get("actual_output_tokens")),
            "actual_cache_read_tokens": _int(session_raw.get("actual_cache_read_tokens")),
            "actual_cache_write_tokens": _int(session_raw.get("actual_cache_write_tokens")),
        },
        "project_summary": {
            "project": project_summary.get("project", ""),
            "sessions": _int(project_summary.get("sessions")),
            "total_saved_tokens": _int(project_summary.get("total_saved_tokens")),
            "best_session_tokens": _int(project_summary.get("best_session_tokens")),
        },
        "tool_breakdown": [
            {
                "tool": t.get("tool", ""),
                "event_count": _int(t.get("event_count")),
                "saved_tokens": _int(t.get("saved_tokens")),
                "raw_tokens_sum": _int(t.get("raw_tokens_sum")),
                "out_tokens_sum": _int(t.get("out_tokens_sum")),
                "credited_event_count": _int(t.get("credited_event_count")),
            }
            for t in (tool_breakdown or [])
        ],
        "last_n_sessions": [
            {
                "ended_at": s.get("ended_at", ""),
                "saved_tokens": _int(s.get("saved_tokens")),
                "actual_input_tokens":      _int(s.get("actual_input_tokens")),
                "actual_output_tokens":     _int(s.get("actual_output_tokens")),
                "actual_cache_read_tokens": _int(s.get("actual_cache_read_tokens")),
                "actual_cache_write_tokens": _int(s.get("actual_cache_write_tokens")),
            }
            for s in (last_n or [])
        ],
        "all_projects": [
            {
                "project": p.get("project", ""),
                "sessions": _int(p.get("sessions")),
                "total_saved_tokens": _int(p.get("total_saved_tokens")),
                "avg_tokens_per_session": _int(p.get("avg_tokens_per_session")),
                # best_tool may genuinely be None when no session_tools rows exist --
                # kept as null in JSON so consumers can distinguish "no data" from
                # an empty string, unlike the text view which just omits it silently.
                "best_tool": p.get("best_tool"),
            }
            for p in (all_projects or [])
        ],
    }
    return json.dumps(doc, indent=2)


def format_csv(session_agg: dict, project_summary: dict, tool_breakdown: list, last_n: list, all_projects: list, table: str = "session") -> str:
    """
    Machine-readable CSV export for spreadsheet import.

    ``table`` selects which data slice to emit:

    * ``"session"``  — one-row summary of the current session (default).
      Columns: project, credited_saved_tokens, raw_tokens_sum, out_tokens_sum,
               event_count, fetch_url_count, overhead_tokens,
               project_total_saved_tokens, project_sessions, project_best_session.
    * ``"tools"``    — per-tool breakdown rows.
      Columns: tool, event_count, saved_tokens, raw_tokens_sum, out_tokens_sum,
               credited_event_count.
    * ``"projects"`` — cross-project all-time rows.
      Columns: project, sessions, total_saved_tokens, avg_tokens_per_session,
               best_tool.

    Uses stdlib ``csv``, so quoting follows RFC 4180 and callers don't need to
    worry about values that contain commas or quotes.

    Raises ValueError for an unrecognised ``table`` value rather than silently
    returning empty output, so callers can surface a clear error rather than
    silently producing nothing.
    """
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")

    def _int(v) -> int:  # type: ignore[no-untyped-def]
        return int(v) if v is not None else 0

    def _safe_str(v: Optional[str]) -> str:
        """
        Prefix text that starts with a formula-trigger character (=, +, -, @)
        with an apostrophe so that spreadsheet applications (Excel, Google
        Sheets, LibreOffice) treat the cell as plain text rather than
        evaluating it as a formula. This is the standard CSV injection
        mitigation -- csv.writer handles RFC-4180 delimiter quoting but does
        not neutralise formula-leading cells. The prefix is visible inside the
        cell only if the spreadsheet application strips it for a non-formula
        cell; most modern applications recognise the convention and hide it.
        An empty or None input returns an empty string unchanged.

        Whitespace bypass: tabs, CR, LF, and ordinary spaces at the start of
        a value are stripped by many spreadsheet applications BEFORE formula
        detection, so "\t=1+1" would be treated as "=1+1" and still execute.
        We check the first NON-whitespace character so these bypass attempts
        are caught regardless of leading whitespace. The whitespace is not
        stripped from the output -- only the check is done against the
        stripped form; the prefix is added to the original value so the cell
        content remains unchanged apart from the leading apostrophe.
        """
        s = v or ""
        first_non_ws = s.lstrip()
        if first_non_ws and first_non_ws[0] in ("=", "+", "-", "@"):
            return "'" + s
        return s

    if table == "session":
        session_raw = session_agg or {}
        w.writerow([
            "project", "credited_saved_tokens", "raw_tokens_sum", "out_tokens_sum",
            "event_count", "fetch_url_count", "overhead_tokens",
            "project_total_saved_tokens", "project_sessions", "project_best_session",
        ])
        w.writerow([
            _safe_str(session_raw.get("project") or project_summary.get("project", "")),
            _int(session_raw.get("credited_saved_tokens")),
            _int(session_raw.get("raw_tokens_sum")),
            _int(session_raw.get("out_tokens_sum")),
            _int(session_raw.get("event_count")),
            _int(session_raw.get("fetch_url_count")),
            _int(session_raw.get("overhead_tokens")),
            _int(project_summary.get("total_saved_tokens")),
            _int(project_summary.get("sessions")),
            _int(project_summary.get("best_session_tokens")),
        ])

    elif table == "tools":
        w.writerow([
            "tool", "event_count", "saved_tokens", "raw_tokens_sum",
            "out_tokens_sum", "credited_event_count",
        ])
        for t in (tool_breakdown or []):
            w.writerow([
                _safe_str(t.get("tool", "")),
                _int(t.get("event_count")),
                _int(t.get("saved_tokens")),
                _int(t.get("raw_tokens_sum")),
                _int(t.get("out_tokens_sum")),
                _int(t.get("credited_event_count")),
            ])

    elif table == "projects":
        w.writerow([
            "project", "sessions", "total_saved_tokens",
            "avg_tokens_per_session", "best_tool",
        ])
        for p in (all_projects or []):
            w.writerow([
                _safe_str(p.get("project", "")),
                _int(p.get("sessions")),
                _int(p.get("total_saved_tokens")),
                _int(p.get("avg_tokens_per_session")),
                # best_tool is None when no session_tools data exists -- emit as
                # empty string so consumers see a blank cell, not the literal "None".
                _safe_str(p.get("best_tool") or ""),
            ])

    else:
        raise ValueError(
            f"Unknown table {table!r}. Valid values: 'session', 'tools', 'projects'."
        )

    return buf.getvalue()
