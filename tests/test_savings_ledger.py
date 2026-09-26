#!/usr/bin/env python3
"""Tests for libs/savings_ledger.py -- current_session_id()'s optional
project filter and record_event()'s project field (issue #35), and the
PRAGMA user_version schema migration scaffold in _connect() (issue #30/#69).

This module had ZERO test coverage before this file (also separately
tracked as issue #40) -- these tests are scoped to the bugs the PRs that
added them actually fixed, not a full audit of the module.

Stdlib-only (unittest, no pytest) and no network:

    .venv/bin/python -m unittest discover -s tests

Points CLAUDE_RUNWAY_SAVINGS_DB at a temp file for every test (same
env-var-redirection approach tests/test_cache_db.py uses for its own
sibling module) so nothing here ever touches the real
~/.claude/claude-runway/ store, and _sessions_dir() -- derived from
resolve_db_path().parent -- lands under that temp dir too.

Issue #213: current_session_id(project=...) now delegates to
session_id_lib.session_id(SessionIdStrategy.SHADOW_FILE, ...), which scans
its OWN, separately-resolved (session_id_lib._sessions_dir(), hardcoded to
~/.claude/claude-runway/sessions -- see that module's docstring) marker
directory, independent of CLAUDE_RUNWAY_SAVINGS_DB. SavingsLedgerTestCase
below also redirects THAT directory to the same temp dir, both so these
tests stay deterministic (a real machine running this very test suite can
have real, live session_id_lib markers sitting under the real
~/.claude/claude-runway/sessions right now) and so project-filtered
fixtures can be set up via session_id_lib.record_shadow_marker() directly
(matching what the CORE hooks/record_session_id.py hook actually writes).
"""

import csv as _csv_mod
import io as _io_mod
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import savings_ledger as L  # noqa: E402
import session_id_lib  # noqa: E402


class SavingsLedgerTestCase(unittest.TestCase):
    """Common temp-DB-dir setup shared by every test class below."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "savings.db"
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_SAVINGS_DB": str(db_path)})
        self._env_patch.start()

        # Issue #213: session_id_lib's own sessions dir is independent of
        # CLAUDE_RUNWAY_SAVINGS_DB (see module docstring above) -- redirect it
        # to the SAME temp dir savings_ledger's own _sessions_dir() resolves
        # to above, so current_session_id(project=...)'s delegation is both
        # deterministic (never touches a real machine's live markers) and
        # testable against fixtures this file controls.
        self._session_id_lib_sessions_patch = mock.patch.object(
            session_id_lib, "_sessions_dir", return_value=db_path.parent / "sessions"
        )
        self._session_id_lib_sessions_patch.start()
        # Same for TRANSCRIPT_SCAN's fallback: without this, a real live Claude
        # Code transcript under ~/.claude/projects/ for a checkout whose
        # basename matches a test's project (e.g. "claude-runway") leaks in.
        self._session_id_lib_projects_patch = mock.patch.object(
            session_id_lib, "_transcript_projects_dir", return_value=db_path.parent / "projects"
        )
        self._session_id_lib_projects_patch.start()

    def tearDown(self):
        self._session_id_lib_projects_patch.stop()
        self._session_id_lib_sessions_patch.stop()
        self._env_patch.stop()
        self._tmpdir.cleanup()


class RecordEventProjectField(SavingsLedgerTestCase):
    def test_project_is_stored_on_the_entry(self):
        L.record_event("sess-1", "hook:Bash", 1000, 100, True, "some cmd", project="repo-a")
        events = L.read_session_events("sess-1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["project"], "repo-a")

    def test_project_defaults_to_none(self):
        # Matches the pre-fix JSONL shape (no project key at all, in effect).
        # Issue #213: this field no longer drives current_session_id()'s
        # project-filtered lookup at all (that now reads session_id_lib's
        # shadow markers instead) -- kept as a data-integrity guard so an
        # omitted project is stored honestly as None, not fabricated.
        L.record_event("sess-1", "hook:Bash", 1000, 100, True, "some cmd")
        events = L.read_session_events("sess-1")
        self.assertIsNone(events[0]["project"])


class CurrentSessionIdProjectFilter(SavingsLedgerTestCase):
    """The core of issue #35's fix: current_session_id(project=...) must not
    hand back a real, unrelated session just because its marker happens to be
    the most recently modified file in the shared sessions dir.

    Issue #213: the project-FILTERED cases below now set up their fixtures
    via session_id_lib.record_shadow_marker() (what the CORE
    hooks/record_session_id.py hook actually writes) rather than
    L.record_event() (this module's own transient compression-event log),
    since current_session_id(project=...) delegates to session_id_lib's
    SHADOW_FILE strategy, which scans the former, not the latter. The
    UNFILTERED cases are UNCHANGED -- that path still scans this module's own
    _sessions_dir() via record_event()-created files, per current_session_id's
    own docstring."""

    def test_returns_none_with_no_sessions_at_all(self):
        self.assertIsNone(L.current_session_id())
        self.assertIsNone(L.current_session_id(project="anything"))

    def test_unfiltered_call_preserves_old_most_recent_file_behavior(self):
        # Regression guard for callers that don't pass project at all.
        L.record_event("sess-old", "hook:Bash", 100, 10, True, project="repo-a")
        L.record_event("sess-new", "hook:Bash", 100, 10, True, project="repo-b")
        # Force sess-new's file to have a strictly later mtime than sess-old's.
        old_path = L._session_jsonl_path("sess-old")
        new_path = L._session_jsonl_path("sess-new")
        os.utime(old_path, (1000, 1000))
        os.utime(new_path, (2000, 2000))
        self.assertEqual(L.current_session_id(), "sess-new")

    def test_project_filter_skips_a_more_recently_touched_other_project(self):
        # The exact failure mode from the issue and my confirmation comment:
        # a DIFFERENT project's session is the most recently modified marker,
        # but the caller asked for a specific project -- it must not be
        # substituted in.
        # Use recent relative timestamps so issue #236's stale-marker fallback
        # does not fire (markers within TTL are returned directly, not
        # passed to TRANSCRIPT_SCAN): sess-other is 30s more recent than
        # sess-mine, but sess-mine is still well within the 168h TTL.
        import time as _time
        now = _time.time()
        session_id_lib.record_shadow_marker("sess-mine", project="claude-runway")
        session_id_lib.record_shadow_marker("sess-other", project="Acme.Support.Nx")
        old_path = session_id_lib._shadow_marker_path("sess-mine")
        new_path = session_id_lib._shadow_marker_path("sess-other")
        os.utime(old_path, (now - 60, now - 60))
        os.utime(new_path, (now - 30, now - 30))  # sess-other is the more recent marker

        # Filtered by the ACTUAL project: correctly finds the older marker,
        # not the more-recently-touched unrelated one.
        self.assertEqual(L.current_session_id(project="claude-runway"), "sess-mine")
        self.assertEqual(L.current_session_id(project="Acme.Support.Nx"), "sess-other")

    def test_project_filter_returns_none_when_nothing_matches(self):
        # No live session at all for the requested project -- must return
        # None (caller shows "no live session"), not substitute an unrelated
        # session just because it's the only one that exists.
        session_id_lib.record_shadow_marker("sess-other", project="some-other-repo")
        self.assertIsNone(L.current_session_id(project="claude-runway"))

    def test_project_filter_never_matches_a_marker_for_a_different_basename(self):
        # A marker recorded under a project whose basename differs from the
        # filter must not match -- same "no substitution" guarantee as
        # session_id_lib's own SHADOW_FILE tests, exercised here through the
        # delegating caller.
        session_id_lib.record_shadow_marker("sess-legacy", project="/some/other/path")
        self.assertIsNone(L.current_session_id(project="claude-runway"))


class SessionIdDelegatesToSessionIdLibTest(SavingsLedgerTestCase):
    """Issue #213: current_session_id(project=...) must come from
    libs/session_id_lib.py's SessionIdStrategy.SHADOW_FILE rather than this
    module re-implementing its own marker scan -- asserting on the exact
    returned session_id (not just "some session_id came back") is what
    actually proves delegation happened, matching PR #218's precedent for the
    sibling PROXY-strategy migration (issue #214)."""

    def test_matches_session_id_lib_directly_for_the_same_project(self):
        session_id_lib.record_shadow_marker("sess-abc", project="claude-runway")
        expected = session_id_lib.session_id(
            session_id_lib.SessionIdStrategy.SHADOW_FILE, project="claude-runway"
        )
        self.assertEqual(expected, "sess-abc")  # sanity: the fixture itself is valid
        self.assertEqual(L.current_session_id(project="claude-runway"), expected)


def _old_schema_db_path(db_path: Path) -> None:
    """Builds a `sessions` table matching the schema from before
    raw_tokens_sum/out_tokens_sum/overhead_tokens existed, plus one
    pre-existing row -- the exact shape issue #30 is about. `session_tools`/
    `meta` are deliberately NOT created here either, to also exercise
    _connect()'s CREATE TABLE IF NOT EXISTS for those two."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            credited_saved_tokens INTEGER NOT NULL,
            event_count INTEGER NOT NULL,
            fetch_url_count INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO sessions VALUES ('old-sess', 'old-project', '2026-01-01T00:00:00Z', 42, 3, 0)"
    )
    conn.commit()
    conn.close()


class SchemaMigration(SavingsLedgerTestCase):
    """Issue #30/#69: a pre-existing savings.db from before
    raw_tokens_sum/out_tokens_sum/overhead_tokens existed must not make
    finalize_session raise sqlite3.OperationalError -- and the migration
    scaffold itself must be safe to run against a DB that already has the
    columns but hasn't been through PRAGMA user_version tracking yet."""

    def _db_path(self) -> Path:
        return L.resolve_db_path()

    def test_old_schema_db_no_longer_raises_and_gains_the_columns(self):
        _old_schema_db_path(self._db_path())

        # The exact regression: this used to raise sqlite3.OperationalError.
        result = L.finalize_session("sess1", "myproject", overhead_tokens=5)
        self.assertEqual(result["raw_tokens_sum"], 0)
        self.assertEqual(result["overhead_tokens"], 5)

        conn = sqlite3.connect(str(self._db_path()))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        self.assertTrue({"raw_tokens_sum", "out_tokens_sum", "overhead_tokens"} <= columns)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        conn.close()

    def test_pre_existing_row_survives_the_migration(self):
        # ALTER TABLE ADD COLUMN must backfill the existing row via its
        # DEFAULT 0, not drop or corrupt it -- confirms this is a real
        # in-place migration, not a silent rebuild-and-lose-data.
        _old_schema_db_path(self._db_path())
        L.finalize_session("sess1", "myproject")  # triggers _connect() -> migration

        conn = sqlite3.connect(str(self._db_path()))
        row = conn.execute(
            "SELECT project, credited_saved_tokens, raw_tokens_sum, out_tokens_sum, overhead_tokens "
            "FROM sessions WHERE session_id = 'old-sess'"
        ).fetchone()
        conn.close()
        self.assertEqual(row, ("old-project", 42, 0, 0, 0))

    def test_brand_new_db_gets_latest_schema_and_version_immediately(self):
        # No file exists yet -- _connect()'s CREATE TABLE already produces
        # the full current schema, so there's nothing for a migration to
        # add, but user_version must still end up at SCHEMA_VERSION.
        self.assertFalse(self._db_path().exists())
        L.finalize_session("sess1", "myproject")

        conn = sqlite3.connect(str(self._db_path()))
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        conn.close()

    def test_already_current_columns_but_unversioned_db_is_not_an_error(self):
        # Simulates a DB created by an already-shipped build of this file
        # from BEFORE this migration scaffold existed: the columns are
        # already there (they're in that build's own CREATE TABLE), but
        # PRAGMA user_version was never set, so it reads back as 0 -- same
        # as a genuinely old DB. The migration must detect the columns are
        # already present and skip re-adding them, not raise "duplicate
        # column name."
        conn = sqlite3.connect(str(self._db_path()))
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                credited_saved_tokens INTEGER NOT NULL,
                raw_tokens_sum INTEGER NOT NULL DEFAULT 0,
                out_tokens_sum INTEGER NOT NULL DEFAULT 0,
                event_count INTEGER NOT NULL,
                fetch_url_count INTEGER NOT NULL,
                overhead_tokens INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()
        conn.close()

        # Must not raise "duplicate column name: raw_tokens_sum".
        L.finalize_session("sess1", "myproject")

        conn = sqlite3.connect(str(self._db_path()))
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        conn.close()

    def test_repeated_connects_are_idempotent(self):
        _old_schema_db_path(self._db_path())
        L.finalize_session("sess1", "myproject")
        # Second call must not re-run the migration or error -- the loop in
        # _run_migrations is empty once current_version == SCHEMA_VERSION.
        L.finalize_session("sess2", "myproject")

        conn = sqlite3.connect(str(self._db_path()))
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        conn.close()
        self.assertEqual(count, 3)  # old-sess + sess1 + sess2, nothing lost or duplicated


class MigrationTransactionSafety(SavingsLedgerTestCase):
    """PR #94 review (Copilot): the version check, migration execution, and
    PRAGMA user_version write must be one atomic transaction (BEGIN
    IMMEDIATE ... COMMIT/ROLLBACK) -- a failure partway through must undo
    everything, including any ALTER TABLE a migration already ran, rather
    than leaving the schema half-migrated with a possibly-wrong recorded
    version. Also: a schema version with no registered migration must raise
    loudly, never silently advance user_version past it.

    The concurrent-process race itself (two connections both deciding to
    migrate before either commits) is verified separately, by direct
    reasoning + a live check that SQLite's BEGIN IMMEDIATE actually blocks a
    second connection's own BEGIN IMMEDIATE against the same file -- not
    re-tested here as a threaded/multi-process test, which would trade
    determinism for coverage of behavior SQLite itself already guarantees."""

    def _db_path(self) -> Path:
        return L.resolve_db_path()

    def test_missing_migration_entry_raises_and_rolls_back_everything(self):
        _old_schema_db_path(self._db_path())
        with mock.patch.object(L, "SCHEMA_VERSION", 5):
            # _MIGRATIONS only has entries for versions 1-4 -- version 5
            # is deliberately left unregistered to exercise the guard.
            with self.assertRaises(RuntimeError):
                L.finalize_session("sess1", "myproject")

        conn = sqlite3.connect(str(self._db_path()))
        # Rolled back entirely -- migrations 1-3 (which ran FIRST and would
        # have succeeded) must not have stuck, since the whole sequence is
        # one transaction.
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 0)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        self.assertNotIn("raw_tokens_sum", columns)
        conn.close()

    def test_migration_function_failure_rolls_back_its_own_alter(self):
        _old_schema_db_path(self._db_path())

        def _broken_migration(conn):
            conn.execute("ALTER TABLE sessions ADD COLUMN raw_tokens_sum INTEGER NOT NULL DEFAULT 0")
            raise RuntimeError("simulated failure partway through a migration")

        with mock.patch.object(L, "_MIGRATIONS", {1: _broken_migration}):
            with self.assertRaises(RuntimeError):
                L.finalize_session("sess1", "myproject")

        conn = sqlite3.connect(str(self._db_path()))
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 0)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        self.assertNotIn(
            "raw_tokens_sum", columns,
            "a migration's own ALTER TABLE must roll back if it fails partway through",
        )
        conn.close()

    def test_connection_is_closed_when_a_migration_fails(self):
        # _connect() opens the connection before running migrations, so a
        # failure there must close it rather than leak an open handle to the
        # db file (kept alive only by the exception's traceback). Invisible
        # on POSIX, but on Windows an open handle makes the file undeletable
        # -- this is what made the two tests above error in tearDown. Checked
        # directly here so it's a regression guard on every platform.
        _old_schema_db_path(self._db_path())
        opened = []
        real_connect = sqlite3.connect

        def _capturing_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        with mock.patch.object(sqlite3, "connect", _capturing_connect):
            with mock.patch.object(L, "SCHEMA_VERSION", 5):
                with self.assertRaises(RuntimeError):
                    L._connect()

        self.assertEqual(len(opened), 1)
        with self.assertRaises(sqlite3.ProgrammingError):
            opened[0].execute("SELECT 1")  # "Cannot operate on a closed database."

    def test_second_connection_cannot_interleave_a_migration_in_progress(self):
        # Direct check of the mechanism the real fix relies on: once one
        # connection holds BEGIN IMMEDIATE's write lock, a second connection
        # to the SAME file cannot also acquire it -- this is what serializes
        # two processes' _run_migrations calls instead of letting them
        # interleave their read-then-write into the race Copilot found.
        db_path = self._db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn1 = sqlite3.connect(str(db_path), timeout=0.2)
        conn1.execute("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
        conn1.execute("BEGIN IMMEDIATE")
        try:
            conn2 = sqlite3.connect(str(db_path), timeout=0.2)
            with self.assertRaises(sqlite3.OperationalError):
                conn2.execute("BEGIN IMMEDIATE")
            conn2.close()
        finally:
            conn1.execute("ROLLBACK")
            conn1.close()


class FormatDetailViewAcrossProjectsTotal(unittest.TestCase):
    """format_detail_view's "Across projects · all time" grand-total row
    (#95). Pure-function tests -- format_detail_view takes plain dicts/lists
    and returns a string, so no CLAUDE_RUNWAY_SAVINGS_DB fixture is needed
    (unlike the sqlite-backed query functions above).
    """

    # Minimal session_agg/project_summary shared by every case below --
    # only the all_projects list varies per test, so keep the rest fixed
    # and uninteresting.
    # Deliberately a different project name than any all_projects row below,
    # so "This session · <project>" never collides with a per-project-row
    # substring match in the assertions.
    _SESSION_AGG = {"credited_saved_tokens": 0, "event_count": 0, "raw_tokens_sum": 0, "out_tokens_sum": 0, "project": "current-session-project"}
    _PROJECT_SUMMARY = {"project": "current-session-project"}

    def _render(self, all_projects):
        return L.format_detail_view(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], all_projects)

    def test_multiple_projects_sum_correctly(self):
        out = self._render([
            {"project": "acme-a", "sessions": 10, "total_saved_tokens": 115300},
            {"project": "acme-b", "sessions": 8, "total_saved_tokens": 70700},
        ])
        lines = out.splitlines()
        total_line = [ln for ln in lines if ln.strip().startswith("Total")]
        self.assertEqual(len(total_line), 1)
        # 10 + 8 sessions, 115300 + 70700 = 186000 -> "~186.0k" -> "~186k"
        # per _fmt_tokens' own ".0k" collapse rule.
        self.assertIn("18 sessions", total_line[0])
        self.assertIn(L._fmt_tokens(186000), total_line[0])

    def test_single_project_total_matches_its_own_row(self):
        out = self._render([{"project": "acme-a", "sessions": 5, "total_saved_tokens": 42000}])
        lines = out.splitlines()
        project_line = [ln for ln in lines if "acme-a" in ln]
        total_line = [ln for ln in lines if ln.strip().startswith("Total")]
        self.assertEqual(len(project_line), 1)
        self.assertEqual(len(total_line), 1)
        self.assertIn("5 sessions", total_line[0])
        self.assertIn(L._fmt_tokens(42000), total_line[0])

    def test_no_all_projects_means_no_total_row(self):
        out = self._render([])
        self.assertNotIn("Across projects", out)
        self.assertNotIn("Total", out)


class PerToolRawTokensAccumulation(SavingsLedgerTestCase):
    """Issue #65: raw_tokens_sum, out_tokens_sum, and credited_event_count
    accumulated per tool so avg % reduction can be computed and displayed.
    Tests the accumulation logic in both finalize_session (DB path) and
    get_live_session_aggregate (JSONL path).
    """

    def test_finalize_session_accumulates_raw_and_out_tokens_per_tool(self):
        # Two credited events for the same tool -- raw and out must sum both.
        L.record_event("s1", "hook:Bash", raw_tokens=1000, out_tokens=100, credited=True)
        L.record_event("s1", "hook:Bash", raw_tokens=500, out_tokens=50, credited=True)
        L.finalize_session("s1", "proj")

        rows = L.query_session_tool_breakdown("s1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tool"], "hook:Bash")
        self.assertEqual(rows[0]["event_count"], 2)
        self.assertEqual(rows[0]["saved_tokens"], 1350)        # (1000-100) + (500-50)
        self.assertEqual(rows[0]["raw_tokens_sum"], 1500)      # 1000 + 500
        self.assertEqual(rows[0]["out_tokens_sum"], 150)       # 100 + 50
        self.assertEqual(rows[0]["credited_event_count"], 2)

    def test_finalize_session_does_not_credit_uncredited_raw_tokens(self):
        # fetch_url events are uncredited -- their raw/out must NOT feed into
        # the credited aggregates (the % would be meaningless).
        L.record_event("s2", "fetch_url", raw_tokens=2000, out_tokens=200, credited=False)
        L.record_event("s2", "hook:Bash", raw_tokens=800, out_tokens=80, credited=True)
        L.finalize_session("s2", "proj")

        rows = {r["tool"]: r for r in L.query_session_tool_breakdown("s2")}
        self.assertEqual(rows["fetch_url"]["raw_tokens_sum"], 0)      # uncredited -> 0
        self.assertEqual(rows["fetch_url"]["out_tokens_sum"], 0)
        self.assertEqual(rows["fetch_url"]["credited_event_count"], 0)
        self.assertEqual(rows["hook:Bash"]["raw_tokens_sum"], 800)
        self.assertEqual(rows["hook:Bash"]["out_tokens_sum"], 80)
        self.assertEqual(rows["hook:Bash"]["credited_event_count"], 1)

    def test_finalize_session_multiple_tools_tracked_independently(self):
        L.record_event("s3", "hook:Bash", raw_tokens=1000, out_tokens=100, credited=True)
        L.record_event("s3", "mcp:compress_command_output", raw_tokens=3000, out_tokens=450, credited=True)
        L.finalize_session("s3", "proj")

        rows = {r["tool"]: r for r in L.query_session_tool_breakdown("s3")}
        self.assertEqual(rows["hook:Bash"]["raw_tokens_sum"], 1000)
        self.assertEqual(rows["hook:Bash"]["out_tokens_sum"], 100)
        self.assertEqual(rows["mcp:compress_command_output"]["raw_tokens_sum"], 3000)
        self.assertEqual(rows["mcp:compress_command_output"]["out_tokens_sum"], 450)

    def test_get_live_session_aggregate_accumulates_all_credited_fields(self):
        # Same logic as finalize_session but reads from JSONL (mid-session).
        L.record_event("s4", "hook:Bash", raw_tokens=600, out_tokens=60, credited=True)
        L.record_event("s4", "hook:Bash", raw_tokens=400, out_tokens=40, credited=True)
        agg = L.get_live_session_aggregate("s4")

        bash_agg = agg["by_tool"]["hook:Bash"]
        self.assertEqual(bash_agg["raw_tokens_sum"], 1000)        # 600 + 400
        self.assertEqual(bash_agg["out_tokens_sum"], 100)         # 60 + 40
        self.assertEqual(bash_agg["saved_tokens"], 900)           # (600-60) + (400-40)
        self.assertEqual(bash_agg["credited_event_count"], 2)

    def test_get_live_session_aggregate_uncredited_does_not_add_raw_tokens(self):
        L.record_event("s5", "fetch_url", raw_tokens=5000, out_tokens=500, credited=False)
        agg = L.get_live_session_aggregate("s5")
        self.assertEqual(agg["by_tool"]["fetch_url"]["raw_tokens_sum"], 0)
        self.assertEqual(agg["by_tool"]["fetch_url"]["out_tokens_sum"], 0)
        self.assertEqual(agg["by_tool"]["fetch_url"]["credited_event_count"], 0)

    def test_credited_event_with_expanding_summary_uses_real_out_tokens(self):
        # A summary that expands (out > raw): saved_tokens is clamped to 0 by
        # record_event, but raw and out are still recorded faithfully on the
        # event. out_tokens_sum must reflect the REAL out (150), not raw-saved (100-0=100).
        L.record_event("s6", "hook:Bash", raw_tokens=100, out_tokens=150, credited=True)
        L.finalize_session("s6", "proj")

        rows = L.query_session_tool_breakdown("s6")
        self.assertEqual(rows[0]["saved_tokens"], 0)          # clamped at record_event
        self.assertEqual(rows[0]["raw_tokens_sum"], 100)
        self.assertEqual(rows[0]["out_tokens_sum"], 150)      # real out, not raw-saved
        self.assertEqual(rows[0]["credited_event_count"], 1)  # still credited


class PerToolAvgPctFormat(unittest.TestCase):
    """Issue #65: format_detail_view must show avg % reduction per credited
    tool. Pure-function tests -- no DB, no temp dir needed.
    """

    _SESSION_AGG = {
        "credited_saved_tokens": 0, "event_count": 0,
        "raw_tokens_sum": 0, "out_tokens_sum": 0, "project": "p",
    }
    _PROJECT_SUMMARY = {"project": "p"}

    def _render(self, tool_breakdown):
        return L.format_detail_view(
            self._SESSION_AGG, self._PROJECT_SUMMARY, tool_breakdown, [], []
        )

    def test_avg_pct_appears_for_credited_tool(self):
        # 1000 raw, 150 out -> 1 - 150/1000 = 85% reduction.
        breakdown = [{"tool": "hook:Bash", "event_count": 10, "saved_tokens": 850,
                      "raw_tokens_sum": 1000, "out_tokens_sum": 150, "credited_event_count": 10}]
        out = self._render(breakdown)
        # The percentage and "avg reduction" label must both appear on the tool line.
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertEqual(len(tool_lines), 1)
        self.assertIn("avg reduction", tool_lines[0])
        self.assertIn("85%", tool_lines[0])

    def test_avg_pct_zero_raw_does_not_crash(self):
        # Old DB rows backfilled with raw_tokens_sum=0 must display gracefully.
        # saved_tokens=200 ensures the credited branch is taken (heuristic fallback).
        breakdown = [{"tool": "hook:Bash", "event_count": 5, "saved_tokens": 200,
                      "raw_tokens_sum": 0, "out_tokens_sum": 0, "credited_event_count": 5}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertEqual(len(tool_lines), 1)
        # No % shown (denominator is 0), but the line must still appear.
        self.assertNotIn("avg reduction", tool_lines[0])

    def test_uncredited_tool_has_no_avg_pct(self):
        breakdown = [{"tool": "fetch_url", "event_count": 3, "saved_tokens": 0,
                      "raw_tokens_sum": 0, "out_tokens_sum": 0, "credited_event_count": 0}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "fetch_url" in ln]
        self.assertEqual(len(tool_lines), 1)
        self.assertIn("not credited", tool_lines[0])
        self.assertNotIn("avg reduction", tool_lines[0])

    def test_credited_tool_with_zero_savings_shows_0pct_not_uncredited(self):
        # A credited tool whose every summary EXPANDED (out > raw per event) --
        # saved_tokens is clamped to 0, but credited_event_count > 0 means the
        # tool IS credited; it should render "0% avg reduction" not "not credited".
        breakdown = [{"tool": "hook:Bash", "event_count": 2, "saved_tokens": 0,
                      "raw_tokens_sum": 200, "out_tokens_sum": 300, "credited_event_count": 2}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertEqual(len(tool_lines), 1)
        self.assertNotIn("not credited", tool_lines[0])
        self.assertIn("0% avg reduction", tool_lines[0])

    def test_avg_pct_uses_out_tokens_sum_not_raw_minus_saved(self):
        # Regression guard for the Copilot-found bug: out = raw - saved gives
        # the wrong % when any event had a clamped saved_tokens. Two events:
        #   event 1: raw=100, out=150 -> saved=0 (clamped) -> naive out = 100-0=100 (WRONG, real=150)
        #   event 2: raw=100, out=10  -> saved=90
        # Naive reconstructed out: (100+100)-(0+90)=110 -> 1-110/200 = 45%
        # Real out_tokens_sum: 150+10=160            -> 1-160/200 = 20%
        breakdown = [{"tool": "hook:Bash", "event_count": 2, "saved_tokens": 90,
                      "raw_tokens_sum": 200, "out_tokens_sum": 160, "credited_event_count": 2}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        # 1 - 160/200 = 20%
        self.assertIn("20%", tool_lines[0])
        self.assertNotIn("45%", tool_lines[0])

    def test_avg_pct_100_percent(self):
        # out_tokens_sum=0, credited_event_count=1 -> 100% reduction (real zero-output).
        breakdown = [{"tool": "hook:Bash", "event_count": 1, "saved_tokens": 500,
                      "raw_tokens_sum": 500, "out_tokens_sum": 0, "credited_event_count": 1}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertIn("100%", tool_lines[0])

    def test_v2_migrated_row_credited_but_no_pct_shown(self):
        # A v2→v3 migrated row: real saved_tokens and raw_tokens_sum (from v2),
        # but credited_event_count=0 and out_tokens_sum=0 (v3 backfill defaults).
        # The row IS credited (saved_tokens > 0 heuristic), but the avg % must
        # be SUPPRESSED -- computing 1 - 0/raw would yield a fabricated 100%.
        breakdown = [{"tool": "hook:Bash", "event_count": 5, "saved_tokens": 900,
                      "raw_tokens_sum": 1000, "out_tokens_sum": 0, "credited_event_count": 0}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertEqual(len(tool_lines), 1)
        # Must render as credited (not "not credited") but without a % metric.
        self.assertNotIn("not credited", tool_lines[0])
        self.assertNotIn("avg reduction", tool_lines[0])


class SchemaMigrationV2V3(SavingsLedgerTestCase):
    """Schema migrations: v2 adds raw_tokens_sum; v3 adds out_tokens_sum and
    credited_event_count to session_tools (issue #65 and follow-up).
    Mirrors the shape of existing SchemaMigration tests for v1.
    """

    def _db_path(self) -> Path:
        return L.resolve_db_path()

    def _old_v1_db(self) -> None:
        """Build a DB at schema v1 (session_tools without raw_tokens_sum, etc.)."""
        conn = sqlite3.connect(str(self._db_path()))
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                credited_saved_tokens INTEGER NOT NULL,
                raw_tokens_sum INTEGER NOT NULL DEFAULT 0,
                out_tokens_sum INTEGER NOT NULL DEFAULT 0,
                event_count INTEGER NOT NULL,
                fetch_url_count INTEGER NOT NULL,
                overhead_tokens INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE session_tools (
                session_id TEXT NOT NULL,
                tool TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                saved_tokens INTEGER NOT NULL,
                PRIMARY KEY (session_id, tool)
            )
            """
        )
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        # Mark it as at v1 so only v2 and v3 migrations run.
        conn.execute("PRAGMA user_version = 1")
        conn.execute("INSERT INTO session_tools VALUES ('old-sess', 'hook:Bash', 5, 300)")
        conn.commit()
        conn.close()

    def test_v2_v3_migrations_add_all_new_columns(self):
        self._old_v1_db()
        L.finalize_session("new-sess", "proj")

        conn = sqlite3.connect(str(self._db_path()))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(session_tools)").fetchall()}
        self.assertIn("raw_tokens_sum", columns)
        self.assertIn("out_tokens_sum", columns)
        self.assertIn("credited_event_count", columns)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        conn.close()

    def test_v2_v3_migrations_backfill_existing_rows_with_zero(self):
        self._old_v1_db()
        L.finalize_session("new-sess", "proj")

        conn = sqlite3.connect(str(self._db_path()))
        row = conn.execute(
            "SELECT event_count, saved_tokens, raw_tokens_sum, out_tokens_sum, credited_event_count "
            "FROM session_tools WHERE session_id = 'old-sess'"
        ).fetchone()
        conn.close()
        # Existing row must survive with new columns backfilled to 0 (honest "unknown").
        self.assertEqual(row, (5, 300, 0, 0, 0))

    def test_query_session_tool_breakdown_includes_all_new_fields(self):
        L.record_event("s-qbd", "hook:Bash", raw_tokens=1000, out_tokens=100, credited=True)
        L.finalize_session("s-qbd", "proj")

        rows = L.query_session_tool_breakdown("s-qbd")
        self.assertEqual(len(rows), 1)
        self.assertIn("raw_tokens_sum", rows[0])
        self.assertIn("out_tokens_sum", rows[0])
        self.assertIn("credited_event_count", rows[0])
        self.assertEqual(rows[0]["raw_tokens_sum"], 1000)
        self.assertEqual(rows[0]["out_tokens_sum"], 100)
        self.assertEqual(rows[0]["credited_event_count"], 1)


class QueryAllProjectsRicherStats(SavingsLedgerTestCase):
    """Issue #68: query_all_projects() must return avg_tokens_per_session and
    best_tool in addition to the existing {project, sessions, total_saved_tokens}
    -- pure DB-layer tests that exercise the SQL directly."""

    def test_avg_tokens_per_session_correct(self):
        # Two sessions for one project: 100 + 300 = 400 total → avg 200.
        L.record_event("s-avg-1", "hook:Bash", 1000, 900, credited=True)   # saved=100
        L.finalize_session("s-avg-1", "proj-a")
        L.record_event("s-avg-2", "hook:Bash", 1000, 700, credited=True)   # saved=300
        L.finalize_session("s-avg-2", "proj-a")

        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertIn("proj-a", rows)
        self.assertEqual(rows["proj-a"]["sessions"], 2)
        self.assertEqual(rows["proj-a"]["total_saved_tokens"], 400)
        self.assertEqual(rows["proj-a"]["avg_tokens_per_session"], 200)

    def test_best_tool_per_project(self):
        # hook:Bash saves 100, compress_command_output saves 300 -- best_tool must
        # pick compress_command_output regardless of event_count ordering.
        L.record_event("s-bt-1", "hook:Bash", 1000, 900, credited=True)              # saved=100
        L.record_event("s-bt-1", "mcp:compress_command_output", 1000, 700, credited=True)  # saved=300
        L.finalize_session("s-bt-1", "proj-b")

        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertEqual(rows["proj-b"]["best_tool"], "mcp:compress_command_output")

    def test_best_tool_none_when_no_session_tools_rows(self):
        # A project whose session was inserted directly into `sessions` without
        # any corresponding session_tools rows (e.g. a session from before the
        # per-tool breakdown schema existed) -- best_tool must be None, not error.
        #
        # Must initialize the schema first via L._connect(), then insert a bare
        # sessions row WITHOUT a matching session_tools row. Using finalize_session
        # would create session_tools rows, which defeats the purpose of this test.
        conn = L._connect()
        conn.execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, project, ended_at, credited_saved_tokens, raw_tokens_sum, "
            " out_tokens_sum, event_count, fetch_url_count, overhead_tokens) "
            "VALUES ('bare-sess', 'proj-bare', '2026-01-01T00:00:00Z', 50, 0, 0, 1, 0, 0)"
        )
        conn.commit()
        conn.close()

        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertIn("proj-bare", rows)
        self.assertIsNone(rows["proj-bare"]["best_tool"])

    def test_avg_is_zero_for_project_with_zero_savings(self):
        # A single session that saved 0 tokens (e.g. every summary expanded).
        L.finalize_session("s-zero", "proj-zero")
        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertEqual(rows["proj-zero"]["avg_tokens_per_session"], 0)

    def test_multiple_projects_independent(self):
        # Two different projects must not bleed into each other's avg or best_tool.
        L.record_event("s-p1", "hook:Bash", 1000, 500, credited=True)    # saved=500
        L.finalize_session("s-p1", "proj-x")
        L.record_event("s-p2", "mcp:compress_file", 1000, 100, credited=True)  # saved=900
        L.finalize_session("s-p2", "proj-y")

        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertEqual(rows["proj-x"]["avg_tokens_per_session"], 500)
        self.assertEqual(rows["proj-x"]["best_tool"], "hook:Bash")
        self.assertEqual(rows["proj-y"]["avg_tokens_per_session"], 900)
        self.assertEqual(rows["proj-y"]["best_tool"], "mcp:compress_file")

    def test_avg_uses_python_round_not_sql_round(self):
        # Regression guard for the Copilot-found rounding inconsistency: SQLite's
        # ROUND() uses half-away-from-zero (2.5 → 3), Python's round() uses
        # ties-to-even (2.5 → 2). The "Last N sessions" average uses Python's
        # round(); avg_tokens_per_session must also use Python's round() so
        # the two blocks never disagree on identical data. Two sessions:
        #   session 1: saved=2, session 2: saved=3 → sum=5, count=2 → 2.5
        # SQLite ROUND(2.5) = 3; Python round(2.5) = 2.
        L.record_event("s-rnd-1", "hook:Bash", 100, 98, credited=True)   # saved=2
        L.finalize_session("s-rnd-1", "proj-rnd")
        L.record_event("s-rnd-2", "hook:Bash", 100, 97, credited=True)   # saved=3
        L.finalize_session("s-rnd-2", "proj-rnd")

        rows = {r["project"]: r for r in L.query_all_projects()}
        # Python round(2.5) = 2 (ties-to-even); SQLite ROUND(2.5) = 3.
        self.assertEqual(rows["proj-rnd"]["avg_tokens_per_session"], round(2.5))


class FormatDetailViewCrossProjectRicherStats(unittest.TestCase):
    """Issue #68: format_detail_view must show avg_tokens_per_session and
    best_tool in the 'Across projects' block when present.
    Pure-function tests -- no DB or temp dir needed."""

    _SESSION_AGG = {
        "credited_saved_tokens": 0, "event_count": 0,
        "raw_tokens_sum": 0, "out_tokens_sum": 0,
        "project": "current-session-project",
    }
    _PROJECT_SUMMARY = {"project": "current-session-project"}

    def _render(self, all_projects):
        return L.format_detail_view(
            self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], all_projects
        )

    def test_avg_and_best_tool_shown_when_present(self):
        out = self._render([{
            "project": "acme-a", "sessions": 4, "total_saved_tokens": 80000,
            "avg_tokens_per_session": 20000, "best_tool": "hook:Bash",
        }])
        project_lines = [ln for ln in out.splitlines() if "acme-a" in ln]
        self.assertEqual(len(project_lines), 1)
        self.assertIn("avg", project_lines[0])
        self.assertIn("best: hook:Bash", project_lines[0])

    def test_best_tool_suppressed_when_none(self):
        # best_tool=None means no session_tools data -- must not render "best: None".
        out = self._render([{
            "project": "proj-bare", "sessions": 1, "total_saved_tokens": 50,
            "avg_tokens_per_session": 50, "best_tool": None,
        }])
        project_lines = [ln for ln in out.splitlines() if "proj-bare" in ln]
        self.assertEqual(len(project_lines), 1)
        self.assertNotIn("best:", project_lines[0])
        self.assertNotIn("None", project_lines[0])

    def test_avg_shown_even_when_best_tool_absent(self):
        out = self._render([{
            "project": "proj-c", "sessions": 2, "total_saved_tokens": 400,
            "avg_tokens_per_session": 200, "best_tool": None,
        }])
        project_lines = [ln for ln in out.splitlines() if "proj-c" in ln]
        self.assertEqual(len(project_lines), 1)
        self.assertIn("avg", project_lines[0])
        self.assertNotIn("best:", project_lines[0])

    def test_backward_compat_missing_keys_do_not_crash(self):
        # Old-shape dict without the new fields (e.g. tests that were written
        # before issue #68) must not raise -- .get() returns None, suppressing
        # both new fields gracefully.
        out = self._render([
            {"project": "acme-a", "sessions": 10, "total_saved_tokens": 115300},
        ])
        lines = out.splitlines()
        project_line = [ln for ln in lines if "acme-a" in ln]
        self.assertEqual(len(project_line), 1)
        # Old callers keep working -- no crash, no "None" visible in output.
        self.assertNotIn("None", project_line[0])

    def test_zero_avg_is_shown_not_suppressed(self):
        # avg_tokens_per_session=0 is a real value (zero savings), not absent --
        # must display rather than being treated as falsy and hidden.
        out = self._render([{
            "project": "proj-zero", "sessions": 1, "total_saved_tokens": 0,
            "avg_tokens_per_session": 0, "best_tool": None,
        }])
        project_lines = [ln for ln in out.splitlines() if "proj-zero" in ln]
        self.assertEqual(len(project_lines), 1)
        # 0 is not None -- avg suffix must still appear.
        self.assertIn("avg", project_lines[0])


class FormatJson(unittest.TestCase):
    """format_json must return valid JSON with the correct top-level structure.
    Pure-function tests -- no DB or temp dir needed.
    """

    # Shared minimal inputs; tests override only the part they care about.
    _SESSION_AGG = {
        "project": "my-repo",
        "credited_saved_tokens": 1234,
        "raw_tokens_sum": 5000,
        "out_tokens_sum": 3766,
        "event_count": 7,
        "fetch_url_count": 2,
        "overhead_tokens": 300,
    }
    _PROJECT_SUMMARY = {
        "project": "my-repo",
        "sessions": 10,
        "total_saved_tokens": 50000,
        "best_session_tokens": 8000,
    }

    def _render(self, session_agg=None, project_summary=None,
                tool_breakdown=None, last_n=None, all_projects=None):
        return L.format_json(
            session_agg or self._SESSION_AGG,
            project_summary or self._PROJECT_SUMMARY,
            tool_breakdown if tool_breakdown is not None else [],
            last_n if last_n is not None else [],
            all_projects if all_projects is not None else [],
        )

    def test_output_is_valid_json(self):
        out = self._render()
        parsed = json.loads(out)
        self.assertIsInstance(parsed, dict)

    def test_top_level_keys_are_present(self):
        parsed = json.loads(self._render())
        for key in ("session", "project_summary", "tool_breakdown", "last_n_sessions", "all_projects"):
            self.assertIn(key, parsed, f"Missing top-level key: {key}")

    def test_session_values_are_correct(self):
        parsed = json.loads(self._render())
        sess = parsed["session"]
        self.assertEqual(sess["project"], "my-repo")
        self.assertEqual(sess["credited_saved_tokens"], 1234)
        self.assertEqual(sess["raw_tokens_sum"], 5000)
        self.assertEqual(sess["event_count"], 7)
        self.assertEqual(sess["fetch_url_count"], 2)
        self.assertEqual(sess["overhead_tokens"], 300)

    def test_project_summary_values_are_correct(self):
        parsed = json.loads(self._render())
        ps = parsed["project_summary"]
        self.assertEqual(ps["project"], "my-repo")
        self.assertEqual(ps["sessions"], 10)
        self.assertEqual(ps["total_saved_tokens"], 50000)
        self.assertEqual(ps["best_session_tokens"], 8000)

    def test_tool_breakdown_serialised_correctly(self):
        breakdown = [
            {"tool": "hook:Bash", "event_count": 5, "saved_tokens": 900,
             "raw_tokens_sum": 1000, "out_tokens_sum": 100, "credited_event_count": 5},
        ]
        parsed = json.loads(self._render(tool_breakdown=breakdown))
        self.assertEqual(len(parsed["tool_breakdown"]), 1)
        t = parsed["tool_breakdown"][0]
        self.assertEqual(t["tool"], "hook:Bash")
        self.assertEqual(t["saved_tokens"], 900)
        self.assertEqual(t["raw_tokens_sum"], 1000)

    def test_last_n_sessions_serialised_correctly(self):
        last_n = [
            {"ended_at": "2026-09-01T10:00:00Z", "saved_tokens": 500},
            {"ended_at": "2026-09-02T10:00:00Z", "saved_tokens": 800},
        ]
        parsed = json.loads(self._render(last_n=last_n))
        self.assertEqual(len(parsed["last_n_sessions"]), 2)
        self.assertEqual(parsed["last_n_sessions"][0]["saved_tokens"], 500)
        self.assertEqual(parsed["last_n_sessions"][1]["ended_at"], "2026-09-02T10:00:00Z")

    def test_all_projects_best_tool_none_stays_null(self):
        # best_tool=None must round-trip as JSON null, not the string "None".
        all_projects = [
            {"project": "proj-bare", "sessions": 1, "total_saved_tokens": 50,
             "avg_tokens_per_session": 50, "best_tool": None},
        ]
        parsed = json.loads(self._render(all_projects=all_projects))
        self.assertIsNone(parsed["all_projects"][0]["best_tool"])

    def test_all_projects_best_tool_present_is_a_string(self):
        all_projects = [
            {"project": "proj-a", "sessions": 3, "total_saved_tokens": 300,
             "avg_tokens_per_session": 100, "best_tool": "hook:Bash"},
        ]
        parsed = json.loads(self._render(all_projects=all_projects))
        self.assertEqual(parsed["all_projects"][0]["best_tool"], "hook:Bash")

    def test_none_values_in_session_agg_are_coerced_to_zero(self):
        # A partially-filled aggregate (e.g. from a brand-new session with no
        # events yet) must not produce null in the JSON output.
        sparse_agg = {"project": "p", "credited_saved_tokens": None}
        parsed = json.loads(self._render(session_agg=sparse_agg))
        self.assertEqual(parsed["session"]["credited_saved_tokens"], 0)

    def test_empty_lists_produce_empty_json_arrays(self):
        parsed = json.loads(self._render())
        self.assertEqual(parsed["tool_breakdown"], [])
        self.assertEqual(parsed["last_n_sessions"], [])
        self.assertEqual(parsed["all_projects"], [])


class FormatCsv(unittest.TestCase):
    """format_csv must return RFC-4180 CSV with a header row and correct data.
    Pure-function tests -- no DB or temp dir needed.
    """

    _SESSION_AGG = {
        "project": "my-repo",
        "credited_saved_tokens": 1234,
        "raw_tokens_sum": 5000,
        "out_tokens_sum": 3766,
        "event_count": 7,
        "fetch_url_count": 2,
        "overhead_tokens": 300,
    }
    _PROJECT_SUMMARY = {
        "project": "my-repo",
        "sessions": 10,
        "total_saved_tokens": 50000,
        "best_session_tokens": 8000,
    }

    def _parse(self, text: str) -> list:
        """Parse a CSV string into a list-of-lists via stdlib csv."""
        return list(_csv_mod.reader(_io_mod.StringIO(text)))

    def test_session_table_has_header_and_one_data_row(self):
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], [], table="session")
        rows = self._parse(out)
        self.assertEqual(len(rows), 2)  # header + 1 data row

    def test_session_table_header_contains_expected_columns(self):
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], [], table="session")
        rows = self._parse(out)
        header = rows[0]
        self.assertIn("project", header)
        self.assertIn("credited_saved_tokens", header)
        self.assertIn("project_total_saved_tokens", header)

    def test_session_table_data_matches_inputs(self):
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], [], table="session")
        rows = self._parse(out)
        header, data = rows[0], rows[1]
        d = dict(zip(header, data))
        self.assertEqual(d["project"], "my-repo")
        self.assertEqual(int(d["credited_saved_tokens"]), 1234)
        self.assertEqual(int(d["project_total_saved_tokens"]), 50000)
        self.assertEqual(int(d["project_sessions"]), 10)

    def test_tools_table_has_header_and_correct_number_of_rows(self):
        breakdown = [
            {"tool": "hook:Bash", "event_count": 5, "saved_tokens": 900,
             "raw_tokens_sum": 1000, "out_tokens_sum": 100, "credited_event_count": 5},
            {"tool": "fetch_url", "event_count": 3, "saved_tokens": 0,
             "raw_tokens_sum": 0, "out_tokens_sum": 0, "credited_event_count": 0},
        ]
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, breakdown, [], [], table="tools")
        rows = self._parse(out)
        # 1 header + 2 data rows
        self.assertEqual(len(rows), 3)
        self.assertIn("tool", rows[0])

    def test_tools_table_data_matches_breakdown(self):
        breakdown = [
            {"tool": "hook:Bash", "event_count": 5, "saved_tokens": 900,
             "raw_tokens_sum": 1000, "out_tokens_sum": 100, "credited_event_count": 5},
        ]
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, breakdown, [], [], table="tools")
        rows = self._parse(out)
        header, data = rows[0], rows[1]
        d = dict(zip(header, data))
        self.assertEqual(d["tool"], "hook:Bash")
        self.assertEqual(int(d["saved_tokens"]), 900)
        self.assertEqual(int(d["credited_event_count"]), 5)

    def test_projects_table_has_header_and_correct_number_of_rows(self):
        all_projects = [
            {"project": "proj-a", "sessions": 3, "total_saved_tokens": 300,
             "avg_tokens_per_session": 100, "best_tool": "hook:Bash"},
            {"project": "proj-b", "sessions": 1, "total_saved_tokens": 50,
             "avg_tokens_per_session": 50, "best_tool": None},
        ]
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], all_projects, table="projects")
        rows = self._parse(out)
        self.assertEqual(len(rows), 3)  # header + 2 data rows

    def test_projects_table_none_best_tool_is_empty_string(self):
        # None best_tool must render as an empty CSV field, not "None".
        all_projects = [
            {"project": "proj-bare", "sessions": 1, "total_saved_tokens": 50,
             "avg_tokens_per_session": 50, "best_tool": None},
        ]
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], all_projects, table="projects")
        rows = self._parse(out)
        header, data = rows[0], rows[1]
        d = dict(zip(header, data))
        self.assertEqual(d["best_tool"], "")
        self.assertNotIn("None", out)

    def test_default_table_is_session(self):
        # Calling format_csv without the table kwarg must default to "session".
        out_explicit = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], [], table="session")
        out_default = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], [])
        self.assertEqual(out_explicit, out_default)

    def test_unknown_table_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], [], table="invalid")
        self.assertIn("invalid", str(ctx.exception))
        self.assertIn("session", str(ctx.exception))

    def test_empty_tools_table_has_only_header(self):
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], [], table="tools")
        rows = self._parse(out)
        self.assertEqual(len(rows), 1)  # header only, no data rows
        self.assertIn("tool", rows[0])

    def test_values_with_commas_are_quoted(self):
        # A project name containing a comma must be quoted in the CSV output --
        # verifying stdlib csv handles this rather than producing broken output.
        agg = dict(self._SESSION_AGG, project="repo,with,commas")
        ps = dict(self._PROJECT_SUMMARY, project="repo,with,commas")
        out = L.format_csv(agg, ps, [], [], [], table="session")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1]))
        self.assertEqual(d["project"], "repo,with,commas")

    def test_formula_injection_prefix_on_equals_leading_project(self):
        # A project name starting with '=' would be evaluated as a spreadsheet
        # formula if left unchanged. _safe_str must prefix it with an apostrophe.
        agg = dict(self._SESSION_AGG, project="=DANGEROUS()")
        ps = dict(self._PROJECT_SUMMARY, project="=DANGEROUS()")
        out = L.format_csv(agg, ps, [], [], [], table="session")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1]))
        # Apostrophe prefix applied -- the raw cell value now starts with "'".
        self.assertTrue(d["project"].startswith("'"), d["project"])
        self.assertNotEqual(d["project"][0], "=")

    def test_formula_injection_prefix_on_plus_leading_tool(self):
        # Tool names could in principle start with '+' (e.g. a malformed event).
        breakdown = [
            {"tool": "+BAD", "event_count": 1, "saved_tokens": 10,
             "raw_tokens_sum": 100, "out_tokens_sum": 90, "credited_event_count": 1},
        ]
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, breakdown, [], [], table="tools")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1]))
        self.assertTrue(d["tool"].startswith("'"), d["tool"])

    def test_safe_str_does_not_modify_normal_values(self):
        # Ordinary project/tool names must pass through unchanged.
        agg = dict(self._SESSION_AGG, project="my-normal-repo")
        ps = dict(self._PROJECT_SUMMARY, project="my-normal-repo")
        out = L.format_csv(agg, ps, [], [], [], table="session")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1]))
        self.assertEqual(d["project"], "my-normal-repo")

    def test_formula_injection_prefix_on_whitespace_prefixed_formula(self):
        # A tab or space before '=' bypasses a naive s[0] check because
        # spreadsheet applications strip leading whitespace before formula
        # detection. The guard must check the FIRST NON-WHITESPACE character.
        agg = dict(self._SESSION_AGG, project="\t=FORMULA()")
        ps = dict(self._PROJECT_SUMMARY, project="\t=FORMULA()")
        out = L.format_csv(agg, ps, [], [], [], table="session")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1]))
        # Apostrophe must be prepended even though the first char is \t.
        self.assertTrue(d["project"].startswith("'"), f"Expected apostrophe prefix, got: {d['project']!r}")

    def test_formula_injection_prefix_space_prefixed_formula(self):
        # Space before '+' -- same bypass vector as tab.
        agg = dict(self._SESSION_AGG, project=" +1+1")
        ps = dict(self._PROJECT_SUMMARY, project=" +1+1")
        out = L.format_csv(agg, ps, [], [], [], table="session")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1]))
        self.assertTrue(d["project"].startswith("'"), f"Expected apostrophe prefix, got: {d['project']!r}")


class QueryTrend(SavingsLedgerTestCase):
    """Issue #67: query_trend groups sessions by day or week bucket."""

    def test_weekly_buckets_group_same_week_together(self):
        # Two sessions on 2026-09-07 (Mon) and 2026-09-09 (Wed) are in the
        # same ISO week (2026-W37) -- they must land in one bucket.
        conn = L._connect()
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, project, ended_at, credited_saved_tokens, "
            " raw_tokens_sum, out_tokens_sum, event_count, fetch_url_count, overhead_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s-w1", "proj-w", "2026-09-07T09:00:00Z", 100, 1000, 900, 1, 0, 0),
        )
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, project, ended_at, credited_saved_tokens, "
            " raw_tokens_sum, out_tokens_sum, event_count, fetch_url_count, overhead_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s-w2", "proj-w", "2026-09-09T17:00:00Z", 200, 1000, 800, 1, 0, 0),
        )
        conn.commit()
        conn.close()

        rows = L.query_trend("proj-w", bucket="week", n=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sessions"], 2)
        self.assertEqual(rows[0]["total_saved_tokens"], 300)

    def test_weekly_buckets_iso_year_boundary_not_split(self):
        # Regression guard for the %Y-W%W bug Copilot identified in PR #157:
        # 2024-12-30 (Mon) and 2025-01-01 (Wed) are BOTH in ISO week 2025-W01
        # (the week containing the first Thursday of 2025). The broken %Y/%W
        # format would produce 2024-W53 and 2025-W00, splitting them into two
        # buckets. The correct %G/%V format must produce one bucket for both.
        conn = L._connect()
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, project, ended_at, credited_saved_tokens, "
            " raw_tokens_sum, out_tokens_sum, event_count, fetch_url_count, overhead_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s-boundary-1", "proj-boundary", "2024-12-30T10:00:00Z", 100, 1000, 900, 1, 0, 0),
        )
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, project, ended_at, credited_saved_tokens, "
            " raw_tokens_sum, out_tokens_sum, event_count, fetch_url_count, overhead_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s-boundary-2", "proj-boundary", "2025-01-01T14:00:00Z", 200, 1000, 800, 1, 0, 0),
        )
        conn.commit()
        conn.close()

        rows = L.query_trend("proj-boundary", bucket="week", n=10)
        # Both dates are in ISO week 2025-W01 -- must produce exactly ONE bucket.
        self.assertEqual(len(rows), 1, f"Expected 1 bucket (ISO 2025-W01) but got {len(rows)}: {rows}")
        self.assertEqual(rows[0]["sessions"], 2)
        self.assertEqual(rows[0]["total_saved_tokens"], 300)
        # Bucket key must reflect the ISO week year (2025), not the calendar year
        # of the first session (2024).
        self.assertTrue(
            rows[0]["bucket"].startswith("2025-W"),
            f"Expected ISO week-year 2025, got: {rows[0]['bucket']!r}",
        )

    def test_daily_buckets_keep_different_days_separate(self):
        conn = L._connect()
        for session_id, day, saved in (
            ("s-d1", "2026-09-07", 100),
            ("s-d2", "2026-09-08", 200),
            ("s-d3", "2026-09-09", 300),
        ):
            conn.execute(
                "INSERT INTO sessions "
                "(session_id, project, ended_at, credited_saved_tokens, "
                " raw_tokens_sum, out_tokens_sum, event_count, fetch_url_count, overhead_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, "proj-d", f"{day}T12:00:00Z", saved, 1000, 1000 - saved, 1, 0, 0),
            )
        conn.commit()
        conn.close()

        rows = L.query_trend("proj-d", bucket="day", n=10)
        self.assertEqual(len(rows), 3)
        # Oldest first (ascending).
        self.assertEqual(rows[0]["bucket"], "2026-09-07")
        self.assertEqual(rows[0]["total_saved_tokens"], 100)
        self.assertEqual(rows[2]["bucket"], "2026-09-09")
        self.assertEqual(rows[2]["total_saved_tokens"], 300)

    def test_n_caps_returned_buckets(self):
        # Insert 5 sessions on 5 different days; n=3 must return only the
        # 3 most recent buckets (still in ascending order).
        conn = L._connect()
        for i in range(5):
            conn.execute(
                "INSERT INTO sessions "
                "(session_id, project, ended_at, credited_saved_tokens, "
                " raw_tokens_sum, out_tokens_sum, event_count, fetch_url_count, overhead_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"s-cap-{i}", "proj-cap", f"2026-09-0{i + 1}T12:00:00Z", (i + 1) * 100, 1000, 900, 1, 0, 0),
            )
        conn.commit()
        conn.close()

        rows = L.query_trend("proj-cap", bucket="day", n=3)
        self.assertEqual(len(rows), 3)
        # Must be the 3 most recent days: 03, 04, 05.
        self.assertEqual(rows[0]["bucket"], "2026-09-03")
        self.assertEqual(rows[2]["bucket"], "2026-09-05")

    def test_empty_project_returns_empty_list(self):
        rows = L.query_trend("no-such-project", bucket="week", n=10)
        self.assertEqual(rows, [])

    def test_unknown_bucket_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            L.query_trend("proj", bucket="month")
        self.assertIn("month", str(ctx.exception))

    def test_different_projects_do_not_bleed(self):
        # One session for each of two projects on the same day -- they must
        # not appear in each other's trend results.
        conn = L._connect()
        for proj in ("proj-alpha", "proj-beta"):
            conn.execute(
                "INSERT INTO sessions "
                "(session_id, project, ended_at, credited_saved_tokens, "
                " raw_tokens_sum, out_tokens_sum, event_count, fetch_url_count, overhead_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"s-{proj}", proj, "2026-09-07T12:00:00Z", 500, 1000, 500, 1, 0, 0),
            )
        conn.commit()
        conn.close()

        alpha = L.query_trend("proj-alpha", bucket="day", n=10)
        beta = L.query_trend("proj-beta", bucket="day", n=10)
        self.assertEqual(len(alpha), 1)
        self.assertEqual(alpha[0]["total_saved_tokens"], 500)
        self.assertEqual(len(beta), 1)
        self.assertEqual(beta[0]["total_saved_tokens"], 500)


class FormatTrendView(unittest.TestCase):
    """Issue #67: format_trend_view must render a sparkline + table for the
    bucketed trend data returned by query_trend.
    Pure-function tests — no DB or temp dir needed.
    """

    def _rows(self, values, prefix="2026-W3"):
        """Build synthetic weekly trend rows from a list of token-saved values."""
        return [
            {"bucket": f"{prefix}{i}", "sessions": 1, "total_saved_tokens": v}
            for i, v in enumerate(values, 1)
        ]

    def test_empty_rows_returns_no_history_message(self):
        out = L.format_trend_view([], "my-repo", bucket="week")
        self.assertIn("No session history", out)
        self.assertIn("my-repo", out)

    def test_header_shows_project_and_bucket_type(self):
        out = L.format_trend_view(self._rows([100, 200]), "my-repo", bucket="week")
        self.assertIn("Weekly", out)
        self.assertIn("my-repo", out)

    def test_daily_header_says_daily(self):
        rows = [{"bucket": "2026-09-07", "sessions": 1, "total_saved_tokens": 100}]
        out = L.format_trend_view(rows, "my-repo", bucket="day")
        self.assertIn("Daily", out)

    def test_each_bucket_appears_in_output(self):
        rows = [
            {"bucket": "2026-W36", "sessions": 2, "total_saved_tokens": 300},
            {"bucket": "2026-W37", "sessions": 3, "total_saved_tokens": 700},
        ]
        out = L.format_trend_view(rows, "my-repo", bucket="week")
        self.assertIn("2026-W36", out)
        self.assertIn("2026-W37", out)

    def test_total_row_sums_correctly(self):
        rows = [
            {"bucket": "2026-W36", "sessions": 2, "total_saved_tokens": 300},
            {"bucket": "2026-W37", "sessions": 3, "total_saved_tokens": 700},
        ]
        out = L.format_trend_view(rows, "my-repo", bucket="week")
        lines = out.splitlines()
        total_lines = [ln for ln in lines if "Total" in ln]
        self.assertEqual(len(total_lines), 1)
        # 2 + 3 = 5 sessions total; 300 + 700 = 1000 tokens.
        self.assertIn("5", total_lines[0])
        self.assertIn(L._fmt_tokens(1000), total_lines[0])

    def test_sparkline_is_present(self):
        # A sparkline character must appear somewhere in the output.
        out = L.format_trend_view(self._rows([100, 500, 300]), "my-repo")
        spark_chars = set(L._SPARK_CHARS)
        self.assertTrue(
            any(c in spark_chars for c in out),
            "Expected at least one sparkline character in the trend output",
        )

    def test_single_bucket_does_not_crash(self):
        out = L.format_trend_view(
            [{"bucket": "2026-W36", "sessions": 1, "total_saved_tokens": 500}],
            "my-repo",
        )
        self.assertIn("2026-W36", out)
        self.assertIn("Total", out)

    def test_bucket_count_shown_in_summary_line(self):
        rows = self._rows([100, 200, 300])
        out = L.format_trend_view(rows, "my-repo", bucket="week")
        # The summary line (e.g. "(3 weeks · 3 sessions)") must show the count.
        self.assertIn("3 week", out)


class ParseTranscriptTokenCounts(unittest.TestCase):
    """parse_transcript_token_counts: happy path, failure modes, edge cases."""

    def _make_transcript(self, entries: list) -> str:
        """Build a JSONL transcript string from a list of raw dicts."""
        import tempfile, os as _os
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
        f.close()
        return f.name

    def _assistant(self, inp=0, out=0, cr=0, cw=0):
        return {
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cache_read_input_tokens": cr,
                    "cache_creation_input_tokens": cw,
                }
            }
        }

    def test_none_path_returns_none(self):
        self.assertIsNone(L.parse_transcript_token_counts(None))

    def test_missing_file_returns_none(self):
        self.assertIsNone(L.parse_transcript_token_counts("/nonexistent/path.jsonl"))

    def test_sums_all_assistant_turns(self):
        path = self._make_transcript([
            self._assistant(inp=10, out=5, cr=100, cw=20),
            self._assistant(inp=3,  out=8, cr=50,  cw=0),
        ])
        try:
            result = L.parse_transcript_token_counts(path)
            self.assertIsNotNone(result)
            self.assertEqual(result["input"],      13)
            self.assertEqual(result["output"],     13)
            self.assertEqual(result["cache_read"], 150)
            self.assertEqual(result["cache_write"], 20)
        finally:
            import os as _os; _os.unlink(path)

    def test_skips_non_assistant_entries(self):
        path = self._make_transcript([
            {"type": "user", "message": {"usage": {"input_tokens": 9999}}},
            self._assistant(inp=5, out=2, cr=10, cw=3),
        ])
        try:
            result = L.parse_transcript_token_counts(path)
            self.assertIsNotNone(result)
            self.assertEqual(result["input"], 5)  # user entry must not contribute
        finally:
            import os as _os; _os.unlink(path)

    def test_malformed_lines_skipped_valid_lines_counted(self):
        path = self._make_transcript([self._assistant(inp=7, out=3, cr=20, cw=5)])
        # Prepend a bad line by writing the file manually.
        with open(path, "r+", encoding="utf-8") as f:
            content = f.read()
            f.seek(0)
            f.write("not valid json\n" + content)
        try:
            result = L.parse_transcript_token_counts(path)
            self.assertIsNotNone(result)
            self.assertEqual(result["input"], 7)
        finally:
            import os as _os; _os.unlink(path)

    def test_no_assistant_entries_returns_none(self):
        path = self._make_transcript([
            {"type": "user", "message": {"content": "hello"}},
            {"type": "mode", "mode": "normal"},
        ])
        try:
            self.assertIsNone(L.parse_transcript_token_counts(path))
        finally:
            import os as _os; _os.unlink(path)

    def test_assistant_entry_missing_usage_returns_none(self):
        path = self._make_transcript([
            {"type": "assistant", "message": {"content": "no usage block"}},
        ])
        try:
            self.assertIsNone(L.parse_transcript_token_counts(path))
        finally:
            import os as _os; _os.unlink(path)

    def test_empty_file_returns_none(self):
        import tempfile, os as _os
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.close()
        try:
            self.assertIsNone(L.parse_transcript_token_counts(f.name))
        finally:
            _os.unlink(f.name)


class SchemaMigrationV4(SavingsLedgerTestCase):
    """Schema migration v4: actual_* token columns added to sessions (issue #165)."""

    def _db_path(self) -> Path:
        return L.resolve_db_path()

    def _v3_db(self) -> None:
        """Build a DB at schema v3 (sessions without actual_* columns)."""
        conn = sqlite3.connect(str(self._db_path()))
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                credited_saved_tokens INTEGER NOT NULL,
                raw_tokens_sum INTEGER NOT NULL DEFAULT 0,
                out_tokens_sum INTEGER NOT NULL DEFAULT 0,
                event_count INTEGER NOT NULL,
                fetch_url_count INTEGER NOT NULL,
                overhead_tokens INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute("CREATE TABLE IF NOT EXISTS session_tools "
                     "(session_id TEXT NOT NULL, tool TEXT NOT NULL, event_count INTEGER NOT NULL, "
                     "saved_tokens INTEGER NOT NULL, raw_tokens_sum INTEGER NOT NULL DEFAULT 0, "
                     "out_tokens_sum INTEGER NOT NULL DEFAULT 0, "
                     "credited_event_count INTEGER NOT NULL DEFAULT 0, "
                     "PRIMARY KEY (session_id, tool))")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("PRAGMA user_version = 3")
        conn.execute(
            "INSERT INTO sessions VALUES ('old-sess', 'old-proj', '2026-01-01T00:00:00Z', 42, 0, 0, 1, 0, 0)"
        )
        conn.commit()
        conn.close()

    def test_v4_migration_adds_all_four_columns(self):
        self._v3_db()
        L.finalize_session("new-sess", "proj")

        conn = sqlite3.connect(str(self._db_path()))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        conn.close()
        for col in ("actual_input_tokens", "actual_output_tokens",
                    "actual_cache_read_tokens", "actual_cache_write_tokens"):
            self.assertIn(col, columns, f"Missing column after v4 migration: {col}")

    def test_v4_migration_backfills_existing_rows_with_zero(self):
        self._v3_db()
        L.finalize_session("new-sess", "proj")

        conn = sqlite3.connect(str(self._db_path()))
        row = conn.execute(
            "SELECT actual_input_tokens, actual_output_tokens, "
            "actual_cache_read_tokens, actual_cache_write_tokens "
            "FROM sessions WHERE session_id = 'old-sess'"
        ).fetchone()
        conn.close()
        self.assertEqual(row, (0, 0, 0, 0))

    def test_finalize_session_stores_actual_tokens(self):
        actual = {"input": 100, "output": 50, "cache_read": 2000, "cache_write": 300}
        L.finalize_session("s-at", "proj", actual_tokens=actual)

        conn = L._connect()
        row = conn.execute(
            "SELECT actual_input_tokens, actual_output_tokens, "
            "actual_cache_read_tokens, actual_cache_write_tokens "
            "FROM sessions WHERE session_id = 's-at'"
        ).fetchone()
        conn.close()
        self.assertEqual(row, (100, 50, 2000, 300))

    def test_finalize_session_no_actual_tokens_stores_zeros(self):
        L.finalize_session("s-no-at", "proj")

        conn = L._connect()
        row = conn.execute(
            "SELECT actual_input_tokens, actual_output_tokens, "
            "actual_cache_read_tokens, actual_cache_write_tokens "
            "FROM sessions WHERE session_id = 's-no-at'"
        ).fetchone()
        conn.close()
        self.assertEqual(row, (0, 0, 0, 0))

    def test_finalize_session_return_dict_includes_actual_fields(self):
        actual = {"input": 10, "output": 5, "cache_read": 200, "cache_write": 30}
        result = L.finalize_session("s-ret", "proj", actual_tokens=actual)
        self.assertEqual(result["actual_input_tokens"],       10)
        self.assertEqual(result["actual_output_tokens"],       5)
        self.assertEqual(result["actual_cache_read_tokens"], 200)
        self.assertEqual(result["actual_cache_write_tokens"],  30)

    def test_schema_version_is_4(self):
        self.assertEqual(L.SCHEMA_VERSION, 4)


class FormatActualTokenBlock(unittest.TestCase):
    """_format_actual_token_block: display block for Anthropic-processed token counts."""

    def _agg(self, inp=0, out=0, cr=0, cw=0, saved=0):
        return {
            "actual_input_tokens":       inp,
            "actual_output_tokens":      out,
            "actual_cache_read_tokens":  cr,
            "actual_cache_write_tokens": cw,
            "credited_saved_tokens":     saved,
        }

    def test_returns_empty_when_all_zero(self):
        self.assertEqual(L._format_actual_token_block(self._agg()), [])

    def test_block_present_when_any_nonzero(self):
        lines = L._format_actual_token_block(self._agg(inp=100))
        self.assertTrue(len(lines) > 0)
        self.assertTrue(any("Tokens Anthropic processed" in ln for ln in lines))

    def test_project_name_in_header(self):
        lines = L._format_actual_token_block(self._agg(inp=100), project="my-repo")
        self.assertTrue(any("my-repo" in ln for ln in lines))

    def test_all_four_token_types_shown(self):
        lines = L._format_actual_token_block(self._agg(inp=10, out=5, cr=200, cw=30))
        joined = "\n".join(lines)
        self.assertIn("Fresh input",  joined)
        self.assertIn("Cache writes", joined)
        self.assertIn("Cache reads",  joined)
        self.assertIn("Output",       joined)

    def test_relative_cost_legend_always_shown(self):
        lines = L._format_actual_token_block(self._agg(inp=10, out=5, cr=200, cw=30))
        joined = "\n".join(lines)
        self.assertIn("1.00×", joined)
        self.assertIn("1.25×", joined)
        self.assertIn("0.10×", joined)
        self.assertIn("5.00×", joined)

    def test_savings_rate_shown_when_nonzero_saved(self):
        lines = L._format_actual_token_block(self._agg(inp=100, out=50, cr=1000, cw=200, saved=300))
        joined = "\n".join(lines)
        self.assertIn("Local compression avoided", joined)
        self.assertIn("of total processed", joined)

    def test_savings_rate_absent_when_saved_is_zero(self):
        lines = L._format_actual_token_block(self._agg(inp=100, out=50, cr=1000, cw=200, saved=0))
        joined = "\n".join(lines)
        self.assertNotIn("Local compression avoided", joined)

    def test_cache_efficiency_shown_when_input_nonzero(self):
        # cache_read=700, input=100 -> ratio=7
        lines = L._format_actual_token_block(self._agg(inp=100, cr=700))
        joined = "\n".join(lines)
        self.assertIn("Cache efficiency", joined)
        self.assertIn("7×", joined)

    def test_cache_efficiency_absent_when_input_zero(self):
        # No fresh input -> ratio undefined -> must not show or divide by zero.
        lines = L._format_actual_token_block(self._agg(cr=500))
        joined = "\n".join(lines)
        self.assertNotIn("Cache efficiency", joined)

    def test_percentages_sum_to_100(self):
        # Verify that the four % values written into the block sum to 100.0%
        # (allowing for rounding). Each percentage is computed as val/total*100.
        lines = L._format_actual_token_block(self._agg(inp=10, out=5, cr=80, cw=5))
        # 10+5+80+5=100 -> all pcts are clean integers; just check the Total line.
        self.assertTrue(any("100.0%" in ln for ln in lines))

    def test_fmt_tokens_millions(self):
        # _fmt_tokens must format values >= 1_000_000 as M, not as a huge k count.
        self.assertEqual(L._fmt_tokens(1_000_000), "~1M")
        self.assertEqual(L._fmt_tokens(110_600_000), "~110.6M")
        self.assertEqual(L._fmt_tokens(5_000_000), "~5M")


class ActualTokensReadPath(SavingsLedgerTestCase):
    """Issue #167: actual_* token counts written by finalize_session must be
    readable via query_last_n_sessions and visible in format_detail_view /
    format_json in a subsequent session.

    DB-backed tests use the same SavingsLedgerTestCase temp-dir fixture.
    format_detail_view / format_json tests are pure-function (no DB needed)
    and are included here for grouping -- they call the functions directly
    with hand-crafted last_n lists.
    """

    # Shared session_agg with all-zero actual_* (live session, always 0).
    _SESSION_AGG = {
        "credited_saved_tokens": 0, "event_count": 0,
        "raw_tokens_sum": 0, "out_tokens_sum": 0,
        "project": "test-proj",
        "actual_input_tokens": 0, "actual_output_tokens": 0,
        "actual_cache_read_tokens": 0, "actual_cache_write_tokens": 0,
    }
    _PROJECT_SUMMARY = {"project": "test-proj"}

    # A last_n entry with real actual_* data -- the shape query_last_n_sessions
    # returns after the fix.
    _SESSION_WITH_ACTUALS = {
        "ended_at": "2026-09-10T19:00:00Z",
        "saved_tokens": 1800,
        "actual_input_tokens": 1000,
        "actual_output_tokens": 200,
        "actual_cache_read_tokens": 5000,
        "actual_cache_write_tokens": 100,
    }
    _SESSION_ALL_ZEROS = {
        "ended_at": "2026-09-08T15:00:00Z",
        "saved_tokens": 7200,
        "actual_input_tokens": 0,
        "actual_output_tokens": 0,
        "actual_cache_read_tokens": 0,
        "actual_cache_write_tokens": 0,
    }

    # --- DB-layer tests ---

    def test_query_last_n_sessions_includes_actual_tokens(self):
        # finalize_session stores actual_* via the actual_tokens kwarg;
        # query_last_n_sessions must return them.
        L.finalize_session(
            "s-act", "proj-act",
            actual_tokens={"input": 1000, "output": 200, "cache_read": 5000, "cache_write": 100},
        )
        rows = L.query_last_n_sessions("proj-act")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["actual_input_tokens"], 1000)
        self.assertEqual(r["actual_output_tokens"], 200)
        self.assertEqual(r["actual_cache_read_tokens"], 5000)
        self.assertEqual(r["actual_cache_write_tokens"], 100)

    def test_query_last_n_sessions_zeros_when_no_transcript_parsing(self):
        # Sessions without CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS land all-zero;
        # the query must return them as 0 (not None or absent).
        L.finalize_session("s-noact", "proj-noact")
        rows = L.query_last_n_sessions("proj-noact")
        r = rows[0]
        for key in ("actual_input_tokens", "actual_output_tokens",
                    "actual_cache_read_tokens", "actual_cache_write_tokens"):
            self.assertEqual(r[key], 0, f"{key} should be 0")

    # --- format_detail_view pure-function tests ---

    def test_format_detail_view_renders_breakdown_for_most_recent_session_with_data(self):
        out = L.format_detail_view(
            self._SESSION_AGG, self._PROJECT_SUMMARY, [],
            [self._SESSION_ALL_ZEROS, self._SESSION_WITH_ACTUALS], [],
        )
        # Header must appear and include the date from the most-recent session.
        self.assertIn("Tokens Anthropic processed", out)
        self.assertIn("2026-09-10", out)

    def test_format_detail_view_skips_older_session_when_most_recent_has_data(self):
        # Only ONE breakdown block should appear -- for the most recent session.
        out = L.format_detail_view(
            self._SESSION_AGG, self._PROJECT_SUMMARY, [],
            [self._SESSION_ALL_ZEROS, self._SESSION_WITH_ACTUALS], [],
        )
        self.assertEqual(out.count("Tokens Anthropic processed"), 1)
        self.assertNotIn("2026-09-08", out)  # older session date must not appear

    def test_format_detail_view_falls_back_to_older_session_when_most_recent_has_zeros(self):
        # Most recent session has no actual_* data; the next most recent that does
        # should be shown instead.
        older_with_actuals = dict(self._SESSION_WITH_ACTUALS, ended_at="2026-09-08T15:00:00Z")
        recent_no_actuals = dict(self._SESSION_ALL_ZEROS, ended_at="2026-09-10T19:00:00Z")
        out = L.format_detail_view(
            self._SESSION_AGG, self._PROJECT_SUMMARY, [],
            [older_with_actuals, recent_no_actuals], [],
        )
        self.assertIn("Tokens Anthropic processed", out)
        self.assertIn("2026-09-08", out)

    def test_format_detail_view_skips_breakdown_when_all_actuals_are_zero(self):
        out = L.format_detail_view(
            self._SESSION_AGG, self._PROJECT_SUMMARY, [],
            [self._SESSION_ALL_ZEROS], [],
        )
        self.assertNotIn("Tokens Anthropic processed", out)

    def test_format_detail_view_skips_breakdown_when_last_n_is_empty(self):
        out = L.format_detail_view(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], [])
        self.assertNotIn("Tokens Anthropic processed", out)

    # --- format_json pure-function tests ---

    def test_format_json_session_block_includes_actual_token_fields(self):
        parsed = json.loads(L.format_json(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], []))
        sess = parsed["session"]
        for key in ("actual_input_tokens", "actual_output_tokens",
                    "actual_cache_read_tokens", "actual_cache_write_tokens"):
            self.assertIn(key, sess, f"Missing session key: {key}")
            self.assertEqual(sess[key], 0)

    def test_format_json_last_n_sessions_includes_actual_token_fields(self):
        parsed = json.loads(L.format_json(
            self._SESSION_AGG, self._PROJECT_SUMMARY, [],
            [self._SESSION_WITH_ACTUALS], [],
        ))
        row = parsed["last_n_sessions"][0]
        self.assertEqual(row["actual_input_tokens"], 1000)
        self.assertEqual(row["actual_output_tokens"], 200)
        self.assertEqual(row["actual_cache_read_tokens"], 5000)
        self.assertEqual(row["actual_cache_write_tokens"], 100)

    def test_format_json_last_n_sessions_actual_fields_absent_coerces_to_zero(self):
        # A last_n entry without actual_* keys (e.g. an old-shape dict from before
        # this fix) must produce 0 in JSON, not raise KeyError.
        old_shape = {"ended_at": "2026-09-01T00:00:00Z", "saved_tokens": 500}
        parsed = json.loads(L.format_json(
            self._SESSION_AGG, self._PROJECT_SUMMARY, [], [old_shape], [],
        ))
        row = parsed["last_n_sessions"][0]
        for key in ("actual_input_tokens", "actual_output_tokens",
                    "actual_cache_read_tokens", "actual_cache_write_tokens"):
            self.assertEqual(row[key], 0, f"{key} should be 0 for old-shape entry")


if __name__ == "__main__":
    unittest.main()
