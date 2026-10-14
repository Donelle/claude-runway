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
import time
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


class ShadowMarkersAreNeverLedgers(SavingsLedgerTestCase):
    """Issue #273: session_id_lib's `session_<id>.jsonl` shadow markers share
    the default sessions dir with this module's `<id>.jsonl` ledgers (the
    base class points both modules at the same temp dir, as in a default
    install). The unfiltered current_session_id() used to return the newest
    `*.jsonl`'s stem -- very often a marker, since markers are refreshed on
    every lifecycle event -- and get_live_session_aggregate() would then parse
    the marker as an event log."""

    def _ledger_then_fresher_marker(self):
        L.record_event("sess-real", "compress_file", 1000, 100, True, project="claude-runway")
        session_id_lib.record_shadow_marker("sess-real", project="/repos/claude-runway")
        ledger = L._session_jsonl_path("sess-real")
        marker = session_id_lib._shadow_marker_path("sess-real")
        self.assertEqual(ledger.parent, marker.parent)  # sanity: one shared dir
        os.utime(ledger, (1000, 1000))
        os.utime(marker, (2000, 2000))  # the marker is the newest *.jsonl
        return marker

    def test_unfiltered_lookup_skips_a_fresher_marker(self):
        self._ledger_then_fresher_marker()
        self.assertEqual(L.current_session_id(), "sess-real")

    def test_unfiltered_lookup_with_only_markers_returns_none(self):
        session_id_lib.record_shadow_marker("sess-a", project="/repos/claude-runway")
        self.assertIsNone(L.current_session_id())

    def test_marker_stem_is_never_parsed_as_an_event_log(self):
        marker = self._ledger_then_fresher_marker()
        self.assertTrue(marker.exists())  # sanity: there IS a file to misread
        stem = marker.stem
        self.assertEqual(L.read_session_events(stem), [])
        agg = L.get_live_session_aggregate(stem)
        self.assertEqual(agg["by_tool"], {})
        self.assertEqual(agg["event_count"], 0)
        # The real ledger next to it is unaffected.
        self.assertEqual(len(L.read_session_events("sess-real")), 1)

    def test_prefix_is_the_one_the_marker_writer_uses(self):
        # Writer and readers share session_id_lib.SESSION_MARKER_PREFIX, so a
        # change to the marker filename can't silently reopen this bug.
        marker = session_id_lib._shadow_marker_path("sess-x")
        self.assertTrue(marker.name.startswith(session_id_lib.SESSION_MARKER_PREFIX))
        self.assertTrue(L._is_shadow_marker_name(marker.name))
        self.assertIsNone(L._artifact_session_id(marker.name))
        self.assertFalse(L._is_shadow_marker_name(L._session_jsonl_path("sess-x").name))


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
    raw_chars_sum/out_chars_sum existed must not make
    finalize_session raise sqlite3.OperationalError -- and the migration
    scaffold itself must be safe to run against a DB that already has the
    columns but hasn't been through PRAGMA user_version tracking yet."""

    def _db_path(self) -> Path:
        return L.resolve_db_path()

    def test_old_schema_db_no_longer_raises_and_gains_the_columns(self):
        _old_schema_db_path(self._db_path())

        # The exact regression: this used to raise sqlite3.OperationalError.
        result = L.finalize_session("sess1", "myproject")
        self.assertEqual(result["raw_chars_sum"], 0)
        self.assertNotIn("overhead_tokens", result)

        conn = sqlite3.connect(str(self._db_path()))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        self.assertTrue({"raw_chars_sum", "out_chars_sum"} <= columns)
        # Issue #439: the schema-overhead estimate column is gone.
        self.assertNotIn("overhead_tokens", columns)
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
            "SELECT project, credited_saved_chars, raw_chars_sum, out_chars_sum "
            "FROM sessions WHERE session_id = 'old-sess'"
        ).fetchone()
        conn.close()
        # 42 saved tokens -> 147 chars via v6's tokens x 3.5 backfill; the token
        # columns themselves are gone after v7 (issue #438).
        self.assertEqual(row, ("old-project", 147, 0, 0))

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
        # Derived, not hardcoded: the first version past the last registered
        # migration is always "unregistered".
        unregistered = max(L._MIGRATIONS) + 1
        with mock.patch.object(L, "SCHEMA_VERSION", unregistered):
            # _MIGRATIONS has no entry for this version -- deliberately left
            # unregistered to exercise the guard.
            with self.assertRaises(RuntimeError):
                L.finalize_session("sess1", "myproject")

        conn = sqlite3.connect(str(self._db_path()))
        # Rolled back entirely -- the registered migrations (which ran FIRST and
        # would have succeeded) must not have stuck, since the whole sequence is
        # one transaction.
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 0)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        self.assertNotIn("raw_tokens_sum", columns)
        conn.close()

    def test_migration_function_failure_rolls_back_its_own_alter(self):
        _old_schema_db_path(self._db_path())

        def _broken_migration(conn):
            conn.execute("ALTER TABLE sessions ADD COLUMN raw_chars_sum INTEGER NOT NULL DEFAULT 0")
            raise RuntimeError("simulated failure partway through a migration")

        with mock.patch.object(L, "_MIGRATIONS", {1: _broken_migration}):
            with self.assertRaises(RuntimeError):
                L.finalize_session("sess1", "myproject")

        conn = sqlite3.connect(str(self._db_path()))
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 0)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        self.assertNotIn(
            "raw_chars_sum", columns,
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
            # Derived, not hardcoded: 5 is a registered migration now, so the
            # first unregistered version is the one past the last entry.
            with mock.patch.object(L, "SCHEMA_VERSION", max(L._MIGRATIONS) + 1):
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
    _SESSION_AGG = {"credited_saved_chars": 0, "event_count": 0, "raw_chars_sum": 0, "out_chars_sum": 0, "project": "current-session-project"}
    _PROJECT_SUMMARY = {"project": "current-session-project"}

    def _render(self, all_projects):
        return L.format_detail_view(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], all_projects)

    def test_multiple_projects_sum_correctly(self):
        out = self._render([
            {"project": "acme-a", "sessions": 10, "total_raw_chars": 200000, "total_out_chars": 84700},
            {"project": "acme-b", "sessions": 8, "total_raw_chars": 100000, "total_out_chars": 29300},
        ])
        lines = out.splitlines()
        total_line = [ln for ln in lines if ln.strip().startswith("Total")]
        self.assertEqual(len(total_line), 1)
        # 10 + 8 sessions; 300,000 chars before, 114,000 after -> 62% reduction.
        self.assertIn("18 sessions", total_line[0])
        self.assertIn("300,000", total_line[0])
        self.assertIn("114,000", total_line[0])
        self.assertIn("62% reduction", total_line[0])

    def test_single_project_total_matches_its_own_row(self):
        out = self._render([{"project": "acme-a", "sessions": 5, "total_raw_chars": 100000, "total_out_chars": 58000}])
        lines = out.splitlines()
        project_line = [ln for ln in lines if "acme-a" in ln]
        total_line = [ln for ln in lines if ln.strip().startswith("Total")]
        self.assertEqual(len(project_line), 1)
        self.assertEqual(len(total_line), 1)
        self.assertIn("5 sessions", total_line[0])
        for line in (project_line[0], total_line[0]):
            self.assertIn("100,000 → 58,000 chars", line)
            self.assertIn("42% reduction", line)

    def test_no_all_projects_means_no_total_row(self):
        out = self._render([])
        self.assertNotIn("Across projects", out)
        self.assertNotIn("Total", out)


class PerToolRawTokensAccumulation(SavingsLedgerTestCase):
    """Issue #65: raw_chars_sum, out_chars_sum, and credited_event_count
    accumulated per tool so avg % reduction can be computed and displayed.
    Tests the accumulation logic in both finalize_session (DB path) and
    get_live_session_aggregate (JSONL path).
    """

    def test_finalize_session_accumulates_raw_and_out_chars_per_tool(self):
        # Two credited events for the same tool -- raw and out must sum both.
        L.record_event("s1", "hook:Bash", raw_chars=1000, out_chars=100, credited=True)
        L.record_event("s1", "hook:Bash", raw_chars=500, out_chars=50, credited=True)
        L.finalize_session("s1", "proj")

        rows = L.query_session_tool_breakdown("s1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tool"], "hook:Bash")
        self.assertEqual(rows[0]["event_count"], 2)
        self.assertEqual(rows[0]["saved_chars"], 1350)        # (1000-100) + (500-50)
        self.assertEqual(rows[0]["raw_chars_sum"], 1500)      # 1000 + 500
        self.assertEqual(rows[0]["out_chars_sum"], 150)       # 100 + 50
        self.assertEqual(rows[0]["credited_event_count"], 2)

    def test_finalize_session_does_not_credit_uncredited_raw_chars(self):
        # fetch_url events are uncredited -- their raw/out must NOT feed into
        # the credited aggregates (the % would be meaningless).
        L.record_event("s2", "fetch_url", raw_chars=2000, out_chars=200, credited=False)
        L.record_event("s2", "hook:Bash", raw_chars=800, out_chars=80, credited=True)
        L.finalize_session("s2", "proj")

        rows = {r["tool"]: r for r in L.query_session_tool_breakdown("s2")}
        self.assertEqual(rows["fetch_url"]["raw_chars_sum"], 0)      # uncredited -> 0
        self.assertEqual(rows["fetch_url"]["out_chars_sum"], 0)
        self.assertEqual(rows["fetch_url"]["credited_event_count"], 0)
        self.assertEqual(rows["hook:Bash"]["raw_chars_sum"], 800)
        self.assertEqual(rows["hook:Bash"]["out_chars_sum"], 80)
        self.assertEqual(rows["hook:Bash"]["credited_event_count"], 1)

    def test_finalize_session_multiple_tools_tracked_independently(self):
        L.record_event("s3", "hook:Bash", raw_chars=1000, out_chars=100, credited=True)
        L.record_event("s3", "mcp:compress_command_output", raw_chars=3000, out_chars=450, credited=True)
        L.finalize_session("s3", "proj")

        rows = {r["tool"]: r for r in L.query_session_tool_breakdown("s3")}
        self.assertEqual(rows["hook:Bash"]["raw_chars_sum"], 1000)
        self.assertEqual(rows["hook:Bash"]["out_chars_sum"], 100)
        self.assertEqual(rows["mcp:compress_command_output"]["raw_chars_sum"], 3000)
        self.assertEqual(rows["mcp:compress_command_output"]["out_chars_sum"], 450)

    def test_get_live_session_aggregate_accumulates_all_credited_fields(self):
        # Same logic as finalize_session but reads from JSONL (mid-session).
        L.record_event("s4", "hook:Bash", raw_chars=600, out_chars=60, credited=True)
        L.record_event("s4", "hook:Bash", raw_chars=400, out_chars=40, credited=True)
        agg = L.get_live_session_aggregate("s4")

        bash_agg = agg["by_tool"]["hook:Bash"]
        self.assertEqual(bash_agg["raw_chars_sum"], 1000)        # 600 + 400
        self.assertEqual(bash_agg["out_chars_sum"], 100)         # 60 + 40
        self.assertEqual(bash_agg["saved_chars"], 900)           # (600-60) + (400-40)
        self.assertEqual(bash_agg["credited_event_count"], 2)

    def test_get_live_session_aggregate_uncredited_does_not_add_raw_chars(self):
        L.record_event("s5", "fetch_url", raw_chars=5000, out_chars=500, credited=False)
        agg = L.get_live_session_aggregate("s5")
        self.assertEqual(agg["by_tool"]["fetch_url"]["raw_chars_sum"], 0)
        self.assertEqual(agg["by_tool"]["fetch_url"]["out_chars_sum"], 0)
        self.assertEqual(agg["by_tool"]["fetch_url"]["credited_event_count"], 0)

    def test_credited_event_with_expanding_summary_uses_real_out_chars(self):
        # A summary that expands (out > raw): saved_chars is clamped to 0 by
        # record_event, but raw and out are still recorded faithfully on the
        # event. out_chars_sum must reflect the REAL out (150), not raw-saved (100-0=100).
        L.record_event("s6", "hook:Bash", raw_chars=100, out_chars=150, credited=True)
        L.finalize_session("s6", "proj")

        rows = L.query_session_tool_breakdown("s6")
        self.assertEqual(rows[0]["saved_chars"], 0)          # clamped at record_event
        self.assertEqual(rows[0]["raw_chars_sum"], 100)
        self.assertEqual(rows[0]["out_chars_sum"], 150)      # real out, not raw-saved
        self.assertEqual(rows[0]["credited_event_count"], 1)  # still credited


class PerToolAvgPctFormat(unittest.TestCase):
    """Issue #65: format_detail_view must show avg % reduction per credited
    tool. Pure-function tests -- no DB, no temp dir needed.
    """

    _SESSION_AGG = {
        "credited_saved_chars": 0, "event_count": 0,
        "raw_chars_sum": 0, "out_chars_sum": 0, "project": "p",
    }
    _PROJECT_SUMMARY = {"project": "p"}

    def _render(self, tool_breakdown):
        return L.format_detail_view(
            self._SESSION_AGG, self._PROJECT_SUMMARY, tool_breakdown, [], []
        )

    def test_avg_pct_appears_for_credited_tool(self):
        # 1000 raw, 150 out -> 1 - 150/1000 = 85% reduction.
        breakdown = [{"tool": "hook:Bash", "event_count": 10, "saved_chars": 850,
                      "raw_chars_sum": 1000, "out_chars_sum": 150, "credited_event_count": 10}]
        out = self._render(breakdown)
        # The percentage and "reduction" label must both appear on the tool line.
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertEqual(len(tool_lines), 1)
        self.assertIn("reduction", tool_lines[0])
        self.assertIn("85%", tool_lines[0])

    def test_avg_pct_zero_raw_does_not_crash(self):
        # Old DB rows backfilled with raw_chars_sum=0 must display gracefully.
        # credited_event_count=5 puts the tool in the credited branch; raw=0 then
        # suppresses the percentage.
        breakdown = [{"tool": "hook:Bash", "event_count": 5, "saved_chars": 200,
                      "raw_chars_sum": 0, "out_chars_sum": 0, "credited_event_count": 5}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertEqual(len(tool_lines), 1)
        # No % shown (denominator is 0), but the line must still appear.
        self.assertNotIn("reduction", tool_lines[0])

    def test_uncredited_tool_has_no_avg_pct(self):
        breakdown = [{"tool": "fetch_url", "event_count": 3, "saved_chars": 0,
                      "raw_chars_sum": 0, "out_chars_sum": 0, "credited_event_count": 0}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "fetch_url" in ln]
        self.assertEqual(len(tool_lines), 1)
        self.assertIn("not credited", tool_lines[0])
        self.assertNotIn("reduction", tool_lines[0])

    def test_credited_tool_with_zero_savings_shows_0pct_not_uncredited(self):
        # A credited tool whose every summary EXPANDED (out > raw per event) --
        # saved_chars is clamped to 0, but credited_event_count > 0 means the
        # tool IS credited; it should render "0% reduction" not "not credited".
        breakdown = [{"tool": "hook:Bash", "event_count": 2, "saved_chars": 0,
                      "raw_chars_sum": 200, "out_chars_sum": 300, "credited_event_count": 2}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertEqual(len(tool_lines), 1)
        self.assertNotIn("not credited", tool_lines[0])
        self.assertIn("0% reduction", tool_lines[0])

    def test_avg_pct_uses_out_chars_sum_not_raw_minus_saved(self):
        # Regression guard for the Copilot-found bug: out = raw - saved gives
        # the wrong % when any event had a clamped saved_chars. Two events:
        #   event 1: raw=100, out=150 -> saved=0 (clamped) -> naive out = 100-0=100 (WRONG, real=150)
        #   event 2: raw=100, out=10  -> saved=90
        # Naive reconstructed out: (100+100)-(0+90)=110 -> 1-110/200 = 45%
        # Real out_chars_sum: 150+10=160            -> 1-160/200 = 20%
        breakdown = [{"tool": "hook:Bash", "event_count": 2, "saved_chars": 90,
                      "raw_chars_sum": 200, "out_chars_sum": 160, "credited_event_count": 2}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        # 1 - 160/200 = 20%
        self.assertIn("20%", tool_lines[0])
        self.assertNotIn("45%", tool_lines[0])

    def test_avg_pct_100_percent(self):
        # out_chars_sum=0, credited_event_count=1 -> 100% reduction (real zero-output).
        breakdown = [{"tool": "hook:Bash", "event_count": 1, "saved_chars": 500,
                      "raw_chars_sum": 500, "out_chars_sum": 0, "credited_event_count": 1}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertIn("100%", tool_lines[0])

    def test_v2_migrated_row_credited_but_no_pct_shown(self):
        # A v2→v3 migrated row: real saved_chars and raw_chars_sum (from v2),
        # but credited_event_count=0 and out_chars_sum=0 (v3 backfill defaults).
        # The row IS credited (saved_chars > 0 heuristic), but the avg % must
        # be SUPPRESSED -- computing 1 - 0/raw would yield a fabricated 100%.
        breakdown = [{"tool": "hook:Bash", "event_count": 5, "saved_chars": 900,
                      "raw_chars_sum": 1000, "out_chars_sum": 0, "credited_event_count": 0}]
        out = self._render(breakdown)
        tool_lines = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertEqual(len(tool_lines), 1)
        # Must render as credited (not "not credited") but without a % metric.
        self.assertNotIn("not credited", tool_lines[0])
        self.assertNotIn("reduction", tool_lines[0])


class SchemaMigrationV2V3(SavingsLedgerTestCase):
    """Schema migrations: v2 adds the raw-size column and v3 the out-size column
    plus credited_event_count to session_tools (issue #65 and follow-up); v6/v7
    later turn the size columns into raw_chars_sum/out_chars_sum, which these
    tests assert on.
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
        # Mark it as at v1 so the migrations from v2 onward run.
        conn.execute("PRAGMA user_version = 1")
        conn.execute("INSERT INTO session_tools VALUES ('old-sess', 'hook:Bash', 5, 300)")
        conn.commit()
        conn.close()

    def test_v2_v3_migrations_add_all_new_columns(self):
        self._old_v1_db()
        L.finalize_session("new-sess", "proj")

        conn = sqlite3.connect(str(self._db_path()))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(session_tools)").fetchall()}
        self.assertIn("raw_chars_sum", columns)
        self.assertIn("out_chars_sum", columns)
        self.assertIn("credited_event_count", columns)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        conn.close()

    def test_v2_v3_migrations_backfill_existing_rows_with_zero(self):
        self._old_v1_db()
        L.finalize_session("new-sess", "proj")

        conn = sqlite3.connect(str(self._db_path()))
        row = conn.execute(
            "SELECT event_count, saved_chars, raw_chars_sum, out_chars_sum, credited_event_count "
            "FROM session_tools WHERE session_id = 'old-sess'"
        ).fetchone()
        conn.close()
        # Existing row must survive with the unknown columns at 0 (honest "unknown");
        # its 300 saved tokens carry over as 1050 chars (v6's tokens x 3.5 backfill).
        self.assertEqual(row, (5, 1050, 0, 0, 0))

    def test_query_session_tool_breakdown_includes_all_new_fields(self):
        L.record_event("s-qbd", "hook:Bash", raw_chars=1000, out_chars=100, credited=True)
        L.finalize_session("s-qbd", "proj")

        rows = L.query_session_tool_breakdown("s-qbd")
        self.assertEqual(len(rows), 1)
        self.assertIn("raw_chars_sum", rows[0])
        self.assertIn("out_chars_sum", rows[0])
        self.assertIn("credited_event_count", rows[0])
        self.assertEqual(rows[0]["raw_chars_sum"], 1000)
        self.assertEqual(rows[0]["out_chars_sum"], 100)
        self.assertEqual(rows[0]["credited_event_count"], 1)


class QueryAllProjectsRicherStats(SavingsLedgerTestCase):
    """Issue #68/#438: query_all_projects() returns, per project, the characters
    before/after/saved and best_tool -- pure DB-layer tests that exercise the SQL
    directly."""

    def test_totals_are_summed_across_sessions(self):
        L.record_event("s-avg-1", "hook:Bash", 1000, 900, credited=True)   # saved=100
        L.finalize_session("s-avg-1", "proj-a")
        L.record_event("s-avg-2", "hook:Bash", 1000, 700, credited=True)   # saved=300
        L.finalize_session("s-avg-2", "proj-a")

        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertIn("proj-a", rows)
        self.assertEqual(rows["proj-a"]["sessions"], 2)
        self.assertEqual(rows["proj-a"]["total_raw_chars"], 2000)
        self.assertEqual(rows["proj-a"]["total_out_chars"], 1600)
        self.assertEqual(rows["proj-a"]["total_saved_chars"], 400)

    def test_no_estimated_token_fields_are_returned(self):
        L.record_event("s-x", "hook:Bash", 1000, 900, credited=True)
        L.finalize_session("s-x", "proj-x")
        row = L.query_all_projects()[0]
        self.assertFalse([k for k in row if k.endswith("_tokens") or "tokens_per" in k], row)

    def test_best_tool_per_project(self):
        # hook:Bash saves 100, compress_command_output saves 300 -- best_tool must
        # pick compress_command_output regardless of event_count ordering.
        L.record_event("s-bt-1", "hook:Bash", 1000, 900, credited=True)              # saved=100
        L.record_event("s-bt-1", "mcp:compress_command_output", 1000, 700, credited=True)  # saved=300
        L.finalize_session("s-bt-1", "proj-b")

        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertEqual(rows["proj-b"]["best_tool"], "mcp:compress_command_output")

    def test_best_tool_ranks_by_aggregate_not_sum_of_clamped_savings(self):
        # hook:Bash saves 800 in one session and grows its content by 400 in another.
        # Each session's stored saving is clamped at 0 (800 and 0, summing to 800),
        # but the tool's real aggregate is 1100 -> 700 = 400. compress_file's
        # aggregate is 500, so it is the better tool even though hook:Bash's summed
        # per-session saved_chars is higher.
        L.record_event("s-agg-1", "hook:Bash", 1000, 200, credited=True)
        L.record_event("s-agg-1", "mcp:compress_file", 1000, 500, credited=True)
        L.finalize_session("s-agg-1", "proj-agg")
        L.record_event("s-agg-2", "hook:Bash", 100, 500, credited=True)
        L.finalize_session("s-agg-2", "proj-agg")

        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertEqual(rows["proj-agg"]["best_tool"], "mcp:compress_file")

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
            "(session_id, project, ended_at, credited_saved_chars, raw_chars_sum, "
            " out_chars_sum, event_count, fetch_url_count) "
            "VALUES ('bare-sess', 'proj-bare', '2026-01-01T00:00:00Z', 50, 0, 0, 1, 0)"
        )
        conn.commit()
        conn.close()

        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertIn("proj-bare", rows)
        self.assertIsNone(rows["proj-bare"]["best_tool"])
        # Credited events but no recorded size: counted as unsized, not as a 0-char total.
        self.assertEqual(rows["proj-bare"]["unsized_sessions"], 1)
        self.assertEqual(rows["proj-bare"]["total_saved_chars"], 0)

    def test_zero_totals_for_project_with_no_credited_events(self):
        L.finalize_session("s-zero", "proj-zero")
        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertEqual(
            (rows["proj-zero"]["total_raw_chars"], rows["proj-zero"]["total_out_chars"],
             rows["proj-zero"]["total_saved_chars"]),
            (0, 0, 0),
        )

    def test_multiple_projects_independent(self):
        # Two different projects must not bleed into each other's totals or best_tool.
        L.record_event("s-p1", "hook:Bash", 1000, 500, credited=True)    # saved=500
        L.finalize_session("s-p1", "proj-x")
        L.record_event("s-p2", "mcp:compress_file", 1000, 100, credited=True)  # saved=900
        L.finalize_session("s-p2", "proj-y")

        rows = {r["project"]: r for r in L.query_all_projects()}
        self.assertEqual(rows["proj-x"]["total_saved_chars"], 500)
        self.assertEqual(rows["proj-x"]["best_tool"], "hook:Bash")
        self.assertEqual(rows["proj-y"]["total_saved_chars"], 900)
        self.assertEqual(rows["proj-y"]["best_tool"], "mcp:compress_file")


class FormatDetailViewCrossProjectRicherStats(unittest.TestCase):
    """Issue #68/#438: format_detail_view shows characters before/after, percent
    reduction and best_tool in the 'Across projects' block when present.
    Pure-function tests -- no DB or temp dir needed."""

    _SESSION_AGG = {
        "credited_saved_chars": 0, "event_count": 0,
        "raw_chars_sum": 0, "out_chars_sum": 0,
        "project": "current-session-project",
    }
    _PROJECT_SUMMARY = {"project": "current-session-project"}

    def _render(self, all_projects):
        return L.format_detail_view(
            self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], all_projects
        )

    def test_before_after_reduction_and_best_tool_shown_when_present(self):
        out = self._render([{
            "project": "acme-a", "sessions": 4, "total_raw_chars": 80000,
            "total_out_chars": 20000, "best_tool": "hook:Bash",
        }])
        project_lines = [ln for ln in out.splitlines() if "acme-a" in ln]
        self.assertEqual(len(project_lines), 1)
        self.assertIn("80,000 → 20,000 chars", project_lines[0])
        self.assertIn("75% reduction", project_lines[0])
        self.assertIn("best: hook:Bash", project_lines[0])

    def test_best_tool_suppressed_when_none(self):
        # best_tool=None means no session_tools data -- must not render "best: None".
        out = self._render([{
            "project": "proj-bare", "sessions": 1, "total_raw_chars": 100,
            "total_out_chars": 50, "best_tool": None,
        }])
        project_lines = [ln for ln in out.splitlines() if "proj-bare" in ln]
        self.assertEqual(len(project_lines), 1)
        self.assertNotIn("best:", project_lines[0])
        self.assertNotIn("None", project_lines[0])

    def test_backward_compat_missing_keys_do_not_crash(self):
        # An old-shape dict without the character totals must not raise -- .get()
        # supplies 0, shown as "n/a" rather than a fabricated percentage.
        out = self._render([{"project": "acme-a", "sessions": 10}])
        lines = out.splitlines()
        project_line = [ln for ln in lines if "acme-a" in ln]
        self.assertEqual(len(project_line), 1)
        self.assertIn("n/a reduction", project_line[0])
        self.assertNotIn("None", project_line[0])

    def test_zero_raw_is_na_not_a_fabricated_percentage(self):
        out = self._render([{
            "project": "proj-zero", "sessions": 1, "total_raw_chars": 0,
            "total_out_chars": 0, "best_tool": None,
        }])
        project_lines = [ln for ln in out.splitlines() if "proj-zero" in ln]
        self.assertEqual(len(project_lines), 1)
        self.assertIn("0 → 0 chars", project_lines[0])
        self.assertIn("n/a reduction", project_lines[0])

    def test_output_has_no_estimated_token_wording(self):
        out = self._render([{
            "project": "acme-a", "sessions": 4, "total_raw_chars": 80000,
            "total_out_chars": 20000, "best_tool": "hook:Bash",
        }])
        self.assertNotIn("~", out)
        self.assertNotIn("tokens avoided", out)


class FormatJson(unittest.TestCase):
    """format_json must return valid JSON with the correct top-level structure.
    Pure-function tests -- no DB or temp dir needed.
    """

    # Shared minimal inputs; tests override only the part they care about.
    _SESSION_AGG = {
        "project": "my-repo",
        "credited_saved_chars": 1234,
        "raw_chars_sum": 5000,
        "out_chars_sum": 3766,
        "event_count": 7,
        "fetch_url_count": 2,
    }
    _PROJECT_SUMMARY = {
        "project": "my-repo",
        "sessions": 10,
        "total_saved_chars": 50000,
        "best_session_chars": 8000,
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
        self.assertEqual(sess["credited_saved_chars"], 1234)
        self.assertEqual(sess["raw_chars_sum"], 5000)
        self.assertEqual(sess["event_count"], 7)
        self.assertEqual(sess["fetch_url_count"], 2)
        self.assertNotIn("overhead_tokens", sess)  # issue #439

    def test_project_summary_values_are_correct(self):
        parsed = json.loads(self._render())
        ps = parsed["project_summary"]
        self.assertEqual(ps["project"], "my-repo")
        self.assertEqual(ps["sessions"], 10)
        self.assertEqual(ps["total_saved_chars"], 50000)
        self.assertEqual(ps["best_session_chars"], 8000)

    def test_tool_breakdown_serialised_correctly(self):
        breakdown = [
            {"tool": "hook:Bash", "event_count": 5, "saved_chars": 900,
             "raw_chars_sum": 1000, "out_chars_sum": 100, "credited_event_count": 5},
        ]
        parsed = json.loads(self._render(tool_breakdown=breakdown))
        self.assertEqual(len(parsed["tool_breakdown"]), 1)
        t = parsed["tool_breakdown"][0]
        self.assertEqual(t["tool"], "hook:Bash")
        self.assertEqual(t["saved_chars"], 900)
        self.assertEqual(t["raw_chars_sum"], 1000)

    def test_last_n_sessions_serialised_correctly(self):
        last_n = [
            {"ended_at": "2026-09-01T10:00:00Z", "saved_chars": 500},
            {"ended_at": "2026-09-02T10:00:00Z", "saved_chars": 800},
        ]
        parsed = json.loads(self._render(last_n=last_n))
        self.assertEqual(len(parsed["last_n_sessions"]), 2)
        self.assertEqual(parsed["last_n_sessions"][0]["saved_chars"], 500)
        self.assertEqual(parsed["last_n_sessions"][1]["ended_at"], "2026-09-02T10:00:00Z")

    def test_all_projects_best_tool_none_stays_null(self):
        # best_tool=None must round-trip as JSON null, not the string "None".
        all_projects = [
            {"project": "proj-bare", "sessions": 1, "total_saved_chars": 50,
             "avg_tokens_per_session": 50, "best_tool": None},
        ]
        parsed = json.loads(self._render(all_projects=all_projects))
        self.assertIsNone(parsed["all_projects"][0]["best_tool"])

    def test_all_projects_best_tool_present_is_a_string(self):
        all_projects = [
            {"project": "proj-a", "sessions": 3, "total_saved_chars": 300,
             "avg_tokens_per_session": 100, "best_tool": "hook:Bash"},
        ]
        parsed = json.loads(self._render(all_projects=all_projects))
        self.assertEqual(parsed["all_projects"][0]["best_tool"], "hook:Bash")

    def test_none_values_in_session_agg_are_coerced_to_zero(self):
        # A partially-filled aggregate (e.g. from a brand-new session with no
        # events yet) must not produce null in the JSON output.
        sparse_agg = {"project": "p", "credited_saved_chars": None}
        parsed = json.loads(self._render(session_agg=sparse_agg))
        self.assertEqual(parsed["session"]["credited_saved_chars"], 0)

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
        "credited_saved_chars": 1234,
        "raw_chars_sum": 5000,
        "out_chars_sum": 3766,
        "event_count": 7,
        "fetch_url_count": 2,
    }
    _PROJECT_SUMMARY = {
        "project": "my-repo",
        "sessions": 10,
        "total_saved_chars": 50000,
        "best_session_chars": 8000,
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
        self.assertIn("credited_saved_chars", header)
        self.assertIn("project_total_saved_chars", header)

    def test_session_table_data_matches_inputs(self):
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], [], table="session")
        rows = self._parse(out)
        header, data = rows[0], rows[1]
        d = dict(zip(header, data, strict=False))
        self.assertEqual(d["project"], "my-repo")
        self.assertEqual(int(d["credited_saved_chars"]), 1234)
        self.assertEqual(int(d["project_total_saved_chars"]), 50000)
        self.assertEqual(int(d["project_sessions"]), 10)

    def test_tools_table_has_header_and_correct_number_of_rows(self):
        breakdown = [
            {"tool": "hook:Bash", "event_count": 5, "saved_chars": 900,
             "raw_chars_sum": 1000, "out_chars_sum": 100, "credited_event_count": 5},
            {"tool": "fetch_url", "event_count": 3, "saved_chars": 0,
             "raw_chars_sum": 0, "out_chars_sum": 0, "credited_event_count": 0},
        ]
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, breakdown, [], [], table="tools")
        rows = self._parse(out)
        # 1 header + 2 data rows
        self.assertEqual(len(rows), 3)
        self.assertIn("tool", rows[0])

    def test_tools_table_data_matches_breakdown(self):
        breakdown = [
            {"tool": "hook:Bash", "event_count": 5, "saved_chars": 900,
             "raw_chars_sum": 1000, "out_chars_sum": 100, "credited_event_count": 5},
        ]
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, breakdown, [], [], table="tools")
        rows = self._parse(out)
        header, data = rows[0], rows[1]
        d = dict(zip(header, data, strict=False))
        self.assertEqual(d["tool"], "hook:Bash")
        self.assertEqual(int(d["saved_chars"]), 900)
        self.assertEqual(int(d["credited_event_count"]), 5)

    def test_projects_table_has_header_and_correct_number_of_rows(self):
        all_projects = [
            {"project": "proj-a", "sessions": 3, "total_saved_chars": 300,
             "avg_tokens_per_session": 100, "best_tool": "hook:Bash"},
            {"project": "proj-b", "sessions": 1, "total_saved_chars": 50,
             "avg_tokens_per_session": 50, "best_tool": None},
        ]
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], all_projects, table="projects")
        rows = self._parse(out)
        self.assertEqual(len(rows), 3)  # header + 2 data rows

    def test_projects_table_none_best_tool_is_empty_string(self):
        # None best_tool must render as an empty CSV field, not "None".
        all_projects = [
            {"project": "proj-bare", "sessions": 1, "total_saved_chars": 50,
             "avg_tokens_per_session": 50, "best_tool": None},
        ]
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, [], [], all_projects, table="projects")
        rows = self._parse(out)
        header, data = rows[0], rows[1]
        d = dict(zip(header, data, strict=False))
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
        d = dict(zip(rows[0], rows[1], strict=False))
        self.assertEqual(d["project"], "repo,with,commas")

    def test_formula_injection_prefix_on_equals_leading_project(self):
        # A project name starting with '=' would be evaluated as a spreadsheet
        # formula if left unchanged. _safe_str must prefix it with an apostrophe.
        agg = dict(self._SESSION_AGG, project="=DANGEROUS()")
        ps = dict(self._PROJECT_SUMMARY, project="=DANGEROUS()")
        out = L.format_csv(agg, ps, [], [], [], table="session")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1], strict=False))
        # Apostrophe prefix applied -- the raw cell value now starts with "'".
        self.assertTrue(d["project"].startswith("'"), d["project"])
        self.assertNotEqual(d["project"][0], "=")

    def test_formula_injection_prefix_on_plus_leading_tool(self):
        # Tool names could in principle start with '+' (e.g. a malformed event).
        breakdown = [
            {"tool": "+BAD", "event_count": 1, "saved_chars": 10,
             "raw_chars_sum": 100, "out_chars_sum": 90, "credited_event_count": 1},
        ]
        out = L.format_csv(self._SESSION_AGG, self._PROJECT_SUMMARY, breakdown, [], [], table="tools")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1], strict=False))
        self.assertTrue(d["tool"].startswith("'"), d["tool"])

    def test_safe_str_does_not_modify_normal_values(self):
        # Ordinary project/tool names must pass through unchanged.
        agg = dict(self._SESSION_AGG, project="my-normal-repo")
        ps = dict(self._PROJECT_SUMMARY, project="my-normal-repo")
        out = L.format_csv(agg, ps, [], [], [], table="session")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1], strict=False))
        self.assertEqual(d["project"], "my-normal-repo")

    def test_formula_injection_prefix_on_whitespace_prefixed_formula(self):
        # A tab or space before '=' bypasses a naive s[0] check because
        # spreadsheet applications strip leading whitespace before formula
        # detection. The guard must check the FIRST NON-WHITESPACE character.
        agg = dict(self._SESSION_AGG, project="\t=FORMULA()")
        ps = dict(self._PROJECT_SUMMARY, project="\t=FORMULA()")
        out = L.format_csv(agg, ps, [], [], [], table="session")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1], strict=False))
        # Apostrophe must be prepended even though the first char is \t.
        self.assertTrue(d["project"].startswith("'"), f"Expected apostrophe prefix, got: {d['project']!r}")

    def test_formula_injection_prefix_space_prefixed_formula(self):
        # Space before '+' -- same bypass vector as tab.
        agg = dict(self._SESSION_AGG, project=" +1+1")
        ps = dict(self._PROJECT_SUMMARY, project=" +1+1")
        out = L.format_csv(agg, ps, [], [], [], table="session")
        rows = self._parse(out)
        d = dict(zip(rows[0], rows[1], strict=False))
        self.assertTrue(d["project"].startswith("'"), f"Expected apostrophe prefix, got: {d['project']!r}")


class QueryTrend(SavingsLedgerTestCase):
    """Issue #67: query_trend groups sessions by day or week bucket."""

    def test_weekly_buckets_group_same_week_together(self):
        # Two sessions on 2026-09-07 (Mon) and 2026-09-09 (Wed) are in the
        # same ISO week (2026-W37) -- they must land in one bucket.
        conn = L._connect()
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, project, ended_at, credited_saved_chars, "
            " raw_chars_sum, out_chars_sum, event_count, fetch_url_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s-w1", "proj-w", "2026-09-07T09:00:00Z", 100, 1000, 900, 1, 0),
        )
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, project, ended_at, credited_saved_chars, "
            " raw_chars_sum, out_chars_sum, event_count, fetch_url_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s-w2", "proj-w", "2026-09-09T17:00:00Z", 200, 1000, 800, 1, 0),
        )
        conn.commit()
        conn.close()

        rows = L.query_trend("proj-w", bucket="week", n=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sessions"], 2)
        self.assertEqual(rows[0]["total_saved_chars"], 300)

    def test_weekly_buckets_iso_year_boundary_not_split(self):
        # Regression guard for the %Y-W%W bug Copilot identified in PR #157:
        # 2024-12-30 (Mon) and 2025-01-01 (Wed) are BOTH in ISO week 2025-W01
        # (the week containing the first Thursday of 2025). The broken %Y/%W
        # format would produce 2024-W53 and 2025-W00, splitting them into two
        # buckets. The correct %G/%V format must produce one bucket for both.
        conn = L._connect()
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, project, ended_at, credited_saved_chars, "
            " raw_chars_sum, out_chars_sum, event_count, fetch_url_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s-boundary-1", "proj-boundary", "2024-12-30T10:00:00Z", 100, 1000, 900, 1, 0),
        )
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, project, ended_at, credited_saved_chars, "
            " raw_chars_sum, out_chars_sum, event_count, fetch_url_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s-boundary-2", "proj-boundary", "2025-01-01T14:00:00Z", 200, 1000, 800, 1, 0),
        )
        conn.commit()
        conn.close()

        rows = L.query_trend("proj-boundary", bucket="week", n=10)
        # Both dates are in ISO week 2025-W01 -- must produce exactly ONE bucket.
        self.assertEqual(len(rows), 1, f"Expected 1 bucket (ISO 2025-W01) but got {len(rows)}: {rows}")
        self.assertEqual(rows[0]["sessions"], 2)
        self.assertEqual(rows[0]["total_saved_chars"], 300)
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
                "(session_id, project, ended_at, credited_saved_chars, "
                " raw_chars_sum, out_chars_sum, event_count, fetch_url_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, "proj-d", f"{day}T12:00:00Z", saved, 1000, 1000 - saved, 1, 0),
            )
        conn.commit()
        conn.close()

        rows = L.query_trend("proj-d", bucket="day", n=10)
        self.assertEqual(len(rows), 3)
        # Oldest first (ascending).
        self.assertEqual(rows[0]["bucket"], "2026-09-07")
        self.assertEqual(rows[0]["total_saved_chars"], 100)
        self.assertEqual(rows[2]["bucket"], "2026-09-09")
        self.assertEqual(rows[2]["total_saved_chars"], 300)

    def test_n_caps_returned_buckets(self):
        # Insert 5 sessions on 5 different days; n=3 must return only the
        # 3 most recent buckets (still in ascending order).
        conn = L._connect()
        for i in range(5):
            conn.execute(
                "INSERT INTO sessions "
                "(session_id, project, ended_at, credited_saved_chars, "
                " raw_chars_sum, out_chars_sum, event_count, fetch_url_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"s-cap-{i}", "proj-cap", f"2026-09-0{i + 1}T12:00:00Z", (i + 1) * 100, 1000, 900, 1, 0),
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
                "(session_id, project, ended_at, credited_saved_chars, "
                " raw_chars_sum, out_chars_sum, event_count, fetch_url_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"s-{proj}", proj, "2026-09-07T12:00:00Z", 500, 1000, 500, 1, 0),
            )
        conn.commit()
        conn.close()

        alpha = L.query_trend("proj-alpha", bucket="day", n=10)
        beta = L.query_trend("proj-beta", bucket="day", n=10)
        self.assertEqual(len(alpha), 1)
        self.assertEqual(alpha[0]["total_saved_chars"], 500)
        self.assertEqual(len(beta), 1)
        self.assertEqual(beta[0]["total_saved_chars"], 500)


class FormatTrendView(unittest.TestCase):
    """Issue #67/#438: format_trend_view must render a sparkline + table of
    characters before/after and percent reduction for the bucketed trend data
    returned by query_trend.
    Pure-function tests — no DB or temp dir needed.
    """

    @staticmethod
    def _row(bucket, sessions, raw, saved):
        return {"bucket": bucket, "sessions": sessions, "total_raw_chars": raw,
                "total_out_chars": raw - saved, "total_saved_chars": saved}

    def _rows(self, values, prefix="2026-W3"):
        """Build synthetic weekly trend rows from a list of characters-saved values."""
        return [self._row(f"{prefix}{i}", 1, v * 10, v) for i, v in enumerate(values, 1)]

    def test_empty_rows_returns_no_history_message(self):
        out = L.format_trend_view([], "my-repo", bucket="week")
        self.assertIn("No session history", out)
        self.assertIn("my-repo", out)

    def test_header_shows_project_and_bucket_type(self):
        out = L.format_trend_view(self._rows([100, 200]), "my-repo", bucket="week")
        self.assertIn("Weekly", out)
        self.assertIn("my-repo", out)

    def test_daily_header_says_daily(self):
        rows = [self._row("2026-09-07", 1, 1000, 100)]
        out = L.format_trend_view(rows, "my-repo", bucket="day")
        self.assertIn("Daily", out)

    def test_each_bucket_appears_in_output(self):
        rows = [self._row("2026-W36", 2, 1000, 300), self._row("2026-W37", 3, 2000, 700)]
        out = L.format_trend_view(rows, "my-repo", bucket="week")
        self.assertIn("2026-W36", out)
        self.assertIn("2026-W37", out)

    def test_each_bucket_shows_before_after_and_reduction(self):
        rows = [self._row("2026-W36", 2, 1000, 300)]
        line = [ln for ln in L.format_trend_view(rows, "my-repo", bucket="week").splitlines() if "2026-W36" in ln][0]
        self.assertIn("1,000", line)   # before
        self.assertIn("700", line)     # after
        self.assertIn("30%", line)     # reduction

    def test_total_row_sums_correctly(self):
        rows = [self._row("2026-W36", 2, 1000, 300), self._row("2026-W37", 3, 2000, 700)]
        out = L.format_trend_view(rows, "my-repo", bucket="week")
        lines = out.splitlines()
        total_lines = [ln for ln in lines if "Total" in ln]
        self.assertEqual(len(total_lines), 1)
        # 2 + 3 = 5 sessions; 3,000 chars before, 2,000 after -> 33% reduction.
        self.assertIn("5", total_lines[0])
        self.assertIn("3,000", total_lines[0])
        self.assertIn("2,000", total_lines[0])
        self.assertIn("33%", total_lines[0])

    def test_no_estimated_token_wording(self):
        out = L.format_trend_view(self._rows([100, 200]), "my-repo", bucket="week")
        self.assertNotIn("Tokens avoided", out)
        self.assertNotIn("~", out)

    def test_sparkline_is_present(self):
        # A sparkline character must appear somewhere in the output.
        out = L.format_trend_view(self._rows([100, 500, 300]), "my-repo")
        spark_chars = set(L._SPARK_CHARS)
        self.assertTrue(
            any(c in spark_chars for c in out),
            "Expected at least one sparkline character in the trend output",
        )

    def test_single_bucket_does_not_crash(self):
        out = L.format_trend_view([self._row("2026-W36", 1, 1000, 500)], "my-repo")
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
        import tempfile
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
        f.close()
        return f.name

    def _assistant(self, inp=0, out=0, cr=0, cw=0, msg_id=None):
        return {
            "type": "assistant",
            "message": {
                **({"id": msg_id} if msg_id else {}),
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

    def test_dedups_lines_sharing_a_message_id(self):
        """#298: one line per content block, all repeating the turn's usage."""
        path = self._make_transcript([
            # Turn A: text block + two tool_use blocks = 3 lines, same usage.
            self._assistant(inp=10, out=5, cr=100, cw=20, msg_id="msg_A"),
            self._assistant(inp=10, out=5, cr=100, cw=20, msg_id="msg_A"),
            self._assistant(inp=10, out=5, cr=100, cw=20, msg_id="msg_A"),
            # Turn B: single line.
            self._assistant(inp=3, out=8, cr=50, cw=0, msg_id="msg_B"),
        ])
        try:
            result = L.parse_transcript_token_counts(path)
            self.assertEqual(result, {"input": 13, "output": 13,
                                      "cache_read": 150, "cache_write": 20})
        finally:
            import os as _os; _os.unlink(path)

    def test_lines_without_message_id_are_all_counted(self):
        """No id means no safe dedup key: count every line, don't merge them."""
        path = self._make_transcript([
            self._assistant(inp=4, msg_id=None),
            self._assistant(inp=4, msg_id=None),
            self._assistant(inp=1, msg_id="msg_A"),
            self._assistant(inp=1, msg_id="msg_A"),
        ])
        try:
            self.assertEqual(L.parse_transcript_token_counts(path)["input"], 9)
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


class ParseSessionTokenCounts(unittest.TestCase):
    """parse_session_token_counts (#364): main transcript + subagent transcripts.

    Builds the real on-disk layout Claude Code uses --
    <tmp>/<session_id>.jsonl  and  <tmp>/<session_id>/subagents/agent-<id>.jsonl
    -- so the directory derivation from the transcript path is exercised, not
    stubbed.
    """

    SESSION = "sess-1"

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.main = self.root / f"{self.SESSION}.jsonl"
        self.sub_dir = self.root / self.SESSION / "subagents"

    @staticmethod
    def _assistant(inp=0, out=0, cr=0, cw=0, msg_id=None):
        return {
            "type": "assistant",
            "message": {
                **({"id": msg_id} if msg_id else {}),
                "usage": {
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cache_read_input_tokens": cr,
                    "cache_creation_input_tokens": cw,
                },
            },
        }

    def _write(self, path: Path, entries: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")

    def test_none_and_empty_path_return_none(self):
        self.assertIsNone(L.parse_session_token_counts(None))
        self.assertIsNone(L.parse_session_token_counts(""))

    def test_main_only_matches_single_file_parser(self):
        self._write(self.main, [self._assistant(inp=10, out=5, cr=100, cw=20, msg_id="m1")])
        self.assertEqual(
            L.parse_session_token_counts(self.main),
            L.parse_transcript_token_counts(self.main),
        )
        self.assertEqual(
            L.parse_session_token_counts(self.main),
            {"input": 10, "output": 5, "cache_read": 100, "cache_write": 20},
        )

    def test_sums_main_and_every_subagent_file(self):
        self._write(self.main, [self._assistant(inp=1, out=2, cr=100, cw=10, msg_id="m1")])
        self._write(self.sub_dir / "agent-aaa.jsonl", [self._assistant(inp=3, out=4, cr=200, cw=20, msg_id="a1")])
        self._write(self.sub_dir / "agent-bbb.jsonl", [self._assistant(inp=5, out=6, cr=300, cw=30, msg_id="b1")])
        self.assertEqual(
            L.parse_session_token_counts(self.main),
            {"input": 9, "output": 12, "cache_read": 600, "cache_write": 60},
        )

    def test_ignores_non_transcript_files_in_subagents_dir(self):
        self._write(self.main, [self._assistant(inp=1, msg_id="m1")])
        self._write(self.sub_dir / "agent-aaa.jsonl", [self._assistant(inp=2, msg_id="a1")])
        # The real directory also holds a .meta.json per agent; a stray file must
        # not be parsed as a transcript either.
        (self.sub_dir / "agent-aaa.meta.json").write_text(
            json.dumps(self._assistant(inp=9999, msg_id="meta")), encoding="utf-8")
        (self.sub_dir / "notes.jsonl").write_text(
            json.dumps(self._assistant(inp=9999, msg_id="stray")) + "\n", encoding="utf-8")
        self.assertEqual(L.parse_session_token_counts(self.main)["input"], 3)

    def test_unreadable_subagent_file_is_skipped_others_still_count(self):
        self._write(self.main, [self._assistant(inp=1, msg_id="m1")])
        self._write(self.sub_dir / "agent-good.jsonl", [self._assistant(inp=2, msg_id="g1")])
        (self.sub_dir / "agent-bad.jsonl").write_text("not json at all\n\x00\x01\n", encoding="utf-8")
        self.assertEqual(L.parse_session_token_counts(self.main)["input"], 3)

    def test_missing_main_transcript_still_returns_subagent_totals(self):
        # Main file absent/unparseable must not discard real subagent usage.
        self._write(self.sub_dir / "agent-aaa.jsonl", [self._assistant(inp=7, out=1, cr=50, cw=5, msg_id="a1")])
        self.assertEqual(
            L.parse_session_token_counts(self.main),
            {"input": 7, "output": 1, "cache_read": 50, "cache_write": 5},
        )

    def test_nothing_anywhere_returns_none(self):
        self.assertIsNone(L.parse_session_token_counts(self.main))
        self._write(self.sub_dir / "agent-aaa.jsonl", [{"type": "user", "message": {"content": "hi"}}])
        self.assertIsNone(L.parse_session_token_counts(self.main))

    def test_dedup_is_scoped_per_file_not_across_files(self):
        # #298 collapses one turn's per-content-block lines WITHIN a file. The same
        # message.id in two different files is two separate transcripts' turns.
        self._write(self.main, [
            self._assistant(inp=10, msg_id="same"),
            self._assistant(inp=10, msg_id="same"),  # duplicate block line: counted once
        ])
        self._write(self.sub_dir / "agent-aaa.jsonl", [self._assistant(inp=10, msg_id="same")])
        self.assertEqual(L.parse_session_token_counts(self.main)["input"], 20)

    def test_failure_listing_subagents_dir_falls_back_to_main_only(self):
        self._write(self.main, [self._assistant(inp=4, msg_id="m1")])
        self._write(self.sub_dir / "agent-aaa.jsonl", [self._assistant(inp=100, msg_id="a1")])
        with mock.patch.object(Path, "glob", side_effect=OSError("boom")):
            self.assertEqual(L.parse_session_token_counts(self.main)["input"], 4)


class RecordEventAgentFields(SavingsLedgerTestCase):
    """Issue #365: record_event writes agent_id/agent_type only when present."""

    def test_omitted_when_absent_so_old_shape_is_unchanged(self):
        L.record_event("s", "hook:Bash", 1000, 100, True, "cmd")
        event = L.read_session_events("s")[0]
        self.assertNotIn("agent_id", event)
        self.assertNotIn("agent_type", event)

    def test_both_written_when_given(self):
        L.record_event("s", "hook:Bash", 1000, 100, True, "cmd", agent_id="a1", agent_type="Explore")
        event = L.read_session_events("s")[0]
        self.assertEqual(event["agent_id"], "a1")
        self.assertEqual(event["agent_type"], "Explore")

    def test_empty_values_are_treated_as_absent(self):
        L.record_event("s", "hook:Bash", 1000, 100, True, "cmd", agent_id="", agent_type=None)
        event = L.read_session_events("s")[0]
        self.assertNotIn("agent_id", event)
        self.assertNotIn("agent_type", event)

    def test_agent_type_alone_is_stored(self):
        # `claude --agent X` main thread: agent_type present, no agent_id.
        L.record_event("s", "hook:Bash", 1000, 100, True, "cmd", agent_type="my-agent")
        event = L.read_session_events("s")[0]
        self.assertEqual(event["agent_type"], "my-agent")
        self.assertNotIn("agent_id", event)

    def test_overlong_values_are_truncated(self):
        L.record_event("s", "hook:Bash", 1000, 100, True, "cmd", agent_id="x" * 500, agent_type="y" * 500)
        event = L.read_session_events("s")[0]
        self.assertEqual(len(event["agent_id"]), 200)
        self.assertEqual(len(event["agent_type"]), 200)


class AggregateByAgent(unittest.TestCase):
    """_aggregate_by_agent: grouping, credited-only sums, subagent detection."""

    @staticmethod
    def _ev(saved=100, raw=None, out=100, credited=True, **agent):
        # The saving is derived from before/after, so build `raw` from the wanted saving.
        raw = out + saved if raw is None else raw
        return {"tool": "hook:Bash", "raw_chars": raw, "out_chars": out,
                "saved_chars": saved if credited else 0, "credited": credited, **agent}

    def test_no_agent_fields_lands_in_the_main_row(self):
        rows = L._aggregate_by_agent([self._ev(), self._ev()])
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["is_subagent"], rows[0]["agent_type"]), (0, ""))
        self.assertEqual(rows[0]["event_count"], 2)
        self.assertEqual(rows[0]["saved_chars"], 200)

    def test_subagent_detected_by_agent_id(self):
        rows = L._aggregate_by_agent([self._ev(), self._ev(agent_id="a1", agent_type="Explore")])
        by_key = {(r["is_subagent"], r["agent_type"]): r for r in rows}
        self.assertEqual(set(by_key), {(0, ""), (1, "Explore")})

    def test_agent_type_without_agent_id_is_not_a_subagent(self):
        # The `claude --agent X` main-thread case: must NOT be filed as a subagent.
        rows = L._aggregate_by_agent([self._ev(agent_type="my-agent")])
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["is_subagent"], rows[0]["agent_type"]), (0, "my-agent"))

    def test_main_agent_session_and_same_typed_subagent_do_not_collide(self):
        rows = L._aggregate_by_agent([
            self._ev(agent_type="X"),                    # main thread started with --agent X
            self._ev(agent_id="a1", agent_type="X"),     # a subagent of the same type
        ])
        self.assertEqual({(r["is_subagent"], r["agent_type"]) for r in rows}, {(0, "X"), (1, "X")})

    def test_same_type_different_agent_ids_merge_into_one_row(self):
        rows = L._aggregate_by_agent([
            self._ev(agent_id="a1", agent_type="Explore"),
            self._ev(agent_id="a2", agent_type="Explore"),
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_count"], 2)
        self.assertEqual(rows[0]["saved_chars"], 200)

    def test_uncredited_events_count_as_events_but_not_savings(self):
        rows = L._aggregate_by_agent([
            self._ev(),
            self._ev(credited=False, raw=999, out=1),
        ])
        self.assertEqual(rows[0]["event_count"], 2)
        self.assertEqual(rows[0]["credited_event_count"], 1)
        self.assertEqual(rows[0]["saved_chars"], 100)
        self.assertEqual(rows[0]["raw_chars_sum"], 200)  # uncredited raw excluded

    def test_sorted_by_saved_chars_descending(self):
        rows = L._aggregate_by_agent([
            self._ev(saved=10),
            self._ev(saved=500, agent_id="a1", agent_type="Big"),
            self._ev(saved=50, agent_id="a2", agent_type="Mid"),
        ])
        self.assertEqual([r["agent_type"] for r in rows], ["Big", "Mid", ""])

    def test_empty_events(self):
        self.assertEqual(L._aggregate_by_agent([]), [])


class SessionAgentsRollup(SavingsLedgerTestCase):
    """Issue #365: finalize_session persists the per-agent split; schema v5."""

    def _mixed_session(self, sid="s1"):
        L.record_event(sid, "hook:Bash", 1000, 100, True, "main cmd")
        L.record_event(sid, "hook:Bash", 2000, 200, True, "sub cmd", agent_id="a1", agent_type="general-purpose")
        L.record_event(sid, "hook:Bash", 3000, 300, True, "sub cmd 2", agent_id="a2", agent_type="general-purpose")

    def test_finalize_writes_main_and_subagent_rows(self):
        self._mixed_session()
        agg = L.finalize_session("s1", "proj")
        self.assertEqual({(r["is_subagent"], r["agent_type"]) for r in agg["by_agent"]},
                         {(0, ""), (1, "general-purpose")})
        rows = {(r["is_subagent"], r["agent_type"]): r for r in L.query_session_agent_breakdown("s1")}
        self.assertEqual(rows[(0, "")]["saved_chars"], 900)
        self.assertEqual(rows[(1, "general-purpose")]["saved_chars"], 1800 + 2700)
        self.assertEqual(rows[(1, "general-purpose")]["credited_event_count"], 2)

    def test_main_plus_subagent_rows_sum_to_the_session_total(self):
        self._mixed_session()
        agg = L.finalize_session("s1", "proj")
        self.assertEqual(sum(r["saved_chars"] for r in agg["by_agent"]), agg["credited_saved_chars"])

    def test_refinalizing_replaces_rather_than_duplicates(self):
        self._mixed_session()
        L.finalize_session("s1", "proj", delete_jsonl=False)
        L.finalize_session("s1", "proj")
        self.assertEqual(len(L.query_session_agent_breakdown("s1")), 2)

    def test_session_without_events_for_a_given_id_has_no_rows(self):
        self.assertEqual(L.query_session_agent_breakdown("never-existed"), [])

    def test_live_aggregate_includes_by_agent_without_writing(self):
        self._mixed_session("live")
        agg = L.get_live_session_aggregate("live")
        self.assertEqual(len(agg["by_agent"]), 2)
        self.assertEqual(L.query_session_agent_breakdown("live"), [])  # read-only

    def test_fresh_db_has_the_table_at_current_version(self):
        L.finalize_session("fresh", "proj")
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertTrue(L._table_exists(conn, "session_agents"))
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        conn.close()

    def test_v4_database_migrates_to_v5_with_data_intact(self):
        db_path = L.resolve_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, project TEXT NOT NULL, ended_at TEXT NOT NULL, "
            "credited_saved_tokens INTEGER NOT NULL, raw_tokens_sum INTEGER NOT NULL DEFAULT 0, "
            "out_tokens_sum INTEGER NOT NULL DEFAULT 0, event_count INTEGER NOT NULL, fetch_url_count INTEGER NOT NULL, "
            "overhead_tokens INTEGER NOT NULL DEFAULT 0, actual_input_tokens INTEGER NOT NULL DEFAULT 0, "
            "actual_output_tokens INTEGER NOT NULL DEFAULT 0, actual_cache_read_tokens INTEGER NOT NULL DEFAULT 0, "
            "actual_cache_write_tokens INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "CREATE TABLE session_tools (session_id TEXT NOT NULL, tool TEXT NOT NULL, event_count INTEGER NOT NULL, "
            "saved_tokens INTEGER NOT NULL, raw_tokens_sum INTEGER NOT NULL DEFAULT 0, "
            "out_tokens_sum INTEGER NOT NULL DEFAULT 0, credited_event_count INTEGER NOT NULL DEFAULT 0, "
            "PRIMARY KEY (session_id, tool))"
        )
        conn.execute("INSERT INTO sessions (session_id, project, ended_at, credited_saved_tokens, event_count, "
                     "fetch_url_count) VALUES ('old', 'p', '2026-09-01T00:00:00Z', 42, 3, 0)")
        conn.execute("PRAGMA user_version = 4")
        conn.commit()
        conn.close()

        L.finalize_session("new", "p")  # opens the DB -> runs the v5 (and later) migrations

        conn = sqlite3.connect(str(db_path))
        self.assertTrue(L._table_exists(conn, "session_agents"))
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        self.assertEqual(
            conn.execute("SELECT credited_saved_chars FROM sessions WHERE session_id = 'old'").fetchone()[0], 147)
        self.assertEqual(L.query_session_agent_breakdown("old"), [])  # no backfill
        conn.close()


class FormatDetailViewByAgent(unittest.TestCase):
    """Issue #365: the "Main session vs subagents" block in /my-savings detail."""

    _PROJECT_SUMMARY = {"project": "proj"}

    def _render(self, by_agent):
        agg = {"credited_saved_chars": 0, "event_count": 0, "raw_chars_sum": 0,
               "out_chars_sum": 0, "project": "proj"}
        if by_agent is not None:
            agg["by_agent"] = by_agent
        return L.format_detail_view(agg, self._PROJECT_SUMMARY, [], [], [])

    @staticmethod
    def _row(is_sub, agent_type, saved, credited_events):
        # before = 10x the saving, so the figures shown are distinguishable from it.
        return {"is_subagent": is_sub, "agent_type": agent_type, "event_count": credited_events,
                "saved_chars": saved, "raw_chars_sum": saved * 10, "out_chars_sum": saved * 9,
                "credited_event_count": credited_events}

    def test_block_shown_when_a_subagent_earned_savings(self):
        out = self._render([self._row(1, "Explore", 3000, 4), self._row(0, "", 1000, 2)])
        self.assertIn("Main session vs subagents", out)
        main_line = [ln for ln in out.splitlines() if ln.strip().startswith("main session")][0]
        self.assertIn("10,000 → 9,000 chars", main_line)
        self.assertIn("2 events", main_line)
        sub_line = [ln for ln in out.splitlines() if "subagent · Explore" in ln][0]
        self.assertIn("30,000 → 27,000 chars", sub_line)
        self.assertIn("4 events", sub_line)

    def test_block_absent_without_subagents(self):
        self.assertNotIn("Main session vs subagents", self._render([self._row(0, "", 1000, 2)]))

    def test_block_absent_when_by_agent_missing_or_empty(self):
        # Old-shape aggregates (and every pre-#365 caller) don't carry by_agent.
        self.assertNotIn("Main session vs subagents", self._render(None))
        self.assertNotIn("Main session vs subagents", self._render([]))

    def test_main_agent_session_is_not_listed_as_a_subagent(self):
        out = self._render([self._row(0, "my-agent", 500, 1), self._row(1, "Explore", 100, 1)])
        self.assertNotIn("subagent · my-agent", out)

    def test_block_absent_when_the_only_subagent_activity_saved_nothing(self):
        # Regression for the Copilot review on PR #367: a subagent whose only
        # activity was uncredited (e.g. fetch_url) has a by_agent row with 0
        # saved / 0 credited events -- that must NOT render a block of
        # zeros, even though the main session did earn savings.
        out = self._render([self._row(0, "", 900, 1), self._row(1, "Explore", 0, 0)])
        self.assertNotIn("Main session vs subagents", out)
        self.assertNotIn("subagent · Explore", out)

    def test_zero_savings_subagent_type_is_not_listed_beside_a_productive_one(self):
        out = self._render([
            self._row(0, "", 900, 1),
            self._row(1, "Explore", 3000, 4),
            self._row(1, "Idle", 0, 0),
        ])
        self.assertIn("subagent · Explore", out)
        self.assertNotIn("subagent · Idle", out)

    def test_subagent_without_a_type_gets_a_plain_label(self):
        out = self._render([self._row(1, "", 100, 1)])
        self.assertTrue(any(ln.strip().startswith("subagent ") and "·" not in ln for ln in out.splitlines()))


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

    def test_schema_version_includes_v4(self):
        # Was `== 4`; every later bump (v5 for #365) would otherwise have to edit
        # this test just to keep asserting that v4's migration is still present.
        self.assertGreaterEqual(L.SCHEMA_VERSION, 4)
        self.assertIn(4, L._MIGRATIONS)


class FormatActualTokenBlock(unittest.TestCase):
    """_format_actual_token_block: display block for Anthropic-processed token counts."""

    def _agg(self, inp=0, out=0, cr=0, cw=0, saved=0):
        return {
            "actual_input_tokens":       inp,
            "actual_output_tokens":      out,
            "actual_cache_read_tokens":  cr,
            "actual_cache_write_tokens": cw,
            "credited_saved_chars":     saved,
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

    def test_no_compression_rate_line_even_when_saved_nonzero(self):
        # Issue #436: the line divided an estimate by a measurement; it must
        # not come back, whether or not savings were credited.
        for saved in (300, 0):
            lines = L._format_actual_token_block(self._agg(inp=100, out=50, cr=1000, cw=200, saved=saved))
            joined = "\n".join(lines)
            self.assertNotIn("Local compression avoided", joined)
            self.assertNotIn("of total processed", joined)

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
        "credited_saved_chars": 0, "event_count": 0,
        "raw_chars_sum": 0, "out_chars_sum": 0,
        "project": "test-proj",
        "actual_input_tokens": 0, "actual_output_tokens": 0,
        "actual_cache_read_tokens": 0, "actual_cache_write_tokens": 0,
    }
    _PROJECT_SUMMARY = {"project": "test-proj"}

    # A last_n entry with real actual_* data -- the shape query_last_n_sessions
    # returns after the fix.
    _SESSION_WITH_ACTUALS = {
        "ended_at": "2026-09-10T19:00:00Z",
        "saved_chars": 1800,
        "actual_input_tokens": 1000,
        "actual_output_tokens": 200,
        "actual_cache_read_tokens": 5000,
        "actual_cache_write_tokens": 100,
    }
    _SESSION_ALL_ZEROS = {
        "ended_at": "2026-09-08T15:00:00Z",
        "saved_chars": 7200,
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
        old_shape = {"ended_at": "2026-09-01T00:00:00Z", "saved_chars": 500}
        parsed = json.loads(L.format_json(
            self._SESSION_AGG, self._PROJECT_SUMMARY, [], [old_shape], [],
        ))
        row = parsed["last_n_sessions"][0]
        for key in ("actual_input_tokens", "actual_output_tokens",
                    "actual_cache_read_tokens", "actual_cache_write_tokens"):
            self.assertEqual(row[key], 0, f"{key} should be 0 for old-shape entry")


class OrphanedLedgerRecovery(SavingsLedgerTestCase):
    """Issue #306: a session that never reached a graceful SessionEnd (crash,
    force-kill) left its <session_id>.jsonl ledger behind forever -- its
    credited savings never reached savings.db and nothing deleted the file.
    recover_orphaned_sessions() rolls up and deletes any ledger older than the
    shadow-marker TTL. The TTL env var is cleared/pinned per test so a
    developer shell's own CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS can't leak in."""

    def setUp(self):
        super().setUp()
        ttl_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS": ""})
        ttl_patch.start()
        self.addCleanup(ttl_patch.stop)
        self.now = time.time()
        # Derived from the module, never hardcoded -- the default may change.
        self.ttl_seconds = session_id_lib._ttl_hours() * 3600

    def _age(self, session_id, seconds):
        path = L._session_jsonl_path(session_id)
        ts = self.now - seconds
        os.utime(path, (ts, ts))
        return path, ts

    def _claims(self):
        return sorted(L._sessions_dir().glob("*.jsonl.claimed-*"))

    def _session_row(self, session_id):
        conn = L._connect()  # creates the schema if no roll-up ever ran
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        conn.close()
        return row

    def test_stale_orphan_is_rolled_up_and_deleted(self):
        # The regression: on main this ledger was never rolled up or removed.
        L.record_event("crashed", "compress_file", 1000, 100, True, project="proj")
        L.record_event("crashed", "compress_file", 400, 100, True, project="proj")
        path, _ = self._age("crashed", self.ttl_seconds + 60)

        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)

        self.assertFalse(path.exists())
        row = self._session_row("crashed")
        self.assertIsNotNone(row)
        self.assertEqual(row["credited_saved_chars"], 1200)
        self.assertEqual(row["event_count"], 2)
        self.assertEqual(row["project"], "proj")
        self.assertEqual(L.query_project_summary("proj")["total_saved_chars"], 1200)

    def test_ended_at_is_the_ledgers_last_write_not_now(self):
        L.record_event("crashed", "compress_file", 1000, 100, True, project="proj")
        _, ts = self._age("crashed", self.ttl_seconds * 4)
        L.recover_orphaned_sessions(now=self.now)
        self.assertEqual(
            self._session_row("crashed")["ended_at"],
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        )

    def test_fresh_ledger_is_left_alone(self):
        L.record_event("live", "compress_file", 1000, 100, True, project="proj")
        path, _ = self._age("live", self.ttl_seconds - 60)
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 0)
        self.assertTrue(path.exists())
        self.assertIsNone(self._session_row("live"))

    def test_excluded_session_is_never_recovered(self):
        # The SessionEnd hook excludes the session that is ending -- it's
        # finalized by the normal path, with its own cwd-derived project.
        L.record_event("ending", "compress_file", 1000, 100, True, project="proj")
        path, _ = self._age("ending", self.ttl_seconds + 60)
        self.assertEqual(L.recover_orphaned_sessions(exclude_session_id="ending", now=self.now), 0)
        self.assertTrue(path.exists())
        self.assertIsNone(self._session_row("ending"))

    def test_shadow_markers_are_never_touched(self):
        # session_id_lib's markers share this directory by default.
        session_id_lib.record_shadow_marker("other", project="/x/proj")
        marker = session_id_lib._shadow_marker_path("other")
        self.assertEqual(marker.parent, L._sessions_dir())  # same dir, so the skip is really exercised
        ts = self.now - self.ttl_seconds * 2
        os.utime(marker, (ts, ts))
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 0)
        self.assertTrue(marker.exists())
        self.assertIsNone(self._session_row("session_other"))

    def test_project_falls_back_to_unknown(self):
        L.record_event("crashed", "compress_file", 1000, 100, True)  # project=None
        self._age("crashed", self.ttl_seconds + 60)
        L.recover_orphaned_sessions(now=self.now)
        self.assertEqual(self._session_row("crashed")["project"], "unknown")

    def test_empty_ledger_is_deleted_without_a_row(self):
        path = L._session_jsonl_path("empty")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json\n", encoding="utf-8")
        self._age("empty", self.ttl_seconds + 60)
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)
        self.assertFalse(path.exists())
        self.assertIsNone(self._session_row("empty"))

    def test_ttl_env_override_is_honored(self):
        L.record_event("crashed", "compress_file", 1000, 100, True, project="proj")
        path, _ = self._age("crashed", 2 * 3600)  # 2h old: fresh under the default
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 0)
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS": "1"}):
            self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)
        self.assertFalse(path.exists())

    def _leave_claim(self, session_id):
        """What a roll-up that failed or died leaves behind: the ledger renamed
        to a claim (rename keeps its mtime)."""
        path = L._session_jsonl_path(session_id)
        claim = path.with_name(path.name + ".claimed-" + "ab" * 16)
        os.rename(path, claim)
        return claim

    def test_append_after_claim_survives_in_a_fresh_ledger(self):
        # PR #412 review (round 1): an append racing the recovery used to be
        # deleted with the file. The ledger is renamed to a claim before it's
        # read, so the append lands in a NEW ledger at the old path.
        L.record_event("busy", "compress_file", 1000, 100, True, project="proj")
        path, _ = self._age("busy", self.ttl_seconds + 60)
        real_write = L._write_session_rows

        def append_then_write(*args, **kwargs):
            L.record_event("busy", "compress_file", 300, 100, True, project="proj")
            return real_write(*args, **kwargs)

        with mock.patch.object(L, "_write_session_rows", side_effect=append_then_write):
            self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)
        self.assertEqual(self._session_row("busy")["credited_saved_chars"], 900)
        self.assertTrue(path.exists())
        self.assertEqual([e["raw_chars"] for e in L.read_session_events("busy")], [300])
        self.assertEqual(self._claims(), [])

    def test_finalize_session_aggregates_a_given_snapshot_without_rereading(self):
        L.record_event("snap", "compress_file", 1000, 100, True, project="proj")
        snapshot = L.read_session_events("snap")
        L._session_jsonl_path("snap").unlink()  # gone, as after a concurrent recovery
        agg = L.finalize_session("snap", "proj", events=snapshot)
        self.assertEqual(agg["credited_saved_chars"], 900)
        self.assertEqual(self._session_row("snap")["credited_saved_chars"], 900)

    def test_second_owner_finds_nothing_and_leaves_the_row_alone(self):
        # PR #412 review (round 1): a second finalizer re-reading a ledger the
        # first had already consumed wrote an all-zero row over it.
        L.record_event("x", "compress_file", 1000, 100, True, project="proj")
        self._age("x", self.ttl_seconds + 60)
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)  # owner A
        self.assertIsNone(L.finalize_owned_session("x", stale_before=self.now))  # owner B
        self.assertIsNone(L.finalize_owned_session("x", "proj"))  # or the session's own end
        row = self._session_row("x")
        self.assertEqual((row["credited_saved_chars"], row["event_count"]), (900, 1))

    def test_finalizers_are_serialized_on_the_db_write_lock(self):
        # PR #412 review (round 3): per-file claims let two workers each own
        # part of one session. Ownership is now savings.db's write lock: while
        # another finalizer holds it, nobody can take this session's files.
        L.record_event("x", "compress_file", 1000, 100, True, project="proj")
        path, _ = self._age("x", self.ttl_seconds + 60)
        holder = L._connect()
        holder.execute("BEGIN IMMEDIATE")
        real_connect = sqlite3.connect
        try:
            with mock.patch.object(sqlite3, "connect", side_effect=lambda *a, **k: real_connect(*a, timeout=0.05)):
                with self.assertRaises(sqlite3.OperationalError):
                    L.finalize_owned_session("x", stale_before=self.now)
                self.assertEqual(L.recover_orphaned_sessions(now=self.now), 0)
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        self.assertTrue(path.exists())  # untouched -- not claimed, not read
        self.assertEqual(self._claims(), [])
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)  # once the lock is free

    def test_leftover_claim_and_later_ledger_roll_up_together(self):
        # PR #412 review (round 2): a roll-up dies after claiming, the session
        # then writes a fresh ledger, and later both are stale. Rolled up one
        # file at a time, INSERT OR REPLACE kept only the last file's events.
        L.record_event("s", "compress_file", 1000, 100, True, project="proj")  # saves 900
        self._age("s", self.ttl_seconds * 3)
        self._leave_claim("s")
        L.record_event("s", "compress_file", 500, 100, True, project="proj")   # saves 400
        path, _ = self._age("s", self.ttl_seconds * 2)
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)
        row = self._session_row("s")
        self.assertEqual((row["credited_saved_chars"], row["event_count"]), (1300, 2))
        self.assertEqual(self._claims(), [])
        self.assertFalse(path.exists())

    def test_graceful_end_includes_its_own_leftover_claim(self):
        # PR #412 review (round 3): the SessionEnd hook excludes its own
        # session from recovery, so it must roll up that session's leftover
        # claim itself -- otherwise a later sweep finalized the old claim on
        # its own and replaced the newer row with the older subset.
        L.record_event("s", "compress_file", 1000, 100, True, project="proj")
        self._age("s", self.ttl_seconds * 3)
        self._leave_claim("s")
        L.record_event("s", "compress_file", 500, 100, True, project="proj")  # resumed, fresh
        self.assertEqual(L.recover_orphaned_sessions(exclude_session_id="s", now=self.now), 0)
        agg = L.finalize_owned_session("s", "proj")
        self.assertEqual(agg["credited_saved_chars"], 1300)
        self.assertEqual(L.recover_orphaned_sessions(now=self.now + self.ttl_seconds * 2), 0)
        row = self._session_row("s")
        self.assertEqual((row["credited_saved_chars"], row["event_count"]), (1300, 2))
        self.assertEqual(L._session_artifacts("s"), [])

    def test_failed_roll_up_leaves_everything_for_the_next_owner(self):
        L.record_event("f", "compress_file", 1000, 100, True, project="proj")
        self._age("f", self.ttl_seconds + 60)
        with mock.patch.object(L, "_write_session_rows", side_effect=sqlite3.OperationalError("disk I/O error")):
            self.assertEqual(L.recover_orphaned_sessions(now=self.now), 0)
        self.assertIsNone(self._session_row("f"))
        self.assertEqual(len(self._claims()), 1)  # rolled back, events kept
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)
        self.assertEqual(self._session_row("f")["credited_saved_chars"], 900)
        self.assertEqual(self._claims(), [])

    def test_roll_up_that_died_after_commit_is_redone_idempotently(self):
        # Committed but never got to delete its files: the next owner's
        # combined snapshot is the same set, so the row comes out the same.
        L.record_event("d", "compress_file", 1000, 100, True, project="proj")
        self._age("d", self.ttl_seconds + 60)
        with mock.patch.object(Path, "unlink", lambda self, missing_ok=False: None):
            self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)
        self.assertEqual(len(self._claims()), 1)
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)
        row = self._session_row("d")
        self.assertEqual((row["credited_saved_chars"], row["event_count"]), (900, 1))
        self.assertEqual(self._claims(), [])

    def _actual(self, session_id):
        row = self._session_row(session_id)
        return (row["actual_input_tokens"], row["actual_output_tokens"],
                row["actual_cache_read_tokens"], row["actual_cache_write_tokens"])

    def test_recovered_then_resumed_session_accumulates(self):
        # PR #412 review (round 5): a crashed session is recovered (900), later
        # resumed, and its next clean end used to REPLACE the row with only
        # the resumed events (200) -- the recovered savings were lost again.
        L.record_event("s", "compress_file", 1000, 100, True, project="proj")
        self._age("s", self.ttl_seconds + 60)
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)
        L.record_event("s", "compress_file", 300, 100, True, project="proj")   # resumed
        L.record_event("s", "fetch_url", 50, 10, False, project="proj")
        agg = L.finalize_owned_session("s", "proj")
        self.assertEqual(agg["credited_saved_chars"], 200)  # the summary: this run
        row = self._session_row("s")
        self.assertEqual(
            (row["credited_saved_chars"], row["raw_chars_sum"], row["out_chars_sum"],
             row["event_count"], row["fetch_url_count"]),
            (1100, 1300, 200, 2, 1),
        )
        tools = {t["tool"]: t for t in L.query_session_tool_breakdown("s")}
        self.assertEqual(tools["compress_file"]["saved_chars"], 1100)
        self.assertEqual(L.query_project_summary("proj")["total_saved_chars"], 1100)
        conn = L._connect()
        agents = conn.execute(
            "SELECT event_count, saved_chars FROM session_agents WHERE session_id = 's'"
        ).fetchall()
        conn.close()
        self.assertEqual(agents, [(3, 1100)])

    def test_session_ending_once_writes_the_same_row_as_before(self):
        # The accumulate path must not change a normal, single-end session.
        for sid in ("plain", "owned"):
            L.record_event(sid, "compress_file", 1000, 100, True, project="proj")
            L.record_event(sid, "fetch_url", 50, 10, False, project="proj")
        counts = {"input": 1, "output": 2, "cache_read": 3, "cache_write": 4}
        L.finalize_session("plain", "proj", actual_tokens=counts)
        L.finalize_owned_session("owned", "proj", actual_tokens=counts)
        skip = {"session_id", "ended_at"}
        plain, owned = self._session_row("plain"), self._session_row("owned")
        self.assertEqual({k: plain[k] for k in plain.keys() if k not in skip},
                         {k: owned[k] for k in owned.keys() if k not in skip})
        strip = lambda rows: [{k: v for k, v in r.items() if k != "session_id"} for r in rows]  # noqa: E731
        self.assertEqual(strip(L.query_session_tool_breakdown("plain")),
                         strip(L.query_session_tool_breakdown("owned")))

    def test_resumed_session_transcript_counts(self):
        # A resumed session reuses its transcript, so a later parse covers the
        # whole session and replaces earlier counts; no parse (None) keeps the
        # counts already recorded for this session rather than zeroing them.
        L.record_event("r", "compress_file", 1000, 100, True, project="proj")
        L.finalize_owned_session("r", "proj", actual_tokens={"input": 11, "output": 22, "cache_read": 33, "cache_write": 44})
        L.record_event("r", "compress_file", 500, 100, True, project="proj")
        L.finalize_owned_session("r", "proj", actual_tokens=None)
        self.assertEqual(self._session_row("r")["credited_saved_chars"], 1300)
        self.assertEqual(self._actual("r"), (11, 22, 33, 44))
        L.record_event("r", "compress_file", 200, 100, True, project="proj")
        L.finalize_owned_session("r", "proj", actual_tokens={"input": 50, "output": 60, "cache_read": 70, "cache_write": 80})
        self.assertEqual(self._session_row("r")["credited_saved_chars"], 1400)
        self.assertEqual(self._actual("r"), (50, 60, 70, 80))

    def test_redo_after_commit_never_counts_a_claim_twice(self):
        # Accumulating must not double-count a claim whose roll-up committed
        # but died before deleting it -- even with an existing row and a new
        # ledger alongside it.
        L.record_event("d", "compress_file", 1000, 100, True, project="proj")
        L.finalize_owned_session("d", "proj")                                    # 900, first run
        L.record_event("d", "compress_file", 500, 100, True, project="proj")
        with mock.patch.object(Path, "unlink", lambda self, missing_ok=False: None):
            L.finalize_owned_session("d", "proj")                                # +400, died before delete
        self.assertEqual(len(self._claims()), 1)
        L.record_event("d", "compress_file", 300, 100, True, project="proj")    # +200, resumed again
        L.finalize_owned_session("d", "proj")
        row = self._session_row("d")
        self.assertEqual((row["credited_saved_chars"], row["event_count"]), (1500, 3))
        self.assertEqual(self._claims(), [])
        self.assertEqual(L.get_meta("consumed_claim:" + "x", None), None)
        conn = L._connect()
        leftover = conn.execute("SELECT COUNT(*) FROM meta WHERE key LIKE 'consumed_claim:%'").fetchone()[0]
        conn.close()
        self.assertEqual(leftover, 0)  # bookkeeping cleaned up once the files are gone

    def test_recovered_project_is_the_newest_events_not_the_last_file_read(self):
        # PR #412 review (round 6): artifacts are listed by filename, so the
        # ledger "s.jsonl" is read before the older "s.jsonl.claimed-*", and
        # taking the last event in list order filed the row under the OLD
        # claim's project after the resumed session moved projects.
        old = {"ts": "2026-01-01T00:00:00Z", "tool": "compress_file", "credited": True,
               "raw_chars": 1000, "out_chars": 100, "saved_chars": 900, "project": "old-proj"}
        new = dict(old, ts="2026-01-02T00:00:00Z", project="new-proj")
        path = L._session_jsonl_path("s")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(old) + "\n", encoding="utf-8")
        self._age("s", self.ttl_seconds * 3)
        self._leave_claim("s")
        path.write_text(json.dumps(new) + "\n", encoding="utf-8")
        self._age("s", self.ttl_seconds * 2)
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)
        self.assertEqual(self._session_row("s")["project"], "new-proj")

    def test_orphan_project_ties_and_missing_ts(self):
        self.assertEqual(L._orphan_project([]), "unknown")
        self.assertEqual(L._orphan_project([{"project": "a"}, {"project": "b"}]), "b")  # no ts: position
        same = "2026-01-01T00:00:00Z"
        self.assertEqual(L._orphan_project([{"ts": same, "project": "a"}, {"ts": same, "project": "b"}]), "b")
        self.assertEqual(L._orphan_project([{"ts": "2026-01-02T00:00:00Z", "project": "a"},
                                            {"ts": "2026-01-01T00:00:00Z", "project": "b"},
                                            {"ts": "2026-01-03T00:00:00Z", "project": ""}]), "a")

    def test_claim_only_session_counts_as_having_artifacts(self):
        # PR #412 review (round 4): the SessionEnd hook checked only the
        # ledger, so a session whose only artifact was a leftover claim exited
        # before its own finalize (and recovery skips the ending session).
        L.record_event("c", "compress_file", 1000, 100, True, project="proj")
        self._leave_claim("c")
        self.assertEqual(L.read_session_events("c"), [])
        self.assertTrue(L.has_session_artifacts("c"))
        self.assertEqual(L.finalize_owned_session("c", "proj")["credited_saved_chars"], 900)
        self.assertFalse(L.has_session_artifacts("c"))

    def test_live_again_by_the_time_the_lock_is_held_is_skipped(self):
        L.record_event("l", "compress_file", 1000, 100, True, project="proj")
        path = L._session_jsonl_path("l")  # fresh mtime: written just now
        self.assertIsNone(L.finalize_owned_session("l", stale_before=self.now - self.ttl_seconds))
        self.assertTrue(path.exists())
        self.assertIsNone(self._session_row("l"))

    def test_claims_are_invisible_to_current_session_id(self):
        L.record_event("c", "compress_file", 1000, 100, True, project="proj")
        self._leave_claim("c")
        self.assertIsNone(L.current_session_id())

    def test_one_bad_ledger_does_not_block_the_rest(self):
        for sid in ("a", "b"):
            L.record_event(sid, "compress_file", 1000, 100, True, project="proj")
            self._age(sid, self.ttl_seconds + 60)
        real_write = L._write_session_rows

        def fail_for_a(conn, session_id, *args, **kwargs):
            if session_id == "a":
                raise sqlite3.OperationalError("disk I/O error")
            return real_write(conn, session_id, *args, **kwargs)

        with mock.patch.object(L, "_write_session_rows", side_effect=fail_for_a):
            self.assertEqual(L.recover_orphaned_sessions(now=self.now), 1)
        self.assertEqual(len(L._session_artifacts("a")), 1)  # kept for a later retry
        self.assertIsNone(self._session_row("a"))
        self.assertIsNotNone(self._session_row("b"))

    def test_missing_sessions_dir_is_a_noop(self):
        self.assertFalse(L._sessions_dir().exists())
        self.assertEqual(L.recover_orphaned_sessions(now=self.now), 0)


class ExactCharCountsAndToolUseId(SavingsLedgerTestCase):
    """Issues #437/#438: exact character counts and the tool-call id are the only
    size fields recorded, rolled up at SessionEnd and in orphan recovery. A
    pre-v6 database migrates with a tokens x 3.5 backfill, then v7 drops the
    estimated-token columns."""

    def test_event_stores_exact_chars_and_no_token_estimates(self):
        L.record_event("s", "hook:Bash", 3501, 351, True, "cmd")
        e = L.read_session_events("s")[0]
        self.assertEqual((e["raw_chars"], e["out_chars"], e["saved_chars"]), (3501, 351, 3150))
        for key in ("raw_tokens", "out_tokens", "saved_tokens"):
            self.assertNotIn(key, e)

    def test_saved_chars_clamped_and_zero_when_uncredited(self):
        L.record_event("s", "hook:Bash", 350, 525, True)
        L.record_event("s", "fetch_url", 350, 35, False)
        clamped, uncredited = L.read_session_events("s")
        self.assertEqual(clamped["saved_chars"], 0)
        self.assertEqual((uncredited["raw_chars"], uncredited["saved_chars"]), (350, 0))

    def test_tool_use_id_stored_when_present_and_empty_when_absent(self):
        L.record_event("s", "hook:Bash", 1000, 100, True, tool_use_id="toolu_01")
        L.record_event("s", "hook:Bash", 1000, 100, True, tool_use_id="")
        L.record_event("s", "hook:Bash", 1000, 100, True, tool_use_id=None)
        with_id, empty, none = L.read_session_events("s")
        self.assertEqual(with_id["tool_use_id"], "toolu_01")
        self.assertNotIn("tool_use_id", empty)
        self.assertNotIn("tool_use_id", none)

    def _append_legacy_event(self, session_id):
        """A line as a pre-#438 build wrote it: estimated tokens (and, from #437,
        sometimes chars) but none of the fields this version reads."""
        path = L._session_jsonl_path(session_id)
        legacy = {"ts": "2026-10-01T00:00:00Z", "tool": "hook:Bash", "raw_tokens": 10, "out_tokens": 10,
                  "saved_tokens": 0, "credited": True, "source": "", "project": None}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(legacy) + "\n")

    def test_finalize_rolls_chars_up_into_all_three_tables(self):
        L.record_event("s", "hook:Bash", 3500, 350, True)
        L.record_event("s", "hook:Bash", 7000, 700, True, agent_id="a1", agent_type="Explore")
        L.record_event("s", "fetch_url", 17500, 175, False)  # uncredited: excluded
        self._append_legacy_event("s")                       # no char fields: contributes nothing
        agg = L.finalize_session("s", "proj")
        self.assertEqual((agg["raw_chars_sum"], agg["out_chars_sum"], agg["credited_saved_chars"]),
                         (10500, 1050, 9450))
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(
            conn.execute("SELECT raw_chars_sum, out_chars_sum, credited_saved_chars FROM sessions").fetchone(),
            (10500, 1050, 9450))
        self.assertEqual(
            conn.execute("SELECT raw_chars_sum, out_chars_sum, saved_chars FROM session_tools "
                         "WHERE tool = 'hook:Bash'").fetchone(), (10500, 1050, 9450))
        self.assertEqual(
            conn.execute("SELECT raw_chars_sum, out_chars_sum, saved_chars FROM session_tools "
                         "WHERE tool = 'fetch_url'").fetchone(), (0, 0, 0))
        agents = {(r[0], r[1]): r[2:] for r in conn.execute(
            "SELECT is_subagent, agent_type, raw_chars_sum, out_chars_sum, saved_chars FROM session_agents")}
        self.assertEqual(agents[(1, "Explore")], (7000, 700, 6300))
        self.assertEqual(agents[(0, "")], (3500, 350, 3150))
        conn.close()

    def test_live_aggregate_reports_chars(self):
        L.record_event("s", "hook:Bash", 3500, 350, True)
        agg = L.get_live_session_aggregate("s")
        self.assertEqual((agg["raw_chars_sum"], agg["out_chars_sum"], agg["credited_saved_chars"]),
                         (3500, 350, 3150))
        self.assertEqual(agg["by_tool"]["hook:Bash"]["saved_chars"], 3150)

    def test_aggregates_carry_no_estimated_token_fields(self):
        L.record_event("s", "hook:Bash", 3500, 350, True, agent_id="a1", agent_type="Explore")
        agg = L.get_live_session_aggregate("s")
        flat_keys = set(agg) | set(agg["by_tool"]["hook:Bash"]) | set(agg["by_agent"][0])
        self.assertFalse(
            {k for k in flat_keys if k in ("raw_tokens_sum", "out_tokens_sum", "saved_tokens", "credited_saved_tokens")},
            flat_keys)

    def test_corrupt_char_field_does_not_sink_the_rollup(self):
        L.record_event("s", "hook:Bash", 3500, 350, True)
        path = L._session_jsonl_path("s")
        entry = json.loads(path.read_text().splitlines()[0])
        entry["raw_chars"] = "lots"
        path.write_text(json.dumps(entry) + "\n")
        agg = L.finalize_session("s", "proj")
        # An event whose size can't be trusted is not counted at all, so the figures
        # (and the event count beside them) never describe a partial set.
        self.assertEqual((agg["raw_chars_sum"], agg["credited_saved_chars"], agg["event_count"]), (0, 0, 0))

    def test_orphan_recovery_rolls_chars_up(self):
        ttl_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS": ""})
        ttl_patch.start()
        self.addCleanup(ttl_patch.stop)
        now = time.time()
        L.record_event("crashed", "compress_file", 3500, 350, True, project="proj")
        ts = now - session_id_lib._ttl_hours() * 3600 - 60
        os.utime(L._session_jsonl_path("crashed"), (ts, ts))
        self.assertEqual(L.recover_orphaned_sessions(now=now), 1)
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(
            conn.execute("SELECT raw_chars_sum, out_chars_sum, credited_saved_chars FROM sessions "
                         "WHERE session_id = 'crashed'").fetchone(), (3500, 350, 3150))
        conn.close()

    def test_accumulating_rollups_add_chars(self):
        L.record_event("s", "hook:Bash", 3500, 350, True)
        L.finalize_owned_session("s", "proj")
        L.record_event("s", "hook:Bash", 3500, 350, True)
        L.finalize_owned_session("s", "proj")
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(
            conn.execute("SELECT raw_chars_sum, credited_saved_chars FROM sessions").fetchone(), (7000, 6300))
        self.assertEqual(
            conn.execute("SELECT raw_chars_sum, saved_chars FROM session_tools").fetchone(), (7000, 6300))
        conn.close()

    def _make_v5_db(self):
        """A database exactly as v5 left it: no character columns anywhere."""
        conn = sqlite3.connect(str(L.resolve_db_path()))
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, project TEXT NOT NULL, ended_at TEXT NOT NULL, "
            "credited_saved_tokens INTEGER NOT NULL, raw_tokens_sum INTEGER NOT NULL DEFAULT 0, "
            "out_tokens_sum INTEGER NOT NULL DEFAULT 0, event_count INTEGER NOT NULL, fetch_url_count INTEGER NOT NULL, "
            "overhead_tokens INTEGER NOT NULL DEFAULT 0, actual_input_tokens INTEGER NOT NULL DEFAULT 0, "
            "actual_output_tokens INTEGER NOT NULL DEFAULT 0, actual_cache_read_tokens INTEGER NOT NULL DEFAULT 0, "
            "actual_cache_write_tokens INTEGER NOT NULL DEFAULT 0)")
        conn.execute(
            "CREATE TABLE session_tools (session_id TEXT NOT NULL, tool TEXT NOT NULL, event_count INTEGER NOT NULL, "
            "saved_tokens INTEGER NOT NULL, raw_tokens_sum INTEGER NOT NULL DEFAULT 0, "
            "out_tokens_sum INTEGER NOT NULL DEFAULT 0, credited_event_count INTEGER NOT NULL DEFAULT 0, "
            "PRIMARY KEY (session_id, tool))")
        conn.execute(
            "CREATE TABLE session_agents (session_id TEXT NOT NULL, is_subagent INTEGER NOT NULL, "
            "agent_type TEXT NOT NULL DEFAULT '', event_count INTEGER NOT NULL, saved_tokens INTEGER NOT NULL, "
            "raw_tokens_sum INTEGER NOT NULL DEFAULT 0, out_tokens_sum INTEGER NOT NULL DEFAULT 0, "
            "credited_event_count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (session_id, is_subagent, agent_type))")
        # known before/after (odd token counts so x3.5 lands on a .5 and rounding is exercised)
        conn.execute("INSERT INTO sessions (session_id, project, ended_at, credited_saved_tokens, raw_tokens_sum, "
                     "out_tokens_sum, event_count, fetch_url_count, overhead_tokens, actual_input_tokens) "
                     "VALUES ('known', 'p', '2026-09-01T00:00:00Z', 900, 1001, 101, 3, 0, 55, 777)")
        # v1-era row: saved known, before/after unknown (0)
        conn.execute("INSERT INTO sessions (session_id, project, ended_at, credited_saved_tokens, event_count, "
                     "fetch_url_count) VALUES ('legacy', 'p', '2026-08-01T00:00:00Z', 42, 3, 0)")
        # v2-era tool row: raw known, out unknown
        conn.execute("INSERT INTO session_tools (session_id, tool, event_count, saved_tokens, raw_tokens_sum) "
                     "VALUES ('legacy', 'hook:Bash', 3, 42, 100)")
        conn.execute("INSERT INTO session_agents (session_id, is_subagent, agent_type, event_count, saved_tokens, "
                     "raw_tokens_sum, out_tokens_sum) VALUES ('known', 1, 'Explore', 2, 300, 400, 100)")
        conn.execute("PRAGMA user_version = 5")
        conn.commit()
        conn.close()

    def test_v5_database_migrates_with_tokens_times_3_5_backfill(self):
        self._make_v5_db()
        conn = L._connect()  # runs the v6 migration (backfill), then v7 (drops the token columns), then v8/v9
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        sess = {r[0]: r[1:] for r in conn.execute(
            "SELECT session_id, raw_chars_sum, out_chars_sum, credited_saved_chars FROM sessions")}
        # 1001 * 3.5 = 3503.5 -> 3504 (ROUND half away from zero); 101 * 3.5 = 353.5 -> 354; 900 * 3.5 = 3150
        self.assertEqual(sess["known"], (3504, 354, 3150))
        # unknown before/after stay unknown (0); the saved figure is still backfilled (42 * 3.5 = 147)
        self.assertEqual(sess["legacy"], (0, 0, 147))
        tool = conn.execute("SELECT raw_chars_sum, out_chars_sum, saved_chars FROM session_tools").fetchone()
        self.assertEqual(tool, (350, 0, 147))
        agent = conn.execute("SELECT raw_chars_sum, out_chars_sum, saved_chars FROM session_agents").fetchone()
        self.assertEqual(agent, (1400, 350, 1050))
        conn.close()

    def test_v7_drops_estimated_token_columns_and_keeps_everything_else(self):
        self._make_v5_db()
        conn = L._connect()
        dropped = {"credited_saved_tokens", "raw_tokens_sum", "out_tokens_sum", "saved_tokens"}
        for table in ("sessions", "session_tools", "session_agents"):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            self.assertFalse(cols & dropped, (table, cols & dropped))
        # Real transcript usage is NOT an estimate and stays. (The schema-overhead
        # estimate column went in v8, issue #439 -- see SchemaOverheadRemovedV8.)
        row = conn.execute(
            "SELECT actual_input_tokens, event_count FROM sessions WHERE session_id = 'known'"
        ).fetchone()
        self.assertEqual(row, (777, 3))
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
        self.assertTrue({"actual_input_tokens", "actual_output_tokens", "actual_cache_read_tokens",
                         "actual_cache_write_tokens"} <= cols)
        self.assertEqual(conn.execute("SELECT credited_event_count FROM session_tools").fetchone()[0], 0)
        conn.close()

    def test_v7_rebuilt_tables_keep_their_keys(self):
        # The rebuilds must keep the PRIMARY KEYs the upserts in _write_session_rows
        # (ON CONFLICT(...)) depend on.
        self._make_v5_db()
        L.record_event("known", "hook:Bash", 3500, 350, True)
        L.finalize_owned_session("known", "p")
        L.record_event("known", "hook:Bash", 3500, 350, True)
        L.finalize_owned_session("known", "p")  # second roll-up hits ON CONFLICT on all three tables
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(
            conn.execute("SELECT raw_chars_sum, credited_saved_chars FROM sessions "
                         "WHERE session_id = 'known'").fetchone(), (3504 + 7000, 3150 + 6300))
        self.assertEqual(
            conn.execute("SELECT raw_chars_sum, saved_chars FROM session_tools "
                         "WHERE session_id = 'known' AND tool = 'hook:Bash'").fetchone(), (7000, 6300))
        conn.close()

    def test_v6_database_with_both_column_sets_migrates_to_v7(self):
        # A DB at v6 has the token columns AND the char columns side by side.
        self._make_v5_db()
        conn = sqlite3.connect(str(L.resolve_db_path()))
        conn.close()
        with mock.patch.object(L, "SCHEMA_VERSION", 6):
            L._connect().close()
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 6)
        self.assertIn("raw_tokens_sum", {r[1] for r in conn.execute("PRAGMA table_info(sessions)")})
        conn.close()
        with mock.patch.object(L, "SCHEMA_VERSION", 7):
            L._connect().close()  # now v7
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 7)
        self.assertNotIn("raw_tokens_sum", {r[1] for r in conn.execute("PRAGMA table_info(sessions)")})
        self.assertEqual(
            conn.execute("SELECT credited_saved_chars FROM sessions WHERE session_id = 'known'").fetchone(), (3150,))
        conn.close()

    def test_migrated_ledger_still_rolls_up_new_sessions_and_keeps_old_data(self):
        self._make_v5_db()
        L.record_event("new", "hook:Bash", 3500, 350, True)
        L.finalize_session("new", "p")
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(
            conn.execute("SELECT credited_saved_chars, raw_chars_sum FROM sessions "
                         "WHERE session_id = 'known'").fetchone(), (3150, 3504))
        self.assertEqual(
            conn.execute("SELECT credited_saved_chars FROM sessions WHERE session_id = 'new'").fetchone(), (3150,))
        conn.close()
        self.assertEqual(L.query_project_summary("p")["sessions"], 3)

    def test_migration_does_not_overwrite_existing_char_columns(self):
        # A DB whose table already has the column (e.g. created by this file's own
        # CREATE TABLE while user_version lagged) must keep its real exact counts.
        self._make_v5_db()
        conn = sqlite3.connect(str(L.resolve_db_path()))
        conn.execute("ALTER TABLE sessions ADD COLUMN raw_chars_sum INTEGER NOT NULL DEFAULT 0")
        # 3500 chars is what 1001 estimated tokens come from (x 3.5 = 3503.5); the v6 backfill
        # would have written 3504 over it.
        conn.execute("UPDATE sessions SET raw_chars_sum = 3500 WHERE session_id = 'known'")
        conn.commit()
        conn.close()
        conn = L._connect()
        self.assertEqual(
            conn.execute("SELECT raw_chars_sum FROM sessions WHERE session_id = 'known'").fetchone(), (3500,))
        conn.close()

    def test_fresh_database_has_char_columns_and_no_token_estimates(self):
        conn = L._connect()
        for table in ("sessions", "session_tools", "session_agents"):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            self.assertTrue({"raw_chars_sum", "out_chars_sum"} <= cols, table)
            self.assertFalse(cols & {"raw_tokens_sum", "out_tokens_sum", "saved_tokens", "credited_saved_tokens"}, table)
        conn.close()


class ConsistentCharacterFigures(SavingsLedgerTestCase):
    """PR #446 review: the figures shown together must describe the same events
    (counted events only), the saving must agree with the before/after beside it,
    and history with no recorded size must be reported as left out rather than
    folded into a total that looks complete."""

    def _append_legacy_event(self, session_id):
        path = L._session_jsonl_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tool": "hook:Bash", "raw_tokens": 10, "out_tokens": 10,
                                "saved_tokens": 0, "credited": True}) + "\n")

    def _insert_unsized_session(self, session_id, project, ended_at="2026-01-01T00:00:00Z"):
        """A pre-v6 row: credited events, but before/after unknown (0)."""
        conn = L._connect()
        conn.execute(
            "INSERT INTO sessions (session_id, project, ended_at, credited_saved_chars, event_count, fetch_url_count) "
            "VALUES (?, ?, ?, 147, 3, 0)", (session_id, project, ended_at))
        conn.commit()
        conn.close()

    # -- finding 1: only events with a recorded size are counted --

    def test_credited_event_without_chars_is_not_counted_anywhere(self):
        L.record_event("s", "hook:Bash", 3500, 350, True)
        self._append_legacy_event("s")
        live = L.get_live_session_aggregate("s")
        self.assertEqual(live["event_count"], 1)
        self.assertEqual(live["by_tool"]["hook:Bash"]["credited_event_count"], 1)
        self.assertEqual(live["by_tool"]["hook:Bash"]["event_count"], 2)  # still a call that happened
        self.assertEqual(live["by_agent"][0]["credited_event_count"], 1)
        agg = L.finalize_session("s", "proj")
        self.assertEqual(agg["event_count"], 1)
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(conn.execute("SELECT event_count FROM sessions").fetchone(), (1,))
        self.assertEqual(
            conn.execute("SELECT credited_event_count FROM session_tools").fetchone(), (1,))
        conn.close()

    def test_a_session_of_only_unsized_events_has_no_counted_events(self):
        self._append_legacy_event("s")
        self.assertEqual(L.get_live_session_aggregate("s")["event_count"], 0)

    # -- finding 2: the saving is derived from before/after, never summed --

    def test_saving_is_derived_from_the_totals_not_summed_per_event(self):
        L.record_event("s", "hook:Bash", 100, 150, True)   # expands: its own saving is clamped to 0
        L.record_event("s", "hook:Bash", 100, 10, True)    # saves 90 on its own
        # 200 -> 160 is a saving of 40, not 0 + 90.
        live = L.get_live_session_aggregate("s")
        self.assertEqual(live["credited_saved_chars"], 40)
        self.assertEqual(live["by_tool"]["hook:Bash"]["saved_chars"], 40)
        self.assertEqual(live["by_agent"][0]["saved_chars"], 40)
        agg = L.finalize_session("s", "proj")
        self.assertEqual(agg["credited_saved_chars"], 40)
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(conn.execute("SELECT credited_saved_chars FROM sessions").fetchone(), (40,))
        self.assertEqual(conn.execute("SELECT saved_chars FROM session_tools").fetchone(), (40,))
        self.assertEqual(conn.execute("SELECT saved_chars FROM session_agents").fetchone(), (40,))
        conn.close()
        self.assertEqual(L.query_project_summary("proj")["best_session_chars"], 40)
        self.assertEqual(L.query_project_summary("proj")["total_saved_chars"], 40)
        self.assertEqual(L.query_last_n_sessions("proj")[0]["saved_chars"], 40)

    def test_accumulating_rollups_rederive_the_saving_from_the_summed_totals(self):
        L.record_event("s", "hook:Bash", 100, 150, True)   # first roll-up: 100 -> 150, saves 0
        L.finalize_owned_session("s", "proj")
        L.record_event("s", "hook:Bash", 100, 10, True)    # second: 100 -> 10, saves 90
        L.finalize_owned_session("s", "proj")
        # Together 200 -> 160 = 40; adding the two clamped savings would give 90.
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(
            conn.execute("SELECT raw_chars_sum, out_chars_sum, credited_saved_chars FROM sessions").fetchone(),
            (200, 160, 40))
        self.assertEqual(conn.execute("SELECT saved_chars FROM session_tools").fetchone(), (40,))
        self.assertEqual(conn.execute("SELECT saved_chars FROM session_agents").fetchone(), (40,))
        conn.close()

    def test_trend_and_all_projects_derive_the_saving_from_their_totals(self):
        L.record_event("a", "hook:Bash", 100, 150, True)
        L.finalize_session("a", "proj")
        L.record_event("b", "hook:Bash", 100, 10, True)
        L.finalize_session("b", "proj")
        (bucket,) = L.query_trend("proj", bucket="week")
        self.assertEqual((bucket["total_raw_chars"], bucket["total_out_chars"], bucket["total_saved_chars"]),
                         (200, 160, 40))
        (row,) = [r for r in L.query_all_projects() if r["project"] == "proj"]
        self.assertEqual(row["total_saved_chars"], 40)

    # -- finding 3: history with no recorded size is reported, not folded in --

    def test_project_summary_leaves_unsized_sessions_out_and_counts_them(self):
        self._insert_unsized_session("legacy", "p")
        L.record_event("new", "hook:Bash", 3500, 350, True)
        L.finalize_session("new", "p")
        ps = L.query_project_summary("p")
        self.assertEqual(ps["sessions"], 2)
        self.assertEqual((ps["total_raw_chars"], ps["total_out_chars"]), (3500, 350))
        self.assertEqual(ps["unsized_sessions"], 1)

    def test_simple_view_says_how_many_sessions_were_left_out(self):
        self._insert_unsized_session("legacy", "p")
        L.record_event("new", "hook:Bash", 3500, 350, True)
        L.finalize_session("new", "p")
        text = L.format_simple_view({"project": "p", "event_count": 0}, L.query_project_summary("p"))
        self.assertIn("Note: 1 of 2 sessions have no recorded size and are left out", text)

    def test_simple_view_has_no_note_when_every_session_is_sized(self):
        L.record_event("new", "hook:Bash", 3500, 350, True)
        L.finalize_session("new", "p")
        text = L.format_simple_view({"project": "p", "event_count": 0}, L.query_project_summary("p"))
        self.assertNotIn("no recorded size", text)

    def test_a_row_with_before_known_but_after_unknown_is_not_a_hundred_percent(self):
        # v2-era shape after the v6 backfill: before known, after 0 (unknown).
        conn = L._connect()
        conn.execute(
            "INSERT INTO sessions (session_id, project, ended_at, credited_saved_chars, raw_chars_sum, "
            "out_chars_sum, event_count, fetch_url_count) VALUES ('v2', 'p', '2026-01-01T00:00:00Z', 147, 350, 0, 3, 0)")
        conn.commit()
        conn.close()
        ps = L.query_project_summary("p")
        self.assertEqual((ps["total_raw_chars"], ps["total_out_chars"], ps["unsized_sessions"]), (0, 0, 1))
        self.assertIn("n/a", L.format_simple_view({"project": "p"}, ps))

    def test_trend_buckets_count_unsized_sessions_and_note_them(self):
        self._insert_unsized_session("legacy", "p", ended_at="2026-09-07T10:00:00Z")
        conn = L._connect()
        conn.execute(
            "INSERT INTO sessions (session_id, project, ended_at, credited_saved_chars, raw_chars_sum, "
            "out_chars_sum, event_count, fetch_url_count) VALUES ('new', 'p', '2026-09-08T10:00:00Z', 900, 1000, 100, 1, 0)")
        conn.commit()
        conn.close()
        (bucket,) = L.query_trend("p", bucket="week")
        self.assertEqual((bucket["sessions"], bucket["unsized_sessions"]), (2, 1))
        self.assertEqual((bucket["total_raw_chars"], bucket["total_out_chars"]), (1000, 100))
        self.assertIn("Note: 1 of 2 sessions have no recorded size", L.format_trend_view([bucket], "p"))

    def test_all_projects_counts_unsized_sessions(self):
        self._insert_unsized_session("legacy", "p")
        L.record_event("new", "hook:Bash", 3500, 350, True)
        L.finalize_session("new", "p")
        (row,) = L.query_all_projects()
        self.assertEqual((row["sessions"], row["unsized_sessions"]), (2, 1))
        self.assertEqual((row["total_raw_chars"], row["total_out_chars"]), (3500, 350))
        out = L.format_detail_view({"project": "p"}, {"project": "p"}, [], [], [row])
        self.assertIn("1 unsized session(s) left out", out)

    def test_unsized_sessions_reach_json_and_csv(self):
        ps = {"project": "p", "sessions": 2, "unsized_sessions": 1}
        parsed = json.loads(L.format_json({"project": "p"}, ps, [], [], [dict(ps, best_tool=None)]))
        self.assertEqual(parsed["project_summary"]["unsized_sessions"], 1)
        self.assertEqual(parsed["all_projects"][0]["unsized_sessions"], 1)
        rows = list(_csv_mod.reader(_io_mod.StringIO(L.format_csv({"project": "p"}, ps, [], [], [ps], table="projects"))))
        self.assertEqual(dict(zip(rows[0], rows[1], strict=False))["unsized_sessions"], "1")
        rows = list(_csv_mod.reader(_io_mod.StringIO(L.format_csv({"project": "p"}, ps, [], [], [], table="session"))))
        self.assertEqual(dict(zip(rows[0], rows[1], strict=False))["project_unsized_sessions"], "1")


    # -- round 2: migration re-derives, counts match their metrics, exports carry the percent --

    def _make_v6_db(self):
        """A v6 database: token AND character columns, with the saving stored as the
        SUM of per-event clamped savings (events 100->150 and 100->10 -> 90, although
        200 -> 160 saves 40), plus a pre-v6 row with no recorded size."""
        conn = sqlite3.connect(str(L.resolve_db_path()))
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, project TEXT NOT NULL, ended_at TEXT NOT NULL, "
            "credited_saved_tokens INTEGER NOT NULL, raw_tokens_sum INTEGER NOT NULL DEFAULT 0, "
            "out_tokens_sum INTEGER NOT NULL DEFAULT 0, event_count INTEGER NOT NULL, fetch_url_count INTEGER NOT NULL, "
            "overhead_tokens INTEGER NOT NULL DEFAULT 0, actual_input_tokens INTEGER NOT NULL DEFAULT 0, "
            "actual_output_tokens INTEGER NOT NULL DEFAULT 0, actual_cache_read_tokens INTEGER NOT NULL DEFAULT 0, "
            "actual_cache_write_tokens INTEGER NOT NULL DEFAULT 0, raw_chars_sum INTEGER NOT NULL DEFAULT 0, "
            "out_chars_sum INTEGER NOT NULL DEFAULT 0, credited_saved_chars INTEGER NOT NULL DEFAULT 0)")
        conn.execute(
            "CREATE TABLE session_tools (session_id TEXT NOT NULL, tool TEXT NOT NULL, event_count INTEGER NOT NULL, "
            "saved_tokens INTEGER NOT NULL, raw_tokens_sum INTEGER NOT NULL DEFAULT 0, "
            "out_tokens_sum INTEGER NOT NULL DEFAULT 0, credited_event_count INTEGER NOT NULL DEFAULT 0, "
            "raw_chars_sum INTEGER NOT NULL DEFAULT 0, out_chars_sum INTEGER NOT NULL DEFAULT 0, "
            "saved_chars INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (session_id, tool))")
        conn.execute(
            "CREATE TABLE session_agents (session_id TEXT NOT NULL, is_subagent INTEGER NOT NULL, "
            "agent_type TEXT NOT NULL DEFAULT '', event_count INTEGER NOT NULL, saved_tokens INTEGER NOT NULL, "
            "raw_tokens_sum INTEGER NOT NULL DEFAULT 0, out_tokens_sum INTEGER NOT NULL DEFAULT 0, "
            "credited_event_count INTEGER NOT NULL DEFAULT 0, raw_chars_sum INTEGER NOT NULL DEFAULT 0, "
            "out_chars_sum INTEGER NOT NULL DEFAULT 0, saved_chars INTEGER NOT NULL DEFAULT 0, "
            "PRIMARY KEY (session_id, is_subagent, agent_type))")
        conn.execute("INSERT INTO sessions (session_id, project, ended_at, credited_saved_tokens, event_count, "
                     "fetch_url_count, raw_chars_sum, out_chars_sum, credited_saved_chars) "
                     "VALUES ('sized', 'p', '2026-09-01T00:00:00Z', 26, 2, 0, 200, 160, 90)")
        conn.execute("INSERT INTO sessions (session_id, project, ended_at, credited_saved_tokens, event_count, "
                     "fetch_url_count, credited_saved_chars) "
                     "VALUES ('unsized', 'p', '2026-08-01T00:00:00Z', 42, 3, 0, 147)")
        conn.execute("INSERT INTO session_tools (session_id, tool, event_count, saved_tokens, credited_event_count, "
                     "raw_chars_sum, out_chars_sum, saved_chars) VALUES ('sized', 'hook:Bash', 2, 26, 2, 200, 160, 90)")
        conn.execute("INSERT INTO session_tools (session_id, tool, event_count, saved_tokens, credited_event_count, "
                     "raw_chars_sum, out_chars_sum, saved_chars) VALUES ('unsized', 'hook:Bash', 3, 42, 0, 0, 0, 147)")
        conn.execute("INSERT INTO session_agents (session_id, is_subagent, agent_type, event_count, saved_tokens, "
                     "credited_event_count, raw_chars_sum, out_chars_sum, saved_chars) "
                     "VALUES ('sized', 0, '', 2, 26, 2, 200, 160, 90)")
        conn.execute("PRAGMA user_version = 6")
        conn.commit()
        conn.close()

    def test_v7_migration_rederives_the_saving_for_sized_rows_and_keeps_it_for_unsized(self):
        self._make_v6_db()
        conn = L._connect()
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        sess = dict(conn.execute("SELECT session_id, credited_saved_chars FROM sessions"))
        self.assertEqual(sess, {"sized": 40, "unsized": 147})  # 200 -> 160 saves 40; unsized keeps its backfill
        tools = {(r[0], r[1]): r[2] for r in conn.execute("SELECT session_id, tool, saved_chars FROM session_tools")}
        self.assertEqual(tools, {("sized", "hook:Bash"): 40, ("unsized", "hook:Bash"): 147})
        self.assertEqual(conn.execute("SELECT saved_chars FROM session_agents").fetchone(), (40,))
        conn.close()
        # The unsized row's 147 backfill is left out of the best session, like the other figures
        # (the "left out of the figures above" note would otherwise be untrue).
        self.assertEqual(L.query_project_summary("p")["best_session_chars"], 40)

    def test_v7_migration_reconstructs_v6_rows_whose_tokens_exceed_their_characters(self):
        # After #437 a not-yet-restarted MCP server still sent token-only footers and the
        # hook recorded them, so a v6 row can be token-only (characters 0) or mixed
        # (characters partial). Dropping the token columns must not lose that.
        self._make_v6_db()
        conn = sqlite3.connect(str(L.resolve_db_path()))
        insert = ("INSERT INTO sessions (session_id, project, ended_at, credited_saved_tokens, raw_tokens_sum, "
                  "out_tokens_sum, event_count, fetch_url_count, raw_chars_sum, out_chars_sum, credited_saved_chars) "
                  "VALUES (?, 'p', '2026-09-02T00:00:00Z', ?, ?, ?, ?, 0, ?, ?, ?)")
        conn.execute(insert, ("tokenonly", 900, 1001, 101, 2, 0, 0, 0))
        conn.execute(insert, ("mixed", 1800, 2000, 200, 2, 3500, 350, 3150))      # half its events had characters
        conn.execute(insert, ("covered", 900, 1000, 100, 1, 3501, 351, 3150))     # characters already cover the tokens
        conn.execute("INSERT INTO session_tools (session_id, tool, event_count, saved_tokens, raw_tokens_sum, "
                     "out_tokens_sum, credited_event_count, raw_chars_sum, out_chars_sum, saved_chars) "
                     "VALUES ('tokenonly', 'compress_file', 2, 900, 1001, 101, 2, 0, 0, 0)")
        conn.commit()
        conn.close()
        conn = L._connect()
        rows = {r[0]: r[1:] for r in conn.execute(
            "SELECT session_id, raw_chars_sum, out_chars_sum, credited_saved_chars FROM sessions")}
        # tokens x 3.5 (1001 -> 3504, 101 -> 354: ROUND half away from zero), saving re-derived
        self.assertEqual(rows["tokenonly"], (3504, 354, 3150))
        self.assertEqual(rows["mixed"], (7000, 700, 6300))      # the whole row, from tokens
        self.assertEqual(rows["covered"], (3501, 351, 3150))    # exact characters left untouched
        self.assertEqual(rows["sized"], (200, 160, 40))         # no token data: unchanged
        self.assertEqual(conn.execute(
            "SELECT raw_chars_sum, out_chars_sum, saved_chars FROM session_tools WHERE session_id = 'tokenonly'"
        ).fetchone(), (3504, 354, 3150))
        conn.close()
        ps = L.query_project_summary("p")
        self.assertEqual(ps["unsized_sessions"], 1)  # only the original pre-v6 unsized row; token-only is now sized

    def test_v7_reconstruction_slack_counts_credited_events_only_for_tool_and_agent_rows(self):
        # session_agents/session_tools event_count also counts uncredited calls (fetch_url) but their
        # token sums don't, so scaling the rounding slack with it would hide a small token-only row.
        self._make_v6_db()
        conn = sqlite3.connect(str(L.resolve_db_path()))
        conn.execute("INSERT INTO session_agents (session_id, is_subagent, agent_type, event_count, saved_tokens, "
                     "raw_tokens_sum, out_tokens_sum, credited_event_count, raw_chars_sum, out_chars_sum, saved_chars) "
                     "VALUES ('sized', 1, 'Explore', 100, 8, 10, 2, 1, 0, 0, 0)")  # 99 fetch_url calls + 1 credited
        conn.execute("INSERT INTO session_tools (session_id, tool, event_count, saved_tokens, raw_tokens_sum, "
                     "out_tokens_sum, credited_event_count, raw_chars_sum, out_chars_sum, saved_chars) "
                     "VALUES ('sized', 'compress_file', 100, 8, 10, 2, 1, 0, 0, 0)")
        conn.commit()
        conn.close()
        conn = L._connect()
        self.assertEqual(conn.execute(
            "SELECT raw_chars_sum, out_chars_sum, saved_chars FROM session_agents WHERE agent_type = 'Explore'"
        ).fetchone(), (35, 7, 28))   # 10 x 3.5, 2 x 3.5
        self.assertEqual(conn.execute(
            "SELECT raw_chars_sum, out_chars_sum, saved_chars FROM session_tools WHERE tool = 'compress_file'"
        ).fetchone(), (35, 7, 28))
        conn.close()

    def test_by_tool_rows_show_the_events_their_figures_cover(self):
        # 5 calls, but only 3 contributed a before/after: the row must say 3.
        breakdown = [{"tool": "hook:Bash", "event_count": 5, "credited_event_count": 3,
                      "raw_chars_sum": 3000, "out_chars_sum": 300, "saved_chars": 2700}]
        out = L.format_detail_view({"project": "p", "event_count": 3}, {"project": "p"}, breakdown, [], [])
        (line,) = [ln for ln in out.splitlines() if "hook:Bash" in ln]
        self.assertIn("  3 events", line)
        self.assertNotIn("  5 events", line)
        self.assertIn("3,000 → 300 chars", line)

    def test_json_exports_the_percent_reduction_at_every_level(self):
        session = {"project": "p", "raw_chars_sum": 1000, "out_chars_sum": 250}
        ps = {"project": "p", "total_raw_chars": 2000, "total_out_chars": 1000}
        tool = {"tool": "hook:Bash", "raw_chars_sum": 400, "out_chars_sum": 100}
        proj = {"project": "p", "total_raw_chars": 50, "total_out_chars": 50}
        none = {"project": "q"}
        parsed = json.loads(L.format_json(session, ps, [tool], [], [proj, none]))
        self.assertEqual(parsed["session"]["reduction_percent"], 75)
        self.assertEqual(parsed["project_summary"]["reduction_percent"], 50)
        self.assertEqual(parsed["tool_breakdown"][0]["reduction_percent"], 75)
        self.assertEqual(parsed["all_projects"][0]["reduction_percent"], 0)
        self.assertIsNone(parsed["all_projects"][1]["reduction_percent"])  # no before size: null, not 100

    def test_csv_exports_the_percent_reduction_at_every_level(self):
        session = {"project": "p", "raw_chars_sum": 1000, "out_chars_sum": 250}
        ps = {"project": "p", "total_raw_chars": 2000, "total_out_chars": 1000}
        tool = {"tool": "hook:Bash", "raw_chars_sum": 400, "out_chars_sum": 100}
        none = {"project": "q"}

        def row(table, **kw):
            rows = list(_csv_mod.reader(_io_mod.StringIO(L.format_csv(session, ps, [tool], [], [none], table=table))))
            return dict(zip(rows[0], rows[1], strict=False))

        sess = row("session")
        self.assertEqual((sess["reduction_percent"], sess["project_reduction_percent"]), ("75", "50"))
        self.assertEqual(row("tools")["reduction_percent"], "75")
        self.assertEqual(row("projects")["reduction_percent"], "")  # unknown: empty cell


    # -- round 3: a resumed pre-v6 session stays unsized --

    def test_resumed_unsized_session_stays_unsized_and_keeps_its_saving(self):
        self._insert_unsized_session("old", "p")  # event_count 3, saved 147, before/after unknown
        conn = L._connect()
        conn.execute("INSERT INTO session_tools (session_id, tool, event_count, credited_event_count, "
                     "raw_chars_sum, out_chars_sum, saved_chars) VALUES ('old', 'hook:Bash', 3, 0, 350, 0, 147)")
        conn.commit()
        conn.close()
        L.record_event("old", "hook:Bash", 3500, 350, True, project="p")
        agg = L.finalize_owned_session("old", "p")
        # The summary of THIS run still reports its own exact figures...
        self.assertEqual((agg["raw_chars_sum"], agg["out_chars_sum"], agg["event_count"]), (3500, 350, 1))
        conn = sqlite3.connect(str(L.resolve_db_path()))
        # ...but the stored row stays unsized: events add, characters and the saving don't.
        self.assertEqual(
            conn.execute("SELECT event_count, raw_chars_sum, out_chars_sum, credited_saved_chars FROM sessions "
                         "WHERE session_id = 'old'").fetchone(), (4, 0, 0, 147))
        self.assertEqual(
            conn.execute("SELECT event_count, credited_event_count, raw_chars_sum, out_chars_sum, saved_chars "
                         "FROM session_tools WHERE session_id = 'old'").fetchone(), (4, 0, 350, 0, 147))
        self.assertEqual(
            conn.execute("SELECT event_count, raw_chars_sum, saved_chars FROM session_agents "
                         "WHERE session_id = 'old'").fetchone(), (1, 0, 0))
        conn.close()
        ps = L.query_project_summary("p")
        self.assertEqual((ps["unsized_sessions"], ps["total_raw_chars"]), (1, 0))  # still reported as left out

    def test_resumed_sized_session_still_accumulates(self):
        L.record_event("s", "hook:Bash", 3500, 350, True, project="p")
        L.finalize_owned_session("s", "p")
        L.record_event("s", "hook:Bash", 3500, 350, True, project="p")
        L.finalize_owned_session("s", "p")
        ps = L.query_project_summary("p")
        self.assertEqual((ps["total_raw_chars"], ps["total_out_chars"], ps["unsized_sessions"]), (7000, 700, 0))


class FormatSimpleViewCharacters(unittest.TestCase):
    """Issue #438: the simple view (and so the SessionEnd summary) reports exact
    characters before and after compression and the percent reduction -- never an
    estimated token count."""

    _SESSION = {"project": "proj", "event_count": 3, "raw_chars_sum": 12000, "out_chars_sum": 3000,
                "credited_saved_chars": 9000, "fetch_url_count": 0}
    _PROJECT = {"project": "proj", "sessions": 4, "total_raw_chars": 50000, "total_out_chars": 20000,
                "total_saved_chars": 30000, "best_session_chars": 11000}

    def _lines(self, session=None, project=None):
        return L.format_simple_view(session or self._SESSION, project or self._PROJECT).splitlines()

    def test_session_block_shows_before_after_and_percent(self):
        text = "\n".join(self._lines())
        self.assertRegex(text, r"Characters before\s+12,000")
        self.assertRegex(text, r"Characters after\s+3,000")
        self.assertRegex(text, r"Reduction\s+75%\s+\(3 events\)")

    def test_project_block_shows_before_after_and_percent(self):
        text = "\n".join(self._lines())
        self.assertRegex(text, r"Characters before\s+50,000")
        self.assertRegex(text, r"Characters after\s+20,000")
        self.assertRegex(text, r"Reduction\s+60%")
        self.assertRegex(text, r"Best session \(saved\)\s+11,000")

    def test_no_estimated_token_wording_or_approximation_marker(self):
        text = "\n".join(self._lines())
        self.assertNotIn("tokens avoided", text)
        self.assertNotIn("Token Savings", text)
        self.assertNotIn("~", text)

    def test_no_raw_size_reports_na_not_a_fabricated_percentage(self):
        session = dict(self._SESSION, raw_chars_sum=0, out_chars_sum=0, credited_saved_chars=0, event_count=0)
        self.assertRegex("\n".join(self._lines(session=session)), r"Reduction\s+n/a")

    def test_an_expanding_summary_is_clamped_to_zero_percent(self):
        session = dict(self._SESSION, raw_chars_sum=100, out_chars_sum=150)
        self.assertRegex("\n".join(self._lines(session=session)), r"Reduction\s+0%\s+\(3 events\)")

    def test_no_tool_overhead_line_even_when_the_aggregate_carries_one(self):
        # Issue #439: the "Tool overhead .../turn" line is gone. A stray
        # overhead_tokens key (e.g. from an older caller) must not bring it back.
        session = dict(self._SESSION, overhead_tokens=2500)
        text = "\n".join(self._lines(session=session))
        self.assertNotIn("Tool overhead", text)
        self.assertNotIn("/turn", text)


class SchemaOverheadRemovedV8(SavingsLedgerTestCase):
    """Issue #439: `sessions.overhead_tokens` and the `schema_overhead_tokens` meta
    row held a chars / 3.5 estimate of a per-turn cost real transcripts show is not
    paid (the tools are deferred). v8 drops them; the rest of each row survives."""

    _V7_SESSIONS = (
        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, project TEXT NOT NULL, ended_at TEXT NOT NULL, "
        "event_count INTEGER NOT NULL, fetch_url_count INTEGER NOT NULL, "
        "overhead_tokens INTEGER NOT NULL DEFAULT 0, actual_input_tokens INTEGER NOT NULL DEFAULT 0, "
        "actual_output_tokens INTEGER NOT NULL DEFAULT 0, actual_cache_read_tokens INTEGER NOT NULL DEFAULT 0, "
        "actual_cache_write_tokens INTEGER NOT NULL DEFAULT 0, raw_chars_sum INTEGER NOT NULL DEFAULT 0, "
        "out_chars_sum INTEGER NOT NULL DEFAULT 0, credited_saved_chars INTEGER NOT NULL DEFAULT 0)"
    )

    def _make_v7_db(self):
        conn = sqlite3.connect(str(L.resolve_db_path()))
        conn.execute(self._V7_SESSIONS)
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO sessions VALUES ('known', 'p', '2026-09-01T00:00:00Z', 3, 1, 2500, 777, 8, 9, 10, 5000, 1000, 4000)"
        )
        conn.execute("INSERT INTO meta VALUES ('schema_overhead_tokens', '2500')")
        conn.execute("INSERT INTO meta VALUES ('keep-me', 'yes')")
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
        conn.close()

    def test_v7_database_loses_the_column_and_the_meta_row_but_keeps_every_other_value(self):
        self._make_v7_db()
        conn = L._connect()
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)")]
        self.assertNotIn("overhead_tokens", cols)
        self.assertEqual(tuple(cols), L._SESSIONS_COLUMNS)
        self.assertEqual(
            conn.execute("SELECT * FROM sessions").fetchone(),
            ("known", "p", "2026-09-01T00:00:00Z", 3, 1, 777, 8, 9, 10, 5000, 1000, 4000))
        self.assertIsNone(conn.execute("SELECT 1 FROM meta WHERE key = 'schema_overhead_tokens'").fetchone())
        self.assertIsNotNone(conn.execute("SELECT 1 FROM meta WHERE key = 'keep-me'").fetchone())
        conn.close()

    def test_a_fresh_database_never_has_the_column(self):
        conn = L._connect()
        self.assertNotIn("overhead_tokens", {r[1] for r in conn.execute("PRAGMA table_info(sessions)")})
        conn.close()

    def test_a_v8_database_reconnects_without_error(self):
        self._make_v7_db()
        L._connect().close()
        L._connect().close()  # second connect: nothing left to do, nothing may raise

    def test_the_removed_helpers_are_gone(self):
        self.assertFalse(hasattr(L, "get_schema_overhead_tokens"))


if __name__ == "__main__":
    unittest.main()
