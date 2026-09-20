#!/usr/bin/env python3
"""Tests for libs/memory_events_lib.py (issue #179) -- the passive,
append-only recall/remember usage log, deliberately kept as its own SQLite
file separate from libs/savings_ledger.py's savings.db.

Stdlib-only (unittest, no pytest), no network, real on-disk SQLite (a temp
file per test, same env-var-redirection approach test_savings_ledger.py and
test_cache_db.py use for their own sibling modules):

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import memory_events_lib as ev  # noqa: E402


class ResolveDbPath(unittest.TestCase):
    def test_defaults_under_claude_runway_home_dir(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_RUNWAY_MEMORY_EVENTS_DB", None)
            expected = Path.home() / ".claude" / "claude-runway" / "memory-events.db"
            self.assertEqual(ev.resolve_db_path(), expected)

    def test_default_is_a_different_file_than_savings_db_and_cache_db(self):
        # The whole point of this module (issue #179's own explicit
        # rationale): never the same file as savings_ledger's or cache_db's,
        # so wiping one never touches the others.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_RUNWAY_MEMORY_EVENTS_DB", None)
            path = ev.resolve_db_path()
            self.assertNotEqual(path, Path.home() / ".claude" / "claude-runway" / "savings.db")
            self.assertNotEqual(path, Path.home() / ".claude" / "claude-runway" / "cache.db")

    def test_absolute_override_is_used_as_is(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_MEMORY_EVENTS_DB": "/custom/memory-events.db"}, clear=False):
            self.assertEqual(ev.resolve_db_path(), Path("/custom/memory-events.db"))

    def test_relative_override_is_anchored_to_home_not_cwd(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_MEMORY_EVENTS_DB": "relative/mem.db"}, clear=False):
            self.assertEqual(ev.resolve_db_path(), Path.home() / "relative" / "mem.db")

    def test_expanduser_override_is_honored(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_MEMORY_EVENTS_DB": "~/custom/mem.db"}, clear=False):
            self.assertEqual(ev.resolve_db_path(), Path.home() / "custom" / "mem.db")


class TrackingEnabled(unittest.TestCase):
    def test_defaults_to_true_unlike_savings_tracker(self):
        # Deliberately the opposite default of savings_ledger.tracking_enabled()
        # -- see memory_events_lib.py's own docstring for why.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS", None)
            self.assertTrue(ev.tracking_enabled())

    def test_explicit_falsy_values_disable_it(self):
        for value in ("0", "false", "False", "no", "NO"):
            with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS": value}, clear=False):
                self.assertFalse(ev.tracking_enabled(), f"expected {value!r} to disable tracking")

    def test_other_values_leave_it_enabled(self):
        for value in ("1", "true", "yes", "anything-else"):
            with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS": value}, clear=False):
                self.assertTrue(ev.tracking_enabled(), f"expected {value!r} to leave tracking enabled")


class RecordMemoryEventTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "memory-events.db"
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_MEMORY_EVENTS_DB": str(self._db_path)})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def _all_rows(self):
        # A rejected event_type (see RecordMemoryEventValidation below)
        # raises before _connect() ever runs, so the table -- and even the
        # db file itself -- may not exist yet. That's equivalent to "no rows
        # written," not a real error worth surfacing from this helper.
        if not self._db_path.exists():
            return []
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT event_type, point_id, repo, kind, summary_created_at, "
                "event_timestamp, turn, session_id, project FROM memory_events"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        conn.close()
        return rows


class RecordMemoryEventRoundTrip(RecordMemoryEventTestCase):
    def test_insert_then_read_back(self):
        ev.record_memory_event(
            event_type="remember",
            point_id="abc123",
            repo="proj-a",
            kind="lesson",
            summary_created_at=1700000000.0,
            session_id="sess-1",
            project="proj-a",
            turn=1,
        )
        rows = self._all_rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row[0], "remember")
        self.assertEqual(row[1], "abc123")
        self.assertEqual(row[2], "proj-a")
        self.assertEqual(row[3], "lesson")
        self.assertEqual(row[4], 1700000000.0)
        self.assertTrue(row[5])  # event_timestamp is a non-empty string
        self.assertEqual(row[6], 1)
        self.assertEqual(row[7], "sess-1")
        self.assertEqual(row[8], "proj-a")

    def test_multiple_events_share_a_turn(self):
        # Mirrors how tools/memory_bank_mcp_server.py's recall() logs one row
        # per hit but a single shared turn value for the whole call.
        for point_id in ("p1", "p2", "p3"):
            ev.record_memory_event(
                event_type="recall", point_id=point_id, repo="proj-a", kind="idea",
                summary_created_at=None, session_id="sess-1", project="proj-a", turn=5,
            )
        rows = self._all_rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual({r[6] for r in rows}, {5})

    def test_point_id_is_coerced_to_string(self):
        # Qdrant point ids can be int or str/UUID -- storage must not choke
        # on a non-string id.
        ev.record_memory_event(
            event_type="recall", point_id=42, repo=None, kind=None,
            summary_created_at=None, session_id="sess-1", project=None, turn=1,
        )
        rows = self._all_rows()
        self.assertEqual(rows[0][1], "42")

    def test_none_point_id_stays_none(self):
        ev.record_memory_event(
            event_type="remember", point_id=None, repo="proj-a", kind="lesson",
            summary_created_at=None, session_id="sess-1", project="proj-a", turn=1,
        )
        rows = self._all_rows()
        self.assertIsNone(rows[0][1])


class RecordMemoryEventValidation(RecordMemoryEventTestCase):
    def test_unknown_event_type_raises(self):
        with self.assertRaises(ValueError):
            ev.record_memory_event(
                event_type="forget", point_id="p1", repo="proj-a", kind="lesson",
                summary_created_at=None, session_id="sess-1", project="proj-a", turn=1,
            )
        # A rejected event_type must never reach storage.
        self.assertEqual(self._all_rows(), [])


class RecordMemoryEventFailsOpen(unittest.TestCase):
    def test_unwritable_db_path_does_not_raise(self):
        # Pointing at a path whose PARENT is actually a file (not a
        # directory) makes mkdir(parents=True) raise -- record_memory_event
        # must swallow that rather than propagate it, since a broken usage
        # log must never break the remember/recall call it's logging.
        with tempfile.TemporaryDirectory() as tmpdir:
            blocking_file = Path(tmpdir) / "not-a-directory"
            blocking_file.write_text("blocking")
            bad_path = blocking_file / "sub" / "memory-events.db"
            with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_MEMORY_EVENTS_DB": str(bad_path)}):
                try:
                    ev.record_memory_event(
                        event_type="remember", point_id="p1", repo="proj-a", kind="lesson",
                        summary_created_at=None, session_id="sess-1", project="proj-a", turn=1,
                    )
                except Exception as e:  # pragma: no cover -- this is exactly what must NOT happen
                    self.fail(f"record_memory_event raised instead of failing open: {e}")


class ConnectionCleanupOnFailure(unittest.TestCase):
    """Regression for PR #197 review: a failure AFTER sqlite3.connect()
    succeeds (schema init, or the INSERT itself) must not leak the open
    connection -- neither `_connect()` nor `record_memory_event` can rely on
    a bare post-block `conn.close()` line, since an exception skips past it."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "memory-events.db"
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_MEMORY_EVENTS_DB": str(db_path)})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_connect_closes_connection_when_schema_init_fails(self):
        fake_conn = MagicMock()
        fake_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        with mock.patch("memory_events_lib.sqlite3.connect", return_value=fake_conn):
            with self.assertRaises(sqlite3.OperationalError):
                ev._connect()
        fake_conn.close.assert_called_once()

    def test_record_memory_event_closes_connection_when_insert_fails(self):
        fake_conn = MagicMock()
        fake_conn.__enter__ = MagicMock(return_value=fake_conn)
        fake_conn.__exit__ = MagicMock(return_value=False)
        # The 4 schema-setup execute() calls _connect() makes (CREATE TABLE +
        # 3 CREATE INDEX) succeed; the 5th call -- the INSERT, made inside
        # record_memory_event's own `with conn:` block -- is the one that
        # fails.
        calls = {"n": 0}

        def _execute(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 4:
                raise sqlite3.OperationalError("database is locked")
            return MagicMock()

        fake_conn.execute.side_effect = _execute
        with mock.patch("memory_events_lib.sqlite3.connect", return_value=fake_conn):
            try:
                ev.record_memory_event(
                    event_type="remember", point_id="p1", repo="proj-a", kind="lesson",
                    summary_created_at=None, session_id="sess-1", project="proj-a", turn=1,
                )
            except Exception as e:  # pragma: no cover -- must fail open, never raise
                self.fail(f"record_memory_event raised instead of failing open: {e}")
        fake_conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
