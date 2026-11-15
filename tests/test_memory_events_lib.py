#!/usr/bin/env python3
"""Tests for libs/memory_events_lib.py -- the passive,
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
import uuid
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
        # The whole point of this module (its explicit
        # rationale): never the same file as savings_ledger's or cache_db's,
        # so wiping one never touches the others.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_RUNWAY_MEMORY_EVENTS_DB", None)
            path = ev.resolve_db_path()
            self.assertNotEqual(path, Path.home() / ".claude" / "claude-runway" / "savings.db")
            self.assertNotEqual(path, Path.home() / ".claude" / "claude-runway" / "cache.db")

    def test_absolute_override_is_used_as_is(self):
        # "/custom/..." has no drive letter, so it isn't absolute on Windows
        # (see the same test in test_cache_db.py); abspath() makes it
        # genuinely absolute there and is a no-op on POSIX.
        custom = os.path.abspath("/custom/memory-events.db")
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_MEMORY_EVENTS_DB": custom}, clear=False):
            self.assertEqual(ev.resolve_db_path(), Path(custom))

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
        # Mirrors how mcpservers/memory_bank_mcp_server.py's recall() logs one row
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


# One memory's id in the two spellings this log used to carry:
# remember_point's uuid4().hex, and the dashed form Qdrant returns on recall.
HEX_ID = "6fed0f3aa24c4b8e9d1c2b3a4f5e6d7c"
DASHED_ID = "6fed0f3a-a24c-4b8e-9d1c-2b3a4f5e6d7c"


class CanonicalPointId(unittest.TestCase):
    def test_hex_becomes_lowercase_dashed(self):
        self.assertEqual(ev.canonical_point_id(HEX_ID), DASHED_ID)

    def test_uppercase_hex_and_uppercase_dashed_become_lowercase_dashed(self):
        self.assertEqual(ev.canonical_point_id(HEX_ID.upper()), DASHED_ID)
        self.assertEqual(ev.canonical_point_id(DASHED_ID.upper()), DASHED_ID)

    def test_dashed_is_unchanged(self):
        self.assertEqual(ev.canonical_point_id(DASHED_ID), DASHED_ID)

    def test_uuid5_id_from_transfer_and_copy_is_unchanged(self):
        # transfer_memories/copy_memories mint str(uuid5(...)): already dashed.
        minted = str(uuid.uuid5(uuid.NAMESPACE_DNS, "example.org"))
        self.assertEqual(ev.canonical_point_id(minted), minted)

    def test_none_stays_none(self):
        self.assertIsNone(ev.canonical_point_id(None))

    def test_values_that_are_not_a_uuid_come_back_as_plain_strings(self):
        not_uuids = [
            (42, "42"),  # an integer Qdrant point id
            ("", ""),
            ("p1", "p1"),
            ("abc123", "abc123"),
            ("g" * 32, "g" * 32),  # 32 characters but not hex
            ("a" * 31, "a" * 31),
            ("a" * 33, "a" * 33),
            (DASHED_ID[:-1], DASHED_ID[:-1]),  # one character short
            (DASHED_ID.replace("-", "_"), DASHED_ID.replace("-", "_")),
            ("{" + DASHED_ID + "}", "{" + DASHED_ID + "}"),  # uuid.UUID would accept these
            ("urn:uuid:" + DASHED_ID, "urn:uuid:" + DASHED_ID),
            (HEX_ID + "\n", HEX_ID + "\n"),  # fullmatch, not a trailing-newline-tolerant $
        ]
        for given, expected in not_uuids:
            with self.subTest(given=given):
                self.assertEqual(ev.canonical_point_id(given), expected)


class RecordMemoryEventCanonicalizesPointId(RecordMemoryEventTestCase):
    def _record(self, event_type, point_id):
        ev.record_memory_event(
            event_type=event_type, point_id=point_id, repo="proj-a", kind="lesson",
            summary_created_at=None, session_id="sess-1", project="proj-a", turn=1,
        )

    def test_remember_and_recall_of_one_memory_store_identical_ids_and_join(self):
        # The defect: remember logged the undashed hex, recall logged the
        # dashed form, and a plain join matched zero rows.
        self._record("remember", HEX_ID)
        self._record("recall", DASHED_ID)
        rows = self._all_rows()
        self.assertEqual({r[1] for r in rows}, {DASHED_ID})
        conn = sqlite3.connect(str(self._db_path))
        try:
            joined = conn.execute(
                "SELECT COUNT(*) FROM memory_events m JOIN memory_events r "
                "ON r.point_id = m.point_id "
                "WHERE m.event_type = 'remember' AND r.event_type = 'recall'"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(joined, 1)

    def test_explicit_dashed_id_is_stored_unchanged(self):
        # The transfer_memories / copy_memories shape.
        minted = str(uuid.uuid5(uuid.NAMESPACE_DNS, "example.org"))
        self._record("remember", minted)
        self.assertEqual(self._all_rows()[0][1], minted)

    def test_a_non_uuid_id_is_stored_as_given(self):
        self._record("remember", "p1")
        self.assertEqual(self._all_rows()[0][1], "p1")


class PointIdBackfill(RecordMemoryEventTestCase):
    """The one-time rewrite in `_connect()` of rows logged before the canonical form."""

    def _connect_once(self):
        ev._connect().close()

    def _raw(self, sql, params=()):
        conn = sqlite3.connect(str(self._db_path))
        try:
            with conn:
                return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def _insert(self, point_id, event_type="remember"):
        self._raw(
            "INSERT INTO memory_events (event_type, point_id, event_timestamp, turn, session_id) "
            "VALUES (?, ?, '2026-01-01T00:00:00Z', 1, 's')",
            (event_type, point_id),
        )

    def _ids(self):
        return [r[0] for r in self._raw("SELECT point_id FROM memory_events ORDER BY id")]

    def _version(self):
        return self._raw("PRAGMA user_version")[0][0]

    def _set_version(self, version):
        self._raw(f"PRAGMA user_version = {int(version)}")

    def test_a_new_db_is_marked_migrated_without_touching_anything(self):
        self._connect_once()
        self.assertEqual(self._version(), 1)
        self.assertEqual(self._ids(), [])

    def test_old_hex_rows_are_rewritten_and_everything_else_is_left_alone(self):
        self._connect_once()
        self._set_version(0)  # what a file written before the canonical form looks like
        seeded = [
            HEX_ID,
            HEX_ID.upper(),
            DASHED_ID,
            "ab",  # short non-UUID ids other tests/fixtures write
            "abc123",
            "g" * 32,  # 32 characters, not hex
            "a" * 31,
            None,
        ]
        for point_id in seeded:
            self._insert(point_id)
        self._connect_once()
        self.assertEqual(
            self._ids(),
            [DASHED_ID, DASHED_ID, DASHED_ID, "ab", "abc123", "g" * 32, "a" * 31, None],
        )
        self.assertEqual(self._version(), 1)

    def test_the_rewrite_runs_once_not_on_every_connection(self):
        self._connect_once()
        self._set_version(0)
        self._insert(HEX_ID)
        self._connect_once()
        self.assertEqual(self._ids(), [DASHED_ID])
        # A raw undashed row added AFTER the migration is not rewritten: the
        # version gate stops it, which is what keeps the per-event cost to one
        # integer read instead of a table scan.
        self._insert(HEX_ID)
        self._connect_once()
        self.assertEqual(self._ids(), [DASHED_ID, HEX_ID])
        self.assertEqual(self._version(), 1)

    def test_a_second_connection_changes_nothing(self):
        self._connect_once()
        self._set_version(0)
        for point_id in (HEX_ID, DASHED_ID, "p1"):
            self._insert(point_id)
        self._connect_once()
        first = self._ids()
        self._connect_once()
        self.assertEqual(self._ids(), first)

    def test_a_newer_user_version_is_left_alone(self):
        self._connect_once()
        self._set_version(7)
        self._insert(HEX_ID)
        self._connect_once()
        self.assertEqual(self._ids(), [HEX_ID])
        self.assertEqual(self._version(), 7)

    def test_recording_into_an_old_db_migrates_it_so_remember_and_recall_join(self):
        # End to end: an old file holding an undashed remember row, then a
        # recall of that memory logged by the fixed writer.
        self._connect_once()
        self._set_version(0)
        self._insert(HEX_ID, event_type="remember")
        ev.record_memory_event(
            event_type="recall", point_id=DASHED_ID, repo="proj-a", kind="lesson",
            summary_created_at=None, session_id="sess-1", project="proj-a", turn=2,
        )
        joined = self._raw(
            "SELECT COUNT(*) FROM memory_events m JOIN memory_events r ON r.point_id = m.point_id "
            "WHERE m.event_type = 'remember' AND r.event_type = 'recall'"
        )[0][0]
        self.assertEqual(joined, 1)


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
    """Regression for a review finding: a failure AFTER sqlite3.connect()
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


class RecordMemoryMetricTest(unittest.TestCase):
    """record_memory_metric() writes a plain per-call-attempt
    tally into metrics.db via libs/metrics_lib.py's MetricsStore --
    deliberately separate from record_memory_event/memory-events.db above
    (that table stays entirely untouched by this function)."""

    def test_valid_event_types_call_metrics_store_increment(self):
        for event_type in ("recall", "remember", "forget"):
            with mock.patch("memory_events_lib.metrics_lib.MetricsStore") as mock_store_cls:
                mock_store = mock_store_cls.return_value
                ev.record_memory_metric(event_type, session_id="sess-1")
            mock_store_cls.assert_called_once_with()
            mock_store.increment.assert_called_once_with(
                metric_id="memory-bank", event_type=event_type, session_id="sess-1"
            )

    def test_session_id_defaults_to_none(self):
        with mock.patch("memory_events_lib.metrics_lib.MetricsStore") as mock_store_cls:
            mock_store = mock_store_cls.return_value
            ev.record_memory_metric("recall")
        mock_store.increment.assert_called_once_with(
            metric_id="memory-bank", event_type="recall", session_id=None
        )

    def test_forget_is_valid_here_even_though_record_memory_event_rejects_it(self):
        # The whole point of _VALID_METRIC_EVENT_TYPES being wider than
        # _VALID_EVENT_TYPES: a plain call count for "forget" carries none of
        # the "autonomous removal" concern that excludes it from the richer
        # per-point memory_events log.
        with mock.patch("memory_events_lib.metrics_lib.MetricsStore") as mock_store_cls:
            ev.record_memory_metric("forget")
        mock_store_cls.return_value.increment.assert_called_once()

    def test_unknown_event_type_raises_without_touching_metrics_store(self):
        with mock.patch("memory_events_lib.metrics_lib.MetricsStore") as mock_store_cls:
            with self.assertRaises(ValueError):
                ev.record_memory_metric("something-else")
        mock_store_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
