#!/usr/bin/env python3
"""Tests for libs/metrics_lib.py (issue #208) -- the shared, generic
cross-tool metrics store: a MetricsStore class over ONE SQLite table any
domain can write append-only events into.

Stdlib-only (unittest, no pytest), no network, real on-disk SQLite (a temp
file per test, same env-var-redirection approach test_savings_ledger.py and
test_memory_events_lib.py use for their own sibling modules):

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import metrics_lib as M  # noqa: E402


class ResolveDbPath(unittest.TestCase):
    def test_defaults_under_claude_runway_home_dir(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_RUNWAY_METRICS_DB", None)
            expected = Path.home() / ".claude" / "claude-runway" / "metrics.db"
            self.assertEqual(M.resolve_db_path(), expected)

    def test_default_is_a_different_file_than_savings_and_memory_events_db(self):
        # Deliberately its own file (same reasoning memory_events_lib's own
        # analogous test documents): wiping one must never touch the others.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_RUNWAY_METRICS_DB", None)
            path = M.resolve_db_path()
            self.assertNotEqual(path, Path.home() / ".claude" / "claude-runway" / "savings.db")
            self.assertNotEqual(path, Path.home() / ".claude" / "claude-runway" / "memory-events.db")

    def test_absolute_override_is_used_as_is(self):
        # os.path.abspath resolves to a drive-letter path on Windows so
        # Path.is_absolute() returns True on both platforms, matching what
        # production code's resolve_db_path() actually checks.
        abs_path = os.path.abspath("/custom/metrics.db")
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": abs_path}, clear=False):
            self.assertEqual(M.resolve_db_path(), Path(abs_path))

    def test_relative_override_is_anchored_to_home_not_cwd(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": "relative/metrics.db"}, clear=False):
            self.assertEqual(M.resolve_db_path(), Path.home() / "relative" / "metrics.db")

    def test_expanduser_override_is_honored(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": "~/custom/metrics.db"}, clear=False):
            self.assertEqual(M.resolve_db_path(), Path.home() / "custom" / "metrics.db")


class MetricsStoreTestCase(unittest.TestCase):
    """Base case: a MetricsStore pinned to a temp file via explicit db_path
    (not an env-var patch) -- exercises the "pass db_path directly" path
    documented in MetricsStore.__init__, which is also what every test below
    relies on to avoid cross-test interference."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "metrics.db"
        self.store = M.MetricsStore(db_path=self._db_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _all_rows(self):
        if not self._db_path.exists():
            return []
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT metric_id, event_type, value, metadata, event_timestamp, session_id FROM metrics"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        conn.close()
        return rows


class RecordRoundTrip(MetricsStoreTestCase):
    def test_insert_then_read_back(self):
        self.store.record(
            "autowork", "ticket_merged", value=1.0,
            metadata={"issue": 208}, session_id="sess-1",
        )
        rows = self._all_rows()
        self.assertEqual(len(rows), 1)
        metric_id, event_type, value, metadata, event_timestamp, session_id = rows[0]
        self.assertEqual(metric_id, "autowork")
        self.assertEqual(event_type, "ticket_merged")
        self.assertEqual(value, 1.0)
        self.assertEqual(json.loads(metadata), {"issue": 208})
        self.assertTrue(event_timestamp)  # non-empty ISO timestamp string
        self.assertEqual(session_id, "sess-1")

    def test_defaults_value_to_one_and_metadata_to_none(self):
        self.store.record("autowork", "review_round")
        rows = self._all_rows()
        self.assertEqual(rows[0][2], 1.0)
        self.assertIsNone(rows[0][3])

    def test_multiple_records_all_persisted(self):
        for i in range(3):
            self.store.record("autowork", "ticket_merged", value=1.0)
        rows = self._all_rows()
        self.assertEqual(len(rows), 3)

    def test_non_serializable_metadata_fails_open_and_writes_nothing(self):
        # An object with no JSON representation (e.g. a raw set) must not
        # raise -- and must not silently write a garbage row either.
        self.store.record("autowork", "ticket_merged", metadata={"bad": {1, 2, 3}})
        self.assertEqual(self._all_rows(), [])


class EventTimestampOverride(MetricsStoreTestCase):
    """Issue #209: record() gained an optional event_timestamp override so
    tools/migrate_autowork_metrics.py can backfill historical Qdrant-derived
    events under their OWN original date, instead of every migrated row
    collapsing onto the migration's own run-time."""

    def test_explicit_event_timestamp_is_used_verbatim(self):
        self.store.record("autowork", "ticket_merged", event_timestamp="2026-08-15T00:00:00Z")
        rows = self._all_rows()
        self.assertEqual(rows[0][4], "2026-08-15T00:00:00Z")

    def test_omitting_it_keeps_stamping_current_time(self):
        # Existing behavior for every current call site must be unchanged --
        # this just confirms the new parameter defaulting to None doesn't
        # alter the pre-existing "stamp with now" path.
        with mock.patch.object(M.time, "gmtime", return_value=M.time.struct_time((2026, 1, 2, 3, 4, 5, 0, 1, 0))):
            self.store.record("autowork", "ticket_merged")
        rows = self._all_rows()
        self.assertEqual(rows[0][4], "2026-01-02T03:04:05Z")

    def test_invalid_format_is_rejected_and_writes_nothing(self):
        self.store.record("autowork", "ticket_merged", event_timestamp="2026-08-15")  # missing time component
        self.assertEqual(self._all_rows(), [])

    def test_non_iso_garbage_is_rejected_and_writes_nothing(self):
        self.store.record("autowork", "ticket_merged", event_timestamp="not-a-timestamp")
        self.assertEqual(self._all_rows(), [])

    def test_shape_valid_but_calendar_invalid_timestamp_is_rejected(self):
        """PR #246 review (Copilot): a regex checking only `\\d{4}-\\d{2}-\\d{2}
        T\\d{2}:\\d{2}:\\d{2}Z` shape would accept "2026-99-99T99:99:99Z"
        (no such month/day/hour/minute/second) and persist it, which later
        crashes trend()'s datetime.fromisoformat parsing. strptime performs
        real calendar/time validation instead of just character-class
        matching -- confirmed by reproduction before the fix."""
        self.store.record("autowork", "ticket_merged", event_timestamp="2026-99-99T99:99:99Z")
        self.assertEqual(self._all_rows(), [])

    def test_calendar_invalid_date_with_valid_time_is_also_rejected(self):
        # Narrower case of the above: only the date portion is impossible
        # (Feb 30 doesn't exist), time portion is fine on its own.
        self.store.record("autowork", "ticket_merged", event_timestamp="2026-02-30T12:00:00Z")
        self.assertEqual(self._all_rows(), [])

    def test_non_zero_padded_timestamp_is_rejected(self):
        """PR #246 review, round 2 (Copilot): datetime.strptime alone is
        lenient about zero-padding and happily parses "2026-8-5T01:02:03Z",
        which datetime.fromisoformat (what trend() actually calls) then
        rejects -- confirmed by reproduction. A strict round-trip
        (strftime the parsed value back and require an exact match) is
        what actually enforces the canonical zero-padded shape."""
        self.store.record("autowork", "ticket_merged", event_timestamp="2026-8-5T01:02:03Z")
        self.assertEqual(self._all_rows(), [])

    def test_canonical_zero_padded_timestamp_still_accepted(self):
        # The round-trip check must not reject genuinely canonical values.
        self.store.record("autowork", "ticket_merged", event_timestamp="2026-08-05T01:02:03Z")
        rows = self._all_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][4], "2026-08-05T01:02:03Z")

    def test_ordinary_values_around_a_rejected_one_are_unaffected(self):
        self.store.record("autowork", "ticket_merged", value=1.0)
        self.store.record("autowork", "ticket_merged", event_timestamp="garbage")
        self.store.record("autowork", "ticket_merged", value=2.0)
        rows = self._all_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual({r[2] for r in rows}, {1.0, 2.0})


class NonFiniteValueRejection(MetricsStoreTestCase):
    """Regression for PR #222 review: a non-finite value (inf/-inf/nan)
    must never be persisted -- one such row poisons SUM(value) for every
    future summary()/by_event_type()/trend() read of that metric_id, and
    format_summary_view's int(v) check raises OverflowError on it."""

    def test_positive_infinity_is_rejected_and_writes_nothing(self):
        self.store.record("autowork", "ticket_merged", value=float("inf"))
        self.assertEqual(self._all_rows(), [])

    def test_negative_infinity_is_rejected_and_writes_nothing(self):
        self.store.record("autowork", "ticket_merged", value=float("-inf"))
        self.assertEqual(self._all_rows(), [])

    def test_nan_is_rejected_and_writes_nothing(self):
        self.store.record("autowork", "ticket_merged", value=float("nan"))
        self.assertEqual(self._all_rows(), [])

    def test_increment_with_infinite_by_is_also_rejected(self):
        # increment()/decrement() are thin sugar over record() -- the
        # rejection must apply through that path too, not just the direct
        # record() call.
        self.store.increment("autowork", "ticket_merged", by=float("inf"))
        self.store.decrement("autowork", "ticket_merged", by=float("inf"))
        self.assertEqual(self._all_rows(), [])

    def test_ordinary_values_around_a_rejected_one_are_unaffected(self):
        # A rejected non-finite value must not corrupt or block subsequent
        # ordinary writes to the same metric_id.
        self.store.record("autowork", "ticket_merged", value=1.0)
        self.store.record("autowork", "ticket_merged", value=float("inf"))
        self.store.record("autowork", "ticket_merged", value=2.0)
        summary = self.store.summary("autowork")
        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["total_value"], 3.0)
        # Must not raise -- this is exactly what the reviewer's reproduction
        # showed breaking before the fix.
        M.format_summary_view(summary)


class IncrementDecrement(MetricsStoreTestCase):
    def test_increment_stores_positive_delta(self):
        self.store.increment("autowork", "open_tickets", by=3.0)
        rows = self._all_rows()
        self.assertEqual(rows[0][2], 3.0)

    def test_increment_defaults_by_to_one(self):
        self.store.increment("autowork", "open_tickets")
        rows = self._all_rows()
        self.assertEqual(rows[0][2], 1.0)

    def test_decrement_stores_negative_delta(self):
        self.store.decrement("autowork", "open_tickets", by=2.0)
        rows = self._all_rows()
        self.assertEqual(rows[0][2], -2.0)

    def test_current_value_is_sum_not_a_mutable_counter(self):
        # The whole point of this module's append-only design: three
        # increments and one decrement produce FOUR rows, and "current
        # value" is only ever derived by summing them at read time.
        self.store.increment("autowork", "open_tickets", by=1.0)
        self.store.increment("autowork", "open_tickets", by=1.0)
        self.store.increment("autowork", "open_tickets", by=1.0)
        self.store.decrement("autowork", "open_tickets", by=1.0)
        rows = self._all_rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual(self.store.summary("autowork")["total_value"], 2.0)


class RecordFailsOpen(unittest.TestCase):
    def test_unwritable_db_path_does_not_raise(self):
        # Same technique memory_events_lib's analogous test uses: point at a
        # path whose PARENT is actually a file, so mkdir(parents=True) raises.
        with tempfile.TemporaryDirectory() as tmpdir:
            blocking_file = Path(tmpdir) / "not-a-directory"
            blocking_file.write_text("blocking")
            bad_path = blocking_file / "sub" / "metrics.db"
            store = M.MetricsStore(db_path=bad_path)
            try:
                store.record("autowork", "ticket_merged")
            except Exception as e:  # pragma: no cover -- must not happen
                self.fail(f"record() raised instead of failing open: {e}")


class SummaryTestCase(MetricsStoreTestCase):
    def test_empty_metric_id_reports_zero_and_none_timestamps(self):
        summary = self.store.summary("no-such-metric")
        self.assertEqual(summary["metric_id"], "no-such-metric")
        self.assertEqual(summary["event_count"], 0)
        self.assertEqual(summary["total_value"], 0.0)
        self.assertIsNone(summary["first_event_at"])
        self.assertIsNone(summary["last_event_at"])

    def test_aggregates_across_event_types(self):
        self.store.record("autowork", "ticket_merged", value=1.0)
        self.store.record("autowork", "ticket_blocked", value=1.0)
        self.store.record("autowork", "ticket_merged", value=1.0)
        summary = self.store.summary("autowork")
        self.assertEqual(summary["event_count"], 3)
        self.assertEqual(summary["total_value"], 3.0)

    def test_different_metric_ids_do_not_bleed(self):
        self.store.record("autowork", "ticket_merged", value=5.0)
        self.store.record("memory-bank", "recall", value=1.0)
        self.assertEqual(self.store.summary("autowork")["total_value"], 5.0)
        self.assertEqual(self.store.summary("memory-bank")["total_value"], 1.0)


class ByEventType(MetricsStoreTestCase):
    def test_empty_metric_id_returns_empty_list(self):
        self.assertEqual(self.store.by_event_type("no-such-metric"), [])

    def test_breaks_down_by_event_type_ordered_by_total_value_desc(self):
        self.store.record("autowork", "ticket_merged", value=1.0)
        self.store.record("autowork", "ticket_merged", value=1.0)
        self.store.record("autowork", "ticket_merged", value=1.0)
        self.store.record("autowork", "review_round", value=1.0)
        rows = self.store.by_event_type("autowork")
        self.assertEqual(len(rows), 2)
        # Largest total_value first.
        self.assertEqual(rows[0]["event_type"], "ticket_merged")
        self.assertEqual(rows[0]["event_count"], 3)
        self.assertEqual(rows[0]["total_value"], 3.0)
        self.assertEqual(rows[1]["event_type"], "review_round")
        self.assertEqual(rows[1]["event_count"], 1)


class Trend(MetricsStoreTestCase):
    """Mirrors test_savings_ledger.py's QueryTrend test suite -- same
    bucketing semantics (ISO week via datetime.isocalendar(), day via the
    first 10 chars of the ISO timestamp)."""

    def _insert(self, metric_id, event_type, event_timestamp, value):
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, metric_id TEXT NOT NULL, "
            "event_type TEXT NOT NULL, value REAL NOT NULL, metadata TEXT, "
            "event_timestamp TEXT NOT NULL, session_id TEXT)"
        )
        conn.execute(
            "INSERT INTO metrics (metric_id, event_type, value, event_timestamp) VALUES (?, ?, ?, ?)",
            (metric_id, event_type, value, event_timestamp),
        )
        conn.commit()
        conn.close()

    def test_weekly_buckets_group_same_week_together(self):
        self._insert("autowork", "x", "2026-09-07T09:00:00Z", 1.0)
        self._insert("autowork", "x", "2026-09-09T17:00:00Z", 2.0)
        rows = self.store.trend("autowork", bucket="week", n=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_count"], 2)
        self.assertEqual(rows[0]["total_value"], 3.0)

    def test_weekly_buckets_iso_year_boundary_not_split(self):
        # Same regression this repo's savings_ledger trend test guards --
        # 2024-12-30 and 2025-01-01 are both ISO week 2025-W01.
        self._insert("autowork", "x", "2024-12-30T10:00:00Z", 1.0)
        self._insert("autowork", "x", "2025-01-01T14:00:00Z", 2.0)
        rows = self.store.trend("autowork", bucket="week", n=10)
        self.assertEqual(len(rows), 1, f"Expected 1 bucket but got {len(rows)}: {rows}")
        self.assertTrue(rows[0]["bucket"].startswith("2025-W"), rows[0]["bucket"])

    def test_daily_buckets_keep_different_days_separate(self):
        for day, value in (("2026-09-07", 1.0), ("2026-09-08", 2.0), ("2026-09-09", 3.0)):
            self._insert("autowork", "x", f"{day}T12:00:00Z", value)
        rows = self.store.trend("autowork", bucket="day", n=10)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["bucket"], "2026-09-07")
        self.assertEqual(rows[0]["total_value"], 1.0)
        self.assertEqual(rows[2]["bucket"], "2026-09-09")
        self.assertEqual(rows[2]["total_value"], 3.0)

    def test_n_caps_returned_buckets(self):
        for i in range(5):
            self._insert("autowork", "x", f"2026-09-0{i + 1}T12:00:00Z", float(i + 1))
        rows = self.store.trend("autowork", bucket="day", n=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["bucket"], "2026-09-03")
        self.assertEqual(rows[2]["bucket"], "2026-09-05")

    def test_empty_metric_id_returns_empty_list(self):
        self.assertEqual(self.store.trend("no-such-metric", bucket="week", n=10), [])

    def test_unknown_bucket_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.store.trend("autowork", bucket="month")
        self.assertIn("month", str(ctx.exception))

    def test_different_metric_ids_do_not_bleed(self):
        self._insert("autowork", "x", "2026-09-07T12:00:00Z", 5.0)
        self._insert("memory-bank", "x", "2026-09-07T12:00:00Z", 1.0)
        autowork_rows = self.store.trend("autowork", bucket="day", n=10)
        memory_rows = self.store.trend("memory-bank", bucket="day", n=10)
        self.assertEqual(len(autowork_rows), 1)
        self.assertEqual(autowork_rows[0]["total_value"], 5.0)
        self.assertEqual(len(memory_rows), 1)
        self.assertEqual(memory_rows[0]["total_value"], 1.0)


class _NoFetchAllCursor:
    """Wraps a real sqlite3.Cursor so a test can assert trend() never calls
    .fetchall() on it -- sqlite3.Cursor is a C extension type and can't be
    monkeypatched directly (setattr on it raises "cannot set 'fetchall'
    attribute of immutable type 'sqlite3.Cursor'", confirmed live), so this
    wraps the real cursor at the Python level instead and forwards
    everything except fetchall()."""

    def __init__(self, cursor):
        self._cursor = cursor

    def __iter__(self):
        return iter(self._cursor)

    def fetchall(self):
        raise AssertionError("trend() must not call fetchall()")

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _NoFetchAllConnection:
    """Wraps a real sqlite3.Connection so every cursor .execute() returns
    goes through _NoFetchAllCursor above."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, *args, **kwargs):
        return _NoFetchAllCursor(self._conn.execute(*args, **kwargs))

    def close(self):
        self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class TrendDoesNotMaterializeFullHistory(MetricsStoreTestCase):
    """Regression for PR #222 review: trend() must not call fetchall() up
    front -- that would load the metric_id's ENTIRE history into memory
    before the bucket-limiting loop even runs, scaling with total row count
    rather than with n. It must iterate the cursor directly instead, so
    that breaking out of the aggregation loop actually stops further rows
    from being pulled."""

    def test_trend_never_calls_fetchall(self):
        for i in range(20):
            self.store.record("autowork", "x", value=1.0)
        original_connect = M._connect
        with mock.patch("metrics_lib._connect", side_effect=lambda p: _NoFetchAllConnection(original_connect(p))):
            rows = self.store.trend("autowork", bucket="day", n=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_count"], 20)

    def test_idx_metrics_trend_index_exists(self):
        # Covers `WHERE metric_id = ? ORDER BY event_timestamp DESC` directly
        # (metric_id, event_timestamp DESC) -- so that access pattern doesn't
        # require a full table scan + separate sort step.
        self.store.record("autowork", "x", value=1.0)
        conn = sqlite3.connect(str(self._db_path))
        names = {row[1] for row in conn.execute("PRAGMA index_list(metrics)").fetchall()}
        conn.close()
        self.assertIn("idx_metrics_trend", names)


class LazyPathResolution(unittest.TestCase):
    """MetricsStore() with no explicit db_path re-resolves resolve_db_path()
    on every call rather than caching it at construction -- so a caller (or
    a test) that changes CLAUDE_RUNWAY_METRICS_DB after constructing the
    store still sees the new location, not a stale one."""

    def test_env_var_change_after_construction_is_honored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = M.MetricsStore()  # no explicit db_path
            first_path = Path(tmpdir) / "first.db"
            second_path = Path(tmpdir) / "second.db"
            with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": str(first_path)}):
                store.record("autowork", "x")
            with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": str(second_path)}):
                store.record("autowork", "y")
            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())


class FormattingHelpers(unittest.TestCase):
    def test_format_summary_view_with_no_events(self):
        text = M.format_summary_view({"metric_id": "autowork", "event_count": 0, "total_value": 0.0, "first_event_at": None, "last_event_at": None})
        self.assertIn("autowork", text)
        self.assertIn("No events recorded yet", text)

    def test_format_summary_view_with_events(self):
        text = M.format_summary_view({
            "metric_id": "autowork", "event_count": 3, "total_value": 3.0,
            "first_event_at": "2026-09-01T00:00:00Z", "last_event_at": "2026-09-07T00:00:00Z",
        })
        self.assertIn("3", text)
        self.assertNotIn("No events recorded yet", text)

    def test_format_by_event_type_view_empty(self):
        text = M.format_by_event_type_view("autowork", [])
        self.assertIn("No events recorded yet", text)

    def test_format_by_event_type_view_with_rows(self):
        text = M.format_by_event_type_view("autowork", [
            {"event_type": "ticket_merged", "event_count": 3, "total_value": 3.0},
        ])
        self.assertIn("ticket_merged", text)

    def test_format_trend_view_empty(self):
        text = M.format_trend_view("autowork", [])
        self.assertIn("No event history yet", text)

    def test_format_trend_view_with_rows(self):
        text = M.format_trend_view("autowork", [
            {"bucket": "2026-W37", "event_count": 2, "total_value": 3.0},
        ], bucket="week")
        self.assertIn("2026-W37", text)

    def test_format_json_roundtrips(self):
        raw = M.format_json("summary", {"metric_id": "autowork", "event_count": 1, "total_value": 1.0})
        doc = json.loads(raw)
        self.assertEqual(doc["view"], "summary")
        self.assertEqual(doc["data"]["metric_id"], "autowork")


if __name__ == "__main__":
    unittest.main()
