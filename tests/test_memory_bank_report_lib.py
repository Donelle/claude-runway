#!/usr/bin/env python3
"""Tests for libs/memory_bank_report_lib.py and tools/memory_bank_report.py
(issue #338) -- the read-only baseline report over memory-events.db.

Stdlib-only, no network, real on-disk SQLite in a temp dir. Rows are seeded
through memory_events_lib's own _connect() (so the schema under test is the
real one) with direct INSERTs where a test needs to control timestamps:

    .venv/bin/python -m unittest discover -s tests
"""

import calendar
import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "libs"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import memory_bank_report  # noqa: E402
import memory_bank_report_lib as rl  # noqa: E402
import memory_events_lib as ev  # noqa: E402


class _DbTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "memory-events.db"
        self._env = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_MEMORY_EVENTS_DB": str(self.db_path)})
        self._env.start()
        ev._connect().close()  # create the real schema

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def add(self, event_type, point_id, ts, repo="proj", project="proj", session="s1", turn=1):
        conn = sqlite3.connect(str(self.db_path))
        with conn:
            conn.execute(
                "INSERT INTO memory_events (event_type, point_id, repo, kind, summary_created_at, "
                "event_timestamp, turn, session_id, project) VALUES (?, ?, ?, 'lesson', NULL, ?, ?, ?, ?)",
                (event_type, point_id, repo, ts, turn, session, project),
            )
        conn.close()

    def ro(self):
        conn = rl.open_read_only(self.db_path)
        assert conn is not None
        self.addCleanup(conn.close)
        return conn


class DeadMemoryRatioTest(_DbTestCase):
    def test_overall_and_per_repo(self):
        self.add("remember", "a", "2026-09-01T00:00:00Z", repo="proj")
        self.add("remember", "b", "2026-09-01T00:00:00Z", repo="proj")
        self.add("remember", "c", "2026-09-01T00:00:00Z", repo="general")
        self.add("recall", "a", "2026-09-02T00:00:00Z", repo="proj")
        result = rl.dead_memory_ratio(self.ro())
        self.assertEqual(result["overall"], {"remembered": 3, "dead": 2, "ratio": 2 / 3})
        self.assertEqual(result["by_repo"]["proj"]["dead"], 1)
        self.assertEqual(result["by_repo"]["general"], {"remembered": 1, "dead": 1, "ratio": 1.0})

    def test_dashed_and_undashed_ids_of_the_same_memory_match(self):
        # Regression: remember logs uuid4().hex, Qdrant returns the dashed
        # form on recall -- a raw string compare matched nothing.
        hex_id = "6fed0f3aa24c45d89b6b49a249e5c2c7"
        dashed = "6FED0F3A-A24C-45D8-9B6B-49A249E5C2C7"
        self.add("remember", hex_id, "2026-09-01T00:00:00Z")
        self.add("recall", dashed, "2026-09-02T00:00:00Z")
        self.assertEqual(rl.dead_memory_ratio(self.ro())["overall"]["dead"], 0)

    def test_recall_before_remember_does_not_count(self):
        self.add("recall", "a", "2026-08-01T00:00:00Z")
        self.add("remember", "a", "2026-09-01T00:00:00Z")
        self.assertEqual(rl.dead_memory_ratio(self.ro())["overall"]["dead"], 1)

    def test_recall_only_points_are_not_counted_as_remembered(self):
        self.add("recall", "old", "2026-09-02T00:00:00Z")
        self.assertEqual(rl.dead_memory_ratio(self.ro())["overall"],
                         {"remembered": 0, "dead": 0, "ratio": None})

    def test_min_age_days_skips_young_memories(self):
        now = calendar.timegm(time.strptime("2026-10-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"))
        self.add("remember", "old", "2026-09-01T00:00:00Z")
        self.add("remember", "young", "2026-09-30T00:00:00Z")
        result = rl.dead_memory_ratio(self.ro(), min_age_days=7, now=now)
        self.assertEqual(result["overall"]["remembered"], 1)

    def test_non_finite_or_negative_min_age_is_rejected(self):
        # PR #424 review: nan silently disabled the filter; inf overflowed gmtime().
        for bad in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(ValueError):
                rl.dead_memory_ratio(self.ro(), min_age_days=bad)

    def test_huge_finite_min_age_counts_nothing_instead_of_overflowing(self):
        self.add("remember", "a", "2026-09-01T00:00:00Z")
        result = rl.dead_memory_ratio(self.ro(), min_age_days=1e300)
        self.assertEqual(result["overall"]["remembered"], 0)

    def test_excluded_projects_and_null_project_kept(self):
        self.add("remember", "t", "2026-09-01T00:00:00Z", project="test-collection")
        self.add("remember", "n", "2026-09-01T00:00:00Z", project=None)
        result = rl.dead_memory_ratio(self.ro())
        self.assertEqual(result["overall"]["remembered"], 1)
        result = rl.dead_memory_ratio(self.ro(), excluded_projects=())
        self.assertEqual(result["overall"]["remembered"], 2)

    def test_recall_from_excluded_project_does_not_revive_a_memory(self):
        self.add("remember", "a", "2026-09-01T00:00:00Z")
        self.add("recall", "a", "2026-09-02T00:00:00Z", project="test-collection")
        self.assertEqual(rl.dead_memory_ratio(self.ro())["overall"]["dead"], 1)

    def test_generator_exclusions_apply_to_recall_rows_too(self):
        # PR #424 review: a generator was consumed by the first clause, so
        # the excluded project's recall still revived the memory.
        self.add("remember", "a", "2026-09-01T00:00:00Z")
        self.add("recall", "a", "2026-09-02T00:00:00Z", project="x")
        result = rl.dead_memory_ratio(self.ro(), excluded_projects=(p for p in ["x"]))
        self.assertEqual(result["overall"]["dead"], 1)


class RecallPayloadTrendTest(_DbTestCase):
    def test_groups_hits_into_calls_by_session_and_turn(self):
        for pid in ("a", "b", "c"):
            self.add("recall", pid, "2026-09-01T00:00:00Z", session="s1", turn=1)
        self.add("recall", "a", "2026-09-01T00:00:01Z", session="s1", turn=2)
        self.add("recall", "a", "2026-09-01T00:00:01Z", session="s2", turn=1, project="other")
        result = rl.recall_payload_trend(self.ro())
        self.assertEqual(result["overall"], [
            {"bucket": "2026-W36", "calls": 3, "hits": 5, "avg_hits_per_call": 5 / 3},
        ])
        self.assertEqual(result["by_project"]["proj"][0]["calls"], 2)
        self.assertEqual(result["by_project"]["other"][0]["hits"], 1)

    def test_remember_rows_are_ignored(self):
        self.add("remember", "a", "2026-09-01T00:00:00Z")
        self.assertEqual(rl.recall_payload_trend(self.ro())["overall"], [])

    def test_week_bucket_across_year_boundary_uses_iso_week(self):
        self.add("recall", "a", "2026-12-31T00:00:00Z")
        self.assertEqual(rl.recall_payload_trend(self.ro())["overall"][0]["bucket"], "2026-W53")

    def test_month_buckets_are_ascending(self):
        self.add("recall", "a", "2026-10-02T00:00:00Z", turn=2)
        self.add("recall", "a", "2026-09-02T00:00:00Z", turn=1)
        buckets = [r["bucket"] for r in rl.recall_payload_trend(self.ro(), bucket="month")["overall"]]
        self.assertEqual(buckets, ["2026-09", "2026-10"])

    def test_unknown_bucket_raises(self):
        with self.assertRaises(ValueError):
            rl.recall_payload_trend(self.ro(), bucket="day")


class BuildReportTest(_DbTestCase):
    def test_missing_db_is_reported_and_not_created(self):
        missing = Path(self._tmp.name) / "nope.db"
        report = rl.build_report(missing)
        self.assertFalse(report["available"])
        self.assertFalse(missing.exists())
        self.assertFalse(report["score_trend"]["available"])

    def test_db_without_table_is_reported(self):
        empty = Path(self._tmp.name) / "empty.db"
        sqlite3.connect(str(empty)).close()
        self.assertFalse(rl.build_report(empty)["available"])

    def test_report_is_read_only(self):
        self.add("remember", "a", "2026-09-01T00:00:00Z")
        conn = self.ro()
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("DELETE FROM memory_events")

    def test_all_sections_read_inside_one_transaction(self):
        # PR #424 review: every section must see one snapshot, so each
        # query has to run inside the same open read transaction.
        self.add("remember", "a", "2026-09-01T00:00:00Z")
        seen = []

        def _spy(real):
            def wrapper(conn, *args, **kwargs):
                seen.append(conn.in_transaction)
                return real(conn, *args, **kwargs)
            return wrapper

        with mock.patch.object(rl, "_data_window", _spy(rl._data_window)), \
             mock.patch.object(rl, "dead_memory_ratio", _spy(rl.dead_memory_ratio)), \
             mock.patch.object(rl, "recall_payload_trend", _spy(rl.recall_payload_trend)):
            rl.build_report(self.db_path)
        self.assertEqual(seen, [True, True, True])

    def test_full_report_counts_and_text(self):
        self.add("remember", "a", "2026-09-01T00:00:00Z")
        self.add("recall", "a", "2026-09-02T00:00:00Z")
        self.add("recall", "x", "2026-09-02T00:00:00Z", project="test-collection", session="t")
        report = rl.build_report(self.db_path)
        self.assertTrue(report["available"])
        self.assertEqual(report["data_window"]["rows_included"], 2)
        self.assertEqual(report["data_window"]["rows_excluded"], 1)
        text = rl.format_report(report)
        self.assertIn("overall: 0/1 dead (0.0%)", text)
        self.assertIn("Score trend: Not available", text)
        self.assertIn("avg 1.00 hits/call", text)


class CliTest(_DbTestCase):
    def _run(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = memory_bank_report.main(list(argv))
        return rc, out.getvalue()

    def test_defaults_read_the_resolved_db_path(self):
        self.add("remember", "a", "2026-09-01T00:00:00Z")
        rc, out = self._run()
        self.assertEqual(rc, 0)
        self.assertIn(str(self.db_path), out)
        self.assertIn("overall: 1/1 dead", out)

    def test_json_and_exclusion_flags(self):
        self.add("remember", "t", "2026-09-01T00:00:00Z", project="test-collection")
        self.add("remember", "o", "2026-09-01T00:00:00Z", project="other")
        _, out = self._run("--json", "--no-default-excludes", "--exclude-project", "other")
        report = json.loads(out)
        self.assertEqual(report["excluded_projects"], ["other"])
        self.assertEqual(report["dead_memory"]["overall"]["remembered"], 1)

    def test_negative_or_non_finite_min_age_is_rejected(self):
        for bad in ("-1", "inf", "nan"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                memory_bank_report.parse_args(["--min-age-days", bad])


if __name__ == "__main__":
    unittest.main()
