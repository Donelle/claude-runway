#!/usr/bin/env python3
"""Tests for tools/migrate_autowork_metrics.py (issue #209) -- the one-time
migration of my-gh-autowork's Qdrant-stored per-ticket metrics into the
shared metrics.db (libs/metrics_lib.py, issue #208).

Stdlib-only (unittest, no pytest), no network and no real Qdrant/LM Studio --
QdrantClient and _collections_for_project are mocked out, and MetricsStore
is pinned to a real temp-file SQLite db (same "real on-disk SQLite in a temp
file" approach test_metrics_lib.py itself uses), so these tests exercise the
real read-back/idempotency logic without needing any live service:

    .venv/bin/python -m unittest discover -s tests
"""

import argparse
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import migrate_autowork_metrics as mod  # noqa: E402
import metrics_lib as metrics_lib_mod  # noqa: E402


def _point(point_id, information=None, document=None, date="2026-09-20", payload_overrides=None):
    """Builds a fake Qdrant Record/ScoredPoint-shaped object -- only `.id`
    and `.payload` are ever accessed by the code under test, mirroring the
    real qdrant-client objects' shape closely enough for these tests."""
    payload = {"project": "claude-runway-autowork-metrics", "label": f"issue-x-{point_id}", "date": date}
    if information is not None:
        payload["information"] = json.dumps(information)
    if document is not None:
        payload["document"] = json.dumps(document)
    if payload_overrides:
        payload.update(payload_overrides)
    return SimpleNamespace(id=point_id, payload=payload)


def _args(**overrides):
    defaults = dict(
        project="claude-runway-autowork-metrics",
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        metrics_db=None,
        dry_run=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class MigrateTestCase(unittest.TestCase):
    """Base case: MetricsStore pinned to a temp file (same convention
    test_metrics_lib.py uses), and QdrantClient/_collections_for_project
    mocked so no real network call is ever made."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "metrics.db"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run_migrate(self, points, collections=("conversation-compacts-claude-runway-autowork-metrics-fc6e9795",), **arg_overrides):
        """Runs migrate() with a fake single-page scroll returning `points`
        for the first (and only) resolved collection, and empty results for
        any other collection name in `collections`. Returns (exit_code, stdout)."""
        fake_client = mock.MagicMock()

        def fake_scroll(collection_name, limit, offset, with_payload):
            if offset is not None:
                return [], None
            if collection_name == collections[0]:
                return list(points), None
            return [], None

        fake_client.scroll.side_effect = fake_scroll
        args = _args(metrics_db=str(self._db_path), **arg_overrides)
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "_collections_for_project", return_value=list(collections)):
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exit_code = mod.migrate(args)
        # Combined -- this helper's callers care about "did this message get
        # printed somewhere," not which stream specifically (migrate() splits
        # ordinary progress to stdout and warnings/failures to stderr).
        return exit_code, stdout_buf.getvalue() + stderr_buf.getvalue()

    def _all_metrics_rows(self):
        if not self._db_path.exists():
            return []
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT metric_id, event_type, value, metadata, event_timestamp FROM metrics"
            ).fetchall()
        finally:
            conn.close()
        return rows


class OutcomeMapping(MigrateTestCase):
    def test_merged_maps_to_ticket_merged(self):
        points = [_point("p1", information={"issue": 1, "pr": 10, "outcome": "MERGED", "model": "m", "rounds": 0, "manual_interventions": 0, "wall_clock_s": 1.0, "subagent_tokens": 1, "tool_calls": 1, "findings": []})]
        exit_code, _ = self._run_migrate(points)
        self.assertEqual(exit_code, 0)
        rows = self._all_metrics_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "autowork")
        self.assertEqual(rows[0][1], "ticket_merged")
        self.assertEqual(rows[0][2], 1.0)

    def test_blocked_maps_to_ticket_blocked(self):
        points = [_point("p1", information={"issue": 2, "pr": None, "outcome": "BLOCKED", "model": "m", "rounds": 0, "manual_interventions": 0, "wall_clock_s": 1.0, "subagent_tokens": 1, "tool_calls": 1, "findings": []})]
        self._run_migrate(points)
        rows = self._all_metrics_rows()
        self.assertEqual(rows[0][1], "ticket_blocked")

    def test_failed_maps_to_ticket_failed(self):
        points = [_point("p1", information={"issue": 3, "pr": None, "outcome": "FAILED", "model": "m", "rounds": 0, "manual_interventions": 0, "wall_clock_s": 1.0, "subagent_tokens": 1, "tool_calls": 1, "findings": []})]
        self._run_migrate(points)
        rows = self._all_metrics_rows()
        self.assertEqual(rows[0][1], "ticket_failed")

    def test_unrecognized_outcome_maps_to_ticket_unknown(self):
        points = [_point("p1", information={"issue": 4, "outcome": "WEIRD"})]
        self._run_migrate(points)
        rows = self._all_metrics_rows()
        self.assertEqual(rows[0][1], "ticket_unknown")

    def test_missing_outcome_maps_to_ticket_unknown(self):
        points = [_point("p1", information={"issue": 5})]
        self._run_migrate(points)
        rows = self._all_metrics_rows()
        self.assertEqual(rows[0][1], "ticket_unknown")

    def test_outcome_matching_is_case_insensitive(self):
        points = [_point("p1", information={"issue": 6, "outcome": "merged"})]
        self._run_migrate(points)
        rows = self._all_metrics_rows()
        self.assertEqual(rows[0][1], "ticket_merged")


class EventTimestampPreservation(MigrateTestCase):
    def test_uses_the_points_own_date_not_run_time(self):
        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"}, date="2026-08-15")]
        self._run_migrate(points)
        rows = self._all_metrics_rows()
        self.assertEqual(rows[0][4], "2026-08-15T00:00:00Z")

    def test_malformed_date_falls_back_without_crashing(self):
        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"}, date="not-a-date")]
        exit_code, stderr_or_stdout = self._run_migrate(points)
        self.assertEqual(exit_code, 0)
        rows = self._all_metrics_rows()
        self.assertEqual(len(rows), 1)
        # Falls back to *some* valid ISO 8601 UTC timestamp rather than
        # storing the malformed value verbatim (which would later break
        # trend()'s datetime.fromisoformat parsing).
        import re
        self.assertRegex(rows[0][4], r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

    def test_missing_date_falls_back_without_crashing(self):
        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"})]
        points[0].payload.pop("date")
        exit_code, _ = self._run_migrate(points)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(self._all_metrics_rows()), 1)

    def test_shape_valid_but_calendar_invalid_date_falls_back_without_crashing(self):
        """PR #246 review (Copilot): a regex checking only `\\d{4}-\\d{2}-\\d{2}`
        shape would accept "2026-99-99" (calendar-invalid: no such month/day)
        and persist it verbatim as event_timestamp, which later crashes
        MetricsStore.trend()'s datetime.fromisoformat parsing. Confirmed by
        reproduction before the fix (strptime replacing the regex)."""
        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"}, date="2026-99-99")]
        exit_code, _ = self._run_migrate(points)
        self.assertEqual(exit_code, 0)
        rows = self._all_metrics_rows()
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0][4], "2026-99-99T00:00:00Z")
        self.assertRegex(rows[0][4], r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
        # And the fallback value must itself be a REAL, parseable timestamp --
        # this is the exact check trend() performs downstream.
        from datetime import datetime
        datetime.fromisoformat(rows[0][4].replace("Z", "+00:00"))

    def test_non_zero_padded_date_falls_back_without_crashing(self):
        """PR #246 review, round 2 (Copilot): datetime.strptime alone is
        lenient about zero-padding and happily parses "2026-8-5" (non-
        canonical), which datetime.fromisoformat then rejects downstream --
        confirmed by reproduction. Must fall back, not persist verbatim."""
        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"}, date="2026-8-5")]
        exit_code, _ = self._run_migrate(points)
        self.assertEqual(exit_code, 0)
        rows = self._all_metrics_rows()
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0][4], "2026-8-5T00:00:00Z")
        from datetime import datetime
        datetime.fromisoformat(rows[0][4].replace("Z", "+00:00"))


class MetadataShape(MigrateTestCase):
    def test_metadata_preserves_original_fields_and_adds_traceability(self):
        info = {"issue": 7, "pr": 20, "outcome": "MERGED", "model": "claude-sonnet-5", "rounds": 2, "manual_interventions": 0, "wall_clock_s": 5.0, "subagent_tokens": 100, "tool_calls": 3, "findings": [{"category": "x", "reproduced": True, "fixed": True}]}
        points = [_point("point-abc", information=info)]
        self._run_migrate(points)
        rows = self._all_metrics_rows()
        metadata = json.loads(rows[0][3])
        for key, value in info.items():
            self.assertEqual(metadata[key], value)
        self.assertEqual(metadata["qdrant_point_id"], "point-abc")
        self.assertEqual(metadata["qdrant_collection"], "conversation-compacts-claude-runway-autowork-metrics-fc6e9795")


class LegacyDocumentFallback(MigrateTestCase):
    def test_falls_back_to_document_field_when_information_missing(self):
        points = [_point("p1", document={"issue": 8, "outcome": "MERGED"})]
        exit_code, _ = self._run_migrate(points)
        self.assertEqual(exit_code, 0)
        rows = self._all_metrics_rows()
        self.assertEqual(len(rows), 1)
        metadata = json.loads(rows[0][3])
        self.assertEqual(metadata["issue"], 8)


class UnparseablePoints(MigrateTestCase):
    """PR #246 review (Copilot): an unparseable point must not crash the
    whole run (every OTHER point still gets migrated), but it must ALSO not
    let the run report PASS/exit 0 -- a source record silently dropped from
    the migration is exactly the "incomplete migration goes unnoticed"
    outcome the row-count verification exists to catch. An earlier version
    subtracted unparseable points from `expected`, which made verification
    pass even though fewer rows landed in metrics.db than Qdrant points
    existed."""

    def test_point_with_no_information_or_document_does_not_crash_the_run(self):
        points = [
            _point("bad"),  # neither information nor document set
            _point("good", information={"issue": 9, "outcome": "MERGED"}),
        ]
        exit_code, stdout = self._run_migrate(points)
        rows = self._all_metrics_rows()
        self.assertEqual(len(rows), 1)  # the good point still migrated
        self.assertIn("Skipped 1 point", stdout)
        # But verification must fail: 2 Qdrant points found, only 1 row landed.
        self.assertEqual(exit_code, 1)
        self.assertIn("FAIL", stdout)

    def test_unparseable_json_does_not_crash_the_run_but_fails_verification(self):
        points = [_point("bad", payload_overrides={"information": "{not json"})]
        exit_code, stdout = self._run_migrate(points)
        self.assertEqual(len(self._all_metrics_rows()), 0)
        self.assertIn("Skipped 1 point", stdout)
        self.assertEqual(exit_code, 1)
        self.assertIn("FAIL", stdout)


class Idempotency(MigrateTestCase):
    def test_rerunning_does_not_duplicate_rows(self):
        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"})]
        self._run_migrate(points)
        self.assertEqual(len(self._all_metrics_rows()), 1)
        exit_code, stdout = self._run_migrate(points)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(self._all_metrics_rows()), 1)
        self.assertIn("already migrated", stdout)

    def test_a_new_point_added_after_first_run_is_picked_up_on_rerun(self):
        p1 = _point("p1", information={"issue": 1, "outcome": "MERGED"})
        p2 = _point("p2", information={"issue": 2, "outcome": "FAILED"})
        self._run_migrate([p1])
        self.assertEqual(len(self._all_metrics_rows()), 1)
        self._run_migrate([p1, p2])
        self.assertEqual(len(self._all_metrics_rows()), 2)

    def test_pre_existing_empty_db_file_with_no_schema_does_not_crash(self):
        """PR #246 review (Copilot): --metrics-db can point at a path that
        already exists as a file (e.g. touch'd, or opened by some other
        process) but has no `metrics` table yet. _already_migrated_keys
        used to assume "file exists" meant "has a queryable metrics table,"
        and crashed with sqlite3.OperationalError: no such table: metrics
        before MetricsStore.record() ever got a chance to create it.
        Confirmed by reproduction before the fix."""
        # Create an empty sqlite file with no schema at all.
        conn = sqlite3.connect(str(self._db_path))
        conn.close()
        self.assertTrue(self._db_path.exists())

        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"})]
        exit_code, _ = self._run_migrate(points)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(self._all_metrics_rows()), 1)


class DryRunNeverWrites(MigrateTestCase):
    def test_dry_run_writes_nothing(self):
        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"})]
        exit_code, stdout = self._run_migrate(points, dry_run=True)
        self.assertEqual(exit_code, 0)
        self.assertEqual(self._all_metrics_rows(), [])
        self.assertIn("Dry run", stdout)
        self.assertIn("Nothing written", stdout)

    def test_dry_run_never_constructs_a_real_metrics_store_write(self):
        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"})]
        with mock.patch.object(metrics_lib_mod.MetricsStore, "record") as fake_record:
            self._run_migrate(points, dry_run=True)
        fake_record.assert_not_called()


class VerificationResult(MigrateTestCase):
    def test_passes_when_counts_match(self):
        points = [
            _point("p1", information={"issue": 1, "outcome": "MERGED"}),
            _point("p2", information={"issue": 2, "outcome": "FAILED"}),
        ]
        exit_code, stdout = self._run_migrate(points)
        self.assertEqual(exit_code, 0)
        self.assertIn("PASS", stdout)

    def test_fails_when_a_migrated_row_is_deleted_out_from_under_it(self):
        """Not a realistic normal-operation scenario -- this pins the
        verification's actual failure path (rather than only ever exercising
        the passing path) by tampering with metrics.db directly between the
        migration writes and the verification read."""
        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"})]

        # Patch _autowork_row_count to simulate a mismatch without needing a
        # real deletion race -- isolates the exit-code/message contract from
        # how a mismatch could arise in practice.
        with mock.patch.object(mod, "_autowork_row_count", return_value=0):
            exit_code, stdout = self._run_migrate(points)
        self.assertEqual(exit_code, 1)
        self.assertIn("FAIL", stdout)


class SharedCollectionIsolation(MigrateTestCase):
    """PR #246 review (Copilot, high severity): _collections_for_project
    resolves COLLECTION names that hold this project's history under any
    casing -- it does NOT guarantee every point inside a resolved
    collection belongs to this project (compact_store's own `collection`
    param is a caller-controlled override, so a collection CAN be shared
    across projects). compact_find defends against this with a
    case-insensitive per-point filter on the payload's own `project` field;
    an earlier version of _fetch_points appended every point unconditionally,
    which would have migrated a foreign project's payload in as an
    "autowork" metric. Confirmed by reproduction before the fix."""

    def test_point_belonging_to_a_different_project_in_a_shared_collection_is_excluded(self):
        fake_client = mock.MagicMock()

        def fake_scroll(collection_name, limit, offset, with_payload):
            if offset is not None:
                return [], None
            return [
                _point("mine", information={"issue": 1, "outcome": "MERGED"}),
                _point("foreign", information={"issue": 999, "outcome": "MERGED"}, payload_overrides={"project": "some-other-project"}),
            ], None

        fake_client.scroll.side_effect = fake_scroll
        args = _args(metrics_db=str(self._db_path))
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "_collections_for_project", return_value=["shared-collection"]):
            with redirect_stdout(io.StringIO()):
                exit_code = mod.migrate(args)
        self.assertEqual(exit_code, 0)
        rows = self._all_metrics_rows()
        self.assertEqual(len(rows), 1)
        metadata = json.loads(rows[0][3])
        self.assertEqual(metadata["issue"], 1)

    def test_project_match_is_case_insensitive(self):
        fake_client = mock.MagicMock()

        def fake_scroll(collection_name, limit, offset, with_payload):
            if offset is not None:
                return [], None
            return [
                _point("mine", information={"issue": 1, "outcome": "MERGED"}, payload_overrides={"project": "Claude-Runway-Autowork-Metrics"}),
            ], None

        fake_client.scroll.side_effect = fake_scroll
        args = _args(metrics_db=str(self._db_path))
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "_collections_for_project", return_value=["col"]):
            with redirect_stdout(io.StringIO()):
                exit_code = mod.migrate(args)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(self._all_metrics_rows()), 1)


class MultipleCollectionsResolved(MigrateTestCase):
    def test_points_from_every_resolved_collection_are_counted(self):
        fake_client = mock.MagicMock()

        def fake_scroll(collection_name, limit, offset, with_payload):
            if offset is not None:
                return [], None
            if collection_name == "col-a":
                return [_point("a1", information={"issue": 1, "outcome": "MERGED"})], None
            if collection_name == "col-b":
                return [_point("b1", information={"issue": 2, "outcome": "FAILED"})], None
            return [], None

        fake_client.scroll.side_effect = fake_scroll
        args = _args(metrics_db=str(self._db_path))
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "_collections_for_project", return_value=["col-a", "col-b"]):
            with redirect_stdout(io.StringIO()):
                exit_code = mod.migrate(args)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(self._all_metrics_rows()), 2)


class CollisionAcrossCollections(MigrateTestCase):
    """PR #246 review (Copilot, round 3, high severity): a Qdrant point id
    is only unique WITHIN its own collection, not guaranteed unique across
    every collection this script can resolve for one project (casing-drift
    aggregation can resolve more than one). Keying idempotency/verification
    by point id alone under-counted real migrated rows and could make
    verification never pass; confirmed by reproduction before the fix."""

    def _two_collections_same_point_id(self):
        fake_client = mock.MagicMock()

        def fake_scroll(collection_name, limit, offset, with_payload):
            if offset is not None:
                return [], None
            if collection_name == "col-a":
                return [SimpleNamespace(id="SAME-ID", payload={
                    "project": "claude-runway-autowork-metrics", "date": "2026-09-01",
                    "information": json.dumps({"issue": 1, "outcome": "MERGED"}),
                })], None
            if collection_name == "col-b":
                return [SimpleNamespace(id="SAME-ID", payload={
                    "project": "claude-runway-autowork-metrics", "date": "2026-09-02",
                    "information": json.dumps({"issue": 2, "outcome": "FAILED"}),
                })], None
            return [], None

        fake_client.scroll.side_effect = fake_scroll
        return fake_client

    def test_same_point_id_in_two_collections_both_migrate_and_verify_passes(self):
        fake_client = self._two_collections_same_point_id()
        args = _args(metrics_db=str(self._db_path))
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "_collections_for_project", return_value=["col-a", "col-b"]):
            with redirect_stdout(io.StringIO()) as buf:
                exit_code = mod.migrate(args)
        self.assertEqual(exit_code, 0)
        self.assertIn("PASS", buf.getvalue())
        rows = self._all_metrics_rows()
        self.assertEqual(len(rows), 2)  # both really written, not collapsed to 1

    def test_rerun_after_partial_migration_does_not_lose_the_unmigrated_copy(self):
        """The sharper failure mode: a prior run only got as far as
        migrating collection A's copy (e.g. crashed before reaching B).
        A point-id-only skip check would have wrongly treated B's copy as
        "already migrated" too on the next run, silently losing it."""
        fake_client = self._two_collections_same_point_id()
        args = _args(metrics_db=str(self._db_path))
        # First "run": only resolve col-a, simulating a partial prior run.
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "_collections_for_project", return_value=["col-a"]):
            with redirect_stdout(io.StringIO()):
                mod.migrate(args)
        self.assertEqual(len(self._all_metrics_rows()), 1)

        # Second run: both collections now resolve -- col-b's copy (same
        # point id, different collection) must still get migrated.
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "_collections_for_project", return_value=["col-a", "col-b"]):
            with redirect_stdout(io.StringIO()) as buf:
                exit_code = mod.migrate(args)
        self.assertEqual(exit_code, 0)
        self.assertIn("PASS", buf.getvalue())
        self.assertEqual(len(self._all_metrics_rows()), 2)


class ConcurrentRunGuard(MigrateTestCase):
    """PR #246 review (Copilot, "previously missed" -- concurrent
    migrations): the idempotency check is a read performed before any
    inserts, with no lock between the read and the writes -- two
    concurrent real runs could both see the same "not yet migrated" set
    and both insert the same points, producing duplicate rows the
    set-based verifier then collapses back down, potentially misreporting
    PASS. An exclusive lock file (created atomically, removed when the
    run finishes) now guards the whole non-dry-run body."""

    def test_lock_file_is_created_during_and_removed_after_a_real_run(self):
        points = [_point("p1", information={"issue": 1, "outcome": "MERGED"})]
        seen_lock_exists_during_run = []
        real_record = metrics_lib_mod.MetricsStore.record

        def spy_record(self, *a, **kw):
            seen_lock_exists_during_run.append(mod._lock_path_for(self).exists())
            return real_record(self, *a, **kw)

        with mock.patch.object(metrics_lib_mod.MetricsStore, "record", spy_record):
            exit_code, _ = self._run_migrate(points)
        self.assertEqual(exit_code, 0)
        self.assertEqual(seen_lock_exists_during_run, [True])  # held during the write
        store = metrics_lib_mod.MetricsStore(db_path=self._db_path)
        self.assertFalse(mod._lock_path_for(store).exists())  # released afterward

    def test_second_concurrent_real_run_fails_fast_instead_of_racing(self):
        store = metrics_lib_mod.MetricsStore(db_path=self._db_path)
        lock_path = mod._lock_path_for(store)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        mod._acquire_lock(lock_path)
        try:
            points = [_point("p1", information={"issue": 1, "outcome": "MERGED"})]
            exit_code, stdout = self._run_migrate(points)
            self.assertEqual(exit_code, 1)
            self.assertIn("FAIL", stdout)
            self.assertIn(str(lock_path), stdout)
            self.assertEqual(self._all_metrics_rows(), [])  # never even attempted a write
        finally:
            lock_path.unlink()

    def test_dry_run_never_takes_the_lock_even_if_one_is_held(self):
        store = metrics_lib_mod.MetricsStore(db_path=self._db_path)
        lock_path = mod._lock_path_for(store)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        mod._acquire_lock(lock_path)
        try:
            points = [_point("p1", information={"issue": 1, "outcome": "MERGED"})]
            exit_code, stdout = self._run_migrate(points, dry_run=True)
            self.assertEqual(exit_code, 0)
            self.assertIn("Dry run", stdout)
        finally:
            lock_path.unlink()

    def test_lock_is_released_even_when_the_run_fails(self):
        points = [_point("bad", payload_overrides={"information": "{not json"})]
        exit_code, _ = self._run_migrate(points)
        self.assertEqual(exit_code, 1)  # unparseable point -> verification FAIL
        store = metrics_lib_mod.MetricsStore(db_path=self._db_path)
        self.assertFalse(mod._lock_path_for(store).exists())


if __name__ == "__main__":
    unittest.main()
