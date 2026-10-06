#!/usr/bin/env python3
"""Tests for libs/qdrant_collection_hints.py's persistent hint cache.

Stdlib-only (unittest, no pytest) and no network -- this is a pure SQLite
cache layer, so testing it needs neither Qdrant nor an embedding model:

    .venv/bin/python -m unittest discover -s tests

Why this file exists (issue #81): list_collections surfaces a per-collection
description hint so a model can pick the right collection on its own. The
cache exists specifically to make that cheap across sessions/restarts --
these tests pin down the two properties that actually matter for that: a
miss is genuinely distinguishable from a cached-empty hit (otherwise
"never fetched" and "fetched, found nothing" would look identical and the
caller couldn't know whether to skip a live fetch), and a repeated set()
overwrites rather than accumulating duplicate rows.

Storage is the shared cache DB from libs/cache_db.py (see
tests/test_cache_db.py for that module's own path-resolution tests) --
these tests only exercise the collection_hints table this module keeps
inside it.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import qdrant_collection_hints  # noqa: E402
from qdrant_collection_hints import (  # noqa: E402
    get_cached_descriptions,
    get_cached_search_limit,
    get_cached_search_limits,
    set_cached_description,
    set_cached_search_limit,
)

# Path-resolution tests for the shared cache DB itself live in
# tests/test_cache_db.py -- this file only covers the collection_hints
# table's own read/write behavior against it.


class CollectionHintsCache(unittest.TestCase):
    """Each test gets its own temp DB file so tests can't see each other's
    cached rows -- CLAUDE_RUNWAY_CACHE_DB (libs/cache_db.py's override) is
    set fresh per test.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "cache.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_CACHE_DB": db_path}, clear=False)
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_miss_is_absent_not_empty_string(self):
        result = get_cached_descriptions("http://localhost:6333", ["never-cached"])
        self.assertNotIn("never-cached", result)

    def test_set_then_get_round_trips(self):
        set_cached_description("http://localhost:6333", "my-repo", "Backend API for widgets")
        result = get_cached_descriptions("http://localhost:6333", ["my-repo"])
        self.assertEqual(result["my-repo"], "Backend API for widgets")

    def test_cached_empty_description_is_a_hit_not_a_miss(self):
        # A collection genuinely has no description set -- list_collections
        # caches "" itself (see its own docstring) so it isn't re-fetched
        # every call. That must read back as present, not absent.
        set_cached_description("http://localhost:6333", "no-hint-repo", "")
        result = get_cached_descriptions("http://localhost:6333", ["no-hint-repo"])
        self.assertIn("no-hint-repo", result)
        self.assertEqual(result["no-hint-repo"], "")

    def test_repeated_set_overwrites_rather_than_duplicating(self):
        set_cached_description("http://localhost:6333", "my-repo", "first description")
        set_cached_description("http://localhost:6333", "my-repo", "second description")
        result = get_cached_descriptions("http://localhost:6333", ["my-repo"])
        self.assertEqual(result["my-repo"], "second description")

    def test_batch_lookup_returns_only_requested_names_that_are_cached(self):
        set_cached_description("http://localhost:6333", "repo-a", "A")
        set_cached_description("http://localhost:6333", "repo-b", "B")
        result = get_cached_descriptions("http://localhost:6333", ["repo-a", "repo-b", "repo-c"])
        self.assertEqual(result, {"repo-a": "A", "repo-b": "B"})

    def test_different_qdrant_url_is_a_separate_cache_entry(self):
        # Same collection name, two different instances -- must not collide.
        set_cached_description("http://localhost:6333", "shared-name", "instance one's content")
        set_cached_description("http://otherhost:6333", "shared-name", "instance two's content")
        result_one = get_cached_descriptions("http://localhost:6333", ["shared-name"])
        result_two = get_cached_descriptions("http://otherhost:6333", ["shared-name"])
        self.assertEqual(result_one["shared-name"], "instance one's content")
        self.assertEqual(result_two["shared-name"], "instance two's content")

    def test_empty_collections_list_returns_empty_dict_without_querying(self):
        self.assertEqual(get_cached_descriptions("http://localhost:6333", []), {})


class CacheFailsOpen(unittest.TestCase):
    """Copilot review on PR #82 caught this: a disposable cache must never
    turn into a reported failure for list_collections/set_collection_
    description, since by the time either cache write runs, Qdrant itself
    -- the authoritative source -- has already been read or written
    successfully. A cache read failure must fall through to "nothing
    cached" (triggering a live Qdrant fetch upstream), not raise; a cache
    write failure must be swallowed (logged, not raised), not turn an
    already-successful Qdrant operation into an apparent tool failure.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "cache.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_CACHE_DB": db_path}, clear=False)
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_get_returns_empty_dict_instead_of_raising_on_db_error(self):
        with mock.patch.object(
            qdrant_collection_hints, "_connect", side_effect=sqlite3.OperationalError("disk I/O error")
        ):
            result = get_cached_descriptions("http://localhost:6333", ["some-repo"])
        self.assertEqual(result, {})

    def test_set_does_not_raise_on_db_error(self):
        with mock.patch.object(
            qdrant_collection_hints, "_connect", side_effect=sqlite3.OperationalError("disk I/O error")
        ):
            set_cached_description("http://localhost:6333", "some-repo", "a description")  # must not raise

    def test_set_does_not_raise_on_oserror(self):
        # Covers a read-only home directory / unwritable parent dir, which
        # surfaces as OSError from Path.mkdir() inside cache_db.connect(),
        # not a sqlite3 exception.
        with mock.patch.object(qdrant_collection_hints, "_connect", side_effect=OSError("Read-only file system")):
            set_cached_description("http://localhost:6333", "some-repo", "a description")  # must not raise

    def test_non_string_description_is_normalized_to_empty_string(self):
        # Qdrant collection metadata is arbitrary JSON -- a description read
        # back from it (list_collections's cache-miss path) could legally be
        # None, a dict, or a list. Reproduced directly that binding any of
        # these into the TEXT NOT NULL column raises IntegrityError (None)
        # or ProgrammingError (dict/list) with no normalization -- this
        # covers all three actually surviving as a no-op, not an exception.
        for bad_value in (None, {"nested": "object"}, [1, 2, 3]):
            with self.subTest(bad_value=bad_value):
                set_cached_description("http://localhost:6333", "weird-metadata-repo", bad_value)  # must not raise
                result = get_cached_descriptions("http://localhost:6333", ["weird-metadata-repo"])
                self.assertEqual(result["weird-metadata-repo"], "")


class _TempCacheDb(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "cache.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_CACHE_DB": self.db_path}, clear=False)
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()


URL = "http://localhost:6333"


class SearchLimitCache(_TempCacheDb):
    def test_miss_is_absent(self):
        self.assertEqual(get_cached_search_limits(URL, ["x"]), {})
        self.assertIsNone(get_cached_search_limit(URL, "x"))

    def test_round_trip_and_overwrite(self):
        set_cached_search_limit(URL, "repo", 5)
        set_cached_search_limit(URL, "repo", 9)
        self.assertEqual(get_cached_search_limit(URL, "repo"), 9)

    def test_writers_do_not_clobber_each_other(self):
        set_cached_description(URL, "repo", "desc")
        set_cached_search_limit(URL, "repo", 7)
        self.assertEqual(get_cached_descriptions(URL, ["repo"]), {"repo": "desc"})
        self.assertEqual(get_cached_search_limit(URL, "repo"), 7)
        set_cached_description(URL, "repo", "new desc")
        self.assertEqual(get_cached_search_limit(URL, "repo"), 7)
        set_cached_search_limit(URL, "repo", 8)
        self.assertEqual(get_cached_descriptions(URL, ["repo"]), {"repo": "new desc"})

    def test_limit_only_row_is_a_description_miss(self):
        set_cached_search_limit(URL, "repo", 3)
        self.assertEqual(get_cached_descriptions(URL, ["repo"]), {})

    def test_description_only_row_is_a_limit_miss(self):
        set_cached_description(URL, "repo", "desc")
        self.assertEqual(get_cached_search_limits(URL, ["repo"]), {})

    def test_cleared_description_still_a_hit_alongside_limit(self):
        set_cached_search_limit(URL, "repo", 3)
        set_cached_description(URL, "repo", "")
        self.assertEqual(get_cached_descriptions(URL, ["repo"]), {"repo": ""})

    def test_non_integer_limit_is_skipped_without_raising(self):
        for bad in (None, "5", 1.5, True):
            with self.subTest(bad=bad):
                set_cached_search_limit(URL, "repo", bad)  # type: ignore[arg-type]
        self.assertEqual(get_cached_search_limits(URL, ["repo"]), {})

    def test_set_and_get_fail_open(self):
        with mock.patch.object(qdrant_collection_hints, "_connect", side_effect=sqlite3.OperationalError("boom")):
            set_cached_search_limit(URL, "repo", 5)
            self.assertEqual(get_cached_search_limits(URL, ["repo"]), {})


class SchemaMigration(_TempCacheDb):
    def _columns(self):
        conn = sqlite3.connect(self.db_path)
        try:
            return {r[1]: r for r in conn.execute("PRAGMA table_info(collection_hints)").fetchall()}
        finally:
            conn.close()

    def _version(self):
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT version FROM schema_versions WHERE name = 'collection_hints'"
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def test_fresh_db_gets_latest_schema_and_version(self):
        set_cached_description(URL, "repo", "d")
        cols = self._columns()
        self.assertIn("search_limit", cols)
        self.assertEqual(cols["description"][3], 0)  # nullable
        self.assertEqual(self._version(), qdrant_collection_hints.SCHEMA_VERSION)

    def test_legacy_table_is_dropped_and_recreated(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE collection_hints (qdrant_url TEXT NOT NULL, collection TEXT NOT NULL, "
            "description TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (qdrant_url, collection))"
        )
        conn.execute("INSERT INTO collection_hints VALUES (?, 'old', 'stale', 't')", (URL,))
        conn.commit()
        conn.close()
        self.assertEqual(get_cached_descriptions(URL, ["old"]), {})  # disposable cache: row gone
        cols = self._columns()
        self.assertIn("search_limit", cols)
        self.assertEqual(cols["description"][3], 0)
        self.assertEqual(self._version(), qdrant_collection_hints.SCHEMA_VERSION)
        set_cached_search_limit(URL, "old", 4)  # new shape is usable
        self.assertEqual(get_cached_search_limit(URL, "old"), 4)

    def test_older_process_does_not_lower_a_newer_stored_version(self):
        # PR #379 review: an older build opening a cache already migrated to
        # a newer version used to reset the stored version to its own.
        set_cached_description(URL, "repo", "d")
        conn = sqlite3.connect(self.db_path)
        newer = qdrant_collection_hints.SCHEMA_VERSION + 1
        conn.execute("UPDATE schema_versions SET version = ? WHERE name = 'collection_hints'", (newer,))
        conn.commit()
        conn.close()
        set_cached_description(URL, "repo", "d2")  # this build is older than `newer`
        self.assertEqual(self._version(), newer)
        self.assertEqual(get_cached_descriptions(URL, ["repo"]), {"repo": "d2"})

    def test_stale_higher_version_is_reset_when_table_is_recreated(self):
        # PR #379 review: schema_versions holds a higher version but the
        # collection_hints table is missing. The table we create is this
        # build's schema, so the stored version must be reset to match it --
        # keeping the higher one would let a later newer build skip migrations.
        set_cached_description(URL, "repo", "d")
        conn = sqlite3.connect(self.db_path)
        conn.execute("DROP TABLE collection_hints")
        conn.execute(
            "UPDATE schema_versions SET version = ? WHERE name = 'collection_hints'",
            (qdrant_collection_hints.SCHEMA_VERSION + 5,),
        )
        conn.commit()
        conn.close()
        set_cached_description(URL, "repo", "d2")  # recreates the table
        self.assertEqual(self._version(), qdrant_collection_hints.SCHEMA_VERSION)
        self.assertEqual(get_cached_descriptions(URL, ["repo"]), {"repo": "d2"})

    def test_rerun_does_not_wipe_data(self):
        set_cached_description(URL, "repo", "keep me")
        self.assertEqual(get_cached_descriptions(URL, ["repo"]), {"repo": "keep me"})
        self.assertEqual(get_cached_descriptions(URL, ["repo"]), {"repo": "keep me"})

    def test_already_v1_table_without_version_row_is_not_wiped(self):
        set_cached_description(URL, "repo", "keep me")
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM schema_versions")
        conn.commit()
        conn.close()
        self.assertEqual(get_cached_descriptions(URL, ["repo"]), {"repo": "keep me"})
        self.assertEqual(self._version(), qdrant_collection_hints.SCHEMA_VERSION)

    def test_does_not_touch_pragma_user_version_of_shared_db(self):
        set_cached_description(URL, "repo", "d")
        conn = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 0)
        finally:
            conn.close()

    def test_missing_migration_entry_raises_and_rolls_back(self):
        set_cached_description(URL, "repo", "d")
        current = qdrant_collection_hints.SCHEMA_VERSION
        with mock.patch.object(qdrant_collection_hints, "SCHEMA_VERSION", current + 1):
            with mock.patch.object(qdrant_collection_hints, "_MIGRATIONS", {}):
                with self.assertRaises(RuntimeError):
                    qdrant_collection_hints._connect()
        self.assertEqual(self._version(), current)


class ConnectionHygiene(_TempCacheDb):
    def test_connection_closed_even_when_the_statement_fails(self):
        for call in (
            lambda: set_cached_description(URL, "r", "d"),
            lambda: set_cached_search_limit(URL, "r", 1),
            lambda: get_cached_descriptions(URL, ["r"]),
            lambda: get_cached_search_limits(URL, ["r"]),
        ):
            fake = mock.MagicMock()
            fake.execute.side_effect = sqlite3.OperationalError("boom")
            with mock.patch.object(qdrant_collection_hints, "_connect", return_value=fake):
                call()  # fails open
            fake.close.assert_called_once()

    def test_no_write_lock_when_already_current(self):
        set_cached_description(URL, "r", "d")  # migrates / creates
        conn = sqlite3.connect(self.db_path)
        statements = []
        conn.set_trace_callback(statements.append)
        try:
            qdrant_collection_hints._run_migrations(conn)
        finally:
            conn.close()
        self.assertFalse([s for s in statements if "BEGIN IMMEDIATE" in s.upper()])

    def test_every_version_up_to_schema_version_has_a_migration(self):
        # Lockstep guard: the RuntimeError above is a loud dev-time failure,
        # so make sure this build never ships one.
        for version in range(1, qdrant_collection_hints.SCHEMA_VERSION + 1):
            self.assertIn(version, qdrant_collection_hints._MIGRATIONS)

    def test_failing_rollback_does_not_mask_the_original_error(self):
        class Boom(Exception):
            pass

        real = sqlite3.connect(self.db_path)

        class Conn:
            # Delegates everything, but makes ROLLBACK itself fail and
            # forces an error inside the migration body.
            def execute(self, sql, *a):
                if sql == "ROLLBACK":
                    raise sqlite3.OperationalError("rollback failed")
                if "schema_versions" in sql and sql.startswith("CREATE"):
                    raise Boom("original")
                return real.execute(sql, *a)

        try:
            with mock.patch.object(qdrant_collection_hints, "_is_current", return_value=False):
                with self.assertRaises(Boom):
                    qdrant_collection_hints._run_migrations(Conn())
        finally:
            real.close()


class ComputeSearchLimitTests(unittest.TestCase):
    """compute_search_limit (issue #330): pure tier lookup, boundaries exclusive."""

    def test_each_tier_and_boundary(self):
        cases = [
            (0, 10), (1, 10), (999, 10),
            (1_000, 15), (4_999, 15),
            (5_000, 25), (14_999, 25),
            (15_000, 40), (1_000_000, 40),
        ]
        for count, expected in cases:
            with self.subTest(point_count=count):
                self.assertEqual(qdrant_collection_hints.compute_search_limit(count), expected)

    def test_negative_count_uses_smallest_tier(self):
        self.assertEqual(qdrant_collection_hints.compute_search_limit(-5), 10)


if __name__ == "__main__":
    unittest.main()
