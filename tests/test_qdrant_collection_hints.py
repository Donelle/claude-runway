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
    set_cached_description,
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


if __name__ == "__main__":
    unittest.main()
