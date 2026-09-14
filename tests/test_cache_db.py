#!/usr/bin/env python3
"""Tests for libs/cache_db.py's shared cache-DB path resolution.

Stdlib-only (unittest, no pytest) and no network -- pure path-resolution
logic, no actual SQLite I/O happens here (that's exercised indirectly via
each real caller's own tests, e.g. tests/test_qdrant_collection_hints.py):

    .venv/bin/python -m unittest discover -s tests

Why this file exists: cache_db.py is deliberately a SEPARATE file from
libs/savings_ledger.py's savings.db -- one the user wants to keep forever,
one that's fine to delete anytime. Pinning resolve_cache_db_path()'s
default here (and confirming CLAUDE_RUNWAY_CACHE_DB is a distinct env var
from CLAUDE_RUNWAY_SAVINGS_DB) is what keeps that split from silently
drifting back together later.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from cache_db import resolve_cache_db_path  # noqa: E402


class ResolveCacheDbPath(unittest.TestCase):
    def test_defaults_under_claude_runway_home_dir(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_RUNWAY_CACHE_DB", None)
            expected = Path.home() / ".claude" / "claude-runway" / "cache.db"
            self.assertEqual(resolve_cache_db_path(), expected)

    def test_default_is_a_different_file_than_savings_db(self):
        # The whole point of this module: never the same path as
        # savings_ledger's DB, so "delete the cache" can never accidentally
        # also delete the savings history sitting next to it.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_RUNWAY_CACHE_DB", None)
            self.assertNotEqual(
                resolve_cache_db_path(),
                Path.home() / ".claude" / "claude-runway" / "savings.db",
            )

    def test_absolute_override_is_used_as_is(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_CACHE_DB": "/custom/cache.db"}, clear=False):
            self.assertEqual(resolve_cache_db_path(), Path("/custom/cache.db"))

    def test_relative_override_is_anchored_to_home_not_cwd(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_CACHE_DB": "relative/cache.db"}, clear=False):
            self.assertEqual(resolve_cache_db_path(), Path.home() / "relative" / "cache.db")

    def test_expanduser_override_is_honored(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_CACHE_DB": "~/custom/cache.db"}, clear=False):
            self.assertEqual(resolve_cache_db_path(), Path.home() / "custom" / "cache.db")


if __name__ == "__main__":
    unittest.main()
