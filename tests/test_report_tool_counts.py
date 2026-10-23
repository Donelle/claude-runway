#!/usr/bin/env python3
"""Tests for tools/report_tool_counts.py -- the CLI (GROW-05)
that EVALUATION.md's Track A/B prose and the CI lint step both point at
instead of a hardcoded tool count/token estimate. Verifies it reports a
live, nonzero count AND schema size (in characters) for each
real server (all four are this repo's own) and
correctly signals failure (nonzero exit) if a server can't be
constructed/enumerated, rather than silently reporting zero -- a silent
zero would be indistinguishable from "this server genuinely has no
tools," which would defeat the entire point of this being a smoke check.

Stdlib-only (unittest, no pytest) and no network -- report_tool_counts.py
only imports/constructs each server (none of which connect to a live
Qdrant/LM Studio just to enumerate their already-registered tools; the
qdrant server builds its embedding model on first use, so importing it loads
none).

    .venv/bin/python -m unittest discover -s tests
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import report_tool_counts as _rtc  # noqa: E402


class ReportRepoServerReturnsLiveCountsAndSchemaSizes(unittest.TestCase):
    def test_local_compress_reports_a_positive_count_and_schema_size(self):
        count, chars = _rtc._report_repo_server("compress_mcp_server")
        self.assertGreater(count, 0)
        self.assertGreater(chars, 0)

    def test_codebase_indexer_reports_a_positive_count_and_schema_size(self):
        count, chars = _rtc._report_repo_server("ingest_mcp_server")
        self.assertGreater(count, 0)
        self.assertGreater(chars, 0)

    def test_unimportable_module_raises_rather_than_reporting_zero(self):
        # A silent 0 here would be indistinguishable from "this server
        # genuinely has no tools" -- the whole point of this script as a
        # CI smoke check is that an import/enumeration failure must be
        # loud, not swallowed into a misleadingly low number.
        # ModuleNotFoundError (not a blind Exception -- ruff B017) is what
        # _report_repo_server's own __import__(module_name) actually raises
        # for a nonexistent module.
        with self.assertRaises(ModuleNotFoundError):
            _rtc._report_repo_server("this_module_does_not_exist_anywhere")


class ReportQdrantServerCoversTheServerThatReplacedTheThirdPartyOne(unittest.TestCase):
    """The `qdrant` server used to be the unpinned third-party package's, whose
    tool count this repo could not control, so the report had to construct it
    specially. It is this repo's own `qdrant_mcp_server.py` now and goes through
    the same live enumeration as the others."""

    def test_reports_five_tools_without_a_live_connection(self):
        count, chars = _rtc._report_repo_server("qdrant_mcp_server")
        self.assertEqual(count, 5)
        self.assertGreater(chars, 0)

    def test_pinned_to_the_actual_tool_names_not_just_the_count(self):
        # A count-only assertion would still pass if a tool were renamed while
        # the count stayed at 5 -- the silent drift this report exists to catch.
        import qdrant_mcp_server
        from mcp_tool_introspect import list_registered_tools

        names = {fn.__name__ for fn in list_registered_tools(qdrant_mcp_server.mcp)}
        self.assertEqual(
            names, {"qdrant_find", "qdrant_store", "compact_store", "compact_find", "compact_prune"}
        )

    def test_local_compress_no_longer_registers_the_compact_tools(self):
        import compress_mcp_server
        from mcp_tool_introspect import list_registered_tools

        names = {fn.__name__ for fn in list_registered_tools(compress_mcp_server.mcp)}
        self.assertFalse(names & {"compact_store", "compact_find", "compact_prune"})


class AnalyticsServerIsReported(unittest.TestCase):
    """The savings and metrics tools moved off `local-compress` onto the opt-in
    `analytics` server; both must show their own live count."""

    def test_analytics_reports_the_five_moved_tools(self):
        count, chars = _rtc._report_repo_server("analytics_mcp_server")
        self.assertEqual(count, 5)
        self.assertGreater(chars, 0)

    def test_local_compress_reports_five_tools_and_none_of_the_moved_ones(self):
        import compress_mcp_server
        from mcp_tool_introspect import list_registered_tools

        names = {fn.__name__ for fn in list_registered_tools(compress_mcp_server.mcp)}
        self.assertEqual(
            names,
            {"list_local_models", "compress_file", "compress_command_output", "fetch_url", "compress_text"},
        )

    def test_main_prints_both_counts(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            _rtc.main()
        output = captured.getvalue()
        self.assertIn("local-compress (compress_mcp_server.py): 5 tools", output)
        self.assertIn("analytics (analytics_mcp_server.py): 5 tools", output)


class CodebaseReviewerServerIsReported(unittest.TestCase):
    def test_it_reports_its_two_tools(self):
        count, chars = _rtc._report_repo_server("codebase_reviewer_mcp_server")
        self.assertEqual(count, 2)  # list_reviewers and review
        self.assertGreater(chars, 0)

    def test_main_prints_its_line(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            _rtc.main()
        self.assertIn("codebase-reviewer (codebase_reviewer_mcp_server.py): 2 tools", captured.getvalue())


class MainReportsAllServersAndExitsCleanly(unittest.TestCase):
    def test_main_prints_all_server_labels_and_returns_zero(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            exit_code = _rtc.main()

        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("local-compress", output)
        self.assertIn("analytics", output)
        self.assertIn("codebase-indexer", output)
        self.assertIn("memory-bank", output)
        self.assertIn("qdrant (qdrant_mcp_server.py): 5 tools", output)
        self.assertNotIn("standalone", output)
        self.assertIn("tools", output)
        self.assertIn("tool-description characters (schema proxy)", output)
        # No token estimate is printed any more.
        self.assertNotIn("schema tokens", output)

    def test_main_reports_a_specific_live_count_matching_direct_introspection(self):
        # Cross-check against the same underlying utility used elsewhere
        # (libs/mcp_tool_introspect.py) so this test would fail if the two
        # ever disagreed -- the exact drift this whole issue exists to
        # prevent.
        import compress_mcp_server as _cms
        from mcp_tool_introspect import tool_count

        expected = tool_count(_cms.mcp)
        captured = io.StringIO()
        with redirect_stdout(captured):
            _rtc.main()
        self.assertIn(f"local-compress (compress_mcp_server.py): {expected} tools", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
