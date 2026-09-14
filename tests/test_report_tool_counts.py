#!/usr/bin/env python3
"""Tests for tools/report_tool_counts.py -- the CLI (issue #52/GROW-05)
that EVALUATION.md's Track A/B prose and the CI lint step both point at
instead of a hardcoded tool count/token estimate. Verifies it reports a
live, nonzero count AND token estimate for each real server (this repo's
own two, plus the standalone third-party `mcp-server-qdrant` package) and
correctly signals failure (nonzero exit) if a server can't be
constructed/enumerated, rather than silently reporting zero -- a silent
zero would be indistinguishable from "this server genuinely has no
tools," which would defeat the entire point of this being a smoke check.

Stdlib-only (unittest, no pytest) and no network -- report_tool_counts.py
only imports/constructs each server (none of which connect to a live
Qdrant/LM Studio just to enumerate their already-registered tools -- see
the module's own docstring for how the standalone qdrant server avoids
that too).

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


class ReportRepoServerReturnsLiveCountsAndTokenEstimates(unittest.TestCase):
    def test_local_compress_reports_a_positive_count_and_tokens(self):
        count, tokens = _rtc._report_repo_server("compress_mcp_server")
        self.assertGreater(count, 0)
        self.assertGreater(tokens, 0)

    def test_codebase_indexer_reports_a_positive_count_and_tokens(self):
        count, tokens = _rtc._report_repo_server("ingest_mcp_server")
        self.assertGreater(count, 0)
        self.assertGreater(tokens, 0)

    def test_unimportable_module_raises_rather_than_reporting_zero(self):
        # A silent 0 here would be indistinguishable from "this server
        # genuinely has no tools" -- the whole point of this script as a
        # CI smoke check is that an import/enumeration failure must be
        # loud, not swallowed into a misleadingly low number.
        with self.assertRaises(Exception):
            _rtc._report_repo_server("this_module_does_not_exist_anywhere")


class ReportStandaloneQdrantServerCoversTheThirdPartyHalf(unittest.TestCase):
    """PR #120 review finding: mcp-server-qdrant is unpinned in
    requirements.txt, so its own tool count isn't a repo-controlled
    invariant the way a hardcoded "2" in EVALUATION.md's prose implied.
    This introspects the actual installed package's server directly,
    rather than assuming it never changes.
    """

    def test_reports_the_two_known_qdrant_tools_without_a_live_connection(self):
        count, tokens = _rtc._report_standalone_qdrant_server()
        self.assertEqual(count, 2)
        self.assertGreater(tokens, 0)

    def test_pinned_to_the_actual_tool_names_not_just_the_count(self):
        # PR #120 review finding: a count-only assertion would still pass
        # if mcp-server-qdrant renamed a tool while keeping the count at 2
        # (e.g. "qdrant-find" -> "qdrant-search") -- that's exactly the
        # kind of silent drift this whole issue exists to catch, so this
        # constructs the same server directly and checks the actual
        # registered names, not just how many there are.
        import asyncio
        from unittest.mock import patch

        from mcp_server_qdrant.mcp_server import QdrantMCPServer
        from mcp_server_qdrant.settings import EmbeddingProviderSettings, QdrantSettings, ToolSettings

        # Patch out the embedding-provider factory so construction stays
        # network-free -- the provider is never invoked here (no real tool
        # calls are made), and FastEmbedProvider.__init__ would otherwise
        # trigger a model download on a cold runner.
        with patch("mcp_server_qdrant.mcp_server.create_embedding_provider", return_value=object()):
            srv = QdrantMCPServer(
                tool_settings=ToolSettings(),
                qdrant_settings=QdrantSettings(),
                embedding_provider_settings=EmbeddingProviderSettings(),
            )
        # QdrantMCPServer is a subclass of the standalone fastmcp.FastMCP
        # (not our MCPServer), so _tool_manager is not available. Use the
        # async list_tools() coroutine, which is the public API across all
        # fastmcp versions and returns objects with a .name attribute.
        tools = asyncio.run(srv.list_tools())
        names = {t.name for t in tools}
        self.assertEqual(names, {"qdrant-find", "qdrant-store"})


class MainReportsAllThreeServersAndExitsCleanly(unittest.TestCase):
    def test_main_prints_all_three_server_labels_and_returns_zero(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            exit_code = _rtc.main()

        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("local-compress", output)
        self.assertIn("codebase-indexer", output)
        self.assertIn("qdrant", output)
        self.assertIn("tools", output)
        self.assertIn("schema tokens", output)

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
