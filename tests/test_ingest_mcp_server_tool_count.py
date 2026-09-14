#!/usr/bin/env python3
"""Tests for ingest_mcp_server.py's _log_registered_tool_count() -- issue
#52/GROW-05's Track A counterpart to compress_mcp_server.py's
_log_fixed_overhead(). Before this, EVALUATION.md's Track A section had
no live self-reporting mechanism at all for the codebase-indexer server's
tool count, which is exactly why it drifted stale twice (issue #23's
"6 total" correction, then a further drift to 7 once
set_collection_description shipped after that fix landed).

Stdlib-only (unittest, no pytest) and no network -- _log_registered_tool_count()
never touches Qdrant/FastEmbed; it only enumerates already-registered tool
functions via libs/mcp_tool_introspect.py.

    .venv/bin/python -m unittest discover -s tests
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

# Minimal env so module-level DEFAULT_* constants don't explode on import
# -- same pattern already established in test_qdrant_ingest_lib.py for the
# identical reason (these all default gracefully to None/empty when unset,
# this just avoids depending on real infra being up).
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "test-collection")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

import ingest_mcp_server as _ims  # noqa: E402


class LogRegisteredToolCountReportsTheLiveCount(unittest.TestCase):
    def test_logs_the_actual_current_tool_count_to_stderr(self):
        captured = io.StringIO()
        with redirect_stderr(captured):
            _ims._log_registered_tool_count()

        output = captured.getvalue()
        expected_count = _ims.tool_count(_ims.mcp)
        self.assertGreater(expected_count, 0)
        self.assertIn(str(expected_count), output)
        self.assertIn("codebase-indexer", output)

    def test_never_writes_to_stdout(self):
        # This server's own module docstring bans stdout entirely (stdio
        # transport reserves it for MCP protocol messages) -- a print()
        # without file=sys.stderr here would silently corrupt the MCP
        # protocol stream in real use, not just look wrong in a log.
        captured_stdout = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(captured_stdout):
            _ims._log_registered_tool_count()
        self.assertEqual(captured_stdout.getvalue(), "")

    def test_a_newly_registered_tool_is_reflected_without_code_changes(self):
        before = _ims.tool_count(_ims.mcp)

        @_ims.mcp.tool()
        def _test_probe_tool_for_count_regression() -> str:
            """Registered after this test file was last edited."""
            return "x"

        captured = io.StringIO()
        with redirect_stderr(captured):
            _ims._log_registered_tool_count()

        self.assertIn(str(before + 1), captured.getvalue())


if __name__ == "__main__":
    unittest.main()
