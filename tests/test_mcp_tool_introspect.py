#!/usr/bin/env python3
"""Tests for libs/mcp_tool_introspect.py -- the shared utility extracted
in issue #52/GROW-05 so compress_mcp_server.py, ingest_mcp_server.py, and
tools/report_tool_counts.py all enumerate an MCPServer's registered tools
through one identical code path instead of duplicating (and independently
drifting on) the same "mcp._tool_manager.list_tools()" logic in three
places.

Stdlib-only (unittest, no pytest) and no network -- exercises a small
ad-hoc MCPServer instance built just for these tests, not either of the
real server modules (those are covered by their own test files).

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from mcp_tool_introspect import describe_tools, list_registered_tools, tool_count  # noqa: E402


class ListRegisteredToolsReflectsCurrentRegistrations(unittest.TestCase):
    def test_empty_server_has_no_tools(self):
        mcp = MCPServer("empty-test-server")
        self.assertEqual(list_registered_tools(mcp), [])
        self.assertEqual(tool_count(mcp), 0)

    def test_registered_tools_are_returned_by_name(self):
        mcp = MCPServer("test-server")

        @mcp.tool()
        def alpha() -> str:
            """First tool."""
            return "a"

        @mcp.tool()
        def beta(x: int) -> str:
            """Second tool."""
            return str(x)

        names = {fn.__name__ for fn in list_registered_tools(mcp)}
        self.assertEqual(names, {"alpha", "beta"})
        self.assertEqual(tool_count(mcp), 2)

    def test_a_tool_added_after_the_first_query_is_picked_up_on_the_next_one(self):
        # The whole point of enumerating live instead of caching a count:
        # a tool registered later must show up without any code change to
        # this module -- the exact regression class issues #22/#23 filed
        # against the OLD hardcoded-tuple/hardcoded-doc-number approach.
        mcp = MCPServer("test-server")
        self.assertEqual(tool_count(mcp), 0)

        @mcp.tool()
        def _late_addition() -> str:
            """Registered after the first tool_count() call above."""
            return "x"

        self.assertEqual(tool_count(mcp), 1)
        self.assertIn("_late_addition", {fn.__name__ for fn in list_registered_tools(mcp)})


class DescribeToolsIncludesNameSignatureAndDocstring(unittest.TestCase):
    def test_blob_contains_name_signature_and_docstring_for_each_tool(self):
        mcp = MCPServer("test-server")

        @mcp.tool()
        def widget(count: int, label: str = "x") -> str:
            """A widget-shaped tool for testing describe_tools()."""
            return label * count

        blob = describe_tools(mcp)
        self.assertIn("widget", blob)
        self.assertIn("count", blob)
        self.assertIn("label", blob)
        self.assertIn("A widget-shaped tool for testing describe_tools().", blob)

    def test_blob_grows_when_a_new_tool_is_registered(self):
        mcp = MCPServer("test-server")

        @mcp.tool()
        def one() -> str:
            """Tool one."""
            return "1"

        first_blob = describe_tools(mcp)

        @mcp.tool()
        def two() -> str:
            """Tool two, registered after the first describe_tools() call."""
            return "2"

        second_blob = describe_tools(mcp)
        self.assertGreater(len(second_blob), len(first_blob))
        self.assertIn("two", second_blob)

    def test_tool_with_no_docstring_does_not_raise(self):
        mcp = MCPServer("test-server")

        @mcp.tool()
        def undocumented() -> str:
            return "no docstring here"

        # Must not raise even though __doc__ is None -- describe_tools()
        # falls back to an empty string per tool.
        blob = describe_tools(mcp)
        self.assertIn("undocumented", blob)


if __name__ == "__main__":
    unittest.main()
