#!/usr/bin/env python3
"""Layout checks for the `mcpservers/` folder.

The MCP server scripts Claude Code launches live in `mcpservers/`, apart
from `tools/`, which holds the CLIs a person runs. Nothing in the language
enforces that split, so this pins it:

    .venv/bin/python -m unittest discover -s tests

Stdlib only, no network, and it imports no server (the tool-count tests already
do that), so it runs in a bare checkout.
"""

import glob
import os
import unittest
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPECTED_SERVERS = {
    "analytics_mcp_server.py",
    "codebase_reviewer_mcp_server.py",
    "compress_mcp_server.py",
    "ingest_mcp_server.py",
    "memory_bank_mcp_server.py",
    "qdrant_mcp_server.py",
}


class McpServersFolder(unittest.TestCase):
    def test_the_servers_are_in_mcpservers(self):
        found = {p.name for p in (ROOT / "mcpservers").glob("*_mcp_server.py")}
        self.assertEqual(found, EXPECTED_SERVERS)

    def test_no_server_script_remains_in_tools(self):
        # A hard move, no stubs: a leftover copy or forwarding stub in tools/
        # would keep an old .mcp.json path working by accident and hide that
        # `upgrade` never rewrote it.
        self.assertEqual(sorted(glob.glob(str(ROOT / "tools" / "*_mcp_server.py"))), [])

    def test_mcpservers_holds_only_the_servers(self):
        names = {p.name for p in (ROOT / "mcpservers").iterdir() if p.name != "__pycache__"}
        self.assertEqual(names, EXPECTED_SERVERS)

    def test_mcpservers_has_no_init_py(self):
        # With an __init__.py the folder becomes a package and, were it ever
        # named `mcp`, it would shadow the SDK package the servers import.
        self.assertFalse((ROOT / "mcpservers" / "__init__.py").exists())

    def test_no_directory_named_mcp_at_the_top_level(self):
        self.assertFalse((ROOT / "mcp").exists())

    def test_each_server_still_reaches_libs_from_its_new_folder(self):
        # Every server inserts `dirname(dirname(__file__))/libs` into sys.path.
        # That arithmetic is only right while the script sits exactly one level
        # below the repo root, as it does in mcpservers/.
        for name in sorted(EXPECTED_SERVERS):
            text = (ROOT / "mcpservers" / name).read_text(encoding="utf-8")
            self.assertIn("libs", text, name)
            self.assertIn("os.path.dirname(os.path.dirname(os.path.abspath(__file__)))", text, name)
        self.assertTrue((ROOT / "libs").is_dir())


class SetupPyShipsMcpServers(unittest.TestCase):
    def test_data_files_includes_the_mcpservers_directory(self):
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('_flat_data_files("mcpservers", "*.py")', text)


class NoStaleServerPaths(unittest.TestCase):
    """No tracked text names a server under the old tools/ path. The one
    deliberate exception is the migration that rewrites that path."""

    ALLOWED = {
        "libs/upgrade_lib.py",
        "tests/test_upgrade_lib.py",
        "tests/test_mcpservers_layout.py",
    }

    def test_no_old_server_path_outside_the_migration_and_its_tests(self):
        import re
        import subprocess

        try:
            tracked = subprocess.run(
                ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.split("\n")
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        pattern = re.compile(r"tools[/\\]+[a-z_]+_mcp_server")
        offenders = []
        for rel in tracked:
            if not rel or rel in self.ALLOWED:
                continue
            path = ROOT / rel
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if pattern.search(text):
                offenders.append(rel)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
