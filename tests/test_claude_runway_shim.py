#!/usr/bin/env python3
"""Tests for claude_runway/__init__.py, the console-script shim.

Pins that the entry points load <env>/src/tools/<name>.py by exact path, so an
unrelated regular `tools` package elsewhere on sys.path (another dependency's,
or one on PYTHONPATH) can't shadow the namespace-package <env>/src/tools.

Stdlib-only (unittest), no network.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import claude_runway  # noqa: E402


class ToolsNamespaceShadowing(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        root = Path(self._tmpdir.name)

        # Fake install prefix: <prefix>/src/tools/doctor.py, no __init__.py,
        # exactly like the wheel's data-file layout.
        self.prefix = root / "env"
        (self.prefix / "src" / "tools").mkdir(parents=True)
        (self.prefix / "src" / "tools" / "doctor.py").write_text("def main():\n    return 7\n", encoding="utf-8")

        # An unrelated REGULAR `tools` package on sys.path.
        decoy = root / "site"
        (decoy / "tools").mkdir(parents=True)
        (decoy / "tools" / "__init__.py").write_text("", encoding="utf-8")

        saved_path = list(sys.path)
        saved_tools = {k: v for k, v in sys.modules.items() if k == "tools" or k.startswith("tools.")}
        for k in saved_tools:
            del sys.modules[k]
        sys.path.append(str(decoy))

        def _restore():
            sys.path[:] = saved_path
            for k in [k for k in sys.modules if k == "tools" or k.startswith("tools.")]:
                del sys.modules[k]
            sys.modules.update(saved_tools)
            sys.modules.pop("_claude_runway_tool_doctor", None)

        self.addCleanup(_restore)

    def test_decoy_regular_package_really_would_shadow_a_package_import(self):
        # Guards the premise: without the shim's by-path loading, the decoy wins.
        sys.path.insert(0, str(self.prefix / "src"))
        with self.assertRaises(ModuleNotFoundError):
            __import__("tools.doctor")

    def test_entry_point_loads_the_shipped_file_despite_the_decoy(self):
        with mock.patch.object(sys, "prefix", str(self.prefix)):
            self.assertEqual(claude_runway.doctor_main(), 7)


if __name__ == "__main__":
    unittest.main()
