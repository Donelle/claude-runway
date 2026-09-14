#!/usr/bin/env python3
"""Tests for tools/doctor.py's CLI wiring (issue #49) -- argument parsing,
exit codes, and report formatting on top of libs/doctor_lib.py, which has
its own dedicated tests in test_doctor_lib.py.

Stdlib-only (unittest), no network:

    .venv/bin/python -m unittest discover -s tests
"""

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tools"))


def _load_doctor():
    """Loads a fresh copy of tools/doctor.py by file path -- same pattern
    test_setup_project_cli.py uses, since tools/ isn't normally an
    importable package."""
    spec = importlib.util.spec_from_file_location("doctor", REPO_ROOT / "tools" / "doctor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DoctorCli(unittest.TestCase):
    def setUp(self):
        self.mod = _load_doctor()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-project"
        self.target_repo.mkdir()

    def _write_mcp_json(self, local_compress_env=None):
        servers = {"qdrant": {"command": "x", "env": {}}}
        if local_compress_env is not None:
            servers["local-compress"] = {"command": "x", "env": local_compress_env}
        (self.target_repo / ".mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")

    def test_nonexistent_target_repo_exits_nonzero(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            exit_code = self.mod.main([str(self.target_repo / "does-not-exist")])
        self.assertEqual(exit_code, 1)
        self.assertIn("does not exist", buf.getvalue())

    def test_malformed_mcp_json_exits_one_not_zero(self):
        # Regression for the finding from PR #119 review: a malformed
        # .mcp.json (here, "mcpServers": null) previously either exited 0
        # (indistinguishable from a valid --qdrant-only setup) or crashed
        # outright with an uncaught AttributeError.
        (self.target_repo / ".mcp.json").write_text(json.dumps({"mcpServers": None}), encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = self.mod.main([str(self.target_repo)])
        self.assertEqual(exit_code, 1)
        self.assertIn("error:", buf.getvalue())

    def test_no_local_compress_configured_exits_zero(self):
        self._write_mcp_json(local_compress_env=None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = self.mod.main([str(self.target_repo)])
        self.assertEqual(exit_code, 0)
        self.assertIn("nothing to check", buf.getvalue())

    def test_clean_config_exits_zero(self):
        self._write_mcp_json(local_compress_env={"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1"})
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1"}, clear=False):
            buf = io.StringIO()
            with redirect_stdout(buf):
                exit_code = self.mod.main([str(self.target_repo)])
        self.assertEqual(exit_code, 0)
        self.assertIn("OK", buf.getvalue())

    def test_mismatch_exits_one_and_prints_both_values(self):
        self._write_mcp_json(local_compress_env={"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:9999/v1"})
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1"}, clear=False):
            buf = io.StringIO()
            with redirect_stdout(buf):
                exit_code = self.mod.main([str(self.target_repo)])
        output = buf.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("CLAUDE_RUNWAY_LMSTUDIO_URL", output)
        self.assertIn("http://localhost:9999/v1", output)
        self.assertIn("http://localhost:1234/v1", output)

    def test_target_repo_argument_is_required(self):
        with self.assertRaises(SystemExit) as ctx:
            self.mod.parse_args([])
        self.assertEqual(ctx.exception.code, 2)  # argparse's standard usage-error exit code


if __name__ == "__main__":
    unittest.main()
