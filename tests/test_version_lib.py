#!/usr/bin/env python3
"""Tests for libs/version_lib.py and the `--version` flag on the three CLIs
(issue #348).

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import os
import sys
import unittest
from importlib import metadata
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import version_lib  # noqa: E402


class GetVersion(unittest.TestCase):
    def test_uses_installed_distribution_metadata_when_not_in_a_checkout(self):
        with (
            mock.patch.object(version_lib, "_checkout_root", return_value=None),
            mock.patch.object(version_lib.metadata, "version", return_value="1.2.3.dev4+gabc1234"),
        ):
            self.assertEqual(version_lib.get_version(), "1.2.3.dev4+gabc1234")

    def test_checkout_wins_over_an_also_installed_distribution(self):
        # Copilot review (PR #372): running a clone with an interpreter that
        # ALSO has claude-runway installed must report the clone's version.
        with (
            mock.patch.object(version_lib, "_checkout_root", return_value=version_lib.Path(REPO_ROOT)),
            mock.patch.object(version_lib, "_describe_checkout", return_value="v0.1.0-5-gabc1234"),
            mock.patch.object(version_lib.metadata, "version", return_value="9.9.9"),
        ):
            self.assertEqual(version_lib.get_version(), "v0.1.0-5-gabc1234 (source checkout)")

    def test_checkout_without_a_usable_vcs_falls_back_to_installed_metadata(self):
        with (
            mock.patch.object(version_lib, "_checkout_root", return_value=version_lib.Path(REPO_ROOT)),
            mock.patch.object(version_lib, "_describe_checkout", return_value=None),
            mock.patch.object(version_lib.metadata, "version", return_value="1.2.3"),
        ):
            self.assertEqual(version_lib.get_version(), "1.2.3")

    def test_reports_unknown_when_neither_source_works(self):
        with (
            mock.patch.object(version_lib, "_checkout_root", return_value=None),
            mock.patch.object(version_lib.metadata, "version", side_effect=metadata.PackageNotFoundError),
        ):
            self.assertIn("unknown", version_lib.get_version())

    def test_checkout_root_detects_this_repo_and_rejects_a_copy_without_dot_git(self):
        self.assertEqual(version_lib._checkout_root(), version_lib.Path(REPO_ROOT).resolve())
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "src", "libs"))
            copied = os.path.join(d, "src", "libs", "version_lib.py")
            shutil.copy(os.path.join(REPO_ROOT, "libs", "version_lib.py"), copied)
            code = (
                "import importlib.util, sys;"
                f"s = importlib.util.spec_from_file_location('v', {copied!r});"
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m);"
                "print(m._checkout_root())"
            )
            import subprocess

            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(out.stdout.strip(), "None", out.stderr)

    def test_describe_checkout_survives_a_missing_vcs_binary(self):
        with mock.patch.object(version_lib.subprocess, "run", side_effect=FileNotFoundError):
            self.assertIsNone(version_lib._describe_checkout(version_lib.Path(REPO_ROOT)))


class VersionFlagOnEachCli(unittest.TestCase):
    def _run_version(self, parse_fn, argv):
        buf = io.StringIO()
        with (
            mock.patch.object(version_lib, "get_version", return_value="9.9.9-test"),
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(buf),
            self.assertRaises(SystemExit) as cm,
        ):
            parse_fn()
        self.assertEqual(cm.exception.code, 0)
        return buf.getvalue()

    def test_doctor_version_flag(self):
        import doctor

        out = self._run_version(lambda: doctor.parse_args(["--version"]), ["x"])
        self.assertIn("claude-runway-doctor", out)

    def test_setup_version_flag(self):
        import setup_project

        out = self._run_version(setup_project.parse_args, ["x", "--version"])
        self.assertIn("claude-runway-setup", out)

    def test_ingest_version_flag(self):
        import ingest_to_qdrant

        out = self._run_version(ingest_to_qdrant.parse_args, ["x", "--version"])
        self.assertIn("claude-runway-ingest", out)


class IngestVersionWorksWithBrokenDependencies(unittest.TestCase):
    def test_version_exits_zero_even_when_qdrant_imports_fail(self):
        # Copilot review (PR #372): the dependency import guard in
        # ingest_to_qdrant.py exit(1)s before parse_args() ever runs, so
        # --version must be answered ahead of it. Shadow both third-party
        # modules with ones that raise ImportError (PYTHONPATH wins over
        # site-packages) to simulate a half-broken install.
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as stub_dir:
            for name in ("mcp_server_qdrant", "qdrant_client"):
                os.makedirs(os.path.join(stub_dir, name))
                with open(os.path.join(stub_dir, name, "__init__.py"), "w", encoding="utf-8") as f:
                    f.write("raise ImportError('simulated broken dependency')\n")
            env = {**os.environ, "PYTHONPATH": stub_dir}
            result = subprocess.run(
                [sys.executable, os.path.join(REPO_ROOT, "tools", "ingest_to_qdrant.py"), "--version"],
                capture_output=True,
                text=True,
                env=env,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("claude-runway-ingest", result.stdout)


if __name__ == "__main__":
    unittest.main()
