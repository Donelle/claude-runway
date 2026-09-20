#!/usr/bin/env python3
"""Tests for pyproject.toml (issue #50) -- pins the 3 console-script entry
points the issue actually names (ingest_to_qdrant.py, setup_project.py,
doctor.py) so a future edit can't silently drop one, and confirms the
declared packages/package-data match the flat libs/tools/templates/hooks
layout every script's own `sys.path`/`Path(__file__)` arithmetic assumes
(see pyproject.toml's own `[tool.setuptools]` comments for why that
arithmetic needs those four directories to stay siblings under
site-packages -- none of the four has an `__init__.py`; that same comment
block explains why one was tried and then deliberately removed).

Stdlib-only (tomllib, Python 3.11+) -- skips itself on an older interpreter
rather than adding a new third-party TOML-parsing dependency just for this
one test:

    .venv/bin/python -m unittest discover -s tests
"""

import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPROJECT_PATH = os.path.join(REPO_ROOT, "pyproject.toml")

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None


@unittest.skipIf(tomllib is None, "tomllib requires Python 3.11+")
class PyprojectTomlIsWellFormed(unittest.TestCase):
    def setUp(self):
        with open(PYPROJECT_PATH, "rb") as f:
            self.data = tomllib.load(f)

    def test_declares_the_three_named_console_scripts(self):
        # The issue's own proposal text names exactly these three tools --
        # not the MCP servers, which stay path-invoked from a project's own
        # .mcp.json (a separate concern this issue doesn't cover).
        scripts = self.data["project"]["scripts"]
        self.assertEqual(scripts.get("claude-runway-ingest"), "tools.ingest_to_qdrant:main")
        self.assertEqual(scripts.get("claude-runway-setup"), "tools.setup_project:main")
        self.assertEqual(scripts.get("claude-runway-doctor"), "tools.doctor:main")

    def test_packages_include_all_four_sibling_directories(self):
        # libs/tools/templates/hooks must stay siblings under site-packages
        # for the __file__-relative path arithmetic in tools/*.py and
        # hooks/*.py to keep resolving correctly post-install -- see
        # pyproject.toml's own [tool.setuptools] comment block for the full
        # reasoning (including why none of the four has an __init__.py).
        packages = set(self.data["tool"]["setuptools"]["packages"])
        self.assertEqual(packages, {"libs", "tools", "templates", "hooks"})

    def test_templates_glob_is_declared_as_package_data(self):
        # templates/*.template files aren't .py -- without this, they'd
        # silently be dropped from the built wheel even though the
        # "templates" package itself is declared above (confirmed by
        # building a wheel without this section during development).
        package_data = self.data["tool"]["setuptools"]["package-data"]
        self.assertIn("*.template", package_data.get("templates", []))

    def test_core_dependencies_match_requirements_txt_core_section(self):
        # Not a byte-for-byte diff against requirements.txt (that file has
        # its own comments/optional section this doesn't need to mirror
        # exactly) -- just pins that the same core packages
        # tools/ingest_to_qdrant.py actually imports are declared here too.
        deps = " ".join(self.data["project"]["dependencies"])
        for expected in ("mcp", "mcp-server-qdrant", "qdrant-client", "fastembed", "pydantic_settings"):
            self.assertIn(expected, deps)

    def test_local_compress_dependencies_are_base_not_optional(self):
        # PR #136 review (Copilot): claude-runway-setup's OWN default
        # behavior (no --qdrant-only) generates a .mcp.json that enables
        # tools/compress_mcp_server.py and its hooks, pointed at this same
        # install's interpreter -- confirmed live that leaving
        # openai/requests/trafilatura as an opt-in extra meant a plain
        # default `pipx install claude-runway` + `claude-runway-setup init`
        # (no flags) produced a config whose local-compress MCP server
        # crashes at import time (ModuleNotFoundError: trafilatura). These
        # 3 must be base dependencies, not behind a "compress" extra a
        # first-time user has no reason to know to ask for.
        deps = " ".join(self.data["project"]["dependencies"])
        for expected in ("openai", "requests", "trafilatura"):
            self.assertIn(expected, deps)
        self.assertNotIn("compress", self.data["project"].get("optional-dependencies", {}))

    def test_memory_bank_files_are_present_in_declared_packages(self):
        # Issue #182: the memory-bank MCP server (tools/memory_bank_mcp_server.py,
        # backed by libs/memory_bank_lib.py and libs/memory_events_lib.py --
        # added later than this file, in issue #175/PR #178) must ship in a
        # pip-installed package the same way ingest_mcp_server.py/
        # compress_mcp_server.py already do. setuptools packages by whole
        # DIRECTORY here (see the [tool.setuptools] comment block above),
        # not by an explicit file list, so there's no per-file declaration
        # that could omit these -- confirmed directly by building a real
        # wheel and inspecting its contents during this issue's own testing.
        # This test pins that fact against the files actually on disk, so a
        # future change to `packages` (e.g. narrowing it, or moving these
        # files out of tools/libs) can't silently drop the memory-bank
        # server from a real wheel again without failing here first.
        packages = set(self.data["tool"]["setuptools"]["packages"])
        for rel_path in (
            "tools/memory_bank_mcp_server.py",
            "libs/memory_bank_lib.py",
            "libs/memory_events_lib.py",
        ):
            package_dir, _, _filename = rel_path.partition("/")
            self.assertIn(package_dir, packages)
            self.assertTrue(
                os.path.isfile(os.path.join(REPO_ROOT, rel_path)),
                f"{rel_path} missing on disk",
            )

    def test_pathspec_is_base_not_optional(self):
        # PR #136 review (Copilot), second round: same class of finding as
        # the compress trio above, but security-relevant rather than just a
        # startup crash. libs/qdrant_ingest_lib.py's respect_gitignore=True
        # default silently falls back to "don't filter at all" (no error) if
        # pathspec isn't importable -- and --qdrant-api-key writes a real
        # credential into a project's .mcp.json that this repo's own docs
        # tell users to gitignore. Leaving pathspec an opt-in extra meant
        # claude-runway-ingest would silently NOT respect that gitignore
        # rule by default, indexing the credential into Qdrant -- and would
        # have made this pip-installed default *worse* than the existing
        # documented clone-based default (`uv pip install -r
        # requirements.txt`), which already includes pathspec.
        deps = " ".join(self.data["project"]["dependencies"])
        self.assertIn("pathspec", deps)
        self.assertNotIn("gitignore", self.data["project"].get("optional-dependencies", {}))


if __name__ == "__main__":
    unittest.main()
