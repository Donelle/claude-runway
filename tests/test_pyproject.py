#!/usr/bin/env python3
"""Tests for pyproject.toml/setup.py (issues #50, #206) -- pins the 3
console-script entry points the issue actually names (ingest_to_qdrant.py,
setup_project.py, doctor.py), confirms the `claude_runway` shim package
they now route through is declared and present, and builds a REAL wheel to
confirm libs/tools/hooks/templates/skills land under
<wheel>.data/data/src/{libs,tools,hooks,templates,skills} -- issue #206's
whole point, replacing the old top-level `[tool.setuptools] packages`
layout that put these five directly in site-packages.

setup.py's data_files() is real Python code that runs at build time (a
glob, not a hand-typed list -- see that file's own docstring for why), so
it can't be pinned by only parsing pyproject.toml's static TOML tables the
way the old version of this file did; the wheel-content tests below
actually build a wheel and inspect it, which is the only thing that can
catch setup.py itself regressing (e.g. back to a single shared "src/skills"
target, which silently drops every skill but one -- confirmed live during
this issue's own development, see setup.py's docstring).

Stdlib-only for the TOML-parsing tests (tomllib, Python 3.11+) -- skips
itself on an older interpreter rather than adding a new third-party
TOML-parsing dependency just for this one test. The wheel-building tests
need `build`/`wheel`/`setuptools` (requirements-dev.txt) and are skipped
(not failed) if those aren't importable, so a bare `requirements.txt`-only
environment doesn't spuriously fail unittest discovery.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPROJECT_PATH = os.path.join(REPO_ROOT, "pyproject.toml")

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

try:
    import build as _build_pkg  # noqa: F401
    import wheel as _wheel_pkg  # noqa: F401

    _BUILD_TOOLING_AVAILABLE = True
except ImportError:
    _BUILD_TOOLING_AVAILABLE = False


@unittest.skipIf(tomllib is None, "tomllib requires Python 3.11+")
class PyprojectTomlIsWellFormed(unittest.TestCase):
    def setUp(self):
        with open(PYPROJECT_PATH, "rb") as f:
            self.data = tomllib.load(f)

    def test_declares_the_three_named_console_scripts_via_the_shim(self):
        # Issue #206: these no longer point straight at tools.*:main --
        # tools/ isn't an importable site-packages package anymore (it
        # ships as wheel data under <env>/src/tools instead), so each
        # entry point routes through the claude_runway shim, which puts
        # <env>/src on sys.path before calling the real tools/*.py main().
        scripts = self.data["project"]["scripts"]
        self.assertEqual(scripts.get("claude-runway-ingest"), "claude_runway:ingest_main")
        self.assertEqual(scripts.get("claude-runway-setup"), "claude_runway:setup_main")
        self.assertEqual(scripts.get("claude-runway-doctor"), "claude_runway:doctor_main")

    def test_claude_runway_shim_is_the_only_declared_package(self):
        # Issue #206: libs/tools/templates/hooks/skills moved OUT of
        # `[tool.setuptools] packages` (previously all five, plus this one
        # implicitly via site-packages, were top-level packages -- the
        # generic-namespace layout this issue replaces). claude_runway is
        # the one real, importable site-packages package left -- it exists
        # purely to bridge [project.scripts] to the data-files layout (see
        # claude_runway/__init__.py's own docstring).
        packages = set(self.data["tool"]["setuptools"]["packages"])
        self.assertEqual(packages, {"claude_runway"})

    def test_no_stale_package_data_section(self):
        # templates/*.template and skills/*/SKILL.md used to need
        # `[tool.setuptools.package-data]` globs because they were bundled
        # INSIDE a declared package. Issue #206 ships them as data_files
        # instead (see setup.py) -- package-data has nothing left to do and
        # its presence would be a stale, confusing leftover of the old
        # layout, so pin its absence rather than silently letting it
        # reappear unnoticed.
        self.assertNotIn("package-data", self.data.get("tool", {}).get("setuptools", {}))

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


def _build_wheel() -> Optional[str]:
    """Builds a real wheel via `python -m build --wheel --no-isolation` into
    a fresh temp dir, returning the .whl path. Reused across every test in
    BuiltWheelHasExpectedDataLayout via setUpClass so the (multi-second)
    build only runs once per test session, not once per test method."""
    out_dir = tempfile.mkdtemp(prefix="claude_runway_wheel_test_")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", out_dir],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    whls = [f for f in os.listdir(out_dir) if f.endswith(".whl")]
    assert len(whls) == 1, f"expected exactly one wheel in {out_dir}, found {whls}"
    return os.path.join(out_dir, whls[0])


@unittest.skipUnless(
    _BUILD_TOOLING_AVAILABLE,
    "build/wheel not installed -- run `uv pip install -r requirements-dev.txt` to enable this test",
)
class BuiltWheelHasExpectedDataLayout(unittest.TestCase):
    """Issue #206's core acceptance criterion: a real built wheel must ship
    libs/tools/hooks/templates/skills as DATA files landing under
    <env>/src/{libs,tools,hooks,templates,skills} at install time (wheel
    spec: the "data" category installs relative to sys.prefix) -- not as
    site-packages packages. Confirmed live (issue #206's own research) that
    this "data" category's install-to-sys.prefix rule is a wheel-spec
    guarantee, not OS-specific, so inspecting the wheel's own RECORD/namelist
    here is a legitimate stand-in for actually installing it on each of
    Windows/macOS/Linux (the CI matrix in .github/workflows/tests.yml's
    packaging-smoke job covers the real per-OS install+run separately)."""

    @classmethod
    def setUpClass(cls):
        cls.whl_path = _build_wheel()
        with zipfile.ZipFile(cls.whl_path) as z:
            cls.names = z.namelist()

    @classmethod
    def tearDownClass(cls):
        if cls.whl_path and os.path.exists(cls.whl_path):
            os.remove(cls.whl_path)

    def _data_src_names(self) -> list:
        # The wheel's data category directory is named
        # "<dist-name>-<version>.data/data/..." -- match on the
        # "/data/src/" suffix rather than hardcoding the version, so this
        # doesn't need updating every version bump.
        return [n for n in self.names if ".data/data/src/" in n]

    def test_claude_runway_shim_ships_as_a_real_package_not_data(self):
        self.assertIn("claude_runway/__init__.py", self.names)

    def test_libs_tools_hooks_templates_ship_under_env_src(self):
        data_names = self._data_src_names()
        for directory, filename in (
            ("libs", "setup_project_lib.py"),
            ("tools", "setup_project.py"),
            ("tools", "doctor.py"),
            ("tools", "ingest_to_qdrant.py"),
            ("hooks", "compress_bash_output.py"),
            ("hooks", "redirect_webfetch_to_fetch_url.py"),
            ("hooks", "record_session_id.py"),
            ("templates", "mcp.json.template"),
            ("templates", "settings.json.template"),
            ("templates", "CLAUDE.md.template"),
        ):
            with self.subTest(directory=directory, filename=filename):
                self.assertTrue(
                    any(n.endswith(f"/data/src/{directory}/{filename}") for n in data_names),
                    f"{directory}/{filename} not found under any .data/data/src/{directory}/ in the built wheel",
                )

    def test_memory_bank_files_are_present_in_the_built_wheel(self):
        # Issue #182's guarantee (the memory-bank MCP server must ship the
        # same way ingest_mcp_server.py/compress_mcp_server.py do) still
        # holds under the new data_files layout -- confirm directly against
        # the real wheel rather than just the files existing on disk, which
        # wouldn't catch setup.py's glob failing to pick them up.
        data_names = self._data_src_names()
        for rel_path in (
            "tools/memory_bank_mcp_server.py",
            "libs/memory_bank_lib.py",
            "libs/memory_events_lib.py",
        ):
            with self.subTest(rel_path=rel_path):
                self.assertTrue(
                    any(n.endswith(f"/data/src/{rel_path}") for n in data_names),
                    f"{rel_path} missing from the built wheel's data files",
                )

    def test_every_skill_subdirectory_ships_distinctly_not_collapsed(self):
        # Regression guard for the exact bug found during this issue's own
        # development: a single shared "src/skills" data_files target
        # copies every matched SKILL.md into the SAME target directory by
        # basename alone, silently collapsing every skill but the
        # alphabetically-last one into one file. setup.py instead emits one
        # data_files tuple per skills/<name>/ subdirectory -- confirm every
        # real skill directory on disk actually made it into the wheel
        # DISTINCTLY (own subdirectory, own SKILL.md), not just that "a"
        # SKILL.md exists somewhere.
        skills_dir = os.path.join(REPO_ROOT, "skills")
        real_skill_names = sorted(
            name for name in os.listdir(skills_dir) if os.path.isdir(os.path.join(skills_dir, name))
        )
        self.assertGreater(len(real_skill_names), 1, "expected more than one skill on disk to make this test meaningful")
        data_names = self._data_src_names()
        for name in real_skill_names:
            with self.subTest(skill=name):
                self.assertTrue(
                    any(n.endswith(f"/data/src/skills/{name}/SKILL.md") for n in data_names),
                    f"skills/{name}/SKILL.md missing from the built wheel (or collapsed into another skill's entry)",
                )


if __name__ == "__main__":
    unittest.main()
