#!/usr/bin/env python3
"""Tests for skills/my-resume/SKILL.md -- issue #71's proactive file pre-loading.

The skill is a markdown document of prose instructions, not a Python module, so
the test strategy is twofold:

  1. Content checks: assert that the SKILL.md contains the key behavioural
     requirements introduced in issue #71, the same way test_pyproject.py pins
     pyproject.toml content. A future editor who removes or rewrites the step
     will get an explicit failure here rather than a silent regression.

  2. Path-extraction and scope-filtering logic: the skill tells Claude to apply
     a regex-like heuristic to pull file paths from the 'Important files and
     locations' section, then skip any path that resolves outside the current
     project directory. Reference Python implementations of both heuristics live
     here so their boundary cases are testable without running Claude itself.

Stdlib-only (unittest, no pytest), no network, no LM Studio, no Qdrant.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import re
import tempfile
import unittest
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_PATH = os.path.join(REPO_ROOT, "skills", "my-resume", "SKILL.md")

# ---------------------------------------------------------------------------
# Reference implementation of the path-extraction heuristic described in the
# skill.  This mirrors what the skill instructs Claude to do so the boundary
# cases are testable without invoking Claude.
# ---------------------------------------------------------------------------

_FILE_EXT_RE = re.compile(
    r"\.(py|md|ts|tsx|js|jsx|json|yaml|yml|toml|txt|sh|cs|go|rb|sql)$",
    re.IGNORECASE,
)

# A path must contain a slash or start with a rooted prefix AND end in a
# known extension.  This keeps bare class/function names (no slash, no ext)
# from matching while still catching relative paths like `libs/foo.py`.
_PATH_CANDIDATES_RE = re.compile(
    r"""
    (?:
        /?[\w\-\.]+(?:/[\w\-\.]+)+   # relative: one-or-more slash-separated components
        | ~/[\w\-\./]+                # home-relative: ~/...
        | \./[\w\-\./]+               # explicit dot-relative: ./...
        | /[\w\-\./]+                 # absolute: /...
    )
    """,
    re.VERBOSE,
)


def is_in_project_scope(path: str, cwd: Optional[str] = None) -> bool:
    """
    Return True only when the canonically resolved path stays inside `cwd`
    (defaults to the real current working directory).

    A textual prefix check ("no leading /") is insufficient because:
    - Relative traversals like `dir/../../private/secrets.json` can escape.
    - In-tree symlinks can resolve to targets outside the project.

    Home-relative paths (starting with ~/) are also rejected before any
    resolution attempt, since os.path.join does not expand ~ and the literal
    string "~" would otherwise be treated as a subdirectory name, masking
    the out-of-scope nature of the intended path.

    This implementation mirrors what the skill instructs Claude to do:
    resolve the path to its real absolute form (following symlinks and
    collapsing `..` components) and verify the result starts with the
    resolved project root (with a trailing separator to prevent prefix
    collisions like /project-data matching a root of /project).
    """
    # Reject home-relative paths before any resolution: os.path.join does not
    # expand ~, so join(cwd, "~/.config/x") resolves as <cwd>/~/.config/x --
    # a path that textually starts with cwd but is clearly not the intended
    # ~/... target.  Expanding with os.path.expanduser first and then checking
    # would also work, but rejecting early is simpler and equally correct.
    if path.startswith("~/") or path == "~":
        return False
    if cwd is None:
        cwd = os.getcwd()
    resolved_cwd = os.path.realpath(cwd) + os.sep
    try:
        resolved_path = os.path.realpath(os.path.join(cwd, path))
    except (ValueError, OSError):
        return False
    return resolved_path.startswith(resolved_cwd)


def extract_important_file_paths(
    compact_text: str, max_paths: int = 5, cwd: Optional[str] = None
) -> list:
    """
    Extract up to `max_paths` in-project file paths from the
    '## Important files and locations' section of a /my-compact summary.

    Returns a list of unique, in-scope path strings in the order they first
    appeared.  Paths that resolve outside the project root are silently
    excluded -- see is_in_project_scope().
    """
    # Locate the section.
    section_match = re.search(
        r"^##\s+Important files and locations\s*$(.+?)(?=^##\s|\Z)",
        compact_text,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return []

    section_body = section_match.group(1)

    # Extract path-shaped tokens, apply canonical scope filter.
    seen: dict = {}
    for m in _PATH_CANDIDATES_RE.finditer(section_body):
        candidate = m.group(0).strip("`'\",. ")
        if not _FILE_EXT_RE.search(candidate):
            continue
        if not is_in_project_scope(candidate, cwd=cwd):
            continue
        if candidate not in seen:
            seen[candidate] = None
            if len(seen) >= max_paths:
                break

    return list(seen.keys())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class SkillFileExists(unittest.TestCase):
    """Guard against the file being renamed or deleted accidentally."""

    def test_skill_file_is_present(self):
        self.assertTrue(
            os.path.isfile(SKILL_PATH),
            f"skills/my-resume/SKILL.md not found at {SKILL_PATH}",
        )


class SkillContentRequirements(unittest.TestCase):
    """Pin the key behavioural requirements from issue #71 in the SKILL.md text."""

    def setUp(self):
        with open(SKILL_PATH, encoding="utf-8") as f:
            self.content = f.read()

    def test_skill_references_important_files_section(self):
        # The step must explicitly name the section it parses.
        self.assertIn(
            "Important files and locations",
            self.content,
            "SKILL.md must reference '## Important files and locations'",
        )

    def test_skill_instructs_read_without_asking(self):
        # The key behavioural change: Read immediately, do NOT ask.
        lower = self.content.lower()
        self.assertIn(
            "do not ask",
            lower,
            "SKILL.md must instruct Claude NOT to ask the user before calling Read",
        )

    def test_skill_caps_at_five_paths(self):
        # The 5-path cap prevents context bloat on large compacts.
        self.assertIn(
            "5",
            self.content,
            "SKILL.md must specify a cap of 5 paths",
        )

    def test_skill_instructs_silent_skip_on_missing_file(self):
        # Paths may no longer exist -- the skill must not report errors for them.
        self.assertIn(
            "silently",
            self.content.lower(),
            "SKILL.md must instruct Claude to skip unreadable paths silently",
        )

    def test_skill_instructs_read_call(self):
        # Step must explicitly say to call Read.
        self.assertIn(
            "Read",
            self.content,
            "SKILL.md must tell Claude to call `Read`",
        )

    def test_skill_restricts_to_in_project_paths(self):
        # Security requirement: only relative paths (inside the project) should
        # be auto-read.  Absolute (/...) and home-relative (~/) paths must be
        # excluded to prevent loading sensitive files like ~/.claude/settings.json
        # without user confirmation.  The SKILL.md must document this constraint.
        lower = self.content.lower()
        self.assertTrue(
            "relative" in lower or "current project" in lower or "current working" in lower,
            "SKILL.md must restrict auto-Read to paths inside the current project",
        )
        # Also verify that absolute/home-relative paths are explicitly called out.
        self.assertTrue(
            "absolute" in lower or "~/" in self.content,
            "SKILL.md must document that absolute or home-relative paths are skipped",
        )

    def test_invite_continuation_still_present(self):
        # The original step 5 invitation must still exist (renumbered to step 6).
        self.assertIn(
            "restored the context from your previous session",
            self.content,
            "SKILL.md must still include the continuation invitation",
        )


class ProjectScopeFilter(unittest.TestCase):
    """
    Exercises is_in_project_scope() -- the canonical scope filter the skill
    prescribes (Copilot round 1 and round 2 findings).

    Uses a real temp directory so os.path.realpath can follow the actual
    filesystem -- required to verify path-traversal cases correctly.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "libs"), exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_relative_path_is_in_scope(self):
        self.assertTrue(is_in_project_scope("libs/local_compress_lib.py", cwd=self.tmp))

    def test_explicit_dot_relative_is_in_scope(self):
        self.assertTrue(is_in_project_scope("./libs/ingest.py", cwd=self.tmp))

    def test_absolute_path_outside_project_is_out_of_scope(self):
        # An absolute path to a location outside the project must be rejected.
        outside = tempfile.gettempdir() + "/secret.json"
        self.assertFalse(is_in_project_scope(outside, cwd=self.tmp))

    def test_home_relative_path_is_out_of_scope(self):
        # ~/... expands to the home directory, outside the project.
        self.assertFalse(is_in_project_scope("~/.claude/settings.json", cwd=self.tmp))

    def test_dotdot_traversal_is_out_of_scope(self):
        # A traversal that escapes the project root via .. must be rejected.
        # This is what a textual prefix check ("no leading /") would miss.
        self.assertFalse(
            is_in_project_scope("libs/../../etc/passwd", cwd=self.tmp),
            "Path traversal via .. must be blocked by canonical resolution",
        )

    def test_deep_relative_path_inside_project_is_in_scope(self):
        nested = os.path.join(self.tmp, "a", "b")
        os.makedirs(nested, exist_ok=True)
        self.assertTrue(is_in_project_scope("a/b/file.py", cwd=self.tmp))

    def test_prefix_collision_guarded_by_separator(self):
        # A sibling directory whose name begins with the same prefix as the
        # project root must not be treated as in-scope.  The trailing separator
        # in resolved_cwd prevents /project-data from matching /project.
        sibling = self.tmp + "-sibling"
        os.makedirs(sibling, exist_ok=True)
        try:
            self.assertFalse(
                is_in_project_scope(sibling + "/secret.py", cwd=self.tmp),
                "A sibling directory with a shared prefix must not match as in-scope",
            )
        finally:
            import shutil
            shutil.rmtree(sibling, ignore_errors=True)


class PathExtractionHeuristic(unittest.TestCase):
    """
    Exercises extract_important_file_paths() -- the reference Python
    implementation of the combined path-extraction + scope-filter heuristic.
    """

    def _compact(self, section_body: str) -> str:
        """Wrap section_body in a minimal compact structure."""
        return (
            "PROJECT: test\n"
            "DATE: 2026-01-01\n"
            "\n"
            "## What we were working on\n"
            "Some work.\n"
            "\n"
            "## Important files and locations\n"
            f"{section_body}\n"
            "\n"
            "## Unresolved questions\n"
            "None.\n"
        )

    def test_extracts_relative_paths_with_extension(self):
        body = "- `libs/local_compress_lib.py` — compress logic\n- `tools/compress_mcp_server.py`"
        paths = extract_important_file_paths(self._compact(body))
        self.assertIn("libs/local_compress_lib.py", paths)
        self.assertIn("tools/compress_mcp_server.py", paths)

    def test_skips_absolute_paths(self):
        # Absolute paths are out-of-scope -- the scope filter must remove them.
        body = "- `/home/user/tools/claude-runway/libs/qdrant_ingest_lib.py`"
        paths = extract_important_file_paths(self._compact(body))
        self.assertEqual(paths, [], f"Absolute paths must be excluded, got: {paths}")

    def test_skips_home_relative_paths(self):
        # ~/... paths are out-of-scope.
        body = "- `~/.claude/settings.json`\n- `libs/savings_ledger.py`"
        paths = extract_important_file_paths(self._compact(body))
        self.assertNotIn("~/.claude/settings.json", paths)
        # The relative path from the same body should still be extracted.
        self.assertIn("libs/savings_ledger.py", paths)

    def test_extracts_skill_paths(self):
        body = "- `skills/my-resume/SKILL.md`\n- `skills/my-compact/SKILL.md`"
        paths = extract_important_file_paths(self._compact(body))
        self.assertIn("skills/my-resume/SKILL.md", paths)
        self.assertIn("skills/my-compact/SKILL.md", paths)

    def test_does_not_extract_bare_symbol_names(self):
        # Class names and function names without a path separator must be ignored.
        body = "- `LocalCompressLib` class\n- `section_is_verbatim` function\n- `libs/qdrant_retry.py`"
        paths = extract_important_file_paths(self._compact(body))
        self.assertNotIn("LocalCompressLib", paths)
        self.assertNotIn("section_is_verbatim", paths)
        self.assertIn("libs/qdrant_retry.py", paths)

    def test_caps_at_five_paths(self):
        lines = "\n".join(
            f"- `libs/file{i}.py`" for i in range(10)
        )
        paths = extract_important_file_paths(self._compact(lines))
        self.assertLessEqual(len(paths), 5)

    def test_returns_empty_when_section_missing(self):
        compact_no_section = (
            "PROJECT: test\n"
            "## What we were working on\n"
            "Some work.\n"
        )
        self.assertEqual(extract_important_file_paths(compact_no_section), [])

    def test_returns_empty_for_empty_section(self):
        paths = extract_important_file_paths(self._compact(""))
        self.assertEqual(paths, [])

    def test_deduplicates_repeated_paths(self):
        body = "- `libs/local_compress_lib.py`\n- `libs/local_compress_lib.py`"
        paths = extract_important_file_paths(self._compact(body))
        self.assertEqual(paths.count("libs/local_compress_lib.py"), 1)

    def test_extracts_yaml_and_toml_paths(self):
        # Include a clear slash-separated path with a known extension.
        # pyproject.toml has no slash so the reference impl skips it (it
        # requires at least one slash for relative paths).  templates/mcp.json
        # has both a slash AND a known .json extension so it is extracted.
        body = "- `pyproject.toml`\n- `templates/mcp.json`\n- `hooks/compress_bash_output.py`"
        paths = extract_important_file_paths(self._compact(body))
        self.assertTrue(
            any("templates/mcp.json" in p for p in paths),
            f"Expected templates/mcp.json in {paths}",
        )
        self.assertTrue(
            any("compress_bash_output.py" in p for p in paths),
            f"Expected hooks/compress_bash_output.py in {paths}",
        )

    def test_preserves_order_of_appearance(self):
        body = "- `libs/a.py`\n- `tools/b.py`\n- `hooks/c.py`"
        paths = extract_important_file_paths(self._compact(body))
        self.assertEqual(paths, ["libs/a.py", "tools/b.py", "hooks/c.py"])

    def test_mixed_scope_paths(self):
        # A section with both in-scope and out-of-scope paths: only relative ones returned.
        body = (
            "- `/etc/config.json`\n"
            "- `libs/cache_db.py`\n"
            "- `~/.claude/settings.json`\n"
            "- `tools/doctor.py`\n"
        )
        paths = extract_important_file_paths(self._compact(body))
        self.assertIn("libs/cache_db.py", paths)
        self.assertIn("tools/doctor.py", paths)
        self.assertNotIn("/etc/config.json", paths)
        self.assertNotIn("~/.claude/settings.json", paths)


if __name__ == "__main__":
    unittest.main()
