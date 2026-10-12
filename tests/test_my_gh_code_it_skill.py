#!/usr/bin/env python3
"""Tests for .claude/skills/my-gh-code-it/SKILL.md and .claude/skills/my-gh-pr/SKILL.md
and docs/development-workflow.md -- issue #286's port of the bin-vs-Scripts venv-path
resolver (issue #230's defect class, originally fixed only for /my-gh-autowork).

Those three documents are prose instructions, not Python modules, so this mirrors
tests/test_my_gh_autowork_skill.py's content-pinning strategy: assert the resolver's
key strings remain present and that no bare `.venv/bin/python` invocation survives in
the executable command block, so a future edit that reintroduces the hardcoded POSIX
path -- which fails with command-not-found on a Windows checkout, the platform this
repo is actually dogfooded on -- gets an explicit failure here instead of a silent
regression. The resolver's own shell logic is also re-implemented in Python and
exercised directly, the same dual strategy test_my_gh_autowork_skill.py uses.

Stdlib-only (unittest, no pytest), no network, no LM Studio, no Qdrant.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_IT_PATH = os.path.join(REPO_ROOT, ".claude", "skills", "my-gh-code-it", "SKILL.md")
PR_PATH = os.path.join(REPO_ROOT, ".claude", "skills", "my-gh-pr", "SKILL.md")
DEV_WORKFLOW_PATH = os.path.join(REPO_ROOT, "docs", "development-workflow.md")

# The exact resolver line shared by the two documents that actually embed it --
# my-gh-code-it's SKILL.md (step 13) and docs/development-workflow.md (issue #286).
# my-gh-pr's SKILL.md deliberately carries only platform-neutral wording (it names both
# venv layouts rather than embedding the resolver), so it is pinned by
# PrValidateWordingRequirements below, NOT by RESOLVER_LINE. Matches the form already
# pinned for /my-gh-autowork in tests/test_my_gh_autowork_skill.py's
# WorktreeVenvInstallContentRequirements neighborhood.
RESOLVER_LINE = (
    "PYTHON=$(if [ -f .venv/Scripts/python ]; then echo .venv/Scripts/python; "
    "else echo .venv/bin/python; fi)"
)


def resolve_python(scripts_exists):
    """Reference implementation of the shell resolver's branch logic: returns the
    interpreter path the `if [ -f .venv/Scripts/python ]` test selects, given whether
    that Windows path exists. Mirrors test_my_gh_autowork_skill.py's practice of
    exercising a pinned shell snippet's logic directly, not just its text presence."""
    return ".venv/Scripts/python" if scripts_exists else ".venv/bin/python"


class SkillFilesExist(unittest.TestCase):
    """Guard against any of the three documents being renamed or deleted."""

    def test_code_it_skill_is_present(self):
        self.assertTrue(
            os.path.isfile(CODE_IT_PATH),
            f".claude/skills/my-gh-code-it/SKILL.md not found at {CODE_IT_PATH}",
        )

    def test_pr_skill_is_present(self):
        self.assertTrue(
            os.path.isfile(PR_PATH),
            f".claude/skills/my-gh-pr/SKILL.md not found at {PR_PATH}",
        )

    def test_dev_workflow_doc_is_present(self):
        self.assertTrue(
            os.path.isfile(DEV_WORKFLOW_PATH),
            f"docs/development-workflow.md not found at {DEV_WORKFLOW_PATH}",
        )


class CodeItResolverRequirements(unittest.TestCase):
    """Pin issue #286's resolver in /my-gh-code-it's step 13."""

    def setUp(self):
        with open(CODE_IT_PATH, encoding="utf-8") as f:
            self.content = f.read()

    def test_has_bin_vs_scripts_resolver(self):
        self.assertIn(RESOLVER_LINE, self.content)

    def test_step_13_command_block_has_no_bare_bin_python_invocation(self):
        # The fenced ```bash block in step 13 must run tools through $PYTHON, never
        # a hardcoded `.venv/bin/python -m ...` (which is the Windows-breaking form).
        # The resolver line itself legitimately contains the literal `.venv/bin/python`
        # as its else-branch value, so strip that one line out before checking.
        blocks = re.findall(r"```bash\n(.*?)```", self.content, re.DOTALL)
        joined = "\n".join(blocks)
        joined_without_resolver = joined.replace(RESOLVER_LINE, "")
        self.assertNotIn(
            ".venv/bin/python -m",
            joined_without_resolver,
            "a hardcoded `.venv/bin/python -m ...` invocation survives in a command block",
        )
        self.assertNotIn(
            ".venv/bin/ruff",
            joined_without_resolver,
            "a hardcoded `.venv/bin/ruff` invocation survives in a command block",
        )
        self.assertNotIn(
            ".venv/bin/pyright",
            joined_without_resolver,
            "a hardcoded `.venv/bin/pyright` invocation survives in a command block",
        )

    def test_tools_invoked_through_resolved_python(self):
        # ruff/pyright run as `$PYTHON -m ...`, matching autowork's single-interpreter
        # approach (so only one allow rule, not three, is needed per platform).
        self.assertIn("$PYTHON -m unittest discover -s tests", self.content)
        self.assertIn("$PYTHON -m ruff check libs tools hooks tests", self.content)
        self.assertIn("$PYTHON -m pyright", self.content)

    def test_pyright_pythonpath_points_at_resolved_python(self):
        # --pythonpath must point at $PYTHON, not a hardcoded `.venv/bin/python`.
        self.assertIn('--pythonpath "$PYTHON"', self.content)
        self.assertNotIn("--pythonpath .venv/bin/python", self.content)

    def test_recovery_install_uses_resolved_python(self):
        # The requirements-dev.txt recovery install must also target $PYTHON, not a
        # hardcoded path -- the worktree-wrong-venv reason it carries `--python` at
        # all is the same reason it must resolve the path per platform.
        self.assertIn('uv pip install --python "$PYTHON"', self.content)
        self.assertNotIn("--python .venv/bin/python", self.content)

    def test_documents_the_windows_failure_mode(self):
        # The "why" comment must survive so a future reader understands the resolver
        # isn't gratuitous -- it's the issue #286 / #230 Windows command-not-found fix.
        self.assertIn("#286", self.content)
        self.assertIn(".venv/Scripts/", self.content)


class PrValidateWordingRequirements(unittest.TestCase):
    """Pin /my-gh-pr's validate-artifact wording as platform-neutral (issue #286)."""

    def setUp(self):
        with open(PR_PATH, encoding="utf-8") as f:
            self.content = f.read()

    def test_no_hardcoded_bin_python_unittest_command(self):
        # The old wording hardcoded `.venv/bin/python -m unittest discover -s tests`,
        # which reads as a copy-pasteable command that fails on Windows.
        self.assertNotIn(".venv/bin/python -m unittest", self.content)

    def test_mentions_both_platform_paths(self):
        # The wording must acknowledge both venv layouts rather than privileging one.
        self.assertIn(".venv/Scripts/python", self.content)
        self.assertIn(".venv/bin/python", self.content)
        self.assertIn("#286", self.content)


class DevWorkflowResolverRequirements(unittest.TestCase):
    """Pin docs/development-workflow.md's linting/type-checking resolver (issue #286)."""

    def setUp(self):
        with open(DEV_WORKFLOW_PATH, encoding="utf-8") as f:
            self.content = f.read()

    def test_has_bin_vs_scripts_resolver(self):
        self.assertIn(RESOLVER_LINE, self.content)

    def test_no_hardcoded_bin_ruff_or_pyright_commands(self):
        # The old instructions hardcoded `.venv/bin/ruff check ...` and
        # `.venv/bin/pyright --pythonpath .venv/bin/python`.
        self.assertNotIn(".venv/bin/ruff", self.content)
        self.assertNotIn(".venv/bin/pyright", self.content)
        self.assertNotIn("--pythonpath .venv/bin/python", self.content)

    def test_checks_run_through_resolved_python(self):
        self.assertIn("$PYTHON -m ruff check libs tools hooks tests", self.content)
        self.assertIn('$PYTHON -m pyright --pythonpath "$PYTHON"', self.content)
        self.assertIn("#286", self.content)


class ResolverLogic(unittest.TestCase):
    """Exercise the shell resolver's branch logic directly (not just its text)."""

    def test_windows_branch_selects_scripts(self):
        self.assertEqual(resolve_python(scripts_exists=True), ".venv/Scripts/python")

    def test_posix_branch_selects_bin(self):
        self.assertEqual(resolve_python(scripts_exists=False), ".venv/bin/python")


if __name__ == "__main__":
    unittest.main()
