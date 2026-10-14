"""Guard: tracked files must not cite backlog tickets or pull requests.

A reader installing, using or maintaining the tool gets nothing from
"fixed in #374" or "see PR 107", and such references go stale when a ticket
closes or is renumbered. Every tracked Markdown file (README, docs/,
EVALUATION.md, skills/, .claude/skills/, .github/) and every other tracked
text file (code, comments, docstrings, templates, CI and packaging config) is
therefore kept free of them, and this test fails if one creeps back in.
State the fact and the reason directly instead.

What counts as a reference (the rules are intentionally narrow):

  * ``#123`` -- a "#" directly followed by digits (``Fixes #<n>`` with a
    placeholder is fine, as are headings like "# 1. Setup" with a space, anchors
    like ``#step-1``, and ``&#123;`` HTML entities). A bare hex colour such as
    ``#000000`` would be flagged too; none exist in the docs, and writing one
    in a doc would mean rewording it.
  * ``issue 12``, ``issues #12``, ``PR 12``, ``PRs 12``, ``pull request 12``,
    ``ticket 12`` -- the word followed by a number, with an optional "#".
  * ``owner/repo#12`` (covered by ``#123``).
  * ``/issues/123`` and ``/pull/123`` URL paths. API paths with a placeholder
    (``/issues/<N>/comments``) or without a number (``/issues/comments/<id>``)
    are not references and are allowed.

Excluded: ``.plans/`` (gitignored, never shipped), and the fixture data
listed in ``CODE_FIXTURE_ALLOWLIST`` below (numbers that are test input, not
citations).
"""

import os
import re
import subprocess
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories skipped when falling back to a filesystem walk (no git available).
# `git ls-files` already excludes all of these, being untracked or ignored.
_SKIP_DIRS = {".git", ".venv", ".plans", "node_modules", "__pycache__", "worktrees"}

TICKET_REF = re.compile(
    r"""
      (?<!&)\#\d+                                   # #123 (not the &#123; entity)
    | \b(?:issues?|PRs?|pull\ requests?|tickets?)   # issue 12 / PR #12 / ticket 12
        \s*\#?\s*\d+
    | /(?:issues|pull)/\d+                          # .../issues/12, .../pull/12
    """,
    re.IGNORECASE | re.VERBOSE,
)


def tracked_markdown_files(root: str = REPO_ROOT) -> list[str]:
    """Repo-relative paths of every tracked ``*.md`` file, minus ``.plans/``."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.md"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
        paths = [p for p in out.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        # Not a git checkout (e.g. a source tarball): walk the tree instead.
        paths = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                if name.endswith(".md"):
                    rel = os.path.relpath(os.path.join(dirpath, name), root)
                    paths.append(rel.replace(os.sep, "/"))
    return sorted(p for p in paths if not p.startswith(".plans/"))


def find_ticket_refs(text: str) -> list[tuple[int, str]]:
    """``(line number, matched text)`` for every ticket/PR reference in ``text``."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in TICKET_REF.finditer(line):
            hits.append((lineno, m.group(0)))
    return hits


class TicketRefPatternTests(unittest.TestCase):
    """Pin the matching rules so they cannot drift silently in either direction."""

    def test_flags_reference_forms(self):
        for sample in [
            "fixed in #374",
            "(issue #221)",
            "see issue 12 for detail",
            "issues #22/#23",
            "Copilot review, PR #319",
            "pull request 7",
            "PR 21",
            "owner/repo#52089",
            "anthropics/claude-code#52089",
            "[#205](https://github.com/Donelle/claude-runway/issues/205)",
            "https://github.com/Donelle/claude-runway/pull/12",
            "tracked in ticket 9",
        ]:
            with self.subTest(sample=sample):
                self.assertTrue(find_ticket_refs(sample), sample)

    def test_allows_ordinary_hash_and_number_use(self):
        for sample in [
            "# Heading",
            "## 1. Setup",
            "[Known limitations](#known-limitations)",
            "[step](#step-1)",
            "1. first item",
            "&#123; entity",
            "Fixes #<issue-number>",
            "/my-gh-pr <issue-number>",
            "https://github.com/Donelle/claude-runway/issues/<N>",
            "gh api repos/OWNER/REPO/issues/comments/<id> --method DELETE",
            "gh api repos/OWNER/REPO/issues/<PR>/comments",
            "issue-<n>-<HHMMSS> marker",
            "the issue tracker has 3 columns",
            "a PR with 5 files",
            "step 4 of 6",
        ]:
            with self.subTest(sample=sample):
                self.assertEqual(find_ticket_refs(sample), [], sample)


class NoTicketRefsInMarkdownTests(unittest.TestCase):
    def test_tracked_markdown_found(self):
        files = tracked_markdown_files()
        self.assertIn("README.md", files)
        self.assertFalse([f for f in files if f.startswith(".plans/")])

    def test_no_ticket_or_pr_references_in_tracked_markdown(self):
        offenders = []
        for rel in tracked_markdown_files():
            path = os.path.join(REPO_ROOT, rel)
            if not os.path.isfile(path):  # tracked but deleted in the working tree
                continue
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for lineno, match in find_ticket_refs(text):
                offenders.append(f"{rel}:{lineno}: {match!r}")
        self.assertEqual(
            offenders,
            [],
            "Markdown must not cite backlog tickets or PRs; state the fact directly "
            "instead (see this test's docstring for what counts):\n" + "\n".join(offenders),
        )


# Non-Markdown files whose matches are test input rather than citations. A value
# of None allows every match in the file; a set allows only those exact strings.
CODE_FIXTURE_ALLOWLIST: dict[str, set[str] | None] = {
    # this file's own sample strings
    "tests/test_docs_no_ticket_refs.py": None,
    # prompts and descriptions fed to the autowork cost matcher
    "tests/test_autowork_cost_lib.py": None,
    # an identifier inside a compression fixture
    "tests/test_local_compress_lib.py": {"PR 481200"},
}


def tracked_code_files(root: str = REPO_ROOT) -> list[str]:
    """Repo-relative paths of every tracked non-Markdown file, minus ``.plans/``."""
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
        paths = [p for p in out.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        paths = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                paths.append(rel.replace(os.sep, "/"))
    return sorted(p for p in paths if not p.endswith(".md") and not p.startswith(".plans/"))


class NoTicketRefsInCodeTests(unittest.TestCase):
    def test_tracked_code_found(self):
        files = tracked_code_files()
        self.assertIn("libs/savings_ledger.py", files)
        self.assertFalse([f for f in files if f.endswith(".md")])

    def test_no_ticket_or_pr_references_in_tracked_code(self):
        offenders = []
        for rel in tracked_code_files():
            allowed = CODE_FIXTURE_ALLOWLIST.get(rel, set())
            if allowed is None:
                continue
            path = os.path.join(REPO_ROOT, rel)
            if not os.path.isfile(path):  # tracked but deleted in the working tree
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:  # binary file
                continue
            for lineno, match in find_ticket_refs(text):
                if match not in allowed:
                    offenders.append(f"{rel}:{lineno}: {match!r}")
        self.assertEqual(
            offenders,
            [],
            "Code, comments, docstrings and config must not cite backlog tickets or PRs; "
            "state the fact directly instead (see this test's docstring for what counts):\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
