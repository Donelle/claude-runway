#!/usr/bin/env python3
"""Tests for skills/cr-pr-feedback/SKILL.md, the shareable, GitHub-only PR
feedback skill.

A skill has no unit test of its behaviour, so these pin what can be checked
mechanically:

  - the text is generic: it names no repository or organisation (the repo
    comes from the checkout at run time), and no reply template @-mentions
    the reviewer bot;
  - its fetch step lists the four commands in the order a later transcript
    reader relies on, and documents the marker line it prints;
  - the skill's `name:` matches its directory;
  - `init --install-skills` installs the REAL shipped skill, and `upgrade`
    refreshes a modified installed copy but does not install it where it is
    absent (the existing rule), against temp directories only.

The wheel containing the skill and the docs listing it are covered by
tests/test_pyproject.py and tests/test_docs_coverage.py, which iterate every
directory in skills/.

Stdlib-only, no network:

    .venv/bin/python -m unittest discover -s tests
"""

import argparse
import importlib.util
import io
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "cr-pr-feedback"
SKILL_PATH = SKILL_DIR / "SKILL.md"


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _load_setup_project():
    # Same by-file-path load tests/test_setup_project_cli.py uses: tools/ is
    # not importable by name.
    spec = importlib.util.spec_from_file_location("setup_project_pr_feedback", REPO_ROOT / "tools" / "setup_project.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Never prompt, even when a developer runs this suite from a terminal.
    mod._stdin_is_interactive = lambda: False
    return mod


class SkillIsGeneric(unittest.TestCase):
    def test_skill_file_exists(self):
        self.assertTrue(SKILL_PATH.is_file(), f"{SKILL_PATH} not found")

    def test_name_matches_directory(self):
        match = re.match(r"---\n(.*?)\n---\n", _skill_text(), re.DOTALL)
        self.assertIsNotNone(match, "no frontmatter block")
        names = re.findall(r"^name:\s*(\S+)\s*$", match.group(1), re.MULTILINE)
        self.assertEqual(names, [SKILL_DIR.name])

    def test_names_no_repository_or_organisation(self):
        text = _skill_text()
        for forbidden in ("CVNA-SandboxOrg", "claude-runway", "claude_runway", "ClaudeRunway"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_every_repo_bound_command_takes_the_repo_from_the_checkout(self):
        text = _skill_text()
        # No `-R <repo>` and no hardcoded `repos/<owner>/<name>` path: the
        # only legal forms are gh's own {owner}/{repo} placeholders.
        self.assertIsNone(re.search(r"\s-R\s", text), "a command passes -R")
        self.assertIsNone(re.search(r"repos/(?!\{owner\}/\{repo\})[A-Za-z0-9_.-]+/", text), "a hardcoded repos/<owner>/ path")
        self.assertIn("gh repo view --json owner,name", text)

    def test_never_mentions_the_reviewer_bot_by_handle(self):
        text = _skill_text()
        # Not in a reply template, and not anywhere else either: a bare
        # handle in the file is one copy-paste away from a posted reply.
        self.assertIsNone(re.search(r"@copilot", text, re.IGNORECASE))
        for template in re.findall(r"```\n(Checked this.*?|Confirmed this.*?)\n```", text, re.DOTALL):
            self.assertNotIn("@", template)

    def test_states_the_no_mention_rule(self):
        self.assertIn("Do not @-mention the automated reviewer", _skill_text())

    def test_reply_endpoint_keeps_the_pr_segment(self):
        self.assertIn('pulls/<N>/comments/<comment_id>/replies', _skill_text())


class ThreadsAndCleanupAreSafe(unittest.TestCase):
    def test_thread_listing_follows_the_cursor_through_every_page(self):
        # A single `first: 100` page hides every thread past the 100th, so an
        # addressed finding there could never be resolved.
        text = _skill_text()
        listing = next(line for line in text.splitlines() if "reviewThreads(" in line)
        self.assertIn("gh api graphql --paginate", listing)
        self.assertIn("$endCursor: String", listing)
        self.assertIn("after: $endCursor", listing)
        self.assertIn("pageInfo { hasNextPage endCursor }", listing)

    def test_cleanup_deletes_only_confirmed_error_noise_and_keeps_findings_open(self):
        text = _skill_text()
        self.assertIn("Delete only comments you have read and confirmed", text)
        self.assertIn("CI output", text)
        self.assertIn("Deleting the noise does not close the finding", text)
        # The old wording treated the interrupted finding as resolved either way.
        self.assertNotIn("is resolved either way", text)


class FetchStepIsFixed(unittest.TestCase):
    COMMANDS = (
        "gh pr view <N> --json number,title,body,state,mergeable,reviewDecision,reviews",
        'gh api "repos/{owner}/{repo}/pulls/<N>/reviews" --paginate',
        'gh api "repos/{owner}/{repo}/pulls/<N>/comments" --paginate',
        'gh api "repos/{owner}/{repo}/issues/<N>/comments" --paginate',
    )

    def test_four_fetch_commands_appear_once_each_in_order(self):
        text = _skill_text()
        positions = []
        for command in self.COMMANDS:
            with self.subTest(command=command):
                self.assertEqual(text.count(command), 1)
                positions.append(text.index(command))
        self.assertEqual(positions, sorted(positions))

    def test_the_fetch_is_one_named_step_run_once_per_round(self):
        text = _skill_text()
        self.assertIn("one named step, the fetch", text)
        self.assertIn("once per round", text)

    def test_the_review_id_is_the_numeric_one_from_the_rest_reviews_output(self):
        text = _skill_text()
        # Never `latestReviews` (it keeps one review per reviewer), and never the PRR_ node id:
        # the REST fetch shows the numeric one, so the reader ties the two without matching.
        self.assertNotIn("`latestReviews` array", text)
        self.assertIn("numeric `id` field of the newest entry in the output of the second command", text)
        self.assertIn("not the `PRR_...` node id", text)

    def test_marker_line_is_documented_with_its_none_case(self):
        text = _skill_text()
        self.assertIn("pr-feedback-fetch: pr=<N> copilot-review=<id>", text)
        self.assertIn("copilot-review=none", text)

    def test_marker_is_printed_by_an_echo_call_so_it_lands_in_a_tool_result(self):
        # The reader looks for the marker only in tool results; text the model writes in its
        # reply is an assistant message and is never read. So the skill must have the line come
        # out of a command, as its own call after the four fetches, not as reply text.
        text = _skill_text()
        echo = 'echo "pr-feedback-fetch: pr=<N> copilot-review=<id>"'
        self.assertEqual(text.count(echo), 1)
        # After the last fetch command, and not inside the fetch block of four.
        self.assertGreater(text.index(echo), text.index(FetchStepIsFixed.COMMANDS[-1]))
        start = text.rfind("```bash", 0, text.index(FetchStepIsFixed.COMMANDS[0]))
        block = text[start:text.index("```", start + 7)]
        self.assertIn(FetchStepIsFixed.COMMANDS[-1], block)
        self.assertNotIn("pr-feedback-fetch", block)
        self.assertIn("own fifth call", text)
        # The old instruction, which put the line in the reply where the reader never looks.
        self.assertNotIn("as plain text in your reply", text)
        self.assertIn("So do not print it in your reply as well", text)

    def test_analytics_is_optional_and_nothing_is_recorded(self):
        text = _skill_text()
        self.assertIn("record_metric", text)
        self.assertIn("This skill records nothing else", text)
        self.assertIn("if it is absent nothing changes", text)


class ShipsLikeTheOtherProductSkills(unittest.TestCase):
    def setUp(self):
        self.mod = _load_setup_project()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        root = Path(self._tmpdir.name)
        self.skills = root / "skills-dest"
        # Path.home() is patched so a refresh can never touch the real ~/.claude.
        home_patch = mock.patch.object(Path, "home", return_value=root / "fake-home")
        home_patch.start()
        self.addCleanup(home_patch.stop)

    def _installed(self) -> Path:
        return self.skills / "cr-pr-feedback" / "SKILL.md"

    def test_init_install_skills_installs_the_real_shipped_skill(self):
        args = argparse.Namespace(target_repo=None, install_skills=True, skills_dir=str(self.skills), dry_run=False)
        with mock.patch.object(self.mod, "TOOLS_REPO_DIR", REPO_ROOT), redirect_stdout(io.StringIO()):
            self.mod.cmd_init(args)
        self.assertEqual(self._installed().read_bytes(), SKILL_PATH.read_bytes())

    def test_upgrade_refreshes_a_modified_installed_copy(self):
        self._installed().parent.mkdir(parents=True)
        self._installed().write_text("stale local edit\n", encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            done = self.mod.refresh_installed_skills(self.skills, tools_repo_dir=REPO_ROOT, auto_yes=True, dry_run=False)
        self.assertEqual(done, ["cr-pr-feedback"])
        self.assertEqual(self._installed().read_bytes(), SKILL_PATH.read_bytes())

    def test_upgrade_does_not_install_it_where_it_is_absent(self):
        with redirect_stdout(io.StringIO()):
            done = self.mod.refresh_installed_skills(self.skills, tools_repo_dir=REPO_ROOT, auto_yes=True, dry_run=False)
        self.assertEqual(done, [])
        self.assertFalse(self._installed().exists())


if __name__ == "__main__":
    unittest.main()
