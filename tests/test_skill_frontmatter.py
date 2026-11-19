#!/usr/bin/env python3
"""Tests that every shipped skills/*/SKILL.md carries a `name`/`description`
header, and that installing or upgrading handles it the way the installer
documents.

Without a header Claude Code lists a skill as "Skill: <name>" and cannot tell
when it applies. Against temp directories only, stdlib-only, no network:

    .venv/bin/python -m unittest discover -s tests
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from libs import setup_project_lib as lib  # noqa: E402

SKILLS_ROOT = REPO_ROOT / "skills"
SKILL_NAMES = sorted(p.name for p in SKILLS_ROOT.iterdir() if (p / "SKILL.md").is_file())


def _header(text: str) -> "dict[str, str]":
    match = re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, re.DOTALL)
    assert match, "no frontmatter block"
    out: dict = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


class EveryShippedSkillHasAHeader(unittest.TestCase):
    def test_there_are_skills_to_check(self):
        self.assertGreaterEqual(len(SKILL_NAMES), 8)

    def test_name_matches_directory_and_description_is_present(self):
        for name in SKILL_NAMES:
            with self.subTest(skill=name):
                text = (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")
                header = _header(text)
                self.assertEqual(header.get("name"), name)
                description = header.get("description", "")
                # One double-quoted line, so a colon inside it cannot break YAML.
                self.assertRegex(description, r'^".+"$')
                self.assertGreater(len(description.strip('"').strip()), 20)

    def test_heading_is_still_the_first_body_line(self):
        for name in SKILL_NAMES:
            with self.subTest(skill=name):
                text = (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")
                self.assertIn(f"\n---\n\n# Skill: {name}\n", text.replace("\r\n", "\n"))


class InstallAndUpgradeHandleTheHeader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.skills_dir = Path(self._tmp.name)

    def _plan(self, name: str):
        plans = lib.plan_skill_installs(REPO_ROOT, self.skills_dir)
        return next(p for p in plans if p.name == name)

    def _installed(self, name: str, text: str) -> Path:
        dest = self.skills_dir / name / "SKILL.md"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(text.encode("utf-8"))
        return dest

    def test_fresh_install_writes_the_shipped_header(self):
        plan = self._plan("cr-compact")
        self.assertEqual(plan.action, "install")
        self.assertEqual(_header(plan.content)["name"], "cr-compact")

    def test_headerless_installed_copy_is_updated_to_the_shipped_header(self):
        shipped = (SKILLS_ROOT / "cr-compact" / "SKILL.md").read_bytes().decode("utf-8")
        body = shipped[shipped.index("# Skill: cr-compact"):]
        self._installed("cr-compact", body)
        plan = self._plan("cr-compact")
        self.assertEqual(plan.action, "update")
        self.assertFalse(plan.frontmatter_preserved)
        self.assertEqual(_header(plan.content)["name"], "cr-compact")
        self.assertTrue(plan.content.startswith("---"))

    def test_hand_edited_installed_header_is_kept(self):
        shipped = (SKILLS_ROOT / "cr-compact" / "SKILL.md").read_bytes().decode("utf-8")
        body = shipped[shipped.index("# Skill: cr-compact"):]
        mine = '---\nname: cr-compact\ndescription: "my own words"\n---\n\n'
        self._installed("cr-compact", mine + body)
        plan = self._plan("cr-compact")
        self.assertEqual(plan.action, "up_to_date")
        self.assertTrue(plan.frontmatter_preserved)
        self.assertIn("my own words", plan.content)


if __name__ == "__main__":
    unittest.main()
