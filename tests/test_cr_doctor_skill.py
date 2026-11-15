#!/usr/bin/env python3
"""Tests for skills/cr-doctor/SKILL.md, the skill that runs tools/doctor.py.

A skill has no unit test of its behaviour, so these pin what can be checked
mechanically: the instructions the skill depends on (the exit-code contract,
the two missing-prerequisite messages, never reading .mcp.json whole), that the
doctor script it names really exists, and that `init --install-skills` finds it
without any hardcoded skill list.

    .venv/bin/python -m unittest discover -s tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_PATH = REPO_ROOT / "skills" / "cr-doctor" / "SKILL.md"

sys.path.insert(0, str(REPO_ROOT / "libs"))

import setup_project_lib as lib  # noqa: E402


class CrDoctorSkillTest(unittest.TestCase):
    def setUp(self):
        self.text = SKILL_PATH.read_text(encoding="utf-8")

    def test_names_the_script_it_runs_and_that_script_exists(self):
        self.assertIn("tools/doctor.py", self.text)
        self.assertTrue((REPO_ROOT / "tools" / "doctor.py").is_file())

    def test_reports_pass_and_fail_by_exit_code(self):
        self.assertIn("Exit 0", self.text)
        self.assertIn("Exit 1", self.text)

    def test_fails_clearly_without_the_tools_repo_or_venv(self):
        self.assertIn("`CLAUDE_RUNWAY_DIR` is not set", self.text)
        self.assertIn("virtual environment is missing", self.text)
        self.assertIn(".venv/bin/python", self.text)
        self.assertIn("Scripts\\python.exe", self.text)

    def test_quoting_rules_for_both_platforms(self):
        self.assertIn("single quotes", self.text)
        self.assertIn("& '<venv-python>'", self.text)

    def test_never_reads_mcp_json_and_never_runs_the_update_unasked(self):
        self.assertIn("Do not read `.mcp.json`", self.text)
        self.assertIn("/cr-setup update", self.text)
        self.assertIn("Do not run it unasked", self.text)

    def test_init_install_skills_picks_it_up_from_the_real_skills_dir(self):
        with tempfile.TemporaryDirectory() as d:
            plan = {p.name: p for p in lib.plan_skill_installs(REPO_ROOT, Path(d))}
        self.assertIn("cr-doctor", plan)
        self.assertEqual(plan["cr-doctor"].action, "install")


if __name__ == "__main__":
    unittest.main()
