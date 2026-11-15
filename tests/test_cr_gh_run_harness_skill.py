#!/usr/bin/env python3
"""Tests for .claude/skills/cr-gh-run-harness/SKILL.md and the agent it starts, .claude/agents/harness-runner.md.

The skill is prose instructions, not a Python module, so this mirrors tests/test_cr_gh_code_it_skill.py's
content-pinning strategy: assert the commands and cautions a future edit must not lose, so a reworded
skill that drops one fails here and not in front of a person with a model loading.

Stdlib-only (unittest, no pytest), no network, no LM Studio.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_PATH = os.path.join(REPO_ROOT, ".claude", "skills", "cr-gh-run-harness", "SKILL.md")
AGENT_PATH = os.path.join(REPO_ROOT, ".claude", "agents", "harness-runner.md")

# The same resolver cr-gh-code-it embeds: Windows `uv venv` makes .venv/Scripts/, Linux/macOS .venv/bin/.
RESOLVER_LINE = (
    "PYTHON=$(if [ -f .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe; "
    "else echo .venv/bin/python; fi)"
)


def read_skill() -> str:
    with open(SKILL_PATH, encoding="utf-8") as handle:
        return handle.read()


def read_agent() -> str:
    with open(AGENT_PATH, encoding="utf-8") as handle:
        return handle.read()


class SkillShape(unittest.TestCase):
    def setUp(self):
        self.text = read_skill()

    def test_frontmatter_names_the_skill(self):
        self.assertTrue(self.text.startswith("---\nname: cr-gh-run-harness\n"))
        self.assertRegex(self.text, r'(?m)^description: ".+"$')

    def test_resolves_the_interpreter_for_both_venv_layouts(self):
        self.assertIn(RESOLVER_LINE, self.text)

    def test_no_command_hardcodes_the_posix_interpreter_path(self):
        for block in re.findall(r"```bash\n(.*?)```", self.text, re.DOTALL):
            for line in block.splitlines():
                if line.lstrip().startswith("PYTHON="):
                    continue
                self.assertNotIn(".venv/bin/python", line, line)

    def test_every_command_that_uses_python_resolves_it_in_the_same_block(self):
        # Shell variables do not persist from one Bash call to the next, so a command block that uses $PYTHON
        # must define it itself: a variable set in an earlier block is empty by then and the runner would be
        # executed directly, without the venv.
        blocks = [b for b in re.findall(r"```bash\n(.*?)```", self.text, re.DOTALL) if "$PYTHON" in b]
        self.assertGreaterEqual(len(blocks), 1)
        for block in blocks:
            self.assertIn(RESOLVER_LINE, block, block)
            self.assertLess(block.index(RESOLVER_LINE), block.index("$PYTHON harness"), block)

    def test_lists_the_configs_before_running_anything(self):
        self.assertIn("run_live_tests.py --list", self.text)
        self.assertLess(self.text.index("--list"), self.text.index("harness-runner"))

    def test_does_not_carry_a_second_copy_of_the_run_procedure(self):
        # The command that starts a run and the monitor that watches it live in the agent only.
        self.assertNotIn("run_live_tests.py <config-name>", self.text)
        self.assertNotIn("heartbeat", self.text)
        self.assertNotIn("Monitor", self.text)

    def test_launches_the_agent_by_its_exact_name_and_lets_the_agent_file_set_model_and_effort(self):
        self.assertIn("exact name `harness-runner`", self.text)
        self.assertIn("do not pass `model` or `effort`", self.text)

    def test_one_config_per_agent_and_never_two_runs_at_once(self):
        self.assertIn("One config per agent", self.text)
        self.assertIn("only after the previous one has reported", self.text)

    def test_a_blocked_agent_is_reported_not_worked_around(self):
        self.assertIn("BLOCKED:", self.text)
        self.assertIn("do not run anything yourself", self.text)

    def test_asks_which_config_and_how_many_runs_instead_of_guessing(self):
        self.assertIn("AskUserQuestion", self.text)
        self.assertIn("how many runs", self.text)
        self.assertIn("the run count", self.text)  # handed to the agent, which turns it into --runs

    def test_runs_in_the_background_and_does_not_poll(self):
        self.assertIn("run_in_background", self.text)
        self.assertIn("rather than polling", self.text)

    def test_keeps_the_preflight_guard(self):
        self.assertIn("Do not add `--skip-checks`", self.text)

    def test_explains_the_exit_codes_and_that_3_is_not_a_pass(self):
        for code in ("`0`", "`1`", "`3`"):
            self.assertIn(code, self.text)
        self.assertIn("not a pass", self.text)

    def test_reports_the_log_path(self):
        self.assertIn("harness/live-tests/logs/", self.text)

    def test_never_edits_baselines_to_make_a_run_pass(self):
        self.assertIn("Do not edit configs, baselines or code to make a failing run pass", self.text)


class HarnessRunnerAgent(unittest.TestCase):
    def setUp(self):
        self.text = read_agent()
        self.frontmatter = self.text.split("---\n")[1]

    def test_the_name_and_description(self):
        self.assertTrue(self.text.startswith("---\nname: harness-runner\n"))
        self.assertRegex(self.text, r'(?m)^description: ".+Started only by that skill; not for general tasks\."$')

    def test_it_is_a_small_model_at_low_effort_because_the_work_is_mechanical(self):
        self.assertIn("model: claude-haiku-5-5", self.frontmatter)
        self.assertIn("effort: low", self.frontmatter)

    def test_it_turns_the_run_count_into_the_runs_flag(self):
        self.assertIn("--runs <N>", self.text)

    def test_it_can_run_commands_and_watch_them_but_cannot_edit_anything(self):
        tools = re.search(r"(?m)^tools: (.+)$", self.frontmatter).group(1)
        names = {t.strip() for t in tools.split(",")}
        self.assertEqual(names, {"Bash", "Read", "ToolSearch", "Monitor"})
        self.assertIn("You have no tool that edits a file, on purpose", self.text)

    def test_it_resolves_the_interpreter_in_every_command_that_uses_it(self):
        blocks = [b for b in re.findall(r"```bash\n(.*?)```", self.text, re.DOTALL) if "$PYTHON" in b]
        self.assertEqual(len(blocks), 2)  # the --list check and the run
        for block in blocks:
            self.assertIn(RESOLVER_LINE, block, block)
            self.assertLess(block.index(RESOLVER_LINE), block.index("$PYTHON harness"), block)
        for block in re.findall(r"```bash\n(.*?)```", self.text, re.DOTALL):
            for line in block.splitlines():
                if not line.lstrip().startswith("PYTHON="):
                    self.assertNotIn(".venv/bin/python", line, line)

    def test_it_checks_the_config_can_run_before_loading_a_model(self):
        self.assertIn("run_live_tests.py --list", self.text)
        self.assertLess(self.text.index("--list"), self.text.index("run_live_tests.py <config-name>"))
        self.assertIn("`NOT RUNNABLE`", self.text)

    def test_it_runs_the_one_config_it_is_given_and_never_picks_or_runs_all(self):
        self.assertIn("BLOCKED: no config name", self.text)
        self.assertIn('never run "every config"', self.text)

    def test_it_never_skips_the_preflight_and_never_edits_to_make_a_run_pass(self):
        self.assertIn("Do not add `--skip-checks`", self.text)
        self.assertIn("Do not edit configs, baselines or code to make a failing run pass", self.text)

    def test_it_starts_the_run_in_the_background_and_watches_it_instead_of_polling(self):
        self.assertIn("`run_in_background`", self.text)
        self.assertIn("select:Monitor", self.text)
        self.assertIn("timeout_ms", self.text)
        self.assertIn("1800000", self.text)
        self.assertIn("[heartbeat", self.text)
        self.assertIn("Do not poll the output file yourself", self.text)

    def test_the_monitor_filters_out_the_http_noise_and_ends_with_the_run(self):
        self.assertIn("_client\\.py|localhost:1234|ompletions", self.text)
        self.assertIn("^===== results", self.text)
        self.assertIn("pgrep -f run_live_tests.py", self.text)

    def test_it_says_in_a_line_where_the_run_is_and_does_not_call_a_quiet_run_a_hang(self):
        self.assertIn("write ONE short line", self.text)
        self.assertIn("Do not report it as a hang unless the process is gone", self.text)

    def test_the_report_has_the_exit_code_the_results_the_log_path_and_the_unexpected(self):
        for code in ("`0`", "`1`", "`3`"):
            self.assertIn(code, self.text)
        self.assertIn("not a pass", self.text)
        self.assertIn("`[baseline]`", self.text)
        self.assertIn("`[mechanics]`", self.text)
        self.assertIn("harness/live-tests/logs/", self.text)

    def test_it_does_not_suggest_fixes(self):
        self.assertIn("do not suggest a fix", self.text)

    def test_it_cites_no_ticket_or_pr_number(self):
        self.assertIsNone(re.search(r"(?<![\w`])#\d+\b", self.text))


class ReferencedFilesExist(unittest.TestCase):
    def test_the_runner_and_configs_folder_the_skill_names_exist(self):
        self.assertTrue(os.path.isfile(os.path.join(REPO_ROOT, "harness", "live-tests", "run_live_tests.py")))
        self.assertTrue(os.path.isdir(os.path.join(REPO_ROOT, "harness", "live-tests", "configs")))

    def test_the_workflow_doc_lists_the_skill_and_its_agent(self):
        with open(os.path.join(REPO_ROOT, "docs", "development-workflow.md"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("| `cr-gh-run-harness` |", text)
        self.assertIn(".claude/agents/harness-runner.md", text)


if __name__ == "__main__":
    unittest.main()
