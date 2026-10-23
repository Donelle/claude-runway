#!/usr/bin/env python3
"""`init`, `upgrade` and `doctor` support for the opt-in `codebase-reviewer`
server.

What these pin, because each is a way to get it quietly wrong:
  - `REVIEWERS_FILE` is required and has no default: neither `init` nor `upgrade`
    ever writes a default path or a reviewers file, and both warn when it is not
    set.
  - An existing block (and the user's own path in it) survives a re-run of `init`.
  - `doctor` reports a missing, blank or invalid reviewers file and exits 1.

Stdlib-only (unittest) and no network:

    .venv/bin/python -m unittest discover -s tests
"""

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import doctor as doctor_cli  # noqa: E402
from doctor_lib import run_doctor  # noqa: E402
from setup_project_lib import (  # noqa: E402
    REVIEWERS_FILE_REQUIRED_WARNING,
    ask_codebase_reviewer,
    build_mcp_servers,
    find_unresolved_placeholders,
    has_codebase_reviewer_block,
    load_json,
    merge_mcp_json,
    run_setup,
)
from test_setup_project_cli import _init_args, _load_setup_project  # noqa: E402
from upgrade_lib import CODEBASE_REVIEWER_OFFER_ID, run_upgrade  # noqa: E402

MCP_TEMPLATE = load_json(REPO_ROOT / "templates" / "mcp.json.template")
HOME = Path("/home/user")
VENV_PYTHON = Path("/home/user/tools/claude-runway/.venv/bin/python")
BLOCK = "codebase-reviewer"


def _build(**overrides):
    kwargs = dict(
        collection_name="my-project",
        venv_python=VENV_PYTHON,
        tools_repo_dir=Path("/home/user/tools/claude-runway"),
        home_dir=HOME,
    )
    kwargs.update(overrides)
    return build_mcp_servers(MCP_TEMPLATE, **kwargs)


def _valid_reviewers() -> dict:
    return {
        "rubric": [{"name": "r1", "level": "low", "definition": "d", "score": 2}],
        "reviewers": [
            {"name": "q", "model": "m", "prompt": "p", "active": True, "type": ["reasoning"], "loadOnDemand": False}
        ],
    }


class TemplateBlock(unittest.TestCase):
    def test_the_template_has_the_block_with_the_documented_env_and_no_default_path(self):
        env = MCP_TEMPLATE["mcpServers"][BLOCK]["env"]
        self.assertEqual(
            set(env), {"REVIEWERS_FILE", "MAX_REVIEWERS", "MAX_REVIEW_ROUNDS", "QDRANT_URL", "QDRANT_API_KEY"}
        )
        self.assertEqual(env["REVIEWERS_FILE"], "")
        self.assertEqual(env["MAX_REVIEWERS"], "3")
        self.assertEqual(env["MAX_REVIEW_ROUNDS"], "3")

    def test_the_template_block_points_at_the_new_server_script(self):
        self.assertEqual(
            MCP_TEMPLATE["mcpServers"][BLOCK]["args"], ["/absolute/path/to/mcpservers/codebase_reviewer_mcp_server.py"]
        )
        self.assertTrue((REPO_ROOT / "mcpservers" / "codebase_reviewer_mcp_server.py").is_file())


class BuildMcpServers(unittest.TestCase):
    def test_the_block_is_opt_in_and_absent_by_default(self):
        self.assertNotIn(BLOCK, _build())
        self.assertNotIn(BLOCK, _build(reviewers_file="/x/r.json"))  # the path alone is not the opt-in here

    def test_the_flag_writes_the_block_with_no_placeholders(self):
        servers = _build(codebase_reviewer=True)
        self.assertIn(BLOCK, servers)
        self.assertEqual(find_unresolved_placeholders(servers), [])
        block = servers[BLOCK]
        self.assertEqual(block["command"], VENV_PYTHON.as_posix())
        self.assertEqual(block["args"], ["/home/user/tools/claude-runway/mcpservers/codebase_reviewer_mcp_server.py"])

    def test_reviewers_file_is_blank_unless_given_never_a_default_path(self):
        self.assertEqual(_build(codebase_reviewer=True)[BLOCK]["env"]["REVIEWERS_FILE"], "")
        env = _build(codebase_reviewer=True, reviewers_file="/my/own/reviewers.json")[BLOCK]["env"]
        self.assertEqual(env["REVIEWERS_FILE"], "/my/own/reviewers.json")

    def test_limits_keep_the_template_values_unless_given(self):
        env = _build(codebase_reviewer=True)[BLOCK]["env"]
        self.assertEqual((env["MAX_REVIEWERS"], env["MAX_REVIEW_ROUNDS"]), ("3", "3"))
        env = _build(codebase_reviewer=True, max_reviewers="1", max_review_rounds="5")[BLOCK]["env"]
        self.assertEqual((env["MAX_REVIEWERS"], env["MAX_REVIEW_ROUNDS"]), ("1", "5"))

    def test_qdrant_settings_follow_the_other_servers(self):
        env = _build(codebase_reviewer=True, qdrant_url="http://q:6333", qdrant_api_key="k")[BLOCK]["env"]
        self.assertEqual((env["QDRANT_URL"], env["QDRANT_API_KEY"]), ("http://q:6333", "k"))

    def test_it_is_independent_of_local_compress_and_the_savings_tracker(self):
        servers = _build(codebase_reviewer=True, include_compress=False)
        self.assertIn(BLOCK, servers)
        self.assertNotIn("local-compress", servers)

    def test_the_block_is_not_owned_so_a_plain_merge_never_removes_an_existing_one(self):
        existing = {"mcpServers": {BLOCK: {"command": "mine", "env": {"REVIEWERS_FILE": "/mine.json"}}}}
        merged = merge_mcp_json(existing, _build())
        self.assertEqual(merged["mcpServers"][BLOCK]["env"]["REVIEWERS_FILE"], "/mine.json")


class RunSetup(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.target = Path(self._tmp.name) / "proj"
        self.target.mkdir()

    def _setup(self, mcp_json=None, **kwargs):
        if mcp_json is not None:
            (self.target / ".mcp.json").write_text(json.dumps(mcp_json), encoding="utf-8")
        return run_setup(self.target, REPO_ROOT, home_dir=HOME, **kwargs)

    def _warnings(self, result):
        return [c for c in result.changes if REVIEWERS_FILE_REQUIRED_WARNING in c]

    def test_a_plain_init_writes_no_reviewer_block_and_no_warning(self):
        result = self._setup()
        self.assertNotIn(BLOCK, result.mcp_json["mcpServers"])
        self.assertEqual(self._warnings(result), [])

    def test_the_flag_without_a_path_writes_a_blank_setting_and_warns(self):
        result = self._setup(codebase_reviewer=True)
        self.assertEqual(result.mcp_json["mcpServers"][BLOCK]["env"]["REVIEWERS_FILE"], "")
        (warning,) = self._warnings(result)
        self.assertIn("REQUIRED", warning)
        self.assertIn("no default", warning)
        self.assertIn("No default path", warning)

    def test_a_path_implies_the_block_and_needs_no_warning(self):
        result = self._setup(reviewers_file="/my/reviewers.json")
        self.assertEqual(result.mcp_json["mcpServers"][BLOCK]["env"]["REVIEWERS_FILE"], "/my/reviewers.json")
        self.assertEqual(self._warnings(result), [])

    def test_nothing_is_created_beyond_the_two_config_files(self):
        self._setup(codebase_reviewer=True)
        # run_setup is pure; nothing it does may create a reviewers file either.
        self.assertEqual(sorted(p.name for p in self.target.iterdir()), [])

    def test_a_rerun_without_the_flag_keeps_the_block_and_the_users_values(self):
        first = self._setup(codebase_reviewer=True, reviewers_file="/mine/r.json")
        block = first.mcp_json["mcpServers"][BLOCK]
        block["env"]["MAX_REVIEWERS"] = "1"
        block["env"]["MAX_REVIEW_ROUNDS"] = "6"
        again = self._setup(mcp_json=first.mcp_json)
        env = again.mcp_json["mcpServers"][BLOCK]["env"]
        self.assertEqual(env["REVIEWERS_FILE"], "/mine/r.json")
        self.assertEqual((env["MAX_REVIEWERS"], env["MAX_REVIEW_ROUNDS"]), ("1", "6"))
        self.assertEqual(self._warnings(again), [])

    def test_a_new_path_replaces_the_carried_one(self):
        first = self._setup(reviewers_file="/old.json")
        again = self._setup(mcp_json=first.mcp_json, reviewers_file="/new.json")
        self.assertEqual(again.mcp_json["mcpServers"][BLOCK]["env"]["REVIEWERS_FILE"], "/new.json")

    def test_a_kept_block_with_no_path_still_warns(self):
        first = self._setup(codebase_reviewer=True)
        again = self._setup(mcp_json=first.mcp_json)
        self.assertEqual(len(self._warnings(again)), 1)

    def _kept_block(self, **env_changes):
        """A first run's config with its reviewer block's env edited by hand:
        a value sets that key, None removes it."""
        first = self._setup(codebase_reviewer=True)
        env = first.mcp_json["mcpServers"][BLOCK]["env"]
        for key, value in env_changes.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return first.mcp_json

    def test_keys_absent_from_the_kept_block_stay_absent_so_the_shell_value_still_applies(self):
        # An absent key inherits the shell's value; writing "" or the template's
        # "3" there would mask a working export.
        existing = self._kept_block(REVIEWERS_FILE=None, MAX_REVIEWERS=None, MAX_REVIEW_ROUNDS=None)
        again = self._setup(mcp_json=existing)
        env = again.mcp_json["mcpServers"][BLOCK]["env"]
        for key in ("REVIEWERS_FILE", "MAX_REVIEWERS", "MAX_REVIEW_ROUNDS"):
            self.assertNotIn(key, env)
        # ...and no "REVIEWERS_FILE is not set" warning: the shell may supply it.
        self.assertEqual(self._warnings(again), [])

    def test_an_existing_block_with_no_env_at_all_is_kept_with_every_key_absent(self):
        # No `env` means everything inherits the shell; it is an existing block
        # with nothing set, not "no block", so it must not get template values.
        existing = self._kept_block()
        del existing["mcpServers"][BLOCK]["env"]
        again = self._setup(mcp_json=existing)
        env = again.mcp_json["mcpServers"][BLOCK]["env"]
        self.assertNotIn("REVIEWERS_FILE", env)
        self.assertNotIn("MAX_REVIEWERS", env)
        self.assertNotIn("MAX_REVIEW_ROUNDS", env)
        self.assertEqual(self._warnings(again), [])

    def test_an_existing_block_with_no_env_still_takes_an_explicit_path(self):
        existing = self._kept_block()
        del existing["mcpServers"][BLOCK]["env"]
        again = self._setup(mcp_json=existing, reviewers_file="/new.json")
        self.assertEqual(again.mcp_json["mcpServers"][BLOCK]["env"]["REVIEWERS_FILE"], "/new.json")

    def test_a_blank_reviewers_file_in_the_kept_block_stays_blank_and_still_warns(self):
        existing = self._kept_block(REVIEWERS_FILE="")
        again = self._setup(mcp_json=existing)
        self.assertEqual(again.mcp_json["mcpServers"][BLOCK]["env"]["REVIEWERS_FILE"], "")
        self.assertEqual(len(self._warnings(again)), 1)

    def test_the_users_limits_are_kept_exactly(self):
        existing = self._kept_block(MAX_REVIEWERS="1", MAX_REVIEW_ROUNDS="")
        env = self._setup(mcp_json=existing).mcp_json["mcpServers"][BLOCK]["env"]
        self.assertEqual((env["MAX_REVIEWERS"], env["MAX_REVIEW_ROUNDS"]), ("1", ""))

    def test_custom_env_entries_in_the_kept_block_survive(self):
        existing = self._kept_block(CLAUDE_RUNWAY_LMSTUDIO_URL="http://custom:1234/v1", MY_OWN_SETTING="x")
        env = self._setup(mcp_json=existing).mcp_json["mcpServers"][BLOCK]["env"]
        self.assertEqual(env["CLAUDE_RUNWAY_LMSTUDIO_URL"], "http://custom:1234/v1")
        self.assertEqual(env["MY_OWN_SETTING"], "x")

    def test_the_dry_run_preview_does_not_echo_the_users_custom_env_values(self):
        existing = self._kept_block(MY_SECRET_TOKEN="hunter2")
        result = self._setup(mcp_json=existing)
        self.assertNotIn("hunter2", json.dumps(result.generated_mcp_servers))
        self.assertEqual(result.mcp_json["mcpServers"][BLOCK]["env"]["MY_SECRET_TOKEN"], "hunter2")

    def test_the_qdrant_settings_still_follow_this_runs_flags(self):
        existing = self._kept_block(QDRANT_URL="http://old:6333")
        env = self._setup(mcp_json=existing, qdrant_url="http://new:6333").mcp_json["mcpServers"][BLOCK]["env"]
        self.assertEqual(env["QDRANT_URL"], "http://new:6333")

    def test_an_explicit_reviewers_file_beats_a_kept_blank_or_absent_one(self):
        for value in ("", None):
            with self.subTest(existing=value):
                existing = self._kept_block(REVIEWERS_FILE=value)
                env = self._setup(mcp_json=existing, reviewers_file="/new.json").mcp_json["mcpServers"][BLOCK]["env"]
                self.assertEqual(env["REVIEWERS_FILE"], "/new.json")

    def test_qdrant_only_does_not_remove_it(self):
        first = self._setup(reviewers_file="/mine/r.json")
        again = self._setup(mcp_json=first.mcp_json, include_compress=False)
        self.assertEqual(again.mcp_json["mcpServers"][BLOCK]["env"]["REVIEWERS_FILE"], "/mine/r.json")

    def test_dry_run_preview_lists_the_block(self):
        result = self._setup(codebase_reviewer=True)
        self.assertIn(BLOCK, result.generated_mcp_servers)
        self.assertIn(BLOCK, result.changes[0])


class AskCodebaseReviewer(unittest.TestCase):
    def _ask(self, answer, **kw):
        out: list = []
        return ask_codebase_reviewer(input_fn=lambda prompt: answer, print_fn=out.append, **kw), "\n".join(out)

    def test_yes_and_no(self):
        self.assertTrue(self._ask("y")[0])
        self.assertTrue(self._ask("YES")[0])
        self.assertFalse(self._ask("n")[0])
        self.assertFalse(self._ask("whatever")[0])

    def test_blank_takes_the_default(self):
        self.assertFalse(self._ask("")[0])
        self.assertTrue(self._ask("", default_enabled=True)[0])

    def test_a_closed_stdin_takes_the_default(self):
        def eof(prompt):
            raise EOFError

        self.assertFalse(ask_codebase_reviewer(input_fn=eof, print_fn=lambda *_: None))

    def test_the_question_says_the_file_is_required_and_has_no_default(self):
        _, text = self._ask("n")
        self.assertIn("REVIEWERS_FILE is required and has no default", text)


class InitCli(unittest.TestCase):
    def setUp(self):
        self.mod = _load_setup_project()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.target = Path(self._tmp.name) / "proj"
        self.target.mkdir()

    def _init(self, **over):
        args = _init_args(self.target, dry_run=False, **over)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        return buf.getvalue()

    def _mcp(self):
        return json.loads((self.target / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]

    def test_default_init_adds_no_reviewer_block_and_asks_nothing(self):
        out = self._init()
        self.assertNotIn(BLOCK, self._mcp())
        self.assertNotIn("REVIEWERS_FILE", out)

    def test_flag_writes_a_blank_setting_and_warns_without_writing_a_default_file(self):
        before = {p.name for p in self.target.iterdir()}
        out = self._init(codebase_reviewer=True)
        self.assertEqual(self._mcp()[BLOCK]["env"]["REVIEWERS_FILE"], "")
        self.assertIn("REVIEWERS_FILE", out)
        self.assertIn("REQUIRED and has no default", out)
        # Only the files init always writes: no reviewers file anywhere.
        self.assertEqual({p.name for p in self.target.iterdir()} - before, {".mcp.json", ".claude", "CLAUDE.md"})
        self.assertEqual(list(self.target.rglob("*reviewers*")), [])

    def test_reviewers_file_flag_writes_the_given_path_verbatim_with_no_warning(self):
        out = self._init(reviewers_file="~/my/reviewers.json")
        self.assertEqual(self._mcp()[BLOCK]["env"]["REVIEWERS_FILE"], "~/my/reviewers.json")
        self.assertNotIn("REQUIRED and has no default", out)

    def test_dry_run_with_the_flag_warns_and_writes_nothing(self):
        args = _init_args(self.target, dry_run=True, codebase_reviewer=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        self.assertFalse((self.target / ".mcp.json").exists())
        self.assertIn("REQUIRED and has no default", buf.getvalue())

    def test_a_rerun_keeps_the_block(self):
        self._init(reviewers_file="/mine.json")
        self._init()
        self.assertEqual(self._mcp()[BLOCK]["env"]["REVIEWERS_FILE"], "/mine.json")

    def test_the_parser_accepts_both_flags(self):
        with mock.patch.object(
            sys, "argv", ["setup_project.py", "init", str(self.target), "--codebase-reviewer", "--reviewers-file", "/r.json"]
        ):
            parsed = self.mod.parse_args()
        self.assertTrue(parsed.codebase_reviewer)
        self.assertEqual(parsed.reviewers_file, "/r.json")
        with mock.patch.object(sys, "argv", ["setup_project.py", "init", str(self.target)]):
            parsed = self.mod.parse_args()
        self.assertFalse(parsed.codebase_reviewer)
        self.assertEqual(parsed.reviewers_file, "")


class UpgradeCliWiring(unittest.TestCase):
    def setUp(self):
        self.mod = _load_setup_project()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.target = root / "proj"
        self.target.mkdir()
        result = run_setup(self.target, REPO_ROOT, home_dir=HOME)
        (self.target / ".mcp.json").write_text(json.dumps(result.mcp_json), encoding="utf-8")
        patch = mock.patch.object(Path, "home", return_value=root / "fake-home")
        patch.start()
        self.addCleanup(patch.stop)

    def _offer_passed(self, *, tty, **flags):
        self.mod._stdin_is_interactive = lambda: tty
        captured: dict = {}

        def fake_run_upgrade(target_repo, tools_repo_dir, venv_python, **kw):
            captured.update(kw)
            return []

        args = argparse.Namespace(
            target_repo=str(self.target),
            dry_run=flags.get("dry_run", False),
            auto_yes=flags.get("auto_yes", False),
            skills_dir=str(Path(self._tmp.name) / "skills"),
        )
        with mock.patch.object(self.mod, "run_upgrade", side_effect=fake_run_upgrade), redirect_stdout(io.StringIO()):
            self.mod.cmd_upgrade(args)
        return captured["offer_reviewer_fn"]

    def test_an_interactive_run_passes_the_question(self):
        self.assertIs(self._offer_passed(tty=True), self.mod.ask_codebase_reviewer)

    def test_no_terminal_auto_yes_and_dry_run_pass_none(self):
        self.assertIsNone(self._offer_passed(tty=False))
        self.assertIsNone(self._offer_passed(tty=True, auto_yes=True))
        self.assertIsNone(self._offer_passed(tty=True, dry_run=True))


class UpgradeOffer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.target = Path(self._tmp.name) / "proj"
        self.target.mkdir()
        self.mcp_path = self.target / ".mcp.json"

    def _init_project(self, **kwargs):
        result = run_setup(self.target, REPO_ROOT, home_dir=HOME, **kwargs)
        self.mcp_path.write_text(json.dumps(result.mcp_json), encoding="utf-8")
        (self.target / ".claude").mkdir(exist_ok=True)
        (self.target / ".claude" / "settings.json").write_text(json.dumps(result.settings_json), encoding="utf-8")

    def _servers(self):
        return json.loads(self.mcp_path.read_text(encoding="utf-8"))["mcpServers"]

    def _run(self, **kwargs):
        buf = io.StringIO()
        with redirect_stdout(buf):
            applied = run_upgrade(self.target, REPO_ROOT, VENV_PYTHON, **kwargs)
        return applied, buf.getvalue()

    def test_a_yes_adds_the_block_with_a_blank_file_and_warns(self):
        self._init_project()
        asked = []
        applied, out = self._run(offer_reviewer_fn=lambda: asked.append(1) or True)
        self.assertEqual(asked, [1])
        self.assertEqual(applied, [CODEBASE_REVIEWER_OFFER_ID])
        env = self._servers()[BLOCK]["env"]
        self.assertEqual(env["REVIEWERS_FILE"], "")
        self.assertIn("REVIEWERS_FILE is not set", out)
        self.assertIn("REQUIRED and has no default", out)

    def test_it_writes_no_default_path_and_no_reviewers_file(self):
        self._init_project()
        before = sorted(str(p.relative_to(self.target)) for p in self.target.rglob("*"))
        self._run(offer_reviewer_fn=lambda: True)
        after = sorted(str(p.relative_to(self.target)) for p in self.target.rglob("*"))
        self.assertEqual(after, before)
        self.assertEqual(self._servers()[BLOCK]["env"]["REVIEWERS_FILE"], "")

    def test_the_block_runs_the_new_script_under_the_given_interpreter(self):
        self._init_project()
        self._run(offer_reviewer_fn=lambda: True)
        block = self._servers()[BLOCK]
        self.assertEqual(block["command"], VENV_PYTHON.as_posix())
        self.assertEqual(block["args"], [(REPO_ROOT / "mcpservers" / "codebase_reviewer_mcp_server.py").as_posix()])
        self.assertEqual(find_unresolved_placeholders(block), [])

    def test_the_qdrant_settings_follow_the_projects_own_qdrant_block(self):
        self._init_project(qdrant_url="http://q:7000", qdrant_api_key="secret")
        self._run(offer_reviewer_fn=lambda: True)
        env = self._servers()[BLOCK]["env"]
        self.assertEqual((env["QDRANT_URL"], env["QDRANT_API_KEY"]), ("http://q:7000", "secret"))

    def test_a_no_changes_nothing(self):
        self._init_project()
        before = self.mcp_path.read_text(encoding="utf-8")
        applied, out = self._run(offer_reviewer_fn=lambda: False)
        self.assertEqual(applied, [])
        self.assertEqual(self.mcp_path.read_text(encoding="utf-8"), before)
        self.assertIn("No pending upgrades", out)

    def test_it_is_not_asked_when_there_is_no_function_dry_run_or_auto_yes(self):
        self._init_project()

        def explode():
            raise AssertionError("must not be asked")

        self._run()
        self._run(dry_run=True, offer_reviewer_fn=explode)
        self._run(auto_yes=True, offer_reviewer_fn=explode)
        self.assertNotIn(BLOCK, self._servers())

    def test_it_is_not_asked_when_the_project_already_has_the_block(self):
        self._init_project(reviewers_file="/mine.json")

        def explode():
            raise AssertionError("must not be asked")

        applied, _ = self._run(offer_reviewer_fn=explode)
        self.assertEqual(applied, [])
        self.assertEqual(self._servers()[BLOCK]["env"]["REVIEWERS_FILE"], "/mine.json")

    def test_other_servers_and_settings_are_untouched(self):
        self._init_project()
        before_servers = self._servers()
        before_settings = (self.target / ".claude" / "settings.json").read_text(encoding="utf-8")
        self._run(offer_reviewer_fn=lambda: True)
        after = self._servers()
        self.assertEqual({k: v for k, v in after.items() if k != BLOCK}, before_servers)
        self.assertEqual((self.target / ".claude" / "settings.json").read_text(encoding="utf-8"), before_settings)

    def test_the_helper_agrees_with_what_it_added(self):
        self._init_project()
        self._run(offer_reviewer_fn=lambda: True)
        self.assertTrue(has_codebase_reviewer_block(json.loads(self.mcp_path.read_text(encoding="utf-8"))))


class Doctor(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.target = Path(self._tmp.name) / "proj"
        self.target.mkdir()
        self.home = Path(self._tmp.name) / "home"

    def _config(self, env=None, include_block=True):
        servers: dict = {"qdrant": {"command": "p", "type": "stdio", "args": ["x"], "env": {}}}
        if include_block:
            servers[BLOCK] = {"command": "p", "type": "stdio", "args": ["x"], "env": env if env is not None else {}}
        (self.target / ".mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")

    def _file(self, data=None, name="reviewers.json") -> str:
        path = self.target / name
        path.write_text(data if isinstance(data, str) else json.dumps(data or _valid_reviewers()), encoding="utf-8")
        return str(path)

    def _doctor(self, shell_env=None):
        return run_doctor(self.target, shell_env=shell_env or {}, home_dir=self.home)

    def test_no_block_means_nothing_to_check_and_exit_zero(self):
        self._config(include_block=False)
        result = self._doctor()
        self.assertFalse(result.reviewer_checked)
        self.assertIsNone(result.reviewers_file_problem)
        self.assertEqual(self._main(), 0)

    def test_a_blank_reviewers_file_is_a_problem(self):
        self._config({"REVIEWERS_FILE": ""})
        result = self._doctor()
        self.assertTrue(result.reviewer_checked)
        self.assertIn("REVIEWERS_FILE is not set", result.reviewers_file_problem or "")

    def test_a_block_with_no_key_and_no_shell_value_is_a_problem(self):
        self._config({})
        self.assertIn("REVIEWERS_FILE is not set", self._doctor().reviewers_file_problem or "")

    def test_a_missing_file_names_the_path(self):
        missing = str(self.target / "nope.json")
        self._config({"REVIEWERS_FILE": missing})
        self.assertIn(missing, self._doctor().reviewers_file_problem or "")

    def test_an_invalid_file_lists_every_problem(self):
        data = _valid_reviewers()
        data["rubric"][0]["level"] = "critical"
        data["reviewers"][0]["active"] = "yes"
        self._config({"REVIEWERS_FILE": self._file(data)})
        problem = self._doctor().reviewers_file_problem or ""
        self.assertIn("'level' must be one of", problem)
        self.assertIn("'active' is required", problem)

    def test_a_valid_file_is_ok_and_summarised(self):
        self._config({"REVIEWERS_FILE": self._file()})
        result = self._doctor()
        self.assertTrue(result.reviewer_checked)
        self.assertIsNone(result.reviewers_file_problem)
        self.assertIn("1 reviewer(s), 1 rubric entry", result.reviewers_file_summary)

    def test_a_key_absent_from_the_block_inherits_the_shell_value(self):
        self._config({})
        result = self._doctor({"REVIEWERS_FILE": self._file()})
        self.assertIsNone(result.reviewers_file_problem)

    def test_a_blank_key_in_the_block_masks_the_shell_value(self):
        self._config({"REVIEWERS_FILE": ""})
        self.assertIn("REVIEWERS_FILE is not set", self._doctor({"REVIEWERS_FILE": self._file()}).reviewers_file_problem or "")

    def test_a_relative_path_resolves_against_the_project_directory(self):
        self._file(name="rel.json")
        self._config({"REVIEWERS_FILE": "rel.json"})
        self.assertIsNone(self._doctor().reviewers_file_problem)

    def test_a_bad_numeric_setting_is_a_problem_too(self):
        self._config({"REVIEWERS_FILE": self._file(), "MAX_REVIEWERS": "lots"})
        self.assertIn("MAX_REVIEWERS", self._doctor().reviewers_file_problem or "")

    def _main(self, shell_env=None):
        with mock.patch.dict(os.environ, shell_env or {}), redirect_stdout(io.StringIO()):
            return doctor_cli.main([str(self.target)])

    def test_the_cli_exits_one_and_explains_for_a_problem(self):
        self._config({"REVIEWERS_FILE": ""})
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = doctor_cli.main([str(self.target)])
        self.assertEqual(code, 1)
        self.assertIn("reviewers file cannot be used", buf.getvalue())
        self.assertIn("REVIEWERS_FILE is not set", buf.getvalue())

    def test_the_cli_exits_zero_and_says_ok_for_a_valid_file(self):
        self._config({"REVIEWERS_FILE": self._file()})
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = doctor_cli.main([str(self.target)])
        self.assertEqual(code, 0)
        self.assertIn("Reviewers file OK:", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
