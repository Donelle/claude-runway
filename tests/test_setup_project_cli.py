#!/usr/bin/env python3
"""Tests for tools/setup_project.py's `_hook_env_reminders`.

Stdlib-only (unittest, no pytest) and no network -- pure function over an
argparse Namespace, so this needs neither a real target repo nor a real
venv:

    .venv/bin/python -m unittest discover -s tests

Why this file exists (PR #113 review): --track-savings/--lmstudio-model/
--lmstudio-url/--savings-db only reach the local-compress MCP server's env
block in the generated .mcp.json -- they do NOT reach the hook scripts
written into .claude/settings.json, since Claude Code hook entries have no
env field of their own and instead inherit the shell's environment
unfiltered. The CLI previously reported these options as fully applied with
no indication that the hook-side half of the same setting still needs a
matching shell export -- _hook_env_reminders is what now surfaces that gap
to the user instead of leaving it silent.

Also covers _write_json's atomic-write behavior and cmd_init's initial-index
reminder (both also found in PR #113 review): _write_json previously opened
the destination directly with "w", truncating it immediately, so an
interruption or disk-full error mid-write could destroy the very
.mcp.json/.claude/settings.json content this script's merge logic exists to
preserve; and the printed "run the initial index" reminder previously showed
a literal "<name>" placeholder with no --qdrant-url, which either fails to
copy-paste as-is or silently indexes the wrong Qdrant instance.
"""

import argparse
import importlib.util
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


def _load_setup_project():
    """Loads a fresh copy of tools/setup_project.py by file path, the same
    pattern tests/test_compress_mcp_server.py uses -- this module isn't
    normally importable by name since it lives under tools/, not libs/."""
    spec = importlib.util.spec_from_file_location("setup_project", REPO_ROOT / "tools" / "setup_project.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        track_savings=False,
        savings_db="",
        lmstudio_model="",
        lmstudio_url="http://localhost:1234/v1",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class HookEnvReminders(unittest.TestCase):
    def setUp(self):
        self.mod = _load_setup_project()

    def test_all_defaults_produce_no_reminders(self):
        self.assertEqual(self.mod._hook_env_reminders(_args()), [])

    def test_track_savings_flags_the_shell_export(self):
        reminders = self.mod._hook_env_reminders(_args(track_savings=True))
        self.assertIn("CLAUDE_RUNWAY_TRACK_SAVINGS=1", reminders)

    def test_custom_lmstudio_model_flags_the_shell_export(self):
        reminders = self.mod._hook_env_reminders(_args(lmstudio_model="google/gemma-3-4b"))
        self.assertIn("CLAUDE_RUNWAY_LMSTUDIO_MODEL=google/gemma-3-4b", reminders)

    def test_custom_lmstudio_url_flags_the_shell_export(self):
        reminders = self.mod._hook_env_reminders(_args(lmstudio_url="http://localhost:9999/v1"))
        self.assertIn("CLAUDE_RUNWAY_LMSTUDIO_URL=http://localhost:9999/v1", reminders)

    def test_default_lmstudio_url_is_not_flagged(self):
        # The default value itself isn't something the user "set" -- only a
        # value that actually differs from the default implies a shell
        # export is needed to match it.
        reminders = self.mod._hook_env_reminders(_args(lmstudio_url="http://localhost:1234/v1"))
        self.assertEqual([r for r in reminders if r.startswith("CLAUDE_RUNWAY_LMSTUDIO_URL")], [])

    def test_savings_db_flags_the_shell_export(self):
        reminders = self.mod._hook_env_reminders(_args(savings_db="/custom/savings.db"))
        self.assertIn("CLAUDE_RUNWAY_SAVINGS_DB=/custom/savings.db", reminders)


class WriteJsonAtomic(unittest.TestCase):
    def setUp(self):
        self.mod = _load_setup_project()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = Path(self._tmpdir.name) / "sub" / "config.json"

    def test_writes_valid_json_that_round_trips(self):
        self.mod._write_json(self.path, {"a": 1, "b": [1, 2, 3]})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"a": 1, "b": [1, 2, 3]})

    def test_no_leftover_temp_file_after_success(self):
        self.mod._write_json(self.path, {"a": 1})
        self.assertEqual(list(self.path.parent.glob(".*.tmp")), [])

    def test_failure_mid_write_leaves_original_file_untouched(self):
        # Regression for the finding from PR #113 review: writing directly
        # to the destination truncated it immediately, so a failure partway
        # through (disk full, interruption, or here, a non-serializable
        # value) could destroy the caller's existing content instead of
        # just failing to update it.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('{"original": true}', encoding="utf-8")

        class Unserializable:
            pass

        with self.assertRaises(TypeError):
            self.mod._write_json(self.path, {"bad": Unserializable()})

        self.assertEqual(self.path.read_text(encoding="utf-8"), '{"original": true}')
        self.assertEqual(list(self.path.parent.glob(".*.tmp")), [])


def _init_args(target_repo, **overrides) -> argparse.Namespace:
    defaults = dict(
        target_repo=str(target_repo),
        collection_name=None,
        qdrant_url="http://localhost:6333",
        qdrant_api_key="",
        collection_description="",
        include_extensions="",
        exclude_dirs="",
        lmstudio_url="http://localhost:1234/v1",
        lmstudio_model="",
        track_savings=False,
        savings_db="",
        compact_collection="",
        memory_bank_collection="",
        memory_bank_id="",
        qdrant_only=False,
        skip_hooks=False,
        dry_run=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class DryRunNeverLeaksUnrelatedSecrets(unittest.TestCase):
    """Regression for the finding from PR #113 review: --dry-run previously
    printed the FULL merged .mcp.json/.claude/settings.json, so an unrelated
    existing MCP server's secret (an API token in its own env block) was
    printed to the terminal/CI log right along with this toolkit's own
    generated content."""

    def setUp(self):
        self.mod = _load_setup_project()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-project"
        self.target_repo.mkdir()
        (self.target_repo / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"github": {"command": "x", "env": {"GITHUB_TOKEN": "super-secret-value"}}}}),
            encoding="utf-8",
        )

    def test_dry_run_output_never_contains_the_unrelated_secret(self):
        args = _init_args(self.target_repo, dry_run=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        output = buf.getvalue()
        # The secret value must never appear in dry-run output.
        self.assertNotIn("super-secret-value", output)
        # The pre-existing server's token env var key must also not appear --
        # its presence would indicate the server's own env block was leaked.
        self.assertNotIn("GITHUB_TOKEN", output)
        # Note: the bare word "github" is NOT checked here because it appears
        # legitimately in the hook matcher string (mcp__github__* tool
        # patterns this toolkit generates). The two assertions above are the
        # actual sentinels -- they guard the secret value and env var key
        # directly rather than using the server name as a proxy.

    def test_real_write_still_preserves_the_unrelated_server(self):
        # The fix must only change what gets PRINTED, not what gets WRITTEN
        # -- a real write still has to preserve the user's own config.
        args = _init_args(self.target_repo, dry_run=False)
        self.mod.cmd_init(args)
        written = json.loads((self.target_repo / ".mcp.json").read_text(encoding="utf-8"))
        self.assertEqual(written["mcpServers"]["github"]["env"]["GITHUB_TOKEN"], "super-secret-value")


class QdrantApiKeyNeverLeaksInOwnOutput(unittest.TestCase):
    """Regression for the finding from PR #113 review: even after
    generated_mcp_servers stopped including UNRELATED existing secrets, it
    still included the REAL --qdrant-api-key value the user just typed on
    this exact command line, so --dry-run (and the printed follow-up
    ingest_to_qdrant.py command) both echoed that credential straight to
    the terminal/CI log. Also covers the "don't commit .mcp.json" warning
    that must replace the normal commit instruction once a real key is set."""

    def setUp(self):
        self.mod = _load_setup_project()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-project"
        self.target_repo.mkdir()

    def test_dry_run_redacts_the_key_in_the_preview(self):
        args = _init_args(self.target_repo, qdrant_api_key="my-real-secret", dry_run=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        output = buf.getvalue()
        self.assertNotIn("my-real-secret", output)
        self.assertIn(self.mod._REDACTED, output)

    def test_index_command_reminder_uses_a_placeholder_not_the_real_key(self):
        args = _init_args(self.target_repo, qdrant_api_key="my-real-secret", dry_run=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        output = buf.getvalue()
        self.assertNotIn("my-real-secret", output)
        self.assertIn("--qdrant-api-key", output)
        self.assertIn("<your-qdrant-api-key>", output)

    def test_real_write_still_puts_the_real_key_on_disk(self):
        # The fix must only change what gets PRINTED, not what gets WRITTEN.
        args = _init_args(self.target_repo, qdrant_api_key="my-real-secret", dry_run=False)
        self.mod.cmd_init(args)
        written = json.loads((self.target_repo / ".mcp.json").read_text(encoding="utf-8"))
        self.assertEqual(written["mcpServers"]["qdrant"]["env"]["QDRANT_API_KEY"], "my-real-secret")

    def test_commit_warning_replaces_the_normal_commit_instruction(self):
        args = _init_args(self.target_repo, qdrant_api_key="my-real-secret", dry_run=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        output = buf.getvalue()
        self.assertIn("Do NOT commit .mcp.json as-is", output)
        self.assertNotIn("Commit .mcp.json to the repo", output)

    def test_no_key_set_shows_the_normal_commit_instruction(self):
        args = _init_args(self.target_repo, dry_run=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        output = buf.getvalue()
        self.assertIn("Commit .mcp.json to the repo", output)
        self.assertNotIn("Do NOT commit .mcp.json as-is", output)


class InitialIndexReminder(unittest.TestCase):
    """cmd_init's printed 'run the initial index' reminder (PR #113 review:
    it previously showed a literal '<name>' placeholder and no --qdrant-url,
    which either fails to copy-paste as-is or silently indexes the wrong
    Qdrant instance when a custom --qdrant-url was actually configured)."""

    def setUp(self):
        self.mod = _load_setup_project()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-project"
        self.target_repo.mkdir()

    def test_uses_resolved_collection_name_and_qdrant_url_not_a_placeholder(self):
        args = _init_args(
            self.target_repo, collection_name="custom-collection", qdrant_url="http://localhost:9999"
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        output = buf.getvalue()
        self.assertIn("--collection custom-collection", output)
        self.assertIn("--qdrant-url http://localhost:9999", output)
        self.assertNotIn("<name>", output)

    def test_default_collection_name_is_the_derived_slug(self):
        args = _init_args(self.target_repo)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        self.assertIn("--collection my-project", buf.getvalue())

    def test_includes_include_extensions_and_exclude_dirs_when_set(self):
        # Regression for the finding from PR #113 review: the reminder
        # previously omitted these entirely, so copying it indexed a
        # different corpus than the codebase-indexer config this run
        # actually generated.
        args = _init_args(self.target_repo, include_extensions=".py,.md", exclude_dirs="fixtures,generated")
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        output = buf.getvalue()
        self.assertIn("--include-ext .py,.md", output)
        self.assertIn("--exclude-dirs fixtures,generated", output)

    def test_omits_include_ext_and_exclude_dirs_flags_when_unset(self):
        args = _init_args(self.target_repo)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        output = buf.getvalue()
        self.assertNotIn("--include-ext", output)
        self.assertNotIn("--exclude-dirs", output)

    def test_target_repo_path_with_a_space_is_shell_quoted(self):
        # Regression for the finding from PR #113 review: raw interpolation
        # of target_repo would silently split into multiple shell
        # arguments if the path contained a space.
        spacey_repo = self.target_repo.parent / "my project"
        spacey_repo.mkdir()
        args = _init_args(spacey_repo)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        output = buf.getvalue()
        # cmd_init resolves target_repo (Path.resolve()) before using it --
        # match that here rather than the pre-resolve path. Use the same
        # _quote_command helper the production code uses so the assertion
        # uses platform-appropriate quoting (subprocess.list2cmdline on
        # Windows, shlex.quote on POSIX) rather than always expecting
        # POSIX single-quotes even when running on Windows.
        expected = self.mod._quote_command(
            ["--repo-path", str(spacey_repo.resolve())], windows=(os.name == "nt")
        )
        self.assertIn(expected, output)

    def test_windows_uses_cmd_exe_style_quoting_not_posix_single_quotes(self):
        # Regression for the finding from PR #113 review: shlex.quote
        # produces POSIX single-quoted output, which cmd.exe treats as
        # LITERAL characters rather than quoting -- README documents
        # Windows cmd.exe usage too, so a path with a space needs
        # subprocess.list2cmdline's double-quote style there instead.
        #
        # Exercises _quote_command directly (passing windows=True as an
        # explicit parameter) rather than patching os.name on cmd_init --
        # confirmed live that patching os.name globally breaks pathlib's
        # own Path() instantiation on a real POSIX test machine, since
        # cmd_init also resolves target_repo via Path() earlier in the
        # same function.
        argv = ["python", "tools/ingest_to_qdrant.py", "--repo-path", "/tmp/my project", "--dry-run"]
        result = self.mod._quote_command(argv, windows=True)
        self.assertEqual(result, self.mod.subprocess.list2cmdline(argv))
        self.assertNotIn("'/tmp/my project'", result)  # not POSIX-quoted
        self.assertIn('"/tmp/my project"', result)  # cmd.exe-style double quotes

    def test_posix_uses_shlex_quote(self):
        argv = ["python", "tools/ingest_to_qdrant.py", "--repo-path", "/tmp/my project", "--dry-run"]
        result = self.mod._quote_command(argv, windows=False)
        self.assertEqual(result, " ".join(self.mod.shlex.quote(a) for a in argv))
        self.assertIn("'/tmp/my project'", result)


class HookEnvReminderShownForSkipHooks(unittest.TestCase):
    """Regression for the finding from PR #113 review: --skip-hooks leaves
    settings.json (and any hooks a PRIOR full setup already wrote there)
    completely untouched -- unlike --qdrant-only, it does NOT remove them --
    so the reminder is still just as relevant there. Gating it on
    include_hooks silenced it for --skip-hooks + --track-savings/etc, even
    though any existing hooks would keep running with stale/default values."""

    def setUp(self):
        self.mod = _load_setup_project()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-project"
        self.target_repo.mkdir()

    def test_skip_hooks_with_track_savings_still_shows_the_reminder(self):
        args = _init_args(self.target_repo, skip_hooks=True, track_savings=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        self.assertIn("CLAUDE_RUNWAY_TRACK_SAVINGS=1", buf.getvalue())

    def test_qdrant_only_with_track_savings_does_not_show_the_reminder(self):
        # qdrant-only actively removes toolkit hooks (clean_hooks_if_unused),
        # so there's genuinely nothing left for the reminder to be about.
        args = _init_args(self.target_repo, qdrant_only=True, track_savings=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.mod.cmd_init(args)
        self.assertNotIn("CLAUDE_RUNWAY_TRACK_SAVINGS", buf.getvalue())


class CompactCollectionFlag(unittest.TestCase):
    """--compact-collection: defaults to the shared historical
    'conversation-compacts' when unset (NOT a value derived from the
    resolved collection name -- see the PR #130 review regression guard in
    test_setup_project_lib.py's BuildMcpServers for why deriving it would
    silently orphan every already-configured project's conversation-compact
    history the moment its .mcp.json gets regenerated), and is written
    verbatim when given explicitly."""

    def setUp(self):
        self.mod = _load_setup_project()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-project"
        self.target_repo.mkdir()

    def test_defaults_to_the_shared_historical_name(self):
        args = _init_args(self.target_repo, dry_run=False)
        self.mod.cmd_init(args)
        written = json.loads((self.target_repo / ".mcp.json").read_text(encoding="utf-8"))
        self.assertEqual(
            written["mcpServers"]["local-compress"]["env"]["COMPACT_COLLECTION"], "conversation-compacts"
        )

    def test_rerun_with_no_flag_does_not_drift_between_runs(self):
        # Regression for the finding from PR #130 review: a plain re-run
        # with no new flags must produce the SAME COMPACT_COLLECTION both
        # times on an already-configured project, or its conversation-compact
        # history becomes unreachable (see compress_mcp_server.py's
        # _collections_for_project, which scans for siblings by this exact
        # prefix).
        first_args = _init_args(self.target_repo, dry_run=False)
        self.mod.cmd_init(first_args)
        first_written = json.loads((self.target_repo / ".mcp.json").read_text(encoding="utf-8"))

        second_args = _init_args(self.target_repo, dry_run=False)
        self.mod.cmd_init(second_args)
        second_written = json.loads((self.target_repo / ".mcp.json").read_text(encoding="utf-8"))

        self.assertEqual(
            first_written["mcpServers"]["local-compress"]["env"]["COMPACT_COLLECTION"],
            second_written["mcpServers"]["local-compress"]["env"]["COMPACT_COLLECTION"],
        )

    def test_explicit_value_is_used_verbatim(self):
        args = _init_args(self.target_repo, compact_collection="my-custom-compacts", dry_run=False)
        self.mod.cmd_init(args)
        written = json.loads((self.target_repo / ".mcp.json").read_text(encoding="utf-8"))
        self.assertEqual(
            written["mcpServers"]["local-compress"]["env"]["COMPACT_COLLECTION"], "my-custom-compacts"
        )


class QdrantOnlyAndSkipHooksAreMutuallyExclusive(unittest.TestCase):
    """Regression for the finding from PR #113 review: --qdrant-only
    ACTIVELY REMOVES existing toolkit hooks while --skip-hooks promises to
    leave settings.json completely untouched -- genuinely conflicting
    semantics that were previously accepted together with --qdrant-only
    silently winning. Rejected up front by argparse instead."""

    def test_parse_args_rejects_both_flags_together(self):
        mod = _load_setup_project()
        with mock.patch.object(sys, "argv", ["setup_project.py", "init", "/tmp/whatever", "--qdrant-only", "--skip-hooks"]):
            with self.assertRaises(SystemExit) as ctx:
                mod.parse_args()
        self.assertEqual(ctx.exception.code, 2)  # argparse's standard usage-error exit code


class QdrantOnlyWriteOrderSafety(unittest.TestCase):
    """Regression for the finding from PR #113 review: --qdrant-only wrote
    .mcp.json (removing local-compress) BEFORE settings.json (removing the
    stale hooks). If the second write then failed, local-compress was
    already gone but the stale PreToolUse hook was still active -- denying
    WebFetch and redirecting to a now-unconfigured fetch_url, exactly the
    inconsistent state --qdrant-only exists to prevent. Hook cleanup now
    happens FIRST for --qdrant-only, so a failure on the second write
    instead leaves local-compress still fully configured (a "not yet
    qdrant-only" state, not a broken one)."""

    def setUp(self):
        self.mod = _load_setup_project()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-project"
        self.target_repo.mkdir()
        # Simulate a prior FULL setup (local-compress + hooks) already in place.
        self.mod.cmd_init(_init_args(self.target_repo, dry_run=False))
        self.original_mcp_json = (self.target_repo / ".mcp.json").read_text(encoding="utf-8")
        self.assertIn("local-compress", self.original_mcp_json)

    def test_settings_json_written_before_mcp_json_for_qdrant_only(self):
        write_order = []
        original_write_json = self.mod._write_json

        def _tracking_write_json(path, data):
            write_order.append(path.name)
            original_write_json(path, data)

        with mock.patch.object(self.mod, "_write_json", side_effect=_tracking_write_json):
            self.mod.cmd_init(_init_args(self.target_repo, qdrant_only=True, dry_run=False))

        self.assertEqual(write_order, ["settings.json", ".mcp.json"])

    def test_mcp_json_still_has_local_compress_if_settings_json_write_fails(self):
        original_write_json = self.mod._write_json

        def _fail_on_mcp_json(path, data):
            if path.name == ".mcp.json":
                raise OSError("simulated disk-full error")
            original_write_json(path, data)

        with mock.patch.object(self.mod, "_write_json", side_effect=_fail_on_mcp_json):
            with self.assertRaises(OSError):
                self.mod.cmd_init(_init_args(self.target_repo, qdrant_only=True, dry_run=False))

        # settings.json's write (now first) already succeeded and removed
        # the stale hooks...
        settings = json.loads((self.target_repo / ".claude" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(settings["hooks"]["PreToolUse"], [])
        # ...but .mcp.json's write never got a chance to run, so
        # local-compress is still fully configured -- the safer partial
        # state, not "hooks gone but local-compress still active".
        self.assertEqual((self.target_repo / ".mcp.json").read_text(encoding="utf-8"), self.original_mcp_json)


class PipInstalledVenvPython(unittest.TestCase):
    """`_pip_installed_venv_python` (issue #50): `run_setup`'s own
    `venv_python_path` default assumes the documented clone workflow's
    `<tools-repo>/.venv` sibling directory -- this is what falls back to
    `sys.executable` instead when that assumption doesn't hold, which is
    exactly the case for a `pipx`/`uvx`/plain `pip install claude-runway`
    (no local clone, so no `.venv` sibling to find)."""

    def setUp(self):
        self.mod = _load_setup_project()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_returns_none_when_tools_repo_dir_has_a_venv(self):
        # None tells run_setup to use ITS OWN pure default -- the
        # documented clone workflow, unchanged by this issue.
        fake_tools_repo_dir = Path(self._tmpdir.name)
        (fake_tools_repo_dir / ".venv").mkdir()
        with mock.patch.object(self.mod, "TOOLS_REPO_DIR", fake_tools_repo_dir):
            self.assertIsNone(self.mod._pip_installed_venv_python())

    def test_returns_sys_executable_when_no_venv_present(self):
        fake_tools_repo_dir = Path(self._tmpdir.name)  # no .venv subdir created
        with mock.patch.object(self.mod, "TOOLS_REPO_DIR", fake_tools_repo_dir):
            self.assertEqual(self.mod._pip_installed_venv_python(), Path(self.mod.sys.executable))


class PipInstalledLayoutReminder(unittest.TestCase):
    """cmd_init's printed reminders adapt for a pip/pipx/uvx install (issue
    #50): there's no local clone to say "venv activated" about, and
    claude-runway-ingest is on PATH instead of a `python tools/....py` path
    that wouldn't even exist in that layout. Mocks `_pip_installed_venv_python`
    directly (rather than manipulating a real `.venv` next to this repo's own
    checkout) so this doesn't disturb the real clone-workflow templates/tools
    paths `run_setup` still needs to resolve for the rest of cmd_init to work."""

    def setUp(self):
        self.mod = _load_setup_project()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-project"
        self.target_repo.mkdir()

    def test_pip_installed_layout_uses_the_console_script_reminder(self):
        fake_python = Path("/opt/pipx/venvs/claude-runway/bin/python")
        with mock.patch.object(self.mod, "_pip_installed_venv_python", return_value=fake_python):
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.mod.cmd_init(_init_args(self.target_repo))
            output = buf.getvalue()
        self.assertIn("claude-runway-ingest --repo-path", output)
        self.assertNotIn("venv activated", output)
        self.assertNotIn("ingest_to_qdrant.py", output)

    def test_pip_installed_layout_passes_the_override_into_run_setup(self):
        fake_python = Path("/opt/pipx/venvs/claude-runway/bin/python")
        original_run_setup = self.mod.run_setup
        captured = {}

        def _capture_run_setup(*args, **kwargs):
            captured.update(kwargs)
            return original_run_setup(*args, **kwargs)

        with mock.patch.object(self.mod, "_pip_installed_venv_python", return_value=fake_python), \
             mock.patch.object(self.mod, "run_setup", side_effect=_capture_run_setup):
            self.mod.cmd_init(_init_args(self.target_repo))
        self.assertEqual(captured["venv_python"], fake_python)
        # ...and that override actually lands on the generated qdrant
        # server's command, sibling to mcp-server-qdrant -- not just passed
        # through and ignored.
        mcp_json_path = self.target_repo / ".mcp.json"
        # dry_run=True by default in _init_args, so nothing was written --
        # re-run for real to inspect the actual generated file.
        with mock.patch.object(self.mod, "_pip_installed_venv_python", return_value=fake_python):
            self.mod.cmd_init(_init_args(self.target_repo, dry_run=False))
        mcp_json = json.loads(mcp_json_path.read_text(encoding="utf-8"))
        self.assertEqual(
            mcp_json["mcpServers"]["qdrant"]["command"],
            "/opt/pipx/venvs/claude-runway/bin/mcp-server-qdrant",
        )

    def test_clone_based_layout_keeps_the_original_reminder(self):
        # None (this repo's real TOOLS_REPO_DIR/.venv exists in this test
        # environment -- see Step 15 bootstrap) documents the
        # zero-behavior-change case for the existing documented workflow.
        with mock.patch.object(self.mod, "_pip_installed_venv_python", return_value=None):
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.mod.cmd_init(_init_args(self.target_repo))
            output = buf.getvalue()
        self.assertIn("venv activated", output)
        self.assertIn("ingest_to_qdrant.py", output)


class MainIsTheConsoleScriptEntryPoint(unittest.TestCase):
    """`main()` (issue #50) is what pyproject.toml's `claude-runway-setup`
    console script points `module:function` at -- pinning that it does
    exactly what the `if __name__ == "__main__":` guard always did
    (`parse_args()` then `args.func(args)`), just pulled into a callable so
    there's something for an entry point to reference."""

    def test_main_parses_args_and_dispatches_to_its_func(self):
        mod = _load_setup_project()
        fake_func = mock.MagicMock()
        fake_args = argparse.Namespace(func=fake_func)
        with mock.patch.object(mod, "parse_args", return_value=fake_args) as fake_parse_args:
            mod.main()
        fake_parse_args.assert_called_once_with()
        fake_func.assert_called_once_with(fake_args)


if __name__ == "__main__":
    unittest.main()
