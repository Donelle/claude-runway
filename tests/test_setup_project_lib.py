#!/usr/bin/env python3
"""Tests for libs/setup_project_lib.py (issue #48's setup/init script).

Stdlib-only (unittest, no pytest) and no network -- everything here is pure
dict/path logic over real template files already in this repo (`templates/
mcp.json.template`, `templates/settings.json.template`) plus tmp target
directories, so it needs neither Qdrant nor LM Studio:

    .venv/bin/python -m unittest discover -s tests

Why this file exists: a setup script whose entire point is "stop hand-typing
placeholders wrong" has to be trustworthy about actually finishing the job
(no leftover REPLACE-WITH-*/absolute/path/to/... text) and about not
clobbering config it doesn't own. Two interesting cases pinned here are both
about mcp_server_qdrant_path(): a literal REPLACE-WITH-VENV-PYTHON text
substitution on mcp.json.template's qdrant "command" field (as README's own
Installation step 3 instructs) would produce a broken, doubled
".../bin/python/bin/mcp-server-qdrant" path -- a human following the README
verbatim would produce this same broken config; separately (found in PR #113
review, confirmed by inspecting a real venv's bin/ directory), the actual
installed console-script is HYPHENATED (mcp-server-qdrant), not the
underscored mcp_server_qdrant that's only the Python import package name and
never exists as a file on disk.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

from setup_project_lib import (  # noqa: E402
    build_mcp_servers,
    build_settings_hooks,
    default_collection_name,
    find_unresolved_placeholders,
    load_json,
    mcp_server_qdrant_path,
    merge_mcp_json,
    merge_settings_hooks,
    run_setup,
    strip_toolkit_hooks,
    venv_python_path,
)

MCP_TEMPLATE = load_json(REPO_ROOT / "templates" / "mcp.json.template")
SETTINGS_TEMPLATE = load_json(REPO_ROOT / "templates" / "settings.json.template")


class DefaultCollectionName(unittest.TestCase):
    def test_simple_dir_name_lowercased(self):
        self.assertEqual(default_collection_name(Path("/repos/MyProject")), "myproject")

    def test_punctuation_and_spaces_collapse_to_single_hyphens(self):
        self.assertEqual(default_collection_name(Path("/repos/My Cool App!!")), "my-cool-app")

    def test_empty_after_sanitizing_falls_back(self):
        self.assertEqual(default_collection_name(Path("/repos/___")), "project")


class VenvPythonPath(unittest.TestCase):
    def test_posix_layout(self):
        tools_dir = Path("/home/user/tools/claude-runway")
        self.assertEqual(venv_python_path(tools_dir, windows=False), tools_dir / ".venv" / "bin" / "python")

    def test_windows_layout(self):
        tools_dir = Path("/home/user/tools/claude-runway")
        self.assertEqual(
            venv_python_path(tools_dir, windows=True), tools_dir / ".venv" / "Scripts" / "python.exe"
        )


class McpServerQdrantPath(unittest.TestCase):
    def test_is_a_sibling_of_the_interpreter_not_a_nested_bin(self):
        venv_python = Path("/home/user/tools/claude-runway/.venv/bin/python")
        result = mcp_server_qdrant_path(venv_python)
        self.assertEqual(result, Path("/home/user/tools/claude-runway/.venv/bin/mcp-server-qdrant"))
        # The bug this fixes: a literal text substitution of the raw
        # template's "REPLACE-WITH-VENV-PYTHON/bin/mcp-server-qdrant" using
        # the full interpreter path would produce this doubled path instead.
        self.assertNotIn("python/bin/mcp-server-qdrant", result.as_posix())

    def test_uses_the_hyphenated_console_script_name_not_the_import_package_name(self):
        # Regression for the finding from PR #113 review: pip registers the
        # entry point as HYPHENATED mcp-server-qdrant; the underscored
        # mcp_server_qdrant is only the Python import package name and
        # never exists as a file in the venv's bin/ directory. Confirmed
        # directly by inspecting a real venv's installed console scripts.
        venv_python = Path("/home/user/tools/claude-runway/.venv/bin/python")
        result = mcp_server_qdrant_path(venv_python)
        self.assertEqual(result.name, "mcp-server-qdrant")
        self.assertNotEqual(result.name, "mcp_server_qdrant")

    def test_windows_gets_exe_suffix(self):
        venv_python = Path("/home/user/tools/claude-runway/.venv/Scripts/python.exe")
        result = mcp_server_qdrant_path(venv_python)
        self.assertEqual(
            result, Path("/home/user/tools/claude-runway/.venv/Scripts/mcp-server-qdrant.exe")
        )


class FindUnresolvedPlaceholders(unittest.TestCase):
    def test_detects_replace_with_marker(self):
        self.assertEqual(find_unresolved_placeholders({"a": "REPLACE-WITH-X"}), ["REPLACE-WITH-X"])

    def test_detects_absolute_path_to_marker(self):
        self.assertEqual(
            find_unresolved_placeholders(["/absolute/path/to/thing.py"]), ["/absolute/path/to/thing.py"]
        )

    def test_clean_structure_returns_empty(self):
        self.assertEqual(find_unresolved_placeholders({"a": ["b", {"c": "d"}], "e": 1}), [])


class BuildMcpServers(unittest.TestCase):
    def _build(self, **overrides):
        kwargs = dict(
            collection_name="my-project",
            venv_python=Path("/home/user/tools/claude-runway/.venv/bin/python"),
            tools_repo_dir=Path("/home/user/tools/claude-runway"),
            home_dir=Path("/home/user"),
        )
        kwargs.update(overrides)
        return build_mcp_servers(MCP_TEMPLATE, **kwargs)

    def test_no_placeholders_survive(self):
        servers = self._build()
        self.assertEqual(find_unresolved_placeholders(servers), [])

    def test_qdrant_command_is_the_sibling_script_not_the_interpreter(self):
        servers = self._build()
        self.assertEqual(
            servers["qdrant"]["command"],
            "/home/user/tools/claude-runway/.venv/bin/mcp-server-qdrant",
        )

    def test_collection_name_applied_to_both_qdrant_and_indexer(self):
        servers = self._build()
        self.assertEqual(servers["qdrant"]["env"]["COLLECTION_NAME"], "my-project")
        self.assertEqual(servers["codebase-indexer"]["env"]["COLLECTION_NAME"], "my-project")

    def test_fastembed_cache_path_uses_home_dir_not_a_literal_tilde(self):
        servers = self._build()
        self.assertEqual(
            servers["qdrant"]["env"]["FASTEMBED_CACHE_PATH"],
            "/home/user/.claude/claude-runway/fastembed-cache",
        )

    def test_include_compress_false_omits_the_server(self):
        servers = self._build(include_compress=False)
        self.assertNotIn("local-compress", servers)
        # Omitting it shouldn't disturb the other two.
        self.assertIn("qdrant", servers)
        self.assertIn("codebase-indexer", servers)

    def test_original_template_dict_is_not_mutated(self):
        before = json.dumps(MCP_TEMPLATE, sort_keys=True)
        self._build()
        after = json.dumps(MCP_TEMPLATE, sort_keys=True)
        self.assertEqual(before, after)

    def test_qdrant_api_key_applied_to_all_three_servers(self):
        # Regression for the finding from PR #113 review: an authenticated
        # remote-Qdrant setup needs QDRANT_API_KEY on every server that
        # talks to Qdrant, not just local-compress (which already had the
        # field) -- qdrant/codebase-indexer didn't even have it before.
        servers = self._build(qdrant_api_key="my-secret-key")
        self.assertEqual(servers["qdrant"]["env"]["QDRANT_API_KEY"], "my-secret-key")
        self.assertEqual(servers["codebase-indexer"]["env"]["QDRANT_API_KEY"], "my-secret-key")
        self.assertEqual(servers["local-compress"]["env"]["QDRANT_API_KEY"], "my-secret-key")

    def test_qdrant_api_key_defaults_to_blank_for_unauthenticated_local_qdrant(self):
        servers = self._build()
        self.assertEqual(servers["qdrant"]["env"]["QDRANT_API_KEY"], "")
        self.assertEqual(servers["codebase-indexer"]["env"]["QDRANT_API_KEY"], "")

    def test_compact_collection_defaults_to_the_shared_historical_name(self):
        # Regression guard for the finding from PR #130 review: deriving
        # this default from collection_name (a per-project value) would
        # silently change COMPACT_COLLECTION on every ALREADY-configured
        # project the moment its .mcp.json gets regenerated, since every
        # project's local-compress.env.COMPACT_COLLECTION has only ever
        # held this one constant so far (this function never touched the
        # key at all before PR #130) -- orphaning that project's entire
        # conversation-compact history, since compress_mcp_server.py's
        # sibling-scan/canonical-name computation both key off this exact
        # string. A per-project default is still worth having for NEW
        # setups -- that's what /my-setup-clauderunway's own Q5 computes
        # and passes explicitly instead, the same way it always explicitly
        # passes --collection-name rather than relying on this function's
        # own bare fallback.
        servers = self._build()
        self.assertEqual(servers["local-compress"]["env"]["COMPACT_COLLECTION"], "conversation-compacts")

    def test_compact_collection_explicit_value_used_verbatim(self):
        servers = self._build(compact_collection="my-custom-compacts")
        self.assertEqual(servers["local-compress"]["env"]["COMPACT_COLLECTION"], "my-custom-compacts")

    def test_compact_collection_never_left_blank(self):
        # Regression guard: os.environ.get("COMPACT_COLLECTION", "conversation-compacts")
        # in compress_mcp_server.py only falls back when the key is ABSENT,
        # not when it's present-but-empty -- an explicit "" here would build
        # a collection name with a leading stray hyphen at runtime.
        servers = self._build(compact_collection="")
        self.assertNotEqual(servers["local-compress"]["env"]["COMPACT_COLLECTION"], "")

    def test_memory_bank_server_generated_with_this_projects_repo_tag(self):
        servers = self._build()
        self.assertIn("memory-bank", servers)
        self.assertEqual(servers["memory-bank"]["env"]["MEMORY_BANK_ID"], "my-project")

    def test_memory_bank_id_explicit_value_overrides_collection_name(self):
        # Regression for PR #178 review (fifth pass): an earlier version
        # hardcoded MEMORY_BANK_ID to always equal collection_name with no
        # independent parameter at all, so a project that later changed
        # --collection-name would silently retag all future memories while
        # every existing memory stayed under the old identifier.
        servers = self._build(memory_bank_id="my-preserved-identity")
        self.assertEqual(servers["memory-bank"]["env"]["MEMORY_BANK_ID"], "my-preserved-identity")

    def test_memory_bank_id_defaults_to_collection_name_when_blank(self):
        servers = self._build(collection_name="my-project", memory_bank_id="")
        self.assertEqual(servers["memory-bank"]["env"]["MEMORY_BANK_ID"], "my-project")

    def test_memory_bank_collection_defaults_to_shared_name(self):
        # Deliberately NOT derived from collection_name -- every project on
        # a machine must agree on this exact value for memories to actually
        # be shared/found across projects (issue #175).
        servers = self._build()
        self.assertEqual(servers["memory-bank"]["env"]["MEMORY_BANK_COLLECTION"], "memory-bank")
        self.assertEqual(servers["codebase-indexer"]["env"]["MEMORY_BANK_COLLECTION"], "memory-bank")

    def test_memory_bank_collection_explicit_value_used_verbatim(self):
        servers = self._build(memory_bank_collection="my-custom-memory-bank")
        self.assertEqual(servers["memory-bank"]["env"]["MEMORY_BANK_COLLECTION"], "my-custom-memory-bank")
        self.assertEqual(servers["codebase-indexer"]["env"]["MEMORY_BANK_COLLECTION"], "my-custom-memory-bank")

    def test_memory_bank_collection_never_left_blank(self):
        # Same "present-but-empty defeats os.environ.get's own fallback"
        # class of bug as compact_collection above.
        servers = self._build(memory_bank_collection="")
        self.assertNotEqual(servers["memory-bank"]["env"]["MEMORY_BANK_COLLECTION"], "")

    def test_memory_bank_qdrant_api_key_applied(self):
        servers = self._build(qdrant_api_key="my-secret-key")
        self.assertEqual(servers["memory-bank"]["env"]["QDRANT_API_KEY"], "my-secret-key")


def _find_block_args(hooks: dict, event: str, script_name: str) -> list:
    """Test helper: finds the args list of whichever block under `event`
    references `script_name` by basename -- position-independent, matching
    build_settings_hooks' own basename-matching approach rather than
    assuming a fixed array index (issue #198 added a second block per
    event, so a fixed index would be fragile against future reordering)."""
    for block in hooks.get(event, []):
        for inner in block.get("hooks", []):
            args = inner.get("args") or []
            if args and Path(args[-1]).name == script_name:
                return args
    raise AssertionError(f"no block found for {script_name!r} under {event!r}")


class BuildSettingsHooks(unittest.TestCase):
    def _build(self, include_compress: bool = True):
        return build_settings_hooks(
            SETTINGS_TEMPLATE,
            venv_python=Path("/home/user/tools/claude-runway/.venv/bin/python"),
            tools_repo_dir=Path("/home/user/tools/claude-runway"),
            include_compress=include_compress,
        )

    def test_no_placeholders_survive(self):
        self.assertEqual(find_unresolved_placeholders(self._build()), [])

    def test_each_hook_points_at_its_own_script(self):
        hooks = self._build()
        self.assertEqual(
            _find_block_args(hooks, "PostToolUse", "compress_bash_output.py"),
            ["/home/user/tools/claude-runway/hooks/compress_bash_output.py"],
        )
        self.assertEqual(
            _find_block_args(hooks, "PreToolUse", "redirect_webfetch_to_fetch_url.py"),
            ["/home/user/tools/claude-runway/hooks/redirect_webfetch_to_fetch_url.py"],
        )
        self.assertEqual(
            _find_block_args(hooks, "SessionEnd", "session_end_savings.py"),
            ["/home/user/tools/claude-runway/hooks/session_end_savings.py"],
        )

    def test_core_hook_present_under_both_events_regardless_of_compress(self):
        # Issue #198: record_session_id.py is CORE/base install -- present
        # (and correctly patched) under both PostToolUse and SessionEnd
        # whether or not local-compress is included.
        for include_compress in (True, False):
            hooks = self._build(include_compress=include_compress)
            self.assertEqual(
                _find_block_args(hooks, "PostToolUse", "record_session_id.py"),
                ["/home/user/tools/claude-runway/hooks/record_session_id.py"],
            )
            self.assertEqual(
                _find_block_args(hooks, "SessionEnd", "record_session_id.py"),
                ["/home/user/tools/claude-runway/hooks/record_session_id.py"],
            )

    def test_include_compress_false_drops_compress_gated_blocks(self):
        hooks = self._build(include_compress=False)
        with self.assertRaises(AssertionError):
            _find_block_args(hooks, "PostToolUse", "compress_bash_output.py")
        with self.assertRaises(AssertionError):
            _find_block_args(hooks, "PreToolUse", "redirect_webfetch_to_fetch_url.py")
        with self.assertRaises(AssertionError):
            _find_block_args(hooks, "SessionEnd", "session_end_savings.py")
        # PreToolUse's event key survives as an empty list rather than being
        # deleted -- so a later merge_settings_hooks call still processes
        # this event and strips any STALE compress-gated block a prior full
        # setup left in an existing settings.json (see this module's
        # build_settings_hooks docstring).
        self.assertEqual(hooks["PreToolUse"], [])
        self.assertEqual(find_unresolved_placeholders(hooks), [])


class MergeMcpJson(unittest.TestCase):
    def test_preserves_unrelated_existing_server(self):
        existing = {"mcpServers": {"github": {"command": "some-other-mcp-server"}}}
        merged = merge_mcp_json(existing, {"qdrant": {"command": "x"}})
        self.assertIn("github", merged["mcpServers"])
        self.assertIn("qdrant", merged["mcpServers"])

    def test_overwrites_a_stale_previously_generated_server(self):
        existing = {"mcpServers": {"qdrant": {"command": "stale-path"}}}
        merged = merge_mcp_json(existing, {"qdrant": {"command": "fresh-path"}})
        self.assertEqual(merged["mcpServers"]["qdrant"]["command"], "fresh-path")

    def test_empty_existing_produces_just_the_generated_servers(self):
        merged = merge_mcp_json({}, {"qdrant": {"command": "x"}})
        self.assertEqual(merged, {"mcpServers": {"qdrant": {"command": "x"}}})

    def test_omitting_an_owned_server_this_run_removes_it_from_existing(self):
        # Regression for the finding from PR #113 review: re-running with
        # --qdrant-only after a prior run had configured local-compress
        # previously left the stale local-compress entry in place, since a
        # plain dict .update() only ever adds/overwrites keys it's given
        # and never removes anything absent from generated_servers.
        existing = {
            "mcpServers": {
                "qdrant": {"command": "old"},
                "local-compress": {"command": "old-compress"},
                "github": {"command": "unrelated"},
            }
        }
        generated_servers_qdrant_only = {"qdrant": {"command": "new"}, "codebase-indexer": {"command": "new-indexer"}}
        merged = merge_mcp_json(existing, generated_servers_qdrant_only)
        self.assertNotIn("local-compress", merged["mcpServers"])
        self.assertIn("github", merged["mcpServers"])  # untouched -- not a toolkit-owned key
        self.assertEqual(merged["mcpServers"]["qdrant"]["command"], "new")


class MergeSettingsHooks(unittest.TestCase):
    def test_preserves_unrelated_existing_hook_block(self):
        existing = {"hooks": {"PostToolUse": [{"matcher": "MyOwnTool", "hooks": [{"command": "custom"}]}]}}
        generated = {"PostToolUse": [{"matcher": "Bash", "hooks": [{"command": "x", "args": ["y"]}]}]}
        merged = merge_settings_hooks(existing, generated)
        matchers = {b["matcher"] for b in merged["hooks"]["PostToolUse"]}
        self.assertEqual(matchers, {"MyOwnTool", "Bash"})

    def test_rerunning_does_not_duplicate_the_same_block(self):
        generated = {
            "PostToolUse": [
                {"matcher": "Bash", "hooks": [{"command": "x", "args": ["/tools/hooks/compress_bash_output.py"]}]}
            ]
        }
        once = merge_settings_hooks({}, generated)
        twice = merge_settings_hooks(once, generated)
        self.assertEqual(len(twice["hooks"]["PostToolUse"]), 1)

    def test_rerun_after_tools_repo_moved_replaces_the_stale_block_not_duplicates(self):
        # Regression for the finding from PR #113 review: matching by exact
        # command/args (rather than by script basename) meant a moved
        # tools-repo checkout / recreated venv left the OLD, now-broken
        # block in place alongside a fresh duplicate.
        generated_before_move = {
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"command": "/old/.venv/bin/python", "args": ["/old/hooks/compress_bash_output.py"]}],
                }
            ]
        }
        once = merge_settings_hooks({}, generated_before_move)

        generated_after_move = {
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"command": "/new/.venv/bin/python", "args": ["/new/hooks/compress_bash_output.py"]}],
                }
            ]
        }
        twice = merge_settings_hooks(once, generated_after_move)

        blocks = twice["hooks"]["PostToolUse"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["hooks"][0]["command"], "/new/.venv/bin/python")

    def test_a_users_custom_inner_hook_sharing_a_block_survives(self):
        # Regression for the finding from PR #113 review: a block's "hooks"
        # list can hold more than one entry under the same matcher. The
        # earlier whole-block check discarded the ENTIRE block -- including
        # a user's own unrelated custom hook -- the moment any ONE inner
        # entry matched a toolkit script name.
        existing = {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "Bash|Grep|WebFetch|Glob|WebSearch|mcp__local-compress__.*",
                        "hooks": [
                            {"command": "/old/venv/python", "args": ["/old/hooks/compress_bash_output.py"]},
                            {"command": "my-custom-tool", "args": ["my_custom_hook.py"]},
                        ],
                    }
                ]
            }
        }
        generated = {
            "PostToolUse": [
                {
                    "matcher": "Bash|Grep|WebFetch|Glob|WebSearch|mcp__local-compress__.*",
                    "hooks": [{"command": "/new/venv/python", "args": ["/new/hooks/compress_bash_output.py"]}],
                }
            ]
        }
        merged = merge_settings_hooks(existing, generated)

        all_commands = [h["command"] for b in merged["hooks"]["PostToolUse"] for h in b["hooks"]]
        self.assertIn("my-custom-tool", all_commands)  # survived
        self.assertIn("/new/venv/python", all_commands)  # refreshed
        self.assertNotIn("/old/venv/python", all_commands)  # stale entry replaced


class StripToolkitHooks(unittest.TestCase):
    def test_removes_toolkit_owned_blocks_preserves_unrelated_ones(self):
        existing = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": "MyOwnTool", "hooks": [{"command": "custom", "args": ["my_own.py"]}]},
                    {"matcher": "Bash", "hooks": [{"command": "x", "args": ["/tools/hooks/compress_bash_output.py"]}]},
                ],
                "PreToolUse": [
                    {"matcher": "WebFetch", "hooks": [{"command": "x", "args": ["/tools/hooks/redirect_webfetch_to_fetch_url.py"]}]}
                ],
            }
        }
        cleaned = strip_toolkit_hooks(existing)
        self.assertEqual(len(cleaned["hooks"]["PostToolUse"]), 1)
        self.assertEqual(cleaned["hooks"]["PostToolUse"][0]["matcher"], "MyOwnTool")
        self.assertEqual(cleaned["hooks"]["PreToolUse"], [])

    def test_no_hooks_key_at_all_is_a_no_op(self):
        existing = {"some_other_setting": True}
        self.assertEqual(strip_toolkit_hooks(existing), existing)

    def test_nothing_toolkit_owned_present_is_unchanged(self):
        existing = {"hooks": {"PostToolUse": [{"matcher": "MyOwnTool", "hooks": [{"command": "custom", "args": ["my_own.py"]}]}]}}
        self.assertEqual(strip_toolkit_hooks(existing), existing)

    def test_a_users_custom_inner_hook_sharing_a_block_survives(self):
        # Same regression as MergeSettingsHooks' equivalent test, but for
        # the "actively remove, nothing to replace with" path used by
        # --qdrant-only.
        existing = {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "Bash|Grep|WebFetch|Glob|WebSearch|mcp__local-compress__.*",
                        "hooks": [
                            {"command": "/old/venv/python", "args": ["/old/hooks/compress_bash_output.py"]},
                            {"command": "my-custom-tool", "args": ["my_custom_hook.py"]},
                        ],
                    }
                ]
            }
        }
        cleaned = strip_toolkit_hooks(existing)
        blocks = cleaned["hooks"]["PostToolUse"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["hooks"], [{"command": "my-custom-tool", "args": ["my_custom_hook.py"]}])


class RunSetupEndToEnd(unittest.TestCase):
    """Exercises run_setup() against the REAL templates in this repo, using
    tmp directories so nothing here touches the repo's own working tree."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-target-project"
        self.target_repo.mkdir()

    def test_fresh_target_gets_both_files_with_no_placeholders(self):
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertEqual(find_unresolved_placeholders(result.mcp_json), [])
        self.assertIsNotNone(result.settings_json)
        self.assertEqual(find_unresolved_placeholders(result.settings_json), [])

    def test_collection_name_defaults_from_target_dir_name(self):
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertEqual(
            result.mcp_json["mcpServers"]["qdrant"]["env"]["COLLECTION_NAME"], "my-target-project"
        )

    def test_compact_collection_defaults_to_the_shared_historical_name(self):
        # See BuildMcpServers' equivalent test (PR #130 review regression
        # guard) for why this deliberately does NOT vary with the resolved
        # collection name here -- a re-run of run_setup() with no
        # --compact-collection must be a no-op for this field on an
        # already-configured project, not silently orphan its history.
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertEqual(
            result.mcp_json["mcpServers"]["local-compress"]["env"]["COMPACT_COLLECTION"],
            "conversation-compacts",
        )

    def test_collection_name_colliding_with_general_sentinel_warns(self):
        # Issue #175: a project resolved/named "general" would be
        # indistinguishable from memory-bank's reserved cross-project tag.
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), collection_name="general"
        )
        self.assertTrue(any("general" in c and "WARNING" in c for c in result.changes))

    def test_normal_collection_name_does_not_warn(self):
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertFalse(any("WARNING" in c for c in result.changes))

    def test_general_collision_check_keys_off_memory_bank_id_not_collection_name(self):
        # Regression for PR #178 review (fifth pass): since memory_bank_id
        # is now independently settable, the "general" collision check must
        # track the value that actually ends up as the repo tag -- not
        # collection_name, which is now only its fallback default.
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"),
            collection_name="general", memory_bank_id="my-real-identity",
        )
        self.assertFalse(any("WARNING" in c for c in result.changes))

        result2 = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"),
            collection_name="my-project", memory_bank_id="general",
        )
        self.assertTrue(any("general" in c and "WARNING" in c for c in result2.changes))

    def test_collection_name_colliding_with_memory_bank_collection_warns(self):
        # Regression for PR #178 review: a target repo directory named
        # e.g. "memory-bank" with both COLLECTION_NAME and
        # MEMORY_BANK_COLLECTION left at their defaults resolves them to the
        # SAME string, which makes ingest_mcp_server.py's reserved-name guard
        # refuse every index_repo/sync_repo call for this project.
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), collection_name="memory-bank"
        )
        self.assertTrue(any("MEMORY_BANK_COLLECTION" in c and "WARNING" in c for c in result.changes))

    def test_distinct_memory_bank_collection_avoids_the_warning(self):
        # Same colliding COLLECTION_NAME, but an explicit --memory-bank-collection
        # that differs -- no warning should fire.
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"),
            collection_name="memory-bank", memory_bank_collection="shared-memories",
        )
        self.assertFalse(any("WARNING" in c for c in result.changes))

    def test_rerun_with_no_compact_collection_flag_does_not_drift_between_runs(self):
        # Regression for the finding from PR #130 review: a plain re-run
        # (no new flags -- the exact "safe to re-run" workflow this
        # module's own docstring advertises) of an already-configured
        # project that has only ever used the shared default must produce
        # the SAME value both times, or its stored conversation-compact
        # history becomes unreachable (see compress_mcp_server.py's
        # _collections_for_project, which scans for siblings by this exact
        # prefix). Deriving the default from collection_name would have
        # failed this the moment build_mcp_servers() started resolving
        # COMPACT_COLLECTION at all, since every existing project's
        # .mcp.json holds only the old constant already.
        first = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        (self.target_repo / ".mcp.json").write_text(json.dumps(first.mcp_json), encoding="utf-8")

        second = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertEqual(
            first.mcp_json["mcpServers"]["local-compress"]["env"]["COMPACT_COLLECTION"],
            second.mcp_json["mcpServers"]["local-compress"]["env"]["COMPACT_COLLECTION"],
        )

    def test_qdrant_only_omits_compress_server_and_settings_json(self):
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), include_compress=False, include_hooks=False)
        self.assertNotIn("local-compress", result.mcp_json["mcpServers"])
        self.assertIsNone(result.settings_json)

    def test_cli_style_qdrant_only_still_writes_core_hooks(self):
        # Issue #198: the CLI's --qdrant-only keeps include_hooks=True (only
        # --skip-hooks sets it False) and passes include_compress=False --
        # record_session_id.py (CORE/base install) must still be written,
        # while the compress-gated hooks must not.
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), include_compress=False, include_hooks=True
        )
        self.assertIsNotNone(result.settings_json)
        self.assertEqual(
            _find_block_args(result.settings_json["hooks"], "PostToolUse", "record_session_id.py"),
            [f"{REPO_ROOT.as_posix()}/hooks/record_session_id.py"],
        )
        self.assertEqual(
            _find_block_args(result.settings_json["hooks"], "SessionEnd", "record_session_id.py"),
            [f"{REPO_ROOT.as_posix()}/hooks/record_session_id.py"],
        )
        with self.assertRaises(AssertionError):
            _find_block_args(result.settings_json["hooks"], "PostToolUse", "compress_bash_output.py")
        with self.assertRaises(AssertionError):
            _find_block_args(result.settings_json["hooks"], "PreToolUse", "redirect_webfetch_to_fetch_url.py")
        with self.assertRaises(AssertionError):
            _find_block_args(result.settings_json["hooks"], "SessionEnd", "session_end_savings.py")

    def test_cli_style_qdrant_only_rerun_strips_stale_compress_hooks(self):
        # A prior FULL setup wrote compress-gated hooks; a later --qdrant-only
        # -style rerun (include_hooks=True, include_compress=False) must
        # strip those stale blocks while keeping the core one, all through
        # the normal merge path -- no separate clean_hooks_if_unused branch
        # needed anymore for this case (issue #198).
        first = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        (self.target_repo / ".claude").mkdir()
        (self.target_repo / ".claude" / "settings.json").write_text(
            json.dumps(first.settings_json), encoding="utf-8"
        )

        second = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), include_compress=False, include_hooks=True
        )
        self.assertEqual(len(second.settings_json["hooks"]["PostToolUse"]), 1)
        self.assertEqual(second.settings_json["hooks"]["PreToolUse"], [])
        self.assertEqual(len(second.settings_json["hooks"]["SessionEnd"]), 1)
        self.assertEqual(
            _find_block_args(second.settings_json["hooks"], "PostToolUse", "record_session_id.py"),
            [f"{REPO_ROOT.as_posix()}/hooks/record_session_id.py"],
        )

    def test_merges_with_an_already_existing_mcp_json(self):
        (self.target_repo / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"github": {"command": "unrelated"}}}), encoding="utf-8"
        )
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertIn("github", result.mcp_json["mcpServers"])
        self.assertIn("qdrant", result.mcp_json["mcpServers"])

    def test_rerun_does_not_duplicate_hook_blocks(self):
        first = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        (self.target_repo / ".claude").mkdir()
        (self.target_repo / ".claude" / "settings.json").write_text(
            json.dumps(first.settings_json), encoding="utf-8"
        )
        second = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        # PostToolUse/SessionEnd each carry TWO blocks by default (issue #198's
        # always-on core record_session_id.py block, alongside the
        # compress-gated one); PreToolUse has no core hook, so still just 1.
        self.assertEqual(len(second.settings_json["hooks"]["PostToolUse"]), 2)
        self.assertEqual(len(second.settings_json["hooks"]["PreToolUse"]), 1)
        self.assertEqual(len(second.settings_json["hooks"]["SessionEnd"]), 2)

    def test_generated_mcp_servers_never_includes_an_unrelated_existing_secret(self):
        # Regression for the finding from PR #113 review: the full merged
        # mcp_json legitimately preserves an unrelated pre-existing server
        # (that's the whole point of merge_mcp_json), but a caller
        # PREVIEWING changes must be able to show only what THIS run
        # generated -- never the full merged document -- or an unrelated
        # server's secret sitting in its own env block gets printed right
        # along with it.
        (self.target_repo / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"github": {"command": "x", "env": {"GITHUB_TOKEN": "super-secret-value"}}}}),
            encoding="utf-8",
        )
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))

        # The full merged document correctly still has it (real write must
        # preserve the user's own config)...
        self.assertIn("super-secret-value", json.dumps(result.mcp_json))
        # ...but the generated-only preview must not.
        self.assertNotIn("github", result.generated_mcp_servers)
        self.assertNotIn("super-secret-value", json.dumps(result.generated_mcp_servers))

    def test_rerun_with_qdrant_only_removes_a_previously_configured_compress_server(self):
        first = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertIn("local-compress", first.mcp_json["mcpServers"])
        (self.target_repo / ".mcp.json").write_text(json.dumps(first.mcp_json), encoding="utf-8")

        second = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), include_compress=False, include_hooks=False
        )
        self.assertNotIn("local-compress", second.mcp_json["mcpServers"])
        self.assertIn("qdrant", second.mcp_json["mcpServers"])

    def test_rerun_with_qdrant_only_also_removes_stale_hooks_from_a_prior_full_setup(self):
        # Regression for the finding from PR #113 review: --qdrant-only
        # correctly stopped ADDING local-compress-dependent hooks, but a
        # prior full setup's PostToolUse/PreToolUse/SessionEnd hooks were
        # left completely untouched -- still firing (and PreToolUse still
        # denying WebFetch in favor of a now-unconfigured fetch_url) even
        # though "qdrant only" was just requested.
        first = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        (self.target_repo / ".claude").mkdir()
        (self.target_repo / ".claude" / "settings.json").write_text(
            json.dumps(first.settings_json), encoding="utf-8"
        )

        second = run_setup(
            self.target_repo,
            REPO_ROOT,
            home_dir=Path("/home/user"),
            include_compress=False,
            include_hooks=False,
            clean_hooks_if_unused=True,
        )
        self.assertIsNotNone(second.settings_json)
        self.assertEqual(second.settings_json["hooks"]["PostToolUse"], [])
        self.assertEqual(second.settings_json["hooks"]["PreToolUse"], [])
        self.assertEqual(second.settings_json["hooks"]["SessionEnd"], [])

    def test_qdrant_only_on_a_fresh_target_does_not_create_settings_json(self):
        # clean_hooks_if_unused=True has nothing to do (and shouldn't
        # invent a file) when there was never a settings.json to clean.
        result = run_setup(
            self.target_repo,
            REPO_ROOT,
            home_dir=Path("/home/user"),
            include_compress=False,
            include_hooks=False,
            clean_hooks_if_unused=True,
        )
        self.assertIsNone(result.settings_json)
        self.assertFalse((self.target_repo / ".claude" / "settings.json").exists())

    def test_rerun_after_tools_repo_moved_replaces_stale_hooks_not_duplicates(self):
        first = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        (self.target_repo / ".claude").mkdir()
        (self.target_repo / ".claude" / "settings.json").write_text(
            json.dumps(first.settings_json), encoding="utf-8"
        )

        # Simulate the tools repo having moved to a different absolute path
        # (or the venv having been recreated elsewhere) between runs --
        # templates_dir is pinned to the REAL location so template loading
        # still works even though tools_repo_dir itself is now a fake path.
        moved_tools_repo = REPO_ROOT.parent / "claude-runway-moved"
        second = run_setup(
            self.target_repo,
            moved_tools_repo,
            templates_dir=REPO_ROOT / "templates",
            home_dir=Path("/home/user"),
        )

        post_hooks = second.settings_json["hooks"]["PostToolUse"]
        # Two blocks (core record_session_id.py + compress_bash_output.py),
        # not duplicated -- issue #198 added the second block per event.
        self.assertEqual(len(post_hooks), 2)
        for block in post_hooks:
            # Normalize path separators before comparing: JSON settings may
            # store paths with forward slashes on Windows for cross-platform
            # compatibility, while str(Path) uses the OS separator.
            actual = block["hooks"][0]["args"][0].replace("\\", "/")
            self.assertIn(moved_tools_repo.as_posix(), actual)


if __name__ == "__main__":
    unittest.main()
