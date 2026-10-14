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
clobbering config it doesn't own.
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
    default_skills_dir,
    find_unresolved_placeholders,
    load_json,
    merge_mcp_json,
    merge_settings_hooks,
    plan_skill_installs,
    split_skill_frontmatter,
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

    def test_qdrant_runs_this_repos_own_server_under_the_venv_interpreter(self):
        servers = self._build()
        self.assertEqual(servers["qdrant"]["command"], "/home/user/tools/claude-runway/.venv/bin/python")
        self.assertEqual(
            servers["qdrant"]["args"], ["/home/user/tools/claude-runway/tools/qdrant_mcp_server.py"]
        )

    def test_qdrant_block_no_longer_carries_the_third_party_servers_settings(self):
        env = self._build()["qdrant"]["env"]
        for gone in ("HF_HUB_OFFLINE", "FASTEMBED_CACHE_PATH", "TOOL_FIND_DESCRIPTION", "TOOL_STORE_DESCRIPTION"):
            self.assertNotIn(gone, env)

    def test_no_server_command_is_a_console_script(self):
        for name, block in self._build().items():
            self.assertEqual(block["command"], "/home/user/tools/claude-runway/.venv/bin/python", name)
            self.assertTrue(block["args"][0].endswith("_mcp_server.py"), name)

    def test_qdrant_block_carries_the_compact_and_lm_studio_settings(self):
        env = self._build(lmstudio_url="http://lm:1234/v1", lmstudio_model="my-model")["qdrant"]["env"]
        self.assertEqual(env["COMPACT_COLLECTION"], "conversation-compacts")
        self.assertEqual(env["CLAUDE_RUNWAY_LMSTUDIO_URL"], "http://lm:1234/v1")
        self.assertEqual(env["CLAUDE_RUNWAY_LMSTUDIO_MODEL"], "my-model")
        self.assertIn("COMPACT_EMBEDDING_MODEL", env)

    def test_qdrant_only_still_writes_the_lm_studio_settings_into_the_qdrant_block(self):
        # compact_store compresses through LM Studio when it can, so the qdrant
        # block carries the values even with no local-compress server.
        servers = self._build(include_compress=False, lmstudio_model="my-model")
        self.assertNotIn("local-compress", servers)
        self.assertEqual(servers["qdrant"]["env"]["CLAUDE_RUNWAY_LMSTUDIO_MODEL"], "my-model")

    def test_local_compress_block_no_longer_carries_qdrant_or_compact_settings(self):
        env = self._build()["local-compress"]["env"]
        for gone in ("QDRANT_URL", "QDRANT_API_KEY", "COMPACT_COLLECTION", "EMBEDDING_MODEL"):
            self.assertNotIn(gone, env)

    def test_analytics_block_is_opt_in_and_absent_by_default(self):
        self.assertNotIn("analytics", self._build())
        self.assertNotIn("analytics", self._build(track_savings=False, savings_db="/x/s.db", metrics_db="/x/m.db"))

    def test_track_savings_writes_the_analytics_block(self):
        servers = self._build(track_savings=True)
        block = servers["analytics"]
        self.assertEqual(block["command"], "/home/user/tools/claude-runway/.venv/bin/python")
        self.assertEqual(block["args"], ["/home/user/tools/claude-runway/tools/analytics_mcp_server.py"])
        # No DB override: neither key is written (a blank one would mask inheritance).
        self.assertEqual(block["env"], {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"})
        # local-compress keeps its own TRACK_SAVINGS: its footer feeds the tracker.
        self.assertEqual(servers["local-compress"]["env"]["CLAUDE_RUNWAY_TRACK_SAVINGS"], "1")
        self.assertEqual(find_unresolved_placeholders(servers), [])

    def test_analytics_block_passes_through_the_db_overrides(self):
        env = self._build(track_savings=True, savings_db="/x/s.db", metrics_db="/x/m.db")["analytics"]["env"]
        self.assertEqual(env["CLAUDE_RUNWAY_SAVINGS_DB"], "/x/s.db")
        self.assertEqual(env["CLAUDE_RUNWAY_METRICS_DB"], "/x/m.db")

    def test_metrics_db_override_reaches_memory_bank_too(self):
        # memory-bank writes metric_id="memory-bank" into the same store the
        # analytics server reads; both must see the same override.
        servers = self._build(track_savings=True, metrics_db="/x/m.db")
        self.assertEqual(servers["analytics"]["env"]["CLAUDE_RUNWAY_METRICS_DB"], "/x/m.db")
        self.assertEqual(servers["memory-bank"]["env"]["CLAUDE_RUNWAY_METRICS_DB"], "/x/m.db")
        # ...and without tracking there is no analytics block, but memory-bank still gets it.
        servers = self._build(metrics_db="/x/m.db")
        self.assertNotIn("analytics", servers)
        self.assertEqual(servers["memory-bank"]["env"]["CLAUDE_RUNWAY_METRICS_DB"], "/x/m.db")

    def test_memory_bank_has_no_metrics_key_when_not_overridden(self):
        # A blank value would mask the inherited shell value, so it is omitted.
        self.assertNotIn("CLAUDE_RUNWAY_METRICS_DB", self._build(track_savings=True)["memory-bank"]["env"])

    def test_analytics_needs_local_compress(self):
        # The tracker records compression events, so --qdrant-only never gets it.
        self.assertNotIn("analytics", self._build(track_savings=True, include_compress=False))

    def test_analytics_is_owned_so_a_rerun_without_track_savings_removes_it(self):
        existing = {"mcpServers": {"analytics": {"command": "old"}, "my-own": {"command": "x"}}}
        merged = merge_mcp_json(existing, self._build())
        self.assertNotIn("analytics", merged["mcpServers"])
        self.assertIn("my-own", merged["mcpServers"])

    def test_compact_embedding_model_blank_leaves_the_template_default(self):
        env = self._build(compact_embedding_model="")["qdrant"]["env"]
        self.assertEqual(env["COMPACT_EMBEDDING_MODEL"], MCP_TEMPLATE["mcpServers"]["qdrant"]["env"]["COMPACT_EMBEDDING_MODEL"])

    def test_compact_embedding_model_explicit_value_used_verbatim(self):
        env = self._build(compact_embedding_model="BAAI/bge-small-en")["qdrant"]["env"]
        self.assertEqual(env["COMPACT_EMBEDDING_MODEL"], "BAAI/bge-small-en")
        # ... and never leaks into the code-index model.
        self.assertNotEqual(env["EMBEDDING_MODEL"], "BAAI/bge-small-en")

    def test_collection_name_applied_to_both_qdrant_and_indexer(self):
        servers = self._build()
        self.assertEqual(servers["qdrant"]["env"]["COLLECTION_NAME"], "my-project")
        self.assertEqual(servers["codebase-indexer"]["env"]["COLLECTION_NAME"], "my-project")

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
        # talks to Qdrant -- qdrant, codebase-indexer and memory-bank (the
        # local-compress server no longer talks to Qdrant at all).
        servers = self._build(qdrant_api_key="my-secret-key")
        self.assertEqual(servers["qdrant"]["env"]["QDRANT_API_KEY"], "my-secret-key")
        self.assertEqual(servers["codebase-indexer"]["env"]["QDRANT_API_KEY"], "my-secret-key")
        self.assertEqual(servers["memory-bank"]["env"]["QDRANT_API_KEY"], "my-secret-key")

    def test_qdrant_api_key_defaults_to_blank_for_unauthenticated_local_qdrant(self):
        servers = self._build()
        self.assertEqual(servers["qdrant"]["env"]["QDRANT_API_KEY"], "")
        self.assertEqual(servers["codebase-indexer"]["env"]["QDRANT_API_KEY"], "")

    def test_compact_collection_defaults_to_the_shared_historical_name(self):
        # Regression guard for the finding from PR #130 review: deriving
        # this default from collection_name (a per-project value) would
        # silently change COMPACT_COLLECTION on every ALREADY-configured
        # project the moment its .mcp.json gets regenerated, since every
        # project's COMPACT_COLLECTION has only ever
        # held this one constant so far (this function never touched the
        # key at all before PR #130) -- orphaning that project's entire
        # conversation-compact history, since qdrant_mcp_server.py's
        # sibling-scan/canonical-name computation both key off this exact
        # string. A per-project default is still worth having for NEW
        # setups -- that's what /my-setup-clauderunway's own Q5 computes
        # and passes explicitly instead, the same way it always explicitly
        # passes --collection-name rather than relying on this function's
        # own bare fallback.
        servers = self._build()
        self.assertEqual(servers["qdrant"]["env"]["COMPACT_COLLECTION"], "conversation-compacts")

    def test_compact_collection_explicit_value_used_verbatim(self):
        servers = self._build(compact_collection="my-custom-compacts")
        self.assertEqual(servers["qdrant"]["env"]["COMPACT_COLLECTION"], "my-custom-compacts")

    def test_compact_collection_never_left_blank(self):
        # Regression guard: os.environ.get("COMPACT_COLLECTION", "conversation-compacts")
        # in qdrant_mcp_server.py only falls back when the key is ABSENT,
        # not when it's present-but-empty -- an explicit "" here would build
        # a collection name with a leading stray hyphen at runtime.
        servers = self._build(compact_collection="")
        self.assertNotEqual(servers["qdrant"]["env"]["COMPACT_COLLECTION"], "")

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
            _find_block_args(hooks, "PostToolUse", "compress_output.py"),
            ["/home/user/tools/claude-runway/hooks/compress_output.py"],
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
        # Issues #198, #231: record_session_id.py is CORE/base install --
        # present (and correctly patched) under SessionStart and SessionEnd
        # whether or not local-compress is included.  It is NOT in PostToolUse
        # (moved from PostToolUse '.*' to SessionStart in issue #231).
        for include_compress in (True, False):
            hooks = self._build(include_compress=include_compress)
            self.assertEqual(
                _find_block_args(hooks, "SessionStart", "record_session_id.py"),
                ["/home/user/tools/claude-runway/hooks/record_session_id.py"],
            )
            self.assertEqual(
                _find_block_args(hooks, "SessionEnd", "record_session_id.py"),
                ["/home/user/tools/claude-runway/hooks/record_session_id.py"],
            )
            with self.assertRaises(AssertionError):
                _find_block_args(hooks, "PostToolUse", "record_session_id.py")

    def test_include_compress_false_drops_compress_gated_blocks(self):
        hooks = self._build(include_compress=False)
        with self.assertRaises(AssertionError):
            _find_block_args(hooks, "PostToolUse", "compress_output.py")
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
                {"matcher": "Bash", "hooks": [{"command": "x", "args": ["/tools/hooks/compress_output.py"]}]}
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
                    "hooks": [{"command": "/old/.venv/bin/python", "args": ["/old/hooks/compress_output.py"]}],
                }
            ]
        }
        once = merge_settings_hooks({}, generated_before_move)

        generated_after_move = {
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"command": "/new/.venv/bin/python", "args": ["/new/hooks/compress_output.py"]}],
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
                            {"command": "/old/venv/python", "args": ["/old/hooks/compress_output.py"]},
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
                    "hooks": [{"command": "/new/venv/python", "args": ["/new/hooks/compress_output.py"]}],
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
                    {"matcher": "Bash", "hooks": [{"command": "x", "args": ["/tools/hooks/compress_output.py"]}]},
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
                            {"command": "/old/venv/python", "args": ["/old/hooks/compress_output.py"]},
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
            result.mcp_json["mcpServers"]["qdrant"]["env"]["COMPACT_COLLECTION"],
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
        # (no new flags -- the exact "safe to re-run" workflow
        # tools/setup_project.py's own docstring advertises) of an already-configured
        # project that has only ever used the shared default must produce
        # the SAME value both times, or its stored conversation-compact
        # history becomes unreachable (see qdrant_mcp_server.py's
        # _collections_for_project, which scans for siblings by this exact
        # prefix). Deriving the default from collection_name would have
        # failed this the moment build_mcp_servers() started resolving
        # COMPACT_COLLECTION at all, since every existing project's
        # .mcp.json holds only the old constant already.
        first = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        (self.target_repo / ".mcp.json").write_text(json.dumps(first.mcp_json), encoding="utf-8")

        second = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertEqual(
            first.mcp_json["mcpServers"]["qdrant"]["env"]["COMPACT_COLLECTION"],
            second.mcp_json["mcpServers"]["qdrant"]["env"]["COMPACT_COLLECTION"],
        )

    def test_qdrant_only_omits_compress_server_and_settings_json(self):
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), include_compress=False, include_hooks=False)
        self.assertNotIn("local-compress", result.mcp_json["mcpServers"])
        self.assertIsNone(result.settings_json)

    def test_cli_style_qdrant_only_still_writes_core_hooks(self):
        # Issues #198, #231: the CLI's --qdrant-only keeps include_hooks=True
        # (only --skip-hooks sets it False) and passes include_compress=False
        # -- record_session_id.py (CORE/base install) must still be written
        # under SessionStart and SessionEnd, while compress-gated hooks must not.
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), include_compress=False, include_hooks=True
        )
        self.assertIsNotNone(result.settings_json)
        self.assertEqual(
            _find_block_args(result.settings_json["hooks"], "SessionStart", "record_session_id.py"),
            [f"{REPO_ROOT.as_posix()}/hooks/record_session_id.py"],
        )
        self.assertEqual(
            _find_block_args(result.settings_json["hooks"], "SessionEnd", "record_session_id.py"),
            [f"{REPO_ROOT.as_posix()}/hooks/record_session_id.py"],
        )
        with self.assertRaises(AssertionError):
            _find_block_args(result.settings_json["hooks"], "PostToolUse", "record_session_id.py")
        with self.assertRaises(AssertionError):
            _find_block_args(result.settings_json["hooks"], "PostToolUse", "compress_output.py")
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
        # record_session_id.py lives under SessionStart (issue #231), so
        # PostToolUse held only the compress block, now stripped → empty
        # list. SessionStart has the core record_session_id.py
        # block. SessionEnd has record_session_id.py only (no savings hook).
        self.assertEqual(len(second.settings_json["hooks"]["SessionStart"]), 1)
        self.assertEqual(len(second.settings_json["hooks"]["PostToolUse"]), 0)
        self.assertEqual(second.settings_json["hooks"]["PreToolUse"], [])
        self.assertEqual(len(second.settings_json["hooks"]["SessionEnd"]), 1)
        self.assertEqual(
            _find_block_args(second.settings_json["hooks"], "SessionStart", "record_session_id.py"),
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
        # SessionStart: 1 core block (record_session_id.py, issue #231).
        # PostToolUse: 1 compress block (compress_output.py only; core
        #   record_session_id.py moved from PostToolUse to SessionStart in #231).
        # PreToolUse: 1 block (redirect_webfetch_to_fetch_url.py).
        # SessionEnd: 2 blocks (record_session_id.py + session_end_savings.py).
        self.assertEqual(len(second.settings_json["hooks"]["SessionStart"]), 1)
        self.assertEqual(len(second.settings_json["hooks"]["PostToolUse"]), 1)
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

        # PostToolUse: 1 block (compress_output.py); record_session_id.py
        # moved to SessionStart in issue #231 -- no longer in PostToolUse.
        # SessionStart: 1 block (record_session_id.py).
        # Both must reference the moved tools repo path, not the old one.
        post_hooks = second.settings_json["hooks"]["PostToolUse"]
        self.assertEqual(len(post_hooks), 1)
        for block in post_hooks:
            actual = block["hooks"][0]["args"][0].replace("\\", "/")
            self.assertIn(moved_tools_repo.as_posix(), actual)

        start_hooks = second.settings_json["hooks"]["SessionStart"]
        self.assertEqual(len(start_hooks), 1)
        for block in start_hooks:
            actual = block["hooks"][0]["args"][0].replace("\\", "/")
            self.assertIn(moved_tools_repo.as_posix(), actual)

    # --- Issue #279: init re-run must not reset a customized EMBEDDING_MODEL ---

    _EMBED_SERVERS = ("qdrant", "codebase-indexer", "memory-bank")
    _CUSTOM_MODEL = "BAAI/bge-small-en"

    def _models(self, result):
        return {s: result.mcp_json["mcpServers"][s]["env"]["EMBEDDING_MODEL"] for s in self._EMBED_SERVERS}

    def _write_customized_existing_config(self, model):
        first = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        cfg = first.mcp_json
        for s in self._EMBED_SERVERS:
            cfg["mcpServers"][s]["env"]["EMBEDDING_MODEL"] = model
        (self.target_repo / ".mcp.json").write_text(json.dumps(cfg))

    def test_fresh_target_uses_template_default_embedding_model(self):
        default = MCP_TEMPLATE["mcpServers"]["qdrant"]["env"]["EMBEDDING_MODEL"]
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertEqual(set(self._models(result).values()), {default})

    def test_rerun_preserves_customized_embedding_model_in_all_blocks(self):
        self._write_customized_existing_config(self._CUSTOM_MODEL)
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertEqual(set(self._models(result).values()), {self._CUSTOM_MODEL})
        self.assertTrue(any("Preserved the existing EMBEDDING_MODEL" in c for c in result.changes))
        self.assertFalse(any("WARNING" in c for c in result.changes))

    def test_explicit_embedding_model_overrides_existing_and_warns(self):
        self._write_customized_existing_config(self._CUSTOM_MODEL)
        new_model = "sentence-transformers/all-mpnet-base-v2"
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), embedding_model=new_model)
        models = self._models(result)
        # Per-project code collections move to the new model; memory-bank's model is
        # its own value (issue #428) and is untouched.
        self.assertEqual(models["qdrant"], new_model)
        self.assertEqual(models["codebase-indexer"], new_model)
        self.assertEqual(models["memory-bank"], self._CUSTOM_MODEL)
        self.assertTrue(any("EMBEDDING_MODEL changed" in c and "WARNING" in c for c in result.changes))
        self.assertFalse(any("memory-bank EMBEDDING_MODEL changed" in c for c in result.changes))

    def _drop_memory_bank_block(self):
        cfg = json.loads((self.target_repo / ".mcp.json").read_text())
        del cfg["mcpServers"]["memory-bank"]
        (self.target_repo / ".mcp.json").write_text(json.dumps(cfg))

    def _set_memory_bank_model(self, model):
        cfg = json.loads((self.target_repo / ".mcp.json").read_text())
        cfg["mcpServers"]["memory-bank"]["env"]["EMBEDDING_MODEL"] = model
        (self.target_repo / ".mcp.json").write_text(json.dumps(cfg))

    _OTHER_MODEL = "sentence-transformers/all-mpnet-base-v2"

    def test_explicit_model_change_without_memory_bank_block_uses_template_default(self):
        # Issue #428: no memory-bank block means memory-bank's model comes from the
        # template default, never from the code-side model.
        self._write_customized_existing_config(self._CUSTOM_MODEL)
        self._drop_memory_bank_block()
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), embedding_model=self._OTHER_MODEL)
        models = self._models(result)
        self.assertEqual(models["qdrant"], self._OTHER_MODEL)
        self.assertEqual(models["memory-bank"], MCP_TEMPLATE["mcpServers"]["memory-bank"]["env"]["EMBEDDING_MODEL"])

    def test_no_memory_bank_block_does_not_inherit_custom_code_model(self):
        # A re-run (no flags) against a config whose code blocks are customized but
        # which has no memory-bank block must not copy the code-side model across.
        self._write_customized_existing_config(self._CUSTOM_MODEL)
        self._drop_memory_bank_block()
        models = self._models(run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user")))
        self.assertEqual(models["qdrant"], self._CUSTOM_MODEL)
        self.assertEqual(models["memory-bank"], MCP_TEMPLATE["mcpServers"]["memory-bank"]["env"]["EMBEDDING_MODEL"])

    def test_code_side_model_is_never_read_from_memory_bank_block(self):
        # Issue #428: only the memory-bank block has a custom model; the code side
        # must fall back to the template default rather than read it.
        first = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        cfg = first.mcp_json
        cfg["mcpServers"]["memory-bank"]["env"]["EMBEDDING_MODEL"] = self._CUSTOM_MODEL
        for s in ("qdrant", "codebase-indexer"):
            del cfg["mcpServers"][s]["env"]["EMBEDDING_MODEL"]
        (self.target_repo / ".mcp.json").write_text(json.dumps(cfg))
        models = self._models(run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user")))
        default = MCP_TEMPLATE["mcpServers"]["qdrant"]["env"]["EMBEDDING_MODEL"]
        self.assertEqual(models["qdrant"], default)
        self.assertEqual(models["codebase-indexer"], default)
        self.assertEqual(models["memory-bank"], self._CUSTOM_MODEL)

    def test_explicit_memory_bank_model_overrides_and_warns_without_touching_code_side(self):
        self._write_customized_existing_config(self._CUSTOM_MODEL)
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), memory_bank_embedding_model=self._OTHER_MODEL
        )
        models = self._models(result)
        self.assertEqual(models["memory-bank"], self._OTHER_MODEL)
        self.assertEqual(models["qdrant"], self._CUSTOM_MODEL)
        self.assertEqual(models["codebase-indexer"], self._CUSTOM_MODEL)
        self.assertTrue(
            any("memory-bank EMBEDDING_MODEL changed" in c and "WARNING" in c and "EVERY project" in c
                for c in result.changes)
        )
        self.assertFalse(any(c.startswith("WARNING: EMBEDDING_MODEL changed") for c in result.changes))

    def test_explicit_memory_bank_model_same_as_existing_does_not_warn(self):
        self._write_customized_existing_config(self._CUSTOM_MODEL)
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), memory_bank_embedding_model=self._CUSTOM_MODEL
        )
        self.assertFalse(any("WARNING" in c for c in result.changes))

    def test_explicit_memory_bank_model_on_fresh_target_does_not_warn(self):
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), memory_bank_embedding_model=self._CUSTOM_MODEL
        )
        self.assertEqual(self._models(result)["memory-bank"], self._CUSTOM_MODEL)
        self.assertFalse(any("WARNING" in c for c in result.changes))

    def test_rerun_preserves_memory_bank_model_when_code_model_differs(self):
        self._write_customized_existing_config(self._CUSTOM_MODEL)
        self._set_memory_bank_model(self._OTHER_MODEL)
        models = self._models(run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user")))
        self.assertEqual(models["memory-bank"], self._OTHER_MODEL)
        self.assertEqual(models["qdrant"], self._CUSTOM_MODEL)

    def test_explicit_embedding_model_same_as_existing_does_not_warn(self):
        self._write_customized_existing_config(self._CUSTOM_MODEL)
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), embedding_model=self._CUSTOM_MODEL
        )
        self.assertEqual(set(self._models(result).values()), {self._CUSTOM_MODEL})
        self.assertFalse(any("WARNING" in c for c in result.changes))

    def test_explicit_embedding_model_on_fresh_target_does_not_warn(self):
        result = run_setup(
            self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), embedding_model=self._CUSTOM_MODEL
        )
        models = self._models(result)
        self.assertEqual({models["qdrant"], models["codebase-indexer"]}, {self._CUSTOM_MODEL})
        self.assertEqual(models["memory-bank"], MCP_TEMPLATE["mcpServers"]["memory-bank"]["env"]["EMBEDDING_MODEL"])
        self.assertFalse(any("WARNING" in c for c in result.changes))

    def test_existing_config_without_embedding_model_falls_back_to_template(self):
        default = MCP_TEMPLATE["mcpServers"]["qdrant"]["env"]["EMBEDDING_MODEL"]
        (self.target_repo / ".mcp.json").write_text(json.dumps({"mcpServers": {"qdrant": {"env": {}}}}))
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        self.assertEqual(set(self._models(result).values()), {default})

    def test_existing_model_falls_back_to_another_block_when_qdrant_has_none(self):
        # The qdrant block's value wins when present (the issue's stated
        # source of truth); a missing/blank key there falls through to the
        # next toolkit block rather than to the template default.
        first = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        cfg = first.mcp_json
        del cfg["mcpServers"]["qdrant"]["env"]["EMBEDDING_MODEL"]
        cfg["mcpServers"]["codebase-indexer"]["env"]["EMBEDDING_MODEL"] = self._CUSTOM_MODEL
        (self.target_repo / ".mcp.json").write_text(json.dumps(cfg))
        result = run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"))
        models = self._models(result)
        self.assertEqual({models["qdrant"], models["codebase-indexer"]}, {self._CUSTOM_MODEL})

    def test_build_mcp_servers_embedding_model_blank_keeps_template_default(self):
        default = MCP_TEMPLATE["mcpServers"]["qdrant"]["env"]["EMBEDDING_MODEL"]
        servers = build_mcp_servers(
            MCP_TEMPLATE, collection_name="c", venv_python=Path("/v/bin/python"),
            tools_repo_dir=REPO_ROOT, home_dir=Path("/home/user"),
        )
        for s in self._EMBED_SERVERS:
            self.assertEqual(servers[s]["env"]["EMBEDDING_MODEL"], default)


class DefaultSkillsDir(unittest.TestCase):
    def test_is_dot_claude_skills_under_home(self):
        self.assertEqual(default_skills_dir(Path("/home/user")), Path("/home/user/.claude/skills"))


class PlanSkillInstalls(unittest.TestCase):
    """`plan_skill_installs` (issue #205) -- pure, read-only planning over a
    fake tools-repo `skills/` layout (never this real repo's own skills/,
    so these tests don't need updating whenever a skill is added/renamed
    here)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tools_repo_dir = Path(self._tmpdir.name) / "tools-repo"
        self.skills_dir = Path(self._tmpdir.name) / "dest-skills"
        (self.tools_repo_dir / "skills" / "skill-a").mkdir(parents=True)
        (self.tools_repo_dir / "skills" / "skill-a" / "SKILL.md").write_text("skill a v1", encoding="utf-8")
        (self.tools_repo_dir / "skills" / "skill-b").mkdir(parents=True)
        (self.tools_repo_dir / "skills" / "skill-b" / "SKILL.md").write_text("skill b v1", encoding="utf-8")

    def test_missing_skills_dir_in_tools_repo_returns_empty_list(self):
        empty_tools_repo = Path(self._tmpdir.name) / "no-skills-here"
        empty_tools_repo.mkdir()
        self.assertEqual(plan_skill_installs(empty_tools_repo, self.skills_dir), [])

    def test_fresh_dest_reports_install_for_every_skill(self):
        plans = plan_skill_installs(self.tools_repo_dir, self.skills_dir)
        self.assertEqual({p.name for p in plans}, {"skill-a", "skill-b"})
        self.assertTrue(all(p.action == "install" for p in plans))
        # Sorted by name -- stable, deterministic output.
        self.assertEqual([p.name for p in plans], ["skill-a", "skill-b"])

    def test_byte_identical_dest_reports_up_to_date(self):
        (self.skills_dir / "skill-a").mkdir(parents=True)
        (self.skills_dir / "skill-a" / "SKILL.md").write_text("skill a v1", encoding="utf-8")
        plans = plan_skill_installs(self.tools_repo_dir, self.skills_dir)
        by_name = {p.name: p for p in plans}
        self.assertEqual(by_name["skill-a"].action, "up_to_date")
        self.assertEqual(by_name["skill-b"].action, "install")

    def test_differing_dest_reports_update(self):
        (self.skills_dir / "skill-a").mkdir(parents=True)
        (self.skills_dir / "skill-a" / "SKILL.md").write_text("skill a v0-stale", encoding="utf-8")
        plans = plan_skill_installs(self.tools_repo_dir, self.skills_dir)
        by_name = {p.name: p for p in plans}
        self.assertEqual(by_name["skill-a"].action, "update")
        self.assertEqual(by_name["skill-a"].content, "skill a v1")

    def test_directory_without_skill_md_is_skipped(self):
        (self.tools_repo_dir / "skills" / "not-a-skill").mkdir()
        plans = plan_skill_installs(self.tools_repo_dir, self.skills_dir)
        self.assertEqual({p.name for p in plans}, {"skill-a", "skill-b"})

    def test_never_writes_anything_itself(self):
        # Purely a planning step -- the dest directory shouldn't even be
        # created by calling this, let alone populated.
        plan_skill_installs(self.tools_repo_dir, self.skills_dir)
        self.assertFalse(self.skills_dir.exists())

    def test_crlf_source_content_is_preserved_verbatim(self):
        # Regression for Copilot review finding on PR #224: source.read_text()
        # opens in universal-newlines text mode, silently translating a
        # CRLF source's line endings to bare "\n" -- lossy regardless of
        # platform, and (combined with a text-mode write) the root cause of
        # a Windows install perpetually re-reporting "update" for content
        # that never actually changed. Reading raw bytes and decoding
        # keeps the ORIGINAL line endings intact in `content`.
        crlf_path = self.tools_repo_dir / "skills" / "skill-a" / "SKILL.md"
        crlf_path.write_bytes(b"line one\r\nline two\r\n")
        plans = plan_skill_installs(self.tools_repo_dir, self.skills_dir)
        plan_a = next(p for p in plans if p.name == "skill-a")
        self.assertEqual(plan_a.content, "line one\r\nline two\r\n")

    def test_non_utf8_dest_is_reported_as_update_not_a_crash(self):
        # Regression for Copilot review finding on PR #224: a pre-existing
        # dest SKILL.md saved in a non-UTF-8 encoding (e.g. a user's editor
        # defaulting to latin-1, or plain corruption) previously raised
        # UnicodeDecodeError from a decoding comparison, aborting the ENTIRE
        # --install-skills run before the documented "different -> update"
        # path ever got a chance to run for this or any other skill.
        (self.skills_dir / "skill-a").mkdir(parents=True)
        (self.skills_dir / "skill-a" / "SKILL.md").write_bytes(b"\xff\xfe not valid utf-8")
        plans = plan_skill_installs(self.tools_repo_dir, self.skills_dir)
        by_name = {p.name: p for p in plans}
        self.assertEqual(by_name["skill-a"].action, "update")
        self.assertEqual(by_name["skill-b"].action, "install")


class SkillFrontmatterHandling(unittest.TestCase):
    """Issue #449: shipped skills have no frontmatter, installed ones do. The
    installed copy's frontmatter is preserved and only the body is compared,
    so there is no perpetual false "update" and no frontmatter is deleted."""

    FM = b'---\nname: skill-a\ndescription: "hand edited description"\n---\n\n'

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tools_repo_dir = Path(self._tmpdir.name) / "tools-repo"
        self.skills_dir = Path(self._tmpdir.name) / "dest-skills"
        (self.tools_repo_dir / "skills" / "skill-a").mkdir(parents=True)
        self.src = self.tools_repo_dir / "skills" / "skill-a" / "SKILL.md"
        self.src.write_bytes(b"# Skill: a\nbody v2\n")
        (self.skills_dir / "skill-a").mkdir(parents=True)
        self.dest = self.skills_dir / "skill-a" / "SKILL.md"

    def _plan(self):
        return plan_skill_installs(self.tools_repo_dir, self.skills_dir)[0]

    def test_split_roundtrips_and_handles_no_frontmatter(self):
        data = self.FM + b"# Skill: a\n"
        prefix, body = split_skill_frontmatter(data)
        self.assertEqual(prefix + body, data)
        self.assertEqual(body, b"# Skill: a\n")
        self.assertEqual(split_skill_frontmatter(b"# Skill: a\n"), (b"", b"# Skill: a\n"))
        self.assertEqual(split_skill_frontmatter(b""), (b"", b""))

    def test_bom_before_the_opening_fence_is_still_frontmatter(self):
        # Copilot review on PR #451: a Windows editor's UTF-8 BOM must not
        # make the block look absent (which would delete it on refresh).
        bom = b"\xef\xbb\xbf"
        data = bom + self.FM + b"# Skill: a\n"
        prefix, body = split_skill_frontmatter(data)
        self.assertEqual(prefix, bom + self.FM)
        self.assertEqual(body, b"# Skill: a\n")
        self.dest.write_bytes(bom + self.FM + b"stale\n")
        plan = self._plan()
        self.assertEqual(plan.action, "update")
        self.assertEqual(plan.content.encode("utf-8"), bom + self.FM + b"# Skill: a\nbody v2\n")
        self.dest.write_bytes(plan.content.encode("utf-8"))
        self.assertEqual(self._plan().action, "up_to_date")

    def test_unclosed_fence_is_not_frontmatter(self):
        data = b"---\nname: x\n# Skill: a\n"
        self.assertEqual(split_skill_frontmatter(data), (b"", data))

    def test_installed_copy_with_frontmatter_and_same_body_is_up_to_date(self):
        self.dest.write_bytes(self.FM + b"# Skill: a\nbody v2\n")
        plan = self._plan()
        self.assertEqual(plan.action, "up_to_date")
        self.assertTrue(plan.frontmatter_preserved)

    def test_installed_copy_without_frontmatter_and_same_body_is_up_to_date(self):
        self.dest.write_bytes(b"# Skill: a\nbody v2\n")
        plan = self._plan()
        self.assertEqual(plan.action, "up_to_date")
        self.assertFalse(plan.frontmatter_preserved)

    def test_stale_body_updates_and_keeps_the_users_frontmatter(self):
        self.dest.write_bytes(self.FM + b"# Skill: a\nbody v1 mentions EVALUATION.md\n")
        plan = self._plan()
        self.assertEqual(plan.action, "update")
        self.assertEqual(plan.content.encode("utf-8"), self.FM + b"# Skill: a\nbody v2\n")
        self.assertNotIn("EVALUATION.md", plan.content)

    def test_stale_copy_without_frontmatter_gets_none_invented(self):
        self.dest.write_bytes(b"old\n")
        plan = self._plan()
        self.assertEqual(plan.action, "update")
        self.assertEqual(plan.content, "# Skill: a\nbody v2\n")

    def test_hand_edited_body_is_reported_as_update(self):
        self.dest.write_bytes(self.FM + b"# Skill: a\nbody v2 plus my edit\n")
        self.assertEqual(self._plan().action, "update")

    def test_crlf_installed_copy_keeps_crlf_frontmatter(self):
        fm = self.FM.replace(b"\n", b"\r\n")
        self.dest.write_bytes(fm + b"# Skill: a\nbody v2\n")
        self.assertEqual(self._plan().action, "up_to_date")
        self.dest.write_bytes(fm + b"stale\n")
        plan = self._plan()
        self.assertEqual(plan.action, "update")
        self.assertTrue(plan.content.encode("utf-8").startswith(fm))

    def test_non_utf8_frontmatter_round_trips(self):
        fm = b'---\nname: skill-a\ndescription: "caf\xe9"\n---\n\n'
        self.dest.write_bytes(fm + b"stale\n")
        plan = self._plan()
        self.assertEqual(plan.content.encode("utf-8", errors="surrogateescape"), fm + b"# Skill: a\nbody v2\n")

    def test_source_gaining_frontmatter_never_overwrites_the_installed_block(self):
        # Copilot review on PR #451: a shipped skill that later gains its own
        # frontmatter must not replace the installed copy's frontmatter.
        self.src.write_bytes(self.FM + b"body\n")
        installed_fm = b"---\nname: other\ndescription: mine\n---\n\n"
        self.dest.write_bytes(installed_fm + b"stale\n")
        plan = self._plan()
        self.assertEqual(plan.action, "update")
        self.assertEqual(plan.content.encode("utf-8"), installed_fm + b"body\n")
        # ...and once refreshed it is up to date, not a perpetual update.
        self.dest.write_bytes(plan.content.encode("utf-8"))
        self.assertEqual(self._plan().action, "up_to_date")

    def test_source_with_frontmatter_installs_whole_when_installed_has_none(self):
        self.src.write_bytes(self.FM + b"body\n")
        self.dest.write_bytes(b"stale, no frontmatter\n")
        plan = self._plan()
        self.assertEqual(plan.content.encode("utf-8"), self.FM + b"body\n")


class CompactEmbeddingModelOnInit(unittest.TestCase):
    """The compact embedding model is sticky -- an `init` re-run must never move
    an existing install's compacts onto another model. It lives in the `qdrant`
    block's COMPACT_EMBEDDING_MODEL now; installs from before the compact tools
    moved held it in the `local-compress` block's EMBEDDING_MODEL."""

    LEGACY = "BAAI/bge-small-en"

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "proj"
        self.target_repo.mkdir()

    def _model(self, result):
        return result.mcp_json["mcpServers"]["qdrant"]["env"]["COMPACT_EMBEDDING_MODEL"]

    def _write(self, servers):
        (self.target_repo / ".mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")

    def _run(self, **kwargs):
        return run_setup(self.target_repo, REPO_ROOT, home_dir=Path("/home/user"), **kwargs)

    def test_brand_new_install_uses_the_project_model(self):
        result = self._run()
        self.assertEqual(self._model(result), "sentence-transformers/all-MiniLM-L6-v2")

    def test_brand_new_install_with_a_custom_project_model_uses_it_for_compacts_too(self):
        result = self._run(embedding_model="c/d")
        self.assertEqual(self._model(result), "c/d")

    def test_pre_move_install_without_the_key_is_pinned_to_the_legacy_model(self):
        self._write({"qdrant": {"env": {}}, "local-compress": {"env": {"COMPACT_COLLECTION": "conversation-compacts"}}})
        self.assertEqual(self._model(self._run()), self.LEGACY)

    def test_pre_move_value_in_the_local_compress_block_is_carried_over(self):
        self._write({"qdrant": {"env": {"EMBEDDING_MODEL": "a/b"}},
                     "local-compress": {"env": {"EMBEDDING_MODEL": "x/custom"}}})
        result = self._run(embedding_model="c/d")
        self.assertEqual(self._model(result), "x/custom")
        self.assertEqual(result.mcp_json["mcpServers"]["qdrant"]["env"]["EMBEDDING_MODEL"], "c/d")

    def test_the_qdrant_blocks_own_value_wins_over_a_stale_local_compress_one(self):
        self._write({"qdrant": {"env": {"COMPACT_EMBEDDING_MODEL": "x/current"}},
                     "local-compress": {"env": {"EMBEDDING_MODEL": "x/stale"}}})
        self.assertEqual(self._model(self._run()), "x/current")

    def test_a_rerun_keeps_the_value_even_when_the_project_model_changes(self):
        self._write({"qdrant": {"env": {"EMBEDDING_MODEL": "a/b", "COMPACT_EMBEDDING_MODEL": self.LEGACY}}})
        result = self._run(embedding_model="c/d")
        self.assertEqual(self._model(result), self.LEGACY)
        self.assertEqual(result.mcp_json["mcpServers"]["qdrant"]["env"]["EMBEDDING_MODEL"], "c/d")

    def test_qdrant_only_rerun_keeps_the_value_after_local_compress_is_removed(self):
        self._write({"qdrant": {"env": {"COMPACT_EMBEDDING_MODEL": self.LEGACY}}, "local-compress": {"env": {}}})
        result = self._run(include_compress=False)
        self.assertNotIn("local-compress", result.mcp_json["mcpServers"])
        self.assertEqual(self._model(result), self.LEGACY)

    def test_an_existing_config_without_local_compress_is_pinned_to_the_legacy_model(self):
        # A full setup later re-run with --qdrant-only deletes the local-compress
        # block but leaves its compacts in Qdrant, embedded with the legacy model,
        # so a missing block does not prove none was stored. Pinning legacy for a
        # project that never stored one costs nothing; the project model would
        # orphan one that did.
        self._write({"qdrant": {"env": {"EMBEDDING_MODEL": "a/b"}}})
        self.assertEqual(self._model(self._run()), self.LEGACY)

    def test_a_previous_full_then_qdrant_only_config_keeps_the_legacy_model(self):
        self._write({"qdrant": {"command": "/v/bin/mcp-server-qdrant", "env": {"COLLECTION_NAME": "p"}}})
        result = self._run(include_compress=False, embedding_model="c/d")
        self.assertEqual(self._model(result), self.LEGACY)
        self.assertNotIn("local-compress", result.mcp_json["mcpServers"])

    def test_a_config_with_only_unrelated_servers_counts_as_empty(self):
        self._write({"github": {"command": "gh-mcp"}})
        self.assertEqual(self._model(self._run()), "sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()
