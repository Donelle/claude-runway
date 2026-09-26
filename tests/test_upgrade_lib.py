#!/usr/bin/env python3
"""Tests for libs/upgrade_lib.py (issue #225's `upgrade` subcommand).

Stdlib-only (unittest, no pytest) and no network -- exercises the REAL
templates in this repo (`templates/mcp.json.template`/`settings.json.
template`) against tmp target directories, the same pattern
tests/test_setup_project_lib.py's RunSetupEndToEnd uses for run_setup():

    .venv/bin/python -m unittest discover -s tests

Covers, per this ticket's own test-strategy section: each migration's
`detect` in both its "before" and "after" states; `apply` making exactly the
one stated change (diffed against the input); `run_upgrade(auto_yes=True)`
applying every pending migration and skipping ones already applied;
idempotency (running twice leaves nothing pending the second time);
`--dry-run` writing nothing while still reporting what's pending; and a
declined migration reappearing on the next run.

Also pins the specific real-world regression `_apply_record_session_id_
hooks_missing`'s own docstring describes: a project that adopted
local-compress BEFORE #198 shipped has a REAL `compress_bash_output.py`
PostToolUse block and no core block at all -- applying this migration must
ADD the core block without touching that existing one.
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

from upgrade_lib import (  # noqa: E402
    MIGRATIONS,
    MigrationContext,
    is_project_configured,
    pending_migrations,
    run_upgrade,
)

_VENV_PYTHON = Path("/fake/venv/bin/python")


def _ctx() -> MigrationContext:
    return MigrationContext(tools_repo_dir=REPO_ROOT, venv_python=_VENV_PYTHON)


def _migration(migration_id: str):
    return next(m for m in MIGRATIONS if m.id == migration_id)


class MemoryBankServerMissing(unittest.TestCase):
    def test_detect_true_when_memory_bank_key_absent(self):
        migration = _migration("memory-bank-server-missing")
        mcp_json = {"mcpServers": {"qdrant": {}, "codebase-indexer": {}}}
        self.assertTrue(migration.detect(mcp_json, {}))

    def test_detect_false_when_memory_bank_key_present(self):
        migration = _migration("memory-bank-server-missing")
        mcp_json = {"mcpServers": {"qdrant": {}, "memory-bank": {}}}
        self.assertFalse(migration.detect(mcp_json, {}))

    def test_apply_adds_only_the_memory_bank_key(self):
        migration = _migration("memory-bank-server-missing")
        before = {
            "mcpServers": {
                "qdrant": {
                    "command": "/fake/venv/bin/mcp-server-qdrant",
                    "env": {
                        "QDRANT_URL": "http://remote-qdrant:6333",
                        "QDRANT_API_KEY": "real-key",
                        "COLLECTION_NAME": "my-project",
                    },
                },
                "codebase-indexer": {
                    "env": {"COLLECTION_NAME": "my-project", "MEMORY_BANK_COLLECTION": "memory-bank"}
                },
            }
        }
        after_mcp, after_settings = migration.apply(before, {}, _ctx())

        # Every pre-existing key is untouched, byte-for-byte.
        self.assertEqual(after_mcp["mcpServers"]["qdrant"], before["mcpServers"]["qdrant"])
        self.assertEqual(after_mcp["mcpServers"]["codebase-indexer"], before["mcpServers"]["codebase-indexer"])
        self.assertEqual(after_settings, {})

        # The one new key derives QDRANT_URL/QDRANT_API_KEY from the
        # existing qdrant block (not a hardcoded localhost default), and
        # MEMORY_BANK_ID from the existing COLLECTION_NAME.
        mb = after_mcp["mcpServers"]["memory-bank"]
        self.assertEqual(mb["env"]["QDRANT_URL"], "http://remote-qdrant:6333")
        self.assertEqual(mb["env"]["QDRANT_API_KEY"], "real-key")
        self.assertEqual(mb["env"]["MEMORY_BANK_ID"], "my-project")
        self.assertEqual(mb["env"]["MEMORY_BANK_COLLECTION"], "memory-bank")

        # The original dict passed in is never mutated in place.
        self.assertNotIn("memory-bank", before["mcpServers"])

    def test_apply_is_idempotent_via_detect(self):
        migration = _migration("memory-bank-server-missing")
        before = {"mcpServers": {"qdrant": {"env": {"COLLECTION_NAME": "my-project"}}, "codebase-indexer": {"env": {}}}}
        after_mcp, _ = migration.apply(before, {}, _ctx())
        self.assertFalse(migration.detect(after_mcp, {}))

    def test_apply_preserves_a_customized_embedding_model(self):
        # Regression for the finding from Copilot review on PR #227:
        # build_mcp_servers has no embedding_model parameter at all -- every
        # server it generates just inherits mcp.json.template's own
        # hardcoded default. A project whose qdrant/codebase-indexer blocks
        # were hand-customized to a different model (the documented, if
        # manual, way to change this) must get a memory-bank block that
        # matches THAT model, not silently reset to the template default,
        # or it can mismatch whatever model actually created the shared
        # memory-bank collection on this machine.
        migration = _migration("memory-bank-server-missing")
        before = {
            "mcpServers": {
                "qdrant": {"env": {"COLLECTION_NAME": "my-project", "EMBEDDING_MODEL": "custom/other-model"}},
                "codebase-indexer": {"env": {"COLLECTION_NAME": "my-project"}},
            }
        }
        after_mcp, _ = migration.apply(before, {}, _ctx())
        self.assertEqual(after_mcp["mcpServers"]["memory-bank"]["env"]["EMBEDDING_MODEL"], "custom/other-model")

    def test_apply_falls_back_to_the_template_default_when_no_existing_model_is_recorded(self):
        migration = _migration("memory-bank-server-missing")
        before = {
            "mcpServers": {
                "qdrant": {"env": {"COLLECTION_NAME": "my-project"}},
                "codebase-indexer": {"env": {"COLLECTION_NAME": "my-project"}},
            }
        }
        after_mcp, _ = migration.apply(before, {}, _ctx())
        self.assertEqual(
            after_mcp["mcpServers"]["memory-bank"]["env"]["EMBEDDING_MODEL"],
            "sentence-transformers/all-MiniLM-L6-v2",
        )


class RecordSessionIdHooksMissing(unittest.TestCase):
    def test_detect_true_when_no_hook_references_the_script(self):
        migration = _migration("record-session-id-hooks-missing")
        settings_json = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "x", "args": ["/x/hooks/compress_bash_output.py"]}]}
                ]
            }
        }
        self.assertTrue(migration.detect({}, settings_json))

    def test_detect_false_when_already_present(self):
        # Both events (SessionStart + SessionEnd) must be registered for
        # this to count as "already applied" -- see the partial-registration
        # tests below for why each event is checked independently.
        migration = _migration("record-session-id-hooks-missing")
        settings_json = {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "x", "args": ["/x/hooks/record_session_id.py"]}]}
                ],
                "SessionEnd": [
                    {"hooks": [{"type": "command", "command": "x", "args": ["/x/hooks/record_session_id.py"]}]}
                ],
            }
        }
        self.assertFalse(migration.detect({}, settings_json))

    def test_detect_true_on_completely_empty_settings(self):
        migration = _migration("record-session-id-hooks-missing")
        self.assertTrue(migration.detect({}, {}))

    def test_apply_adds_core_blocks_without_touching_existing_compress_hook(self):
        # Regression guard for the exact scenario _apply_record_session_id_
        # hooks_missing's own docstring describes: a pre-#198 project with a
        # REAL compress_bash_output.py PostToolUse block and NO core block.
        # A naive merge_settings_hooks-based apply would delete this block --
        # this must not happen.
        migration = _migration("record-session-id-hooks-missing")
        existing_compress_block = {
            "matcher": "Bash|Grep|WebFetch|Glob|WebSearch",
            "hooks": [
                {
                    "type": "command",
                    "command": "/old/venv/bin/python",
                    "args": ["/old/tools-repo/hooks/compress_bash_output.py"],
                }
            ],
        }
        existing_pretooluse_block = {
            "matcher": "WebFetch",
            "hooks": [
                {
                    "type": "command",
                    "command": "/old/venv/bin/python",
                    "args": ["/old/tools-repo/hooks/redirect_webfetch_to_fetch_url.py"],
                }
            ],
        }
        before_settings = {
            "hooks": {
                "PostToolUse": [copy.deepcopy(existing_compress_block)],
                "PreToolUse": [copy.deepcopy(existing_pretooluse_block)],
            }
        }
        _, after_settings = migration.apply({}, before_settings, _ctx())

        # The pre-existing compress hook (PostToolUse) and the PreToolUse
        # block are both still there, byte-for-byte.
        self.assertIn(existing_compress_block, after_settings["hooks"]["PostToolUse"])
        self.assertEqual(after_settings["hooks"]["PreToolUse"], [existing_pretooluse_block])

        # A NEW core block was appended for record_session_id.py under both
        # SessionStart and SessionEnd.
        for event in ("SessionStart", "SessionEnd"):
            scripts = [
                Path(a).name
                for block in after_settings["hooks"][event]
                for hook in block["hooks"]
                for a in hook["args"]
            ]
            self.assertIn("record_session_id.py", scripts)

        # The core block's command/args are resolved from THIS MigrationContext.
        start_hooks = [h for b in after_settings["hooks"]["SessionStart"] for h in b["hooks"]]
        core_hook = next(h for h in start_hooks if Path(h["args"][-1]).name == "record_session_id.py")
        # build_settings_hooks emits command/args as POSIX strings (.as_posix());
        # use the same form here to avoid Windows backslash vs forward-slash mismatch.
        self.assertEqual(core_hook["command"], _VENV_PYTHON.as_posix())
        self.assertTrue(core_hook["args"][-1].endswith("hooks/record_session_id.py"))

        # The original dict passed in is never mutated in place.
        self.assertEqual(before_settings["hooks"]["PostToolUse"], [existing_compress_block])

    def test_apply_on_empty_settings_creates_hooks_key(self):
        migration = _migration("record-session-id-hooks-missing")
        _, after_settings = migration.apply({}, {}, _ctx())
        scripts = [
            Path(a).name
            for block in after_settings["hooks"]["SessionStart"]
            for hook in block["hooks"]
            for a in hook["args"]
        ]
        self.assertIn("record_session_id.py", scripts)

    def test_detect_true_when_only_session_end_is_registered(self):
        # Regression for the finding from Copilot review on PR #227: a
        # single boolean "does record_session_id.py appear ANYWHERE"
        # treated this partially-configured state as already up to date,
        # even though SessionStart -- the event that actually creates/
        # refreshes the SHADOW_FILE marker -- is missing.
        migration = _migration("record-session-id-hooks-missing")
        settings_json = {
            "hooks": {
                "SessionEnd": [
                    {"hooks": [{"type": "command", "command": "x", "args": ["/x/hooks/record_session_id.py"]}]}
                ]
            }
        }
        self.assertTrue(migration.detect({}, settings_json))

    def test_detect_true_when_only_session_start_is_registered(self):
        # Symmetric case: SessionStart present but SessionEnd absent.
        migration = _migration("record-session-id-hooks-missing")
        settings_json = {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "x", "args": ["/x/hooks/record_session_id.py"]}]}
                ]
            }
        }
        self.assertTrue(migration.detect({}, settings_json))

    def test_detect_true_when_old_posttooluse_wildcard_present_but_sessionstart_absent(self):
        # A project configured by issue #198 (old PostToolUse '.*' design)
        # before issue #231 shipped: has PostToolUse+SessionEnd but not
        # SessionStart -- both _RECORD_SESSION_ID_EVENTS are still missing.
        migration = _migration("record-session-id-hooks-missing")
        settings_json = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": ".*", "hooks": [{"type": "command", "command": "x", "args": ["/x/hooks/record_session_id.py"]}]}
                ],
                "SessionEnd": [
                    {"hooks": [{"type": "command", "command": "x", "args": ["/x/hooks/record_session_id.py"]}]}
                ],
            }
        }
        self.assertTrue(migration.detect({}, settings_json))

    def test_apply_adds_only_the_missing_event_without_duplicating_the_present_one(self):
        migration = _migration("record-session-id-hooks-missing")
        existing_session_end_block = {
            "hooks": [{"type": "command", "command": "/real/venv/bin/python", "args": ["/real/tools-repo/hooks/record_session_id.py"]}]
        }
        before_settings = {"hooks": {"SessionEnd": [copy.deepcopy(existing_session_end_block)]}}

        _, after_settings = migration.apply({}, before_settings, _ctx())

        # SessionEnd is untouched -- still exactly the one pre-existing block.
        self.assertEqual(after_settings["hooks"]["SessionEnd"], [existing_session_end_block])
        # SessionStart got the missing core block added.
        start_scripts = [
            Path(a).name
            for block in after_settings["hooks"]["SessionStart"]
            for hook in block["hooks"]
            for a in hook["args"]
        ]
        self.assertIn("record_session_id.py", start_scripts)
        # detect() now reports this migration as fully applied.
        self.assertFalse(migration.detect({}, after_settings))


class HfHubOfflineMissing(unittest.TestCase):
    def test_detect_true_when_key_absent(self):
        migration = _migration("hf-hub-offline-missing")
        mcp_json = {"mcpServers": {"qdrant": {"env": {"FASTEMBED_CACHE_PATH": "/x"}}}}
        self.assertTrue(migration.detect(mcp_json, {}))

    def test_detect_false_when_present_even_if_blank(self):
        # Present-but-blank is the "already applied, not yet warmed" state --
        # NOT the same as missing entirely.
        migration = _migration("hf-hub-offline-missing")
        mcp_json = {"mcpServers": {"qdrant": {"env": {"HF_HUB_OFFLINE": ""}}}}
        self.assertFalse(migration.detect(mcp_json, {}))

    def test_detect_false_when_present_and_set(self):
        migration = _migration("hf-hub-offline-missing")
        mcp_json = {"mcpServers": {"qdrant": {"env": {"HF_HUB_OFFLINE": "1"}}}}
        self.assertFalse(migration.detect(mcp_json, {}))

    def test_apply_adds_blank_value_only(self):
        migration = _migration("hf-hub-offline-missing")
        before = {"mcpServers": {"qdrant": {"env": {"COLLECTION_NAME": "my-project"}}}}
        after_mcp, after_settings = migration.apply(before, {}, _ctx())
        self.assertEqual(after_mcp["mcpServers"]["qdrant"]["env"]["HF_HUB_OFFLINE"], "")
        self.assertEqual(after_mcp["mcpServers"]["qdrant"]["env"]["COLLECTION_NAME"], "my-project")
        self.assertEqual(after_settings, {})
        self.assertNotIn("HF_HUB_OFFLINE", before["mcpServers"]["qdrant"]["env"])


class RecordSessionIdSessionstart(unittest.TestCase):
    """Tests for the record-session-id-sessionstart migration (issue #231):
    removes the stale PostToolUse '.*' block for record_session_id.py."""

    def test_detect_true_when_posttooluse_wildcard_present_for_record_session_id(self):
        migration = _migration("record-session-id-sessionstart")
        settings_json = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": ".*", "hooks": [{"type": "command", "command": "x", "args": ["/x/hooks/record_session_id.py"]}]}
                ]
            }
        }
        self.assertTrue(migration.detect({}, settings_json))

    def test_detect_false_when_no_wildcard_posttooluse_block(self):
        migration = _migration("record-session-id-sessionstart")
        settings_json = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": "Bash|Grep", "hooks": [{"args": ["/x/hooks/compress_bash_output.py"]}]}
                ],
                "SessionStart": [{"hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}],
                "SessionEnd": [{"hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}],
            }
        }
        self.assertFalse(migration.detect({}, settings_json))

    def test_detect_false_when_wildcard_present_but_for_different_script(self):
        # A user's own '.*' PostToolUse block for an unrelated script must not
        # be detected as the stale record_session_id.py block.
        migration = _migration("record-session-id-sessionstart")
        settings_json = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": ".*", "hooks": [{"args": ["/user/hooks/custom_hook.py"]}]}
                ]
            }
        }
        self.assertFalse(migration.detect({}, settings_json))

    def test_detect_false_on_empty_settings(self):
        migration = _migration("record-session-id-sessionstart")
        self.assertFalse(migration.detect({}, {}))

    # Helper: a minimal SessionStart block to satisfy the safety guard -- the apply
    # function requires SessionStart to be present before it removes the PostToolUse
    # fallback (PR #232 Copilot review finding).
    _SESSION_START_BLOCK = {"hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}

    def test_apply_removes_only_the_wildcard_record_session_id_block(self):
        migration = _migration("record-session-id-sessionstart")
        stale_block = {
            "matcher": ".*",
            "hooks": [{"type": "command", "command": "/old/python", "args": ["/old/hooks/record_session_id.py"]}],
        }
        compress_block = {
            "matcher": "Bash|Grep",
            "hooks": [{"type": "command", "command": "/old/python", "args": ["/old/hooks/compress_bash_output.py"]}],
        }
        before_settings = {
            "hooks": {
                "SessionStart": [copy.deepcopy(self._SESSION_START_BLOCK)],  # guard requires this
                "PostToolUse": [copy.deepcopy(stale_block), copy.deepcopy(compress_block)],
            }
        }
        _, after_settings = migration.apply({}, before_settings, _ctx())

        post_blocks = after_settings["hooks"]["PostToolUse"]
        # The stale '.*' block is gone.
        self.assertNotIn(stale_block, post_blocks)
        # The compress block is untouched.
        self.assertIn(compress_block, post_blocks)
        # The original dict is never mutated in place.
        self.assertIn(stale_block, before_settings["hooks"]["PostToolUse"])

    def test_apply_leaves_empty_posttooluse_list_when_only_block_was_stale(self):
        migration = _migration("record-session-id-sessionstart")
        before_settings = {
            "hooks": {
                "SessionStart": [copy.deepcopy(self._SESSION_START_BLOCK)],
                "PostToolUse": [
                    {"matcher": ".*", "hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}
                ],
            }
        }
        _, after_settings = migration.apply({}, before_settings, _ctx())
        self.assertEqual(after_settings["hooks"]["PostToolUse"], [])

    def test_apply_does_not_touch_user_wildcard_block_for_other_scripts(self):
        migration = _migration("record-session-id-sessionstart")
        user_block = {"matcher": ".*", "hooks": [{"args": ["/user/hooks/my_custom.py"]}]}
        before_settings = {
            "hooks": {
                "SessionStart": [copy.deepcopy(self._SESSION_START_BLOCK)],
                "PostToolUse": [
                    {"matcher": ".*", "hooks": [{"args": ["/x/hooks/record_session_id.py"]}]},
                    copy.deepcopy(user_block),
                ],
            }
        }
        _, after_settings = migration.apply({}, before_settings, _ctx())
        self.assertIn(user_block, after_settings["hooks"]["PostToolUse"])

    def test_apply_is_noop_when_sessionstart_absent(self):
        # Regression guard for the finding from Copilot review on PR #232:
        # if the user declines record-session-id-hooks-missing (the migration that
        # adds SessionStart) but accepts this removal, the apply must be a safe
        # no-op -- leaving the PostToolUse '.*' block in place rather than
        # producing a config with no SHADOW_FILE recorder at all.
        migration = _migration("record-session-id-sessionstart")
        stale_block = {
            "matcher": ".*",
            "hooks": [{"args": ["/x/hooks/record_session_id.py"]}],
        }
        before_settings = {
            "hooks": {
                "PostToolUse": [copy.deepcopy(stale_block)],
                # No SessionStart -- simulates declining record-session-id-hooks-missing
            }
        }
        _, after_settings = migration.apply({}, before_settings, _ctx())

        # The stale block must still be present -- no-op.
        self.assertIn(stale_block, after_settings["hooks"]["PostToolUse"])
        # detect() still returns True (will re-appear when SessionStart is configured).
        self.assertTrue(migration.detect({}, after_settings))

    def test_apply_preserves_unrelated_inner_hook_in_the_same_wildcard_block(self):
        # Regression guard for the finding from Copilot review on PR #232: a
        # user who manually added a second inner hook to the same '.*' block
        # (sharing it with the toolkit's record_session_id.py entry) must not
        # have their custom hook silently deleted by this migration.  Only the
        # record_session_id.py inner hook is removed; the block survives with
        # the remaining custom hook intact.
        migration = _migration("record-session-id-sessionstart")
        user_inner_hook = {"type": "command", "command": "/user/python", "args": ["/user/hooks/my_audit.py"]}
        before_settings = {
            "hooks": {
                "SessionStart": [copy.deepcopy(self._SESSION_START_BLOCK)],  # guard requires this
                "PostToolUse": [
                    {
                        "matcher": ".*",
                        "hooks": [
                            {"type": "command", "command": "/x/python", "args": ["/x/hooks/record_session_id.py"]},
                            copy.deepcopy(user_inner_hook),
                        ],
                    }
                ],
            }
        }
        _, after_settings = migration.apply({}, before_settings, _ctx())

        post_blocks = after_settings["hooks"]["PostToolUse"]
        # The block survived (not dropped entirely) because the user's hook remains.
        self.assertEqual(len(post_blocks), 1)
        remaining_inner = post_blocks[0]["hooks"]
        # record_session_id.py is gone.
        self.assertFalse(any(Path(h["args"][-1]).name == "record_session_id.py" for h in remaining_inner))
        # The user's inner hook is intact.
        self.assertIn(user_inner_hook, remaining_inner)
        # The original dict is never mutated in place.
        self.assertEqual(len(before_settings["hooks"]["PostToolUse"][0]["hooks"]), 2)

    def test_apply_is_idempotent_via_detect(self):
        migration = _migration("record-session-id-sessionstart")
        before_settings = {
            "hooks": {
                "SessionStart": [copy.deepcopy(self._SESSION_START_BLOCK)],  # guard requires this
                "PostToolUse": [
                    {"matcher": ".*", "hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}
                ],
            }
        }
        _, after_settings = migration.apply({}, before_settings, _ctx())
        self.assertFalse(migration.detect({}, after_settings))


class PendingMigrations(unittest.TestCase):
    def test_all_pending_on_a_fully_stale_project(self):
        # A project with the old PostToolUse '.*' block (issue #198 config)
        # plus no memory-bank / HF_HUB_OFFLINE: all four migrations pending.
        mcp_json = {"mcpServers": {"qdrant": {"env": {}}, "codebase-indexer": {"env": {}}}}
        settings_json = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": ".*", "hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}
                ]
            }
        }
        pending = pending_migrations(mcp_json, settings_json)
        self.assertEqual({m.id for m in pending}, {m.id for m in MIGRATIONS})

    def test_none_pending_on_a_fully_up_to_date_project(self):
        mcp_json = {
            "mcpServers": {
                "qdrant": {"env": {"HF_HUB_OFFLINE": ""}},
                "codebase-indexer": {"env": {}},
                "memory-bank": {"env": {}},
            }
        }
        settings_json = {
            "hooks": {
                "SessionStart": [{"hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}],
                "SessionEnd": [{"hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}],
            }
        }
        self.assertEqual(pending_migrations(mcp_json, settings_json), [])


class IsProjectConfigured(unittest.TestCase):
    """`is_project_configured` (Copilot review, PR #227): the CLI-level and
    run_upgrade-internal guard against `upgrade` writing broken partial
    config into a directory that was never `init`'d."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_false_when_mcp_json_does_not_exist(self):
        target = Path(self._tmpdir.name) / "no-mcp-json"
        target.mkdir()
        self.assertFalse(is_project_configured(target))

    def test_false_when_mcp_json_has_no_qdrant_server(self):
        target = Path(self._tmpdir.name) / "foreign-mcp-json"
        target.mkdir()
        (target / ".mcp.json").write_text(json.dumps({"mcpServers": {"some-other-server": {}}}), encoding="utf-8")
        self.assertFalse(is_project_configured(target))

    def test_true_when_qdrant_server_present(self):
        target = Path(self._tmpdir.name) / "real-project"
        target.mkdir()
        (target / ".mcp.json").write_text(json.dumps({"mcpServers": {"qdrant": {}}}), encoding="utf-8")
        self.assertTrue(is_project_configured(target))

    def test_true_for_a_qdrant_only_project_with_no_memory_bank_or_compress(self):
        target = Path(self._tmpdir.name) / "qdrant-only-project"
        target.mkdir()
        (target / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"qdrant": {}, "codebase-indexer": {}}}), encoding="utf-8"
        )
        self.assertTrue(is_project_configured(target))


class RunUpgradeEndToEnd(unittest.TestCase):
    """Exercises run_upgrade() against real target repo directories on disk,
    the same tmpdir pattern RunSetupEndToEnd (test_setup_project_lib.py)
    uses."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-target-project"
        self.target_repo.mkdir()
        self._write_stale_config()

    def _write_stale_config(self):
        # A project configured by issues #175/#198/#221 but BEFORE #231:
        # has memory-bank and HF_HUB_OFFLINE absent (pre-#175/#221), plus
        # the OLD PostToolUse '.*' registration for record_session_id.py
        # (issue #198's original design, superseded by SessionStart in #231).
        # This exercises all four migrations in one end-to-end run.
        (self.target_repo / ".mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "qdrant": {
                            "command": "/old/venv/bin/mcp-server-qdrant",
                            "env": {"QDRANT_URL": "http://localhost:6333", "COLLECTION_NAME": "my-target-project"},
                        },
                        "codebase-indexer": {"env": {"COLLECTION_NAME": "my-target-project"}},
                    }
                }
            ),
            encoding="utf-8",
        )
        (self.target_repo / ".claude").mkdir()
        (self.target_repo / ".claude" / "settings.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "PostToolUse": [
                            {
                                "matcher": ".*",  # stale record_session_id.py registration (issue #198)
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "/old/venv/bin/python",
                                        "args": ["/old/tools-repo/hooks/record_session_id.py"],
                                    }
                                ],
                            },
                            {
                                "matcher": "Bash",
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "/old/venv/bin/python",
                                        "args": ["/old/tools-repo/hooks/compress_bash_output.py"],
                                    }
                                ],
                            },
                        ],
                        "SessionEnd": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "/old/venv/bin/python",
                                        "args": ["/old/tools-repo/hooks/record_session_id.py"],
                                    }
                                ]
                            }
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )

    def _read_configs(self):
        mcp_json = json.loads((self.target_repo / ".mcp.json").read_text(encoding="utf-8"))
        settings_json = json.loads((self.target_repo / ".claude" / "settings.json").read_text(encoding="utf-8"))
        return mcp_json, settings_json

    def test_auto_yes_applies_every_pending_migration(self):
        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        self.assertEqual(set(applied), {m.id for m in MIGRATIONS})

        mcp_json, settings_json = self._read_configs()
        self.assertIn("memory-bank", mcp_json["mcpServers"])
        self.assertEqual(mcp_json["mcpServers"]["qdrant"]["env"]["HF_HUB_OFFLINE"], "")
        # record_session_id.py must now live under SessionStart (not PostToolUse).
        start_scripts = [
            Path(a).name
            for block in settings_json["hooks"]["SessionStart"]
            for hook in block["hooks"]
            for a in hook["args"]
        ]
        self.assertIn("record_session_id.py", start_scripts)
        # compress_bash_output.py's PostToolUse block is still intact.
        post_scripts = [
            Path(a).name
            for block in settings_json["hooks"]["PostToolUse"]
            for hook in block["hooks"]
            for a in hook["args"]
        ]
        self.assertIn("compress_bash_output.py", post_scripts)
        # The stale '.*' PostToolUse block for record_session_id.py was removed.
        wildcard_scripts = [
            Path(a).name
            for block in settings_json["hooks"]["PostToolUse"]
            for hook in block["hooks"]
            for a in hook["args"]
            if block.get("matcher") == ".*"
        ]
        self.assertNotIn("record_session_id.py", wildcard_scripts)

    def test_idempotent_second_run_has_nothing_pending(self):
        run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        second_applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        self.assertEqual(second_applied, [])

    def test_dry_run_writes_nothing_but_reports_pending(self):
        original_mcp = (self.target_repo / ".mcp.json").read_text(encoding="utf-8")
        original_settings = (self.target_repo / ".claude" / "settings.json").read_text(encoding="utf-8")

        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, dry_run=True)

        self.assertEqual(set(applied), {m.id for m in MIGRATIONS})
        self.assertEqual((self.target_repo / ".mcp.json").read_text(encoding="utf-8"), original_mcp)
        self.assertEqual(
            (self.target_repo / ".claude" / "settings.json").read_text(encoding="utf-8"), original_settings
        )

    def test_dry_run_does_not_prompt(self):
        # A prompt_fn that raises would fail the test if run_upgrade ever
        # called it under dry_run -- confirming dry-run truly never asks.
        def _explode(migration, index, total):
            raise AssertionError("dry_run must never prompt")

        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, dry_run=True, prompt_fn=_explode)
        self.assertEqual(set(applied), {m.id for m in MIGRATIONS})

    def test_declining_a_migration_leaves_it_pending_next_run(self):
        declined_id = "hf-hub-offline-missing"

        def _decline_one(migration, index, total):
            return migration.id != declined_id

        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, prompt_fn=_decline_one)
        self.assertNotIn(declined_id, applied)
        self.assertIn("memory-bank-server-missing", applied)

        mcp_json, _ = self._read_configs()
        self.assertNotIn("HF_HUB_OFFLINE", mcp_json["mcpServers"]["qdrant"]["env"])

        # Declined migration reappears as pending on the next run.
        second_pending = pending_migrations(*self._read_configs())
        self.assertIn(declined_id, {m.id for m in second_pending})

    def test_prompt_fn_receives_index_and_total(self):
        seen = []

        def _accept_all(migration, index, total):
            seen.append((index, total))
            return True

        run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, prompt_fn=_accept_all)
        self.assertEqual(len(seen), len(MIGRATIONS))
        totals = {t for _, t in seen}
        self.assertEqual(totals, {len(MIGRATIONS)})
        self.assertEqual([i for i, _ in seen], list(range(1, len(MIGRATIONS) + 1)))

    def test_refuses_a_brand_new_never_configured_project(self):
        # Regression for the finding from Copilot review on PR #227: an
        # earlier version treated a missing/empty .mcp.json the same as
        # "{}", so every migration's detect() fired unconditionally and
        # wrote a BROKEN partial config (a qdrant server block with no
        # command/type at all) for a directory that was never `init`'d.
        # `upgrade` must now refuse outright and write nothing.
        empty_target = Path(self._tmpdir.name) / "brand-new-project"
        empty_target.mkdir()
        applied = run_upgrade(empty_target, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        self.assertEqual(applied, [])
        self.assertFalse((empty_target / ".mcp.json").exists())
        self.assertFalse((empty_target / ".claude").exists())

    def test_refuses_an_mcp_json_with_no_qdrant_server(self):
        # An .mcp.json that exists but has no 'qdrant' key at all (e.g. a
        # hand-edited or otherwise foreign file) is treated the same as
        # "not a toolkit-configured project" -- not a crash, not a partial
        # write.
        target = Path(self._tmpdir.name) / "foreign-mcp-json"
        target.mkdir()
        (target / ".mcp.json").write_text(json.dumps({"mcpServers": {"some-other-server": {}}}), encoding="utf-8")
        applied = run_upgrade(target, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        self.assertEqual(applied, [])
        written = json.loads((target / ".mcp.json").read_text(encoding="utf-8"))
        self.assertEqual(written, {"mcpServers": {"some-other-server": {}}})

    def test_no_pending_migrations_writes_nothing_and_returns_empty(self):
        # Fully up-to-date project -- run once to get there, then track
        # whether a second run touches the files' mtimes at all.
        run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        mcp_mtime = (self.target_repo / ".mcp.json").stat().st_mtime_ns
        settings_mtime = (self.target_repo / ".claude" / "settings.json").stat().st_mtime_ns

        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True)

        self.assertEqual(applied, [])
        self.assertEqual((self.target_repo / ".mcp.json").stat().st_mtime_ns, mcp_mtime)
        self.assertEqual((self.target_repo / ".claude" / "settings.json").stat().st_mtime_ns, settings_mtime)


if __name__ == "__main__":
    unittest.main()
