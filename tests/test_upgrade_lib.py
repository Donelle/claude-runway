#!/usr/bin/env python3
"""Tests for libs/upgrade_lib.py (the `upgrade` subcommand).

Stdlib-only (unittest, no pytest) and no network -- exercises the REAL
templates in this repo (`templates/mcp.json.template`/`settings.json.
template`) against tmp target directories, the same pattern
tests/test_setup_project_lib.py's RunSetupEndToEnd uses for run_setup():

    .venv/bin/python -m unittest discover -s tests

Covers each migration's
`detect` in both its "before" and "after" states; `apply` making exactly the
one stated change (diffed against the input); `run_upgrade(auto_yes=True)`
applying every pending migration and skipping ones already applied;
idempotency (running twice leaves nothing pending the second time);
`--dry-run` writing nothing while still reporting what's pending; and a
declined migration reappearing on the next run.

Also pins the specific real-world regression `_apply_record_session_id_
hooks_missing`'s own docstring describes: a project that adopted
local-compress BEFORE the session-id hook shipped has a REAL `compress_output.py`
PostToolUse block and no core block at all -- applying this migration must
ADD the core block without touching that existing one.
"""

import copy
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

from qdrant_ingest_lib import MANIFEST_FILENAME, manifest_to_json  # noqa: E402
from setup_project_lib import MalformedConfigError, run_setup  # noqa: E402
from upgrade_lib import (  # noqa: E402
    COMPACT_TOOLS_MATCHER,
    MIGRATIONS,
    REPO_MIGRATIONS,
    MigrationContext,
    is_project_configured,
    pending_migrations,
    pending_repo_migrations,
    run_upgrade,
)

_VENV_PYTHON = Path("/fake/venv/bin/python")


def _ctx() -> MigrationContext:
    return MigrationContext(tools_repo_dir=REPO_ROOT, venv_python=_VENV_PYTHON)


def _migration(migration_id: str):
    return next(m for m in MIGRATIONS if m.id == migration_id)


# The compress hook's matcher as `init` wrote it before the compact tools moved
# to the qdrant server: it lists the local-compress tools, not the compact tools.
_OLD_MATCHER = "Bash|Grep|WebFetch|Glob|WebSearch|mcp__local-compress__.*"


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
        # Pins the CURRENT, legacy behavior, not a desired
        # contract: the migration copies the code-index block's
        # EMBEDDING_MODEL into the new memory-bank block. That copy came from
        # the Copilot review (build_mcp_servers' embedding_model
        # parameter defaults to blank and the migration doesn't pass it, so
        # without the copy a hand-customized project got the template
        # default). Memory-bank's model is independent of the
        # code-index model, so this assertion should change when that copy is
        # removed.
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


class CodebaseIndexerMemoryBankCollectionMissing(unittest.TestCase):
    """`upgrade` never added MEMORY_BANK_COLLECTION to an existing
    codebase-indexer block, leaving index_repo/sync_repo's memory-bank guard
    protecting only the default name for a project with a custom one."""

    _ID = "codebase-indexer-memory-bank-collection-missing"

    def test_detect_true_when_key_absent(self):
        mcp_json = {"mcpServers": {"codebase-indexer": {"env": {"COLLECTION_NAME": "p"}}}}
        self.assertTrue(_migration(self._ID).detect(mcp_json, {}))

    def test_detect_true_when_indexer_has_no_env_block_at_all(self):
        mcp_json = {"mcpServers": {"codebase-indexer": {}}}
        self.assertTrue(_migration(self._ID).detect(mcp_json, {}))

    def test_detect_false_when_key_present_and_set(self):
        for value in ("memory-bank", "custom-mb"):
            mcp_json = {"mcpServers": {"codebase-indexer": {"env": {"MEMORY_BANK_COLLECTION": value}}}}
            self.assertFalse(_migration(self._ID).detect(mcp_json, {}), value)

    def test_detect_true_when_key_present_but_blank(self):
        # A blank value is what the template writes by default; it leaves the
        # memory-bank guard disabled, so it must count as unconfigured.
        mcp_json = {"mcpServers": {"codebase-indexer": {"env": {"MEMORY_BANK_COLLECTION": ""}}}}
        self.assertTrue(_migration(self._ID).detect(mcp_json, {}))

    def test_detect_false_when_no_codebase_indexer_block(self):
        mcp_json = {"mcpServers": {"qdrant": {"env": {}}, "memory-bank": {"env": {}}}}
        self.assertFalse(_migration(self._ID).detect(mcp_json, {}))
        self.assertFalse(_migration(self._ID).detect({}, {}))

    def test_apply_uses_default_with_default_memory_bank_block(self):
        before = {
            "mcpServers": {
                "codebase-indexer": {"env": {"COLLECTION_NAME": "p"}},
                "memory-bank": {"env": {"MEMORY_BANK_COLLECTION": "memory-bank"}},
            }
        }
        after, after_settings = _migration(self._ID).apply(before, {"x": 1}, _ctx())
        self.assertEqual(
            after["mcpServers"]["codebase-indexer"]["env"],
            {"COLLECTION_NAME": "p", "MEMORY_BANK_COLLECTION": "memory-bank"},
        )
        self.assertEqual(after["mcpServers"]["memory-bank"], before["mcpServers"]["memory-bank"])
        self.assertEqual(after_settings, {"x": 1})
        # Input never mutated in place.
        self.assertNotIn("MEMORY_BANK_COLLECTION", before["mcpServers"]["codebase-indexer"]["env"])

    def test_apply_copies_custom_value_from_memory_bank_block(self):
        before = {
            "mcpServers": {
                "codebase-indexer": {"env": {"COLLECTION_NAME": "p"}},
                "memory-bank": {"env": {"MEMORY_BANK_COLLECTION": "custom-mb"}},
            }
        }
        after, _ = _migration(self._ID).apply(before, {}, _ctx())
        self.assertEqual(after["mcpServers"]["codebase-indexer"]["env"]["MEMORY_BANK_COLLECTION"], "custom-mb")

    def test_apply_defaults_when_no_memory_bank_block(self):
        before = {"mcpServers": {"codebase-indexer": {}}}
        after, _ = _migration(self._ID).apply(before, {}, _ctx())
        self.assertEqual(after["mcpServers"]["codebase-indexer"]["env"], {"MEMORY_BANK_COLLECTION": "memory-bank"})
        self.assertNotIn("memory-bank", after["mcpServers"])

    def test_apply_is_idempotent_via_detect(self):
        migration = _migration(self._ID)
        before = {"mcpServers": {"codebase-indexer": {"env": {}}}}
        after, _ = migration.apply(before, {}, _ctx())
        self.assertFalse(migration.detect(after, {}))


class RecordSessionIdHooksMissing(unittest.TestCase):
    def test_detect_true_when_no_hook_references_the_script(self):
        migration = _migration("record-session-id-hooks-missing")
        settings_json = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "x", "args": ["/x/hooks/compress_output.py"]}]}
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
        # hooks_missing's own docstring describes: a project from before the session-id hook, with a
        # REAL compress_output.py PostToolUse block and NO core block.
        # A naive merge_settings_hooks-based apply would delete this block --
        # this must not happen.
        migration = _migration("record-session-id-hooks-missing")
        existing_compress_block = {
            "matcher": "Bash|Grep|WebFetch|Glob|WebSearch",
            "hooks": [
                {
                    "type": "command",
                    "command": "/old/venv/bin/python",
                    "args": ["/old/tools-repo/hooks/compress_output.py"],
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
        # Regression for the finding from Copilot review: a
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
        # A project configured with the old PostToolUse '.*' design,
        # before SessionStart replaced it: has PostToolUse+SessionEnd but not
        # SessionStart -- the PostToolUse entry doesn't count for that event,
        # so _RECORD_SESSION_ID_EVENTS still has SessionStart missing.
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


class ObsoleteMigrationsAreGone(unittest.TestCase):
    """`hf-hub-offline-missing` and `local-compress-embedding-model-missing` were
    replaced by `qdrant-server-replaced`: the new server needs no HF_HUB_OFFLINE
    (the embedding provider loads from the local cache without the network on
    its own), and the compact embedding model now lives in the qdrant block,
    where that migration carries it over."""

    def test_neither_old_migration_is_offered_any_more(self):
        ids = {m.id for m in MIGRATIONS}
        self.assertNotIn("hf-hub-offline-missing", ids)
        self.assertNotIn("local-compress-embedding-model-missing", ids)

    def test_a_block_without_hf_hub_offline_is_not_pending_on_its_own(self):
        mcp_json = {"mcpServers": {"qdrant": {"command": "/venv/bin/python", "args": ["/t/mcpservers/qdrant_mcp_server.py"],
                                              "env": {"COLLECTION_NAME": "p"}},
                                   "memory-bank": {"env": {}}, "codebase-indexer": {"env": {"MEMORY_BANK_COLLECTION": "m"}}}}
        settings = {"hooks": {"SessionStart": [{"hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}],
                              "SessionEnd": [{"hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}]}}
        self.assertEqual(pending_migrations(mcp_json, settings), [])


class RecordSessionIdSessionstart(unittest.TestCase):
    """Tests for the record-session-id-sessionstart migration:
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
                    {"matcher": "Bash|Grep", "hooks": [{"args": ["/x/hooks/compress_output.py"]}]}
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
    # fallback.
    _SESSION_START_BLOCK = {"hooks": [{"args": ["/x/hooks/record_session_id.py"]}]}

    def test_apply_removes_only_the_wildcard_record_session_id_block(self):
        migration = _migration("record-session-id-sessionstart")
        stale_block = {
            "matcher": ".*",
            "hooks": [{"type": "command", "command": "/old/python", "args": ["/old/hooks/record_session_id.py"]}],
        }
        compress_block = {
            "matcher": "Bash|Grep",
            "hooks": [{"type": "command", "command": "/old/python", "args": ["/old/hooks/compress_output.py"]}],
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
        # Regression guard for the finding from Copilot review:
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
        # Regression guard for the finding from Copilot review: a
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


class QdrantServerReplaced(unittest.TestCase):
    """The `qdrant` block must be rewritten from the third-party console script
    to this repo's own server, with the compact settings moved over from
    `local-compress`, WITHOUT orphaning compacts already stored: they live under
    a model-named vector, so the compact embedding model has to carry over."""

    _ID = "qdrant-server-replaced"
    _SERVER = f"{REPO_ROOT.as_posix()}/mcpservers/qdrant_mcp_server.py"

    def _migration(self):
        return _migration(self._ID)

    def _pre_split(self):
        """A realistic pre-split `.mcp.json`, as `init` wrote it before the move."""
        return {
            "mcpServers": {
                "qdrant": {
                    "command": "/old/venv/bin/mcp-server-qdrant",
                    "type": "stdio",
                    "env": {
                        "QDRANT_URL": "http://remote-qdrant:6333",
                        "QDRANT_API_KEY": "real-key",
                        "COLLECTION_NAME": "my-project",
                        "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
                        "FASTEMBED_CACHE_PATH": "/home/user/.claude/claude-runway/fastembed-cache",
                        "HF_HUB_OFFLINE": "1",
                        "TOOL_FIND_DESCRIPTION": "custom find text",
                        "TOOL_STORE_DESCRIPTION": "custom store text",
                    },
                },
                "codebase-indexer": {"command": "/old/venv/bin/python", "env": {"COLLECTION_NAME": "my-project"}},
                "memory-bank": {"command": "/old/venv/bin/python", "env": {"MEMORY_BANK_ID": "my-project"}},
                "local-compress": {
                    "command": "/old/venv/bin/python",
                    "env": {
                        "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://lm-host:1234/v1",
                        "CLAUDE_RUNWAY_LMSTUDIO_MODEL": "qwen-small",
                        "QDRANT_URL": "http://remote-qdrant:6333",
                        "QDRANT_API_KEY": "real-key",
                        "COMPACT_COLLECTION": "my-compacts",
                        "EMBEDDING_MODEL": "BAAI/bge-small-en",
                        "CLAUDE_RUNWAY_TRACK_SAVINGS": "1",
                    },
                },
            }
        }

    # --- detect ---

    def test_detect_true_for_the_console_script_command(self):
        self.assertTrue(self._migration().detect(self._pre_split(), {}))

    def test_detect_true_for_a_windows_exe_command(self):
        mcp_json = {"mcpServers": {"qdrant": {"command": "C:/tools/.venv/Scripts/mcp-server-qdrant.exe", "env": {}}}}
        self.assertTrue(self._migration().detect(mcp_json, {}))

    def test_detect_true_when_a_launcher_runs_the_package_through_args(self):
        mcp_json = {"mcpServers": {"qdrant": {"command": "uvx", "args": ["mcp-server-qdrant"], "env": {}}}}
        self.assertTrue(self._migration().detect(mcp_json, {}))

    def test_detect_false_once_the_block_runs_the_own_server(self):
        after, _ = self._migration().apply(self._pre_split(), {}, _ctx())
        self.assertFalse(self._migration().detect(after, {}))

    def test_detect_false_without_a_qdrant_block_or_with_a_malformed_one(self):
        for mcp_json in ({}, {"mcpServers": {}}, {"mcpServers": {"qdrant": None}}, {"mcpServers": {"qdrant": "x"}},
                         {"mcpServers": {"qdrant": {"command": 7, "args": "nope"}}}):
            with self.subTest(mcp_json=mcp_json):
                self.assertFalse(self._migration().detect(mcp_json, {}))

    def test_detect_false_for_an_unrelated_custom_launcher(self):
        mcp_json = {"mcpServers": {"qdrant": {"command": "/opt/my-own-qdrant-wrapper", "env": {}}}}
        self.assertFalse(self._migration().detect(mcp_json, {}))

    # --- apply: the rewritten qdrant block ---

    def test_apply_rewrites_command_args_and_env_exactly(self):
        after, after_settings = self._migration().apply(self._pre_split(), {}, _ctx())
        block = after["mcpServers"]["qdrant"]
        self.assertEqual(block["command"], "/fake/venv/bin/python")
        self.assertEqual(block["type"], "stdio")
        self.assertEqual(block["args"], [self._SERVER])
        self.assertEqual(
            block["env"],
            {
                "QDRANT_URL": "http://remote-qdrant:6333",
                "QDRANT_API_KEY": "real-key",
                "COLLECTION_NAME": "my-project",
                "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
                "COMPACT_COLLECTION": "my-compacts",
                "COMPACT_EMBEDDING_MODEL": "BAAI/bge-small-en",
                "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://lm-host:1234/v1",
                "CLAUDE_RUNWAY_LMSTUDIO_MODEL": "qwen-small",
            },
        )
        self.assertEqual(after_settings, {})

    def test_apply_drops_what_the_new_server_does_not_read(self):
        env = self._migration().apply(self._pre_split(), {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
        for gone in ("HF_HUB_OFFLINE", "FASTEMBED_CACHE_PATH", "TOOL_FIND_DESCRIPTION", "TOOL_STORE_DESCRIPTION"):
            self.assertNotIn(gone, env)

    def test_apply_touches_no_other_server_and_never_mutates_its_input(self):
        before = self._pre_split()
        snapshot = copy.deepcopy(before)
        after, _ = self._migration().apply(before, {}, _ctx())
        self.assertEqual(before, snapshot)
        for name in ("codebase-indexer", "memory-bank", "local-compress"):
            self.assertEqual(after["mcpServers"][name], snapshot["mcpServers"][name], name)

    def test_apply_is_idempotent(self):
        once, _ = self._migration().apply(self._pre_split(), {}, _ctx())
        self.assertFalse(self._migration().detect(once, {}))
        twice, _ = self._migration().apply(once, {}, _ctx())
        self.assertEqual(twice, once)

    # --- apply: the compact embedding model must never move ---

    def test_an_install_that_never_set_the_model_is_pinned_to_the_legacy_one(self):
        # Compacts stored before the setting existed were embedded with
        # BAAI/bge-small-en whatever the code-index model is.
        before = self._pre_split()
        del before["mcpServers"]["local-compress"]["env"]["EMBEDDING_MODEL"]
        env = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
        self.assertEqual(env["COMPACT_EMBEDDING_MODEL"], "BAAI/bge-small-en")
        self.assertEqual(env["EMBEDDING_MODEL"], "sentence-transformers/all-mpnet-base-v2")

    def test_a_blank_local_compress_model_counts_as_never_set(self):
        before = self._pre_split()
        before["mcpServers"]["local-compress"]["env"]["EMBEDDING_MODEL"] = "  "
        env = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
        self.assertEqual(env["COMPACT_EMBEDDING_MODEL"], "BAAI/bge-small-en")

    def test_a_model_the_user_set_in_local_compress_is_carried_over(self):
        before = self._pre_split()
        before["mcpServers"]["local-compress"]["env"]["EMBEDDING_MODEL"] = "x/custom-compact-model"
        env = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
        self.assertEqual(env["COMPACT_EMBEDDING_MODEL"], "x/custom-compact-model")

    def test_a_missing_local_compress_block_does_not_prove_no_compact_was_ever_stored(self):
        # A full setup later re-run with `init --qdrant-only` deletes the
        # local-compress block but leaves its compacts in Qdrant, embedded with the
        # legacy model. Guessing the project model here would orphan them; guessing
        # legacy for a project that never stored one costs nothing.
        before = self._pre_split()
        del before["mcpServers"]["local-compress"]
        env = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
        self.assertEqual(env["COMPACT_EMBEDDING_MODEL"], "BAAI/bge-small-en")
        self.assertEqual(env["EMBEDDING_MODEL"], "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(env["COMPACT_COLLECTION"], "conversation-compacts")
        self.assertEqual(env["CLAUDE_RUNWAY_LMSTUDIO_URL"], "http://localhost:1234/v1")
        self.assertEqual(env["CLAUDE_RUNWAY_LMSTUDIO_MODEL"], "")

    # --- apply: the Qdrant URL and key must survive wherever they were kept ---

    def _without_qdrant_connection(self, *names):
        """The pre-split config with URL and key removed from the named blocks."""
        before = self._pre_split()
        for name in names:
            for key in ("QDRANT_URL", "QDRANT_API_KEY"):
                before["mcpServers"][name]["env"].pop(key, None)
        return before

    def test_a_key_kept_only_in_local_compress_survives(self):
        before = self._without_qdrant_connection("qdrant")
        env = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
        self.assertEqual(env["QDRANT_API_KEY"], "real-key")
        self.assertEqual(env["QDRANT_URL"], "http://remote-qdrant:6333")

    def test_a_key_kept_only_in_the_indexer_or_memory_bank_survives(self):
        for holder in ("codebase-indexer", "memory-bank"):
            with self.subTest(holder=holder):
                before = self._without_qdrant_connection("qdrant", "local-compress")
                before["mcpServers"][holder]["env"].update(
                    {"QDRANT_URL": "http://other:6333", "QDRANT_API_KEY": f"key-from-{holder}"}
                )
                env = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
                self.assertEqual(env["QDRANT_API_KEY"], f"key-from-{holder}")
                self.assertEqual(env["QDRANT_URL"], "http://other:6333")

    def test_the_old_qdrant_blocks_own_values_win_over_the_other_blocks(self):
        before = self._pre_split()
        before["mcpServers"]["local-compress"]["env"].update({"QDRANT_URL": "http://stale:6333", "QDRANT_API_KEY": "stale"})
        env = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
        self.assertEqual((env["QDRANT_URL"], env["QDRANT_API_KEY"]), ("http://remote-qdrant:6333", "real-key"))

    def test_a_blank_key_in_the_qdrant_block_falls_through_to_one_that_is_set(self):
        before = self._pre_split()
        before["mcpServers"]["qdrant"]["env"]["QDRANT_API_KEY"] = ""
        env = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
        self.assertEqual(env["QDRANT_API_KEY"], "real-key")

    def test_an_unauthenticated_install_stays_unauthenticated(self):
        before = self._without_qdrant_connection("qdrant", "local-compress")
        env = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
        self.assertEqual(env["QDRANT_API_KEY"], "")
        self.assertEqual(env["QDRANT_URL"], "http://localhost:6333")

    # --- apply: fallbacks for a sparse, hand-built block ---

    def test_collection_and_model_fall_back_to_the_codebase_indexer_block(self):
        before = {
            "mcpServers": {
                "qdrant": {"command": "/old/bin/mcp-server-qdrant", "env": {}},
                "codebase-indexer": {"env": {"COLLECTION_NAME": "from-indexer", "EMBEDDING_MODEL": "x/indexer-model"}},
            }
        }
        env = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]["env"]
        self.assertEqual(env["COLLECTION_NAME"], "from-indexer")
        self.assertEqual(env["EMBEDDING_MODEL"], "x/indexer-model")
        self.assertEqual(env["QDRANT_URL"], "http://localhost:6333")
        self.assertEqual(env["QDRANT_API_KEY"], "")

    def test_a_block_with_no_env_at_all_still_gets_a_complete_one(self):
        before = {"mcpServers": {"qdrant": {"command": "/old/bin/mcp-server-qdrant"}}}
        block = self._migration().apply(before, {}, _ctx())[0]["mcpServers"]["qdrant"]
        self.assertEqual(block["args"], [self._SERVER])
        self.assertEqual(
            set(block["env"]),
            {"QDRANT_URL", "QDRANT_API_KEY", "COLLECTION_NAME", "EMBEDDING_MODEL", "COMPACT_COLLECTION",
             "COMPACT_EMBEDDING_MODEL", "CLAUDE_RUNWAY_LMSTUDIO_URL", "CLAUDE_RUNWAY_LMSTUDIO_MODEL"},
        )

    def test_no_placeholder_survives_in_the_rewritten_block(self):
        from setup_project_lib import find_unresolved_placeholders

        block = self._migration().apply(self._pre_split(), {}, _ctx())[0]["mcpServers"]["qdrant"]
        self.assertEqual(find_unresolved_placeholders(block), [])


class PendingMigrations(unittest.TestCase):
    def test_all_pending_on_a_fully_stale_project(self):
        # A project with the old PostToolUse '.*' block (the original config)
        # plus no memory-bank and the third-party qdrant server: every
        # migration pending.
        mcp_json = {
            "mcpServers": {
                "qdrant": {"command": "/old/venv/bin/mcp-server-qdrant", "env": {}},
                # Still launched from the old tools/ folder: mcp-server-scripts-moved too.
                "codebase-indexer": {"args": ["/old/tools-repo/tools/ingest_mcp_server.py"], "env": {}},
                # Tracking on, no analytics block: analytics-server-missing too;
                # the leftover SAVINGS_DB key: local-compress-savings-db-inert too.
                "local-compress": {"env": {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1", "CLAUDE_RUNWAY_SAVINGS_DB": ""}},
            }
        }
        settings_json = {
            "hooks": {
                "PostToolUse": [
                    {"matcher": ".*", "hooks": [{"args": ["/x/hooks/record_session_id.py"]}]},
                    {"matcher": _OLD_MATCHER, "hooks": [{"args": ["/x/hooks/compress_bash_output.py"]}]},
                ]
            }
        }
        pending = pending_migrations(mcp_json, settings_json)
        self.assertEqual({m.id for m in pending}, {m.id for m in MIGRATIONS})

    def test_none_pending_on_a_fully_up_to_date_project(self):
        mcp_json = {
            "mcpServers": {
                "qdrant": {"command": "/venv/bin/python", "args": ["/t/mcpservers/qdrant_mcp_server.py"], "env": {}},
                "codebase-indexer": {"env": {"MEMORY_BANK_COLLECTION": "memory-bank"}},
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
    """`is_project_configured`: the CLI-level and
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
        # A project configured before SessionStart replaced PostToolUse:
        # has memory-bank absent and the third-party qdrant server, plus
        # the OLD PostToolUse '.*' registration for record_session_id.py
        # (the original design, superseded by SessionStart).
        # This exercises every migration in one end-to-end run.
        (self.target_repo / ".mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "qdrant": {
                            "command": "/old/venv/bin/mcp-server-qdrant",
                            "env": {"QDRANT_URL": "http://localhost:6333", "COLLECTION_NAME": "my-target-project"},
                        },
                        # Launched from the old tools/ folder, before the servers moved
                        # to mcpservers/.
                        "codebase-indexer": {
                            "args": ["/old/tools-repo/tools/ingest_mcp_server.py"],
                            "env": {"COLLECTION_NAME": "my-target-project"},
                        },
                        # No EMBEDDING_MODEL in the local-compress env: compacts were stored
                        # under the legacy model.
                        # Savings tracking is on with a non-default DB, so the analytics
                        # server is offered and carries that DB over.
                        "local-compress": {"env": {
                            "COMPACT_COLLECTION": "conversation-compacts",
                            "CLAUDE_RUNWAY_TRACK_SAVINGS": "1",
                            "CLAUDE_RUNWAY_SAVINGS_DB": "/data/savings.db",
                        }},
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
                                "matcher": ".*",  # stale record_session_id.py registration
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "/old/venv/bin/python",
                                        "args": ["/old/tools-repo/hooks/record_session_id.py"],
                                    }
                                ],
                            },
                            {
                                "matcher": _OLD_MATCHER,
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
        qdrant = mcp_json["mcpServers"]["qdrant"]
        self.assertEqual(qdrant["command"], "/fake/venv/bin/python")
        self.assertTrue(qdrant["args"][0].endswith("mcpservers/qdrant_mcp_server.py"))
        self.assertEqual(qdrant["env"]["COMPACT_EMBEDDING_MODEL"], "BAAI/bge-small-en")
        self.assertEqual(qdrant["env"]["COMPACT_COLLECTION"], "conversation-compacts")
        # The server that was still on the old tools/ path moved with its args
        # otherwise untouched (mcp-server-scripts-moved).
        self.assertEqual(
            mcp_json["mcpServers"]["codebase-indexer"]["args"], ["/old/tools-repo/mcpservers/ingest_mcp_server.py"]
        )
        analytics = mcp_json["mcpServers"]["analytics"]
        self.assertEqual(analytics["command"], "/fake/venv/bin/python")
        self.assertTrue(analytics["args"][0].endswith("mcpservers/analytics_mcp_server.py"))
        self.assertEqual(
            analytics["env"],
            {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1", "CLAUDE_RUNWAY_SAVINGS_DB": "/data/savings.db"},
        )
        # ...and only then was the inert local-compress copy removed,
        # in the same run, leaving the setting local-compress does read.
        compress_env = mcp_json["mcpServers"]["local-compress"]["env"]
        self.assertNotIn("CLAUDE_RUNWAY_SAVINGS_DB", compress_env)
        self.assertEqual(compress_env["CLAUDE_RUNWAY_TRACK_SAVINGS"], "1")
        # record_session_id.py must now live under SessionStart (not PostToolUse).
        start_scripts = [
            Path(a).name
            for block in settings_json["hooks"]["SessionStart"]
            for hook in block["hooks"]
            for a in hook["args"]
        ]
        self.assertIn("record_session_id.py", start_scripts)
        # compress_output.py's PostToolUse block is still intact.
        post_scripts = [
            Path(a).name
            for block in settings_json["hooks"]["PostToolUse"]
            for hook in block["hooks"]
            for a in hook["args"]
        ]
        self.assertIn("compress_output.py", post_scripts)
        # ...and its matcher now also routes the compact tools to it.
        compress_matchers = [
            block["matcher"]
            for block in settings_json["hooks"]["PostToolUse"]
            if any(Path(a).name == "compress_output.py" for hook in block["hooks"] for a in hook["args"])
        ]
        self.assertEqual(compress_matchers, [f"{_OLD_MATCHER}|{COMPACT_TOOLS_MATCHER}"])
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
        declined_id = "qdrant-server-replaced"

        def _decline_one(migration, index, total):
            return migration.id != declined_id

        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, prompt_fn=_decline_one)
        self.assertNotIn(declined_id, applied)
        self.assertIn("memory-bank-server-missing", applied)

        mcp_json, _ = self._read_configs()
        self.assertEqual(mcp_json["mcpServers"]["qdrant"]["command"], "/old/venv/bin/mcp-server-qdrant")

        # Declined migration reappears as pending on the next run.
        second_pending = pending_migrations(*self._read_configs())
        self.assertIn(declined_id, {m.id for m in second_pending})

    def test_guarded_no_op_is_not_reported_as_applied(self):
        # Declining the migration that ADDS SessionStart makes
        # record-session-id-sessionstart's safety guard no-op its removal.
        # It must stay pending, not be counted (and reported) as applied.
        def _decline_add(migration, index, total):
            return migration.id != "record-session-id-hooks-missing"

        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, prompt_fn=_decline_add)
        self.assertNotIn("record-session-id-sessionstart", applied)
        self.assertIn("memory-bank-server-missing", applied)

        _, settings_json = self._read_configs()
        wildcard_blocks = [b for b in settings_json["hooks"]["PostToolUse"] if b.get("matcher") == ".*"]
        self.assertEqual(len(wildcard_blocks), 1, "PostToolUse fallback must survive without SessionStart")
        self.assertIn("record-session-id-sessionstart", {m.id for m in pending_migrations(*self._read_configs())})

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
        # Regression for the finding from Copilot review: an
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


class RunUpgradeWritesOnlyChangedFiles(unittest.TestCase):
    """run_upgrade must write only the documents an applied
    migration actually changed, and must not report a no-op apply as applied."""

    _MCP_COMPACT = '{"mcpServers":{"qdrant":{"command":"/x/bin/mcp-server-qdrant","env":{"COLLECTION_NAME":"p"}}}}'

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "proj"
        self.target_repo.mkdir()
        self.mcp_path = self.target_repo / ".mcp.json"
        self.settings_path = self.target_repo / ".claude" / "settings.json"
        # Deliberately NOT indent-2, so a re-serialization is detectable as a byte change.
        self.mcp_path.write_text(self._MCP_COMPACT, encoding="utf-8")

    def test_mcp_only_migration_does_not_create_settings_json(self):
        # A --skip-hooks project: no .claude/settings.json at all.
        self.assertFalse(self.settings_path.exists())

        applied = run_upgrade(
            self.target_repo,
            REPO_ROOT,
            _VENV_PYTHON,
            prompt_fn=lambda m, i, t: m.id == "qdrant-server-replaced",
        )
        self.assertEqual(applied, ["qdrant-server-replaced"])
        self.assertFalse(self.settings_path.exists(), "settings.json must not be materialized")
        written = json.loads(self.mcp_path.read_text(encoding="utf-8"))
        self.assertEqual(written["mcpServers"]["qdrant"]["command"], "/fake/venv/bin/python")

    def test_settings_only_migration_leaves_mcp_json_bytes_untouched(self):
        self.settings_path.parent.mkdir()
        self.settings_path.write_text("{}", encoding="utf-8")

        applied = run_upgrade(
            self.target_repo,
            REPO_ROOT,
            _VENV_PYTHON,
            prompt_fn=lambda m, i, t: m.id == "record-session-id-hooks-missing",
        )
        self.assertEqual(applied, ["record-session-id-hooks-missing"])
        self.assertEqual(self.mcp_path.read_text(encoding="utf-8"), self._MCP_COMPACT)
        self.assertIn("hooks", json.loads(self.settings_path.read_text(encoding="utf-8")))

    def test_noop_apply_after_declined_prerequisite_is_not_reported_or_written(self):
        # Only the stale PostToolUse '.*' block is present; the prerequisite
        # (record-session-id-hooks-missing) is declined, so the sessionstart
        # migration's guard makes its apply a no-op.
        self.settings_path.parent.mkdir()
        settings_text = json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {"matcher": ".*", "hooks": [{"type": "command", "args": ["/x/hooks/record_session_id.py"]}]}
                    ]
                }
            }
        )
        self.settings_path.write_text(settings_text, encoding="utf-8")

        applied = run_upgrade(
            self.target_repo,
            REPO_ROOT,
            _VENV_PYTHON,
            prompt_fn=lambda m, i, t: m.id == "record-session-id-sessionstart",
        )
        self.assertEqual(applied, [])
        self.assertEqual(self.settings_path.read_text(encoding="utf-8"), settings_text)
        self.assertEqual(self.mcp_path.read_text(encoding="utf-8"), self._MCP_COMPACT)


class IndexManifestOldFormat(unittest.TestCase):
    """The sync manifest is versioned, `sync_repo` ignores an
    old-format one by itself, and `upgrade` removes the stale file and says the
    next sync will re-index. It stays config-only: nothing here embeds anything.

    A fully migrated project is built first (run_upgrade with auto_yes on a
    minimal config), so the only thing left pending is the manifest."""

    _OLD_MANIFEST = json.dumps({"a.py": "hash-a", "docs/b.md": "hash-b"})

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "proj"
        self.target_repo.mkdir()
        (self.target_repo / ".mcp.json").write_text(
            '{"mcpServers":{"qdrant":{"command":"/x/bin/mcp-server-qdrant","env":{"COLLECTION_NAME":"p"}}}}',
            encoding="utf-8",
        )
        (self.target_repo / ".claude").mkdir()
        (self.target_repo / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
        run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        self.manifest = self.target_repo / MANIFEST_FILENAME

    def _ids(self, repo=None):
        return [m.id for m in pending_repo_migrations(repo or self.target_repo)]

    def test_registered_with_a_stable_id(self):
        self.assertEqual([m.id for m in REPO_MIGRATIONS], ["index-manifest-old-format"])

    def test_not_pending_without_a_manifest(self):
        self.assertEqual(self._ids(), [])

    def test_pending_for_an_old_flat_manifest(self):
        self.manifest.write_text(self._OLD_MANIFEST, encoding="utf-8")
        self.assertEqual(self._ids(), ["index-manifest-old-format"])

    def test_not_pending_for_a_new_format_manifest(self):
        self.manifest.write_text(manifest_to_json({"a.py": "hash-a"}), encoding="utf-8")
        self.assertEqual(self._ids(), [])

    def test_not_pending_for_empty_unreadable_or_foreign_manifests(self):
        for text in ("{}", "not json", "[1, 2]", json.dumps({"a.py": 5})):
            with self.subTest(text=text):
                self.manifest.write_text(text, encoding="utf-8")
                self.assertEqual(self._ids(), [])

    def test_dry_run_reports_it_and_deletes_nothing(self):
        self.manifest.write_text(self._OLD_MANIFEST, encoding="utf-8")
        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, dry_run=True)
        self.assertEqual(applied, ["index-manifest-old-format"])
        self.assertTrue(self.manifest.exists())

    def test_auto_yes_deletes_it_and_a_second_run_has_nothing_pending(self):
        self.manifest.write_text(self._OLD_MANIFEST, encoding="utf-8")
        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        self.assertEqual(applied, ["index-manifest-old-format"])
        self.assertFalse(self.manifest.exists())
        self.assertEqual(run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True), [])

    def test_idempotent_against_a_manifest_the_next_sync_wrote(self):
        """The reason the manifest is versioned rather than just deleted once:
        after the rebuild writes a new-format one, `upgrade` must leave it alone."""
        self.manifest.write_text(self._OLD_MANIFEST, encoding="utf-8")
        run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        self.manifest.write_text(manifest_to_json({"a.py": "fresh"}), encoding="utf-8")
        self.assertEqual(run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True), [])
        self.assertTrue(self.manifest.exists())

    def test_declining_keeps_the_file_and_it_stays_pending(self):
        self.manifest.write_text(self._OLD_MANIFEST, encoding="utf-8")
        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, prompt_fn=lambda m, i, t: False)
        self.assertEqual(applied, [])
        self.assertTrue(self.manifest.exists())
        self.assertEqual(self._ids(), ["index-manifest-old-format"])

    def test_is_numbered_after_the_config_migrations_and_prompted_like_them(self):
        self.manifest.write_text(self._OLD_MANIFEST, encoding="utf-8")
        # Re-open a config gap so both kinds are pending in one run.
        (self.target_repo / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
        seen = []

        def _accept(migration, index, total):
            seen.append((migration.id, index, total))
            return True

        run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, prompt_fn=_accept)
        ids = [i for i, _, _ in seen]
        self.assertEqual(ids[-1], "index-manifest-old-format")
        self.assertEqual([n for _, n, _ in seen], list(range(1, len(seen) + 1)))
        self.assertEqual({t for _, _, t in seen}, {len(seen)})

    def test_description_and_note_state_the_cost(self):
        migration = REPO_MIGRATIONS[0]
        self.assertIn("re-embeds the WHOLE repo", migration.note)
        self.assertIn("no per-file cap", migration.note)
        self.assertIn("Nothing is embedded", migration.description)

    def test_a_project_with_only_the_manifest_pending_is_not_reported_as_up_to_date(self):
        self.manifest.write_text(self._OLD_MANIFEST, encoding="utf-8")
        applied = run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, dry_run=True)
        self.assertNotEqual(applied, [])


class CompressHookMatcherCoversCompactTools(unittest.TestCase):
    """The compact tools moved from `mcp__local-compress__` to `mcp__qdrant__`.
    A compress-hook matcher `init` wrote before that no longer reaches the hook
    for them after `qdrant-server-replaced`, so `upgrade` must add the new
    alternative, idempotently, without touching anything it does not own."""

    _ID = "compress-hook-matcher-covers-compact-tools"
    _HOOK = "/old/tools-repo/hooks/compress_output.py"

    def _settings(self, matcher, script=None, extra_blocks=()):
        script = script or self._HOOK
        return {"hooks": {"PostToolUse": [
            {"matcher": matcher, "hooks": [{"type": "command", "command": "py", "args": [script]}]},
            *extra_blocks,
        ]}}

    def _matcher_of(self, settings):
        return settings["hooks"]["PostToolUse"][0]["matcher"]

    def test_the_constant_is_exactly_what_the_shipped_template_carries(self):
        template = json.loads((REPO_ROOT / "templates" / "settings.json.template").read_text(encoding="utf-8"))
        [block] = [b for b in template["hooks"]["PostToolUse"]
                   if any(Path(a).name == "compress_output.py" for h in b["hooks"] for a in h["args"])]
        self.assertIn(COMPACT_TOOLS_MATCHER, block["matcher"])
        # ...directly after the local-compress alternative, which is where apply splices it in.
        self.assertIn(f"mcp__local-compress__.*|{COMPACT_TOOLS_MATCHER}", block["matcher"])

    def test_detect_true_for_a_matcher_written_before_the_move(self):
        self.assertTrue(_migration(self._ID).detect({}, self._settings(_OLD_MATCHER)))

    def test_detect_true_for_the_legacy_hook_name_too(self):
        settings = self._settings(_OLD_MATCHER, script="/old/tools-repo/hooks/compress_bash_output.py")
        self.assertTrue(_migration(self._ID).detect({}, settings))

    def test_detect_false_once_the_matcher_has_the_alternative(self):
        self.assertFalse(_migration(self._ID).detect({}, self._settings(f"{_OLD_MATCHER}|{COMPACT_TOOLS_MATCHER}")))

    def test_detect_false_for_a_matcher_the_user_wrote_by_hand(self):
        for matcher in ("Bash", "Bash|Grep", "mcp__github__.*"):
            with self.subTest(matcher=matcher):
                self.assertFalse(_migration(self._ID).detect({}, self._settings(matcher)))

    def test_detect_false_without_a_compress_hook_or_without_settings(self):
        other = {"hooks": {"PostToolUse": [{"matcher": _OLD_MATCHER, "hooks": [{"args": ["/x/my_own_hook.py"]}]}]}}
        self.assertFalse(_migration(self._ID).detect({}, other))
        self.assertFalse(_migration(self._ID).detect({}, {}))
        self.assertFalse(_migration(self._ID).detect({}, {"hooks": {"PostToolUse": [{"hooks": []}]}}))  # no matcher key

    def test_apply_splices_the_alternative_in_right_after_the_local_compress_one(self):
        github = "mcp__github__(?:search_code|get_issue)$"
        before = self._settings(f"Bash|WebFetch|mcp__local-compress__.*|{github}|mcp__example_tools__.*")
        after_mcp, after = _migration(self._ID).apply({"mcpServers": {}}, before, _ctx())
        self.assertEqual(
            self._matcher_of(after),
            f"Bash|WebFetch|mcp__local-compress__.*|{COMPACT_TOOLS_MATCHER}|{github}|mcp__example_tools__.*",
        )
        self.assertEqual(after_mcp, {"mcpServers": {}})

    def test_apply_touches_only_the_matcher_and_never_mutates_its_input(self):
        other_block = {"matcher": "Bash", "hooks": [{"command": "my-hook", "args": ["my_hook.py"]}]}
        before = self._settings(_OLD_MATCHER, extra_blocks=(other_block,))
        snapshot = copy.deepcopy(before)
        _, after = _migration(self._ID).apply({}, before, _ctx())
        self.assertEqual(before, snapshot)
        expected = copy.deepcopy(snapshot)
        expected["hooks"]["PostToolUse"][0]["matcher"] = f"{_OLD_MATCHER}|{COMPACT_TOOLS_MATCHER}"
        self.assertEqual(after, expected)

    def test_apply_is_idempotent(self):
        once = _migration(self._ID).apply({}, self._settings(_OLD_MATCHER), _ctx())[1]
        self.assertFalse(_migration(self._ID).detect({}, once))
        twice = _migration(self._ID).apply({}, once, _ctx())[1]
        self.assertEqual(twice, once)
        self.assertEqual(self._matcher_of(twice).count(COMPACT_TOOLS_MATCHER), 1)

    def test_it_fixes_every_compress_hook_block_that_needs_it(self):
        second = {"matcher": _OLD_MATCHER, "hooks": [{"command": "py", "args": [self._HOOK]}]}
        before = self._settings(_OLD_MATCHER, extra_blocks=(second,))
        after = _migration(self._ID).apply({}, before, _ctx())[1]
        self.assertEqual([b["matcher"] for b in after["hooks"]["PostToolUse"]], [f"{_OLD_MATCHER}|{COMPACT_TOOLS_MATCHER}"] * 2)


class AnalyticsServerMissing(unittest.TestCase):
    """The savings and metrics tools moved from `local-compress` to the opt-in
    `analytics` server. A project that already tracks savings is offered the
    block so `/my-savings` keeps working; a project that does not is not."""

    _ID = "analytics-server-missing"

    def _mcp(self, **compress_env):
        return {"mcpServers": {
            "qdrant": {"command": "/venv/bin/python", "args": ["/t/mcpservers/qdrant_mcp_server.py"], "env": {}},
            "local-compress": {"command": "/venv/bin/python", "env": dict(compress_env)},
        }}

    def test_detect_true_when_tracking_is_on_and_there_is_no_analytics_block(self):
        for value in ("1", "true", "TRUE", " yes "):
            with self.subTest(value=value):
                self.assertTrue(_migration(self._ID).detect(self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS=value), {}))

    def test_detect_false_when_tracking_is_off_blank_or_unset(self):
        for env in ({}, {"CLAUDE_RUNWAY_TRACK_SAVINGS": ""}, {"CLAUDE_RUNWAY_TRACK_SAVINGS": "0"},
                    {"CLAUDE_RUNWAY_TRACK_SAVINGS": "maybe"}, {"CLAUDE_RUNWAY_TRACK_SAVINGS": 1}):
            with self.subTest(env=env):
                self.assertFalse(_migration(self._ID).detect(self._mcp(**env), {}))

    def test_detect_false_without_a_local_compress_block_or_with_an_odd_shape(self):
        self.assertFalse(_migration(self._ID).detect({}, {}))
        self.assertFalse(_migration(self._ID).detect({"mcpServers": {"qdrant": {}}}, {}))
        self.assertFalse(_migration(self._ID).detect({"mcpServers": None}, {}))
        self.assertFalse(_migration(self._ID).detect({"mcpServers": {"local-compress": None}}, {}))

    def test_detect_false_once_the_analytics_block_exists(self):
        mcp = self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1")
        mcp["mcpServers"]["analytics"] = {"command": "/venv/bin/python"}
        self.assertFalse(_migration(self._ID).detect(mcp, {}))

    def test_apply_adds_exactly_the_analytics_block(self):
        before = self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1")
        settings = {"hooks": {"SessionStart": []}}
        after, after_settings = _migration(self._ID).apply(before, settings, _ctx())
        self.assertEqual(after_settings, settings)
        block = after["mcpServers"]["analytics"]
        self.assertEqual(block["command"], "/fake/venv/bin/python")
        self.assertEqual(block["type"], "stdio")
        self.assertEqual(block["args"], [(REPO_ROOT / "mcpservers" / "analytics_mcp_server.py").as_posix()])
        # No DB override was set, so neither key is written (a blank one would mask inheritance).
        self.assertEqual(block["env"], {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"})
        expected = copy.deepcopy(before)
        expected["mcpServers"]["analytics"] = block
        self.assertEqual(after, expected)

    def test_apply_carries_over_both_db_overrides_and_never_mutates_its_input(self):
        before = self._mcp(
            CLAUDE_RUNWAY_TRACK_SAVINGS="1", CLAUDE_RUNWAY_SAVINGS_DB=" /data/s.db ", CLAUDE_RUNWAY_METRICS_DB="/data/m.db"
        )
        snapshot = copy.deepcopy(before)
        after = _migration(self._ID).apply(before, {}, _ctx())[0]
        self.assertEqual(before, snapshot)
        self.assertEqual(
            after["mcpServers"]["analytics"]["env"],
            {
                "CLAUDE_RUNWAY_TRACK_SAVINGS": "1",
                "CLAUDE_RUNWAY_SAVINGS_DB": "/data/s.db",
                "CLAUDE_RUNWAY_METRICS_DB": "/data/m.db",
            },
        )

    def test_apply_carries_the_metrics_db_over_from_memory_bank(self):
        # `init --metrics-db` writes the override into memory-bank in every mode,
        # and into analytics only with --track-savings, so a project that never
        # tracked savings has it only there. Dropping it would split the data:
        # memory-bank writes the custom file, analytics reads the default one.
        before = self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1")
        before["mcpServers"]["memory-bank"] = {"env": {"CLAUDE_RUNWAY_METRICS_DB": "/data/m.db"}}
        after = _migration(self._ID).apply(before, {}, _ctx())[0]
        self.assertEqual(
            after["mcpServers"]["analytics"]["env"],
            {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1", "CLAUDE_RUNWAY_METRICS_DB": "/data/m.db"},
        )

    def test_metrics_db_is_carried_over_verbatim_not_trimmed(self):
        # metrics_lib.resolve_db_path() does not strip and memory-bank keeps its
        # own value as written, so a trimmed copy would be a different file.
        for name in ("memory-bank", "local-compress"):
            with self.subTest(source=name):
                before = self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1")
                before["mcpServers"].setdefault(name, {"env": {}})["env"]["CLAUDE_RUNWAY_METRICS_DB"] = " /data/m.db "
                after = _migration(self._ID).apply(before, {}, _ctx())[0]
                self.assertEqual(after["mcpServers"]["analytics"]["env"]["CLAUDE_RUNWAY_METRICS_DB"], " /data/m.db ")

    def test_metrics_db_prefers_memory_bank_then_falls_back_to_local_compress(self):
        both = self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1", CLAUDE_RUNWAY_METRICS_DB="/legacy/m.db")
        both["mcpServers"]["memory-bank"] = {"env": {"CLAUDE_RUNWAY_METRICS_DB": "/data/m.db"}}
        after = _migration(self._ID).apply(both, {}, _ctx())[0]
        self.assertEqual(after["mcpServers"]["analytics"]["env"]["CLAUDE_RUNWAY_METRICS_DB"], "/data/m.db")
        # A blank memory-bank value is not an override: the legacy one is used.
        both["mcpServers"]["memory-bank"] = {"env": {"CLAUDE_RUNWAY_METRICS_DB": "  "}}
        after = _migration(self._ID).apply(both, {}, _ctx())[0]
        self.assertEqual(after["mcpServers"]["analytics"]["env"]["CLAUDE_RUNWAY_METRICS_DB"], "/legacy/m.db")

    def test_apply_leaves_no_placeholder_behind(self):
        from setup_project_lib import find_unresolved_placeholders

        after = _migration(self._ID).apply(self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1"), {}, _ctx())[0]
        self.assertEqual(find_unresolved_placeholders(after["mcpServers"]["analytics"]), [])

    def test_a_pre_change_config_is_migrated_and_a_second_run_changes_nothing(self):
        # The pre-change shape: tracking on in local-compress, no analytics block.
        pre_change = self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1", CLAUDE_RUNWAY_SAVINGS_DB="/data/s.db")
        self.assertTrue(_migration(self._ID).detect(pre_change, {}))

        once = _migration(self._ID).apply(pre_change, {}, _ctx())[0]
        self.assertIn("analytics", once["mcpServers"])
        self.assertEqual(once["mcpServers"]["analytics"]["env"]["CLAUDE_RUNWAY_SAVINGS_DB"], "/data/s.db")
        self.assertFalse(_migration(self._ID).detect(once, {}))
        self.assertNotIn(self._ID, {m.id for m in pending_migrations(once, {})})

        twice = _migration(self._ID).apply(once, {}, _ctx())[0]
        self.assertEqual(twice, once)

    def test_it_never_touches_the_local_compress_block(self):
        # That block keeps TRACK_SAVINGS: the compression server's savings metrics, which
        # the hook turns into savings events, still depends on it.
        before = self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1", CLAUDE_RUNWAY_SAVINGS_DB="/data/s.db")
        after = _migration(self._ID).apply(before, {}, _ctx())[0]
        self.assertEqual(after["mcpServers"]["local-compress"], before["mcpServers"]["local-compress"])


class LocalCompressSavingsDbInert(unittest.TestCase):
    """The `local-compress` server never opens the savings database,
    so the `CLAUDE_RUNWAY_SAVINGS_DB` every older `init` wrote into its block is
    inert. `upgrade` removes it, but only after `analytics-server-missing` has
    carried a real override over into the block that does read it."""

    _ID = "local-compress-savings-db-inert"

    def _mcp(self, analytics=False, **compress_env):
        servers = {
            "qdrant": {"command": "/venv/bin/python", "args": ["/t/mcpservers/qdrant_mcp_server.py"], "env": {}},
            "local-compress": {"command": "/venv/bin/python", "env": dict(compress_env)},
        }
        if analytics:
            servers["analytics"] = {"command": "/venv/bin/python", "env": {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}}
        return {"mcpServers": servers}

    def test_detect_true_whenever_the_key_is_present_blank_or_not(self):
        for value in ("", "/data/s.db"):
            with self.subTest(value=value):
                self.assertTrue(_migration(self._ID).detect(self._mcp(CLAUDE_RUNWAY_SAVINGS_DB=value), {}))

    def test_detect_false_without_the_key_or_with_an_odd_shape(self):
        self.assertFalse(_migration(self._ID).detect(self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1"), {}))
        self.assertFalse(_migration(self._ID).detect({}, {}))
        self.assertFalse(_migration(self._ID).detect({"mcpServers": None}, {}))
        self.assertFalse(_migration(self._ID).detect({"mcpServers": {"local-compress": None}}, {}))
        self.assertFalse(_migration(self._ID).detect({"mcpServers": {"local-compress": {"env": []}}}, {}))
        # The key in some other block is not this migration's business.
        mcp = self._mcp(analytics=True)
        mcp["mcpServers"]["analytics"]["env"]["CLAUDE_RUNWAY_SAVINGS_DB"] = "/data/s.db"
        self.assertFalse(_migration(self._ID).detect(mcp, {}))

    def test_apply_removes_only_that_key_and_never_mutates_its_input(self):
        before = self._mcp(
            CLAUDE_RUNWAY_LMSTUDIO_URL="http://lm:1234/v1",
            CLAUDE_RUNWAY_TRACK_SAVINGS="",
            CLAUDE_RUNWAY_SAVINGS_DB="",
        )
        settings = {"hooks": {"SessionStart": []}}
        snapshot = copy.deepcopy(before)
        after, after_settings = _migration(self._ID).apply(before, settings, _ctx())
        self.assertEqual(before, snapshot)
        self.assertEqual(after_settings, settings)
        expected = copy.deepcopy(before)
        del expected["mcpServers"]["local-compress"]["env"]["CLAUDE_RUNWAY_SAVINGS_DB"]
        self.assertEqual(after, expected)
        self.assertFalse(_migration(self._ID).detect(after, {}))

    def test_apply_is_a_no_op_while_the_analytics_carry_over_is_still_pending(self):
        # Tracking on, no analytics block: removing the key now would lose the
        # override analytics-server-missing is about to carry over.
        before = self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1", CLAUDE_RUNWAY_SAVINGS_DB="/data/s.db")
        after = _migration(self._ID).apply(before, {}, _ctx())[0]
        self.assertEqual(after, before)
        # Once the analytics block exists, the removal goes ahead.
        with_analytics = self._mcp(analytics=True, CLAUDE_RUNWAY_TRACK_SAVINGS="1", CLAUDE_RUNWAY_SAVINGS_DB="/x")
        removed = _migration(self._ID).apply(with_analytics, {}, _ctx())[0]
        self.assertNotIn("CLAUDE_RUNWAY_SAVINGS_DB", removed["mcpServers"]["local-compress"]["env"])

    def test_apply_removes_it_when_tracking_is_off(self):
        # No analytics block is ever offered then, so nothing waits on the value.
        before = self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="", CLAUDE_RUNWAY_SAVINGS_DB="/data/s.db")
        after = _migration(self._ID).apply(before, {}, _ctx())[0]
        self.assertNotIn("CLAUDE_RUNWAY_SAVINGS_DB", after["mcpServers"]["local-compress"]["env"])

    def test_it_comes_after_analytics_server_missing(self):
        ids = [m.id for m in MIGRATIONS]
        self.assertLess(ids.index("analytics-server-missing"), ids.index(self._ID))

    def _run(self, mcp_json, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        target = Path(tmp.name)
        (target / ".mcp.json").write_text(json.dumps(mcp_json), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            applied = run_upgrade(target, REPO_ROOT, _VENV_PYTHON, **kwargs)
        return applied, json.loads((target / ".mcp.json").read_text(encoding="utf-8"))

    def test_one_run_carries_the_override_over_then_removes_the_inert_copy(self):
        applied, after = self._run(
            self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1", CLAUDE_RUNWAY_SAVINGS_DB="/data/s.db"), auto_yes=True
        )
        # (This minimal config has other migrations pending too; only these two matter here.)
        ours = [i for i in applied if i in ("analytics-server-missing", self._ID)]
        self.assertEqual(ours, ["analytics-server-missing", self._ID])
        self.assertEqual(after["mcpServers"]["analytics"]["env"]["CLAUDE_RUNWAY_SAVINGS_DB"], "/data/s.db")
        self.assertEqual(after["mcpServers"]["local-compress"]["env"], {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"})

    def test_declining_the_carry_over_keeps_the_value_and_leaves_this_pending(self):
        def accept_only_this_one(migration, index, total):
            # Declines analytics-server-missing (and the unrelated others this
            # minimal config also has pending), accepts only this migration.
            return migration.id == self._ID

        before = self._mcp(CLAUDE_RUNWAY_TRACK_SAVINGS="1", CLAUDE_RUNWAY_SAVINGS_DB="/data/s.db")
        applied, after = self._run(before, prompt_fn=accept_only_this_one)
        self.assertEqual(applied, [])
        self.assertEqual(after, before)
        self.assertIn(self._ID, {m.id for m in pending_migrations(after, {})})


class CompressHookRenamed(unittest.TestCase):
    """compress_bash_output.py -> compress_output.py."""

    OLD = "/old/tools-repo/hooks/compress_bash_output.py"
    NEW = "/old/tools-repo/hooks/compress_output.py"

    def _settings(self, blocks):
        return {"hooks": {"PostToolUse": blocks}}

    def test_detect_true_for_old_name_false_for_new(self):
        m = _migration("compress-hook-renamed")
        old = self._settings([{"matcher": "Bash", "hooks": [{"command": "py", "args": [self.OLD]}]}])
        new = self._settings([{"matcher": "Bash", "hooks": [{"command": "py", "args": [self.NEW]}]}])
        self.assertTrue(m.detect({}, old))
        self.assertFalse(m.detect({}, new))
        self.assertFalse(m.detect({}, {}))

    def test_apply_rewrites_only_the_filename_keeping_everything_else(self):
        m = _migration("compress-hook-renamed")
        block = {
            "matcher": "Bash|Grep|custom",
            "hooks": [
                {"type": "command", "command": "/v/python", "args": [self.OLD]},
                {"type": "command", "command": "mine", "args": ["--flag"]},
            ],
        }
        settings = self._settings([block])
        before = copy.deepcopy(settings)
        _, out = m.apply({}, settings, _ctx())
        self.assertEqual(settings, before, "apply must not mutate its input")
        got = out["hooks"]["PostToolUse"]
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["matcher"], "Bash|Grep|custom")
        self.assertEqual(got[0]["hooks"][0]["args"], [self.NEW])
        self.assertEqual(got[0]["hooks"][0]["command"], "/v/python")
        self.assertEqual(got[0]["hooks"][1], {"type": "command", "command": "mine", "args": ["--flag"]})
        self.assertFalse(m.detect({}, out))

    def test_apply_is_idempotent(self):
        m = _migration("compress-hook-renamed")
        settings = self._settings([{"hooks": [{"args": [self.OLD]}]}])
        _, once = m.apply({}, settings, _ctx())
        _, twice = m.apply({}, once, _ctx())
        self.assertEqual(once, twice)

    def test_does_not_touch_other_events_hooks(self):
        m = _migration("compress-hook-renamed")
        settings = {
            "hooks": {
                "PostToolUse": [{"hooks": [{"args": [self.OLD]}]}],
                "SessionEnd": [{"hooks": [{"args": ["/x/hooks/session_end_savings.py"]}]}],
            }
        }
        _, out = m.apply({}, settings, _ctx())
        self.assertEqual(out["hooks"]["SessionEnd"], settings["hooks"]["SessionEnd"])

    def test_drops_stale_duplicate_when_new_name_already_registered(self):
        m = _migration("compress-hook-renamed")
        settings = self._settings(
            [
                {"matcher": "Bash", "hooks": [{"args": [self.OLD]}]},
                {"matcher": "Bash", "hooks": [{"args": [self.NEW]}]},
            ]
        )
        _, out = m.apply({}, settings, _ctx())
        self.assertEqual(out["hooks"]["PostToolUse"], [{"matcher": "Bash", "hooks": [{"args": [self.NEW]}]}])

    def test_legacy_hook_with_a_different_matcher_is_renamed_not_dropped(self):
        # Copilot review: a legacy Bash block next to a new-named
        # Grep block is not a duplicate; dropping it disabled Bash compression.
        m = _migration("compress-hook-renamed")
        settings = self._settings(
            [
                {"matcher": "Bash", "hooks": [{"args": [self.OLD]}]},
                {"matcher": "Grep", "hooks": [{"args": [self.NEW]}]},
            ]
        )
        _, out = m.apply({}, settings, _ctx())
        self.assertEqual(
            out["hooks"]["PostToolUse"],
            [
                {"matcher": "Bash", "hooks": [{"args": [self.NEW]}]},
                {"matcher": "Grep", "hooks": [{"args": [self.NEW]}]},
            ],
        )

    def test_duplicate_drop_keeps_a_block_that_still_holds_other_hooks(self):
        m = _migration("compress-hook-renamed")
        settings = self._settings(
            [
                {"hooks": [{"args": [self.OLD]}, {"args": ["--mine"]}]},
                {"hooks": [{"args": [self.NEW]}]},
            ]
        )
        _, out = m.apply({}, settings, _ctx())
        self.assertEqual(out["hooks"]["PostToolUse"][0]["hooks"], [{"args": ["--mine"]}])

    def test_setup_lib_still_recognizes_the_legacy_name_as_toolkit_owned(self):
        """setup_project_lib must still RECOGNIZE the old name as toolkit-owned."""
        from setup_project_lib import _is_toolkit_owned_hook

        self.assertTrue(_is_toolkit_owned_hook({"args": [self.OLD]}))
        self.assertTrue(_is_toolkit_owned_hook({"args": [self.NEW]}))


class McpServerScriptsMoved(unittest.TestCase):
    """The five server scripts moved from tools/ to mcpservers/. A config
    written before the move names the old path, so every server fails to start
    until `upgrade` rewrites it. Detection is by that old path alone, which is
    what makes a re-run a no-op."""

    _ID = "mcp-server-scripts-moved"
    _SCRIPTS = (
        ("qdrant", "qdrant_mcp_server"),
        ("codebase-indexer", "ingest_mcp_server"),
        ("memory-bank", "memory_bank_mcp_server"),
        ("local-compress", "compress_mcp_server"),
        ("analytics", "analytics_mcp_server"),
    )

    def _old(self, root: str = "/home/u/claude-runway") -> dict:
        return {
            "mcpServers": {
                name: {
                    "command": "/venv/bin/python",
                    "type": "stdio",
                    "args": [f"{root}/tools/{stem}.py"],
                    "env": {"KEEP": name},
                }
                for name, stem in self._SCRIPTS
            }
        }

    def _apply(self, mcp_json: dict) -> dict:
        return _migration(self._ID).apply(mcp_json, {}, _ctx())[0]

    def test_detect_true_for_a_config_naming_the_old_paths(self):
        self.assertTrue(_migration(self._ID).detect(self._old(), {}))

    def test_detect_true_when_only_one_server_still_has_the_old_path(self):
        mcp_json = self._apply(self._old())
        mcp_json["mcpServers"]["analytics"]["args"] = ["/home/u/claude-runway/tools/analytics_mcp_server.py"]
        self.assertTrue(_migration(self._ID).detect(mcp_json, {}))

    def test_apply_rewrites_each_of_the_five_paths(self):
        after = self._apply(self._old())["mcpServers"]
        for name, stem in self._SCRIPTS:
            self.assertEqual(after[name]["args"], [f"/home/u/claude-runway/mcpservers/{stem}.py"], name)

    def test_apply_changes_only_the_script_path(self):
        before = self._old()
        after = self._apply(before)
        for name, _ in self._SCRIPTS:
            for key in ("command", "type", "env"):
                self.assertEqual(after["mcpServers"][name][key], before["mcpServers"][name][key], (name, key))
        # And it never mutates its input.
        self.assertEqual(before, self._old())

    def test_a_current_config_is_left_alone(self):
        current = self._apply(self._old())
        self.assertFalse(_migration(self._ID).detect(current, {}))
        self.assertEqual(self._apply(current), current)

    def test_apply_is_idempotent(self):
        once = self._apply(self._old())
        self.assertEqual(self._apply(once), once)

    def test_a_pipx_env_path_is_rewritten_too(self):
        mcp_json = self._old("/home/u/.local/pipx/venvs/claude-runway/src")
        after = self._apply(mcp_json)["mcpServers"]["qdrant"]["args"]
        self.assertEqual(after, ["/home/u/.local/pipx/venvs/claude-runway/src/mcpservers/qdrant_mcp_server.py"])

    def test_a_windows_backslash_path_keeps_its_separators(self):
        mcp_json = {
            "mcpServers": {
                "qdrant": {"command": "python", "args": ["C:\\repos\\claude-runway\\tools\\qdrant_mcp_server.py"]}
            }
        }
        migration = _migration(self._ID)
        self.assertTrue(migration.detect(mcp_json, {}))
        after = self._apply(mcp_json)["mcpServers"]["qdrant"]["args"]
        self.assertEqual(after, ["C:\\repos\\claude-runway\\mcpservers\\qdrant_mcp_server.py"])

    def test_a_users_own_script_in_a_tools_folder_is_not_touched(self):
        mcp_json = {
            "mcpServers": {
                "mine": {"command": "python", "args": ["/home/u/tools/my_server.py"]},
                # Right directory, but not one of the five moved scripts.
                "other": {"command": "python", "args": ["/home/u/claude-runway/tools/doctor.py"]},
                # One of the five names, but not under a `tools` path segment.
                "lookalike": {"command": "python", "args": ["/home/u/mytools/qdrant_mcp_server.py"]},
            }
        }
        self.assertFalse(_migration(self._ID).detect(mcp_json, {}))
        self.assertEqual(self._apply(mcp_json), mcp_json)

    def test_a_block_without_args_or_with_odd_shapes_does_not_break_detect_or_apply(self):
        mcp_json = {"mcpServers": {"a": {"command": "x"}, "b": {"args": None}, "c": "not-a-dict", "d": {"args": [1, None]}}}
        migration = _migration(self._ID)
        self.assertFalse(migration.detect(mcp_json, {}))
        self.assertEqual(self._apply(mcp_json), mcp_json)
        self.assertFalse(migration.detect({}, {}))

    def test_settings_json_is_returned_untouched(self):
        settings = {"hooks": {"SessionStart": []}}
        _, settings_after = _migration(self._ID).apply(self._old(), settings, _ctx())
        self.assertEqual(settings_after, settings)

    def test_run_upgrade_lists_it_in_dry_run_then_applies_it_and_a_rerun_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / ".mcp.json").write_text(json.dumps(self._old(REPO_ROOT.as_posix())), encoding="utf-8")
            before = (target / ".mcp.json").read_bytes()
            listed = run_upgrade(target, REPO_ROOT, _VENV_PYTHON, dry_run=True)
            self.assertIn(self._ID, listed)
            self.assertEqual((target / ".mcp.json").read_bytes(), before)

            applied = run_upgrade(target, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
            self.assertIn(self._ID, applied)
            written = json.loads((target / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
            for name, stem in self._SCRIPTS:
                script = Path(written[name]["args"][0])
                self.assertEqual(script.parent.name, "mcpservers", name)
                # The rewritten path is where the script really lives.
                self.assertTrue((REPO_ROOT / "mcpservers" / f"{stem}.py").is_file(), stem)
            self.assertEqual(run_upgrade(target, REPO_ROOT, _VENV_PYTHON, auto_yes=True), [])


class UpgradeChainFromPreSplitProject(unittest.TestCase):
    """A project written before the `qdrant`/`analytics` split: the `qdrant`
    block launches the third-party console script, savings tracking is on and
    there is no `analytics` block. `upgrade` must leave it exactly as a fresh
    `init --track-savings` would, with no trace of the old launcher."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "presplit"
        self.target_repo.mkdir()
        lmstudio = {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1", "CLAUDE_RUNWAY_LMSTUDIO_MODEL": ""}
        (self.target_repo / ".mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "qdrant": {
                            "command": "/old/venv/bin/mcp-server-qdrant",
                            "type": "stdio",
                            "env": {
                                "QDRANT_URL": "http://localhost:6333",
                                "QDRANT_API_KEY": "",
                                "COLLECTION_NAME": "presplit",
                                "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
                                "HF_HUB_OFFLINE": "1",
                            },
                        },
                        "local-compress": {
                            "command": "/old/venv/bin/python",
                            "type": "stdio",
                            "args": ["/old/tools/compress_mcp_server.py"],
                            "env": {
                                **lmstudio,
                                "COMPACT_COLLECTION": "conversation-compacts",
                                "CLAUDE_RUNWAY_TRACK_SAVINGS": "1",
                                "CLAUDE_RUNWAY_SAVINGS_DB": "",
                            },
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

    def _mcp(self):
        return json.loads((self.target_repo / ".mcp.json").read_text(encoding="utf-8"))

    def test_upgrade_leaves_no_third_party_reference_and_matches_a_fresh_init(self):
        from setup_project_lib import run_setup

        run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        text = (self.target_repo / ".mcp.json").read_text(encoding="utf-8")
        for stale in ("mcp-server-qdrant", "mcp_server_qdrant", "fastmcp"):
            self.assertNotIn(stale, text)

        fresh_target = Path(self._tmpdir.name) / "fresh"
        fresh_target.mkdir()
        fresh = run_setup(
            fresh_target, REPO_ROOT, collection_name="presplit", venv_python=_VENV_PYTHON, track_savings=True
        ).mcp_json["mcpServers"]
        upgraded = self._mcp()["mcpServers"]

        self.assertEqual(upgraded["analytics"], fresh["analytics"])
        # The pre-split project also launched its compress server from the old
        # tools/ folder; the path migration moves that to mcpservers/ in the
        # same run, with nothing else in the block touched.
        self.assertEqual(upgraded["local-compress"]["args"], ["/old/mcpservers/compress_mcp_server.py"])
        # Compacts that predate the setting stay pinned to the model they were
        # stored with; that is the one deliberate difference from a new install.
        fresh_qdrant = copy.deepcopy(fresh["qdrant"])
        fresh_qdrant["env"]["COMPACT_EMBEDDING_MODEL"] = "BAAI/bge-small-en"
        self.assertEqual(upgraded["qdrant"], fresh_qdrant)

    def test_a_second_upgrade_changes_nothing(self):
        run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        before = (self.target_repo / ".mcp.json").read_bytes()
        self.assertEqual(run_upgrade(self.target_repo, REPO_ROOT, _VENV_PYTHON, auto_yes=True), [])
        self.assertEqual((self.target_repo / ".mcp.json").read_bytes(), before)


class MalformedExistingConfig(unittest.TestCase):
    """upgrade refuses a malformed existing config with the shared
    MalformedConfigError, before prompting or writing anything."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target = Path(self._tmpdir.name) / "project"
        (self.target / ".claude").mkdir(parents=True)
        self.mcp_path = self.target / ".mcp.json"
        self.settings_path = self.target / ".claude" / "settings.json"

    def test_is_project_configured_raises_for_a_bad_mcp_json(self):
        for label, text in [
            ("mcpServers null", '{"mcpServers": null}'),
            ("mcpServers list", '{"mcpServers": []}'),
            ("top level list", "[]"),
            ("invalid json", "{not json"),
        ]:
            with self.subTest(label):
                self.mcp_path.write_text(text, encoding="utf-8")
                with self.assertRaises(MalformedConfigError):
                    is_project_configured(self.target)

    def test_run_upgrade_raises_for_a_bad_mcp_json_and_writes_nothing(self):
        self.mcp_path.write_text('{"mcpServers": null}', encoding="utf-8")
        with self.assertRaises(MalformedConfigError) as ctx:
            run_upgrade(self.target, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        self.assertIn(str(self.mcp_path), str(ctx.exception))
        self.assertEqual(self.mcp_path.read_text(encoding="utf-8"), '{"mcpServers": null}')
        self.assertFalse(self.settings_path.exists())

    def test_run_upgrade_raises_for_a_bad_settings_json_and_writes_nothing(self):
        mcp_text = json.dumps({"mcpServers": {"qdrant": {}}})
        self.mcp_path.write_text(mcp_text, encoding="utf-8")
        for label, text in [
            ("hooks null", '{"hooks": null}'),
            ("hooks list", '{"hooks": []}'),
            ("event null", '{"hooks": {"PostToolUse": null}}'),
            ("invalid json", "{not json"),
        ]:
            with self.subTest(label):
                self.settings_path.write_text(text, encoding="utf-8")
                with self.assertRaises(MalformedConfigError) as ctx:
                    run_upgrade(self.target, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
                self.assertIn(str(self.settings_path), str(ctx.exception))
                self.assertEqual(self.mcp_path.read_text(encoding="utf-8"), mcp_text)
                self.assertEqual(self.settings_path.read_text(encoding="utf-8"), text)

    def test_server_args_that_are_not_a_list_raise_instead_of_a_typeerror(self):
        # Review finding on the PR: `"args": 1` passed the loader and then
        # crashed list(args) in the qdrant-server migration's detect.
        self.mcp_path.write_text(json.dumps({"mcpServers": {"qdrant": {"args": 1}}}), encoding="utf-8")
        with self.assertRaises(MalformedConfigError):
            run_upgrade(self.target, REPO_ROOT, _VENV_PYTHON, auto_yes=True)

    def test_a_list_matcher_on_a_legacy_compress_hook_raises_instead_of_a_typeerror(self):
        # Review finding on the PR: a list `matcher` passed the loader and then
        # was put in a set by the compress-hook-renamed migration (unhashable).
        self.mcp_path.write_text(json.dumps({"mcpServers": {"qdrant": {"command": "x"}}}), encoding="utf-8")
        legacy = {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": ["Bash", "Grep"],
                        "hooks": [{"type": "command", "command": "py", "args": ["/x/hooks/compress_bash_output.py"]}],
                    }
                ]
            }
        }
        self.settings_path.write_text(json.dumps(legacy), encoding="utf-8")
        with self.assertRaises(MalformedConfigError) as ctx:
            run_upgrade(self.target, REPO_ROOT, _VENV_PYTHON, auto_yes=True)
        self.assertIn('"matcher" is a list', str(ctx.exception))


class SavingsTrackerOffer(unittest.TestCase):
    """A project that never tracked savings has no `analytics`
    server, so /my-savings and /my-metrics fail with an unknown-tool error and
    nothing says why. An interactive `upgrade` asks (the caller passes
    `offer_savings_fn`); a yes turns tracking on and adds the block through
    `analytics-server-missing`, without a second question."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target = Path(self._tmpdir.name) / "proj"
        self.target.mkdir()
        self.mcp_path = self.target / ".mcp.json"

    def _init_project(self, **kwargs):
        # A fully current config as a bare `init` writes it: nothing else is pending.
        result = run_setup(self.target, REPO_ROOT, home_dir=Path("/home/user"), **kwargs)
        self.mcp_path.write_text(json.dumps(result.mcp_json), encoding="utf-8")
        (self.target / ".claude").mkdir(exist_ok=True)
        (self.target / ".claude" / "settings.json").write_text(json.dumps(result.settings_json), encoding="utf-8")

    def _servers(self):
        return json.loads(self.mcp_path.read_text(encoding="utf-8"))["mcpServers"]

    def _run(self, **kwargs):
        buf = io.StringIO()
        with redirect_stdout(buf):
            applied = run_upgrade(self.target, REPO_ROOT, _VENV_PYTHON, **kwargs)
        return applied, buf.getvalue()

    def test_precondition_a_bare_init_is_fully_current_and_has_no_analytics(self):
        self._init_project()
        self.assertNotIn("analytics", self._servers())
        applied, out = self._run(auto_yes=True)
        self.assertEqual(applied, [])
        self.assertIn("No pending upgrades", out)

    def test_yes_enables_tracking_adds_analytics_and_does_not_ask_twice(self):
        self._init_project()
        asked = []

        def _unexpected(migration, index, total):
            raise AssertionError(f"asked a second time about {migration.id}")

        applied, out = self._run(prompt_fn=_unexpected, offer_savings_fn=lambda: asked.append(1) or True)
        self.assertEqual(asked, [1])
        self.assertEqual(applied, ["analytics-server-missing"])
        servers = self._servers()
        self.assertEqual(servers["local-compress"]["env"]["CLAUDE_RUNWAY_TRACK_SAVINGS"], "1")
        self.assertEqual(servers["analytics"]["env"]["CLAUDE_RUNWAY_TRACK_SAVINGS"], "1")
        # The block is the one `init --track-savings` writes, not a variant of it.
        expected = run_setup(
            Path(self._tmpdir.name) / "other", REPO_ROOT, home_dir=Path("/home/user"), track_savings=True,
        ).generated_mcp_servers["analytics"]
        self.assertEqual(servers["analytics"]["args"], expected["args"])
        # The hooks read the shell, not .mcp.json, so the export is called out.
        self.assertIn("Export CLAUDE_RUNWAY_TRACK_SAVINGS=1 in your shell", out)

    def test_yes_keeps_a_custom_metrics_db_so_memory_bank_and_analytics_share_one_file(self):
        # Project set up with `init --metrics-db` and tracking off: the override
        # lives only in the memory-bank block. The new analytics block must read
        # the same file, or /my-metrics would show an empty default database.
        self._init_project(metrics_db="/data/metrics.db")
        self.assertEqual(self._servers()["memory-bank"]["env"]["CLAUDE_RUNWAY_METRICS_DB"], "/data/metrics.db")
        self._run(prompt_fn=lambda m, i, t: True, offer_savings_fn=lambda: True)
        servers = self._servers()
        self.assertEqual(servers["analytics"]["env"]["CLAUDE_RUNWAY_METRICS_DB"], "/data/metrics.db")
        self.assertEqual(servers["memory-bank"]["env"]["CLAUDE_RUNWAY_METRICS_DB"], "/data/metrics.db")

    def test_no_leaves_the_project_untouched(self):
        self._init_project()
        before = self.mcp_path.read_text(encoding="utf-8")
        applied, out = self._run(prompt_fn=lambda m, i, t: True, offer_savings_fn=lambda: False)
        self.assertEqual(applied, [])
        self.assertEqual(self.mcp_path.read_text(encoding="utf-8"), before)
        self.assertIn("No pending upgrades", out)

    def test_no_offer_function_never_asks(self):
        # The library default: a caller that passes nothing keeps tracking off.
        self._init_project()
        applied, _ = self._run(prompt_fn=lambda m, i, t: True)
        self.assertEqual(applied, [])
        self.assertNotIn("analytics", self._servers())

    def test_auto_yes_and_dry_run_never_ask(self):
        self._init_project()

        def _explode():
            raise AssertionError("must not ask")

        for kwargs in ({"auto_yes": True}, {"dry_run": True}):
            with self.subTest(**kwargs):
                applied, _ = self._run(offer_savings_fn=_explode, **kwargs)
                self.assertEqual(applied, [])
                self.assertNotIn("analytics", self._servers())

    def test_not_asked_when_the_project_already_tracks_or_has_the_block(self):
        def _explode():
            raise AssertionError("must not ask")

        # Tracking already on, block missing: analytics-server-missing's own offer.
        self._init_project(track_savings=True)
        servers = json.loads(self.mcp_path.read_text(encoding="utf-8"))
        del servers["mcpServers"]["analytics"]
        self.mcp_path.write_text(json.dumps(servers), encoding="utf-8")
        applied, _ = self._run(prompt_fn=lambda m, i, t: True, offer_savings_fn=_explode)
        self.assertEqual(applied, ["analytics-server-missing"])
        # Fully set up: nothing to ask or do.
        applied, _ = self._run(prompt_fn=lambda m, i, t: True, offer_savings_fn=_explode)
        self.assertEqual(applied, [])

    def test_not_asked_for_a_qdrant_only_project(self):
        # No local-compress block means no tracker to offer.
        self._init_project(include_compress=False)
        self.assertNotIn("local-compress", self._servers())

        def _explode():
            raise AssertionError("must not ask")

        applied, _ = self._run(prompt_fn=lambda m, i, t: True, offer_savings_fn=_explode)
        self.assertNotIn("analytics", self._servers())
        self.assertNotIn("analytics-server-missing", applied)


if __name__ == "__main__":
    unittest.main()
