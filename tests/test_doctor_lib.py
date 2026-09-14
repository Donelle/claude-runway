#!/usr/bin/env python3
"""Tests for libs/doctor_lib.py -- the validator behind tools/doctor.py
that diffs .mcp.json's local-compress env block against the shell for the
four dual-config CLAUDE_RUNWAY_* vars (issue #49).

Stdlib-only (unittest), no network, no real .mcp.json fixture beyond
temp-dir JSON this test writes itself:

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

import doctor_lib  # noqa: E402


class LoadLocalCompressEnv(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.mcp_json_path = Path(self._tmpdir.name) / ".mcp.json"

    def test_missing_file_returns_none(self):
        self.assertIsNone(doctor_lib.load_local_compress_env(self.mcp_json_path))

    def test_invalid_json_raises_malformed_error_not_none(self):
        # Regression for the finding from PR #119 review: invalid/unreadable
        # JSON previously returned None, indistinguishable from a genuine
        # --qdrant-only setup with nothing to check -- so the CLI's
        # advertised "exit 1 on a problem" gate silently exited 0 on a
        # broken .mcp.json instead of reporting it.
        self.mcp_json_path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(doctor_lib.MalformedMcpJsonError):
            doctor_lib.load_local_compress_env(self.mcp_json_path)

    def test_top_level_not_a_dict_raises_malformed_error(self):
        self.mcp_json_path.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertRaises(doctor_lib.MalformedMcpJsonError):
            doctor_lib.load_local_compress_env(self.mcp_json_path)

    def test_mcp_servers_null_raises_malformed_error_not_attribute_error(self):
        # Regression for the finding from PR #119 review: "mcpServers":
        # null (present but None) previously crashed with an uncaught
        # AttributeError from the blind data.get("mcpServers", {}).get(...)
        # chain -- .get()'s default only applies when the KEY is missing,
        # not when it's present-but-None, so chaining .get() straight onto
        # that None blew up instead of returning any reportable outcome.
        self.mcp_json_path.write_text(json.dumps({"mcpServers": None}), encoding="utf-8")
        with self.assertRaises(doctor_lib.MalformedMcpJsonError):
            doctor_lib.load_local_compress_env(self.mcp_json_path)

    def test_mcp_servers_as_a_list_raises_malformed_error_not_attribute_error(self):
        self.mcp_json_path.write_text(json.dumps({"mcpServers": [1, 2, 3]}), encoding="utf-8")
        with self.assertRaises(doctor_lib.MalformedMcpJsonError):
            doctor_lib.load_local_compress_env(self.mcp_json_path)

    def test_local_compress_not_a_dict_raises_malformed_error(self):
        self.mcp_json_path.write_text(json.dumps({"mcpServers": {"local-compress": "not-an-object"}}), encoding="utf-8")
        with self.assertRaises(doctor_lib.MalformedMcpJsonError):
            doctor_lib.load_local_compress_env(self.mcp_json_path)

    def test_local_compress_env_not_a_dict_raises_malformed_error(self):
        self.mcp_json_path.write_text(
            json.dumps({"mcpServers": {"local-compress": {"command": "x", "env": "not-an-object"}}}),
            encoding="utf-8",
        )
        with self.assertRaises(doctor_lib.MalformedMcpJsonError):
            doctor_lib.load_local_compress_env(self.mcp_json_path)

    def test_local_compress_explicit_null_value_raises_malformed_error_not_none(self):
        # Regression for the finding from PR #119 review (round 2):
        # {"local-compress": null} previously came back from
        # mcp_servers.get("local-compress") as None -- INDISTINGUISHABLE
        # from the key being genuinely absent (a real --qdrant-only setup)
        # -- so this malformed config silently printed "nothing to check"
        # and exited 0 instead of surfacing the real problem.
        self.mcp_json_path.write_text(json.dumps({"mcpServers": {"local-compress": None}}), encoding="utf-8")
        with self.assertRaises(doctor_lib.MalformedMcpJsonError):
            doctor_lib.load_local_compress_env(self.mcp_json_path)

    def test_env_value_that_is_not_a_string_raises_malformed_error(self):
        # Regression for the finding from PR #119 review (round 2):
        # a non-string env value (here, an int) previously either crashed
        # downstream inside _is_truthy_track_savings()'s .strip() call, or
        # (for a JSON null) was silently misread by _normalize() as "key
        # absent" even though the key was genuinely present with a bad
        # value -- producing a false "OK" instead of a reportable error.
        for bad_value in (1, None, True, [1, 2], {"nested": "dict"}):
            with self.subTest(bad_value=bad_value):
                self.mcp_json_path.write_text(
                    json.dumps(
                        {"mcpServers": {"local-compress": {"command": "x", "env": {"CLAUDE_RUNWAY_TRACK_SAVINGS": bad_value}}}}
                    ),
                    encoding="utf-8",
                )
                with self.assertRaises(doctor_lib.MalformedMcpJsonError):
                    doctor_lib.load_local_compress_env(self.mcp_json_path)

    def test_no_mcp_servers_key_returns_none(self):
        self.mcp_json_path.write_text(json.dumps({}), encoding="utf-8")
        self.assertIsNone(doctor_lib.load_local_compress_env(self.mcp_json_path))

    def test_no_local_compress_server_returns_none(self):
        # e.g. a --qdrant-only setup: qdrant/codebase-indexer configured, no local-compress.
        self.mcp_json_path.write_text(
            json.dumps({"mcpServers": {"qdrant": {"command": "x", "env": {}}}}), encoding="utf-8"
        )
        self.assertIsNone(doctor_lib.load_local_compress_env(self.mcp_json_path))

    def test_local_compress_with_no_env_block_returns_empty_dict(self):
        self.mcp_json_path.write_text(
            json.dumps({"mcpServers": {"local-compress": {"command": "x"}}}), encoding="utf-8"
        )
        self.assertEqual(doctor_lib.load_local_compress_env(self.mcp_json_path), {})

    def test_local_compress_env_block_returned_as_is(self):
        env = {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1"}
        self.mcp_json_path.write_text(
            json.dumps({"mcpServers": {"local-compress": {"command": "x", "env": env}}}), encoding="utf-8"
        )
        self.assertEqual(doctor_lib.load_local_compress_env(self.mcp_json_path), env)


class CheckDualEnvVars(unittest.TestCase):
    def setUp(self):
        self.home_dir = Path("/home/testuser")

    def _check(self, mcp_env, shell_env):
        return doctor_lib.check_dual_env_vars(mcp_env, shell_env, home_dir=self.home_dir)

    def test_var_absent_from_mcp_env_is_never_flagged(self):
        # Absent means "inherits from the shell" -- the exact same source
        # this check compares against, so it can never diverge.
        mismatches = self._check({}, {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "google/gemma-3-4b"})
        self.assertEqual(mismatches, [])

    def test_both_sides_default_is_not_a_mismatch(self):
        mcp_env = {
            "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1",
            "CLAUDE_RUNWAY_LMSTUDIO_MODEL": "",
            "CLAUDE_RUNWAY_TRACK_SAVINGS": "",
            "CLAUDE_RUNWAY_SAVINGS_DB": "",
        }
        self.assertEqual(self._check(mcp_env, {}), [])

    def test_both_sides_matching_non_default_is_not_a_mismatch(self):
        mcp_env = {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:9999/v1"}
        shell_env = {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:9999/v1"}
        self.assertEqual(self._check(mcp_env, shell_env), [])

    def test_url_mismatch_is_flagged(self):
        mcp_env = {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:9999/v1"}
        shell_env = {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1"}
        mismatches = self._check(mcp_env, shell_env)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].var, "CLAUDE_RUNWAY_LMSTUDIO_URL")
        self.assertEqual(mismatches[0].mcp_json_value, "http://localhost:9999/v1")
        self.assertEqual(mismatches[0].shell_value, "http://localhost:1234/v1")

    def test_url_present_but_blank_on_mcp_side_is_a_literal_broken_value_not_the_default(self):
        # os.environ.get(key, default) only substitutes default when the
        # key is missing entirely -- present-but-blank stays blank in the
        # real subprocess, which is exactly the divergence worth flagging.
        mcp_env = {"CLAUDE_RUNWAY_LMSTUDIO_URL": ""}
        mismatches = self._check(mcp_env, {})  # shell unset -> real default
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].mcp_json_value, "")
        self.assertEqual(mismatches[0].shell_value, doctor_lib.DEFAULT_LMSTUDIO_URL)

    def test_url_missing_from_mcp_env_defaults_identically_on_both_sides(self):
        self.assertEqual(self._check({}, {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1"}), [])

    def test_model_blank_on_mcp_side_vs_pinned_shell_model_is_flagged(self):
        # Real behavior: the MCP server would try to auto-detect (and could
        # error on multiple loaded models) while the hooks force a specific
        # pinned model -- a genuine, worth-surfacing split.
        mcp_env = {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": ""}
        shell_env = {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "google/gemma-3-4b"}
        mismatches = self._check(mcp_env, shell_env)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].mcp_json_value, doctor_lib._AUTO_DETECT_MODEL)
        self.assertEqual(mismatches[0].shell_value, "google/gemma-3-4b")

    def test_model_pinned_on_mcp_side_vs_blank_shell_is_flagged(self):
        mcp_env = {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "google/gemma-3-4b"}
        mismatches = self._check(mcp_env, {})
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].mcp_json_value, "google/gemma-3-4b")
        self.assertEqual(mismatches[0].shell_value, doctor_lib._AUTO_DETECT_MODEL)

    def test_model_matching_pinned_value_is_not_a_mismatch(self):
        mcp_env = {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "google/gemma-3-4b"}
        shell_env = {"CLAUDE_RUNWAY_LMSTUDIO_MODEL": "google/gemma-3-4b"}
        self.assertEqual(self._check(mcp_env, shell_env), [])

    def test_track_savings_on_one_side_only_is_flagged(self):
        mcp_env = {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}
        mismatches = self._check(mcp_env, {})  # shell unset -> off
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].var, "CLAUDE_RUNWAY_TRACK_SAVINGS")
        self.assertEqual(mismatches[0].mcp_json_value, "on")
        self.assertEqual(mismatches[0].shell_value, "off")

    def test_track_savings_accepted_truthy_spellings_are_all_equivalent(self):
        # "1"/"true"/"yes" (case-insensitive, whitespace-trimmed) must not
        # be flagged against each other -- they all mean "on".
        for mcp_val, shell_val in [("1", "true"), ("TRUE", "yes"), (" Yes ", "1")]:
            with self.subTest(mcp_val=mcp_val, shell_val=shell_val):
                mismatches = self._check(
                    {"CLAUDE_RUNWAY_TRACK_SAVINGS": mcp_val}, {"CLAUDE_RUNWAY_TRACK_SAVINGS": shell_val}
                )
                self.assertEqual(mismatches, [])

    def test_savings_db_both_blank_is_not_a_mismatch(self):
        # The default needs no coordination -- both sides derive it identically.
        self.assertEqual(self._check({"CLAUDE_RUNWAY_SAVINGS_DB": ""}, {}), [])

    def test_savings_db_overridden_on_mcp_side_only_is_flagged(self):
        mcp_env = {"CLAUDE_RUNWAY_SAVINGS_DB": "/custom/savings.db"}
        mismatches = self._check(mcp_env, {})
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].mcp_json_value, "/custom/savings.db")
        self.assertEqual(mismatches[0].shell_value, str(self.home_dir / ".claude" / "claude-runway" / "savings.db"))

    def test_savings_db_overridden_differently_on_both_sides_is_flagged(self):
        mcp_env = {"CLAUDE_RUNWAY_SAVINGS_DB": "/custom/a.db"}
        shell_env = {"CLAUDE_RUNWAY_SAVINGS_DB": "/custom/b.db"}
        mismatches = self._check(mcp_env, shell_env)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].mcp_json_value, "/custom/a.db")
        self.assertEqual(mismatches[0].shell_value, "/custom/b.db")

    def test_savings_db_overridden_identically_on_both_sides_is_not_a_mismatch(self):
        mcp_env = {"CLAUDE_RUNWAY_SAVINGS_DB": "/custom/a.db"}
        shell_env = {"CLAUDE_RUNWAY_SAVINGS_DB": "/custom/a.db"}
        self.assertEqual(self._check(mcp_env, shell_env), [])

    def test_savings_db_tilde_path_matches_its_expanded_absolute_equivalent(self):
        # Regression for the finding from PR #119 review: this previously
        # compared the raw override strings as-is, so "~/savings.db" and
        # "/home/testuser/savings.db" -- the SAME real database, per
        # savings_ledger.resolve_db_path()'s own expanduser() call -- were
        # reported as a mismatch.
        mcp_env = {"CLAUDE_RUNWAY_SAVINGS_DB": "~/savings.db"}
        shell_env = {"CLAUDE_RUNWAY_SAVINGS_DB": str(self.home_dir / "savings.db")}
        self.assertEqual(self._check(mcp_env, shell_env), [])

    def test_savings_db_relative_path_is_anchored_to_home_dir_before_comparing(self):
        # resolve_db_path() anchors a still-relative override to Path.home()
        # -- "relative/savings.db" and "/home/testuser/relative/savings.db"
        # are the same real path.
        mcp_env = {"CLAUDE_RUNWAY_SAVINGS_DB": "relative/savings.db"}
        shell_env = {"CLAUDE_RUNWAY_SAVINGS_DB": str(self.home_dir / "relative" / "savings.db")}
        self.assertEqual(self._check(mcp_env, shell_env), [])

    def test_savings_db_tilde_path_still_flagged_against_a_genuinely_different_path(self):
        # The fix must only stop FALSE positives -- a real divergence still
        # needs to be reported.
        mcp_env = {"CLAUDE_RUNWAY_SAVINGS_DB": "~/savings.db"}
        shell_env = {"CLAUDE_RUNWAY_SAVINGS_DB": "/completely/different/path.db"}
        mismatches = self._check(mcp_env, shell_env)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].mcp_json_value, str(self.home_dir / "savings.db"))
        self.assertEqual(mismatches[0].shell_value, "/completely/different/path.db")

    def test_multiple_mismatches_all_reported(self):
        mcp_env = {
            "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:9999/v1",
            "CLAUDE_RUNWAY_TRACK_SAVINGS": "1",
        }
        mismatches = self._check(mcp_env, {})
        self.assertEqual({m.var for m in mismatches}, {"CLAUDE_RUNWAY_LMSTUDIO_URL", "CLAUDE_RUNWAY_TRACK_SAVINGS"})


class RunDoctor(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.target_repo = Path(self._tmpdir.name) / "my-project"
        self.target_repo.mkdir()
        self.home_dir = Path("/home/testuser")

    def _write_mcp_json(self, local_compress_env=None):
        servers = {"qdrant": {"command": "x", "env": {}}}
        if local_compress_env is not None:
            servers["local-compress"] = {"command": "x", "env": local_compress_env}
        (self.target_repo / ".mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")

    def test_no_local_compress_server_reports_not_configured(self):
        self._write_mcp_json(local_compress_env=None)
        result = doctor_lib.run_doctor(self.target_repo, shell_env={}, home_dir=self.home_dir)
        self.assertFalse(result.local_compress_configured)
        self.assertEqual(result.mismatches, [])

    def test_missing_mcp_json_reports_not_configured_not_an_error(self):
        result = doctor_lib.run_doctor(self.target_repo, shell_env={}, home_dir=self.home_dir)
        self.assertFalse(result.local_compress_configured)
        self.assertIsNone(result.config_error)

    def test_malformed_mcp_json_is_reported_as_a_config_error_not_swallowed(self):
        # Regression for the finding from PR #119 review: a malformed
        # .mcp.json (here, "mcpServers": null) previously either returned
        # the same "nothing to check" result as a genuine --qdrant-only
        # setup, or crashed outright -- run_doctor must instead surface it
        # distinctly via config_error so the CLI can exit nonzero.
        (self.target_repo / ".mcp.json").write_text(json.dumps({"mcpServers": None}), encoding="utf-8")
        result = doctor_lib.run_doctor(self.target_repo, shell_env={}, home_dir=self.home_dir)
        self.assertIsNotNone(result.config_error)
        self.assertFalse(result.local_compress_configured)
        self.assertEqual(result.mismatches, [])

    def test_clean_config_reports_no_mismatches(self):
        self._write_mcp_json(local_compress_env={"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1"})
        result = doctor_lib.run_doctor(
            self.target_repo, shell_env={"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1"}, home_dir=self.home_dir
        )
        self.assertTrue(result.local_compress_configured)
        self.assertEqual(result.mismatches, [])

    def test_mismatched_config_is_surfaced(self):
        self._write_mcp_json(local_compress_env={"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"})
        result = doctor_lib.run_doctor(self.target_repo, shell_env={}, home_dir=self.home_dir)
        self.assertTrue(result.local_compress_configured)
        self.assertEqual(len(result.mismatches), 1)
        self.assertEqual(result.mismatches[0].var, "CLAUDE_RUNWAY_TRACK_SAVINGS")

    def test_defaults_to_real_home_dir_when_not_overridden(self):
        # Smoke test only -- just confirms the default parameter path
        # doesn't blow up and produces SOME savings-db-shaped default
        # (Path.home() varies per machine, so we can't assert an exact
        # value here without duplicating resolve_db_path's own logic).
        self._write_mcp_json(local_compress_env={"CLAUDE_RUNWAY_SAVINGS_DB": "/custom/a.db"})
        result = doctor_lib.run_doctor(self.target_repo, shell_env={})
        self.assertTrue(result.local_compress_configured)
        self.assertEqual(len(result.mismatches), 1)
        self.assertTrue(result.mismatches[0].shell_value.endswith("savings.db"))


if __name__ == "__main__":
    unittest.main()
