#!/usr/bin/env python3
"""Tests for the per-model split in libs/savings_ledger.py (schema
v9): the migration from every earlier schema version, the roll-up into
`session_models`, accumulation across a resumed session, and the display.

The join and ratio logic itself is tested in tests/test_transcript_attribution_lib.py;
here the transcript index is handed in as a plain dict.

Stdlib-only, no network:

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import sqlite3
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import savings_ledger as L  # noqa: E402
# Module import (not `from ... import Class`) so unittest does not collect the
# borrowed fixtures' own tests a second time from this module.
import test_savings_ledger as base  # noqa: E402


def _call(model, after):
    return {"model": model, "turn_id": "msg", "timestamp": "2026-10-01T00:00:00Z", "turns_after": after}


def _index(samples=None):
    return {
        "calls": {"ta": _call("model-a", 2), "tb": _call("model-a", 1), "tc": _call("model-b", 4)},
        "marker_ids": set(),
        "samples": samples if samples is not None else {"model-a": [2.0] * 5, "model-b": [3.0, 3.0]},
    }


class SessionModelsV9(base.SavingsLedgerTestCase):
    # Read by the borrowed `_make_v7_db` below through `self`.
    _V7_SESSIONS = base.SchemaOverheadRemovedV8._V7_SESSIONS
    _COLUMNS = ["session_id", "model", "event_count", "raw_chars_sum", "out_chars_sum", "saved_chars",
                "derived_tokens", "chars_per_token", "ratio_samples"]

    def _record(self, sid="s1"):
        L.record_event(sid, "hook:Bash", 1000, 200, True, tool_use_id="ta")
        L.record_event(sid, "hook:Bash", 600, 100, True, tool_use_id="tb")
        L.record_event(sid, "hook:Grep", 900, 90, True, tool_use_id="tc")
        L.record_event(sid, "hook:Grep", 400, 40, True, tool_use_id="not-in-transcript")

    def _rows(self, sid="s1"):
        conn = sqlite3.connect(str(L.resolve_db_path()))
        rows = {r[1]: r for r in conn.execute("SELECT * FROM session_models WHERE session_id = ?", (sid,))}
        conn.close()
        return rows

    # -- migration from every earlier schema ---------------------------------

    def _assert_migrated(self, conn):
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], L.SCHEMA_VERSION)
        self.assertEqual([r[1] for r in conn.execute("PRAGMA table_info(session_models)")], self._COLUMNS)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM session_models").fetchone()[0], 0)

    def test_a_fresh_database_has_the_table_in_its_final_shape(self):
        conn = L._connect()
        self._assert_migrated(conn)
        conn.close()

    def test_v5_v6_and_v7_databases_reach_the_current_schema(self):
        makers = {
            5: lambda: base.ExactCharCountsAndToolUseId._make_v5_db(self),
            6: lambda: base.ConsistentCharacterFigures._make_v6_db(self),
            7: lambda: base.SchemaOverheadRemovedV8._make_v7_db(self),
        }
        for version, make in makers.items():
            with self.subTest(version=version):
                if L.resolve_db_path().exists():
                    L.resolve_db_path().unlink()
                make()
                conn = L._connect()
                self._assert_migrated(conn)
                conn.close()

    def test_a_v8_database_reaches_the_current_schema(self):
        base.SchemaOverheadRemovedV8._make_v7_db(self)
        with mock.patch.object(L, "SCHEMA_VERSION", 8):
            L._connect().close()  # exactly v8: no session_models migration has run
        raw = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(raw.execute("PRAGMA user_version").fetchone()[0], 8)
        raw.close()
        conn = L._connect()
        self._assert_migrated(conn)
        conn.close()

    def test_migration_keeps_existing_rows_and_the_table_works_afterwards(self):
        base.SchemaOverheadRemovedV8._make_v7_db(self)
        L.record_event("s1", "hook:Bash", 1000, 200, True, tool_use_id="ta")
        L.finalize_owned_session("s1", "p", attribution=_index())
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(conn.execute("SELECT raw_chars_sum FROM sessions WHERE session_id = 'known'").fetchone(), (5000,))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM session_models").fetchone()[0], 1)
        conn.close()

    def test_reconnecting_to_a_current_database_changes_nothing(self):
        self._record()
        L.finalize_owned_session("s1", "p", attribution=_index())
        before = self._rows()
        L._connect().close()
        self.assertEqual(self._rows(), before)

    # -- roll-up ----------------------------------------------------------

    def test_rollup_writes_one_row_per_model_plus_the_unattributed_bucket(self):
        self._record()
        agg = L.finalize_owned_session("s1", "p", attribution=_index())
        rows = self._rows()
        self.assertEqual(set(rows), {"model-a", "model-b", ""})
        a = rows["model-a"]
        self.assertEqual(a[2:6], (2, 1600, 300, 1300))
        # derived: (800 * 2 + 500 * 1) / 2.0 chars per token, from 5 samples
        self.assertEqual(a[6:9], (1050, 2.0, 5))
        # model-b has only 2 samples: no ratio, so NULL -- not 0
        self.assertEqual(rows["model-b"][2:9], (1, 900, 90, 810, None, None, 2))
        # the event whose id is not in the transcript lands in the unattributed bucket
        self.assertEqual(rows[""][2:9], (1, 400, 40, 360, None, None, 0))
        # the rows add up to the session
        conn = sqlite3.connect(str(L.resolve_db_path()))
        self.assertEqual(sum(r[3] for r in rows.values()),
                         conn.execute("SELECT raw_chars_sum FROM sessions").fetchone()[0])
        conn.close()
        self.assertEqual(agg["attribution"], {"unjoined": 1, "marker_without_event": []})
        self.assertEqual([r["model"] for r in agg["by_model"]], ["model-a", "model-b", ""])

    def test_no_transcript_means_no_model_rows(self):
        self._record()
        agg = L.finalize_owned_session("s1", "p")
        self.assertEqual(self._rows(), {})
        self.assertNotIn("by_model", agg)

    def test_replacing_a_session_replaces_its_model_rows(self):
        self._record()
        L.finalize_session("s1", "p", delete_jsonl=False, attribution=_index())
        self.assertEqual(len(self._rows()), 3)
        L.finalize_session("s1", "p", attribution=None)  # a roll-up with no transcript
        self.assertEqual(self._rows(), {})

    def test_query_helpers_return_the_stored_rows(self):
        self._record()
        L.finalize_owned_session("s1", "p", attribution=_index())
        rows = L.query_session_model_breakdown("s1")
        self.assertEqual([r["model"] for r in rows], ["model-a", "model-b", ""])
        recent = L.query_recent_model_breakdown("p")
        self.assertEqual([r["model"] for r in recent["models"]], ["model-a", "model-b", ""])
        self.assertIsNone(L.query_recent_model_breakdown("other-project"))

    # -- accumulation -----------------------------------------------------

    def test_a_resumed_session_adds_exact_characters_but_drops_the_derived_figure(self):
        # The first roll-up's events are gone, and the transcript
        # has grown since, so a stored derived figure cannot be carried across.
        L.record_event("s1", "hook:Bash", 1000, 200, True, tool_use_id="ta")
        L.finalize_owned_session("s1", "p", attribution=_index())
        self.assertEqual(self._rows()["model-a"][6], 800 * 2 / 2.0)
        L.record_event("s1", "hook:Bash", 600, 100, True, tool_use_id="tb")
        L.finalize_owned_session("s1", "p", attribution=_index())
        a = self._rows()["model-a"]
        self.assertEqual(a[2:6], (2, 1600, 300, 1300))
        self.assertIsNone(a[6])

    def test_a_resume_adding_a_new_model_also_has_no_derived_figure(self):
        samples = {"model-a": [2.0] * 5, "model-b": [3.0] * 5}
        L.record_event("s1", "hook:Bash", 1000, 200, True, tool_use_id="ta")
        L.finalize_owned_session("s1", "p", attribution=_index(samples=samples))
        L.record_event("s1", "hook:Grep", 900, 90, True, tool_use_id="tc")
        L.finalize_owned_session("s1", "p", attribution=_index(samples=samples))
        rows = self._rows()
        self.assertIsNone(rows["model-b"][6])  # covers only the new events, not the session
        self.assertEqual(rows["model-b"][2], 1)

    def test_a_resume_invalidates_every_stored_row_and_the_returned_summary_rows(self):
        # model-b has no new event in the resume but its
        # stored derived figure is just as stale; the summary must agree with the DB.
        samples = {"model-a": [2.0] * 5, "model-b": [3.0] * 5}
        L.record_event("s1", "hook:Bash", 1000, 200, True, tool_use_id="ta")
        L.record_event("s1", "hook:Grep", 900, 90, True, tool_use_id="tc")
        L.finalize_owned_session("s1", "p", attribution=_index(samples=samples))
        self.assertIsNotNone(self._rows()["model-b"][6])
        L.record_event("s1", "hook:Bash", 600, 100, True, tool_use_id="tb")
        agg = L.finalize_owned_session("s1", "p", attribution=_index(samples=samples))
        self.assertIsNone(self._rows()["model-b"][6])
        self.assertIsNone(self._rows()["model-a"][6])
        self.assertTrue(all(r["derived_tokens"] is None for r in agg["by_model"]))

    def test_the_marker_cross_check_is_skipped_on_a_resume(self):
        # The transcript's marker results cover the whole session
        # but a resume's `events` hold only the new roll-up's.
        index = _index()
        index["marker_ids"] = {"ta", "tb"}
        L.record_event("s1", "hook:Bash", 1000, 200, True, tool_use_id="ta")
        first = L.finalize_owned_session("s1", "p", attribution=index)
        self.assertEqual(first["attribution"]["marker_without_event"], ["tb"])  # a real miss on a first roll-up
        L.record_event("s1", "hook:Bash", 600, 100, True, tool_use_id="tb")
        second = L.finalize_owned_session("s1", "p", attribution=index)
        self.assertEqual(second["attribution"]["marker_without_event"], [])

    def test_a_resume_without_a_transcript_drops_the_split_that_no_longer_adds_up(self):
        # 1,000 characters in the split beside 1,600 in the session.
        L.record_event("s1", "hook:Bash", 1000, 200, True, tool_use_id="ta")
        L.finalize_owned_session("s1", "p", attribution=_index())
        self.assertEqual(len(self._rows()), 1)
        L.record_event("s1", "hook:Bash", 600, 100, True, tool_use_id="tb")
        L.finalize_owned_session("s1", "p")
        self.assertEqual(self._rows(), {})

    def test_a_resume_without_a_transcript_and_without_new_credited_events_keeps_the_split(self):
        L.record_event("s1", "hook:Bash", 1000, 200, True, tool_use_id="ta")
        L.finalize_owned_session("s1", "p", attribution=_index())
        L.record_event("s1", "fetch_url", 5000, 500, False)  # logged, never credited: nothing to cover
        L.finalize_owned_session("s1", "p")
        self.assertEqual(self._rows()["model-a"][2], 1)

    def test_a_session_rolled_up_without_model_rows_does_not_get_partial_ones_on_resume(self):
        # Finalized earlier with no transcript (or before v9): its row has events
        # but no model rows, so rows for only the later events would read as the
        # whole session.
        L.record_event("s1", "hook:Bash", 1000, 200, True, tool_use_id="ta")
        L.finalize_owned_session("s1", "p")
        L.record_event("s1", "hook:Bash", 600, 100, True, tool_use_id="tb")
        L.finalize_owned_session("s1", "p", attribution=_index())
        self.assertEqual(self._rows(), {})

    def test_a_resumed_unsized_historical_session_stays_without_model_rows(self):
        base.SchemaOverheadRemovedV8._make_v7_db(self)  # 'known': 3 credited events
        L._connect().close()
        conn = sqlite3.connect(str(L.resolve_db_path()))
        conn.execute("UPDATE sessions SET raw_chars_sum = 0, out_chars_sum = 0 WHERE session_id = 'known'")  # unsized
        conn.commit()
        conn.close()
        L.record_event("known", "hook:Bash", 1000, 200, True, tool_use_id="ta")
        L.finalize_owned_session("known", "p", attribution=_index())
        self.assertEqual(self._rows("known"), {})

    def test_a_prior_row_with_only_uncredited_events_does_not_block_model_rows(self):
        L.record_event("s1", "fetch_url", 5000, 500, False)
        L.finalize_owned_session("s1", "p", attribution=_index())
        L.record_event("s1", "hook:Bash", 1000, 200, True, tool_use_id="ta")
        L.finalize_owned_session("s1", "p", attribution=_index())
        self.assertEqual(self._rows()["model-a"][2], 1)

    # -- display ----------------------------------------------------------

    def _agg(self):
        self._record()
        return L.finalize_owned_session("s1", "p", attribution=_index())

    def test_session_end_summary_shows_exact_characters_per_model_and_labels_the_derived_figure(self):
        agg = self._agg()
        agg["project"] = "p"
        text = L.format_simple_view(agg, {"project": "p"})
        self.assertIn("By model · this session", text)
        self.assertIn("1,600 → 300 chars", text)
        self.assertIn("derived: ~1.1k tokens", text)
        self.assertIn("2.00 chars/token", text)
        self.assertIn("5 samples", text)
        self.assertIn("not a measurement", text)
        self.assertIn("unattributed", text)
        self.assertIn("1 event(s) could not be matched", text)

    def test_the_derived_line_is_omitted_for_a_model_without_a_ratio(self):
        agg = self._agg()
        agg["project"] = "p"
        lines = L.format_simple_view(agg, {"project": "p"}).splitlines()
        i = next(n for n, line in enumerate(lines) if line.strip().startswith("model-b"))
        self.assertNotIn("derived", lines[i + 1])

    def test_marker_results_with_no_event_are_noted(self):
        row = {"model": "m", "event_count": 1, "raw_chars_sum": 10, "out_chars_sum": 5, "saved_chars": 5,
               "derived_tokens": None, "chars_per_token": None, "ratio_samples": 0}
        text = "\n".join(L._format_model_block([row], "By model", {"unjoined": 0, "marker_without_event": ["x", "y"]}))
        self.assertIn("2 compressed result(s) in the transcript have no ledger event", text)

    def test_nothing_is_shown_without_rows(self):
        self.assertEqual(L._format_model_block([], "By model"), [])

    def test_detail_view_falls_back_to_the_most_recent_finished_session(self):
        self._agg()
        recent = L.query_recent_model_breakdown("p")
        live = {"credited_saved_chars": 0, "raw_chars_sum": 0, "out_chars_sum": 0, "event_count": 0,
                "fetch_url_count": 0, "project": "p"}
        text = L.format_detail_view(live, {"project": "p"}, [], [], [], recent)
        self.assertIn("By model · most recent finished session (", text)
        self.assertIn("model-a", text)
        self.assertNotIn("By model", L.format_detail_view(live, {"project": "p"}, [], [], []))

    def test_json_carries_the_split_only_when_there_is_one(self):
        agg = self._agg()
        doc = json.loads(L.format_json(agg, {"project": "p"}, [], [], []))
        models = {m["model"]: m for m in doc["model_breakdown"]["models"]}
        self.assertEqual(doc["model_breakdown"]["source"], "this_session")
        self.assertEqual(models["model-a"]["derived_tokens"], 1050)
        self.assertIsNone(models["model-b"]["derived_tokens"])
        self.assertEqual(models["model-a"]["raw_chars_sum"], 1600)
        self.assertNotIn("model_breakdown", json.loads(L.format_json({}, {"project": "p"}, [], [], [])))


class MySavingsSkillContent(unittest.TestCase):
    def test_the_installed_skill_does_not_point_at_evaluation_md(self):
        # The skill is installed into ~/.claude/skills/, where the repo's
        # EVALUATION.md does not exist, so a pointer to it cannot be followed.
        skill = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills", "my-savings", "SKILL.md")
        with open(skill, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("EVALUATION.md", text)
        self.assertNotIn("Track D", text)


if __name__ == "__main__":
    unittest.main()
