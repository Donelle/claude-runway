"""Tests for libs/autowork_cost_lib.py and tools/autowork_cost.py (issue #455).
Fixture transcripts in a temp dir only -- never the real ~/.claude/projects."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "libs"))
sys.path.insert(0, str(REPO / "tools"))

import autowork_cost_lib as C  # noqa: E402
import autowork_cost as CLI  # noqa: E402
import metrics_lib  # noqa: E402


def iso(s):
    return f"2026-10-10T{s}Z"


def epoch(s):
    return datetime.fromisoformat(iso(s).replace("Z", "+00:00")).timestamp()


def asst(mid, model, usage, ts, blocks=None):
    """One content block per line, each repeating the message's usage (as Claude Code writes it)."""
    blocks = blocks or [{"type": "text", "text": "x"}]
    return [{"type": "assistant", "timestamp": ts,
             "message": {"id": mid, "model": model, "usage": usage, "content": [b]}} for b in blocks]


def usage(i, cc, cr, o):
    return {"input_tokens": i, "cache_creation_input_tokens": cc, "cache_read_input_tokens": cr, "output_tokens": o}


def write_agent(root, session, agent_id, tool_use_id, description, issue, entries):
    d = Path(root) / "proj" / session / "subagents"
    d.mkdir(parents=True, exist_ok=True)
    prompt = {"type": "user", "timestamp": iso("00:00:00"),
              "message": {"role": "user", "content": (
                  "You are autonomously working a GitHub issue on Donelle/claude-runway, end to end.\n\n"
                  f"## Step 0 — determine the target ticket\n{issue}\n(The orchestrator named this issue explicitly.)\n")}}
    flat = [prompt]
    for e in entries:
        flat.extend(e if isinstance(e, list) else [e])
    (d / f"agent-{agent_id}.jsonl").write_text("\n".join(json.dumps(e) for e in flat) + "\n", encoding="utf-8")
    (d / f"agent-{agent_id}.meta.json").write_text(
        json.dumps({"toolUseId": tool_use_id, "description": description}), encoding="utf-8")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "projects"
        self.db = Path(self._tmp.name) / "metrics.db"
        self.store = metrics_lib.MetricsStore(self.db)

    def tearDown(self):
        self._tmp.cleanup()

    def basic(self, agent_id="aaa", tool_use_id="toolu_1", issue=7, description="Autowork ticket 7"):
        write_agent(self.root, "s1", agent_id, tool_use_id, description, issue, [
            asst("m1", "model-a", usage(2, 100, 0, 10), iso("01:00:00"), [
                {"type": "thinking", "thinking": "abc"}, {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]),
            asst("m2", "model-b", usage(3, 0, 100, 20), iso("01:00:30"), [{"type": "tool_use", "id": "t2", "name": "Read", "input": {}}]),
            asst("m3", "model-a", usage(1, 0, 200, 5), iso("01:01:00")),
        ])
        return self.root / "proj" / "s1" / "subagents" / f"agent-{agent_id}.jsonl"


class ParseAttempt(Base):
    def test_totals_per_model_dedup_by_message_id(self):
        a = C.parse_attempt(self.basic())
        # m1 has two content blocks (two lines), counted once.
        self.assertEqual(a["by_model"]["model-a"], {"requests": 2, "input_tokens": 3, "cache_creation_tokens": 100,
                                                    "cache_read_tokens": 200, "output_tokens": 15})
        self.assertEqual(a["by_model"]["model-b"]["cache_read_tokens"], 100)
        self.assertEqual(a["tool_calls"], 2)
        self.assertEqual(a["wall_clock_s"], 3660)
        self.assertEqual(a["thinking_chars"], 3)
        t = C.totals(a["by_model"])
        self.assertEqual(t["total_tokens"], 3 + 100 + 200 + 15 + 3 + 100 + 20)
        self.assertAlmostEqual(t["cache_read_share"], 300 / t["total_tokens"])

    def test_streamed_message_takes_the_max_not_the_sum(self):
        write_agent(self.root, "s1", "bbb", "toolu_2", "d", 8, [
            asst("m1", "model-a", usage(1, 0, 0, 1), iso("01:00:00")),
            asst("m1", "model-a", usage(1, 0, 0, 40), iso("01:00:01")),
        ])
        a = C.parse_attempt(self.root / "proj" / "s1" / "subagents" / "agent-bbb.jsonl")
        self.assertEqual(a["by_model"]["model-a"]["output_tokens"], 40)
        self.assertEqual(a["by_model"]["model-a"]["requests"], 1)

    def test_assistant_messages_without_usage_are_not_a_zero_cost_run(self):
        write_agent(self.root, "s1", "nou", "toolu_nu", "d", 9, [
            asst("m1", "model-a", None, iso("01:00:00"), [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]),
            asst("m2", "model-a", {}, iso("01:00:05")),
        ])
        self.assertIsNone(C.parse_attempt(self.root / "proj" / "s1" / "subagents" / "agent-nou.jsonl"))
        # and `record` therefore marks the row unavailable instead of recording subagent_tokens=0
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            CLI.main(["--db", str(self.db), "--projects-root", str(self.root), "record", "--issue", "9", "--outcome", "merged", "--agent-id", "nou"])
        meta = self.store.detail("autowork")[0]["metadata"]
        self.assertIs(meta["cost"]["available"], False)
        self.assertNotIn("subagent_tokens", meta)

    def test_synthetic_and_garbage_are_skipped_and_empty_is_none(self):
        write_agent(self.root, "s1", "ccc", "toolu_3", "d", 9, [asst("m1", "<synthetic>", usage(5, 5, 5, 5), iso("01:00:00"))])
        self.assertIsNone(C.parse_attempt(self.root / "proj" / "s1" / "subagents" / "agent-ccc.jsonl"))
        self.assertIsNone(C.parse_attempt(self.root / "nope.jsonl"))


class Linking(Base):
    def test_issue_from_the_templates_step_0_number_first(self):
        # The real template substitutes a bare number under the Step 0 heading.
        real = "intro\n## Step 0 — determine the target ticket\n455\n(The orchestrator named this issue.)\n"
        self.assertEqual(C.issue_of(real, ""), 455)
        self.assertEqual(C.issue_of(real + "The issue to work is #9 (x)", "issue 3"), 455)
        # An auto-pick prompt has no number there, so it carries no issue.
        auto = "## Step 0 — determine the target ticket\nPick the highest-priority open issue on x — of ANY label\n"
        self.assertIsNone(C.issue_of(auto, "Autowork ticket 1 (auto-pick)"))

    def test_issue_from_prompt_then_description(self):
        self.assertEqual(C.issue_of("The issue to work is #455 (x)", "Autowork ticket 1"), 455)
        self.assertEqual(C.issue_of("", "issue-12-followup"), 12)
        self.assertIsNone(C.issue_of("", "Autowork ticket 1 (auto-pick)"))

    def test_locate_by_agent_id_and_tool_use_id(self):
        p = self.basic()
        self.assertEqual(C.locate(self.root, agent_id="aaa"), str(p))
        self.assertEqual(C.locate(self.root, tool_use_id="toolu_1"), str(p))
        self.assertIsNone(C.locate(self.root, agent_id="zzz", tool_use_id="nope"))

    def test_other_repositorys_transcript_never_matches_a_row(self):
        write_agent(self.root, "s1", "oth", "toolu_x", "issue-7-followup", 7,
                    [asst("m1", "model-a", usage(1, 0, 0, 1), iso("01:00:00"))])
        path = self.root / "proj" / "s1" / "subagents" / "agent-oth.jsonl"
        path.write_text(path.read_text().replace("Donelle/claude-runway", "SomeOrg/other-repo"), encoding="utf-8")
        a = C.parse_attempt(path)
        self.assertIsNone(a["issue"])
        row = ticket_row("issue-7-010100", iso("01:01:00"), {"issue": 7})
        self.assertIsNone(C.match_rows([row], [a])[id(row)])

    def test_followup_is_its_own_attempt(self):
        self.basic()
        write_agent(self.root, "s1", "fff", "toolu_9", "issue-7-followup", 7,
                    [asst("m1", "model-a", usage(1, 0, 0, 1), iso("02:00:00"))])
        attempts = C.load_attempts(self.root)
        self.assertEqual(sorted(a["kind"] for a in attempts), ["attempt", "followup"])
        self.assertEqual({a["issue"] for a in attempts}, {7})


def ticket_row(session, ts, meta, event="ticket_merged"):
    return {"event_type": event, "value": 1.0, "metadata": meta, "event_timestamp": ts, "session_id": session}


class Backfill(Base):
    def test_matches_nearest_before_and_never_double_claims(self):
        self.basic()  # issue 7, ends 01:01:00
        write_agent(self.root, "s1", "ddd", "toolu_4", "d", 7, [asst("m1", "model-a", usage(1, 0, 0, 1), iso("05:00:00"))])
        attempts = C.load_attempts(self.root)
        rows = [ticket_row("issue-7-010200", iso("01:02:00"), {"issue": 7}),
                ticket_row("issue-7-050100", iso("05:01:00"), {"issue": 7})]
        m = C.match_rows(rows, attempts)
        self.assertEqual(m[id(rows[0])]["agent_id"], "aaa")
        self.assertEqual(m[id(rows[1])]["agent_id"], "ddd")

    def test_pre_row_attempt_beats_one_that_finished_after_the_row(self):
        # Both finish inside the slack window of the row; the real one finished first.
        write_agent(self.root, "s1", "pre", "toolu_p", "d", 7, [asst("m1", "model-a", usage(1, 0, 0, 1), iso("01:00:00"))])
        write_agent(self.root, "s1", "post", "toolu_q", "d", 7, [asst("m1", "model-a", usage(1, 0, 0, 1), iso("01:03:00"))])
        row = ticket_row("issue-7-010100", iso("01:01:00"), {"issue": 7})
        self.assertEqual(C.match_rows([row], C.load_attempts(self.root))[id(row)]["agent_id"], "pre")

    def test_old_transcript_does_not_stand_in_for_a_missing_one(self):
        # The row's own transcript is gone; an attempt of the same issue that ended
        # more than the lookback earlier must leave the row unmatched.
        write_agent(self.root, "s1", "old", "toolu_old", "d", 7, [asst("m1", "model-a", usage(1, 0, 0, 1), iso("01:00:00"))])
        row = ticket_row("issue-7-late", iso("03:30:00"), {"issue": 7})  # 2.5h after, lookback is 1h
        self.assertIsNone(C.match_rows([row], C.load_attempts(self.root))[id(row)])
        near = ticket_row("issue-7-near", iso("01:20:00"), {"issue": 7})
        self.assertEqual(C.match_rows([near], C.load_attempts(self.root))[id(near)]["agent_id"], "old")

    def test_future_candidates_are_a_fallback_earliest_first(self):
        write_agent(self.root, "s1", "f1", "toolu_a", "d", 7, [asst("m1", "model-a", usage(1, 0, 0, 1), iso("01:03:00"))])
        write_agent(self.root, "s1", "f2", "toolu_b", "d", 7, [asst("m1", "model-a", usage(1, 0, 0, 1), iso("01:05:00"))])
        row = ticket_row("issue-7-010100", iso("01:01:00"), {"issue": 7})
        self.assertEqual(C.match_rows([row], C.load_attempts(self.root))[id(row)]["agent_id"], "f1")

    def test_wall_clock_spans_prompt_to_last_entry(self):
        # One response, 30s after the prompt: the old assistant-only span said 0s.
        write_agent(self.root, "s1", "one", "toolu_o", "d", 7, [asst("m1", "model-a", usage(1, 0, 0, 1), iso("00:00:30"))])
        a = C.parse_attempt(self.root / "proj" / "s1" / "subagents" / "agent-one.jsonl")
        self.assertEqual(a["wall_clock_s"], 30)

    def test_row_with_figures_is_untouched_and_keeps_its_attempt(self):
        self.basic()
        attempts = C.load_attempts(self.root)
        done = ticket_row("issue-7-010200", iso("01:02:00"), {"issue": 7, "subagent_tokens": 5, "tool_calls": 1, "wall_clock_s": 2})
        missing = ticket_row("issue-7-010300", iso("01:03:00"), {"issue": 7})
        actions = C.plan_backfill([done, missing], attempts)
        # `done` claims the only attempt first, so `missing` must be unavailable, not a duplicate.
        self.assertEqual(len(actions), 1)
        self.assertIsNone(actions[0]["attempt"])
        self.assertIs(actions[0]["metadata"]["cost"]["available"], False)

    def test_issue_falls_back_to_session_id_when_no_metadata(self):
        self.basic()
        row = {"event_type": "ticket_merged", "value": 1.0, "metadata": None, "event_timestamp": iso("01:02:00"), "session_id": "issue-7-010200"}
        actions = C.plan_backfill([row], C.load_attempts(self.root))
        self.assertEqual(actions[0]["attempt"]["agent_id"], "aaa")

    def test_row_before_the_transcript_ended_does_not_match(self):
        self.basic()
        row = ticket_row("issue-7-000100", iso("00:01:00"), {"issue": 7})
        self.assertIsNone(C.match_rows([row], C.load_attempts(self.root))[id(row)])

    def test_unavailable_cost_event_is_retried_when_a_transcript_appears(self):
        row = ticket_row("issue-7-010200", iso("01:02:00"), {"issue": 7})
        first = C.plan_backfill([row], [])
        self.assertIsNone(first[0]["attempt"])
        gone = ticket_row(row["session_id"], row["event_timestamp"], first[0]["metadata"], C.COST_EVENT)
        # still nothing matching: no duplicate unavailable event piles up
        self.assertEqual(C.plan_backfill([row, gone], []), [])
        # the transcript turns up later: an available event is appended, and the report prefers it
        self.basic()
        again = C.plan_backfill([row, gone], C.load_attempts(self.root))
        self.assertEqual(again[0]["attempt"]["agent_id"], "aaa")
        recovered = ticket_row(row["session_id"], row["event_timestamp"], again[0]["metadata"], C.COST_EVENT)
        rep = C.build_report([row, gone, recovered])
        self.assertEqual([t["status"] for t in rep["tickets"]], ["split"])
        self.assertEqual(C.plan_backfill([row, gone, recovered], C.load_attempts(self.root)), [])

    def test_row_recorded_unavailable_is_not_re_marked_unavailable(self):
        row = ticket_row("issue-9-1", iso("01:02:00"), C.ticket_metadata(None, {"issue": 9}))
        self.assertEqual(C.plan_backfill([row], []), [])

    def test_existing_cost_row_makes_backfill_idempotent(self):
        self.basic()
        attempts = C.load_attempts(self.root)
        row = ticket_row("issue-7-010200", iso("01:02:00"), {"issue": 7})
        first = C.plan_backfill([row], attempts)
        cost = ticket_row(row["session_id"], row["event_timestamp"], first[0]["metadata"], C.COST_EVENT)
        self.assertEqual(C.plan_backfill([row, cost], attempts), [])

    def test_compare_reports_old_vs_transcript(self):
        self.basic()
        total = C.totals(C.parse_attempt(self.root / "proj" / "s1" / "subagents" / "agent-aaa.jsonl")["by_model"])["total_tokens"]
        row = ticket_row("issue-7-010200", iso("01:02:00"), {"issue": 7, "subagent_tokens": total // 2, "tool_calls": 2, "wall_clock_s": 60})
        out = C.compare_existing([row], C.load_attempts(self.root))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["ratio_total"], (total // 2) / total)
        self.assertIn("Median old/transcript ratio", C.format_compare(out))


class Report(Base):
    def test_split_legacy_and_unavailable_are_kept_apart(self):
        a = C.parse_attempt(self.basic())
        split = ticket_row("issue-7-1", iso("01:02:00"), C.ticket_metadata(a, {"issue": 7}))
        legacy = ticket_row("issue-8-1", iso("02:00:00"), {"issue": 8, "subagent_tokens": 1000})
        gone = ticket_row("issue-9-1", iso("03:00:00"), C.ticket_metadata(None, {"issue": 9}))
        late = ticket_row("issue-10-1", iso("04:00:00"), {"issue": 10})
        late_cost = ticket_row("issue-10-1", iso("04:00:00"), {"issue": 10, **C.figures(a), "cost": C.cost_metadata(a, "transcript_backfill")}, C.COST_EVENT)
        rep = C.build_report([split, legacy, gone, late, late_cost])
        statuses = {t["issue"]: t["status"] for t in rep["tickets"]}
        self.assertEqual(statuses, {7: "split", 8: "legacy", 9: "unavailable", 10: "split"})
        self.assertEqual(rep["per_model"]["model-a"]["tickets"], 2)
        self.assertEqual(rep["legacy"], 1)
        self.assertEqual(rep["unavailable"], 1)
        text = C.format_report(rep)
        self.assertIn("UNAVAILABLE", text)
        self.assertIn("cache reads", text)

    def test_recovered_cost_event_beats_an_initially_unavailable_row(self):
        a = C.parse_attempt(self.basic())
        gone = ticket_row("issue-7-1", iso("01:02:00"), C.ticket_metadata(None, {"issue": 7}))
        recovered = ticket_row("issue-7-1", iso("01:02:00"),
                               {"issue": 7, **C.figures(a), "cost": C.cost_metadata(a, "transcript_backfill")}, C.COST_EVENT)
        rep = C.build_report([gone, recovered])
        self.assertEqual([t["status"] for t in rep["tickets"]], ["split"])
        self.assertEqual(rep["unavailable"], 0)
        self.assertIn("model-a", rep["per_model"])

    def test_empty(self):
        self.assertIn("No autowork", C.format_report(C.build_report([])))


class Cli(Base):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = CLI.main(["--db", str(self.db), "--projects-root", str(self.root), *argv])
        return rc, out.getvalue(), err.getvalue()

    def test_record_computes_figures_from_transcript(self):
        self.basic()
        rc, out, _ = self.run_cli("record", "--issue", "7", "--outcome", "merged", "--pr", "70", "--agent-id", "aaa", "--rounds", "2")
        self.assertEqual((rc, out.strip()), (0, "OK"))
        row = self.store.detail("autowork")[0]
        meta = row["metadata"]
        self.assertEqual(row["event_type"], "ticket_merged")
        self.assertEqual(meta["tool_calls"], 2)
        self.assertEqual(meta["wall_clock_s"], 3660)
        self.assertEqual(meta["rounds"], 2)
        t = C.totals(meta["cost"]["by_model"])
        # same scale as the legacy hand-copied rows: everything but cache reads
        self.assertEqual(meta["subagent_tokens"], t["total_tokens"] - t["cache_read_tokens"])
        self.assertGreater(t["cache_read_tokens"], 0)
        self.assertTrue(row["session_id"].startswith("issue-7-"))

    def test_record_by_tool_use_id_and_missing_transcript_still_records(self):
        self.basic()
        self.assertEqual(self.run_cli("record", "--issue", "7", "--outcome", "blocked", "--tool-use-id", "toolu_1")[0], 0)
        rc, out, err = self.run_cli("record", "--issue", "8", "--outcome", "failed", "--agent-id", "missing")
        self.assertEqual((rc, out.strip()), (0, "OK"))
        self.assertIn("not found", err)
        meta = self.store.detail("autowork")[0]["metadata"]
        self.assertIs(meta["cost"]["available"], False)
        self.assertNotIn("subagent_tokens", meta)

    def test_backfill_dry_run_then_apply_then_idempotent(self):
        self.basic()
        self.store.record("autowork", "ticket_merged", 1.0, {"issue": 7}, session_id="issue-7-010200", event_timestamp=iso("01:02:00"))
        self.store.record("autowork", "ticket_merged", 1.0, {"issue": 99}, session_id="issue-99-010200", event_timestamp=iso("01:02:00"))
        rc, out, _ = self.run_cli("backfill")
        self.assertIn("Dry run", out)
        self.assertEqual(len(self.store.detail("autowork")), 2)
        self.run_cli("backfill", "--apply")
        rows = self.store.detail("autowork")
        costs = [r for r in rows if r["event_type"] == "ticket_cost"]
        self.assertEqual(len(costs), 2)
        self.assertEqual(sum(r["value"] for r in costs), 0.0)
        self.assertEqual({r["metadata"]["cost"]["available"] for r in costs}, {True, False})
        # original ticket rows untouched
        self.assertEqual({str(r["metadata"]) for r in rows if r["event_type"] == "ticket_merged"},
                         {str({"issue": 7}), str({"issue": 99})})
        self.run_cli("backfill", "--apply")
        self.assertEqual(len(self.store.detail("autowork")), 4)
        rc, out, _ = self.run_cli("report", "--format", "json")
        self.assertEqual(sorted(t["status"] for t in json.loads(out)["tickets"]), ["split", "unavailable"])

    def test_record_refuses_a_transcript_from_another_issue(self):
        self.basic()  # agent aaa is issue 7
        rc, out, err = self.run_cli("record", "--issue", "8", "--outcome", "merged", "--agent-id", "aaa")
        self.assertEqual((rc, out.strip()), (0, "OK"))
        self.assertIn("#7, not #8", err)
        meta = self.store.detail("autowork")[0]["metadata"]
        self.assertIs(meta["cost"]["available"], False)
        self.assertIn("#7", meta["cost"]["reason"])
        self.assertNotIn("subagent_tokens", meta)

    def test_record_accepts_an_autopick_transcript_with_no_issue(self):
        write_agent(self.root, "s1", "auto", "toolu_au", "Autowork ticket 1 (auto-pick)", "",
                    [asst("m1", "model-a", usage(1, 0, 0, 1), iso("01:00:00"))])
        self.assertIsNone(C.parse_attempt(self.root / "proj" / "s1" / "subagents" / "agent-auto.jsonl")["issue"])
        self.assertEqual(self.run_cli("record", "--issue", "8", "--outcome", "merged", "--agent-id", "auto")[1].strip(), "OK")
        self.assertTrue(self.store.detail("autowork")[0]["metadata"]["cost"]["available"])

    def test_record_rejects_bad_findings(self):
        self.assertEqual(self.run_cli("record", "--issue", "1", "--outcome", "merged", "--agent-id", "a", "--findings", "{")[0], 2)


if __name__ == "__main__":
    unittest.main()
