#!/usr/bin/env python3
"""Tests for libs/transcript_attribution_lib.py: the join of
compression events to the transcript's model and turn, the per-model chars-per-
token ratio, and the per-request counting across a compaction.

Stdlib-only, no network, no live model:

    .venv/bin/python -m unittest discover -s tests

Transcripts are built from small synthetic entries shaped like the real ones
(assistant entries with message.id/model/usage and one line per content block;
user entries with tool_result blocks; a `system`/`compact_boundary` entry).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import transcript_attribution_lib as T  # noqa: E402


def asst(msg_id, model="model-a", ctx=1000, out=50, tool_ids=(), ts="2026-10-01T00:00:00Z", text=False):
    """The lines one assistant turn writes: one per content block, each repeating
    the whole turn's usage (that is how Claude Code writes them)."""
    usage = {"input_tokens": 2, "cache_read_input_tokens": ctx - 2, "cache_creation_input_tokens": 0, "output_tokens": out}
    blocks = [{"type": "tool_use", "id": t, "name": "Bash", "input": {}} for t in tool_ids]
    if text or not blocks:
        blocks.insert(0, {"type": "text", "text": "ok"})
    return [
        {"type": "assistant", "timestamp": ts, "message": {"id": msg_id, "model": model, "usage": usage, "content": [b]}}
        for b in blocks
    ]


def result(tool_use_id, text):
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}]}}


def boundary():
    return {"type": "system", "subtype": "compact_boundary", "content": "Conversation compacted"}


def write_transcript(path, entries):
    flat = []
    for e in entries:
        flat.extend(e if isinstance(e, list) else [e])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(e) for e in flat) + "\n", encoding="utf-8")


class TranscriptTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.main = self.dir / "sess.jsonl"

    def tearDown(self):
        self._tmp.cleanup()


class JoinToTurnAndModel(TranscriptTestCase):
    def test_each_call_gets_its_turn_model_and_timestamp(self):
        write_transcript(self.main, [
            asst("m1", "model-a", tool_ids=["t1"], ts="2026-10-01T00:00:01Z"),
            result("t1", "x" * 10),
            asst("m2", "model-b", tool_ids=["t2"], ts="2026-10-01T00:00:02Z"),
            result("t2", "y" * 10),
            asst("m3", "model-b", ts="2026-10-01T00:00:03Z"),
        ])
        idx = T.read_transcript(self.main)
        self.assertEqual(idx["calls"]["t1"], {"model": "model-a", "turn_id": "m1",
                                              "timestamp": "2026-10-01T00:00:01Z", "turns_after": 2})
        self.assertEqual(idx["calls"]["t2"]["model"], "model-b")
        self.assertEqual(idx["calls"]["t2"]["turn_id"], "m2")

    def test_parallel_calls_on_separate_lines_with_results_between_them(self):
        # Claude Code runs tools as they stream, so a result can land between two
        # content-block lines of the SAME assistant message.
        a = asst("m1", tool_ids=["t1", "t2"])
        write_transcript(self.main, [a[0], result("t1", "r"), a[1], result("t2", "r"), asst("m2")])
        idx = T.read_transcript(self.main)
        self.assertEqual(idx["calls"]["t1"]["turn_id"], "m1")
        self.assertEqual(idx["calls"]["t2"]["turn_id"], "m1")
        self.assertEqual(idx["calls"]["t2"]["turns_after"], 1)

    def test_subagent_transcripts_are_joined_too(self):
        write_transcript(self.main, [asst("m1", "model-a", tool_ids=["t1"])])
        write_transcript(self.dir / "sess" / "subagents" / "agent-x.jsonl", [asst("s1", "model-b", tool_ids=["t9"])])
        idx = T.read_transcript(self.main)
        self.assertEqual(idx["calls"]["t1"]["model"], "model-a")
        self.assertEqual(idx["calls"]["t9"]["model"], "model-b")

    def test_synthetic_messages_and_garbage_lines_are_skipped(self):
        write_transcript(self.main, [asst("m1", "<synthetic>", tool_ids=["t0"]), asst("m2", tool_ids=["t1"])])
        with open(self.main, "a", encoding="utf-8") as f:
            f.write("not json\n")
        idx = T.read_transcript(self.main)
        self.assertNotIn("t0", idx["calls"])
        self.assertIn("t1", idx["calls"])

    def test_missing_or_empty_transcript_is_none(self):
        self.assertIsNone(T.read_transcript(None))
        self.assertIsNone(T.read_transcript(self.dir / "nope.jsonl"))
        write_transcript(self.main, [{"type": "mode"}])
        self.assertIsNone(T.read_transcript(self.main))


class MarkerCrossCheck(TranscriptTestCase):
    def test_compressed_results_are_collected_by_their_marker(self):
        write_transcript(self.main, [
            asst("m1", tool_ids=["t1", "t2"]),
            result("t1", "[compressed 5000 -> 400 chars across 1 chunk(s), ~92% smaller]\n\nsummary"),
            result("t2", "plain output mentioning [compressed 1 -> 1 chars"),
            asst("m2"),
        ])
        self.assertEqual(T.read_transcript(self.main)["marker_ids"], {"t1"})

    def test_a_marker_result_with_no_event_is_reported_and_so_is_a_failed_join(self):
        index = {"calls": {"t1": {"model": "model-a", "turn_id": "m1", "timestamp": "", "turns_after": 1}},
                 "marker_ids": {"t1", "t7"}, "samples": {}}
        events = [
            {"tool_use_id": "t1", "raw_chars": 1000, "out_chars": 100},
            {"tool_use_id": "gone", "raw_chars": 500, "out_chars": 50},   # not in the transcript
            {"raw_chars": 200, "out_chars": 20},                         # no tool_use_id at all
        ]
        r = T.attribute_events(events, index)
        self.assertEqual(r["unjoined"], 2)
        self.assertEqual(r["marker_without_event"], ["t7"])
        by = {row["model"]: row for row in r["by_model"]}
        self.assertEqual(by["model-a"]["event_count"], 1)
        self.assertEqual(by[T.UNATTRIBUTED]["event_count"], 2)
        # The rows always add up to the characters of every event given.
        self.assertEqual(sum(row["raw_chars_sum"] for row in r["by_model"]), 1700)
        self.assertEqual(sum(row["out_chars_sum"] for row in r["by_model"]), 170)
        # The unattributed bucket is listed last.
        self.assertEqual(r["by_model"][-1]["model"], T.UNATTRIBUTED)

    def test_marker_check_uses_every_event_not_only_counted_ones(self):
        index = {"calls": {}, "marker_ids": {"t1"}, "samples": {}}
        r = T.attribute_events([], index, all_events=[{"tool_use_id": "t1", "credited": False}])
        self.assertEqual(r["marker_without_event"], [])


class PerModelRollup(unittest.TestCase):
    def test_rollup_per_model_with_derived_saving_not_summed_per_event(self):
        index = {"calls": {
            "a": {"model": "m1", "turn_id": "1", "timestamp": "", "turns_after": 1},
            "b": {"model": "m1", "turn_id": "2", "timestamp": "", "turns_after": 1},
            "c": {"model": "m2", "turn_id": "3", "timestamp": "", "turns_after": 1},
        }, "marker_ids": set(), "samples": {}}
        events = [
            {"tool_use_id": "a", "raw_chars": 100, "out_chars": 150},   # grew: saving clamps to 0
            {"tool_use_id": "b", "raw_chars": 100, "out_chars": 10},
            {"tool_use_id": "c", "raw_chars": 300, "out_chars": 30},
        ]
        by = {r["model"]: r for r in T.attribute_events(events, index)["by_model"]}
        self.assertEqual((by["m1"]["event_count"], by["m1"]["raw_chars_sum"], by["m1"]["out_chars_sum"]), (2, 200, 160))
        self.assertEqual(by["m1"]["saved_chars"], 40)  # max(0, 200 - 160), not 0 + 90
        self.assertEqual(by["m2"]["saved_chars"], 270)


class RatioDerivation(TranscriptTestCase):
    def _session(self, n_turns, model="model-a"):
        """n_turns requests; every one after the first follows a 3000-character
        result, and context grows by exactly 1200 tokens beyond what the model
        wrote (ratio 2.5)."""
        entries, ctx = [], 1000
        for i in range(n_turns):
            entries.append(asst(f"m{i}", model, ctx=ctx, out=50, tool_ids=[f"t{i}"]))
            entries.append(result(f"t{i}", "z" * 3000))
            ctx += 50 + 1200
        entries.append(asst("last", model, ctx=ctx, out=50))
        write_transcript(self.main, entries)

    def test_ratio_is_the_observed_chars_per_token(self):
        self._session(6)
        idx = T.read_transcript(self.main)
        self.assertEqual(len(idx["samples"]["model-a"]), 6)
        ratios = T.ratio_by_model(idx["samples"])
        self.assertAlmostEqual(ratios["model-a"]["chars_per_token"], 2.5)
        self.assertEqual(ratios["model-a"]["samples"], 6)

    def test_too_few_samples_gives_no_ratio_and_no_derived_figure(self):
        self._session(T.MIN_RATIO_SAMPLES - 1)
        idx = T.read_transcript(self.main)
        self.assertEqual(T.ratio_by_model(idx["samples"]), {})
        event = {"tool_use_id": "t0", "raw_chars": 5000, "out_chars": 500}
        row = T.attribute_events([event], idx)["by_model"][0]
        self.assertEqual(row["model"], "model-a")
        self.assertIsNone(row["derived_tokens"])      # omitted, not 0
        self.assertIsNone(row["chars_per_token"])
        self.assertEqual(row["ratio_samples"], T.MIN_RATIO_SAMPLES - 1)

    def test_derived_tokens_use_the_models_own_ratio(self):
        self._session(T.MIN_RATIO_SAMPLES)
        idx = T.read_transcript(self.main)
        # call t0 is followed by MIN_RATIO_SAMPLES more requests (t1.., and 'last')
        turns_after = idx["calls"]["t0"]["turns_after"]
        self.assertEqual(turns_after, T.MIN_RATIO_SAMPLES)
        row = T.attribute_events([{"tool_use_id": "t0", "raw_chars": 5000, "out_chars": 1000}], idx)["by_model"][0]
        self.assertEqual(row["derived_tokens"], round(4000 * turns_after / 2.5))
        self.assertAlmostEqual(row["chars_per_token"], 2.5)

    def test_samples_that_are_not_clean_tool_result_growth_are_left_out(self):
        entries, ctx = [], 1000
        for i in range(8):
            entries.append(asst(f"m{i}", ctx=ctx, out=50, tool_ids=[f"t{i}"]))
            if i == 2:
                entries.append(result(f"t{i}", "[compressed 9000 -> 3000 chars across 1 chunk(s)]" + "q" * 2950))
            elif i == 3:
                entries.append(result(f"t{i}", "z" * 1999))            # below MIN_SAMPLE_CHARS
            elif i == 4:
                entries.append(result(f"t{i}", "z" * 3000))
                entries.append({"type": "user", "message": {"role": "user", "content": "a typed prompt"}})
            elif i == 5:
                entries.append(result(f"t{i}", [{"type": "image", "source": {}}]))  # not characters
            else:
                entries.append(result(f"t{i}", "z" * 3000))
            ctx += 50 + 1200
        entries.append(asst("last", ctx=ctx, out=50))
        write_transcript(self.main, entries)
        # Samples form only after results 0, 1, 6, 7 (the others are excluded).
        self.assertEqual(len(T.read_transcript(self.main)["samples"]["model-a"]), 4)

    def test_a_model_change_between_two_requests_is_not_a_sample(self):
        write_transcript(self.main, [
            asst("m0", "model-a", ctx=1000, tool_ids=["t0"]), result("t0", "z" * 3000),
            asst("m1", "model-b", ctx=2250),
        ])
        self.assertEqual(T.read_transcript(self.main)["samples"], {})

    def test_a_shrinking_context_is_not_a_sample(self):
        write_transcript(self.main, [
            asst("m0", ctx=5000, tool_ids=["t0"]), result("t0", "z" * 3000), asst("m1", ctx=900),
        ])
        self.assertEqual(T.read_transcript(self.main)["samples"], {})


class CountingAcrossCompaction(TranscriptTestCase):
    def test_a_saving_counts_once_per_request_until_the_session_compacts(self):
        write_transcript(self.main, [
            asst("m1"),
            asst("m2", tool_ids=["early"]), result("early", "r"),
            asst("m3"), asst("m4"),
            boundary(),
            asst("m5", tool_ids=["late"]), result("late", "r"),
            asst("m6"),
        ])
        calls = T.read_transcript(self.main)["calls"]
        # 'early' was in context for m3 and m4; the compaction then dropped it.
        self.assertEqual(calls["early"]["turns_after"], 2)
        # 'late' is carried by m6 only.
        self.assertEqual(calls["late"]["turns_after"], 1)

    def test_with_no_compaction_a_saving_counts_until_the_session_ends(self):
        write_transcript(self.main, [asst("m1", tool_ids=["t"]), result("t", "r"), asst("m2"), asst("m3"), asst("m4")])
        self.assertEqual(T.read_transcript(self.main)["calls"]["t"]["turns_after"], 3)

    def test_a_call_in_the_last_request_was_never_carried(self):
        write_transcript(self.main, [asst("m1"), asst("m2", tool_ids=["t"])])
        self.assertEqual(T.read_transcript(self.main)["calls"]["t"]["turns_after"], 0)

    def test_the_summary_entry_alone_also_marks_a_compaction(self):
        write_transcript(self.main, [
            asst("m1", tool_ids=["t"]), result("t", "r"), asst("m2"),
            {"type": "user", "isCompactSummary": True, "message": {"role": "user", "content": "summary"}},
            asst("m3"),
        ])
        self.assertEqual(T.read_transcript(self.main)["calls"]["t"]["turns_after"], 1)

    def test_a_ratio_sample_is_never_taken_across_a_compaction(self):
        write_transcript(self.main, [
            asst("m1", ctx=1000, tool_ids=["t"]), result("t", "z" * 3000), boundary(), asst("m2", ctx=2250),
        ])
        self.assertEqual(T.read_transcript(self.main)["samples"], {})

    def test_derived_tokens_scale_with_the_requests_carrying_the_result(self):
        index = {"calls": {
            "x": {"model": "m", "turn_id": "1", "timestamp": "", "turns_after": 3},
            "y": {"model": "m", "turn_id": "2", "timestamp": "", "turns_after": 0},
        }, "marker_ids": set(), "samples": {"m": [2.0] * 5}}
        events = [{"tool_use_id": "x", "raw_chars": 1100, "out_chars": 100},
                  {"tool_use_id": "y", "raw_chars": 9100, "out_chars": 100}]
        row = T.attribute_events(events, index)["by_model"][0]
        self.assertEqual(row["derived_tokens"], round(1000 * 3 / 2.0))  # y was never carried: adds 0


if __name__ == "__main__":
    unittest.main()
