"""Tests for libs/context_breakdown_lib.py and tools/context_breakdown.py.
Fixture transcripts in a temp dir only -- never the real ~/.claude/projects."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "libs"))
sys.path.insert(0, str(REPO / "tools"))

import context_breakdown_lib as C  # noqa: E402
import context_breakdown as CLI  # noqa: E402


def usage(total):
    return {"input_tokens": 2, "cache_creation_input_tokens": total - 2, "cache_read_input_tokens": 0, "output_tokens": 5}


def asst(mid, total, blocks):
    """One content block per line, each repeating the message's usage, as Claude Code writes it."""
    return [{"type": "assistant", "message": {"id": mid, "model": "claude-sonnet-5-5", "usage": usage(total), "content": [b]}}
            for b in blocks]


def att(**kw):
    return {"type": "attachment", "attachment": kw}


def user_text(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def tool_result(tid, text):
    return {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}}


def write(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def start_entries():
    return [
        user_text("p" * 1000),
        att(type="skill_listing", content="s" * 3000),
        att(type="instructions", files=[{"path": "a", "content": "i" * 2000}, {"path": "b", "content": "j" * 500}]),
        att(type="deferred_tools_delta", addedLines=["x" * 99, "y" * 99]),
        att(type="environment", snapshot={"cwd": "/w"}),
        att(type="prompt_snapshot", systemPrompt=["q" * 400, "r" * 100], cliPrefix="c" * 10,
            tools=[{"name": "Bash", "description": "d" * 300}, {"name": "Artifact", "description": "e" * 700}]),
    ]


def transcript(rates=None, requests=6):
    """A transcript whose growth per request is exactly `rates` times the recorded
    characters, so the fit has an exact answer. Requests alternate Read and Bash
    results of varying size."""
    rates = rates or {"result_Read": 0.4, "result_Bash": 0.5, "tool_use_input": 0.3}
    entries = start_entries()
    ctx = 5000
    out = list(entries)
    out += asst("m0", ctx, [{"type": "tool_use", "id": "t0", "name": "Read", "input": {"file_path": "/x/README.md"}}])
    prev_use = len(json.dumps({"file_path": "/x/README.md"})) + len("Read")
    names = ["Read"] + ["Read" if i % 2 else "Bash" for i in range(1, requests)]
    for i in range(1, requests):
        name = names[i]
        size = 1000 * i + 300
        out.append(tool_result(f"t{i - 1}", "r" * size))
        ctx = round(ctx + rates["result_" + names[i - 1]] * size
                    + rates["tool_use_input"] * prev_use + 100)
        inp = {"file_path": "/y/lib.py", "limit": 50} if name == "Read" and i % 3 == 0 else {"command": "ls " + "z" * i * 17}
        out += asst(f"m{i}", ctx, [{"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}])
        prev_use = len(json.dumps(inp)) + len(name)
    return out


class StartTests(unittest.TestCase):
    def test_components_and_tools(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            write(p, transcript())
            s = C.start_components(p)
        self.assertEqual(s["context0"], 5000)
        c = s["components"]
        self.assertEqual(c["prompt"], 1000)
        self.assertEqual(c["skill_listing"], 3000)
        self.assertEqual(c["instructions"], 2500)
        self.assertEqual(c["deferred_tool_names"], 199)
        self.assertEqual(c["system_prompt"], 510)
        self.assertEqual(set(s["tools"]), {"Bash", "Artifact"})
        self.assertEqual(c["tool_definitions"], sum(s["tools"].values()))

    def test_itemize_adds_up_exactly_with_tools_as_remainder(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            write(p, transcript())
            it = C.itemize_start(C.start_components(p))
        self.assertEqual(sum(it["items"].values()), it["context0"])
        self.assertEqual(it["items"]["prompt"], round(1000 * 0.37))
        self.assertAlmostEqual(sum(it["tool_definitions_by_tool"].values()), it["items"]["tool_definitions"], delta=1)

    def test_rates_override(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            write(p, transcript())
            it = C.itemize_start(C.start_components(p), {"prompt": 1.0})
        self.assertEqual(it["items"]["prompt"], 1000)

    def test_no_request_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            write(p, [user_text("hi")])
            self.assertIsNone(C.start_components(p))
            self.assertEqual(C.read_requests(p), [])
            self.assertEqual(C.growth_features(p), [])
            self.assertIsNone(C.reconcile(p, {"per_char": {}, "per_request": 0}))

    def test_missing_file_fails_open(self):
        self.assertIsNone(C.start_components("/no/such/file.jsonl"))


class GrowthTests(unittest.TestCase):
    def test_streamed_message_counts_once_with_max_usage(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            lines = asst("m0", 100, [{"type": "thinking", "thinking": "", "signature": "s" * 10},
                                     {"type": "tool_use", "id": "t", "name": "Bash", "input": {}}])
            lines[0]["message"]["usage"] = usage(90)  # an in-progress write of the same message
            write(p, lines)
            reqs = C.read_requests(p)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0]["context"], 100)

    def test_chars_land_on_the_request_that_carries_them(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            write(p, transcript(requests=3))
            g = C.growth_features(p)
        self.assertEqual([x["index"] for x in g], [1, 2])
        # request 1 carries the first Read result and the first response's tool input
        self.assertEqual(g[0]["chars"]["result_Read"], 1300)
        self.assertEqual(g[0]["chars"]["tool_use_input"], len(json.dumps({"file_path": "/x/README.md"})) + 4)
        self.assertEqual(g[1]["chars"]["result_Read"], 2300)
        self.assertEqual(g[0]["chars"]["result_Bash"], 0)

    def test_result_written_between_blocks_of_one_response_goes_to_the_next_request(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            lines = asst("m0", 100, [{"type": "tool_use", "id": "a", "name": "Read", "input": {}}])
            lines.append(tool_result("a", "x" * 50))  # streamed tool execution
            lines += asst("m0", 100, [{"type": "tool_use", "id": "b", "name": "Read", "input": {}}])
            lines += asst("m1", 200, [{"type": "text", "text": "done"}])
            write(p, lines)
            g = C.growth_features(p)
        self.assertEqual(g[0]["chars"]["result_Read"], 50)
        self.assertEqual(g[0]["delta"], 100)

    def test_not_in_context_attachments_add_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            lines = asst("m0", 100, [{"type": "tool_use", "id": "a", "name": "Bash", "input": {}}])
            lines.append(att(type="hook_success", content="h" * 500, stdout="h" * 500))
            lines.append(att(type="hook_additional_context", content="c" * 40))
            lines.append(att(type="agent_listing_delta", addedLines=["l" * 70]))
            lines += asst("m1", 200, [{"type": "text", "text": "done"}])
            write(p, lines)
            chars = C.growth_features(p)[0]["chars"]
        self.assertEqual(chars["hook_context"], 40)
        self.assertEqual(chars["agent_listing"], 70)
        self.assertEqual(sum(chars.values()), 40 + 70 + chars["tool_use_input"])


class DeferredSchemaAndCompactionTests(unittest.TestCase):
    def test_loaded_tool_schema_is_its_own_growth_class(self):
        entries = [{"name": "mcp__x__y", "description": "d" * 300}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            lines = asst("m0", 100, [{"type": "tool_use", "id": "a", "name": "ToolSearch", "input": {}}])
            lines.append(tool_result("a", "stub"))
            lines.append(att(type="deferred_tools_record", entries=entries, toolInputCopies=[{"id": "a", "copy": "wire"}]))
            lines.append(att(type="deferred_tools_record", entries=[], toolInputCopies=[]))
            lines += asst("m1", 900, [{"type": "text", "text": "done"}])
            write(p, lines)
            chars = C.growth_features(p)[0]["chars"]
        self.assertEqual(chars["deferred_schema"], len(json.dumps(entries)))
        self.assertEqual(chars["result_other"], 4)

    def test_growth_row_across_a_compaction_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            lines = asst("m0", 5000, [{"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": "/a/b.py"}}])
            lines.append(tool_result("a", "x" * 400))
            lines += asst("m1", 5200, [{"type": "tool_use", "id": "b", "name": "Read", "input": {"file_path": "/c/d.py"}}])
            lines.append({"type": "system", "subtype": "compact_boundary"})
            lines.append({"type": "user", "isCompactSummary": True, "message": {"role": "user", "content": "summary"}})
            lines += asst("m2", 900, [{"type": "tool_use", "id": "c", "name": "Bash", "input": {}}])
            lines.append(tool_result("c", "y" * 100))
            lines += asst("m3", 1000, [{"type": "text", "text": "done"}])
            write(p, lines)
            g = C.growth_features(p)
            calls = C.read_calls(p)
        self.assertEqual([x["index"] for x in g], [1, 3])  # the row across the compaction is gone
        self.assertEqual(g[0]["delta"], 200)
        by_file = {c["file"]: c for c in calls}
        self.assertEqual(by_file["/a/b.py"]["carried_requests"], 1)  # m1 only, not m2 or m3
        self.assertEqual(by_file["/c/d.py"]["carried_requests"], 0)

    def test_reconcile_growth_excludes_a_compaction_row(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            lines = asst("m0", 5000, [{"type": "tool_use", "id": "a", "name": "Bash", "input": {}}])
            lines += asst("m1", 5200, [{"type": "text", "text": "x"}])
            lines.append({"type": "system", "subtype": "compact_boundary"})
            lines += asst("m2", 900, [{"type": "text", "text": "y"}])
            write(p, lines)
            r = C.reconcile(p, {"per_char": {}, "per_request": 0.0})
        self.assertEqual(r["growth"], 200)


class FitTests(unittest.TestCase):
    def test_least_squares_recovers_a_known_line(self):
        rows = [[float(x), 1.0] for x in range(1, 20)]
        ys = [3.0 * r[0] + 7.0 for r in rows]
        a, b = C.least_squares(rows, ys)
        self.assertAlmostEqual(a, 3.0, places=4)
        self.assertAlmostEqual(b, 7.0, places=2)

    def test_all_zero_column_comes_back_zero(self):
        rows = [[float(x), 0.0] for x in range(1, 10)]
        a, b = C.least_squares(rows, [2.0 * r[0] for r in rows])
        self.assertAlmostEqual(a, 2.0, places=4)
        self.assertEqual(b, 0.0)

    def test_empty(self):
        self.assertEqual(C.least_squares([], []), [])

    def test_calibration_recovers_rates_and_reconcile_closes(self):
        rates = {"result_Read": 0.4, "result_Bash": 0.5, "tool_use_input": 0.3}
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for k, n in enumerate((14, 17, 21)):
                p = Path(d) / f"agent-{k}.jsonl"
                write(p, transcript(rates, requests=n))
                paths.append(p)
            cal = C.calibrate_growth(paths)
            self.assertAlmostEqual(cal["per_char"]["result_Read"], 0.4, delta=0.02)
            self.assertAlmostEqual(cal["per_char"]["result_Bash"], 0.5, delta=0.02)
            self.assertAlmostEqual(cal["per_char"]["tool_use_input"], 0.3, delta=0.05)
            self.assertAlmostEqual(cal["per_request"], 100, delta=15)
            r = C.reconcile(paths[0], cal)
        self.assertEqual(r["growth"], r["final"] - r["context0"])
        self.assertLess(abs(r["remainder"]), 0.02 * r["growth"])
        self.assertEqual(r["explained"] + r["remainder"], r["growth"])

    def test_calibration_with_no_data(self):
        cal = C.calibrate_growth([])
        self.assertEqual(cal["rows"], 0)
        self.assertEqual(cal["per_request"], 0.0)


class ReadTests(unittest.TestCase):
    def test_read_calls_flag_ranged_and_size_and_carry(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            write(p, transcript(requests=5))
            calls = C.read_calls(p)
        by_file = {(c["file"], c["ranged"]): c for c in calls}
        whole = by_file[("/x/README.md", False)]
        self.assertEqual(whole["chars"], 1300)
        self.assertEqual(whole["carried_requests"], 4)
        self.assertTrue(by_file[("/y/lib.py", True)]["ranged"])


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = CLI.main(list(argv))
        return rc, buf.getvalue()

    def test_start_prints_items_and_tools(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agent-a.jsonl"
            write(p, transcript())
            rc, out = self.run_cli("start", str(p))
        self.assertEqual(rc, 0)
        self.assertIn("request 0 = 5,000 tokens", out)
        self.assertIn("tool_definitions", out)
        self.assertIn("Artifact", out)

    def test_calibrate_growth_and_reads_on_a_temp_root(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "proj" / "sess" / "subagents"
            sub.mkdir(parents=True)
            for k in range(3):
                write(sub / f"agent-a{k}.jsonl", transcript(requests=12 + k))
            rc, out = self.run_cli("calibrate", "--projects-root", d)
            self.assertEqual(rc, 0)
            self.assertIn("3 transcripts", out)
            rc, out = self.run_cli("growth", str(sub / "agent-a0.jsonl"), "--projects-root", d)
            self.assertEqual(rc, 0)
            self.assertIn("remainder", out)
            rc, out = self.run_cli("reads", "--projects-root", d, "--min-chars", "1000")
            self.assertEqual(rc, 0)
            self.assertIn("whole-file Reads", out)
            self.assertIn("README", out)


if __name__ == "__main__":
    unittest.main()
