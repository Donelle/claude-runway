"""Tests for libs/transcript_context_lib.py and tools/measure_startup_context.py.

Stdlib-only, no network. The transcripts are small fixtures shaped like the real
Claude Code format (the entry types and keys seen in real transcripts: an
`attachment` entry of type `deferred_tools_delta` / `deferred_tools_record`, and
`assistant` entries with `message.usage`), written to a temp directory.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import measure_startup_context as cli  # noqa: E402
import transcript_context_lib as T  # noqa: E402


def _delta(added=(), removed=(), readded=()):
    return {"type": "attachment", "attachment": {
        "type": "deferred_tools_delta", "addedNames": list(added),
        "readdedNames": list(readded), "removedNames": list(removed)}}


def _record(names_to_schema):
    return {"type": "attachment", "attachment": {"type": "deferred_tools_record", "entries": [
        {"name": n, "description": d, "input_schema": {"type": "object"}} for n, d in names_to_schema.items()]}}


def _assistant(model="m1", inp=2, read=0, write=0, mid="msg_1", content=None):
    return {"type": "assistant", "message": {
        "id": mid, "model": model, "content": content or [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": inp, "cache_read_input_tokens": read, "cache_creation_input_tokens": write}}}


def _tool_use(name):
    return [{"type": "tool_use", "name": name, "id": "t1", "input": {}}]


class TranscriptFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self._n = 0

    def write(self, entries, raw_extra=""):
        self._n += 1
        path = self.dir / f"s{self._n}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n" + raw_extra, encoding="utf-8")
        return path


class ServerOfTests(unittest.TestCase):
    def test_extracts_the_server_between_the_mcp_prefix_and_the_tool(self):
        self.assertEqual(T.server_of("mcp__codebase-indexer__sync_repo"), "codebase-indexer")
        self.assertEqual(T.server_of("mcp__claude_ai_Claude_Docs__read"), "claude_ai_Claude_Docs")

    def test_non_mcp_and_malformed_names_have_no_server(self):
        self.assertIsNone(T.server_of("WebFetch"))
        self.assertIsNone(T.server_of("mcp__noseparator"))


class StartOfSessionTests(TranscriptFixture):
    def test_sums_the_three_input_side_counters_of_the_first_assistant_turn(self):
        p = self.write([_assistant(inp=2, read=24000, write=31000), _assistant(inp=9, read=99999, mid="msg_2")])
        s = T.start_of_session(p)
        self.assertEqual(s["total"], 2 + 24000 + 31000)
        self.assertEqual((s["input"], s["cache_read"], s["cache_write"]), (2, 24000, 31000))
        self.assertEqual(s["model"], "m1")

    def test_skips_synthetic_placeholder_turns(self):
        p = self.write([_assistant(model="<synthetic>", inp=0), _assistant(model="real", inp=5, read=10)])
        self.assertEqual(T.start_of_session(p)["model"], "real")

    def test_no_assistant_turn_is_no_data_not_zero(self):
        self.assertIsNone(T.start_of_session(self.write([_delta(added=["WebFetch"])])))

    def test_missing_file_and_garbled_lines_fail_open(self):
        self.assertIsNone(T.start_of_session(self.dir / "nope.jsonl"))
        p = self.write([_assistant(inp=3)], raw_extra="{not json\n")
        self.assertEqual(T.start_of_session(p)["total"], 3)

    def test_undecodable_bytes_are_skipped_not_raised(self):
        # A partially written / corrupt transcript must not abort a whole survey:
        # the bad bytes become an unparseable line and are skipped.
        path = self.dir / "bad.jsonl"
        good = json.dumps(_assistant(inp=3)).encode("utf-8")
        path.write_bytes(b"\xff\xfe not utf-8\n" + good + b"\n")
        self.assertEqual(T.start_of_session(path)["total"], 3)

    def test_a_non_numeric_usage_counter_means_no_data_not_a_crash(self):
        # Valid JSON, malformed counter: the format is internal and
        # may change, so this must read as "unavailable", never abort a survey.
        bad = {"type": "assistant", "message": {"model": "m", "usage": {"input_tokens": "lots"}}}
        self.assertIsNone(T.start_of_session(self.write([bad])))
        self.assertIsNone(T.summarize(self.write([bad])))
        wrapped = {"type": "assistant", "message": {"model": "m", "usage": {"input_tokens": [1]}}}
        self.assertIsNone(T.start_of_session(self.write([wrapped])))

    def test_a_missing_or_non_string_model_means_no_data_not_an_unknown_bucket(self):
        # An "unknown" bucket could mix models in compare(), and an
        # unhashable model value would crash it.
        for model in (None, "", ["x"], {"a": 1}, 7):
            bad = {"type": "assistant", "message": {"model": model, "usage": {"input_tokens": 3}}}
            self.assertIsNone(T.start_of_session(self.write([bad])), model)
        no_key = {"type": "assistant", "message": {"usage": {"input_tokens": 3}}}
        self.assertIsNone(T.summarize(self.write([no_key])))

    def test_an_infinite_counter_means_no_data_not_a_crash(self):
        p = self.dir / "inf.jsonl"
        p.write_text('{"type": "assistant", "message": {"model": "m", "usage": {"input_tokens": Infinity}}}\n',
                     encoding="utf-8")
        self.assertIsNone(T.start_of_session(p))

    def test_deferred_names_are_those_listed_before_the_first_turn_only(self):
        p = self.write([
            _delta(added=["a", "b", "c"], removed=[]),
            _delta(added=[], removed=["b"], readded=["d"]),
            _assistant(),
            _delta(added=["late"]),
        ])
        self.assertEqual(T.start_of_session(p)["deferred_names"], {"a", "c", "d"})


class SummarizeTests(TranscriptFixture):
    def test_counts_toolkit_tools_among_the_deferred_names(self):
        p = self.write([
            _delta(added=["WebFetch", "mcp__qdrant__qdrant-find", "mcp__memory-bank__recall", "mcp__github__x"]),
            _assistant(inp=1, read=100),
        ])
        s = T.summarize(p)
        self.assertEqual(s["deferred_tool_count"], 4)
        self.assertEqual(s["toolkit_deferred_count"], 2)
        self.assertEqual(s["toolkit_servers_deferred"], ["memory-bank", "qdrant"])
        self.assertEqual(s["toolkit_used_not_deferred"], [])

    def test_a_toolkit_tool_called_but_never_listed_as_deferred_is_flagged(self):
        # Ambiguous, so flagged and not classified: its schema was sent up front, or its
        # server connected after the first turn (the transcript does not say which).
        p = self.write([
            _delta(added=["WebFetch"]),
            _assistant(content=_tool_use("mcp__memory-bank__remember")),
        ])
        self.assertEqual(T.summarize(p)["toolkit_used_not_deferred"], ["mcp__memory-bank__remember"])

    def test_a_deferred_toolkit_tool_that_was_called_is_not_flagged(self):
        p = self.write([
            _delta(added=["mcp__qdrant__qdrant-find"]),
            _assistant(content=_tool_use("mcp__qdrant__qdrant-find")),
        ])
        self.assertEqual(T.summarize(p)["toolkit_used_not_deferred"], [])

    def test_no_data_is_none(self):
        self.assertIsNone(T.summarize(self.write([])))


class LoadedSchemaTests(TranscriptFixture):
    def test_sizes_only_toolkit_tools_and_keeps_the_first_record_per_tool(self):
        p = self.write([
            _record({"mcp__qdrant__qdrant-find": "x" * 100, "mcp__github__list_issues": "y" * 100}),
            _record({"mcp__qdrant__qdrant-find": "z" * 5000}),
        ])
        sizes = T.loaded_schema_chars(p)
        self.assertEqual(set(sizes), {"mcp__qdrant__qdrant-find"})
        self.assertGreater(sizes["mcp__qdrant__qdrant-find"], 100)
        self.assertLess(sizes["mcp__qdrant__qdrant-find"], 1000)

    def test_a_malformed_entries_container_is_skipped_not_raised(self):
        # Valid JSON whose `entries` is not a list must not abort a survey.
        for bad in ("oops", 7, {"name": "mcp__qdrant__qdrant-find"}, True):
            p = self.write([{"type": "attachment", "attachment": {"type": "deferred_tools_record", "entries": bad}},
                            _record({"mcp__qdrant__qdrant-find": "d"})])
            self.assertEqual(set(T.loaded_schema_chars(p)), {"mcp__qdrant__qdrant-find"}, bad)

    def test_a_session_that_loaded_nothing_has_no_sizes(self):
        self.assertEqual(T.loaded_schema_chars(self.write([_assistant()])), {})


class OptionalServerTests(TranscriptFixture):
    CORE = ["mcp__qdrant__qdrant-find", "mcp__codebase-indexer__sync_repo",
            "mcp__memory-bank__recall", "mcp__local-compress__compress_file"]

    def test_constants_keep_core_four_and_list_analytics_as_optional(self):
        self.assertEqual(len(T.TOOLKIT_SERVERS), 4)
        self.assertNotIn("analytics", T.TOOLKIT_SERVERS)
        self.assertEqual(T.OPTIONAL_TOOLKIT_SERVERS, ("analytics",))

    def test_analytics_tools_are_toolkit_tools(self):
        names = ["mcp__analytics__get_metrics", "mcp__analytics__savings_summary", "WebFetch"]
        self.assertEqual(T.toolkit_tool_names(names), {"mcp__analytics__get_metrics", "mcp__analytics__savings_summary"})

    def test_a_loaded_analytics_schema_contributes_its_size(self):
        p = self.write([_record({"mcp__analytics__get_metrics": "x" * 100})])
        self.assertEqual(set(T.loaded_schema_chars(p)), {"mcp__analytics__get_metrics"})

    def test_a_call_to_an_analytics_tool_counts_as_toolkit_use(self):
        p = self.write([_delta(added=["WebFetch"]),
                        _assistant(content=_tool_use("mcp__analytics__get_metrics"))])
        self.assertEqual(T.toolkit_tools_used(p), {"mcp__analytics__get_metrics"})
        self.assertEqual(T.summarize(p)["toolkit_used_not_deferred"], ["mcp__analytics__get_metrics"])

    def test_summarize_and_compare_count_deferred_analytics_tools(self):
        w = self.write([_delta(added=["mcp__analytics__get_metrics"]), _assistant(read=1000)])
        wo = self.write([_delta(added=["WebFetch"]), _assistant(read=500)])
        s = T.summarize(w)
        self.assertEqual(s["toolkit_deferred_count"], 1)
        self.assertEqual(s["toolkit_servers_deferred"], ["analytics"])
        res = T.compare([w], [wo])["models"]["m1"]
        self.assertEqual(res["with_toolkit_deferred"], [1])

    def test_explicit_servers_argument_still_restricts_matching(self):
        self.assertEqual(T.toolkit_tool_names(["mcp__analytics__get_metrics"], T.TOOLKIT_SERVERS), set())


class CompareTests(TranscriptFixture):
    def test_difference_is_with_minus_without_per_model(self):
        w1 = self.write([_delta(added=["mcp__qdrant__qdrant-find"]), _assistant(model="m1", read=1000)])
        w2 = self.write([_delta(added=["mcp__qdrant__qdrant-find"]), _assistant(model="m1", read=1200)])
        wo = self.write([_delta(added=["WebFetch"]), _assistant(model="m1", read=1050)])
        res = T.compare([w1, w2], [wo])["models"]["m1"]
        # Each session's total is its 2 input tokens plus the cache read.
        self.assertEqual(res["with_mean"], 1102)
        self.assertEqual(res["without_mean"], 1052)
        self.assertEqual(res["difference"], 50)
        self.assertEqual(res["with_toolkit_deferred"], [1, 1])

    def test_models_are_never_mixed_and_a_one_sided_model_has_no_difference(self):
        w = self.write([_assistant(model="m1", read=1000)])
        wo = self.write([_assistant(model="m2", read=500)])
        res = T.compare([w], [wo])["models"]
        self.assertIsNone(res["m1"]["difference"])
        self.assertIsNone(res["m2"]["difference"])

    def test_unreadable_transcripts_are_reported_as_skipped(self):
        empty = self.write([])
        res = T.compare([empty], [])
        self.assertEqual(res["skipped"], [str(empty)])
        self.assertEqual(res["models"], {})


class CliTests(TranscriptFixture):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(argv)
        return code, buf.getvalue()

    def test_with_without_reports_the_difference_and_the_caveat(self):
        w = self.write([_delta(added=["mcp__qdrant__qdrant-find"]), _assistant(read=2000)])
        wo = self.write([_delta(added=["WebFetch"]), _assistant(read=1500)])
        code, out = self._run(["--with", str(w), "--without", str(wo)])
        self.assertEqual(code, 0)
        self.assertIn("+500 tokens", out)
        self.assertIn("same model", out)

    def test_a_without_transcript_that_lists_toolkit_tools_is_rejected_with_no_number(self):
        # The controlled figure must not be printed from a session
        # whose own transcript shows the toolkit was connected.
        w = self.write([_delta(added=["mcp__qdrant__qdrant-find"]), _assistant(read=2000)])
        bad = self.write([_delta(added=["mcp__memory-bank__recall"]), _assistant(read=1500)])
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(io.StringIO()) as err:
            code = cli.main(["--with", str(w), "--without", str(bad)])
        self.assertEqual(code, 2)
        self.assertNotIn("difference", buf.getvalue())
        self.assertIn("names toolkit tools as deferred", err.getvalue())

    def test_a_without_transcript_that_called_a_toolkit_tool_is_rejected_too(self):
        # a call proves the toolkit was connected, even with no
        # first-turn deferred listing (up front, or connected later: neither is "removed").
        w = self.write([_delta(added=["mcp__qdrant__qdrant-find"]), _assistant(read=2000)])
        bad = self.write([_delta(added=["WebFetch"]),
                          _assistant(read=1500, content=_tool_use("mcp__memory-bank__recall"))])
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(io.StringIO()) as err:
            code = cli.main(["--with", str(w), "--without", str(bad)])
        self.assertEqual(code, 2)
        self.assertNotIn("difference", buf.getvalue())
        self.assertIn("called a toolkit tool", err.getvalue())

    def test_with_without_flags_a_toolkit_tool_that_was_not_deferred(self):
        w = self.write([_delta(added=["WebFetch"]), _assistant(content=_tool_use("mcp__memory-bank__recall"))])
        wo = self.write([_assistant()])
        _, out = self._run(["--with", str(w), "--without", str(wo)])
        self.assertIn("not listed as deferred", out)

    def test_survey_groups_sessions_and_says_it_is_observational(self):
        proj = self.dir / "proj"
        proj.mkdir()
        (proj / "a.jsonl").write_text("\n".join(json.dumps(e) for e in [
            _delta(added=["mcp__qdrant__qdrant-find", "WebFetch"]), _record({"mcp__qdrant__qdrant-find": "d"}),
            _assistant(read=3000)]), encoding="utf-8")
        (proj / "b.jsonl").write_text("\n".join(json.dumps(e) for e in [
            _delta(added=["WebFetch"]), _assistant(read=2000)]), encoding="utf-8")
        # No deferred-tools data at all: not comparable, must be left out of both groups.
        (proj / "c.jsonl").write_text(json.dumps(_assistant(read=9999)), encoding="utf-8")
        code, out = self._run(["--survey", str(self.dir)])
        self.assertEqual(code, 0)
        self.assertRegex(out, r"listed as deferred at the first turn\s*: 1 session")
        self.assertRegex(out, r"no toolkit tool listed \(absence NOT verified\)\s*: 1 session")
        self.assertIn("OBSERVATIONAL", out)
        self.assertIn("+1,000 tokens", out)
        self.assertIn("1 of 3 session(s) that parsed recorded", out)
        # Per-server completeness, not just "at least one tool".
        self.assertRegex(out, r"qdrant: 1")
        self.assertRegex(out, r"memory-bank: 0")
        self.assertRegex(out, r"all 4 servers: 0")

    def test_survey_all_servers_line_ignores_the_optional_analytics_server(self):
        # A session with the four core servers and no analytics still counts as "all 4";
        # analytics is listed on its own line with its own count.
        proj = self.dir / "proj"
        proj.mkdir()
        core = OptionalServerTests.CORE
        (proj / "core.jsonl").write_text("\n".join(json.dumps(e) for e in [
            _delta(added=core), _assistant(read=100)]), encoding="utf-8")
        (proj / "tracked.jsonl").write_text("\n".join(json.dumps(e) for e in [
            _delta(added=core + ["mcp__analytics__get_metrics"]), _assistant(read=100)]), encoding="utf-8")
        _, out = self._run(["--survey", str(self.dir)])
        self.assertRegex(out, r"all 4 servers: 2")
        self.assertRegex(out, r"analytics \(optional[^)]*\): 1")
        self.assertNotIn("all_four", out)

    def test_survey_does_not_guess_a_session_that_called_a_toolkit_tool_into_either_group(self):
        # Its server may simply have connected after the first turn, so a call with no
        # first-turn deferred listing proves neither "present" nor "absent".
        proj = self.dir / "proj"
        proj.mkdir()
        (proj / "late.jsonl").write_text("\n".join(json.dumps(e) for e in [
            _delta(added=["WebFetch"]), _assistant(read=500, content=_tool_use("mcp__qdrant__qdrant-find"))]),
            encoding="utf-8")
        _, out = self._run(["--survey", str(self.dir)])
        self.assertRegex(out, r"unclassified \(called a toolkit tool, none listed\)\s*: 1 session")
        self.assertRegex(out, r"listed as deferred at the first turn\s*: 0 session")
        self.assertRegex(out, r"absence NOT verified\)\s*: 0 session")
        self.assertIn("connected after the first turn", out)

    def test_survey_counts_schema_loads_from_sessions_outside_the_listed_group(self):
        # A deferred_tools_record is direct evidence of a load, so it counts even in a
        # session the survey could not classify.
        proj = self.dir / "proj"
        proj.mkdir()
        (proj / "late.jsonl").write_text("\n".join(json.dumps(e) for e in [
            _delta(added=["WebFetch"]), _record({"mcp__qdrant__qdrant-find": "d"}),
            _assistant(read=500, content=_tool_use("mcp__qdrant__qdrant-find"))]), encoding="utf-8")
        _, out = self._run(["--survey", str(self.dir)])
        self.assertRegex(out, r"unclassified \(called a toolkit tool, none listed\)\s*: 1 session")
        self.assertIn("1 of 1 session(s) that parsed recorded", out)

    def test_survey_of_an_empty_directory_reports_no_data(self):
        code, out = self._run(["--survey", str(self.dir)])
        self.assertEqual(code, 0)
        self.assertIn("nothing to compare", out)


if __name__ == "__main__":
    unittest.main()
