#!/usr/bin/env python3
"""Tests for mcpservers/analytics_mcp_server.py.

Stdlib-only (unittest, no pytest) and no network -- needs neither a live model
nor a real database (every test points the metrics/savings DB at a temp path):

    .venv/bin/python -m unittest discover -s tests

The five tools were moved here from the compression server, so most of the
coverage below is that server's old test classes for them, retargeted.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "mcpservers"))


def _load_analytics_mcp_server():
    """Loads a FRESH copy of the module per call (its own MCPServer instance)."""
    spec = importlib.util.spec_from_file_location(
        "analytics_mcp_server", REPO_ROOT / "mcpservers" / "analytics_mcp_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RegisteredToolSet(unittest.TestCase):
    def test_the_registered_tool_set_is_the_documented_one(self):
        mod = _load_analytics_mcp_server()
        tool_names = {info.name for info in mod.mcp._tool_manager.list_tools()}
        self.assertEqual(
            tool_names,
            {"savings_summary", "savings_detail", "savings_trend", "get_metrics", "record_metric"},
        )

    def test_the_server_is_named_analytics(self):
        # The name is the `mcp__analytics__` prefix permission rules and the
        # compress hook's exemption key off.
        mod = _load_analytics_mcp_server()
        self.assertEqual(mod.mcp.name, "analytics")

    def test_loading_it_imports_no_model_client_or_qdrant_stack(self):
        # The point of a separate server: it needs only `mcp` and the two SQLite
        # libraries, so it must not pull in the compression or Qdrant stacks.
        source = (REPO_ROOT / "mcpservers" / "analytics_mcp_server.py").read_text()
        for forbidden in ("openai", "lmstudio_client", "local_compress_lib", "qdrant_client", "fastembed", "trafilatura"):
            self.assertNotIn(f"import {forbidden}", source)
            self.assertNotIn(f"from {forbidden}", source)


class SavingsToolsReportTrackingOff(unittest.TestCase):
    """With tracking off the three savings tools return an explanatory string
    (not an exception), and never touch the database."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._env_patch = mock.patch.dict(
            os.environ,
            {
                "CLAUDE_RUNWAY_TRACK_SAVINGS": "",
                "CLAUDE_RUNWAY_SAVINGS_DB": str(Path(self._tmpdir.name) / "savings.db"),
            },
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_each_savings_tool_says_tracking_is_off(self):
        mod = _load_analytics_mcp_server()
        for result in (mod.savings_summary(), mod.savings_detail(), mod.savings_trend()):
            self.assertIn("Savings tracking is off", result)
        self.assertFalse((Path(self._tmpdir.name) / "savings.db").exists())

    def test_savings_summary_reads_the_ledger_when_tracking_is_on(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}):
            mod = _load_analytics_mcp_server()
            result = mod.savings_summary(project="no-such-project", format="json")
        data = json.loads(result)
        self.assertIn("project_summary", data)
        self.assertEqual(data["session"]["project"], "no-such-project")

    def test_bad_arguments_are_input_errors(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}):
            mod = _load_analytics_mcp_server()
            self.assertIn("unrecognised format", mod.savings_summary(format="xml"))
            self.assertIn("unrecognised bucket", mod.savings_trend(bucket="month"))


class RecordMetricAcceptsDictOrJsonStringMetadata(unittest.TestCase):
    """record_metric's `metadata` parameter was previously documented and
    typed as a JSON string only, but some MCP clients (confirmed live:
    Claude Code itself) silently coerce a string argument that happens to
    parse as a JSON object into a native dict before this tool ever sees
    it -- bypassing that declared `string` schema type entirely. The fix
    below widens the parameter to accept either shape directly, rather than
    erroring on the coerced-dict case with "metadata must be valid JSON" (a
    `TypeError` from `json.loads(dict)`, caught and misreported as a
    JSON-parse failure even though the value was never invalid to begin
    with)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = str(Path(self._tmpdir.name) / "metrics.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": self._db_path})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_metadata_as_json_string_still_works(self):
        mod = _load_analytics_mcp_server()
        result = mod.record_metric(
            "autowork", "ticket_merged", metadata='{"issue": 211, "pr": 260}'
        )
        self.assertEqual(result, "OK")
        rows = mod.metrics_lib.MetricsStore().detail("autowork")
        self.assertEqual(rows[0]["metadata"], {"issue": 211, "pr": 260})

    def test_metadata_as_already_parsed_dict_is_accepted(self):
        mod = _load_analytics_mcp_server()
        result = mod.record_metric(
            "autowork", "ticket_merged", metadata={"issue": 211, "pr": 260}
        )
        self.assertEqual(result, "OK")
        rows = mod.metrics_lib.MetricsStore().detail("autowork")
        self.assertEqual(rows[0]["metadata"], {"issue": 211, "pr": 260})

    def test_metadata_as_list_is_still_rejected(self):
        mod = _load_analytics_mcp_server()
        result = mod.record_metric("autowork", "ticket_merged", metadata='["not", "a", "dict"]')
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("must be a JSON object", result)

    def test_metadata_as_invalid_json_string_is_still_rejected(self):
        mod = _load_analytics_mcp_server()
        result = mod.record_metric("autowork", "ticket_merged", metadata="{not valid json")
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("must be valid JSON", result)

    def test_metadata_omitted_still_works(self):
        mod = _load_analytics_mcp_server()
        result = mod.record_metric("autowork", "ticket_merged")
        self.assertEqual(result, "OK")


class GetMetricsSessionIdFilter(unittest.TestCase):
    """get_metrics gained an optional session_id passthrough for
    view="summary"/"by_event_type" -- covers the MCP tool surface on top of
    tests/test_metrics_lib.py's own MetricsStore-level coverage of the
    underlying filter."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = str(Path(self._tmpdir.name) / "metrics.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": self._db_path})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_summary_view_narrows_to_session_id(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-248-100000")
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="summary", session_id="issue-248-100000", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(data["event_count"], 1)
        self.assertEqual(data["total_value"], 1.0)

    def test_by_event_type_view_narrows_to_session_id(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-248-100000")
        store.record("autowork", "ticket_blocked", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="by_event_type", session_id="issue-248-100000", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["event_type"], "ticket_merged")

    def test_omitting_session_id_keeps_unfiltered_summary(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-248-100000")
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="summary", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(data["event_count"], 2)

    def test_trend_view_ignores_session_id_without_erroring(self):
        # session_id has no effect on trend() --
        # passing it must not raise or otherwise break the trend view.
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-248-100000")
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="trend", bucket="day", session_id="issue-248-100000", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        # Both rows count toward the trend regardless of session_id -- it's
        # simply not applied there.
        self.assertEqual(sum(r["event_count"] for r in data), 2)

    def test_summary_view_empty_filter_names_the_session_not_the_whole_metric(self):
        """Copilot review (round 4): a session_id filter that
        matches nothing must not claim the metric_id as a whole has no
        events -- it has plenty, just not under this session_id."""
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="summary", session_id="no-such-session", format="text")
        self.assertIn("no-such-session", result)
        self.assertNotIn("No events recorded yet for this metric_id.", result)

    def test_by_event_type_view_empty_filter_names_the_session_not_the_whole_metric(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="by_event_type", session_id="no-such-session", format="text")
        self.assertIn("no-such-session", result)
        self.assertNotIn("No events recorded yet for this metric_id.", result)


class GetMetricsDetailView(unittest.TestCase):
    """get_metrics gained a "detail" view -- raw per-event rows
    with deserialized metadata, most recent first. Covers the MCP tool
    surface on top of tests/test_metrics_lib.py's own MetricsStore-level
    coverage of the underlying detail()/format_detail_view()."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = str(Path(self._tmpdir.name) / "metrics.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": self._db_path})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_detail_view_returns_rows_with_metadata(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, metadata={"model": "claude-sonnet-5", "rounds": 5})

        result = mod.get_metrics("autowork", view="detail", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["metadata"], {"model": "claude-sonnet-5", "rounds": 5})

    def test_n_is_reused_as_the_row_limit(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        for i in range(5):
            store.record("autowork", "x", value=1.0, event_timestamp=f"2026-09-0{i + 1}T00:00:00Z")

        result = mod.get_metrics("autowork", view="detail", n=2, format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["event_timestamp"], "2026-09-05T00:00:00Z")

    def test_detail_view_narrows_to_session_id(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-100000")
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-250-110000")

        result = mod.get_metrics("autowork", view="detail", session_id="issue-249-100000", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["session_id"], "issue-249-100000")

    def test_detail_view_empty_filter_names_the_session_not_the_whole_metric(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="detail", session_id="no-such-session", format="text")
        self.assertIn("no-such-session", result)
        self.assertNotIn("No events recorded yet for this metric_id.", result)

    def test_unrecognised_view_error_lists_detail_as_valid(self):
        mod = _load_analytics_mcp_server()
        result = mod.get_metrics("autowork", view="bogus")
        self.assertIn("detail", result)

    def test_direct_negative_limit_is_rejected_not_unbounded(self):
        # get_metrics's own n<1 check already prevents this via the MCP
        # surface -- this exercises detail()'s own independent guard (see
        # test_metrics_lib.py's Detail class for the direct-library-call
        # reproduction this backs).
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "x", value=1.0)
        with self.assertRaises(ValueError):
            store.detail("autowork", limit=-1)


class GetMetricsBucketAndNScopedToRelevantViews(unittest.TestCase):
    """Copilot review: bucket/n are documented as "ignored" for
    views that don't consume them, but were validated unconditionally --
    an irrelevant bucket="month" or n=0 alongside view="summary" rejected
    an otherwise-valid call. Scoped so only the views that actually use
    each argument validate it."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = str(Path(self._tmpdir.name) / "metrics.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": self._db_path})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_summary_ignores_irrelevant_bad_bucket(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0)
        result = mod.get_metrics("autowork", view="summary", bucket="month", format="json")
        self.assertNotIn("Error:", result)

    def test_summary_ignores_irrelevant_bad_n(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0)
        result = mod.get_metrics("autowork", view="summary", n=0, format="json")
        self.assertNotIn("Error:", result)

    def test_by_event_type_ignores_irrelevant_bad_bucket(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0)
        result = mod.get_metrics("autowork", view="by_event_type", bucket="month", format="json")
        self.assertNotIn("Error:", result)

    def test_detail_ignores_irrelevant_bad_bucket(self):
        mod = _load_analytics_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0)
        result = mod.get_metrics("autowork", view="detail", bucket="month", format="json")
        self.assertNotIn("Error:", result)

    def test_trend_still_validates_bucket(self):
        mod = _load_analytics_mcp_server()
        result = mod.get_metrics("autowork", view="trend", bucket="month")
        self.assertIn("Error:", result)

    def test_trend_still_validates_n(self):
        mod = _load_analytics_mcp_server()
        result = mod.get_metrics("autowork", view="trend", n=0)
        self.assertIn("Error:", result)

    def test_detail_still_validates_n_since_n_is_its_row_limit(self):
        mod = _load_analytics_mcp_server()
        result = mod.get_metrics("autowork", view="detail", n=0)
        self.assertIn("Error:", result)


class WordingForTheMovedToolsStaysInSync(unittest.TestCase):
    """The template and the two skills name the five tools and say where they
    live; this pins that wording so it cannot drift from the server again."""

    _TOOLS = ("savings_summary", "savings_detail", "savings_trend", "get_metrics", "record_metric")

    def _read(self, *parts):
        return (REPO_ROOT.joinpath(*parts)).read_text(encoding="utf-8")

    def test_claude_md_template_has_an_opt_in_analytics_section(self):
        text = self._read("templates", "CLAUDE.md.template")
        self.assertIn("## Analytics (opt-in MCP server)", text)
        section = text.split("## Analytics (opt-in MCP server)", 1)[1]
        for name in self._TOOLS:
            self.assertIn(f"`{name}", section)
        # A missing block means an error, not missing data.
        self.assertIn("unknown-tool error", section)
        self.assertIn("init --track-savings", section)
        for cli in ("tools/memory_bank_report.py", "tools/autowork_cost.py report",
                    "tools/context_breakdown.py", "tools/report_tool_counts.py"):
            self.assertIn(cli, section)

    def test_local_compression_section_no_longer_claims_the_moved_tools(self):
        text = self._read("templates", "CLAUDE.md.template")
        compression = text.split("## Local compression (LM Studio MCP)", 1)[1].split("## Analytics", 1)[0]
        self.assertIn("Five tools from the `local-compress` server", compression)
        self.assertNotIn("Four tools", compression)

    def test_my_savings_names_the_three_savings_tools_and_the_analytics_server(self):
        text = self._read("skills", "my-savings", "SKILL.md")
        for name in ("savings_summary", "savings_detail", "savings_trend"):
            self.assertIn(name, text)
        self.assertIn("`analytics`", text)
        self.assertNotIn("from the `local-compress` server", text)

    def test_my_metrics_lists_writers_points_at_savings_and_the_clis(self):
        text = self._read("skills", "my-metrics", "SKILL.md")
        self.assertIn("`autowork`", text)
        self.assertIn("ticket_cost", text)
        self.assertIn("`memory-bank`", text)
        self.assertIn("/my-savings", text)
        self.assertIn("`analytics`", text)
        self.assertNotIn("from the `local-compress` server", text)
        for cli in ("tools/autowork_cost.py report", "tools/memory_bank_report.py",
                    "tools/context_breakdown.py", "tools/report_tool_counts.py"):
            self.assertIn(cli, text)

    def test_autowork_skill_says_record_metric_needs_the_analytics_server(self):
        text = self._read(".claude", "skills", "my-gh-autowork", "SKILL.md")
        self.assertIn("`analytics` MCP server", text)
        self.assertNotIn("mcp__local-compress__record_metric", text)

    def test_setup_skill_reads_the_metrics_override_from_every_block_that_can_hold_it(self):
        text = self._read("skills", "my-setup-clauderunway", "SKILL.md")
        line = next(l for l in text.splitlines() if l.startswith("- `CLAUDE_RUNWAY_METRICS_DB` from"))
        for block in ("analytics.env", "memory-bank.env", "local-compress.env"):
            self.assertIn(block, line)

    def test_the_mcp_template_has_an_opt_in_analytics_block(self):
        template = json.loads(self._read("templates", "mcp.json.template"))
        block = template["mcpServers"]["analytics"]
        self.assertEqual(block["args"], ["/absolute/path/to/mcpservers/analytics_mcp_server.py"])
        self.assertIn("OPTIONAL", block["_comment"])


if __name__ == "__main__":
    unittest.main()
