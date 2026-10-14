#!/usr/bin/env python3
"""
MCP server `analytics`: the opt-in reporting tools over this toolkit's two
local SQLite stores. It holds five tools and nothing else:

  - savings_summary, savings_detail, savings_trend: read the savings tracker
    (libs/savings_ledger.py), the three views `/my-savings` renders.
  - get_metrics, record_metric: read and write the shared cross-tool metrics
    store (libs/metrics_lib.py), which `/my-metrics` and `/my-gh-autowork` use.

None of them uses LM Studio or Qdrant, so this server needs only the `mcp`
package: no embedding model is loaded and no model client is built. It is
registered separately from `local-compress` so that a project that never looks
at savings or metrics does not pay for these five tools' schemas in every
session. `init --track-savings` writes the block; without it the tools do not
exist, and calling one is an unknown-tool error, not an empty result.

Recording does NOT go through this server. Savings events are written by the
`local-compress` server's footer and by hooks/compress_output.py, and rolled
up by hooks/session_end_savings.py, whether or not this server is configured.
Memory-bank's counters reach metrics_lib through libs/memory_events_lib.py.
The one writer that uses a tool here is `record_metric`, for anything
tools/autowork_cost.py record cannot cover.

Project minimum: Python 3.11.

Register in .mcp.json (or let `setup_project.py init --track-savings` do it):
    "analytics": {
      "command": "python",
      "args": ["/absolute/path/to/tools/analytics_mcp_server.py"],
      "env": {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}
    }
CLAUDE_RUNWAY_SAVINGS_DB and CLAUDE_RUNWAY_METRICS_DB are optional overrides.
"""

import os
import sys
from typing import Optional, Union

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from mcp.server.mcpserver import MCPServer

# Both libraries are optional at import time: an install missing one must not
# crash server startup over an opt-in feature. The explicit `= None` in each
# except branch keeps the name always bound, and every tool below checks the
# matching availability flag first, so Pyright can narrow the type.
try:
    import savings_ledger
    _SAVINGS_LEDGER_AVAILABLE = True
except ImportError:
    savings_ledger = None  # type: ignore[assignment]
    _SAVINGS_LEDGER_AVAILABLE = False

try:
    import metrics_lib
    _METRICS_LIB_AVAILABLE = True
except ImportError:
    metrics_lib = None  # type: ignore[assignment]
    _METRICS_LIB_AVAILABLE = False

mcp = MCPServer("analytics")


def _require_savings_tracking() -> Optional[str]:
    if not _SAVINGS_LEDGER_AVAILABLE:
        return "Error: libs/savings_ledger.py is not available in this install -- the savings tracker requires it."
    assert savings_ledger is not None  # _SAVINGS_LEDGER_AVAILABLE implies this
    if not savings_ledger.tracking_enabled():
        return (
            "Savings tracking is off (CLAUDE_RUNWAY_TRACK_SAVINGS is not set). "
            "Set CLAUDE_RUNWAY_TRACK_SAVINGS=1 in this server's env (in .mcp.json) and export it in the shell "
            "environment that launches `claude` too (hook entries in .claude/settings.json have no env field "
            "of their own, so that's the only way compress_output.py sees it)."
        )
    return None


@mcp.tool()
def savings_summary(project: Optional[str] = None, format: str = "text") -> str:
    """
    SIMPLE view of the opt-in savings tracker: this session's characters
    before and after local compression and the percent reduction, plus this
    project's all-time totals. Called by /my-savings with no argument.

    Character counts recorded from now on are exact; older all-time history may
    include approximate figures (rows from before schema v6, or stale token-only
    footers, were reconstructed as tokens x 3.5) or rows with no size at all,
    which are left out and counted as unsized. What they would have cost in tokens is not
    estimated (a token count depends on the model's vocabulary, which is not
    public). The counterfactual itself -- what the raw content would have cost
    vs. the compressed result actually used -- is still observed online, not a
    controlled A/B measurement; see EVALUATION.md's Track D for that distinction.
    Only local-compression events are credited; qdrant-find activity is
    intentionally never counted here (its counterfactual isn't observable).

    project defaults to the current working directory's folder name.

    format controls the output format:
      "text" (default) — human-readable text, same as before.
      "json"           — JSON object with five keys: "session", "project_summary",
                         and "tool_breakdown"/"last_n_sessions"/"all_projects" as
                         empty arrays (those are summary-only fields not populated
                         by this tool; use savings_detail for the full set).
      "csv"            — CSV with a header row and one data row for the session.
    """
    err = _require_savings_tracking()
    if err:
        return err
    assert savings_ledger is not None  # _require_savings_tracking() returning None implies this
    # Validate before any DB access so a bad argument produces a clear input
    # error and never reaches the generic except handler below, which would
    # misleadingly report the issue as database corruption.
    if format not in ("text", "json", "csv"):
        return f"Error: unrecognised format {format!r}. Valid values: 'text', 'json', 'csv'."
    try:
        project = project or savings_ledger.project_name_from_cwd(os.getcwd())
        # project-filtered: a live session belonging to a DIFFERENT project
        # must never be picked here just because its JSONL is the most
        # recently touched file in the shared sessions dir -- see
        # current_session_id's docstring and issue #35. Falling through to
        # the zero-value aggregate below (same as the "no live session at
        # all" case) is what keeps the project label below always accurate.
        session_id = savings_ledger.current_session_id(project=project)
        # Annotated `dict` (not inferred from the ternary's two branches,
        # whose value types differ -- get_live_session_aggregate() returns a
        # bare `dict`, the literal fallback is `dict[str, int]`) so the
        # `session_agg["project"] = project` str assignment below type-checks.
        session_agg: dict = (
            savings_ledger.get_live_session_aggregate(session_id) if session_id
            else {"credited_saved_chars": 0, "raw_chars_sum": 0, "out_chars_sum": 0, "event_count": 0, "fetch_url_count": 0}
        )
        session_agg["project"] = project
        project_summary = savings_ledger.query_project_summary(project)
        if format == "json":
            return savings_ledger.format_json(session_agg, project_summary, [], [], [])
        if format == "csv":
            return savings_ledger.format_csv(session_agg, project_summary, [], [], [], table="session")
        return savings_ledger.format_simple_view(session_agg, project_summary)
    except Exception as e:
        # This is an opt-in, best-effort feature -- a corrupted/locked/missing
        # savings DB must not fail the whole tool call (or worse, the server
        # process). Return a readable error instead of letting it raise.
        return f"Error: could not read savings data ({e}). The savings database may be corrupted or inaccessible; this doesn't affect anything else."


@mcp.tool()
def savings_detail(project: Optional[str] = None, format: str = "text", csv_table: str = "session") -> str:
    """
    DETAIL view of the opt-in savings tracker: this session's per-tool
    breakdown, the per-model split of the most recent finished session (exact
    characters before and after, plus a token figure labelled as derived), a sparkline of the last 10 sessions for this project, and a
    cross-project comparison of characters before and after compression. Called by
    `/my-savings detail`. See savings_summary's docstring for the same
    counterfactual-estimate caveat.

    project defaults to the current working directory's folder name.

    format controls the output format:
      "text" (default) — human-readable text, same as before.
      "json"           — JSON object with the data blocks (including the
                         per-model split when there is one, but not the
                         text view's main-vs-subagent breakdown).
      "csv"            — CSV whose content depends on csv_table (see below).

    csv_table selects which table to export when format="csv":
      "session"  (default) — one-row session summary.
      "tools"              — per-tool breakdown rows.
      "projects"           — cross-project all-time rows.
    """
    err = _require_savings_tracking()
    if err:
        return err
    assert savings_ledger is not None  # _require_savings_tracking() returning None implies this
    # Validate both enum parameters before any DB access so bad arguments
    # produce clear input errors rather than silently returning text or
    # hitting the generic except handler below (which would misleadingly
    # report the issue as database corruption).
    if format not in ("text", "json", "csv"):
        return f"Error: unrecognised format {format!r}. Valid values: 'text', 'json', 'csv'."
    if format == "csv" and csv_table not in ("session", "tools", "projects"):
        return f"Error: unrecognised csv_table {csv_table!r}. Valid values: 'session', 'tools', 'projects'."
    try:
        project = project or savings_ledger.project_name_from_cwd(os.getcwd())
        # See savings_summary's identical comment -- project-filtered lookup,
        # issue #35.
        session_id = savings_ledger.current_session_id(project=project)
        # Annotated `dict` here too (see savings_summary's identical comment)
        # so the "project" key assignment below type-checks either way.
        session_agg: dict
        if session_id:
            session_agg = savings_ledger.get_live_session_aggregate(session_id)
            # Sorted by saved_chars descending -- by_tool.items() is otherwise in
            # dict insertion order (first-tool-seen order), which would make the
            # ordering inconsistent with query_session_tool_breakdown's explicit
            # "ORDER BY saved_chars DESC" for a finalized (post-SessionEnd) session.
            tool_breakdown = sorted(
                (
                    {
                        "tool": t,
                        "event_count": a["event_count"],
                        "saved_chars": a["saved_chars"],
                        # raw/out/credited_event_count feed the reduction % and
                        # credited-vs-uncredited detection in format_detail_view --
                        # all three must be forwarded here or the live (pre-SessionEnd)
                        # path silently suppresses the metric.
                        "raw_chars_sum": a.get("raw_chars_sum", 0),
                        "out_chars_sum": a.get("out_chars_sum", 0),
                        "credited_event_count": a.get("credited_event_count", 0),
                    }
                    for t, a in session_agg.get("by_tool", {}).items()
                ),
                key=lambda t: t["saved_chars"],
                reverse=True,
            )
        else:
            session_agg = {"credited_saved_chars": 0, "raw_chars_sum": 0, "out_chars_sum": 0, "event_count": 0, "fetch_url_count": 0}
            tool_breakdown = []
        session_agg["project"] = project
        project_summary = savings_ledger.query_project_summary(project)
        last_n = savings_ledger.query_last_n_sessions(project, n=10)
        all_projects = savings_ledger.query_all_projects()
        # The session in progress has no transcript to join yet, so the per-model
        # split comes from the most recent finished session (issue #440).
        model_breakdown = savings_ledger.query_recent_model_breakdown(project)
        if format == "json":
            return savings_ledger.format_json(session_agg, project_summary, tool_breakdown, last_n, all_projects, model_breakdown)
        if format == "csv":
            return savings_ledger.format_csv(session_agg, project_summary, tool_breakdown, last_n, all_projects, table=csv_table)
        return savings_ledger.format_detail_view(session_agg, project_summary, tool_breakdown, last_n, all_projects, model_breakdown)
    except Exception as e:
        # Same reasoning as savings_summary's try/except -- best-effort feature,
        # must not fail the tool call over a DB problem.
        return f"Error: could not read savings data ({e}). The savings database may be corrupted or inaccessible; this doesn't affect anything else."


@mcp.tool()
def savings_trend(project: Optional[str] = None, bucket: str = "week", n: int = 12) -> str:
    """
    TREND view of the opt-in savings tracker: characters before and after
    compression, aggregated by day or week rather than individual sessions, so patterns are visible even
    once a project has accumulated enough history that per-session rows are
    too noisy to read. Called by `/my-savings trend`.

    See savings_summary's docstring for the same counterfactual-estimate
    caveat that applies here.

    project defaults to the current working directory's folder name.

    bucket: "week" (default) — groups sessions by ISO week (YYYY-Www).
            "day"            — groups sessions by calendar day (YYYY-MM-DD).

    n: maximum number of buckets to show (default 12, i.e. ~3 months of weeks
       or ~2 weeks of days). Older buckets are dropped when the project has
       more history than this.
    """
    err = _require_savings_tracking()
    if err:
        return err
    assert savings_ledger is not None  # _require_savings_tracking() returning None implies this
    if bucket not in ("day", "week"):
        return f"Error: unrecognised bucket {bucket!r}. Valid values: 'day', 'week'."
    if not isinstance(n, int) or n < 1:
        return "Error: n must be a positive integer."
    try:
        project = project or savings_ledger.project_name_from_cwd(os.getcwd())
        trend_rows = savings_ledger.query_trend(project, bucket=bucket, n=n)
        return savings_ledger.format_trend_view(trend_rows, project, bucket=bucket)
    except Exception as e:
        # Same reasoning as savings_summary's try/except -- best-effort feature,
        # must not fail the tool call over a DB problem.
        return f"Error: could not read savings data ({e}). The savings database may be corrupted or inaccessible; this doesn't affect anything else."


@mcp.tool()
def get_metrics(metric_id: str, view: str = "summary", bucket: str = "week", n: int = 12, format: str = "text", session_id: Optional[str] = None) -> str:
    """
    Reads the shared cross-tool metrics store (issue #208): ONE generic
    SQLite table any domain in this toolkit can write into via
    `libs/metrics_lib.py`'s `MetricsStore` class, aggregated here into
    whichever `view` is requested. Deliberately the ONLY new tool this
    ticket adds (see `libs/metrics_lib.py`'s module docstring for why a
    single generic dispatch tool matters here specifically) -- a future
    domain writing a new `metric_id` never needs a new tool, only a new
    value passed to this one.

    As of issue #210, my-gh-autowork's per-ticket outcome logging writes
    into this store under metric_id="autowork". Any metric_id with no
    recorded events returns an honest "no events recorded yet" result.

    metric_id: the domain to query (e.g. "autowork", "memory-bank") --
      whatever string a future writer used when calling `record()`/
      `increment()`/`decrement()`.

    view controls which aggregation to return:
      "summary"       (default) — total event count + total value across
                       every event_type for this metric_id.
      "by_event_type"           — per-event_type breakdown, largest total
                       value first.
      "trend"                   — day/week-bucketed history (see `bucket`/`n`).
      "detail"        (issue #249) — raw per-event rows, most-recent first,
                       with each row's `metadata` deserialized and rendered
                       (see `n` below for how many rows). This is the view
                       for the richer per-run detail a domain stores in
                       `metadata` (e.g. my-gh-autowork's model/rounds/
                       wall_clock_s/findings) that summary/by_event_type
                       never surface, since those two only ever aggregate
                       `value`.

    bucket ("day" or "week", default "week") only applies to view="trend" —
      ignored otherwise.

    n (default 12) means "max buckets" for view="trend", or "max rows" for
      view="detail" — ignored for "summary"/"by_event_type". Reused across
      both rather than adding a second row-count parameter, since exactly
      one of the two meanings ever applies for a given call.

    session_id (issue #248, extended to "detail" by issue #249), when given,
      narrows view="summary"/"by_event_type"/"detail" down to rows matching
      this exact session_id AND metric_id — e.g. one specific
      `my-gh-autowork` attempt's own `f"issue-{n}-{HHMMSS}"` marker (issue
      #210), instead of that metric_id's entire history. Omitting it (the
      default) keeps today's metric_id-only aggregate/listing unchanged.
      Ignored for view="trend" — per-session filtering doesn't compose with
      time-bucketed grouping (out of scope for issue #248; see that issue
      for why), so a session_id passed alongside view="trend" has no effect
      on the result.

    format controls the output format:
      "text" (default) — human-readable text.
      "json"            — JSON object with keys "view"/"data".
    """
    if not _METRICS_LIB_AVAILABLE:
        return "Error: libs/metrics_lib.py is not available in this install -- get_metrics requires it."
    assert metrics_lib is not None  # _METRICS_LIB_AVAILABLE implies this
    if view not in ("summary", "by_event_type", "trend", "detail"):
        return f"Error: unrecognised view {view!r}. Valid values: 'summary', 'by_event_type', 'trend', 'detail'."
    if format not in ("text", "json"):
        return f"Error: unrecognised format {format!r}. Valid values: 'text', 'json'."
    # Copilot review on this PR: bucket/n are documented as "ignored" for
    # views that don't use them, but were validated unconditionally
    # regardless of view -- so an irrelevant bucket="month" or n=0 rejected
    # a perfectly good view="summary" call, contradicting that doc. Scoped
    # to only the views that actually consume each argument: bucket is
    # trend-only; n applies to trend (bucket count) AND detail (row limit).
    if view == "trend" and bucket not in ("day", "week"):
        return f"Error: unrecognised bucket {bucket!r}. Valid values: 'day', 'week'."
    if view in ("trend", "detail") and (not isinstance(n, int) or n < 1):
        return "Error: n must be a positive integer."
    try:
        store = metrics_lib.MetricsStore()
        if view == "summary":
            summary_data = store.summary(metric_id, session_id=session_id)
            return metrics_lib.format_json("summary", summary_data) if format == "json" else metrics_lib.format_summary_view(summary_data, session_id=session_id)
        if view == "by_event_type":
            by_type_data = store.by_event_type(metric_id, session_id=session_id)
            return metrics_lib.format_json("by_event_type", by_type_data) if format == "json" else metrics_lib.format_by_event_type_view(metric_id, by_type_data, session_id=session_id)
        if view == "detail":
            detail_data = store.detail(metric_id, session_id=session_id, limit=n)
            return metrics_lib.format_json("detail", detail_data) if format == "json" else metrics_lib.format_detail_view(metric_id, detail_data, session_id=session_id)
        # view == "trend"
        trend_data = store.trend(metric_id, bucket=bucket, n=n)
        return metrics_lib.format_json("trend", trend_data) if format == "json" else metrics_lib.format_trend_view(metric_id, trend_data, bucket=bucket)
    except Exception as e:
        # Same reasoning as savings_summary's try/except -- best-effort feature,
        # must not fail the tool call over a DB problem.
        return f"Error: could not read metrics data ({e}). The metrics database may be corrupted or inaccessible; this doesn't affect anything else."


@mcp.tool()
def record_metric(
    metric_id: str,
    event_type: str,
    value: float = 1.0,
    metadata: Optional[Union[str, dict]] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Writes one append-only event to the shared cross-tool metrics store
    (issue #210 — the write counterpart to get_metrics). Any domain in
    this toolkit can call this to record an event. my-gh-autowork's
    per-ticket outcome logging was the first consumer (cut over from
    compact_store by this ticket); it now goes through
    tools/autowork_cost.py record, which writes the same metric_id="autowork"
    rows, and this tool remains for anything that CLI can't cover.

    metric_id: the domain (e.g. "autowork", "memory-bank") — caller-defined,
      no fixed enum.
    event_type: domain-defined event kind (e.g. "ticket_merged", "ticket_blocked").
    value: a signed numeric delta (default 1.0). Must be finite — non-finite
      values (inf/-inf/nan) are rejected by MetricsStore.record() and logged to
      stderr; this tool surfaces that as an "Error:..." return instead.
    metadata: optional domain-specific extra fields, as either a JSON string
      (e.g. '{"issue": 42, "pr": 101, "rounds": 2}') or an already-parsed
      dict — the schema accepts both shapes directly. A string is validated
      as JSON and must decode to an object; an invalid JSON string, or one
      that decodes to a non-object (a list/scalar), is returned as an
      "Error:..." string rather than raising. The dict form exists because,
      back when this parameter was typed as a plain string, some MCP clients
      silently coerced a string argument that happened to look like JSON into
      a native object before this tool ever saw it, bypassing that
      string-only schema entirely and failing every such call outright
      (confirmed live: Claude Code itself does this coercion for a
      JSON-object-shaped string argument). Accepting both shapes now means a
      call succeeds regardless of which one the calling client actually
      sends.
    session_id: optional caller-supplied session identifier — stored on the
      row for later per-session filtering via get_metrics(view="summary"/
      "by_event_type"/"detail", session_id=...) (issue #248, extended to
      "detail" by issue #249).

    Returns "OK" on success, or an "Error:..." string on any failure —
    including input-validation failures (non-finite value, invalid metadata
    JSON) AND DB write failures (MetricsStore.record() returns False, which
    this tool maps to an "Error:..." string). DB write failures are also
    logged to stderr and never raised (fail-open: a broken/locked/corrupt
    metrics db must never break the actual domain operation being measured),
    but the "Error:..." return is the only signal the tool's caller sees.
    """
    if not _METRICS_LIB_AVAILABLE:
        return "Error: libs/metrics_lib.py is not available in this install -- record_metric requires it."
    assert metrics_lib is not None  # _METRICS_LIB_AVAILABLE implies this
    import json as _json
    import math as _math
    if not _math.isfinite(value):
        return f"Error: value must be finite (got {value!r})."
    metadata_dict: Optional[dict] = None
    if isinstance(metadata, dict):
        metadata_dict = metadata
    elif metadata is not None:
        try:
            metadata_dict = _json.loads(metadata)
        except (ValueError, TypeError) as e:
            return f"Error: metadata must be valid JSON if supplied ({e})."
        if not isinstance(metadata_dict, dict):
            return "Error: metadata must be a JSON object (dict), not an array or scalar."
    store = metrics_lib.MetricsStore()
    # record() fails open (returns False, logs to stderr) -- a broken metrics
    # db must never look like a tool call failure for the domain operation
    # that triggered the logging. Map a False return to an "Error:..." string
    # so the SKILL.md "Error:" check can detect it, rather than unconditionally
    # returning "OK" even when the write silently failed (Copilot review on
    # PR #247: an always-"OK" return made the SKILL.md error check unreachable
    # for the most important failure mode — an unwritable or locked db path).
    ok = store.record(
        metric_id=metric_id,
        event_type=event_type,
        value=value,
        metadata=metadata_dict,
        session_id=session_id,
    )
    if not ok:
        return f"Error: could not write metric {metric_id}/{event_type} to the database (details logged to server stderr)."
    return "OK"


if __name__ == "__main__":
    mcp.run()
