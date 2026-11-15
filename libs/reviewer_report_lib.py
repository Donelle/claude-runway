"""
Reads the `codebase-reviewer` database (`reviewer_db_lib.py`) and answers three questions without
hand-written SQL:

  1. Which model combination suits my machine, performance-wise?   -> `review_times` (and `--this-machine`)
  2. What is the response time for model X, per request and per stage?   -> `response_times`
  3. I want my review cycle under N minutes: which combination gets there?   -> `under_target`

A "model combination" is the models of each stage, in stage order (`reasoning:a+b|verifier:c`); it is
derived from what actually ran, so every stage (reasoning, verifier, classifier, and any added later) is
grouped the same way. Every view groups by stage as a value, never by a fixed list of stages.

How a review's time is split: each model load and unload is timed on its own (`load`). A stage's time
as reported here is the stage's wall time MINUS the loads and unloads that happened inside it, so that
load + reasoning + verifying (+ classifying) adds up to the review rather than counting a load twice.
What is left over (grounding, reading files, the model unloads at the very end) is `other`.

Only runs that finished are used for review times; a run that ended in an error, or returned `incomplete`
(a stage got no answer from any reviewer), is in the database but would say nothing about how long a
review takes. Response times use answered requests only; the count of
timed-out and failed requests is shown beside them so a median is never read without it.

Reads raise (`ReportError`) like `metrics_lib`'s: a caller that wants a message catches it. Standard
library only.
"""

import functools
import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Optional

import reviewer_db_lib


class ReportError(Exception):
    """The database cannot be read. The message says where it looked."""


def open_database(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Opens the database read-only. A missing file is a `ReportError` (nothing has been recorded yet), not
    an empty database created by the act of asking."""
    path = db_path or reviewer_db_lib.resolve_db_path()
    if not path.exists():
        raise ReportError(f"no codebase-reviewer database at {path}: nothing has been recorded yet")
    conn = None
    try:
        # as_uri() escapes the path: a `#` or `?` in a file name would otherwise end the path and drop mode=ro.
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=reviewer_db_lib.BUSY_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1 FROM runs LIMIT 1")
    except sqlite3.Error as exc:
        if conn is not None:
            conn.close()
        raise ReportError(f"cannot read the codebase-reviewer database at {path} ({exc})") from exc
    return conn


def percentile(values: list, fraction: float) -> float:
    """Nearest-rank percentile of a non-empty list (p90 is `percentile(v, 0.9)`)."""
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _this_machine_id(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute(
        "SELECT id FROM machines WHERE fingerprint = ?", (reviewer_db_lib.machine_fingerprint(reviewer_db_lib.machine_info()),)
    ).fetchone()
    return row["id"] if row else None


def _machine_clause(conn: sqlite3.Connection, this_machine: bool) -> tuple:
    """`(sql fragment, params)` limiting `runs` (aliased `r`) to this machine; a machine never seen gives no rows."""
    if not this_machine:
        return "", ()
    machine = _this_machine_id(conn)
    if machine is None:
        return " AND 0", ()
    return " AND r.machine_id = ?", (machine,)


# ---------------------------------------------------------------------------
# 2. response time per model and role
# ---------------------------------------------------------------------------


def _column(rows: list, key: str) -> list:
    return [r[key] for r in rows]


def _median(values: list, digits: int = 0) -> Optional[float]:
    """The median of the numbers in `values` (None entries ignored), or None when there are none."""
    numbers = [v for v in values if v is not None]
    if not numbers:
        return None
    result = statistics.median(numbers)
    # Half rounds up (Python's own round() sends 8.5 to 8, which reads as a mistake in a report).
    return round(result + 1e-9, digits) if digits else int(math.floor(result + 0.5))


def _p90(values: list) -> Optional[float]:
    numbers = [v for v in values if v is not None]
    return percentile(numbers, 0.9) if numbers else None


def response_times(conn: sqlite3.Connection, *, model: Optional[str] = None, this_machine: bool = False) -> list:
    """Per model and role: the response time of answered requests (n, median, p90, max, in seconds), how many requests
    timed out or failed, and the timing and size figures of the answered ones, in milliseconds and tokens:

      latency_ms     submission to the final token, measured by the reviewer (includes any wait in LM Studio's queue)
      ttft_ms        time to first token, as LM Studio reports it
      inter_token_ms average time between tokens, from LM Studio's tokens per second
      prompt_tokens, reply_tokens   what LM Studio counted (the reply count includes a reasoning model's thinking)

    each as a median (and p90 for latency and first token). A figure no request reported is None. Sorted by model then role."""
    clause, params = _machine_clause(conn, this_machine)
    sql = ("SELECT q.model, q.role, q.outcome, q.wall_s, q.latency_ms, q.ttft_ms, q.inter_token_ms, q.tokens_per_second, "
           "q.prompt_tokens, q.reply_tokens FROM requests q JOIN runs r ON r.id = q.run_id WHERE 1 = 1" + clause)
    args = list(params)
    if model:
        sql += " AND q.model = ?"
        args.append(model)
    grouped: dict = {}
    for row in conn.execute(sql, args):
        entry = grouped.setdefault((row["model"], row["role"]), {"answered": [], "timed_out": 0, "failed": 0})
        if row["outcome"] == reviewer_db_lib.ANSWERED:
            entry["answered"].append(row)
        elif row["outcome"] == reviewer_db_lib.TIMED_OUT:
            entry["timed_out"] += 1
        else:
            entry["failed"] += 1
    rows = []
    for (name, role), entry in sorted(grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        answered = entry["answered"]
        times = [r["wall_s"] for r in answered]
        column = functools.partial(_column, answered)

        rows.append({
            "model": name,
            "role": role,
            "answered": len(times),
            "median_s": round(statistics.median(times), 2) if times else None,
            "p90_s": round(percentile(times, 0.9), 2) if times else None,
            "max_s": round(max(times), 2) if times else None,
            "timed_out": entry["timed_out"],
            "failed": entry["failed"],
            "latency_ms": {"median": _median(column("latency_ms")), "p90": _p90(column("latency_ms"))},
            "ttft_ms": {"median": _median(column("ttft_ms")), "p90": _p90(column("ttft_ms"))},
            "inter_token_ms": {"median": _median(column("inter_token_ms"), 1)},
            "tokens_per_second": {"median": _median(column("tokens_per_second"), 1)},
            "prompt_tokens": {"median": _median(column("prompt_tokens"))},
            "reply_tokens": {"median": _median(column("reply_tokens"))},
        })
    return rows


# ---------------------------------------------------------------------------
# 1 and 3. review time per model combination
# ---------------------------------------------------------------------------


def _review_rows(conn: sqlite3.Connection, this_machine: bool) -> list:
    """One dict per finished run: combination, wall time, load time, per-stage run time (loads removed), other."""
    clause, params = _machine_clause(conn, this_machine)
    runs = conn.execute(
        "SELECT r.id, r.combination, r.wall_s FROM runs r WHERE r.outcome = 'finished' AND r.combination IS NOT NULL "
        "AND r.combination != ''" + clause,
        params,
    ).fetchall()
    out = []
    for run in runs:
        loads = conn.execute("SELECT started_at, wall_s FROM model_loads WHERE run_id = ?", (run["id"],)).fetchall()
        stages = conn.execute("SELECT stage, started_at, ended_at, wall_s FROM stages WHERE run_id = ?", (run["id"],)).fetchall()
        stage_s: dict = {}
        for stage in stages:
            inside = sum(l["wall_s"] for l in loads if stage["started_at"] <= l["started_at"] <= stage["ended_at"])
            stage_s[stage["stage"]] = stage_s.get(stage["stage"], 0.0) + max(0.0, stage["wall_s"] - inside)
        load_s = sum(l["wall_s"] for l in loads)
        other = max(0.0, run["wall_s"] - load_s - sum(stage_s.values()))
        out.append({"combination": run["combination"], "wall_s": run["wall_s"], "load_s": load_s, "stage_s": stage_s, "other_s": other})
    return out


def review_times(conn: sqlite3.Connection, *, this_machine: bool = False) -> list:
    """Per model combination: how many finished reviews it has, the median review wall time, and the split of
    its middle review (the one whose wall time is the lower median) into load, one part per stage, and other.

    The split is one real review's, so its parts add up to that review's wall time. Medians of each part taken
    on their own would not: three reviews of (load, reasoning, verifier) = (100, 1, 1), (1, 100, 1) and
    (1, 1, 100) all take 102 s, but each part's median is 1 s. With an even number of reviews the middle
    review's wall time can differ a little from the median shown beside it, so the split carries its own.
    Fastest median first."""
    grouped: dict = {}
    for row in _review_rows(conn, this_machine):
        grouped.setdefault(row["combination"], []).append(row)
    result = []
    for combination, rows in grouped.items():
        middle_wall = statistics.median_low(r["wall_s"] for r in rows)
        middle = next(r for r in rows if r["wall_s"] == middle_wall)
        result.append({
            "combination": combination,
            "reviews": len(rows),
            "median_wall_s": round(statistics.median(r["wall_s"] for r in rows), 1),
            "middle_review": {
                "wall_s": round(middle["wall_s"], 1),
                "load_s": round(middle["load_s"], 1),
                "stage_s": {name: round(seconds, 1) for name, seconds in middle["stage_s"].items()},
                "other_s": round(middle["other_s"], 1),
            },
        })
    return sorted(result, key=lambda item: (item["median_wall_s"], item["combination"]))


def under_target(conn: sqlite3.Connection, *, minutes: float, this_machine: bool = False) -> list:
    """The combinations with at least one recorded review under `minutes`, each with how many of its reviews
    came in under it and how many it has in all (the evidence the claim rests on). Best share first."""
    if not (isinstance(minutes, (int, float)) and math.isfinite(minutes) and minutes > 0):
        raise ReportError(f"target_minutes must be a number above 0 (got {minutes!r})")
    limit = minutes * 60.0
    grouped: dict = {}
    for row in _review_rows(conn, this_machine):
        grouped.setdefault(row["combination"], []).append(row["wall_s"])
    result = []
    for combination, walls in grouped.items():
        under = sum(1 for w in walls if w <= limit)
        if under:
            result.append({
                "combination": combination,
                "under": under,
                "reviews": len(walls),
                "median_wall_s": round(statistics.median(walls), 1),
                "slowest_s": round(max(walls), 1),
            })
    return sorted(result, key=lambda item: (-item["under"] / item["reviews"], item["median_wall_s"], item["combination"]))


# ---------------------------------------------------------------------------
# Showing the answers
# ---------------------------------------------------------------------------

VIEWS = ("response_times", "review_times", "under_target")


def _seconds(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}s"


def _ms(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:g} ms"


def _timing_line(r: dict) -> str:
    """The millisecond and token figures of one `response_times` row, or "" when no request reported any."""
    latency, ttft = r["latency_ms"], r["ttft_ms"]
    parts = []
    if latency["median"] is not None:
        parts.append(f"latency median {_ms(latency['median'])} (p90 {_ms(latency['p90'])})")
    if ttft["median"] is not None:
        parts.append(f"first token median {_ms(ttft['median'])} (p90 {_ms(ttft['p90'])})")
    if r["inter_token_ms"]["median"] is not None:
        rate = r["tokens_per_second"]["median"]
        parts.append(f"between tokens median {_ms(r['inter_token_ms']['median'])}" + (f" ({rate:g} tokens/s)" if rate is not None else ""))
    tokens_in, tokens_out = r["prompt_tokens"]["median"], r["reply_tokens"]["median"]
    if tokens_in is not None or tokens_out is not None:
        parts.append(f"tokens in/out median {tokens_in if tokens_in is not None else '-'}/{tokens_out if tokens_out is not None else '-'}")
    return "; ".join(parts)


def format_response_times(rows: list, *, scope: str = "") -> str:
    if not rows:
        return f"No answered requests recorded{scope}."
    lines = [f"Response time per request, by model and role{scope}:"]
    for r in rows:
        extra = []
        if r["timed_out"]:
            extra.append(f"{r['timed_out']} timed out")
        if r["failed"]:
            extra.append(f"{r['failed']} failed")
        lines.append(
            f"- {r['model']} [{r['role']}]: {r['answered']} answered, median {_seconds(r['median_s'])}, "
            f"p90 {_seconds(r['p90_s'])}, max {_seconds(r['max_s'])}" + (f" ({', '.join(extra)})" if extra else "")
        )
        timing = _timing_line(r)
        if timing:
            lines.append(f"    {timing}")
    return "\n".join(lines)


def format_review_times(rows: list, *, scope: str = "") -> str:
    if not rows:
        return f"No finished reviews recorded{scope}."
    lines = [f"Review time per model combination{scope}, fastest first (median of its finished reviews, and how its "
             "middle review's time splits):"]
    for r in rows:
        split = r["middle_review"]
        parts = [f"load {_seconds(split['load_s'])}"] + [f"{n} {_seconds(s)}" for n, s in split["stage_s"].items()]
        parts.append(f"other {_seconds(split['other_s'])}")
        lines.append(f"- {r['combination']}: {_seconds(r['median_wall_s'])} over {r['reviews']} review(s); "
                     f"middle review {_seconds(split['wall_s'])} = {', '.join(parts)}")
    return "\n".join(lines)


def format_under_target(rows: list, minutes: float, *, scope: str = "") -> str:
    if not rows:
        return f"No recorded review{scope} finished within {minutes:g} minute(s)."
    lines = [f"Combinations with a review within {minutes:g} minute(s){scope}:"]
    for r in rows:
        lines.append(
            f"- {r['combination']}: {r['under']} of {r['reviews']} review(s) under the target "
            f"(median {_seconds(r['median_wall_s'])}, slowest {_seconds(r['slowest_s'])})"
        )
    return "\n".join(lines)


def report(
    view: str,
    *,
    model: Optional[str] = None,
    target_minutes: Optional[float] = None,
    this_machine: bool = False,
    as_json: bool = False,
    db_path: Optional[Path] = None,
) -> str:
    """One view as text (or JSON). `view` is one of `VIEWS`. Raises `ReportError` for a bad view or an
    unreadable database."""
    if view not in VIEWS:
        raise ReportError(f"unknown view {view!r}; known views: {', '.join(VIEWS)}")
    scope = " on this machine" if this_machine else ""
    conn = open_database(db_path)
    try:
        try:
            if view == "response_times":
                rows = response_times(conn, model=model or None, this_machine=this_machine)
                return json.dumps(rows, indent=2) if as_json else format_response_times(rows, scope=scope)
            if view == "review_times":
                rows = review_times(conn, this_machine=this_machine)
                return json.dumps(rows, indent=2) if as_json else format_review_times(rows, scope=scope)
            if target_minutes is None:
                raise ReportError("the under_target view needs target_minutes (a number above 0)")
            rows = under_target(conn, minutes=target_minutes, this_machine=this_machine)
            return json.dumps(rows, indent=2) if as_json else format_under_target(rows, target_minutes, scope=scope)
        except sqlite3.Error as exc:
            raise ReportError(f"cannot read the codebase-reviewer database ({exc})") from exc
    finally:
        conn.close()
