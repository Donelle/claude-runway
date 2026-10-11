"""
Read-only baseline report over memory-bank's EXISTING usage log (issue #338,
Batch A of the #328 staleness audit) -- turns `libs/memory_events_lib.py`'s
`memory-events.db` into the three numbers the #328 decision gate needs,
without any schema change and without touching memory-bank's behavior:

- **Dead-memory ratio**: of all distinct remembered `point_id`s, the fraction
  that have ZERO `recall` events referencing them at or after the time they
  were remembered -- overall and per `repo` tag (the memory's own tag, which
  can be the reserved `"general"` sentinel).
- **Recall payload trend**: per week/month bucket, how many hits a recall
  call returns on average -- overall and per `project` (the CALLING
  project's `MEMORY_BANK_ID`; a single call can return hits from several
  repo tags, so per-call numbers are grouped by caller, not by hit repo).
- **Score trend**: reported as UNAVAILABLE, with the reason -- see
  `SCORE_TREND_UNAVAILABLE` below.

What the existing data can and can't say -- stated here, and repeated in the
report's own output, so nobody reads more into the numbers than they hold
(the counting-semantics trap a past EVALUATION.md fix fell into, PR #323):

- `memory_events` logs one row per returned recall HIT, not per call; every
  hit from one `recall()` call shares that call's `(session_id, turn)`, so a
  call is reconstructed here as a `GROUP BY session_id, turn`. A recall that
  returned NOTHING (or failed) writes no row at all, so "average hits per
  call" here is per NON-EMPTY call. `metrics.db`'s `memory-bank` tally
  (`/my-metrics memory-bank trend`) does count every call attempt, best-effort,
  but carries no project/repo detail, so it can't be broken down per repo and
  isn't mixed in here.
- No column records a hit's `score`/`effective_score`, or the size of the
  response a call returned. Payload is therefore reported as a HIT COUNT,
  not bytes/tokens, and the score trend can't be computed at all without a
  schema change, which this ticket explicitly rules out.
- A "dead" memory may also have been `forget`-ed (not logged in this table)
  or be too new to have had a chance to be recalled -- `min_age_days` exists
  for the second case; nothing in the existing data distinguishes the first.
- The same memory's id is logged in two spellings (undashed hex on its
  remember row, dashed UUID on its recall rows) -- see dead_memory_ratio's
  comment. Ids are matched in a normalized form for that reason.

Never creates or writes the db: it's opened via SQLite's read-only URI mode,
unlike `memory_events_lib._connect()`, which creates the file and schema on
first use. A report run on a machine that has never logged an event says so
instead of leaving an empty `memory-events.db` behind as a side effect.
"""

import math
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

# Project ids excluded by default. "test-collection" is the MEMORY_BANK_ID
# tests/test_memory_bank_mcp_server.py sets; until issue #338 that module did
# not redirect CLAUDE_RUNWAY_MEMORY_EVENTS_DB, so every local test run appended
# fake "proj-a"/"new-id" rows into the developer's REAL memory-events.db
# (confirmed live: ~70% of one dogfood machine's rows). The test fix stops new
# pollution; this default keeps the rows already written from skewing the
# baseline. Overridable (see tools/memory_bank_report.py's
# --no-default-excludes) in case a real project ever uses this id.
DEFAULT_EXCLUDED_PROJECTS = ("test-collection",)

SCORE_TREND_UNAVAILABLE = (
    "Not available from existing data: memory_events records no score/effective_score "
    "for recall hits, so a score trend needs a schema change (out of scope for #338's "
    "baseline, which is existing-data only)."
)

_VALID_BUCKETS = ("week", "month")


def open_read_only(db_path: Path) -> Optional[sqlite3.Connection]:
    """
    Opens `db_path` read-only, or returns None when the file doesn't exist.
    The existence check comes first because SQLite's `mode=ro` still raises
    (rather than returning None) on a missing file, and a missing file is a
    normal state here ("nothing logged yet"), not an error.
    """
    if not db_path.is_file():
        return None
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)


def _exclusion_clause(excluded_projects: Iterable[str], column: str = "project") -> tuple:
    """
    SQL fragment + params that keep rows whose project is NOT excluded. A
    NULL project is kept explicitly: `project NOT IN (...)` alone evaluates
    to NULL (falsy) for a NULL project, which would silently drop rows the
    caller never asked to exclude.
    """
    excluded = tuple(excluded_projects)
    if not excluded:
        return "1 = 1", ()
    placeholders = ", ".join("?" for _ in excluded)
    return f"({column} IS NULL OR {column} NOT IN ({placeholders}))", excluded


def _bucket_key(event_timestamp: str, bucket: str) -> str:
    """
    event_timestamp is ISO 8601 UTC ("2026-09-24T12:00:00Z", written by
    memory_events_lib). Week keys use Python's isocalendar(), same as
    metrics_lib.MetricsStore.trend() and for the same reason (SQLite's
    strftime can't produce correct ISO week keys across year boundaries).
    """
    if bucket == "month":
        return event_timestamp[:7]
    dt = datetime.fromisoformat(event_timestamp.replace("Z", "+00:00"))
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _ratio(dead: int, total: int) -> Optional[float]:
    return (dead / total) if total else None


def dead_memory_ratio(
    conn: sqlite3.Connection,
    excluded_projects: Iterable[str] = DEFAULT_EXCLUDED_PROJECTS,
    min_age_days: float = 0.0,
    now: Optional[float] = None,
) -> dict:
    """
    {"overall": {...}, "by_repo": {repo: {...}}}, each {...} being
    {"remembered", "dead", "ratio"}. "remembered" counts distinct point_ids
    with a remember row; "dead" those with no recall row for the same
    point_id at or after their (earliest) remember timestamp. A recall row
    from BEFORE the remember row can only come from a re-used point_id and
    doesn't count as reuse of this memory.

    min_age_days > 0 only counts memories remembered at least that long ago
    (relative to `now`, default the current time), since a memory written
    an hour ago being "dead" says nothing yet. Memories recalled but never
    remembered while logging was on (created before issue #179) aren't in
    either count -- there's no remember row to anchor them to.
    """
    # Materialized once (PR #424 review): both clauses below consume it, and
    # a generator would be empty by the second, letting excluded projects'
    # recall rows count as reuse.
    excluded = tuple(excluded_projects)
    where_rem, rem_params = _exclusion_clause(excluded, "project")
    where_rec, rec_params = _exclusion_clause(excluded, "r.project")
    # Non-finite input is rejected rather than interpreted (PR #424 review):
    # nan fails every comparison, so it silently disabled the age filter,
    # and inf reached time.gmtime() and raised OverflowError.
    if not math.isfinite(min_age_days) or min_age_days < 0:
        raise ValueError(f"min_age_days must be a finite number >= 0 (got {min_age_days!r}).")
    cutoff = None
    if min_age_days > 0:
        current = time.time() if now is None else now
        cutoff_secs = current - min_age_days * 86400
        # A cutoff before the epoch means no memory is old enough. ""
        # sorts before every ISO timestamp, so `MIN(event_timestamp) <= ""`
        # matches nothing. This also keeps a huge but finite value away
        # from gmtime(), which overflows the same way inf did.
        cutoff = "" if cutoff_secs < 0 else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cutoff_secs))
    age_clause = "HAVING MIN(event_timestamp) <= ?" if cutoff is not None else ""
    age_params: tuple = (cutoff,) if cutoff is not None else ()
    # Ids are compared in a normalized form (lowercase, dashes stripped), not
    # as stored. remember_point mints `uuid.uuid4().hex` (32 hex chars, no
    # dashes) and that is what the remember row logs, but Qdrant returns
    # every point id in canonical dashed UUID form, so the SAME memory's
    # recall rows carry "6fed0f3a-a24c-..." -- confirmed live, a plain
    # `r.point_id = rem.point_id` matched nothing and reported every memory
    # as dead. Normalizing here covers rows already logged both ways.
    norm = "REPLACE(LOWER({col}), '-', '')"
    rows = conn.execute(
        f"""
        WITH rem AS (
            SELECT {norm.format(col='point_id')} AS pid, MIN(repo) AS repo, MIN(event_timestamp) AS ts
            FROM memory_events
            WHERE event_type = 'remember' AND point_id IS NOT NULL AND {where_rem}
            GROUP BY pid
            {age_clause}
        ),
        rec AS (
            SELECT {norm.format(col='r.point_id')} AS pid, MAX(r.event_timestamp) AS last_ts
            FROM memory_events r
            WHERE r.event_type = 'recall' AND r.point_id IS NOT NULL AND {where_rec}
            GROUP BY pid
        )
        SELECT rem.repo, rec.last_ts IS NOT NULL AND rec.last_ts >= rem.ts
        FROM rem LEFT JOIN rec ON rec.pid = rem.pid
        """,
        rem_params + age_params + rec_params,
    ).fetchall()

    by_repo: dict = {}
    for repo, recalled in rows:
        key = repo if repo is not None else "(none)"
        entry = by_repo.setdefault(key, {"remembered": 0, "dead": 0})
        entry["remembered"] += 1
        if not recalled:
            entry["dead"] += 1
    for entry in by_repo.values():
        entry["ratio"] = _ratio(entry["dead"], entry["remembered"])
    total = sum(e["remembered"] for e in by_repo.values())
    dead = sum(e["dead"] for e in by_repo.values())
    return {
        "overall": {"remembered": total, "dead": dead, "ratio": _ratio(dead, total)},
        "by_repo": dict(sorted(by_repo.items())),
    }


def recall_payload_trend(
    conn: sqlite3.Connection,
    bucket: str = "week",
    excluded_projects: Iterable[str] = DEFAULT_EXCLUDED_PROJECTS,
) -> dict:
    """
    {"overall": [row, ...], "by_project": {project: [row, ...]}}, rows
    ascending by bucket, each {"bucket", "calls", "hits", "avg_hits_per_call"}.

    A "call" is one distinct (session_id, turn) among recall rows -- see this
    module's docstring for why that's per NON-EMPTY call only. A call's
    bucket is its earliest row's timestamp (every row of one call is written
    within the same request, so in practice they're identical).
    """
    if bucket not in _VALID_BUCKETS:
        raise ValueError(f"Unknown bucket {bucket!r}. Valid values: {_VALID_BUCKETS}.")
    where, params = _exclusion_clause(excluded_projects)
    rows = conn.execute(
        f"""
        SELECT MIN(project), MIN(event_timestamp), COUNT(*)
        FROM memory_events
        WHERE event_type = 'recall' AND {where}
        GROUP BY session_id, turn
        """,
        params,
    ).fetchall()

    overall: dict = {}
    by_project: dict = {}
    for project, ts, hits in rows:
        key = _bucket_key(ts, bucket)
        project_key = project if project is not None else "(none)"
        for agg in (overall, by_project.setdefault(project_key, {})):
            entry = agg.setdefault(key, {"calls": 0, "hits": 0})
            entry["calls"] += 1
            entry["hits"] += hits

    def _rows(agg: dict) -> list:
        return [
            {"bucket": k, "calls": v["calls"], "hits": v["hits"],
             "avg_hits_per_call": v["hits"] / v["calls"]}
            for k, v in sorted(agg.items())
        ]

    return {
        "overall": _rows(overall),
        "by_project": {p: _rows(agg) for p, agg in sorted(by_project.items())},
    }


def _data_window(conn: sqlite3.Connection, excluded_projects: Iterable[str]) -> dict:
    where, params = _exclusion_clause(excluded_projects)
    included, first, last = conn.execute(
        f"SELECT COUNT(*), MIN(event_timestamp), MAX(event_timestamp) FROM memory_events WHERE {where}",
        params,
    ).fetchone()
    (total,) = conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()
    return {"rows_included": included, "rows_excluded": total - included, "first_event": first, "last_event": last}


def build_report(
    db_path: Path,
    bucket: str = "week",
    excluded_projects: Iterable[str] = DEFAULT_EXCLUDED_PROJECTS,
    min_age_days: float = 0.0,
    now: Optional[float] = None,
) -> dict:
    """
    The full report as a plain dict (JSON-serializable). `available` is False
    when the db file doesn't exist or has no memory_events table yet -- both
    mean "nothing logged yet," reported rather than raised.
    """
    if bucket not in _VALID_BUCKETS:
        raise ValueError(f"Unknown bucket {bucket!r}. Valid values: {_VALID_BUCKETS}.")
    excluded = tuple(excluded_projects)
    base = {
        "db_path": str(db_path),
        "bucket": bucket,
        "excluded_projects": list(excluded),
        "min_age_days": min_age_days,
        "score_trend": {"available": False, "reason": SCORE_TREND_UNAVAILABLE},
    }
    conn = open_read_only(db_path)
    if conn is None:
        return {**base, "available": False, "reason": "no memory-events.db found (nothing logged yet)"}
    try:
        # One read transaction for the whole report (PR #424 review), so
        # every query sees the same snapshot. Without it each SELECT runs in
        # its own implicit transaction, and an event the memory-bank server
        # logs mid-report can show up in some sections but not others
        # (reproduced: two counts on one autocommit connection differed
        # across an interleaved insert). The snapshot holds a shared lock;
        # a concurrent writer waits on sqlite3's default 5s busy timeout,
        # far longer than this report's millisecond-scale queries take.
        # Closing the connection in `finally` ends the transaction.
        conn.execute("BEGIN")
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_events'"
        ).fetchone()
        if not has_table:
            return {**base, "available": False, "reason": "memory-events.db has no memory_events table yet"}
        return {
            **base,
            "available": True,
            "data_window": _data_window(conn, excluded),
            "dead_memory": dead_memory_ratio(conn, excluded, min_age_days=min_age_days, now=now),
            "recall_payload": recall_payload_trend(conn, bucket, excluded),
        }
    finally:
        conn.close()


def _pct(ratio: Optional[float]) -> str:
    return "n/a" if ratio is None else f"{ratio * 100:.1f}%"


def _payload_lines(rows: list, indent: str) -> list:
    if not rows:
        return [f"{indent}(no recall calls with hits)"]
    return [
        f"{indent}{r['bucket']}: {r['calls']} call(s), {r['hits']} hit(s), "
        f"avg {r['avg_hits_per_call']:.2f} hits/call"
        for r in rows
    ]


def format_report(report: dict) -> str:
    lines = ["memory-bank baseline report (issue #338)", f"Source: {report['db_path']}"]
    excluded = report["excluded_projects"]
    lines.append(f"Excluded projects: {', '.join(excluded) if excluded else '(none)'}")
    if not report["available"]:
        lines.append(f"No data: {report['reason']}")
        lines.append(f"Score trend: {report['score_trend']['reason']}")
        return "\n".join(lines)

    window = report["data_window"]
    lines.append(
        f"Data window: {window['first_event'] or '-'} .. {window['last_event'] or '-'} "
        f"({window['rows_included']} row(s) included, {window['rows_excluded']} excluded)"
    )

    dead = report["dead_memory"]
    age = report["min_age_days"]
    lines.append("")
    lines.append(
        "Dead-memory ratio (remembered memories never recalled since"
        + (f"; only memories at least {age:g} day(s) old" if age else "")
        + "):"
    )
    o = dead["overall"]
    lines.append(f"  overall: {o['dead']}/{o['remembered']} dead ({_pct(o['ratio'])})")
    for repo, e in dead["by_repo"].items():
        lines.append(f"  {repo}: {e['dead']}/{e['remembered']} dead ({_pct(e['ratio'])})")

    payload = report["recall_payload"]
    lines.append("")
    lines.append(
        f"Recall payload by {report['bucket']} (hits per NON-EMPTY call; empty calls and "
        "response size aren't logged):"
    )
    lines.append("  overall:")
    lines.extend(_payload_lines(payload["overall"], "    "))
    for project, rows in payload["by_project"].items():
        lines.append(f"  {project} (calling project):")
        lines.extend(_payload_lines(rows, "    "))

    lines.append("")
    lines.append(f"Score trend: {report['score_trend']['reason']}")
    return "\n".join(lines)
