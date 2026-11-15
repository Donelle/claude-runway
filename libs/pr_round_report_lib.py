"""
Read-only report over the `pr-review-rounds` events in the shared metrics store
(`metrics.db`, written by `libs/pr_round_record_lib.py`): what Claude spent on each Copilot
review round, per repository, with the rounds split into those answered with the local
codebase-reviewer ("reviewer used") and those answered without it ("not used"). It is the
quick view behind `EVALUATION.md`'s Track G; the Track's queries add the cut-off and the
like-for-like rules this report does not know about (it cannot tell a merged pull request
from a closed one, which only GitHub can).

WHAT IS COUNTED
  A ROUND, not a pull request. Only `pr_round` rows of kind `copilot` are the rounds being
  compared; the `local` rows are the reviewer's own cost and are summed apart, per
  repository, but only the ones whose `review_pr` is the row's own PR (the call named it).
  A `review` call that names no PR is attached by the reader to every PR the session fetched
  feedback for, so one call is a row under each of them, with a window cut by that PR's own
  rounds: adding them would count one call once per PR. Those rows are reported as an
  ambiguous group of their own (`local_rounds_unnamed`), outside the cost totals. `pr_feedback_fetch` rows are a finer grain of the same spend and are never read
  (adding them to the rounds would count it twice).
  A review that two sessions both fetched is two rows (each with its own session's spend).
  They are one round here, keyed by repository, pull request and review (the numeric
  GitHub id when the row has it, else the id the transcript showed), with each token field
  summed over its rows, never across fields. A round is "reviewer used" when any of its rows
  says so.

THE GROUPS
  - `used`: `local_reviewer_used` is true.
  - `not_used`: it is false. This means "no local review captured for this round", not proof
    the tool was not used: a review made in a session whose transcript was not read shows as
    false and is never corrected (see docs/metrics.md).
  - `unknown`: the row has no such field (written before it existed, `schema` 1). Never
    folded into `not_used`.
  Rows are grouped by `repo` and `repo_source`: a repository (`remote`), a project folder name
  (`project`) and a row with no repository at all (null) are three different kinds of key,
  and a null is never merged into a repository.

FIGURES
  The four token fields (`input_tokens`, `cache_creation_tokens`, `cache_read_tokens`,
  `output_tokens`) are kept apart and never summed into one number: cache reads dominate by
  orders of magnitude. Per group and field: how many rounds had a figure, their total,
  median, smallest and largest. A round whose last window has no end (`window_open`) is a
  floor; they are counted per group and can be left out (`exclude_open`).

Never creates or writes the database: it is opened through SQLite's read-only URI mode, so a
machine that has recorded nothing is told so instead of getting an empty file. Standard
library only.
"""

import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Optional

METRIC_ID = "pr-review-rounds"
EVENT_ROUND = "pr_round"
KIND_COPILOT = "copilot"
KIND_LOCAL = "local"

TOKEN_FIELDS = ("input_tokens", "cache_creation_tokens", "cache_read_tokens", "output_tokens")
GROUPS = ("used", "not_used", "unknown")
_GROUP_LABELS = {"used": "reviewer used", "not_used": "reviewer not used", "unknown": "unknown (no flag on the row)"}
_SOURCE_ORDER = {"remote": 0, "project": 1, None: 2}


def open_read_only(db_path: Path) -> Optional[sqlite3.Connection]:
    """Opens `db_path` read-only, or None when the file does not exist (a normal state:
    nothing recorded yet, not an error)."""
    if not db_path.is_file():
        return None
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)


def _number(value) -> Optional[float]:
    # A non-finite figure (a hand-edited row; json.loads accepts NaN) would poison every total
    # and make --json emit a bare NaN, which is not JSON: it is left out like a missing one.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _repo_key(meta: dict) -> tuple:
    """`(repo, repo_source)`: a value that is not a non-empty string is null, so a malformed
    row is grouped with the null ones instead of getting a key of its own."""
    repo = meta.get("repo")
    source = meta.get("repo_source")
    if not (isinstance(repo, str) and repo):
        return (None, None)
    return (repo, source if source in ("remote", "project") else None)


def _round_key(meta: dict, session_id, repo_key: tuple) -> tuple:
    """What makes two rows one round: the repository, the pull request and the review. A Copilot
    row with no review id at all cannot be matched to another session's row, so it is its own
    round (its session and place among the session's rounds stand in for the id)."""
    review = meta.get("github_review_id") or meta.get("review_id")
    if review:
        return (repo_key, meta.get("pr"), str(review))
    return (repo_key, meta.get("pr"), None, session_id, meta.get("round"))


def _fold(rounds: dict, key: tuple, meta: dict, flagged: bool) -> None:
    entry = rounds.setdefault(key, {"pr": meta.get("pr"), "open": False, "used": None,
                                    **{f: None for f in TOKEN_FIELDS}})
    if meta.get("window_open") is True:
        entry["open"] = True
    if flagged:
        entry["used"] = bool(entry["used"]) or meta.get("local_reviewer_used") is True
    for field in TOKEN_FIELDS:
        value = _number(meta.get(field))
        if value is not None:
            entry[field] = (entry[field] or 0) + value


def _stats(values: list) -> dict:
    if not values:
        return {"n": 0, "total": None, "median": None, "min": None, "max": None}
    return {"n": len(values), "total": sum(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def _summarise(entries: list) -> dict:
    return {"rounds": len(entries),
            "open_windows": sum(1 for e in entries if e["open"]),
            **{f: _stats([e[f] for e in entries if e[f] is not None]) for f in TOKEN_FIELDS}}


def label_of(repo, source) -> str:
    """How a group is named in the text report."""
    if repo is None:
        return "(no repository or project recorded)"
    return f"{repo} ({'repository' if source == 'remote' else 'project folder, not a repository' if source == 'project' else 'source not recorded'})"


def aggregate(rows: list, repo: Optional[str] = None, exclude_open: bool = False) -> dict:
    """The report body from `(metadata JSON text, session_id)` rows of the `pr_round` events.
    `repo` keeps only the rows whose `repo` is that exact value; `exclude_open` leaves out the
    rounds whose last window had no end. Rows that are not JSON objects are counted, not read."""
    copilot: dict = {}
    local: dict = {}
    local_unnamed: dict = {}
    unreadable = 0
    for text, session_id in rows:
        try:
            meta = json.loads(text) if isinstance(text, str) else None
        except ValueError:
            meta = None
        if not isinstance(meta, dict):
            unreadable += 1
            continue
        key = _repo_key(meta)
        if repo is not None and key[0] != repo:
            continue
        kind = meta.get("kind")
        named = _number(meta.get("review_pr")) is not None and meta.get("review_pr") == meta.get("pr")
        if kind == KIND_COPILOT:
            _fold(copilot.setdefault(key, {}), _round_key(meta, session_id, key), meta, "local_reviewer_used" in meta)
        elif kind == KIND_LOCAL:
            # A local round is told apart by its place among its session's local rounds.
            bucket = local if named else local_unnamed
            _fold(bucket.setdefault(key, {}), (key, meta.get("pr"), session_id, meta.get("kind_number")), meta, False)
    repos = []
    for key in sorted(set(copilot) | set(local) | set(local_unnamed), key=lambda k: (_SOURCE_ORDER[k[1]], str(k[0]))):
        rounds = [e for e in copilot.get(key, {}).values() if not (exclude_open and e["open"])]
        local_rounds = [e for e in local.get(key, {}).values() if not (exclude_open and e["open"])]
        unnamed_rounds = [e for e in local_unnamed.get(key, {}).values() if not (exclude_open and e["open"])]
        grouped: dict = {g: [] for g in GROUPS}
        for e in rounds:
            grouped["unknown" if e["used"] is None else "used" if e["used"] else "not_used"].append(e)
        repos.append({
            "repo": key[0], "repo_source": key[1], "label": label_of(*key),
            "copilot_rounds": len(rounds), "pull_requests": len({e["pr"] for e in rounds}),
            "groups": {g: _summarise(grouped[g]) for g in GROUPS},
            "local_rounds": _summarise(local_rounds),
            "local_rounds_unnamed": _summarise(unnamed_rounds),
        })
    return {"repos": repos, "unreadable_rows": unreadable}


def build_report(db_path: Path, repo: Optional[str] = None, exclude_open: bool = False) -> dict:
    """The full report as a plain, JSON-serialisable dict. `available` is False when the file
    does not exist or has no `metrics` table, both reported rather than raised. One read
    transaction, so every figure comes from the same snapshot."""
    base = {"db_path": str(db_path), "repo_filter": repo, "exclude_open": exclude_open}
    try:
        conn = open_read_only(db_path)
    except sqlite3.Error as e:
        return {**base, "available": False, "reason": f"metrics.db could not be opened ({type(e).__name__})"}
    if conn is None:
        return {**base, "available": False, "reason": "no metrics.db found (nothing recorded yet)"}
    try:
        conn.execute("BEGIN")
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'metrics'").fetchone():
            return {**base, "available": False, "reason": "metrics.db has no metrics table yet"}
        rows = conn.execute("SELECT metadata, session_id FROM metrics WHERE metric_id = ? AND event_type = ? ORDER BY id",
                            (METRIC_ID, EVENT_ROUND)).fetchall()
    except sqlite3.Error as e:
        # A file that is not a database, or one a writer holds locked for longer than the busy
        # timeout (a session-end capture running as the report does): said, not a traceback.
        return {**base, "available": False, "reason": f"metrics.db could not be read ({type(e).__name__}: {str(e)[:100]})"}
    finally:
        conn.close()
    if not rows:
        return {**base, "available": False, "reason": "no pr-review-rounds rounds recorded yet"}
    return {**base, "available": True, **aggregate(rows, repo, exclude_open)}


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def format_report(report: dict) -> str:
    lines = ["pr-review-rounds report: Claude's tokens per Copilot review round, by repository and by whether the "
             "local reviewer was used", f"Source: {report['db_path']}"]
    if report["repo_filter"]:
        lines.append(f"Only rows whose repo is {report['repo_filter']}")
    if not report["available"]:
        lines.append(f"No data: {report['reason']}")
        return "\n".join(lines)
    lines.append("A row is one Copilot round (rounds seen by several sessions are one, each token field summed); the four "
                 "token fields are never added together. 'not used' means no local review was captured for the round.")
    if report["exclude_open"]:
        lines.append("Rounds whose last window has no end are left out.")
    if not report["repos"]:
        lines.append("No rounds match.")
    for r in report["repos"]:
        lines.append("")
        lines.append(f"{r['label']}: {r['copilot_rounds']} Copilot round(s) over {r['pull_requests']} pull request(s)")
        for g in GROUPS:
            s = r["groups"][g]
            if not s["rounds"]:
                continue
            note = f", {s['open_windows']} with an open last window (a floor)" if s["open_windows"] else ""
            lines.append(f"  {_GROUP_LABELS[g]}: {s['rounds']} round(s){note}")
            for f in TOKEN_FIELDS:
                t = s[f]
                lines.append(f"    {f}: median {_fmt(t['median'])} (min {_fmt(t['min'])}, max {_fmt(t['max'])}), "
                             f"total {_fmt(t['total'])} over {t['n']} round(s)")
        loc = r["local_rounds"]
        if loc["rounds"]:
            lines.append(f"  local review rounds that name their pull request (the reviewer's own cost): {loc['rounds']}")
            for f in TOKEN_FIELDS:
                lines.append(f"    {f}: total {_fmt(loc[f]['total'])} over {loc[f]['n']} round(s)")
        unnamed = r["local_rounds_unnamed"]
        if unnamed["rounds"]:
            lines.append(f"  local review rows that name no pull request: {unnamed['rounds']}, not added to the cost above "
                         "(one call is a row under every pull request the session fetched feedback for, so its tokens "
                         "would count once per pull request)")
            for f in TOKEN_FIELDS:
                lines.append(f"    {f}: total {_fmt(unnamed[f]['total'])} over {unnamed[f]['n']} row(s), counted once per pull request")
    if report["unreadable_rows"]:
        lines.append("")
        lines.append(f"{report['unreadable_rows']} row(s) had no readable metadata and were not counted.")
    return "\n".join(lines)
