#!/usr/bin/env python3
"""
Shared, dependency-free derivation of /my-gh-autowork's per-ticket cost from the
SUBAGENT TRANSCRIPTS, instead of the orchestrator copying figures out
of the Agent call's `usage` block by hand (which nothing enforced: 22 of 91 logged
runs had no figures at all, and the one total it did record had no cache split).

What a subagent transcript holds (`<session>/subagents/agent-<id>.jsonl`, the same
layout libs/transcript_attribution_lib.py reads) -- all recorded by Claude Code:
  - each assistant message: `message.id`, `message.model`, `message.usage`
    (`input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`,
    `output_tokens`), `timestamp`, and `tool_use` content blocks;
  - the sibling `agent-<id>.meta.json`: `toolUseId` (the orchestrator's `Agent`
    call), `description`, and sometimes `model`.

HOW A TRANSCRIPT IS LINKED TO A TICKET (one rule, chosen on purpose):
  - An ATTEMPT is one subagent file. A re-attempt of an issue and a follow-up
    subagent (`issue-N-followup`) are each their own `Agent` call, so each is its
    own attempt with its own agent id; they are never merged into one total.
  - The caller identifies the attempt by `agent_id` or by `tool_use_id` (the meta
    file's `toolUseId`). That is exact and is what `record` uses.
  - The ISSUE NUMBER, needed only to match historical rows (backfill/compare),
    is read, only when that prompt names Donelle/claude-runway (issue
    numbers are per repository and discovery scans every Claude project), from the
    number the template substitutes on the line after its "## Step 0" heading
    (the stable form; an auto-pick prompt has no number there), else a stray "The
    issue to work is #N" sentence, else a `#N`/`issue N` in the meta `description`.
    None of them present means the attempt has no issue and never matches a row.
    `record` refuses to file an attempt whose own issue differs from `--issue`.

Figures here are COUNTS OF TOKENS THE API REPORTED, not estimates. Two caveats:
  - Claude Code writes every content block of one API response as its own line, each
    repeating that response's `usage`; usage is therefore taken once per
    `message.id` (field-wise max, since a streamed message can be written in
    progress first), never summed per line.
  - `output_tokens` already INCLUDES thinking tokens. The API reports no separate
    thinking count, so none is invented here; `thinking_chars` (the characters of
    thinking text actually stored in the transcript, often empty) is only a hint.

Not included: the orchestrator's own cost (its transcript is the main session file,
not a subagent file).

The transcript format is internal to Claude Code and undocumented; it may change
with any release. Every reader here fails open (None / empty) rather than raising.
"""

import glob
import json
import os
import re
import statistics
from datetime import datetime
from typing import Optional

import transcript_context_lib

COST_EVENT = "ticket_cost"

_TOKEN_KEYS = (
    ("input_tokens", "input_tokens"),
    ("cache_creation_input_tokens", "cache_creation_tokens"),
    ("cache_read_input_tokens", "cache_read_tokens"),
    ("output_tokens", "output_tokens"),
)
_FIELDS = tuple(k for _, k in _TOKEN_KEYS)

# The repository every autowork subagent prompt names (the template's first line).
REPO_MARKER = "Donelle/claude-runway"

# The template's "## Step 0" heading is followed by the bare substituted number
# (SKILL.md's "{ISSUE_NUMBER}" line); that is the stable form, so it comes first.
_ISSUE_STEP0_RE = re.compile(r"## Step 0[^\n]*\n[ \t]*#?(\d+)[ \t]*(?:\n|\Z)")
_ISSUE_PROMPT_RE = re.compile(r"The issue to work is #(\d+)")
_ISSUE_DESC_RE = re.compile(r"(?:issue[\s#-]*|#)(\d+)\b", re.IGNORECASE)
_SESSION_ISSUE_RE = re.compile(r"\Aissue-(\d+)-")

# A row is logged just after its subagent finishes; allow for clock skew between
# the orchestrator's `date` and the transcript's UTC timestamps being read back.
MATCH_SLACK_S = 600
# How long before a row an attempt may have finished and still be its attempt. The
# orchestrator logs right after the final report; across the 2026-10-13 store the
# gap was at most 93s, so an hour tolerates a slow orchestrator without letting a
# stale transcript of the same issue stand in for a missing one.
MATCH_LOOKBACK_S = 3600

# COLD-CACHE REBUILD (documented in docs/metrics.md): a request that wrote a large
# cache block and read nothing, arriving more than REBUILD_GAP_S after the previous
# request started. The prompt cache appears to lapse after about five minutes idle,
# so a long single Bash wait makes the next request re-write the whole context at
# the (dearer) cache-write rate. The first request of a run is never a rebuild (it
# has no previous request and always writes). "Large" is REBUILD_MIN_CREATION
# tokens, well under a real context (a subagent starts near 56k) and well over the
# few hundred tokens an ordinary turn adds.
REBUILD_GAP_S = 300
REBUILD_MIN_CREATION = 10_000


def _count(v) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def _parse_ts(s) -> Optional[float]:
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _prompt_text(message) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content
                         if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str))
    return ""


def issue_of(prompt: str, description: str) -> Optional[int]:
    m = (_ISSUE_STEP0_RE.search(prompt or "") or _ISSUE_PROMPT_RE.search(prompt or "")
         or _ISSUE_DESC_RE.search(description or ""))
    return int(m.group(1)) if m else None


def cache_stats(requests: list) -> dict:
    """`requests` is [(start_ts or None, cache_creation_tokens, cache_read_tokens)] in
    transcript order, one per distinct API response. Returns
    {"longest_gap_s": int|None, "cold_rebuilds": int}: the longest start-to-start gap
    between consecutive requests that both have a timestamp, and the number of
    requests meeting the rebuild definition above."""
    longest = None
    rebuilds = 0
    prev = None
    for ts, creation, read in requests:
        if ts is not None and prev is not None:
            gap = ts - prev
            if longest is None or gap > longest:
                longest = gap
            if gap > REBUILD_GAP_S and read == 0 and creation >= REBUILD_MIN_CREATION:
                rebuilds += 1
        if ts is not None:
            prev = ts
    return {"longest_gap_s": round(longest) if longest is not None else None, "cold_rebuilds": rebuilds}


def parse_attempt(jsonl_path) -> Optional[dict]:
    """One subagent file -> an attempt dict, or None when it has no assistant turn.
    {"agent_id", "tool_use_id", "description", "issue", "kind", "first_ts",
     "last_ts", "wall_clock_s", "tool_calls", "thinking_chars", "by_model"}.
    `by_model` is {model: {requests, input_tokens, cache_creation_tokens,
    cache_read_tokens, output_tokens}}; `requests` counts distinct API responses."""
    jsonl_path = str(jsonl_path)
    base = os.path.basename(jsonl_path)
    agent_id = base[len("agent-"):-len(".jsonl")] if base.startswith("agent-") and base.endswith(".jsonl") else base
    description: str = ""
    tool_use_id: str = ""
    try:
        with open(jsonl_path[:-len(".jsonl")] + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta, dict):
            d, u = meta.get("description"), meta.get("toolUseId")
            description = d if isinstance(d, str) else ""
            tool_use_id = u if isinstance(u, str) else ""
    except (OSError, ValueError):
        pass

    usage_by_id: dict = {}   # message.id -> {"model", **fields}
    start_by_id: dict = {}   # message.id -> earliest timestamp (the request's start)
    anon = 0
    tool_ids: set = set()
    thinking_chars = 0
    prompt = ""
    first_ts = last_ts = None
    for obj in transcript_context_lib.iter_entries(jsonl_path):
        # Wall clock spans the WHOLE file, prompt to final entry, taken before any
        # entry is filtered by kind: measuring only assistant messages made a
        # one-response agent 0s and dropped the first request's latency.
        ts = _parse_ts(obj.get("timestamp"))
        if ts is not None:
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)
        message = obj.get("message")
        if not isinstance(message, dict):
            continue
        kind = obj.get("type")
        if kind == "user" and not prompt:
            prompt = _prompt_text(message)
            continue
        if kind != "assistant" or message.get("model") == "<synthetic>":
            continue
        usage = message.get("usage")
        # A usage slot exists only for a message that carries a non-empty usage
        # object. A readable transcript with none (the format is internal and can
        # change) must degrade to "no figures", not to an available zero-cost run.
        if isinstance(usage, dict) and usage:
            mid = message.get("id")
            if not isinstance(mid, str) or not mid:
                anon += 1
                mid = f"\0anon{anon}"
            model = message.get("model") if isinstance(message.get("model"), str) and message.get("model") else "unknown"
            slot = usage_by_id.setdefault(mid, {"model": model, **{k: 0 for k in _FIELDS}})
            for src, dst in _TOKEN_KEYS:
                slot[dst] = max(slot[dst], _count(usage.get(src)))
            if ts is not None:
                start_by_id[mid] = min(ts, start_by_id.get(mid, ts))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                    tool_ids.add(block["id"])
                elif block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
                    thinking_chars += len(block["thinking"])
    if not usage_by_id:
        return None
    by_model: dict = {}
    for slot in usage_by_id.values():
        row = by_model.setdefault(slot["model"], {"requests": 0, **{k: 0 for k in _FIELDS}})
        row["requests"] += 1
        for k in _FIELDS:
            row[k] += slot[k]
    wall = round(last_ts - first_ts) if first_ts is not None and last_ts is not None else None
    return {
        "agent_id": agent_id, "tool_use_id": tool_use_id, "description": description,
        # Issue numbers are not unique across repositories, and discovery scans every
        # Claude project, so an attempt only carries an issue (and so can only match
        # a row) when its own prompt names this repository.
        "issue": issue_of(prompt, description) if REPO_MARKER in prompt else None,
        "kind": "followup" if re.search(r"follow-?up", description, re.IGNORECASE) else "attempt",
        "first_ts": first_ts, "last_ts": last_ts, "wall_clock_s": wall,
        "tool_calls": len(tool_ids), "thinking_chars": thinking_chars, "by_model": by_model,
        **cache_stats([(start_by_id.get(mid), slot["cache_creation_tokens"], slot["cache_read_tokens"])
                       for mid, slot in usage_by_id.items()]),
    }


def default_projects_root() -> str:
    return os.path.join(os.path.expanduser("~"), ".claude", "projects")


def subagent_files(projects_root) -> list:
    """Every subagent transcript under `<root>/<project>/<session>/subagents/`."""
    return sorted(glob.glob(os.path.join(str(projects_root), "*", "*", "subagents", "agent-*.jsonl")))


def locate(projects_root, agent_id: Optional[str] = None, tool_use_id: Optional[str] = None) -> Optional[str]:
    """Path of the subagent file for an agent id (exact, from the file name) or a
    tool-use id (the meta file's `toolUseId`); None when neither is found."""
    files = subagent_files(projects_root)
    if agent_id:
        want = f"agent-{agent_id}.jsonl"
        for f in files:
            if os.path.basename(f) == want:
                return f
    if tool_use_id:
        for f in files:
            try:
                with open(f[:-len(".jsonl")] + ".meta.json", encoding="utf-8") as m:
                    if json.load(m).get("toolUseId") == tool_use_id:
                        return f
            except (OSError, ValueError, AttributeError):
                continue
    return None


def load_attempts(projects_root) -> list:
    out = []
    for f in subagent_files(projects_root):
        try:
            a = parse_attempt(f)
        except Exception:
            a = None
        if a:
            out.append(a)
    return out


# --------------------------------------------------------------------------
# Totals and the metadata a ticket row carries
# --------------------------------------------------------------------------

def totals(by_model: dict) -> dict:
    t: dict = {k: sum(r.get(k, 0) for r in by_model.values()) for k in _FIELDS}
    t["requests"] = sum(r.get("requests", 0) for r in by_model.values())
    t["total_tokens"] = sum(t[k] for k in _FIELDS)
    t["cache_read_share"] = (t["cache_read_tokens"] / t["total_tokens"]) if t["total_tokens"] else None
    return t


def cost_metadata(attempt: dict, source: str = "transcript") -> dict:
    """The `cost` object stored on a row: per-model split, never one lump."""
    t = totals(attempt["by_model"])
    return {
        "source": source, "available": True,
        "agent_id": attempt["agent_id"], "tool_use_id": attempt["tool_use_id"], "kind": attempt["kind"],
        "by_model": attempt["by_model"], "cache_read_share": t["cache_read_share"],
        "thinking_chars": attempt["thinking_chars"],
        # Stored at record time because the report reads rows, not transcripts. Rows
        # written before these fields existed simply lack them (shown as n/a).
        "longest_gap_s": attempt.get("longest_gap_s"), "cold_rebuilds": attempt.get("cold_rebuilds"),
    }


def figures(attempt: dict) -> dict:
    """The three flat figures the orchestrator used to type in by hand, now
    computed. `subagent_tokens` is input + cache creation + output, i.e. everything
    EXCEPT cache reads: `compare` showed the hand-copied figures on the
    existing rows match that definition (median 101% of it) and are about 3.6% of
    the all-counter total, so this keeps old and new rows on one scale. The cache
    reads, which are most of the volume, are in `cost.by_model` and the report."""
    t = totals(attempt["by_model"])
    out = {"subagent_tokens": t["total_tokens"] - t["cache_read_tokens"], "tool_calls": attempt["tool_calls"]}
    if attempt["wall_clock_s"] is not None:
        out["wall_clock_s"] = attempt["wall_clock_s"]
    return out


def primary_model(attempt: dict) -> str:
    return max(attempt["by_model"].items(), key=lambda kv: (kv[1]["output_tokens"], kv[1]["requests"]))[0]


def ticket_metadata(attempt: Optional[dict], base: dict, reason: str = "subagent transcript not found or has no usage data") -> dict:
    """`base` (issue/pr/outcome/rounds/findings/manual_interventions, supplied by
    the orchestrator) plus the computed figures. With no attempt, `base` is kept and
    marked unavailable (with `reason`), so a missing or refused transcript is never
    mistaken for a free run."""
    meta = dict(base)
    if attempt is None:
        meta["cost"] = {"source": "transcript", "available": False, "reason": reason}
        return meta
    meta.update(figures(attempt))
    meta.setdefault("model", primary_model(attempt))
    meta["cost"] = cost_metadata(attempt)
    return meta


# --------------------------------------------------------------------------
# Matching historical rows to attempts (backfill / compare)
# --------------------------------------------------------------------------

def row_issue(row: dict) -> Optional[int]:
    meta = row.get("metadata") or {}
    issue = meta.get("issue")
    if isinstance(issue, int) and not isinstance(issue, bool):
        return issue
    m = _SESSION_ISSUE_RE.match(row.get("session_id") or "")
    return int(m.group(1)) if m else None


def has_figures(row: dict) -> bool:
    meta = row.get("metadata") or {}
    return isinstance(meta.get("subagent_tokens"), (int, float)) and not isinstance(meta.get("subagent_tokens"), bool)


def match_rows(rows: list, attempts: list) -> dict:
    """{id(row): attempt or None}. Rows are claimed oldest first; each attempt serves
    ONE row, the unclaimed attempt for that issue that finished nearest before the
    row was logged. Run over ALL rows (with and without figures) so a row that has
    none can never take the attempt that an already-complete row maps to."""
    claimed: set = set()
    result: dict = {}
    for row in sorted(rows, key=lambda r: r.get("event_timestamp") or ""):
        issue = row_issue(row)
        row_ts = _parse_ts(row.get("event_timestamp"))
        best = None
        if issue is not None and row_ts is not None:
            for a in attempts:
                if a["issue"] != issue or a["agent_id"] in claimed or a["last_ts"] is None:
                    continue
                # Bounded in BOTH directions: an old unclaimed transcript for the same
                # issue must not be assigned to a row whose own transcript is gone
                # (that row is unavailable, not someone else's cost; Copilot review).
                if a["last_ts"] > row_ts + MATCH_SLACK_S or a["last_ts"] < row_ts - MATCH_LOOKBACK_S:
                    continue
                # Any attempt that finished before the row beats one that finished
                # after it (a later re-attempt or follow-up inside the slack window
                # must not take the row from its real attempt); among pre-row
                # candidates the latest wins, and future ones are only a clock-skew
                # fallback, the earliest of them winning.
                if best is None:
                    best = a
                    continue
                a_pre, b_pre = a["last_ts"] <= row_ts, best["last_ts"] <= row_ts
                if a_pre != b_pre:
                    if a_pre:
                        best = a
                elif a_pre and a["last_ts"] > best["last_ts"]:
                    best = a
                elif not a_pre and a["last_ts"] < best["last_ts"]:
                    best = a
        if best:
            claimed.add(best["agent_id"])
        result[id(row)] = best
    return result


def plan_backfill(rows: list, attempts: list) -> list:
    """One action per ticket row that lacks figures and has no usable `ticket_cost`
    row yet: {"row", "attempt"|None, "metadata"}. Rows that already have figures are
    never touched. Append-only: the action is a NEW `ticket_cost` event.

    A session whose `ticket_cost` event is unavailable is retried: when a transcript
    matches it now, a new available event is appended (the report prefers it). While
    nothing matches, it is skipped, so re-running never piles up unavailable
    duplicates."""
    done = {r.get("session_id") for r in rows if r.get("event_type") == COST_EVENT and _cost_available(r)}
    done_unavailable = {r.get("session_id") for r in rows if r.get("event_type") == COST_EVENT and not _cost_available(r)}
    ticket_rows = [r for r in rows if str(r.get("event_type", "")).startswith("ticket_") and r.get("event_type") != COST_EVENT]
    matches = match_rows(ticket_rows, attempts)
    actions = []
    for row in sorted(ticket_rows, key=lambda r: r.get("event_timestamp") or ""):
        if has_figures(row) or row.get("session_id") in done:
            continue
        attempt = matches[id(row)]
        own_cost = (row.get("metadata") or {}).get("cost")
        already_unavailable = row.get("session_id") in done_unavailable or (
            isinstance(own_cost, dict) and own_cost.get("available") is False)
        if attempt is None and already_unavailable:
            continue
        issue = row_issue(row)
        if attempt:
            meta = {"issue": issue, "figures_for": row.get("event_type"), **figures(attempt),
                    "model": primary_model(attempt), "cost": cost_metadata(attempt, "transcript_backfill")}
        else:
            meta = {"issue": issue, "figures_for": row.get("event_type"),
                    "cost": {"source": "transcript_backfill", "available": False,
                             "reason": "no subagent transcript matched this row"}}
        actions.append({"row": row, "attempt": attempt, "metadata": meta})
    return actions


def compare_existing(rows: list, attempts: list) -> list:
    """For rows that already carry figures, the transcript total beside the old one:
    [{"session_id", "issue", "old_tokens", "transcript_total", "transcript_no_reads",
      "transcript_no_cache", "ratio_total", "ratio_no_reads", "ratio_no_cache", ...}]. Reports only; changes nothing."""
    ticket_rows = [r for r in rows if str(r.get("event_type", "")).startswith("ticket_") and r.get("event_type") != COST_EVENT]
    matches = match_rows(ticket_rows, attempts)
    out = []
    for row in ticket_rows:
        if not has_figures(row):
            continue
        a = matches[id(row)]
        if a is None:
            continue
        t = totals(a["by_model"])
        meta = row["metadata"]
        old = meta["subagent_tokens"]
        no_cache = t["input_tokens"] + t["output_tokens"]
        no_reads = t["input_tokens"] + t["cache_creation_tokens"] + t["output_tokens"]
        out.append({
            "session_id": row.get("session_id"), "issue": row_issue(row), "old_tokens": old,
            "transcript_total": t["total_tokens"], "transcript_no_cache": no_cache,
            "ratio_total": (old / t["total_tokens"]) if t["total_tokens"] else None,
            "ratio_no_cache": (old / no_cache) if no_cache else None,
            "transcript_no_reads": no_reads, "ratio_no_reads": (old / no_reads) if no_reads else None,
            "old_tool_calls": meta.get("tool_calls"), "transcript_tool_calls": a["tool_calls"],
            "old_wall_clock_s": meta.get("wall_clock_s"), "transcript_wall_clock_s": a["wall_clock_s"],
        })
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _cost_available(row: dict) -> bool:
    cost = (row.get("metadata") or {}).get("cost")
    return isinstance(cost, dict) and bool(cost.get("available"))


def build_report(rows: list) -> dict:
    """Per-ticket and per-model totals over autowork rows. A ticket's figures come
    from a `ticket_cost` row sharing its session_id when there is one, else from
    its own row's `cost` metadata. Rows with only the legacy single total are counted as `legacy`
    (no model or cache split exists for them) and are left out of the per-model
    table and the cache share rather than guessed into it."""
    # A `ticket_cost` event for the row's session always wins over the row's own
    # `cost` (a row first recorded with available=false can be recovered later by
    # backfill; Copilot review). Should one session have several, an
    # available one wins over an unavailable one.
    cost_rows: dict = {}
    for r in rows:
        if r.get("event_type") != COST_EVENT:
            continue
        sid = r.get("session_id")
        prior = cost_rows.get(sid)
        if prior is None or (not _cost_available(prior) and _cost_available(r)):
            cost_rows[sid] = r
    tickets = []
    per_model: dict = {}
    unavailable = legacy = 0
    for row in sorted((r for r in rows if str(r.get("event_type", "")).startswith("ticket_") and r.get("event_type") != COST_EVENT),
                      key=lambda r: r.get("event_timestamp") or ""):
        meta = row.get("metadata") or {}
        cost = meta.get("cost")
        source_meta = meta
        if row.get("session_id") in cost_rows:
            source_meta = cost_rows[row["session_id"]].get("metadata") or {}
            cost = source_meta.get("cost")
        entry = {"session_id": row.get("session_id"), "issue": row_issue(row), "event_type": row.get("event_type"),
                 "timestamp": row.get("event_timestamp")}
        if isinstance(cost, dict) and cost.get("available") and isinstance(cost.get("by_model"), dict):
            t = totals(cost["by_model"])
            entry.update(status="split", total_tokens=t["total_tokens"], cache_read_share=t["cache_read_share"],
                         tool_calls=source_meta.get("tool_calls"), wall_clock_s=source_meta.get("wall_clock_s"),
                         kind=cost.get("kind"), cache_creation_tokens=t["cache_creation_tokens"],
                         longest_gap_s=cost.get("longest_gap_s"), cold_rebuilds=cost.get("cold_rebuilds"))
            for model, r in cost["by_model"].items():
                acc = per_model.setdefault(model, {"tickets": 0, "requests": 0, **{k: 0 for k in _FIELDS}})
                acc["tickets"] += 1
                acc["requests"] += _count(r.get("requests"))
                for k in _FIELDS:
                    acc[k] += _count(r.get(k))
        elif isinstance(cost, dict) and cost.get("available") is False:
            entry["status"] = "unavailable"
            unavailable += 1
        elif has_figures(row):
            entry.update(status="legacy", total_tokens=meta["subagent_tokens"], tool_calls=meta.get("tool_calls"),
                         wall_clock_s=meta.get("wall_clock_s"))
            legacy += 1
        else:
            entry["status"] = "no_figures"
        tickets.append(entry)
    overall = totals(per_model)
    shares = [t["cache_read_share"] for t in tickets if t.get("cache_read_share") is not None]
    return {"tickets": tickets, "per_model": per_model, "overall": overall, "legacy": legacy, "unavailable": unavailable,
            "median_ticket_cache_read_share": statistics.median(shares) if shares else None}


def _n(v) -> str:
    return "n/a" if v is None else f"{v:,.0f}"


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def format_report(rep: dict) -> str:
    lines = ["Autowork cost, derived from subagent transcripts", ""]
    if not rep["tickets"]:
        return "No autowork ticket rows recorded yet."
    lines.append("Per ticket:")
    for t in rep["tickets"]:
        if t["status"] == "split":
            lines.append(f"  #{t['issue']} {t['event_type']}  {_n(t['total_tokens'])} tokens, cache reads {_pct(t['cache_read_share'])}, "
                         f"{_n(t['tool_calls'])} tool calls, {_n(t['wall_clock_s'])}s, cache creation {_n(t['cache_creation_tokens'])}, "
                         f"longest gap {_n(t['longest_gap_s'])}s, cold-cache rebuilds {_n(t['cold_rebuilds'])} ({t['session_id']})")
        elif t["status"] == "legacy":
            lines.append(f"  #{t['issue']} {t['event_type']}  {_n(t['total_tokens'])} tokens (hand-copied total, no model or cache split) ({t['session_id']})")
        elif t["status"] == "unavailable":
            lines.append(f"  #{t['issue']} {t['event_type']}  UNAVAILABLE: transcript gone ({t['session_id']})")
        else:
            lines.append(f"  #{t['issue']} {t['event_type']}  no figures recorded ({t['session_id']})")
    lines += ["", "Per model (tickets with a transcript-derived split only):"]
    for model, r in sorted(rep["per_model"].items(), key=lambda kv: -sum(kv[1][k] for k in _FIELDS)):
        total = sum(r[k] for k in _FIELDS)
        share = r["cache_read_tokens"] / total if total else None
        lines.append(f"  {model}: {r['tickets']} tickets, {_n(r['requests'])} requests, input {_n(r['input_tokens'])}, "
                     f"cache creation {_n(r['cache_creation_tokens'])}, cache reads {_n(r['cache_read_tokens'])}, "
                     f"output {_n(r['output_tokens'])} (cache reads {_pct(share)} of {_n(total)})")
    o = rep["overall"]
    lines += ["", f"Overall: {_n(o['total_tokens'])} tokens, cache reads {_pct(o['cache_read_share'])}; "
              f"median ticket cache-read share {_pct(rep['median_ticket_cache_read_share'])}.",
              f"Left out of the model table: {rep['legacy']} legacy hand-copied rows, {rep['unavailable']} rows whose transcripts are gone.",
              "Subagent cost only; the orchestrator's own session is not included. Output tokens include thinking."]
    return "\n".join(lines)


def format_compare(rows: list) -> str:
    if not rows:
        return "No existing figures could be matched to a transcript."
    lines = ["Existing hand-copied figures vs. transcript totals (old / transcript):"]
    for r in rows:
        lines.append(f"  {r['session_id']} #{r['issue']}: old {_n(r['old_tokens'])} vs all-token total {_n(r['transcript_total'])} "
                     f"({_pct(r['ratio_total'])}), vs all but cache reads {_n(r['transcript_no_reads'])} ({_pct(r['ratio_no_reads'])}), "
                     f"vs input+output {_n(r['transcript_no_cache'])} ({_pct(r['ratio_no_cache'])}); "
                     f"tool calls {r['old_tool_calls']} vs {r['transcript_tool_calls']}")
    for key, label in (("ratio_total", "all four counters"), ("ratio_no_reads", "all but cache reads"),
                       ("ratio_no_cache", "input + output only")):
        vals = [r[key] for r in rows if r[key] is not None]
        if vals:
            lines.append(f"Median old/transcript ratio against {label}: {_pct(statistics.median(vals))} "
                         f"(min {_pct(min(vals))}, max {_pct(max(vals))}, {len(vals)} rows)")
    return "\n".join(lines)
