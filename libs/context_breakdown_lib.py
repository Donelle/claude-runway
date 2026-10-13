#!/usr/bin/env python3
"""
Shared, dependency-free itemization of a subagent transcript's CONTEXT: what the
fixed starting context (request 0) is made of, and what each later request added.

Why this exists: a subagent's first request is already about 56k tokens and its
final context is two to four times that. Nothing said what the 56k holds or what
makes the context grow, so every figure was a guess. A transcript records more
than the API's token counts: it also stores the attachments Claude Code injects
(skill list, CLAUDE.md files, agent list, MCP server instructions, deferred tool
names) and a `prompt_snapshot` attachment whose `tools` list is the full tool
definitions sent to the model. This module reads those and joins them to the
API's own per-request token counts.

Three things worth knowing before trusting any number from here:

  - The transcript stores TEXT, not tokens. Turning characters into tokens needs a
    ratio, and the ratio differs by content (tool definitions and code tokenize
    far denser than prose). `calibrate` fits those ratios from the API's own
    counts (a least-squares fit over many requests) instead of assuming one.
  - `usage.output_tokens` in a transcript is a streaming-start snapshot, not the
    final count (a 109-request run sums to about 1.6k although every response
    carries a tool call). So the growth of the context between two requests is NOT
    "previous output + new input" with a known output; the assistant's own content
    is one more class of added text and gets its own fitted ratio.
  - Extended-thinking blocks are stored with EMPTY text and a long `signature`;
    the signature's length is the only recorded hint of how much thinking entered
    the context, so it is used as that class's size (a proxy, fitted like the rest).

The transcript format is internal to Claude Code and undocumented; it may change
with any release. Every reader here fails open (None / empty / zero) rather than
raising. Same stance as libs/transcript_context_lib.py, whose reader it shares.
"""

import bisect
import collections
import json
from typing import Optional

import transcript_context_lib

# Text classes that make a request's context grow after request 0. The fit uses
# these as its columns; add one here and `growth_features` and the fit follow.
ASSISTANT_CLASSES = ("tool_use_input", "thinking_signature", "assistant_text")
USER_CLASSES = ("result_Read", "result_Bash", "result_Edit", "result_Write", "result_other",
                "agent_listing", "mcp_instructions", "edited_file_notice", "hook_context", "deferred_schema",
                "user_text")
# Attachment types whose content is NOT in the model's context (bookkeeping the
# fit found to contribute nothing: hook_success carries the hook's own stdout,
# which the harness does not re-send; a real hook result arrives as
# `hook_additional_context`). `deferred_tools_record` is listed because its
# `toolInputCopies` are bookkeeping, but its `entries` ARE content (the full
# schema of a tool loaded on demand; the ToolSearch result itself is only a
# stub), so `growth_features` counts those as their own class, `deferred_schema`.
NOT_IN_CONTEXT = ("hook_success", "prompt_snapshot", "deferred_tools_record", "total_tokens_reminder")


def _segments(entries: list, reqs: list) -> list:
    """Compaction segment of each request: how many compaction markers
    (`compact_boundary` system entries or `isCompactSummary` user entries, the same
    two libs/transcript_attribution_lib.py uses) precede it. A result does not
    survive a compaction and the context drops across it, so a growth row or a
    carried read never spans two segments. Subagent transcripts have none today;
    a main-session transcript can."""
    marks = []
    for idx, obj in enumerate(entries):
        if (obj.get("type") == "system" and obj.get("subtype") == "compact_boundary") or obj.get("isCompactSummary"):
            marks.append(idx)
    return [bisect.bisect_left(marks, r["first_index"]) for r in reqs]


def _blocks_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    n = 0
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict):
                n += len(b["text"]) if b.get("type") == "text" and isinstance(b.get("text"), str) else len(json.dumps(b))
    return n


def _count(v) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def read_requests(path) -> list:
    """One dict per distinct API response, in order of first appearance:
    {"id", "first_index", "context"} where `context` is input + cache creation +
    cache reads (the whole prompt the API processed for that request), taken once
    per `message.id` as the field-wise max (a streamed message is written in
    several lines). Empty list when there is none."""
    out: list = []
    pos: dict = {}
    for idx, obj in enumerate(transcript_context_lib.iter_entries(path)):
        message = obj.get("message")
        if obj.get("type") != "assistant" or not isinstance(message, dict) or message.get("model") == "<synthetic>":
            continue
        usage = message.get("usage")
        mid = message.get("id")
        if not isinstance(usage, dict) or not usage or not isinstance(mid, str) or not mid:
            continue
        ctx = sum(_count(usage.get(k)) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        if mid not in pos:
            pos[mid] = len(out)
            out.append({"id": mid, "first_index": idx, "context": ctx})
        else:
            out[pos[mid]]["context"] = max(out[pos[mid]]["context"], ctx)
    return out


def _attachment_text_chars(att: dict) -> int:
    """Characters of an attachment that the model sees. Types not listed render as
    their JSON size only when `unknown_counts` says so (see `start_components`)."""
    t = att.get("type")
    if t == "skill_listing":
        return len(att.get("content") or "")
    if t == "instructions":
        return sum(len(f.get("content") or "") for f in att.get("files") or [] if isinstance(f, dict))
    if t == "deferred_tools_delta":
        return len("\n".join(x for x in att.get("addedLines") or [] if isinstance(x, str)))
    if t == "agent_listing_delta":
        return len("\n".join(x for x in att.get("addedLines") or [] if isinstance(x, str)))
    if t == "mcp_instructions_delta":
        return sum(len(x) for x in att.get("addedBlocks") or [] if isinstance(x, str))
    if t == "environment":
        return len(json.dumps(att.get("snapshot")))
    if t == "session_context":
        ctx = att.get("context")
        return sum(len(v) for v in ctx.values() if isinstance(v, str)) if isinstance(ctx, dict) else 0
    if t == "model":
        return len(att.get("text") or "")
    if t == "date":
        return len(att.get("date") or "")
    if t == "edited_text_file":
        return len(att.get("snippet") or "")
    if t == "hook_additional_context":
        c = att.get("content")
        return len(c) if isinstance(c, str) else len(json.dumps(c))
    return len(json.dumps(att))


def start_components(path) -> Optional[dict]:
    """Characters of everything recorded before request 0, by item, plus the tool
    definitions (full JSON length of each tool, from the first `prompt_snapshot`
    that carries a `tools` list). Returns
    {"context0", "components": {name: chars}, "tools": {tool: chars}} or None when the
    transcript has no request. Items: prompt (the first user turn), user_other,
    system_prompt, skill_listing, instructions (CLAUDE.md files), deferred_tool_names,
    environment, session_context, other_attachments. The tool list is stored beside
    them under "tool_definitions" in `components` (the sum) and per tool in "tools"."""
    reqs = read_requests(path)
    if not reqs:
        return None
    first = reqs[0]["first_index"]
    comp: collections.Counter = collections.Counter()
    tools: dict = {}
    system_chars = 0
    seen_prompt = False
    for idx, obj in enumerate(transcript_context_lib.iter_entries(path)):
        t = obj.get("type")
        if t == "attachment":
            att = obj.get("attachment")
            if not isinstance(att, dict):
                continue
            at = att.get("type")
            if at == "prompt_snapshot":
                sp = att.get("systemPrompt")
                if isinstance(sp, list) and not system_chars:
                    system_chars = sum(len(s) for s in sp if isinstance(s, str)) + len(att.get("cliPrefix") or "")
                tl = att.get("tools")
                if isinstance(tl, list) and tl and not tools:
                    for tool in tl:
                        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                            tools[tool["name"]] = len(json.dumps(tool))
                continue
            if idx >= first or at in NOT_IN_CONTEXT or at in ("agent_listing_delta", "mcp_instructions_delta"):
                continue
            key = {"skill_listing": "skill_listing", "instructions": "instructions",
                   "deferred_tools_delta": "deferred_tool_names", "environment": "environment",
                   "session_context": "session_context"}.get(str(at), "other_attachments")
            comp[key] += _attachment_text_chars(att)
        elif t == "user" and idx < first:
            message = obj.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            text = content if isinstance(content, str) else "\n".join(
                b["text"] for b in content or [] if isinstance(b, dict) and isinstance(b.get("text"), str))
            if not seen_prompt:
                comp["prompt"] += len(text)
                seen_prompt = True
            else:
                comp["user_other"] += len(text)
    comp["system_prompt"] = system_chars
    comp["tool_definitions"] = sum(tools.values())
    return {"context0": reqs[0]["context"], "components": dict(comp), "tools": tools}


# Tokens per character measured by putting each text, verbatim, in front of a
# no-op subagent through a Read of a file and subtracting a control read (see
# docs/metrics.md, "What a subagent starts with"). The skill list and CLAUDE.md
# rates carry the Read tool's line-number prefixes, which add roughly 0.5k tokens
# to the skill list; plain prose and the harness's own text are close to 0.37.
DEFAULT_START_RATES = {
    "prompt": 0.37, "user_other": 0.37, "system_prompt": 0.37, "skill_listing": 0.366,
    "instructions": 0.325, "deferred_tool_names": 0.37, "environment": 0.37,
    "session_context": 0.37, "other_attachments": 0.37,
}


# Fitted tokens per character of a Read result (about 2.5 characters per token),
# used to size what a whole-file read costs while it is carried in the context.
READ_TOKENS_PER_CHAR = 0.397


def itemize_start(start: dict, rates: Optional[dict] = None) -> dict:
    """Turns `start_components` output into tokens that add up to request 0.
    Every item except the tool definitions is its characters times a measured
    tokens-per-character rate. The tool definitions are NOT given a rate: the API
    does not tokenize them as the JSON text recorded in the transcript (the same
    text read through a user turn costs about 1.4 times as much), so their share is
    what is LEFT of request 0, and it also absorbs the estimation error of every
    other line. `tool_definitions_by_tool` splits it by each tool's share of the
    recorded JSON characters, an assumption stated here, not a measurement.
    Returns {"context0", "items": {name: tokens}, "tool_definitions_by_tool"}."""
    r = dict(DEFAULT_START_RATES)
    r.update(rates or {})
    comps = start["components"]
    items = {k: round(v * r.get(k, 0.37)) for k, v in comps.items() if k != "tool_definitions"}
    items["tool_definitions"] = start["context0"] - sum(items.values())
    total_chars = sum(start["tools"].values()) or 1
    by_tool = {n: round(items["tool_definitions"] * c / total_chars) for n, c in start["tools"].items()}
    return {"context0": start["context0"], "items": items, "tool_definitions_by_tool": by_tool}


def growth_features(path) -> list:
    """Per request after the first: {"index", "delta", "chars": {class: chars}}.
    `delta` is this request's context minus the previous one's. `chars` is the
    recorded text that entered the context between them: the assistant's own
    previous response (tool inputs, thinking signatures, text) and everything that
    arrived on the user side (tool results by tool, attachments, hook context).
    Entries are placed by position: an entry belongs to the first request that
    starts after it, and a response's own blocks belong to the NEXT request's
    context (a tool result can be written between two blocks of one response)."""
    reqs = read_requests(path)
    if len(reqs) < 2:
        return []
    starts = [r["first_index"] for r in reqs]
    index_of = {r["id"]: k for k, r in enumerate(reqs)}
    feats = [collections.Counter() for _ in reqs]
    tool_name: dict = {}
    entries = list(transcript_context_lib.iter_entries(path))
    for obj in entries:
        first_msg = obj.get("message")
        if obj.get("type") == "assistant" and isinstance(first_msg, dict):
            for b in first_msg.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and isinstance(b.get("id"), str):
                    tool_name[b["id"]] = str(b.get("name"))
    for idx, obj in enumerate(entries):
        t = obj.get("type")
        raw = obj.get("message")
        message: dict = raw if isinstance(raw, dict) else {}
        if t == "assistant":
            k = index_of.get(message.get("id"))
            if k is None:
                continue
            for b in message.get("content") or []:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "tool_use":
                    feats[k]["tool_use_input"] += len(json.dumps(b.get("input"))) + len(str(b.get("name")))
                elif bt == "thinking":
                    feats[k]["thinking_signature"] += len(b.get("signature") or "") + len(b.get("thinking") or "")
                elif bt == "text":
                    feats[k]["assistant_text"] += len(b.get("text") or "")
            continue
        k = bisect.bisect_right(starts, idx)
        if k >= len(reqs):
            continue
        f = feats[k]
        if t == "user":
            content = message.get("content")
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_result":
                        name = tool_name.get(b.get("tool_use_id"), "")
                        cls = "result_" + name if name in ("Read", "Bash", "Edit", "Write") else "result_other"
                        f[cls] += _blocks_chars(b.get("content"))
                    else:
                        f["user_text"] += len(json.dumps(b))
            elif isinstance(content, str):
                f["user_text"] += len(content)
        elif t == "attachment" and isinstance(obj.get("attachment"), dict):
            att = obj["attachment"]
            at = att.get("type")
            if at == "deferred_tools_record":
                entries_json = att.get("entries")
                if entries_json:
                    f["deferred_schema"] += len(json.dumps(entries_json))
                continue
            if at in NOT_IN_CONTEXT:
                continue
            key = {"agent_listing_delta": "agent_listing", "mcp_instructions_delta": "mcp_instructions",
                   "edited_text_file": "edited_file_notice", "hook_additional_context": "hook_context"}.get(str(at), "user_text")
            f[key] += _attachment_text_chars(att)
    segment = _segments(entries, reqs)
    out = []
    for i in range(1, len(reqs)):
        if segment[i] != segment[i - 1]:
            continue  # the context was rebuilt here; the difference is not growth
        chars: collections.Counter = collections.Counter()
        for c in ASSISTANT_CLASSES:
            chars[c] = feats[i - 1][c]
        for c in USER_CLASSES:
            chars[c] = feats[i][c]
        out.append({"index": i, "delta": reqs[i]["context"] - reqs[i - 1]["context"], "chars": dict(chars)})
    return out


# --------------------------------------------------------------------------
# Least-squares calibration (stdlib only)
# --------------------------------------------------------------------------

def least_squares(rows: list, targets: list, ridge: float = 1e-6) -> list:
    """Ordinary least squares via the normal equations and Gaussian elimination.
    `rows` is a list of equal-length feature lists; returns one coefficient per
    feature. A tiny ridge term keeps a column of all zeros from making the system
    singular (its coefficient comes back 0). Not for ill-conditioned problems."""
    if not rows:
        return []
    n = len(rows[0])
    m: list[list[float]] = [[float(sum(r[i] * r[j] for r in rows)) + (ridge if i == j else 0.0) for j in range(n)]
                            for i in range(n)]
    v: list[float] = [float(sum(r[i] * y for r, y in zip(rows, targets, strict=True))) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        v[i], v[p] = v[p], v[i]
        if not m[i][i]:
            continue
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                if f:
                    for c in range(i, n):
                        m[r][c] -= f * m[i][c]
                    v[r] -= f * v[i]
    return [v[i] / m[i][i] if m[i][i] else 0.0 for i in range(n)]


def calibrate_growth(paths) -> dict:
    """Fit tokens-per-character for each growth class over every request of every
    transcript in `paths`, plus a per-request constant (the wrapper tokens each
    request adds that no recorded text explains). Returns
    {"per_char": {class: tokens/char}, "per_request": tokens, "rows": n,
     "rmse": tokens}. Classes that never occur get 0."""
    classes = ASSISTANT_CLASSES + USER_CLASSES
    rows, targets = [], []
    for p in paths:
        for g in growth_features(p):
            rows.append([float(g["chars"].get(c, 0)) for c in classes] + [1.0])
            targets.append(float(g["delta"]))
    if not rows:
        return {"per_char": {c: 0.0 for c in classes}, "per_request": 0.0, "rows": 0, "rmse": 0.0}
    coef = least_squares(rows, targets)
    res = [y - sum(c * x for c, x in zip(coef, r, strict=True)) for r, y in zip(rows, targets, strict=True)]
    return {"per_char": dict(zip(classes, coef[:-1], strict=True)), "per_request": coef[-1], "rows": len(rows),
            "rmse": (sum(r * r for r in res) / len(res)) ** 0.5}


def reconcile(path, calibration: dict) -> Optional[dict]:
    """Splits one transcript's growth after request 0 into what the calibrated
    classes explain, per class, and the remainder. Returns
    {"context0", "final", "growth", "by_class": {class: tokens}, "per_request": tokens,
     "explained", "remainder", "chars": {class: chars}, "worst": [(request_index,
     delta, predicted)] largest under-explained requests} or None for a transcript
    with fewer than two requests."""
    reqs = read_requests(path)
    gf = growth_features(path)
    if len(reqs) < 2 or not gf:
        return None
    per_char = calibration["per_char"]
    by_class: collections.Counter = collections.Counter()
    chars: collections.Counter = collections.Counter()
    worst = []
    for g in gf:
        pred = calibration["per_request"]
        for c, n in g["chars"].items():
            t = n * per_char.get(c, 0.0)
            by_class[c] += t
            chars[c] += n
            pred += t
        worst.append((g["index"], g["delta"], round(pred)))
    explained = sum(by_class.values()) + calibration["per_request"] * len(gf)
    # The sum of the rows used, which is final minus first unless a compaction
    # removed rows (those are left out of both sides).
    growth = sum(g["delta"] for g in gf)
    worst.sort(key=lambda w: w[2] - w[1])
    return {"context0": reqs[0]["context"], "final": reqs[-1]["context"], "growth": growth,
            "by_class": {k: round(v) for k, v in by_class.items()},
            "per_request": round(calibration["per_request"] * len(gf)),
            "explained": round(explained), "remainder": round(growth - explained),
            "chars": dict(chars), "worst": [w for w in worst if w[1] - w[2] > 0][::-1][:5] or worst[-5:][::-1]}


# --------------------------------------------------------------------------
# Big reads
# --------------------------------------------------------------------------

def read_calls(path) -> list:
    """Every Read tool call in a transcript: [{"file", "ranged", "chars", "carried_requests"}].
    `ranged` is True when the call carried an offset or a limit. `chars` is the
    size of the result text (0 when the result is missing). Paths are returned as
    recorded; callers should report only a base name or a count, never content."""
    entries = list(transcript_context_lib.iter_entries(path))
    reqs = read_requests(path)
    index_of = {r["id"]: k for k, r in enumerate(reqs)}
    segment = _segments(entries, reqs)
    calls: dict = {}
    for obj in entries:
        message = obj.get("message")
        if obj.get("type") == "assistant" and isinstance(message, dict):
            k = index_of.get(message.get("id"))
            # Only the requests of the same compaction segment carry the result.
            carried = sum(1 for j in range(k + 1, len(reqs)) if segment[j] == segment[k]) if k is not None else 0
            for b in message.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Read" and isinstance(b.get("id"), str):
                    raw_inp = b.get("input")
                    inp: dict = raw_inp if isinstance(raw_inp, dict) else {}
                    calls[b["id"]] = {"file": str(inp.get("file_path") or ""),
                                      "ranged": inp.get("offset") is not None or inp.get("limit") is not None, "chars": 0,
                                      # Requests that carry this result: every later request
                                      # up to the next compaction.
                                      "carried_requests": carried}
    for obj in entries:
        message = obj.get("message")
        if obj.get("type") == "user" and isinstance(message, dict) and isinstance(message.get("content"), list):
            for b in message["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") in calls:
                    calls[b["tool_use_id"]]["chars"] = _blocks_chars(b.get("content"))
    return list(calls.values())
