#!/usr/bin/env python3
"""
Shared, dependency-free join of the savings tracker's compression events to the
session transcript: which MODEL and which TURN each compression
happened under, plus a per-model chars-per-token ratio measured from the same
transcript.

Why this lives at SessionEnd, not in the PostToolUse hook: the hook payload has
no model, and Claude Code writes the transcript asynchronously, so at hook time
the current turn may not be in it yet. By SessionEnd the transcript is complete.

What the transcript gives (all of it recorded by Claude Code, none estimated):
  - each assistant entry: `message.id` (the turn), `message.model`, `timestamp`,
    `message.usage`, and one `tool_use` block per call, whose `id` is the same
    `tool_use_id` the PostToolUse payload carries (stored on the event by the compress hook);
  - each tool result (a user entry's `tool_result` block): which call it answers
    and its text. A compressed result's text starts with `[compressed A -> B
    chars`, which is how the join is cross-checked.

Two figures come out of it and must never be conflated:
  - EXACT: the characters before and after compression, grouped per model. Those
    are counts the tool made, so the join only decides which group they land in.
  - DERIVED: a token figure. A saving cannot be read off the transcript, because
    the uncompressed result never entered context and was never counted. It is
    derived by dividing the characters removed by this session's OWN observed
    chars-per-token for that model (see ratio_by_model), and counting each saving
    once for every request that carried it. It is not a measurement, and it is
    left out when the session has too few samples to derive a ratio.

The transcript format is internal to Claude Code and undocumented; it may change
with any release. Every reader here fails open (None / empty) rather than raising.
"""

import re
import statistics
from pathlib import Path
from typing import Optional

import transcript_context_lib

# A model's ratio needs at least this many samples, or no derived figure is shown.
# Few samples make a median that one odd turn can move, and the figure it feeds is
# already derived: showing nothing is more honest than showing a shaky number.
MIN_RATIO_SAMPLES = 5

# A turn only becomes a ratio sample when the tool results it added total at least
# this many characters. Context also grows by things this reader does not size
# (system reminders, attachments), a roughly fixed amount per turn; against a
# large result that is noise, against a small one it dominates the ratio.
MIN_SAMPLE_CHARS = 2000

# The unattributed bucket: events with no matching transcript turn (no
# tool_use_id, or one the transcript does not contain).
UNATTRIBUTED = ""

_MARKER_RE = re.compile(r"\A\s*\[compressed \d+ -> \d+ chars")


def _count(v) -> int:
    """A usage counter as an int; anything else (the format is internal and may
    change) counts as 0 rather than raising."""
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def _result_text(content) -> tuple:
    """(characters, text_only, head) for one tool_result's content. `content` is a
    string or a list of blocks; `text_only` is False when any block is not text
    (an image), since its size in context cannot be measured in characters."""
    if isinstance(content, str):
        return len(content), True, content[:80]
    chars = 0
    head = ""
    text_only = True
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                if not head:
                    head = block["text"][:80]
                chars += len(block["text"])
            else:
                text_only = False
    else:
        text_only = False
    return chars, text_only, head


def _parse_file(path) -> Optional[dict]:
    """One transcript file (the main session's, or one subagent's -- each has its
    own context, so each is read on its own). Returns
    {"calls": {tool_use_id: {...}}, "marker_ids": set, "samples": {model: [chars/token]}}
    or None when the file has no assistant turn."""
    turns: list = []          # one per distinct assistant message.id, in order
    by_message: dict = {}     # message.id -> index into turns
    call_turn: dict = {}      # tool_use_id -> index into turns
    marker_ids: set = set()
    samples: dict = {}
    segment = 0               # bumped at each compaction: a result does not survive one
    pending_chars = 0         # tool-result characters since the last turn started
    pending_valid = True      # False once anything unmeasurable joined them
    prev: Optional[dict] = None
    anon = 0

    for obj in transcript_context_lib.iter_entries(path):
        kind = obj.get("type")
        if kind == "system" and obj.get("subtype") == "compact_boundary":
            segment += 1
            pending_chars, pending_valid, prev = 0, True, None
            continue
        message = obj.get("message")
        if not isinstance(message, dict):
            continue
        if kind == "user":
            if obj.get("isCompactSummary"):
                # The summary replaces the earlier context; the boundary entry
                # normally came first, but either one marks it.
                if prev is not None or pending_chars:
                    segment += 1
                pending_chars, pending_valid, prev = 0, True, None
                continue
            content = message.get("content")
            has_result = False
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    has_result = True
                    chars, text_only, head = _result_text(block.get("content"))
                    tool_use_id = block.get("tool_use_id")
                    if _MARKER_RE.match(head):
                        # A compressed result's text is a model summary, not raw
                        # tool output: it is not a fair sample of raw text.
                        if isinstance(tool_use_id, str):
                            marker_ids.add(tool_use_id)
                        pending_valid = False
                    elif not text_only:
                        pending_valid = False
                    pending_chars += chars
            if not has_result:
                # A typed prompt: context grew by something this reader cannot
                # size, so the next growth is not attributable to tool results.
                pending_valid = False
            continue
        if kind != "assistant" or message.get("model") == "<synthetic>":
            continue
        usage = message.get("usage")
        model = message.get("model")
        message_id = message.get("id")
        key = message_id if isinstance(message_id, str) and message_id else None
        if key is None:
            anon += 1
            key = f"\0anon{anon}"
        idx = by_message.get(key)
        if idx is None:
            ctx = out = None
            if isinstance(usage, dict) and usage:
                ctx = (_count(usage.get("input_tokens")) + _count(usage.get("cache_read_input_tokens"))
                       + _count(usage.get("cache_creation_input_tokens")))
                out = _count(usage.get("output_tokens"))
            turn = {
                "id": message_id if isinstance(message_id, str) else "",
                "model": model if isinstance(model, str) and model else UNATTRIBUTED,
                "timestamp": obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else "",
                "ctx": ctx, "out": out, "segment": segment,
            }
            # A ratio sample: the growth in real context between two consecutive
            # requests of one model, less what the model itself wrote in between,
            # against the characters of the (uncompressed, text-only) tool results
            # that arrived in between.
            if (prev is not None and pending_valid and pending_chars >= MIN_SAMPLE_CHARS
                    and ctx is not None and prev["ctx"] is not None and prev["out"] is not None
                    and turn["model"] and turn["model"] == prev["model"]
                    and prev["segment"] == segment):
                growth = ctx - (prev["ctx"] + prev["out"])
                if growth > 0:
                    samples.setdefault(turn["model"], []).append(pending_chars / growth)
            pending_chars, pending_valid = 0, True
            by_message[key] = idx = len(turns)
            turns.append(turn)
            prev = turn
        # Every content block of a turn is its own line (see
        # savings_ledger.parse_transcript_token_counts), so the tool_use blocks
        # are collected for a turn already seen too.
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                    call_turn.setdefault(block["id"], idx)

    if not turns:
        return None
    # Requests that came after each turn within the same compaction segment: the
    # ones that carried its results.
    per_segment: dict = {}
    for t in turns:
        per_segment[t["segment"]] = per_segment.get(t["segment"], 0) + 1
    seen_in_segment: dict = {}
    after: list = []
    for t in turns:
        seen_in_segment[t["segment"]] = seen_in_segment.get(t["segment"], 0) + 1
        after.append(per_segment[t["segment"]] - seen_in_segment[t["segment"]])
    calls = {
        tuid: {"model": turns[i]["model"], "turn_id": turns[i]["id"],
               "timestamp": turns[i]["timestamp"], "turns_after": after[i]}
        for tuid, i in call_turn.items()
    }
    return {"calls": calls, "marker_ids": marker_ids, "samples": samples}


def read_transcript(transcript_path) -> Optional[dict]:
    """The attribution index for a whole SESSION: the main transcript plus every
    subagent transcript beside it (`<session>/subagents/agent-*.jsonl`, the same
    layout savings_ledger.parse_session_token_counts reads). Each file is parsed
    separately -- a subagent has its own context, so turns, compactions and ratio
    samples never cross files -- then merged.

    Returns {"calls": {tool_use_id: {"model", "turn_id", "timestamp",
    "turns_after"}}, "marker_ids": set of tool_use_ids whose result text starts
    with the compression marker, "samples": {model: [chars per token, ...]}}, or
    None when nothing could be read. Always fails open -- never raises.
    """
    if not transcript_path:
        return None
    files = [Path(transcript_path)]
    try:
        files += sorted((Path(transcript_path).with_suffix("") / "subagents").glob("agent-*.jsonl"))
    except Exception:
        pass  # keep the main transcript
    merged: dict = {"calls": {}, "marker_ids": set(), "samples": {}}
    found = False
    for f in files:
        try:
            part = _parse_file(f)
        except Exception:
            continue
        if part is None:
            continue
        found = True
        for tuid, info in part["calls"].items():
            merged["calls"].setdefault(tuid, info)
        merged["marker_ids"] |= part["marker_ids"]
        for model, values in part["samples"].items():
            merged["samples"].setdefault(model, []).extend(values)
    return merged if found else None


def ratio_by_model(samples: dict, minimum: int = MIN_RATIO_SAMPLES) -> dict:
    """{model: {"chars_per_token": median, "samples": n}} for each model with at
    least `minimum` samples. A model with fewer is absent: no ratio, so no derived
    figure. The median, not the mean: a turn whose growth includes something this
    reader cannot see (an attachment) is an outlier, not a measurement."""
    return {
        model: {"chars_per_token": statistics.median(values), "samples": len(values)}
        for model, values in samples.items() if len(values) >= minimum and statistics.median(values) > 0
    }


def attribute_events(events: list, index: Optional[dict], all_events: Optional[list] = None) -> dict:
    """
    Join `events` (the ledger's COUNTED events: credited, with a recorded before
    and after) to the transcript index and roll them up per model.

    Returns {"by_model": [row, ...], "unjoined": int, "marker_without_event": [id, ...]}.
    A row is {"model", "event_count", "raw_chars_sum", "out_chars_sum",
    "saved_chars", "derived_tokens", "chars_per_token", "ratio_samples"}; model
    UNATTRIBUTED ("") collects the events that did not join, so the rows always
    add up to the same characters as the session itself.

    `saved_chars` is derived from the row's own before/after (max(0, before -
    after)), like every other saving in the tracker. `derived_tokens` (None when
    the model has no ratio, or for the unattributed row) is the DERIVED figure:
    each event's saving, clamped at 0, times the number of requests that carried
    its result (the turns after the call, within one compaction segment -- the
    result is written into context on the next request and read on each one
    after, until the session ends or compacts), divided by the model's observed
    chars per token. Every saving counted once per request, so it is comparable
    with the real "tokens processed" total. Rounded to a whole token.

    `all_events` (default `events`) is every ledger event of the session, joined
    or not, credited or not: it is the set a transcript result with the
    compression marker is checked against (`marker_without_event`: marker results
    the ledger has no event for -- an event that never reached the ledger).
    """
    calls = (index or {}).get("calls", {})
    ratios = ratio_by_model((index or {}).get("samples", {}))
    rows: dict = {}
    weighted: dict = {}
    unjoined = 0
    for e in events:
        tuid = e.get("tool_use_id")
        info = calls.get(tuid) if isinstance(tuid, str) else None
        model = info["model"] if info else UNATTRIBUTED
        if info is None:
            unjoined += 1
        row = rows.setdefault(model, {"model": model, "event_count": 0, "raw_chars_sum": 0, "out_chars_sum": 0})
        row["event_count"] += 1
        row["raw_chars_sum"] += e["raw_chars"]
        row["out_chars_sum"] += e["out_chars"]
        if info:
            weighted[model] = weighted.get(model, 0) + max(0, e["raw_chars"] - e["out_chars"]) * info["turns_after"]
    out = []
    for model, row in rows.items():
        row["saved_chars"] = max(0, row["raw_chars_sum"] - row["out_chars_sum"])
        ratio = ratios.get(model) if model != UNATTRIBUTED else None
        if ratio:
            row["derived_tokens"] = round(weighted.get(model, 0) / ratio["chars_per_token"])
            row["chars_per_token"] = ratio["chars_per_token"]
            row["ratio_samples"] = ratio["samples"]
        else:
            row["derived_tokens"] = None
            row["chars_per_token"] = None
            # Kept even without a ratio, so a caller can say how close it was.
            row["ratio_samples"] = len((index or {}).get("samples", {}).get(model, [])) if model != UNATTRIBUTED else 0
        out.append(row)
    out.sort(key=lambda r: (r["model"] == UNATTRIBUTED, -r["saved_chars"], r["model"]))
    known = {e.get("tool_use_id") for e in (events if all_events is None else all_events) if isinstance(e.get("tool_use_id"), str)}
    marker_without_event = sorted((index or {}).get("marker_ids", set()) - known)
    return {"by_model": out, "unjoined": unjoined, "marker_without_event": marker_without_event}
