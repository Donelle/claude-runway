#!/usr/bin/env python3
"""
Shared, dependency-free reading of what a Claude Code session transcript
records about its STARTING context -- the measured replacement for
the old chars / 3.5 "tool schema overhead" estimate.

Two things in a transcript are real, not estimated:

- The first assistant turn's `message.usage` (`input_tokens` +
  `cache_read_input_tokens` + `cache_creation_input_tokens`) is the token count
  the API reported for the whole fixed prefix of the session: system prompt,
  built-in tool definitions, any MCP tool definitions that are sent up front,
  CLAUDE.md and the first user message. It is one number for the whole prefix,
  so one transcript can never attribute a share of it to this toolkit; only a
  PAIR of sessions (toolkit servers connected vs. removed) can, by difference.
- `deferred_tools_delta` attachments list, by tool name, the tools Claude Code
  held back from that prefix and loads on demand (ToolSearch) instead. A tool
  named there does not have its schema in the starting context, so it is not
  "paid every turn". `deferred_tools_record` entries carry the full schema of
  tools that were later loaded, which gives their real size.

The transcript format is internal to Claude Code and undocumented; it may change
with any release. Every reader here fails open (None / empty) rather than raising,
and the CLI built on this (tools/measure_startup_context.py) says "no data"
instead of guessing. Same stance as savings_ledger.parse_transcript_token_counts.
"""

import json
from pathlib import Path
from typing import Iterable, Optional

# The core MCP servers this toolkit's templates always register
# (templates/mcp.json.template). A tool name in a transcript is
# `mcp__<server>__<tool>`. "Every always-on server" statistics (the survey's
# per-server counts and its "all N servers" line) use this tuple alone.
TOOLKIT_SERVERS = ("qdrant", "codebase-indexer", "memory-bank", "local-compress")

# Servers that exist only when the user opted in, so most projects and every older
# transcript lack them. `analytics` is written by `init --track-savings` (and offered
# by `upgrade` to a project that already tracks savings). Its tools still belong to
# the toolkit when present, so tool-name matching counts them, but they stay out of
# the always-on statistics: otherwise those would fall toward zero for a reason
# unrelated to what they measure.
OPTIONAL_TOOLKIT_SERVERS = ("analytics",)

# What tool-name matching uses by default: the core servers plus the optional ones.
ALL_TOOLKIT_SERVERS = TOOLKIT_SERVERS + OPTIONAL_TOOLKIT_SERVERS


def _entries(path):
    """Yield each parsed JSONL object of a transcript; skip blank/garbled lines."""
    try:
        # errors="replace": a partially written or corrupt transcript must degrade to
        # an undecodable (so skipped) line, never abort a whole survey.
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, RecursionError):
                    # JSONDecodeError is a ValueError; so is the int-conversion limit
                    # error for an absurdly long number. Deeply nested input raises
                    # RecursionError. Any of them is just an unreadable line.
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def iter_entries(path):
    """Public name for the shared line reader, for sibling libs that read the same
    transcript (libs/transcript_attribution_lib.py, and
    libs/autowork_cost_lib.py): same fail-open
    behavior, one reader so the two can't drift on what counts as a readable line."""
    return _entries(path)


def _first_usage(obj) -> Optional[dict]:
    """The usage block of an assistant entry that carries one, else None. Skips
    `<synthetic>` messages (Claude Code's own placeholders, all-zero usage)."""
    if obj.get("type") != "assistant":
        return None
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("model") == "<synthetic>":
        return None
    usage = message.get("usage")
    return usage if isinstance(usage, dict) and usage else None


def server_of(tool_name: str) -> Optional[str]:
    """`mcp__<server>__<tool>` -> `<server>`; None for a non-MCP name. Server
    names may contain hyphens but never a double underscore."""
    if not tool_name.startswith("mcp__"):
        return None
    rest = tool_name[len("mcp__"):]
    server, sep, _tool = rest.partition("__")
    return server if sep else None


def start_of_session(path) -> Optional[dict]:
    """
    What the transcript records at its first assistant turn:
    {"model", "input", "cache_read", "cache_write", "total", "deferred_names"}.
    `total` is the three input-side counters summed -- the whole prefix the API
    processed for that turn. `deferred_names` is the set of tool names held back
    from it, as of that turn (see _apply_delta). None when there is no assistant
    turn with usage.
    """
    deferred: set = set()
    for obj in _entries(path):
        if obj.get("type") == "attachment":
            attachment = obj.get("attachment")
            if isinstance(attachment, dict) and attachment.get("type") == "deferred_tools_delta":
                _apply_delta(deferred, attachment)
            continue
        usage = _first_usage(obj)
        if usage is None:
            continue
        try:
            inp = int(usage.get("input_tokens", 0) or 0)
            cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
            cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            # Valid JSON but a counter that is not a number (the format is internal
            # and may change): the starting context is unavailable, not a crash.
            return None
        model = obj["message"].get("model")
        if not isinstance(model, str) or not model:
            # A difference is only meaningful within one model (see compare()), so a
            # turn with no usable model identity is unavailable data, not an
            # "unknown" bucket that could silently mix different models.
            return None
        return {
            "model": model,
            "input": inp,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "total": inp + cache_read + cache_write,
            "deferred_names": deferred,
        }
    return None


def _apply_delta(names: set, attachment: dict) -> None:
    """Folds one `deferred_tools_delta` into the running deferred-name set:
    added and re-added names join it, removed names leave it."""
    for key in ("addedNames", "readdedNames"):
        value = attachment.get(key)
        if isinstance(value, list):
            names.update(n for n in value if isinstance(n, str))
    removed = attachment.get("removedNames")
    if isinstance(removed, list):
        names.difference_update(n for n in removed if isinstance(n, str))


def toolkit_tool_names(names: Iterable[str], servers: Iterable[str] = ALL_TOOLKIT_SERVERS) -> set:
    """The subset of `names` that belong to one of the toolkit's MCP servers."""
    wanted = set(servers)
    return {n for n in names if server_of(n) in wanted}


def loaded_schema_chars(path, servers: Iterable[str] = ALL_TOOLKIT_SERVERS) -> dict:
    """
    Size, in characters, of the toolkit tools' schemas that were recorded as
    loaded into this session on demand: `deferred_tools_record` entries carry
    each loaded tool's name/description/input_schema. Returns {tool_name: chars}
    (JSON length of the entry; the first record per tool wins). Empty means only
    that no on-demand load was RECORDED. It does not mean the schemas cost nothing:
    a schema sent up front leaves no such record, and its cost is inside the first
    turn's usage, where one transcript cannot size it separately.
    """
    wanted = set(servers)
    sizes: dict = {}
    for obj in _entries(path):
        if obj.get("type") != "attachment":
            continue
        attachment = obj.get("attachment")
        if not isinstance(attachment, dict) or attachment.get("type") != "deferred_tools_record":
            continue
        entries = attachment.get("entries")
        if not isinstance(entries, list):
            continue  # changed/malformed container: skip the record, never abort the scan
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and server_of(name) in wanted and name not in sizes:
                sizes[name] = len(json.dumps(entry, ensure_ascii=False))
    return sizes


def toolkit_tools_used(path, servers: Iterable[str] = ALL_TOOLKIT_SERVERS) -> set:
    """Names of toolkit tools the session actually CALLED (assistant `tool_use`
    blocks). A tool that was called but not listed as deferred at the first turn is
    AMBIGUOUS: either its schema was sent up front, or its server connected after
    the first turn. The transcript does not say which, so summarize() only flags it
    (`toolkit_used_not_deferred`) and callers must not conclude either."""
    wanted = set(servers)
    used: set = set()
    for obj in _entries(path):
        if obj.get("type") != "assistant":
            continue
        message = obj.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name")
                if isinstance(name, str) and server_of(name) in wanted:
                    used.add(name)
    return used


def summarize(path, servers: Iterable[str] = ALL_TOOLKIT_SERVERS) -> Optional[dict]:
    """One session's start-of-session facts plus how many of its tools are the
    toolkit's, and how many of those start deferred. None when there is no data."""
    start = start_of_session(path)
    if start is None:
        return None
    toolkit = toolkit_tool_names(start["deferred_names"], servers)
    used_not_deferred = toolkit_tools_used(path, servers) - toolkit
    return {
        "toolkit_used_not_deferred": sorted(used_not_deferred),
        "path": str(path),
        "model": start["model"],
        "total": start["total"],
        "input": start["input"],
        "cache_read": start["cache_read"],
        "cache_write": start["cache_write"],
        "deferred_tool_count": len(start["deferred_names"]),
        "toolkit_deferred_count": len(toolkit),
        "toolkit_servers_deferred": sorted({srv for srv in (server_of(n) for n in toolkit) if srv}),
    }


def _mean(values: list) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compare(with_paths, without_paths, servers: Iterable[str] = ALL_TOOLKIT_SERVERS) -> dict:
    """
    Per model, the mean starting context of the `with_paths` sessions (toolkit
    servers connected) against the `without_paths` sessions (removed), and the
    difference. Returns {"models": {model: {...}}, "skipped": [path, ...]}.

    Grouped by model on purpose: the system prompt and tokenizer differ between
    models, so a with-vs-without difference across two models is not a measurement
    of anything. A model present on only one side gets no difference.
    """
    sides = {"with": [], "without": []}
    skipped = []
    for label, paths in (("with", with_paths), ("without", without_paths)):
        for p in paths:
            s = summarize(p, servers)
            if s is None:
                skipped.append(str(p))
            else:
                sides[label].append(s)
    models: dict = {}
    for label, rows in sides.items():
        for row in rows:
            models.setdefault(row["model"], {"with": [], "without": []})[label].append(row)
    out: dict = {}
    for model, groups in sorted(models.items()):
        with_mean = _mean([r["total"] for r in groups["with"]])
        without_mean = _mean([r["total"] for r in groups["without"]])
        out[model] = {
            "with_count": len(groups["with"]),
            "without_count": len(groups["without"]),
            "with_mean": with_mean,
            "without_mean": without_mean,
            "difference": None if with_mean is None or without_mean is None else with_mean - without_mean,
            "with_toolkit_deferred": [r["toolkit_deferred_count"] for r in groups["with"]],
            "without_toolkit_deferred": [r["toolkit_deferred_count"] for r in groups["without"]],
        }
    return {"models": out, "skipped": skipped}


def find_transcripts(root) -> list:
    """Every main-session transcript (`*.jsonl`, not a subagent's) directly under
    each project directory of `root`, newest last."""
    base = Path(root)
    found = []
    for p in base.glob("*/*.jsonl"):
        try:
            if p.is_file():
                found.append((p.stat().st_mtime, p))
        except OSError:
            continue  # vanished (or unreadable) between listing and stat
    return [p for _mtime, p in sorted(found, key=lambda pair: pair[0])]
