#!/usr/bin/env python3
"""
Shared, dependency-free introspection over an MCPServer's registered
`@mcp.tool()` functions -- so any caller that needs "how many tools does
this server expose right now, and roughly how big are their schemas" gets
a LIVE answer instead of a hardcoded number that quietly goes stale as
tools are added over time.

This exists because that exact drift already happened twice on this repo:

- Issue #22 (BUG-02): `compress_mcp_server.py`'s `_log_fixed_overhead()`
  hardcoded a 5-tool tuple. `compact_store`/`compact_find`/
  `savings_summary`/`savings_detail` shipped later and were never added to
  it -- the /my-savings "tool overhead" annotation silently understated
  the real cost for months, with no error to notice.
- Issue #23 (BUG-03): EVALUATION.md's Track A/B prose separately hardcoded
  "6 total" and "5 total" tool counts that drifted the same way.

Issue #22's fix replaced compress_mcp_server.py's own hardcoded tuple with
an inline enumeration of `mcp._tool_manager.list_tools()`. This module
extracts that same logic into one shared place (issue #52/GROW-05) so
every caller that needs a live tool count -- `compress_mcp_server.py`,
`ingest_mcp_server.py`, `tools/report_tool_counts.py`, and EVALUATION.md's
own "how to check this" prose -- goes through the identical code path and
can't drift relative to each other.

Deliberately STDLIB-ONLY (no third-party imports, not even
`local_compress_lib.estimate_tokens`) so importing this from
`ingest_mcp_server.py`'s side doesn't drag in `openai` for a project that
only wants the Qdrant memory piece -- same reasoning `libs/doctor_lib.py`'s
own docstring already gives for not importing `local_compress_lib.py`
directly.
"""

import inspect
from typing import Any, Callable, List


def list_registered_tools(mcp: Any) -> List[Callable]:
    """
    Returns every function currently registered as an `@mcp.tool()` on an
    MCPServer instance, in registration order.

    Reaches into MCPServer's private `_tool_manager` attribute rather than
    the public `list_tools()` coroutine -- the public method is async, and
    this needs to run at plain module/startup scope (before `mcp.run()`
    starts an event loop), the same synchronous-context constraint
    `compress_mcp_server.py`'s original fix already worked around this
    way. Same private-attribute tradeoff already accepted in
    `qdrant_batch_store.py` for the identical reason: no public
    synchronous alternative exists.
    """
    return [info.fn for info in mcp._tool_manager.list_tools()]


def describe_tools(mcp: Any) -> str:
    """
    Name + signature + docstring for every registered tool, concatenated
    into one blob -- a proxy for the real JSON tool schema Claude Code
    actually sends (MCPServer doesn't expose the real schema synchronously
    either, so this is the same approximation `_log_fixed_overhead()`
    already used before this extraction).
    """
    return "\n".join(
        f"{fn.__name__}{inspect.signature(fn)}\n{fn.__doc__ or ''}"
        for fn in list_registered_tools(mcp)
    )


def tool_count(mcp: Any) -> int:
    """Convenience for callers that only need the count, not the blob."""
    return len(list_registered_tools(mcp))
