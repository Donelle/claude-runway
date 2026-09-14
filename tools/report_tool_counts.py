#!/usr/bin/env python3
"""
Prints the LIVE `@mcp.tool()` count AND an estimated schema-token cost for
every MCP server this repo's users connect (this repo's own
`compress_mcp_server.py`/`ingest_mcp_server.py`, PLUS the standalone
third-party `mcp-server-qdrant` package's server) -- instead of a number
hand-copied into EVALUATION.md that goes stale as tools are added. See
libs/mcp_tool_introspect.py's docstring for the history of that exact
failure mode (issues #22/#23), and issue #52/GROW-05 for why one shared,
reusable utility was built rather than a third place to fix the same
number by hand.

Usage:
    python tools/report_tool_counts.py

Run this instead of trusting (or re-deriving by hand) any hardcoded tool
count or token estimate in EVALUATION.md's Track A/B "measure the fixed
overhead" steps -- that's exactly what those sections now point at. Also
used as a CI smoke check (see .github/workflows/tests.yml): a nonzero
exit code here means one of the three servers failed to construct or
enumerate its own tools, which is a real regression worth failing the
build on.

Reports a token estimate, not just a count, because a count alone can't
substitute for Step A4/B4's actual requirement (PR #120 review, issue
#52): tool schemas differ in size, so "N tools" doesn't tell you how many
tokens that costs per turn the way `schema_overhead_tokens` does. Uses
the exact same `describe_tools()` blob + `estimate_tokens()` approach
`compress_mcp_server.py`'s own `_log_fixed_overhead()` uses, so this
script's number and that one can never disagree.

The standalone `mcp-server-qdrant` package's server is included here too
(not just this repo's own two servers) because that package is UNPINNED
in requirements.txt -- its own tool count ("qdrant-find", "qdrant-store",
currently 2) is not a repo-controlled invariant the way a hardcoded "2"
in EVALUATION.md's prose implied. Constructing `QdrantMCPServer` directly
(patching out the embedding provider factory so no real model loads,
and using default `QdrantSettings()`/`ToolSettings()` so no live Qdrant
connection is required) is the only way to check this live rather than
assuming it never changes -- same "verify, don't assume" standard the
rest of this fix applies to this repo's own two servers. This does couple
to `mcp_server_qdrant`'s internal `QdrantMCPServer`/`ToolSettings`/
`QdrantSettings` classes, which aren't a documented public API -- but
`ingest_mcp_server.py` already couples to that same package's internals
elsewhere (`QdrantConnector`, `FastEmbedProvider`), so this isn't a new
class of risk for this repo, just the same one applied one level deeper.
If a future `mcp-server-qdrant` release renames or restructures these,
this check fails loudly (nonzero exit, clear stderr message) rather than
silently reporting a stale number -- which is the whole point.

Deliberately never calls any server's own `mcp.run()`/`.run()` (or
connects to a live Qdrant/LM Studio) -- this never starts a real stdio
server and never blocks on stdin, so it's safe to run from a plain
terminal or CI runner with no other infrastructure up.

Note: importing `compress_mcp_server` requires `openai`/`requests`/
`trafilatura` (see requirements.txt); importing `ingest_mcp_server` and
constructing the standalone qdrant server both require
`mcp-server-qdrant`/`qdrant-client`. All are already required for the CI
job's `pip install -r requirements.txt` step to succeed at all, so no
extra dependency is introduced here.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
LIBS_DIR = os.path.join(REPO_ROOT, "libs")

sys.path.insert(0, LIBS_DIR)
sys.path.insert(0, TOOLS_DIR)

from mcp_tool_introspect import describe_tools, tool_count  # noqa: E402


def _report_repo_server(module_name: str) -> tuple:
    """
    Imports one of THIS repo's own tool modules and returns
    (tool_count, estimated_schema_tokens), or raises -- callers decide
    whether a failure here should be fatal (main() below treats it as
    fatal, since a broken import means the numbers can't be trusted at
    all, which is worse than an unknown number).

    ingest_mcp_server.py needs a few env vars set before import (its
    module-level DEFAULT_* constants read them, though all default
    gracefully to None/empty) -- same minimal env test_qdrant_ingest_lib.py
    already sets for the identical reason, so this mirrors an already-
    established pattern rather than inventing a new one.
    """
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    os.environ.setdefault("COLLECTION_NAME", "report-tool-counts-placeholder")
    os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    module = __import__(module_name)
    # estimate_tokens lives in local_compress_lib, which compress_mcp_server
    # already forces this process to import -- reusing it here (rather than
    # duplicating the char/3.5 ratio a third time) keeps this number backed
    # by the exact same formula _log_fixed_overhead() uses.
    from local_compress_lib import estimate_tokens

    mcp = module.mcp
    return tool_count(mcp), estimate_tokens(describe_tools(mcp))


def _report_standalone_qdrant_server() -> tuple:
    """
    Constructs the third-party `mcp-server-qdrant` package's own server
    directly (never via a live Qdrant/LM Studio connection -- see this
    module's docstring for why this is safe) and returns
    (tool_count, estimated_schema_tokens), or raises.

    `QdrantMCPServer` is a subclass of the standalone `fastmcp.FastMCP`
    (not this repo's own `mcp.server.mcpserver.MCPServer`), so it does not
    expose the `_tool_manager` attribute that `mcp_tool_introspect` reaches
    into. Use the async `list_tools()` coroutine instead, which is the
    public API across all versions of the standalone fastmcp package.

    `FunctionTool` objects from `list_tools()` expose a `.fn` attribute
    (the underlying Python function) the same way `mcp_tool_introspect`
    does for `MCPServer` instances, so the schema-token blob is built in
    the same format (name + signature + docstring) for a consistent
    three-way comparison.

    The embedding provider factory is patched out for this introspection-
    only call to prevent `FastEmbedProvider.__init__` from triggering a
    real model download -- the provider is stored on `QdrantConnector` and
    only invoked when an actual find/store tool call happens, which this
    script never makes.
    """
    import asyncio
    import inspect
    from unittest.mock import patch

    from local_compress_lib import estimate_tokens
    from mcp_server_qdrant.mcp_server import QdrantMCPServer
    from mcp_server_qdrant.settings import EmbeddingProviderSettings, QdrantSettings, ToolSettings

    with patch("mcp_server_qdrant.mcp_server.create_embedding_provider", return_value=object()):
        srv = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider_settings=EmbeddingProviderSettings(),
        )
    tools = asyncio.run(srv.list_tools())
    # FunctionTool objects expose .fn (the underlying Python function) the same
    # way mcp_tool_introspect.list_registered_tools() does for our MCPServer
    # instances, so build the describe_tools()-equivalent blob in the same
    # format (name + signature + docstring) so the three reported estimates are
    # computed from comparable subsets.
    blob = "\n".join(
        f"{t.name}{inspect.signature(t.fn)}\n{t.fn.__doc__ or ''}"
        for t in tools
    )
    return len(tools), estimate_tokens(blob)


def main() -> int:
    exit_code = 0

    for label, module_name in (
        ("local-compress", "compress_mcp_server"),
        ("codebase-indexer", "ingest_mcp_server"),
    ):
        try:
            count, tokens = _report_repo_server(module_name)
            print(f"{label} ({module_name}.py): {count} tools, ~{tokens} schema tokens")
        except Exception as e:
            print(f"{label} ({module_name}.py): FAILED to enumerate tools -- {e}", file=sys.stderr)
            exit_code = 1

    try:
        count, tokens = _report_standalone_qdrant_server()
        print(f"qdrant (standalone mcp-server-qdrant package): {count} tools, ~{tokens} schema tokens")
    except Exception as e:
        print(f"qdrant (standalone mcp-server-qdrant package): FAILED to enumerate tools -- {e}", file=sys.stderr)
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
