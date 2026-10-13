#!/usr/bin/env python3
"""
Prints the LIVE `@mcp.tool()` count AND the size, in characters, of each
tool's description text (name + signature + docstring, a proxy for the JSON
tool schema -- see below) for every MCP server this repo's users connect (this repo's own
`qdrant_mcp_server.py`/`compress_mcp_server.py`/`ingest_mcp_server.py`/
`memory_bank_mcp_server.py`) -- instead of a number
hand-copied into EVALUATION.md that goes stale as tools are added. See
libs/mcp_tool_introspect.py's docstring for the history of that exact
failure mode (issues #22/#23), and issue #52/GROW-05 for why one shared,
reusable utility was built rather than a third place to fix the same
number by hand.

Usage:
    python tools/report_tool_counts.py

Run this instead of trusting (or re-deriving by hand) any hardcoded tool
count in EVALUATION.md's Track A/B "measure the fixed overhead" steps --
that's exactly what those sections now point at. For what the toolkit really
adds to a session's starting context, use tools/measure_startup_context.py,
which reads it from session transcripts; this script only reports how many
tools there are and how big their descriptions are. Also
used as a CI smoke check (see .github/workflows/tests.yml): a nonzero
exit code here means one of the four servers failed to construct or
enumerate its own tools, which is a real regression worth failing the
build on.

Reports a size, not just a count, because a count alone says little (PR
#120 review, issue #52): tool schemas differ in size. The size is in
CHARACTERS, not tokens (issue #439): a token count depends on the model's
vocabulary, which is not public, and the chars / 3.5 estimate this script used
to print was presented as a per-turn cost that real transcripts show is not
paid when Claude Code defers these tools. The blob measured is the
`describe_tools()` one (name + signature + docstring), a proxy for the real
JSON schema, so it is a relative size, not what the API receives.

Deliberately never calls any server's own `mcp.run()`/`.run()` (or
connects to a live Qdrant/LM Studio) -- this never starts a real stdio
server and never blocks on stdin, so it's safe to run from a plain
terminal or CI runner with no other infrastructure up.

Note: importing `compress_mcp_server` requires `openai`/`requests`/
`trafilatura` (see requirements.txt); importing `qdrant_mcp_server`,
`ingest_mcp_server` and `memory_bank_mcp_server` requires `qdrant-client`
and `fastembed`. All are already required for the CI job's
`pip install -r requirements.txt` step to succeed at all, so no extra
dependency is introduced here. Importing `qdrant_mcp_server` loads no
embedding model (it is built on the first `qdrant-find`/`qdrant-store`
call), so this stays network-free.
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
    (tool_count, schema_chars), or raises -- callers decide
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
    mcp = module.mcp
    return tool_count(mcp), len(describe_tools(mcp))


def main() -> int:
    exit_code = 0

    for label, module_name in (
        ("local-compress", "compress_mcp_server"),
        ("qdrant", "qdrant_mcp_server"),
        ("codebase-indexer", "ingest_mcp_server"),
        ("memory-bank", "memory_bank_mcp_server"),
    ):
        try:
            count, chars = _report_repo_server(module_name)
            print(f"{label} ({module_name}.py): {count} tools, {chars:,} tool-description characters (schema proxy)")
        except Exception as e:
            print(f"{label} ({module_name}.py): FAILED to enumerate tools -- {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
