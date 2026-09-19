#!/usr/bin/env python3
"""
MCP server exposing `remember`/`recall`/`forget` -- durable, cross-session
knowledge storage independent of the code index (issue #175). Where
`codebase-indexer` remembers what a repo's CODE looks like, this server
remembers what a project (or, via `general=True`, every project) has
LEARNED: a cited incident's root cause, a design decision and its rationale,
a correction a reviewer made -- the kind of thing worth surviving a
`/compact` or a fresh session, not just a single conversation.

Project minimum: Python 3.12 (policy floor; the dependency chain supports >=3.10).

Setup:
    pip install "mcp[cli]" mcp-server-qdrant qdrant-client --break-system-packages

Register in .mcp.json (see templates/mcp.json.template's "memory-bank"
block). Reuses the same QDRANT_URL/QDRANT_API_KEY/EMBEDDING_MODEL env var
names the qdrant/codebase-indexer servers already read, so one project
config keeps every Qdrant-talking server aligned -- but MEMORY_BANK_COLLECTION
is its own, separate collection name: every project shares ONE memory-bank
collection (not each project's own code-index collection), tagged by writing
project (see libs/memory_bank_lib.py's module docstring for why, and
issue #175's comments for the design history behind that choice).

IMPORTANT: this server uses stdio transport, meaning stdout is reserved for
MCP protocol messages. Nothing in this file should ever call print() --
progress and results must be returned as tool output, not printed.
"""

import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from mcp.server.mcpserver import MCPServer, Context
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from qdrant_client import QdrantClient

from qdrant_ingest_lib import ensure_persistent_fastembed_cache
from qdrant_retry import call_with_retry
from qdrant_model_check import check_embedding_model_mismatch
from mcp_tool_introspect import tool_count
import memory_bank_lib as mb

# Must run before any FastEmbedProvider(...) construction below -- see
# ingest_mcp_server.py's identical call and issue #77 for why.
ensure_persistent_fastembed_cache()

# Same env var names the qdrant/codebase-indexer servers already read, so a
# single project .mcp.json keeps every Qdrant-talking server aligned.
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# The ONE shared collection every project's memory-bank tools write into --
# deliberately NOT derived from this project's own MEMORY_BANK_ID below (see
# libs/memory_bank_lib.py's docstring).
DEFAULT_MEMORY_BANK_COLLECTION = os.environ.get("MEMORY_BANK_COLLECTION", "memory-bank")

# A plain identifier for THIS project -- doesn't have to match any real
# Qdrant collection name, used purely to resolve this project's `repo` tag
# (see memory_bank_lib.resolve_repo). setup_project.py defaults it to the
# same value as codebase-indexer's own COLLECTION_NAME for convenience, but
# they're independent settings -- nothing enforces they stay identical.
DEFAULT_MEMORY_BANK_ID = os.environ.get("MEMORY_BANK_ID")


mcp = MCPServer("memory-bank")


def _log_registered_tool_count():
    """Best-effort startup log (stderr only) -- see this repo's other MCP
    servers' identical helper for why this can't ever call print()."""
    try:
        n = tool_count(mcp)
        print(f"[claude-runway] memory-bank registered {n} tools", file=sys.stderr)
    except Exception as e:
        print(f"[claude-runway] could not count memory-bank tools: {e}", file=sys.stderr)


@mcp.tool()
async def remember(
    summary: str,
    description: str,
    kind: str,
    general: bool = False,
    collection: Optional[str] = None,
    ctx: Context = None,  # type: ignore[assignment]
) -> str:
    """
    Persist a distilled finding for future sessions -- a bug's root cause, a
    design decision and its rationale, a correction a reviewer made, or
    anything else worth surviving a fresh session. Writes into ONE shared
    collection across every project (not this project's own code index),
    tagged by this project unless general=True.

    summary: a concise summarization -- the ONLY text actually embedded and
    searched. Keep it short and phrased the way a future query would be
    phrased, not a verbatim dump.
    description: the verbatim full text of what's being remembered --
    stored as payload, returned in full on a hit, but never itself embedded.
    Provenance (a cited ticket, a file path) belongs here, not in a separate
    field.
    kind: a fully open, model-decided classification (e.g. "lesson",
    "decision", "idea", "correction") -- no fixed enum, pick whatever fits.
    general: set True ONLY when the user explicitly signals this applies
    across every project -- e.g. "remember this globally," "this applies to
    every project," "remember this across all repos." Default False (scoped
    to this project) otherwise; don't infer general=True from the content
    alone, only from an explicit signal like that.

    collection defaults to this server's configured MEMORY_BANK_COLLECTION
    env var -- only pass this explicitly to target a different collection.
    """
    collection = collection or DEFAULT_MEMORY_BANK_COLLECTION
    repo, error = mb.resolve_repo(DEFAULT_MEMORY_BANK_ID, general)
    if error:
        return error

    client = QdrantClient(url=DEFAULT_QDRANT_URL, api_key=DEFAULT_QDRANT_API_KEY)
    embedding_provider = FastEmbedProvider(DEFAULT_EMBEDDING_MODEL)

    # No separate top-level mismatch check here (PR #178 review): ensure_collection
    # (called by remember_point below) now does an equivalent check itself, positioned
    # right before the actual write rather than earlier -- checking here too would just
    # be a redundant round-trip with a WIDER race window, not extra safety.
    point_id, mismatch = await mb.remember_point(
        client, embedding_provider, collection,
        summary=summary, description=description, kind=kind,
        repo=repo, embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    if mismatch:
        return mismatch
    return f"Remembered (id={point_id}, repo='{repo}', kind='{kind}')."


@mcp.tool()
async def recall(
    query: str,
    kind: Optional[str] = None,
    repo: Optional[str] = None,
    all_repos: bool = False,
    collection: Optional[str] = None,
    limit: int = 5,
    ctx: Context = None,  # type: ignore[assignment]
) -> str:
    """
    Semantically search stored memories. Defaults to THIS project's own
    memories plus general (cross-project) ones -- the same precision a
    per-project store would give, without missing anything explicitly
    tagged as applying everywhere.

    Pass repo=<some other project's MEMORY_BANK_ID> to look at a
    SPECIFIC other project's memories instead (e.g. the user names it
    directly), or all_repos=True to search across every project at once.
    Don't guess a repo name -- only pass one the user actually named, or
    call list_collections-equivalent context if unsure (there is no
    discovery tool for repo names here by design -- see issue #175 for why
    automatic cross-repo discovery was rejected).

    kind narrows further if given. limit caps how many hits come back.
    """
    collection = collection or DEFAULT_MEMORY_BANK_COLLECTION
    caller_repo, error = mb.resolve_repo(DEFAULT_MEMORY_BANK_ID, general=False)
    if error and not (repo or all_repos):
        # A caller with no valid own repo can still explicitly search a
        # NAMED other repo or all_repos -- only the (repo=None,
        # all_repos=False) default scope actually needs caller_repo resolved.
        return error
    if limit < 1:
        return f"Error: limit must be 1 or greater (got {limit})."

    client = QdrantClient(url=DEFAULT_QDRANT_URL, api_key=DEFAULT_QDRANT_API_KEY)

    if not call_with_retry(client.collection_exists, collection):
        # Return immediately rather than falling through to recall_points,
        # which does its OWN separate collection_exists check right before
        # querying (PR #178 review): a concurrent remember() could create the
        # collection under a DIFFERENT embedding model in the gap between
        # this check and that one, making recall_points proceed past ITS
        # check and call query_points with the wrong vector name -- an
        # opaque Qdrant error instead of this function's own clean mismatch
        # message. A collection that doesn't exist yet has no memories for
        # this caller regardless, so returning here is exactly what
        # recall_points would have reported anyway, without reopening that
        # window. Checked BEFORE constructing FastEmbedProvider below (PR
        # #178 review, second pass): that constructor eagerly loads/downloads
        # the ONNX model (TextEmbedding's default lazy_load=False), so
        # recalling against a not-yet-existing collection would otherwise pay
        # for a model load it never needed just to return "no results."
        return f"No memories found for '{query}'."

    embedding_provider = FastEmbedProvider(DEFAULT_EMBEDDING_MODEL)
    mismatch = check_embedding_model_mismatch(
        client, collection, embedding_provider, recovery_hint=mb.MISMATCH_RECOVERY_HINT
    )
    if mismatch:
        return mismatch

    results = await mb.recall_points(
        client, embedding_provider, collection, query,
        caller_repo=caller_repo or "", repo=repo, all_repos=all_repos, kind=kind, limit=limit,
    )
    if not results:
        return f"No memories found for '{query}'."

    # JSON, not hand-rolled XML-ish text (PR #178 review, eighth pass):
    # summary/description/kind are arbitrary stored text (see
    # memory_bank_lib.py's module docstring -- kind is fully model-decided,
    # summary/description are never sanitized), so interpolating them into
    # unescaped tags let one memory's content break record boundaries or
    # spoof a fake <memory> block. json.dumps escapes every string it's
    # given, so this isn't a per-field patch job -- the whole class of
    # "what if this text contains a delimiter" bugs doesn't apply to a
    # properly-serialized format.
    return json.dumps({"query": query, "results": results}, indent=2)


@mcp.tool()
def forget(
    point_id: Optional[str] = None,
    wipe_all: bool = False,
    confirm: bool = False,
    collection: Optional[str] = None,
) -> str:
    """
    Delete memories. Two mutually exclusive modes -- passing both or
    neither of point_id/wipe_all is an error:

    point_id: delete exactly one memory by exact id (e.g. one returned by a
    prior recall call). Deleting a memory that belongs to a DIFFERENT repo
    than this project's own (this includes a general/cross-project memory)
    requires confirm=True -- always confirm with the user before passing
    confirm=True here, don't set it automatically.

    wipe_all: bulk-clear ALL of THIS project's own memories (never general/
    cross-project ones -- those can only be removed individually via
    point_id). A call without confirm=True only returns a count of what
    would be deleted; pass confirm=True to actually delete. Always show the
    count to the user and get explicit confirmation before re-calling with
    confirm=True.
    """
    if bool(point_id) == bool(wipe_all):
        return "Error: pass exactly one of point_id or wipe_all, not both or neither."

    collection = collection or DEFAULT_MEMORY_BANK_COLLECTION
    caller_repo, error = mb.resolve_repo(DEFAULT_MEMORY_BANK_ID, general=False)
    if error:
        return error

    client = QdrantClient(url=DEFAULT_QDRANT_URL, api_key=DEFAULT_QDRANT_API_KEY)

    if point_id:
        return mb.forget_point(client, collection, point_id, caller_repo=caller_repo, confirm=confirm)

    count, deleted = mb.wipe_memory_bank(client, collection, caller_repo=caller_repo, confirm=confirm)
    if not deleted:
        return (
            f"{count} memor{'y' if count == 1 else 'ies'} belonging to repo '{caller_repo}' would be deleted. "
            f"Pass confirm=True to actually delete them."
        )
    return (
        f"Deleted {count} memor{'y' if count == 1 else 'ies'} belonging to repo '{caller_repo}' "
        f"(count reflects what this call found and targeted, not a separately-verified post-delete total)."
    )


if __name__ == "__main__":
    _log_registered_tool_count()
    mcp.run()
