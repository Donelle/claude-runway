#!/usr/bin/env python3
"""
MCP server exposing repo indexing as callable tools, so Claude (in Claude
Code) can trigger re-indexing on demand -- e.g. right after cloning a repo,
or when it notices qdrant-find is coming back empty -- instead of you
running ingest_to_qdrant.py by hand every time.

Setup:
    pip install "mcp[cli]" mcp-server-qdrant qdrant-client --break-system-packages

Register in .mcp.json (adjust the path to wherever you save this file). Set
QDRANT_URL / COLLECTION_NAME / EMBEDDING_MODEL here -- these are the SAME env
var names mcp-server-qdrant itself reads, so one project config drives both
servers and they can't drift apart:
{
  "mcpServers": {
    "codebase-indexer": {
      "command": "python",
      "args": ["/absolute/path/to/ingest_mcp_server.py"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "<this-project's-collection-name>",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
      }
    }
  }
}

With COLLECTION_NAME set per-project this way, every tool call below defaults
to that project's collection automatically -- you don't pass collection on
every call, and opening a different project (different .mcp.json) uses a
different collection with no manual switching.

IMPORTANT: this server uses stdio transport, meaning stdout is reserved for
MCP protocol messages. Nothing in this file should ever call print() --
progress and results must be returned as tool output, not printed. (This is
also why this server is a separate file from ingest_to_qdrant.py, which
prints freely as a normal CLI script.)
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs"))

from mcp.server.fastmcp import FastMCP, Context
from mcp_server_qdrant.qdrant import QdrantConnector, Entry
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from qdrant_client import QdrantClient, models

from qdrant_ingest_lib import build_entries, chunk_file, compute_file_hashes

MANIFEST_FILENAME = ".qdrant_index_manifest.json"


def _load_manifest(repo: Path) -> dict:
    manifest_path = repo / MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_manifest(repo: Path, manifest: dict) -> None:
    (repo / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))

# Same env var names mcp-server-qdrant reads, so a single project .mcp.json
# config keeps this server and the qdrant-find/qdrant-store server aligned.
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
DEFAULT_COLLECTION = os.environ.get("COLLECTION_NAME")
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _parse_csv_set(value: Optional[str]) -> Optional[set]:
    return {v.strip() for v in value.split(",") if v.strip()} if value else None


# Project-specific file filtering, settable once via .mcp.json so every tool
# call picks it up without repeating it per call.
DEFAULT_INCLUDE_EXTENSIONS = _parse_csv_set(os.environ.get("INDEX_INCLUDE_EXTENSIONS"))
DEFAULT_EXTRA_EXCLUDE_DIRS = _parse_csv_set(os.environ.get("INDEX_EXCLUDE_DIRS"))

mcp = FastMCP("codebase-indexer")


@mcp.tool()
async def index_repo(
    repo_path: str,
    collection: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    scope: str = "both",
    include_extensions: Optional[str] = None,
    exclude_dirs: Optional[str] = None,
    respect_gitignore: bool = True,
    chunk_lines: int = 50,
    overlap: int = 10,
    reset: bool = False,
    ctx: Context = None,
) -> str:
    """
    Bulk-index a repo's code and docs into a Qdrant collection so qdrant-find
    can retrieve them later. Use this when: a collection is missing or empty
    and qdrant-find is returning nothing useful, a repo was just cloned and
    hasn't been indexed yet, or the code has changed enough that the
    existing index is stale.

    collection/qdrant_url/embedding_model default to this project's
    COLLECTION_NAME/QDRANT_URL/EMBEDDING_MODEL env vars if not given -- only
    pass these explicitly to target a different project's collection.

    include_extensions/exclude_dirs are comma-separated strings (e.g.
    "'.py,.md'" / "'fixtures,generated'") and default to this project's
    INDEX_INCLUDE_EXTENSIONS/INDEX_EXCLUDE_DIRS env vars if not given.
    include_extensions overrides scope entirely when set. exclude_dirs adds
    to (not replaces) the built-in defaults (.git, node_modules, dist, etc).
    respect_gitignore additionally skips whatever the repo's own .gitignore
    excludes (silently ignored if there's no .gitignore).

    embedding_model MUST match the EMBEDDING_MODEL configured on the
    qdrant-find/qdrant-store MCP server, or search relevance will silently
    break (different models produce incompatible vector spaces).

    Set reset=true to delete all existing entries in the collection first --
    otherwise re-running this on an already-indexed repo creates duplicate
    chunks. Do NOT set reset=true if the collection is shared with other
    data (e.g. conversation-memory summaries) you don't want wiped.

    This can take a while on large repos since every chunk is embedded
    locally. Reports progress via MCP progress notifications as it runs (not
    all clients render these visibly, but it keeps the call from timing out
    on large repos). Returns a summary once indexing completes. For a big
    first-time index where you want to watch progress in a terminal, prefer
    running ingest_to_qdrant.py directly instead of this tool.
    """
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        return f"Error: {repo} is not a directory."

    collection = collection or DEFAULT_COLLECTION
    if not collection:
        return (
            "Error: no collection specified and no COLLECTION_NAME env var "
            "configured for this project's .mcp.json. Pass collection explicitly "
            "or set COLLECTION_NAME in this server's env block."
        )
    qdrant_url = qdrant_url or DEFAULT_QDRANT_URL
    qdrant_api_key = qdrant_api_key or DEFAULT_QDRANT_API_KEY
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    if reset:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        if client.collection_exists(collection):
            client.delete_collection(collection)

    embedding_provider = FastEmbedProvider(embedding_model)
    connector = QdrantConnector(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection,
        embedding_provider=embedding_provider,
    )

    raw_entries = build_entries(
        repo, scope, chunk_lines, overlap,
        include_extensions=_parse_csv_set(include_extensions) or DEFAULT_INCLUDE_EXTENSIONS,
        extra_exclude_dirs=_parse_csv_set(exclude_dirs) or DEFAULT_EXTRA_EXCLUDE_DIRS,
        respect_gitignore=respect_gitignore,
    )
    total = len(raw_entries)
    for i, (content, metadata) in enumerate(raw_entries, 1):
        await connector.store(Entry(content=content, metadata=metadata), collection_name=collection)
        if ctx is not None and (i % 10 == 0 or i == total):
            await ctx.report_progress(progress=i, total=total, message=f"Indexed {i}/{total} chunks")

    code_count = sum(1 for _, m in raw_entries if m["type"] == "code")
    doc_count = sum(1 for _, m in raw_entries if m["type"] == "doc")
    return (
        f"Indexed {len(raw_entries)} chunks from {repo} into collection '{collection}' "
        f"({code_count} code chunks, {doc_count} doc chunks). "
        f"{'Collection was reset before indexing.' if reset else 'Existing entries were kept -- rerun with reset=true if you suspect duplicates.'}"
    )


@mcp.tool()
async def sync_repo(
    repo_path: str,
    collection: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    scope: str = "both",
    include_extensions: Optional[str] = None,
    exclude_dirs: Optional[str] = None,
    respect_gitignore: bool = True,
    chunk_lines: int = 50,
    overlap: int = 10,
    ctx: Context = None,
) -> str:
    """
    Incrementally update a Qdrant collection to match the repo's current
    state -- only re-embeds files that changed since the last sync, deletes
    entries for files that were removed, and skips unchanged files entirely.

    include_extensions/exclude_dirs/respect_gitignore work the same as in
    index_repo and default to the same INDEX_INCLUDE_EXTENSIONS/
    INDEX_EXCLUDE_DIRS env vars. IMPORTANT: keep these consistent between
    calls for the same repo -- changing them changes which files "exist" as
    far as the manifest is concerned, which can make previously-indexed
    files look removed or vice versa.

    Use this for routine re-indexing (e.g. after editing a handful of files,
    or as a habit at the start of a session) -- it's far cheaper than
    index_repo since it doesn't touch or re-embed anything unchanged. Use
    index_repo with reset=true instead when: this is the first time indexing
    this repo, chunk_lines/overlap/embedding_model changed and everything
    needs re-chunking consistently, or the index and manifest seem to have
    drifted (e.g. after manually editing points in Qdrant).

    Tracks a small manifest file (.qdrant_index_manifest.json) in the repo
    root to know what's already indexed -- safe to gitignore, don't hand-edit it.

    collection/qdrant_url/embedding_model default to this project's
    COLLECTION_NAME/QDRANT_URL/EMBEDDING_MODEL env vars if not given.

    Reports progress via MCP progress notifications per changed file.
    """
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        return f"Error: {repo} is not a directory."

    collection = collection or DEFAULT_COLLECTION
    if not collection:
        return (
            "Error: no collection specified and no COLLECTION_NAME env var "
            "configured for this project's .mcp.json."
        )
    qdrant_url = qdrant_url or DEFAULT_QDRANT_URL
    qdrant_api_key = qdrant_api_key or DEFAULT_QDRANT_API_KEY
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    resolved_include_extensions = _parse_csv_set(include_extensions) or DEFAULT_INCLUDE_EXTENSIONS
    resolved_exclude_dirs = _parse_csv_set(exclude_dirs) or DEFAULT_EXTRA_EXCLUDE_DIRS

    manifest = _load_manifest(repo)
    current_hashes = compute_file_hashes(
        repo, scope,
        include_extensions=resolved_include_extensions,
        extra_exclude_dirs=resolved_exclude_dirs,
        respect_gitignore=respect_gitignore,
    )

    changed = [f for f, h in current_hashes.items() if manifest.get(f) != h]
    removed = [f for f in manifest if f not in current_hashes]
    unchanged_count = len(current_hashes) - len(changed)

    if not changed and not removed:
        return f"No changes since last sync. {unchanged_count} files already up to date in '{collection}'."

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    if client.collection_exists(collection):
        for rel in changed + removed:
            client.delete(
                collection_name=collection,
                points_selector=models.Filter(
                    must=[models.FieldCondition(key="metadata.file_path", match=models.MatchValue(value=rel))]
                ),
            )

    embedding_provider = FastEmbedProvider(embedding_model)
    connector = QdrantConnector(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection,
        embedding_provider=embedding_provider,
    )

    new_chunk_count = 0
    for i, rel in enumerate(changed, 1):
        for content, metadata in chunk_file(repo / rel, rel, chunk_lines, overlap):
            await connector.store(Entry(content=content, metadata=metadata), collection_name=collection)
            new_chunk_count += 1
        if ctx is not None:
            await ctx.report_progress(progress=i, total=len(changed), message=f"Re-indexed {i}/{len(changed)} changed files")

    _save_manifest(repo, current_hashes)

    return (
        f"Synced '{collection}': {len(changed)} file(s) re-indexed ({new_chunk_count} chunks), "
        f"{len(removed)} file(s) removed, {unchanged_count} file(s) unchanged and skipped."
    )


@mcp.tool()
def preview_index(
    repo_path: str,
    scope: str = "both",
    include_extensions: Optional[str] = None,
    exclude_dirs: Optional[str] = None,
    respect_gitignore: bool = True,
    chunk_lines: int = 50,
    overlap: int = 10,
    limit: int = 10,
) -> str:
    """
    Preview how a repo would be chunked WITHOUT writing anything to Qdrant.
    Use this before index_repo on a repo you haven't indexed before, to
    sanity-check chunk boundaries and metadata (especially file_path and
    line_range) -- and to sanity-check include_extensions/exclude_dirs are
    actually filtering the way you expect -- before committing to a real
    index run.
    """
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        return f"Error: {repo} is not a directory."

    entries = build_entries(
        repo, scope, chunk_lines, overlap,
        include_extensions=_parse_csv_set(include_extensions) or DEFAULT_INCLUDE_EXTENSIONS,
        extra_exclude_dirs=_parse_csv_set(exclude_dirs) or DEFAULT_EXTRA_EXCLUDE_DIRS,
        respect_gitignore=respect_gitignore,
    )
    lines = [f"{len(entries)} total chunks would be created. Showing first {min(limit, len(entries))}:\n"]
    for content, metadata in entries[:limit]:
        preview = content[:200].replace("\n", " ")
        lines.append(f"[{metadata['type']}] {metadata['file_path']} ({metadata['line_range']}): {preview}...")
    return "\n".join(lines)


@mcp.tool()
def get_collection_info(
    collection: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> str:
    """
    Check whether a Qdrant collection exists and how many points it holds.
    Use this to confirm a previous index_repo call actually wrote data, or
    to check whether a collection is empty before relying on qdrant-find
    against it.

    collection/qdrant_url default to this project's COLLECTION_NAME/QDRANT_URL
    env vars if not given.
    """
    collection = collection or DEFAULT_COLLECTION
    if not collection:
        return "Error: no collection specified and no COLLECTION_NAME env var configured for this project."
    qdrant_url = qdrant_url or DEFAULT_QDRANT_URL
    qdrant_api_key = qdrant_api_key or DEFAULT_QDRANT_API_KEY

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    if not client.collection_exists(collection):
        return f"Collection '{collection}' does not exist yet."
    info = client.get_collection(collection)
    return f"Collection '{collection}': {info.points_count} points, status={info.status}."


if __name__ == "__main__":
    mcp.run()
