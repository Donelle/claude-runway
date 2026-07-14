#!/usr/bin/env python3
"""
Bulk-index a repo's code and docs into the same Qdrant collection your
mcp-server-qdrant MCP server reads from, using the *same* embedding
provider the server uses (so vectors are compatible with qdrant-find).

Usage:
    pip install mcp-server-qdrant

    python ingest_to_qdrant.py \
        --repo-path /path/to/your/repo \
        --collection my-collection \
        --qdrant-url http://localhost:6333 \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2

Run this once to seed the collection, then re-run periodically (or on a
schedule) to pick up changes. Re-running without --reset will add duplicate
chunks; pass --reset to wipe the collection before indexing.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs"))

try:
    from mcp_server_qdrant.qdrant import QdrantConnector, Entry
    from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
    from qdrant_client import QdrantClient
except ImportError:
    print(
        "Missing dependency. Install with:\n"
        "    pip install mcp-server-qdrant qdrant-client --break-system-packages\n",
        file=sys.stderr,
    )
    sys.exit(1)

from qdrant_ingest_lib import build_entries


async def ingest(args):
    if args.reset:
        client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
        if client.collection_exists(args.collection):
            client.delete_collection(args.collection)
            print(f"Reset: deleted existing collection '{args.collection}'.")

    embedding_provider = FastEmbedProvider(args.embedding_model)
    connector = QdrantConnector(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        embedding_provider=embedding_provider,
    )

    repo_path = Path(args.repo_path).resolve()
    include_extensions = set(args.include_ext.split(",")) if args.include_ext else None
    extra_exclude_dirs = set(args.exclude_dirs.split(",")) if args.exclude_dirs else None
    raw_entries = build_entries(
        repo_path, args.scope, args.chunk_lines, args.overlap,
        include_extensions=include_extensions,
        extra_exclude_dirs=extra_exclude_dirs,
        respect_gitignore=not args.no_gitignore,
    )
    entries = [Entry(content=content, metadata=metadata) for content, metadata in raw_entries]

    print(f"Found {len(entries)} chunks to index from {repo_path}")
    if args.dry_run:
        for e in entries[:10]:
            print("---")
            print(e.metadata)
            print(e.content[:200])
        print(f"... dry run, not storing. ({len(entries)} total)")
        return

    for i, entry in enumerate(entries, 1):
        await connector.store(entry, collection_name=args.collection)
        if i % 25 == 0 or i == len(entries):
            print(f"  stored {i}/{len(entries)}")

    print("Done.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-path", required=True, help="Path to the repo to index")
    p.add_argument("--collection", required=True, help="Qdrant collection name (must match your MCP server's COLLECTION_NAME)")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--qdrant-api-key", default=None)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Must match your MCP server's EMBEDDING_MODEL exactly")
    p.add_argument("--scope", choices=["code", "docs", "both"], default="both",
                    help="Ignored if --include-ext is given")
    p.add_argument("--include-ext", default=None,
                    help="Comma-separated file extensions to ingest, overriding --scope entirely, "
                         "e.g. '.py,.md,.proto' (leading dot optional)")
    p.add_argument("--exclude-dirs", default=None,
                    help="Comma-separated folder names to skip, IN ADDITION TO the built-in defaults "
                         "(.git, node_modules, venv, dist, build, etc.) -- e.g. 'fixtures,generated'")
    p.add_argument("--no-gitignore", action="store_true",
                    help="Don't apply the repo's own .gitignore on top of the exclude rules above "
                         "(gitignore is respected by default if the optional 'pathspec' package is installed)")
    p.add_argument("--chunk-lines", type=int, default=50)
    p.add_argument("--overlap", type=int, default=10)
    p.add_argument("--dry-run", action="store_true", help="Preview chunks without writing to Qdrant")
    p.add_argument("--reset", action="store_true", help="Delete the collection before indexing (avoids duplicate chunks on re-runs)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(ingest(parse_args()))
