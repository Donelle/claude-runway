#!/usr/bin/env python3
"""
Bulk-index a repo's code and docs into the same Qdrant collection your
mcp-server-qdrant MCP server reads from, using the *same* embedding
provider the server uses (so vectors are compatible with qdrant-find).

Usage:
    pip install mcp-server-qdrant

    python tools/ingest_to_qdrant.py \
        --repo-path /path/to/your/repo \
        --collection my-collection \
        --qdrant-url http://localhost:6333 \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2

Run this once to seed the collection, then re-run periodically (or on a
schedule) to pick up changes. Re-running without --reset will add duplicate
chunks; pass --reset to wipe the collection's (non-memory-bank) points before indexing.

Project minimum: Python 3.12 (policy floor; the dependency chain supports >=3.10).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

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

from qdrant_ingest_lib import build_entries, ensure_persistent_fastembed_cache, validate_chunk_params
from qdrant_batch_store import store_batch, ensure_file_path_index, FIELD_INDEXES
from qdrant_model_check import check_embedding_model_mismatch
from qdrant_retry import call_with_retry
import memory_bank_lib as mb

# Must run before FastEmbedProvider(...) is constructed in ingest() below --
# see issue #77 and the function's own docstring for why this can't just be
# a .mcp.json/shell env value.
ensure_persistent_fastembed_cache()

# Same env var + default the MCP servers read (tools/ingest_mcp_server.py's
# DEFAULT_MEMORY_BANK_COLLECTION), so the CLI's --reset guard below agrees
# with index_repo's about which collection is the shared memory bank. Read
# at import time like the server does; an interactive shell that doesn't
# export MEMORY_BANK_COLLECTION falls back to the "memory-bank" default,
# which is the name a stock install uses anyway.
DEFAULT_MEMORY_BANK_COLLECTION = os.environ.get("MEMORY_BANK_COLLECTION", "memory-bank")


async def ingest(args):
    # Issue #265: --reset used to be a raw delete_collection with none of
    # the protections index_repo(reset=True) got in issue #175 / PR #178.
    # Layer 1 (name refusal), ported from index_repo: pointing --reset at
    # the shared memory-bank collection would irreversibly destroy every
    # project's durable memories. Checked BEFORE the dry-run guard and
    # before any Qdrant client exists, so it needs no network and a
    # refused command fails fast even when combined with --dry-run.
    if args.reset and mb.is_memory_bank_collection_name(args.collection, DEFAULT_MEMORY_BANK_COLLECTION):
        print(
            f"Error: '{args.collection}' is configured as the shared memory-bank "
            f"collection (MEMORY_BANK_COLLECTION). Refusing to --reset it -- that "
            f"would wipe every project's durable memories. Use a different "
            f"--collection for this project's code index.",
            file=sys.stderr,
        )
        sys.exit(1)
    embedding_provider = None

    # --dry-run is documented (both in --help and this preview print below)
    # as writing NOTHING to Qdrant -- so all collection maintenance
    # (reset-delete, file_path index backfill) must stay entirely behind
    # this guard, not just the actual store_batch() call further down. PR
    # #121 review (Copilot): an earlier version of this constructed
    # QdrantClient and called collection_exists/ensure_file_path_index
    # unconditionally, which broke an offline dry run outright and, for an
    # online one, silently created a real payload index despite the
    # "preview only" contract -- confirmed by reproducing it directly
    # (mocking QdrantClient showed collection_exists/get_collection/
    # create_payload_index all firing with dry_run=True). This also fixes
    # a narrower, pre-existing version of the same bug: `--reset
    # --dry-run` already deleted the real collection before this PR, for
    # the same reason (the old code's `if args.reset:` block ran before
    # the dry_run check too).
    if not args.dry_run:
        # Constructed unconditionally (not just inside `if args.reset:`)
        # since the non-reset branch below also needs it for the
        # file_path payload-index backfill (issue #54).
        client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
        if args.reset:
            if call_with_retry(client.collection_exists, args.collection):
                embedding_provider = FastEmbedProvider(args.embedding_model)
                # Layer 3 (issue #265), ported from index_repo: validate the
                # embedding model BEFORE deleting anything. Otherwise a
                # model change (or an unnamed-vector legacy collection)
                # would wipe the old index first and only then fail on the
                # store below, leaving nothing. fail_closed=True because
                # this is a pre-delete safety check -- an inconclusive
                # result must block the delete, not count as "compatible."
                mismatch = check_embedding_model_mismatch(
                    client, args.collection, embedding_provider, fail_closed=True
                )
                if mismatch:
                    print(mismatch, file=sys.stderr)
                    sys.exit(1)
                # Layer 2 (issue #265), ported from index_repo: ALWAYS a
                # filtered delete, never delete_collection, so any
                # metadata.source == "memory-bank" points survive even if
                # this collection somehow holds them. Unconditional (no
                # count-then-act) to avoid the race documented in
                # index_repo: a remember() landing between a count and a
                # delete_collection would be destroyed anyway.
                call_with_retry(
                    client.delete,
                    collection_name=args.collection,
                    points_selector=mb.memory_bank_exclusion_filter(),
                )
                print(
                    f"Reset: deleted existing non-memory-bank points from '{args.collection}' "
                    f"(the collection itself and any memory-bank points are kept)."
                )
        elif client.collection_exists(args.collection):
            # FIELD_INDEXES below only takes effect when QdrantConnector's
            # own _ensure_collection_exists() creates a BRAND NEW
            # collection -- an already-existing collection (any repo
            # indexed before this fix, or simply re-run without --reset)
            # never hits that branch again, so it needs this explicit
            # backfill instead (issue #54).
            if ensure_file_path_index(client, args.collection):
                print(f"Backfilled the metadata.file_path payload index on '{args.collection}'.")

    if embedding_provider is None:
        embedding_provider = FastEmbedProvider(args.embedding_model)
    connector = QdrantConnector(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        embedding_provider=embedding_provider,
        field_indexes=FIELD_INDEXES,
    )

    repo_path = Path(args.repo_path).resolve()
    include_extensions = set(args.include_ext.split(",")) if args.include_ext else None
    extra_exclude_dirs = set(args.exclude_dirs.split(",")) if args.exclude_dirs else None
    raw_entries, skipped = build_entries(
        repo_path, args.scope, args.chunk_lines, args.overlap,
        include_extensions=include_extensions,
        extra_exclude_dirs=extra_exclude_dirs,
        respect_gitignore=not args.no_gitignore,
    )
    entries = [Entry(content=content, metadata=metadata) for content, metadata in raw_entries]

    print(f"Found {len(entries)} chunks to index from {repo_path}")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable file(s):")
        for rel, reason in skipped:
            print(f"  - {rel}: {reason}")
    if args.dry_run:
        for e in entries[:10]:
            print("---")
            print(e.metadata)
            print(e.content[:200])
        print(f"... dry run, not storing. ({len(entries)} total)")
        return

    async def _print_progress(stored, total):
        print(f"  stored {stored}/{total}")

    # Batched embed+upsert instead of one connector.store() call per chunk --
    # see libs/qdrant_batch_store.py's module docstring for the full
    # diagnosis: real-world testing on a large repo found thousands of
    # individual round-trips in this loop hit a hard connection/request
    # ceiling at a deterministic failure point that a bigger retry count
    # (issue #75/#76) didn't move, confirming retrying more doesn't help
    # once you're actually at the ceiling -- only batching does.
    await store_batch(connector, entries, collection_name=args.collection, progress_callback=_print_progress)

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
    p.add_argument("--reset", action="store_true", help="Delete the collection's existing points before indexing (avoids duplicate chunks on re-runs). "
                         "Like index_repo(reset=True) (issue #175), this is a filtered delete that keeps the "
                         "collection itself and any memory-bank points, refuses the configured "
                         "MEMORY_BANK_COLLECTION outright, and aborts before deleting on an embedding-model "
                         "mismatch -- so it can no longer rebuild a collection's vector schema after an "
                         "EMBEDDING_MODEL change (drop the collection manually for that)")
    args = p.parse_args()
    # See validate_chunk_params's docstring (issue #26) -- overlap >=
    # chunk_lines silently multiplies chunk count by roughly chunk_lines,
    # with no error anywhere otherwise. p.error() is argparse's own idiom:
    # prints usage + this message to stderr and exits 2, same as any other
    # bad-argument combination argparse itself would catch.
    err = validate_chunk_params(args.chunk_lines, args.overlap)
    if err:
        p.error(err)
    return args


def main() -> None:
    """
    Entry point for both `python tools/ingest_to_qdrant.py` (the
    `if __name__` guard below) and the `claude-runway-ingest` console
    script pip/pipx/uvx installs (issue #50) -- pulled out into its own
    function purely so `[project.scripts]` in pyproject.toml has something
    to point `module:function` at; behavior is unchanged either way.
    """
    asyncio.run(ingest(parse_args()))


if __name__ == "__main__":
    main()
