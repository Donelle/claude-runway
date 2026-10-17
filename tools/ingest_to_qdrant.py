#!/usr/bin/env python3
"""
Bulk-index a repo's code and docs into the same Qdrant collection your
`qdrant` MCP server reads from, using the *same* embedding
provider the server uses (so vectors are compatible with qdrant-find).

Usage:
    pip install qdrant-client fastembed pydantic_settings

    python tools/ingest_to_qdrant.py \
        --repo-path /path/to/your/repo \
        --collection my-collection \
        --qdrant-url http://localhost:6333 \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2

--qdrant-url/--qdrant-api-key/--embedding-model/--include-ext/--exclude-dirs
fall back to the same QDRANT_URL/QDRANT_API_KEY/EMBEDDING_MODEL/
INDEX_INCLUDE_EXTENSIONS/INDEX_EXCLUDE_DIRS env vars the codebase-indexer MCP
server reads, read from this shell's environment (not from .mcp.json).
QDRANT_API_KEY is used only when --qdrant-url is QDRANT_URL (or the built-in
default); any other --qdrant-url needs its own --qdrant-api-key.

Chunking is by the embedding model's own token window, not by line count:
the model embeds only the first N tokens of a text (N is read
from the loaded model), so each chunk is cut to fit that window. --overlap is
therefore a number of tokens, and --chunk-lines is accepted but ignored. Every
file is indexed with no per-file cap, so a minified or generated file can become
thousands of chunks: the run prints each file's chunk count as it goes and names
the largest files at the end. Even --dry-run loads the embedding model, to read
that window.

--write-manifest additionally leaves sync_repo's manifest
(.qdrant_index_manifest.json, in --repo-path) behind, so the first sync_repo
after this index re-embeds only what changed instead of every file a second
time. It is an explicit opt-in because the manifest is one file per checkout and
does not record which collection it describes: pass it only when --collection
(on --qdrant-url) is the collection sync_repo uses for this checkout, i.e. the
`codebase-indexer` block's COLLECTION_NAME on its QDRANT_URL in .mcp.json. It
writes only when this run leaves the collection holding exactly what it indexed
(--reset, or a collection that was absent or empty), never on --dry-run or after
a failed run. Without the flag the CLI writes no manifest, and the next
sync_repo re-chunks and re-embeds every file.

Run this once to seed the collection, then re-run periodically (or on a
schedule) to pick up changes. Re-running without --reset will add duplicate
chunks; pass --reset to wipe the collection's unprotected points (everything except memory-bank points and compacts carrying the source marker) before indexing.

Project minimum: Python 3.12 (policy floor; the dependency chain supports >=3.10).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from version_lib import version_string  # noqa: E402

# Answer --version BEFORE the dependency imports
# below. Those exit(1) on a missing/broken fastembed/qdrant_client --
# exactly the half-broken environment where someone most needs to report which
# version they have. parse_args() still declares the flag (so --help lists it).
if len(sys.argv) == 2 and sys.argv[1] == "--version":
    print(version_string("claude-runway-ingest"))
    sys.exit(0)

try:
    from qdrant_connector import QdrantConnector, Entry
    from qdrant_embeddings import FastEmbedProvider
    from qdrant_client import QdrantClient
except ImportError:
    print(
        "Missing dependency. Install with:\n"
        "    pip install qdrant-client fastembed pydantic_settings --break-system-packages\n",
        file=sys.stderr,
    )
    sys.exit(1)

from qdrant_ingest_lib import (
    MANIFEST_FILENAME,
    TokenBudgetError,
    build_entries,
    chunk_position,
    compute_file_hashes,
    ensure_persistent_fastembed_cache,
    format_largest_files,
    ingest_env_defaults,
    parse_csv_set,
    validate_token_overlap,
)
from qdrant_batch_store import store_batch, ensure_file_path_index, FIELD_INDEXES
from qdrant_index_prep import prepare_collection_for_index
from qdrant_model_check import check_embedding_model_mismatch
from qdrant_retry import call_with_retry
from qdrant_credentials import resolve_qdrant_credentials
from sync_manifest_lib import (
    discard_manifest,
    index_baseline,
    invalidation_error,
    save_manifest,
    unindexed_hashed_files,
)
import memory_bank_lib as mb

# Must run before FastEmbedProvider(...) is constructed in ingest() below --
# see the function's own docstring for why this can't just be
# a .mcp.json/shell env value.
ensure_persistent_fastembed_cache()

# Same env var + default the MCP servers read (tools/ingest_mcp_server.py's
# DEFAULT_MEMORY_BANK_COLLECTION, both via ingest_env_defaults), so the CLI's
# --reset guard below agrees with index_repo's about which collection is the
# shared memory bank. Read at import time like the server does; an
# interactive shell that doesn't export MEMORY_BANK_COLLECTION falls back to
# the "memory-bank" default, which is the name a stock install uses anyway
# (a custom name that lives only in .mcp.json is not picked up here).
DEFAULT_MEMORY_BANK_COLLECTION = ingest_env_defaults()["memory_bank_collection"]


async def ingest(args):
    # --Reset used to be a raw delete_collection with none of
    # the protections index_repo(reset=True) has.
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
    # --dry-run is documented (both in --help and this preview print below)
    # as writing NOTHING to Qdrant -- so all collection maintenance
    # (reset-delete, file_path index backfill) must stay entirely behind
    # this guard, not just the actual store_batch() call further down. An earlier version of this constructed
    # QdrantClient and called collection_exists/ensure_file_path_index
    # unconditionally, which broke an offline dry run outright and, for an
    # online one, silently created a real payload index despite the
    # "preview only" contract. `--reset --dry-run` likewise must not delete.
    embedding_provider = None
    budgets: dict = {}
    repo_path = Path(args.repo_path).resolve()

    # --write-manifest: a preview writes nothing, the manifest
    # included, so the flag only counts on a real run.
    write_manifest = args.write_manifest and not args.dry_run
    # How many points the collection held before this run added to it. Stays 0
    # for a collection that does not exist yet; --reset never consults it (the
    # wipe makes this run's own chunks the whole index).
    points_before = {"count": 0}

    def _record_existing_count(existing_count: int):
        # prepare_collection_for_index's existing_points_guard, used only to
        # observe: the CLI has never refused to index into a populated
        # collection, so this always returns None (no error).
        points_before["count"] = existing_count
        if existing_count == 0:
            # prepare_collection_for_index passes `points_count or 0`, and
            # Qdrant can report no count at all (None), which is not "empty".
            # An apparent zero is what lets a run replace the manifest, so it
            # is confirmed with an exact count first: a populated collection
            # misread as empty would lose the manifest that tracks its deleted
            # files.
            points_before["count"] = call_with_retry(
                client.count, collection_name=args.collection, exact=True
            ).count
        return None

    def _invalidate_manifest():
        # An error string, in the shape prepare_collection_for_index aborts on.
        # discard_manifest is looked up here, at call time, so it stays
        # patchable as this module's own name.
        return invalidation_error(repo_path, discard_manifest)

    def _provider_with_budget(model_name: str):
        # Built together so that on --reset an unreadable token
        # window fails BEFORE the filtered delete runs.
        provider = FastEmbedProvider(model_name)
        budgets["budget"] = provider.build_token_budget(args.overlap)
        return provider

    if not args.dry_run:
        client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
        # The same preparation index_repo runs (libs/
        # qdrant_index_prep.py) -- --reset's pre-delete model check (Layer 3,
        # fail_closed) and always-filtered delete (Layer 2), or
        # the file_path index backfill -- with EVERY Qdrant call
        # retry-wrapped. This copy used to skip call_with_retry on the
        # non-reset collection_exists/ensure_file_path_index, so one vpnkit
        # dropped connection (README's Known limitations) aborted the
        # first-big-index run docs/installation.md recommends. The CLI never
        # refuses to index into a non-empty collection (its docstring documents
        # --reset instead), so the existing_points_guard is passed only under
        # --write-manifest, and then only to count what is already there.
        try:
            prep = prepare_collection_for_index(
                client,
                args.collection,
                args.embedding_model,
                reset=args.reset,
                provider_factory=_provider_with_budget,
                retry=call_with_retry,
                model_check=check_embedding_model_mismatch,
                backfill=ensure_file_path_index,
                existing_points_guard=_record_existing_count if write_manifest else None,
                # After the model check, before the destructive delete: a reset
                # that cannot remove the old manifest does not start, so nothing
                # is deleted and a manifest never outlives the index it described.
                before_reset_delete=_invalidate_manifest if write_manifest and args.reset else None,
            )
        except TokenBudgetError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if prep.error:
            print(prep.error, file=sys.stderr)
            sys.exit(1)
        embedding_provider = prep.embedding_provider
        if prep.reset_deleted:
            print(
                f"Reset: deleted existing points from '{args.collection}' except protected ones "
                f"(the collection itself, memory-bank points and marked conversation compacts are kept; "
                f"compacts stored before the marker existed carry none and were deleted)."
            )
        if prep.backfilled:
            print(f"Backfilled the metadata.file_path payload index on '{args.collection}'.")

    # The token budget is needed before chunking, dry run included, so the
    # model is loaded here for a preview too. Reuses the provider
    # --reset's model check already built, rather than loading it twice.
    try:
        if embedding_provider is None:
            embedding_provider = _provider_with_budget(args.embedding_model)
    except TokenBudgetError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    budget = budgets["budget"]

    # Whether this run will leave the collection holding exactly what it
    # indexes, so that a manifest written from it is true: after a reset, or
    # into a collection that was absent or empty. A run into a populated one
    # only adds chunks, so it writes none.
    writes_baseline = write_manifest and (args.reset or points_before["count"] == 0)
    if writes_baseline:
        # From the first stored chunk the old manifest describes a collection
        # that no longer exists, so a run that fails or is cancelled partway
        # must not leave it behind: removing it HERE, not only replacing it at
        # the end, is what guarantees that. A reset on an existing collection
        # already did this before the wipe (the hook above), which is the
        # stricter point; this covers the runs with nothing to wipe and costs
        # nothing when it is already gone.
        manifest_error = _invalidate_manifest()
        if manifest_error:
            print(manifest_error, file=sys.stderr)
            sys.exit(1)

    # parse_csv_set strips each entry and drops blanks -- a bare
    # split(",") turned "--include-ext '.py, .md'" into {".py", " .md"}, which
    # normalizes to ". .md" and silently indexed no markdown at all.
    include_extensions = parse_csv_set(args.include_ext)
    extra_exclude_dirs = parse_csv_set(args.exclude_dirs)
    file_chunk_counts: dict = {}

    current_hashes: dict = {}
    if writes_baseline:
        # BEFORE any file is chunked, never after: a file edited between the two
        # must end up with a hash older than its chunks, so the next sync
        # re-indexes it (see sync_manifest_lib.index_baseline). Same filters as
        # build_entries below, and the CLI's own terminal progress.
        print("Hashing files for the sync manifest...", flush=True)

        def _print_hash_progress(visited: int) -> None:
            print(f"  hashed {visited} file(s)...", flush=True)

        current_hashes, _hash_skipped = compute_file_hashes(
            repo_path, args.scope,
            include_extensions=include_extensions,
            extra_exclude_dirs=extra_exclude_dirs,
            respect_gitignore=not args.no_gitignore,
            extra_exclude_files={MANIFEST_FILENAME},
            on_progress=_print_hash_progress,
        )

    def _print_file(rel: str, chunk_count: int) -> None:
        # Per-file progress: with no per-file cap, a file that
        # expands into thousands of chunks should be visible as it happens.
        file_chunk_counts[rel] = chunk_count
        print(f"  {rel}: {chunk_count} chunk(s)", flush=True)

    raw_entries, skipped = build_entries(
        repo_path, args.scope, args.chunk_lines, args.overlap,
        budget=budget,
        on_file=_print_file,
        include_extensions=include_extensions,
        extra_exclude_dirs=extra_exclude_dirs,
        respect_gitignore=not args.no_gitignore,
        # sync_repo's manifest is a .json file at the repo root,
        # so without this the CLI embeds it whenever a repo previously
        # maintained by sync_repo doesn't gitignore it -- the exact bug
        # fixed for the server's three build_entries/compute_file_hashes
        # call sites.
        extra_exclude_files={MANIFEST_FILENAME},
    )
    entries = [Entry(content=content, metadata=metadata) for content, metadata in raw_entries]

    print(
        f"Found {len(entries)} chunks to index from {repo_path} "
        f"(each cut to the model's {budget.max_tokens}-token window)"
    )
    largest = format_largest_files(file_chunk_counts)
    if largest:
        print(largest)
    if args.chunk_lines != 50:
        print(f"Note: --chunk-lines {args.chunk_lines} was ignored -- chunks are cut by token window, not line count.")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable file(s):")
        for rel, reason in skipped:
            print(f"  - {rel}: {reason}")
    if args.dry_run:
        for e in entries[:10]:
            print("---")
            print(e.metadata, f"[{chunk_position(e.metadata)}]")
            print(e.content[:200])
        print(f"... dry run, not storing. ({len(entries)} total)")
        if args.write_manifest:
            print("--write-manifest has no effect on a dry run: no manifest was written or removed.")
        return

    # The provider was built above, before chunking, because the token budget
    # comes from it.
    connector = QdrantConnector(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        embedding_provider=embedding_provider,
        field_indexes=FIELD_INDEXES,
    )

    async def _print_progress(stored, total):
        print(f"  stored {stored}/{total}")

    # Batched embed+upsert instead of one connector.store() call per chunk --
    # see libs/qdrant_batch_store.py's module docstring for the full
    # diagnosis: real-world testing on a large repo found thousands of
    # individual round-trips in this loop hit a hard connection/request
    # ceiling at a deterministic failure point that a bigger retry count
    # didn't move, confirming retrying more doesn't help
    # once you're actually at the ceiling -- only batching does.
    await store_batch(connector, entries, collection_name=args.collection, progress_callback=_print_progress)

    print("Done.")

    if not args.write_manifest:
        return
    if not writes_baseline:
        print(
            f"The sync manifest was not written: the collection already held {points_before['count']} point(s) "
            f"this run added to, so no manifest from it could say what is indexed (any existing manifest was "
            f"left alone). Run with --reset to index from a clean baseline."
        )
        return
    # Reached only after the last chunk was stored: a run that failed earlier
    # raised out of store_batch above and writes nothing (the old manifest is
    # already gone), and the entry/skip lists below describe a finished run.
    chunked_files = {rel for rel, count in file_chunk_counts.items() if count}
    missed = unindexed_hashed_files(repo_path, current_hashes, skipped, chunked_files)
    baseline = index_baseline(current_hashes, skipped, chunked_files, missed)
    try:
        save_manifest(repo_path, baseline)
        print(f"Wrote the sync manifest ({len(baseline)} file(s)) so sync_repo can pick up from here.")
    except OSError as e:
        # The chunks are already stored and searchable, so a manifest that
        # cannot be written (read-only checkout) is reported, not raised. The
        # next sync_repo finds no manifest and rebuilds, as without the flag.
        print(f"Warning: could not write the sync manifest ({e}); the next sync_repo will rebuild the index.")


def parse_args():
    # Every default below used to be hardcoded, so the CLI
    # ignored the QDRANT_URL/QDRANT_API_KEY/EMBEDDING_MODEL/
    # INDEX_INCLUDE_EXTENSIONS/INDEX_EXCLUDE_DIRS the MCP server honors. Both
    # now resolve them through the same ingest_env_defaults(). Precedence: an
    # explicit flag, then the env var, then the built-in default. Resolved
    # here (not at import) so it reflects the environment at parse time.
    # The help strings deliberately never interpolate %(default)s: the API
    # key's default can be a real secret from the environment.
    env = ingest_env_defaults()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version=version_string("claude-runway-ingest"))
    p.add_argument("--repo-path", required=True, help="Path to the repo to index")
    p.add_argument("--collection", required=True, help="Qdrant collection name (must match your MCP server's COLLECTION_NAME)")
    p.add_argument("--qdrant-url", default=env["qdrant_url"],
                    help="Default: $QDRANT_URL if set, else http://localhost:6333")
    p.add_argument("--qdrant-api-key", default=None,
                    help="Default: $QDRANT_API_KEY if set, but only when --qdrant-url is the configured URL "
                         "($QDRANT_URL, or the built-in default). Any other --qdrant-url gets no key unless this "
                         "flag is passed")
    p.add_argument("--embedding-model", default=env["embedding_model"],
                    help="Must match your MCP server's EMBEDDING_MODEL exactly. Default: $EMBEDDING_MODEL "
                         "if set, else sentence-transformers/all-MiniLM-L6-v2. Only the shell environment "
                         "is read, not .mcp.json -- pass this flag (or export the var) when the project's "
                         ".mcp.json sets a different model")
    p.add_argument("--scope", choices=["code", "docs", "both"], default="both",
                    help="Ignored if --include-ext (or $INDEX_INCLUDE_EXTENSIONS) is given")
    p.add_argument("--include-ext", default=env["include_extensions"],
                    help="Comma-separated file extensions to ingest, overriding --scope entirely, "
                         "e.g. '.py,.md,.proto' (leading dot optional, spaces around commas are fine). "
                         "Default: $INDEX_INCLUDE_EXTENSIONS if set")
    p.add_argument("--exclude-dirs", default=env["exclude_dirs"],
                    help="Comma-separated folder names to skip, IN ADDITION TO the built-in defaults "
                         "(.git, node_modules, venv, dist, build, etc.) -- e.g. 'fixtures,generated'. "
                         "Default: $INDEX_EXCLUDE_DIRS if set")
    p.add_argument("--no-gitignore", action="store_true",
                    help="Don't apply the repo's own .gitignore on top of the exclude rules above "
                         "(gitignore is respected by default if the optional 'pathspec' package is installed)")
    p.add_argument("--chunk-lines", type=int, default=50,
                    help="Ignored: chunks are cut to the embedding model's token window, not by line "
                         "count. Kept so existing command lines keep working")
    p.add_argument("--overlap", type=int, default=10,
                    help="Tokens repeated between consecutive chunks (was lines before token-window chunking). "
                         "Must be smaller than the model's usable window")
    p.add_argument("--dry-run", action="store_true", help="Preview chunks without writing to Qdrant")
    p.add_argument("--write-manifest", action="store_true",
                    help="Also write sync_repo's manifest (.qdrant_index_manifest.json in --repo-path), so the first "
                         "sync_repo after this index re-embeds only what changed instead of every file. Pass it only "
                         "when --collection (on --qdrant-url) is the collection sync_repo uses for this checkout "
                         "(the codebase-indexer block's COLLECTION_NAME on its QDRANT_URL in .mcp.json): the manifest "
                         "is one file per checkout and does not say which collection it describes. Written only with "
                         "--reset or into a collection that was absent or empty; never on --dry-run or after a failed "
                         "run. The old manifest is removed before anything is deleted or stored. Into a populated "
                         "collection without --reset, the manifest is left alone and a message says why")
    p.add_argument("--reset", action="store_true", help="Delete the collection's existing points before indexing (avoids duplicate chunks on re-runs). "
                         "Like index_repo(reset=True), this is a filtered delete that keeps the "
                         "collection itself, any memory-bank points and marked conversation compacts, refuses the configured "
                         "MEMORY_BANK_COLLECTION outright, and aborts before deleting on an embedding-model "
                         "mismatch -- so it can no longer rebuild a collection's vector schema after an "
                         "EMBEDDING_MODEL change (drop the collection manually for that)")
    args = p.parse_args()
    # The same key rule the MCP tools follow (libs/qdrant_credentials.py): an
    # explicit --qdrant-api-key always wins, $QDRANT_API_KEY goes only to the
    # configured URL ($QDRANT_URL, or the built-in default), and any other
    # --qdrant-url gets no key, so the exported credential never reaches a
    # host named on the command line.
    args.qdrant_url, args.qdrant_api_key = resolve_qdrant_credentials(
        args.qdrant_url,
        args.qdrant_api_key,
        default_url=env["qdrant_url"],
        default_api_key=env["qdrant_api_key"],
    )
    # A negative overlap is meaningless. The upper bound is the model's window,
    # which is only known once the model is loaded, so ingest() checks it then.
    # p.error() is argparse's own idiom: prints usage + this message to stderr
    # and exits 2, same as any other bad-argument combination argparse itself
    # would catch.
    err = validate_token_overlap(args.overlap)
    if err:
        p.error(err)
    return args


def main() -> None:
    """
    Entry point for both `python tools/ingest_to_qdrant.py` (the
    `if __name__` guard below) and the `claude-runway-ingest` console
    script pip/pipx/uvx installs -- pulled out into its own
    function so the `claude_runway:ingest_main` shim, which is what
    `[project.scripts]` in pyproject.toml points at, has
    something to call; behavior is unchanged either way.
    """
    asyncio.run(ingest(parse_args()))


if __name__ == "__main__":
    main()
