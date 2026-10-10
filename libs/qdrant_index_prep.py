"""
Shared "prepare a collection for a full index" step (issue #304), used by
both tools/ingest_mcp_server.py's index_repo and tools/ingest_to_qdrant.py's
ingest().

Before this module, each of those hand-rolled the same sequence -- on reset,
check the embedding model then do a filtered (memory-bank-preserving) delete;
otherwise backfill the metadata.file_path payload index on an existing
collection -- and the CLI's copy drifted: its non-reset collection_exists and
ensure_file_path_index calls skipped call_with_retry, so the exact vpnkit
idle-connection drop libs/qdrant_retry.py exists for aborted the CLI's
first-big-index path (the one docs/installation.md recommends). One shared
code path means a future fix to this sequence lands in both callers at once.

What deliberately stays with each caller, because the two differ on purpose:
  - the memory-bank NAME refusal (index_repo refuses that collection
    regardless of reset; the CLI only refuses it with --reset),
  - --dry-run (the CLI never calls this at all on a dry run, so a preview
    touches neither Qdrant nor the embedding model),
  - how an error is surfaced (index_repo returns the string; the CLI prints
    it and exits 1) and what gets printed on success.

Deliberately a separate module from qdrant_ingest_lib.py, which documents
itself as dependency-free; this one needs qdrant_retry (httpx) and
memory_bank_lib (qdrant-client).
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import memory_bank_lib as mb
from qdrant_batch_store import ensure_file_path_index
from qdrant_model_check import check_embedding_model_mismatch
from qdrant_retry import call_with_retry


@dataclass
class IndexPreparation:
    """Outcome of prepare_collection_for_index.

    error: non-None means the caller must abort with this message (an
        embedding-model mismatch before a reset delete, or whatever
        existing_points_guard returned). Nothing destructive has run then.
    embedding_provider: the provider constructed for the reset branch's
        pre-delete model check, or None if none was needed. Callers reuse it
        instead of loading the model a second time.
    reset_deleted: True if the filtered reset delete ran.
    backfilled: True if the metadata.file_path payload index was newly
        created on an existing collection (issue #54).
    """

    error: Optional[str] = None
    embedding_provider: Any = None
    reset_deleted: bool = False
    backfilled: bool = False


def prepare_collection_for_index(
    client,
    collection: str,
    embedding_model: str,
    *,
    reset: bool,
    provider_factory: Callable[[str], Any],
    existing_points_guard: Optional[Callable[[int], Optional[str]]] = None,
    retry: Callable[..., Any] = call_with_retry,
    model_check: Callable[..., Optional[str]] = check_embedding_model_mismatch,
    backfill: Callable[..., bool] = ensure_file_path_index,
) -> IndexPreparation:
    """
    Run the pre-index collection maintenance for a full index of `collection`.

    Every Qdrant call goes through `retry` (call_with_retry by default) --
    any call here can be the first one after the CPU-bound model load or a
    long idle gap, which is when vpnkit drops the pooled connection (see
    libs/qdrant_retry.py).

    reset=True and the collection exists: build the embedding provider
    (only here -- the non-reset path never needs it before indexing, so an
    early return from existing_points_guard doesn't pay for a model load),
    validate the model BEFORE deleting anything (fail_closed=True: an
    inconclusive result must block a destructive delete, issue #175/#265),
    then ALWAYS a filtered delete that preserves memory-bank and marked
    conversation-compact points -- never delete_collection, and no
    count-then-act, to avoid the race documented in index_repo (PR #178).

    reset=False and the collection exists: if existing_points_guard is
    given, call it with the collection's current point count; a non-None
    return aborts (index_repo's issue #58 duplicate-risk warning). Then
    backfill the metadata.file_path payload index, since FIELD_INDEXES only
    applies when QdrantConnector creates a brand-new collection (issue #54).

    The collection not existing is a no-op either way: QdrantConnector
    creates it (with FIELD_INDEXES) on the first store.

    provider_factory/retry/model_check/backfill are injectable so each
    caller passes its OWN module-level names, keeping those the patch point
    its existing tests already use (e.g. ingest_mcp_server.FastEmbedProvider).
    """
    result = IndexPreparation()
    if reset:
        if retry(client.collection_exists, collection):
            result.embedding_provider = provider_factory(embedding_model)
            mismatch = model_check(client, collection, result.embedding_provider, fail_closed=True)
            if mismatch:
                result.error = mismatch
                return result
            retry(client.delete, collection_name=collection, points_selector=mb.memory_bank_exclusion_filter())
            result.reset_deleted = True
    elif retry(client.collection_exists, collection):
        if existing_points_guard is not None:
            info = retry(client.get_collection, collection)
            guard_error = existing_points_guard(info.points_count or 0)
            if guard_error:
                result.error = guard_error
                return result
        result.backfilled = bool(retry(backfill, client, collection))
    return result
