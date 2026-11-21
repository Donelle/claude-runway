"""
Collection-metadata mark for the collections the codebase-indexer owns.

`cb_indexer_index_repo`, `cb_indexer_sync_repo` and tools/ingest_to_qdrant.py
write code and doc chunks into a collection, and a reset or a sync deletes
points from it. Before this mark existed the only thing stopping them from
doing that to a collection they had no business in was a NAME check against
the indexer's own copy of the memory-bank server's MEMORY_BANK_COLLECTION. That
copy had to be kept equal to the memory-bank block by hand, and it protected
exactly one name: a compact collection, or anything else already in Qdrant,
could still be indexed into or reset. Collection names are local choices
(COLLECTION_NAME, MEMORY_BANK_COLLECTION and COMPACT_COLLECTION can all be
anything), so no name-based guard can generalise.

Instead the indexer marks every collection it creates (QdrantConnector writes
`code_index_metadata()` in the same create_collection request, see
libs/qdrant_connector.py), and `check_code_index_ownership` refuses an
EXISTING collection that does not carry the mark, whatever it is called and
whatever reset/force say. That covers the memory bank, compact collections and
any other collection without the indexer knowing their names.

Same mechanism as libs/qdrant_compact_marker.py: Qdrant collection metadata,
which `update_collection(metadata=...)` merges key by key, so the mark, a
stored `description`/`search_limit` hint and the compact marker cannot erase
one another. A separate key from the compact marker's `claude_runway_internal`
on purpose: that key means "hide this collection from searches", which a code
index must never be.

Unlike the compact marker's helpers this check fails CLOSED: it guards writes
and deletes, so a collection whose metadata cannot be read is refused rather
than treated as ours.

Collections indexed before the mark existed have none. One is ADOPTED (marked,
then accepted) when it holds at least one point with `metadata.file_path`
(written by the indexer for every chunk; the memory-bank server never writes
it) and no point whose `metadata.source` is protected (memory-bank or
conversation-compact), and is not marked as a compact collection. An
unmarked EMPTY collection is not recognised and is refused: nothing in it says
the indexer made it.
"""

import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from qdrant_client import models

import memory_bank_lib as mb
from qdrant_compact_marker import is_marked_compact
from qdrant_retry import call_with_retry

# Namespaced like the compact marker's key, and a value rather than a bare
# boolean so another kind of collection could share the key later.
MARKER_KEY = "claude_runway_kind"
MARKER_VALUE = "code-index"

# Written by the indexer on every code and doc chunk (libs/qdrant_ingest_lib.py)
# and indexed (libs/qdrant_batch_store.py FIELD_INDEXES), so the adoption
# probe below is a cheap indexed lookup on a real code collection.
FILE_PATH_FIELD = "metadata.file_path"


def code_index_metadata() -> dict:
    """The collection metadata the indexer writes when it creates a collection.
    A fresh dict each call, so a caller can never mutate a shared default."""
    return {MARKER_KEY: MARKER_VALUE}


def is_marked_code_index(metadata: Optional[dict]) -> bool:
    """True when a collection's metadata dict (`info.config.metadata`, which
    may be None) carries the code-index mark."""
    return isinstance(metadata, dict) and metadata.get(MARKER_KEY) == MARKER_VALUE


@dataclass
class OwnershipCheck:
    """Outcome of check_code_index_ownership.

    error: non-None means the caller must refuse with this message, before
        writing or deleting anything.
    adopted: True when an unmarked collection of indexer-written chunks was
        recognised and marked by this call.
    """

    error: Optional[str] = None
    adopted: bool = False


def _not_ours(collection: str, why: str) -> str:
    return (
        f"Error: collection '{collection}' already exists and was not created by the codebase-indexer "
        f"({why}). Refusing to write into it (index, reset, sync or store) -- use a different COLLECTION_NAME (or "
        f"collection argument) for this project's code index."
    )


def metadata_ownership_error(collection: str, metadata: Optional[dict]) -> Optional[str]:
    """
    The metadata-only half of the ownership decision: None when `metadata`
    says the collection is the indexer's own, else the refusal message. The
    compact marker is checked FIRST: metadata updates merge keys, so one
    collection can carry both marks (an indexer-created collection that
    compact_store later wrote into and marked under the same name), and a
    compact-marked collection is never ours, whatever else it carries.

    Passed to QdrantConnector as `existing_collection_check`, so the write
    path re-applies exactly this rule to a collection's current metadata
    right before each batch, with no adoption: adoption is decided once, by
    check_code_index_ownership, before a run starts.
    """
    if is_marked_compact(metadata):
        return _not_ours(collection, "it is marked as a conversation-compact collection")
    if is_marked_code_index(metadata):
        return None
    return _not_ours(collection, "it has no code-index mark")


def _has_point(client, collection: str, scroll_filter: models.Filter, retry: Callable[..., Any]) -> bool:
    points, _next = retry(
        client.scroll,
        collection_name=collection,
        scroll_filter=scroll_filter,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(points) > 0


def check_code_index_ownership(
    client, collection: str, *, retry: Callable[..., Any] = call_with_retry
) -> OwnershipCheck:
    """
    Decides whether the indexer may write into (or delete from) `collection`.

    - It does not exist: accepted. QdrantConnector creates it with the mark.
    - It exists and carries the compact marker: refused, even if it also
      carries the code-index mark.
    - It exists and carries the mark: accepted.
    - It exists unmarked, holds a point with metadata.file_path and none with a
      protected metadata.source: adopted (the mark is written) and accepted.
    - Anything else that exists (empty, the memory bank under any name,
      unrelated points): refused.
    - Any Qdrant error along the way: refused (fails closed), with the error
      in the message.

    Every Qdrant call goes through `retry`, injectable so each caller passes
    its own module-level name and keeps its existing patch point.
    """
    try:
        if not retry(client.collection_exists, collection):
            return OwnershipCheck()
        info = retry(client.get_collection, collection)
        metadata = info.config.metadata
        # Compact marker first, then the mark (see metadata_ownership_error).
        if is_marked_compact(metadata) or is_marked_code_index(metadata):
            return OwnershipCheck(metadata_ownership_error(collection, metadata))
        protected = models.Filter(
            should=[
                models.FieldCondition(key=mb.SOURCE_FIELD, match=models.MatchValue(value=mb.MEMORY_BANK_SOURCE)),
                models.FieldCondition(key=mb.SOURCE_FIELD, match=models.MatchValue(value=mb.COMPACT_SOURCE)),
            ]
        )
        if _has_point(client, collection, protected, retry):
            return OwnershipCheck(
                _not_ours(collection, "it holds memory-bank or conversation-compact points")
            )
        has_file_chunks = models.Filter(
            must_not=[models.IsEmptyCondition(is_empty=models.PayloadField(key=FILE_PATH_FIELD))]
        )
        if not _has_point(client, collection, has_file_chunks, retry):
            return OwnershipCheck(
                _not_ours(collection, "it has no code-index mark and holds no indexed file chunks")
            )
        retry(client.update_collection, collection_name=collection, metadata=code_index_metadata())
    except Exception as e:
        return OwnershipCheck(
            f"Error: could not confirm that collection '{collection}' was created by the codebase-indexer "
            f"({e}). Refusing to write into it until its metadata can be read."
        )
    print(
        f"[claude-runway] adopted '{collection}' as a codebase-indexer collection: it predates the "
        f"code-index mark, holds indexed file chunks and no memory-bank or compact points",
        file=sys.stderr,
    )
    return OwnershipCheck(adopted=True)
