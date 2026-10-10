"""
Collection-metadata marker for conversation-compact collections (issue #397).

`compact_store` (tools/compress_mcp_server.py) writes saved conversation
summaries for /my-compact and /my-resume into their own collections, which
are only ever read back by project through `compact_find` -- never by
semantic search. Nothing used to say so: `find_in_collection` would search
them and `list_collections` listed them next to real codebase collections as
if they were search targets. This module is the ONE place that defines how
such a collection is marked internal, so the writer (compress_mcp_server.py)
and the two readers (ingest_mcp_server.py's `find_in_collection` /
`list_collections`) cannot drift apart on the key, the value, or the
read/write rules. (libs/qdrant_collection_hints.py is a description cache
and not a fit for this.)

The marker is Qdrant collection metadata, the same mechanism
`set_collection_description` uses. `update_collection(metadata=...)` merges
top-level keys rather than replacing the dict (confirmed against a live
Qdrant instance for this issue, and noted in `set_collection_description`'s
docstring), so writing the marker cannot erase
a stored `description`/`search_limit` hint, and writing those cannot erase
the marker.

Metadata rather than a collection-name prefix is deliberate: a consumer
repo's custom `COMPACT_COLLECTION` prefix would be missed by a hardcoded name
match, whereas the marker travels with the collection.

Every helper here fails open (a Qdrant hiccup must never block a store or a
search), reporting to stderr instead.
"""

import sys
from typing import Optional

from qdrant_retry import call_with_retry

# Namespaced so it can't collide with anyone else's metadata keys, and
# deliberately NOT a bare boolean so a future second kind of internal
# collection can reuse the key with a different value.
MARKER_KEY = "claude_runway_internal"
MARKER_VALUE = "conversation-compacts"


def is_marked_compact(metadata: Optional[dict]) -> bool:
    """True when a collection's metadata dict (`info.config.metadata`, which
    may be None) carries the compact marker."""
    return isinstance(metadata, dict) and metadata.get(MARKER_KEY) == MARKER_VALUE


def collection_is_marked_compact(client, collection: str) -> bool:
    """Reads the collection's metadata and returns whether it is marked.
    Fails open to False -- an unreadable collection is treated as an
    ordinary one, exactly as before this marker existed."""
    try:
        info = call_with_retry(client.get_collection, collection)
        return is_marked_compact(info.config.metadata)
    except Exception as e:
        print(f"[claude-runway] could not read compact marker for '{collection}': {e}", file=sys.stderr)
        return False


def mark_compact_collection(client, collection: str) -> bool:
    """
    Idempotently marks `collection` as an internal compact collection.
    Called by `compact_store` on EVERY write, so a legacy collection created
    before this marker existed gets marked on its next /my-compact (no
    backfill, by decision -- issue #397). Skips the network write when the
    marker is already present. Sends ONLY the marker key, relying on Qdrant's
    metadata merge so a stored description/search_limit survives. Returns
    True when the collection is marked on return, False when it could not be
    (logged to stderr, never raised).
    """
    try:
        info = call_with_retry(client.get_collection, collection)
        if is_marked_compact(info.config.metadata):
            return True
        call_with_retry(client.update_collection, collection_name=collection, metadata={MARKER_KEY: MARKER_VALUE})
        return True
    except Exception as e:
        print(f"[claude-runway] could not mark '{collection}' as a compact collection: {e}", file=sys.stderr)
        return False
