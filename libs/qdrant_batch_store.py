"""
Batched embed+upsert for `QdrantConnector`-based indexing.

FOUND BY REAL-WORLD TESTING: bumping call_with_retry from 1 retry to
MAX_ATTEMPTS=4 (see qdrant_retry.py) made ZERO difference to a connection
drop deep in the per-chunk upsert loop on a large repo -- both the 2-attempt
and 4-attempt runs failed at the EXACT SAME point (5,249 successful upserts,
then every remaining attempt failed). An identical deterministic failure
point across separate runs with different retry counts rules out a
probabilistic transient drop (more attempts would have pushed the failure
point later, or eliminated it) -- it's a hard resource ceiling (local
ephemeral ports / Docker Desktop's vpnkit connection-tracking table,
consistent with the same mechanism already confirmed for the delete loop)
being hit at a consistent cumulative connection/request count. No amount of
retrying helps once you're actually at that ceiling; only reducing the
NUMBER of round-trips does.

`QdrantConnector.store()` accepts one Entry per call, so a large index would
make one embed + one upsert HTTP round-trip per chunk. This module batches
them through the connector's public `client`, `embedding_provider`,
`default_collection_name` and `ensure_collection_exists` (the connector is
this repo's own, in qdrant_connector.py), rather than constructing a second,
redundant AsyncQdrantClient/connection pool of its own.

embed_documents() already natively batches (passes the whole list straight
to FastEmbed's own batch-optimized passage_embed) -- embedding a batch of
N documents in one call is also a real CPU throughput win, not just an API
convenience, on top of collapsing N upsert HTTP round-trips into one.
"""

from typing import Awaitable, Callable, Optional

from qdrant_client import models

from qdrant_connector import METADATA_PATH, make_point
from qdrant_retry import async_call_with_retry

# Chosen to comfortably clear the connection-exhaustion ceiling this module
# exists to avoid (the failure this replaces happened at ~5,249 individual
# calls) while keeping each single upsert request's payload (points +
# embeddings) a reasonable size. Matches DELETE_BATCH_SIZE's role in
# ingest_mcp_server.py for the delete side of the same problem.
UPSERT_BATCH_SIZE = 250

# Must match the nested key sync_repo's own delete-by-filter already uses
# (tools/ingest_mcp_server.py's models.FieldCondition(key="metadata.file_path",
# ...)): store()/store_batch() write {"document": ..., METADATA_PATH:
# entry.metadata}, and entry.metadata itself has a "file_path" key (see
# qdrant_ingest_lib.py's chunk_file), so the indexable field is the NESTED
# "metadata.file_path", not the bare "file_path".
FILE_PATH_INDEX_FIELD = f"{METADATA_PATH}.file_path"

# Passed to QdrantConnector(..., field_indexes=FIELD_INDEXES) so a BRAND NEW
# collection gets this index automatically the moment
# QdrantConnector.ensure_collection_exists() creates it. That method only
# ever applies field_indexes on the branch where the collection doesn't
# already exist yet -- it's never re-applied to a collection that's already
# there. That's the common case in practice (every repo indexed before this
# index existed), which is why ensure_file_path_index() below exists as a
# separate, explicit backfill -- this constant alone does not fix an
# already-existing collection.
FIELD_INDEXES = {FILE_PATH_INDEX_FIELD: models.PayloadSchemaType.KEYWORD}


async def store_batch(
    connector,
    entries: list,
    *,
    collection_name: Optional[str] = None,
    batch_size: int = UPSERT_BATCH_SIZE,
    progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
) -> int:
    """
    Embeds and upserts `entries` (a list of qdrant_connector.Entry)
    in batches of `batch_size`, instead of QdrantConnector.store()'s one
    embed+upsert HTTP round-trip per entry. Returns the number of entries
    stored (== len(entries) on success). Each batch's upsert call is
    retried via qdrant_retry (see its module docstring) since batching
    reduces exhaustion risk but doesn't eliminate the possibility of an
    individual transient drop.

    progress_callback, if given, is awaited as
    progress_callback(stored_so_far, total) after each batch -- callers
    that report MCP progress (to avoid client-side call timeouts on large
    repos) should pass one instead of trying to report per-chunk, since
    there's no longer a per-chunk round-trip to hang a progress report off.
    """
    if not entries:
        return 0

    collection_name = collection_name or connector.default_collection_name
    assert collection_name is not None
    await connector.ensure_collection_exists(collection_name)

    vector_name = connector.embedding_provider.get_vector_name()
    stored = 0
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        contents = [entry.content for entry in batch]
        embeddings = await connector.embedding_provider.embed_documents(contents)
        points = [
            make_point(vector_name, embedding, entry)
            # strict=False: identical to pre-ruff behavior (a bare zip()
            # already silently truncates to the shorter iterable); ruff's
            # B905 adoption must not change runtime behavior.
            for entry, embedding in zip(batch, embeddings, strict=False)
        ]
        await async_call_with_retry(connector.client.upsert, collection_name=collection_name, points=points)
        stored += len(batch)
        if progress_callback is not None:
            await progress_callback(stored, len(entries))
    return stored


def ensure_file_path_index(client, collection_name: str) -> bool:
    """
    Idempotently backfills the KEYWORD payload index on
    FILE_PATH_INDEX_FIELD for an ALREADY-EXISTING collection (issue #54).

    Without this, sync_repo's per-file delete-by-filter -- and any other
    metadata.file_path-scoped query -- full-scans every point in the
    collection on every changed/removed file. FIELD_INDEXES (passed to
    QdrantConnector's constructor) only creates this index automatically
    when a collection is freshly created; it silently never applies to a
    collection that already exists, which is the common case for any repo
    indexed before this fix shipped. This function covers that gap.

    Takes a plain sync `QdrantClient` (not QdrantConnector) since that's
    what index_repo/sync_repo already construct for their own
    collection_exists/delete calls -- no separate client needed.

    Checks the collection's current payload_schema first rather than
    unconditionally calling create_payload_index every time: Qdrant builds
    a payload index by scanning every existing point once, so blindly
    reissuing this call on every sync_repo run would just move the exact
    full-scan cost this issue exists to eliminate from the delete path to
    here instead.

    Callers must confirm collection_name already exists before calling this
    -- same precondition QdrantConnector's own ensure_collection_exists
    has; get_collection() errors on a missing collection rather than
    returning something falsy.

    Returns True if the index was newly created, False if it already
    existed.
    """
    info = client.get_collection(collection_name)
    if FILE_PATH_INDEX_FIELD in (info.payload_schema or {}):
        return False
    client.create_payload_index(
        collection_name=collection_name,
        field_name=FILE_PATH_INDEX_FIELD,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    return True
