"""
Async Qdrant connector used by the `codebase-indexer` server's
`cb_indexer_*` tools and `tools/ingest_to_qdrant.py`.

This is the repo's own `QdrantConnector`/`Entry`, written to replace the ones
the third-party Qdrant MCP server package provided and this repo used to
import. The on-disk format is kept exactly as that package wrote it, because existing collections depend on it:

  - one named vector per point, named by the embedding provider
    (`"fast-" + model basename`, see `qdrant_embeddings.py`), cosine distance,
    size taken from the provider;
  - payload `{"document": <content>, "metadata": <metadata>}`;
  - point ids are `uuid.uuid4().hex`.

`tests/test_qdrant_connector.py` pins each of those, including reading back a
point shaped the way the old package wrote it.

Unlike the package it replaces, the pieces the callers used to reach through
private attributes (the client, the default collection, the provider and
ensure-collection) are public here. `libs/qdrant_batch_store.py` builds its
batched writes on them.
"""

import json
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient, models

# Payload keys. METADATA_PATH is also the prefix of the nested keys the callers
# filter on (e.g. "metadata.file_path"), so it lives here once.
DOCUMENT_KEY = "document"
METADATA_PATH = "metadata"


@dataclass
class Entry:
    """One chunk to store, or one search hit."""

    content: str
    metadata: Optional[Any] = None


def new_point_id() -> str:
    return uuid.uuid4().hex


def make_point(vector_name: str, embedding: list[float], entry: Entry) -> models.PointStruct:
    """Builds one point in the on-disk format described in the module docstring."""
    return models.PointStruct(
        id=new_point_id(),
        vector={vector_name: embedding},
        payload={DOCUMENT_KEY: entry.content, METADATA_PATH: entry.metadata},
    )


class QdrantConnector:
    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: Optional[str],
        embedding_provider,
        qdrant_local_path: Optional[str] = None,
        field_indexes: Optional[dict[str, models.PayloadSchemaType]] = None,
        collection_metadata: Optional[dict[str, Any]] = None,
    ):
        self.qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self.qdrant_api_key = qdrant_api_key
        self.default_collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self.field_indexes = field_indexes
        # Written in the create_collection request itself, never afterwards,
        # so a collection this connector creates cannot exist for a moment
        # without it. The codebase-indexer passes its ownership mark here
        # (libs/qdrant_code_index_marker.py): a separate update after the
        # create could fail and leave an unmarked collection the indexer would
        # then refuse as not its own.
        self.collection_metadata = collection_metadata
        # Collections this connector has already seen carry
        # `collection_metadata` (it created them, or checked them once), so
        # the check below costs one read per collection, not one per batch.
        self._metadata_verified: set = set()

    async def get_collection_names(self) -> list[str]:
        response = await self.client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: Optional[str] = None) -> None:
        """Embeds and upserts ONE entry. Bulk writes should use
        `qdrant_batch_store.store_batch` instead, which collapses the
        per-entry round-trips."""
        collection_name = collection_name or self.default_collection_name
        assert collection_name is not None
        await self.ensure_collection_exists(collection_name)

        embeddings = await self.embedding_provider.embed_documents([entry.content])
        vector_name = self.embedding_provider.get_vector_name()
        await self.client.upsert(
            collection_name=collection_name,
            points=[make_point(vector_name, embeddings[0], entry)],
        )

    async def search(
        self,
        query: str,
        *,
        collection_name: Optional[str] = None,
        limit: int = 10,
        query_filter: Optional[models.Filter] = None,
    ) -> list[Entry]:
        collection_name = collection_name or self.default_collection_name
        assert collection_name is not None
        if not await self.client.collection_exists(collection_name):
            return []

        query_vector = await self.embedding_provider.embed_query(query)
        vector_name = self.embedding_provider.get_vector_name()

        results = await self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )
        # `payload` is Optional in qdrant_client's types; a point this code
        # wrote always has one, and a KeyError on "document" for a foreign
        # point is the same failure the old package had.
        return [
            Entry(
                content=(point.payload or {})[DOCUMENT_KEY],
                metadata=(point.payload or {}).get(METADATA_PATH),
            )
            for point in results.points
        ]

    async def ensure_collection_exists(self, collection_name: str) -> None:
        """Creates the collection (with any `collection_metadata`, and any
        `field_indexes`) if it is missing.

        `field_indexes` and `collection_metadata` are applied ONLY when this
        call creates the collection, never to one that already exists;
        `qdrant_batch_store.ensure_file_path_index` is the explicit backfill
        for the indexes, and `qdrant_code_index_marker
        .check_code_index_ownership` the one for the codebase-indexer's mark.

        Safe against a concurrent creator: two MCP/CLI processes indexing a
        missing collection can both see it absent, and the loser's
        `create_collection` then fails with a conflict. If the collection
        exists after a failed create, the other process won; its schema is
        checked against this provider's and indexing carries on, otherwise
        the original error is re-raised.

        With `collection_metadata`, a collection this call did not create
        (it already existed, or another process won the create) must carry
        every one of those keys, checked once per collection per connector;
        otherwise this raises before anything is written. The codebase-
        indexer checks ownership before it starts, but chunking and embedding
        can take minutes, and a memory-bank, compact or other writer creating
        the same name in that gap must not have code chunks written into its
        collection.
        """
        if await self.client.collection_exists(collection_name):
            await self._require_collection_metadata(collection_name)
            return
        vector_name = self.embedding_provider.get_vector_name()
        vector_size = self.embedding_provider.get_vector_size()
        try:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
                },
                metadata=dict(self.collection_metadata) if self.collection_metadata else None,
            )
        except Exception:
            if not await self.client.collection_exists(collection_name):
                raise
            await self._require_matching_schema(collection_name, vector_name, vector_size)
            await self._require_collection_metadata(collection_name)
            return
        self._metadata_verified.add(collection_name)
        for field_name, field_type in (self.field_indexes or {}).items():
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )

    async def _require_collection_metadata(self, collection_name: str) -> None:
        """Raises unless `collection_name` carries every `collection_metadata`
        key with its value. A no-op without `collection_metadata`, and after
        the first success for a collection."""
        if not self.collection_metadata or collection_name in self._metadata_verified:
            return
        info = await self.client.get_collection(collection_name)
        metadata = info.config.metadata
        missing = [
            key for key, value in self.collection_metadata.items()
            if not isinstance(metadata, dict) or metadata.get(key) != value
        ]
        if missing:
            raise RuntimeError(
                f"Collection '{collection_name}' exists without the metadata this writer requires "
                f"({', '.join(sorted(missing))}): it was created by another writer, not this one. "
                f"Refusing to write into it."
            )
        self._metadata_verified.add(collection_name)

    async def _require_matching_schema(self, collection_name: str, vector_name: str, vector_size: int) -> None:
        """Raises if the collection a concurrent creator won has no vector
        named `vector_name` of `vector_size` (the winner used a different
        embedding model), instead of letting the next upsert fail with an
        opaque Qdrant error."""
        info = await self.client.get_collection(collection_name)
        vectors = info.config.params.vectors
        found = vectors.get(vector_name) if isinstance(vectors, dict) else None
        if found is None or found.size != vector_size:
            present = sorted(vectors) if isinstance(vectors, dict) else "an unnamed vector"
            raise RuntimeError(
                f"Collection '{collection_name}' was created concurrently with a different embedding "
                f"schema: expected vector '{vector_name}' of size {vector_size}, found {present}. "
                f"Every writer to one collection must use the same EMBEDDING_MODEL."
            )


# --- find/store, served as cb_indexer_find_repo / cb_indexer_store_repo ---------
#
# These were the bodies of the `qdrant` server's `qdrant-find`/`qdrant-store`
# tools; they live here so the `codebase-indexer` server (which owns the same
# collection) can serve them without importing the `qdrant` server. Nothing
# here imports MCP, and nothing reads the environment: the caller passes the
# collection in. Their return values are byte-for-byte what those tools
# returned, because a prompt or caller may already parse them.

# How many hits a find returns when the caller passes no limit. The package
# this replaced defaulted to 10 and took no limit at all; the tool takes an
# optional one because repetitive content (generated clients, JSON fixtures,
# lock files) can fill a fixed top 10 with near-duplicates and push the chunk
# that was asked for past it.
SEARCH_LIMIT = 10

NO_COLLECTION_ERROR = (
    "Error: COLLECTION_NAME is not set in this server's env block (.mcp.json), so there is no "
    "project collection to use. Run `claude-runway-setup init` or set it by hand."
)


def format_entry(entry: Entry) -> str:
    """One search hit, in the shape the package this repo replaced returned,
    so a caller or prompt that learned to read it keeps working."""
    entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
    return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"


async def find_entries(
    connector: QdrantConnector, collection_name: Optional[str], query: str, limit: int = SEARCH_LIMIT
) -> list[str]:
    """Semantic search over `collection_name`, formatted for a tool reply."""
    if not collection_name:
        return [NO_COLLECTION_ERROR]
    if limit < 1:
        return [f"Error: limit must be 1 or greater (got {limit})."]
    entries = await connector.search(query, collection_name=collection_name, limit=limit)
    if not entries:
        return [f"No information found for the query '{query}'"]
    return [f"Results for the query '{query}'"] + [format_entry(e) for e in entries]


async def store_entry(
    connector: QdrantConnector,
    collection_name: Optional[str],
    information: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Embeds and stores one note in `collection_name`, returning the reply text."""
    if not collection_name:
        return NO_COLLECTION_ERROR
    await connector.store(Entry(content=information, metadata=metadata), collection_name=collection_name)
    return f"Remembered: {information} in collection {collection_name}"
