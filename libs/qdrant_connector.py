"""
Async Qdrant connector used by `index_repo`/`sync_repo`/`find_in_collection`
and `tools/ingest_to_qdrant.py`.

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
    ):
        self.qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self.qdrant_api_key = qdrant_api_key
        self.default_collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self.field_indexes = field_indexes

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
        """Creates the collection (and any `field_indexes`) if it is missing.

        `field_indexes` are applied ONLY when this call creates the collection,
        never to one that already exists; `qdrant_batch_store
        .ensure_file_path_index` is the explicit backfill for those.

        Safe against a concurrent creator: two MCP/CLI processes indexing a
        missing collection can both see it absent, and the loser's
        `create_collection` then fails with a conflict. If the collection
        exists after a failed create, the other process won; its schema is
        checked against this provider's and indexing carries on, otherwise
        the original error is re-raised.
        """
        if await self.client.collection_exists(collection_name):
            return
        vector_name = self.embedding_provider.get_vector_name()
        vector_size = self.embedding_provider.get_vector_size()
        try:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
                },
            )
        except Exception:
            if not await self.client.collection_exists(collection_name):
                raise
            await self._require_matching_schema(collection_name, vector_name, vector_size)
            return
        for field_name, field_type in (self.field_indexes or {}).items():
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )

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
