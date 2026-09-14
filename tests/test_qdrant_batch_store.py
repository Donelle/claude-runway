#!/usr/bin/env python3
"""Tests for libs/qdrant_batch_store.py (issue #75/#76 follow-up).

Stdlib-only (unittest, no pytest), no network -- uses a fake
QdrantConnector-shaped object (exposing the same "private" attributes
store_batch deliberately reaches into -- see the module docstring) instead
of a real one, since a real QdrantConnector needs a live Qdrant instance.
End-to-end verification against a real Qdrant instance (point counts,
search correctness at 5,500-entry scale matching the actual failure point)
was done manually during development -- see the PR/issue discussion; these
tests cover the batching/retry/progress logic in isolation.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from qdrant_client.http.exceptions import ResponseHandlingException  # noqa: E402
import httpx  # noqa: E402

from qdrant_batch_store import (  # noqa: E402
    store_batch,
    ensure_file_path_index,
    METADATA_PATH,
    FILE_PATH_INDEX_FIELD,
)


class _Entry:
    """Stands in for mcp_server_qdrant.qdrant.Entry -- store_batch only
    reads .content/.metadata, so a plain object is enough."""

    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata


class _FakeEmbeddingProvider:
    def __init__(self, vector_name="fast-test-model", dim=4):
        self.vector_name = vector_name
        self.dim = dim
        self.embed_calls = []  # list of the `documents` list passed each call

    async def embed_documents(self, documents):
        self.embed_calls.append(list(documents))
        return [[float(i)] * self.dim for i in range(len(documents))]

    def get_vector_name(self):
        return self.vector_name


class _FakeAsyncClient:
    def __init__(self, fail_first_n_upserts=0, fail_exc=None):
        self.upsert_calls = []  # list of (collection_name, points) per call
        self._fail_first_n = fail_first_n_upserts
        self._fail_exc = fail_exc or httpx.RemoteProtocolError("disconnected")
        self._upsert_attempts = 0

    async def upsert(self, collection_name, points):
        self._upsert_attempts += 1
        if self._upsert_attempts <= self._fail_first_n:
            raise self._fail_exc
        self.upsert_calls.append((collection_name, points))


class _FakeConnector:
    """Exposes exactly the attributes store_batch reaches into -- see
    qdrant_batch_store.py's module docstring for why those are "private"
    (single-underscore) on the real QdrantConnector."""

    def __init__(self, collection_name="test-collection", **client_kwargs):
        self._default_collection_name = collection_name
        self._embedding_provider = _FakeEmbeddingProvider()
        self._client = _FakeAsyncClient(**client_kwargs)
        self.ensure_calls = []

    async def _ensure_collection_exists(self, collection_name):
        self.ensure_calls.append(collection_name)


class _FakeCollectionInfo:
    """Stands in for qdrant_client.http.models.CollectionInfo --
    ensure_file_path_index only ever reads .payload_schema."""

    def __init__(self, payload_schema=None):
        self.payload_schema = payload_schema or {}


class _FakeSyncClient:
    """Stands in for the plain (sync) QdrantClient index_repo/sync_repo
    already construct for their own collection_exists/delete calls --
    ensure_file_path_index only ever calls get_collection/
    create_payload_index on it."""

    def __init__(self, existing_payload_schema=None):
        self._collections = {"test-collection": _FakeCollectionInfo(existing_payload_schema)}
        self.create_payload_index_calls = []  # list of kwargs dicts

    def get_collection(self, collection_name):
        return self._collections[collection_name]

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.create_payload_index_calls.append({
            "collection_name": collection_name,
            "field_name": field_name,
            "field_schema": field_schema,
        })
        # Mirror real Qdrant behavior: the field now shows up in
        # payload_schema, so a second call would see it as already indexed.
        self._collections[collection_name].payload_schema[field_name] = object()


def _run(coro):
    return asyncio.run(coro)


class StoreBatch(unittest.TestCase):
    def test_empty_entries_returns_zero_and_makes_no_calls(self):
        connector = _FakeConnector()
        stored = _run(store_batch(connector, []))
        self.assertEqual(stored, 0)
        self.assertEqual(connector.ensure_calls, [])
        self.assertEqual(connector._client.upsert_calls, [])

    def test_all_entries_stored_across_multiple_batches(self):
        connector = _FakeConnector()
        entries = [_Entry(f"content {i}", {"file_path": f"f{i}.py"}) for i in range(23)]
        stored = _run(store_batch(connector, entries, batch_size=10))
        self.assertEqual(stored, 23)
        # 23 entries at batch_size=10 -> 3 upsert calls (10, 10, 3), not 23
        # individual ones -- this IS the fix: collapsing N round-trips into
        # ceil(N/batch_size).
        self.assertEqual(len(connector._client.upsert_calls), 3)
        total_points = sum(len(points) for _, points in connector._client.upsert_calls)
        self.assertEqual(total_points, 23)

    def test_ensures_collection_exists_exactly_once_not_per_batch(self):
        connector = _FakeConnector()
        entries = [_Entry(f"c{i}") for i in range(25)]
        _run(store_batch(connector, entries, batch_size=10))
        self.assertEqual(connector.ensure_calls, ["test-collection"])

    def test_uses_connectors_default_collection_when_not_given(self):
        connector = _FakeConnector(collection_name="my-default-collection")
        _run(store_batch(connector, [_Entry("x")]))
        self.assertEqual(connector.ensure_calls, ["my-default-collection"])
        self.assertEqual(connector._client.upsert_calls[0][0], "my-default-collection")

    def test_explicit_collection_name_overrides_default(self):
        connector = _FakeConnector(collection_name="default-coll")
        _run(store_batch(connector, [_Entry("x")], collection_name="other-coll"))
        self.assertEqual(connector.ensure_calls, ["other-coll"])
        self.assertEqual(connector._client.upsert_calls[0][0], "other-coll")

    def test_point_payload_and_vector_shape_matches_qdrant_connector(self):
        """Payload/vector shape must exactly match what
        mcp_server_qdrant.qdrant.QdrantConnector.store()/search() write and
        read -- a mismatch wouldn't error, it would just silently write
        payloads search()/qdrant-find can't parse metadata back out of."""
        connector = _FakeConnector()
        entry = _Entry("hello world", {"file_path": "a.py", "line_range": "1-2"})
        _run(store_batch(connector, [entry]))
        _, points = connector._client.upsert_calls[0]
        self.assertEqual(len(points), 1)
        point = points[0]
        self.assertEqual(point.payload["document"], "hello world")
        self.assertEqual(point.payload[METADATA_PATH], {"file_path": "a.py", "line_range": "1-2"})
        self.assertIn(connector._embedding_provider.vector_name, point.vector)

    def test_embed_documents_called_with_the_whole_batch_at_once(self):
        """embed_documents natively batches -- confirms store_batch passes
        the whole batch's contents in one call rather than embedding one
        document at a time (a real CPU throughput win, not just fewer HTTP
        calls)."""
        connector = _FakeConnector()
        entries = [_Entry(f"doc {i}") for i in range(7)]
        _run(store_batch(connector, entries, batch_size=100))
        self.assertEqual(connector._embedding_provider.embed_calls, [[f"doc {i}" for i in range(7)]])

    def test_progress_callback_invoked_after_each_batch_with_running_total(self):
        connector = _FakeConnector()
        entries = [_Entry(f"c{i}") for i in range(25)]
        progress_calls = []

        async def on_progress(stored, total):
            progress_calls.append((stored, total))

        _run(store_batch(connector, entries, batch_size=10, progress_callback=on_progress))
        self.assertEqual(progress_calls, [(10, 25), (20, 25), (25, 25)])

    def test_no_progress_callback_is_fine(self):
        connector = _FakeConnector()
        stored = _run(store_batch(connector, [_Entry("x")], progress_callback=None))
        self.assertEqual(stored, 1)

    @patch("qdrant_retry.asyncio.sleep", new_callable=AsyncMock)
    def test_transient_upsert_failure_is_retried_via_qdrant_retry(self, mock_sleep):
        """Confirms store_batch actually wires each batch's upsert through
        async_call_with_retry, not just calling connector._client.upsert
        directly -- a batch failing once shouldn't lose the whole run."""
        connector = _FakeConnector(fail_first_n_upserts=1)
        stored = _run(store_batch(connector, [_Entry("x"), _Entry("y")]))
        self.assertEqual(stored, 2)
        self.assertEqual(len(connector._client.upsert_calls), 1, "one logical batch, despite the underlying retry")

    @patch("qdrant_retry.asyncio.sleep", new_callable=AsyncMock)
    def test_wrapped_transient_failure_is_also_retried(self, mock_sleep):
        """Same as above but with the REAL qdrant-client failure shape
        (ResponseHandlingException-wrapped) -- see qdrant_retry.py."""
        connector = _FakeConnector(
            fail_first_n_upserts=1,
            fail_exc=ResponseHandlingException(httpx.ConnectError("dead")),
        )
        stored = _run(store_batch(connector, [_Entry("x")]))
        self.assertEqual(stored, 1)


class EnsureFilePathIndex(unittest.TestCase):
    """Tests for ensure_file_path_index() (issue #54)."""

    def test_creates_index_when_missing(self):
        client = _FakeSyncClient(existing_payload_schema={})
        created = ensure_file_path_index(client, "test-collection")
        self.assertTrue(created)
        self.assertEqual(len(client.create_payload_index_calls), 1)
        call = client.create_payload_index_calls[0]
        self.assertEqual(call["collection_name"], "test-collection")
        self.assertEqual(call["field_name"], FILE_PATH_INDEX_FIELD)

    def test_field_name_is_the_nested_metadata_path_not_the_bare_key(self):
        """The issue's own proposal text suggested the bare "file_path" key
        -- confirmed by reading the real payload shape that the actual
        indexable field is nested under "metadata", matching what
        sync_repo's existing delete-by-filter already queries."""
        self.assertEqual(FILE_PATH_INDEX_FIELD, f"{METADATA_PATH}.file_path")

    def test_skips_create_when_index_already_exists(self):
        client = _FakeSyncClient(existing_payload_schema={FILE_PATH_INDEX_FIELD: object()})
        created = ensure_file_path_index(client, "test-collection")
        self.assertFalse(created)
        self.assertEqual(client.create_payload_index_calls, [])

    def test_second_call_after_first_creation_is_a_no_op(self):
        """Confirms this is safe to call on every sync_repo/index_repo run
        without re-triggering Qdrant's full-collection index-build scan
        each time -- the whole point of checking payload_schema first."""
        client = _FakeSyncClient(existing_payload_schema={})
        first = ensure_file_path_index(client, "test-collection")
        second = ensure_file_path_index(client, "test-collection")
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(client.create_payload_index_calls), 1)


if __name__ == "__main__":
    unittest.main()
