#!/usr/bin/env python3
"""Tests for issue #397: conversation-compact collections are marked internal
(libs/qdrant_compact_marker.py), written by compact_store and honored by
find_in_collection/list_collections; and local-compress reads EMBEDDING_MODEL
and checks for a model mismatch.

Stdlib-only (unittest), no network: QdrantClient and the embedding provider
are faked.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "test-collection")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

import ingest_mcp_server as _ims  # noqa: E402
import qdrant_compact_marker as marker  # noqa: E402
from qdrant_client.http.models import models as _m  # noqa: E402
from test_compress_mcp_server import (  # noqa: E402
    _FakeEmbeddingProvider,
    _load_compress_mcp_server,
    _make_fake_qdrant_client_class,
    _run,
)

_MARKED = {marker.MARKER_KEY: marker.MARKER_VALUE}


def _client(metadata=None) -> MagicMock:
    client = MagicMock()
    client.get_collection.return_value.config.metadata = metadata
    return client


class MarkerHelpers(unittest.TestCase):
    def test_is_marked_compact(self):
        self.assertTrue(marker.is_marked_compact(dict(_MARKED, description="x")))
        for metadata in (None, {}, {"description": "x"}, {marker.MARKER_KEY: "other"}, "nope"):
            with self.subTest(metadata=metadata):
                self.assertFalse(marker.is_marked_compact(metadata))

    def test_mark_writes_only_the_marker_key(self):
        # Qdrant merges metadata patches, so a stored description survives;
        # resending a read-back snapshot would risk clobbering concurrent keys.
        client = _client({"description": "keep me"})
        self.assertTrue(marker.mark_compact_collection(client, "c"))
        client.update_collection.assert_called_once_with(collection_name="c", metadata=_MARKED)

    def test_mark_is_idempotent_no_write_when_already_marked(self):
        client = _client(dict(_MARKED))
        self.assertTrue(marker.mark_compact_collection(client, "c"))
        client.update_collection.assert_not_called()

    def test_mark_fails_open(self):
        client = _client({})
        client.update_collection.side_effect = RuntimeError("boom")
        with patch("sys.stderr"):
            self.assertFalse(marker.mark_compact_collection(client, "c"))

    def test_read_fails_open_to_not_compact(self):
        client = MagicMock()
        client.get_collection.side_effect = RuntimeError("boom")
        with patch("sys.stderr"):
            self.assertFalse(marker.collection_is_marked_compact(client, "c"))
        self.assertTrue(marker.collection_is_marked_compact(_client(dict(_MARKED)), "c"))


def _metadata_fake_client_class(vectors_config=None):
    """The shared fake client plus get_collection/update_collection with real
    Qdrant's merge-don't-replace metadata semantics."""
    base = _make_fake_qdrant_client_class()
    if vectors_config is None:
        # Default: the collection already holds the fake provider's vector, so
        # the model-mismatch check passes.
        vectors_config = {
            _FakeEmbeddingProvider.VECTOR_NAME: _m.VectorParams(
                size=_FakeEmbeddingProvider.VECTOR_SIZE, distance=_m.Distance.COSINE
            )
        }
    metadata_store: dict = {}
    updates: list = []

    class _Client(base):
        metadata = metadata_store
        update_calls = updates

        def get_collection(self, collection_name):
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata=metadata_store.get(collection_name),
                    params=SimpleNamespace(vectors=vectors_config),
                )
            )

        def update_collection(self, collection_name, metadata=None):
            updates.append((collection_name, metadata))
            metadata_store.setdefault(collection_name, {}).update(metadata or {})

    return _Client


class CompactStoreMarksCollection(unittest.TestCase):
    def _module(self, **client_kwargs):
        mod = _load_compress_mcp_server()
        client_cls = _metadata_fake_client_class(**client_kwargs)
        mod.QdrantClient = client_cls
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod, client_cls

    def test_store_marks_the_collection_internal(self):
        mod, client_cls = self._module()
        _run(mod.compact_store(information="x", project="my-project", label="l", date="2026-08-27"))
        [(name, metadata)] = client_cls.metadata.items()
        self.assertEqual(metadata, _MARKED)
        self.assertTrue(marker.is_marked_compact(metadata))
        self.assertTrue(name.startswith(mod.COMPACT_COLLECTION))

    def test_every_write_is_idempotent(self):
        mod, client_cls = self._module()
        for label in ("a", "b"):
            _run(mod.compact_store(information="x", project="my-project", label=label, date="2026-08-27"))
        self.assertEqual(len(client_cls.update_calls), 1)

    def test_legacy_unmarked_collection_gets_marked_on_next_store_keeping_its_description(self):
        mod, client_cls = self._module()
        # A pre-#397 collection: exists, has a description hint, no marker.
        col = f"{mod.COMPACT_COLLECTION}-my-project"
        client_cls().collections[col] = {}
        client_cls.metadata[col] = {"description": "legacy"}
        _run(mod.compact_store(information="x", project="my-project", label="l", date="2026-08-27"))
        self.assertEqual(client_cls.metadata[col], dict(_MARKED, description="legacy"))

    def test_marking_failure_does_not_lose_the_compact(self):
        mod, client_cls = self._module()

        def _boom(self, collection_name, metadata=None):
            raise RuntimeError("metadata write refused")

        client_cls.update_collection = _boom
        with patch("sys.stderr"):
            result = _run(mod.compact_store(information="x", project="my-project", label="l", date="2026-08-27"))
        self.assertTrue(result.startswith("Stored compact"))
        self.assertEqual(sum(len(c) for c in client_cls().collections.values()), 1)


class CompactModelMismatch(unittest.TestCase):
    """compact_store/compact_find now call check_embedding_model_mismatch."""

    def _module(self):
        mod = _load_compress_mcp_server()
        # The collection holds the OLD model's vector; the fake provider asks
        # for a different name, as a changed EMBEDDING_MODEL would.
        client_cls = _metadata_fake_client_class(
            vectors_config={"fast-bge-small-en": _m.VectorParams(size=3, distance=_m.Distance.COSINE)}
        )
        mod.QdrantClient = client_cls
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        return mod, client_cls

    def test_store_refuses_a_collection_created_under_another_model(self):
        mod, client_cls = self._module()
        col = f"{mod.COMPACT_COLLECTION}-my-project"
        client_cls().collections[col] = {}
        result = _run(mod.compact_store(information="x", project="my-project", label="l", date="2026-08-27"))
        self.assertIn("embedding model mismatch", result)
        self.assertIn("fast-bge-small-en", result)
        self.assertIn("EMBEDDING_MODEL", result)  # compact-specific hint, not "other repo's .mcp.json"
        self.assertNotIn("other repo", result)
        self.assertEqual(client_cls().collections[col], {})  # nothing written
        self.assertEqual(client_cls.update_calls, [])  # and not marked either

    def test_find_query_reports_the_mismatch(self):
        mod, client_cls = self._module()
        col = f"{mod.COMPACT_COLLECTION}-my-project"
        client_cls().collections[col] = {}
        result = _run(mod.compact_find(project="my-project", query="anything"))
        self.assertIn("embedding model mismatch", result)

    def test_matching_model_still_stores(self):
        mod = _load_compress_mcp_server()
        client_cls = _metadata_fake_client_class(
            vectors_config={
                _FakeEmbeddingProvider.VECTOR_NAME: _m.VectorParams(
                    size=_FakeEmbeddingProvider.VECTOR_SIZE, distance=_m.Distance.COSINE
                )
            }
        )
        mod.QdrantClient = client_cls
        mod.FastEmbedProvider = _FakeEmbeddingProvider
        client_cls().collections[f"{mod.COMPACT_COLLECTION}-my-project"] = {}
        result = _run(mod.compact_store(information="x", project="my-project", label="l", date="2026-08-27"))
        self.assertTrue(result.startswith("Stored compact"))


class CompactEmbeddingModelEnv(unittest.TestCase):
    def _load(self, value=None):
        with mock.patch.dict(os.environ):
            os.environ.pop("EMBEDDING_MODEL", None)
            if value is not None:
                os.environ["EMBEDDING_MODEL"] = value
            return _load_compress_mcp_server().COMPACT_EMBEDDING_MODEL

    def test_unset_falls_back_to_the_legacy_model(self):
        self.assertEqual(self._load(), "BAAI/bge-small-en")

    def test_blank_falls_back_to_the_legacy_model(self):
        self.assertEqual(self._load(""), "BAAI/bge-small-en")

    def test_reads_embedding_model(self):
        self.assertEqual(self._load("sentence-transformers/all-MiniLM-L6-v2"),
                         "sentence-transformers/all-MiniLM-L6-v2")


class FindInCollectionIgnoresCompacts(unittest.TestCase):
    def _find(self, metadata):
        client = _client(metadata)
        client.collection_exists.return_value = True
        with patch.object(_ims, "QdrantClient", return_value=client), \
                patch.object(_ims, "FastEmbedProvider") as provider, \
                patch.object(_ims, "QdrantConnector") as connector, \
                patch.object(_ims, "check_embedding_model_mismatch", return_value="MISMATCH ERROR") as check:
            result = asyncio.run(_ims.find_in_collection("my query", "conversation-compacts-x"))
        return result, provider, connector, check

    def test_marked_collection_returns_the_plain_empty_result(self):
        result, provider, connector, check = self._find(dict(_MARKED))
        # Identical to a genuinely empty search -- no error, no special wording.
        self.assertEqual(
            result, "No results found in collection 'conversation-compacts-x' for query 'my query'."
        )
        provider.assert_not_called()
        connector.assert_not_called()
        check.assert_not_called()

    def test_unmarked_collection_is_searched_as_before(self):
        result, _provider, _connector, check = self._find({"description": "code"})
        self.assertEqual(result, "MISMATCH ERROR")
        check.assert_called_once()


class ListCollectionsHidesCompacts(unittest.TestCase):
    def _list(self, metadata_by_name, **kwargs):
        client = MagicMock()
        client.get_collections.return_value.collections = [SimpleNamespace(name=n) for n in metadata_by_name]

        def _get(name):
            return SimpleNamespace(
                points_count=7, config=SimpleNamespace(metadata=metadata_by_name[name])
            )

        client.get_collection.side_effect = _get
        with patch.object(_ims, "QdrantClient", return_value=client), \
                patch.object(_ims, "get_cached_descriptions", return_value={}), \
                patch.object(_ims, "set_cached_description"):
            return _ims.list_collections(**kwargs)

    def test_marked_collections_are_skipped_entirely(self):
        out = self._list({"code-repo": {"description": "code"}, "their-prefix-compacts": dict(_MARKED)})
        self.assertIn("code-repo", out)
        self.assertNotIn("their-prefix-compacts", out)

    def test_hidden_even_with_counts_and_descriptions_off(self):
        out = self._list(
            {"code-repo": None, "conversation-compacts-x": dict(_MARKED)},
            include_counts=False, include_descriptions=False,
        )
        self.assertIn("code-repo", out)
        self.assertNotIn("conversation-compacts-x", out)

    def test_only_compacts_reads_as_no_collections(self):
        out = self._list({"conversation-compacts-x": dict(_MARKED)})
        self.assertTrue(out.startswith("No collections found"))


if __name__ == "__main__":
    unittest.main()
