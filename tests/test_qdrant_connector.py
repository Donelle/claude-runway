#!/usr/bin/env python3
"""Tests for libs/qdrant_connector.py and libs/qdrant_embeddings.py.

These two modules replaced the third-party Qdrant MCP server package's
connector and embedding provider, so the main job of this file is to pin the on-disk format that existing collections depend on:

  - vector name: "fast-" + the model's basename, lowercased
  - payload: {"document": <content>, "metadata": <metadata>}
  - distance: cosine, one named vector, size from the provider
  - point id: uuid4().hex

and to prove a point shaped the way that package wrote it reads back through
the connector. The tests run against qdrant-client's in-process `:memory:` mode
and a fake embedding provider, so they need no Qdrant server and no model
download.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import json
import os
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from qdrant_client import AsyncQdrantClient, models  # noqa: E402

import qdrant_embeddings  # noqa: E402
from qdrant_connector import (  # noqa: E402
    DOCUMENT_KEY,
    METADATA_PATH,
    NO_COLLECTION_ERROR,
    SEARCH_LIMIT,
    Entry,
    QdrantConnector,
    find_entries,
    format_entry,
    make_point,
    store_entry,
)
from qdrant_embeddings import FastEmbedProvider, vector_name_for_model  # noqa: E402

def _run(coro):
    return asyncio.run(coro)


class _FakeProvider:
    """Deterministic stand-in for FastEmbedProvider: one unit vector per text,
    keyed off the text so a query for the same text is the nearest hit."""

    def __init__(self, vector_name="fast-test-model", size=4):
        self._name = vector_name
        self._size = size

    def _vec(self, text):
        v = [0.0] * self._size
        v[sum(map(ord, text)) % self._size] = 1.0
        return v

    async def embed_documents(self, documents):
        return [self._vec(d) for d in documents]

    async def embed_query(self, query):
        return self._vec(query)

    def get_vector_name(self):
        return self._name

    def get_vector_size(self):
        return self._size


def _connector(provider=None, collection="coll", **kwargs):
    return QdrantConnector(":memory:", None, collection, provider or _FakeProvider(), **kwargs)


class OnDiskFormat(unittest.TestCase):
    def test_vector_name_is_fast_plus_lowercased_basename(self):
        self.assertEqual(vector_name_for_model("BAAI/bge-small-en"), "fast-bge-small-en")
        self.assertEqual(vector_name_for_model("sentence-transformers/All-MiniLM-L6-v2"), "fast-all-minilm-l6-v2")
        self.assertEqual(vector_name_for_model("no-slash-Model"), "fast-no-slash-model")

    def test_payload_keys_are_document_and_metadata(self):
        self.assertEqual(DOCUMENT_KEY, "document")
        self.assertEqual(METADATA_PATH, "metadata")
        point = make_point("fast-x", [0.1, 0.2], Entry("body", {"file_path": "a.py"}))
        self.assertEqual(point.payload, {"document": "body", "metadata": {"file_path": "a.py"}})
        self.assertEqual(point.vector, {"fast-x": [0.1, 0.2]})

    def test_point_ids_are_uuid4_hex(self):
        ids = {make_point("v", [0.0], Entry("x")).id for _ in range(50)}
        self.assertEqual(len(ids), 50, "ids must be unique")
        for point_id in ids:
            self.assertRegex(str(point_id), r"^[0-9a-f]{32}$")

    def test_new_collection_has_one_named_cosine_vector_sized_by_provider(self):
        async def go():
            c = _connector(_FakeProvider("fast-test-model", 7))
            await c.ensure_collection_exists("coll")
            info = await c.client.get_collection("coll")
            return info.config.params.vectors

        vectors = _run(go())
        self.assertEqual(list(vectors), ["fast-test-model"])
        self.assertEqual(vectors["fast-test-model"].size, 7)
        self.assertEqual(vectors["fast-test-model"].distance, models.Distance.COSINE)

    def test_collection_metadata_is_written_in_the_create_request(self):
        # The codebase-indexer's ownership mark (libs/qdrant_code_index_marker.py)
        # rides on this, so a collection it creates is never briefly unmarked.
        async def go(exists):
            c = _connector(collection_metadata={"claude_runway_kind": "code-index"})
            c.client = mock.AsyncMock()
            c.client.collection_exists.return_value = exists
            await c.ensure_collection_exists("coll")
            return c.client

        created = _run(go(False))
        self.assertEqual(
            created.create_collection.call_args.kwargs["metadata"], {"claude_runway_kind": "code-index"}
        )
        # Nothing is written onto an existing collection (it is checked
        # instead, see the tests below).
        created.update_collection.assert_not_awaited()

    def _existing(self, metadata, *, lose_race=False):
        """A connector requiring the code-index mark, over a mock client whose
        collection already exists (or, with lose_race, is created by another
        writer between this connector's existence check and its create)."""
        c = _connector(_FakeProvider("fast-x", 4), collection_metadata={"claude_runway_kind": "code-index"})
        c.client = mock.AsyncMock()
        if lose_race:
            c.client.collection_exists.side_effect = [False, True]
            c.client.create_collection.side_effect = RuntimeError("already exists")
        else:
            c.client.collection_exists.return_value = True
        info = mock.MagicMock()
        info.config.params.vectors = {"fast-x": models.VectorParams(size=4, distance=models.Distance.COSINE)}
        info.config.metadata = metadata
        c.client.get_collection.return_value = info
        return c

    def test_an_existing_collection_without_the_metadata_is_refused(self):
        # A memory-bank or compact collection created under this name after
        # the indexer's ownership check, while it was chunking and embedding.
        for lose_race in (False, True):
            for metadata in (None, {}, {"claude_runway_kind": "other"}, {"description": "x"}):
                with self.subTest(lose_race=lose_race, metadata=metadata):
                    c = self._existing(metadata, lose_race=lose_race)
                    with self.assertRaises(RuntimeError) as caught:
                        _run(c.ensure_collection_exists("coll"))
                    self.assertIn("Refusing to write", str(caught.exception))

    def test_an_existing_collection_with_the_metadata_is_used_and_checked_once(self):
        for lose_race in (False, True):
            with self.subTest(lose_race=lose_race):
                c = self._existing({"claude_runway_kind": "code-index", "description": "d"}, lose_race=lose_race)
                _run(c.ensure_collection_exists("coll"))
                if not lose_race:
                    _run(c.ensure_collection_exists("coll"))
                # The schema check (race only) plus one metadata read, never per batch.
                self.assertEqual(c.client.get_collection.await_count, 2 if lose_race else 1)

    def test_without_collection_metadata_an_existing_collection_is_not_read(self):
        c = _connector()
        c.client = mock.AsyncMock()
        c.client.collection_exists.return_value = True
        _run(c.ensure_collection_exists("coll"))
        c.client.get_collection.assert_not_awaited()

    def test_no_collection_metadata_sends_none(self):
        async def go():
            c = _connector()
            c.client = mock.AsyncMock()
            c.client.collection_exists.return_value = False
            await c.ensure_collection_exists("coll")
            return c.client

        self.assertIsNone(_run(go()).create_collection.call_args.kwargs["metadata"])

    def test_collection_metadata_round_trips_in_process(self):
        async def go():
            c = _connector(collection_metadata={"claude_runway_kind": "code-index"})
            await c.ensure_collection_exists("coll")
            return (await c.client.get_collection("coll")).config.metadata

        self.assertEqual(_run(go()), {"claude_runway_kind": "code-index"})

    def test_store_writes_the_documented_payload_and_vector_name(self):
        async def go():
            c = _connector()
            await c.store(Entry("hello", {"file_path": "a.py"}))
            records, _ = await c.client.scroll("coll", with_vectors=True)
            return records

        [record] = _run(go())
        self.assertEqual(record.payload, {"document": "hello", "metadata": {"file_path": "a.py"}})
        self.assertEqual(list(record.vector), ["fast-test-model"])
        # What a client reports back is not the string we stored: Qdrant
        # canonicalizes UUID ids, and qdrant-client 1.20.0's in-memory mode
        # returns the dashed form where 1.18/1.19 echoed the undashed one. So
        # check the id is a UUID whose hex is 32 characters, whichever spelling
        # comes back; the stored spelling (uuid4().hex) is pinned directly,
        # without a client, by test_point_ids_are_uuid4_hex above.
        self.assertRegex(uuid.UUID(str(record.id)).hex, r"^[0-9a-f]{32}$")

    def test_embedding_provider_vector_size_matches_fastembeds_model_table(self):
        """get_vector_size leans on a private fastembed classmethod; this fails
        loudly if a fastembed upgrade renames it. No model is loaded: the
        description table is read from the class."""
        provider = object.__new__(FastEmbedProvider)
        provider.model_name = "BAAI/bge-small-en"
        provider.embedding_model = mock.Mock(model_name="BAAI/bge-small-en")
        provider.embedding_model._get_model_description = qdrant_embeddings.TextEmbedding._get_model_description
        self.assertEqual(provider.get_vector_size(), 384)
        self.assertEqual(provider.get_vector_name(), "fast-bge-small-en")


class FastEmbedProviderBehavior(unittest.TestCase):
    def _provider(self):
        fake_model = mock.Mock(model_name="BAAI/Bge-Small-EN")
        fake_model.passage_embed.side_effect = lambda docs: [mock.Mock(tolist=lambda d=d: [float(len(d))]) for d in docs]
        fake_model.query_embed.side_effect = lambda qs: [mock.Mock(tolist=lambda: [9.0]) for _ in qs]
        with mock.patch.object(qdrant_embeddings, "TextEmbedding", return_value=fake_model) as ctor:
            provider = FastEmbedProvider("BAAI/Bge-Small-EN")
        ctor.assert_called_once_with("BAAI/Bge-Small-EN", local_files_only=True)
        return provider, fake_model

    def test_documents_use_passage_embed_and_queries_use_query_embed(self):
        provider, model = self._provider()
        self.assertEqual(_run(provider.embed_documents(["ab", "abcd"])), [[2.0], [4.0]])
        self.assertEqual(_run(provider.embed_query("q")), [9.0])
        model.passage_embed.assert_called_once_with(["ab", "abcd"])
        model.query_embed.assert_called_once_with(["q"])

    def test_vector_name_comes_from_the_loaded_models_name(self):
        provider, _ = self._provider()
        self.assertEqual(provider.get_vector_name(), "fast-bge-small-en")


class LoadsFromTheLocalCacheFirst(unittest.TestCase):
    """The provider must not pay for a huggingface.co round-trip when the model
    is already cached (measured 59.6s on a slow network against 2.0s without
    it), and must still download on a cold cache. This replaced the
    `HF_HUB_OFFLINE` entry servers used to need in their config."""

    def test_a_warm_cache_is_loaded_with_no_network_attempt(self):
        model = mock.Mock(model_name="BAAI/bge-small-en")
        ctor = mock.Mock(return_value=model)
        self.assertIs(qdrant_embeddings.load_text_embedding("BAAI/bge-small-en", ctor), model)
        ctor.assert_called_once_with("BAAI/bge-small-en", local_files_only=True)

    def test_a_cold_cache_falls_back_to_a_normal_download(self):
        model = mock.Mock(model_name="BAAI/bge-small-en")
        ctor = mock.Mock(side_effect=[ValueError("model not in the local cache"), model])
        self.assertIs(qdrant_embeddings.load_text_embedding("BAAI/bge-small-en", ctor), model)
        self.assertEqual(
            ctor.call_args_list,
            [mock.call("BAAI/bge-small-en", local_files_only=True), mock.call("BAAI/bge-small-en")],
        )

    def test_a_failed_download_is_not_swallowed(self):
        ctor = mock.Mock(side_effect=[ValueError("not cached"), OSError("no network")])
        with self.assertRaises(OSError):
            qdrant_embeddings.load_text_embedding("BAAI/bge-small-en", ctor)

    def test_the_provider_constructor_uses_the_module_level_class(self):
        # Looked up at call time, so replacing qdrant_embeddings.TextEmbedding
        # (as the tests above and other servers' tests do) takes effect.
        model = mock.Mock(model_name="m")
        with mock.patch.object(qdrant_embeddings, "TextEmbedding", return_value=model) as ctor:
            self.assertIs(FastEmbedProvider("m").embedding_model, model)
        ctor.assert_called_once_with("m", local_files_only=True)


class ConnectorBehavior(unittest.TestCase):
    def test_search_on_a_missing_collection_returns_nothing(self):
        self.assertEqual(_run(_connector().search("anything")), [])

    def test_search_returns_entries_with_content_and_metadata(self):
        async def go():
            c = _connector()
            await c.store(Entry("alpha", {"file_path": "a.py"}))
            await c.store(Entry("beta", None))
            return await c.search("alpha", limit=5)

        hits = _run(go())
        self.assertEqual(hits[0], Entry("alpha", {"file_path": "a.py"}))
        self.assertEqual({h.content for h in hits}, {"alpha", "beta"})

    def test_search_honors_limit_and_query_filter(self):
        async def go():
            c = _connector()
            for i in range(6):
                await c.store(Entry(f"doc {i}", {"file_path": f"f{i % 2}.py"}))
            limited = await c.search("doc", limit=2)
            flt = models.Filter(must=[models.FieldCondition(key="metadata.file_path", match=models.MatchValue(value="f1.py"))])
            filtered = await c.search("doc", limit=10, query_filter=flt)
            return limited, filtered

        limited, filtered = _run(go())
        self.assertEqual(len(limited), 2)
        self.assertEqual(len(filtered), 3)
        self.assertTrue(all(h.metadata["file_path"] == "f1.py" for h in filtered))

    def test_field_indexes_are_applied_only_when_the_collection_is_created(self):
        # qdrant-client's in-process mode ignores payload indexes, so this one
        # records calls on a mock client instead.
        async def go(exists):
            c = _connector(field_indexes={"metadata.file_path": models.PayloadSchemaType.KEYWORD})
            c.client = mock.AsyncMock()
            c.client.collection_exists.return_value = exists
            await c.ensure_collection_exists("coll")
            return c.client

        created = _run(go(False))
        created.create_collection.assert_awaited_once()
        created.create_payload_index.assert_awaited_once_with(
            collection_name="coll", field_name="metadata.file_path", field_schema=models.PayloadSchemaType.KEYWORD
        )
        existing = _run(go(True))
        existing.create_collection.assert_not_awaited()
        existing.create_payload_index.assert_not_awaited()

    def _racing_connector(self, winner_provider):
        """A connector whose first existence check says 'missing' even though
        another process (using `winner_provider`) has just created the
        collection, so its own create_collection fails with a conflict."""

        async def go():
            c = _connector(_FakeProvider("fast-test-model", 4))
            winner = _connector(winner_provider)
            winner.client = c.client
            await winner.ensure_collection_exists("coll")
            real = c.client.collection_exists
            seen = {"n": 0}

            async def racing(name):
                seen["n"] += 1
                return False if seen["n"] == 1 else await real(name)

            c.client.collection_exists = racing
            await c.ensure_collection_exists("coll")

        return go

    def test_losing_a_creation_race_to_a_matching_schema_carries_on(self):
        _run(self._racing_connector(_FakeProvider("fast-test-model", 4))())

    def test_losing_a_creation_race_to_a_different_model_raises_a_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "concurrently with a different embedding schema"):
            _run(self._racing_connector(_FakeProvider("fast-other-model", 4))())

    def test_a_create_failure_with_no_collection_afterwards_is_reraised(self):
        async def go():
            c = _connector()
            c.client = mock.AsyncMock()
            c.client.collection_exists.return_value = False
            c.client.create_collection.side_effect = ConnectionError("boom")
            await c.ensure_collection_exists("coll")

        with self.assertRaisesRegex(ConnectionError, "boom"):
            _run(go())

    def test_default_collection_and_trailing_slash_url(self):
        c = QdrantConnector("http://localhost:6333/", None, "dflt", _FakeProvider())
        self.assertEqual(c.default_collection_name, "dflt")
        self.assertEqual(c.qdrant_url, "http://localhost:6333")

    def test_get_collection_names(self):
        async def go():
            c = _connector()
            await c.ensure_collection_exists("one")
            return await c.get_collection_names()

        self.assertEqual(_run(go()), ["one"])


class ReadsPointsInThePreExistingFormat(unittest.TestCase):
    def test_a_hand_built_point_in_the_old_on_disk_shape_reads_back(self):
        # Written with the raw client, not the connector, so it stands for a
        # collection that already exists on a user's machine.
        async def go():
            provider = _FakeProvider()
            c = _connector(provider)
            await c.ensure_collection_exists("coll")
            await c.client.upsert(
                "coll",
                points=[
                    models.PointStruct(
                        id="0123456789abcdef0123456789abcdef",
                        vector={provider.get_vector_name(): provider._vec("legacy text")},
                        payload={"document": "legacy text", "metadata": {"file_path": "x.py"}},
                    )
                ],
            )
            return await c.search("legacy text")

        [hit] = _run(go())
        self.assertEqual(hit, Entry("legacy text", {"file_path": "x.py"}))


class FindAndStoreHelpers(unittest.TestCase):
    """`find_entries` / `store_entry` are what `cb_indexer_find_repo` /
    `cb_indexer_store_repo` return, moved here from the `qdrant` server's
    `qdrant-find` / `qdrant-store`. They pin the exact reply text, because a
    prompt or caller may already parse it."""

    def _conn(self):
        return QdrantConnector(":memory:", None, "proj", _FakeProvider())

    def test_store_reply_and_find_round_trip(self):
        async def go():
            c = self._conn()
            reply = await store_entry(c, "proj", "auth is in middleware", {"file_path": "a.py"})
            hits = await find_entries(c, "proj", "auth is in middleware")
            return reply, hits

        reply, hits = _run(go())
        self.assertEqual(reply, "Remembered: auth is in middleware in collection proj")
        self.assertEqual(hits[0], "Results for the query 'auth is in middleware'")
        self.assertEqual(
            hits[1],
            "<entry><content>auth is in middleware</content><metadata>"
            + json.dumps({"file_path": "a.py"})
            + "</metadata></entry>",
        )

    def test_the_stored_point_is_in_the_on_disk_format(self):
        async def go():
            c = self._conn()
            await store_entry(c, "proj", "a note", {"source": "design"})
            points, _ = await c.client.scroll("proj", limit=10, with_payload=True)
            return points

        [point] = _run(go())
        self.assertEqual(point.payload, {DOCUMENT_KEY: "a note", METADATA_PATH: {"source": "design"}})

    def test_no_metadata_gives_an_empty_metadata_element(self):
        self.assertEqual(
            format_entry(Entry("x", None)), "<entry><content>x</content><metadata></metadata></entry>"
        )

    def test_nothing_found_and_a_missing_collection_say_so(self):
        c = self._conn()
        self.assertEqual(_run(find_entries(c, "nope", "q")), ["No information found for the query 'q'"])

    def test_limit_defaults_to_ten_and_below_one_is_an_error(self):
        async def go():
            c = self._conn()
            for i in range(SEARCH_LIMIT + 3):
                await store_entry(c, "proj", f"note {i}", None)
            return (
                len(await find_entries(c, "proj", "note 0")) - 1,
                len(await find_entries(c, "proj", "note 0", limit=2)) - 1,
                await find_entries(c, "proj", "note 0", limit=0),
            )

        default, explicit, bad = _run(go())
        self.assertEqual((SEARCH_LIMIT, default, explicit), (10, 10, 2))
        self.assertTrue(bad[0].startswith("Error: limit must be 1 or greater (got 0)"))

    def test_an_unset_collection_is_reported_not_raised(self):
        c = self._conn()
        self.assertEqual(_run(find_entries(c, None, "q")), [NO_COLLECTION_ERROR])
        self.assertEqual(_run(store_entry(c, None, "x")), NO_COLLECTION_ERROR)
        self.assertTrue(NO_COLLECTION_ERROR.startswith("Error: COLLECTION_NAME is not set"))

    def test_a_note_stays_in_its_own_collection(self):
        async def go():
            c = self._conn()
            await store_entry(c, "a", "only in a")
            return await find_entries(c, "b", "only in a")

        self.assertEqual(_run(go()), ["No information found for the query 'only in a'"])


class OldClientTypeStillUsable(unittest.TestCase):
    def test_connector_exposes_a_real_async_client(self):
        self.assertIsInstance(_connector().client, AsyncQdrantClient)


if __name__ == "__main__":
    unittest.main()
