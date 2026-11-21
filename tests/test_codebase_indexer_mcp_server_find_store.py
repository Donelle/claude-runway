#!/usr/bin/env python3
"""Tests for the `cb_indexer_find_repo` / `cb_indexer_store_repo` tools of
mcpservers/codebase_indexer_mcp_server.py, and for the tool set that server
registers.

These two tools were `qdrant-find` / `qdrant-store` on the `qdrant` server; the
tests below are those tests, moved with the tools, so they pin that the replies
and the stored points are exactly what they were. Stdlib-only, no network: a
real QdrantConnector on qdrant-client's in-process `:memory:` mode and a fake
embedding provider.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import importlib.util
import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "mcpservers"))
sys.path.insert(0, str(REPO_ROOT / "tests"))  # fake_ownership_client
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "test-collection")

from fake_ownership_client import FakeOwnershipClient, code_chunk, memory_point, unrelated_point  # noqa: E402
from qdrant_code_index_marker import OwnershipCheck, code_index_metadata  # noqa: E402
from qdrant_compact_marker import MARKER_KEY as COMPACT_KEY, MARKER_VALUE as COMPACT_VALUE  # noqa: E402
from qdrant_connector import NO_COLLECTION_ERROR, SEARCH_LIMIT, QdrantConnector  # noqa: E402

EXPECTED_TOOLS = {
    "cb_indexer_index_repo",
    "cb_indexer_sync_repo",
    "cb_indexer_preview_index",
    "cb_indexer_get_collection_info",
    "cb_indexer_find_in_collection",
    "cb_indexer_set_collection_description",
    "cb_indexer_list_collections",
    "cb_indexer_find_repo",
    "cb_indexer_store_repo",
}

OLD_NAMES = (
    "index_repo", "sync_repo", "preview_index", "get_collection_info", "find_in_collection",
    "set_collection_description", "list_collections", "qdrant-find", "qdrant-store",
)


def _load_server():
    """A FRESH copy of the module per call, so a test that replaces an
    attribute cannot leak it into another."""
    spec = importlib.util.spec_from_file_location(
        "codebase_indexer_mcp_server_under_test", REPO_ROOT / "mcpservers" / "codebase_indexer_mcp_server.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(coro):
    return asyncio.run(coro)


class RegisteredToolSet(unittest.TestCase):
    """Exactly the nine `cb_indexer_*` tools, and no old name: a breaking
    rename with no aliases, so an old name must not answer."""

    def test_the_registered_tool_set_is_exactly_the_nine_cb_indexer_tools(self):
        names = {info.name for info in _load_server().mcp._tool_manager.list_tools()}
        self.assertEqual(names, EXPECTED_TOOLS)

    def test_no_old_tool_name_is_registered(self):
        names = {info.name for info in _load_server().mcp._tool_manager.list_tools()}
        self.assertFalse(names & set(OLD_NAMES))

    def test_the_server_keeps_its_name_so_the_tool_prefix_is_unchanged(self):
        self.assertEqual(_load_server().mcp.name, "codebase-indexer")

    def test_find_and_store_expose_the_same_arguments_as_qdrant_find_and_store(self):
        tools = {t.name: t for t in asyncio.run(_load_server().mcp.list_tools())}
        # `limit` is optional: a call that passes just `query` must keep working.
        self.assertEqual(set(tools["cb_indexer_find_repo"].input_schema["properties"]), {"query", "limit"})
        self.assertEqual(tools["cb_indexer_find_repo"].input_schema["required"], ["query"])
        self.assertEqual(
            set(tools["cb_indexer_store_repo"].input_schema["properties"]), {"information", "metadata"}
        )
        self.assertEqual(tools["cb_indexer_store_repo"].input_schema["required"], ["information"])

    def test_descriptions_carry_the_guidance_qdrant_find_and_store_had(self):
        tools = {t.name: t for t in asyncio.run(_load_server().mcp.list_tools())}
        find = tools["cb_indexer_find_repo"].description or ""
        store = tools["cb_indexer_store_repo"].description or ""
        self.assertIn("BEFORE grepping", find)
        self.assertIn("Do NOT use for exact string", find)
        self.assertIn("line_number", find)
        self.assertIn("char_range", find)
        self.assertIn("line_range", find)
        self.assertIn("file_path and line_number", store)
        self.assertIn("speculative", store)
        self.assertIn("optional", store)
        self.assertNotIn("ALWAYS", store)
        self.assertIn("cb_indexer_store_repo", find)
        self.assertIn("cb_indexer_find_repo", store)

    def test_no_tool_description_names_an_old_tool(self):
        # A description that sends the model to a tool that no longer exists
        # fails at the call, not here, so check every one.
        old = re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in OLD_NAMES) + r")(?![\w-])")
        for tool in asyncio.run(_load_server().mcp.list_tools()):
            with self.subTest(tool=tool.name):
                self.assertIsNone(old.search(tool.description or ""))


class _VectorProvider:
    """A deterministic embedding provider for the real QdrantConnector: one
    unit vector per text, so a query for the same text is the nearest hit."""

    def get_vector_name(self):
        return "fast-test-model"

    def get_vector_size(self):
        return 4

    def _vec(self, text):
        v = [0.0] * 4
        v[sum(map(ord, text)) % 4] = 1.0
        return v

    async def embed_documents(self, documents):
        return [self._vec(d) for d in documents]

    async def embed_query(self, query):
        return self._vec(query)


class CbIndexerFindAndStoreTools(unittest.TestCase):
    """cb_indexer_find_repo / cb_indexer_store_repo against a real
    QdrantConnector on qdrant-client's in-process `:memory:` mode, so the tool
    bodies and the connector are exercised together without a server or a
    model."""

    def _server(self, collection="proj"):
        mod = _load_server()
        mod.DEFAULT_COLLECTION = collection
        # Marks what it creates, as the real store connector does.
        mod._find_store_connector = QdrantConnector(
            ":memory:", None, collection, _VectorProvider(), collection_metadata=code_index_metadata()
        )
        # The ownership guard runs on a separate sync client (tested in
        # StoreRefusesForeignCollections below); the in-memory async client
        # above cannot be shared with it, so it is accepted here.
        mod.check_code_index_ownership = lambda *a, **kw: OwnershipCheck()
        return mod

    def test_store_then_find_round_trips_content_and_metadata(self):
        mod = self._server()
        reply = _run(
            mod.cb_indexer_store_repo(
                "auth is handled in middleware", {"file_path": "src/auth.py", "line_range": "10-40"}
            )
        )
        self.assertEqual(reply, "Remembered: auth is handled in middleware in collection proj")
        results = _run(mod.cb_indexer_find_repo("auth is handled in middleware"))
        self.assertEqual(results[0], "Results for the query 'auth is handled in middleware'")
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[1],
            "<entry><content>auth is handled in middleware</content><metadata>"
            + json.dumps({"file_path": "src/auth.py", "line_range": "10-40"})
            + "</metadata></entry>",
        )

    def test_an_entry_stored_without_metadata_has_an_empty_metadata_element(self):
        mod = self._server()
        _run(mod.cb_indexer_store_repo("note", None))
        results = _run(mod.cb_indexer_find_repo("note"))
        self.assertEqual(results[1], "<entry><content>note</content><metadata></metadata></entry>")

    def test_find_with_nothing_stored_says_so(self):
        mod = self._server()
        self.assertEqual(
            _run(mod.cb_indexer_find_repo("anything")), ["No information found for the query 'anything'"]
        )

    def test_find_limit_defaults_to_ten_and_an_explicit_one_wins(self):
        """Repetitive content can fill a fixed top 10 and push the wanted
        chunk past it, so the caller can ask for more (or fewer)."""
        mod = self._server()
        self.assertEqual(SEARCH_LIMIT, 10)
        for i in range(SEARCH_LIMIT + 5):
            _run(mod.cb_indexer_store_repo(f"note {i}", None))
        self.assertEqual(len(_run(mod.cb_indexer_find_repo("note 0"))) - 1, SEARCH_LIMIT)
        self.assertEqual(len(_run(mod.cb_indexer_find_repo("note 0", limit=3))) - 1, 3)
        self.assertEqual(
            len(_run(mod.cb_indexer_find_repo("note 0", limit=SEARCH_LIMIT + 5))) - 1, SEARCH_LIMIT + 5
        )

    def test_find_rejects_a_limit_below_one(self):
        mod = self._server()
        for bad in (0, -1):
            with self.subTest(limit=bad):
                results = _run(mod.cb_indexer_find_repo("note", limit=bad))
                self.assertEqual(len(results), 1)
                self.assertTrue(results[0].startswith("Error: limit must be 1 or greater"))

    def test_store_then_find_round_trips_the_new_position_metadata(self):
        mod = self._server()
        metadata = {"file_path": "src/auth.py", "line_number": 12, "char_range": "340-910"}
        _run(mod.cb_indexer_store_repo("auth is handled in middleware", metadata))
        results = _run(mod.cb_indexer_find_repo("auth is handled in middleware"))
        self.assertIn(json.dumps(metadata), results[1])

    def test_both_tools_report_a_missing_collection_name_instead_of_crashing(self):
        mod = self._server()
        mod.DEFAULT_COLLECTION = None
        self.assertTrue(_run(mod.cb_indexer_find_repo("q"))[0].startswith("Error: COLLECTION_NAME is not set"))
        self.assertTrue(
            _run(mod.cb_indexer_store_repo("x", None)).startswith("Error: COLLECTION_NAME is not set")
        )

    def test_tools_use_the_default_collection_of_this_project_only(self):
        mod = self._server("project-a")
        _run(mod.cb_indexer_store_repo("only in a", None))
        # A second server for another project sharing the same in-memory client
        # must not see it: each tool is scoped to its own COLLECTION_NAME.
        other = self._server("project-b")
        other._find_store_connector.client = mod._find_store_connector.client
        self.assertEqual(
            _run(other.cb_indexer_find_repo("only in a")), ["No information found for the query 'only in a'"]
        )

    def test_a_stored_note_has_the_same_payload_shape_qdrant_store_wrote(self):
        # Notes land in the code-index collection exactly as before, so
        # sync_repo's handling of them (report points whose metadata.type is
        # not code/doc; delete one carrying a re-indexed file's file_path) is
        # unchanged by serving store from this server.
        mod = self._server()
        _run(mod.cb_indexer_store_repo("a note", {"source": "design"}))
        client = mod._find_store_connector.client
        points, _ = _run(client.scroll("proj", limit=10, with_payload=True))
        self.assertEqual([p.payload for p in points], [{"document": "a note", "metadata": {"source": "design"}}])


class StoreRefusesForeignCollections(unittest.TestCase):
    """cb_indexer_store_repo runs the indexer's ownership guard before it
    stores anything: a note must never be upserted into the memory bank, a
    compact collection or any other collection the indexer did not create."""

    def _store(self, client):
        mod = _load_server()
        mod.DEFAULT_COLLECTION = "shared-thing"
        mod.QdrantClient = lambda **kw: client
        mod.call_with_retry = lambda fn, *a, **kw: fn(*a, **kw)
        connector = mock.MagicMock()
        mod._get_find_store_connector = lambda: connector
        stored = []

        async def _store_entry(*args):
            stored.append(args)
            return "Remembered"

        mod.store_entry = _store_entry
        return _run(mod.cb_indexer_store_repo("a note", None)), stored

    def test_foreign_collections_are_refused_before_anything_is_stored(self):
        for label, client in (
            ("memory bank", FakeOwnershipClient(payloads=[memory_point()])),
            ("marked compact collection", FakeOwnershipClient(metadata={COMPACT_KEY: COMPACT_VALUE})),
            ("empty unmarked collection", FakeOwnershipClient()),
            ("unrelated points", FakeOwnershipClient(payloads=[unrelated_point()])),
            ("unreadable metadata", FakeOwnershipClient(fail_on="get_collection")),
        ):
            with self.subTest(label=label):
                result, stored = self._store(client)
                self.assertTrue(result.startswith("Error"), result)
                self.assertIn("'shared-thing'", result)
                self.assertEqual(stored, [])
                self.assertEqual(client.updates, [])

    def test_own_missing_and_legacy_collections_are_stored_into(self):
        for label, client, adopted in (
            ("marked", FakeOwnershipClient(metadata=code_index_metadata()), False),
            ("missing", FakeOwnershipClient(exists=False), False),
            ("legacy code collection", FakeOwnershipClient(payloads=[code_chunk()]), True),
        ):
            with self.subTest(label=label):
                result, stored = self._store(client)
                self.assertEqual(result, "Remembered")
                self.assertEqual(len(stored), 1)
                self.assertEqual(client.updates, [code_index_metadata()] if adopted else [])


class FindStoreDoNotLoadTheModelAtStartup(unittest.TestCase):
    def test_importing_the_module_constructs_no_embedding_model(self):
        # Loading the ONNX model takes seconds (and a download on a cold
        # cache); the server must answer `initialize` first.
        import qdrant_embeddings

        with mock.patch.object(
            qdrant_embeddings, "TextEmbedding", side_effect=AssertionError("model loaded at import")
        ):
            _load_server()

    def test_an_unset_collection_is_reported_without_loading_the_model(self):
        # The connector cache is left EMPTY on purpose: seeding it (as the
        # round-trip tests do) would hide an order bug where the model is
        # loaded before the collection is checked.
        mod = _load_server()
        mod.DEFAULT_COLLECTION = None
        loaded = []

        def _provider(model_name):
            loaded.append(model_name)
            raise RuntimeError("the model must not be loaded for an unset collection")

        mod.FastEmbedProvider = _provider
        self.assertEqual(_run(mod.cb_indexer_find_repo("q")), [NO_COLLECTION_ERROR])
        self.assertEqual(_run(mod.cb_indexer_store_repo("x", None)), NO_COLLECTION_ERROR)
        self.assertEqual(loaded, [])

    def test_the_connector_is_built_on_first_use_and_then_reused(self):
        mod = _load_server()
        mod.DEFAULT_COLLECTION = "proj"
        mod.DEFAULT_EMBEDDING_MODEL = "some/model"
        built = []

        class _Provider:
            def __init__(self, model_name):
                self.model_name = model_name

        class _Connector:
            def __init__(self, url, key, collection, provider, collection_metadata=None,
                         existing_collection_check=None):
                built.append((url, key, collection, provider.model_name))

            async def search(self, query, *, collection_name, limit):
                return []

        mod.FastEmbedProvider = _Provider
        mod.QdrantConnector = _Connector
        self.assertEqual(built, [])
        _run(mod.cb_indexer_find_repo("a"))
        _run(mod.cb_indexer_find_repo("b"))
        self.assertEqual(built, [(mod.DEFAULT_QDRANT_URL, mod.DEFAULT_QDRANT_API_KEY, "proj", "some/model")])


if __name__ == "__main__":
    unittest.main()
