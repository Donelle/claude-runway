#!/usr/bin/env python3
"""Tests that every mcpservers/ingest_mcp_server.py tool taking qdrant_url/
qdrant_api_key resolves them through the shared credential rule, so the
configured QDRANT_API_KEY is never sent to a caller-supplied host. Asserts on
the kwargs each tool hands to QdrantClient. Stdlib-only, no network: the client
constructor is replaced by one that records its kwargs and stops the tool.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "mcpservers"))

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "test-collection")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

import ingest_mcp_server as _ims  # noqa: E402

CONFIGURED_URL = "http://configured-qdrant:6333"
CONFIGURED_KEY = "configured-secret"
FOREIGN_URL = "https://attacker.example"


class _Stop(Exception):
    """Raised by the fake QdrantClient so the tool stops right after building it."""


class IngestToolCredentialsTest(unittest.TestCase):
    def setUp(self):
        # Pinned here, not read from the environment, so the test means the
        # same thing on a machine with or without QDRANT_API_KEY exported.
        for name, value in (("DEFAULT_QDRANT_URL", CONFIGURED_URL), ("DEFAULT_QDRANT_API_KEY", CONFIGURED_KEY)):
            p = patch.object(_ims, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _tools(self):
        # (name, factory taking the credential kwargs). index_repo/sync_repo
        # need a real directory; the others need only a collection name.
        repo = self.tmp.name
        return [
            ("index_repo", lambda **kw: _ims.index_repo(repo_path=repo, collection="c", **kw)),
            ("sync_repo", lambda **kw: _ims.sync_repo(repo_path=repo, collection="c", **kw)),
            ("get_collection_info", lambda **kw: _ims.get_collection_info(collection="c", **kw)),
            ("find_in_collection", lambda **kw: _ims.find_in_collection("q", "c", **kw)),
            ("set_collection_description", lambda **kw: _ims.set_collection_description("c", "d", **kw)),
            ("list_collections", lambda **kw: _ims.list_collections(**kw)),
        ]

    def _client_kwargs(self, make_call, **creds):
        with patch.object(_ims, "QdrantClient", side_effect=_Stop) as client_cls:
            with self.assertRaises(_Stop):
                asyncio.run(make_call(**creds))
        client_cls.assert_called_once()
        return client_cls.call_args.kwargs

    def test_every_credentialed_tool_is_covered(self):
        # Guard against a seventh tool taking qdrant_url being added without a
        # case here: any @mcp.tool with a qdrant_api_key parameter must be
        # listed in _tools().
        import inspect

        def _parameters(obj):
            # Some callables in the module namespace have no introspectable
            # signature (a builtin-derived exception class imported from a lib).
            try:
                return inspect.signature(obj).parameters
            except (ValueError, TypeError):
                return {}

        covered = {name for name, _ in self._tools()}
        taking_key = {
            name
            for name, obj in vars(_ims).items()
            if callable(obj)
            and not name.startswith("_")
            and "qdrant_api_key" in _parameters(obj)
            and getattr(obj, "__module__", None) == _ims.__name__
        }
        self.assertEqual(covered, taking_key)

    def test_foreign_url_without_key_gets_no_key(self):
        for name, make_call in self._tools():
            with self.subTest(tool=name):
                kwargs = self._client_kwargs(make_call, qdrant_url=FOREIGN_URL)
                self.assertEqual(kwargs, {"url": FOREIGN_URL, "api_key": None})

    def test_default_url_without_key_gets_configured_key(self):
        for name, make_call in self._tools():
            with self.subTest(tool=name, how="omitted"):
                kwargs = self._client_kwargs(make_call)
                self.assertEqual(kwargs, {"url": CONFIGURED_URL, "api_key": CONFIGURED_KEY})
            with self.subTest(tool=name, how="explicit default url"):
                kwargs = self._client_kwargs(make_call, qdrant_url=CONFIGURED_URL)
                self.assertEqual(kwargs, {"url": CONFIGURED_URL, "api_key": CONFIGURED_KEY})

    def test_explicit_key_passes_through_for_any_url(self):
        for name, make_call in self._tools():
            with self.subTest(tool=name, url="foreign"):
                kwargs = self._client_kwargs(make_call, qdrant_url=FOREIGN_URL, qdrant_api_key="own-key")
                self.assertEqual(kwargs, {"url": FOREIGN_URL, "api_key": "own-key"})
            with self.subTest(tool=name, url="default"):
                kwargs = self._client_kwargs(make_call, qdrant_api_key="own-key")
                self.assertEqual(kwargs, {"url": CONFIGURED_URL, "api_key": "own-key"})


if __name__ == "__main__":
    unittest.main()
