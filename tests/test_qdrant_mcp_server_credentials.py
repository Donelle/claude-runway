#!/usr/bin/env python3
"""Tests that every mcpservers/qdrant_mcp_server.py tool taking qdrant_url/
qdrant_api_key (the three compact tools) resolves them through the shared
credential rule, so the configured QDRANT_API_KEY is never sent to a
caller-supplied host. Asserts on the kwargs each tool hands to QdrantClient.
Stdlib-only, no network: the client constructor is replaced by one that records
its kwargs and stops the tool, and compact_store's compression step is stubbed
so it never reaches for LM Studio. Touches no files.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import importlib.util
import inspect
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "mcpservers"))

CONFIGURED_URL = "http://configured-qdrant:6333"
CONFIGURED_KEY = "configured-secret"
FOREIGN_URL = "https://attacker.example"


async def _identity_compression(information):
    return information, "stored uncompressed: test"


def _load_module():
    # A fresh copy per test, like tests/test_qdrant_mcp_server.py, so patching
    # one module's globals cannot leak into another test.
    spec = importlib.util.spec_from_file_location("qdrant_mcp_server_creds", REPO_ROOT / "mcpservers" / "qdrant_mcp_server.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._compress_for_storage = _identity_compression
    return mod


class _Stop(Exception):
    """Raised by the fake QdrantClient so the tool stops right after building it."""


class CompactToolCredentialsTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        # Pinned here, not read from the environment, so the test means the
        # same thing on a machine with or without QDRANT_API_KEY exported.
        for name, value in (("COMPACT_QDRANT_URL", CONFIGURED_URL), ("COMPACT_QDRANT_API_KEY", CONFIGURED_KEY)):
            p = patch.object(self.mod, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _tools(self):
        # (name, factory taking the credential kwargs).
        m = self.mod
        return [
            ("compact_store", lambda **kw: m.compact_store("info", "proj", "2026-01-02", **kw)),
            ("compact_find", lambda **kw: m.compact_find("proj", **kw)),
            ("compact_prune", lambda **kw: m.compact_prune("proj", keep_last_n=1, **kw)),
        ]

    def _client_kwargs(self, make_call, **creds):
        with patch.object(self.mod, "QdrantClient", side_effect=_Stop) as client_cls:
            with self.assertRaises(_Stop):
                asyncio.run(make_call(**creds))
        client_cls.assert_called_once()
        return client_cls.call_args.kwargs

    def test_every_credentialed_tool_is_covered(self):
        # Guard against a new tool taking qdrant_api_key being added without a
        # case here: any function in the module with a qdrant_api_key
        # parameter must be listed in _tools().
        def _parameters(obj):
            try:
                return inspect.signature(obj).parameters
            except (ValueError, TypeError):
                return {}

        covered = {name for name, _ in self._tools()}
        taking_key = {
            name
            for name, obj in vars(self.mod).items()
            if callable(obj)
            and not name.startswith("_")
            and "qdrant_api_key" in _parameters(obj)
            and getattr(obj, "__module__", None) == self.mod.__name__
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

    def test_unconfigured_key_stays_none(self):
        # A project with no QDRANT_API_KEY must still send no key to its own URL.
        with patch.object(self.mod, "COMPACT_QDRANT_API_KEY", None):
            for name, make_call in self._tools():
                with self.subTest(tool=name):
                    kwargs = self._client_kwargs(make_call)
                    self.assertEqual(kwargs, {"url": CONFIGURED_URL, "api_key": None})

    def test_docstrings_tell_callers_a_different_url_needs_its_own_key(self):
        for name, _ in self._tools():
            with self.subTest(tool=name):
                doc = " ".join((getattr(self.mod, name).__doc__ or "").split())
                self.assertIn("different qdrant_url needs its own qdrant_api_key", doc)


if __name__ == "__main__":
    unittest.main()
