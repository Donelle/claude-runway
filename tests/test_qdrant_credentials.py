#!/usr/bin/env python3
"""Tests for libs/qdrant_credentials.py: the one rule deciding whether the
configured QDRANT_API_KEY may accompany a tool call's qdrant_url. Stdlib only,
no network.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

from qdrant_credentials import resolve_qdrant_credentials  # noqa: E402

DEFAULT_URL = "http://localhost:6333"
DEFAULT_KEY = "configured-secret"


def _resolve(url, key, default_key=DEFAULT_KEY):
    return resolve_qdrant_credentials(url, key, default_url=DEFAULT_URL, default_api_key=default_key)


class ResolveQdrantCredentialsTest(unittest.TestCase):
    def test_foreign_url_without_key_gets_no_key(self):
        # The leak: this used to come back with the configured key attached.
        self.assertEqual(_resolve("https://attacker.example", None), ("https://attacker.example", None))

    def test_foreign_url_with_empty_key_gets_no_key(self):
        # "" is "not given", same as None.
        self.assertEqual(_resolve("https://attacker.example", ""), ("https://attacker.example", None))

    def test_no_url_and_no_key_gets_default_url_and_key(self):
        self.assertEqual(_resolve(None, None), (DEFAULT_URL, DEFAULT_KEY))
        self.assertEqual(_resolve("", ""), (DEFAULT_URL, DEFAULT_KEY))

    def test_explicit_default_url_without_key_gets_default_key(self):
        self.assertEqual(_resolve(DEFAULT_URL, None), (DEFAULT_URL, DEFAULT_KEY))

    def test_trailing_slash_on_default_url_still_gets_default_key(self):
        # Same endpoint, so the project's own Qdrant must not lose its key.
        self.assertEqual(_resolve(DEFAULT_URL + "/", None), (DEFAULT_URL + "/", DEFAULT_KEY))

    def test_explicit_key_passes_through_for_any_url(self):
        self.assertEqual(_resolve("https://other.example", "own-key"), ("https://other.example", "own-key"))
        self.assertEqual(_resolve(DEFAULT_URL, "own-key"), (DEFAULT_URL, "own-key"))
        self.assertEqual(_resolve(None, "own-key"), (DEFAULT_URL, "own-key"))

    def test_lookalike_urls_do_not_get_default_key(self):
        # Only an exact endpoint match earns the key: no prefix, port, scheme,
        # or userinfo trick may widen it.
        for url in (
            "http://localhost:6333.evil.example",
            "http://localhost:63330",
            "https://localhost:6333",
            "http://localhost:6333@evil.example",
            "http://localhost:6333/../x",
            "http://localhost:6334",
        ):
            with self.subTest(url=url):
                self.assertEqual(_resolve(url, None), (url, None))

    def test_unconfigured_default_key_stays_none(self):
        self.assertEqual(_resolve(None, None, default_key=None), (DEFAULT_URL, None))
        self.assertEqual(_resolve(DEFAULT_URL, None, default_key=None), (DEFAULT_URL, None))


if __name__ == "__main__":
    unittest.main()
