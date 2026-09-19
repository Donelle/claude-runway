#!/usr/bin/env python3
"""Tests for check_embedding_model_mismatch in libs/qdrant_model_check.py
(issue #56; relocated here from tools/ingest_mcp_server.py -- issue #175 --
so tools/memory_bank_mcp_server.py can reuse the identical check).

Stdlib-only (unittest, no pytest) and no network -- all QdrantClient and
FastEmbedProvider calls are mocked, so these run without a live Qdrant
instance or model download.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import qdrant_model_check as _qmc  # noqa: E402
from qdrant_client.http.models import models as _m  # noqa: E402


def _make_provider(vector_name: str, vector_size: int, model_name: str = "test-model") -> MagicMock:
    """Build a mock FastEmbedProvider with controlled name/size."""
    p = MagicMock()
    p.model_name = model_name
    p.get_vector_name.return_value = vector_name
    p.get_vector_size.return_value = vector_size
    return p


def _make_client_named(vectors: dict) -> MagicMock:
    """Return a mocked QdrantClient whose get_collection returns named vectors."""
    info = MagicMock()
    info.config.params.vectors = vectors
    client = MagicMock()
    client.get_collection.return_value = info
    return client


def _make_client_single(size: int) -> MagicMock:
    """Return a mocked QdrantClient whose get_collection returns an unnamed VectorParams."""
    info = MagicMock()
    info.config.params.vectors = _m.VectorParams(size=size, distance=_m.Distance.COSINE)
    client = MagicMock()
    client.get_collection.return_value = info
    return client


def _make_client_none_vectors() -> MagicMock:
    """Return a mocked QdrantClient whose get_collection returns vectors=None."""
    info = MagicMock()
    info.config.params.vectors = None
    client = MagicMock()
    client.get_collection.return_value = info
    return client


class CheckEmbeddingModelMismatchTest(unittest.TestCase):
    # call_with_retry in this module wraps the client call; patch it to pass through.
    def setUp(self):
        # Patch call_with_retry so it just calls fn(*args, **kwargs).
        self._patcher = patch.object(
            _qmc, "call_with_retry", side_effect=lambda fn, *a, **kw: fn(*a, **kw)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    # --- Named-vector collections (mcp-server-qdrant's format) ---

    def test_named_vector_matching_name_and_size_returns_none(self):
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = _make_client_named({
            "fast-all-minilm-l6-v2": _m.VectorParams(size=384, distance=_m.Distance.COSINE)
        })
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNone(result)

    def test_named_vector_wrong_name_returns_error(self):
        provider = _make_provider("fast-all-minilm-l6-v2", 384, model_name="sentence-transformers/all-MiniLM-L6-v2")
        client = _make_client_named({
            "fast-bge-small-en-v1.5": _m.VectorParams(size=384, distance=_m.Distance.COSINE)
        })
        result = _qmc.check_embedding_model_mismatch(client, "myrepo", provider)
        self.assertIsNotNone(result)
        self.assertIn("mismatch", result)
        self.assertIn("fast-all-minilm-l6-v2", result)
        self.assertIn("myrepo", result)

    def test_named_vector_right_name_wrong_size_returns_error(self):
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = _make_client_named({
            "fast-all-minilm-l6-v2": _m.VectorParams(size=768, distance=_m.Distance.COSINE)
        })
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNotNone(result)
        self.assertIn("384", result)
        self.assertIn("768", result)

    def test_empty_named_vector_dict_returns_error(self):
        # An empty {} vectors config indicates a sparse-only collection with no
        # dense vectors. QdrantConnector.search would still attempt using=fast-*
        # and fail at query time -- return an actionable error immediately instead.
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = _make_client_named({})
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNotNone(result)
        self.assertIn("mismatch", result)

    # --- Single unnamed VectorParams (older / third-party ingestion) ---
    # QdrantConnector.search always passes using=get_vector_name(), which Qdrant
    # cannot resolve for an unnamed-vector collection, so BOTH cases return an error.

    def test_single_vector_matching_size_returns_error_about_unnamed_format(self):
        # Even when dimensions match, the query would fail downstream because
        # QdrantConnector uses named-vector search (using="fast-*") which Qdrant
        # can't resolve on an unnamed/default-vector collection.
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = _make_client_single(384)
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNotNone(result)
        self.assertIn("unnamed", result)
        # PR #178 review (third pass): index_repo(reset=true) no longer
        # recreates an existing collection's schema at all -- must recommend
        # dropping/recreating the collection directly, not reset=true.
        self.assertIn("Drop this collection", result)
        self.assertNotIn("Re-index it using index_repo(reset=true)", result)

    def test_single_vector_wrong_size_also_returns_error_about_unnamed_format(self):
        # Dimension mismatch within an unnamed-vector collection is secondary:
        # the collection is already incompatible at the format level.
        provider = _make_provider("fast-all-minilm-l6-v2", 384, model_name="sentence-transformers/all-MiniLM-L6-v2")
        client = _make_client_single(1536)
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNotNone(result)
        self.assertIn("unnamed", result)

    # --- None vectors config (sparse-only with no dense vector config) ---

    def test_none_vectors_config_returns_error(self):
        # vectors=None is not inconclusive -- it means no dense vectors at all
        # (sparse-only collection), which cannot satisfy a named dense-vector search.
        provider = _make_provider("fast-all-minilm-l6-v2", 384, model_name="sentence-transformers/all-MiniLM-L6-v2")
        client = _make_client_none_vectors()
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNotNone(result)
        self.assertIn("mismatch", result)
        self.assertIn("no dense vectors", result)

    # --- Fail-open cases ---

    def test_get_collection_raises_returns_none(self):
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = MagicMock()
        client.get_collection.side_effect = RuntimeError("connection refused")
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNone(result)

    def test_get_vector_size_raises_returns_none(self):
        """Model-description lookup failure (e.g. unknown model name) must fail open."""
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        provider.get_vector_size.side_effect = ValueError("unknown model")
        client = _make_client_named({
            "fast-all-minilm-l6-v2": _m.VectorParams(size=384, distance=_m.Distance.COSINE)
        })
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNone(result)

    # --- fail_closed=True: the two callers (index_repo's and sync_repo's pre-delete checks) that must NOT fail open ---

    def test_fail_closed_returns_error_on_exception_instead_of_none(self):
        # Regression for PR #178 review (fourth pass): a destructive-delete
        # caller must never treat "couldn't verify" the same as "confirmed
        # compatible" -- that's exactly the gap that let reset=True erase an
        # index before a genuine mismatch was ever actually detected.
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = MagicMock()
        client.get_collection.side_effect = RuntimeError("connection refused")
        result = _qmc.check_embedding_model_mismatch(client, "col", provider, fail_closed=True)
        self.assertIsNotNone(result)
        self.assertIn("Error", result)

    def test_fail_closed_still_returns_none_on_genuine_match(self):
        # fail_closed must only change the EXCEPTION path -- a real,
        # successfully-verified match must still return None either way.
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = _make_client_named({
            "fast-all-minilm-l6-v2": _m.VectorParams(size=384, distance=_m.Distance.COSINE)
        })
        result = _qmc.check_embedding_model_mismatch(client, "col", provider, fail_closed=True)
        self.assertIsNone(result)

    def test_fail_closed_still_returns_error_string_on_genuine_mismatch(self):
        # fail_closed must not change the DEFINITIVE-mismatch path either --
        # only the "inconclusive" (exception) path is inverted.
        provider = _make_provider("fast-all-minilm-l6-v2", 384, model_name="sentence-transformers/all-MiniLM-L6-v2")
        client = _make_client_named({
            "fast-bge-small-en-v1.5": _m.VectorParams(size=384, distance=_m.Distance.COSINE)
        })
        result = _qmc.check_embedding_model_mismatch(client, "col", provider, fail_closed=True)
        self.assertIsNotNone(result)
        self.assertIn("mismatch", result)

    def test_default_still_fails_open_on_exception(self):
        # Explicit regression pin: fail_closed defaults to False, so every
        # EXISTING caller (find_in_collection, memory_bank_lib.py) is
        # unaffected by this change.
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = MagicMock()
        client.get_collection.side_effect = RuntimeError("connection refused")
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNone(result)

    def test_unrecognized_vectors_config_shape_fails_closed(self):
        # Regression for PR #178 review (seventh pass): a vectors_config
        # shape that's neither None, dict, nor VectorParams (a future/other
        # qdrant-client type this function doesn't recognize) used to fall
        # through every branch with no explicit return, silently reaching
        # the shared `return None` -- treating "we don't know what this is"
        # as "confirmed compatible" even under fail_closed=True. Verified
        # live: fail_closed=True now returns an actionable error instead.
        class _UnrecognizedShape:
            pass

        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = _make_client_named(_UnrecognizedShape())
        # _make_client_named passes its arg straight through as
        # info.config.params.vectors, so this works even though it's not
        # really a dict.
        result_closed = _qmc.check_embedding_model_mismatch(client, "col", provider, fail_closed=True)
        self.assertIsNotNone(result_closed)
        self.assertIn("Error", result_closed)

    def test_unrecognized_vectors_config_shape_still_fails_open_by_default(self):
        # fail_closed=False (the default) must be unaffected -- an
        # unrecognized shape is inconclusive, not a definitive mismatch.
        class _UnrecognizedShape:
            pass

        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = _make_client_named(_UnrecognizedShape())
        result_open = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNone(result_open)

    # --- Error message quality ---

    def test_wrong_name_error_lists_present_vector_names(self):
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = _make_client_named({
            "fast-bge-large-en": _m.VectorParams(size=1024, distance=_m.Distance.COSINE),
            "fast-bge-small-en": _m.VectorParams(size=512, distance=_m.Distance.COSINE),
        })
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNotNone(result)
        # Both present names should appear so the user knows what to choose from
        self.assertIn("fast-bge-large-en", result)
        self.assertIn("fast-bge-small-en", result)

    def test_error_mentions_mcp_json_hint(self):
        """Error should nudge the user toward checking the other repo's .mcp.json."""
        provider = _make_provider("fast-all-minilm-l6-v2", 384)
        client = _make_client_named({
            "fast-bge-small-en-v1.5": _m.VectorParams(size=384, distance=_m.Distance.COSINE)
        })
        result = _qmc.check_embedding_model_mismatch(client, "col", provider)
        self.assertIsNotNone(result)
        self.assertIn(".mcp.json", result)


if __name__ == "__main__":
    unittest.main()
