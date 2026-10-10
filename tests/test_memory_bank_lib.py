#!/usr/bin/env python3
"""Tests for libs/memory_bank_lib.py (issue #175).

Stdlib-only (unittest, no pytest), no network -- uses MagicMock/AsyncMock
stand-ins for QdrantClient/FastEmbedProvider, the same style
tests/test_qdrant_batch_store.py uses. A live round-trip against a real
Qdrant instance (remember -> recall with default/explicit/all_repos scoping,
count, wipe, forget's same-repo/cross-repo/confirm paths, a malformed
point_id) was also run manually during development -- these tests pin the
same logic in isolation.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import memory_bank_lib as mb  # noqa: E402
from qdrant_client import models  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_provider(vector_name="fast-test-model", dim=4):
    p = MagicMock()
    p.get_vector_name.return_value = vector_name
    p.get_vector_size.return_value = dim
    p.embed_documents = AsyncMock(return_value=[[0.1] * dim])
    p.embed_query = AsyncMock(return_value=[0.1] * dim)
    return p


class _PatchRetryMixin:
    """call_with_retry in memory_bank_lib just calls fn(*args, **kwargs) for
    these tests -- no real transient-connection behavior under test here."""

    def setUp(self):
        self._patcher = patch.object(mb, "call_with_retry", side_effect=lambda fn, *a, **kw: fn(*a, **kw))
        self._patcher.start()
        self.addCleanup(self._patcher.stop)


class ResolveRepoTest(unittest.TestCase):
    def test_general_true_always_resolves_to_sentinel(self):
        repo, err = mb.resolve_repo("anything", general=True)
        self.assertEqual(repo, mb.GENERAL_REPO)
        self.assertIsNone(err)

    def test_general_true_ignores_missing_default_collection(self):
        repo, err = mb.resolve_repo(None, general=True)
        self.assertEqual(repo, mb.GENERAL_REPO)
        self.assertIsNone(err)

    def test_unset_collection_name_errors(self):
        repo, err = mb.resolve_repo(None, general=False)
        self.assertIsNone(repo)
        self.assertIn("Error", err)

    def test_unedited_placeholder_errors(self):
        repo, err = mb.resolve_repo("ANY-NAME-YOU-WANT", general=False)
        self.assertIsNone(repo)
        self.assertIn("Error", err)

    def test_collision_with_general_sentinel_errors(self):
        repo, err = mb.resolve_repo("general", general=False)
        self.assertIsNone(repo)
        self.assertIn("Error", err)

    def test_normal_collection_name_resolves_cleanly(self):
        repo, err = mb.resolve_repo("my-project", general=False)
        self.assertEqual(repo, "my-project")
        self.assertIsNone(err)


class IsMemoryBankCollectionNameTest(unittest.TestCase):
    def test_matches(self):
        self.assertTrue(mb.is_memory_bank_collection_name("memory-bank", "memory-bank"))

    def test_does_not_match(self):
        self.assertFalse(mb.is_memory_bank_collection_name("my-code-repo", "memory-bank"))


class MemoryBankExclusionFilterTest(unittest.TestCase):
    def test_excludes_memory_bank_source(self):
        f = mb.memory_bank_exclusion_filter()
        # Issue #282: conversation compacts are protected by the same fragment.
        self.assertEqual(len(f.must_not), 2)
        for cond in f.must_not:
            self.assertEqual(cond.key, mb.SOURCE_FIELD)
        self.assertEqual(
            {c.match.value for c in f.must_not}, {mb.MEMORY_BANK_SOURCE, mb.COMPACT_SOURCE}
        )


def _matching_collection_info(vector_name="fast-test-model", dim=4):
    """A get_collection() return value whose vector schema matches
    _make_provider()'s default (name/dim), for exercising ensure_collection's
    "already exists, schema matches" path without a real mismatch firing."""
    info = MagicMock()
    info.config.params.vectors = {vector_name: models.VectorParams(size=dim, distance=models.Distance.COSINE)}
    return info


class EnsureCollectionTest(_PatchRetryMixin, unittest.TestCase):
    def test_noop_when_already_exists_and_schema_matches(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = _matching_collection_info()
        provider = _make_provider()
        result = mb.ensure_collection(client, "col", provider)
        self.assertIsNone(result)
        client.create_collection.assert_not_called()

    def test_already_exists_but_schema_mismatch_returns_error(self):
        # A collection matching this NAME already exists, but under a
        # different model's vector name -- must be reported, not silently
        # treated as ready to use.
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = _matching_collection_info(vector_name="fast-other-model", dim=384)
        provider = _make_provider(vector_name="fast-test-model", dim=4)
        result = mb.ensure_collection(client, "col", provider)
        self.assertIsNotNone(result)
        self.assertIn("Error", result)
        client.create_collection.assert_not_called()

    def test_creates_with_matching_vector_schema_when_missing(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        provider = _make_provider(vector_name="fast-x", dim=99)
        result = mb.ensure_collection(client, "col", provider)
        self.assertIsNone(result)
        client.create_collection.assert_called_once()
        _, kwargs = client.create_collection.call_args
        self.assertEqual(kwargs["collection_name"], "col")
        vectors_config = kwargs["vectors_config"]
        self.assertIn("fast-x", vectors_config)
        self.assertEqual(vectors_config["fast-x"].size, 99)

    def test_concurrent_creation_conflict_treated_as_success_when_schema_matches(self):
        """A concurrent creator winning the race raises from create_collection --
        re-checking existence afterward and finding a MATCHING schema must
        return None (success), not raise."""
        client = MagicMock()
        client.collection_exists.side_effect = [False, True]  # missing, then exists after conflict
        client.create_collection.side_effect = RuntimeError("409 Conflict")
        client.get_collection.return_value = _matching_collection_info()
        provider = _make_provider()
        result = mb.ensure_collection(client, "col", provider)
        self.assertIsNone(result)

    def test_concurrent_creation_conflict_with_mismatched_winner_returns_error(self):
        # Regression for PR #178 review finding: two projects with DIFFERENT
        # EMBEDDING_MODEL values race the first remember() call. The loser's
        # create_collection raises (winner got there first); the PREVIOUS
        # version just re-checked existence and returned success unconditionally,
        # silently proceeding to an upsert that would fail opaquely against the
        # winner's actual (different) schema. Must now return the SAME clear
        # mismatch error check_embedding_model_mismatch produces elsewhere.
        client = MagicMock()
        client.collection_exists.side_effect = [False, True]
        client.create_collection.side_effect = RuntimeError("409 Conflict")
        client.get_collection.return_value = _matching_collection_info(vector_name="fast-winner-model", dim=768)
        loser_provider = _make_provider(vector_name="fast-loser-model", dim=4)
        result = mb.ensure_collection(client, "col", loser_provider)
        self.assertIsNotNone(result)
        self.assertIn("Error", result)

    def test_genuine_failure_still_propagates(self):
        """If the collection STILL doesn't exist after create_collection raised,
        it wasn't a concurrent-creation race -- the real error must propagate."""
        client = MagicMock()
        client.collection_exists.side_effect = [False, False]
        client.create_collection.side_effect = RuntimeError("real failure")
        provider = _make_provider()
        with self.assertRaises(RuntimeError):
            mb.ensure_collection(client, "col", provider)


class EnsureMemoryBankIndexesTest(_PatchRetryMixin, unittest.TestCase):
    def test_creates_missing_indexes(self):
        client = MagicMock()
        info = MagicMock()
        info.payload_schema = {}
        client.get_collection.return_value = info
        mb.ensure_memory_bank_indexes(client, "col")
        created_fields = {kw["field_name"] for _, kw in client.create_payload_index.call_args_list}
        self.assertEqual(created_fields, {mb.SOURCE_FIELD, mb.REPO_FIELD})

    def test_skips_existing_indexes(self):
        client = MagicMock()
        info = MagicMock()
        info.payload_schema = {mb.SOURCE_FIELD: MagicMock(), mb.REPO_FIELD: MagicMock()}
        client.get_collection.return_value = info
        mb.ensure_memory_bank_indexes(client, "col")
        client.create_payload_index.assert_not_called()


class RememberPointTest(_PatchRetryMixin, unittest.TestCase):
    def test_payload_shape(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        info = MagicMock()
        info.payload_schema = {mb.SOURCE_FIELD: MagicMock(), mb.REPO_FIELD: MagicMock()}
        client.get_collection.return_value = _matching_collection_info(vector_name="fast-x", dim=4)
        client.get_collection.return_value.payload_schema = info.payload_schema
        provider = _make_provider(vector_name="fast-x", dim=4)

        point_id, created_at, error = _run(
            mb.remember_point(
                client, provider, "col",
                summary="short summary", description="verbatim full text",
                kind="lesson", repo="my-project", embedding_model="model-x",
            )
        )
        self.assertIsNone(error)
        self.assertTrue(point_id)
        client.upsert.assert_called_once()
        _, kwargs = client.upsert.call_args
        point = kwargs["points"][0]
        self.assertEqual(point.payload["document"], "short summary")
        meta = point.payload["metadata"]
        self.assertEqual(meta["source"], mb.MEMORY_BANK_SOURCE)
        self.assertEqual(meta["kind"], "lesson")
        self.assertEqual(meta["description"], "verbatim full text")
        self.assertEqual(meta["repo"], "my-project")
        self.assertEqual(meta["embedding_model"], "model-x")
        self.assertIn("fast-x", point.vector)
        self.assertNotIn("created_at", meta)

        # Issue #177: no weight passed -> stored as DEFAULT_WEIGHT (1.0), in
        # the SAME initial upsert as source/kind/description/etc -- there is
        # no window where the point exists without a weight at all, unlike
        # created_at below.
        self.assertEqual(meta["weight"], mb.DEFAULT_WEIGHT)

        # pending=True is written in the SAME atomic upsert that creates the
        # point (PR #178 review, sixth pass) -- there is no window where the
        # point exists without it, unlike created_at below.
        self.assertIs(meta["pending"], True)

        # created_at is stamped -- and pending cleared -- in a SEPARATE call
        # made AFTER the upsert above returns (PR #178 review, fifth and
        # sixth passes) -- not included in the initial payload, so a value
        # read there is captured before that write's own round-trip
        # completes, reopening the exact race window wipe_memory_bank's
        # cutoff and pending exclusion exist to close.
        client.set_payload.assert_called_once()
        _, set_payload_kwargs = client.set_payload.call_args
        self.assertEqual(set_payload_kwargs["collection_name"], "col")
        self.assertEqual(set_payload_kwargs["points"], [point_id])
        self.assertEqual(set_payload_kwargs["key"], "metadata")
        self.assertIsInstance(set_payload_kwargs["payload"]["created_at"], float)
        self.assertAlmostEqual(set_payload_kwargs["payload"]["created_at"], time.time(), delta=5)
        self.assertIs(set_payload_kwargs["payload"]["pending"], False)

        # Issue #179, PR #197 review: the returned created_at must be the
        # EXACT same value stamped into the set_payload call above, not a
        # separately re-sampled time.time() -- that's the whole point of
        # returning it at all.
        self.assertEqual(created_at, set_payload_kwargs["payload"]["created_at"])

    def test_explicit_weight_is_stored_verbatim(self):
        # Issue #177: an explicit weight (not the default) must be stored
        # exactly as given, not coerced/clamped -- this lower-level helper
        # trusts the value verbatim; validation (rejecting negative/
        # non-finite weights) happens at the remember() tool boundary
        # instead (PR #199 review, fourth pass: an earlier version of this
        # comment claimed weight validation was out of scope for this
        # ticket entirely, which went stale once that validation was added
        # in a later round -- see RememberWeightValidationTest in
        # tests/test_memory_bank_mcp_server.py for that coverage).
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = _matching_collection_info(vector_name="fast-x", dim=4)
        provider = _make_provider(vector_name="fast-x", dim=4)

        _run(
            mb.remember_point(
                client, provider, "col",
                summary="s", description="d", kind="lesson",
                repo="my-project", embedding_model="model-x", weight=2.5,
            )
        )
        _, kwargs = client.upsert.call_args
        meta = kwargs["points"][0].payload["metadata"]
        self.assertEqual(meta["weight"], 2.5)

    def _remember(self, **extra):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = _matching_collection_info(vector_name="fast-x", dim=4)
        provider = _make_provider(vector_name="fast-x", dim=4)
        result = _run(
            mb.remember_point(
                client, provider, "col",
                summary="s", description="d", kind="lesson",
                repo="my-project", embedding_model="model-x", **extra,
            )
        )
        return client, result

    def test_explicit_point_id_is_used_for_upsert_and_returned(self):
        # Issue #334: a caller-supplied ID must be used verbatim for both the
        # upsert and the follow-up set_payload, and returned unchanged.
        client, (point_id, _, error) = self._remember(point_id="deterministic-id")
        self.assertIsNone(error)
        self.assertEqual(point_id, "deterministic-id")
        _, kwargs = client.upsert.call_args
        self.assertEqual(kwargs["points"][0].id, "deterministic-id")
        _, sp = client.set_payload.call_args
        self.assertEqual(sp["points"], ["deterministic-id"])

    def test_omitted_point_id_still_generates_random_uuid_hex(self):
        # Issue #334: default behavior unchanged -- fresh random hex IDs.
        _, (id1, _, _) = self._remember()
        _, (id2, _, _) = self._remember()
        self.assertRegex(id1, r"^[0-9a-f]{32}$")
        self.assertNotEqual(id1, id2)

    def test_schema_mismatch_returns_error_without_upserting(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = _matching_collection_info(vector_name="fast-other-model", dim=768)
        provider = _make_provider(vector_name="fast-x", dim=4)

        point_id, created_at, error = _run(
            mb.remember_point(
                client, provider, "col",
                summary="s", description="d", kind="lesson",
                repo="my-project", embedding_model="model-x",
            )
        )
        self.assertIsNone(point_id)
        self.assertIsNone(created_at)
        self.assertIsNotNone(error)
        self.assertIn("Error", error)
        client.upsert.assert_not_called()
        client.set_payload.assert_not_called()


class RecallPointsTest(_PatchRetryMixin, unittest.TestCase):
    def _client_with_empty_results(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = []
        client.query_points.return_value = response
        return client

    def test_returns_empty_list_when_collection_missing(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        provider = _make_provider()
        results = _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a"))
        self.assertEqual(results, [])
        client.query_points.assert_not_called()

    def test_default_scope_filters_caller_repo_or_general(self):
        client = self._client_with_empty_results()
        provider = _make_provider()
        _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a"))
        _, kwargs = client.query_points.call_args
        query_filter = kwargs["query_filter"]
        should_values = {c.match.value for c in query_filter.should}
        self.assertEqual(should_values, {"proj-a", mb.GENERAL_REPO})
        must_keys = {c.key for c in query_filter.must}
        self.assertIn(mb.SOURCE_FIELD, must_keys)

    def test_explicit_repo_overrides_default_scope(self):
        client = self._client_with_empty_results()
        provider = _make_provider()
        _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a", repo="proj-b"))
        _, kwargs = client.query_points.call_args
        query_filter = kwargs["query_filter"]
        self.assertIsNone(query_filter.should)
        must_values = {c.match.value for c in query_filter.must}
        self.assertIn("proj-b", must_values)
        self.assertNotIn("proj-a", must_values)

    def test_all_repos_applies_no_repo_constraint(self):
        client = self._client_with_empty_results()
        provider = _make_provider()
        _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a", all_repos=True))
        _, kwargs = client.query_points.call_args
        query_filter = kwargs["query_filter"]
        self.assertIsNone(query_filter.should)
        must_keys = [c.key for c in query_filter.must]
        self.assertNotIn(mb.REPO_FIELD, must_keys)

    def test_kind_narrows_further(self):
        client = self._client_with_empty_results()
        provider = _make_provider()
        _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a", kind="idea"))
        _, kwargs = client.query_points.call_args
        must_kind_values = {c.match.value for c in kwargs["query_filter"].must if c.key == mb.KIND_FIELD}
        self.assertEqual(must_kind_values, {"idea"})

    def test_excludes_pending_points(self):
        # Regression for PR #178 review (seventh pass): a point is visible to
        # search the instant remember_point's initial upsert returns, before
        # its follow-up call clears pending -- without this exclusion, recall
        # could surface a record that wipe_memory_bank can never remove
        # (pending points are always wipe-ineligible).
        client = self._client_with_empty_results()
        provider = _make_provider()
        _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a"))
        _, kwargs = client.query_points.call_args
        query_filter = kwargs["query_filter"]
        self.assertEqual(len(query_filter.must_not), 1)
        cond = query_filter.must_not[0]
        self.assertEqual(cond.key, mb.PENDING_FIELD)
        self.assertEqual(cond.match.value, True)

    def test_maps_hit_fields_from_payload(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        point = MagicMock()
        point.id = "abc"
        point.score = 0.9
        point.payload = {
            "document": "summary text",
            "metadata": {
                "description": "full text",
                "kind": "lesson",
                "repo": "proj-a",
                "embedding_model": "model-x",
                "created_at": 1700000000.0,
            },
        }
        response = MagicMock()
        response.points = [point]
        client.query_points.return_value = response
        provider = _make_provider()
        results = _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a"))
        self.assertEqual(len(results), 1)
        hit = results[0]
        self.assertEqual(hit["summary"], "summary text")
        self.assertEqual(hit["description"], "full text")
        self.assertEqual(hit["kind"], "lesson")
        self.assertEqual(hit["repo"], "proj-a")
        self.assertEqual(hit["embedding_model"], "model-x")
        self.assertEqual(hit["score"], 0.9)
        self.assertEqual(hit["created_at"], 1700000000.0)

        # Issue #177: no metadata.weight on this point -> defaults to
        # DEFAULT_WEIGHT (1.0), a true no-op, and effective_score is the
        # NORMALIZED score (raw similarity rescaled to [0, 1], PR #199
        # review) scaled by that default -- not the raw score directly.
        self.assertEqual(hit["weight"], mb.DEFAULT_WEIGHT)
        self.assertEqual(hit["effective_score"], mb._normalize_similarity(0.9) * mb.DEFAULT_WEIGHT)

    def test_created_at_is_none_for_a_legacy_point_missing_the_field(self):
        # Issue #179: a point written before metadata.created_at existed at
        # all must not fabricate a value -- None is the honest signal that
        # this memory predates the field, not "created at time zero."
        client = MagicMock()
        client.collection_exists.return_value = True
        point = MagicMock()
        point.id = "legacy"
        point.score = 0.5
        point.payload = {
            "document": "summary text",
            "metadata": {
                "description": "full text",
                "kind": "lesson",
                "repo": "proj-a",
                "embedding_model": "model-x",
            },
        }
        response = MagicMock()
        response.points = [point]
        client.query_points.return_value = response
        provider = _make_provider()
        results = _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a"))
        self.assertIsNone(results[0]["created_at"])


def _hit_point(point_id, score, weight=None):
    """A MagicMock query_points hit with the given raw score/weight, for
    exercising recall_points' weight-based re-ranking (issue #177)."""
    point = MagicMock()
    point.id = point_id
    point.score = score
    metadata = {
        "description": f"desc-{point_id}",
        "kind": "lesson",
        "repo": "proj-a",
        "embedding_model": "model-x",
    }
    if weight is not None:
        metadata["weight"] = weight
    point.payload = {"document": f"summary-{point_id}", "metadata": metadata}
    return point


class NormalizeSimilarityTest(unittest.TestCase):
    """Issue #177/#272: raw cosine similarity is floored at 0 (no offset)
    BEFORE being multiplied by weight -- multiplying a possibly-negative raw
    score directly would invert the weight semantics, while a (raw+1)/2
    rescale gives every realistic hit a 0.5 floor that makes weight>=2.0 a
    ranking takeover."""

    def test_floors_negatives_and_passes_nonnegatives_through(self):
        self.assertEqual(mb._normalize_similarity(-1.0), 0.0)
        self.assertEqual(mb._normalize_similarity(-0.1), 0.0)
        self.assertEqual(mb._normalize_similarity(0.0), 0.0)
        self.assertEqual(mb._normalize_similarity(0.3), 0.3)
        self.assertEqual(mb._normalize_similarity(1.0), 1.0)


class RecallPointsWeightRerankingTest(_PatchRetryMixin, unittest.TestCase):
    """Issue #177: recall_points over-fetches by raw similarity, re-ranks by
    effective_score = normalize_similarity(score) * weight, then truncates
    to `limit`."""

    def test_overfetches_by_configured_multiplier(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = []
        client.query_points.return_value = response
        provider = _make_provider()
        _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a", limit=5))
        _, kwargs = client.query_points.call_args
        self.assertEqual(kwargs["limit"], 5 * mb._RECALL_OVERFETCH_MULTIPLIER)

    def test_higher_weight_outranks_higher_raw_similarity(self):
        # "b" has the higher RAW similarity (0.9) but weight=1.0; "a" has
        # lower raw similarity (0.5) but a much higher weight (3.0) --
        # effective_score (0.5*3.0=1.5 vs 0.9*1.0=0.9)
        # must put "a" first, which raw-similarity-only ranking would never do.
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = [
            _hit_point("b", score=0.9, weight=1.0),
            _hit_point("a", score=0.5, weight=3.0),
        ]
        client.query_points.return_value = response
        provider = _make_provider()
        results = _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a", limit=5))
        self.assertEqual([r["id"] for r in results], ["a", "b"])
        self.assertEqual(results[0]["effective_score"], 0.5 * 3.0)
        self.assertEqual(results[1]["effective_score"], 0.9 * 1.0)

    def test_missing_weight_on_a_hit_defaults_to_1_0(self):
        # A legacy point with no metadata.weight at all must default to
        # DEFAULT_WEIGHT (1.0), not an implicit penalty or boost -- its
        # effective_score is normalize_similarity(score) * 1.0 (PR #199
        # review, third pass: NOT the raw score directly, since weight=1.0
        # is a no-op for RANKING among equally-weighted hits, not for the
        # raw score's own numeric value -- see recall_points' docstring).
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = [_hit_point("legacy", score=0.7, weight=None)]
        client.query_points.return_value = response
        provider = _make_provider()
        results = _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a"))
        self.assertEqual(results[0]["weight"], mb.DEFAULT_WEIGHT)
        self.assertEqual(results[0]["effective_score"], mb._normalize_similarity(0.7))

    def test_negative_raw_similarity_does_not_invert_weight_ordering(self):
        # Regression for PR #199 review: cosine similarity CAN be negative.
        # Multiplying a negative raw score directly by weight would let a
        # weight=0 "de-emphasized" hit (score=-0.8 -> -0.8*0=0) rank ABOVE a
        # normal weight=1 hit with a less-negative raw score (score=-0.1 ->
        # -0.1*1=-0.1). Flooring at 0 makes both score exactly 0; the tie is
        # broken by raw score (issue #272), so "normal" (-0.1) still ranks
        # above "stale-deemphasized" (-0.8).
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = [
            _hit_point("stale-deemphasized", score=-0.8, weight=0.0),
            _hit_point("normal", score=-0.1, weight=1.0),
        ]
        client.query_points.return_value = response
        provider = _make_provider()
        results = _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a"))
        self.assertEqual([r["id"] for r in results], ["normal", "stale-deemphasized"])
        self.assertEqual(results[1]["effective_score"], 0.0)

    def test_weight_zero_ranks_below_positive_similarity_default_hit(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = [
            _hit_point("stale", score=0.95, weight=0.0),
            _hit_point("normal", score=0.05, weight=1.0),
        ]
        client.query_points.return_value = response
        results = _run(mb.recall_points(client, _make_provider(), "col", "query", caller_repo="proj-a"))
        self.assertEqual([r["id"] for r in results], ["normal", "stale"])

    def test_weight_zero_is_last_even_against_negative_similarity_tie(self):
        # PR #359 review: both score effective 0; raw-score tie-break alone
        # would put the weight=0 hit (raw 0.95) above the default hit (-0.1).
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = [
            _hit_point("stale", score=0.95, weight=0.0),
            _hit_point("neg-default", score=-0.1, weight=1.0),
        ]
        client.query_points.return_value = response
        results = _run(mb.recall_points(client, _make_provider(), "col", "query", caller_repo="proj-a"))
        self.assertEqual([r["id"] for r in results], ["neg-default", "stale"])

    def test_default_weight_hits_keep_plain_similarity_order_including_negatives(self):
        # Constraint from #272: default-weight ordering must not shift. All
        # negative scores floor to effective 0, so the raw-score tie-break
        # has to preserve their relative order.
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = [
            _hit_point("neg-low", score=-0.7, weight=None),
            _hit_point("pos", score=0.2, weight=1.0),
            _hit_point("neg-high", score=-0.1, weight=1.0),
        ]
        client.query_points.return_value = response
        results = _run(mb.recall_points(client, _make_provider(), "col", "query", caller_repo="proj-a"))
        self.assertEqual([r["id"] for r in results], ["pos", "neg-high", "neg-low"])

    def test_high_weight_low_similarity_does_not_beat_high_similarity_default(self):
        # Issue #272 regression: under the old (raw+1)/2 rescale,
        # weight=2.0/raw=0.1 scored 2.0*0.55=1.1 and beat weight=1.0/raw=0.9
        # (0.95). Now it is 0.2 vs 0.9.
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = [
            _hit_point("relevant", score=0.9, weight=1.0),
            _hit_point("boosted-irrelevant", score=0.1, weight=2.0),
            _hit_point("unrelated-boosted", score=0.0, weight=2.0),
        ]
        client.query_points.return_value = response
        results = _run(mb.recall_points(client, _make_provider(), "col", "query", caller_repo="proj-a"))
        self.assertEqual([r["id"] for r in results], ["relevant", "boosted-irrelevant", "unrelated-boosted"])
        self.assertEqual(results[2]["effective_score"], 0.0)

    def test_weight_is_a_boost_among_comparable_similarity(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = [
            _hit_point("plain", score=0.80, weight=1.0),
            _hit_point("boosted", score=0.70, weight=1.5),
        ]
        client.query_points.return_value = response
        results = _run(mb.recall_points(client, _make_provider(), "col", "query", caller_repo="proj-a"))
        self.assertEqual([r["id"] for r in results], ["boosted", "plain"])

    def test_score_field_stays_raw_similarity(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = [_hit_point("n", score=-0.4, weight=1.0)]
        client.query_points.return_value = response
        results = _run(mb.recall_points(client, _make_provider(), "col", "query", caller_repo="proj-a"))
        self.assertEqual(results[0]["score"], -0.4)
        self.assertEqual(results[0]["effective_score"], 0.0)

    def test_truncates_reranked_results_to_limit(self):
        # More over-fetched candidates than `limit` -- final list must be
        # exactly `limit` long, in effective_score order, even though the
        # winner by weight arrives from Qdrant sorted LAST by raw similarity.
        client = MagicMock()
        client.collection_exists.return_value = True
        response = MagicMock()
        response.points = [
            _hit_point("hi-raw-1", score=0.95, weight=1.0),
            _hit_point("hi-raw-2", score=0.90, weight=1.0),
            _hit_point("hi-raw-3", score=0.85, weight=1.0),
            _hit_point("low-raw-high-weight", score=0.10, weight=20.0),
        ]
        client.query_points.return_value = response
        provider = _make_provider()
        results = _run(mb.recall_points(client, provider, "col", "query", caller_repo="proj-a", limit=2))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "low-raw-high-weight")
        self.assertEqual(results[1]["id"], "hi-raw-1")


class CountMemoryBankPointsTest(_PatchRetryMixin, unittest.TestCase):
    def test_returns_zero_when_collection_missing(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        self.assertEqual(mb.count_memory_bank_points(client, "col"), 0)
        client.count.assert_not_called()

    def test_unscoped_counts_all_memory_bank_points(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.count.return_value = MagicMock(count=7)
        result = mb.count_memory_bank_points(client, "col")
        self.assertEqual(result, 7)
        _, kwargs = client.count.call_args
        must_keys = [c.key for c in kwargs["count_filter"].must]
        self.assertEqual(must_keys, [mb.SOURCE_FIELD])

    def test_repo_scoped_adds_repo_condition(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.count.return_value = MagicMock(count=2)
        mb.count_memory_bank_points(client, "col", repo="proj-a")
        _, kwargs = client.count.call_args
        must = kwargs["count_filter"].must
        self.assertEqual({c.key for c in must}, {mb.SOURCE_FIELD, mb.REPO_FIELD})

    def test_excludes_pending_points(self):
        # Regression for PR #178 review (seventh pass): wipe_memory_bank's
        # dry-run (confirm=False) calls this function, while its confirmed
        # path uses _own_repo_filter -- which already excludes pending. Without
        # this exclusion here too, the dry-run count and the immediately
        # following confirmed delete could disagree on the exact same snapshot.
        client = MagicMock()
        client.collection_exists.return_value = True
        client.count.return_value = MagicMock(count=1)
        mb.count_memory_bank_points(client, "col", repo="proj-a")
        _, kwargs = client.count.call_args
        must_not = kwargs["count_filter"].must_not
        self.assertEqual(len(must_not), 1)
        self.assertEqual(must_not[0].key, mb.PENDING_FIELD)
        self.assertEqual(must_not[0].match.value, True)


class ForgetPointTest(_PatchRetryMixin, unittest.TestCase):
    def _record(self, source=mb.MEMORY_BANK_SOURCE, repo="proj-a"):
        rec = MagicMock()
        rec.payload = {"metadata": {"source": source, "repo": repo}}
        return rec

    def test_collection_missing(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        result = mb.forget_point(client, "col", "id1", caller_repo="proj-a", confirm=False)
        self.assertIn("Error", result)

    def test_point_not_found(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.retrieve.return_value = []
        result = mb.forget_point(client, "col", "id1", caller_repo="proj-a", confirm=False)
        self.assertIn("Error", result)
        client.delete.assert_not_called()

    def test_malformed_point_id_treated_as_not_found(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.retrieve.side_effect = RuntimeError("400 Bad Request: not a valid point ID")
        result = mb.forget_point(client, "col", "not-a-uuid", caller_repo="proj-a", confirm=False)
        self.assertIn("Error", result)
        client.delete.assert_not_called()

    def test_refuses_non_memory_bank_point(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.retrieve.return_value = [self._record(source="something-else")]
        result = mb.forget_point(client, "col", "id1", caller_repo="proj-a", confirm=False)
        self.assertIn("not a memory-bank point", result)
        client.delete.assert_not_called()

    def test_same_repo_deletes_without_confirm(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.retrieve.return_value = [self._record(repo="proj-a")]
        result = mb.forget_point(client, "col", "id1", caller_repo="proj-a", confirm=False)
        self.assertIn("Deleted", result)
        client.delete.assert_called_once()

    def test_cross_repo_requires_confirm(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.retrieve.return_value = [self._record(repo="proj-b")]
        result = mb.forget_point(client, "col", "id1", caller_repo="proj-a", confirm=False)
        self.assertIn("confirm", result.lower())
        self.assertNotIn("Deleted", result)
        client.delete.assert_not_called()

    def test_cross_repo_with_confirm_deletes(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.retrieve.return_value = [self._record(repo="proj-b")]
        result = mb.forget_point(client, "col", "id1", caller_repo="proj-a", confirm=True)
        self.assertIn("Deleted", result)
        client.delete.assert_called_once()

    def test_general_tagged_point_treated_as_cross_repo(self):
        """A "general" point is never "this project's own" -- deleting it
        needs confirm=True the same as any other different repo."""
        client = MagicMock()
        client.collection_exists.return_value = True
        client.retrieve.return_value = [self._record(repo=mb.GENERAL_REPO)]
        result = mb.forget_point(client, "col", "id1", caller_repo="proj-a", confirm=False)
        self.assertNotIn("Deleted", result)


def _scroll_page(ids, next_offset=None):
    records = [MagicMock(id=i) for i in ids]
    return records, next_offset


class WipeMemoryBankTest(_PatchRetryMixin, unittest.TestCase):
    def test_without_confirm_only_counts(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.count.return_value = MagicMock(count=3)
        count, deleted = mb.wipe_memory_bank(client, "col", caller_repo="proj-a", confirm=False)
        self.assertEqual(count, 3)
        self.assertFalse(deleted)
        client.delete.assert_not_called()
        client.scroll.assert_not_called()

    def test_confirmed_deletes_exact_snapshotted_ids_scoped_to_own_repo(self):
        # Regression for PR #178 review: the confirmed path now snapshots
        # exact point ids via scroll() and deletes precisely those, rather
        # than counting once and then deleting by a live filter (a separate,
        # later request that could pick up a point inserted in between).
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = _scroll_page(["a", "b", "c"], next_offset=None)
        count, deleted = mb.wipe_memory_bank(client, "col", caller_repo="proj-a", confirm=True)
        self.assertEqual(count, 3)
        self.assertTrue(deleted)
        client.delete.assert_called_once()
        _, kwargs = client.delete.call_args
        self.assertEqual(set(kwargs["points_selector"].points), {"a", "b", "c"})
        _, scroll_kwargs = client.scroll.call_args
        scoped = scroll_kwargs["scroll_filter"].must
        values = {c.match.value for c in scoped}
        self.assertEqual(values, {mb.MEMORY_BANK_SOURCE, "proj-a"})

    def test_confirmed_paginates_through_scroll(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.side_effect = [
            _scroll_page(["a", "b"], next_offset="page2"),
            _scroll_page(["c"], next_offset=None),
        ]
        count, deleted = mb.wipe_memory_bank(client, "col", caller_repo="proj-a", confirm=True)
        self.assertEqual(count, 3)
        self.assertTrue(deleted)
        self.assertEqual(client.scroll.call_count, 2)
        _, kwargs = client.delete.call_args
        self.assertEqual(set(kwargs["points_selector"].points), {"a", "b", "c"})

    def test_confirmed_but_nothing_matches_skips_delete_call(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = _scroll_page([], next_offset=None)
        count, deleted = mb.wipe_memory_bank(client, "col", caller_repo="proj-a", confirm=True)
        self.assertEqual(count, 0)
        self.assertTrue(deleted)
        client.delete.assert_not_called()

    def test_confirmed_on_nonexistent_collection_skips_scroll_and_delete(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        count, deleted = mb.wipe_memory_bank(client, "col", caller_repo="proj-a", confirm=True)
        self.assertEqual(count, 0)
        self.assertTrue(deleted)
        client.scroll.assert_not_called()

    def test_confirmed_scroll_filter_includes_created_at_cutoff(self):
        # Regression for PR #178 review (second wipe-related pass): scroll()
        # has no cross-page snapshot guarantee, and point ids are random
        # UUIDs (not insertion-ordered), so a remember() landing mid-scroll
        # of a large (>1 page) wipe could otherwise sort into a not-yet-
        # visited page and get swept up despite being created after the wipe
        # began. A created_at cutoff captured before scrolling starts closes
        # that regardless of paging -- but must still match points that
        # PREDATE this field's existence (via IsEmptyCondition), or every
        # memory written before this fix would become permanently
        # unwipeable via wipe_all.
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = _scroll_page(["a"], next_offset=None)
        before = time.time()
        mb.wipe_memory_bank(client, "col", caller_repo="proj-a", confirm=True)
        after = time.time()
        _, scroll_kwargs = client.scroll.call_args
        scroll_filter = scroll_kwargs["scroll_filter"]
        should = scroll_filter.should
        self.assertEqual(len(should), 2)
        range_conditions = [c for c in should if getattr(c, "range", None) is not None]
        self.assertEqual(len(range_conditions), 1)
        self.assertEqual(range_conditions[0].key, mb.CREATED_AT_FIELD)
        self.assertLessEqual(before, range_conditions[0].range.lte)
        self.assertLessEqual(range_conditions[0].range.lte, after)
        empty_conditions = [c for c in should if isinstance(c, models.IsEmptyCondition)]
        self.assertEqual(len(empty_conditions), 1)
        self.assertEqual(empty_conditions[0].is_empty.key, mb.CREATED_AT_FIELD)

    def test_confirmed_scroll_filter_excludes_pending_points(self):
        # Regression for PR #178 review (sixth pass): the created_at cutoff
        # above closes the mid-scroll-pagination gap, but a point can still
        # be mid-write (upserted with pending=True, created_at not yet
        # stamped) when a wipe's cutoff is captured. Excluding pending==True
        # unconditionally closes that follow-on gap regardless of timing.
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = _scroll_page(["a"], next_offset=None)
        mb.wipe_memory_bank(client, "col", caller_repo="proj-a", confirm=True)
        _, scroll_kwargs = client.scroll.call_args
        scroll_filter = scroll_kwargs["scroll_filter"]
        self.assertEqual(len(scroll_filter.must_not), 1)
        cond = scroll_filter.must_not[0]
        self.assertEqual(cond.key, mb.PENDING_FIELD)
        self.assertEqual(cond.match.value, True)


class ScrollCollectionTest(_PatchRetryMixin, unittest.TestCase):
    @staticmethod
    def _rec(pid, payload):
        r = MagicMock()
        r.id = pid
        r.payload = payload
        return r

    @staticmethod
    def _good(repo="proj-a", **extra):
        return {"document": "sum", "metadata": {"repo": repo, "description": "d", "kind": "lesson", **extra}}

    def test_missing_collection_reports_error_and_never_scrolls(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        points, errors = mb.scroll_collection(client, "gone")
        self.assertEqual(points, [])
        self.assertEqual(len(errors), 1)
        client.scroll.assert_not_called()

    def test_follows_next_offset_until_none(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.side_effect = [
            ([self._rec("a", self._good())], "off1"),
            ([self._rec("b", self._good())], "off2"),
            ([self._rec("c", self._good())], None),
        ]
        points, errors = mb.scroll_collection(client, "col", batch_size=1)
        self.assertEqual([p["id"] for p in points], ["a", "b", "c"])
        self.assertEqual(errors, [])
        offsets = [c.kwargs["offset"] for c in client.scroll.call_args_list]
        self.assertEqual(offsets, [None, "off1", "off2"])
        for c in client.scroll.call_args_list:
            self.assertEqual(c.kwargs["limit"], 1)
            self.assertIs(c.kwargs["with_payload"], True)
            self.assertIs(c.kwargs["with_vectors"], False)
            self.assertNotIn("scroll_filter", c.kwargs)

    def test_non_memory_bank_shaped_points_are_errors_not_crashes(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = (
            [
                self._rec("ok", self._good()),
                self._rec("no-repo", {"document": "x", "metadata": {"kind": "lesson"}}),
                self._rec("no-meta", {"document": "x"}),
                self._rec("no-payload", None),
            ],
            None,
        )
        points, errors = mb.scroll_collection(client, "col")
        self.assertEqual([p["id"] for p in points], ["ok"])
        self.assertEqual({e["id"] for e in errors}, {"no-repo", "no-meta", "no-payload"})

    def test_weight_defaults_when_absent_and_fields_mapped(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = (
            [self._rec("a", self._good(weight=0.0, created_at=5.0)), self._rec("b", self._good())],
            None,
        )
        points, _ = mb.scroll_collection(client, "col")
        self.assertEqual(points[0]["weight"], 0.0)
        self.assertEqual(points[0]["created_at"], 5.0)
        self.assertEqual(points[0]["summary"], "sum")
        self.assertEqual(points[1]["weight"], mb.DEFAULT_WEIGHT)


class TransferPointIdTest(unittest.TestCase):
    def test_deterministic_and_source_specific(self):
        a = mb.transfer_point_id("legacy", "p1")
        self.assertEqual(a, mb.transfer_point_id("legacy", "p1"))
        self.assertNotEqual(a, mb.transfer_point_id("legacy", "p2"))
        self.assertNotEqual(a, mb.transfer_point_id("other-legacy", "p1"))

    def test_is_a_valid_uuid_string(self):
        import uuid
        value = mb.transfer_point_id("legacy", 42)
        self.assertEqual(str(uuid.UUID(value)), value)

    def test_namespace_is_pinned(self):
        # Changing the namespace would make every earlier transfer look new on
        # a re-run and duplicate it -- pin one concrete derived value.
        self.assertEqual(
            mb.transfer_point_id("legacy", "p1"),
            str(__import__("uuid").uuid5(mb.TRANSFER_ID_NAMESPACE, "legacy:p1")),
        )
        self.assertEqual(str(mb.TRANSFER_ID_NAMESPACE), "6f1c2b7e-4d3a-5e8f-9a0b-1c2d3e4f5a6b")


class TransferPointsTest(_PatchRetryMixin, unittest.TestCase):
    """Issue #336: transfer_points' repo boundary, check-before-write dedup,
    dry-run, and per-point error handling. remember_point is mocked -- its own
    write behavior is covered by RememberPointTest above."""

    SOURCE = "legacy-memory-bank"
    TARGET = "memory-bank"
    ME = "proj-a"

    @staticmethod
    def _rec(pid, repo, **meta):
        r = MagicMock()
        r.id = pid
        doc = meta.pop("document", "sum-" + str(pid))
        r.payload = {"document": doc, "metadata": {"repo": repo, "description": "d", "kind": "lesson", **meta}}
        return r

    def _client(self, records, target_exists=True, present_source_ids=(), source_exists=True):
        client = MagicMock()
        client.collection_exists.side_effect = lambda name: source_exists if name == self.SOURCE else target_exists
        client.scroll.return_value = (records, None)
        present = {mb.transfer_point_id(self.SOURCE, sid) for sid in present_source_ids}

        def _retrieve(collection_name, ids, **kw):
            out = []
            for i in ids:
                if i in present:
                    r = MagicMock()
                    r.id = i
                    out.append(r)
            return out

        client.retrieve.side_effect = _retrieve
        return client

    def _transfer(self, client, dry_run, remember=None):
        remember = remember or AsyncMock(return_value=("id", 1.0, None))
        with patch.object(mb, "remember_point", new=remember):
            report = _run(mb.transfer_points(
                client, _make_provider(), self.SOURCE, self.TARGET,
                caller_repo=self.ME, embedding_model="m", dry_run=dry_run,
            ))
        return report, remember

    def test_repo_mapping_buckets(self):
        records = [
            self._rec("own1", self.ME), self._rec("own2", self.ME),
            self._rec("gen1", mb.GENERAL_REPO),
            self._rec("f1", "proj-b"), self._rec("f2", "proj-b"), self._rec("f3", "proj-c"),
        ]
        report, remember = self._transfer(self._client(records), dry_run=False)
        self.assertEqual(report["migrated"], {"general": 1, "project": 2})
        self.assertEqual(report["skipped_foreign_repo"], {"proj-b": 2, "proj-c": 1})
        self.assertEqual(report["errors"], [])
        self.assertNotIn("error", report)
        written_repos = sorted(c.kwargs["repo"] for c in remember.call_args_list)
        self.assertEqual(written_repos, [mb.GENERAL_REPO, self.ME, self.ME])

    def test_foreign_points_are_never_written_or_looked_up(self):
        client = self._client([self._rec("f1", "proj-b")])
        report, remember = self._transfer(client, dry_run=False)
        remember.assert_not_called()
        client.retrieve.assert_not_called()
        self.assertEqual(report["migrated"], {"general": 0, "project": 0})

    def test_dry_run_writes_nothing_and_reports_would_migrate(self):
        records = [self._rec("own1", self.ME), self._rec("gen1", mb.GENERAL_REPO)]
        report, remember = self._transfer(self._client(records), dry_run=True)
        remember.assert_not_called()
        self.assertEqual(report["would_migrate"], {"general": 1, "project": 1})
        self.assertNotIn("migrated", report)
        self.assertIs(report["dry_run"], True)

    def test_dry_run_needs_no_embedding_provider(self):
        report = _run(mb.transfer_points(
            self._client([self._rec("own1", self.ME)]), None, self.SOURCE, self.TARGET,
            caller_repo=self.ME, embedding_model="m", dry_run=True,
        ))
        self.assertEqual(report["would_migrate"]["project"], 1)

    def test_already_present_points_are_skipped_not_rewritten(self):
        records = [self._rec("own1", self.ME), self._rec("own2", self.ME)]
        client = self._client(records, present_source_ids=["own1"])
        report, remember = self._transfer(client, dry_run=False)
        self.assertEqual(report["already_present"], 1)
        self.assertEqual(report["migrated"]["project"], 1)
        remember.assert_called_once()
        self.assertEqual(remember.call_args.kwargs["point_id"], mb.transfer_point_id(self.SOURCE, "own2"))

    def test_dry_run_also_reports_already_present(self):
        client = self._client([self._rec("own1", self.ME)], present_source_ids=["own1"])
        report, _ = self._transfer(client, dry_run=True)
        self.assertEqual(report["already_present"], 1)
        self.assertEqual(report["would_migrate"]["project"], 0)

    def test_missing_target_skips_retrieve(self):
        client = self._client([self._rec("own1", self.ME)], target_exists=False)
        report, _ = self._transfer(client, dry_run=False)
        client.retrieve.assert_not_called()
        self.assertEqual(report["migrated"]["project"], 1)

    def test_written_fields_come_from_the_source_point(self):
        records = [self._rec("own1", self.ME, document="the summary", description="the desc", kind="decision", weight=2.0)]
        _, remember = self._transfer(self._client(records), dry_run=False)
        kw = remember.call_args.kwargs
        self.assertEqual(kw["summary"], "the summary")
        self.assertEqual(kw["description"], "the desc")
        self.assertEqual(kw["kind"], "decision")
        self.assertEqual(kw["weight"], 2.0)
        self.assertEqual(kw["embedding_model"], "m")
        self.assertEqual(remember.call_args.args[2], self.TARGET)

    def test_invalid_fields_go_to_errors_not_the_target(self):
        records = [
            self._rec("empty-summary", self.ME, document="  "),
            self._rec("no-kind", self.ME, kind=None),
            self._rec("bad-desc", self.ME, description=5),
            self._rec("neg-weight", self.ME, weight=-1),
            self._rec("nan-weight", self.ME, weight=float("nan")),
            self._rec("str-weight", self.ME, weight="2"),
            self._rec("bool-weight", self.ME, weight=True),
            self._rec("ok", self.ME),
        ]
        report, remember = self._transfer(self._client(records), dry_run=False)
        self.assertEqual(
            {e["id"] for e in report["errors"]},
            {"empty-summary", "no-kind", "bad-desc", "neg-weight", "nan-weight", "str-weight", "bool-weight"},
        )
        self.assertEqual(report["migrated"]["project"], 1)
        remember.assert_called_once()

    def test_scroll_errors_are_reported(self):
        bad = MagicMock()
        bad.id = "no-meta"
        bad.payload = {"document": "x"}
        report, _ = self._transfer(self._client([bad, self._rec("own1", self.ME)]), dry_run=False)
        self.assertEqual([e["id"] for e in report["errors"]], ["no-meta"])
        self.assertEqual(report["migrated"]["project"], 1)

    def test_one_failing_write_does_not_abort_the_run(self):
        records = [self._rec("own1", self.ME), self._rec("own2", self.ME)]
        remember = AsyncMock(side_effect=[RuntimeError("boom"), ("id", 1.0, None)])
        report, _ = self._transfer(self._client(records), dry_run=False, remember=remember)
        self.assertEqual(report["migrated"]["project"], 1)
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("boom", report["errors"][0]["error"])

    def test_embedding_mismatch_stops_the_run(self):
        records = [self._rec("own1", self.ME), self._rec("own2", self.ME)]
        remember = AsyncMock(return_value=(None, None, "Error: mismatch"))
        report, _ = self._transfer(self._client(records), dry_run=False, remember=remember)
        self.assertEqual(report["error"], "Error: mismatch")
        self.assertEqual(remember.call_count, 1)
        self.assertEqual(report["migrated"]["project"], 0)

    def test_missing_source_reports_error_without_scrolling(self):
        client = self._client([], source_exists=False)
        report, remember = self._transfer(client, dry_run=False)
        self.assertIn("does not exist", report["error"])
        client.scroll.assert_not_called()
        remember.assert_not_called()

    def test_source_equal_to_target_is_refused(self):
        client = self._client([])
        with patch.object(mb, "remember_point", new=AsyncMock()) as remember:
            report = _run(mb.transfer_points(
                client, _make_provider(), self.TARGET, self.TARGET,
                caller_repo=self.ME, embedding_model="m", dry_run=False,
            ))
        self.assertIn("error", report)
        client.scroll.assert_not_called()
        remember.assert_not_called()

    @staticmethod
    def _aliases(mapping):
        resp = MagicMock()
        resp.aliases = [MagicMock(alias_name=a, collection_name=c) for a, c in mapping.items()]
        return resp

    def test_source_alias_of_target_is_refused(self):
        # PR #422 review: Qdrant resolves an alias anywhere a collection name
        # goes, so an alias of memory-bank must not pass the self-transfer guard.
        client = self._client([self._rec("own1", self.ME)])
        client.get_aliases.return_value = self._aliases({self.SOURCE: self.TARGET})
        report, remember = self._transfer(client, dry_run=False)
        self.assertIn("alias", report["error"])
        client.scroll.assert_not_called()
        remember.assert_not_called()

    def test_target_alias_of_source_is_refused(self):
        client = self._client([self._rec("own1", self.ME)])
        client.get_aliases.return_value = self._aliases({self.TARGET: self.SOURCE})
        report, remember = self._transfer(client, dry_run=False)
        self.assertIn("error", report)
        client.scroll.assert_not_called()
        remember.assert_not_called()

    def test_unrelated_aliases_do_not_block(self):
        client = self._client([self._rec("own1", self.ME)])
        client.get_aliases.return_value = self._aliases({"something": "else", self.SOURCE: "legacy-v2"})
        report, _ = self._transfer(client, dry_run=False)
        self.assertNotIn("error", report)
        self.assertEqual(report["migrated"]["project"], 1)

    def test_all_operations_use_resolved_backing_collections(self):
        # PR #422 review (second pass): after the alias check, every Qdrant
        # call must use the resolved backing names, so repointing an alias
        # mid-run can't redirect the scroll or the writes.
        client = self._client([self._rec("own1", self.ME)])
        client.collection_exists.side_effect = lambda name: True
        client.get_aliases.return_value = self._aliases({self.SOURCE: "legacy-v1", self.TARGET: "memory-bank-v2"})
        report, remember = self._transfer(client, dry_run=False)
        self.assertNotIn("error", report)
        self.assertEqual(client.scroll.call_args.kwargs["collection_name"], "legacy-v1")
        self.assertEqual(client.retrieve.call_args.kwargs["collection_name"], "memory-bank-v2")
        self.assertEqual(remember.call_args.args[2], "memory-bank-v2")
        exists_names = {c.args[0] for c in client.collection_exists.call_args_list}
        self.assertEqual(exists_names, {"legacy-v1", "memory-bank-v2"})
        # Ids still hash the REQUESTED source name, so they don't change with
        # whichever backing collection an alias happens to point at.
        self.assertEqual(remember.call_args.kwargs["point_id"], mb.transfer_point_id(self.SOURCE, "own1"))

    def test_pending_point_in_target_is_rewritten_not_counted_present(self):
        # PR #422 review (second pass): an id left pending=True by a run whose
        # set_payload failed must be finished on the next run, not counted as
        # already_present forever (pending points are hidden from recall).
        client = self._client([self._rec("own1", self.ME), self._rec("own2", self.ME)])
        stuck_id = mb.transfer_point_id(self.SOURCE, "own1")
        done_id = mb.transfer_point_id(self.SOURCE, "own2")
        client.retrieve.side_effect = lambda collection_name, ids, **kw: [
            MagicMock(id=stuck_id, payload={"metadata": {"pending": True}}),
            MagicMock(id=done_id, payload={"metadata": {"pending": False}}),
        ]
        report, remember = self._transfer(client, dry_run=False)
        self.assertEqual(report["already_present"], 1)
        self.assertEqual(report["migrated"]["project"], 1)
        remember.assert_called_once()
        self.assertEqual(remember.call_args.kwargs["point_id"], stuck_id)
        self.assertEqual(client.retrieve.call_args.kwargs["with_payload"], [mb.PENDING_FIELD])

    def test_unreadable_aliases_fail_closed(self):
        client = self._client([self._rec("own1", self.ME)])
        client.get_aliases.side_effect = RuntimeError("down")
        report, remember = self._transfer(client, dry_run=False)
        self.assertIn("aliases", report["error"])
        client.scroll.assert_not_called()
        remember.assert_not_called()

    def test_never_writes_to_or_deletes_from_the_source(self):
        records = [self._rec("own1", self.ME)]
        client = self._client(records)
        _, remember = self._transfer(client, dry_run=False)
        for call in remember.call_args_list:
            self.assertNotEqual(call.args[2], self.SOURCE)
        client.delete.assert_not_called()
        client.delete_collection.assert_not_called()
        client.upsert.assert_not_called()

    def test_rerun_after_success_writes_nothing(self):
        # End-to-end idempotency: ids the first run wrote are what the second
        # run's retrieve finds, so a second real run is all already_present.
        records = [self._rec("own1", self.ME), self._rec("gen1", mb.GENERAL_REPO)]
        _, first = self._transfer(self._client(records), dry_run=False)
        written = [c.kwargs["point_id"] for c in first.call_args_list]
        client = self._client(records)
        client.retrieve.side_effect = lambda collection_name, ids, **kw: [
            MagicMock(id=i) for i in ids if i in written
        ]
        report, second = self._transfer(client, dry_run=False)
        second.assert_not_called()
        self.assertEqual(report["already_present"], 2)


if __name__ == "__main__":
    unittest.main()
