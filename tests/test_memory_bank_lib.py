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
        self.assertEqual(len(f.must_not), 1)
        cond = f.must_not[0]
        self.assertEqual(cond.key, mb.SOURCE_FIELD)
        self.assertEqual(cond.match.value, mb.MEMORY_BANK_SOURCE)


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

        point_id, error = _run(
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

    def test_schema_mismatch_returns_error_without_upserting(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = _matching_collection_info(vector_name="fast-other-model", dim=768)
        provider = _make_provider(vector_name="fast-x", dim=4)

        point_id, error = _run(
            mb.remember_point(
                client, provider, "col",
                summary="s", description="d", kind="lesson",
                repo="my-project", embedding_model="model-x",
            )
        )
        self.assertIsNone(point_id)
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


if __name__ == "__main__":
    unittest.main()
