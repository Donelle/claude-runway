#!/usr/bin/env python3
"""Tests for libs/qdrant_code_index_marker.py: the codebase-indexer's own
collection mark and the one ownership guard index/sync/CLI/description sync
share.

The fake client (tests/fake_ownership_client.py) answers `scroll` by
evaluating the filter the guard sends against a list of payloads, the way
Qdrant would, so each case is decided by the points a collection holds rather
than by a canned answer.

Stdlib-only (unittest, no pytest), no network.

    .venv/bin/python -m unittest discover -s tests
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # fake_ownership_client

import qdrant_code_index_marker as marker  # noqa: E402
from fake_ownership_client import (  # noqa: E402
    FakeOwnershipClient as FakeClient,
    code_chunk as _code_chunk,
    compact_point as _compact,
    memory_point as _memory,
    no_retry as _no_retry,
    unrelated_point,
)
from qdrant_compact_marker import MARKER_KEY as COMPACT_KEY, MARKER_VALUE as COMPACT_VALUE  # noqa: E402


def _check(client):
    with redirect_stderr(io.StringIO()) as err:
        result = marker.check_code_index_ownership(client, "coll", retry=_no_retry)
    return result, err.getvalue()


class MarkHelpers(unittest.TestCase):
    def test_metadata_carries_the_mark_and_is_a_fresh_dict(self):
        first = marker.code_index_metadata()
        self.assertTrue(marker.is_marked_code_index(first))
        first["other"] = 1
        self.assertEqual(marker.code_index_metadata(), {marker.MARKER_KEY: marker.MARKER_VALUE})

    def test_is_marked_rejects_none_and_other_values(self):
        self.assertFalse(marker.is_marked_code_index(None))
        self.assertFalse(marker.is_marked_code_index({}))
        self.assertFalse(marker.is_marked_code_index({marker.MARKER_KEY: "something-else"}))
        self.assertFalse(marker.is_marked_code_index({COMPACT_KEY: COMPACT_VALUE}))

    def test_key_is_not_the_compact_markers_key(self):
        # The compact key means "hide from search"; a code index must never carry it.
        self.assertNotEqual(marker.MARKER_KEY, COMPACT_KEY)


class OwnershipAccepted(unittest.TestCase):
    def test_missing_collection_is_accepted_without_writing(self):
        client = FakeClient(exists=False)
        result, _ = _check(client)
        self.assertIsNone(result.error)
        self.assertFalse(result.adopted)
        self.assertEqual(client.calls, ["collection_exists"])

    def test_marked_collection_is_accepted_without_writing(self):
        client = FakeClient(metadata={"description": "d", **marker.code_index_metadata()}, payloads=[_memory()])
        result, _ = _check(client)
        self.assertIsNone(result.error)
        self.assertFalse(result.adopted)
        self.assertEqual(client.updates, [])
        self.assertNotIn("scroll", client.calls)

    def test_unmarked_code_collection_is_adopted(self):
        # The issue's step-3 decision: indexed file chunks and nothing protected.
        # A cb_indexer_store_repo note (no file_path, no source) does not block it.
        note = {"document": "n", "metadata": {"topic": "x"}}
        client = FakeClient(metadata={"description": "d"}, payloads=[note, _code_chunk()])
        result, err = _check(client)
        self.assertIsNone(result.error)
        self.assertTrue(result.adopted)
        # Only the mark is sent, relying on Qdrant's metadata merge.
        self.assertEqual(client.updates, [marker.code_index_metadata()])
        self.assertIn("adopted 'coll'", err)


class OwnershipRefused(unittest.TestCase):
    def _assert_refused(self, client, fragment):
        result, _ = _check(client)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("'coll'", result.error)
        self.assertIn(fragment, result.error)
        self.assertFalse(result.adopted)
        self.assertEqual(client.updates, [])

    def test_memory_bank_collection_under_any_name(self):
        self._assert_refused(FakeClient(payloads=[_memory(), _memory()]), "not created by the codebase-indexer")

    def test_memory_bank_points_alongside_code_chunks_are_not_adopted(self):
        self._assert_refused(FakeClient(payloads=[_code_chunk(), _memory()]), "memory-bank")

    def test_marked_compact_collection(self):
        # Refused on the marker alone, even if it somehow held file chunks.
        client = FakeClient(metadata={COMPACT_KEY: COMPACT_VALUE}, payloads=[_code_chunk()])
        self._assert_refused(client, "conversation-compact collection")

    def test_compact_marker_wins_over_the_code_index_mark(self):
        # Metadata merges keys, so one collection can carry both marks.
        client = FakeClient(metadata={COMPACT_KEY: COMPACT_VALUE, **marker.code_index_metadata()})
        self._assert_refused(client, "conversation-compact collection")

    def test_unmarked_compact_points(self):
        self._assert_refused(FakeClient(payloads=[_compact()]), "conversation-compact points")

    def test_empty_unmarked_collection(self):
        self._assert_refused(FakeClient(payloads=[]), "no indexed file chunks")

    def test_unrelated_points(self):
        client = FakeClient(metadata={"description": "x"}, payloads=[unrelated_point()])
        self._assert_refused(client, "no indexed file chunks")

    def test_every_read_failure_fails_closed(self):
        for step in ("collection_exists", "get_collection", "scroll", "update_collection"):
            with self.subTest(step=step):
                client = FakeClient(payloads=[_code_chunk()], fail_on=step)
                result, _ = _check(client)
                self.assertIsNotNone(result.error)
                assert result.error is not None
                self.assertIn("could not confirm", result.error)
                self.assertIn(f"{step} failed", result.error)
                self.assertFalse(result.adopted)

    def test_calls_go_through_retry(self):
        seen = []

        def recording_retry(fn, *args, **kwargs):
            seen.append(fn.__name__)
            return fn(*args, **kwargs)

        client = FakeClient(payloads=[_code_chunk()])
        with redirect_stderr(io.StringIO()):
            marker.check_code_index_ownership(client, "coll", retry=recording_retry)
        self.assertEqual(seen, ["collection_exists", "get_collection", "scroll", "scroll", "update_collection"])


class AgainstInProcessQdrant(unittest.TestCase):
    """The same decisions against qdrant-client's in-process engine, so the
    filters the guard sends (is-empty on metadata.file_path, a should over
    metadata.source) are checked against Qdrant's own semantics, not only
    the fake's."""

    def _collection(self, payloads, metadata=None):
        from qdrant_client import QdrantClient, models

        client = QdrantClient(":memory:")
        client.create_collection(
            "coll",
            vectors_config={"v": models.VectorParams(size=2, distance=models.Distance.COSINE)},
            metadata=metadata,
        )
        if payloads:
            client.upsert("coll", points=[
                models.PointStruct(id=i, vector={"v": [1.0, 0.0]}, payload=p) for i, p in enumerate(payloads)
            ])
        return client

    def _check(self, client):
        with redirect_stderr(io.StringIO()):
            return marker.check_code_index_ownership(client, "coll", retry=_no_retry)

    def test_decisions(self):
        cases = (
            ("code chunks are adopted", [_code_chunk(), {"document": "n", "metadata": {}}], None, True),
            ("memory bank is refused", [_memory()], None, False),
            ("memory bank plus code is refused", [_code_chunk(), _memory()], None, False),
            ("compact points are refused", [_compact()], None, False),
            ("empty is refused", [], None, False),
            ("unrelated is refused", [unrelated_point()], None, False),
            ("marked is accepted", [_memory()], marker.code_index_metadata(), True),
            ("compact marker is refused", [_code_chunk()], {COMPACT_KEY: COMPACT_VALUE}, False),
        )
        for label, payloads, metadata, accepted in cases:
            with self.subTest(label=label):
                client = self._collection(payloads, metadata)
                result = self._check(client)
                self.assertEqual(result.error is None, accepted, result.error)
                if accepted:
                    self.assertTrue(marker.is_marked_code_index(client.get_collection("coll").config.metadata))

    def test_adoption_keeps_an_existing_description(self):
        client = self._collection([_code_chunk()], {"description": "kept"})
        self.assertTrue(self._check(client).adopted)
        self.assertEqual(
            client.get_collection("coll").config.metadata,
            {"description": "kept", **marker.code_index_metadata()},
        )


if __name__ == "__main__":
    unittest.main()
