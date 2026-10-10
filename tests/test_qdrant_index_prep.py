#!/usr/bin/env python3
"""Tests for libs/qdrant_index_prep.py's prepare_collection_for_index (issue
#304) -- the pre-index collection maintenance shared by index_repo and
tools/ingest_to_qdrant.py.

Stdlib-only, no network: the Qdrant client is a MagicMock and every other
collaborator is injected, so these pin which calls happen and in what order.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sys
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import httpx  # noqa: E402
from qdrant_client.http.exceptions import ResponseHandlingException  # noqa: E402

import memory_bank_lib as mb  # noqa: E402
import qdrant_index_prep as prep_mod  # noqa: E402


def _passthrough(fn, *a, **kw):
    return fn(*a, **kw)


class PrepareCollectionForIndex(unittest.TestCase):
    def _prep(self, client, *, reset, mismatch=None, guard=None, backfilled=False, retry=_passthrough):
        self.provider_factory = mock.MagicMock(name="provider_factory")
        self.model_check = mock.MagicMock(return_value=mismatch)
        self.backfill = mock.MagicMock(return_value=backfilled)
        return prep_mod.prepare_collection_for_index(
            client,
            "coll",
            "model-x",
            reset=reset,
            provider_factory=self.provider_factory,
            existing_points_guard=guard,
            retry=retry,
            model_check=self.model_check,
            backfill=self.backfill,
        )

    def test_reset_checks_model_then_does_the_filtered_delete(self):
        client = mock.MagicMock()
        client.collection_exists.return_value = True
        result = self._prep(client, reset=True)
        self.assertIsNone(result.error)
        self.assertTrue(result.reset_deleted)
        self.provider_factory.assert_called_once_with("model-x")
        self.assertIs(result.embedding_provider, self.provider_factory.return_value)
        self.assertTrue(self.model_check.call_args.kwargs["fail_closed"])
        client.delete.assert_called_once_with(
            collection_name="coll", points_selector=mb.memory_bank_exclusion_filter()
        )
        client.delete_collection.assert_not_called()
        self.backfill.assert_not_called()

    def test_reset_mismatch_returns_error_and_deletes_nothing(self):
        client = mock.MagicMock()
        client.collection_exists.return_value = True
        result = self._prep(client, reset=True, mismatch="Error: mismatch")
        self.assertEqual(result.error, "Error: mismatch")
        self.assertFalse(result.reset_deleted)
        client.delete.assert_not_called()

    def test_missing_collection_is_a_no_op_and_loads_no_model(self):
        for reset in (True, False):
            with self.subTest(reset=reset):
                client = mock.MagicMock()
                client.collection_exists.return_value = False
                result = self._prep(client, reset=reset, guard=mock.MagicMock())
                self.assertIsNone(result.error)
                self.assertIsNone(result.embedding_provider)
                self.provider_factory.assert_not_called()
                client.delete.assert_not_called()
                client.get_collection.assert_not_called()
                self.backfill.assert_not_called()

    def test_non_reset_backfills_and_never_loads_the_model(self):
        client = mock.MagicMock()
        client.collection_exists.return_value = True
        result = self._prep(client, reset=False, backfilled=True)
        self.assertIsNone(result.error)
        self.assertTrue(result.backfilled)
        self.backfill.assert_called_once_with(client, "coll")
        self.provider_factory.assert_not_called()
        # No guard given (the CLI's case): no point-count lookup either.
        client.get_collection.assert_not_called()

    def test_guard_error_aborts_before_the_backfill(self):
        client = mock.MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value.points_count = 7
        guard = mock.MagicMock(return_value="Warning: 7 points")
        result = self._prep(client, reset=False, guard=guard)
        guard.assert_called_once_with(7)
        self.assertEqual(result.error, "Warning: 7 points")
        self.backfill.assert_not_called()
        self.provider_factory.assert_not_called()

    def test_guard_passing_continues_to_the_backfill(self):
        client = mock.MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value.points_count = None  # Qdrant can report None
        guard = mock.MagicMock(return_value=None)
        result = self._prep(client, reset=False, guard=guard)
        guard.assert_called_once_with(0)
        self.assertIsNone(result.error)
        self.backfill.assert_called_once()

    def test_default_retry_survives_a_wrapped_transient_drop(self):
        """With the real call_with_retry, a qdrant-client-wrapped vpnkit drop
        on any step is retried rather than aborting the index."""
        drop = ResponseHandlingException(httpx.RemoteProtocolError("Server disconnected"))
        client = mock.MagicMock()
        client.collection_exists.side_effect = [drop, True]
        backfill = mock.MagicMock(side_effect=[drop, True])
        with mock.patch("qdrant_retry.RETRY_BACKOFF_SECONDS", 0):
            result = prep_mod.prepare_collection_for_index(
                client, "coll", "model-x", reset=False,
                provider_factory=mock.MagicMock(), backfill=backfill,
            )
        self.assertEqual(client.collection_exists.call_count, 2)
        self.assertEqual(backfill.call_count, 2)
        self.assertTrue(result.backfilled)


if __name__ == "__main__":
    unittest.main()
