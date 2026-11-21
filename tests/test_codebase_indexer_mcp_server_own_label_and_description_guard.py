"""cb_indexer_list_collections' "own" label and the startup description sync's ownership guard.

Both are small correctness gaps: "own" was decided by collection name alone, so
the same name on a different Qdrant instance (an explicit qdrant_url override)
was mislabeled; and the description sync lacked the guard that
cb_indexer_index_repo/cb_indexer_sync_repo have, so a misconfigured COLLECTION_NAME
stamped the code-index description on the shared memory-bank collection. The
guard is now the codebase-indexer's ownership mark
(libs/qdrant_code_index_marker.py), so it covers any collection the indexer did
not create, whatever its name, not just the memory bank's.
"""
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # fake_ownership_client

from mcpservers import codebase_indexer_mcp_server as _ims  # noqa: E402
from fake_ownership_client import FakeOwnershipClient, code_chunk, memory_point, unrelated_point  # noqa: E402
from qdrant_code_index_marker import code_index_metadata  # noqa: E402
from qdrant_compact_marker import MARKER_KEY as COMPACT_KEY, MARKER_VALUE as COMPACT_VALUE  # noqa: E402

OWN_URL = "http://own-qdrant:6333"
OTHER_URL = "http://other-qdrant:6333"


def _client(names):
    client = MagicMock()
    client.get_collections.return_value.collections = [SimpleNamespace(name=n) for n in names]
    client.get_collection.side_effect = lambda n: SimpleNamespace(
        points_count=1, config=SimpleNamespace(metadata=None)
    )
    return client


class OwnLabelRequiresSameInstance(unittest.TestCase):
    def _list(self, url):
        with patch.object(_ims, "DEFAULT_COLLECTION", "proj"), \
                patch.object(_ims, "DEFAULT_QDRANT_URL", OWN_URL), \
                patch.object(_ims, "DEFAULT_QDRANT_API_KEY", None), \
                patch.object(_ims, "QdrantClient", return_value=_client(["proj"])), \
                patch.object(_ims, "get_cached_descriptions", return_value={}), \
                patch.object(_ims, "set_cached_description"):
            return _ims.list_collections(qdrant_url=url)

    def test_same_name_same_instance_is_own(self):
        self.assertIn("(this project's own)", self._list(OWN_URL))

    def test_same_name_on_a_different_instance_is_not_own(self):
        self.assertNotIn("(this project's own)", self._list(OTHER_URL))

    def test_trailing_slash_on_the_same_instance_is_still_own(self):
        self.assertIn("(this project's own)", self._list(OWN_URL + "/"))


class DescriptionSyncOnlyStampsOwnCollections(unittest.TestCase):
    """No MEMORY_BANK_COLLECTION is set or consulted: the refusal comes from
    the collection carrying no code-index mark."""

    def _sync(self, client):
        with patch.object(_ims, "DEFAULT_COLLECTION_DESCRIPTION", "desc"), \
                patch.object(_ims, "QdrantClient", return_value=client), \
                patch.object(_ims, "set_cached_description") as cached, \
                redirect_stderr(io.StringIO()) as err:
            _ims._sync_static_collection_description("shared-mb", OWN_URL, None)
        return cached, err.getvalue()

    def test_foreign_collections_are_never_stamped(self):
        for label, client in (
            ("memory bank under a non-default name", FakeOwnershipClient(payloads=[memory_point()])),
            ("marked compact collection", FakeOwnershipClient(metadata={COMPACT_KEY: COMPACT_VALUE})),
            ("empty unmarked collection", FakeOwnershipClient()),
            ("unrelated points", FakeOwnershipClient(payloads=[unrelated_point()])),
        ):
            with self.subTest(label=label):
                cached, err = self._sync(client)
                # FakeOwnershipClient records every metadata write the guard
                # makes and has no other write method, so nothing was stamped.
                self.assertEqual(client.updates, [])
                cached.assert_not_called()
                self.assertIn("not syncing COLLECTION_DESCRIPTION", err)
                self.assertIn("not created by the codebase-indexer", err)

    def test_unreadable_metadata_fails_closed(self):
        cached, err = self._sync(FakeOwnershipClient(fail_on="scroll"))
        cached.assert_not_called()
        self.assertIn("could not confirm", err)

    def test_missing_collection_is_a_no_op(self):
        client = FakeOwnershipClient(exists=False)
        cached, _ = self._sync(client)
        self.assertEqual(client.calls, ["collection_exists"])
        cached.assert_not_called()

    def test_marked_collection_still_syncs(self):
        client = MagicMock()
        client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(metadata=code_index_metadata()))
        with patch.object(_ims, "DEFAULT_COLLECTION_DESCRIPTION", "desc"), \
                patch.object(_ims, "QdrantClient", return_value=client), \
                patch.object(_ims, "set_cached_description"), \
                redirect_stderr(io.StringIO()):
            _ims._sync_static_collection_description("proj", OWN_URL, None)
        client.update_collection.assert_called_once_with(collection_name="proj", metadata={"description": "desc"})

    def test_unmarked_code_collection_is_adopted_then_stamped(self):
        client = FakeOwnershipClient(payloads=[code_chunk()])
        # After adoption the description write goes through update_collection too.
        cached, _ = self._sync(client)
        self.assertEqual(client.updates, [code_index_metadata(), {"description": "desc"}])
        cached.assert_called_once()


if __name__ == "__main__":
    unittest.main()
