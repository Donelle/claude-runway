"""list_collections' "own" label and the startup description sync's memory-bank guard.

Both are small correctness gaps: "own" was decided by collection name alone, so
the same name on a different Qdrant instance (an explicit qdrant_url override)
was mislabeled; and the description sync lacked the memory-bank-name guard that
index_repo/sync_repo have, so a COLLECTION_NAME == MEMORY_BANK_COLLECTION
misconfiguration stamped the code-index description on the shared collection.
"""
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcpservers import codebase_indexer_mcp_server as _ims  # noqa: E402

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


class DescriptionSyncSkipsMemoryBank(unittest.TestCase):
    def _sync(self, collection, memory_bank):
        client = MagicMock()
        with patch.object(_ims, "DEFAULT_COLLECTION_DESCRIPTION", "desc"), \
                patch.object(_ims, "DEFAULT_MEMORY_BANK_COLLECTION", memory_bank), \
                patch.object(_ims, "QdrantClient", return_value=client), \
                patch.object(_ims, "set_cached_description") as cached, \
                redirect_stderr(io.StringIO()) as err:
            _ims._sync_static_collection_description(collection, OWN_URL, None)
        return client, cached, err.getvalue()

    def test_memory_bank_collection_is_never_touched(self):
        client, cached, err = self._sync("shared-mb", "shared-mb")
        client.update_collection.assert_not_called()
        client.collection_exists.assert_not_called()
        cached.assert_not_called()
        self.assertIn("memory-bank", err)

    def test_ordinary_collection_still_syncs(self):
        client = MagicMock()
        client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(metadata={}))
        with patch.object(_ims, "DEFAULT_COLLECTION_DESCRIPTION", "desc"), \
                patch.object(_ims, "DEFAULT_MEMORY_BANK_COLLECTION", "shared-mb"), \
                patch.object(_ims, "QdrantClient", return_value=client), \
                patch.object(_ims, "set_cached_description"), \
                redirect_stderr(io.StringIO()):
            _ims._sync_static_collection_description("proj", OWN_URL, None)
        client.update_collection.assert_called_once()


if __name__ == "__main__":
    unittest.main()
