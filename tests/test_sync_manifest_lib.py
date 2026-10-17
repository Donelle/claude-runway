#!/usr/bin/env python3
"""Tests for libs/sync_manifest_lib.py, the manifest helpers the MCP server and
the ingest CLI share. The server's own behaviour through them is covered by the
test_ingest_mcp_server_* files; these pin the shared module itself, and that
both callers use the very same functions (so the format cannot drift).

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import sync_manifest_lib as lib  # noqa: E402
from qdrant_ingest_lib import (  # noqa: E402
    MANIFEST_FILENAME,
    MANIFEST_FORMAT_VERSION,
    MANIFEST_PRESERVED_KEY,
    manifest_to_json,
)


class ManifestFileTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        self.path = self.repo / MANIFEST_FILENAME


class LoadManifest(ManifestFileTestCase):
    def test_no_manifest_is_an_empty_map(self):
        self.assertEqual(lib.load_manifest(self.repo), {})

    def test_a_saved_manifest_round_trips(self):
        lib.save_manifest(self.repo, {"a.py": "h1", "b.py": "h2"})
        self.assertEqual(lib.load_manifest(self.repo), {"a.py": "h1", "b.py": "h2"})

    def test_the_old_flat_format_is_ignored(self):
        self.path.write_text(json.dumps({"a.py": "h1"}), encoding="utf-8")
        self.assertEqual(lib.load_manifest(self.repo), {})

    def test_unreadable_junk_is_an_empty_map_not_an_error(self):
        for content in (b"{not json", b"\xff\xfe\x00bad", b""):
            with self.subTest(content=content):
                self.path.write_bytes(content)
                self.assertEqual(lib.load_manifest(self.repo), {})

    def test_the_preserved_set_comes_from_the_same_single_read_as_the_map(self):
        lib.save_manifest(self.repo, {"a.py": "h1"}, preserved={"kept.py"})
        reads = []
        real_read_text = Path.read_text

        def counting(path, *a, **kw):
            if path.name == MANIFEST_FILENAME:
                reads.append(path)
            return real_read_text(path, *a, **kw)

        preserved: set = set()
        with mock.patch.object(Path, "read_text", counting):
            files = lib.load_manifest(self.repo, preserved)
        self.assertEqual(files, {"a.py": "h1"})
        self.assertEqual(preserved, {"kept.py"})
        self.assertEqual(len(reads), 1)

    def test_the_preserved_set_is_left_empty_when_the_map_is(self):
        self.path.write_text(json.dumps({"a.py": "h1"}), encoding="utf-8")  # old format
        preserved: set = set()
        self.assertEqual(lib.load_manifest(self.repo, preserved), {})
        self.assertEqual(preserved, set())


class SaveManifest(ManifestFileTestCase):
    def test_the_format_version_is_unchanged(self):
        # A bump forces a full rebuild on every project already indexed.
        self.assertEqual(MANIFEST_FORMAT_VERSION, 2)
        lib.save_manifest(self.repo, {"a.py": "h"})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["manifest_format"], 2)

    def test_it_writes_exactly_what_manifest_to_json_produces(self):
        lib.save_manifest(self.repo, {"a.py": "h"})
        self.assertEqual(self.path.read_text(encoding="utf-8"), manifest_to_json({"a.py": "h"}))

    def test_the_preserved_key_is_additive_and_written_only_when_non_empty(self):
        lib.save_manifest(self.repo, {"a.py": "h"})
        self.assertNotIn(MANIFEST_PRESERVED_KEY, json.loads(self.path.read_text(encoding="utf-8")))
        lib.save_manifest(self.repo, {"a.py": "h"}, preserved={"z.py", "k.py"})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))[MANIFEST_PRESERVED_KEY], ["k.py", "z.py"])


class DiscardManifest(ManifestFileTestCase):
    def test_removes_an_existing_manifest(self):
        lib.save_manifest(self.repo, {"a.py": "h"})
        lib.discard_manifest(self.repo)
        self.assertFalse(self.path.exists())

    def test_a_missing_manifest_is_not_an_error(self):
        lib.discard_manifest(self.repo)

    def test_a_manifest_that_cannot_be_removed_raises(self):
        lib.save_manifest(self.repo, {"a.py": "h"})
        with mock.patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            with self.assertRaises(OSError):
                lib.discard_manifest(self.repo)


class InvalidationError(ManifestFileTestCase):
    def test_none_when_the_manifest_was_removed(self):
        lib.save_manifest(self.repo, {"a.py": "h"})
        self.assertIsNone(lib.invalidation_error(self.repo))
        self.assertFalse(self.path.exists())

    def test_the_message_when_it_cannot_be_removed(self):
        def boom(repo):
            raise OSError("permission denied")

        message = lib.invalidation_error(self.repo, boom)
        self.assertTrue(message.startswith("Error: could not remove the old sync manifest"))
        self.assertIn(MANIFEST_FILENAME, message)
        self.assertIn("permission denied", message)
        self.assertIn("Nothing was deleted or indexed", message)

    def test_it_uses_the_remover_it_is_given(self):
        remover = mock.MagicMock()
        self.assertIsNone(lib.invalidation_error(self.repo, remover))
        remover.assert_called_once_with(self.repo)


class IndexBaseline(unittest.TestCase):
    def test_every_hashed_file_keeps_its_hash(self):
        self.assertEqual(lib.index_baseline({"a": "h1", "b": "h2"}, [], {"a": 1, "b": 2}), {"a": "h1", "b": "h2"})

    def test_a_chunk_failure_is_left_out(self):
        self.assertEqual(lib.index_baseline({"a": "h1", "b": "h2"}, [("b", "boom")], {"a": 1}), {"a": "h1"})

    def test_a_chunked_file_without_a_hash_is_unverified(self):
        baseline = lib.index_baseline({"a": "h1"}, [], {"a", "unhashable", "late"})
        self.assertEqual(baseline, {"a": "h1", "unhashable": lib.UNVERIFIED_HASH, "late": lib.UNVERIFIED_HASH})

    def test_a_hashed_file_without_chunks_keeps_its_hash(self):
        self.assertEqual(lib.index_baseline({"empty": "h"}, [], set()), {"empty": "h"})

    def test_a_missed_file_is_unverified_and_a_ghost_is_not_added(self):
        baseline = lib.index_baseline({"a": "h1", "missed": "h2"}, [], {"a"}, missed={"missed", "ghost"})
        self.assertEqual(baseline, {"a": "h1", "missed": lib.UNVERIFIED_HASH})

    def test_it_does_not_modify_the_hashes_it_was_given(self):
        hashes = {"a": "h1"}
        lib.index_baseline(hashes, [], {"a", "b"})
        self.assertEqual(hashes, {"a": "h1"})

    def test_a_mapping_of_counts_works_as_the_chunked_files(self):
        self.assertEqual(lib.index_baseline({"a": "h"}, [], {"a": 3, "b": 1}), {"a": "h", "b": lib.UNVERIFIED_HASH})


class UnindexedHashedFiles(ManifestFileTestCase):
    def test_text_without_chunks_is_missed_and_empty_files_are_not(self):
        (self.repo / "text.txt").write_text("words")
        (self.repo / "empty.txt").write_text("")
        (self.repo / "blank.txt").write_text(" \n ")
        hashes = {"text.txt": "h", "empty.txt": "h", "blank.txt": "h"}
        self.assertEqual(lib.unindexed_hashed_files(self.repo, hashes, [], set()), {"text.txt"})

    def test_chunked_and_chunk_failed_files_are_not_missed(self):
        (self.repo / "chunked.txt").write_text("words")
        (self.repo / "failed.txt").write_text("words")
        hashes = {"chunked.txt": "h", "failed.txt": "h"}
        self.assertEqual(
            lib.unindexed_hashed_files(self.repo, hashes, [("failed.txt", "boom")], {"chunked.txt"}), set()
        )

    def test_a_file_that_cannot_be_read_now_counts_as_missed(self):
        hashes = {"gone.txt": "h"}
        self.assertEqual(lib.unindexed_hashed_files(self.repo, hashes, [], set()), {"gone.txt"})

    def test_the_content_check_is_injectable(self):
        checker = mock.MagicMock(return_value=True)
        missed = lib.unindexed_hashed_files(self.repo, {"a.txt": "h"}, [], set(), has_indexable_content=checker)
        self.assertEqual(missed, {"a.txt"})
        checker.assert_called_once_with(self.repo / "a.txt")


class ServerAndCliShareTheseFunctions(unittest.TestCase):
    """One implementation, two callers: the format cannot drift."""

    def test_the_server_uses_the_shared_functions(self):
        sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
        os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        import ingest_mcp_server as ims

        self.assertIs(ims._load_manifest, lib.load_manifest)
        self.assertIs(ims._save_manifest, lib.save_manifest)
        self.assertIs(ims._discard_manifest, lib.discard_manifest)
        self.assertIs(ims._index_baseline, lib.index_baseline)
        self.assertIs(ims.UNVERIFIED_HASH, lib.UNVERIFIED_HASH)

    def test_the_cli_uses_the_shared_functions(self):
        sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
        import ingest_to_qdrant as cli

        self.assertIs(cli.save_manifest, lib.save_manifest)
        self.assertIs(cli.discard_manifest, lib.discard_manifest)
        self.assertIs(cli.index_baseline, lib.index_baseline)
        self.assertIs(cli.unindexed_hashed_files, lib.unindexed_hashed_files)


if __name__ == "__main__":
    unittest.main()
