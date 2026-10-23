#!/usr/bin/env python3
"""Tests for the files a rebuild keeps but does not scan.

A rebuild preserves the chunks of a file that is still on disk but outside its
scope. Those paths are recorded in the manifest under an additive key, apart
from the {file: hash} map, so a later sync can notice when one is deleted and
remove its chunks.

The sync_repo tests run against qdrant-client's in-process `:memory:` mode, so
the real filters (including the memory-bank/compact exclusion) are exercised
rather than asserted against a mock. The embedding model is replaced by a
stand-in (tests/fake_token_budget.py). No test depends on the order the repo
scan yields files in.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "mcpservers"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "test-collection")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

import ingest_mcp_server as _ims  # noqa: E402
from fake_token_budget import word_budget  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402
from qdrant_ingest_lib import (  # noqa: E402
    MANIFEST_FILENAME,
    MANIFEST_FORMAT_VERSION,
    MANIFEST_PRESERVED_KEY,
    files_removed_since_last_sync,
    manifest_to_json,
    parse_manifest,
    parse_manifest_preserved,
)


def _run(coro):
    return asyncio.run(coro)


def _provider_cls(max_tokens=20):
    cls = MagicMock(name="FastEmbedProvider")
    cls.return_value.build_token_budget.return_value = word_budget(max_tokens)
    return cls


def _local_collection(points):
    client = QdrantClient(":memory:")
    client.create_collection("col", vectors_config={"v": models.VectorParams(size=2, distance=models.Distance.COSINE)})
    client.upsert(
        "col",
        [
            models.PointStruct(id=i, vector={"v": [1.0, 0.0]}, payload={"document": "text", "metadata": md})
            for i, md in points.items()
        ],
    )
    return client


def _ids(client):
    return sorted(p.id for p in client.scroll("col", limit=1000)[0])


def _manifest_doc(repo: Path) -> dict:
    return json.loads((repo / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def _kept(repo: Path) -> set:
    """The kept-but-untracked set _load_manifest reports for `repo`."""
    kept: set = set()
    _ims._load_manifest(repo, kept)
    return kept


def _reader_that_predates_the_key(data):
    """parse_manifest exactly as it was before MANIFEST_PRESERVED_KEY existed:
    it looks at the format version and the "files" map and nothing else."""
    if not isinstance(data, dict) or data.get("manifest_format") != 2:
        return None
    files = data.get("files")
    if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
        return None
    return files


# ---------------------------------------------------------------------------
# The manifest format
# ---------------------------------------------------------------------------


class PreservedKeyInTheManifest(unittest.TestCase):
    def test_the_format_version_is_not_bumped(self):
        """A bump makes every project rebuild again (the cost the owner is about
        to pay once, across many projects). The key is additive."""
        self.assertEqual(MANIFEST_FORMAT_VERSION, 2)

    def test_round_trip(self):
        doc = json.loads(manifest_to_json({"a.py": "h"}, {"z.md", "b.md"}))
        self.assertEqual(doc[MANIFEST_PRESERVED_KEY], ["b.md", "z.md"])
        self.assertEqual(parse_manifest_preserved(doc), {"b.md", "z.md"})
        self.assertEqual(parse_manifest(doc), {"a.py": "h"})

    def test_an_empty_set_leaves_the_key_out_so_the_manifest_is_what_it_was(self):
        self.assertEqual(manifest_to_json({"a.py": "h"}), manifest_to_json({"a.py": "h"}, set()))
        self.assertNotIn(MANIFEST_PRESERVED_KEY, json.loads(manifest_to_json({"a.py": "h"}, ())))

    def test_a_manifest_without_the_key_reads_as_an_empty_set(self):
        self.assertEqual(parse_manifest_preserved(json.loads(manifest_to_json({"a.py": "h"}))), set())
        self.assertEqual(parse_manifest_preserved({"manifest_format": 2, "files": {}}), set())

    def test_a_reader_that_predates_the_key_still_loads_the_hash_map(self):
        files = {"a.py": "h1", "dir/b.md": "h2"}
        doc = json.loads(manifest_to_json(files, {"kept.md"}))
        self.assertEqual(_reader_that_predates_the_key(doc), files)
        # ...and so does the current reader, which is the same code path.
        self.assertEqual(parse_manifest(doc), files)

    def test_an_unusable_or_malformed_value_reads_as_an_empty_set(self):
        base = {"manifest_format": MANIFEST_FORMAT_VERSION, "files": {"a.py": "h"}}
        for value in ("kept.md", 3, {"kept.md": "h"}, None):
            with self.subTest(value=value):
                self.assertEqual(parse_manifest_preserved({**base, MANIFEST_PRESERVED_KEY: value}), set())
        # Entries of the wrong type are dropped one by one, not the whole list.
        mixed = {**base, MANIFEST_PRESERVED_KEY: ["kept.md", 5, "", None]}
        self.assertEqual(parse_manifest_preserved(mixed), {"kept.md"})

    def test_a_document_this_code_does_not_understand_has_no_preserved_set(self):
        for doc in (
            {"a.py": "h"},
            {"manifest_format": MANIFEST_FORMAT_VERSION + 1, "files": {}, MANIFEST_PRESERVED_KEY: ["x.md"]},
            {"manifest_format": MANIFEST_FORMAT_VERSION, "files": ["a.py"], MANIFEST_PRESERVED_KEY: ["x.md"]},
            [],
            None,
        ):
            with self.subTest(doc=doc):
                self.assertEqual(parse_manifest_preserved(doc), set())

    def test_the_set_is_never_part_of_what_files_removed_since_last_sync_looks_at(self):
        files = {"a.py": "h"}
        doc = json.loads(manifest_to_json(files, {"kept.md"}))
        removed = files_removed_since_last_sync(parse_manifest(doc), {"a.py": "h"}, [])
        self.assertEqual(removed, [])

    def test_save_and_load_through_the_server_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _ims._save_manifest(repo, {"a.py": "h"}, {"kept.md"})
            self.assertEqual(_ims._load_manifest(repo), {"a.py": "h"})
            self.assertEqual(_kept(repo), {"kept.md"})
            _ims._save_manifest(repo, {"a.py": "h"})
            self.assertEqual(_kept(repo), set())
            self.assertNotIn(MANIFEST_PRESERVED_KEY, _manifest_doc(repo))

    def test_loading_the_set_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.assertEqual(_kept(repo), set())
            for blob in (b"not json", b"[1]", b"\xff\xfe\x00"):
                (repo / MANIFEST_FILENAME).write_bytes(blob)
                self.assertEqual(_kept(repo), set())


# ---------------------------------------------------------------------------
# The sweep reports what it kept
# ---------------------------------------------------------------------------


class SweepReportsPreservedPaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)

    def test_an_indexed_file_on_disk_outside_the_scan_is_reported(self):
        (self.repo / "README.md").write_text("# hi")
        (self.repo / "a.py").write_text("x")
        client = _local_collection(
            {
                1: {"file_path": "README.md", "type": "doc"},
                2: {"file_path": "a.py", "type": "code"},
                3: {"file_path": "gone.md", "type": "doc"},
            }
        )
        kept: set = set()
        removed = _ims._delete_orphaned_chunks(client, "col", self.repo, {"a.py"}, kept)
        self.assertEqual(removed, ["gone.md"])
        self.assertEqual(kept, {"README.md"})
        self.assertEqual(_ids(client), [1, 2])

    def test_a_file_that_cannot_be_checked_is_kept_and_reported(self):
        (self.repo / "locked.md").write_text("x")
        client = _local_collection({1: {"file_path": "locked.md", "type": "doc"}})
        real_lstat = os.lstat

        def lstat(path, *a, **kw):
            if str(path).endswith("locked.md"):
                raise PermissionError("cannot search the parent directory")
            return real_lstat(path, *a, **kw)

        kept: set = set()
        with patch("os.lstat", side_effect=lstat):
            removed = _ims._delete_orphaned_chunks(client, "col", self.repo, (), kept)
        self.assertEqual(removed, [])
        self.assertEqual(kept, {"locked.md"})
        self.assertEqual(_ids(client), [1])

    def test_non_chunk_points_are_not_reported(self):
        client = _local_collection(
            {1: {"file_path": "note.md", "note": "stored by hand"}, 2: {"source": "memory-bank", "file_path": "m.md"}}
        )
        (self.repo / "note.md").write_text("x")
        kept: set = set()
        _ims._delete_orphaned_chunks(client, "col", self.repo, (), kept)
        self.assertEqual(kept, set())

    def test_the_new_arguments_are_optional(self):
        client = _local_collection({1: {"file_path": "gone.md", "type": "doc"}})
        self.assertEqual(_ims._delete_orphaned_chunks(client, "col", self.repo), ["gone.md"])

    def test_gone_paths_use_the_lstat_based_check(self):
        (self.repo / "here.md").write_text("x")
        self.assertEqual(_ims._preserved_paths_gone(self.repo, ["here.md", "nope.md", "a/b.md"]), ["a/b.md", "nope.md"])


# ---------------------------------------------------------------------------
# sync_repo, end to end against an in-memory collection
# ---------------------------------------------------------------------------


class SyncRepoTracksPreservedPaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        (self.repo / "a.py").write_text("print('a')")
        (self.repo / "README.md").write_text("# readme words")
        (self.repo / "NOTES.md").write_text("# notes words")
        self.client = _local_collection(
            {
                1: {"file_path": "a.py", "type": "code"},
                2: {"file_path": "README.md", "type": "doc"},
                3: {"file_path": "NOTES.md", "type": "doc"},
                4: {"file_path": "gone.md", "type": "doc"},
                # Not indexer chunks: the deletion must leave them alone.
                5: {"source": "memory-bank", "file_path": "README.md"},
                6: {"source": "conversation-compact", "file_path": "README.md"},
                7: {"file_path": "other.md", "note": "kept"},
            }
        )
        self.stored: list = []

        next_id = [1000]

        async def _store(connector, entries, **kw):
            # Really store the chunks, so a re-indexed file ends up with chunks
            # in the collection as it would in production.
            self.stored.extend(entries)
            points = []
            for entry in entries:
                next_id[0] += 1
                points.append(models.PointStruct(
                    id=next_id[0], vector={"v": [1.0, 0.0]},
                    payload={"document": entry.content, "metadata": entry.metadata},
                ))
            if points:
                self.client.upsert("col", points)
            return len(entries)

        for target, kwargs in (
            ("store_batch", {"new": AsyncMock(side_effect=_store)}),
            ("QdrantConnector", {}),
            ("ensure_file_path_index", {}),
            ("check_embedding_model_mismatch", {"return_value": None}),
            ("_recompute_search_limit", {}),
        ):
            patcher = patch.object(_ims, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _sync(self, include=".py", provider_cls=None):
        self.stored.clear()
        provider_cls = provider_cls or _provider_cls()
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=provider_cls):
            return _run(_ims.sync_repo(repo_path=str(self.repo), collection="col", include_extensions=include))

    def _rebuild(self):
        """The first sync after an update: no manifest, narrower scope than the
        index was built with (code only; the index also holds docs)."""
        self.assertFalse((self.repo / MANIFEST_FILENAME).exists())
        return self._sync(".py")

    def _chunk_files(self):
        """The files, other than a.py (always in scope, so never what these
        tests are about), that still have indexer chunks in the collection."""
        return sorted(
            {
                (p.payload or {}).get("metadata", {}).get("file_path")
                for p in self.client.scroll("col", limit=1000)[0]
                if (p.payload or {}).get("metadata", {}).get("type") in ("code", "doc")
            }
            - {"a.py"}
        )

    def _non_chunk_ids(self):
        return [i for i in _ids(self.client) if 5 <= i <= 7]

    # -- the rebuild records what it kept -------------------------------------------

    def test_a_rebuild_with_a_narrower_scope_records_the_paths_it_kept(self):
        result = self._rebuild()
        doc = _manifest_doc(self.repo)
        self.assertEqual(set(doc["files"]), {"a.py"})
        self.assertEqual(doc[MANIFEST_PRESERVED_KEY], ["NOTES.md", "README.md"])
        self.assertEqual(doc["manifest_format"], MANIFEST_FORMAT_VERSION)
        self.assertIn("Rebuild kept the chunks of 2 file(s) still on disk but outside this scan", result)
        # Their chunks are still searchable; the one for the deleted file is not.
        self.assertEqual(self._chunk_files(), ["NOTES.md", "README.md"])

    def test_a_rebuild_with_nothing_kept_writes_no_key(self):
        (self.repo / "README.md").unlink()
        (self.repo / "NOTES.md").unlink()
        result = self._rebuild()
        self.assertNotIn(MANIFEST_PRESERVED_KEY, _manifest_doc(self.repo))
        self.assertNotIn("Rebuild kept", result)

    def test_a_rebuild_that_finds_nothing_to_index_still_records_what_it_kept(self):
        (self.repo / "a.py").unlink()
        result = self._rebuild()
        self.assertIn("No changes since last sync", result)
        doc = _manifest_doc(self.repo)
        self.assertEqual(doc["files"], {})
        self.assertEqual(doc[MANIFEST_PRESERVED_KEY], ["NOTES.md", "README.md"])

    def test_a_file_that_is_in_the_rebuilds_scan_is_tracked_not_kept(self):
        self._sync(".py,.md")
        doc = _manifest_doc(self.repo)
        self.assertEqual(set(doc["files"]), {"a.py", "README.md", "NOTES.md"})
        self.assertNotIn(MANIFEST_PRESERVED_KEY, doc)

    def test_an_unreadable_file_in_the_scan_stays_tracked_rather_than_kept(self):
        real_read_bytes = Path.read_bytes

        def read_bytes(path, *a, **kw):
            if path.name == "README.md":
                raise OSError("simulated read failure")
            return real_read_bytes(path, *a, **kw)

        with patch.object(Path, "read_bytes", read_bytes):
            self._sync(".py,.md")
        doc = _manifest_doc(self.repo)
        self.assertEqual(doc["files"]["README.md"], _ims.UNVERIFIED_HASH)
        self.assertNotIn(MANIFEST_PRESERVED_KEY, doc)

    # -- later syncs ---------------------------------------------------------------

    def test_deleting_a_kept_file_removes_its_chunks_on_the_next_sync(self):
        self._rebuild()
        (self.repo / "README.md").unlink()
        result = self._sync(".py")
        self.assertIn("Also removed the chunks of 1 previously kept file(s) since deleted from disk: README.md.", result)
        self.assertEqual(self._chunk_files(), ["NOTES.md"])
        # The other kept file is still watched; the deleted one is not.
        self.assertEqual(_manifest_doc(self.repo)[MANIFEST_PRESERVED_KEY], ["NOTES.md"])
        # Memory-bank points and conversation compacts survive even when they
        # carry the deleted file's file_path, and so does a note about another file.
        self.assertEqual(self._non_chunk_ids(), [5, 6, 7])

    def test_a_stored_note_carrying_the_deleted_files_path_is_deleted_with_it(self):
        """Same as for any removed file: the per-file delete excludes only the
        memory-bank and conversation-compact sources, not every point that is not
        a chunk, and the rebuild's own warning says so. The installation guide
        must not promise more than that."""
        self.client.upsert("col", [models.PointStruct(
            id=8, vector={"v": [1.0, 0.0]},
            payload={"document": "a note", "metadata": {"file_path": "README.md", "note": "stored with qdrant-store"}},
        )])
        self._rebuild()
        (self.repo / "README.md").unlink()
        self._sync(".py")
        self.assertNotIn(8, _ids(self.client))
        self.assertEqual(self._non_chunk_ids(), [5, 6, 7])

    def test_deleting_the_last_kept_file_drops_the_key(self):
        self._rebuild()
        (self.repo / "README.md").unlink()
        (self.repo / "NOTES.md").unlink()
        self._sync(".py")
        self.assertNotIn(MANIFEST_PRESERVED_KEY, _manifest_doc(self.repo))
        self.assertEqual(self._chunk_files(), [])
        # ...and the next sync has nothing left to do or load a model for.
        provider_cls = _provider_cls()
        self.assertIn("No changes since last sync", self._sync(".py", provider_cls=provider_cls))
        provider_cls.assert_not_called()

    def test_a_kept_file_that_is_still_there_is_never_swept(self):
        """A narrower scan must not delete the chunks of a file that exists,
        however many syncs run: that is the whole point of keeping it apart from
        the hash map files_removed_since_last_sync reads."""
        self._rebuild()
        for _ in range(3):
            provider_cls = _provider_cls()
            result = self._sync(".py", provider_cls=provider_cls)
            self.assertIn("No changes since last sync", result)
            provider_cls.assert_not_called()
            self.assertEqual(self._chunk_files(), ["NOTES.md", "README.md"])
            self.assertEqual(_manifest_doc(self.repo)[MANIFEST_PRESERVED_KEY], ["NOTES.md", "README.md"])

    def test_a_sync_that_scans_a_different_set_still_never_removes_a_present_kept_file(self):
        self._rebuild()
        (self.repo / "b.py").write_text("print('b')")
        result = self._sync(".py")
        self.assertIn("1 file(s) re-indexed", result)
        self.assertEqual(self._chunk_files(), ["NOTES.md", "README.md", "b.py"])
        self.assertEqual(_manifest_doc(self.repo)[MANIFEST_PRESERVED_KEY], ["NOTES.md", "README.md"])

    def test_widening_the_scope_moves_a_kept_file_into_normal_tracking(self):
        self._rebuild()
        result = self._sync(".py,.md")
        self.assertIn("2 file(s) re-indexed", result)
        self.assertEqual({e.metadata["file_path"] for e in self.stored}, {"README.md", "NOTES.md"})
        doc = _manifest_doc(self.repo)
        self.assertEqual(set(doc["files"]), {"a.py", "README.md", "NOTES.md"})
        self.assertNotIn(MANIFEST_PRESERVED_KEY, doc)
        # Normal tracking: deleting it is now an ordinary removal.
        (self.repo / "README.md").unlink()
        self.assertIn("1 file(s) removed", self._sync(".py,.md"))
        self.assertEqual(self._chunk_files(), ["NOTES.md"])

    def test_widening_for_one_file_keeps_watching_the_other(self):
        self._rebuild()
        (self.repo / "NOTES.md").rename(self.repo / "NOTES.txt")
        self._sync(".py,.md")
        doc = _manifest_doc(self.repo)
        self.assertEqual(set(doc["files"]), {"a.py", "README.md"})
        self.assertNotIn(MANIFEST_PRESERVED_KEY, doc)

    def test_a_kept_file_whose_reindex_fails_to_chunk_stays_kept(self):
        """Its old chunks were not replaced, and with no hash entry written it
        would otherwise fall out of both lists."""
        self._rebuild()
        real_chunk_file = _ims.chunk_file

        def chunk_file(path, rel, *a, **kw):
            if rel == "README.md":
                raise OSError("simulated chunk failure")
            return real_chunk_file(path, rel, *a, **kw)

        with patch.object(_ims, "chunk_file", side_effect=chunk_file):
            self._sync(".py,.md")
        doc = _manifest_doc(self.repo)
        self.assertNotIn("README.md", doc["files"])
        self.assertEqual(doc[MANIFEST_PRESERVED_KEY], ["README.md"])
        self.assertEqual(self._chunk_files(), ["NOTES.md", "README.md"])

    def test_a_permission_error_keeps_the_file_and_its_chunks(self):
        self._rebuild()
        (self.repo / "README.md").unlink()
        real_lstat = os.lstat

        def lstat(path, *a, **kw):
            if str(path).endswith("README.md"):
                raise PermissionError("cannot search the parent directory")
            return real_lstat(path, *a, **kw)

        with patch("os.lstat", side_effect=lstat):
            result = self._sync(".py")
        self.assertIn("No changes since last sync", result)
        self.assertEqual(self._chunk_files(), ["NOTES.md", "README.md"])
        self.assertEqual(_manifest_doc(self.repo)[MANIFEST_PRESERVED_KEY], ["NOTES.md", "README.md"])
        # Once it can be checked again, the deletion is noticed.
        self._sync(".py")
        self.assertEqual(self._chunk_files(), ["NOTES.md"])

    def test_a_path_that_is_in_both_the_hash_map_and_the_set_is_tracked_normally(self):
        """Never reported as removed through the set, and never both at once."""
        (self.repo / "README.md").unlink()
        _ims._save_manifest(self.repo, {"a.py": _sha(self.repo / "a.py"), "README.md": "h"}, {"README.md"})
        result = self._sync(".py")
        self.assertIn("1 file(s) removed", result)
        doc = _manifest_doc(self.repo)
        self.assertEqual(set(doc["files"]), {"a.py"})
        self.assertNotIn(MANIFEST_PRESERVED_KEY, doc)

    def test_the_deletion_goes_through_the_same_checks_as_a_removed_file(self):
        """A schema mismatch stops the delete, as it does for a tracked removal."""
        self._rebuild()
        (self.repo / "README.md").unlink()
        with patch.object(_ims, "check_embedding_model_mismatch", return_value="Error: embedding model mismatch"):
            result = self._sync(".py")
        self.assertIn("Error", result)
        self.assertEqual(self._chunk_files(), ["NOTES.md", "README.md"])

    def test_an_old_reader_of_the_saved_manifest_still_sees_the_hash_map(self):
        self._rebuild()
        doc = _manifest_doc(self.repo)
        self.assertIn(MANIFEST_PRESERVED_KEY, doc)
        self.assertEqual(_reader_that_predates_the_key(doc), {"a.py": _sha(self.repo / "a.py")})

    def test_a_manifest_written_before_the_key_existed_syncs_as_before(self):
        _ims._save_manifest(self.repo, {"a.py": _sha(self.repo / "a.py")})
        provider_cls = _provider_cls()
        result = self._sync(".py", provider_cls=provider_cls)
        self.assertIn("No changes since last sync", result)
        provider_cls.assert_not_called()
        self.assertNotIn(MANIFEST_PRESERVED_KEY, _manifest_doc(self.repo))

    def test_the_hashes_and_the_kept_set_come_from_one_read_of_the_manifest(self):
        """Another process saving the manifest can leave it truncated for a moment.
        Read twice (the hashes before the scan, the kept set after it), a sync
        could see a good hash map and an empty kept set, then save a manifest
        without the watched paths: their chunks would then outlive a deletion,
        since ordinary syncs do not sweep. One read makes a bad read lose both, so
        the existing rebuild recovery applies."""
        self._rebuild()
        (self.repo / "b.py").write_text("print('b')")
        manifest_path = self.repo / MANIFEST_FILENAME
        real_hash = _ims._hash_files_with_progress

        async def hash_while_the_manifest_is_being_rewritten(*a, **kw):
            result = await real_hash(*a, **kw)
            manifest_path.write_text("", encoding="utf-8")  # the other writer's write_text, mid-flight
            return result

        with patch.object(_ims, "_hash_files_with_progress", side_effect=hash_while_the_manifest_is_being_rewritten):
            self._sync(".py")
        doc = _manifest_doc(self.repo)
        self.assertEqual(doc[MANIFEST_PRESERVED_KEY], ["NOTES.md", "README.md"])

    def test_an_ordinary_sync_never_scrolls_the_whole_collection(self):
        """The orphan sweep stays rebuild-only: re-checking the kept set is one
        lstat per path, not a scan of the collection."""
        self._rebuild()
        (self.repo / "README.md").unlink()
        with patch.object(_ims, "_delete_orphaned_chunks") as sweep:
            self._sync(".py")
        sweep.assert_not_called()


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# index_repo's baseline
# ---------------------------------------------------------------------------


class IndexRepoBaselineHasNoKeptPaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        (self.repo / "a.txt").write_text("a few words here")

        async def _store(connector, entries, **kw):
            return len(entries)

        for target, kwargs in (
            ("store_batch", {"new": AsyncMock(side_effect=_store)}),
            ("QdrantConnector", {}),
            ("ensure_file_path_index", {}),
            ("check_embedding_model_mismatch", {"return_value": None}),
            ("_recompute_search_limit", {}),
            ("DEFAULT_COLLECTION", {"new": "c"}),
        ):
            patcher = patch.object(_ims, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = MagicMock()
        self.client.collection_exists.return_value = True
        self.client.get_collection.return_value = MagicMock(points_count=3)
        self.client.count.return_value = MagicMock(count=3)

    def _index(self, **kwargs):
        with patch.object(_ims, "QdrantClient", return_value=self.client), \
             patch.object(_ims, "FastEmbedProvider", new=_provider_cls()):
            return _run(_ims.index_repo(
                repo_path=str(self.repo), collection="c", scope="docs", include_extensions=".txt", **kwargs,
            ))

    def test_a_reset_index_writes_a_baseline_with_no_kept_paths(self):
        _ims._save_manifest(self.repo, {"old.txt": "h"}, {"kept.md"})
        result = self._index(reset=True)
        self.assertIn("Wrote the sync manifest (1 file(s))", result)
        doc = _manifest_doc(self.repo)
        self.assertEqual(set(doc["files"]), {"a.txt"})
        self.assertNotIn(MANIFEST_PRESERVED_KEY, doc)
        self.assertEqual(_kept(self.repo), set())

    def test_a_clean_start_into_an_empty_collection_does_the_same(self):
        self.client.get_collection.return_value = MagicMock(points_count=0)
        self.client.count.return_value = MagicMock(count=0)
        _ims._save_manifest(self.repo, {"old.txt": "h"}, {"kept.md"})
        self._index()
        self.assertNotIn(MANIFEST_PRESERVED_KEY, _manifest_doc(self.repo))

    def test_the_baseline_is_written_without_a_kept_set(self):
        with patch.object(_ims, "_save_manifest", wraps=_ims._save_manifest) as save:
            self._index(reset=True)
        save.assert_called_once()
        self.assertEqual(len(save.call_args.args), 2)
        self.assertNotIn("preserved", save.call_args.kwargs)

    def test_a_forced_run_into_a_populated_collection_leaves_the_kept_set_alone_too(self):
        _ims._save_manifest(self.repo, {"old.txt": "h"}, {"kept.md"})
        result = self._index(force=True)
        self.assertIn("The sync manifest was not written", result)
        self.assertEqual(_kept(self.repo), {"kept.md"})


# ---------------------------------------------------------------------------
# The workaround note is gone from every place that carried it
# ---------------------------------------------------------------------------


class WorkaroundNoteIsReplacedConsistently(unittest.TestCase):
    """The advice to run the first rebuild with the scope you normally index
    with existed because nothing tracked the files a narrower rebuild kept.
    README, docs/installation.md, the upgrade migration's note and sync_repo's
    docstring all carried it; they must now agree that those files are tracked."""

    def _texts(self):
        import upgrade_lib

        def read(*parts):
            return " ".join((Path(REPO_ROOT).joinpath(*parts)).read_text(encoding="utf-8").split())

        migration = next(m for m in upgrade_lib.REPO_MIGRATIONS if m.id == "index-manifest-old-format")
        return {
            "README.md": read("README.md"),
            "docs/installation.md": read("docs", "installation.md"),
            "the upgrade migration note": " ".join(migration.note.split()),
            "sync_repo's docstring": " ".join((_ims.sync_repo.__doc__ or "").split()),
        }

    def test_none_of_them_still_asks_for_the_same_scope_or_says_removal_goes_unnoticed(self):
        for where, text in self._texts().items():
            with self.subTest(where=where):
                low = text.lower()
                self.assertNotIn("same scope you normally index with", low)
                self.assertNotIn("until the collection is reset or rebuilt again", low)
                self.assertNotIn("no sync removes", low)
                self.assertNotIn("no `sync_repo` removes", low)
                self.assertNotIn("is not recorded in the new manifest", low)
                self.assertNotIn("is not tracked in the new manifest", low)

    def test_the_installation_guide_does_not_promise_that_all_stored_notes_survive(self):
        text = self._texts()["docs/installation.md"]
        self.assertNotIn("stored notes and conversation compacts are left alone", text)
        self.assertIn("is deleted with it", text)

    def test_each_of_them_says_the_kept_file_is_recorded(self):
        for where, text in self._texts().items():
            with self.subTest(where=where):
                self.assertRegex(text.lower(), r"recorded[^.]*\bkept\b|\bkept\b[^.]*recorded")


if __name__ == "__main__":
    unittest.main()
