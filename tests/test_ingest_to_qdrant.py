#!/usr/bin/env python3
"""Tests for tools/ingest_to_qdrant.py's ingest() -- specifically its
--dry-run contract (PR #121 review, Copilot, issue #54 follow-up).

Stdlib-only (unittest, no pytest), no network and no real Qdrant/FastEmbed
-- QdrantClient, FastEmbedProvider, and QdrantConnector are all mocked out,
since these tests only care about WHICH calls happen, not their real I/O.

    .venv/bin/python -m unittest discover -s tests
"""

import argparse
import asyncio
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import ingest_to_qdrant as mod  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        repo_path=REPO_ROOT,
        collection="test-collection",
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        scope="both",
        include_ext=".md",  # narrow scope -- keeps build_entries() fast/small
        exclude_dirs=None,
        no_gitignore=True,
        chunk_lines=50,
        overlap=10,
        dry_run=False,
        reset=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run(coro):
    return asyncio.run(coro)


class DryRunNeverTouchesQdrant(unittest.TestCase):
    """--dry-run is documented (--help, and the "not storing" print in
    ingest() itself) as a pure preview -- these pin that contract after a
    real regression: an earlier version of the file_path payload-index
    backfill (issue #54) unconditionally constructed a QdrantClient and
    called collection_exists/ensure_file_path_index even when dry_run was
    set, which broke an offline dry run outright and, for an online one,
    silently created a real payload index despite the "preview only"
    contract. Confirmed by reproduction before fixing (see PR #121)."""

    def test_plain_dry_run_constructs_no_qdrant_client_at_all(self):
        fake_client_cls = mock.MagicMock()
        with mock.patch.object(mod, "QdrantClient", fake_client_cls), \
             mock.patch.object(mod, "FastEmbedProvider"), \
             mock.patch.object(mod, "QdrantConnector"):
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=True, reset=False)))
        fake_client_cls.assert_not_called()

    def test_reset_dry_run_also_constructs_no_qdrant_client(self):
        """Narrower pre-existing version of the same bug, also fixed here:
        --reset --dry-run used to delete the real collection for real
        before this fix, since the old reset-delete block ran ahead of the
        dry_run check too."""
        fake_client_cls = mock.MagicMock()
        with mock.patch.object(mod, "QdrantClient", fake_client_cls), \
             mock.patch.object(mod, "FastEmbedProvider"), \
             mock.patch.object(mod, "QdrantConnector"):
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=True, reset=True)))
        fake_client_cls.assert_not_called()

    def test_dry_run_never_calls_ensure_file_path_index(self):
        with mock.patch.object(mod, "QdrantClient"), \
             mock.patch.object(mod, "FastEmbedProvider"), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "ensure_file_path_index") as fake_ensure:
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=True, reset=False)))
        fake_ensure.assert_not_called()

    def test_dry_run_never_calls_store_batch(self):
        with mock.patch.object(mod, "QdrantClient"), \
             mock.patch.object(mod, "FastEmbedProvider"), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "store_batch") as fake_store_batch:
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=True, reset=False)))
        fake_store_batch.assert_not_called()


class NonDryRunStillPerformsCollectionMaintenance(unittest.TestCase):
    """Confirms the dry_run guard didn't accidentally swallow the real
    (non-preview) behavior too."""

    def test_non_dry_run_backfills_the_index_on_an_existing_collection(self):
        fake_client = mock.MagicMock()
        fake_client.collection_exists.return_value = True
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "FastEmbedProvider"), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "ensure_file_path_index") as fake_ensure, \
             mock.patch.object(mod, "store_batch", new=mock.AsyncMock(return_value=0)):
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=False, reset=False)))
        fake_ensure.assert_called_once_with(fake_client, "test-collection")

    def test_non_dry_run_reset_still_deletes_the_real_collection(self):
        fake_client = mock.MagicMock()
        fake_client.collection_exists.return_value = True
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "FastEmbedProvider"), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "store_batch", new=mock.AsyncMock(return_value=0)):
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=False, reset=True)))
        fake_client.delete_collection.assert_called_once_with("test-collection")


class MainIsTheConsoleScriptEntryPoint(unittest.TestCase):
    """`main()` (issue #50) is what pyproject.toml's `claude-runway-ingest`
    console script points `module:function` at -- pinning that it does
    exactly what the `if __name__ == "__main__":` guard always did
    (`asyncio.run(ingest(parse_args()))`), just pulled into a callable so
    there's something for an entry point to reference. Mocks `parse_args`/
    `ingest`/`asyncio.run` themselves rather than exercising a real ingest,
    since that's already covered by the classes above -- this only needs to
    confirm the wiring, not re-test ingest()'s own behavior."""

    def test_main_parses_args_builds_and_runs_the_ingest_coroutine(self):
        fake_args = _args()
        fake_coro = object()
        # ingest() is `async def`, so mock.patch.object would otherwise
        # auto-detect that and hand back an AsyncMock -- whose own
        # return_value, once called, is a real coroutine wrapping fake_coro
        # rather than fake_coro itself, breaking the identity check below.
        # An explicit plain MagicMock sidesteps that auto-detection, since
        # main() only ever passes ingest(...)'s return value straight into
        # asyncio.run() without awaiting it directly itself.
        fake_ingest = mock.MagicMock(return_value=fake_coro)
        with mock.patch.object(mod, "parse_args", return_value=fake_args) as fake_parse_args, \
             mock.patch.object(mod, "ingest", new=fake_ingest), \
             mock.patch.object(mod.asyncio, "run") as fake_run:
            mod.main()
        fake_parse_args.assert_called_once_with()
        fake_ingest.assert_called_once_with(fake_args)
        fake_run.assert_called_once_with(fake_coro)


if __name__ == "__main__":
    unittest.main()
