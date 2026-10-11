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
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
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

    def _run_reset(self, fake_client, collection="test-collection", mismatch=None):
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "FastEmbedProvider"), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "check_embedding_model_mismatch", return_value=mismatch) as fake_check, \
             mock.patch.object(mod, "store_batch", new=mock.AsyncMock(return_value=0)):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                _run(mod.ingest(_args(dry_run=False, reset=True, collection=collection)))
        return fake_check

    def test_non_dry_run_reset_uses_filtered_delete_never_delete_collection(self):
        """Issue #265: pins the index_repo-style filtered delete (this used
        to pin a raw delete_collection)."""
        fake_client = mock.MagicMock()
        fake_client.collection_exists.return_value = True
        fake_check = self._run_reset(fake_client)
        fake_client.delete_collection.assert_not_called()
        fake_client.delete.assert_called_once()
        kwargs = fake_client.delete.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], "test-collection")
        self.assertEqual(kwargs["points_selector"], mod.mb.memory_bank_exclusion_filter())
        # fail_closed=True is what makes an inconclusive check block the delete.
        self.assertTrue(fake_check.call_args.kwargs["fail_closed"])

    def test_reset_on_missing_collection_deletes_nothing(self):
        fake_client = mock.MagicMock()
        fake_client.collection_exists.return_value = False
        self._run_reset(fake_client)
        fake_client.delete.assert_not_called()
        fake_client.delete_collection.assert_not_called()

    def test_model_mismatch_blocks_reset_before_any_delete(self):
        fake_client = mock.MagicMock()
        fake_client.collection_exists.return_value = True
        with self.assertRaises(SystemExit) as cm:
            self._run_reset(fake_client, mismatch="Error: embedding model mismatch")
        self.assertEqual(cm.exception.code, 1)
        fake_client.delete.assert_not_called()
        fake_client.delete_collection.assert_not_called()


def _transient_drop():
    """The shape a real vpnkit drop actually reaches callers in: qdrant-client
    wraps the raw httpx error in ResponseHandlingException (see
    libs/qdrant_retry.py's docstring)."""
    import httpx
    from qdrant_client.http.exceptions import ResponseHandlingException
    return ResponseHandlingException(httpx.RemoteProtocolError("Server disconnected"))


class SharedPreparationWithIndexRepo(unittest.TestCase):
    """Issue #304: the CLI now runs index_repo's shared preparation
    (libs/qdrant_index_prep.py). These failed on main before the fix: a
    --dry-run loaded the embedding model, and the non-reset path's
    collection_exists/ensure_file_path_index weren't retry-wrapped, so one
    dropped connection aborted the whole run."""

    def setUp(self):
        # No real sleeps between retry attempts.
        patcher = mock.patch("qdrant_retry.RETRY_BACKOFF_SECONDS", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dry_run_never_constructs_the_embedding_provider(self):
        for reset in (False, True):
            with self.subTest(reset=reset), \
                 mock.patch.object(mod, "QdrantClient"), \
                 mock.patch.object(mod, "FastEmbedProvider") as fake_provider_cls, \
                 mock.patch.object(mod, "QdrantConnector") as fake_connector_cls:
                with redirect_stdout(io.StringIO()):
                    _run(mod.ingest(_args(dry_run=True, reset=reset)))
                fake_provider_cls.assert_not_called()
                fake_connector_cls.assert_not_called()

    def _run_non_reset(self, fake_client):
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "FastEmbedProvider"), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "store_batch", new=mock.AsyncMock(return_value=0)) as fake_store:
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=False, reset=False)))
        return fake_store

    def test_transient_drop_on_collection_exists_is_retried(self):
        fake_client = mock.MagicMock()
        fake_client.collection_exists.side_effect = [_transient_drop(), False]
        fake_store = self._run_non_reset(fake_client)
        self.assertEqual(fake_client.collection_exists.call_count, 2)
        fake_store.assert_awaited_once()

    def test_transient_drop_inside_the_index_backfill_is_retried(self):
        fake_client = mock.MagicMock()
        fake_client.collection_exists.return_value = True
        info = mock.MagicMock()
        info.payload_schema = {}
        fake_client.get_collection.side_effect = [_transient_drop(), info]
        fake_store = self._run_non_reset(fake_client)
        fake_client.create_payload_index.assert_called_once()
        fake_store.assert_awaited_once()

    def test_reset_reuses_the_model_checks_provider_instead_of_loading_twice(self):
        fake_client = mock.MagicMock()
        fake_client.collection_exists.return_value = True
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "FastEmbedProvider") as fake_provider_cls, \
             mock.patch.object(mod, "QdrantConnector") as fake_connector_cls, \
             mock.patch.object(mod, "check_embedding_model_mismatch", return_value=None), \
             mock.patch.object(mod, "store_batch", new=mock.AsyncMock(return_value=0)):
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=False, reset=True)))
        fake_provider_cls.assert_called_once()
        self.assertIs(
            fake_connector_cls.call_args.kwargs["embedding_provider"], fake_provider_cls.return_value
        )


class ResetRefusesMemoryBankCollection(unittest.TestCase):
    """Issue #265: --reset against the configured memory-bank collection is
    refused outright, before any Qdrant client is even constructed."""

    def _assert_refused(self, **overrides):
        fake_client_cls = mock.MagicMock()
        err = io.StringIO()
        with mock.patch.object(mod, "QdrantClient", fake_client_cls), \
             mock.patch.object(mod, "DEFAULT_MEMORY_BANK_COLLECTION", "my-shared-bank"), \
             mock.patch.object(mod, "FastEmbedProvider"), \
             mock.patch.object(mod, "QdrantConnector"):
            with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                _run(mod.ingest(_args(collection="my-shared-bank", reset=True, **overrides)))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("memory-bank", err.getvalue())
        fake_client_cls.assert_not_called()

    def test_reset_of_memory_bank_collection_is_refused(self):
        self._assert_refused(dry_run=False)

    def test_refusal_also_applies_alongside_dry_run(self):
        self._assert_refused(dry_run=True)

    def test_non_reset_ingest_into_memory_bank_name_is_not_this_guards_concern(self):
        """Scope pin: only --reset is guarded here (issue #265)."""
        fake_client = mock.MagicMock()
        fake_client.collection_exists.return_value = False
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "DEFAULT_MEMORY_BANK_COLLECTION", "my-shared-bank"), \
             mock.patch.object(mod, "FastEmbedProvider"), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "store_batch", new=mock.AsyncMock(return_value=0)):
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(collection="my-shared-bank", reset=False)))


def _dry_run_indexed_paths(args) -> set:
    """Runs a real --dry-run ingest() over args.repo_path (no Qdrant, no
    embedding model -- both mocked) and returns the set of file_path values
    build_entries actually produced, plus the kwargs it was called with."""
    captured = {}
    real_build_entries = mod.build_entries

    def _recording_build_entries(*a, **kw):
        captured["kwargs"] = kw
        entries, skipped = real_build_entries(*a, **kw)
        captured["paths"] = {m["file_path"] for _, m in entries}
        return entries, skipped

    with mock.patch.object(mod, "QdrantClient"), \
         mock.patch.object(mod, "FastEmbedProvider"), \
         mock.patch.object(mod, "QdrantConnector"), \
         mock.patch.object(mod, "build_entries", side_effect=_recording_build_entries):
        with redirect_stdout(io.StringIO()):
            _run(mod.ingest(argparse.Namespace(**{**vars(args), "dry_run": True})))
    return captured


class CliParityWithMcpServer(unittest.TestCase):
    """Issue #276: three ways the CLI had drifted from ingest_mcp_server.py.
    The csv, manifest and env-fallback tests failed on main before the fix;
    the flags-win, no-env-defaults and --help tests are guards on the new
    defaults (precedence, unchanged built-ins, no secret echoed in help)."""

    # --- 1. csv parsing --------------------------------------------------

    def test_include_ext_with_spaces_still_indexes_every_listed_extension(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "a.py").write_text("x = 1\n")
            (repo / "b.md").write_text("# Title\n\nbody\n")
            captured = _dry_run_indexed_paths(
                _args(repo_path=d, include_ext=".py, .md", no_gitignore=True)
            )
        self.assertEqual(captured["paths"], {"a.py", "b.md"})

    def test_exclude_dirs_with_spaces_still_excludes_every_listed_dir(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            for sub in ("keep", "fixtures", "generated"):
                (repo / sub).mkdir()
                (repo / sub / "m.py").write_text("x = 1\n")
            captured = _dry_run_indexed_paths(
                _args(repo_path=d, include_ext=".py", exclude_dirs="fixtures, generated")
            )
        self.assertEqual(captured["paths"], {os.path.join("keep", "m.py")})

    # --- 2. manifest self-indexing (#91) ---------------------------------

    def test_cli_never_indexes_the_sync_manifest(self):
        """Mirrors test_qdrant_ingest_lib.ManifestSelfIndex for the CLI path:
        a repo previously maintained by sync_repo has the manifest at its
        root, and .json is in CODE_EXTENSIONS."""
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "main.py").write_text("x = 1\n")
            (repo / "config.json").write_text('{"k": "v"}\n')
            (repo / ".qdrant_index_manifest.json").write_text('{"main.py": "h"}\n')
            captured = _dry_run_indexed_paths(
                _args(repo_path=d, include_ext=None, scope="code", no_gitignore=True)
            )
        self.assertIn("main.py", captured["paths"])
        self.assertIn("config.json", captured["paths"])
        self.assertNotIn(".qdrant_index_manifest.json", captured["paths"])

    # --- 3. env var defaults ---------------------------------------------

    def _parse(self, argv, env):
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(sys, "argv", ["ingest_to_qdrant.py", *argv]):
            return mod.parse_args()

    BASE_ARGV = ["--repo-path", "/tmp/repo", "--collection", "c"]

    def test_parse_args_falls_back_to_every_env_var_the_server_reads(self):
        args = self._parse(self.BASE_ARGV, {
            "QDRANT_URL": "http://remote:6333",
            "QDRANT_API_KEY": "secret",
            "EMBEDDING_MODEL": "BAAI/bge-small-en",
            "INDEX_INCLUDE_EXTENSIONS": ".py,.md",
            "INDEX_EXCLUDE_DIRS": "fixtures",
        })
        self.assertEqual(args.qdrant_url, "http://remote:6333")
        self.assertEqual(args.qdrant_api_key, "secret")
        self.assertEqual(args.embedding_model, "BAAI/bge-small-en")
        self.assertEqual(args.include_ext, ".py,.md")
        self.assertEqual(args.exclude_dirs, "fixtures")

    def test_parse_args_flags_win_over_env(self):
        args = self._parse(self.BASE_ARGV + [
            "--qdrant-url", "http://flag:6333",
            "--qdrant-api-key", "flag-key",
            "--embedding-model", "flag/model",
            "--include-ext", ".rs",
            "--exclude-dirs", "flagdir",
        ], {
            "QDRANT_URL": "http://remote:6333",
            "QDRANT_API_KEY": "secret",
            "EMBEDDING_MODEL": "BAAI/bge-small-en",
            "INDEX_INCLUDE_EXTENSIONS": ".py,.md",
            "INDEX_EXCLUDE_DIRS": "fixtures",
        })
        self.assertEqual(args.qdrant_url, "http://flag:6333")
        self.assertEqual(args.qdrant_api_key, "flag-key")
        self.assertEqual(args.embedding_model, "flag/model")
        self.assertEqual(args.include_ext, ".rs")
        self.assertEqual(args.exclude_dirs, "flagdir")

    def test_parse_args_built_in_defaults_with_no_env(self):
        args = self._parse(self.BASE_ARGV, {})
        self.assertEqual(args.qdrant_url, "http://localhost:6333")
        self.assertIsNone(args.qdrant_api_key)
        self.assertEqual(args.embedding_model, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertIsNone(args.include_ext)
        self.assertIsNone(args.exclude_dirs)

    def test_help_never_prints_the_env_api_key(self):
        out = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(out):
            self._parse(["--help"], {"QDRANT_API_KEY": "super-secret-key"})
        self.assertNotIn("super-secret-key", out.getvalue())


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
