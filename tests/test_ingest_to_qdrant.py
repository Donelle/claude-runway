#!/usr/bin/env python3
"""Tests for tools/ingest_to_qdrant.py's ingest() -- specifically its
--dry-run contract.

Stdlib-only (unittest, no pytest), no network and no real Qdrant/FastEmbed
-- QdrantClient, FastEmbedProvider, and QdrantConnector are all mocked out,
since these tests only care about WHICH calls happen, not their real I/O.

    .venv/bin/python -m unittest discover -s tests
"""

import argparse
import asyncio
import hashlib
import io
import json
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # fake_token_budget

import ingest_to_qdrant as mod  # noqa: E402
from fake_token_budget import word_budget  # noqa: E402
import qdrant_ingest_lib as lib  # noqa: E402
from qdrant_ingest_lib import MANIFEST_FILENAME, MANIFEST_FORMAT_VERSION, TokenBudgetError  # noqa: E402
from sync_manifest_lib import UNVERIFIED_HASH, discard_manifest, load_manifest, save_manifest  # noqa: E402


def _provider_mock() -> mock.MagicMock:
    """A stand-in FastEmbedProvider class whose instances hand out a model-free
    word-count TokenBudget: chunking needs a budget even for a
    dry run, and a bare MagicMock would hand chunk_stream a MagicMock."""
    cls = mock.MagicMock(name="FastEmbedProvider")
    cls.return_value.build_token_budget.return_value = word_budget(max_tokens=20)
    return cls


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
        write_manifest=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run(coro):
    return asyncio.run(coro)


class DryRunNeverTouchesQdrant(unittest.TestCase):
    """--dry-run is documented (--help, and the "not storing" print in
    ingest() itself) as a pure preview -- these pin that contract after a
    real regression: an earlier version of the file_path payload-index
    backfill unconditionally constructed a QdrantClient and
    called collection_exists/ensure_file_path_index even when dry_run was
    set, which broke an offline dry run outright and, for an online one,
    silently created a real payload index despite the "preview only"
    contract. Confirmed by reproduction before fixing."""

    def test_plain_dry_run_constructs_no_qdrant_client_at_all(self):
        fake_client_cls = mock.MagicMock()
        with mock.patch.object(mod, "QdrantClient", fake_client_cls), \
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
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
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
             mock.patch.object(mod, "QdrantConnector"):
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=True, reset=True)))
        fake_client_cls.assert_not_called()

    def test_dry_run_never_calls_ensure_file_path_index(self):
        with mock.patch.object(mod, "QdrantClient"), \
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "ensure_file_path_index") as fake_ensure:
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=True, reset=False)))
        fake_ensure.assert_not_called()

    def test_dry_run_never_calls_store_batch(self):
        with mock.patch.object(mod, "QdrantClient"), \
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
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
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "ensure_file_path_index") as fake_ensure, \
             mock.patch.object(mod, "store_batch", new=mock.AsyncMock(return_value=0)):
            with redirect_stdout(io.StringIO()):
                _run(mod.ingest(_args(dry_run=False, reset=False)))
        fake_ensure.assert_called_once_with(fake_client, "test-collection")

    def _run_reset(self, fake_client, collection="test-collection", mismatch=None):
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "check_embedding_model_mismatch", return_value=mismatch) as fake_check, \
             mock.patch.object(mod, "store_batch", new=mock.AsyncMock(return_value=0)):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                _run(mod.ingest(_args(dry_run=False, reset=True, collection=collection)))
        return fake_check

    def test_non_dry_run_reset_uses_filtered_delete_never_delete_collection(self):
        """Pins the index_repo-style filtered delete (this used
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
    """The CLI now runs index_repo's shared preparation
    (libs/qdrant_index_prep.py). These failed on main before the fix: a
    --dry-run loaded the embedding model, and the non-reset path's
    collection_exists/ensure_file_path_index weren't retry-wrapped, so one
    dropped connection aborted the whole run."""

    def setUp(self):
        # No real sleeps between retry attempts.
        patcher = mock.patch("qdrant_retry.RETRY_BACKOFF_SECONDS", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dry_run_loads_the_model_for_its_window_but_builds_no_connector(self):
        """Chunking is by the model's token window, so even a
        preview has to load the model to read it (it never embeds anything, so
        no connector)."""
        for reset in (False, True):
            with self.subTest(reset=reset), \
                 mock.patch.object(mod, "QdrantClient"), \
                 mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()) as fake_provider_cls, \
                 mock.patch.object(mod, "QdrantConnector") as fake_connector_cls:
                with redirect_stdout(io.StringIO()):
                    _run(mod.ingest(_args(dry_run=True, reset=reset)))
                fake_provider_cls.assert_called_once()
                fake_provider_cls.return_value.build_token_budget.assert_called_once_with(10)
                fake_connector_cls.assert_not_called()

    def _run_non_reset(self, fake_client):
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
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
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()) as fake_provider_cls, \
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
    """--Reset against the configured memory-bank collection is
    refused outright, before any Qdrant client is even constructed."""

    def _assert_refused(self, **overrides):
        fake_client_cls = mock.MagicMock()
        err = io.StringIO()
        with mock.patch.object(mod, "QdrantClient", fake_client_cls), \
             mock.patch.object(mod, "DEFAULT_MEMORY_BANK_COLLECTION", "my-shared-bank"), \
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
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
        """Scope pin: only --reset is guarded here."""
        fake_client = mock.MagicMock()
        fake_client.collection_exists.return_value = False
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "DEFAULT_MEMORY_BANK_COLLECTION", "my-shared-bank"), \
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
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
         mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
         mock.patch.object(mod, "QdrantConnector"), \
         mock.patch.object(mod, "build_entries", side_effect=_recording_build_entries):
        with redirect_stdout(io.StringIO()):
            _run(mod.ingest(argparse.Namespace(**{**vars(args), "dry_run": True})))
    return captured


class CliParityWithMcpServer(unittest.TestCase):
    """Three ways the CLI had drifted from ingest_mcp_server.py.
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

    # --- 2. manifest self-indexing ---------------------------------

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

    def test_env_api_key_is_not_sent_to_a_different_qdrant_url(self):
        # The exported key belongs to $QDRANT_URL. A --qdrant-url naming any
        # other host must not carry it there.
        args = self._parse(self.BASE_ARGV + ["--qdrant-url", "https://other.example"], {
            "QDRANT_URL": "http://configured:6333",
            "QDRANT_API_KEY": "secret",
        })
        self.assertEqual(args.qdrant_url, "https://other.example")
        self.assertIsNone(args.qdrant_api_key)

    def test_explicit_api_key_is_kept_for_a_different_qdrant_url(self):
        args = self._parse(self.BASE_ARGV + [
            "--qdrant-url", "https://other.example", "--qdrant-api-key", "other-key",
        ], {"QDRANT_URL": "http://configured:6333", "QDRANT_API_KEY": "secret"})
        self.assertEqual(args.qdrant_api_key, "other-key")

    def test_env_api_key_kept_when_qdrant_url_flag_names_the_configured_url(self):
        # Spelling out the configured URL, trailing slash and all, is still the
        # configured endpoint.
        args = self._parse(self.BASE_ARGV + ["--qdrant-url", "http://configured:6333/"], {
            "QDRANT_URL": "http://configured:6333",
            "QDRANT_API_KEY": "secret",
        })
        self.assertEqual(args.qdrant_api_key, "secret")

    def test_env_api_key_goes_to_the_built_in_default_url_when_qdrant_url_is_unset(self):
        args = self._parse(self.BASE_ARGV, {"QDRANT_API_KEY": "secret"})
        self.assertEqual(args.qdrant_url, "http://localhost:6333")
        self.assertEqual(args.qdrant_api_key, "secret")
        foreign = self._parse(self.BASE_ARGV + ["--qdrant-url", "https://other.example"],
                              {"QDRANT_API_KEY": "secret"})
        self.assertIsNone(foreign.qdrant_api_key)

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


class TokenBudgetChunkingInTheCli(unittest.TestCase):
    """The CLI chunks by the model's token window, fails before any
    delete when the window is unreadable, and reports per-file progress."""

    def _unreadable_window_provider(self):
        cls = mock.MagicMock(name="FastEmbedProvider")
        cls.return_value.build_token_budget.side_effect = TokenBudgetError(
            "Cannot read the token window of embedding model 'x/y': no truncation max_length"
        )
        return cls

    def test_unreadable_window_on_reset_fails_before_the_delete(self):
        fake_client = mock.MagicMock()
        fake_client.collection_exists.return_value = True
        err = io.StringIO()
        with mock.patch.object(mod, "QdrantClient", return_value=fake_client), \
             mock.patch.object(mod, "FastEmbedProvider", new=self._unreadable_window_provider()), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "check_embedding_model_mismatch", return_value=None), \
             mock.patch.object(mod, "store_batch", new=mock.AsyncMock(return_value=0)):
            with redirect_stderr(err), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                _run(mod.ingest(_args(dry_run=False, reset=True)))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("token window", err.getvalue())
        fake_client.delete.assert_not_called()

    def test_unreadable_window_fails_a_dry_run_too(self):
        err = io.StringIO()
        with mock.patch.object(mod, "QdrantClient"), \
             mock.patch.object(mod, "FastEmbedProvider", new=self._unreadable_window_provider()), \
             mock.patch.object(mod, "build_entries") as fake_build:
            with redirect_stderr(err), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                _run(mod.ingest(_args(dry_run=True)))
        fake_build.assert_not_called()

    def test_entries_carry_the_new_position_metadata_and_a_budget_is_passed(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text(" ".join(f"w{i}" for i in range(50)) + "\n")
            captured = {}
            real_build_entries = mod.build_entries

            def _recording(*a, **kw):
                captured["budget"] = kw.get("budget")
                entries, skipped = real_build_entries(*a, **kw)
                captured["metadata"] = [m for _, m in entries]
                return entries, skipped

            with mock.patch.object(mod, "QdrantClient"), \
                 mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
                 mock.patch.object(mod, "QdrantConnector"), \
                 mock.patch.object(mod, "build_entries", side_effect=_recording):
                with redirect_stdout(io.StringIO()):
                    _run(mod.ingest(_args(repo_path=d, include_ext=".txt", dry_run=True)))
        self.assertIsNotNone(captured["budget"])
        self.assertGreater(len(captured["metadata"]), 1)
        for md in captured["metadata"]:
            self.assertIn("line_number", md)
            self.assertIn("char_range", md)
            self.assertNotIn("line_range", md)

    def test_prints_per_file_chunk_counts_and_the_largest_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "big.txt").write_text(" ".join(f"w{i}" for i in range(100)) + "\n")
            (Path(d) / "small.txt").write_text("just a few words\n")
            out = io.StringIO()
            with mock.patch.object(mod, "QdrantClient"), \
                 mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
                 mock.patch.object(mod, "QdrantConnector"):
                with redirect_stdout(out):
                    _run(mod.ingest(_args(repo_path=d, include_ext=".txt", dry_run=True)))
        text = out.getvalue()
        self.assertIn("big.txt: 5 chunk(s)", text)
        self.assertIn("small.txt: 1 chunk(s)", text)
        self.assertIn("Largest files by chunk count: big.txt (5), small.txt (1).", text)
        self.assertIn("20-token window", text)

    def test_non_default_chunk_lines_is_reported_as_ignored(self):
        out = io.StringIO()
        with mock.patch.object(mod, "QdrantClient"), \
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
             mock.patch.object(mod, "QdrantConnector"):
            with redirect_stdout(out):
                _run(mod.ingest(_args(dry_run=True, chunk_lines=7)))
        self.assertIn("--chunk-lines 7 was ignored", out.getvalue())

    def test_negative_overlap_is_a_parse_error(self):
        with mock.patch.object(sys, "argv", ["ingest_to_qdrant.py", "--repo-path", ".", "--collection", "c", "--overlap", "-1"]), \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            mod.parse_args()
        self.assertEqual(cm.exception.code, 2)

    def test_overlap_larger_than_a_line_chunk_is_fine_now_that_it_is_tokens(self):
        """The old check rejected overlap >= chunk_lines; overlap is tokens now."""
        with mock.patch.object(sys, "argv", ["ingest_to_qdrant.py", "--repo-path", ".", "--collection", "c",
                                             "--overlap", "60", "--chunk-lines", "50"]):
            args = mod.parse_args()
        self.assertEqual(args.overlap, 60)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WriteManifestInTheCli(unittest.TestCase):
    """The opt-in --write-manifest flag leaves sync_repo's manifest
    baseline behind, like index_repo does when it starts from a clean slate.

    Every test indexes a throwaway repo in a temp directory, never REPO_ROOT, so
    the real checkout's manifest is not touched whichever way a test goes. Where
    a test needs a manifest to already exist it plants a sentinel one and then
    checks what became of it. Paths are compared as sets: nothing here depends
    on the order files are scanned in."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        (self.repo / "big.txt").write_text(" ".join(f"w{i}" for i in range(100)) + "\n")
        (self.repo / "small.txt").write_text("just a few words\n")
        self.client = mock.MagicMock()
        self.client.collection_exists.return_value = False
        self.store = mock.AsyncMock(return_value=0)
        self.events: list = []
        # No real sleeps between retry attempts.
        patcher = mock.patch("qdrant_retry.RETRY_BACKOFF_SECONDS", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    # -- helpers ---------------------------------------------------------------

    def _plant(self, files=None):
        """A sentinel manifest, written with the real writer."""
        sentinel = files or {"sentinel.txt": "sentinel-hash"}
        save_manifest(self.repo, sentinel)
        return sentinel

    def _manifest_path(self) -> Path:
        return self.repo / MANIFEST_FILENAME

    def _saved_doc(self) -> dict:
        return json.loads(self._manifest_path().read_text(encoding="utf-8"))

    def _saved_files(self) -> dict:
        return self._saved_doc()["files"]

    def _populated(self, points=7, reported=...):
        """An existing collection holding `points` points. `reported` is the
        count get_collection gives (default: the same); None is what Qdrant
        returns when it does not report one."""
        self.client.collection_exists.return_value = True
        self.client.get_collection.return_value = mock.MagicMock(
            points_count=points if reported is ... else reported
        )
        self.client.count.return_value = mock.MagicMock(count=points)

    def _index(self, mismatch=None, check_effect=None, **overrides):
        """Run ingest() over the temp repo; returns (stdout, stderr). The same
        text stays on self.out/self.err for a run that exits."""
        args = _args(**{
            "repo_path": str(self.repo), "include_ext": ".txt", "write_manifest": True, **overrides,
        })
        out, err = self.out, self.err = io.StringIO(), io.StringIO()
        with mock.patch.object(mod, "QdrantClient", return_value=self.client), \
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "ensure_file_path_index", return_value=False), \
             mock.patch.object(mod, "check_embedding_model_mismatch", return_value=mismatch,
                               side_effect=check_effect), \
             mock.patch.object(mod, "store_batch", new=self.store):
            with redirect_stdout(out), redirect_stderr(err):
                _run(mod.ingest(args))
        return out.getvalue(), err.getvalue()

    # -- the flag is opt-in -----------------------------------------------------

    def test_the_flag_is_off_by_default_and_on_when_passed(self):
        for argv, expected in (([], False), (["--write-manifest"], True)):
            with self.subTest(argv=argv), \
                 mock.patch.object(sys, "argv", ["ingest_to_qdrant.py", "--repo-path", ".", "--collection", "c", *argv]):
                self.assertEqual(mod.parse_args().write_manifest, expected)

    def test_help_explains_the_flag_and_why_it_is_opt_in(self):
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["ingest_to_qdrant.py", "--help"]), \
             self.assertRaises(SystemExit), redirect_stdout(out):
            mod.parse_args()
        text = " ".join(out.getvalue().split())
        self.assertIn("--write-manifest", text)
        self.assertIn("does not say which collection it describes", text)
        self.assertIn("never on --dry-run or after a failed run", text)

    def test_without_the_flag_a_fresh_index_writes_nothing_and_says_nothing_about_a_manifest(self):
        out, _ = self._index(write_manifest=False)
        self.assertFalse(self._manifest_path().exists())
        self.assertNotIn("manifest", out.lower())
        self.client.count.assert_not_called()

    def test_without_the_flag_an_existing_manifest_is_left_exactly_as_it_was(self):
        sentinel = self._plant()
        before = self._manifest_path().read_bytes()
        self.client.collection_exists.return_value = True
        self._index(write_manifest=False, reset=True)
        self.assertEqual(self._manifest_path().read_bytes(), before)
        self.assertEqual(self._saved_files(), sentinel)
        self.client.delete.assert_called_once()  # the reset itself still ran

    def test_without_the_flag_nothing_is_hashed_and_no_hook_is_installed(self):
        self.client.collection_exists.return_value = True
        with mock.patch.object(mod, "compute_file_hashes", wraps=mod.compute_file_hashes) as hashes, \
             mock.patch.object(mod, "prepare_collection_for_index", wraps=mod.prepare_collection_for_index) as prep:
            self._index(write_manifest=False, reset=True)
        hashes.assert_not_called()
        self.assertIsNone(prep.call_args.kwargs["existing_points_guard"])
        self.assertIsNone(prep.call_args.kwargs["before_reset_delete"])

    # -- the clean cases write a versioned baseline -----------------------------

    def test_a_fresh_index_into_an_absent_collection_writes_a_versioned_baseline(self):
        out, _ = self._index()
        self.assertIn("Wrote the sync manifest (2 file(s))", out)
        doc = self._saved_doc()
        self.assertEqual(doc["manifest_format"], MANIFEST_FORMAT_VERSION)
        self.assertEqual(
            doc["files"], {"big.txt": _sha(self.repo / "big.txt"), "small.txt": _sha(self.repo / "small.txt")}
        )

    def test_the_baseline_has_no_preserved_untracked_key_and_loads_through_the_shared_reader(self):
        self._index()
        # Additive key written only when non-empty, and a fresh index has none.
        self.assertNotIn("preserved_untracked", self._saved_doc())
        preserved: set = set()
        files = load_manifest(self.repo, preserved)
        self.assertEqual(set(files), {"big.txt", "small.txt"})
        self.assertEqual(preserved, set())

    def test_it_goes_through_the_one_writer_the_server_uses(self):
        with mock.patch.object(mod, "save_manifest", wraps=mod.save_manifest) as save:
            self._index()
        save.assert_called_once()
        self.assertEqual(save.call_args.args[0], self.repo.resolve())
        self.assertEqual(set(save.call_args.args[1]), {"big.txt", "small.txt"})
        self.assertIs(mod.save_manifest, save_manifest)

    def test_a_reset_replaces_the_old_manifest_with_the_new_baseline(self):
        self._plant({"old.txt": "h"})
        self.client.collection_exists.return_value = True
        out, _ = self._index(reset=True)
        self.assertIn("Wrote the sync manifest (2 file(s))", out)
        self.assertEqual(set(self._saved_files()), {"big.txt", "small.txt"})
        self.client.delete.assert_called_once()

    def test_the_old_manifest_is_removed_between_the_model_check_and_the_delete(self):
        self._plant({"old.txt": "h"})
        self.client.collection_exists.return_value = True
        self.client.delete.side_effect = lambda **kw: self.events.append("delete")

        def discard(repo):
            self.events.append("discard")
            discard_manifest(repo)

        with mock.patch.object(mod, "discard_manifest", side_effect=discard):
            self._index(reset=True, check_effect=lambda *a, **kw: self.events.append("check"))
        # Once before the wipe, once more as the clean-start guarantee.
        self.assertEqual(self.events[:3], ["check", "discard", "delete"])
        self.assertEqual(self.events.count("delete"), 1)

    def test_a_reset_of_an_absent_collection_still_writes_the_baseline(self):
        out, _ = self._index(reset=True)
        self.assertIn("Wrote the sync manifest", out)
        self.client.delete.assert_not_called()

    def test_an_existing_but_empty_collection_counts_as_a_clean_start(self):
        self._populated(points=0)
        out, _ = self._index()
        self.assertIn("Wrote the sync manifest", out)
        self.assertEqual(set(self._saved_files()), {"big.txt", "small.txt"})
        # An apparent zero is confirmed with an exact count before it is trusted.
        self.client.count.assert_called_once()
        self.assertTrue(self.client.count.call_args.kwargs["exact"])

    def test_an_unreported_point_count_on_an_empty_collection_still_writes_the_baseline(self):
        self._populated(points=0, reported=None)
        out, _ = self._index()
        self.assertIn("Wrote the sync manifest", out)

    # -- a populated collection without --reset leaves the manifest alone -------

    def test_a_populated_collection_without_reset_leaves_the_manifest_alone_and_says_why(self):
        self._populated(points=7)
        sentinel = self._plant()
        before = self._manifest_path().read_bytes()
        out, _ = self._index()
        self.assertIn("The sync manifest was not written", out)
        self.assertIn("7 point(s)", out)
        self.assertIn("--reset", out)
        self.assertEqual(self._manifest_path().read_bytes(), before)
        self.assertEqual(self._saved_files(), sentinel)
        self.store.assert_awaited_once()  # the chunks were still stored

    def test_a_populated_collection_without_reset_writes_none_when_there_was_none(self):
        self._populated(points=7)
        self._index()
        self.assertFalse(self._manifest_path().exists())

    def test_nothing_is_hashed_when_no_baseline_will_be_written(self):
        self._populated(points=7)
        with mock.patch.object(mod, "compute_file_hashes", wraps=mod.compute_file_hashes) as hashes:
            self._index()
        hashes.assert_not_called()

    def test_an_unreported_point_count_is_not_taken_for_an_empty_collection(self):
        """Qdrant can report points_count as None, which the preparation step
        turns into 0. A populated collection read as empty would lose the
        manifest that tracks its deleted files."""
        self._populated(points=5, reported=None)
        sentinel = self._plant({"deleted.txt": "h", "big.txt": "stale"})
        out, _ = self._index()
        self.assertIn("The sync manifest was not written", out)
        self.assertIn("5 point(s)", out)
        self.assertEqual(self._saved_files(), sentinel)

    # -- dry run -----------------------------------------------------------------

    def test_a_dry_run_writes_and_removes_nothing_and_never_contacts_qdrant(self):
        sentinel = self._plant()
        before = self._manifest_path().read_bytes()
        for reset in (False, True):
            with self.subTest(reset=reset):
                client_cls = mock.MagicMock()
                out = io.StringIO()
                with mock.patch.object(mod, "QdrantClient", client_cls), \
                     mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
                     mock.patch.object(mod, "QdrantConnector"), \
                     mock.patch.object(mod, "save_manifest") as save, \
                     mock.patch.object(mod, "discard_manifest") as discard, \
                     mock.patch.object(mod, "compute_file_hashes") as hashes, \
                     redirect_stdout(out):
                    _run(mod.ingest(_args(
                        repo_path=str(self.repo), include_ext=".txt", write_manifest=True,
                        dry_run=True, reset=reset,
                    )))
                client_cls.assert_not_called()
                save.assert_not_called()
                discard.assert_not_called()
                hashes.assert_not_called()
                self.assertIn("--write-manifest has no effect on a dry run", out.getvalue())
                self.assertEqual(self._manifest_path().read_bytes(), before)
                self.assertEqual(self._saved_files(), sentinel)

    # -- failed runs --------------------------------------------------------------

    def test_a_run_that_fails_while_storing_writes_no_manifest_and_leaves_none_behind(self):
        self._plant({"big.txt": _sha(self.repo / "big.txt"), "small.txt": _sha(self.repo / "small.txt")})
        self.store.side_effect = RuntimeError("qdrant went away")
        with self.assertRaises(RuntimeError):
            self._index()
        # Chunks may have been stored before it died, so the old manifest no
        # longer describes the collection: a sync must rebuild, not trust it.
        self.assertFalse(self._manifest_path().exists())

    def test_a_reset_that_fails_after_the_wipe_leaves_no_manifest(self):
        self._plant({"old.txt": "h"})
        self.client.collection_exists.return_value = True
        self.store.side_effect = RuntimeError("qdrant went away")
        with self.assertRaises(RuntimeError):
            self._index(reset=True)
        self.client.delete.assert_called_once()
        self.assertFalse(self._manifest_path().exists())

    def test_a_failure_while_chunking_writes_no_manifest(self):
        with mock.patch.object(mod, "build_entries", side_effect=RuntimeError("chunker blew up")):
            with self.assertRaises(RuntimeError):
                self._index()
        self.assertFalse(self._manifest_path().exists())
        self.store.assert_not_awaited()

    def test_a_reset_refused_by_a_model_mismatch_deletes_nothing_and_keeps_the_manifest(self):
        sentinel = self._plant()
        self.client.collection_exists.return_value = True
        with self.assertRaises(SystemExit) as cm:
            self._index(reset=True, mismatch="Error: embedding model mismatch")
        self.assertEqual(cm.exception.code, 1)
        self.client.delete.assert_not_called()
        self.store.assert_not_awaited()
        self.assertEqual(self._saved_files(), sentinel)

    def test_a_reset_that_cannot_remove_the_old_manifest_does_not_start(self):
        """Carrying on would leave a manifest claiming files are indexed that
        the wipe removed; a failure after it would never replace it."""
        sentinel = self._plant({"old.txt": "h"})
        self.client.collection_exists.return_value = True
        with mock.patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            with self.assertRaises(SystemExit) as cm:
                self._index(reset=True)
        self.assertEqual(cm.exception.code, 1)
        self.client.delete.assert_not_called()
        self.store.assert_not_awaited()
        self.assertEqual(self._saved_files(), sentinel)
        self.assertTrue(self.err.getvalue().startswith("Error: could not remove the old sync manifest"),
                        self.err.getvalue())
        self.assertIn("permission denied", self.err.getvalue())
        self.assertIn("Nothing was deleted or indexed", self.err.getvalue())

    def test_a_clean_start_that_cannot_remove_the_old_manifest_does_not_start_either(self):
        sentinel = self._plant({"old.txt": "h"})
        with mock.patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            with self.assertRaises(SystemExit) as cm:
                self._index()
        self.assertEqual(cm.exception.code, 1)
        self.store.assert_not_awaited()
        self.assertEqual(self._saved_files(), sentinel)

    def test_a_missing_manifest_is_not_an_error_for_the_removal(self):
        out, _ = self._index()
        self.assertIn("Wrote the sync manifest", out)

    def test_an_unwritable_manifest_is_reported_not_raised(self):
        with mock.patch.object(mod, "save_manifest", side_effect=OSError("read-only file system")):
            out, _ = self._index()
        self.assertIn("Done.", out)
        self.assertIn("could not write the sync manifest (read-only file system)", out)
        self.assertIn("the next sync_repo will rebuild", out)

    # -- the baseline's contents ---------------------------------------------------

    def test_hashes_are_taken_before_any_file_is_chunked(self):
        """A file edited while the run is going must end up with a hash OLDER
        than its indexed chunks (so the next sync re-indexes it), never newer."""
        original = _sha(self.repo / "small.txt")
        real_chunk_file = lib.chunk_file

        def edit_then_chunk(path, rel, *a, **kw):
            if rel == "small.txt":
                path.write_text("edited while the index was running")
            return real_chunk_file(path, rel, *a, **kw)

        with mock.patch.object(lib, "chunk_file", side_effect=edit_then_chunk):
            self._index()
        self.assertEqual(self._saved_files()["small.txt"], original)
        self.assertNotEqual(self._saved_files()["small.txt"], _sha(self.repo / "small.txt"))

    def test_a_file_that_fails_to_chunk_is_left_out_so_the_next_sync_retries_it(self):
        real_chunk_file = lib.chunk_file

        def chunk_file(path, rel, *a, **kw):
            if rel == "small.txt":
                raise OSError("simulated chunk failure")
            return real_chunk_file(path, rel, *a, **kw)

        with mock.patch.object(lib, "chunk_file", side_effect=chunk_file):
            out, _ = self._index()
        self.assertIn("Skipped 1 unreadable file(s)", out)
        self.assertEqual(set(self._saved_files()), {"big.txt"})

    def test_a_file_that_cannot_be_hashed_but_was_chunked_stays_tracked_and_retryable(self):
        real_read_bytes = Path.read_bytes

        def read_bytes(path, *a, **kw):
            if path.name == "small.txt":
                raise OSError("simulated read failure")
            return real_read_bytes(path, *a, **kw)

        with mock.patch.object(Path, "read_bytes", read_bytes):
            self._index()
        files = self._saved_files()
        self.assertEqual(files["small.txt"], UNVERIFIED_HASH)
        self.assertEqual(files["big.txt"], _sha(self.repo / "big.txt"))

    def test_a_file_created_after_the_hash_scan_is_tracked_as_unverified(self):
        """Chunking walks the tree again, so it can store chunks for a file the
        hashing never saw. Leaving it out would mean deleting it before the next
        sync left its chunks searchable forever."""
        real_hashes = mod.compute_file_hashes

        def hash_then_create(*a, **kw):
            result = real_hashes(*a, **kw)
            (self.repo / "late.txt").write_text("created after the hash scan")
            return result

        with mock.patch.object(mod, "compute_file_hashes", side_effect=hash_then_create):
            self._index()
        files = self._saved_files()
        self.assertEqual(files["late.txt"], UNVERIFIED_HASH)
        self.assertEqual(files["big.txt"], _sha(self.repo / "big.txt"))

    def test_a_file_the_chunking_walk_never_reached_is_unverified_not_up_to_date(self):
        """It was hashed, so its hash would make the next sync call it unchanged
        although it was never indexed."""
        real_build_entries = mod.build_entries

        def build_without_small(*a, **kw):
            real_on_file = kw["on_file"]
            kw["on_file"] = lambda rel, count: None if rel == "small.txt" else real_on_file(rel, count)
            entries, skipped = real_build_entries(*a, **kw)
            return [e for e in entries if e[1]["file_path"] != "small.txt"], skipped

        with mock.patch.object(mod, "build_entries", side_effect=build_without_small):
            self._index()
        files = self._saved_files()
        self.assertEqual(files["small.txt"], UNVERIFIED_HASH)
        self.assertEqual(files["big.txt"], _sha(self.repo / "big.txt"))

    def test_a_file_with_nothing_to_index_keeps_its_hash(self):
        (self.repo / "empty.txt").write_text("")
        (self.repo / "blank.txt").write_text("  \n\n ")
        self._index()
        files = self._saved_files()
        self.assertEqual(files["empty.txt"], _sha(self.repo / "empty.txt"))
        self.assertEqual(files["blank.txt"], _sha(self.repo / "blank.txt"))

    def test_the_baseline_is_scoped_by_the_same_filters_as_the_run(self):
        (self.repo / "skipme").mkdir()
        (self.repo / "skipme" / "inner.txt").write_text("words in an excluded directory")
        (self.repo / "other.md").write_text("# not selected by --include-ext\n")
        self._index(exclude_dirs="skipme")
        self.assertEqual(set(self._saved_files()), {"big.txt", "small.txt"})

    def test_the_manifest_file_never_hashes_itself(self):
        self._plant({"x.json": "h"})
        # An extension filter that would match the manifest's own .json suffix.
        # The remover is patched out so the old manifest is still on disk when
        # the hash scan runs: the exclusion is what keeps it out, not its removal.
        with mock.patch.object(mod, "discard_manifest"):
            self._index(include_ext=".txt,.json")
        self.assertNotIn(MANIFEST_FILENAME, self._saved_files())
        self.assertEqual(set(self._saved_files()), {"big.txt", "small.txt"})

    def test_hash_progress_is_printed_to_the_terminal(self):
        out, _ = self._index()
        self.assertIn("Hashing files for the sync manifest", out)
        with mock.patch.object(lib, "HASH_PROGRESS_EVERY", 1), \
             mock.patch.object(mod, "compute_file_hashes", wraps=mod.compute_file_hashes) as hashes:
            self._index()
        self.assertTrue(callable(hashes.call_args.kwargs["on_progress"]))

    # -- the patch points: the real checkout's manifest is never touched -----------

    def test_with_the_writer_and_remover_patched_the_planted_manifest_is_never_modified(self):
        sentinel = self._plant()
        before = self._manifest_path().read_bytes()
        self.client.collection_exists.return_value = True
        with mock.patch.object(mod, "save_manifest") as save, mock.patch.object(mod, "discard_manifest") as discard:
            out, _ = self._index(reset=True)
        self.assertGreaterEqual(discard.call_count, 1)
        for call in discard.call_args_list:
            self.assertEqual(call.args, (self.repo.resolve(),))
        save.assert_called_once()
        self.assertIn("Wrote the sync manifest", out)
        self.assertEqual(self._manifest_path().read_bytes(), before)
        self.assertEqual(self._saved_files(), sentinel)

    def test_indexing_the_real_checkout_with_the_writer_and_remover_patched_leaves_its_manifest_alone(self):
        real_manifest = Path(REPO_ROOT) / MANIFEST_FILENAME
        existed = real_manifest.exists()
        before = real_manifest.read_bytes() if existed else None
        args = _args(include_ext=".md", write_manifest=True, reset=True)
        self.client.collection_exists.return_value = True
        with mock.patch.object(mod, "QdrantClient", return_value=self.client), \
             mock.patch.object(mod, "FastEmbedProvider", new=_provider_mock()), \
             mock.patch.object(mod, "QdrantConnector"), \
             mock.patch.object(mod, "check_embedding_model_mismatch", return_value=None), \
             mock.patch.object(mod, "store_batch", new=self.store), \
             mock.patch.object(mod, "save_manifest") as save, \
             mock.patch.object(mod, "discard_manifest") as discard, \
             redirect_stdout(io.StringIO()):
            _run(mod.ingest(args))
        save.assert_called_once()
        discard.assert_called()
        self.assertEqual(real_manifest.exists(), existed)
        if existed:
            self.assertEqual(real_manifest.read_bytes(), before)

    # -- docs ----------------------------------------------------------------------

    def test_the_cli_docstring_says_what_the_flag_does_and_why_it_is_opt_in(self):
        doc = " ".join((mod.__doc__ or "").split())
        self.assertIn("--write-manifest", doc)
        self.assertIn("does not record which collection it describes", doc)
        self.assertIn("never on --dry-run or after a failed run", doc)
        self.assertIn("the next sync_repo re-chunks and re-embeds every file", doc)


class MainIsTheConsoleScriptEntryPoint(unittest.TestCase):
    """`main()` is what pyproject.toml's `claude-runway-ingest`
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
