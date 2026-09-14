#!/usr/bin/env python3
"""Tests for hooks/redirect_webfetch_to_fetch_url.py.

Stdlib-only (unittest, no pytest) and no network -- OpenAI's client is
stubbed via mock.patch.object on the hook module's own OpenAI reference, the
same monkeypatch-and-restore approach test_local_compress_lib.py's
SectionCompressionEndToEnd uses for LM Studio calls:

    .venv/bin/python -m unittest discover -s tests

Issue #29: _lmstudio_reachable only checked that LM Studio responded at all,
not that a model could actually be resolved -- so in a 0-or-2+-model state
with nothing pinned, this hook would deny WebFetch and fetch_url would then
fail too, for the exact same reason. These tests pin the fixed decision tree
against every input combination that matters, including the one a bare
`len(models.data) == 1` check (the issue's literal suggested fix) would
still get wrong: a PINNED model with 2+ models loaded (fetch_url would
actually succeed there, using the pinned name) and an UNREACHABLE server
with a model pinned (the pin must not bypass reachability itself, since
resolve_model's own pinned branch never checks it -- see the fix's
docstring for why that matters specifically for this hook).

Issue #64: _DeniedUrlCache records denied URLs with a timestamp so a retry
WebFetch call for the same URL within the TTL is allowed through (fetch_url
was presumably tried and failed between the two calls). After allowing, the
entry is deleted (one-shot skip) so the next call is denied again normally.
"""

import datetime
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))

import redirect_webfetch_to_fetch_url as hook  # noqa: E402


def _fake_models(*ids):
    return SimpleNamespace(data=[SimpleNamespace(id=i) for i in ids])


class FakeOpenAI:
    """Stand-in for openai.OpenAI. Constructed with the same kwargs the hook
    passes (base_url/api_key/timeout, all ignored here); .models.list()
    returns `response` or raises `error`, set per test via the class
    attributes below rather than per-instance, since the hook constructs a
    fresh client on every call."""
    response = None
    error = None

    def __init__(self, *args, **kwargs):
        pass

    class _Models:
        def list(self):
            if FakeOpenAI.error is not None:
                raise FakeOpenAI.error
            return FakeOpenAI.response

    @property
    def models(self):
        return FakeOpenAI._Models()


class LmstudioReachable(unittest.TestCase):
    def setUp(self):
        FakeOpenAI.response = None
        FakeOpenAI.error = None
        patcher = mock.patch.object(hook, "OpenAI", FakeOpenAI)
        patcher.start()
        self.addCleanup(patcher.stop)
        # DEFAULT_MODEL/stale_env_warning are plain names imported into the
        # hook's own module namespace -- patching them there (not on
        # local_compress_lib) is what _lmstudio_reachable actually looks up.
        self._orig_default_model = hook.DEFAULT_MODEL
        self._orig_stale_env_warning = hook.stale_env_warning
        self.addCleanup(self._restore_hook_globals)

    def _restore_hook_globals(self):
        hook.DEFAULT_MODEL = self._orig_default_model
        hook.stale_env_warning = self._orig_stale_env_warning

    def _pin(self, model_id):
        hook.DEFAULT_MODEL = model_id

    def _stale(self, is_stale):
        hook.stale_env_warning = lambda: ("stale env var set" if is_stale else None)

    def test_unreachable_returns_false(self):
        # Regression guard: the original "server doesn't respond" case.
        FakeOpenAI.error = ConnectionError("LM Studio is down")
        self._pin(None)
        self.assertFalse(hook._lmstudio_reachable(None))

    def test_reachable_exactly_one_model_no_pin_returns_true(self):
        # Regression guard: the case that already worked before the fix.
        FakeOpenAI.response = _fake_models("model-a")
        self._pin(None)
        self._stale(False)
        self.assertTrue(hook._lmstudio_reachable(None))

    def test_reachable_zero_models_no_pin_returns_false(self):
        # The bug: reachable, but resolve_model would refuse (no model loaded).
        FakeOpenAI.response = _fake_models()
        self._pin(None)
        self._stale(False)
        self.assertFalse(hook._lmstudio_reachable(None))

    def test_reachable_multiple_models_no_pin_returns_false(self):
        # The bug: reachable, but resolve_model can't auto-detect among 2+.
        FakeOpenAI.response = _fake_models("model-a", "model-b")
        self._pin(None)
        self._stale(False)
        self.assertFalse(hook._lmstudio_reachable(None))

    def test_reachable_multiple_models_with_pin_returns_true(self):
        # The case a bare `len(models.data) == 1` check (the issue's literal
        # suggested fix) would still get WRONG: resolve_model would succeed
        # here via the pin, never even looking at the loaded count.
        FakeOpenAI.response = _fake_models("model-a", "model-b")
        self._pin("pinned-model")
        self.assertTrue(hook._lmstudio_reachable(None))

    def test_unreachable_even_with_pin_returns_false(self):
        # The pin must NOT bypass the reachability check itself -- unlike
        # resolve_model's own pinned branch, which never calls .models.list()
        # at all. This hook has no later real call to fall back on, so it
        # has to check reachability unconditionally first.
        FakeOpenAI.error = ConnectionError("LM Studio is down")
        self._pin("pinned-model")
        self.assertFalse(hook._lmstudio_reachable(None))

    def test_stale_env_var_returns_false(self):
        # Reachable and even exactly one model loaded, but resolve_model
        # would still hard-error on an unmigrated old env var name.
        FakeOpenAI.response = _fake_models("model-a")
        self._pin(None)
        self._stale(True)
        self.assertFalse(hook._lmstudio_reachable(None))

    def test_stale_env_var_with_pin_returns_false(self):
        # PR #131 review (Copilot) on issue #41's fix: resolve_model() now
        # checks stale_env_warning() unconditionally, BEFORE its
        # explicit_model/DEFAULT_MODEL short-circuits -- so a pinned model
        # no longer bypasses a stale-env hard-fail there. This hook's
        # DEFAULT_MODEL check used to run before its own stale_env_warning
        # check, so it still reported "reachable" for a pinned model with
        # some OTHER stale var present, denying WebFetch and redirecting to
        # fetch_url -- which would then immediately fail on the very same
        # stale check resolve_model now runs first. Must return False here
        # so this hook fails open (lets WebFetch through) instead.
        FakeOpenAI.response = _fake_models("model-a")
        self._pin("pinned-model")
        self._stale(True)
        self.assertFalse(hook._lmstudio_reachable(None))


class DeniedUrlCache(unittest.TestCase):
    """Tests for _DeniedUrlCache — the TTL-based failed-URL cache (issue #64).

    Each test uses a temporary file-backed SQLite DB (not in-memory) so the
    cache implementation can open and close connections normally without the DB
    disappearing. A fresh temp file is created in setUp and deleted in
    tearDown -- nothing touches the real cache.db on disk.
    """

    def setUp(self):
        import sqlite3
        # Temporary file DB: the implementation closes connections between
        # calls, which would destroy an in-memory DB. A temp file survives
        # open/close cycles and is cleaned up in tearDown.
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = self._tmp.name
        self._cache = hook._DeniedUrlCache()
        # Monkeypatch _connect to use the temp file instead of the real cache.db.
        self._cache._connect = self._tmp_connect

    def _tmp_connect(self):
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.execute(hook._DeniedUrlCache._TABLE_DDL)
        return conn

    def tearDown(self):
        try:
            os.unlink(self._db_path)
        except OSError:
            pass

    def _insert_denied(self, url, denied_at_iso):
        """Helper: directly insert a row into the temp cache table (by digest)."""
        import sqlite3
        digest = hook._DeniedUrlCache._digest(url)
        conn = sqlite3.connect(self._db_path)
        conn.execute(hook._DeniedUrlCache._TABLE_DDL)
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO webfetch_denied_urls (url_digest, denied_at) VALUES (?, ?)",
                (digest, denied_at_iso),
            )
        conn.close()

    def _rows(self):
        """Helper: return all rows in the cache table."""
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.execute(hook._DeniedUrlCache._TABLE_DDL)
        rows = conn.execute(
            "SELECT url_digest FROM webfetch_denied_urls"
        ).fetchall()
        conn.close()
        return rows

    # --- check_and_clear_if_retrying -----------------------------------------

    def test_url_not_in_cache_returns_false(self):
        # A URL never denied before: not a retry.
        result = self._cache.check_and_clear_if_retrying("https://example.com", 3600.0)
        self.assertFalse(result)

    def test_url_denied_recently_returns_true_and_deletes_entry(self):
        # The core feature: a URL denied within the TTL is a retry after
        # fetch_url failed -- allow WebFetch through, delete the cache entry.
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._insert_denied("https://auth.example.com", now_iso)

        result = self._cache.check_and_clear_if_retrying("https://auth.example.com", 3600.0)

        self.assertTrue(result)
        # Entry must be gone so the NEXT call for this URL is denied again.
        self.assertEqual(self._rows(), [])

    def test_url_denied_beyond_ttl_returns_false_and_deletes_entry(self):
        # An expired entry is treated as a miss (URL is not "recently denied").
        # The stale row should still be deleted to keep the table tidy.
        old_iso = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(seconds=7200)
        ).isoformat()
        self._insert_denied("https://example.com/old", old_iso)

        result = self._cache.check_and_clear_if_retrying("https://example.com/old", 3600.0)

        self.assertFalse(result)
        self.assertEqual(self._rows(), [])  # stale entry cleaned up

    def test_after_allow_next_call_returns_false(self):
        # One-shot skip: after the entry is consumed by a True return,
        # a subsequent call for the same URL sees no entry and returns False
        # (the URL will be denied again normally from that point).
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._insert_denied("https://js.example.com", now_iso)

        first = self._cache.check_and_clear_if_retrying("https://js.example.com", 3600.0)
        second = self._cache.check_and_clear_if_retrying("https://js.example.com", 3600.0)

        self.assertTrue(first)
        self.assertFalse(second)

    def test_different_url_not_affected(self):
        # Only the exact URL is matched; another URL in the cache is unaffected.
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._insert_denied("https://a.example.com", now_iso)

        result = self._cache.check_and_clear_if_retrying("https://b.example.com", 3600.0)

        self.assertFalse(result)
        # The unrelated entry should still be there.
        self.assertEqual(len(self._rows()), 1)

    def test_check_fails_open_on_db_error(self):
        # If the DB connection raises, check_and_clear_if_retrying must return
        # False (fail open) rather than propagating the exception.
        def _broken_connect():
            raise OSError("disk full")

        self._cache._connect = _broken_connect
        result = self._cache.check_and_clear_if_retrying("https://example.com", 3600.0)
        self.assertFalse(result)

    # --- record_denied -------------------------------------------------------

    def test_record_denied_inserts_url_and_returns_true(self):
        result = self._cache.record_denied("https://new.example.com")
        self.assertTrue(result)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        # The stored value must be the SHA-256 digest, not the plaintext URL.
        # URLs can contain credentials or signed query strings -- they must
        # never appear verbatim in the on-disk cache.
        import hashlib
        expected_digest = hashlib.sha256(b"https://new.example.com").hexdigest()
        self.assertEqual(rows[0][0], expected_digest)
        self.assertNotIn("new.example.com", rows[0][0])

    def test_record_denied_updates_existing_url(self):
        # Upsert: recording a denial for a URL already in the cache refreshes
        # its timestamp rather than inserting a duplicate.
        old_iso = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(seconds=7200)
        ).isoformat()
        self._insert_denied("https://example.com", old_iso)

        result = self._cache.record_denied("https://example.com")

        self.assertTrue(result)
        rows = self._rows()
        self.assertEqual(len(rows), 1)  # still one row, not two

    def test_record_denied_returns_false_on_db_error(self):
        # A DB write failure must return False so main() can allow WebFetch
        # through: if the record can't be written, no retry-allow is possible,
        # so denying would permanently trap Claude in the loop.
        def _broken_connect():
            raise OSError("disk full")

        self._cache._connect = _broken_connect
        result = self._cache.record_denied("https://example.com")
        self.assertFalse(result)

    # --- _failed_url_ttl -----------------------------------------------------

    def test_default_ttl_is_one_hour(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL", None)
            self.assertEqual(hook._failed_url_ttl(), hook._DEFAULT_FAILED_URL_TTL_SECONDS)

    def test_ttl_env_var_is_honoured(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL": "120"}, clear=False):
            self.assertEqual(hook._failed_url_ttl(), 120.0)

    def test_invalid_ttl_env_var_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL": "not-a-number"}, clear=False):
            self.assertEqual(hook._failed_url_ttl(), hook._DEFAULT_FAILED_URL_TTL_SECONDS)

    def test_nan_ttl_falls_back_to_default(self):
        # nan would make markers never expire -- must be rejected.
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL": "nan"}, clear=False):
            self.assertEqual(hook._failed_url_ttl(), hook._DEFAULT_FAILED_URL_TTL_SECONDS)

    def test_inf_ttl_falls_back_to_default(self):
        # inf would make markers never expire -- must be rejected.
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL": "inf"}, clear=False):
            self.assertEqual(hook._failed_url_ttl(), hook._DEFAULT_FAILED_URL_TTL_SECONDS)

    def test_negative_ttl_falls_back_to_default(self):
        # Negative TTL would prevent any retry from being allowed -- must be rejected.
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL": "-1"}, clear=False):
            self.assertEqual(hook._failed_url_ttl(), hook._DEFAULT_FAILED_URL_TTL_SECONDS)

    def test_zero_ttl_falls_back_to_default(self):
        # Zero TTL means every entry is immediately expired -- same as no cache.
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL": "0"}, clear=False):
            self.assertEqual(hook._failed_url_ttl(), hook._DEFAULT_FAILED_URL_TTL_SECONDS)


if __name__ == "__main__":
    unittest.main()
