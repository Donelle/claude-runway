"""Tests for libs/lmstudio_client.py: the one module that imports `openai` and
builds the bounded client that libs/local_compress_lib.py receives as an
argument, plus the guarantee that local_compress_lib itself imports without
`openai` installed."""

import os
import subprocess
import sys
import unittest

LIBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs")
sys.path.insert(0, LIBS)

from lmstudio_client import build_client  # noqa: E402


class ClientOptionalTimeout(unittest.TestCase):
    """build_client() takes an optional `timeout` for a caller that needs a
    shorter per-request budget than the resolved default. These test the REAL
    build_client() (not a stub) against the actual openai.OpenAI constructor,
    confirming: (a) the one-argument call, with no timeout, still builds a
    client for the given base URL, and (b) passing timeout actually reaches
    the constructed client's own .timeout attribute, rather than being
    silently accepted and dropped."""

    def test_no_timeout_arg_matches_pre_existing_single_arg_behavior(self):
        c = build_client("http://localhost:1234/v1")
        self.assertEqual(str(c.base_url), "http://localhost:1234/v1/")

    def test_explicit_timeout_is_applied_to_the_client(self):
        c = build_client("http://localhost:1234/v1", timeout=2.5)
        # openai.OpenAI stores an explicit `timeout=` constructor arg as a
        # plain float on `.timeout` (rather than wrapping it in its
        # httpx.Timeout default), so this is a direct, non-fragile check.
        self.assertEqual(c.timeout, 2.5)


class ClientIsAlwaysBounded(unittest.TestCase):
    """build_client() used to pass neither timeout nor max_retries, so
    the OpenAI SDK's own defaults (600s read timeout, 2 retries) let a
    reachable-but-hung LM Studio stall the PostToolUse hook for ~30 minutes.
    Invariant under test: every client is built with a finite timeout and
    max_retries=0, whichever way the timeout is supplied."""

    ENV = "CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS"

    def setUp(self):
        self._saved = os.environ.pop(self.ENV, None)

    def tearDown(self):
        os.environ.pop(self.ENV, None)
        if self._saved is not None:
            os.environ[self.ENV] = self._saved

    def test_default_client_has_bounded_timeout_and_no_retries(self):
        import local_compress_lib as L
        c = build_client("http://localhost:1234/v1")
        self.assertEqual(c.timeout, L.DEFAULT_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(c.max_retries, 0)
        # Guard against silently regressing toward the SDK's 600s default.
        self.assertLess(c.timeout, 600)

    def test_explicit_timeout_client_also_has_no_retries(self):
        c = build_client("http://localhost:1234/v1", timeout=2.5)
        self.assertEqual(c.timeout, 2.5)
        self.assertEqual(c.max_retries, 0)

    def test_invalid_explicit_timeout_falls_back_to_bounded_default(self):
        """The explicit-arg path must apply the same
        finite-and-positive rule as the env var, not bypass it."""
        import local_compress_lib as L
        for bad in (float("inf"), float("nan"), 0, -1):
            with self.subTest(value=bad):
                c = build_client("http://localhost:1234/v1", timeout=bad)
                self.assertEqual(c.timeout, L.DEFAULT_REQUEST_TIMEOUT_SECONDS)
                self.assertEqual(c.max_retries, 0)

    def test_env_var_overrides_default_timeout(self):
        os.environ[self.ENV] = " 12.5 "
        self.assertEqual(build_client("http://localhost:1234/v1").timeout, 12.5)

    def test_invalid_env_values_fall_back_to_default(self):
        import local_compress_lib as L
        for bad in ("", "abc", "0", "-5", "inf", "nan"):
            with self.subTest(value=bad):
                os.environ[self.ENV] = bad
                self.assertEqual(L.resolve_request_timeout(), L.DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_hung_server_fails_within_the_timeout_budget(self):
        """End to end against a socket that accepts but never responds: the
        real complete() must return None (fail open) within the configured
        budget rather than blocking. Bounded by an outer join timeout so a
        regression fails this test instead of hanging the suite."""
        import socket
        import threading
        import time
        import local_compress_lib as L

        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(5)
        conns = []

        def accept_loop():
            while True:
                try:
                    conns.append(srv.accept()[0])
                except OSError:
                    return

        threading.Thread(target=accept_loop, daemon=True).start()
        os.environ[self.ENV] = "1"
        result = {}

        def call():
            start = time.monotonic()
            result["out"] = L.complete(build_client(f"http://127.0.0.1:{srv.getsockname()[1]}/v1"), "m", "s", "x")
            result["elapsed"] = time.monotonic() - start

        t = threading.Thread(target=call, daemon=True)
        try:
            t.start()
            t.join(timeout=10)
            self.assertFalse(t.is_alive(), "complete() hung past the configured timeout")
            self.assertIsNone(result["out"])
            # With retries on, 1s x 3 attempts (plus backoff) would exceed this.
            self.assertLess(result["elapsed"], 3)
        finally:
            srv.close()
            for c in conns:
                c.close()


class TimeoutTranslation(unittest.TestCase):
    """The library tells a timeout from an outage via its own CompletionTimeout;
    this module is what turns openai's timeout into it."""

    def _client_raising(self, exc):
        from types import SimpleNamespace
        import lmstudio_client

        def create(**kwargs):
            raise exc

        inner = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            timeout=3.0,
            max_retries=0,
        )
        return lmstudio_client.LMStudioClient(inner)

    def _request(self):
        import httpx
        return httpx.Request("POST", "http://localhost:1234/v1")

    def test_openai_timeout_becomes_completion_timeout(self):
        import openai
        from local_compress_lib import CompletionTimeout
        c = self._client_raising(openai.APITimeoutError(request=self._request()))
        with self.assertRaises(CompletionTimeout):
            c.chat.completions.create(model="m", messages=[])

    def test_connection_error_is_not_translated(self):
        import openai
        from local_compress_lib import CompletionTimeout
        c = self._client_raising(openai.APIConnectionError(request=self._request()))
        with self.assertRaises(openai.APIConnectionError) as ctx:
            c.chat.completions.create(model="m", messages=[])
        self.assertNotIsInstance(ctx.exception, CompletionTimeout)

    def test_other_attributes_are_the_wrapped_clients_own(self):
        c = build_client("http://localhost:1234/v1", timeout=2.5)
        self.assertEqual(c.timeout, 2.5)
        self.assertTrue(hasattr(c, "models"))


class LibraryImportsWithoutOpenai(unittest.TestCase):
    """local_compress_lib must import and serve its non-model functions
    (config checks, model discovery) without `openai`: it is standard-library
    only, and the WebFetch redirect hook imports nothing else from the
    compression stack. libs/lmstudio_client.py is the one module that needs
    `openai`. Run in a fresh interpreter with `openai` made unimportable."""

    BLOCK = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'openai' or name.startswith('openai.'):\n"
        "            raise ImportError('openai is blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "sys.path.insert(0, %r)\n"
    ) % LIBS

    def _run(self, code):
        return subprocess.run(
            [sys.executable, "-I", "-c", self.BLOCK + code],
            capture_output=True, text=True, timeout=60,
        )

    def test_local_compress_lib_imports_and_serves_non_model_calls(self):
        r = self._run(
            "import local_compress_lib as L\n"
            "assert 'openai' not in sys.modules\n"
            "assert L.resolve_request_timeout() > 0\n"
            "assert callable(L.fetch_loaded_models) and callable(L.stale_env_warning)\n"
        )
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_shared_client_module_is_what_needs_openai(self):
        r = self._run("import lmstudio_client\n")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("openai is blocked", r.stderr)

    def test_completion_timeout_is_defined_by_the_library(self):
        r = self._run(
            "import local_compress_lib as L\n"
            "assert issubclass(L.CompletionTimeout, Exception)\n"
        )
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
