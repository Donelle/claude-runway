"""Tests for libs/openai_lib.py: the OpenAI-shaped request functions and the bounded client builder, which the
codebase-reviewer and the compression tools share.

The request functions (`complete`, `yes_probability`) and their side channels are also exercised through the
reviewers in tests/test_reviewer_recording.py and tests/test_yes_probability.py, and through compression in
tests/test_local_compress_lib.py. This file holds what belongs to the module itself: the exact request sent, the
failure reasons, the timeout translation, and the layering guarantees (what imports what)."""

import ast
import asyncio
import os
import subprocess
import sys
import threading
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS = os.path.join(ROOT, "libs")
sys.path.insert(0, LIBS)
sys.path.insert(0, os.path.join(ROOT, "harness"))

import local_compress_lib  # noqa: E402
import openai_lib  # noqa: E402


def _response(text="hello"):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])


class RecordingClient:
    """An OpenAI-shaped client that records every create() call's keyword arguments and returns, or raises,
    what it is given."""

    def __init__(self, outcome=None):
        self.calls = []
        self._outcome = _response() if outcome is None else outcome

        def create(**kwargs):
            self.calls.append(kwargs)
            if isinstance(self._outcome, Exception):
                raise self._outcome
            return self._outcome

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))


class HttpError(Exception):
    """Shaped like the SDK's status errors: a `status_code` and the parsed response `body`."""

    def __init__(self, status_code, body):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code
        self.body = body


class CompleteSendsExactlyTheRequestItAlwaysSent(unittest.TestCase):
    """Moving `complete` must not change a request. Both the reviewers and the compression tools rely on it:
    a system message, a user message, `temperature=0` and nothing else."""

    def test_messages_and_temperature(self):
        client = RecordingClient()
        self.assertEqual(openai_lib.complete(client, "the-model", "be brief", "the content"), "hello")
        self.assertEqual(client.calls, [{
            "model": "the-model",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "the content"},
            ],
            "temperature": 0,
        }])

    def test_a_reply_with_no_content_is_an_empty_string_not_a_failure(self):
        self.assertEqual(openai_lib.complete(RecordingClient(_response(None)), "m", "s", "c"), "")
        self.assertEqual(openai_lib.take_failure(), (None, None))

    def test_compression_sends_that_same_request(self):
        """Not a second definition: compression binds the one function, and a compression run's own request has
        the same shape and temperature 0."""
        self.assertIs(local_compress_lib.complete, openai_lib.complete)
        self.assertIs(local_compress_lib.take_failure, openai_lib.take_failure)
        local_compress_lib.clear_compression_cache()  # a cached result would make no request at all
        self.addCleanup(local_compress_lib.clear_compression_cache)
        client = RecordingClient(_response("summary"))
        text = "line of build output\n" * 400
        asyncio.run(local_compress_lib.compress(text, model="m", client=client))
        self.assertTrue(client.calls, "compress made no model request")
        for call in client.calls:
            self.assertEqual(call["temperature"], 0)
            self.assertEqual([m["role"] for m in call["messages"]], ["system", "user"])
            self.assertEqual(set(call), {"model", "messages", "temperature"})

    def test_yes_probability_asks_for_three_tokens_with_probabilities(self):
        client = RecordingClient(types.SimpleNamespace(choices=[]))
        self.assertIsNone(openai_lib.yes_probability(client, "m", "s", "q"))
        call = client.calls[0]
        self.assertEqual(call["temperature"], 0)
        self.assertEqual((call["max_tokens"], call["logprobs"], call["top_logprobs"]), (3, True, 10))
        self.assertEqual(openai_lib.take_failure(), ("no_probabilities", None))


class FailureReasonsAreReportedFromTheSharedThreadLocal(unittest.TestCase):
    """A failed request keeps the `Optional[str]` contract (None) and records why, on one thread-local, for
    whoever reads it on that thread: the reviewers and the compression tools alike."""

    def test_timeout(self):
        self.assertIsNone(openai_lib.complete(RecordingClient(openai_lib.CompletionTimeout("slow")), "m", "s", "c"))
        self.assertEqual(openai_lib.take_failure(), ("timeout", None))

    def test_a_400_is_a_bad_request_carrying_the_servers_message_as_sent(self):
        error = HttpError(400, {"error": "context overflow"})
        self.assertIsNone(openai_lib.complete(RecordingClient(error), "m", "s", "c"))
        self.assertEqual(openai_lib.take_failure(), ("bad_request", "context overflow"))

    def test_anything_else_is_other(self):
        for error in (OSError("connection refused"), HttpError(404, {"error": "no such model"}), HttpError(500, None)):
            with self.subTest(error=error):
                self.assertIsNone(openai_lib.complete(RecordingClient(error), "m", "s", "c"))
                self.assertEqual(openai_lib.take_failure(), ("other", None))

    def test_reading_clears_and_peeking_does_not(self):
        openai_lib.complete(RecordingClient(openai_lib.CompletionTimeout("slow")), "m", "s", "c")
        self.assertEqual(openai_lib.peek_failure(), ("timeout", None))
        self.assertEqual(openai_lib.peek_failure(), ("timeout", None))
        self.assertEqual(openai_lib.take_failure_kind(), "timeout")
        self.assertEqual(openai_lib.take_failure(), (None, None))

    def test_a_success_clears_a_stale_reason(self):
        openai_lib.complete(RecordingClient(OSError("down")), "m", "s", "c")
        openai_lib.complete(RecordingClient(), "m", "s", "c")
        self.assertEqual(openai_lib.take_failure(), (None, None))

    def test_the_reason_is_per_thread(self):
        """The relevance classifier and the reviewer scheduler make requests from worker threads concurrently; a
        failure on one must not be read by another."""
        openai_lib.take_failure()
        seen = {}

        def worker():
            openai_lib.complete(RecordingClient(openai_lib.CompletionTimeout("slow")), "m", "s", "c")
            seen["worker"] = openai_lib.take_failure()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertEqual(seen["worker"], ("timeout", None))
        self.assertEqual(openai_lib.take_failure(), (None, None))


class BuildOpenAIClientTranslatesTheSdksTimeout(unittest.TestCase):
    """The request functions tell a timeout from an outage by CompletionTimeout; the client built here is what
    turns the `openai` SDK's own timeout into it. (Moved from tests/test_lmstudio_client.py.)"""

    @staticmethod
    def _wrapped(exc, timeout_error):
        def create(**kwargs):
            raise exc

        inner = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)),
            timeout=3.0,
            max_retries=0,
        )
        return openai_lib.OpenAIClient(inner, timeout_error)

    @staticmethod
    def _request():
        import httpx
        return httpx.Request("POST", "http://localhost:1234/v1")

    def test_openai_timeout_becomes_completion_timeout(self):
        import openai
        c = self._wrapped(openai.APITimeoutError(request=self._request()), openai.APITimeoutError)
        with self.assertRaises(openai_lib.CompletionTimeout):
            c.chat.completions.create(model="m", messages=[])

    def test_connection_error_is_not_translated(self):
        import openai
        c = self._wrapped(openai.APIConnectionError(request=self._request()), openai.APITimeoutError)
        with self.assertRaises(openai.APIConnectionError) as ctx:
            c.chat.completions.create(model="m", messages=[])
        self.assertNotIsInstance(ctx.exception, openai_lib.CompletionTimeout)

    def test_the_built_client_is_bounded_and_otherwise_the_sdks_own(self):
        c = openai_lib.build_openai_client("http://localhost:1234/v1", "key", 2.5)
        self.assertEqual((c.timeout, c.max_retries), (2.5, openai_lib.CLIENT_MAX_RETRIES))
        self.assertEqual(openai_lib.CLIENT_MAX_RETRIES, 0)
        self.assertTrue(hasattr(c, "models"))

    def test_an_unbounded_timeout_is_refused_rather_than_built(self):
        for bad in (float("inf"), float("nan"), 0, -1):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                openai_lib.build_openai_client("http://localhost:1234/v1", "key", bad)


class TheServersTextIsKeptAsSentAndRedactedWhereItIsShown(unittest.TestCase):
    """`openai_lib` does not decide what is safe to show. The detail it records is the server's text,
    unmodified, and every place that puts it in a message passes it through
    credential_redaction_lib.display_safe. A 400's text can echo the prompt, credentials included."""

    SECRET = "sk-" + "a1B2c3D4" * 5

    def _seed_a_400(self, *, probability=False):
        error = HttpError(400, {"error": f"too long; the prompt held api_key={self.SECRET}\nand more"})
        call = openai_lib.yes_probability if probability else openai_lib.complete
        self.assertIsNone(call(RecordingClient(error), "m", "s", "c"))

    def _assert_safe(self, text):
        self.assertNotIn(self.SECRET, text)
        self.assertNotIn(self.SECRET[:20], text)
        self.assertIn("[REDACTED-CREDENTIAL]", text)
        self.assertNotIn("\n", text)

    def test_openai_lib_records_the_text_exactly_as_sent(self):
        self._seed_a_400()
        kind, detail = openai_lib.take_failure()
        self.assertEqual(kind, "bad_request")
        self.assertEqual(detail, f"too long; the prompt held api_key={self.SECRET}\nand more")

    def test_the_reasoning_reviewer_redacts_it(self):
        import reasoning_reviewer_lib
        self._seed_a_400()
        self._assert_safe(reasoning_reviewer_lib._failure_text())

    def test_the_verifier_redacts_it(self):
        import verifier_reviewer_lib
        self._seed_a_400()
        self._assert_safe(verifier_reviewer_lib._failure_text())

    def test_the_classifier_redacts_it(self):
        import classifier_reviewer_lib
        self._seed_a_400(probability=True)
        said, kept, log = classifier_reviewer_lib._failure_text("m")
        for text in (said, kept):
            self._assert_safe(text)
        self.assertNotIn(self.SECRET, log)

    def test_the_harness_probe_redacts_it(self):
        import harness_lib
        error = HttpError(400, {"error": f"too long; the prompt held api_key={self.SECRET}"})
        loader = mock.Mock()
        with mock.patch.object(harness_lib, "load_server", return_value=mock.Mock(lmstudio_base_url=lambda: "http://x/v1")), \
                mock.patch("lmstudio_model_loader_lib.ModelLoader", return_value=loader), \
                mock.patch("lmstudio_client.build_client", return_value=RecordingClient(error)):
            problem = harness_lib.probe_answers_yes_no("m", on_demand=False, context=1000)
        assert problem is not None
        self.assertNotIn(self.SECRET, problem)
        self.assertNotIn(self.SECRET[:20], problem)
        self.assertIn("[REDACTED-CREDENTIAL]", problem)


# The files on the reviewer side. None may import the compression library, so a request setting added
# for a reviewer is never a change to compression (and the reverse).
REVIEWER_SIDE = (
    "libs/reasoning_reviewer_lib.py",
    "libs/verifier_reviewer_lib.py",
    "libs/classifier_reviewer_lib.py",
    "mcpservers/codebase_reviewer_mcp_server.py",
)


def _reviewer_side_files():
    files = [os.path.join(ROOT, rel) for rel in REVIEWER_SIDE]
    harness = os.path.join(ROOT, "harness")
    files += [os.path.join(harness, n) for n in sorted(os.listdir(harness)) if n.endswith(".py")]
    return files


def _imported_modules(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    names = set()
    for node in ast.walk(tree):  # walk, not iter_child_nodes: the harness imports inside functions
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


class ReviewersDoNotImportCompression(unittest.TestCase):
    def test_no_reviewer_side_file_imports_local_compress_lib(self):
        files = _reviewer_side_files()
        self.assertGreater(len(files), len(REVIEWER_SIDE), "the harness/*.py files were not found")
        for path in files:
            with self.subTest(file=os.path.relpath(path, ROOT)):
                self.assertNotIn("local_compress_lib", _imported_modules(path))

    def test_importing_the_reviewers_does_not_load_it_transitively_either(self):
        """A direct import is not the only way in: a reviewer module could reach compression through something it
        imports. Import each in a fresh interpreter and check what got loaded."""
        for rel in REVIEWER_SIDE + ("harness/harness_lib.py",):
            name = os.path.splitext(os.path.basename(rel))[0]
            with self.subTest(module=rel):
                code = (
                    "import sys\n"
                    f"sys.path[:0] = [{LIBS!r}, {os.path.join(ROOT, 'harness')!r}, {os.path.join(ROOT, 'mcpservers')!r}]\n"
                    f"import {name}\n"
                    "assert 'local_compress_lib' not in sys.modules, 'local_compress_lib was loaded'\n"
                )
                r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, r.stderr[-1500:])

    def test_compression_and_the_reviewers_read_one_failure_state(self):
        """The thread-local is defined once, in openai_lib: a reason recorded during a real compression run is the
        one a reviewer reading `openai_lib.take_failure()` would see. Driven through the compression library's own
        request path (its `complete` binding and its own take_failure() read), not by calling complete() directly."""
        self.assertIs(local_compress_lib.take_failure, openai_lib.take_failure)
        openai_lib.take_failure()
        local_compress_lib.clear_compression_cache()  # an earlier test compressed similar text; force real requests
        self.addCleanup(local_compress_lib.clear_compression_cache)
        seen = []
        real_take = local_compress_lib.take_failure

        def spy():
            # Peek at the shared state as compression reads it, then let compression consume it as it would.
            seen.append(openai_lib.peek_failure())
            return real_take()

        text = "line of build output\n" * 400
        with mock.patch.object(local_compress_lib, "take_failure", spy):
            out = asyncio.run(local_compress_lib.compress(
                text, model="m", client=RecordingClient(openai_lib.CompletionTimeout("slow")),
            ))
        self.assertIn(("timeout", None), seen)
        self.assertIn("timed out", out)


if __name__ == "__main__":
    unittest.main()
