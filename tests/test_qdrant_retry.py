#!/usr/bin/env python3
"""Tests for libs/qdrant_retry.py (issue #75).

Stdlib-only (unittest, no pytest), no network and no real Qdrant/Docker --
uses fake sync/async functions that raise on their first N calls to stand in
for a dropped pooled connection, since the actual Docker Desktop vpnkit
idle-reap behavior this guards against isn't something a unit test can
reliably simulate (see the module docstring and 75-plan.md's "Known
limitation" section).

time.sleep/asyncio.sleep are patched out everywhere here so the backoff
between attempts (real behavior, added after real-world testing showed a
single retry wasn't enough -- see qdrant_retry.py's module docstring)
doesn't slow this test file down; MAX_ATTEMPTS * RETRY_BACKOFF_SECONDS
would otherwise add real wall-clock delay to the exhausted-retry tests.

    .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import httpx  # noqa: E402
from qdrant_client.http.exceptions import ResponseHandlingException  # noqa: E402

import qdrant_retry  # noqa: E402
from qdrant_retry import call_with_retry, async_call_with_retry  # noqa: E402

MAX_ATTEMPTS = qdrant_retry.MAX_ATTEMPTS  # read from the module, not hardcoded, so tests track it if ever tuned


class _FlakyCounter:
    """Raises `exc` on its first `fail_times` calls, then returns `result`."""

    def __init__(self, fail_times, exc, result="ok"):
        self.fail_times = fail_times
        self.exc = exc
        self.result = result
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return self.result


class _AsyncFlakyCounter(_FlakyCounter):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


@patch("qdrant_retry.time.sleep")
class CallWithRetrySync(unittest.TestCase):
    def test_success_on_first_try_calls_once(self, mock_sleep):
        fn = _FlakyCounter(fail_times=0, exc=httpx.ConnectError("unused"))
        result = call_with_retry(fn, 1, key="value")
        self.assertEqual(result, "ok")
        self.assertEqual(fn.calls, 1)
        mock_sleep.assert_not_called()

    def test_retry_succeeds_calls_twice(self, mock_sleep):
        fn = _FlakyCounter(fail_times=1, exc=httpx.RemoteProtocolError("Server disconnected"))
        result = call_with_retry(fn)
        self.assertEqual(result, "ok")
        self.assertEqual(fn.calls, 2, "must succeed as soon as an attempt succeeds, not keep retrying")
        self.assertEqual(mock_sleep.call_count, 1, "exactly one backoff between the 2 attempts made")

    def test_succeeds_on_the_last_allowed_attempt(self, mock_sleep):
        """The scenario the MAX_ATTEMPTS bump was made for: real-world
        testing found a single retry (2 total attempts) wasn't enough over
        a large enough sequential loop -- a run can fail its first
        MAX_ATTEMPTS-1 attempts and still succeed within the bound."""
        fn = _FlakyCounter(fail_times=MAX_ATTEMPTS - 1, exc=httpx.ConnectError("flaky"))
        result = call_with_retry(fn)
        self.assertEqual(result, "ok")
        self.assertEqual(fn.calls, MAX_ATTEMPTS)
        self.assertEqual(mock_sleep.call_count, MAX_ATTEMPTS - 1)

    def test_retry_exhausted_reraises(self, mock_sleep):
        fn = _FlakyCounter(fail_times=MAX_ATTEMPTS, exc=httpx.ConnectError("still down"))
        with self.assertRaises(httpx.ConnectError):
            call_with_retry(fn)
        self.assertEqual(fn.calls, MAX_ATTEMPTS, "must stop at MAX_ATTEMPTS, not retry forever")
        self.assertEqual(mock_sleep.call_count, MAX_ATTEMPTS - 1, "no backoff after the final failed attempt")

    def test_non_transient_error_is_not_retried(self, mock_sleep):
        fn = _FlakyCounter(fail_times=1, exc=ValueError("unrelated bug"))
        with self.assertRaises(ValueError):
            call_with_retry(fn)
        self.assertEqual(fn.calls, 1, "a non-transient error must fail immediately, no retry")
        mock_sleep.assert_not_called()

    def test_args_and_kwargs_are_forwarded_on_retry(self, mock_sleep):
        seen = []

        def fn(*args, **kwargs):
            seen.append((args, kwargs))
            if len(seen) == 1:
                raise httpx.ConnectError("dead")
            return "ok"

        result = call_with_retry(fn, "a", b=2)
        self.assertEqual(result, "ok")
        self.assertEqual(seen, [(("a",), {"b": 2}), (("a",), {"b": 2})])

    # The tests below exercise qdrant-client's REAL failure shape -- caught
    # on PR review, confirmed by reading qdrant-client's source: both its
    # sync and async API clients wrap EVERY transport exception in
    # ResponseHandlingException(source=<original>), so a fake that raises
    # the raw httpx error directly (as the tests above do) doesn't actually
    # prove the retry fires against a real qdrant-client call. These do.

    def test_wrapped_transient_error_retries_and_succeeds(self, mock_sleep):
        fn = _FlakyCounter(fail_times=1, exc=ResponseHandlingException(httpx.RemoteProtocolError("disconnected")))
        result = call_with_retry(fn)
        self.assertEqual(result, "ok")
        self.assertEqual(fn.calls, 2, "must unwrap ResponseHandlingException.source to see the transient error")

    def test_wrapped_transient_error_exhausted_reraises_the_wrapper(self, mock_sleep):
        fn = _FlakyCounter(fail_times=MAX_ATTEMPTS, exc=ResponseHandlingException(httpx.ConnectError("still down")))
        with self.assertRaises(ResponseHandlingException):
            call_with_retry(fn)
        self.assertEqual(fn.calls, MAX_ATTEMPTS, "must stop at MAX_ATTEMPTS")

    def test_wrapped_non_transient_error_is_not_retried(self, mock_sleep):
        fn = _FlakyCounter(fail_times=1, exc=ResponseHandlingException(ValueError("unrelated, just happens to be wrapped")))
        with self.assertRaises(ResponseHandlingException):
            call_with_retry(fn)
        self.assertEqual(fn.calls, 1, "a wrapped non-transient source must fail immediately, no retry")


@patch("qdrant_retry.asyncio.sleep", new_callable=AsyncMock)
class AsyncCallWithRetry(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_success_on_first_try_calls_once(self, mock_sleep):
        fn = _AsyncFlakyCounter(fail_times=0, exc=httpx.ConnectError("unused"))
        result = self._run(async_call_with_retry(fn, 1, key="value"))
        self.assertEqual(result, "ok")
        self.assertEqual(fn.calls, 1)
        mock_sleep.assert_not_called()

    def test_retry_succeeds_calls_twice(self, mock_sleep):
        fn = _AsyncFlakyCounter(fail_times=1, exc=httpx.RemoteProtocolError("Server disconnected"))
        result = self._run(async_call_with_retry(fn))
        self.assertEqual(result, "ok")
        self.assertEqual(fn.calls, 2, "must succeed as soon as an attempt succeeds, not keep retrying")
        self.assertEqual(mock_sleep.call_count, 1)

    def test_succeeds_on_the_last_allowed_attempt(self, mock_sleep):
        fn = _AsyncFlakyCounter(fail_times=MAX_ATTEMPTS - 1, exc=httpx.ConnectError("flaky"))
        result = self._run(async_call_with_retry(fn))
        self.assertEqual(result, "ok")
        self.assertEqual(fn.calls, MAX_ATTEMPTS)
        self.assertEqual(mock_sleep.call_count, MAX_ATTEMPTS - 1)

    def test_retry_exhausted_reraises(self, mock_sleep):
        fn = _AsyncFlakyCounter(fail_times=MAX_ATTEMPTS, exc=httpx.ConnectError("still down"))
        with self.assertRaises(httpx.ConnectError):
            self._run(async_call_with_retry(fn))
        self.assertEqual(fn.calls, MAX_ATTEMPTS, "must stop at MAX_ATTEMPTS, not retry forever")

    def test_non_transient_error_is_not_retried(self, mock_sleep):
        fn = _AsyncFlakyCounter(fail_times=1, exc=ValueError("unrelated bug"))
        with self.assertRaises(ValueError):
            self._run(async_call_with_retry(fn))
        self.assertEqual(fn.calls, 1, "a non-transient error must fail immediately, no retry")
        mock_sleep.assert_not_called()

    def test_wrapped_transient_error_retries_and_succeeds(self, mock_sleep):
        fn = _AsyncFlakyCounter(fail_times=1, exc=ResponseHandlingException(httpx.RemoteProtocolError("disconnected")))
        result = self._run(async_call_with_retry(fn))
        self.assertEqual(result, "ok")
        self.assertEqual(fn.calls, 2, "must unwrap ResponseHandlingException.source to see the transient error")

    def test_wrapped_transient_error_exhausted_reraises_the_wrapper(self, mock_sleep):
        fn = _AsyncFlakyCounter(fail_times=MAX_ATTEMPTS, exc=ResponseHandlingException(httpx.ConnectError("still down")))
        with self.assertRaises(ResponseHandlingException):
            self._run(async_call_with_retry(fn))
        self.assertEqual(fn.calls, MAX_ATTEMPTS, "must stop at MAX_ATTEMPTS")

    def test_wrapped_non_transient_error_is_not_retried(self, mock_sleep):
        fn = _AsyncFlakyCounter(fail_times=1, exc=ResponseHandlingException(ValueError("unrelated, just happens to be wrapped")))
        with self.assertRaises(ResponseHandlingException):
            self._run(async_call_with_retry(fn))
        self.assertEqual(fn.calls, 1, "a wrapped non-transient source must fail immediately, no retry")


if __name__ == "__main__":
    unittest.main()
