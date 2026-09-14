"""
Retry helper for transient Qdrant connection drops (issue #75).

Docker Desktop for Mac's userspace networking proxy (vpnkit) can silently
close an idle pooled keep-alive HTTP connection to Qdrant during the
CPU-bound gap while the local embedding model loads (FastEmbedProvider
construction) -- vpnkit's idle-connection reaper doesn't honor TCP
keep-alives as "non-idle" activity, so this can happen even on a connection
the client believes is still healthy. The client then raises
httpx.RemoteProtocolError ("Server disconnected without sending a
response") or httpx.ConnectError on the next call that tries to reuse that
dead connection.

Confirmed via Qdrant's own container logs (see issue #75) that the failing
request never reaches the container at all -- the connection is torn down
between the client and the container, not by Qdrant. That makes retrying
safe here: httpx evicts a connection from its pool once it fails, so a
retried call opens a fresh one rather than reusing the same dead one.

CAUGHT ON PR REVIEW (real bug, confirmed by reading qdrant-client's source
directly, not just trusting the reviewer): qdrant-client's own API client
--  both ApiClient.send_inner (sync) and AsyncApiClient.send_inner (async),
which back QdrantClient and QdrantConnector's AsyncQdrantClient
respectively -- wrap EVERY transport exception in
qdrant_client.http.exceptions.ResponseHandlingException(source=<original
exception>), rather than letting it propagate directly:

    try:
        response = self._client.send(request)
    except Exception as e:
        raise ResponseHandlingException(e)

So the raw httpx.RemoteProtocolError/ConnectError this module was written to
catch is NEVER what actually reaches call_with_retry/async_call_with_retry
in real usage -- it's always wrapped. The original version of this file
only caught the unwrapped types, so the retry silently never fired against
a real Qdrant client at all; the unit tests didn't catch this because they
called fake functions that raised the raw exception directly instead of
going through qdrant-client's real wrapping. Fixed by unwrapping
ResponseHandlingException.source (a plain attribute, not nested further)
and checking IT against the transient types, in addition to the raw types
in case some future/other call path raises them unwrapped.

FOUND BY REAL-WORLD TESTING ON A LARGE REPO (10,532+ files): a single retry
(2 total attempts) still isn't enough on a big enough sequential loop.
sync_repo's delete loop batching (see DELETE_BATCH_SIZE in
ingest_mcp_server.py) fixed the failure there, but the SAME class of drop
then reliably resurfaced deep in the per-chunk upsert loop instead --
observed failing near call ~5,249 in one run, after thousands of prior
calls all returned 200 OK. The mechanism is statistical, not a stuck
state: even a small per-call drop probability compounds across thousands
of sequential calls until eventually two consecutive attempts (original +
single retry) both land on a dead connection. Confirmed this isn't a
resource leak or degraded container state (which a Docker restart would
fix) -- the failure point tracked with how many requests had been made in
that run (~60, ~10,532, ~5,249+ across different runs), not with elapsed
time or session history. A bounded retry loop targets the actual
mechanism: MAX_ATTEMPTS consecutive drops in a row is vanishingly
unlikely even though back-to-back pairs eventually happen over a long
enough single-retry sequence.

Deliberately a separate module from qdrant_ingest_lib.py, which documents
itself as intentionally dependency-free (no qdrant-client/mcp_server_qdrant
imports) for easy unit testing. This module's whole purpose is retrying an
HTTP-level failure, so importing httpx here is unavoidable.
"""

import asyncio
import time

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException

# Both exception types are confirmed reachable from this code: QdrantClient
# (sync) uses httpx.Client internally, and QdrantConnector (async, from
# mcp_server_qdrant) uses AsyncQdrantClient (async httpx) internally. In
# practice both arrive wrapped in ResponseHandlingException -- see
# _is_transient below -- but the raw types are also checked directly in
# case some call path ever raises them unwrapped.
TRANSIENT_HTTP_ERRORS = (httpx.RemoteProtocolError, httpx.ConnectError)

# Total attempts (the original call plus retries), not just "retries" --
# bumped from 2 (a single retry) after real-world testing showed that's
# still not enough over a large enough sequential loop (see module
# docstring). A short backoff between attempts gives vpnkit's connection
# state a moment to settle rather than hammering it with an instant retry.
MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 0.2


def _is_transient(exc: Exception) -> bool:
    """True if `exc` is (or wraps, via ResponseHandlingException.source) one
    of TRANSIENT_HTTP_ERRORS. See module docstring for why the wrapped case
    is the one that actually matters in real usage."""
    if isinstance(exc, TRANSIENT_HTTP_ERRORS):
        return True
    if isinstance(exc, ResponseHandlingException) and isinstance(exc.source, TRANSIENT_HTTP_ERRORS):
        return True
    return False


def call_with_retry(fn, *args, **kwargs):
    """Call a sync function, retrying on a transient disconnect up to
    MAX_ATTEMPTS total attempts, with a short backoff between attempts."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_transient(e) or attempt == MAX_ATTEMPTS:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS)


async def async_call_with_retry(fn, *args, **kwargs):
    """Call an async function, retrying on a transient disconnect up to
    MAX_ATTEMPTS total attempts, with a short backoff between attempts."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if not _is_transient(e) or attempt == MAX_ATTEMPTS:
                raise
            await asyncio.sleep(RETRY_BACKOFF_SECONDS)
