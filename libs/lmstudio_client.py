"""
The one place that imports `openai` for local compression.

libs/local_compress_lib.py is stdlib-only and takes the model client as an
argument; this module builds that client so the setup is not copied into every
caller (the compression MCP server and the compress hook). Importing this
module is what requires `openai`, so a Qdrant-only install, which has no
`openai`, never touches it.

Two things live here and nowhere else:
  - the bound on every request: a finite timeout (CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS,
    or an explicit one) and `max_retries=0`, because the SDK's own defaults (a
    600s read timeout plus 2 retries) let a hung LM Studio stall the
    synchronous PostToolUse hook for ~30 minutes. The constants and the
    "is this timeout usable" rule are read from local_compress_lib so the
    error wording there ("timed out after Ns") and the client always agree.
  - the translation of `openai`'s timeout exception into the library's own
    CompletionTimeout, so the library can say "timed out" versus "unreachable"
    without importing `openai`.
"""

from typing import Optional

from openai import APITimeoutError, OpenAI

from local_compress_lib import (
    CLIENT_MAX_RETRIES,
    DEFAULT_BASE_URL,
    CompletionTimeout,
    is_bounded_timeout,
    resolve_request_timeout,
)


class _Completions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, *args, **kwargs):
        try:
            return self._inner.create(*args, **kwargs)
        except APITimeoutError as e:
            # APITimeoutError subclasses APIConnectionError in the SDK; only
            # the timeout becomes CompletionTimeout, everything else passes
            # through for the library's generic failure handling.
            raise CompletionTimeout(str(e)) from e


class _Chat:
    def __init__(self, inner):
        self.completions = _Completions(inner.completions)


class LMStudioClient:
    """An `openai.OpenAI` client whose chat completions raise CompletionTimeout
    on a timeout. Every other attribute (`timeout`, `max_retries`, `models`,
    ...) is the wrapped client's own."""

    def __init__(self, inner: OpenAI):
        self._inner = inner
        self.chat = _Chat(inner.chat)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def build_client(base_url: Optional[str], timeout: Optional[float] = None) -> LMStudioClient:
    """
    Builds the bounded client. `timeout` overrides the resolved per-request
    budget for a caller that needs a shorter one; when omitted,
    resolve_request_timeout() supplies it. An explicit value that isn't finite
    and > 0 is ignored in favor of the resolved budget (same rule as the env
    var). max_retries is always 0. `base_url` falls back to the library's
    DEFAULT_BASE_URL.
    """
    return LMStudioClient(
        OpenAI(
            base_url=base_url or DEFAULT_BASE_URL,
            api_key="lm-studio",
            timeout=timeout if timeout is not None and is_bounded_timeout(timeout) else resolve_request_timeout(),
            max_retries=CLIENT_MAX_RETRIES,
        )
    )
