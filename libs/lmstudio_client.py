"""
What is specific to LM Studio when this repo talks to a local model: where it
listens, how long one request may take, and the client built for it.

Everything about the request itself (the OpenAI-shaped completion call, the
failure and usage side channels, the bounded client and its timeout
translation) lives in libs/openai_lib.py, which this module imports to build the
client. Nothing here imports `openai`: the SDK is imported only when a client is
built (`build_client`), so this module, like libs/local_compress_lib.py which
reads its constants from here, imports in an environment without `openai` (a
Qdrant-only install).

Two things live here and nowhere else:
  - the LM Studio configuration: the base URL (CLAUDE_RUNWAY_LMSTUDIO_URL), and
    the per-request timeout (CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS, or an
    explicit one) with its default. The default is finite and the client is
    built with `max_retries=0`, because the SDK's own defaults (a 600s read
    timeout plus 2 retries) let a hung LM Studio stall the synchronous
    PostToolUse hook for ~30 minutes. "Is this timeout usable" is
    libs/openai_lib.py's `is_bounded_timeout`, applied to both the env var and an
    explicit argument so neither path can smuggle in an unbounded client.
  - the LM Studio api key the SDK insists on (LM Studio ignores it).
"""

import os
from typing import Optional

from openai_lib import OpenAIClient, build_openai_client, is_bounded_timeout

# The client type callers annotate with; the translation of the SDK's timeout
# into the library's CompletionTimeout is libs/openai_lib.py's.
LMStudioClient = OpenAIClient

DEFAULT_BASE_URL = os.environ.get("CLAUDE_RUNWAY_LMSTUDIO_URL", "http://localhost:1234/v1")

# Per-request budget for every LM Studio completion call, in seconds. A
# non-streaming completion only returns once generation finishes, so this is
# effectively the total budget for one request. That matters most for the
# PostToolUse hook, which runs synchronously in Claude Code's own tool-result
# path, so a hang there stalls the whole session, not just one tool call.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
REQUEST_TIMEOUT_ENV_VAR = "CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS"

_API_KEY = "lm-studio"


def resolve_request_timeout() -> float:
    """
    Returns the per-request timeout in seconds: CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS
    if it parses as a finite number > 0, else DEFAULT_REQUEST_TIMEOUT_SECONDS.
    Read at call time (not import time) so a test or long-lived MCP server
    process sees the current environment. An unset, blank, non-numeric,
    non-finite or non-positive value silently falls back to the default
    rather than raising: this runs inside the hook, where a bad env value
    must never crash it (and 0/negative/inf/nan would otherwise mean
    "no real bound" to the SDK, defeating the point of this setting).
    """
    raw = os.environ.get(REQUEST_TIMEOUT_ENV_VAR, "")
    try:
        value = float(raw.strip())
    except ValueError:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    return value if is_bounded_timeout(value) else DEFAULT_REQUEST_TIMEOUT_SECONDS


def build_client(base_url: Optional[str], timeout: Optional[float] = None) -> LMStudioClient:
    """
    Builds the bounded client. `timeout` overrides the resolved per-request
    budget for a caller that needs a shorter one; when omitted,
    resolve_request_timeout() supplies it. An explicit value that isn't finite
    and > 0 is ignored in favor of the resolved budget (same rule as the env
    var). max_retries is always 0. `base_url` falls back to DEFAULT_BASE_URL.
    Importing `openai` happens here, so a missing SDK surfaces as an ImportError
    from this call.
    """
    return build_openai_client(
        base_url or DEFAULT_BASE_URL,
        _API_KEY,
        timeout if timeout is not None and is_bounded_timeout(timeout) else resolve_request_timeout(),
    )
