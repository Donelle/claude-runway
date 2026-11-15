"""
Talking to an OpenAI-compatible chat-completions endpoint, with no knowledge of
which server is behind it (libs/lmstudio_client.py adds the LM Studio specifics).

Two halves, both importable without `openai` installed:
  - the request functions (`complete`, `yes_probability`) and their side
    channels (why the last request on this thread failed, its token counts).
    They take the client as an argument and only call
    `client.chat.completions.create(...)`, so they depend on no SDK. They live
    here, and not in libs/local_compress_lib.py, so the codebase-reviewer
    (libs/reasoning_reviewer_lib.py, libs/verifier_reviewer_lib.py,
    libs/classifier_reviewer_lib.py, the harness) can send a request without
    importing the compression library, and a change to a request setting here is
    a change to a model call, not to compression. Compression imports these too,
    so the reviewers and the compression tools read ONE failure state.
  - `build_openai_client`, which builds the bounded client. It imports
    `openai` inside the function, so importing this module never needs the SDK
    (a Qdrant-only install has none); only building a client does.
"""

import math
import threading
from typing import Any, Optional, Tuple


# Every client built here is BOUNDED: a finite timeout and no retries. A
# non-streaming completion only returns once generation finishes, so the
# timeout is effectively the total budget for one request. Retries are off
# because a local server that timed out or refused will not be fixed by
# immediately asking again (it would only multiply the stall), and every caller
# already fails open on an error. The SDK's own defaults (a 600s read timeout
# plus 2 retries) let a hung server stall the synchronous PostToolUse hook for
# ~30 minutes.
CLIENT_MAX_RETRIES = 0


class CompletionTimeout(Exception):
    """A model request did not finish within its time budget.

    The neutral, library-owned stand-in for a client SDK's own timeout
    exception: whichever client the caller passes in must raise this (and
    only this) for a timeout, so the request functions below can tell "timed
    out" from "could not connect" without importing the SDK. The client built by
    `build_openai_client` does that translation for the `openai` SDK.
    """


def is_bounded_timeout(value: float) -> bool:
    """
    The single definition of a usable timeout, applied to BOTH the env var
    and an explicit `build_client(timeout=...)` argument
    (libs/lmstudio_client.py) so neither path can smuggle in an unbounded client: finite and > 0. The
    SDK would treat inf as no bound, and 0/negative/nan as broken or
    immediate-fail.
    """
    return math.isfinite(value) and value > 0


# Why the most recent complete() or yes_probability() call on THIS thread
# failed, as (kind, detail), or (None, None); take_failure() lists the kinds.
# Both functions deliberately keep their Optional return contract (many callers
# and tests rely on None meaning "the request failed"), so the reason travels on
# this side channel instead of widening the return type.
# Thread-local because the relevance classifier and the reviewer scheduler make
# requests from worker threads concurrently; read it with take_failure() (or
# peek_failure() to leave it in place) on the same thread that made the call.
_failure_state = threading.local()


def take_failure() -> Tuple[Optional[str], Optional[str]]:
    """Return and clear (kind, detail) for this thread's last complete() failure.

    kind:
    "timeout" -- the request exceeded the per-request timeout (LM Studio may
    be busy, e.g. its parallel-request slots are all taken and the request sat
    in its queue, or unreachable: the SDK's timeout error also covers a
    connection that never got established); "bad_request" -- LM Studio was
    reached and answered HTTP 400 (most often the chunk does not
    fit the loaded model's context window, so the server IS running and
    "check it's still running" is the wrong advice); "other" -- anything else
    (connection refused, model not found, ...); None -- no reason was
    recorded, e.g. complete() was replaced by a test stub, or nothing failed.
    Callers must treat None as "unknown" and fall back to the generic failure
    wording.

    detail: the server's own message for a "bad_request", EXACTLY as the server sent it (extracted by
    server_error_text, nothing removed or shortened), otherwise None; or, for "no_answer", the model's
    likeliest first tokens. This module does not decide what is safe to show: the text can echo the model's own
    input, including a credential, so whoever puts it in a message or a log passes it through
    credential_redaction_lib.display_safe first.

    Clearing on read means a stale reason can never leak into a later,
    unrelated failure.
    """
    kind = getattr(_failure_state, "kind", None)
    detail = getattr(_failure_state, "detail", None)
    _failure_state.kind = None
    _failure_state.detail = None
    return kind, detail


def peek_failure() -> Tuple[Optional[str], Optional[str]]:
    """Like take_failure(), but leaves the reason in place. For a caller that only RECORDS why a request
    failed (the codebase-reviewer's run database) and must not steal the reason from the code that words
    the user-facing message with take_failure()."""
    return getattr(_failure_state, "kind", None), getattr(_failure_state, "detail", None)


# The token counts of this thread's last successful complete(), on the same side channel as the failure
# reason and for the same reason: the return type stays the reply text. Cleared at the start of every call,
# so a stub that replaces complete() can never leave a previous call's counts behind.
_usage_state = threading.local()


def take_usage() -> Optional[dict]:
    """Return and clear the token counts and timing of this thread's last successful complete() call, or None
    when the server reported nothing. Keys (each an int, float, string or None):

      prompt_tokens, completion_tokens, reasoning_tokens   from the response's `usage`
      ttft_ms               time to first token, from `stats.time_to_first_token` (seconds on the wire)
      inter_token_ms        average time between tokens: 1000 / `stats.tokens_per_second`
      tokens_per_second     `stats.tokens_per_second`
      generation_ms         `stats.generation_time`. What it covers depends on the runtime: LM Studio's llama.cpp (GGUF)
                            runtime counts the first token's wait in it, its MLX runtime does not (checked live), so use
                            the measured latency for a total and `inter_token_ms` for the speed of generation
      stop_reason           `stats.stop_reason` ("eosFound", "maxPredictedTokensReached", ...)

    `completion_tokens` is what the server says it generated; whether that includes a reasoning model's
    thinking depends on the server (LM Studio's does), and `reasoning_tokens` is filled only when the response
    breaks it out. The `stats` half exists only on LM Studio's native `/api/v0/chat/completions`; its
    OpenAI-compatible `/v1` path returns an empty `stats`, so a client built on `/v1` gets the token counts only."""
    usage = getattr(_usage_state, "usage", None)
    _usage_state.usage = None
    return usage


def _usage_from(response) -> Optional[dict]:
    """The token counts and timing of an OpenAI-shaped response, or None. Duck-typed, never raises."""
    try:
        raw = getattr(response, "usage", None)
        stats = getattr(response, "stats", None)

        def field(obj, name):
            if obj is None:
                return None
            return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

        def count(obj, name):
            value = field(obj, name)
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        def number(obj, name):
            value = field(obj, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value) if math.isfinite(value) and value >= 0 else None

        ttft = number(stats, "time_to_first_token")
        generation = number(stats, "generation_time")
        rate = number(stats, "tokens_per_second")
        reason = field(stats, "stop_reason")
        usage = {
            "prompt_tokens": count(raw, "prompt_tokens"),
            "completion_tokens": count(raw, "completion_tokens"),
            "reasoning_tokens": count(field(raw, "completion_tokens_details"), "reasoning_tokens"),
            "ttft_ms": round(ttft * 1000) if ttft is not None else None,
            "inter_token_ms": round(1000.0 / rate, 2) if rate else None,
            "tokens_per_second": round(rate, 2) if rate is not None else None,
            "generation_ms": round(generation * 1000) if generation is not None else None,
            "stop_reason": reason[:60] if isinstance(reason, str) and reason else None,
        }
        return usage if any(v is not None for v in usage.values()) else None
    except Exception:
        return None


def take_failure_kind() -> Optional[str]:
    """Return and clear why this thread's last complete() call failed (the
    kind half of take_failure(); see there for the values). Reading clears the
    detail too, so a caller that wants the detail uses take_failure()."""
    return take_failure()[0]


def server_error_text(exc: BaseException) -> Optional[str]:
    """The server's message for an HTTP error, as sent, or None if there is none.

    Source order: `exc.body` first (the OpenAI SDK parses the response JSON
    into it: LM Studio sends `{"error": "<text>"}`, which arrives as the bare
    string, while an OpenAI-shaped `{"error": {"message": ...}}` arrives as a
    dict), then `exc.message`/`str(exc)`. `.body` goes first because for
    LM Studio's string-shaped error the SDK's `.message` is the noisy
    "Error code: 400 - {'error': ...}" repr rather than the text itself.

    Not redacted, collapsed or truncated here: what is safe to show depends on where it is shown, and that is
    the caller's to decide (credential_redaction_lib.display_safe), not this module's.
    """
    body = getattr(exc, "body", None)
    text: Optional[str] = None
    if isinstance(body, str):
        text = body
    elif isinstance(body, dict):
        for candidate in (body.get("message"), body.get("error")):
            if isinstance(candidate, str):
                text = candidate
                break
            if isinstance(candidate, dict) and isinstance(candidate.get("message"), str):
                text = candidate["message"]
                break
    if not text:
        message = getattr(exc, "message", None)
        text = message if isinstance(message, str) and message else str(exc)
    return text if text.strip() else None


def complete(oai_client, model: str, system: str, content: str) -> Optional[str]:
    """Returns the completion text, or None if the request failed for any reason
    (connection error, model not found, timeout, HTTP 400, etc.) -- callers must
    check for None rather than assume the server is up because an earlier step
    (finding a model, building the client) succeeded: neither makes a request, so
    a dead server is only caught here, at the actual request.

    On failure the reason is also recorded for take_failure(): the client signals
    a timeout by raising CompletionTimeout (a timeout may subclass a connection
    error in the client's own SDK, so the client translates it explicitly rather
    than this function inferring it from "not a connection error"). `oai_client`
    is any object with an OpenAI-shaped `chat.completions.create(...)`; the
    caller builds it."""
    _failure_state.kind = None
    _failure_state.detail = None
    _usage_state.usage = None
    try:
        response = oai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0,
        )
    except CompletionTimeout:
        _failure_state.kind = "timeout"
        return None
    except Exception as e:
        # An HTTP 400 means LM Studio was reached and refused THIS request
        # -- typically a chunk whose token count overflows the
        # loaded model's context window. Detected by duck-typing the
        # `status_code` the OpenAI SDK's status errors carry, because this
        # module never imports a client SDK. Everything else (connection
        # refused, a missing model's 404, ...) stays "other".
        if getattr(e, "status_code", None) == 400:
            _failure_state.kind = "bad_request"
            _failure_state.detail = server_error_text(e)
        else:
            _failure_state.kind = "other"
        return None
    _usage_state.usage = _usage_from(response)
    return response.choices[0].message.content or ""


# The least share of the first token's probability that must sit on "yes" and "no" for yes_probability() to trust the
# split between them. A model that starts with some other word (a reasoning model's "<think>", a preamble) has put its
# probability somewhere this reading does not look.
MIN_ANSWER_MASS = 0.5
_TOP_LOGPROBS = 10


def yes_probability(oai_client, model: str, system: str, content: str) -> Optional[float]:
    """Asks one yes/no question and returns the model's probability of "yes", from 0 to 1, or None if there is none.

    The request is made at temperature 0 and asks for one token and its token probabilities; the answer is
    p(yes) / (p(yes) + p(no)) over the first token's top candidates (case and a leading space ignored), so it is the model's
    own split between the two answers, not a word it chose to write. The same question therefore gives the same number every
    time. None means no number could be read, and the reason is recorded for take_failure() as in complete(), with two
    kinds of its own: "no_probabilities" (the runtime or the model returned no token probabilities) and "no_answer"
    (the first token was something other than yes or no: less than MIN_ANSWER_MASS of its probability sat on them; the
    detail then names the likeliest first tokens and their shares)."""
    _failure_state.kind = None
    _failure_state.detail = None
    _usage_state.usage = None
    try:
        response = oai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0,
            # At least 3: LM Studio returns no token probabilities for a one-token reply.
            max_tokens=3,
            logprobs=True,
            top_logprobs=_TOP_LOGPROBS,
        )
    except CompletionTimeout:
        _failure_state.kind = "timeout"
        return None
    except Exception as e:
        if getattr(e, "status_code", None) == 400:
            _failure_state.kind = "bad_request"
            _failure_state.detail = server_error_text(e)
        else:
            _failure_state.kind = "other"
        return None
    _usage_state.usage = _usage_from(response)
    try:
        candidates = response.choices[0].logprobs.content[0].top_logprobs
    except (AttributeError, IndexError, TypeError):
        candidates = None
    if not candidates:
        _failure_state.kind = "no_probabilities"
        return None
    yes = no = 0.0
    for candidate in candidates:
        word = str(candidate.token).strip().lower()
        if word == "yes":
            yes += math.exp(candidate.logprob)
        elif word == "no":
            no += math.exp(candidate.logprob)
    if yes + no < MIN_ANSWER_MASS:
        _failure_state.kind = "no_answer"
        likeliest = sorted(candidates, key=lambda c: c.logprob, reverse=True)[:3]
        _failure_state.detail = ", ".join(f"{str(c.token)!r} {math.exp(c.logprob):.0%}" for c in likeliest)
        return None
    return yes / (yes + no)


class _Completions:
    def __init__(self, inner, timeout_error: "type[BaseException]"):
        self._inner = inner
        self._timeout_error = timeout_error

    def create(self, *args, **kwargs):
        try:
            return self._inner.create(*args, **kwargs)
        except self._timeout_error as e:
            # The SDK's timeout error subclasses its connection error; only the
            # timeout becomes CompletionTimeout, everything else passes through
            # for the request functions' generic failure handling.
            raise CompletionTimeout(str(e)) from e


class _Chat:
    def __init__(self, inner, timeout_error: "type[BaseException]"):
        self.completions = _Completions(inner.completions, timeout_error)


class OpenAIClient:
    """An `openai.OpenAI` client whose chat completions raise CompletionTimeout
    on a timeout. Every other attribute (`timeout`, `max_retries`, `models`,
    ...) is the wrapped client's own."""

    def __init__(self, inner: Any, timeout_error: "type[BaseException]"):
        self._inner = inner
        self.chat = _Chat(inner.chat, timeout_error)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def build_openai_client(base_url: str, api_key: str, timeout: float) -> OpenAIClient:
    """
    Builds the bounded client: `timeout` seconds per request and no retries.
    Raises ValueError for a timeout that is not finite and > 0 (an unbounded
    client is the failure this module exists to prevent); callers resolve a
    usable one first. `openai` is imported here, not at the top of the module.
    """
    if not is_bounded_timeout(timeout):
        raise ValueError(f"timeout must be finite and > 0, got {timeout!r}")
    from openai import APITimeoutError, OpenAI

    return OpenAIClient(
        OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=CLIENT_MAX_RETRIES),
        APITimeoutError,
    )
