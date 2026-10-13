#!/usr/bin/env python3
"""
compress_file, compress_command_output, and fetch_url have been validated
against real sources (real files/commands, and a real Wikipedia article via
a real local model) across multiple rounds of testing -- see
libs/local_compress_lib.py's docstrings for the specific failures found and
fixed along the way. compress_text and list_local_models are simpler and
less exercised. Results will still vary by which local model you have
loaded -- testing surfaced real, model-dependent quality differences (see
classify_relevant's docstring), so don't assume another model behaves
identically to whatever was used during this repo's own testing.

MCP server that compresses large, mechanical text (build/test logs, big
diffs, verbose command output) using a LOCAL LM Studio model *before* it
reaches Claude's context. The point: this costs zero Claude API tokens --
only local compute -- and Claude only pays to read the (hopefully much
smaller) compressed result.

Project minimum: Python 3.11 (policy floor; the dependency chain supports >=3.10).

Setup:
    1. Install LM Studio (https://lmstudio.ai), load a small model, start
       its local server (Server tab -> Start). Defaults to
       http://localhost:1234/v1.
    2. pip install "mcp[cli]" openai requests trafilatura --break-system-packages
       (or just pip install -r requirements.txt from the repo root -- requests
       and trafilatura are needed specifically for fetch_url; omitting them
       causes this server to crash on startup with a ModuleNotFoundError,
       which Claude Code surfaces as the much less obvious "MCP error
       -32000: Connection closed")
    3. Register in .mcp.json:
       {
         "mcpServers": {
           "local-compress": {
             "command": "python",
             "args": ["/absolute/path/to/tools/compress_mcp_server.py"],
             "env": {
               "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:1234/v1",
               "CLAUDE_RUNWAY_LMSTUDIO_MODEL": "<exact model id loaded in LM Studio>"
             }
       NOTE: these must ALSO be exported at the OS/shell level, with matching
       values, for hooks/*.py to see them -- a hook entry in .claude/settings.json
       has no `env` field of its own. Setting only one side is the single most
       common misconfiguration here; see the README's "Environment variables"
       section. The pre-rename names (LMSTUDIO_BASE_URL / LMSTUDIO_MODEL /
       HOOK_COMPRESS_THRESHOLD_CHARS) are no longer read.
           }
         }
       }

DESIGN DECISIONS:
  1. fetch_url, for the same reason compress_command_output exists
     instead of "run Bash then compress_text the output" -- Claude Code's
     built-in WebFetch tool already runs its own extraction/summarization,
     but that step happens on ANTHROPIC's infrastructure, not locally, with
     no way to inspect or steer how it summarizes (confirmed empirically:
     a 2MB fetched page came back as a ~1300 char WebFetch result, already
     processed before this project's tools ever saw it). fetch_url instead
     does the fetch AND the review/summarization step itself, entirely
     server-side and entirely on the local LM Studio model -- Claude only
     ever sees the final compressed result, and the local model's read of
     the page is steerable via `focus` the same way compress_file/
     compress_command_output already are.
"""

import codecs
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import requests
import trafilatura
from mcp.server.mcpserver import MCPServer, Context

from lmstudio_client import build_client
from local_compress_lib import (
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_FOCUS,
    DEFAULT_WEB_FOCUS,  # defined in the lib so _looks_non_selective can see it
    compress as _compress_impl,
    fetch_loaded_models,
    redact_credentials,
)

# savings_ledger is optional at import time (e.g. an older checkout of just
# this file without libs/savings_ledger.py) -- fails open to "tracking off"
# rather than crashing server startup over an opt-in feature. The explicit
# `savings_ledger = None` in the except branch (issue #297, adopting Pyright)
# keeps the name always bound -- every function below that touches
# savings_ledger checks _SAVINGS_LEDGER_AVAILABLE/TRACK_SAVINGS first and
# then asserts it's not None, which is always true at runtime (that's exactly
# what _SAVINGS_LEDGER_AVAILABLE records) but lets Pyright narrow the type
# instead of reporting a possibly-unbound/Optional-access error.
try:
    import savings_ledger
    _SAVINGS_LEDGER_AVAILABLE = True
except ImportError:
    savings_ledger = None  # type: ignore[assignment]
    _SAVINGS_LEDGER_AVAILABLE = False

if _SAVINGS_LEDGER_AVAILABLE:
    assert savings_ledger is not None
    TRACK_SAVINGS = savings_ledger.tracking_enabled()
else:
    TRACK_SAVINGS = False

# metrics_lib is likewise optional at import time (issue #208) -- same
# fail-open reasoning as savings_ledger above: an older checkout of just this
# file, without libs/metrics_lib.py, must not crash server startup over a
# brand-new, still domain-empty store. Same `= None` reasoning as above.
try:
    import metrics_lib
    _METRICS_LIB_AVAILABLE = True
except ImportError:
    metrics_lib = None  # type: ignore[assignment]
    _METRICS_LIB_AVAILABLE = False

# Issue #39: cap on how much of a fetch_url response body is ever read into
# memory. max_total_chars is a post-extraction cap and timeout_seconds is a
# per-read/connect timeout, not a total-duration cap, so neither stops a URL
# that resolves to a multi-GB file (a redirect to a video/zip/dataset, a
# poorly-bounded API response, or simply the wrong link) from downloading in
# full first. 10 MB is comfortably above any real HTML/article page
# (fetch_url's actual target) and comfortably below "this is clearly not an
# article anymore."
MAX_FETCH_URL_BYTES = 10 * 1024 * 1024

mcp = MCPServer("local-compress")


def _append_savings_footer(outer_text: str, inner_result: str, tool: str, raw_text: str, credited: bool, source: str = "") -> str:
    """
    Appends a machine-readable savings footer to a credited tool's return
    value, ONLY when CLAUDE_RUNWAY_TRACK_SAVINGS is on and actual compression
    happened. Detection reuses compress()'s own self-documenting "[compressed
    X -> Y chars]" prefix (see local_compress_lib.compress's docstring)
    rather than re-deriving "did compression actually happen" separately.

    inner_result is what compress() returned (used only for that detection);
    outer_text is the full string this tool actually returns to Claude
    (which may wrap inner_result with an exit-code/fetch-size line) -- the
    footer is appended there, and out_chars is measured on it, since that's
    what Claude will actually see.

    hooks/compress_output.py is the one that STRIPS this footer before
    Claude ever reads it and logs the numbers to the savings ledger -- this
    MCP server can't write the ledger itself because (unlike a hook) it has
    no access to Claude Code's session_id. When tracking is off, this is a
    pure no-op, so tool output stays byte-identical to before this feature
    existed.

    `source` is redacted (not just truncated) before being embedded here
    (PR #135 review, round 3): it's the raw caller-supplied value --
    compress_command_output passes the actual shell command, which can
    itself carry a credential (e.g. `curl -H 'Authorization: Bearer ghp_...'`)
    even when the compressed BODY the command produced is properly
    redacted. This footer gets persisted into the savings ledger by
    hooks/compress_output.py, so an unredacted `source` would turn a
    transient secret in a command line into a durable one in that database
    -- same class of problem redact_and_disclose() already exists to
    prevent for compressed content itself. Redacted over the FULL source
    before truncating to 200 chars, not the other way around, so a
    credential straddling that 200-char cutoff is still caught in full
    before the cutoff can split it.
    """
    if not TRACK_SAVINGS or not inner_result.startswith("[compressed"):
        return outer_text
    redacted_source, _ = redact_credentials(source) if source else ("", 0)
    footer = json.dumps({
        "tool": tool,
        # Exact character counts (issues #437, #438). The chars / 3.5 token
        # estimates this footer used to carry are gone: a token count depends on
        # the model's vocabulary, which is not public.
        "raw_chars": len(raw_text),
        "out_chars": len(outer_text),
        "credited": credited,
        "source": redacted_source[:200],
    })
    return f"{outer_text}\n<!--CLAUDE_RUNWAY_SAVINGS:{footer}-->"


@mcp.tool()
def list_local_models(base_url: Optional[str] = None) -> str:
    """
    List the model ID(s) currently LOADED (not merely downloaded) on the
    local LM Studio server. Use this to find the exact model string for
    compress_text's model param, or to check LM Studio is actually reachable
    before relying on it.

    Issue #296: this used to report every model /v1/models returned, which
    is every DOWNLOADED model (LM Studio's default is JIT loading, so a
    downloaded model isn't necessarily loaded) -- misleading when the output
    is used directly as a model= value that must name something actually
    loaded. Now uses local_compress_lib.fetch_loaded_models, which prefers
    LM Studio's native /api/v0/models endpoint (true per-model load state)
    and only falls back to /v1/models -- unchanged, same old limitation --
    when /api/v0 isn't available; the reply says explicitly when that
    fallback happened, since load state can't be confirmed there.
    """
    try:
        ids, source = fetch_loaded_models(base_url)
    except Exception as e:
        return f"Could not reach LM Studio at {base_url or DEFAULT_BASE_URL}: {e}"
    if not ids:
        return "LM Studio is reachable but no model is loaded."
    if source == "v0":
        return "Loaded chat models: " + ", ".join(ids)
    return (
        "Models available (load state unknown -- LM Studio's /api/v0 endpoint wasn't "
        "reachable, so this can't confirm which are actually loaded): " + ", ".join(ids)
    )


async def _compress(
    text: str,
    focus: str,
    skip_if_under_chars: int,
    chunk_chars: int,
    max_total_chars: int,
    model: Optional[str],
    base_url: Optional[str],
    ctx: Optional[Context],
    max_chars: Optional[int] = None,
    preserve_identifiers: bool = False,
    preserve_sections: bool = False,
) -> str:
    """
    Thin wrapper so the rest of this file's tool signatures stay unchanged.
    Truncation for positional asks ("the lead section," "the introduction")
    now happens inside local_compress_lib.compress() itself -- both an
    explicit max_chars and auto-detection of positional-looking focus text
    are handled there (see its docstring for the full history: a position-
    aware classifier was tried first, but a real test against a real model
    still returned a Techniques-section digest mislabeled as "the lead
    section," so a deterministic keyword-based shortcut was added as the
    reliable path, with the classifier kept as a secondary aid for
    positional phrasings the keyword list doesn't catch).
    """
    # This server owns the `openai` client (libs/lmstudio_client.py); the
    # compression library only receives it. Built per call because base_url
    # is a per-call tool parameter.
    return await _compress_impl(
        text, focus, skip_if_under_chars, chunk_chars, max_total_chars, max_chars,
        model, base_url, preserve_identifiers=preserve_identifiers,
        preserve_sections=preserve_sections, ctx=ctx, client=build_client(base_url),
    )


def _decode_file_bytes(data: bytes) -> str:
    """Decode a file's raw bytes without silently corrupting them (issue #267).

    The old read, `read_text(encoding="utf-8", errors="ignore")`, pinned the
    encoding but still dropped every byte it couldn't decode, so the model
    summarized subtly corrupted input with nothing to notice: a cp1252/latin-1
    file lost every accented character, and a UTF-16LE file (Windows
    PowerShell 5.1's `>` default) arrived NUL-interleaved because NUL is
    valid UTF-8. Honor a BOM first (stdlib codecs strip it, so it never
    reaches the model), then decode as UTF-8 with errors="replace" so any
    remaining undecodable byte shows up as a visible U+FFFD marker instead of
    vanishing.
    """
    # UTF-32 BOMs must be tested before UTF-16's: UTF-32LE's BOM
    # (FF FE 00 00) begins with UTF-16LE's (FF FE).
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return data.decode("utf-32", errors="replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


@mcp.tool()
async def compress_file(
    file_path: str,
    focus: str = DEFAULT_FOCUS,
    skip_if_under_chars: int = 2000,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_total_chars: int = 2_000_000,
    max_chars: Optional[int] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    # MCPServer injects context by matching the `Context` annotation on a
    # parameter name. `ctx: Context = None` keeps the framework-visible type
    # annotation while suppressing a "None isn't a valid Context default"
    # complaint per-line, avoiding the need to widen to Optional[Context].
    # Confirmed still needed under Pyright (issue #297 -- replaced mypy):
    # removing the ignore surfaces Pyright's own equivalent
    # reportArgumentType error here, just phrased differently than mypy's.
    # Same reasoning at every other `ctx` parameter below and in
    # ingest_mcp_server.py/memory_bank_mcp_server.py.
    ctx: Context = None,  # type: ignore[assignment]
) -> str:
    """
    PREFER THIS OVER Read for large, mostly-mechanical files (logs, verbose
    dumps, big diffs saved to disk) where you don't need exact original
    text. This server reads the file itself -- the raw contents never pass
    through your own context, only the compressed summary does, which is
    what actually makes this save tokens (compress_text does not, since you
    have to already hold the text to call it).

    Do NOT use this for source code you intend to edit -- read that
    directly with Read so you have exact, unmodified content. A summary is
    for deciding where to look next, not for editing from.

    If the summary reveals something needing precise follow-up (an exact
    line number, exact error text to grep for), read that specific file/
    section directly afterward rather than treating the summary as
    authoritative for exact text.

    POSITIONAL asks ("the first N lines," "the beginning of the file") are
    handled automatically -- the underlying classifier knows each chunk's
    position in the file and weighs that alongside content. max_chars is
    available as an explicit override if you want a hard guarantee of
    "only look at the first N chars" regardless of what the classifier
    decides -- see the _compress() docstring in this file for why that's a
    fallback now, not the primary mechanism.
    """
    path = Path(file_path)
    if not path.is_file():
        return f"Error: {path} is not a file."
    text = _decode_file_bytes(path.read_bytes())
    result = await _compress(text, focus, skip_if_under_chars, chunk_chars, max_total_chars, model, base_url, ctx, max_chars)
    return _append_savings_footer(result, result, tool="compress_file", raw_text=text, credited=True, source=file_path)


@mcp.tool()
async def compress_command_output(
    command: str,
    cwd: Optional[str] = None,
    focus: str = DEFAULT_FOCUS,
    skip_if_under_chars: int = 2000,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_total_chars: int = 2_000_000,
    max_chars: Optional[int] = None,
    timeout_seconds: int = 300,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    preserve_identifiers: bool = True,
    preserve_sections: bool = False,
    ctx: Context = None,  # type: ignore[assignment]  # see compress_file's ctx comment above
) -> str:
    """
    PREFER THIS OVER Bash for commands expected to produce large, mostly-
    mechanical output (test suites, builds, verbose linters) where you want
    the gist/failures, not the full log. This server runs the command and
    captures output itself -- the raw stdout/stderr never pass through your
    own context, only the compressed summary does, which is what actually
    makes this save tokens.

    Do NOT use this for commands whose exact output you need verbatim (e.g.
    a command whose output you're about to parse programmatically, or one
    producing a diff you intend to apply). If the summary indicates
    something needs precise follow-up, re-run a narrower command or read the
    specific file directly rather than treating the summary as authoritative
    for exact text.

    POSITIONAL asks ("just the first part of the output," "the header
    before it starts logging") are handled automatically by the underlying
    classifier, which weighs each chunk's position alongside its content.
    max_chars is available as an explicit override -- see compress_file's
    docstring.

    Returns the exit code alongside the compressed output, since exit code
    alone often tells you whether you even need to read further.

    preserve_identifiers defaults to True HERE (unlike compress_file/
    compress_text, which default it off) -- issue #46: shell command output
    (curl responses, printenv, config dumps, an API token embedded in JSON)
    is the single surface most likely to contain a raw secret, and
    local_compress_lib.compress()'s credential redaction runs whenever
    EITHER preserve_identifiers OR preserve_sections is on -- setting
    preserve_identifiers=False does NOT disable redaction by itself if
    preserve_sections=True is also passed. Set preserve_identifiers to
    False only for a pure gist-style ask over output you're confident
    carries no sensitive values, where dragging along every identifier
    would work against the summary's readability -- and leave
    preserve_sections off too if you want redaction off entirely.
    """
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout_seconds}s."
    except Exception as e:
        return f"Error running command: {e}"

    combined = proc.stdout + (f"\n--- stderr ---\n{proc.stderr}" if proc.stderr else "")
    result = await _compress(
        combined, focus, skip_if_under_chars, chunk_chars, max_total_chars, model, base_url, ctx, max_chars,
        preserve_identifiers=preserve_identifiers, preserve_sections=preserve_sections,
    )
    wrapped = f"[exit code {proc.returncode}]\n{result}"
    return _append_savings_footer(wrapped, result, tool="compress_command_output", raw_text=combined, credited=True, source=command)


@mcp.tool()
async def fetch_url(
    url: str,
    focus: str = DEFAULT_WEB_FOCUS,
    skip_if_under_chars: int = 2000,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_total_chars: int = 2_000_000,
    max_chars: Optional[int] = None,
    timeout_seconds: int = 30,
    max_response_bytes: int = MAX_FETCH_URL_BYTES,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    ctx: Context = None,  # type: ignore[assignment]  # see compress_file's ctx comment above
) -> str:
    """
    PREFER THIS OVER WebFetch when you want the page reviewed and summarized
    by your LOCAL LM Studio model instead of by Anthropic's own WebFetch
    extraction step. This server fetches the page and extracts the main
    content itself -- the raw HTML never passes through your own context,
    only the compressed summary does. Use `focus` to steer exactly what to
    extract (default targets general fact/number/name preservation).

    Unlike WebFetch, which already runs its own summarization but on
    Anthropic's infrastructure with no way to inspect or customize it, this
    tool's review step is entirely local, so you can point it at what you
    actually need from the page.

    POSITIONAL asks (the lead/intro section, the first paragraph, an
    abstract) are handled automatically through `focus` -- e.g. focus=
    "summarize the lead section" -- the underlying classifier is told each
    chunk's position in the document and weighs that alongside content, so
    it can tell "chunk 1, plausibly the introduction" apart from "chunk 45,
    topically similar but not the introduction." Confirmed against a real
    Wikipedia article: a scripted test correctly kept only the genuinely-
    first chunk for a "lead section" focus, where an earlier content-only
    version of the classifier had pulled in a 2024 news item from deep in
    the article body just because it read as topically similar. Topic-based
    asks (a specific fact, a named concept) also work well -- verified
    against the same real page: a narrow factual focus correctly extracted
    just the matching detail, and a focus for a topic absent from the page
    correctly returned an explicit not-found error instead of fabricating
    something. If a particular local model doesn't weigh position well
    despite this, max_chars is available as an explicit override to force
    "only the first N chars" regardless of the classifier's judgment.

    Do NOT use this for pages requiring authentication, JavaScript
    rendering, or session/cookie handling -- this issues a plain
    unauthenticated GET and extracts from the resulting static HTML. Use
    WebFetch instead for those cases.

    Do NOT use this when you need exact page text verbatim (e.g. quoting a
    passage precisely) -- a summary is for gist/lookup, not for exact
    quotation. If precise text is needed, re-fetch with WebFetch instead.

    max_response_bytes caps how much of the response body this will ever
    materialize into memory (default 10 MB, see MAX_FETCH_URL_BYTES) -- a
    response that declares a larger Content-Length is refused before any
    body is read, and one with no (or an understated) Content-Length is
    aborted mid-stream the moment the running byte count crosses the cap.
    Issue #39: without this, a URL that happens to resolve to a very large
    resource (a redirect to a video/zip/dataset, a poorly-bounded API
    response) would fully download into memory before anything checked its
    size, since timeout_seconds is a per-read/connect timeout, not a
    total-duration cap.
    """
    try:
        # PR #124 review (Copilot): a bare `resp = requests.get(...)` left
        # the connection unclosed on any exception raised by
        # raise_for_status()/iter_content() below (a normal read timeout or
        # a broken chunked response, not something exotic) -- those
        # exceptions propagated straight to the `except Exception` below
        # without ever calling resp.close(), leaking a socket in this
        # long-running server on every such failure. requests.Response
        # supports the context-manager protocol specifically to release the
        # underlying connection on ANY exit path (normal return, `return`
        # inside the block, or an exception) -- confirmed live: a fake
        # response whose iter_content() raised mid-stream left .close()
        # never called before this fix, called reliably after.
        with requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": "Mozilla/5.0 (compatible; local-compress-fetch/1.0)"},
            stream=True,
        ) as resp:
            resp.raise_for_status()

            # Fast path: if the server tells us up front it's too big, refuse
            # before reading a single byte of the body. Not load-bearing by
            # itself -- a malformed/absent/lying Content-Length falls through
            # to the incremental check below, which is the one that actually
            # enforces the cap in every case.
            content_length = resp.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_bytes = int(content_length)
                except ValueError:
                    declared_bytes = None
                if declared_bytes is not None and declared_bytes > max_response_bytes:
                    return (
                        f"Error fetching {url}: response declares {declared_bytes} "
                        f"bytes via Content-Length, over the {max_response_bytes}-byte "
                        "cap -- refusing to download. Use WebFetch instead if you "
                        "genuinely need this content."
                    )

            # Read incrementally rather than resp.content/resp.text, which
            # would materialize the entire body regardless of size before we
            # ever get a chance to check it. This is the check that actually
            # matters when Content-Length is missing or understated (the
            # common case for a streamed/chunked response) -- confirmed
            # live: a server is free to omit or lie about this header
            # entirely.
            chunks = []
            total_bytes = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > max_response_bytes:
                    return (
                        f"Error fetching {url}: response body exceeded the "
                        f"{max_response_bytes}-byte cap while streaming (no reliable "
                        "Content-Length was available to catch this earlier) -- "
                        "refusing to download further. Use WebFetch instead if you "
                        "genuinely need this content."
                    )
                chunks.append(chunk)
            raw_bytes = b"".join(chunks)

            encoding = resp.encoding
            # Issue #266: requests derives resp.encoding from the header
            # alone, and for any text/* Content-Type WITHOUT a charset
            # parameter it reports the RFC 2616 legacy default "ISO-8859-1"
            # (confirmed live) rather than None. That is truthy, so the
            # detection below was skipped for the most common real-world
            # case -- a UTF-8 page sent as bare `text/html` with the charset
            # only in <meta> -- and Latin-1 decoding (which never raises)
            # silently turned every multi-byte sequence into mojibake.
            # Invariant: the header's ISO-8859-1 is only trusted when the
            # server EXPLICITLY declared a charset; otherwise it is requests'
            # placeholder and is treated exactly like "no encoding known".
            content_type = resp.headers.get("Content-Type", "") or ""
            implicit_latin1 = (
                bool(encoding)
                and encoding.lower() == "iso-8859-1"
                and "charset" not in content_type.lower()
            )
            if not encoding or implicit_latin1:
                # PR #124 review (Copilot): a response with no Content-Type
                # charset at all (resp.encoding is None) used to fall back,
                # via resp.text, to apparent_encoding -- chardet/
                # charset_normalizer detection over the full body -- not a
                # blind UTF-8 assumption. Reproduced directly: a headerless
                # Windows-1252-encoded page containing a curly single-quote
                # (a realistic legacy-page byte, not an edge case) decoded
                # to a bare U+FFFD replacement character under straight
                # UTF-8, silently corrupting text handed to extraction.
                # apparent_encoding is a thin wrapper around
                # chardet.detect(self.content) -- populate resp's own
                # content cache with the bytes already downloaded (bounded
                # by max_response_bytes) so detection runs over those
                # instead of trying to re-read an already-exhausted stream.
                resp._content = raw_bytes
                encoding = resp.apparent_encoding
    except Exception as e:
        return f"Error fetching {url}: {e}"

    # PR #124 review (Copilot): resp.text's OWN implementation catches
    # (LookupError, TypeError) around this exact decode and falls back to
    # UTF-8 -- a server is free to send a Content-Type charset that isn't a
    # real Python codec name (confirmed live: `Content-Type: ...;
    # charset=invalid-bogus-charset` makes requests set resp.encoding to
    # that literal string, and decoding with it raises LookupError). Losing
    # that fallback when switching off resp.text would have turned a
    # malformed-but-realistic response into an unhandled crash instead of
    # the graceful error every other failure path here returns. Also covers
    # apparent_encoding itself returning None (detection genuinely
    # inconclusive), which decode() would otherwise reject with TypeError.
    try:
        html_text = raw_bytes.decode(encoding or "utf-8", errors="replace")
    except (LookupError, TypeError):
        html_text = raw_bytes.decode("utf-8", errors="replace")

    text = trafilatura.extract(html_text, url=url) or ""
    if not text.strip():
        # trafilatura found no "article-like" main content (common on
        # non-article pages) -- fall back to a crude tag strip rather than
        # returning nothing.
        text = re.sub(r"<[^>]+>", " ", html_text)
        text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return f"Error: no extractable text content found at {url}."

    result = await _compress(text, focus, skip_if_under_chars, chunk_chars, max_total_chars, model, base_url, ctx, max_chars)
    wrapped = f"[fetched {url}, {len(raw_bytes)} raw bytes]\n{result}"
    # credited=False: fetch_url's honest counterfactual is WebFetch's OWN
    # server-side summary (already compressed, per this repo's own testing),
    # not the raw page -- so we log the event for visibility but never count
    # it toward the headline savings number. See EVALUATION.md Track D.
    return _append_savings_footer(wrapped, result, tool="fetch_url", raw_text=text, credited=False, source=url)


@mcp.tool()
async def compress_text(
    text: str,
    focus: str = DEFAULT_FOCUS,
    skip_if_under_chars: int = 2000,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_total_chars: int = 2_000_000,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    preserve_identifiers: bool = False,
    preserve_sections: bool = False,
    ctx: Context = None,  # type: ignore[assignment]  # see compress_file's ctx comment above
) -> str:
    """
    Compress text you already hold (e.g. your own long draft) via a local
    model. IMPORTANT: this does NOT save Claude tokens if you had to read
    something into context just to call this -- by that point you've
    already paid the input cost to read it, and pay output tokens again to
    pass it as this argument. For files or command output, use compress_file
    or compress_command_output instead, which read the source themselves so
    the raw content never has to pass through you at all.

    Set preserve_identifiers=true when the result is an artifact someone will
    later grep -- a handoff summary, a set of notes to resume from. It checks
    after compressing that every identifier-shaped token in the source (file
    paths, colon-delimited config keys, dotted namespaces, CamelCase names,
    CLI flags, long numeric ids) still appears, and re-appends verbatim any
    the model dropped. A real 9,876-char handoff summary lost the Options
    segment from a colon-delimited config key and three entire sections despite a focus
    that explicitly demanded verbatim identifiers -- at that size the text is
    a single chunk, so there is no reduce step to catch it. Leave it off for
    gist-style asks, where dragging along every identifier defeats the point.

    Set preserve_sections=true for '## '-headed structured input (a handoff
    summary, a filled-in template). It compresses one section at a time,
    passes exactness-critical sections through untouched, and re-emits every
    heading from the source, so a section cannot silently disappear. This
    addresses the other half of the same failure: the same real handoff lost
    three ENTIRE sections -- all of its unresolved questions, all of its
    environment notes, and a user instruction -- while still reading as
    complete. Safe to leave on for unstructured text, which falls back to
    ordinary whole-text compression. The qdrant server's compact_store sets
    both flags itself when it compresses a conversation summary, so
    /my-compact no longer calls this tool.
    """
    return await _compress(
        text, focus, skip_if_under_chars, chunk_chars, max_total_chars,
        model, base_url, ctx, preserve_identifiers=preserve_identifiers,
        preserve_sections=preserve_sections,
    )


def _require_savings_tracking() -> Optional[str]:
    if not _SAVINGS_LEDGER_AVAILABLE:
        return "Error: libs/savings_ledger.py is not available in this install -- the savings tracker requires it."
    assert savings_ledger is not None  # _SAVINGS_LEDGER_AVAILABLE implies this
    if not savings_ledger.tracking_enabled():
        return (
            "Savings tracking is off (CLAUDE_RUNWAY_TRACK_SAVINGS is not set). "
            "Set CLAUDE_RUNWAY_TRACK_SAVINGS=1 in this server's env (in .mcp.json) and export it in the shell "
            "environment that launches `claude` too (hook entries in .claude/settings.json have no env field "
            "of their own, so that's the only way compress_output.py sees it)."
        )
    return None


@mcp.tool()
def savings_summary(project: Optional[str] = None, format: str = "text") -> str:
    """
    SIMPLE view of the opt-in savings tracker: this session's characters
    before and after local compression and the percent reduction, plus this
    project's all-time totals. Called by /my-savings with no argument.

    Character counts recorded from now on are exact; older all-time history may
    include approximate figures (rows from before schema v6, or stale token-only
    footers, were reconstructed as tokens x 3.5) or rows with no size at all,
    which are left out and counted as unsized. What they would have cost in tokens is not
    estimated (a token count depends on the model's vocabulary, which is not
    public). The counterfactual itself -- what the raw content would have cost
    vs. the compressed result actually used -- is still observed online, not a
    controlled A/B measurement; see EVALUATION.md's Track D for that distinction.
    Only local-compression events are credited; qdrant-find activity is
    intentionally never counted here (its counterfactual isn't observable).

    project defaults to the current working directory's folder name.

    format controls the output format:
      "text" (default) — human-readable text, same as before.
      "json"           — JSON object with five keys: "session", "project_summary",
                         and "tool_breakdown"/"last_n_sessions"/"all_projects" as
                         empty arrays (those are summary-only fields not populated
                         by this tool; use savings_detail for the full set).
      "csv"            — CSV with a header row and one data row for the session.
    """
    err = _require_savings_tracking()
    if err:
        return err
    assert savings_ledger is not None  # _require_savings_tracking() returning None implies this
    # Validate before any DB access so a bad argument produces a clear input
    # error and never reaches the generic except handler below, which would
    # misleadingly report the issue as database corruption.
    if format not in ("text", "json", "csv"):
        return f"Error: unrecognised format {format!r}. Valid values: 'text', 'json', 'csv'."
    try:
        project = project or savings_ledger.project_name_from_cwd(os.getcwd())
        # project-filtered: a live session belonging to a DIFFERENT project
        # must never be picked here just because its JSONL is the most
        # recently touched file in the shared sessions dir -- see
        # current_session_id's docstring and issue #35. Falling through to
        # the zero-value aggregate below (same as the "no live session at
        # all" case) is what keeps the project label below always accurate.
        session_id = savings_ledger.current_session_id(project=project)
        # Annotated `dict` (not inferred from the ternary's two branches,
        # whose value types differ -- get_live_session_aggregate() returns a
        # bare `dict`, the literal fallback is `dict[str, int]`) so the
        # `session_agg["project"] = project` str assignment below type-checks.
        session_agg: dict = (
            savings_ledger.get_live_session_aggregate(session_id) if session_id
            else {"credited_saved_chars": 0, "raw_chars_sum": 0, "out_chars_sum": 0, "event_count": 0, "fetch_url_count": 0}
        )
        session_agg["project"] = project
        project_summary = savings_ledger.query_project_summary(project)
        if format == "json":
            return savings_ledger.format_json(session_agg, project_summary, [], [], [])
        if format == "csv":
            return savings_ledger.format_csv(session_agg, project_summary, [], [], [], table="session")
        return savings_ledger.format_simple_view(session_agg, project_summary)
    except Exception as e:
        # This is an opt-in, best-effort feature -- a corrupted/locked/missing
        # savings DB must not fail the whole tool call (or worse, the server
        # process). Return a readable error instead of letting it raise.
        return f"Error: could not read savings data ({e}). The savings database may be corrupted or inaccessible; this doesn't affect anything else."


@mcp.tool()
def savings_detail(project: Optional[str] = None, format: str = "text", csv_table: str = "session") -> str:
    """
    DETAIL view of the opt-in savings tracker: this session's per-tool
    breakdown, the per-model split of the most recent finished session (exact
    characters before and after, plus a token figure labelled as derived), a sparkline of the last 10 sessions for this project, and a
    cross-project comparison of characters before and after compression. Called by
    `/my-savings detail`. See savings_summary's docstring for the same
    counterfactual-estimate caveat.

    project defaults to the current working directory's folder name.

    format controls the output format:
      "text" (default) — human-readable text, same as before.
      "json"           — JSON object with the data blocks (including the
                         per-model split when there is one, but not the
                         text view's main-vs-subagent breakdown).
      "csv"            — CSV whose content depends on csv_table (see below).

    csv_table selects which table to export when format="csv":
      "session"  (default) — one-row session summary.
      "tools"              — per-tool breakdown rows.
      "projects"           — cross-project all-time rows.
    """
    err = _require_savings_tracking()
    if err:
        return err
    assert savings_ledger is not None  # _require_savings_tracking() returning None implies this
    # Validate both enum parameters before any DB access so bad arguments
    # produce clear input errors rather than silently returning text or
    # hitting the generic except handler below (which would misleadingly
    # report the issue as database corruption).
    if format not in ("text", "json", "csv"):
        return f"Error: unrecognised format {format!r}. Valid values: 'text', 'json', 'csv'."
    if format == "csv" and csv_table not in ("session", "tools", "projects"):
        return f"Error: unrecognised csv_table {csv_table!r}. Valid values: 'session', 'tools', 'projects'."
    try:
        project = project or savings_ledger.project_name_from_cwd(os.getcwd())
        # See savings_summary's identical comment -- project-filtered lookup,
        # issue #35.
        session_id = savings_ledger.current_session_id(project=project)
        # Annotated `dict` here too (see savings_summary's identical comment)
        # so the "project" key assignment below type-checks either way.
        session_agg: dict
        if session_id:
            session_agg = savings_ledger.get_live_session_aggregate(session_id)
            # Sorted by saved_chars descending -- by_tool.items() is otherwise in
            # dict insertion order (first-tool-seen order), which would make the
            # ordering inconsistent with query_session_tool_breakdown's explicit
            # "ORDER BY saved_chars DESC" for a finalized (post-SessionEnd) session.
            tool_breakdown = sorted(
                (
                    {
                        "tool": t,
                        "event_count": a["event_count"],
                        "saved_chars": a["saved_chars"],
                        # raw/out/credited_event_count feed the reduction % and
                        # credited-vs-uncredited detection in format_detail_view --
                        # all three must be forwarded here or the live (pre-SessionEnd)
                        # path silently suppresses the metric.
                        "raw_chars_sum": a.get("raw_chars_sum", 0),
                        "out_chars_sum": a.get("out_chars_sum", 0),
                        "credited_event_count": a.get("credited_event_count", 0),
                    }
                    for t, a in session_agg.get("by_tool", {}).items()
                ),
                key=lambda t: t["saved_chars"],
                reverse=True,
            )
        else:
            session_agg = {"credited_saved_chars": 0, "raw_chars_sum": 0, "out_chars_sum": 0, "event_count": 0, "fetch_url_count": 0}
            tool_breakdown = []
        session_agg["project"] = project
        project_summary = savings_ledger.query_project_summary(project)
        last_n = savings_ledger.query_last_n_sessions(project, n=10)
        all_projects = savings_ledger.query_all_projects()
        # The session in progress has no transcript to join yet, so the per-model
        # split comes from the most recent finished session (issue #440).
        model_breakdown = savings_ledger.query_recent_model_breakdown(project)
        if format == "json":
            return savings_ledger.format_json(session_agg, project_summary, tool_breakdown, last_n, all_projects, model_breakdown)
        if format == "csv":
            return savings_ledger.format_csv(session_agg, project_summary, tool_breakdown, last_n, all_projects, table=csv_table)
        return savings_ledger.format_detail_view(session_agg, project_summary, tool_breakdown, last_n, all_projects, model_breakdown)
    except Exception as e:
        # Same reasoning as savings_summary's try/except -- best-effort feature,
        # must not fail the tool call over a DB problem.
        return f"Error: could not read savings data ({e}). The savings database may be corrupted or inaccessible; this doesn't affect anything else."


@mcp.tool()
def savings_trend(project: Optional[str] = None, bucket: str = "week", n: int = 12) -> str:
    """
    TREND view of the opt-in savings tracker: characters before and after
    compression, aggregated by day or week rather than individual sessions, so patterns are visible even
    once a project has accumulated enough history that per-session rows are
    too noisy to read. Called by `/my-savings trend`.

    See savings_summary's docstring for the same counterfactual-estimate
    caveat that applies here.

    project defaults to the current working directory's folder name.

    bucket: "week" (default) — groups sessions by ISO week (YYYY-Www).
            "day"            — groups sessions by calendar day (YYYY-MM-DD).

    n: maximum number of buckets to show (default 12, i.e. ~3 months of weeks
       or ~2 weeks of days). Older buckets are dropped when the project has
       more history than this.
    """
    err = _require_savings_tracking()
    if err:
        return err
    assert savings_ledger is not None  # _require_savings_tracking() returning None implies this
    if bucket not in ("day", "week"):
        return f"Error: unrecognised bucket {bucket!r}. Valid values: 'day', 'week'."
    if not isinstance(n, int) or n < 1:
        return "Error: n must be a positive integer."
    try:
        project = project or savings_ledger.project_name_from_cwd(os.getcwd())
        trend_rows = savings_ledger.query_trend(project, bucket=bucket, n=n)
        return savings_ledger.format_trend_view(trend_rows, project, bucket=bucket)
    except Exception as e:
        # Same reasoning as savings_summary's try/except -- best-effort feature,
        # must not fail the tool call over a DB problem.
        return f"Error: could not read savings data ({e}). The savings database may be corrupted or inaccessible; this doesn't affect anything else."


@mcp.tool()
def get_metrics(metric_id: str, view: str = "summary", bucket: str = "week", n: int = 12, format: str = "text", session_id: Optional[str] = None) -> str:
    """
    Reads the shared cross-tool metrics store (issue #208): ONE generic
    SQLite table any domain in this toolkit can write into via
    `libs/metrics_lib.py`'s `MetricsStore` class, aggregated here into
    whichever `view` is requested. Deliberately the ONLY new tool this
    ticket adds (see `libs/metrics_lib.py`'s module docstring for why a
    single generic dispatch tool matters here specifically) -- a future
    domain writing a new `metric_id` never needs a new tool, only a new
    value passed to this one.

    As of issue #210, my-gh-autowork's per-ticket outcome logging writes
    into this store under metric_id="autowork". Any metric_id with no
    recorded events returns an honest "no events recorded yet" result.

    metric_id: the domain to query (e.g. "autowork", "memory-bank") --
      whatever string a future writer used when calling `record()`/
      `increment()`/`decrement()`.

    view controls which aggregation to return:
      "summary"       (default) — total event count + total value across
                       every event_type for this metric_id.
      "by_event_type"           — per-event_type breakdown, largest total
                       value first.
      "trend"                   — day/week-bucketed history (see `bucket`/`n`).
      "detail"        (issue #249) — raw per-event rows, most-recent first,
                       with each row's `metadata` deserialized and rendered
                       (see `n` below for how many rows). This is the view
                       for the richer per-run detail a domain stores in
                       `metadata` (e.g. my-gh-autowork's model/rounds/
                       wall_clock_s/findings) that summary/by_event_type
                       never surface, since those two only ever aggregate
                       `value`.

    bucket ("day" or "week", default "week") only applies to view="trend" —
      ignored otherwise.

    n (default 12) means "max buckets" for view="trend", or "max rows" for
      view="detail" — ignored for "summary"/"by_event_type". Reused across
      both rather than adding a second row-count parameter, since exactly
      one of the two meanings ever applies for a given call.

    session_id (issue #248, extended to "detail" by issue #249), when given,
      narrows view="summary"/"by_event_type"/"detail" down to rows matching
      this exact session_id AND metric_id — e.g. one specific
      `my-gh-autowork` attempt's own `f"issue-{n}-{HHMMSS}"` marker (issue
      #210), instead of that metric_id's entire history. Omitting it (the
      default) keeps today's metric_id-only aggregate/listing unchanged.
      Ignored for view="trend" — per-session filtering doesn't compose with
      time-bucketed grouping (out of scope for issue #248; see that issue
      for why), so a session_id passed alongside view="trend" has no effect
      on the result.

    format controls the output format:
      "text" (default) — human-readable text.
      "json"            — JSON object with keys "view"/"data".
    """
    if not _METRICS_LIB_AVAILABLE:
        return "Error: libs/metrics_lib.py is not available in this install -- get_metrics requires it."
    assert metrics_lib is not None  # _METRICS_LIB_AVAILABLE implies this
    if view not in ("summary", "by_event_type", "trend", "detail"):
        return f"Error: unrecognised view {view!r}. Valid values: 'summary', 'by_event_type', 'trend', 'detail'."
    if format not in ("text", "json"):
        return f"Error: unrecognised format {format!r}. Valid values: 'text', 'json'."
    # Copilot review on this PR: bucket/n are documented as "ignored" for
    # views that don't use them, but were validated unconditionally
    # regardless of view -- so an irrelevant bucket="month" or n=0 rejected
    # a perfectly good view="summary" call, contradicting that doc. Scoped
    # to only the views that actually consume each argument: bucket is
    # trend-only; n applies to trend (bucket count) AND detail (row limit).
    if view == "trend" and bucket not in ("day", "week"):
        return f"Error: unrecognised bucket {bucket!r}. Valid values: 'day', 'week'."
    if view in ("trend", "detail") and (not isinstance(n, int) or n < 1):
        return "Error: n must be a positive integer."
    try:
        store = metrics_lib.MetricsStore()
        if view == "summary":
            summary_data = store.summary(metric_id, session_id=session_id)
            return metrics_lib.format_json("summary", summary_data) if format == "json" else metrics_lib.format_summary_view(summary_data, session_id=session_id)
        if view == "by_event_type":
            by_type_data = store.by_event_type(metric_id, session_id=session_id)
            return metrics_lib.format_json("by_event_type", by_type_data) if format == "json" else metrics_lib.format_by_event_type_view(metric_id, by_type_data, session_id=session_id)
        if view == "detail":
            detail_data = store.detail(metric_id, session_id=session_id, limit=n)
            return metrics_lib.format_json("detail", detail_data) if format == "json" else metrics_lib.format_detail_view(metric_id, detail_data, session_id=session_id)
        # view == "trend"
        trend_data = store.trend(metric_id, bucket=bucket, n=n)
        return metrics_lib.format_json("trend", trend_data) if format == "json" else metrics_lib.format_trend_view(metric_id, trend_data, bucket=bucket)
    except Exception as e:
        # Same reasoning as savings_summary's try/except -- best-effort feature,
        # must not fail the tool call over a DB problem.
        return f"Error: could not read metrics data ({e}). The metrics database may be corrupted or inaccessible; this doesn't affect anything else."


@mcp.tool()
def record_metric(
    metric_id: str,
    event_type: str,
    value: float = 1.0,
    metadata: Optional[Union[str, dict]] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Writes one append-only event to the shared cross-tool metrics store
    (issue #210 — the write counterpart to get_metrics). Any domain in
    this toolkit can call this to record an event. my-gh-autowork's
    per-ticket outcome logging was the first consumer (cut over from
    compact_store by this ticket); it now goes through
    tools/autowork_cost.py record, which writes the same metric_id="autowork"
    rows, and this tool remains for anything that CLI can't cover.

    metric_id: the domain (e.g. "autowork", "memory-bank") — caller-defined,
      no fixed enum.
    event_type: domain-defined event kind (e.g. "ticket_merged", "ticket_blocked").
    value: a signed numeric delta (default 1.0). Must be finite — non-finite
      values (inf/-inf/nan) are rejected by MetricsStore.record() and logged to
      stderr; this tool surfaces that as an "Error:..." return instead.
    metadata: optional domain-specific extra fields, as either a JSON string
      (e.g. '{"issue": 42, "pr": 101, "rounds": 2}') or an already-parsed
      dict — the schema accepts both shapes directly. A string is validated
      as JSON and must decode to an object; an invalid JSON string, or one
      that decodes to a non-object (a list/scalar), is returned as an
      "Error:..." string rather than raising. The dict form exists because,
      back when this parameter was typed as a plain string, some MCP clients
      silently coerced a string argument that happened to look like JSON into
      a native object before this tool ever saw it, bypassing that
      string-only schema entirely and failing every such call outright
      (confirmed live: Claude Code itself does this coercion for a
      JSON-object-shaped string argument). Accepting both shapes now means a
      call succeeds regardless of which one the calling client actually
      sends.
    session_id: optional caller-supplied session identifier — stored on the
      row for later per-session filtering via get_metrics(view="summary"/
      "by_event_type"/"detail", session_id=...) (issue #248, extended to
      "detail" by issue #249).

    Returns "OK" on success, or an "Error:..." string on any failure —
    including input-validation failures (non-finite value, invalid metadata
    JSON) AND DB write failures (MetricsStore.record() returns False, which
    this tool maps to an "Error:..." string). DB write failures are also
    logged to stderr and never raised (fail-open: a broken/locked/corrupt
    metrics db must never break the actual domain operation being measured),
    but the "Error:..." return is the only signal the tool's caller sees.
    """
    if not _METRICS_LIB_AVAILABLE:
        return "Error: libs/metrics_lib.py is not available in this install -- record_metric requires it."
    assert metrics_lib is not None  # _METRICS_LIB_AVAILABLE implies this
    import json as _json
    import math as _math
    if not _math.isfinite(value):
        return f"Error: value must be finite (got {value!r})."
    metadata_dict: Optional[dict] = None
    if isinstance(metadata, dict):
        metadata_dict = metadata
    elif metadata is not None:
        try:
            metadata_dict = _json.loads(metadata)
        except (ValueError, TypeError) as e:
            return f"Error: metadata must be valid JSON if supplied ({e})."
        if not isinstance(metadata_dict, dict):
            return "Error: metadata must be a JSON object (dict), not an array or scalar."
    store = metrics_lib.MetricsStore()
    # record() fails open (returns False, logs to stderr) -- a broken metrics
    # db must never look like a tool call failure for the domain operation
    # that triggered the logging. Map a False return to an "Error:..." string
    # so the SKILL.md "Error:" check can detect it, rather than unconditionally
    # returning "OK" even when the write silently failed (Copilot review on
    # PR #247: an always-"OK" return made the SKILL.md error check unreachable
    # for the most important failure mode — an unwritable or locked db path).
    ok = store.record(
        metric_id=metric_id,
        event_type=event_type,
        value=value,
        metadata=metadata_dict,
        session_id=session_id,
    )
    if not ok:
        return f"Error: could not write metric {metric_id}/{event_type} to the database (details logged to server stderr)."
    return "OK"


if __name__ == "__main__":
    mcp.run()
