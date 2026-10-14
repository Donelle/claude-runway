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
from typing import Optional

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
# `savings_ledger = None` in the except branch (adopting Pyright)
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

# Cap on how much of a fetch_url response body is ever read into
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

    `source` is redacted (not just truncated) before being embedded here:
    it's the raw caller-supplied value --
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
        # Exact character counts. The chars / 3.5 token
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

    This used to report every model /v1/models returned, which
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
    """Decode a file's raw bytes without silently corrupting them.

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
    # Confirmed still needed under Pyright (which replaced mypy):
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
    compress_text, which default it off) -- Shell command output
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
    Without this, a URL that happens to resolve to a very large
    resource (a redirect to a video/zip/dataset, a poorly-bounded API
    response) would fully download into memory before anything checked its
    size, since timeout_seconds is a per-read/connect timeout, not a
    total-duration cap.
    """
    try:
        # a bare `resp = requests.get(...)` left
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
            # Requests derives resp.encoding from the header
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
                # a response with no Content-Type
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

    # resp.text's OWN implementation catches
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


if __name__ == "__main__":
    mcp.run()
