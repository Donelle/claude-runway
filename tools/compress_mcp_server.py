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
       section, and local_compress_lib.stale_env_warning for the guard against
       the pre-rename names (LMSTUDIO_BASE_URL / LMSTUDIO_MODEL /
       HOOK_COMPRESS_THRESHOLD_CHARS), which are no longer read.
           }
         }
       }

OPEN DESIGN QUESTIONS (marked NOTE: inline) -- flagging these rather than
silently picking an answer, since this is meant to be iterated on:
  1. RESOLVED: inputs under skip_if_under_chars return silently unchanged.
     The compressed path always self-documents via the "[compressed X -> Y
     chars]" prefix, so a caller can already tell skip vs. compress just by
     checking for that prefix -- no separate flag needed.
  2. RESOLVED: model resolves in this order -- explicit model= param, then
     CLAUDE_RUNWAY_LMSTUDIO_MODEL env var, then auto-detect IF exactly one model is
     loaded in LM Studio. Auto-detect refuses to guess (clear error instead)
     when zero or multiple models are loaded, since silently picking "the
     first one" could quietly use the wrong model. Re-checked live on every
     call rather than cached, since you can swap models in LM Studio without
     restarting this server.
  3. RESOLVED: large inputs are chunked and summarized piece-by-piece
     (map-reduce), not truncated. Truncation was a real bug, not just a
     simplification -- logs put failures at the END, so cutting the tail is
     exactly wrong for the content this tool targets. Every chunk gets
     summarized; if there's more than one chunk, a final reduce pass
     combines the per-chunk summaries into one coherent result. Costs more
     local time/compute for huge inputs, which is the correct tradeoff since
     local compute is free against the actual budget (Claude tokens).
     A max_total_chars safety ceiling still exists, but it REFUSES with a
     clear error rather than silently dropping content.
  4. RESOLVED (with a bigger fix than expected): compress_text(text=...)
     requires Claude to already hold the raw content to pass it as an
     argument -- meaning the token cost was already paid (input to read it
     in, output to re-emit it as the argument) BEFORE compress_text ever
     runs. That's strictly worse than doing nothing, for the exact case this
     tool targets. Fixed by adding compress_file and compress_command_output,
     which read the file / run the command server-side -- the raw content
     never has to pass through Claude at all, only the compressed result
     does. compress_text is kept only for the narrower case of compressing
     content Claude already legitimately holds (e.g. its own long draft)
     -- it does NOT save tokens if the input had to be read into context
     first just to call it. Usage guidance (this same item) now points
     Claude at compress_file/compress_command_output as the default choice.
  5. ADDED: fetch_url, for the same reason compress_command_output exists
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

import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import requests
import trafilatura
from mcp.server.mcpserver import MCPServer, Context
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from qdrant_client import QdrantClient, models as qdrant_models

from local_compress_lib import (
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_FOCUS,
    client as _client,
    compress as _compress_impl,
    derive_compact_label,
    estimate_tokens,
    redact_credentials,
)
from mcp_tool_introspect import describe_tools, tool_count
from qdrant_retry import call_with_retry

# savings_ledger is optional at import time (e.g. an older checkout of just
# this file without libs/savings_ledger.py) -- fails open to "tracking off"
# rather than crashing server startup over an opt-in feature.
try:
    import savings_ledger
    _SAVINGS_LEDGER_AVAILABLE = True
except ImportError:
    _SAVINGS_LEDGER_AVAILABLE = False

TRACK_SAVINGS = _SAVINGS_LEDGER_AVAILABLE and savings_ledger.tracking_enabled()

# metrics_lib is likewise optional at import time (issue #208) -- same
# fail-open reasoning as savings_ledger above: an older checkout of just this
# file, without libs/metrics_lib.py, must not crash server startup over a
# brand-new, still domain-empty store.
try:
    import metrics_lib
    _METRICS_LIB_AVAILABLE = True
except ImportError:
    _METRICS_LIB_AVAILABLE = False

COMPACT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COMPACT_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COMPACT_COLLECTION = os.environ.get("COMPACT_COLLECTION", "conversation-compacts")

# Historical default from qdrant-client's QdrantFastembedMixin (the
# QdrantClient.add()/.query() convenience methods compact_store/compact_find
# used to call directly -- deprecated since ~1.7, actually removed in 1.19.0,
# which is what broke this repo's own new CI on its first run: see issue
# #51's PR). Neither tool ever called set_model(), so every compact stored
# before this fix was embedded with exactly this model, under the vector
# name FastEmbedProvider.get_vector_name() derives from it
# ("fast-bge-small-en") -- changing either here would silently orphan every
# previously stored compact rather than erroring.
COMPACT_EMBEDDING_MODEL = "BAAI/bge-small-en"

DEFAULT_WEB_FOCUS = (
    "Extract and summarize the information on this page most relevant to "
    "what was asked. Preserve specific facts, numbers, names, dates, and "
    "technical details verbatim where possible. Drop navigation, ads, and "
    "boilerplate."
)

# Issue #39: fetch_url used to call requests.get() with no stream=True, no
# Content-Length check, and no size cap of any kind -- resp.text/resp.content
# both materialize the ENTIRE body into memory unconditionally before
# max_total_chars (a post-extraction cap) ever gets a chance to run.
# timeout_seconds is a per-read/connect timeout, not a total-duration cap, so
# a large file streamed at steady throughput never trips it -- a URL that
# happens to resolve to a multi-GB file (a redirect to a video/zip/dataset, a
# poorly-bounded API response, or simply the wrong link) would fully download
# into memory first. 10 MB is comfortably above any real HTML/article page
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
    footer is appended there, and out_tokens is measured on it, since that's
    what Claude will actually see.

    hooks/compress_bash_output.py is the one that STRIPS this footer before
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
    hooks/compress_bash_output.py, so an unredacted `source` would turn a
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
        "raw_tokens": estimate_tokens(raw_text),
        "out_tokens": estimate_tokens(outer_text),
        "credited": credited,
        "source": redacted_source[:200],
    })
    return f"{outer_text}\n<!--CLAUDE_RUNWAY_SAVINGS:{footer}-->"


def _log_fixed_overhead():
    """
    Estimates the fixed per-turn context cost of this server's tool schemas
    (name + signature + docstring, as a proxy for the real JSON schema
    Claude Code sends) and stashes it via savings_ledger.set_meta so
    hooks/session_end_savings.py and the savings_summary/savings_detail
    tools can report it without recomputing it or importing this module.
    Runs once at startup -- self-updating on every restart, so it never
    goes stale the way a hardcoded constant would if a tool's docstring or
    signature changes. Best-effort: never blocks server startup.

    Enumerates this server's own @mcp.tool()-decorated functions via
    libs/mcp_tool_introspect.py (a shared utility, issue #52/GROW-05)
    instead of a hardcoded tuple -- a fixed tuple silently went stale here
    once already (issue #22: written against 5 tools, never updated when
    compact_store/compact_find/savings_summary/savings_detail shipped,
    understating the real overhead for months). See that module's own
    docstring for why it reaches into MCPServer's private "_tool_manager"
    attribute instead of the public (but async) list_tools().
    """
    if not TRACK_SAVINGS:
        return
    try:
        blob = describe_tools(mcp)
        overhead = estimate_tokens(blob)
        savings_ledger.set_meta("schema_overhead_tokens", str(overhead))
        print(f"[claude-runway] local-compress schema overhead ~{overhead} tokens ({tool_count(mcp)} tools)", file=sys.stderr)
    except Exception as e:
        print(f"[claude-runway] could not estimate schema overhead: {e}", file=sys.stderr)


@mcp.tool()
def list_local_models(base_url: Optional[str] = None) -> str:
    """
    List model IDs currently available on the local LM Studio server. Use
    this to find the exact model string for compress_text's model param, or
    to check LM Studio is actually reachable before relying on it.
    """
    try:
        models = _client(base_url).models.list()
    except Exception as e:
        return f"Could not reach LM Studio at {base_url or DEFAULT_BASE_URL}: {e}"
    ids = [m.id for m in models.data]
    return ("Available local models: " + ", ".join(ids)) if ids else "LM Studio is reachable but no model is loaded."


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
    return await _compress_impl(
        text, focus, skip_if_under_chars, chunk_chars, max_total_chars, max_chars,
        model, base_url, preserve_identifiers=preserve_identifiers,
        preserve_sections=preserve_sections, ctx=ctx,
    )


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
    # annotation while suppressing mypy's "None isn't a valid Context default"
    # complaint per-line, avoiding the need to widen to Optional[Context].
    # Same reasoning at every other `ctx` parameter below and in
    # ingest_mcp_server.py.
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
    # Explicit encoding: otherwise a non-ASCII file decodes per the platform
    # locale and errors="ignore" silently drops the bytes it can't handle, so
    # the model summarizes subtly corrupted input and reports nothing wrong.
    text = path.read_text(encoding="utf-8", errors="ignore")
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
            if not encoding:
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
    ordinary whole-text compression. /my-compact sets both flags.
    """
    return await _compress(
        text, focus, skip_if_under_chars, chunk_chars, max_total_chars,
        model, base_url, ctx, preserve_identifiers=preserve_identifiers,
        preserve_sections=preserve_sections,
    )


def _normalize_project(project: str) -> str:
    """
    Case-fold a project string (issue #37), so "MyProject" and "myproject"
    derive the identical CANONICAL collection name via _sanitize_project
    (its only caller). casefold() (not .lower()) is used deliberately --
    it's the Unicode-aware form intended for caseless comparison (e.g. it
    correctly equates German "ß" with "ss", which .lower() does not).

    IMPORTANT (round 2 of issue #37, flagged in PR review on #123 as a
    docstring/implementation mismatch that a prior version of this
    function's docs left in place): this is NOT called anywhere else in
    compact_store/compact_find, and cross-casing identity for a project
    with EXISTING history is deliberately NOT handled by normalizing a
    stored value at all -- compact_store writes the RAW `project` argument
    into both the payload's `project` field and _compact_point_id, and an
    explicit `collection` override on compact_find filters on the RAW
    value too. Case-insensitive matching against existing data instead
    happens entirely via case-insensitive COMPARISON at read/resolution
    time (see _collections_for_project, and the per-entry `.casefold()`
    filtering in compact_find's default path) -- there is no single
    "normalized form" that every stored entry is expected to already be
    in, since entries written under different historical casings
    legitimately coexist and are matched by comparison, not by rewriting
    history to agree with each other.
    """
    return project.casefold()


def _sanitize_project(project: str) -> str:
    """
    Replace characters not valid in Qdrant collection names with hyphens,
    then append a short hash suffix derived from the ORIGINAL (unsanitized,
    but already case-folded -- see below) project string.

    The suffix exists to close a real collision: two distinct project
    strings that differ only in punctuation/whitespace -- "my.project" and
    "my-project", or "my project", or "my@project" -- all sanitize to the
    identical "my-project" without it, since every disallowed character
    independently collapses to the same hyphen. That would put two distinct
    projects' /my-compact entries in the same underlying Qdrant collection
    (issue #45). Hashing the pre-sanitization string (not the lossy
    sanitized one) makes that specific punctuation/whitespace-driven
    collision extremely unlikely in practice, because the hash input still
    carries the distinction the sanitization step just threw away -- but
    this is NOT an absolute guarantee: the suffix is only the first 8 hex
    chars of a SHA-256 digest, a 32-bit (~4.3 billion value) space, so two
    different original strings can still coincidentally share the same
    truncated hash (verified directly: brute-forcing "project-<n>" strings
    hits a real collision by n=161010, in line with the ~2^16 birthday-bound
    expectation for a 32-bit space). Don't reason about this as
    collision-proof isolation; it only defends against the one collapse
    class described above, not against 32-bit hash birthday collisions in
    general.

    Case-folds `project` FIRST (issue #37), before either the hyphen
    substitution or the hash, so "MyProject" and "myproject" -- unlike the
    punctuation variants above, which are deliberately kept distinct --
    collapse onto the identical sanitized string AND hash. This is the
    opposite of the punctuation case on purpose: punctuation differences
    are treated as different projects, casing differences are treated as
    the same project. Without this, a casing drift between /my-compact and
    /my-resume (different terminal profile, a renamed directory recreated
    with different casing, etc.) pointed at an entirely different,
    nonexistent collection, producing a misleading "no collection found"
    message instead of just finding the right one.

    Deterministic (same input -> same output every call) is required here:
    compact_store and compact_find each call this independently and must
    land on the same collection name for the same project, or /my-resume
    would never find what /my-compact just stored.
    """
    import hashlib as _hashlib
    import re as _re
    normalized = _normalize_project(project)
    sanitized = _re.sub(r"[^a-zA-Z0-9_-]", "-", normalized)
    suffix = _hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}-{suffix}"


def _collections_for_project(client: QdrantClient, project: str) -> list:
    """
    Every existing '<COMPACT_COLLECTION>-*' collection that already holds
    THIS logical project's history, case-insensitively.

    Round 2 of issue #37 (flagged in PR review on #123): an earlier
    version of this fix picked exactly ONE collection per call (a
    "legacy vs. canonical" choice), which still silently fragmented
    history across a genuine casing drift -- reproduced directly: a
    pre-existing "MyProject" collection plus a later /my-compact call
    passing "myproject" landed on the NEW case-folded canonical
    collection (since a lowercase input's legacy-style and canonical
    names coincide, so the old resolver's "does the legacy name exist"
    check never even looked for "MyProject"), after which lowercase
    reads only ever saw the new collection and uppercase reads only ever
    saw the old one -- two disjoint views of the same project, exactly
    the "does not resolve to the same entries as promised" failure the
    review called out.

    This replaces that single-collection choice with a case-insensitive
    SCAN across every collection that currently exists: the canonical
    (case-folded) name is checked first, then every OTHER
    '<COMPACT_COLLECTION>-*' collection is checked via
    _collection_contains_project, which scans for ANY stored point whose
    `project` payload field case-folds to match -- not just its most
    recent point (an earlier version of this scan did exactly that, which
    is its own bug; see _collection_contains_project's docstring, round
    4). Comparing against REAL stored values (rather than re-deriving a
    name algorithmically, as the round-1 fix did) is what lets this find
    a match under ANY historical casing, not just the one non-case-folded
    algorithm this file used before issue #37 -- there could in principle
    be more than two differently-cased collections for the same project
    by now.

    Deliberately uncapped (unlike _candidate_hint, which caps the number
    of DISPLAYED "did you mean" candidates at a handful, purely for
    AskUserQuestion's option-count limit) -- this is used for actual data
    resolution, not just a suggestion, so truncating it could silently
    hide a real match instead of merely shortening a hint. Compact-history
    collections are per-project and few in practice (this is a personal
    productivity tool called a handful of times per session, not a hot
    path), so the extra Qdrant round-trips this costs are an acceptable,
    deliberate trade of a little latency for actually-correct results,
    not an oversight.

    Deliberately does NOT swallow errors the way _sibling_compact_collections
    (a pure best-effort DISPLAY hint) does (flagged in PR review on #123):
    this function's result feeds an AUTHORITATIVE decision -- which
    collection compact_store writes into, and what compact_find reports
    existing at all -- so silently treating a failed get_collections()
    call (all retries exhausted) as "no other collections exist" would let
    compact_store create a second, fragmenting canonical collection right
    next to a real one it just couldn't see, or let compact_find falsely
    report a project as never-compacted. Letting the exception propagate
    means the caller sees a real error instead of confidently wrong data.
    _collection_contains_project (called below) is the same story: it
    doesn't swallow scroll failures either, so a failure probing one
    candidate collection here also propagates rather than being silently
    treated as "this collection doesn't match."

    Returns collection names only. The canonical name is trusted by
    construction (deterministic, hash-suffixed, one project = one name --
    the same isolation guarantee the pre-issue-#37 design always relied
    on). Every OTHER returned name is only PROBABLY project-exclusive --
    proven only by containing AT LEAST ONE matching point (one match
    proves containment, not exclusivity: the rest of that collection's
    points could still belong to other projects entirely, e.g. an
    explicitly shared collection someone deliberately created via the
    `collection` override), never by any persisted metadata that actually
    distinguishes a dedicated collection from a shared one -- so
    compact_find still applies a per-entry case-insensitive filter on top
    of whatever this returns, rather than trusting collection membership
    alone as isolation (see compact_find's own comment at its filtering
    step). The explicit `collection` override on compact_store/compact_find
    bypasses this function entirely and keeps its own exact-match filter,
    since that path is a deliberately shared, caller-controlled collection
    from the start.
    """
    canonical_col = f"{COMPACT_COLLECTION}-{_sanitize_project(project)}"
    matches = []
    if call_with_retry(client.collection_exists, canonical_col):
        matches.append(canonical_col)
    target = project.casefold()
    response = call_with_retry(client.get_collections)
    all_names = [c.name for c in response.collections if c.name.startswith(f"{COMPACT_COLLECTION}-")]
    for name in all_names:
        if name == canonical_col:
            continue
        if _collection_contains_project(client, name, target):
            matches.append(name)
    return matches


def _collection_contains_project(client: QdrantClient, collection_name: str, target_casefolded: str) -> bool:
    """
    True if ANY point in `collection_name` has a `project` payload field
    matching `target_casefolded` case-insensitively.

    _collections_for_project used to decide a candidate collection's
    identity purely from its MOST RECENT point (via
    _most_recent_sibling_entry) -- wrong for a genuinely shared collection
    where a DIFFERENT project's entry happens to carry a LATER date than
    this project's own (flagged in PR review on #123, round 4): that
    older, real history would be silently excluded from the match
    entirely, so compact_find would omit it and compact_store could
    create a second, fragmenting canonical collection alongside a shared
    one that actually already holds this project's data. Scanning for ANY
    matching point, not just the newest one, is what actually answers
    "does this collection hold this project's history" -- using recency
    as a stand-in for identity was the bug.

    Scans the WHOLE collection (paginated, same pattern as
    _most_recent_sibling_entry -- and returns as soon as a match is found,
    so this is often cheaper in practice, not more expensive, despite
    being more thorough in the worst case). Deliberately does NOT catch
    exceptions (no try/except) for the same reason _collections_for_project
    itself doesn't: this feeds an authoritative decision, not a
    best-effort hint, so a scroll failure must propagate rather than be
    silently treated as "no match here."
    """
    offset = None
    while True:
        batch, offset = call_with_retry(
            client.scroll, collection_name=collection_name, limit=100, offset=offset, with_payload=True,
        )
        for p in batch:
            if (p.payload or {}).get("project", "").casefold() == target_casefolded:
                return True
        if offset is None:
            break
    return False


def _collection_project_values(client: QdrantClient, collection_name: str, target_casefolded: str) -> list:
    """
    Every DISTINCT literal `project` payload value stored in
    `collection_name` that case-insensitively matches `target_casefolded`
    -- there can be more than one (e.g. "MyProject" and "myproject" both
    landed in the same collection over time, since compact_store
    converges future writes onto whichever existing collection already
    matches, regardless of the exact casing each individual call passed).

    Used to build an EXACT, Qdrant-SIDE MatchAny filter for a non-canonical
    collection in compact_find's semantic query path (issue #37 round 5,
    flagged in PR review on #123): an earlier version instead over-fetched
    a larger, but still fundamentally CAPPED, number of results and
    filtered client-side afterward -- which the review correctly pointed
    out just moves the truncation-before-filter bug to a bigger threshold
    rather than eliminating it (a long-lived shared collection with more
    higher-scoring OTHER-project points than the over-fetch cap would
    still silently omit real matches). Filtering AT the Qdrant query
    itself, before its own internal ranking/limit ever truncates
    anything, is what actually eliminates the class of bug -- not a
    bigger buffer.

    Unlike _collection_contains_project (which only needs to prove
    non-emptiness and can short-circuit on the first match), this needs
    EVERY distinct value to build a correct filter, so it scans the whole
    collection unconditionally. Deliberately does NOT catch exceptions,
    for the same reason the rest of this call chain doesn't: this feeds
    an authoritative decision.
    """
    values = set()
    offset = None
    while True:
        batch, offset = call_with_retry(
            client.scroll, collection_name=collection_name, limit=100, offset=offset, with_payload=True,
        )
        for p in batch:
            v = (p.payload or {}).get("project", "")
            if v.casefold() == target_casefolded:
                values.add(v)
        if offset is None:
            break
    return list(values)


# Fixed namespace for _compact_point_id's uuid5 derivation -- any constant
# UUID works here (uuid5's own spec just requires SOME namespace), this one
# is simply namespaced under this project's own URL so it doesn't collide
# with some other subsystem's unrelated uuid5 usage that happened to pick
# the same namespace by coincidence.
_COMPACT_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "compact_store:01a0a23a-3633-7309-96ee-244a4002cd96"
)


def _compact_point_id(project: str, label: str, date: str) -> str:
    """
    Deterministic Qdrant point id for a compact, derived from
    project+label+date (issue #36).

    compact_store used to assign every stored compact a fresh uuid.uuid4(),
    which is why a retried /my-compact call (an MCP transport hiccup, or a
    user/model re-running /my-compact before confirming the first call
    succeeded -- /my-compact's own SKILL.md calls compact_store exactly
    once per run, so the retry has to come from outside that skill) landed
    on a second, indistinguishable point next to the first: two random ids
    never collide, so Qdrant just inserted both. Deriving the id instead of
    randomizing it means a retry with the SAME project/label/date always
    computes the SAME id, so client.upsert() overwrites the earlier point
    in place rather than inserting a sibling -- upsert-by-deterministic-id,
    the approach the issue's own suggested fix names first.

    uuid5 (not uuid4) is required here specifically because it has to be a
    pure function of its inputs -- same project/label/date must always
    produce the same id on every call, on every process, forever, or this
    wouldn't dedupe anything. The null-byte separator between fields (\\x00)
    is deliberately not something project/label/date names are expected to
    contain, so "a" + "b|c" can't collide with "a|b" + "c" the way a plain
    "|" join could.

    This does NOT dedupe two intentionally different compacts that happen
    to share the same label and date (e.g. saved twice in one day under
    the same label on purpose) -- that's an upsert-overwrite by design, not
    a bug: project+label+date IS the identity key the issue asks for.
    """
    key = f"{project}\x00{label}\x00{date}"
    return str(uuid.uuid5(_COMPACT_ID_NAMESPACE, key))


@mcp.tool()
async def compact_store(
    information: str,
    project: str,
    date: str,
    label: str = "",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection: Optional[str] = None,
) -> str:
    """
    Store a conversation compact in a dedicated Qdrant collection, isolated
    from the codebase index. The collection defaults to
    '<COMPACT_COLLECTION>-<sanitized-project>-<hash8>' (e.g.
    'conversation-compacts-Acme-Support-TicketsApi-3f9a2b1c'), giving each
    project its own collection so compact_find never sees another project's
    entries. Non-alphanumeric characters in the project name (dots, spaces,
    etc.) are replaced with hyphens to satisfy Qdrant naming rules, and an
    8-char hash of the original (pre-sanitization) project string is
    appended so that two distinct project names differing only in
    punctuation/whitespace (e.g. "my.project" vs "my-project") are extremely
    unlikely to collapse onto the same collection (issue #45) -- not an
    absolute guarantee, since the suffix is only a 32-bit truncated hash;
    see _sanitize_project's docstring for why. `project` is also case-folded
    before any of the above (issue #37), so "MyProject" and "myproject" are
    treated as the SAME project (the opposite of the punctuation case,
    which treats near-identical strings as different projects) for any
    BRAND-NEW project. If a collection already exists for this project
    under ANY casing (checked by reading a stored entry's own `project`
    payload field case-insensitively, not by re-deriving a name), writes
    continue landing in that existing collection instead of fragmenting
    into a new one -- see _collections_for_project. Pass collection
    explicitly to override.

    Called by /my-compact instead of qdrant-store. project, label, and date
    are stored as payload fields and used by compact_find for native filtering
    and sorting. The collection is created automatically on first use.

    `label` is optional (issue #72): when omitted or blank, the label is
    derived server-side from the "## What we were working on" section of
    `information` (the first sentence, up to 80 chars) using the same
    section-splitting logic that already parses the compact template.  This
    removes label-quality variance that came from asking each calling model to
    extract the first sentence on its own.  Pass an explicit `label` to
    override the derived value (e.g. from a /my-compact argument).

    Idempotent (issue #36): the stored point's id is deterministically
    derived from project+label+date (see _compact_point_id), not a random
    uuid4. A retried call with the same three values upserts onto the same
    id instead of inserting a second, indistinguishable point, so an
    external retry (an MCP transport hiccup, or a re-run before confirming
    the first call succeeded) can't silently duplicate a compact in the
    /my-resume picker.

    Requires mcp-server-qdrant + fastembed (both already installed via
    requirements.txt's Core section). QDRANT_URL / QDRANT_API_KEY /
    COMPACT_COLLECTION env vars are read from this server's .mcp.json env block.
    """
    import re as _re
    if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return f"Error: date must be ISO format YYYY-MM-DD (got {date!r}). compact_find sorts lexicographically, so non-ISO dates produce incorrect ordering."

    # Derive a label server-side when the caller didn't supply one (issue #72).
    # Uses the same split_sections logic that parse_compact already applies,
    # so this is deterministic and consistent across every calling model.
    if not label.strip():
        label = derive_compact_label(information) or "(unlabeled)"

    client = QdrantClient(
        url=qdrant_url or COMPACT_QDRANT_URL,
        api_key=(qdrant_api_key or COMPACT_QDRANT_API_KEY) or None,
    )
    canonical_col = f"{COMPACT_COLLECTION}-{_sanitize_project(project)}"
    if collection:
        # Caller-controlled override -- use exactly as given, no
        # case-insensitive resolution (the caller already fully controls
        # both sides of this lookup).
        col = collection
    else:
        # Reuse whichever existing collection (under ANY casing) already
        # holds this project's history, case-insensitively (issue #37
        # round 2 -- see _collections_for_project's docstring for why a
        # single legacy-vs-canonical choice wasn't enough: it could still
        # fragment history across a genuine casing drift). Prefer the
        # canonical (case-folded) collection if it's among the matches,
        # so writes keep converging on ONE collection going forward;
        # otherwise reuse whichever existing match was found. Falls back
        # to creating the canonical collection fresh if nothing matches
        # yet (a brand-new project).
        matches = _collections_for_project(client, project)
        col = canonical_col if canonical_col in matches else (matches[0] if matches else canonical_col)
    embedding_provider = FastEmbedProvider(COMPACT_EMBEDDING_MODEL)
    vector_name = embedding_provider.get_vector_name()
    # Every direct QdrantClient network call below is wrapped in
    # call_with_retry -- constructing FastEmbedProvider (loads the ONNX
    # model) and embed_documents() are both CPU-bound gaps during which
    # Docker Desktop's vpnkit can reap an idle pooled connection, so the
    # very next Qdrant call can raise a ResponseHandlingException on a
    # connection the client believes is still healthy. See issue #75/#76 /
    # libs/qdrant_retry.py, and ingest_mcp_server.py's identical pattern.
    if not call_with_retry(client.collection_exists, col):
        call_with_retry(
            client.create_collection,
            collection_name=col,
            vectors_config={
                vector_name: qdrant_models.VectorParams(
                    size=embedding_provider.get_vector_size(),
                    distance=qdrant_models.Distance.COSINE,
                )
            },
        )
    [embedding] = await embedding_provider.embed_documents([information])
    # Deterministic, not random (issue #36) -- see _compact_point_id's
    # docstring for why this is what makes a retry an overwrite instead of
    # a duplicate insert. Uses the raw `project` argument, same as before
    # issue #37 -- point-id dedup only ever needs to matter within a
    # single call's own retries, which always pass identical arguments,
    # so there's no cross-casing dedup need here the way there is for
    # collection/payload resolution above.
    point_id = _compact_point_id(project, label, date)
    call_with_retry(
        client.upsert,
        collection_name=col,
        points=[
            qdrant_models.PointStruct(
                id=point_id,
                vector={vector_name: embedding},
                # "document" mirrors the payload shape the now-removed
                # QdrantClient.add() used to write -- compact_find's
                # scroll-path fallback (.get("information", .get("document",
                # ""))) still checks for it on entries stored before this fix.
                # payload's "project" is the RAW argument (issue #37 round
                # 2) -- it doesn't need to agree with any particular casing
                # scheme, because nothing downstream expects it to already
                # BE normalized: _collections_for_project reads this exact
                # raw value back to decide whether a collection matches a
                # project (case-insensitively), and compact_find's default
                # (non-explicit-collection) read path applies its OWN
                # case-insensitive PER-ENTRY filter against this same raw
                # value (round 3, flagged in PR review on #123 -- an
                # earlier version of this comment claimed no filter was
                # applied at all, which stopped being true once that
                # per-entry filter was added). The exact literal value is
                # also useful as a human-readable record of what this
                # specific call actually passed.
                payload={"document": information, "project": project, "label": label, "date": date, "information": information},
            )
        ],
    )
    return f"Stored compact in '{col}': project={project!r}, date={date!r}, label={label!r}, id={point_id!r}"


def _sibling_compact_collections(client: QdrantClient, exclude: list) -> list:
    """
    List other '<COMPACT_COLLECTION>-*' collections besides everything in
    `exclude` (issue #37's recovery path). Used when NOTHING was found for
    a given project under any casing -- either this project was genuinely
    never compacted, or the project name has drifted entirely (a renamed
    directory, a different worktree checkout) from whatever it was when
    /my-compact ran (a pure casing drift never reaches here at all -- see
    _collections_for_project, which already finds any casing variant
    directly, so this is purely for a genuinely different name). `exclude`
    is always a list, never a bare string -- iterating a bare string would
    silently exclude by individual CHARACTER instead of by name. Returns
    an empty list (never raises) if the get_collections() call itself
    fails, so this stays purely additive -- a Qdrant hiccup here degrades
    back to the plain "not found" message instead of blowing up the whole
    call.

    Returns ALL matching sibling names, uncapped. An earlier version
    capped this list to `max_collections` (3) raw names before
    _candidate_hint ever got a chance to probe and skip the ones that
    turn out empty or unprobeable -- `compact_store` creates a collection
    before it has any points, so an empty sibling is a realistic state,
    and 3 empty siblings could hide a 4th with real saved compacts from
    ever being considered (flagged in PR review on #123, round 6). The
    actual display cap belongs in _candidate_hint below, applied AFTER
    filtering, not here.
    """
    try:
        response = call_with_retry(client.get_collections)
    except Exception:
        return []
    prefix = f"{COMPACT_COLLECTION}-"
    excluded = set(exclude)
    return [c.name for c in response.collections if c.name.startswith(prefix) and c.name not in excluded]


def _most_recent_sibling_entry(client: QdrantClient, collection_name: str):
    """
    Fetch one collection's most-recently-dated compact, keyed by ISO date
    string (lexicographic sort, same assumption compact_find's own
    date-sort already relies on). Used ONLY by _candidate_hint, a
    best-effort DISPLAY hint -- authoritative project resolution
    (_collections_for_project) now uses _collection_contains_project
    instead, precisely because "most recent point" is the wrong question
    to ask when deciding whether a collection belongs to a project at all
    (round 4, see that function's docstring); it's still exactly the
    right question for "what should we SHOW the user as this candidate's
    most recent activity," which is all _candidate_hint needs. This
    function itself does NOT swallow failures -- a scroll error
    propagates to the caller, and _candidate_hint wraps its own call in a
    try/except (a failed hint probe should degrade gracefully, never
    block the underlying result) rather than this shared helper deciding
    that for every caller.

    Pages through the WHOLE collection (mirroring compact_find's own
    scroll loop for its primary path) rather than reading a single capped
    page -- an earlier version of this function read only the first
    100-point page, which is ID-ordered, not date-ordered, so a
    collection with more than 100 points could have its actual
    most-recently-dated entry sitting on a later page and never get
    considered (flagged in PR review on #123, round 1). Returns None (not
    an exception) only for the genuinely different case of a collection
    that scrolled successfully but turned out to be empty -- that's a
    real, valid result, not a failure.
    """
    all_points = []
    offset = None
    while True:
        batch, offset = call_with_retry(
            client.scroll, collection_name=collection_name, limit=100, offset=offset, with_payload=True,
        )
        all_points.extend(batch)
        if offset is None:
            break
    if not all_points:
        return None
    best = max(all_points, key=lambda p: (p.payload or {}).get("date", ""))
    payload = best.payload or {}
    return payload.get("date", "?"), payload.get("label", "(no label)"), payload.get("project", "?")


def _candidate_hint(client: QdrantClient, exclude: list) -> str:
    """
    Builds the "did you mean one of these other projects" suffix shared by
    both of compact_find's fallback paths (collection-not-found and
    zero-results-within-an-existing-collection). `exclude` is every
    collection name already considered for this call (so it's never
    re-offered as if it were a NEW candidate) -- a list, since the
    default (non-explicit-collection) path can have considered more than
    one collection (issue #37 round 2's aggregation; see
    _collections_for_project). Returns "" when no sibling collections
    exist or none of them have any entries, so callers can just
    concatenate this onto their base message unconditionally.

    This function treats a failed _most_recent_sibling_entry probe as
    "skip this candidate" rather than letting it propagate (issue #37
    round 3, flagged in PR review on #123) -- deliberately: this function
    is a pure best-effort DISPLAY hint layered on top of an
    already-decided result (compact_find has already determined nothing
    matched before this ever runs), so one bad sibling collection
    degrading the suggestion list is the right failure mode. Contrast
    _collections_for_project's own resolution helper,
    _collection_contains_project, which deliberately does NOT catch its
    exceptions, for exactly the opposite reason (that path is
    authoritative, not a suggestion).

    The 3-candidate cap is applied HERE, to `hints` (candidates that
    actually turned out to have content), not to the raw sibling-name
    list _sibling_compact_collections returns -- capping the raw list
    first (an earlier version's bug, round 6) meant 3 empty or unprobeable
    siblings could hide a 4th with real saved compacts from ever being
    probed at all, so the recovery path could wrongly report zero
    candidates even when one genuinely existed further down the list.
    Stopping the loop itself once 3 real hints are collected (rather than
    probing every sibling and slicing afterward) also avoids scrolling
    entire collections that will just be discarded once the cap is hit.

    3, NOT an arbitrary round number: Claude Code's built-in
    AskUserQuestion tool hard-caps every question at 2-4 options total
    (confirmed directly against the CLI's own bundled schema, not just
    the reviewer's claim on PR #123 -- verified via `strings` against the
    installed `claude` binary, which contains the literal schema
    description "2-4 options" and "do not exceed them even if the user
    requests more"). my-resume/SKILL.md's "If no match is found" flow
    presents these candidates as AskUserQuestion options PLUS one extra
    "None of these" option, so up to 3 real candidates + 1 fallback stays
    within that hard 4-option ceiling.
    """
    max_candidates = 3
    hints: list = []
    for name in _sibling_compact_collections(client, exclude):
        if len(hints) >= max_candidates:
            break
        try:
            entry = _most_recent_sibling_entry(client, name)
        except Exception:
            continue
        if entry is None:
            continue
        date, label, stored_project = entry
        hints.append(f"  - project '{stored_project}' (most recent: {date} — {label})")
    if not hints:
        return ""
    return (
        "\n\nDid you mean one of these other projects, which DO have saved compacts "
        "(this can happen if the working directory was renamed, or you're in a "
        "different checkout/worktree)?\n" + "\n".join(hints)
    )


@mcp.tool()
async def compact_find(
    project: str,
    query: Optional[str] = None,
    limit: int = 10,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection: Optional[str] = None,
) -> str:
    """
    Retrieve conversation compacts for a project from its dedicated Qdrant
    collection ('<COMPACT_COLLECTION>-<sanitized-project>-<hash8>' by
    default -- see compact_store / _sanitize_project for why the hash suffix
    is there). The collection is project-scoped so results are already
    isolated; the payload filter on 'project' is an additional safeguard if
    a shared collection is passed explicitly via the collection param.

    Called by /my-resume instead of qdrant-find. Returns compacts sorted by
    date descending (most recent first), each with its date, label, and full
    content, ready for /my-resume to present as a selection list.

    When query is omitted, uses scroll (payload filter only, no vector
    ranking) so date-sort is reliable and recent entries are never buried by
    semantic distance. When query is provided, uses semantic search to narrow
    results to the closest match — suited for keyword-based selection like
    '/my-resume auth refactor'. limit caps results in both cases.

    Each entry's header also includes a short suffix of its own Qdrant
    point id (issue #36). Going forward, compact_store derives that id
    deterministically from project+label+date, so a retried call with the
    SAME three values now overwrites the existing point instead of
    inserting a second one -- there is nothing left to disambiguate for
    compacts stored under this scheme. The suffix exists for the LEGACY
    case: any point stored before this fix still carries its old random
    uuid4, and because compact_store used to accept unlimited calls with
    identical project+label+date as unlimited separate inserts, two such
    pre-fix points can legitimately have identical date/label already
    sitting in the collection today, which would otherwise render as
    indistinguishable entries in /my-resume's selection UI. Surfacing the
    id suffix here is what lets that UI (or a human reading raw output)
    tell those specific pre-existing entries apart.

    `project` is case-folded before deriving the canonical collection name
    (issue #37), so "MyProject" and "myproject" always land on the same
    NEW-style collection for a brand-new project. For a project with
    EXISTING history, this AGGREGATES entries across every
    '<COMPACT_COLLECTION>-*' collection that case-insensitively matches
    `project` -- not just one -- so a casing drift between /my-compact and
    /my-resume, or even multiple different historical casings accumulated
    over time, still surfaces everything instead of silently showing only
    whichever single collection happened to match (round 2 of issue #37,
    flagged in PR review on #123: picking exactly one collection could
    still fragment history across a genuine casing drift). See
    _collections_for_project. For a project with NO existing history under
    any casing, both the collection-not-found path and the
    zero-matching-entries path list any sibling '<COMPACT_COLLECTION>-*'
    collections' most recent entries as candidates, instead of just
    failing flatly -- these are necessarily a genuinely different project
    name at that point (a renamed working directory, a different worktree
    checkout), since any same-project casing variant would already have
    been picked up by the aggregation above.
    """
    client = QdrantClient(
        url=qdrant_url or COMPACT_QDRANT_URL,
        api_key=(qdrant_api_key or COMPACT_QDRANT_API_KEY) or None,
    )
    canonical_col = f"{COMPACT_COLLECTION}-{_sanitize_project(project)}"

    if collection:
        # Caller-controlled override -- a single, explicit, possibly
        # SHARED collection, so (unlike the default path below) isolation
        # isn't already guaranteed by construction and the exact-match
        # payload filter is the only thing keeping it project-scoped.
        col_list = [collection]
        project_filter = qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(key="project", match=qdrant_models.MatchValue(value=project))]
        )
        # See compact_store's identical comment -- every direct
        # QdrantClient call here is wrapped in call_with_retry for the
        # same CPU-bound-gap-then-dropped-connection reason (issue
        # #75/#76 / libs/qdrant_retry.py).
        if not call_with_retry(client.collection_exists, collection):
            return (
                f"No compacts collection found ('{collection}') for project '{project}'. "
                "Run /my-compact first to create it."
            ) + _candidate_hint(client, [collection])
    else:
        # Every collection (under ANY casing) that already holds this
        # project's history -- see _collections_for_project's docstring
        # for why this scans rather than picks one. No Qdrant-level
        # project_filter here (each collection this returns is EXPECTED
        # to already be scoped to this one project) -- but see the
        # case-insensitive PER-ENTRY filter applied below, after fetching,
        # which is the actual isolation guarantee for this path (issue
        # #37 round 3, flagged in PR review on #123). Every match
        # _collections_for_project returns -- INCLUDING the canonical
        # one -- is only PROVEN project-exclusive by inspecting stored
        # data, not by any persisted metadata that distinguishes a
        # dedicated collection from a shared one someone deliberately (or
        # accidentally) wrote into via compact_store's own `collection`
        # override: an earlier version of this comment treated the
        # canonical collection as exempt from that risk ("trusted by
        # construction"), which held for RESOLUTION (deciding a
        # collection belongs to this project) but not for ISOLATION
        # (nothing stops another project's points from landing inside it)
        # -- round 6's review caught the semantic query path still acting
        # on that stale exemption (see its own comment at the MatchAny
        # filter below). Without a per-entry check here too, such a
        # collection could leak another project's entries into this
        # result.
        col_list = _collections_for_project(client, project)
        project_filter = None
        if not col_list:
            return (
                f"No compacts collection found ('{canonical_col}') for project '{project}'. "
                "Run /my-compact first to create it."
            ) + _candidate_hint(client, [canonical_col])

    # Case-insensitive per-entry isolation for the default (non-explicit-
    # collection) path -- see the comment above where col_list was built.
    # None when `collection` was explicit (that path already has its own
    # exact-match Qdrant-level project_filter and doesn't need this).
    case_insensitive_target = None if collection else project.casefold()

    if query:
        # Semantic search — preserve relevance order, do not date-sort.
        embedding_provider = FastEmbedProvider(COMPACT_EMBEDDING_MODEL)
        query_vector = await embedding_provider.embed_query(query)
        raw = []
        for col in col_list:
            # For the default (non-explicit-collection) path, Qdrant's
            # OWN `limit` truncates results BEFORE any client-side filter
            # could ever see them -- so filtering has to happen AT the
            # query itself (server-side), not after, or a genuinely
            # SHARED collection with more than `limit` higher-ranked
            # points from a DIFFERENT project could fill the entire
            # returned page with points a client-side filter would then
            # remove, silently reporting no matches despite this
            # project's own (lower-ranked) real entries still sitting in
            # that same collection (flagged in PR review on #123, round
            # 5 -- an earlier version tried a bigger but still
            # fundamentally capped over-fetch instead, which the review
            # correctly pointed out just moves the same bug to a larger
            # threshold rather than eliminating it). This filter is now
            # built and applied for EVERY collection in col_list,
            # including the canonical one -- an earlier version exempted
            # the canonical collection as "trusted by construction," but
            # `collection` is a caller-controlled override on
            # compact_store's WRITE side too, so nothing actually
            # prevents another project's points from being written under
            # this exact canonical name; without a filter here, those
            # foreign points could rank ahead of the real ones and
            # reproduce the identical truncation-before-filter bug this
            # exemption was meant to sidestep (flagged in PR review on
            # #123, round 6). Every collection gets an EXACT MatchAny
            # filter built from its own actual stored project values
            # (there can be more than one casing variant within a single
            # collection -- see _collection_project_values), so Qdrant
            # itself only ever ranks and truncates already-isolated
            # points. A collection with zero matching literal values
            # (should not happen for anything _collections_for_project
            # returned, but defensively possible if data changed between
            # resolution and query) is skipped outright rather than
            # queried with an empty MatchAny, which Qdrant would treat as
            # "match nothing" anyway -- skipping just avoids the wasted
            # round trip.
            if case_insensitive_target is not None:
                values = _collection_project_values(client, col, case_insensitive_target)
                if not values:
                    continue
                col_filter: Optional[qdrant_models.Filter] = qdrant_models.Filter(
                    must=[qdrant_models.FieldCondition(key="project", match=qdrant_models.MatchAny(any=values))]
                )
            else:
                col_filter = project_filter
            raw.extend(call_with_retry(
                client.query_points,
                collection_name=col,
                query=query_vector,
                using=embedding_provider.get_vector_name(),
                query_filter=col_filter,
                limit=limit,
                with_payload=True,
            ).points)
        if case_insensitive_target is not None:
            # Defensive double-check, not the primary isolation mechanism
            # anymore (that's the per-collection MatchAny filter above) --
            # cheap, and matches this file's existing defense-in-depth
            # style (e.g. the retry-wrapped calls throughout).
            raw = [r for r in raw if (r.payload or {}).get("project", "").casefold() == case_insensitive_target]
        # Only matters when col_list has more than one entry (a genuine
        # historical casing-drift case) -- each collection's own results
        # already arrive relevance-sorted, so re-sort the COMBINED set by
        # score before capping to `limit`, rather than just concatenating
        # and cutting off partway through the second collection's results.
        # Real Qdrant ScoredPoint objects always carry `.score`; getattr's
        # default only matters for this file's own test fakes.
        raw.sort(key=lambda r: getattr(r, "score", 0), reverse=True)
        raw = raw[:limit]
        entries = [
            (r.id, (r.payload or {}).get("date", ""), (r.payload or {}).get("label", ""),
             (r.payload or {}).get("information", (r.payload or {}).get("document", "")))
            for r in raw
        ]
    else:
        # No query — page through ALL matching points in EVERY resolved
        # collection so date-sort covers the full combined set, not just
        # the first scroll page of the first collection (which is
        # ID-ordered, not date-ordered, so a cap-then-sort would miss
        # newer entries stored with higher IDs).
        all_points = []
        for col in col_list:
            offset = None
            while True:
                batch, offset = call_with_retry(
                    client.scroll,
                    collection_name=col,
                    scroll_filter=project_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )
                all_points.extend(batch)
                if offset is None:
                    break
        if case_insensitive_target is not None:
            all_points = [p for p in all_points if (p.payload or {}).get("project", "").casefold() == case_insensitive_target]
        entries = [
            (
                p.id, (p.payload or {}).get("date", ""), (p.payload or {}).get("label", ""),
                (p.payload or {}).get("information", (p.payload or {}).get("document", "")),
            )
            # qdrant-client types payload as Optional[dict] -- a point written
            # by this codebase always has one, but the client's own stubs
            # don't guarantee it, so this guards against a genuinely payload-
            # less point crashing the whole call instead of just losing that
            # one entry's fields.
            for p in all_points
        ]
        # Date is now index 1 (id is 0) -- sort key updated to match.
        entries.sort(key=lambda e: e[1], reverse=True)
        entries = entries[:limit]

    if not entries:
        return f"No compacts found for project '{project}'." + _candidate_hint(client, col_list)

    lines = [f"Found {len(entries)} compact(s) for project '{project}':\n"]
    for i, (point_id, date, label, information) in enumerate(entries, 1):
        # Short disambiguator, not the full id -- mirrors _sanitize_project's
        # existing hash8-suffix convention elsewhere in this file. Strip
        # dashes first so the 8 chars taken are hex digits, not separators.
        id_suffix = str(point_id).replace("-", "")[:8]
        lines.append(f"--- {i}. {date or '?'} — {label or '(no label)'} (id: {id_suffix}) ---")
        lines.append(information or "(no content)")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def compact_prune(
    project: str,
    keep_last_n: Optional[int] = None,
    older_than_days: Optional[int] = None,
    dry_run: bool = True,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection: Optional[str] = None,
) -> str:
    """
    Delete stored conversation compacts for a project, either by keeping
    only the N most recent entries or by removing entries older than a
    given number of days.

    Exactly one of `keep_last_n` or `older_than_days` must be provided;
    passing both, or neither, is an error.

    `dry_run` defaults to True -- reports what WOULD be deleted without
    removing anything. Pass dry_run=False explicitly to actually delete.
    This is the most important safety default: a prune call that accidentally
    matched the wrong project, or a keep_last_n that was set too aggressively,
    can be caught and aborted before any data is lost.

    `keep_last_n=N` retains the N most recent compacts by parsed ISO date
    and deletes the rest. Compacts whose stored date is missing, malformed,
    or calendar-invalid (e.g. "2026-99-99") cannot be date-ranked and are
    always preserved outside the N quota -- `keep_last_n=0` does NOT
    delete such entries. `keep_last_n=1` keeps only the most recently
    dated parseable compact (plus any unparseable ones).

    `older_than_days=N` deletes any compact whose parsed date is strictly
    before (today − N days). Entries whose date is today, in the future,
    or unparseable are never deleted by this mode. `older_than_days=0`
    deletes any entry dated before today (today-dated and future-dated
    entries survive). `older_than_days=1` deletes entries dated before
    yesterday (today's and yesterday's survive), etc.

    Uses the same project collection resolution (case-insensitive, multi-
    collection aggregation) as compact_find -- see _collections_for_project's
    docstring. The `collection` override bypasses that resolution in favour
    of a caller-controlled exact-match filter, same as compact_store and
    compact_find.

    Only individual POINTS are deleted, never the collection itself.
    """
    import datetime as _dt

    if (keep_last_n is None) == (older_than_days is None):
        return (
            "Error: provide exactly one of keep_last_n or older_than_days, not both or neither. "
            "Example: compact_prune(project='myproject', keep_last_n=5) "
            "or compact_prune(project='myproject', older_than_days=30)."
        )
    if keep_last_n is not None and keep_last_n < 0:
        return f"Error: keep_last_n must be a non-negative integer (got {keep_last_n!r})."
    if older_than_days is not None and older_than_days < 0:
        return f"Error: older_than_days must be a non-negative integer (got {older_than_days!r})."

    client = QdrantClient(
        url=qdrant_url or COMPACT_QDRANT_URL,
        api_key=(qdrant_api_key or COMPACT_QDRANT_API_KEY) or None,
    )
    canonical_col = f"{COMPACT_COLLECTION}-{_sanitize_project(project)}"

    if collection:
        col_list = [collection]
        project_filter = qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(key="project", match=qdrant_models.MatchValue(value=project))]
        )
        if not call_with_retry(client.collection_exists, collection):
            return (
                f"No compacts collection found ('{collection}') for project '{project}'. "
                "Run /my-compact first to create it."
            )
        case_insensitive_target: Optional[str] = None
    else:
        col_list = _collections_for_project(client, project)
        project_filter = None
        if not col_list:
            return (
                f"No compacts collection found ('{canonical_col}') for project '{project}'. "
                "Run /my-compact first to create it."
            )
        case_insensitive_target = project.casefold()

    # Gather every matching point across all resolved collections, with
    # collection membership so we know WHERE each point lives for deletion.
    # qdrant_client stubs type scroll() results as list[Record] (which is not
    # an importable public type in every version we support) -- use a plain
    # untyped list and suppress the two attribute accesses mypy flags below,
    # the same pattern call_with_retry's own callers use throughout this file.
    all_points: list = []  # list of (collection_name, Record)
    for col in col_list:
        offset = None
        while True:
            batch, offset = call_with_retry(
                client.scroll,
                collection_name=col,
                scroll_filter=project_filter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for p in batch:
                if case_insensitive_target is not None:
                    if (p.payload or {}).get("project", "").casefold() != case_insensitive_target:
                        continue
                all_points.append((col, p))
            if offset is None:
                break

    if not all_points:
        return f"No compacts found for project '{project}' — nothing to prune."

    def _parse_date(p) -> Optional[_dt.date]:  # type: ignore[no-untyped-def]
        """Parse the payload date field; return None for unparseable values."""
        try:
            return _dt.date.fromisoformat((p.payload or {}).get("date", ""))
        except (ValueError, TypeError):
            return None

    # Separate entries with valid parseable dates from those without.
    # Unparseable dates (malformed, missing, or calendar-invalid values like
    # "2026-99-99") must NOT participate in date-based ranking:
    # - For keep_last_n: a malformed date string sorts ahead of every valid
    #   2026 date lexicographically, so including it in the sort would
    #   silently treat it as "newest" and discard the genuinely newest real
    #   compact instead (Copilot review, round 1).
    # - For older_than_days: these are already excluded in that branch, but
    #   keeping the separation uniform makes the intent explicit in both paths.
    # Entries with unparseable dates are always preserved outside the quota.
    parseable: list = []
    unparseable_kept = 0
    for col, p in all_points:
        if _parse_date(p) is not None:
            parseable.append((col, p))
        else:
            unparseable_kept += 1

    # Sort parseable entries by date descending (most recent first).
    # All entries in `parseable` have a non-None _parse_date -- the list was
    # built by filtering out None results above, so the cast is safe.
    parseable.sort(key=lambda cp: _parse_date(cp[1]) or _dt.date.min, reverse=True)

    # Determine which points to delete.
    if keep_last_n is not None:
        # Keep the first keep_last_n parseable entries, delete the rest.
        to_delete = parseable[keep_last_n:]
        kept_count = min(keep_last_n, len(parseable)) + unparseable_kept
    else:
        # older_than_days: delete any parseable point whose date is strictly
        # more than older_than_days days before today.
        today = _dt.date.today()
        try:
            cutoff = today - _dt.timedelta(days=older_than_days)  # type: ignore[arg-type]
        except OverflowError:
            # older_than_days is astronomically large — nothing is that old.
            return (
                f"Nothing to prune for project '{project}': "
                f"older_than_days={older_than_days} exceeds the representable date range; "
                "no compact could be that old."
            )
        to_delete = []
        kept_count = unparseable_kept
        for col, p in parseable:
            entry_date = _parse_date(p)  # guaranteed non-None (parseable list)
            if entry_date < cutoff:  # type: ignore[operator]
                to_delete.append((col, p))
            else:
                kept_count += 1

    if not to_delete:
        return (
            f"Nothing to prune for project '{project}': "
            f"all {len(all_points)} compact(s) satisfy the retention criteria."
        )

    # Build a human-readable summary of what will be (or was) deleted.
    def _entry_summary(col: str, p) -> str:  # type: ignore[no-untyped-def]
        payload = (p.payload or {})
        date = payload.get("date", "?")
        label = payload.get("label", "(no label)")
        id_suffix = str(p.id).replace("-", "")[:8]
        return f"  {date} — {label} (id: {id_suffix}) [{col}]"

    summary_lines = [_entry_summary(col, p) for col, p in to_delete]
    summary_text = "\n".join(summary_lines)

    mode_desc = (
        f"keep_last_n={keep_last_n}" if keep_last_n is not None
        else f"older_than_days={older_than_days}"
    )

    if dry_run:
        return (
            f"[DRY RUN] Would delete {len(to_delete)} compact(s) for project '{project}' "
            f"({mode_desc}), retaining {kept_count}:\n"
            f"{summary_text}\n\n"
            "Pass dry_run=False to actually delete."
        )

    # Group deletes by collection to minimise round trips.
    by_col: dict[str, list] = {}
    for col, p in to_delete:
        by_col.setdefault(col, []).append(p.id)  # type: ignore[attr-defined]

    for col, ids in by_col.items():
        call_with_retry(
            client.delete,
            collection_name=col,
            points_selector=qdrant_models.PointIdsList(points=ids),
        )

    return (
        f"Deleted {len(to_delete)} compact(s) for project '{project}' "
        f"({mode_desc}), retained {kept_count}:\n"
        f"{summary_text}"
    )


def _require_savings_tracking() -> Optional[str]:
    if not _SAVINGS_LEDGER_AVAILABLE:
        return "Error: libs/savings_ledger.py is not available in this install -- the savings tracker requires it."
    if not savings_ledger.tracking_enabled():
        return (
            "Savings tracking is off (CLAUDE_RUNWAY_TRACK_SAVINGS is not set). "
            "Set CLAUDE_RUNWAY_TRACK_SAVINGS=1 in this server's env (in .mcp.json) and export it in the shell "
            "environment that launches `claude` too (hook entries in .claude/settings.json have no env field "
            "of their own, so that's the only way compress_bash_output.py sees it)."
        )
    return None


@mcp.tool()
def savings_summary(project: Optional[str] = None, format: str = "text") -> str:
    """
    SIMPLE view of the opt-in savings tracker: this session's estimated
    context tokens avoided by local compression, plus this project's
    all-time totals. Called by /my-savings with no argument.

    This is an ONLINE ESTIMATE of a counterfactual (what raw content would
    have cost vs. the compressed result actually used), not a controlled
    A/B measurement -- see EVALUATION.md's Track D for that distinction.
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
        session_agg = (
            savings_ledger.get_live_session_aggregate(session_id) if session_id
            else {"credited_saved_tokens": 0, "raw_tokens_sum": 0, "out_tokens_sum": 0, "event_count": 0, "fetch_url_count": 0, "overhead_tokens": savings_ledger.get_schema_overhead_tokens()}
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
    breakdown, a sparkline of the last 10 sessions for this project, and a
    cross-project comparison of total tokens avoided. Called by
    `/my-savings detail`. See savings_summary's docstring for the same
    counterfactual-estimate caveat.

    project defaults to the current working directory's folder name.

    format controls the output format:
      "text" (default) — human-readable text, same as before.
      "json"           — JSON object with all five data blocks.
      "csv"            — CSV whose content depends on csv_table (see below).

    csv_table selects which table to export when format="csv":
      "session"  (default) — one-row session summary.
      "tools"              — per-tool breakdown rows.
      "projects"           — cross-project all-time rows.
    """
    err = _require_savings_tracking()
    if err:
        return err
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
        if session_id:
            session_agg = savings_ledger.get_live_session_aggregate(session_id)
            # Sorted by saved_tokens descending -- by_tool.items() is otherwise in
            # dict insertion order (first-tool-seen order), which would make the
            # ordering inconsistent with query_session_tool_breakdown's explicit
            # "ORDER BY saved_tokens DESC" for a finalized (post-SessionEnd) session.
            tool_breakdown = sorted(
                (
                    {
                        "tool": t,
                        "event_count": a["event_count"],
                        "saved_tokens": a["saved_tokens"],
                        # raw/out/credited_event_count feed avg % reduction and
                        # credited-vs-uncredited detection in format_detail_view --
                        # all three must be forwarded here or the live (pre-SessionEnd)
                        # path silently suppresses the metric.
                        "raw_tokens_sum": a.get("raw_tokens_sum", 0),
                        "out_tokens_sum": a.get("out_tokens_sum", 0),
                        "credited_event_count": a.get("credited_event_count", 0),
                    }
                    for t, a in session_agg.get("by_tool", {}).items()
                ),
                key=lambda t: t["saved_tokens"],
                reverse=True,
            )
        else:
            session_agg = {"credited_saved_tokens": 0, "raw_tokens_sum": 0, "out_tokens_sum": 0, "event_count": 0, "fetch_url_count": 0, "overhead_tokens": savings_ledger.get_schema_overhead_tokens()}
            tool_breakdown = []
        session_agg["project"] = project
        project_summary = savings_ledger.query_project_summary(project)
        last_n = savings_ledger.query_last_n_sessions(project, n=10)
        all_projects = savings_ledger.query_all_projects()
        if format == "json":
            return savings_ledger.format_json(session_agg, project_summary, tool_breakdown, last_n, all_projects)
        if format == "csv":
            return savings_ledger.format_csv(session_agg, project_summary, tool_breakdown, last_n, all_projects, table=csv_table)
        return savings_ledger.format_detail_view(session_agg, project_summary, tool_breakdown, last_n, all_projects)
    except Exception as e:
        # Same reasoning as savings_summary's try/except -- best-effort feature,
        # must not fail the tool call over a DB problem.
        return f"Error: could not read savings data ({e}). The savings database may be corrupted or inaccessible; this doesn't affect anything else."


@mcp.tool()
def savings_trend(project: Optional[str] = None, bucket: str = "week", n: int = 12) -> str:
    """
    TREND view of the opt-in savings tracker: tokens avoided aggregated by
    day or week rather than individual sessions, so patterns are visible even
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
    metadata: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Writes one append-only event to the shared cross-tool metrics store
    (issue #210 — the write counterpart to get_metrics). Any domain in
    this toolkit can call this to record an event; the primary consumer
    as of this ticket is my-gh-autowork's per-ticket outcome logging
    (cut over from compact_store by this ticket).

    metric_id: the domain (e.g. "autowork", "memory-bank") — caller-defined,
      no fixed enum.
    event_type: domain-defined event kind (e.g. "ticket_merged", "ticket_blocked").
    value: a signed numeric delta (default 1.0). Must be finite — non-finite
      values (inf/-inf/nan) are rejected by MetricsStore.record() and logged to
      stderr; this tool surfaces that as an "Error:..." return instead.
    metadata: optional JSON string of domain-specific extra fields (e.g.
      '{"issue": 42, "pr": 101, "rounds": 2}'). Must be valid JSON if supplied;
      an invalid JSON string is returned as an "Error:..." string rather than
      raising.
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
    import json as _json
    import math as _math
    if not _math.isfinite(value):
        return f"Error: value must be finite (got {value!r})."
    metadata_dict: Optional[dict] = None
    if metadata is not None:
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
    _log_fixed_overhead()
    mcp.run()
