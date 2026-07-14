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
             "args": ["/absolute/path/to/compress_mcp_server.py"],
             "env": {
               "LMSTUDIO_BASE_URL": "http://localhost:1234/v1",
               "LMSTUDIO_MODEL": "<exact model id loaded in LM Studio>"
             }
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
     LMSTUDIO_MODEL env var, then auto-detect IF exactly one model is
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

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs"))

import requests
import trafilatura
from mcp.server.fastmcp import FastMCP, Context

from local_compress_lib import (
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_FOCUS,
    client as _client,
    compress as _compress_impl,
)

DEFAULT_WEB_FOCUS = (
    "Extract and summarize the information on this page most relevant to "
    "what was asked. Preserve specific facts, numbers, names, dates, and "
    "technical details verbatim where possible. Drop navigation, ads, and "
    "boilerplate."
)

mcp = FastMCP("local-compress")


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
    return await _compress_impl(text, focus, skip_if_under_chars, chunk_chars, max_total_chars, max_chars, model, base_url, ctx)


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
    ctx: Context = None,
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
    text = path.read_text(errors="ignore")
    return await _compress(text, focus, skip_if_under_chars, chunk_chars, max_total_chars, model, base_url, ctx, max_chars)


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
    ctx: Context = None,
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
    result = await _compress(combined, focus, skip_if_under_chars, chunk_chars, max_total_chars, model, base_url, ctx, max_chars)
    return f"[exit code {proc.returncode}]\n{result}"


@mcp.tool()
async def fetch_url(
    url: str,
    focus: str = DEFAULT_WEB_FOCUS,
    skip_if_under_chars: int = 2000,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_total_chars: int = 2_000_000,
    max_chars: Optional[int] = None,
    timeout_seconds: int = 30,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    ctx: Context = None,
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
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": "Mozilla/5.0 (compatible; local-compress-fetch/1.0)"},
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching {url}: {e}"

    text = trafilatura.extract(resp.text, url=url) or ""
    if not text.strip():
        # trafilatura found no "article-like" main content (common on
        # non-article pages) -- fall back to a crude tag strip rather than
        # returning nothing.
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return f"Error: no extractable text content found at {url}."

    result = await _compress(text, focus, skip_if_under_chars, chunk_chars, max_total_chars, model, base_url, ctx, max_chars)
    return f"[fetched {url}, {len(resp.content)} raw bytes]\n{result}"


@mcp.tool()
async def compress_text(
    text: str,
    focus: str = DEFAULT_FOCUS,
    skip_if_under_chars: int = 2000,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_total_chars: int = 2_000_000,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Compress text you already hold (e.g. your own long draft) via a local
    model. IMPORTANT: this does NOT save Claude tokens if you had to read
    something into context just to call this -- by that point you've
    already paid the input cost to read it, and pay output tokens again to
    pass it as this argument. For files or command output, use compress_file
    or compress_command_output instead, which read the source themselves so
    the raw content never has to pass through you at all.
    """
    return await _compress(text, focus, skip_if_under_chars, chunk_chars, max_total_chars, model, base_url, ctx)


if __name__ == "__main__":
    mcp.run()
