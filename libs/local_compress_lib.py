"""
Shared, MCP-independent compression logic used by both:
  - compress_mcp_server.py     (MCP server exposing compress_file/compress_command_output/etc.)
  - hooks/compress_bash_output.py  (PostToolUse hook that compresses ANY large Bash output)

Kept free of the `mcp` package dependency on purpose -- the hook script only
needs `openai`, not the full MCP SDK, since it's invoked as a plain command
hook, not an MCP server. ctx is accepted as a loosely-typed optional object
with an async report_progress(progress, total, message) method; pass None
when there's no MCP Context available (e.g. from the hook).
"""

import os
from typing import Optional

from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = os.environ.get("LMSTUDIO_MODEL")  # intentionally no fallback -- see compress_mcp_server.py NOTE 2
DEFAULT_CHUNK_CHARS = 12_000  # conservative per-call size -- local models vary widely in context window

DEFAULT_FOCUS = (
    "Summarize the key information. Preserve anything that looks like an "
    "error, failure, stack trace location, or actionable detail. Drop "
    "repetitive or boilerplate lines."
)

NOT_RELEVANT_MARKER = "[NOT RELEVANT]"

# Keyword-based, deterministic detection for positional focuses ("the lead
# section," "the introduction"). This is a step down in elegance from
# letting the model reason about chunk position on its own (classify_relevant
# below does receive chunk_index/chunk_count and IS instructed to weigh
# position), but real testing showed that isn't reliable: against a real
# Wikipedia article and a real local model, a focus of "summarize the lead
# section" still returned a detailed digest of the article's Techniques and
# Applications/Regulation sections (search algorithms, Bayesian networks,
# SVMs, RLHF, legal frameworks) -- clearly not the lead, which is a short,
# non-technical overview. The model was pattern-matching on "sounds like a
# broad summary" rather than actually using the "you are chunk 4 of 16"
# context it was given. Since that failure mode depends on a specific
# model's instruction-following quality and can't be fixed by prompting
# alone, this keyword check guarantees correctness for phrasings we can
# recognize with certainty, by bypassing the classifier for them entirely
# rather than hoping it uses position correctly. The position-aware
# classifier remains in place as a secondary aid for positional phrasings
# this list doesn't happen to catch.
_POSITIONAL_FOCUS_HINTS = (
    "lead section", "lede", "introduction", "intro paragraph", "intro section",
    "opening paragraph", "opening section", "beginning of", "first paragraph",
    "first section", "first few paragraphs", "abstract", "tl;dr", "tldr",
    "top of the page", "top of the document", "start of the",
)


def _looks_positional(focus: str) -> bool:
    lowered = focus.lower()
    return any(hint in lowered for hint in _POSITIONAL_FOCUS_HINTS)


def _find_first_heading_boundary(text: str, min_offset: int = 300, max_offset: int = 20_000) -> Optional[int]:
    """
    Heuristic boundary detection for "the lead section" and similar
    positional asks: find the char offset of the first apparent section
    heading and truncate there, instead of guessing a fixed length.

    Confirmed necessary in testing, not just theoretically nicer: a fixed
    _AUTO_POSITIONAL_CHARS window (tried at both 12_000 and 4000) reliably
    excluded distant content, but still bled past the ACTUAL lead section
    into the next section(s) -- e.g. Wikipedia's "Goals" section and the
    start of "Reasoning and problem-solving" got pulled in alongside a
    ~1700-char real lead, because 4000 chars simply isn't how long that
    particular article's lead happens to be. Lead-section length varies a
    lot by document; a fixed count can't track that, but a structural
    signal can.

    The heuristic: text extracted by trafilatura (and many plain-text/
    markdown-ish documents generally) renders section headings as short
    standalone lines sitting between longer prose paragraphs -- no
    sentence-ending punctuation, flanked by substantial text above and
    below. A line matching that pattern is a much more reliable "this is
    where the next section starts" signal than any guessed character count.

    Returns None if no confident heading-like line is found within
    max_offset chars (callers should fall back to a fixed default in that
    case -- this heuristic won't fire for content with no heading structure
    at all, e.g. a plain log file, which is the correct behavior since
    there's no boundary to detect there).
    """
    lines = text.split("\n")
    offset = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if offset > max_offset:
            return None
        if offset >= min_offset and stripped and len(stripped) < 60:
            looks_like_heading = not stripped.endswith((".", "!", "?", ":", ")", "]", ","))
            prev_is_prose = i > 0 and len(lines[i - 1].strip()) > 80
            next_is_prose = i + 1 < len(lines) and len(lines[i + 1].strip()) > 80
            if looks_like_heading and prev_is_prose and next_is_prose:
                return offset
        offset += len(line) + 1  # +1 for the newline chunk_text/split doesn't preserve
    return None


# Used when a positional focus is auto-detected and the caller didn't
# already pass an explicit max_chars -- a reasonable size for "the
# beginning" of most documents without needing to know the document's
# actual length in advance.
_AUTO_POSITIONAL_CHARS = 4000

# Applied to each individual chunk (the "map" step), AFTER the dedicated
# relevance classifier below has already confirmed the chunk is worth
# extracting from. Kept as a second line of defense (belt-and-suspenders,
# not the primary mechanism) -- see _classify_relevant's docstring for why
# the classifier, not this marker, is what's actually relied on now.
_MAP_ANTI_CHAT_INSTRUCTION = (
    "Respond ONLY with the compressed content itself -- no greetings, no "
    "meta-commentary, no questions, and never respond as if you are having "
    "a conversation. If this specific chunk has nothing relevant to the "
    f"focus below, respond with exactly this and nothing else: {NOT_RELEVANT_MARKER}"
)

# Applied to the final reduce step, which only ever sees already-filtered,
# already-relevant chunk summaries -- so no NOT_RELEVANT escape hatch here,
# just the same anti-chat instruction (small models can drift into chatty
# responses even without a relevance mismatch).
_REDUCE_ANTI_CHAT_INSTRUCTION = (
    "Respond ONLY with the combined summary itself -- no greetings, no "
    "meta-commentary, no questions, and never respond as if you are having "
    "a conversation."
)


def client(base_url: Optional[str]) -> OpenAI:
    return OpenAI(base_url=base_url or DEFAULT_BASE_URL, api_key="lm-studio")


def chunk_text(text: str, chunk_chars: int):
    return [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)] or [""]


def complete(oai_client: OpenAI, model: str, system: str, content: str) -> Optional[str]:
    """Returns the completion text, or None if the request failed for any reason
    (connection error, model not found, etc.) -- callers must check for None
    rather than assume this always succeeds just because resolve_model did.
    resolve_model only validates connectivity when it has to auto-detect; an
    explicit model= or LMSTUDIO_MODEL skips that check entirely, so a dead
    LM Studio server is only caught here, at the actual request."""
    try:
        response = oai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0,
        )
    except Exception:
        return None
    return response.choices[0].message.content or ""


def classify_relevant(
    oai_client: OpenAI, model: str, focus: str, chunk: str, chunk_index: int = 1, chunk_count: int = 1,
    truncated: bool = False,
) -> Optional[bool]:
    """
    Dedicated relevance classifier, run BEFORE extraction on every chunk.

    Earlier design relied on a single combined call per chunk: "extract what
    matches this focus, or respond with a NOT_RELEVANT marker if nothing
    does." In testing (fetch_url against a long Wikipedia article with a
    narrow focus like "summarize the lead section"), models often described
    irrelevance in free prose instead of the exact marker -- e.g. "this
    chunk doesn't mention X, it discusses Y instead" -- which didn't match
    the marker check and silently slipped through as if it were a real
    summary. The final reduce step happened to still produce a correct
    answer by synthesizing across several similar "not found here" chunk
    summaries, but only by accident -- the filtering itself wasn't working.

    A dedicated classification call with a constrained one-word answer
    (YES/NO) is far more reliable to parse than hoping free-form extraction
    output either contains an exact marker or doesn't. This doubles the
    number of local model calls per chunk (one to classify, one to extract),
    but that's free against the actual budget this project cares about
    (Claude tokens) -- same reasoning as chunking instead of truncating
    (see compress_mcp_server.py's RESOLVED note #3).

    Returns True/False, or None if the request itself failed -- callers
    should treat None as a hard error (same as any other local model
    failure in this file), not a silent skip, so a flaky LM Studio server
    doesn't quietly drop content instead of surfacing the problem.

    Prompt is deliberately biased toward NO. Testing (fetch_url against a
    Wikipedia article with a focus of "chocolate cake recipes" -- an article
    that mentions no such thing) still got 2 of 16 chunks classified as
    relevant. The final answer was still correct (the extraction + reduce
    steps recovered by synthesizing "this isn't covered" from the 2
    generic, off-focus chunk summaries that leaked through), so this wasn't
    a user-visible failure, but it means the classifier defaults toward
    "yes" under uncertainty -- a common bias in small chat models being
    asked a yes/no question. Explicitly instructing it to default to NO
    unless the match is clear and direct counteracts that bias.

    chunk_index/chunk_count give the classifier POSITIONAL awareness, not
    just content. This matters for asks like "the lead section" or "the
    introduction," which aren't about content TOPIC at all -- every chunk of
    a document about subject X is topically "about X," including chunks
    nowhere near the actual start. Relying on a caller to separately notice
    "this is a positional ask" and pass a different parameter (an earlier
    version of this tool required explicitly passing max_chars for these
    cases) doesn't hold up in practice -- it's an extra thing to remember on
    every call, and got flagged as impractical. Telling the classifier its
    own position in the document lets it reason about positional asks
    automatically in the same single YES/NO pass used for topical ones, with
    no special-casing required from the caller. max_chars (see
    compress_mcp_server.py) still exists as an explicit override for anyone
    who wants a hard guarantee and to skip the classifier's judgment call
    entirely, but it's no longer the only way to get positional asks right.

    `truncated` matters for a real bug found in testing: when a positional
    focus triggers compress()'s auto-truncation (see _looks_positional) and
    the truncated slice happens to fit in a single chunk, the OLD position
    note said "this is the entire document" -- which is false, and actively
    misleading for a focus like "the introductory paragraphs BEFORE THE
    TABLE OF CONTENTS": a model told "this is everything, nothing comes
    after" has no consistent way to agree that what it's looking at comes
    "before" something else, since it was just told there IS no "something
    else." Confirmed as the actual cause of a real failure: the classifier
    returned NO for a single truncated chunk that demonstrably WAS the real
    lead section, on a document deliberately truncated for exactly that
    focus. `truncated=True` fixes the framing to honestly describe a
    beginning-of-a-larger-document slice instead.
    """
    if chunk_count > 1:
        position_note = (
            f"This is chunk {chunk_index} of {chunk_count} from a larger document, in the "
            "same order as the original (chunk 1 is the very beginning, the last chunk is the "
            "very end) -- use this position if the focus asks about a specific part of the "
            "document (e.g. \"the introduction\" means early chunks, \"the conclusion\" means "
            "late chunks)."
        )
    elif truncated:
        position_note = (
            "This is the BEGINNING portion of a larger document -- the rest was intentionally "
            "left out for this request, not because the document ends here. Treat this as "
            "consistent with the document having more content after what you're seeing."
        )
    else:
        position_note = "This is the entire document (not split into multiple chunks)."
    answer = complete(
        oai_client, model,
        "You are a strict relevance classifier. Most chunks of a large "
        "document will NOT be relevant to any given focus -- when in doubt, "
        "answer NO. Only answer YES if this specific chunk directly and "
        "explicitly contains information matching the focus (considering "
        "both its content AND its position in the document, when the focus "
        "implies a position), not just a loosely related topic. Respond "
        "with EXACTLY one word and nothing else: YES or NO.",
        f"Focus: {focus!r}\n\n{position_note}\n\nDoes this chunk directly and explicitly "
        f"contain information matching that focus?\n\n---\n\n{chunk}",
    )
    if answer is None:
        return None
    normalized = answer.strip().upper()
    if normalized.startswith("YES"):
        return True
    if normalized.startswith("NO"):
        return False
    # Unparseable response (rare, but small local models can be inconsistent
    # about following the one-word-answer instruction exactly) -- fail safe
    # by treating it as relevant rather than silently dropping real content
    # over a formatting hiccup. Worst case this costs one wasted extraction
    # call, which is still free against the actual budget.
    return True


def resolve_model(explicit_model: Optional[str], base_url: Optional[str]):
    """
    Returns (model_id, error_message). Exactly one will be None.
    Resolution order: explicit param > LMSTUDIO_MODEL env var > auto-detect
    (only when exactly one model is currently loaded in LM Studio).
    """
    if explicit_model:
        return explicit_model, None
    if DEFAULT_MODEL:
        return DEFAULT_MODEL, None

    try:
        models = client(base_url).models.list()
    except Exception as e:
        return None, f"Could not reach LM Studio at {base_url or DEFAULT_BASE_URL} to auto-detect a model: {e}"

    ids = [m.id for m in models.data]
    if len(ids) == 1:
        return ids[0], None
    if not ids:
        return None, "LM Studio is reachable but no model is loaded. Load one in LM Studio, or pass model= explicitly."
    return None, (
        f"Multiple models are loaded ({', '.join(ids)}) -- can't auto-detect which one to use. "
        "Pass model= explicitly, or set LMSTUDIO_MODEL to pin a default."
    )


async def compress(
    text: str,
    focus: str = DEFAULT_FOCUS,
    skip_if_under_chars: int = 2000,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_total_chars: int = 2_000_000,
    max_chars: Optional[int] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    ctx=None,
) -> str:
    """
    Shared map-reduce compression logic. Returns the ORIGINAL text unchanged
    if it's under skip_if_under_chars, or an "Error: ..." string (never
    raises) if compression can't proceed -- callers should treat a return
    value starting with "Error:" as a signal to fall back to the original
    text rather than lose it.

    max_chars, if given, truncates to the first N chars of `text` before
    anything else happens. If not given but `focus` looks positional (see
    _looks_positional/_POSITIONAL_FOCUS_HINTS), it's auto-applied at
    _AUTO_POSITIONAL_CHARS -- this makes positional asks ("summarize the
    lead section") correct automatically, without requiring the caller to
    separately recognize "this needs max_chars" and pass it explicitly
    (that requirement was tried first and got flagged as impractical -- see
    compress_mcp_server.py's fetch_url docstring history). This is a
    deterministic shortcut, not a judgment call left to the model: real
    testing showed a position-aware classifier (still used below for
    phrasings this keyword list doesn't catch) isn't reliably followed by
    every local model -- one real test still returned a technically-detailed
    digest of a Techniques/Applications section mislabeled as "the lead
    section," because the model pattern-matched on tone rather than using
    the chunk-position context it was given.
    """
    auto_truncated = False
    if max_chars is None and _looks_positional(focus):
        # Prefer a structural boundary (first apparent section heading) over
        # a fixed character count -- confirmed necessary in testing, not
        # just theoretically nicer: even a corrected fixed window (min(),
        # not max(), of chunk_chars/_AUTO_POSITIONAL_CHARS -- an earlier bug
        # here used max(), which always evaluated to chunk_chars=12_000 and
        # made _AUTO_POSITIONAL_CHARS dead code) still bled past a real
        # ~1700-char Wikipedia lead into its "Goals" and "Techniques"
        # sections, because a fixed guess can't track how long any given
        # document's actual lead happens to be. Falls back to the fixed
        # window only when no confident heading-like boundary is found
        # (e.g. plain log/command output with no heading structure at all).
        boundary = _find_first_heading_boundary(text)
        max_chars = boundary if boundary is not None else min(chunk_chars, _AUTO_POSITIONAL_CHARS)
        auto_truncated = True

    # Whether `text` has been deliberately scoped down to a specific window
    # via max_chars, whether the caller passed it explicitly or it was just
    # auto-detected above. Once true, classify_relevant is skipped entirely
    # for every resulting chunk -- see the classification call site for why:
    # real testing showed the classifier actively working AGAINST
    # already-correct scoping. Two real failures confirmed this: with the
    # boundary-detection fix correctly finding the true ~2434-char lead, the
    # classifier STILL rejected it (a chunk we'd already deterministically
    # confirmed was exactly right). And when a caller widened max_chars to
    # 20_000 chars (crossing into the article's Techniques section), the
    # classifier let through a chunk full of SVMs/CNNs/Bayesian-network
    # detail while rejecting the chunk that actually contained the real
    # lead. Once max_chars has scoped what to look at, asking the
    # classifier to re-judge relevance within that window adds pure risk of
    # it vetoing content already known to be correct, with no upside --
    # there's nothing left to filter for once the window is deliberately
    # scoped.
    chars_limited = max_chars is not None
    if max_chars is not None:
        text = text[:max_chars]

    if len(text) < skip_if_under_chars:
        return text

    if len(text) > max_total_chars:
        return (
            f"Error: input is {len(text)} chars, over the max_total_chars safety limit "
            f"({max_total_chars}). Refusing rather than silently dropping content -- "
            "pre-filter the input yourself or raise max_total_chars if you're sure."
        )

    model, error = resolve_model(model, base_url)
    if error:
        return f"Error: {error}"

    original_len = len(text)
    oai_client = client(base_url)
    chunks = chunk_text(text, chunk_chars)

    chunk_summaries = []
    skipped = 0
    for i, chunk in enumerate(chunks, 1):
        if chars_limited:
            # Already deliberately scoped via max_chars -- don't ask the
            # classifier to re-judge relevance within a window we've
            # already decided is the right one to look at.
            relevant = True
        else:
            if ctx is not None:
                await ctx.report_progress(progress=i - 1, total=len(chunks), message=f"Checking relevance {i}/{len(chunks)}")
            relevant = classify_relevant(
                oai_client, model, focus, chunk, chunk_index=i, chunk_count=len(chunks), truncated=auto_truncated,
            )
        if relevant is None:
            return f"Error: LM Studio request failed while checking relevance (chunk {i}/{len(chunks)}) -- check it's still running at {base_url or DEFAULT_BASE_URL}."
        if relevant is False:
            skipped += 1
            continue

        if ctx is not None:
            await ctx.report_progress(progress=i - 1, total=len(chunks), message=f"Compressing chunk {i}/{len(chunks)}")
        summary = complete(
            oai_client, model,
            f"You compress text for another AI to read next. {_MAP_ANTI_CHAT_INSTRUCTION}\n\nFocus: {focus}",
            chunk,
        )
        if summary is None:
            return f"Error: LM Studio request failed (chunk {i}/{len(chunks)}) -- check it's still running at {base_url or DEFAULT_BASE_URL}."
        summary = summary.strip()
        # Belt-and-suspenders: even a chunk the classifier called relevant
        # can still come back marked NOT_RELEVANT from the extraction call
        # itself (the two calls can disagree) -- honor that if it happens.
        if summary.upper() == NOT_RELEVANT_MARKER.upper() or (
            len(summary) < len(NOT_RELEVANT_MARKER) + 20 and "NOT RELEVANT" in summary.upper()
        ):
            skipped += 1
            continue
        chunk_summaries.append(summary)

    if not chunk_summaries:
        truncated_note = (
            f" NOTE: focus looked positional, so only the first {original_len} chars of the "
            "source were considered (auto-truncated) -- this error means nothing relevant was "
            "found WITHIN that truncated portion, not that the whole document was searched. "
            "Pass a larger max_chars explicitly if the target content might be further in."
            if auto_truncated else ""
        )
        return (
            f"Error: none of the {len(chunks)} chunk(s) contained content relevant to the "
            f"requested focus ({focus!r}).{truncated_note} The page/content may genuinely not "
            "cover this, or the relevant part may have been split awkwardly across chunks -- "
            "try a broader focus, or a smaller chunk_chars if the target content is short and "
            "near a chunk boundary."
        )

    if len(chunk_summaries) == 1:
        compressed = chunk_summaries[0]
    else:
        if ctx is not None:
            await ctx.report_progress(progress=len(chunks), total=len(chunks) + 1, message="Combining chunk summaries...")
        combined = "\n\n".join(f"[part {i + 1}/{len(chunk_summaries)}]\n{s}" for i, s in enumerate(chunk_summaries))
        compressed = complete(
            oai_client, model,
            "You are combining partial summaries of one larger document into a single coherent "
            f"summary. {_REDUCE_ANTI_CHAT_INSTRUCTION}\n\nFocus: {focus}",
            combined,
        )
        if compressed is None:
            return f"Error: LM Studio request failed while combining chunk summaries -- check it's still running at {base_url or DEFAULT_BASE_URL}."
        compressed = compressed.strip()

    if ctx is not None:
        await ctx.report_progress(progress=1, total=1, message="Done")

    ratio = round(100 * (1 - len(compressed) / original_len)) if original_len else 0
    skip_note = f", {skipped} chunk(s) skipped as not relevant to the focus" if skipped else ""
    truncate_note = (
        f" [note: focus looked positional, so only the first {original_len} chars of the "
        "source were considered -- pass max_chars explicitly to override this]"
        if auto_truncated else ""
    )
    return f"[compressed {original_len} -> {len(compressed)} chars across {len(chunks)} chunk(s){skip_note}, ~{ratio}% smaller]{truncate_note}\n\n{compressed}"
