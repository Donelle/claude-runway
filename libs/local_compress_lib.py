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

import asyncio
import hashlib
import os
import re
from typing import Optional

from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("CLAUDE_RUNWAY_LMSTUDIO_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = os.environ.get("CLAUDE_RUNWAY_LMSTUDIO_MODEL")  # intentionally no fallback -- see compress_mcp_server.py NOTE 2
DEFAULT_CHUNK_CHARS = 12_000  # conservative per-call size -- local models vary widely in context window

# Renamed from the bare LMSTUDIO_*/HOOK_* names to a CLAUDE_RUNWAY_* namespace.
# The rename matters because these have to be exported at the OS/shell level for
# the hook scripts to see them at all (hook entries in .claude/settings.json have
# no `env` field, unlike MCP servers in .mcp.json) -- and an unnamespaced
# LMSTUDIO_MODEL sitting in a global shell profile is both easy to collide with
# and impossible to attribute to this toolkit.
#
# The rename is a hard cutover: the old names are NOT read as a fallback. That
# would leave two live name sets forever. But an unmigrated config must not fail
# SILENTLY -- the bug that motivated all of this was LMSTUDIO_MODEL being set in
# .mcp.json where the hook couldn't read it, which produced a confusing
# "Multiple models are loaded, can't auto-detect" error rather than anything
# pointing at the real cause. So stale_env_warning() detects "old name set, new
# name missing" and callers surface it: resolve_model turns it into a hard error,
# and the hooks attach it via additionalContext (they must fail open).
# The third element is where the NEW name actually has to be set, which is not
# uniform and must not be described as if it were: CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS
# is read ONLY by hooks/compress_bash_output.py, so an .mcp.json entry for it is
# inert and telling someone to add one sends them to the wrong file. The other two
# are read via this module, which BOTH the MCP server and the compress hook import.
_BOTH = ("set it in both .mcp.json's env block (for the MCP server) and as a shell "
         "export (for the hooks), with matching values")
_SHELL_ONLY = ("set it as a shell export only -- it's read solely by the compress hook, "
               "so an .mcp.json entry for it has no effect")
_RENAMED_ENV_VARS = (
    ("LMSTUDIO_BASE_URL", "CLAUDE_RUNWAY_LMSTUDIO_URL", _BOTH),
    ("LMSTUDIO_MODEL", "CLAUDE_RUNWAY_LMSTUDIO_MODEL", _BOTH),
    ("HOOK_COMPRESS_THRESHOLD_CHARS", "CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS", _SHELL_ONLY),
)


def stale_env_warning() -> Optional[str]:
    """
    Returns a human-readable warning if any pre-rename env var is set while its
    new counterpart is missing -- i.e. a config that was never migrated and is
    now being silently ignored. Returns None when there's nothing to report.

    Deliberately only fires when the NEW name is absent: if both are set, the new
    one wins and the leftover old one is harmless noise, not a misconfiguration
    worth interrupting anyone over.

    The "where to set it" advice is per-variable rather than one blanket sentence,
    because it genuinely differs -- see _RENAMED_ENV_VARS.
    """
    stale = [
        f"{old} is set but no longer read -- rename it to {new} ({where})"
        for old, new, where in _RENAMED_ENV_VARS
        if os.environ.get(old) and not os.environ.get(new)
    ]
    if not stale:
        return None
    return (
        "ClaudeRunway env vars were renamed to a CLAUDE_RUNWAY_* namespace: "
        + "; ".join(stale)
        + ". See the README's 'Environment variables' section."
    )

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


# Non-selective focuses: "compress all of this," not "find the part about X."
#
# classify_relevant exists to FILTER -- to drop chunks that don't contain the
# specific thing the caller asked for. That premise only holds when the focus
# names a target. When the focus is "summarize this," every chunk is in scope
# by definition, there is nothing to filter, and running a NO-biased
# classifier over it can only subtract.
#
# This isn't hypothetical. DEFAULT_FOCUS reads as an instruction, but its most
# concrete clause ("preserve anything that looks like an error, failure, stack
# trace location") reads to a strict classifier as the selective criterion --
# so it answers the question "does this chunk contain an error?" and ignores
# the "Summarize the key information" clause entirely. Measured against a real
# local model (gemma-4-e2b, temperature=0, 8 trials each): a clean 60-line
# restore log classified NO 8/8, a 5000-char README chunk NO 8/8. Appending a
# single `error CS0246:` line to that same clean log flipped it to YES. The
# practical effect was that the PostToolUse hook only ever compressed FAILING
# output -- successful builds, passing test runs, docs, and diffs all fell
# through to "none of the chunk(s) contained content relevant to the requested
# focus" and were returned raw, after paying for the round-trip. That is the
# common case, not the edge case.
#
# Same reasoning as the chars_limited bypass in compress(): once we know
# there's nothing to filter, asking the classifier to re-judge adds pure
# downside risk over content already known to be in scope. Kept as a
# deterministic keyword check rather than a smarter prompt for the same
# reason _POSITIONAL_FOCUS_HINTS is -- this failure mode depends on a given
# model's instruction-following and can't be reliably prompted away.
# Matched against the WHOLE normalized focus, never as a substring. Substring
# matching was tried first and is wrong in the one direction that matters: a
# generic phrase is almost always a prefix of a perfectly selective request.
# "summarize the key information about auth", "what happened with the retry
# logic", "the gist of the auth module" and "did it fail to connect to redis"
# all contain a phrase from this set while clearly naming a target -- and
# skipping classification for them would silently stop filtering, handing back
# a summary of the whole document instead of the part that was asked for. A
# selective focus that gets treated as non-selective fails quietly and looks
# like a correct answer, which is the same property that made the bug this
# whole predicate exists to fix so hard to notice.
#
# Note "tl;dr" is deliberately absent: _POSITIONAL_FOCUS_HINTS already claims
# it, and a positional ask is selective (it names a part) -- it needs the
# auto-truncation path, not this bypass.
_GENERIC_FOCUS_PHRASES = frozenset({
    "summarize", "summarise", "summary",
    "summarize this", "summarise this", "summarize it", "summarize the output",
    "summarize the key information", "key information",
    "the gist", "give me the gist",
    "overall summary", "general summary",
    "what happened", "did it succeed", "did it fail", "pass or fail",
})


def _normalize_focus(focus: str) -> str:
    """Lowercase, collapse whitespace, drop surrounding punctuation and any
    leading politeness/imperative filler, so "Please summarize this." and
    "  summarize this  " both reduce to "summarize this".

    The filler group is wrapped in a repeating `(?:...)+` rather than
    applied once, so a DOUBLED prefix like "Just please summarize this" or
    "Please just summarize this" has every leading filler word stripped,
    not just the first -- a single non-repeating re.sub left one layer
    behind ("please summarize this"), which then failed to match
    DEFAULT_FOCUS/_GENERIC_FOCUS_PHRASES in _looks_non_selective and wrongly
    ran the per-chunk classifier against the leftover fragment as if it
    were a real selection target (issue #42)."""
    normalized = re.sub(r"\s+", " ", focus.strip().lower()).strip(" .!?")
    return re.sub(r"^(?:(?:please|just|can you|could you)\s+)+", "", normalized).strip()


def _looks_non_selective(focus: str) -> bool:
    """True if `focus` asks for the whole thing compressed rather than for a
    specific part of it -- in which case classification is skipped entirely.

    DEFAULT_FOCUS is matched explicitly rather than by phrase, since it's the
    value every caller gets when they pass no focus at all, it's by
    construction non-selective, and it's the only focus the PostToolUse hook
    ever sends -- so the hook path stays covered regardless of how
    conservative the phrase set below is."""
    if not focus or not focus.strip():
        return True
    normalized = _normalize_focus(focus)
    if normalized == _normalize_focus(DEFAULT_FOCUS):
        return True
    return normalized in _GENERIC_FOCUS_PHRASES


def _nearest_nonblank_stripped(lines: list, start: int, step: int) -> Optional[str]:
    """
    Walks from `start` in `step` direction (-1 or +1), skipping blank lines,
    and returns the first non-blank line's stripped text -- or None if the
    walk runs off the end without finding one.

    Needed because a heading is normally separated from surrounding prose by
    a blank line (standard Markdown, including this repo's own README) --
    checking only the line immediately adjacent to a candidate heading missed
    that far more often than intended (issue #28).
    """
    j = start
    while 0 <= j < len(lines) and not lines[j].strip():
        j += step
    return lines[j].strip() if 0 <= j < len(lines) else None


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
    The prose lines don't have to be immediately adjacent -- a blank line
    (or several) in between is the normal case, not the exception, so the
    nearest NON-BLANK line on each side is what actually gets checked
    (issue #28; see _nearest_nonblank_stripped).

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
            prev_line = _nearest_nonblank_stripped(lines, i - 1, -1)
            next_line = _nearest_nonblank_stripped(lines, i + 1, 1)
            prev_is_prose = prev_line is not None and len(prev_line) > 80
            next_is_prose = next_line is not None and len(next_line) > 80
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


# Section handling for structured input (see compress(preserve_sections=True)).
#
# /my-compact fills in a fixed six-heading template, so the section boundaries
# here are AUTHORED, not inferred -- which is what makes this a guarantee rather
# than another heuristic. The identifier guard had to guess what an identifier
# looks like; there is nothing to guess about a heading we wrote ourselves.
#
# Fence tracking follows the same CommonMark rules as the doc chunker, for the
# same reason: a '## ' line inside a fenced sample is not a section boundary.
_SECTION_HEADING_RE = re.compile(r"^ {0,3}##\s+\S")
_SECTION_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")

# Sections whose value is exactness rather than prose, passed through untouched.
# "Important files and locations" is a list of identifiers by definition, and
# the other two are short lists where every line is load-bearing -- compressing
# them is all risk and no meaningful saving. The prose wins live in the other
# three sections ("What we were working on", "Key decisions made", "Current
# state / progress"), which is where compression is applied.
_VERBATIM_SECTION_HINTS = (
    "important file", "unresolved", "open task", "next step",
)

# Bullet lines, used for the structural half of the verbatim decision.
#
# Heading names alone were not enough, found by testing against a rebuilt
# version of the real incident: "## Environment and workflow notes" is not one
# of the six template headings, so it was compressed, and the model dropped
# "ADO has a 4000-char cap" from inside it -- the same section the incident lost
# outright. Matching more heading names would have patched that one case and
# left the next off-template section exposed.
#
# The real distinction isn't which heading a section has, it's what the body IS.
# A paragraph can be tightened without losing facts. A bullet list is a set of
# discrete facts, and "shorter" there means merging or dropping items. So a
# body that is mostly bullets is passed through regardless of its heading, which
# covers sections this template never anticipated.
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")


def _is_mostly_bullets(body: str) -> bool:
    lines = [l for l in body.splitlines() if l.strip()]
    if not lines:
        return False
    bullets = sum(1 for l in lines if _BULLET_LINE_RE.match(l))
    return bullets * 2 >= len(lines)


def split_prose_bullet_runs(body: str) -> list:
    """
    Split a section body into contiguous runs of prose lines and bullet lines.

    Returns a list of (is_bullet_run: bool, text: str) tuples. Blank lines are
    attached to the run that follows them (so a blank line between a bullet
    block and a prose block travels with the prose), keeping bullet blocks
    tight while giving prose runs the surrounding whitespace they need.

    Why attach blanks to what follows: bullet lists as authored rarely have
    trailing blank lines before the next structural element, but prose
    paragraphs typically have a leading blank line as an inter-paragraph
    separator. Attaching to the follower preserves that convention.

    A body with no bullets returns a single prose run. A body that is
    entirely bullets returns a single bullet run. Only a body with at least
    one prose line AND at least one bullet line produces two or more runs,
    which is the case `_compress_each_section` uses this for.
    """
    lines = body.splitlines()
    if not lines:
        return [(False, body)]

    runs: list = []
    # Collect pending blank lines to prepend to the next non-blank run.
    pending_blanks: list = []
    current_is_bullet: bool | None = None
    current_lines: list = []

    for line in lines:
        if not line.strip():
            # Blank line -- defer to the next non-blank line's run type.
            pending_blanks.append(line)
            continue
        is_bullet = bool(_BULLET_LINE_RE.match(line))
        if current_is_bullet is None:
            # First non-blank line: start the first run.
            current_is_bullet = is_bullet
            current_lines = pending_blanks + [line]
            pending_blanks = []
        elif is_bullet == current_is_bullet:
            # Same run type: absorb pending blanks and continue.
            current_lines.extend(pending_blanks)
            current_lines.append(line)
            pending_blanks = []
        else:
            # Run type changed: flush current run, start a new one.
            # The accumulated blanks travel with the NEW run (attach-to-follower).
            runs.append((current_is_bullet, "\n".join(current_lines)))
            current_is_bullet = is_bullet
            current_lines = pending_blanks + [line]
            pending_blanks = []

    # Flush the last run. Any trailing blank lines attach to the last run
    # (there is no following run to attach to).
    if current_is_bullet is not None:
        current_lines.extend(pending_blanks)
        runs.append((current_is_bullet, "\n".join(current_lines)))
    elif pending_blanks:
        # Body was entirely blank lines (no non-blank line was ever seen).
        # Return a single prose run containing the blanks, which is the
        # least surprising behaviour (prose rather than bullet).
        runs.append((False, "\n".join(pending_blanks)))

    return runs or [(False, body)]


def section_has_mixed_runs(body: str) -> bool:
    """
    True if the section body contains at least one prose run AND at least one
    bullet run (i.e., it is neither pure prose nor pure bullets).

    This is the condition under which `_compress_each_section` splits the body
    into runs rather than sending it whole to the model or passing it through
    verbatim, recovering compression ratio on the prose portions without
    weakening the 'a list of facts is never compressed' guarantee.
    """
    runs = split_prose_bullet_runs(body)
    has_prose = any(not is_bullet for is_bullet, _ in runs)
    has_bullets = any(is_bullet for is_bullet, _ in runs)
    return has_prose and has_bullets


def client(base_url: Optional[str]) -> OpenAI:
    return OpenAI(base_url=base_url or DEFAULT_BASE_URL, api_key="lm-studio")


def split_sections(text: str):
    """
    Split `text` into (preamble, [(heading_line, body), ...]) on '## ' headings.

    Returns ("", []) worth of nothing -- specifically (text, []) -- when there
    are no headings, so callers can fall back to whole-text compression instead
    of special-casing unstructured input.
    """
    lines = text.splitlines()
    fence = None
    heading_indexes = []
    for i, line in enumerate(lines):
        match = _SECTION_FENCE_RE.match(line)
        if match:
            marker, trailing = match.group(1), match.group(2)
            if fence is None:
                fence = (marker[0], len(marker))
            elif (
                marker[0] == fence[0]
                and len(marker) >= fence[1]
                and not trailing.strip()
            ):
                fence = None
            continue
        if fence is None and _SECTION_HEADING_RE.match(line):
            heading_indexes.append(i)
    if not heading_indexes:
        return text, []
    preamble = "\n".join(lines[: heading_indexes[0]]).strip()
    bounds = heading_indexes + [len(lines)]
    sections = []
    for k, start in enumerate(heading_indexes):
        body = "\n".join(lines[start + 1 : bounds[k + 1]]).strip()
        sections.append((lines[start].strip(), body))
    return preamble, sections


def section_is_verbatim(heading: str, body: str = "") -> bool:
    """
    True if this section should be passed through instead of compressed.

    Two independent reasons to pass a section through: its heading names a
    known exactness-critical section, or its body is a list of discrete facts
    rather than prose (see _is_mostly_bullets for why the structural test is
    the more general of the two).
    """
    lowered = heading.lower()
    if any(hint in lowered for hint in _VERBATIM_SECTION_HINTS):
        return True
    return _is_mostly_bullets(body)


def missing_sections(original: str, summary: str) -> list:
    """
    Headings present in `original` but absent from `summary`.

    The structural counterpart to missing_identifiers(). This is what turns
    "three whole sections silently disappeared" into a detectable condition: a
    heading either appears in the output or it does not.
    """
    _, sections = split_sections(original)
    return [h for h, _ in sections if h not in summary]


def restore_missing_sections(original: str, summary: str) -> str:
    """
    Append verbatim any section of `original` whose heading is absent from
    `summary`.

    Belt-and-suspenders behind the per-section compression path: even if that
    orchestration regresses, a dropped section is restored rather than lost.
    Over-restoration (a heading the model merely reformatted) costs some
    compression ratio and never costs content, which is the correct direction
    for an artifact someone resumes work from.
    """
    _, sections = split_sections(original)
    absent = [(h, b) for h, b in sections if h not in summary]
    if not absent:
        return summary
    restored = "\n\n".join(f"{h}\n{b}".rstrip() for h, b in absent)
    return (
        f"{summary}\n\n"
        "[SECTIONS DROPPED BY COMPRESSION -- restored verbatim from the source]\n"
        f"{restored}"
    )


_COMPACT_WORKING_ON_HEADING = "## What we were working on"
_COMPACT_LABEL_MAX_CHARS = 80


def derive_compact_label(information: str) -> str:
    """
    Derive a short, deterministic label from a structured compact summary.

    Looks for the "## What we were working on" section produced by the
    /my-compact skill template (split_sections parses the same '## ' heading
    convention the template uses) and extracts the first sentence from its
    body -- split on '.', '!', or '?', hard-capped at _COMPACT_LABEL_MAX_CHARS
    characters.  Falls back to the first non-blank line of that section if no
    sentence-terminator is found, and falls back further to the whole preamble
    if the section itself is absent.

    Returns "" when nothing meaningful can be extracted, so callers can
    substitute a fallback (e.g. "(unlabeled)") rather than storing a blank
    label that renders unreadably in the /my-resume picker.

    Pure string logic -- no model call, no I/O, always the same output for
    the same input.
    """
    _, sections = split_sections(information)
    body = ""
    for heading, section_body in sections:
        if heading.strip() == _COMPACT_WORKING_ON_HEADING:
            body = section_body.strip()
            break

    # If the structured section is absent, try the preamble (unstructured
    # one-shot summaries still benefit from a label, even if it's cruder).
    if not body:
        preamble, _ = split_sections(information)
        body = preamble.strip()

    if not body:
        return ""

    # Extract the first sentence across wrapped lines: join ALL body lines
    # with a single space first, then find the sentence terminator. Operating
    # on just `body.splitlines()[0]` truncates at the first physical newline,
    # which cuts off a sentence that the author wrapped for readability (the
    # COMPACT_TEMPLATE itself does exactly this: "Aligning the OA publisher
    # with the provisioned\nOrderNotification Service Bus topology.").
    # The joined form is only used for sentence-boundary detection; the
    # first physical line is kept as the no-terminator fallback so callers
    # that write un-punctuated one-liners still get something readable.
    joined = " ".join(line.strip() for line in body.splitlines() if line.strip())
    first_line = body.splitlines()[0].strip()

    # Sentence split: find the earliest '.', '!', '?' that ends a sentence
    # (followed by whitespace or end of string) in the JOINED text.
    match = re.search(r"[.!?](?=\s|$)", joined)
    if match:
        label = joined[: match.start()].strip()
    else:
        # No sentence terminator anywhere in the body -- use the first
        # physical line rather than the full joined blob, which could be
        # an arbitrarily long prose paragraph.
        label = first_line

    return label[:_COMPACT_LABEL_MAX_CHARS].strip()


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (chars / 3.5) used ONLY by the opt-in savings
    tracker (see CLAUDE_RUNWAY_TRACK_SAVINGS in savings_ledger.py) -- never
    for any compression decision in this file. Deliberately the same crude
    ratio on both sides of every savings comparison (raw vs. compressed),
    so error in the ratio itself mostly cancels out of the resulting
    percentage. Claude's exact tokenizer isn't public, so a "more precise"
    local tokenizer wouldn't actually be more accurate here -- just
    differently wrong.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 3.5))


# ---------------------------------------------------------------------------
# In-session compression cache (issue #60)
# ---------------------------------------------------------------------------
#
# Skips a redundant LM Studio round-trip when the exact same content is
# compressed more than once in a session -- e.g. the same file re-read, the
# same URL re-fetched, the same command re-run with identical output.
#
# Design: an in-memory dict keyed on a SHA-256 digest of the inputs that
# determine the output (text, focus, resolved model id, max_chars,
# auto_truncated, chunk_chars, preserve_identifiers, preserve_sections,
# effective base_url). Sized-capped so pathological
# caller patterns don't accumulate unbounded memory; the cap is enforced by
# evicting the oldest entry (insertion-order dicts in Python 3.7+, so the
# oldest is always `next(iter(_compression_cache))`).
#
# Why in-memory rather than SQLite (which savings_ledger uses):
#   - Compression results don't need to persist across sessions -- the source
#     content (a file, a URL, a command) may change, so yesterday's cache
#     entry for today's content is not useful.
#   - No file-coordination overhead: MCP-server and hook processes are separate
#     OS processes; a shared SQLite DB would require locking across them for
#     every cache hit/miss, adding latency to the common case (no hit).
#   - The common pattern is: Claude reads a file, notices the output is large,
#     asks this server to compress it, then the PostToolUse hook ALSO compresses
#     the same output. Two processes, same session, same content -- exactly the
#     case that would be a cache hit if they shared storage, but the hook's own
#     in-process cache already covers the same-process case (e.g. the MCP server
#     compresses the same file twice in one session).
#   - In practice, each process's own in-memory cache is sufficient for the
#     within-session deduplication this issue targets. A future cross-process
#     cache can be layered on top if measured to matter.
#
# model is included in the key: the same content compressed with a different
# model (user swapped models mid-session) must NOT share a cache entry, since
# model quality and style differ. max_chars matters because a truncated
# compress() of the same content yields a different (shorter) result.
# effective_url (base_url or DEFAULT_BASE_URL) is included because model IDs
# are not globally unique across instances -- two different LM Studio servers
# can expose the same ID for different model weights, so the URL distinguishes
# entries from different physical servers.

# Public so tests can inspect and clear it without importing private names.
# `dict` preserves insertion order (Python 3.7+), which is what makes the
# oldest-entry eviction below O(1).
_compression_cache: dict = {}

# Two independent bounds keep the cache's memory footprint under control for
# the long-lived MCP server process:
#
# _CACHE_MAX_ENTRIES: maximum number of entries. 256 is comfortably above
# any realistic session's unique compression count.
#
# _CACHE_MAX_VALUE_CHARS: maximum characters for a SINGLE cached value. A
# preserve_sections result can be close to the 2,000,000-char input limit when
# most sections pass through verbatim -- caching 256 of those would consume
# well over 500 MB of the server process's memory. Entries larger than this
# threshold are never stored (the cost of re-compressing them is bounded, and
# the compression ratio is low anyway if the result is nearly input-sized).
# 100,000 chars is ~70-100 KB, comfortably above any real compressed summary
# while rejecting pathologically large verbatim-passthrough results.
_CACHE_MAX_ENTRIES = 256
_CACHE_MAX_VALUE_CHARS = 100_000


def _compression_cache_key(
    text: str,
    focus: str,
    model: Optional[str],
    max_chars: Optional[int],
    preserve_identifiers: bool,
    preserve_sections: bool,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    auto_truncated: bool = False,
    effective_url: str = DEFAULT_BASE_URL,
) -> str:
    """
    SHA-256 hex digest of the arguments that fully determine compress()'s
    output. Returns a 64-char hex string suitable for use as a dict key.

    Uses \\x00 as a field separator, which cannot appear in any of the
    string arguments under normal usage (focus is human-entered prose, model
    is an alphanumeric id) -- so "ab" + "cd" can never collide with "a" +
    "bcd" the way a plain concatenation could.

    max_chars, auto_truncated, preserve_identifiers, preserve_sections,
    chunk_chars, and effective_url are all included because each of those
    affects the returned string:
    - max_chars: by the time this function is called, `compress()` has already
      mutated a positional None into the auto-detected boundary integer -- so
      an originally-None max_chars and an explicit max_chars equal to the same
      boundary would produce the same `max_chars` value here. auto_truncated
      distinguishes these two cases, because their results differ: an
      auto-detected call adds a "[note: focus looked positional...]" suffix
      while an explicit call adds a "[note: max_chars=N truncated...]" suffix
      (or none at all if the limit wasn't reached).
    - preserve_identifiers and preserve_sections gate identifier-repair and
      redaction.
    - chunk_chars determines chunk boundaries, which changes what each LM
      Studio call sees and thus the summary it returns, whether a reduce
      step runs (one chunk vs. many), and the reported "across N chunk(s)"
      count in the output string.
    - effective_url identifies the physical LM Studio server. Model IDs are
      not globally unique across instances -- two different servers can expose
      the same ID for entirely different weights. Including the URL prevents a
      request to server B from being served by server A's cached result.
      Callers pass `base_url or DEFAULT_BASE_URL` so a None base_url (use
      the default) and an explicit default-value base_url hash identically.

    skip_if_under_chars is NOT included -- it controls whether compression
    runs at all, but the cache is only populated when compression actually
    ran (under-threshold calls return before `_cache_put` is ever reached),
    so its value can't affect what's in the cache.
    """
    blob = (
        f"{text}\x00{focus}\x00{model or ''}\x00{max_chars}\x00"
        f"{preserve_identifiers}\x00{preserve_sections}\x00{chunk_chars}\x00"
        f"{auto_truncated}\x00{effective_url}"
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    """Return the cached result for `key`, or None if not present."""
    return _compression_cache.get(key)


def _cache_put(key: str, value: str) -> None:
    """
    Store `value` under `key`, subject to two independent size guards:

    1. Per-entry character limit (_CACHE_MAX_VALUE_CHARS): oversized values
       (e.g. a preserve_sections result where most sections passed through
       verbatim, keeping the output nearly as large as the input) are silently
       not stored. Caching them would let 256 such entries consume hundreds of
       megabytes in the long-lived MCP server process; skipping them instead
       means the next call re-compresses, which is safe because compression
       did little work anyway when the result is input-sized.

    2. Entry-count limit (_CACHE_MAX_ENTRIES): evicts the oldest-inserted
       entry first once the count cap would be exceeded, keeping the cache
       bounded regardless of how varied the inputs are.

    Eviction policy is FIFO (first-in, first-out), NOT LRU: _cache_get() does
    NOT update insertion order (dict.__getitem__ is a read-only lookup that
    leaves the dict's own ordering intact), so a frequently hit old entry is
    still evicted before a newer one that has never been hit. This is an
    intentional simplicity tradeoff -- O(1) put and get, no bookkeeping, and
    for a session-scoped 256-entry cache the difference between FIFO and LRU
    is rarely observable in practice (a session typically compresses far fewer
    than 256 distinct content blobs).

    Re-inserting an existing key would update the value in place WITHOUT
    changing its insertion position (dict.__setitem__ on an already-present
    key is an in-place update) -- but in practice this never happens, because
    a cache HIT returns the stored value without calling _cache_put again. It
    could only arise from a SHA-256 collision across different inputs, which
    this library makes negligibly unlikely.
    """
    if len(value) > _CACHE_MAX_VALUE_CHARS:
        # Oversized result -- skip caching. The entry count cap below is
        # unaffected; do not evict an existing entry just because this one
        # is too large to store.
        return
    if len(_compression_cache) >= _CACHE_MAX_ENTRIES:
        oldest_key = next(iter(_compression_cache))
        del _compression_cache[oldest_key]
    _compression_cache[key] = value


def clear_compression_cache() -> None:
    """
    Remove all entries from the in-session cache.

    Exposed primarily for tests (so each test case starts with a clean slate
    without relying on module reload) and for any caller that needs to
    force-invalidate on a known content change (e.g. after a file has been
    edited and the caller knows the previous result is stale).
    """
    _compression_cache.clear()


def chunk_text(text: str, chunk_chars: int):
    return [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)] or [""]


# Token shapes whose value is entirely in being byte-exact. Deliberately
# conservative: a missed identifier costs one un-restored token, while a
# false positive appends prose noise to every compaction, so each pattern
# requires structural evidence (a separator, a case transition, a length
# floor) rather than "looks technical".
_IDENTIFIER_PATTERNS = (
    # Paths with a file extension, POSIX or Windows separators:
    # src/utils/auth.ts, Documentation/x/y.md, src\Validators\Invoice.cs.
    # The separator before the filename is [/\\], not just / -- this repo
    # supports Windows (docs/windows-setup.md), where a handoff summary's
    # paths are backslash-style and would otherwise go unverified.
    # The optional drive prefix keeps an absolute Windows path whole: without
    # it, C:\repo\src\a.cs is captured as \repo\src\a.cs, which still matches
    # as a substring but restores an incomplete path when the summary drops it.
    r"(?:[A-Za-z]:)?[\w.\-/\\]+[/\\][\w.\-]+\.[A-Za-z0-9]{1,8}",
    # Colon-delimited config/secret keys, 2+ separators:
    # AppSecretKey:Options:OrderNotificationServiceBus:ConnectionKey
    r"[A-Za-z][\w\-]*(?::[A-Za-z][\w\-]*){2,}",
    # Dotted namespaces / assemblies, 2+ dots: Acme.Billing.OrdersApi
    r"[A-Za-z][\w\-]*(?:\.[A-Za-z][\w\-]*){2,}",
    # CamelCase with at least two humps, which is what class and method names
    # look like: GetUser, UserService, InvoiceStatusNotificationRequestDtoValidator.
    # {1,} (2 humps total), not {2,} (3+) -- the original 3-hump minimum missed
    # ordinary 2-segment .NET-style names like UserService/ApiClient/GetUser.
    r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+){1,}\b",
    # snake_case / SCREAMING_SNAKE with a real separator, optional leading
    # underscore: compute_file_hashes, _EXACT_CMDS, __init__. Requiring at
    # least one underscore is what keeps ordinary prose out -- a hyphenated
    # word like "well-known" has no underscore and never matches here.
    r"\b_{0,2}[A-Za-z][A-Za-z0-9]*(?:_+[A-Za-z0-9]+)+_{0,2}\b",
    # CLI flags, where a renamed or dropped flag is a broken command:
    # --dry-run, -o, -h. Single-letter flags are included (trailing part is
    # optional), because dropping the -o from `az ... -o json` changes what
    # the command returns. The (?<!\w) lookbehind is what keeps this off
    # ordinary hyphenation -- in "state-of-the-art" and "UTF-8" the character
    # before the dash is a word character, so no match is attempted.
    r"(?<!\w)--?[A-Za-z][\w\-]*",
    # Long digit runs: ticket/PR/build ids. Short numbers are excluded on
    # purpose -- requiring "732" to survive would fire on ordinary prose.
    r"\b\d{5,}\b",
)
_IDENTIFIER_RE = re.compile("|".join(f"(?:{p})" for p in _IDENTIFIER_PATTERNS))

# Words matching a pattern above that carry no exactness value. Kept small on
# purpose; this is a noise filter, not a stopword list.
_IDENTIFIER_STOPLIST = frozenset({
    "e.g.", "i.e.", "etc.", "--help", "--version",
})

# Credential shapes, which must NEVER be collected. This is a security filter,
# not a noise filter, and it is the one place where over-collection is unsafe.
#
# The failure it prevents, reproduced before this existed: several of the
# patterns above match real secrets -- the snake_case one accepts
# "ghp_<...>" and "sk_live_<...>", the dotted one accepts a three-part JWT.
# When the summarizer omitted a secret (which it did -- dropping it was
# accidentally protective), append_missing_identifiers put it BACK, verbatim,
# and /my-compact enables this feature unconditionally and then persists the
# result to Qdrant. So the repair step converted a transient secret in a
# session into a durable one in a vector store, and made a leak out of the
# summarizer's accidental good behavior.
#
# Prefix matching rather than entropy scoring on purpose: an entropy heuristic
# would also flag commit SHAs, build ids, and hashes -- exactly the
# identifiers worth restoring. A missed exotic credential shape is a gap; a
# dropped commit SHA is a regression in the feature's whole point.
_CREDENTIAL_PREFIXES = (
    # GitHub
    "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_",
    # OpenAI / Anthropic style
    "sk-", "sk-ant-", "sk-proj-",
    # Stripe
    "sk_live_", "sk_test_", "rk_live_", "rk_test_", "pk_live_",
    # Slack
    "xoxb-", "xoxp-", "xoxa-", "xoxr-", "xoxs-", "xapp-",
    # GitLab / npm / Docker / HuggingFace / Shopify
    "glpat-", "npm_", "dckr_pat_", "hf_", "shpat_", "shpss_",
    # AWS access key ids / Google API keys
    "AKIA", "ASIA", "AIza",
    # GCP — OAuth2 refresh tokens use the "1//" prefix (documented in Google's
    # OAuth2 reference; the body is base64url after the slashes):
    # 1//<base64url-body>
    "1//",
)

# Credential VALUES, matched wherever they appear -- not anchored to the start
# of a word.
#
# Anchoring to word-start was the second wrong answer here (the first was
# checking only the collected token, which missed fragments of hyphenated
# secrets). A secret is very often not a standalone word, and 6 of 8 realistic
# forms leaked the full value in testing:
#
#     GITHUB_TOKEN=ghp_<...>            leaked
#     export GITHUB_TOKEN=ghp_<...>     leaked
#     {"token":"ghp_<...>"}             leaked
#     --with-token=ghp_<...>            leaked
#     https://x:ghp_<...>@github.com/   leaked
#     TOKEN="ghp_<...>"                 leaked
#
# Worse, credential_token_count() returned 0 for those, so the "withheld"
# disclosure did not fire either -- silently wrong rather than visibly
# incomplete. Matching the value itself covers every wrapper without needing to
# enumerate assignment, JSON, YAML, flag, and URL syntaxes separately.
#
# The (?<![A-Za-z0-9_]) lookbehind is load-bearing: without it the short "sk-"
# prefix matches inside ordinary words ("task-oriented" -> "sk-oriented"),
# which would mark innocent text as a credential span and suppress the real
# identifiers sitting in it.
_CREDENTIAL_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    # A known prefix followed by enough body to be a real token. Short
    # prefixes like "sk-" still capture the whole of "sk-ant-api03-<...>",
    # because the body class includes the hyphen.
    + "|".join(re.escape(p) for p in _CREDENTIAL_PREFIXES) + r")[A-Za-z0-9_\-]{6,}"
    # base64url-encoded JSON: a JWT header and up to two following segments.
    r"|(?<![A-Za-z0-9_])eyJ[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+){0,2}"
    # PEM private keys: the WHOLE block, header through footer.
    #
    # Matching only the header left the key body outside the credential span,
    # where it could be collected as an identifier and restored verbatim. The
    # CamelCase pattern is \b-anchored and base64 is one long word with no
    # internal boundaries, so a match can only begin at a line's first
    # character -- which makes this probabilistic rather than certain, and is
    # why a typical sample looks clean. Measured: ~1.5% of random 64-char
    # base64 lines begin with a CamelCase-shaped run, so for a 25-line 2048-bit
    # key the chance that at least one line leaks is ~31%, and ~53% for a
    # 50-line 4096-bit key. A leaked line is 64 characters of key material.
    #
    # The \Z fallback covers an unterminated block (a truncated summary), where
    # everything after the header is treated as key material. That
    # over-suppresses -- identifiers past the header go unverified -- which is
    # the correct direction: an unverified identifier is a gap, leaked key
    # material is a breach.
    r"|-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?"
    r"(?:-----END[A-Z ]*PRIVATE KEY-----|\Z)"
    # Azure Databricks PATs: "dapi" followed by EXACTLY 32 hex characters.
    # Excluded from _CREDENTIAL_PREFIXES (which appends a generic
    # [A-Za-z0-9_\-]{6,} body) because "dapi" alone would then match
    # ordinary identifiers like dapi_request, dapi_handler, etc. -- names
    # common in API code. The documented format is strict hex only, so the
    # dedicated pattern is both more accurate and false-positive-free.
    # The right-side boundary (?![A-Za-z0-9_]) is load-bearing: without it a
    # 40-char git-style hash beginning with "dapi" would match its first 32
    # characters and be silently redacted even though it's not a real token.
    r"|(?<![A-Za-z0-9_])dapi[0-9a-fA-F]{32}(?![A-Za-z0-9_])"
)


def looks_like_credential(token: str) -> bool:
    """True if `token` contains something shaped like a secret.

    Deliberately errs toward withholding: a legitimate identifier wrongly
    withheld is one unverified token, while a credential wrongly restored is a
    secret written into a durable store.
    """
    return bool(_CREDENTIAL_VALUE_RE.search(token))


def _credential_spans(text: str) -> list:
    """Character spans of credential values anywhere in `text`."""
    return [m.span() for m in _CREDENTIAL_VALUE_RE.finditer(text)]


_REDACTION_PLACEHOLDER = "[REDACTED-CREDENTIAL]"


def redact_credentials(text: str):
    """
    Replace every credential value in `text` with a placeholder.
    Returns (redacted_text, number_of_replacements).

    Excluding credentials from identifier collection is not sufficient on its
    own, because that only governs what gets RE-APPENDED. Content can reach the
    output by being copied rather than regenerated -- the preamble, a verbatim
    section, a failed-call fallback -- and a model can also echo a secret out of
    its own input. Redacting the finished artifact is the only step that covers
    all of those routes at once.

    Applied only in the artifact-producing modes (preserve_identifiers /
    preserve_sections), which are the ones whose output gets persisted. A
    throwaway gist of a web page is not worth rewriting.
    """
    spans = _credential_spans(text)
    if not spans:
        return text, 0
    pieces, last = [], 0
    for start, end in spans:  # finditer spans are ordered and non-overlapping
        pieces.append(text[last:start])
        pieces.append(_REDACTION_PLACEHOLDER)
        last = end
    pieces.append(text[last:])
    return "".join(pieces), len(spans)


def redact_and_disclose(text: str) -> str:
    """
    Redact credentials and state how many were removed, or return `text`
    unchanged when there were none.

    The disclosure is derived from replacements actually made in THIS text,
    not from what was detected in the source. The earlier version counted the
    source and printed "withheld and NOT restored" -- which, once verbatim
    passthrough existed, appeared directly alongside a secret it had not
    removed. A false assurance is worse than none.
    """
    redacted, count = redact_credentials(text)
    if not count:
        return redacted
    return (
        f"{redacted}\n\n"
        f"[{count} credential-shaped value(s) redacted from this artifact]"
    )


def identifier_tokens(text: str) -> list:
    """
    Identifier-shaped tokens in `text`, in first-appearance order.

    Used to verify that a lossy summary preserved the strings a reader would
    later grep for. This is a detector, not a parser: it over-collects mildly
    (an ordinary hyphenated word can look like a flag) and under-collects
    deliberately (bare short numbers are skipped).

    Over-collection is harmless for ordinary tokens -- re-appending one that
    was already present changes nothing -- with ONE exception that is not
    harmless at all: credentials. Several of these patterns match real secrets,
    and restoring a secret the summarizer had dropped writes it into a durable
    artifact. Those are excluded here via looks_like_credential(), so they are
    never collected and therefore never restored.
    """
    credential_spans = _credential_spans(text)
    seen: dict[str, None] = {}  # emulates an ordered set -- keys are what matters, list(seen) preserves insertion order
    for match in _IDENTIFIER_RE.finditer(text):
        token = match.group(0).strip(".,;:)(»«\"'`")
        # Flags get a shorter floor than everything else: "-o" is only two
        # characters but dropping it changes what a command returns, while a
        # two-character non-flag token is noise. A single floor of 4 silently
        # discarded every single-letter flag no matter what the regex matched.
        floor = 2 if token.startswith("-") else 4
        if len(token) < floor or token in _IDENTIFIER_STOPLIST:
            continue
        # Skip anything overlapping a credential word, which covers both a
        # whole secret and a fragment of one.
        start, end = match.span()
        if any(start < c_end and c_start < end for c_start, c_end in credential_spans):
            continue
        seen.setdefault(token, None)
    return list(seen)


def credential_token_count(text: str) -> int:
    """
    How many distinct credential-shaped words appear in `text`.

    Exists so the repair step can say that something was deliberately not
    restored, without printing it. A silent omission would leave a reader
    believing the restored set is complete.
    """
    return len({text[start:end] for start, end in _credential_spans(text)})


def missing_identifiers(original: str, summary: str) -> list:
    """
    Identifier tokens present in `original` but absent from `summary`.

    Comparison is case-sensitive and substring-based: a token reformatted by
    the model (wrapped in backticks, say) still matches, while one whose
    characters changed does not -- and a changed identifier is exactly the
    corruption worth catching. A token that is a substring of a longer
    surviving token counts as present, which is correct: the grep succeeds.
    """
    return [t for t in identifier_tokens(original) if t not in summary]


def append_missing_identifiers(original: str, summary: str, limit: int = 200) -> str:
    """
    Append any identifiers the summary dropped, verbatim, as a labeled block.

    Repair rather than rejection: the summary's prose is still useful, so the
    exact strings are restored beside it instead of discarding the whole
    result. The block is labeled so a reader knows these came from the source
    and not from the model. `limit` caps the block, and a truncation is
    reported rather than silent -- an unbounded append on a pathological input
    would defeat the point of compressing at all.
    """
    missing = missing_identifiers(original, summary)
    if not missing:
        return summary
    shown, dropped = missing[:limit], max(0, len(missing) - limit)
    note = f" ({dropped} more not shown)" if dropped else ""
    return (
        f"{summary}\n\n"
        f"[VERBATIM IDENTIFIERS DROPPED BY COMPRESSION -- restored from the source"
        f"{note}]\n" + "\n".join(shown)
    )


# Trailing totals/result lines -- often the single most decision-relevant
# fact in a command's output ("did the whole thing pass?"), and measurably
# the kind of thing a small local model drops while everything around it
# still reads as a complete summary.
#
# Reproduced directly: a real pytest failure log's map-step summary listed
# all 3 failures correctly but omitted the final "3 failed, 39 passed in
# 8.72s" line entirely -- across the default focus AND a focus explicitly
# asking for a bullet-per-fact format. The fix was tried as a prompt change
# first (appending "always preserve the final totals line" to the focus),
# and that DID stop the drop, but it also made the model markedly more
# conservative everywhere else: on a clean, all-passing Gradle build, the
# unmodified focus correctly aggregated 8 passing tests into "All tests
# PASSED. BUILD SUCCESSFUL in 47s." (208 chars); the same input with the
# added instruction started listing every individual passing test instead
# (433-618 chars across repeated trials) despite the instruction only
# mentioning the trailing line. A single added clause changed the model's
# overall compression behavior, not just its handling of the last line --
# the exact failure mode this file's other preserve_* features exist to
# route around (see preserve_identifiers/preserve_sections' docstrings):
# a small model can't be prompted into protecting one fact without that
# request bleeding into how it treats everything else. So, same as those
# features, this is measured and repaired after the fact rather than
# requested up front -- it doesn't touch the model's summarization
# behavior at all, so it carries none of that side effect.
#
# Each pattern requires a recognizable, structural shape (a known keyword
# plus digits, not just prose that happens to mention a number) for the
# same reason _IDENTIFIER_PATTERNS does: a missed shape costs one
# un-restored line, but a false positive risks appending something that
# only looked like a result line.
_TRAILING_SUMMARY_PATTERNS = (
    # pytest / unittest: "3 failed, 39 passed in 8.72s", "39 passed in 1.02s"
    r"\d+\s+(?:passed|failed|error|skipped|warning)s?"
    r"(?:,\s*\d+\s+(?:passed|failed|error|skipped|warning)s?)*\s+in\s+[\d.]+s",
    # Jest/Mocha-style totals block: "Tests:       3 failed, 39 passed, 42 total"
    r"Tests:\s*\d+\s+failed,\s*\d+\s+passed,\s*\d+\s+total",
    # go test: a tab/space-separated "ok  <package>  0.003s" line, or a bare
    # PASS/FAIL as the entire line (anchored so it can't match "PASS" as a
    # substring of ordinary prose).
    r"^ok\s+\S+\s+[\d.]+s\s*$",
    r"^(?:PASS|FAIL)\s*$",
    # Gradle/Maven
    r"BUILD (?:SUCCESSFUL|FAILED|SUCCESS|FAILURE)\b[^\n]*",
    # dotnet test
    r"(?:Passed|Failed)!\s*-\s*Failed:\s*\d+,\s*Passed:\s*\d+,\s*Skipped:\s*\d+,\s*Total:\s*\d+[^\n]*",
    # RSpec: "10 examples, 2 failures", optionally with pending
    r"\d+\s+examples?,\s*\d+\s+failures?(?:,\s*\d+\s+pending)?",
    # Playwright: "N passed (dur)" / "N failed (dur)" / bare "N failed" --
    # the parenthesised duration is the primary distinguisher (separates from
    # the pytest pattern above, which uses " in Xs"), but an all-failing run
    # emits a bare "N failed" anchored to end-of-line.
    # Duration forms Playwright emits (from its @playwright/test source):
    #   Xs      "4.2s"  "30s"
    #   Xms     "487ms"
    #   Xm Ys   "1m 2.3s"          (compound minutes+seconds)
    #   X.Ym    "1.7m"             (fractional minutes for very long runs)
    # The duration sub-expression covers all four: \d+\.?\d*(?:ms|s|m\s+\d+\.?\d*s|m)
    r"(?:\d+\s+(?:passed|failed)\s+\(\d+\.?\d*(?:ms|s|m\s+\d+\.?\d*s|m)\)|\d+\s+failed$)",
    # Cypress: "N passing (dur)" / "N failing" / "N pending" -- the
    # parenthesised wall-clock duration is only shown for "passing"; "failing"
    # and "pending" appear on their own lines without a duration.
    # Real examples from cypress run output:
    #   "  3 passing (4s)"   "  3 passing (487ms)"   "  3 passing (1m)"
    #   "  1 failing"        "  2 pending"
    # "failing"/"pending" anchored to end-of-line so ordinary prose
    # ("3 failing builds were fixed") doesn't match.
    # Duration forms (Mocha's ms module, used by Cypress):
    #   Xms    "487ms"
    #   Xs     "4s"
    #   Xm     "1m"               (minute-only for runs >= 60 s)
    #   Xm Ys  "1m 4s"            (compound, rounded to nearest second)
    r"(?:\d+\s+passing\s+\(\d+\.?\d*(?:ms|s|m\s+\d+\.?\d*s|m)\)|\d+\s+(?:failing|pending)$)",
    # JUnit 5 (Launcher / console standalone): summary lines emitted by the
    # ConsoleLauncher look like:
    #   "[         3 tests found           ]"
    #   "[         2 tests started         ]"
    #   "[         1 tests failed          ]"
    # The brackets plus "tests" keyword plus a known status word are the
    # structural anchors -- plain prose can't produce this shape.
    r"\[\s*\d+\s+tests?\s+(?:found|started|failed|passed|aborted|skipped|successful)\s*\]",
)
_TRAILING_SUMMARY_RE = re.compile("|".join(f"(?:{p})" for p in _TRAILING_SUMMARY_PATTERNS), re.MULTILINE)

# How far back from the end of the text to look. Keeps this feature scoped to
# what it's for -- restoring THE result of the whole run -- rather than
# matching a totals-shaped line that happens to appear mid-log (e.g. one
# retry's result in a multi-attempt CI step), which is not the answer to
# "how did this end."
#
# 500 was too narrow: a CI log with a trailing coverage table or
# artifact-upload confirmation (typically 800-1200 chars) could push the real
# pass/fail line past the window, causing a silent miss (#47). 1500 covers
# those realistic noise patterns while staying well short of where a stale
# retry result would appear.
_TRAILING_SUMMARY_WINDOW = 1500


def find_trailing_summary_line(text: str, window: int = _TRAILING_SUMMARY_WINDOW) -> Optional[str]:
    """
    Returns the totals/result line nearest the end of `text` (e.g. "3 failed,
    39 passed in 8.72s", "BUILD SUCCESSFUL in 47s"), or None if the tail
    doesn't contain one of the recognized shapes.

    Only the last `window` chars are searched, and the LAST match within
    that window wins if more than one appears -- both narrow this to "the
    actual final result," not any totals-shaped line anywhere near the end.
    """
    matches = list(_TRAILING_SUMMARY_RE.finditer(text[-window:]))
    return matches[-1].group(0).strip() if matches else None


def append_trailing_summary_if_missing(original: str, summary: str) -> str:
    """
    If `original` ends with a totals/result line that isn't already present
    in `summary`, append it, unmodified, as a labeled `[RESULT] ...` line --
    same convention as append_missing_identifiers' labeled block and
    restore_missing_sections' bracketed header, so a reader can tell this
    line was restored rather than produced by the model.

    A no-op, not a truncation risk: most compressions never touch this at
    all, since the model usually keeps this line on its own (confirmed by
    testing -- it isn't dropped every time, just often enough to matter).
    This only fires on the specific gap the docstring above measured.
    """
    line = find_trailing_summary_line(original)
    if not line or line in summary:
        return summary
    return f"{summary}\n\n[RESULT] {line}"


def complete(oai_client: OpenAI, model: str, system: str, content: str) -> Optional[str]:
    """Returns the completion text, or None if the request failed for any reason
    (connection error, model not found, etc.) -- callers must check for None
    rather than assume this always succeeds just because resolve_model did.
    resolve_model only validates connectivity when it has to auto-detect; an
    explicit model= or CLAUDE_RUNWAY_LMSTUDIO_MODEL skips that check entirely, so
    a dead LM Studio server is only caught here, at the actual request."""
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
    Resolution order: explicit param > CLAUDE_RUNWAY_LMSTUDIO_MODEL env var >
    auto-detect (only when exactly one model is currently loaded in LM Studio).
    """
    # Checked unconditionally, before the explicit_model/DEFAULT_MODEL
    # short-circuits below -- not just on the auto-detect branch reached only
    # when neither of those is set. A stale var can be about something
    # OTHER than the model itself (e.g. LMSTUDIO_BASE_URL never renamed to
    # CLAUDE_RUNWAY_LMSTUDIO_URL) while the model resolves fine via the new
    # CLAUDE_RUNWAY_LMSTUDIO_MODEL var -- if we returned early on the model
    # before ever checking this, that other partial migration would stay
    # silent and this whole call would go on to use the wrong base URL with
    # no diagnosis of why (issue #41). stale_env_warning() itself only fires
    # per-pair (old set + that SAME pair's new name missing), so a
    # successfully migrated model var never masks or gets masked by a
    # still-stale URL var, and vice versa.
    stale = stale_env_warning()
    if stale:
        return None, stale

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
        "Pass model= explicitly, or set CLAUDE_RUNWAY_LMSTUDIO_MODEL to pin a default. "
        "NOTE: that var must be set in BOTH .mcp.json's env block (for the MCP server) "
        "and exported at the shell level (for the hooks) -- setting only one leaves the "
        "other half hitting this exact error. See the README's 'Environment variables' section."
    )


async def _compress_each_section(text, focus, oai_client, model, base_url, preserve_identifiers, ctx):
    """
    Compress a headed document section by section, or return None if it has no
    headings (so the caller falls back to whole-text compression).

    Why per-section instead of one call over the whole thing: a single 10k-char
    call asked to be shorter can only get shorter by discarding content, and
    whole sections are the cheapest thing to discard. That is the reported
    failure -- a real handoff lost all of its Unresolved Questions, all of its
    Environment notes, and a user instruction, while reading as complete. Each
    section on its own is small enough that no such sacrifice is needed.

    Verbatim sections are never sent to the model at all (see
    _VERBATIM_SECTION_HINTS), and every heading is re-emitted from the source
    rather than from model output, so a section cannot silently vanish.

    The returned string is always success-shaped ("[compressed N -> M
    chars...]") EXCEPT when every section actually SENT a request that FAILED
    outright (complete() itself returned None) and none succeeded -- that
    case gets an "[LM Studio appears unreachable ...]" line prepended instead,
    so a total outage during this run is distinguishable from a normal one
    (issue #43) without regressing the content-preserving fallback itself:
    each section's original body is still preserved either way, never
    dropped -- though "preserved" means kept out of the model's summarizing
    pass, not a promise of byte-for-byte identity: redact_and_disclose()
    still runs over the whole result afterward regardless of which path got
    it there, so a credential-shaped value in a fallback section is still
    replaced and disclosed exactly as it would be anywhere else (PR #134
    review, round 3) -- the warning text below is worded to reflect that.

    A "NOT RELEVANT" verdict is deliberately NOT counted toward that signal,
    even though it also falls back to the original body: it's a valid,
    successful response from a reachable model, just one that doesn't apply
    to this section, so a document whose every attempted section happens to
    be judged not relevant must not be reported as an outage (PR #134
    review) -- LM Studio answered every request just fine. Same reasoning
    applies to a genuinely empty/whitespace-only response: per complete()'s
    own contract, it returns None specifically for a request failure
    (connection error, model not found, etc.) and the completion's actual
    text -- which can legitimately be "" -- on a real, successful response.
    Checking `is None` BEFORE stripping (rather than folding None and "" and
    "   " together via `(summary or "").strip()`) is what makes that
    distinction possible; collapsing them, as an earlier version of this
    function did, made an empty-but-successful response indistinguishable
    from an actual outage (also PR #134 review).
    """
    preamble, sections = split_sections(text)
    if not sections:
        return None

    original_len = len(text)
    parts = [preamble] if preamble else []
    compressed_count = verbatim_count = 0
    request_failed_count = not_relevant_count = empty_response_count = 0

    for i, (heading, body) in enumerate(sections, 1):
        if ctx is not None:
            await ctx.report_progress(
                progress=i - 1, total=len(sections),
                message=f"Section {i}/{len(sections)}: {heading[:40]}",
            )
        if not body:
            parts.append(heading)
            continue
        if section_is_verbatim(heading, body):
            parts.append(f"{heading}\n{body}")
            verbatim_count += 1
            continue
        # Mixed section (prose + bullet runs): compress only the prose runs and
        # keep each bullet run verbatim -- recovering compression ratio on the
        # prose portions without weakening the 'a list of facts is never
        # compressed' guarantee (issue #61).
        if section_has_mixed_runs(body):
            runs = split_prose_bullet_runs(body)
            run_parts = []
            prose_compressed = False   # at least one prose run was actually compressed
            prose_failed = False       # at least one prose run had a request failure
            for is_bullet, run_text in runs:
                if is_bullet:
                    run_parts.append(run_text)
                    continue
                # Prose run: send to the model, fall back to verbatim on any
                # failure -- same contract as the whole-section path below.
                raw = complete(
                    oai_client, model,
                    f"You compress one section of a structured handoff document "
                    f"for another AI to read next. "
                    f"{_MAP_ANTI_CHAT_INSTRUCTION}\n\nFocus: {focus}",
                    run_text,
                )
                if raw is None:
                    run_parts.append(run_text)
                    prose_failed = True
                    continue
                stripped = raw.strip()
                if not stripped or "NOT RELEVANT" in stripped.upper():
                    run_parts.append(run_text)
                    continue
                # Preserve any leading blank lines from run_text: they are
                # structural separators attached to this run by
                # split_prose_bullet_runs (e.g. the blank line between a
                # bullet block and the prose that follows it). The model
                # summary replaces only the non-blank content; the blanks
                # that open the run stay in place.
                leading_blanks = run_text[: len(run_text) - len(run_text.lstrip("\n"))]
                run_parts.append(leading_blanks + stripped)
                prose_compressed = True
            assembled = "\n".join(run_parts)
            parts.append(f"{heading}\n{assembled}")
            # Counter semantics: prefer the outcome that best describes what
            # actually happened. A section that had at least one real
            # compression is counted as compressed even if other prose runs
            # fell back. A section where NO prose run was compressed but at
            # least one failed outright gets request_failed_count. The
            # section is left uncounted (no stat bump) only in the edge case
            # where all prose runs fell back due to empty/NOT-RELEVANT
            # responses and none failed outright -- that is a normal
            # (if unproductive) result, equivalent to the whole-section
            # NOT-RELEVANT path (which increments not_relevant_count, but
            # at run granularity we don't have per-run NOT-RELEVANT
            # tracking, so we leave the section uncounted rather than
            # misattribute it).
            if prose_compressed:
                compressed_count += 1
            elif prose_failed:
                request_failed_count += 1
            continue
        raw_summary = complete(
            oai_client, model,
            f"You compress one section of a structured handoff document for another AI to "
            f"read next. {_MAP_ANTI_CHAT_INSTRUCTION}\n\nFocus: {focus}",
            body,
        )
        # complete() returns None ONLY for a genuine request failure (see its
        # own docstring) -- checked here, before any stripping, so that a
        # real (if unhelpful) "" or whitespace-only response from a reachable
        # model is never conflated with the model being unreachable.
        if raw_summary is None:
            parts.append(f"{heading}\n{body}")
            request_failed_count += 1
            continue
        summary = raw_summary.strip()
        # A failed call, an empty/whitespace response, or a NOT_RELEVANT
        # verdict must not delete the section. Falling back to the original
        # body keeps the artifact complete at the cost of some ratio -- the
        # correct direction here. All three are tracked separately (not
        # folded into one "failed" bucket) precisely so the outage signal
        # below can tell "the request itself failed" apart from "the request
        # succeeded" (empty, or not relevant, either way a live response).
        if not summary:
            parts.append(f"{heading}\n{body}")
            empty_response_count += 1
            continue
        if "NOT RELEVANT" in summary.upper():
            parts.append(f"{heading}\n{body}")
            not_relevant_count += 1
            continue
        parts.append(f"{heading}\n{summary}")
        compressed_count += 1

    compressed = "\n\n".join(parts).strip()

    # Belt-and-suspenders: headings are emitted from the source above, so this
    # should never fire. It exists so that a future regression in this function
    # degrades to a bigger artifact rather than a lossy one.
    compressed = restore_missing_sections(text, compressed)
    if preserve_identifiers:
        compressed = append_missing_identifiers(text, compressed)
    # Last, so it covers every route content took to get here: the preamble, a
    # verbatim section, a failed-call fallback, and anything the model echoed.
    compressed = redact_and_disclose(compressed)

    if ctx is not None:
        await ctx.report_progress(progress=1, total=1, message="Done")

    ratio = round(100 * (1 - len(compressed) / original_len)) if original_len else 0
    notes = [f"{len(sections)} section(s)", f"{compressed_count} compressed"]
    if verbatim_count:
        notes.append(f"{verbatim_count} kept verbatim")
    if request_failed_count:
        notes.append(f"{request_failed_count} kept verbatim after a failed call")
    if empty_response_count:
        notes.append(f"{empty_response_count} kept verbatim after an empty response")
    if not_relevant_count:
        notes.append(f"{not_relevant_count} kept verbatim as not relevant")
    result = (
        f"[compressed {original_len} -> {len(compressed)} chars across "
        f"{', '.join(notes)}, ~{ratio}% smaller]\n\n{compressed}"
    )

    # Every section whose request was actually SENT came back an outright
    # failure, and nothing succeeded (not compressed, not a valid
    # NOT-RELEVANT verdict, not even an empty-but-live response) --
    # distinguish this from a normal, healthy run rather than returning the
    # same success-shaped string either way (issue #43). This is NOT
    # "request_failed_count == len(sections)": a section that's verbatim by
    # heading/structure is never sent to the model at all, so a document
    # with, say, four verbatim sections and two prose ones that both failed
    # would otherwise never trip a literal all-sections check, even though
    # every model call that WAS made failed outright. And it is deliberately
    # NOT triggered by not_relevant_count or empty_response_count alone (PR
    # #134 review, both rounds): either one means the request reached a live
    # model and got an actual response back, so a document whose attempted
    # sections are all judged not relevant, or all come back empty, is a
    # normal (if unproductive) run, not an outage -- conflating either with a
    # real request failure would misreport a healthy LM Studio as down.
    # compressed_count == 0 and not_relevant_count == 0 and
    # empty_response_count == 0 (nothing succeeded, in any of the three
    # non-failure ways) plus request_failed_count > 0 (at least one call was
    # actually attempted and failed outright) is the precise "every attempted
    # request actually failed" condition, regardless of how many sections
    # were verbatim. Deliberately prepended rather than appended, and
    # deliberately NOT made part of the "[compressed ...]" line itself:
    # compress_mcp_server.py's savings-tracker footer credits a result only
    # when it starts with "[compressed" (see _append_savings_footer) -- and
    # no compression actually happened here, so it must not qualify for
    # credit either.
    #
    # "no section was compressed" rather than "returned unchanged" in the
    # wording below: redact_and_disclose() (below, and unconditionally on
    # every path through this function) still runs over the whole result
    # regardless of why a section fell back to its original body, so a
    # credential-shaped value in a fallback section IS still replaced and
    # disclosed here exactly as it would be anywhere else -- "unchanged"
    # would be a false claim in that case (PR #134 review, round 3).
    if (
        compressed_count == 0
        and not_relevant_count == 0
        and empty_response_count == 0
        and request_failed_count > 0
    ):
        result = (
            "[LM Studio appears unreachable -- no section was compressed; "
            f"original content preserved]\n\n{result}"
        )
    return result


async def compress(
    text: str,
    focus: str = DEFAULT_FOCUS,
    skip_if_under_chars: int = 2000,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_total_chars: int = 2_000_000,
    max_chars: Optional[int] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    preserve_identifiers: bool = False,
    preserve_sections: bool = False,
    ctx=None,
) -> str:
    """
    Shared map-reduce compression logic. Returns the ORIGINAL text unchanged
    if it's under skip_if_under_chars, or an "Error: ..." string (never
    raises) if compression can't proceed -- callers should treat a return
    value starting with "Error:" as a signal to fall back to the original
    text rather than lose it.

    EXCEPTION to "unchanged" above (PR #135 review, round 2): with
    preserve_identifiers=True or preserve_sections=True, redact_and_disclose()
    still runs on this under-threshold path too, so a credential-shaped value
    is replaced and a "[N credential-shaped value(s) redacted...]" note
    appended even though no summarization happened. "Unchanged" means "not
    summarized," not "byte-for-byte identical regardless of content" -- the
    same distinction the "[LM Studio appears unreachable...]" wording below
    already draws for the total-outage fallback path.

    With preserve_sections=True, there is a THIRD possible return shape
    besides the original text and the usual "[compressed N -> M chars...]"
    success string: if every section whose request was actually attempted
    failed outright and nothing succeeded, the result is prepended with
    "[LM Studio appears unreachable -- no section was compressed; original
    content preserved]" (issue #43; see _compress_each_section's own
    docstring for the precise condition and why a NOT-RELEVANT verdict or an
    empty-but-live response don't count toward it -- and for why the wording
    says "preserved" rather than "unchanged": redact_and_disclose() still
    runs regardless, so a credential-shaped value is still replaced even in
    a fallback section, PR #134 review round 3). This is NOT an
    "Error:"-prefixed string -- the content is fully present and usable
    either way -- so a caller parsing only for "Error:" vs. everything-else
    still works unchanged; a caller that specifically wants to detect this
    distinct total-outage signal should check for that "[LM Studio appears
    unreachable" prefix instead
    (PR #134 review).

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

    Relevance classification is skipped entirely for a non-selective focus
    (see _looks_non_selective) -- "summarize this" asks for all of it, so
    there is no subset to filter down to and a NO-biased classifier can only
    subtract. This was measured, not assumed: with DEFAULT_FOCUS, a clean
    build log and a README chunk were both classified NOT relevant 8/8 at
    temperature=0, because the classifier read the focus's "preserve
    anything that looks like an error" clause as its selection criterion.
    The hook therefore compressed only FAILING output and returned every
    successful build, passing test run, and doc read raw.

    Always (not gated by any flag): if the source ends with a recognized
    totals/result line (a pytest/jest/go-test/gradle/dotnet/rspec-style
    "N passed, M failed", "BUILD SUCCESSFUL", etc.), it's restored verbatim
    if the summary dropped it (see append_trailing_summary_if_missing). This
    is the same "measure, don't just prompt harder" reasoning as
    preserve_identifiers below, applied to one specific fact that's cheap to
    guarantee unconditionally: a real pytest failure log's summary correctly
    listed all 3 failures but dropped the final "3 failed, 39 passed in
    8.72s" line. Fixing that by appending "always keep the final totals line"
    to the focus text worked for that case, but on a clean, all-passing
    build the SAME added instruction made the model list every individual
    passing test instead of aggregating them into "All tests PASSED"
    (208 -> 400+ chars) -- protecting one line by prompting made the model
    more conservative about compressing everything else. The deterministic
    check has no such side effect, because it never touches how the model
    summarizes.

    preserve_identifiers verifies, after compressing, that every
    identifier-shaped token in the source survived, and re-appends verbatim
    any that didn't (see append_missing_identifiers). Off by default because
    it's wrong for gist-style asks -- summarizing a web page shouldn't drag a
    list of its URLs along. Turn it on when the output is an artifact someone
    will later grep, which is what /my-compact produces.

    Why a deterministic check rather than a stronger prompt: the /my-compact
    focus already demands "Retain exact file paths, class/function/variable
    names ... without modification" and asks for a "context-lossless" summary,
    and a real 9,876-char handoff still came back with the Options segment
    dropped from a colon-delimited config key, three whole sections missing, and a stray
    triple-quote. At that size the text is ONE chunk, so the entire result is
    a single call to a 2-4B local model with no reduce step to catch anything.
    A summarizer cannot be prompted into being lossless; this measures the
    output instead of trusting it, the same way _EXACT_CMDS measures the
    command instead of trusting a size threshold.

    preserve_sections compresses a '## '-headed document one section at a time
    instead of in one pass, passing exactness-critical sections through
    untouched and re-emitting every heading from the source. It addresses the
    other half of the same reported failure: not a mangled identifier, but
    three ENTIRE sections disappearing while the result still read as complete.
    A single large call can only get shorter by discarding content, and whole
    sections are the cheapest thing to discard; per-section calls remove that
    pressure. Falls back to whole-text compression when there are no headings,
    so it is safe to leave on for input that turns out to be unstructured.

    The two flags compose, and /my-compact sets both: preserve_sections keeps
    the structure, preserve_identifiers keeps the exact strings inside it.
    Neither makes the result literally lossless -- prose inside a compressed
    section is still rewritten -- they bound the loss to within a section
    instead of across the document.
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
    # Whether the slice below actually dropped anything -- chars_limited only
    # means a limit was *selected*, not that it was reached. A short
    # positional input, or an explicit max_chars >= len(text), sets
    # chars_limited/auto_truncated but text[:max_chars] is a no-op. Computed
    # before the slice so the comparison is against the real original length.
    # Written as "max_chars is not None and ..." rather than "chars_limited
    # and ..." so mypy can narrow max_chars from Optional[int] to int within
    # this same expression -- it can't do that narrowing through a separate
    # bool variable, even though chars_limited means exactly the same thing.
    was_truncated = max_chars is not None and len(text) > max_chars
    if max_chars is not None:
        text = text[:max_chars]

    # Nothing to filter for -- the caller asked for all of it compressed, not
    # for a specific part. See _looks_non_selective for the measured failure
    # this avoids (DEFAULT_FOCUS rejecting every chunk of any output that
    # didn't happen to contain an error).
    non_selective = _looks_non_selective(focus)

    if len(text) < skip_if_under_chars:
        # `text` may already be a truncated slice (was_truncated), not the
        # true original, at this point -- returning it bare here would
        # contradict this function's own contract ("returns the ORIGINAL
        # text unchanged if under skip_if_under_chars") and look
        # indistinguishable from "the whole document was already this
        # short." Disclose truncation the same way the full pipeline does
        # further down (see auto_truncated's truncate_note below), whether
        # it was auto-detected from a positional focus or came from an
        # explicit max_chars the caller passed themselves -- issue #24
        # reproduced both as silently truncated with no note. Gated on
        # was_truncated, not chars_limited: a limit that was merely selected
        # but never reached (e.g. a short positional input, or an explicit
        # max_chars >= len(text)) truncated nothing, so a disclosure note
        # here would itself be a false claim (caught in PR #85 review).
        #
        # Redacts here in EITHER artifact-producing mode (PR #135 review,
        # Copilot, both rounds): this early return happens BEFORE the main
        # preserve_identifiers block AND before _compress_each_section
        # further down ever run, so a caller relying on either flag to get
        # credential redaction was leaving every input shorter than
        # skip_if_under_chars (the common case -- most command output is
        # short) completely unredacted. Gated on `preserve_identifiers or
        # preserve_sections`, not `preserve_identifiers` alone -- round 2
        # caught that preserve_sections=True by itself (no
        # preserve_identifiers) still leaked here, even though the NORMAL
        # (over-threshold) section path always redacts unconditionally
        # regardless of preserve_identifiers (see _compress_each_section).
        # "unchanged" in this function's own contract above means "not
        # summarized," not "byte-for-byte identical regardless of what it
        # contains" -- same distinction already drawn for the total-outage
        # fallback path elsewhere in this file, and now documented in this
        # function's own docstring too (PR #135 review, round 2).
        #
        # Captured BEFORE redact_and_disclose below can change text's length
        # (replacing a credential with a shorter/longer placeholder, or
        # appending a disclosure line) -- the truncate_note reports where
        # the SOURCE was cut, which redaction has no bearing on.
        truncated_len = len(text)
        if preserve_identifiers or preserve_sections:
            text = redact_and_disclose(text)
        if was_truncated:
            truncate_note = (
                f" [note: focus looked positional, so only the first {truncated_len} chars of the "
                "source were considered (auto-truncated) -- pass max_chars explicitly to override this]"
                if auto_truncated else
                f" [note: max_chars={max_chars} truncated the source to {truncated_len} chars before this result]"
            )
            return f"{text}{truncate_note}"
        return text

    if len(text) > max_total_chars:
        return (
            f"Error: input is {len(text)} chars, over the max_total_chars safety limit "
            f"({max_total_chars}). Refusing rather than silently dropping content -- "
            "pre-filter the input yourself or raise max_total_chars if you're sure."
        )

    # Cache check: skip model resolution and all LM Studio calls if this exact
    # (text, focus, model, max_chars, preserve_*) combination was compressed
    # earlier in this session. `model` here is the caller-supplied value (or
    # None), not the resolved id -- resolve_model() handles auto-detection
    # internally, and is called once BEFORE the cache key is computed below so
    # the resolved id is part of the key (two calls that auto-detect to the
    # SAME model share a cache entry; calls that resolve to DIFFERENT models do
    # not, even if the caller passed None for both).
    model, error = resolve_model(model, base_url)
    if error:
        return f"Error: {error}"

    # The resolved model id is now known -- compute the cache key and check
    # before any LM Studio COMPRESSION work. Note that resolve_model() above
    # may itself have contacted LM Studio (models.list() for auto-detection),
    # so a cache hit does not avoid that round-trip -- only the compress/
    # classify calls below are skipped. See _compression_cache_key for why
    # each argument is included or excluded.
    #
    # auto_truncated and max_chars are passed separately:
    # - By this point max_chars may have been mutated from None (positional
    #   auto-detect) to the integer boundary it detected -- so max_chars alone
    #   can no longer distinguish an auto-detected truncation from an explicit
    #   one equal to the same value. auto_truncated carries that distinction.
    # - effective_url (base_url or DEFAULT_BASE_URL) keeps two different LM
    #   Studio servers with the same model id from sharing a cache entry via
    #   the MCP tools' per-call base_url parameter.
    cache_key = _compression_cache_key(
        text, focus, model, max_chars, preserve_identifiers, preserve_sections,
        chunk_chars, auto_truncated, base_url or DEFAULT_BASE_URL,
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        # Return the cached result directly. Only the LM Studio compression
        # calls are skipped -- resolve_model() above already ran (and may have
        # called models.list() for auto-detection), so a cache hit does NOT
        # avoid that round-trip. ctx progress reporting is also skipped:
        # there's nothing to report progress on since no LM Studio work runs.
        # The caller sees the same string it would have gotten from a live
        # compression call.
        return cached

    original_len = len(text)
    oai_client = client(base_url)

    if preserve_sections:
        section_result = await _compress_each_section(
            text, focus, oai_client, model, base_url, preserve_identifiers, ctx
        )
        if section_result is not None:
            # Only cache fully successful section runs. Two cases skip caching:
            #
            # 1. Total outage: a result prepended with "[LM Studio appears
            #    unreachable ...]" means every model request that was SENT failed
            #    outright. Caching this would make repeated calls keep returning
            #    the degraded preserved-content output instead of retrying once
            #    LM Studio recovers. The "[LM Studio appears unreachable" prefix
            #    is the one and only marker _compress_each_section emits for this
            #    case, so checking for it here is both necessary and sufficient.
            #
            # 2. Partial failure: a result containing "kept verbatim after a
            #    failed call" means at least one prose section had a transient
            #    model failure and fell back to its original body. Caching this
            #    would permanently suppress retries for those sections -- the
            #    next call would serve the preserved-content fallback instead of
            #    potentially getting a real summary for the sections that failed.
            #    Detected via _compress_each_section's own disclosed fallback
            #    note, which is always present in the summary stats line when
            #    any call actually failed (distinct from verbatim-by-structure,
            #    which reports "kept verbatim" without the "after a failed call"
            #    suffix, and from empty/NOT-RELEVANT responses, which report
            #    "after an empty response" / "as not relevant" -- neither of
            #    those indicate a transient failure worth retrying).
            _is_fully_successful = (
                not section_result.startswith("[LM Studio appears unreachable")
                and "kept verbatim after a failed call" not in section_result
            )
            if _is_fully_successful:
                _cache_put(cache_key, section_result)
            return section_result
        # No headings found -- fall through to ordinary whole-text compression.

    chunks = chunk_text(text, chunk_chars)

    # Phase 1: relevance classification.
    # When classification is needed (selective focus, not already scoped by
    # max_chars), fire all chunks concurrently rather than waiting for each
    # round-trip before starting the next. classify_relevant() is synchronous
    # (uses the blocking openai client), so we dispatch each call on the
    # default thread-pool executor and gather the results -- same semantics
    # as running them in parallel threads, but compatible with the surrounding
    # async function. See issue #59.
    #
    # Note: actual parallelism is bounded by the executor's thread pool size
    # and by LM Studio's own request concurrency limit, so the speedup is
    # "significantly fewer round-trip latencies than N sequential calls,"
    # not a guaranteed "exactly one." In practice either bound is far above
    # the few chunks most real inputs produce.
    #
    # Bypass conditions (chars_limited or non_selective) are unchanged: if
    # classification is skipped entirely, relevance_results stays None and the
    # extraction loop treats every chunk as relevant, exactly as before.
    relevance_results: Optional[list] = None  # None -> classify-all bypass active
    if not (chars_limited or non_selective) and len(chunks) > 0:
        if ctx is not None:
            await ctx.report_progress(
                progress=0, total=len(chunks) + 1,
                message=f"Checking relevance ({len(chunks)} chunk(s), concurrent)",
            )
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(
                None, classify_relevant,
                oai_client, model, focus, chunk, i, len(chunks), auto_truncated,
            )
            for i, chunk in enumerate(chunks, 1)
        ]
        relevance_results = list(await asyncio.gather(*futures))
        # Check for any None (request failure) before proceeding.
        # Report the first failed chunk index for parity with the old error.
        for i, result in enumerate(relevance_results, 1):
            if result is None:
                return (
                    f"Error: LM Studio request failed while checking relevance "
                    f"(chunk {i}/{len(chunks)}) -- check it's still running at "
                    f"{base_url or DEFAULT_BASE_URL}."
                )

    # Phase 2: extraction -- sequential, in document order, for relevant chunks.
    chunk_summaries = []
    skipped = 0
    for i, chunk in enumerate(chunks, 1):
        if relevance_results is not None:
            relevant: Optional[bool] = relevance_results[i - 1]
            # None already handled above; only True/False arrive here.
        else:
            # chars_limited or non_selective -- classification was bypassed.
            relevant = True
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

    compressed: Optional[str]
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

    # Unconditional (unlike preserve_identifiers/preserve_sections below,
    # which are opt-in): this repairs one specific, high-value fact -- the
    # run's own final result -- rather than asserting general losslessness,
    # so it's cheap and safe to always check. Checked against `text` (the
    # post-truncation slice actually summarized), matching preserve_identifiers'
    # reasoning just below: a result line outside the window was never in
    # scope to begin with.
    compressed = append_trailing_summary_if_missing(text, compressed)

    if preserve_identifiers:
        # Checked against `text` (post-truncation) rather than the pre-max_chars
        # original: an identifier outside the window was never in scope, and
        # re-appending it would assert coverage the summary never claimed.
        compressed = append_missing_identifiers(text, compressed)
    # Redaction is gated on EITHER preservation flag, not preserve_identifiers
    # alone (PR #135 review, round 4): preserve_sections=True falls through
    # to this same ordinary whole-text path whenever _compress_each_section
    # finds no headings to split on (see the `if preserve_sections:` block
    # above) -- so with preserve_sections=True and preserve_identifiers left
    # at its default False, this whole-text fallback used to skip redaction
    # entirely, even though preserve_sections is just as much an artifact-
    # producing mode as preserve_identifiers is. Reproduced directly: 3000
    # chars of unstructured (no-heading) text with preserve_sections=True,
    # preserve_identifiers=False, and a model that echoes a credential back
    # came out with the secret intact. Identifier RESTORATION stays gated on
    # preserve_identifiers alone, per its own docstring -- only the
    # redaction step widens to cover both flags, mirroring
    # _compress_each_section's own unconditional (relative to
    # preserve_identifiers) redaction on the path that DOES find headings.
    if preserve_identifiers or preserve_sections:
        # A model can echo a secret out of its own input regardless of
        # which path produced `compressed`.
        compressed = redact_and_disclose(compressed)

    if ctx is not None:
        await ctx.report_progress(progress=1, total=1, message="Done")

    ratio = round(100 * (1 - len(compressed) / original_len)) if original_len else 0
    skip_note = f", {skipped} chunk(s) skipped as not relevant to the focus" if skipped else ""
    truncate_note = (
        f" [note: focus looked positional, so only the first {original_len} chars of the "
        "source were considered -- pass max_chars explicitly to override this]"
        if auto_truncated else ""
    )
    result = f"[compressed {original_len} -> {len(compressed)} chars across {len(chunks)} chunk(s){skip_note}, ~{ratio}% smaller]{truncate_note}\n\n{compressed}"
    _cache_put(cache_key, result)
    return result
