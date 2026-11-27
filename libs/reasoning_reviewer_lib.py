"""
The reasoning stage of the `codebase-reviewer` server: build each reasoning
reviewer's prompt, feed it a diff that has been cut to fit its context, and
collect what every reviewer found.

What this module does, in the order a review uses it:

  1. `split_diff` / `chunk_file_diff` cut a diff per file, then per hunk, so
     each piece fits a reviewer's context, and show each file as its NEW version
     with absolute line numbers (removed lines become a one-line marker). A cut
     never lands inside a line, and every chunk starts with its `File:` line so
     the model knows which file it is reading.
  2. `instruction_files_for` gathers the repo's own review rules: the
     `.github/instructions/*.instructions.md` files whose `applyTo` matches a
     changed path (the same files Copilot reads). `build_system_prompt` puts the
     reviewer's lens, those rules, any recalled lessons and the required output
     format together.
  3. `run_reasoning_review` runs the active reasoning reviewers through
     `reviewer_scheduler_lib.run_stages` (so grouping, parallelism and model
     loading are the scheduler's), and `collect_findings` keeps EVERY finding
     from EVERY reviewer: there is no vote threshold, so a finding one reviewer
     alone saw is not lost. Findings are grouped by class and duplicates merge.

Findings at this stage are unverified and unclassified: `verifier_reviewer_lib`
and `classifier_reviewer_lib` are the stages that go on from here.

Standard library only, apart from the sibling libs. The model client and the
completion call are passed in, so tests run with a stub and no LM Studio.
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import openai_lib
from credential_redaction_lib import display_safe
from codebase_reviewer_config_lib import DEFAULT_REPLY_FORMAT, Reviewer, ReviewersConfig, load_prompt
from lmstudio_model_loader_lib import ModelLoader
from reply_translator_lib import TranslatorError, translate
from reviewer_db_lib import NULL_RECORDER, hash_text
from reviewer_need_lib import ASKED, LIMIT, NeedSession, need_instructions, reference_budget
from reviewer_scheduler_lib import run_stages

# Sizing. A token count depends on the model's vocabulary, which is not known
# here, so chunk sizes use a deliberately low characters-per-token guess: a
# chunk that is a little small costs one more request, one that is too big is
# refused by LM Studio (an HTTP 400). `chunk_chars` overrides the whole
# calculation for a caller that knows better, except that for a reviewer with
# `tools` it is a ceiling: it never takes the room kept for the lookups' results.
CHARS_PER_TOKEN_GUESS = 3
# Assumed when a reviewer sets no `contextLength` (a model the user loaded keeps
# whatever context they gave it, which the server cannot see).
ASSUMED_CONTEXT_TOKENS = 8192
# Share of the context kept back for the model's reply.
REPLY_RESERVE_DIVISOR = 4
MIN_CHUNK_CHARS = 500
# A diff that needs more chunks than this per reviewer is refused with advice to
# review a smaller range, rather than sent as hundreds of requests.
MAX_CHUNKS = 100
# Each rules file (instructions, skill) is cut to this many characters.
MAX_RULES_CHARS = 20000
# A quote shorter than this (after whitespace is squeezed) says nothing about where the code is: `if` or
# `x` is in every file. The verifier stage's grounding step treats one as ungrounded, and merging two
# reports of a claim must not pick one.
MIN_QUOTE_CHARS = 3

UNCLASSIFIED_CLASS = "uncategorized"


class ReviewError(Exception):
    """The review cannot run at all (as opposed to one reviewer failing)."""


@dataclass(frozen=True)
class DiffSection:
    path: str
    text: str


@dataclass(frozen=True)
class Alternate:
    """Another claim about the same place and class, kept beside the finding it was merged into
    instead of being dropped, with the reviewers who made it."""

    claim: str
    failing_input: str
    reviewers: tuple
    line_start: int
    line_end: int
    # The code the claim accuses, as the reviewer copied it; "" when it gave none.
    quote: str = ""


@dataclass(frozen=True)
class Finding:
    file: str
    line_start: int
    line_end: int
    cls: str
    claim: str
    failing_input: str
    # The reviewers who made THIS claim, in the order they were merged.
    reviewers: tuple
    # Different claims (or the same claim with a different failing input) at overlapping lines in
    # the same class: two defects can share a class, so the second is never discarded.
    alternates: tuple = ()
    # The exact line (or short span) of code the claim accuses, copied by the reviewer. The verifier
    # stage checks that it occurs in the cited file before any model reads the finding; "" means the
    # reviewer gave none, which that check treats as ungrounded.
    quote: str = ""


@dataclass(frozen=True)
class FindingGroup:
    cls: str
    findings: tuple


@dataclass
class ReplyCount:
    """How one reviewer's requests went, in every stage: how many it sent, how many got a reply, and how many of those replies
    the reader could not read (a reply that is not in the stage's format, a translator that failed, a verdict that does not meet
    the standard, no usable probability). A reply that asks for a lookup (`need:`) is read as such, so it counts as readable.
    The readable ones are `replies - unreadable`. It is what the structured results file reports as "how many replies were
    readable" (`libs/reviewer_files_lib.py`), so a reviewer that answered but whose answers could not be read is told apart from
    one that found nothing."""

    requests: int = 0
    replies: int = 0
    unreadable: int = 0

    @property
    def readable(self) -> int:
        return self.replies - self.unreadable


@dataclass
class ReviewerResult:
    """What one reviewer produced across all of its chunks."""

    reviewer: str
    findings: list = field(default_factory=list)
    # Problems that did not lose the whole reviewer: a malformed reply, an item
    # skipped, one chunk whose request failed.
    problems: list = field(default_factory=list)
    # Notes that are not problems: lookups of the `need:` loop that were cut to fit the reviewer's budget, or a reviewer with
    # `tools` that ran without them.
    warnings: list = field(default_factory=list)
    chunks: int = 0
    counts: ReplyCount = field(default_factory=ReplyCount)


@dataclass
class ReasoningReview:
    groups: list = field(default_factory=list)
    # One line per reviewer that produced nothing usable, or per problem.
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    reviewers_run: list = field(default_factory=list)
    chunks: int = 0
    # The scheduler's outcome for every reasoning reviewer that was asked, in the order they ran (`ReviewerOutcome`: the reviewer,
    # its `ReviewerResult` or the error that stopped it). `groups` are the findings after merging; this keeps each reviewer's own
    # findings, which the structured results file saves so that a process can merge several reviewers' results itself.
    outcomes: list = field(default_factory=list)

    @property
    def finding_count(self) -> int:
        return sum(len(g.findings) for g in self.groups)


# ---------------------------------------------------------------------------
# Splitting a diff
# ---------------------------------------------------------------------------

_DIFF_HEADER = re.compile(r"^diff --git ", re.MULTILINE)
_HUNK_HEADER = re.compile(r"^@@ ", re.MULTILINE)


_GIT_ESCAPES = {"a": 7, "b": 8, "f": 12, "n": 10, "r": 13, "t": 9, "v": 11, "\\": 92, '"': 34}


def _unquote_git_path(token: str) -> str:
    """A path as git prints it. Git wraps a name in double quotes, with C-style escapes, when it holds
    a non-ASCII byte (as octal `\\303\\251`), a quote, a backslash or a control character, so
    `café.py` appears as `"caf\\303\\251.py"`. An unquoted name is returned as it is."""
    token = token.strip()
    if len(token) < 2 or not (token.startswith('"') and token.endswith('"')):
        return token
    body = token[1:-1]
    raw = bytearray()
    i = 0
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body):
            following = body[i + 1]
            if following in "01234567":
                j = i + 1
                digits = ""
                while j < len(body) and len(digits) < 3 and body[j] in "01234567":
                    digits += body[j]
                    j += 1
                raw.append(int(digits, 8) & 0xFF)
                i = j
                continue
            if following in _GIT_ESCAPES:
                raw.append(_GIT_ESCAPES[following])
                i += 2
                continue
        raw.extend(char.encode("utf-8"))
        i += 1
    return raw.decode("utf-8", errors="replace")


def _strip_prefix(path: str, prefix: str) -> str:
    return path[len(prefix):] if path.startswith(prefix) else path


def _header_new_path(header: str) -> str:
    """The new path from a `diff --git <old> <new>` line, quoted or not; "" when it cannot be read."""
    rest = header[len("diff --git "):] if header.startswith("diff --git ") else ""
    if not rest:
        return ""
    if rest.startswith('"'):
        end = 1
        while end < len(rest) and not (rest[end] == '"' and rest[end - 1] != "\\"):
            end += 1
        second = rest[end + 1:].strip()
    elif ' "b/' in rest:
        second = rest[rest.index(' "b/') + 1:].strip()
    else:
        match = re.match(r"a/(.+) b/(.+)$", rest)
        second = ("b/" + match.group(2)) if match else ""
    return _strip_prefix(_unquote_git_path(second), "b/") if second else ""


def _section_path(text: str) -> str:
    """The path a diff section is about: the new path, or the old one for a deletion. Quoted names
    (non-ASCII, quotes, control characters) are decoded."""
    new = re.search(r"^\+\+\+ (.+)$", text, re.MULTILINE)
    if new and new.group(1).strip() != "/dev/null":
        return _strip_prefix(_unquote_git_path(new.group(1)), "b/")
    old = re.search(r"^--- (.+)$", text, re.MULTILINE)
    if old and old.group(1).strip() != "/dev/null":
        return _strip_prefix(_unquote_git_path(old.group(1)), "a/")
    header = text.split("\n", 1)[0]
    # A binary, renamed-only or mode-only file has no ---/+++ lines: read the `diff --git` line.
    return _header_new_path(header) or header


def split_diff(diff_text: str) -> list:
    """The diff cut into one `DiffSection` per file, in the order they appear.

    Text before the first `diff --git` line (there is none in `git diff`
    output) is ignored. An empty diff is no sections."""
    starts = [m.start() for m in _DIFF_HEADER.finditer(diff_text)]
    sections = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(diff_text)
        text = diff_text[start:end]
        sections.append(DiffSection(path=_section_path(text), text=text))
    return sections


_NUMBERED_LINE = re.compile(r"^( *\d+) [+ ] ")


def _cut_lines(text: str, limit: int) -> list:
    """`text` (rendered lines, each ending in a newline) as pieces of at most `limit` characters,
    cut between lines. A single line longer than `limit` is split into fragments: the first keeps its
    line number, and every later fragment repeats that number with a `>` mark (`   12 > ...`), so a
    request holding only a fragment still knows which line it belongs to."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    fitted: list = []
    for line in lines:
        if len(line) + 1 <= limit:
            fitted.append(line + "\n")
            continue
        numbered = _NUMBERED_LINE.match(line)
        continuation = f"{numbered.group(1)} > " if numbered else ""
        fitted.append(line[: limit - 1] + "\n")
        rest = line[limit - 1:]
        room = max(limit - 1 - len(continuation), 1)
        while rest:
            fitted.append(continuation + rest[:room] + "\n")
            rest = rest[room:]
    pieces: list = []
    current = ""
    for line in fitted:
        if current and len(current) + len(line) > limit:
            pieces.append(current)
            current = ""
        current += line
    if current:
        pieces.append(current)
    return pieces


_HUNK_LINE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
# Shown between two hunks that are not adjacent in the file.
GAP_MARKER = "      ..."


def _render_section(section: DiffSection) -> tuple:
    """`(title, hunks)`: one file's change shown as its NEW version, with line numbers.

    Each line a reviewer sees is `<new line number> <+ or blank> <code>`: `+` for a line the
    change added or modified, blank for unchanged context. A run of removed lines becomes one
    `~ N line(s) removed here` marker, so a model never reads deleted code as current code (a
    unified diff's `-` lines were taken for current code in live runs). Because every line
    carries its own number, a hunk cut into several chunks still gives absolute positions.

    A hunk that only deletes lines and has no surviving line beside it (`context_lines=0`) would
    show nothing at all, so its removed lines ARE listed, with `-`, after
    `~ N line(s) removed after line L`: the question there is whether removing them was safe.

    `hunks` is empty when the file has nothing to review: binary, mode-only, renamed without
    edits, or deleted (no new version to read). `run_reasoning_review` reports those files."""
    # Split on "\n" only: str.splitlines() would also split inside a code line holding a form feed or similar.
    lines = section.text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    title = f"File: {section.path}"
    header = lines[: next((i for i, line in enumerate(lines) if _HUNK_LINE.match(line)), len(lines))]
    if any(line.startswith("deleted file mode") or line.strip() == "+++ /dev/null" for line in header):
        return title, []
    if any(line.startswith("new file mode") for line in header):
        title += "\n(new file)"
    hunks: list = []
    current: Optional[list] = None
    removed_texts: list = []  # the removed lines of the run in progress
    removed_in_hunk: list = []
    new_line = 0
    hunk_start = 0
    numbered_in_hunk = False
    reviewable = False

    def flush_run() -> None:
        nonlocal removed_texts
        if current is not None and removed_texts:
            current.append(f"      ~ {len(removed_texts)} line(s) removed here")
        removed_texts = []

    def close() -> None:
        nonlocal numbered_in_hunk, removed_in_hunk, reviewable
        if current is None:
            return
        flush_run()
        if not numbered_in_hunk and removed_in_hunk:
            # Nothing of this hunk survives next to the removal: list what was removed.
            lead = [current[0]] if current and current[0] == GAP_MARKER else []
            current[:] = lead + [f"      ~ {len(removed_in_hunk)} line(s) removed after line {hunk_start}:"] + [f"      - {text}" for text in removed_in_hunk]
            reviewable = True
        hunks.append("\n".join(current) + "\n")
        removed_in_hunk = []
        numbered_in_hunk = False

    for line in lines:
        start = _HUNK_LINE.match(line)
        if start:
            close()
            current = [GAP_MARKER] if hunks else []
            new_line = hunk_start = int(start.group(1))
            continue
        if current is None or line.startswith("\\"):
            continue
        if line.startswith("-"):
            removed_texts.append(line[1:])
            removed_in_hunk.append(line[1:])
            continue
        flush_run()
        mark = "+" if line.startswith("+") else " "
        current.append(f"{new_line:>5} {mark} {line[1:]}")
        numbered_in_hunk = True
        reviewable = True
        new_line += 1
    close()
    return title, (hunks if reviewable else [])


def chunk_file_diff(section: DiffSection, max_chars: int) -> list:
    """`section` as chunks of at most about `max_chars` characters, each showing the file's new
    version with line numbers (see `_render_section`). No chunks when there is nothing to review.

    A file that fits is one chunk. Otherwise the cut is at hunk boundaries, and a hunk bigger
    than the limit is cut at line boundaries (only a single line over the limit is cut inside
    the line); every chunk starts with the `File:` line so a reviewer always knows the file. A
    chunk is the title plus at least one piece even when that is over the limit, rather than
    dropping the title."""
    title, hunks = _render_section(section)
    if not hunks:
        return []
    header = title + "\n"
    room = max(max_chars - len(header), MIN_CHUNK_CHARS)
    pieces: list = []
    for hunk in hunks:
        pieces.extend(_cut_lines(hunk, room) if len(hunk) > room else [hunk])
    chunks: list = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > room:
            chunks.append(header + current)
            current = ""
        current += piece
    if current:
        chunks.append(header + current)
    return chunks


def chunk_diff(diff_text: str, max_chars: int) -> list:
    """Every chunk of the whole diff, in file order."""
    chunks: list = []
    for section in split_diff(diff_text):
        chunks.extend(chunk_file_diff(section, max_chars))
    return chunks


def chunk_chars_for(reviewer: Reviewer, system_prompt: str, override: Optional[int] = None, reserve: int = 0) -> int:
    """How many characters of diff one request to `reviewer` can carry.

    `override` (the caller's `chunk_chars`) wins when there is no `reserve`, exactly as given. Otherwise: the reviewer's
    `contextLength` (or `ASSUMED_CONTEXT_TOKENS`), minus a share kept for the
    reply, at `CHARS_PER_TOKEN_GUESS` characters a token, minus the system
    prompt. Raises `ReviewError` when the prompt alone leaves less than
    `MIN_CHUNK_CHARS`, which is the reviewer's context being too small for its
    own prompt, not something chunking can fix.

    `reserve` is room kept back for something else in the same request (the `need:` loop's reference block); it is
    taken off before anything is sized, and 0 for every reviewer without `tools`, so their sizes are what they always were. With
    a `reserve`, an `override` is a ceiling: it is used as given when it fits in what the reserve leaves, and is cut to that room
    when it does not (a follow-up request carries the chunk and the reference block, so a chunk that took the reserve's room would
    send a request larger than the sizing allows). When the room left is under `MIN_CHUNK_CHARS` and the override needs cutting,
    it is refused as above."""
    if override is not None and not reserve:
        return max(override, 1)
    context = reviewer.context_length or ASSUMED_CONTEXT_TOKENS
    usable_tokens = context - context // REPLY_RESERVE_DIVISOR
    room = usable_tokens * CHARS_PER_TOKEN_GUESS - len(system_prompt) - reserve
    if override is not None and max(override, 1) <= room:
        return max(override, 1)
    if room < MIN_CHUNK_CHARS:
        kept = f" and {reserve} kept for what its lookups return" if reserve else ""
        raise ReviewError(
            f"reviewer {reviewer.name!r}: its prompt and rules take about {len(system_prompt)} characters{kept}, "
            f"which leaves under {MIN_CHUNK_CHARS} for the diff in a {context}-token context; "
            "raise its contextLength, shorten its prompt, or pass chunk_chars"
        )
    return room


def loop_enabled(reviewer: Reviewer, executor: object) -> bool:
    """Whether `reviewer` gets the `need:` feedback loop: it has `tools` and there is an executor to run them."""
    return bool(reviewer.tools) and executor is not None


def need_format_for(reviewer: Reviewer, role: str) -> str:
    """The format a reviewer asks for lookups in: the one it replies in, so it is never asked to switch. A verifier always
    replies in JSON; a reasoning reviewer in its `replyFormat`."""
    return "json" if role == "verifier" else reviewer.reply_format


def instructions_from_file(reviewer: Reviewer, role: str) -> bool:
    """Whether the reviewer's `promptFile` carries `role`'s instructions for how to reply, and so for how to ask for lookups,
    in place of the built-in ones.

    Always for a reasoning reviewer (its file replaces the built-in reply instructions). For a verifier only when verifier is the
    reviewer's ONLY type: a reviewer that is also a reasoning reviewer shares its file with that role, whose reply instructions
    would not suit the verifier, so its verifier keeps the built-in instructions after the file, as it always did. False
    for a reviewer without a `promptFile`."""
    if not reviewer.prompt_file:
        return False
    return role == "reasoning" or tuple(reviewer.roles) == ("verifier",)


def with_need_instructions(reviewer: Reviewer, system: str, executor: object, role: str = "reasoning") -> str:
    """`system`, plus the section that tells a tools reviewer how to ask (in the format it replies in). Unchanged (the same
    string) when the loop is off, and when the reviewer's `promptFile` carries the instructions (`instructions_from_file`): then
    the file has to say how to ask, and the server adds nothing after it."""
    if not loop_enabled(reviewer, executor) or instructions_from_file(reviewer, role):
        return system
    return system + "\n\n" + need_instructions(reviewer.tools, need_format_for(reviewer, role), role)


def loop_reserve(reviewer: Reviewer, system: str, executor: object) -> int:
    """The characters of `reviewer`'s room reserved for what its lookups return (`contextShare` of what its system prompt
    leaves), 0 when the loop is off. `system` is the prompt as it will be sent, instructions included."""
    if not loop_enabled(reviewer, executor):
        return 0
    return reference_budget(reviewer, chunk_chars_for(reviewer, system))


def loop_unavailable(reviewer: Reviewer, executor: object) -> Optional[str]:
    """A warning when `reviewer` has `tools` but there is nothing to run them (so it runs as if it had none), else None."""
    if reviewer.tools and executor is None:
        return f"reviewer {reviewer.name!r} has 'tools' but no tool executor was given, so it ran without lookups"
    return None


# ---------------------------------------------------------------------------
# The repo's review rules
# ---------------------------------------------------------------------------


# An `applyTo` glob longer than this is ignored (matches nothing): real ones are a few dozen characters.
MAX_GLOB_CHARS = 512


def _wild_match(pattern: str, text: str) -> bool:
    """`pattern` (`*` any run of characters, `?` one character, anything else literal) against one path
    segment. The classic two-pointer match: it remembers only the last `*` instead of trying every
    split, so the work is at most len(pattern) * len(text), never exponential."""
    pi = ti = 0
    star = -1
    mark = 0
    while ti < len(text):
        if pi < len(pattern) and pattern[pi] == "*":
            star = pi
            mark = ti
            pi += 1
        elif pi < len(pattern) and (pattern[pi] == "?" or pattern[pi] == text[ti]):
            pi += 1
            ti += 1
        elif star != -1:
            pi = star + 1
            mark += 1
            ti = mark
        else:
            return False
    while pi < len(pattern) and pattern[pi] == "*":
        pi += 1
    return pi == len(pattern)


def _glob_match(pattern: str, path: str) -> bool:
    """Whether a GitHub-style `applyTo` glob matches a posix relative path.

    `*` matches within one directory, a `**` segment matches any number of directories (none at all
    included, so `**/*.py` matches `a.py`), `?` one character, and everything else is literal. A
    segment that merely contains `**` (`**.py`) reads as `**/` plus that segment with `*`.

    This replaces a regex translation that backtracked exponentially: a handful of `**/` segments,
    or a handful of `*a` runs inside one segment, made a small instructions file stall the whole
    review. Here the work is bounded by the number of pattern segments times path segments (each
    compared in bounded time), and `**` positions are memoized."""
    if len(pattern) > MAX_GLOB_CHARS:
        return False
    segments: list = []
    for segment in pattern.split("/"):
        if segment == "**":
            if not segments or segments[-1] != "**":  # `**/**` is `**/`
                segments.append("**")
        elif "**" in segment:
            if segment.startswith("**") and (not segments or segments[-1] != "**"):
                segments.append("**")
            segments.append(re.sub(r"\*+", "*", segment))
        else:
            segments.append(segment)
    parts = path.split("/")
    memo: dict = {}

    def match(si: int, pi: int) -> bool:
        key = (si, pi)
        if key in memo:
            return memo[key]
        if si == len(segments):
            result = pi == len(parts)
        elif segments[si] == "**":
            if si == len(segments) - 1:
                result = pi < len(parts)  # a trailing `**` covers at least one more path segment
            else:
                result = any(match(si + 1, k) for k in range(pi, len(parts) + 1))
        elif pi == len(parts):
            result = False
        else:
            result = _wild_match(segments[si], parts[pi]) and match(si + 1, pi + 1)
        memo[key] = result
        return result

    try:
        return match(0, 0)
    except RecursionError:  # a path hundreds of directories deep: treat as no match, not a crash
        return False


def _split_front_matter(text: str) -> tuple:
    """`(fields, body)` for a markdown file with `---` front matter.

    Only flat `key: value` lines are read (that is all `applyTo` and `name`
    need); no front matter is `({}, text)`."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            fields = {}
            for line in lines[1:index]:
                key, sep, value = line.partition(":")
                if sep:
                    fields[key.strip()] = value.strip().strip("\"'")
            return fields, "".join(lines[index + 1 :])
    return {}, text


def applies_to(apply_to: Optional[str], changed_paths: Iterable[str]) -> bool:
    """Whether an instructions file with this `applyTo` covers any changed path.

    A missing or blank `applyTo` applies to every path (assumed to be how
    Copilot treats a file without one). The value is comma-separated globs."""
    paths = [p.replace("\\", "/") for p in changed_paths]
    if not apply_to or not apply_to.strip():
        return True
    patterns = [p.strip() for p in apply_to.split(",") if p.strip()]
    return any(_glob_match(pattern, path) for pattern in patterns for path in paths)


def _bounded(text: str) -> str:
    if len(text) <= MAX_RULES_CHARS:
        return text
    return text[:MAX_RULES_CHARS] + "\n[truncated]"


# Upper bound on the bytes read from a rules file. `_bounded` cuts the TEXT to MAX_RULES_CHARS afterwards,
# but reading must stop first: a link to a device such as /dev/zero would otherwise never finish.
MAX_READ_BYTES = MAX_RULES_CHARS * 4


GIT_TIMEOUT_SECONDS = 10


def _is_tracked(root: Path, relative: Path) -> bool:
    """Whether git tracks `relative` (a path inside `root`). Anything uncertain (no git, not a
    repository, a timeout) counts as not tracked, so a doubtful link target is refused."""
    try:
        done = subprocess.run(
            ["git", "--literal-pathspecs", "-C", str(root), "ls-files", "--error-unmatch", "--", relative.as_posix()],
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _read_text(path: Path, root: Optional[Path] = None) -> Optional[str]:
    """The text of a rules file, or None when it cannot be used.

    With `root` (every automatically discovered rules file), the path is resolved first, so a symlink
    to a file, or through a directory, outside the repository is refused: the repository controls
    these files, and a link must not send a private file to the model. Staying inside the checkout is
    not enough: a link to `.env` or to anything under `.git/` would still read a local secret, so a
    link must also point at a file git tracks (an in-repository link to a tracked rules file is fine),
    and nothing under `.git` is ever read. The target must be a regular file, and at most
    MAX_READ_BYTES are read. A prompt file the user configures explicitly is read elsewhere and is
    not subject to this."""
    try:
        target = path.resolve(strict=True)
        if root is not None:
            root_real = root.resolve()
            if not target.is_relative_to(root_real):
                return None
            relative = target.relative_to(root_real)
            if relative.parts[:1] == (".git",):
                return None
            if target != root_real / path.relative_to(root) and not _is_tracked(root_real, relative):
                return None
        if not target.is_file():
            return None
        with open(target, "rb") as handle:
            data = handle.read(MAX_READ_BYTES + 1)
    except (OSError, RuntimeError):
        # RuntimeError: before Python 3.13, Path.resolve() reports a symlink loop as RuntimeError
        # rather than OSError, and a self-linked rules file must be skipped, not abort the review.
        return None
    truncated = len(data) > MAX_READ_BYTES
    data = data[:MAX_READ_BYTES]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut by the read limit is not a bad file.
        if truncated and exc.start >= len(data) - 3:
            return data[: exc.start].decode("utf-8")
        return None


def instruction_files_for(repo_root: "str | Path", changed_paths: Sequence[str]) -> list:
    """`(relative path, text)` of each `.github/instructions/*.instructions.md`
    whose `applyTo` matches at least one changed path, in file-name order. The
    text is the file's body (its front matter is dropped). An unreadable file
    is skipped."""
    root = Path(repo_root)
    found = []
    for path in sorted((root / ".github" / "instructions").glob("*.instructions.md")):
        text = _read_text(path, root)
        if text is None:
            continue
        fields, body = _split_front_matter(text)
        if applies_to(fields.get("applyTo"), changed_paths):
            found.append((path.relative_to(root).as_posix(), _bounded(body.strip())))
    return found


INPUT_FORMAT = """\
Each request shows one file's change as the file's NEW version: a "File:" line, then lines with their
line numbers on the left. A "+" after the number marks a line this change added or modified; an
unmarked line is unchanged context. "~ N line(s) removed here" marks where lines were deleted: they
are gone, so never report on them. A ">" in place of the "+" continues a line that was too long for one
request. "..." means lines were skipped. When a change only deletes code, the deleted lines are listed
with a "-": judge whether removing them is safe. Report defects that this change introduces or exposes."""

OUTPUT_FORMAT = """\
Reply with ONLY a JSON object, no other text:
{"findings": [{"file": "<path from the File: line>", "line_start": <int>, "line_end": <int>,
  "class": "<short-kebab-case name for the kind of defect>",
  "claim": "<what is wrong, in one or two sentences>",
  "quote": "<the exact line of code the claim is about, copied character for character from the file>",
  "failing_input": "<a concrete input or state that makes it fail>"}]}
Report each defect you can describe with a concrete failing input. Use the line numbers printed at the left of each line.
The quote is checked against the file: copy the code itself, without the line number or the "+" mark, and do not paraphrase it.
A finding whose quote is not in the file is thrown away.
If you find nothing, reply {"findings": []}."""

# The plain-lines reply format, the default. The `quote` field carries code, which is full of `"`,
# `\` and string joins, and small models often fail to escape it inside a JSON string; a line that ends where
# the code ends has nothing to escape. These are the exact instructions the format experiments used.
OUTPUT_FORMAT_LINES = """\
Reply with ONLY finding blocks, no other text. Write one block for each defect. A block is these lines, in this order:
FINDING
file: <path from the File: line>
lines: <first line>-<last line>, using the line numbers printed at the left
class: <short-kebab-case name for the kind of defect>
claim: <what is wrong, in one or two sentences>
quote: <the exact line of code the claim is about, copied character for character from the file>
failing_input: <a concrete input or state that makes it fail>
END
Write the code as it is, on one line: no escaping and no extra quotes around it. A finding whose quote is not in the file is thrown away.
Report each defect you can describe with a concrete failing input.
If you find nothing, reply with the single word NONE."""

OUTPUT_FORMATS = {"json": OUTPUT_FORMAT, "lines": OUTPUT_FORMAT_LINES}


def build_system_prompt(
    reviewer: Reviewer,
    instructions: Sequence[tuple] = (),
    lessons: Sequence[str] = (),
) -> str:
    """The system prompt for one reasoning reviewer: its own prompt (lens), then
    the repo's rules, then lessons recalled from past reviews, then the input
    format, then the output format for the reviewer's `reply_format`. A reviewer
    with a `promptFile` is not given the output format: the file replaces it, so
    it has to describe the format the reviewer's replies are read in.
    `lessons` is empty until the scoring database exists."""
    parts = [load_prompt(reviewer)]
    for path, text in instructions:
        parts.append(f"## Repository review instructions: {path}\n{text}")
    if lessons:
        parts.append("## Lessons from earlier reviews of this repository\n" + "\n".join(f"- {x}" for x in lessons))
    parts.append(INPUT_FORMAT)
    if not reviewer.prompt_file:
        parts.append(OUTPUT_FORMATS[reviewer.reply_format])
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Reading and merging findings
# ---------------------------------------------------------------------------


def _normalize_class(raw: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(raw or "").lower()).strip("-")
    return text or UNCLASSIFIED_CLASS


def _as_line(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _without_thinking(text: str) -> str:
    """The reply without a reasoning model's leading `<think>...</think>` envelope(s).

    A reasoning model (the DeepSeek-R1 distills) writes its chain of thought first, and that often
    holds braces and brackets, which would otherwise be taken for the start of the JSON. Only an
    envelope at the very start of the reply is removed: a literal `<think>` inside the JSON (a finding
    about code that parses such tags) is part of the payload and is kept. An envelope that is never
    closed means the reply was cut off while thinking, so there is no answer."""
    rest = text.lstrip()
    if not rest.startswith(_THINK_OPEN):
        return text
    while rest.startswith(_THINK_OPEN):
        end = rest.find(_THINK_CLOSE)
        if end == -1:
            return ""
        rest = rest[end + len(_THINK_CLOSE):].lstrip()
    return rest


def _extract_json(text: str) -> object:
    """The JSON in a model reply: the whole reply, a fenced block, or the
    outermost `{...}`/`[...]`. Raises `ValueError` when there is none."""
    stripped = _without_thinking(text).strip()
    fence = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    try:
        return json.loads(stripped)
    except ValueError:
        pass
    # The outermost container wins: in prose around a bare array (`Here you go: [{"claim": ...}]`) the
    # array starts before the object inside it, and decoding the inner object first would lose the list.
    containers = sorted(
        ((stripped.find(opener), stripped.rfind(closer)) for opener, closer in (("{", "}"), ("[", "]"))),
        key=lambda span: span[0] if span[0] >= 0 else len(stripped),
    )
    for start, end in containers:
        if 0 <= start < end:
            try:
                return json.loads(stripped[start : end + 1])
            except ValueError:
                continue
    raise ValueError("no JSON object in the reply")


_LINE_KEYS = ("file", "lines", "line", "class", "claim", "quote", "failing_input")
# A key at the start of a line, ignoring leading punctuation such as `**`, `__` or `-` and any between the key and its
# colon. The match after the key is lazy so the FIRST colon ends the key: a value that itself starts with
# punctuation and holds a colon (`quote: "/": 0,`, `failing_input: ':'`) keeps all of it.
_KEY_LINE = re.compile(r"^[\W_]*(" + "|".join(_LINE_KEYS) + r")[\W_]*?\s*:\s*(.*)$", re.IGNORECASE)
_FINDING_LINE = re.compile(r"\W*FINDING\W*", re.IGNORECASE)
_END_LINE = re.compile(r"\W*END\W*", re.IGNORECASE)
_NONE_REPLY = re.compile(r"\W*NONE\W*", re.IGNORECASE)
# The keys whose value is free text and may legitimately be written twice in one block (a model that
# does so is elaborating, not starting a second finding).
_REPEATABLE_KEYS = ("claim", "quote", "failing_input")


def _items_from_lines(text: str) -> list:
    """The findings in a plain-lines reply, as the dicts `_findings_from_items` reads.

    A block starts at a `FINDING` line, or at a key with no block open, and ends at `END`. A line is a
    `key: value` pair when it starts with one of the known keys (case does not matter, nor punctuation such as
    `**` or `-` before it); any other line continues the value above it. `lines` gives the first and last
    integers in it. Tolerates a code fence and a leading `<think>` envelope. A reply with no block that is
    the single word `NONE` is no findings. Raises `ValueError` when there is neither a block nor `NONE`."""
    body = _without_thinking(text).strip()
    fence = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", body, re.DOTALL)
    if fence:
        body = fence.group(1).strip()
    items: list = []
    current: Optional[dict] = None
    key = ""
    for raw in body.splitlines():
        line = raw.strip()
        if _FINDING_LINE.fullmatch(line):
            current, key = {}, ""
            items.append(current)
            continue
        if _END_LINE.fullmatch(line):
            current, key = None, ""
            continue
        match = _KEY_LINE.match(line)
        if match:
            name = match.group(1).lower()
            if current is None or (name in current and name not in _REPEATABLE_KEYS):
                # A field with no FINDING line before it, or a field written twice that is not free text,
                # starts a block of its own.
                current = {}
                items.append(current)
            key = name
            value = match.group(2)
            # Markdown bold written around the colon (`**file:** a.py`) leaves its closing marker at the start of
            # the value. It is stripped only when the key's bold is still open at the colon: in `**quote**: **d,`
            # the bold closed before the colon, so the `**` is the code's own (dictionary unpacking) and is kept.
            opening = line[: match.start(1)]
            before_colon = line[match.end(1) : match.start(2)].split(":", 1)[0]
            for marker in ("**", "__"):
                if marker in opening and marker not in before_colon and value.startswith(marker):
                    value = value[len(marker):]
            current[key] = value.strip()
        elif current is not None and key and line:
            current[key] += " " + line
    for item in items:
        span = item.pop("lines", None) or item.pop("line", None) or ""
        numbers = [int(n) for n in re.findall(r"\d+", span)]
        item["line_start"] = numbers[0] if numbers else 0
        item["line_end"] = numbers[-1] if numbers else item["line_start"]
    if items or _NONE_REPLY.fullmatch(body):
        return items
    raise ValueError("no finding blocks in the reply")


def _findings_from_items(items: list, reviewer: str, default_file: str) -> tuple:
    """`(findings, problems)` from the parsed items of a reply, whatever format it was in."""
    findings: list = []
    problems: list = []
    for index, item in enumerate(items):
        claim = item.get("claim") if isinstance(item, dict) else None
        if not isinstance(claim, str) or not claim.strip():
            problems.append(f"reviewer {reviewer!r}: finding {index} skipped (no 'claim')")
            continue
        start = _as_line(item.get("line_start"))
        end = _as_line(item.get("line_end")) or start
        file = item.get("file")
        failing = item.get("failing_input")
        quote = item.get("quote")
        findings.append(
            Finding(
                file=file.strip() if isinstance(file, str) and file.strip() else default_file,
                line_start=start,
                line_end=max(end, start),
                cls=_normalize_class(item.get("class")),
                claim=claim.strip(),
                failing_input=failing.strip() if isinstance(failing, str) else "",
                reviewers=(reviewer,),
                quote=quote.strip() if isinstance(quote, str) else "",
            )
        )
    return findings, problems


def parse_findings(text: str, reviewer: str, default_file: str = "", reply_format: str = DEFAULT_REPLY_FORMAT) -> tuple:
    """`(findings, problems)` from one model reply in `reply_format` (`lines` or `json`).

    JSON is read tolerantly (fences and prose around it, a bare list); lines as `_items_from_lines` says. A
    reply with no findings block and no JSON is one problem and no findings; an item without a `claim` is
    skipped with a problem. Never raises."""
    if reply_format == "lines":
        try:
            items = _items_from_lines(text)
        except ValueError:
            return [], [f"reviewer {reviewer!r}: the reply had no FINDING blocks and was not NONE"]
        return _findings_from_items(items, reviewer, default_file)
    try:
        data = _extract_json(text)
    except ValueError:
        return [], [f"reviewer {reviewer!r}: the reply was not JSON"]
    items = data.get("findings") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], [f"reviewer {reviewer!r}: the reply has no 'findings' list"]
    return _findings_from_items(items, reviewer, default_file)


def _same_finding(a: Finding, b: Finding) -> bool:
    """Two findings are one when they are in the same file and class and their
    line ranges overlap. A finding with no line number matches only another
    with none, and only when the claims are the same text."""
    if a.file != b.file or a.cls != b.cls:
        return False
    if a.line_start == 0 or b.line_start == 0:
        return a.line_start == b.line_start == 0 and " ".join(a.claim.lower().split()) == " ".join(b.claim.lower().split())
    return a.line_start <= b.line_end and b.line_start <= a.line_end


def _norm(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".")


def _squeeze(text: str) -> str:
    return " ".join(text.split())


def _same_quote(a: str, b: str) -> bool:
    """Two quotes of one claim are the same code when either is empty or, with whitespace squeezed, one
    contains the other. Quotes that differ are different code: the grounding step checks each one against
    the file, and merging them to the first would lose a valid quote whenever the first is not in the file."""
    a, b = _squeeze(a), _squeeze(b)
    return not a or not b or a in b or b in a


def _better_quote(a: str, b: str) -> str:
    """The quote to keep when two reports of one claim merge: the shorter, which is likelier to be exact
    in the file, but never one too short for grounding to place (`[]` inside `def f(items=[]):`)
    while the other is long enough. An empty quote loses to any other."""
    candidates = [q for q in (a, b) if q.strip()]
    usable = [q for q in candidates if len(_squeeze(q)) >= MIN_QUOTE_CHARS]
    return min(usable or candidates, key=lambda q: len(_squeeze(q)), default="")


def _merge(a: Finding, b: Finding) -> Finding:
    """`b` folded into `a`. The same claim only adds `b`'s reviewers (and a failing input `a` lacked).
    A different claim, the same claim with a different failing input, or the same claim quoting different
    code is kept as an `Alternate` with its own reviewers, so two defects that share a class and
    overlapping lines both survive and each is credited only to the reviewers who raised it."""
    pairs: list = [[a.claim, a.failing_input, list(a.reviewers), a.line_start, a.line_end, a.quote]]
    pairs += [[x.claim, x.failing_input, list(x.reviewers), x.line_start, x.line_end, x.quote] for x in a.alternates]
    incoming = [[b.claim, b.failing_input, list(b.reviewers), b.line_start, b.line_end, b.quote]]
    incoming += [[x.claim, x.failing_input, list(x.reviewers), x.line_start, x.line_end, x.quote] for x in b.alternates]
    for claim, failing, reviewers, start, end, quote in incoming:
        for pair in pairs:
            same_claim = _norm(pair[0]) == _norm(claim)
            # Failing inputs are concrete reproducers (a path, a shell command): compared exactly, since
            # `Config.json` and `config.json`, or `a id` and `a\nid`, are different inputs. Only claims,
            # which are prose, are compared loosely.
            same_input = not pair[1] or not failing or pair[1].strip() == failing.strip()
            if same_claim and same_input and _same_quote(pair[5], quote):
                pair[2] += [r for r in reviewers if r not in pair[2]]
                pair[1] = pair[1] or failing
                pair[5] = _better_quote(pair[5], quote)
                # The repeated claim may cover lines the first report did not: keep the whole range.
                pair[3] = min(pair[3], start) if pair[3] and start else pair[3] or start
                pair[4] = max(pair[4], end)
                break
        else:
            pairs.append([claim, failing, reviewers, start, end, quote])
    primary, rest = pairs[0], pairs[1:]
    return Finding(
        file=a.file,
        line_start=min(a.line_start, b.line_start) if a.line_start and b.line_start else a.line_start or b.line_start,
        line_end=max(a.line_end, b.line_end),
        cls=a.cls,
        claim=primary[0],
        failing_input=primary[1],
        reviewers=tuple(primary[2]),
        alternates=tuple(Alternate(c, f, tuple(r), st, en, q) for c, f, r, st, en, q in rest),
        quote=primary[5],
    )


def collect_findings(findings: Iterable[Finding]) -> list:
    """Every finding kept, duplicates merged, grouped by class.

    There is deliberately no vote threshold: a finding only one reviewer raised
    stays. Groups are ordered by class name and findings by file and line, so
    the same input always gives the same output."""
    merged: list = []
    for finding in findings:
        for index, existing in enumerate(merged):
            if _same_finding(existing, finding):
                merged[index] = _merge(existing, finding)
                break
        else:
            merged.append(finding)
    by_class: dict = {}
    for finding in merged:
        by_class.setdefault(finding.cls, []).append(finding)
    return [
        FindingGroup(cls=cls, findings=tuple(sorted(items, key=lambda f: (f.file, f.line_start, f.claim))))
        for cls, items in sorted(by_class.items())
    ]


# ---------------------------------------------------------------------------
# Running the reviewers
# ---------------------------------------------------------------------------

CompleteFn = Callable[[object, str, str, str], Optional[str]]


def _failure_text() -> str:
    kind, detail = openai_lib.take_failure()
    detail = display_safe(detail)  # the server's text, as sent: redacted here, where it is shown
    suffix = f": {detail}" if detail else ""
    return f"LM Studio request failed ({kind or 'unknown'}){suffix}"


def finish_request(req: Any, reply: Optional[str]) -> None:
    """Closes a recorded request with what the model returned: the token counts of a reply, or why there
    was none. Reads the failure reason without clearing it, so the message wording that follows still
    gets it. A request that failed after the model had answered (a classifier reply with no readable
    probability) keeps its token counts; one that never got a reply has none to keep. Never raises."""
    try:
        if reply is None:
            req.finish(None, failure_kind=openai_lib.peek_failure()[0], usage=openai_lib.take_usage())
        else:
            req.finish(reply, usage=openai_lib.take_usage())
    except Exception:
        pass


def describe_finding(finding: Finding) -> list:
    """A finding as lines for the session log: where it is, its class and claim, the failing input and the quoted code."""
    where = finding.file or "(unknown file)"
    if finding.line_start:
        where += f":{finding.line_start}" + (f"-{finding.line_end}" if finding.line_end != finding.line_start else "")
    lines = [f"{where} [{finding.cls}] {finding.claim}"]
    if finding.failing_input:
        lines.append(f"  failing input: {finding.failing_input}")
    if finding.quote:
        lines.append(f"  quote: {finding.quote}")
    return lines


def prompt_hash_of(reviewer: Reviewer, loop: bool = False) -> Optional[str]:
    """A hash of the reviewer's own prompt text (its lens, and its reply instructions when a promptFile
    carries them), so a recorded request names the prompt it used without storing it.

    A request made in the `need:` loop (`loop`) hashes the reviewer's tools and its `contextShare` in as well, so the same model
    with and without lookups, or with two different reference budgets, is two setups and their timings are never mixed.
    Without the loop the hash is what it always was."""
    try:
        text = load_prompt(reviewer)
        if loop:
            # The share sets the reference budget, so two shares are two setups; the default is hashed too (a hash kept
            # only for the default would make its absence mean something).
            text += "\ntools: " + ",".join(reviewer.tools) + f"\ncontextShare: {reviewer.context_share:g}"
        return hash_text(text)
    except Exception:
        return None


def _chunk_default_file(chunk: str) -> str:
    first = chunk.split("\n", 1)[0]
    return first[len("File: "):].strip() if first.startswith("File: ") else ""


def _review_one(
    reviewer: Reviewer,
    chunks_for: Callable[[Reviewer, str], list],
    client: object,
    complete: CompleteFn,
    instructions: Sequence[tuple],
    lessons: Sequence[str],
    recorder: Any = NULL_RECORDER,
    executor: object = None,
) -> ReviewerResult:
    """One reviewer over every chunk. Raises when it produced nothing at all
    (every request failed); partial failures are recorded as problems.

    A reviewer with `tools` (and an `executor`) may answer a chunk with `need:` lines instead of findings: the chunk is then asked
    again with what it asked for in front of it (`reviewer_need_lib`), up to the round limit. A reviewer without `tools` makes
    one request per chunk, with exactly the prompt it always had."""
    loop = loop_enabled(reviewer, executor)
    system = with_need_instructions(reviewer, build_system_prompt(reviewer, instructions, lessons), executor)
    budget = loop_reserve(reviewer, system, executor)
    chunks = chunks_for(reviewer, system)
    result = ReviewerResult(reviewer=reviewer.name, chunks=len(chunks))
    unavailable = loop_unavailable(reviewer, executor)
    if unavailable:
        result.warnings.append(unavailable)
    failed = 0
    last_failure = ""
    prompt_hash = prompt_hash_of(reviewer, loop) if recorder is not NULL_RECORDER else None
    for number, chunk in enumerate(chunks, start=1):
        session = NeedSession(reviewer, executor, budget, reply_format=need_format_for(reviewer, "reasoning")) if loop else None
        asking = True
        while asking:
            asking = False
            message = session.message(chunk) if session is not None else chunk
            with recorder.request(
                reviewer=reviewer.name, role="reasoning", model=reviewer.model, subject=("chunk", number),
                prompt_chars=len(system) + len(message), reply_format=reviewer.reply_format, translator=reviewer.translator,
                prompt_hash=prompt_hash,
            ) as req:
                result.counts.requests += 1
                reply = complete(client, reviewer.model, system, message)
                finish_request(req, reply)
                if reply is None:
                    failed += 1
                    last_failure = _failure_text()
                    result.problems.append(f"reviewer {reviewer.name!r}: chunk {number}/{len(chunks)}: {last_failure}")
                    req.problem(last_failure)
                    continue
                result.counts.replies += 1
                if session is not None:
                    outcome = session.read(reply, req)
                    if outcome == ASKED:
                        asking = True
                        continue
                    if outcome == LIMIT:
                        problem = "it kept asking for more context after the round limit, so this chunk has no answer from it"
                        result.problems.append(f"reviewer {reviewer.name!r}: chunk {number}/{len(chunks)}: {problem}")
                        req.problem(problem)
                        continue
                if reviewer.translator:
                    try:
                        reply = translate(reply, reviewer.translator)
                    except TranslatorError as exc:
                        # No fallback to the raw reply: the translator is what makes it readable.
                        result.counts.unreadable += 1
                        result.problems.append(f"reviewer {reviewer.name!r}: chunk {number}/{len(chunks)}: the translator {exc}")
                        req.problem(f"the translator {exc.brief}")
                        # The caller's result keeps what the translator printed; the session log's copy of it must not.
                        recorder.withhold(exc.printed)
                        continue
                found, problems = parse_findings(reply, reviewer.name, _chunk_default_file(chunk), reviewer.reply_format)
                if found or not problems:  # a reply that could not be read says nothing about what it found
                    req.note(f"raised {len(found)} finding(s)" if found else "raised no finding",
                             detail=tuple(line for f in found for line in describe_finding(f)))
                else:
                    result.counts.unreadable += 1
                result.findings.extend(found)
                result.problems.extend(f"chunk {number}/{len(chunks)}: {p}" for p in problems)
                for problem in problems:
                    req.problem(problem)
        if session is not None:
            result.warnings.extend(w for w in session.warnings if w not in result.warnings)
    if chunks and failed == len(chunks):
        raise ReviewError(f"every request failed; the last one: {last_failure}")
    return result


def run_reasoning_review(
    config: ReviewersConfig,
    loader: ModelLoader,
    client: object,
    diff_text: str,
    repo_root: "str | Path",
    max_reviewers: int,
    *,
    chunk_chars: Optional[int] = None,
    lessons: Sequence[str] = (),
    complete: Optional[CompleteFn] = None,
    recorder: Any = NULL_RECORDER,
    executor: object = None,
) -> ReasoningReview:
    """Runs every active reasoning reviewer over `diff_text` and collects the findings.

    `executor` (a `reviewer_tools_lib.ToolExecutor`) is what a reviewer with `tools` asks for lookups through;
    without one, such a reviewer runs without lookups and says so in the review's warnings.

    Raises `ReviewError` when no reasoning reviewer is active, the diff is
    empty, or it needs more than `MAX_CHUNKS` chunks. A model that is not
    loaded raises the scheduler's `ModelNotLoadedError` before anything runs.
    A reviewer that fails does not stop the others: it is reported in
    `errors` and the rest of the findings still come back."""
    if not config.active_reviewers("reasoning"):
        raise ReviewError("no active reasoning reviewer is configured; nothing to run")
    sections = split_diff(diff_text)
    if not sections:
        raise ReviewError("the diff is empty: there is nothing to review")
    if not any(_render_section(section)[1] for section in sections):
        raise ReviewError("the diff has no text changes to review (only binary, deleted, renamed or mode-only files)")
    changed = [s.path for s in sections]
    instructions = instruction_files_for(repo_root, changed)
    complete_fn = complete or openai_lib.complete
    chunk_counts: dict = {}

    def chunks_for(reviewer: Reviewer, system: str) -> list:
        size = chunk_chars_for(reviewer, system, chunk_chars, reserve=loop_reserve(reviewer, system, executor))
        chunks = chunk_diff(diff_text, size)
        if len(chunks) > MAX_CHUNKS:
            raise ReviewError(
                f"the diff needs {len(chunks)} requests per reviewer at {size} characters each "
                f"(the limit is {MAX_CHUNKS}); review a smaller range, or raise chunk_chars"
            )
        chunk_counts[reviewer.name] = len(chunks)
        return chunks

    def run_reviewer(reviewer: Reviewer, role: str) -> ReviewerResult:
        return _review_one(reviewer, chunks_for, client, complete_fn, instructions, lessons, recorder, executor)

    report = run_stages(config, loader, run_reviewer, max_reviewers, roles=("reasoning",), recorder=recorder)
    review = ReasoningReview(warnings=list(report.warnings), outcomes=list(report.outcomes))
    unsent = [section.path for section in sections if not _render_section(section)[1]]
    if unsent:
        shown = ", ".join(unsent[:10]) + (f" and {len(unsent) - 10} more" if len(unsent) > 10 else "")
        review.warnings.append(
            f"{len(unsent)} changed file(s) had no new text to review and were not sent: {shown} "
            "(deleted, binary, renamed-only or mode-only files)"
        )
    collected: list = []
    for outcome in report.outcomes:
        name = outcome.reviewer.name
        if outcome.error is not None:
            review.errors.append(f"reviewer {name!r}: {outcome.error}")
            continue
        result = outcome.value
        assert isinstance(result, ReviewerResult)
        review.reviewers_run.append(name)
        review.errors.extend(result.problems)
        review.warnings.extend(w for w in result.warnings if w not in review.warnings)
        collected.extend(result.findings)
    review.chunks = max(chunk_counts.values(), default=0)
    review.groups = collect_findings(collected)
    return review


def stage_status_line(followed_by: Sequence[str] = ()) -> str:
    """What the reasoning stage's findings have been through, given the stages that went on from them in the same
    review (`followed_by`, any of "verifier" and "classifier"). With none, the findings are plain reasoning output and
    the line says nothing was checked; with some, it names what each later stage did to them and what is still missing,
    so a pipeline that ran every stage is never described as having run none."""
    ran = [stage for stage in ("verifier", "classifier") if stage in followed_by]
    if not ran:
        return "These findings are NOT verified and NOT classified: no verifier or classifier stage ran."
    missing = [word for stage, word in (("verifier", "verified"), ("classifier", "classified")) if stage not in ran]
    before = " and ".join({"verifier": "verification", "classifier": "classification"}[stage] for stage in ran)
    line = f"These are the reasoning stage's findings, before {before}: the result of the later stage(s) is reported separately."
    if missing:
        line += f" They are NOT {' and NOT '.join(missing)}: no {'verifier' if 'verified' in missing else 'classifier'} stage ran."
    return line


def incomplete_review_line(problems: Sequence[str]) -> Optional[str]:
    """The line every review's text opens with when any stage reported a problem, or None for a clean run.

    The result text is the only thing a calling model reads, and the detail list (`Problems (N):`) sits at the end, after
    the finding count. A small caller was seen reading `0 finding(s)` on the first line and reporting "no bugs" when the
    only reviewer's reply had been unreadable, so the failure state goes first, before any count. The count is of the
    same list the formatter prints under "Problems", so the two always agree. Warnings do not trigger it: they are notes
    about a run, not a failure. No language-specific text: the reviewer is programming-language agnostic by rule."""
    if not problems:
        return None
    return (
        f"INCOMPLETE REVIEW: {len(problems)} problem(s) (listed under \"Problems\" at the end); findings may be missing, "
        "so do NOT report this review as clean."
    )


def format_review(review: ReasoningReview, *, followed_by: Sequence[str] = ()) -> str:
    """The review as text for the calling model: findings by class, then errors.

    `followed_by` names the stages that run on these findings in the same review (the `review` tool's "reasoning"
    type passes none; a pipeline that goes on to verify and classify passes both), so the opening line says what the
    findings have and have not been through."""
    header = incomplete_review_line(review.errors)
    lines = [] if header is None else [header]
    lines += [
        f"Reasoning review: {review.finding_count} finding(s) in {len(review.groups)} class(es) "
        f"from {len(review.reviewers_run)} reviewer(s) ({', '.join(review.reviewers_run) or 'none'}), "
        f"up to {review.chunks} diff chunk(s) each.",
        stage_status_line(followed_by),
    ]
    for group in review.groups:
        lines.append("")
        lines.append(f"## {group.cls} ({len(group.findings)})")
        for f in group.findings:
            where = f.file or "(unknown file)"
            if f.line_start:
                where += f":{f.line_start}" + (f"-{f.line_end}" if f.line_end != f.line_start else "")
            lines.append(f"- {where} [{', '.join(f.reviewers)}] {f.claim}")
            if f.failing_input:
                lines.append(f"    failing input: {f.failing_input}")
            for alt in f.alternates:
                at = f"{alt.line_start}" + (f"-{alt.line_end}" if alt.line_end != alt.line_start else "") if alt.line_start else "?"
                lines.append(f"    also, at {at} [{', '.join(alt.reviewers)}]: {alt.claim}")
                if alt.failing_input:
                    lines.append(f"        failing input: {alt.failing_input}")
    if review.errors:
        lines.append("")
        lines.append(f"Problems ({len(review.errors)}):")
        lines.extend(f"- {e}" for e in review.errors)
    if review.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {w}" for w in review.warnings)
    return "\n".join(lines)


__all__ = [
    "ReviewError",
    "DiffSection",
    "Finding",
    "FindingGroup",
    "ReplyCount",
    "ReviewerResult",
    "ReasoningReview",
    "split_diff",
    "Alternate",
    "chunk_file_diff",
    "chunk_diff",
    "chunk_chars_for",
    "applies_to",
    "instruction_files_for",
    "build_system_prompt",
    "OUTPUT_FORMAT",
    "OUTPUT_FORMAT_LINES",
    "parse_findings",
    "describe_finding",
    "collect_findings",
    "run_reasoning_review",
    "format_review",
    "incomplete_review_line",
    "stage_status_line",
]
