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
  2. `instruction_files_for` / `review_skills` gather the repo's own review
     rules: the `.github/instructions/*.instructions.md` files whose `applyTo`
     matches a changed path (the same files Copilot reads), and any review
     skill. `build_system_prompt` puts the reviewer's lens, those rules, any
     recalled lessons and the required output format together.
  3. `run_reasoning_review` runs the active reasoning reviewers through
     `reviewer_scheduler_lib.run_stages` (so grouping, parallelism and model
     loading are the scheduler's), and `collect_findings` keeps EVERY finding
     from EVERY reviewer: there is no vote threshold, so a finding one reviewer
     alone saw is not lost. Findings are grouped by class and duplicates merge.

Findings at this stage are unverified and unclassified: the verifier and
classifier stages are separate, later pieces of work.

Standard library only, apart from the sibling libs. The model client and the
completion call are passed in, so tests run with a stub and no LM Studio.
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import local_compress_lib
from codebase_reviewer_config_lib import Reviewer, ReviewersConfig, load_prompt
from lmstudio_model_loader_lib import ModelLoader
from reviewer_scheduler_lib import run_stages

# Sizing. A token count depends on the model's vocabulary, which is not known
# here, so chunk sizes use a deliberately low characters-per-token guess: a
# chunk that is a little small costs one more request, one that is too big is
# refused by LM Studio (an HTTP 400). `chunk_chars` overrides the whole
# calculation for a caller that knows better.
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


@dataclass(frozen=True)
class FindingGroup:
    cls: str
    findings: tuple


@dataclass
class ReviewerResult:
    """What one reviewer produced across all of its chunks."""

    reviewer: str
    findings: list = field(default_factory=list)
    # Problems that did not lose the whole reviewer: a malformed reply, an item
    # skipped, one chunk whose request failed.
    problems: list = field(default_factory=list)
    chunks: int = 0


@dataclass
class ReasoningReview:
    groups: list = field(default_factory=list)
    # One line per reviewer that produced nothing usable, or per problem.
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    reviewers_run: list = field(default_factory=list)
    chunks: int = 0

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


def chunk_chars_for(reviewer: Reviewer, system_prompt: str, override: Optional[int] = None) -> int:
    """How many characters of diff one request to `reviewer` can carry.

    `override` (the caller's `chunk_chars`) wins. Otherwise: the reviewer's
    `contextLength` (or `ASSUMED_CONTEXT_TOKENS`), minus a share kept for the
    reply, at `CHARS_PER_TOKEN_GUESS` characters a token, minus the system
    prompt. Raises `ReviewError` when the prompt alone leaves less than
    `MIN_CHUNK_CHARS`, which is the reviewer's context being too small for its
    own prompt, not something chunking can fix."""
    if override is not None:
        return max(override, 1)
    context = reviewer.context_length or ASSUMED_CONTEXT_TOKENS
    usable_tokens = context - context // REPLY_RESERVE_DIVISOR
    room = usable_tokens * CHARS_PER_TOKEN_GUESS - len(system_prompt)
    if room < MIN_CHUNK_CHARS:
        raise ReviewError(
            f"reviewer {reviewer.name!r}: its prompt and rules take about {len(system_prompt)} characters, "
            f"which leaves under {MIN_CHUNK_CHARS} for the diff in a {context}-token context; "
            "raise its contextLength, shorten its prompt, or pass chunk_chars"
        )
    return room


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


def review_skills(repo_root: "str | Path") -> list:
    """`(relative path, text)` of each review skill: a `SKILL.md` under
    `.claude/skills/` or `skills/` whose directory name or front-matter `name`
    contains "review"."""
    root = Path(repo_root)
    found = []
    for base in (root / ".claude" / "skills", root / "skills"):
        for path in sorted(base.glob("*/SKILL.md")):
            text = _read_text(path, root)
            if text is None:
                continue
            fields, body = _split_front_matter(text)
            name = f"{path.parent.name} {fields.get('name', '')}".lower()
            if "review" in name:
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
  "failing_input": "<a concrete input or state that makes it fail>"}]}
Report each defect you can describe with a concrete failing input. Use the line numbers printed at the left of each line.
If you find nothing, reply {"findings": []}."""


def build_system_prompt(
    reviewer: Reviewer,
    instructions: Sequence[tuple] = (),
    skills: Sequence[tuple] = (),
    lessons: Sequence[str] = (),
) -> str:
    """The system prompt for one reasoning reviewer: its own prompt (lens), then
    the repo's rules, then lessons recalled from past reviews, then the output
    format. `lessons` is empty until the scoring database exists."""
    parts = [load_prompt(reviewer)]
    for label, items in (("Repository review instructions", instructions), ("Repository review skills", skills)):
        for path, text in items:
            parts.append(f"## {label}: {path}\n{text}")
    if lessons:
        parts.append("## Lessons from earlier reviews of this repository\n" + "\n".join(f"- {x}" for x in lessons))
    parts.append(INPUT_FORMAT)
    parts.append(OUTPUT_FORMAT)
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


def parse_findings(text: str, reviewer: str, default_file: str = "") -> tuple:
    """`(findings, problems)` from one model reply.

    Tolerant of fences and prose around the JSON, and of a bare list. A reply
    with no JSON is one problem and no findings; an item without a `claim` is
    skipped with a problem. Never raises."""
    try:
        data = _extract_json(text)
    except ValueError:
        return [], [f"reviewer {reviewer!r}: the reply was not JSON"]
    items = data.get("findings") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], [f"reviewer {reviewer!r}: the reply has no 'findings' list"]
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
        findings.append(
            Finding(
                file=file.strip() if isinstance(file, str) and file.strip() else default_file,
                line_start=start,
                line_end=max(end, start),
                cls=_normalize_class(item.get("class")),
                claim=claim.strip(),
                failing_input=failing.strip() if isinstance(failing, str) else "",
                reviewers=(reviewer,),
            )
        )
    return findings, problems


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


def _merge(a: Finding, b: Finding) -> Finding:
    """`b` folded into `a`. The same claim only adds `b`'s reviewers (and a failing input `a` lacked).
    A different claim, or the same claim with a different failing input, is kept as an `Alternate`
    with its own reviewers, so two defects that share a class and overlapping lines both survive and
    each is credited only to the reviewers who raised it."""
    pairs: list = [[a.claim, a.failing_input, list(a.reviewers), a.line_start, a.line_end]]
    pairs += [[x.claim, x.failing_input, list(x.reviewers), x.line_start, x.line_end] for x in a.alternates]
    incoming = [[b.claim, b.failing_input, list(b.reviewers), b.line_start, b.line_end]]
    incoming += [[x.claim, x.failing_input, list(x.reviewers), x.line_start, x.line_end] for x in b.alternates]
    for claim, failing, reviewers, start, end in incoming:
        for pair in pairs:
            same_claim = _norm(pair[0]) == _norm(claim)
            # Failing inputs are concrete reproducers (a path, a shell command): compared exactly, since
            # `Config.json` and `config.json`, or `a id` and `a\nid`, are different inputs. Only claims,
            # which are prose, are compared loosely.
            same_input = not pair[1] or not failing or pair[1].strip() == failing.strip()
            if same_claim and same_input:
                pair[2] += [r for r in reviewers if r not in pair[2]]
                pair[1] = pair[1] or failing
                # The repeated claim may cover lines the first report did not: keep the whole range.
                pair[3] = min(pair[3], start) if pair[3] and start else pair[3] or start
                pair[4] = max(pair[4], end)
                break
        else:
            pairs.append([claim, failing, reviewers, start, end])
    primary, rest = pairs[0], pairs[1:]
    return Finding(
        file=a.file,
        line_start=min(a.line_start, b.line_start) if a.line_start and b.line_start else a.line_start or b.line_start,
        line_end=max(a.line_end, b.line_end),
        cls=a.cls,
        claim=primary[0],
        failing_input=primary[1],
        reviewers=tuple(primary[2]),
        alternates=tuple(Alternate(c, f, tuple(r), st, en) for c, f, r, st, en in rest),
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
    kind, detail = local_compress_lib.take_failure()
    suffix = f": {detail}" if detail else ""
    return f"LM Studio request failed ({kind or 'unknown'}){suffix}"


def _chunk_default_file(chunk: str) -> str:
    first = chunk.split("\n", 1)[0]
    return first[len("File: "):].strip() if first.startswith("File: ") else ""


def _review_one(
    reviewer: Reviewer,
    chunks_for: Callable[[Reviewer, str], list],
    client: object,
    complete: CompleteFn,
    instructions: Sequence[tuple],
    skills: Sequence[tuple],
    lessons: Sequence[str],
) -> ReviewerResult:
    """One reviewer over every chunk. Raises when it produced nothing at all
    (every request failed); partial failures are recorded as problems."""
    system = build_system_prompt(reviewer, instructions, skills, lessons)
    chunks = chunks_for(reviewer, system)
    result = ReviewerResult(reviewer=reviewer.name, chunks=len(chunks))
    failed = 0
    last_failure = ""
    for number, chunk in enumerate(chunks, start=1):
        reply = complete(client, reviewer.model, system, chunk)
        if reply is None:
            failed += 1
            last_failure = _failure_text()
            result.problems.append(f"reviewer {reviewer.name!r}: chunk {number}/{len(chunks)}: {last_failure}")
            continue
        found, problems = parse_findings(reply, reviewer.name, _chunk_default_file(chunk))
        result.findings.extend(found)
        result.problems.extend(f"chunk {number}/{len(chunks)}: {p}" for p in problems)
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
) -> ReasoningReview:
    """Runs every active reasoning reviewer over `diff_text` and collects the findings.

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
    skills = review_skills(repo_root)
    complete_fn = complete or local_compress_lib.complete
    chunk_counts: dict = {}

    def chunks_for(reviewer: Reviewer, system: str) -> list:
        size = chunk_chars_for(reviewer, system, chunk_chars)
        chunks = chunk_diff(diff_text, size)
        if len(chunks) > MAX_CHUNKS:
            raise ReviewError(
                f"the diff needs {len(chunks)} requests per reviewer at {size} characters each "
                f"(the limit is {MAX_CHUNKS}); review a smaller range, or raise chunk_chars"
            )
        chunk_counts[reviewer.name] = len(chunks)
        return chunks

    def run_reviewer(reviewer: Reviewer, role: str) -> ReviewerResult:
        return _review_one(reviewer, chunks_for, client, complete_fn, instructions, skills, lessons)

    report = run_stages(config, loader, run_reviewer, max_reviewers, roles=("reasoning",))
    review = ReasoningReview(warnings=list(report.warnings))
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
        collected.extend(result.findings)
    review.chunks = max(chunk_counts.values(), default=0)
    review.groups = collect_findings(collected)
    return review


def format_review(review: ReasoningReview) -> str:
    """The review as text for the calling model: findings by class, then errors."""
    lines = [
        f"Reasoning review: {review.finding_count} finding(s) in {len(review.groups)} class(es) "
        f"from {len(review.reviewers_run)} reviewer(s) ({', '.join(review.reviewers_run) or 'none'}), "
        f"up to {review.chunks} diff chunk(s) each.",
        "These findings are NOT verified and NOT classified: no verifier or classifier stage ran.",
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
    "ReviewerResult",
    "ReasoningReview",
    "split_diff",
    "Alternate",
    "chunk_file_diff",
    "chunk_diff",
    "chunk_chars_for",
    "applies_to",
    "instruction_files_for",
    "review_skills",
    "build_system_prompt",
    "parse_findings",
    "collect_findings",
    "run_reasoning_review",
    "format_review",
]
