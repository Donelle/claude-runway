"""
Shared, side-effect-free chunking helpers used by both:
  - ingest_to_qdrant.py   (standalone CLI script)
  - ingest_mcp_server.py  (MCP server exposing the same logic as tools)

Kept dependency-free (no qdrant-client / mcp_server_qdrant imports here) so
it's easy to unit test and reuse. No print() calls in this file on purpose --
ingest_mcp_server.py runs over stdio, where stray prints corrupt the protocol.
"""

import functools
import hashlib
import os
import re
from pathlib import Path
from typing import Generator, Optional

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rb", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".swift", ".kt", ".scala",
    ".sql", ".sh", ".yaml", ".yml", ".json",
}
DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}

# Only these get heading-based chunking; every other DOC_EXTENSIONS member is
# line-chunked (still typed "doc"). '#' means "comment", not "heading", in a
# .txt -- a requirements.txt of '# ...' lines chunked to ONE LINE PER COMMENT
# while this set didn't exist, producing embeddings of single comment lines.
# .rst marks headings with underlines rather than '#', so it never had headings
# to find here in the first place.
MARKDOWN_EXTENSIONS = {".md", ".mdx"}

# Curly-brace-block languages (from CODE_EXTENSIONS) that get boundary-aware
# chunking via brace_boundary_indexes() instead of a pure fixed-line window
# (issue #55) -- every CODE_EXTENSIONS member whose blocks are delimited by
# '{'/'}' rather than indentation or a non-brace keyword (.rb's 'end', for
# example, isn't covered here -- out of the issue's named scope of "at least
# C#, Python, TypeScript", and there's no brace signal to hook a heuristic on).
BRACE_BLOCK_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".php", ".swift", ".kt", ".scala", ".rs",
}

# Indentation-delimited-block languages that get boundary-aware chunking via
# python_boundary_indexes() instead of brace_boundary_indexes() (issue #55) --
# Python has no braces, so "depth returns to zero" has to mean something else
# (see python_boundary_indexes' own docstring).
INDENT_BLOCK_EXTENSIONS = {".py"}

EXCLUDE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "dist", "build", "__pycache__",
    ".next", "target", "vendor", ".idea", ".vscode", "coverage",
}


def normalize_extensions(extensions) -> set:
    """Accepts extensions with or without a leading dot, case-insensitive."""
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}


def load_gitignore_spec(repo_path: Path):
    """
    Loads repo_path/.gitignore as a matchable spec, if present and the
    optional `pathspec` package is installed. Returns None otherwise (caller
    should treat None as "no gitignore filtering") -- this is a nice-to-have,
    not a hard dependency.
    """
    gitignore_path = repo_path / ".gitignore"
    if not gitignore_path.exists():
        return None
    try:
        import pathspec
    except ImportError:
        return None
    lines = gitignore_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def iter_files(
    repo_path: Path,
    extensions: set,
    extra_exclude_dirs: Optional[set] = None,
    gitignore_spec=None,
    extra_exclude_files: Optional[set] = None,
):
    """
    extra_exclude_dirs adds to (doesn't replace) the built-in EXCLUDE_DIRS.
    gitignore_spec, if provided (see load_gitignore_spec), additionally skips
    anything the repo's own .gitignore would exclude.
    extra_exclude_files skips any file whose name (basename, not full path)
    appears in the set -- intended for caller-owned bookkeeping files like the
    sync manifest that should never be indexed regardless of scope, extension
    filter, or gitignore state.
    """
    exclude_dirs = EXCLUDE_DIRS | (extra_exclude_dirs or set())
    exclude_files = extra_exclude_files or set()
    for path in repo_path.rglob("*"):
        if path.is_dir():
            continue
        # Exclude on the path RELATIVE TO repo_path, not the absolute path --
        # repo_path.rglob() yields absolute paths, so path.parts includes every
        # ancestor directory of the repo (e.g. checked out under ~/build/myrepo,
        # "build" would match EXCLUDE_DIRS and silently exclude everything).
        rel = path.relative_to(repo_path)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        if path.name in exclude_files:
            continue
        if gitignore_spec is not None and gitignore_spec.match_file(str(rel)):
            continue
        if path.suffix.lower() in extensions:
            yield path


def validate_chunk_params(chunk_size: int, overlap: int) -> Optional[str]:
    """
    Returns an error message describing what's wrong, or None if chunk_size/
    overlap are safe to pass to chunk_lines()/chunk_markdown() below.

    overlap >= chunk_size collapses chunk_lines()'s step (chunk_size -
    overlap) to 1 -- silently turning "chunk every N lines with M overlap"
    into "chunk every line," multiplying the number of chunks (and
    embedding/storage cost) by roughly chunk_size -- confirmed real, not
    theoretical (issue #26). chunk_size < 1 is the same class of defect from
    the other direction: chunk_file's `end <= start` on every line then
    yields an empty (or near-empty) chunk per line instead of raising
    cleanly. overlap < 0 has no valid meaning either.

    Called once at every CLI/tool boundary that accepts these two params
    (ingest_to_qdrant.py's argparse setup; ingest_mcp_server.py's
    index_repo/sync_repo/preview_index) rather than enforced inside
    chunk_lines()/chunk_markdown() themselves -- that way the error names the
    actual bad input immediately, at the point the caller can still act on
    it, instead of surfacing however deep the chunking call stack happens to
    be by the time it's reached.
    """
    if chunk_size < 1:
        return f"chunk_lines must be at least 1 (got {chunk_size})."
    if overlap < 0:
        return f"overlap must be 0 or greater (got {overlap})."
    if overlap >= chunk_size:
        return (
            f"overlap ({overlap}) must be smaller than chunk_lines ({chunk_size}) -- "
            "otherwise every chunk boundary advances by only 1 line, multiplying the "
            f"number of chunks (and embedding/storage cost) by roughly {chunk_size}x."
        )
    return None


def chunk_lines(text: str, chunk_size: int, overlap: int):
    """
    Fixed-size line-based chunking with overlap. Language-agnostic.

    Precondition: overlap < chunk_size (and chunk_size >= 1, overlap >= 0) --
    see validate_chunk_params, which every caller of this function is
    expected to have already checked. Not re-enforced here on purpose (see
    that function's own docstring for why).
    """
    lines = text.splitlines()
    if not lines:
        return
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(lines), step):
        end = min(start + chunk_size, len(lines))
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            yield chunk, start + 1, end  # 1-indexed line numbers
        if end == len(lines):
            break


def _strip_c_style_line(
    line: str,
    in_block_comment: bool,
    in_template_string: bool = False,
    in_verbatim_string: bool = False,
    ext: str = "",
):
    """
    Best-effort scan of a single line of C-family source (JS/TS/Go/Java/
    C/C++/C#/PHP/Swift/Kotlin/Scala/Rust all share this comment/string
    grammar closely enough for a boundary heuristic). Returns (stripped,
    in_block_comment, in_template_string, in_verbatim_string) where
    `stripped` is the same length as `line` with every string-literal and
    comment character replaced by a space -- blanked, not removed, so
    column positions never shift -- so a later brace-count pass never
    miscounts a '{'/'}' that only exists inside a string or a comment. The
    three `in_*` flags carry a `/* ... */`, a backtick template/raw string
    literal, or a C# verbatim string (`@"..."`) spanning multiple lines
    into the next call.

    Backtick literals (JS/TS template strings, Go raw strings) get the same
    treatment as '/'"' strings, including backslash-escape handling for the
    closing backtick -- correct for JS/TS (where '\\`' is a real escape);
    for Go, where a raw string has no escape syntax at all, this can in
    principle misjudge a literal '\\' immediately followed by a backtick as
    an escaped delimiter rather than the string's actual close. Accepted
    tradeoff: that's a much rarer pattern than an ordinary '${...}'/brace-
    containing template literal (issue #55, PR #137 review) -- correctly
    treating THAT common case matters far more than the rare Go edge case.

    C# verbatim strings (`@"..."`) are the one common multi-line plain-
    quote form worth handling explicitly (PR #137 review): unlike an
    ordinary `"..."`, a bare newline inside one is legal, and `""` (a
    doubled quote) is an escaped literal `"` rather than the closing
    delimiter -- so it needs its own state, tracked separately from the
    single-line '"'/"'" handling below, which is unaffected. Recognized
    only via the `@"` marker specifically, which doesn't collide with any
    other supported language's meaning of a standalone '@' (a TS/Python-
    style decorator is always its own token followed by an identifier, not
    immediately before a quote).

    `ext` is the lowercase file extension (e.g. `".php"`) used for the one
    language-specific comment form that can't be handled uniformly: PHP's `#`
    line comment. When `ext == ".php"`, a `#` that is NOT currently inside a
    block comment or string AND is NOT immediately followed by `[` is treated
    as a line comment (remainder of the line blanked), exactly like `//`. The
    `#[` exclusion is required because PHP 8 attributes begin with `#[` (e.g.
    `#[Route("/home")]`) -- those are NOT comments, so blanking them would
    miss the surrounding function's opening brace and fabricate a spurious
    boundary. This treatment is intentionally PHP-only: in C/C++/C#, `#`
    introduces a preprocessor directive (`#if`, `#include`, `#region`, ...);
    in Rust, `#` introduces an attribute (`#[derive(...)]`). A blanket rule
    would corrupt scanning for those languages, which is why the issue
    explicitly ruled it out (#138). All other `ext` values leave `#` alone.

    Not a real tokenizer: unusual per-language escape rules, nested block
    comments (most C-family languages don't allow those anyway), and other
    multiline literal forms this scanner doesn't special-case (Java text
    blocks, C++ raw strings, JS/TS regex literals) can still miscount in
    rare cases -- properly handling those needs per-language lexical rules
    (a materially bigger change than this shared, extension-agnostic
    scanner takes on), so they're an accepted, documented tradeoff rather
    than something this function claims to get right. See
    chunk_code_by_boundaries' own docstring for why a missed boundary is
    always the safe failure mode here, never a fabricated one -- the
    residual risk from these forms is a fabricated one in rare cases, never
    content corruption (chunk_code_by_boundaries only ever cuts at an exact
    line boundary).
    """
    result = []
    i, n = 0, len(line)
    while i < n:
        if in_block_comment:
            end = line.find("*/", i)
            if end == -1:
                result.append(" " * (n - i))
                i = n
            else:
                result.append(" " * (end + 2 - i))
                i = end + 2
                in_block_comment = False
            continue
        if in_template_string:
            j = i
            while j < n:
                if line[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if line[j] == "`":
                    j += 1
                    in_template_string = False
                    break
                j += 1
            result.append(" " * (j - i))
            i = j
            continue
        if in_verbatim_string:
            j = i
            while j < n:
                if line[j] == '"':
                    if j + 1 < n and line[j + 1] == '"':
                        j += 2  # doubled quote -- an escaped literal '"', string stays open
                        continue
                    j += 1
                    in_verbatim_string = False
                    break
                j += 1
            result.append(" " * (j - i))
            i = j
            continue
        ch = line[i]
        if ch == "#" and ext == ".php" and not (i + 1 < n and line[i + 1] == "["):
            # PHP line comment: '#' through end of line, same treatment as '//'.
            # PHP 8 attributes start with '#[' (e.g. '#[Route(...)]') -- those
            # are NOT comments, so '#[' is explicitly excluded from this branch.
            result.append(" " * (n - i))
            i = n
        elif ch == "/" and i + 1 < n and line[i + 1] == "/":
            result.append(" " * (n - i))
            i = n
        elif ch == "/" and i + 1 < n and line[i + 1] == "*":
            in_block_comment = True
            result.append("  ")
            i += 2
        elif ch == "`":
            in_template_string = True
            result.append(" ")
            i += 1
        elif ch == "@" and i + 1 < n and line[i + 1] == '"':
            in_verbatim_string = True
            result.append("  ")
            i += 2
        elif ch in ("'", '"'):
            quote = ch
            j = i + 1
            while j < n:
                if line[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if line[j] == quote:
                    j += 1
                    break
                j += 1
            result.append(" " * (j - i))
            i = j
        else:
            result.append(ch)
            i += 1
    return "".join(result), in_block_comment, in_template_string, in_verbatim_string


def _strip_python_style_line(line: str, triple_quote):
    """
    Same purpose as _strip_c_style_line, for Python's comment/string grammar
    instead: '#' line comments (no '/* */' block-comment syntax), plus
    triple-quoted strings (''' or \"\"\") that can span many lines --
    `triple_quote` (None, or the 3-char marker currently open) carries that
    span across the call boundary the same way in_block_comment does above.
    """
    result = []
    i, n = 0, len(line)
    while i < n:
        if triple_quote:
            end = line.find(triple_quote, i)
            if end == -1:
                result.append(" " * (n - i))
                i = n
            else:
                result.append(" " * (end + 3 - i))
                i = end + 3
                triple_quote = None
            continue
        ch = line[i]
        if ch == "#":
            result.append(" " * (n - i))
            i = n
        elif line[i:i + 3] in ('"""', "'''"):
            marker = line[i:i + 3]
            close = line.find(marker, i + 3)
            if close == -1:
                triple_quote = marker
                result.append(" " * (n - i))
                i = n
            else:
                result.append(" " * (close + 3 - i))
                i = close + 3
        elif ch in ("'", '"'):
            quote = ch
            j = i + 1
            while j < n:
                if line[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if line[j] == quote:
                    j += 1
                    break
                j += 1
            result.append(" " * (j - i))
            i = j
        else:
            result.append(ch)
            i += 1
    return "".join(result), triple_quote


# Common declaration-site modifier keywords that can precede a container
# keyword (public/private/static/... class Foo, pub struct Foo, export
# class Foo, ...). Consumed left-to-right so any number/combination can
# appear before the real container keyword.
_CONTAINER_MODIFIER_RE = re.compile(
    r"^(public|private|protected|internal|static|abstract|sealed|partial|readonly|"
    r"final|export|default|virtual|override|async|unsafe|pub|const)\b\s*"
)

# Keywords that open a "container" scope -- a class/struct/interface/etc.
# whose direct children are members (methods, fields, nested types), not
# statements. Covers class (C#/Java/TS/JS/Kotlin/Scala/Swift), struct/
# interface/enum (C#/Java/Go/TS/Swift/Rust), namespace/module (C#/TS),
# record (C#), impl/trait (Rust), protocol (Swift), object (Kotlin/TS
# singleton). Not exhaustive -- a missed keyword just means that
# container's members fall back to depth-0-only boundaries (the pre-fix
# behavior), never a wrong split.
#
# Requires actual WHITESPACE right after the keyword, not just a '\b' word
# boundary (PR #137 review): '\b' is satisfied between a word char and ANY
# non-word char, including '.' -- so '\bmodule\b' matched the start of
# 'module.exports = function () {' (a completely ordinary CommonJS
# expression, not a TypeScript module declaration), tagging that
# function's own brace 'container' and turning every line in its body into
# a false member-level boundary (confirmed by direct reproduction).
#
# The optional identifier group plus lookahead disambiguates a genuine
# declaration from the SAME keyword used as a plain TYPE NAME in a variable
# declaration (PR #137 review, second finding): 'object value = new Foo {'
# or 'struct Point p = {' both have whitespace right after the keyword too
# (so the earlier '\s'-only fix doesn't catch them), but neither is a
# container declaration -- 'object'/'struct' there just names the
# variable's TYPE, and what follows (another identifier, then '=') is
# nothing like a real declaration's shape. A real declaration is always
# EITHER "keyword IDENTIFIER" followed directly by '{'/':'/'<' (or nothing
# else on the line, for Allman style) -- e.g. 'class Foo {', 'class Foo<T>',
# 'class Foo : Base {' -- OR just "keyword" with no identifier at all
# followed directly by one of those (Kotlin's anonymous
# `object : Base() {}`, which has no name). 'object value = ...' fails
# both shapes: after the optional identifier ("value") comes '=', not one
# of '{'/':'/'<'/end-of-line; and skipping the identifier entirely, the
# very next thing after "object " is "v" (from "value"), also not one of
# those. Confirmed by direct reproduction (an object-initializer's own
# body was fully mistagged 'container' before this fix).
_CONTAINER_KEYWORD_RE = re.compile(
    r"^(?:class|struct|interface|namespace|record|enum|impl|trait|protocol|module|object)"
    r"\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*)?(?=[{:<]|$)"
)


def _line_declares_container(stripped_line: str) -> bool:
    """
    True only if `stripped_line` is ITSELF a container declaration --
    the container keyword must be the line's own first real token (after
    any leading modifiers), not merely present anywhere on the line.

    PR #137 review: matching the keyword anywhere via `.search()` treated
    a type ANNOTATION/parameter/generic-bound mentioning one of these words
    as if it opened a container -- e.g. `function process(value: object) {`
    (TypeScript, "object" as a type) or `void foo(struct Point p) {` (C,
    "struct" as a parameter's type name) tagged that FUNCTION's own brace
    "container", which then made every single line inside its body look
    like a safe member-level boundary (confirmed by direct reproduction:
    every body line was flagged). Anchoring at the line's start (skipping
    only recognized modifier words) fixes both of those. Known, accepted
    remaining gap: a single-line `template<class T> class Foo {` (C++)
    still tags 'other' instead of 'container', since "template" isn't a
    recognized modifier -- rare style (real C++ almost always puts the
    template<> parameter list on its own line, which this handles
    correctly), and strictly better than the pre-fix behavior it replaces
    (which matched the WRONG "class" -- the one inside `<class T>` -- for
    the right answer by accident, but had the exact same failure mode as
    the bugs above for every other case).
    """
    text = stripped_line.lstrip()
    while True:
        m = _CONTAINER_MODIFIER_RE.match(text)
        if not m:
            break
        text = text[m.end():]
    return bool(_CONTAINER_KEYWORD_RE.match(text))


def brace_boundary_indexes(lines, ext: str = "") -> set:
    """
    0-indexed line numbers after which it's safe to end a chunk for a
    curly-brace-block language (BRACE_BLOCK_EXTENSIONS) -- see
    python_boundary_indexes for indentation-delimited languages.

    A safe boundary is a line that leaves the CURRENT scope sitting either
    at file-level (no open brace) or directly inside a "container" scope
    (class/struct/interface/... -- see _line_declares_container) -- i.e.
    between two sibling members (two methods, a field and a method, ...) of
    that container, never inside a method body or a control-flow block
    (if/for/while/switch/lambda/...). PR #137 review (issue #55 follow-up):
    the original version only recognized depth-0 (file-level) as safe,
    which meant a whole class/namespace longer than chunk_size never had
    ANY boundary candidate inside it at all -- the exact "reduce mid-class
    splits" case the issue asks for. Each currently-open brace is tagged
    "container" (if its own opening line is ITSELF a container declaration,
    K&R- or Allman-style -- see _line_declares_container) or "other"
    (everything else -- methods, if/for/while/lambdas, object initializers,
    ...) as it opens; a line is a boundary only when the INNERMOST
    currently-open scope (or no open scope at all) is tagged "container",
    which is what keeps a multi-line if/for/lambda body from ever being
    mistaken for a member boundary -- it's tagged "other," so nothing
    inside it can look safe until it (and everything nested in it) actually
    closes back out to the enclosing container.

    A line is only a candidate while PAREN depth (from '(' and ')') is also
    0 (PR #137 review: a multi-line method/constructor SIGNATURE -- e.g.
    Allman-style `void Bar(\n    int a,\n    int b\n)` -- doesn't open any
    NEW brace of its own while its parameter list spans several lines, so
    without this check every one of those lines still looked like a safe
    member-level boundary, right up to (and including) the line closing the
    signature's own ')' -- confirmed by direct reproduction, and the review
    specifically flagged a chunk cut landing immediately before the
    method's own '{' as a result).

    Even with paren-depth tracked, the line ending a (possibly multi-line)
    signature's closing ')' is STILL not safe if the method/class's own
    Allman-style '{' is the very next line -- cutting there separates the
    declaration from its own body just as much as cutting mid-signature
    does. A final lookahead pass removes any candidate whose next real
    line of code is a bare '{' (skipping only genuinely blank/comment
    lines to find it), which also covers a class/struct declaration whose
    own opening brace is Allman-style on the following line.

    `ext` is the lowercase file extension (e.g. `".php"`) passed through to
    `_strip_c_style_line` for PHP-specific `#` line-comment handling (issue
    #138). All other extensions leave `#` untouched (see _strip_c_style_line).

    Strings and comments (including backtick template/raw-string literals
    -- see _strip_c_style_line) are stripped first so a brace inside one is
    never miscounted -- the same "safe direction" philosophy as
    heading_line_indexes' fence tracking above: a missed real boundary just
    falls back to a fixed-line cut (see chunk_code_by_boundaries), it never
    invents a wrong one.
    """
    n = len(lines)
    boundaries = set()
    in_block_comment = False
    in_template_string = False
    in_verbatim_string = False
    frame_tags: list = []  # one entry per currently open '{', outer to inner
    pending_container = False
    paren_depth = 0
    stripped_lines: list = [""] * n
    for i, line in enumerate(lines):
        stripped, in_block_comment, in_template_string, in_verbatim_string = _strip_c_style_line(
            line, in_block_comment, in_template_string, in_verbatim_string, ext=ext
        )
        stripped_lines[i] = stripped
        if _line_declares_container(stripped):
            pending_container = True
        for ch in stripped:
            if ch == "{":
                frame_tags.append("container" if pending_container else "other")
                pending_container = False
            elif ch == "}":
                if frame_tags:
                    frame_tags.pop()
            elif ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(paren_depth - 1, 0)
            elif ch == ";":
                pending_container = False
        # len(frame_tags) tracks depth exactly (one entry pushed per '{',
        # popped per '}'), so "no open frames" already implies depth <= 0 --
        # no separate depth check needed on top of in_container_scope.
        in_container_scope = not frame_tags or frame_tags[-1] == "container"
        if in_container_scope and paren_depth == 0:
            boundaries.add(i)

    filtered = set()
    for i in boundaries:
        j = i + 1
        while j < n and not stripped_lines[j].strip():
            j += 1
        if j < n and stripped_lines[j].lstrip().startswith("{"):
            continue
        filtered.add(i)
    return filtered


# A real top-level (or nested) class declaration -- matched against the
# STRIPPED line (comments/strings already blanked), so a line that's really
# pure string/docstring content (e.g. the word "class" appearing inside a
# multi-line docstring) can never false-match here.
_PY_CLASS_RE = re.compile(r"^\s*class\s+\w")


def python_boundary_indexes(lines) -> set:
    """
    0-indexed line numbers after which it's safe to end a chunk for Python
    source.

    Two independent checks, both must pass:

    1. Bracket depth: tracks unclosed '(', '[', '{' spanning a line into the
       next (a multi-line function call, list, or dict literal) -- a line
       is only a candidate when that depth is back to 0, so a chunk never
       splits mid-multi-line-expression.
    2. Indentation level: the next real line of code must return to EITHER
       column 0 (top level -- between module-level statements/defs/classes)
       OR the member indentation level of the nearest still-open class body
       (between two sibling methods/fields of the same class) -- tracked via
       a small indentation stack keyed on `class` declarations
       (_PY_CLASS_RE). PR #137 review (issue #55 follow-up): the original
       version only recognized column 0, so a whole class body longer than
       chunk_size never had ANY boundary candidate between its own methods
       -- the exact "reduce mid-class splits" case the issue asks for.

    A line is EXCLUDED from being a candidate if it's still inside an
    unterminated triple-quoted string when it ends (PR #137 review finding:
    a line like `return \"\"\"` -- unterminated on this line -- must never be
    treated as safe just because the *string's own content* on a later line
    happens to look unindented; that content isn't real code at all). For
    the same reason, a line that STARTED already inside an open
    triple-quoted string (pure string payload, not code) never participates
    in the indentation stack above -- its raw leading whitespace reflects
    the string's content, not real code structure, and could otherwise
    spuriously pop a class context that hasn't actually closed.

    A line is also EXCLUDED if it's part of a decorator statement (`@foo`,
    possibly spanning multiple lines via unclosed parens, e.g.
    `@app.route(\n    "/x",\n)`) -- PR #137 review: a decorator sits at the
    same indentation as the def/class it decorates, so without this check
    it satisfied the indentation-level rule above and could get split from
    its own target (confirmed by direct reproduction: `['@route', 'def
    handler():', ...]` flagged the '@route' line itself as a boundary).

    A comment-only line is treated the same as a blank line everywhere
    above -- both when scanning for "the next real line of code" and when
    updating the class-indentation stack -- never as real code (PR #137
    review: Python comments carry no indentation meaning at all, so a
    comment sitting at a class's member-indentation level in between two
    statements that are STILL part of the same method's body was
    previously read as if it were a real dedent back to member level,
    fabricating a boundary mid-method; confirmed by direct reproduction).
    A line's own STRIPPED form (comments/strings already blanked) is
    checked for this, so a comment-only line -- which strips down to pure
    whitespace -- is detected uniformly the same way a truly blank line
    already is.

    IMPORTANT distinction (PR #137 review): a line that strips down to
    whitespace because it's a REAL bare string-literal statement (the
    entire body of a function whose only content is a multi-line
    docstring) is NOT "ignorable" the same way a blank/comment line is --
    it's the function's actual (only) body
    statement, and its own raw indentation is real code structure, not
    something to skip past while hunting for "the next real line." Without
    this distinction, the scan for foo()'s own header line would skip
    straight over its entire multi-line docstring body (mistaking it for
    ignorable filler) and land on the FOLLOWING top-level def, making
    foo()'s header look boundary-safe -- splitting it from its own body
    (confirmed by direct reproduction). Tracked via whether a line has any
    triple-quote interaction at all (opens, continues, or closes one) --
    such a line is always treated as "real," even when its own stripped
    text is blank.

    Strings and comments are stripped first (_strip_python_style_line) so a
    bracket character inside a string literal, a '#' comment, or a
    triple-quoted docstring is never miscounted.
    """
    n = len(lines)
    depth = 0
    triple_quote = None
    depths = [0] * n
    triple_quote_after: list = [None] * n
    triple_quote_entering: list = [None] * n
    body_indent_after: list = [None] * n
    decorator_flags: list = [False] * n
    stripped_lines: list = [""] * n
    class_stack: list = []  # dicts: {"header_indent": int, "body_indent": Optional[int]}
    current_is_decorator = False
    for i, line in enumerate(lines):
        entering_depth = depth
        entering_triple_quote = triple_quote
        triple_quote_entering[i] = entering_triple_quote
        stripped, triple_quote = _strip_python_style_line(line, triple_quote)
        stripped_lines[i] = stripped
        depth += (
            stripped.count("(") + stripped.count("[") + stripped.count("{")
            - stripped.count(")") - stripped.count("]") - stripped.count("}")
        )
        depths[i] = depth
        triple_quote_after[i] = triple_quote

        if entering_depth == 0:
            # A new statement starts here (the previous one, if any, had
            # fully closed its own brackets by the end of the prior line) --
            # decide fresh whether THIS statement is a decorator invocation.
            # A continuation line (entering_depth != 0, still inside a
            # multi-line decorator's own parens) keeps whatever value this
            # was already set to, so the whole span stays tagged together.
            current_is_decorator = stripped.lstrip().startswith("@")
        decorator_flags[i] = current_is_decorator

        # stripped.strip() (not line.strip()) so a comment-only line -- real
        # text before stripping, pure whitespace after -- never updates the
        # class-indentation stack as if it were a real statement either.
        if stripped.strip() and not entering_triple_quote:
            indent = len(line) - len(line.lstrip())
            while class_stack and indent <= class_stack[-1]["header_indent"]:
                class_stack.pop()
            if class_stack and class_stack[-1]["body_indent"] is None:
                class_stack[-1]["body_indent"] = indent
            if _PY_CLASS_RE.match(stripped):
                class_stack.append({"header_indent": indent, "body_indent": None})
        body_indent_after[i] = class_stack[-1]["body_indent"] if class_stack else None

    def _is_ignorable(j: int) -> bool:
        # Blank or comment-only -- AND no triple-quote interaction at all
        # (didn't open, continue, or close one here). A line that strips to
        # whitespace ONLY because it's entirely inside/opening/closing a
        # triple-quoted string is real code (a bare string-literal
        # statement), never ignorable filler to skip past.
        return (
            not stripped_lines[j].strip()
            and not triple_quote_entering[j]
            and not triple_quote_after[j]
        )

    boundaries = set()
    for i in range(n):
        if depths[i] > 0 or triple_quote_after[i] or decorator_flags[i]:
            continue
        j = i + 1
        while j < n and _is_ignorable(j):
            j += 1
        if j == n:
            boundaries.add(i)
            continue
        next_indent = len(lines[j]) - len(lines[j].lstrip())
        member_indent = body_indent_after[i]
        if next_indent == 0 or (member_indent is not None and next_indent == member_indent):
            boundaries.add(i)
    return boundaries


def chunk_code_by_boundaries(text: str, chunk_size: int, overlap: int, boundary_indexes_fn):
    """
    Boundary-aware variant of chunk_lines() (issue #55): keeps the same
    chunk_size/overlap *budget*, but nudges each chunk's actual end line
    back to the nearest safe boundary (per boundary_indexes_fn -- see
    brace_boundary_indexes/python_boundary_indexes above) instead of an
    arbitrary fixed line count, so a function/class body is far less likely
    to get split across two chunks.

    The boundary search only looks within the back half of the chunk_size
    window (from chunk_size//2 lines in, up to the fixed-window end) --
    searching the WHOLE window back to `start` would let an early boundary
    (e.g. right after a one-line import near the top) nudge every chunk
    down to a tiny size, defeating chunk_size as a real budget. If no
    boundary exists in that back-half window at all -- one very long
    unbroken function, or minified/generated code with no boundaries -- this
    falls back to the exact fixed-line cut chunk_lines() would have made, so
    a chunk never grows unboundedly and this never gets stuck.

    Precondition: same as chunk_lines() -- overlap < chunk_size (and
    chunk_size >= 1, overlap >= 0); see validate_chunk_params, which every
    caller of this function is expected to have already checked.
    """
    lines = text.splitlines()
    if not lines:
        return
    n = len(lines)
    boundaries = boundary_indexes_fn(lines)
    start = 0
    while start < n:
        target_end = min(start + chunk_size, n)
        end = target_end
        if target_end < n:
            lookback_floor = start + max(chunk_size // 2, 1)
            for idx in range(target_end - 1, lookback_floor - 1, -1):
                if idx in boundaries:
                    end = idx + 1  # end is exclusive; boundary idx is inclusive
                    break
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            yield chunk, start + 1, end
        if end >= n:
            break
        # Guard against a boundary-nudged `end` landing so close to `start`
        # that end - overlap <= start -- chunk_lines() can't hit this (its
        # end is always exactly start + chunk_size), but a boundary-nudged
        # end can be much smaller, so this must force forward progress
        # explicitly rather than relying on the fixed-window math above.
        start = max(end - overlap, start + 1)


# A real ATX heading: up to 3 leading spaces, 1-6 '#', then whitespace or end
# of line. Requiring that trailing space is what separates a heading from a
# '#!/usr/bin/env' shebang, a '#include', or a '#comment' with no space -- all
# of which matched the old bare startswith("#") test.
_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:\s|$)")

# Fence delimiters, tracked so a '# comment' inside a ``` block is not read as
# a heading. That's the common shape in a README full of shell and Python
# samples, where each commented sample line would otherwise start a section.
#
# Captures the run of markers and whatever follows, because which fences OPEN
# and which CLOSE depends on both (see below).
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def heading_line_indexes(lines):
    """
    Indexes of lines that are real ATX headings, ignoring anything inside a
    fenced code block.

    Closing a fence follows CommonMark: only a run of the SAME character, at
    least as long as the opener, with nothing but whitespace after it. A naive
    "any fence toggles" rule breaks the standard way of showing markdown inside
    markdown -- a ````-fenced block containing a ``` sample -- where the inner
    ``` closes the block early and every '#' line after it reads as a heading.
    Tracking the character matters too, or a ~~~ line ends a ``` block.

    An unbalanced fence leaves the remainder of the file treated as code, which
    costs some heading splits but never invents them -- the safe direction,
    since a false heading fragments a chunk mid-thought.
    """
    indexes = []
    fence = None  # (marker char, marker length) while inside a block
    for i, line in enumerate(lines):
        match = _FENCE_RE.match(line)
        if match:
            marker, trailing = match.group(1), match.group(2)
            if fence is None:
                # Opening fence. A trailing info string ("```bash") is allowed.
                fence = (marker[0], len(marker))
            elif (
                marker[0] == fence[0]
                and len(marker) >= fence[1]
                and not trailing.strip()
            ):
                fence = None
            # Any other fence-looking line is content inside the open block.
            continue
        if fence is None and _ATX_HEADING_RE.match(line):
            indexes.append(i)
    return indexes


def chunk_markdown(text: str, fallback_chunk_size: int, fallback_overlap: int):
    """
    Split markdown on headings; falls back to line chunking when there aren't
    at least two headings to split on.

    Content before the first heading -- YAML frontmatter, an intro paragraph --
    is emitted as its own chunk. It used to be skipped entirely: chunking began
    at the first heading, so those lines were never indexed and nothing
    reported them missing.
    """
    lines = text.splitlines()
    heading_indexes = heading_line_indexes(lines)
    if len(heading_indexes) < 2:
        yield from chunk_lines(text, fallback_chunk_size, fallback_overlap)
        return
    preamble = "\n".join(lines[: heading_indexes[0]]).strip()
    if preamble:
        yield preamble, 1, heading_indexes[0]
    bounds = heading_indexes + [len(lines)]
    for i in range(len(heading_indexes)):
        start, end = bounds[i], bounds[i + 1]
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            yield chunk, start + 1, end


def scoped_extensions(scope: str, include_extensions: Optional[set] = None) -> set:
    """
    Returns the file extensions to include. If include_extensions is given,
    it's used as-is (normalized) and scope is ignored -- explicit inclusion
    always wins over the code/docs/both default buckets.
    """
    if include_extensions:
        return normalize_extensions(include_extensions)
    extensions = set()
    if scope in ("code", "both"):
        extensions |= CODE_EXTENSIONS
    if scope in ("docs", "both"):
        extensions |= DOC_EXTENSIONS
    return extensions


def chunk_file(path: Path, rel_path: str, chunk_size: int, overlap: int):
    """
    Chunk a single file, picking the code or doc chunker based on extension.
    Yields (content, metadata) tuples. Used both for full-repo builds and for
    re-chunking individual changed files during an incremental sync.
    """
    # encoding is explicit on purpose. Without it, decoding follows the
    # platform locale (cp1252 on a default Windows box, UTF-8 on macOS/Linux),
    # so the same file yields different chunk text on different machines --
    # and with errors="ignore" the difference is silently dropped bytes rather
    # than an exception. compute_file_hashes() reads BYTES, so the hash stays
    # identical across platforms while the indexed content diverges: an
    # incremental sync sees no change and never corrects it.
    text = path.read_text(encoding="utf-8", errors="ignore")
    suffix = path.suffix.lower()
    file_type = "doc" if suffix in DOC_EXTENSIONS else "code"
    if suffix in MARKDOWN_EXTENSIONS:
        generator = chunk_markdown(text, chunk_size, overlap)
    elif suffix in BRACE_BLOCK_EXTENSIONS:
        # Pass the extension to brace_boundary_indexes so PHP's '#' line
        # comments are recognized correctly without affecting other languages
        # (issue #138 -- see _strip_c_style_line's `ext` parameter).
        generator = chunk_code_by_boundaries(
            text, chunk_size, overlap,
            functools.partial(brace_boundary_indexes, ext=suffix),
        )
    elif suffix in INDENT_BLOCK_EXTENSIONS:
        generator = chunk_code_by_boundaries(text, chunk_size, overlap, python_boundary_indexes)
    else:
        generator = chunk_lines(text, chunk_size, overlap)
    for chunk, start, end in generator:
        yield chunk, {
            "file_path": rel_path,
            "line_range": f"{start}-{end}",
            "type": file_type,
        }


def build_entries(
    repo_path: Path,
    scope: str,
    chunk_size: int,
    overlap: int,
    include_extensions: Optional[set] = None,
    extra_exclude_dirs: Optional[set] = None,
    respect_gitignore: bool = True,
    extra_exclude_files: Optional[set] = None,
) -> tuple:
    """
    Returns (entries, skipped): entries is a list of (content, metadata)
    tuples for every matching file across the whole repo that could actually
    be read; skipped is a list of (rel_path, error message) for any file
    that raised while being read/chunked (broken symlink, permission error,
    etc.) -- one bad file no longer aborts the whole run (issue #27).
    scope is one of "code", "docs", "both" -- ignored if include_extensions
    is given. extra_exclude_dirs adds project-specific folders to skip on
    top of the built-in EXCLUDE_DIRS. extra_exclude_files skips individual
    files by name (see iter_files). respect_gitignore additionally skips
    anything the repo's own .gitignore excludes (silently a no-op if there's
    no .gitignore or the optional pathspec package isn't installed).
    """
    gitignore_spec = load_gitignore_spec(repo_path) if respect_gitignore else None
    extensions = scoped_extensions(scope, include_extensions)
    entries = []
    skipped = []
    for f in iter_files(repo_path, extensions, extra_exclude_dirs, gitignore_spec, extra_exclude_files):
        rel = str(f.relative_to(repo_path))
        try:
            entries.extend(chunk_file(f, rel, chunk_size, overlap))
        except OSError as e:
            skipped.append((rel, str(e)))
    return entries, skipped


def iter_entries(
    repo_path: Path,
    scope: str,
    chunk_size: int,
    overlap: int,
    include_extensions: Optional[set] = None,
    extra_exclude_dirs: Optional[set] = None,
    respect_gitignore: bool = True,
    extra_exclude_files: Optional[set] = None,
) -> Generator:
    """
    Generator variant of build_entries() -- yields one (content, metadata)
    tuple per chunk as each file is processed, instead of accumulating the
    whole repo's chunks in memory before returning.

    Errors for a single file are yielded as (None, (rel_path, error_message))
    sentinel tuples so the caller can separate good chunks from bad files
    without losing either. Callers that need to keep a running error list
    should check `content is None` on each yielded item:

        for content, metadata_or_err in iter_entries(...):
            if content is None:
                rel, err = metadata_or_err
                skipped.append((rel, err))
            else:
                # content and metadata are both valid

    Unlike build_entries(), this never holds more than one file's chunks in
    memory at a time, which matters for very large repos (issue #57). It also
    enables progress reporting DURING the chunking phase -- the caller can
    count yielded items and report progress without waiting for the entire
    repo to be scanned first.

    All filtering parameters (scope, include_extensions, extra_exclude_dirs,
    respect_gitignore, extra_exclude_files) work identically to build_entries()
    -- keep them consistent between calls for the same repo.
    """
    gitignore_spec = load_gitignore_spec(repo_path) if respect_gitignore else None
    extensions = scoped_extensions(scope, include_extensions)
    for f in iter_files(repo_path, extensions, extra_exclude_dirs, gitignore_spec, extra_exclude_files):
        rel = str(f.relative_to(repo_path))
        try:
            yield from chunk_file(f, rel, chunk_size, overlap)
        except OSError as e:
            yield None, (rel, str(e))


def compute_file_hashes(
    repo_path: Path,
    scope: str,
    include_extensions: Optional[set] = None,
    extra_exclude_dirs: Optional[set] = None,
    respect_gitignore: bool = True,
    extra_exclude_files: Optional[set] = None,
) -> tuple:
    """
    Returns (hashes, skipped). hashes maps relative file path -> sha256 hex
    digest of its current content, for every matching file that could
    actually be read -- used to detect which files changed since the last
    sync without re-embedding anything. skipped is a list of (rel_path,
    error message) for any file that raised while being hashed (broken
    symlink, permission error, etc.) -- one bad file no longer aborts the
    whole sync (issue #27); callers must NOT treat a skipped file as
    "removed" just because it's absent from hashes -- see
    files_removed_since_last_sync below. Filtering params match
    build_entries -- keep them consistent between calls or the diff will
    look like every file changed. extra_exclude_files skips individual
    files by name (see iter_files).
    """
    gitignore_spec = load_gitignore_spec(repo_path) if respect_gitignore else None
    extensions = scoped_extensions(scope, include_extensions)
    hashes = {}
    skipped = []
    for f in iter_files(repo_path, extensions, extra_exclude_dirs, gitignore_spec, extra_exclude_files):
        rel = str(f.relative_to(repo_path))
        try:
            hashes[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError as e:
            skipped.append((rel, str(e)))
    return hashes, skipped


def files_removed_since_last_sync(manifest_files, current_hashes: dict, skipped_paths) -> list:
    """
    Which manifest-tracked files sync_repo should treat as removed.

    A file counts as removed only if it's genuinely gone from disk (or no
    longer matches the scope/extension/exclude filters) -- i.e. iter_files
    simply didn't yield it, AND hashing it didn't raise either. A file that
    failed to hash (skipped_paths, from compute_file_hashes) is neither
    "removed" nor "changed": its current state is unknown, so its existing
    embeddings are left alone rather than deleted -- treating an unreadable
    file as removed would silently delete good, previously-indexed content
    over what may just be a transient read error (issue #27).
    """
    skipped_set = set(skipped_paths)
    return [f for f in manifest_files if f not in current_hashes and f not in skipped_set]


def ensure_persistent_fastembed_cache() -> None:
    """
    fastembed's own define_cache_dir() (third-party, not this repo's code)
    defaults an unset FASTEMBED_CACHE_PATH to the OS temp directory
    (tempfile.gettempdir()/"fastembed_cache"), which gets wiped on reboot or
    periodic temp cleanup -- silently reintroducing a network dependency on
    huggingface.co the next time a FastEmbedProvider is constructed (issue
    #77). Point it at a persistent, toolkit-owned directory instead, unless
    the caller already set their own value -- .mcp.json's codebase-indexer
    env block or a shell export both take priority via setdefault().

    Must run BEFORE FastEmbedProvider(...)/TextEmbedding(...) is constructed
    anywhere: fastembed reads this env var directly at that point rather than
    accepting it as a constructor parameter, so setting it any later has no
    effect on that construction.

    ~/.claude/claude-runway/fastembed-cache matches the convention already
    established by libs/savings_ledger.py's resolve_db_path() (same home-dir
    namespace, same reasoning -- stable across reinstalls/moves of this tools
    repo, no accidental-commit risk). Deliberately NOT wired through
    templates/mcp.json.template as the issue originally suggested: fastembed's
    define_cache_dir() does not call .expanduser() on the env var's value, so
    a literal "~/..." string placed in JSON or a shell export would create a
    directory literally named "~" in the process's cwd instead of expanding
    to the home directory. Resolving Path.home() here in code sidesteps that
    landmine entirely and needs zero per-project configuration.
    """
    os.environ.setdefault(
        "FASTEMBED_CACHE_PATH",
        str(Path.home() / ".claude" / "claude-runway" / "fastembed-cache"),
    )


def batched(items: list, size: int):
    """
    Yields successive `size`-length slices of `items` (the final slice may be
    shorter). Used by sync_repo to group per-file Qdrant delete filters into
    chunks instead of one client.delete() call per file -- issue #75/#76
    found that thousands of individual round-trips in a tight loop (not an
    idle gap) can exhaust local ephemeral ports / Docker Desktop's
    connection-tracking table, breaking the very next connection attempt.
    Confirmed empirically that a single filter covering thousands of
    conditions can time out server-side, so this can't just be "one big
    request" either -- both extremes (one call per item, one call for
    everything) are wrong; chunking is the fix.
    """
    for i in range(0, len(items), size):
        yield items[i:i + size]
