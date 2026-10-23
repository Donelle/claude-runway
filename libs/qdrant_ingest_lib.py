"""
Shared, side-effect-free chunking helpers used by both:
  - ingest_to_qdrant.py   (standalone CLI script)
  - ingest_mcp_server.py  (MCP server exposing the same logic as tools)

Kept dependency-free (no qdrant-client / fastembed imports here) so
it's easy to unit test and reuse. No print() calls in this file on purpose --
ingest_mcp_server.py runs over stdio, where stray prints corrupt the protocol.
"""

import bisect
import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Optional, Sequence

# A complete Rust char literal, tried at a `'` in .rs files only:
# `'x'` (one non-quote, non-backslash character) or a backslash escape --
# `'\n'`, `'\''`, `'\\'`, `'\x41'`, `'\u{1F600}'`. Rust char literals are short
# and always close; a lifetime or loop label (`'a`, `'static`, `'outer:`) is
# followed by an identifier and never has a second `'` right after one
# character, so it fails this match and is treated as a plain tick.
_RUST_CHAR_LITERAL = re.compile(
    r"'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^'\\])'"
)

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
# -- every CODE_EXTENSIONS member whose blocks are delimited by
# '{'/'}' rather than indentation or a non-brace keyword (.rb's 'end', for
# example, isn't covered here -- out of the issue's named scope of "at least
# C#, Python, TypeScript", and there's no brace signal to hook a heuristic on).
BRACE_BLOCK_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".php", ".swift", ".kt", ".scala", ".rs",
}

# Indentation-delimited-block languages that get boundary-aware chunking via
# python_boundary_indexes() instead of brace_boundary_indexes() --
# Python has no braces, so "depth returns to zero" has to mean something else
# (see python_boundary_indexes' own docstring).
INDENT_BLOCK_EXTENSIONS = {".py"}

EXCLUDE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "dist", "build", "__pycache__",
    ".next", "target", "vendor", ".idea", ".vscode", "coverage",
}


# sync_repo's bookkeeping file at the repo root. Lives here rather than in
# ingest_mcp_server.py so the CLI can exclude it too: .json is in
# CODE_EXTENSIONS, so every build_entries/compute_file_hashes caller has to
# pass extra_exclude_files={MANIFEST_FILENAME} or it embeds the manifest
# itself -- the CLI was the one caller that didn't.
MANIFEST_FILENAME = ".qdrant_index_manifest.json"

# Side files a caller-owned bookkeeping file such as the manifest has next to
# it: the lock that serializes its writers (`<name>.lock`, which stays on disk
# between runs) and the temp files its atomic write goes through
# (`<name>.<random>.tmp`, normally gone once the write finishes, but a crash
# between creating and renaming one can leave it behind). iter_files skips both
# for every name in extra_exclude_files, so neither can be hashed or indexed
# even under an extension list that happens to include .lock or .tmp.
BOOKKEEPING_LOCK_SUFFIX = ".lock"
BOOKKEEPING_TEMP_SUFFIX = ".tmp"
MANIFEST_LOCK_FILENAME = MANIFEST_FILENAME + BOOKKEEPING_LOCK_SUFFIX


def is_excluded_file_name(name: str, exclude_files) -> bool:
    """
    Whether iter_files skips a file called `name`: it is one of
    `exclude_files`, or one of their side files (see BOOKKEEPING_LOCK_SUFFIX).
    The temp-file match needs both the `<name>.` prefix and the .tmp suffix, so
    an unrelated file that merely starts with an excluded name is still indexed.
    """
    for base in exclude_files:
        if name == base or name == base + BOOKKEEPING_LOCK_SUFFIX:
            return True
        if name.startswith(base + ".") and name.endswith(BOOKKEEPING_TEMP_SUFFIX):
            return True
    return False

# How many files compute_file_hashes visits between two on_progress calls.
# Reading and hashing one file is quick, so a per-file report
# would cost more MCP round-trips than the hashing itself on a large repo;
# every few hundred files is frequent enough that the call never looks frozen
# and rare enough to be noise-free.
HASH_PROGRESS_EVERY = 200

# The manifest is versioned. It is a wrapper, not the flat
# {file: hash} map it used to be, and it keeps the SAME filename on purpose:
# users already gitignore ".qdrant_index_manifest.json", and a new name would
# put a new untracked file into every repo. The manifest only tracks file
# CONTENT, so it cannot tell that the chunker changed; bumping the version is
# what makes the first sync after a chunker change rebuild the index instead
# of reporting "No changes" and serving stale chunks forever. A manifest in any
# other shape (the old flat map, or a newer version this code does not know) is
# ignored, which makes every file count as changed. Keys inside "files" stay
# file paths only: files_removed_since_last_sync treats every key as a file.
MANIFEST_FORMAT_VERSION = 2
MANIFEST_FORMAT_KEY = "manifest_format"
MANIFEST_FILES_KEY = "files"
# Additive key beside "files": paths a rebuild found indexed and still on disk
# but outside its scan, so it neither re-embedded nor hashed them. They are kept
# apart from the hash map on purpose, because files_removed_since_last_sync
# treats every key of that map as a file the scan should yield, and a narrower
# scope would then delete a still-present file's chunks. The key is absent when
# the set is empty, and a reader that predates it never looks at it, so adding
# it needed NO MANIFEST_FORMAT_VERSION bump: the version is what forces a full
# rebuild on every project that has the toolkit installed, and a manifest that
# is merely richer must not do that.
MANIFEST_PRESERVED_KEY = "preserved_untracked"

# Built-in fallbacks for ingest_env_defaults() below -- what both the MCP
# server and the CLI use when the matching env var is unset or blank.
FALLBACK_QDRANT_URL = "http://localhost:6333"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_MEMORY_BANK_COLLECTION = "memory-bank"


def parse_csv_set(value: Optional[str]) -> Optional[set]:
    """
    Parses a comma-separated list (INDEX_INCLUDE_EXTENSIONS, --include-ext,
    an index_repo include_extensions argument, ...) into a set, stripping
    whitespace around each entry and dropping empty ones. Returns None for
    a None/empty value or one with no non-blank entries, so callers can
    write `parse_csv_set(x) or default`.

    Shared by ingest_mcp_server.py and ingest_to_qdrant.py:
    the CLI used to do a bare split(","), so ".py, .md" became
    {".py", " .md"} -> normalize_extensions -> {".py", ". .md"} and every
    .md file was silently skipped.
    """
    if not value:
        return None
    return {v.strip() for v in value.split(",") if v.strip()} or None


def ingest_env_defaults(environ=None) -> dict:
    """
    The env-var-backed defaults every ingestion entry point shares, resolved
    in one place so the MCP server and the CLI can't disagree about them
    again (the CLI used to hardcode every one of these and
    ignore the env entirely). Keys: qdrant_url, qdrant_api_key,
    embedding_model, include_extensions, exclude_dirs,
    memory_bank_collection.

    Reads the same env var names qdrant_mcp_server and memory_bank_mcp_server
    read (QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL, MEMORY_BANK_COLLECTION)
    plus this repo's INDEX_INCLUDE_EXTENSIONS/INDEX_EXCLUDE_DIRS. A blank
    value counts as unset -- the rule MEMORY_BANK_COLLECTION already had --
    since templates/mcp.json.template declares several of these as "" and a
    blank URL or model name is never a usable value (a blank API key means
    "no key", i.e. None). include_extensions/exclude_dirs are returned RAW
    (the comma-separated string, or None) so the CLI can use them directly as
    argparse defaults; run them through parse_csv_set to get sets.

    Only reads the process environment. For the MCP server that is its
    .mcp.json env block plus whatever it inherited; for the CLI it is the
    shell it was run from, which normally does NOT carry .mcp.json's values;
    for the memory-bank collection name the CLI closes that gap with
    resolve_memory_bank_collection below.
    """
    env = os.environ if environ is None else environ
    return {
        "qdrant_url": env.get("QDRANT_URL") or FALLBACK_QDRANT_URL,
        "qdrant_api_key": env.get("QDRANT_API_KEY") or None,
        "embedding_model": env.get("EMBEDDING_MODEL") or FALLBACK_EMBEDDING_MODEL,
        "include_extensions": env.get("INDEX_INCLUDE_EXTENSIONS") or None,
        "exclude_dirs": env.get("INDEX_EXCLUDE_DIRS") or None,
        "memory_bank_collection": env.get("MEMORY_BANK_COLLECTION") or FALLBACK_MEMORY_BANK_COLLECTION,
    }


# The .mcp.json server blocks that carry MEMORY_BANK_COLLECTION, in the order
# resolve_memory_bank_collection consults them. setup_project_lib writes the
# same value into both; codebase-indexer comes first because its copy is the
# one index_repo/sync_repo actually guard with, and memory-bank (the server
# that owns the collection) is the fallback for a hand-edited config that only
# set it there. When a hand edit leaves the two blocks naming different
# collections, both names are reserved: memory-bank's is where memories are
# stored, and codebase-indexer's is the one index_repo guards.
MEMORY_BANK_MCP_JSON_SERVERS = ("codebase-indexer", "memory-bank")


@dataclass(frozen=True)
class MemoryBankCollection:
    """Which collection is the shared memory bank, and where that came from.
    `source` is human-readable, for an error message; `warning` is set when a
    .mcp.json that exists could not be read and was skipped, or when its
    sources disagree. `also_reserved` holds a (name, source) pair for each
    other collection a .mcp.json block names; use `reserved_source` to check a
    collection against all of them."""

    name: str
    source: str
    warning: Optional[str] = None
    also_reserved: tuple = ()

    def reserved_source(self, collection: str) -> Optional[str]:
        """Where `collection` is configured as the memory bank, or None if it
        is not one of the reserved names."""
        for name, source in ((self.name, self.source), *self.also_reserved):
            if collection == name:
                return source
        return None


def resolve_memory_bank_collection(
    explicit: Optional[str], repo_path, environ=None
) -> MemoryBankCollection:
    """
    Resolves the shared memory-bank collection name for a caller that, unlike
    the MCP servers, does not run with .mcp.json's env block:
    ingest_to_qdrant.py runs from a normal shell, and this toolkit never asks
    for MEMORY_BANK_COLLECTION to be exported there -- the value lives in
    .mcp.json. Reading only the shell meant a project with a custom name had
    its memory bank refused by index_repo but not by the CLI.

    `name` comes from the first non-blank of:
      1. `explicit` (the CLI's --memory-bank-collection)
      2. MEMORY_BANK_COLLECTION in `environ` (default os.environ); a blank or
         whitespace-only value counts as unset
      3. `<repo_path>/.mcp.json`, each MEMORY_BANK_MCP_JSON_SERVERS block's env
      4. FALLBACK_MEMORY_BANK_COLLECTION ("memory-bank")

    Every .mcp.json block is read even when step 1 or 2 supplies the name,
    because those blocks say where the MCP servers actually store memories,
    and an override does not move them. Any name a block gives that differs
    from `name` goes in `also_reserved`, and `warning` reports the
    disagreement, so the CLI refuses all of them.

    `repo_path` may be None to skip reading .mcp.json. The file is read with
    doctor_lib.load_server_env, the loader `doctor` already uses for the same
    file, so the two cannot disagree about what a well-formed block is. It is
    imported here rather than at module level to keep this module free of
    import-time dependencies (the MCP server imports it too). A missing file
    or server block falls through silently. A malformed one never raises:
    this guard must not turn a broken .mcp.json into a crashed index run, so
    the block is skipped and `warning` says so, and the caller decides whether
    to print it.
    """
    env = os.environ if environ is None else environ
    # Whitespace-only counts as unset too, not just "" (PR review): a value of
    # "  " is truthy, and returning it would let it outrank .mcp.json's real
    # name and leave that collection unprotected. Every source here strips.
    env_value = (env.get("MEMORY_BANK_COLLECTION") or "").strip()
    # (name, source for an error message, short label for the warnings)
    if explicit and explicit.strip():
        primary = (explicit.strip(), "--memory-bank-collection", "--memory-bank-collection")
    elif env_value:
        primary = (env_value, "the MEMORY_BANK_COLLECTION environment variable", "the shell")
    else:
        primary = None
    # A dict, not a list, to drop repeats: a file that is not valid JSON (or
    # has a malformed "mcpServers") fails with the same message for every
    # server block, and saying it twice adds nothing.
    problems: dict = {}
    found = []  # (name, source, label), in MEMORY_BANK_MCP_JSON_SERVERS order
    if repo_path is not None:
        from doctor_lib import MalformedMcpJsonError, load_server_env

        mcp_json_path = Path(repo_path) / ".mcp.json"
        for server in MEMORY_BANK_MCP_JSON_SERVERS:
            try:
                server_env = load_server_env(mcp_json_path, server)
            except MalformedMcpJsonError as exc:
                problems[str(exc)] = None
                continue
            value = (server_env or {}).get("MEMORY_BANK_COLLECTION", "").strip()
            if value:
                found.append((value, f"MEMORY_BANK_COLLECTION in {mcp_json_path}'s \"{server}\" block",
                              f"the \"{server}\" block"))
    if primary is None and found:
        primary = found[0]
    if primary is None:
        return MemoryBankCollection(
            FALLBACK_MEMORY_BANK_COLLECTION,
            "the built-in default",
            _unreadable_mcp_json_warning(problems, f"using the default '{FALLBACK_MEMORY_BANK_COLLECTION}'"),
        )
    name, source, label = primary
    others: dict = {}  # name -> source, first block wins, so a name repeated across blocks is listed once
    for other, other_source, _ in found:
        if other != name:
            others.setdefault(other, other_source)
    # A block that could not be read is still worth reporting, even though
    # another source supplied the name.
    warnings = [_unreadable_mcp_json_warning(problems, f"using '{name}' from {label}")]
    if others:
        sources = [primary] + [entry for entry in found if entry is not primary]
        named = ", ".join(f"{entry_label} says '{entry_name}'" for entry_name, _, entry_label in sources)
        warnings.append(f"MEMORY_BANK_COLLECTION disagrees across its sources ({named}); refusing all of them")
    return MemoryBankCollection(name, source, "; ".join(w for w in warnings if w) or None, tuple(others.items()))


def _unreadable_mcp_json_warning(problems: dict, outcome: str) -> Optional[str]:
    if not problems:
        return None
    return f"could not read MEMORY_BANK_COLLECTION from .mcp.json ({'; '.join(problems)}); {outcome}"


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
    filter, or gitignore state. Each listed name's lock and atomic-write temp
    files are skipped with it (see is_excluded_file_name).
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
        if is_excluded_file_name(path.name, exclude_files):
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
    theoretical. chunk_size < 1 is the same class of defect from
    the other direction: chunk_lines()'s `end <= start` window then yields
    nothing at all (0) or a nonsensical chunk (negative) instead of raising
    cleanly. overlap < 0 has no valid meaning either.

    Only for callers of the line-window chunkers. No indexing entry point
    calls it any more: they chunk by the embedding model's token
    window, where `overlap` is a number of tokens and `chunk_lines` sets
    nothing, so they check validate_token_overlap instead. It was called once
    at every CLI/tool boundary rather than enforced inside
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
    containing template literal -- correctly
    treating THAT common case matters far more than the rare Go edge case.

    C# verbatim strings (`@"..."`) are the one common multi-line plain-
    quote form worth handling explicitly: unlike an
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
    explicitly ruled it out. All other `ext` values leave `#` alone.

    Rust's `'` is ambiguous: it opens a char literal (`'x'`,
    `'{'`) but is also the tick of every lifetime (`<'a>`, `&'a str`,
    `'static`) and loop label (`'outer:`). When `ext == ".rs"`, a `'` is
    only treated as a string opener if a complete char literal starts there
    (`_RUST_CHAR_LITERAL`); otherwise it is kept as an ordinary character and
    scanning continues. Treating every `'` as an opener, as the other
    languages here do, made a signature like `fn f<'a>(x: &'a str) -> &'a str {`
    swallow its own `{` from the third tick onward, so every line of the body
    looked like a depth-zero boundary -- a fabricated boundary, not a missed
    one. A char literal this regex misses (none in valid Rust) would at worst
    leave a stray brace visible, never hide one.

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
        elif ch == "'" and ext == ".rs":
            # Rust: a char literal, or a lifetime/label tick.
            m = _RUST_CHAR_LITERAL.match(line, i)
            if m:
                result.append(" " * (m.end() - i))
                i = m.end()
            else:
                result.append(ch)
                i += 1
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
# boundary: '\b' is satisfied between a word char and ANY
# non-word char, including '.' -- so '\bmodule\b' matched the start of
# 'module.exports = function () {' (a completely ordinary CommonJS
# expression, not a TypeScript module declaration), tagging that
# function's own brace 'container' and turning every line in its body into
# a false member-level boundary (confirmed by direct reproduction).
#
# The optional identifier group plus lookahead disambiguates a genuine
# declaration from the SAME keyword used as a plain TYPE NAME in a variable
# declaration: 'object value = new Foo {'
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

    Matching the keyword anywhere via `.search()` treated
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
    (if/for/while/switch/lambda/...). The original version only recognized depth-0 (file-level) as safe,
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
    0 (a multi-line method/constructor SIGNATURE -- e.g.
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
    `_strip_c_style_line` for PHP-specific `#` line-comment handling.
    All other extensions leave `#` untouched (see _strip_c_style_line).

    Strings and comments (including backtick template/raw-string literals
    -- see _strip_c_style_line) are stripped first so a brace inside one is
    never miscounted -- the same "safe direction" philosophy as
    heading_line_indexes' fence tracking below: a missed real boundary just
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
       (_PY_CLASS_RE). The original
       version only recognized column 0, so a whole class body longer than
       chunk_size never had ANY boundary candidate between its own methods
       -- the exact "reduce mid-class splits" case the issue asks for.

    A line is EXCLUDED from being a candidate if it's still inside an
    unterminated triple-quoted string when it ends (review finding:
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
    `@app.route(\n    "/x",\n)`) -- a decorator sits at the
    same indentation as the def/class it decorates, so without this check
    it satisfied the indentation-level rule above and could get split from
    its own target (confirmed by direct reproduction: `['@route', 'def
    handler():', ...]` flagged the '@route' line itself as a boundary).

    A comment-only line is treated the same as a blank line everywhere
    above -- both when scanning for "the next real line of code" and when
    updating the class-indentation stack -- never as real code
    (Python comments carry no indentation meaning at all, so a
    comment sitting at a class's member-indentation level in between two
    statements that are STILL part of the same method's body was
    previously read as if it were a real dedent back to member level,
    fabricating a boundary mid-method; confirmed by direct reproduction).
    A line's own STRIPPED form (comments/strings already blanked) is
    checked for this, so a comment-only line -- which strips down to pure
    whitespace -- is detected uniformly the same way a truly blank line
    already is.

    IMPORTANT distinction: a line that strips down to
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
    Boundary-aware variant of chunk_lines(): keeps the same
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


class TokenBudgetError(ValueError):
    """The token budget for chunking cannot be established.

    Raised before anything is chunked, deleted or written, and never worked
    around: falling back to line-based chunking, or to an invented window size,
    would silently reproduce the bug token-budget chunking exists to fix (most
    of every chunk never reaching its vector).
    """


def validate_token_overlap(overlap: int, max_tokens: Optional[int] = None) -> Optional[str]:
    """
    Error message for an unusable token overlap, or None.

    `overlap` is measured in TOKENS now (it used to be lines), so the upper
    bound is the embedding window, which is only known once the model is loaded:
    callers check `overlap >= 0` at the tool boundary (max_tokens=None), where
    the failure is cheapest, and the window bound when the budget is built. An
    overlap of max_tokens or more would make every chunk start inside the
    previous one's first token and never advance.
    """
    if overlap < 0:
        return f"overlap must be 0 or greater (got {overlap})."
    if max_tokens is not None and overlap >= max_tokens:
        return (
            f"overlap ({overlap} tokens) must be smaller than the embedding window's usable size "
            f"({max_tokens} tokens) -- otherwise a chunk could never advance past the previous one."
        )
    return None


@dataclass(frozen=True)
class TokenBudget:
    """How big a chunk may be, in the embedding model's own tokens.

    The library stays dependency-free, so it never imports a tokenizer: the
    indexer builds one of these from the loaded embedding model
    (libs/qdrant_embeddings.py's FastEmbedProvider.build_token_budget) and
    passes it in.

    max_tokens: the usable window, i.e. the model's truncation limit minus the
        special tokens the tokenizer adds around every text. A chunk of at most
        this many tokens is embedded in full.
    overlap_tokens: how many tokens the next chunk repeats from the end of the
        previous one, at any cut.
    token_spans: text -> [(start, end), ...], the character offsets of every
        real token in `text` (no special tokens), in order. The count of a text
        is len(token_spans(text)). Offsets, not just a count, because the cut
        has to land on a character position: a minified file is one line, so
        "cut after line N" has nothing to cut at.
    """

    max_tokens: int
    overlap_tokens: int
    token_spans: Callable[[str], list]

    def __post_init__(self):
        if self.max_tokens < 1:
            raise TokenBudgetError(f"The usable embedding window must be at least 1 token (got {self.max_tokens}).")
        err = validate_token_overlap(self.overlap_tokens, self.max_tokens)
        if err:
            raise TokenBudgetError(err)


def _choose_cut(text: str, pos: int, cut_min: int, cut_max: int, preferred: Sequence[int]) -> int:
    """
    Where a chunk that starts at `pos` should end, given that the token budget
    allows it to end anywhere up to `cut_max` and a cut earlier than `cut_min`
    would waste too much of the window.

    Prefers, in order, within [cut_min, cut_max]: (1) the last offset in
    `preferred` (a markdown heading, or the end of a safe code boundary from
    brace_boundary_indexes/python_boundary_indexes), (2) just after the last
    newline, (3) a whitespace boundary, (4) `cut_max` exactly. Never requires a
    preferred point to exist: a minified file or one long unbroken token has
    none, and cutting at the budget is then the only correct answer.
    """
    if cut_min >= cut_max:
        return cut_max
    i = bisect.bisect_right(preferred, cut_max) - 1
    if i >= 0 and preferred[i] >= cut_min and preferred[i] > pos:
        return preferred[i]
    newline = text.rfind("\n", max(cut_min - 1, pos), cut_max)
    if newline != -1:
        return newline + 1
    n = len(text)
    for p in range(cut_max, cut_min - 1, -1):
        if p >= n or text[p].isspace() or text[p - 1].isspace():
            return p
    return cut_max


def chunk_stream(text: str, budget: TokenBudget, preferred_cuts: Sequence[int] = ()):
    """
    Token-budgeted chunking over a stream of characters. Yields
    (chunk_text, start, end): `text[start:end] == chunk_text`, offsets into
    `text` as given (end exclusive).

    Replaces the line-window chunkers for every indexing entry point. Lines are
    the wrong unit: the embedder reads tokens and silently drops everything past
    its window, so a 50-line chunk of a JSON file had ~10% of its text
    searchable, and a minified file (one line) was one chunk with only its first
    window searchable. Here a chunk is whatever fits the model's window, wherever
    the line breaks fall.

    The chunk is not stripped, so offsets and stored text always agree (the old
    chunkers stripped, which shifted the start past leading blank lines). A
    chunk that is only whitespace is dropped.

    Each chunk is re-tokenized on its own after the cut and shortened until it
    fits: cutting in the middle of a word can make the shorter text tokenize into
    MORE tokens than the same characters did inside the longer one, so counting
    inside the window alone does not guarantee the stored text fits.

    Overlap (budget.overlap_tokens) is measured in tokens and applied at every
    cut, including cuts at a preferred point.

    A run of non-whitespace longer than the tokenizer's per-word limit comes
    back as ONE (unknown) token covering all of it, so a chunk can be far longer
    in characters than the window suggests, but never longer in tokens.
    """
    n = len(text)
    # Offset past which nothing but whitespace remains: a chunk that would be
    # only the overlap repeated over trailing whitespace is a duplicate.
    content_end = len(text.rstrip())
    if content_end == 0:
        return
    max_tokens = budget.max_tokens
    pos = 0
    while pos < n:
        # Ask the tokenizer about a bounded slice, not the whole rest of the
        # file (a minified bundle is millions of characters). Start from a
        # generous guess and double until the slice holds more than one chunk's
        # worth of tokens or reaches the end of the text.
        window_chars = max_tokens * 4
        while True:
            window_end = min(pos + window_chars, n)
            spans = budget.token_spans(text[pos:window_end])
            if len(spans) > max_tokens or window_end >= n:
                break
            window_chars *= 2
        if len(spans) <= max_tokens:
            end = n  # the rest of the text fits in one chunk
        else:
            cut_max = pos + spans[max_tokens - 1][1]
            cut_min = pos + spans[max(max_tokens // 2, 1) - 1][1]
            end = _choose_cut(text, pos, cut_min, cut_max, preferred_cuts)
        # Never a zero-length chunk, whatever the offsets say.
        end = max(end, pos + 1)
        chunk = text[pos:end]
        chunk_spans = budget.token_spans(chunk)
        while len(chunk_spans) > max_tokens:
            # Strictly shorter on every pass. The end of the max_tokens-th
            # token is NOT enough on its own: a byte-level or byte-fallback
            # tokenizer can give several tokens the same character span, so
            # that offset can be the end of the chunk already being counted
            # and the loop would recount the same slice forever.
            shorter = min(pos + chunk_spans[max_tokens - 1][1], end - 1)
            if shorter <= pos:
                raise TokenBudgetError(
                    f"A single character {text[pos]!r} (offset {pos}) needs more than the embedding window's "
                    f"{max_tokens} usable tokens, so no chunk can hold it. Indexing is stopped rather than "
                    f"storing text the model cannot read."
                )
            end = shorter
            chunk = text[pos:end]
            chunk_spans = budget.token_spans(chunk)
        if chunk.strip():
            yield chunk, pos, end
        if end >= content_end:
            return
        overlap = budget.overlap_tokens
        next_pos = pos + chunk_spans[len(chunk_spans) - overlap][0] if 0 < overlap < len(chunk_spans) else end
        pos = next_pos if next_pos > pos else end


def _line_end_offsets(text: str) -> list:
    """Offset just past each line of text.splitlines(), newline included --
    the same line split brace_boundary_indexes/python_boundary_indexes see."""
    ends = []
    total = 0
    for line in text.splitlines(keepends=True):
        total += len(line)
        ends.append(total)
    return ends


def _preferred_cuts(text: str, suffix: str) -> list:
    """
    Character offsets where a chunk may preferably end, for chunk_stream:
    the start of every markdown heading, or the end of every line
    after which the code boundary heuristics say a cut is safe. Empty for any
    other file type, which then cuts at a newline, whitespace or the budget.
    """
    if suffix in MARKDOWN_EXTENSIONS:
        starts = [0] + _line_end_offsets(text)[:-1]
        return sorted(starts[i] for i in heading_line_indexes(text.splitlines()))
    if suffix in BRACE_BLOCK_EXTENSIONS:
        indexes = brace_boundary_indexes(text.splitlines(), ext=suffix)
    elif suffix in INDENT_BLOCK_EXTENSIONS:
        indexes = python_boundary_indexes(text.splitlines())
    else:
        return []
    ends = _line_end_offsets(text)
    return sorted(ends[i] for i in indexes)


def chunk_position(metadata) -> str:
    """
    Human-readable position of a chunk from its metadata, tolerant of both
    shapes: `line_number` + `char_range` on chunks indexed after
    token-budget chunking, `line_range` on points that predate it and keep it
    until their collection is re-indexed. A collection can also be another
    repo's, found with find_in_collection, which may not be re-indexed yet.
    Returns "" for a point carrying neither (a note stored with no position).
    """
    if not isinstance(metadata, dict):
        return ""
    line_number = metadata.get("line_number")
    char_range = metadata.get("char_range")
    if line_number is not None:
        position = f"line {line_number}"
        if char_range:
            position += f", chars {char_range}"
        return position
    line_range = metadata.get("line_range")
    if line_range:
        return f"lines {line_range}"
    return ""


def format_largest_files(chunk_counts: dict, top: int = 5) -> str:
    """
    One sentence naming the files that produced the most chunks, or "" when
    there are none. Token-budget chunking has no per-file cap by
    owner decision -- a minified bundle or a data dump simply becomes thousands
    of chunks -- so the indexing entry points name the biggest files in their
    result, which is how a run that expanded unexpectedly gets noticed.
    """
    ranked = sorted(chunk_counts.items(), key=lambda item: (-item[1], item[0]))[:top]
    if not ranked:
        return ""
    listing = ", ".join(f"{rel} ({count})" for rel, count in ranked)
    return f"Largest files by chunk count: {listing}."


def file_has_indexable_content(path: Path) -> bool:
    """
    True if chunking `path` would produce at least one chunk, i.e. it holds
    something other than whitespace after the same decoding chunk_file uses.
    Independent of any token budget, so sync_repo's index-consistency probe can
    run before the embedding model is loaded (the "no changes" path never loads
    it). Raises OSError like chunk_file.
    """
    return bool(path.read_text(encoding="utf-8", errors="ignore").strip())


def manifest_to_json(files: dict, preserved=()) -> str:
    """
    Serialize a {file: hash} map as the versioned manifest, plus the set of
    preserved-but-untracked paths (MANIFEST_PRESERVED_KEY) when there are any.
    The key is left out when the set is empty, so a manifest that has none is
    byte-for-byte what it was before the key existed.
    """
    doc = {MANIFEST_FORMAT_KEY: MANIFEST_FORMAT_VERSION, MANIFEST_FILES_KEY: files}
    if preserved:
        doc[MANIFEST_PRESERVED_KEY] = sorted(set(preserved))
    return json.dumps(doc, indent=2, sort_keys=True)


def parse_manifest(data) -> Optional[dict]:
    """
    The {file: hash} map inside a parsed manifest document, or None when
    `data` is not a manifest this code version understands: the
    old flat map, a newer format version, or anything malformed. None means
    "ignore it", which makes every file count as changed.
    """
    if not isinstance(data, dict) or data.get(MANIFEST_FORMAT_KEY) != MANIFEST_FORMAT_VERSION:
        return None
    files = data.get(MANIFEST_FILES_KEY)
    if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
        return None
    return files


def parse_manifest_preserved(data) -> set:
    """
    The set of preserved-but-untracked paths in a parsed manifest document
    (MANIFEST_PRESERVED_KEY), or an empty set when there are none: a manifest
    written before the key existed, one without it, one this code does not
    understand at all (parse_manifest says None), or a value of the wrong shape.
    Empty is the safe reading in every case: it only means those paths are not
    watched for deletion, which is what happened before the key existed.
    """
    if parse_manifest(data) is None:
        return set()
    raw = data.get(MANIFEST_PRESERVED_KEY)
    if not isinstance(raw, list):
        return set()
    return {path for path in raw if isinstance(path, str) and path}


def is_old_format_manifest(data) -> bool:
    """
    True for a parsed manifest document in the old flat {file: hash} shape
    that holds at least one entry. `claude-runway-setup upgrade`
    uses it to find a manifest worth removing; an empty one is the same as no
    manifest, and a malformed or newer-format one is not "old".
    """
    return (
        isinstance(data, dict)
        and bool(data)
        and MANIFEST_FORMAT_KEY not in data
        and all(isinstance(k, str) and isinstance(v, str) for k, v in data.items())
    )


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


def chunk_file(path: Path, rel_path: str, chunk_size: int, overlap: int, budget: Optional[TokenBudget] = None):
    """
    Chunk a single file, picking the code or doc chunker based on extension.
    Yields (content, metadata) tuples. Used both for full-repo builds and for
    re-chunking individual changed files during an incremental sync.

    With a `budget` the file is cut by the embedding model's token
    window over a stream of characters (chunk_stream), `chunk_size` and
    `overlap` are ignored (the overlap is budget.overlap_tokens), and each
    chunk's metadata carries `line_number` (1-based line of the file where the
    chunk's text begins) and `char_range` ("start-end": character offsets of the
    chunk's text in the decoded file, end exclusive, so text[start:end] is the
    stored text). Every indexing entry point passes a budget.

    Without one, the older line-window chunkers run and metadata carries
    `line_range` instead; they stay for callers that exercise the pure
    functions, and no indexing entry point may use them, because their chunks
    are sized in lines while the embedder reads tokens.
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
    if budget is not None:
        # Offsets are into `text` exactly as decoded above (UTF-8, undecodable
        # bytes dropped, newlines normalized to \n by read_text), never bytes.
        line_number = 1
        counted_to = 0
        for chunk, start, end in chunk_stream(text, budget, _preferred_cuts(text, suffix)):
            # Chunks start at strictly increasing offsets, so the line number
            # is a running count of the newlines passed since the last chunk.
            line_number += text.count("\n", counted_to, start)
            counted_to = start
            yield chunk, {
                "file_path": rel_path,
                "line_number": line_number,
                "char_range": f"{start}-{end}",
                "type": file_type,
            }
        return
    if suffix in MARKDOWN_EXTENSIONS:
        generator = chunk_markdown(text, chunk_size, overlap)
    elif suffix in BRACE_BLOCK_EXTENSIONS:
        # Pass the extension to brace_boundary_indexes so PHP's '#' line
        # comments are recognized correctly without affecting other languages
        # (see _strip_c_style_line's `ext` parameter).
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
    budget: Optional[TokenBudget] = None,
    on_file: Optional[Callable[[str, int], None]] = None,
) -> tuple:
    """
    Returns (entries, skipped): entries is a list of (content, metadata)
    tuples for every matching file across the whole repo that could actually
    be read; skipped is a list of (rel_path, error message) for any file
    that raised while being read/chunked (broken symlink, permission error,
    etc.) -- one bad file no longer aborts the whole run.
    scope is one of "code", "docs", "both" -- ignored if include_extensions
    is given. extra_exclude_dirs adds project-specific folders to skip on
    top of the built-in EXCLUDE_DIRS. extra_exclude_files skips individual
    files by name (see iter_files). respect_gitignore additionally skips
    anything the repo's own .gitignore excludes (silently a no-op if there's
    no .gitignore or the optional pathspec package isn't installed).

    budget is passed through to chunk_file. on_file, if given, is
    called as on_file(rel_path, chunk_count) after each file that was chunked,
    so a caller can show per-file progress: with no per-file chunk cap, one
    minified file can become thousands of chunks.
    """
    gitignore_spec = load_gitignore_spec(repo_path) if respect_gitignore else None
    extensions = scoped_extensions(scope, include_extensions)
    entries = []
    skipped = []
    for f in iter_files(repo_path, extensions, extra_exclude_dirs, gitignore_spec, extra_exclude_files):
        rel = str(f.relative_to(repo_path))
        try:
            before = len(entries)
            entries.extend(chunk_file(f, rel, chunk_size, overlap, budget))
            if on_file is not None:
                on_file(rel, len(entries) - before)
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
    budget: Optional[TokenBudget] = None,
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
    memory at a time, which matters for very large repos. It also
    enables progress reporting DURING the chunking phase -- the caller can
    count yielded items and report progress without waiting for the entire
    repo to be scanned first.

    All filtering parameters (scope, include_extensions, extra_exclude_dirs,
    respect_gitignore, extra_exclude_files) work identically to build_entries()
    -- keep them consistent between calls for the same repo. budget is passed
    through to chunk_file.
    """
    gitignore_spec = load_gitignore_spec(repo_path) if respect_gitignore else None
    extensions = scoped_extensions(scope, include_extensions)
    for f in iter_files(repo_path, extensions, extra_exclude_dirs, gitignore_spec, extra_exclude_files):
        rel = str(f.relative_to(repo_path))
        try:
            yield from chunk_file(f, rel, chunk_size, overlap, budget)
        except OSError as e:
            yield None, (rel, str(e))


def compute_file_hashes(
    repo_path: Path,
    scope: str,
    include_extensions: Optional[set] = None,
    extra_exclude_dirs: Optional[set] = None,
    respect_gitignore: bool = True,
    extra_exclude_files: Optional[set] = None,
    on_progress: Optional[Callable[[int], None]] = None,
    progress_every: int = HASH_PROGRESS_EVERY,
) -> tuple:
    """
    Returns (hashes, skipped). hashes maps relative file path -> sha256 hex
    digest of its current content, for every matching file that could
    actually be read -- used to detect which files changed since the last
    sync without re-embedding anything. skipped is a list of (rel_path,
    error message) for any file that raised while being hashed (broken
    symlink, permission error, etc.) -- one bad file no longer aborts the
    whole sync; callers must NOT treat a skipped file as
    "removed" just because it's absent from hashes -- see
    files_removed_since_last_sync below. Filtering params match
    build_entries -- keep them consistent between calls or the diff will
    look like every file changed. extra_exclude_files skips individual
    files by name (see iter_files).

    on_progress, if given, is called with the running number of
    files visited (hashed or skipped) after every progress_every-th one. It is
    a plain synchronous callable because this function runs in a worker
    thread: the caller decides how to get the figure to its client. There is no
    total because the directory walk is lazy, and no call for the final count,
    which the caller already has from the return value. An exception from the
    callback propagates and stops the scan: the callers' only failure is a
    cancelled call, which should stop the work, not be swallowed.
    """
    gitignore_spec = load_gitignore_spec(repo_path) if respect_gitignore else None
    extensions = scoped_extensions(scope, include_extensions)
    hashes = {}
    skipped = []
    visited = 0
    for f in iter_files(repo_path, extensions, extra_exclude_dirs, gitignore_spec, extra_exclude_files):
        rel = str(f.relative_to(repo_path))
        try:
            hashes[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError as e:
            skipped.append((rel, str(e)))
        visited += 1
        if on_progress is not None and progress_every > 0 and visited % progress_every == 0:
            on_progress(visited)
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
    over what may just be a transient read error.
    """
    skipped_set = set(skipped_paths)
    return [f for f in manifest_files if f not in current_hashes and f not in skipped_set]


def ensure_persistent_fastembed_cache() -> None:
    """
    fastembed's own define_cache_dir() (third-party, not this repo's code)
    defaults an unset FASTEMBED_CACHE_PATH to the OS temp directory
    (tempfile.gettempdir()/"fastembed_cache"), which gets wiped on reboot or
    periodic temp cleanup -- silently reintroducing a network dependency on
    huggingface.co the next time a FastEmbedProvider is constructed.
    Point it at a persistent, toolkit-owned directory instead, unless
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
    chunks instead of one client.delete() call per file
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
