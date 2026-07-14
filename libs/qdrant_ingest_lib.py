"""
Shared, side-effect-free chunking helpers used by both:
  - ingest_to_qdrant.py   (standalone CLI script)
  - ingest_mcp_server.py  (MCP server exposing the same logic as tools)

Kept dependency-free (no qdrant-client / mcp_server_qdrant imports here) so
it's easy to unit test and reuse. No print() calls in this file on purpose --
ingest_mcp_server.py runs over stdio, where stray prints corrupt the protocol.
"""

import hashlib
from pathlib import Path
from typing import Optional

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rb", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".swift", ".kt", ".scala",
    ".sql", ".sh", ".yaml", ".yml", ".json",
}
DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}

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
    lines = gitignore_path.read_text(errors="ignore").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def iter_files(
    repo_path: Path,
    extensions: set,
    extra_exclude_dirs: Optional[set] = None,
    gitignore_spec=None,
):
    """
    extra_exclude_dirs adds to (doesn't replace) the built-in EXCLUDE_DIRS.
    gitignore_spec, if provided (see load_gitignore_spec), additionally skips
    anything the repo's own .gitignore would exclude.
    """
    exclude_dirs = EXCLUDE_DIRS | (extra_exclude_dirs or set())
    for path in repo_path.rglob("*"):
        if path.is_dir():
            continue
        if any(part in exclude_dirs for part in path.parts):
            continue
        rel = path.relative_to(repo_path)
        if gitignore_spec is not None and gitignore_spec.match_file(str(rel)):
            continue
        if path.suffix.lower() in extensions:
            yield path


def chunk_lines(text: str, chunk_size: int, overlap: int):
    """Fixed-size line-based chunking with overlap. Language-agnostic."""
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


def chunk_markdown(text: str, fallback_chunk_size: int, fallback_overlap: int):
    """Split markdown on headings; falls back to line chunking if no headings."""
    lines = text.splitlines()
    heading_indexes = [i for i, l in enumerate(lines) if l.lstrip().startswith("#")]
    if len(heading_indexes) < 2:
        yield from chunk_lines(text, fallback_chunk_size, fallback_overlap)
        return
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
    text = path.read_text(errors="ignore")
    if path.suffix.lower() in DOC_EXTENSIONS:
        generator = chunk_markdown(text, chunk_size, overlap)
        file_type = "doc"
    else:
        generator = chunk_lines(text, chunk_size, overlap)
        file_type = "code"
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
):
    """
    Returns a list of (content, metadata) tuples for every matching file
    across the whole repo. scope is one of "code", "docs", "both" -- ignored
    if include_extensions is given. extra_exclude_dirs adds project-specific
    folders to skip on top of the built-in EXCLUDE_DIRS. respect_gitignore
    additionally skips anything the repo's own .gitignore excludes (silently
    a no-op if there's no .gitignore or the optional pathspec package isn't
    installed).
    """
    gitignore_spec = load_gitignore_spec(repo_path) if respect_gitignore else None
    extensions = scoped_extensions(scope, include_extensions)
    entries = []
    for f in iter_files(repo_path, extensions, extra_exclude_dirs, gitignore_spec):
        rel = str(f.relative_to(repo_path))
        entries.extend(chunk_file(f, rel, chunk_size, overlap))
    return entries


def compute_file_hashes(
    repo_path: Path,
    scope: str,
    include_extensions: Optional[set] = None,
    extra_exclude_dirs: Optional[set] = None,
    respect_gitignore: bool = True,
) -> dict:
    """
    Maps relative file path -> sha256 hex digest of its current content, for
    every matching file. Used to detect which files changed since the last
    sync without re-embedding anything. Filtering params match build_entries
    -- keep them consistent between calls or the diff will look like every
    file changed.
    """
    gitignore_spec = load_gitignore_spec(repo_path) if respect_gitignore else None
    extensions = scoped_extensions(scope, include_extensions)
    hashes = {}
    for f in iter_files(repo_path, extensions, extra_exclude_dirs, gitignore_spec):
        rel = str(f.relative_to(repo_path))
        hashes[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    return hashes
