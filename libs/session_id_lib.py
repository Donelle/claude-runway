"""
Shared module classifying every way this toolkit's MCP-server/hook code can
get or approximate a session_id -- including all the path/directory/slug
logic behind each strategy -- behind a single entry point, so a future
caller picks the right strategy deliberately instead of reinventing one,
copying the wrong one, or reaching into another module's private directory
layout (issue #198).

Four strategies exist:

| Type | Strategy        | Mechanism                                                                                                  | Needs a hook? | Real ID? | Risk |
|------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|----------|------|
| 1    | HOOK_PAYLOAD    | Claude Code hands every hook `session_id`/`transcript_path` directly on stdin -- trivial extraction.        | Yes (hook only) | Yes -- authoritative | None |
| 2    | SHADOW_FILE     | `hooks/record_session_id.py` writes a small presence marker (`session_<id>.jsonl`) into the sessions directory `savings_ledger.py` also uses (own filename prefix, no shared format); an MCP tool call scans for the most-recently-modified marker, optionally filtered by project. | Yes (recoverable elsewhere) | Yes, recovered from what the hook wrote | Only available where the hook ran |
| 3    | TRANSCRIPT_SCAN | Claude Code itself continuously writes `~/.claude/projects/<slug>/<session-id>.jsonl` for every session, hook or no hook. `<slug>` is the project's absolute path with every `/` replaced by `-`. Picks the most-recently-modified file's stem. | No | Yes -- zero hook dependency | Depends on an undocumented, empirically-observed internal convention (macOS-verified only) |
| 4    | PROXY           | One `uuid.uuid4().hex` generated once per process and cached, reused for every event that process logs.     | No | No -- proxy, spans the whole process | None -- pure in-memory |

Decision rule:
  - You ARE a hook -> Type 1 (HOOK_PAYLOAD). Always correct, always available.
  - You're MCP-server code (no hook access), need the real boundary, and a
    hook already feeds a shadow file for this domain -> Type 2 (SHADOW_FILE).
  - You're MCP-server code, need the real boundary, no hook available, and
    willing to accept dependency on Claude Code's undocumented
    project-directory layout -> Type 3 (TRANSCRIPT_SCAN).
  - You're MCP-server code, a coarse per-process grouping key is good
    enough, and you want zero dependency on anything external -> Type 4
    (PROXY).

`project`'s contract is the SAME value for every strategy that uses it -- an
absolute path, never a bare name -- because the two callers need different
derivations from it: TRANSCRIPT_SCAN uses the path as-is to compute
`<slug>`; SHADOW_FILE internally reduces it to `Path(project).name` to
filter marker files by project. Callers never pass a bare project name
themselves; passing anything other than an absolute path (or omitting it,
which falls back to `os.getcwd()`) is a caller error.

This module owns ALL the underlying logic internally (the shadow-sessions
directory location and filename convention, the `<slug>` derivation for
transcript scanning, and the proxy's module-level cache) -- callers only
ever pick a `SessionIdStrategy` and, where relevant, pass the hook's own
payload dict or a `project` filter.

Deliberately independent of `libs/savings_ledger.py` -- no import, even
though SHADOW_FILE uses the SAME sessions directory that module's own
`_sessions_dir()` resolves to. This is a directory-CONVENTION match, not a
code dependency: this module's own `session_*.jsonl` marker files never
share a filename with, read, or write `savings_ledger`'s bare
`<session_id>.jsonl` event logs, and wiring the two together is explicitly
tracked separately (issue #213) rather than done here.

Cleanup (two layers, both driven by `hooks/record_session_id.py`):
  - Layer 1 (graceful): a `SessionEnd` registration of the same hook script
    deletes its own session's marker on normal exit (see
    `delete_shadow_marker`).
  - Layer 2 (crash-safety net, write side): the `PostToolUse` registration
    opportunistically sweeps markers older than
    `CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS` (default 48) -- see
    `sweep_stale_shadow_markers`/`should_sweep`/`mark_swept`. Bounded to run
    at most once per hour (a sentinel file's mtime, not a counter) so sweep
    cost doesn't scale with total sessions-directory size on every single
    tool call. The hook always refreshes its OWN marker before sweeping
    (see that script), so a live session can never observe its own marker
    as stale and prune itself, regardless of how long it's been since that
    session's last tool call relative to the TTL.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
from enum import IntEnum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Strategy enum + unified entry point
# ---------------------------------------------------------------------------


class SessionIdStrategy(IntEnum):
    HOOK_PAYLOAD = 1
    SHADOW_FILE = 2
    TRANSCRIPT_SCAN = 3
    PROXY = 4


def session_id(
    strategy: SessionIdStrategy,
    *,
    hook_payload: Optional[dict] = None,
    project: Optional[str] = None,
) -> Optional[str]:
    """Single entry point -- pick a strategy, get back a session_id (or an
    approximation of one), or None if that strategy couldn't resolve one
    right now. For any RECOGNIZED strategy, never raises: every strategy
    below fails toward None on any I/O problem (missing directory,
    unreadable file, malformed marker) rather than propagating, since this
    is meant to be safe to call from deep inside MCP-server/hook logic
    where an exception would break normal operation over what's ultimately
    a best-effort lookup. An UNRECOGNIZED strategy (not one of the
    `SessionIdStrategy` members) is a caller bug, not a best-effort lookup
    that can fail open -- that case deliberately raises `ValueError` below
    instead of silently returning `None`, so a typo'd/invalid strategy is
    loud rather than indistinguishable from "couldn't resolve one"."""
    if strategy == SessionIdStrategy.HOOK_PAYLOAD:
        return _session_id_from_hook_payload(hook_payload)
    if strategy == SessionIdStrategy.SHADOW_FILE:
        return _session_id_from_shadow_file(project)
    if strategy == SessionIdStrategy.TRANSCRIPT_SCAN:
        return _session_id_from_transcript_scan(project)
    if strategy == SessionIdStrategy.PROXY:
        return _proxy_session_id()
    raise ValueError(f"Unknown SessionIdStrategy: {strategy!r}")


def _resolve_project(project: Optional[str]) -> Optional[str]:
    """
    Shared `project` fallback for SHADOW_FILE and TRANSCRIPT_SCAN (the two
    strategies whose contract says omitting `project` falls back to
    `os.getcwd()`): returns `project` unchanged if given AND absolute,
    otherwise tries `os.getcwd()` -- which can itself raise `OSError` (e.g.
    this process's current working directory was deleted/renamed out from
    under it, confirmed a real possibility, not a hypothetical). That would
    otherwise propagate straight out of a call this module documents as
    never raising for a recognized strategy, so it's caught here, once,
    rather than duplicated in each strategy that needs the same fallback.
    Returns None (not the failing call's exception, and not a value that
    violates the contract) when even the fallback can't be resolved --
    callers treat that exactly like any other unresolvable case.

    This module's own `project` contract (see module docstring) says the
    value is always an absolute path, never a bare/relative one -- a
    caller passing something like `"app"` is a caller error, but the
    contract wasn't actually ENFORCED here: a relative value used to pass
    through unchanged, silently corrupting SHADOW_FILE's basename filter
    (matching an unrelated project that happens to share that relative
    string) and TRANSCRIPT_SCAN's `<slug>` derivation. Rejecting it (fail
    toward None, same as an unresolvable `os.getcwd()`) instead of
    accepting it unchanged closes that gap.
    """
    if project is not None:
        return project if os.path.isabs(project) else None
    try:
        return os.getcwd()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Type 1: HOOK_PAYLOAD
# ---------------------------------------------------------------------------


def _session_id_from_hook_payload(hook_payload: Optional[dict]) -> Optional[str]:
    if not hook_payload:
        return None
    value = hook_payload.get("session_id")
    return value if isinstance(value, str) and value else None


# ---------------------------------------------------------------------------
# Type 2: SHADOW_FILE -- shared, always-populated marker directory
# ---------------------------------------------------------------------------

# Same directory savings_ledger.py resolves as resolve_db_path().parent /
# "sessions" for its own DEFAULT db location -- kept as an independent
# literal here (not imported) so this module never depends on
# savings_ledger's own CLAUDE_RUNWAY_SAVINGS_DB override or db-path logic.
# See module docstring: same directory CONVENTION, not a code dependency.
_MARKER_FILENAME_PREFIX = "session_"


def _sessions_dir() -> Path:
    return Path.home() / ".claude" / "claude-runway" / "sessions"


def _sanitize_session_id(raw_session_id: str) -> str:
    """
    session_id ends up directly in a filename below -- if it ever contained
    a path separator or ".." (a malformed hook payload, a manual/adversarial
    invocation), the resulting path could escape _sessions_dir() entirely.
    Replacing anything that isn't a-z/A-Z/0-9/hyphen/underscore keeps this
    confined regardless of what session_id turns out to be, without
    changing behavior for the UUID-like IDs Claude Code actually sends. Same
    approach as savings_ledger._sanitize_session_id -- duplicated rather
    than imported, per this module's deliberate independence from that one.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw_session_id)


def _shadow_marker_path(raw_session_id: str) -> Path:
    return _sessions_dir() / f"{_MARKER_FILENAME_PREFIX}{_sanitize_session_id(raw_session_id)}.jsonl"


def _read_shadow_marker_project(path: Path) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as f:
            line = f.readline()
        data = json.loads(line)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        # UnicodeDecodeError (invalid UTF-8 bytes) is a ValueError subclass,
        # NOT an OSError -- readline() raises it directly rather than
        # returning a value, so it needs its own arm here. A marker with
        # invalid encoding is just as "malformed" as one with invalid JSON;
        # letting it escape this function would break a recognized
        # strategy's fail-toward-None contract for its MCP caller.
        return None
    if not isinstance(data, dict):
        # A marker can be syntactically valid JSON (e.g. "null", "[1]", a
        # bare number/string) without being the object this module writes --
        # a corrupted/truncated write, or something else entirely landing at
        # this path. That still counts as "malformed marker" under this
        # module's fail-toward-None contract, so guard against it explicitly
        # rather than letting dict-only .get() below raise AttributeError.
        return None
    project = data.get("project")
    return project if isinstance(project, str) else None


def record_shadow_marker(raw_session_id: str, project: str) -> None:
    """
    Writes/OVERWRITES (never appends) this session's Type-2 shadow marker
    with just `{"project": "<absolute project path>"}` -- called by
    `hooks/record_session_id.py` on (broad-matcher) PostToolUse to keep this
    session's marker fresh. Every marker always carries a real `project` tag
    regardless of what else is configured, so project-filtered lookups work
    for every project, not only ones that also have `savings_ledger` writing
    richer content.

    Published atomically (write to a same-directory temp file, then
    `os.replace()` over the real path) rather than truncate-in-place
    (`open(path, "w")`) -- `open(..., "w")` truncates the file to zero
    bytes synchronously, before any content is written, so a concurrent
    SHADOW_FILE reader could open this exact path in that window and see
    empty/partial content, get treated as a malformed marker, and
    incorrectly fall back to an older marker (or None) instead of this
    fresh one. `os.replace()` is atomic on the same filesystem (POSIX
    rename(2) semantics; the same guarantee holds on Windows), so a reader
    only ever sees the complete old file or the complete new one, never
    something in between.
    """
    path = _shadow_marker_path(raw_session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"project": project}) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def delete_shadow_marker(raw_session_id: str) -> None:
    """Layer 1 graceful cleanup -- unlinks this session's own marker on
    normal SessionEnd. Silent no-op if it's already gone."""
    _shadow_marker_path(raw_session_id).unlink(missing_ok=True)


def _strip_marker_prefix(stem: str) -> str:
    if stem.startswith(_MARKER_FILENAME_PREFIX):
        return stem[len(_MARKER_FILENAME_PREFIX):]
    return stem


def _session_id_from_shadow_file(project: Optional[str]) -> Optional[str]:
    """
    Scans the sessions directory for the most-recently-modified
    `session_*.jsonl` marker, filtered by project (reduced to
    `Path(project).name`, matching how `hooks/record_session_id.py` derives
    the tag it writes into each marker). Per this module's `project`
    contract (issue #198), omitting `project` falls back to `os.getcwd()`
    the same way every other strategy that takes a `project` does -- it
    does NOT disable filtering. Returns None if the directory doesn't
    exist, is empty, or no marker matches the (given-or-cwd-derived)
    project -- never substitutes a DIFFERENTLY-NAMED project's session id,
    the same caution savings_ledger.current_session_id() applies for the
    identical reason.

    Known limitation (project-scoping is by directory BASENAME, not full
    path): two distinct projects that happen to share the same final path
    component (e.g. `/Users/alice/app` and `/Users/bob/other/app`) are
    indistinguishable to this filter and can match each other's marker --
    confirmed live. This mirrors the SAME basename-only derivation the
    ticket's `project` contract specifies for this strategy (issue #198),
    so it isn't unique to this function; if this ever needs tightening,
    do it by having `record_shadow_marker`/this function compare full
    paths instead, not by reinterpreting `project` differently per
    strategy.
    """
    d = _sessions_dir()
    if not d.is_dir():
        return None
    dated = []
    for p in d.glob(f"{_MARKER_FILENAME_PREFIX}*.jsonl"):
        try:
            dated.append((p.stat().st_mtime, p))
        except OSError:
            continue  # a concurrent SessionEnd cleanup can delete a marker mid-scan
    if not dated:
        return None
    dated.sort(key=lambda pair: pair[0], reverse=True)

    effective_project = _resolve_project(project)
    if effective_project is None:
        # project was omitted AND os.getcwd() itself failed (e.g. this
        # process's cwd was deleted/renamed out from under it) -- nothing
        # left to filter by, so fail toward None like every other
        # unresolvable case rather than letting OSError propagate.
        return None
    project_filter = Path(effective_project).name
    for _, p in dated:
        marker_project = _read_shadow_marker_project(p)
        if marker_project is not None and Path(marker_project).name == project_filter:
            return _strip_marker_prefix(p.stem)
    return None


# --- Layer 2: crash-safety sweep, rate-limited via a sentinel file ----------

_DEFAULT_SESSION_MARKER_TTL_HOURS = 48.0
_SWEEP_INTERVAL_SECONDS = 3600  # bound sweep cost to roughly once per hour per project
_SENTINEL_FILENAME = ".last_sweep"


def _ttl_hours() -> float:
    """
    Parses `CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS`, falling back to the
    default for anything unset/blank/unparseable -- AND for any parsed
    value that isn't strictly positive or is NaN. `float()` happily parses
    "0", "-1", and "nan" without raising, but a non-positive TTL makes the
    very next sweep treat even the marker the CURRENT hook invocation just
    refreshed as stale (confirmed live: `(now - mtime) > 0` is true for any
    mtime strictly in the past, which self-refresh always is by the time
    the comparison runs), defeating the "self-refresh-before-sweep" safety
    guarantee the whole two-layer cleanup design depends on. NaN compares
    False against everything, which merely disables sweeping silently
    instead -- not unsafe, but not what a NaN-typo'd value was going for
    either, so it's rejected the same way.
    """
    raw = os.environ.get("CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS")
    if raw is None or not raw.strip():
        return _DEFAULT_SESSION_MARKER_TTL_HOURS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SESSION_MARKER_TTL_HOURS
    if not (value > 0):  # also rejects NaN, since every NaN comparison is False
        return _DEFAULT_SESSION_MARKER_TTL_HOURS
    return value


def _sentinel_path() -> Path:
    return _sessions_dir() / _SENTINEL_FILENAME


def should_sweep(now: Optional[float] = None) -> bool:
    """True if more than `_SWEEP_INTERVAL_SECONDS` has elapsed since the last
    sweep (or none has ever run) -- the hook checks this before bothering to
    list+stat every marker in the directory, so the cost of sweeping scales
    with elapsed time, not with how many tool calls happen in between."""
    now = time.time() if now is None else now
    try:
        return (now - _sentinel_path().stat().st_mtime) > _SWEEP_INTERVAL_SECONDS
    except OSError:
        return True  # no sentinel yet -- sweep once to create it


def mark_swept(now: Optional[float] = None) -> None:
    p = _sentinel_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    if now is not None:
        os.utime(p, (now, now))


def sweep_stale_shadow_markers(now: Optional[float] = None) -> int:
    """
    Deletes every `session_*.jsonl` marker older than the TTL
    (`CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS`, default 48h). Only ever
    considers this module's own `session_*.jsonl` markers -- never touches
    a bare `<session_id>.jsonl` file, since those belong to
    `savings_ledger`'s own event log (present in the same directory for
    tracking-enabled sessions) and are cleaned up separately by that
    module's own `finalize_session()` at SessionEnd.

    Safe by construction ONLY if the caller refreshes its own marker first
    (see `hooks/record_session_id.py`): a live session's own next tool call
    refreshes its file's mtime, so anything old enough to cross the TTL can
    only be a dead session's leftover marker.

    Known limitation (cross-PROCESS TOCTOU, narrowed but not eliminated):
    the staleness check and the unlink below are two separate syscalls, so
    a DIFFERENT session's own hook process can call `record_shadow_marker`
    (self-refresh) in the gap between them -- `os.replace()` there swaps in
    a fresh file at the same path, but `unlink()` here operates on the PATH,
    not the specific (now-stale) inode this sweep decided to delete, so it
    would delete that fresh file anyway. A re-check immediately before
    unlinking (below) shrinks this window from "the time between stat and
    unlink for every marker in the directory" down to "the time between two
    back-to-back stat calls on the same marker" -- narrow enough to be
    exceedingly unlikely in practice, but not a hard guarantee the way a
    real interprocess lock (e.g. `flock`/`fcntl`) would be. Deliberately not
    adding one: this module's whole design intentionally avoids
    cross-process coordination for this marker mechanism (see issue #198's
    "no schema/migration concerns... trivial to reason about" rationale for
    staying file-based over e.g. SQLite) -- the residual race here is the
    same category of trade-off, and the worst case is self-healing anyway
    (the affected session's own next tool call recreates its marker on the
    very next `PostToolUse` hook invocation).

    Returns the number of markers deleted (mainly useful for tests).
    """
    now = time.time() if now is None else now
    ttl_seconds = _ttl_hours() * 3600
    d = _sessions_dir()
    if not d.is_dir():
        return 0
    deleted = 0
    for p in d.glob(f"{_MARKER_FILENAME_PREFIX}*.jsonl"):
        try:
            mtime = p.stat().st_mtime
            if (now - mtime) <= ttl_seconds:
                continue
            if p.stat().st_mtime != mtime:
                continue  # refreshed by another process between the two stat() calls above
            p.unlink()
            deleted += 1
        except OSError:
            continue  # already gone, or a transient race -- not fatal to the sweep
    return deleted


# ---------------------------------------------------------------------------
# Type 3: TRANSCRIPT_SCAN -- Claude Code's own per-project transcript dir
# ---------------------------------------------------------------------------


def _transcript_projects_dir() -> Path:
    return Path.home() / ".claude" / "projects"


def _project_slug(project: Optional[str]) -> Optional[str]:
    """
    Claude Code's own (undocumented, empirically observed) convention:
    `<slug>` is the project's absolute path with every `/` replaced by `-`.
    Verified directly against this machine's own `~/.claude/projects/`
    (macOS). NOT verified on Windows -- the `/`-separator convention (and
    how a Windows-style path would map to it, if at all) is unconfirmed
    there; don't depend on this for a Windows-installed project without
    checking first.

    Returns None (rather than raising) if `project` is omitted and
    `os.getcwd()` itself fails -- see `_resolve_project`.
    """
    p = _resolve_project(project)
    if p is None:
        return None
    return str(p).replace("/", "-")


def _session_id_from_transcript_scan(project: Optional[str]) -> Optional[str]:
    """
    Picks the most-recently-modified `.jsonl` directly under
    `~/.claude/projects/<slug>/` and returns its filename stem (the real
    session_id) -- or None if that directory doesn't exist, is empty, or
    nothing in it can be stat()'d. Zero hook dependency, but depends on
    Claude Code's own undocumented internal file layout -- see this
    module's docstring / `_project_slug`'s docstring for the known caveats.
    """
    slug = _project_slug(project)
    if slug is None:
        return None
    d = _transcript_projects_dir() / slug
    if not d.is_dir():
        return None
    dated = []
    for p in d.glob("*.jsonl"):
        try:
            dated.append((p.stat().st_mtime, p))
        except OSError:
            continue
    if not dated:
        return None
    dated.sort(key=lambda pair: pair[0], reverse=True)
    return dated[0][1].stem


# ---------------------------------------------------------------------------
# Type 4: PROXY -- process-lifetime cached UUID
# ---------------------------------------------------------------------------

_process_proxy_id: Optional[str] = None
_process_proxy_id_lock = threading.Lock()


def _proxy_session_id() -> str:
    """
    Every call within the same process must return the IDENTICAL cached
    string (that's what makes this usable as a grouping key) -- a bare
    `if _process_proxy_id is None: ...` check-then-set is not atomic, so
    two concurrent first callers (e.g. an MCP server handling overlapping
    tool calls on separate threads) could both observe `None` and each
    mint their own UUID, silently violating that guarantee (confirmed live
    with a threaded repro). Double-checked locking closes that window
    without paying lock overhead on the (overwhelmingly common) already-
    cached path.
    """
    global _process_proxy_id
    if _process_proxy_id is None:
        with _process_proxy_id_lock:
            if _process_proxy_id is None:
                _process_proxy_id = uuid.uuid4().hex
    return _process_proxy_id
