#!/usr/bin/env python3
"""
Keeps `libs/session_id_lib.py`'s Type-2 (SHADOW_FILE) shadow marker fresh
(issue #198) -- the one hook whose entire job is making SHADOW_FILE an
actually-populated, permanent strategy, rather than inert scanning logic
pointed at a directory nothing writes to. Before this hook existed, the
sessions directory was only ever populated as a side effect of the opt-in
savings tracker (`hooks/compress_bash_output.py`, itself gated behind
`CLAUDE_RUNWAY_TRACK_SAVINGS`) -- this hook is part of the CORE/base
install, so every project gets a working SHADOW_FILE strategy regardless of
whether local-compress/the savings tracker is configured at all.

Registered under TWO separate events in `templates/settings.json.template`
(see `libs/setup_project_lib.py`'s `_CORE_HOOK_SCRIPTS`), both pointing at
this SAME script:

  - `PostToolUse` (broad matcher `.*` -- fires often enough that the marker's
    mtime tracks real recent activity, not just session start): refreshes
    this session's OWN marker first, then -- rate-limited via a sentinel
    file, roughly once per hour -- sweeps any marker older than
    `CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS` (default 48). Refreshing self
    BEFORE sweeping guarantees a live session can never observe its own
    marker as stale and prune itself, regardless of how long it's been
    since this session's last tool call relative to the TTL.
  - `SessionEnd`: Layer 1 graceful cleanup -- deletes this session's own
    marker on normal exit. A sibling delete to, but independent of,
    `savings_ledger.finalize_session()`'s own unlink of its differently
    named `<session_id>.jsonl` file.

The two payload shapes are told apart via `hook_event_name`, which Claude
Code includes on every hook invocation's stdin payload; PostToolUse-shaped
payloads (which additionally carry `tool_name`/`tool_response`) are the
fallback if that field is ever absent, so this never depends on exactly one
signal to tell the events apart.

Writes NOTHING to stdout on the ordinary path (no `updatedToolOutput`, no
`additionalContext`) -- this hook never touches tool output, only a side
file. Fails open, unconditionally and silently: a broken sessions directory
(unwritable, disk full, permissions) must never surface as a broken tool
call or a broken session shutdown, since this hook's own job is a pure
side-effect nobody's else's correctness depends on synchronously.

Setup: see templates/settings.json.template's core PostToolUse/SessionEnd
blocks -- nothing needs to be copied into a target project, same convention
as every other hook in this repo.

Env vars: `CLAUDE_RUNWAY_SESSION_MARKER_TTL_HOURS` (shell-only, default 48;
see libs/session_id_lib.py's `_ttl_hours()`).
"""

import json
import os
import sys

# Same redundant path-resolution order as compress_bash_output.py/
# session_end_savings.py, so this script also works if copied standalone or
# if the repo layout shifts -- see compress_bash_output.py's comment for the
# full rationale.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_candidates = [os.environ.get("TOOLS_REPO_DIR"), os.path.join(_root, "libs"), _root, _here]
for _dir in _candidates:
    if _dir:
        sys.path.insert(0, _dir)

try:
    import session_id_lib
except ImportError:
    # Fail open, silently -- this hook has no output contract to preserve
    # (unlike compress_bash_output.py, which must report why on stderr since
    # a broken import there silently drops real compression); a missing
    # libs/session_id_lib.py just means Type 2 isn't populated this run.
    sys.exit(0)


def _handle_post_tool_use(payload: dict) -> None:
    raw_session_id = payload.get("session_id")
    if not raw_session_id:
        return
    cwd = payload.get("cwd") or ""
    if cwd:
        # Refresh THIS session's own marker first, unconditionally, before
        # any sweep runs below -- see module docstring for why this
        # ordering is what keeps a live session from ever pruning itself.
        session_id_lib.record_shadow_marker(raw_session_id, project=cwd)
    if session_id_lib.should_sweep():
        session_id_lib.sweep_stale_shadow_markers()
        session_id_lib.mark_swept()


def _handle_session_end(payload: dict) -> None:
    raw_session_id = payload.get("session_id")
    if not raw_session_id:
        return
    session_id_lib.delete_shadow_marker(raw_session_id)


def _dispatch(payload: dict) -> None:
    event = payload.get("hook_event_name")
    # Fallback discriminator for a payload shape without hook_event_name:
    # PostToolUse payloads always carry tool_name, SessionEnd payloads never do.
    is_session_end = event == "SessionEnd" or (event is None and "tool_name" not in payload)
    if is_session_end:
        _handle_session_end(payload)
    else:
        _handle_post_tool_use(payload)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)  # can't parse input -- fail open, nothing to do

    try:
        _dispatch(payload)
    except Exception:
        # Unconditional fail-open (issue #198, same philosophy as
        # compress_bash_output.py's top-level guard) -- this hook's job is a
        # pure side effect; a bug here must never surface as a broken tool
        # call or a broken session shutdown.
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
