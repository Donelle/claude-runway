#!/usr/bin/env python3
"""
SessionEnd hook for ClaudeRunway's opt-in savings tracker. Rolls the current
session's transient JSONL ledger (written during the session by
compress_output.py -- see libs/savings_ledger.py) into one row of the
perpetual SQLite store, then surfaces a short summary to the user via
`systemMessage` in this hook's JSON output.

Verified against Claude Code v2.1.218: SessionEnd supports `systemMessage`
(not `additionalContext`, which SessionEnd does NOT support) as the way to
show text to the user, and receives session_id/transcript_path/cwd/reason on
stdin. See libs/savings_ledger.py's module docstring for why this is an
ONLINE ESTIMATE of a counterfactual, not a measured A/B delta (EVALUATION.md
Track D covers that distinction).

No-op (silent, zero output) when CLAUDE_RUNWAY_TRACK_SAVINGS is off, or when
this session logged no credited compression events -- an empty summary isn't
worth printing every time a session ends. Either way (tracking on), it first
silently rolls up any OTHER session's ledger orphaned by a crash (issue #306,
see savings_ledger.recover_orphaned_sessions) -- that never prints anything.

Setup: register in .claude/settings.json (see settings.json.template):
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "REPLACE-WITH-VENV-PYTHON",
            "args": ["/absolute/path/to/tools-repo/hooks/session_end_savings.py"]
          }
        ]
      }
    ]
  }
}
"""

import json
import os
import sys

# Same redundant path-resolution order as compress_output.py, so this
# script also works if copied standalone or if the repo layout shifts --
# see that file's comment for the full rationale.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_candidates = [os.environ.get("TOOLS_REPO_DIR"), os.path.join(_root, "libs"), _root, _here]
for _dir in _candidates:
    if _dir:
        sys.path.insert(0, _dir)

try:
    import savings_ledger
except ImportError:
    sys.exit(0)  # savings tracker isn't installed -- silent no-op, nothing to report


def _parse_transcript_enabled() -> bool:
    return os.environ.get("CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS", "").strip().lower() in ("1", "true", "yes")


def main():
    # Force UTF-8 on stdin before the read: Claude Code sends the payload as
    # UTF-8, but Windows Python defaults stdin to the locale codepage
    # (cp1252), under which a cp1252-unmappable byte raises UnicodeDecodeError
    # -- a ValueError the guard below swallows into a silent fail-open (issue
    # #263). Guarded because a non-reconfigurable stream (e.g. io.StringIO in
    # the unit tests) has no reconfigure(); fail open rather than crash.
    _reconfigure = getattr(sys.stdin, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass  # already-consumed/detached stream -- fail open
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    if not savings_ledger.tracking_enabled():
        sys.exit(0)

    session_id = payload.get("session_id")
    cwd = payload.get("cwd", "")
    transcript_path = payload.get("transcript_path")
    if not session_id:
        sys.exit(0)

    # Issue #306: roll up any OTHER session's ledger that was orphaned by a
    # crash/force-kill (it never got a SessionEnd of its own, so this hook is
    # the only chance it has of reaching savings.db). Runs before the
    # "nothing logged this session" exit below on purpose -- recovery mustn't
    # depend on whether the session that happens to be ending compressed
    # anything. This session is excluded: it's finalized by the normal path
    # below (which also picks up any leftover claim of it). Separately guarded so a recovery failure can never cost this
    # session its own roll-up or summary.
    try:
        savings_ledger.recover_orphaned_sessions(exclude_session_id=session_id)
    except Exception:
        pass

    # Any artifact, not just the ledger (#306): a roll-up of this session that
    # failed or died can have left only a claim of it, which recovery skips
    # for the session that is ending -- so this hook must still take it.
    # Guarded like the rest of this hook: any error here fails open (exit 0,
    # no output) and leaves the files for a later roll-up, never a traceback
    # at session shutdown.
    try:
        has_artifacts = savings_ledger.has_session_artifacts(session_id)
    except Exception:
        sys.exit(0)
    if not has_artifacts:
        sys.exit(0)  # nothing logged this session -- don't print an empty summary

    # Parse actual token counts from the transcript if the opt-in env var is set.
    # parse_session_token_counts also folds in this session's subagent
    # transcripts (#364) -- transcript_path alone is only the MAIN session's file.
    # Fails open -- any parse failure leaves actual_tokens as None. None means
    # "no new counts": finalize_owned_session writes 0s for a session with no
    # row yet, and keeps the counts an existing row already has (a resumed
    # session's earlier roll-up) rather than zeroing them.
    # STOPGAP: remove when #164 is resolved (Stop hook will expose these directly).
    actual_tokens = None
    if _parse_transcript_enabled() and transcript_path:
        actual_tokens = savings_ledger.parse_session_token_counts(transcript_path)

    # This hook is meant to be best-effort and fail open, same philosophy as
    # compress_output.py -- a savings-tracker problem (a corrupted DB
    # file, a disk error, a locked SQLite connection) must never surface as a
    # crashed hook or a missing systemMessage at normal session shutdown.
    try:
        project = savings_ledger.project_name_from_cwd(cwd)
        overhead = savings_ledger.get_schema_overhead_tokens()
        # finalize_owned_session, not finalize_session (#306): the same
        # ownership protocol orphan recovery uses -- serialized on savings.db's
        # write lock, and rolling up this session's ledger TOGETHER with any
        # claim of it a failed/dead recovery left behind, since rolling them
        # up separately would let one overwrite the other. None means another
        # process already consumed this session's events; that roll-up stands.
        session_agg = savings_ledger.finalize_owned_session(
            session_id, project, overhead_tokens=overhead, actual_tokens=actual_tokens,
        )
        if not session_agg:
            sys.exit(0)
        session_agg["project"] = project
        # #307: events existing isn't enough to print -- `event_count` counts
        # CREDITED events only (see _write_session_rows), so a fetch_url-only
        # session (logged but never credited) rolls up into the perpetual
        # store above but stays silent here instead of printing a "0 tokens
        # avoided" summary, matching the documented no-op behavior.
        if not session_agg.get("event_count"):
            sys.exit(0)
        project_summary = savings_ledger.query_project_summary(project)
        summary = savings_ledger.format_simple_view(session_agg, project_summary)
    except Exception:
        sys.exit(0)

    print(json.dumps({"systemMessage": summary}))
    sys.exit(0)


if __name__ == "__main__":
    main()
