#!/usr/bin/env python3
"""
SessionEnd hook for ClaudeRunway's opt-in savings tracker. Rolls the current
session's transient JSONL ledger (written during the session by
compress_bash_output.py -- see libs/savings_ledger.py) into one row of the
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
worth printing every time a session ends.

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

# Same redundant path-resolution order as compress_bash_output.py, so this
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

    events = savings_ledger.read_session_events(session_id)
    if not events:
        sys.exit(0)  # nothing logged this session -- don't print an empty summary

    # Parse actual token counts from the transcript if the opt-in env var is set.
    # Fails open -- any parse failure leaves actual_tokens as None, and
    # finalize_session stores 0s for the actual_* columns rather than erroring.
    # STOPGAP: remove when #164 is resolved (Stop hook will expose these directly).
    actual_tokens = None
    if _parse_transcript_enabled() and transcript_path:
        actual_tokens = savings_ledger.parse_transcript_token_counts(transcript_path)

    # This hook is meant to be best-effort and fail open, same philosophy as
    # compress_bash_output.py -- a savings-tracker problem (a corrupted DB
    # file, a disk error, a locked SQLite connection) must never surface as a
    # crashed hook or a missing systemMessage at normal session shutdown.
    try:
        project = savings_ledger.project_name_from_cwd(cwd)
        overhead = savings_ledger.get_schema_overhead_tokens()
        session_agg = savings_ledger.finalize_session(
            session_id, project, overhead_tokens=overhead, actual_tokens=actual_tokens
        )
        session_agg["project"] = project
        project_summary = savings_ledger.query_project_summary(project)
        summary = savings_ledger.format_simple_view(session_agg, project_summary)
    except Exception:
        sys.exit(0)

    print(json.dumps({"systemMessage": summary}))
    sys.exit(0)


if __name__ == "__main__":
    main()
