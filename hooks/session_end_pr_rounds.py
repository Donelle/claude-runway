#!/usr/bin/env python3
"""
SessionEnd hook that records the pull-request review rounds of the session that is ending to the
shared metrics store (`metric_id` `pr-review-rounds`), so the baseline of what Claude spends per
review round grows with no step for anyone to remember. The recording itself is
`tools/capture_pr_rounds.py` (see `libs/pr_round_capture_lib.py` for what it reads, what it
records and what it deliberately does not).

OPT-IN: a silent no-op unless `CLAUDE_RUNWAY_CAPTURE_PR_ROUNDS` is `1`, `true` or `yes`. The hook
is installed by `setup_project init`/`upgrade` like the other core hooks, but it starts a process,
calls `gh` (the network) and writes outside the project, so it stays off until you switch it on:
export the variable at the OS/shell level (a hook entry has no `env` field, so the shell is the
only channel; see docs/environment-variables.md).

WHAT THIS HOOK DOES, AND WHY IT DOES SO LITTLE
  It reads the payload Claude Code sends on stdin (`session_id`, `transcript_path`, `cwd`),
  checks it, and starts the worker DETACHED (`python -I tools/capture_pr_rounds.py --transcript
  <path>`, in the session's `cwd` so `gh` finds the repository) and returns at once. It never
  opens the transcript itself and never waits: finding the rounds needs several `gh` calls that
  take seconds, and a SessionEnd hook runs under a short timeout while the session is closing.
  The worker's outcome goes to its own log (`pr-round-capture.log` beside the metrics database),
  not to this hook's output.

FAILS OPEN, ALWAYS: every error (an unreadable payload, a missing transcript, a worker that will
not start, anything unexpected) ends in exit 0 with no stdout, and the reason is a line in the
log when the opt-in is on. It never raises and never blocks the session ending.

The transcript is untrusted data. This process does not read it; the worker is started with
`-I` (isolated mode) and treats its text as data, never as instructions.

Setup: see templates/settings.json.template's core SessionEnd blocks (nothing is copied into the
target project):
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "REPLACE-WITH-VENV-PYTHON",
            "args": ["/absolute/path/to/tools-repo/hooks/session_end_pr_rounds.py"]
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

# Same redundant path-resolution order as the other hooks (see compress_output.py's comment).
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_candidates = [os.environ.get("TOOLS_REPO_DIR"), os.path.join(_root, "libs"), _root, _here]
for _dir in _candidates:
    if _dir:
        sys.path.insert(0, _dir)

try:
    import pr_round_capture_lib as capture_lib
except ImportError:
    sys.exit(0)  # the capture isn't installed -- silent no-op

# Where the worker lives, relative to this file (the repo layout, and the pip-installed one
# `<env>/src/{hooks,tools}`, both keep `hooks/` and `tools/` side by side).
TOOLS_DIR = os.path.join(_root, "tools")


def _problem(payload: dict) -> "str | None":
    """Why this payload cannot be captured, or None when it can."""
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return "no session_id in the payload"
    path = payload.get("transcript_path")
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        return "no absolute transcript_path in the payload"
    if not path.endswith(".jsonl"):
        return "transcript_path is not a .jsonl file"
    if not os.path.isfile(path):
        return "the transcript file does not exist"
    cwd = payload.get("cwd")
    if not isinstance(cwd, str) or not os.path.isdir(cwd):
        # Without the session's folder `gh` cannot tell which repository the PR is in.
        return "the session's cwd is missing or gone"
    return None


def handle(payload, *, spawn=None, log_path=None) -> None:
    """Start the worker for this payload's transcript. `spawn` and `log_path` are injectable for
    the tests. Never raises."""
    log = log_path if log_path is not None else capture_lib.default_log_path()
    try:
        if not isinstance(payload, dict):
            capture_lib.log_note(log, "hook: the payload is not a JSON object; nothing captured")
            return
        problem = _problem(payload)
        if problem:
            capture_lib.log_note(log, f"hook: nothing captured: {problem}")
            return
        script = os.path.join(TOOLS_DIR, "capture_pr_rounds.py")
        if not os.path.isfile(script):
            capture_lib.log_note(log, "hook: nothing captured: tools/capture_pr_rounds.py is missing")
            return
        python = sys.executable
        if not python:
            capture_lib.log_note(log, "hook: nothing captured: no Python interpreter path to start the worker with")
            return
        command = capture_lib.worker_command(python, TOOLS_DIR, payload["transcript_path"])
        (spawn or capture_lib.spawn_worker)(command, payload["cwd"])
    except Exception as e:  # noqa: BLE001 -- fail open: a session end must never fail on this hook
        capture_lib.log_note(log, f"hook: could not start the worker: {type(e).__name__}")


def main() -> None:
    # Force UTF-8 on stdin before the read (Windows defaults to the locale codepage; see
    # session_end_savings.py for the history). Guarded: a non-reconfigurable stream has none.
    _reconfigure = getattr(sys.stdin, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    try:
        if not capture_lib.capture_enabled():
            sys.exit(0)
        try:
            payload = json.load(sys.stdin)
        except (json.JSONDecodeError, ValueError):
            capture_lib.log_note(capture_lib.default_log_path(), "hook: the payload is not valid JSON; nothing captured")
            sys.exit(0)
        handle(payload)
    except Exception:  # noqa: BLE001 -- includes anything the log itself could throw
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
