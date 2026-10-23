#!/usr/bin/env python3
"""
Doctor/validator for a target project's dual-config `CLAUDE_RUNWAY_*` env
vars -- diffs `.mcp.json`'s `local-compress` env block (and the `qdrant`
block's LM Studio values, which `compact_store` compresses with, and the
`analytics` block's savings DB override, which its savings tools read)
against the live shell environment for each variable the "Environment
variables" doc marks "both" (`CLAUDE_RUNWAY_LMSTUDIO_URL`/`_MODEL`/
`_TRACK_SAVINGS` in `local-compress`, `_SAVINGS_DB` in `analytics`), flagging
any mismatch instead of relying on a human to notice that the MCP-server
side (`compress_file`/`fetch_url`, the savings tools) and the hook side
(`compress_output.py`/`session_end_savings.py`) have quietly drifted apart
(GROW-02 in `.plans/code-review-2026-08-18.md`).

Usage:
    python tools/doctor.py /path/to/target-repo

Run this from the SAME shell (or an equivalent one with the same exports)
that launches `claude` for that project -- it reads `os.environ` from
whatever process runs it, matching the README's own "set both" advice.

Also flags a .claude/settings.json still running the renamed-away
`compress_bash_output.py` hook (`upgrade` fixes it), and, for a
project with a `codebase-reviewer` server, a missing, blank, unreadable or
invalid reviewers file (REVIEWERS_FILE is required and has no default).

Exit code is 1 if any mismatch (or stale hook, or unusable reviewers file) is found, 0 otherwise (including when
there is nothing to check -- e.g. a `--qdrant-only` setup whose `qdrant` block
carries no dual-config values -- which is a valid state, not a misconfiguration) -- usable as a CI/
pre-flight gate, not just an interactive report.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from doctor_lib import DoctorResult, run_doctor  # noqa: E402
from version_lib import version_string  # noqa: E402


def _legacy_hook_note(result: DoctorResult) -> str:
    """Extra section for a settings.json still running the pre-rename
    compress_bash_output.py -- empty string when there's nothing to flag."""
    if not result.legacy_compress_hook_events:
        return ""
    events = ", ".join(result.legacy_compress_hook_events)
    return (
        "\n\nStale hook: .claude/settings.json still runs hooks/compress_bash_output.py "
        f"(under {events}), which was renamed to hooks/compress_output.py -- the old path "
        "no longer exists, so output compression is silently not running. Fix: "
        "`claude-runway-setup upgrade <target_repo>` (migration compress-hook-renamed)."
    )


def _reviewers_file_note(result: DoctorResult) -> str:
    """Extra section for a project with a `codebase-reviewer` block: whether its
    reviewers file (REVIEWERS_FILE, required, no default) loads. Empty when the
    project has no such block."""
    if not result.reviewer_checked:
        return ""
    if result.reviewers_file_problem:
        return (
            "\n\nProblem: the codebase-reviewer server's reviewers file cannot be used, so it would run no review.\n"
            f"{result.reviewers_file_problem}\n"
            "Fix: set REVIEWERS_FILE in that server's env block in .mcp.json to your own reviewers JSON file "
            "(it has no default), and correct what is listed above."
        )
    return f"\n\nReviewers file OK: {result.reviewers_file_summary}."


def format_report(result: DoctorResult) -> str:
    return _format_env_report(result) + _reviewers_file_note(result) + _legacy_hook_note(result)


def _format_env_report(result: DoctorResult) -> str:
    if result.config_error:
        # Distinct from "no local-compress configured" below -- this means
        # .mcp.json itself is malformed/unreadable, not a valid config that
        # simply omits local-compress. Found in review: the two
        # were previously conflated, so a broken .mcp.json silently
        # reported "nothing to check" and exited 0 instead of surfacing
        # the real problem.
        return f"error: {result.config_error}"
    if not (result.local_compress_configured or result.qdrant_checked or result.analytics_checked):
        return (
            f"No 'local-compress' server found in {result.mcp_json_path} -- nothing to check "
            "(expected if this project only uses the Qdrant memory piece, e.g. --qdrant-only)."
        )
    if not result.mismatches:
        return f"OK -- all dual-config env vars agree between {result.mcp_json_path} and the shell."

    lines = [
        f"Found {len(result.mismatches)} dual-config env var mismatch(es) between "
        f"{result.mcp_json_path} and the shell:",
        "",
    ]
    for m in result.mismatches:
        lines.append(f"  {m.var}")
        lines.append(f"    .mcp.json ({m.server}): {m.mcp_json_value!r}")
        lines.append(f"    shell (os.environ): {m.shell_value!r}")
        lines.append("")
    lines.append(
        "These need matching values in both places -- see README.md's 'Environment variables' "
        "/ 'Keeping them in sync' sections for what each divergence actually does."
    )
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=version_string("claude-runway-doctor"))
    parser.add_argument("target_repo", help="Path to the project repo whose .mcp.json to check")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    Also what the `claude-runway-doctor` console script ends up
    calling: `[project.scripts]` in pyproject.toml points at
    `claude_runway:doctor_main`, a small shim that loads this
    file by path and calls this function, which already took this shape for
    `python tools/doctor.py` (the `if __name__` guard below).
    """
    args = parse_args(argv)
    target_repo = Path(args.target_repo).resolve()
    if not target_repo.is_dir():
        print(f"error: target repo path does not exist or is not a directory: {target_repo}", file=sys.stderr)
        return 1

    result = run_doctor(target_repo, shell_env=os.environ)
    print(format_report(result))
    return (
        1
        if (
            result.config_error
            or result.mismatches
            or result.legacy_compress_hook_events
            or result.reviewers_file_problem
        )
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
