#!/usr/bin/env python3
"""
Doctor/validator for a target project's dual-config `CLAUDE_RUNWAY_*` env
vars -- diffs `.mcp.json`'s `local-compress` env block against the live
shell environment for each variable README's "Environment variables" table
marks "both" (`CLAUDE_RUNWAY_LMSTUDIO_URL`/`_MODEL`/`_TRACK_SAVINGS`/
`_SAVINGS_DB`), flagging any mismatch instead of relying on a human to
notice that the MCP-server side (`compress_file`/`fetch_url`) and the
hook side (`compress_bash_output.py`/`session_end_savings.py`) have quietly
drifted apart (issue #49, GROW-02 in `.plans/code-review-2026-08-18.md`).

Usage:
    python tools/doctor.py /path/to/target-repo

Run this from the SAME shell (or an equivalent one with the same exports)
that launches `claude` for that project -- it reads `os.environ` from
whatever process runs it, matching the README's own "set both" advice.

Exit code is 1 if any mismatch is found, 0 otherwise (including when
`local-compress` isn't configured at all -- e.g. a `--qdrant-only` setup --
which is a valid state, not a misconfiguration) -- usable as a CI/
pre-flight gate, not just an interactive report.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from doctor_lib import DoctorResult, run_doctor  # noqa: E402


def format_report(result: DoctorResult) -> str:
    if result.config_error:
        # Distinct from "no local-compress configured" below -- this means
        # .mcp.json itself is malformed/unreadable, not a valid config that
        # simply omits local-compress. Found in PR #119 review: the two
        # were previously conflated, so a broken .mcp.json silently
        # reported "nothing to check" and exited 0 instead of surfacing
        # the real problem.
        return f"error: {result.config_error}"
    if not result.local_compress_configured:
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
        lines.append(f"    .mcp.json (local-compress): {m.mcp_json_value!r}")
        lines.append(f"    shell (os.environ):         {m.shell_value!r}")
        lines.append("")
    lines.append(
        "These need matching values in both places -- see README.md's 'Environment variables' "
        "/ 'Keeping them in sync' sections for what each divergence actually does."
    )
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target_repo", help="Path to the project repo whose .mcp.json to check")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    Also the entry point pip/pipx/uvx wires up as the `claude-runway-doctor`
    console script (issue #50) -- `[project.scripts]` in pyproject.toml
    points straight at this existing function, no new wrapper needed since
    it already took this shape for `python tools/doctor.py` (the
    `if __name__` guard below).
    """
    args = parse_args(argv)
    target_repo = Path(args.target_repo).resolve()
    if not target_repo.is_dir():
        print(f"error: target repo path does not exist or is not a directory: {target_repo}", file=sys.stderr)
        return 1

    result = run_doctor(target_repo, shell_env=os.environ)
    print(format_report(result))
    return 1 if (result.config_error or result.mismatches) else 0


if __name__ == "__main__":
    sys.exit(main())
