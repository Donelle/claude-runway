#!/usr/bin/env python3
"""
CLI: Claude's tokens per Copilot review round, per repository, with the rounds split into those
answered with the local codebase-reviewer ("reviewer used") and those answered without it
("not used"), read from the `pr-review-rounds` events in the shared metrics store. See
`libs/pr_round_report_lib.py` for what is counted and what the groups mean, and `EVALUATION.md`'s
Track G for how to read it.

    python tools/pr_round_report.py [--db PATH] [--repo OWNER/NAME] [--exclude-open] [--json]

It groups rounds, not pull requests, and keeps rows with no repository in a group of their own.
Reads `CLAUDE_RUNWAY_METRICS_DB` if set, else `~/.claude/claude-runway/metrics.db` (override with
`--db`). Opens the database read-only and never creates or writes it. Exit 0 whether or not
there is data (a store with nothing in it says so), 2 for a bad argument.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import metrics_lib  # noqa: E402
import pr_round_report_lib as report_lib  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Claude's tokens per Copilot review round, by repository and by "
                                             "whether the local reviewer was used.")
    ap.add_argument("--db", type=Path, default=None,
                    help="metrics.db path (default: CLAUDE_RUNWAY_METRICS_DB, else ~/.claude/claude-runway/metrics.db).")
    ap.add_argument("--repo", default=None, metavar="OWNER/NAME",
                    help="Only rows whose recorded repo is this exact value (a project name works too).")
    ap.add_argument("--exclude-open", action="store_true",
                    help="Leave out rounds whose last window has no end (their figures are a floor).")
    ap.add_argument("--json", action="store_true", help="Print the report as JSON instead of text.")
    args = ap.parse_args(argv)
    if args.repo is not None and not args.repo.strip():
        print("Error: --repo must not be empty.", file=sys.stderr)
        return 2
    db_path = args.db.expanduser() if args.db else metrics_lib.resolve_db_path()
    report = report_lib.build_report(db_path, repo=args.repo, exclude_open=args.exclude_open)
    print(json.dumps(report, indent=2) if args.json else report_lib.format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
