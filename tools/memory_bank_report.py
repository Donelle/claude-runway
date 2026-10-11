#!/usr/bin/env python3
"""
Baseline memory-bank staleness report (issue #338) -- dead-memory ratio and
recall payload trend, per repo/project, computed read-only from the existing
`memory-events.db` usage log. See `libs/memory_bank_report_lib.py` for what
each number means and, as importantly, what the existing data can't answer
(score trend, response size, zero-hit calls).

Usage:
    python tools/memory_bank_report.py                  # weekly buckets
    python tools/memory_bank_report.py --bucket month --min-age-days 7
    python tools/memory_bank_report.py --json

Reads the same file the memory-bank server writes: `CLAUDE_RUNWAY_MEMORY_EVENTS_DB`
if set, else `~/.claude/claude-runway/memory-events.db` (override with --db).
Never creates or writes that file.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import memory_bank_report_lib as report_lib  # noqa: E402
import memory_events_lib  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline memory-bank staleness report (issue #338).")
    parser.add_argument("--db", type=Path, default=None,
                        help="memory-events.db path (default: the memory-bank server's own resolution).")
    parser.add_argument("--bucket", choices=("week", "month"), default="week",
                        help="Time bucket for the recall payload trend (default: week).")
    parser.add_argument("--min-age-days", type=float, default=0.0,
                        help="Only count memories remembered at least this many days ago in the dead-memory ratio.")
    parser.add_argument("--exclude-project", action="append", default=[], metavar="PROJECT",
                        help="Exclude rows logged by this calling project (repeatable).")
    parser.add_argument("--no-default-excludes", action="store_true",
                        help=f"Don't exclude {', '.join(report_lib.DEFAULT_EXCLUDED_PROJECTS)} "
                             "(the test-suite's project id) by default.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON instead of text.")
    args = parser.parse_args(argv)
    # math.isfinite too, not just >= 0 (PR #424 review): argparse's float()
    # accepts "inf"/"nan", and neither is a usable age.
    if not math.isfinite(args.min_age_days) or args.min_age_days < 0:
        parser.error("--min-age-days must be a finite number, 0 or greater")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    excluded = [] if args.no_default_excludes else list(report_lib.DEFAULT_EXCLUDED_PROJECTS)
    excluded += [p for p in args.exclude_project if p not in excluded]
    db_path = args.db.expanduser() if args.db else memory_events_lib.resolve_db_path()
    report = report_lib.build_report(
        db_path, bucket=args.bucket, excluded_projects=excluded, min_age_days=args.min_age_days,
    )
    print(json.dumps(report, indent=2) if args.json else report_lib.format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
