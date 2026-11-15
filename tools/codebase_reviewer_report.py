#!/usr/bin/env python3
"""
Shows what the codebase-reviewer's runs cost, from its own database (CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB).

    python tools/codebase_reviewer_report.py response_times [--model ID] [--this-machine]
    python tools/codebase_reviewer_report.py review_times [--this-machine]
    python tools/codebase_reviewer_report.py under_target --minutes 10 [--this-machine]

`response_times` is the response time of model X per request, by role. `review_times` is the review wall time
per model combination, split into load and each stage. `under_target` lists the combinations whose recorded
reviews came in under a time you give. Add `--json` for machine-readable output. The same views are the
codebase-reviewer server's `review_report` tool. Standard library only; exits 1 with a message when there is
nothing to read yet.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import reviewer_report_lib  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report what codebase-reviewer runs cost.")
    parser.add_argument("view", choices=reviewer_report_lib.VIEWS)
    parser.add_argument("--model", help="response_times: only this model id")
    parser.add_argument("--minutes", type=float, help="under_target: the target review time in minutes")
    parser.add_argument("--this-machine", action="store_true", help="only runs recorded on this machine")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        text = reviewer_report_lib.report(
            args.view, model=args.model, target_minutes=args.minutes, this_machine=args.this_machine, as_json=args.as_json
        )
    except reviewer_report_lib.ReportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
