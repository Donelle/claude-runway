#!/usr/bin/env python3
"""
CLI: record one PR's review rounds from one session transcript to the shared metrics store
(`metric_id` `pr-review-rounds`; see docs/metrics.md and libs/pr_round_record_lib.py).
Stdlib only, and `gh` for the GitHub fields.

    python tools/record_pr_rounds.py --transcript PATH --pr N [--repo OWNER/NAME]
        [--merge-time ISO] [--db PATH] [--dry-run] [--format text|json]

Run it from a checkout of the PR's repository (or pass --repo) so `gh` knows which one. It
is idempotent: running it again over the same transcript writes nothing. It never fails the
shell on a GitHub or store problem; it says so in its output and exits 0. Exit 2 is a bad
argument, 1 is a transcript that could not be read.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import metrics_lib  # noqa: E402
import pr_round_record_lib as P  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Record one PR's review rounds from a session transcript.")
    ap.add_argument("--transcript", required=True, help="the session transcript (.jsonl)")
    ap.add_argument("--pr", required=True, type=int, help="the pull request number")
    ap.add_argument("--repo", help="owner/name for the gh calls (default: the current checkout's remote)")
    ap.add_argument("--merge-time", help="ISO time that ends the last window (default: the transcript's last line)")
    ap.add_argument("--db", help="metrics database path (default: CLAUDE_RUNWAY_METRICS_DB or ~/.claude/claude-runway/metrics.db)")
    ap.add_argument("--dry-run", action="store_true", help="build the events and print them; write nothing")
    ap.add_argument("--format", choices=("text", "json"), default="text")
    args = ap.parse_args(argv)
    if args.pr <= 0:
        print("Error: --pr must be a positive integer.", file=sys.stderr)
        return 2
    if args.repo is not None and not P.valid_repo(args.repo):
        print(f"Error: --repo must look like owner/name, got {args.repo!r}.", file=sys.stderr)
        return 2
    store = metrics_lib.MetricsStore(Path(args.db) if args.db else None)
    summary = P.record_pr_rounds(args.transcript, args.pr, repo=args.repo, merge_time=args.merge_time,
                                 store=store, dry_run=args.dry_run)
    if args.format == "json":
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"PR {summary['pr']}: {summary['rounds']} round(s), {summary['fetch_calls']} fetch call(s) found, "
              f"{summary['fetch_events']} with an event"
              f"{', session ' + summary['session_id'] if summary['session_id'] else ''}")
        if summary["repo"]:
            print(f"repo: {summary['repo']} ({summary['repo_source']})")
        if args.dry_run:
            print(f"dry run: {len(summary['events'])} event(s) built, none written")
        else:
            print(f"written {summary['written']}, already recorded {summary['existing']}, "
                  f"failed {summary['failed']}, not attempted {summary['not_attempted']}")
        if summary["skipped_fetches"]:
            print(f"{summary['skipped_fetches']} fetch call(s) were not read (unrecognised output)")
        for err in summary["github_errors"]:
            print(f"GitHub field left null: {err}")
        if summary["store_error"]:
            print(f"Store problem: {summary['store_error']}")
        if not summary["ok"]:
            print("The transcript could not be read.")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
