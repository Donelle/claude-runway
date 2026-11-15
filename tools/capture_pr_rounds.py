#!/usr/bin/env python3
"""
CLI: record the pull-request review rounds in ONE session transcript to the shared metrics store
(see libs/pr_round_capture_lib.py, libs/pr_round_record_lib.py and docs/metrics.md). This is the
worker `hooks/session_end_pr_rounds.py` starts, detached, when a session ends; it can also be run
by hand on any transcript.

    python -I tools/capture_pr_rounds.py --transcript PATH [--repo OWNER/NAME]
        [--include-open] [--dry-run] [--db PATH] [--log PATH | --no-log] [--format text|json]

Run it with `python -I`: the transcript is untrusted data and isolated mode keeps PYTHONPATH, the
user site and the working directory off the import path. Only the named transcript is read, and
only for reading. `gh` finds the repository from the current directory unless `--repo` is given.

A PR that is still open is skipped (its last window would be stored as open and never corrected;
a later session end or a backfill after the merge records it); `--include-open` records it anyway.
`--dry-run` builds every event and writes nothing (the database is not even opened, and nothing
is logged). Each run that found something to say appends one line to the log (default: beside
the metrics database, `pr-round-capture.log`). Re-running writes nothing twice.

Exit 0 when the run finished (a skipped PR or a GitHub problem is in the report, not an error),
1 when the transcript could not be opened or the store failed, 2 for a bad argument. A hook that
starts this ignores the exit code either way.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import metrics_lib  # noqa: E402
import pr_round_capture_lib as C  # noqa: E402
import pr_round_record_lib as P  # noqa: E402


def format_text(report: dict, dry_run: bool) -> str:
    lines = [f"{'DRY RUN (nothing is written)' if dry_run else 'CAPTURE'}: {report['transcript']}"]
    for e in report["prs"]:
        lines.append(f"  PR {e['pr']}: {e['state']}, {e['rounds']} round(s), {e['fetch_events']} fetch event(s)")
    for s in report["skipped"]:
        lines.append(f"  skipped{' PR ' + str(s['pr']) if s['pr'] else ''}: {s['reason']}")
    for note in report["notes"]:
        lines.append(f"  note: {note}")
    if report["error"]:
        lines.append(f"Error: {report['error']}")
    for err in report["github_errors"]:
        lines.append(f"GitHub field left null: {err}")
    if report["store_error"]:
        lines.append(f"Store problem: {report['store_error']}")
    lines.append(f"PRs: {report['candidates']} candidate(s), {len(report['prs'])} "
                 f"{'that would be recorded' if dry_run else 'recorded'}; {report['unrecognised_fetches']} fetch call(s) "
                 f"not read (unrecognised)")
    if not dry_run:
        lines.append(f"Store: written {report['written']}, already recorded {report['existing']}, failed {report['failed']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Record the PR review rounds in one session transcript.")
    ap.add_argument("--transcript", required=True, help="the session transcript (a .jsonl file) to read")
    ap.add_argument("--repo", help="owner/name for the gh calls (default: the current checkout's remote)")
    ap.add_argument("--include-open", action="store_true",
                    help="also record a PR that is still open, its last window stored as open and never corrected")
    ap.add_argument("--dry-run", action="store_true", help="build the events and write nothing")
    ap.add_argument("--db", help="metrics database path (default: CLAUDE_RUNWAY_METRICS_DB or ~/.claude/claude-runway/metrics.db)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--log", help="log file (default: pr-round-capture.log beside the metrics database)")
    group.add_argument("--no-log", action="store_true", help="write no log line")
    ap.add_argument("--format", choices=("text", "json"), default="text")
    args = ap.parse_args(argv)
    if args.repo is not None and not P.valid_repo(args.repo):
        print(f"Error: --repo must look like owner/name, got {args.repo!r}.", file=sys.stderr)
        return 2
    if args.db and args.dry_run:
        print("Error: --db only matters without --dry-run; a dry run never opens the database.", file=sys.stderr)
        return 2
    db_path = Path(args.db) if args.db else None
    store = None if args.dry_run else metrics_lib.MetricsStore(db_path)
    report = C.capture(args.transcript, repo=args.repo, store=store, gh=P.run_gh, include_open=args.include_open,
                       dry_run=args.dry_run, deadline=time.monotonic() + C.DEFAULT_TIME_BUDGET_SECONDS)
    if not args.dry_run and not args.no_log and C.has_something_to_log(report):
        C.log_note(args.log or C.default_log_path(db_path), C.summary_line(report))
    print(json.dumps(report, indent=2, default=str) if args.format == "json" else format_text(report, args.dry_run))
    return 1 if (report["error"] or report["store_error"] or report["failed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
