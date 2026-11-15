#!/usr/bin/env python3
"""
CLI: backfill the shared metrics store's `pr-review-rounds` events from a project's main-session
transcripts that still exist (see libs/pr_round_backfill_lib.py, libs/pr_round_record_lib.py and
docs/metrics.md). Stdlib only, and `gh` for the GitHub fields.

    python -I tools/backfill_pr_rounds.py [--project-dir DIR] [--repo OWNER/NAME]
        [--write] [--include-open] [--db PATH] [--format text|json]

A DRY RUN is the default: every transcript is read and every event built, and nothing is
written (the metrics database is not even opened). `--write` records. Re-running writes nothing
twice. Transcripts are only opened for reading; this never modifies, moves or deletes one.
Run it from a checkout of the PR's repository (or pass --repo) so `gh` knows which one.

`--project-dir` is the folder under ~/.claude/projects that holds the project's transcripts;
the default is the folder of the checkout you run it in (the MAIN checkout, from a linked
worktree). A PR that is still open is skipped (its last window would be stored as open and never
corrected); `--include-open` records it anyway. Exit 0 on a finished run (a skipped PR or a
GitHub problem is in the report, not an error), 1 when the folder is missing or the store failed,
2 for a bad argument.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import metrics_lib  # noqa: E402
import pr_round_backfill_lib as B  # noqa: E402
import pr_round_record_lib as P  # noqa: E402


def format_text(report: dict) -> str:
    """The report as text: the plan (or the result), then the counts."""
    lines = [f"{'WRITE' if report['write'] else 'DRY RUN (nothing is written; pass --write to record)'}: "
             f"{report['project_dir']}"]
    for pr in sorted(report["prs"]):
        e = report["prs"][pr]
        lines.append(f"  PR {pr}: {e['state']}{' ' + e['end'] if e['end'] else ''}, {e['transcripts']} transcript(s), "
                     f"{e['rounds']} round(s), {e['fetch_events']} fetch event(s)")
    if report["skipped"]:
        lines.append("Skipped:")
        for s in report["skipped"]:
            lines.append(f"  {s['transcript']}{' PR ' + str(s['pr']) if s['pr'] else ''}: {s['reason']}")
    for pair in report["shared_tool_use_ids"]:
        lines.append(f"WARNING: {pair['a']} and {pair['b']} share {pair['ids']} tool-use id(s); a resumed session "
                     f"written as a new file would, and its local rounds would count twice")
    for err in report["github_errors"]:
        lines.append(f"GitHub field left null: {err}")
    if report["store_error"]:
        lines.append(f"Store problem: {report['store_error']}")
    lines.append(f"Transcripts: {report['transcripts_found']} found, {report['transcripts_read']} read, "
                 f"{len(report['transcripts_unreadable'])} unreadable, {report['transcripts_with_feedback']} with a feedback "
                 f"fetch, {report['transcripts_without_feedback']} without")
    lines.append(f"PRs: {report['prs_found']} found, {report['prs_recorded']} "
                 f"{'recorded' if report['write'] else 'that would be recorded'}, "
                 f"{len(report['skipped'])} transcript/PR pair(s) skipped, {report['candidates_without_fetch']} "
                 f"other PR number(s) named in a command with no feedback fetch")
    lines.append(f"Events: {report['rounds']} round(s), {report['fetch_events']} fetch event(s), "
                 f"{report['events_built']} built; {report['unrecognised_fetches']} fetch call(s) not read (unrecognised)")
    if report["write"]:
        lines.append(f"Store: written {report['written']}, already recorded {report['existing']}, "
                     f"failed {report['failed']}, not attempted {report['not_attempted']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Backfill PR review rounds from a project's session transcripts.")
    ap.add_argument("--project-dir", help="folder holding the project's transcripts (default: this checkout's folder under ~/.claude/projects)")
    ap.add_argument("--projects-root", help="the projects folder the default is derived under (default: ~/.claude/projects)")
    ap.add_argument("--repo", help="owner/name for the gh calls (default: the current checkout's remote)")
    ap.add_argument("--write", action="store_true", help="record the events (default: dry run, nothing written)")
    ap.add_argument("--include-open", action="store_true",
                    help="also record a PR that is still open, its last window stored as open and never corrected")
    ap.add_argument("--db", help="metrics database path (default: CLAUDE_RUNWAY_METRICS_DB or ~/.claude/claude-runway/metrics.db)")
    ap.add_argument("--format", choices=("text", "json"), default="text")
    args = ap.parse_args(argv)
    if args.repo is not None and not P.valid_repo(args.repo):
        print(f"Error: --repo must look like owner/name, got {args.repo!r}.", file=sys.stderr)
        return 2
    if args.db and not args.write:
        print("Error: --db only matters with --write; a dry run never opens the database.", file=sys.stderr)
        return 2
    project_dir = args.project_dir or B.default_project_dir(projects_root=args.projects_root)
    if not os.path.isdir(project_dir):
        print(f"Error: no transcripts folder at {project_dir} (pass --project-dir).", file=sys.stderr)
        return 1
    store = metrics_lib.MetricsStore(Path(args.db) if args.db else None) if args.write else None
    report = B.backfill(project_dir, repo=args.repo, store=store, gh=P.run_gh, write=args.write,
                        include_open=args.include_open)
    print(json.dumps(report, indent=2, default=str) if args.format == "json" else format_text(report))
    return 1 if report["store_error"] or report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
