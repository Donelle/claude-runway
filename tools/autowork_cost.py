#!/usr/bin/env python3
"""
CLI for /my-gh-autowork's transcript-derived per-ticket cost (issue #455). Stdlib
only. Reads subagent transcripts under ~/.claude/projects (or --projects-root) and
writes the shared metrics store (metric_id="autowork").

    python tools/autowork_cost.py record --issue N --outcome merged \
        (--agent-id ID | --tool-use-id toolu_...) [--pr P] [--rounds N] \
        [--manual-interventions N] [--findings JSON]
        Logs one ticket row whose wall_clock_s / subagent_tokens / tool_calls /
        model and per-model `cost` split are COMPUTED from that subagent's
        transcript instead of typed in. A transcript that cannot be found still
        records the row, marked cost.available=false, so a missing figure is never
        mistaken for a free run. Prints "OK" or "Error: ..."; always exits 0 for a
        recorded row (logging must never change what the run does next).

    python tools/autowork_cost.py backfill [--apply]
        Rows with no figures get a NEW `ticket_cost` event (append-only; rows that
        already have figures are never touched). Dry run unless --apply. Rows whose
        transcript is gone are marked unavailable.

    python tools/autowork_cost.py compare
        Existing hand-copied figures beside the transcript totals, so the old data
        can be judged. Changes nothing.

    python tools/autowork_cost.py report [--format json]
        Per-ticket and per-model totals and the cache-read share.
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import autowork_cost_lib as C  # noqa: E402
import metrics_lib  # noqa: E402

METRIC_ID = "autowork"


def _store(args) -> "metrics_lib.MetricsStore":
    from pathlib import Path
    return metrics_lib.MetricsStore(Path(args.db) if args.db else None)


def _rows(store) -> list:
    return store.detail(METRIC_ID, limit=1_000_000)


def cmd_record(args) -> int:
    root = args.projects_root or C.default_projects_root()
    path = C.locate(root, agent_id=args.agent_id, tool_use_id=args.tool_use_id)
    attempt = None
    if path:
        try:
            attempt = C.parse_attempt(path)
        except Exception:
            attempt = None
    base = {"issue": args.issue, "pr": args.pr, "outcome": args.outcome.upper(),
            "rounds": args.rounds, "manual_interventions": args.manual_interventions}
    if args.findings:
        try:
            findings = json.loads(args.findings)
        except ValueError as e:
            print(f"Error: --findings must be JSON ({e}).")
            return 2
        base["findings"] = findings
    reason = "subagent transcript not found or has no usage data"
    if attempt is not None and attempt["issue"] is not None and attempt["issue"] != args.issue:
        # A valid agent id from another ticket must not file its usage under this one.
        reason = f"transcript {attempt['agent_id']} is for issue #{attempt['issue']}, not #{args.issue}"
        attempt = None
    meta = C.ticket_metadata(attempt, base, reason)
    if attempt is None:
        print(f"Warning: {reason}; recording the row marked cost.available=false.", file=sys.stderr)
    ok = _store(args).record(METRIC_ID, "ticket_" + args.outcome.lower(), 1.0, meta,
                             session_id=f"issue-{args.issue}-{time.strftime('%H%M%S')}")
    print("OK" if ok else f"Error: could not write metric {METRIC_ID}/ticket_{args.outcome.lower()} to the database.")
    return 0


def cmd_backfill(args) -> int:
    store = _store(args)
    rows = _rows(store)
    attempts = C.load_attempts(args.projects_root or C.default_projects_root())
    actions = C.plan_backfill(rows, attempts)
    found = sum(1 for a in actions if a["attempt"])
    print(f"{len(actions)} rows lack figures: {found} matched to a transcript, {len(actions) - found} unavailable.")
    for a in actions:
        r = a["row"]
        print(f"  {r.get('session_id')} #{C.row_issue(r)}: " + (f"agent {a['attempt']['agent_id']}" if a["attempt"] else "unavailable"))
    if not args.apply:
        print("Dry run; pass --apply to write.")
        return 0
    failed = 0
    for a in actions:
        r = a["row"]
        if not store.record(METRIC_ID, C.COST_EVENT, 0.0, a["metadata"], session_id=r.get("session_id"),
                            event_timestamp=r.get("event_timestamp")):
            failed += 1
    print(f"Wrote {len(actions) - failed} ticket_cost rows" + (f"; {failed} failed." if failed else "."))
    return 1 if failed else 0


def cmd_compare(args) -> int:
    rows = _rows(_store(args))
    attempts = C.load_attempts(args.projects_root or C.default_projects_root())
    print(C.format_compare(C.compare_existing(rows, attempts)))
    return 0


def cmd_report(args) -> int:
    rep = C.build_report(_rows(_store(args)))
    print(json.dumps(rep, indent=2) if args.format == "json" else C.format_report(rep))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="metrics database path (default: the shared metrics.db)")
    p.add_argument("--projects-root", help="Claude Code projects dir (default ~/.claude/projects)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--issue", type=int, required=True)
    r.add_argument("--outcome", choices=["merged", "blocked", "failed"], required=True)
    r.add_argument("--pr", type=int)
    r.add_argument("--agent-id")
    r.add_argument("--tool-use-id")
    r.add_argument("--rounds", type=int, default=0)
    r.add_argument("--manual-interventions", type=int, default=0)
    r.add_argument("--findings")
    b = sub.add_parser("backfill")
    b.add_argument("--apply", action="store_true")
    sub.add_parser("compare")
    rp = sub.add_parser("report")
    rp.add_argument("--format", choices=["text", "json"], default="text")
    args = p.parse_args(argv)
    if args.cmd == "record" and not (args.agent_id or args.tool_use_id):
        p.error("record needs --agent-id or --tool-use-id")
    return {"record": cmd_record, "backfill": cmd_backfill, "compare": cmd_compare, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
