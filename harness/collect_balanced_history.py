#!/usr/bin/env python3
"""Record a balanced set of Copilot's past reviews, so the history folder can be rebuilt on any clone.

    .venv/bin/python harness/collect_balanced_history.py              # pick 9 of the last 150 merged PRs and record them
    .venv/bin/python harness/collect_balanced_history.py --dry-run    # only show the pick
    .venv/bin/python harness/collect_balanced_history.py --count 12 --scan 200 --exclude 552

`harness/history/` is not committed (git ignores it), so a clone starts with none. This picks the PRs for it by a fixed
rule instead of by hand: every kind of change (docs, shell, Python, other), size, number of Copilot rounds and number of
inline findings gets at least one PR (see `history_lib.choose_balanced`), and then records them with
`collect_copilot_history.py`. The same PRs give the same set.

READ ONLY, like the collector: `gh pr list` and GETs of each PR's reviews and comments, then the collector's own GETs and
`git fetch`. Nothing here requests a review.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_copilot_history as collector  # noqa: E402
import history_lib  # noqa: E402

DEFAULT_COUNT = 9
DEFAULT_SCAN = 150


def profiles_of(scan: int, repo: str = collector.REPO) -> list:
    """A profile for each of the last `scan` merged PRs that Copilot reviewed (an open PR is still changing, so it is left out)."""
    prs = collector.gh_json("pr", "list", "-R", repo, "--state", "merged", "--limit", str(scan), "--json", "number,title,additions,deletions,files")
    assert isinstance(prs, list)
    found = []
    for pr in prs:
        number = pr["number"]
        reviews = collector._flatten(collector.gh_json("api", f"repos/{repo}/pulls/{number}/reviews", "--paginate", "--slurp"))
        review_ids = {r["id"] for r in reviews if history_lib.is_copilot((r.get("user") or {}).get("login"))}
        if not review_ids:
            continue
        comments = collector._flatten(collector.gh_json("api", f"repos/{repo}/pulls/{number}/comments", "--paginate", "--slurp"))
        inline = sum(1 for c in comments if c.get("pull_request_review_id") in review_ids and not c.get("in_reply_to_id") and c.get("path"))
        found.append(history_lib.profile(pr, len(review_ids), inline))
    return found


def describe(profiles: list, chosen: list) -> list:
    by_number = {p["pr"]: p for p in profiles}
    return [f"#{n}: {' / '.join(by_number[n]['traits'])} ({by_number[n]['rounds']} round(s), {by_number[n]['inline']} finding(s)) {by_number[n]['title'][:60]}"
            for n in chosen]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"how many PRs to record (default {DEFAULT_COUNT})")
    parser.add_argument("--scan", type=int, default=DEFAULT_SCAN, help=f"how many recent merged PRs to choose from (default {DEFAULT_SCAN})")
    parser.add_argument("--exclude", type=int, action="append", default=[], metavar="PR", help="a PR never to pick (repeatable)")
    parser.add_argument("--out", default=str(collector.DEFAULT_OUT), help="directory for the snapshots")
    parser.add_argument("--dry-run", action="store_true", help="show the pick and record nothing")
    args = parser.parse_args(argv)
    profiles = profiles_of(args.scan)
    excluded = set(args.exclude)
    chosen = history_lib.choose_balanced(profiles, args.count, excluded)
    if not chosen:
        print("no merged PR in the scan has a Copilot review with an inline finding", file=sys.stderr)
        return 1
    print(f"{len(chosen)} of {len(profiles)} reviewed merged PRs (of the last {args.scan}), kind / size / rounds / findings:")
    print("\n".join("  " + line for line in describe(profiles, chosen)))
    if args.dry_run:
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = 0
    recorded: set = set()
    while True:
        # A PR that was retargeted cannot be replayed (Copilot saw a base GitHub no longer records), so it is dropped and
        # the next best PR takes its place.
        retargeted = set()
        for number in chosen:
            if number in recorded:
                continue
            snapshot = collector.collect(number)
            if snapshot["base_changes"]:
                retargeted.add(number)
                print(f"PR #{number}: retargeted {snapshot['base_changes']} time(s), so it cannot be replayed; choosing another")
                continue
            path = out / f"pr-{number}.json"
            path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            failures = collector.fetch_reviewed_commits(snapshot)
            for failure in failures:
                print(f"PR #{number}: {failure}", file=sys.stderr)
            status = status or (1 if failures else 0)
            recorded.add(number)
            print(f"PR #{number}: {len(snapshot['rounds'])} Copilot review(s) -> {path}")
        if not retargeted:
            for number in recorded - set(chosen):  # a pick that the replacements pushed out of the set
                (out / f"pr-{number}.json").unlink(missing_ok=True)
            return status
        excluded |= retargeted
        chosen = history_lib.choose_balanced(profiles, args.count, excluded)


if __name__ == "__main__":
    raise SystemExit(main())
