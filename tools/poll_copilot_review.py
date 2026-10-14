#!/usr/bin/env python3
"""
One bounded wait for Copilot's review of a specific commit (used by /my-gh-autowork's
subagent prompt, Step 19). Prints `review:<id>` lines for new reviews, or `NOT_YET`.

    python tools/poll_copilot_review.py --pr <PR> --head-sha <40-char sha> \
        --me <your login> [--seen review:1,review:2] [--iterations 12] [--interval 15]

Exit 0 with ids or NOT_YET; exit 2 for a bad argument (e.g. a short SHA); exit 1 with
`ERROR: ...` when every attempt to reach GitHub failed (not a timeout, not "clean").
Each call is meant to run in its own Bash call (at most ~3 minutes) so the prompt
cache stays warm; see libs/copilot_review_poll_lib.py for why.
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import copilot_review_poll_lib as P  # noqa: E402


def main(argv=None, fetch=None, sleep=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pr", type=int, required=True)
    p.add_argument("--head-sha", required=True)
    p.add_argument("--me", default="", help="your own login; your reviews never count")
    p.add_argument("--seen", action="append", default=[], help="seen review ids (review:<id> or bare), comma separated; repeatable")
    p.add_argument("--iterations", type=int, default=P.DEFAULT_ITERATIONS)
    p.add_argument("--interval", type=float, default=P.DEFAULT_INTERVAL_S)
    p.add_argument("--repo", default=P.DEFAULT_REPO)
    args = p.parse_args(argv)
    if not P.validate_sha(args.head_sha):
        print("ERROR: --head-sha must be the full 40-character lowercase hex SHA from `git rev-parse HEAD`.")
        return 2
    if args.iterations < 1 or args.interval < 0 or args.iterations * args.interval > P.MAX_CALL_SECONDS:
        print(f"ERROR: one call may wait at most {P.MAX_CALL_SECONDS}s (the prompt cache lapses after about five minutes); "
              "make several calls instead.")
        return 2
    fetch_fn = fetch or (lambda: P.gh_fetch_reviews(args.repo, args.pr))
    ids, errors, looks = P.poll(fetch_fn, args.head_sha, args.seen, args.me or None, args.iterations, args.interval,
                         sleep or time.sleep)
    if ids:
        for rid in ids:
            print(f"review:{rid}")
        return 0
    if errors >= looks:
        print("ERROR: could not reach GitHub on any attempt; this is not a timeout and not a clean review.")
        return 1
    print(P.NOT_YET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
