#!/usr/bin/env python3
"""Record Copilot's existing reviews of one or more PRs as a snapshot the harnesses can score against.

    .venv/bin/python harness/collect_copilot_history.py 552
    .venv/bin/python harness/collect_copilot_history.py 540 552 --out harness/history

READ ONLY. Every call is a GET: the PR's metadata, its reviews and its review
comments, plus `git fetch` of the PR head and of every reviewed commit by SHA, so they exist
locally. It never requests a review, tags a reviewer or re-requests one, so it
cannot start a Copilot review or spend review credits. The harness replays
history; it does not ask Copilot for anything new.

Writes `<out>/pr-<N>.json` (default `harness/history/`).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import history_lib  # noqa: E402

REPO = "Donelle/claude-runway"
DEFAULT_OUT = Path(__file__).resolve().parent / "history"
REPO_ROOT = Path(__file__).resolve().parent.parent


def gh_json(*args: str) -> object:
    """Run `gh` for a read and parse its JSON. Only ever called with `api` GET paths and `pr view`."""
    done = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"gh {' '.join(args[:3])} failed: {done.stderr.strip()[:300]}")
    return json.loads(done.stdout or "null")


def _flatten(pages: object) -> list:
    """`gh api --paginate --slurp` returns a list of pages, each a list of items."""
    assert isinstance(pages, list)
    return [item for page in pages for item in (page if isinstance(page, list) else [page])]


def collect(number: int, repo: str = REPO) -> dict:
    meta = gh_json("pr", "view", str(number), "-R", repo, "--json", "number,title,baseRefOid,headRefOid")
    reviews = gh_json("api", f"repos/{repo}/pulls/{number}/reviews", "--paginate", "--slurp")
    comments = gh_json("api", f"repos/{repo}/pulls/{number}/comments", "--paginate", "--slurp")
    timeline = gh_json("api", f"repos/{repo}/issues/{number}/timeline", "--paginate", "--slurp")
    assert isinstance(meta, dict)
    retargets = sum(1 for event in _flatten(timeline) if event.get("event") == "base_ref_changed")
    return history_lib.build_snapshot(meta, _flatten(reviews), _flatten(comments), repo, retargets)


def _have_commit(sha: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, check=False).returncode == 0


def fetch_reviewed_commits(snapshot: dict) -> list:
    """Make every commit a replay needs exist locally; returns one plain message per commit that could not be had.

    The PR's current head is not enough: a commit Copilot reviewed that a later force-push or rebase
    discarded is no longer on `pull/N/head`. GitHub still serves it by its full SHA, so each recorded review
    commit and the base are fetched by SHA, and a failure is reported rather than assumed away."""
    number = snapshot["pr"]
    wanted = [("pull request head", f"pull/{number}/head")]
    shas = [("base", snapshot.get("base_sha"))] + [(f"round {i} review commit", r.get("commit")) for i, r in enumerate(snapshot["rounds"], 1)]
    wanted += [(label, sha) for label, sha in shas if sha]
    failures = []
    for label, ref in wanted:
        if ref and not ref.startswith("pull/") and _have_commit(ref):
            continue
        done = subprocess.run(["git", "-C", str(REPO_ROOT), "fetch", "-q", "origin", ref], capture_output=True, text=True, check=False)
        if done.returncode != 0 or (not ref.startswith("pull/") and not _have_commit(ref)):
            failures.append(f"could not fetch the {label} ({ref[:12]}): {done.stderr.strip()[:200] or 'not present after fetch'}; replaying it will fail")
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pr", type=int, nargs="+", help="PR number(s) in " + REPO)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="directory for the snapshots")
    parser.add_argument("--no-fetch", action="store_true", help="do not `git fetch` the PR head (the reviewed commits may then be missing locally)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = 0
    for number in args.pr:
        snapshot = collect(number)
        rounds = snapshot["rounds"]
        path = out / f"pr-{number}.json"
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        if not args.no_fetch:
            failures = fetch_reviewed_commits(snapshot)
            for failure in failures:
                print(f"PR #{number}: {failure}", file=sys.stderr)
            status = status or (1 if failures else 0)
        counts = ", ".join(f"round {i}: {len(r['findings'])}" for i, r in enumerate(rounds, 1)) or "no Copilot review"
        print(f"PR #{number}: {len(rounds)} Copilot review(s) ({counts}) -> {path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
