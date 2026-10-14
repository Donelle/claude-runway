#!/usr/bin/env python3
"""
Bounded wait for GitHub Copilot's review of one specific commit on a PR, used by
/my-gh-autowork's subagent prompt (its Step 19 and the follow-up pattern that reuses it).

WHY A HELPER, AND WHY BOUNDED: the subagent's prompt cache lapses after roughly five
minutes idle. One Bash call that sleeps for nine minutes therefore makes the next
request re-write the whole context at the cache-write rate. The wait is instead made
of several short calls (each at most ~3 minutes, so the cache stays warm), and shell
state does not persist between Bash calls, so everything a call needs comes in as
arguments and the answer comes out as plain text:

  - one or more `review:<id>` lines: Copilot's review(s) of exactly `head_sha` that are
    not in the seen set (the caller carries on to read the findings), or
  - `NOT_YET`: nothing arrived in this call's window; the caller makes another call
    with the SAME head SHA and seen ids. After the last allowed call it reports FAILED,
    never "clean".

The matching rules are the ones the old inline loop had: the reviewer must be
`copilot-pull-request-reviewer[bot]`, the review's `commit_id` must equal the full
40-character HEAD SHA exactly (a short SHA can never match, so it is rejected up front
instead of silently burning the whole window), and the caller's own reviews never count.

Stdlib only; the `gh` call is injectable so tests never touch the network.
"""

import json
import re
import subprocess
import time
from typing import Callable, Iterable, Optional

COPILOT_LOGIN = "copilot-pull-request-reviewer[bot]"
DEFAULT_REPO = "Donelle/claude-runway"
# Per call: 12 x 15s = 180s of sleep, well under the ~5 minute cache lifetime, and a
# subagent makes at most MAX_CALLS of them, so the overall window stays about 9 minutes.
DEFAULT_ITERATIONS = 12
DEFAULT_INTERVAL_S = 15
MAX_CALLS = 3
# The most one call may wait on its own. The prompt cache lapses after about five
# minutes, so a longer single call defeats the point; tests pin this ceiling.
MAX_CALL_SECONDS = 240
# One `gh` request may not hang the call: it is cut off after this long and counted as a
# failed look. The poll's own deadline (below) covers sleeping AND fetching, so a run of
# slow fetches cannot stretch one call past MAX_CALL_SECONDS (worst case: the deadline
# plus one fetch timeout).
FETCH_TIMEOUT_S = 30
DEADLINE_S = MAX_CALL_SECONDS - FETCH_TIMEOUT_S - 10

NOT_YET = "NOT_YET"
_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")


class FetchError(Exception):
    pass


def validate_sha(sha: str) -> bool:
    return bool(_SHA_RE.match(sha or ""))


def normalize_seen(seen: Iterable[str]) -> set:
    """Accept `review:123`, `123`, or a comma/space separated mix; return bare ids."""
    out = set()
    for item in seen:
        for part in re.split(r"[,\s]+", str(item)):
            if part:
                out.add(part.split(":", 1)[1] if part.startswith("review:") else part)
    return out


def parse_paginated(text: str) -> list:
    """`gh api --paginate` prints one JSON array per page, back to back."""
    decoder = json.JSONDecoder()
    items: list = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        value, i = decoder.raw_decode(text, i)
        if isinstance(value, list):
            items.extend(value)
        else:
            items.append(value)
    return items


def gh_fetch_reviews(repo: str, pr: int, run: Callable = subprocess.run) -> list:
    try:
        proc = run(["gh", "api", f"repos/{repo}/pulls/{pr}/reviews", "--paginate"],
                   capture_output=True, text=True, timeout=FETCH_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError) as e:
        # A stalled `gh` or one that cannot start is a failed look, not a crash.
        raise FetchError(f"{type(e).__name__}: {e}"[:300]) from e
    if proc.returncode != 0:
        raise FetchError((proc.stderr or "gh api failed").strip()[:300])
    try:
        return parse_paginated(proc.stdout)
    except ValueError as e:
        raise FetchError(f"unreadable gh output: {e}") from e


def new_review_ids(reviews: list, head_sha: str, seen: set, me: Optional[str]) -> list:
    out = []
    for r in reviews:
        if not isinstance(r, dict):
            continue
        user = (r.get("user") or {}).get("login")
        if user != COPILOT_LOGIN or (me and user == me):
            continue
        if r.get("commit_id") != head_sha or r.get("id") is None:
            continue
        rid = str(r["id"])
        if rid not in seen and rid not in out:
            out.append(rid)
    return out


def poll(fetch: Callable[[], list], head_sha: str, seen: Iterable[str], me: Optional[str],
         iterations: int = DEFAULT_ITERATIONS, interval_s: float = DEFAULT_INTERVAL_S,
         sleep: Callable[[float], None] = time.sleep, deadline_s: float = DEADLINE_S,
         clock: Callable[[], float] = time.monotonic) -> tuple:
    """Returns (ids, errors, looks): the new review ids found (empty means NOT_YET), the
    number of looks whose fetch failed, and how many looks were made in all. The loop also stops once `deadline_s` of
    wall-clock time (sleeping plus fetching) has passed, so one call never outlasts the
    prompt cache however slow GitHub is. A failed fetch is retried next iteration;
    the caller treats "every iteration failed" as an error, not as a timeout."""
    seen_ids = normalize_seen(seen)
    errors = looks = 0
    start = clock()
    for _ in range(max(1, iterations)):
        looks += 1
        try:
            ids = new_review_ids(fetch(), head_sha, seen_ids, me)
        except FetchError:
            errors += 1
            ids = []
        if ids:
            return ids, errors, looks
        # Sleep after every empty look, the last one included, so a full NOT_YET call
        # really covers iterations x interval (three calls are the documented nine minutes).
        if clock() - start + interval_s > deadline_s:
            break
        sleep(interval_s)
    return [], errors, looks
