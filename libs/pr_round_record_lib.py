#!/usr/bin/env python3
"""
Records the review rounds of one pull request, and the feedback fetch calls inside them, to
the shared metrics store (`metrics.db`, `libs/metrics_lib.py`), enriched from GitHub. This
turns `libs/pr_round_cost_lib.py`'s reading of a session transcript into stored data, which
is what a before/after baseline of "what does Claude spend answering each review round" is.
That reader is used exactly as it is (its contract is fixed; this module never edits or
second-guesses what it found), and nothing here parses a transcript itself beyond reading
the `cwd` of its first entry for the project-name fallback (see below).

WHAT IS WRITTEN
  `metric_id` `pr-review-rounds`, `value` 1.0 (the store aggregates only COUNT(*) and
  SUM(value), so every figure lives in `metadata`), `session_id` the transcript's session,
  `event_timestamp` the moment the round or call started (so a backfill buckets by when the
  work happened, not by when it was recorded). Two event types:
  - `pr_round`: one per round, of kind `copilot` or `local`. A `copilot` round also says
    whether the codebase-reviewer was used while its feedback was being answered
    (`local_reviewer_used`, below).
  - `pr_feedback_fetch`: one per feedback fetch call, including a call before the first round
    (`round` None). A fetch call's window runs to the next fetch call or the end of its round
    (see the reader), so a round's fetch windows tile it; the two event types describe the
    same spend at two grains, so SUM them separately, never together.
  Every field is listed, with its meaning, in docs/metrics.md.

WHERE THE ROW CAME FROM (`repo`, `repo_source`, on both event types)
  `metrics.db` is one store for every project, so each event says which repository it is
  about. `repo` is `owner/name` and `repo_source` is `remote` when the repository is known:
  the `repo` the caller passed for the `gh` calls, else what `gh repo view` says for the
  checkout the recorder runs in. When neither is available `repo` is the project name the
  toolkit already uses for a session (`savings_ledger.project_name_from_cwd`: the last
  component of the first `cwd` the transcript records) and `repo_source` is `project`, so a
  folder name is never mistaken for a repository. With neither, both are null: nothing is
  guessed. A session started in a subfolder or a worktree gives that folder's name, which is
  the fallback's limit.

WHETHER THE REVIEWER WAS USED (`local_reviewer_used`, `pr_round` rows of kind `copilot` only)
  A round is: Copilot feedback arrives, changes are made, the changes are pushed. The flag
  says whether the codebase-reviewer's `review` tool was used while the changes that answered
  this round's feedback were made. Concretely it is true when a `local` round starts after
  this Copilot round began and before the next Copilot round begins (strictly: a tie with a
  Copilot round's start sets nothing, see `reviewer_used_flags`), else false. It is per
  round, never per PR (the tool can be switched off part-way through a PR). Decisions:
  - A `review` call before the first Copilot round (a check before or at opening the PR)
    belongs to no Copilot round and sets no flag; it remains its own `local` row.
  - The final round, with no changes after it, is false unless a `review` call follows it in
    the transcript (nothing is guessed about what the call was for).
  - `local` rows and fetch rows do not carry the field.
  - It is computed from the transcript being read, so a review made in a session whose
    transcript is not read (the backfill skips subagent transcripts, and the SessionEnd
    capture is opt-in) shows as false, and the row is never updated afterwards: false means
    "no local review captured for this round", not proof the tool was not used. A `review`
    call that names no PR is attached by the reader to every PR the session fetched
    feedback for, so it can set the flag on a round of another PR.
  None of the three fields is in an idempotency key. They are additive, and `schema` is 2 so
  that a row written before they existed (schema 1: unknown) is told apart from one that has
  them (false).

IDEMPOTENT
  Before writing, `MetricsStore.record_if_absent` looks for a row with the same key, in the
  same write transaction as the insert, so a second run over the same transcript (a session
  end and then a backfill) writes nothing, even when two run at once:
    pr_round:           (pr, kind, review_id, call_time); a local round, which has no review id and
                        can share its call time with a parallel call in one response, also
                        (kind_number, session_id)
    pr_feedback_fetch:  (pr, review_id, call_time, tool_use_id)
  `call_time` is the transcript's own timestamp text for the marker call. None of the GitHub
  fields is in a key, so a run that could not reach GitHub finds the rows an earlier run wrote.
  A PR's rounds can span sessions: each session contributes the rounds and calls its own
  transcript holds. A review that two sessions both fetched is therefore two `pr_round` rows
  (each with its own session's tokens and its own call time), joined by `review_id` and
  `copilot_review_number`. A Copilot round's key holds no session id, on purpose (see
  docs/metrics.md): a resumed session whose transcript copies another's history must not count
  that round twice. The cost is that two sessions whose marker calls for the same review carry
  the same call-time text (to the millisecond) are one row, and the second session's spend for
  it is not recorded.

FAILS OPEN
  `record_pr_rounds` never raises and always returns a summary dict. A `gh` call that fails
  (not installed, not signed in, offline, a rate limit, a PR in another repository) leaves
  the fields it would have filled null and is listed in `github_errors`; nothing is guessed.
  An unreadable or locked store stops the run at the first failed write and is reported in
  `store_error`; the remaining events are counted as `not_attempted`, not written.

KNOWN LIMITS (each also in docs/metrics.md)
  - A window that ends at the transcript's last line (no merge time was given) is recorded as
    it stood, with `window_open` true, and is never corrected: a session resumed later can add
    work to that last round and its last fetch call, and a re-run finds the existing key and
    writes nothing. Rewriting or appending a correction would make COUNT(*) and SUM(value)
    wrong and a round's total disagree with its fetch calls' totals, so the row says instead
    that it may be short. Every other window is bounded by the next round, the next fetch call
    or the merge and cannot grow. Pass the PR's merge time to record a round's final figures.
  - A row written while GitHub was unreachable keeps its null GitHub fields: the store is
    append-only and the key excludes them, so a later run that could reach GitHub finds the
    row and does not upgrade it. `github` on the row says which it is (`ok`, `partial` or
    `unavailable`).
  - PR size is the PR's size when it is recorded (additions, deletions, changed files),
    not at the moment of the review; a PR that grew after a review is recorded with its
    later size on every row.
  - Push time is the earliest check suite GitHub created for the reviewed commit, which is
    when GitHub first saw that commit, not the committer date; null for a commit with no
    check suite. The wait from push to review is null when either end is, or when the review predates it.
  - Findings raised are the review's inline comments (counted by `pull_request_review_id`)
    and the number of open issues its body lists, parsed from the body's own "N open issues"
    line, which only the newer Copilot overview format has (null for an older body).
  - `copilot_review_number` is the review's place among the PR's Copilot reviews on GitHub
    (1-based, by submission time), which does not depend on which transcript saw it. `round`
    is the reader's own number within ONE transcript, so a second session starts again at 1.
  - The repository is stored on every row but is not part of a key (the reader does not
    check it either), so group by it: a PR number alone is not unique across repositories.
    A key holds a review id, which is, and a call time, so rows of two repositories do not
    collide in practice. A row written by a run that could only name the project keeps that
    value if a later run could name the repository (the store is append-only).
  - The metric is Claude's cost only. Copilot's own cost has no field on a review and is not
    recorded.

Standard library plus `metrics_lib`, `pr_round_cost_lib` and `transcript_context_lib` (only to
read the `cwd` field for the project-name fallback; the rounds are the reader's); the `gh`
call is injectable so the tests need no network.
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Callable, Optional

import metrics_lib
import pr_round_cost_lib
import savings_ledger
import transcript_context_lib

METRIC_ID = "pr-review-rounds"
EVENT_ROUND = "pr_round"
EVENT_FETCH = "pr_feedback_fetch"
SCHEMA_VERSION = 2   # 2: repo, repo_source and local_reviewer_used added (additive)

# What the metadata of both event types carries from the reader's window figures.
_WINDOW_FIELDS = ("messages", "tool_calls", "wall_clock_s", "model", "input_tokens", "cache_creation_tokens",
                  "cache_read_tokens", "output_tokens", "by_model")

_COPILOT_LOGINS = ("copilot-pull-request-reviewer", "copilot")
_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_OPEN_ISSUES_RE = re.compile(r"<strong>(\d+) open issues?</strong>")
_EFFORT_RE = re.compile(r"Review effort:\s*([A-Za-z][\w -]*?)\s*</sub>")

GH_TIMEOUT_SECONDS = 60

REPO_SOURCE_REMOTE = "remote"
REPO_SOURCE_PROJECT = "project"

# `gh` runner: the arguments after `gh`, the stdout back; raises GhError on any failure.
GhRunner = Callable[[list], str]


class GhError(Exception):
    """A `gh` call that failed, for any reason; the message says why, briefly."""


def run_gh(args: list) -> str:
    """Run `gh <args>` (no shell) and return stdout. Every failure is a GhError: a missing
    binary, a timeout, a non-zero exit."""
    try:
        done = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=GH_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise GhError(f"gh could not run: {type(e).__name__}") from e
    if done.returncode != 0:
        raise GhError(f"gh exited {done.returncode}: {done.stderr.strip()[:200]}")
    return done.stdout


def valid_repo(repo) -> bool:
    """Whether `repo` looks like `owner/name`; it goes into a gh API path, so nothing else is passed on."""
    return isinstance(repo, str) and _REPO_RE.match(repo) is not None


def transcript_cwd(transcript_path) -> Optional[str]:
    """The `cwd` of the first transcript entry that records one: the directory the session
    started in (a later entry can differ after a `cd`). None when the file has none or cannot
    be read. Only this one field is read, as data: nothing in the transcript is followed."""
    try:
        for entry in transcript_context_lib.iter_entries(transcript_path):
            cwd = entry.get("cwd") if isinstance(entry, dict) else None
            if isinstance(cwd, str) and cwd.strip():
                return cwd
    except Exception:  # noqa: BLE001 -- the transcript format is internal; a passive recorder fails open
        return None
    return None


def _gh_remote_repo(gh: GhRunner) -> Optional[str]:
    """`owner/name` of the checkout `gh` runs in, or None. Not a GitHub question the row is
    short of an answer to: a failure here only means the project-name fallback is used, so it
    is not added to `github_errors` (a `gh` that is down shows up as the errors of the calls
    that need it)."""
    try:
        out = gh(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])
    except Exception:  # noqa: BLE001 -- GhError, or any fault in an injected runner
        return None
    name = out.strip() if isinstance(out, str) else ""
    return name if valid_repo(name) else None


def repo_fields(transcript_path, gh: GhRunner, repo: Optional[str] = None) -> dict:
    """`{"repo", "repo_source"}` for the events of one transcript: the repository as
    `owner/name` with source `remote` (the caller's `repo`, else the checkout's remote through
    `gh`), else the project name from the transcript's `cwd` with source `project`, else both
    None. See the module docstring."""
    name = repo if repo else _gh_remote_repo(gh)
    if name:
        return {"repo": name, "repo_source": REPO_SOURCE_REMOTE}
    cwd = transcript_cwd(transcript_path)
    project = savings_ledger.project_name_from_cwd(cwd) if cwd else ""
    if project:
        return {"repo": project, "repo_source": REPO_SOURCE_PROJECT}
    return {"repo": None, "repo_source": None}


def reviewer_used_flags(rounds: list) -> dict:
    """Round number -> whether the codebase-reviewer was used while that Copilot round's
    feedback was answered, for every Copilot round in `rounds` (the reader's, in time order).
    True when a `local` round starts strictly after that Copilot round began and strictly
    before the next Copilot round begins. A `local` round before the first Copilot round
    belongs to none. A tie belongs to none either: parallel tool calls issued in one response
    share a timestamp, so a `review` call tied with a Copilot round's first fetch is neither
    after it nor before it, and where it sits in the transcript says nothing about which came
    first, so it sets no flag rather than a flag that depends on that order."""
    flags: dict = {}
    copilot = [(r["number"], r["start_ts"]) for r in rounds if r["kind"] == pr_round_cost_lib.KIND_COPILOT]
    for number, _ in copilot:
        flags[number] = False
    for rnd in rounds:
        if rnd["kind"] == pr_round_cost_lib.KIND_COPILOT:
            continue
        when = rnd["start_ts"]
        if any(start == when for _, start in copilot):
            continue
        before = [number for number, start in copilot if start < when]
        if before:
            flags[before[-1]] = True
    return flags


def _epoch_to_iso(ts) -> Optional[str]:
    if not isinstance(ts, (int, float)) or isinstance(ts, bool):
        return None
    try:
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


def _parse_iso(s) -> Optional[float]:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).timestamp()


def _is_copilot_review(review: dict) -> bool:
    if review.get("is_copilot_code_review") is True:
        return True
    user = review.get("user")
    login = user.get("login") if isinstance(user, dict) else None
    if not isinstance(login, str):
        return False
    name = login.lower()
    return (name[:-5] if name.endswith("[bot]") else name) in _COPILOT_LOGINS


# --------------------------------------------------------------------------
# GitHub, read lazily, each source once, every failure remembered not raised
# --------------------------------------------------------------------------

class GitHubFacts:
    """What GitHub says about one PR, fetched on first use and cached (a failure is cached
    too, so one outage is one error, not one per round). Every getter returns None when its
    source could not be read; the reasons collect in `errors`."""

    def __init__(self, pr: int, gh: GhRunner = run_gh, repo: Optional[str] = None) -> None:
        if repo is not None and not valid_repo(repo):
            raise ValueError(f"repo must look like owner/name, got {repo!r}")
        self.pr = pr
        self._gh = gh
        # `{owner}/{repo}` is gh's own placeholder, filled from the current checkout's remote.
        self._prefix = f"repos/{repo}" if repo else "repos/{owner}/{repo}"
        self._repo_flag = ["-R", repo] if repo else []
        self.errors: list = []
        self._cache: dict = {}

    def _once(self, name: str, load: Callable[[], object]):
        if name not in self._cache:
            try:
                self._cache[name] = load()
            except (GhError, ValueError, TypeError, KeyError) as e:
                self.errors.append(f"{name}: {str(e)[:200]}")
                self._cache[name] = None
        return self._cache[name]

    def _lines(self, args: list) -> list:
        """One JSON value per line, which is what `gh api --paginate --jq '<program>'` prints
        (a `--paginate` without `--jq` would print one array per page, concatenated)."""
        out = []
        for line in self._gh(args).splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def state(self) -> str:
        """`ok` when no read failed, `unavailable` when the review list itself failed (nothing
        about a review could be filled), else `partial`."""
        if not self.errors:
            return "ok"
        return "unavailable" if "reviews" in self._cache and self._cache["reviews"] is None else "partial"

    def reviews(self) -> Optional[list]:
        def load() -> list:
            found = self._lines(["api", f"{self._prefix}/pulls/{self.pr}/reviews", "--paginate", "--jq", ".[]"])
            return [r for r in found if isinstance(r, dict)]
        return self._once("reviews", load)

    def inline_counts(self) -> Optional[dict]:
        """Inline review comments per review id (as a string), over every comment on the PR."""
        def load() -> dict:
            counts: dict = {}
            found = self._lines(["api", f"{self._prefix}/pulls/{self.pr}/comments", "--paginate",
                                 "--jq", ".[] | {r: .pull_request_review_id}"])
            for item in found:
                rid = item.get("r") if isinstance(item, dict) else None
                if rid is not None:
                    counts[str(rid)] = counts.get(str(rid), 0) + 1
            return counts
        return self._once("inline_comments", load)

    def size(self) -> Optional[dict]:
        def load() -> dict:
            data = json.loads(self._gh(["pr", "view", str(self.pr), *self._repo_flag,
                                        "--json", "additions,deletions,changedFiles"]))
            return {"additions": data.get("additions"), "deletions": data.get("deletions"),
                    "changed_files": data.get("changedFiles")}
        return self._once("pr_size", load)

    def push_time(self, sha) -> Optional[float]:
        """Epoch seconds of the earliest check suite GitHub created for commit `sha`."""
        if not isinstance(sha, str) or not _SHA_RE.match(sha):
            return None

        def load() -> Optional[float]:
            stamps = [_parse_iso(line.strip().strip('"')) for line in self._gh(
                ["api", f"{self._prefix}/commits/{sha}/check-suites", "--jq", ".check_suites[].created_at"]).splitlines()
                if line.strip()]
            known = [t for t in stamps if t is not None]
            return min(known) if known else None
        return self._once(f"check_suites {sha[:7]}", load)


def _find_review(reviews: list, ids: set) -> Optional[dict]:
    """The GitHub review listed under any of `ids` (its numeric id or its `PRR_` node id)."""
    return next((r for r in reviews if str(r.get("id")) in ids or r.get("node_id") in ids), None)


def _review_facts(rnd: dict, gh: GitHubFacts) -> dict:
    """The Copilot-only fields of one round, from GitHub. All null for anything not found."""
    facts: dict = {"copilot_review_number": None, "effort": None, "inline_comments": None,
                   "body_open_issues": None, "had_findings": None, "github_commit_id": None,
                   "push_time": None, "wait_from_push_s": None, "github_review_id": None,
                   "github_submitted_at": None, "github_state": None}
    if rnd["kind"] != pr_round_cost_lib.KIND_COPILOT:
        return facts
    reviews = gh.reviews()
    if reviews is None:
        return facts
    ids = {str(i) for i in [rnd.get("review_id"), *rnd.get("review_ids", [])] if i}
    match = _find_review(reviews, ids)
    if match is None:
        return facts
    # What GitHub says about the review itself, for a round whose transcript output showed
    # nothing of it (a marker-only fetch has the id alone). The transcript's own value wins
    # when it has one; the idempotency key holds neither, so filling them in changes no key.
    facts["github_submitted_at"] = match.get("submitted_at") if isinstance(match.get("submitted_at"), str) else None
    facts["github_state"] = match.get("state") if isinstance(match.get("state"), str) else None
    copilot = sorted((r for r in reviews if _is_copilot_review(r)),
                     key=lambda r: (str(r.get("submitted_at") or ""), str(r.get("id"))))
    facts["github_review_id"] = str(match.get("id")) if match.get("id") is not None else None
    facts["copilot_review_number"] = next((i + 1 for i, r in enumerate(copilot) if r is match), None)
    raw_body = match.get("body")
    body: str = raw_body if isinstance(raw_body, str) else ""
    effort = match.get("copilot_code_review_effort")
    if not (isinstance(effort, str) and effort):
        m = _EFFORT_RE.search(body)
        effort = m.group(1).lower() if m else None
    facts["effort"] = effort
    m = _OPEN_ISSUES_RE.search(body)
    facts["body_open_issues"] = int(m.group(1)) if m else (0 if "No open issues" in body else None)
    counts = gh.inline_counts()
    inline = None if counts is None else counts.get(str(match.get("id")), 0)
    facts["inline_comments"] = inline
    if (inline or 0) > 0 or (facts["body_open_issues"] or 0) > 0:
        facts["had_findings"] = True
    elif inline == 0 or facts["body_open_issues"] == 0:
        facts["had_findings"] = False
    commit = match.get("commit_id") if isinstance(match.get("commit_id"), str) else None
    facts["github_commit_id"] = commit
    pushed = gh.push_time(commit or rnd.get("commit_id"))
    facts["push_time"] = _epoch_to_iso(pushed)
    # The same submission time the row stores (the transcript's, else GitHub's), so the two
    # stored figures cannot disagree.
    submitted = _parse_iso(rnd.get("submitted_at")) or _parse_iso(match.get("submitted_at"))
    if pushed is not None and submitted is not None and submitted >= pushed:
        facts["wait_from_push_s"] = round(submitted - pushed)
    return facts


# --------------------------------------------------------------------------
# Building the events
# --------------------------------------------------------------------------

def _window(rec: dict) -> dict:
    return {k: rec.get(k) for k in _WINDOW_FIELDS}


def _merge_given(merge_time) -> bool:
    """Whether `merge_time` is one the reader accepts (it ends the last window there); anything
    else it ignores, and the last window then runs to the transcript's last line."""
    if isinstance(merge_time, bool):
        return False
    if isinstance(merge_time, (int, float)):
        return True
    return _parse_iso(merge_time) is not None


def build_events(found: dict, gh: GitHubFacts, merge_given: bool = False, repo_info: Optional[dict] = None) -> list:
    """The events for one `find_rounds` result: a list of
    `{"event_type", "key", "metadata", "event_timestamp", "session_id"}`, rounds first, then
    fetch calls. Reads GitHub through `gh`; writes nothing. A review's `submitted_at` and
    `review_state` come from the transcript when its output showed them, else from GitHub (a
    marker-only fetch names the review id and nothing more), else stay null. A window is
    `window_open` when it ends at the transcript's last line because no merge time closed it:
    a session resumed later can add work to it, which a re-run would not record (see the
    module docstring); every other window is bounded by the next round, the next fetch call
    or the merge, and cannot grow. `repo_info` is `repo_fields`' result, stamped on every event
    (null `repo` and `repo_source` when omitted)."""
    pr = found["pr"]
    where = {"repo": None, "repo_source": None, **(repo_info or {})}
    reviewer_used = reviewer_used_flags(found["rounds"])
    session = found.get("session_id")
    size = gh.size()
    last_line = found.get("end_ts")

    def is_open(end_ts) -> bool:
        return not merge_given and last_line is not None and end_ts == last_line

    size_fields = {"pr_additions": size["additions"] if size else None,
                   "pr_deletions": size["deletions"] if size else None,
                   "pr_changed_files": size["changed_files"] if size else None}
    events: list = []
    round_by_number: dict = {}
    for rnd in found["rounds"]:
        facts = _review_facts(rnd, gh)
        copilot = rnd["kind"] == pr_round_cost_lib.KIND_COPILOT
        review_id = rnd.get("review_id") if copilot else None
        meta = {
            "schema": SCHEMA_VERSION, "pr": pr, **where, "kind": rnd["kind"], "round": rnd["number"],
            "kind_number": rnd["kind_number"], "copilot_review_number": facts["copilot_review_number"],
            "review_id": review_id, "github_review_id": facts["github_review_id"], "review_ids": list(rnd.get("review_ids", [])) if copilot else [],
            "submitted_at": (rnd.get("submitted_at") or facts["github_submitted_at"]) if copilot else None,
            "commit_id": (facts["github_commit_id"] or rnd.get("commit_id")) if copilot else None,
            "review_state": (rnd.get("review_state") or facts["github_state"]) if copilot else None,
            "effort": facts["effort"], "inline_comments": facts["inline_comments"],
            "body_open_issues": facts["body_open_issues"], "had_findings": facts["had_findings"],
            "call_time": rnd.get("start"), "window_end": _epoch_to_iso(rnd.get("end_ts")),
            "window_open": is_open(rnd.get("end_ts")),
            "push_time": facts["push_time"], "wait_from_push_s": facts["wait_from_push_s"],
            "fetch_calls": len(rnd.get("fetch_calls", [])),
            "review_type": rnd.get("review_type") if not copilot else None,
            "review_pr": rnd.get("pr") if not copilot else None,
            "review_range": rnd.get("range") if not copilot else None,
            "result_chars": rnd.get("result_chars") if not copilot else None,
            **size_fields, **_window(rnd), "session_id": session,
            "github": None,  # stamped by _stamp_github_state once every question has been asked
        }
        if copilot:
            # A local round is not given the flag: it is the reviewer's own cost, not a round
            # that was answered with or without it.
            meta["local_reviewer_used"] = reviewer_used[rnd["number"]]
        # A local round has no review id, and two `review` calls issued in one assistant
        # response share a timestamp, so (pr, kind, None, call_time) would drop the second as
        # "already recorded" together with its tokens. Its place among the local rounds of its
        # session tells them apart, and stays the same on every re-read of the same transcript.
        key_fields = ("pr", "kind", "review_id", "call_time") + (() if copilot else ("kind_number", "session_id"))
        events.append({"event_type": EVENT_ROUND, "metadata": meta, "session_id": session,
                       "key": {k: meta[k] for k in key_fields},
                       "event_timestamp": _epoch_to_iso(rnd.get("start_ts"))})
        round_by_number[rnd["number"]] = rnd
    fetches = [f for rnd in found["rounds"] for f in rnd["fetch_calls"]] + list(found["orphan_fetch_calls"])
    for call in sorted(fetches, key=lambda f: f["ts"]):
        rnd = round_by_number.get(call["round"])
        submitted = call.get("submitted_at")
        if submitted is None and call.get("review_id"):
            # A marker-only call shows the review id and nothing else; GitHub knows the time.
            reviews = gh.reviews()
            listed = _find_review(reviews, {str(call["review_id"])}) if reviews else None
            if listed is not None and isinstance(listed.get("submitted_at"), str):
                submitted = listed["submitted_at"]
        meta = {
            "schema": SCHEMA_VERSION, "pr": pr, **where, "round": call["round"],
            "round_kind": rnd["kind"] if rnd else None, "tool_use_id": call["tool_use_id"],
            "review_id": call.get("review_id"), "submitted_at": submitted,
            "call_time": call["time"], "window_end": _epoch_to_iso(call.get("end_ts")),
            "window_open": is_open(call.get("end_ts")),
            "sources": list(call["sources"]), "recognised": call["recognised"],
            **size_fields, **_window(call), "session_id": session,
        }
        events.append({"event_type": EVENT_FETCH, "metadata": meta, "session_id": session,
                       "key": {k: meta[k] for k in ("pr", "review_id", "call_time", "tool_use_id")},
                       "event_timestamp": _epoch_to_iso(call["ts"])})
    return events


def _stamp_github_state(events: list, gh: GitHubFacts) -> None:
    """Say on every pr_round row whether GitHub answered every question it asked: `ok` (no
    failure), `partial` (some) or `unavailable` (the review list itself could not be read)."""
    state = gh.state()
    for ev in events:
        if ev["event_type"] == EVENT_ROUND:
            ev["metadata"]["github"] = state


def record_pr_rounds(transcript_path, pr: int, *, repo: Optional[str] = None, merge_time=None,
                     store: Optional[metrics_lib.MetricsStore] = None, gh: GhRunner = run_gh,
                     dry_run: bool = False) -> dict:
    """Find PR `pr`'s review rounds in the transcript at `transcript_path` and record them.

    `merge_time` is passed to the reader (it ends the last window). `repo` is `owner/name` for
    the `gh` calls; omitted, `gh` takes it from the current directory's checkout. `store`
    defaults to the shared `metrics.db`. `dry_run` builds the events and writes nothing
    (they are returned under `events`).

    Returns `{"ok", "pr", "session_id", "rounds", "fetch_calls", "fetch_events", "written",
    "existing", "failed", "not_attempted", "github_errors", "store_error", "skipped_fetches",
    "events", "repo", "repo_source"}`; `fetch_calls` and `skipped_fetches` are the reader's counts of all fetch calls
    and of those it did not read, `fetch_events` how many `pr_feedback_fetch` events were built
    (a call it could not place or whose command it did not read has none); `ok` is False only
    when the transcript could not be read. Never raises."""
    summary: dict = {"ok": False, "pr": pr, "session_id": None, "rounds": 0, "fetch_calls": 0,
                     "fetch_events": 0, "written": 0, "existing": 0, "failed": 0, "not_attempted": 0,
                     "github_errors": [], "store_error": None, "skipped_fetches": 0, "events": [],
                     "repo": None, "repo_source": None}
    if repo is not None and not valid_repo(repo):
        summary["store_error"] = f"repo must look like owner/name, got {repo!r}; nothing was read or written"
        return summary
    try:
        found = pr_round_cost_lib.find_rounds(transcript_path, pr, merge_time=merge_time)
        summary["ok"] = bool(found.get("ok"))
        summary["session_id"] = found.get("session_id")
        if not found.get("ok"):
            return summary
        summary["rounds"] = len(found["rounds"])
        summary["skipped_fetches"] = found.get("skipped_fetches", 0)
        facts = GitHubFacts(pr, gh, repo)
        # A transcript with nothing to record asks nobody anything, `gh repo view` included.
        where = (repo_fields(transcript_path, gh, repo) if found["rounds"] or found["orphan_fetch_calls"]
                 else {"repo": None, "repo_source": None})
        summary["repo"], summary["repo_source"] = where["repo"], where["repo_source"]
        events = build_events(found, facts, _merge_given(merge_time), where)
        _stamp_github_state(events, facts)
        summary["github_errors"] = list(facts.errors)
        # The reader's own total, so `fetch_calls` and `skipped_fetches` describe one population
        # (a wrapped command it counted but did not read has no event, and is in both).
        summary["fetch_calls"] = found.get("fetch_calls", 0)
        summary["fetch_events"] = sum(1 for e in events if e["event_type"] == EVENT_FETCH)
        if dry_run:
            summary["events"] = events
            return summary
        target = store if store is not None else metrics_lib.MetricsStore()
        for i, ev in enumerate(events):
            outcome = target.record_if_absent(METRIC_ID, ev["event_type"], ev["key"], 1.0, ev["metadata"],
                                              ev["session_id"], ev["event_timestamp"])
            if outcome == "written":
                summary["written"] += 1
            elif outcome == "exists":
                summary["existing"] += 1
            else:
                # The store is the likely cause (locked, unreadable, unwritable); the reason is
                # on stderr. Stop rather than log the same failure once per event.
                summary["failed"] += 1
                summary["not_attempted"] = len(events) - i - 1
                summary["store_error"] = f"{ev['event_type']} could not be written; stopped (see stderr)"
                break
    except Exception as e:  # noqa: BLE001 -- broad on purpose: a passive recorder must never break its caller
        print(f"[claude-runway] pr round recording failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        summary["store_error"] = summary["store_error"] or f"unexpected {type(e).__name__}"
    return summary
