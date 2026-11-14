#!/usr/bin/env python3
"""
Backfills the shared metrics store's `pr-review-rounds` events (`libs/pr_round_record_lib.py`)
from the session transcripts that still exist for a project. Live capture only adds data from
now on; the baseline (what Claude spent per review round before the codebase-reviewer) is in
transcripts that Claude Code deletes after a retention period, so this reads them while they
are there.

WHAT IT DOES
  1. Lists the project's MAIN-session transcripts: the `<session id>.jsonl` files directly in
     the project folder. Subagent transcripts live below `<session id>/subagents/` and are not
     read (out of scope for the epic).
  2. DISCOVERS which pull requests each transcript fetched feedback for. Neither reader takes
     "any PR": `libs/pr_round_cost_lib.py`'s `find_rounds` needs the number. A candidate comes
     from the command text (`pulls/<N>/reviews|comments`, `issues/<N>/comments`,
     `gh pr view <N>`), from a `pr-feedback-fetch: pr=<N>` line in a Bash result, or from a
     `review` call's `params.pr`. A candidate is only a lead: it can be a false one (a PR number
     in a command that merely mentions it), and `find_rounds` decides whether the transcript
     holds anything for it. A pair is kept only with PR-specific evidence, a feedback fetch the
     reader recognised or a `review` call naming the PR (`_has_evidence`): the reader attributes
     a `review` call that names no PR to every PR it is asked about, which would otherwise
     record that review against any PR whose number merely appears in some command.
  3. For each (transcript, PR) with something readable, looks up the PR on GitHub for the time
     that ends the last window, then records through `record_pr_rounds` (used as it is, never
     edited), which is idempotent: a second run, or a run after the session-end capture, writes
     nothing twice.
  Transcripts are untrusted data and are only ever opened for reading, never modified, moved
  or deleted; nothing in one is followed or executed.

DRY RUN BY DEFAULT
  `backfill(..., write=False)` builds every event (it reads GitHub, which is read-only) and
  writes nothing; the store is not even opened. `write=True` records.

DECISIONS
  - A PR that is still OPEN is skipped unless `include_open` is set. A window that ends at a
    transcript's last line with no end time is stored with `window_open` true and is never
    corrected (a re-run finds the key and writes nothing), so recording a PR before it is
    merged would leave a permanently short last round. Skipping it costs nothing: the next
    backfill, after the merge, records it with the merge time. A PR CLOSED without merging
    ends at its closing time, which is final. `include_open` records the open PR as it stands,
    `window_open` true, for someone who prefers a partial row to none.
  - The end time is clamped to the transcript's own last line. A PR merged days after its
    session ended must not stretch that session's last window (and its wall clock) across the
    gap; with the clamp the window ends where the session's work did, and the row is not
    `window_open` because the session is over.
  - A lookup that fails (no such PR in this repository, `gh` missing or signed out, a rate
    limit) skips that transcript's PR and says so: nothing is recorded without its end time.
  - Several transcripts naming one PR are each recorded (a PR's rounds can span sessions).
    That is correct while each session id has its own history. If Claude Code ever wrote a
    resumed session as a NEW file carrying the old history under a new session id, a local
    round (its key holds `session_id`) would count twice while a Copilot round (its key does
    not) would not. None of the transcripts checked so far does this; the report lists any two
    transcripts that share a Bash tool-use id, which a copied history would, so a dry run shows
    it before anything is written.

KNOWN LIMITS
  - The repository is not checked: a PR number from another repository fetched in this
    project's sessions is looked up as this repository's PR with that number (a miss is
    reported and skipped; a hit of the same number is recorded as this one's, as the readers do).
  - A candidate needs a command or marker the discovery recognises; a session that fetched a
    PR's feedback in another way is not found. What the readers cannot read is counted in the
    report as unrecognised fetches, never guessed at.
  - Only a transcript's file name decides it is a main session; a stray `*.jsonl` that is not
    one reads as a transcript with nothing in it.

Standard library plus `pr_round_record_lib`, `pr_round_cost_lib`, `transcript_context_lib`,
`autowork_cost_lib` (only its projects-folder layout) and `metrics_lib`; the `gh` call is
injectable so the tests need no network.
"""

import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import autowork_cost_lib
import metrics_lib
import pr_round_cost_lib
import pr_round_record_lib as P
import transcript_context_lib

# Where a PR number shows in a Bash command the reviews/comments fetches use.
# At most 9 digits: a transcript is untrusted, and `int()` of a longer run of digits can raise.
_CMD_PR_RE = re.compile(r"/(?:pulls|issues)/(\d{1,9})/(?:reviews|comments)\b|\bgh[ \t]+pr[ \t]+view[ \t]+#?(\d{1,9})\b")
_MARKER_PR_RE = re.compile(r"pr-feedback-fetch:[ \t]*pr=(\d{1,9})\b")
_MAX_COMMAND_CHARS = 20000   # a longer command is not one of the shapes any reader knows


def default_projects_root() -> str:
    return autowork_cost_lib.default_projects_root()


def project_folder_name(project_path) -> str:
    """The folder name Claude Code gives a project under `~/.claude/projects`: the project's
    absolute path with every character that is not a letter or digit turned into a hyphen
    (`/Users/me/repos/my_repo` -> `-Users-me-repos-my-repo`)."""
    return re.sub(r"[^A-Za-z0-9]", "-", os.path.abspath(str(project_path)))


def default_project_dir(cwd=None, projects_root=None) -> str:
    """The transcripts folder of the project whose checkout contains `cwd`. In a linked git
    worktree that is the MAIN checkout's folder (a worktree is its own project to Claude
    Code, and its sessions are not the project's own), found through git's common directory;
    anywhere else, `cwd` itself."""
    here = os.path.abspath(str(cwd)) if cwd else os.getcwd()
    root = here
    try:
        done = subprocess.run(["git", "-C", here, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                              capture_output=True, text=True, timeout=30, check=False)
        common = done.stdout.strip()
        if done.returncode == 0 and common and os.path.basename(common) == ".git":
            root = os.path.dirname(common)
    except (OSError, subprocess.SubprocessError):
        pass
    return os.path.join(str(projects_root) if projects_root else default_projects_root(), project_folder_name(root))


def main_transcripts(project_dir) -> list:
    """Every main-session transcript of a project folder, sorted by name. Subagent
    transcripts are in folders below it and are not matched."""
    return sorted(p for p in glob.glob(os.path.join(glob.escape(str(project_dir)), "*.jsonl")) if os.path.isfile(p))


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))
    return ""


def scan_transcript(path) -> dict:
    """One read-only pass over a transcript for the candidate PRs, its last timestamp and its
    Bash tool-use ids. `{"ok", "prs", "last_ts", "bash_ids"}`; `ok` is False only when the
    file cannot be opened (a garbled line is skipped by the shared reader, like everywhere
    else, so a wholly garbled file is an empty transcript, not an unreadable one)."""
    out: dict = {"ok": False, "prs": set(), "last_ts": None, "bash_ids": set()}
    try:
        with open(str(path), "rb"):
            pass
    except OSError:
        return out
    out["ok"] = True
    bash_ids: set = out["bash_ids"]
    prs: set = out["prs"]
    for obj in transcript_context_lib.iter_entries(path):
        ts = _epoch(obj.get("timestamp"))
        if ts is not None:
            out["last_ts"] = ts if out["last_ts"] is None else max(out["last_ts"], ts)
        message = obj.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                name = block.get("name")
                raw_input = block.get("input")
                tin: dict = raw_input if isinstance(raw_input, dict) else {}
                if name == "Bash":
                    bash_ids.add(block["id"])
                    command = tin.get("command")
                    if isinstance(command, str) and len(command) <= _MAX_COMMAND_CHARS:
                        for m in _CMD_PR_RE.finditer(command):
                            prs.add(int(m.group(1) or m.group(2)))
                elif name == pr_round_cost_lib.REVIEW_TOOL:
                    named = _review_call_pr(tin)
                    if named:
                        prs.add(named)
            elif (block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str)
                  and block["tool_use_id"] in bash_ids):
                text = _text_of(block.get("content"))
                if "pr-feedback-fetch:" in text:
                    prs.update(int(m.group(1)) for m in _MARKER_PR_RE.finditer(text))
    prs.discard(0)
    return out


class _MemoGh:
    """Wraps a `gh` runner so one run asks GitHub each question once (a PR's reviews, comments,
    size and check suites do not change during a backfill, and the same PR often turns up in
    several transcripts). A failure is remembered too, so one outage is one attempt."""

    def __init__(self, gh) -> None:
        self._gh = gh
        self._cache: dict = {}

    def __call__(self, args: list) -> str:
        key = tuple(args)
        if key not in self._cache:
            try:
                self._cache[key] = (True, self._gh(args))
            except P.GhError as e:
                self._cache[key] = (False, str(e))
        ok, value = self._cache[key]
        if not ok:
            raise P.GhError(value)
        return value


def _iso(ts) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(text) -> Optional[float]:
    """Epoch seconds of an ISO 8601 text (a naive one is UTC), None for anything else."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).timestamp()


def _review_call_pr(tool_input: dict) -> Optional[int]:
    """The PR a `review` call names in `params.pr` (an int or a digit string, `#` allowed)."""
    params = tool_input.get("params")
    raw = params.get("pr") if isinstance(params, dict) else None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if 0 < raw < 10 ** 9 else None
    # ASCII digits only: `str.isdigit` also accepts characters `int()` then rejects.
    if isinstance(raw, str) and re.fullmatch(r"#?[0-9]{1,9}", raw.strip()):
        return int(raw.strip().lstrip("#"))
    return None


def lookup_pr(pr: int, gh, repo: Optional[str] = None) -> dict:
    """What ends PR `pr`'s last window: `{"ok", "state", "end", "error"}`. `end` is the epoch
    seconds of the merge (MERGED) or the closing (CLOSED), None while OPEN. `ok` is False when
    the PR could not be looked up or the answer was not understood; `error` says why."""
    args = ["pr", "view", str(pr), *(["-R", repo] if repo else []), "--json", "state,mergedAt,closedAt"]
    try:
        data = json.loads(gh(args))
    except P.GhError as e:
        return {"ok": False, "state": None, "end": None, "error": str(e)[:200]}
    except (ValueError, TypeError):
        return {"ok": False, "state": None, "end": None, "error": "gh printed something that is not JSON"}
    state = data.get("state") if isinstance(data, dict) else None
    if state not in ("OPEN", "MERGED", "CLOSED"):
        return {"ok": False, "state": None, "end": None, "error": f"unexpected state {state!r}"}
    when = data.get("mergedAt") if state == "MERGED" else data.get("closedAt") if state == "CLOSED" else None
    end = _epoch(when)
    if state != "OPEN" and end is None:
        return {"ok": False, "state": state, "end": None, "error": f"{state} PR with no usable time"}
    return {"ok": True, "state": state, "end": end, "error": None}


def _new_report(project_dir) -> dict:
    return {
        "project_dir": str(project_dir), "write": False,
        "transcripts_found": 0, "transcripts_read": 0, "transcripts_unreadable": [],
        "transcripts_with_feedback": 0, "transcripts_without_feedback": 0,
        "prs_found": 0, "prs_recorded": 0, "candidates_without_fetch": 0, "prs": {},
        "pairs": [], "skipped": [],
        "rounds": 0, "fetch_events": 0, "events_built": 0, "unrecognised_fetches": 0,
        "written": 0, "existing": 0, "failed": 0, "not_attempted": 0,
        "github_errors": [], "store_error": None, "shared_tool_use_ids": [],
    }


def _shared_tool_use_ids(by_name: dict) -> list:
    """Pairs of transcripts that carry the same Bash tool-use id, which a session written as a
    new file with copied history would: `[{"a", "b", "ids"}]`."""
    names = sorted(by_name)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = by_name[a] & by_name[b]
            if common:
                out.append({"a": a, "b": b, "ids": len(common)})
    return out


def backfill(project_dir, *, repo: Optional[str] = None, store: Optional[metrics_lib.MetricsStore] = None,
             gh=P.run_gh, write: bool = False, include_open: bool = False) -> dict:
    """Backfill the transcripts directly in `project_dir`. `write` False (the default) is a
    dry run: everything is read and built, nothing is written, and `store` is not touched.
    Returns the report: counts of what was read, found, skipped (with each reason under
    `skipped`) and recorded, one entry per recorded PR under `prs`, and the per-pair
    summaries under `pairs`. Never raises."""
    report = _new_report(project_dir)
    report["write"] = write
    try:
        _backfill(report, project_dir, repo, store, _MemoGh(gh), write, include_open)
    except Exception as e:  # noqa: BLE001 -- a backfill must report, never traceback halfway through a baseline
        print(f"[claude-runway] pr round backfill failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        report["store_error"] = report["store_error"] or f"unexpected {type(e).__name__}"
    return report


def _has_evidence(found: dict, pr: int) -> bool:
    """Whether a transcript holds something specific to PR `pr`: a feedback fetch of it that the
    reader recognised (every such call is matched to this PR by its command or its marker), or a
    local `review` call that names it. A `review` call naming NO PR is not evidence: the reader
    attributes it to every PR it is asked about, so a number that merely appears in some other
    command (a `gh pr view <N> --json state`) would otherwise record that review's tokens against
    an unrelated PR once it is merged, and the same work again under each such number. (A PR
    that does have a fetch still gets such unnamed local rounds, which is the reader's contract.)"""
    if found["orphan_fetch_calls"] or any(r["fetch_calls"] for r in found["rounds"]):
        return True
    return any(r["kind"] == pr_round_cost_lib.KIND_LOCAL and r.get("pr") == pr for r in found["rounds"])


def _skip(report: dict, transcript: str, pr: Optional[int], reason: str) -> None:
    report["skipped"].append({"transcript": transcript, "pr": pr, "reason": reason})


def _backfill(report: dict, project_dir, repo, store, gh, write: bool, include_open: bool) -> None:
    if repo is not None and not P.valid_repo(repo):
        report["store_error"] = f"repo must look like owner/name, got {repo!r}; nothing was read or written"
        return
    paths = main_transcripts(project_dir)
    report["transcripts_found"] = len(paths)
    target = (store if store is not None else metrics_lib.MetricsStore()) if write else None
    lookups: dict = {}
    bash_by_name: dict = {}
    pr_seen: set = set()
    recorded_prs: set = set()
    for path in paths:
        name = os.path.basename(path)
        try:
            scanned = scan_transcript(path)
        except Exception:  # noqa: BLE001 -- an odd line in one untrusted transcript must not end the run
            scanned = {"ok": False, "prs": set(), "last_ts": None, "bash_ids": set()}
        if not scanned["ok"]:
            report["transcripts_unreadable"].append(name)
            _skip(report, name, None, "the transcript could not be opened")
            continue
        report["transcripts_read"] += 1
        bash_by_name[name] = scanned["bash_ids"]
        had_feedback = False
        for pr in sorted(scanned["prs"]):
            found = pr_round_cost_lib.find_rounds(path, pr)
            if not found.get("ok"):
                _skip(report, name, pr, "the transcript could not be read for this PR")
                continue
            # The reader's own count of fetch calls it did not read (an output shape it does not
            # recognise, a command off its list): counted once per transcript and PR, whatever
            # becomes of the pair below, so the figure does not depend on how the run went.
            unread = found.get("skipped_fetches", 0)
            report["unrecognised_fetches"] += unread
            if not _has_evidence(found, pr):
                # Named in a command but never fetched in a shape either reader recognises, and no
                # review call names it: see `_has_evidence`.
                if unread:
                    _skip(report, name, pr, f"{unread} fetch call(s) the reader does not recognise, none it can read")
                else:
                    # Only a number in a command that is no fetch of this PR's feedback: nothing to say about it.
                    report["candidates_without_fetch"] += 1
                continue
            had_feedback = True
            pr_seen.add(pr)
            if report["store_error"]:
                _skip(report, name, pr, "not attempted: the store failed earlier in this run")
                continue
            if pr not in lookups:
                lookups[pr] = lookup_pr(pr, gh, repo)
            looked = lookups[pr]
            if not looked["ok"]:
                _skip(report, name, pr, f"the PR could not be looked up: {looked['error']}")
                continue
            if looked["state"] == "OPEN" and not include_open:
                _skip(report, name, pr, "the PR is still open; a window with no end time is stored as open and "
                                        "never corrected (run again after the merge, or pass include_open)")
                continue
            merge_time = None
            if looked["end"] is not None:
                last = scanned["last_ts"]
                # Never past the transcript's own last line (see the module docstring).
                # The epoch itself, not an ISO text of it: transcript timestamps carry
                # milliseconds, and cutting the clamp to the whole second would drop the last
                # message (and any call) of that second from the window, for good, since a
                # written row is never rewritten. The reader and recorder accept a number.
                merge_time = min(looked["end"], last) if last is not None else looked["end"]
            summary = P.record_pr_rounds(path, pr, repo=repo, merge_time=merge_time, store=target, gh=gh,
                                         dry_run=not write)
            if _absorb(report, name, pr, looked, summary, write):
                recorded_prs.add(pr)
        if had_feedback:
            report["transcripts_with_feedback"] += 1
        else:
            report["transcripts_without_feedback"] += 1
    report["prs_found"] = len(pr_seen)
    report["prs_recorded"] = len(recorded_prs)
    report["shared_tool_use_ids"] = _shared_tool_use_ids(
        {n: ids for n, ids in bash_by_name.items() if ids})


def _absorb(report: dict, name: str, pr: int, looked: dict, summary: dict, write: bool) -> bool:
    """Fold one `record_pr_rounds` summary into the report. True when the pair counts as
    recorded (or, in a dry run, as one that would be): it built events and the store did not fail."""
    if not summary["ok"]:
        _skip(report, name, pr, "the transcript could not be read for this PR")
        return False
    events = summary["events"] if not write else None
    report["rounds"] += summary["rounds"]
    report["fetch_events"] += summary["fetch_events"]
    report["events_built"] += (len(events) if events is not None else summary["written"] + summary["existing"]
                               + summary["failed"] + summary["not_attempted"])
    for key in ("written", "existing", "failed", "not_attempted"):
        report[key] += summary[key]
    for err in summary["github_errors"]:
        if err not in report["github_errors"]:
            report["github_errors"].append(err)
    if summary["store_error"]:
        report["store_error"] = summary["store_error"]
    if summary["rounds"] + summary["fetch_events"] == 0:
        # Every fetch of this PR in this transcript came after its merge or closing.
        _skip(report, name, pr, "nothing in this transcript falls before the PR's merge or closing time")
        return False
    entry = report["prs"].setdefault(pr, {"pr": pr, "state": looked["state"], "end": _iso(looked["end"]) if looked["end"] is not None else None,
                                         "transcripts": 0, "rounds": 0, "fetch_events": 0})
    entry["transcripts"] += 1
    entry["rounds"] += summary["rounds"]
    entry["fetch_events"] += summary["fetch_events"]
    report["pairs"].append({"transcript": name, "pr": pr, "session_id": summary["session_id"],
                            "rounds": summary["rounds"], "fetch_calls": summary["fetch_calls"],
                            "fetch_events": summary["fetch_events"], "written": summary["written"],
                            "existing": summary["existing"], "failed": summary["failed"],
                            "skipped_fetches": summary["skipped_fetches"]})
    return not (write and summary["store_error"])
