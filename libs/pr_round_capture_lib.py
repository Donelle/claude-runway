#!/usr/bin/env python3
"""
Captures the pull-request review rounds of ONE session transcript into the shared metrics store
(`pr-review-rounds`, `libs/pr_round_record_lib.py`) when the session ends, so the baseline of what
Claude spends per review round keeps growing with no step for anyone to remember. The backfill
(`libs/pr_round_backfill_lib.py`) records the past; this records each session as it finishes.

It is the same pipeline the backfill runs per (transcript, PR) pair, for the one transcript a
`SessionEnd` payload names, and it reuses that pipeline's parts as they are (their contracts are
fixed and nothing here edits them):
  1. `pr_round_backfill_lib.scan_transcript`: the candidate PR numbers (a lead, not a verdict),
     and the transcript's last timestamp.
  2. `pr_round_cost_lib.find_rounds`: what the transcript really holds for a candidate. A pair
     is kept only with PR-specific evidence (`pr_round_backfill_lib._has_evidence`).
  3. `pr_round_backfill_lib.lookup_pr`: the PR's state and the time that ends its last window.
  4. `pr_round_record_lib.record_pr_rounds`: the idempotent write.

HOW A ROUND IS RECOGNISED (the marker versus the whitelist)
  The reader starts a Copilot round from a feedback fetch it recognises in exactly two ways:
  the marker line `/cr-pr-feedback` prints (`pr-feedback-fetch: pr=<N> copilot-review=<id>`, from
  a Bash `echo` so it lands in a tool result), or a command matching, whole, its closed whitelist
  of exact command shapes. A session that used `/cr-pr-feedback` is therefore captured through
  the marker whatever the commands looked like. A session that fetched feedback by hand is
  captured only if its commands were one of the whitelisted shapes. A session that did NEITHER
  (a compound such as `cd <repo> && gh api ...`, a wrapped or piped fetch) records no Copilot
  round: the reader counts those calls as unrecognised and never guesses, and this module puts
  that count in the log line (`unrecognised=<n>`) so the gap is visible rather than silent. A
  local `review` call that names the PR is evidence on its own and records a `local` round.

OPEN PULL REQUESTS ARE NOT RECORDED AT SESSION END
  A window that ends at a transcript's last line with no end time is stored with `window_open`
  true and is never corrected (the recorder finds the key and writes nothing), so recording a PR
  while it is open would leave a permanently short last round. The capture therefore skips an
  open PR, as the backfill does: a later session end that finds the PR merged or closed (a
  resumed session ends again), or a backfill after the merge, records it with its end time. The
  cost is that a session which ends before the merge records nothing until one of those runs;
  `include_open` (`--include-open` on the CLI, never set by the hook) records it as it stands.
  A merged or closed PR ends at its merge or closing time, clamped to the transcript's last line
  so a merge days after the session does not stretch its last window.

SAFETY (this runs from a SessionEnd hook on every session that has it switched on)
  - `capture` never raises and always returns a report. The HOOK exits 0 on anything and prints
    nothing. The worker CLI it starts reports honestly instead (exit 1 when the transcript
    cannot be opened or the store fails, 2 for a bad argument), and the hook never looks at
    that code, so a worker failure never reaches the session end.
  - The transcript is untrusted data: it is only opened for reading, nothing in it is followed or
    executed, and no transcript text reaches the log (only counts, PR numbers, the file's base
    name and `gh`'s own short errors, with control characters stripped).
  - Only the one transcript named is read; the project folder is never listed.
  - `capture_pr_rounds.py` runs it as `python -I`, detached, so the session end is never held up
    by `gh` calls; `deadline` bounds the work between PRs.

Standard library plus `pr_round_backfill_lib`, `pr_round_cost_lib`, `pr_round_record_lib` and
`metrics_lib`; the `gh` call is injectable so the tests need no network.
"""

import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

import metrics_lib
import pr_round_backfill_lib as B
import pr_round_cost_lib
import pr_round_record_lib as P

ENABLE_ENV_VAR = "CLAUDE_RUNWAY_CAPTURE_PR_ROUNDS"
_TRUTHY = ("1", "true", "yes")
LOG_FILE_NAME = "pr-round-capture.log"
MAX_LOG_BYTES = 200_000
# Wall-clock budget for one capture, checked between PRs (each `gh` call has its own timeout).
DEFAULT_TIME_BUDGET_SECONDS = 300.0


def capture_enabled(environ=None) -> bool:
    """Whether the opt-in is on (`1`, `true` or `yes`, any case, whitespace trimmed)."""
    env = os.environ if environ is None else environ
    return str(env.get(ENABLE_ENV_VAR, "")).strip().lower() in _TRUTHY


def default_log_path(db_path=None) -> Path:
    """The log sits beside the metrics database it describes (so an override of the database
    moves it too): `<database folder>/pr-round-capture.log`."""
    base = Path(db_path) if db_path else metrics_lib.resolve_db_path()
    return base.parent / LOG_FILE_NAME


def _clean(text, limit: int = 200) -> str:
    """One line of printable ASCII, whatever went in (a session id, a `gh` error): nothing from
    a transcript can inject a line break or a control sequence into the log."""
    one_line = re.sub(r"\s+", " ", str(text))
    return re.sub(r"[^\x20-\x7e]", "?", one_line)[:limit]


LOG_LOCK_WAIT_SECONDS = 2.0


def _try_lock(fd: int) -> bool:
    """One non-blocking attempt at an exclusive OS lock on the open file `fd`."""
    try:
        if sys.platform == "win32":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


@contextmanager
def _log_lock(path: Path, wait: Optional[float] = None):
    """Serialize the log's size check, rotation and append across processes, with a bounded wait.

    The lock is an OS-managed one (`flock`; `msvcrt.locking` on Windows) on a separate, stable
    file (`<log>.lock`, created if absent and never deleted). The OS releases it when the holder
    closes the file or dies, so there is no stale lock to detect and break, which is where a
    lock-file scheme (create-exclusive, then delete) goes wrong: two waiters that both judge a
    lock stale can delete each other's fresh one. It yields True when held and False when it
    could not be taken within `wait` seconds (or the lock file cannot be opened), so the caller
    can still append without rotating rather than drop its line or hang a session end. A
    rename-based rotation without a lock cannot keep a concurrent append: a writer that opened
    the log just before it was renamed writes into a file nobody reads."""
    wait = LOG_LOCK_WAIT_SECONDS if wait is None else wait
    held = False
    try:
        fd: Optional[int] = os.open(str(path.with_name(path.name + ".lock")), os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        fd = None
    if fd is not None:
        give_up = time.monotonic() + wait
        while True:
            if _try_lock(fd):
                held = True
                break
            if time.monotonic() >= give_up:
                break
            time.sleep(0.02)
    try:
        yield held
    finally:
        if fd is not None:
            os.close(fd)  # closing releases the lock


def log_note(log_path, message: str) -> None:
    """Append one timestamped line to the log. Bounded (past `MAX_LOG_BYTES` the old log is kept
    as `<name>.1`, replacing the previous one) and fail open: an unwritable log is no error.
    The size check, the rotation and the append happen under `_log_lock`, so two session ends
    at once neither replace the archive with a fresh log nor lose a line; if the lock cannot be
    taken in time the line is still appended, without rotating (the log may pass the bound by
    a line or two, which is harmless)."""
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{stamp} {_clean(message, 600)}\n"
        with _log_lock(path) as held:
            if held:
                try:
                    if path.stat().st_size > MAX_LOG_BYTES:
                        os.replace(path, path.with_name(path.name + ".1"))
                except OSError:
                    pass
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:  # noqa: BLE001 -- the log is a courtesy; a session end must never fail on it
        pass


def _new_report(transcript_path) -> dict:
    return {
        "ok": False, "transcript": os.path.basename(str(transcript_path)), "session_id": None,
        "candidates": 0, "prs": [], "skipped": [], "unrecognised_fetches": 0, "notes": [],
        "written": 0, "existing": 0, "failed": 0, "github_errors": [], "store_error": None, "error": None,
    }


def _skip(report: dict, pr: Optional[int], reason: str) -> None:
    report["skipped"].append({"pr": pr, "reason": reason})


def capture(transcript_path, *, repo: Optional[str] = None, store: Optional[metrics_lib.MetricsStore] = None,
            gh: P.GhRunner = P.run_gh, include_open: bool = False, dry_run: bool = False,
            deadline: Optional[float] = None, clock: Callable[[], float] = time.monotonic) -> dict:
    """Record the review rounds of every PR the transcript at `transcript_path` fetched feedback
    for (or reviewed locally). `repo` is `owner/name` for the `gh` calls (omitted, `gh` takes it
    from the current directory), `store` defaults to the shared `metrics.db`, `dry_run` builds
    everything and writes nothing (the store is not opened), `deadline` is a `clock()` value
    after which the remaining PRs are skipped.

    Returns `{"ok", "transcript", "session_id", "candidates", "prs", "skipped",
    "unrecognised_fetches", "notes", "written", "existing", "failed", "github_errors",
    "store_error", "error"}`: `ok` False only when the transcript could not be opened, `prs`
    one entry per recorded PR, `skipped` each (PR, reason) not recorded, `notes` anything worth
    a log line that is not a skip (an empty or unreadable transcript). Never raises."""
    report = _new_report(transcript_path)
    try:
        _capture(report, transcript_path, repo, store, gh, include_open, dry_run, deadline, clock)
    except Exception as e:  # noqa: BLE001 -- a session end must never fail on a capture
        report["error"] = f"unexpected {type(e).__name__}"
    return report


def _capture(report: dict, transcript_path, repo, store, gh, include_open: bool, dry_run: bool,
             deadline, clock) -> None:
    if repo is not None and not P.valid_repo(repo):
        report["error"] = f"repo must look like owner/name, got {_clean(repo, 80)!r}; nothing was read or written"
        return
    scanned = B.scan_transcript(transcript_path)
    if not scanned["ok"]:
        report["error"] = "the transcript could not be opened"
        return
    report["ok"] = True
    if scanned["last_ts"] is None:
        try:
            empty = os.path.getsize(str(transcript_path)) == 0
        except OSError:
            empty = False
        # A file with content but not one timestamped entry is not a transcript we can read
        # (the shared reader skips a garbled line silently, so say so here).
        report["notes"].append("the transcript is empty" if empty else "no readable entries (malformed transcript?)")
    report["candidates"] = len(scanned["prs"])
    target = None if dry_run else (store if store is not None else metrics_lib.MetricsStore())
    lookups: dict = {}
    for pr in sorted(scanned["prs"]):
        if deadline is not None and clock() > deadline:
            _skip(report, pr, "the time budget for this capture ran out")
            continue
        found = pr_round_cost_lib.find_rounds(transcript_path, pr)
        if not found.get("ok"):
            _skip(report, pr, "the transcript could not be read for this PR")
            continue
        report["session_id"] = report["session_id"] or found.get("session_id")
        unread = found.get("skipped_fetches", 0)
        report["unrecognised_fetches"] += unread
        # `_has_evidence` is the backfill's own test (a fetch the reader recognised, or a local
        # `review` call naming the PR); reusing it keeps the two from disagreeing about which
        # pairs hold a round. A number that merely appears in some other command is dropped
        # without a word: nothing to say about it.
        if not B._has_evidence(found, pr):
            if unread:
                _skip(report, pr, f"{unread} fetch call(s) the reader does not recognise, none it can read")
            continue
        if pr not in lookups:
            lookups[pr] = B.lookup_pr(pr, gh, repo)
        looked = lookups[pr]
        if not looked["ok"]:
            _skip(report, pr, f"the PR could not be looked up: {looked['error']}")
            continue
        if looked["state"] == "OPEN" and not include_open:
            _skip(report, pr, "the PR is still open; recorded by a later session end or a backfill after the merge")
            continue
        merge_time = None
        if looked["end"] is not None:
            last = scanned["last_ts"]
            # The epoch itself, never past the transcript's own last line (the backfill's rule
            # and reason: see its module docstring).
            merge_time = min(looked["end"], last) if last is not None else looked["end"]
        summary = P.record_pr_rounds(transcript_path, pr, repo=repo, merge_time=merge_time, store=target, gh=gh,
                                     dry_run=dry_run)
        _absorb(report, pr, looked, summary)
        if report["store_error"]:
            # The store is the likely cause for every later write too; stop rather than repeat it.
            break


def _absorb(report: dict, pr: int, looked: dict, summary: dict) -> None:
    """Fold one `record_pr_rounds` summary into the report."""
    if not summary["ok"]:
        _skip(report, pr, "the transcript could not be read for this PR")
        return
    for key in ("written", "existing", "failed"):
        report[key] += summary[key]
    for err in summary["github_errors"]:
        if err not in report["github_errors"]:
            report["github_errors"].append(err)
    if summary["store_error"]:
        report["store_error"] = summary["store_error"]
        # The recorder still returns the rounds and calls it PLANNED. A PR whose write failed is
        # not a recorded PR (the log would say `recorded=` beside `written=0`): list it as a
        # skip with what did get through, so `prs` holds only PRs the store accepted.
        _skip(report, pr, f"not fully recorded, the store failed ({summary['written']} written, "
                          f"{summary['existing']} already there, {summary['failed']} failed, "
                          f"{summary['not_attempted']} not attempted)")
        return
    if summary["rounds"] + summary["fetch_events"] == 0:
        _skip(report, pr, "nothing in this transcript falls before the PR's merge or closing time")
        return
    report["prs"].append({"pr": pr, "state": looked["state"], "rounds": summary["rounds"],
                          "fetch_events": summary["fetch_events"], "written": summary["written"],
                          "existing": summary["existing"], "failed": summary["failed"]})


def has_something_to_log(report: dict) -> bool:
    """False for the ordinary session that fetched no feedback: it gets no log line, or the log
    would grow one line per session end."""
    return bool(report["error"] or report["store_error"] or report["notes"] or report["prs"]
                or report["skipped"] or report["unrecognised_fetches"] or report["github_errors"])


def summary_line(report: dict) -> str:
    """One log line for a capture: counts, PR numbers and short errors, never transcript text."""
    parts = [f"transcript={_clean(report['transcript'], 80)}", f"session={_clean(report['session_id'] or '-', 64)}"]
    if report["error"]:
        parts.append(f"error={_clean(report['error'])}")
    if report["prs"]:
        parts.append("recorded=" + ",".join(f"{e['pr']}({e['rounds']}r/{e['fetch_events']}f)" for e in report["prs"]))
        parts.append(f"written={report['written']} existing={report['existing']} failed={report['failed']}")
    if report["skipped"]:
        parts.append("skipped=" + "; ".join(
            (f"PR {s['pr']}: " if s["pr"] else "") + _clean(s["reason"], 160) for s in report["skipped"][:5])
            + (f" (+{len(report['skipped']) - 5} more)" if len(report["skipped"]) > 5 else ""))
    if report["unrecognised_fetches"]:
        parts.append(f"unrecognised={report['unrecognised_fetches']}")
    for note in report["notes"]:
        parts.append(f"note={_clean(note, 120)}")
    for err in report["github_errors"][:3]:
        parts.append(f"github={_clean(err, 160)}")
    if report["store_error"]:
        parts.append(f"store_error={_clean(report['store_error'], 160)}")
    return " ".join(parts)


# --- spawning the worker (the SessionEnd hook's whole job) -------------------------------------

_WINDOWS_DETACH_FLAGS = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW


def worker_command(python: str, tools_dir, transcript_path) -> list:
    """The command line of the detached worker. `-I` (isolated mode) because it reads a
    transcript, which is untrusted: no PYTHONPATH, no user site, no script-directory import."""
    return [python, "-I", str(Path(tools_dir) / "capture_pr_rounds.py"), "--transcript", str(transcript_path)]


def spawn_worker(command: list, cwd, popen=subprocess.Popen) -> None:
    """Start `command` detached, with no inherited stdin/stdout/stderr (it logs to its own file),
    in `cwd` (so `gh` finds the repository the session was in). Returns at once without waiting;
    raises OSError if the process cannot start, which the caller turns into a log note."""
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                    "cwd": str(cwd), "close_fds": True}
    if sys.platform == "win32":
        kwargs["creationflags"] = _WINDOWS_DETACH_FLAGS
    else:
        kwargs["start_new_session"] = True
    popen(command, **kwargs)
