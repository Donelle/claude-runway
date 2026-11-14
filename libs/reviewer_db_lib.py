"""
The `codebase-reviewer` server's own SQLite database: what every review run cost.

It answers three questions without hand-written SQL (see `reviewer_report_lib.py`): which model
combination suits this machine, how long model X takes per request, and which combination gets a
review under a target time. The server owns this database, so it works whether or not the analytics
server is installed, and it is the place later pieces of the reviewer add their own tables (scoring,
rubric signals, Copilot feedback intake, replay results). Those tables will reference the identifiers
created here: `runs.id` (one review run) and `reviewer_setups.id` (a model plus its reply format,
translator and prompt hash, so two setups are never mixed together).

Nothing here is specific to one stage. A stage and a role are VALUES in a
column, never a column or a table per stage; what a request was about is a `(kind, id)` pair (a diff
chunk, a finding, a finding and a rubric entry). A stage added later records and reports
through the same calls with no change to this file or to the reports.

What is recorded, per run: start, end and outcome; the review type; the repository and diff range; the
number of chunks and findings; the settings that change the cost; the machine; and the model combination.
Per classified finding (the classifier stage): which rubric entry and level it took, the average probability (`confidence`), any overlap, and each
classifier's own probability for every entry, in the `classifications` table (never the claim text: this database holds no model-written text).
Per `need:` lookup a reviewer or verifier asked for: the request it was asked in, the round, the tool's name, the size
of the result, how much of it reached the model (`shown_chars`) and whether the tool cut it or it was an error, in `tool_calls`; and on the request itself, which round of the loop it was
(`need_round`) and whether it ended the loop by hitting the round limit (`need_limit_hit`), and whether the reply was a verifier's "cannot tell" verdict (`need_cannot_tell`). Never what the model wrote as the
arguments, never what came back, and never a tool name the model made up (an off-menu name is stored as `(not on the menu)`).
Per stage: start and end. Per request: start and end, reviewer, model, role, what it was about, prompt
and reply size in characters, prompt and reply tokens, the latency (submission to the final token, measured
here), the time to first token and the average time between tokens (both from LM Studio's own stats, in
milliseconds), and whether it was answered, timed out, refused (HTTP 400) or failed. Per model load and unload: start, end,
the context length it was loaded with, and the model's details from LM Studio (the `models` row is the model as first
seen; the context length is a setting of each load, so it is kept on the load).

A run that returns but in which some stage got no answer from any reviewer (every request failed, or its model never
loaded) is recorded as `incomplete`, not `finished`, so the review-time reports never count it as a fast review.

Writes FAIL OPEN, like `metrics_lib`: a recording problem is logged to stderr and never stops or slows a
review. Events are held in memory while the run goes (so a model request never waits on SQLite) and written
once, in one transaction, when the run closes, which is also when it closes after a failure part-way. Several
Claude sessions on one machine share the database file, so it runs in WAL mode with a short busy timeout; a
locked database costs at most that timeout, once, at the end of a run.

What went wrong with a reply (an unreadable reply, a translator failure, an empty reply) is kept in the request's row
as a count and a one-line summary, so it outlives the log file. The readable record of what was evaluated goes to a LOG
FILE, one per SESSION in a `codebase-reviewer-logs` folder next to the database, named after the session id
(`runs.session_id`, the id on the summary event in `metrics.db`), written like the harness's logs: each review is a
section with every request (stage, reviewer, model, which chunk or finding, outcome, timing, tokens), the findings each
reasoning reviewer raised, what grounding dropped, each verifier's verdict with its reasoning, the problems a reply
produced, the model loads, and the result returned to the caller. Every stage writes into the same file through the
same `note` call, so a stage added later appears in it with no change here. A session's
reviews append to its one file, so the files a machine accumulates grow with the number of sessions, not the number of
reviews; concurrent sessions write different files; files beyond the newest 200, or older than 30 days, are pruned when
a run closes. Requests are identified in it by the first eight characters of the request id in the `requests` table.

Beside the full record, each closed run writes one summary event to the shared `metrics.db`
(`metric_id` "codebase-reviewer", `event_type` "review", value the wall time in seconds), and, when
`CLAUDE_RUNWAY_TRACK_SAVINGS` is on, the characters of diff the local models read against the characters
returned to the caller to the savings ledger. Both are derived from this database's record and fail open.

Standard library only apart from the sibling libs imported lazily for those two derived writes.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

# 10: requests.need_cannot_tell. A file at the old version is only migrated when its version differs, so a new column bumps this.
SCHEMA_VERSION = 10

# A locked database costs a run at most this long, once, when it closes.
BUSY_TIMEOUT_SECONDS = 2.0

LOG_DIR_NAME = "codebase-reviewer-logs"
LOG_LINE_CHARS = 600
LOG_DETAIL_LINES = 12
LOG_RESULT_CHARS = 20000
WITHHELD = "[translator output withheld]"
LOG_KEEP_FILES = 200
LOG_KEEP_DAYS = 30

METRIC_ID = "codebase-reviewer"
SUMMARY_EVENT = "review"

FINISHED = "finished"
INCOMPLETE = "incomplete"

ANSWERED = "answered"
TIMED_OUT = "timed_out"
REFUSED = "refused"
FAILED = "failed"

# libs/openai_lib.py's failure kinds -> the outcome recorded for the request.
_FAILURE_OUTCOMES = {"timeout": TIMED_OUT, "bad_request": REFUSED}


def resolve_db_path() -> Path:
    """`CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB` if set, else `~/.claude/claude-runway/codebase-reviewer.db`.
    A relative override is anchored to the home directory, not whichever working directory a process
    happens to have (same rule as the other databases here)."""
    override = os.environ.get("CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.home() / path
        return path
    return Path.home() / ".claude" / "claude-runway" / "codebase-reviewer.db"


def log_dir(db_path: Path) -> Path:
    """The folder of per-run log files, next to the database."""
    return db_path.parent / LOG_DIR_NAME


_SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL UNIQUE,
    chip TEXT,
    memory_bytes INTEGER,
    os TEXT,
    arch TEXT,
    lmstudio_version TEXT,
    first_seen REAL NOT NULL,
    host TEXT
);
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    publisher TEXT,
    arch TEXT,
    format TEXT,
    quantization TEXT,
    max_context_length INTEGER,
    loaded_context_length INTEGER,
    first_seen REAL NOT NULL,
    UNIQUE (model_id, format, quantization, max_context_length)
);
CREATE TABLE IF NOT EXISTS reviewer_setups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_hash TEXT NOT NULL UNIQUE,
    model TEXT NOT NULL,
    reply_format TEXT,
    translator TEXT,
    prompt_hash TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    review_type TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    wall_s REAL NOT NULL,
    outcome TEXT NOT NULL,
    error TEXT,
    repo TEXT,
    base_ref TEXT,
    head_ref TEXT,
    branch TEXT,
    pr INTEGER,
    chunks INTEGER,
    findings INTEGER,
    confirmed INTEGER,
    timeouts INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    settings TEXT,
    combination TEXT,
    machine_id INTEGER REFERENCES machines (id),
    session_id TEXT
);
CREATE TABLE IF NOT EXISTS stages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs (id),
    stage TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    wall_s REAL NOT NULL,
    models TEXT
);
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs (id),
    stage TEXT,
    setup_id INTEGER REFERENCES reviewer_setups (id),
    reviewer TEXT,
    model TEXT,
    role TEXT,
    subject_kind TEXT,
    subject_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    wall_s REAL NOT NULL,
    prompt_chars INTEGER,
    reply_chars INTEGER,
    prompt_tokens INTEGER,
    reply_tokens INTEGER,
    reasoning_tokens INTEGER,
    latency_ms INTEGER,
    ttft_ms INTEGER,
    inter_token_ms REAL,
    tokens_per_second REAL,
    generation_ms INTEGER,
    stop_reason TEXT,
    outcome TEXT NOT NULL,
    problems INTEGER NOT NULL DEFAULT 0,
    problem_summary TEXT,
    log_ref TEXT,
    need_round INTEGER,
    need_limit_hit INTEGER,
    need_cannot_tell INTEGER
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs (id),
    request_id TEXT NOT NULL REFERENCES requests (id),
    round INTEGER NOT NULL,
    tool TEXT NOT NULL,
    result_chars INTEGER NOT NULL,
    truncated INTEGER NOT NULL DEFAULT 0,
    error INTEGER NOT NULL DEFAULT 0,
    shown_chars INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls (run_id);
CREATE TABLE IF NOT EXISTS model_loads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs (id),
    model_id TEXT NOT NULL,
    action TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    wall_s REAL NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT,
    model_row_id INTEGER REFERENCES models (id),
    loaded_context_length INTEGER
);
CREATE TABLE IF NOT EXISTS classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs (id),
    finding_no INTEGER NOT NULL,
    file TEXT,
    line_start INTEGER,
    line_end INTEGER,
    finding_class TEXT,
    verification TEXT,
    status TEXT NOT NULL,
    entry TEXT,
    level TEXT,
    confidence REAL,
    overlap INTEGER NOT NULL DEFAULT 0,
    overlapping TEXT,
    answers TEXT,
    scores TEXT
);
CREATE INDEX IF NOT EXISTS idx_classifications_run ON classifications (run_id);
CREATE INDEX IF NOT EXISTS idx_requests_run ON requests (run_id);
CREATE INDEX IF NOT EXISTS idx_requests_model_role ON requests (model, role);
CREATE INDEX IF NOT EXISTS idx_stages_run ON stages (run_id);
CREATE INDEX IF NOT EXISTS idx_loads_run ON model_loads (run_id);
CREATE INDEX IF NOT EXISTS idx_runs_combination ON runs (combination);
"""


# Columns added after schema version 1. `CREATE TABLE IF NOT EXISTS` leaves an existing table as it was, so a
# database made by an earlier version gets these added here; without that, every insert into it would fail (and,
# failing open, quietly record nothing).
_ADDED_COLUMNS = (
    ("requests", "latency_ms", "INTEGER"),
    ("requests", "ttft_ms", "INTEGER"),
    ("requests", "inter_token_ms", "REAL"),
    ("requests", "tokens_per_second", "REAL"),
    ("requests", "generation_ms", "INTEGER"),
    ("requests", "stop_reason", "TEXT"),
    ("requests", "problem_summary", "TEXT"),
    ("model_loads", "loaded_context_length", "INTEGER"),
    ("machines", "host", "TEXT"),
    ("classifications", "scores", "TEXT"),
    ("requests", "need_round", "INTEGER"),
    ("requests", "need_limit_hit", "INTEGER"),
    ("requests", "need_cannot_tell", "INTEGER"),
    ("tool_calls", "shown_chars", "INTEGER"),
)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, column, kind in _ADDED_COLUMNS:
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            if (table, column) == ("requests", "latency_ms"):
                # The same measurement the older rows already hold as seconds in wall_s.
                conn.execute("UPDATE requests SET latency_ms = CAST(ROUND(wall_s * 1000) AS INTEGER) WHERE latency_ms IS NULL")


def connect(db_path: Path, *, timeout: Optional[float] = None) -> sqlite3.Connection:
    """Opens the database and makes sure its schema exists. WAL, because several sessions share the file.
    A connection already opened is closed again before an error propagates (no leaked handle)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_SECONDS if timeout is None else timeout)
    try:
        # A file already at this version needs no schema work, so the run's write is the only time it takes the write
        # lock: a locked database then costs at most one busy timeout. Reading the version takes no write lock.
        if conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION:
            return conn
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        # Under the write lock: two sessions opening an older file at once would otherwise both see a column
        # missing, and the second ALTER would fail with "duplicate column name". The second now waits and
        # finds the column already there.
        conn.execute("BEGIN IMMEDIATE")
        _add_missing_columns(conn)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.commit()
    except sqlite3.Error:
        conn.close()
        raise
    return conn


def _warn(message: str) -> None:
    print(f"codebase-reviewer database: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------

_machine_cache: Optional[dict] = None
_machine_lock = threading.Lock()
# Runs of one session append to one log file; this keeps two reviews finishing at once from interleaving lines.
_log_lock = threading.Lock()


def _run_text(command: list, timeout: float = 3.0) -> Optional[str]:
    try:
        done = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = done.stdout.decode("utf-8", errors="replace").strip()
    return text if done.returncode == 0 and text else None


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_lmstudio_version(text: str) -> Optional[str]:
    """What identifies the LM Studio build in the output of `lms version`: that command prints a coloured banner
    and a `CLI commit:` line (no version number), so the commit is what is recorded, as `lms CLI commit <hash>`. If a
    newer `lms` prints a version number, that number is used instead. None when neither is there."""
    clean = _ANSI.sub("", text)
    number = re.search(r"\bv?(\d+\.\d+\.\d+[\w.+-]*)", clean)
    if number:
        return number.group(1)[:100]
    commit = re.search(r"CLI commit:\s*([0-9A-Za-z]+)", clean)
    return f"lms CLI commit {commit.group(1)}" if commit else None


def _lmstudio_version() -> Optional[str]:
    """LM Studio's build from its `lms` command line tool, when that is installed."""
    lms = shutil.which("lms")
    if not lms:
        return None
    text = _run_text([lms, "version"])
    return parse_lmstudio_version(text) if text else None


def _chip_and_memory() -> tuple:
    system = platform.system()
    chip: Optional[str] = None
    memory: Optional[int] = None
    if system == "Darwin":
        chip = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        raw = _run_text(["sysctl", "-n", "hw.memsize"])
        memory = int(raw) if raw and raw.isdigit() else None
    elif system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    chip = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("MemTotal:"):
                    memory = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            pass
    if not chip:
        chip = platform.processor() or None
    return chip, memory


def machine_info() -> dict:
    """What this machine is, captured once per process: chip, memory, OS and LM Studio version.
    Any field it cannot find is None; this never raises."""
    global _machine_cache
    with _machine_lock:
        if _machine_cache is None:
            try:
                chip, memory = _chip_and_memory()
                _machine_cache = {
                    "chip": chip,
                    "memory_bytes": memory,
                    "os": f"{platform.system()} {platform.release()}".strip(),
                    "arch": platform.machine() or None,
                    "lmstudio_version": _lmstudio_version(),
                }
            except Exception:
                _machine_cache = {"chip": None, "memory_bytes": None, "os": None, "arch": None, "lmstudio_version": None}
        return dict(_machine_cache)


def machine_fingerprint(info: dict) -> str:
    """A stable id for a machine, so results from different machines are never pooled by mistake. A remote LM Studio
    host is part of it; this machine's own fingerprint has no host, so it is the same as before hosts were recorded."""
    parts = [str(info.get(key)) for key in ("chip", "memory_bytes", "os", "arch", "lmstudio_version")]
    if info.get("host"):
        parts.append(f"host={info['host']}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


def lmstudio_host(url: Optional[str]) -> Optional[str]:
    """The host of an LM Studio URL when it is another computer, or None when it is this one (or unknown). Timing
    belongs to the computer that ran the model, so a remote one is recorded as its own machine."""
    if not url:
        return None
    try:
        from urllib.parse import urlsplit

        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return None
    if host in _LOCAL_HOSTS or host.startswith("127."):
        return None
    try:
        import socket

        if host in {socket.gethostname().lower(), socket.getfqdn().lower()}:
            return None
    except OSError:
        pass
    return host


def machine_for(lmstudio_url: Optional[str]) -> dict:
    """The machine a run's model requests ran on: this one, or for a remote LM Studio that host, whose chip, memory and
    OS this machine cannot know (left None)."""
    host = lmstudio_host(lmstudio_url)
    if host is None:
        return machine_info()
    return {"chip": None, "memory_bytes": None, "os": None, "arch": None, "lmstudio_version": None, "host": host}


# What differs from one problem line to the next without saying anything new: which reviewer, which chunk or finding.
_PROBLEM_PREFIX = re.compile(r"^(?:reviewer '[^']*': )?(?:(?:chunk|finding) \d+/\d+: )?")
MAX_SUMMARY_CHARS = 500


def summarize_problems(problems: list) -> Optional[str]:
    """One line saying what went wrong with a request's reply, durable in the `requests` row after the log files are
    pruned: each distinct problem once, with how many times it happened, e.g. `the reply was not JSON (x2); the translator
    exited with status 1`. The reviewer name and the chunk or finding number are left out (the row already says which
    request it is). None when there was no problem; cut with an ellipsis at `MAX_SUMMARY_CHARS`."""
    counts: dict = {}
    for text in problems:
        line = " ".join(_PROBLEM_PREFIX.sub("", str(text), count=1).split())
        if line:
            counts[line] = counts.get(line, 0) + 1
    if not counts:
        return None
    summary = "; ".join(f"{line} (x{n})" if n > 1 else line for line, n in counts.items())
    return summary if len(summary) <= MAX_SUMMARY_CHARS else summary[: MAX_SUMMARY_CHARS - 1].rstrip() + "\u2026"


def hash_text(text: str) -> str:
    """A short, stable hash of prompt text (a reviewer setup carries this, never the text)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class _NullHandle:
    id = ""

    def finish(self, reply, *, failure_kind=None, usage=None) -> None:
        pass

    def problem(self, text: str) -> None:
        pass

    def note(self, text: str, detail: tuple = ()) -> None:
        pass

    def need(self, round_number: int, limit_hit: bool = False) -> None:
        pass

    def cannot_tell(self) -> None:
        pass

    def tool_call(self, tool: str, result_chars: int, *, truncated: bool = False, error: bool = False, round_number: int = 0,
                  shown_chars: Optional[int] = None) -> None:
        pass

    def __enter__(self) -> "_NullHandle":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class NullRecorder:
    """Records nothing. The default wherever a recorder is optional, so a caller that does not pass one
    (and every existing test) behaves exactly as before."""

    def stage(self, name: str, models: tuple = ()):
        return _NullHandle()

    def request(self, **fields):
        return _NullHandle()

    def model_loaded(self, model: str, started: float, ended: float, ok: bool, error: Optional[str], details: Optional[dict]) -> None:
        pass

    def model_unloaded(self, model: str, started: float, ended: float, ok: bool, error: Optional[str]) -> None:
        pass

    def model_seen(self, details: Optional[dict]) -> None:
        pass

    def reviewer_failed(self, stage: str) -> None:
        pass

    def withhold(self, text: str) -> None:
        pass

    def note(self, stage: str, text: str, *, reviewer: str = "", detail: tuple = ()) -> None:
        pass

    def set_result_text(self, text: str) -> None:
        pass

    def set_counts(self, **counts) -> None:
        pass

    def add_classification(self, **record) -> None:
        pass

    def set_sizes(self, diff_chars: Optional[int] = None, result_chars: Optional[int] = None) -> None:
        pass

    def close(self, outcome: str = "finished", *, error: Optional[str] = None) -> None:
        """What `RunRecorder.close` is to the orchestrator: called once when a run ends, whichever way. Without it a run
        given no recorder (`RunState`'s default) would raise `AttributeError` after a successful review, and would hide
        the real error of a failed one behind that same `AttributeError`."""
        pass


NULL_RECORDER = NullRecorder()


class _Request:
    """One model request in flight. `finish` is called once with the reply (None when the request failed)."""

    def __init__(self, recorder: "RunRecorder", row: dict):
        self._recorder = recorder
        self._row = row
        self.id = row["id"]
        self._done = False

    def finish(self, reply: Optional[str], *, failure_kind: Optional[str] = None, usage: Optional[dict] = None) -> None:
        try:
            if self._done:
                return
            self._done = True
            row = self._row
            row["ended_at"] = self._recorder.now()
            row["wall_s"] = max(0.0, row["ended_at"] - row["started_at"])
            # Submission to the final token, measured here, so it includes any wait in LM Studio's queue.
            row["latency_ms"] = round(row["wall_s"] * 1000)
            if reply is None:
                row["outcome"] = _FAILURE_OUTCOMES.get(failure_kind or "", FAILED)
            else:
                row["outcome"] = ANSWERED
                row["reply_chars"] = len(reply)
            if usage:
                row["prompt_tokens"] = usage.get("prompt_tokens")
                row["reply_tokens"] = usage.get("completion_tokens")
                row["reasoning_tokens"] = usage.get("reasoning_tokens")
                for key in ("ttft_ms", "inter_token_ms", "tokens_per_second", "generation_ms", "stop_reason"):
                    row[key] = usage.get(key)
            self._recorder.add_request(row)
        except Exception as exc:
            _warn(f"could not record a request ({type(exc).__name__}: {exc})")

    def problem(self, text: str) -> None:
        """A problem this request's reply produced; counted in the row, summarized there, and written to the log."""
        try:
            self._row["problems"] = int(self._row.get("problems") or 0) + 1
            self._row.setdefault("problem_texts", []).append(str(text))
            self._recorder.add_note(request=self.id, stage=self._row.get("stage") or "", reviewer=self._row.get("reviewer") or "",
                                    kind="problem", text=str(text))
        except Exception as exc:
            _warn(f"could not record a problem ({type(exc).__name__}: {exc})")

    def note(self, text: str, detail: tuple = ()) -> None:
        """Something this request's reply showed (the findings it raised, a verifier's verdict and reasoning), written to
        the log under the request's line. `detail` is further lines."""
        try:
            self._recorder.add_note(request=self.id, stage=self._row.get("stage") or "", reviewer=self._row.get("reviewer") or "",
                                    kind="note", text=text, detail=detail)
        except Exception as exc:
            _warn(f"could not record a note ({type(exc).__name__}: {exc})")

    def need(self, round_number: int, limit_hit: bool = False) -> None:
        """Marks this request as one of the `need:` loop's: `round_number` is 0 for the first ask and n for the ask made after n
        rounds of lookups; `limit_hit` is true when the reply to it still only asked, after the round limit."""
        try:
            self._row["need_round"] = int(round_number)
            self._row["need_limit_hit"] = 1 if limit_hit else 0
        except Exception as exc:
            _warn(f"could not record a need round ({type(exc).__name__}: {exc})")

    def cannot_tell(self) -> None:
        """Marks this request's reply as a verifier's "cannot tell" verdict: not a vote but a request to look something up
        (with or without lookups named). Counted per request, so a run can say how many replies used the answer."""
        try:
            self._row["need_cannot_tell"] = 1
        except Exception as exc:
            _warn(f"could not record a cannot-tell answer ({type(exc).__name__}: {exc})")

    def tool_call(self, tool: str, result_chars: int, *, truncated: bool = False, error: bool = False, round_number: int = 0,
                  shown_chars: Optional[int] = None) -> None:
        """One lookup this request's reply asked for: the tool's name (a fixed name, never text the model wrote), how many
        characters came back, whether the tool cut it (`truncated`) or it was an error, and how many of those characters reached the
        model (`shown_chars`; None when not known): the reference budget can cut a result the tool did not, and `truncated` does not
        say so. Written to `tool_calls` when the run closes."""
        try:
            self._row.setdefault("tool_calls", []).append(
                {"round": int(round_number), "tool": str(tool), "result_chars": int(result_chars),
                 "truncated": 1 if truncated else 0, "error": 1 if error else 0,
                 "shown_chars": None if shown_chars is None else int(shown_chars)}
            )
        except Exception as exc:
            _warn(f"could not record a tool call ({type(exc).__name__}: {exc})")

    def __enter__(self) -> "_Request":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # A request that raised before it could be finished is a failed one, not a missing one.
        if not self._done:
            self.finish(None)
        return False


class _Stage:
    def __init__(self, recorder: "RunRecorder", name: str, models: tuple):
        self._recorder = recorder
        self._name = name
        self._models = models
        self._started = 0.0

    def __enter__(self) -> "_Stage":
        self._started = self._recorder.now()
        self._recorder.enter_stage(self._name)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self._recorder.add_stage(self._name, self._started, self._recorder.now(), self._models)
        except Exception as err:
            _warn(f"could not record a stage ({type(err).__name__}: {err})")
        return False


class RunRecorder:
    """Collects one review run's events in memory and writes them when it closes.

    Use it as a context manager, so the run is written even when it ends in an error:

        with RunRecorder(source="mcp", review_type="verifier", repo=..., base_ref=...) as recorder:
            ... run the review, passing `recorder` down ...
            recorder.set_counts(chunks=3, findings=7, confirmed=2)

    Thread-safe: reviewers run on a thread pool. Nothing in here raises into the review."""

    def __init__(
        self,
        *,
        source: str,
        review_type: str,
        repo: Optional[str] = None,
        base_ref: Optional[str] = None,
        head_ref: Optional[str] = None,
        branch: Optional[str] = None,
        pr: Optional[int] = None,
        settings: Optional[dict] = None,
        session_id: Optional[str] = None,
        lmstudio_url: Optional[str] = None,
        db_path: Optional[Path] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.run_id = uuid.uuid4().hex
        self.source = source
        self.review_type = review_type
        self.repo = repo
        self.base_ref = base_ref
        self.head_ref = head_ref
        self.branch = branch
        self.pr = pr
        self.settings = dict(settings or {})
        self.session_id = session_id
        # Where the model requests went: a remote LM Studio is recorded as its own machine (see machine_for).
        self.lmstudio_url = lmstudio_url
        self._db_path = db_path
        self._clock = clock
        self._lock = threading.Lock()
        self._started = clock()
        self._stages: list = []
        self._requests: list = []
        self._loads: list = []
        self._models: dict = {}
        self._notes: list = []
        self._seq = 0  # the order things were recorded in, for events at the same instant
        self._result_text: Optional[str] = None
        self._stage_names: list = []
        self._failed_stages: set = set()  # stages in which a reviewer failed, whether or not it sent a request
        self._withheld: list = []  # text the caller may see but the log never holds (what a translator printed)
        self._counts: dict = {}
        self._classifications: list = []
        self._diff_chars: Optional[int] = None
        self._result_chars: Optional[int] = None
        self._closed = False

    # -- the clock and the buffers (called by the helper objects above) -----

    def now(self) -> float:
        return self._clock()

    def enter_stage(self, name: str) -> None:
        with self._lock:
            if name not in self._stage_names:
                self._stage_names.append(name)

    def add_stage(self, name: str, started: float, ended: float, models: tuple) -> None:
        with self._lock:
            self._stages.append({"stage": name, "started_at": started, "ended_at": ended,
                                 "wall_s": max(0.0, ended - started), "models": sorted(set(models))})

    def add_request(self, row: dict) -> None:
        with self._lock:
            self._seq += 1
            row["seq"] = self._seq
            self._requests.append(row)

    def add_note(self, *, request: Optional[str], stage: str, reviewer: str, kind: str, text: str, detail: tuple = ()) -> None:
        with self._lock:
            self._seq += 1
            self._notes.append({"seq": self._seq, "at": self.now(), "request": request, "stage": stage, "reviewer": reviewer, "kind": kind,
                                "text": text, "detail": [str(line) for line in detail]})

    # -- what a review calls ------------------------------------------------

    def stage(self, name: str, models: tuple = ()) -> _Stage:
        """`with recorder.stage("reasoning", models=(...)):` times one stage. The name is a value, so a
        stage that does not exist yet records the same way."""
        return _Stage(self, name, tuple(models))

    def request(
        self,
        *,
        reviewer: str = "",
        role: str = "",
        model: str = "",
        stage: Optional[str] = None,
        subject: tuple = ("", ""),
        prompt_chars: Optional[int] = None,
        reply_format: Optional[str] = None,
        translator: Optional[str] = None,
        prompt_hash: Optional[str] = None,
    ) -> _Request:
        """`with recorder.request(...) as req:` times one model request; call `req.finish(reply, ...)`
        with the reply (None when it failed). `stage` defaults to the role."""
        kind, subject_id = subject
        row = {
            "id": uuid.uuid4().hex,
            "stage": stage or role or None,
            "reviewer": reviewer,
            "model": model,
            "role": role,
            "subject_kind": kind,
            "subject_id": str(subject_id),
            "started_at": self.now(),
            "prompt_chars": prompt_chars,
            "reply_chars": None,
            "prompt_tokens": None,
            "reply_tokens": None,
            "reasoning_tokens": None,
            "latency_ms": None,
            "ttft_ms": None,
            "inter_token_ms": None,
            "tokens_per_second": None,
            "generation_ms": None,
            "stop_reason": None,
            "problems": 0,
            "setup": {"model": model, "reply_format": reply_format or None, "translator": translator or None,
                      "prompt_hash": prompt_hash},
        }
        return _Request(self, row)

    def note(self, stage: str, text: str, *, reviewer: str = "", detail: tuple = ()) -> None:
        """A line in the session log that belongs to the stage and not to one request (what grounding dropped, what the
        classifier decided). The stage is a value, so a stage that does not exist yet writes the same way."""
        try:
            self.add_note(request=None, stage=stage, reviewer=reviewer, kind="note", text=text, detail=detail)
        except Exception as exc:
            _warn(f"could not record a note ({type(exc).__name__}: {exc})")

    def set_result_text(self, text: str) -> None:
        """The result returned to the caller, written at the end of the run's section of the log."""
        with self._lock:
            self._result_text = text

    def model_loaded(self, model: str, started: float, ended: float, ok: bool, error: Optional[str], details: Optional[dict]) -> None:
        self._add_load("load", model, started, ended, ok, error, details)

    def model_unloaded(self, model: str, started: float, ended: float, ok: bool, error: Optional[str]) -> None:
        self._add_load("unload", model, started, ended, ok, error, None)

    def withhold(self, text: str) -> None:
        """Text that may appear in the result returned to the caller but must not be written to the session log: what a
        failing translator printed, which can echo the model's reply. The log replaces it with WITHHELD."""
        try:
            if text and text.strip():
                with self._lock:
                    self._withheld.append(text)
        except Exception as exc:
            _warn(f"could not record text to withhold ({type(exc).__name__}: {exc})")

    def reviewer_failed(self, stage: str) -> None:
        """A reviewer of `stage` ended in an error. Counted as the stage having tried, so a stage in which every reviewer
        failed before sending a request (a diff too big for it) makes the run incomplete, not finished."""
        try:
            with self._lock:
                self._failed_stages.add(stage)
        except Exception as exc:
            _warn(f"could not record a reviewer failure ({type(exc).__name__}: {exc})")

    def model_seen(self, details: Optional[dict]) -> None:
        """A model that was already loaded (by the user) and is used as it is: its details are kept, no load is recorded."""
        try:
            if details and isinstance(details.get("id"), str):
                with self._lock:
                    self._models[details["id"]] = details
        except Exception as exc:
            _warn(f"could not record a model ({type(exc).__name__}: {exc})")

    def _add_load(self, action: str, model: str, started: float, ended: float, ok: bool, error: Optional[str], details: Optional[dict]) -> None:
        try:
            with self._lock:
                self._seq += 1
                context = details.get("loaded_context_length") if details else None
                self._loads.append({"seq": self._seq, "model_id": model, "action": action, "started_at": started, "ended_at": ended,
                                    "wall_s": max(0.0, ended - started), "ok": 1 if ok else 0, "error": error,
                                    "loaded_context_length": context if isinstance(context, int) else None})
                if details and isinstance(details.get("id"), str):
                    self._models[details["id"]] = details
        except Exception as exc:
            _warn(f"could not record a model load ({type(exc).__name__}: {exc})")

    def set_counts(self, **counts) -> None:
        """chunks, findings and confirmed, as the review learns them."""
        with self._lock:
            self._counts.update({k: v for k, v in counts.items() if v is not None})

    def add_classification(self, **record) -> None:
        """One finding's outcome from the classifier stage, written to `classifications` when the run closes. Keys: finding_no,
        file, line_start, line_end, finding_class, verification, status, entry, level, confidence, overlap, overlapping (a list
        of the other entries that fit), answers (one per classifier, each with its probability for every entry) and
        scores (the average per entry): both JSON-able lists, so any average can be recomputed. The claim text is not
        recorded here: this database holds no model-written text, and the session log shows the finding."""
        with self._lock:
            self._classifications.append(record)

    def set_sizes(self, diff_chars: Optional[int] = None, result_chars: Optional[int] = None) -> None:
        """The characters of diff the models read and the characters returned to the caller (the savings record)."""
        with self._lock:
            if diff_chars is not None:
                self._diff_chars = diff_chars
            if result_chars is not None:
                self._result_chars = result_chars

    # -- closing --------------------------------------------------------------

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close("finished")
        else:
            self.close("errored", error=f"{exc_type.__name__}: {exc}")
        return False

    def combination(self) -> str:
        """The model combination of this run: `stage:model+model|stage:model`, stages in the order they ran."""
        with self._lock:
            by_stage: dict = {name: set() for name in self._stage_names}
            for stage in self._stages:
                by_stage.setdefault(stage["stage"], set()).update(stage["models"])
            for req in self._requests:
                if req.get("stage") and req.get("model"):
                    by_stage.setdefault(req["stage"], set()).add(req["model"])
        return "|".join(f"{name}:{'+'.join(sorted(models))}" for name, models in by_stage.items() if models)

    def close(self, outcome: str = "finished", *, error: Optional[str] = None) -> Optional[dict]:
        """Writes the run, then its derived summary event and savings record. Idempotent. Returns the
        summary (also for a test), or None when the run was already closed. Never raises."""
        with self._lock:
            if self._closed:
                return None
            self._closed = True
        try:
            return self._close(outcome, error)
        except Exception as exc:
            _warn(f"could not record the run ({type(exc).__name__}: {exc})")
            return None

    def _unanswered_stages(self) -> list:
        """The stages that ran and tried (a request was sent, a model load failed inside them, or a reviewer failed)
        but in which no request was answered. A review with such a stage returned without doing that stage's work."""
        with self._lock:
            out = []
            for stage in self._stages:
                name = stage["stage"]
                requests = [r for r in self._requests if r.get("stage") == name]
                failed_load = any(not load["ok"] and load["action"] == "load"
                                  and stage["started_at"] <= load["started_at"] <= stage["ended_at"] for load in self._loads)
                tried = requests or failed_load or name in self._failed_stages
                if tried and not any(r.get("outcome") == ANSWERED for r in requests) and name not in out:
                    out.append(name)
            return out

    def _close(self, outcome: str, error: Optional[str]) -> dict:
        ended = self.now()
        if outcome == FINISHED:
            unanswered = self._unanswered_stages()
            if unanswered:
                outcome = INCOMPLETE
                error = error or f"no reviewer answered in: {', '.join(unanswered)}"
        machine = machine_for(self.lmstudio_url)
        timeouts = sum(1 for r in self._requests if r.get("outcome") == TIMED_OUT)
        failures = sum(1 for r in self._requests if r.get("outcome") in (FAILED, REFUSED))
        stage_times: dict = {}
        for stage in self._stages:
            stage_times[stage["stage"]] = round(stage_times.get(stage["stage"], 0.0) + stage["wall_s"], 3)
        load_s = round(sum(load["wall_s"] for load in self._loads), 3)
        summary = {
            "run_id": self.run_id,
            "source": self.source,
            "review_type": self.review_type,
            "outcome": outcome,
            "wall_s": round(ended - self._started, 3),
            "stage_s": stage_times,
            "load_s": load_s,
            "combination": self.combination(),
            "chunks": self._counts.get("chunks"),
            "findings": self._counts.get("findings"),
            "confirmed": self._counts.get("confirmed"),
            **self._classification_counts(),
            "requests": len(self._requests),
            "timeouts": timeouts,
            "failures": failures,
            "session_id": self.session_id,
            "repo": self.repo,
            "branch": self.branch,
            "pr": self.pr,
        }
        db_path = self._db_path or resolve_db_path()
        try:
            self._write_database(db_path, ended, outcome, error, machine, summary)
        except Exception as exc:
            _warn(f"could not write {db_path} ({type(exc).__name__}: {exc})")
        try:
            self._write_log(db_path, ended, outcome, error, summary)
        except Exception as exc:
            _warn(f"could not write the run log ({type(exc).__name__}: {exc})")
        self._write_summary_event(summary)
        self._write_savings()
        return summary

    def _classification_counts(self) -> dict:
        """What the classifier stage did in this run, for the summary event: findings per level, unclassified and overlaps.
        Nothing (no keys) when the stage did not run, so a review without classifiers is not reported as having
        classified zero findings. Counts only: the rubric's entry names and definitions are the user's own text and stay in the
        local `classifications` table."""
        if "classifier" not in self._stage_names:
            return {}
        counts = {f"classified_{level}": 0 for level in ("low", "moderate", "high")}
        counts.update(unclassified=0, overlaps=0)
        for record in self._classifications:
            if record["status"] == "classified" and f"classified_{record.get('level')}" in counts:
                counts[f"classified_{record['level']}"] += 1
            elif record["status"] == "unclassified":
                counts["unclassified"] += 1
            if record.get("overlap"):
                counts["overlaps"] += 1
        return counts

    def _write_database(self, db_path: Path, ended: float, outcome: str, error: Optional[str], machine: dict, summary: dict) -> None:
        conn = connect(db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO machines (fingerprint, chip, memory_bytes, os, arch, lmstudio_version, first_seen, host) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (machine_fingerprint(machine), machine["chip"], machine["memory_bytes"], machine["os"], machine["arch"],
                     machine["lmstudio_version"], ended, machine.get("host")),
                )
                machine_row = conn.execute("SELECT id FROM machines WHERE fingerprint = ?", (machine_fingerprint(machine),)).fetchone()
                model_rows = {model: self._model_row(conn, details, ended) for model, details in self._models.items()}
                conn.execute(
                    "INSERT INTO runs (id, source, review_type, started_at, ended_at, wall_s, outcome, error, repo, base_ref, head_ref, "
                    "branch, pr, chunks, findings, confirmed, timeouts, failures, settings, combination, machine_id, session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (self.run_id, self.source, self.review_type, self._started, ended, summary["wall_s"], outcome, error, self.repo,
                     self.base_ref, self.head_ref, self.branch, self.pr, summary["chunks"], summary["findings"], summary["confirmed"],
                     summary["timeouts"], summary["failures"], json.dumps(self.settings, sort_keys=True, default=str),
                     summary["combination"], machine_row[0] if machine_row else None, self.session_id),
                )
                conn.executemany(
                    "INSERT INTO stages (run_id, stage, started_at, ended_at, wall_s, models) VALUES (?, ?, ?, ?, ?, ?)",
                    [(self.run_id, s["stage"], s["started_at"], s["ended_at"], s["wall_s"], json.dumps(s["models"])) for s in self._stages],
                )
                conn.executemany(
                    "INSERT INTO model_loads (run_id, model_id, action, started_at, ended_at, wall_s, ok, error, model_row_id, "
                    "loaded_context_length) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(self.run_id, load["model_id"], load["action"], load["started_at"], load["ended_at"], load["wall_s"], load["ok"],
                      load["error"], model_rows.get(load["model_id"]), load["loaded_context_length"]) for load in self._loads],
                )
                request_rows = []
                for req in self._requests:
                    request_rows.append((
                        req["id"], self.run_id, req.get("stage"), self._setup_row(conn, req["setup"]), req.get("reviewer"),
                        req.get("model"), req.get("role"), req.get("subject_kind"), req.get("subject_id"), req["started_at"],
                        req["ended_at"], req["wall_s"], req.get("prompt_chars"), req.get("reply_chars"), req.get("prompt_tokens"),
                        req.get("reply_tokens"), req.get("reasoning_tokens"), req.get("latency_ms"), req.get("ttft_ms"),
                        req.get("inter_token_ms"), req.get("tokens_per_second"), req.get("generation_ms"), req.get("stop_reason"),
                        req["outcome"], req.get("problems") or 0, summarize_problems(req.get("problem_texts") or []), req["id"],
                        req.get("need_round"), req.get("need_limit_hit"), req.get("need_cannot_tell"),
                    ))
                conn.executemany(
                    "INSERT INTO requests (id, run_id, stage, setup_id, reviewer, model, role, subject_kind, subject_id, started_at, "
                    "ended_at, wall_s, prompt_chars, reply_chars, prompt_tokens, reply_tokens, reasoning_tokens, latency_ms, ttft_ms, "
                    "inter_token_ms, tokens_per_second, generation_ms, stop_reason, outcome, problems, problem_summary, log_ref, need_round, need_limit_hit, need_cannot_tell) "
                    "VALUES (" + ", ".join("?" * 30) + ")",
                    request_rows,
                )
                conn.executemany(
                    "INSERT INTO tool_calls (run_id, request_id, round, tool, result_chars, truncated, error, shown_chars) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [(self.run_id, req["id"], call["round"], call["tool"], call["result_chars"], call["truncated"], call["error"],
                      call["shown_chars"])
                     for req in self._requests for call in req.get("tool_calls") or []],
                )
                conn.executemany(
                    "INSERT INTO classifications (run_id, finding_no, file, line_start, line_end, finding_class, verification, status, "
                    "entry, level, confidence, overlap, overlapping, answers, scores) VALUES (" + ", ".join("?" * 15) + ")",
                    [(self.run_id, c["finding_no"], c.get("file"), c.get("line_start"), c.get("line_end"), c.get("finding_class"),
                      c.get("verification"), c["status"], c.get("entry"), c.get("level"), c.get("confidence"), 1 if c.get("overlap") else 0,
                      json.dumps(c.get("overlapping") or []), json.dumps(c.get("answers") or [], default=str),
                      json.dumps(c.get("scores") or [], default=str))
                     for c in self._classifications],
                )
        finally:
            conn.close()

    @staticmethod
    def _model_row(conn: sqlite3.Connection, details: dict, now: float) -> Optional[int]:
        key = (details.get("id"), details.get("compatibility_type"), details.get("quantization"), details.get("max_context_length"))
        conn.execute(
            "INSERT OR IGNORE INTO models (model_id, format, quantization, max_context_length, publisher, arch, loaded_context_length, "
            "first_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (*key, details.get("publisher"), details.get("arch"), details.get("loaded_context_length"), now),
        )
        row = conn.execute(
            "SELECT id FROM models WHERE model_id IS ? AND format IS ? AND quantization IS ? AND max_context_length IS ?", key
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _setup_row(conn: sqlite3.Connection, setup: dict) -> int:
        digest = hash_text(json.dumps(setup, sort_keys=True, default=str))
        conn.execute(
            "INSERT OR IGNORE INTO reviewer_setups (setup_hash, model, reply_format, translator, prompt_hash) VALUES (?, ?, ?, ?, ?)",
            (digest, setup.get("model") or "", setup.get("reply_format"), setup.get("translator"), setup.get("prompt_hash")),
        )
        return conn.execute("SELECT id FROM reviewer_setups WHERE setup_hash = ?", (digest,)).fetchone()[0]

    def _write_log(self, db_path: Path, ended: float, outcome: str, error: Optional[str], summary: dict) -> None:
        text = self.render_log(ended, outcome, error, summary)
        folder = log_dir(db_path)
        folder.mkdir(parents=True, exist_ok=True)
        with _log_lock:
            with open(folder / log_file_name(self.session_id), "a", encoding="utf-8") as handle:
                handle.write(text)
        prune_logs(folder)

    def render_log(self, ended: float, outcome: str, error: Optional[str], summary: dict) -> str:
        """This run's section of the session log, as text to read: a header, then in time order every request with the
        notes under it, the stage-level notes and the model loads, then the result returned and a footer."""
        started = self._started

        def offset(at: float) -> str:
            return f"[{max(0.0, at - started):7.1f}s]"

        def one_line(text: object, limit: int = LOG_LINE_CHARS) -> str:
            line = " ".join(str(text).split())
            return line if len(line) <= limit else line[: limit - 1].rstrip() + "\u2026"

        with self._lock:
            requests, loads, notes, result = list(self._requests), list(self._loads), list(self._notes), self._result_text
            withheld = sorted(set(self._withheld), key=len, reverse=True)
        # Before anything is cut, so a long text cannot survive in part.
        for text in withheld:
            result = result.replace(text, WITHHELD) if result else result
        children: dict = {}
        for note in notes:
            if note["request"]:
                children.setdefault(note["request"], []).append(note)

        def note_lines(note: dict, indent: str) -> list:
            label = "problem: " if note["kind"] == "problem" else ""
            lines = [f"{indent}{label}{one_line(note['text'])}"]
            lines.extend(f"{indent}    {one_line(line)}" for line in note["detail"][:LOG_DETAIL_LINES])
            return lines

        items: list = []
        for req in requests:
            who = f"{req.get('reviewer') or '?'} ({req.get('model') or '?'})"
            subject = f"{req['subject_kind']} {req['subject_id']}" if req.get("subject_kind") else "-"
            line = f"{offset(req['started_at'])} {req.get('stage') or '-'} | {who} | {subject} | {req['outcome']}"
            if req["outcome"] == ANSWERED or req.get("latency_ms") is not None:
                line += f" in {req['wall_s']:.1f}s"
            if req.get("ttft_ms") is not None:
                line += f", first token {req['ttft_ms']} ms"
            if req.get("inter_token_ms") is not None:
                line += f", {req['inter_token_ms']:g} ms/token"
            if req.get("prompt_tokens") is not None or req.get("reply_tokens") is not None:
                line += f", tokens {req.get('prompt_tokens') if req.get('prompt_tokens') is not None else '-'} in / " \
                        f"{req.get('reply_tokens') if req.get('reply_tokens') is not None else '-'} out"
                if req.get("reasoning_tokens"):
                    line += f" ({req['reasoning_tokens']} thinking)"
            if req.get("stop_reason") and req["stop_reason"] != "eosFound":
                line += f", stopped: {req['stop_reason']}"
            line += f"  [request {req['id'][:8]}]"
            lines = [line]
            for note in sorted(children.get(req["id"], []), key=lambda n: n["at"]):
                lines.extend(note_lines(note, " " * 11))
            items.append((req["started_at"], req.get("seq", 0), lines))
        for note in notes:
            if not note["request"]:
                items.append((note["at"], note["seq"], [f"{offset(note['at'])} {note['stage'] or '-'} | " + note_lines(note, "")[0].lstrip()]
                              + note_lines(note, " " * 11)[1:]))
        for load in loads:
            what = "loaded" if load["action"] == "load" else "unloaded"
            state = f"{what} in {load['wall_s']:.1f}s" if load["ok"] else f"FAILED to be {what}: {one_line(load['error'] or '')}"
            items.append((load["started_at"], load["seq"], [f"{offset(load['started_at'])} model | {load['model_id']} | {state}"]))
        items.sort(key=lambda item: (item[0], item[1]))

        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))
        out = [f"=== run {self.run_id} | {stamp} | {self.review_type} review via {self.source} ==="]
        where = [f"repo {self.repo}" if self.repo else "", f"range {self.base_ref}...{self.head_ref}" if self.base_ref or self.head_ref else "",
                 f"branch {self.branch}" if self.branch else "", f"pr {self.pr}" if self.pr else ""]
        out.append("    " + "  ".join(part for part in where if part))
        if self.settings:
            out.append("    settings: " + ", ".join(f"{k}={v}" for k, v in sorted(self.settings.items())))
        for _, _, lines in items:
            out.extend(lines)
        if result:
            out.append("--- result returned to the caller ---")
            shown = result if len(result) <= LOG_RESULT_CHARS else result[:LOG_RESULT_CHARS] + "\n\u2026 (cut)"
            out.extend(shown.splitlines())
            out.append("--- end of result ---")
        parts = [f"{outcome} in {summary['wall_s']:.1f}s", summary.get("combination") or "no model used",
                 f"{summary['requests']} request(s): {summary['timeouts']} timed out, {summary['failures']} failed"]
        if summary.get("load_s"):
            parts.append(f"model loading {summary['load_s']:.1f}s")
        if error:
            parts.append(f"error: {one_line(error)}")
        out.append("=== " + " | ".join(parts) + " ===")
        return "\n".join(out) + "\n\n"

    def _write_summary_event(self, summary: dict) -> None:
        """One event in the shared metrics.db, derived from the run just written. Fails open."""
        try:
            import metrics_lib

            metrics_lib.MetricsStore().record(
                METRIC_ID,
                SUMMARY_EVENT,
                value=float(summary["wall_s"]),
                metadata={k: v for k, v in summary.items() if k != "wall_s"} | {"wall_s": summary["wall_s"]},
                session_id=summary.get("session_id"),
            )
        except Exception as exc:
            _warn(f"could not write the summary event ({type(exc).__name__}: {exc})")

    def _write_savings(self) -> None:
        """The characters of diff the local models read against what came back to the caller, when savings
        tracking is on. Skipped when either size is unknown or no live session of the calling project can be
        found. Fails open.

        The calling project is this process's working directory (an MCP server starts in the project of the
        session that launched it, which is how the analytics server finds it too), never the reviewed
        repository, which can be another checkout. The unfiltered lookup would take the newest ledger of ANY
        project and credit the review to a session that did not ask for it."""
        try:
            if self._diff_chars is None or self._result_chars is None:
                return
            import savings_ledger

            if not savings_ledger.tracking_enabled():
                return
            project = savings_ledger.project_name_from_cwd(os.getcwd())
            session = savings_ledger.current_session_id(project=project)
            if not session:
                return
            savings_ledger.record_event(session, "codebase-reviewer", self._diff_chars, self._result_chars, True, source=self.source,
                                        project=project)
        except Exception as exc:
            _warn(f"could not write the savings record ({type(exc).__name__}: {exc})")


def log_file_name(session_id: Optional[str]) -> str:
    """The log file of a session: its id, made safe for a file name. A run with no session id (none could be
    worked out) shares one file per process rather than getting a file of its own."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (session_id or ""))[:100]
    return f"{safe or 'no-session-' + str(os.getpid())}.log"


def prune_logs(folder: Path, *, keep: int = LOG_KEEP_FILES, max_age_days: float = LOG_KEEP_DAYS, now: Optional[float] = None) -> int:
    """Deletes run logs beyond the newest `keep` and any older than `max_age_days`. Returns how many it
    deleted. A file it cannot delete is left; never raises."""
    removed = 0
    try:
        listed = [(p.stat().st_mtime, p) for p in folder.glob("*.log") if p.is_file()]
        listed.sort(key=lambda pair: pair[0], reverse=True)
        cutoff = (time.time() if now is None else now) - max_age_days * 86400
        for index, (mtime, path) in enumerate(listed):
            try:
                if index >= keep or mtime < cutoff:
                    # Another server may have appended to this file since it was listed (its session came back):
                    # it is then one of the newest, so it is kept. The check narrows the race to the instant between
                    # this stat and the unlink; a lock across processes would cost more than one section of a log.
                    if path.stat().st_mtime != mtime:
                        continue
                    path.unlink()
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed



GIT_TIMEOUT_SECONDS = 5


def branch_of(repo_path: str, ref: str) -> Optional[str]:
    """The branch name `ref` names in the repository, or None (detached, not a branch, no git, slow git).
    Best effort and bounded: it only labels a run."""
    try:
        done = subprocess.run(
            ["git", "-C", os.path.expanduser(repo_path), "rev-parse", "--abbrev-ref", ref],
            capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name = done.stdout.decode("utf-8", errors="replace").strip()
    return name if done.returncode == 0 and name and name != "HEAD" else None


def new_run_recorder(
    *,
    source: str,
    review_type: str,
    repo_path: str,
    base_ref: str,
    head_ref: str,
    pr: Optional[int] = None,
    settings: Optional[dict] = None,
    lmstudio_url: Optional[str] = None,
) -> RunRecorder:
    """A recorder for one review run, labelled with the repository, the diff range, the branch of `head_ref`
    and this process's session id. `lmstudio_url` is where the model requests go (a remote one is recorded as its
    own machine). Never raises: what it cannot work out is left blank."""
    try:
        import session_id_lib

        session = session_id_lib.session_id(session_id_lib.SessionIdStrategy.PROXY)
    except Exception:
        session = None
    return RunRecorder(
        source=source,
        review_type=review_type,
        repo=os.path.abspath(os.path.expanduser(repo_path)),
        base_ref=base_ref,
        head_ref=head_ref,
        branch=branch_of(repo_path, head_ref),
        pr=pr,
        settings=settings,
        session_id=session,
        lmstudio_url=lmstudio_url,
    )
