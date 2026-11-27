#!/usr/bin/env python3
"""Run Pi (the coding agent) in a throwaway Docker container against the codebase-reviewer's MCP server, with caps.

Why this exists: the harness scripts call the reviewer's libraries directly, so they never exercise the path an agent
takes: finding the `review` tool over MCP and calling it. This does, with Pi as the agent, and it keeps the whole run in a
container: it never sees `~/.claude` or `~/.pi`, and the only thing of the host checkout it can write is the harness database
folder (`harness/live-tests/logs/db/`, ignored by Git). On the host the driver itself writes a temporary scratch folder that
it deletes, that database when it is missing, and the `--log` file, which is refused under `~/.claude` and `~/.pi` and never
overwrites a file. Pi has no turn or tool-call limit of its own, so the driver enforces them: it reads Pi's JSON event stream
and runs `docker kill` when a cap is reached.

What Pi is asked to review is a change from the `harness-live-tests` branch (`--change`, default `change-6`; see harness/README.md,
"A real repository with seeded bugs"): real code with a seeded bug and a decoy, not a made-up case. The container is given a
scratch repository that holds only the branch's base and that change, fetched from a checkout that has the tags (`--repo`,
default this one). The checkout is not mounted (of this one, only the reviewer's own `libs/` and `mcpservers/` are, read-only,
and the harness database folder, writable), and neither is the answer key, which sits in it under `harness/live-tests/`. The JSON
summary names the change, so `score_run.py` can score it against the key without being told.

One session runs one review of one type (`--type`: reasoning, verifier or classifier) in one container. A verifier or classifier
session starts from a findings file (`--findings-file`), of which the container gets a checked copy. The driver does that one job:
reading configs for a whole run, choosing reviewers, loading models, combining and scoring belong to whoever calls it. It runs
from the command line, and `run_session(SessionSpec(...), stop=...)` is the same session as a Python function.

A session passes only when Pi's `review` call that RAN was the one the session is for (type, repository, refs and findings file)
and the results file of that review came back.

A development tool, not product code (CI does not lint or type-check `harness/`; the logic here is covered by
`tests/test_harness_pi_driver.py` with a fake `docker`, no Docker needed). See harness/pi/README.md.

    .venv/bin/python harness/pi/pi_driver.py --build --model <lm-studio-id>

Standard library only. Exit code: 0 = Pi finished, its review call was this session's and the results came back; 1 = it finished
without that (no review call ran, every call errored, a call that ran was another review, no results file), Pi aborted, or the
container failed; 2 = a bad argument, a failed preflight, a checkout without the branch's tags or docker could not be started
(nothing was run); 3 = a cap, the time limit or a stop request stopped the run.
"""

import argparse
import contextlib
import copy
import json
import os
import posixpath
import queue
import shlex
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PI_DIR = Path(__file__).resolve().parent

# `libs/` (the reviewer's config validator, `runway_defaults`, the LM Studio helpers) and `harness/` (the live scoring module) are
# imported by bare name, as everywhere in this repository. The only harness module used is `live_scoring_lib`, which depends on
# nothing else there: neither `cases.py` nor `harness_lib.py` is imported, here or through it, since the branch is the only test
# code and `cases.py` is on its way out.
for _directory in (REPO_ROOT / "libs", REPO_ROOT / "harness"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import live_scoring_lib as scoring  # noqa: E402
from codebase_reviewer_config_lib import ROLES, ConfigError, read_settings, validate_reviewers  # noqa: E402

DEFAULT_IMAGE = "claude-runway-pi:dev"
# What the container calls the host. Docker Desktop provides the name; on Linux `--add-host ...:host-gateway` does.
CONTAINER_HOST_ALIAS = "host.docker.internal"
# Hosts that mean "this machine". Inside a container that is the container itself, so a URL with one of them cannot be
# handed to the container as it is (see `container_url`).
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

# The caps. Defaults come from the spike's own guards: a turn cap alone does not stop a flood of tool calls inside one
# turn (seen in the spike on what Pi is), so there are three: turns, tool calls and wall-clock time.
DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_TOOL_CALLS = 30
DEFAULT_TIMEOUT_SECONDS = 600

# The change to review when `--change` is not given: two lines added and two removed in one file, one bug and one decoy, both
# judgeable from the diff alone (see harness/README.md), so a run measures whether the CALLER can drive the review tool, with
# a result small enough to read by eye. A bigger change from the branch is one `--change` away.
DEFAULT_CHANGE = "change-6"
DEFAULT_CONTEXT_LENGTH = 16384
# The most one `git` call on the checkout or the scratch repository may take. A local fetch of two small commits takes well under a
# second, so this only ends a call that is stuck (a lock, a prompt); it is not a tuning knob.
GIT_TIMEOUT_SECONDS = 120

# Where things are inside the container. Nothing here is a host path.
C_REVIEWER = "/opt/reviewer"  # `libs/` and `mcpservers/` of this checkout, read-only
C_REPO = "/work/repo"  # the scratch repository (the branch's base and one change), read-only
C_CONFIG = "/config"  # the reviewers file, read-only
C_AGENT = "/agent"  # PI_CODING_AGENT_DIR: Pi's settings, models.json and mcp.json for this run only
C_RESULTS = "/results"  # where the review tool saves its structured results (REVIEW_OUTPUT_DIR), for this run only
C_DB = "/db"  # the harness database and the reviewer's text logs, shared by every session and kept between runs
C_FINDINGS = "/findings"  # a verifier or classifier session only: a copy of its findings file, read-only
C_FINDINGS_FILE = f"{C_FINDINGS}/findings.json"
C_VENV_PYTHON = "/opt/venv/bin/python"
# The image's own `timeout`: Pi runs under it, so a container cannot outlive its time limit even if this process is killed
# outright (on Windows a killed process runs no cleanup). It fires BACKSTOP_MARGIN_SECONDS after the session's limit, so the
# driver's own cap, which also records why the run ended, goes first.
C_TIMEOUT = "/usr/bin/timeout"
BACKSTOP_MARGIN_SECONDS = 15
KILL_AFTER_SECONDS = 10  # `timeout` sends SIGKILL this long after its SIGTERM, if the process is still there
TIMEOUT_EXIT_CODE = 124  # `timeout`'s own exit code for a command it had to end

# Folders of a checkout, as path parts. The database folder is the ONE writable thing of the checkout a container may be given.
# It is `db/` and never `logs/` above it: older logs can print the answer key.
HARNESS_DB_SUBDIR = ("harness", "live-tests", "logs", "db")
DB_FILE_NAME = "codebase-reviewer.db"
# The only folders of a checkout a reviewer's `promptFile` may be copied from.
PROMPT_SUBDIRS = (("harness", "prompts"), ("harness", "live-tests", "prompts"))

# How long one `docker kill`, `rm` or `ps` may take before the driver gives up on it.
DOCKER_COMMAND_TIMEOUT_SECONDS = 30
REMOVE_ATTEMPTS = 3
REMOVE_PAUSE_SECONDS = 1.0
CLIENT_EXIT_SECONDS = 10  # how long a killed `docker` client is given to be gone (it takes milliseconds)

PROVIDER = "lmstudio"
MCP_SERVER_NAME = "reviewer"
# Pi names an MCP tool mcp__<server>__<tool>; --tools takes the pattern to expose the server's tools and nothing else MCP.
REVIEW_TOOL = f"mcp__{MCP_SERVER_NAME}__review"
# How the review tool words a refusal: `review()` in mcpservers/codebase_reviewer_mcp_server.py returns "Error: ..." as a normal
# result for an invalid reviewers file, bad parameters or nothing to run (its docstring promises the prefix).
REVIEW_REFUSAL_PREFIX = "Error:"
# Pi's own read-only tools. Never `bash`, `edit` or `write`: the run needs to look, not change.
READ_ONLY_TOOLS = ("read", "grep", "find", "ls")

# {repo} is replaced with the repository's path inside the container, {base} and {head} with the two tags it holds, and
# {findings} with the findings file's path (plain replace, so the JSON braces need no escaping).
DEFAULT_PROMPT = (
    "An MCP tool named review is available. Call it once with type \"reasoning\" and params "
    '{"repo_path": "{repo}", "base_ref": "{base}", "head_ref": "{head}"}. '
    "Then list the findings it returns, one per line. Do not edit any file."
)
# The same words for a verifier or classifier session, which starts from a findings file.
FINDINGS_PROMPT = (
    "An MCP tool named review is available. Call it once with type \"{type}\" and params "
    '{"repo_path": "{repo}", "base_ref": "{base}", "head_ref": "{head}", "findings_file": "{findings}"}. '
    "Then list the findings it returns, one per line. Do not edit any file."
)


def default_prompt(review_type: str) -> str:
    """What Pi is asked in a session of `review_type`, with the placeholders still in it (see `fill_prompt`)."""
    return DEFAULT_PROMPT if review_type == "reasoning" else FINDINGS_PROMPT.replace("{type}", review_type)


def fill_prompt(prompt: str, head_ref: str, findings_file: Optional[str]) -> str:
    return (prompt.replace("{repo}", C_REPO).replace("{base}", scoring.BASE_REF).replace("{head}", head_ref)
            .replace("{findings}", findings_file or ""))


class SessionError(Exception):
    """A session could not be set up, so nothing ran: a bad argument, a file the container must not be given, a checkout without
    the branch's tags, or no `docker`. The command line reports it as a usage error (exit 2)."""


def _shown(value: object, limit: int = 120) -> str:
    """`value` for a message about what a model sent: its repr, cut, since a model can send text of any length."""
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


@dataclass(frozen=True)
class ExpectedCall:
    """The one `review` call a session is for, as the container spells it."""

    review_type: str
    head_ref: str
    findings_file: Optional[str] = None
    repo_path: str = C_REPO
    base_ref: str = scoring.BASE_REF

    def mismatches(self, args: object) -> list:
        """What is wrong with the arguments of a `review` call; empty when it is this session's call.

        The refs are checked as well as the type and the findings file: swapped refs would review another diff and the run would
        still look like a review. `head_ref` left out (or "HEAD") is accepted, because the container's repository has the change
        checked out, so it is the same commit. Other params (`context_lines`, `chunk_chars`, `pr`) are the caller's choice."""
        if not isinstance(args, dict):
            return ["its arguments were not an object"]
        problems = []
        if args.get("type") != self.review_type:
            problems.append(f"type was {_shown(args.get('type'))}, not {self.review_type!r}")
        params = args.get("params")
        if not isinstance(params, dict):
            return [*problems, "params was not an object"]
        repo = params.get("repo_path")
        if not isinstance(repo, str) or posixpath.normpath(repo) != self.repo_path:
            problems.append(f"repo_path was {_shown(repo)}, not {self.repo_path!r}")
        if params.get("base_ref") != self.base_ref:
            problems.append(f"base_ref was {_shown(params.get('base_ref'))}, not {self.base_ref!r}")
        if params.get("head_ref", "HEAD") not in (self.head_ref, "HEAD"):
            problems.append(f"head_ref was {_shown(params.get('head_ref'))}, not {self.head_ref!r}")
        if params.get("findings_file") != self.findings_file:
            wanted = "none" if self.findings_file is None else repr(self.findings_file)
            problems.append(f"findings_file was {_shown(params.get('findings_file'))}, not {wanted}")
        return problems

# ---------------------------------------------------------------------------
# Pi's event stream
# ---------------------------------------------------------------------------


class LineSplitter:
    """Turns chunks of bytes into complete lines, splitting on LF only.

    Pi's JSONL framing is strict: a record ends at LF (an optional CR before it is stripped). The Unicode line and
    paragraph separators (U+2028, U+2029) are valid INSIDE a JSON string, so `str.splitlines()` or `readline` would cut a
    record in two. A chunk may end mid-record, and even mid-way through a multi-byte character, so bytes are kept until
    the LF arrives and decoded only then."""

    def __init__(self) -> None:
        self._pending = b""

    def feed(self, chunk: bytes) -> list:
        self._pending += chunk
        *complete, self._pending = self._pending.split(b"\n")
        return [self._decode(line) for line in complete]

    def flush(self) -> list:
        """What is left when the stream ends: a last record with no LF, if there is one."""
        rest, self._pending = self._pending, b""
        return [self._decode(rest)] if rest else []

    @staticmethod
    def _decode(raw: bytes) -> str:
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        return raw.decode("utf-8", errors="replace")


def parse_event(line: str) -> Optional[dict]:
    """The record as a dict, or None for a blank line, text that is not JSON, or JSON that is not an object."""
    if not line.strip():
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def message_text(event: dict) -> str:
    """The text of an assistant `message_end` record, or ''. Content is a list of blocks (or, for a user message, a string)."""
    message = event.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")


def tool_result_text(event: dict) -> str:
    """The text of a `tool_execution_end` record's result (its content blocks joined), in full: it is the evidence a run
    is judged by, and a cut with no marker would pass for the whole result."""
    result = event.get("result")
    content = result.get("content") if isinstance(result, dict) else None
    if not isinstance(content, list):
        return ""
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


class CapTracker:
    """Counts what Pi does and says when a cap has been reached.

    A cap is reached when the work BEYOND it starts: with a limit of 10 turns, the 11th `turn_start` trips it, so a run
    that finishes in its 10th turn is a completed run and not a capped one. Time is checked apart (`out_of_time`), because
    a model that is thinking sends no events at all, and the wall clock must stop it all the same."""

    def __init__(self, max_turns: int, max_tool_calls: int, timeout_seconds: float, clock: Callable[[], float] = time.monotonic,
                 expected: Optional[ExpectedCall] = None) -> None:
        self.expected = expected  # the session's `review` call; None checks no arguments
        self.good_calls = 0  # review calls that ran, with this session's arguments, and did not error
        self.call_errors: list = []  # one line for each review call that ran with other arguments
        self._open_calls: dict = {}  # toolCallId -> the arguments of the review calls that have started and not ended
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.timeout_seconds = timeout_seconds
        self._clock = clock
        self._started = clock()
        self.turns_started = 0
        self.turns = 0  # finished (turn_end)
        self.tool_calls = 0
        self.review_calls = 0
        self.review_errors = 0
        self.settled = False
        self.aborted = False  # Pi settled because the run was aborted, which is not a finished run
        self.final_text = ""
        self.review_text = ""
        self._review_ran = False  # whether `review_text` came from a call that ran

    def elapsed(self) -> float:
        return self._clock() - self._started

    def remaining(self) -> float:
        return self.timeout_seconds - self.elapsed()

    def out_of_time(self) -> bool:
        return self.remaining() <= 0

    def observe(self, event: dict) -> Optional[str]:
        """Takes one record; returns the `stopped_by` reason ("turns" or "tool_calls") if it reached a cap, else None."""
        kind = event.get("type")
        if kind == "turn_start":
            self.turns_started += 1
            if self.turns_started > self.max_turns:
                return "turns"
        elif kind == "turn_end":
            self.turns += 1
        elif kind == "tool_execution_start":
            self.tool_calls += 1
            if event.get("toolName") == REVIEW_TOOL:
                self.review_calls += 1
                self._open_calls.setdefault(event.get("toolCallId"), []).append(event.get("args"))
            if self.tool_calls > self.max_tool_calls:
                return "tool_calls"
        elif kind == "tool_execution_end":
            if event.get("toolName") == REVIEW_TOOL:
                # What the reviewer itself returned, kept because the caller's own summary of it can drop most of it. What Pi
                # says about a call it refused is kept only until a call runs, so a stray refused call after the review cannot
                # replace the review (`_check_call` says how the two are told apart).
                text = tool_result_text(event)
                ran = "durationMs" in event
                if ran or not self._review_ran:
                    self.review_text = text
                    self._review_ran = ran
                # The review tool reports its own refusals (a bad reviewers file, nothing to run) as ordinary text that
                # starts with "Error:" and an isError of false, so a refused review would otherwise read as a success.
                errored = bool(event.get("isError")) or text.lstrip().startswith(REVIEW_REFUSAL_PREFIX)
                if errored:
                    self.review_errors += 1
                self._check_call(event, errored)
        elif kind == "message_end":
            text = message_text(event)
            if text:
                self.final_text = text
        elif kind == "agent_settled":
            self.settled = True
            self.aborted = bool(event.get("aborted"))
        return None

    def _check_call(self, event: dict, errored: bool) -> None:
        """Judges the arguments of a `review` call when it ends, the first moment Pi says whether the tool ran.

        Pi sends `tool_execution_start` BEFORE it validates the arguments, so a call it refuses has a start record too. Only the
        end record of a call it executed carries `durationMs`; one refused for bad arguments, an unknown tool, a block, an abort
        or a truncated message has none (Pi 1.6.0, `agent-loop.js`). A refused call ran nothing, so it is not judged: a refused
        call followed by a good retry is not a failure. If a later Pi stopped sending `durationMs`, no call would count as good,
        and the session would fail for lack of one rather than pass unchecked."""
        pending = self._open_calls.get(event.get("toolCallId"), [])
        if "durationMs" not in event:
            if pending:
                pending.pop()  # a refused call ends straight after its own start, so the newest arguments are its own
            return
        # An end with no start has unknown arguments, which are not this session's. Calls that share an id can end in any order,
        # so every one of their arguments has to be right.
        candidates = list(pending) or [None]
        if pending:
            pending.pop(0)
        if self.expected is None:
            return
        problems = list(dict.fromkeys(problem for args in candidates for problem in self.expected.mismatches(args)))
        if problems:
            self.call_errors.append("a review call that ran was not this session's: " + "; ".join(problems))
        elif not errored:
            self.good_calls += 1


@dataclass
class RunResult:
    change: str = ""  # the branch change that was reviewed (`change-6`): what score_run.py reads the answers for
    review_type: str = ""
    review_id: str = ""
    stopped_by: str = "completed"  # completed | turns | tool_calls | timeout | stopped | aborted | error
    turns: int = 0
    tool_calls: int = 0
    review_calls: int = 0
    review_errors: int = 0
    elapsed_seconds: float = 0.0
    exit_code: Optional[int] = None  # the container client's, when it ended on its own (None when the driver stopped it)
    final_text: str = ""  # what the caller model said at the end: its summary, which may leave most of the result out
    review_text: str = ""  # what the reviewer's own tool returned: the result to read
    call_errors: list = field(default_factory=list)  # review calls that ran with arguments other than this session's
    failures: list = field(default_factory=list)  # what else keeps the run from passing (no results, a container maybe left)
    results: list = field(default_factory=list)  # the review tool's results files, as parsed JSON
    stderr_tail: list = field(default_factory=list)

    def exit_status(self) -> int:
        """The driver's own exit code: see the module docstring."""
        if self.stopped_by in ("turns", "tool_calls", "timeout", "stopped"):
            return 3
        if self.stopped_by == "completed" and self.review_calls > 0 and not self.call_errors and not self.failures:
            # Every call to the reviewer errored: the run worked and the review did not, which must not read as a pass.
            return 1 if self.review_errors >= self.review_calls else 0
        return 1


# ---------------------------------------------------------------------------
# The container command
# ---------------------------------------------------------------------------


@dataclass
class Mounts:
    """Host directories the container sees, each with where it appears and whether it can be written."""

    libs: Path
    mcpservers: Path
    repo: Path
    config: Path
    agent: Path
    results: Path
    db: Path
    findings: Optional[Path] = None  # only a verifier or classifier session has one

    def as_list(self) -> list:
        """(host path, container path, writable) for every mount. This list is ALL the container can see of the host."""
        mounts = [
            (self.libs, f"{C_REVIEWER}/libs", False),
            (self.mcpservers, f"{C_REVIEWER}/mcpservers", False),
            (self.repo, C_REPO, False),
            (self.config, C_CONFIG, False),
            (self.agent, C_AGENT, True),
            (self.results, C_RESULTS, True),
            (self.db, C_DB, True),
        ]
        if self.findings is not None:
            mounts.append((self.findings, C_FINDINGS, False))
        return mounts


def harness_db_dir(checkout: Path) -> Path:
    return checkout.joinpath(*HARNESS_DB_SUBDIR)


def reviewers_file(model: str, context_length: int, load_on_demand: bool) -> dict:
    """A reviewers file with one reasoning reviewer. The model and context are arguments, never constants, so a bigger
    machine changes the command line and nothing else."""
    return {
        # The server refuses a reviewers file without a `rubric` key; an empty one is valid until a classifier is active.
        "rubric": [],
        "reviewers": [
            {
                "name": "general",
                "model": model,
                "prompt": "You review a Python change for real defects: wrong results, crashes on valid input, injection, resource leaks.",
                "active": True,
                "type": ["reasoning"],
                "loadOnDemand": load_on_demand,
                "contextLength": context_length,
            }
        ]
    }


def _within(path: Path, folder: Path) -> bool:
    """Whether `path` is `folder` or inside it (both resolved by the caller)."""
    return path == folder or folder in path.parents


def prompt_file_problem(source: Path, home: Path, checkouts: tuple) -> Optional[str]:
    """Why a reviewer's `promptFile` must not be copied into the container; None when it may be.

    The copy can be read inside the container, so this guards as much as `unsafe_mounts` does. A prompt is a `.md` file (which
    the answer key, a JSON file, is not). It may not come from `~/.claude` or `~/.pi`, and inside a checkout only from the two
    prompt folders: elsewhere there are old logs and notes on the seeded bugs. Symlinks and `..` are resolved first."""
    resolved = source.resolve()
    if resolved.suffix.lower() != ".md":
        return "is not a .md file"
    for label, folder in (("~/.claude", home / ".claude"), ("~/.pi", home / ".pi")):
        if _within(resolved, folder.resolve()):
            return f"is under {label}"
    for checkout in checkouts:
        root = checkout.resolve()
        if _within(resolved, root) and not any(_within(resolved, root.joinpath(*parts)) for parts in PROMPT_SUBDIRS):
            allowed = " or ".join("/".join(parts) + "/" for parts in PROMPT_SUBDIRS)
            return f"is in the checkout {root} but not under {allowed}"
    if _is_not_a_regular_file(resolved):
        return "is not a regular file"
    return None


def _is_not_a_regular_file(path: Path) -> bool:
    """Whether something is at `path` that is not a plain file: asked BEFORE reading, because reading a named pipe waits for a
    writer, possibly for ever, and a `stop` signal is not watched yet. A missing file is left to whoever reads it."""
    return path.exists() and not path.is_file()


def _container_lacks(reviewers: object, rubric: object, where: str) -> None:
    """Raises ValueError for a file a config names that the driver does not copy into the container."""
    for reviewer in reviewers if isinstance(reviewers, list) else []:
        if isinstance(reviewer, dict) and "translator" in reviewer:
            raise ValueError(f"{where}: reviewer {reviewer.get('name')!r} uses translator, a file the container does not have")
    for entry in rubric if isinstance(rubric, list) else []:  # anything else is left for the validator to name, not iterated
        if isinstance(entry, dict) and "definitionFile" in entry:
            raise ValueError(f"{where}: a rubric entry uses definitionFile, a file the container does not have")


def has_active(reviewers: dict, review_type: str) -> bool:
    return any(r.get("active", True) and review_type in r["type"] for r in reviewers["reviewers"])


def reviewers_from_config(path: Path, review_type: str = "reasoning") -> dict:
    """The reviewers file out of a `harness/live-tests/configs/*.json` config, so the models and contexts the harness
    already records are not copied into a second place.

    A live-test config is a reviewers file plus a `stages` key the server would reject, so `stages` is dropped and the rest
    is checked by the server's own validator. A `promptFile` is made an absolute path (it is relative to the config's folder),
    to be copied into the container with the session; `translator` and `definitionFile` name files the driver does not copy, so
    a config that uses one is refused rather than run half-configured. The config needs an active reviewer of `review_type`;
    its other reviewers are passed on as they are (choosing among them is the caller's job). Raises ValueError with the reason."""
    if _is_not_a_regular_file(Path(path)):
        raise ValueError(f"cannot read {path}: it is not a regular file")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("reviewers"), list) or not data["reviewers"]:
        raise ValueError(f"{path} has no 'reviewers' list")
    # The harness fills in its own rubric for a config that has none; the server does not, and refuses a file without the key.
    # An empty one is valid until a classifier reviewer is active (the validator then says so).
    rubric = data.get("rubric", [])
    _container_lacks(data["reviewers"], rubric, str(path))
    folder = Path(path).resolve().parent
    entries = []
    for entry in data["reviewers"]:
        if isinstance(entry, dict):
            entry = dict(entry)
            raw = entry.get("promptFile")
            if isinstance(raw, str) and raw.strip():
                source = Path(os.path.expanduser(raw.strip()))
                source = (source if source.is_absolute() else folder / source).resolve()
                problem = prompt_file_problem(source, Path.home(), (REPO_ROOT,))
                if problem:
                    raise ValueError(f"{path}: reviewer {entry.get('name')!r}: promptFile {source} {problem}")
                entry["promptFile"] = str(source)
        entries.append(entry)
    reviewers = {"rubric": rubric, "reviewers": entries}
    try:
        validate_reviewers(reviewers, path=str(path))
    except ConfigError as exc:
        raise ValueError(str(exc)) from exc
    # The review tool refuses a review type with no reviewer for it with a plain "Error:" result, after a container has started.
    if not has_active(reviewers, review_type):
        raise ValueError(f"{path} has no active {review_type} reviewer: the review tool would refuse a {review_type} review")
    return reviewers


def container_url(host_url: str, alias: str = CONTAINER_HOST_ALIAS) -> str:
    """`host_url` as the container must spell it.

    A loopback address (`localhost`, `127.0.0.1`...) means "this machine", which inside a container is the container, so
    only the host name is swapped for `alias`; the scheme, port and path are kept, so an LM Studio on another port works
    without a flag. Any other host is already reachable from the container and comes back unchanged. Raises ValueError for
    a URL whose port is not a number."""
    parts = urllib.parse.urlsplit(host_url)
    if (parts.hostname or "").lower() not in LOOPBACK_HOSTS:
        return host_url
    userinfo, at = parts.netloc.rpartition("@")[:2]
    port = f":{parts.port}" if parts.port is not None else ""
    return urllib.parse.urlunsplit(parts._replace(netloc=f"{userinfo}{at}{alias}{port}"))


def models_in(reviewers: dict, review_type: Optional[str] = None) -> list:
    """The distinct models of the active reviewers (of `review_type`, when given), in file order."""
    return list(dict.fromkeys(r["model"] for r in reviewers["reviewers"]
                              if r.get("active", True) and (review_type is None or review_type in r["type"])))


def models_file(model: str, context_length: int, lmstudio_url: str) -> dict:
    """Pi's models.json: LM Studio as an OpenAI-compatible endpoint. The key is a placeholder; LM Studio ignores it."""
    return {
        "providers": {
            PROVIDER: {
                "baseUrl": lmstudio_url,
                "api": "openai-completions",
                "apiKey": "lm-studio",
                "models": [{"id": model, "contextWindow": context_length}],
            }
        }
    }


def pi_arguments(model: str, prompt: str) -> list:
    """Pi's own arguments for one non-interactive run with a JSON event stream."""
    tools = ",".join([*READ_ONLY_TOOLS, f"mcp__{MCP_SERVER_NAME}__*"])
    return [
        "--mode", "json", "--no-session", "--thinking", "off",
        "--provider", PROVIDER, "--model", model,
        "--tools", tools,
        # Nothing from outside the run shapes it: no AGENTS.md/CLAUDE.md, skills or prompt templates found on disk.
        "--no-context-files", "--no-skills", "--no-prompt-templates",
        "-p", "--", prompt,
    ]


def mcp_file(lmstudio_url: str, mcp_timeout: Optional[int] = None, lmstudio_timeout: Optional[int] = None,
             review_id: str = "") -> dict:
    """Pi's mcp.json, registering the reviewer MCP server: the file `pi mcp add` would write, in the same shape.

    The driver writes it into the agent scratch folder instead of running `pi mcp add` in the container, so the LM Studio
    URL, which can carry a user name and password, is never on a command line (host `ps`, `docker inspect`, --dry-run).

    Two different limits, each left out when None so its own default holds (60 seconds for both): `mcp_timeout` is Pi's
    `timeout` for one request to this server, which a whole `review` call must fit in while the server sends no progress;
    `lmstudio_timeout` is the reviewer's limit for one completion request to LM Studio. A session passes both (its own time
    limit, unless told otherwise), so Pi does not give up and call again while a review is still running.

    The results folder and the review ID are the review tool's own settings: the model that calls it never sees or copies
    either."""
    env = {
        "REVIEWERS_FILE": f"{C_CONFIG}/reviewers.json",
        "CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB": f"{C_DB}/{DB_FILE_NAME}",
        "CLAUDE_RUNWAY_LMSTUDIO_URL": lmstudio_url,
        "REVIEW_OUTPUT_DIR": C_RESULTS,
    }
    if review_id:
        env["REVIEW_ID"] = review_id
    if lmstudio_timeout is not None:
        env["CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS"] = str(lmstudio_timeout)
    server = {
        "command": C_VENV_PYTHON,
        "args": [f"{C_REVIEWER}/mcpservers/codebase_reviewer_mcp_server.py"],
        "env": env,
        "exposure": "direct",
    }
    if mcp_timeout is not None:
        server["timeout"] = mcp_timeout
    return {"mcpServers": {MCP_SERVER_NAME: server}}


def display_url(url: str) -> str:
    """`url` safe to print: a user name and password in it become `***`. Nothing else is changed."""
    parts = urllib.parse.urlsplit(url)
    _userinfo, at, host = parts.netloc.rpartition("@")
    if not at:
        return url
    return urllib.parse.urlunsplit(parts._replace(netloc=f"***@{host}"))


def user_arguments() -> list:
    """`docker run`'s `--user` for this system. On Linux the files a container writes into a mounted folder would be root's
    and the driver could not remove them. Windows has no such ids (and Docker Desktop maps ownership itself)."""
    getuid, getgid = getattr(os, "getuid", None), getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return []
    return ["--user", f"{getuid()}:{getgid()}"]


def docker_run_command(
    *, docker: str, image: str, name: str, mounts: Mounts, model: str, prompt: str,
    memory: str, cpus: str, pids_limit: int, time_limit_seconds: int,
) -> list:
    """The whole `docker run` command line, as a list (never a shell string, so no quoting problem and no injection).

    It carries no LM Studio URL: that is in the models.json and mcp.json the driver wrote into the agent scratch folder."""
    command = [
        # --init: a small init process is PID 1, and it forwards signals and reaps children. Without it the entrypoint,
        # `timeout`, would be PID 1, which the kernel gives no default signal handling.
        docker, "run", "--rm", "--init", "--name", name,
        "--memory", memory, "--cpus", cpus, "--pids-limit", str(pids_limit),
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        # Docker Desktop already knows this name; plain Linux Docker needs to be told.
        "--add-host", "host.docker.internal:host-gateway",
        "--workdir", "/work",
    ]
    command += user_arguments()
    command += [
        "--env", f"PI_CODING_AGENT_DIR={C_AGENT}",
        "--env", "PI_TELEMETRY=0", "--env", "PI_OFFLINE=1", "--env", "PI_SKIP_VERSION_CHECK=1",
        "--env", "HOME=/tmp",
    ]
    # --mount rather than -v: a Windows path (C:\...) contains the colon that -v uses to split its fields.
    for host, inside, writable in mounts.as_list():
        spec = f"type=bind,source={host},target={inside}"
        command += ["--mount", spec if writable else spec + ",readonly"]
    command += ["--entrypoint", C_TIMEOUT, image, f"--kill-after={KILL_AFTER_SECONDS}", str(time_limit_seconds),
                "pi", *pi_arguments(model, prompt)]
    return command


def unsafe_mounts(mounts: Mounts, home: Path, checkout: Path) -> list:
    """Reasons a mount list must not be used; empty when it is fine.

    The kit promises never to expose the checkout, `~/.claude` or `~/.pi`. A mount overlaps one of them when it is the
    folder, a folder above it (which contains it) or a folder inside it. There are two exceptions inside the checkout. The
    reviewer's own code, `libs/` and `mcpservers/`, read-only, because the MCP server cannot run without them. And the
    harness database folder, `harness/live-tests/logs/db/`, writable, as the database mount only: not `logs/` above it,
    whose older logs can print the answer key, and not a folder inside it. All three must be the checkout's real folders: a
    symlink in their place (or above the database folder) could point anywhere outside the three roots, and Docker would
    mount whatever it points to."""
    checkout = checkout.resolve()
    code = {checkout / "libs", checkout / "mcpservers"}
    database = harness_db_dir(checkout)
    forbidden = {"the checkout": checkout, "~/.claude": (home / ".claude").resolve(), "~/.pi": (home / ".pi").resolve()}
    problems = []
    for name, host in (("libs", mounts.libs), ("mcpservers", mounts.mcpservers)):
        if host.is_symlink() or host.resolve() != checkout / name:
            problems.append(f"{host} is not the checkout's own {name}/ folder ({checkout / name})")
    if mounts.db.is_symlink() or mounts.db.resolve() != database:
        problems.append(f"{mounts.db} is not the checkout's own harness database folder ({database})")
    for host, inside, writable in mounts.as_list():
        resolved = host.resolve()
        if resolved in code and not writable:
            continue
        for label, path in forbidden.items():
            if label == "the checkout" and inside == C_DB and resolved == database:
                continue
            if resolved == path or path in resolved.parents or resolved in path.parents:
                problems.append(f"{host} overlaps {label} ({path})")
    return problems


def unsafe_log(path: Path, home: Path) -> Optional[str]:
    """Why `--log` must not write to `path`, or None when it may.

    The log is written by the driver on the host, so the container's mount checks do not cover it. Pi's own folder
    (`~/.pi`, where its credentials live) and `~/.claude` are never written to, whatever the user typed; the checkout is
    allowed, since a log kept beside the harness is natural. Symlinks and `..` are resolved first."""
    resolved = path.resolve()
    for label, protected in (("~/.claude", (home / ".claude").resolve()), ("~/.pi", (home / ".pi").resolve())):
        if resolved == protected or protected in resolved.parents:
            return f"--log {path} is inside {label} ({protected}), which the driver never writes to"
    return None


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


def run_capped(
    command: list, tracker: CapTracker, *, spawn: Callable, kill_container: Callable, stderr_path: Path,
    on_event: Callable[[dict], None] = lambda event: None, clock: Callable[[], float] = time.monotonic,
    stop: Optional[threading.Event] = None, remove_container: Callable[[], Optional[str]] = lambda: None,
) -> RunResult:
    """Starts `command`, reads its JSON stream and stops it when a cap is reached or `stop` is set.

    `spawn(command, stderr_file)` returns an object with `stdout` (a binary stream with `read1`), `wait(timeout)` and `kill()`;
    `kill_container()` stops the container itself (killing the `docker` client alone would leave it running), and
    `remove_container()` removes whatever container is left, returning what is wrong when it cannot tell that none is. All three
    are arguments so a test can fake them. `stop` is how a caller ends a run: no operating system signal is involved, since on
    Windows a killed process runs no cleanup. A `stop` that is already set runs nothing."""
    result = RunResult()
    if stop is not None and stop.is_set():
        result.stopped_by = "stopped"
        return result
    chunks: queue.Queue = queue.Queue()
    with open(stderr_path, "wb") as stderr_file:
        process = spawn(command, stderr_file)
        stopped_by: Optional[str] = None

        def clean_up() -> None:
            """After a run that did not end cleanly. The client goes first, and is waited for, so that one the daemon has not
            been asked by yet never creates the container; then the container; then whatever the client left behind, because
            `docker kill` fails on a container that does not exist yet and does nothing to one that was created and never
            started. What the client asked for before it exited is out of reach here (see `stop_client`)."""
            problems: list = []
            try:
                if not stop_client(process):
                    problems.append(f"the docker client did not exit within {CLIENT_EXIT_SECONDS} s of being killed, so it may "
                                    "still start the container; check `docker ps -a`")
            finally:
                # In a `finally`: a second Ctrl+C that ends the wait must not leave the container running.
                kill_container()
                problems.append(remove_container())
                for problem in problems:
                    if problem and problem not in result.failures:
                        print(f"warning: {problem}", file=sys.stderr)
                        result.failures.append(problem)

        # Everything from here to the end of the wait is one protected block: the container must stop on EVERY way out, not
        # only when a cap trips (see the `except` below). Starting it straight after `spawn` leaves no gap before it.
        try:

            def read_stdout() -> None:
                # A thread, so the wall clock can fire while Pi is silent: a blocking read on this thread would wait forever.
                try:
                    while True:
                        chunk = process.stdout.read1(65536)
                        if not chunk:
                            break
                        chunks.put(chunk)
                except (OSError, ValueError):
                    pass  # the pipe was closed under us: the container is gone
                chunks.put(None)

            threading.Thread(target=read_stdout, daemon=True).start()

            splitter = LineSplitter()
            ended = False
            while not ended and stopped_by is None:
                if stop is not None and stop.is_set():
                    stopped_by = "stopped"
                    break
                if tracker.out_of_time():
                    stopped_by = "timeout"
                    break
                try:
                    chunk = chunks.get(timeout=max(0.05, min(0.5, tracker.remaining())))
                except queue.Empty:
                    continue
                lines = splitter.flush() if chunk is None else splitter.feed(chunk)
                ended = chunk is None
                for line in lines:
                    event = parse_event(line)
                    if event is None:
                        continue
                    # Count first, then report, so what `on_event` prints (turn 1, turn 2...) already includes this event.
                    stopped_by = tracker.observe(event)
                    on_event(event)
                    if stopped_by:
                        break
            if stopped_by:
                clean_up()
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                clean_up()
                exit_code = process.wait(timeout=30)
            if stopped_by is None:
                result.exit_code = exit_code
                if not tracker.settled or exit_code != 0:
                    # The stream ended without a clean finish. A `docker` client that was killed or crashed ends its output
                    # while the container goes on running with no caps, so make sure it is gone (nothing to do if it is).
                    clean_up()
        except BaseException:
            # An exception (Ctrl+C, a failed log write) while reading or waiting: without this the container would run on
            # uncapped while the caller deletes the folders it has mounted.
            clean_up()
            raise

    # A settle that says `aborted` is not a finished run, whatever the container's exit code was: it must not read as a pass.
    # Nor is a settle followed by a failing container (Pi settles before it shuts its MCP server down, so an OOM kill or a
    # crash during that shutdown still ends nonzero): only a settle and a clean container exit count as completed. A stream
    # that ends without a settle is an error even when the container exits 0, since Pi never said it had finished. An exit
    # with the `timeout` command's code is the backstop in the container ending a run that outlived its limit. A Pi that
    # ignores the backstop's first signal is killed KILL_AFTER_SECONDS later, and the exit is then 137, which stays `error`
    # on purpose: an out-of-memory kill gives 137 too. The driver's own time cap is due BACKSTOP_MARGIN_SECONDS before the
    # backstop, so while the driver is running its cap is what should end a run that is out of time.
    result.stopped_by = stopped_by or (
        "timeout" if result.exit_code == TIMEOUT_EXIT_CODE
        else "aborted" if tracker.aborted
        else "completed" if tracker.settled and result.exit_code == 0
        else "error"
    )
    result.turns, result.tool_calls = tracker.turns, tracker.tool_calls
    result.review_calls, result.review_errors = tracker.review_calls, tracker.review_errors
    result.call_errors = list(tracker.call_errors)
    result.elapsed_seconds = round(tracker.elapsed(), 1)
    result.final_text, result.review_text = tracker.final_text, tracker.review_text
    try:
        result.stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    except OSError:
        pass
    return result


def _docker(docker: str, *args: str) -> Optional[subprocess.CompletedProcess]:
    """`docker <args>`, or None when it could not run or did not answer in time. A docker daemon that does not answer must
    not hang the driver on its way out."""
    try:
        return subprocess.run([docker, *args], capture_output=True, check=False, timeout=DOCKER_COMMAND_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return None


def stop_client(process) -> bool:
    """Kills the `docker` client `process` (an object with `kill()` and `wait(timeout)`) and waits until it has exited; False
    when it has not within CLIENT_EXIT_SECONDS. Do it BEFORE removing the client's container: `kill()` returns while the
    client may still be alive, and a live one can still ask the daemon to create or start the container after a removal has
    looked and found nothing. What it asked for before it exited the daemon still carries out: a create sent just before the
    kill can leave a container that is never started, which can appear after a removal has looked."""
    with contextlib.suppress(OSError):
        process.kill()
    try:
        process.wait(timeout=CLIENT_EXIT_SECONDS)
    except subprocess.TimeoutExpired:
        return False
    return True


def kill_container(docker: str, name: str) -> None:
    if _docker(docker, "kill", name) is None:
        print(f"warning: docker kill {name} did not run or did not answer in {DOCKER_COMMAND_TIMEOUT_SECONDS} s", file=sys.stderr)


def remove_container(docker: str, name: str, *, sleep: Callable[[float], None] = time.sleep) -> Optional[str]:
    """Removes the container `name`, running or only created, and checks that it is gone. Returns None when it is, else what to
    tell the user. `--rm` removes a container that ran and ended by itself, and may still be doing so when this is called, hence
    the retries."""
    for attempt in range(REMOVE_ATTEMPTS):
        _docker(docker, "rm", "--force", name)
        listed = _docker(docker, "ps", "--all", "--quiet", "--filter", f"name={name}")
        if listed is not None and listed.returncode == 0 and not listed.stdout.strip():
            return None
        if attempt + 1 < REMOVE_ATTEMPTS:
            sleep(REMOVE_PAUSE_SECONDS)
    return f"could not confirm that the container {name} is gone; check `docker ps -a`"


class LaunchError(SessionError):
    """`docker` could not be started at all (not installed, not on PATH): nothing ran, so it is a usage error, not a failed run."""


def _spawn(command: list, stderr_file) -> subprocess.Popen:
    try:
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file, stdin=subprocess.DEVNULL)
    except OSError as exc:
        raise LaunchError(f"could not start {command[0]!r}: {exc}") from exc


def _remove_tree(directory: Path) -> None:
    """Removes the scratch directory. Git's object files are read-only, which Windows will not delete without a chmod."""
    def retry(function, path, _error):
        os.chmod(path, stat.S_IWRITE)
        function(path)
    shutil.rmtree(directory, onexc=retry)


class ChangeRepoError(SessionError):
    """The repository of a change could not be built: no such checkout, a tag it lacks, or a `git` that failed. Nothing ran, so it
    is a usage error (exit 2), not a failed run."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """`git -C repo ...` with the user's own git configuration left out. Raises ChangeRepoError only when git could not run at all
    (missing, timed out); a nonzero exit is the caller's to read.

    Left out (`GIT_CONFIG_GLOBAL`, `GIT_CONFIG_SYSTEM`) because this builds a repository meant to hold two commits and nothing
    else, and it is read by a container: a global `core.hooksPath` would run a hook on the checkout, `core.autocrlf` would
    rewrite the files the reviewer reads, and `init.templateDir` would add files. `GIT_TERMINAL_PROMPT=0`: never ask a person."""
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_TERMINAL_PROMPT": "0"}
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              check=False, timeout=GIT_TIMEOUT_SECONDS, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ChangeRepoError(f"could not run git ({type(exc).__name__}: {exc})") from exc


def _first_line(done: subprocess.CompletedProcess) -> str:
    """The first line git said, from stderr (where its errors go), else stdout; a fixed phrase when it said nothing."""
    lines = (done.stderr.strip() or done.stdout.strip()).splitlines()
    return lines[0] if lines else "no message"


def change_tag(change: str, key_path: Optional[Path] = None):
    """The answer key's `LiveTag` for `change` (`change-6`, written without the branch's prefix).

    The key is read here, on the host, only to learn which changes exist; it is never copied or mounted (see `unsafe_mounts` and
    `build_change_repo`). Raises ValueError naming the known changes for one that is not listed, and `LiveMetadataError` (also a
    ValueError) when the key itself cannot be read."""
    tags = scoring.load_answer_key(key_path)
    tag = next((t for t in tags if t.tag_id == change), None)
    if tag is None:
        raise ValueError(f"unknown change {change!r}; the answer key lists {', '.join(t.tag_id for t in tags)}")
    return tag


def check_checkout(checkout: Path, tag) -> None:
    """Raises ChangeRepoError unless `checkout` is a git repository that has the branch's base and `tag` as commits.

    Run before anything slow (the model preflight, `--build`), so a clone that does not carry the live branch stops at once with
    the exact `git fetch` that gets it, as the other harness runners do."""
    probe = _git(checkout, "rev-parse", "--git-dir")
    if probe.returncode != 0:
        raise ChangeRepoError(f"{checkout} cannot be used as the checkout that has the {scoring.PREFIX}* tags (--repo): {_first_line(probe)}")
    try:
        missing = scoring.missing_refs(checkout, (tag,))
    except (OSError, subprocess.SubprocessError) as exc:
        raise ChangeRepoError(f"could not run git ({type(exc).__name__}: {exc})") from exc
    if missing:
        raise ChangeRepoError(scoring.missing_refs_message(checkout, missing))


def build_change_repo(parent: Path, checkout: Path, tag) -> Path:
    """A repository at `parent / "repo"` that holds the branch's base and `tag`'s change, and nothing else of `checkout`.

    `git fetch` of just the two tags into an empty repository brings only the commits reachable from them (the base, and the one
    commit on it) with their files. The rest of the checkout's history, its branches, its other tags and every file that is not on
    the branch never reach the folder the container is given: the answer key, which sits in `harness/` and not on the branch,
    least of all. The change is then checked out (detached), so the repository is a real checkout of that change: the reviewer
    reads the repository's own review rules (`.github/instructions/`) from the working tree, as it does in any checkout it is
    pointed at, and Pi's read-only tools see the files. Raises ChangeRepoError."""
    check_checkout(checkout, tag)
    repo = parent / "repo"
    repo.mkdir()
    for args in (
        ("init", "-q"),
        # --no-tags: only the two refs named, not any other tag that happens to point at what they bring.
        ("fetch", "-q", "--no-tags", str(checkout), *(f"refs/tags/{ref}:refs/tags/{ref}" for ref in (scoring.BASE_REF, tag.head))),
        ("checkout", "-q", "--detach", f"refs/tags/{tag.head}"),
    ):
        done = _git(repo, *args)
        if done.returncode != 0:
            raise ChangeRepoError(f"git {args[0]} failed while building the repository of {tag.head} from {checkout}: {_first_line(done)}")
    return repo


def copy_prompt_files(reviewers: dict, config_dir: Path, checkouts: tuple) -> dict:
    """The reviewers file the container gets: `reviewers`, with each `promptFile` copied under `config_dir` and renamed to
    match. Raises SessionError for one that must not be copied (see `prompt_file_problem`) or cannot be.

    `reviewers` may come from a caller that built it itself and not from `reviewers_from_config`, so the checks are made
    here, where the copy is."""
    try:
        _container_lacks(reviewers.get("reviewers"), reviewers.get("rubric"), "the session's reviewers")
    except ValueError as exc:
        raise SessionError(str(exc)) from exc
    copied = copy.deepcopy(reviewers)
    entries = copied.get("reviewers")
    for number, reviewer in enumerate(entries if isinstance(entries, list) else []):
        if not isinstance(reviewer, dict) or "promptFile" not in reviewer:
            continue
        source = Path(os.path.expanduser(str(reviewer["promptFile"])))
        if not source.is_absolute():
            raise SessionError(f"promptFile {str(source)!r} is relative: a session needs an absolute path (reviewers_from_config "
                               "makes one from a config's own folder)")
        problem = prompt_file_problem(source, Path.home(), checkouts)
        if problem:
            raise SessionError(f"promptFile {source} {problem}")
        name = f"prompts/{number}.md"  # relative: the server resolves it against the reviewers file's own folder
        target = config_dir / name
        target.parent.mkdir(exist_ok=True)
        try:
            target.write_bytes(source.read_bytes())
        except OSError as exc:
            raise SessionError(f"cannot copy promptFile {source}: {exc}") from exc
        reviewer["promptFile"] = name
    return copied


def copy_findings_file(source: Path, review_type: str, folder: Path) -> None:
    """Copies the findings file `source` to `folder/findings.json` and checks the COPY, which is what the container is given,
    with the review tool's own reader. Raises SessionError.

    A copy, because the source may sit under `harness/live-tests/logs/`, which no mount may reach, and because the container
    then gets this one file and nothing beside it. The reader refuses any key it does not know, so a file that is not a
    findings file (the answer key, say) never gets in, and it never repeats a value from the file."""
    # libs/ is on sys.path since this module was imported; imported here to keep the driver's own imports light.
    from reasoning_reviewer_lib import ReviewError
    from reviewer_files_lib import MAX_FINDINGS_FILE_BYTES, read_findings_file
    if not source.is_file():
        raise SessionError(f"findings file {source} is not a file")
    folder.mkdir()
    target = folder / "findings.json"
    try:
        with open(source, "rb") as handle:
            target.write_bytes(handle.read(MAX_FINDINGS_FILE_BYTES + 1))  # one byte over: the reader refuses it as too long
        read_findings_file(str(target), review_type)
    except OSError as exc:
        raise SessionError(f"cannot read the findings file {source}: {exc}") from exc
    except ReviewError as exc:
        # The reader names the copy; the user knows the file by its own path.
        raise SessionError(str(exc).replace(repr(str(target)), repr(str(source))).replace(str(target), str(source))) from exc


def prepare_scratch(scratch: Path, checkout: Path, tag, model: str, reviewers: dict, context_length: int, lmstudio_url: str,
                    mcp_timeout: Optional[int] = None, lmstudio_timeout: Optional[int] = None, *, review_type: str = "reasoning",
                    findings_file: Optional[Path] = None, review_id: str = "") -> Mounts:
    """Fills `scratch` with everything the container mounts and returns the Mounts. Raises SessionError (ChangeRepoError for
    the repository). The one mount that is not in `scratch` or this checkout's code is the harness database folder, which is
    only named here: `ensure_harness_db` creates it, just before a container starts."""
    repo = build_change_repo(scratch, checkout, tag)
    config, agent, results = scratch / "config", scratch / "agent", scratch / "results"
    for directory in (config, agent, results):
        directory.mkdir()
    in_container = copy_prompt_files(reviewers, config, (REPO_ROOT, checkout))
    try:
        # The server's own validator, on the file as it will see it, so a problem is found before a container starts.
        validate_reviewers(in_container, base_dir=config, path="the session's reviewers")
    except ConfigError as exc:
        raise SessionError(str(exc)) from exc
    if not has_active(in_container, review_type):
        raise SessionError(f"the session's reviewers have no active {review_type} reviewer")
    (config / "reviewers.json").write_text(json.dumps(in_container, indent=2), encoding="utf-8")
    (agent / "models.json").write_text(json.dumps(models_file(model, context_length, lmstudio_url), indent=2), encoding="utf-8")
    (agent / "mcp.json").write_text(
        json.dumps(mcp_file(lmstudio_url, mcp_timeout, lmstudio_timeout, review_id), indent=2), encoding="utf-8")
    findings = None
    if findings_file is not None:
        findings = scratch / "findings"
        copy_findings_file(Path(findings_file), review_type, findings)
    return Mounts(REPO_ROOT / "libs", REPO_ROOT / "mcpservers", repo, config, agent, results, harness_db_dir(REPO_ROOT), findings)


# Held while the database is created, so a second session of this process cannot start its container on a half-made file.
_DB_LOCK = threading.Lock()


def ensure_harness_db(folder: Path) -> Path:
    """Creates the harness database in `folder` when it is missing, and otherwise leaves it completely alone. Returns its path.

    It has to exist before a container starts: containers creating it together lost records (epic "codebase-reviewer
    harness: an orchestrator...", shared database rules). An existing file is NOT opened, because a container of another
    session may be writing it: host writes during a run lost records and damaged the database there, and closing a host
    connection to a WAL database can checkpoint it, unaware of locks held inside Docker's virtual machine (not tested, so
    not risked). `connect` creates the schema and adds no record."""
    path = folder / DB_FILE_NAME
    with _DB_LOCK:
        if not path.exists():
            from reviewer_db_lib import connect  # libs/ is on sys.path since this module was imported
            try:
                connect(path).close()
            except (OSError, sqlite3.Error) as exc:
                raise SessionError(f"cannot create the harness database {path}: {exc}") from exc
    return path


def read_results(folder: Path) -> tuple:
    """`(results, problems)`: the review tool's results files in `folder`, as parsed JSON, and what could not be read.

    The folder is one the container can write, so only a regular file is read: a link left in it must not make the host read
    some other file into the summary."""
    results: list = []
    problems: list = []
    for path in sorted(folder.glob("*.json")):  # the tool's temporary files end in .tmp
        if path.is_symlink() or not path.is_file():
            problems.append(f"results file {path.name} is not a regular file, so it was not read")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"results file {path.name} cannot be read ({type(exc).__name__}: {exc})")
            continue
        if isinstance(data, dict):
            results.append(data)
        else:
            problems.append(f"results file {path.name} is not a JSON object")
    return results, problems


def session_failures(tracker: CapTracker, results: list, review_type: str) -> list:
    """Why a run Pi finished is still not a passing session, beyond the calls `tracker.call_errors` already names."""
    failures = []
    if not tracker.good_calls and not tracker.call_errors:
        failures.append("no review call with this session's arguments ran and succeeded (a call Pi refuses before running "
                        "it does not count)")
    types = [str(r.get("type")) for r in results]
    if review_type not in types:
        failures.append(f"no results file of the {review_type} review came back")
    others = sorted({t for t in types if t != review_type})
    if others:
        failures.append(f"a results file of another review type came back: {', '.join(others)}")
    return failures


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionSpec:
    """Everything one session needs. Whoever makes it has chosen the reviewers and loaded what must be loaded: the session
    runs one review of `review_type` on `change`, in one container."""

    model: str  # Pi's calling model, as LM Studio lists it
    reviewers: dict  # the reviewers file's content (`rubric` and `reviewers`); a promptFile in it is an absolute host path
    lmstudio_url: str  # LM Studio as the CONTAINER sees it (see `container_url`)
    review_type: str = "reasoning"
    change: str = DEFAULT_CHANGE
    checkout: Optional[Path] = None  # a checkout with the branch's tags; None is this one
    findings_file: Optional[Path] = None  # required for a verifier or classifier session, refused for a reasoning one
    review_id: str = ""  # groups the run records of one review; stored by the review tool itself
    context_length: int = DEFAULT_CONTEXT_LENGTH  # Pi's own model entry; the reviewers carry theirs
    prompt: Optional[str] = None  # None is `default_prompt(review_type)`; the call check applies to any prompt
    max_turns: int = DEFAULT_MAX_TURNS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    mcp_timeout: Optional[int] = None  # None is `timeout`
    lmstudio_timeout: Optional[int] = None  # None is `timeout`
    image: str = DEFAULT_IMAGE
    docker: str = "docker"
    memory: str = "4g"
    cpus: str = "2"
    pids_limit: int = 256


@dataclass
class PreparedSession:
    name: str  # the container's
    command: list
    mounts: Mounts
    scratch: Path
    expected: ExpectedCall


def _check_spec(spec: SessionSpec) -> None:
    if spec.review_type not in ROLES:
        raise SessionError(f"unknown review type {spec.review_type!r}; the types are {', '.join(ROLES)}")
    if spec.review_type == "reasoning" and spec.findings_file is not None:
        raise SessionError("a reasoning session takes no findings file: the review starts from the diff")
    if spec.review_type != "reasoning" and spec.findings_file is None:
        raise SessionError(f"a {spec.review_type} session needs a findings file: without one the review tool runs the earlier "
                           "stages too, and a session runs one reviewer")
    if min(spec.max_turns, spec.max_tool_calls, spec.timeout) < 1:
        raise SessionError("max_turns, max_tool_calls and timeout must each be 1 or more")
    if any(value is not None and value < 1 for value in (spec.mcp_timeout, spec.lmstudio_timeout)):
        raise SessionError("mcp_timeout and lmstudio_timeout must each be 1 or more")
    try:
        read_settings({"REVIEW_ID": spec.review_id})  # the server's own rule for a review ID
    except ConfigError as exc:
        raise SessionError(str(exc)) from exc


@contextlib.contextmanager
def prepared_session(spec: SessionSpec):
    """Builds one session's files and `docker run` command and removes the files again on the way out. Nothing is started and
    the harness database is not touched. Raises SessionError."""
    _check_spec(spec)
    try:
        tag = change_tag(spec.change)
    except ValueError as exc:
        raise SessionError(str(exc)) from exc
    checkout = Path(spec.checkout).expanduser().resolve() if spec.checkout else REPO_ROOT
    scratch = Path(tempfile.mkdtemp(prefix="claude-runway-pi-"))
    try:
        mounts = prepare_scratch(
            scratch, checkout, tag, spec.model, spec.reviewers, spec.context_length, spec.lmstudio_url,
            spec.timeout if spec.mcp_timeout is None else spec.mcp_timeout,
            spec.timeout if spec.lmstudio_timeout is None else spec.lmstudio_timeout,
            review_type=spec.review_type, findings_file=spec.findings_file, review_id=spec.review_id)
        problems = unsafe_mounts(mounts, Path.home(), REPO_ROOT)
        if problems:
            raise SessionError("refusing to start, a mount would expose the host:\n  " + "\n  ".join(problems))
        findings = C_FINDINGS_FILE if spec.findings_file is not None else None
        prompt = fill_prompt(spec.prompt or default_prompt(spec.review_type), tag.head, findings)
        # The random part: one caller may start several sessions in the same second, from the same process.
        name = f"claude-runway-pi-{os.getpid()}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        command = docker_run_command(
            docker=spec.docker, image=spec.image, name=name, mounts=mounts, model=spec.model, prompt=prompt,
            memory=spec.memory, cpus=spec.cpus, pids_limit=spec.pids_limit,
            time_limit_seconds=spec.timeout + BACKSTOP_MARGIN_SECONDS)
        yield PreparedSession(name, command, mounts, scratch, ExpectedCall(spec.review_type, tag.head, findings))
    finally:
        _remove_tree(scratch)


def run_session(spec: SessionSpec, *, stop: Optional[threading.Event] = None,
                on_event: Callable[[dict, CapTracker], None] = lambda event, tracker: None,
                spawn: Optional[Callable] = None) -> RunResult:
    """Runs one session and returns how it went, with the review tool's results. Setting `stop` ends it and removes its
    container. Raises SessionError when it could not be set up; nothing ran then. The run's record is in the harness database
    (the driver does not read it: reading while a container writes has not been tested)."""
    with prepared_session(spec) as session:
        ensure_harness_db(session.mounts.db)
        tracker = CapTracker(spec.max_turns, spec.max_tool_calls, spec.timeout, expected=session.expected)
        result = run_capped(
            session.command, tracker, spawn=spawn or _spawn, stop=stop,
            kill_container=lambda: kill_container(spec.docker, session.name),
            remove_container=lambda: remove_container(spec.docker, session.name),
            stderr_path=session.scratch / "pi-stderr.log", on_event=lambda event: on_event(event, tracker))
        result.change, result.review_type, result.review_id = spec.change, spec.review_type, spec.review_id
        # Read before the scratch folder goes.
        result.results, unreadable = read_results(session.mounts.results)
        result.failures += unreadable
        if result.stopped_by == "completed":
            result.failures += session_failures(tracker, result.results, spec.review_type)
        return result


def model_known(base_url: str, model: str) -> Optional[bool]:
    """True/False when LM Studio lists/does not list `model`; None when it cannot be asked.

    LM Studio's own REST API is served at the server root, not under the OpenAI-compatible `/v1` (`rest_root` drops it)."""
    from lmstudio_model_loader_lib import rest_root  # libs/ is on sys.path since this module was imported
    try:
        with urllib.request.urlopen(f"{rest_root(base_url)}/api/v0/models", timeout=10) as response:
            return model in {m["id"] for m in json.load(response)["data"]}
    except (OSError, ValueError, KeyError):
        return None


def host_model_check(host_url: str, model: str) -> Optional[str]:
    """A problem with `model` on the host's LM Studio, or None. Not being able to ask is not a problem: it only warns."""
    known = model_known(host_url, model)
    if known is False:
        return f"LM Studio at {display_url(host_url)} does not list a model named {model!r} (`lms ls` shows the ids)"
    if known is None:
        print(f"warning: could not ask LM Studio at {display_url(host_url)} whether {model!r} exists; going on", file=sys.stderr)
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Pi in a throwaway container against the codebase-reviewer, with caps.")
    p.add_argument("--model", required=True, help="the model Pi calls the reviewer with, as LM Studio lists it (`lms ls`)")
    p.add_argument("--type", choices=ROLES, default="reasoning", help="the review the session runs (default reasoning). verifier and "
                   "classifier run that one stage on --findings-file, and need a --config with an active reviewer of the type")
    p.add_argument("--findings-file", type=Path, metavar="PATH", help="the findings file a verifier or classifier session starts "
                   "from (libs/reviewer_files_lib.py has the format). It is checked, and the container gets a read-only copy")
    p.add_argument("--review-id", default="", help="an ID stored in the run's record, to group the sessions of one review")
    p.add_argument("--config", type=Path, help="a harness/live-tests/configs/*.json file: its reviewers (and rubric) become the reviewer's "
                   "REVIEWERS_FILE, so models and contexts are read from where the harness records them. Instead of --reviewer-model")
    p.add_argument("--reviewer-model", help="the model behind the one reviewer built from the arguments; default: the same as --model")
    p.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH,
                   help=f"context tokens: Pi's model entry, and the reviewer built from the arguments (default {DEFAULT_CONTEXT_LENGTH}); "
                   "match what LM Studio loaded. A --config carries its own reviewer contexts")
    p.add_argument("--load-on-demand", action="store_true",
                   help="the reviewer built from the arguments may ask LM Studio to load its model (default: it must already be loaded)")
    p.add_argument("--change", default=DEFAULT_CHANGE,
                   help="the change of the harness-live-tests branch the reviewer is asked to review, named as in the answer key "
                   f"(default {DEFAULT_CHANGE}: two lines added, two removed; one bug and one decoy, both judgeable from the diff)")
    p.add_argument("--repo", type=Path, metavar="PATH",
                   help="a checkout that has the harness-live-tests tags (default: this one). Only the base and the change are fetched "
                   "from it into a scratch repository, and that repository is what the container sees, not the checkout. One without the "
                   "tags is refused with the `git fetch` that gets them")
    p.add_argument("--prompt", help="what Pi is asked; {repo} is replaced by the repository's path in the container, "
                   "{base} and {head} by the two tags it holds (the branch's base and the change), {findings} by the findings file's "
                   "path (nothing in a reasoning session). The review call is checked against the session whatever the prompt says")
    p.add_argument("--host-lmstudio-url", help="LM Studio as this machine sees it, for the preflight check "
                   "(default: CLAUDE_RUNWAY_LMSTUDIO_URL, else LM Studio's own default)")
    p.add_argument("--lmstudio-url", help="LM Studio as the container sees it "
                   f"(default: the host URL with localhost swapped for {CONTAINER_HOST_ALIAS}, port and path kept)")
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="wall-clock seconds for the whole run")
    p.add_argument("--mcp-timeout", type=int, help="seconds Pi waits for one request to the reviewer server, so for a whole "
                   "`review` call (default: --timeout, so Pi does not give up and call again while a review is running)")
    p.add_argument("--lmstudio-timeout", type=int, help="seconds the reviewer waits for one completion request to LM Studio "
                   "(default: --timeout)")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--build", action="store_true", help="build the image from harness/pi/Dockerfile first")
    p.add_argument("--docker", default="docker", help="the docker executable")
    p.add_argument("--memory", default="4g")
    p.add_argument("--cpus", default="2")
    p.add_argument("--pids-limit", type=int, default=256)
    p.add_argument("--log", type=Path, help="write every event Pi emitted to this new file (JSON lines); an existing file, "
                   "or one under ~/.claude or ~/.pi, is refused")
    p.add_argument("--dry-run", action="store_true", help="print the docker command and exit; nothing is built or run")
    p.add_argument("--no-preflight", action="store_true", help="skip asking the host's LM Studio whether the model exists")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.max_turns, args.max_tool_calls, args.timeout) < 1:
        print("error: --max-turns, --max-tool-calls and --timeout must each be 1 or more", file=sys.stderr)
        return 2
    if any(value is not None and value < 1 for value in (args.mcp_timeout, args.lmstudio_timeout)):
        print("error: --mcp-timeout and --lmstudio-timeout must each be 1 or more", file=sys.stderr)
        return 2
    if args.log:
        problem = unsafe_log(args.log, Path.home())
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 2
    if args.config and (args.reviewer_model or args.load_on_demand):
        print("error: --config carries its own reviewers; do not combine it with --reviewer-model or --load-on-demand", file=sys.stderr)
        return 2
    if args.type == "reasoning" and args.findings_file:
        print("error: --findings-file is for --type verifier or classifier: a reasoning review starts from the diff", file=sys.stderr)
        return 2
    if args.type != "reasoning" and not (args.findings_file and args.config):
        print(f"error: --type {args.type} needs --findings-file (the findings it works on) and --config (its reviewer; the one "
              "built from the arguments is a reasoning reviewer)", file=sys.stderr)
        return 2
    if args.config:
        try:
            reviewers = reviewers_from_config(args.config, args.type)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        reviewers = reviewers_file(args.reviewer_model or args.model, args.context_length, args.load_on_demand)
    from runway_defaults import LMSTUDIO_URL  # libs/ is on sys.path since this module was imported
    host_url = args.host_lmstudio_url or LMSTUDIO_URL
    try:
        lmstudio_in_container = args.lmstudio_url or container_url(host_url)
    except ValueError as exc:  # a port that is not a number
        print(f"error: LM Studio URL: {exc}", file=sys.stderr)
        return 2
    # The change and the checkout that has it are settled before anything slow (the model preflight, `--build`): a clone without the
    # branch's tags is told so, with the `git fetch` that gets them, in the first second and not after an image build.
    try:
        tag = change_tag(args.change)
        checkout = args.repo.expanduser().resolve() if args.repo else REPO_ROOT
        check_checkout(checkout, tag)
    except (ValueError, ChangeRepoError) as exc:  # a change the answer key lacks, an unreadable key (LiveMetadataError), a bad checkout
        print(f"error: {exc}", file=sys.stderr)
        return 2
    spec = SessionSpec(
        model=args.model, reviewers=reviewers, lmstudio_url=lmstudio_in_container, review_type=args.type, change=args.change,
        checkout=checkout, findings_file=args.findings_file.expanduser() if args.findings_file else None, review_id=args.review_id,
        context_length=args.context_length, prompt=args.prompt, max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
        timeout=args.timeout, mcp_timeout=args.mcp_timeout, lmstudio_timeout=args.lmstudio_timeout, image=args.image,
        docker=args.docker, memory=args.memory, cpus=args.cpus, pids_limit=args.pids_limit)
    # The rest of the session's settings and its findings file, also before anything slow.
    try:
        _check_spec(spec)
        if spec.findings_file is not None:
            with tempfile.TemporaryDirectory(prefix="claude-runway-pi-") as tmp:
                copy_findings_file(spec.findings_file, spec.review_type, Path(tmp) / "findings")
    except SessionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.no_preflight and not args.dry_run:
        for model in dict.fromkeys([args.model, *models_in(reviewers, args.type)]):
            problem = host_model_check(host_url, model)
            if problem:
                print(f"error: {problem}", file=sys.stderr)
                return 2

    if args.build and not args.dry_run:
        try:
            build = subprocess.run([args.docker, "build", "-t", args.image, str(PI_DIR)])
        except OSError as exc:
            print(f"error: could not start {args.docker!r}: {exc}", file=sys.stderr)
            return 2
        if build.returncode != 0:
            print("error: docker build failed", file=sys.stderr)
            return 2

    if args.dry_run:
        try:
            with prepared_session(spec) as session:
                print(shlex.join(session.command))
        except SessionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"LM Studio as the container sees it (in the agent folder's models.json and mcp.json): "
              f"{display_url(lmstudio_in_container)}", file=sys.stderr)
        return 0

    try:
        # "x", not "w": a mistyped path must not truncate a file that is already there.
        log_file = open(args.log, "x", encoding="utf-8") if args.log else None
    except FileExistsError:
        print(f"error: --log {args.log} already exists; the driver does not overwrite a file. Choose a new name.", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot write --log {args.log}: {exc}", file=sys.stderr)
        return 2

    def on_event(event: dict, tracker: CapTracker) -> None:
        if log_file:
            log_file.write(json.dumps(event) + "\n")
        kind = event.get("type")
        if kind == "turn_start":
            print(f"[{tracker.elapsed():5.1f}s] turn {tracker.turns_started}", file=sys.stderr)
        elif kind == "tool_execution_start":
            print(f"[{tracker.elapsed():5.1f}s]   tool {event.get('toolName')}", file=sys.stderr)

    try:
        result = run_session(spec, on_event=on_event)
    except SessionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if log_file and log_file.tell() == 0:
            # Nothing ran, so the log this run made is empty: left behind, it would get the same command refused next time.
            log_file.close()
            args.log.unlink(missing_ok=True)
        return 2
    finally:
        if log_file:
            log_file.close()
    print(json.dumps(asdict(result), indent=2))
    return result.exit_status()


if __name__ == "__main__":
    sys.exit(main())
