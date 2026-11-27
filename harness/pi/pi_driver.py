#!/usr/bin/env python3
"""Run Pi (the coding agent) in a throwaway Docker container against the codebase-reviewer's MCP server, with caps.

Why this exists: the harness scripts call the reviewer's libraries directly, so they never exercise the path an agent
takes: finding the `review` tool over MCP and calling it. This does, with Pi as the agent, and it keeps the whole run in a
container so the host checkout, `~/.claude` and `~/.pi` are never touched (the container never sees them, and `--log`, the one
file the driver itself writes, is refused under `~/.claude` and `~/.pi` and never overwrites a file). Pi has no turn or tool-call limit of its own,
so the driver enforces them: it reads Pi's JSON event stream and runs `docker kill` when a cap is reached.

What Pi is asked to review is a change from the `harness-live-tests` branch (`--change`, default `change-6`; see harness/README.md,
"A real repository with seeded bugs"): real code with a seeded bug and a decoy, not a made-up case. The container is given a
scratch repository that holds only the branch's base and that change, fetched from a checkout that has the tags (`--repo`,
default this one). The checkout is not mounted (of this one, only the reviewer's own `libs/` and `mcpservers/` are, read-only, as
before), and neither is the answer key, which sits in it under `harness/live-tests/`. The JSON summary names the change, so
`score_run.py` can score it against the key without being told.

A development tool, not product code (CI does not lint or type-check `harness/`; the logic here is covered by
`tests/test_harness_pi_driver.py` with a fake `docker`, no Docker needed). See harness/pi/README.md.

    .venv/bin/python harness/pi/pi_driver.py --build --model <lm-studio-id>

Standard library only. Exit code: 0 = Pi finished and the reviewer returned a result; 1 = it finished without calling the
reviewer, every call to it errored, Pi aborted, or the container failed; 2 = a bad argument, a failed preflight, a checkout
without the branch's tags or docker could not be started (nothing was run); 3 = a cap stopped the run.
"""

import argparse
import json
import os
import queue
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
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
C_STATE = "/state"  # the reviewer's database and logs for this run only
C_VENV_PYTHON = "/opt/venv/bin/python"

PROVIDER = "lmstudio"
MCP_SERVER_NAME = "reviewer"
# Pi names an MCP tool mcp__<server>__<tool>; --tools takes the pattern to expose the server's tools and nothing else MCP.
REVIEW_TOOL = f"mcp__{MCP_SERVER_NAME}__review"
# How the review tool words a refusal: `review()` in mcpservers/codebase_reviewer_mcp_server.py returns "Error: ..." as a normal
# result for an invalid reviewers file, bad parameters or nothing to run (its docstring promises the prefix).
REVIEW_REFUSAL_PREFIX = "Error:"
# Pi's own read-only tools. Never `bash`, `edit` or `write`: the run needs to look, not change.
READ_ONLY_TOOLS = ("read", "grep", "find", "ls")

# {repo} is replaced with the repository's path inside the container, {base} and {head} with the two tags it holds (plain
# replace, so the JSON braces need no escaping).
DEFAULT_PROMPT = (
    "An MCP tool named review is available. Call it once with type \"reasoning\" and params "
    '{"repo_path": "{repo}", "base_ref": "{base}", "head_ref": "{head}"}. '
    "Then list the findings it returns, one per line. Do not edit any file."
)

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

    def __init__(self, max_turns: int, max_tool_calls: int, timeout_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
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
            if self.tool_calls > self.max_tool_calls:
                return "tool_calls"
        elif kind == "tool_execution_end":
            if event.get("toolName") == REVIEW_TOOL:
                # What the reviewer itself returned, kept because the caller's own summary of it can drop most of it.
                text = tool_result_text(event)
                self.review_text = text
                # The review tool reports its own refusals (a bad reviewers file, nothing to run) as ordinary text that
                # starts with "Error:" and an isError of false, so a refused review would otherwise read as a success.
                if event.get("isError") or text.lstrip().startswith(REVIEW_REFUSAL_PREFIX):
                    self.review_errors += 1
        elif kind == "message_end":
            text = message_text(event)
            if text:
                self.final_text = text
        elif kind == "agent_settled":
            self.settled = True
            self.aborted = bool(event.get("aborted"))
        return None


@dataclass
class RunResult:
    change: str = ""  # the branch change that was reviewed (`change-6`): what score_run.py reads the answers for
    stopped_by: str = "completed"  # completed | turns | tool_calls | timeout | aborted | error
    turns: int = 0
    tool_calls: int = 0
    review_calls: int = 0
    review_errors: int = 0
    elapsed_seconds: float = 0.0
    exit_code: Optional[int] = None  # the container client's, when it ended on its own
    final_text: str = ""  # what the caller model said at the end: its summary, which may leave most of the result out
    review_text: str = ""  # what the reviewer's own tool returned: the result to read
    stderr_tail: list = field(default_factory=list)

    def exit_status(self) -> int:
        """The driver's own exit code: see the module docstring."""
        if self.stopped_by in ("turns", "tool_calls", "timeout"):
            return 3
        if self.stopped_by == "completed" and self.review_calls > 0:
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
    state: Path

    def as_list(self) -> list:
        """(host path, container path, writable) for every mount. This list is ALL the container can see of the host."""
        return [
            (self.libs, f"{C_REVIEWER}/libs", False),
            (self.mcpservers, f"{C_REVIEWER}/mcpservers", False),
            (self.repo, C_REPO, False),
            (self.config, C_CONFIG, False),
            (self.agent, C_AGENT, True),
            (self.state, C_STATE, True),
        ]


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


def reviewers_from_config(path: Path) -> dict:
    """The reviewers file out of a `harness/live-tests/configs/*.json` config, so the models and contexts the harness
    already records are not copied into a second place.

    A live-test config is a reviewers file plus a `stages` key the server would reject, so `stages` is dropped and the rest
    is checked by the server's own validator. A path inside the config (`promptFile`, `translator`, `definitionFile`) would
    point at a file that is not in the container, so a config that uses one is refused rather than run half-configured.
    Raises ValueError with the reason."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("reviewers"), list) or not data["reviewers"]:
        raise ValueError(f"{path} has no 'reviewers' list")
    for reviewer in data["reviewers"]:
        used = [key for key in ("promptFile", "translator") if isinstance(reviewer, dict) and key in reviewer]
        if used:
            raise ValueError(f"{path}: reviewer {reviewer.get('name')!r} uses {', '.join(used)}, a file the container does not have")
    # The harness fills in its own rubric for a config that has none; the server does not, and refuses a file without the key.
    # An empty one is valid until a classifier reviewer is active (the validator then says so).
    rubric = data.get("rubric", [])
    if isinstance(rubric, list):  # anything else is left for the validator below to name, not iterated here
        for entry in rubric:
            if isinstance(entry, dict) and "definitionFile" in entry:
                raise ValueError(f"{path}: a rubric entry uses definitionFile, a file the container does not have")
    reviewers = {"rubric": rubric, "reviewers": data["reviewers"]}
    from codebase_reviewer_config_lib import ConfigError, validate_reviewers  # libs/ is on sys.path since this module was imported
    try:
        validate_reviewers(reviewers, path=str(path))
    except ConfigError as exc:
        raise ValueError(str(exc)) from exc
    # The driver always asks for review type "reasoning" (it passes no findings file, the only way a verifier or classifier review
    # starts without the reasoning stage), so a config with no active reasoning reviewer (a verifier-only or classifier-only one)
    # would be refused by the server.
    if not any(r.get("active", True) and "reasoning" in r["type"] for r in reviewers["reviewers"]):
        raise ValueError(f"{path} has no active reasoning reviewer: the review tool would refuse it, because the reasoning "
                         "review this driver asks for needs one")
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


def models_in(reviewers: dict) -> list:
    """The distinct models of the active reviewers, in file order."""
    return list(dict.fromkeys(r["model"] for r in reviewers["reviewers"] if r.get("active", True)))


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


def mcp_file(lmstudio_url: str, mcp_timeout: Optional[int] = None, lmstudio_timeout: Optional[int] = None) -> dict:
    """Pi's mcp.json, registering the reviewer MCP server: the file `pi mcp add` would write, in the same shape.

    The driver writes it into the agent scratch folder instead of running `pi mcp add` in the container, so the LM Studio
    URL, which can carry a user name and password, is never on a command line (host `ps`, `docker inspect`, --dry-run).

    Two different limits, each left out when None so its own default holds (60 seconds for both): `mcp_timeout` is Pi's
    `timeout` for one request to this server, which a whole `review` call must fit in while the server sends no progress;
    `lmstudio_timeout` is the reviewer's limit for one completion request to LM Studio."""
    env = {
        "REVIEWERS_FILE": f"{C_CONFIG}/reviewers.json",
        "CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB": f"{C_STATE}/codebase-reviewer.db",
        "CLAUDE_RUNWAY_LMSTUDIO_URL": lmstudio_url,
    }
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


def docker_run_command(
    *, docker: str, image: str, name: str, mounts: Mounts, model: str, prompt: str,
    memory: str, cpus: str, pids_limit: int,
) -> list:
    """The whole `docker run` command line, as a list (never a shell string, so no quoting problem and no injection).

    It carries no LM Studio URL: that is in the models.json and mcp.json the driver wrote into the agent scratch folder."""
    command = [
        # --init: Pi runs as PID 1, which the kernel gives no default signal handling, so a stop signal would never reach it;
        # the small init process forwards signals and reaps children.
        docker, "run", "--rm", "--init", "--name", name,
        "--memory", memory, "--cpus", cpus, "--pids-limit", str(pids_limit),
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        # Docker Desktop already knows this name; plain Linux Docker needs to be told.
        "--add-host", "host.docker.internal:host-gateway",
        "--workdir", "/work",
    ]
    # On Linux the files the container writes into the scratch mounts would be root's and the driver could not remove them.
    getuid = getattr(os, "getuid", None)
    if getuid is not None:
        command += ["--user", f"{getuid()}:{os.getgid()}"]
    command += [
        "--env", f"PI_CODING_AGENT_DIR={C_AGENT}",
        "--env", "PI_TELEMETRY=0", "--env", "PI_OFFLINE=1", "--env", "PI_SKIP_VERSION_CHECK=1",
        "--env", "HOME=/tmp",
    ]
    # --mount rather than -v: a Windows path (C:\...) contains the colon that -v uses to split its fields.
    for host, inside, writable in mounts.as_list():
        spec = f"type=bind,source={host},target={inside}"
        command += ["--mount", spec if writable else spec + ",readonly"]
    command += ["--entrypoint", "pi", image, *pi_arguments(model, prompt)]
    return command


def unsafe_mounts(mounts: Mounts, home: Path, checkout: Path) -> list:
    """Reasons a mount list must not be used; empty when it is fine.

    The kit promises never to expose the checkout, `~/.claude` or `~/.pi`. A mount overlaps one of them when it is the
    folder, a folder above it (which contains it) or a folder inside it. The one exception is the reviewer's own code:
    `libs/` and `mcpservers/` of the checkout, read-only, because the MCP server cannot run without them. Those two
    must be the checkout's real folders: a symlink in their place could point anywhere outside the three roots, and
    Docker would mount whatever it points to."""
    checkout = checkout.resolve()
    code = {checkout / "libs", checkout / "mcpservers"}
    forbidden = {"the checkout": checkout, "~/.claude": (home / ".claude").resolve(), "~/.pi": (home / ".pi").resolve()}
    problems = []
    for name, host in (("libs", mounts.libs), ("mcpservers", mounts.mcpservers)):
        if host.is_symlink() or host.resolve() != checkout / name:
            problems.append(f"{host} is not the checkout's own {name}/ folder ({checkout / name})")
    for host, _inside, writable in mounts.as_list():
        resolved = host.resolve()
        if resolved in code and not writable:
            continue
        for label, path in forbidden.items():
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
) -> RunResult:
    """Starts `command`, reads its JSON stream and stops it when a cap is reached.

    `spawn(command, stderr_file)` returns an object with `stdout` (a binary stream with `read1`), `wait(timeout)` and `kill()`,
    and `kill_container()` stops the container itself (killing the `docker` client alone would leave it running). Both are
    arguments so a test can fake them."""
    result = RunResult()
    chunks: queue.Queue = queue.Queue()
    with open(stderr_path, "wb") as stderr_file:
        process = spawn(command, stderr_file)
        stopped_by: Optional[str] = None
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
                kill_container()
            try:
                result.exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Killing the `docker` client alone would leave the container running, so stop the container first.
                kill_container()
                process.kill()
                result.exit_code = process.wait(timeout=30)
            if stopped_by is None and (not tracker.settled or result.exit_code != 0):
                # The stream ended without a clean finish. A `docker` client that was killed or crashed ends its output while
                # the container goes on running with no caps, so make sure it is stopped (a no-op if it already exited).
                kill_container()
        except BaseException:
            # An exception (Ctrl+C, a failed log write) while reading or waiting. Stop the client first, so that if the
            # daemon has not created the container yet it never will (`docker kill` fails on a name that does not exist yet),
            # then the container itself, which would otherwise run on uncapped while the caller deletes the folders it has
            # mounted.
            try:
                process.kill()
            except OSError:
                pass
            kill_container()
            raise

    # A settle that says `aborted` is not a finished run, whatever the container's exit code was: it must not read as a pass.
    # Nor is a settle followed by a failing container (Pi settles before it shuts its MCP server down, so an OOM kill or a
    # crash during that shutdown still ends nonzero): only a settle and a clean container exit count as completed. A stream
    # that ends without a settle is an error even when the container exits 0, since Pi never said it had finished.
    result.stopped_by = stopped_by or (
        "aborted" if tracker.aborted else "completed" if tracker.settled and result.exit_code == 0 else "error"
    )
    result.turns, result.tool_calls = tracker.turns, tracker.tool_calls
    result.review_calls, result.review_errors = tracker.review_calls, tracker.review_errors
    result.elapsed_seconds = round(tracker.elapsed(), 1)
    result.final_text, result.review_text = tracker.final_text, tracker.review_text
    try:
        result.stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    except OSError:
        pass
    return result


class LaunchError(Exception):
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


class ChangeRepoError(Exception):
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


def prepare_scratch(scratch: Path, checkout: Path, tag, model: str, reviewers: dict, context_length: int, lmstudio_url: str,
                    mcp_timeout: Optional[int] = None, lmstudio_timeout: Optional[int] = None) -> Mounts:
    """Fills `scratch` with everything the container mounts and returns the Mounts. Raises ChangeRepoError."""
    repo = build_change_repo(scratch, checkout, tag)
    config, agent, state = scratch / "config", scratch / "agent", scratch / "state"
    for directory in (config, agent, state):
        directory.mkdir()
    (config / "reviewers.json").write_text(json.dumps(reviewers, indent=2), encoding="utf-8")
    (agent / "models.json").write_text(json.dumps(models_file(model, context_length, lmstudio_url), indent=2), encoding="utf-8")
    (agent / "mcp.json").write_text(json.dumps(mcp_file(lmstudio_url, mcp_timeout, lmstudio_timeout), indent=2), encoding="utf-8")
    return Mounts(REPO_ROOT / "libs", REPO_ROOT / "mcpservers", repo, config, agent, state)


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
                   "{base} and {head} by the two tags it holds (the branch's base and the change)")
    p.add_argument("--host-lmstudio-url", help="LM Studio as this machine sees it, for the preflight check "
                   "(default: CLAUDE_RUNWAY_LMSTUDIO_URL, else LM Studio's own default)")
    p.add_argument("--lmstudio-url", help="LM Studio as the container sees it "
                   f"(default: the host URL with localhost swapped for {CONTAINER_HOST_ALIAS}, port and path kept)")
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="wall-clock seconds for the whole run")
    p.add_argument("--mcp-timeout", type=int, help="seconds Pi waits for one request to the reviewer server, so for a whole "
                   "`review` call (default: not set, which is Pi's 60). --timeout still bounds the run")
    p.add_argument("--lmstudio-timeout", type=int, help="seconds the reviewer waits for one completion request to LM Studio "
                   "(default: not set, which is the reviewer's 60)")
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
    if args.config:
        try:
            reviewers = reviewers_from_config(args.config)
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
    if not args.no_preflight and not args.dry_run:
        for model in dict.fromkeys([args.model, *models_in(reviewers)]):
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

    scratch = Path(tempfile.mkdtemp(prefix="claude-runway-pi-"))
    try:
        try:
            mounts = prepare_scratch(scratch, checkout, tag, args.model, reviewers, args.context_length, lmstudio_in_container,
                                     args.mcp_timeout, args.lmstudio_timeout)
        except ChangeRepoError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        problems = unsafe_mounts(mounts, Path.home(), REPO_ROOT)
        if problems:
            print("error: refusing to start, a mount would expose the host:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
        prompt = (args.prompt or DEFAULT_PROMPT).replace("{repo}", C_REPO).replace("{base}", scoring.BASE_REF).replace("{head}", tag.head)
        name = f"claude-runway-pi-{os.getpid()}-{int(time.time())}"
        command = docker_run_command(
            docker=args.docker, image=args.image, name=name, mounts=mounts, model=args.model, prompt=prompt,
            memory=args.memory, cpus=args.cpus, pids_limit=args.pids_limit,
        )
        if args.dry_run:
            print(shlex.join(command))
            print(f"LM Studio as the container sees it (in the agent folder's models.json and mcp.json): "
                  f"{display_url(lmstudio_in_container)}", file=sys.stderr)
            return 0

        tracker = CapTracker(args.max_turns, args.max_tool_calls, args.timeout)
        try:
            # "x", not "w": a mistyped path must not truncate a file that is already there.
            log_file = open(args.log, "x", encoding="utf-8") if args.log else None
        except FileExistsError:
            print(f"error: --log {args.log} already exists; the driver does not overwrite a file. Choose a new name.", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"error: cannot write --log {args.log}: {exc}", file=sys.stderr)
            return 2

        def on_event(event: dict) -> None:
            if log_file:
                log_file.write(json.dumps(event) + "\n")
            kind = event.get("type")
            if kind == "turn_start":
                print(f"[{tracker.elapsed():5.1f}s] turn {tracker.turns_started}", file=sys.stderr)
            elif kind == "tool_execution_start":
                print(f"[{tracker.elapsed():5.1f}s]   tool {event.get('toolName')}", file=sys.stderr)

        def kill_container() -> None:
            try:
                subprocess.run([args.docker, "kill", name], capture_output=True, check=False, timeout=30)
            except subprocess.TimeoutExpired:
                # A docker daemon that does not answer must not hang the driver on its way out; say so, since the container
                # may still be running.
                print(f"warning: docker kill {name} did not answer in 30 s; check `docker ps`", file=sys.stderr)

        try:
            result = run_capped(command, tracker, spawn=_spawn, kill_container=kill_container,
                                stderr_path=scratch / "pi-stderr.log", on_event=on_event)
        except LaunchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            if log_file:
                log_file.close()
        result.change = tag.tag_id  # names what was reviewed, so score_run.py needs no flag to find its answers
        print(json.dumps(asdict(result), indent=2))
        return result.exit_status()
    finally:
        _remove_tree(scratch)


if __name__ == "__main__":
    sys.exit(main())
