"""
The tool executor of the `codebase-reviewer` server, and the in-process repo tools it runs.

Why it exists. The local models that review a change see only the diff, so when they need more of the repository they have
to guess. This module is the one place that reads it for them. Everything that later needs a lookup goes through the same
executor: the feedback loop where a reviewer asks for more context, the MCP client and native tool calls.

**Programming-language agnostic, by rule.** The tool facilitates a review that the MODELS do, so nothing here interprets
code as any particular language: no keyword or builtin lists, no notion of what a definition, a call or a function looks
like, no indentation rules, no test-file naming conventions. It finds text and returns it; what the text means is for the
model to work out. (An earlier version of this issue's work recognised definitions by `def`/`class`/`function`/`fn`/`func`
and silently found nothing in C, C++, Java or C#; that was removed. See the branch `archive/596-context-gatherer`.)

The contract, which every caller relies on:

  * `ToolExecutor(repo_root, head).run(name, args) -> ToolResult`: a tool name and its arguments in; result text out, with
    whether it was cut (`truncated`) and whether it failed (`error`). **It never raises.** An unknown tool, bad arguments,
    a path outside the repo, a path git does not track, a file that is not at the head, a bad regex, a git timeout: each
    comes back as `ToolResult(error=True)` text a model or a log can read.
  * **Read-only, always.** Only `git show`, `git grep`, `git ls-tree` and `git ls-files` run. Nothing is written.
  * **One pinned commit.** Every tool reads the files as they are at `head`, never the working tree, so a run that takes
    minutes sees the same code even when a branch moves or a file is edited, and an untracked file (a `.env`, a scratch
    file) cannot be reached at all.
  * **Tracked files only, anywhere in the repo.** This is a different reader from `verifier_reviewer_lib.git_file_reader`,
    which reads only the change's own files and stays that way: grounding must refuse a finding whose quote is in a file the
    change did not touch.
  * **A cut or a failure is never silent.** A result that was shortened says where, in its text, and sets `truncated`; a
    line that was clipped says how much; a failure is an error result, never an empty answer.

The tools: `read_file(path, start_line, end_line)` and `grep(pattern, path_glob)` (an extended regular expression, matched by
`git grep`, so the same for every language).

Standard library only, apart from `ReviewError` from the reasoning stage.
"""

import posixpath
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from reasoning_reviewer_lib import ReviewError

GIT_TIMEOUT_SECONDS = 10
# A tool's result text is cut to this many characters, with `truncated` set.
MAX_TOOL_RESULT_CHARS = 8000
# `read_file` returns at most this many lines per call; ask again with a later `start_line` for the rest.
MAX_READ_LINES = 400
# A file bigger than this is not read (a generated bundle or a dataset is no help to a reviewer).
MAX_FILE_BYTES = 2_000_000
# `grep` returns at most this many matching lines, each cut to MAX_LINE_CHARS.
GREP_MAX_LINES = 50
MAX_LINE_CHARS = 300
MAX_GREP_PATTERN_CHARS = 500
MAX_GLOB_CHARS = 200
# `grep` reads at most this much git output; it shows far less (GREP_MAX_LINES lines).
MAX_GREP_OUTPUT_BYTES = 400_000

TOOL_NAMES = ("read_file", "grep")

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class _Limited:
    """A git command's output read up to a limit: the exit code, what was read, its first stderr bytes, and whether the
    command was stopped because it had more."""

    returncode: int
    stdout: bytes
    stderr: bytes
    over: bool


@dataclass(frozen=True)
class ToolResult:
    """What a tool call returned: the text, whether it was cut to fit, and whether it is an error message."""

    text: str
    truncated: bool = False
    error: bool = False


class LookupFailed(Exception):
    """A lookup at the head could not give an answer: git is missing, it ran out of time, or it returned an error. Not the
    same as 'no match' (which is an answer). `ToolExecutor.run` turns it into an error result."""


_LINE_CUT = re.compile(r" \.\.\.\[line cut: \d+ more characters\]")


def clip_line(line: str) -> str:
    """`line` as shown to a model: at most MAX_LINE_CHARS characters, and when it was longer the cut is SAID, in the line
    itself, with how much was left out. A line clipped without a word would show a model shortened code (a condition, an
    argument list) as if it were whole."""
    if len(line) <= MAX_LINE_CHARS:
        return line
    return f"{line[:MAX_LINE_CHARS]} ...[line cut: {len(line) - MAX_LINE_CHARS} more characters]"


def has_line_cut(text: str) -> bool:
    """Whether `text` holds a line `clip_line` shortened. A tool result with one is `truncated`."""
    return bool(_LINE_CUT.search(text))


def _numbered(lines: list, first: int, marked: Optional[range] = None) -> str:
    """`lines` as `<number> <mark> <code>`, numbered from `first`: the format a reviewer already reads in a diff chunk and
    a verifier in its code window. `>` marks the lines in `marked`. A line longer than MAX_LINE_CHARS is clipped
    visibly (`clip_line`)."""
    out = []
    for offset, line in enumerate(lines):
        number = first + offset
        mark = ">" if marked is not None and number in marked else " "
        out.append(f"{number:>5} {mark} {clip_line(line)}")
    return "\n".join(out)


def _cut(text: str, limit: int) -> "tuple[str, bool]":
    """`text` cut to at most `limit` characters at a line boundary; `(text, whether it was cut)`."""
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    newline = cut.rfind("\n")
    return (cut[:newline] if newline > 0 else cut), True


class ToolExecutor:
    """Runs the repo tools at one pinned commit. Build one per review run; it keeps the file list and the files it has read,
    so a file asked for several times costs one `git show`."""

    def __init__(self, repo_root: "str | Path", head: str):
        # A head that starts with `-` would be read by git as an option, and whitespace or NUL is never in a commit id.
        if not head or head.startswith("-") or any(c.isspace() or c == "\0" for c in head):
            raise ReviewError(f"head {head!r} is not a usable git ref")
        self.root = Path(repo_root)
        self.head = head
        self._tracked: Optional[frozenset] = None
        self._files: dict = {}
        self._tools: dict = {
            "read_file": self._tool_read_file,
            "grep": self._tool_grep,
        }

    # -- the one entry point --------------------------------------------------------------------------------------

    def run(self, name: object, args: object = None) -> ToolResult:
        """Runs tool `name` with `args` (a dict of its arguments). Never raises."""
        try:
            tool: Optional[Callable[..., ToolResult]] = self._tools.get(name) if isinstance(name, str) else None
            if tool is None:
                return _error(f"unknown tool {name!r}; the tools are: {', '.join(TOOL_NAMES)}")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                return _error(f"the arguments of {name} must be an object of named values (got {type(args).__name__})")
            return tool(args)
        except Exception as exc:  # the contract: a model or a loop gets text, never a traceback
            return _error(f"{name if isinstance(name, str) else 'the tool'} failed: {type(exc).__name__}: {exc}")

    # -- git, read-only ---------------------------------------------------------------------------------------------

    def _git(self, args: list) -> Optional["subprocess.CompletedProcess[bytes]"]:
        """The result of a git command in the repo, or None when git is missing or too slow."""
        try:
            return subprocess.run(
                # quotePath off: otherwise git prints a non-ASCII path as `"\303\274.py"` and a lookup would miss the file.
                ["git", "-c", "core.quotePath=false", "-C", str(self.root), *args],
                capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    def _git_limited(self, args: list, max_bytes: int) -> Optional["_Limited"]:
        """Like `_git`, but never holds more than `max_bytes` of the command's output: it reads that much, then stops the
        command (`over` says it did). For `git grep` of a pattern that matches everywhere, where the whole output of a big
        repo would otherwise be read into memory before any cap applied. None when git is missing or ran out of time."""
        try:
            proc = subprocess.Popen(
                ["git", "-c", "core.quotePath=false", "-C", str(self.root), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except OSError:
            return None
        timed_out: list = []

        def stop() -> None:
            timed_out.append(True)
            proc.kill()

        timer = threading.Timer(GIT_TIMEOUT_SECONDS, stop)
        timer.start()
        try:
            assert proc.stdout is not None and proc.stderr is not None
            out = proc.stdout.read(max_bytes + 1)
            over = len(out) > max_bytes
            if over:
                proc.kill()
            err = proc.stderr.read(4000)
            proc.wait()
        except (OSError, ValueError):
            proc.kill()
            proc.wait()
            return None
        finally:
            timer.cancel()
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        if timed_out:
            return None
        if over:
            # Drop the line the limit cut in half, and report success: the output read so far is valid.
            return _Limited(0, out[:max_bytes].rsplit(b"\n", 1)[0], err, True)
        return _Limited(proc.returncode, out, err, False)

    def tracked_paths(self) -> frozenset:
        """Every file in the head commit's tree. Only an ANSWER is cached: git that is missing, too slow or failing raises
        LookupFailed and the next call asks again, instead of an empty list making every file 'not tracked' for the run."""
        if self._tracked is None:
            done = self._git(["ls-tree", "-r", "-z", "--name-only", self.head])
            if done is None:
                raise LookupFailed("the list of files could not be read: git timed out or is not available")
            if done.returncode != 0:
                reason = done.stderr.decode("utf-8", errors="replace").strip().splitlines()
                raise LookupFailed(f"the list of files could not be read: {reason[0][:200] if reason else 'git ls-tree returned an error'}")
            self._tracked = frozenset(p for p in done.stdout.decode("utf-8", errors="replace").split("\0") if p)
        return self._tracked

    def _check_path(self, path: object) -> "tuple[str, Optional[str]]":
        """`(clean path, None)` for a file in the head's tree, else `("", why not)`. The reasons are separate so a loop (or a
        person reading the log) can tell a typo from a path that is not allowed."""
        if not isinstance(path, str) or not path.strip():
            return "", "path must be a non-empty string"
        if "\0" in path:
            return "", "path contains a NUL character"
        if path.startswith("/") or _WINDOWS_DRIVE.match(path):
            return "", f"{path!r} is outside the repository: give a path relative to the repository root"
        clean = posixpath.normpath(path[2:] if path.startswith("./") else path)
        if clean == ".." or clean.startswith("../"):
            return "", f"{path!r} is outside the repository"
        if clean == ".":
            return "", f"{path!r} is the repository root, a directory, not a file"
        try:
            tracked = self.tracked_paths()
        except LookupFailed as exc:
            return "", f"{clean!r} could not be checked: {exc}"  # not 'not a tracked file': nothing is known about it
        if clean in tracked:
            return clean, None
        if any(p.startswith(clean + "/") for p in tracked):
            return "", f"{clean!r} is a directory, not a file"
        listed = self._git(["--literal-pathspecs", "ls-files", "--error-unmatch", "--", clean])
        if listed is not None and listed.returncode == 0:
            return "", f"{clean!r} is tracked but is not in the commit under review ({self.head[:12]}): it was added or removed since"
        return "", f"{clean!r} is not a tracked file of this repository"

    def _read(self, path: object) -> "tuple[Optional[str], Optional[str]]":
        """`(text, None)` or `(None, why not)`. A file that cannot be read is not retried within a run."""
        clean, problem = self._check_path(path)
        if problem is not None:
            return None, problem
        if clean not in self._files:
            # Bounded: git is stopped once it has handed back MAX_FILE_BYTES, so a large tracked dataset or generated
            # artifact is never buffered whole before the size check (`over`) refuses it.
            done = self._git_limited(["show", f"{self.head}:{clean}"], MAX_FILE_BYTES)
            if done is None:
                self._files[clean] = (None, f"git could not read {clean!r} (git is missing or timed out)")
            elif done.over:
                self._files[clean] = (None, f"{clean!r} is larger than {MAX_FILE_BYTES} bytes and is not read")
            elif done.returncode != 0:
                self._files[clean] = (None, f"{clean!r} is not readable at the head commit")
            elif b"\0" in done.stdout[:8000]:
                self._files[clean] = (None, f"{clean!r} is a binary file")
            else:
                self._files[clean] = (done.stdout.decode("utf-8", errors="replace"), None)
        return self._files[clean]

    # -- the tools ------------------------------------------------------------------------------------------------------

    def _tool_read_file(self, args: dict) -> ToolResult:
        problem = _unexpected(args, "read_file", {"path", "start_line", "end_line"})
        if problem:
            return _error(problem)
        if "path" not in args:
            return _error("read_file needs a 'path'")
        start = args.get("start_line", 1)
        end = args.get("end_line")
        for label, value in (("start_line", start), ("end_line", end)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                return _error(f"{label} must be a whole number of 1 or more (got {value!r})")
        text, problem = self._read(args["path"])
        if text is None:
            return _error(problem or "the file cannot be read")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if not lines:
            return ToolResult("(the file is empty)")
        if start > len(lines):
            return _error(f"start_line {start} is past the end of the file, which has {len(lines)} line(s)")
        last = min(end if end is not None else len(lines), len(lines))
        if last < start:
            return _error(f"end_line {end} is before start_line {start}")
        capped = min(last, start + MAX_READ_LINES - 1)
        shown, cut = _cut(_numbered(lines[start - 1 : capped], start), MAX_TOOL_RESULT_CHARS)
        truncated = cut or capped < last or has_line_cut(shown)
        if truncated:
            shown_last = start + shown.count("\n")
            shown += f"\n[cut: showing lines {start}-{shown_last} of {len(lines)}; ask for later lines with start_line]"
        return ToolResult(shown, truncated=truncated)

    def _tool_grep(self, args: dict) -> ToolResult:
        problem = _unexpected(args, "grep", {"pattern", "path_glob"})
        if problem:
            return _error(problem)
        pattern, glob = args.get("pattern"), args.get("path_glob") or ""
        if not isinstance(pattern, str) or not pattern:
            return _error("grep needs a non-empty 'pattern'")
        if len(pattern) > MAX_GREP_PATTERN_CHARS or "\0" in pattern or "\n" in pattern:
            return _error(f"the pattern must be one line of at most {MAX_GREP_PATTERN_CHARS} characters")
        if not isinstance(glob, str) or len(glob) > MAX_GLOB_CHARS or "\0" in glob:
            return _error("path_glob must be a short string")
        # A glob is a path pattern, nothing else: `:(...)` is git's own pathspec syntax and `..` leaves the repo.
        if glob.startswith((":", "/")) or ".." in glob.split("/") or _WINDOWS_DRIVE.match(glob):
            return _error(f"path_glob {glob!r} must be a plain pattern relative to the repository root, for example 'libs/*.py'")
        command = ["grep", "-n", "-I", "-E", "--no-color", "-e", pattern, self.head, "--"]
        if glob:
            command.append(f":(glob){glob}")
        done = self._git_limited(command, MAX_GREP_OUTPUT_BYTES)
        if done is None:
            return _error("grep timed out or git is not available")
        if done.returncode == 1:
            return ToolResult("no matches")
        if done.returncode != 0:
            reason = done.stderr.decode("utf-8", errors="replace").strip().splitlines()
            return _error(f"grep failed: {reason[0][:200] if reason else 'git grep returned an error'}")
        prefix = f"{self.head}:"
        lines = [clip_line(raw[len(prefix):] if raw.startswith(prefix) else raw) for raw in done.stdout.decode("utf-8", errors="replace").split("\n") if raw]
        shown = lines[:GREP_MAX_LINES]
        text, cut = _cut("\n".join(shown), MAX_TOOL_RESULT_CHARS)
        # A clipped line counts: the matched text may be in the part that was left out.
        truncated = cut or done.over or len(lines) > len(shown) or has_line_cut(text)
        if truncated:
            counted = f"at least {len(lines)}" if done.over else str(len(lines))
            text += f"\n[cut: {counted} matching line(s), showing the first {text.count(chr(10)) + 1}]"
        return ToolResult(text, truncated=truncated)


def _error(text: str) -> ToolResult:
    return ToolResult(text, error=True)


def _unexpected(args: dict, tool: str, allowed: set) -> Optional[str]:
    extra = sorted(str(k) for k in args if k not in allowed)
    return f"{tool} does not take {', '.join(repr(k) for k in extra)} (it takes: {', '.join(sorted(allowed))})" if extra else None


__all__ = [
    "TOOL_NAMES",
    "MAX_TOOL_RESULT_CHARS",
    "ToolResult",
    "LookupFailed",
    "ToolExecutor",
    "clip_line",
    "has_line_cut",
]
