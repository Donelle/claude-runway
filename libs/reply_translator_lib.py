"""Runs a reviewer's `translator` script on one model reply.

The reply readers in `reasoning_reviewer_lib` are strict on purpose: they read `lines` or `json` and
do not try to guess every format a model might invent, which would never end. A model whose natural
format is neither is still usable when its reviewer names a `translator`: a script that is given the
model's reply and prints the same findings in the format the reviewer's `replyFormat` selects.

The script gets the reply on stdin (UTF-8) and its stdout (UTF-8) is what the reader sees. It runs
once per reply, with no shell, in its own directory, with a small environment, and under a time and
an output limit, because it is a program the server starts on its own:

- a `.py` file runs under the server's own Python, anything else is run directly (it has to be
  executable, which the config validator checks);
- the environment is an allowlist, so the server's own settings (`QDRANT_API_KEY`, and the like) are
  not handed to it;
- it is stopped after TRANSLATOR_TIMEOUT_SECONDS, or as soon as it writes more than
  MAX_TRANSLATED_BYTES, together with anything it started (its own process group on POSIX, `taskkill /T` on
  Windows). The deadline covers a child that keeps the output pipe open after the translator itself has
  exited: reading ends only when every holder of the pipe has closed it, so the deadline is enforced on the
  group and not on the translator's own exit;
- when it has finished, anything it left running is stopped too (POSIX), so a child that redirected its output
  and carried on cannot outlive the reply;
- on POSIX the deadline does not depend on stopping anything at all: the three pipes are served by one loop
  that waits with `select` and gives up at the deadline, so even a helper that left the group (its own session,
  out of reach of the group signal) and still holds a pipe cannot keep the review waiting. The pipes are then
  closed, which such a helper sees as a broken pipe. Windows cannot `select` on a pipe, so there the pipes are
  served by threads and the deadline relies on `taskkill /T` reaching whatever holds them.

It runs with the user's privileges. It is named only in the reviewers file, which the user owns, and
is never taken from the repository under review: point it only at a script you trust.

A translator that fails is a `TranslatorError` carrying a plain explanation. There is no fallback to
the raw reply: the caller records the problem and the chunk yields no findings.

Stdlib only.
"""

import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

TRANSLATOR_TIMEOUT_SECONDS = 30
MAX_TRANSLATED_BYTES = 1_000_000
# How much of a failing script's stderr goes into the explanation.
STDERR_SHOWN_CHARS = 200
_STDERR_KEPT_BYTES = 8192
_READ_CHUNK = 65536
# How often the select loop checks whether the translator has exited once its output has closed.
_EXIT_POLL_SECONDS = 0.02

# The variables a translator can see. `PYTHONUTF8` is set for it, so a Python script reads and writes UTF-8 on
# every platform; the rest are passed through only when the server has them.
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR")


class TranslatorError(Exception):
    """A translator did not produce usable output; `str()` says why, in one line. `brief` is the same reason
    without anything the translator printed (its stderr can echo the model's reply), for a durable record, and
    `printed` is what it printed as it appears in the message ("" when nothing), so a record can withhold it."""

    def __init__(self, message: str, *, brief: Optional[str] = None, printed: str = ""):
        super().__init__(message)
        self.brief = brief or message
        self.printed = printed


def _environment() -> dict:
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env["PYTHONUTF8"] = "1"
    return env


def _command(script: Path) -> list:
    if script.suffix.lower() == ".py":
        return [sys.executable, "-X", "utf8", str(script)]
    return [str(script)]


def _stop(process: "subprocess.Popen") -> None:
    """Kills the translator and everything it started, whether or not the translator itself is still running.

    A child can keep the output pipe open after the translator exits, so the group has to be stopped even then.
    The pid is only safe to signal while this process has not reaped it: until `wait()` or `poll()` reaps the
    translator, its pid (and so its process group id) cannot be reused, so callers stop the group before
    reaping, and a translator already reaped is left alone."""
    if process.returncode is not None:
        return
    try:
        if os.name == "nt":
            # `kill()` ends only the translator; /T takes its children with it.
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _stop_leftovers(process: "subprocess.Popen") -> None:
    """Stops whatever the translator started that is still running, once it has been reaped.

    A child that redirected its own output and carried on closes nothing the reader is waiting for, so it would
    outlive the reply, and it can hold the translator's input and error pipes open, leaving the threads that feed
    and drain them blocked. The translator is reaped by now, so its pid is no longer reserved by it; the process
    group, though, keeps the id from being reused for as long as any member is alive, and when none is left the
    signal finds no group. (The one gap: another process taking the id in the instant between the reaping and
    this call.) Windows has no group to signal here, and `taskkill /T` needs the parent, so a child started by a
    translator there is only stopped while the translator is running."""
    if os.name == "nt":
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass


def _feed(process: "subprocess.Popen", data: bytes) -> None:
    stdin = process.stdin
    assert stdin is not None
    try:
        stdin.write(data)
    except OSError:
        # The script closed its input or exited without reading it; its exit status says what happened.
        pass
    finally:
        try:
            stdin.close()
        except OSError:
            pass


def _drain(process: "subprocess.Popen", kept: bytearray) -> None:
    """Reads stderr to its end, keeping only the first bytes: a chatty script must not fill a pipe and hang."""
    stream = process.stderr
    assert stream is not None
    while True:
        try:
            chunk = os.read(stream.fileno(), _READ_CHUNK)
        except OSError:
            return
        if not chunk:
            return
        if len(kept) < _STDERR_KEPT_BYTES:
            kept += chunk[: _STDERR_KEPT_BYTES - len(kept)]


def _pump_with_select(process: "subprocess.Popen", data: bytes, deadline: float, limit_bytes: int,
                      output: bytearray, stderr_kept: bytearray) -> str:
    """Feeds stdin and reads stdout and stderr until stdout has closed and the translator has exited, without ever
    blocking past `deadline`.

    Returns "done", "timeout" (the deadline passed first) or "too_big" (stdout passed `limit_bytes`). Nothing here
    blocks on a pipe: a write waits until the pipe can take some, a read until there is something to read, and
    every wait is bounded by the time left. The reply is complete once stdout is closed and the translator has
    exited; stdin and stderr are not waited on beyond that, since something the translator left behind may hold
    them. POSIX only. "done" leaves the translator reaped."""
    stdin, stdout, stderr = process.stdin, process.stdout, process.stderr
    assert stdin is not None and stdout is not None and stderr is not None
    selector = selectors.DefaultSelector()
    view = memoryview(data)
    sent = 0
    if view:
        os.set_blocking(stdin.fileno(), False)
        selector.register(stdin, selectors.EVENT_WRITE)
    else:
        stdin.close()
    selector.register(stdout, selectors.EVENT_READ)
    selector.register(stderr, selectors.EVENT_READ)
    stdout_open = True
    try:
        while True:
            if not stdout_open and process.poll() is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            # Once stdout has closed only the translator's exit is awaited, and that has no descriptor to wait on:
            # check it again every few milliseconds.
            wait = remaining if stdout_open else min(remaining, _EXIT_POLL_SECONDS)
            if not selector.get_map():
                time.sleep(wait)
                continue
            for key, _events in selector.select(wait):
                if key.fileobj is stdin:
                    try:
                        sent += os.write(stdin.fileno(), view[sent : sent + _READ_CHUNK])
                    except BlockingIOError:
                        continue
                    except OSError:
                        # The script closed its input or exited without reading it; its exit status says what happened.
                        sent = len(view)
                    if sent >= len(view):
                        selector.unregister(stdin)
                        stdin.close()
                    continue
                chunk = os.read(key.fd, _READ_CHUNK)
                if not chunk:
                    selector.unregister(key.fileobj)
                    if key.fileobj is stdout:
                        stdout_open = False
                elif key.fileobj is stdout:
                    output += chunk
                    if len(output) > limit_bytes:
                        return "too_big"
                elif len(stderr_kept) < _STDERR_KEPT_BYTES:
                    stderr_kept += chunk[: _STDERR_KEPT_BYTES - len(stderr_kept)]
        # Whatever the translator wrote to stderr before it exited is already in the pipe: take it without waiting.
        if any(key.fileobj is stderr for key in selector.get_map().values()):
            os.set_blocking(stderr.fileno(), False)
            while len(stderr_kept) < _STDERR_KEPT_BYTES:
                try:
                    chunk = os.read(stderr.fileno(), _READ_CHUNK)
                except OSError:
                    break
                if not chunk:
                    break
                stderr_kept += chunk[: _STDERR_KEPT_BYTES - len(stderr_kept)]
        return "done"
    finally:
        selector.close()


def _pump_with_threads(process: "subprocess.Popen", data: bytes, limit_seconds: float, limit_bytes: int,
                       output: bytearray, stderr_kept: bytearray) -> str:
    """The same as `_pump_with_select`, for Windows, where `select` does not take a pipe: stdin and stderr are
    served by threads, and a timer stops the translator's process tree at the deadline, which closes the pipe
    the loop below is reading."""
    timed_out = threading.Event()
    guard = threading.Lock()
    reading = [True]

    def on_timeout() -> None:
        with guard:
            if reading[0]:
                timed_out.set()
                _stop(process)

    timer = threading.Timer(limit_seconds, on_timeout)
    feeder = threading.Thread(target=_feed, args=(process, data), daemon=True)
    drainer = threading.Thread(target=_drain, args=(process, stderr_kept), daemon=True)
    timer.start()
    feeder.start()
    drainer.start()
    outcome = "done"
    try:
        stdout = process.stdout
        assert stdout is not None
        while True:
            try:
                chunk = os.read(stdout.fileno(), _READ_CHUNK)
            except OSError:
                break
            if not chunk:
                break
            output += chunk
            if len(output) > limit_bytes:
                outcome = "too_big"
                break
    finally:
        with guard:
            reading[0] = False
        timer.cancel()
    feeder.join(1)
    drainer.join(1)
    return "timeout" if timed_out.is_set() else outcome


def translate(
    reply: str,
    translator: str,
    *,
    timeout: Optional[float] = None,
    max_bytes: Optional[int] = None,
) -> str:
    """The text `translator` prints for `reply`. Raises `TranslatorError` when it cannot be used."""
    limit_seconds = TRANSLATOR_TIMEOUT_SECONDS if timeout is None else timeout
    limit_bytes = MAX_TRANSLATED_BYTES if max_bytes is None else max_bytes
    script = Path(translator)
    try:
        process = subprocess.Popen(
            _command(script),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(script.parent),
            env=_environment(),
            # Its own process group, so a timeout can stop what it started as well.
            start_new_session=os.name != "nt",
        )
    except (OSError, ValueError) as exc:
        raise TranslatorError(f"could not be started ({getattr(exc, 'strerror', None) or exc})") from exc

    deadline = time.monotonic() + limit_seconds
    data = reply.encode("utf-8")
    stderr_kept = bytearray()
    output = bytearray()
    try:
        if os.name == "nt":
            outcome = _pump_with_threads(process, data, limit_seconds, limit_bytes, output, stderr_kept)
        else:
            outcome = _pump_with_select(process, data, deadline, limit_bytes, output, stderr_kept)
        if outcome != "done":
            _stop(process)
        else:
            # The pipes are closed, but the translator may not have exited (it can close its output and carry
            # on): the rest of the deadline applies to that wait too.
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                outcome = "timeout"
                _stop(process)
    finally:
        _stop(process)
        process.wait()
        _stop_leftovers(process)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    timed_out = outcome == "timeout"
    too_big = outcome == "too_big"
    if timed_out:
        raise TranslatorError(f"did not finish within {limit_seconds:g} seconds and was stopped")
    if too_big:
        raise TranslatorError(f"wrote more than {limit_bytes} bytes and was stopped")
    if process.returncode != 0:
        detail = bytes(stderr_kept).decode("utf-8", "replace").strip()[:STDERR_SHOWN_CHARS]
        suffix = f": {detail}" if detail else ""
        raise TranslatorError(f"exited with status {process.returncode}{suffix}", brief=f"exited with status {process.returncode}",
                              printed=detail)
    try:
        text = bytes(output).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranslatorError("wrote output that is not UTF-8 text") from exc
    if not text.strip():
        raise TranslatorError("wrote nothing")
    return text


__all__ = [
    "MAX_TRANSLATED_BYTES",
    "TRANSLATOR_TIMEOUT_SECONDS",
    "TranslatorError",
    "translate",
]
