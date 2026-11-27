#!/usr/bin/env python3
"""Tests for harness/pi/pi_driver.py: the capped driver that runs Pi in a container against the codebase-reviewer.

No Docker, no LM Studio, no network: the process the driver starts is replaced by a fake that plays back canned Pi events,
and the container kill by a recorder. What this proves is the driver's own logic (reading Pi's stream, counting, stopping,
and which host folders the container can see); that Pi, Docker and the MCP server work together is checked by hand (see
harness/pi/README.md).

Stdlib-only (unittest, no pytest).

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
PI_DIR = REPO_ROOT / "harness" / "pi"
for _directory in (PI_DIR, REPO_ROOT / "libs", REPO_ROOT / "harness"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import live_scoring_lib as scoring  # noqa: E402
import pi_driver as d  # noqa: E402
import score_run  # noqa: E402

# What stands in for the answer key's text, and for another change's own text. Neither may ever reach the container's repository.
KEY_MARKER = "ANSWER-KEY-MARKER-4f9d1c"
OTHER_CHANGE_MARKER = "CHANGE-5-ONLY-8a2e7b"
BASE_FILES = {
    "libs/version_lib.py": "base\n",
    ".github/instructions/rules.instructions.md": '---\napplyTo: "**/*.py"\n---\nA rule.\n',
}


def checkout_git(repo: Path, *args: str) -> str:
    """git in a fixture repository, with the user's own configuration left out (a signing key or hook of theirs must not matter)."""
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    done = subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
                          check=True, capture_output=True, text=True, errors="replace", env=env)
    return done.stdout


def make_checkout(parent: Path, change_ids=("change-5", "change-6")) -> Path:
    """A throwaway checkout shaped like the real one. `main` holds a file where the answer key is
    (`harness/live-tests/answer_key.json`, with a marker for its text). The branch's tags sit on a history that shares nothing with
    `main`: an orphan base, then one commit per change on top of the base, each changing `libs/version_lib.py`, as the real branch
    does. Back on `main` at the end, as a person's checkout is."""
    repo = parent / "checkout"
    repo.mkdir()
    checkout_git(repo, "init", "-q", "-b", "main")
    key = repo / "harness" / "live-tests" / "answer_key.json"
    key.parent.mkdir(parents=True)
    key.write_text(KEY_MARKER + "\n", encoding="utf-8")
    checkout_git(repo, "add", "-A")
    checkout_git(repo, "commit", "-q", "-m", "main")
    checkout_git(repo, "checkout", "-q", "--orphan", "snapshot")
    checkout_git(repo, "rm", "-rfq", ".")
    for relative, text in BASE_FILES.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    checkout_git(repo, "add", "-A")
    checkout_git(repo, "commit", "-q", "-m", "base")
    checkout_git(repo, "tag", scoring.BASE_REF)
    for change in change_ids:
        checkout_git(repo, "checkout", "-q", "--detach", scoring.BASE_REF)
        (repo / "libs" / "version_lib.py").write_text(f"{change}\n" + (OTHER_CHANGE_MARKER + "\n" if change == "change-5" else ""), encoding="utf-8")
        checkout_git(repo, "commit", "-q", "-am", change)
        checkout_git(repo, "tag", scoring.PREFIX + change)
    checkout_git(repo, "checkout", "-q", "main")
    return repo


# One checkout for the whole module: the driver only reads from it, and building it takes a dozen git calls.
_FIXTURE: dict = {}


def setUpModule():
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # git's object files are read-only, which Windows will not delete
    _FIXTURE["tmp"] = tmp
    _FIXTURE["checkout"] = make_checkout(Path(tmp.name))


def tearDownModule():
    _FIXTURE.pop("tmp").cleanup()


def checkout() -> Path:
    return _FIXTURE["checkout"]


def event(kind: str, **fields) -> bytes:
    return (json.dumps({"type": kind, **fields}) + "\n").encode("utf-8")


def turn(tool_names=()) -> bytes:
    """One turn as Pi emits it: turn_start, a tool call and its result for each name, then turn_end."""
    out = event("turn_start")
    for name in tool_names:
        out += event("tool_execution_start", toolCallId="c", toolName=name, args={})
        out += event("tool_execution_end", toolCallId="c", toolName=name, result={}, isError=False)
    return out + event("turn_end", message={"role": "assistant", "content": []}, toolResults=[])


def final_message(text: str) -> bytes:
    return event("message_end", message={"role": "assistant", "content": [{"type": "text", "text": text}]})


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeStdout:
    """A binary stream with `read1`: hands out the canned chunks, then either ends or, for a silent Pi, waits for a kill."""

    def __init__(self, chunks, hang: bool = False) -> None:
        self._chunks: queue.Queue = queue.Queue()
        for chunk in chunks:
            self._chunks.put(chunk)
        if not hang:
            self._chunks.put(b"")

    def read1(self, _size: int) -> bytes:
        return self._chunks.get()

    def close_stream(self) -> None:
        self._chunks.put(b"")


class FakeProcess:
    def __init__(self, chunks, hang: bool = False, returncode: int = 0, wait_times_out: bool = False) -> None:
        self.stdout = FakeStdout(chunks, hang)
        self.returncode = returncode
        self.killed = False
        self._wait_times_out = wait_times_out  # the first wait() raises TimeoutExpired, as a docker client that will not end does

    def wait(self, timeout=None) -> int:
        if self._wait_times_out:
            self._wait_times_out = False
            raise subprocess.TimeoutExpired("docker", timeout)
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class Recorder:
    """Stands in for `docker kill`: records that it ran and lets a hung fake stream end, as a killed container's does."""

    def __init__(self, process: FakeProcess) -> None:
        self.process, self.calls = process, 0

    def __call__(self) -> None:
        self.calls += 1
        self.process.stdout.close_stream()


def run(chunks, *, max_turns=10, max_tool_calls=30, timeout=600, hang=False, clock=None, returncode=0):
    """Runs the driver's loop over canned output; returns (result, kill recorder, tracker)."""
    clock = clock or FakeClock()
    tracker = d.CapTracker(max_turns, max_tool_calls, timeout, clock)
    process = FakeProcess(chunks, hang, returncode)
    recorder = Recorder(process)
    with tempfile.TemporaryDirectory() as tmp:
        result = d.run_capped(["docker", "run"], tracker, spawn=lambda command, stderr: process, kill_container=recorder,
                              stderr_path=Path(tmp) / "stderr.log")
    return result, recorder, tracker


class LineSplitterTests(unittest.TestCase):
    def test_splits_on_lf_and_strips_a_cr(self):
        self.assertEqual(d.LineSplitter().feed(b'{"a":1}\r\n{"b":2}\n'), ['{"a":1}', '{"b":2}'])

    def test_a_record_split_across_chunks_is_joined(self):
        splitter = d.LineSplitter()
        self.assertEqual(splitter.feed(b'{"a":'), [])
        self.assertEqual(splitter.feed(b'1}\n'), ['{"a":1}'])

    def test_a_multibyte_character_split_across_chunks_is_not_corrupted(self):
        record = '{"t":"é"}\n'.encode("utf-8")
        cut = record.index("é".encode("utf-8")) + 1  # between the two bytes of é
        splitter = d.LineSplitter()
        self.assertEqual(splitter.feed(record[:cut]), [])
        self.assertEqual(splitter.feed(record[cut:]), ['{"t":"é"}'])

    def test_unicode_line_separators_inside_a_string_do_not_split_the_record(self):
        # U+2028 and U+2029 are legal inside a JSON string. str.splitlines() would cut this record in three. The characters
        # are built with chr() because they are invisible and a file write once stripped them from the source, which left
        # a test that passed for any splitter; the assertions below fail if they are ever missing again.
        value = f"a{chr(0x2028)}b{chr(0x2029)}c"
        record = json.dumps({"t": value}, ensure_ascii=False)
        self.assertIn(chr(0x2028), record)
        self.assertIn(chr(0x2029), record)
        self.assertEqual(len(record.splitlines()), 3)  # what a splitlines-based reader would have made of it
        lines = d.LineSplitter().feed((record + "\n").encode("utf-8"))
        self.assertEqual(lines, [record])
        self.assertEqual(d.parse_event(lines[0]), {"t": value})

    def test_flush_returns_a_last_record_with_no_lf(self):
        splitter = d.LineSplitter()
        splitter.feed(b'{"a":1}')
        self.assertEqual(splitter.flush(), ['{"a":1}'])
        self.assertEqual(splitter.flush(), [])


class ParseEventTests(unittest.TestCase):
    def test_blank_non_json_and_non_object_lines_are_not_events(self):
        for line in ("", "   ", "not json", "[1, 2]", "3", '"text"'):
            self.assertIsNone(d.parse_event(line), line)

    def test_an_object_is_an_event(self):
        self.assertEqual(d.parse_event('{"type":"turn_start"}'), {"type": "turn_start"})

    def test_message_text_joins_the_text_blocks_of_an_assistant_message_only(self):
        content = [{"type": "text", "text": "a"}, {"type": "toolCall"}, {"type": "text", "text": "b"}]
        self.assertEqual(d.message_text({"message": {"role": "assistant", "content": content}}), "ab")
        self.assertEqual(d.message_text({"message": {"role": "user", "content": "hi"}}), "")
        self.assertEqual(d.message_text({"message": {"role": "assistant", "content": "plain"}}), "plain")
        self.assertEqual(d.message_text({}), "")


class CapTrackerTests(unittest.TestCase):
    def test_finishing_in_the_last_allowed_turn_is_not_a_cap(self):
        tracker = d.CapTracker(2, 30, 600, FakeClock())
        for _ in range(2):
            self.assertIsNone(tracker.observe({"type": "turn_start"}))
            tracker.observe({"type": "turn_end"})
        self.assertEqual((tracker.turns_started, tracker.turns), (2, 2))

    def test_the_turn_after_the_limit_trips_it(self):
        tracker = d.CapTracker(2, 30, 600, FakeClock())
        reasons = [tracker.observe({"type": "turn_start"}) for _ in range(3)]
        self.assertEqual(reasons, [None, None, "turns"])

    def test_a_flood_of_tool_calls_inside_one_turn_trips_the_tool_cap(self):
        # The case a turn cap alone misses: one turn, endless tool calls.
        tracker = d.CapTracker(10, 3, 600, FakeClock())
        tracker.observe({"type": "turn_start"})
        reasons = [tracker.observe({"type": "tool_execution_start", "toolName": "read"}) for _ in range(4)]
        self.assertEqual(reasons, [None, None, None, "tool_calls"])
        self.assertEqual(tracker.turns_started, 1)

    def test_the_clock_runs_out_without_any_event(self):
        clock = FakeClock()
        tracker = d.CapTracker(10, 30, 60, clock)
        self.assertFalse(tracker.out_of_time())
        clock.now = 61
        self.assertTrue(tracker.out_of_time())

    def test_review_calls_and_errors_are_counted_by_the_reviewer_tool_name(self):
        tracker = d.CapTracker(10, 30, 600, FakeClock())
        tracker.observe({"type": "tool_execution_start", "toolName": "read"})
        tracker.observe({"type": "tool_execution_start", "toolName": d.REVIEW_TOOL})
        tracker.observe({"type": "tool_execution_end", "toolName": d.REVIEW_TOOL, "isError": True})
        tracker.observe({"type": "tool_execution_end", "toolName": "read", "isError": True})
        self.assertEqual((tracker.tool_calls, tracker.review_calls, tracker.review_errors), (2, 1, 1))

    def test_the_last_assistant_text_is_kept_and_agent_settled_is_noted(self):
        tracker = d.CapTracker(10, 30, 600, FakeClock())
        tracker.observe(json.loads(final_message("first")))
        tracker.observe(json.loads(final_message("last")))
        tracker.observe({"type": "agent_settled", "aborted": False})
        self.assertEqual((tracker.final_text, tracker.settled), ("last", True))


class RunCappedTests(unittest.TestCase):
    def test_a_run_that_calls_the_reviewer_and_settles_completes(self):
        chunks = [turn([d.REVIEW_TOOL]), final_message("two findings"), event("agent_settled", aborted=False)]
        result, kill, _ = run(chunks)
        self.assertEqual(result.stopped_by, "completed")
        self.assertEqual((result.turns, result.tool_calls, result.review_calls), (1, 1, 1))
        self.assertEqual(result.final_text, "two findings")
        self.assertEqual(kill.calls, 0)
        self.assertEqual(result.exit_status(), 0)

    def test_an_aborted_settle_is_not_a_completed_run(self):
        # Pi settles with aborted=true when the run was cut short; whatever the container's exit code, that is not a pass.
        for container_exit in (0, 1, 130):
            result, _, tracker = run([turn([d.REVIEW_TOOL]), event("agent_settled", aborted=True)], returncode=container_exit)
            self.assertTrue(tracker.settled and tracker.aborted)
            self.assertEqual((result.stopped_by, result.exit_status()), ("aborted", 1), container_exit)

    def test_a_settle_followed_by_a_failing_container_is_an_error_not_a_pass(self):
        # Pi settles before it shuts its MCP server down, so an OOM kill (137) or a crash in that shutdown ends nonzero after a
        # clean settle: the run must not read as success, as the docs say a failed container is exit 1.
        for container_exit in (1, 137):
            result, _, tracker = run([turn([d.REVIEW_TOOL]), event("agent_settled", aborted=False)], returncode=container_exit)
            self.assertTrue(tracker.settled)
            self.assertEqual((result.stopped_by, result.exit_status()), ("error", 1), container_exit)

    def test_a_settle_that_is_not_aborted_still_completes(self):
        result, _, tracker = run([turn([d.REVIEW_TOOL]), event("agent_settled", aborted=False)])
        self.assertFalse(tracker.aborted)
        self.assertEqual((result.stopped_by, result.exit_status()), ("completed", 0))

    def test_a_run_that_never_calls_the_reviewer_is_not_a_pass(self):
        result, _, _ = run([turn(["read"]), event("agent_settled", aborted=False)])
        self.assertEqual(result.stopped_by, "completed")
        self.assertEqual(result.exit_status(), 1)

    def test_the_turn_cap_kills_the_container(self):
        result, kill, _ = run([turn() * 5], max_turns=2)
        self.assertEqual(result.stopped_by, "turns")
        self.assertEqual(kill.calls, 1)
        self.assertEqual(result.exit_status(), 3)

    def test_the_tool_call_cap_kills_the_container_inside_one_turn(self):
        flood = event("turn_start") + event("tool_execution_start", toolName="read") * 50
        result, kill, _ = run([flood], max_tool_calls=5)
        self.assertEqual((result.stopped_by, result.turns), ("tool_calls", 0))
        self.assertEqual((kill.calls, result.exit_status()), (1, 3))

    def test_a_silent_pi_is_stopped_by_the_wall_clock(self):
        # No bytes ever arrive, so only the clock can stop it. The clock is already past the limit.
        clock = FakeClock()
        clock.now = 1000
        tracker = d.CapTracker(10, 30, 600, clock)
        tracker._started = 0  # started long ago, so the 600 s are used up
        process = FakeProcess([], hang=True)
        recorder = Recorder(process)
        with tempfile.TemporaryDirectory() as tmp:
            result = d.run_capped(["docker"], tracker, spawn=lambda command, stderr: process, kill_container=recorder,
                                  stderr_path=Path(tmp) / "e.log")
        self.assertEqual((result.stopped_by, recorder.calls, result.exit_status()), ("timeout", 1, 3))

    def test_a_stream_that_ends_without_a_settle_is_an_error_and_the_container_is_stopped(self):
        # The output of a `docker` client that was killed or crashed ends while its container goes on running with no caps, so
        # a stream that ends without a clean finish is followed by a `docker kill` (a no-op if the container already exited).
        result, kill, _ = run([turn()], returncode=137)
        self.assertEqual((result.stopped_by, kill.calls, result.exit_status()), ("error", 1, 1))
        # A clean container exit does not make it a finished run: without a settle, Pi never said it was done.
        result, kill, _ = run([turn()], returncode=0)
        self.assertEqual((result.stopped_by, kill.calls, result.exit_status()), ("error", 1, 1))

    def test_a_settle_followed_by_a_failing_container_also_stops_it(self):
        result, kill, _ = run([turn(), event("agent_settled", aborted=False)], returncode=137)
        self.assertEqual((result.stopped_by, kill.calls), ("error", 1))

    def test_an_interrupt_stops_the_client_before_the_container(self):
        # If the daemon has not created the container yet `docker kill` fails, so the client must already be dead by then or it
        # goes on to create the container.
        order = []

        class Process(FakeProcess):
            def kill(self):
                order.append("client")
                super().kill()

        process = Process([turn()], hang=True)
        recorder = Recorder(process)

        def kill_container():
            order.append("container")
            recorder()

        def interrupt(_event):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(KeyboardInterrupt):
            d.run_capped(["docker"], d.CapTracker(10, 30, 600, FakeClock()), spawn=lambda c, e: process,
                         kill_container=kill_container, stderr_path=Path(tmp) / "e.log", on_event=interrupt)
        self.assertEqual(order, ["client", "container"])

    def test_an_interrupt_straight_after_the_client_starts_is_covered_too(self):
        process = FakeProcess([turn()], hang=True)
        recorder = Recorder(process)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(d.threading, "Thread", side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            d.run_capped(["docker"], d.CapTracker(10, 30, 600, FakeClock()), spawn=lambda c, e: process,
                         kill_container=recorder, stderr_path=Path(tmp) / "e.log")
        self.assertEqual((recorder.calls, process.killed), (1, True))

    def test_noise_between_events_is_ignored(self):
        chunks = [b"warning: something\n", b"\n", turn([d.REVIEW_TOOL]), b"{broken\n", event("agent_settled", aborted=False)]
        result, _, _ = run(chunks)
        self.assertEqual((result.stopped_by, result.review_calls), ("completed", 1))

    def test_events_split_across_reads_are_still_counted(self):
        whole = turn([d.REVIEW_TOOL]) + event("agent_settled", aborted=False)
        result, _, _ = run([whole[:17], whole[17:60], whole[60:]])
        self.assertEqual((result.turns, result.review_calls, result.stopped_by), (1, 1, "completed"))

    def test_the_reviewers_own_result_is_returned_beside_the_callers_summary(self):
        end = event("tool_execution_end", toolName=d.REVIEW_TOOL, isError=False,
                    result={"content": [{"type": "text", "text": "2 finding(s)\n- backup.py:6-8 shell injection"}]})
        result, _, _ = run([event("turn_start"), event("tool_execution_start", toolName=d.REVIEW_TOOL), end,
                            final_message("injection, off by one"), event("agent_settled", aborted=False)])
        self.assertEqual(result.final_text, "injection, off by one")
        self.assertEqual(result.review_text, "2 finding(s)\n- backup.py:6-8 shell injection")

    def test_a_refusal_the_review_tool_returns_as_text_counts_as_a_review_error(self):
        # The tool answers a refused review with ordinary text starting "Error:" and isError false (a bad reviewers file,
        # nothing to run). Without this the run would exit 0.
        for text in ("Error: no active reasoning reviewer is configured; nothing to run", "  Error: Reviewers file 'x' is invalid"):
            end = event("tool_execution_end", toolName=d.REVIEW_TOOL, isError=False, result={"content": [{"type": "text", "text": text}]})
            result, _, _ = run([event("turn_start"), event("tool_execution_start", toolName=d.REVIEW_TOOL), end,
                                event("agent_settled", aborted=False)])
            self.assertEqual((result.review_calls, result.review_errors, result.exit_status()), (1, 1, 1), text)

    def test_a_result_that_only_mentions_an_error_later_is_not_a_refusal(self):
        text = "Reasoning review: 1 finding(s)\n## error-handling (1)\n- a.py:3 Error: swallowed exception"
        end = event("tool_execution_end", toolName=d.REVIEW_TOOL, isError=False, result={"content": [{"type": "text", "text": text}]})
        result, _, _ = run([event("turn_start"), event("tool_execution_start", toolName=d.REVIEW_TOOL), end,
                            event("agent_settled", aborted=False)])
        self.assertEqual((result.review_errors, result.exit_status()), (0, 0))

    def test_a_long_reviewer_result_is_returned_whole(self):
        # It was once cut at 4,000 characters with no marker, which would have passed for the whole result.
        text = "finding " * 2000
        end = event("tool_execution_end", toolName=d.REVIEW_TOOL, isError=False, result={"content": [{"type": "text", "text": text}]})
        result, _, _ = run([event("turn_start"), end, event("agent_settled", aborted=False)])
        self.assertEqual(result.review_text, text)

    def test_when_every_review_call_errored_the_run_is_not_a_pass(self):
        errored = d.RunResult(stopped_by="completed", review_calls=2, review_errors=2)
        mixed = d.RunResult(stopped_by="completed", review_calls=2, review_errors=1)
        self.assertEqual((errored.exit_status(), mixed.exit_status()), (1, 0))

    def test_another_tools_result_is_not_taken_for_the_reviewers(self):
        end = event("tool_execution_end", toolName="read", isError=False, result={"content": [{"type": "text", "text": "file text"}]})
        result, _, _ = run([event("turn_start"), end, event("agent_settled", aborted=False)])
        self.assertEqual(result.review_text, "")

    def test_an_event_is_counted_before_it_is_reported(self):
        # The progress line says "turn N": N must already include the turn that has just started (it once printed turn 0).
        seen = []
        tracker = d.CapTracker(10, 30, 600, FakeClock())
        process = FakeProcess([turn() + turn() + event("agent_settled", aborted=False)])
        with tempfile.TemporaryDirectory() as tmp:
            d.run_capped(["docker"], tracker, spawn=lambda command, stderr: process, kill_container=Recorder(process),
                         stderr_path=Path(tmp) / "e.log",
                         on_event=lambda e: seen.append(tracker.turns_started) if e.get("type") == "turn_start" else None)
        self.assertEqual(seen, [1, 2])

    def test_an_exception_while_reading_still_stops_the_container(self):
        # A failed log write, say: without this the container would go on, uncapped, while its folders are deleted.
        process = FakeProcess([turn(), event("agent_settled", aborted=False)])
        recorder = Recorder(process)

        def explode(_event):
            raise OSError("log write failed")

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(OSError):
            d.run_capped(["docker"], d.CapTracker(10, 30, 600, FakeClock()), spawn=lambda c, e: process,
                         kill_container=recorder, stderr_path=Path(tmp) / "e.log", on_event=explode)
        self.assertEqual(recorder.calls, 1)

    def test_ctrl_c_while_waiting_for_the_client_after_the_stream_ended_still_stops_the_container(self):
        # The stream has drained and the read loop's cleanup has finished; the interrupt lands in the final wait.
        class InterruptedWait(FakeProcess):
            def wait(self, timeout=None):
                raise KeyboardInterrupt

        process = InterruptedWait([turn(), event("agent_settled", aborted=False)])
        recorder = Recorder(process)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(KeyboardInterrupt):
            d.run_capped(["docker"], d.CapTracker(10, 30, 600, FakeClock()), spawn=lambda c, e: process,
                         kill_container=recorder, stderr_path=Path(tmp) / "e.log")
        self.assertEqual((recorder.calls, process.killed), (1, True))

    def test_ctrl_c_while_reading_still_stops_the_container(self):
        process = FakeProcess([turn()], hang=True)
        recorder = Recorder(process)

        def interrupt(_event):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(KeyboardInterrupt):
            d.run_capped(["docker"], d.CapTracker(10, 30, 600, FakeClock()), spawn=lambda c, e: process,
                         kill_container=recorder, stderr_path=Path(tmp) / "e.log", on_event=interrupt)
        self.assertEqual(recorder.calls, 1)

    def test_a_client_that_will_not_end_has_its_container_stopped_and_not_only_the_client(self):
        process = FakeProcess([event("agent_settled", aborted=False)], wait_times_out=True)
        recorder = Recorder(process)
        with tempfile.TemporaryDirectory() as tmp:
            result = d.run_capped(["docker"], d.CapTracker(10, 30, 600, FakeClock()), spawn=lambda c, e: process,
                                  kill_container=recorder, stderr_path=Path(tmp) / "e.log")
        self.assertEqual((recorder.calls, process.killed), (1, True))
        self.assertEqual(result.stopped_by, "completed")

    def test_a_run_that_ends_by_itself_never_runs_docker_kill(self):
        result, kill, _ = run([turn([d.REVIEW_TOOL]), event("agent_settled", aborted=False)])
        self.assertEqual((result.stopped_by, kill.calls), ("completed", 0))

    def test_the_stderr_tail_is_returned(self):
        process = FakeProcess([event("agent_settled", aborted=False)])
        tracker = d.CapTracker(10, 30, 600, FakeClock())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.log"

            def spawn(command, stderr_file):
                stderr_file.write(b"reviewer: connected\n")
                stderr_file.flush()
                return process

            result = d.run_capped(["docker"], tracker, spawn=spawn, kill_container=Recorder(process), stderr_path=path)
        self.assertEqual(result.stderr_tail, ["reviewer: connected"])


def mounts_in(root: Path) -> d.Mounts:
    paths = {name: root / name for name in ("repo", "config", "agent", "state")}
    for path in paths.values():
        path.mkdir()
    return d.Mounts(d.REPO_ROOT / "libs", d.REPO_ROOT / "mcpservers", **paths)


def command_for(mounts: d.Mounts) -> list:
    return d.docker_run_command(
        docker="docker", image="img", name="n", mounts=mounts, model="m", prompt="p", memory="4g", cpus="2", pids_limit=256,
    )


class DockerCommandTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mounts = mounts_in(Path(self._tmp.name))
        self.command = command_for(self.mounts)

    def mount_specs(self) -> list:
        return [self.command[i + 1] for i, part in enumerate(self.command) if part == "--mount"]

    def test_the_reviewer_code_the_repo_and_the_config_are_read_only_and_only_two_mounts_are_writable(self):
        specs = self.mount_specs()
        self.assertEqual(len(specs), 6)
        writable = [spec for spec in specs if not spec.endswith(",readonly")]
        self.assertEqual(sorted(re.search(r"target=([^,]+)", s).group(1) for s in writable), [d.C_AGENT, d.C_STATE])

    def test_mounts_use_the_mount_flag_never_v_so_a_windows_path_cannot_be_misread(self):
        self.assertNotIn("-v", self.command)
        self.assertNotIn("--volume", self.command)

    def test_the_resource_limits_the_environment_and_the_hardening_are_present(self):
        text = " ".join(self.command)
        for expected in ("--memory 4g", "--cpus 2", "--pids-limit 256", "--cap-drop ALL", "no-new-privileges", "--init",
                         f"PI_CODING_AGENT_DIR={d.C_AGENT}", "PI_TELEMETRY=0", "PI_OFFLINE=1", "PI_SKIP_VERSION_CHECK=1", "--rm"):
            self.assertIn(expected, text)

    def test_pi_is_limited_to_read_only_tools_plus_the_reviewer_server(self):
        tools = self.command[self.command.index("--tools") + 1]
        self.assertEqual(tools, "read,grep,find,ls,mcp__reviewer__*")
        for forbidden in ("bash", "edit", "write"):
            self.assertNotIn(forbidden, tools.split(","))

    def test_the_prompt_is_one_argument_after_the_double_dash_not_text_in_a_shell_string(self):
        prompt = 'it\'s "quoted"; rm -rf /'
        command = d.docker_run_command(docker="docker", image="img", name="n", mounts=self.mounts, model="m", prompt=prompt,
                                       memory="4g", cpus="2", pids_limit=1)
        self.assertNotIn("sh", command[command.index("--entrypoint") + 1])  # no shell to splice it into...
        self.assertEqual(command.count(prompt), 1)
        self.assertEqual(command[-1], prompt)  # ...and handed over whole, as the last argument after "--"
        self.assertEqual(command[-2], "--")

    def test_the_mcp_file_registers_the_reviewer_with_the_scratch_database(self):
        server = d.mcp_file("http://host.docker.internal:1234/v1")["mcpServers"][d.MCP_SERVER_NAME]
        self.assertEqual(server["command"], d.C_VENV_PYTHON)
        self.assertEqual(server["exposure"], "direct")
        self.assertEqual(server["env"]["CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB"], f"{d.C_STATE}/codebase-reviewer.db")
        self.assertEqual(server["env"]["REVIEWERS_FILE"], f"{d.C_CONFIG}/reviewers.json")
        self.assertEqual(server["env"]["CLAUDE_RUNWAY_LMSTUDIO_URL"], "http://host.docker.internal:1234/v1")

    def test_the_mcp_file_leaves_both_timeouts_to_their_own_defaults_unless_asked(self):
        server = d.mcp_file("http://h:1234/v1")["mcpServers"][d.MCP_SERVER_NAME]
        self.assertNotIn("timeout", server)
        self.assertNotIn("CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS", server["env"])

    def test_each_timeout_goes_where_the_program_that_enforces_it_reads_it(self):
        # Pi reads `timeout` (seconds) off the server entry; the reviewer reads its own limit from its environment.
        server = d.mcp_file("http://h:1234/v1", mcp_timeout=300)["mcpServers"][d.MCP_SERVER_NAME]
        self.assertEqual(server["timeout"], 300)
        self.assertNotIn("CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS", server["env"])
        server = d.mcp_file("http://h:1234/v1", lmstudio_timeout=120)["mcpServers"][d.MCP_SERVER_NAME]
        self.assertNotIn("timeout", server)
        self.assertEqual(server["env"]["CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS"], "120")

    def test_display_url_hides_a_user_name_and_password_and_nothing_else(self):
        self.assertEqual(d.display_url("http://alice:s3cret@host.docker.internal:1234/v1"), "http://***@host.docker.internal:1234/v1")
        self.assertEqual(d.display_url("http://alice@h:1/v1"), "http://***@h:1/v1")
        self.assertEqual(d.display_url("http://h:1234/v1"), "http://h:1234/v1")

    def test_the_driver_source_spells_each_event_name_it_counts(self):
        # Only a spelling guard: it reads the driver's own source, so it cannot tell whether Pi uses these names. They come
        # from Pi's docs (json.md) and were seen in a real run's event log; the cap and counting tests above feed the same
        # names in, so a misspelling in the driver alone would make those fail.
        source = (PI_DIR / "pi_driver.py").read_text(encoding="utf-8")
        for name in ("turn_start", "turn_end", "tool_execution_start", "tool_execution_end", "agent_settled", "message_end"):
            self.assertIn(f'"{name}"', source)


class UnsafeMountsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.home = self.root / "home"
        (self.home / ".claude").mkdir(parents=True)
        (self.home / ".pi" / "agent").mkdir(parents=True)

    def fresh_mounts(self) -> d.Mounts:
        """A default Mounts in a new subdirectory, so one test can build several."""
        return mounts_in(Path(tempfile.mkdtemp(dir=self.root)))

    def test_the_default_layout_is_safe(self):
        self.assertEqual(d.unsafe_mounts(self.fresh_mounts(), self.home, d.REPO_ROOT), [])

    def test_the_checkout_itself_is_refused(self):
        mounts = self.fresh_mounts()
        mounts.repo = d.REPO_ROOT
        self.assertTrue(d.unsafe_mounts(mounts, self.home, d.REPO_ROOT))

    def test_a_folder_that_contains_the_checkout_is_refused(self):
        mounts = self.fresh_mounts()
        mounts.state = d.REPO_ROOT.parent
        self.assertTrue(d.unsafe_mounts(mounts, self.home, d.REPO_ROOT))

    def test_only_libs_and_mcpservers_may_come_from_inside_the_checkout_and_only_read_only(self):
        mounts = self.fresh_mounts()
        mounts.repo = d.REPO_ROOT / "tests"
        self.assertTrue(d.unsafe_mounts(mounts, self.home, d.REPO_ROOT))
        # The same folder as a writable mount is refused too: the exception is for read-only code.
        mounts = self.fresh_mounts()
        mounts.agent = d.REPO_ROOT / "libs"
        self.assertTrue(d.unsafe_mounts(mounts, self.home, d.REPO_ROOT))

    def test_claude_and_pi_folders_are_refused_whatever_the_mount_is_called(self):
        for sub in (".claude", ".pi", ".pi/agent"):
            mounts = self.fresh_mounts()
            mounts.agent = self.home / sub
            self.assertTrue(d.unsafe_mounts(mounts, self.home, d.REPO_ROOT), sub)

    def test_a_folder_that_contains_the_home_claude_folder_is_refused(self):
        mounts = self.fresh_mounts()
        mounts.state = self.home
        self.assertTrue(d.unsafe_mounts(mounts, self.home, d.REPO_ROOT))

    def fake_checkout(self) -> Path:
        checkout = Path(tempfile.mkdtemp(dir=self.root))
        (checkout / "libs").mkdir()
        (checkout / "mcpservers").mkdir()
        return checkout

    def test_the_reviewer_code_of_a_real_checkout_is_accepted(self):
        checkout = self.fake_checkout()
        mounts = self.fresh_mounts()
        mounts.libs, mounts.mcpservers = checkout / "libs", checkout / "mcpservers"
        self.assertEqual(d.unsafe_mounts(mounts, self.home, checkout), [])

    def test_a_symlinked_reviewer_folder_is_refused_wherever_it_points(self):
        # Without this check a link to an unrelated folder (say ~/.ssh) passed: it overlaps none of the three roots.
        outside = Path(tempfile.mkdtemp(dir=self.root))
        for name in ("libs", "mcpservers"):
            for target in (outside, self.home / ".claude"):
                checkout = self.fake_checkout()
                (checkout / name).rmdir()
                try:
                    (checkout / name).symlink_to(target, target_is_directory=True)
                except OSError as error:  # Windows without the symlink privilege
                    self.skipTest(f"cannot create a symlink here: {error}")
                mounts = self.fresh_mounts()
                mounts.libs, mounts.mcpservers = checkout / "libs", checkout / "mcpservers"
                problems = d.unsafe_mounts(mounts, self.home, checkout)
                self.assertTrue(any(f"own {name}/ folder" in p for p in problems), (name, target, problems))

    def test_reviewer_code_from_another_folder_is_refused(self):
        checkout = self.fake_checkout()
        mounts = self.fresh_mounts()
        mounts.libs = self.fake_checkout() / "libs"
        mounts.mcpservers = checkout / "mcpservers"
        self.assertTrue(d.unsafe_mounts(mounts, self.home, checkout))


class ReviewersFileTests(unittest.TestCase):
    def test_the_built_file_is_one_reasoning_reviewer_with_the_model_and_context_given(self):
        data = d.reviewers_file("some-model", 4096, True)
        (reviewer,) = data["reviewers"]
        self.assertEqual((reviewer["model"], reviewer["contextLength"], reviewer["loadOnDemand"], reviewer["type"]),
                         ("some-model", 4096, True, ["reasoning"]))

    def test_the_built_file_is_accepted_by_the_servers_own_validator(self):
        from codebase_reviewer_config_lib import validate_reviewers
        # The function's own output, as is. This once added the missing `rubric` key itself before validating, which hid
        # that the default (no --config) file was refused by the server.
        validate_reviewers(d.reviewers_file("m", 8192, False))

    def test_the_models_file_points_pi_at_lm_studio_with_the_model_and_context(self):
        data = d.models_file("a/b", 8192, "http://host.docker.internal:1234/v1")
        provider = data["providers"][d.PROVIDER]
        self.assertEqual((provider["api"], provider["baseUrl"]), ("openai-completions", "http://host.docker.internal:1234/v1"))
        self.assertEqual(provider["models"], [{"id": "a/b", "contextWindow": 8192}])


class ConfigTests(unittest.TestCase):
    def write(self, data) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    REVIEWER = {"name": "g", "model": "m1", "prompt": "p", "active": True, "type": ["reasoning"], "loadOnDemand": True, "contextLength": 8192}

    def test_stages_is_dropped_and_an_empty_rubric_is_added(self):
        path = self.write({"stages": ["reasoning"], "reviewers": [self.REVIEWER]})
        self.assertEqual(d.reviewers_from_config(path), {"rubric": [], "reviewers": [self.REVIEWER]})

    def test_every_real_non_qwen_reasoning_config_in_the_repo_converts(self):
        # The point of --config: the models the harness already records are used as they are.
        configs = sorted((REPO_ROOT / "harness" / "live-tests" / "configs").glob("reasoning-*.json"))
        # A `.baselines.json` beside a config is its score sidecar, not a config; a prompt-file config needs a file the container lacks.
        usable = [c for c in configs if ".baselines." not in c.name and "promptfile" not in c.name and "qwen" not in c.name]
        self.assertTrue(usable)
        for config in usable:
            with self.subTest(config=config.name):
                self.assertTrue(d.models_in(d.reviewers_from_config(config)))

    def test_a_config_using_a_file_the_container_lacks_is_refused(self):
        for key in ("promptFile", "translator"):
            path = self.write({"reviewers": [{**self.REVIEWER, key: "x.md"}]})
            with self.assertRaisesRegex(ValueError, key):
                d.reviewers_from_config(path)

    def test_a_config_with_no_active_reasoning_reviewer_is_refused(self):
        # The driver asks for review type "reasoning" and every type starts with that stage, so the server would refuse it
        # with a plain "Error:" result, which a run would then have reported as a pass.
        for reviewers in ([{**self.REVIEWER, "type": ["verifier"]}], [{**self.REVIEWER, "active": False}]):
            with self.subTest(reviewers=reviewers), self.assertRaisesRegex(ValueError, "no active reasoning reviewer"):
                d.reviewers_from_config(self.write({"reviewers": reviewers}))

    def test_a_config_whose_reasoning_reviewer_sits_beside_others_is_accepted(self):
        verifier = {**self.REVIEWER, "name": "v", "type": ["verifier"]}
        result = d.reviewers_from_config(self.write({"reviewers": [self.REVIEWER, verifier]}))
        self.assertEqual([r["name"] for r in result["reviewers"]], ["g", "v"])

    def test_a_rubric_that_is_not_a_list_is_refused_not_iterated(self):
        for rubric in (None, "text", 3, {"name": "x"}):
            with self.subTest(rubric=rubric), self.assertRaises(ValueError):
                d.reviewers_from_config(self.write({"rubric": rubric, "reviewers": [self.REVIEWER]}))

    def test_a_rubric_that_is_not_a_list_is_a_usage_error_not_a_traceback(self):
        path = self.write({"rubric": None, "reviewers": [self.REVIEWER]})
        completed = subprocess.run([sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m", "--config", str(path)],
                                   capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_missing_unreadable_and_invalid_configs_say_so(self):
        with self.assertRaisesRegex(ValueError, "cannot read"):
            d.reviewers_from_config(Path("does-not-exist.json"))
        with self.assertRaisesRegex(ValueError, "no 'reviewers'"):
            d.reviewers_from_config(self.write({"stages": []}))
        with self.assertRaises(ValueError):
            d.reviewers_from_config(self.write({"reviewers": [{"name": "g"}]}))

    def test_models_in_lists_active_reviewers_once_each_in_order(self):
        data = {"reviewers": [{**self.REVIEWER, "model": "b"}, {**self.REVIEWER, "model": "a"}, {**self.REVIEWER, "model": "b"},
                              {**self.REVIEWER, "model": "off", "active": False}]}
        self.assertEqual(d.models_in(data), ["b", "a"])


class ContainerUrlTests(unittest.TestCase):
    """The container cannot use `localhost` (that is itself), so a loopback URL has its host swapped and nothing else."""

    def test_localhost_becomes_the_container_alias_and_the_port_and_path_are_kept(self):
        self.assertEqual(d.container_url("http://localhost:1234/v1"), "http://host.docker.internal:1234/v1")

    def test_another_port_is_not_overwritten_with_a_default(self):
        self.assertEqual(d.container_url("http://localhost:5555/v1"), "http://host.docker.internal:5555/v1")

    def test_every_loopback_spelling_is_swapped(self):
        for host in ("localhost", "LOCALHOST", "127.0.0.1", "0.0.0.0", "[::1]"):
            self.assertEqual(d.container_url(f"http://{host}:8/x"), "http://host.docker.internal:8/x", host)

    def test_a_url_with_no_port_gets_none_added(self):
        self.assertEqual(d.container_url("https://localhost/v1"), "https://host.docker.internal/v1")

    def test_a_host_the_container_can_already_reach_is_left_alone(self):
        for url in ("http://192.168.1.20:1234/v1", "http://my-box.lan:1234/v1", "http://host.docker.internal:1234/v1"):
            self.assertEqual(d.container_url(url), url)

    def test_credentials_in_the_url_are_kept(self):
        self.assertEqual(d.container_url("http://user:pw@localhost:1234/v1"), "http://user:pw@host.docker.internal:1234/v1")

    def test_a_different_alias_can_be_given(self):
        self.assertEqual(d.container_url("http://localhost:1/v1", alias="host.containers.internal"), "http://host.containers.internal:1/v1")

    def test_a_port_that_is_not_a_number_is_an_error(self):
        with self.assertRaises(ValueError):
            d.container_url("http://localhost:abc/v1")


class LmStudioUrlTests(unittest.TestCase):
    def test_the_driver_writes_neither_the_address_nor_the_port_itself(self):
        # The point of runway_defaults: an assumption about where LM Studio is lives in one file, not here.
        source = (PI_DIR / "pi_driver.py").read_text(encoding="utf-8")
        for assumption in (":1234", "localhost:", "127.0.0.1:"):
            self.assertNotIn(assumption, source)

    def test_a_dry_run_follows_the_environment_variable(self):
        environment = {**os.environ, "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:5555/v1"}
        completed = subprocess.run([sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m",
                                    "--repo", str(checkout())],
                                   capture_output=True, text=True, timeout=60, env=environment)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("http://host.docker.internal:5555/v1", completed.stderr)

    def test_a_password_in_the_url_reaches_neither_the_command_nor_any_output(self):
        # The URL goes into files in the agent scratch folder; the command line is visible to `ps` and `docker inspect`.
        environment = {**os.environ, "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://alice:s3cret@localhost:5555/v1"}
        completed = subprocess.run([sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m",
                                    "--repo", str(checkout())],
                                   capture_output=True, text=True, timeout=60, env=environment)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("s3cret", completed.stdout + completed.stderr)
        self.assertIn("http://***@host.docker.internal:5555/v1", completed.stderr)

    def test_the_preflight_messages_hide_a_password_in_the_url(self):
        url = "http://alice:s3cret@127.0.0.1:1/v1"
        for known in (False, None):
            with mock.patch.object(d, "model_known", return_value=known), \
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                problem = d.host_model_check(url, "m") or ""
            self.assertNotIn("s3cret", problem + stderr.getvalue(), known)

    def test_a_bad_port_in_the_environment_is_a_usage_error_not_a_traceback(self):
        environment = {**os.environ, "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:abc/v1"}
        completed = subprocess.run([sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m",
                                    "--repo", str(checkout())],
                                   capture_output=True, text=True, timeout=60, env=environment)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("LM Studio URL", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


class MainTests(unittest.TestCase):
    def test_a_dry_run_prints_the_command_and_leaves_no_scratch_directory(self):
        before = set(Path(tempfile.gettempdir()).glob("claude-runway-pi-*"))
        completed = subprocess.run(
            [sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m", "--repo", str(checkout())],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(completed.stdout.startswith("docker run --rm"))
        self.assertIn("--mount", completed.stdout)
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("claude-runway-pi-*")), before)

    def test_an_unknown_change_and_a_nonpositive_cap_are_usage_errors(self):
        for extra in (["--change", "nope"], ["--max-turns", "0"], ["--timeout", "0"], ["--mcp-timeout", "0"],
                      ["--lmstudio-timeout", "-1"]):
            completed = subprocess.run([sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m",
                                        "--repo", str(checkout()), *extra],
                                       capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 2, extra)

    def test_the_timeout_flags_reach_the_mcp_file_the_container_is_given(self):
        written = {}

        def keep_then_remove(directory):
            written.update(json.loads((directory / "agent" / "mcp.json").read_text(encoding="utf-8")))
            remove(directory)

        remove = d._remove_tree
        with mock.patch.object(d, "_remove_tree", keep_then_remove), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = d.main(["--dry-run", "--model", "m", "--repo", str(checkout()), "--mcp-timeout", "300", "--lmstudio-timeout", "120"])
        self.assertEqual(code, 0)
        server = written["mcpServers"][d.MCP_SERVER_NAME]
        self.assertEqual((server["timeout"], server["env"]["CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS"]), (300, "120"))

    def test_config_cannot_be_combined_with_a_reviewer_model(self):
        config = REPO_ROOT / "harness" / "live-tests" / "configs" / "reasoning-gguf-seed-coder-8b.json"
        completed = subprocess.run(
            [sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m", "--config", str(config), "--reviewer-model", "x"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(completed.returncode, 2)


class UnsafeLogTests(unittest.TestCase):
    """`--log` is written by the driver on the host, so the container's mount checks do not cover it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.home = self.root / "home"
        (self.home / ".claude").mkdir(parents=True)
        (self.home / ".pi" / "agent").mkdir(parents=True)

    def test_paths_under_claude_and_pi_are_refused(self):
        for target in (".claude/events.jsonl", ".pi/events.jsonl", ".pi/agent/auth.json", ".pi/agent/not-yet/x.jsonl", ".pi", ".claude"):
            self.assertTrue(d.unsafe_log(self.home / target, self.home), target)

    def test_dot_dot_cannot_reach_around_the_check(self):
        self.assertTrue(d.unsafe_log(self.home / "elsewhere" / ".." / ".pi" / "agent" / "settings.json", self.home))

    def test_a_path_in_the_checkout_or_a_temp_folder_is_allowed(self):
        self.assertIsNone(d.unsafe_log(d.REPO_ROOT / "pi-events.jsonl", self.home))
        self.assertIsNone(d.unsafe_log(self.root / "events.jsonl", self.home))
        self.assertIsNone(d.unsafe_log(self.home / ".pi-notes.jsonl", self.home))  # a sibling that only shares a prefix

    def test_the_driver_refuses_such_a_log_before_running_anything(self):
        env = dict(os.environ, HOME=str(self.home), USERPROFILE=str(self.home))
        target = self.home / ".pi" / "agent" / "settings.json"
        target.write_text("precious", encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--model", "m", "--no-preflight", "--dry-run", "--log", str(target)],
            capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("~/.pi", completed.stderr)
        self.assertEqual(target.read_text(encoding="utf-8"), "precious")


class FailurePathTests(unittest.TestCase):
    """What happens when something outside the run is wrong: no docker, a bad log path, a model LM Studio lacks."""

    def run_driver(self, *extra):
        return subprocess.run([sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--model", "m", "--no-preflight",
                               "--repo", str(checkout()), *extra],
                              capture_output=True, text=True, timeout=120)

    def scratch_dirs(self) -> set:
        return set(Path(tempfile.gettempdir()).glob("claude-runway-pi-*"))

    def test_a_docker_that_cannot_be_started_is_a_usage_error_not_a_traceback(self):
        before = self.scratch_dirs()
        completed = self.run_driver("--docker", "no-such-docker-xyz")
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("could not start", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(self.scratch_dirs(), before)  # the scratch folder is removed on this path too

    def test_spawning_a_missing_executable_raises_launch_error(self):
        with tempfile.TemporaryFile() as stderr_file, self.assertRaises(d.LaunchError):
            d._spawn(["no-such-docker-xyz", "run"], stderr_file)

    def test_a_log_path_that_cannot_be_written_is_a_usage_error_not_a_traceback(self):
        before = self.scratch_dirs()
        with tempfile.TemporaryDirectory() as tmp:
            completed = self.run_driver("--log", str(Path(tmp) / "missing-dir" / "events.jsonl"))
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("--log", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(self.scratch_dirs(), before)

    def test_an_existing_log_file_is_refused_and_left_as_it_was(self):
        before = self.scratch_dirs()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            log.write_text("keep me", encoding="utf-8")
            completed = self.run_driver("--log", str(log))
            self.assertEqual(log.read_text(encoding="utf-8"), "keep me")
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("already exists", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(self.scratch_dirs(), before)

    def test_read_only_files_in_the_scratch_folder_are_still_removed(self):
        # Git's object files are read-only, which Windows will not delete without a chmod.
        directory = Path(tempfile.mkdtemp(prefix="claude-runway-pi-test-"))
        locked = directory / "objects" / "ab" / "cdef"
        locked.parent.mkdir(parents=True)
        locked.write_text("x", encoding="utf-8")
        os.chmod(locked, 0o444)
        d._remove_tree(directory)
        self.assertFalse(directory.exists())

    def test_a_model_lm_studio_does_not_list_is_a_preflight_problem(self):
        with mock.patch.object(d, "model_known", return_value=False):
            problem = d.host_model_check("http://h:1/v1", "ghost-model")
        self.assertIn("ghost-model", problem)

    def test_when_lm_studio_cannot_be_asked_the_run_goes_on_with_a_warning(self):
        err = io.StringIO()
        with mock.patch.object(d, "model_known", return_value=None), contextlib.redirect_stderr(err):
            problem = d.host_model_check("http://h:1/v1", "some-model")
        self.assertIsNone(problem)
        self.assertIn("could not ask", err.getvalue())

    def test_a_known_model_is_no_problem(self):
        with mock.patch.object(d, "model_known", return_value=True):
            self.assertIsNone(d.host_model_check("http://h:1/v1", "m"))


class ChangeRepositoryTests(unittest.TestCase):
    """The repository the container reviews: only the branch's base and the chosen change, fetched from a checkout."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.scratch = Path(self._tmp.name)
        self.tag = d.change_tag("change-6")
        self.repo = d.build_change_repo(self.scratch, checkout(), self.tag)

    def git(self, *args: str) -> str:
        return checkout_git(self.repo, *args)

    def commit_of(self, ref: str) -> str:
        return checkout_git(checkout(), "rev-parse", f"{ref}^{{commit}}").strip()

    def test_it_holds_the_base_and_the_change_and_nothing_else(self):
        base, change = self.commit_of(scoring.BASE_REF), self.commit_of("harness-live-tests/change-6")
        self.assertEqual(sorted(self.git("rev-list", "--all").split()), sorted([base, change]))
        self.assertEqual(sorted(self.git("for-each-ref", "--format=%(refname)").split()),
                         ["refs/tags/harness-live-tests/base", "refs/tags/harness-live-tests/change-6"])
        # Every object in the store is reachable from those two tags: no stray history, no other change's files.
        stored = set(self.git("cat-file", "--batch-all-objects", "--batch-check=%(objectname)").split())
        reachable = {line.split()[0] for line in self.git("rev-list", "--objects", "--all").splitlines() if line.strip()}
        self.assertEqual(stored, reachable)
        # Copied, not borrowed: nothing in it points back at the checkout it came from.
        self.assertFalse((self.repo / ".git" / "objects" / "info" / "alternates").exists())

    def test_the_change_is_checked_out_so_the_working_tree_is_the_change(self):
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.commit_of("harness-live-tests/change-6"))
        self.assertEqual(self.git("rev-parse", "--abbrev-ref", "HEAD").strip(), "HEAD")  # detached
        self.assertEqual(self.git("status", "--porcelain"), "")
        self.assertEqual((self.repo / "libs" / "version_lib.py").read_text(encoding="utf-8"), "change-6\n")
        # The repository's own review rules are in the working tree, where the reviewer reads them from.
        self.assertTrue((self.repo / ".github" / "instructions" / "rules.instructions.md").is_file())

    def test_the_reviewer_finds_the_repositorys_own_review_rules_in_it(self):
        # Why the change is checked out and not only fetched: the reviewer reads `.github/instructions/` from the working tree, so a
        # repository with no files would have been reviewed with no rules, unlike the real checkout it stands in for.
        import reasoning_reviewer_lib
        rules = reasoning_reviewer_lib.instruction_files_for(self.repo, ["libs/version_lib.py"])
        self.assertEqual([name for name, _text in rules], [".github/instructions/rules.instructions.md"])

    def test_the_diff_to_review_is_the_base_against_the_change(self):
        diff = self.git("diff", "--name-only", f"{scoring.BASE_REF}...harness-live-tests/change-6")
        self.assertEqual(diff.split(), ["libs/version_lib.py"])

    def test_nothing_from_main_or_from_another_change_is_anywhere_in_it(self):
        # The text that stands in for the answer key, and another change's own text, looked for in every object the repository
        # has (loose or packed) and in every file of its working tree.
        everything = self.git("cat-file", "--batch-all-objects", "--batch")
        files = "".join(p.read_text(encoding="utf-8", errors="replace") for p in self.repo.rglob("*") if p.is_file() and ".git" not in p.parts)
        for marker in (KEY_MARKER, OTHER_CHANGE_MARKER):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, everything)
                self.assertNotIn(marker, files)
        self.assertFalse((self.repo / "harness").exists())
        for commit in (self.commit_of(scoring.BASE_REF), self.commit_of("harness-live-tests/change-6")):
            self.assertNotIn("answer_key.json", self.git("ls-tree", "-r", "--name-only", commit))

    def test_another_change_gets_its_own_repository(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            repo = d.build_change_repo(Path(tmp), checkout(), d.change_tag("change-5"))
            self.assertEqual((repo / "libs" / "version_lib.py").read_text(encoding="utf-8"), f"change-5\n{OTHER_CHANGE_MARKER}\n")
            refs = checkout_git(repo, "for-each-ref", "--format=%(refname)").split()
            self.assertEqual(sorted(refs), ["refs/tags/harness-live-tests/base", "refs/tags/harness-live-tests/change-5"])

    def test_a_tag_the_checkout_lacks_stops_with_the_exact_fetch_command_and_builds_nothing(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            short = make_checkout(Path(tmp), change_ids=("change-5",))  # has the base and change-5, not change-6
            scratch = Path(tmp) / "scratch"
            scratch.mkdir()
            with self.assertRaises(d.ChangeRepoError) as caught:
                d.build_change_repo(scratch, short, self.tag)
            self.assertEqual(list(scratch.iterdir()), [])
        message = str(caught.exception)
        self.assertIn("harness-live-tests/change-6", message)
        self.assertNotIn("harness-live-tests/change-5", message)  # present, so not named as missing
        self.assertIn(scoring.FETCH_COMMAND, message)
        self.assertIn(str(short), message)

    def test_a_checkout_with_no_branch_at_all_names_the_base_too(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            bare = Path(tmp) / "plain-clone"
            bare.mkdir()
            checkout_git(bare, "init", "-q")
            with self.assertRaises(d.ChangeRepoError) as caught:
                d.check_checkout(bare, self.tag)
        self.assertIn("harness-live-tests/base, harness-live-tests/change-6", str(caught.exception))

    def test_a_folder_that_is_not_a_repository_is_refused_with_gits_own_reason(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            # Do not let git climb out of the temporary folder into some repository the machine happens to keep above it.
            with mock.patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": str(Path(tmp).parent)}), self.assertRaises(d.ChangeRepoError) as caught:
                d.check_checkout(Path(tmp), self.tag)
        self.assertIn(f"{tmp} cannot be used as the checkout that has the harness-live-tests/* tags", str(caught.exception))
        self.assertIn("not a git repository", str(caught.exception))

    def test_a_path_that_does_not_exist_is_refused(self):
        with self.assertRaises(d.ChangeRepoError) as caught:
            d.check_checkout(self.scratch / "no-such-folder", self.tag)
        self.assertIn("no-such-folder", str(caught.exception))

    def test_a_git_that_cannot_run_or_hangs_is_a_clean_error_not_a_traceback(self):
        for failure in (FileNotFoundError("git"), subprocess.TimeoutExpired("git", 1)):
            with self.subTest(failure=type(failure).__name__), mock.patch.object(d.subprocess, "run", side_effect=failure), \
                    self.assertRaises(d.ChangeRepoError) as caught:
                d.check_checkout(checkout(), self.tag)
            self.assertIn("could not run git", str(caught.exception))
        # The tag check goes through the scoring module's own git call.
        with mock.patch.object(d.scoring, "missing_refs", side_effect=FileNotFoundError("git")), self.assertRaises(d.ChangeRepoError) as caught:
            d.check_checkout(checkout(), self.tag)
        self.assertIn("could not run git", str(caught.exception))

    def test_a_fetch_that_fails_says_which_step_and_what_git_said(self):
        real = d._git

        def failing_fetch(repo, *args):
            if args[0] == "fetch":
                return subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: the remote hung up\nhint: more\n")
            return real(repo, *args)

        parent = self.scratch / "again"
        parent.mkdir()
        with mock.patch.object(d, "_git", failing_fetch), self.assertRaises(d.ChangeRepoError) as caught:
            d.build_change_repo(parent, checkout(), self.tag)
        message = str(caught.exception)
        self.assertIn("git fetch failed", message)
        self.assertIn("fatal: the remote hung up", message)
        self.assertNotIn("hint: more", message)  # the first line only

    def test_the_users_git_configuration_does_not_reach_the_repository(self):
        # A global autocrlf would rewrite the files the reviewer reads, and a global hooks path would run a hook on the checkout.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            hooks = Path(tmp) / "hooks"
            hooks.mkdir()
            ran = Path(tmp) / "hook-ran"
            hook = hooks / "post-checkout"
            hook.write_text(f"#!/bin/sh\ntouch '{ran}'\n", encoding="utf-8")
            hook.chmod(0o755)
            config = Path(tmp) / "gitconfig"
            config.write_text(f"[core]\n\tautocrlf = true\n\thooksPath = {hooks.as_posix()}\n", encoding="utf-8")
            target = Path(tmp) / "target"
            target.mkdir()
            with mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": str(config)}):
                repo = d.build_change_repo(target, checkout(), self.tag)
            self.assertNotIn(b"\r", (repo / "libs" / "version_lib.py").read_bytes())
            self.assertFalse(ran.exists())


class NoMountReachesTheAnswerKey(unittest.TestCase):
    """The answer key is in this checkout under `harness/live-tests/`; the container may see `libs/`, `mcpservers/` and scratch folders."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.mounts = d.prepare_scratch(self.root, checkout(), d.change_tag("change-6"), "m", d.reviewers_file("m", 8192, False), 8192,
                                        "http://host.docker.internal:1234/v1")

    def test_no_folder_the_container_is_given_holds_the_key(self):
        key = scoring.ANSWER_KEY_PATH.resolve()
        self.assertTrue(key.is_file())
        for host, inside, _writable in self.mounts.as_list():
            with self.subTest(mount=inside):
                resolved = host.resolve()
                self.assertNotEqual(resolved, key)
                self.assertNotIn(resolved, key.parents)

    def test_the_same_holds_for_the_sources_on_the_command_line_the_container_is_started_with(self):
        command = d.docker_run_command(docker="docker", image="img", name="n", mounts=self.mounts, model="m", prompt="p",
                                       memory="4g", cpus="2", pids_limit=1)
        sources = [re.search(r"source=([^,]+)", command[i + 1]).group(1) for i, part in enumerate(command) if part == "--mount"]
        self.assertEqual(len(sources), 6)
        key = scoring.ANSWER_KEY_PATH.resolve()
        for source in sources:
            with self.subTest(source=source):
                self.assertNotIn(Path(source).resolve(), [key, *key.parents])

    def test_the_checkout_the_repository_came_from_is_not_mounted_either(self):
        source = checkout().resolve()
        for host, inside, _writable in self.mounts.as_list():
            with self.subTest(mount=inside):
                resolved = host.resolve()
                self.assertNotIn(resolved, [source, *source.parents])  # not the checkout, and not a folder above it
                self.assertNotIn(source, resolved.parents)  # not something inside it

    def test_a_mount_of_the_key_or_of_a_folder_it_is_in_is_refused(self):
        home = self.root / "home"
        key = scoring.ANSWER_KEY_PATH
        for folder in (key, key.parent, key.parent.parent, REPO_ROOT / "harness" / "pi"):
            for field_name in ("repo", "state"):  # a read-only mount and a writable one
                mounts = mounts_in(Path(tempfile.mkdtemp(dir=self.root)))
                setattr(mounts, field_name, folder)
                with self.subTest(folder=str(folder), mount=field_name):
                    self.assertTrue(d.unsafe_mounts(mounts, home, d.REPO_ROOT))

    def test_the_key_is_in_harness_and_neither_of_the_two_folders_of_this_checkout_that_are_mounted(self):
        key = scoring.ANSWER_KEY_PATH.resolve()
        self.assertIn((REPO_ROOT / "harness").resolve(), key.parents)
        for mounted in (self.mounts.libs, self.mounts.mcpservers):
            self.assertNotIn(mounted.resolve(), key.parents)


class ChangeOptionTests(unittest.TestCase):
    """`--change` and `--repo` through `main`, in this process."""

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = d.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def scratch_dirs(self) -> set:
        return set(Path(tempfile.gettempdir()).glob("claude-runway-pi-*"))

    def test_the_default_change_is_change_6_and_the_prompt_names_its_two_tags(self):
        self.assertEqual(d.DEFAULT_CHANGE, "change-6")
        code, out, _err = self.run_main("--dry-run", "--model", "m", "--repo", str(checkout()))
        self.assertEqual(code, 0)
        prompt = shlex.split(out)[-1]
        self.assertIn('"repo_path": "/work/repo"', prompt)
        self.assertIn('"base_ref": "harness-live-tests/base"', prompt)
        self.assertIn('"head_ref": "harness-live-tests/change-6"', prompt)
        for placeholder in ("{repo}", "{base}", "{head}"):
            self.assertNotIn(placeholder, prompt)  # none is left unfilled

    def test_change_picks_the_change_and_a_custom_prompt_may_use_the_placeholders(self):
        code, out, _err = self.run_main("--dry-run", "--model", "m", "--repo", str(checkout()), "--change", "change-5",
                                        "--prompt", "review {base}..{head} in {repo}")
        self.assertEqual(code, 0)
        self.assertEqual(shlex.split(out)[-1], "review harness-live-tests/base..harness-live-tests/change-5 in /work/repo")

    def test_an_unknown_change_is_a_usage_error_that_names_the_known_ones_and_runs_nothing(self):
        before = self.scratch_dirs()
        code, out, err = self.run_main("--dry-run", "--model", "m", "--repo", str(checkout()), "--change", "change-9")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("unknown change 'change-9'", err)
        self.assertIn("change-1, change-2, change-3, change-4, change-5, change-6", err)
        self.assertEqual(self.scratch_dirs(), before)

    def test_a_change_written_with_the_branch_prefix_is_not_one(self):
        code, _out, err = self.run_main("--dry-run", "--model", "m", "--repo", str(checkout()), "--change", "harness-live-tests/change-6")
        self.assertEqual(code, 2)
        self.assertIn("unknown change", err)

    def test_a_checkout_without_the_tags_is_stopped_with_the_fetch_command_before_anything_slow(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            plain = Path(tmp) / "plain-clone"
            plain.mkdir()
            checkout_git(plain, "init", "-q")
            before = self.scratch_dirs()
            # Neither the model preflight nor the image build may run first: with `--build` and a `docker` that does not exist,
            # a late check would end with "could not start", and a late preflight would call the patched function.
            with mock.patch.object(d, "host_model_check", side_effect=AssertionError("the preflight ran first")):
                code, out, err = self.run_main("--model", "m", "--repo", str(plain), "--build", "--docker", "no-such-docker-xyz")
        self.assertEqual((code, out), (2, ""))
        self.assertIn(scoring.FETCH_COMMAND, err)
        self.assertNotIn("could not start", err)
        self.assertEqual(self.scratch_dirs(), before)

    def test_a_bad_port_is_still_reported_when_the_checkout_would_be_refused_too(self):
        # Arguments are checked before the checkout is: the cheaper check goes first.
        code, _out, err = self.run_main("--dry-run", "--model", "m", "--repo", "/no/such/checkout", "--host-lmstudio-url", "http://localhost:abc/v1")
        self.assertEqual(code, 2)
        self.assertIn("LM Studio URL", err)

    def test_a_missing_default_checkout_is_a_clean_message_naming_the_fetch(self):
        # Without --repo the checkout is this one; stand in for a clone that lacks the tags by pointing the default somewhere without them.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            plain = Path(tmp) / "clone"
            plain.mkdir()
            checkout_git(plain, "init", "-q")
            with mock.patch.object(d, "REPO_ROOT", plain):
                code, _out, err = self.run_main("--dry-run", "--model", "m")
        self.assertEqual(code, 2)
        self.assertIn(f"Run this in {plain}:", err)
        self.assertIn(scoring.FETCH_COMMAND, err)

    def test_the_old_case_flag_is_gone(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            d.build_parser().parse_args(["--model", "m", "--case", "seeded-three"])
        self.assertEqual(caught.exception.code, 2)


class ChangeInTheSummaryTests(unittest.TestCase):
    """A driver run on a change is scored by score_run.py against the key, with a fake Pi in place of the container."""

    REVIEW = ("Reasoning review: 1 finding(s) in 1 class(es) from 1 reviewer(s) (general), up to 1 diff chunk(s) each.\n"
              "These findings are not verified.\n\n## logic (1)\n- libs/version_lib.py:35 [general] the success test is inverted\n")

    def run_driver(self, *extra, review=REVIEW):
        chunks = [
            event("turn_start"),
            event("tool_execution_start", toolCallId="c", toolName=d.REVIEW_TOOL, args={}),
            event("tool_execution_end", toolCallId="c", toolName=d.REVIEW_TOOL, result={"content": [{"type": "text", "text": review}]}, isError=False),
            event("turn_end", message={"role": "assistant", "content": []}, toolResults=[]),
            final_message("one finding"),
            event("agent_settled", aborted=False),
        ]
        process = FakeProcess(chunks)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(d, "_spawn", lambda command, stderr_file: process), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = d.main(["--model", "m", "--no-preflight", "--repo", str(checkout()), *extra])
        return code, json.loads(out.getvalue())

    def test_the_summary_names_the_change_that_was_reviewed(self):
        code, summary = self.run_driver()
        self.assertEqual((code, summary["change"], summary["stopped_by"]), (0, "change-6", "completed"))
        self.assertEqual(self.run_driver("--change", "change-5")[1]["change"], "change-5")

    def test_the_change_leads_the_json_so_a_person_reading_it_sees_what_was_reviewed(self):
        _code, summary = self.run_driver()
        self.assertEqual(next(iter(summary)), "change")

    def test_score_run_scores_it_against_the_keys_answers_with_no_flag(self):
        _code, summary = self.run_driver()
        score = score_run.score_summary(summary)
        self.assertEqual((score.change, score.readable, score.found, score.missed, score.on_decoys), ("change-6", True, ["B7"], [], []))
        self.assertIn("change=change-6", score_run.one_line(summary, score))

    def test_the_same_run_on_the_clean_change_is_all_false_alarms(self):
        _code, summary = self.run_driver("--change", "change-5", review=self.REVIEW.replace("libs/version_lib.py", "libs/qdrant_retry.py"))
        score = score_run.score_summary(summary)
        self.assertEqual((score.change, len(score.clean_false_alarms), score.found, score.missed), ("change-5", 1, [], []))


class NeitherCasesNorHarnessLibTests(unittest.TestCase):
    def test_running_the_driver_and_the_scorer_imports_neither_cases_nor_harness_lib(self):
        # In a new interpreter: this process has long since imported both for the other harness tests. A dry run goes through every
        # step before docker, including building the repository from the tags; scoring goes through the key.
        program = (
            "import sys\n"
            f"sys.path.insert(0, {str(PI_DIR)!r})\n"
            "import pi_driver, score_run\n"
            f"code = pi_driver.main(['--dry-run', '--model', 'm', '--no-preflight', '--repo', {str(checkout())!r}])\n"
            "score_run.score_summary({'change': 'change-6', 'review_text': 'Reasoning review: 0 finding(s)\\n'})\n"
            "print('RESULT', code, sorted(m for m in ('cases', 'harness_lib', 'live_branch_lib') if m in sys.modules))\n"
        )
        completed = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip().splitlines()[-1], "RESULT 0 []")

    def test_neither_source_names_either_module_in_an_import(self):
        for name in ("pi_driver.py", "score_run.py"):
            source = (PI_DIR / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertNotRegex(source, r"(?m)^\s*(import|from)\s+(cases|harness_lib|live_branch_lib)\b")


class ModelKnownTests(unittest.TestCase):
    def listing(self, *ids) -> io.BytesIO:
        return io.BytesIO(json.dumps({"data": [{"id": model} for model in ids]}).encode("utf-8"))

    def test_a_listed_model_is_known_and_an_unlisted_one_is_not(self):
        with mock.patch.object(d.urllib.request, "urlopen", return_value=self.listing("a", "b")) as opened:
            self.assertIs(d.model_known("http://host:1234/v1", "a"), True)
        self.assertEqual(opened.call_args.args[0], "http://host:1234/api/v0/models")  # the REST API is at the root, not under /v1
        with mock.patch.object(d.urllib.request, "urlopen", return_value=self.listing("a")):
            self.assertIs(d.model_known("http://host:1234/v1/", "z"), False)

    def test_a_server_that_cannot_be_asked_or_answers_oddly_is_unknown_not_a_no(self):
        for outcome in (urllib.error.URLError("refused"), TimeoutError("slow"), io.BytesIO(b"not json"), io.BytesIO(b'{"nodata": 1}')):
            with self.subTest(outcome=repr(outcome)[:30]):
                kwargs = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
                with mock.patch.object(d.urllib.request, "urlopen", **kwargs):
                    self.assertIsNone(d.model_known("http://host:1234/v1", "a"))


class DockerfileTests(unittest.TestCase):
    """Cheap guards against drift in the things the image must keep; building it is checked by hand."""

    def setUp(self):
        self.text = (PI_DIR / "Dockerfile").read_text(encoding="utf-8")

    def test_it_keeps_the_pieces_the_run_depends_on(self):
        for expected in ("--ignore-scripts", "--python 3.12", "safe.directory", "ARG PI_VERSION=", "ripgrep", "mcp[cli]>=2.0.0"):
            self.assertIn(expected, self.text)

    def test_it_never_bakes_the_checkout_or_a_credential_in(self):
        self.assertNotRegex(self.text, r"(?m)^\s*(COPY|ADD)\s+(?!--from)")
        self.assertNotIn("API_KEY", self.text)


if __name__ == "__main__":
    unittest.main()
