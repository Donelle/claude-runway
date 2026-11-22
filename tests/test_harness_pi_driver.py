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
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
PI_DIR = REPO_ROOT / "harness" / "pi"
for _directory in (PI_DIR, REPO_ROOT / "libs", REPO_ROOT / "harness"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import pi_driver as d  # noqa: E402


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

    def test_a_container_that_dies_without_settling_is_an_error(self):
        result, kill, _ = run([turn()], returncode=137)
        self.assertEqual((result.stopped_by, kill.calls, result.exit_status()), ("error", 0, 1))

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
        docker="docker", image="img", name="n", mounts=mounts, model="m", prompt="p", lmstudio_url="http://host.docker.internal:1234/v1",
        memory="4g", cpus="2", pids_limit=256,
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
                                       lmstudio_url="u", memory="4g", cpus="2", pids_limit=1)
        script = command[command.index("-c") + 1]
        self.assertNotIn(prompt, script)  # never spliced into the shell text...
        self.assertNotIn("rm -rf", script)
        self.assertEqual(command[-1], prompt)  # ...but handed over whole, as the last argument after "--"
        self.assertEqual(command[-2], "--")

    def test_the_container_script_registers_the_reviewer_with_the_scratch_database_and_keeps_stdout_clean(self):
        script = d.container_script("http://host.docker.internal:1234/v1")
        self.assertIn(f"CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB={d.C_STATE}/codebase-reviewer.db", script)
        self.assertIn(f"REVIEWERS_FILE={d.C_CONFIG}/reviewers.json", script)
        self.assertIn(">&2 && exec pi", script)

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


class ReviewersFileTests(unittest.TestCase):
    def test_the_built_file_is_one_reasoning_reviewer_with_the_model_and_context_given(self):
        data = d.reviewers_file("some-model", 4096, True)
        (reviewer,) = data["reviewers"]
        self.assertEqual((reviewer["model"], reviewer["contextLength"], reviewer["loadOnDemand"], reviewer["type"]),
                         ("some-model", 4096, True, ["reasoning"]))

    def test_the_built_file_is_accepted_by_the_servers_own_validator(self):
        from codebase_reviewer_config_lib import validate_reviewers
        validate_reviewers({"rubric": [], **d.reviewers_file("m", 8192, False)})

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
        completed = subprocess.run([sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m"],
                                   capture_output=True, text=True, timeout=60, env=environment)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CLAUDE_RUNWAY_LMSTUDIO_URL=http://host.docker.internal:5555/v1", completed.stdout)

    def test_a_bad_port_in_the_environment_is_a_usage_error_not_a_traceback(self):
        environment = {**os.environ, "CLAUDE_RUNWAY_LMSTUDIO_URL": "http://localhost:abc/v1"}
        completed = subprocess.run([sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m"],
                                   capture_output=True, text=True, timeout=60, env=environment)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("LM Studio URL", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


class MainTests(unittest.TestCase):
    def test_a_dry_run_prints_the_command_and_leaves_no_scratch_directory(self):
        before = set(Path(tempfile.gettempdir()).glob("claude-runway-pi-*"))
        completed = subprocess.run(
            [sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(completed.stdout.startswith("docker run --rm"))
        self.assertIn("--mount", completed.stdout)
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("claude-runway-pi-*")), before)

    def test_an_unknown_case_and_a_nonpositive_cap_are_usage_errors(self):
        for extra in (["--case", "nope"], ["--max-turns", "0"], ["--timeout", "0"]):
            completed = subprocess.run([sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m", *extra],
                                       capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 2, extra)

    def test_config_cannot_be_combined_with_a_reviewer_model(self):
        config = REPO_ROOT / "harness" / "live-tests" / "configs" / "reasoning-seed-coder-8b.json"
        completed = subprocess.run(
            [sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--dry-run", "--model", "m", "--config", str(config), "--reviewer-model", "x"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(completed.returncode, 2)


class FailurePathTests(unittest.TestCase):
    """What happens when something outside the run is wrong: no docker, a bad log path, a model LM Studio lacks."""

    def run_driver(self, *extra):
        return subprocess.run([sys.executable, "-I", str(PI_DIR / "pi_driver.py"), "--model", "m", "--no-preflight", *extra],
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
        import harness_lib
        with mock.patch.object(harness_lib, "model_known", return_value=False):
            problem = d.host_model_check("http://h:1/v1", "ghost-model")
        self.assertIn("ghost-model", problem)

    def test_when_lm_studio_cannot_be_asked_the_run_goes_on_with_a_warning(self):
        import harness_lib
        err = io.StringIO()
        with mock.patch.object(harness_lib, "model_known", return_value=None), contextlib.redirect_stderr(err):
            problem = d.host_model_check("http://h:1/v1", "some-model")
        self.assertIsNone(problem)
        self.assertIn("could not ask", err.getvalue())

    def test_a_known_model_is_no_problem(self):
        import harness_lib
        with mock.patch.object(harness_lib, "model_known", return_value=True):
            self.assertIsNone(d.host_model_check("http://h:1/v1", "m"))


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
