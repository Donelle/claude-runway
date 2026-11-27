#!/usr/bin/env python3
"""Tests for harness/orchestrator/shared_db_check.py: several containers recording into one shared reviewer database.

No Docker: each "container" is the script's own writer code run in a host process, on the folders the command would have
mounted. So the recording code, the barrier that makes the writers overlap, and the checks afterwards are real; what only a
real run can show, how Docker shares the folder between containers, is what the script is for.

Stdlib-only (unittest, no pytest).

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
for _directory in (REPO_ROOT / "harness" / "orchestrator", REPO_ROOT / "harness" / "pi", REPO_ROOT / "libs"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import pi_driver  # noqa: E402
import shared_db_check as check  # noqa: E402


def mounts_of(command: list) -> dict:
    """{path in the container: (host path, read-only)} of a `docker run` command."""
    found = {}
    for spec in (command[i + 1] for i, part in enumerate(command) if part == "--mount"):
        fields = dict(part.split("=", 1) for part in spec.split(",") if "=" in part)
        found[fields["target"]] = (Path(fields["source"]), spec.endswith(",readonly"))
    return found


class HostWriters:
    """A `spawn` that runs a command's writer code on the host. `home` takes what the recorder writes outside the database
    (its metrics and savings files), so a test never touches the real `~/.claude`."""

    def __init__(self, home: Path, script: str = "", exit_early: tuple = ()) -> None:
        self.home, self.script, self.exit_early = home, script, exit_early
        self.commands: list = []
        self.processes: list = []
        self.events: list = []

    def __call__(self, command: list, stderr_file):
        self.commands.append(command)
        self.events.append("spawn")
        mounts = mounts_of(command)
        database_dir, libs = mounts[pi_driver.C_DB][0], mounts[check.C_LIBS][0]
        at = command.index("-c")
        script, arguments = self.script or command[at + 1], command[at + 2:]
        if int(arguments[0]) in self.exit_early:
            script = "import sys; sys.exit('writer broke')"
        environment = {
            **os.environ, "HOME": str(self.home), "USERPROFILE": str(self.home),
            "CLAUDE_RUNWAY_METRICS_DB": str(self.home / "metrics.db"), "CLAUDE_RUNWAY_SAVINGS_DB": str(self.home / "savings.db"),
            "CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB": str(database_dir / pi_driver.DB_FILE_NAME),
            "CHECK_LIBS": str(libs), "CHECK_BARRIER": str(database_dir), "CHECK_BARRIER_SECONDS": "60",
        }
        process = subprocess.Popen([sys.executable, "-c", script, *arguments], stdout=subprocess.DEVNULL, stderr=stderr_file,
                                   env=environment)
        self.processes.append(process)
        wait = process.wait

        def waited(timeout=None):
            code = wait(timeout=timeout)
            self.events.append("exited")
            return code

        process.wait = waited
        return process


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name).resolve()
        self.home = self.folder / "home"
        self.home.mkdir()

    def trial(self, spawn, **changes) -> "check.TrialResult":
        values = {"writers": 3, "records": 4, "timeout": 120, "docker": "docker", "image": "img", "spawn": spawn,
                  "remove": lambda name: None}
        values.update(changes)
        return check.run_trial(self.folder / "db", **values)


class CommandTests(unittest.TestCase):
    def command(self) -> list:
        return check.writer_command(docker="docker", image="img", name="n", libs=Path("/x/libs"), database_dir=Path("/x/db"),
                                    index=2, writers=6, records=25, timeout=300)

    def test_a_writer_sees_only_the_libs_read_only_and_the_database_folder(self):
        mounts = mounts_of(self.command())
        self.assertEqual({target: read_only for target, (_source, read_only) in mounts.items()},
                         {check.C_LIBS: True, pi_driver.C_DB: False})
        self.assertEqual(mounts[pi_driver.C_DB][0], Path("/x/db"))  # the folder, never the database file in it

    def test_it_has_no_network_runs_under_timeout_and_names_its_database_in_the_mounted_folder(self):
        command = self.command()
        text = " ".join(command[:command.index("-c")])
        for expected in ("--rm", "--network none", "--cap-drop ALL", "HOME=/tmp",
                         f"CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB={pi_driver.C_DB}/{pi_driver.DB_FILE_NAME}",
                         f"--entrypoint {pi_driver.C_TIMEOUT} img --kill-after={pi_driver.KILL_AFTER_SECONDS} 300"):
            self.assertIn(expected, text)
        self.assertEqual(command[-3:], ["2", "6", "25"])


class TrialTests(Base):
    def test_writers_recording_at_once_save_every_record_through_the_real_recorder(self):
        writers = HostWriters(self.home)
        result = self.trial(writers)
        self.assertEqual((result.saved, result.integrity, result.exit_codes), (12, "ok", [0, 0, 0]), result.writer_errors)
        self.assertTrue(result.passed)
        with contextlib.closing(sqlite3.connect(self.folder / "db" / pi_driver.DB_FILE_NAME)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM runs").fetchone(), (12,))  # no record of the host's own
        self.assertEqual(len({command[command.index("--name") + 1] for command in writers.commands}), 3)

    def test_the_host_opens_the_database_only_before_the_first_container_and_after_the_last(self):
        writers = HostWriters(self.home)
        real = sqlite3.connect

        def watched(*args, **kwargs):
            writers.events.append("host opens")
            return real(*args, **kwargs)

        with mock.patch.object(sqlite3, "connect", watched):
            self.assertTrue(self.trial(writers).passed)
        events = writers.events
        first, last = events.index("spawn"), len(events) - 1 - events[::-1].index("exited")
        self.assertIn("host opens", events[:first])  # created before any container starts
        self.assertNotIn("host opens", events[first:last + 1])
        self.assertIn("host opens", events[last + 1:])

    def test_lost_records_fail_the_trial_even_when_every_writer_exits_0(self):
        # A failed database write only prints a warning in the real recorder, so exit codes alone would miss it.
        writers = HostWriters(self.home, script="import sys; print('could not write', file=sys.stderr)")
        result = self.trial(writers)
        self.assertEqual((result.saved, result.exit_codes, result.passed), (0, [0, 0, 0], False))
        self.assertIn("could not write", result.writer_errors)
        self.assertIn("0 of 12 records saved", check.describe(1, 1, result))
        self.assertIn("could not write", check.describe(1, 1, result))

    def test_a_writer_that_fails_fails_the_trial(self):
        result = self.trial(HostWriters(self.home, script="import sys; sys.exit(1)"))
        self.assertEqual((result.exit_codes, result.passed), ([1, 1, 1], False))

    def test_a_damaged_database_fails_the_trial_and_says_why(self):
        script = ("import os, sys; from pathlib import Path; "
                  "Path(os.environ['CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB']).write_bytes(b'garbage' * 200)")
        result = self.trial(HostWriters(self.home, script=script), writers=1)
        self.assertFalse(result.passed)
        self.assertIn("cannot be read", result.problem)

    def test_a_docker_that_refuses_the_container_is_not_a_failed_check(self):
        writers = HostWriters(self.home, script="import sys; print('no such image', file=sys.stderr); sys.exit(125)")
        with self.assertRaisesRegex(check.CannotRun, "no such image"):
            self.trial(writers)

    def test_an_interrupt_removes_every_container_already_started(self):
        removed: list = []
        writers = HostWriters(self.home, script="import time; time.sleep(60)")

        def spawn(command, stderr_file):
            if len(writers.processes) == 2:
                raise KeyboardInterrupt
            return writers(command, stderr_file)

        with self.assertRaises(KeyboardInterrupt):
            self.trial(spawn, remove=removed.append)
        for process in writers.processes:
            process.communicate(timeout=30)
        self.assertEqual(len(removed), 2)


class MainTests(Base):
    def run_main(self, *argv, spawn=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = check.main(["--folder", str(self.folder / "logs"), "--writers", "2", "--records", "2", *argv],
                              spawn=spawn or HostWriters(self.home))
        return code, out.getvalue(), err.getvalue()

    def test_every_trial_passing_is_exit_0_and_the_throwaway_folders_are_deleted(self):
        code, out, err = self.run_main("--trials", "2")
        self.assertEqual(code, 0, err + out)
        self.assertIn("2 of 2 trial(s) passed", out)
        self.assertEqual(list((self.folder / "logs").iterdir()), [])

    def test_a_failing_trial_is_exit_1_and_its_folder_is_kept(self):
        code, out, _err = self.run_main("--trials", "1", spawn=HostWriters(self.home, exit_early=(1,), script="pass"))
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)
        self.assertEqual(len(list((self.folder / "logs").iterdir())), 1)

    def test_no_docker_is_exit_2(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), mock.patch.object(check.pi_driver, "_docker", return_value=None):
            code = check.main(["--folder", str(self.folder / "logs"), "--docker", "no-such-docker-xyz", "--trials", "1"])
        self.assertEqual(code, 2, out.getvalue())
        self.assertIn("could not start", err.getvalue())
        self.assertEqual(list((self.folder / "logs").iterdir()), [])

    def test_a_trial_docker_refuses_leaves_no_folder_behind(self):
        refusing = HostWriters(self.home, script="import sys; print('no such image', file=sys.stderr); sys.exit(125)")
        code, _out, err = self.run_main("--trials", "1", spawn=refusing)
        self.assertEqual(code, 2)
        self.assertIn("no such image", err)
        self.assertEqual(list((self.folder / "logs").iterdir()), [])

    def test_an_interrupted_trial_leaves_no_folder_behind(self):
        seen: list = []

        def interrupted(database_dir, **kwargs):
            pi_driver.ensure_harness_db(database_dir)  # as far as a trial gets before its first container
            seen.append(database_dir)
            raise KeyboardInterrupt

        with mock.patch.object(check, "run_trial", interrupted), self.assertRaises(KeyboardInterrupt):
            self.run_main("--trials", "1")
        self.assertFalse(seen[0].parent.exists())

    def test_a_folder_that_cannot_be_deleted_is_named(self):
        with mock.patch.object(check.pi_driver, "_remove_tree", side_effect=OSError("in use")):
            code, _out, err = self.run_main("--trials", "1")
        (kept,) = (self.folder / "logs").iterdir()
        self.assertEqual(code, 0)
        self.assertIn(str(kept), err)

    def test_a_count_below_one_is_exit_2(self):
        self.assertEqual(self.run_main("--trials", "0")[0], 2)

    def test_the_default_folder_is_the_logs_folder_and_never_the_harness_database(self):
        seen: list = []

        def fake_trial(database_dir, **kwargs):
            seen.append(database_dir)
            raise check.CannotRun("stop here")

        with mock.patch.object(check, "run_trial", fake_trial), mock.patch.object(check, "REPO_ROOT", self.folder / "root"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(check.main([]), 2)
        logs = self.folder / "root" / "harness" / "live-tests" / "logs"
        self.assertEqual(seen[0].parent.parent, logs)
        self.assertNotEqual(seen[0], pi_driver.harness_db_dir(self.folder / "root"))


if __name__ == "__main__":
    unittest.main()
