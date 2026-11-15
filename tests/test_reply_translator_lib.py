#!/usr/bin/env python3
"""Tests for libs/reply_translator_lib.py.

Stdlib-only (unittest), no network. Each test writes a throwaway script to a temp directory and runs it
for real under the interpreter running the tests, since what is under test is the process handling:
stdin and stdout, limits, the environment and the failure messages.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

from reply_translator_lib import TranslatorError, translate  # noqa: E402


class TranslatorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def script(self, body: str, name: str = "tr.py") -> str:
        path = self.dir / name
        path.write_text(body, encoding="utf-8")
        return str(path)


class Translating(TranslatorTestCase):
    def test_the_reply_goes_in_on_stdin_and_the_output_comes_back(self):
        path = self.script("import sys\nsys.stdout.write(sys.stdin.read().upper())\n")
        self.assertEqual(translate("finding: x", path), "FINDING: X")

    def test_non_ascii_text_survives_both_ways(self):
        path = self.script("import sys\nsys.stdout.write(sys.stdin.read() + ' ✓')\n")
        self.assertEqual(translate("héllo ☃", path), "héllo ☃ ✓")

    def test_a_reply_larger_than_a_pipe_buffer_does_not_deadlock(self):
        path = self.script("import sys\ndata = sys.stdin.read()\nsys.stdout.write(str(len(data)))\n")
        self.assertEqual(translate("x" * 3_000_000, path), "3000000")

    def test_a_script_that_answers_without_reading_its_input_is_fine(self):
        path = self.script("print('NONE')\n")
        self.assertEqual(translate("y" * 2_000_000, path).strip(), "NONE")

    def test_it_runs_in_the_scripts_own_directory(self):
        path = self.script("import os\nprint(os.getcwd())\n")
        self.assertEqual(os.path.realpath(translate("", path).strip()), os.path.realpath(str(self.dir)))

    def test_a_path_with_a_space_and_shell_characters_still_runs(self):
        # No shell is involved, so nothing in the path is interpreted.
        folder = self.dir / "a b; echo pwned"
        folder.mkdir()
        path = folder / "tr.py"
        path.write_text("print('ok')\n", encoding="utf-8")
        self.assertEqual(translate("", str(path)).strip(), "ok")

    @unittest.skipIf(os.name == "nt", "needs a shebang script")
    def test_a_script_that_is_not_python_runs_directly(self):
        path = self.script("#!/bin/sh\ncat\n", "convert")
        os.chmod(path, 0o755)
        self.assertEqual(translate("as is", path), "as is")


class Environment(TranslatorTestCase):
    def test_the_servers_own_settings_are_not_handed_to_it(self):
        path = self.script("import os\nprint(os.environ.get('QDRANT_API_KEY', 'absent'), os.environ.get('PATH') is not None)\n")
        with mock.patch.dict(os.environ, {"QDRANT_API_KEY": "secret"}):
            self.assertEqual(translate("", path).strip(), "absent True")

    def test_a_python_script_reads_and_writes_utf8(self):
        path = self.script("import os\nprint(os.environ.get('PYTHONUTF8'))\n")
        self.assertEqual(translate("", path).strip(), "1")


class ChildProcesses(TranslatorTestCase):
    """A translator can leave a child holding the output pipe; the deadline still has to hold."""

    def _child_script(self, pid_file: Path) -> str:
        return self.script(
            "import subprocess, sys\n"
            f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
            "print('FINDING')\n"
        )

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @unittest.skipIf(os.name == "nt", "process groups are POSIX")
    def test_a_child_that_keeps_stdout_open_after_the_translator_exits_hits_the_deadline(self):
        pid_file = self.dir / "child.pid"
        path = self._child_script(pid_file)
        started = time.monotonic()
        with self.assertRaises(TranslatorError) as caught:
            translate("", path, timeout=1)
        self.assertLess(time.monotonic() - started, 10)
        self.assertIn("did not finish within 1 seconds", str(caught.exception))
        # The child is stopped too, not left running in the background.
        pid = int(pid_file.read_text())
        for _ in range(40):
            if not self._alive(pid):
                break
            time.sleep(0.05)
        self.assertFalse(self._alive(pid))

    @unittest.skipIf(os.name == "nt", "process groups are POSIX")
    def test_a_child_left_running_after_a_normal_exit_is_stopped_and_leaves_no_thread_behind(self):
        # The child redirects its own stdout, so the output pipe closes and the reply reads fine, but it keeps the
        # translator's stdin and stderr pipes: with a big reply the feeding thread blocks on it for as long as it lives.
        pid_file = self.dir / "child.pid"
        path = self.script(
            "import subprocess, sys\n"
            f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=subprocess.DEVNULL)\n"
            f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
            "print('NONE')\n"
        )
        threads_before = threading.active_count()
        started = time.monotonic()
        self.assertEqual(translate("x" * 300_000, path, timeout=10).strip(), "NONE")
        self.assertLess(time.monotonic() - started, 1.5)  # not held up by the threads' join timeouts
        pid = int(pid_file.read_text())
        for _ in range(40):
            if not self._alive(pid):
                break
            time.sleep(0.05)
        self.assertFalse(self._alive(pid))
        for _ in range(40):
            if threading.active_count() <= threads_before:
                break
            time.sleep(0.05)
        self.assertLessEqual(threading.active_count(), threads_before)

    @unittest.skipIf(os.name == "nt", "select on pipes is POSIX")
    def test_a_helper_in_its_own_session_holding_stdout_does_not_hold_the_review_past_the_deadline(self):
        # The group signal cannot reach a process that started its own session, so only a bounded read helps.
        pid_file = self.dir / "helper.pid"
        path = self.script(
            "import subprocess, sys\n"
            "helper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)\n"
            f"open({str(pid_file)!r}, 'w').write(str(helper.pid))\n"
            "print('FINDING')\n"
        )
        started = time.monotonic()
        try:
            with self.assertRaises(TranslatorError) as caught:
                translate("", path, timeout=1)
            self.assertLess(time.monotonic() - started, 5)
            self.assertIn("did not finish within 1 seconds", str(caught.exception))
        finally:
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), 9)
                except OSError:
                    pass

    @unittest.skipIf(os.name == "nt", "select on pipes is POSIX")
    def test_a_helper_in_its_own_session_holding_stdin_and_stderr_does_not_block_a_finished_reply(self):
        pid_file = self.dir / "helper.pid"
        path = self.script(
            "import subprocess, sys\n"
            "helper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=subprocess.DEVNULL,\n"
            "                          start_new_session=True)\n"
            f"open({str(pid_file)!r}, 'w').write(str(helper.pid))\n"
            "print('NONE')\n"
        )
        started = time.monotonic()
        try:
            # stdout closed and the translator exited, so the reply is complete: the stdin and stderr the helper still
            # holds are not waited on.
            self.assertEqual(translate("x" * 300_000, path, timeout=10).strip(), "NONE")
            self.assertLess(time.monotonic() - started, 3)
        finally:
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), 9)
                except OSError:
                    pass

    @unittest.skipIf(os.name == "nt", "process groups are POSIX")
    def test_a_translator_that_closes_its_output_and_then_hangs_hits_the_deadline(self):
        path = self.script("import os, time\nprint('FINDING', flush=True)\nos.close(1)\ntime.sleep(30)\n")
        started = time.monotonic()
        with self.assertRaises(TranslatorError) as caught:
            translate("", path, timeout=1)
        self.assertLess(time.monotonic() - started, 10)
        self.assertIn("did not finish within 1 seconds", str(caught.exception))

    @unittest.skipIf(os.name == "nt", "process groups are POSIX")
    def test_a_child_that_floods_the_pipe_after_the_translator_exits_hits_the_output_limit(self):
        path = self.script(
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import sys\\nwhile True:\\n sys.stdout.write(\"x\" * 10000); sys.stdout.flush()'])\n"
        )
        with self.assertRaises(TranslatorError) as caught:
            translate("", path, max_bytes=50_000, timeout=20)
        self.assertIn("wrote more than 50000 bytes", str(caught.exception))


class Failing(TranslatorTestCase):
    def test_a_non_zero_exit_reports_the_status_and_the_start_of_stderr(self):
        path = self.script("import sys\nsys.stderr.write('boom ' + 'x' * 500)\nsys.exit(3)\n")
        with self.assertRaises(TranslatorError) as caught:
            translate("", path)
        message = str(caught.exception)
        self.assertIn("exited with status 3: boom", message)
        self.assertLess(len(message), 260)
        # What is kept on record leaves out what the translator printed: its stderr can echo the model's reply.
        self.assertEqual(caught.exception.brief, "exited with status 3")
        self.assertTrue(caught.exception.printed.startswith("boom "))
        self.assertIn(caught.exception.printed, message)  # exactly as it appears, so a record can withhold it

    def test_a_script_that_hangs_is_stopped_at_the_timeout(self):
        path = self.script("import time\ntime.sleep(60)\n")
        started = time.monotonic()
        with self.assertRaises(TranslatorError) as caught:
            translate("", path, timeout=0.5)
        self.assertLess(time.monotonic() - started, 10)
        self.assertIn("did not finish within 0.5 seconds", str(caught.exception))

    def test_output_over_the_limit_is_stopped(self):
        path = self.script("import sys\nwhile True:\n    sys.stdout.write('x' * 10000)\n    sys.stdout.flush()\n")
        with self.assertRaises(TranslatorError) as caught:
            translate("", path, max_bytes=50_000)
        self.assertIn("wrote more than 50000 bytes", str(caught.exception))

    def test_output_that_is_not_utf8_is_refused(self):
        path = self.script("import sys\nsys.stdout.buffer.write(b'\\xff\\xfe bad')\n")
        with self.assertRaises(TranslatorError) as caught:
            translate("", path)
        self.assertIn("not UTF-8", str(caught.exception))

    def test_empty_output_is_refused(self):
        path = self.script("pass\n")
        with self.assertRaises(TranslatorError) as caught:
            translate("", path)
        self.assertIn("wrote nothing", str(caught.exception))

    def test_a_script_that_cannot_be_started_says_so(self):
        with self.assertRaises(TranslatorError) as caught:
            translate("", str(self.dir / "gone"))
        self.assertIn("could not be started", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
