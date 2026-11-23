#!/usr/bin/env python3
"""Tests for `harness_lib.isolate_reviewer_db`: a harness run records in a throwaway reviewer database (and metrics and
savings database) unless the caller chose one, and never in the real ones under `~/.claude`.

Stdlib-only (unittest), no network, no LM Studio: LM Studio's REST API is `tests/fake_lmstudio.py` and the model's answer is a
stub, as in `test_harness_classifier.py`, whose set-up this reuses. The home folder is a temporary one, so "the default
database" is a path in a sandbox that the test can assert was never created; the real `~/.claude` is not touched.

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_TESTS = REPO_ROOT / "harness" / "live-tests"
for _directory in (LIVE_TESTS, REPO_ROOT / "harness", REPO_ROOT / "libs", REPO_ROOT / "tests"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import harness_lib as h  # noqa: E402
import run_live_tests as rlt  # noqa: E402
import test_harness_classifier as tc  # noqa: E402
import test_harness_live_tests as tl  # noqa: E402
from harness_lib import REVIEWER_DB_ENV, SIDE_STORE_ENVS, isolate_reviewer_db  # noqa: E402

METRICS_ENV, SAVINGS_ENV = (variable for variable, _ in SIDE_STORE_ENVS)
ALL_ENVS = (REVIEWER_DB_ENV, METRICS_ENV, SAVINGS_ENV)
DEFAULT_DB = Path(".claude") / "claude-runway" / "codebase-reviewer.db"


def sandbox_home(case: unittest.TestCase, folder: Path) -> Path:
    """Make `Path.home()` a folder of the test's own, so the real default paths can be asserted never to be created."""
    home = folder / "home"
    home.mkdir()
    patcher = mock.patch.object(Path, "home", return_value=home)
    patcher.start()
    case.addCleanup(patcher.stop)
    return home


class Sandboxed(unittest.TestCase):
    """A home folder that is a temp folder, and the three variables unset: what a harness run starts from."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name).resolve()
        # `os.environ` is restored at cleanup, so what the helper sets does not leak into other tests.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for variable in ALL_ENVS:
            os.environ.pop(variable, None)
        self.home = sandbox_home(self, self.dir)

    def isolate(self) -> tuple:
        """`(path, announced)`: what the helper returns, and what it printed."""
        said: list = []
        path = isolate_reviewer_db(said.append)
        # The throwaway folder is under the system temp folder, not inside this test's own: remove it when the test ends.
        self.addCleanup(shutil.rmtree, path.parent, True)
        return path, said


class Helper(Sandboxed):
    def test_unset_points_the_variable_at_a_new_folder_under_the_system_temp_folder(self):
        path, _ = self.isolate()
        self.assertEqual(os.environ[REVIEWER_DB_ENV], str(path))
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.parent.parent, Path(tempfile.gettempdir()))
        self.assertTrue(path.parent.name.startswith("claude-runway-harness-"))
        self.assertEqual(path.name, "codebase-reviewer.db")

    def test_unset_does_not_make_the_real_default_the_chosen_path(self):
        path, _ = self.isolate()
        self.assertNotEqual(path, self.home / DEFAULT_DB)
        self.assertFalse((self.home / ".claude").exists())

    def test_the_metrics_and_savings_databases_move_with_it_into_the_same_folder(self):
        path, _ = self.isolate()
        self.assertEqual(Path(os.environ[METRICS_ENV]), path.parent / "metrics.db")
        self.assertEqual(Path(os.environ[SAVINGS_ENV]), path.parent / "savings.db")

    def test_a_metrics_or_savings_path_the_caller_set_is_kept_in_throwaway_mode(self):
        mine = str(self.dir / "mine-metrics.db")
        os.environ[METRICS_ENV] = mine
        path, _ = self.isolate()
        self.assertEqual(os.environ[METRICS_ENV], mine)
        self.assertEqual(Path(os.environ[SAVINGS_ENV]), path.parent / "savings.db")

    def test_a_callers_path_is_kept_and_is_what_resolve_db_path_returns(self):
        import reviewer_db_lib
        mine = self.dir / "mine" / "r.db"
        os.environ[REVIEWER_DB_ENV] = str(mine)
        path, _ = self.isolate()
        self.assertEqual(path, mine)
        self.assertEqual(os.environ[REVIEWER_DB_ENV], str(mine))
        self.assertEqual(reviewer_db_lib.resolve_db_path(), mine)

    def test_a_caller_who_chose_the_reviewer_database_leaves_the_metrics_and_savings_ones_alone(self):
        os.environ[REVIEWER_DB_ENV] = str(self.dir / "mine.db")
        self.isolate()
        self.assertNotIn(METRICS_ENV, os.environ)
        self.assertNotIn(SAVINGS_ENV, os.environ)

    def test_a_relative_path_is_anchored_at_the_home_folder_as_the_reviewer_reads_it(self):
        import reviewer_db_lib
        os.environ[REVIEWER_DB_ENV] = str(Path("runs") / "x.db")
        path, said = self.isolate()
        self.assertEqual(path, self.home / "runs" / "x.db")
        self.assertEqual(path, reviewer_db_lib.resolve_db_path())
        self.assertIn(str(self.home / "runs" / "x.db"), said[0])

    def test_an_empty_variable_counts_as_unset_as_resolve_db_path_treats_it(self):
        os.environ[REVIEWER_DB_ENV] = ""
        path, _ = self.isolate()
        self.assertEqual(os.environ[REVIEWER_DB_ENV], str(path))
        self.assertEqual(path.parent.parent, Path(tempfile.gettempdir()))

    def test_the_chosen_path_and_the_log_folder_are_announced_in_both_cases(self):
        import reviewer_db_lib
        path, said = self.isolate()
        self.assertEqual(len(said), 1)
        self.assertIn(str(path), said[0])
        self.assertIn(str(reviewer_db_lib.log_dir(path)), said[0])
        self.assertIn("throwaway", said[0])
        mine = self.dir / "mine.db"
        os.environ[REVIEWER_DB_ENV] = str(mine)
        said.clear()
        isolate_reviewer_db(said.append)
        self.assertEqual(len(said), 1)
        self.assertIn(str(mine), said[0])
        self.assertNotIn("throwaway", said[0])

    def test_two_runs_get_two_folders(self):
        first, _ = self.isolate()
        for variable in ALL_ENVS:
            os.environ.pop(variable)
        second, _ = self.isolate()
        self.assertNotEqual(first.parent, second.parent)

    def test_the_folder_is_created_new_and_a_name_that_exists_is_never_reused(self):
        # A name made from the time and the process id can be guessed: another user (or a link) could be there first, and the
        # databases would then be written into a folder they own. The folder is created atomically, so one that exists is skipped.
        with mock.patch.object(h.time, "strftime", return_value="20991231-235959"):
            first, _ = self.isolate()
            os.environ.pop(REVIEWER_DB_ENV)
            second, _ = self.isolate()
        self.assertTrue(first.parent.is_dir() and second.parent.is_dir())
        self.assertNotEqual(first.parent, second.parent)
        self.assertTrue(first.parent.name.startswith("claude-runway-harness-20991231-235959-"))

    def test_importing_the_module_does_not_touch_the_variables(self):
        env = {k: v for k, v in os.environ.items() if k not in ALL_ENVS}
        code = ("import sys, os; sys.path[:0] = [r'%s', r'%s']; import harness_lib; "
                "print([v for v in %r if v in os.environ])" % (REPO_ROOT / "harness", REPO_ROOT / "libs", ALL_ENVS))
        done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, check=False)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "[]"), done.stderr)


class AStageScriptRecordsInTheThrowawayDatabase(tc.Isolated):
    """The ticket's check: a stage script run with a fake LM Studio and the variables unset leaves the default paths alone."""

    def setUp(self):
        super().setUp()
        # `Isolated` pins all three to temp files; these tests are about what a run does when they are not set.
        for variable in ALL_ENVS:
            os.environ.pop(variable)
        self.home = sandbox_home(self, self.dir)

    def test_a_run_with_the_variables_unset_writes_only_under_the_system_temp_folder(self):
        self.lmstudio(lambda client, model, system, content: tc.perfect_probability(content))
        code, out = self.run_main("--config", self.config(), "--runs", "1")
        self.assertEqual(code, 0, out)
        throwaway = Path(os.environ[REVIEWER_DB_ENV])
        self.addCleanup(shutil.rmtree, throwaway.parent, True)
        self.assertEqual(throwaway.parent.parent, Path(tempfile.gettempdir()))
        self.assertTrue(throwaway.is_file(), "the run recorded nothing in the throwaway database")
        # The summary event a closed run writes went to the throwaway metrics database, not the shared one.
        self.assertTrue((throwaway.parent / "metrics.db").is_file(), "the run's summary event did not reach the throwaway metrics database")
        self.assertIn(f"reviewer database: {throwaway}", out)  # said at the start so the logs can be found
        # Nothing was created under the home folder: no default database, no metrics database, no log folder.
        self.assertFalse((self.home / ".claude").exists())

    def test_a_run_with_the_reviewer_variable_set_records_where_the_caller_said(self):
        mine = self.dir / "mine" / "r.db"
        os.environ[REVIEWER_DB_ENV] = str(mine)
        os.environ[METRICS_ENV] = str(self.dir / "mine-metrics.db")
        self.lmstudio(lambda client, model, system, content: tc.perfect_probability(content))
        code, out = self.run_main("--config", self.config(), "--runs", "1")
        self.assertEqual(code, 0, out)
        self.assertTrue(mine.is_file())
        self.assertTrue((self.dir / "mine-metrics.db").is_file())
        self.assertEqual(os.environ[REVIEWER_DB_ENV], str(mine))
        self.assertNotIn(SAVINGS_ENV, os.environ)
        self.assertFalse((self.home / ".claude").exists())

    def test_the_no_model_exits_neither_announce_nor_choose_a_database(self):
        for argv in (("--check",), ("--config", self.config(), "--show-config")):
            code, out = self.run_main(*argv)
            self.assertEqual(code, 0)
            self.assertNotIn("reviewer database", out)
            for variable in ALL_ENVS:
                self.assertNotIn(variable, os.environ)


class AFailedPreflightChoosesNothing(unittest.TestCase):
    """A stage script whose preflight fails (here LM Studio is not reachable) must not announce a database or create a folder:
    the run never starts, so a throwaway database named in its output would be one it never used."""

    SCRIPTS = (("live_reasoning.py",), ("live_verifier.py",), ("live_classifier.py",), ("live_all_reviewers.py",),
               ("live_routes.py", "--model", "some-model"))

    def test_every_stage_script_says_nothing_and_creates_nothing_when_lmstudio_is_not_reachable(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp).resolve()
            env = {k: v for k, v in os.environ.items() if k not in ALL_ENVS}
            env.update({"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://127.0.0.1:9/v1", "TMP": str(temp), "TEMP": str(temp), "TMPDIR": str(temp)})
            for script, *args in self.SCRIPTS:
                with self.subTest(script=script):
                    done = subprocess.run([sys.executable, "-I", str(REPO_ROOT / "harness" / script), *args], capture_output=True,
                                          text=True, env=env, cwd=REPO_ROOT, check=False, timeout=120)
                    output = done.stdout + done.stderr
                    self.assertIn("not reachable", output)
                    self.assertNotIn("reviewer database", output)
                    self.assertEqual(sorted(temp.iterdir()), [], "a failed preflight left a folder in the temp folder")


class TheRunnerSetsItForEveryScriptItStarts(tl.Folder):
    """`run_live_tests.main` with the scripts stubbed out: what matters is the environment each would be started with."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for variable in ALL_ENVS:
            os.environ.pop(variable, None)

    def run_runner(self) -> tuple:
        tl.write_config(self.dir, "one", ["verifier"])
        tl.write_config(self.dir, "two", ["verifier"])
        seen: list = []

        def run_script(config, runs, timeout, log, env):
            seen.append(tuple(env.get(variable) for variable in ALL_ENVS))
            return 0, "loaded before: ['x']\nloaded after:  ['x']\n"

        out = io.StringIO()
        with (mock.patch.object(rlt, "CONFIGS", self.dir), mock.patch.object(rlt, "lmstudio_models", lambda url: {"some-model"}),
              mock.patch.object(rlt, "run_script", run_script), contextlib.redirect_stdout(out)):
            rlt.main(["--skip-checks"])
        return seen, out.getvalue()

    def test_every_config_gets_the_same_throwaway_databases_under_the_temp_folder(self):
        seen, out = self.run_runner()
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], seen[1])
        reviewer, metrics, savings = (Path(p) for p in seen[0])
        self.addCleanup(shutil.rmtree, reviewer.parent, True)
        self.assertEqual(reviewer.parent.parent, Path(tempfile.gettempdir()))
        self.assertEqual((metrics.parent, savings.parent), (reviewer.parent, reviewer.parent))
        self.assertIn(f"reviewer database: {reviewer}", out)

    def test_a_callers_path_reaches_every_script_unchanged(self):
        mine = str(self.dir / "mine.db")
        os.environ[REVIEWER_DB_ENV] = mine
        seen, _ = self.run_runner()
        self.assertEqual(seen, [(mine, None, None)] * 2)

    def test_a_run_that_cannot_start_chooses_no_database(self):
        tl.write_config(self.dir, "one", ["verifier"])
        with (mock.patch.object(rlt, "CONFIGS", self.dir), mock.patch.object(rlt, "lmstudio_models", lambda url: None),
              contextlib.redirect_stdout(io.StringIO()) as out):
            code = rlt.main(["--skip-checks"])
        self.assertEqual(code, rlt.EXIT_NOT_RUN)
        self.assertNotIn("reviewer database", out.getvalue())
        for variable in ALL_ENVS:
            self.assertNotIn(variable, os.environ)


if __name__ == "__main__":
    unittest.main()
