#!/usr/bin/env python3
"""Tests for `runway_defaults.isolate_reviewer_db`: a harness run records in a throwaway reviewer database unless the caller
chose one, and never in the real one under `~/.claude`.

Stdlib-only (unittest), no network, no LM Studio: LM Studio's REST API is `tests/fake_lmstudio.py` and the model's answer is a
stub, as in `test_harness_classifier.py`, whose set-up this reuses. The home folder is a temporary one, so "the default
database" is a path in a sandbox that the test can assert was never created; the real `~/.claude` is not touched.

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import os
import shutil
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

import run_live_tests as rlt  # noqa: E402
import runway_defaults  # noqa: E402
import test_harness_classifier as tc  # noqa: E402
import test_harness_live_tests as tl  # noqa: E402
from runway_defaults import REVIEWER_DB_ENV, isolate_reviewer_db  # noqa: E402

DEFAULT_DB = Path(".claude") / "claude-runway" / "codebase-reviewer.db"


class Sandboxed(unittest.TestCase):
    """A home folder that is a temp folder, and the reviewer-database variable unset: what a harness run starts from."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name).resolve()
        self.home = self.dir / "home"
        self.home.mkdir()
        # `os.environ` is restored at cleanup, so the variable the helper sets does not leak into other tests.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(REVIEWER_DB_ENV, None)
        home = mock.patch.object(Path, "home", return_value=self.home)
        home.start()
        self.addCleanup(home.stop)

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

    def test_a_callers_path_is_kept_and_is_what_resolve_db_path_returns(self):
        import reviewer_db_lib
        mine = self.dir / "mine" / "r.db"
        os.environ[REVIEWER_DB_ENV] = str(mine)
        path, _ = self.isolate()
        self.assertEqual(path, mine)
        self.assertEqual(os.environ[REVIEWER_DB_ENV], str(mine))
        self.assertEqual(reviewer_db_lib.resolve_db_path(), mine)

    def test_an_empty_variable_counts_as_unset_as_resolve_db_path_treats_it(self):
        os.environ[REVIEWER_DB_ENV] = ""
        path, _ = self.isolate()
        self.assertEqual(os.environ[REVIEWER_DB_ENV], str(path))
        self.assertEqual(path.parent.parent, Path(tempfile.gettempdir()))

    def test_the_chosen_path_is_announced_in_both_cases(self):
        path, said = self.isolate()
        self.assertEqual(len(said), 1)
        self.assertIn(str(path), said[0])
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
        os.environ.pop(REVIEWER_DB_ENV)
        with mock.patch.object(runway_defaults.os, "getpid", return_value=os.getpid() + 1):
            second, _ = self.isolate()
        self.assertNotEqual(first.parent, second.parent)

    def test_importing_the_module_does_not_touch_the_variable(self):
        import importlib
        importlib.reload(runway_defaults)
        self.assertNotIn(REVIEWER_DB_ENV, os.environ)


class AStageScriptRecordsInTheThrowawayDatabase(tc.Isolated):
    """The ticket's check: a stage script run with a fake LM Studio and the variable unset leaves the default path alone."""

    def setUp(self):
        super().setUp()
        os.environ.pop(REVIEWER_DB_ENV)  # `Isolated` pins it to a temp file; this test is about it being unset
        self.home = self.dir / "home"
        self.home.mkdir()
        home = mock.patch.object(Path, "home", return_value=self.home)
        home.start()
        self.addCleanup(home.stop)

    def test_a_run_with_the_variable_unset_writes_only_under_the_system_temp_folder(self):
        self.lmstudio(lambda client, model, system, content: tc.perfect_probability(content))
        code, out = self.run_main("--config", self.config(), "--runs", "1")
        self.assertEqual(code, 0, out)
        throwaway = Path(os.environ[REVIEWER_DB_ENV])
        self.addCleanup(shutil.rmtree, throwaway.parent, True)
        self.assertEqual(throwaway.parent.parent, Path(tempfile.gettempdir()))
        self.assertTrue(throwaway.is_file(), "the run recorded nothing in the throwaway database")
        self.assertIn(f"reviewer database: {throwaway}", out)  # said at the start so the logs can be found
        # The real default, and the folder around it, were never created.
        self.assertFalse((self.home / DEFAULT_DB).exists())
        self.assertFalse((self.home / ".claude").exists())
        self.assertFalse((self.home / ".claude" / "claude-runway" / "codebase-reviewer-logs").exists())

    def test_a_run_with_the_variable_set_records_where_the_caller_said(self):
        mine = self.dir / "mine" / "r.db"
        os.environ[REVIEWER_DB_ENV] = str(mine)
        self.lmstudio(lambda client, model, system, content: tc.perfect_probability(content))
        code, out = self.run_main("--config", self.config(), "--runs", "1")
        self.assertEqual(code, 0, out)
        self.assertTrue(mine.is_file())
        self.assertEqual(os.environ[REVIEWER_DB_ENV], str(mine))
        self.assertFalse((self.home / ".claude").exists())

    def test_the_no_model_exits_neither_announce_nor_choose_a_database(self):
        for argv in (("--check",), ("--config", self.config(), "--show-config")):
            code, out = self.run_main(*argv)
            self.assertEqual(code, 0)
            self.assertNotIn("reviewer database", out)
            self.assertNotIn(REVIEWER_DB_ENV, os.environ)


class TheRunnerSetsItForEveryScriptItStarts(tl.Folder):
    """`run_live_tests.main` with the scripts stubbed out: what matters is the environment each would be started with."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(REVIEWER_DB_ENV, None)

    def run_runner(self) -> tuple:
        tl.write_config(self.dir, "one", ["verifier"])
        tl.write_config(self.dir, "two", ["verifier"])
        seen: list = []

        def run_script(config, runs, timeout, log, env):
            seen.append(env.get(REVIEWER_DB_ENV))
            return 0, "loaded before: ['x']\nloaded after:  ['x']\n"

        out = io.StringIO()
        with (mock.patch.object(rlt, "CONFIGS", self.dir), mock.patch.object(rlt, "lmstudio_models", lambda url: {"some-model"}),
              mock.patch.object(rlt, "run_script", run_script), contextlib.redirect_stdout(out)):
            rlt.main(["--skip-checks"])
        return seen, out.getvalue()

    def test_every_config_gets_the_same_throwaway_database_under_the_temp_folder(self):
        seen, out = self.run_runner()
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(Path(seen[0]).parent.parent, Path(tempfile.gettempdir()))
        self.addCleanup(shutil.rmtree, Path(seen[0]).parent, True)
        self.assertIn(f"reviewer database: {seen[0]}", out)

    def test_a_callers_path_reaches_every_script_unchanged(self):
        mine = str(self.dir / "mine.db")
        os.environ[REVIEWER_DB_ENV] = mine
        seen, _ = self.run_runner()
        self.assertEqual(seen, [mine, mine])

    def test_a_run_that_cannot_start_chooses_no_database(self):
        tl.write_config(self.dir, "one", ["verifier"])
        with (mock.patch.object(rlt, "CONFIGS", self.dir), mock.patch.object(rlt, "lmstudio_models", lambda url: None),
              contextlib.redirect_stdout(io.StringIO()) as out):
            code = rlt.main(["--skip-checks"])
        self.assertEqual(code, rlt.EXIT_NOT_RUN)
        self.assertNotIn("reviewer database", out.getvalue())
        self.assertNotIn(REVIEWER_DB_ENV, os.environ)


if __name__ == "__main__":
    unittest.main()
