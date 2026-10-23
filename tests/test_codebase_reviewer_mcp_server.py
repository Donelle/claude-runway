#!/usr/bin/env python3
"""Tests for mcpservers/codebase_reviewer_mcp_server.py.

Stdlib-only (unittest) and no network: nothing here needs LM Studio, a reviewers
file outside a temp dir, or a model.

    .venv/bin/python -m unittest discover -s tests
"""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "mcpservers"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import local_compress_lib  # noqa: E402
from fake_lmstudio import FakeLmStudio  # noqa: E402
from lmstudio_model_loader_lib import ModelLoader  # noqa: E402
from reasoning_reviewer_lib import split_diff  # noqa: E402

SERVER_PATH = REPO_ROOT / "mcpservers" / "codebase_reviewer_mcp_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("codebase_reviewer_mcp_server", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reviewers(**over) -> dict:
    data = {
        "rubric": [{"name": "r1", "level": "low", "definition": "d", "score": 2}],
        "reviewers": [
            {"name": "quoting", "model": "model-a", "prompt": "p", "active": True, "type": ["reasoning"], "loadOnDemand": False},
            {
                "name": "judge",
                "model": "model-b",
                "prompt": "p",
                "active": True,
                "type": ["classifier"],
                "loadOnDemand": True,
                "contextLength": 8192,
            },
        ],
    }
    data.update(over)
    return data


class Shape(unittest.TestCase):
    def test_the_server_is_named_codebase_reviewer(self):
        self.assertEqual(_load_server().mcp.name, "codebase-reviewer")

    def test_the_registered_tool_set_is_the_documented_one(self):
        mod = _load_server()
        self.assertEqual({t.name for t in mod.mcp._tool_manager.list_tools()}, {"list_reviewers", "review"})

    def test_it_imports_no_other_server(self):
        # Standalone: installable or left out on its own.
        tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertEqual([m for m in imported if m.endswith("_mcp_server")], [])

    def test_it_does_not_load_the_qdrant_or_embedding_stack(self):
        tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        self.assertFalse(roots & {"qdrant_client", "fastembed", "trafilatura", "requests"})

    def test_it_reaches_libs_the_way_every_server_does(self):
        text = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn("os.path.dirname(os.path.dirname(os.path.abspath(__file__)))", text)

    def test_it_runs_as_a_script(self):
        self.assertIn('if __name__ == "__main__":', SERVER_PATH.read_text(encoding="utf-8"))


class Client(unittest.TestCase):
    def test_the_client_is_the_bounded_one_from_lmstudio_client(self):
        mod = _load_server()
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://lm:9999/v1"}):
            client = mod.new_client()
        self.assertEqual(client.max_retries, 0)  # no retries, like every other server's client
        self.assertEqual(str(client.base_url).rstrip("/"), "http://lm:9999/v1")
        self.assertLess(client.timeout, 600)

    def test_the_base_url_comes_from_the_environment_per_call(self):
        mod = _load_server()
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://one/v1"}):
            self.assertEqual(mod.lmstudio_base_url(), "http://one/v1")
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://two/v1"}):
            self.assertEqual(mod.lmstudio_base_url(), "http://two/v1")


class ListReviewers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.mod = _load_server()

    def _env(self, **env):
        base = {k: v for k, v in os.environ.items() if k not in ("REVIEWERS_FILE", "MAX_REVIEWERS", "MAX_REVIEW_ROUNDS")}
        base.update(env)
        return mock.patch.dict(os.environ, base, clear=True)

    def _file(self, data) -> str:
        path = self.dir / "reviewers.json"
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return str(path)

    def test_no_reviewers_file_is_an_error_and_there_is_no_default(self):
        with self._env():
            out = self.mod.list_reviewers()
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("REVIEWERS_FILE is not set", out)
        self.assertIn("no default", out)

    def test_a_missing_file_is_an_error_naming_the_path(self):
        missing = str(self.dir / "absent.json")
        with self._env(REVIEWERS_FILE=missing):
            out = self.mod.list_reviewers()
        self.assertTrue(out.startswith("Error:"))
        self.assertIn(missing, out)

    def test_an_invalid_file_lists_every_problem(self):
        data = _reviewers()
        data["rubric"][0]["level"] = "critical"
        data["reviewers"][0]["active"] = "yes"
        with self._env(REVIEWERS_FILE=self._file(data)):
            out = self.mod.list_reviewers()
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("'level' must be one of", out)
        self.assertIn("'active' is required", out)

    def test_a_bad_setting_is_an_error_even_with_a_good_file(self):
        with self._env(REVIEWERS_FILE=self._file(_reviewers()), MAX_REVIEWERS="many"):
            out = self.mod.list_reviewers()
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("MAX_REVIEWERS", out)

    def _two_reasoning(self, prompt_b="p", model_b="model-a"):
        return _reviewers(reviewers=[
            {"name": "one", "model": "model-a", "prompt": "p", "active": True, "type": ["reasoning"], "loadOnDemand": False},
            {"name": "two", "model": model_b, "prompt": prompt_b, "active": True, "type": ["reasoning"], "loadOnDemand": False},
        ])

    def test_a_single_reasoning_reviewer_gets_an_advisory_that_it_covers_everything(self):
        with self._env(REVIEWERS_FILE=self._file(_reviewers())):
            out = self.mod.list_reviewers()
        self.assertIn("Advisories:", out)
        self.assertIn("Only one reasoning reviewer is active ('quoting')", out)
        self.assertIn("broad prompt", out)

    def test_two_reviewers_with_different_prompts_get_no_advisory(self):
        with self._env(REVIEWERS_FILE=self._file(self._two_reasoning(prompt_b="a different lens"))):
            out = self.mod.list_reviewers()
        self.assertNotIn("Advisories", out)

    def test_two_reviewers_with_the_same_model_and_prompt_are_reported_as_repeating_each_other(self):
        with self._env(REVIEWERS_FILE=self._file(self._two_reasoning())):
            out = self.mod.list_reviewers()
        self.assertIn("Reviewers 'one', 'two' use the same model ('model-a') and the same prompt", out)

    def test_the_same_prompt_on_different_models_is_a_second_opinion_not_a_duplicate(self):
        with self._env(REVIEWERS_FILE=self._file(self._two_reasoning(model_b="model-b"))):
            out = self.mod.list_reviewers()
        self.assertNotIn("Advisories", out)

    def test_prompts_that_differ_only_in_whitespace_count_as_the_same(self):
        data = self._two_reasoning(prompt_b="p  \n")
        with self._env(REVIEWERS_FILE=self._file(data)):
            out = self.mod.list_reviewers()
        self.assertIn("the same prompt", out)

    def test_a_prompt_file_that_vanishes_after_validation_does_not_make_list_reviewers_raise(self):
        lens = self.dir / "lens.md"
        lens.write_text("look at injection\n", encoding="utf-8")
        data = _reviewers(reviewers=[{"name": "a", "model": "m", "promptFile": str(lens), "active": True, "type": ["reasoning"], "loadOnDemand": False}])
        real_load = self.mod.load_reviewers_file

        def load_then_lose_the_file(path):
            config = real_load(path)
            lens.unlink()  # validation passed; the file is gone before the prompt is read again
            return config

        with self._env(REVIEWERS_FILE=self._file(data)), mock.patch.object(self.mod, "load_reviewers_file", load_then_lose_the_file):
            out = self.mod.list_reviewers()
        self.assertFalse(out.startswith("Error:"), out)
        self.assertIn("Stage plan", out)

    def test_a_valid_setup_shows_settings_rubric_reviewers_and_the_stage_plan(self):
        with self._env(REVIEWERS_FILE=self._file(_reviewers()), MAX_REVIEWERS="2", MAX_REVIEW_ROUNDS="4"):
            out = self.mod.list_reviewers()
        self.assertFalse(out.startswith("Error:"))
        self.assertIn("MAX_REVIEWERS=2, MAX_REVIEW_ROUNDS=4", out)
        self.assertIn("r1 [low, min confidence 2]", out)
        self.assertIn("Stage plan", out)
        self.assertIn("1. reasoning", out)
        self.assertIn("2. classifier", out)
        self.assertIn("model model-b (loaded on demand): judge", out)
        self.assertNotIn("Model check", out)

    def test_it_loads_no_model_and_makes_no_request_by_default(self):
        with self._env(REVIEWERS_FILE=self._file(_reviewers())), mock.patch.object(
            self.mod, "ModelLoader", side_effect=AssertionError("must not contact LM Studio")
        ):
            self.mod.list_reviewers()

    def _with_fake(self, fake):
        return mock.patch.object(self.mod, "ModelLoader", lambda url: ModelLoader(url, http=fake))

    def test_check_models_reports_when_every_resident_model_is_loaded(self):
        fake = FakeLmStudio(loaded=["model-a"])
        with self._env(REVIEWERS_FILE=self._file(_reviewers())), self._with_fake(fake):
            out = self.mod.list_reviewers(check_models=True)
        self.assertIn("every model that must already be loaded is loaded", out)
        self.assertEqual(fake.posts("/load"), [])  # checking never loads anything

    def test_check_models_names_a_resident_model_that_is_not_loaded(self):
        fake = FakeLmStudio(loaded=[])
        with self._env(REVIEWERS_FILE=self._file(_reviewers())), self._with_fake(fake):
            out = self.mod.list_reviewers(check_models=True)
        self.assertIn("NOT loaded", out)
        self.assertIn("'model-a'", out)
        self.assertNotIn("'model-b'", out)  # on demand: not required to be loaded

    def test_check_models_reports_an_unreachable_lm_studio(self):
        fake = FakeLmStudio(fail_on={"/api/v0/models": ConnectionRefusedError("refused")})
        with self._env(REVIEWERS_FILE=self._file(_reviewers())), self._with_fake(fake):
            out = self.mod.list_reviewers(check_models=True)
        self.assertIn("could not ask LM Studio", out)


class ReviewTool(unittest.TestCase):
    """The one `review(type, params)` tool and its `reasoning` type."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.mod = _load_server()

    def _env(self, **env):
        base = {k: v for k, v in os.environ.items() if k not in ("REVIEWERS_FILE", "MAX_REVIEWERS", "MAX_REVIEW_ROUNDS")}
        base.update(env)
        return mock.patch.dict(os.environ, base, clear=True)

    def _reviewers_file(self, data=None) -> str:
        data = data or {
            "rubric": [],
            "reviewers": [
                {"name": "quoting", "model": "model-a", "prompt": "look at quoting", "active": True,
                 "type": ["reasoning"], "loadOnDemand": False},
            ],
        }
        path = self.dir / "reviewers.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def test_the_type_is_the_first_parameter_and_params_the_second(self):
        tool = {t.name: t for t in self.mod.mcp._tool_manager.list_tools()}["review"]
        self.assertEqual(list(tool.parameters["properties"])[:2], ["type", "params"])
        self.assertEqual(tool.parameters["required"], ["type"])

    def test_an_unknown_type_lists_the_known_ones(self):
        out = self.mod.review("verifier", {"repo_path": "."})
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("unknown review type 'verifier'", out)
        self.assertIn("known types: reasoning", out)

    def test_empty_params_returns_the_types_help_instead_of_an_error(self):
        for params in ({}, None):
            with self.subTest(params=params):
                out = self.mod.review("reasoning", params)
                self.assertFalse(out.startswith("Error:"))
                for key in ("repo_path", "base_ref", "head_ref", "context_lines", "chunk_chars"):
                    self.assertIn(key, out)
                self.assertIn("required", out)
                self.assertIn("example: review(type=\"reasoning\"", out)

    def test_every_problem_is_reported_at_once(self):
        out = self.mod.review("reasoning", {"base_ref": 5, "bogus": 1, "context_lines": True, "chunk_chars": 0, "head_ref": " "})
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("unknown key 'bogus'", out)
        self.assertIn("missing required key 'repo_path'", out)
        self.assertIn("'base_ref' must be a str (got 5)", out)
        self.assertIn("'context_lines' must be a int (got True)", out)  # a bool is not a number
        self.assertIn("'chunk_chars' must be 1 or more", out)
        self.assertIn("'head_ref' must not be blank", out)
        self.assertIn("6 problems", out)

    def test_params_that_is_not_an_object_is_an_error(self):
        self.assertIn("params must be an object", self.mod.review("reasoning", ["repo_path"]))  # type: ignore[arg-type]

    def test_validate_params_fills_defaults_and_leaves_optional_keys_out(self):
        values, problems = self.mod.validate_params(self.mod.REVIEW_TYPES["reasoning"], {"repo_path": "r", "base_ref": "main"})
        self.assertEqual(problems, [])
        self.assertEqual(values, {"repo_path": "r", "base_ref": "main", "head_ref": "HEAD", "context_lines": 20})

    def test_a_missing_reviewers_file_is_an_error_not_a_traceback(self):
        with self._env():
            out = self.mod.review("reasoning", {"repo_path": str(self.dir), "base_ref": "main"})
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("REVIEWERS_FILE is not set", out)

    def test_a_bad_repo_path_or_ref_is_an_error_naming_it(self):
        with self._env(REVIEWERS_FILE=self._reviewers_file()):
            gone = self.mod.review("reasoning", {"repo_path": str(self.dir / "nope"), "base_ref": "main"})
            option = self.mod.review("reasoning", {"repo_path": str(self.dir), "base_ref": "--output=/tmp/x"})
        self.assertIn("is not a directory", gone)
        self.assertIn("base_ref '--output=/tmp/x' is not a usable git ref", option)

    def test_a_model_that_must_be_loaded_but_is_not_is_an_error(self):
        fake = FakeLmStudio(loaded=[])
        with self._env(REVIEWERS_FILE=self._reviewers_file()), mock.patch.object(
            self.mod, "ModelLoader", lambda url: ModelLoader(url, http=fake)
        ), mock.patch.object(self.mod, "_git_diff", return_value="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+x\n"):
            out = self.mod.review("reasoning", {"repo_path": str(self.dir), "base_ref": "main"})
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("not loaded in LM Studio", out)

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_a_real_git_range_is_reviewed_and_only_findings_come_back(self):
        repo = self.dir / "repo"
        repo.mkdir()

        def git(*args):
            env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                   "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)

        git("init", "-q", "-b", "main")
        (repo / "app.py").write_text("print('one')\nprint('two')\nprint('three')\n", encoding="utf-8")
        git("add", "."); git("commit", "-q", "-m", "base")
        git("checkout", "-q", "-b", "feature")
        (repo / "app.py").write_text("print('one')\nprint('CHANGED')\nprint('three')\n", encoding="utf-8")
        git("commit", "-q", "-am", "change")
        seen = []

        def complete(client, model, system, content):
            seen.append(content)
            return json.dumps({"findings": [{"file": "app.py", "line_start": 2, "line_end": 2, "class": "output",
                                             "claim": "prints a constant", "failing_input": "any"}]})

        fake = FakeLmStudio(loaded=["model-a"])
        with self._env(REVIEWERS_FILE=self._reviewers_file()), mock.patch.object(
            self.mod, "ModelLoader", lambda url: ModelLoader(url, http=fake)
        ), mock.patch.object(self.mod, "new_client", return_value=object()), mock.patch.object(
            local_compress_lib, "complete", complete
        ):
            out = self.mod.review("reasoning", {"repo_path": str(repo), "base_ref": "main", "head_ref": "feature", "context_lines": 1})
        self.assertFalse(out.startswith("Error:"), out)
        self.assertIn("app.py:2 [quoting] prints a constant", out)
        self.assertEqual(len(seen), 1)
        # The reviewer is shown the new version with real line numbers; the removed line is not shown.
        self.assertIn("    2 + print('CHANGED')", seen[0])
        self.assertNotIn("print('two')", seen[0])
        self.assertIn("~ 1 line(s) removed here", seen[0])
        self.assertIn("    3   print('three')", seen[0])  # unchanged context: numbered, no "+" mark
        # The diff the model read is not in what comes back to the caller.
        self.assertNotIn("print('CHANGED')", out)

    def test_the_diff_uses_standard_prefixes_whatever_the_repositorys_git_config_says(self):
        # With diff.noprefix a change to `b/auth.py` is printed as `+++ b/auth.py` and read back as `auth.py`.
        for setting in (None, ("diff.noprefix", "true"), ("diff.mnemonicPrefix", "true"), ("diff.srcPrefix", "X/"), ("diff.dstPrefix", "Y/")):
            with self.subTest(setting=setting):
                repo = self.dir / f"repo-{setting[0] if setting else 'default'}"
                repo.mkdir()
                env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
                       "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

                def git(*args, repo=repo, env=env):
                    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)

                git("init", "-q", "-b", "main")
                (repo / "b").mkdir()
                (repo / "b" / "auth.py").write_text("x = 1\n", encoding="utf-8")
                git("add", ".")
                git("commit", "-q", "-m", "base")
                git("checkout", "-q", "-b", "feature")
                (repo / "b" / "auth.py").write_text("x = 2\n", encoding="utf-8")
                git("commit", "-q", "-am", "change")
                if setting:
                    git("config", *setting)
                with mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}):
                    diff = self.mod._git_diff(str(repo), "main", "feature", 3)
                self.assertIn("diff --git a/b/auth.py b/b/auth.py", diff)
                self.assertIn("+++ b/b/auth.py", diff)
                self.assertEqual([section.path for section in split_diff(diff)], ["b/auth.py"])

    def test_the_diff_runs_no_command_the_repositorys_config_names(self):
        # A textconv or external diff driver in the reviewed repository's config is a command git would run.
        repo = self.dir / "repo-drivers"
        repo.mkdir()
        marker = self.dir / "driver-ran"
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

        def git(*args):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)

        git("init", "-q", "-b", "main")
        (repo / ".gitattributes").write_text("*.txt diff=conv\n", encoding="utf-8")
        (repo / "a.txt").write_text("one\n", encoding="utf-8")
        git("add", ".")
        git("commit", "-q", "-m", "base")
        git("checkout", "-q", "-b", "feature")
        (repo / "a.txt").write_text("two\n", encoding="utf-8")
        git("commit", "-q", "-am", "change")
        driver = f"sh -c 'touch \"{marker}\"; tr a-z A-Z < \"$0\"'"
        git("config", "diff.conv.textconv", driver)
        git("config", "diff.external", f"sh -c 'touch \"{marker}\"'")
        with mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}):
            diff = self.mod._git_diff(str(repo), "main", "feature", 3)
        self.assertFalse(marker.exists())
        self.assertIn("+two", diff)
        self.assertNotIn("TWO", diff)

    def test_the_default_range_end_is_head_and_the_context_default_is_20(self):
        captured = {}

        def fake_diff(repo_path, base_ref, head_ref, context_lines):
            captured.update(base=base_ref, head=head_ref, context=context_lines)
            raise self.mod.ReviewError("stop here")

        with self._env(REVIEWERS_FILE=self._reviewers_file()), mock.patch.object(self.mod, "_git_diff", fake_diff):
            out = self.mod.review("reasoning", {"repo_path": str(self.dir), "base_ref": "origin/main"})
        self.assertEqual(captured, {"base": "origin/main", "head": "HEAD", "context": 20})
        self.assertEqual(out, "Error: stop here")


if __name__ == "__main__":
    unittest.main()
