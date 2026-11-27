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
import sqlite3
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

import openai_lib  # noqa: E402
from fake_lmstudio import FakeLmStudio  # noqa: E402
from lmstudio_model_loader_lib import ModelLoader  # noqa: E402
from reasoning_reviewer_lib import Finding, split_diff  # noqa: E402
from reviewer_files_lib import findings_document  # noqa: E402

SERVER_PATH = REPO_ROOT / "mcpservers" / "codebase_reviewer_mcp_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("codebase_reviewer_mcp_server", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reviewers(**over) -> dict:
    data = {
        "rubric": [{"name": "r1", "level": "low", "definition": "d", "score": 0.4}],
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
        self.assertEqual({t.name for t in mod.mcp._tool_manager.list_tools()}, {"list_reviewers", "review", "review_report"})

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
        # LM Studio's native API, not the OpenAI-compatible /v1: only the native one returns each response's timing.
        self.assertEqual(str(client.base_url).rstrip("/"), "http://lm:9999/api/v0")
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
        # The message quotes the path with repr(), which doubles every backslash in a
        # Windows path; compare against the same quoted form so the check holds on both.
        self.assertIn(repr(missing), out)

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
        self.assertIn("r1 [low, min average probability 0.4]", out)
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
        return mock.patch.object(self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw))

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
        # Every review writes its run record, a metrics event and maybe a savings record: point them at the temp dir.
        isolate = mock.patch.dict(os.environ, {
            "CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB": str(self.dir / "reviewer.db"),
            "CLAUDE_RUNWAY_METRICS_DB": str(self.dir / "metrics.db"),
            "CLAUDE_RUNWAY_SAVINGS_DB": str(self.dir / "savings.db"),
            "CLAUDE_RUNWAY_TRACK_SAVINGS": "",
        })
        isolate.start()
        self.addCleanup(isolate.stop)

    def _env(self, **env):
        base = {k: v for k, v in os.environ.items() if k not in ("REVIEWERS_FILE", "MAX_REVIEWERS", "MAX_REVIEW_ROUNDS")}
        base.update(env)
        return mock.patch.dict(os.environ, base, clear=True)

    def _reviewers_file(self, data=None) -> str:
        data = data or {
            "rubric": [],
            "reviewers": [
                {"name": "quoting", "model": "model-a", "prompt": "look at quoting", "active": True,
                 "type": ["reasoning"], "loadOnDemand": False, "replyFormat": "json"},
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
        out = self.mod.review("summarizer", {"repo_path": "."})
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("unknown review type 'summarizer'", out)
        self.assertIn("known types: reasoning, verifier, classifier", out)

    def test_the_classifier_type_takes_the_same_keys_as_the_other_two_and_shows_its_help_without_params(self):
        # The keys that describe the diff are the same for all three types; the classifier adds only the path of a findings file.
        shared = self.mod.REVIEW_TYPES["reasoning"].params
        self.assertEqual(self.mod.REVIEW_TYPES["classifier"].params[: len(shared)], shared)
        self.assertEqual([p.name for p in self.mod.REVIEW_TYPES["classifier"].params[len(shared):]], ["findings_file"])
        out = self.mod.review("classifier", {})
        self.assertFalse(out.startswith("Error:"))
        self.assertIn("review type 'classifier'", out)
        for key in ("repo_path", "base_ref", "head_ref", "context_lines", "chunk_chars"):
            self.assertIn(key, out)

    def test_the_verifier_type_takes_the_same_keys_as_reasoning_and_shows_its_help_without_params(self):
        # The keys that describe the diff are the same for all three types; the verifier adds only the path of a findings file.
        shared = self.mod.REVIEW_TYPES["reasoning"].params
        self.assertEqual(self.mod.REVIEW_TYPES["verifier"].params[: len(shared)], shared)
        self.assertEqual([p.name for p in self.mod.REVIEW_TYPES["verifier"].params[len(shared):]], ["findings_file"])
        out = self.mod.review("verifier", {})
        self.assertFalse(out.startswith("Error:"))
        self.assertIn("review type 'verifier'", out)
        for key in ("repo_path", "base_ref", "head_ref", "context_lines", "chunk_chars"):
            self.assertIn(key, out)

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
        with self._env(REVIEWERS_FILE=self._reviewers_file()), mock.patch.object(self.mod, "_resolve_commit", return_value="0" * 40):
            gone = self.mod.review("reasoning", {"repo_path": str(self.dir / "nope"), "base_ref": "main"})
            option = self.mod.review("reasoning", {"repo_path": str(self.dir), "base_ref": "--output=/tmp/x"})
        self.assertIn("is not a directory", gone)
        self.assertIn("base_ref '--output=/tmp/x' is not a usable git ref", option)

    def test_a_model_that_must_be_loaded_but_is_not_is_an_error(self):
        fake = FakeLmStudio(loaded=[])
        with self._env(REVIEWERS_FILE=self._reviewers_file()), mock.patch.object(
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw)
        ), mock.patch.object(self.mod, "_git_diff", return_value="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+x\n"), \
                mock.patch.object(self.mod, "_resolve_commit", return_value="0" * 40):
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
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw)
        ), mock.patch.object(self.mod, "new_client", return_value=object()), mock.patch.object(
            openai_lib, "complete", complete
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

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_the_verifier_type_grounds_and_verifies_a_real_git_range(self):
        repo = self.dir / "repo"
        repo.mkdir()

        def git(*args):
            env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                   "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)

        git("init", "-q", "-b", "main")
        (repo / "app.py").write_text("print('one')\nprint('two')\n", encoding="utf-8")
        git("add", "."); git("commit", "-q", "-m", "base")
        git("checkout", "-q", "-b", "feature")
        (repo / "app.py").write_text("print('one')\nos.system(cmd)\n", encoding="utf-8")
        git("commit", "-q", "-am", "change")
        # Work in progress on disk that is NOT in the reviewed range: the verifier must read the committed file.
        (repo / "app.py").write_text("uncommitted = True\n", encoding="utf-8")
        shown = []

        def complete(client, model, system, content):
            if "checking ONE reported defect" in system:
                shown.append(content)
                return json.dumps({"verdict": "confirmed", "failing_input": "cmd='a; b'", "wrong_result": "runs b", "reasoning": "traced"})
            return json.dumps({"findings": [
                {"file": "app.py", "line_start": 1, "line_end": 1, "class": "injection", "claim": "shell injection",
                 "failing_input": "cmd='a; b'", "quote": "os.system(cmd)"},
                {"file": "app.py", "line_start": 1, "line_end": 1, "class": "style", "claim": "made up", "failing_input": "x",
                 "quote": "this line does not exist"},
            ]})

        data = {"rubric": [], "reviewers": [
            {"name": "finder", "model": "model-a", "prompt": "look", "active": True, "type": ["reasoning"], "loadOnDemand": False,
             "replyFormat": "json"},
            {"name": "checker", "model": "model-b", "prompt": "check", "active": True, "type": ["verifier"], "loadOnDemand": False},
        ]}
        fake = FakeLmStudio(loaded=["model-a", "model-b"])
        with self._env(REVIEWERS_FILE=self._reviewers_file(data)), mock.patch.object(
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw)
        ), mock.patch.object(self.mod, "new_client", return_value=object()), mock.patch.object(
            openai_lib, "complete", complete
        ):
            out = self.mod.review("verifier", {"repo_path": str(repo), "base_ref": "main", "head_ref": "feature"})
        self.assertFalse(out.startswith("Error:"), out)
        self.assertIn("1 confirmed, 0 disputed, 0 unchecked; dropped 0 rejected and 1 ungrounded", out)
        self.assertIn("app.py:2 [injection] [finder] shell injection", out)  # the model said line 1; the quote is on line 2
        self.assertIn("checker: failing input: cmd='a; b' -> runs b", out)
        self.assertIn("ungrounded: its quote is not in app.py", out)
        self.assertEqual(len(shown), 1)
        self.assertIn("    2 > os.system(cmd)", shown[0])
        self.assertNotIn("uncommitted", shown[0])

    def test_a_verifier_model_that_is_not_loaded_is_an_error_not_a_traceback(self):
        data = {"rubric": [], "reviewers": [
            {"name": "finder", "model": "model-a", "prompt": "look", "active": True, "type": ["reasoning"], "loadOnDemand": False,
             "replyFormat": "json"},
            {"name": "checker", "model": "model-b", "prompt": "check", "active": True, "type": ["verifier"], "loadOnDemand": False},
        ]}
        fake = FakeLmStudio(loaded=["model-a"])
        with self._env(REVIEWERS_FILE=self._reviewers_file(data)), mock.patch.object(
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw)
        ), mock.patch.object(self.mod, "new_client", return_value=object()), mock.patch.object(
            self.mod, "_git_diff", return_value="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+x\n"
        ), mock.patch.object(self.mod, "_resolve_commit", return_value="0" * 40):
            out = self.mod.review("verifier", {"repo_path": str(self.dir), "base_ref": "main"})
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("'model-b'", out)

    def test_the_classifiers_requests_go_through_the_openai_compatible_url_because_the_native_one_returns_no_token_probabilities(self):
        urls = []
        with self._env(CLAUDE_RUNWAY_LMSTUDIO_URL="http://host:1234/v1"), mock.patch.object(
            self.mod, "build_client", side_effect=lambda url, *a, **k: urls.append(url) or object()
        ):
            self.mod.new_client()
            self.mod.new_probability_client()
        self.assertEqual(urls, ["http://host:1234/api/v0", "http://host:1234/v1"])

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_the_classifier_type_classifies_what_the_verifiers_left_standing_in_a_real_git_range(self):
        repo = self.dir / "repo"
        repo.mkdir()

        def git(*args):
            env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                   "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)

        git("init", "-q", "-b", "main")
        (repo / "app.py").write_text("print('one')\nprint('two')\n", encoding="utf-8")
        git("add", "."); git("commit", "-q", "-m", "base")
        git("checkout", "-q", "-b", "feature")
        (repo / "app.py").write_text("print('one')\nos.system(cmd)\n", encoding="utf-8")
        git("commit", "-q", "-am", "change")
        sorted_ = []

        def probability(client, model, system, content):
            sorted_.append((system, content))
            return {"injection": 0.8, "crash": 0.9}[content.split("Category: ")[1].split(" (")[0]]

        def complete(client, model, system, content, **options):
            if "checking ONE reported defect" in system:
                return json.dumps({"verdict": "confirmed", "failing_input": "cmd='a; b'", "wrong_result": "runs b", "reasoning": "traced"})
            return json.dumps({"findings": [
                {"file": "app.py", "line_start": 2, "line_end": 2, "class": "injection", "claim": "shell injection",
                 "failing_input": "cmd='a; b'", "quote": "os.system(cmd)"},
                {"file": "app.py", "line_start": 1, "line_end": 1, "class": "style", "claim": "made up", "failing_input": "x",
                 "quote": "this line does not exist"},
            ]})

        data = {"rubric": [
            {"name": "injection", "level": "high", "score": 0.6, "definition": "input reaches a shell"},
            {"name": "crash", "level": "moderate", "score": 0.6, "definition": "valid input raises"},
        ], "reviewers": [
            {"name": "finder", "model": "model-a", "prompt": "look", "active": True, "type": ["reasoning"], "loadOnDemand": False,
             "replyFormat": "json"},
            {"name": "checker", "model": "model-b", "prompt": "check", "active": True, "type": ["verifier"], "loadOnDemand": False},
            {"name": "sorter", "model": "model-c", "prompt": "sort", "active": True, "type": ["classifier"], "loadOnDemand": False},
        ]}
        fake = FakeLmStudio(loaded=["model-a", "model-b", "model-c"])
        with self._env(REVIEWERS_FILE=self._reviewers_file(data)), mock.patch.object(
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw)
        ), mock.patch.object(self.mod, "new_client", return_value=object()), mock.patch.object(
            openai_lib, "complete", complete
        ), mock.patch.object(openai_lib, "yes_probability", probability):
            out = self.mod.review("classifier", {"repo_path": str(repo), "base_ref": "main", "head_ref": "feature"})
        self.assertFalse(out.startswith("Error:"), out)
        self.assertIn("1 classified, 0 without a rubric level; dropped 0 unclassified, 0 rejected and 1 ungrounded", out)
        self.assertIn("app.py:2 [injection] [finder] shell injection (verifier: confirmed)", out)
        self.assertIn("injection [high], average probability 0.80 over 1 classifier(s); also fits crash (took the higher level)", out)
        self.assertEqual(len(sorted_), 2)  # the ungrounded finding never reached the classifier: one finding, once per rubric entry
        self.assertIn("Category: injection (level: high)\nDefinition: input reaches a shell", sorted_[0][1])

    def test_a_classifier_model_that_is_not_loaded_is_an_error_not_a_traceback(self):
        data = {"rubric": [{"name": "injection", "level": "high", "score": 0.6, "definition": "input reaches a shell"}], "reviewers": [
            {"name": "finder", "model": "model-a", "prompt": "look", "active": True, "type": ["reasoning"], "loadOnDemand": False,
             "replyFormat": "json"},
            {"name": "sorter", "model": "model-c", "prompt": "sort", "active": True, "type": ["classifier"], "loadOnDemand": False},
        ]}
        fake = FakeLmStudio(loaded=["model-a"])
        with self._env(REVIEWERS_FILE=self._reviewers_file(data)), mock.patch.object(
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw)
        ), mock.patch.object(self.mod, "new_client", return_value=object()), mock.patch.object(
            self.mod, "_git_diff", return_value="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+x\n"
        ), mock.patch.object(self.mod, "_resolve_commit", return_value="0" * 40):
            out = self.mod.review("classifier", {"repo_path": str(self.dir), "base_ref": "main"})
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("'model-c'", out)

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_the_verifier_reads_the_commit_the_diff_was_cut_from_even_if_head_moves_mid_run(self):
        # A run takes minutes on local models; another session can check out a branch meanwhile.
        repo = self.dir / "repo"
        repo.mkdir()

        def git(*args):
            env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                   "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)

        git("init", "-q", "-b", "main")
        (repo / "app.py").write_text("print('one')\n", encoding="utf-8")
        git("add", "."); git("commit", "-q", "-m", "base")
        git("checkout", "-q", "-b", "feature")
        (repo / "app.py").write_text("print('one')\nos.system(cmd)\n", encoding="utf-8")
        git("commit", "-q", "-am", "change")
        shown = []

        def complete(client, model, system, content):
            if "checking ONE reported defect" in system:
                shown.append(content)
                return json.dumps({"verdict": "confirmed", "failing_input": "cmd='a; b'", "wrong_result": "runs b", "reasoning": "traced"})
            git("checkout", "-q", "main")  # HEAD leaves the reviewed commit while the reasoning stage is running
            return json.dumps({"findings": [{"file": "app.py", "line_start": 2, "line_end": 2, "class": "injection",
                                             "claim": "shell injection", "failing_input": "cmd='a; b'", "quote": "os.system(cmd)"}]})

        data = {"rubric": [], "reviewers": [
            {"name": "finder", "model": "model-a", "prompt": "look", "active": True, "type": ["reasoning"], "loadOnDemand": False,
             "replyFormat": "json"},
            {"name": "checker", "model": "model-b", "prompt": "check", "active": True, "type": ["verifier"], "loadOnDemand": False},
        ]}
        fake = FakeLmStudio(loaded=["model-a", "model-b"])
        with self._env(REVIEWERS_FILE=self._reviewers_file(data)), mock.patch.object(
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw)
        ), mock.patch.object(self.mod, "new_client", return_value=object()), mock.patch.object(
            openai_lib, "complete", complete
        ):
            out = self.mod.review("verifier", {"repo_path": str(repo), "base_ref": "main", "head_ref": "HEAD"})
        self.assertFalse(out.startswith("Error:"), out)
        self.assertIn("1 confirmed, 0 disputed, 0 unchecked; dropped 0 rejected and 0 ungrounded", out)
        self.assertEqual(len(shown), 1)
        self.assertIn("os.system(cmd)", shown[0])

    def test_a_head_ref_that_is_not_a_commit_or_looks_like_an_option_is_refused_before_any_model_runs(self):
        for ref in ("--output=x", "no-such-ref", "a b"):
            with self.subTest(ref=ref):
                with self.assertRaises(self.mod.ReviewError):
                    self.mod._resolve_commit(str(self.dir), ref)

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

        # The head is pinned for every type: the diff is cut from the commit `HEAD` named, not from the moving name.
        with self._env(REVIEWERS_FILE=self._reviewers_file()), mock.patch.object(self.mod, "_git_diff", fake_diff), \
                mock.patch.object(self.mod, "_resolve_commit", return_value="0" * 40) as resolve:
            out = self.mod.review("reasoning", {"repo_path": str(self.dir), "base_ref": "origin/main"})
        resolve.assert_called_once_with(str(self.dir), "HEAD")
        self.assertEqual(captured, {"base": "origin/main", "head": "0" * 40, "context": 20})
        self.assertEqual(out, "Error: stop here")


def _feature_repo(root: Path) -> Path:
    """A repository whose `feature` branch changes `app.py` (its second line becomes `os.system(cmd)`), to review against `main`."""
    repo = root / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)

    git("init", "-q", "-b", "main")
    (repo / "app.py").write_text("print('one')\nprint('two')\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    git("checkout", "-q", "-b", "feature")
    (repo / "app.py").write_text("print('one')\nos.system(cmd)\n", encoding="utf-8")
    git("commit", "-q", "-am", "change")
    return repo


REAL = Finding("app.py", 1, 1, "injection", "shell injection", "cmd='a; b'", ("finder",), (), "os.system(cmd)")
INVENTED = Finding("app.py", 1, 1, "style", "made up", "x", ("finder",), (), "this line does not exist")


@unittest.skipUnless(shutil.which("git"), "needs git")
class ServerRunWithStubbedModels(unittest.TestCase):
    """Base of the tests that run `review` end to end over a real git repository with stubbed models. The reviewers file has a
    verifier and a classifier and NO reasoning reviewer, so a request for the reasoning stage could only come from a bug; `requests`
    lists the stage each model request was for."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.mod = _load_server()
        self.repo = _feature_repo(self.dir)
        self.db = self.dir / "reviewer.db"
        self.results = self.dir / "results"
        isolate = mock.patch.dict(os.environ, {
            "CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB": str(self.db), "CLAUDE_RUNWAY_METRICS_DB": str(self.dir / "metrics.db"),
            "CLAUDE_RUNWAY_SAVINGS_DB": str(self.dir / "savings.db"), "CLAUDE_RUNWAY_TRACK_SAVINGS": "",
        })
        isolate.start()
        self.addCleanup(isolate.stop)
        self.requests: list = []
        self.reviewers = self.reviewers_file()

    def reviewers_file(self, finder=False, name="reviewers.json") -> str:
        data = {
            "rubric": [{"name": "injection", "level": "high", "score": 0.6, "definition": "input reaches a shell"}],
            "reviewers": [
                {"name": "checker", "model": "model-b", "prompt": "check", "active": True, "type": ["verifier"], "loadOnDemand": False},
                {"name": "sorter", "model": "model-c", "prompt": "sort", "active": True, "type": ["classifier"], "loadOnDemand": False},
            ],
        }
        if finder:
            data["reviewers"].insert(0, {"name": "finder", "model": "model-a", "prompt": "look", "active": True, "type": ["reasoning"],
                                         "loadOnDemand": False, "replyFormat": "json"})
        path = self.dir / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def write_findings(self, content, name="findings.json") -> str:
        path = self.dir / name
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return str(path)

    def findings_file(self, findings, statuses=None) -> str:
        return self.write_findings(findings_document(findings, statuses))

    def complete(self, client, model, system, content):
        if "checking ONE reported defect" in system:
            self.requests.append("verifier")
            return json.dumps({"verdict": "confirmed", "failing_input": "cmd='a; b'", "wrong_result": "runs b", "reasoning": "traced"})
        self.requests.append("reasoning")
        return json.dumps({"findings": [{"file": "app.py", "line_start": 2, "line_end": 2, "class": "injection", "claim": "shell injection",
                                         "failing_input": "cmd='a; b'", "quote": "os.system(cmd)"}]})

    def probability(self, client, model, system, content):
        self.requests.append("classifier")
        return 0.8

    def env(self, **env):
        base = {k: v for k, v in os.environ.items() if k not in ("REVIEWERS_FILE", "MAX_REVIEWERS", "MAX_REVIEW_ROUNDS", "REVIEW_ID",
                                                                 "REVIEW_OUTPUT_DIR")}
        base.update(env)
        return mock.patch.dict(os.environ, base, clear=True)

    def review(self, review_type, params=None, *, reviewers=None, loaded=("model-b", "model-c"), **env) -> str:
        fake = FakeLmStudio(loaded=list(loaded))
        with self.env(REVIEWERS_FILE=reviewers or self.reviewers, **env), mock.patch.object(
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw)
        ), mock.patch.object(self.mod, "new_client", return_value=object()), mock.patch.object(
            self.mod, "new_probability_client", return_value=object()
        ), mock.patch.object(openai_lib, "complete", self.complete), mock.patch.object(openai_lib, "yes_probability", self.probability):
            return self.mod.review(review_type, {"repo_path": str(self.repo), "base_ref": "main", "head_ref": "feature", **(params or {})})

    def runs(self) -> list:
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute("SELECT * FROM runs ORDER BY started_at, rowid").fetchall()
        finally:
            conn.close()

    def head(self) -> str:
        return subprocess.run(["git", "-C", str(self.repo), "rev-parse", "feature"], check=True, capture_output=True, text=True).stdout.strip()


class OneStageFromAFindingsFile(ServerRunWithStubbedModels):
    def test_the_verifier_alone_grounds_the_findings_in_the_file_and_asks_only_its_verifiers(self):
        out = self.review("verifier", {"findings_file": self.findings_file([REAL, INVENTED])})
        self.assertFalse(out.startswith("Error:"), out)
        self.assertEqual(self.requests, ["verifier"])  # no reasoning request, and the invented finding never reached a model
        self.assertIn("1 confirmed, 0 disputed, 0 unchecked; dropped 0 rejected and 1 ungrounded", out)
        self.assertIn("app.py:2 [injection] [finder] shell injection", out)  # the file said line 1; the quote is on line 2
        self.assertIn("ungrounded: its quote is not in app.py", out)
        self.assertIn("The findings were read from a findings file: the reasoning stage did not run.", out)
        run = self.runs()[0]
        self.assertEqual((run["review_type"], run["outcome"], run["findings"], run["chunks"], run["confirmed"]), ("verifier", "finished", 2, 0, 1))

    def test_without_the_file_the_same_reviewers_file_is_still_refused_for_having_no_reasoning_reviewer(self):
        out = self.review("verifier")
        self.assertTrue(out.startswith("Error:"), out)
        self.assertIn("no active reasoning reviewer", out)
        self.assertEqual(self.requests, [])

    def test_the_classifier_alone_asks_only_its_classifiers_about_findings_that_already_stand(self):
        out = self.review("classifier", {"findings_file": self.findings_file([REAL], ["disputed"])})
        self.assertFalse(out.startswith("Error:"), out)
        self.assertEqual(self.requests, ["classifier"])  # no reasoning, no verifier request, and no grounding of the quote
        self.assertIn("1 classified, 0 without a rubric level", out)
        self.assertIn("shell injection (verifier: disputed)", out)
        self.assertIn("injection [high], average probability 0.80 over 1 classifier(s)", out)
        self.assertIn("Each finding's verification status is the one its findings file gave: no verifier ran in this review.", out)
        self.assertIn("the reasoning and verifier stages did not run", out)

    def test_the_classifier_does_not_need_the_verifiers_model_to_be_loaded(self):
        out = self.review("classifier", {"findings_file": self.findings_file([REAL], ["confirmed"])},
                          reviewers=self.reviewers_file(finder=True), loaded=("model-c",))
        self.assertFalse(out.startswith("Error:"), out)
        self.assertEqual(self.requests, ["classifier"])

    def test_chunk_chars_is_accepted_and_has_no_effect_with_a_findings_file(self):
        out = self.review("verifier", {"findings_file": self.findings_file([REAL]), "chunk_chars": 10})
        self.assertFalse(out.startswith("Error:"), out)
        self.assertEqual(self.requests, ["verifier"])

    def test_the_findings_are_not_merged_or_changed_before_the_stage_runs(self):
        twin = Finding("app.py", 2, 2, "injection", "the same place, said differently", "", ("finder",), (), "os.system(cmd)")
        out = self.review("verifier", {"findings_file": self.findings_file([REAL, twin])})
        self.assertEqual(self.requests, ["verifier", "verifier"])  # two findings on one line stay two: merging is the reasoning stage's job
        self.assertIn("2 confirmed", out)

    def test_a_findings_file_that_is_wrong_is_an_error_naming_it_with_every_problem_and_no_model_runs(self):
        path = self.write_findings({"findings": [{"file": "app.py"}, "not an object"]})
        out = self.review("verifier", {"findings_file": path})
        self.assertTrue(out.startswith("Error:"), out)
        self.assertIn(repr(path), out)
        self.assertIn("findings[0]: missing 'line_start'", out)
        self.assertIn("findings[1]: must be an object (got str)", out)
        self.assertEqual(self.requests, [])
        run = self.runs()[0]
        self.assertEqual(run["outcome"], "refused")
        self.assertIn("is not usable", run["error"])

    def test_a_findings_file_that_is_not_there_is_an_error_not_a_traceback(self):
        out = self.review("classifier", {"findings_file": str(self.dir / "absent.json")})
        self.assertTrue(out.startswith("Error:") and "is not a file" in out, out)
        self.assertEqual(self.runs()[0]["outcome"], "refused")

    def test_a_classifier_file_with_a_rejected_finding_is_refused(self):
        path = self.write_findings({"findings": [dict(findings_document([REAL])["findings"][0], verification="rejected")]})
        out = self.review("classifier", {"findings_file": path})
        self.assertIn("the verifiers rejected it", out)
        self.assertEqual(self.requests, [])

    def test_the_reasoning_type_has_no_earlier_stage_and_refuses_the_key(self):
        out = self.mod.review("reasoning", {"repo_path": ".", "base_ref": "main", "findings_file": "x.json"})
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("unknown key 'findings_file'", out)

    def test_a_blank_or_non_string_findings_file_is_refused_with_the_other_problems(self):
        out = self.mod.review("verifier", {"repo_path": ".", "base_ref": 5, "findings_file": "  "})
        self.assertIn("'findings_file' must not be blank", out)
        self.assertIn("'base_ref' must be a str (got 5)", out)
        self.assertIn("'findings_file' must be a str (got 3)", self.mod.review("classifier", {"repo_path": ".", "base_ref": "m", "findings_file": 3}))

    def test_the_tool_describes_the_new_key_for_the_two_types_that_take_it(self):
        for review_type in ("verifier", "classifier"):
            help_text = self.mod.review(review_type, {})
            self.assertIn("findings_file (str, optional)", help_text)
            self.assertIn("only the", help_text)
        self.assertNotIn("findings_file", self.mod.review("reasoning", {}))
        description = " ".join({t.name: t for t in self.mod.mcp._tool_manager.list_tools()}["review"].description.split())
        self.assertIn("findings_file", description)
        self.assertIn("PATH of a findings file, never the findings themselves", description)

    def test_a_run_from_a_findings_file_writes_no_savings_record_because_no_model_read_the_diff(self):
        import savings_ledger

        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}), mock.patch.object(
            savings_ledger, "current_session_id", return_value="live"
        ):
            self.review("verifier", {"findings_file": self.findings_file([REAL])})
        self.assertEqual(list((self.dir / "sessions").glob("*.jsonl")), [])


class ResultsFileAndReviewId(ServerRunWithStubbedModels):
    def setUp(self):
        super().setUp()
        self.supplied = self.findings_file([REAL, INVENTED])

    def test_with_an_output_folder_a_review_saves_its_complete_results_and_returns_the_same_text(self):
        plain = self.review("verifier", {"findings_file": self.supplied})
        saved = self.review("verifier", {"findings_file": self.supplied}, REVIEW_OUTPUT_DIR=str(self.results), REVIEW_ID="review-7")
        self.assertEqual(saved, plain)  # the caller's text is unchanged, and never names the file or the folder
        self.assertNotIn(str(self.results), saved)
        (path,) = list(self.results.glob("*.json"))
        run = self.runs()[-1]
        self.assertEqual(path.name, f"verifier-{run['id']}.json")  # the run's own id, so a file joins to its record
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((data["version"], data["type"], data["run_id"], data["review_id"], data["head"], data["steps"]),
                         (1, "verifier", run["id"], "review-7", self.head(), ["ground", "verify"]))
        self.assertEqual([k for k in data if k in ("reasoning", "verifier", "classifier")], ["verifier"])
        self.assertEqual([f["claim"] for f in data["verifier"]["findings"]], ["shell injection"])
        self.assertEqual(data["verifier"]["findings"][0]["quote"], "os.system(cmd)")  # what the text leaves out
        self.assertEqual([u["reason"] for u in data["verifier"]["ungrounded"]], ["its quote is not in app.py"])
        (checker,) = data["verifier"]["reviewers"]
        self.assertEqual((checker["name"], checker["requests"], checker["replies"], checker["readable"]), ("checker", 1, 1, 1))
        self.assertEqual(checker["decisions"][0]["finding"], 0)
        self.assertEqual(list(self.results.glob(".*")), [])  # no temporary file is left behind

    def test_every_review_type_saves_its_own_stages(self):
        reviewers = self.reviewers_file(finder=True)
        self.review("reasoning", reviewers=reviewers, loaded=("model-a", "model-b", "model-c"), REVIEW_OUTPUT_DIR=str(self.results))
        self.review("classifier", {"findings_file": self.findings_file([REAL], ["confirmed"])}, REVIEW_OUTPUT_DIR=str(self.results))
        sections = {p.name.split("-")[0]: [k for k in json.loads(p.read_text(encoding="utf-8")) if k in ("reasoning", "verifier", "classifier")]
                    for p in self.results.glob("*.json")}
        self.assertEqual(sections, {"reasoning": ["reasoning"], "classifier": ["classifier"]})

    def test_without_the_settings_nothing_is_written_and_the_record_is_exactly_what_it_was(self):
        self.review("verifier", {"findings_file": self.supplied})
        self.assertFalse(self.results.exists())
        run = self.runs()[0]
        self.assertEqual(json.loads(run["settings"]), {"chunk_chars": None, "context_lines": 20, "max_reviewers": 3})
        self.assertEqual(run["source"], "mcp")

    def test_the_review_id_is_stored_in_the_runs_settings_and_the_rest_of_the_record_is_unchanged(self):
        self.review("verifier", {"findings_file": self.supplied}, REVIEW_ID="  review-7  ")
        run = self.runs()[0]
        self.assertEqual(json.loads(run["settings"]), {"chunk_chars": None, "context_lines": 20, "max_reviewers": 3, "review_id": "review-7"})
        self.assertEqual((run["source"], run["outcome"]), ("mcp", "finished"))

    def test_every_run_of_one_review_carries_the_same_id(self):
        self.review("verifier", {"findings_file": self.supplied}, REVIEW_ID="review-7")
        self.review("classifier", {"findings_file": self.findings_file([REAL], ["confirmed"])}, REVIEW_ID="review-7")
        self.assertEqual({json.loads(r["settings"])["review_id"] for r in self.runs()}, {"review-7"})
        self.assertEqual(len(self.runs()), 2)

    def test_a_review_refused_by_a_bad_reviewers_file_still_carries_its_review_id(self):
        bad = self.dir / "broken.json"
        bad.write_text("{not json", encoding="utf-8")
        out = self.review("verifier", {"findings_file": self.supplied}, reviewers=str(bad), REVIEW_ID="review-7")
        self.assertTrue(out.startswith("Error:"), out)
        run = self.runs()[0]
        self.assertEqual(run["outcome"], "refused")
        self.assertEqual(json.loads(run["settings"])["review_id"], "review-7")

    def test_a_review_id_that_is_not_printable_text_is_refused_before_anything_runs(self):
        for bad in ("two\nlines", "x" * 201):
            with self.subTest(bad=bad[:10]):
                out = self.review("verifier", {"findings_file": self.supplied}, REVIEW_ID=bad)
                self.assertTrue(out.startswith("Error: Invalid codebase-reviewer settings: REVIEW_ID must be printable text"), out)
                self.assertEqual(self.requests, [])

    def test_an_output_folder_that_cannot_be_used_refuses_the_review_before_any_model_runs(self):
        blocker = self.dir / "a-file"
        blocker.write_text("x", encoding="utf-8")
        out = self.review("verifier", {"findings_file": self.supplied}, REVIEW_OUTPUT_DIR=str(blocker))
        self.assertTrue(out.startswith("Error: REVIEW_OUTPUT_DIR"), out)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.runs()[0]["outcome"], "refused")

    def test_results_that_cannot_be_written_leave_the_findings_in_the_text_with_a_note_and_the_run_finished(self):
        plain = self.review("verifier", {"findings_file": self.supplied})
        with mock.patch.object(self.mod, "write_results", side_effect=OSError(28, "No space left on device")), mock.patch("sys.stderr"):
            out = self.review("verifier", {"findings_file": self.supplied}, REVIEW_OUTPUT_DIR=str(self.results))
        self.assertEqual(out, plain + "\n\nNote: the structured results could not be saved (OSError); the findings above are complete.")
        self.assertNotIn(str(self.results), out)
        self.assertEqual([r["outcome"] for r in self.runs()], ["finished", "finished"])

    def test_a_review_that_is_refused_writes_no_results_file(self):
        out = self.review("verifier", {"findings_file": self.supplied}, loaded=("model-c",), REVIEW_OUTPUT_DIR=str(self.results))
        self.assertTrue(out.startswith("Error:") and "'model-b'" in out, out)
        self.assertEqual(list(self.results.glob("*.json")), [])

    def test_a_results_file_names_a_run_that_failed_part_way_with_its_error(self):
        # The only verifier's model fails every request: the review still finishes (incomplete), and the file says which reviewer failed.
        def failing(client, model, system, content):
            return None

        fake = FakeLmStudio(loaded=["model-b", "model-c"])
        with self.env(REVIEWERS_FILE=self.reviewers, REVIEW_OUTPUT_DIR=str(self.results)), mock.patch.object(
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw)
        ), mock.patch.object(self.mod, "new_client", return_value=object()), mock.patch.object(openai_lib, "complete", failing):
            out = self.mod.review("verifier", {"repo_path": str(self.repo), "base_ref": "main", "head_ref": "feature",
                                               "findings_file": self.supplied})
        self.assertIn("INCOMPLETE REVIEW", out)
        (path,) = list(self.results.glob("*.json"))
        section = json.loads(path.read_text(encoding="utf-8"))["verifier"]
        (checker,) = section["reviewers"]
        self.assertIn("every request failed", checker["error"])
        self.assertNotIn("requests", checker)
        self.assertTrue(any("'checker'" in e for e in section["errors"]))
        self.assertEqual(self.runs()[0]["outcome"], "incomplete")


if __name__ == "__main__":
    unittest.main()
