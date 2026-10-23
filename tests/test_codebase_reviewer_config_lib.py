#!/usr/bin/env python3
"""Tests for libs/codebase_reviewer_config_lib.py: the codebase-reviewer
server's settings and the reviewers file.

Stdlib-only (unittest) and no network:

    .venv/bin/python -m unittest discover -s tests
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

from codebase_reviewer_config_lib import (  # noqa: E402
    LEVELS,
    ROLES,
    ConfigError,
    describe_config,
    load_prompt,
    load_reviewers_file,
    read_settings,
    validate_reviewers,
)


def _valid() -> dict:
    return {
        "rubric": [
            {"name": "misleading-content", "level": "low", "definition": "a misleading message.", "score": 2},
            {"name": "data-loss", "level": "high", "definition": "loses data.", "score": 4},
        ],
        "reviewers": [
            {
                "name": "quoting",
                "model": "model-a",
                "prompt": "look at quoting",
                "active": True,
                "type": ["reasoning", "verifier"],
                "loadOnDemand": False,
            },
            {
                "name": "classifier",
                "model": "model-b",
                "prompt": "classify",
                "active": True,
                "type": ["classifier"],
                "loadOnDemand": True,
                "contextLength": 32768,
            },
        ],
    }


def _problems(data, **kwargs) -> list:
    with unittest.TestCase().assertRaises(ConfigError) as ctx:
        validate_reviewers(data, **kwargs)
    return ctx.exception.problems


class FixedVocabularies(unittest.TestCase):
    def test_levels_are_low_moderate_high_in_that_order(self):
        self.assertEqual(LEVELS, ("low", "moderate", "high"))

    def test_roles_are_in_stage_order(self):
        self.assertEqual(ROLES, ("reasoning", "verifier", "classifier"))


class ValidFile(unittest.TestCase):
    def test_a_valid_config_parses(self):
        config = validate_reviewers(_valid())
        self.assertEqual([e.name for e in config.rubric], ["misleading-content", "data-loss"])
        self.assertEqual(config.rubric[1].level, "high")
        self.assertEqual(config.rubric[1].score, 4)
        quoting, classifier = config.reviewers
        self.assertEqual(quoting.roles, ("reasoning", "verifier"))
        self.assertFalse(quoting.load_on_demand)
        self.assertIsNone(quoting.context_length)
        self.assertEqual(classifier.context_length, 32768)
        self.assertTrue(classifier.load_on_demand)

    def test_roles_are_normalised_to_stage_order(self):
        data = _valid()
        data["reviewers"][0]["type"] = ["verifier", "reasoning"]
        self.assertEqual(validate_reviewers(data).reviewers[0].roles, ("reasoning", "verifier"))

    def test_active_reviewers_filters_by_role_and_active_flag(self):
        data = _valid()
        data["reviewers"].append(
            {"name": "off", "model": "model-c", "prompt": "p", "active": False, "type": ["reasoning"], "loadOnDemand": False}
        )
        config = validate_reviewers(data)
        self.assertEqual([r.name for r in config.active_reviewers()], ["quoting", "classifier"])
        self.assertEqual([r.name for r in config.active_reviewers("reasoning")], ["quoting"])
        self.assertEqual([r.name for r in config.active_reviewers("classifier")], ["classifier"])

    def test_several_entries_may_share_a_level(self):
        data = _valid()
        data["rubric"].append({"name": "other-low", "level": "low", "definition": "d", "score": 1})
        self.assertEqual(len(validate_reviewers(data).rubric), 3)

    def test_empty_lists_are_allowed(self):
        config = validate_reviewers({"rubric": [], "reviewers": []})
        self.assertEqual(config.rubric, ())
        self.assertEqual(config.reviewers, ())

    def test_describe_config_lists_rubric_and_reviewers(self):
        text = describe_config(validate_reviewers(_valid()))
        self.assertIn("misleading-content [low, min confidence 2]", text)
        self.assertIn("classifier: active, classifier, model model-b (loaded on demand, context 32768)", text)
        self.assertIn("quoting: active, reasoning/verifier, model model-a (must already be loaded)", text)


class SchemaErrors(unittest.TestCase):
    def test_top_level_must_be_an_object(self):
        self.assertTrue(any("top level" in p for p in _problems([])))

    def test_missing_required_top_level_keys_are_both_reported(self):
        problems = _problems({})
        self.assertIn("missing required key 'rubric'", problems)
        self.assertIn("missing required key 'reviewers'", problems)

    def test_unknown_top_level_key(self):
        data = _valid()
        data["reviewer"] = []
        self.assertTrue(any("unknown top-level key 'reviewer'" in p for p in _problems(data)))

    def test_unknown_level_is_an_error_naming_the_allowed_ones(self):
        data = _valid()
        data["rubric"][0]["level"] = "critical"
        problems = _problems(data)
        self.assertEqual(len(problems), 1)
        self.assertIn("'level' must be one of low, moderate, high", problems[0])
        self.assertIn("'critical'", problems[0])

    def test_duplicate_rubric_names(self):
        data = _valid()
        data["rubric"][1]["name"] = "misleading-content"
        problems = _problems(data)
        self.assertEqual(len(problems), 1)
        self.assertIn("duplicate rubric name 'misleading-content'", problems[0])

    def test_duplicate_reviewer_names(self):
        data = _valid()
        data["reviewers"][1]["name"] = "quoting"
        problems = _problems(data)
        self.assertTrue(any("duplicate reviewer name 'quoting'" in p for p in problems))

    def test_score_must_be_a_whole_number_from_1_to_5(self):
        for bad in (0, 6, -1, 2.5, "3", True, None):
            with self.subTest(score=bad):
                data = _valid()
                data["rubric"][0]["score"] = bad
                self.assertTrue(any("'score' must be a whole number from 1 to 5" in p for p in _problems(data)))
        for good in (1, 5):
            data = _valid()
            data["rubric"][0]["score"] = good
            validate_reviewers(data)

    def test_blank_or_missing_required_strings(self):
        for key in ("name", "definition"):
            for value in ("", "  ", None, 3):
                with self.subTest(rubric_key=key, value=value):
                    data = _valid()
                    data["rubric"][0][key] = value
                    self.assertTrue(any(f"'{key}' is required" in p for p in _problems(data)))
        for key in ("name", "model", "prompt"):
            for value in ("", " ", None, 3):
                with self.subTest(reviewer_key=key, value=value):
                    data = _valid()
                    data["reviewers"][0][key] = value
                    self.assertTrue(any(f"'{key}' is required" in p for p in _problems(data)))

    def test_active_and_load_on_demand_must_be_booleans(self):
        for key in ("active", "loadOnDemand"):
            for value in ("true", 1, None):
                with self.subTest(key=key, value=value):
                    data = _valid()
                    data["reviewers"][0][key] = value
                    self.assertTrue(any(f"'{key}' is required and must be true or false" in p for p in _problems(data)))
            data = _valid()
            del data["reviewers"][0][key]
            self.assertTrue(any(f"'{key}' is required" in p for p in _problems(data)))

    def test_type_must_be_a_non_empty_list_of_known_roles(self):
        for bad in ("reasoning", [], None, 3):
            with self.subTest(type=bad):
                data = _valid()
                data["reviewers"][0]["type"] = bad
                self.assertTrue(any("'type' is required and must be a non-empty list" in p for p in _problems(data)))
        data = _valid()
        data["reviewers"][0]["type"] = ["reasoning", "judge"]
        self.assertTrue(any("unknown role(s) ['judge']" in p for p in _problems(data)))
        data["reviewers"][0]["type"] = ["reasoning", "reasoning"]
        self.assertTrue(any("more than once" in p for p in _problems(data)))

    def test_context_length_must_be_a_positive_whole_number(self):
        for bad in (0, -5, 1.5, "4096", True):
            with self.subTest(contextLength=bad):
                data = _valid()
                data["reviewers"][1]["contextLength"] = bad
                self.assertTrue(any("'contextLength' must be a whole number of 1 or more" in p for p in _problems(data)))

    def test_unknown_keys_in_an_entry_are_reported(self):
        # A misspelt `loadOnDemand` would otherwise quietly mean something else.
        data = _valid()
        data["reviewers"][0]["loadondemand"] = True
        data["rubric"][0]["severity"] = 3
        problems = _problems(data)
        self.assertTrue(any("unknown key 'loadondemand'" in p for p in problems))
        self.assertTrue(any("unknown key 'severity'" in p for p in problems))

    def test_entries_must_be_objects_and_lists_must_be_lists(self):
        problems = _problems({"rubric": ["x"], "reviewers": [3]})
        self.assertTrue(any(p.startswith("rubric[0]: must be an object") for p in problems))
        self.assertTrue(any(p.startswith("reviewers[0]: must be an object") for p in problems))
        problems = _problems({"rubric": {}, "reviewers": "x"})
        self.assertTrue(any("'rubric' must be a list" in p for p in problems))
        self.assertTrue(any("'reviewers' must be a list" in p for p in problems))

    def test_every_problem_is_reported_at_once(self):
        data = _valid()
        data["rubric"][0]["level"] = "critical"
        data["rubric"][1]["name"] = "misleading-content"
        data["rubric"][1]["score"] = 9
        data["reviewers"][0]["model"] = ""
        data["reviewers"][1]["type"] = ["judge"]
        data["reviewers"][1]["loadOnDemand"] = "yes"
        with self.assertRaises(ConfigError) as ctx:
            validate_reviewers(data, path="/x/reviewers.json")
        problems = ctx.exception.problems
        self.assertGreaterEqual(len(problems), 6)
        message = str(ctx.exception)
        self.assertIn("/x/reviewers.json", message)
        self.assertIn(f"({len(problems)} problems)", message)
        for problem in problems:
            self.assertIn(problem, message)

    def test_reviewers_sharing_a_model_must_agree_on_load_on_demand(self):
        data = _valid()
        data["reviewers"][1]["model"] = "model-a"
        problems = _problems(data)
        self.assertEqual(len(problems), 1)
        self.assertIn("share model 'model-a' but disagree on 'loadOnDemand'", problems[0])

    def test_reviewers_sharing_an_on_demand_model_must_agree_on_context_length(self):
        data = _valid()
        second = copy.deepcopy(data["reviewers"][1])
        second["name"] = "second"
        second["contextLength"] = 8192
        data["reviewers"].append(second)
        problems = _problems(data)
        self.assertEqual(len(problems), 1)
        self.assertIn("disagree on 'contextLength'", problems[0])

    def test_context_length_is_ignored_for_a_model_that_is_not_loaded_on_demand(self):
        data = _valid()
        data["reviewers"][0]["contextLength"] = 4096
        second = copy.deepcopy(data["reviewers"][0])
        second["name"] = "second"
        second["contextLength"] = 8192
        data["reviewers"].append(second)
        validate_reviewers(data)

    def test_an_active_classifier_needs_a_rubric(self):
        data = _valid()
        data["rubric"] = []
        self.assertTrue(any("active classifier" in p for p in _problems(data)))
        data["reviewers"][1]["active"] = False
        validate_reviewers(data)


class DefinitionFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_a_readable_relative_definition_file_resolves_against_base_dir(self):
        (self.dir / "full.md").write_text("a long definition", encoding="utf-8")
        data = _valid()
        data["rubric"][0]["definitionFile"] = "full.md"
        config = validate_reviewers(data, base_dir=self.dir)
        self.assertEqual(config.rubric[0].definition_file, str(self.dir / "full.md"))
        self.assertEqual(config.rubric[1].definition_file, "")

    def test_a_missing_definition_file_is_an_error_naming_it(self):
        data = _valid()
        data["rubric"][0]["definitionFile"] = "nope.md"
        problems = _problems(data, base_dir=self.dir)
        self.assertEqual(len(problems), 1)
        self.assertIn("'definitionFile'", problems[0])
        self.assertIn("nope.md", problems[0])

    def test_a_directory_is_not_a_definition_file(self):
        data = _valid()
        data["rubric"][0]["definitionFile"] = str(self.dir)
        self.assertTrue(any("cannot be read" in p for p in _problems(data, base_dir=self.dir)))

    def test_definition_file_must_be_a_non_empty_string(self):
        data = _valid()
        data["rubric"][0]["definitionFile"] = ""
        self.assertTrue(any("'definitionFile' must be a non-empty path" in p for p in _problems(data)))


class LoadFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _write(self, content, name="reviewers.json") -> Path:
        path = self.dir / name
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return path

    def test_a_valid_file_loads_and_records_its_path(self):
        path = self._write(_valid())
        config = load_reviewers_file(str(path))
        self.assertEqual(config.path, str(path))
        self.assertEqual(len(config.reviewers), 2)

    def test_the_path_is_required_and_has_no_default(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError) as ctx:
                    load_reviewers_file(value)
                self.assertIn("REVIEWERS_FILE is not set", str(ctx.exception))
                self.assertIn("no default", str(ctx.exception))

    def test_a_missing_file_names_the_path(self):
        missing = self.dir / "absent.json"
        with self.assertRaises(ConfigError) as ctx:
            load_reviewers_file(str(missing))
        self.assertIn(str(missing), str(ctx.exception))
        self.assertIn("does not exist", str(ctx.exception))

    def test_a_directory_is_not_a_file(self):
        with self.assertRaises(ConfigError) as ctx:
            load_reviewers_file(str(self.dir))
        self.assertIn(str(self.dir), str(ctx.exception))
        self.assertIn("not a file", str(ctx.exception))

    def test_an_unreadable_file_names_the_path_and_the_cause(self):
        path = self._write(_valid())
        path.chmod(0)
        self.addCleanup(path.chmod, 0o600)
        if os.access(path, os.R_OK):
            self.skipTest("running as a user that can read a mode-000 file")
        with self.assertRaises(ConfigError) as ctx:
            load_reviewers_file(str(path))
        self.assertIn(str(path), str(ctx.exception))
        self.assertIn("cannot be read", str(ctx.exception))

    def test_a_file_that_is_not_utf8_text_is_unreadable(self):
        path = self.dir / "binary.json"
        path.write_bytes(b"\xff\xfe\x00\x80")
        with self.assertRaises(ConfigError) as ctx:
            load_reviewers_file(str(path))
        self.assertIn("cannot be read", str(ctx.exception))

    def test_invalid_json_names_the_path_and_the_parse_error(self):
        path = self._write("{not json")
        with self.assertRaises(ConfigError) as ctx:
            load_reviewers_file(str(path))
        self.assertIn(str(path), str(ctx.exception))
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_content_problems_are_all_reported_with_the_path(self):
        data = _valid()
        data["rubric"][0]["level"] = "bad"
        data["reviewers"][0]["active"] = "yes"
        path = self._write(data)
        with self.assertRaises(ConfigError) as ctx:
            load_reviewers_file(str(path))
        self.assertEqual(len(ctx.exception.problems), 2)
        self.assertIn(str(path), str(ctx.exception))

    def test_a_relative_path_resolves_against_base_dir(self):
        self._write(_valid())
        config = load_reviewers_file("reviewers.json", base_dir=self.dir)
        self.assertEqual(config.path, str(self.dir / "reviewers.json"))

    def test_definition_file_is_relative_to_the_reviewers_file_not_the_cwd(self):
        sub = self.dir / "sub"
        sub.mkdir()
        (sub / "full.md").write_text("text", encoding="utf-8")
        data = _valid()
        data["rubric"][0]["definitionFile"] = "full.md"
        (sub / "reviewers.json").write_text(json.dumps(data), encoding="utf-8")
        config = load_reviewers_file(str(sub / "reviewers.json"), base_dir=self.dir)
        self.assertEqual(config.rubric[0].definition_file, str(sub / "full.md"))


class Settings(unittest.TestCase):
    def test_defaults_when_unset_or_blank(self):
        for env in ({}, {"MAX_REVIEWERS": "", "MAX_REVIEW_ROUNDS": "  "}):
            with self.subTest(env=env):
                settings = read_settings(env)
                self.assertEqual(settings.max_reviewers, 3)
                self.assertEqual(settings.max_review_rounds, 3)
                self.assertEqual(settings.reviewers_file, "")

    def test_values_are_read(self):
        settings = read_settings(
            {
                "REVIEWERS_FILE": "/r.json",
                "MAX_REVIEWERS": "1",
                "MAX_REVIEW_ROUNDS": " 7 ",
                "QDRANT_URL": "http://q:6333",
                "QDRANT_API_KEY": "k",
            }
        )
        self.assertEqual(settings.reviewers_file, "/r.json")
        self.assertEqual(settings.max_reviewers, 1)
        self.assertEqual(settings.max_review_rounds, 7)
        self.assertEqual(settings.qdrant_url, "http://q:6333")
        self.assertEqual(settings.qdrant_api_key, "k")

    def test_a_bad_value_is_an_error_not_a_silent_default(self):
        for bad in ("three", "0", "-2", "1.5"):
            with self.subTest(value=bad):
                with self.assertRaises(ConfigError) as ctx:
                    read_settings({"MAX_REVIEWERS": bad})
                self.assertIn("MAX_REVIEWERS", str(ctx.exception))

    def test_every_bad_setting_is_reported_together(self):
        with self.assertRaises(ConfigError) as ctx:
            read_settings({"MAX_REVIEWERS": "x", "MAX_REVIEW_ROUNDS": "y"})
        self.assertEqual(len(ctx.exception.problems), 2)

    def test_not_requiring_the_reviewers_file_here(self):
        # Whether it is set is checked where the file is loaded, so a caller that
        # wants only the numbers (or doctor reporting a blank value) can read them.
        self.assertEqual(read_settings({}).reviewers_file, "")


class PromptSources(unittest.TestCase):
    """`prompt`, `promptFile`, or both: one of the two is required."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        (self.dir / "lens.md").write_text("The lens from a file.\n", encoding="utf-8")

    def _config(self, **change):
        data = _valid()
        reviewer = data["reviewers"][0]
        for key, value in change.items():
            if value is _ABSENT:
                reviewer.pop(key, None)
            else:
                reviewer[key] = value
        return validate_reviewers(data, base_dir=self.dir)

    def _problems_for(self, **change):
        data = _valid()
        for key, value in change.items():
            if value is _ABSENT:
                data["reviewers"][0].pop(key, None)
            else:
                data["reviewers"][0][key] = value
        return _problems(data, base_dir=self.dir)

    def test_prompt_alone_still_works(self):
        reviewer = self._config().reviewers[0]
        self.assertEqual((reviewer.prompt, reviewer.prompt_file), ("look at quoting", ""))
        self.assertEqual(load_prompt(reviewer), "look at quoting")

    def test_prompt_file_alone_is_enough_and_resolves_against_base_dir(self):
        reviewer = self._config(prompt=_ABSENT, promptFile="lens.md").reviewers[0]
        self.assertEqual(reviewer.prompt, "")
        self.assertEqual(reviewer.prompt_file, str(self.dir / "lens.md"))
        self.assertEqual(load_prompt(reviewer), "The lens from a file.")

    def test_both_are_merged_into_one_prompt_text_first(self):
        reviewer = self._config(promptFile="lens.md").reviewers[0]
        self.assertEqual(load_prompt(reviewer), "look at quoting\n\nThe lens from a file.")

    def test_neither_is_an_error_naming_both_keys(self):
        problems = self._problems_for(prompt=_ABSENT)
        self.assertEqual(len(problems), 1)
        self.assertIn("one of 'prompt' or 'promptFile' is required", problems[0])
        self.assertIn("'quoting'", problems[0])

    def test_a_blank_prompt_is_an_error_even_when_a_prompt_file_is_given(self):
        for value in ("", "  ", None, 3):
            with self.subTest(value=value):
                problems = self._problems_for(prompt=value, promptFile="lens.md")
                self.assertTrue(any("'prompt' is required and must be a non-empty string" in p for p in problems), problems)

    def test_a_missing_prompt_file_is_an_error_naming_it_even_with_a_prompt(self):
        problems = self._problems_for(promptFile="nope.md")
        self.assertEqual(len(problems), 1)
        self.assertIn("'promptFile'", problems[0])
        self.assertIn("nope.md", problems[0])

    def test_a_directory_is_not_a_prompt_file(self):
        self.assertTrue(any("cannot be read" in p for p in self._problems_for(promptFile=str(self.dir))))

    def test_a_blank_prompt_file_is_an_error(self):
        (self.dir / "empty.md").write_text("  \n\n", encoding="utf-8")
        problems = self._problems_for(prompt=_ABSENT, promptFile="empty.md")
        self.assertTrue(any("'promptFile'" in p and "is empty" in p for p in problems), problems)

    def test_prompt_file_must_be_a_non_empty_string(self):
        for value in ("", 3, None):
            with self.subTest(value=value):
                self.assertTrue(any("'promptFile' must be a non-empty path" in p for p in self._problems_for(promptFile=value)))

    def test_prompt_file_is_an_allowed_key(self):
        self.assertFalse(any("unknown key" in p for p in self._problems_for(prompt=_ABSENT, promptFile="nope.md")))

    def test_load_prompt_reads_the_file_at_run_time(self):
        reviewer = self._config(prompt=_ABSENT, promptFile="lens.md").reviewers[0]
        (self.dir / "lens.md").write_text("edited since validation", encoding="utf-8")
        self.assertEqual(load_prompt(reviewer), "edited since validation")

    def test_load_prompt_reports_a_file_that_went_away_or_went_blank(self):
        reviewer = self._config(prompt=_ABSENT, promptFile="lens.md").reviewers[0]
        (self.dir / "lens.md").write_text("   ", encoding="utf-8")
        with self.assertRaises(ConfigError) as blank:
            load_prompt(reviewer)
        self.assertIn("'quoting'", str(blank.exception))
        (self.dir / "lens.md").unlink()
        with self.assertRaises(ConfigError) as gone:
            load_prompt(reviewer)
        self.assertIn("cannot be read", str(gone.exception))
        self.assertIn("lens.md", str(gone.exception))

    def test_describe_config_says_where_each_prompt_comes_from(self):
        text = describe_config(self._config(promptFile="lens.md"))
        self.assertIn("prompt: inline text + " + str(self.dir / "lens.md"), text)
        self.assertIn("prompt: inline text", text)


_ABSENT = object()


if __name__ == "__main__":
    unittest.main()
