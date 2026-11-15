#!/usr/bin/env python3
"""Tests for the reply-format plumbing in harness/harness_lib.py: `replyFormat` and `translator` in a --config
file, and the `--reply-format` / `--translator` flags.

Stdlib-only (unittest), no network, no LM Studio.

    .venv/bin/python -m unittest discover -s tests
"""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _directory in (REPO_ROOT / "harness", REPO_ROOT / "libs"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import harness_lib as h  # noqa: E402


class ReplyFormatPlumbing(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name).resolve()  # macOS temp dirs are symlinked under /private
        (self.dir / "tr.py").write_text("print('x')\n", encoding="utf-8")

    def _config(self, **change) -> str:
        entry = {"name": "r", "model": "m", "prompt": "p", "active": True, "type": ["reasoning"], "loadOnDemand": True}
        entry.update(change)
        path = self.dir / "cfg.json"
        path.write_text(json.dumps({"reviewers": [entry]}), encoding="utf-8")
        return str(path)

    def _args(self, *argv) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        h.add_model_options(parser, config=True)
        return parser.parse_args(list(argv))

    def test_a_config_entry_carries_the_format_and_translator_through(self):
        config = h.load_config(self._config(replyFormat="json", translator="tr.py"))
        entry = config.reviewers[0]
        self.assertEqual(entry["replyFormat"], "json")
        self.assertEqual(entry["translator"], str(self.dir / "tr.py"))

    def test_a_config_with_neither_gets_the_servers_default_and_no_translator(self):
        entry = h.load_config(self._config()).reviewers[0]
        self.assertEqual(entry["replyFormat"], "lines")
        self.assertNotIn("translator", entry)

    def test_a_bad_format_in_a_config_is_reported_by_the_servers_own_validator(self):
        with self.assertRaises(Exception) as caught:
            h.load_config(self._config(replyFormat="xml"))
        self.assertIn("'replyFormat' must be one of lines, json", str(caught.exception))

    def test_the_flags_become_keys_on_the_reasoning_reviewer_only(self):
        args = h.finish_args(self._args("--reply-format", "json", "--translator", str(self.dir / "tr.py")))
        reasoning = h.stage_reviewers(args, "reasoning")[0]
        self.assertEqual((reasoning["replyFormat"], reasoning["translator"]), ("json", str(self.dir / "tr.py")))
        for stage in ("verifier", "classifier"):
            other = h.stage_reviewers(args, stage)[0]
            self.assertNotIn("replyFormat", other)
            self.assertNotIn("translator", other)

    def test_without_the_flags_the_entry_leaves_the_server_default_alone(self):
        entry = h.stage_reviewers(h.finish_args(self._args()), "reasoning")[0]
        self.assertNotIn("replyFormat", entry)
        self.assertNotIn("translator", entry)

    def test_an_unknown_format_flag_is_refused(self):
        with self.assertRaises(SystemExit):
            self._args("--reply-format", "xml")

    def test_a_flag_next_to_a_config_that_names_reviewers_is_refused(self):
        for flag in (("--reply-format", "json"), ("--translator", "tr.py")):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit) as caught:
                    h.finish_args(self._args("--config", self._config(), *flag))
                self.assertIn(flag[0], str(caught.exception))

    def test_show_config_says_how_the_reply_is_read(self):
        plain = h.describe_config(h.finish_args(self._args("--config", self._config())))
        self.assertIn("reply: lines", plain)
        text = h.describe_config(h.finish_args(self._args("--config", self._config(replyFormat="json", translator="tr.py"))))
        self.assertIn("reply: json via " + str(self.dir / "tr.py"), text)


class NoConfigReviewers(unittest.TestCase):
    """`live_all_reviewers.py` builds its own reviewers when there is no --config: the flags must reach them."""

    def test_the_flags_reach_every_reasoning_entry_it_builds(self):
        import live_all_reviewers as live

        script = str(REPO_ROOT / "harness" / "prompts" / "README.md")  # any existing path: it is only forwarded here
        args = live.parse_args(["--reply-format", "json", "--translator", script])
        entries, _ = live.build_config(args)
        self.assertGreater(len(entries), 1)
        for entry in entries:
            self.assertEqual((entry["replyFormat"], entry["translator"]), ("json", script))

    def test_without_the_flags_the_servers_default_applies(self):
        import live_all_reviewers as live

        entries, _ = live.build_config(live.parse_args([]))
        for entry in entries:
            self.assertNotIn("replyFormat", entry)
            self.assertNotIn("translator", entry)


if __name__ == "__main__":
    unittest.main()
