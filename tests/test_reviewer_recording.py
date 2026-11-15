#!/usr/bin/env python3
"""Tests that a review writes its record: the token side channel in `openai_lib`, the model loader's
observer, the reasoning and verifier request paths, and the `codebase-reviewer` server end to end.

Stdlib-only (unittest), no network: LM Studio's REST API is `tests/fake_lmstudio.py` and the completion client is a
fake. Every database, metrics store and log folder is a temporary one.

    .venv/bin/python -m unittest discover -s tests
"""

import dataclasses
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "mcpservers"))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "harness"))

import harness_lib as h  # noqa: E402
import openai_lib  # noqa: E402
import metrics_lib  # noqa: E402
import reviewer_db_lib as db  # noqa: E402
import verifier_reviewer_lib  # noqa: E402
from classifier_reviewer_lib import classify_findings  # noqa: E402
from codebase_reviewer_config_lib import Reviewer, ReviewersConfig, RubricEntry  # noqa: E402
from fake_lmstudio import FakeLmStudio  # noqa: E402
from lmstudio_model_loader_lib import ModelLoader, reviewer_api_url  # noqa: E402
from reasoning_reviewer_lib import Finding, chunk_diff, format_review, run_reasoning_review  # noqa: E402
from verifier_reviewer_lib import verify_findings  # noqa: E402

BASE = "http://localhost:1234/v1"
SERVER_PATH = REPO_ROOT / "mcpservers" / "codebase_reviewer_mcp_server.py"

DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+x = 1\n"
FINDING_JSON = json.dumps({"findings": [{"file": "a.py", "line_start": 1, "line_end": 1, "class": "output", "claim": "constant",
                                         "failing_input": "any", "quote": "x = 1"}]})


class Yes:
    """A reply item for `FakeClient`: the model's first token was Yes with this probability and No with the rest."""

    def __init__(self, probability):
        self.probability = probability


def probability_response(probability, usage=None, stats=None):
    """A chat completion whose first token had Yes at `probability` and No at the rest, as LM Studio returns token probabilities."""
    top = [SimpleNamespace(token="Yes", logprob=math.log(probability)), SimpleNamespace(token="No", logprob=math.log(1 - probability))]
    logprobs = SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)])
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Yes"), logprobs=logprobs)], usage=usage, stats=stats)


class FakeClient:
    """An OpenAI-shaped client: each call returns the next reply (or raises it), with a `usage` block. A `Yes` item is answered
    with token probabilities, as the classifier asks for them."""

    def __init__(self, replies, usage=True, stats=None):
        self._replies = iter(replies)
        self._usage = usage
        self._stats = stats
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        item = next(self._replies)
        if isinstance(item, BaseException):
            raise item
        usage = (
            SimpleNamespace(prompt_tokens=100, completion_tokens=20, completion_tokens_details=SimpleNamespace(reasoning_tokens=5))
            if self._usage else None
        )
        if isinstance(item, Yes):
            return probability_response(item.probability, usage, self._stats)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item))], usage=usage, stats=self._stats)


class ProbabilityClient:
    """A client for the classifier stage alone: `chooser(model, content)` is the probability of yes for each request."""

    def __init__(self, chooser):
        self.chooser = chooser
        self.requests: list = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        content = kwargs["messages"][1]["content"]
        self.requests.append(content)
        probability = self.chooser(kwargs["model"], content)
        if probability is None:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Yes"), logprobs=None)], usage=None, stats=None)
        return probability_response(probability)


def entry_of(content):
    return re.search(r"^Category: (\S+) \(level", content, re.MULTILINE).group(1)


def by_claim(table):
    """A chooser: `table` maps a piece of a finding's claim to {entry: probability of yes}, 0.05 for an entry not named; a claim
    mapped to None gets no probabilities at all."""
    def chooser(model, content):
        for claim, values in table.items():
            if f"Claim: {claim}" in content:
                return None if values is None else values.get(entry_of(content), 0.05)
        raise AssertionError(content)
    return chooser


def reviewer(name="quoting", model="model-a", roles=("reasoning",), on_demand=False, reply_format="json"):
    return Reviewer(name=name, model=model, prompt="look at it", active=True, roles=tuple(roles), load_on_demand=on_demand,
                    context_length=None, reply_format=reply_format)


class IsolatedEnv(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.db_path = self.dir / "r.db"
        patcher = mock.patch.dict(os.environ, {
            "CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB": str(self.db_path),
            "CLAUDE_RUNWAY_METRICS_DB": str(self.dir / "metrics.db"),
            "CLAUDE_RUNWAY_SAVINGS_DB": str(self.dir / "savings.db"),
            "CLAUDE_RUNWAY_TRACK_SAVINGS": "",
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def rows(self, sql, *args):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(sql, args).fetchall()
        finally:
            conn.close()


class TokenSideChannel(unittest.TestCase):
    def test_a_successful_completion_leaves_its_token_counts(self):
        client = FakeClient(["hello"])
        self.assertEqual(openai_lib.complete(client, "m", "sys", "user"), "hello")
        self.assertEqual(openai_lib.take_usage(),
                         {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 5, "ttft_ms": None, "inter_token_ms": None, "tokens_per_second": None, "generation_ms": None, "stop_reason": None})
        self.assertIsNone(openai_lib.take_usage())  # cleared on read

    def test_a_response_without_usage_leaves_none(self):
        openai_lib.complete(FakeClient(["hello"], usage=False), "m", "sys", "user")
        self.assertIsNone(openai_lib.take_usage())

    def test_usage_given_as_a_dict_is_read_too(self):
        response = SimpleNamespace(usage={"prompt_tokens": 7, "completion_tokens": 3})
        self.assertEqual(openai_lib._usage_from(response),
                         {"prompt_tokens": 7, "completion_tokens": 3, "reasoning_tokens": None, "ttft_ms": None, "inter_token_ms": None, "tokens_per_second": None, "generation_ms": None, "stop_reason": None})

    # What LM Studio's native /api/v0/chat/completions returned for a real request (seed-coder-8b-instruct, 2026-10-31).
    NATIVE_STATS = {"tokens_per_second": 49.426261390425196, "time_to_first_token": 0.147636, "generation_time": 1.543655,
                    "stop_reason": "eosFound"}

    def test_the_native_apis_stats_become_milliseconds_and_an_average_time_between_tokens(self):
        openai_lib.complete(FakeClient(["ok"], stats=self.NATIVE_STATS), "m", "sys", "user")
        usage = openai_lib.take_usage()
        assert usage is not None
        self.assertEqual(usage["ttft_ms"], 148)
        self.assertEqual(usage["generation_ms"], 1544)
        self.assertEqual(usage["tokens_per_second"], 49.43)
        self.assertEqual(usage["inter_token_ms"], 20.23)  # 1000 / 49.43
        self.assertEqual(usage["stop_reason"], "eosFound")
        self.assertEqual((usage["prompt_tokens"], usage["completion_tokens"]), (100, 20))

    def test_the_openai_compatible_paths_empty_stats_leave_the_timing_empty(self):
        openai_lib.complete(FakeClient(["ok"], stats={}), "m", "sys", "user")
        usage = openai_lib.take_usage()
        assert usage is not None
        self.assertEqual([usage[k] for k in ("ttft_ms", "inter_token_ms", "tokens_per_second", "generation_ms", "stop_reason")], [None] * 5)
        self.assertEqual(usage["completion_tokens"], 20)

    def test_stats_alone_are_kept_when_a_response_has_no_usage(self):
        usage = openai_lib._usage_from(SimpleNamespace(usage=None, stats=self.NATIVE_STATS))
        assert usage is not None
        self.assertEqual((usage["ttft_ms"], usage["prompt_tokens"]), (148, None))

    def test_a_zero_or_missing_rate_gives_no_time_between_tokens_not_a_division_error(self):
        for stats in ({"tokens_per_second": 0}, {"time_to_first_token": 0.5}, {"tokens_per_second": float("nan")},
                      {"tokens_per_second": float("inf")}, {"tokens_per_second": -3}, {"tokens_per_second": "fast"},
                      {"tokens_per_second": True}):
            with self.subTest(stats=stats):
                usage = openai_lib._usage_from(SimpleNamespace(usage=None, stats=stats))
                self.assertIsNone(usage["inter_token_ms"] if usage else None)

    def test_a_stop_reason_is_a_short_string_or_nothing(self):
        long = openai_lib._usage_from(SimpleNamespace(usage=None, stats={"stop_reason": "x" * 500}))
        assert long is not None
        self.assertEqual(len(long["stop_reason"]), 60)
        self.assertIsNone(openai_lib._usage_from(SimpleNamespace(usage=None, stats={"stop_reason": 7})))

    def test_odd_usage_values_are_ignored_not_raised(self):
        self.assertIsNone(openai_lib._usage_from(SimpleNamespace(usage=SimpleNamespace(prompt_tokens="many", completion_tokens=True))))
        self.assertIsNone(openai_lib._usage_from(SimpleNamespace()))

    def test_a_failed_call_does_not_leave_the_previous_calls_counts(self):
        openai_lib.complete(FakeClient(["ok"]), "m", "s", "u")
        openai_lib.complete(FakeClient([OSError("down")]), "m", "s", "u")
        self.assertIsNone(openai_lib.take_usage())

    def test_peek_failure_reads_without_clearing_so_the_message_wording_still_gets_it(self):
        openai_lib.complete(FakeClient([openai_lib.CompletionTimeout("slow")]), "m", "s", "u")
        self.assertEqual(openai_lib.peek_failure(), ("timeout", None))
        self.assertEqual(openai_lib.peek_failure(), ("timeout", None))
        self.assertEqual(openai_lib.take_failure(), ("timeout", None))
        self.assertEqual(openai_lib.peek_failure(), (None, None))


class NativeApiUrl(unittest.TestCase):
    def test_the_reviewer_talks_to_the_native_api_not_the_openai_compatible_one(self):
        self.assertEqual(reviewer_api_url("http://localhost:1234/v1"), "http://localhost:1234/api/v0")
        self.assertEqual(reviewer_api_url("http://localhost:1234/v1/"), "http://localhost:1234/api/v0")
        self.assertEqual(reviewer_api_url("http://lm.example:9000"), "http://lm.example:9000/api/v0")

    def test_the_server_builds_its_client_on_that_url(self):
        mod = _load_server()
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://box:1234/v1"}), mock.patch.object(
            mod, "build_client", return_value="client"
        ) as build:
            self.assertEqual(mod.new_client(), "client")
        build.assert_called_once_with("http://box:1234/api/v0")


class LoaderObserver(unittest.TestCase):
    class Observer:
        def __init__(self):
            self.events = []

        def model_loaded(self, model, started, ended, ok, error, details):
            self.events.append(("loaded", model, ok, error, details))

        def model_unloaded(self, model, started, ended, ok, error):
            self.events.append(("unloaded", model, ok, error))

        def model_seen(self, details):
            self.events.append(("seen", details))

    def test_a_load_and_an_unload_are_reported_with_the_models_details(self):
        fake = FakeLmStudio(loaded=[])
        seen = self.Observer()
        loader = ModelLoader(BASE, http=fake, observer=seen)
        loader.acquire("model-a", load_on_demand=True)
        loader.release("model-a")
        self.assertEqual([e[0] for e in seen.events], ["loaded", "unloaded"])
        self.assertEqual((seen.events[0][1], seen.events[0][2]), ("model-a", True))
        self.assertEqual(seen.events[0][4]["id"], "model-a")  # what LM Studio lists for it

    def test_a_model_the_user_loaded_is_seen_not_loaded(self):
        seen = self.Observer()
        loader = ModelLoader(BASE, http=FakeLmStudio(loaded=["mine"]), observer=seen)
        loader.acquire("mine", load_on_demand=True)
        self.assertEqual([e[0] for e in seen.events], ["seen"])

    def test_a_failed_load_is_reported_and_still_raises(self):
        fake = FakeLmStudio(loaded=[], capacity=0)
        seen = self.Observer()
        loader = ModelLoader(BASE, http=fake, observer=seen)
        with self.assertRaises(Exception) as ctx:
            loader.acquire("big", load_on_demand=True)
        self.assertIn("could not load model 'big'", str(ctx.exception))
        self.assertEqual((seen.events[0][0], seen.events[0][2]), ("loaded", False))
        self.assertIn("out of memory", seen.events[0][3])

    def test_an_observer_that_raises_changes_nothing(self):
        class Broken:
            def __getattr__(self, name):
                raise RuntimeError("observer is broken")

        fake = FakeLmStudio(loaded=[])
        loader = ModelLoader(BASE, http=fake, observer=Broken())
        self.assertTrue(loader.acquire("model-a", load_on_demand=True))
        self.assertTrue(loader.release("model-a"))

    def test_without_an_observer_no_extra_request_is_made(self):
        fake = FakeLmStudio(loaded=[])
        loader = ModelLoader(BASE, http=fake)
        loader.acquire("model-a", load_on_demand=True)
        loader.release("model-a")
        gets = [c for c in fake.calls if c[0] == "GET"]
        self.assertEqual(len(gets), 1)  # the one "what is loaded" check, as before


class ReasoningAndVerifierRecording(IsolatedEnv):
    def test_a_reasoning_run_records_its_stage_requests_and_loads(self):
        config = ReviewersConfig(rubric=(), reviewers=(reviewer(on_demand=True),))
        fake = FakeLmStudio(loaded=[])
        rec = db.RunRecorder(source="test", review_type="reasoning")
        with rec:
            result = run_reasoning_review(config, ModelLoader(BASE, http=fake, observer=rec), FakeClient([FINDING_JSON]), DIFF,
                                          self.dir, 1, recorder=rec)
            rec.set_counts(chunks=result.chunks, findings=result.finding_count)
        self.assertEqual(result.finding_count, 1)
        run = self.rows("SELECT * FROM runs")[0]
        self.assertEqual((run["outcome"], run["combination"], run["findings"]), ("finished", "reasoning:model-a", 1))
        request = self.rows("SELECT * FROM requests")[0]
        self.assertEqual((request["role"], request["reviewer"], request["model"], request["outcome"]), ("reasoning", "quoting", "model-a", "answered"))
        self.assertEqual((request["prompt_tokens"], request["reply_tokens"], request["reasoning_tokens"]), (100, 20, 5))
        self.assertEqual((request["subject_kind"], request["subject_id"]), ("chunk", "1"))
        self.assertGreater(request["prompt_chars"], len(DIFF))
        self.assertEqual([r["action"] for r in self.rows("SELECT action FROM model_loads ORDER BY id")], ["load", "unload"])
        setup = self.rows("SELECT * FROM reviewer_setups")[0]
        self.assertEqual((setup["reply_format"], setup["translator"]), ("json", None))
        self.assertTrue(setup["prompt_hash"])
        self.assertEqual(len(self.rows("SELECT * FROM stages")), 1)

    def _session_log(self, rec):
        return (db.log_dir(self.db_path) / db.log_file_name(rec.session_id)).read_text(encoding="utf-8")

    def test_the_session_log_shows_the_findings_a_reasoning_reviewer_raised(self):
        config = ReviewersConfig(rubric=(), reviewers=(reviewer(),))
        rec = db.RunRecorder(source="test", review_type="reasoning", session_id="sess-r")
        with rec:
            run_reasoning_review(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])), FakeClient([FINDING_JSON]), DIFF,
                                 self.dir, 1, recorder=rec)
        text = self._session_log(rec)
        self.assertIn("reasoning | quoting (model-a) | chunk 1 | answered", text)
        self.assertIn("raised 1 finding(s)", text)
        self.assertIn("a.py:1 [output] constant", text)
        self.assertIn("failing input: any", text)
        self.assertIn("quote: x = 1", text)

    def test_a_reviewer_that_fails_before_sending_a_request_makes_the_run_incomplete(self):
        # The diff needs more requests than a reviewer may send: it fails without a request or a load, yet the review
        # returned having reviewed nothing, so it must not be recorded as a finished (and fast) review.
        diff = "".join(f"diff --git a/f{i}.py b/f{i}.py\n--- a/f{i}.py\n+++ b/f{i}.py\n@@ -1 +1 @@\n+x{i} = {'y' * 400}\n"
                       for i in range(150))
        config = ReviewersConfig(rubric=(), reviewers=(reviewer(),))
        rec = db.RunRecorder(source="test", review_type="reasoning")
        with rec:
            result = run_reasoning_review(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])), FakeClient([]), diff,
                                          self.dir, 1, chunk_chars=1000, recorder=rec)
        self.assertTrue(any("requests per reviewer" in e for e in result.errors))
        self.assertEqual(self.rows("SELECT * FROM requests"), [])
        run = self.rows("SELECT outcome, error FROM runs")[0]
        self.assertEqual((run["outcome"], run["error"]), ("incomplete", "no reviewer answered in: reasoning"))

    def test_what_a_failing_translator_printed_is_shown_to_the_caller_but_never_recorded(self):
        # The translator echoes the reply on stderr and fails: the caller's problem keeps that detail, but the
        # session log and the row's summary keep only the exit status, never the model's reply.
        script = self.dir / "translator.py"
        script.write_text("#!" + sys.executable + "\nimport sys\nsys.stderr.write('bad reply: ' + sys.stdin.read())\nsys.exit(1)\n",
                          encoding="utf-8")
        script.chmod(0o755)
        config = ReviewersConfig(rubric=(), reviewers=(dataclasses.replace(reviewer(), translator=str(script)),))
        rec = db.RunRecorder(source="test", review_type="reasoning", session_id="sess-t")
        with rec:
            result = run_reasoning_review(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])),
                                          FakeClient(["SECRET_FROM_THE_REPLY"]), DIFF, self.dir, 1, recorder=rec)
            text = format_review(result)
            rec.set_result_text(text)  # what the server does: the log keeps a copy of the result returned
        self.assertIn("SECRET_FROM_THE_REPLY", text)  # the caller still sees why the translator failed
        summary = self.rows("SELECT problem_summary FROM requests")[0]["problem_summary"]
        self.assertEqual(summary, "the translator exited with status 1")
        log = self._session_log(rec)
        self.assertNotIn("SECRET_FROM_THE_REPLY", log)
        self.assertIn(f"the translator exited with status 1: {db.WITHHELD}", log)

    def test_an_unreadable_reply_is_logged_as_a_problem_and_not_as_raising_no_finding(self):
        config = ReviewersConfig(rubric=(), reviewers=(reviewer(),))
        rec = db.RunRecorder(source="test", review_type="reasoning", session_id="sess-u")
        with rec:
            run_reasoning_review(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])), FakeClient(["not json at all"]), DIFF,
                                 self.dir, 1, recorder=rec)
        text = self._session_log(rec)
        self.assertIn("problem: reviewer 'quoting': the reply was not JSON", text)
        self.assertNotIn("raised no finding", text)

    def test_the_session_log_shows_grounding_and_each_verifiers_verdict_with_its_reasoning(self):
        config = ReviewersConfig(rubric=(), reviewers=(reviewer("v", model="model-v", roles=("verifier",)),))
        kept = Finding(file="a.py", line_start=1, line_end=1, cls="output", claim="constant", failing_input="any", quote="x = 1",
                       reviewers=("quoting",))
        dropped = Finding(file="a.py", line_start=1, line_end=1, cls="output", claim="invented", failing_input="any",
                          quote="this code is not in the file", reviewers=("quoting",))
        verdict = json.dumps({"verdict": "rejected", "failing_input": "", "wrong_result": "", "reasoning": "the constant is intended"})
        rec = db.RunRecorder(source="test", review_type="verifier", session_id="sess-v")
        with rec:
            verify_findings(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-v"])), FakeClient([verdict]),
                            [kept, dropped], lambda path: "x = 1\n", 1, recorder=rec)
        text = self._session_log(rec)
        self.assertIn("grounding | 1 of 2 finding(s) kept; 1 dropped as ungrounded", text)
        self.assertIn("grounding | dropped:", text)
        self.assertIn("a.py:1 [output] invented", text)
        self.assertIn("verifier | v (model-v) | finding 1 | answered", text)
        self.assertIn("rejected: the constant is intended", text)
        self.assertLess(text.index("grounding | 1 of 2"), text.index("verifier | v (model-v)"))

    def test_the_timing_lm_studio_reports_is_recorded_with_the_request(self):
        config = ReviewersConfig(rubric=(), reviewers=(reviewer(),))
        stats = {"tokens_per_second": 50.0, "time_to_first_token": 0.2, "generation_time": 1.5, "stop_reason": "eosFound"}
        rec = db.RunRecorder(source="test", review_type="reasoning")
        with rec:
            run_reasoning_review(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])), FakeClient([FINDING_JSON], stats=stats),
                                 DIFF, self.dir, 1, recorder=rec)
        row = self.rows("SELECT * FROM requests")[0]
        self.assertEqual((row["ttft_ms"], row["inter_token_ms"], row["tokens_per_second"], row["generation_ms"], row["stop_reason"]),
                         (200, 20.0, 50.0, 1500, "eosFound"))
        self.assertEqual((row["prompt_tokens"], row["reply_tokens"]), (100, 20))
        self.assertGreaterEqual(row["latency_ms"], 0)  # measured here, around the call

    def test_a_timeout_a_refusal_and_an_unreadable_reply_are_recorded_as_such(self):
        config = ReviewersConfig(rubric=(), reviewers=(reviewer(),))
        big = DIFF + "".join(f"+line {n} {'x' * 100}\n" for n in range(60))
        timeout = openai_lib.CompletionTimeout("slow")
        refused = type("Refused", (Exception,), {"status_code": 400})("too long")
        chunks = len(chunk_diff(big, 600))
        self.assertGreaterEqual(chunks, 4)
        client = FakeClient([timeout, refused, "this is not json at all"] + [FINDING_JSON] * (chunks - 3))
        rec = db.RunRecorder(source="test", review_type="reasoning")
        with rec:
            result = run_reasoning_review(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])), client, big, self.dir, 1,
                                          chunk_chars=600, recorder=rec)
        self.assertEqual(client.calls, chunks)
        outcomes = [r["outcome"] for r in self.rows("SELECT outcome FROM requests ORDER BY rowid")]
        self.assertEqual(sorted(outcomes), sorted(["refused", "timed_out"] + ["answered"] * (chunks - 2)))
        run = self.rows("SELECT timeouts, failures FROM runs")[0]
        self.assertEqual((run["timeouts"], run["failures"]), (1, 1))
        # The failure wording still reaches the result (peeking did not steal the reason).
        self.assertTrue(any("timeout" in e for e in result.errors), result.errors)
        self.assertGreaterEqual(sum(r["problems"] for r in self.rows("SELECT problems FROM requests")), 3)
        # Each failed or unreadable request keeps a one-line summary in its row, with no reviewer name or chunk number.
        summaries = [r["problem_summary"] for r in self.rows("SELECT problem_summary FROM requests WHERE problems > 0")]
        self.assertTrue(any("timeout" in x for x in summaries), summaries)
        self.assertTrue(any("the reply was not JSON" in x for x in summaries), summaries)
        self.assertFalse(any("quoting" in x or "chunk " in x for x in summaries), summaries)
        self.assertTrue(list(db.log_dir(self.db_path).glob("*.log")))

    def test_the_verifier_records_one_request_per_finding(self):
        config = ReviewersConfig(rubric=(), reviewers=(reviewer("v", model="model-v", roles=("verifier",)),))
        finding = Finding(file="a.py", line_start=1, line_end=1, cls="output", claim="constant", failing_input="any", quote="x = 1",
                          reviewers=["quoting"])
        verdict = json.dumps({"verdict": "confirm", "failing_input": "any", "wrong_result": "1", "reasoning": "yes"})
        rec = db.RunRecorder(source="test", review_type="verifier")
        with rec:
            review = verify_findings(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-v"]), observer=rec), FakeClient([verdict]),
                                     [finding], lambda path: "x = 1\n", 1, recorder=rec)
        self.assertEqual(len(review.verified), 1)
        request = self.rows("SELECT * FROM requests")[0]
        self.assertEqual((request["role"], request["stage"], request["subject_kind"], request["subject_id"], request["model"]),
                         ("verifier", "verifier", "finding", "1", "model-v"))
        self.assertEqual(self.rows("SELECT combination FROM runs")[0]["combination"], "verifier:model-v")

    def test_without_a_recorder_nothing_is_written(self):
        config = ReviewersConfig(rubric=(), reviewers=(reviewer(),))
        run_reasoning_review(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"])), FakeClient([FINDING_JSON]), DIFF, self.dir, 1)
        self.assertFalse(self.db_path.exists())


class HarnessRecordsOneRunPerReview(IsolatedEnv):
    """The harness runs reasoning and the verifier as two calls; a pipeline review is ONE run in the database, with
    the model combination of both stages, when the caller hands both calls the same recorder."""

    def setUp(self):
        super().setUp()
        verdict = json.dumps({"verdict": "confirm", "failing_input": "any", "wrong_result": "1", "reasoning": "yes"})
        self.fake_lmstudio = FakeLmStudio(loaded=["model-a", "model-v"])
        self.client = FakeClient([FINDING_JSON, verdict, FINDING_JSON, verdict])
        server = SimpleNamespace(lmstudio_base_url=lambda: BASE, _git_diff=lambda *args: DIFF)
        for patcher in (
            mock.patch.object(h, "load_server", return_value=server),
            mock.patch("lmstudio_client.build_client", return_value=self.client),
            mock.patch("lmstudio_model_loader_lib._urllib_http", self.fake_lmstudio),
            mock.patch("verifier_reviewer_lib.git_file_reader", return_value=lambda path: "x = 1\n"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.args = SimpleNamespace(context_lines=3, chunk_chars=None)
        self.entry = h.reviewer_entry("general", "model-a", prompt="look", prompt_file=None, on_demand=False, reply_format="json")
        self.verifier = h.reviewer_entry("v", "model-v", prompt="check", prompt_file=None, roles=("verifier",), on_demand=False)

    def test_each_call_records_its_own_run_when_given_no_recorder(self):
        recorder = h.Recorder()
        findings, review = h.run_reasoning(self.dir, "main", "HEAD", [self.entry], self.args, recorder)
        h.run_verifier(self.dir, "HEAD", [f for g in review.groups for f in g.findings], [self.verifier], self.args, recorder)
        runs = self.rows("SELECT source, review_type, outcome, combination FROM runs ORDER BY started_at")
        self.assertEqual([tuple(r) for r in runs],
                         [("harness", "reasoning", "finished", "reasoning:model-a"), ("harness", "verifier", "finished", "verifier:model-v")])

    def test_one_recorder_given_to_both_calls_makes_one_run_with_both_stages(self):
        recorder = h.Recorder()
        with db.new_run_recorder(source="harness", review_type="verifier", repo_path=str(self.dir), base_ref="main", head_ref="HEAD") as run:
            findings, review = h.run_reasoning(self.dir, "main", "HEAD", [self.entry], self.args, recorder, run_recorder=run)
            h.run_verifier(self.dir, "HEAD", [f for g in review.groups for f in g.findings], [self.verifier], self.args, recorder,
                           run_recorder=run)
            self.assertFalse(self.db_path.exists())  # neither call closed (so wrote) the run it was handed
        runs = self.rows("SELECT review_type, outcome, combination, findings, confirmed FROM runs")
        self.assertEqual([tuple(r) for r in runs], [("verifier", "finished", "reasoning:model-a|verifier:model-v", 1, 1)])
        self.assertEqual(sorted(r["role"] for r in self.rows("SELECT role FROM requests")), ["reasoning", "verifier"])
        self.assertEqual(len(self.rows("SELECT * FROM stages")), 2)

    def test_a_bad_ref_in_the_harness_is_recorded_as_refused_like_the_server_does(self):
        from reasoning_reviewer_lib import ReviewError

        def bad_diff(*args):
            raise ReviewError("git diff main...nope failed: unknown revision")

        server = SimpleNamespace(lmstudio_base_url=lambda: BASE, _git_diff=bad_diff)
        with mock.patch.object(h, "load_server", return_value=server), self.assertRaises(ReviewError):
            h.run_reasoning(self.dir, "main", "nope", [self.entry], self.args, h.Recorder())
        run = self.rows("SELECT source, outcome, error FROM runs")[0]
        self.assertEqual((run["source"], run["outcome"]), ("harness", "refused"))
        self.assertIn("unknown revision", run["error"])

    def test_a_run_handed_in_is_not_closed_by_a_failing_call(self):
        run = db.new_run_recorder(source="harness", review_type="verifier", repo_path=str(self.dir), base_ref="main", head_ref="HEAD")
        with mock.patch("reasoning_reviewer_lib.run_reasoning_review", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                h.run_reasoning(self.dir, "main", "HEAD", [self.entry], self.args, h.Recorder(), run_recorder=run)
        self.assertFalse(self.db_path.exists())  # the call that failed did not close the run it was handed
        run.close("errored", error="boom")
        self.assertEqual(self.rows("SELECT outcome FROM runs")[0]["outcome"], "errored")


def _load_server():
    spec = importlib.util.spec_from_file_location("codebase_reviewer_mcp_server_recording", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ServerRecordsEveryReview(IsolatedEnv):
    def setUp(self):
        super().setUp()
        self.mod = _load_server()

    def _reviewers_file(self, on_demand=False):
        data = {"rubric": [], "reviewers": [{"name": "quoting", "model": "model-a", "prompt": "look at quoting", "active": True,
                                             "type": ["reasoning"], "loadOnDemand": on_demand, "replyFormat": "json"}]}
        path = self.dir / "reviewers.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def _review(self, client, fake, *, on_demand=False, params=None):
        env = {"REVIEWERS_FILE": self._reviewers_file(on_demand), "MAX_REVIEWERS": "1"}
        with mock.patch.dict(os.environ, env), mock.patch.object(
            self.mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=fake, **kw)
        ), mock.patch.object(self.mod, "new_client", return_value=client), mock.patch.object(self.mod, "_git_diff", return_value=DIFF), \
                mock.patch.object(self.mod, "_resolve_commit", return_value="0" * 40):
            return self.mod.review("reasoning", {"repo_path": str(self.dir), "base_ref": "main", **(params or {})})

    def test_a_review_through_the_tool_writes_its_run_stage_request_and_loads(self):
        out = self._review(FakeClient([FINDING_JSON]), FakeLmStudio(loaded=[]), on_demand=True, params={"pr": 42})
        self.assertFalse(out.startswith("Error:"), out)
        run = self.rows("SELECT * FROM runs")[0]
        self.assertEqual((run["source"], run["review_type"], run["outcome"], run["pr"], run["findings"], run["chunks"]),
                         ("mcp", "reasoning", "finished", 42, 1, 1))
        self.assertEqual(json.loads(run["settings"])["max_reviewers"], 1)
        self.assertEqual(self.rows("SELECT outcome, prompt_tokens FROM requests")[0]["outcome"], "answered")
        self.assertEqual([r["action"] for r in self.rows("SELECT action FROM model_loads ORDER BY id")], ["load", "unload"])
        self.assertTrue(run["session_id"])

    def test_the_session_log_ends_a_review_with_the_result_the_caller_was_given(self):
        out = self._review(FakeClient([FINDING_JSON]), FakeLmStudio(loaded=["model-a"]), params={"pr": 9})
        run = self.rows("SELECT session_id FROM runs")[0]
        text = (db.log_dir(self.db_path) / db.log_file_name(run["session_id"])).read_text(encoding="utf-8")
        self.assertIn("pr 9", text)
        self.assertIn("raised 1 finding(s)", text)
        start = text.index("--- result returned to the caller ---") + len("--- result returned to the caller ---\n")
        self.assertEqual(text[start:text.index("\n--- end of result ---")], out)

    def test_the_summary_event_reaches_the_shared_metrics_store(self):
        self._review(FakeClient([FINDING_JSON]), FakeLmStudio(loaded=["model-a"]), params={"pr": 5})
        event = metrics_lib.MetricsStore().detail("codebase-reviewer", limit=5)[0]
        self.assertEqual(event["event_type"], "review")
        self.assertEqual((event["metadata"]["pr"], event["metadata"]["findings"], event["metadata"]["combination"]), (5, 1, "reasoning:model-a"))
        self.assertTrue(event["session_id"])

    def test_a_review_that_times_out_part_way_is_still_recorded(self):
        out = self._review(FakeClient([openai_lib.CompletionTimeout("slow")]), FakeLmStudio(loaded=["model-a"]))
        self.assertIn("timeout", out)
        run = self.rows("SELECT outcome, error, timeouts FROM runs")[0]
        # Its only request timed out: the review returned, but reviewed nothing, so it is not a finished review.
        self.assertEqual((run["outcome"], run["timeouts"]), ("incomplete", 1))
        self.assertIn("no reviewer answered in: reasoning", run["error"])
        self.assertEqual(self.rows("SELECT outcome FROM requests")[0]["outcome"], "timed_out")

    def test_a_review_whose_on_demand_model_never_loads_is_incomplete(self):
        out = self._review(FakeClient([]), FakeLmStudio(loaded=[], capacity=0), on_demand=True)
        self.assertIn("could not load", out)
        self.assertEqual(self.rows("SELECT outcome FROM runs")[0]["outcome"], "incomplete")

    def test_a_review_against_a_remote_lm_studio_is_filed_under_that_host(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_LMSTUDIO_URL": "http://gpu-box.local:1234/v1"}):
            self._review(FakeClient([FINDING_JSON]), FakeLmStudio(loaded=["model-a"]))
        machine = self.rows("SELECT m.host, m.chip FROM runs r JOIN machines m ON m.id = r.machine_id")[0]
        self.assertEqual((machine["host"], machine["chip"]), ("gpu-box.local", None))

    def test_a_review_refused_by_a_bad_reviewers_file_is_recorded_as_refused(self):
        bad = self.dir / "broken.json"
        bad.write_text("{not json", encoding="utf-8")
        with mock.patch.dict(os.environ, {"REVIEWERS_FILE": str(bad)}):
            out = self.mod.review("reasoning", {"repo_path": str(self.dir), "base_ref": "main"})
        self.assertTrue(out.startswith("Error:"), out)
        run = self.rows("SELECT outcome, error FROM runs")[0]
        self.assertEqual(run["outcome"], "refused")
        self.assertTrue(run["error"])

    def test_a_review_refused_before_it_starts_is_recorded_as_refused(self):
        out = self._review(FakeClient([]), FakeLmStudio(loaded=[]))  # the model must be loaded and is not
        self.assertTrue(out.startswith("Error:"))
        run = self.rows("SELECT outcome, error FROM runs")[0]
        self.assertEqual(run["outcome"], "refused")
        self.assertIn("not loaded", run["error"])

    def test_a_crash_inside_the_review_is_recorded_as_errored_and_still_raises(self):
        # The orchestrator's `reason` step is what calls the reasoning stage now.
        with mock.patch("reviewer_orchestrator_lib.run_reasoning_review", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self._review(FakeClient([]), FakeLmStudio(loaded=["model-a"]))
        self.assertEqual(self.rows("SELECT outcome FROM runs")[0]["outcome"], "errored")

    def test_a_recording_failure_changes_neither_the_result_nor_the_speed(self):
        baseline = self._review(FakeClient([FINDING_JSON]), FakeLmStudio(loaded=["model-a"]))
        started = time.monotonic()
        with mock.patch.object(db, "connect", side_effect=sqlite3.OperationalError("disk I/O error")), mock.patch.object(
            metrics_lib.MetricsStore, "record", side_effect=sqlite3.OperationalError("locked")
        ), mock.patch("sys.stderr"):
            again = self._review(FakeClient([FINDING_JSON]), FakeLmStudio(loaded=["model-a"]))
        self.assertEqual(again, baseline)
        self.assertLess(time.monotonic() - started, 5.0)

    def test_the_savings_record_is_written_when_tracking_is_on(self):
        import savings_ledger

        with mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_TRACK_SAVINGS": "1"}), mock.patch.object(
            savings_ledger, "current_session_id", return_value="live"
        ):
            out = self._review(FakeClient([FINDING_JSON]), FakeLmStudio(loaded=["model-a"]))
        files = list((self.dir / "sessions").glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        entry = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((entry["raw_chars"], entry["out_chars"]), (len(DIFF), len(out)))

    def test_pr_must_be_a_positive_whole_number(self):
        for bad in (0, "7", True):
            with self.subTest(pr=bad):
                out = self.mod.review("reasoning", {"repo_path": ".", "base_ref": "main", "pr": bad})
                self.assertTrue(out.startswith("Error:"))
                self.assertIn("'pr'", out)

    def test_review_report_answers_from_what_was_recorded(self):
        self._review(FakeClient([FINDING_JSON]), FakeLmStudio(loaded=["model-a"]))
        self.assertIn("model-a [reasoning]", self.mod.review_report("response_times"))
        self.assertIn("reasoning:model-a", self.mod.review_report())
        self.assertIn("1 of 1 review(s)", self.mod.review_report("under_target", target_minutes=10))
        self.assertIn("reasoning:model-a", self.mod.review_report(this_machine=True))

    def test_review_report_errors_come_back_as_text(self):
        self.assertTrue(self.mod.review_report().startswith("Error: no codebase-reviewer database"))
        self._review(FakeClient([FINDING_JSON]), FakeLmStudio(loaded=["model-a"]))
        self.assertIn("target_minutes", self.mod.review_report("under_target"))
        self.assertIn("unknown view", self.mod.review_report("nope"))


class ClassifierRecording(IsolatedEnv):
    """The classifier stage writes into the reviewer's own database, the file `CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB` names."""

    RUBRIC = (RubricEntry("injection", "high", "input reaches a shell", 0.6), RubricEntry("crash", "moderate", "valid input raises", 0.6),
              RubricEntry("style", "low", "naming only", 0.8))

    @staticmethod
    def finding(claim="shell injection", cls="injection", file="a.py", start=1, end=1):
        return Finding(file=file, line_start=start, line_end=end, cls=cls, claim=claim, failing_input="any", quote="x = 1",
                       reviewers=["quoting"])

    def classify(self, chooser, findings, *, statuses=None, record=True, extra_roles=()):
        config = ReviewersConfig(rubric=self.RUBRIC, reviewers=(
            reviewer("sorter", model="model-s", roles=("classifier", *extra_roles)),))
        rec = db.RunRecorder(source="test", review_type="classifier")
        with rec:
            review = classify_findings(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-s"]), observer=rec),
                                       ProbabilityClient(chooser), findings, 1, statuses=statuses,
                                       recorder=rec if record else db.NULL_RECORDER)
        return review

    def test_each_classification_is_a_row_in_the_database_the_environment_names(self):
        findings = [self.finding(), self.finding("it crashes", "crash", start=3, end=4), self.finding("a name", "style", file="b.py", start=7, end=7)]
        chooser = by_claim({"shell injection": {"injection": 0.9, "crash": 0.7}, "it crashes": {"crash": 0.2}, "a name": {"style": 0.9}})
        self.classify(chooser, findings, statuses=["confirmed", "disputed", "unchecked"])
        self.assertTrue(self.db_path.exists())  # the file CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB points at
        rows = self.rows("SELECT * FROM classifications ORDER BY finding_no")
        self.assertEqual([(r["finding_no"], r["file"], r["line_start"], r["line_end"], r["finding_class"], r["verification"], r["status"],
                           r["entry"], r["level"], r["confidence"], r["overlap"]) for r in rows],
                         [(1, "a.py", 1, 1, "injection", "confirmed", "classified", "injection", "high", 0.9, 1),
                          (2, "a.py", 3, 4, "crash", "disputed", "unclassified", None, None, None, 0),
                          (3, "b.py", 7, 7, "style", "unchecked", "classified", "style", "low", 0.9, 0)])
        self.assertEqual(json.loads(rows[0]["overlapping"]), ["crash"])
        answers = json.loads(rows[1]["answers"])
        self.assertEqual(answers[0]["reviewer"], "sorter")
        # Every entry's probability is in the answer, so any average can be recomputed.
        # Stored as they came, not rounded (the probability crosses the wire as a log), so compared to the places that matter.
        self.assertEqual([a["entry"] for a in answers[0]["probabilities"]], ["injection", "crash", "style"])
        for stored, expected in zip(answers[0]["probabilities"], (0.05, 0.2, 0.05), strict=True):
            self.assertAlmostEqual(stored["probability"], expected, places=9)
        scores = json.loads(rows[0]["scores"])
        self.assertEqual([(s["entry"], s["reached"]) for s in scores], [("injection", True), ("crash", True), ("style", False)])
        for stored, expected in zip(scores, (0.9, 0.7, 0.05), strict=True):
            self.assertAlmostEqual(stored["mean"], expected, places=9)
        self.assertEqual(self.rows("SELECT id FROM runs")[0]["id"], rows[0]["run_id"])

    def test_no_claim_text_is_stored_in_the_table(self):
        self.classify(by_claim({"SECRET-CLAIM-TEXT": {"injection": 0.9}}), [self.finding(claim="SECRET-CLAIM-TEXT")])
        columns = {row["name"] for row in self.rows("PRAGMA table_info(classifications)")}
        self.assertFalse({"claim", "quote", "failing_input"} & columns)
        raw = self.db_path.read_bytes()
        self.assertNotIn(b"SECRET-CLAIM-TEXT", raw)

    def test_each_finding_is_one_classifier_request_per_rubric_entry_in_the_classifier_stage(self):
        self.classify(by_claim({"shell injection": {"injection": 0.9}, "second": {}}), [self.finding(), self.finding("second")])
        requests = self.rows("SELECT role, stage, subject_kind, subject_id, model FROM requests ORDER BY subject_id")
        self.assertEqual([tuple(r) for r in requests],
                         [("classifier", "classifier", "finding", subject, "model-s") for subject in ("1", "1", "1", "2", "2", "2")])
        self.assertEqual(self.rows("SELECT combination FROM runs")[0]["combination"], "classifier:model-s")
        self.assertIn("classifier", [r["stage"] for r in self.rows("SELECT stage FROM stages")])

    def test_the_session_log_shows_each_classification_with_its_finding(self):
        self.classify(by_claim({"shell injection": {"injection": 0.9, "crash": 0.7}}), [self.finding()])
        session = self.rows("SELECT session_id FROM runs")[0]["session_id"]
        text = (db.log_dir(self.db_path) / db.log_file_name(session)).read_text(encoding="utf-8")
        self.assertIn("classified as injection [high], average probability 0.90 over 1 classifier(s) (overlap, also fits crash)", text)
        self.assertIn("a.py:1 [injection] shell injection", text)

    def test_the_summary_event_carries_the_counts_and_no_rubric_text(self):
        findings = [self.finding(f"claim {n}") for n in range(6)]
        chooser = by_claim({
            "claim 0": {"injection": 0.9, "crash": 0.9},  # classified high, an overlap
            "claim 1": {"crash": 0.9},                    # classified moderate
            "claim 2": {"style": 0.9},                    # classified low
            "claim 3": {"style": 0.5},                    # below its score: unclassified
            "claim 4": {},                                # fits nothing: unclassified
            "claim 5": None,                              # no probability: unchecked, counted nowhere
        })
        self.classify(chooser, findings)
        event = metrics_lib.MetricsStore().detail("codebase-reviewer", limit=5)[0]
        meta = event["metadata"]
        self.assertEqual((meta["classified_high"], meta["classified_moderate"], meta["classified_low"], meta["unclassified"],
                          meta["overlaps"]), (1, 1, 1, 2, 1))
        self.assertNotIn("disputes", meta)
        text = json.dumps(meta)
        for private in ("input reaches a shell", "valid input raises", "naming only"):
            self.assertNotIn(private, text)

    def test_two_classifiers_that_split_are_averaged_and_recorded_with_each_answer(self):
        config = ReviewersConfig(rubric=self.RUBRIC, reviewers=(reviewer("s1", model="model-s", roles=("classifier",)),
                                                              reviewer("s2", model="model-t", roles=("classifier",))))
        rec = db.RunRecorder(source="test", review_type="classifier")
        by_model = {"model-s": "injection", "model-t": "crash"}
        client = ProbabilityClient(lambda model, content: 0.9 if entry_of(content) == by_model[model] else 0.05)
        with rec:
            classify_findings(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-s", "model-t"]), observer=rec), client,
                              [self.finding()], 2, recorder=rec)
        row = self.rows("SELECT * FROM classifications")[0]
        self.assertEqual((row["status"], row["entry"]), ("unclassified", None))  # (0.9 + 0.05) / 2 = 0.475 is under both scores of 0.6
        self.assertEqual({a["reviewer"] for a in json.loads(row["answers"])}, {"s1", "s2"})
        scores = json.loads(row["scores"])
        self.assertEqual([s["entry"] for s in scores], ["injection", "crash", "style"])
        for stored, expected in zip(scores, (0.475, 0.475, 0.05), strict=True):
            self.assertAlmostEqual(stored["mean"], expected, places=9)
        meta = metrics_lib.MetricsStore().detail("codebase-reviewer", limit=5)[0]["metadata"]
        self.assertEqual(meta["unclassified"], 1)
        self.assertNotIn("disputes", meta)

    def test_an_average_over_classifiers_is_stored_as_it_is(self):
        config = ReviewersConfig(rubric=self.RUBRIC, reviewers=(reviewer("s1", model="model-s", roles=("classifier",)),
                                                              reviewer("s2", model="model-t", roles=("classifier",))))
        rec = db.RunRecorder(source="test", review_type="classifier")
        client = ProbabilityClient(lambda model, content: ({"model-s": 0.9, "model-t": 0.7}[model] if entry_of(content) == "injection" else 0.05))
        with rec:
            classify_findings(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-s", "model-t"]), observer=rec), client,
                              [self.finding()], 2, recorder=rec)
        self.assertAlmostEqual(self.rows("SELECT confidence FROM classifications")[0]["confidence"], 0.8)

    @staticmethod
    def first_token_client(token, usage=None):
        """A client whose every reply begins with `token` (so no yes or no probability can be read), with a `usage` block."""
        top = [SimpleNamespace(token=token, logprob=math.log(0.9))]
        logprobs = SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)])
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=token), logprobs=logprobs)], usage=usage, stats=None)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))

    def test_a_reply_whose_probability_could_not_be_read_still_records_its_token_counts(self):
        client = self.first_token_client("The", SimpleNamespace(prompt_tokens=123, completion_tokens=3))
        config = ReviewersConfig(rubric=self.RUBRIC, reviewers=(reviewer("sorter", model="model-s", roles=("classifier",)),))
        rec = db.RunRecorder(source="test", review_type="classifier")
        with rec:
            classify_findings(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-s"]), observer=rec), client, [self.finding()], 1,
                              probability_client=client, recorder=rec)
        rows = self.rows("SELECT prompt_tokens, reply_tokens, outcome FROM requests WHERE role = 'classifier'")
        self.assertEqual({(r["prompt_tokens"], r["reply_tokens"]) for r in rows}, {(123, 3)})
        self.assertNotIn("answered", {r["outcome"] for r in rows})

    def test_a_probability_is_stored_as_it_is_so_the_saved_answers_reproduce_the_outcome(self):
        self.classify(by_claim({"shell injection": {"injection": 0.59996}}), [self.finding()])  # a hair under its score of 0.6
        row = self.rows("SELECT * FROM classifications")[0]
        self.assertEqual(row["status"], "unclassified")
        (answer,) = json.loads(row["answers"])
        self.assertEqual(answer["probabilities"][0], {"entry": "injection", "probability": 0.59996})
        score = json.loads(row["scores"])[0]
        self.assertEqual((score["mean"], score["reached"]), (0.59996, False))

    def test_what_a_model_began_a_reply_with_reaches_the_caller_but_not_the_database_or_the_log(self):
        client = self.first_token_client("SECRET-FIRST-TOKEN")
        config = ReviewersConfig(rubric=self.RUBRIC, reviewers=(reviewer("sorter", model="model-s", roles=("classifier",)),))
        rec = db.RunRecorder(source="test", review_type="classifier")
        with rec:
            review = classify_findings(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-s"]), observer=rec), client, [self.finding()],
                                       1, probability_client=client, recorder=rec)
            rec.set_result_text("\n".join(review.errors))
        self.assertTrue(any("SECRET-FIRST-TOKEN" in e for e in review.errors))  # the caller is told what the model began with
        session = self.rows("SELECT session_id FROM runs")[0]["session_id"]
        log = (db.log_dir(self.db_path) / db.log_file_name(session)).read_text(encoding="utf-8")
        self.assertNotIn("SECRET-FIRST-TOKEN", log)
        self.assertNotIn(b"SECRET-FIRST-TOKEN", self.db_path.read_bytes())
        self.assertIn("began its reply with something other than Yes or No", log)

    def test_a_request_is_recorded_as_a_probability_reply(self):
        self.classify(by_claim({"shell injection": {"injection": 0.9}}), [self.finding()])
        self.assertEqual(self.rows("SELECT COUNT(*) AS n FROM requests WHERE role = 'classifier'")[0]["n"], len(self.RUBRIC))

    def test_a_review_with_no_classifier_stage_reports_no_classification_counts(self):
        config = ReviewersConfig(rubric=(), reviewers=(reviewer(),))
        rec = db.RunRecorder(source="test", review_type="reasoning")
        with rec:
            run_reasoning_review(config, ModelLoader(BASE, http=FakeLmStudio(loaded=["model-a"]), observer=rec), FakeClient([FINDING_JSON]),
                                 DIFF, self.dir, 1, recorder=rec)
        meta = metrics_lib.MetricsStore().detail("codebase-reviewer", limit=5)[0]["metadata"]
        self.assertFalse([k for k in meta if k.startswith("classified_") or k in ("unclassified", "overlaps")])
        self.assertEqual(self.rows("SELECT COUNT(*) AS n FROM classifications")[0]["n"], 0)

    def test_with_no_classifier_active_the_findings_are_recorded_as_unchecked_and_the_stage_is_not_counted(self):
        config = ReviewersConfig(rubric=self.RUBRIC, reviewers=(reviewer("sorter", model="model-s", roles=("classifier",)),))
        config = ReviewersConfig(rubric=config.rubric, reviewers=(Reviewer(**{**vars(config.reviewers[0]), "active": False}),))
        rec = db.RunRecorder(source="test", review_type="classifier")
        with rec:
            classify_findings(config, ModelLoader(BASE, http=FakeLmStudio(loaded=[])), FakeClient([]), [self.finding()], 1, recorder=rec)
        self.assertEqual([r["status"] for r in self.rows("SELECT status FROM classifications")], ["unchecked"])
        meta = metrics_lib.MetricsStore().detail("codebase-reviewer", limit=5)[0]["metadata"]
        self.assertNotIn("classified_high", meta)  # the stage never ran

    def test_a_database_made_before_the_table_existed_gets_it_and_keeps_its_runs(self):
        with db.RunRecorder(source="test", review_type="reasoning"):
            pass
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DROP TABLE classifications")
        conn.execute("PRAGMA user_version=5")
        conn.commit()
        conn.close()
        self.assertEqual(len(self.rows("SELECT id FROM runs")), 1)
        self.classify(by_claim({"shell injection": {"injection": 0.9}}), [self.finding()])
        self.assertEqual(len(self.rows("SELECT id FROM runs")), 2)
        self.assertEqual(len(self.rows("SELECT id FROM classifications")), 1)
        conn = sqlite3.connect(str(self.db_path))
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], db.SCHEMA_VERSION)
        finally:
            conn.close()

    def test_a_database_made_at_the_last_version_gets_the_scores_column_and_keeps_its_rows(self):
        chooser = by_claim({"shell injection": {"injection": 0.9}})
        self.classify(chooser, [self.finding()])
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("ALTER TABLE classifications DROP COLUMN scores")
        conn.execute("PRAGMA user_version=6")
        conn.commit()
        conn.close()
        self.classify(chooser, [self.finding()])
        rows = self.rows("SELECT scores FROM classifications ORDER BY id")
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["scores"])
        self.assertIn("injection", rows[1]["scores"])

    def test_a_review_through_the_tool_records_its_classifications(self):
        mod = _load_server()
        data = {"rubric": [{"name": "injection", "level": "high", "score": 0.6, "definition": "input reaches a shell"}], "reviewers": [
            {"name": "quoting", "model": "model-a", "prompt": "look", "active": True, "type": ["reasoning", "verifier", "classifier"],
             "loadOnDemand": False, "replyFormat": "json"}]}
        path = self.dir / "reviewers.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        verdict = json.dumps({"verdict": "confirmed", "failing_input": "any", "wrong_result": "1", "reasoning": "yes"})
        client = FakeClient([FINDING_JSON, verdict, Yes(0.9)])  # one rubric entry, so one classifier request
        with mock.patch.dict(os.environ, {"REVIEWERS_FILE": str(path), "MAX_REVIEWERS": "1"}), mock.patch.object(
            mod, "ModelLoader", lambda url, **kw: ModelLoader(url, http=FakeLmStudio(loaded=["model-a"]), **kw)
        ), mock.patch.object(mod, "new_client", return_value=client), mock.patch.object(
            mod, "new_probability_client", return_value=client
        ), mock.patch.object(mod, "_git_diff", return_value=DIFF), mock.patch.object(
            mod, "_resolve_commit", return_value="0" * 40
        ), mock.patch.object(verifier_reviewer_lib, "git_file_reader", lambda *a, **k: lambda p: "x = 1\n"):
            out = mod.review("classifier", {"repo_path": str(self.dir), "base_ref": "main"})
        self.assertFalse(out.startswith("Error:"), out)
        run = self.rows("SELECT review_type, outcome, findings, confirmed FROM runs")[0]
        self.assertEqual(tuple(run), ("classifier", "finished", 1, 1))
        row = self.rows("SELECT status, entry, level, confidence, verification FROM classifications")[0]
        self.assertEqual(tuple(row), ("classified", "injection", "high", 0.9, "confirmed"))


if __name__ == "__main__":
    unittest.main()
