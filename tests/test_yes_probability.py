#!/usr/bin/env python3
"""Tests for `openai_lib.yes_probability`: the model's probability of "yes", read from token probabilities.

Stdlib-only (unittest), no network: the client is a fake with an OpenAI-shaped `chat.completions.create`.

    .venv/bin/python -m unittest discover -s tests
"""

import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

import openai_lib as lib  # noqa: E402
from openai_lib import MIN_ANSWER_MASS, CompletionTimeout, yes_probability  # noqa: E402


def candidates(**probabilities):
    """top_logprobs entries from {token: probability}; a token's key uses `_` for a leading space."""
    return [SimpleNamespace(token=token.replace("_", " "), logprob=math.log(p)) for token, p in probabilities.items()]


def response(top):
    """A chat completion whose first token had the candidates `top` (None: no `logprobs` at all)."""
    logprobs = None if top is None else SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)])
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Yes"), logprobs=logprobs)], usage=None)


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.requests: list = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class StatusError(Exception):
    status_code = 400


class YesProbability(unittest.TestCase):
    def ask(self, result):
        client = FakeClient(result)
        return yes_probability(client, "m", "system text", "user text"), client

    def test_it_is_the_share_of_yes_among_yes_and_no(self):
        value, _ = self.ask(response(candidates(Yes=0.6, No=0.2)))
        self.assertAlmostEqual(value, 0.75)

    def test_case_and_a_leading_space_are_ignored_and_the_variants_are_added_up(self):
        value, _ = self.ask(response(candidates(Yes=0.3, _yes=0.1, YES=0.1, No=0.3, _no=0.2)))
        self.assertAlmostEqual(value, 0.5 / 1.0)

    def test_tokens_other_than_yes_and_no_are_ignored(self):
        value, _ = self.ask(response(candidates(Yes=0.45, No=0.05, The=0.3, I=0.1)))
        self.assertAlmostEqual(value, 0.9)

    def test_only_yes_is_one_and_only_no_is_zero(self):
        self.assertEqual(self.ask(response(candidates(Yes=0.9)))[0], 1.0)
        self.assertEqual(self.ask(response(candidates(No=0.9)))[0], 0.0)

    def test_the_same_reply_always_gives_the_same_number(self):
        top = candidates(Yes=0.62, No=0.31)
        self.assertEqual(self.ask(response(top))[0], self.ask(response(top))[0])

    def test_the_request_is_at_temperature_0_with_token_probabilities_and_room_for_the_runtime_to_return_them(self):
        _, client = self.ask(response(candidates(Yes=0.9, No=0.1)))
        (request,) = client.requests
        self.assertEqual(request["temperature"], 0)
        self.assertIs(request["logprobs"], True)
        self.assertGreaterEqual(request["top_logprobs"], 2)
        self.assertGreaterEqual(request["max_tokens"], 3)  # LM Studio returns no probabilities for a one-token reply
        self.assertEqual([m["role"] for m in request["messages"]], ["system", "user"])
        self.assertEqual([m["content"] for m in request["messages"]], ["system text", "user text"])

    def test_a_reply_without_token_probabilities_is_none_and_says_so(self):
        for top in (None, []):
            with self.subTest(top=top):
                value, _ = self.ask(response(top))
                self.assertIsNone(value)
                self.assertEqual(lib.take_failure()[0], "no_probabilities")

    def test_a_first_token_that_is_not_yes_or_no_is_none_and_says_so(self):
        value, _ = self.ask(response(candidates(**{"<think>": 0.9, "Yes": 0.01, "No": 0.01})))
        self.assertIsNone(value)
        self.assertEqual(lib.take_failure()[0], "no_answer")
        self.assertLess(0.02, MIN_ANSWER_MASS)

    def test_the_failure_names_the_likeliest_first_tokens_and_their_shares(self):
        top = candidates(**{"": 0.56, "The": 0.04, "In": 0.03, "Output": 0.02, "Yes": 0.01})
        self.assertIsNone(self.ask(response(top))[0])
        kind, detail = lib.take_failure()
        self.assertEqual(kind, "no_answer")
        self.assertEqual(detail, "'' 56%, 'The' 4%, 'In' 3%")

    def test_a_timeout_is_none_and_recorded_as_one(self):
        value, _ = self.ask(CompletionTimeout())
        self.assertIsNone(value)
        self.assertEqual(lib.take_failure()[0], "timeout")

    def test_a_refused_request_is_none_and_recorded_as_a_bad_request(self):
        value, _ = self.ask(StatusError("context too long"))
        self.assertIsNone(value)
        self.assertEqual(lib.take_failure()[0], "bad_request")

    def test_any_other_failure_is_none_and_recorded_as_other(self):
        value, _ = self.ask(ConnectionError("refused"))
        self.assertIsNone(value)
        self.assertEqual(lib.take_failure()[0], "other")

    def test_a_success_clears_the_last_failure(self):
        self.ask(ConnectionError("refused"))
        self.ask(response(candidates(Yes=0.9, No=0.1)))
        self.assertEqual(lib.take_failure(), (None, None))


if __name__ == "__main__":
    unittest.main()
