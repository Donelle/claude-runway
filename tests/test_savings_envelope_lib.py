#!/usr/bin/env python3
"""Tests for libs/savings_envelope_lib.py, the shared definition of the savings
envelope. Stdlib only, no network:

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import savings_envelope_lib as lib  # noqa: E402

METRICS = {"tool": "compress_file", "raw_chars": 10, "out_chars": 5, "credited": True, "source": "f"}


def _env(text="kept", metrics=METRICS):
    return json.dumps({"compressed_text": text, "metrics": metrics})


class IsSavingsEnvelope(unittest.TestCase):
    def test_accepts_the_exact_shape(self):
        self.assertTrue(lib.is_savings_envelope({"compressed_text": "x", "metrics": dict(METRICS)}))

    def test_count_fields_are_not_checked(self):
        for counts in ({}, {"raw_chars": None, "out_chars": None}, {"raw_chars": "lots", "out_chars": True}):
            with self.subTest(counts=counts):
                metrics = {"tool": "t", "credited": False, "source": "", **counts}
                self.assertTrue(lib.is_savings_envelope({"compressed_text": "x", "metrics": metrics}))

    def test_rejects_anything_else(self):
        for value in (
            None, "x", 3, [], {}, {"compressed_text": "x"}, {"metrics": dict(METRICS)},
            {"compressed_text": "x", "metrics": dict(METRICS), "extra": 1},
            {"compressed_text": 1, "metrics": dict(METRICS)},
            {"compressed_text": "x", "metrics": []},
            {"compressed_text": "x", "metrics": dict(METRICS, tool=1)},
            {"compressed_text": "x", "metrics": dict(METRICS, credited=1)},
            {"compressed_text": "x", "metrics": dict(METRICS, source=None)},
        ):
            with self.subTest(value=value):
                self.assertFalse(lib.is_savings_envelope(value))


class ParseEnvelopeText(unittest.TestCase):
    def test_decodes_the_envelope_bare_or_wrapped_by_the_mcp_layer(self):
        for text in (_env(), json.dumps({"result": _env()})):
            with self.subTest(text=text[:30]):
                self.assertEqual(lib.parse_envelope_text(text)["compressed_text"], "kept")

    def test_does_not_decode_a_third_layer_or_repair_anything(self):
        for text in (
            json.dumps({"result": json.dumps({"result": _env()})}),
            _env()[:-1], "", "not json", "[1]", "3", None, 5,
        ):
            with self.subTest(text=text):
                self.assertIsNone(lib.parse_envelope_text(text))

    def test_very_deep_nesting_is_not_an_envelope_and_does_not_raise(self):
        self.assertIsNone(lib.parse_envelope_text("[" * 100000))


class TextIsEnvelope(unittest.TestCase):
    def test_true_for_text_the_hook_would_unwrap(self):
        for text in (_env(), json.dumps({"result": _env()}), json.dumps(json.loads(_env()), indent=2)):
            with self.subTest(text=text[:30]):
                self.assertTrue(lib.text_is_envelope(text))

    def test_false_for_ordinary_text(self):
        for text in (
            "", "plain", '{"name": "pkg"}', "[1]", '{"result": "x"}', "[compressed 1 -> 1 chars]\nx",
            json.dumps({"result": json.dumps({"result": _env()})}), None,
        ):
            with self.subTest(text=text):
                self.assertFalse(lib.text_is_envelope(text))

    def test_agrees_with_the_parser_for_the_wrapped_form(self):
        # The guard exists so that nothing returned bare parses as an envelope
        # once the MCP layer wraps it; check that equivalence directly.
        for text in (_env(), json.dumps({"result": _env()}), "plain", '{"a": 1}'):
            with self.subTest(text=text[:30]):
                wrapped = json.dumps({"result": text})
                self.assertEqual(lib.text_is_envelope(text), lib.parse_envelope_text(wrapped) is not None
                                 or lib.parse_envelope_text(text) is not None)


if __name__ == "__main__":
    unittest.main()
