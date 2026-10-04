#!/usr/bin/env python3
"""Tests for libs/model_pricing.py: model-name normalization, tier selection
and per-turn cost math. Stdlib-only, no network.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

import model_pricing as P  # noqa: E402

M = 1_000_000


def _cost(*args, **kwargs) -> float:
    cost = P.turn_cost_usd(*args, **kwargs)
    assert cost is not None
    return cost


class NormalizeModel(unittest.TestCase):
    def test_transcript_ids_and_display_names_map_to_table_keys(self):
        cases = {
            "claude-haiku-4-5-20251001": "claude-haiku-4-5",
            "claude-opus-5-5[1m]": "claude-opus-5-5",
            "Claude Sonnet 4.6": "claude-sonnet-4-6",
            "GPT-5.6 Sol": "gpt-5-6-sol",
            "GPT-5.3-Codex": "gpt-5-3-codex",
            "MAI-Code-1.1-Flash": "mai-code-1-1-flash",
            "Kimi K3": "kimi-k3",
        }
        for raw, key in cases.items():
            self.assertEqual(P.normalize_model(raw), key, raw)
            self.assertIsNotNone(P.lookup_rates(raw), raw)

    def test_empty_and_unknown(self):
        self.assertEqual(P.normalize_model(None), "")
        self.assertIsNone(P.lookup_rates(None))
        self.assertIsNone(P.lookup_rates("<synthetic>"))
        self.assertIsNone(P.turn_cost_usd("not-a-model", 10, 0, 0, 10))


class RateTable(unittest.TestCase):
    def test_every_entry_is_well_formed(self):
        for key, tiers in P.RATES.items():
            self.assertEqual(P.normalize_model(key), key, key)
            self.assertEqual(tiers[0][0], 0, key)
            thresholds = [t[0] for t in tiers]
            self.assertEqual(thresholds, sorted(set(thresholds)), key)
            for tier in tiers:
                self.assertEqual(len(tier), 5, key)


class TurnCost(unittest.TestCase):
    def test_four_token_types_priced_separately(self):
        # Claude Sonnet 5.5: $2 in, $0.20 cached, $2.50 write, $10 out.
        cost = _cost("claude-sonnet-5-5", M, M, M, M)
        self.assertAlmostEqual(cost, 2.00 + 0.20 + 2.50 + 10.00)

    def test_no_cache_write_rate_bills_writes_as_input(self):
        # GPT-5 mini has no separate cache write cost.
        self.assertAlmostEqual(_cost("gpt-5-mini", 0, 0, M, 0), 0.25)

    def test_long_context_tier_applies_above_threshold_only(self):
        # GPT-5.4: <= 272K prompt -> $2.50 in; > 272K -> $5.00 in, $22.50 out.
        at = _cost("gpt-5.4", 272_000, 0, 0, 0)
        self.assertAlmostEqual(at, 272_000 * 2.50 / M)
        above = _cost("gpt-5.4", 100_000, 172_001, 0, 1000)
        self.assertAlmostEqual(above, (100_000 * 5.00 + 172_001 * 0.50 + 1000 * 22.50) / M)

    def test_fast_flag_falls_back_to_standard_rate_without_fast_entry(self):
        self.assertAlmostEqual(_cost("claude-opus-4-8", M, 0, 0, 0, fast=True), 10.00)
        self.assertAlmostEqual(_cost("claude-opus-5", M, 0, 0, 0, fast=True), 5.00)

    def test_ai_credits(self):
        self.assertAlmostEqual(P.usd_to_ai_credits(1.23), 123)


if __name__ == "__main__":
    unittest.main()
