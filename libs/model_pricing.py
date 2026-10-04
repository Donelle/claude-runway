"""
Per-token Anthropic model rates for turning the savings tracker's actual token
counts into an estimated dollar cost. Used by libs/savings_ledger.py, which
prices each assistant turn of a transcript as it parses it.

The rates are GitHub Copilot's published per-token rates for Anthropic models
(USD per 1M tokens; 1 GitHub AI credit = $0.01). Only Anthropic models are
listed: Claude Code transcripts only ever contain Claude turns. Copilot bills
each request on four token types -- fresh input, cached input (cache reads),
cache writes and output -- which map one-to-one onto the four `message.usage`
counters Claude Code writes to its transcript, so no ratio guessing is needed.

The table is a snapshot, not a live feed: rates change, so update RATES (and
RATES_AS_OF) when Copilot publishes new ones. A model missing from the table
is reported as unpriced rather than guessed at.
"""

import re
from typing import Optional

RATES_SOURCE = "GitHub Copilot"
RATES_AS_OF = "2026-10-04"

# USD per GitHub AI credit.
USD_PER_AI_CREDIT = 0.01

# (input, cached_input, cache_write, output), all USD per 1M tokens. Keys are
# normalized model names (see normalize_model); "-fast" marks a model's
# fast-mode rate.
_M = 1_000_000
RATES: dict = {
    "claude-haiku-4-5":     (1.00,  0.10,  1.25,  5.00),
    "claude-sonnet-4":      (3.00,  0.30,  3.75, 15.00),
    "claude-sonnet-4-6":    (3.00,  0.30,  3.75, 15.00),
    "claude-opus-4-8":      (5.00,  0.50,  6.25, 25.00),
    "claude-opus-4-8-fast": (10.00, 1.00, 12.50, 50.00),
    "claude-opus-5":        (5.00,  0.50,  6.25, 25.00),
    "claude-opus-5-5":      (4.00,  0.20,  5.00, 20.00),
    "claude-sonnet-5":      (2.00,  0.20,  2.50, 10.00),
    "claude-sonnet-5-5":    (2.00,  0.20,  2.50, 10.00),
    "claude-fable-5":       (10.00, 1.00, 12.50, 50.00),
    "claude-fable-5-1":     (10.00, 0.25, 12.50, 50.00),
}

_DATE_SUFFIX = re.compile(r"-\d{8}$")


def normalize_model(model: Optional[str]) -> str:
    """
    Map a model name as it appears anywhere (a transcript's `message.model`
    such as "claude-haiku-4-5-20251001" or "claude-opus-5-5[1m]", or a display
    name such as "Claude Sonnet 4.6") onto a RATES key: lowercase, a trailing
    context-window tag like "[1m]" and a trailing -YYYYMMDD snapshot date
    dropped, and every run of spaces, dots or underscores turned into "-".
    """
    if not model:
        return ""
    name = model.strip().lower()
    name = re.sub(r"\[[^\]]*\]$", "", name)
    name = re.sub(r"[\s._]+", "-", name).strip("-")
    return _DATE_SUFFIX.sub("", name)


def lookup_rates(model: Optional[str]) -> Optional[tuple]:
    """Rate tuple for `model`, or None when it isn't in the table."""
    return RATES.get(normalize_model(model))


def turn_cost_usd(model: Optional[str], input_tokens: int, cache_read_tokens: int,
                  cache_write_tokens: int, output_tokens: int, fast: bool = False) -> Optional[float]:
    """
    Estimated USD cost of ONE request (assistant turn), or None when the model
    is unpriced. `fast` selects a model's fast-mode rate where one exists
    (the transcript's `usage.speed` reads "fast" on those turns).
    """
    rates = None
    if fast:
        rates = lookup_rates(normalize_model(model) + "-fast")
    if rates is None:
        rates = lookup_rates(model)
    if rates is None:
        return None
    inp, cached, write, out = rates
    return (input_tokens * inp + cache_read_tokens * cached
            + cache_write_tokens * write + output_tokens * out) / _M


def usd_to_ai_credits(usd: float) -> float:
    return usd / USD_PER_AI_CREDIT
