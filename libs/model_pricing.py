"""
Per-token model rates for turning the savings tracker's actual token counts
into an estimated dollar cost. Used by libs/savings_ledger.py, which prices
each assistant turn of a transcript as it parses it.

The rates are GitHub Copilot's published per-token rates for additional usage
(USD per 1M tokens; 1 GitHub AI credit = $0.01). Copilot bills each request on
four token types -- fresh input, cached input (cache reads), cache writes and
output -- which map one-to-one onto the four `message.usage` counters Claude
Code writes to its transcript, so no ratio guessing is needed.

Long-context tiers: some models switch to a higher rate for the WHOLE request
once its prompt exceeds a threshold. "Prompt" here is every input-side token
of the request (fresh + cache read + cache write), which is why tiers are
chosen per turn, not from session totals -- a session of many small turns
never crosses the threshold even when its summed input does.

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

# One tier: (prompt-token threshold above which it applies, or 0 for the
# default tier, input, cached_input, cache_write, output), all USD per 1M
# tokens. cache_write is None where the vendor charges no separate write cost;
# those tokens are then billed at the fresh input rate, since they were sent
# as input either way.
# Keys are normalized model names (see normalize_model). Tiers are listed
# lowest threshold first.
_M = 1_000_000
RATES: dict = {
    # OpenAI
    "gpt-5-mini":        [(0,       0.25,  0.025, None,  2.00)],
    "gpt-5-3-codex":     [(0,       1.75,  0.175, None, 14.00)],
    "gpt-5-4":           [(0,       2.50,  0.25,  None, 15.00), (272_000,  5.00, 0.50, None, 22.50)],
    "gpt-5-4-mini":      [(0,       0.75,  0.075, None,  4.50)],
    "gpt-5-4-nano":      [(0,       0.20,  0.02,  None,  1.25)],
    "gpt-5-5":           [(0,       5.00,  0.50,  None, 30.00), (272_000, 10.00, 1.00, None, 45.00)],
    "gpt-5-6-luna":      [(0,       0.20,  0.02,  0.25,  1.20), (200_000,  0.40, 0.04, 0.50,  1.80)],
    "gpt-5-6-sol":       [(0,       4.00,  0.40,  5.00, 20.00), (272_000,  8.00, 0.80, 10.00, 30.00)],
    "gpt-5-6-terra":     [(0,       2.00,  0.20,  2.50, 12.00), (272_000,  4.00, 0.40, 5.00, 18.00)],
    "gpt-6-astra":       [(0,      10.00,  1.00, 12.50, 50.00), (272_000, 20.00, 2.00, 25.00, 75.00)],
    "gpt-6-luna":        [(0,       0.10,  0.01, 0.125,  0.50), (272_000,  0.20, 0.02, 0.25,  0.75)],
    "gpt-6-sol":         [(0,       2.00,  0.20,  2.50, 10.00), (272_000,  4.00, 0.40, 5.00, 15.00)],
    "gpt-6-1-sol":       [(0,       2.00,  0.10,  2.50, 10.00), (272_000,  4.00, 0.20, 5.00, 15.00)],
    # Anthropic
    "claude-haiku-4-5":       [(0,  1.00, 0.10,  1.25,  5.00)],
    "claude-sonnet-4":        [(0,  3.00, 0.30,  3.75, 15.00)],
    "claude-sonnet-4-6":      [(0,  3.00, 0.30,  3.75, 15.00)],
    "claude-opus-4-8":        [(0,  5.00, 0.50,  6.25, 25.00)],
    "claude-opus-4-8-fast":   [(0, 10.00, 1.00, 12.50, 50.00)],
    "claude-opus-5":          [(0,  5.00, 0.50,  6.25, 25.00)],
    "claude-opus-5-5":        [(0,  4.00, 0.20,  5.00, 20.00)],
    "claude-sonnet-5":        [(0,  2.00, 0.20,  2.50, 10.00)],
    "claude-sonnet-5-5":      [(0,  2.00, 0.20,  2.50, 10.00)],
    "claude-fable-5":         [(0, 10.00, 1.00, 12.50, 50.00)],
    "claude-fable-5-1":       [(0, 10.00, 0.25, 12.50, 50.00)],
    # Google (promotional pricing through 2026-12-31)
    "gemini-3-7-flash":  [(0,       0.75,  0.075, None,  3.75)],
    "gemini-3-8-flash":  [(0,       0.75,  0.075, None,  3.75)],
    # Microsoft
    "mai-code-1-1-flash": [(0,      0.20,  0.02,  None,  1.20)],
    # xAI
    "grok-4-5":          [(0,       2.00,  0.50,  None,  6.00), (200_000,  4.00, 1.00, None, 12.00)],
    "grok-4-6":          [(0,       2.00,  0.50,  None,  6.00), (200_000,  4.00, 1.00, None, 12.00)],
    "grok-4-7":          [(0,       2.00,  0.50,  None,  6.00), (200_000,  4.00, 1.00, None, 12.00)],
    # Moonshot AI
    "kimi-k3":           [(0,       3.00,  0.30,  None, 15.00)],
}

_DATE_SUFFIX = re.compile(r"-\d{8}$")


def normalize_model(model: Optional[str]) -> str:
    """
    Map a model name as it appears anywhere (a transcript's `message.model`
    such as "claude-haiku-4-5-20251001" or "claude-opus-5-5[1m]", or a display
    name such as "GPT-5.6 Sol") onto a RATES key: lowercase, a trailing
    context-window tag like "[1m]" and a trailing -YYYYMMDD snapshot date
    dropped, and every run of spaces, dots or underscores turned into "-".
    """
    if not model:
        return ""
    name = model.strip().lower()
    name = re.sub(r"\[[^\]]*\]$", "", name)
    name = re.sub(r"[\s._]+", "-", name).strip("-")
    return _DATE_SUFFIX.sub("", name)


def lookup_rates(model: Optional[str]) -> Optional[list]:
    """Tier list for `model`, or None when it isn't in the table."""
    return RATES.get(normalize_model(model))


def turn_cost_usd(model: Optional[str], input_tokens: int, cache_read_tokens: int,
                  cache_write_tokens: int, output_tokens: int, fast: bool = False) -> Optional[float]:
    """
    Estimated USD cost of ONE request (assistant turn), or None when the model
    is unpriced. `fast` selects a model's fast-mode rate where one exists
    (the transcript's `usage.speed` reads "fast" on those turns).
    """
    tiers = None
    if fast:
        tiers = lookup_rates(normalize_model(model) + "-fast")
    if tiers is None:
        tiers = lookup_rates(model)
    if tiers is None:
        return None
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    tier = tiers[0]
    for candidate in tiers:
        if prompt_tokens > candidate[0]:
            tier = candidate
    _, inp, cached, write, out = tier
    if write is None:
        write = inp
    return (input_tokens * inp + cache_read_tokens * cached
            + cache_write_tokens * write + output_tokens * out) / _M


def usd_to_ai_credits(usd: float) -> float:
    return usd / USD_PER_AI_CREDIT
