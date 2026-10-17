"""
A model-free TokenBudget for tests: "tokens" are whitespace-
separated words, so no embedding model has to be loaded and chunk sizes are easy
to reason about. Not a test module (no TestCase), imported by the ones that need
a budget.
"""

import re

from qdrant_ingest_lib import TokenBudget

_WORD = re.compile(r"\S+")


def word_spans(text: str) -> list:
    """One (start, end) span per whitespace-separated word, like a tokenizer's
    offsets for a model whose tokens are words."""
    return [m.span() for m in _WORD.finditer(text)]


def word_budget(max_tokens: int = 20, overlap_tokens: int = 0) -> TokenBudget:
    return TokenBudget(max_tokens=max_tokens, overlap_tokens=overlap_tokens, token_spans=word_spans)
