"""
FastEmbed embedding provider used by every Qdrant-backed tool in this repo.

This is the repo's own `FastEmbedProvider`, written to replace the one the
third-party Qdrant MCP server package provided and this repo used to import.
It keeps that class's behavior
exactly, because existing collections were written under it and must stay
readable:

  - The named vector in a collection is `"fast-" + <model basename, lowercased>`
    (so `BAAI/bge-small-en` -> `fast-bge-small-en`). A different name would make
    every existing collection look like it has no vector for the new code.
  - `embed_documents` uses FastEmbed's `passage_embed` and `embed_query` uses
    `query_embed`; some models embed the two differently, so they are not
    interchangeable.
  - Embedding runs in the default executor, because FastEmbed is synchronous
    and CPU-bound and would otherwise block the MCP server's event loop.

Constructing a provider loads the ONNX model (and downloads it on a cold cache),
so callers construct it as late as they can.

It is also where the indexer learns how much of a text the model actually reads.
FastEmbed truncates every text to a token window taken from the model's own
tokenizer configuration and silently drops the rest, so a chunk longer than the
window is only searchable by its first part. `token_window()` reads that limit
from the loaded model and `build_token_budget()` turns it into the budget the
chunker cuts to, so a different EMBEDDING_MODEL changes the chunk size with no
code or config change. The reads reach into private FastEmbed attributes and are
kept in this one class, failing with an error that names the model.

It loads from the local cache first, with no network access, and only goes to
the network when the cache does not hold the model. Without that, a cache HIT
still made a live huggingface.co round-trip to resolve the revision, which was
measured at 59.6s on a slow network against 2.0s with it skipped, long enough
to pass Claude Code's MCP connection timeout. Doing it in code means no server
needs an `HF_HUB_OFFLINE` entry in its config, and a cold cache still downloads.
"""

import asyncio
from typing import Any

from fastembed import TextEmbedding

from qdrant_ingest_lib import TokenBudget, TokenBudgetError

VECTOR_NAME_PREFIX = "fast-"


class TokenWindowError(TokenBudgetError):
    """The embedding model's token window could not be read.

    Names the model and what was missing. Chunking must stop here: the window
    is what decides how much of a chunk the embedder actually reads, and a
    guessed value (or line-based chunking) would quietly bring back the bug
    that most of every chunk never reaches its vector.
    """


def load_text_embedding(model_name: str, text_embedding_cls=None) -> Any:
    """Builds the FastEmbed model, from the local cache when it is complete.

    `local_files_only=True` is a per-call argument that fastembed passes to
    `huggingface_hub.snapshot_download`, whose network call is gated on that
    literal parameter, so it works regardless of when `huggingface_hub` was
    imported (unlike setting `HF_HUB_OFFLINE` in `os.environ`, which
    `huggingface_hub` reads once at import). On a cold or incomplete cache it
    raises immediately, and the second call downloads as before.
    """
    # Looked up at call time (not a default argument) so a test or caller that
    # replaces this module's `TextEmbedding` is honored.
    cls = text_embedding_cls or TextEmbedding
    try:
        return cls(model_name, local_files_only=True)
    except Exception:
        return cls(model_name)


def vector_name_for_model(model_name: str) -> str:
    """The named-vector key a collection uses for `model_name`.

    Pure and importable on its own so tests can pin the on-disk format without
    loading a model.
    """
    return f"{VECTOR_NAME_PREFIX}{model_name.split('/')[-1].lower()}"


class FastEmbedProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_model: Any = load_text_embedding(model_name)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.passage_embed(documents))
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        return embeddings[0].tolist()

    def get_vector_name(self) -> str:
        return vector_name_for_model(self.embedding_model.model_name)

    def get_vector_size(self) -> int:
        # `_get_model_description` is private to fastembed, the one remaining
        # reach into a third-party internal. It is isolated here so a rename
        # breaks one line, and tests/test_qdrant_connector.py exercises it
        # against the installed fastembed.
        return self.embedding_model._get_model_description(self.model_name).dim

    # -- token window -------------------------------------------
    #
    # Everything below reaches into fastembed attributes that are not public
    # API (`embedding_model.model.tokenizer`). It lives here, next to
    # `_get_model_description` above, so a fastembed change breaks in one
    # place, with an error naming the model, and tests/test_qdrant_embeddings.py
    # runs it against the installed fastembed.

    def _shared_tokenizer(self) -> Any:
        """The tokenizer fastembed itself embeds with (truncating, padding)."""
        try:
            return self.embedding_model.model.tokenizer
        except AttributeError as e:
            raise TokenWindowError(
                f"Cannot read the token window of embedding model '{self.model_name}': the loaded model has "
                f"no tokenizer at embedding_model.model.tokenizer ({e}). This looks like a fastembed change "
                f"or an unsupported model; indexing is stopped rather than guessing a chunk size."
            ) from e

    def token_window(self) -> int:
        """The model's own token limit: the `max_length` fastembed applies with
        `enable_truncation`, derived from the model's tokenizer_config.json
        (the stricter of `model_max_length` and `max_length`). It counts the
        special tokens the tokenizer adds around every text."""
        tokenizer = self._shared_tokenizer()
        truncation = getattr(tokenizer, "truncation", None)
        if not isinstance(truncation, dict) or "max_length" not in truncation:
            raise TokenWindowError(
                f"Cannot read the token window of embedding model '{self.model_name}': its tokenizer has no "
                f"truncation max_length (truncation setting: {truncation!r}). Indexing is stopped rather than "
                f"guessing a chunk size."
            )
        window = truncation["max_length"]
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise TokenWindowError(
                f"Cannot read the token window of embedding model '{self.model_name}': truncation max_length "
                f"is {window!r}, not a positive integer. Indexing is stopped rather than guessing a chunk size."
            )
        return window

    def build_token_budget(self, overlap_tokens: int) -> TokenBudget:
        """
        The TokenBudget the indexing entry points chunk with.

        max_tokens is the model's window minus the special tokens the tokenizer
        adds around every text, so a chunk of that size is embedded in full.

        Counting uses a SEPARATE copy of the tokenizer with truncation and
        padding off. The shared one truncates and pads, so len(ids) is always
        the window size whatever the text length, and switching truncation off
        on it would change what gets embedded. The copy is made through the
        tokenizer's own serialization (the same tokenizer.json content), so it
        cannot differ from the shared one except in those two settings.

        What is counted is the chunk text itself, and that is what FastEmbed
        embeds: `embed_documents` calls `passage_embed`, which in every model
        class FastEmbed ships is `TextEmbeddingBase.passage_embed` (a plain
        `embed`, no text added; a model card's "prefixes necessary" is advice
        for the caller, not something FastEmbed applies) except the multitask
        Jina v3 class, which selects a task adapter and adds no tokens.
        tests/test_qdrant_embeddings.py pins that against the installed
        FastEmbed, so a release that starts adding text (an E5-style
        "passage: " prefix) fails a test instead of silently truncating chunks.

        Raises TokenWindowError (a TokenBudgetError) if any of that cannot be
        determined, and TokenBudgetError for an overlap the window cannot take.
        """
        window = self.token_window()
        tokenizer = self._shared_tokenizer()
        try:
            specials = tokenizer.num_special_tokens_to_add(False)
            counting = tokenizer.__class__.from_str(tokenizer.to_str())
            counting.no_truncation()
            counting.no_padding()
        except (AttributeError, TypeError, ValueError) as e:
            raise TokenWindowError(
                f"Cannot build a token counter for embedding model '{self.model_name}': {e}. Indexing is "
                f"stopped rather than guessing a chunk size."
            ) from e
        usable = window - specials
        if usable < 1:
            raise TokenWindowError(
                f"Embedding model '{self.model_name}' has a token window of {window} and its tokenizer adds "
                f"{specials} special tokens per text, leaving no room for content."
            )

        def token_spans(text: str) -> list:
            return list(counting.encode(text, add_special_tokens=False).offsets)

        return TokenBudget(max_tokens=usable, overlap_tokens=overlap_tokens, token_spans=token_spans)
