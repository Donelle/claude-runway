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

VECTOR_NAME_PREFIX = "fast-"


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
