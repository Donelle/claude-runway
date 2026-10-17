#!/usr/bin/env python3
"""Tests for FastEmbedProvider's token window and token budget.

`build_token_budget` reaches into fastembed attributes that are not public API
(`embedding_model.model.tokenizer`), so these run it against REAL `tokenizers`
objects: a tiny WordPiece tokenizer built in memory, configured the way
fastembed configures a model's tokenizer (truncation to a window, padding, and
a post-processor adding [CLS]/[SEP]). That needs no model download and still
breaks if the `tokenizers` API the provider relies on changes. One more test
runs against the real default model when it is already in the local fastembed
cache, and is skipped otherwise.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sys
import types
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))

import qdrant_embeddings  # noqa: E402
from qdrant_embeddings import FastEmbedProvider, TokenWindowError  # noqa: E402
from qdrant_ingest_lib import TokenBudgetError, chunk_stream  # noqa: E402

WINDOW = 16


def _real_tokenizer(window: int = WINDOW, truncation: bool = True):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "alpha": 4, "beta": 5, "gamma": 6, "delta": 7, "##s": 8}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    if truncation:
        tokenizer.enable_truncation(max_length=window)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    return tokenizer


def _provider(tokenizer, model_name: str = "test/model") -> FastEmbedProvider:
    """A FastEmbedProvider whose embedding model is just enough of a model for
    the token-window reads, with no model load."""
    fake_model = types.SimpleNamespace(model=types.SimpleNamespace(tokenizer=tokenizer), model_name=model_name)
    with mock.patch.object(qdrant_embeddings, "TextEmbedding", return_value=fake_model):
        return FastEmbedProvider(model_name)


class TokenWindow(unittest.TestCase):
    def test_reads_the_truncation_max_length_of_the_loaded_models_tokenizer(self):
        self.assertEqual(_provider(_real_tokenizer(window=WINDOW)).token_window(), WINDOW)
        self.assertEqual(_provider(_real_tokenizer(window=40)).token_window(), 40)

    def test_a_model_with_no_tokenizer_names_the_model_and_what_is_missing(self):
        fake_model = types.SimpleNamespace(model=types.SimpleNamespace(), model_name="acme/embed-v9")
        with mock.patch.object(qdrant_embeddings, "TextEmbedding", return_value=fake_model):
            provider = FastEmbedProvider("acme/embed-v9")
        with self.assertRaises(TokenWindowError) as cm:
            provider.token_window()
        message = str(cm.exception)
        self.assertIn("acme/embed-v9", message)
        self.assertIn("embedding_model.model.tokenizer", message)
        self.assertIn("rather than guessing", message)

    def test_a_tokenizer_without_truncation_names_the_model(self):
        provider = _provider(_real_tokenizer(truncation=False), "acme/no-trunc")
        with self.assertRaises(TokenWindowError) as cm:
            provider.token_window()
        self.assertIn("acme/no-trunc", str(cm.exception))
        self.assertIn("truncation", str(cm.exception))

    def test_a_non_positive_or_non_integer_max_length_is_rejected(self):
        for bad in (0, -4, 1.5, True, None, "128"):
            with self.subTest(max_length=bad):
                tokenizer = types.SimpleNamespace(truncation={"max_length": bad})
                with self.assertRaises(TokenWindowError):
                    _provider(tokenizer).token_window()

    def test_it_is_a_token_budget_error_so_every_entry_point_catches_it(self):
        self.assertTrue(issubclass(TokenWindowError, TokenBudgetError))


class BuildTokenBudget(unittest.TestCase):
    def test_usable_budget_is_the_window_minus_the_special_tokens(self):
        budget = _provider(_real_tokenizer(window=WINDOW)).build_token_budget(3)
        self.assertEqual(budget.max_tokens, WINDOW - 2)  # [CLS] and [SEP]
        self.assertEqual(budget.overlap_tokens, 3)

    def test_counting_uses_a_separate_tokenizer_without_truncation_or_padding(self):
        tokenizer = _real_tokenizer(window=WINDOW)
        budget = _provider(tokenizer).build_token_budget(0)
        long_text = " ".join(["alpha"] * 100)
        # The shared tokenizer truncates and pads: it can never report more
        # than the window, which is exactly why it cannot be used to count.
        self.assertEqual(len(tokenizer.encode(long_text).ids), WINDOW)
        self.assertEqual(len(budget.token_spans(long_text)), 100)
        # And building the counter did not change what gets embedded.
        self.assertEqual(tokenizer.truncation["max_length"], WINDOW)
        self.assertIsNotNone(tokenizer.padding)

    def test_spans_are_character_offsets_of_real_tokens_only(self):
        budget = _provider(_real_tokenizer()).build_token_budget(0)
        text = "alpha  beta\ngammas"
        spans = budget.token_spans(text)
        self.assertEqual([text[s:e] for s, e in spans], ["alpha", "beta", "gamma", "s"])

    def test_no_special_tokens_leak_into_the_spans(self):
        budget = _provider(_real_tokenizer()).build_token_budget(0)
        self.assertEqual(budget.token_spans(""), [])
        self.assertEqual(len(budget.token_spans("alpha")), 1)

    def test_every_chunk_fits_the_embedded_window_in_full(self):
        """The point of the ticket: what is stored is what gets embedded."""
        tokenizer = _real_tokenizer(window=WINDOW)
        budget = _provider(tokenizer).build_token_budget(2)
        text = "\n".join(" ".join(["alpha", "beta", "gamma", "delta"] * 3) for _ in range(40))
        chunks = list(chunk_stream(text, budget))
        self.assertGreater(len(chunks), 10)
        for chunk, _, _ in chunks:
            ids = tokenizer.encode(chunk).ids
            self.assertLess(ids.count(0), len(ids), "padding only appears when the chunk is shorter than the window")
            real_tokens = [i for i in ids if i not in (0, 2, 3)]
            self.assertEqual(len(real_tokens), len(budget.token_spans(chunk)), "the embedder truncated this chunk")

    def test_overlap_not_smaller_than_the_usable_window_is_an_error(self):
        provider = _provider(_real_tokenizer(window=WINDOW))
        with self.assertRaises(TokenBudgetError) as cm:
            provider.build_token_budget(WINDOW - 2)
        self.assertIn("overlap", str(cm.exception))

    def test_a_window_with_no_room_for_content_is_an_error(self):
        provider = _provider(_real_tokenizer(window=2))
        with self.assertRaises(TokenWindowError) as cm:
            provider.build_token_budget(0)
        self.assertIn("special tokens", str(cm.exception))

    def test_a_tokenizer_that_cannot_be_copied_is_a_clear_error_not_a_fallback(self):
        tokenizer = types.SimpleNamespace(truncation={"max_length": 128})  # no to_str/from_str
        with self.assertRaises(TokenWindowError) as cm:
            _provider(tokenizer, "acme/odd").build_token_budget(0)
        self.assertIn("acme/odd", str(cm.exception))


class PassageEmbeddingAddsNoText(unittest.TestCase):
    """The budget counts the chunk text alone, so FastEmbed must embed exactly
    that text. `FastEmbedProvider.embed_documents` goes through `passage_embed`;
    a model class that prepended text there (an E5-style "passage: " prefix)
    would make a full-budget chunk silently lose its tail to truncation.

    Checked against the installed FastEmbed's own model-class registry, with no
    model download: a release that adds such a class or override fails here and
    the budget has to account for it (or reject that model)."""

    def _classes(self):
        from fastembed.text.text_embedding import TextEmbedding

        registry = getattr(TextEmbedding, "EMBEDDINGS_REGISTRY", None)
        if not registry:
            self.skipTest("this FastEmbed has no TextEmbedding.EMBEDDINGS_REGISTRY to inspect")
        return registry

    def test_every_model_class_embeds_passages_as_plain_embed(self):
        import inspect

        from fastembed.text.text_embedding_base import TextEmbeddingBase

        for cls in self._classes():
            with self.subTest(model_class=cls.__name__):
                if cls.passage_embed is TextEmbeddingBase.passage_embed:
                    continue
                # The one allowed override selects a task adapter: no text added.
                source = inspect.getsource(cls.passage_embed)
                self.assertIn("task_id", source, f"{cls.__name__}.passage_embed is overridden: account for it")
                self.assertNotIn("passage:", source)

    def test_no_model_class_applies_a_text_prefix_when_embedding(self):
        import inspect

        for cls in self._classes():
            with self.subTest(model_class=cls.__name__):
                self.assertNotIn("passage:", inspect.getsource(cls))


class RealDefaultModel(unittest.TestCase):
    """Against the real default model, when it is already cached locally (no
    download is attempted: CI without the cache skips this)."""

    def test_window_and_budget_of_the_default_model(self):
        name = "sentence-transformers/all-MiniLM-L6-v2"
        qdrant_embeddings_module = qdrant_embeddings
        from qdrant_ingest_lib import ensure_persistent_fastembed_cache

        ensure_persistent_fastembed_cache()
        try:
            model = qdrant_embeddings_module.TextEmbedding(name, local_files_only=True)
        except Exception as e:  # not cached here
            self.skipTest(f"{name} is not in the local fastembed cache: {type(e).__name__}")
        with mock.patch.object(qdrant_embeddings_module, "TextEmbedding", return_value=model):
            provider = FastEmbedProvider(name)
        window = provider.token_window()
        self.assertGreater(window, 2)
        budget = provider.build_token_budget(10)
        self.assertEqual(budget.max_tokens, window - 2)
        text = " ".join(f"identifier_{i} = compute(value_{i}, other_{i})" for i in range(500))
        for chunk, _, _ in chunk_stream(text, budget):
            self.assertLessEqual(len(budget.token_spans(chunk)), budget.max_tokens)
            # What fastembed would embed (truncated, with [CLS]/[SEP]) is the
            # whole chunk: nothing was cut off.
            self.assertEqual(
                len(model.model.tokenizer.encode(chunk).ids), len(budget.token_spans(chunk)) + 2
            )


if __name__ == "__main__":
    unittest.main()
