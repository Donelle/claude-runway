#!/usr/bin/env python3
"""Tests for token-budget chunking in libs/qdrant_ingest_lib.py.

Stdlib-only and model-free: "tokens" are whitespace-separated words (see
tests/fake_token_budget.py), or single characters where a test needs to cut
inside a run of text with no whitespace at all.

    .venv/bin/python -m unittest discover -s tests

Why this exists: the line-window chunkers sized chunks in lines while the
embedder reads tokens and drops everything past its window, so most of every
chunk could never be matched by meaning. The property pinned here is that every
chunk fits the window in full, for every line structure (including a file that
is one enormous line), with offsets and line numbers that agree with the text.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "libs"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qdrant_ingest_lib as lib  # noqa: E402
from fake_token_budget import word_budget, word_spans  # noqa: E402
from qdrant_ingest_lib import (  # noqa: E402
    MANIFEST_FILENAME,
    TokenBudget,
    TokenBudgetError,
    build_entries,
    chunk_file,
    chunk_position,
    chunk_stream,
    file_has_indexable_content,
    format_largest_files,
    is_old_format_manifest,
    iter_entries,
    manifest_to_json,
    parse_manifest,
    validate_token_overlap,
)


def char_budget(max_tokens: int, overlap_tokens: int = 0) -> TokenBudget:
    """Every character is a token, so a cut has to land inside a run of text."""
    return TokenBudget(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        token_spans=lambda text: [(i, i + 1) for i in range(len(text))],
    )


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


class StreamInvariants(unittest.TestCase):
    """What must hold for any text, whatever its line structure."""

    def _check(self, text, budget, preferred=()):
        chunks = list(chunk_stream(text, budget, preferred))
        for chunk, start, end in chunks:
            self.assertEqual(text[start:end], chunk, "offsets and stored text must agree")
            self.assertLessEqual(len(budget.token_spans(chunk)), budget.max_tokens, "chunk exceeds the window")
            self.assertTrue(chunk.strip())
        # Nothing but whitespace is ever skipped: every non-space character is
        # inside some chunk.
        covered = bytearray(len(text))
        for _, start, end in chunks:
            for i in range(start, end):
                covered[i] = 1
        for i, ch in enumerate(text):
            if not ch.isspace():
                self.assertTrue(covered[i], f"character {i} ({ch!r}) is in no chunk")
        # Chunks advance.
        starts = [s for _, s, _ in chunks]
        self.assertEqual(starts, sorted(set(starts)))
        return chunks

    def test_short_text_is_one_chunk_equal_to_the_whole_text(self):
        text = "a b c"
        self.assertEqual(self._check(text, word_budget(10)), [(text, 0, len(text))])

    def test_many_lines(self):
        text = "\n".join(words(7, f"l{i}_") for i in range(40))
        self._check(text, word_budget(20))

    def test_one_enormous_line(self):
        """A minified file: the whole text is a single line, so a line-window
        chunker made it one chunk of which only the first window was searchable."""
        text = words(5000)
        chunks = self._check(text, word_budget(25))
        self.assertGreaterEqual(len(chunks), 5000 // 25)

    def test_no_whitespace_at_all_cuts_at_the_budget_exactly(self):
        text = "x" * 1000
        chunks = self._check(text, char_budget(64))
        self.assertTrue(all(len(c) <= 64 for c, _, _ in chunks))
        self.assertEqual([len(c) for c, _, _ in chunks[:-1]], [64] * (len(chunks) - 1))

    def test_with_overlap(self):
        self._check(words(300), word_budget(30, overlap_tokens=7))
        self._check("y" * 500, char_budget(40, overlap_tokens=9))

    def test_blank_lines_and_indentation(self):
        text = "\n\n   ".join(words(11, f"p{i}_") for i in range(30))
        self._check(text, word_budget(15, overlap_tokens=2))

    def test_unicode(self):
        text = " ".join("héllo wörld 日本語 😀".split()) * 1
        text = " ".join([text] * 60)
        self._check(text, word_budget(12, overlap_tokens=1))

    def test_empty_and_whitespace_only_text_yield_nothing(self):
        for text in ("", "   ", "\n\n\t \n"):
            with self.subTest(text=text):
                self.assertEqual(list(chunk_stream(text, word_budget(5))), [])


class OverlapIsInTokens(unittest.TestCase):
    def test_next_chunk_repeats_the_last_overlap_tokens(self):
        text = words(100)
        chunks = list(chunk_stream(text, word_budget(10, overlap_tokens=3)))
        self.assertGreater(len(chunks), 3)
        for (prev, _, _), (nxt, _, _) in zip(chunks, chunks[1:], strict=False):
            self.assertEqual(nxt.split()[:3], prev.split()[-3:])

    def test_no_overlap_means_adjacent_chunks(self):
        chunks = list(chunk_stream(words(100), word_budget(10)))
        for (_, _, end), (_, start, _) in zip(chunks, chunks[1:], strict=False):
            self.assertEqual(start, end)

    def test_overlap_applies_at_a_preferred_cut_too(self):
        lines = [words(6, f"s{i}_") for i in range(10)]
        text = "\n".join(lines)
        ends = []
        pos = 0
        for line in lines:
            pos += len(line) + 1
            ends.append(pos)
        chunks = list(chunk_stream(text, word_budget(10, overlap_tokens=2), preferred_cuts=ends))
        for (prev, _, _), (nxt, _, _) in zip(chunks, chunks[1:], strict=False):
            self.assertEqual(nxt.split()[:2], prev.split()[-2:])

    def test_a_trailing_overlap_only_chunk_is_not_emitted(self):
        """Text that ends in whitespace must not produce a final chunk that is
        just the previous chunk's overlap again."""
        text = words(20) + "\n\n\n"
        chunks = list(chunk_stream(text, word_budget(10, overlap_tokens=3)))
        last_words = chunks[-1][0].split()
        self.assertEqual(last_words[-1], "w19")
        self.assertEqual(sum(c.split().count("w19") for c, _, _ in chunks), 1)


class PreferredCutPoints(unittest.TestCase):
    def test_prefers_a_preferred_offset_in_the_back_half(self):
        text = words(40)
        # A preferred cut at the start of word 16 (inside a 20-word budget's
        # back half: words 10..20).
        offset = text.index("w16")
        chunks = list(chunk_stream(text, word_budget(20), preferred_cuts=[offset]))
        self.assertEqual(chunks[0][2], offset)

    def test_ignores_a_preferred_offset_in_the_front_half(self):
        """A boundary right after an early import must not shrink every chunk."""
        text = words(40)
        offset = text.index("w3")
        chunks = list(chunk_stream(text, word_budget(20), preferred_cuts=[offset]))
        self.assertGreater(chunks[0][2], offset)
        self.assertEqual(len(chunks[0][0].split()), 20)

    def test_takes_the_last_preferred_offset_that_fits(self):
        text = words(40)
        offsets = [text.index("w12"), text.index("w15"), text.index("w19"), text.index("w25")]
        chunks = list(chunk_stream(text, word_budget(20), preferred_cuts=offsets))
        self.assertEqual(chunks[0][2], text.index("w19"))

    def test_newline_beats_plain_whitespace(self):
        text = words(12) + "\n" + words(12, "z") + " " + words(12, "q")
        chunks = list(chunk_stream(text, word_budget(20)))
        self.assertTrue(chunks[0][0].endswith("\n"), repr(chunks[0][0][-10:]))

    def test_whitespace_beats_cutting_a_word(self):
        text = "abcdefghij " * 30
        chunks = list(chunk_stream(text, char_budget(25)))
        for _chunk, _, end in chunks[:-1]:
            self.assertTrue(text[end - 1].isspace() or text[end].isspace())

    def test_no_boundary_anywhere_still_cuts_at_the_budget(self):
        chunks = list(chunk_stream("z" * 90, char_budget(30)))
        self.assertEqual([len(c) for c, _, _ in chunks], [30, 30, 30])

    def test_chunks_are_not_stripped(self):
        text = "\n\n  alpha beta"
        self.assertEqual(list(chunk_stream(text, word_budget(10))), [(text, 0, len(text))])


class ChunkThatRetokenizesLonger(unittest.TestCase):
    """Cutting inside a word can make the shorter text tokenize into more
    tokens than the same characters did inside the longer one (WordPiece), so a
    chunk is re-counted on its own and shortened until it fits."""

    def test_a_chunk_that_counts_more_standing_alone_is_shortened(self):
        def spans(text):
            out = word_spans(text)
            # A word that ends the text and is longer than 3 characters splits
            # in two, the way a trailing fragment can.
            if out and out[-1][1] == len(text) and out[-1][1] - out[-1][0] > 3:
                s, e = out[-1]
                out = out[:-1] + [(s, s + 2), (s + 2, e)]
            return out

        budget = TokenBudget(max_tokens=6, overlap_tokens=0, token_spans=spans)
        text = " ".join(["word"] * 40)
        chunks = list(chunk_stream(text, budget))
        self.assertGreater(len(chunks), 3)
        for chunk, start, end in chunks:
            self.assertLessEqual(len(spans(chunk)), 6)
            self.assertEqual(text[start:end], chunk)


class TokenizersWhoseSpansDoNotShrinkTheChunk(unittest.TestCase):
    """Byte-level / byte-fallback tokenizers can give several tokens the SAME
    character span. Cutting at the end of the max_tokens-th token can then leave
    the end of the chunk unchanged, and the re-count loop must still make
    progress instead of recounting the same slice forever."""

    def _budget(self, tokens_per_char: int, max_tokens: int, calls: list):
        def spans(text):
            calls.append(len(text))
            if len(calls) > 5000:
                raise AssertionError("chunk_stream is not terminating")
            return [(i, i + 1) for i in range(len(text)) for _ in range(tokens_per_char)]

        return TokenBudget(max_tokens=max_tokens, overlap_tokens=0, token_spans=spans)

    def test_four_tokens_per_character_terminates_with_chunks_that_fit(self):
        calls: list = []
        budget = self._budget(4, 126, calls)
        text = "x" * 200
        chunks = list(chunk_stream(text, budget))
        self.assertEqual([len(c) for c, _, _ in chunks], [31, 31, 31, 31, 31, 31, 14])
        for chunk, start, end in chunks:
            self.assertLessEqual(len(budget.token_spans(chunk)), 126)
            self.assertEqual(text[start:end], chunk)
        self.assertEqual("".join(c for c, _, _ in chunks), text)

    def test_with_overlap_too(self):
        calls: list = []
        budget = TokenBudget(
            max_tokens=126, overlap_tokens=8,
            token_spans=lambda t: [(i, i + 1) for i in range(len(t)) for _ in range(4)],
        )
        chunks = list(chunk_stream("y" * 300, budget))
        self.assertGreater(len(chunks), 8)
        for chunk, _, _ in chunks:
            self.assertLessEqual(len(budget.token_spans(chunk)), 126)
        self.assertEqual(calls, [])

    def test_a_character_that_cannot_fit_fails_explicitly(self):
        calls: list = []
        budget = self._budget(200, 126, calls)  # one character is 200 tokens > 126
        with self.assertRaises(TokenBudgetError) as cm:
            list(chunk_stream("abc", budget))
        self.assertIn("single character", str(cm.exception))
        self.assertLess(len(calls), 50)

    def test_zero_width_spans_never_loop(self):
        calls: list = []

        def spans(text):
            calls.append(1)
            if len(calls) > 5000:
                raise AssertionError("chunk_stream is not terminating")
            return [(0, 0)] * (len(text) // 2)

        budget = TokenBudget(max_tokens=3, overlap_tokens=0, token_spans=spans)
        try:
            list(chunk_stream("abcdefghijkl", budget))
        except TokenBudgetError:
            pass  # failing explicitly is fine; hanging is not


class TokenBudgetValidation(unittest.TestCase):
    def test_overlap_must_be_smaller_than_the_window(self):
        for overlap in (10, 11):
            with self.subTest(overlap=overlap), self.assertRaises(TokenBudgetError) as cm:
                word_budget(10, overlap_tokens=overlap)
            self.assertIn("overlap", str(cm.exception))

    def test_window_must_hold_a_token(self):
        with self.assertRaises(TokenBudgetError):
            TokenBudget(max_tokens=0, overlap_tokens=0, token_spans=word_spans)

    def test_negative_overlap_is_rejected_without_a_window(self):
        self.assertIsNotNone(validate_token_overlap(-1))
        self.assertIsNone(validate_token_overlap(0))
        self.assertIsNone(validate_token_overlap(60))  # bigger than chunk_lines' 50: fine, it is tokens now

    def test_overlap_check_against_a_window(self):
        self.assertIsNone(validate_token_overlap(9, 10))
        self.assertIn("10", validate_token_overlap(10, 10) or "")


class ChunkFileWithABudget(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _write(self, name, text):
        path = self.dir / name
        path.write_bytes(text.encode("utf-8"))
        return path

    def _chunks(self, name, budget):
        return list(chunk_file(self.dir / name, name, 50, 10, budget))

    def test_metadata_shape(self):
        self._write("a.txt", words(60))
        for _, md in self._chunks("a.txt", word_budget(20)):
            self.assertEqual(set(md), {"file_path", "line_number", "char_range", "type"})
            self.assertEqual(md["type"], "doc")
            self.assertNotIn("line_range", md)

    def test_char_range_slices_the_decoded_file_exactly(self):
        text = "\n".join(words(9, f"r{i}_") for i in range(30))
        self._write("a.txt", text)
        decoded = (self.dir / "a.txt").read_text(encoding="utf-8", errors="ignore")
        for chunk, md in self._chunks("a.txt", word_budget(14)):
            start, end = map(int, md["char_range"].split("-"))
            self.assertEqual(decoded[start:end], chunk)

    def test_line_number_is_the_line_where_the_chunk_text_begins(self):
        text = "\n".join(words(9, f"r{i}_") for i in range(30))
        self._write("a.txt", text)
        chunks = self._chunks("a.txt", word_budget(14, overlap_tokens=2))
        self.assertGreater(len(chunks), 4)
        for chunk, md in chunks:
            start = int(md["char_range"].split("-")[0])
            self.assertEqual(md["line_number"], text.count("\n", 0, start) + 1)
            first_word = chunk.split()[0]
            self.assertIn(first_word, text.splitlines()[md["line_number"] - 1])

    def test_a_one_line_file_has_line_number_one_for_every_chunk(self):
        """In a one-line file every chunk used to read line_range 1-1; char_range
        is what tells them apart."""
        self._write("min.txt", words(400))
        chunks = self._chunks("min.txt", word_budget(20))
        self.assertGreater(len(chunks), 10)
        self.assertEqual({md["line_number"] for _, md in chunks}, {1})
        self.assertEqual(len({md["char_range"] for _, md in chunks}), len(chunks))

    def test_crlf_files_report_offsets_into_the_normalized_text(self):
        self._write("w.txt", "one two three\r\nfour five six\r\nseven eight nine\r\n")
        chunks = self._chunks("w.txt", word_budget(4))
        decoded = (self.dir / "w.txt").read_text(encoding="utf-8", errors="ignore")
        self.assertNotIn("\r", decoded)
        for chunk, md in chunks:
            start, end = map(int, md["char_range"].split("-"))
            self.assertEqual(decoded[start:end], chunk)

    def test_undecodable_bytes_are_dropped_before_offsets_are_counted(self):
        path = self.dir / "bad.txt"
        path.write_bytes(b"alpha \xff\xfe beta gamma delta epsilon zeta eta theta")
        chunks = list(chunk_file(path, "bad.txt", 50, 10, word_budget(3)))
        decoded = path.read_text(encoding="utf-8", errors="ignore")
        for chunk, md in chunks:
            start, end = map(int, md["char_range"].split("-"))
            self.assertEqual(decoded[start:end], chunk)

    def test_markdown_prefers_to_end_chunks_at_headings(self):
        sections = [f"# Section {i}\n" + words(7, f"s{i}_") + "\n" for i in range(8)]
        self._write("d.md", "".join(sections))
        chunks = self._chunks("d.md", word_budget(30))
        self.assertGreater(len(chunks), 1)
        for chunk, _ in chunks[1:]:
            # Every chunk after the first starts at a heading: the cut landed
            # right before one, inside the back half of the window.
            self.assertTrue(chunk.startswith("#"), repr(chunk[:20]))

    def test_python_prefers_to_end_chunks_at_top_level_boundaries(self):
        funcs = [f"def f{i}(a):\n    return a + {i}\n\n\n" for i in range(12)]
        self._write("m.py", "".join(funcs))
        chunks = self._chunks("m.py", word_budget(18))
        self.assertGreater(len(chunks), 1)
        for chunk, _ in chunks[1:]:
            self.assertTrue(chunk.lstrip("\n").startswith("def "), repr(chunk[:20]))

    def test_brace_language_prefers_to_end_chunks_between_members(self):
        members = [f"    void M{i}() {{\n        Run({i});\n    }}\n" for i in range(12)]
        self._write("c.cs", "class C {\n" + "".join(members) + "}\n")
        chunks = self._chunks("c.cs", word_budget(20))
        self.assertGreater(len(chunks), 1)
        for chunk, _ in chunks[:-1]:
            self.assertTrue(chunk.rstrip().endswith("}"), repr(chunk[-20:]))

    def test_without_a_budget_the_old_line_chunkers_still_run(self):
        self._write("a.txt", "\n".join(f"line {i}" for i in range(120)))
        chunks = list(chunk_file(self.dir / "a.txt", "a.txt", 50, 10))
        self.assertIn("line_range", chunks[0][1])
        self.assertNotIn("line_number", chunks[0][1])

    def test_build_entries_and_iter_entries_pass_the_budget_through(self):
        self._write("a.txt", words(60))
        budget = word_budget(20)
        entries, skipped = build_entries(self.dir, "docs", 50, 10, budget=budget)
        self.assertEqual(skipped, [])
        self.assertTrue(all("char_range" in md for _, md in entries))
        streamed = [item for item in iter_entries(self.dir, "docs", 50, 10, budget=budget)]
        self.assertEqual([c for c, _ in streamed], [c for c, _ in entries])

    def test_build_entries_reports_each_file_to_on_file(self):
        self._write("big.txt", words(100))
        self._write("small.txt", "a few words")
        seen = {}
        build_entries(self.dir, "docs", 50, 10, budget=word_budget(20), on_file=seen.__setitem__)
        self.assertEqual(seen, {"big.txt": 5, "small.txt": 1})


class ChunkPositionReader(unittest.TestCase):
    """Readers must accept the new and the old metadata shape: points keep
    line_range until their collection is re-indexed, and find_in_collection can
    search another repo's collection that has not been."""

    def test_new_shape(self):
        self.assertEqual(
            chunk_position({"file_path": "a.py", "line_number": 12, "char_range": "340-910"}),
            "line 12, chars 340-910",
        )

    def test_new_shape_without_char_range(self):
        self.assertEqual(chunk_position({"line_number": 3}), "line 3")

    def test_old_shape(self):
        self.assertEqual(chunk_position({"file_path": "a.py", "line_range": "10-40"}), "lines 10-40")

    def test_new_shape_wins_when_both_are_present(self):
        self.assertTrue(chunk_position({"line_number": 1, "line_range": "1-9"}).startswith("line 1"))

    def test_neither_shape_and_non_dicts_give_an_empty_string(self):
        for metadata in ({}, {"source": "https://x"}, None, "text", 5):
            with self.subTest(metadata=metadata):
                self.assertEqual(chunk_position(metadata), "")


class LargestFiles(unittest.TestCase):
    def test_ranks_by_chunk_count_then_name_and_caps_the_list(self):
        counts = {f"f{i}.js": i for i in range(1, 9)}
        text = format_largest_files(counts)
        self.assertTrue(text.startswith("Largest files by chunk count: f8.js (8), f7.js (7)"))
        self.assertNotIn("f3.js", text)
        self.assertEqual(text.count("("), 5)

    def test_empty_is_empty(self):
        self.assertEqual(format_largest_files({}), "")


class IndexableContent(unittest.TestCase):
    def test_whitespace_only_files_have_none(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "e.py").write_text("")
            (Path(d) / "w.py").write_text("\n \t\n")
            (Path(d) / "x.py").write_text("x = 1\n")
            self.assertFalse(file_has_indexable_content(Path(d) / "e.py"))
            self.assertFalse(file_has_indexable_content(Path(d) / "w.py"))
            self.assertTrue(file_has_indexable_content(Path(d) / "x.py"))

    def test_unreadable_raises_oserror_like_chunk_file(self):
        with self.assertRaises(OSError):
            file_has_indexable_content(Path("/definitely/not/here.py"))


class VersionedManifest(unittest.TestCase):
    def test_round_trip(self):
        files = {"a.py": "h1", "dir/b.md": "h2"}
        import json

        self.assertEqual(parse_manifest(json.loads(manifest_to_json(files))), files)

    def test_keeps_the_filename_users_already_gitignore(self):
        self.assertEqual(MANIFEST_FILENAME, ".qdrant_index_manifest.json")

    def test_keys_inside_files_are_paths_only(self):
        import json

        doc = json.loads(manifest_to_json({"a.py": "h"}))
        self.assertEqual(set(doc["files"]), {"a.py"})
        self.assertEqual(doc["manifest_format"], lib.MANIFEST_FORMAT_VERSION)

    def test_old_flat_newer_and_malformed_documents_are_ignored(self):
        for doc in (
            {"a.py": "h1"},
            {},
            {"manifest_format": 1, "files": {"a.py": "h"}},
            {"manifest_format": lib.MANIFEST_FORMAT_VERSION + 1, "files": {"a.py": "h"}},
            {"manifest_format": lib.MANIFEST_FORMAT_VERSION},
            {"manifest_format": lib.MANIFEST_FORMAT_VERSION, "files": ["a.py"]},
            {"manifest_format": lib.MANIFEST_FORMAT_VERSION, "files": {"a.py": 3}},
            [],
            "text",
            None,
        ):
            with self.subTest(doc=doc):
                self.assertIsNone(parse_manifest(doc))

    def test_old_format_detection(self):
        self.assertTrue(is_old_format_manifest({"a.py": "h"}))
        self.assertFalse(is_old_format_manifest({}))
        self.assertFalse(is_old_format_manifest({"manifest_format": 2, "files": {}}))
        self.assertFalse(is_old_format_manifest({"a.py": 5}))
        self.assertFalse(is_old_format_manifest(["a.py"]))


if __name__ == "__main__":
    unittest.main()
