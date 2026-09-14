#!/usr/bin/env python3
"""Tests for libs/qdrant_ingest_lib.py chunking.

Stdlib-only (unittest, no pytest) and no network -- the chunkers are pure
functions over text, so this needs neither Qdrant nor an embedding model:

    .venv/bin/python -m unittest discover -s tests

Why this file exists: all four bugs pinned here were SILENT. Nothing errors
when a chunker splits in the wrong place, skips lines, or explodes chunk
count -- the index just quietly contains worse content (or costs far more to
build), and a later qdrant-find returns a comment line instead of an answer.
The first three were found by indexing this repo with its own indexer:

  1. '.txt' routed to the markdown chunker, whose heading test was a bare
     startswith("#"). This repo's comment-heavy requirements.txt came out as
     9 single-line chunks out of 12.
  2. '#' comments inside fenced code blocks counted as headings, so a README
     of shell samples fragmented mid-sample.
  3. Content before the first heading was dropped outright. A .instructions.md
     file with YAML frontmatter was indexed starting at line 5.
  4. overlap >= chunk_size collapsed chunk_lines()'s step to 1, silently
     turning "chunk every N lines" into "chunk every line" -- multiplying
     embedding/storage cost by roughly chunk_size with no error anywhere
     (issue #26). Caught by code review, not by indexing this repo, since
     this repo's own defaults never hit the bad case.
"""

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from qdrant_ingest_lib import (  # noqa: E402
    _strip_c_style_line,
    batched,
    brace_boundary_indexes,
    build_entries,
    chunk_code_by_boundaries,
    chunk_file,
    chunk_lines,
    chunk_markdown,
    compute_file_hashes,
    ensure_persistent_fastembed_cache,
    files_removed_since_last_sync,
    heading_line_indexes,
    iter_entries,
    iter_files,
    python_boundary_indexes,
    validate_chunk_params,
)

# Chunk params matching the CLI defaults closely enough to be representative.
CHUNK_SIZE, OVERLAP = 50, 10


class HeadingDetection(unittest.TestCase):
    """heading_line_indexes must find real ATX headings and nothing else."""

    def test_real_headings(self):
        lines = [
            "# Title",  # 0
            "text",
            "## Section",  # 2
            "###### Deepest",  # 3
            "   ### Indented three spaces is still a heading",  # 4
            "#",  # 5 -- bare hash, '#' then end of line
        ]
        self.assertEqual(heading_line_indexes(lines), [0, 2, 3, 4, 5])

    def test_non_headings(self):
        lines = [
            "#!/usr/bin/env python3",
            "#include <stdio.h>",
            "#comment-with-no-space",
            "####### seven hashes exceeds h6",
            "    # four spaces of indent is a code block, not a heading",
            "not a heading at all",
        ]
        self.assertEqual(
            heading_line_indexes(lines),
            [],
            "none of these are ATX headings, but all matched the old startswith('#')",
        )

    def test_hashes_inside_fenced_code_are_ignored(self):
        lines = [
            "# Real Heading",  # 0
            "",
            "```bash",
            "# install the thing",
            "pip install foo  # inline",
            "```",
            "",
            "## Another Real Heading",  # 7
            "~~~python",
            "# a tilde fence counts too",
            "~~~",
        ]
        self.assertEqual(heading_line_indexes(lines), [0, 7])

    def test_unbalanced_fence_suppresses_rather_than_invents(self):
        lines = ["# Heading", "```bash", "# not a heading", "## also not"]
        self.assertEqual(
            heading_line_indexes(lines),
            [0],
            "an unclosed fence should lose splits, never fabricate them",
        )

    def test_longer_fence_is_not_closed_by_a_shorter_one(self):
        # The standard way to show markdown inside markdown: a ````-fenced
        # block containing a ``` sample. Found in PR review -- the original
        # "any fence toggles" rule closed the outer block at the inner ```
        # and then read every '#' line after it as a heading.
        lines = [
            "# Real Heading",         # 0
            "````markdown",           # 1  opens a 4-backtick fence
            "```bash",                # 2  inner fence, must not close it
            "# not a heading",        # 3
            "```",                    # 4  inner close, must not close it
            "## also not a heading",  # 5
            "````",                   # 6  this closes the outer fence
            "## Real Heading Two",    # 7
        ]
        self.assertEqual(heading_line_indexes(lines), [0, 7])

    def test_fence_character_must_match_to_close(self):
        lines = ["# H", "```bash", "~~~", "# not a heading", "```", "## H2"]
        self.assertEqual(
            heading_line_indexes(lines),
            [0, 5],
            "a ~~~ line must not close a ``` block",
        )

    def test_closing_fence_may_not_carry_an_info_string(self):
        # An opening fence may have an info string ("```bash"); a closing one
        # may not, so a second "```bash" opens nothing and closes nothing.
        lines = ["# H", "```bash", "# not a heading", "```bash", "# still not", "```", "## H2"]
        self.assertEqual(heading_line_indexes(lines), [0, 6])

    def test_longer_closing_fence_still_closes(self):
        lines = ["# H", "```", "# not a heading", "`````", "## H2"]
        self.assertEqual(
            heading_line_indexes(lines),
            [0, 4],
            "a closing run longer than the opener is valid",
        )


class MarkdownChunking(unittest.TestCase):
    def _chunks(self, text):
        return list(chunk_markdown(text, CHUNK_SIZE, OVERLAP))

    def test_preamble_before_first_heading_is_kept(self):
        text = "\n".join([
            "---",
            "applyTo: '**/*.py'",
            "---",
            "",
            "# First Heading",
            "body",
            "",
            "## Second Heading",
            "more body",
        ])
        chunks = self._chunks(text)
        joined = " ".join(c for c, _, _ in chunks)
        self.assertIn("applyTo", joined, "frontmatter must be indexed, not dropped")
        first_content, first_start, _ = chunks[0]
        self.assertEqual(first_start, 1, "the preamble chunk starts at line 1")
        self.assertIn("---", first_content)

    def test_no_preamble_chunk_when_file_opens_with_a_heading(self):
        text = "# A\nbody\n\n## B\nbody"
        starts = [start for _, start, _ in self._chunks(text)]
        self.assertEqual(starts, [1, 4], "no empty leading chunk when there's no preamble")

    def test_splits_on_headings(self):
        text = "# A\na body\n## B\nb body\n## C\nc body"
        chunks = self._chunks(text)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[0][0].startswith("# A"))
        self.assertTrue(chunks[1][0].startswith("## B"))

    def test_falls_back_to_line_chunking_with_fewer_than_two_headings(self):
        text = "\n".join(["# Only One Heading"] + [f"line {i}" for i in range(120)])
        chunks = self._chunks(text)
        self.assertGreater(len(chunks), 1, "a long single-heading doc still gets split by lines")
        self.assertEqual(chunks[0][1], 1)

    def test_code_fence_comments_do_not_fragment_a_section(self):
        text = "\n".join([
            "# Install",
            "```bash",
            "# step one",
            "brew install foo",
            "# step two",
            "foo init",
            "```",
            "## Next",
            "done",
        ])
        chunks = self._chunks(text)
        self.assertEqual(len(chunks), 2, "the fenced block stays inside its own section")
        self.assertIn("step two", chunks[0][0], "the whole sample travels with its heading")


class BraceBoundaryDetection(unittest.TestCase):
    """brace_boundary_indexes must find real curly-brace depth-zero points
    and never miscount a brace hiding inside a string literal or a comment
    (issue #55)."""

    def test_braces_inside_strings_and_line_comments_are_not_counted(self):
        lines = [
            'void Foo() {',
            '    var s = "{ not a brace }";',
            '    // comment with a brace {',
            '}',
        ]
        self.assertEqual(brace_boundary_indexes(lines), {3})

    def test_brace_inside_multiline_block_comment_is_not_counted(self):
        lines = [
            'void Foo() {',
            '    int x = 1; /* {',
            '       still comment',
            '    end */ }',
        ]
        self.assertEqual(
            brace_boundary_indexes(lines),
            {3},
            "the '{' inside the block comment must not open an extra, "
            "never-closed brace that prevents depth from returning to 0",
        )

    def test_top_level_statements_between_functions_are_boundaries(self):
        lines = [
            'function a() {',  # 0 depth 1
            '    return 1;',   # 1
            '}',                # 2 depth 0 -- boundary
            '',                 # 3 depth 0 -- boundary
            'function b() {',  # 4 depth 1
            '    return 2;',   # 5
            '}',                # 6 depth 0 -- boundary
        ]
        self.assertEqual(brace_boundary_indexes(lines), {2, 3, 6})

    def test_brace_inside_backtick_template_literal_is_not_counted(self):
        # PR #137 review: a template/raw-string literal (JS/TS/Go) can
        # contain a brace character that must not decrement tracked depth
        # and fabricate a boundary inside the surrounding function.
        lines = [
            'function a() {',
            '    const text = `}`;',
            '    return 1;',
            '}',
        ]
        self.assertEqual(
            brace_boundary_indexes(lines),
            {3},
            "the '}' inside the backtick literal must not close the function early",
        )

    def test_multiline_backtick_template_literal_is_not_counted(self):
        lines = [
            'function a() {',
            '    const text = `',
            '    a brace in here: }',
            '    `;',
            '    return 1;',
            '}',
        ]
        self.assertEqual(
            brace_boundary_indexes(lines),
            {5},
            "a brace inside a multi-line backtick literal must not be counted either",
        )

    def test_methods_inside_a_class_get_boundaries_between_them(self):
        # PR #137 review: depth only ever returning to 0 meant a whole
        # class/namespace longer than chunk_size had NO boundary candidate
        # between its own methods -- the core "reduce mid-class splits"
        # case the issue is about. A closing '}' that returns to directly
        # inside a class/struct/interface/... body must count too, while a
        # control-flow block (if/for/while/...) must not be confused for one.
        lines = [
            'namespace App {',            # 0
            '    public class Foo {',      # 1
            '        public int A() {',    # 2 -- inside method, not boundary
            '            if (true) {',     # 3 -- control-flow, not boundary
            '                return 1;',   # 4
            '            }',                # 5 -- closes 'if', still inside method body, not boundary
            '        }',                    # 6 -- closes method A -- boundary (back to class body)
            '',                             # 7 -- boundary
            '        public int B() {',    # 8 -- inside method, not boundary
            '            return 2;',       # 9
            '        }',                    # 10 -- closes method B -- boundary
            '    }',                        # 11 -- closes class -- boundary
            '}',                            # 12 -- closes namespace -- boundary
        ]
        boundaries = brace_boundary_indexes(lines)
        self.assertIn(6, boundaries, "must find the safe point between method A and method B")
        self.assertNotIn(2, boundaries, "must not treat entering method A's own body as safe")
        self.assertNotIn(3, boundaries, "must not treat entering an if-block as a member boundary")
        self.assertNotIn(5, boundaries, "closing the if-block must not look like closing a method")
        self.assertNotIn(9, boundaries, "must not treat a statement inside method B as safe")

    def test_container_keyword_in_a_type_annotation_is_not_a_declaration(self):
        # PR #137 review: matching the container keyword anywhere on the
        # line (rather than requiring it be the line's OWN declaration
        # keyword) tagged a function whose signature merely MENTIONS one of
        # these words -- e.g. a TypeScript "object" type annotation -- as a
        # container. That mistagged EVERY line in its body as a safe
        # boundary (confirmed by direct reproduction before this fix: all 5
        # lines of the fixture below came back as boundaries).
        lines = [
            'function process(value: object) {',  # 0
            '    const x = 1;',                     # 1
            '    const y = 2;',                     # 2
            '    return x + y;',                    # 3
            '}',                                     # 4
        ]
        boundaries = brace_boundary_indexes(lines)
        self.assertEqual(
            boundaries, {4},
            "a type annotation mentioning 'object' must not tag the function itself a container",
        )

    def test_container_keyword_as_a_parameter_type_is_not_a_declaration(self):
        lines = [
            'void foo(struct Point p) {',
            '    do_thing();',
            '}',
        ]
        boundaries = brace_boundary_indexes(lines)
        self.assertEqual(
            boundaries, {2},
            "'struct' appearing as a parameter's type name must not tag the function a container",
        )

    def test_declaration_with_modifier_keyword_is_still_recognized(self):
        # Regression guard the other direction: a real declaration prefixed
        # by one or more modifier keywords must still be tagged container.
        lines = [
            'public static class Foo {',
            '    public int A() {',
            '        return 1;',
            '    }',
            '',
            '    public int B() {',
            '        return 2;',
            '    }',
            '}',
        ]
        boundaries = brace_boundary_indexes(lines)
        self.assertIn(3, boundaries, "must still find the boundary between method A and method B")

    def test_brace_inside_csharp_verbatim_string_is_not_counted(self):
        # PR #137 review: a C# verbatim string (@"...") can legally span
        # multiple lines -- unlike an ordinary "...", which cannot -- and a
        # brace inside one must not be counted (confirmed by direct
        # reproduction before this fix: a false boundary appeared right at
        # the string-content line).
        lines = [
            'void Foo() {',
            '    string s = @"',
            '    a brace in a verbatim string: }',
            '    ";',
            '    return;',
            '}',
        ]
        self.assertEqual(
            brace_boundary_indexes(lines), {5},
            "the '}' inside the verbatim string must not close the function early",
        )

    def test_csharp_verbatim_string_doubled_quote_escape_does_not_close_early(self):
        # A doubled '""' inside a verbatim string is an escaped literal '"',
        # not the closing delimiter.
        lines = [
            'void Foo() {',
            '    string s = @"a ""quoted"" word with a brace: }";',
            '    return;',
            '}',
        ]
        self.assertEqual(brace_boundary_indexes(lines), {3})

    def test_module_dot_exports_is_not_a_container_declaration(self):
        # PR #137 review: '\b' is satisfied between a word char and ANY
        # non-word char, including '.', so '\bmodule\b' matched the start
        # of an ordinary CommonJS 'module.exports = function () {'
        # expression -- confirmed by direct reproduction: every line of the
        # fixture below came back as a boundary before this fix.
        lines = [
            'module.exports = function () {',
            '    const x = 1;',
            '    const y = 2;',
            '    return x + y;',
            '};',
        ]
        self.assertEqual(
            brace_boundary_indexes(lines), {4},
            "'module.exports' must not be read as a TypeScript module declaration",
        )

    def test_kotlin_anonymous_object_expression_is_still_recognized(self):
        # Regression guard the other direction: Kotlin's anonymous object
        # expression ('object : Base() {') has a real space after the
        # keyword (unlike 'module.exports') and must still be tagged.
        lines = ['object : Base() {', '    fun a() {', '        return 1', '    }', '}']
        boundaries = brace_boundary_indexes(lines)
        self.assertIn(3, boundaries, "must still recognize a real anonymous-object declaration")

    def test_container_keyword_as_a_variable_type_is_not_a_declaration(self):
        # PR #137 review (round 4): the keyword can have real whitespace
        # after it and STILL not be a declaration -- 'object value = ...'
        # uses "object" as a plain TYPE NAME for a variable, not to open a
        # container. Confirmed by direct reproduction: every line of the
        # object-initializer body below was flagged before this fix.
        lines = [
            'void Foo() {',
            '    object value = new Foo {',
            '        A = 1,',
            '        B = 2,',
            '    };',
            '    return;',
            '}',
        ]
        boundaries = brace_boundary_indexes(lines)
        self.assertEqual(
            boundaries, {6},
            "'object value = ...' must not tag the initializer's own brace a container",
        )

    def test_struct_as_a_variable_type_is_not_a_declaration(self):
        lines = ['void foo() {', '    struct Point p = { 1, 2 };', '    return;', '}']
        boundaries = brace_boundary_indexes(lines)
        self.assertEqual(boundaries, {3})

    def test_allman_style_multiline_method_signature_is_never_split_from_its_body(self):
        # PR #137 review (round 4): a multi-line method signature doesn't
        # open any NEW brace of its own while its parameter list spans
        # several lines, so every one of those lines (including the line
        # that closes the signature's own ')') still looked boundary-safe,
        # right up to the line before the method's own Allman-style '{' --
        # confirmed by direct reproduction (a chunk_size=6 cut landed
        # immediately before the '{', separating signature from body).
        lines = [
            'public class Foo',      # 0 -- Allman: own '{' is next line, not safe
            '{',                      # 1
            '    public void Bar(',  # 2 -- mid-signature, not safe
            '        int a,',        # 3 -- mid-signature, not safe
            '        int b',         # 4 -- mid-signature, not safe
            '    )',                  # 5 -- signature just closed, but body's '{' is next -- not safe
            '    {',                  # 6
            '        DoSomething();',# 7
            '    }',                  # 8 -- safe (back to class body)
            '}',                       # 9 -- safe (end of file)
        ]
        boundaries = brace_boundary_indexes(lines)
        for bad in (0, 2, 3, 4, 5):
            self.assertNotIn(bad, boundaries, f"line {bad} must not be a boundary")
        self.assertIn(8, boundaries)
        self.assertIn(9, boundaries)

    def test_krandr_style_multiline_signature_still_gets_the_paren_depth_guard(self):
        # Same signature shape, but K&R style (brace on the same line as
        # the closing paren) -- the paren-depth guard alone (without
        # needing the Allman lookahead) must still keep the mid-signature
        # lines out of the boundary set.
        lines = [
            'public class Foo {',       # 0
            '    public void Bar(',    # 1 -- mid-signature, not safe
            '        int a,',           # 2 -- mid-signature, not safe
            '        int b) {',        # 3 -- signature closes AND opens body on the same line
            '        DoSomething();',  # 4
            '    }',                    # 5 -- safe
            '}',                         # 6 -- safe
        ]
        boundaries = brace_boundary_indexes(lines)
        self.assertNotIn(1, boundaries)
        self.assertNotIn(2, boundaries)
        self.assertIn(5, boundaries)
        self.assertIn(6, boundaries)


class PythonBoundaryDetection(unittest.TestCase):
    """python_boundary_indexes must not treat a line inside a multi-line
    call/list/dict expression as a boundary (issue #55), and must find the
    real top-level gap between two function definitions."""

    def test_multiline_call_args_are_not_false_boundaries(self):
        lines = [
            'def foo():',          # 0
            '    x = some_call(',  # 1 depth 1 -- inside the call
            '        1,',          # 2 depth 1
            '        2,',          # 3 depth 1
            '    )',                # 4 depth 0, next line indented -- not boundary
            '    return x',         # 5 depth 0, next real line is top-level -- boundary
            '',                     # 6 blank, next real line top-level -- boundary
            'def bar():',           # 7 next line indented -- not boundary
            '    return 2',        # 8 last line -- boundary
        ]
        self.assertEqual(python_boundary_indexes(lines), {5, 6, 8})

    def test_bracket_char_inside_string_or_comment_does_not_count(self):
        lines = [
            'def foo():',
            '    s = "( not a paren"  # comment with a ( too',
            '    return s',
        ]
        # depth never actually goes positive despite the '(' characters
        # above, since both are inside a string/comment -- so line 1 (index
        # 1) must still be eligible as a boundary once its next line is
        # checked; there's no unmatched bracket masking it.
        boundaries = python_boundary_indexes(lines)
        self.assertIn(2, boundaries, "the final line is always a safe boundary")

    def test_methods_inside_a_class_get_boundaries_between_them(self):
        # PR #137 review: requiring column-0 indentation meant a whole
        # class body longer than chunk_size had NO boundary candidate
        # between its own sibling methods -- the exact "reduce mid-class
        # splits" case the issue is about.
        lines = [
            'class Foo:',           # 0 -- right after header, not boundary
            '    def a(self):',    # 1 -- header itself, not boundary
            '        return 1',    # 2 -- end of method a -- boundary
            '',                     # 3 -- boundary
            '    def b(self):',    # 4 -- header itself, not boundary
            '        return 2',    # 5 -- last line -- boundary
        ]
        boundaries = python_boundary_indexes(lines)
        self.assertIn(2, boundaries, "must find the safe point between method a and method b")
        self.assertNotIn(0, boundaries)
        self.assertNotIn(1, boundaries)
        self.assertNotIn(4, boundaries)

    def test_triple_quoted_string_content_is_never_a_boundary(self):
        # PR #137 review: a line like `return """` (unterminated on that
        # line) must never be treated as safe just because the STRING'S OWN
        # unindented payload on a later line happens to look like column-0
        # top-level code -- that's string content, not real code.
        lines = [
            'def foo():',
            '    return """',
            'some unindented payload line',
            '"""',
        ]
        boundaries = python_boundary_indexes(lines)
        self.assertNotIn(1, boundaries, "must not end a chunk mid-way through opening the docstring")
        self.assertNotIn(2, boundaries, "the string's own content is never a boundary")

    def test_triple_quoted_string_does_not_corrupt_class_indentation_tracking(self):
        # A docstring line that happens to look unindented (column 0) must
        # not be mistaken for a real top-level statement that dedents out
        # of (and thereby corrupts tracking of) the enclosing class body.
        lines = [
            'class Foo:',
            '    def a(self):',       # 1
            '        return """',     # 2 -- opens the docstring, not a boundary
            'unindented docstring payload, not real code',  # 3 -- string content, not a boundary
            '        """',            # 4 -- docstring closes here -- boundary (still member-level, not module-level)
            '',                        # 5 -- boundary
            '    def b(self):',       # 6 -- header itself, not boundary
            '        return 2',       # 7 -- last line -- boundary
        ]
        boundaries = python_boundary_indexes(lines)
        self.assertIn(4, boundaries, "must find the safe point right after the docstring genuinely closes")
        self.assertNotIn(2, boundaries, "must not end a chunk mid-way through opening the docstring")
        self.assertNotIn(3, boundaries, "the string's own unindented-looking content is never a boundary")
        self.assertNotIn(6, boundaries, "method b's own header, whose next line is its own body, is not a boundary")

    def test_decorator_is_never_split_from_its_definition(self):
        # PR #137 review: a decorator sits at the same indentation as the
        # def/class it decorates, so it satisfied the column-0 rule and was
        # flagged as a boundary itself -- confirmed by direct reproduction
        # before this fix (index 0, the '@route' line, was in the result).
        lines = ['@route', 'def handler():', '    return 1']
        boundaries = python_boundary_indexes(lines)
        self.assertNotIn(0, boundaries, "the decorator line itself must never be a boundary")

    def test_multiline_decorator_is_never_split_from_its_definition(self):
        lines = ['@app.route(', '    "/x",', ')', 'def handler():', '    return 1']
        boundaries = python_boundary_indexes(lines)
        self.assertNotIn(2, boundaries, "the decorator's own closing line must not be a boundary either")

    def test_stacked_decorators_are_never_split_from_each_other_or_the_def(self):
        lines = ['@first', '@second', 'def handler():', '    return 1']
        boundaries = python_boundary_indexes(lines)
        self.assertNotIn(0, boundaries)
        self.assertNotIn(1, boundaries)

    def test_comment_only_line_is_not_treated_as_a_real_dedent(self):
        # PR #137 review: Python comments carry no indentation meaning --
        # a comment sitting at a class's member-indentation level, in
        # between two lines that are BOTH still part of the same method's
        # body, must not be read as a real dedent back to member level
        # (confirmed by direct reproduction: index 2, "x = 1", was flagged
        # as a boundary before this fix, even though method a continues
        # with "return x" right after the comment).
        lines = [
            'class Foo:',              # 0
            '    def a(self):',        # 1
            '        x = 1',           # 2 -- must NOT be a boundary
            '    # a misleadingly-dedented comment, method a continues below',  # 3
            '        return x',        # 4
            '',                          # 5
            '    def b(self):',        # 6
            '        return 2',        # 7
        ]
        boundaries = python_boundary_indexes(lines)
        self.assertNotIn(2, boundaries, "method a's body continues past the comment -- not a real boundary")

    def test_comment_only_line_does_not_corrupt_class_indentation_stack(self):
        # A comment that happens to be fully dedented (column 0) must not
        # be mistaken for a real top-level statement that pops the
        # enclosing class context.
        lines = [
            'class Foo:',
            '    def a(self):',
            '        x = 1',
            '# a fully-dedented comment, method a still continues below',
            '        return x',
            '',
            '    def b(self):',
            '        return 2',
        ]
        boundaries = python_boundary_indexes(lines)
        self.assertNotIn(2, boundaries)
        self.assertIn(4, boundaries, "the real boundary between method a and method b must still be found")

    def test_docstring_only_function_body_is_never_split_from_its_header(self):
        # PR #137 review (round 4): a function whose ENTIRE body is a
        # multi-line docstring strips down to whitespace on every one of
        # those lines -- previously indistinguishable from blank/comment
        # filler, so the scan for "the next real line" after 'def foo():'
        # skipped straight over the whole docstring and landed on the
        # FOLLOWING top-level def, making foo()'s header look
        # boundary-safe (confirmed by direct reproduction: index 0 was
        # flagged before this fix).
        lines = [
            'def foo():',
            '    """',
            '    a multi-line docstring',
            '    """',
            '',
            'def bar():',
            '    return 2',
        ]
        boundaries = python_boundary_indexes(lines)
        self.assertNotIn(0, boundaries, "foo()'s header must not be split from its own docstring body")

    def test_docstring_only_method_body_inside_a_class_is_never_split(self):
        lines = [
            'class Foo:',
            '    def a(self):',
            '        """',
            '        just a docstring, no other body',
            '        """',
            '',
            '    def b(self):',
            '        return 2',
        ]
        boundaries = python_boundary_indexes(lines)
        self.assertNotIn(1, boundaries, "method a's header must not be split from its own docstring body")


class ChunkCodeByBoundaries(unittest.TestCase):
    """chunk_code_by_boundaries must nudge chunk ends to a real boundary
    instead of an arbitrary fixed line count, and must always fall back
    safely (never hang, never grow unboundedly) when no boundary exists in
    the lookback window (issue #55)."""

    def test_does_not_split_either_function_across_chunks(self):
        text = "\n".join([
            "function a() {",
            "    return 1;",
            "}",
            "",
            "function b() {",
            "    return 2;",
            "}",
        ])
        chunks = list(chunk_code_by_boundaries(text, chunk_size=5, overlap=1, boundary_indexes_fn=brace_boundary_indexes))
        self.assertEqual(len(chunks), 2)
        self.assertIn("function a", chunks[0][0])
        self.assertNotIn("function b", chunks[0][0], "must not split function b's opening line into chunk 1")
        self.assertIn("function b", chunks[1][0])
        self.assertIn("return 2", chunks[1][0], "function b's body must stay whole in chunk 2")

    def test_falls_back_to_fixed_cut_when_no_boundary_in_window(self):
        # One giant, never-closing-until-the-end function -- no brace
        # boundary exists anywhere in the early lookback windows, so this
        # must behave exactly like chunk_lines() for those chunks rather
        # than searching forever or growing the chunk past chunk_size.
        lines = ["function a() {"] + [f"    line{i};" for i in range(100)] + ["}"]
        text = "\n".join(lines)
        chunks = list(chunk_code_by_boundaries(text, chunk_size=20, overlap=5, boundary_indexes_fn=brace_boundary_indexes))
        self.assertGreater(len(chunks), 1)
        first_start, first_end = chunks[0][1], chunks[0][2]
        self.assertEqual(
            first_end - first_start + 1, 20,
            "with no boundary in the lookback window, the cut must match the fixed chunk_size exactly",
        )

    def test_empty_text_yields_nothing(self):
        self.assertEqual(list(chunk_code_by_boundaries("", 50, 10, brace_boundary_indexes)), [])

    def test_covers_every_line_with_no_gaps_or_infinite_loop(self):
        lines = ["function a() {"] + [f"    line{i};" for i in range(100)] + ["}"]
        text = "\n".join(lines)
        chunks = list(chunk_code_by_boundaries(text, chunk_size=20, overlap=5, boundary_indexes_fn=brace_boundary_indexes))
        self.assertEqual(chunks[-1][2], len(lines), "the last chunk must reach the final line")


class FileRouting(unittest.TestCase):
    """chunk_file picks the chunker by extension and the type by doc-ness."""

    def _chunk_path(self, name, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_text(text)
            return list(chunk_file(p, name, CHUNK_SIZE, OVERLAP))

    def test_txt_comments_are_not_treated_as_headings(self):
        # Shaped like this repo's requirements.txt: the bug that started this.
        text = "\n".join([
            "# Core -- required for the Qdrant piece",
            "# install these even if you skip local-compress",
            "mcp[cli]",
            "mcp-server-qdrant",
            "# Local-compress piece only",
            "openai",
        ])
        chunks = self._chunk_path("requirements.txt", text)
        self.assertEqual(len(chunks), 1, "a short .txt is one line-chunk, not one per comment")
        content, meta = chunks[0]
        self.assertEqual(meta["type"], "doc", ".txt is still a doc, just line-chunked")
        self.assertIn("mcp[cli]", content)
        self.assertIn("openai", content)

    def test_single_line_chunks_are_gone(self):
        text = "\n".join(f"# comment line {i}" for i in range(12))
        chunks = self._chunk_path("requirements.txt", text)
        spans = [int(m["line_range"].split("-")[1]) - int(m["line_range"].split("-")[0]) + 1
                 for _, m in chunks]
        self.assertNotIn(1, spans, "12 comment lines previously became 12 one-line chunks")

    def test_rst_is_line_chunked(self):
        # .rst headings are underlines, so there was never anything for the
        # markdown chunker to find -- but '#' comments would still have split it.
        text = "\n".join(["Title", "=====", "", "# not an rst heading", "body"])
        chunks = self._chunk_path("guide.rst", text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][1]["type"], "doc")

    def test_markdown_still_gets_heading_chunking(self):
        text = "# A\nbody\n## B\nbody"
        chunks = self._chunk_path("README.md", text)
        self.assertEqual(len(chunks), 2, ".md must keep heading-based chunking")
        self.assertEqual(chunks[0][1]["type"], "doc")

    def test_code_is_line_chunked_and_typed_code(self):
        text = "\n".join(["#!/usr/bin/env python3", "# a comment", "x = 1"])
        chunks = self._chunk_path("script.py", text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][1]["type"], "code")

    def _chunk_path_with_params(self, name, text, chunk_size, overlap):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_text(text)
            return list(chunk_file(p, name, chunk_size, overlap))

    def test_cs_uses_boundary_aware_chunking_not_a_mid_function_split(self):
        # Issue #55: a brace-block language must not split a function body
        # across two chunks the way a pure fixed-line window would.
        text = "\n".join([
            "public class Foo {",
            "    public int A() {",
            "        return 1;",
            "    }",
            "",
            "    public int B() {",
            "        return 2;",
            "    }",
            "}",
        ])
        chunks = self._chunk_path_with_params("Foo.cs", text, chunk_size=5, overlap=1)
        self.assertGreater(len(chunks), 1, "small chunk_size must still produce multiple chunks")
        for content, _ in chunks:
            if "public int A()" in content:
                self.assertIn("return 1", content, "method A must not be split from its own body")
            if "public int B()" in content:
                self.assertIn("return 2", content, "method B must not be split from its own body")

    def test_ts_uses_boundary_aware_chunking_not_a_mid_function_split(self):
        text = "\n".join([
            "function a() {",
            "    return 1;",
            "}",
            "",
            "function b() {",
            "    return 2;",
            "}",
        ])
        chunks = self._chunk_path_with_params("app.ts", text, chunk_size=5, overlap=1)
        self.assertGreater(len(chunks), 1)
        for content, _ in chunks:
            if "function a" in content:
                self.assertIn("return 1", content)
            if "function b" in content:
                self.assertIn("return 2", content)

    def test_py_uses_boundary_aware_chunking_not_a_mid_def_split(self):
        text = "\n".join([
            "def foo():",
            "    return 1",
            "",
            "def bar():",
            "    return 2",
        ])
        chunks = self._chunk_path_with_params("app.py", text, chunk_size=3, overlap=1)
        self.assertGreater(len(chunks), 1)
        for content, meta in chunks:
            self.assertEqual(meta["type"], "code")
            if "def foo" in content:
                self.assertIn("return 1", content)
            if "def bar" in content:
                self.assertIn("return 2", content)

    def test_rb_unaffected_stays_on_fixed_line_chunking(self):
        # Ruby's 'end' keyword isn't a brace, and is out of the issue's
        # named scope -- must keep the original fixed-line behavior.
        text = "\n".join(["def foo", "  1", "end", "", "def bar", "  2", "end"])
        chunks = self._chunk_path_with_params("app.rb", text, chunk_size=50, overlap=10)
        self.assertEqual(len(chunks), 1, ".rb is short enough to stay one chunk either way")
        self.assertEqual(chunks[0][1]["type"], "code")

    def test_non_ascii_survives_regardless_of_platform_locale(self):
        """chunk_file must decode as UTF-8, not as the platform default.

        This passes either way on a UTF-8 machine, which is exactly why the bug
        (found in PR review) went unnoticed: with no explicit encoding, decoding
        follows the locale, so on a default Windows box (cp1252) these bytes
        decode to mojibake or -- with errors="ignore" -- vanish silently. The
        assertion is what fails there without the fix.

        The trap is that compute_file_hashes() reads BYTES, so the hash is
        identical across platforms while the decoded chunk text differs: an
        incremental sync sees no change and never repairs the bad content.
        """
        text = "# Café ☕\n\nnaïve résumé — em dash, ünlaut\n\n## Ω section\nbody"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "unicode.md"
            p.write_text(text, encoding="utf-8")
            chunks = list(chunk_file(p, "unicode.md", CHUNK_SIZE, OVERLAP))
        joined = " ".join(c for c, _ in chunks)
        for token in ("Café", "☕", "naïve", "résumé", "—", "ünlaut", "Ω"):
            self.assertIn(token, joined, f"{token!r} must survive decoding intact")


class IterFilesExcludeDirs(unittest.TestCase):
    """iter_files must exclude on the path RELATIVE TO repo_path, not the
    absolute path (issue #21) -- otherwise an ancestor directory that happens
    to share a name with EXCLUDE_DIRS (e.g. a repo checked out under
    ~/build/myrepo) silently excludes every file in the repo, with no error.
    """

    def test_ancestor_dir_matching_exclude_list_does_not_exclude_the_repo(self):
        with tempfile.TemporaryDirectory() as d:
            # "build" is in EXCLUDE_DIRS -- but only as an ancestor here, not
            # as a directory inside the repo itself.
            repo_path = Path(d) / "build" / "myrepo"
            repo_path.mkdir(parents=True)
            (repo_path / "main.py").write_text("x = 1")

            found = list(iter_files(repo_path, {".py"}))

            self.assertEqual(
                [p.name for p in found],
                ["main.py"],
                "a file inside the repo must be indexed even though the repo's "
                "own ancestor path contains a directory name in EXCLUDE_DIRS",
            )

    def test_in_repo_excluded_dir_is_still_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            repo_path = Path(d) / "myrepo"
            (repo_path / "node_modules" / "pkg").mkdir(parents=True)
            (repo_path / "node_modules" / "pkg" / "file.py").write_text("x = 1")
            (repo_path / "main.py").write_text("x = 1")

            found = list(iter_files(repo_path, {".py"}))

            self.assertEqual(
                [p.name for p in found],
                ["main.py"],
                "a genuinely in-repo excluded directory must still be skipped "
                "after checking the relative path instead of the absolute one",
            )


class Batched(unittest.TestCase):
    """batched() groups sync_repo's per-file deletes into fixed-size chunks
    (issue #75/#76) -- confirmed empirically that neither extreme works:
    one client.delete() call per file exhausts connections on a large repo,
    and one call covering every file can time out server-side.
    """

    def test_exact_multiple_splits_evenly(self):
        items = list(range(10))
        self.assertEqual(list(batched(items, 5)), [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    def test_remainder_yields_a_shorter_final_batch(self):
        items = list(range(7))
        self.assertEqual(list(batched(items, 3)), [[0, 1, 2], [3, 4, 5], [6]])

    def test_empty_list_yields_nothing(self):
        self.assertEqual(list(batched([], 5)), [])

    def test_batch_size_larger_than_list_yields_one_batch(self):
        items = [1, 2, 3]
        self.assertEqual(list(batched(items, 100)), [[1, 2, 3]])

    def test_batch_size_one_yields_one_item_per_batch(self):
        items = [1, 2, 3]
        self.assertEqual(list(batched(items, 1)), [[1], [2], [3]])


class BuildEntriesSkipsUnreadableFiles(unittest.TestCase):
    """Issue #27: a single unreadable file (broken symlink, permission
    error, ...) must not abort the whole indexing run -- build_entries logs
    it and skips it, with the rest of the repo still indexed.

    Uses a selective Path.read_text patch instead of a real broken symlink
    so the test is deterministic and portable (symlink creation needs extra
    privileges on Windows)."""

    def _patched_read_text(self, bad_name):
        real_read_text = Path.read_text

        def flaky(self_path, *args, **kwargs):
            if self_path.name == bad_name:
                raise OSError(f"simulated unreadable file: {bad_name}")
            return real_read_text(self_path, *args, **kwargs)

        return flaky

    def test_unreadable_file_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "good.py").write_text("x = 1")
            (repo / "bad.py").write_text("y = 2")

            with mock.patch.object(Path, "read_text", self._patched_read_text("bad.py")):
                entries, skipped = build_entries(repo, "code", CHUNK_SIZE, OVERLAP)

            self.assertEqual(
                [m["file_path"] for _, m in entries], ["good.py"],
                "the readable file must still be indexed despite the other one failing",
            )
            self.assertEqual([rel for rel, _ in skipped], ["bad.py"])
            self.assertIn("simulated unreadable file", skipped[0][1])

    def test_no_skipped_files_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "good.py").write_text("x = 1")
            entries, skipped = build_entries(repo, "code", CHUNK_SIZE, OVERLAP)
            self.assertEqual(len(entries), 1)
            self.assertEqual(skipped, [])


class ComputeFileHashesSkipsUnreadableFiles(unittest.TestCase):
    """Same guarantee as BuildEntriesSkipsUnreadableFiles, for the hashing
    path sync_repo relies on to detect changed files (issue #27)."""

    def _patched_read_bytes(self, bad_name):
        real_read_bytes = Path.read_bytes

        def flaky(self_path, *args, **kwargs):
            if self_path.name == bad_name:
                raise OSError(f"simulated unreadable file: {bad_name}")
            return real_read_bytes(self_path, *args, **kwargs)

        return flaky

    def test_unreadable_file_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "good.py").write_text("x = 1")
            (repo / "bad.py").write_text("y = 2")

            with mock.patch.object(Path, "read_bytes", self._patched_read_bytes("bad.py")):
                hashes, skipped = compute_file_hashes(repo, "code")

            self.assertEqual(list(hashes.keys()), ["good.py"])
            self.assertEqual([rel for rel, _ in skipped], ["bad.py"])

    def test_no_skipped_files_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "good.py").write_text("x = 1")
            hashes, skipped = compute_file_hashes(repo, "code")
            self.assertEqual(list(hashes.keys()), ["good.py"])
            self.assertEqual(skipped, [])


class FilesRemovedSinceLastSync(unittest.TestCase):
    """Issue #27: a file that failed to hash must not be reported as
    'removed' -- sync_repo would otherwise delete its still-good existing
    embeddings over what may just be a transient read error. Only a file
    genuinely absent from BOTH current_hashes and skipped_paths (i.e.
    iter_files didn't yield it at all -- deleted, renamed, or filtered out)
    counts as removed."""

    def test_file_absent_from_current_hashes_and_not_skipped_is_removed(self):
        manifest = {"gone.py": "oldhash", "kept.py": "stillhash"}
        current_hashes = {"kept.py": "stillhash"}
        self.assertEqual(
            files_removed_since_last_sync(manifest, current_hashes, []),
            ["gone.py"],
        )

    def test_skipped_file_is_not_reported_as_removed(self):
        manifest = {"flaky.py": "oldhash", "kept.py": "stillhash"}
        current_hashes = {"kept.py": "stillhash"}  # flaky.py failed to hash
        self.assertEqual(
            files_removed_since_last_sync(manifest, current_hashes, ["flaky.py"]),
            [],
            "a file that merely failed to hash must not be treated as removed",
        )

    def test_no_manifest_entries_yields_nothing_removed(self):
        self.assertEqual(files_removed_since_last_sync({}, {"a.py": "h"}, []), [])


class EnsurePersistentFastembedCache(unittest.TestCase):
    """ensure_persistent_fastembed_cache() must default FASTEMBED_CACHE_PATH
    to a persistent, toolkit-owned directory when unset (issue #77 -- the
    fastembed default is the OS temp dir, wiped on reboot/cleanup), but never
    override a value the caller already set via .mcp.json or the shell.
    """

    def test_defaults_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FASTEMBED_CACHE_PATH", None)
            ensure_persistent_fastembed_cache()
            expected = str(Path.home() / ".claude" / "claude-runway" / "fastembed-cache")
            self.assertEqual(os.environ["FASTEMBED_CACHE_PATH"], expected)

    def test_respects_existing_override(self):
        with mock.patch.dict(os.environ, {"FASTEMBED_CACHE_PATH": "/custom/cache/dir"}, clear=False):
            ensure_persistent_fastembed_cache()
            self.assertEqual(os.environ["FASTEMBED_CACHE_PATH"], "/custom/cache/dir")


class ValidateChunkParams(unittest.TestCase):
    """Issue #26: overlap >= chunk_size collapses chunk_lines()'s step to 1,
    silently multiplying chunk count. validate_chunk_params is the shared
    check every CLI/tool boundary calls before chunking anything."""

    def test_valid_params_return_none(self):
        self.assertIsNone(validate_chunk_params(CHUNK_SIZE, OVERLAP))

    def test_overlap_equal_to_chunk_size_is_rejected(self):
        err = validate_chunk_params(50, 50)
        self.assertIsNotNone(err)
        self.assertIn("50", err)

    def test_overlap_greater_than_chunk_size_is_rejected(self):
        err = validate_chunk_params(10, 20)
        self.assertIsNotNone(err)

    def test_chunk_size_below_one_is_rejected(self):
        for bad in (0, -5):
            with self.subTest(chunk_size=bad):
                err = validate_chunk_params(bad, 0)
                self.assertIsNotNone(err)

    def test_negative_overlap_is_rejected(self):
        err = validate_chunk_params(50, -1)
        self.assertIsNotNone(err)

    def test_zero_overlap_is_valid(self):
        # overlap=0 means "no overlap," a legitimate, common configuration --
        # must not be confused with the negative/too-large cases above.
        self.assertIsNone(validate_chunk_params(50, 0))


class ChunkLinesOverlapExplosion(unittest.TestCase):
    """Demonstrates the actual defect mechanism directly (issue #26), not
    just that validate_chunk_params rejects the bad input -- pins the ROOT
    CAUSE so a future refactor of chunk_lines() can't silently reintroduce
    this while validate_chunk_params stays correct in isolation."""

    def test_overlap_equal_to_chunk_size_yields_near_one_chunk_per_line(self):
        text = "\n".join(f"line {i}" for i in range(200))
        chunks = list(chunk_lines(text, chunk_size=10, overlap=10))
        # With a VALID overlap of, say, 5 (step = chunk_size - overlap = 5),
        # 200 lines would yield ~40 chunks. Here overlap=10 == chunk_size=10,
        # so step collapses to max(10 - 10, 1) = 1 -- one chunk starting at
        # nearly every line instead, bounded by len(lines), not by any sane
        # chunk math.
        self.assertGreater(len(chunks), 150)

    def test_healthy_overlap_yields_the_expected_chunk_count(self):
        # Regression guard the other direction: a VALID overlap must keep
        # producing the normal, small chunk count -- confirms the explosion
        # above is specific to the bad-overlap case, not a general property
        # of chunk_lines() itself.
        text = "\n".join(f"line {i}" for i in range(200))
        chunks = list(chunk_lines(text, chunk_size=10, overlap=5))
        self.assertLess(len(chunks), 50)


class McpToolsValidateChunkParamsBeforeAnyWork(unittest.TestCase):
    """Issue #26: a shared validator only closes the bug if every boundary
    actually calls it. Imports ingest_mcp_server.py directly and calls each
    tool with an invalid chunk_lines/overlap pair -- no live Qdrant/FastEmbed
    needed, since the validation check is the very first thing each function
    does, before any network call or even a filesystem check."""

    @classmethod
    def setUpClass(cls):
        import sys as _sys
        tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
        _sys.path.insert(0, tools_dir)
        # Minimal env so module-level code (DEFAULT_* constants) doesn't
        # explode on import -- these tools already default gracefully to
        # None/empty when unset, this just avoids relying on real infra.
        os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
        os.environ.setdefault("COLLECTION_NAME", "test-collection")
        os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        import ingest_mcp_server as _ims
        cls.ims = _ims

    def test_index_repo_rejects_bad_overlap_before_touching_qdrant(self):
        import asyncio
        result = asyncio.run(self.ims.index_repo(repo_path=".", chunk_lines=10, overlap=10))
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("overlap", result)

    def test_sync_repo_rejects_bad_overlap_before_touching_qdrant(self):
        import asyncio
        result = asyncio.run(self.ims.sync_repo(repo_path=".", chunk_lines=10, overlap=15))
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("overlap", result)

    def test_preview_index_rejects_bad_overlap_before_touching_qdrant(self):
        result = self.ims.preview_index(repo_path=".", chunk_lines=10, overlap=10)
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("overlap", result)

    def test_index_repo_still_rejects_bad_repo_path_when_chunk_params_are_fine(self):
        # Regression guard: confirms the new check is additive, not a
        # replacement for the existing repo.is_dir() validation.
        import asyncio
        result = asyncio.run(self.ims.index_repo(repo_path="/definitely/not/a/real/path", chunk_lines=50, overlap=10))
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("is not a directory", result)


class IndexRepoDocumentsTheOrphanedChunkLimitation(unittest.TestCase):
    """Issue #31: index_repo(reset=False) never detects/removes chunks for
    files deleted from the repo, unlike sync_repo. Decided (see .plans/31-
    plan.md) to document this loudly rather than change the behavior --
    sync_repo already exists specifically for ongoing/incremental
    re-indexing and already tracks a manifest to handle deletions
    correctly; index_repo is the bulk/first-time/full-rebuild tool. This
    test is a regression guard for the warning itself: a future docstring
    edit can't silently drop it without failing a test, even though
    nothing here checks runtime behavior (there isn't any behavior change
    to check)."""

    @classmethod
    def setUpClass(cls):
        tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
        sys.path.insert(0, tools_dir)
        os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
        os.environ.setdefault("COLLECTION_NAME", "test-collection")
        os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        import ingest_mcp_server as _ims
        cls.ims = _ims

    def test_docstring_warns_about_deleted_files_and_points_at_sync_repo(self):
        doc = self.ims.index_repo.__doc__
        self.assertIsNotNone(doc)
        self.assertIn("DELETED", doc)
        self.assertIn("sync_repo", doc)


class SyncRepoPreservesEmbeddingsOnChunkAndHashFailures(unittest.TestCase):
    """PR #90 review (Copilot), issue #27 follow-up. Two real bugs the
    original try/except-and-skip fix introduced, both now fixed:

    1. sync_repo deleted a changed file's OLD embeddings BEFORE attempting to
       chunk its new content. A file that hashed fine but then failed to
       chunk (the same hash/chunk race compute_file_hashes is exposed to)
       ended up with its embeddings deleted and nothing to replace them --
       zero searchable content until a later successful sync, unlike the
       preservation hash-skipped files already got.
    2. _save_manifest overwrites the WHOLE manifest with current_hashes. A
       hash- or chunk-skipped file was simply absent from current_hashes, so
       its manifest entry vanished outright -- if that file was later
       genuinely deleted from disk, no future sync could ever detect it as
       "removed" (it no longer appears in the manifest to compare against),
       so its now-actually-stale embeddings would never get cleaned up.

    Mocks the Qdrant/FastEmbed layer (no live services needed) so this
    exercises sync_repo's actual chunk/delete/manifest ordering end to end,
    not just the pure helpers in isolation."""

    @classmethod
    def setUpClass(cls):
        tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
        sys.path.insert(0, tools_dir)
        os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
        os.environ.setdefault("COLLECTION_NAME", "test-collection")
        os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        import ingest_mcp_server as _ims
        cls.ims = _ims

    @staticmethod
    def _sha(content):
        return hashlib.sha256(content.encode()).hexdigest()

    def test_chunk_failure_preserves_embeddings_and_manifest_entry(self):
        import asyncio

        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "unchanged.py").write_text("same = 1")
            (repo / "real_change.py").write_text("new_content = 2")
            (repo / "flaky_hash.py").write_text("hash_will_fail = 3")
            (repo / "flaky_chunk.py").write_text("chunk_will_fail = 4")

            # Old manifest: unchanged.py's hash matches its real content
            # (so it stays "unchanged"); the other three have stale hashes
            # so they'd normally be classified "changed" this sync.
            manifest = {
                "unchanged.py": self._sha("same = 1"),
                "real_change.py": "stale-old-hash",
                "flaky_hash.py": "old-hash-for-flaky-hash",
                "flaky_chunk.py": "old-hash-for-flaky-chunk",
            }
            self.ims._save_manifest(repo, manifest)

            real_read_bytes = Path.read_bytes
            real_read_text = Path.read_text

            def flaky_read_bytes(self_path, *a, **kw):
                if self_path.name == "flaky_hash.py":
                    raise OSError("simulated hash failure")
                return real_read_bytes(self_path, *a, **kw)

            def flaky_read_text(self_path, *a, **kw):
                if self_path.name == "flaky_chunk.py":
                    raise OSError("simulated chunk failure")
                return real_read_text(self_path, *a, **kw)

            fake_client = MagicMock()
            fake_client.collection_exists.return_value = True

            with mock.patch.object(Path, "read_bytes", flaky_read_bytes), \
                 mock.patch.object(Path, "read_text", flaky_read_text), \
                 mock.patch.object(self.ims, "QdrantClient", return_value=fake_client), \
                 mock.patch.object(self.ims, "FastEmbedProvider", return_value=MagicMock()), \
                 mock.patch.object(self.ims, "QdrantConnector", return_value=MagicMock()), \
                 mock.patch.object(self.ims, "store_batch", new=AsyncMock(return_value=None)):
                result = asyncio.run(self.ims.sync_repo(
                    repo_path=str(repo),
                    collection="test-sync-collection",
                    qdrant_url="http://fake-qdrant:6333",
                    scope="code",
                    # Scoped to .py only so the manifest file itself
                    # (.qdrant_index_manifest.json -- .json is in
                    # CODE_EXTENSIONS, and this repo has no .gitignore to
                    # exclude it) doesn't get self-indexed as a "changed"
                    # file, which would throw off this test's exact counts.
                    # Pre-existing, unrelated to issue #27/this PR.
                    include_extensions=".py",
                ))

            # Only real_change.py actually produced replacement chunks --
            # flaky_chunk.py must not be double-counted as both "re-indexed"
            # and "skipped" (finding 3).
            self.assertIn("1 file(s) re-indexed", result)
            self.assertIn("1 file(s) unchanged", result)
            self.assertIn("0 file(s) removed", result)
            self.assertIn("Skipped 2 unreadable file(s)", result)

            # flaky_chunk.py's OLD embeddings must survive -- it must not
            # appear in any delete call despite being in `changed` (finding 1).
            deleted_paths = set()
            for call in fake_client.delete.call_args_list:
                filt = call.kwargs["points_selector"]
                for cond in filt.should:
                    deleted_paths.add(cond.match.value)
            self.assertIn("real_change.py", deleted_paths)
            self.assertNotIn("flaky_chunk.py", deleted_paths)
            self.assertNotIn("flaky_hash.py", deleted_paths)

            # The saved manifest must retain flaky_hash.py's and
            # flaky_chunk.py's PRIOR hashes -- not dropped (which would
            # permanently orphan their embeddings from removal-detection on
            # a later genuine delete), not overwritten with a hash that
            # doesn't match what's actually still embedded (finding 2).
            saved_manifest = self.ims._load_manifest(repo)
            self.assertEqual(saved_manifest["flaky_hash.py"], "old-hash-for-flaky-hash")
            self.assertEqual(saved_manifest["flaky_chunk.py"], "old-hash-for-flaky-chunk")
            self.assertEqual(saved_manifest["unchanged.py"], self._sha("same = 1"))
            self.assertEqual(saved_manifest["real_change.py"], self._sha("new_content = 2"))


class ManifestSelfIndex(unittest.TestCase):
    """Issue #91: .qdrant_index_manifest.json has a .json extension that
    matches CODE_EXTENSIONS, so iter_files yields it when the repo's
    .gitignore doesn't list it. The manifest content (a flat map of
    file->sha256) gets embedded as source and perpetually looks "changed"
    to sync_repo (its own hash changes on every _save_manifest write).

    Fix: extra_exclude_files parameter on iter_files/build_entries/
    compute_file_hashes; ingest_mcp_server passes {MANIFEST_FILENAME}.
    """

    MANIFEST = ".qdrant_index_manifest.json"

    def _make_repo(self, tmp):
        """Returns a repo Path with main.py and the manifest file."""
        repo = Path(tmp)
        (repo / "main.py").write_text("x = 1")
        (repo / self.MANIFEST).write_text('{"main.py": "somehash"}')
        return repo

    # --- Reproduce the bug (no exclusion) -----------------------------------

    def test_iter_files_without_exclusion_yields_manifest(self):
        """Reproduces the exact script from the issue body: without
        extra_exclude_files, iter_files yields the manifest alongside source."""
        with tempfile.TemporaryDirectory() as d:
            repo = self._make_repo(d)
            names = {f.name for f in iter_files(repo, {".py", ".json"})}
            # Both must be present for the bug to exist -- this is the
            # pre-fix behavior we're pinning, so the fix is meaningful.
            self.assertIn("main.py", names)
            self.assertIn(self.MANIFEST, names, (
                "Without extra_exclude_files the manifest is yielded -- "
                "this confirms the bug scenario from issue #91 is real"
            ))

    # --- Verify the fix -----------------------------------------------------

    def test_iter_files_excludes_manifest_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            repo = self._make_repo(d)
            names = {f.name for f in iter_files(
                repo, {".py", ".json"},
                extra_exclude_files={self.MANIFEST},
            )}
            self.assertIn("main.py", names)
            self.assertNotIn(self.MANIFEST, names)

    def test_iter_files_extra_exclude_files_does_not_drop_unrelated_json(self):
        """Excluding just the manifest must not exclude other .json files."""
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "config.json").write_text('{"key": "val"}')
            (repo / self.MANIFEST).write_text('{}')
            names = {f.name for f in iter_files(
                repo, {".json"},
                extra_exclude_files={self.MANIFEST},
            )}
            self.assertIn("config.json", names)
            self.assertNotIn(self.MANIFEST, names)

    def test_build_entries_excludes_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            repo = self._make_repo(d)
            entries, _ = build_entries(
                repo, "code", CHUNK_SIZE, OVERLAP,
                extra_exclude_files={self.MANIFEST},
            )
            indexed = {m["file_path"] for _, m in entries}
            self.assertIn("main.py", indexed)
            self.assertNotIn(self.MANIFEST, indexed)

    def test_compute_file_hashes_excludes_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            repo = self._make_repo(d)
            hashes, _ = compute_file_hashes(
                repo, "code",
                extra_exclude_files={self.MANIFEST},
            )
            self.assertIn("main.py", hashes)
            self.assertNotIn(self.MANIFEST, hashes)

    def test_extra_exclude_files_none_does_not_affect_normal_behavior(self):
        """Passing extra_exclude_files=None (the default) leaves behavior
        unchanged -- no accidental breakage for callers that don't pass it."""
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "main.py").write_text("x = 1")
            names = {f.name for f in iter_files(repo, {".py"}, extra_exclude_files=None)}
            self.assertEqual(names, {"main.py"})

    # --- Integration: ingest_mcp_server passes MANIFEST_FILENAME -----------

    def test_ingest_mcp_server_constant_matches_expected_filename(self):
        """Guards the filename constant -- if MANIFEST_FILENAME ever changes,
        this fails loudly rather than silently re-exposing the bug (issue #91)."""
        tools_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"
        )
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
        os.environ.setdefault("COLLECTION_NAME", "test-collection")
        os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        import ingest_mcp_server as _ims
        self.assertEqual(
            _ims.MANIFEST_FILENAME,
            ".qdrant_index_manifest.json",
            "MANIFEST_FILENAME must still be '.qdrant_index_manifest.json' -- "
            "update extra_exclude_files call sites if this ever intentionally changes",
        )


class IterEntries(unittest.TestCase):
    """iter_entries() is a generator variant of build_entries() (issue #57) --
    same content, but yields chunks one at a time rather than accumulating the
    whole repo into memory upfront. Errors are yielded as (None, (rel, err))
    sentinels instead of accumulated into a separate list."""

    def test_yields_same_content_as_build_entries(self):
        # For a well-formed repo, iter_entries must produce exactly the same
        # (content, metadata) tuples that build_entries returns in its entries
        # list -- different ordering is fine (iter_files' rglob order) but
        # content and metadata must be identical once sorted by file_path.
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "a.py").write_text("x = 1\ny = 2\n")
            (repo / "b.py").write_text("z = 3\n")

            build_result, build_skipped = build_entries(repo, "code", CHUNK_SIZE, OVERLAP)
            self.assertEqual(build_skipped, [])

            iter_result = []
            iter_skipped = []
            for content, metadata_or_err in iter_entries(repo, "code", CHUNK_SIZE, OVERLAP):
                if content is None:
                    iter_skipped.append(metadata_or_err)
                else:
                    iter_result.append((content, metadata_or_err))

            self.assertEqual(iter_skipped, [])
            # Sort both by file_path so ordering differences don't matter.
            self.assertEqual(
                sorted(iter_result, key=lambda x: x[1]["file_path"]),
                sorted(build_result, key=lambda x: x[1]["file_path"]),
                "iter_entries must yield the same chunks as build_entries",
            )

    def test_unreadable_file_yields_sentinel_not_exception(self):
        # A file that raises OSError while being chunked must yield a
        # (None, (rel, err)) sentinel -- the caller decides what to do with
        # it, and the rest of the repo continues to be yielded normally.
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "good.py").write_text("x = 1")
            (repo / "bad.py").write_text("y = 2")

            real_read_text = Path.read_text

            def flaky(self_path, *args, **kwargs):
                if self_path.name == "bad.py":
                    raise OSError("simulated unreadable file")
                return real_read_text(self_path, *args, **kwargs)

            good_chunks = []
            errors = []
            with mock.patch.object(Path, "read_text", flaky):
                for content, metadata_or_err in iter_entries(repo, "code", CHUNK_SIZE, OVERLAP):
                    if content is None:
                        errors.append(metadata_or_err)
                    else:
                        good_chunks.append((content, metadata_or_err))

            self.assertEqual(len(errors), 1, "exactly one sentinel for the bad file")
            self.assertEqual(errors[0][0], "bad.py")
            self.assertIn("simulated unreadable file", errors[0][1])
            self.assertEqual(
                [m["file_path"] for _, m in good_chunks],
                ["good.py"],
                "the readable file must still be yielded despite the other one failing",
            )

    def test_empty_repo_yields_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            result = list(iter_entries(Path(d), "code", CHUNK_SIZE, OVERLAP))
            self.assertEqual(result, [], "no files to chunk, no output at all")

    def test_is_a_generator_not_a_list(self):
        # Confirms iter_entries() returns a generator object rather than a
        # pre-built list -- the whole point of issue #57 is that chunks flow
        # out incrementally, not after all files are processed.
        import types
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "a.py").write_text("x = 1")
            result = iter_entries(repo, "code", CHUNK_SIZE, OVERLAP)
            self.assertIsInstance(result, types.GeneratorType)


class PhpHashCommentScanning(unittest.TestCase):
    """PHP's '#' line comment must be treated as a comment when ext='.php'
    and must NOT be treated as one for other brace-block languages where '#'
    has a different meaning (issue #138).

    The bug: _strip_c_style_line had no language-specific handling for '#',
    so a PHP comment like '# }' was parsed as a real closing brace by
    brace_boundary_indexes, popping the function's frame and fabricating
    spurious chunk boundaries on every subsequent line of that function body.
    """

    def test_strip_c_style_line_blanks_hash_for_php(self):
        # A PHP '# }' line: the '#' and everything after it must be blanked.
        result, bc, ts, vs = _strip_c_style_line("# }", False, ext=".php")
        self.assertEqual(result, "   ", "the entire '# }' must be blanked for PHP")
        self.assertFalse(bc)

    def test_strip_c_style_line_hash_not_a_comment_for_cs(self):
        # In C#, '#' is a preprocessor directive start, not a line comment --
        # the brace after it is real and must NOT be blanked.
        result, bc, ts, vs = _strip_c_style_line("# }", False, ext=".cs")
        self.assertIn("}", result, "the '}' must survive for C# where '#' is a preprocessor token")

    def test_strip_c_style_line_hash_not_a_comment_for_rs(self):
        # In Rust, '#' introduces an attribute (#[derive(...)]), not a comment.
        result, bc, ts, vs = _strip_c_style_line("#[derive(Debug)]", False, ext=".rs")
        self.assertIn("[", result, "Rust '#[...]' is an attribute, not a comment -- must not be blanked")

    def test_strip_c_style_line_hash_not_a_comment_for_c(self):
        # In C, '#' is a preprocessor directive (#include, #if, ...).
        result, bc, ts, vs = _strip_c_style_line("#include <stdio.h>", False, ext=".c")
        self.assertIn("i", result, "C '#include' preprocessor directive must not be blanked")

    def test_brace_boundary_indexes_php_hash_comment_does_not_fabricate_boundary(self):
        # The exact scenario from the issue: a PHP function body contains a
        # '# }' comment. Without the fix, that '}' pops the function's frame,
        # making every subsequent line appear to be at file-level (a spurious
        # "safe" boundary). With the fix, the '#' starts a comment, the '}'
        # is blanked, and the frame is never popped -- the boundary appears
        # only at the real closing '}' on the last line.
        lines = [
            "function foo() {",   # 0 -- opens function frame
            "    # }",             # 1 -- PHP comment; the '}' must NOT close the frame
            "    $x = 1;",         # 2 -- still inside function, not a boundary
            "    $y = 2;",         # 3 -- still inside function, not a boundary
            "}",                    # 4 -- real closing brace, depth 0 -- boundary
        ]
        boundaries = brace_boundary_indexes(lines, ext=".php")
        self.assertNotIn(1, boundaries, "line 1 is a PHP comment line, not a boundary")
        self.assertNotIn(2, boundaries, "line 2 is still inside the function body")
        self.assertNotIn(3, boundaries, "line 3 is still inside the function body")
        self.assertIn(4, boundaries, "the real closing brace must still be a boundary")

    def test_brace_boundary_indexes_php_hash_with_two_functions(self):
        # Two sibling PHP functions; the first contains a '# }' comment.
        # The boundary between them must be found at the right place.
        lines = [
            "function foo() {",   # 0
            "    # this comment has a brace: }",  # 1 -- must not close foo
            "    return 1;",       # 2
            "}",                    # 3 -- closes foo, boundary
            "",                     # 4 -- boundary
            "function bar() {",   # 5
            "    return 2;",       # 6
            "}",                    # 7 -- closes bar, boundary
        ]
        boundaries = brace_boundary_indexes(lines, ext=".php")
        self.assertIn(3, boundaries, "must find the boundary between foo and bar")
        self.assertIn(4, boundaries)
        self.assertIn(7, boundaries)
        self.assertNotIn(1, boundaries, "the comment line with a stray '}' must not be a boundary")
        self.assertNotIn(2, boundaries)

    def test_php8_attribute_hash_bracket_is_not_treated_as_a_comment(self):
        # PHP 8 attributes begin with '#[' (e.g. '#[Route("/home")]').
        # The '#[' token must NOT be treated as a comment start, so the
        # attribute line's content (including any braces in the attribute's
        # arguments) is NOT blanked, and the function that follows is scanned
        # correctly.
        result, bc, ts, vs = _strip_c_style_line('#[Route("/home")]', False, ext=".php")
        self.assertNotEqual(result.strip(), "", "PHP 8 '#[...]' attribute must not be blanked as a comment")
        self.assertIn("[", result, "the '[' inside '#[...]' must survive stripping")

    def test_brace_boundary_indexes_php8_attribute_does_not_fabricate_boundary(self):
        # A PHP 8 attribute on its own line precedes the function it decorates.
        # The attribute line must not be treated as a comment (blanking the
        # whole line including the following '['), and the function's opening
        # brace on line 1 must still be counted -- so lines 2 and 3 (inside
        # the function body) are NOT boundaries, and line 4 (the real close) IS.
        lines = [
            '#[Route("/home")]',  # 0 -- PHP 8 attribute, NOT a comment
            'function handle() {',  # 1 -- opens function
            '    $x = 1;',           # 2 -- inside body, not a boundary
            '    return $x;',        # 3 -- inside body, not a boundary
            '}',                      # 4 -- real close, boundary
        ]
        boundaries = brace_boundary_indexes(lines, ext=".php")
        self.assertNotIn(2, boundaries, "line 2 is inside the function body")
        self.assertNotIn(3, boundaries, "line 3 is inside the function body")
        self.assertIn(4, boundaries, "the real closing brace must be a boundary")

    def test_chunk_file_php_hash_comment_does_not_split_function(self):
        # End-to-end via chunk_file: a PHP function with a '# }' comment must
        # not get split at that comment line when the chunk_size is small
        # enough to force a boundary decision inside the function.
        text = "\n".join([
            "<?php",
            "function foo() {",
            "    # this comment has a closing brace: }",
            "    $x = 1;",
            "    $y = 2;",
            "    $z = 3;",
            "}",
            "",
            "function bar() {",
            "    return 2;",
            "}",
        ])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "example.php"
            p.write_text(text, encoding="utf-8")
            chunks = list(chunk_file(p, "example.php", chunk_size=6, overlap=1))
        # foo()'s body must travel with foo()'s opening line, never split at
        # the comment line mid-body.
        for content, _ in chunks:
            if "function foo" in content:
                self.assertIn("$x = 1", content, "foo's body must stay with its header")
                self.assertIn("$y = 2", content)


if __name__ == "__main__":
    unittest.main()
