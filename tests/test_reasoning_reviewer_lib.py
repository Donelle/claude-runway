#!/usr/bin/env python3
"""Tests for libs/reasoning_reviewer_lib.py.

Stdlib-only (unittest), no network: the model client is a stub and LM Studio's
REST API is `tests/fake_lmstudio.py`.

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from codebase_reviewer_config_lib import Reviewer, ReviewersConfig  # noqa: E402
from fake_lmstudio import FakeLmStudio  # noqa: E402
from lmstudio_model_loader_lib import ModelLoader, ModelNotLoadedError  # noqa: E402
from reasoning_reviewer_lib import (  # noqa: E402
    GAP_MARKER,
    MAX_CHUNKS,
    MAX_READ_BYTES,
    MIN_CHUNK_CHARS,
    DiffSection,
    Finding,
    ReviewError,
    applies_to,
    ReasoningReview,
    _read_text,
    _unquote_git_path,
    build_system_prompt,
    chunk_chars_for,
    chunk_diff,
    chunk_file_diff,
    collect_findings,
    format_review,
    instruction_files_for,
    parse_findings,
    review_skills,
    run_reasoning_review,
    split_diff,
)

BASE = "http://localhost:1234/v1"


def rv(name="quoting", model="model-a", roles=("reasoning",), *, prompt="look at quoting", prompt_file="",
       active=True, on_demand=False, context=None) -> Reviewer:
    return Reviewer(
        name=name,
        model=model,
        prompt=prompt,
        active=active,
        roles=tuple(r for r in ("reasoning", "verifier", "classifier") if r in roles),
        load_on_demand=on_demand,
        context_length=context,
        prompt_file=prompt_file,
    )


def cfg(*reviewers) -> ReviewersConfig:
    return ReviewersConfig(rubric=(), reviewers=tuple(reviewers))


def file_diff(path, hunks, header_extra="") -> str:
    """A `git diff`-shaped section for one file; each hunk is a list of body lines."""
    out = [f"diff --git a/{path} b/{path}\n", f"index 111..222 100644{header_extra}\n", f"--- a/{path}\n", f"+++ b/{path}\n"]
    line = 1
    for hunk in hunks:
        out.append(f"@@ -{line},{len(hunk)} +{line},{len(hunk)} @@\n")
        out.extend(f"{text}\n" for text in hunk)
        line += len(hunk) + 10
    return "".join(out)


def reply(*findings) -> str:
    return json.dumps({"findings": list(findings)})


def finding(file="a.py", start=3, end=3, cls="quoting", claim="a claim", failing="x y"):
    return {"file": file, "line_start": start, "line_end": end, "class": cls, "claim": claim, "failing_input": failing}


class SplitDiff(unittest.TestCase):
    def test_one_section_per_file_in_order(self):
        diff = file_diff("a.py", [["+one"]]) + file_diff("dir/b.md", [["-two", "+three"]])
        sections = split_diff(diff)
        self.assertEqual([s.path for s in sections], ["a.py", "dir/b.md"])
        self.assertEqual("".join(s.text for s in sections), diff)

    def test_an_empty_diff_is_no_sections(self):
        self.assertEqual(split_diff(""), [])

    def test_a_deleted_file_is_named_by_its_old_path(self):
        text = "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
        self.assertEqual(split_diff(text)[0].path, "gone.py")

    def test_a_binary_file_is_named_from_its_header(self):
        text = "diff --git a/img.png b/img.png\nBinary files a/img.png and b/img.png differ\n"
        self.assertEqual(split_diff(text)[0].path, "img.png")


class Chunking(unittest.TestCase):
    MODIFIED = (
        "diff --git a/m.py b/m.py\nindex 1..2 100644\n--- a/m.py\n+++ b/m.py\n"
        "@@ -10,5 +10,4 @@ def f():\n keep one\n-old line\n-old two\n+new line\n keep two\n keep three\n"
    )

    def test_a_file_that_fits_is_one_chunk_starting_with_its_file_line(self):
        section = split_diff(file_diff("a.py", [["+x"]]))[0]
        self.assertEqual(chunk_file_diff(section, 10_000), ["File: a.py\n    1 + x\n"])

    def test_a_file_is_shown_as_its_new_version_with_numbers_marks_and_a_removal_marker(self):
        chunk = chunk_file_diff(split_diff(self.MODIFIED)[0], 10_000)[0]
        self.assertEqual(
            chunk,
            "File: m.py\n"
            "   10   keep one\n"
            "      ~ 2 line(s) removed here\n"
            "   11 + new line\n"
            "   12   keep two\n"
            "   13   keep three\n",
        )

    def test_removed_code_is_never_shown_so_it_cannot_be_read_as_current(self):
        chunk = chunk_file_diff(split_diff(self.MODIFIED)[0], 10_000)[0]
        self.assertNotIn("old line", chunk)
        self.assertNotIn("old two", chunk)

    def test_a_trailing_removal_and_the_no_newline_marker_are_handled(self):
        text = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1,3 +1,2 @@\n keep\n-gone\n\\ No newline at end of file\n+tail\n"
        chunk = chunk_file_diff(split_diff(text)[0], 10_000)[0]
        self.assertEqual(chunk, "File: m.py\n    1   keep\n      ~ 1 line(s) removed here\n    2 + tail\n")

    def test_a_code_line_holding_a_form_feed_stays_one_numbered_line(self):
        text = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1,2 @@\n+a\x0cb\n+c\n"
        chunk = chunk_file_diff(split_diff(text)[0], 10_000)[0]
        self.assertEqual(chunk, "File: m.py\n    1 + a\x0cb\n    2 + c\n")

    def test_a_big_file_is_cut_at_hunk_boundaries_and_every_chunk_starts_with_its_file_line(self):
        hunks = [[f"+line {h}-{i}" for i in range(20)] for h in range(6)]
        section = split_diff(file_diff("big.py", hunks))[0]
        chunks = chunk_file_diff(section, 900)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.startswith("File: big.py\n"), chunk[:40])
        # Nothing lost, nothing repeated, and each line keeps its absolute number in the file
        # (hunk h starts at line 1 + 30h in these test diffs).
        for h in range(6):
            for i in range(20):
                shown = f"{1 + 30 * h + i:>5} + line {h}-{i}\n"
                self.assertEqual(sum(shown in c for c in chunks), 1, (h, i))

    def test_hunks_that_are_not_adjacent_are_separated_by_a_gap_marker(self):
        section = split_diff(file_diff("big.py", [["+a"], ["+b"], ["+c"]]))[0]
        (chunk,) = chunk_file_diff(section, 10_000)
        self.assertEqual(chunk.count(GAP_MARKER), 2)
        self.assertLess(chunk.index("+ a"), chunk.index(GAP_MARKER))

    def test_a_single_hunk_bigger_than_the_limit_is_cut_at_lines_and_continuations_keep_absolute_numbers(self):
        section = split_diff(file_diff("big.py", [[f"+line number {i:03d}" for i in range(200)]]))[0]
        chunks = chunk_file_diff(section, 1500)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.endswith("\n"))
            for line in chunk.splitlines():
                self.assertTrue(line.startswith("File: ") or re.match(r"^ *\d+ [+ ] ", line), line)
        # The last chunk's lines are at their real positions, not restarted from 1.
        joined = "".join(chunks)
        for i in range(200):
            self.assertEqual(joined.count(f"{1 + i:>5} + line number {i:03d}\n"), 1)
        self.assertIn("  200 + line number 199\n", chunks[-1])

    def test_one_line_longer_than_the_limit_is_the_only_thing_cut_mid_line(self):
        section = DiffSection("a.py", "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+" + "x" * 5000 + "\n")
        chunks = chunk_file_diff(section, 1000)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks).count("x"), 5000)

    def test_the_fragments_of_an_over_long_line_each_repeat_its_number_with_a_continuation_mark(self):
        long_line = "y" * 5000
        text = f"diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -41,0 +41,2 @@\n+first\n+{long_line}\n"
        chunks = chunk_file_diff(split_diff(text)[0], 1000)
        self.assertGreater(len(chunks), 2)
        fragments = [line for chunk in chunks for line in chunk.splitlines() if line.startswith("   42 ")]
        self.assertTrue(fragments[0].startswith("   42 + y"))
        self.assertTrue(all(line.startswith("   42 > y") for line in fragments[1:]))
        self.assertEqual("".join(line[len("   42 + "):] for line in fragments), long_line)
        for chunk in chunks:
            self.assertTrue(chunk.startswith("File: a.py\n"))
            self.assertLessEqual(len(chunk), 1000 + len("File: a.py\n"))

    def test_a_deletion_only_edit_with_no_surviving_context_is_listed_not_dropped(self):
        text = "diff --git a/auth.py b/auth.py\n--- a/auth.py\n+++ b/auth.py\n@@ -5,2 +4,0 @@ def f():\n-    if not user.is_admin:\n-        raise Forbidden\n"
        self.assertEqual(
            chunk_file_diff(split_diff(text)[0], 10_000),
            ["File: auth.py\n      ~ 2 line(s) removed after line 4:\n      -     if not user.is_admin:\n      -         raise Forbidden\n"],
        )

    def test_a_deletion_with_context_is_still_only_a_marker(self):
        chunk = chunk_file_diff(split_diff(self.MODIFIED)[0], 10_000)[0]
        self.assertIn("~ 2 line(s) removed here", chunk)
        self.assertNotIn("old line", chunk)

    def test_a_pure_deletion_hunk_after_another_hunk_keeps_its_gap_marker(self):
        text = ("diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1,2 @@\n x\n+y\n"
                "@@ -20,2 +21,0 @@\n-guard one\n-guard two\n")
        (chunk,) = chunk_file_diff(split_diff(text)[0], 10_000)
        self.assertEqual(chunk.splitlines()[3:], [GAP_MARKER, "      ~ 2 line(s) removed after line 21:", "      - guard one", "      - guard two"])

    def test_a_new_file_is_marked_as_one(self):
        text = "diff --git a/n.py b/n.py\nnew file mode 100644\n--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+x\n"
        self.assertEqual(chunk_file_diff(split_diff(text)[0], 10_000), ["File: n.py\n(new file)\n    1 + x\n"])

    def test_files_with_nothing_to_read_produce_no_chunks(self):
        deleted = "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x\n-y\n"
        binary = "diff --git a/i.png b/i.png\nBinary files a/i.png and b/i.png differ\n"
        mode_only = "diff --git a/s.sh b/s.sh\nold mode 100644\nnew mode 100755\n"
        for text in (deleted, binary, mode_only):
            with self.subTest(text=text[:30]):
                self.assertEqual(chunk_file_diff(split_diff(text)[0], 10_000), [])

    def test_chunk_diff_covers_every_file_in_order(self):
        diff = file_diff("a.py", [["+one"]]) + file_diff("b.py", [["+two"]])
        chunks = chunk_diff(diff, 10_000)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("File: a.py\n"))
        self.assertTrue(chunks[1].startswith("File: b.py\n"))

    def test_chunk_size_comes_from_the_context_length_and_the_prompt(self):
        small = chunk_chars_for(rv(context=4096), "s" * 100)
        large = chunk_chars_for(rv(context=32768), "s" * 100)
        self.assertGreater(large, small)
        self.assertLess(chunk_chars_for(rv(context=4096), "s" * 1000), small)

    def test_an_explicit_chunk_chars_wins(self):
        self.assertEqual(chunk_chars_for(rv(context=4096), "s" * 100, 1234), 1234)

    def test_a_context_too_small_for_its_own_prompt_is_an_error_naming_the_reviewer(self):
        with self.assertRaises(ReviewError) as caught:
            chunk_chars_for(rv(name="tiny", context=1024), "s" * 5000)
        self.assertIn("'tiny'", str(caught.exception))
        self.assertIn(str(MIN_CHUNK_CHARS), str(caught.exception))


class InstructionFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / ".github" / "instructions").mkdir(parents=True)

    def _instruction(self, name, apply_to, body):
        front = f'---\napplyTo: "{apply_to}"\n---\n\n' if apply_to is not None else ""
        (self.root / ".github" / "instructions" / name).write_text(front + body, encoding="utf-8")

    def test_applies_to_globs(self):
        self.assertTrue(applies_to("**/*.py", ["libs/a.py"]))
        self.assertTrue(applies_to("**/*.py", ["a.py"]))
        self.assertFalse(applies_to("**/*.py", ["README.md"]))
        self.assertTrue(applies_to("**", ["anything/at/all.txt"]))
        self.assertTrue(applies_to("*.py", ["a.py"]))
        self.assertFalse(applies_to("*.py", ["libs/a.py"]))  # `*` stays inside one directory
        self.assertTrue(applies_to("docs/**/*.md", ["docs/a/b/c.md"]))
        self.assertTrue(applies_to("docs/**/*.md", ["docs/c.md"]))
        self.assertFalse(applies_to("docs/**/*.md", ["other/c.md"]))
        self.assertTrue(applies_to("**/*.md, **/*.py", ["x/a.py"]))
        self.assertTrue(applies_to("src/?.py", ["src/a.py"]))
        self.assertFalse(applies_to("a.py", ["bxa.py"]))  # literal text is anchored and not a regex

    def test_glob_edge_cases_keep_their_meaning(self):
        self.assertTrue(applies_to("**.py", ["a/b/c.py"]))  # `**` inside a segment reads as `**/` plus `*`
        self.assertFalse(applies_to("**.py", ["a/b/c.md"]))
        self.assertTrue(applies_to("docs/**", ["docs/a/b.md"]))
        self.assertFalse(applies_to("docs/**", ["docs"]))  # a trailing `**` needs something under docs/
        self.assertTrue(applies_to("**/**/x.py", ["x.py"]))
        self.assertTrue(applies_to("a/**/b/**/c.py", ["a/x/y/b/z/c.py"]))
        self.assertFalse(applies_to("a/**/b/**/c.py", ["a/x/y/z/c.py"]))
        self.assertTrue(applies_to("*a*b?", ["xaxbz"]))
        self.assertFalse(applies_to("*a*b?", ["xaxb"]))
        self.assertTrue(applies_to("*.py", ["a*b.py"]))  # a `*` in a file name is just a character
        self.assertFalse(applies_to("[ab].py", ["a.py"]))  # brackets are literal, not a character class

    def test_a_glob_that_made_the_old_regex_run_for_minutes_is_answered_at_once(self):
        # Twelve `**/` followed by `*.py` against forty directories; and thirty `*a` runs in one segment.
        cases = [("**/" * 12 + "*.py", "dir/" * 40 + "file.txt"), ("*a" * 30 + "*b", "a" * 120), ("**/" * 40 + "x", "d/" * 60 + "y")]
        for pattern, path in cases:
            with self.subTest(pattern=pattern[:20]):
                started = time.monotonic()
                self.assertFalse(applies_to(pattern, [path]))
                self.assertLess(time.monotonic() - started, 1.0)

    def test_the_same_pathological_globs_still_match_when_they_should(self):
        self.assertTrue(applies_to("**/" * 12 + "*.py", ["dir/" * 40 + "file.py"]))
        self.assertTrue(applies_to("*a" * 10 + "*b", ["a" * 30 + "b"]))

    def test_a_glob_longer_than_the_limit_matches_nothing(self):
        self.assertFalse(applies_to("a" * 600, ["a" * 600]))

    def test_a_missing_or_blank_apply_to_applies_to_everything(self):
        self.assertTrue(applies_to(None, ["a.md"]))
        self.assertTrue(applies_to("  ", ["a.md"]))

    def test_only_files_that_apply_to_a_changed_path_are_returned(self):
        self._instruction("python.instructions.md", "**/*.py", "Python rules")
        self._instruction("docs.instructions.md", "**/*.md", "Docs rules")
        py = instruction_files_for(self.root, ["libs/a.py"])
        self.assertEqual([(p, t) for p, t in py], [(".github/instructions/python.instructions.md", "Python rules")])
        both = instruction_files_for(self.root, ["libs/a.py", "README.md"])
        self.assertEqual([p for p, _ in both], [".github/instructions/docs.instructions.md", ".github/instructions/python.instructions.md"])
        self.assertEqual(instruction_files_for(self.root, ["data.json"]), [])

    def test_a_file_without_apply_to_applies_to_every_change(self):
        self._instruction("all.instructions.md", None, "General rules")
        self.assertEqual(instruction_files_for(self.root, ["data.json"])[0][1], "General rules")

    def test_no_instructions_folder_is_no_files(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(instruction_files_for(empty, ["a.py"]), [])

    def test_a_very_long_file_is_cut(self):
        self._instruction("long.instructions.md", "**", "x" * 50_000)
        text = instruction_files_for(self.root, ["a.py"])[0][1]
        self.assertLess(len(text), 21_000)
        self.assertTrue(text.endswith("[truncated]"))

    def test_review_skills_are_found_by_directory_or_front_matter_name(self):
        for base in (".claude/skills", "skills"):
            (self.root / base).mkdir(parents=True)
        (self.root / ".claude/skills/code-review").mkdir()
        (self.root / ".claude/skills/code-review/SKILL.md").write_text("---\nname: x\n---\nReview steps", encoding="utf-8")
        (self.root / "skills/audit").mkdir()
        (self.root / "skills/audit/SKILL.md").write_text("---\nname: PR Review helper\n---\nAudit steps", encoding="utf-8")
        (self.root / "skills/deploy").mkdir()
        (self.root / "skills/deploy/SKILL.md").write_text("---\nname: deploy\n---\nDeploy steps", encoding="utf-8")
        found = review_skills(self.root)
        self.assertEqual(
            sorted(found),
            [(".claude/skills/code-review/SKILL.md", "Review steps"), ("skills/audit/SKILL.md", "Audit steps")],
        )


class PromptAssembly(unittest.TestCase):
    def test_the_prompt_has_the_lens_then_the_rules_then_the_output_format(self):
        prompt = build_system_prompt(
            rv(prompt="Look at quoting"),
            instructions=[(".github/instructions/py.instructions.md", "Only flag disasters")],
            skills=[("skills/review/SKILL.md", "Review steps")],
        )
        order = [prompt.index(x) for x in ("Look at quoting", "Only flag disasters", "Review steps", '"findings"')]
        self.assertEqual(order, sorted(order))
        self.assertIn(".github/instructions/py.instructions.md", prompt)

    def test_the_prompt_explains_the_numbered_new_version_before_the_output_format(self):
        prompt = build_system_prompt(rv())
        self.assertLess(prompt.index("NEW version"), prompt.index('"findings"'))
        self.assertIn("never report on them", prompt)

    def test_prompt_and_prompt_file_are_merged_with_the_text_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lens.md"
            path.write_text("From the file\n", encoding="utf-8")
            both = build_system_prompt(rv(prompt="Inline text", prompt_file=str(path)))
            only_file = build_system_prompt(rv(prompt="", prompt_file=str(path)))
        self.assertIn("Inline text\n\nFrom the file", both)
        self.assertIn("From the file", only_file)
        self.assertNotIn("Inline text", only_file)

    def test_the_prompt_file_is_read_at_run_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lens.md"
            path.write_text("first version", encoding="utf-8")
            reviewer = rv(prompt="", prompt_file=str(path))
            self.assertIn("first version", build_system_prompt(reviewer))
            path.write_text("second version", encoding="utf-8")
            self.assertIn("second version", build_system_prompt(reviewer))

    def test_lessons_are_included_when_given_and_absent_otherwise(self):
        self.assertNotIn("Lessons", build_system_prompt(rv()))
        self.assertIn("- check the quoting of paths", build_system_prompt(rv(), lessons=["check the quoting of paths"]))


class ParseFindings(unittest.TestCase):
    def test_a_plain_reply(self):
        found, problems = parse_findings(reply(finding()), "quoting")
        self.assertEqual(problems, [])
        self.assertEqual(found, [Finding("a.py", 3, 3, "quoting", "a claim", "x y", ("quoting",))])

    def test_fenced_and_prose_wrapped_replies_and_a_bare_list(self):
        body = reply(finding())
        for text in (f"```json\n{body}\n```", f"Here you go:\n{body}\nDone.", json.dumps([finding()])):
            with self.subTest(text=text[:20]):
                self.assertEqual(len(parse_findings(text, "q")[0]), 1)

    def test_a_bare_array_wrapped_in_prose_is_read(self):
        reply = 'Here are my findings: [{"claim": "lost update", "line_start": 3}]'
        findings, problems = parse_findings(reply, "q")
        self.assertEqual(([f.claim for f in findings], problems), (["lost update"], []))

    def test_an_object_with_a_findings_list_in_prose_is_still_read(self):
        reply = 'Sure! {"findings": [{"claim": "x", "line_start": 1}, {"claim": "y", "line_start": 2}]} Hope that helps.'
        self.assertEqual([f.claim for f in parse_findings(reply, "q")[0]], ["x", "y"])

    def test_the_outer_container_is_tried_first_whichever_kind_it_is(self):
        array_first = 'ok [{"claim": "a", "line_start": 1}, {"claim": "b", "line_start": 2}] and {"note": "x"}'
        self.assertEqual([f.claim for f in parse_findings(array_first, "q")[0]], ["a", "b"])

    def test_class_is_normalised(self):
        found, _ = parse_findings(reply(finding(cls="Shell Quoting!!")), "q")
        self.assertEqual(found[0].cls, "shell-quoting")
        found, _ = parse_findings(reply(finding(cls="")), "q")
        self.assertEqual(found[0].cls, "uncategorized")

    def test_missing_lines_and_file_fall_back(self):
        found, _ = parse_findings(json.dumps({"findings": [{"claim": "c"}]}), "q", default_file="from-chunk.py")
        self.assertEqual((found[0].file, found[0].line_start, found[0].line_end), ("from-chunk.py", 0, 0))

    def test_an_end_before_the_start_is_corrected(self):
        found, _ = parse_findings(reply(finding(start=9, end=2)), "q")
        self.assertEqual((found[0].line_start, found[0].line_end), (9, 9))

    def test_a_reply_that_is_not_json_is_a_problem_not_an_exception(self):
        found, problems = parse_findings("I could not find anything wrong.", "q")
        self.assertEqual(found, [])
        self.assertEqual(len(problems), 1)
        self.assertIn("not JSON", problems[0])

    def test_an_item_without_a_claim_is_skipped_and_the_rest_kept(self):
        found, problems = parse_findings(json.dumps({"findings": [{"file": "a.py"}, finding()]}), "q")
        self.assertEqual(len(found), 1)
        self.assertEqual(len(problems), 1)
        self.assertIn("finding 0 skipped", problems[0])

    def test_json_without_a_findings_list_is_a_problem(self):
        self.assertIn("no 'findings' list", parse_findings('{"result": 1}', "q")[1][0])

    def test_an_empty_findings_list_is_no_findings_and_no_problem(self):
        self.assertEqual(parse_findings('{"findings": []}', "q"), ([], []))


def f(reviewer, file="a.py", start=3, end=3, cls="quoting", claim="a claim", failing="x"):
    return Finding(file, start, end, cls, claim, failing, (reviewer,))


class CollectFindings(unittest.TestCase):
    def test_every_finding_from_every_reviewer_is_kept_with_no_vote_threshold(self):
        groups = collect_findings([
            f("one", file="a.py", cls="quoting", claim="only one reviewer saw this"),
            f("two", file="b.py", cls="state", claim="and this one only the other saw"),
        ])
        self.assertEqual(sum(len(g.findings) for g in groups), 2)

    def test_findings_are_grouped_by_class_in_a_stable_order(self):
        groups = collect_findings([f("a", cls="state"), f("a", file="b.py", cls="quoting"), f("a", file="c.py", cls="state")])
        self.assertEqual([(g.cls, len(g.findings)) for g in groups], [("quoting", 1), ("state", 2)])
        self.assertEqual([x.file for x in groups[1].findings], ["a.py", "c.py"])

    def test_overlapping_findings_with_different_claims_are_both_kept_each_credited_to_its_own_reviewer(self):
        # Two defects can share a class and overlapping lines (truncation and a lost concurrent update
        # are both "state"): the second must not be discarded, nor credited to the first's claim.
        groups = collect_findings([
            f("one", start=10, end=14, claim="the file is truncated before it is written", failing="crash mid-write"),
            f("two", start=12, end=20, claim="two writers lose each other's update", failing="two concurrent calls"),
        ])
        self.assertEqual(len(groups[0].findings), 1)
        merged = groups[0].findings[0]
        self.assertEqual((merged.claim, merged.reviewers, merged.failing_input), ("the file is truncated before it is written", ("one",), "crash mid-write"))
        self.assertEqual((merged.line_start, merged.line_end), (10, 20))
        (alternate,) = merged.alternates
        self.assertEqual((alternate.claim, alternate.reviewers, alternate.failing_input), ("two writers lose each other's update", ("two",), "two concurrent calls"))
        self.assertEqual((alternate.line_start, alternate.line_end), (12, 20))

    def test_the_same_claim_in_other_words_of_case_spacing_and_a_final_stop_merges_the_reviewers(self):
        merged = collect_findings([f("one", claim="Not thread safe."), f("two", claim="not  thread   safe")])[0].findings[0]
        self.assertEqual(merged.reviewers, ("one", "two"))
        self.assertEqual(merged.alternates, ())

    def test_a_failing_input_the_first_reviewer_lacked_is_taken_from_the_second(self):
        merged = collect_findings([f("one", claim="same", failing=""), f("two", claim="same", failing="the input")])[0].findings[0]
        self.assertEqual((merged.failing_input, merged.alternates), ("the input", ()))

    def test_the_same_claim_with_a_different_failing_input_keeps_both_inputs(self):
        merged = collect_findings([f("one", claim="same", failing="input A"), f("two", claim="same", failing="input B")])[0].findings[0]
        self.assertEqual(merged.failing_input, "input A")
        self.assertEqual([(a.failing_input, a.reviewers) for a in merged.alternates], [("input B", ("two",))])

    def test_failing_inputs_are_compared_exactly_not_loosely(self):
        # Concrete reproducers differ by case (paths) and whitespace (shell commands); only claims are prose.
        for first, second in (("Config.json", "config.json"), ("a id", "a\nid"), ("a  b", "a b")):
            with self.subTest(first=first, second=second):
                merged = collect_findings([f("one", claim="same", failing=first), f("two", claim="same", failing=second)])[0].findings[0]
                self.assertEqual(merged.failing_input, first)
                self.assertEqual([(a.failing_input, a.reviewers) for a in merged.alternates], [(second, ("two",))])

    def test_identical_failing_inputs_apart_from_surrounding_whitespace_merge(self):
        merged = collect_findings([f("one", claim="same", failing="x y"), f("two", claim="same", failing=" x y\n")])[0].findings[0]
        self.assertEqual((merged.reviewers, merged.alternates), (("one", "two"), ()))

    def test_a_repeated_alternate_keeps_the_whole_line_range_it_was_reported_over(self):
        groups = collect_findings([
            f("one", start=10, end=30, claim="A", failing="a"),
            f("two", start=12, end=14, claim="B", failing="b"),
            f("three", start=13, end=20, claim="B", failing="b"),
            f("four", start=8, end=11, claim="B", failing="b"),
        ])
        (merged,) = groups[0].findings
        (alternate,) = merged.alternates
        self.assertEqual((alternate.reviewers, alternate.line_start, alternate.line_end), (("two", "three", "four"), 8, 20))
        self.assertEqual((merged.line_start, merged.line_end), (8, 30))

    def test_a_repeated_alternate_adds_its_reviewer_instead_of_another_alternate(self):
        merged = collect_findings([f("one", claim="A"), f("two", claim="B"), f("three", claim="B")])[0].findings[0]
        self.assertEqual([(a.claim, a.reviewers) for a in merged.alternates], [("B", ("two", "three"))])

    def test_the_output_shows_each_alternate_with_its_reviewers_and_failing_input(self):
        groups = collect_findings([f("one", start=10, end=12, claim="truncated", failing="a"), f("two", start=11, end=11, claim="lost update", failing="b")])
        text = format_review(ReasoningReview(groups=groups, reviewers_run=["one", "two"]))
        self.assertIn("a.py:10-12 [one] truncated", text)
        self.assertIn("also, at 11 [two]: lost update", text)
        self.assertIn("failing input: b", text)

    def test_the_same_reviewer_is_listed_once(self):
        merged = collect_findings([f("one"), f("one")])[0].findings[0]
        self.assertEqual(merged.reviewers, ("one",))

    def test_different_class_file_or_distant_lines_do_not_merge(self):
        self.assertEqual(len(collect_findings([f("a", cls="x"), f("b", cls="y")])), 2)
        self.assertEqual(len(collect_findings([f("a", file="a.py"), f("b", file="b.py")])[0].findings), 2)
        self.assertEqual(len(collect_findings([f("a", start=1, end=2), f("b", start=50, end=60)])[0].findings), 2)

    def test_findings_without_a_line_merge_only_when_the_claim_is_the_same_text(self):
        same = collect_findings([f("a", start=0, end=0, claim="Same  claim"), f("b", start=0, end=0, claim="same claim")])
        self.assertEqual(len(same[0].findings), 1)
        different = collect_findings([f("a", start=0, end=0, claim="one"), f("b", start=0, end=0, claim="two")])
        self.assertEqual(len(different[0].findings), 2)
        mixed = collect_findings([f("a", start=0, end=0), f("b", start=5, end=5)])
        self.assertEqual(len(mixed[0].findings), 2)

    def test_no_findings_is_no_groups(self):
        self.assertEqual(collect_findings([]), [])


class RulesFilesAreNotAWayOutOfTheRepository(unittest.TestCase):
    """Instruction files and review skills are found automatically in a repository someone else controls."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "repo"
        (self.root / ".github" / "instructions").mkdir(parents=True)
        self.outside = self.base / "private.txt"
        self.outside.write_text("SECRET-TOKEN-12345\n", encoding="utf-8")

    def test_an_instruction_file_that_links_outside_the_repository_is_refused(self):
        (self.root / ".github" / "instructions" / "leak.instructions.md").symlink_to(self.outside)
        self.assertEqual(instruction_files_for(self.root, ["a.py"]), [])

    def test_a_review_skill_that_links_outside_the_repository_is_refused(self):
        (self.root / "skills" / "code-review").mkdir(parents=True)
        (self.root / "skills" / "code-review" / "SKILL.md").symlink_to(self.outside)
        self.assertEqual(review_skills(self.root), [])

    def test_a_symlinked_directory_that_leads_outside_is_refused(self):
        elsewhere = self.base / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "x.instructions.md").write_text("OUTSIDE\n", encoding="utf-8")
        (self.root / ".github" / "instructions").rmdir()
        (self.root / ".github" / "instructions").symlink_to(elsewhere)
        self.assertEqual(instruction_files_for(self.root, ["a.py"]), [])

    def test_a_self_linked_file_or_directory_is_skipped_not_fatal(self):
        # Before Python 3.13, Path.resolve() raises RuntimeError (not OSError) for a symlink loop.
        (self.root / ".github" / "instructions" / "loop.instructions.md").symlink_to("loop.instructions.md")
        (self.root / "skills" / "code-review").mkdir(parents=True)
        (self.root / "skills" / "code-review" / "SKILL.md").symlink_to("SKILL.md")
        (self.root / "skills" / "loopdir-review").symlink_to("loopdir-review")
        self.assertEqual(instruction_files_for(self.root, ["a.py"]), [])
        self.assertEqual(review_skills(self.root), [])

    def test_a_loop_alongside_a_good_file_does_not_hide_the_good_file(self):
        (self.root / ".github" / "instructions" / "a-loop.instructions.md").symlink_to("a-loop.instructions.md")
        (self.root / ".github" / "instructions" / "z-good.instructions.md").write_text("GOOD RULE\n", encoding="utf-8")
        self.assertEqual(instruction_files_for(self.root, ["a.py"]), [(".github/instructions/z-good.instructions.md", "GOOD RULE")])

    def _git(self, *args):
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True, env=env)

    def _tracked_rule(self, name="rules.md"):
        real = self.root / "docs" / name
        real.parent.mkdir(exist_ok=True)
        real.write_text("---\napplyTo: \"**/*.py\"\n---\nINSIDE RULE\n", encoding="utf-8")
        return real

    def test_a_link_to_a_file_git_tracks_inside_the_repository_is_allowed(self):
        self._git("init", "-q", "-b", "main")
        real = self._tracked_rule()
        (self.root / ".github" / "instructions" / "linked.instructions.md").symlink_to(real)
        self._git("add", "-A")
        self.assertEqual(instruction_files_for(self.root, ["a.py"]), [(".github/instructions/linked.instructions.md", "INSIDE RULE")])

    def test_a_link_to_a_tracked_file_whose_name_has_glob_characters_is_allowed(self):
        self._git("init", "-q", "-b", "main")
        real = self._tracked_rule("rules[1].md")
        (self.root / ".github" / "instructions" / "linked.instructions.md").symlink_to(real)
        self._git("add", "-A")
        self.assertEqual(len(instruction_files_for(self.root, ["a.py"])), 1)

    def test_a_link_to_a_secret_that_stays_inside_the_checkout_is_refused(self):
        # Containment is not safety: .env is inside the repository but is not part of it.
        self._git("init", "-q", "-b", "main")
        (self.root / ".gitignore").write_text(".env\n", encoding="utf-8")
        (self.root / ".env").write_text("API_KEY=sk-live-123\n", encoding="utf-8")
        (self.root / ".github" / "instructions" / "env.instructions.md").symlink_to("../../.env")
        self._git("add", "-A")
        self.assertEqual(instruction_files_for(self.root, ["a.py"]), [])

    def test_a_link_into_the_git_directory_is_refused(self):
        self._git("init", "-q", "-b", "main")
        (self.root / ".github" / "instructions" / "cfg.instructions.md").symlink_to("../../.git/config")
        self.assertEqual(instruction_files_for(self.root, ["a.py"]), [])
        (self.root / "skills" / "code-review").mkdir(parents=True)
        (self.root / "skills" / "code-review" / "SKILL.md").symlink_to("../../.git/HEAD")
        self.assertEqual(review_skills(self.root), [])

    def test_an_untracked_link_target_is_refused_even_when_it_is_ordinary_text(self):
        self._git("init", "-q", "-b", "main")
        real = self._tracked_rule()  # created but never added
        (self.root / ".github" / "instructions" / "linked.instructions.md").symlink_to(real)
        self.assertEqual(instruction_files_for(self.root, ["a.py"]), [])

    def test_a_link_in_a_directory_that_is_not_a_git_repository_is_refused(self):
        real = self._tracked_rule()
        (self.root / ".github" / "instructions" / "linked.instructions.md").symlink_to(real)
        self.assertEqual(instruction_files_for(self.root, ["a.py"]), [])

    def test_an_ordinary_untracked_rules_file_is_still_read(self):
        # Only links are checked against git: a plain file in the rules folder is the rules file itself.
        self._git("init", "-q", "-b", "main")
        (self.root / ".github" / "instructions" / "mine.instructions.md").write_text("MY RULE\n", encoding="utf-8")
        self.assertEqual(instruction_files_for(self.root, ["a.py"]), [(".github/instructions/mine.instructions.md", "MY RULE")])

    def test_a_target_that_is_not_a_regular_file_is_skipped_without_being_opened(self):
        if not hasattr(os, "mkfifo") or not hasattr(signal, "SIGALRM"):
            self.skipTest("needs a POSIX FIFO and SIGALRM")
        fifo = self.root / "pipe"
        os.mkfifo(fifo)
        (self.root / ".github" / "instructions" / "pipe.instructions.md").symlink_to(fifo)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(AssertionError("opened a FIFO: this would block forever")))
        signal.alarm(5)
        try:
            self.assertEqual(instruction_files_for(self.root, ["a.py"]), [])
        finally:
            signal.alarm(0)

    def test_at_most_the_read_limit_is_read_from_a_big_file(self):
        big = self.root / ".github" / "instructions" / "big.instructions.md"
        big.write_text("x" * (MAX_READ_BYTES * 3), encoding="utf-8")
        text = _read_text(big, self.root)
        assert text is not None
        self.assertEqual(len(text), MAX_READ_BYTES)

    def test_a_multi_byte_character_cut_by_the_read_limit_is_not_a_bad_file(self):
        big = self.root / "wide.md"
        big.write_text("é" * MAX_READ_BYTES, encoding="utf-8")  # 2 bytes each: the limit lands mid-character
        self.assertIsNotNone(_read_text(big, self.root))

    def test_a_file_that_is_not_utf8_is_skipped(self):
        bad = self.root / "bad.md"
        bad.write_bytes(b"\xff\xfe not utf-8")
        self.assertIsNone(_read_text(bad, self.root))

    def test_without_a_root_the_path_is_read_wherever_it_points(self):
        # A prompt file the user names themselves is their own decision, outside this check.
        self.assertEqual(_read_text(self.outside), "SECRET-TOKEN-12345\n")


class GitPathQuoting(unittest.TestCase):
    def test_unquoting_decodes_octal_utf8_escapes_quotes_backslashes_and_control_characters(self):
        cases = {
            '"caf\\303\\251.py"': "café.py",
            '"we\\"ird.py"': 'we"ird.py',
            '"back\\\\slash.py"': "back\\slash.py",
            '"tab\\there.py"': "tab\there.py",
            '"\\346\\227\\245\\346\\234\\254.py"': "日本.py",
            "plain.py": "plain.py",
            "dir/with space.py": "dir/with space.py",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(_unquote_git_path(raw), expected)

    def test_a_section_with_quoted_names_in_each_header_form_is_named_by_the_decoded_path(self):
        new_file = 'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\n--- /dev/null\n+++ "b/caf\\303\\251.py"\n@@ -0,0 +1 @@\n+x\n'
        modified = 'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\n--- "a/caf\\303\\251.py"\n+++ "b/caf\\303\\251.py"\n@@ -1 +1 @@\n-x\n+y\n'
        deleted = 'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\ndeleted file mode 100644\n--- "a/caf\\303\\251.py"\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n'
        header_only = 'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\nold mode 100644\nnew mode 100755\n'
        binary = 'diff --git "a/img\\303\\251.png" "b/img\\303\\251.png"\nBinary files differ\n'
        renamed = 'diff --git a/old.py "b/caf\\303\\251.py"\nsimilarity index 100%\nrename from old.py\nrename to "caf\\303\\251.py"\n'
        for name, text, expected in [("new", new_file, "café.py"), ("modified", modified, "café.py"), ("deleted", deleted, "café.py"),
                                     ("header only", header_only, "café.py"), ("binary", binary, "imgé.png"), ("renamed", renamed, "café.py")]:
            with self.subTest(form=name):
                self.assertEqual(split_diff(text)[0].path, expected)

    def test_a_name_with_a_space_keeps_working_with_git_trailing_tab(self):
        text = "diff --git a/my file.py b/my file.py\n--- a/my file.py\t\n+++ b/my file.py\t\n@@ -1 +1 @@\n-a\n+b\n"
        self.assertEqual(split_diff(text)[0].path, "my file.py")

    def test_real_git_output_for_non_ascii_and_quoted_names_is_decoded_and_matches_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
                   "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

            def git(*args):
                return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env, text=True).stdout

            git("init", "-q", "-b", "main")
            (repo / "a.txt").write_text("x\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-q", "-m", "base")
            git("checkout", "-q", "-b", "feature")
            for name in ("café.py", 'we"ird.py', "plain.py"):
                (repo / name).write_text("print(1)\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-q", "-m", "change")
            paths = [section.path for section in split_diff(git("diff", "main...feature"))]
        self.assertEqual(paths, ["café.py", "plain.py", 'we"ird.py'])
        self.assertTrue(applies_to("**/*.py", paths))


class StubComplete:
    """A `complete` stand-in: calls recorded, replies chosen per (model, chunk)."""

    def __init__(self, replies):
        self.replies = replies  # callable(model, system, content) -> str | None
        self.calls: list = []
        self._lock = threading.Lock()

    def __call__(self, client, model, system, content):
        with self._lock:
            self.calls.append((model, system, content))
        return self.replies(model, system, content)


class RunReasoningReview(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / ".github" / "instructions").mkdir(parents=True)
        (self.root / ".github" / "instructions" / "py.instructions.md").write_text(
            '---\napplyTo: "**/*.py"\n---\nOnly flag disasters', encoding="utf-8"
        )
        self.diff = file_diff("a.py", [["+one"]]) + file_diff("b.md", [["+two"]])

    def _run(self, config, replies, *, loaded=("model-a", "model-b"), diff=None, max_reviewers=3, **kw):
        fake = FakeLmStudio(loaded=loaded)
        stub = StubComplete(replies)
        result = run_reasoning_review(
            config, ModelLoader(BASE, http=fake), object(), self.diff if diff is None else diff, self.root, max_reviewers, complete=stub, **kw
        )
        return result, stub, fake

    def test_the_reviewers_findings_all_come_back_grouped(self):
        config = cfg(rv("one", "model-a"), rv("two", "model-b", prompt="look at state"))

        def replies(model, system, content):
            if model == "model-a":
                return reply(finding(file="a.py", cls="quoting", claim="seen by one"))
            return reply(finding(file="b.md", start=1, end=1, cls="state", claim="seen by two"))

        result, stub, _ = self._run(config, replies)
        self.assertEqual({g.cls for g in result.groups}, {"quoting", "state"})
        self.assertEqual(result.finding_count, 2)
        self.assertEqual(sorted(result.reviewers_run), ["one", "two"])
        self.assertEqual(result.errors, [])
        text = format_review(result)
        self.assertIn("seen by one", text)
        self.assertIn("NOT verified", text)

    def test_only_active_reasoning_reviewers_run(self):
        config = cfg(
            rv("on", "model-a"),
            rv("off", "model-a", active=False),
            rv("judge", "model-b", roles=("classifier",)),
        )
        result, stub, _ = self._run(config, lambda *a: reply())
        self.assertEqual(result.reviewers_run, ["on"])
        self.assertEqual({call[0] for call in stub.calls}, {"model-a"})

    def test_each_reviewer_gets_its_own_prompt_and_the_applicable_rules(self):
        config = cfg(rv("one", "model-a", prompt="LENS-ONE"), rv("two", "model-b", prompt="LENS-TWO"))
        _, stub, _ = self._run(config, lambda *a: reply())
        by_model = {model: system for model, system, _ in stub.calls}
        self.assertIn("LENS-ONE", by_model["model-a"])
        self.assertNotIn("LENS-TWO", by_model["model-a"])
        self.assertIn("LENS-TWO", by_model["model-b"])
        for system in by_model.values():
            self.assertIn("Only flag disasters", system)  # a.py changed, so the .py instructions apply

    def test_a_markdown_only_diff_leaves_out_the_python_instructions(self):
        _, stub, _ = self._run(cfg(rv()), lambda *a: reply(), diff=file_diff("README.md", [["+x"]]))
        self.assertNotIn("Only flag disasters", stub.calls[0][1])

    def test_a_diff_bigger_than_the_context_is_sent_in_several_requests(self):
        big = file_diff("big.py", [[f"+line {h}-{i}" for i in range(20)] for h in range(8)])
        result, stub, _ = self._run(cfg(rv()), lambda *a: reply(), diff=big, chunk_chars=1000)
        self.assertGreater(len(stub.calls), 1)
        self.assertEqual(result.chunks, len(stub.calls))
        sent = "".join(content for _, _, content in stub.calls)
        for h in range(8):
            self.assertEqual(sent.count(f"{1 + 30 * h:>5} + line {h}-0\n"), 1)

    def test_files_that_could_not_be_sent_are_reported_as_a_warning(self):
        binary = "diff --git a/i.png b/i.png\nBinary files a/i.png and b/i.png differ\n"
        deleted = "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
        result, stub, _ = self._run(cfg(rv()), lambda *a: reply(), diff=file_diff("a.py", [["+x"]]) + binary + deleted)
        self.assertEqual(len(stub.calls), 1)
        (warning,) = [w for w in result.warnings if "no new text" in w]
        self.assertIn("2 changed file(s)", warning)
        self.assertIn("i.png", warning)
        self.assertIn("gone.py", warning)

    def test_a_deletion_only_edit_is_sent_to_the_reviewer_with_no_warning(self):
        text = "diff --git a/auth.py b/auth.py\n--- a/auth.py\n+++ b/auth.py\n@@ -5,2 +4,0 @@\n-    if not user.is_admin:\n-        raise Forbidden\n"
        result, stub, _ = self._run(cfg(rv()), lambda *a: reply(), diff=text + file_diff("b.py", [["+y"]]))
        self.assertEqual(len(stub.calls), 2)
        self.assertIn("      - ", stub.calls[0][2])
        self.assertFalse([w for w in result.warnings if "no new text" in w])

    def test_a_diff_with_no_text_to_read_is_refused_with_the_reason(self):
        binary = "diff --git a/i.png b/i.png\nBinary files a/i.png and b/i.png differ\n"
        with self.assertRaises(ReviewError) as caught:
            self._run(cfg(rv()), lambda *a: reply(), diff=binary)
        self.assertIn("no text changes", str(caught.exception))

    def test_the_same_finding_in_two_chunks_is_merged(self):
        big = file_diff("big.py", [[f"+line {h}-{i}" for i in range(20)] for h in range(8)])
        result, stub, _ = self._run(cfg(rv()), lambda *a: reply(finding(file="big.py")), diff=big, chunk_chars=1000)
        self.assertGreater(len(stub.calls), 1)
        self.assertEqual(result.finding_count, 1)

    def test_a_chunk_without_a_file_in_the_reply_is_attributed_to_the_chunks_file(self):
        result, _, _ = self._run(cfg(rv()), lambda *a: json.dumps({"findings": [{"claim": "c"}]}), diff=file_diff("only.py", [["+x"]]))
        self.assertEqual(result.groups[0].findings[0].file, "only.py")

    def test_a_failed_reviewer_is_reported_and_the_others_findings_still_come_back(self):
        config = cfg(rv("good", "model-a"), rv("bad", "model-b"))
        result, _, _ = self._run(config, lambda model, *a: reply(finding()) if model == "model-a" else None)
        self.assertEqual(result.finding_count, 1)
        self.assertEqual(result.reviewers_run, ["good"])
        self.assertTrue(any("'bad'" in e and "every request failed" in e for e in result.errors), result.errors)

    def test_one_failed_chunk_keeps_that_reviewers_other_chunks(self):
        big = file_diff("big.py", [[f"+line {h}-{i}" for i in range(20)] for h in range(8)])
        calls = {"n": 0}

        def replies(model, system, content):
            calls["n"] += 1
            return None if calls["n"] == 1 else reply(finding(file="big.py"))

        result, _, _ = self._run(cfg(rv()), replies, diff=big, chunk_chars=1000)
        self.assertEqual(result.finding_count, 1)
        self.assertTrue(any("chunk 1/" in e for e in result.errors), result.errors)

    def test_a_malformed_reply_is_a_problem_and_loses_nothing_else(self):
        config = cfg(rv("good", "model-a"), rv("rambler", "model-b"))
        result, _, _ = self._run(config, lambda model, *a: reply(finding()) if model == "model-a" else "no idea")
        self.assertEqual(result.finding_count, 1)
        self.assertTrue(any("not JSON" in e for e in result.errors))

    def test_no_active_reasoning_reviewer_is_an_error(self):
        with self.assertRaises(ReviewError) as caught:
            self._run(cfg(rv(roles=("classifier",))), lambda *a: reply())
        self.assertIn("no active reasoning reviewer", str(caught.exception))

    def test_an_empty_diff_is_an_error(self):
        with self.assertRaises(ReviewError) as caught:
            self._run(cfg(rv()), lambda *a: reply(), diff="")
        self.assertIn("empty", str(caught.exception))

    def test_a_diff_needing_too_many_requests_is_reported_per_reviewer_with_advice(self):
        big = file_diff("big.py", [[f"+line {h}-{i}" for i in range(20)] for h in range(MAX_CHUNKS * 5)])
        result, stub, _ = self._run(cfg(rv()), lambda *a: reply(), diff=big, chunk_chars=1000)
        self.assertEqual(stub.calls, [])
        self.assertTrue(any("smaller range" in e for e in result.errors), result.errors)

    def test_a_resident_model_that_is_not_loaded_stops_before_anything_runs(self):
        stub = StubComplete(lambda *a: reply())
        fake = FakeLmStudio(loaded=())
        with self.assertRaises(ModelNotLoadedError):
            run_reasoning_review(cfg(rv()), ModelLoader(BASE, http=fake), object(), self.diff, self.root, 3, complete=stub)
        self.assertEqual(stub.calls, [])

    def test_an_on_demand_model_is_loaded_for_the_run_and_unloaded_after(self):
        fake = FakeLmStudio(loaded=())
        stub = StubComplete(lambda *a: reply(finding()))
        result = run_reasoning_review(
            cfg(rv("demand", "model-x", on_demand=True, context=16384)),
            ModelLoader(BASE, http=fake), object(), self.diff, self.root, 3, complete=stub,
        )
        self.assertEqual(result.finding_count, 1)
        self.assertEqual(len(fake.posts("/api/v1/models/load")), 1)
        self.assertEqual(len(fake.posts("/api/v1/models/unload")), 1)
        self.assertEqual(fake.loaded, {})

    def test_at_most_max_reviewers_run_at_once(self):
        active = {"now": 0, "peak": 0}
        lock = threading.Lock()
        gate = threading.Event()

        def replies(model, system, content):
            with lock:
                active["now"] += 1
                active["peak"] = max(active["peak"], active["now"])
            gate.wait(0.05)
            with lock:
                active["now"] -= 1
            return reply()

        config = cfg(*[rv(f"r{i}", "model-a") for i in range(5)])
        self._run(config, replies, max_reviewers=2)
        self.assertLessEqual(active["peak"], 2)

    def test_lessons_reach_every_reviewers_prompt(self):
        _, stub, _ = self._run(cfg(rv()), lambda *a: reply(), lessons=["paths have spaces"])
        self.assertIn("paths have spaces", stub.calls[0][1])

    def test_the_formatted_review_lists_files_lines_reviewers_and_failing_input(self):
        result, _, _ = self._run(cfg(rv("one")), lambda *a: reply(finding(file="a.py", start=3, end=7, failing="a b")))
        text = format_review(result)
        self.assertIn("## quoting (1)", text)
        self.assertIn("a.py:3-7 [one] a claim", text)
        self.assertIn("failing input: a b", text)


class ReasoningModelReplies(unittest.TestCase):
    """DeepSeek-R1-style models write `<think>...</think>` first; its braces must not be mistaken for the answer."""

    ANSWER = '{"findings": [{"file": "a.py", "line_start": 3, "claim": "bad slice", "class": "off-by-one"}]}'

    def test_braces_in_the_thinking_do_not_hide_the_json(self):
        reply = f"<think>The code has {{x: 1}} and a list [1, 2]. The slice looks off.</think>\n{self.ANSWER}"
        findings, problems = parse_findings(reply, "r1")
        self.assertEqual([f.claim for f in findings], ["bad slice"])
        self.assertEqual(problems, [])

    def test_a_fenced_answer_after_the_thinking_is_read(self):
        reply = f"<think>hmm {{</think>\n```json\n{self.ANSWER}\n```"
        self.assertEqual(len(parse_findings(reply, "r1")[0]), 1)

    def test_a_reply_cut_off_while_still_thinking_is_a_problem_not_a_finding(self):
        findings, problems = parse_findings('<think>maybe {"findings": [{"claim": "x"}]} but wait', "r1")
        self.assertEqual(findings, [])
        self.assertEqual(len(problems), 1)

    def test_consecutive_leading_envelopes_are_all_removed(self):
        reply = f"<think>a {{</think>\n<think>b }}</think>\n{self.ANSWER}"
        self.assertEqual(len(parse_findings(reply, "r1")[0]), 1)

    def test_a_literal_think_tag_inside_a_failing_input_is_kept(self):
        # A finding about code that parses reasoning tags (this very module) must not lose its input.
        answer = json.dumps({"findings": [{"file": "a.py", "line_start": 3, "claim": "c", "class": "x", "failing_input": "<think>x</think>"}]})
        for reply in (answer, f"<think>planning</think>\n{answer}"):
            with self.subTest(reply=reply[:20]):
                self.assertEqual([x.failing_input for x in parse_findings(reply, "r1")[0]], ["<think>x</think>"])

    def test_an_unmatched_literal_think_tag_in_one_claim_does_not_discard_the_findings(self):
        answer = json.dumps({"findings": [
            {"file": "a.py", "line_start": 3, "claim": "treats a literal <think> as the start of reasoning", "class": "x"},
            {"file": "a.py", "line_start": 9, "claim": "second", "class": "y"},
        ]})
        for reply in (answer, f"<think>plan</think>{answer}"):
            with self.subTest(reply=reply[:20]):
                self.assertEqual(len(parse_findings(reply, "r1")[0]), 2)

    def test_a_think_envelope_that_is_not_at_the_start_is_not_stripped(self):
        reply = 'Here you go: {"findings": [{"claim": "uses <think>x</think> tags", "line_start": 1}]}'
        self.assertEqual([x.claim for x in parse_findings(reply, "q")[0]], ["uses <think>x</think> tags"])

    def test_a_reply_without_thinking_is_unchanged(self):
        self.assertEqual(len(parse_findings(self.ANSWER, "q")[0]), 1)


if __name__ == "__main__":
    unittest.main()
