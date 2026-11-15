#!/usr/bin/env python3
"""Tests for libs/reviewer_tools_lib.py: the tool executor and its two repo tools, `read_file` and `grep`.

Stdlib-only (unittest), no network. The tools run against a real throwaway git repository built by
`tests/reviewer_repo_fixture.py`; nothing here needs LM Studio.

    .venv/bin/python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import reviewer_tools_lib as tools  # noqa: E402
from reasoning_reviewer_lib import ReviewError  # noqa: E402
from reviewer_repo_fixture import build_repo, build_repo_with, git  # noqa: E402


class RepoCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "repo"
        cls.head = build_repo(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.ex = tools.ToolExecutor(self.root, self.head)


class TheExecutorNeverRaises(RepoCase):
    """Every failure comes back as `ToolResult(error=True)` text."""

    def assert_error(self, result, *needles):
        self.assertIsInstance(result, tools.ToolResult)
        self.assertTrue(result.error, result.text)
        for needle in needles:
            self.assertIn(needle, result.text)

    def test_an_unknown_tool_is_an_error_naming_the_real_ones(self):
        self.assert_error(self.ex.run("rm_rf", {}), "unknown tool", "read_file", "grep")

    def test_a_tool_name_that_is_not_a_string_is_an_error(self):
        self.assert_error(self.ex.run(None, {}), "unknown tool")
        self.assert_error(self.ex.run(["read_file"], {}), "unknown tool")

    def test_arguments_that_are_not_an_object_are_an_error(self):
        self.assert_error(self.ex.run("read_file", "app/orders.py"), "must be an object")

    def test_a_missing_argument_is_an_error(self):
        self.assert_error(self.ex.run("read_file", {}), "needs a 'path'")
        self.assert_error(self.ex.run("grep", {}), "non-empty 'pattern'")

    def test_an_argument_the_tool_does_not_take_is_an_error(self):
        self.assert_error(self.ex.run("read_file", {"path": "app/orders.py", "mode": "w"}), "does not take", "mode")

    def test_a_path_outside_the_repo_is_an_error(self):
        self.assert_error(self.ex.run("read_file", {"path": "../outside.txt"}), "outside the repository")
        self.assert_error(self.ex.run("read_file", {"path": "app/../../outside.txt"}), "outside the repository")
        self.assert_error(self.ex.run("read_file", {"path": "/etc/passwd"}), "outside the repository")
        self.assert_error(self.ex.run("read_file", {"path": "C:\\Windows\\win.ini"}), "outside the repository")

    def test_an_untracked_path_is_an_error(self):
        self.assert_error(self.ex.run("read_file", {"path": "app/scratch.py"}), "not a tracked file")
        self.assert_error(self.ex.run("read_file", {"path": ".env"}), "not a tracked file")

    def test_nothing_under_dot_git_is_readable(self):
        self.assert_error(self.ex.run("read_file", {"path": ".git/config"}), "not a tracked file")

    def test_a_file_that_is_staged_but_missing_at_the_head_is_an_error_of_its_own(self):
        self.assert_error(self.ex.run("read_file", {"path": "late.py"}), "not in the commit under review")

    def test_a_directory_is_an_error(self):
        self.assert_error(self.ex.run("read_file", {"path": "app"}), "directory")

    def test_a_bad_regex_is_an_error_not_an_exception(self):
        self.assert_error(self.ex.run("grep", {"pattern": "(unclosed"}), "grep failed")

    def test_a_glob_that_would_smuggle_git_pathspec_syntax_is_refused(self):
        self.assert_error(self.ex.run("grep", {"pattern": "def", "path_glob": ":(top)../*"}), "path_glob")
        self.assert_error(self.ex.run("grep", {"pattern": "def", "path_glob": "../*"}), "path_glob")

    def test_a_head_that_git_would_read_as_an_option_is_refused_up_front(self):
        for bad in ("--output=x", "", "a b", "a\0b"):
            with self.assertRaises(ReviewError):
                tools.ToolExecutor(self.root, bad)

    def test_a_repo_that_is_not_a_git_repository_gives_errors_not_exceptions(self):
        with tempfile.TemporaryDirectory() as empty:
            ex = tools.ToolExecutor(empty, "HEAD")
            # git failing is 'could not be checked', not 'not a tracked file': nothing is known about the file.
            self.assert_error(ex.run("read_file", {"path": "x.py"}), "could not be checked")
            self.assert_error(ex.run("grep", {"pattern": "x"}), "grep failed")


class OnlyTheHeadsTrackedFilesAreVisible(RepoCase):
    def test_read_file_returns_the_head_version_not_the_working_tree(self):
        # The fixture edited pricing.py in the working tree after the commit; the head still has `return total`.
        text = self.ex.run("read_file", {"path": "app/pricing.py"}).text
        self.assertIn("return total\n", text + "\n")
        self.assertNotIn("1000", text)

    def test_a_dot_slash_prefix_is_accepted(self):
        self.assertFalse(self.ex.run("read_file", {"path": "./app/orders.py"}).error)

    def test_read_file_numbers_lines_and_honours_a_range(self):
        result = self.ex.run("read_file", {"path": "app/pricing.py", "start_line": 2, "end_line": 3})
        self.assertFalse(result.error)
        lines = result.text.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].lstrip().startswith("2 "))
        self.assertIn("total = 0", lines[0])
        self.assertIn("for item in items:", lines[1])

    def test_read_file_start_past_the_end_is_an_error(self):
        result = self.ex.run("read_file", {"path": "app/orders.py", "start_line": 500})
        self.assertTrue(result.error)
        self.assertIn("past the end", result.text)

    def test_read_file_rejects_a_bad_line_number(self):
        for bad in (0, -1, True, "3", 2.5):
            self.assertTrue(self.ex.run("read_file", {"path": "app/orders.py", "start_line": bad}).error, bad)

    def test_a_long_file_is_cut_with_a_note_saying_how_to_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_repo(root)
            (root / "big.txt").write_text("\n".join(f"line {i}" for i in range(1, 1001)) + "\n", encoding="utf-8")
            git(root, "add", "big.txt")
            git(root, "commit", "-q", "-m", "big")
            ex = tools.ToolExecutor(root, git(root, "rev-parse", "HEAD"))
            result = ex.run("read_file", {"path": "big.txt"})
            self.assertFalse(result.error)
            self.assertTrue(result.truncated)
            self.assertIn("start_line", result.text)
            self.assertLessEqual(len(result.text), tools.MAX_TOOL_RESULT_CHARS + 200)

    def test_grep_finds_matches_with_paths_and_line_numbers(self):
        result = self.ex.run("grep", {"pattern": "def total_price"})
        self.assertFalse(result.error)
        self.assertIn("app/pricing.py:1:def total_price(items):", result.text)

    def test_grep_honours_a_path_glob(self):
        result = self.ex.run("grep", {"pattern": "total_price", "path_glob": "app/*.py"})
        self.assertIn("app/pricing.py", result.text)
        self.assertNotIn("tests/test_pricing.py", result.text)

    def test_grep_never_sees_an_untracked_file(self):
        self.assertEqual(self.ex.run("grep", {"pattern": "scratch_only"}).text, "no matches")
        self.assertEqual(self.ex.run("grep", {"pattern": "hunter2"}).text, "no matches")

    def test_grep_never_sees_a_staged_but_uncommitted_file(self):
        self.assertEqual(self.ex.run("grep", {"pattern": "late_function"}).text, "no matches")

    def test_grep_with_no_match_is_a_normal_result_not_an_error(self):
        result = self.ex.run("grep", {"pattern": "zzz_not_anywhere"})
        self.assertFalse(result.error)
        self.assertEqual(result.text, "no matches")

    def test_grep_is_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_repo(root)
            (root / "many.txt").write_text("\n".join(f"hit {i}" for i in range(500)) + "\n", encoding="utf-8")
            git(root, "add", "many.txt")
            git(root, "commit", "-q", "-m", "many")
            result = tools.ToolExecutor(root, git(root, "rev-parse", "HEAD")).run("grep", {"pattern": "^hit"})
            self.assertTrue(result.truncated)
            self.assertLessEqual(result.text.count("\n") + 1, tools.GREP_MAX_LINES + 1)


class FoundByTheReview(unittest.TestCase):
    """Defects the pre-PR review found, each pinned by the input that showed it."""

    def executor(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        head = build_repo_with(root, files)
        return tools.ToolExecutor(root, head)

    def test_a_non_ascii_path_is_found_and_not_shown_quoted(self):
        # Without core.quotePath=false git prints `"\303\274.py"`, so the file was silently missed.
        ex = self.executor({"ü.py": "def helper(x):\n    return x\n", "caller.py": "helper(1)\n"})
        self.assertIn("ü.py:1:", ex.run("grep", {"pattern": "def helper"}).text)
        self.assertNotIn('"\\303', ex.run("grep", {"pattern": "def helper"}).text)
        self.assertIn("return x", ex.run("read_file", {"path": "ü.py"}).text)

    def test_an_empty_file_reads_as_empty_not_as_an_error(self):
        ex = self.executor({"e.py": "", "x.py": "a = 1\n"})
        result = ex.run("read_file", {"path": "e.py"})
        self.assertFalse(result.error)
        self.assertIn("empty", result.text)

    def test_the_repository_root_is_a_directory_not_a_staged_file(self):
        ex = self.executor({"x.py": "a = 1\n"})
        for path in (".", "./"):
            result = ex.run("read_file", {"path": path})
            self.assertTrue(result.error)
            self.assertIn("repository root", result.text)
            self.assertNotIn("added or removed", result.text)

    def test_grep_stops_reading_git_output_at_its_limit_and_says_so(self):
        ex = self.executor({"many.txt": "\n".join(f"hit {i} " + "x" * 80 for i in range(5000)) + "\n"})
        with mock.patch.object(tools, "MAX_GREP_OUTPUT_BYTES", 2000):
            result = ex.run("grep", {"pattern": "^hit"})
        self.assertFalse(result.error)
        self.assertTrue(result.truncated)
        self.assertIn("at least", result.text)
        self.assertLessEqual(result.text.count("\n") + 1, tools.GREP_MAX_LINES + 1)

    def test_a_stopped_git_command_does_not_leave_a_wrong_exit_code(self):
        ex = self.executor({"many.txt": "\n".join("hit" for _ in range(3000)) + "\n"})
        done = ex._git_limited(["grep", "-n", "-e", "hit", ex.head, "--"], 100)
        assert done is not None
        self.assertTrue(done.over)
        self.assertEqual(done.returncode, 0)
        self.assertLessEqual(len(done.stdout), 100)
        self.assertFalse(done.stdout.endswith(b"\n"))
        self.assertEqual(ex.run("grep", {"pattern": "zzz"}).text, "no matches")  # the executor still works afterwards


class FoundByCopilot(unittest.TestCase):
    """Defects the Copilot review of the pull request found, each pinned by the input that showed it."""

    TAIL = "TAIL_MARKER_AT_THE_END"
    LONG_LINE = "x = compute(" + "argument, " * 40 + TAIL + ")"  # 435 characters; the part that matters is at the end

    def executor(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        return tools.ToolExecutor(root, build_repo_with(root, files))

    def test_a_long_line_is_clipped_visibly_and_the_result_says_it_was_cut(self):
        ex = self.executor({"long.py": f"def f():\n    {self.LONG_LINE}\n"})
        result = ex.run("read_file", {"path": "long.py"})
        extra = len("    " + self.LONG_LINE) - tools.MAX_LINE_CHARS
        self.assertTrue(result.truncated)
        self.assertFalse(result.error)
        self.assertIn(f"...[line cut: {extra} more characters]", result.text)
        self.assertNotIn(self.TAIL, result.text)  # the text is shorter, and now it says so

    def test_a_line_at_exactly_the_limit_is_not_clipped(self):
        ex = self.executor({"edge.py": "y" * tools.MAX_LINE_CHARS + "\n"})
        result = ex.run("read_file", {"path": "edge.py"})
        self.assertFalse(result.truncated)
        self.assertNotIn("line cut", result.text)
        self.assertIn("y" * tools.MAX_LINE_CHARS, result.text)

    def test_grep_flags_a_clipped_match_even_when_the_searched_text_was_in_the_cut_part(self):
        ex = self.executor({"long.py": f"def f():\n    {self.LONG_LINE}\n"})
        result = ex.run("grep", {"pattern": self.TAIL})
        self.assertFalse(result.error)
        self.assertTrue(result.truncated)
        self.assertIn("line cut:", result.text)
        self.assertNotIn(self.TAIL, result.text)

    def test_grep_of_short_lines_is_not_flagged(self):
        ex = self.executor({"short.py": "def f():\n    return 1\n"})
        result = ex.run("grep", {"pattern": "return"})
        self.assertFalse(result.truncated)

    def test_clip_line_and_has_line_cut_agree(self):
        self.assertEqual(tools.clip_line("short"), "short")
        self.assertFalse(tools.has_line_cut(tools.clip_line("short")))
        clipped = tools.clip_line("z" * (tools.MAX_LINE_CHARS + 7))
        self.assertTrue(clipped.endswith("...[line cut: 7 more characters]"))
        self.assertTrue(tools.has_line_cut(clipped))

    def test_a_file_is_never_buffered_whole_before_the_size_check(self):
        ex = self.executor({"big.txt": "A" * 100_000 + "\n"})
        buffered: list = []
        real = ex._git

        def spy(args):
            if args and args[0] == "show":
                buffered.append("buffered whole by _git")
            return real(args)

        ex._git = spy
        with mock.patch.object(tools, "MAX_FILE_BYTES", 2000):
            result = ex.run("read_file", {"path": "big.txt"})
        self.assertEqual(buffered, [], "`git show` must go through the bounded read, not `_git`, which buffers everything")
        self.assertTrue(result.error)
        self.assertIn("larger than 2000 bytes", result.text)

    def test_a_file_exactly_at_the_limit_is_still_read(self):
        content = "B" * 1999 + "\n"  # 2000 bytes
        ex = self.executor({"edge.txt": content})
        with mock.patch.object(tools, "MAX_FILE_BYTES", 2000):
            result = ex.run("read_file", {"path": "edge.txt"})
        self.assertFalse(result.error, result.text)
        self.assertIn("B" * 300, result.text)

    def test_an_oversized_file_is_refused_once_and_not_retried(self):
        ex = self.executor({"big.txt": "A" * 100_000 + "\n"})
        calls: list = []
        real = ex._git_limited

        def spy(args, max_bytes):
            calls.append(args[0])
            return real(args, max_bytes)

        ex._git_limited = spy
        with mock.patch.object(tools, "MAX_FILE_BYTES", 2000):
            for _ in range(3):
                self.assertTrue(ex.run("read_file", {"path": "big.txt"}).error)
        self.assertEqual(calls.count("show"), 1)


class FailuresAreNotRemembered(unittest.TestCase):
    """A lookup that failed once must not be cached as an answer."""

    def executor(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.head = build_repo_with(root, files)
        return tools.ToolExecutor(root, self.head)

    def test_a_failed_list_of_files_is_not_cached_as_no_files(self):
        ex = self.executor({"a.txt": "one\ntwo\n"})
        with mock.patch.object(ex, "_git", lambda *args, **kwargs: None):  # ls-tree times out
            during = ex.run("read_file", {"path": "a.txt"})
        self.assertTrue(during.error)
        self.assertIn("could not be checked", during.text)
        self.assertNotIn("not a tracked file", during.text)
        after = ex.run("read_file", {"path": "a.txt"})  # git works again: the failure was not remembered
        self.assertFalse(after.error, after.text)
        self.assertIn("one", after.text)


class LanguageAgnostic(unittest.TestCase):
    """The tools find and return text; they never interpret code as any language (decision of 2026-11-08)."""

    FILES = {
        "util.c": "int helper(int x)\n{\n    return x * 2;\n}\n",
        "Service.cs": "public class Service\n{\n    public int Helper(int x) { return x * 2; }\n}\n",
        "Util.java": "class Util {\n    static int helper(int x) { return x * 2; }\n}\n",
        "run.sh": "helper() {\n  echo $((1 * 2))\n}\n",
        "calc.py": "def helper(x):\n    return x * 2\n",
    }

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.ex = tools.ToolExecutor(root, build_repo_with(root, self.FILES))

    def test_grep_finds_a_function_in_every_language_the_same_way(self):
        for path, needle in (("util.c", "int helper"), ("Service.cs", "int Helper"), ("Util.java", "static int helper"), ("run.sh", "helper\\(\\)"), ("calc.py", "def helper")):
            result = self.ex.run("grep", {"pattern": needle})
            self.assertFalse(result.error, result.text)
            self.assertIn(f"{path}:", result.text)

    def test_read_file_returns_any_languages_text_unchanged_apart_from_numbering(self):
        for path, text in self.FILES.items():
            result = self.ex.run("read_file", {"path": path})
            self.assertFalse(result.error, result.text)
            for line in text.splitlines():
                self.assertIn(line, result.text)

    def test_there_are_no_tools_that_read_code_as_a_language(self):
        self.assertEqual(tools.TOOL_NAMES, ("read_file", "grep"))
        self.assertTrue(self.ex.run("definitions", {"names": ["helper"]}).error)
        self.assertTrue(self.ex.run("callers", {"function": "helper"}).error)
