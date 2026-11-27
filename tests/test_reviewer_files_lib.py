#!/usr/bin/env python3
"""Tests for libs/reviewer_files_lib.py: the findings file a stage is run from and the results file it writes.

What they pin down:
  * a finding is one dict in both files, and it reads back exactly (alternates included);
  * the findings-file reader reports EVERY problem at once, names the path, and never repeats a value from the file (the path comes
    from a model, so the file can be any file the server can read), and refuses what a stage must not be given (a rejected finding for
    the classifier, alternates for the classifier);
  * a file that is not a regular file, is too big, is not UTF-8 or is nested too deeply is refused, not read or crashed on;
  * the results file is written whole, never overwrites another review's, and is checked before a review runs.

Stdlib-only (unittest), no network, no LM Studio.

    .venv/bin/python -m unittest discover -s tests
"""

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

import reviewer_files_lib as files  # noqa: E402
from classifier_reviewer_lib import Classification, Rating  # noqa: E402
from codebase_reviewer_config_lib import ConfigError  # noqa: E402
from reasoning_reviewer_lib import Alternate, Finding, ReviewError  # noqa: E402
from verifier_reviewer_lib import Decision  # noqa: E402

SECRET = "hunter2-do-not-repeat"


def finding(claim="runs the input in a shell", cls="injection", *, file="a.py", start=4, end=4, quote="subprocess.run(cmd, shell=True)",
            failing="cmd='a; b'", reviewers=("r1",), alternates=()) -> Finding:
    return Finding(file, start, end, cls, claim, failing, reviewers, alternates, quote)


def entry(**over) -> dict:
    """A valid finding as a findings file holds it, with `over` applied (a key set to None is removed)."""
    data = {"file": "a.py", "line_start": 4, "line_end": 4, "class": "injection", "claim": "runs the input in a shell",
            "failing_input": "cmd='a; b'", "quote": "subprocess.run(cmd, shell=True)", "reviewers": ["r1"]}
    for key, value in over.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


class InTempDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, content, name="findings.json") -> str:
        path = self.dir / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return str(path)


class FindingAsADict(unittest.TestCase):
    def test_a_finding_reads_back_exactly_through_json(self):
        original = finding(alternates=(Alternate("a second claim", "other input", ("r2", "r3"), 5, 6, "items.append(1)"),),
                           reviewers=("r1", "r2"))
        again = files.finding_from_dict(json.loads(json.dumps(files.finding_to_dict(original))))
        self.assertEqual(again, original)

    def test_the_dict_has_every_field_a_stage_needs_and_alternates_only_when_there_are_some(self):
        plain = files.finding_to_dict(finding())
        self.assertEqual(sorted(plain), ["claim", "class", "failing_input", "file", "line_end", "line_start", "quote", "reviewers"])
        self.assertNotIn("alternates", plain)
        with_alt = files.finding_to_dict(finding(alternates=(Alternate("c", "f", ("r2",), 1, 1, "q"),)))
        self.assertEqual(with_alt["alternates"], [{"line_start": 1, "line_end": 1, "claim": "c", "failing_input": "f", "quote": "q",
                                                  "reviewers": ["r2"]}])

    def test_line_end_defaults_to_line_start(self):
        read = files.finding_from_dict(entry(line_start=7, line_end=None))
        self.assertEqual((read.line_start, read.line_end), (7, 7))

    def test_an_empty_quote_and_an_empty_failing_input_are_accepted(self):
        # Grounding drops a finding with no quote, with its own stated reason; a reviewer may also give no failing input.
        read = files.finding_from_dict(entry(quote="", failing_input=""))
        self.assertEqual((read.quote, read.failing_input), ("", ""))

    def test_a_line_number_of_zero_is_accepted_as_no_line(self):
        self.assertEqual(files.finding_from_dict(entry(line_start=0, line_end=None)).line_start, 0)

    def test_every_key_but_line_end_is_needed(self):
        for key in ("file", "line_start", "class", "claim", "failing_input", "quote", "reviewers"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as caught:
                    files.finding_from_dict(entry(**{key: None}))
                self.assertIn(f"missing {key!r}", str(caught.exception))

    def test_every_problem_is_reported_at_once(self):
        with self.assertRaises(ValueError) as caught:
            files.finding_from_dict({"file": 3, "line_start": True, "class": " ", "claim": [], "quote": None, "reviewers": [],
                                     "extra": 1}, "findings[2]")
        text = str(caught.exception)
        for part in ("unknown key(s) 'extra'", "'file' must be a string (got int)", "'line_start' must be a whole number",
                     "'class' must not be blank", "'claim' must be a string (got list)", "missing 'failing_input'",
                     "'quote' must be a string (got NoneType)", "'reviewers' must be a non-empty list"):
            self.assertIn(part, text)
        self.assertTrue(all(piece.startswith("findings[2]") for piece in text.split("; ")))

    def test_a_bool_a_float_or_a_huge_number_is_not_a_line_number(self):
        for bad in (True, 3.0, -1, files.MAX_LINE_NUMBER + 1, "4"):
            with self.subTest(line_start=bad):
                with self.assertRaises(ValueError):
                    files.finding_from_dict(entry(line_start=bad))

    def test_line_end_may_not_come_before_line_start(self):
        with self.assertRaises(ValueError) as caught:
            files.finding_from_dict(entry(line_start=9, line_end=3))
        self.assertIn("'line_end'", str(caught.exception))

    def test_reviewers_must_be_a_non_empty_list_of_names(self):
        for bad in ([], "r1", ["r1", 2], ["r1", " "], None):
            with self.subTest(reviewers=bad):
                with self.assertRaises(ValueError):
                    files.finding_from_dict(entry(reviewers=bad))

    def test_alternates_are_checked_like_findings_and_named_by_position(self):
        good = {"line_start": 5, "claim": "c", "failing_input": "", "quote": "q", "reviewers": ["r2"]}
        read = files.finding_from_dict(entry(alternates=[good]))
        self.assertEqual(read.alternates, (Alternate("c", "", ("r2",), 5, 5, "q"),))
        with self.assertRaises(ValueError) as caught:
            files.finding_from_dict(entry(alternates=[good, {"claim": "c"}]), "findings[0]")
        self.assertIn("findings[0].alternates[1]: missing 'line_start'", str(caught.exception))
        with self.assertRaises(ValueError):
            files.finding_from_dict(entry(alternates="not a list"))

    def test_something_that_is_not_an_object_is_refused(self):
        for bad in ("text", 3, None, [entry()]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    files.finding_from_dict(bad)


class ReadFindingsFile(InTempDir):
    def test_a_verifier_file_is_read_in_file_order(self):
        path = self.write({"findings": [entry(claim="first"), entry(claim="second", line_start=6, line_end=6)]})
        read = files.read_findings_file(path, "verifier")
        self.assertEqual([f.claim for f in read.findings], ["first", "second"])
        self.assertEqual(read.statuses, ())

    def test_a_classifier_file_gives_each_finding_the_status_the_verifier_made_of_it(self):
        path = self.write({"findings": [entry(claim="a", verification="confirmed"), entry(claim="b", verification="disputed"),
                                        entry(claim="c", verification="unchecked")]})
        read = files.read_findings_file(path, "classifier")
        self.assertEqual(read.statuses, ("confirmed", "disputed", "unchecked"))
        self.assertEqual(len(read.findings), 3)

    def test_what_findings_document_writes_the_reader_reads_back(self):
        findings = [finding(claim="first", alternates=(Alternate("c", "f", ("r2",), 5, 6, "q"),)), finding(claim="second", start=0, end=0)]
        read = files.read_findings_file(self.write(files.findings_document(findings)), "verifier")
        self.assertEqual(list(read.findings), findings)
        plain = [finding(claim="first"), finding(claim="second")]
        read = files.read_findings_file(self.write(files.findings_document(plain, ["confirmed", "unchecked"])), "classifier")
        self.assertEqual((list(read.findings), read.statuses), (plain, ("confirmed", "unchecked")))

    def test_findings_document_refuses_statuses_that_do_not_match_the_findings(self):
        with self.assertRaises(ValueError):
            files.findings_document([finding()], ["confirmed", "confirmed"])

    def test_an_empty_list_is_a_file_with_no_findings(self):
        read = files.read_findings_file(self.write({"findings": []}), "verifier")
        self.assertEqual((read.findings, read.statuses), ((), ()))

    def test_the_verifier_reads_alternates_and_ignores_verification(self):
        alt = {"line_start": 5, "claim": "c", "failing_input": "", "quote": "q", "reviewers": ["r2"]}
        # `verification` is read only by the classifier: the verifier stage re-checks the finding whatever it says.
        read = files.read_findings_file(self.write({"findings": [entry(alternates=[alt], verification="rejected")]}), "verifier")
        self.assertEqual(len(read.findings[0].alternates), 1)
        self.assertEqual(read.statuses, ())

    def test_the_classifier_refuses_a_rejected_finding_a_missing_or_unknown_status_and_alternates(self):
        alt = {"line_start": 5, "claim": "c", "failing_input": "", "quote": "q", "reviewers": ["r2"]}
        path = self.write({"findings": [entry(verification="rejected"), entry(), entry(verification="maybe"),
                                        entry(verification=3), entry(verification="confirmed", alternates=[alt])]})
        with self.assertRaises(ReviewError) as caught:
            files.read_findings_file(path, "classifier")
        text = str(caught.exception)
        self.assertIn("findings[0]: the verifiers rejected it", text)
        self.assertIn("findings[1]: missing 'verification'", text)
        self.assertIn("findings[2]: 'verification' must be one of confirmed, disputed, unchecked", text)
        self.assertIn("findings[3]: 'verification' must be one of", text)
        self.assertIn("findings[4]: 'alternates' is not accepted here", text)
        self.assertIn("5 problems", text)

    def test_every_problem_is_reported_at_once_and_the_path_is_named(self):
        path = self.write({"findings": [entry(quote=None), entry(claim=7), "not an object", entry(reviewers=[])], "extra": 1})
        with self.assertRaises(ReviewError) as caught:
            files.read_findings_file(path, "verifier")
        text = str(caught.exception)
        self.assertIn(repr(path), text)
        for part in ("unknown top-level key(s) 'extra'", "findings[0]: missing 'quote'", "findings[1]: 'claim' must be a string (got int)",
                     "findings[2]: must be an object (got str)", "findings[3]: 'reviewers' must be a non-empty list", "5 problems"):
            self.assertIn(part, text)

    def test_no_message_ever_repeats_a_value_from_the_file(self):
        # The path comes from a model: whatever file it names, a message must not copy what is inside it.
        wrong_everywhere = {"file": {"x": SECRET}, "line_start": SECRET, "class": [SECRET], "claim": {"a": SECRET},
                            "failing_input": [SECRET], "quote": 5, "reviewers": [SECRET, 1], "bogus": SECRET}
        bad_alternate = {"claim": SECRET, "line_start": SECRET, "quote": [SECRET], "reviewers": SECRET}
        contents = [
            {"findings": [wrong_everywhere, entry(verification=SECRET, alternates=[bad_alternate]), SECRET]},
            {"findings": SECRET},
            {"findings": [], SECRET: SECRET},
        ]
        for stage in ("verifier", "classifier"):
            for content in contents:
                with self.subTest(stage=stage, content=list(content)):
                    with self.assertRaises(ReviewError) as caught:
                        files.read_findings_file(self.write(content), stage)
                    message = str(caught.exception)
                    # A key NAME may be shown (cut short), so the one file that names a key as long as the secret is left out of this check.
                    if SECRET not in content:
                        self.assertNotIn(SECRET, message)
                    else:
                        self.assertEqual(message.count(SECRET), 1)  # only as the unknown top-level key's name, never its value
        with self.assertRaises(ReviewError) as caught:
            files.read_findings_file(self.write(json.dumps(SECRET)), "verifier")
        self.assertNotIn(SECRET, str(caught.exception))

    def test_a_long_key_name_is_cut_in_the_message(self):
        path = self.write({"findings": [entry(**{"k" * 5000: 1})]})
        with self.assertRaises(ReviewError) as caught:
            files.read_findings_file(path, "verifier")
        self.assertLess(len(str(caught.exception)), 1000)
        self.assertIn("k" * files.KEY_CHARS + "...", str(caught.exception))

    def test_a_long_list_of_problems_is_cut_and_says_how_many_more(self):
        path = self.write({"findings": [entry(quote=None) for _ in range(files.MAX_PROBLEMS_SHOWN + 30)]})
        with self.assertRaises(ReviewError) as caught:
            files.read_findings_file(path, "verifier")
        text = str(caught.exception)
        self.assertEqual(text.count("missing 'quote'"), files.MAX_PROBLEMS_SHOWN)
        self.assertIn(f"... and {30} more", text)
        self.assertIn(f"({files.MAX_PROBLEMS_SHOWN + 30} problems)", text)

    def test_a_file_that_is_not_there_or_is_a_folder_is_refused(self):
        for path in (str(self.dir / "absent.json"), str(self.dir)):
            with self.subTest(path=path):
                with self.assertRaises(ReviewError) as caught:
                    files.read_findings_file(path, "verifier")
                self.assertIn("is not a file", str(caught.exception))
                self.assertIn(repr(path), str(caught.exception))

    def test_a_name_too_long_to_exist_is_an_error_not_a_crash(self):
        # `Path.is_file` raises for this (it answers False only for "not there"), and a model can name any string.
        for path in (str(self.dir / ("x" * 300)), "/" + "/".join(["d" * 200] * 40)):
            with self.subTest(length=len(path)):
                with self.assertRaises(ReviewError) as caught:
                    files.read_findings_file(path, "verifier")
                self.assertIn(repr(path), str(caught.exception))

    @unittest.skipIf(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), "needs POSIX permissions and a non-root user")
    def test_a_file_in_a_folder_the_server_may_not_enter_is_an_error_not_a_crash(self):
        locked = self.dir / "locked"
        locked.mkdir()
        (locked / "findings.json").write_text('{"findings": []}', encoding="utf-8")
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o700)
        with self.assertRaises(ReviewError) as caught:
            files.read_findings_file(str(locked / "findings.json"), "verifier")
        self.assertIn("cannot be read", str(caught.exception))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs named pipes")
    def test_a_pipe_is_refused_instead_of_waited_on_forever(self):
        fifo = self.dir / "pipe"
        os.mkfifo(fifo)
        with self.assertRaises(ReviewError) as caught:
            files.read_findings_file(str(fifo), "verifier")
        self.assertIn("is not a file", str(caught.exception))

    def test_a_file_over_the_limit_is_refused(self):
        path = self.write(b"[" + b" " * 200 + b"]")
        with mock.patch.object(files, "MAX_FINDINGS_FILE_BYTES", 100):
            with self.assertRaises(ReviewError) as caught:
                files.read_findings_file(path, "verifier")
        self.assertIn("is longer than 100 bytes", str(caught.exception))
        self.assertIn("is it the right path", str(caught.exception))

    def test_a_file_that_is_not_utf8_or_not_json_is_refused_without_its_text(self):
        for content, expected in ((b"\xff\xfe\x00bad", "is not UTF-8 text"), (b"{not json " + SECRET.encode(), "is not valid JSON"),
                                  (b"", "is not valid JSON")):
            with self.subTest(content=content[:8]):
                with self.assertRaises(ReviewError) as caught:
                    files.read_findings_file(self.write(content), "verifier")
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn(SECRET, str(caught.exception))

    def test_a_file_nested_too_deeply_to_read_is_refused_not_a_crash(self):
        deep = "[" * 200_000 + "]" * 200_000
        with self.assertRaises(ReviewError) as caught:
            files.read_findings_file(self.write(deep), "verifier")
        self.assertIn("is not valid JSON", str(caught.exception))

    def test_a_byte_order_mark_from_a_windows_tool_is_accepted(self):
        path = self.write(b"\xef\xbb\xbf" + json.dumps({"findings": [entry()]}).encode("utf-8"))
        self.assertEqual(len(files.read_findings_file(path, "verifier").findings), 1)

    def test_the_top_level_must_be_an_object_with_a_findings_list(self):
        for content, expected in (([entry()], "must hold a JSON object with a 'findings' list (got list)"),
                                  ({}, "missing 'findings'"), ({"findings": {"a": 1}}, "'findings' must be a list (got dict)"),
                                  ('"a string"', "must hold a JSON object with a 'findings' list (got str)")):
            with self.subTest(content=content):
                with self.assertRaises(ReviewError) as caught:
                    files.read_findings_file(self.write(content), "verifier")
                self.assertIn(expected, str(caught.exception))

    def test_a_leading_tilde_is_expanded(self):
        self.write({"findings": [entry()]}, "in-home.json")
        with mock.patch.dict(os.environ, {"HOME": str(self.dir), "USERPROFILE": str(self.dir)}):
            read = files.read_findings_file("~/in-home.json", "verifier")
        self.assertEqual(len(read.findings), 1)

    def test_a_stage_that_takes_no_findings_file_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            files.read_findings_file(self.write({"findings": []}), "reasoning")

    def test_the_findings_are_returned_as_written_with_nothing_merged_or_corrected(self):
        # Two findings on the same lines of one class are two findings here: merging is the reasoning stage's job.
        path = self.write({"findings": [entry(claim="same place"), entry(claim="same place again", file="./a.py", line_start=99, line_end=99)]})
        read = files.read_findings_file(path, "verifier")
        self.assertEqual([(f.file, f.line_start) for f in read.findings], [("a.py", 4), ("./a.py", 99)])


class SuppliedReasoning(unittest.TestCase):
    def test_the_findings_stay_as_given_grouped_by_class_without_merging(self):
        a, b, c = finding(claim="a"), finding(claim="b"), finding(claim="c", cls="style")
        review = files.supplied_reasoning([a, b, c], "verifier")
        self.assertEqual({g.cls: g.findings for g in review.groups}, {"injection": (a, b), "style": (c,)})
        self.assertEqual(review.finding_count, 3)
        self.assertEqual(review.chunks, 0)

    def test_the_reviewers_are_the_ones_the_findings_name_in_the_order_they_first_appear(self):
        alt = Alternate("x", "", ("r9", "r1"), 1, 1, "q")
        review = files.supplied_reasoning([finding(reviewers=("r2", "r1")), finding(reviewers=("r3",), alternates=(alt,))], "verifier")
        self.assertEqual(review.reviewers_run, ["r2", "r1", "r3", "r9"])

    def test_the_warning_says_where_the_findings_came_from_and_which_stages_did_not_run(self):
        self.assertEqual(files.supplied_reasoning([], "verifier").warnings,
                         ["The findings were read from a findings file: the reasoning stage did not run."])
        self.assertEqual(files.supplied_reasoning([], "classifier").warnings,
                         ["The findings were read from a findings file: the reasoning and verifier stages did not run."])


class ResultReaders(unittest.TestCase):
    def test_a_decision_and_a_classification_read_back_from_their_dicts_through_json(self):
        decision = Decision("v1", "confirmed", "cmd='a; b'", "runs b", "traced")
        entry_ = {"name": "v1", "decisions": [json.loads(json.dumps(files.decision_to_dict(decision, 3)))]}
        self.assertEqual(files.verifier_decisions(entry_), {3: decision})
        answer = Classification("c1", (Rating("injection", 0.1 + 0.2), Rating("crash", 1 / 3)))
        entry_ = {"name": "c1", "answers": [json.loads(json.dumps(files.classification_to_dict(answer, 0)))]}
        # JSON keeps a float whole, so an average over what was saved is the average of the run.
        self.assertEqual(files.classifier_answers(entry_), {0: answer})

    def test_a_reviewer_that_failed_has_nothing_to_read(self):
        failed = {"name": "x", "error": "ReviewError: every request failed"}
        self.assertEqual(files.reasoning_findings(failed), [])
        self.assertEqual(files.verifier_decisions(failed), {})
        self.assertEqual(files.classifier_answers(failed), {})


class WriteResults(InTempDir):
    RESULTS = {"version": files.RESULTS_VERSION, "type": "verifier", "run_id": "abc123", "head": "0" * 40, "steps": ["ground"]}

    def test_the_file_is_named_for_the_type_and_the_run_and_holds_the_results(self):
        path = files.write_results(self.dir, dict(self.RESULTS))
        self.assertEqual(path, self.dir / "verifier-abc123.json")
        self.assertEqual(files.results_file_name("verifier", "abc123"), "verifier-abc123.json")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.RESULTS)

    def test_two_runs_in_one_folder_do_not_overwrite_each_other(self):
        first = files.write_results(self.dir, dict(self.RESULTS, run_id="one"))
        second = files.write_results(self.dir, dict(self.RESULTS, run_id="two"))
        self.assertNotEqual(first, second)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["verifier-one.json", "verifier-two.json"])

    def test_no_temporary_file_is_left_and_a_json_listing_only_meets_whole_files(self):
        files.write_results(self.dir, dict(self.RESULTS))
        self.assertEqual([p.name for p in self.dir.iterdir()], ["verifier-abc123.json"])

    def test_a_failed_write_leaves_neither_a_file_nor_a_temporary_one(self):
        # NaN is not valid JSON: it is refused rather than written into a file another tool would not read.
        with self.assertRaises(ValueError):
            files.write_results(self.dir, dict(self.RESULTS, bad=float("nan")))
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_non_ascii_text_survives_the_round_trip_including_a_lone_surrogate(self):
        text = "café ✓ \ud800"  # model text can hold anything, a lone surrogate included
        path = files.write_results(self.dir, dict(self.RESULTS, note=text))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["note"], text)

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_the_file_gets_the_default_permissions_not_owner_only_ones(self):
        # A container may write the file and the host read it: a temp-file helper's owner-only mode would lock the reader out.
        old = os.umask(0o022)
        try:
            path = files.write_results(self.dir, dict(self.RESULTS))
        finally:
            os.umask(old)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o044, 0o044)


class PrepareResultsFolder(InTempDir):
    def test_a_missing_folder_is_created_with_its_parents(self):
        target = self.dir / "a" / "b"
        self.assertEqual(files.prepare_results_folder(str(target)), target)
        self.assertTrue(target.is_dir())

    def test_an_existing_folder_is_used_as_it_is(self):
        self.assertEqual(files.prepare_results_folder(str(self.dir)), self.dir)

    def test_a_path_that_is_a_file_is_a_config_error_naming_the_setting_and_the_path(self):
        blocker = self.dir / "file"
        blocker.write_text("x", encoding="utf-8")
        for target in (blocker, blocker / "below"):
            with self.subTest(target=str(target)):
                with self.assertRaises(ConfigError) as caught:
                    files.prepare_results_folder(str(target))
                self.assertIn("REVIEW_OUTPUT_DIR", str(caught.exception))
                self.assertIn(repr(str(target)), str(caught.exception))

    @unittest.skipIf(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), "needs POSIX permissions and a non-root user")
    def test_a_folder_that_cannot_be_written_to_is_a_config_error_before_anything_runs(self):
        locked = self.dir / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        self.addCleanup(locked.chmod, 0o700)
        with self.assertRaises(ConfigError) as caught:
            files.prepare_results_folder(str(locked))
        self.assertIn("cannot be written to", str(caught.exception))

    def test_a_leading_tilde_is_expanded(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.dir), "USERPROFILE": str(self.dir)}):
            self.assertEqual(files.prepare_results_folder("~/results"), self.dir / "results")


if __name__ == "__main__":
    unittest.main()
