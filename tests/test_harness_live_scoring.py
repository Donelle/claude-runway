#!/usr/bin/env python3
"""Tests for harness/live_scoring_lib.py: the answer key of the `harness-live-tests` branch and the rules a review is scored by.

No Docker, no LM Studio, no network, and no clone of the branch: the real key file is read as it is shipped, and findings are
canned. The module is also the one the old runners (`live_branch_lib.py`) and the Pi scorer share, so a few tests pin that they
use this implementation and not a copy.

Stdlib-only (unittest, no pytest).

    .venv/bin/python -m unittest discover -s tests
"""

import ast
import collections
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "harness"
for _directory in (REPO_ROOT / "libs", HARNESS):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import harness_lib  # noqa: E402
import live_branch_lib  # noqa: E402
import live_scoring_lib as s  # noqa: E402

Finding = collections.namedtuple("Finding", "file line_start line_end cls claim")


def finding(file="libs/a.py", start=10, end=None, text="a claim") -> Finding:
    return Finding(file, start, start if end is None else end, "c", text)


def bug(id="B1", file="libs/a.py", line=10, evidence="other-file") -> s.LiveBug:
    return s.LiveBug(id, file, line, "logic", "the claim", evidence)


def decoy(id="D1", file="libs/a.py", line=30, evidence="diff") -> s.LiveDecoy:
    return s.LiveDecoy(id, file, line, "the claim", evidence)


TAG = s.LiveTag("change-1", (bug(),), (decoy(),))
CLEAN = s.LiveTag("change-5", clean=True)


class TheShippedAnswerKey(unittest.TestCase):
    def setUp(self):
        self.tags = s.load_answer_key()

    def test_it_lists_the_six_changes_with_the_totals_harness_readme_gives(self):
        self.assertEqual([t.tag_id for t in self.tags], [f"change-{n}" for n in range(1, 7)])
        self.assertEqual([t.tag_id for t in self.tags if t.clean], ["change-5"])
        bugs = collections.Counter(b.evidence for t in self.tags for b in t.bugs)
        decoys = collections.Counter(d.evidence for t in self.tags for d in t.decoys)
        # harness/README.md, "Totals": 7 bugs (1 diff, 3 same-file, 3 other-file) and 4 decoys (1, 2, 1).
        self.assertEqual((dict(bugs), dict(decoys)), ({"diff": 1, "same-file": 3, "other-file": 3}, {"diff": 1, "same-file": 2, "other-file": 1}))

    def test_change_six_is_one_bug_and_one_decoy_both_judgeable_from_the_diff(self):
        six = next(t for t in self.tags if t.tag_id == "change-6")
        self.assertEqual([(b.id, b.file, b.line, b.evidence) for b in six.bugs], [("B7", "libs/version_lib.py", 35, "diff")])
        self.assertEqual([(d.id, d.file, d.line, d.evidence) for d in six.decoys], [("D4", "libs/version_lib.py", 30, "diff")])

    def test_it_is_the_same_key_as_each_live_sidecar_until_those_go(self):
        # The old runners read the key from `metadata.tags` of three sidecars and the Pi scorer from this file. Two copies of an
        # answer can drift without anything saying so, which is what this pins; the cleanup ticket removes the sidecars, and
        # this test with them.
        sidecars = [p for p in sorted((HARNESS / "live-tests" / "configs").glob("*.baselines.json"))
                    if json.loads(p.read_text(encoding="utf-8")).get("runType") == "live"]
        self.assertTrue(sidecars, "no live sidecar is left: the key is no longer duplicated, so delete this test")
        key = json.loads(s.ANSWER_KEY_PATH.read_text(encoding="utf-8"))
        for path in sidecars:
            with self.subTest(sidecar=path.name):
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["metadata"], key)

    def test_it_is_on_this_side_of_the_repository_where_no_container_mount_reaches(self):
        # The Pi driver mounts `libs/` and `mcpservers/` of this checkout and nothing else of it. The key must not be in either,
        # and must not be on the branch (which has no harness/ at all; see tests/test_harness_pi_driver.py for the mounts).
        key = s.ANSWER_KEY_PATH.resolve()
        self.assertEqual(key, (HARNESS / "live-tests" / "answer_key.json").resolve())
        for mounted in ("libs", "mcpservers"):
            self.assertNotIn((REPO_ROOT / mounted).resolve(), key.parents)

    def test_a_bad_key_file_is_refused_naming_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer_key.json"
            for text in ("{", "[]", '{"tags": []}', '{"tags": [{"tagId": "change-1", "bugs": [], "decoys": []}]}'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(s.LiveMetadataError) as caught:
                    s.load_answer_key(path)
                self.assertIn(str(path), str(caught.exception))

    def test_a_missing_key_file_is_a_clear_error_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(s.LiveMetadataError) as caught:
            s.load_answer_key(Path(tmp) / "nope.json")
        self.assertIn("cannot be read", str(caught.exception))

    def test_a_key_that_is_not_utf8_is_refused_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer_key.json"
            path.write_bytes(b'{"tags": "\x80"}')
            with self.assertRaises(s.LiveMetadataError):
                s.load_answer_key(path)


class ParsingIsStrict(unittest.TestCase):
    def refused(self, data) -> str:
        with self.assertRaises(s.LiveMetadataError) as caught:
            s.parse_live_metadata(data)
        return str(caught.exception)

    def good(self) -> dict:
        return {"tags": [
            {"tagId": "change-1", "bugs": [{"id": "B1", "file": "a.py", "line": 3, "class": "logic", "claim": "x", "evidence": "diff"}], "decoys": []},
            {"tagId": "change-5", "clean": True, "bugs": [], "decoys": []},
        ]}

    def test_a_good_key_reads(self):
        tags = s.parse_live_metadata(self.good())
        self.assertEqual([(t.tag_id, t.clean, len(t.bugs)) for t in tags], [("change-1", False, 1), ("change-5", True, 0)])
        self.assertEqual(tags[0].head, "harness-live-tests/change-1")

    def test_a_misspelt_key_is_refused_so_it_is_not_ignored(self):
        data = self.good()
        data["tags"][0]["bugs"][0]["lines"] = 3
        self.assertIn("unknown key(s) ['lines']", self.refused(data))

    def test_an_empty_tag_must_say_it_is_the_clean_control(self):
        data = self.good()
        data["tags"][1].pop("clean")
        self.assertIn('say "clean": true', self.refused(data))

    def test_a_clean_tag_with_a_bug_is_refused(self):
        data = self.good()
        data["tags"][1]["bugs"] = [dict(data["tags"][0]["bugs"][0], id="B9")]
        self.assertIn("'clean' is true but it lists", self.refused(data))

    def test_ids_are_unique_across_the_whole_file(self):
        data = self.good()
        data["tags"].insert(1, {"tagId": "change-2", "bugs": [dict(data["tags"][0]["bugs"][0])], "decoys": []})
        self.assertIn("used twice", self.refused(data))

    def test_a_tag_is_written_without_the_prefix_and_base_is_never_one(self):
        data = self.good()
        data["tags"][0]["tagId"] = "harness-live-tests/change-1"
        self.assertIn("without the 'harness-live-tests/' prefix", self.refused(data))
        data["tags"][0]["tagId"] = "base"
        self.assertIn("not a tag to review", self.refused(data))

    def test_a_line_is_a_whole_number_from_one_and_never_a_bool(self):
        for bad in (0, -1, 2.5, "3", True):
            data = self.good()
            data["tags"][0]["bugs"][0]["line"] = bad
            with self.subTest(line=bad):
                self.assertIn("'line'", self.refused(data))


class ScoringRules(unittest.TestCase):
    def test_a_finding_at_a_bugs_line_or_within_the_tolerance_is_a_hit(self):
        for line in (10, 13, 7):
            with self.subTest(line=line):
                score = s.score_tag([finding(start=line)], TAG)
                self.assertEqual((score.found, score.missed, score.on_decoys, score.unscored), (["B1"], [], [], []))

    def test_a_finding_beyond_the_tolerance_or_in_another_file_is_unscored_and_the_bug_is_missed(self):
        for case in (finding(start=s.LINE_TOLERANCE + 11), finding(file="libs/other.py", start=10)):
            with self.subTest(case=case):
                score = s.score_tag([case], TAG)
                self.assertEqual((score.found, score.missed, len(score.unscored), score.on_decoys), ([], ["B1"], 1, []))

    def test_a_finding_at_a_decoys_line_is_a_false_alarm(self):
        score = s.score_tag([finding(start=30)], TAG)
        self.assertEqual((score.found, score.on_decoys, score.unscored), ([], ["D1"], []))

    def test_every_finding_on_the_clean_change_is_a_false_alarm_whatever_it_says(self):
        score = s.score_tag([finding(), finding(file="libs/x.py", start=1), finding(start=0)], CLEAN)
        self.assertEqual((len(score.clean_false_positives), score.found, score.missed, score.on_decoys, score.unscored), (3, [], [], [], []))

    def test_a_ranges_gap_is_measured_to_its_nearer_end(self):
        self.assertEqual(s.finding_distance(finding(start=5, end=9), "libs/a.py", 10), 1)
        self.assertEqual(s.finding_distance(finding(start=5, end=6), "libs/a.py", 10), None)  # 4 lines away: past the tolerance
        self.assertEqual(s.finding_distance(finding(start=8, end=12), "libs/a.py", 10), 0)  # the line is inside the range

    def test_a_finding_with_no_line_points_nowhere(self):
        score = s.score_tag([finding(start=0)], TAG)
        self.assertEqual((score.found, len(score.unscored)), ([], 1))

    def test_a_finding_nearer_a_decoy_is_not_credited_to_a_bug_and_a_tie_goes_to_the_bug(self):
        tag = s.LiveTag("t", (bug("B", "f.py", 10),), (decoy("D", "f.py", 12),))
        self.assertEqual(s.score_tag([finding("f.py", 12)], tag).on_decoys, ["D"])
        self.assertEqual(s.score_tag([finding("f.py", 11)], tag).found, ["B"])

    def test_two_findings_on_one_bug_are_one_hit(self):
        self.assertEqual(s.score_tag([finding(start=10), finding(start=11, text="again")], TAG).found, ["B1"])

    def test_two_bugs_at_one_line_are_credited_one_to_one(self):
        tag = s.LiveTag("t", (bug("B", "f.py", 10), bug("C", "f.py", 10)))
        self.assertEqual(s.score_tag([finding("f.py", 10), finding("f.py", 10, text="other")], tag).found, ["B", "C"])

    def test_a_path_matches_by_its_tail_either_way_round(self):
        for reported in ("libs/a.py", "a.py", "checkout/libs/a.py"):
            with self.subTest(reported=reported):
                self.assertEqual(s.score_tag([finding(reported, 10)], TAG).found, ["B1"])
        self.assertEqual(s.score_tag([finding("xlibs/a.py", 10)], TAG).found, [])  # a longer name is not the same file

    def test_the_class_and_wording_a_model_chose_are_never_compared(self):
        odd = Finding("libs/a.py", 10, 10, "something-else-entirely", "words that match nothing")
        self.assertEqual(s.score_tag([odd], TAG).found, ["B1"])

    def test_totals_per_evidence_kind_are_averaged_over_the_runs(self):
        tags = (TAG, CLEAN)
        scores = {"change-1": [s.score_tag([finding(start=10), finding(start=30)], TAG), s.score_tag([], TAG)], "change-5": [s.score_tag([], CLEAN)] * 2}
        self.assertEqual(s.kind_totals(tags, scores), {"diff": (0, 0.0, 1, 0.5), "same-file": (0, 0.0, 0, 0.0), "other-file": (1, 0.5, 0, 0.0)})


class OneImplementationForEveryConsumer(unittest.TestCase):
    def test_the_old_runners_module_reaches_this_one_not_a_copy(self):
        for name in ("score_tag", "kind_totals", "parse_live_metadata", "missing_refs", "LiveTag", "LiveBug", "LiveDecoy", "TagScore",
                     "LiveMetadataError", "BASE_REF", "PREFIX", "FETCH_COMMAND"):
            with self.subTest(name=name):
                self.assertIs(getattr(live_branch_lib, name), getattr(s, name))

    def test_the_synthetic_cases_use_the_same_matching_rule(self):
        self.assertIs(harness_lib.LINE_TOLERANCE, s.LINE_TOLERANCE)
        self.assertIs(harness_lib.nearest_anchor, s.nearest_anchor)

    def test_the_module_imports_nothing_but_the_standard_library(self):
        tree = ast.parse((HARNESS / "live_scoring_lib.py").read_text(encoding="utf-8"))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        self.assertEqual(sorted(imported - set(sys.stdlib_module_names)), [])

    def test_reading_the_key_and_scoring_loads_neither_cases_nor_harness_lib(self):
        # In a new interpreter: this process has long since imported both for other tests.
        program = (
            "import sys\n"
            f"sys.path.insert(0, {str(HARNESS)!r})\n"
            "import live_scoring_lib as s\n"
            "tags = s.load_answer_key()\n"
            "s.score_tag([], tags[0])\n"
            "print(sorted(m for m in ('cases', 'harness_lib', 'live_branch_lib') if m in sys.modules))\n"
        )
        completed = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[]")


class FindingTheBranchMessage(unittest.TestCase):
    def test_it_names_the_checkout_the_missing_refs_and_the_exact_fetch_command(self):
        message = s.missing_refs_message(Path("/some/checkout"), ["harness-live-tests/change-6"])
        self.assertIn("/some/checkout does not have harness-live-tests/change-6", message)
        self.assertIn("git fetch origin harness-live-tests 'refs/tags/harness-live-tests/*:refs/tags/harness-live-tests/*'", message)
        self.assertEqual(s.FETCH_COMMAND, "git fetch origin harness-live-tests 'refs/tags/harness-live-tests/*:refs/tags/harness-live-tests/*'")


if __name__ == "__main__":
    unittest.main()
