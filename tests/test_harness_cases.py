#!/usr/bin/env python3
"""Tests for harness/cases.py: the seeded cases, clean cases and verifier fixtures every harness scores against.

A case with a wrong answer key (a marker that matches no line, a decoy standing on a bug) makes every score
built on it wrong without any error, so the answers are checked here once instead of by each run.

Stdlib-only (unittest), no network, no LM Studio.

    .venv/bin/python -m unittest discover -s tests
"""

import ast
import gc
import sys
import tempfile
import threading
import types
import unittest
import unittest.mock
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _directory in (REPO_ROOT / "harness", REPO_ROOT / "libs"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import cases as c  # noqa: E402
import harness_lib as h  # noqa: E402
import live_verifier as lv  # noqa: E402

# Sizes before the set grew: 2 seeded cases with 9 bugs, 1 clean case, 9 verifier fixtures.
OLD_BUGS, OLD_CLEAN_CASES, OLD_FIXTURES = 9, 1, 9


def seeded() -> list:
    return [case for case in c.ALL_CASES.values() if case.bugs]


def clean() -> list:
    return [case for case in c.ALL_CASES.values() if not case.bugs and not case.decoys]


def lines_of(case: c.Case, anchors: tuple) -> list:
    return [(anchor.id, anchor.file, c.line_of(case.head[anchor.file], anchor.marker)) for anchor in anchors]


class MarkerTests(unittest.TestCase):
    def test_every_bug_and_decoy_marker_is_on_exactly_one_line_of_its_file(self):
        for case in c.ALL_CASES.values():
            for anchor in (*case.bugs, *case.decoys):
                with self.subTest(case=case.name, anchor=anchor.id):
                    self.assertIn(anchor.file, case.head)
                    self.assertGreater(c.line_of(case.head[anchor.file], anchor.marker), 0)  # raises when 0 or 2+ lines match

    def test_no_decoy_marker_is_also_a_bug_marker(self):
        for case in c.ALL_CASES.values():
            bug_markers = {(bug.file, bug.marker) for bug in case.bugs}
            for decoy in case.decoys:
                with self.subTest(case=case.name, decoy=decoy.id):
                    self.assertNotIn((decoy.file, decoy.marker), bug_markers)

    def test_no_decoy_sits_within_the_scoring_tolerance_of_a_bug_or_another_decoy(self):
        # A finding is credited to the nearest anchor within LINE_TOLERANCE lines, so a decoy that close to
        # another anchor can be reported as a bug (or the reverse) with nothing to tell them apart.
        for case in c.ALL_CASES.values():
            bugs, decoys = lines_of(case, case.bugs), lines_of(case, case.decoys)
            for index, (decoy_id, decoy_file, decoy_line) in enumerate(decoys):
                others = [*bugs, *decoys[:index], *decoys[index + 1:]]
                for other_id, other_file, other_line in others:
                    if other_file == decoy_file:
                        with self.subTest(case=case.name, decoy=decoy_id, other=other_id):
                            self.assertGreater(abs(decoy_line - other_line), h.LINE_TOLERANCE)

    def test_no_two_anchors_share_a_line(self):
        for case in c.ALL_CASES.values():
            seen = [(file, line) for _id, file, line in lines_of(case, (*case.bugs, *case.decoys))]
            with self.subTest(case=case.name):
                self.assertEqual(len(seen), len(set(seen)))

    def test_a_marker_the_case_would_not_match_raises(self):
        case = c.Case("x", "", {}, {"a.py": "x = 1\nx = 1\n"}, bugs=(c.Bug("b", "a.py", "x = 1", "logic", "", ""),))
        with self.assertRaises(ValueError):
            c.bug_line(case, case.bugs[0])


class CaseShapeTests(unittest.TestCase):
    def test_ids_are_unique_across_every_case(self):
        # The classifier table is keyed on bare bug ids, and a verifier fixture id is <case>/<id>.
        bug_ids = [bug.id for case in c.ALL_CASES.values() for bug in case.bugs]
        decoy_ids = [decoy.id for case in c.ALL_CASES.values() for decoy in case.decoys]
        self.assertEqual(len(bug_ids), len(set(bug_ids)))
        self.assertEqual(len(decoy_ids), len(set(decoy_ids)))
        self.assertFalse(set(bug_ids) & set(decoy_ids))

    def test_every_python_file_parses_and_imports_only_the_standard_library(self):
        for case in c.ALL_CASES.values():
            packages = {path.split("/")[0].removesuffix(".py") for tree in (case.base, case.head) for path in tree}
            for tree in (case.base, case.head):
                for path, text in tree.items():
                    with self.subTest(case=case.name, path=path):
                        module = ast.parse(text, path)
                        for node in ast.walk(module):
                            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                            for name in names:
                                top = name.split(".")[0]
                                self.assertTrue(top in sys.stdlib_module_names or top in packages, f"{name} is neither stdlib nor part of the case")

    def test_a_seeded_case_changes_something(self):
        for case in c.ALL_CASES.values():
            with self.subTest(case=case.name):
                self.assertNotEqual(case.base, case.head)

    def test_a_clean_case_has_no_answers_and_none_of_its_seeded_siblings_bugs(self):
        for case in clean():
            self.assertEqual((case.bugs, case.decoys), ((), ()), case.name)
        sibling = {"clean-leaks": c.LEAKS, "clean-errors": c.ERRORS, "clean-concurrency": c.CONCURRENCY, "clean-guards-removed": c.GUARDS_REMOVED}
        for name, seeded_case in sibling.items():
            self.assertEqual(c.ALL_CASES[name].base, seeded_case.base, name)
            text = "\n".join(c.ALL_CASES[name].head.values())
            if name == "clean-guards-removed":
                # The seeded bug here is a MISSING guard, so its line survives; what must differ is that every guard is kept.
                for base_text in seeded_case.base.values():
                    guards = [line for line in base_text.splitlines() if "raise ValueError" in line]
                    self.assertTrue(guards and all(line in text.splitlines() for line in guards), f"{name} lost a guard")
                continue
            for bug in seeded_case.bugs:
                with self.subTest(case=name, bug=bug.id):
                    # a bug line that survives in the clean sibling would be reported as a false alarm that is really a bug
                    bug_line_text = next(line for line in seeded_case.head[bug.file].splitlines() if bug.marker in line)
                    self.assertNotIn(bug_line_text, text.splitlines())

    def test_the_deletion_only_cases_only_delete_lines(self):
        for name in ("guards-removed", "clean-guards-removed"):
            case = c.ALL_CASES[name]
            for path, text in case.head.items():
                remaining = iter(case.base[path].splitlines())
                # every head line must appear in base, in the same order, so the diff has no added line
                self.assertTrue(all(line in remaining for line in text.splitlines()), f"{name}:{path} adds or reorders a line")
                self.assertLess(len(text.splitlines()), len(case.base[path].splitlines()), name)


def run_module(case: c.Case, path: str):
    """The case's head file as a live module, so a test can check what a claim says the code does."""
    module = types.ModuleType(path.removesuffix(".py"))
    exec(compile(case.head[path], path, "exec"), module.__dict__)  # noqa: S102 - our own fixture text
    return module


class ClaimsHoldWhenRunTests(unittest.TestCase):
    """A seeded bug that does not misbehave scores a correct rejection as a miss, and a clean case that does misbehave
    scores a correct report as a false alarm, so the claims the cases make are run here instead of trusted."""

    def test_the_retained_handle_is_still_open_after_the_call_and_the_clean_sibling_leaves_none(self):
        # An early return that drops the last reference closes the file on CPython, so the seeded leak has to be one
        # that holds a reference: the handle must still be open once the call has returned.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "one.txt"
            path.write_text("first\nsecond\n", encoding="utf-8")
            seeded_module = run_module(c.LEAKS, "fileutil.py")
            self.assertEqual(seeded_module.first_line(str(path)), "first")
            held = list(seeded_module._recent)
            try:
                self.assertEqual(len(held), 1)
                self.assertFalse(held[0].closed)
            finally:
                for handle in held:
                    handle.close()
            clean_module = run_module(c.CLEAN_LEAKS, "fileutil.py")
            with warnings.catch_warnings():
                warnings.simplefilter("error", ResourceWarning)  # an unclosed file would raise here
                self.assertEqual(clean_module.first_line(str(path)), "first")
                gc.collect()

    def test_the_clean_registry_lets_a_factory_resolve_another_resource(self):
        # A non-reentrant lock held around the factory deadlocks this call: the inner get_or_create waits on the outer's lock.
        registry = run_module(c.CLEAN_CONCURRENCY, "registry.py")
        done: list = []
        worker = threading.Thread(
            target=lambda: done.append(registry.get_or_create("service", lambda: registry.get_or_create("db", object))), daemon=True)
        worker.start()
        worker.join(5)
        self.assertFalse(worker.is_alive(), "nested get_or_create deadlocked")
        self.assertEqual(len(done), 1)


class SizeTests(unittest.TestCase):
    """The comparison stops resting on one case only while the set stays several times its old size."""

    def test_the_set_is_at_least_three_times_the_old_size(self):
        self.assertGreaterEqual(sum(len(case.bugs) for case in c.ALL_CASES.values()), 3 * OLD_BUGS)
        self.assertGreaterEqual(len(clean()), 3 * OLD_CLEAN_CASES)
        self.assertGreaterEqual(len(lv.verifier_cases_for(list(c.fixture_cases().values()))), 3 * OLD_FIXTURES)

    def test_it_covers_the_defect_kinds_and_shapes_the_ticket_names(self):
        classes = {bug.cls for case in c.ALL_CASES.values() for bug in case.bugs}
        for kind in ("shell-injection", "resource-leak", "off-by-one", "error-handling", "race-condition"):
            self.assertIn(kind, classes)
        self.assertTrue(any(len(case.head) > 1 and case.bugs for case in c.ALL_CASES.values()), "no multi-file seeded case")
        self.assertTrue(any(case.bugs and len(case.head) == 1 for case in c.ALL_CASES.values()), "no one-file seeded case")
        self.assertIn("guards-removed", c.ALL_CASES)


class VerifierFixtureTests(unittest.TestCase):
    def test_every_fixture_id_is_unique_and_every_quote_is_in_the_merged_tree(self):
        every = list(c.fixture_cases().values())
        fixtures = c.verifier_cases_for(every)
        self.assertEqual(len({f.id for f in fixtures}), len(fixtures))
        tree = c.merged_case(every).head
        bug_ids = {f"{case.name}/{bug.id}" for case in every for bug in case.bugs}
        for fixture in fixtures:
            with self.subTest(fixture=fixture.id):
                self.assertIn(fixture.quote, tree[fixture.file].splitlines()[fixture.line - 1])
                self.assertEqual(fixture.real, fixture.id in bug_ids)

    def test_every_decoy_fixture_carries_a_failing_input(self):
        for fixture in c.verifier_cases_for(list(c.fixture_cases().values())):
            if not fixture.real:
                with self.subTest(fixture=fixture.id):
                    self.assertTrue(fixture.failing_input)

    def test_a_shared_path_with_different_text_is_refused(self):
        one = c.Case("one", "", {}, {"a.py": "x = 1\n"}, bugs=(c.Bug("b", "a.py", "x", "logic", "", ""),))
        two = c.Case("two", "", {}, {"a.py": "x = 2\n"}, bugs=(c.Bug("b2", "a.py", "x", "logic", "", ""),))
        with self.assertRaises(ValueError):
            c.merged_case([one, two])

    def test_a_shared_path_with_the_same_text_is_merged_once(self):
        one = c.Case("one", "", {}, {"a.py": "x = 1\n"}, bugs=(c.Bug("b", "a.py", "x", "logic", "", ""),))
        two = c.Case("two", "", {}, {"a.py": "x = 1\n"}, decoys=(c.Decoy("d", "a.py", "x", "", ""),))
        merged = c.merged_case([one, two])
        self.assertEqual(merged.head, {"a.py": "x = 1\n"})
        self.assertEqual((len(merged.bugs), len(merged.decoys)), (1, 1))

    def test_one_case_is_returned_as_it_is(self):
        self.assertIs(c.merged_case([c.ORDERS]), c.ORDERS)


class VerifierScriptTests(unittest.TestCase):
    def args(self, *argv: str):
        return lv.parse_args(["--check", *argv])

    def test_the_default_is_still_the_orders_fixtures_the_baselines_were_measured_on(self):
        self.assertEqual([case.name for case in lv.chosen_cases(self.args())], ["orders"])
        self.assertIn("9 verifier fixture(s): 6 real finding(s) to confirm, 3 false finding(s) to reject", lv.summarise(self.args()))

    def test_all_selects_every_case_that_has_answers(self):
        self.assertEqual([case.name for case in lv.chosen_cases(self.args("--case", "all"))], list(c.fixture_cases()))

    def test_case_can_be_repeated_and_a_repeat_counts_once(self):
        chosen = lv.chosen_cases(self.args("--case", "paging", "--case", "orders", "--case", "paging"))
        self.assertEqual([case.name for case in chosen], ["paging", "orders"])

    def test_a_case_with_no_answers_is_not_a_choice(self):
        with self.assertRaises(SystemExit):
            with unittest.mock.patch("sys.stderr"):
                self.args("--case", "clean")

    def test_a_score_counts_over_the_chosen_cases(self):
        chosen = c.verifier_cases_for(lv.chosen_cases(self.args("--case", "paging")))
        verdicts = {f.id: f.real for f in chosen}  # a verifier that is right about every fixture
        text = lv.score_verdicts(verdicts, chosen)
        self.assertIn("real findings confirmed: 4/4", text)
        self.assertIn("false findings rejected: 0/0", text)


if __name__ == "__main__":
    unittest.main()
