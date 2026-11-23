#!/usr/bin/env python3
"""Tests for the `live` run type: harness/live_branch_lib.py, `--run-type` on the harness scripts, and how the
live-tests runner reads a live sidecar.

A tiny temporary git repository with `harness-live-tests/*` tags stands in for the real branch, and canned findings stand in
for a model, so nothing here needs the network or LM Studio.

Stdlib-only (unittest, no pytest).

    .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_TESTS = REPO_ROOT / "harness" / "live-tests"
for _directory in (LIVE_TESTS, REPO_ROOT / "libs", REPO_ROOT / "harness"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import harness_lib as h  # noqa: E402
import live_all_reviewers  # noqa: E402
import live_branch_lib as lb  # noqa: E402
import live_classifier  # noqa: E402
import live_configs as lc  # noqa: E402
import live_reasoning  # noqa: E402
import live_verifier  # noqa: E402
import run_live_tests as rlt  # noqa: E402

MODELS = "loaded before: ['x']\nloaded after:  ['x']\n"


def bug(id="B1", file="libs/a.py", line=10, evidence="other-file") -> dict:
    return {"id": id, "file": file, "line": line, "class": "wrong-argument-order", "claim": "the arguments are swapped", "evidence": evidence}


def decoy(id="D1", file="libs/a.py", line=30, evidence="diff") -> dict:
    return {"id": id, "file": file, "line": line, "claim": "looks racy and is not", "evidence": evidence}


def metadata() -> dict:
    """change-1 has an `other-file` bug and a `diff` decoy; change-6 a `same-file` bug; change-5 is the clean control."""
    return {"tags": [
        {"tagId": "change-1", "bugs": [bug()], "decoys": [decoy()]},
        {"tagId": "change-6", "bugs": [bug("B7", "libs/v.py", 35, "same-file")], "decoys": []},
        {"tagId": "change-5", "clean": True, "bugs": [], "decoys": []},
    ]}


def claim(file="libs/a.py", line=10, cls="c", text="a claim") -> h.Claim:
    return h.Claim(file, line, line, cls, text, ("r",))


def tags() -> tuple:
    return lb.parse_live_metadata(metadata())


def by_id(tag_id: str) -> lb.LiveTag:
    return next(t for t in tags() if t.tag_id == tag_id)


class Folder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)


class ParseMetadata(unittest.TestCase):
    def refused(self, data) -> str:
        with self.assertRaises(lb.LiveMetadataError) as caught:
            lb.parse_live_metadata(data)
        return str(caught.exception)

    def test_reads_every_tag_with_its_bugs_decoys_and_the_clean_property(self):
        found = tags()
        self.assertEqual([t.tag_id for t in found], ["change-1", "change-6", "change-5"])
        self.assertEqual((found[0].bugs[0].id, found[0].bugs[0].evidence, found[0].decoys[0].id), ("B1", "other-file", "D1"))
        self.assertEqual([t.clean for t in found], [False, False, True])
        self.assertEqual(found[0].head, "harness-live-tests/change-1")

    def test_a_tag_that_lists_nothing_must_say_it_is_clean(self):
        data = metadata()
        data["tags"][2].pop("clean")
        self.assertIn('say "clean": true', self.refused(data))

    def test_a_clean_tag_cannot_list_bugs_or_decoys(self):
        data = metadata()
        data["tags"][2]["bugs"] = [bug("B9")]
        self.assertIn("'clean' is true but it lists", self.refused(data))

    def test_a_tag_id_is_written_without_the_prefix_and_is_never_base(self):
        data = metadata()
        data["tags"][0]["tagId"] = "harness-live-tests/change-1"
        self.assertIn("without the 'harness-live-tests/' prefix", self.refused(data))
        data["tags"][0]["tagId"] = "base"
        self.assertIn("not a tag to review", self.refused(data))

    def test_duplicates_and_bad_values_are_refused_naming_the_key(self):
        data = metadata()
        data["tags"][1]["bugs"][0]["id"] = "B1"
        self.assertIn("used twice", self.refused(data))
        data = metadata()
        data["tags"].append(dict(data["tags"][0]))
        self.assertIn("listed twice", self.refused(data))
        data = metadata()
        data["tags"][0]["bugs"][0]["evidence"] = "elsewhere"
        self.assertIn("'evidence'", self.refused(data))
        data = metadata()
        data["tags"][0]["bugs"][0]["line"] = 0
        self.assertIn("'line'", self.refused(data))
        data = metadata()
        data["tags"][0]["bugs"][0]["line"] = True
        self.assertIn("'line'", self.refused(data))

    def test_an_unknown_key_is_refused_so_a_typo_is_not_ignored(self):
        data = metadata()
        data["tags"][0]["bugs"][0]["lines"] = 10
        self.assertIn("unknown key(s) ['lines']", self.refused(data))
        data = metadata()
        data["case"] = ["orders"]
        self.assertIn("unknown key(s)", self.refused(data))

    def test_no_tags_is_refused(self):
        for data in ({}, {"tags": []}, {"tags": "change-1"}, None):
            with self.subTest(data=data):
                self.assertIn("needs 'tags'", self.refused(data))

    def test_only_live_bars_are_allowed_in_expect(self):
        self.assertIsNone(lb.check_bars({"min_other_file_found": 1, "max_clean_false_positives": 0}))
        self.assertIn("bugs_found_of_38", lb.check_bars({"bugs_found_of_38": 3}))


class ScoreTag(unittest.TestCase):
    def test_a_finding_at_a_bugs_line_is_a_hit_within_the_tolerance(self):
        for line in (10, 13, 7):
            with self.subTest(line=line):
                score = lb.score_tag([claim(line=line)], by_id("change-1"))
                self.assertEqual((score.found, score.missed, score.on_decoys, score.unscored), (["B1"], [], [], []))

    def test_a_finding_beyond_the_tolerance_or_in_another_file_is_unscored_not_wrong(self):
        for finding in (claim(line=14), claim(file="libs/other.py", line=10)):
            with self.subTest(finding=finding):
                score = lb.score_tag([finding], by_id("change-1"))
                self.assertEqual((score.found, score.missed, len(score.unscored)), ([], ["B1"], 1))

    def test_a_finding_at_a_decoys_line_is_a_false_positive(self):
        score = lb.score_tag([claim(line=30)], by_id("change-1"))
        self.assertEqual((score.found, score.on_decoys, score.unscored), ([], ["D1"], []))

    def test_any_finding_on_the_clean_control_is_a_false_positive(self):
        score = lb.score_tag([claim(), claim(file="libs/x.py", line=1)], by_id("change-5"))
        self.assertEqual((len(score.clean_false_positives), score.found, score.unscored), (2, [], []))

    def test_a_finding_nearer_a_decoy_is_not_credited_to_a_bug_and_a_tie_goes_to_the_bug(self):
        close = lb.LiveTag("t", (lb.LiveBug("B", "f.py", 10, "c", "x", "diff"),), (lb.LiveDecoy("D", "f.py", 12, "x", "diff"),))
        self.assertEqual(lb.score_tag([claim("f.py", 12)], close).on_decoys, ["D"])
        self.assertEqual(lb.score_tag([claim("f.py", 11)], close).found, ["B"])

    def test_two_findings_on_one_bug_are_one_hit(self):
        score = lb.score_tag([claim(line=10), claim(line=11, text="again")], by_id("change-1"))
        self.assertEqual(score.found, ["B1"])

    def test_a_finding_with_no_line_matches_nothing(self):
        score = lb.score_tag([claim(line=0)], by_id("change-1"))
        self.assertEqual((score.found, len(score.unscored)), ([], 1))


class Reporting(unittest.TestCase):
    def scores(self) -> dict:
        one, six, five = by_id("change-1"), by_id("change-6"), by_id("change-5")
        return {"change-1": [lb.score_tag([claim(line=10), claim(line=30)], one), lb.score_tag([], one)],
                "change-6": [lb.score_tag([claim("libs/v.py", 35)], six), lb.score_tag([claim("libs/v.py", 35)], six)],
                "change-5": [lb.score_tag([claim()], five), lb.score_tag([], five)]}

    def test_totals_are_per_evidence_kind_averaged_over_the_runs(self):
        totals = lb.kind_totals(tags(), self.scores())
        # (bugs, found per run, decoys, alarms per run): other-file B1 found in 1 of 2 runs; same-file B7 in both; diff D1 alarmed once.
        self.assertEqual(totals, {"diff": (0, 0.0, 1, 0.5), "same-file": (1, 1.0, 0, 0.0), "other-file": (1, 0.5, 0, 0.0)})

    def test_the_summary_lines_are_what_the_runner_reads(self):
        text = "\n".join(lb.summary_lines(tags(), self.scores()))
        kinds = {m.group(1): m.groups()[1:] for m in rlt.LIVE_KIND_LINE.finditer(text)}
        self.assertEqual(kinds, {"diff": ("0", "0", "0.5", "1"), "same-file": ("1", "1", "0", "0"), "other-file": ("0.5", "1", "0", "0")})
        self.assertEqual(rlt.LIVE_CLEAN_LINE.search(text).group(1), "0.5")

    def test_a_fractional_average_is_printed_exactly_so_a_bar_met_exactly_is_not_failed_by_rounding(self):
        # Three runs: B1 (other-file) found once, the clean control raised once in two of them. Printed with a 6-digit
        # format these were 0.333333 and 0.666667, and a bar of exactly 1/3 (or 2/3) then failed.
        one, five = by_id("change-1"), by_id("change-5")
        scores = {"change-1": [lb.score_tag([claim(line=10)], one), lb.score_tag([], one), lb.score_tag([], one)],
                  "change-6": [lb.score_tag([], by_id("change-6"))] * 3,
                  "change-5": [lb.score_tag([claim()], five), lb.score_tag([claim()], five), lb.score_tag([], five)]}
        text = "\n".join(lb.summary_lines(tags(), scores))
        self.assertIn(f"live other-file: {1 / 3!r} of 1", text)
        self.assertIn(f"live clean: {2 / 3!r} false positive(s)", text)
        self.assertEqual({m.group(1): float(m.group(2)) for m in rlt.LIVE_KIND_LINE.finditer(text)}["other-file"], 1 / 3)
        self.assertEqual(float(rlt.LIVE_CLEAN_LINE.search(text).group(1)), 2 / 3)

    def test_whole_numbers_and_short_decimals_keep_their_short_form(self):
        self.assertEqual([lb._num(v) for v in (0.0, 1.0, 0.5, 2.0, 0.25)], ["0", "1", "0.5", "2", "0.25"])
        self.assertEqual(lb._num(1 / 3), repr(1 / 3))

    def test_the_report_has_a_block_per_tag_the_runner_can_find(self):
        report = lb.format_live_report(tags(), self.scores())
        self.assertEqual(rlt.LIVE_TAG_BLOCK.findall(report), ["change-1", "change-6", "change-5"])
        self.assertIn("partly B1", report)  # found in one of the two runs
        self.assertIn("false positives on the clean control: 1 over 2 run(s)", report)
        self.assertIn("unscored findings: 0 over 2 run(s)", report)

    def test_an_unscored_finding_is_listed_and_never_counted_against_the_model(self):
        scores = {t.tag_id: [lb.score_tag([claim(file="libs/zzz.py", line=5)] if t.tag_id == "change-1" else [], t)] for t in tags()}
        report = lb.format_live_report(tags(), scores)
        self.assertIn("unscored findings: 1 over 1 run(s)", report)
        self.assertIn("libs/zzz.py:5-5", report)
        self.assertEqual(lb.kind_totals(tags(), scores)["other-file"], (1, 0.0, 0, 0.0))


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args], check=True, capture_output=True)


def make_repo(folder: Path, tag_ids=("change-1", "change-5", "change-6")) -> Path:
    """A throwaway repository with `harness-live-tests/base` and one commit per tag on top of it."""
    repo = folder / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "f.txt").write_text("base\n", encoding="utf-8")
    git(repo, "add", "f.txt")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "tag", lb.BASE_REF)
    for tag_id in tag_ids:
        (repo / "f.txt").write_text(f"{tag_id}\n", encoding="utf-8")
        git(repo, "commit", "-q", "-am", tag_id)
        git(repo, "tag", lb.PREFIX + tag_id)
    return repo


class FindingTheBranch(Folder):
    def test_every_ref_present_passes(self):
        repo = make_repo(self.dir)
        self.assertEqual(lb.missing_refs(repo, tags()), [])
        lb.require_refs(repo, tags())

    def test_a_missing_tag_stops_the_run_with_the_exact_git_fetch_command(self):
        repo = make_repo(self.dir, tag_ids=("change-1", "change-5"))
        with self.assertRaises(SystemExit) as caught:
            lb.require_refs(repo, tags())
        message = str(caught.exception)
        self.assertIn("harness-live-tests/change-6", message)
        self.assertNotIn("harness-live-tests/change-1,", message)
        self.assertIn("git fetch origin harness-live-tests 'refs/tags/harness-live-tests/*:refs/tags/harness-live-tests/*'", message)

    def test_a_checkout_with_no_base_is_missing_it_too(self):
        repo = self.dir / "empty"
        repo.mkdir()
        git(repo, "init", "-q")
        self.assertEqual(lb.missing_refs(repo, tags())[0], lb.BASE_REF)


def write_sidecar(config: Path, run_type="live", meta=None, expect=None) -> None:
    data = {"measured": "2026-11-08", "model": "m", "runType": run_type, "metadata": metadata() if meta is None else meta, "expect": expect or {}}
    lb.sidecar_path(str(config)).write_text(json.dumps(data), encoding="utf-8")


def write_config(folder: Path, stages=("reasoning",), name="t") -> Path:
    roles = tuple(stages)
    reviewers = [{"name": f"{r}-reviewer", "model": "some-model", "prompt": "p", "active": True, "type": [r], "loadOnDemand": True, "contextLength": 8192}
                 for r in roles]
    data: dict = {"stages": list(stages), "reviewers": reviewers}
    if "classifier" in roles:
        data["rubric"] = [{"name": "crash", "level": "high", "score": 0.6, "definition": "an input makes the code raise"}]
    path = folder / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class LiveTarget(Folder):
    def args(self, config=None, repo=None) -> types.SimpleNamespace:
        return types.SimpleNamespace(config=str(config) if config else None, repo=str(repo) if repo else None)

    def refused(self, args) -> str:
        with self.assertRaises(SystemExit) as caught:
            lb.live_target(args)
        return str(caught.exception)

    def test_no_config_has_no_answer_key(self):
        self.assertIn("needs --config", self.refused(self.args()))

    def test_no_sidecar_is_refused(self):
        self.assertIn("t.baselines.json next to the config", self.refused(self.args(write_config(self.dir))))

    def test_a_home_relative_config_path_finds_the_sidecar_beside_the_config(self):
        # `h.load_config` expands `~`, so a config given as `~/configs/review.json` loads; its sidecar must be found the same way.
        home = self.dir / "home"
        (home / "configs").mkdir(parents=True)
        config = write_config(home / "configs", name="review")
        write_sidecar(config)
        repo = make_repo(self.dir)
        with unittest.mock.patch.dict("os.environ", {"HOME": str(home), "USERPROFILE": str(home)}):
            self.assertTrue(lb.sidecar_path("~/configs/review.json").is_file())
            found_repo, found_tags = lb.live_target(self.args("~/configs/review.json", repo))
        self.assertEqual([t.tag_id for t in found_tags], ["change-1", "change-6", "change-5"])
        self.assertEqual(found_repo, repo.resolve())

    def test_the_missing_sidecar_message_names_where_it_looked(self):
        config = write_config(self.dir)
        message = self.refused(self.args(config))
        self.assertIn(f"Looked for {lb.sidecar_path(str(config))}", message)

    def test_a_synthetic_sidecar_is_refused_not_read_as_live(self):
        config = write_config(self.dir)
        write_sidecar(config, run_type="synthetic", meta={"cases": ["orders"]})
        self.assertIn("needs a 'live' one", self.refused(self.args(config)))

    def test_a_bad_live_metadata_names_the_file_and_the_key(self):
        config = write_config(self.dir)
        write_sidecar(config, meta={"tags": [{"tagId": "change-1", "bugs": [], "decoys": []}]})
        message = self.refused(self.args(config))
        self.assertIn("t.baselines.json", message)
        self.assertIn("change-1", message)

    def test_a_missing_tag_is_refused_with_the_fetch_command(self):
        config = write_config(self.dir)
        write_sidecar(config)
        repo = make_repo(self.dir, tag_ids=("change-1",))
        self.assertIn("git fetch", self.refused(self.args(config, repo)))

    def test_returns_the_repo_and_the_tags_in_the_sidecars_order(self):
        config = write_config(self.dir)
        write_sidecar(config)
        repo = make_repo(self.dir)
        found_repo, found_tags = lb.live_target(self.args(config, repo))
        self.assertEqual((found_repo, [t.tag_id for t in found_tags]), (repo.resolve(), ["change-1", "change-6", "change-5"]))


class RunTypeOption(Folder):
    def run_script(self, module, *argv) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        # `main` would make a throwaway reviewer-database folder and set the variables for what it runs: neither belongs to these
        # tests (tests/test_harness_reviewer_db.py covers it), so the helper is a no-op here and nothing is left in the temp folder.
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), unittest.mock.patch.object(h, "isolate_reviewer_db"):
            try:
                code = module.main(list(argv))
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_the_default_is_synthetic_and_nothing_changes_without_the_option(self):
        for module in (live_reasoning, live_all_reviewers, live_verifier, live_classifier):
            with self.subTest(script=module.__name__):
                self.assertEqual(module.parse_args([]).run_type, "synthetic")

    def test_the_verifier_and_classifier_scripts_say_a_live_run_is_not_available(self):
        for module, stage in ((live_verifier, "verifier"), (live_classifier, "classifier")):
            with self.subTest(stage=stage):
                code, _out, err = self.run_script(module, "--run-type", "live", "--check")
                self.assertEqual(code, 2)
                self.assertIn(f"--run-type live is not available for the {stage} stage", err)

    def test_a_synthetic_run_of_those_scripts_is_untouched(self):
        for module in (live_verifier, live_classifier):
            with self.subTest(script=module.__name__):
                code, out, _err = self.run_script(module, "--check")
                self.assertEqual(code, 0)
                self.assertTrue(out.strip())

    def test_a_live_run_refuses_case_and_history(self):
        for module, extra in ((live_reasoning, ["--case", "orders"]), (live_reasoning, ["--history", "x.json"]),
                              (live_all_reviewers, ["--case", "orders"]), (live_all_reviewers, ["--history", "x.json"])):
            with self.subTest(script=module.__name__, extra=extra):
                code, _out, _err = self.run_script(module, "--run-type", "live", *extra)
                self.assertIn("--case and --history belong to a synthetic run", str(code))

    def test_a_live_run_with_a_missing_tag_stops_before_any_model_is_checked_or_loaded(self):
        config = write_config(self.dir)
        write_sidecar(config)
        repo = make_repo(self.dir, tag_ids=("change-1",))
        for module in (live_reasoning, live_all_reviewers):
            with self.subTest(script=module.__name__):
                with unittest.mock.patch.object(h, "preflight_reviewers", side_effect=AssertionError("a model was checked")), \
                     unittest.mock.patch.object(h, "load_server", side_effect=AssertionError("LM Studio was contacted")):
                    code, _out, _err = self.run_script(module, "--config", str(config), "--run-type", "live", "--repo", str(repo))
                self.assertIn("git fetch origin harness-live-tests", str(code))

    def test_a_live_run_without_a_sidecar_stops_the_same_way(self):
        config = write_config(self.dir)
        code, _out, _err = self.run_script(live_reasoning, "--config", str(config), "--run-type", "live")
        self.assertIn("needs t.baselines.json", str(code))

    def test_a_synthetic_run_never_reads_a_sidecar(self):
        # `--show-config` is the run that needs no LM Studio: with a live sidecar beside it, it still resolves as before.
        config = write_config(self.dir)
        write_sidecar(config, meta={"not": "valid"})
        code, out, _err = self.run_script(live_reasoning, "--config", str(config), "--show-config")
        self.assertEqual(code, 0)
        self.assertIn("reasoning-reviewer", out)


class LiveReasoningRun(unittest.TestCase):
    def test_each_tag_is_reviewed_against_base_and_scored_from_the_sidecars_answers(self):
        calls = []

        def fake_run(repo, base, head, reviewers, args, recorder, **_kw):
            calls.append((str(repo), base, head))
            findings = {"harness-live-tests/change-1": [claim(line=10)], "harness-live-tests/change-6": [claim("libs/v.py", 35)],
                        "harness-live-tests/change-5": [claim()]}[head]
            return findings, types.SimpleNamespace(errors=[])

        out = io.StringIO()
        with unittest.mock.patch.object(h, "run_reasoning", fake_run), contextlib.redirect_stdout(out):
            live_reasoning.run_live(types.SimpleNamespace(runs=1), Path("/some/checkout"), tags(), None, [])
        # str(Path("/some/checkout")) is "\some\checkout" on Windows, so build the expected text the same way.
        checkout = str(Path("/some/checkout"))
        self.assertEqual(calls, [(checkout, "harness-live-tests/base", "harness-live-tests/change-1"),
                                 (checkout, "harness-live-tests/base", "harness-live-tests/change-6"),
                                 (checkout, "harness-live-tests/base", "harness-live-tests/change-5")])
        text = out.getvalue()
        self.assertIn("live other-file: 1 of 1 bug(s) found; 0 of 0 decoy alarm(s)", text)
        self.assertIn("live same-file: 1 of 1 bug(s) found", text)
        self.assertIn("live clean: 1 false positive(s) on a clean control per run", text)


class AllReviewersPipeline(Folder):
    """`live_all_reviewers.main` end to end with the model stages stubbed: the synthetic path must print what it printed before
    `--run-type` existed, and the live path must run the same stages once per tag."""

    def run_main(self, argv, findings_for, with_verifier=False) -> tuple:
        calls = []

        def fake_reasoning(repo, base, head, reviewers, args, recorder, **_kw):
            calls.append((base, head))
            review = types.SimpleNamespace(errors=[], reviewers_run=["r"], chunks=1, groups=[])
            return findings_for(head), review

        verification = types.SimpleNamespace(confirmed=[], disputed=[], unchecked=[], rejected=[], ungrounded=[], verifiers_run=["v"],
                                             errors=[], kept=[claim(line=10)], verified=[])
        out = io.StringIO()
        patches = [unittest.mock.patch.object(h, "preflight_reviewers"), unittest.mock.patch.object(h, "loaded_models", return_value=[]),
                   unittest.mock.patch.object(h, "load_server", return_value=types.SimpleNamespace(lmstudio_base_url=lambda: "http://x")),
                   unittest.mock.patch.object(h, "run_reasoning", fake_reasoning), unittest.mock.patch.object(h, "run_verifier", return_value=verification),
                   unittest.mock.patch.object(h, "report_run", return_value="the run report"),
                   unittest.mock.patch.object(live_all_reviewers, "new_run_recorder", lambda **_kw: contextlib.nullcontext()),
                   unittest.mock.patch.object(live_all_reviewers, "format_review", lambda *_a, **_kw: "the review"),
                   unittest.mock.patch.object(live_all_reviewers, "format_verified_review", lambda *_a, **_kw: "the verified review"),
                   unittest.mock.patch.object(live_all_reviewers, "describe_plan", lambda *_a, **_kw: "the plan")]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(unittest.mock.patch.object(h, "isolate_reviewer_db"))  # no throwaway folder, no variables set: see test_harness_reviewer_db.py
            code = live_all_reviewers.main(argv)
        return code, out.getvalue(), calls

    def test_without_the_option_the_orders_case_is_run_and_scored_as_before(self):
        code, text, calls = self.run_main(["--model", "m"], lambda head: [])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("case orders: 6 bug(s), 3 decoy(s), 1 run(s)", text)
        self.assertIn("recall 0%", text)
        self.assertNotIn("live", text.replace("the run report", ""))
        self.assertIn("the run report", text)

    def test_a_synthetic_run_with_a_verifier_prints_the_two_scores_as_before(self):
        config = write_config(self.dir, ("reasoning", "verifier"))
        code, text, _calls = self.run_main(["--config", str(config), "--case", "leaks"], lambda head: [])
        self.assertEqual(code, 0)
        self.assertLess(text.index("===== before verification"), text.index("===== after verification (everything not rejected)"))
        self.assertIn("bug(s) found;", text)
        self.assertNotIn("live tag", text)

    def test_a_live_run_reviews_base_to_each_tag_and_prints_the_per_kind_report(self):
        config = write_config(self.dir)
        write_sidecar(config)
        repo = make_repo(self.dir)
        found = {"harness-live-tests/change-1": [claim(line=10)], "harness-live-tests/change-6": [], "harness-live-tests/change-5": [claim()]}
        code, text, calls = self.run_main(["--config", str(config), "--run-type", "live", "--repo", str(repo)], lambda head: found[head])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("harness-live-tests/base", "harness-live-tests/change-1"), ("harness-live-tests/base", "harness-live-tests/change-6"),
                                 ("harness-live-tests/base", "harness-live-tests/change-5")])
        self.assertEqual(rlt.LIVE_TAG_BLOCK.findall(text), ["change-1", "change-6", "change-5"])
        self.assertIn("live other-file: 1 of 1 bug(s) found; 0 of 0 decoy alarm(s)", text)
        self.assertIn("live same-file: 0 of 1 bug(s) found", text)
        self.assertIn("live clean: 1 false positive(s) on a clean control per run", text)
        self.assertNotIn(lb.LIVE_BEFORE_MARK, text)  # no verifier ran, so one report

    def test_a_live_run_with_a_verifier_scores_before_and_after_it(self):
        config = write_config(self.dir, ("reasoning", "verifier"))
        write_sidecar(config)
        repo = make_repo(self.dir)
        code, text, _calls = self.run_main(["--config", str(config), "--run-type", "live", "--repo", str(repo)], lambda head: [])
        self.assertEqual(code, 0)
        before, after = text.index(lb.LIVE_BEFORE_MARK), text.index(lb.LIVE_AFTER_MARK)
        self.assertLess(before, after)
        # Nothing was reported before the verifier; the stubbed verifier kept a finding at B1's line, so after it the bug is found.
        self.assertIn("live other-file: 0 of 1 bug(s) found", text[before:after])
        self.assertIn("live other-file: 1 of 1 bug(s) found", text[after:])
        self.assertEqual(len(rlt.VERIFIED_LINE.findall(text)), 3)  # one verification per tag


def live_config(folder: Path, stages, expect, meta=None) -> lc.LiveConfig:
    config = write_config(folder, stages)
    write_sidecar(config, expect=expect, meta=meta)
    return lc.describe(config)


def live_report(found_other=1, alarms_diff=0, clean=0, tag_ids=("change-1", "change-6", "change-5")) -> str:
    """What a live run prints: a block per tag, then a line per kind and one for the clean control (built by the library's own writers)."""
    blocks = "".join(f"live tag {t}: 1 bug(s), 0 decoy(s), 1 run(s)\n" for t in tag_ids)
    return (blocks + f"live diff: 0 of 0 bug(s) found; {alarms_diff:g} of 1 decoy alarm(s) (per run, averaged over 1 run(s))\n"
            f"live same-file: 1 of 1 bug(s) found; 0 of 0 decoy alarm(s) (per run, averaged over 1 run(s))\n"
            f"live other-file: {found_other:g} of 1 bug(s) found; 0 of 0 decoy alarm(s) (per run, averaged over 1 run(s))\n"
            f"live clean: {clean:g} false positive(s) on a clean control per run; 0 unscored finding(s) per run (not counted)\n")


class LiveSidecars(Folder):
    BARS = {"min_other_file_found": 1, "min_same_file_found": 1, "max_diff_decoy_alarms": 0, "max_clean_false_positives": 0}

    def test_a_live_sidecar_loads_with_its_tags_and_names_no_cases(self):
        config = write_config(self.dir)
        write_sidecar(config, expect=self.BARS)
        found = lc.load_baselines(config)
        self.assertEqual((found.run_type, found.cases, lc.expected_tags(found)), ("live", (), ["change-1", "change-6", "change-5"]))

    def test_a_sidecar_with_a_synthetic_bar_or_bad_metadata_is_refused_naming_the_file(self):
        config = write_config(self.dir)
        write_sidecar(config, expect={"bugs_found_of_38": 3})
        with self.assertRaises(lc.BaselinesError) as caught:
            lc.load_baselines(config)
        self.assertIn("t.baselines.json", str(caught.exception))
        write_sidecar(config, meta={"tags": ["x"]})
        with self.assertRaises(lc.BaselinesError):
            lc.load_baselines(config)

    def test_the_reasoning_and_pipeline_scripts_run_a_live_sidecar_with_run_type_live(self):
        for stages in (["reasoning"], ["reasoning", "verifier"], ["reasoning", "verifier", "classifier"]):
            with self.subTest(stages=stages):
                config = live_config(self.dir, stages, self.BARS)
                self.assertFalse(config.unrunnable, config.problem)
                self.assertEqual(config.problem, "")
                self.assertEqual(lc.script_args(config), ("--run-type", "live"))

    def test_the_verifier_and_classifier_scripts_are_not_available_for_a_live_sidecar(self):
        for stages in (["verifier"], ["classifier"]):
            with self.subTest(stages=stages):
                config = live_config(self.dir, stages, {})
                self.assertTrue(config.unrunnable)
                self.assertIn(f"not available for the {stages[0]} script", config.problem)
                self.assertIn("t.baselines.json", config.problem)

    def test_a_synthetic_sidecar_is_read_and_run_as_before(self):
        config = write_config(self.dir)
        lc.baselines_path(config).write_text(json.dumps({"measured": "d", "model": "m", "runType": "synthetic", "metadata": {"cases": ["orders"]},
                                                         "expect": {}}), encoding="utf-8")
        described = lc.describe(config)
        self.assertEqual((described.problem, lc.script_args(described)), ("", ("--case", "orders")))


class LiveBarChecks(Folder):
    BARS = LiveSidecars.BARS

    def failing(self, checks) -> list:
        return [c.name for c in checks if not c.ok]

    def test_a_reasoning_run_that_meets_every_bar_passes(self):
        config = live_config(self.dir, ["reasoning"], self.BARS)
        checks = rlt.check_reasoning(live_report() + MODELS, 0, config, 1)
        self.assertEqual(self.failing(checks), [])
        self.assertEqual({c.kind for c in checks if c.name.startswith("`")}, {"baseline"})

    def test_each_bar_fails_on_its_own_kind_and_says_which(self):
        config = live_config(self.dir, ["reasoning"], self.BARS)
        for report, expected in ((live_report(found_other=0), "`other-file` bugs found >= 1"),
                                 (live_report(alarms_diff=1), "`diff` decoy alarms <= 0"),
                                 (live_report(clean=1), "false positives on the clean control <= 0")):
            with self.subTest(expected=expected):
                failing = self.failing(rlt.check_reasoning(report + MODELS, 0, config, 1))
                self.assertEqual(len(failing), 1, failing)
                self.assertIn(expected, failing[0])

    def test_a_bar_whose_line_was_never_printed_fails_instead_of_passing(self):
        config = live_config(self.dir, ["reasoning"], self.BARS)
        report = live_report().replace("live other-file:", "gone:")
        failing = self.failing(rlt.check_reasoning(report + MODELS, 0, config, 1))
        self.assertEqual(len(failing), 1)
        self.assertIn("other-file", failing[0])

    def test_the_run_must_cover_the_tags_the_bars_were_counted_on(self):
        config = live_config(self.dir, ["reasoning"], self.BARS)
        checks = rlt.check_reasoning(live_report(tag_ids=("change-1",)) + MODELS, 0, config, 1)
        failing = [c for c in checks if not c.ok]
        self.assertEqual([c.name for c in failing], [rlt.CHECK_TAGS])
        self.assertIn("change-6", failing[0].detail)

    def test_a_bar_met_exactly_by_a_fractional_average_passes_through_the_whole_report(self):
        one, five = by_id("change-1"), by_id("change-5")
        scores = {"change-1": [lb.score_tag([claim(line=10)], one), lb.score_tag([], one), lb.score_tag([], one)],
                  "change-6": [lb.score_tag([claim("libs/v.py", 35)], by_id("change-6"))] * 3,
                  "change-5": [lb.score_tag([claim()], five), lb.score_tag([claim()], five), lb.score_tag([], five)]}
        report = lb.format_live_report(tags(), scores) + "\n"  # a real run's next line (the loaded models) starts on a line of its own
        bars = {"min_other_file_found": 1 / 3, "min_same_file_found": 1, "max_clean_false_positives": 2 / 3}
        config = live_config(self.dir, ["reasoning"], bars)
        self.assertEqual(self.failing(rlt.check_reasoning(report + MODELS, 0, config, 3)), [])
        # And one hair above the average is still a failure: the exactness cuts both ways.
        stricter = live_config(self.dir, ["reasoning"], {"min_other_file_found": 0.34})
        self.assertEqual(len(self.failing(rlt.check_reasoning(report + MODELS, 0, stricter, 3))), 1)

    def test_no_bars_means_only_mechanics_and_the_tags(self):
        config = live_config(self.dir, ["reasoning"], {})
        checks = rlt.check_reasoning(live_report(found_other=0) + MODELS, 0, config, 1)
        self.assertEqual(self.failing(checks), [])

    def pipeline(self, runs=1, **report) -> str:
        """A pipeline's output over the three tags: one verified line per review, the before report, then the after one."""
        verified = "".join(f"--- run {n} verified: 2 confirmed, 1 disputed, 0 unchecked; dropped 1 rejected, 0 ungrounded\n" for n in range(runs * 3))
        return verified + lb.LIVE_BEFORE_MARK + "\n" + live_report(found_other=1, alarms_diff=1) + lb.LIVE_AFTER_MARK + "\n" + live_report(**report) + MODELS

    def test_a_pipeline_applies_the_bars_to_what_the_verifier_left_standing_and_counts_one_review_per_tag(self):
        config = live_config(self.dir, ["reasoning", "verifier"], self.BARS)
        # Before verification a decoy was alarmed (the bar would fail); after it, it is not.
        self.assertEqual(self.failing(rlt.check_pipeline(self.pipeline(), 0, config, 1)), [])
        failing = self.failing(rlt.check_pipeline(self.pipeline(found_other=0), 0, config, 1))
        self.assertEqual(len(failing), 1)
        self.assertIn("`other-file` bugs found >= 1", failing[0])

    def test_a_pipeline_that_reviewed_fewer_times_than_tags_times_runs_fails(self):
        config = live_config(self.dir, ["reasoning", "verifier"], self.BARS)
        failing = self.failing(rlt.check_pipeline(self.pipeline(runs=1), 0, config, 2))
        self.assertTrue(any("verifier ran after reasoning" in name for name in failing), failing)

    def test_a_pipeline_without_the_after_marker_fails_its_bars(self):
        config = live_config(self.dir, ["reasoning", "verifier"], self.BARS)
        out = self.pipeline().replace(lb.LIVE_AFTER_MARK, "gone")
        self.assertTrue(any("live after verification" in name for name in self.failing(rlt.check_pipeline(out, 0, config, 1))))

    def test_a_synthetic_sidecar_checks_are_unchanged_by_the_live_ones(self):
        config = write_config(self.dir)
        lc.baselines_path(config).write_text(json.dumps({"measured": "d", "model": "m", "runType": "synthetic", "metadata": {"cases": ["orders"]},
                                                         "expect": {}}), encoding="utf-8")
        self.assertEqual(rlt.live_bar_checks("anything", lc.describe(config)), [])


if __name__ == "__main__":
    unittest.main()
