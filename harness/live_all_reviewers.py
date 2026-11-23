#!/usr/bin/env python3
"""Harness 4 of 4: every configured reviewer, against test code with known answers.

    .venv/bin/python harness/live_all_reviewers.py                       # orders case, three lenses on one model
    .venv/bin/python harness/live_all_reviewers.py --config my-reviewers.json --case orders --runs 3   # a reviewers file as it stands
    .venv/bin/python harness/live_all_reviewers.py --config my-config.json --runs 3   # reviewers, rubric and stages from a config

The other three harnesses isolate one stage. This one runs the whole reviewers
file the way the `review` tool would (several reviewers, models shared or
swapped, resident and on-demand mixed) over the four-file `orders` case, and
scores the result: which of its six defects were found, which of its three
decoys drew a false alarm.

Without --config it builds three reasoning reviewers on --model, each
with a narrower lens (injection, bounds and state, general), which also
exercises the scheduler's grouping of reviewers that share a model.

With --config it uses that file's reviewers and rubric as written (a reviewers file is a valid config). Every
stage runs: when verifier reviewers are configured, each run's findings are grounded and verified after the
reasoning stage, and the score is printed before and after, which is where the decoy false alarms should
disappear. When classifier reviewers are configured, the findings that stand (everything the verifiers did not
reject, or every finding when no verifier is configured) are then sorted into the rubric, and each is shown
with its rubric level.

--run-type live runs the same pipeline over each tag of the `harness-live-tests` branch that the sidecar beside --config names
(`<config>.baselines.json`, `runType: live`), instead of over a case from `cases.py`, and scores each tag's findings against the
bugs and decoys in that sidecar, per evidence kind (see live_branch_lib.py). Every stage a config selects runs, in order.

--config also takes a `stages` list: it picks which of reasoning, verifier and classifier to test
(see README.md), and a stage it leaves out is not run even when reviewers are configured for it.
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_lib as h  # noqa: E402
import live_branch_lib as lb  # noqa: E402
from cases import ALL_CASES, RUBRIC  # noqa: E402
from codebase_reviewer_config_lib import load_reviewers_file  # noqa: E402
from reasoning_reviewer_lib import format_review  # noqa: E402
from reviewer_db_lib import new_run_recorder  # noqa: E402
from reviewer_scheduler_lib import describe_plan, plan_stages  # noqa: E402
from verifier_reviewer_lib import REJECTED, VerifiedReview, flatten, format_verified_review  # noqa: E402

LENSES = (
    ("injection", "You review Python changes ONLY for injection: shell, SQL, path or argument injection from untrusted input."),
    ("bounds-and-state", "You review Python changes ONLY for off-by-one errors, bad slicing, empty-input crashes, wrong formulas, "
                         "and state shared between calls such as mutable default arguments or unsynchronised counters."),
    ("general", h.DEFAULT_LENS),
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    h.add_model_options(parser, config=True)
    h.add_run_type_option(parser)
    # No argparse default: a live run refuses --case, which it can only do if "not given" can be told from "orders".
    parser.add_argument("--case", default=None, choices=list(ALL_CASES), help="the code under review (default orders)")
    parser.add_argument("--history", default=None, metavar="SNAPSHOT",
                        help="replay a Copilot review from a snapshot (collect_copilot_history.py) instead of --case, and score parity with it")
    parser.add_argument("--round", type=int, default=1, help="which Copilot review of the snapshot to replay (default 1)")
    raw = sys.argv[1:] if argv is None else argv
    if any(a == "--reviewers-file" or a.startswith("--reviewers-file=") for a in raw):
        # Removed in favor of --config; say where to go instead of argparse's bare "unrecognized arguments".
        parser.error("--reviewers-file was removed: pass the same file with --config (a reviewers file is a valid config as it stands)")
    return h.finish_args(parser.parse_args(argv))


def build_config(args: argparse.Namespace) -> tuple:
    """(reviewer entries, rubric) from --config's reviewers, else three lenses on --model."""
    if args.config_data is not None and args.config_data.reviewers:
        return list(args.config_data.reviewers), list(args.config_data.rubric)
    entries = [h.reviewer_entry(name, args.model, prompt=lens, prompt_file=None, on_demand=not args.resident, context=args.context,
                                reply_format=args.reply_format, translator=args.translator)
               for name, lens in LENSES]
    return entries, list(RUBRIC)


def classified_summary(run: int, classification) -> str:
    """The one line a run's classifier stage prints, which `live-tests/run_live_tests.py` reads: counts by level, then what
    was dropped or left without a rubric level."""
    levels = ", ".join(f"{len(classification.by_level(level))} {level}" for level in ("high", "moderate", "low"))
    return (f"--- run {run} classified: {len(classification.matched)} classified ({levels}), "
            f"{len(classification.unclassified)} unclassified, "
            f"{len(classification.unchecked)} without a rubric level; {len(classification.overlaps)} overlap(s); "
            f"classifiers {classification.classifiers_run}")


def classified_lines(classification) -> list:
    """Each finding the classifier stage saw, with the rubric level it was given (or why it has none)."""
    lines = []
    for item in classification.classified:
        f = item.finding
        where = f"{f.file}:{f.line_start}" if f.line_start else f.file
        if item.status == "classified":
            outcome = f"{item.entry} [{item.level}], average probability {item.confidence:.2f}" + (f"; also fits {', '.join(item.overlapping)}" if item.overlap else "")
        elif item.status == "unclassified":
            outcome = "unclassified (matched no entry; dropped)"
        else:
            outcome = f"no rubric level ({item.note})"
        lines.append(f"  {where} [{f.cls}] {f.claim[:60]} -> {outcome}")
    return lines


def review_target(args: argparse.Namespace, entries: list, rubric: list, stages_in_use: tuple, repo: Path, base: str, head: str,
                  score_findings, recorder: h.Recorder) -> tuple:
    """Every run of the whole pipeline over one `base...head`: reasoning, then the verifiers, then the classifiers.

    `score_findings` turns one run's findings into a score (`harness_lib.score` for a synthetic case, `live_branch_lib.score_tag` for a live
    tag), so the stages and their output are the same for both run types. Returns `(scores, verified_scores, all_findings)`."""
    reasoning, verifiers, classifiers = stages_in_use
    active = len(reasoning)
    scores = []
    verified_scores = []
    all_findings = []
    for run in range(1, args.runs + 1):
        # One review is one run in the reviewer's database: reasoning, then the verifier, then the classifier. The stages
        # share the record, so it holds the model combination of all of them and the time of the whole review.
        with new_run_recorder(source="harness", review_type="classifier" if classifiers else "verifier" if verifiers else "reasoning", repo_path=str(repo),
                              base_ref=base or "", head_ref=head,
                              settings={"max_reviewers": max(active, 1), "chunk_chars": args.chunk_chars,
                                        "context_lines": args.context_lines}) as run_recorder:
            findings, review = h.run_reasoning(repo, base, head, entries, args, recorder, max_reviewers=max(active, 1), rubric=rubric,
                                                run_recorder=run_recorder)
            scores.append(score_findings(findings))
            all_findings.append(findings)
            print(f"--- run {run}: {len(findings)} finding(s) from {review.reviewers_run}, {review.chunks} chunk(s)")
            for error in review.errors:
                print(f"  reviewer error: {error}")
            if run == args.runs:
                print(format_review(review, followed_by=tuple(stage for stage, ran in (("verifier", verifiers), ("classifier", classifiers)) if ran)))
            # One finding per claim, as the verifier stage hands them on: a merged finding's other claims (`alternates`) are
            # classified too, also when there is no verifier to flatten them.
            standing = [unit for group in review.groups for f in group.findings for unit in flatten(f)]
            statuses = [""] * len(standing)
            if verifiers:
                # What the `verifier` review type does after reasoning: ground each finding's quote in the file at `head`, then
                # have the verifiers confirm or reject. The score after is of what is left standing: everything not rejected.
                verification = h.run_verifier(repo, head, [f for group in review.groups for f in group.findings], verifiers, args,
                                                  recorder, max_reviewers=len(verifiers), run_recorder=run_recorder)
                verified_scores.append(score_findings(h.distinct_claims(verification.kept)))
                print(f"--- run {run} verified: {len(verification.confirmed)} confirmed, {len(verification.disputed)} disputed, "
                      f"{len(verification.unchecked)} unchecked; dropped {len(verification.rejected)} rejected, "
                      f"{len(verification.ungrounded)} ungrounded; verifiers {verification.verifiers_run}")
                # Every run, not only the last: a verifier that failed leaves its findings unchecked, which the score after
                # counts as standing, so a dead verifier would otherwise look like a good one.
                for error in verification.errors:
                    print(f"  verifier problem: {error}")
                if run == args.runs:
                    print(format_verified_review(VerifiedReview(review, verification)))
                # What the `classifier` review type sorts: everything the verifiers did not reject.
                kept = [v for v in verification.verified if v.status != REJECTED]
                standing, statuses = [v.finding for v in kept], [v.status for v in kept]
            if classifiers:
                classification = h.run_classifier(standing, classifiers, rubric, args, recorder, statuses=statuses,
                                                  max_reviewers=len(classifiers), run_recorder=run_recorder)
                print(classified_summary(run, classification))
                for error in classification.errors:
                    print(f"  classifier problem: {error}")
                if run == args.runs:
                    for line in classified_lines(classification):
                        print(line)
    return scores, verified_scores, all_findings


def report_live(tags: tuple, scores_by_tag: dict, verified_by_tag: dict, verifiers: list) -> None:
    """The live run's scores: per tag and per evidence kind, before the verifier and (when there is one) after it."""
    if not verifiers:
        print(lb.format_live_report(tags, scores_by_tag))
        return
    print(lb.LIVE_BEFORE_MARK)
    print(lb.format_live_report(tags, scores_by_tag))
    print(lb.LIVE_AFTER_MARK)
    print(lb.format_live_report(tags, verified_by_tag))


def main(argv=None) -> int:
    args = parse_args(argv)
    live = args.run_type == "live"
    if live and (args.case is not None or args.history):
        raise SystemExit("--run-type live reviews the tags in the live sidecar: --case and --history belong to a synthetic run, drop them.")
    if args.show_config:
        print(h.describe_config(args))
        return 0
    entries, rubric = build_config(args)
    stages = h.selected_stages(args)
    if "reasoning" not in stages:
        print(f"stages selected: {list(stages)}; reasoning is not one of them, so no review runs here.")
        return 0
    # Before any model loads: a missing sidecar or tag must stop the run here, not after a load.
    live_repo, live_tags = lb.live_target(args) if live else (h.REPO_ROOT, ())
    # Only the reviewers `run_reasoning` will actually run: a disabled reviewer's model, or one that only
    # takes a stage this script does not run, must not stop the others.
    reasoning = [e for e in entries if e.get("active", True) and "reasoning" in e.get("type", [])]
    # The verifiers run after reasoning (see below). Each is restricted to its verifier role, like `stage_reviewers` does: a
    # reviewer that also has the classifier role would need the rubric to load, which this stage's reviewers file leaves out.
    verifiers = [{**e, "type": ["verifier"]} for e in entries
                 if e.get("active", True) and "verifier" in e.get("type", [])] if "verifier" in stages else []
    # The classifiers run last, on what is left standing. Restricted to their role for the same reason as the verifiers.
    classifiers = [{**e, "type": ["classifier"]} for e in entries
                   if e.get("active", True) and "classifier" in e.get("type", [])] if "classifier" in stages else []
    h.preflight_reviewers(reasoning + verifiers + classifiers, args)
    if args.history:
        case, hist_base, hist_head = h.history_case(args.history, args.round - 1)
    elif live:
        case, hist_base, hist_head = None, None, None
    else:
        case, hist_base, hist_head = ALL_CASES[args.case or "orders"], None, None
    h.isolate_reviewer_db()  # past every exit that runs no review, the preflight included: from here the run records
    base_url = h.load_server().lmstudio_base_url()
    before = h.loaded_models(base_url)
    recorder = h.Recorder()
    in_use = (reasoning, verifiers, classifiers)
    with tempfile.TemporaryDirectory(prefix="live-all-") as tmp:
        scratch = Path(tmp)
        plan_file = h.write_reviewers_file(scratch, entries, rubric)
        print(describe_plan(plan_stages(load_reviewers_file(str(plan_file)))))
        if live:
            scores_by_tag: dict = {}
            verified_by_tag: dict = {}
            for tag in live_tags:
                print(f"===== live {tag.tag_id}: {lb.BASE_REF}...{tag.head}")
                scores, verified, _all = review_target(args, entries, rubric, in_use, live_repo, lb.BASE_REF, tag.head,
                                                       lambda found, tag=tag: lb.score_tag(found, tag), recorder)
                scores_by_tag[tag.tag_id], verified_by_tag[tag.tag_id] = scores, verified
            report_live(live_tags, scores_by_tag, verified_by_tag, verifiers)
            print(h.report_run(args, recorder, base_url, before))
            return 0
        repo, base, head = (h.REPO_ROOT, hist_base, hist_head) if args.history else h.build_repo(scratch, case)
        scores, verified_scores, all_findings = review_target(args, entries, rubric, in_use, repo, base, head,
                                                              lambda found: h.score(found, case), recorder)
    summarize = h.parity_line if args.history else h.seeded_line  # parity speaks of Copilot; a seeded case has bugs and decoys
    print("===== before verification" if verifiers else "=====")
    print(h.format_score(case, scores))
    if args.history or verifiers:
        print(summarize(case, scores))
    if verifiers:
        print("===== after verification (everything not rejected)")
        print(h.format_score(case, verified_scores))
        print(summarize(case, verified_scores))
    if len(reasoning) > 1:
        print("per reviewer (findings each one raised, merged duplicates counted for both):")
        for entry in reasoning:
            alone = [h.score([f for f in run_findings if entry["name"] in f.reviewers], case) for run_findings in all_findings]
            print(f"  {entry['name']:<20} {entry['model']:<34} {summarize(case, alone)}")
    print(h.report_run(args, recorder, base_url, before))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
