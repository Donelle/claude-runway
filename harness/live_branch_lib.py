"""The `live` run type: score reviews of the `harness-live-tests` branch against a baselines sidecar's `metadata`.

A `synthetic` run reviews mock scenarios from `cases.py`, whose answers sit in the code. A `live` run reviews real
changes on the `harness-live-tests` branch (tags `harness-live-tests/change-N` on top of `harness-live-tests/base`).
The answer key is deliberately not on that branch (a model must not be able to grep it), so for these runners it lives in the
config's `<config>.baselines.json`, in `metadata.tags`: for each tag, the bugs and decoys a run is scored against. See
`harness/README.md`, "A real repository with seeded bugs", for the same key in a table for people. The same key, once, is
`harness/live-tests/answer_key.json`, which the Pi scorer reads; the sidecar copies go when these runners do.

The answer key's data model and parser, the scoring rules (a finding near a bug's line is a hit, at a decoy's line a false
positive, on a clean tag a false positive, anything else UNSCORED and never counted against a model), the per-evidence-kind
totals and the branch's ref names are in `live_scoring_lib.py`, which depends on nothing else in the harness. They are imported
here, so a runner reaches them as `live_branch_lib.<name>` as before. What stays here is what only these runners need: reading
the sidecar, finding the tags in a checkout, and the report they print.

Results are reported per evidence kind (`diff`, `same-file`, `other-file`), because that is the point of a live run:
whether a model whose proof is in another file asks for it (the `need:` loop).
"""

import json
from pathlib import Path
from typing import Optional

import harness_lib as h
# Re-exported on purpose: the runners and their tests use these as `lb.<name>` (a name imported here only for that is marked).
from live_scoring_lib import (  # noqa: F401
    BASE_REF,
    EVIDENCE_KINDS,
    FETCH_COMMAND,
    PREFIX,
    LiveBug,
    LiveDecoy,
    LiveMetadataError,
    LiveTag,
    TagScore,
    kind_totals,
    missing_refs,
    missing_refs_message,
    parse_live_metadata,
    score_tag,
)

BASELINES_SUFFIX = ".baselines.json"
# Printed by live_all_reviewers.py around its two reports when a verifier runs; `run_live_tests.py` reads the bars from after the second.
# Not `===== before verification`: that is the synthetic marker, and a substring search for it must not find these.
LIVE_BEFORE_MARK = "===== live before verification"
LIVE_AFTER_MARK = "===== live after verification (everything not rejected)"

# `expect` keys a live sidecar may carry: one bar per evidence kind for bugs found and for decoys alarmed, and one for the
# clean control. Written with underscores (a JSON key), where the kind itself has hyphens.
BAR_KEYS = tuple([f"min_{kind.replace('-', '_')}_found" for kind in EVIDENCE_KINDS]
                 + [f"max_{kind.replace('-', '_')}_decoy_alarms" for kind in EVIDENCE_KINDS]
                 + ["max_clean_false_positives"])


def check_bars(expect: dict) -> Optional[str]:
    """A message when a live sidecar's `expect` has a key that is not a live bar; None when every key is one.

    Only the bars a live run reads are allowed: a synthetic key left in would never be applied, and a missing bar
    would pass without anything saying so."""
    unknown = sorted(set(expect) - set(BAR_KEYS))
    if unknown:
        return f"'expect' has {unknown}, which a live run does not read (the live bars are {list(BAR_KEYS)})"
    return None


# ---------------------------------------------------------------------------
# finding the branch
# ---------------------------------------------------------------------------


def require_refs(repo: Path, tags: tuple) -> None:
    """Stop with the exact `git fetch` command when a tag is missing, before any model is loaded."""
    missing = missing_refs(repo, tags)
    if missing:
        raise SystemExit(missing_refs_message(repo, missing))


def sidecar_path(config: str) -> Path:
    # `~` is expanded here as `harness_lib.load_config` expands it for the config itself: a quoted `--config '~/x.json'` (or an
    # unquoted `--config=~/x.json`, which a shell leaves alone) loads, so its sidecar must be looked for where it really is.
    path = Path(config).expanduser()
    return path.with_name(path.stem + BASELINES_SUFFIX)


def live_target(args) -> tuple:
    """`(repo, tags)` for a `--run-type live` run: the checkout holding the branch, and the tags to review.

    The tags come from the sidecar next to `--config`. Refuses what cannot be a live run, each with its reason, instead
    of falling back to something else."""
    if not args.config:
        raise SystemExit("--run-type live needs --config: what a live run is scored against is the `runType: live` sidecar beside the config "
                         "(<config>.baselines.json), and without one there is no answer key.")
    path = sidecar_path(args.config)
    if not path.is_file():
        raise SystemExit(f"--run-type live needs {path.name} next to the config: it names the tags and the bugs and decoys to score against. "
                         f"Looked for {path}.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SystemExit(f"{path.name} cannot be read as JSON ({exc})") from exc
    if not isinstance(data, dict) or data.get("runType") != "live":
        raise SystemExit(f"{path.name} has runType {data.get('runType') if isinstance(data, dict) else None!r}: --run-type live needs a 'live' one. "
                         "A bar belongs to the run type it was measured on.")
    try:
        tags = parse_live_metadata(data.get("metadata"))
    except LiveMetadataError as exc:
        raise SystemExit(f"{path.name}: {exc}") from exc
    repo = Path(args.repo).expanduser().resolve() if args.repo else h.REPO_ROOT
    require_refs(repo, tags)
    return repo, tags


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def format_tag(tag: LiveTag, scores: list) -> str:
    """One block for a tag across its runs, like `harness_lib.format_score`."""
    runs = len(scores)
    lines = [f"live tag {tag.tag_id}: {len(tag.bugs)} bug(s), {len(tag.decoys)} decoy(s){', clean control' if tag.clean else ''}, {runs} run(s)"]
    for bug in tag.bugs:
        hits = sum(1 for s in scores if bug.id in s.found)
        lines.append(f"  {'FOUND ' if hits == runs else 'partly' if hits else 'MISSED'} {bug.id:<4} {hits}/{runs}  [{bug.evidence}] {bug.claim[:70]}")
    for decoy in tag.decoys:
        hits = sum(1 for s in scores if decoy.id in s.on_decoys)
        lines.append(f"  decoy  {decoy.id:<4} raised {hits}/{runs}  [{decoy.evidence}] (should be 0)")
    if tag.clean:
        flagged = [text for s in scores for text in s.clean_false_positives]
        lines.append(f"  false positives on the clean control: {len(flagged)} over {runs} run(s) (should be 0)")
        lines.extend(f"    - {text}" for text in dict.fromkeys(flagged))
    unscored = [text for s in scores for text in s.unscored]
    lines.append(f"  unscored findings: {len(unscored)} over {runs} run(s) (matched no listed bug or decoy; not counted against the model)")
    lines.extend(f"    - {text}" for text in dict.fromkeys(unscored))
    return "\n".join(lines)


def _num(value: float) -> str:
    """`value` as text that reads back as the same float: `1`, `0.5`, but `0.3333333333333333` and not `0.333333`.

    These numbers are printed and then read back by `run_live_tests.py` to be compared with a bar, so a display rounding
    (`:g` keeps 6 digits) would make a run that exactly meets a bar of one third fail it."""
    short = f"{value:g}"
    return short if float(short) == value else repr(value)


def summary_lines(tags: tuple, scores_by_tag: dict) -> list:
    """The lines `run_live_tests.py` reads (`LIVE_KIND_LINE`, `LIVE_CLEAN_LINE`): per evidence kind, then the clean control and unscored."""
    runs = len(next(iter(scores_by_tag.values()))) if scores_by_tag else 0
    lines = []
    for kind, (bugs, found, decoys, alarms) in kind_totals(tags, scores_by_tag).items():
        lines.append(f"live {kind}: {_num(found)} of {bugs} bug(s) found; {_num(alarms)} of {decoys} decoy alarm(s) (per run, averaged over {runs} run(s))")
    clean = sum(len(s.clean_false_positives) for t in tags for s in scores_by_tag.get(t.tag_id, []))
    unscored = sum(len(s.unscored) for t in tags for s in scores_by_tag.get(t.tag_id, []))
    per = (lambda n: n / runs if runs else 0.0)
    # Only the clean figure is read back and compared; the unscored one is for people, so it keeps the short form.
    lines.append(f"live clean: {_num(per(clean))} false positive(s) on a clean control per run; {per(unscored):g} unscored finding(s) per run (not counted)")
    return lines


def format_live_report(tags: tuple, scores_by_tag: dict) -> str:
    blocks = [format_tag(tag, scores_by_tag.get(tag.tag_id, [])) for tag in tags]
    return "\n".join([*blocks, "===== per evidence kind", *summary_lines(tags, scores_by_tag)])
