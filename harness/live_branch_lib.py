"""The `live` run type: score reviews of the `harness-live-tests` branch against a baselines sidecar's `metadata`.

A `synthetic` run reviews mock scenarios from `cases.py`, whose answers sit in the code. A `live` run reviews real
changes on the `harness-live-tests` branch (tags `harness-live-tests/change-N` on top of `harness-live-tests/base`).
The answer key is deliberately not on that branch (a model must not be able to grep it), so it lives in the config's
`<config>.baselines.json`, in `metadata.tags`: for each tag, the bugs and decoys a run is scored against. See
`harness/README.md`, "A real repository with seeded bugs", for the same key in a table for people.

Scoring (the issue's rules): a finding within `LINE_TOLERANCE` lines of a bug's line, in that file, is a hit; one at a
decoy's line is a false positive; any finding on a clean tag is a false positive; anything else is UNSCORED, reported
but never counted against a model, because the code is real and may have real bugs nobody has listed. A tie between a
bug and a decoy goes to the bug, as in `harness_lib.score`.

Results are reported per evidence kind (`diff`, `same-file`, `other-file`), because that is the point of a live run:
whether a model whose proof is in another file asks for it (the `need:` loop).
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import harness_lib as h

PREFIX = "harness-live-tests/"
BASE_REF = PREFIX + "base"
EVIDENCE_KINDS = ("diff", "same-file", "other-file")
BASELINES_SUFFIX = ".baselines.json"
# Printed by live_all_reviewers.py around its two reports when a verifier runs; `run_live_tests.py` reads the bars from after the second.
# Not `===== before verification`: that is the synthetic marker, and a substring search for it must not find these.
LIVE_BEFORE_MARK = "===== live before verification"
LIVE_AFTER_MARK = "===== live after verification (everything not rejected)"
# What to run when a tag is missing: the README's own command, so a person can paste it.
FETCH_COMMAND = "git fetch origin harness-live-tests 'refs/tags/harness-live-tests/*:refs/tags/harness-live-tests/*'"

# `expect` keys a live sidecar may carry: one bar per evidence kind for bugs found and for decoys alarmed, and one for the
# clean control. Written with underscores (a JSON key), where the kind itself has hyphens.
BAR_KEYS = tuple([f"min_{kind.replace('-', '_')}_found" for kind in EVIDENCE_KINDS]
                 + [f"max_{kind.replace('-', '_')}_decoy_alarms" for kind in EVIDENCE_KINDS]
                 + ["max_clean_false_positives"])


class LiveMetadataError(ValueError):
    """A live sidecar's `metadata` that cannot be used; the message names the key."""


@dataclass(frozen=True)
class LiveBug:
    id: str
    file: str
    line: int
    cls: str
    claim: str
    evidence: str


@dataclass(frozen=True)
class LiveDecoy:
    id: str
    file: str
    line: int
    claim: str
    evidence: str


@dataclass(frozen=True)
class LiveTag:
    tag_id: str  # `change-1`, without the `harness-live-tests/` prefix
    bugs: tuple = ()
    decoys: tuple = ()
    clean: bool = False  # the control: no change in behaviour, so any finding is a false positive

    @property
    def head(self) -> str:
        return PREFIX + self.tag_id


@dataclass
class TagScore:
    """One run's findings against one tag."""

    found: list = field(default_factory=list)  # bug ids hit
    missed: list = field(default_factory=list)
    on_decoys: list = field(default_factory=list)  # decoy ids alarmed
    clean_false_positives: list = field(default_factory=list)  # findings on a clean tag
    unscored: list = field(default_factory=list)  # matched nothing: reported, never counted against a model


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def _string(where: str, data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LiveMetadataError(f"{where}: '{key}' must be a non-empty string")
    return value


def _line(where: str, data: dict) -> int:
    value = data.get("line")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise LiveMetadataError(f"{where}: 'line' must be a whole number from 1")
    return value


def _evidence(where: str, data: dict) -> str:
    value = data.get("evidence")
    if value not in EVIDENCE_KINDS:
        raise LiveMetadataError(f"{where}: 'evidence' is {value!r}, which is none of {list(EVIDENCE_KINDS)}")
    return value


def _only(where: str, data: dict, allowed: tuple) -> None:
    extra = sorted(set(data) - set(allowed))
    if extra:
        raise LiveMetadataError(f"{where}: unknown key(s) {extra} (allowed: {list(allowed)}); a misspelt key would otherwise be ignored")


def parse_live_metadata(metadata) -> tuple:
    """The `LiveTag`s a live sidecar's `metadata` names, or `LiveMetadataError` naming what is wrong.

    Strict on purpose: a typo in an answer key (a misspelt `line`, a bug left out of a tag) would change a score
    without anything saying so."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get("tags"), list) or not metadata["tags"]:
        raise LiveMetadataError("a live run's 'metadata' needs 'tags': a non-empty list, one entry per tag the run is against")
    _only("metadata", metadata, ("tags",))
    tags: list = []
    seen_tags: set = set()
    seen_ids: set = set()
    for position, raw in enumerate(metadata["tags"]):
        where = f"metadata.tags[{position}]"
        if not isinstance(raw, dict):
            raise LiveMetadataError(f"{where} must be an object")
        _only(where, raw, ("tagId", "clean", "bugs", "decoys"))
        tag_id = _string(where, raw, "tagId")
        if "/" in tag_id:
            raise LiveMetadataError(f"{where}: 'tagId' is {tag_id!r}; write it without the '{PREFIX}' prefix (for example 'change-1')")
        if tag_id == "base":
            raise LiveMetadataError(f"{where}: 'base' is what every tag is compared with, not a tag to review")
        if tag_id in seen_tags:
            raise LiveMetadataError(f"{where}: tag {tag_id!r} is listed twice")
        seen_tags.add(tag_id)
        where = f"metadata.tags[{tag_id}]"
        clean = raw.get("clean", False)
        if not isinstance(clean, bool):
            raise LiveMetadataError(f"{where}: 'clean' must be true or false")
        for key in ("bugs", "decoys"):
            if not isinstance(raw.get(key), list):
                raise LiveMetadataError(f"{where}: '{key}' must be a list (empty when the tag has none)")
        bugs, decoys = [], []
        for entry in raw["bugs"]:
            at = f"{where}.bugs"
            if not isinstance(entry, dict):
                raise LiveMetadataError(f"{at}: each entry must be an object")
            _only(at, entry, ("id", "file", "line", "class", "claim", "evidence"))
            bugs.append(LiveBug(_string(at, entry, "id"), _string(at, entry, "file"), _line(at, entry), _string(at, entry, "class"),
                                _string(at, entry, "claim"), _evidence(at, entry)))
        for entry in raw["decoys"]:
            at = f"{where}.decoys"
            if not isinstance(entry, dict):
                raise LiveMetadataError(f"{at}: each entry must be an object")
            _only(at, entry, ("id", "file", "line", "claim", "evidence"))
            decoys.append(LiveDecoy(_string(at, entry, "id"), _string(at, entry, "file"), _line(at, entry), _string(at, entry, "claim"),
                                    _evidence(at, entry)))
        for item_id in [*(b.id for b in bugs), *(d.id for d in decoys)]:
            if item_id in seen_ids:
                raise LiveMetadataError(f"{where}: id {item_id!r} is used twice (ids are unique across the whole file)")
            seen_ids.add(item_id)
        if clean and (bugs or decoys):
            raise LiveMetadataError(f"{where}: 'clean' is true but it lists bugs or decoys; a clean control has none")
        if not clean and not bugs and not decoys:
            raise LiveMetadataError(f"{where}: lists no bugs and no decoys; if it is the clean control say \"clean\": true, so an empty tag is never an accident")
        tags.append(LiveTag(tag_id, tuple(bugs), tuple(decoys), clean))
    return tuple(tags)


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


def missing_refs(repo: Path, tags: tuple) -> list:
    """The refs among `base` and `tags` that `repo` does not have as commits."""
    missing = []
    for ref in (BASE_REF, *(tag.head for tag in tags)):
        found = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                               capture_output=True, text=True, check=False)
        if found.returncode != 0:
            missing.append(ref)
    return missing


def require_refs(repo: Path, tags: tuple) -> None:
    """Stop with the exact `git fetch` command when a tag is missing, before any model is loaded."""
    missing = missing_refs(repo, tags)
    if missing:
        # The README's own command, verbatim, so it can be pasted; it names the place to run it instead of using `git -C`.
        raise SystemExit(f"{repo} does not have {', '.join(missing)}. A single-branch clone does not carry the live branch. "
                         f"Run this in {repo}:\n  {FETCH_COMMAND}")


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
# scoring and reporting
# ---------------------------------------------------------------------------


def _text(finding) -> str:
    return f"{finding.file}:{finding.line_start}-{finding.line_end} [{finding.cls}] {finding.claim[:90]}"


def score_tag(findings: list, tag: LiveTag) -> TagScore:
    """Score one run's findings (as `harness_lib.distinct_claims` gives them) against one tag."""
    result = TagScore()
    bug_at = {bug.id: (bug.file, bug.line) for bug in tag.bugs}
    decoy_at = {decoy.id: (decoy.file, decoy.line) for decoy in tag.decoys}
    found: set = set()
    alarmed: set = set()
    for finding in findings:
        if tag.clean:
            result.clean_false_positives.append(_text(finding))
            continue
        bug = h._nearest(finding, bug_at, frozenset(found))
        decoy = h._nearest(finding, decoy_at)
        if bug is not None and (decoy is None or bug[1] <= decoy[1]):
            found.add(bug[0])
        elif decoy is not None:
            alarmed.add(decoy[0])
        else:
            result.unscored.append(_text(finding))
    result.found = [bug.id for bug in tag.bugs if bug.id in found]
    result.missed = [bug.id for bug in tag.bugs if bug.id not in found]
    result.on_decoys = [decoy.id for decoy in tag.decoys if decoy.id in alarmed]
    return result


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


def kind_totals(tags: tuple, scores_by_tag: dict) -> dict:
    """`{kind: (bugs, found per run, decoys, alarms per run)}`, the per-run values averaged over the runs."""
    runs = len(next(iter(scores_by_tag.values()))) if scores_by_tag else 0
    totals = {}
    for kind in EVIDENCE_KINDS:
        bugs = sum(1 for t in tags for b in t.bugs if b.evidence == kind)
        decoys = sum(1 for t in tags for d in t.decoys if d.evidence == kind)
        found = alarms = 0
        for tag in tags:
            for score in scores_by_tag.get(tag.tag_id, []):
                found += sum(1 for b in tag.bugs if b.evidence == kind and b.id in score.found)
                alarms += sum(1 for d in tag.decoys if d.evidence == kind and d.id in score.on_decoys)
        totals[kind] = (bugs, found / runs if runs else 0.0, decoys, alarms / runs if runs else 0.0)
    return totals


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
