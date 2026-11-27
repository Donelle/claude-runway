"""The answer key of the `harness-live-tests` branch, and how a review is scored against it.

This is the part of the live harness that every consumer needs and that needs nothing else from it: the Pi scorer
(`harness/pi/score_run.py`), the Pi driver, the older runners through `live_branch_lib.py`, and the orchestrator that comes
later. It imports no other harness module. That is the point: `cases.py` (the made-up cases) and `harness_lib.py` (which
imports it) are on their way out, and a tool that only scores a review must not need either of them.

The branch is a frozen copy of this repository (`harness-live-tests/base`) with six one-commit changes on top of it, tagged
`harness-live-tests/change-1` to `change-6`; see `harness/README.md`, "A real repository with seeded bugs". The answer key is
`harness/live-tests/answer_key.json`: for each change, the bugs and decoys a review is scored against. **It is never on the
branch and never given to a container**: the branch has no `harness/`, and the Pi driver mounts only `libs/` and `mcpservers/`
of this checkout, so a model that reads every file it can reach still cannot read the answers.

Scoring rules (a finding is matched by file and nearby line only; the class or wording the model used is never compared):
a finding within `LINE_TOLERANCE` lines of a bug's line, in that file, is a hit; one within that distance of a decoy's line is
a false alarm; any finding on the clean change (`change-5`) is a false alarm; anything else is UNSCORED, reported but never
counted against a model, because the code is real and may have real bugs nobody has listed. A finding as near a bug as a decoy
goes to the bug.

Results can be grouped by evidence kind (`diff`, `same-file`, `other-file`), because the question a live run answers is whether
a model whose proof is in another file asks for it (the `need:` loop).

Standard library only.
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PREFIX = "harness-live-tests/"
BASE_REF = PREFIX + "base"
EVIDENCE_KINDS = ("diff", "same-file", "other-file")
# What to run when a tag is missing: the README's own command, so a person can paste it.
FETCH_COMMAND = "git fetch origin harness-live-tests 'refs/tags/harness-live-tests/*:refs/tags/harness-live-tests/*'"
# Where the key lives, on this side of the repository only (see the module docstring).
ANSWER_KEY_PATH = Path(__file__).resolve().parent / "live-tests" / "answer_key.json"
# Reported lines are the model's reading of a diff, and drift by a few lines.
LINE_TOLERANCE = 3


class LiveMetadataError(ValueError):
    """An answer key (or a live sidecar's `metadata`) that cannot be used; the message names the key."""


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
# the answer key
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
    """The `LiveTag`s a live sidecar's `metadata` (or the answer key file, which holds the same thing) names, or
    `LiveMetadataError` naming what is wrong.

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


def load_answer_key(path: Optional[Path] = None) -> tuple:
    """The `LiveTag`s of the answer key file (default: `harness/live-tests/answer_key.json`), in the file's order.

    The file holds what a live sidecar's `metadata` holds, `{"tags": [...]}`, so the same strict parser reads both and its
    messages say `metadata.tags[...]` for either. Raises `LiveMetadataError` naming the file when it cannot be read or is wrong."""
    path = Path(path) if path is not None else ANSWER_KEY_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise LiveMetadataError(f"the answer key {path} cannot be read as JSON ({exc})") from exc
    try:
        return parse_live_metadata(data)
    except LiveMetadataError as exc:
        raise LiveMetadataError(f"the answer key {path}: {exc}") from exc


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


def missing_refs_message(repo: Path, missing: list) -> str:
    """What to tell a person whose checkout lacks `missing`: the README's own `git fetch`, ready to paste, and where to run it
    (named in the sentence instead of using `git -C`)."""
    return (f"{repo} does not have {', '.join(missing)}. A single-branch clone does not carry the live branch. "
            f"Run this in {repo}:\n  {FETCH_COMMAND}")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def finding_distance(finding, file: str, line: int) -> Optional[int]:
    """Lines between the finding's range and `line` (0 when inside it); None when another file, beyond LINE_TOLERANCE,
    or the finding has no known start line (the parser records a missing or invalid line number as 0, which is not a place)."""
    same_file = finding.file == file or finding.file.endswith("/" + file) or file.endswith("/" + finding.file)
    if not same_file or finding.line_start <= 0:
        return None
    gap = max(finding.line_start - line, line - max(finding.line_end, finding.line_start), 0)
    return gap if gap <= LINE_TOLERANCE else None


def nearest_anchor(finding, anchors: dict, taken: frozenset = frozenset()) -> Optional[tuple]:
    """`(anchor id, distance)` of the anchor `finding` sits nearest to, or None. Two defects a few lines apart must
    not be credited to whichever happens to come first, or an exact hit on the second reads as a miss.

    When several anchors are equally near, the first listed wins, EXCEPT among anchors at the very same file and
    line as that first one (two Copilot comments on one line): there, one not yet credited in `taken` is preferred,
    so two claims at that line are matched one to one. Anchors merely at the same distance from a wide range are
    different places, and a claim there must not be credited with one it did not report."""
    gaps = {anchor_id: finding_distance(finding, file, line) for anchor_id, (file, line) in anchors.items()}
    near = {anchor_id: gap for anchor_id, gap in gaps.items() if gap is not None}
    if not near:
        return None
    lead = min(near, key=lambda anchor_id: near[anchor_id])  # min() keeps the first listed of equals
    same_place = [anchor_id for anchor_id in near if near[anchor_id] == near[lead] and anchors[anchor_id] == anchors[lead]]
    return next((anchor_id for anchor_id in same_place if anchor_id not in taken), lead), near[lead]


def _text(finding) -> str:
    return f"{finding.file}:{finding.line_start}-{finding.line_end} [{finding.cls}] {finding.claim[:90]}"


def score_tag(findings: list, tag: LiveTag) -> TagScore:
    """Score one run's findings (anything with `file`, `line_start`, `line_end`, `cls` and `claim`) against one tag."""
    result = TagScore()
    bug_at = {bug.id: (bug.file, bug.line) for bug in tag.bugs}
    decoy_at = {decoy.id: (decoy.file, decoy.line) for decoy in tag.decoys}
    found: set = set()
    alarmed: set = set()
    for finding in findings:
        if tag.clean:
            result.clean_false_positives.append(_text(finding))
            continue
        bug = nearest_anchor(finding, bug_at, frozenset(found))
        decoy = nearest_anchor(finding, decoy_at)
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
