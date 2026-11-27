#!/usr/bin/env python3
"""Score one `pi_driver.py` run, to compare callers: did a review run, and how many of the change's seeded bugs does its text name.

The driver prints a JSON summary whose `review_text` is the reviewer tool's own result and whose `change` names the change of
the `harness-live-tests` branch that was reviewed (`change-6`). It does not say whether that text is a review at all (a caller's
malformed call or a timeout produces text that is not), nor how many of the change's known defects it names; this does, against
the answer key (`harness/live-tests/answer_key.json`), so a comparison of callers records the same numbers for every run:

    .venv/bin/python harness/pi/pi_driver.py --model <caller> --config <config> > run.json
    .venv/bin/python harness/pi/score_run.py run.json            # or: ... | score_run.py -

A development tool, not product code (CI does not lint or type-check `harness/`; `tests/test_harness_pi_score_run.py` covers
it, no Docker or LM Studio needed). It reads the finding lines the reasoning review prints (`format_review` in
libs/reasoning_reviewer_lib.py: `- file:start[-end] [reviewers] claim`) and scores them with `live_scoring_lib.score_tag`, by
file and nearby line only: a finding near a bug's line is a hit; at a decoy's line it is a false alarm; any finding on the clean
change (`change-5`) is a false alarm; anything else is unscored, shown and never counted against the model. Standard library only.

Exit code: 0 = scored; 2 = the input could not be read, names no change or one the answer key does not list, or the key
itself could not be read.
"""

import argparse
import codecs
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
# `live_scoring_lib` (in harness/) is the only harness module this uses, and it depends on nothing else there: neither `cases.py`
# nor `harness_lib.py` is imported, here or through it.
if str(REPO_ROOT / "harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "harness"))

import live_scoring_lib as scoring  # noqa: E402

# "- backup.py:6-8 [general] claim", or "- backup.py [general] claim" when the reviewer gave no line, or "(unknown file)".
FINDING_LINE = re.compile(r"^- (?P<file>\(unknown file\)|[^\s:\[]+)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)? \[(?P<who>[^\]]*)\] (?P<claim>.*)$")
CLASS_LINE = re.compile(r"^## (?P<cls>.+?) \(\d+\)$")
INCOMPLETE_PREFIX = "INCOMPLETE REVIEW:"
REFUSAL_PREFIX = "Error:"
# The line a reasoning review's text opens with (after an INCOMPLETE REVIEW line, when there is one): `format_review`.
REVIEW_HEADER = "Reasoning review:"


@dataclass
class ParsedFinding:
    """What `live_scoring_lib.score_tag` reads from a finding: where it points, its class and its claim."""

    file: str
    line_start: int
    line_end: int
    cls: str
    claim: str


def parse_review_text(text: str) -> list:
    """The findings a reasoning review's text lists, in order. `also, at ...` alternates are separate defects that share
    lines with a finding, so they are read too (their file is the finding's). A line with no number gets 0, which
    scoring treats as not pointing anywhere."""
    findings: list = []
    cls = ""
    last_file = ""
    for raw in text.splitlines():
        heading = CLASS_LINE.match(raw)
        if heading:
            cls = heading.group("cls")
            continue
        match = FINDING_LINE.match(raw)
        if match:
            start = int(match.group("start") or 0)
            end = int(match.group("end") or start)
            last_file = match.group("file")
            findings.append(ParsedFinding(last_file, start, end, cls, match.group("claim")))
            continue
        alternate = re.match(r"^\s+also, at (?P<start>\d+)(?:-(?P<end>\d+))? \[[^\]]*\]: (?P<claim>.*)$", raw)
        if alternate and last_file:
            start = int(alternate.group("start"))
            findings.append(ParsedFinding(last_file, start, int(alternate.group("end") or start), cls, alternate.group("claim")))
    return findings


@dataclass
class RunScore:
    readable: bool  # the review text is a result, not a refusal, a failure or nothing
    change: str = ""  # the change the run reviewed, as the driver named it
    found: list = field(default_factory=list)  # bug ids a finding sits at
    missed: list = field(default_factory=list)  # bug ids no finding sits at (all of them when the review is not readable)
    on_decoys: list = field(default_factory=list)  # decoy ids a finding sits at: each is a false alarm
    clean_false_alarms: list = field(default_factory=list)  # findings on the clean change (`change-5`): each is a false alarm
    unscored: list = field(default_factory=list)  # findings at no listed bug or decoy: shown, never counted against the model
    findings: int = 0
    # How many of the review's `file:line` places the caller's final message repeats (the caller's summary can drop most of it).
    final_repeats: int = 0
    note: str = ""


def final_repeats(final_text: str, findings: list) -> int:
    """Findings whose `file:line` (the start line alone is enough) the caller's final message names again."""
    return sum(1 for f in findings if f.line_start > 0 and re.search(rf"{re.escape(f.file)}:{f.line_start}\b", final_text or ""))


def score_summary(summary: dict, tags: Optional[tuple] = None) -> RunScore:
    """Scores one driver summary against the answer key's entry for the change it names.

    `tags` is the key as `live_scoring_lib.load_answer_key` returns it; left out, the key file is read. Raises ValueError when
    the summary names no change or one the key does not list, and `LiveMetadataError` (also a ValueError) when the key cannot be
    read; each message says which."""
    change = summary.get("change")
    if not isinstance(change, str) or not change:
        raise ValueError("the summary names no change, so there is no answer key entry to score it against: it is not the output of a "
                         "pi_driver.py that reviews the harness-live-tests branch (an older driver ran a case from cases.py). Run the driver again")
    if tags is None:
        tags = scoring.load_answer_key()
    tag = next((t for t in tags if t.tag_id == change), None)
    if tag is None:
        raise ValueError(f"the summary's change {change!r} is not in the answer key, which lists {', '.join(t.tag_id for t in tags)}")
    every_bug = [bug.id for bug in tag.bugs]
    # A hand-edited file can hold anything here; a number or null reads as no text, not as a crash.
    text = summary.get("review_text")
    text = text if isinstance(text, str) else ""
    final_text = summary.get("final_text")
    final_text = final_text if isinstance(final_text, str) else ""
    stripped = text.lstrip()
    if not stripped:
        return RunScore(False, change, missed=every_bug, note="no review text: the review tool returned nothing")
    if stripped.startswith(REFUSAL_PREFIX):
        return RunScore(False, change, missed=every_bug, note="the review tool refused: " + stripped.splitlines()[0])
    if REVIEW_HEADER not in stripped.splitlines()[0] and not stripped.startswith(INCOMPLETE_PREFIX):
        # Not a review at all: Pi's own "Validation failed for tool ..." (the caller sent the call wrongly), "MCP request timed
        # out after 60000ms", or anything else the tool's result does not open with. A reviewer never ran, or never answered.
        return RunScore(False, change, missed=every_bug, note="no review ran: " + stripped.splitlines()[0][:90])
    findings = parse_review_text(text)
    scored = scoring.score_tag(findings, tag)
    # An INCOMPLETE REVIEW header means a reviewer's reply was unreadable or failed: any bug missing may be that, not the model's eye.
    note = "INCOMPLETE REVIEW: a reviewer reported a problem, so a missed bug may be a failed reply" if stripped.startswith(INCOMPLETE_PREFIX) else ""
    return RunScore(
        readable=not note, change=change, found=scored.found, missed=scored.missed, on_decoys=scored.on_decoys,
        clean_false_alarms=scored.clean_false_positives, unscored=scored.unscored, findings=len(findings),
        final_repeats=final_repeats(final_text, findings), note=note,
    )


def one_line(summary: dict, score: RunScore) -> str:
    """The row to paste into the README's table, as `key=value` pairs."""
    bugs = len(score.found) + len(score.missed)
    return (
        f"change={score.change} stopped_by={summary.get('stopped_by')} readable={'yes' if score.readable else 'NO'} "
        f"found={len(score.found)}/{bugs} decoy_alarms={len(score.on_decoys)} clean_false_alarms={len(score.clean_false_alarms)} "
        f"unscored={len(score.unscored)} findings={score.findings} "
        f"review_calls={summary.get('review_calls')} review_errors={summary.get('review_errors')} "
        f"turns={summary.get('turns')} tool_calls={summary.get('tool_calls')} "
        f"seconds={summary.get('elapsed_seconds')} caller_repeated={score.final_repeats}/{score.findings}"
        + (f" ({score.note})" if score.note else "")
    )


def load_summary(source: str) -> Optional[dict]:
    """The driver's JSON from a file, or from stdin for `-`. None when it is not a JSON object.

    A file is decoded by its byte order mark: Windows PowerShell 5.1's `>` writes UTF-16, and some editors put a mark on
    UTF-8. Raises UnicodeDecodeError for a file in neither."""
    if source == "-":
        raw = sys.stdin.read()
    else:
        data = Path(source).read_bytes()
        raw = data.decode("utf-16" if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig")
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Score one pi_driver.py JSON summary against the answer key's entry for the change it reviewed.")
    parser.add_argument("summary", help="the driver's JSON output file, or - for stdin")
    parser.add_argument("--json", action="store_true", help="print the score as JSON instead of one line")
    args = parser.parse_args(argv)
    try:
        summary = load_summary(args.summary)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"error: cannot read {args.summary}: {exc}", file=sys.stderr)
        return 2
    if summary is None:
        print(f"error: {args.summary} is not a JSON object (the driver prints one on stdout)", file=sys.stderr)
        return 2
    try:
        score = score_summary(summary)
    except ValueError as exc:  # no change, an unknown one, or a key that cannot be read (LiveMetadataError): the message says which
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(score), indent=2) if args.json else one_line(summary, score))
    return 0


if __name__ == "__main__":
    sys.exit(main())
