#!/usr/bin/env python3
"""Score one `pi_driver.py` run, to compare callers: did a review run, and how many seeded bugs does its text name.

The driver prints a JSON summary whose `review_text` is the reviewer tool's own result. It does not say whether that text
is a review at all (a caller's malformed call or a timeout produces text that is not), nor how many of the case's known defects
it names; this does, so a comparison of callers records the same numbers for every run:

    .venv/bin/python harness/pi/pi_driver.py --model <caller> --config <config> > run.json
    .venv/bin/python harness/pi/score_run.py run.json            # or: ... | score_run.py -

A development tool, not product code (CI does not lint or type-check `harness/`; `tests/test_harness_pi_score_run.py` covers
it, no Docker or LM Studio needed). It reads the finding lines the reasoning review prints (`format_review` in
libs/reasoning_reviewer_lib.py: `- file:start[-end] [reviewers] claim`) and matches them to the bugs with the harness's own
`score()`, by file and line. Standard library only.

Exit code: 0 = scored; 2 = the input could not be read or the case is unknown.
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
DEFAULT_CASE = "seeded-three"

# "- backup.py:6-8 [general] claim", or "- backup.py [general] claim" when the reviewer gave no line, or "(unknown file)".
FINDING_LINE = re.compile(r"^- (?P<file>\(unknown file\)|[^\s:\[]+)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)? \[(?P<who>[^\]]*)\] (?P<claim>.*)$")
CLASS_LINE = re.compile(r"^## (?P<cls>.+?) \(\d+\)$")
INCOMPLETE_PREFIX = "INCOMPLETE REVIEW:"
REFUSAL_PREFIX = "Error:"
# The line a reasoning review's text opens with (after an INCOMPLETE REVIEW line, when there is one): `format_review`.
REVIEW_HEADER = "Reasoning review:"


@dataclass
class ParsedFinding:
    """What `harness_lib.score` reads from a finding: where it points, its class and its claim."""

    file: str
    line_start: int
    line_end: int
    cls: str
    claim: str


def parse_review_text(text: str) -> list:
    """The findings a reasoning review's text lists, in order. `also, at ...` alternates are separate defects that share
    lines with a finding, so they are read too (their file is the finding's). A line with no number gets 0, which
    `score()` treats as not pointing anywhere."""
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
    found: list = field(default_factory=list)
    missed: list = field(default_factory=list)
    on_decoys: list = field(default_factory=list)
    other: list = field(default_factory=list)
    findings: int = 0
    # How many of the review's `file:line` places the caller's final message repeats (the caller's summary can drop most of it).
    final_repeats: int = 0
    note: str = ""


def final_repeats(final_text: str, findings: list) -> int:
    """Findings whose `file:line` (the start line alone is enough) the caller's final message names again."""
    return sum(1 for f in findings if f.line_start > 0 and re.search(rf"{re.escape(f.file)}:{f.line_start}\b", final_text or ""))


def score_summary(summary: dict, case_name: str = DEFAULT_CASE) -> RunScore:
    """Scores one driver summary. Raises KeyError for an unknown case."""
    for directory in (REPO_ROOT / "libs", REPO_ROOT / "harness"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import cases
    import harness_lib

    case = cases.ALL_CASES[case_name]
    text = summary.get("review_text") or ""
    stripped = text.lstrip()
    if not stripped:
        return RunScore(False, missed=[b.id for b in case.bugs], note="no review text: the review tool returned nothing")
    if stripped.startswith(REFUSAL_PREFIX):
        return RunScore(False, missed=[b.id for b in case.bugs], note="the review tool refused: " + stripped.splitlines()[0])
    if REVIEW_HEADER not in stripped.splitlines()[0] and not stripped.startswith(INCOMPLETE_PREFIX):
        # Not a review at all: Pi's own "Validation failed for tool ..." (the caller sent the call wrongly), "MCP request timed
        # out after 60000ms", or anything else the tool's result does not open with. A reviewer never ran, or never answered.
        return RunScore(False, missed=[b.id for b in case.bugs], note="no review ran: " + stripped.splitlines()[0][:90])
    findings = parse_review_text(text)
    scored = harness_lib.score(findings, case)
    # An INCOMPLETE REVIEW header means a reviewer's reply was unreadable or failed: any bug missing may be that, not the model's eye.
    note = "INCOMPLETE REVIEW: a reviewer reported a problem, so a missed bug may be a failed reply" if stripped.startswith(INCOMPLETE_PREFIX) else ""
    return RunScore(
        readable=not note, found=scored.found, missed=scored.missed, on_decoys=scored.on_decoys, other=scored.other,
        findings=len(findings), final_repeats=final_repeats(summary.get("final_text", ""), findings), note=note,
    )


def one_line(summary: dict, score: RunScore) -> str:
    """The row to paste into the README's table, as `key=value` pairs."""
    bugs = len(score.found) + len(score.missed)
    return (
        f"stopped_by={summary.get('stopped_by')} readable={'yes' if score.readable else 'NO'} found={len(score.found)}/{bugs} "
        f"findings={score.findings} review_calls={summary.get('review_calls')} review_errors={summary.get('review_errors')} "
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
    parser = argparse.ArgumentParser(description="Score one pi_driver.py JSON summary against its case's seeded bugs.")
    parser.add_argument("summary", help="the driver's JSON output file, or - for stdin")
    parser.add_argument("--case", default=DEFAULT_CASE, help=f"the case the run used (default {DEFAULT_CASE})")
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
        score = score_summary(summary, args.case)
    except KeyError:
        print(f"error: unknown case {args.case!r}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(score), indent=2) if args.json else one_line(summary, score))
    return 0


if __name__ == "__main__":
    sys.exit(main())
