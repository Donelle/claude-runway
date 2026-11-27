"""
The two files a review stage of the `codebase-reviewer` server can be run from and can save: the findings file it starts from, and the
results file it writes.

Why they exist. The stages of one review hand their findings to each other in memory, and the text a review returns leaves out what
the next stage needs: it prints a finding's file, lines, reviewers, claim and failing input, not the quote, and a finding with no quote
is dropped as ungrounded. Running one reviewer per session, in separate processes, needs a way to start a stage from findings written
down earlier and a complete record of what a stage produced, so that another process can combine several single-reviewer results with
the functions the stages already use. Neither file is ever read by the model that calls `review`: the caller passes only the path of a
findings file, and the results go to a folder the server's own settings name.

  * The FINDINGS FILE, `read_findings_file(path, stage)`: a JSON object `{"findings": [...]}`. A finding has `file`, `line_start` (and
    `line_end`, default the same line), `class`, `claim`, `failing_input`, `quote`, `reviewers` (the names of the reviewers who raised
    it) and, for the verifier only, optionally `alternates` (the other claims merged into it, which grounding checks one by one as
    it does in a full review). For the classifier a finding also has `verification`: what the verifier made of it, `confirmed`,
    `disputed` or `unchecked`; a rejected finding is never classified, so a file that holds one is refused. The reader reports every
    problem at once and never repeats a value from the file in a message: the path comes from a model, so the file can be any file the
    server can read. `findings_document(findings, statuses)` is the writer's half, for the code that hands one stage's findings on.
  * The RESULTS FILE, `<review type>-<run id>.json` in the folder `REVIEW_OUTPUT_DIR` names, written by `write_results`: for each stage
    that ran, `reasoning_section`, `verifier_section` or `classifier_section` of its review. What each holds is the raw material of the
    stage's combining function, one reviewer at a time: the reasoning reviewers' own findings before they are merged
    (`collect_findings`), each verifier's decision for each grounded finding (`combine`) and each classifier's probability for every rubric
    entry (`classify_answers`); and, for every reviewer, how many requests it sent, how many got a reply and how many of those replies
    could be read. `reasoning_findings`, `verifier_decisions` and `classifier_answers` turn one reviewer's entry back into the objects
    those functions take.

The same dict (`finding_to_dict`) describes a finding in both files, so a finding a results file holds can go into a findings file.

Standard library only, apart from the sibling libs.
"""

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from classifier_reviewer_lib import Classification, ClassificationReview, Rating
from codebase_reviewer_config_lib import ConfigError
from reasoning_reviewer_lib import Alternate, Finding, FindingGroup, ReasoningReview, ReplyCount, ReviewError
from verifier_reviewer_lib import CONFIRMED, DISPUTED, REJECTED, UNCHECKED, Decision, VerificationReview

RESULTS_VERSION = 1

# A findings file is read whole, into memory. It is a few kilobytes (a finding is a few hundred characters), and its path comes from a
# model, so one past this size is a wrong path (a log, a database) and is refused before it is read.
MAX_FINDINGS_FILE_BYTES = 8 * 1024 * 1024
# A line number past this is no line of any file. It also keeps a number that SQLite's INTEGER cannot hold out of the run's record.
MAX_LINE_NUMBER = 10_000_000
# What the verifier stage can make of a finding that the classifier stage may be given. A rejected finding is dropped there.
STANDING_STATUSES = (CONFIRMED, DISPUTED, UNCHECKED)
# A problem list is cut here, and a key name from the file is shown cut at KEY_CHARS, so a wrong file cannot make a reply of any size.
MAX_PROBLEMS_SHOWN = 20
KEY_CHARS = 40

_FINDING_KEYS = ("file", "line_start", "line_end", "class", "claim", "failing_input", "quote", "reviewers", "alternates", "verification")
_ALTERNATE_KEYS = ("line_start", "line_end", "claim", "failing_input", "quote", "reviewers")
_STAGES = ("verifier", "classifier")


# ---------------------------------------------------------------------------
# A finding as a dict
# ---------------------------------------------------------------------------


def _alternate_to_dict(alternate: Alternate) -> dict:
    return {
        "line_start": alternate.line_start,
        "line_end": alternate.line_end,
        "claim": alternate.claim,
        "failing_input": alternate.failing_input,
        "quote": alternate.quote,
        "reviewers": list(alternate.reviewers),
    }


def finding_to_dict(finding: Finding) -> dict:
    """A finding as a JSON-able dict, the shape both files use. `alternates` is there only when the finding has some."""
    data: dict = {
        "file": finding.file,
        "line_start": finding.line_start,
        "line_end": finding.line_end,
        "class": finding.cls,
        "claim": finding.claim,
        "failing_input": finding.failing_input,
        "quote": finding.quote,
        "reviewers": list(finding.reviewers),
    }
    if finding.alternates:
        data["alternates"] = [_alternate_to_dict(a) for a in finding.alternates]
    return data


def _type_name(value: object) -> str:
    return type(value).__name__


def _shown(text: str) -> str:
    return text if len(text) <= KEY_CHARS else text[:KEY_CHARS] + "..."


def _is_count(value: object, high: int) -> bool:
    """A whole number from 0 to `high`. A bool is not a number here."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= high


def _unknown_keys(data: dict, allowed: Sequence[str], where: str, problems: list) -> None:
    extra = sorted(str(key) for key in data if key not in allowed)
    if extra:
        names = ", ".join(repr(_shown(key)) for key in extra[:5]) + (f" and {len(extra) - 5} more" if len(extra) > 5 else "")
        problems.append(f"{where}: unknown key(s) {names} (allowed: {', '.join(allowed)})")


def _text(data: dict, key: str, where: str, problems: list, *, blank_ok: bool) -> Optional[str]:
    """The string at `key`, or None with a problem recorded: missing, not a string, or blank where that is not allowed."""
    if key not in data:
        problems.append(f"{where}: missing {key!r}")
        return None
    value = data[key]
    if not isinstance(value, str):
        problems.append(f"{where}: {key!r} must be a string (got {_type_name(value)})")
        return None
    if not blank_ok and not value.strip():
        problems.append(f"{where}: {key!r} must not be blank")
        return None
    return value


def _lines(data: dict, where: str, problems: list) -> Optional[tuple]:
    """`(line_start, line_end)`: `line_start` is required (0 means the reviewer gave no line), `line_end` defaults to it."""
    if "line_start" not in data:
        problems.append(f"{where}: missing 'line_start'")
        return None
    start = data["line_start"]
    if not _is_count(start, MAX_LINE_NUMBER):
        problems.append(f"{where}: 'line_start' must be a whole number from 0 to {MAX_LINE_NUMBER} (0 is no line number)")
        return None
    end = data.get("line_end", start)
    if not _is_count(end, MAX_LINE_NUMBER) or end < start:
        problems.append(f"{where}: 'line_end' must be a whole number from 'line_start' to {MAX_LINE_NUMBER}")
        return None
    return start, end


def _reviewers(data: dict, where: str, problems: list) -> Optional[tuple]:
    if "reviewers" not in data:
        problems.append(f"{where}: missing 'reviewers'")
        return None
    value = data["reviewers"]
    if not isinstance(value, list) or not value or not all(isinstance(name, str) and name.strip() for name in value):
        problems.append(f"{where}: 'reviewers' must be a non-empty list of reviewer names (strings)")
        return None
    return tuple(value)


def _alternate(data: object, where: str, problems: list) -> Optional[Alternate]:
    if not isinstance(data, dict):
        problems.append(f"{where}: must be an object (got {_type_name(data)})")
        return None
    before = len(problems)
    _unknown_keys(data, _ALTERNATE_KEYS, where, problems)
    lines = _lines(data, where, problems)
    claim = _text(data, "claim", where, problems, blank_ok=False)
    failing = _text(data, "failing_input", where, problems, blank_ok=True)
    quote = _text(data, "quote", where, problems, blank_ok=True)
    reviewers = _reviewers(data, where, problems)
    if len(problems) != before or lines is None or claim is None or failing is None or quote is None or reviewers is None:
        return None
    return Alternate(claim=claim, failing_input=failing, reviewers=reviewers, line_start=lines[0], line_end=lines[1], quote=quote)


def _finding(data: object, where: str, problems: list, *, alternates_ok: bool) -> Optional[Finding]:
    """The finding `data` describes, or None with every problem found in it added to `problems`. A value is never repeated in a
    problem (only a key name, cut short, or a type), since the file can be any file a model named."""
    if not isinstance(data, dict):
        problems.append(f"{where}: must be an object (got {_type_name(data)})")
        return None
    before = len(problems)
    _unknown_keys(data, _FINDING_KEYS, where, problems)
    file = _text(data, "file", where, problems, blank_ok=True)
    lines = _lines(data, where, problems)
    cls = _text(data, "class", where, problems, blank_ok=False)
    claim = _text(data, "claim", where, problems, blank_ok=False)
    failing = _text(data, "failing_input", where, problems, blank_ok=True)
    quote = _text(data, "quote", where, problems, blank_ok=True)
    reviewers = _reviewers(data, where, problems)
    alternates: list = []
    if "alternates" in data:
        raw = data["alternates"]
        if not alternates_ok:
            problems.append(
                f"{where}: 'alternates' is not accepted here: the classifier takes one claim per finding, so list each alternate as a "
                "finding of its own"
            )
        elif not isinstance(raw, list):
            problems.append(f"{where}: 'alternates' must be a list (got {_type_name(raw)})")
        else:
            built = [_alternate(item, f"{where}.alternates[{number}]", problems) for number, item in enumerate(raw)]
            alternates = [a for a in built if a is not None]
    if len(problems) != before or file is None or lines is None or cls is None or claim is None or failing is None or quote is None or reviewers is None:
        return None
    return Finding(
        file=file, line_start=lines[0], line_end=lines[1], cls=cls, claim=claim, failing_input=failing, reviewers=reviewers,
        alternates=tuple(alternates), quote=quote,
    )


def finding_from_dict(data: object, where: str = "finding") -> Finding:
    """The finding a `finding_to_dict` dict describes (a findings file's entry too: its `verification` key is not read here).
    Raises `ValueError` listing every problem."""
    problems: list = []
    finding = _finding(data, where, problems, alternates_ok=True)
    if finding is None:
        raise ValueError("; ".join(problems))
    return finding


# ---------------------------------------------------------------------------
# The findings file
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FindingsInput:
    """What a findings file holds: the findings in file order and, for the classifier stage, what the verifier stage made of each
    (`statuses`, parallel to `findings`; empty for the verifier stage)."""

    findings: tuple
    statuses: tuple = ()


def findings_document(findings: Sequence[Finding], statuses: Optional[Sequence[str]] = None) -> dict:
    """The JSON-able content of a findings file for `findings`. With `statuses` (what the verifier stage made of each one) every
    finding carries it as `verification`, which is what the classifier stage's file needs."""
    if statuses is not None and len(statuses) != len(findings):
        raise ValueError(f"statuses has {len(statuses)} item(s) for {len(findings)} finding(s)")
    items = []
    for number, finding in enumerate(findings):
        item = finding_to_dict(finding)
        if statuses is not None:
            item["verification"] = statuses[number]
        items.append(item)
    return {"findings": items}


def _problems_error(shown: str, problems: list) -> ReviewError:
    count = len(problems)
    lines = [f"findings_file {shown!r} is not usable ({count} problem{'s' if count != 1 else ''}):"]
    lines.extend(f"  - {problem}" for problem in problems[:MAX_PROBLEMS_SHOWN])
    if count > MAX_PROBLEMS_SHOWN:
        lines.append(f"  - ... and {count - MAX_PROBLEMS_SHOWN} more")
    return ReviewError("\n".join(lines))


def _read_json(path: Path) -> object:
    """The JSON in the file at `path`; a `ReviewError` that names the path says what is wrong otherwise. Only a regular file is read
    (not a pipe or a device, which could never end) and at most `MAX_FINDINGS_FILE_BYTES` of it."""
    shown = str(path)
    try:
        regular = path.is_file()
    except OSError as exc:
        # `is_file` answers False only for a path that is not there. A folder the server may not enter, or a name too long to exist,
        # raises, and a path a model named can be either: that is an `Error:` for the model, not a crash.
        raise ReviewError(f"findings_file {shown!r} cannot be read ({exc.strerror or type(exc).__name__})") from exc
    if not regular:
        raise ReviewError(f"findings_file {shown!r} is not a file")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_FINDINGS_FILE_BYTES + 1)
    except OSError as exc:
        raise ReviewError(f"findings_file {shown!r} cannot be read ({exc.strerror or type(exc).__name__})") from exc
    if len(raw) > MAX_FINDINGS_FILE_BYTES:
        raise ReviewError(
            f"findings_file {shown!r} is longer than {MAX_FINDINGS_FILE_BYTES} bytes, which is more than a findings file holds: "
            "is it the right path?"
        )
    try:
        # utf-8-sig: a file saved by a Windows tool may start with a byte-order mark, which plain UTF-8 JSON reading refuses.
        return json.loads(raw.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ReviewError(f"findings_file {shown!r} is not UTF-8 text") from exc
    except (ValueError, RecursionError) as exc:
        # The JSON reader's message names a position, never the text; a RecursionError is a file nested too deeply to read.
        raise ReviewError(f"findings_file {shown!r} is not valid JSON ({exc})") from exc


def read_findings_file(path: str, stage: str) -> FindingsInput:
    """The findings in the findings file at `path`, for the `stage` ("verifier" or "classifier") that will run on them.

    A leading `~` in `path` is expanded; a relative path is the server's working directory's. Raises `ReviewError`, naming the path
    and listing EVERY problem, when the file is missing, is not a regular file, is too large, is not UTF-8 JSON, or any finding is wrong
    (see the module docstring for what a finding needs). The verifier stage reads `alternates`, which it checks one claim at a time, and
    ignores `verification`; the classifier stage needs `verification` on every finding (a rejected finding is refused, because the
    stage takes the findings that were not rejected) and refuses `alternates`, since it would classify only the first claim of such a
    finding and silently lose the rest. The findings are returned as written: nothing is merged, corrected or dropped here."""
    if stage not in _STAGES:
        raise ValueError(f"unknown stage {stage!r}; a findings file is read for: {', '.join(_STAGES)}")
    resolved = Path(os.path.expanduser(path))
    shown = str(resolved)
    data = _read_json(resolved)
    if not isinstance(data, dict):
        raise ReviewError(f"findings_file {shown!r} must hold a JSON object with a 'findings' list (got {_type_name(data)})")
    problems: list = []
    extra = sorted(key for key in data if key != "findings")
    if extra:
        problems.append(f"unknown top-level key(s) {', '.join(repr(_shown(key)) for key in extra[:5])} (allowed: findings)")
    items = data.get("findings")
    findings: list = []
    statuses: list = []
    if not isinstance(items, list):
        problems.append("missing 'findings'" if "findings" not in data else f"'findings' must be a list (got {_type_name(items)})")
        items = []
    for number, item in enumerate(items):
        where = f"findings[{number}]"
        finding = _finding(item, where, problems, alternates_ok=stage == "verifier")
        if stage == "classifier" and isinstance(item, dict):
            status = item.get("verification")
            if "verification" not in item:
                problems.append(f"{where}: missing 'verification' (what the verifier made of it: {', '.join(STANDING_STATUSES)})")
            elif status == REJECTED:
                problems.append(f"{where}: the verifiers rejected it; the classifier stage takes the findings that were not rejected")
            elif not isinstance(status, str) or status not in STANDING_STATUSES:
                problems.append(f"{where}: 'verification' must be one of {', '.join(STANDING_STATUSES)}")
            else:
                statuses.append(status)
        if finding is not None:
            findings.append(finding)
    if problems:
        raise _problems_error(shown, problems)
    return FindingsInput(findings=tuple(findings), statuses=tuple(statuses))


def supplied_reasoning(findings: Sequence[Finding], stage: str) -> ReasoningReview:
    """The `ReasoningReview` a run that started from supplied findings reports in place of a reasoning stage that did not run: the
    findings grouped by class WITHOUT merging (they are what the caller said to check, so they stay as they were given), the reviewers
    the findings name as the ones who ran, and a warning that says where the findings came from."""
    by_class: dict = {}
    for finding in findings:
        by_class.setdefault(finding.cls, []).append(finding)
    names: list = []
    for finding in findings:
        for name in (*finding.reviewers, *(who for alternate in finding.alternates for who in alternate.reviewers)):
            if name not in names:
                names.append(name)
    skipped = "reasoning" if stage == "verifier" else "reasoning and verifier"
    return ReasoningReview(
        groups=[FindingGroup(cls=cls, findings=tuple(items)) for cls, items in sorted(by_class.items())],
        reviewers_run=names,
        warnings=[f"The findings were read from a findings file: the {skipped} stage{'s' if stage != 'verifier' else ''} did not run."],
    )


# ---------------------------------------------------------------------------
# The results file
# ---------------------------------------------------------------------------


def _counts(counts: ReplyCount) -> dict:
    return {"requests": counts.requests, "replies": counts.replies, "readable": counts.readable}


def decision_to_dict(decision: Decision, finding: int) -> dict:
    """One verifier's decision about the finding at position `finding` of the section's `findings`."""
    return {
        "finding": finding,
        "verdict": decision.verdict,
        "failing_input": decision.failing_input,
        "wrong_result": decision.wrong_result,
        "reasoning": decision.reasoning,
    }


def classification_to_dict(classification: Classification, finding: int) -> dict:
    """One classifier's probability for every rubric entry, for the finding at position `finding` of the section's `findings`."""
    return {
        "finding": finding,
        "probabilities": [{"entry": rating.entry, "probability": rating.probability} for rating in classification.ratings],
    }


def reasoning_section(review: ReasoningReview) -> dict:
    """The reasoning stage's results. Each reviewer's own findings are kept as that reviewer raised them, before any merging: feeding
    every reviewer's `reasoning_findings` to `collect_findings`, in the order the reviewers ran, gives the stage's merged findings.
    A reviewer that failed has an `error` and no counts."""
    reviewers = []
    for outcome in review.outcomes:
        entry: dict = {"name": outcome.reviewer.name}
        if outcome.error is not None:
            entry["error"] = outcome.error
        else:
            result = outcome.value
            entry["chunks"] = result.chunks
            entry.update(_counts(result.counts))
            entry["findings"] = [finding_to_dict(f) for f in result.findings]
        reviewers.append(entry)
    return {
        "chunks": review.chunks,
        "reviewers": reviewers,
        "errors": list(review.errors),
        "warnings": list(review.warnings),
    }


def verifier_section(review: VerificationReview) -> dict:
    """The verifier stage's results: `findings`, the findings that were grounded (their lines corrected to where the quote is), in the
    order every verifier's `finding` position refers to, and `ungrounded`, the findings dropped before any model saw them. For each
    verifier its decisions and the positions of the findings it kept asking about (`gave_up`); feeding every verifier's
    `verifier_decisions` for a position to `combine` gives that finding's status. A verifier that failed has an `error` and no counts."""
    reviewers = []
    for outcome in review.outcomes:
        entry: dict = {"name": outcome.reviewer.name}
        if outcome.error is not None:
            entry["error"] = outcome.error
        else:
            result = outcome.value
            entry.update(_counts(result.counts))
            entry["decisions"] = [decision_to_dict(decision, position) for position, decision in sorted(result.decisions.items())]
            entry["gave_up"] = sorted(result.limit_hit)
        reviewers.append(entry)
    return {
        "findings": [finding_to_dict(v.finding) for v in review.verified],
        "ungrounded": [{"finding": finding_to_dict(u.finding), "reason": u.reason} for u in review.ungrounded],
        "reviewers": reviewers,
        "errors": list(review.errors),
        "warnings": list(review.warnings),
    }


def classifier_section(review: ClassificationReview) -> dict:
    """The classifier stage's results: `findings`, what was put to the classifiers (each with the `verification` status it arrived
    with), in the order every classifier's `finding` position refers to. For each classifier its probability for every rubric entry,
    for each finding it answered; feeding every classifier's `classifier_answers` for a position to `classify_answers` gives that
    finding's outcome. A classifier that failed has an `error` and no counts."""
    reviewers = []
    for outcome in review.outcomes:
        entry: dict = {"name": outcome.reviewer.name}
        if outcome.error is not None:
            entry["error"] = outcome.error
        else:
            result = outcome.value
            entry.update(_counts(result.counts))
            entry["answers"] = [classification_to_dict(answer, position) for position, answer in sorted(result.answers.items())]
        reviewers.append(entry)
    findings = []
    for item in review.classified:
        data = finding_to_dict(item.finding)
        if item.verification:
            data["verification"] = item.verification
        findings.append(data)
    return {
        "findings": findings,
        "reviewers": reviewers,
        "errors": list(review.errors),
        "warnings": list(review.warnings),
    }


def reasoning_findings(entry: dict) -> list:
    """The findings of one reviewer's entry in a results file's `reasoning` section, as `Finding`s (none for a reviewer that failed)."""
    return [finding_from_dict(item, f"findings[{number}]") for number, item in enumerate(entry.get("findings", []))]


def verifier_decisions(entry: dict) -> dict:
    """One verifier's entry in a results file's `verifier` section as `{finding position: Decision}` (empty for a verifier that failed)."""
    return {
        item["finding"]: Decision(entry["name"], item["verdict"], item["failing_input"], item["wrong_result"], item["reasoning"])
        for item in entry.get("decisions", [])
    }


def classifier_answers(entry: dict) -> dict:
    """One classifier's entry in a results file's `classifier` section as `{finding position: Classification}` (empty for a classifier
    that failed). The probabilities are exactly the floats the classifier gave (JSON keeps them whole), so an average over them is
    the average of the run."""
    return {
        item["finding"]: Classification(entry["name"], tuple(Rating(p["entry"], p["probability"]) for p in item["probabilities"]))
        for item in entry.get("answers", [])
    }


def results_file_name(review_type: str, run_id: str) -> str:
    """`<review type>-<run id>.json`: the run id is unique to a run, so reviewers sharing a folder never overwrite each other."""
    return f"{review_type}-{run_id}.json"


def prepare_results_folder(raw: str) -> Path:
    """The folder results are written to, created if it is missing, checked BEFORE a review spends minutes of model time: a folder
    that cannot be used is a `ConfigError` naming it, which the review reports as an `Error:`. A leading `~` is expanded."""
    folder = Path(os.path.expanduser(raw))
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"REVIEW_OUTPUT_DIR {str(folder)!r} cannot be created ({exc.strerror or type(exc).__name__})") from exc
    if not folder.is_dir():
        raise ConfigError(f"REVIEW_OUTPUT_DIR {str(folder)!r} is not a folder")
    if not os.access(folder, os.W_OK | os.X_OK):
        raise ConfigError(f"REVIEW_OUTPUT_DIR {str(folder)!r} cannot be written to")
    return folder


def write_results(folder: Path, results: dict) -> Path:
    """Writes `results` as `results_file_name(...)` in `folder` and returns its path.

    Written to a temporary name in the same folder and renamed into place, so a process listing the folder never meets a half-written file
    (the temporary name ends in `.tmp`, which a `*.json` listing does not match), and several reviews sharing a folder (one process each)
    cannot interfere. Raises `OSError` when it cannot write, and `ValueError` for a value JSON cannot hold (NaN, infinity): the
    caller decides what that means for its review."""
    final = folder / results_file_name(results["type"], results["run_id"])
    temporary = folder / f".{final.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        # "x": never overwrite a file that is already there. A new file gets the process's default permissions (a temp-file helper
        # would make it readable by its owner only, and the reader may be another user: a container writes, the host reads).
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, final)
    except BaseException:
        # Whatever went wrong, no half-written temporary file is left behind; the error itself still reaches the caller.
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return final


__all__ = [
    "RESULTS_VERSION",
    "MAX_FINDINGS_FILE_BYTES",
    "STANDING_STATUSES",
    "FindingsInput",
    "finding_to_dict",
    "finding_from_dict",
    "findings_document",
    "read_findings_file",
    "supplied_reasoning",
    "decision_to_dict",
    "classification_to_dict",
    "reasoning_section",
    "verifier_section",
    "classifier_section",
    "reasoning_findings",
    "verifier_decisions",
    "classifier_answers",
    "results_file_name",
    "prepare_results_folder",
    "write_results",
]
