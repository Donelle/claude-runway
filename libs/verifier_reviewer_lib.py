"""
The verifier stage of the `codebase-reviewer` server: drop the findings that are about code that is not
there, then let reasoning-only verifier models confirm or reject the rest.

What this module does, in the order a verified review uses it:

  1. `ground_findings` is the deterministic step, and runs before any model reads a finding. Every
     finding carries a `quote`: the exact line of code it accuses. The quote must occur in the cited
     file (compared with whitespace normalized). A finding that passes is kept and its line numbers are
     corrected to where the quote really is; one whose quote is missing or not in the file is dropped
     and recorded as *ungrounded*. In live runs of 7B reviewers a large share of the findings were about
     code that is not there (a "bare except" at the lines of a docstring, a "missing encoding", an
     "unused import" of a module that is used), and a string search catches all of those for free,
     leaving the model verifiers only the claims that are about real code.
  2. `verify_findings` shows each remaining finding, with the code it cites, to every active verifier
     through `reviewer_scheduler_lib.run_stages` (so grouping, parallelism and model loading are the
     scheduler's). A verifier is reasoning-only: it must confirm with a concrete failing input and the
     wrong result it produces, or reject. It runs nothing itself: one with `tools` in the reviewers file may instead
     ask the server to look something up (`need:` lines, `reviewer_need_lib`) and is asked again with the result.
  3. `combine` turns the verifiers' answers into one outcome per finding: every verifier that answered
     confirms -> *confirmed*; every one rejects -> *rejected* (dropped, recorded); they disagree -> a
     *dispute*, returned with each verifier's decision and reasoning; none answered (or none is
     active) -> *unchecked*, which is kept and says so. Nothing is dropped silently.
  4. `run_verified_review` runs the reasoning stage first and hands its findings to `verify_findings`.

The cited code is the file as it stands at the head of the reviewed range (`git_file_reader`), and only
for files in the change, so a finding cannot make the server read an arbitrary path.

Standard library only, apart from the sibling libs. The model client and the completion call are passed
in, so tests run with a stub and no LM Studio.
"""

import bisect
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import openai_lib
from credential_redaction_lib import display_safe
from codebase_reviewer_config_lib import Reviewer, ReviewersConfig, load_prompt
from lmstudio_model_loader_lib import ModelLoader
from reasoning_reviewer_lib import (
    MIN_QUOTE_CHARS,
    Finding,
    ReasoningReview,
    ReviewError,
    _extract_json,
    chunk_chars_for,
    describe_finding,
    finish_request,
    loop_enabled,
    loop_reserve,
    instructions_from_file,
    loop_unavailable,
    need_format_for,
    prompt_hash_of,
    run_reasoning_review,
    split_diff,
    with_need_instructions,
)
from reviewer_db_lib import NULL_RECORDER
from reviewer_need_lib import ASKED, LIMIT, NeedSession
from reviewer_scheduler_lib import check_resident_models, plan_stages, run_stages

# Lines of code shown above and below the lines a finding cites; shrunk when the verifier's context is small.
CODE_RADIUS_LINES = 15
# Findings beyond this many are not verified (each costs one request per verifier) and come back unchecked.
MAX_VERIFIED_FINDINGS = 100
# The longest `git show` may take. A file is read whole: `git show` has already held all of it in memory,
# so cutting it short saved nothing and made a quote past the cut look absent from the file.
GIT_TIMEOUT_SECONDS = 10

CONFIRMED = "confirmed"
REJECTED = "rejected"
DISPUTED = "disputed"
UNCHECKED = "unchecked"


@dataclass(frozen=True)
class Ungrounded:
    """A finding dropped before any model saw it, and why."""

    finding: Finding
    reason: str


@dataclass(frozen=True)
class Decision:
    """One verifier's answer about one finding. `verdict` is `confirmed` or `rejected`."""

    reviewer: str
    verdict: str
    failing_input: str = ""
    wrong_result: str = ""
    reasoning: str = ""


@dataclass(frozen=True)
class Verified:
    """One finding and what the verifiers made of it. `status` is `confirmed`, `rejected`, `disputed`
    or `unchecked`; `note` says why for an unchecked one."""

    finding: Finding
    status: str
    decisions: tuple = ()
    note: str = ""


@dataclass
class VerificationReview:
    verified: list = field(default_factory=list)
    ungrounded: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    verifiers_run: list = field(default_factory=list)

    def with_status(self, status: str) -> list:
        return [v for v in self.verified if v.status == status]

    @property
    def confirmed(self) -> list:
        return self.with_status(CONFIRMED)

    @property
    def rejected(self) -> list:
        return self.with_status(REJECTED)

    @property
    def disputed(self) -> list:
        return self.with_status(DISPUTED)

    @property
    def unchecked(self) -> list:
        return self.with_status(UNCHECKED)

    @property
    def kept(self) -> list:
        """The findings still standing: everything except the rejected (and the ungrounded, which are not in `verified`)."""
        return [v.finding for v in self.verified if v.status != REJECTED]


@dataclass
class VerifiedReview:
    """The reasoning stage's review and what verification made of its findings."""

    reasoning: ReasoningReview
    verification: VerificationReview


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def flatten(finding: Finding) -> list:
    """One finding per claim: the merged finding's own claim, then each `Alternate` as a finding of its
    own (same file and class, its own lines, reviewers and quote), so two claims that share a place are
    checked separately."""
    units = [replace(finding, alternates=())]
    for alt in finding.alternates:
        units.append(
            Finding(
                file=finding.file,
                line_start=alt.line_start,
                line_end=alt.line_end,
                cls=finding.cls,
                claim=alt.claim,
                failing_input=alt.failing_input,
                reviewers=alt.reviewers,
                quote=alt.quote,
            )
        )
    return units


def _squash(text: str) -> str:
    return " ".join(text.split())


def locate_quote(quote: str, text: str, line_start: int, line_end: int) -> Optional[tuple]:
    """`(first line, last line)` of where `quote` occurs in `text`, or None.

    Both are compared with every run of whitespace squeezed to one space, so indentation, a different
    wrap and tabs versus spaces do not matter; a quote may span lines. When it occurs more than once,
    the occurrence nearest the claimed lines wins (the first of equals). A claim with no line number
    (0) is as near to every occurrence as to any other, so the first one is taken. A quote shorter than
    `MIN_QUOTE_CHARS` is too short to say where the code is and matches nothing."""
    needle = _squash(quote)
    if len(needle) < MIN_QUOTE_CHARS:
        return None
    pieces: list = []
    starts: list = []
    numbers: list = []
    offset = 0
    for number, line in enumerate(text.split("\n"), start=1):
        squeezed = _squash(line)
        if not squeezed:
            continue
        starts.append(offset)
        numbers.append(number)
        pieces.append(squeezed)
        offset += len(squeezed) + 1  # the joining space
    haystack = " ".join(pieces)

    def lines_of(position: int) -> tuple:
        return (numbers[bisect.bisect_right(starts, position) - 1], numbers[bisect.bisect_right(starts, position + len(needle) - 1) - 1])

    if line_start <= 0:
        position = haystack.find(needle)
        return None if position == -1 else lines_of(position)
    # The nearest occurrence is the last one that starts before the claimed lines or the first one that starts at
    # or after them, so two searches settle it however often the quote repeats (a count cap would, for a quote
    # on thousands of lines, stop short of the occurrence at the claim and move the finding to unrelated code).
    claimed_end = max(line_end, line_start)
    at_or_after = bisect.bisect_left(numbers, line_start)
    anchor = starts[at_or_after] if at_or_after < len(starts) else len(haystack)
    candidates = []
    before = haystack.rfind(needle, 0, anchor - 1 + len(needle))
    if before != -1:
        candidates.append(lines_of(before))
    after = haystack.find(needle, anchor)
    if after != -1:
        candidates.append(lines_of(after))
    if not candidates:
        return None
    # The first of equals: `min` keeps the earlier occurrence on a tie.
    return min(candidates, key=lambda lines: max(lines[0] - claimed_end, line_start - lines[1], 0))


def _strip_dot_slash(path: str) -> str:
    path = path.strip()
    while path.startswith("./"):
        path = path[2:]
    return path


def _clean_path(path: str) -> str:
    """The spelling with `\\` read as `/`: how a model that writes Windows-style separators names a file."""
    return _strip_dot_slash(path.strip().replace("\\", "/"))


def ground_findings(findings: Sequence[Finding], read_file: Callable[[str], Optional[str]]) -> tuple:
    """`(grounded, ungrounded)`. Each finding is first split into one per claim (`flatten`).

    A finding is grounded when its `quote` occurs in the cited file as `read_file(path)` returns it:
    it is kept with its line numbers corrected to the occurrence nearest the claim (a quote found
    elsewhere in the file is still grounded, only the lines move). It is ungrounded, with a reason,
    when it has no quote, the quote is too short to place, the file cannot be read (not in the change,
    not a file) or the quote is not in it."""
    grounded: list = []
    ungrounded: list = []
    for original in findings:
        for finding in flatten(original):
            if not finding.quote.strip():
                ungrounded.append(Ungrounded(finding, "it has no quote of the code it accuses"))
                continue
            if len(_squash(finding.quote)) < MIN_QUOTE_CHARS:
                ungrounded.append(Ungrounded(finding, f"its quote {finding.quote!r} is too short to say where the code is"))
                continue
            # The name as the model wrote it first: `x\b.py` and `x/b.py` can be two files. Only when that is
            # not a changed file is `\` read as `/` (a Windows-style `src\a.py` for the changed `src/a.py`).
            name = _strip_dot_slash(finding.file)
            text = read_file(name) if name else None
            if text is None and name and _clean_path(name) != name:
                name = _clean_path(name)
                text = read_file(name)
            if text is None:
                ungrounded.append(Ungrounded(finding, f"{finding.file or '(no file)'!r} is not a file in this change"))
                continue
            span = locate_quote(finding.quote, text, finding.line_start, finding.line_end)
            if span is None:
                ungrounded.append(Ungrounded(finding, f"its quote is not in {finding.file}"))
                continue
            grounded.append(replace(finding, file=name, line_start=span[0], line_end=span[1]))
    return grounded, ungrounded


def git_file_reader(repo_root: "str | Path", head_ref: str, allowed_paths: Sequence[str]) -> Callable[[str], Optional[str]]:
    """A `read_file(path)` for `ground_findings` and the verifiers: the text of `path` as it is at
    `head_ref`, or None.

    Only the paths in `allowed_paths` (the files of the change) are read, so a finding cannot name its way
    to another file, and the text is the exact version that was reviewed. None also for a file that does
    not exist at `head_ref`, a timeout or no git. The whole file is read, however large, so a quote near
    the end of a big file is found. A `head_ref` that starts with `-` is refused rather than handed to
    git as an option."""
    if head_ref.startswith("-") or any(c.isspace() or c == "\0" for c in head_ref):
        raise ReviewError(f"head_ref {head_ref!r} is not a usable git ref")
    # Exact names only: `x\b.py` and `x/b.py` are two files on POSIX, so a name is never read as another
    # spelling here (`ground_findings` is what tries the model's Windows-style spelling as a second guess).
    allowed = {_strip_dot_slash(p) for p in allowed_paths}
    root = Path(repo_root)

    def read(path: str) -> Optional[str]:
        path = _strip_dot_slash(path)
        if path not in allowed:
            return None
        try:
            done = subprocess.run(
                ["git", "-C", str(root), "show", f"{head_ref}:{path}"],
                capture_output=True,
                timeout=GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            return None
        return done.stdout.decode("utf-8", errors="replace")

    return read


def caching_reader(read_file: Callable[[str], Optional[str]]) -> Callable[[str], Optional[str]]:
    """`read_file` that asks for each path once. It is filled by the grounding step, before any
    verifier thread runs, so the verifiers only read it."""
    cache: dict = {}

    def read(path: str) -> Optional[str]:
        if path not in cache:
            cache[path] = read_file(path)
        return cache[path]

    return read


# ---------------------------------------------------------------------------
# The verifier prompt
# ---------------------------------------------------------------------------

VERIFY_INSTRUCTIONS = """\
You are checking ONE reported defect against the code it cites. You are not reviewing the change: judge only this claim.

Assume the claim is wrong until the code proves it. Work like a skeptical reviewer:
1. Say what would have to be true for the claim to hold: a value, a branch that is taken, a caller, a check that is missing.
2. Check each of those things in the code shown. Code you cannot see (a guard, a lock, a validator elsewhere) may exist: if the claim depends on something being absent and the code shown does not settle it, you cannot confirm the claim.
3. Take the reporter's failing input, or choose a better one, and trace the code line by line with it. Say what the code actually returns, raises or does.
Confirm ONLY when that trace ends in a wrong result: give the concrete input and the wrong result it produces. If the code handles the input correctly, if the claim is about code that is not there, or if you cannot give a failing input, reject the claim.

The request shows the claim, then the file's code with line numbers. A ">" in place of the blank marks the lines the claim is about."""

VERIFY_OUTPUT_FORMAT = """\
Reply with ONLY a JSON object, no other text:
{"verdict": "confirmed" or "rejected",
 "failing_input": "<if confirmed: the concrete input or state that makes the code fail; if rejected: empty>",
 "wrong_result": "<if confirmed: what the code returns or does with it, and why that is wrong; if rejected: empty>",
 "reasoning": "<one or two sentences: the check that decided it>"}
A "confirmed" without both a failing_input and a wrong_result is not accepted."""


def build_verifier_prompt(reviewer: Reviewer) -> str:
    """The system prompt for one verifier: its own prompt (lens), then how to verify, then the reply format.

    A verifier whose only type is `verifier` and that has a `promptFile` gets the file and nothing else (`instructions_from_file`):
    the file carries how to verify and how to reply, the way a reasoning reviewer's file carries its reply instructions, so the
    prompt can be changed per model without touching the server. (The reply is still read by `parse_verdict`, so the file has to
    describe that JSON verdict.) Any other verifier, including one that shares its file with a reasoning role, is as it always was."""
    own = load_prompt(reviewer)
    if instructions_from_file(reviewer, "verifier"):
        return own
    return "\n\n".join([own, VERIFY_INSTRUCTIONS, VERIFY_OUTPUT_FORMAT])


def code_window(text: str, line_start: int, line_end: int, budget: int) -> str:
    """The code a finding cites, numbered, with `>` marking the cited lines, at most about `budget`
    characters: the cited lines with `CODE_RADIUS_LINES` of context either side, the context shrinking
    first when that does not fit, and the cited lines themselves cut (with a marker) as a last resort."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    first = min(max(line_start, 1), max(len(lines), 1))
    last = min(max(line_end, first), max(len(lines), 1))
    rendered = ""
    for radius in range(CODE_RADIUS_LINES, -1, -1):
        low, high = max(1, first - radius), min(len(lines), last + radius)
        rendered = "\n".join(
            f"{number:>5} {'>' if first <= number <= last else ' '} {lines[number - 1]}" for number in range(low, high + 1)
        )
        if len(rendered) <= budget:
            return rendered
    return rendered[: max(budget, 1)] + "\n[cut: the cited code is longer than this verifier's context]"


def build_verifier_message(finding: Finding, code: str) -> str:
    where = f"{finding.file}:{finding.line_start}" + (f"-{finding.line_end}" if finding.line_end != finding.line_start else "")
    return "\n".join(
        [
            f"File: {where}",
            f"Kind of defect: {finding.cls}",
            f"Claim: {finding.claim}",
            f"Failing input the reporter gave: {finding.failing_input or '(none)'}",
            f"The code the claim accuses: {finding.quote}",
            "",
            "Code:",
            code,
        ]
    )


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_verdict(text: str, reviewer: str) -> tuple:
    """`(decision, problem)` from one verifier reply: exactly one of them is None. Never raises.

    Fences, prose and a leading `<think>` envelope around the JSON are tolerated. A `confirmed` with no
    `failing_input` or no `wrong_result` does not meet the standard (a confirmation needs a concrete failing
    input and the wrong result), so it is a problem, not a confirmation."""
    try:
        data = _extract_json(text)
    except ValueError:
        return None, f"reviewer {reviewer!r}: the reply was not JSON"
    if not isinstance(data, dict):
        return None, f"reviewer {reviewer!r}: the reply is not a JSON object"
    word = _text(data.get("verdict")).lower()
    verdict = {"confirmed": CONFIRMED, "confirm": CONFIRMED, "rejected": REJECTED, "reject": REJECTED}.get(word)
    if verdict is None:
        # A fixed sentence, without the value the model wrote: problems are recorded (the session log and the
        # request row's summary), and those never hold the model's reply.
        return None, f"reviewer {reviewer!r}: the verdict is neither 'confirmed' nor 'rejected'"
    failing, wrong = _text(data.get("failing_input")), _text(data.get("wrong_result"))
    if verdict == CONFIRMED and not (failing and wrong):
        return None, f"reviewer {reviewer!r}: it confirmed without a failing input and the wrong result it produces, so the answer is not counted"
    return Decision(reviewer, verdict, failing, wrong, _text(data.get("reasoning"))), None


def combine(decisions: Sequence[Decision]) -> str:
    """One outcome from the verifiers that answered: all confirm -> confirmed, all reject -> rejected,
    any disagreement -> disputed, no answer at all -> unchecked."""
    verdicts = {d.verdict for d in decisions}
    if not verdicts:
        return UNCHECKED
    if verdicts == {CONFIRMED}:
        return CONFIRMED
    if verdicts == {REJECTED}:
        return REJECTED
    return DISPUTED


# ---------------------------------------------------------------------------
# Running the verifiers
# ---------------------------------------------------------------------------

CompleteFn = Callable[[object, str, str, str], Optional[str]]


@dataclass
class VerifierResult:
    """What one verifier produced: its decision for each finding it answered, by position in the list it was given."""

    reviewer: str
    decisions: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)
    # Not problems: lookups cut to fit this verifier's budget, or `tools` it could not use.
    warnings: list = field(default_factory=list)
    # Positions of the findings it kept asking about after the round limit, and so gave no answer on.
    limit_hit: set = field(default_factory=set)


def _failure_text() -> str:
    kind, detail = openai_lib.take_failure()
    detail = display_safe(detail)  # the server's text, as sent: redacted here, where it is shown
    suffix = f": {detail}" if detail else ""
    return f"LM Studio request failed ({kind or 'unknown'}){suffix}"


def _verify_one(
    reviewer: Reviewer,
    findings: Sequence[Finding],
    read_file: Callable[[str], Optional[str]],
    client: object,
    complete: CompleteFn,
    recorder: Any = NULL_RECORDER,
    executor: object = None,
) -> VerifierResult:
    """One verifier over every finding. Raises when it answered nothing at all (every request failed);
    a single failed request or unusable reply is a problem and that finding simply gets no vote from it.

    A verifier with `tools` (and an `executor`) may answer a finding with `need:` lines, or with the verdict `cannot_tell`
    (which may name lookups), instead of a verdict: the finding is then
    asked again with what it asked for in front of it (`reviewer_need_lib`), up to the round limit; a verifier that still only
    asks then gives no vote on that finding (`VerifierResult.limit_hit`). One without `tools` makes one request per finding,
    with exactly the prompt and message it always had."""
    loop = loop_enabled(reviewer, executor)
    system = with_need_instructions(reviewer, build_verifier_prompt(reviewer), executor, "verifier")
    reference = loop_reserve(reviewer, system, executor)
    budget = chunk_chars_for(reviewer, system, reserve=reference)
    result = VerifierResult(reviewer=reviewer.name)
    unavailable = loop_unavailable(reviewer, executor)
    if unavailable:
        result.warnings.append(unavailable)
    failed = 0
    last_failure = ""
    prompt_hash = prompt_hash_of(reviewer, loop) if recorder is not NULL_RECORDER else None
    for index, finding in enumerate(findings):
        text = read_file(finding.file) or ""
        message_head = build_verifier_message(finding, "")
        code = code_window(text, finding.line_start, finding.line_end, max(budget - len(message_head), 1))
        base = build_verifier_message(finding, code)
        session = NeedSession(reviewer, executor, reference, reply_format=need_format_for(reviewer, "verifier"), cannot_tell=True) if loop else None
        asking = True
        while asking:
            asking = False
            message = session.message(base) if session is not None else base
            with recorder.request(
                reviewer=reviewer.name, role="verifier", model=reviewer.model, subject=("finding", index + 1),
                prompt_chars=len(system) + len(message), reply_format="json", translator=None, prompt_hash=prompt_hash,
            ) as req:
                reply = complete(client, reviewer.model, system, message)
                finish_request(req, reply)
                if reply is None:
                    failed += 1
                    last_failure = _failure_text()
                    result.problems.append(f"reviewer {reviewer.name!r}: finding {index + 1}/{len(findings)}: {last_failure}")
                    req.problem(last_failure)
                    continue
                if session is not None:
                    outcome = session.read(reply, req)
                    if outcome == ASKED:
                        asking = True
                        continue
                    if outcome == LIMIT:
                        # Not a problem: the verifier read its instructions and kept asking. It gives no vote, and a finding
                        # nobody else answered says why it is unchecked (`verify_step`).
                        result.limit_hit.add(index)
                        continue
                decision, problem = parse_verdict(reply, reviewer.name)
                if decision is not None:
                    detail = list(describe_finding(finding))
                    if decision.failing_input:
                        detail.append(f"verifier's failing input: {decision.failing_input}")
                    if decision.wrong_result:
                        detail.append(f"verifier's wrong result: {decision.wrong_result}")
                    req.note(f"{decision.verdict}: {decision.reasoning}" if decision.reasoning else decision.verdict, detail=tuple(detail))
                    result.decisions[index] = decision
                elif problem is not None:
                    result.problems.append(f"finding {index + 1}/{len(findings)}: {problem}")
                    req.problem(problem)
        if session is not None:
            result.warnings.extend(w for w in session.warnings if w not in result.warnings)
    if findings and failed == len(findings):
        raise ReviewError(f"every request failed; the last one: {last_failure}")
    return result


def ground_step(findings: Sequence[Finding], read: Callable[[str], Optional[str]], recorder: Any = NULL_RECORDER) -> tuple:
    """The grounding half of `verify_findings`, as its own step: `(grounded, ungrounded)` for `findings`, with the
    "grounding" notes the run's record keeps. It needs no model. `read` should be a `caching_reader`, one per run, so the
    verification step that follows reads each file once more at most from the same cache."""
    grounded, ungrounded = ground_findings(findings, read)
    if findings:
        recorder.note("grounding", f"{len(grounded)} of {len(findings)} finding(s) kept; {len(ungrounded)} dropped as ungrounded")
        for dropped in ungrounded:
            recorder.note("grounding", f"dropped: {dropped.reason}", detail=tuple(describe_finding(dropped.finding)))
    return grounded, ungrounded


def verify_step(
    config: ReviewersConfig,
    loader: ModelLoader,
    client: object,
    grounded: Sequence[Finding],
    ungrounded: list,
    read: Callable[[str], Optional[str]],
    max_reviewers: int,
    *,
    complete: Optional[CompleteFn] = None,
    recorder: Any = NULL_RECORDER,
    executor: object = None,
) -> VerificationReview:
    """The verification half of `verify_findings`, as its own step: has every active verifier confirm or reject each
    of `grounded` (the findings `ground_step` kept; `ungrounded` is what it dropped, carried into the result).

    With no active verifier the grounded findings come back `unchecked` (no model checked them) and no model is loaded.
    At most `MAX_VERIFIED_FINDINGS` are sent to the verifiers; the rest come back unchecked with a warning. A verifier
    that fails does not stop the others: it is reported in `errors`, and a finding it did not answer is decided by the
    rest. A model that is not loaded raises the scheduler's `ModelNotLoadedError` before anything runs.

    `executor` is what a verifier with `tools` asks for lookups through (`reviewer_need_lib`); a finding that every verifier left
    unanswered because they kept asking after the round limit is `unchecked` with a note that says so."""
    review = VerificationReview(ungrounded=ungrounded)
    if not config.active_reviewers("verifier"):
        review.verified = [Verified(f, UNCHECKED, note="no verifier is active") for f in grounded]
        return review
    checked = list(grounded[:MAX_VERIFIED_FINDINGS])
    skipped = list(grounded[MAX_VERIFIED_FINDINGS:])
    if skipped:
        review.warnings.append(
            f"{len(skipped)} finding(s) beyond the first {MAX_VERIFIED_FINDINGS} were not sent to the verifiers "
            "(each costs one request per verifier) and are unchecked"
        )
    if not checked:
        review.verified = [Verified(f, UNCHECKED, note="not sent to the verifiers (over the limit)") for f in skipped]
        return review
    complete_fn = complete or openai_lib.complete

    def run_verifier(reviewer: Reviewer, role: str) -> VerifierResult:
        return _verify_one(reviewer, checked, read, client, complete_fn, recorder, executor)

    report = run_stages(config, loader, run_verifier, max_reviewers, roles=("verifier",), recorder=recorder)
    review.warnings.extend(report.warnings)
    answers: dict = {index: [] for index in range(len(checked))}
    gave_up: set = set()  # positions of findings some verifier kept asking about after the round limit
    for outcome in report.outcomes:
        name = outcome.reviewer.name
        if outcome.error is not None:
            review.errors.append(f"reviewer {name!r}: {outcome.error}")
            continue
        result = outcome.value
        assert isinstance(result, VerifierResult)
        review.verifiers_run.append(name)
        review.errors.extend(result.problems)
        review.warnings.extend(w for w in result.warnings if w not in review.warnings)
        gave_up |= result.limit_hit
        for index, decision in result.decisions.items():
            answers[index].append(decision)
    for index, finding in enumerate(checked):
        decisions = tuple(answers[index])
        status = combine(decisions)
        note = ""
        if status == UNCHECKED:
            note = (
                "a verifier kept asking for more context after the round limit and gave no answer"
                if index in gave_up
                else "no verifier gave a usable answer"
            )
        review.verified.append(Verified(finding, status, decisions, note))
    review.verified.extend(Verified(f, UNCHECKED, note="not sent to the verifiers (over the limit)") for f in skipped)
    return review


def verify_findings(
    config: ReviewersConfig,
    loader: ModelLoader,
    client: object,
    findings: Sequence[Finding],
    read_file: Callable[[str], Optional[str]],
    max_reviewers: int,
    *,
    complete: Optional[CompleteFn] = None,
    recorder: Any = NULL_RECORDER,
    executor: object = None,
) -> VerificationReview:
    """Grounds `findings`, then has every active verifier confirm or reject each one that survives: `ground_step`
    then `verify_step` over one cached reader. The orchestrator runs the two as separate steps; this is the
    one-call form the earlier callers (and `run_verified_review`) keep.

    The grounding step runs whether or not a verifier is active: it needs no model, and a finding about
    code that is not there should not be passed on. See `verify_step` for what happens after it."""
    read = caching_reader(read_file)
    grounded, ungrounded = ground_step(findings, read, recorder)
    return verify_step(
        config, loader, client, grounded, ungrounded, read, max_reviewers, complete=complete, recorder=recorder, executor=executor
    )


def standing(verification: VerificationReview) -> list:
    """What the classifier stage looks at: every verified finding the verifiers did not reject (a `Verified` each)."""
    return [v for v in verification.verified if v.status != REJECTED]


def preflight_models(config: ReviewersConfig, loader: ModelLoader, roles: Sequence[str]) -> None:
    """Checks, before the reasoning stage starts, that the models `roles` need and that must already be loaded are loaded,
    so a model that is not is reported at once and not after minutes of reasoning. Does nothing when no reasoning
    reviewer is active (the reasoning stage then refuses on its own, and a later stage has nothing to run on)."""
    if config.active_reviewers("reasoning"):
        check_resident_models(plan_stages(config, tuple(roles)), loader)


def run_verified_review(
    config: ReviewersConfig,
    loader: ModelLoader,
    client: object,
    diff_text: str,
    repo_root: "str | Path",
    head_ref: str,
    max_reviewers: int,
    *,
    chunk_chars: Optional[int] = None,
    lessons: Sequence[str] = (),
    complete: Optional[CompleteFn] = None,
    read_file: Optional[Callable[[str], Optional[str]]] = None,
    recorder: Any = NULL_RECORDER,
    executor: object = None,
) -> VerifiedReview:
    """The reasoning stage over `diff_text`, then `verify_findings` over what it found.

    `executor` is the `reviewer_tools_lib.ToolExecutor` that reviewers and verifiers with `tools` ask for lookups through (pass
    `ToolExecutor(repo_root, head_ref)`; without one they run without lookups and say so).
    `read_file` defaults to `git_file_reader` at `head_ref` for the files in the diff. The resident
    verifier models are checked BEFORE the reasoning stage starts, so a verifier model that is not loaded
    is reported at once and not after minutes of reasoning. Raises what `run_reasoning_review` raises."""
    preflight_models(config, loader, ("verifier",))
    reasoning = run_reasoning_review(
        config, loader, client, diff_text, repo_root, max_reviewers, chunk_chars=chunk_chars, lessons=lessons, complete=complete,
        recorder=recorder, executor=executor,
    )
    reader = read_file or git_file_reader(repo_root, head_ref, [section.path for section in split_diff(diff_text)])
    findings = [finding for group in reasoning.groups for finding in group.findings]
    verification = verify_findings(
        config, loader, client, findings, reader, max_reviewers, complete=complete, recorder=recorder, executor=executor
    )
    return VerifiedReview(reasoning=reasoning, verification=verification)


# ---------------------------------------------------------------------------
# Showing the result
# ---------------------------------------------------------------------------


def _where(finding: Finding) -> str:
    where = finding.file or "(unknown file)"
    if finding.line_start:
        where += f":{finding.line_start}" + (f"-{finding.line_end}" if finding.line_end != finding.line_start else "")
    return where


def _ordered(items: Sequence[Verified]) -> list:
    return sorted(items, key=lambda v: (v.finding.cls, v.finding.file, v.finding.line_start, v.finding.claim))


def _finding_line(f: Finding) -> str:
    return f"- {_where(f)} [{f.cls}] [{', '.join(f.reviewers)}] {f.claim}"


def evidence_lines(v: Verified) -> list:
    """What the verifiers found for one finding, one indented line each: why it was not checked, the failing input its reviewer
    reported (when no verifier confirmed it), and each verifier's reproducer (confirmed) or verdict and reasoning (otherwise)."""
    lines = []
    if v.status == UNCHECKED and v.note:
        lines.append(f"    not checked: {v.note}")
    elif v.finding.failing_input and v.status != CONFIRMED:
        lines.append(f"    reported failing input: {v.finding.failing_input}")
    for d in v.decisions:
        if v.status == CONFIRMED:
            lines.append(f"    {d.reviewer}: failing input: {d.failing_input} -> {d.wrong_result}")
            continue
        detail = d.reasoning or (f"failing input {d.failing_input} -> {d.wrong_result}" if d.failing_input else "(no reason given)")
        lines.append(f"    {d.reviewer} {d.verdict}: {detail}")
    return lines


def format_verified_review(result: VerifiedReview) -> str:
    """The verified review as text for the calling model: what stands, what is in dispute, what was not
    checked, and (one line each) what was dropped and why."""
    review = result.verification
    reasoning = result.reasoning
    produced = len(review.verified) + len(review.ungrounded)
    lines = [
        f"Verified review: {len(review.confirmed)} confirmed, {len(review.disputed)} disputed, {len(review.unchecked)} unchecked; "
        f"dropped {len(review.rejected)} rejected and {len(review.ungrounded)} ungrounded "
        f"(of {produced} claim(s) from {len(reasoning.reviewers_run)} reasoning reviewer(s): {', '.join(reasoning.reviewers_run) or 'none'}).",
    ]
    if review.verifiers_run:
        lines.append(
            f"Verifiers: {', '.join(review.verifiers_run)}. A confirmed finding was reproduced by every verifier that answered, "
            "with a concrete failing input; a disputed one split them."
        )
    elif review.unchecked:
        lines.append(
            "No verifier ran, so these findings are GROUNDED (the code each one quotes is in the file) but NOT VERIFIED by a model."
        )
    else:
        lines.append("No verifier ran.")
    sections = (
        (f"Confirmed ({len(review.confirmed)})", review.confirmed),
        (f"Disputed ({len(review.disputed)})", review.disputed),
        (f"Unchecked ({len(review.unchecked)})", review.unchecked),
    )
    for title, items in sections:
        if not items:
            continue
        lines.append("")
        lines.append(f"## {title}")
        for v in _ordered(items):
            lines.append(_finding_line(v.finding))
            lines.extend(evidence_lines(v))
    dropped = _ordered(review.rejected)
    if dropped:
        lines.append("")
        lines.append(f"## Dropped: rejected by the verifiers ({len(dropped)})")
        for v in dropped:
            lines.append(_finding_line(v.finding))
            for d in v.decisions:
                lines.append(f"    {d.reviewer} rejected: {d.reasoning or '(no reason given)'}")
    if review.ungrounded:
        lines.append("")
        lines.append(f"## Dropped: ungrounded ({len(review.ungrounded)})")
        for u in sorted(review.ungrounded, key=lambda x: (x.finding.cls, x.finding.file, x.finding.line_start, x.finding.claim)):
            lines.append(_finding_line(u.finding))
            lines.append(f"    ungrounded: {u.reason}")
    problems = [*reasoning.errors, *review.errors]
    if problems:
        lines.append("")
        lines.append(f"Problems ({len(problems)}):")
        lines.extend(f"- {e}" for e in problems)
    warnings = [*reasoning.warnings, *review.warnings]
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {w}" for w in warnings)
    return "\n".join(lines)


__all__ = [
    "evidence_lines",
    "CONFIRMED",
    "REJECTED",
    "DISPUTED",
    "UNCHECKED",
    "Ungrounded",
    "Decision",
    "Verified",
    "VerificationReview",
    "VerifiedReview",
    "flatten",
    "locate_quote",
    "ground_findings",
    "git_file_reader",
    "build_verifier_prompt",
    "build_verifier_message",
    "code_window",
    "parse_verdict",
    "combine",
    "caching_reader",
    "ground_step",
    "verify_step",
    "verify_findings",
    "standing",
    "preflight_models",
    "run_verified_review",
    "format_verified_review",
]
