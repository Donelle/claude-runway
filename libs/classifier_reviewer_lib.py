"""
The classifier stage of the `codebase-reviewer` server: check each finding that stands after verification against the
user's rubric, and give it the level of the entry it fits.

What this module does, in the order a classified review uses it:

  1. `classify_findings` shows each finding (where it is, its class, the claim, the failing input the
     reporter gave, the code it quotes) to every active classifier through
     `reviewer_scheduler_lib.run_stages` (so grouping, parallelism and model loading are the scheduler's).
     For every rubric entry a classifier is asked one yes/no question: does this finding fall under this entry's
     definition? It is shown the entry's name, level and definition, never its `score`, reads no code and runs
     nothing. The answer is not the word the model writes but its probability of "yes", read from the token
     probabilities at temperature 0 (`openai_lib.yes_probability`), so asking again gives the same number.
     A model whose runtime returns no token probabilities cannot be a classifier: that is an error naming it.
  2. `score_entries` turns every classifier's probability for a finding into one number per entry: the plain average
     over the classifiers. An entry matches when that average is at or above its `score`, a probability from 0 to 1.
     The average is rounded to six places before the comparison, so float noise cannot tip a finding either way.
  3. `classify_answers` decides the finding: no entry reached its `score` -> *unclassified* (dropped from
     the result, recorded); one or more did -> *classified*, taking the entry with the highest `level`
     (`LEVELS`, lowest to highest), equal levels going to the higher average and then to the entry that
     comes first in the rubric; the others that reached their `score` are recorded as an overlap; no
     answer from any classifier (or none active) -> *unchecked*, kept and said so. Nothing is dropped
     silently.
  4. `run_classified_review` runs the verified review first (`run_verified_review`) and classifies what is
     left standing: everything the verifiers did not reject.

Every classification is handed to the run's recorder (`recorder.add_classification`), which writes it to the
reviewer's own database, each classifier's probability for every entry included, so any average can be recomputed.

Standard library only, apart from the sibling libs. The model client and the probability call are passed in,
so tests run with a stub and no LM Studio.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import openai_lib
from credential_redaction_lib import display_safe
from codebase_reviewer_config_lib import LEVELS, Reviewer, ReviewersConfig, RubricEntry, load_definition, load_prompt
from lmstudio_model_loader_lib import ModelLoader
from reasoning_reviewer_lib import (
    Finding,
    ReplyCount,
    ReviewError,
    chunk_chars_for,
    describe_finding,
    finish_request,
    incomplete_review_line,
    prompt_hash_of,
)
from reviewer_db_lib import NULL_RECORDER
from reviewer_scheduler_lib import run_stages
from verifier_reviewer_lib import UNCHECKED, VerifiedReview, evidence_lines, preflight_models, run_verified_review, standing

# Findings beyond this many are not classified (each costs one request per rubric entry per classifier) and come back unchecked.
MAX_CLASSIFIED_FINDINGS = 100
# The average probability is rounded to this many places before it is compared with an entry's `score`.
SCORE_PRECISION = 6

CLASSIFIED = "classified"
UNCLASSIFIED = "unclassified"
# `UNCHECKED` is the verifier stage's word for the same thing, imported above: no classifier gave an answer.


@dataclass(frozen=True)
class Rating:
    """A classifier's probability, from 0 to 1, that a finding falls under one rubric entry's definition."""

    entry: str
    probability: float


@dataclass(frozen=True)
class Classification:
    """One classifier's answer for one finding: its probability for every rubric entry, in rubric order."""

    reviewer: str
    ratings: tuple = ()


@dataclass(frozen=True)
class EntryScore:
    """What every classifier's answer for a finding came to for one rubric entry: the average probability, and whether it
    reaches the entry's `score`."""

    entry: str
    mean: float
    reached: bool


@dataclass(frozen=True)
class Outcome:
    """What a finding's answers decide: its `status`, and for a classified one the `entry` and `level` it took and the
    `confidence` (that entry's average probability). `overlap` is true when more than one entry reached its `score`;
    `overlapping` lists the ones that lost to `entry`, in rubric order. `scores` is the average for every entry."""

    status: str
    entry: Optional[str] = None
    level: Optional[str] = None
    confidence: Optional[float] = None
    overlap: bool = False
    overlapping: tuple = ()
    scores: tuple = ()


@dataclass(frozen=True)
class Classified:
    """One finding and what the classifiers made of it. `status` is `classified`, `unclassified` or `unchecked`;
    `verification` is the status the finding arrived with from the verifier stage ("" when it was never verified); `note`
    says why for an unchecked one. `classifications` is every classifier's answer, `scores` the average per entry."""

    finding: Finding
    status: str
    verification: str = ""
    entry: Optional[str] = None
    level: Optional[str] = None
    confidence: Optional[float] = None
    overlap: bool = False
    overlapping: tuple = ()
    scores: tuple = ()
    classifications: tuple = ()
    note: str = ""


@dataclass
class ClassificationReview:
    classified: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    classifiers_run: list = field(default_factory=list)
    # The scheduler's outcome for every classifier that was asked, in the order they ran (`ReviewerOutcome`: the classifier, its
    # `ClassifierResult` or the error that stopped it). `classified` holds each finding's combined outcome; this keeps each classifier's
    # own probabilities, which the structured results file saves so that a process can average several classifiers' results itself.
    outcomes: list = field(default_factory=list)

    def with_status(self, status: str) -> list:
        return [c for c in self.classified if c.status == status]

    @property
    def matched(self) -> list:
        return self.with_status(CLASSIFIED)

    @property
    def unclassified(self) -> list:
        return self.with_status(UNCLASSIFIED)

    @property
    def unchecked(self) -> list:
        return self.with_status(UNCHECKED)

    @property
    def overlaps(self) -> list:
        return [c for c in self.classified if c.overlap]

    def by_level(self, level: str) -> list:
        return [c for c in self.matched if c.level == level]


@dataclass
class ClassifiedReview:
    """The verified review and what classification made of the findings that were left standing."""

    verified: VerifiedReview
    classification: ClassificationReview


# ---------------------------------------------------------------------------
# The classifier prompt
# ---------------------------------------------------------------------------

CLASSIFY_INSTRUCTIONS = """\
You are checking ONE reported defect against ONE category. You are not reviewing the change and not judging whether the claim is true: it has already been checked.

Decide whether the claim, as stated, is the kind of defect the category's definition describes. Do not stretch a definition to make the claim fit.

The request shows the claim, where it is, the failing input the reporter gave and the code it accuses, then the category and its definition.

Reply with ONLY Yes or No."""


def build_entry_text(entry: RubricEntry) -> str:
    """The rubric entry as the classifier reads it: its name, level and full definition. The entry's `score` is not shown:
    it is the server's cut-off, and a model told it can learn to answer to it."""
    return f"Category: {entry.name} (level: {entry.level})\nDefinition: {load_definition(entry)}"


def build_classifier_prompt(reviewer: Reviewer) -> str:
    """The system prompt for one classifier: its own prompt (lens), then the question. The same for every entry; the entry
    comes in the request, after the finding (`build_classifier_message`)."""
    return "\n\n".join([load_prompt(reviewer), CLASSIFY_INSTRUCTIONS])


def build_finding_text(finding: Finding) -> str:
    where = f"{finding.file}:{finding.line_start}" + (f"-{finding.line_end}" if finding.line_end != finding.line_start else "")
    lines = [
        f"File: {where}",
        f"The reporter's own label for it (not a rubric entry): {finding.cls}",
        f"Claim: {finding.claim}",
        f"Failing input the reporter gave: {finding.failing_input or '(none)'}",
    ]
    if finding.quote:
        lines.append(f"The code the claim accuses: {finding.quote}")
    return "\n".join(lines)


def build_classifier_message(finding: Finding, entry: RubricEntry) -> str:
    """The request for one (finding, rubric entry) pair: the finding, then the entry it is checked against."""
    return f"{build_finding_text(finding)}\n\n{build_entry_text(entry)}"


# ---------------------------------------------------------------------------
# Applying the answers
# ---------------------------------------------------------------------------


def reaches(mean: float, score: float) -> bool:
    """Whether an average probability reaches an entry's `score`: rounded to `SCORE_PRECISION` places first, so an average
    that is the `score` up to float noise reaches it. The one rule, used by the classifier and by whatever counts its results."""
    return round(mean, SCORE_PRECISION) >= score


def score_entries(answers: Sequence[Classification], rubric: Sequence[RubricEntry]) -> tuple:
    """One `EntryScore` per rubric entry (in rubric order): the average of the probabilities every classifier gave it, and
    whether that average reaches the entry's `score`. Needs at least one answer. The average is rounded to `SCORE_PRECISION`
    places before the comparison, so an average that is exactly the `score` reaches it."""
    count = len(answers)
    scores = []
    for entry in rubric:
        total = sum(r.probability for a in answers for r in a.ratings if r.entry == entry.name)
        mean = total / count
        scores.append(EntryScore(entry.name, mean, reaches(mean, entry.score)))
    return tuple(scores)


def classify_answers(answers: Sequence[Classification], rubric: Sequence[RubricEntry]) -> Outcome:
    """What the answers for one finding decide. No answers -> unchecked. Entries whose average reaches their `score` match:
    none -> unclassified; one or more -> classified, taking the highest `level` (in `LEVELS` order), equal levels going to
    the higher average and then to the entry first in the rubric. The other matching entries are the overlap."""
    if not answers:
        return Outcome(UNCHECKED)
    scores = score_entries(answers, rubric)
    order = {entry.name: index for index, entry in enumerate(rubric)}
    by_name = {entry.name: entry for entry in rubric}
    reached = [s for s in scores if s.reached]
    if not reached:
        return Outcome(UNCLASSIFIED, scores=scores)
    chosen = max(reached, key=lambda s: (LEVELS.index(by_name[s.entry].level), s.mean, -order[s.entry]))
    others = tuple(s.entry for s in reached if s is not chosen)
    return Outcome(CLASSIFIED, chosen.entry, by_name[chosen.entry].level, chosen.mean, bool(others), others, scores)


# ---------------------------------------------------------------------------
# Running the classifiers
# ---------------------------------------------------------------------------

ProbabilityFn = Callable[..., Optional[float]]
CompleteFn = Callable[..., Optional[str]]


@dataclass
class ClassifierResult:
    """What one classifier produced: its answer for each finding it could answer, by position in the list it was given."""

    reviewer: str
    answers: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)
    counts: ReplyCount = field(default_factory=ReplyCount)


# The failure kinds (`openai_lib.take_failure`) of a request that DID get a reply which the reader then could not use. Every other
# failure (a timeout, a refusal, a connection that never opened) is a request with no reply at all.
_UNREADABLE_REPLY_KINDS = ("no_probabilities", "no_answer")


def _failure_text(model: str) -> tuple:
    """(what the caller is told, what the records keep, what the session log must not show) for the request that just failed.

    What a model began its reply with is model-written text, which the database and the session log never hold, so it is in
    the first and the third only: the records get the same sentence without it."""
    kind, detail = openai_lib.take_failure()
    detail = display_safe(detail)  # the server's text, as sent: redacted here, where it is shown
    if kind == "no_probabilities":
        text = (
            f"model {model!r} returned no token probabilities, which a classifier needs: "
            "use a model and runtime that return them (a reasoning model, whose reply starts with its thinking, does not)"
        )
        return text, text, ""
    if kind == "no_answer":
        began = f" (its likeliest first tokens: {detail})" if detail else ""
        said = f"model {model!r} began its reply with something other than Yes or No"
        advice = ": its chat template or tokenizer does not start a reply with the answer, so check the template in LM Studio or use another model"
        return said + began + advice, said + advice, detail or ""
    suffix = f": {detail}" if detail else ""
    text = f"LM Studio request failed ({kind or 'unknown'}){suffix}"
    return text, text, ""


def _probability(value: float) -> str:
    """A probability as text, two places: 0.93."""
    return f"{value:.2f}"


def _describe_scores(scores: Sequence[EntryScore]) -> str:
    return ", ".join(f"{s.entry} {_probability(s.mean)}" for s in scores)


def _classify_one(
    reviewer: Reviewer,
    findings: Sequence[Finding],
    rubric: Sequence[RubricEntry],
    client: object,
    probability: ProbabilityFn,
    recorder: Any = NULL_RECORDER,
) -> ClassifierResult:
    """One classifier over every finding: one request per rubric entry for each. Raises when it answered nothing at all
    (every request failed); a failed request or an unreadable reply is a problem, and the classifier then gives no answer
    for that finding (an answer needs a probability for every entry, since a missing one would read as "does not fit")."""
    system = build_classifier_prompt(reviewer)
    chunk_chars_for(reviewer, system)  # refuses a context too small for its own prompt, as the other stages do
    result = ClassifierResult(reviewer=reviewer.name)
    failed = 0
    requests = 0
    last_failure = ""
    prompt_hash = prompt_hash_of(reviewer) if recorder is not NULL_RECORDER else None
    for index, finding in enumerate(findings):
        ratings = []
        missing = []
        for entry in rubric:
            message = build_classifier_message(finding, entry)
            requests += 1
            result.counts.requests += 1
            with recorder.request(
                reviewer=reviewer.name, role="classifier", model=reviewer.model, subject=("finding", index + 1),
                prompt_chars=len(system) + len(message), reply_format="probability", translator=None, prompt_hash=prompt_hash,
            ) as req:
                value = probability(client, reviewer.model, system, message)
                finish_request(req, None if value is None else _probability(value))
                if value is None:
                    failed += 1
                    # Read before `_failure_text` takes (and so clears) the failure's kind.
                    if openai_lib.peek_failure()[0] in _UNREADABLE_REPLY_KINDS:
                        result.counts.replies += 1
                        result.counts.unreadable += 1
                    last_failure, recorded, withheld = _failure_text(reviewer.model)
                    recorder.withhold(withheld)
                    missing.append(entry.name)
                    req.problem(recorded)
                    continue
                result.counts.replies += 1
                req.note(f"{entry.name} {_probability(value)}", detail=tuple(describe_finding(finding)))
                ratings.append(Rating(entry.name, value))
        if missing:
            result.problems.append(
                f"reviewer {reviewer.name!r}: finding {index + 1}/{len(findings)}: no probability for {', '.join(missing)}, "
                f"so this classifier gives no answer for it ({last_failure})"
            )
        else:
            result.answers[index] = Classification(reviewer.name, tuple(ratings))
    if requests and failed == requests:
        raise ReviewError(f"every request failed; the last one: {last_failure}")
    return result


def _record(recorder: Any, number: int, item: Classified) -> None:
    finding = item.finding
    recorder.add_classification(
        finding_no=number, file=finding.file, line_start=finding.line_start, line_end=finding.line_end, finding_class=finding.cls,
        verification=item.verification, status=item.status, entry=item.entry, level=item.level,
        confidence=item.confidence,
        overlap=item.overlap, overlapping=list(item.overlapping),
        scores=[{"entry": s.entry, "mean": s.mean, "reached": s.reached} for s in item.scores],
        answers=[{"reviewer": c.reviewer, "probabilities": [{"entry": r.entry, "probability": r.probability} for r in c.ratings]}
                 for c in item.classifications],
    )
    if item.status == CLASSIFIED:
        text = f"classified as {item.entry} [{item.level}], average probability {_probability(item.confidence or 0)} over {len(item.classifications)} classifier(s)"
    elif item.status == UNCLASSIFIED:
        text = f"unclassified: no entry's average probability reached its score, so the finding is dropped ({_describe_scores(item.scores)})"
    else:
        text = f"unchecked: {item.note}"
    if item.overlap:
        text += f" (overlap, also fits {', '.join(item.overlapping)})"
    recorder.note("classifier", text, detail=tuple(describe_finding(item.finding)))


def classify_findings(
    config: ReviewersConfig,
    loader: ModelLoader,
    client: object,
    findings: Sequence[Finding],
    max_reviewers: int,
    *,
    statuses: Optional[Sequence[str]] = None,
    probability: Optional[ProbabilityFn] = None,
    probability_client: Optional[object] = None,
    recorder: Any = NULL_RECORDER,
) -> ClassificationReview:
    """Has every active classifier check each finding against the rubric, and decides each finding from the averages.

    `statuses` is what the verifier stage made of each finding (parallel to `findings`; "" when not given). With no
    active classifier the findings come back `unchecked` (no rubric level) and no model is loaded. At most
    `MAX_CLASSIFIED_FINDINGS` are sent; the rest come back unchecked with a warning. A classifier that fails does not
    stop the others: it is reported in `errors`, and a finding it did not answer is decided by the rest. A model that
    is not loaded raises the scheduler's `ModelNotLoadedError` before anything runs.

    `probability_client` is the client the probability requests go through (default: `client`). It has to be one built on
    LM Studio's OpenAI-compatible `/v1`: the native `/api/v0` that the other stages use for its timing statistics returns
    no token probabilities."""
    review = ClassificationReview()
    verification = list(statuses) if statuses is not None else [""] * len(findings)
    if len(verification) != len(findings):
        raise ValueError(f"statuses has {len(verification)} item(s) for {len(findings)} finding(s)")
    if not config.active_reviewers("classifier"):
        review.classified = [Classified(f, UNCHECKED, v, note="no classifier is active") for f, v in zip(findings, verification, strict=True)]
        _record_all(recorder, review)
        return review
    checked = list(findings[:MAX_CLASSIFIED_FINDINGS])
    skipped = list(zip(findings[MAX_CLASSIFIED_FINDINGS:], verification[MAX_CLASSIFIED_FINDINGS:], strict=True))
    if skipped:
        review.warnings.append(
            f"{len(skipped)} finding(s) beyond the first {MAX_CLASSIFIED_FINDINGS} were not sent to the classifiers "
            "(each costs one request per rubric entry per classifier) and are unchecked"
        )
    probability_fn = probability or openai_lib.yes_probability
    rubric = config.rubric
    if checked:

        def run_classifier(reviewer: Reviewer, role: str) -> ClassifierResult:
            return _classify_one(reviewer, checked, rubric, probability_client or client, probability_fn, recorder)

        report = run_stages(config, loader, run_classifier, max_reviewers, roles=("classifier",), recorder=recorder)
        review.outcomes = list(report.outcomes)
        review.warnings.extend(report.warnings)
        answers: dict = {index: [] for index in range(len(checked))}
        for outcome in report.outcomes:
            name = outcome.reviewer.name
            if outcome.error is not None:
                review.errors.append(f"reviewer {name!r}: {outcome.error}")
                continue
            result = outcome.value
            assert isinstance(result, ClassifierResult)
            review.classifiers_run.append(name)
            review.errors.extend(result.problems)
            for index, given in result.answers.items():
                answers[index].append(given)
        for index, finding in enumerate(checked):
            given = tuple(answers[index])
            decided = classify_answers(given, rubric)
            note = "no classifier gave a usable answer" if decided.status == UNCHECKED else ""
            review.classified.append(Classified(
                finding, decided.status, verification[index], decided.entry, decided.level, decided.confidence,
                overlap=decided.overlap, overlapping=decided.overlapping, scores=decided.scores, classifications=given, note=note,
            ))
    review.classified.extend(
        Classified(f, UNCHECKED, v, note="not sent to the classifiers (over the limit)") for f, v in skipped
    )
    _record_all(recorder, review)
    return review


def _record_all(recorder: Any, review: ClassificationReview) -> None:
    for number, item in enumerate(review.classified, start=1):
        _record(recorder, number, item)


def run_classified_review(
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
    probability: Optional[ProbabilityFn] = None,
    probability_client: Optional[object] = None,
    read_file: Optional[Callable[[str], Optional[str]]] = None,
    recorder: Any = NULL_RECORDER,
    executor: object = None,
) -> ClassifiedReview:
    """The verified review over `diff_text`, then `classify_findings` over every finding the verifiers did not reject.

    The resident classifier models are checked BEFORE the reasoning stage starts (the verifier's are checked inside
    `run_verified_review`, also first), so a classifier model that is not loaded is reported at once and not after
    minutes of reasoning. Raises what `run_verified_review` raises."""
    preflight_models(config, loader, ("classifier",))
    verified = run_verified_review(
        config, loader, client, diff_text, repo_root, head_ref, max_reviewers, chunk_chars=chunk_chars, lessons=lessons,
        complete=complete, read_file=read_file, recorder=recorder, executor=executor,
    )
    kept = standing(verified.verification)
    classification = classify_findings(
        config, loader, client, [v.finding for v in kept], max_reviewers, statuses=[v.status for v in kept],
        probability=probability, probability_client=probability_client, recorder=recorder,
    )
    return ClassifiedReview(verified=verified, classification=classification)


# ---------------------------------------------------------------------------
# Showing the result
# ---------------------------------------------------------------------------


def _where(finding: Finding) -> str:
    where = finding.file or "(unknown file)"
    if finding.line_start:
        where += f":{finding.line_start}" + (f"-{finding.line_end}" if finding.line_end != finding.line_start else "")
    return where


def _ordered(items: Sequence[Classified]) -> list:
    return sorted(items, key=lambda c: (c.finding.cls, c.finding.file, c.finding.line_start, c.finding.claim))


def _finding_line(item: Classified) -> str:
    f = item.finding
    tag = f" (verifier: {item.verification})" if item.verification else ""
    return f"- {_where(f)} [{f.cls}] [{', '.join(f.reviewers)}] {f.claim}{tag}"


def _evidence(item: Classified, verified: Sequence) -> list:
    """What the verifiers found for the finding, as the verified review shows it ([] when the finding was never verified)."""
    found = next((v for v in verified if v.finding == item.finding), None)
    return evidence_lines(found) if found else []


def format_classified_review(result: ClassifiedReview) -> str:
    """The classified review as text for the calling model: the findings that stand, grouped by class, each with the
    rubric entry, level and average probability it was given; what was not classified; and one line each for the findings
    dropped (by the verifiers, or for matching no entry)."""
    review = result.classification
    verified = result.verified
    verification = verified.verification
    reasoning = verified.reasoning
    problems = [*reasoning.errors, *verification.errors, *review.errors]
    # A failure in any of the three stages opens the text, before any count (see `incomplete_review_line`).
    header = incomplete_review_line(problems)
    lines = [] if header is None else [header]
    lines += [
        f"Classified review: {len(review.matched)} classified, {len(review.unchecked)} without a "
        f"rubric level; dropped {len(review.unclassified)} unclassified, {len(verification.rejected)} rejected and "
        f"{len(verification.ungrounded)} ungrounded "
        f"(of {len(verification.verified) + len(verification.ungrounded)} claim(s) from {len(reasoning.reviewers_run)} reasoning "
        f"reviewer(s): {', '.join(reasoning.reviewers_run) or 'none'}).",
    ]
    if review.classifiers_run:
        lines.append(
            f"Classifiers: {', '.join(review.classifiers_run)}. Each answer is the classifier's probability, from 0 to 1, that a "
            "finding falls under a rubric entry's definition. A finding takes an entry's level only when the average of every "
            "classifier's probability for that entry is at or above the entry's score; a finding that fits several entries takes "
            "the highest level."
        )
    elif review.unchecked:
        lines.append("No classifier ran, so these findings carry NO rubric level.")
    else:
        lines.append("No classifier ran.")
    if verification.verifiers_run:
        lines.append(f"Verifiers: {', '.join(verification.verifiers_run)}.")
    elif verification.supplied:
        lines.append("Each finding's verification status is the one its findings file gave: no verifier ran in this review.")
    elif verification.verified:
        lines.append("No verifier ran, so the findings are grounded (the code each one quotes is in the file) but not verified.")
    matched = _ordered(review.matched)
    if matched:
        classes = sorted({c.finding.cls for c in matched})
        for cls in classes:
            lines.append("")
            lines.append(f"## {cls}")
            for item in (c for c in matched if c.finding.cls == cls):
                lines.append(_finding_line(item))
                lines.extend(_evidence(item, verification.verified))
                text = f"    {item.entry} [{item.level}], average probability {_probability(item.confidence or 0)} over {len(item.classifications)} classifier(s)"
                if item.overlap:
                    text += f"; also fits {', '.join(item.overlapping)} (took the higher level)"
                lines.append(text)
    if review.unchecked:
        lines.append("")
        lines.append(f"## No rubric level ({len(review.unchecked)})")
        for item in _ordered(review.unchecked):
            lines.append(_finding_line(item))
            lines.extend(_evidence(item, verification.verified))
            if item.note:
                lines.append(f"    not classified: {item.note}")
    dropped = _ordered(review.unclassified)
    if dropped:
        lines.append("")
        lines.append(f"## Dropped: matched no rubric entry ({len(dropped)})")
        for item in dropped:
            lines.append(_finding_line(item))
            lines.append(f"    average probability per entry: {_describe_scores(item.scores)}")
    if verification.rejected:
        lines.append("")
        lines.append(f"## Dropped: rejected by the verifiers ({len(verification.rejected)})")
        for v in sorted(verification.rejected, key=lambda x: (x.finding.cls, x.finding.file, x.finding.line_start, x.finding.claim)):
            lines.append(f"- {_where(v.finding)} [{v.finding.cls}] {v.finding.claim}")
    if verification.ungrounded:
        lines.append("")
        lines.append(f"## Dropped: ungrounded ({len(verification.ungrounded)})")
        for u in sorted(verification.ungrounded, key=lambda x: (x.finding.cls, x.finding.file, x.finding.line_start, x.finding.claim)):
            lines.append(f"- {_where(u.finding)} [{u.finding.cls}] {u.finding.claim}")
            lines.append(f"    ungrounded: {u.reason}")
    if problems:
        lines.append("")
        lines.append(f"Problems ({len(problems)}):")
        lines.extend(f"- {e}" for e in problems)
    warnings = [*reasoning.warnings, *verification.warnings, *review.warnings]
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {w}" for w in warnings)
    return "\n".join(lines)


__all__ = [
    "CLASSIFIED",
    "UNCLASSIFIED",
    "UNCHECKED",
    "Rating",
    "Classification",
    "EntryScore",
    "Outcome",
    "Classified",
    "ClassificationReview",
    "ClassifiedReview",
    "build_entry_text",
    "build_classifier_prompt",
    "build_finding_text",
    "build_classifier_message",
    "reaches",
    "score_entries",
    "classify_answers",
    "classify_findings",
    "run_classified_review",
    "format_classified_review",
]
