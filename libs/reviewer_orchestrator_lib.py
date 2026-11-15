"""
The orchestrator of the `codebase-reviewer` server: a review as steps over one run's state, with a policy choosing
the next step and a loop that enforces what no policy may break.

Why it exists. Before this module the pipeline was fixed code: `run_classified_review` called `run_verified_review`,
which called `run_reasoning_review`, so nothing chose the next step at run time. Splitting the stages out as steps that
read and update one `RunState` lets a policy (rules in code first, a local model later) pick the next action, while the
loop, not the policy, keeps the invariants. This module changes no behaviour: `RulesPolicy` reproduces today's order
for each review type, and every step calls the same stage functions the `run_*` wrappers call.

What is here:

  * `RunState`: one per `review` call, in memory only (nothing carries across calls). Holds the inputs (config,
    loader, clients, the PINNED head commit, the diff, the recorder) and what each step produced.
  * The steps `reason`, `ground`, `verify`, `classify` and `finish`: each `(state, action) -> None`.
  * `Policy.next_action(state) -> Action`, and `RulesPolicy(review_type)`.
  * `run_orchestrated(state, policy)`: asks the policy for an action, checks it, runs it, repeats; returns the
    formatted result.

Invariants the loop enforces (an action that breaks one is recorded and replaced by `finish`):

  * each step runs at most once, `reason` first; `ground` only after `reason`; `verify` only after `ground`, and only
    on grounded findings; `classify` only after `verify`, and only on findings still standing (not rejected).
  * `targets`, when an action names them, must be findings the step could take. A strict subset is refused for now:
    what becomes of the findings left out is new behaviour, and belongs to a later change.
  * a step cap per run (`MAX_STEPS`); at the cap the run ends with a note in the result.
  * the run always finishes and the recorder always closes, also when a step raises (the exception propagates
    unchanged after the close, so the `review` tool's `Error:` text is the same as before).

Standard library only, apart from the sibling libs. The model clients are passed in, so tests run with stubs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence, override

from classifier_reviewer_lib import (
    ClassificationReview,
    ClassifiedReview,
    ProbabilityFn,
    classify_findings,
    format_classified_review,
)
from codebase_reviewer_config_lib import ConfigError, ReviewersConfig
from lmstudio_model_loader_lib import ModelLoader, ModelLoadError
from reasoning_reviewer_lib import (
    CompleteFn,
    ReasoningReview,
    ReviewError,
    format_review,
    run_reasoning_review,
    split_diff,
)
import verifier_reviewer_lib
from reviewer_db_lib import NULL_RECORDER
from reviewer_tools_lib import ToolExecutor
from verifier_reviewer_lib import (
    UNCHECKED,
    VerificationReview,
    Verified,
    VerifiedReview,
    caching_reader,
    format_verified_review,
    ground_step,
    preflight_models,
    standing,
    verify_step,
)

# The most steps (not counting `finish`) one run may take. The rules policies use at most five; the cap is the guard for
# the policies that come later, which can choose to repeat a step.
MAX_STEPS = 8

REASON = "reason"
GROUND = "ground"
VERIFY = "verify"
CLASSIFY = "classify"
FINISH = "finish"
STEPS = (REASON, GROUND, VERIFY, CLASSIFY)

# The stage a refused action or a hit cap is noted under in the run's record.
NOTE_STAGE = "orchestrator"


@dataclass(frozen=True)
class Action:
    """One thing a policy asks for: a step (or `finish`), and optionally the findings it should work on.
    `targets=None` means every finding the step is allowed to take."""

    name: str
    targets: Optional[tuple] = None


class Policy(Protocol):
    """Chooses the next action from the run's state. A policy may also have a `preflight_roles` attribute: the roles
    (in the order to check them) whose already-loaded models are checked before the first step, so a model that is
    not loaded is reported at once and not after minutes of reasoning."""

    def next_action(self, state: "RunState") -> Action: ...


@dataclass
class RunState:
    """Everything one review run knows. Inputs are set when the run is built; the rest is filled in by the steps."""

    config: ReviewersConfig
    loader: ModelLoader
    client: object
    repo_root: "str | Path"
    # The commit the whole run reads: the diff was cut from it and every later read of a cited file sees the same code.
    head: str
    diff: str
    max_reviewers: int
    chunk_chars: Optional[int] = None
    lessons: Sequence[str] = ()
    probability_client: Optional[object] = None
    recorder: Any = NULL_RECORDER
    # Test seams, the same ones the `run_*` functions have: a stub model call, a stub probability call, a stub file reader.
    complete: Optional[CompleteFn] = None
    probability: Optional[ProbabilityFn] = None
    read_file: Optional[Callable[[str], Optional[str]]] = None
    # The tool executor reviewers with `tools` ask for lookups through. Built once, on first use, from `repo_root` and
    # `head` and only when an active reviewer has `tools`, so a run with none never starts one; a test may set it to a stub.
    executor: object = None

    # What the steps produce.
    reasoning: Optional[ReasoningReview] = None
    findings: list = field(default_factory=list)
    read: Optional[Callable[[str], Optional[str]]] = None
    grounded: Optional[list] = None
    ungrounded: list = field(default_factory=list)
    verification: Optional[VerificationReview] = None
    classification: Optional[ClassificationReview] = None
    # Steps run so far, in order; and what the loop itself had to say (a refused action, the cap), shown as warnings.
    done: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    result: Optional[str] = None


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def _executor(state: RunState, role: str) -> object:
    """The run's tool executor when an active `role` reviewer has `tools`, else None (and none is built)."""
    if state.executor is None and any(r.tools for r in state.config.active_reviewers(role)):
        state.executor = ToolExecutor(state.repo_root, state.head)
    return state.executor


def reason(state: RunState, action: Action) -> None:
    """Every active reasoning reviewer over the diff (`run_reasoning_review`)."""
    state.reasoning = run_reasoning_review(
        state.config, state.loader, state.client, state.diff, state.repo_root, state.max_reviewers,
        chunk_chars=state.chunk_chars, lessons=state.lessons, complete=state.complete, recorder=state.recorder,
        executor=_executor(state, "reasoning"),
    )
    state.findings = [finding for group in state.reasoning.groups for finding in group.findings]


def ground(state: RunState, action: Action) -> None:
    """Drops every finding whose quoted code is not in the file as it is at the pinned head. Needs no model."""
    # Looked up on the module at call time (not imported by name), the same way `run_verified_review` reaches it, so a
    # test that replaces `verifier_reviewer_lib.git_file_reader` reaches both paths.
    reader = state.read_file or verifier_reviewer_lib.git_file_reader(
        state.repo_root, state.head, [section.path for section in split_diff(state.diff)]
    )
    state.read = caching_reader(reader)
    state.grounded, state.ungrounded = ground_step(state.findings, state.read, state.recorder)


def verify(state: RunState, action: Action) -> None:
    """Every active verifier confirms or rejects each grounded finding."""
    assert state.grounded is not None and state.read is not None
    state.verification = verify_step(
        state.config, state.loader, state.client, state.grounded, state.ungrounded, state.read, state.max_reviewers,
        complete=state.complete, recorder=state.recorder, executor=_executor(state, "verifier"),
    )


def classify(state: RunState, action: Action) -> None:
    """Every active classifier checks each finding the verifiers did not reject against the rubric."""
    assert state.verification is not None
    kept = standing(state.verification)
    state.classification = classify_findings(
        state.config, state.loader, state.client, [v.finding for v in kept], state.max_reviewers,
        statuses=[v.status for v in kept], probability=state.probability, probability_client=state.probability_client,
        recorder=state.recorder,
    )


def finish(state: RunState, action: Action) -> None:
    """Formats the result from whatever ran, and gives the run's record its counts, sizes and result text. What the loop
    had to say (`state.notes`) is added to the warnings of the last stage's result, so the caller sees it."""
    reasoning = state.reasoning
    if reasoning is None:
        detail = f": {'; '.join(state.notes)}" if state.notes else ""
        raise ReviewError(f"the review ended before any stage ran{detail}")
    verification = state.verification
    classification = state.classification
    if verification is None and state.grounded is not None:
        # The run ended after `ground` and before `verify` (the step cap, or an action the loop refused). Grounding
        # already decided what stands: report ITS result, not the raw reasoning findings, which would hand back the
        # findings it dropped as ordinary ones and lose the line numbers it corrected. Every surviving finding is
        # unchecked, since no verifier was asked about it; `format_verified_review` says exactly that.
        verification = VerificationReview(
            verified=[Verified(f, UNCHECKED, note="the run ended before the verifiers were asked") for f in state.grounded],
            ungrounded=state.ungrounded,
        )
    # Warnings go on the last result that exists, where the formatter for that kind of review reads them.
    (classification or verification or reasoning).warnings.extend(state.notes)
    if verification is not None and classification is not None:
        text = format_classified_review(ClassifiedReview(verified=VerifiedReview(reasoning, verification), classification=classification))
    elif verification is not None:
        text = format_verified_review(VerifiedReview(reasoning=reasoning, verification=verification))
    else:
        text = format_review(reasoning)
    counts: dict = {"chunks": reasoning.chunks, "findings": reasoning.finding_count}
    if verification is not None:
        counts["confirmed"] = len(verification.confirmed)
    state.recorder.set_counts(**counts)
    state.recorder.set_sizes(len(state.diff), len(text))
    state.recorder.set_result_text(text)
    state.result = text


_STEP_FUNCTIONS: dict = {REASON: reason, GROUND: ground, VERIFY: verify, CLASSIFY: classify, FINISH: finish}


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

# The order each review type has always run in, and the roles whose resident models are checked up front (in this order:
# the classifier's check came before the verifier's, which came before the reasoning stage's own).
_RULES: dict = {
    "reasoning": ((REASON,), ()),
    "verifier": ((REASON, GROUND, VERIFY), ("verifier",)),
    "classifier": ((REASON, GROUND, VERIFY, CLASSIFY), ("classifier", "verifier")),
}


class RulesPolicy(Policy):
    """Today's order for a review type: `reasoning` is reason; `verifier` is reason, ground, verify; `classifier` is
    reason, ground, verify, classify; each then `finish`."""

    def __init__(self, review_type: str):
        if review_type not in _RULES:
            raise ValueError(f"unknown review type {review_type!r}; known types: {', '.join(_RULES)}")
        self.review_type = review_type
        self.plan, self.preflight_roles = _RULES[review_type]

    @override
    def next_action(self, state: RunState) -> Action:
        for name in self.plan:
            if name not in state.done:
                return Action(name)
        return Action(FINISH)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _eligible(state: RunState, name: str) -> list:
    """The findings step `name` is allowed to take, given what has run."""
    if name == GROUND:
        return state.findings
    if name == VERIFY:
        return state.grounded or []
    if name == CLASSIFY:
        return [v.finding for v in standing(state.verification)] if state.verification is not None else []
    return []


# What a step needs to have run before it, and why that matters.
_NEEDS = {
    GROUND: (REASON, "there are no findings to ground before the reasoning stage has run"),
    VERIFY: (GROUND, "a finding must be grounded (its quoted code found in the file) before a verifier is asked about it"),
    CLASSIFY: (VERIFY, "only a finding still standing after verification is classified"),
}


def refusal(state: RunState, action: object) -> Optional[str]:
    """Why `action` is not allowed now, or None when it is. The loop's own check: a policy never needs to repeat it."""
    if not isinstance(action, Action):
        return f"the policy returned {action!r}, which is not an Action"
    name = action.name
    if name == FINISH:
        return None
    if name not in STEPS:
        return f"unknown action {name!r} (steps: {', '.join(STEPS)}, or {FINISH})"
    if name in state.done:
        return f"{name} already ran; each step runs at most once"
    if name in _NEEDS:
        needed, why = _NEEDS[name]
        if needed not in state.done:
            return f"{name} before {needed} is not allowed: {why}"
    if action.targets is None:
        return None
    if name == REASON:
        return "reason takes no targets"
    eligible = _eligible(state, name)
    outside = [t for t in action.targets if t not in eligible]
    if outside:
        return f"{name} was given {len(outside)} target(s) that are not findings it may take now (not grounded, or not standing)"
    if len(action.targets) != len(eligible) or any(f not in action.targets for f in eligible):
        return f"{name} was given {len(action.targets)} of {len(eligible)} eligible finding(s); a subset is not supported yet"
    return None


def _drive(state: RunState, policy: Policy, max_steps: int) -> str:
    for role in getattr(policy, "preflight_roles", ()):
        preflight_models(state.config, state.loader, (role,))
    steps = 0
    while True:
        action = policy.next_action(state)
        problem = refusal(state, action)
        if problem is not None:
            state.recorder.note(NOTE_STAGE, f"refused: {problem}; finishing the run")
            state.notes.append(f"the orchestrator refused an action ({problem}) and ended the run")
            break
        if action.name == FINISH:
            break
        if steps >= max_steps:
            problem = f"the step cap ({max_steps}) was reached before the policy finished"
            state.recorder.note(NOTE_STAGE, f"{problem}; finishing the run")
            state.notes.append(f"{problem}; the run ended there")
            break
        _STEP_FUNCTIONS[action.name](state, action)
        state.done.append(action.name)
        steps += 1
    finish(state, Action(FINISH))
    assert state.result is not None
    return state.result


def run_orchestrated(state: RunState, policy: Policy, *, max_steps: int = MAX_STEPS) -> str:
    """Runs one review: asks `policy` for an action until it says `finish` (or an action is refused, or the cap is hit),
    and returns the formatted result. The run's recorder is closed whichever way this ends: `finished`, `refused` for a
    configuration, model-loading or review error (the same three the server reports as an `Error:`), `errored` for
    anything else; an exception is re-raised unchanged after the close."""
    try:
        text = _drive(state, policy, max_steps)
    except (ConfigError, ModelLoadError, ReviewError) as exc:
        state.recorder.close("refused", error=str(exc))
        raise
    except BaseException as exc:
        state.recorder.close("errored", error=f"{type(exc).__name__}: {exc}")
        raise
    state.recorder.close("finished")
    return text


__all__ = [
    "MAX_STEPS",
    "REASON",
    "GROUND",
    "VERIFY",
    "CLASSIFY",
    "FINISH",
    "STEPS",
    "Action",
    "Policy",
    "RunState",
    "RulesPolicy",
    "reason",
    "ground",
    "verify",
    "classify",
    "finish",
    "refusal",
    "run_orchestrated",
]
