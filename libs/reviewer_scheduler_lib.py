"""
Staged scheduling for the `codebase-reviewer` server: which reviewers run when,
in what groups, and when each `loadOnDemand` model is loaded and unloaded.

The rules (from the design this implements):

  - STAGES run in order: every reasoning reviewer, then every verifier, then
    every classifier, because each stage needs the one before it. A reviewer
    with several roles takes part in each of its stages. A stage nobody is
    active in is skipped.
  - PARALLELISM within a stage is up to `max_reviewers` at once (`1` runs them
    one at a time).
  - GROUPING: reviewers that share a model are scheduled together, so a model is
    loaded once per stage however many reviewers use it.
  - LOADING: a `loadOnDemand` model is loaded just before its group runs and
    unloaded right after, unless the NEXT stage also uses it, in which case it
    stays loaded. Models are unloaded only if this run loaded them (see
    `lmstudio_model_loader_lib`).

Two design choices worth knowing, since the rules above leave room:

  - A `loadOnDemand` model's group runs ALONE: loaded, run, (maybe) unloaded,
    before the next one starts. `loadOnDemand` exists for machines that cannot
    hold every model at once, so two on-demand models must never be loaded at
    the same time just to fill `max_reviewers` slots. Models that are already
    resident (`loadOnDemand` false) cost nothing to hold, so ALL of a stage's
    resident reviewers share one pool, run first, and are what `max_reviewers`
    parallelism applies to; each on-demand group then gets the same limit
    within itself.
  - Within a stage, on-demand groups are ordered so that fewer models have to
    be unloaded and reloaded: a model already held from the previous stage goes
    first (it is loaded, so it costs nothing), and a model the NEXT stage will
    use again goes last (so it can stay loaded straight into that stage with
    nothing in between). Whatever the order, a group that has to load first
    unloads every OTHER model this run still holds, so two on-demand models are
    never loaded together even when one was being kept for the next stage; if
    one of those will not unload, that group is skipped with an error rather
    than loaded on top of it.

Standard library only. The reviewer itself is a callable the caller supplies,
so this module knows nothing about prompts, models' APIs or findings.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from codebase_reviewer_config_lib import ROLES, Reviewer, ReviewersConfig
from lmstudio_model_loader_lib import ModelLoader, ModelLoadError, ModelNotLoadedError


@dataclass(frozen=True)
class ModelGroup:
    """The reviewers of one stage that share a model, in file order."""

    model: str
    load_on_demand: bool
    context_length: Optional[int]
    reviewers: tuple


@dataclass(frozen=True)
class Stage:
    role: str
    # Groups whose model must already be loaded, in first-use order.
    resident: tuple = ()
    # Groups the server loads and unloads itself, in first-use order.
    on_demand: tuple = ()

    def reviewers(self) -> list:
        """Every reviewer in this stage, resident groups first, each group contiguous."""
        return [r for g in (*self.resident, *self.on_demand) for r in g.reviewers]

    def models(self) -> set:
        return {g.model for g in (*self.resident, *self.on_demand)}


@dataclass
class ReviewerOutcome:
    role: str
    reviewer: Reviewer
    value: object = None
    error: Optional[str] = None


@dataclass
class RunReport:
    outcomes: list = field(default_factory=list)
    # Problems that did not stop the run: a model that would not unload.
    warnings: list = field(default_factory=list)


def plan_stages(config: ReviewersConfig, roles: Sequence[str] = ROLES) -> list:
    """The stage plan for the ACTIVE reviewers of `config`. Pure: touches nothing.

    `roles` limits which stages are planned (default all three, in stage order;
    the order of `ROLES` always wins over the order given)."""
    stages = []
    for role in ROLES:
        if role not in roles:
            continue
        by_model: dict = {}
        for reviewer in config.active_reviewers(role):
            by_model.setdefault(reviewer.model, []).append(reviewer)
        resident = []
        on_demand = []
        for model, members in by_model.items():
            # The config validator guarantees reviewers sharing a model agree
            # on loadOnDemand and contextLength, so the first speaks for all.
            first = members[0]
            group = ModelGroup(
                model=model,
                load_on_demand=first.load_on_demand,
                context_length=first.context_length,
                reviewers=tuple(members),
            )
            (on_demand if group.load_on_demand else resident).append(group)
        if resident or on_demand:
            stages.append(Stage(role=role, resident=tuple(resident), on_demand=tuple(on_demand)))
    return stages


def order_for_carry_over(
    groups: Sequence[ModelGroup], held_models: Iterable[str], upcoming_models: Iterable[str] = ()
) -> list:
    """`groups` ordered to minimise reloads, stable within each part: models
    already held first, then models the next stage does not need, then models
    the next stage will use again (last, so they can stay loaded into it)."""
    held = set(held_models)
    upcoming = set(upcoming_models)
    first = [g for g in groups if g.model in held]
    middle = [g for g in groups if g.model not in held and g.model not in upcoming]
    last = [g for g in groups if g.model not in held and g.model in upcoming]
    return first + middle + last


RunReviewerFn = Callable[[Reviewer, str], object]


def _run_segment(reviewers: Sequence[Reviewer], role: str, run_reviewer: RunReviewerFn, max_workers: int) -> list:
    """Runs `reviewers` with at most `max_workers` at once and returns their
    outcomes in the order given. A reviewer that raises becomes an outcome with
    an error; it never stops the others."""

    def one(reviewer: Reviewer) -> ReviewerOutcome:
        try:
            return ReviewerOutcome(role=role, reviewer=reviewer, value=run_reviewer(reviewer, role))
        except Exception as exc:
            return ReviewerOutcome(role=role, reviewer=reviewer, error=f"{type(exc).__name__}: {exc}")

    if not reviewers:
        return []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        return list(pool.map(one, reviewers))


def check_resident_models(stages: Sequence[Stage], loader: ModelLoader) -> None:
    """Raises one `ModelNotLoadedError` naming EVERY resident model that is not
    loaded, before anything has run."""
    missing = []
    for stage in stages:
        for group in stage.resident:
            if group.model in missing:
                continue
            try:
                loader.acquire(group.model, load_on_demand=False)
            except ModelNotLoadedError:
                missing.append(group.model)
    if missing:
        names = ", ".join(repr(m) for m in missing)
        raise ModelNotLoadedError(
            f"model(s) {names} are not loaded in LM Studio, and their reviewers are not loadOnDemand: "
            "load them in LM Studio, or set loadOnDemand to true for the reviewers that use them"
        )


def run_stages(
    config: ReviewersConfig,
    loader: ModelLoader,
    run_reviewer: RunReviewerFn,
    max_reviewers: int,
    *,
    roles: Sequence[str] = ROLES,
) -> RunReport:
    """Runs the stage plan for `config`'s active reviewers.

    `run_reviewer(reviewer, role)` does one reviewer's work for one stage and
    returns its result (or raises). Raises `ModelNotLoadedError` BEFORE any
    reviewer runs if a resident model is not loaded. A `loadOnDemand` model
    that fails to load fails only the reviewers that needed it (their outcomes
    carry the error); the rest of the run continues. Everything this call
    loaded is unloaded before it returns, even on an exception."""
    stages = plan_stages(config, roles)
    report = RunReport()
    check_resident_models(stages, loader)
    try:
        for index, stage in enumerate(stages):
            upcoming = stages[index + 1].models() if index + 1 < len(stages) else set()
            report.outcomes.extend(
                _run_segment([r for g in stage.resident for r in g.reviewers], stage.role, run_reviewer, max_reviewers)
            )
            for group in order_for_carry_over(stage.on_demand, loader.owned_models, upcoming):
                try:
                    # Never load on top of a model this run still holds: the
                    # point of loadOnDemand is a machine that cannot hold both.
                    for other in [m for m in loader.owned_models if m != group.model]:
                        loader.release(other)
                    loader.acquire(group.model, load_on_demand=True, context_length=group.context_length)
                except ModelLoadError as exc:
                    report.outcomes.extend(
                        ReviewerOutcome(role=stage.role, reviewer=r, error=str(exc)) for r in group.reviewers
                    )
                    continue
                report.outcomes.extend(_run_segment(group.reviewers, stage.role, run_reviewer, max_reviewers))
                if group.model not in upcoming:
                    try:
                        loader.release(group.model)
                    except ModelLoadError as exc:
                        report.warnings.append(str(exc))
    finally:
        report.warnings.extend(loader.release_all())
    return report


def describe_plan(stages: Sequence[Stage]) -> str:
    """A readable stage plan for the `list_reviewers` tool."""
    if not stages:
        return "Stage plan: no active reviewers."
    lines = ["Stage plan (stages run in this order):"]
    for number, stage in enumerate(stages, start=1):
        lines.append(f"  {number}. {stage.role}")
        for group in (*stage.resident, *stage.on_demand):
            how = "loaded on demand" if group.load_on_demand else "already loaded"
            names = ", ".join(r.name for r in group.reviewers)
            lines.append(f"     - model {group.model} ({how}): {names}")
    return "\n".join(lines)
