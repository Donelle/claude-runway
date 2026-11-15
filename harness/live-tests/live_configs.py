"""What a live test is, worked out from a file in `configs/`: shared by `run_live_tests.py` and the pre-flight.

A config in `configs/` IS a live test. Nothing about it is written in Python:

- the script that runs it comes from the config's `stages` (`kind_for`);
- the models come from its reviewers;
- the quality bars, when there are any, come from an optional sidecar next to it, `<name>.baselines.json`
  (`load_baselines`). The shared config format has no key for test metadata (`harness_lib` rejects an unknown key, and
  all three scripts read that format), so the bars live beside the config and not in it.
- a sidecar also says what its bars were counted on: a `runType` (`synthetic` or `live`, no default) and a `metadata`
  object. A synthetic run's `metadata.cases` names the cases in `harness/cases.py`, and the runner passes exactly those
  to the script (`script_args`), so a bar is never applied to a different set of cases than it was measured on.

The runner and the pre-flight both call `discover`, so they cannot disagree about which files are configs.

Stdlib only, apart from `harness_lib.load_config`, which `describe` uses to read a config the way the scripts do.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
CONFIGS = HERE / "configs"
BASELINES_SUFFIX = ".baselines.json"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))


@dataclass(frozen=True)
class Kind:
    """One way to run a config: the script, how many runs by default, and the fixed arguments it takes."""

    name: str  # keys the checking function in run_live_tests.py
    script: str  # a script in harness/
    runs: int
    args: tuple = ()


# The stages a config selects decide the script. A set, so the order a config lists them in does not matter.
KINDS = {
    frozenset({"verifier"}): Kind("verifier", "live_verifier.py", runs=3),
    frozenset({"classifier"}): Kind("classifier", "live_classifier.py", runs=2),
    frozenset({"reasoning"}): Kind("reasoning", "live_reasoning.py", runs=2),
    # live_all_reviewers.py runs a stage after reasoning, so every combination with reasoning in it is a pipeline. The
    # classifier sorts what the verifiers left standing, or every finding when no verifier is selected. A combination
    # without reasoning and with more than one stage has no script: the stage scripts each run one stage on fixtures.
    # The `--case orders` is for a config with no sidecar (mechanics only, no bars to misapply). With a sidecar the cases
    # come from its `metadata.cases` instead (`script_args`).
    frozenset({"reasoning", "verifier"}): Kind("pipeline", "live_all_reviewers.py", runs=2, args=("--case", "orders")),
    frozenset({"reasoning", "classifier"}): Kind("pipeline", "live_all_reviewers.py", runs=2, args=("--case", "orders")),
    frozenset({"reasoning", "verifier", "classifier"}): Kind("pipeline", "live_all_reviewers.py", runs=2, args=("--case", "orders")),
}


class BaselinesError(Exception):
    """A baselines file that exists but cannot be used: said out loud, not read as "no baselines"."""


# The two run types a bar can belong to. `synthetic` runs mock scenarios from `harness/cases.py`; `live` runs the
# `harness-live-tests` branch, scored from the tags in its `metadata` (`harness/live_branch_lib.py`).
SYNTHETIC = "synthetic"
LIVE = "live"
RUN_TYPES = (SYNTHETIC, LIVE)
ALL = "all"  # in `metadata.cases`: every case the kind's script can run


@dataclass(frozen=True)
class Baselines:
    measured: str  # the date they were measured
    model: str  # what they were measured with: they are specific to it
    run_type: str  # SYNTHETIC or LIVE: which kind of run the numbers were counted on. No default: see `load_baselines`
    metadata: dict  # what the run type needs to choose its data; a synthetic run's `cases`
    expect: dict = field(default_factory=dict)  # the numbers, by the keys the kind's checking function reads

    @property
    def note(self) -> str:
        return f"baseline measured {self.measured} with {self.model}"

    @property
    def cases(self) -> tuple:
        """The cases a synthetic run was measured on, as written in the file (`all` unexpanded); empty for any other."""
        return tuple(self.metadata.get("cases", ())) if self.run_type == SYNTHETIC else ()

    @property
    def tags(self) -> tuple:
        """The `LiveTag`s a live run was measured on (`live_branch_lib.parse_live_metadata`); empty for any other.
        `load_baselines` has already checked them, so this cannot raise for a loaded file."""
        if self.run_type != LIVE:
            return ()
        import live_branch_lib

        return live_branch_lib.parse_live_metadata(self.metadata)


@dataclass(frozen=True)
class LiveConfig:
    """A config as the runner sees it: runnable (`kind` is set), or not with `problem` saying why.

    `unrunnable` separates a config that is fine but has no script (a combination of stages that none runs) from one that
    is broken (it does not load, or its baselines file is unusable): the first is a note, the second fails a run."""

    name: str
    path: Path
    stages: tuple = ()
    models: tuple = ()
    kind: Optional[Kind] = None
    baselines: Optional[Baselines] = None
    problem: str = ""
    unrunnable: bool = False


def discover(configs_dir: Path = CONFIGS) -> list:
    """Every config in the folder, by file name: `*.json` except the baselines sidecars."""
    return sorted(p for p in Path(configs_dir).glob("*.json") if not p.name.endswith(BASELINES_SUFFIX))


def kind_for(stages) -> Optional[Kind]:
    return KINDS.get(frozenset(stages))


def baselines_path(config: Path) -> Path:
    return config.with_name(config.stem + BASELINES_SUFFIX)


def load_baselines(config: Path) -> Optional[Baselines]:
    """The sidecar's numbers, `None` when the config has no sidecar, `BaselinesError` when it has a broken one."""
    path = baselines_path(config)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BaselinesError(f"{path.name} cannot be read as JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise BaselinesError(f"{path.name} must be a JSON object")
    missing = [key for key in ("measured", "model", "expect") if key not in data]
    if missing:
        raise BaselinesError(f"{path.name} lacks {missing}: bars are specific to the model they were measured with, so it names it and the date")
    if not isinstance(data["expect"], dict) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in data["expect"].values()):
        raise BaselinesError(f"{path.name}: 'expect' must be an object of numbers")
    # No default for a missing `runType`: a bar that does not say which kind of run it was counted on could be read as
    # belonging to one it was not measured on, and then pass or fail for the wrong reason.
    if "runType" not in data:
        raise BaselinesError(f"{path.name} has no 'runType': a bar belongs to the run type it was measured on, so it must say {' or '.join(map(repr, RUN_TYPES))}")
    if data["runType"] not in RUN_TYPES:
        raise BaselinesError(f"{path.name}: 'runType' is {data['runType']!r}, which is neither {' nor '.join(map(repr, RUN_TYPES))}")
    if not isinstance(data.get("metadata"), dict):
        raise BaselinesError(f"{path.name} must have a 'metadata' object (for a synthetic run, the cases the bars were counted on)")
    # Whether `cases` is required depends on the script the config runs (the classifier takes no --case), which the
    # loader does not know: `describe` checks that. What is written is checked here.
    if data["runType"] == SYNTHETIC and "cases" in data["metadata"]:
        _check_cases(path.name, data["metadata"]["cases"])
    if data["runType"] == LIVE:
        # A live answer key is checked as strictly as a synthetic case list: a misspelt key or a bug left out of a tag would
        # change a score without anything saying so. `expect` may carry only the bars a live run reads.
        import live_branch_lib

        try:
            live_branch_lib.parse_live_metadata(data["metadata"])
        except live_branch_lib.LiveMetadataError as exc:
            raise BaselinesError(f"{path.name}: {exc}") from exc
        unknown = live_branch_lib.check_bars(data["expect"])
        if unknown:
            raise BaselinesError(f"{path.name}: {unknown}")
    return Baselines(str(data["measured"]), str(data["model"]), data["runType"], dict(data["metadata"]), dict(data["expect"]))


def _check_cases(file_name: str, cases) -> None:
    """A synthetic run's `metadata.cases` when it has one: a list of names from `harness/cases.py`, or `["all"]` alone."""
    from cases import ALL_CASES

    if not isinstance(cases, list) or not cases or not all(isinstance(c, str) for c in cases):
        raise BaselinesError(f"{file_name}: 'cases' must be a non-empty list of case names (or [\"{ALL}\"])")
    if ALL in cases and len(cases) > 1:
        raise BaselinesError(f"{file_name}: 'cases' mixes \"{ALL}\" with named cases; use [\"{ALL}\"] alone or name them")
    unknown = [c for c in cases if c != ALL and c not in ALL_CASES]
    if unknown:
        raise BaselinesError(f"{file_name}: 'cases' names {unknown}, which harness/cases.py does not have (it has {list(ALL_CASES)})")
    if len(set(cases)) != len(cases):
        raise BaselinesError(f"{file_name}: 'cases' names a case twice")


def expected_cases(kind: Kind, baselines: Baselines) -> list:
    """The case names a sidecar's synthetic bars were counted on, `all` expanded the way this kind's script takes cases.

    The runner passes these names explicitly and compares them with what the output reports, so what `all` means is
    decided here, once, and not by each script's own `--case all` (reasoning has none)."""
    from cases import ALL_CASES, fixture_cases

    cases = list(baselines.cases)
    if cases == [ALL]:
        return list(fixture_cases() if kind.name == "verifier" else ALL_CASES)
    return cases


def expected_tags(baselines: Baselines) -> list:
    """The tag ids (`change-1`, ...) a live sidecar's bars were counted on, in the order it lists them."""
    return [tag.tag_id for tag in baselines.tags]


def script_args(config: "LiveConfig") -> tuple:
    """The fixed arguments the config's script is run with: `--case NAME` for each case its sidecar's bars were counted on
    (so the script runs exactly that set), `--run-type live` for a live sidecar (the script reads its tags from that sidecar
    itself), or the kind's own arguments when there is no sidecar (mechanics only)."""
    assert config.kind is not None
    if config.baselines is not None and config.baselines.run_type == LIVE:
        return ("--run-type", LIVE)
    if config.baselines is None or config.baselines.run_type != SYNTHETIC or config.kind.name == "classifier":
        return config.kind.args
    return tuple(arg for name in expected_cases(config.kind, config.baselines) for arg in ("--case", name))


def _baselines_problem(kind: Kind, baselines: Baselines) -> tuple:
    """`(problem, unrunnable)` for a sidecar the kind's script cannot be run with, or `("", False)`."""
    if baselines.run_type == LIVE:
        if kind.name in ("verifier", "classifier"):
            # Not run as a synthetic one either: a live bar belongs to the run type it was measured on.
            return (f"its baselines are for a 'live' run, which is not available for the {kind.name} script (it runs fixtures from "
                    "harness/cases.py and there are no live fixtures yet); a live bar is never run as a synthetic one", True)
        return ("", False)
    if kind.name != "classifier" and not baselines.cases:
        return (f"a synthetic run's 'metadata' needs 'cases', the cases its bars were counted on (a list of names, or [\"{ALL}\"])", False)
    cases = expected_cases(kind, baselines)
    if kind.name == "classifier" and baselines.cases:
        return ("its baselines name cases, but the classifier script takes no --case: it runs its own fixtures", False)
    if kind.name == "pipeline" and len(cases) != 1:
        return (f"the pipeline script runs one case at a time, but its baselines name {len(cases) or 'no'} ({', '.join(cases) or 'none'}); name exactly one", False)
    if kind.name == "verifier":
        from cases import fixture_cases

        no_fixtures = [c for c in cases if c not in fixture_cases()]
        if no_fixtures:
            return (f"the verifier script takes cases that have fixtures, and {no_fixtures} has none (it has {list(fixture_cases())})", False)
    return ("", False)


def describe(config: Path) -> LiveConfig:
    """Read one config the way the harness scripts do. Never raises: a config that does not load carries `problem`."""
    import harness_lib as h
    from codebase_reviewer_config_lib import ConfigError

    try:
        loaded = h.load_config(str(config))
    except ConfigError as exc:
        return LiveConfig(config.stem, config, problem=str(exc))
    stages = tuple(loaded.stages)
    # Only the reviewers a selected stage runs: a verifier left in a config that selects `reasoning` alone is not
    # loaded by live_reasoning.py, so its model being absent from LM Studio must not stop the run.
    models = tuple(sorted({r["model"] for r in loaded.reviewers if r.get("active", True) and set(r["type"]) & set(stages)}))
    try:
        baselines = load_baselines(config)
    except BaselinesError as exc:
        return LiveConfig(config.stem, config, stages, models, problem=str(exc))
    kind = kind_for(stages)
    if kind is None:
        return LiveConfig(config.stem, config, stages, models, baselines=baselines,
                          problem=f"no live test script runs the stages {list(stages)} (the ones it can run: {sorted(sorted(k) for k in KINDS)})",
                          unrunnable=True)
    if baselines is not None:
        problem, unrunnable = _baselines_problem(kind, baselines)
        if problem:
            return LiveConfig(config.stem, config, stages, models, kind, baselines,
                              problem=f"{baselines_path(config).name}: {problem}", unrunnable=unrunnable)
    return LiveConfig(config.stem, config, stages, models, kind, baselines)
