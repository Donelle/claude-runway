#!/usr/bin/env python3
"""
MCP server `codebase-reviewer`: a local-model review loop that runs on LM Studio
before (and between) Copilot reviews, so the findings Claude pays to read have
already been checked.

What exists today: configuration, the reviewers file, model loading and staged
scheduling, a database of what every review cost, and three tools. `list_reviewers` validates the configuration and shows
what would run. `review(type, params)` runs a review; the types are
"reasoning" (the reasoning reviewers over a git diff), "verifier" (the same, then
a check that each finding's quoted code is really in the file, then the verifier
reviewers confirming or rejecting what is left) and "classifier" (the same again,
then the classifier reviewers matching each finding that stands to the rubric).
Each stage is its own type on the same tool, so its schema stays one entry however
many stages exist. The "verifier" and "classifier" types can instead run alone on
findings read from a findings file (`findings_file` in `params`), and a review can save its
complete results as a file (see REVIEW_OUTPUT_DIR below). `review_report` reads the database every review run
writes (libs/reviewer_db_lib.py): response time per model, review time per model
combination, and which combinations finish under a target time.

It is standalone, laid out like `compress_mcp_server.py` and imports nothing
from any other server, so it can be installed or left out on its own. It shares
only `libs/`: `lmstudio_client.py` (LM Studio's URL and timeout) and `openai_lib.py`
(the request and the bounded client, no retries) for the model client, and the `libs/` modules named below.

Configuration (the server's `env` block in .mcp.json):
  REVIEWERS_FILE      REQUIRED, no default. Path to the reviewers JSON file (the
                      models, prompts and rubric are entirely the user's). A
                      missing, unreadable or invalid file is an error that names
                      the path and what is wrong, and no review runs.
                      See libs/codebase_reviewer_config_lib.py for the schema.
  MAX_REVIEWERS       How many reviewers run at once (default 3; "1" runs them
                      one at a time).
  MAX_REVIEW_ROUNDS   The most rounds of Copilot feedback an unattended run goes
                      through before a human is brought in (default 3).
  QDRANT_URL / QDRANT_API_KEY
                      Where the scoring database will live (a later piece of
                      work); read now so the block is complete.
  CLAUDE_RUNWAY_LMSTUDIO_URL
                      LM Studio's OpenAI-compatible URL, inherited from the shell
                      like the other servers (init does not write it here).
  CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB
                      This server's own database of review runs (default
                      ~/.claude/claude-runway/codebase-reviewer.db); its per-run
                      logs go in a codebase-reviewer-logs folder beside it.
  REVIEW_ID           Optional, unset by default. An ID that groups the runs of one
                      review (a harness that runs each reviewer in its own session
                      sets the same one for all of them). It is stored in the run
                      record's `settings` and named in the results file.
  REVIEW_OUTPUT_DIR   Optional, unset by default. A folder: each review also saves
                      its complete structured results there, one file per review
                      (libs/reviewer_files_lib.py). The caller never sees the
                      file or its name. A folder that cannot be used refuses the
                      review before any model runs.

Model loading is in libs/lmstudio_model_loader_lib.py, scheduling in
libs/reviewer_scheduler_lib.py.

Project minimum: Python 3.12.

Register in .mcp.json (or let `setup_project.py init --codebase-reviewer` do it):
    "codebase-reviewer": {
      "command": "python",
      "args": ["/absolute/path/to/mcpservers/codebase_reviewer_mcp_server.py"],
      "env": {"REVIEWERS_FILE": "/path/you/choose/reviewers.json"}
    }
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from mcp.server.mcpserver import MCPServer

from codebase_reviewer_config_lib import ConfigError, advisories, describe_config, load_reviewers_file, read_settings
from lmstudio_client import DEFAULT_BASE_URL, LMStudioClient, build_client
from lmstudio_model_loader_lib import ModelLoader, ModelLoadError, reviewer_api_url
from reasoning_reviewer_lib import ReviewError
from reviewer_db_lib import RunRecorder, new_run_recorder
from reviewer_files_lib import prepare_results_folder, read_findings_file, write_results
from reviewer_orchestrator_lib import RulesPolicy, RunState, results_of, run_orchestrated, supply_findings
from reviewer_report_lib import ReportError, report as reviewer_report
from reviewer_scheduler_lib import describe_plan, plan_stages

mcp = MCPServer("codebase-reviewer")


def lmstudio_base_url() -> str:
    """LM Studio's OpenAI-compatible URL: `CLAUDE_RUNWAY_LMSTUDIO_URL`, else the library default.
    Read per call (not at import) so a test or a restarted env is honoured."""
    return os.environ.get("CLAUDE_RUNWAY_LMSTUDIO_URL", DEFAULT_BASE_URL)


def new_client() -> LMStudioClient:
    """A bounded LM Studio client (finite timeout, no retries), the same construction the other servers use,
    built on LM Studio's native `/api/v0` so each response carries its timing (see `reviewer_api_url`)."""
    return build_client(reviewer_api_url(lmstudio_base_url()))


def new_probability_client() -> LMStudioClient:
    """The client the classifier's requests go through, built on LM Studio's OpenAI-compatible `/v1`: the native `/api/v0`
    that `new_client` uses returns no token probabilities, which a classifier needs. The cost is that those requests carry
    no timing statistics."""
    return build_client(lmstudio_base_url())


@mcp.tool()
def list_reviewers(check_models: bool = False) -> str:
    """
    Validates this project's codebase-reviewer configuration and shows what a
    review would run: the settings, the rubric, every reviewer, and the stage
    plan (reasoning, then verifier, then classifier; reviewers sharing a model
    grouped; which models are loaded on demand). Runs no review and loads no
    model.

    The reviewers file is REVIEWERS_FILE, required with no default. If it is
    missing, unreadable or invalid this returns an "Error:" naming the file and
    EVERY problem in it, and no review would run until it is fixed.

    check_models: when true, also asks LM Studio what is loaded and reports each
      active reviewer whose model must already be loaded (loadOnDemand false)
      and is not.
    """
    try:
        settings = read_settings(os.environ)
        config = load_reviewers_file(settings.reviewers_file)
    except ConfigError as exc:
        return f"Error: {exc}"
    stages = plan_stages(config)
    sections = [describe_config(config, settings), describe_plan(stages)]
    notes = advisories(config)
    if notes:
        sections.append("Advisories:\n" + "\n".join(f"- {note}" for note in notes))
    if check_models:
        sections.append(_describe_loaded_models(stages))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# `review(type, params)`: one tool, one entry per kind of review
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """One key of a review type's `params`."""

    name: str
    kind: type  # str or int
    doc: str
    required: bool = False
    default: object = None
    minimum: Optional[int] = None


@dataclass(frozen=True)
class ReviewType:
    summary: str
    params: tuple
    run: Callable[[dict], str]


GIT_TIMEOUT_SECONDS = 120


def _resolve_commit(repo_path: str, ref: str) -> str:
    """The commit `ref` names right now, as a full hash. A run takes minutes, and `HEAD` or a branch
    name can move meanwhile (a commit, a checkout in another session), so a review that reads files
    after the diff was cut must name the commit it cut the diff from. Raises `ReviewError` with git's
    own message on a bad path or ref; a ref that starts with `-` is refused rather than handed to git."""
    root = Path(os.path.expanduser(repo_path))
    if not root.is_dir():
        raise ReviewError(f"repo_path {str(root)!r} is not a directory")
    if ref.startswith("-") or any(c.isspace() or c == "\0" for c in ref):
        raise ReviewError(f"head_ref {ref!r} is not a usable git ref")
    try:
        done = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                              capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReviewError(f"git rev-parse could not run in {str(root)!r} ({type(exc).__name__}: {exc})") from exc
    commit = done.stdout.decode("utf-8", errors="replace").strip()
    if done.returncode != 0 or not commit:
        raise ReviewError(f"head_ref {ref!r} is not a commit in {str(root)!r}")
    return commit


def _git_diff(repo_path: str, base_ref: str, head_ref: str, context_lines: int) -> str:
    """`git diff base...head` for the repository at `repo_path`, with
    `context_lines` lines of surrounding code (so reviewers see more than the
    changed lines). Raises `ReviewError` with git's own message on a bad path
    or ref. A ref that starts with `-` is refused rather than handed to git as
    an option."""
    root = Path(os.path.expanduser(repo_path))
    if not root.is_dir():
        raise ReviewError(f"repo_path {str(root)!r} is not a directory")
    for name, ref in (("base_ref", base_ref), ("head_ref", head_ref)):
        if ref.startswith("-") or any(c.isspace() or c == "\0" for c in ref):
            raise ReviewError(f"{name} {ref!r} is not a usable git ref")
    # Explicit prefixes: the parser strips `a/` and `b/`, and a repository or user setting such as
    # `diff.noprefix` would otherwise make a change to `b/auth.py` read as `auth.py`.
    # --no-ext-diff and --no-textconv: a repository's config can name a `diff.external` or
    # `diff.<driver>.textconv` command, which git would run during the diff. Reviewing a repository
    # must not run its commands, and reviewers must see the real content, not a converted form.
    command = ["git", "-C", str(root), "diff", "--no-color", "--no-ext-diff", "--no-textconv",
               "--src-prefix=a/", "--dst-prefix=b/", f"--unified={context_lines}", f"{base_ref}...{head_ref}", "--"]
    try:
        done = subprocess.run(command, capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReviewError(f"git diff could not run in {str(root)!r} ({type(exc).__name__}: {exc})") from exc
    if done.returncode != 0:
        message = done.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise ReviewError(f"git diff {base_ref}...{head_ref} failed in {str(root)!r}: {message[0] if message else 'no message'}")
    return done.stdout.decode("utf-8", errors="replace")


def _recorder_for(review_type: str, values: dict) -> RunRecorder:
    """The run recorder for one review (its own database; see libs/reviewer_db_lib.py). Recording never
    changes a review: every write inside it fails open. It is made before the settings and the reviewers
    file are read, so a review refused by a bad configuration is recorded too."""
    return new_run_recorder(
        source="mcp",
        review_type=review_type,
        repo_path=values["repo_path"],
        base_ref=values["base_ref"],
        head_ref=values["head_ref"],
        pr=values.get("pr"),
        settings={"chunk_chars": values.get("chunk_chars"), "context_lines": values["context_lines"]},
        lmstudio_url=lmstudio_base_url(),
    )


def _configured(recorder: RunRecorder) -> tuple:
    """`(settings, reviewers config)` for one review, read inside its recorded body (a ConfigError is then
    recorded as refused). The setting that changes the cost is added to the run's record, and so is the review ID
    when one is set (`REVIEW_ID`: the record then says which review the run belongs to)."""
    settings = read_settings(os.environ)
    # Before the reviewers file is read, so a run refused by a bad reviewers file still carries its review ID.
    if settings.review_id:
        recorder.settings["review_id"] = settings.review_id
    config = load_reviewers_file(settings.reviewers_file)
    recorder.settings["max_reviewers"] = settings.max_reviewers
    return settings, config


def _recorded(recorder: RunRecorder, body: Callable[[], str]) -> str:
    """Runs `body` and closes the run's record whichever way it ends: a review refused before it could
    run (a bad configuration, a bad range, a model that is not loaded) is recorded as refused, any other
    failure as errored. A review that returns is `finished`, or `incomplete` when a stage got no answer
    from any reviewer (the recorder decides that from what it saw)."""
    try:
        text = body()
    except (ConfigError, ModelLoadError, ReviewError) as exc:
        recorder.close("refused", error=str(exc))
        raise
    except BaseException as exc:
        recorder.close("errored", error=f"{type(exc).__name__}: {exc}")
        raise
    recorder.close("finished")
    return text


def _save_results(folder: Path, state: RunState, review_type: str, recorder: RunRecorder, review_id: str) -> str:
    """Writes a finished run's structured results into `folder`; `""` when it did, else the note to add to the result text.

    Never raises: the review has finished and its findings are in the text, so a results file that cannot be written (a full disk, a
    value JSON cannot hold) must not lose them. The note names the kind of failure and not the file: the calling model never handles it."""
    try:
        write_results(folder, results_of(state, review_type=review_type, run_id=recorder.run_id, review_id=review_id))
    except Exception as exc:
        print(f"codebase-reviewer: could not write the results file ({type(exc).__name__}: {exc})", file=sys.stderr)
        return f"\n\nNote: the structured results could not be saved ({type(exc).__name__}); the findings above are complete."
    return ""


def _run_review(review_type: str, values: dict) -> str:
    """One review of any type, run by the orchestrator with the rules policy (today's order for the type). The head
    commit is pinned for EVERY type, so the diff and every later read of a cited file see the same code even when `HEAD`
    or a branch moves during a run that takes minutes.

    With `findings_file` (the verifier and classifier types) the review starts from the findings in that file and only the type's
    own stage runs (`supply_findings`). With `REVIEW_OUTPUT_DIR` set, the finished review's structured results are saved there."""
    recorder = _recorder_for(review_type, values)

    def body() -> str:
        settings, config = _configured(recorder)
        # Both are checked before git or any model runs, so a results folder or a findings file that cannot be used refuses the
        # review at once, instead of after minutes of model time.
        folder = prepare_results_folder(settings.results_dir) if settings.results_dir else None
        supplied = read_findings_file(values["findings_file"], review_type) if "findings_file" in values else None
        head = _resolve_commit(values["repo_path"], values["head_ref"])
        diff = _git_diff(values["repo_path"], values["base_ref"], head, values["context_lines"])
        state = RunState(
            config=config,
            loader=ModelLoader(lmstudio_base_url(), observer=recorder),
            client=new_client(),
            repo_root=Path(os.path.expanduser(values["repo_path"])),
            head=head,
            diff=diff,
            max_reviewers=settings.max_reviewers,
            chunk_chars=values.get("chunk_chars"),
            # Only the classifier's requests need token probabilities, which the native API the other stages use lacks.
            probability_client=new_probability_client() if review_type == "classifier" else None,
            recorder=recorder,
        )
        if supplied is not None:
            supply_findings(state, supplied, review_type)
        text = run_orchestrated(state, RulesPolicy(review_type))
        if folder is not None:
            text += _save_results(folder, state, review_type, recorder, settings.review_id)
        return text

    return _recorded(recorder, body)


def _run_reasoning(values: dict) -> str:
    return _run_review("reasoning", values)


def _run_verifier(values: dict) -> str:
    return _run_review("verifier", values)


def _run_classifier(values: dict) -> str:
    return _run_review("classifier", values)


# The keys the reasoning, verifier and classifier types share: all start from the same diff.
DIFF_PARAMS = (
    Param("repo_path", str, "path of the git repository to review", required=True),
    Param("base_ref", str, "the base to diff against, e.g. origin/main", required=True),
    Param("head_ref", str, "the end of the range; the default is HEAD", default="HEAD"),
    Param("context_lines", int, "lines of surrounding code shown around each change (0-200)", default=20, minimum=0),
    Param("chunk_chars", int, "characters of diff per request; default is derived from each reviewer's contextLength (for a reviewer with tools it is cut to the room left after the lookups' reserve)", minimum=1),
    Param("pr", int, "the pull request number being reviewed, recorded with the run's timing so it can be joined to PR metrics", minimum=1),
)

# The one key only the verifier and classifier types take: the path of a findings file. The caller passes the path and never the
# findings (it would have to retype them, and the text a review returns leaves out the quote that grounding needs).
FINDINGS_FILE_FOR_VERIFIER = Param(
    "findings_file", str,
    "path of a findings file: only the verifier stage runs, on the findings in it and with no reasoning stage (grounding, then the "
    "verifiers); chunk_chars then has no effect",
)
FINDINGS_FILE_FOR_CLASSIFIER = Param(
    "findings_file", str,
    "path of a findings file of findings that already stand after verification, each with its verification status: only the "
    "classifier stage runs, on those, with no reasoning or verifier stage; chunk_chars then has no effect",
)

REVIEW_TYPES = {
    "reasoning": ReviewType(
        summary=(
            "Runs every active reasoning reviewer over the diff base_ref...head_ref and returns ALL their findings, "
            "grouped by class with duplicates merged. Findings are not verified or classified."
        ),
        params=DIFF_PARAMS,
        run=_run_reasoning,
    ),
    "verifier": ReviewType(
        summary=(
            "Runs the reasoning reviewers over the diff base_ref...head_ref, drops every finding whose quoted code is not in the "
            "file (ungrounded), then has every active verifier reviewer confirm each remaining finding with a concrete failing "
            "input or reject it. Returns the confirmed findings, the disputed ones (verifiers disagreed, with each one's "
            "reasoning), the unchecked ones, and one line each for what was dropped. With no verifier active the findings are "
            "grounded but not verified, and the result says so. With findings_file the reasoning stage does not run: the "
            "findings are read from that file and only grounding and verification run."
        ),
        params=DIFF_PARAMS + (FINDINGS_FILE_FOR_VERIFIER,),
        run=_run_verifier,
    ),
    "classifier": ReviewType(
        summary=(
            "Runs the verifier review (reasoning, grounding, verification), then has every active classifier reviewer check each "
            "finding that was not rejected against every rubric entry: does it fall under that entry's definition? The answer is the "
            "model's probability of yes. A finding takes a rubric entry's level only when the average probability over the classifiers "
            "is at or above the entry's score (0 to 1); "
            "a finding that fits several entries takes the highest level; one that matches no entry is dropped as unclassified. "
            "Returns the findings grouped by class with entry, level and average probability, the findings with no rubric level, "
            "and one line each for what was dropped. With no classifier active the findings carry no rubric level, and the result "
            "says so. With findings_file the reasoning and verifier stages do not run: the findings are read from that file, each "
            "with the status the verifiers gave it, and only classification runs."
        ),
        params=DIFF_PARAMS + (FINDINGS_FILE_FOR_CLASSIFIER,),
        run=_run_classifier,
    ),
}


def validate_params(review_type: ReviewType, raw: dict) -> tuple:
    """`(values, problems)` for a `params` dict against a type's keys.

    EVERY problem is reported, not the first: an unknown key, a missing
    required key, a wrong type (a bool is not a number), a value under its
    minimum. `values` has the defaults filled in and is only meaningful when
    there are no problems."""
    problems: list = []
    known = {p.name: p for p in review_type.params}
    for key in sorted(k for k in raw if k not in known):
        problems.append(f"unknown key {key!r} (allowed: {', '.join(known)})")
    values: dict = {}
    for param in review_type.params:
        if param.name not in raw:
            if param.required:
                problems.append(f"missing required key {param.name!r} ({param.kind.__name__}: {param.doc})")
            elif param.default is not None:
                values[param.name] = param.default
            continue
        value = raw[param.name]
        if isinstance(value, bool) or not isinstance(value, param.kind):
            problems.append(f"{param.name!r} must be a {param.kind.__name__} (got {value!r})")
        elif param.kind is str and not value.strip():
            problems.append(f"{param.name!r} must not be blank")
        elif param.kind is int and param.minimum is not None and value < param.minimum:
            problems.append(f"{param.name!r} must be {param.minimum} or more (got {value!r})")
        else:
            values[param.name] = value
    return values, problems


def review_help(name: str, review_type: ReviewType) -> str:
    """What a type takes: its summary, every key with its type, default and
    meaning, and an example call."""
    lines = [f"review type {name!r}: {review_type.summary}", "params:"]
    for p in review_type.params:
        need = "required" if p.required else ("default " + repr(p.default) if p.default is not None else "optional")
        lines.append(f"  - {p.name} ({p.kind.__name__}, {need}): {p.doc}")
    example = {p.name: ("..." if p.kind is str else 1) for p in review_type.params if p.required}
    lines.append(f'example: review(type="{name}", params={example!r})'.replace("'...'", '"/path/or/ref"'))
    return "\n".join(lines)


@mcp.tool()
def review(type: str, params: Optional[dict] = None) -> str:
    """
    Runs a local-model review of a change on LM Studio, entirely server-side:
    the diff and the files never enter your context, only the findings do.

    type: the kind of review. Known types:
      "reasoning": every active reasoning reviewer reads the diff
        base_ref...head_ref and returns ALL its findings, grouped by class with
        duplicates merged. The findings are NOT verified or classified.
      "verifier": the reasoning stage, then a check that each finding's quoted
        code is really in the file (a finding about code that is not there is
        dropped as ungrounded), then every active verifier reviewer confirms each
        remaining finding with a concrete failing input or rejects it. Returns
        what was confirmed, what was disputed (verifiers disagreed; each one's
        reasoning is shown), what was left unchecked, and one line each for what
        was dropped. Use it instead of "reasoning" when verifier reviewers are
        configured; with none active the findings are grounded but not verified,
        and the result says so. With findings_file the reasoning stage does not
        run: the findings are read from that file, and only grounding and
        verification run.
      "classifier": the verifier review, then every active classifier reviewer
        checks each finding that was not rejected against every entry of the
        user's rubric (does it fall under the entry's definition?), as the
        model's probability of yes. A finding takes an entry's level only when
        the average probability over the classifiers is at or above the entry's
        score (0 to 1); one that fits several entries takes the highest level;
        one that matches none is dropped as unclassified. Returns the findings
        grouped by class with the entry, level and average probability, and one
        line each for what was dropped. Use it instead of
        "verifier" when classifier reviewers are configured; with none active the
        findings carry no rubric level, and the result says so. With findings_file
        the reasoning and verifier stages do not run: the findings are read from
        that file, each with the status the verifiers gave it, and only
        classification runs.
    params: that type's arguments, as a dict.
      "reasoning", "verifier" and "classifier": repo_path (str, required), base_ref (str,
        required, e.g. "origin/main"), head_ref (str, default "HEAD"),
        context_lines (int, default 20: surrounding code shown around each
        change), chunk_chars (int, optional: characters of diff per request; the
        default comes from each reviewer's contextLength), pr (int, optional: the
        pull request number, recorded with the run's timing).
      "verifier" and "classifier" also take findings_file (str, optional): the PATH
        of a findings file, never the findings themselves. Without it the type runs
        every stage up to its own, as above. chunk_chars has no effect with it.

    Call it with an empty params to get the keys, defaults and an example for a
    type. An unknown type, unknown or missing keys and wrong types come back as
    one "Error:" listing every problem. Reviewers, models and rubric come from
    the REVIEWERS_FILE the user configured (see list_reviewers); a missing or
    invalid file is an "Error:" naming it and every problem. A review can take
    minutes on large diffs.
    """
    review_type = REVIEW_TYPES.get(type)
    if review_type is None:
        return f"Error: unknown review type {type!r}; known types: {', '.join(REVIEW_TYPES)}"
    raw = params if params is not None else {}
    if not isinstance(raw, dict):
        return f"Error: params must be an object (got {raw.__class__.__name__}); review(type={type!r}, params={{}}) shows its keys"
    if not raw and any(p.required for p in review_type.params):
        return review_help(type, review_type)
    values, problems = validate_params(review_type, raw)
    if problems:
        lines = [f"Error: invalid params for review type {type!r} ({len(problems)} problem{'s' if len(problems) != 1 else ''}):"]
        lines.extend(f"  - {p}" for p in problems)
        lines.append(f"call review(type={type!r}, params={{}}) to see what it takes")
        return "\n".join(lines)
    try:
        return review_type.run(values)
    except (ConfigError, ModelLoadError, ReviewError) as exc:
        return f"Error: {exc}"


@mcp.tool()
def review_report(view: str = "review_times", model: str = "", target_minutes: float = 0.0, this_machine: bool = False) -> str:
    """
    Shows what past reviews cost, from this server's own database of every run
    (nothing is sent to a model; it only reads what was recorded).

    view:
      "response_times": per model and role, the response time of one request
        (answered count, median, p90, max), how many timed out or failed, and the
        medians of latency, time to first token and time between tokens in
        milliseconds with the input and output token counts. Pass `model` to look
        at one model.
      "review_times": per model combination, the review wall time (median over its
        finished reviews) split into model loading and each stage.
      "under_target": the model combinations with a recorded review finished within
        `target_minutes` minutes, with how many of their reviews that is.
    this_machine: only reviews recorded on this machine, so results from different
      machines are never pooled.

    Answers "which combination suits this machine", "how long does model X take",
    and "which combination gets a review under N minutes". An unreadable database,
    a bad view, or no `target_minutes` for "under_target" comes back as an "Error:".
    """
    try:
        return reviewer_report(
            view, model=model or None, target_minutes=target_minutes if target_minutes > 0 else None, this_machine=this_machine
        )
    except ReportError as exc:
        return f"Error: {exc}"


def _describe_loaded_models(stages: list) -> str:
    try:
        loaded = ModelLoader(lmstudio_base_url()).loaded_model_ids()
    except ModelLoadError as exc:
        return f"Model check: could not ask LM Studio ({exc})"
    missing: list = []
    for stage in stages:
        for group in stage.resident:
            if group.model not in loaded and group.model not in missing:
                missing.append(group.model)
    if not missing:
        return "Model check: every model that must already be loaded is loaded."
    names = ", ".join(repr(m) for m in missing)
    return f"Model check: NOT loaded in LM Studio and not loadOnDemand: {names}. A review would refuse to run."


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
