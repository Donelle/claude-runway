#!/usr/bin/env python3
"""
MCP server `codebase-reviewer`: a local-model review loop that runs on LM Studio
before (and between) Copilot reviews, so the findings Claude pays to read have
already been checked.

What exists today: configuration, the reviewers file, model loading and staged
scheduling, and two tools. `list_reviewers` validates the configuration and shows
what would run. `review(type, params)` runs a review; the one type so far is
"reasoning" (the reasoning reviewers over a git diff). The verifier and
classifier stages are separate, later pieces of work that add their own types to
the same tool, so its schema stays one entry however many stages exist.

It is standalone, laid out like `compress_mcp_server.py` and imports nothing
from any other server, so it can be installed or left out on its own. It shares
only `libs/`: `lmstudio_client.py` (bounded timeouts, no retries) for the model
client, and the three `libs/` modules named below.

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

Model loading is in libs/lmstudio_model_loader_lib.py, scheduling in
libs/reviewer_scheduler_lib.py.

Project minimum: Python 3.11.

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

import local_compress_lib
from codebase_reviewer_config_lib import ConfigError, advisories, describe_config, load_reviewers_file, read_settings
from lmstudio_client import LMStudioClient, build_client
from lmstudio_model_loader_lib import ModelLoader, ModelLoadError
from reasoning_reviewer_lib import ReviewError, format_review, run_reasoning_review
from reviewer_scheduler_lib import describe_plan, plan_stages

mcp = MCPServer("codebase-reviewer")


def lmstudio_base_url() -> str:
    """LM Studio's OpenAI-compatible URL: `CLAUDE_RUNWAY_LMSTUDIO_URL`, else the library default.
    Read per call (not at import) so a test or a restarted env is honoured."""
    return os.environ.get("CLAUDE_RUNWAY_LMSTUDIO_URL", local_compress_lib.DEFAULT_BASE_URL)


def new_client() -> LMStudioClient:
    """A bounded LM Studio client (finite timeout, no retries) for the reviewers
    the later pieces of work add; the same construction the other servers use."""
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
    command = ["git", "-C", str(root), "diff", "--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/",
               f"--unified={context_lines}", f"{base_ref}...{head_ref}", "--"]
    try:
        done = subprocess.run(command, capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReviewError(f"git diff could not run in {str(root)!r} ({type(exc).__name__}: {exc})") from exc
    if done.returncode != 0:
        message = done.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise ReviewError(f"git diff {base_ref}...{head_ref} failed in {str(root)!r}: {message[0] if message else 'no message'}")
    return done.stdout.decode("utf-8", errors="replace")


def _run_reasoning(values: dict) -> str:
    settings = read_settings(os.environ)
    config = load_reviewers_file(settings.reviewers_file)
    diff = _git_diff(values["repo_path"], values["base_ref"], values["head_ref"], values["context_lines"])
    review_result = run_reasoning_review(
        config,
        ModelLoader(lmstudio_base_url()),
        new_client(),
        diff,
        Path(os.path.expanduser(values["repo_path"])),
        settings.max_reviewers,
        chunk_chars=values.get("chunk_chars"),
    )
    return format_review(review_result)


REVIEW_TYPES = {
    "reasoning": ReviewType(
        summary=(
            "Runs every active reasoning reviewer over the diff base_ref...head_ref and returns ALL their findings, "
            "grouped by class with duplicates merged. Findings are not verified or classified."
        ),
        params=(
            Param("repo_path", str, "path of the git repository to review", required=True),
            Param("base_ref", str, "the base to diff against, e.g. origin/main", required=True),
            Param("head_ref", str, "the end of the range; the default is HEAD", default="HEAD"),
            Param("context_lines", int, "lines of surrounding code shown around each change (0-200)", default=20, minimum=0),
            Param("chunk_chars", int, "characters of diff per request; default is derived from each reviewer's contextLength", minimum=1),
        ),
        run=_run_reasoning,
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
        duplicates merged. The findings are NOT verified or classified (the
        verifier and classifier stages are separate types that arrive later).
    params: that type's arguments, as a dict.
      "reasoning": repo_path (str, required), base_ref (str, required, e.g.
        "origin/main"), head_ref (str, default "HEAD"), context_lines (int,
        default 20: surrounding code shown around each change), chunk_chars
        (int, optional: characters of diff per request; the default comes from
        each reviewer's contextLength).

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
