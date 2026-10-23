"""
Configuration for the `codebase-reviewer` MCP server: the environment settings
and the reviewers file (`REVIEWERS_FILE`).

Standard-library only, so `mcpservers/codebase_reviewer_mcp_server.py`,
`libs/doctor_lib.py` and the tests can all load it without the model client or
the MCP SDK.

Two rules shape everything here:

  - `REVIEWERS_FILE` is required and has NO default. The server cannot know
    which models a user has, so it never assumes a path, a model or a prompt. A
    missing, unreadable or invalid file is a `ConfigError` that names the path
    and what is wrong, and no review runs.
  - Validation reports EVERY problem in the file at once, not the first one.
    Fixing a reviewers file one error per round trip is what this avoids, so
    the validators below collect into a list and only raise at the end.

The reviewers file is JSON (MCP `env` values must be strings, so the structured
part cannot live in the env block):

    {
      "rubric": [
        {"name": "misleading-content", "level": "low",
         "definition": "a misleading message, comment or doc; nothing breaks.",
         "definitionFile": "optional/path/to/a/longer/definition.md",
         "score": 2}
      ],
      "reviewers": [
        {"name": "quoting", "model": "<served model id>", "prompt": "...",
         "active": true, "type": ["reasoning", "verifier"], "loadOnDemand": false},
        {"name": "classifier", "model": "<served model id>",
         "promptFile": "optional/path/to/a/prompt.md",
         "active": true, "type": ["classifier"],
         "loadOnDemand": true, "contextLength": 32768}
      ]
    }

A reviewer's prompt is `prompt` (text), `promptFile` (a path to a markdown file),
or both: one of the two is required, and when both are set they are merged into
one prompt, the `prompt` text first, then the file's text (`load_prompt`).

`level` is one of the server's own fixed levels, lowest to highest (`LEVELS`).
`score` is the minimum confidence, 1-5, a classifier needs for a finding to
match that entry (the classifier stage that uses it is a later piece of work;
this module only validates it). `contextLength` is used only when
`loadOnDemand` is true.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union

# The server's own fixed levels, lowest to highest. The user picks one per
# rubric entry; an entry with any other level is a configuration error.
LEVELS = ("low", "moderate", "high")

# The roles a reviewer can take, in the order their stages run: every reasoning
# reviewer first (they produce findings), then every verifier (each confirms or
# drops a finding), then every classifier (each matches a finding to a rubric
# entry). Each stage depends on the one before it.
ROLES = ("reasoning", "verifier", "classifier")

SCORE_MIN = 1
SCORE_MAX = 5

DEFAULT_MAX_REVIEWERS = 3
DEFAULT_MAX_REVIEW_ROUNDS = 3

_RUBRIC_KEYS = frozenset({"name", "level", "definition", "definitionFile", "score"})
_REVIEWER_KEYS = frozenset(
    {"name", "model", "prompt", "promptFile", "active", "type", "loadOnDemand", "contextLength"}
)
_TOP_LEVEL_KEYS = frozenset({"rubric", "reviewers"})


class ConfigError(Exception):
    """The reviewers file, or an environment setting, is unusable.

    `problems` holds every individual problem found (one line each). `str()` is
    one message that names the file and lists all of them, which is what the
    server and `doctor` show."""

    def __init__(self, message: str, problems: Optional[list] = None):
        super().__init__(message)
        self.problems = list(problems) if problems else [message]


@dataclass(frozen=True)
class RubricEntry:
    name: str
    level: str
    definition: str
    score: int
    # Resolved path of the optional free-text definition file; "" when absent.
    # Validated as readable at load time, read by the classifier stage.
    definition_file: str = ""


@dataclass(frozen=True)
class Reviewer:
    name: str
    model: str
    # The inline `prompt` text; "" when the reviewer uses only `promptFile`.
    prompt: str
    active: bool
    roles: tuple
    load_on_demand: bool
    # Used only when the server loads the model itself (load_on_demand).
    context_length: Optional[int] = None
    # Resolved path of the optional `promptFile`; "" when absent. Validated as
    # readable and non-blank at load time, read again by `load_prompt` when a
    # review runs, so an edit to the file applies without a restart.
    prompt_file: str = ""


@dataclass(frozen=True)
class ReviewersConfig:
    rubric: tuple = ()
    reviewers: tuple = ()
    # The file this was read from ("" when built from data directly).
    path: str = ""

    def active_reviewers(self, role: Optional[str] = None) -> list:
        """Active reviewers, in file order, optionally only those with `role`."""
        return [r for r in self.reviewers if r.active and (role is None or role in r.roles)]


@dataclass(frozen=True)
class ReviewerSettings:
    """The environment settings (everything except the reviewers file's body)."""

    reviewers_file: str
    max_reviewers: int = DEFAULT_MAX_REVIEWERS
    max_review_rounds: int = DEFAULT_MAX_REVIEW_ROUNDS
    qdrant_url: str = ""
    qdrant_api_key: str = ""


# ---------------------------------------------------------------------------
# Environment settings
# ---------------------------------------------------------------------------


def _parse_positive_int(raw: str, name: str, problems: list) -> Optional[int]:
    text = raw.strip()
    try:
        value = int(text)
    except ValueError:
        problems.append(f"{name} must be a whole number of 1 or more (got {raw!r})")
        return None
    if value < 1:
        problems.append(f"{name} must be a whole number of 1 or more (got {raw!r})")
        return None
    return value


def read_settings(env: Mapping[str, str]) -> ReviewerSettings:
    """Reads the server's settings from `env` (normally `os.environ`).

    Does NOT require `REVIEWERS_FILE`: whether it is set is checked where the
    file is loaded (`load_reviewers_file`), so a caller that only wants the
    numeric settings, or `doctor` reporting a blank value, can still read the
    rest. `MAX_REVIEWERS` and `MAX_REVIEW_ROUNDS` default when unset or blank;
    a present value that is not a positive whole number is an error rather than
    a silent fallback, since "3" misspelt as "three" quietly running 3 would
    hide the mistake. Every bad setting is reported together."""
    problems: list = []
    max_reviewers = DEFAULT_MAX_REVIEWERS
    raw = env.get("MAX_REVIEWERS", "")
    if raw.strip():
        parsed = _parse_positive_int(raw, "MAX_REVIEWERS", problems)
        if parsed is not None:
            max_reviewers = parsed
    max_rounds = DEFAULT_MAX_REVIEW_ROUNDS
    raw = env.get("MAX_REVIEW_ROUNDS", "")
    if raw.strip():
        parsed = _parse_positive_int(raw, "MAX_REVIEW_ROUNDS", problems)
        if parsed is not None:
            max_rounds = parsed
    if problems:
        raise ConfigError("Invalid codebase-reviewer settings: " + "; ".join(problems), problems)
    return ReviewerSettings(
        reviewers_file=env.get("REVIEWERS_FILE", ""),
        max_reviewers=max_reviewers,
        max_review_rounds=max_rounds,
        qdrant_url=env.get("QDRANT_URL", ""),
        qdrant_api_key=env.get("QDRANT_API_KEY", ""),
    )


# ---------------------------------------------------------------------------
# The reviewers file
# ---------------------------------------------------------------------------


def _int_in_range(value: object, low: int, high: Optional[int] = None) -> Optional[int]:
    """`value` if it is a whole number in range, else None. A bool is not a
    number here: `true` is neither a score nor a context length."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < low or (high is not None and value > high):
        return None
    return value


def _required_string(raw: dict, key: str, where: str, problems: list) -> Optional[str]:
    """`raw[key]` if it is a non-blank string, else records a problem and returns None."""
    value = raw.get(key)
    if isinstance(value, str) and value.strip():
        return value
    problems.append(f"{where}: '{key}' is required and must be a non-empty string")
    return None


def _required_bool(raw: dict, key: str, where: str, problems: list) -> Optional[bool]:
    value = raw.get(key)
    if isinstance(value, bool):
        return value
    problems.append(f"{where}: '{key}' is required and must be true or false (got {value!r})")
    return None


def _unknown_keys(entry: dict, allowed: frozenset, where: str, problems: list) -> None:
    for key in sorted(k for k in entry if k not in allowed):
        problems.append(f"{where}: unknown key {key!r} (allowed: {', '.join(sorted(allowed))})")


def _validate_rubric(rubric: object, base_dir: Path, problems: list) -> list:
    entries: list = []
    if not isinstance(rubric, list):
        problems.append(f"'rubric' must be a list (got {type(rubric).__name__})")
        return entries
    seen: dict = {}
    for index, raw in enumerate(rubric):
        where = f"rubric[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object (got {type(raw).__name__})")
            continue
        _unknown_keys(raw, _RUBRIC_KEYS, where, problems)
        name = _required_string(raw, "name", where, problems)
        if name is not None:
            where = f"rubric[{index}] ({name!r})"
            if name in seen:
                problems.append(f"{where}: duplicate rubric name {name!r} (also rubric[{seen[name]}])")
                name = None
            else:
                seen[name] = index
        level = raw.get("level")
        if level not in LEVELS:
            problems.append(f"{where}: 'level' must be one of {', '.join(LEVELS)} (got {level!r})")
        definition = _required_string(raw, "definition", where, problems)
        score = _int_in_range(raw.get("score"), SCORE_MIN, SCORE_MAX)
        if score is None:
            problems.append(
                f"{where}: 'score' must be a whole number from {SCORE_MIN} to {SCORE_MAX} (got {raw.get('score')!r})"
            )
        definition_file: Optional[str] = ""
        if "definitionFile" in raw:
            definition_file = _readable_definition_file(raw["definitionFile"], base_dir, where, problems)
        if (
            name is not None
            and level in LEVELS
            and definition is not None
            and score is not None
            and definition_file is not None
        ):
            entries.append(
                RubricEntry(
                    name=name, level=level, definition=definition, score=score, definition_file=definition_file
                )
            )
    return entries


def _readable_definition_file(raw_path: object, base_dir: Path, where: str, problems: list) -> Optional[str]:
    """The resolved path of a readable `definitionFile`, or None with a problem recorded."""
    return _readable_file(raw_path, base_dir, "definitionFile", where, problems)


def _readable_file(
    raw_path: object, base_dir: Path, key: str, where: str, problems: list, *, require_text: bool = False
) -> Optional[str]:
    """The resolved path of a readable file named by `key`, or None with a problem recorded.

    `require_text` also rejects a file with nothing but whitespace in it (a
    `promptFile` like that would give a reviewer an empty prompt)."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        problems.append(f"{where}: '{key}' must be a non-empty path string (got {raw_path!r})")
        return None
    resolved = _resolve_path(raw_path, base_dir)
    try:
        # Opened, not just stat'ed: "exists" is not "readable".
        with open(resolved, "r", encoding="utf-8") as f:
            text = f.read() if require_text else f.read(1)
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(f"{where}: '{key}' {str(resolved)!r} cannot be read ({_describe(exc)})")
        return None
    if require_text and not text.strip():
        problems.append(f"{where}: '{key}' {str(resolved)!r} is empty")
        return None
    return str(resolved)


def _validate_roles(raw: dict, where: str, problems: list) -> Optional[tuple]:
    roles_raw = raw.get("type")
    if not isinstance(roles_raw, list) or not roles_raw:
        problems.append(
            f"{where}: 'type' is required and must be a non-empty list of {', '.join(ROLES)} (got {roles_raw!r})"
        )
        return None
    bad = [r for r in roles_raw if r not in ROLES]
    if bad:
        problems.append(f"{where}: 'type' has unknown role(s) {bad!r}; allowed: {', '.join(ROLES)}")
        return None
    if len(set(roles_raw)) != len(roles_raw):
        problems.append(f"{where}: 'type' lists a role more than once ({roles_raw!r})")
        return None
    # Stage order, not the order the user happened to write.
    return tuple(r for r in ROLES if r in roles_raw)


def _validate_prompt_sources(raw: dict, base_dir: Path, where: str, problems: list) -> Optional[tuple]:
    """`(prompt, prompt_file)` for one reviewer, or None with problems recorded.

    One of `prompt` / `promptFile` is required; either may be given, and both
    together are merged by `load_prompt`. A key that is present must be valid
    even when the other one is fine."""
    if "prompt" not in raw and "promptFile" not in raw:
        problems.append(f"{where}: one of 'prompt' or 'promptFile' is required")
        return None
    ok = True
    prompt = ""
    if "prompt" in raw:
        value = raw["prompt"]
        if isinstance(value, str) and value.strip():
            prompt = value
        else:
            problems.append(f"{where}: 'prompt' is required and must be a non-empty string when given (got {value!r})")
            ok = False
    prompt_file: Optional[str] = ""
    if "promptFile" in raw:
        prompt_file = _readable_file(raw["promptFile"], base_dir, "promptFile", where, problems, require_text=True)
        if prompt_file is None:
            ok = False
    return (prompt, prompt_file or "") if ok else None


def _validate_reviewers(reviewers: object, base_dir: Path, problems: list) -> list:
    entries: list = []
    if not isinstance(reviewers, list):
        problems.append(f"'reviewers' must be a list (got {type(reviewers).__name__})")
        return entries
    seen: dict = {}
    for index, raw in enumerate(reviewers):
        where = f"reviewers[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object (got {type(raw).__name__})")
            continue
        _unknown_keys(raw, _REVIEWER_KEYS, where, problems)
        name = _required_string(raw, "name", where, problems)
        if name is not None:
            where = f"reviewers[{index}] ({name!r})"
            if name in seen:
                problems.append(f"{where}: duplicate reviewer name {name!r} (also reviewers[{seen[name]}])")
                name = None
            else:
                seen[name] = index
        model = _required_string(raw, "model", where, problems)
        prompt_sources = _validate_prompt_sources(raw, base_dir, where, problems)
        active = _required_bool(raw, "active", where, problems)
        roles = _validate_roles(raw, where, problems)
        load_on_demand = _required_bool(raw, "loadOnDemand", where, problems)
        context_length: Optional[int] = None
        context_ok = True
        if "contextLength" in raw:
            context_length = _int_in_range(raw["contextLength"], 1)
            if context_length is None:
                problems.append(
                    f"{where}: 'contextLength' must be a whole number of 1 or more (got {raw['contextLength']!r})"
                )
                context_ok = False
        if (
            name is not None
            and model is not None
            and prompt_sources is not None
            and active is not None
            and roles is not None
            and load_on_demand is not None
            and context_ok
        ):
            entries.append(
                Reviewer(
                    name=name,
                    model=model.strip(),
                    prompt=prompt_sources[0],
                    prompt_file=prompt_sources[1],
                    active=active,
                    roles=roles,
                    load_on_demand=load_on_demand,
                    context_length=context_length,
                )
            )
    return entries


def _check_cross_references(entries: list, rubric: list, problems: list) -> None:
    """Problems that need more than one entry to see."""
    # Reviewers sharing a model are scheduled as one group and the model is
    # loaded once, so they have to agree on HOW it is loaded. Two reviewers
    # asking for the same model with different loadOnDemand/contextLength has no
    # single right answer, so it is a configuration error, not a tie-break.
    by_model: dict = {}
    for reviewer in entries:
        by_model.setdefault(reviewer.model, []).append(reviewer)
    for model, group in by_model.items():
        loads = {r.load_on_demand for r in group}
        # contextLength only matters when the server loads the model itself.
        contexts = {r.context_length for r in group if r.load_on_demand}
        if len(loads) > 1:
            names = ", ".join(repr(r.name) for r in group)
            problems.append(
                f"reviewers {names} share model {model!r} but disagree on 'loadOnDemand'; "
                "reviewers sharing a model must agree"
            )
        elif len(contexts) > 1:
            names = ", ".join(repr(r.name) for r in group)
            problems.append(
                f"reviewers {names} share model {model!r} but disagree on 'contextLength'; "
                "reviewers sharing a model must agree"
            )
    if not rubric:
        for reviewer in entries:
            if reviewer.active and "classifier" in reviewer.roles:
                problems.append(
                    f"reviewer {reviewer.name!r} is an active classifier but 'rubric' has no entries to classify against"
                )


def _describe(exc: BaseException) -> str:
    return getattr(exc, "strerror", None) or str(exc) or type(exc).__name__


def _resolve_path(raw: str, base_dir: Path) -> Path:
    expanded = Path(os.path.expanduser(raw.strip()))
    return expanded if expanded.is_absolute() else base_dir / expanded


def validate_reviewers(data: object, *, base_dir: Union[str, Path, None] = None, path: str = "") -> ReviewersConfig:
    """Validates parsed reviewers-file content and returns the config.

    Raises `ConfigError` listing EVERY problem found, not the first. `base_dir`
    is what a relative `definitionFile` or `promptFile` resolves against (the reviewers file's
    own directory when called through `load_reviewers_file`; the current
    directory when omitted). `path` is only used in the error message."""
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    problems: list = []
    rubric: list = []
    reviewers: list = []
    if not isinstance(data, dict):
        problems.append(f"the top level must be a JSON object with 'rubric' and 'reviewers' (got {type(data).__name__})")
    else:
        for key in sorted(k for k in data if k not in _TOP_LEVEL_KEYS):
            problems.append(f"unknown top-level key {key!r} (allowed: rubric, reviewers)")
        for required in ("rubric", "reviewers"):
            if required not in data:
                problems.append(f"missing required key {required!r}")
        if "rubric" in data:
            rubric = _validate_rubric(data["rubric"], base, problems)
        if "reviewers" in data:
            reviewers = _validate_reviewers(data["reviewers"], base, problems)
        _check_cross_references(reviewers, rubric, problems)
    if problems:
        raise ConfigError(_format_problems(path, "is invalid", problems), problems)
    return ReviewersConfig(rubric=tuple(rubric), reviewers=tuple(reviewers), path=path)


def _format_problems(path: str, what: str, problems: list) -> str:
    head = f"Reviewers file {path!r} {what}" if path else f"Reviewers configuration {what}"
    count = len(problems)
    lines = [f"{head} ({count} problem{'s' if count != 1 else ''}):"]
    lines.extend(f"  - {p}" for p in problems)
    return "\n".join(lines)


def load_reviewers_file(path: Optional[str], *, base_dir: Union[str, Path, None] = None) -> ReviewersConfig:
    """Reads and validates the reviewers file at `path` (the `REVIEWERS_FILE` value).

    `path` is required: None or blank is a `ConfigError` saying so, because the
    server has no default and never guesses one. A leading `~` is expanded; a
    relative path resolves against `base_dir` (the server's working directory
    when omitted; `doctor` passes the project directory). Missing, not a
    regular file, unreadable, not UTF-8 text, and not valid JSON are each a
    `ConfigError` that names the path and says which; content problems are all
    reported together by `validate_reviewers`."""
    if path is None or not path.strip():
        message = (
            "REVIEWERS_FILE is not set. The codebase-reviewer server requires it and has no default: "
            "set it to the path of your reviewers JSON file in the server's env block."
        )
        raise ConfigError(message)
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    resolved = _resolve_path(path, base)
    shown = str(resolved)
    try:
        if not resolved.exists():
            raise ConfigError(_format_problems(shown, "does not exist", ["no file at that path"]), ["no file at that path"])
        if not resolved.is_file():
            raise ConfigError(_format_problems(shown, "is not a file", ["that path is not a regular file"]), ["that path is not a regular file"])
        text = resolved.read_text(encoding="utf-8")
    except ConfigError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        problem = f"cannot be read ({_describe(exc)})"
        raise ConfigError(_format_problems(shown, "cannot be read", [problem]), [problem]) from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        problem = f"not valid JSON ({exc})"
        raise ConfigError(_format_problems(shown, "is not valid JSON", [problem]), [problem]) from exc
    return validate_reviewers(data, base_dir=resolved.parent, path=shown)


# ---------------------------------------------------------------------------
# A short, human-readable view (used by the server tool and doctor)
# ---------------------------------------------------------------------------


def _prompt_source(reviewer: Reviewer) -> str:
    if reviewer.prompt and reviewer.prompt_file:
        return f"inline text + {reviewer.prompt_file}"
    return reviewer.prompt_file or "inline text"


def load_prompt(reviewer: Reviewer) -> str:
    """The reviewer's full prompt: `prompt` text and/or `promptFile` content.

    When both are set they are merged into one prompt, the inline text first
    and the file's text after a blank line. The file is read here, at run
    time, not cached from load time, so editing it takes effect on the next
    review. A file that has become unreadable or blank since the reviewers file
    was validated is a `ConfigError` naming the reviewer and the path."""
    parts = []
    if reviewer.prompt.strip():
        parts.append(reviewer.prompt.strip())
    if reviewer.prompt_file:
        try:
            text = Path(reviewer.prompt_file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(
                f"reviewer {reviewer.name!r}: 'promptFile' {reviewer.prompt_file!r} cannot be read ({_describe(exc)})"
            ) from exc
        if text.strip():
            parts.append(text.strip())
    if not parts:
        raise ConfigError(f"reviewer {reviewer.name!r} has no prompt text (is its 'promptFile' empty?)")
    return "\n\n".join(parts)


def advisories(config: ReviewersConfig) -> list:
    """Plain-language notes on how the active REASONING reviewers divide the work. Never an error:
    a setup that triggers one is valid, and what to do about it is the user's call.

    How well a small model reviews depends mostly on how much it is asked to cover at once, and that
    is set by the reviewers' prompts, which the user owns. So nothing here changes what runs; it only
    says what the current setup implies:
    - a single reasoning reviewer has to cover every kind of defect by itself;
    - two reviewers on the same model with the same prompt (and so, at temperature 0, the same
      answers) only repeat each other."""
    reasoning = config.active_reviewers("reasoning")
    notes: list = []
    if len(reasoning) == 1:
        notes.append(
            f"Only one reasoning reviewer is active ({reasoning[0].name!r}), so it must cover every kind of defect "
            "on its own: give it a broad prompt, because a narrow one leaves the rest of the code unchecked. "
            "Small models usually do better on one narrow question at a time: to split the work, add reviewers "
            "on the same model with narrower prompts (reviewers that share a model load it once)."
        )
    by_key: dict = {}
    for reviewer in reasoning:
        try:
            text = " ".join(load_prompt(reviewer).split())
        except (OSError, ConfigError):
            # A promptFile that vanished or became unreadable after validation is reported where it is
            # used (load_prompt wraps the failure in ConfigError); a note about prompts must not raise.
            continue
        by_key.setdefault((reviewer.model, text), []).append(reviewer.name)
    for (model, _text), names in by_key.items():
        if len(names) > 1:
            notes.append(
                f"Reviewers {', '.join(repr(n) for n in names)} use the same model ({model!r}) and the same prompt, so they "
                "will raise the same findings: give each its own lens, or drop all but one."
            )
    return notes


def describe_config(config: ReviewersConfig, settings: Optional[ReviewerSettings] = None) -> str:
    """One text summary of a valid config: settings, rubric and reviewers."""
    lines = []
    if settings is not None:
        lines.append(
            f"Settings: MAX_REVIEWERS={settings.max_reviewers}, MAX_REVIEW_ROUNDS={settings.max_review_rounds}"
        )
    if config.path:
        lines.append(f"Reviewers file: {config.path}")
    lines.append(f"Rubric ({len(config.rubric)}):")
    for entry in config.rubric:
        lines.append(f"  - {entry.name} [{entry.level}, min confidence {entry.score}]")
    lines.append(f"Reviewers ({len(config.reviewers)}):")
    for r in config.reviewers:
        state = "active" if r.active else "inactive"
        load = "loaded on demand" + (f", context {r.context_length}" if r.context_length else "") if r.load_on_demand else "must already be loaded"
        lines.append(f"  - {r.name}: {state}, {'/'.join(r.roles)}, model {r.model} ({load}); prompt: {_prompt_source(r)}")
    return "\n".join(lines)


__all__ = [
    "LEVELS",
    "ROLES",
    "ConfigError",
    "RubricEntry",
    "Reviewer",
    "ReviewersConfig",
    "ReviewerSettings",
    "read_settings",
    "validate_reviewers",
    "load_reviewers_file",
    "load_prompt",
    "describe_config",
    "advisories",
]
