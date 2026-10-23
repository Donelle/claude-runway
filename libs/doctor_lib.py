"""
Shared, dependency-free logic behind `tools/doctor.py` -- a validator that
diffs each dual-config `CLAUDE_RUNWAY_*` env var (the ones
docs/environment-variables.md's table marks "both") between a target repo's
`.mcp.json` `local-compress` env block (and the `qdrant` block, which holds
the LM Studio values `compact_store` compresses with, and the `analytics`
block, which holds the savings DB override) and the live shell
environment, flagging any mismatch instead of relying on a human to notice silently
split behavior (GROW-02 in `.plans/code-review-2026-08-18.md`).
Extends the same "loud instead of silent" idea `local_compress_lib.py`'s
`stale_env_warning()` already applies to the *renamed*-var case, to the
*dual-config-mismatch* case instead.

Deliberately does NOT import `local_compress_lib.py` (which, although it
no longer imports `openai` itself, is the compression library and has no
business loading for a config check): a target repo running only the Qdrant
half of this toolkit (`--qdrant-only`, no local-compress at all) has every
reason to still want this doctor available, and shouldn't need the
compression stack just to check `.mcp.json`/shell agreement. The handful of
default-resolution rules this needs (mirroring `local_compress_lib.py` for
URL/MODEL and `savings_ledger.py` for TRACK_SAVINGS/SAVINGS_DB) are small
enough to reproduce directly here instead -- the same "small, independent,
dependency-free lib" precedent `setup_project_lib.py` already follows.

The key semantic this whole module hinges on (see
docs/environment-variables.md's "Why two places" section): a `CLAUDE_RUNWAY_*` key PRESENT in a checked server's
`env` block in `.mcp.json` -- even set to `""` -- always masks/overrides
whatever the shell would otherwise supply to that subprocess; a key ABSENT
from that block inherits straight from the shell, i.e. from the exact same
`os.environ` this tool already reads. So a var absent from the env block
never needs comparing at all (both sides necessarily agree, by
construction) -- only a PRESENT var can actually diverge, and how a blank
value there behaves differs per variable:
  - `CLAUDE_RUNWAY_LMSTUDIO_URL`: `local_compress_lib.py` resolves this via
    `os.environ.get(key, default)`, so present-but-blank is a literal blank
    value (a broken URL), NOT the default -- Python's `dict.get` only
    substitutes its default when the key is missing entirely, not when
    it's present-but-falsy.
  - `CLAUDE_RUNWAY_LMSTUDIO_MODEL`: no explicit default, but every caller
    treats a falsy value as "no model pinned, try to auto-detect" --
    present-but-blank behaves the same as absent.
  - `CLAUDE_RUNWAY_TRACK_SAVINGS`/`CLAUDE_RUNWAY_SAVINGS_DB`:
    `savings_ledger.py` resolves both via truthy `if override:`/
    `tracking_enabled()` checks, so present-but-blank is again the same as
    absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from codebase_reviewer_config_lib import ConfigError, ReviewersConfig, load_reviewers_file, read_settings
from setup_project_lib import (
    LEGACY_COMPRESS_HOOK_SCRIPT,
    MalformedConfigError,
    check_object_member,
    read_config_object,
)

DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"
_AUTO_DETECT_MODEL = "(auto-detect -- no model pinned)"

# Every CLAUDE_RUNWAY_* var docs/environment-variables.md's table marks "both" -- i.e.
# genuinely needs matching values in .mcp.json AND the shell; which server
# block each is compared in is SERVER_DUAL_ENV_VARS below. Deliberately excluded from this list:
# - CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS: shell-only (compress_output.py
#   is the sole reader); an .mcp.json copy is inert.
# - CLAUDE_RUNWAY_CACHE_DB: read by both codebase-indexer MCP server AND
#   redirect_webfetch_to_fetch_url.py hook. Shell export is the
#   override channel (hooks have no env field in settings.json), not .mcp.json.
#   No block doctor.py checks reads it; flagging a mismatch for it there
#   would be misleading.
# - CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL: hook-only; no .mcp.json counterpart.
# - CLAUDE_RUNWAY_LMSTUDIO_TIMEOUT_SECONDS: read by both halves via
#   libs/lmstudio_client.py's build_client(), but deliberately not declared in
#   the .mcp.json template: the MCP server inherits the shell value, and an
#   absent .mcp.json key vs. a shell export is not a real divergence. Same
#   shell-export-only channel as CLAUDE_RUNWAY_CACHE_DB above.
DUAL_ENV_VARS = (
    "CLAUDE_RUNWAY_LMSTUDIO_URL",
    "CLAUDE_RUNWAY_LMSTUDIO_MODEL",
    "CLAUDE_RUNWAY_TRACK_SAVINGS",
    "CLAUDE_RUNWAY_SAVINGS_DB",
)

# Which of DUAL_ENV_VARS each checked server actually READS, so a copy in a
# block whose server ignores it is never compared: comparing an
# inert copy would flag a harmless difference while missing the real one.
#   - local-compress: the LM Studio values, and TRACK_SAVINGS (it marks a
#     compressed result for the hook to record). Not SAVINGS_DB: it never
#     opens the savings database; `upgrade` removes a leftover copy.
#   - qdrant: the LM Studio values `compact_store` compresses with.
#   - analytics: SAVINGS_DB, the database its savings tools read, which must
#     be the same file the hooks write (from the shell), or the tracker
#     splits across two databases. Its TRACK_SAVINGS is not compared here:
#     `init`/`upgrade` always write it as "1" alongside a local-compress
#     block that carries the same setting, which is already checked above,
#     so comparing it again would only report one mismatch twice.
SERVER_DUAL_ENV_VARS = {
    "local-compress": (
        "CLAUDE_RUNWAY_LMSTUDIO_URL",
        "CLAUDE_RUNWAY_LMSTUDIO_MODEL",
        "CLAUDE_RUNWAY_TRACK_SAVINGS",
    ),
    "qdrant": ("CLAUDE_RUNWAY_LMSTUDIO_URL", "CLAUDE_RUNWAY_LMSTUDIO_MODEL"),
    "analytics": ("CLAUDE_RUNWAY_SAVINGS_DB",),
}


def _is_truthy_track_savings(value: Optional[str]) -> bool:
    """
    Mirrors `savings_ledger.tracking_enabled()`'s exact parsing (accepted
    values `1`/`true`/`yes`, case-insensitive, whitespace-trimmed) without
    importing that module -- see module docstring for why this is
    duplicated rather than shared.
    """
    return (value or "").strip().lower() in ("1", "true", "yes")


def _default_savings_db_path(home_dir: Path) -> str:
    """Mirrors `savings_ledger.resolve_db_path()`'s default location."""
    return str(home_dir / ".claude" / "claude-runway" / "savings.db")


def _normalize(var: str, raw: Optional[str], *, home_dir: Path) -> str:
    """
    Turns a raw env value from EITHER side (`None` meaning the key is
    genuinely absent from that source; `""` meaning present-but-blank) into
    a normalized string that's directly comparable between the two sides --
    applying each var's OWN real default-resolution rule, which is why this
    can't be one generic rule for all four (see module docstring).
    """
    if var == "CLAUDE_RUNWAY_LMSTUDIO_URL":
        # os.environ.get(key, default): substitutes the default ONLY when
        # the key is missing entirely -- present-but-blank stays blank.
        return raw if raw is not None else DEFAULT_LMSTUDIO_URL
    if var == "CLAUDE_RUNWAY_LMSTUDIO_MODEL":
        return raw if raw else _AUTO_DETECT_MODEL
    if var == "CLAUDE_RUNWAY_TRACK_SAVINGS":
        return "on" if _is_truthy_track_savings(raw) else "off"
    if var == "CLAUDE_RUNWAY_SAVINGS_DB":
        return _default_savings_db_path(home_dir) if not raw else str(_expand_and_anchor(raw, home_dir))
    raise ValueError(f"unknown dual env var: {var!r}")  # pragma: no cover -- DUAL_ENV_VARS is the only caller


def _expand_and_anchor(raw: str, home_dir: Path) -> Path:
    """
    Mirrors `savings_ledger.resolve_db_path()`'s handling of a real
    `CLAUDE_RUNWAY_SAVINGS_DB` override: expand a leading `~` and anchor an
    otherwise-relative path to the home directory -- so e.g. `~/savings.db`
    and `/home/user/savings.db` (the SAME real database) normalize to the
    same string here instead of doctor reporting a false mismatch.

    Deliberately does NOT call `Path.expanduser()` directly: that resolves
    `~` against the REAL os-level home directory, silently ignoring the
    `home_dir` parameter every other function in this module already
    accepts for testability (`run_doctor`'s callers, including tests, can
    inject a fake `home_dir` -- `expanduser()` would bypass that one case).
    Found in review: this is the one spot the fake-`home_dir`
    contract would have quietly broken without this manual expansion. Only
    handles the plain `~`/`~/...` forms (not `~otheruser/...`, which
    `expanduser()` resolves via a real system user-database lookup that
    has no equivalent injectable parameter here) -- sufficient for this
    tool's purpose, since a savings-db override naming another user's home
    directory isn't a realistic config to validate.
    """
    if raw == "~" or raw.startswith("~/"):
        raw = str(home_dir) + raw[1:]
    path = Path(raw)
    if not path.is_absolute():
        path = home_dir / path
    return path


@dataclass
class DualEnvMismatch:
    """
    One dual-config var whose EFFECTIVE value (already normalized/
    defaulted per `_normalize`, not the raw string) differs between
    `.mcp.json`'s local-compress env block and the live shell environment.
    """

    var: str
    mcp_json_value: str
    shell_value: str
    # Which `.mcp.json` server block held the value (`local-compress`, `qdrant`
    # or `analytics`).
    server: str = "local-compress"


# `.mcp.json` EXISTS but its shape is invalid enough that this module can't
# safely tell whether `local-compress` is configured or not -- e.g. `"mcpServers":
# null` or a list, `"local-compress"` present but not an object (including an
# explicit `null` value, distinguished from a genuinely ABSENT key -- see
# `load_server_env`), a non-object `env` block, a non-string value inside
# `env`, or the file itself being unreadable/not valid JSON at all.
#
# Deliberately distinct from returning `None` (a MISSING file, or valid JSON
# that simply omits `local-compress` -- e.g. a genuine `--qdrant-only` setup):
# both of those legitimately mean "nothing to check here," but a malformed/
# unreadable config is a real problem the caller must not silently treat the
# same way. Found in review: the previous code returned `None` for a
# malformed shape it hadn't anticipated too, which conflated "nothing
# configured" (exit 0) with "config is broken" (should exit nonzero) -- and
# for the shapes it HADN'T anticipated at all (`"mcpServers": null` or a
# list), it instead raised an uncaught `AttributeError` from the blind
# `.get()` chain, which is strictly worse.
#
# This is the same class `init`/`upgrade` raise for the same files
# (`setup_project_lib.MalformedConfigError`), so the three CLIs report a
# malformed config identically; the old name stays for existing callers.
MalformedMcpJsonError = MalformedConfigError


def load_local_compress_env(mcp_json_path: Path) -> Optional[dict]:
    """The `local-compress` server's `env` block; see `load_server_env`."""
    return load_server_env(mcp_json_path, "local-compress")


def load_server_env(mcp_json_path: Path, server_name: str) -> Optional[dict]:
    """
    Returns the `server_name` server's `env` block from `mcp_json_path`,
    or `None` if the file doesn't exist, or exists and parses as valid
    JSON but simply doesn't configure that server (e.g. no `local-compress`
    in a `--qdrant-only` setup) -- either of which means there's nothing to
    dual-check here. Raises `MalformedMcpJsonError` (see the comment above it) for
    every other bad shape, rather than silently treating it the same as
    "nothing configured," or crashing with an unhandled `AttributeError`
    on a `.get()` chain that assumed every level was already a dict.
    """
    if not mcp_json_path.is_file():
        return None
    # The parse and the top-level/`mcpServers` shape checks are the shared
    # ones `init`/`upgrade` use too, with the same messages.
    data = read_config_object(mcp_json_path)
    mcp_servers = check_object_member(mcp_json_path, data, "mcpServers")

    if server_name not in mcp_servers:
        return None  # valid config, simply no such server configured
    # Checked by KEY MEMBERSHIP, not by whether the resulting value is
    # None -- .get(server_name) would return None for BOTH a
    # genuinely absent key AND a key explicitly present with a JSON `null`
    # value, silently treating an explicit "local-compress": null as if
    # nothing were configured at all (exit 0) instead of the malformed
    # config it actually is. Found in review.
    server = mcp_servers[server_name]
    if not isinstance(server, dict):
        raise MalformedMcpJsonError(
            f"{mcp_json_path}'s \"{server_name}\" server is a {type(server).__name__}, expected an object"
        )

    env = server.get("env", {})
    if not isinstance(env, dict):
        raise MalformedMcpJsonError(
            f"{mcp_json_path}'s \"{server_name}\".\"env\" is a {type(env).__name__}, expected an object"
        )
    # Every value must be a string -- real MCP env blocks (and real shell
    # environments) are string-valued by definition; a JSON `null`/number/
    # bool/list/dict snuck in there is malformed, not a valid config this
    # module can safely reason about. Found in review: without
    # this, a non-string value either crashed downstream (e.g.
    # `CLAUDE_RUNWAY_TRACK_SAVINGS: 1` -- an int -- blew up inside
    # `_is_truthy_track_savings()`'s `.strip()` call) or was silently
    # misread as "absent" (e.g. `CLAUDE_RUNWAY_LMSTUDIO_URL: null` passed
    # `_normalize()`'s `raw is not None` check for "key missing," even
    # though the key was genuinely present -- just with a bad value --
    # producing a false-negative "OK" instead of a reportable error).
    non_string = {k: v for k, v in env.items() if not isinstance(v, str)}
    if non_string:
        bad = ", ".join(f"{k!r} ({type(v).__name__})" for k, v in non_string.items())
        raise MalformedMcpJsonError(
            f"{mcp_json_path}'s \"{server_name}\".\"env\" has non-string value(s): {bad} -- "
            "MCP env blocks must be string-valued"
        )
    return env


def check_dual_env_vars(
    mcp_env: Mapping[str, str], shell_env: Mapping[str, str], *, home_dir: Path, server: str = "local-compress"
) -> list:
    """
    Diffs every var `server` reads (`SERVER_DUAL_ENV_VARS[server]`; a server
    not listed there is a caller bug and raises `KeyError`) between `mcp_env`
    (the `server` block's `.mcp.json` env) and `shell_env`
    (typically `os.environ`), returning one `DualEnvMismatch` per var whose
    effective values disagree. A var absent from `mcp_env` is skipped
    entirely -- it inherits straight from `shell_env` (the exact same
    source this check compares against), so it can never actually diverge.
    """
    mismatches = []
    for var in SERVER_DUAL_ENV_VARS[server]:
        if var not in mcp_env:
            continue
        mcp_norm = _normalize(var, mcp_env[var], home_dir=home_dir)
        shell_norm = _normalize(var, shell_env.get(var), home_dir=home_dir)
        if mcp_norm != shell_norm:
            mismatches.append(
                DualEnvMismatch(var=var, mcp_json_value=mcp_norm, shell_value=shell_norm, server=server)
            )
    return mismatches


@dataclass
class DoctorResult:
    """
    `local_compress_configured=False` with `config_error=None` means
    `mcp_json_path` is simply missing, or parses fine but doesn't
    configure a `local-compress` server at all (e.g. a `--qdrant-only`
    setup) -- not an error, just nothing to check.

    `config_error` non-`None` means `.mcp.json` itself is malformed (see
    `MalformedMcpJsonError`) -- a real problem the caller (`tools/
    doctor.py`) must report and exit nonzero for, NOT the same "nothing to
    check" outcome as a valid config that simply omits `local-compress`.
    `local_compress_configured`/`mismatches` are meaningless in this case
    (left at their defaults) since the config couldn't actually be read.
    """

    mcp_json_path: Path
    local_compress_configured: bool
    mismatches: list = field(default_factory=list)
    # True when the `qdrant` block carries at least one dual-config var (its
    # LM Studio values), so there was something to check there even with no
    # local-compress server (a `--qdrant-only` setup).
    qdrant_checked: bool = False
    # True when the `analytics` block carries a `CLAUDE_RUNWAY_SAVINGS_DB`
    # override, the one server that reads that database.
    analytics_checked: bool = False
    config_error: Optional[str] = None
    # True when `.mcp.json` has a `codebase-reviewer` block, so its reviewers
    # file was checked. `reviewers_file_problem` is then the full message
    # (naming the file and every problem) when it is missing, unreadable or
    # invalid, or when a numeric setting is bad; None when all is well.
    reviewer_checked: bool = False
    reviewers_file_problem: Optional[str] = None
    reviewers_file_summary: str = ""
    # Hook events in .claude/settings.json still invoking the pre-rename
    # compress_bash_output.py (see find_legacy_compress_hook_events).
    legacy_compress_hook_events: list = field(default_factory=list)


def find_legacy_compress_hook_events(target_repo: Path) -> list:
    """
    Event names (e.g. "PostToolUse") in `target_repo`'s
    `.claude/settings.json` with a hook still invoking the old
    `compress_bash_output.py` (renamed to `compress_output.py` --
    the old path no longer exists, so that hook now errors on every call).
    Best-effort and read-only: a missing/unreadable/oddly-shaped settings
    file yields `[]` rather than an error, since this check is an extra on
    top of the .mcp.json check and `.claude/settings.json` is optional
    (`--skip-hooks`). `upgrade` is the fix; this only reports.
    """
    settings_path = Path(target_repo) / ".claude" / "settings.json"
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    hooks = data.get("hooks") if isinstance(data, dict) else None
    if not isinstance(hooks, dict):
        return []
    events = []
    for event, blocks in hooks.items():
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            inner = block.get("hooks") if isinstance(block, dict) else None
            for hook in inner if isinstance(inner, list) else []:
                args = hook.get("args") if isinstance(hook, dict) else None
                if isinstance(args, list) and any(
                    isinstance(a, str) and Path(a).name == LEGACY_COMPRESS_HOOK_SCRIPT for a in args
                ):
                    if event not in events:
                        events.append(event)
    return events


def run_doctor(target_repo: Path, *, shell_env: Mapping[str, str], home_dir: Optional[Path] = None) -> DoctorResult:
    """
    Runs the full dual-env-var check for `target_repo`'s `.mcp.json`
    against `shell_env`. `home_dir` defaults to `Path.home()`, overridable
    for tests so this stays a pure function.
    """
    resolved_home = home_dir if home_dir is not None else Path.home()
    mcp_json_path = Path(target_repo) / ".mcp.json"
    try:
        compress_env = load_server_env(mcp_json_path, "local-compress")
        qdrant_env = load_server_env(mcp_json_path, "qdrant")
        analytics_env = load_server_env(mcp_json_path, "analytics")
        reviewer_env = load_server_env(mcp_json_path, "codebase-reviewer")
    except MalformedMcpJsonError as exc:
        return DoctorResult(mcp_json_path=mcp_json_path, local_compress_configured=False, config_error=str(exc))
    legacy_events = find_legacy_compress_hook_events(Path(target_repo))
    mismatches = []
    if compress_env is not None:
        mismatches += check_dual_env_vars(compress_env, shell_env, home_dir=resolved_home, server="local-compress")
    # The qdrant block holds the LM Studio URL/model compact_store compresses
    # with, which the shell (and the hooks) must agree with just like the
    # local-compress block's copy. Its other vars are not dual-config ones.
    qdrant_checked = _carries_any(qdrant_env, "qdrant")
    if qdrant_env is not None:
        mismatches += check_dual_env_vars(qdrant_env, shell_env, home_dir=resolved_home, server="qdrant")
    # The analytics block holds the savings DB override its tools read; the
    # hooks write that database from the shell's value.
    analytics_checked = _carries_any(analytics_env, "analytics")
    if analytics_env is not None:
        mismatches += check_dual_env_vars(analytics_env, shell_env, home_dir=resolved_home, server="analytics")
    reviewer_problem, reviewer_summary = _check_reviewers_file(reviewer_env, shell_env, Path(target_repo))
    return DoctorResult(
        mcp_json_path=mcp_json_path,
        local_compress_configured=compress_env is not None,
        mismatches=mismatches,
        qdrant_checked=qdrant_checked,
        analytics_checked=analytics_checked,
        reviewer_checked=reviewer_env is not None,
        reviewers_file_problem=reviewer_problem,
        reviewers_file_summary=reviewer_summary,
        legacy_compress_hook_events=legacy_events,
    )


def _check_reviewers_file(
    reviewer_env: Optional[dict], shell_env: Mapping[str, str], target_repo: Path
) -> "tuple[Optional[str], str]":
    """
    For a project with a `codebase-reviewer` block, loads its reviewers file the
    way the server will and returns `(problem, summary)`: `problem` is None
    when the file is fine, else the message that names the file and what is
    wrong (the server's own, so the two never disagree); `summary` describes a
    good file. `(None, "")` when the project has no such block.

    The values the server sees are the shell environment with the block's `env`
    on top: a key PRESENT in the block wins even when blank (blank is then "not
    set", the required-setting error), a key absent from it inherits the shell.
    A relative path resolves against the project directory, where Claude Code
    launches the server from. Settings are checked too: a bad `MAX_REVIEWERS`
    makes the server refuse to run just as a bad file does.
    """
    if reviewer_env is None:
        return None, ""
    effective = {**shell_env, **reviewer_env}
    try:
        read_settings(effective)
        config: ReviewersConfig = load_reviewers_file(effective.get("REVIEWERS_FILE", ""), base_dir=target_repo)
    except ConfigError as exc:
        return str(exc), ""
    return None, f"{config.path} ({len(config.reviewers)} reviewer(s), {len(config.rubric)} rubric entr{'y' if len(config.rubric) == 1 else 'ies'})"


def _carries_any(env: Optional[dict], server: str) -> bool:
    """True when `env` (a server's env block, `None` if the server is not
    configured) holds at least one var `server` reads, i.e. there is
    something of that block to compare against the shell."""
    return env is not None and any(var in env for var in SERVER_DUAL_ENV_VARS[server])
