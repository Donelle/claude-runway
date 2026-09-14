"""
Shared, dependency-free logic behind `tools/doctor.py` -- a validator that
diffs each dual-config `CLAUDE_RUNWAY_*` env var (the ones README's
"Environment variables" table marks "both") between a target repo's
`.mcp.json` `local-compress` env block and the live shell environment,
flagging any mismatch instead of relying on a human to notice silently
split behavior (issue #49, GROW-02 in `.plans/code-review-2026-08-18.md`).
Extends the same "loud instead of silent" idea `local_compress_lib.py`'s
`stale_env_warning()` already applies to the *renamed*-var case, to the
*dual-config-mismatch* case instead.

Deliberately does NOT import `local_compress_lib.py` (which pulls in the
`openai` package at module scope): a target repo running only the Qdrant
half of this toolkit (`--qdrant-only`, no local-compress at all) has every
reason to still want this doctor available, and shouldn't need `openai`
installed just to check `.mcp.json`/shell agreement. The handful of
default-resolution rules this needs (mirroring `local_compress_lib.py` for
URL/MODEL and `savings_ledger.py` for TRACK_SAVINGS/SAVINGS_DB) are small
enough to reproduce directly here instead -- the same "small, independent,
dependency-free lib" precedent `setup_project_lib.py` already follows.

The key semantic this whole module hinges on (see README's "Why two
places" section): a `CLAUDE_RUNWAY_*` key PRESENT in `.mcp.json`'s
local-compress `env` block -- even set to `""` -- always masks/overrides
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

DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"
_AUTO_DETECT_MODEL = "(auto-detect -- no model pinned)"

# Every CLAUDE_RUNWAY_* var README's env-var table marks "both" -- i.e.
# genuinely needs matching values in .mcp.json's local-compress env block
# AND the shell. Deliberately excluded from this list:
# - CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS: shell-only (compress_bash_output.py
#   is the sole reader); an .mcp.json copy is inert.
# - CLAUDE_RUNWAY_CACHE_DB: read by both codebase-indexer MCP server AND
#   redirect_webfetch_to_fetch_url.py hook (issue #64). Shell export is the
#   override channel (hooks have no env field in settings.json), not .mcp.json.
#   doctor.py only checks the local-compress env block; flagging a mismatch
#   for a var that's not a local-compress concern would be misleading.
# - CLAUDE_RUNWAY_WEBFETCH_FAILED_URL_TTL: hook-only; no .mcp.json counterpart.
DUAL_ENV_VARS = (
    "CLAUDE_RUNWAY_LMSTUDIO_URL",
    "CLAUDE_RUNWAY_LMSTUDIO_MODEL",
    "CLAUDE_RUNWAY_TRACK_SAVINGS",
    "CLAUDE_RUNWAY_SAVINGS_DB",
)


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
    Found in PR #119 review: this is the one spot the fake-`home_dir`
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


class MalformedMcpJsonError(Exception):
    """
    Raised when `.mcp.json` EXISTS and (for the JSON-parse step) parses,
    but its shape is invalid enough that this module can't safely tell
    whether `local-compress` is configured or not -- e.g. `"mcpServers":
    null` or a list, `"local-compress"` present but not an object
    (including an explicit `null` value, distinguished from a genuinely
    ABSENT key -- see `load_local_compress_env`), a non-object `env`
    block, a non-string value inside `env`, or the file itself being
    unreadable/not valid JSON at all.

    Deliberately distinct from returning `None` (a MISSING file, or valid
    JSON that simply omits `local-compress` -- e.g. a genuine
    `--qdrant-only` setup): both of those legitimately mean "nothing to
    check here," but a malformed/unreadable config is a real problem the
    caller must not silently treat the same way. Found in PR #119 review:
    the previous code returned `None` for a malformed shape it hadn't
    anticipated too, which conflated "nothing configured" (exit 0) with
    "config is broken" (should exit nonzero) -- and for the shapes it
    HADN'T anticipated at all (`"mcpServers": null` or a list), it instead
    raised an uncaught `AttributeError` from the blind `.get()` chain,
    which is strictly worse: an unhandled crash rather than either
    reportable outcome.
    """


def load_local_compress_env(mcp_json_path: Path) -> Optional[dict]:
    """
    Returns the `local-compress` server's `env` block from `mcp_json_path`,
    or `None` if the file doesn't exist, or exists and parses as valid
    JSON but simply doesn't configure a `local-compress` server (e.g. a
    `--qdrant-only` setup) -- either of which means there's nothing to
    dual-check here. Raises `MalformedMcpJsonError` (see its docstring) for
    every other bad shape, rather than silently treating it the same as
    "nothing configured," or crashing with an unhandled `AttributeError`
    on a `.get()` chain that assumed every level was already a dict.
    """
    if not mcp_json_path.is_file():
        return None
    try:
        with open(mcp_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise MalformedMcpJsonError(f"{mcp_json_path} could not be read as valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise MalformedMcpJsonError(f"{mcp_json_path}'s top level is a {type(data).__name__}, expected an object")

    mcp_servers = data.get("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        raise MalformedMcpJsonError(
            f"{mcp_json_path}'s \"mcpServers\" is a {type(mcp_servers).__name__}, expected an object"
        )

    if "local-compress" not in mcp_servers:
        return None  # valid config, simply no local-compress server configured
    # Checked by KEY MEMBERSHIP, not by whether the resulting value is
    # None -- .get("local-compress") would return None for BOTH a
    # genuinely absent key AND a key explicitly present with a JSON `null`
    # value, silently treating an explicit "local-compress": null as if
    # nothing were configured at all (exit 0) instead of the malformed
    # config it actually is. Found in PR #119 review.
    server = mcp_servers["local-compress"]
    if not isinstance(server, dict):
        raise MalformedMcpJsonError(
            f"{mcp_json_path}'s \"local-compress\" server is a {type(server).__name__}, expected an object"
        )

    env = server.get("env", {})
    if not isinstance(env, dict):
        raise MalformedMcpJsonError(
            f"{mcp_json_path}'s \"local-compress\".\"env\" is a {type(env).__name__}, expected an object"
        )
    # Every value must be a string -- real MCP env blocks (and real shell
    # environments) are string-valued by definition; a JSON `null`/number/
    # bool/list/dict snuck in there is malformed, not a valid config this
    # module can safely reason about. Found in PR #119 review: without
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
            f"{mcp_json_path}'s \"local-compress\".\"env\" has non-string value(s): {bad} -- "
            "MCP env blocks must be string-valued"
        )
    return env


def check_dual_env_vars(
    mcp_env: Mapping[str, str], shell_env: Mapping[str, str], *, home_dir: Path
) -> list:
    """
    Diffs every var in `DUAL_ENV_VARS` between `mcp_env` (the
    `local-compress` server's `.mcp.json` env block) and `shell_env`
    (typically `os.environ`), returning one `DualEnvMismatch` per var whose
    effective values disagree. A var absent from `mcp_env` is skipped
    entirely -- it inherits straight from `shell_env` (the exact same
    source this check compares against), so it can never actually diverge.
    """
    mismatches = []
    for var in DUAL_ENV_VARS:
        if var not in mcp_env:
            continue
        mcp_norm = _normalize(var, mcp_env[var], home_dir=home_dir)
        shell_norm = _normalize(var, shell_env.get(var), home_dir=home_dir)
        if mcp_norm != shell_norm:
            mismatches.append(DualEnvMismatch(var=var, mcp_json_value=mcp_norm, shell_value=shell_norm))
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
    config_error: Optional[str] = None


def run_doctor(target_repo: Path, *, shell_env: Mapping[str, str], home_dir: Optional[Path] = None) -> DoctorResult:
    """
    Runs the full dual-env-var check for `target_repo`'s `.mcp.json`
    against `shell_env`. `home_dir` defaults to `Path.home()`, overridable
    for tests so this stays a pure function.
    """
    resolved_home = home_dir if home_dir is not None else Path.home()
    mcp_json_path = Path(target_repo) / ".mcp.json"
    try:
        env = load_local_compress_env(mcp_json_path)
    except MalformedMcpJsonError as exc:
        return DoctorResult(mcp_json_path=mcp_json_path, local_compress_configured=False, config_error=str(exc))
    if env is None:
        return DoctorResult(mcp_json_path=mcp_json_path, local_compress_configured=False)
    mismatches = check_dual_env_vars(env, shell_env, home_dir=resolved_home)
    return DoctorResult(mcp_json_path=mcp_json_path, local_compress_configured=True, mismatches=mismatches)
