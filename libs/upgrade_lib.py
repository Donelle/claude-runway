"""
Shared logic behind `tools/setup_project.py upgrade` (issue #225) -- a config
MIGRATION tool for an already-configured project, distinct from `init`
(`libs/setup_project_lib.py`'s `run_setup`). `init` is safe to re-run in the
sense that it never clobbers UNRELATED config, but it still resets every
toolkit-owned setting to whatever flags this run's command line passes (or
their defaults) -- a user who forgets `--compact-collection`, or who just
wants to pick up ONE new hook/env var a recent release added, has to
remember and re-specify every option they originally set. `upgrade` instead
compares the target's CURRENT `.mcp.json`/`.claude/settings.json` against
specific, NAMED gaps this toolkit already knows how to fill, tells the user
about each one individually, and applies only the ones they approve --
surgically, with no effect on anything else.

Each `Migration.detect` is the source of truth for whether it still needs
applying -- not its position in `MIGRATIONS`. Running `upgrade` twice is
always safe: after the first apply, `detect` returns False and the migration
silently drops off the pending list on the second run.

Deliberately NOT reusing `setup_project_lib.merge_mcp_json`/
`merge_settings_hooks` for the migrations below, even though the issue this
module implements suggested doing so for the hooks migration specifically --
see `_apply_record_session_id_hooks_missing`'s own docstring for why: those
two functions are RESYNC primitives (built for `init`, which regenerates and
re-merges every toolkit-owned field on every run) and their "strip every
toolkit-owned entry for this event, then re-add only what this run
generated" semantics silently DELETE an unrelated toolkit hook (e.g. an
existing `compress_bash_output.py` PostToolUse block) that a narrower,
single-migration apply never intended to touch. A migration's `apply` only
ever ADDS the one thing its `detect` found missing -- confirmed by writing
the naive merge-based version first and catching it deleting a pre-existing
compress hook in a test before shipping this.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Flat import, matching every other libs/*.py module's own convention (see
# setup_project_lib.py's module docstring for why: libs/ is added to
# sys.path directly by callers, not treated as a real Python package).
from setup_project_lib import build_mcp_servers, build_settings_hooks, load_json


@dataclass
class MigrationContext:
    """The two values every migration's `apply` needs to resolve a fresh
    toolkit-owned path -- the SAME two values `cmd_init` already computes
    (`TOOLS_REPO_DIR`, `_pip_installed_venv_python()` or its pure fallback)
    and passes into `run_setup`, so `cmd_upgrade` needs no new machinery to
    obtain them."""

    tools_repo_dir: Path
    venv_python: Path


@dataclass
class Migration:
    """
    One named, independently-detectable config gap and its fix.

    `id` must be stable and never reused -- it's the only thing a caller (a
    human reading `--dry-run` output, a test) has to refer to a specific
    migration by, across every version of this file.

    `detect(mcp_json, settings_json) -> bool`: True when this migration
    still needs applying. Pure -- no disk access, no mutation of its
    arguments.

    `apply(mcp_json, settings_json, ctx) -> (mcp_json, settings_json)`:
    returns NEW dicts (never mutates the inputs in place -- callers/tests
    rely on being able to compare before/after) with exactly this
    migration's own fix applied, nothing else.
    """

    id: str
    title: str
    kind: str  # "add" | "remove" | "update"
    description: str
    note: str  # printed after apply; blank when there's nothing to customize
    detect: Callable[[dict, dict], bool]
    apply: Callable[[dict, dict, MigrationContext], "tuple[dict, dict]"]


# ---------------------------------------------------------------------------
# memory-bank-server-missing (feature: #175)
# ---------------------------------------------------------------------------


def _detect_memory_bank_server_missing(mcp_json: dict, settings_json: dict) -> bool:
    return "memory-bank" not in mcp_json.get("mcpServers", {})


def _apply_memory_bank_server_missing(
    mcp_json: dict, settings_json: dict, ctx: MigrationContext
) -> "tuple[dict, dict]":
    """
    Adds the `memory-bank` server block, reusing `build_mcp_servers` (the
    SAME function `init` uses) rather than hand-assembling the server's
    fields here, so this can't silently drift from what a fresh `init` run
    would generate. Only the `memory-bank` KEY is taken from the result --
    every other key `build_mcp_servers` also computes (`qdrant`,
    `codebase-indexer`, and `local-compress` if requested) is discarded, so
    this migration never touches anything but the one key its own `detect`
    found missing.

    `QDRANT_URL`/`QDRANT_API_KEY` are derived from the target's EXISTING
    `qdrant` server block (present already -- every already-configured
    project has one; `memory-bank` is the thing that's missing, not
    `qdrant`) rather than a hardcoded default, so an authenticated remote
    Qdrant setup gets the memory-bank server pointed at the SAME instance,
    not silently reset to the unauthenticated localhost default.
    `MEMORY_BANK_COLLECTION` similarly reuses whatever the existing
    `codebase-indexer` block already has recorded (it's the one other place
    this value is written -- see `build_mcp_servers`'s own docstring),
    falling back to the template's own `"memory-bank"` default only if that
    key is somehow itself missing. `MEMORY_BANK_ID` defaults to the
    project's own `COLLECTION_NAME` -- the issue's own specified default,
    matching what a fresh `init` run does when `--memory-bank-id` isn't
    passed.

    `EMBEDDING_MODEL` is likewise taken from the existing `qdrant`/
    `codebase-indexer` block rather than left at whatever
    `build_mcp_servers` bakes in from the CURRENT `mcp.json.template`
    (`build_mcp_servers` has no `embedding_model` parameter at all -- every
    server it generates just inherits the template's own hardcoded default
    verbatim). Found in Copilot review on PR #227 and confirmed by
    reproducing it directly: a project whose `qdrant`/`codebase-indexer`
    blocks were hand-customized to a non-default `EMBEDDING_MODEL` (the
    documented, if manual, way to change this -- see README's Known
    limitations) got a freshly-added `memory-bank` block silently pinned to
    the template's CURRENT default instead, which can mismatch whatever
    model actually created the shared `memory-bank` collection on this
    machine (`docs/memory-bank.md`: "EMBEDDING_MODEL... Must match the model
    that created the shared collection") -- `memory_bank_mcp_server.py`
    detects and reports that mismatch loudly rather than silently returning
    wrong results, but this migration has no reason to manufacture one in
    the first place when the value it needs is sitting right there in the
    project's own existing config.
    """
    mcp_json = copy.deepcopy(mcp_json)
    servers = mcp_json.setdefault("mcpServers", {})
    qdrant_env = servers.get("qdrant", {}).get("env", {})
    indexer_env = servers.get("codebase-indexer", {}).get("env", {})

    qdrant_url = qdrant_env.get("QDRANT_URL") or "http://localhost:6333"
    qdrant_api_key = qdrant_env.get("QDRANT_API_KEY") or ""
    collection_name = qdrant_env.get("COLLECTION_NAME") or indexer_env.get("COLLECTION_NAME") or "project"
    memory_bank_collection = indexer_env.get("MEMORY_BANK_COLLECTION") or "memory-bank"
    embedding_model = qdrant_env.get("EMBEDDING_MODEL") or indexer_env.get("EMBEDDING_MODEL")

    template = load_json(Path(ctx.tools_repo_dir) / "templates" / "mcp.json.template")
    generated = build_mcp_servers(
        template,
        collection_name=collection_name,
        venv_python=ctx.venv_python,
        tools_repo_dir=ctx.tools_repo_dir,
        home_dir=Path.home(),
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        memory_bank_collection=memory_bank_collection,
        memory_bank_id=collection_name,
        include_compress=False,
    )
    memory_bank_server = generated["memory-bank"]
    if embedding_model:
        memory_bank_server["env"]["EMBEDDING_MODEL"] = embedding_model
    servers["memory-bank"] = memory_bank_server
    return mcp_json, settings_json


# ---------------------------------------------------------------------------
# record-session-id-hooks-missing (feature: #198)
# ---------------------------------------------------------------------------


# The two events record_session_id.py must be registered under (issue
# #198): PostToolUse creates/refreshes libs/session_id_lib.py's SHADOW_FILE
# marker, SessionEnd deletes it on graceful exit. Both matter independently
# -- see _missing_record_session_id_events below.
_RECORD_SESSION_ID_EVENTS = ("PostToolUse", "SessionEnd")


def _has_record_session_id_hook_for_event(settings_json: dict, event: str) -> bool:
    for block in settings_json.get("hooks", {}).get(event, []):
        for hook in block.get("hooks", []):
            if any(Path(a).name == "record_session_id.py" for a in hook.get("args", [])):
                return True
    return False


def _missing_record_session_id_events(settings_json: dict) -> "list[str]":
    """
    Which of `_RECORD_SESSION_ID_EVENTS` still lack a record_session_id.py
    registration -- checked PER EVENT, not "does record_session_id.py
    appear ANYWHERE in settings_json's hooks at all". Found in Copilot
    review on PR #227 and confirmed by reproducing it directly: an earlier,
    single boolean `_has_record_session_id_hook` treated the migration as
    already-applied the moment EITHER event had the registration, so a
    partially-configured project with only a SessionEnd entry (no
    PostToolUse) permanently skipped the repair -- PostToolUse is the one
    that actually creates/refreshes the marker, so that project's
    SHADOW_FILE strategy would silently never populate at all, forever,
    since `detect` would keep reporting "nothing to do" on every future
    `upgrade` run too.
    """
    return [event for event in _RECORD_SESSION_ID_EVENTS if not _has_record_session_id_hook_for_event(settings_json, event)]


def _detect_record_session_id_hooks_missing(mcp_json: dict, settings_json: dict) -> bool:
    return bool(_missing_record_session_id_events(settings_json))


def _apply_record_session_id_hooks_missing(
    mcp_json: dict, settings_json: dict, ctx: MigrationContext
) -> "tuple[dict, dict]":
    """
    Appends the CORE `record_session_id.py` block for whichever of
    `_RECORD_SESSION_ID_EVENTS` this migration's `detect` found missing --
    and ONLY those event(s) -- to whatever `settings_json` already has.
    Deliberately per-event, not "always both": a partially-configured
    project that already has (say) `SessionEnd`'s registration but not
    `PostToolUse`'s must get only `PostToolUse` added -- appending BOTH
    unconditionally would duplicate the one that's already there (see
    `_missing_record_session_id_events`'s own docstring for the real-world
    scenario this fixes, found in Copilot review on PR #227).

    Deliberately does NOT go through `setup_project_lib.merge_settings_hooks`
    (unlike what an earlier draft of this migration, following the issue's
    own suggested implementation literally, did) -- that function's
    "strip every toolkit-owned inner hook for an event present in the
    generated dict, then re-add only what was just generated" semantics is
    exactly right for `init`, which regenerates every toolkit-owned block on
    every run, but wrong here: this migration's own `detect` fires precisely
    for a project that adopted local-compress BEFORE #198 shipped, i.e. one
    whose PostToolUse already has a REAL `compress_bash_output.py` block and
    no separate core block at all (the template's own PostToolUse entry for
    record_session_id.py didn't exist yet when that project was set up).
    Calling `build_settings_hooks(..., include_compress=False)` and merging
    the result would treat PostToolUse as "generated -> only the core block"
    and, via `merge_settings_hooks`'s strip-then-replace logic, silently
    DELETE that project's real, working `compress_bash_output.py` entry --
    confirmed by writing that version first and catching it wiping the
    compress hook out of a fixture that had one, in this file's own tests.
    A pure append has no such failure mode: it only ever adds the two core
    blocks this migration's `detect` found missing, never inspects or
    touches any other block.

    Still reuses `build_settings_hooks` itself (not `merge_settings_hooks`)
    purely to get the two core blocks' `command`/`args` resolved against
    THIS `MigrationContext`'s `tools_repo_dir`/`venv_python` -- with
    `include_compress=False`, its own `PostToolUse`/`SessionEnd` results
    contain exactly one block each (the core one; the compress-gated block
    is stripped from the RETURNED dict, not from `settings_json`), so no
    filtering is needed on this end either.
    """
    settings_json = copy.deepcopy(settings_json)
    missing_events = _missing_record_session_id_events(settings_json)
    if not missing_events:
        return mcp_json, settings_json

    template = load_json(Path(ctx.tools_repo_dir) / "templates" / "settings.json.template")
    generated_hooks = build_settings_hooks(
        template, venv_python=ctx.venv_python, tools_repo_dir=ctx.tools_repo_dir, include_compress=False
    )
    hooks = settings_json.setdefault("hooks", {})
    for event in missing_events:
        existing_blocks = hooks.setdefault(event, [])
        existing_blocks.extend(copy.deepcopy(b) for b in generated_hooks[event])
    return mcp_json, settings_json


# ---------------------------------------------------------------------------
# hf-hub-offline-missing (feature: #221)
# ---------------------------------------------------------------------------


def _detect_hf_hub_offline_missing(mcp_json: dict, settings_json: dict) -> bool:
    return "HF_HUB_OFFLINE" not in mcp_json.get("mcpServers", {}).get("qdrant", {}).get("env", {})


def _apply_hf_hub_offline_missing(mcp_json: dict, settings_json: dict, ctx: MigrationContext) -> "tuple[dict, dict]":
    """
    Adds `HF_HUB_OFFLINE: ""` (present-but-blank, matching the template's own
    default) to the `qdrant` server's env block -- never unconditionally
    `"1"` here, since that requires the warm-up confirmation `init` itself
    performs (`run_setup`'s `attempt_fastembed_warmup`), which this migration
    has no way to do without a live network/disk check of its own. Re-run
    `init` afterward (see this migration's `note`) to have that confirmation
    flip it to `"1"` once the cache is actually warm.
    """
    mcp_json = copy.deepcopy(mcp_json)
    qdrant_env = mcp_json.setdefault("mcpServers", {}).setdefault("qdrant", {}).setdefault("env", {})
    qdrant_env["HF_HUB_OFFLINE"] = ""
    return mcp_json, settings_json


# Ordered oldest-feature-first -- order only affects DISPLAY order when more
# than one migration is pending; `detect` alone decides whether each one is
# shown at all.
MIGRATIONS: "list[Migration]" = [
    Migration(
        id="memory-bank-server-missing",
        title="Add memory-bank MCP server to .mcp.json",
        kind="add",
        description=(
            "This project's .mcp.json has no 'memory-bank' server configured -- durable, cross-session "
            "remember/recall/forget storage (issue #175) isn't available here yet. Adds the memory-bank "
            "server block, reusing this project's existing qdrant server's QDRANT_URL/QDRANT_API_KEY."
        ),
        note=(
            "MEMORY_BANK_ID defaults to your collection name -- customize it in .mcp.json if you want a "
            "different per-project tag for recall scoping."
        ),
        detect=_detect_memory_bank_server_missing,
        apply=_apply_memory_bank_server_missing,
    ),
    Migration(
        id="record-session-id-hooks-missing",
        title="Add record_session_id.py hooks to .claude/settings.json",
        kind="add",
        description=(
            "No hook entry in this project's .claude/settings.json references record_session_id.py -- the "
            "CORE/base-install hook (issue #198) that keeps libs/session_id_lib.py's SHADOW_FILE session-id "
            "strategy populated isn't running here yet. Adds the PostToolUse + SessionEnd blocks; any other "
            "existing hooks (including local-compress's own) are left completely untouched."
        ),
        note="",
        detect=_detect_record_session_id_hooks_missing,
        apply=_apply_record_session_id_hooks_missing,
    ),
    Migration(
        id="hf-hub-offline-missing",
        title="Add HF_HUB_OFFLINE to qdrant server env",
        kind="add",
        description=(
            "The qdrant MCP server contacts huggingface.co on every startup even with a fully cached model "
            "-- this env var prevents that once the cache is confirmed populated (issue #221). Adds "
            '"HF_HUB_OFFLINE": "" (blank) to .mcp.json qdrant.env.'
        ),
        note=(
            "Re-run init to auto-set this to '1' once your fastembed cache is confirmed populated, or set it "
            "manually if the model is already cached at FASTEMBED_CACHE_PATH."
        ),
        detect=_detect_hf_hub_offline_missing,
        apply=_apply_hf_hub_offline_missing,
    ),
]


def pending_migrations(mcp_json: dict, settings_json: dict) -> "list[Migration]":
    """Pure helper: every migration whose `detect` currently returns True,
    in `MIGRATIONS`' own display order. Exposed separately from
    `run_upgrade` so a caller/test can compute "what's pending" without
    triggering any prompting or disk writes."""
    return [m for m in MIGRATIONS if m.detect(mcp_json, settings_json)]


def _has_qdrant_server(mcp_json: dict) -> bool:
    """True if `mcp_json` already has this toolkit's OWN base-install
    marker -- a `qdrant` server block, present in EVERY real `init` run
    regardless of `--qdrant-only`/`--skip-hooks` (it's the one server key
    every configured project is guaranteed to have)."""
    return "qdrant" in mcp_json.get("mcpServers", {})


def is_project_configured(target_repo: "Path | str") -> bool:
    """
    Path-based convenience wrapper around `_has_qdrant_server`, for a caller
    (`cmd_upgrade`) that wants to check this BEFORE calling `run_upgrade` at
    all -- mirroring `cmd_init`'s own `target_repo.is_dir()` pre-check, so a
    typo'd/genuinely-unconfigured target gets a clean CLI-level error and
    exit code rather than reaching `run_upgrade`'s own (necessarily
    non-process-exiting, since it's also called directly by tests/scripts)
    guard.

    `upgrade` is documented as a tool for an ALREADY-configured project
    (`init` is what a brand-new project needs first) -- without this check,
    every migration's own `detect` fires on an effectively-empty `{}` (a
    missing `.mcp.json` is treated the same as one with no keys at all) and
    happily writes a BROKEN partial config: found in Copilot review on PR
    #227 and confirmed live, `upgrade <empty-dir> --auto-yes` wrote a
    `qdrant` server block containing only `{"env": {"HF_HUB_OFFLINE": ""}}`
    -- no `command`/`type` at all -- plus a default `memory-bank` block,
    for a directory that had never been `init`'d.
    """
    mcp_json_path = Path(target_repo) / ".mcp.json"
    if not mcp_json_path.exists():
        return False
    return _has_qdrant_server(load_json(mcp_json_path))


def _atomic_write_json(path: Path, data: dict) -> None:
    """Same atomicity guarantee as `tools/setup_project.py`'s own
    `_atomic_write`/`_write_json` (temp file in the same directory, fsync,
    then `os.replace`) -- duplicated here rather than imported, since that
    helper lives in the CLI module (`tools/setup_project.py`), not a
    `libs/*.py` module this file can import the way it imports from
    `setup_project_lib`. Kept deliberately tiny and side-effect-identical so
    it can't drift in a way that matters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2) + "\n"
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _format_migration_block(migration: Migration, index: int, total: int) -> str:
    lines = [f"[{index}/{total}] {migration.kind.upper()}: {migration.title}", f"      {migration.description}"]
    if migration.note:
        lines.append(f"      Note: {migration.note}")
    return "\n".join(lines)


def _console_prompt(migration: Migration, index: int, total: int) -> bool:
    """Default, real interactive prompt -- prints this migration's block
    then reads a Y/n answer from stdin. Blank input (just pressing Enter)
    defaults to Yes, matching the issue's own illustrated "Apply? [Y/n]"."""
    print()
    print(_format_migration_block(migration, index, total))
    answer = input("      Apply? [Y/n] ").strip().lower()
    return answer not in ("n", "no")


def run_upgrade(
    target_repo: "Path | str",
    tools_repo_dir: "Path | str",
    venv_python: "Path | str",
    *,
    auto_yes: bool = False,
    dry_run: bool = False,
    prompt_fn: Optional[Callable[[Migration, int, int], bool]] = None,
) -> "list[str]":
    """
    Reads `target_repo`'s current `.mcp.json`/`.claude/settings.json`
    (a missing file is treated as `{}`, same convention `run_setup` uses),
    computes `pending_migrations`, and -- unless `dry_run` -- applies
    whichever ones are approved (every pending one, if `auto_yes=True`;
    otherwise one at a time via `prompt_fn`, defaulting to a real console
    Y/n prompt) atomically to both files via the same temp-file+`os.replace`
    guarantee `tools/setup_project.py`'s own `_write_json` uses.

    Prints its own interactive framing (a "Checking .../Found N pending..."
    header, one block per pending migration, and a final summary line) --
    unlike `setup_project_lib.run_setup` (a pure computation the CLI prints
    for), this function owns that output directly, since `cmd_upgrade` has
    nothing useful to add around it (the issue this implements describes
    `cmd_upgrade` as simply delegating here) and the per-migration
    accept/decline loop is inherently interleaved with prompting anyway.

    `--dry-run` never prompts (there's nothing to write either way) and
    reports every pending migration as-if approved, purely for the returned
    list/printed description -- nothing is written to disk. `auto_yes=True`
    approves every pending migration with no prompt (for scripting/CI).
    Declined migrations (interactive mode only) are simply not applied and
    reappear on the next run -- no permanent-dismiss mechanism in v1, per
    the issue's own spec.

    Returns the list of migration `id`s that were (or, for `--dry-run`, would
    have been) applied, in `MIGRATIONS`' display order. Returns `[]` with an
    error printed (no exception, no files touched) if `target_repo` has no
    existing toolkit configuration at all -- see `is_project_configured`'s
    own docstring for why this refusal exists; `cmd_upgrade` additionally
    checks this itself BEFORE ever calling here, for a clean CLI exit code,
    but this guard stays here too since `run_upgrade` is also called
    directly (by tests, or any other script), not only through that CLI.
    """
    target_repo = Path(target_repo)
    ctx = MigrationContext(tools_repo_dir=Path(tools_repo_dir), venv_python=Path(venv_python))

    mcp_json_path = target_repo / ".mcp.json"
    settings_path = target_repo / ".claude" / "settings.json"
    mcp_json = load_json(mcp_json_path) if mcp_json_path.exists() else {}
    settings_json = load_json(settings_path) if settings_path.exists() else {}

    print(f"Checking {target_repo} for pending upgrades...")

    if not _has_qdrant_server(mcp_json):
        print(
            f"\nerror: no existing toolkit configuration found at {mcp_json_path} (no 'qdrant' MCP "
            "server block) -- `upgrade` only applies migrations to an ALREADY-configured project. "
            "Run `init` first, then `upgrade` to pick up anything released since."
        )
        return []

    pending = pending_migrations(mcp_json, settings_json)

    if not pending:
        print("\nNo pending upgrades -- .mcp.json/.claude/settings.json already reflect every known migration.")
        return []

    print(f"\nFound {len(pending)} pending upgrade(s):")

    ask = prompt_fn or _console_prompt
    applied: "list[str]" = []
    skipped = 0
    for index, migration in enumerate(pending, start=1):
        if dry_run:
            print()
            print(_format_migration_block(migration, index, len(pending)))
            applied.append(migration.id)
            continue
        if auto_yes:
            print()
            print(_format_migration_block(migration, index, len(pending)))
            print("      Auto-applying (--auto-yes).")
            accept = True
        else:
            accept = ask(migration, index, len(pending))
        if not accept:
            skipped += 1
            continue
        mcp_json, settings_json = migration.apply(mcp_json, settings_json, ctx)
        applied.append(migration.id)

    if dry_run:
        print(f"\nDry run: no changes written. {len(pending)} pending upgrade(s) would be applied.")
        return applied

    if applied:
        _atomic_write_json(mcp_json_path, mcp_json)
        _atomic_write_json(settings_path, settings_json)

    print(f"\nDone. {len(applied)} change(s) applied, {skipped} skipped.")
    return applied
