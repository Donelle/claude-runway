"""
Shared logic behind `tools/setup_project.py upgrade` -- a config
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
existing `compress_output.py` PostToolUse block) that a narrower,
single-migration apply never intended to touch. A migration's `apply` only
ever ADDS the one thing its `detect` found missing -- confirmed by writing
the naive merge-based version first and catching it deleting a pre-existing
compress hook in a test before shipping this.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Flat import, matching every other libs/*.py module's own convention (see
# setup_project_lib.py's module docstring for why: libs/ is added to
# sys.path directly by callers, not treated as a real Python package).
from qdrant_ingest_lib import MANIFEST_FILENAME, is_old_format_manifest
from setup_project_lib import (
    COMPRESS_HOOK_SCRIPT,
    LEGACY_COMPRESS_HOOK_SCRIPT,
    REVIEWERS_FILE_REQUIRED_WARNING,
    SAVINGS_TRACKER_SHELL_EXPORT_REMINDER,
    TRACK_SAVINGS_TRUTHY,
    build_mcp_servers,
    build_settings_hooks,
    has_analytics_block,
    has_codebase_reviewer_block,
    load_existing_mcp_json,
    load_existing_settings_json,
    load_json,
    local_compress_tracks_savings,
    resolve_compact_embedding_model,
)


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


@dataclass
class RepoMigration:
    """
    One named, independently-detectable gap in a FILE inside the target repo
    (not in its `.mcp.json`/`.claude/settings.json`), and its fix.

    Same shape as `Migration` for everything a prompt or a printout reads (`id`,
    `title`, `kind`, `description`, `note`), but `detect(target_repo) -> bool`
    and `apply(target_repo) -> None` take the repo path and touch disk. Kept
    config-only in spirit: an apply may delete or rewrite a bookkeeping file,
    never run indexing or embedding code.
    """

    id: str
    title: str
    kind: str  # "add" | "remove" | "update"
    description: str
    note: str
    detect: Callable[[Path], bool]
    apply: Callable[[Path], None]


# ---------------------------------------------------------------------------
# index-manifest-old-format
# ---------------------------------------------------------------------------


def _detect_old_format_manifest(target_repo: Path) -> bool:
    """
    True when the repo's `.qdrant_index_manifest.json` is in the old flat
    `{file: hash}` format and holds at least one entry.

    The manifest is versioned now (see `qdrant_ingest_lib.MANIFEST_FORMAT_VERSION`),
    so `sync_repo` already ignores an old-format one and rebuilds by itself; this
    migration exists to say so out loud and to remove the stale file, not for
    correctness. Idempotent by construction: once the file is deleted (or
    rewritten by the next sync in the new format) this returns False. An absent,
    unreadable, empty or new-format manifest is not pending.
    """
    manifest_path = Path(target_repo) / MANIFEST_FILENAME
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return is_old_format_manifest(data)


def _apply_old_format_manifest(target_repo: Path) -> None:
    (Path(target_repo) / MANIFEST_FILENAME).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# memory-bank-server-missing
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

    KNOWN LEGACY BEHAVIOR: `EMBEDDING_MODEL` is copied from the
    existing `qdrant`/`codebase-indexer` block instead of being left at what
    `build_mcp_servers` bakes in from the CURRENT `mcp.json.template`
    (`build_mcp_servers`'s `embedding_model` parameter
    defaults to blank, so every server it generates inherits the template's
    own default -- and this migration doesn't pass it). That copy was added
    after a Copilot review: a project whose `qdrant`/
    `codebase-indexer` blocks were hand-customized to a non-default model
    (see README's Known limitations) got a freshly-added `memory-bank` block
    pinned to the template's default, which can mismatch the model that
    created the shared `memory-bank` collection (`docs/memory-bank.md`: the
    collection locks in whichever `EMBEDDING_MODEL` created it first).
    Memory-bank's model is now its own value,
    independent of the code-index model, so the copy is not a correct source
    either and can produce the same mismatch the other way round;
    `memory_bank_mcp_server.py` reports such a mismatch loudly rather than
    returning wrong results.
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
# codebase-indexer-memory-bank-collection-missing
# ---------------------------------------------------------------------------


def _detect_indexer_memory_bank_collection_missing(mcp_json: dict, settings_json: dict) -> bool:
    """
    True only when a `codebase-indexer` server EXISTS and its env lacks the
    key. A project with no indexer block at all (e.g. hand-built with only
    `qdrant`) has nothing for this migration to fix -- creating a whole
    indexer server here would be far outside "touch only what detect found
    missing".
    """
    indexer = mcp_json.get("mcpServers", {}).get("codebase-indexer")
    if indexer is None:
        return False
    return not indexer.get("env", {}).get("MEMORY_BANK_COLLECTION")


def _apply_indexer_memory_bank_collection_missing(
    mcp_json: dict, settings_json: dict, ctx: MigrationContext
) -> "tuple[dict, dict]":
    """
    Adds `MEMORY_BANK_COLLECTION` to the existing `codebase-indexer` env.
    `index_repo`/`sync_repo`'s refusal to touch the shared memory-bank
    collection compares against THIS block's own copy (falling back to the
    default name when unset), so a project whose memories live under a
    custom collection name is unprotected until this key matches it.

    The value is copied from the existing `memory-bank` block when it has
    one (that block is where the memories really live, so it's the source of
    truth for what the guard must protect), else the template's own
    `"memory-bank"` default. Only this one key is written.
    """
    mcp_json = copy.deepcopy(mcp_json)
    servers = mcp_json.setdefault("mcpServers", {})
    memory_bank_env = servers.get("memory-bank", {}).get("env", {})
    value = memory_bank_env.get("MEMORY_BANK_COLLECTION") or "memory-bank"
    indexer_env = servers["codebase-indexer"].setdefault("env", {})
    if not indexer_env.get("MEMORY_BANK_COLLECTION"):
        indexer_env["MEMORY_BANK_COLLECTION"] = value
    return mcp_json, settings_json


# ---------------------------------------------------------------------------
# record-session-id-hooks-missing
# ---------------------------------------------------------------------------


# The two events record_session_id.py must be registered under:
# SessionStart creates/refreshes libs/session_id_lib.py's
# SHADOW_FILE marker once per session lifecycle event (startup/resume/
# clear/compact/fork), SessionEnd deletes it on graceful exit.  Both
# matter independently -- see _missing_record_session_id_events below.
# PostToolUse was the original event but was replaced by SessionStart
# to eliminate the ~400ms-per-tool-call Python startup overhead on
# Windows.  Projects still carrying the old PostToolUse '.*' registration
# are handled by the separate record-session-id-sessionstart migration below.
_RECORD_SESSION_ID_EVENTS = ("SessionStart", "SessionEnd")


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
    review and confirmed by reproducing it directly: an earlier,
    single boolean `_has_record_session_id_hook` treated the migration as
    already-applied the moment EITHER event had the registration, so a
    partially-configured project with only a SessionEnd entry (no
    SessionStart) permanently skipped the repair -- SessionStart is the one
    that actually creates/refreshes the marker (PostToolUse was
    the original event), so that project's SHADOW_FILE
    strategy would silently never populate at all, forever, since `detect`
    would keep reporting "nothing to do" on every future `upgrade` run too.
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
    `SessionStart`'s must get only `SessionStart` added -- appending BOTH
    unconditionally would duplicate the one that's already there (see
    `_missing_record_session_id_events`'s own docstring for the real-world
    scenario this fixes, found in Copilot review).

    Deliberately does NOT go through `setup_project_lib.merge_settings_hooks`
    (unlike what an earlier draft of this migration, following the issue's
    own suggested implementation literally, did) -- that function's
    "strip every toolkit-owned inner hook for an event present in the
    generated dict, then re-add only what was just generated" semantics is
    exactly right for `init`, which regenerates every toolkit-owned block on
    every run, but wrong here: this migration's own `detect` fires precisely
    for a project that adopted local-compress BEFORE the session-id hook shipped, i.e. one
    whose PostToolUse already has a REAL `compress_output.py` block and
    no separate CORE block at all (the template's own SessionStart entry for
    record_session_id.py didn't exist yet when that project was set up --
    it was originally a PostToolUse entry, replaced by SessionStart).
    Calling `build_settings_hooks(..., include_compress=False)`
    and merging the result would treat PostToolUse as "generated -> only the
    core block" and, via `merge_settings_hooks`'s strip-then-replace logic,
    silently DELETE that project's real, working `compress_output.py`
    entry -- confirmed by writing that version first and catching it wiping
    the compress hook out of a fixture that had one, in this file's own tests.
    A pure append has no such failure mode: it only ever adds the core
    blocks this migration's `detect` found missing, never inspects or
    touches any other block.

    Still reuses `build_settings_hooks` itself (not `merge_settings_hooks`)
    purely to get the core blocks' `command`/`args` resolved against THIS
    `MigrationContext`'s `tools_repo_dir`/`venv_python` -- with
    `include_compress=False`, its `SessionStart`/`SessionEnd` results
    contain exactly one block each (the core one; any compress-gated blocks
    are stripped from the RETURNED dict, not from `settings_json`), so no
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
# qdrant-server-replaced (feature: own qdrant MCP server)
# ---------------------------------------------------------------------------
#
# This migration replaces two earlier ones that no longer have anything to do:
# `hf-hub-offline-missing` (the new server never needs HF_HUB_OFFLINE: the
# embedding provider loads from the local cache without the network on its own)
# and `local-compress-embedding-model-missing` (the compact embedding model now
# lives in the `qdrant` block, and this migration carries it over, pinning the
# legacy model exactly as that one did).

_THIRD_PARTY_QDRANT_SCRIPT = "mcp-server-qdrant"
_OWN_QDRANT_SCRIPT = "qdrant_mcp_server.py"


def _launches_third_party_qdrant_server(block: object) -> bool:
    """
    True when a `qdrant` server block still starts the third-party
    `mcp-server-qdrant` console script -- as its `command` (what `init` wrote:
    `<venv>/bin/mcp-server-qdrant`, `.exe` on Windows) or, for a hand-edited
    launcher such as `uvx`/`pipx run`, as one of its `args`. Matched by
    basename so it holds wherever the venv or tool environment lives.
    """
    if not isinstance(block, dict):
        return False
    candidates = [block.get("command")] + list(block.get("args") or [])
    for value in candidates:
        if not isinstance(value, str):
            continue
        name = Path(value).name
        if name.lower().endswith(".exe"):
            name = name[: -len(".exe")]
        if name == _THIRD_PARTY_QDRANT_SCRIPT:
            return True
    return False


def _detect_qdrant_server_replaced(mcp_json: dict, settings_json: dict) -> bool:
    """False for a project with no `qdrant` block (nothing to rewrite) and for
    one that already runs this repo's own server (nothing left to do), which is
    what makes re-running `upgrade` a no-op."""
    return _launches_third_party_qdrant_server(mcp_json.get("mcpServers", {}).get("qdrant"))


def _env_str(env: object, key: str) -> str:
    value = env.get(key) if isinstance(env, dict) else None
    return value if isinstance(value, str) else ""


# Pre-move servers that talk to Qdrant, in the order a missing value is looked
# for. `init` wrote the same URL and key into all of them, but a hand-edited or
# older config can have them in only one (the setup skill checks every one of
# these blocks for the key for exactly that reason), and the compact tools now
# reach Qdrant through the qdrant block alone.
_QDRANT_USING_SERVERS = ("qdrant", "local-compress", "codebase-indexer", "memory-bank")


def _first_qdrant_env_str(servers: dict, key: str) -> str:
    """The first non-blank `key` found across the Qdrant-using pre-move blocks."""
    for name in _QDRANT_USING_SERVERS:
        block = servers.get(name)
        value = _env_str(block.get("env") if isinstance(block, dict) else None, key)
        if value.strip():
            return value
    return ""


def _apply_qdrant_server_replaced(
    mcp_json: dict, settings_json: dict, ctx: MigrationContext
) -> "tuple[dict, dict]":
    """
    Rewrites the `qdrant` block to start `mcpservers/qdrant_mcp_server.py` under the
    venv interpreter, with the env that server reads, reusing `build_mcp_servers`
    (the SAME function `init` uses) so it cannot drift from a fresh `init`.
    Only the `qdrant` KEY is taken from the result; every other server, and the
    whole of `settings.json`, is left as it was.

    What is carried over from the old config, so nothing a user set is reset:
      - `QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME` and `EMBEDDING_MODEL`
        from the old `qdrant` block (`COLLECTION_NAME`/`EMBEDDING_MODEL` fall
        back to the `codebase-indexer` block's, which must agree with them;
        the URL and key fall back through `local-compress`, `codebase-indexer`
        and `memory-bank`, so a credential kept in only one of them survives);
      - the conversation-compact settings from the `local-compress` block,
        where the compact tools used to live: `COMPACT_COLLECTION`, the LM
        Studio URL and model, and the compact embedding model. The model is
        resolved by `resolve_compact_embedding_model`, so an install whose
        compacts were stored before that setting existed is pinned to
        `BAAI/bge-small-en`, never the new default, which would orphan them.
        (The `local-compress` block itself is left alone: it keeps working, it
        just no longer serves the compact tools, and `init` rewrites it.)

    What is dropped, because the new server has no use for it: `HF_HUB_OFFLINE`,
    `FASTEMBED_CACHE_PATH` (set in code now) and the `TOOL_FIND_DESCRIPTION`/
    `TOOL_STORE_DESCRIPTION` texts (the descriptions live in the server).
    """
    mcp_json = copy.deepcopy(mcp_json)
    servers = mcp_json.setdefault("mcpServers", {})
    old_env = servers.get("qdrant", {}).get("env", {})
    indexer_env = servers.get("codebase-indexer", {}).get("env", {})
    compress_env = servers.get("local-compress", {}).get("env", {})
    # The old qdrant block's own values win. A value missing from it is looked
    # for in the other Qdrant-using blocks, so an authenticated install whose
    # key lives only in local-compress (or the indexer) keeps it instead of
    # having the moved compact tools write an empty one.
    qdrant_url = _first_qdrant_env_str(servers, "QDRANT_URL") or "http://localhost:6333"
    qdrant_api_key = _first_qdrant_env_str(servers, "QDRANT_API_KEY")

    collection_name = _env_str(old_env, "COLLECTION_NAME") or _env_str(indexer_env, "COLLECTION_NAME") or "project"
    embedding_model = _env_str(old_env, "EMBEDDING_MODEL") or _env_str(indexer_env, "EMBEDDING_MODEL")

    template = load_json(Path(ctx.tools_repo_dir) / "templates" / "mcp.json.template")
    template_model = template["mcpServers"]["qdrant"]["env"]["EMBEDDING_MODEL"]
    generated = build_mcp_servers(
        template,
        collection_name=collection_name,
        venv_python=ctx.venv_python,
        tools_repo_dir=ctx.tools_repo_dir,
        home_dir=Path.home(),
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        lmstudio_url=_env_str(compress_env, "CLAUDE_RUNWAY_LMSTUDIO_URL") or "http://localhost:1234/v1",
        lmstudio_model=_env_str(compress_env, "CLAUDE_RUNWAY_LMSTUDIO_MODEL"),
        compact_collection=_env_str(compress_env, "COMPACT_COLLECTION"),
        embedding_model=embedding_model,
        compact_embedding_model=resolve_compact_embedding_model(mcp_json, embedding_model or template_model),
        include_compress=False,
    )
    servers["qdrant"] = generated["qdrant"]
    return mcp_json, settings_json


# ---------------------------------------------------------------------------
# record-session-id-sessionstart
# ---------------------------------------------------------------------------


def _detect_record_session_id_posttooluse_wildcard(mcp_json: dict, settings_json: dict) -> bool:
    """
    True when .claude/settings.json still has the OLD PostToolUse '.*' block
    for record_session_id.py (the original registration, superseded
    by SessionStart).  Detects by matcher value AND script name
    so a user's own unrelated '.*' PostToolUse block isn't falsely flagged.
    """
    for block in settings_json.get("hooks", {}).get("PostToolUse", []):
        if block.get("matcher") != ".*":
            continue
        for hook in block.get("hooks", []):
            if any(Path(a).name == "record_session_id.py" for a in hook.get("args", [])):
                return True
    return False


def _apply_record_session_id_posttooluse_wildcard(
    mcp_json: dict, settings_json: dict, ctx: MigrationContext
) -> "tuple[dict, dict]":
    """
    Removes only the record_session_id.py INNER HOOK entry from any PostToolUse
    block whose matcher is '.*' -- the stale registration that fired a Python
    subprocess (~400ms on Windows) on every single tool call.
    Operates at INNER-HOOK granularity (same as _strip_toolkit_owned_inner_hooks
    in setup_project_lib.py), not whole-block: if a user manually added another
    inner hook to the same '.*' block, only the record_session_id.py entry is
    removed and the block survives with the remaining content intact (confirmed
    as the data-loss-safe behaviour by Copilot review).  A block
    that contained ONLY the record_session_id.py entry is dropped entirely,
    leaving 'PostToolUse: []' rather than removing the key -- consistent with
    _strip_toolkit_owned_inner_hooks.  An empty PostToolUse array is harmless.

    Safety guard: returns unchanged (no-op) when SessionStart is not yet present,
    preventing the dangerous state of removing the PostToolUse fallback before
    its replacement is in place (see inline comment below).  Re-run after
    record-session-id-hooks-missing has added the SessionStart block.
    """
    # Safety guard: only remove the stale PostToolUse
    # '.*' block if a SessionStart registration for record_session_id.py is
    # ALREADY PRESENT in the current (in-memory) settings_json.  Without this
    # guard, a user who declines record-session-id-hooks-missing (the migration
    # that ADDS SessionStart) but accepts this removal would end up with no
    # SHADOW_FILE recorder at all -- worse than the original state.
    #
    # run_upgrade passes each migration the cumulative in-memory result of all
    # prior applies, so this guard also handles the normal "both pending, both
    # accepted" case correctly: by the time this apply runs,
    # record-session-id-hooks-missing has already added SessionStart to the
    # in-memory settings_json, so the guard passes and the removal proceeds.
    if not _has_record_session_id_hook_for_event(settings_json, "SessionStart"):
        return mcp_json, settings_json  # no-op; safe to re-run after SessionStart is configured

    settings_json = copy.deepcopy(settings_json)
    hooks = settings_json.get("hooks", {})
    post_blocks = hooks.get("PostToolUse", [])
    surviving = []
    for block in post_blocks:
        if block.get("matcher") == ".*":
            # Strip at INNER-HOOK granularity (same as _strip_toolkit_owned_inner_hooks
            # in setup_project_lib.py) so a user who manually added their own hook
            # to the same '.*' block doesn't lose it.  A whole-block drop would only
            # be correct if the block contained ONLY the record_session_id.py entry --
            # confirmed as the real failure mode by Copilot review.
            remaining_inner = [
                hook for hook in block.get("hooks", [])
                if not any(Path(a).name == "record_session_id.py" for a in hook.get("args", []))
            ]
            if not remaining_inner:
                continue  # nothing left in this block -- drop it entirely
            modified = copy.deepcopy(block)
            modified["hooks"] = remaining_inner
            surviving.append(modified)
        else:
            surviving.append(block)
    hooks["PostToolUse"] = surviving
    return mcp_json, settings_json


# ---------------------------------------------------------------------------
# compress-hook-renamed
# ---------------------------------------------------------------------------


def _is_script_hook(hook: dict, script_name: str) -> bool:
    return any(Path(a).name == script_name for a in hook.get("args", []))


def _detect_compress_hook_renamed(mcp_json: dict, settings_json: dict) -> bool:
    """True when ANY hook, under any event, still invokes the pre-rename
    `compress_bash_output.py` -- a path that no longer exists in the toolkit
    repo, so that hook now fails on every tool call."""
    for blocks in settings_json.get("hooks", {}).values():
        for block in blocks:
            if any(_is_script_hook(h, LEGACY_COMPRESS_HOOK_SCRIPT) for h in block.get("hooks", [])):
                return True
    return False


def _apply_compress_hook_renamed(mcp_json: dict, settings_json: dict, ctx: MigrationContext) -> "tuple[dict, dict]":
    """
    Rewrites ONLY the script filename in each legacy hook's args
    (`.../hooks/compress_bash_output.py` -> `.../hooks/compress_output.py`),
    keeping the directory, the command/venv, the matcher, and every sibling
    inner hook exactly as they were -- the same "never delete a working
    block" discipline as record-session-id-hooks-missing (see its docstring
    for why a regenerate-and-merge would be wrong here: it would clobber a
    project's own matcher/path customizations).

    One guarded exception: if the SAME event AND MATCHER already has a hook
    invoking the new name (hand-added, or from a partial earlier run),
    rewriting would leave two compress hooks double-compressing every call
    matching it, so the stale legacy inner hook is dropped instead (the
    block itself survives if it still holds anything else). Scoped to the
    matcher, not the whole event: a legacy `Bash` block beside a new-named
    `Grep` block is NOT a duplicate -- dropping it would silently disable
    Bash compression.
    """
    settings_json = copy.deepcopy(settings_json)
    for blocks in settings_json.get("hooks", {}).values():
        new_matchers = {
            b.get("matcher")
            for b in blocks
            if any(_is_script_hook(h, COMPRESS_HOOK_SCRIPT) for h in b.get("hooks", []))
        }
        surviving = []
        for block in blocks:
            has_new = block.get("matcher") in new_matchers
            inner = []
            for hook in block.get("hooks", []):
                if _is_script_hook(hook, LEGACY_COMPRESS_HOOK_SCRIPT):
                    if has_new:
                        continue  # the new-named hook is already registered; drop the stale duplicate
                    hook["args"] = [
                        a[: -len(LEGACY_COMPRESS_HOOK_SCRIPT)] + COMPRESS_HOOK_SCRIPT
                        if Path(a).name == LEGACY_COMPRESS_HOOK_SCRIPT
                        else a
                        for a in hook.get("args", [])
                    ]
                inner.append(hook)
            if block.get("hooks") and not inner:
                continue  # block held only the stale duplicate
            if "hooks" in block:
                block["hooks"] = inner
            surviving.append(block)
        blocks[:] = surviving
    return mcp_json, settings_json


# ---------------------------------------------------------------------------
# compress-hook-matcher-covers-compact-tools (follows qdrant-server-replaced)
# ---------------------------------------------------------------------------

# The alternative `templates/settings.json.template`'s compress-hook matcher
# carries for the conversation-compact tools, now served by the `qdrant`
# server. tests/test_upgrade_lib.py pins that the template contains exactly it.
COMPACT_TOOLS_MATCHER = "mcp__qdrant__(?:compact_store|compact_find|compact_prune)$"
_LOCAL_COMPRESS_MATCHER = "mcp__local-compress__.*"


def _compress_hook_blocks_missing_compact_matcher(settings_json: dict) -> "list[dict]":
    """
    Blocks that run the compress hook (under its current or its legacy name)
    with a matcher that lists the `local-compress` tools by the exact text
    `init` writes but not the compact tools' new `mcp__qdrant__` alternative.
    A matcher this toolkit did not write (no `mcp__local-compress__.*` in it)
    is deliberately not matched: it is the user's own, and guessing where to
    splice into it could break it.
    """
    found = []
    for blocks in settings_json.get("hooks", {}).values():
        for block in blocks:
            matcher = block.get("matcher")
            if not isinstance(matcher, str):
                continue
            runs_compress_hook = any(
                _is_script_hook(h, COMPRESS_HOOK_SCRIPT) or _is_script_hook(h, LEGACY_COMPRESS_HOOK_SCRIPT)
                for h in block.get("hooks", [])
            )
            if runs_compress_hook and _LOCAL_COMPRESS_MATCHER in matcher and COMPACT_TOOLS_MATCHER not in matcher:
                found.append(block)
    return found


def _detect_compress_hook_matcher_missing_compact_tools(mcp_json: dict, settings_json: dict) -> bool:
    return bool(_compress_hook_blocks_missing_compact_matcher(settings_json))


def _apply_compress_hook_matcher_missing_compact_tools(
    mcp_json: dict, settings_json: dict, ctx: MigrationContext
) -> "tuple[dict, dict]":
    """
    Adds the compact tools' `mcp__qdrant__` alternative right after
    `mcp__local-compress__.*` in each compress-hook matcher that lacks it. The
    compact tools used to be `mcp__local-compress__` tools, so a matcher `init`
    wrote before they moved stops reaching the hook for them after
    `qdrant-server-replaced`, silently dropping their size-based compression
    (and its savings events) until `init` is re-run. Touches nothing else: not
    the hook's command or args, not any other alternative in the matcher, not
    any other block or hook, and not a matcher the user wrote themselves.
    """
    settings_json = copy.deepcopy(settings_json)
    for block in _compress_hook_blocks_missing_compact_matcher(settings_json):
        block["matcher"] = block["matcher"].replace(
            _LOCAL_COMPRESS_MATCHER, f"{_LOCAL_COMPRESS_MATCHER}|{COMPACT_TOOLS_MATCHER}", 1
        )
    return mcp_json, settings_json


# ---------------------------------------------------------------------------
# analytics-server-missing (follows the savings and metrics tools' move)
# ---------------------------------------------------------------------------

# The values `savings_ledger.tracking_enabled()` accepts: defined once in
# `setup_project_lib` (which `init`'s savings question shares), not duplicated.
_TRACK_SAVINGS_TRUTHY = TRACK_SAVINGS_TRUTHY


def _local_compress_env(mcp_json: dict) -> dict:
    servers = mcp_json.get("mcpServers")
    block = servers.get("local-compress") if isinstance(servers, dict) else None
    env = block.get("env") if isinstance(block, dict) else None
    return env if isinstance(env, dict) else {}


def _detect_analytics_server_missing(mcp_json: dict, settings_json: dict) -> bool:
    """
    True when this project's `local-compress` block has savings tracking on
    (its `CLAUDE_RUNWAY_TRACK_SAVINGS` is a truthy string) but there is no
    `analytics` block. That block now serves `savings_summary`/`savings_detail`/
    `savings_trend`, `get_metrics` and `record_metric`, so `/my-savings` and
    `/my-metrics` stop working after the tools leave `local-compress` until it
    is added. A project with tracking off is deliberately not offered THIS
    migration: the server is opt-in. An interactive `upgrade` asks about the
    tracker separately, before the migrations are listed (`run_upgrade`'s
    `offer_savings_fn`), and a yes turns tracking on so this one then applies.
    """
    servers = mcp_json.get("mcpServers")
    if isinstance(servers, dict) and "analytics" in servers:
        return False
    raw = _local_compress_env(mcp_json).get("CLAUDE_RUNWAY_TRACK_SAVINGS")
    return isinstance(raw, str) and raw.strip().lower() in _TRACK_SAVINGS_TRUTHY


def _existing_metrics_db(mcp_json: dict) -> str:
    """
    The `CLAUDE_RUNWAY_METRICS_DB` override this project already has, so a new
    `analytics` block reads the same file. `init --metrics-db` writes it into
    the `memory-bank` block in EVERY mode (that server writes into the store),
    and into `analytics` only with `--track-savings`; a project that never
    tracked savings therefore has it only in `memory-bank`. An older config
    can have it in `local-compress` instead, which is the fallback. First
    non-blank wins, in the same order the setup skill reads it. Without this,
    adding `analytics` for a project with a custom metrics database would
    silently split the data: `memory-bank` writes the custom file while
    `analytics` reads and writes the default one.
    """
    servers = mcp_json.get("mcpServers")
    if not isinstance(servers, dict):
        return ""
    for name in ("memory-bank", "local-compress"):
        block = servers.get(name)
        value = _env_str(block.get("env") if isinstance(block, dict) else None, "CLAUDE_RUNWAY_METRICS_DB")
        # Blank is checked on the stripped text, but the ORIGINAL is returned:
        # `metrics_lib.resolve_db_path()` does not strip, and `memory-bank` keeps
        # its own value untouched, so trimming here would point `analytics` at a
        # different file than the one `memory-bank` writes.
        if value.strip():
            return value
    return ""


def _apply_analytics_server_missing(
    mcp_json: dict, settings_json: dict, ctx: MigrationContext
) -> "tuple[dict, dict]":
    """
    Adds the `analytics` block, reusing `build_mcp_servers` (the SAME function
    `init --track-savings` uses) so it cannot drift from a fresh `init`. Only
    the `analytics` KEY is taken from the result; every other server, and the
    whole of `settings.json`, is left as it was. A non-blank
    `CLAUDE_RUNWAY_SAVINGS_DB` in the `local-compress` block and a non-blank
    `CLAUDE_RUNWAY_METRICS_DB` in the `memory-bank` block (else `local-compress`)
    are carried over, so the server reads the same files the tools read before
    they moved, and the same metrics file `memory-bank` writes.
    """
    mcp_json = copy.deepcopy(mcp_json)
    servers = mcp_json.setdefault("mcpServers", {})
    compress_env = _local_compress_env(mcp_json)
    template = load_json(Path(ctx.tools_repo_dir) / "templates" / "mcp.json.template")
    generated = build_mcp_servers(
        template,
        collection_name="project",
        venv_python=ctx.venv_python,
        tools_repo_dir=ctx.tools_repo_dir,
        home_dir=Path.home(),
        track_savings=True,
        savings_db=_env_str(compress_env, "CLAUDE_RUNWAY_SAVINGS_DB").strip(),
        metrics_db=_existing_metrics_db(mcp_json),
        include_compress=True,
    )
    servers["analytics"] = generated["analytics"]
    return mcp_json, settings_json


# ---------------------------------------------------------------------------
# local-compress-savings-db-inert
# ---------------------------------------------------------------------------

_SAVINGS_DB_KEY = "CLAUDE_RUNWAY_SAVINGS_DB"


def _detect_local_compress_savings_db_inert(mcp_json: dict, settings_json: dict) -> bool:
    """
    True when the `local-compress` block's env still has a
    `CLAUDE_RUNWAY_SAVINGS_DB` key (blank or not). That server only reads
    `CLAUDE_RUNWAY_TRACK_SAVINGS` (to mark a compressed result for the hook);
    it never opens the savings database, so the key there is inert. Every
    earlier `init` wrote it, blank unless `--savings-db` was given.
    """
    return _SAVINGS_DB_KEY in _local_compress_env(mcp_json)


def _apply_local_compress_savings_db_inert(
    mcp_json: dict, settings_json: dict, ctx: MigrationContext
) -> "tuple[dict, dict]":
    """
    Removes just that one key from the `local-compress` env block; every
    other key, server and the whole of `settings.json` are left as they were.

    A no-op while `analytics-server-missing` is still pending:
    that migration carries this very value over into the new `analytics`
    block, so removing it first (or at all, if the user declined that one)
    would lose a real override before the one server that reads it has it.
    The guard runs here rather than in `detect` because `run_upgrade` computes
    the pending list once, before anything is applied: in a single run where
    both are accepted, `analytics-server-missing` (earlier in `MIGRATIONS`)
    has already added the block by the time this runs, so the removal goes
    ahead; if it was declined, this changes nothing and `run_upgrade` reports
    it as still pending (same pattern as `record-session-id-sessionstart`).
    """
    if _detect_analytics_server_missing(mcp_json, settings_json):
        return mcp_json, settings_json
    mcp_json = copy.deepcopy(mcp_json)
    _local_compress_env(mcp_json).pop(_SAVINGS_DB_KEY, None)
    return mcp_json, settings_json


# ---------------------------------------------------------------------------
# mcp-server-scripts-moved
# ---------------------------------------------------------------------------
#
# The five MCP server scripts moved from `tools/` to `mcpservers/` (same file
# names). Every `.mcp.json` written before that hardcodes
# `<checkout>/tools/<server>.py`, so after a pull all five servers fail to start
# until the path is rewritten. Detection is by the OLD path alone, which is what
# makes a re-run a no-op: once rewritten, nothing matches any more.

_MOVED_MCP_SERVER_SCRIPTS = (
    "compress_mcp_server",
    "analytics_mcp_server",
    "qdrant_mcp_server",
    "ingest_mcp_server",
    "memory_bank_mcp_server",
)

# `tools` as a whole path segment (either separator, so a Windows path written
# with backslashes is caught too) directly followed by one of the five script
# names and end of string. Anchoring on the segment and the exact file name
# means a user's own script that happens to sit in some other `tools/` folder,
# or a path merely ending in `compress_mcp_server.py` elsewhere, is not touched.
_OLD_MCP_SERVER_PATH = re.compile(
    r"(^|[\\/])tools([\\/])((?:" + "|".join(_MOVED_MCP_SERVER_SCRIPTS) + r")\.py)$"
)


def _rewrite_moved_server_path(value: object) -> "Optional[str]":
    """The `mcpservers/` form of an old `tools/<server>.py` path, keeping the
    separator the old path used, or None when `value` is not such a path."""
    if not isinstance(value, str):
        return None
    new, count = _OLD_MCP_SERVER_PATH.subn(r"\1mcpservers\2\3", value)
    return new if count else None


def _detect_mcp_server_scripts_moved(mcp_json: dict, settings_json: dict) -> bool:
    """True when any server block still launches one of the five scripts from
    its old `tools/` location (a bare script path in `args`)."""
    servers = mcp_json.get("mcpServers", {})
    if not isinstance(servers, dict):
        return False
    for block in servers.values():
        if not isinstance(block, dict):
            continue
        for arg in block.get("args") or []:
            if _rewrite_moved_server_path(arg) is not None:
                return True
    return False


def _apply_mcp_server_scripts_moved(
    mcp_json: dict, settings_json: dict, ctx: MigrationContext
) -> "tuple[dict, dict]":
    """
    Rewrites each matching `args` entry from `.../tools/<server>.py` to
    `.../mcpservers/<server>.py`, and nothing else: the venv `command`, every
    `env` value, every other server and the whole of `settings.json` are left
    as they were. Pure path rewriting rather than regenerating the blocks with
    `build_mcp_servers`, so a hand-edited block keeps everything the user put
    in it. A pipx or `uv tool` install has the same `tools/` segment under
    `<env>/src/`, so its paths are rewritten by the same rule.
    """
    mcp_json = copy.deepcopy(mcp_json)
    servers = mcp_json.get("mcpServers", {})
    for block in servers.values():
        if not isinstance(block, dict) or not isinstance(block.get("args"), list):
            continue
        block["args"] = [
            _rewrite_moved_server_path(arg) or arg for arg in block["args"]
        ]
    return mcp_json, settings_json


# List order is both display and apply order when more than one migration is
# pending; `detect` alone decides whether each one is shown at all. Order only
# matters where one migration's guard depends on another:
# record-session-id-sessionstart needs record-session-id-hooks-missing to
# appear (and be applied) earlier in this list, and
# local-compress-savings-db-inert needs analytics-server-missing earlier.
MIGRATIONS: "list[Migration]" = [
    Migration(
        id="memory-bank-server-missing",
        title="Add memory-bank MCP server to .mcp.json",
        kind="add",
        description=(
            "This project's .mcp.json has no 'memory-bank' server configured -- durable, cross-session "
            "remember/recall/forget storage isn't available here yet. Adds the memory-bank "
            "server block, reusing this project's existing qdrant server's QDRANT_URL/QDRANT_API_KEY."
        ),
        note=(
            "MEMORY_BANK_ID defaults to your collection name -- customize it in .mcp.json if you want a "
            "different per-project tag for recall scoping."
        ),
        detect=_detect_memory_bank_server_missing,
        apply=_apply_memory_bank_server_missing,
    ),
    # Placed next to memory-bank-server-missing purely for display grouping --
    # there is no ordering dependency: when both are pending no memory-bank
    # block exists yet, so either order yields the same "memory-bank" default
    # in both places.
    Migration(
        id="codebase-indexer-memory-bank-collection-missing",
        title="Add MEMORY_BANK_COLLECTION to codebase-indexer server env",
        kind="add",
        description=(
            "This project's codebase-indexer server has no MEMORY_BANK_COLLECTION in its env, so "
            "index_repo/sync_repo's guard against touching the shared memory-bank collection only "
            "protects the default name -- unprotected if your memory-bank server uses a custom one. "
            "Adds the key, copied from the memory-bank server's own value if present, "
            'else "memory-bank".'
        ),
        note="",
        detect=_detect_indexer_memory_bank_collection_missing,
        apply=_apply_indexer_memory_bank_collection_missing,
    ),
    Migration(
        id="record-session-id-hooks-missing",
        title="Add record_session_id.py hooks to .claude/settings.json",
        kind="add",
        description=(
            "No hook entry in this project's .claude/settings.json references record_session_id.py for "
            "SessionStart or SessionEnd -- the CORE/base-install hook that keeps "
            "libs/session_id_lib.py's SHADOW_FILE session-id strategy populated isn't running here yet. "
            "Adds the SessionStart + SessionEnd blocks; any other existing hooks (including local-compress's "
            "own) are left completely untouched."
        ),
        note="",
        detect=_detect_record_session_id_hooks_missing,
        apply=_apply_record_session_id_hooks_missing,
    ),
    Migration(
        id="qdrant-server-replaced",
        title="Run this repo's own qdrant MCP server instead of mcp-server-qdrant",
        kind="update",
        description=(
            "The 'qdrant' server in .mcp.json still launches the third-party mcp-server-qdrant console "
            "script. This repo now ships its own (mcpservers/qdrant_mcp_server.py): same qdrant-find and "
            "qdrant-store tools under the same names, plus the conversation-compact tools "
            "(compact_store, compact_find, compact_prune) that /my-compact and /my-resume use, which "
            "have moved there from local-compress. Rewrites the qdrant block to the new command and env, "
            "keeping your Qdrant URL/key, collection and embedding model, and carrying over the compact "
            "collection, LM Studio URL/model and compact embedding model from the local-compress block; "
            "drops HF_HUB_OFFLINE, FASTEMBED_CACHE_PATH and the TOOL_*_DESCRIPTION entries, which the "
            "new server no longer reads."
        ),
        note=(
            "Restart Claude Code to load the new server. Your stored compacts stay readable: "
            "COMPACT_EMBEDDING_MODEL is pinned to the model they were stored with (BAAI/bge-small-en for "
            "installs that never set one). Do NOT change it once you have saved compacts. Re-run "
            "`claude-runway-setup init` to also drop the compact settings the local-compress block no "
            "longer uses; `/my-compact` and `/my-resume` copies installed under ~/.claude/skills/ are "
            "refreshed by `upgrade` itself."
        ),
        detect=_detect_qdrant_server_replaced,
        apply=_apply_qdrant_server_replaced,
    ),
    Migration(
        id="record-session-id-sessionstart",
        title="Remove stale PostToolUse '.*' record_session_id.py hook from .claude/settings.json",
        kind="remove",
        description=(
            "This project's .claude/settings.json has the old PostToolUse '.*' registration for "
            "record_session_id.py, which fires a Python subprocess (~400ms on Windows) on "
            "EVERY tool call. It was replaced by a SessionStart hook that fires exactly once per "
            "session lifecycle event (startup/resume/clear/compact/fork). Removes the stale '.*' block; "
            "any other PostToolUse hooks are left completely untouched. Run 'claude-runway-setup upgrade' "
            "first to add the SessionStart block if it isn't present yet."
        ),
        note=(
            "If the record-session-id-hooks-missing migration is also pending in the same run, apply "
            "it first (it appears earlier in this list, so auto-yes handles ordering automatically). "
            "If you decline record-session-id-hooks-missing, this migration becomes a safe no-op -- "
            "it will not remove the PostToolUse fallback until a SessionStart registration is in place."
        ),
        detect=_detect_record_session_id_posttooluse_wildcard,
        apply=_apply_record_session_id_posttooluse_wildcard,
    ),
    Migration(
        id="compress-hook-renamed",
        title="Point the compress hook at hooks/compress_output.py",
        kind="update",
        description=(
            "This project's .claude/settings.json still runs hooks/compress_bash_output.py, which was "
            "renamed to hooks/compress_output.py -- the old path no longer exists, so "
            "output compression silently stops working. Rewrites just the script filename in each such "
            "hook; the venv path, tools-repo directory, matcher, and every other hook are left untouched."
        ),
        note="",
        detect=_detect_compress_hook_renamed,
        apply=_apply_compress_hook_renamed,
    ),
    Migration(
        id="compress-hook-matcher-covers-compact-tools",
        title="Make the compress hook matcher cover the compact tools on the qdrant server",
        kind="update",
        description=(
            "This project's .claude/settings.json compress hook matches the local-compress tools but not "
            "mcp__qdrant__compact_store/compact_find/compact_prune, where the conversation-compact tools now "
            "live, so their size-based compression stops after the qdrant server moves. Adds that "
            "alternative to the matcher right after mcp__local-compress__.*; every other part of the matcher, "
            "and every other hook, is left untouched. A matcher you wrote by hand (without "
            "mcp__local-compress__.*) is never edited."
        ),
        note="",
        detect=_detect_compress_hook_matcher_missing_compact_tools,
        apply=_apply_compress_hook_matcher_missing_compact_tools,
    ),
    Migration(
        id="analytics-server-missing",
        title="Add the opt-in analytics MCP server to .mcp.json",
        kind="add",
        description=(
            "This project tracks savings (CLAUDE_RUNWAY_TRACK_SAVINGS is on in its local-compress block), "
            "but its .mcp.json has no 'analytics' server. The savings and metrics tools (savings_summary, "
            "savings_detail, savings_trend, get_metrics, record_metric) have moved there from "
            "local-compress, so /my-savings and /my-metrics stop working until it is added. Adds the "
            "analytics block, carrying over any CLAUDE_RUNWAY_SAVINGS_DB override from the "
            "local-compress block and any CLAUDE_RUNWAY_METRICS_DB override from the memory-bank "
            "(or, in an older config, local-compress) block. Recording is unaffected either way: the "
            "hooks still write savings events without it."
        ),
        note=(
            "Restart Claude Code to load the new server. A project that does not track savings is not "
            "offered this migration; an interactive `upgrade` asks about the tracker first, and "
            "`init --track-savings` writes the block for a new setup."
        ),
        detect=_detect_analytics_server_missing,
        apply=_apply_analytics_server_missing,
    ),
    # Must stay AFTER analytics-server-missing: that one reads this key to
    # carry an override over, and this one's apply is a no-op until it has.
    Migration(
        id="local-compress-savings-db-inert",
        title="Remove the unused CLAUDE_RUNWAY_SAVINGS_DB from the local-compress server env",
        kind="remove",
        description=(
            "This project's local-compress block still sets CLAUDE_RUNWAY_SAVINGS_DB, which that server "
            "does not read: it never opens the savings database. The hooks write it (reading the shell "
            "export) and the analytics server reads it (from its own block). Removes just that key; "
            "CLAUDE_RUNWAY_TRACK_SAVINGS, which local-compress does read, stays."
        ),
        note=(
            "If the removed value was a non-default path, keep it in the analytics block and your shell "
            "export; `claude-runway-doctor` compares those two. If analytics-server-missing is also "
            "pending, this changes nothing until that one has carried the value over."
        ),
        detect=_detect_local_compress_savings_db_inert,
        apply=_apply_local_compress_savings_db_inert,
    ),
    Migration(
        id="mcp-server-scripts-moved",
        title="Point the MCP servers at mcpservers/ instead of tools/",
        kind="update",
        description=(
            "This project's .mcp.json launches one or more of the five MCP server scripts "
            "(qdrant, ingest, memory_bank, compress, analytics) from tools/, where they no longer "
            "are: they moved to mcpservers/ (same file names), so those servers fail to start until "
            "the path is rewritten. Rewrites just the script path in each such server's args from "
            ".../tools/<server>.py to .../mcpservers/<server>.py; the venv command, env values, "
            "every other server and .claude/settings.json are left untouched."
        ),
        note=(
            "Restart Claude Code to load the servers from their new location. A pipx or `uv tool` "
            "install needs reinstalling first (`pipx upgrade claude-runway` / `uv tool upgrade "
            "claude-runway`), since the old install has no mcpservers/ directory yet."
        ),
        detect=_detect_mcp_server_scripts_moved,
        apply=_apply_mcp_server_scripts_moved,
    ),
]


# Migrations on files inside the target repo, shown and applied after every
# config migration above.
REPO_MIGRATIONS: "list[RepoMigration]" = [
    RepoMigration(
        id="index-manifest-old-format",
        title="Remove the old-format sync manifest so the next sync_repo rebuilds the index",
        kind="remove",
        description=(
            "This repo has a .qdrant_index_manifest.json in the old format. Chunks are now cut by the "
            "embedding model's token window instead of by line count, and their position metadata is "
            "line_number + char_range instead of line_range, so every file has to be re-chunked and "
            "re-embedded. The manifest only records file content, not how files were chunked, so it cannot "
            "tell. Deletes it: the next sync_repo finds no manifest, rebuilds the whole index, and writes a "
            "new versioned one. Nothing is embedded or sent to Qdrant by this command."
        ),
        note=(
            "Cost: the first sync_repo afterwards re-embeds the WHOLE repo in one long call, and the "
            "collection ends up with several times as many chunks as before -- more for minified or "
            "generated files, which are indexed in full with no per-file cap. It reports per-file "
            "progress and names the files that produced the most chunks. Run it when you are ready. "
            "A file that is still on disk but outside that first rebuild's scope keeps its old chunks "
            "and is recorded as kept in the new manifest, so later syncs remove its chunks if it is "
            "deleted."
        ),
        detect=_detect_old_format_manifest,
        apply=_apply_old_format_manifest,
    ),
]


def pending_repo_migrations(target_repo: "Path | str") -> "list[RepoMigration]":
    """Pure helper: every repo-file migration whose `detect` currently returns
    True for `target_repo`, in `REPO_MIGRATIONS`' order."""
    return [m for m in REPO_MIGRATIONS if m.detect(Path(target_repo))]


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
    happily writes a BROKEN partial config: found in Copilot review and confirmed live,
    `upgrade <empty-dir> --auto-yes` wrote a
    `qdrant` server block with an `env` but no `command`/`type` at all --
    plus a default `memory-bank` block, for a directory that had never been
    `init`'d.
    """
    mcp_json_path = Path(target_repo) / ".mcp.json"
    if not mcp_json_path.exists():
        return False
    return _has_qdrant_server(load_existing_mcp_json(mcp_json_path))


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


def _format_migration_block(migration: "Migration | RepoMigration", index: int, total: int) -> str:
    lines = [f"[{index}/{total}] {migration.kind.upper()}: {migration.title}", f"      {migration.description}"]
    if migration.note:
        lines.append(f"      Note: {migration.note}")
    return "\n".join(lines)


def _console_prompt(migration: "Migration | RepoMigration", index: int, total: int) -> bool:
    """Default, real interactive prompt -- prints this migration's block
    then reads a Y/n answer from stdin. Blank input (just pressing Enter)
    defaults to Yes, matching the issue's own illustrated "Apply? [Y/n]"."""
    print()
    print(_format_migration_block(migration, index, total))
    answer = input("      Apply? [Y/n] ").strip().lower()
    return answer not in ("n", "no")


def _savings_offer_applies(mcp_json: dict) -> bool:
    """
    True when the savings tracker could be offered: the project has
    a `local-compress` block (the tracker needs it; a `--qdrant-only` project
    deliberately has none), no `analytics` block, and tracking is not already
    on. A project with tracking already on is not asked: it is simply missing
    the block, and `analytics-server-missing` offers that on its own.
    """
    servers = mcp_json.get("mcpServers")
    block = servers.get("local-compress") if isinstance(servers, dict) else None
    return (
        isinstance(block, dict)
        and not has_analytics_block(mcp_json)
        and not local_compress_tracks_savings(mcp_json)
    )


def _enable_savings_tracking(mcp_json: dict) -> dict:
    """
    A copy of `mcp_json` with `CLAUDE_RUNWAY_TRACK_SAVINGS=1` in the
    `local-compress` env block -- exactly what `init --track-savings` writes
    there. That makes `analytics-server-missing` detect the project, so the
    `analytics` block itself is added by that migration (the one function,
    `build_mcp_servers`, that `init` also uses), not by a second code path.
    """
    mcp_json = copy.deepcopy(mcp_json)
    block = mcp_json["mcpServers"]["local-compress"]
    if not isinstance(block.get("env"), dict):
        block["env"] = {}
    block["env"]["CLAUDE_RUNWAY_TRACK_SAVINGS"] = "1"
    return mcp_json


# The offer's id in `run_upgrade`'s returned list. Not a `Migration`: those are
# detected from the config alone and reappear on every run until applied, which
# would nag every project that never wants a local review loop. This is asked
# once per interactive run instead, like the savings tracker.
CODEBASE_REVIEWER_OFFER_ID = "codebase-reviewer-server-offer"


def _reviewer_offer_applies(mcp_json: dict) -> bool:
    """True when the `codebase-reviewer` server could be offered: the project has
    none yet."""
    return not has_codebase_reviewer_block(mcp_json)


def _add_codebase_reviewer_block(mcp_json: dict, ctx: MigrationContext) -> dict:
    """
    A copy of `mcp_json` with the `codebase-reviewer` block added through
    `build_mcp_servers` (the one function `init --codebase-reviewer` also uses,
    so the two cannot drift). `REVIEWERS_FILE` is left BLANK: the server
    requires it and has no default, and no default path or reviewers file is
    ever written. The block's Qdrant URL and key follow the project's own
    `qdrant` block; every other server and `settings.json` are untouched.
    """
    mcp_json = copy.deepcopy(mcp_json)
    servers = mcp_json.setdefault("mcpServers", {})
    qdrant_env = servers.get("qdrant", {}).get("env") if isinstance(servers.get("qdrant"), dict) else None
    template = load_json(Path(ctx.tools_repo_dir) / "templates" / "mcp.json.template")
    generated = build_mcp_servers(
        template,
        collection_name="project",
        venv_python=ctx.venv_python,
        tools_repo_dir=ctx.tools_repo_dir,
        home_dir=Path.home(),
        qdrant_url=_env_str(qdrant_env, "QDRANT_URL").strip() or "http://localhost:6333",
        qdrant_api_key=_env_str(qdrant_env, "QDRANT_API_KEY"),
        codebase_reviewer=True,
    )
    servers["codebase-reviewer"] = generated["codebase-reviewer"]
    return mcp_json


def run_upgrade(
    target_repo: "Path | str",
    tools_repo_dir: "Path | str",
    venv_python: "Path | str",
    *,
    auto_yes: bool = False,
    dry_run: bool = False,
    prompt_fn: Optional[Callable[["Migration | RepoMigration", int, int], bool]] = None,
    offer_savings_fn: Optional[Callable[[], bool]] = None,
    offer_reviewer_fn: Optional[Callable[[], bool]] = None,
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

    `offer_savings_fn` is the interactive "enable the savings
    tracker?" question for a project that has never tracked savings and has no
    `analytics` server, so `/my-savings` and `/my-metrics` would fail with an
    unknown-tool error and nothing would say why. It is asked at most once,
    before the migrations are listed, and only when the caller passes one
    AND neither `dry_run` nor `auto_yes` is set: `None` (the default), a
    dry run and `--auto-yes` all keep today's behavior of leaving the tracker
    off. A yes turns tracking on in the `local-compress` block and applies
    `analytics-server-missing` without asking a second time; the rest of the
    flow is unchanged.

    `offer_reviewer_fn` is the same kind of one-time interactive question for
    the opt-in `codebase-reviewer` server, asked (after the savings one) only
    when the project has no such block, the caller passes one, and neither
    `dry_run` nor `auto_yes` is set. A yes adds the block with `REVIEWERS_FILE`
    BLANK and prints a warning that the setting is required and has no default:
    no path and no reviewers file are ever written. The change is written even
    when no migration is pending, and `CODEBASE_REVIEWER_OFFER_ID` is added to
    the returned list.

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
    mcp_json = load_existing_mcp_json(mcp_json_path)
    settings_json = load_existing_settings_json(settings_path)

    print(f"Checking {target_repo} for pending upgrades...")

    if not _has_qdrant_server(mcp_json):
        print(
            f"\nerror: no existing toolkit configuration found at {mcp_json_path} (no 'qdrant' MCP "
            "server block) -- `upgrade` only applies migrations to an ALREADY-configured project. "
            "Run `init` first, then `upgrade` to pick up anything released since."
        )
        return []

    savings_offer_accepted = False
    if (
        offer_savings_fn is not None
        and not dry_run
        and not auto_yes
        and _savings_offer_applies(mcp_json)
        and offer_savings_fn()
    ):
        mcp_json = _enable_savings_tracking(mcp_json)
        savings_offer_accepted = True

    reviewer_offer_accepted = False
    if (
        offer_reviewer_fn is not None
        and not dry_run
        and not auto_yes
        and _reviewer_offer_applies(mcp_json)
        and offer_reviewer_fn()
    ):
        mcp_json = _add_codebase_reviewer_block(mcp_json, ctx)
        reviewer_offer_accepted = True

    pending = pending_migrations(mcp_json, settings_json)
    repo_pending = pending_repo_migrations(target_repo)
    total = len(pending) + len(repo_pending)

    if not total:
        if reviewer_offer_accepted:
            # The offer changed .mcp.json with no migration to carry the write.
            _atomic_write_json(mcp_json_path, mcp_json)
            print("\nAdded the codebase-reviewer MCP server to .mcp.json.")
            print(f"WARNING: {REVIEWERS_FILE_REQUIRED_WARNING}")
            return [CODEBASE_REVIEWER_OFFER_ID]
        print("\nNo pending upgrades -- .mcp.json/.claude/settings.json already reflect every known migration.")
        return []

    print(f"\nFound {total} pending upgrade(s):")

    ask = prompt_fn or _console_prompt
    applied: "list[str]" = []
    skipped = 0
    noop = 0
    # The accepted offer changed mcp_json before any migration ran. The
    # analytics migration it triggers writes it again, but keep the flag
    # honest in case that migration ever stops detecting it.
    mcp_dirty = savings_offer_accepted or reviewer_offer_accepted
    settings_dirty = False
    applied_offer: "list[str]" = [CODEBASE_REVIEWER_OFFER_ID] if reviewer_offer_accepted else []
    for index, migration in enumerate(pending, start=1):
        if dry_run:
            print()
            print(_format_migration_block(migration, index, total))
            applied.append(migration.id)
            continue
        if auto_yes:
            print()
            print(_format_migration_block(migration, index, total))
            print("      Auto-applying (--auto-yes).")
            accept = True
        elif savings_offer_accepted and migration.id == "analytics-server-missing":
            # The user just said yes to exactly this; do not ask again.
            print()
            print(_format_migration_block(migration, index, total))
            print("      Applying (you chose to enable the savings tracker).")
            accept = True
        else:
            accept = ask(migration, index, total)
        if not accept:
            skipped += 1
            continue
        new_mcp, new_settings = migration.apply(mcp_json, settings_json, ctx)
        # Diff each document against its pre-apply value so we know what this
        # migration ACTUALLY touched. Migrations deep-copy before
        # mutating, so comparing by value is reliable.
        mcp_changed = new_mcp != mcp_json
        settings_changed = new_settings != settings_json
        if not (mcp_changed or settings_changed):
            # A guarded no-op apply (e.g. record-session-id-sessionstart when
            # its prerequisite migration was declined) changed nothing: don't
            # report it as applied -- it is still pending and will be detected
            # again next run, matching the module docstring's contract.
            noop += 1
            continue
        # An apply() can deliberately no-op when its safety guard isn't met
        # (e.g. record-session-id-sessionstart won't remove the PostToolUse
        # fallback while no SessionStart block exists). Only count it as
        # applied if detect() agrees the gap is actually closed now. Checked
        # before adopting new_mcp/new_settings so a half-applied migration
        # never leaks into a document another migration later writes.
        if migration.detect(new_mcp, new_settings):
            print(f"      Not applied: its precondition isn't met yet, so {migration.id} stays pending.")
            skipped += 1
            continue
        mcp_json, settings_json = new_mcp, new_settings
        mcp_dirty = mcp_dirty or mcp_changed
        settings_dirty = settings_dirty or settings_changed
        applied.append(migration.id)

    # Write only documents some applied migration actually changed:
    # an unconditional both-file write would materialize a settings.json a
    # --skip-hooks project opted out of, and re-indent an untouched .mcp.json.
    if not dry_run:
        if mcp_dirty:
            _atomic_write_json(mcp_json_path, mcp_json)
        if settings_dirty:
            _atomic_write_json(settings_path, settings_json)

    # Repo-file migrations come after the config ones, numbered on
    # from them. They act on disk, so the config files are already written.
    for index, repo_migration in enumerate(repo_pending, start=len(pending) + 1):
        if dry_run:
            print()
            print(_format_migration_block(repo_migration, index, total))
            applied.append(repo_migration.id)
            continue
        if auto_yes:
            print()
            print(_format_migration_block(repo_migration, index, total))
            print("      Auto-applying (--auto-yes).")
            accept = True
        else:
            accept = ask(repo_migration, index, total)
        if not accept:
            skipped += 1
            continue
        repo_migration.apply(target_repo)
        applied.append(repo_migration.id)

    if dry_run:
        print(f"\nDry run: no changes written. {total} pending upgrade(s) would be applied.")
        return applied

    if savings_offer_accepted:
        print(f"\nSavings tracker: {SAVINGS_TRACKER_SHELL_EXPORT_REMINDER}")
    if reviewer_offer_accepted:
        print(f"\nAdded the codebase-reviewer MCP server to .mcp.json.\nWARNING: {REVIEWERS_FILE_REQUIRED_WARNING}")
        applied = applied_offer + applied

    noop_note = f", {noop} had nothing to change (still pending)" if noop else ""
    print(f"\nDone. {len(applied)} change(s) applied, {skipped} skipped{noop_note}.")
    return applied
