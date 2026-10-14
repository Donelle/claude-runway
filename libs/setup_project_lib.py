"""
Shared, dependency-free logic behind `tools/setup_project.py` -- computing
defaults and patching `templates/mcp.json.template` /
`templates/settings.json.template` into a real, ready-to-use `.mcp.json` /
`.claude/settings.json` for a target project repo, instead of the manual
copy-and-hand-edit flow docs/installation.md's steps 4/4b/4c otherwise require (step 5's manual
CLAUDE.md edit is NOT covered here)
(issue #48). Kept separate from `tools/setup_project.py` (the thin CLI) the
same way `qdrant_ingest_lib.py`/`local_compress_lib.py` are kept separate
from their own CLI/MCP-server callers -- pure functions here are what make
this testable without touching disk beyond reading the template files
themselves.

Deliberately patches KNOWN fields structurally (dict/list traversal) rather
than doing a blind string find-and-replace of the placeholder tokens over
the raw template text, so each field gets exactly the value meant for it
(and `find_unresolved_placeholders` can prove nothing was missed).
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def default_collection_name(repo_path: Path) -> str:
    """
    Derive a Qdrant collection name suggestion from the target repo's own
    directory name, e.g. "My Cool App!" -> "my-cool-app".

    This is only a convenience DEFAULT, not a collision-proofed identifier.
    tools/qdrant_mcp_server.py's `_sanitize_project` is not one either: it
    appends no hash, so names differing only in punctuation/whitespace map to
    the same conversation-compacts collection, and per-project isolation there
    comes from payload filtering on read/prune. Here the collection name is a
    one-time, visible choice a human reviews in the generated `.mcp.json`
    before ever committing it -- so a plain, readable slug is what we want. Pass
    `--collection-name` explicitly if this default would collide with an
    already-indexed project's collection on the same Qdrant instance.
    """
    name = repo_path.resolve().name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name or "project"


def venv_python_path(tools_repo_dir: Path, *, windows: bool = False) -> Path:
    """
    Deterministic path to the tools-repo's OWN venv python interpreter,
    matching the layout `docs/installation.md`'s step 3 creates
    (`uv venv --python 3.12` at the tools-repo root). Takes `windows`
    as an explicit parameter rather than reading `os.name`/`sys.platform`
    directly so this stays a pure function callers (including tests) can
    exercise for either platform without patching a global.
    """
    if windows:
        return tools_repo_dir / ".venv" / "Scripts" / "python.exe"
    return tools_repo_dir / ".venv" / "bin" / "python"


def load_json(path: Path) -> dict:
    """Loads a JSON file (a template, or an already-existing target config)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_PLACEHOLDER_MARKERS = ("REPLACE-WITH-", "/absolute/path/to/")


def find_unresolved_placeholders(value: object) -> list:
    """
    Recursively walks a dict/list/str structure for any string that still
    contains an unresolved REPLACE-WITH-*/`/absolute/path/to/...` marker.
    Used as a correctness check right after patching (a field this module
    forgot to patch would otherwise silently ship a broken generated config,
    indistinguishable at a glance from a hand-edit mistake) and directly by
    tests.
    """
    found: list = []

    def _walk(v: object) -> None:
        if isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, list):
            for x in v:
                _walk(x)
        elif isinstance(v, str):
            if any(marker in v for marker in _PLACEHOLDER_MARKERS):
                found.append(v)

    _walk(value)
    return found


def build_mcp_servers(
    template: dict,
    *,
    collection_name: str,
    venv_python: Path,
    tools_repo_dir: Path,
    home_dir: Path,
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: str = "",
    collection_description: str = "",
    include_extensions: str = "",
    exclude_dirs: str = "",
    lmstudio_url: str = "http://localhost:1234/v1",
    lmstudio_model: str = "",
    track_savings: bool = False,
    savings_db: str = "",
    metrics_db: str = "",
    compact_collection: str = "",
    memory_bank_collection: str = "",
    memory_bank_id: str = "",
    include_compress: bool = True,
    embedding_model: str = "",
    memory_bank_embedding_model: str = "",
    compact_embedding_model: str = "",
) -> dict:
    """
    Returns a fresh `mcpServers` dict (deep-copied from `template`, never
    mutated in place) with every placeholder in the qdrant/codebase-indexer/
    memory-bank/local-compress blocks resolved to a real value. Paths are emitted as
    forward-slash (`Path.as_posix()`) to match the templates' own style,
    which uses forward slashes even in its Windows-facing example paths.

    `compact_collection`, if blank, defaults to the template's own historical
    "conversation-compacts" -- a single bucket shared by every project on
    the machine. This is a shared base prefix: qdrant_mcp_server.py
    appends the sanitized project name at runtime, and per-project isolation
    comes from its payload filtering on read/prune, not from the name being
    unique (see `_collections_for_project`'s sibling-scan there, which finds other
    collections for the same project by matching this exact prefix).
    Deliberately NOT derived from `collection_name` here the way
    `COLLECTION_NAME` on the qdrant/codebase-indexer servers is: found in
    PR #130 review that doing so silently changes this value on EVERY
    already-configured project the moment its `.mcp.json` gets regenerated
    (a plain re-run with no new flags, the exact "safe to re-run" workflow
    this script's own module docstring advertises) -- since every project's
    `COMPACT_COLLECTION` (then in the `local-compress` env, now in the `qdrant`
    env) has only ever held this one constant so far (this function never
    touched the key at all before PR #130), and `merge_mcp_json` replaces the
    whole server block wholesale, not a per-key deep merge. Changing the *base prefix* silently
    orphans that project's entire conversation-compact history: the
    sibling-scan and canonical-name computation both key off this exact
    string, so the moment it changes, `/my-resume` can no longer find
    anything stored under the old prefix -- not an error, just silently
    invisible data going forward. A genuinely per-project default
    (`<collection_name>-conversation`) is still worth having for NEW setups
    -- that's exactly what `/my-setup-clauderunway`'s own Q5 computes and
    passes explicitly via `--compact-collection`, the same way Q2 always
    explicitly passes `--collection-name` rather than leaving it to this
    function's own bare fallback. Direct CLI users bypassing the skill keep
    getting the shared historical default unless they opt in explicitly --
    consistent with every other option here (`--qdrant-url`/`--collection-
    description`/etc.), none of which is "sticky" either; see this
    docstring's own `qdrant_api_key` paragraph below.

    Whatever `compact_collection` resolves to is deliberately never left
    blank in the generated env block, regardless of source: relying on
    `os.environ.get("COMPACT_COLLECTION", "conversation-compacts")`'s own
    fallback in qdrant_mcp_server.py isn't enough on its own, since that
    fallback only fires when the key is genuinely ABSENT, not when it's
    present-but-empty -- an explicit `.mcp.json` env entry set to `""`
    would otherwise build a collection name with a leading stray hyphen
    (`-<sanitized-project>`).

    `memory_bank_collection`, if blank, defaults to the template's own
    literal `"memory-bank"` -- the ONE collection every project's
    memory-bank server shares (issue #175), deliberately NOT derived from
    `collection_name` the same way `COLLECTION_NAME` is: every project on a
    machine needs to agree on this exact string for their memories to
    actually be shared/found across projects, so treating it as per-project
    the way `collection_name` is would defeat the whole point. Never left
    blank in the generated env block for the same reason `compact_collection`
    isn't (see that paragraph above) -- `os.environ.get("MEMORY_BANK_COLLECTION",
    "memory-bank")`'s own fallback only fires when the key is genuinely
    ABSENT, not present-but-empty. Applied to BOTH the `codebase-indexer`
    block (which needs to know this value purely to refuse ever indexing
    code into it) and the `memory-bank` block itself.

    `memory_bank_id`, if blank, defaults to `collection_name` -- a
    convenient one-time default for a brand-new project, since most
    projects have no reason to want it different. But unlike
    `memory_bank_collection` above, this IS meant to vary per project (it's
    the per-project identity tag, not the shared storage location) -- found
    in PR #178 review: an earlier version hardcoded this to always equal
    `collection_name` with no independent parameter at all, so a project
    that later changed `--collection-name` would silently retag all FUTURE
    memories under the new name while every EXISTING memory stayed under
    the old one, becoming invisible to default recall/wipe scope. Passing
    `memory_bank_id` explicitly on a re-run (the caller's job -- this
    function doesn't detect re-runs itself, same as every other setting
    here) preserves the existing identity independent of whatever
    `collection_name` resolves to on that run.

    `qdrant_api_key` is applied to the three Qdrant-talking servers'
    `QDRANT_API_KEY` (`qdrant`, `codebase-indexer` and `memory-bank`, which
    all read the exact same variable so a single project's `.mcp.json` keeps
    every one of them aligned). Found in PR #113 review: since this script
    previously always replaced these servers' `env` blocks wholesale with
    values it computed, an authenticated remote-Qdrant setup's
    manually-added `QDRANT_API_KEY` was silently dropped on every rerun. Passing `--qdrant-api-key` explicitly on every
    run (the same pattern already used for `--qdrant-url`/`--collection-
    name`/etc. -- none of this CLI's other options are "sticky" either) is
    what keeps it from reverting.

    `compact_embedding_model` is the `qdrant` block's `COMPACT_EMBEDDING_MODEL`
    (it was the `local-compress` block's `EMBEDDING_MODEL` before the compact
    tools moved), deliberately separate from `embedding_model`: compacts are
    stored under a model-named vector, so changing it orphans an existing
    install's compacts. Blank leaves the template default. `run_setup` decides
    what to pass (see `resolve_compact_embedding_model`).

    `track_savings` also decides whether the opt-in `analytics` block is
    emitted at all (it is dropped from the result otherwise); `savings_db` and
    `metrics_db` reach that block only when non-blank. A non-blank
    `metrics_db` is also written into the `memory-bank` block, which writes
    into the same shared store, so the two stay on one file.

    The `qdrant` block also carries `lmstudio_url`/`lmstudio_model`, which
    only `compact_store` uses (to compress a summary before storing it), so a
    `--qdrant-only` setup (`include_compress=False`) still writes them: the
    server stores uncompressed when LM Studio is not there.

    `embedding_model` (issue #279), if blank, leaves the template's own
    `EMBEDDING_MODEL` default in place; otherwise it is written into the two
    code-index blocks (`qdrant`, `codebase-indexer`), which must agree on one
    model. It is never written into `memory-bank` (issue #428): that block's
    model is `memory_bank_embedding_model`, a separate value. Like
    everything else here this function does not detect re-runs itself:
    deciding WHICH value to pass (explicit flag > the target's existing
    config > template default) is `run_setup`'s job, see
    `resolve_embedding_model`.
    """
    servers = copy.deepcopy(template["mcpServers"])
    venv_python_str = venv_python.as_posix()
    # Deliberately NOT .resolve()'d here: callers (run_setup/the CLI) are
    # responsible for passing an already-absolute tools_repo_dir. Calling
    # .resolve() on it would additionally walk symlinks/synthetic mount
    # points -- harmless for a real existing directory in the common case,
    # but confirmed live to silently rewrite a path through macOS's "/home"
    # synthetic firmlink into "/System/Volumes/Data/home/..." on a machine
    # where that top-level name isn't a real user directory. Staying a
    # simple, predictable string join avoids that surprise entirely.
    tools_repo_str = Path(tools_repo_dir).as_posix()

    qdrant = servers["qdrant"]
    qdrant["command"] = venv_python_str
    qdrant["args"] = [f"{tools_repo_str}/tools/qdrant_mcp_server.py"]
    qdrant["env"]["QDRANT_URL"] = qdrant_url
    qdrant["env"]["QDRANT_API_KEY"] = qdrant_api_key
    qdrant["env"]["COLLECTION_NAME"] = collection_name
    qdrant["env"]["COMPACT_COLLECTION"] = compact_collection or "conversation-compacts"
    qdrant["env"]["CLAUDE_RUNWAY_LMSTUDIO_URL"] = lmstudio_url
    qdrant["env"]["CLAUDE_RUNWAY_LMSTUDIO_MODEL"] = lmstudio_model
    # The compact embedding model is its own value, never blindly the
    # template's: changing it orphans existing compacts. Blank leaves the
    # template default (a brand-new install).
    if compact_embedding_model:
        qdrant["env"]["COMPACT_EMBEDDING_MODEL"] = compact_embedding_model

    resolved_memory_bank_collection = memory_bank_collection or "memory-bank"
    resolved_memory_bank_id = memory_bank_id or collection_name

    indexer = servers["codebase-indexer"]
    indexer["command"] = venv_python_str
    indexer["args"] = [f"{tools_repo_str}/tools/ingest_mcp_server.py"]
    indexer["env"]["QDRANT_URL"] = qdrant_url
    indexer["env"]["QDRANT_API_KEY"] = qdrant_api_key
    indexer["env"]["COLLECTION_NAME"] = collection_name
    indexer["env"]["COLLECTION_DESCRIPTION"] = collection_description
    indexer["env"]["INDEX_INCLUDE_EXTENSIONS"] = include_extensions
    indexer["env"]["INDEX_EXCLUDE_DIRS"] = exclude_dirs
    indexer["env"]["MEMORY_BANK_COLLECTION"] = resolved_memory_bank_collection

    memory_bank = servers["memory-bank"]
    memory_bank["command"] = venv_python_str
    memory_bank["args"] = [f"{tools_repo_str}/tools/memory_bank_mcp_server.py"]
    memory_bank["env"]["QDRANT_URL"] = qdrant_url
    memory_bank["env"]["QDRANT_API_KEY"] = qdrant_api_key
    memory_bank["env"]["MEMORY_BANK_ID"] = resolved_memory_bank_id
    memory_bank["env"]["MEMORY_BANK_COLLECTION"] = resolved_memory_bank_collection

    # Issue #428: memory-bank's model is its own value, never derived from
    # `embedding_model`. The memory-bank collection is ONE shared collection
    # locked to the model that first created it, so a change made for the
    # per-project code collections must never reach it.
    # `memory-bank` writes its call tally (metric_id="memory-bank") into the same
    # shared metrics database the `analytics` server reads, so a non-default
    # path has to reach both blocks, or a launch without the shell export
    # (a GUI, say) leaves the two on different files and `/my-metrics
    # memory-bank` silently reports partial data. Written only when given (a
    # blank value would mask an inherited shell value), and regardless of
    # `track_savings`, since the memory-bank server needs it either way.
    if metrics_db:
        memory_bank["env"]["CLAUDE_RUNWAY_METRICS_DB"] = metrics_db

    if embedding_model:
        for block in (qdrant, indexer):
            block["env"]["EMBEDDING_MODEL"] = embedding_model
    if memory_bank_embedding_model:
        memory_bank["env"]["EMBEDDING_MODEL"] = memory_bank_embedding_model

    if include_compress:
        compress = servers["local-compress"]
        compress["command"] = venv_python_str
        compress["args"] = [f"{tools_repo_str}/tools/compress_mcp_server.py"]
        compress["env"]["CLAUDE_RUNWAY_LMSTUDIO_URL"] = lmstudio_url
        compress["env"]["CLAUDE_RUNWAY_LMSTUDIO_MODEL"] = lmstudio_model
        compress["env"]["CLAUDE_RUNWAY_TRACK_SAVINGS"] = "1" if track_savings else ""
        compress["env"]["CLAUDE_RUNWAY_SAVINGS_DB"] = savings_db
    else:
        del servers["local-compress"]

    # The `analytics` server (savings_summary/savings_detail/savings_trend,
    # get_metrics, record_metric) is opt-in: it is written only when savings
    # tracking is on, and only alongside local-compress (the tracker records
    # compression events, which need that server and its hooks). Both DB keys
    # are written only when overridden: a blank value would mask the
    # inherited shell value instead of falling through to the default path.
    if track_savings and include_compress:
        analytics = servers["analytics"]
        analytics["command"] = venv_python_str
        analytics["args"] = [f"{tools_repo_str}/tools/analytics_mcp_server.py"]
        analytics["env"]["CLAUDE_RUNWAY_TRACK_SAVINGS"] = "1"
        if savings_db:
            analytics["env"]["CLAUDE_RUNWAY_SAVINGS_DB"] = savings_db
        if metrics_db:
            analytics["env"]["CLAUDE_RUNWAY_METRICS_DB"] = metrics_db
    else:
        del servers["analytics"]

    return servers


# Code-index servers whose env carries `EMBEDDING_MODEL`, in the order
# `existing_embedding_model` consults them. `memory-bank` is deliberately NOT
# here (issue #428): its model belongs to a shared, cross-project collection
# and is resolved on its own, see `resolve_memory_bank_embedding_model`.
_EMBEDDING_MODEL_SERVERS = ("qdrant", "codebase-indexer")


def existing_embedding_model(existing_mcp: dict) -> str:
    """
    Returns the `EMBEDDING_MODEL` already configured in an existing
    `.mcp.json` document (first non-blank string found across the `qdrant`
    and `codebase-indexer` blocks -- never `memory-bank`'s, issue #428), or ""
    if none is set. Tolerant
    of missing/odd-shaped blocks -- a hand-edited file may lack any of them.
    """
    servers = existing_mcp.get("mcpServers") if isinstance(existing_mcp, dict) else None
    if not isinstance(servers, dict):
        return ""
    for name in _EMBEDDING_MODEL_SERVERS:
        block = servers.get(name)
        env = block.get("env") if isinstance(block, dict) else None
        value = env.get("EMBEDDING_MODEL") if isinstance(env, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_embedding_model(explicit: str, existing_mcp: dict, template_default: str) -> str:
    """
    The rule for which `EMBEDDING_MODEL` a (re-)run of `init` writes
    (issue #279), in priority order:
      1. an explicit `--embedding-model` value (the caller opted in);
      2. otherwise the model already configured in the target's existing
         `.mcp.json` -- so a plain re-run preserves a hand-customized model
         instead of resetting it to the template default, the same
         preserve-don't-reset behavior `upgrade` already applies;
      3. otherwise the template's default (a brand-new project).
    Invariant: a re-run never changes the model unless the caller passed it
    explicitly, because changing it invalidates every vector already indexed
    under the old model and breaks agreement with the shared memory-bank
    collection.
    """
    return (explicit or "").strip() or existing_embedding_model(existing_mcp) or template_default


def existing_memory_bank_model(existing_mcp: dict) -> str:
    """
    Returns the `EMBEDDING_MODEL` in an existing `.mcp.json`'s `memory-bank`
    block, or "" if there is no such block or it has no non-blank value.
    Tolerant of missing/odd-shaped blocks, like `existing_embedding_model`.
    """
    servers = existing_mcp.get("mcpServers") if isinstance(existing_mcp, dict) else None
    block = servers.get("memory-bank") if isinstance(servers, dict) else None
    env = block.get("env") if isinstance(block, dict) else None
    value = env.get("EMBEDDING_MODEL") if isinstance(env, dict) else None
    return value.strip() if isinstance(value, str) else ""


def resolve_memory_bank_embedding_model(explicit: str, existing_mcp: dict, template_default: str) -> str:
    """
    Which `EMBEDDING_MODEL` a (re-)run of `init` writes into `memory-bank`
    (issue #428), from memory-bank's own sources only -- never the code-index
    model, because the shared collection is locked to whichever model created
    it and a change made for code search must not make `remember`/`recall`
    fail for every project:
      1. an explicit `--memory-bank-embedding-model` value;
      2. otherwise the existing `memory-bank` block's own value;
      3. otherwise the template's default (a brand-new install, or a config
         with no memory-bank block).
    """
    return (explicit or "").strip() or existing_memory_bank_model(existing_mcp) or template_default


# What an install that predates the compact embedding-model setting actually
# stored its compacts with -- must match
# `qdrant_mcp_server.LEGACY_COMPACT_EMBEDDING_MODEL`. Full model id, never the
# short name.
LEGACY_COMPACT_EMBEDDING_MODEL = "BAAI/bge-small-en"


def _server_env(existing_mcp: dict, name: str) -> "Optional[dict]":
    """The `env` dict of the named server block in an existing `.mcp.json`, or
    None when the block or its env is missing or oddly shaped."""
    servers = existing_mcp.get("mcpServers") if isinstance(existing_mcp, dict) else None
    block = servers.get(name) if isinstance(servers, dict) else None
    env = block.get("env") if isinstance(block, dict) else None
    return env if isinstance(env, dict) else None


def _non_blank(env: "Optional[dict]", key: str) -> str:
    value = env.get(key) if env is not None else None
    return value.strip() if isinstance(value, str) else ""


def resolve_compact_embedding_model(existing_mcp: dict, project_model: str) -> str:
    """
    Which `COMPACT_EMBEDDING_MODEL` a (re-)run of `init`, and the migration that
    moves the compact tools onto the `qdrant` server, writes. Unlike the
    code-index model there is no explicit-override flag: changing it only ever
    orphans stored compacts, so a re-run must never do that on its own:
      1. an existing `qdrant` block's own non-blank `COMPACT_EMBEDDING_MODEL`
         is kept (the compact tools already live there);
      2. otherwise, an existing `local-compress` block is where a pre-move
         install's compacts were configured: its non-blank `EMBEDDING_MODEL`
         is kept;
      3. otherwise, ANY existing toolkit config (a `qdrant` or `local-compress`
         block) is pinned to the legacy model. A missing `local-compress`
         block does not prove no compact was ever stored: a full setup later
         re-run with `--qdrant-only` deletes that block but leaves its
         compacts in Qdrant, and every compact stored before this setting
         existed was embedded with the legacy model. Guessing wrong the other
         way orphans them; guessing legacy for a project that never stored a
         compact costs nothing, since its first compact simply creates the
         collection under that model;
      4. only a config with neither block (genuinely empty, or holding just
         unrelated servers) uses the project model.
    """
    qdrant_value = _non_blank(_server_env(existing_mcp, "qdrant"), "COMPACT_EMBEDDING_MODEL")
    if qdrant_value:
        return qdrant_value
    compress_value = _non_blank(_server_env(existing_mcp, "local-compress"), "EMBEDDING_MODEL")
    if compress_value:
        return compress_value
    servers = existing_mcp.get("mcpServers") if isinstance(existing_mcp, dict) else None
    if isinstance(servers, dict) and any(isinstance(servers.get(name), dict) for name in ("qdrant", "local-compress")):
        return LEGACY_COMPACT_EMBEDDING_MODEL
    return project_model


# (event, script filename) for every hook this toolkit's templates define,
# split into two groups (issue #198, #231):
#   - _CORE_HOOK_SCRIPTS: written for EVERY project regardless of
#     --qdrant-only/local-compress config -- today, just record_session_id.py
#     (registered under SessionStart and SessionEnd), which keeps
#     libs/session_id_lib.py's SHADOW_FILE strategy actually populated.
#     SessionStart replaced PostToolUse (issue #231): the '.*' PostToolUse
#     matcher fired on EVERY tool call (~400ms Python startup × N calls on
#     Windows), while SessionStart fires exactly once per session lifecycle
#     event (startup/resume/clear/compact/fork).
#   - _COMPRESS_HOOK_SCRIPTS: only written when local-compress is configured
#     (include_compress=True below) -- the pre-existing three hooks.
# _TOOLKIT_HOOK_SCRIPTS (both groups combined) is the single source of truth
# for _is_toolkit_owned_hook below (which recognizes a block as ours by
# script filename alone, independent of the mutable path/matcher/array
# position around it) -- keeping every group derived from the same tuples is
# what stops them from silently drifting apart if a script is ever renamed
# or a new hook added.
_CORE_HOOK_SCRIPTS = (
    ("SessionStart", "record_session_id.py"),
    ("SessionEnd", "record_session_id.py"),
)
_COMPRESS_HOOK_SCRIPTS = (
    ("PostToolUse", "compress_output.py"),
    ("PreToolUse", "redirect_webfetch_to_fetch_url.py"),
    ("SessionEnd", "session_end_savings.py"),
)
_TOOLKIT_HOOK_SCRIPTS = _CORE_HOOK_SCRIPTS + _COMPRESS_HOOK_SCRIPTS

# The compress hook's pre-#395 filename. Existing installs' settings.json
# still reference it, so it must stay RECOGNIZED as toolkit-owned (otherwise
# `init` would treat that block as a user's own hook and append a second,
# new-named block beside it, double-compressing every tool call). It is
# deliberately NOT in _COMPRESS_HOOK_SCRIPTS: nothing generates it anymore.
# Shared with upgrade_lib (the rename migration) and doctor_lib (the stale-
# registration check) so the three can't disagree about the old name.
LEGACY_COMPRESS_HOOK_SCRIPT = "compress_bash_output.py"
COMPRESS_HOOK_SCRIPT = "compress_output.py"


def _find_and_patch_block(blocks: list, script_name: str, *, command: str, args: list) -> bool:
    """
    Finds the (single) block in `blocks` whose FIRST inner hook's args
    reference `script_name` by basename, and patches its command/args in
    place. Matching by script basename (not array position) so the
    template's own block ORDER is free to change without this needing to
    track index numbers per event -- the same basename-matching principle
    _is_toolkit_owned_hook already uses for the identical reason (a moved
    tools-repo checkout / recreated venv / reordered template must still be
    recognized). Returns True if a block was found and patched, False
    otherwise (the caller treats "not found" as a template bug, since every
    script in _TOOLKIT_HOOK_SCRIPTS is expected to have exactly one
    matching block in the real settings.json.template).
    """
    for block in blocks:
        inner_hooks = block.get("hooks") or []
        if not inner_hooks:
            continue
        first_args = inner_hooks[0].get("args") or []
        if first_args and Path(first_args[-1]).name == script_name:
            inner_hooks[0]["command"] = command
            inner_hooks[0]["args"] = args
            return True
    return False


def _strip_block_for_script(blocks: list, script_name: str) -> list:
    """Returns `blocks` with any block whose first inner hook references
    `script_name` removed entirely -- used to drop a compress-gated
    template block outright (not just leave it unpatched, which would ship
    with unresolved REPLACE-WITH-VENV-PYTHON/.../ placeholders) when
    include_compress=False."""
    kept = []
    for block in blocks:
        inner_hooks = block.get("hooks") or []
        first_args = inner_hooks[0].get("args") if inner_hooks else []
        if inner_hooks and first_args and Path(first_args[-1]).name == script_name:
            continue
        kept.append(block)
    return kept


def build_settings_hooks(
    template: dict, *, venv_python: Path, tools_repo_dir: Path, include_compress: bool = True
) -> dict:
    """
    Returns a fresh `hooks` dict (SessionStart/PostToolUse/PreToolUse/SessionEnd)
    with every REPLACE-WITH-VENV-PYTHON / /absolute/path/to/tools-repo/...
    placeholder resolved. Mirrors build_mcp_servers()'s "patch known fields,
    don't blind-replace" approach.

    `_CORE_HOOK_SCRIPTS` are ALWAYS patched and kept, regardless of
    `include_compress` (issue #198 -- record_session_id.py is base install).
    `_COMPRESS_HOOK_SCRIPTS` are patched and kept only when
    `include_compress=True`; when False, their template blocks are dropped
    from the returned dict entirely (not merely left unpatched) -- the
    EVENT KEY itself is deliberately still present with a (possibly empty)
    list rather than removed, so a caller merging this into an EXISTING
    settings.json (see merge_settings_hooks) still processes that event and
    strips any stale compress-gated block a PRIOR full setup left there,
    even when this run contributes nothing new for it.
    """
    hooks = copy.deepcopy(template["hooks"])
    venv_python_str = venv_python.as_posix()
    # See build_mcp_servers()'s matching comment -- no .resolve() here either.
    tools_repo_str = Path(tools_repo_dir).as_posix()

    for event, script_name in _CORE_HOOK_SCRIPTS:
        blocks = hooks.get(event, [])
        if not _find_and_patch_block(
            blocks, script_name, command=venv_python_str, args=[f"{tools_repo_str}/hooks/{script_name}"]
        ):
            raise RuntimeError(
                f"setup_project_lib bug: no template block found for core hook {script_name!r} under {event!r}"
            )

    for event, script_name in _COMPRESS_HOOK_SCRIPTS:
        blocks = hooks.get(event, [])
        if include_compress:
            if not _find_and_patch_block(
                blocks, script_name, command=venv_python_str, args=[f"{tools_repo_str}/hooks/{script_name}"]
            ):
                raise RuntimeError(
                    f"setup_project_lib bug: no template block found for {script_name!r} under {event!r}"
                )
        else:
            hooks[event] = _strip_block_for_script(blocks, script_name)

    return hooks


# Every MCP server name this toolkit could ever generate -- NOT just the
# ones present in a given run's generated_servers. merge_mcp_json needs the
# full set (not only what this run produced) to correctly REMOVE a
# toolkit-owned server a prior run added but this run omits (e.g.
# `--qdrant-only` after a previous run had configured local-compress) --
# see merge_mcp_json's docstring.
_OWNED_MCP_SERVER_KEYS = ("qdrant", "codebase-indexer", "memory-bank", "local-compress", "analytics")


def merge_mcp_json(existing: dict, generated_servers: dict, *, owned_keys: tuple = _OWNED_MCP_SERVER_KEYS) -> dict:
    """
    Merges `generated_servers` into an existing `.mcp.json`'s "mcpServers"
    map without touching any OTHER server a project may already have
    configured (e.g. a project-specific custom MCP server) -- only keys in
    `owned_keys` are ever added, overwritten, or removed.

    `owned_keys` is deliberately the full set this toolkit could ever
    generate, not just `generated_servers`' keys THIS run: found in PR #113
    review that omitting a key from `generated_servers` (e.g. running
    `--qdrant-only` after an earlier run had configured `local-compress`)
    previously left that stale entry in place, since a plain dict `.update`
    only ever adds/overwrites keys it's given and never removes anything.
    Any `owned_keys` member missing from `generated_servers` is now
    explicitly dropped from the existing config before the update, so
    `--qdrant-only` actually results in `local-compress` being absent.
    """
    merged = copy.deepcopy(existing)
    servers = merged.setdefault("mcpServers", {})
    for key in owned_keys:
        if key not in generated_servers:
            servers.pop(key, None)
    servers.update(generated_servers)
    return merged


# Every script filename (basename, not full path) this toolkit's hooks
# generate -- used by _is_toolkit_owned_hook below.
_TOOLKIT_HOOK_SCRIPT_NAMES = frozenset(name for _, name in _TOOLKIT_HOOK_SCRIPTS) | {LEGACY_COMPRESS_HOOK_SCRIPT}


def _is_toolkit_owned_hook(hook: dict) -> bool:
    """
    True if this single INNER hook entry (one item of a block's "hooks"
    list, not the whole block) invokes one of this toolkit's own scripts --
    checked by whether any of its args end with one of this toolkit's own
    script FILENAMES (a basename match, not a full-path/command match).

    Basename is the one thing that stays stable across a moved tools-repo
    checkout, a venv recreated at a new path, or a future template matcher
    update -- unlike comparing the old entry's exact command/args strings
    (an earlier approach), which is what let a *stale* entry through
    unrecognized: found in PR #113 review that re-running after any of
    those changes appended a second block instead of replacing the first,
    leaving the stale one's now-nonexistent command wired up to fire on
    every matching event indefinitely.
    """
    return any(Path(a).name in _TOOLKIT_HOOK_SCRIPT_NAMES for a in hook.get("args", []))


def _strip_toolkit_owned_inner_hooks(block: dict) -> Optional[dict]:
    """
    Returns a copy of `block` with any toolkit-owned INNER hook entry (see
    _is_toolkit_owned_hook) removed, or None if nothing is left afterward
    (signaling the caller should drop the whole block object).

    Operates at INNER-HOOK granularity, not whole-BLOCK granularity, because
    a single block's "hooks" list can hold more than one entry sharing the
    same matcher -- e.g. a user manually adding their own custom hook
    alongside this toolkit's `compress_output.py` entry in the SAME
    block object, rather than as a separate top-level block. Found in PR
    #113 review and confirmed by reproducing it directly: an earlier
    whole-block check discarded that entire block -- including the user's
    unrelated custom entry -- the moment ANY inner hook in it matched a
    toolkit script name, silently deleting configuration this toolkit does
    not own.
    """
    remaining = [h for h in block.get("hooks", []) if not _is_toolkit_owned_hook(h)]
    if not remaining:
        return None
    residual = copy.deepcopy(block)
    residual["hooks"] = remaining
    return residual


def merge_settings_hooks(existing: dict, generated_hooks: dict) -> dict:
    """
    Merges `generated_hooks` into an existing `.claude/settings.json`'s
    "hooks" map, preserving any unrelated existing hook blocks/inner hook
    entries (e.g. a project's own custom PostToolUse hook, even one sharing
    a block with this toolkit's own) and REPLACING -- not duplicating --
    whatever this toolkit generated in a prior run. "Generated by this
    toolkit before" is decided at the INNER-HOOK level by
    _strip_toolkit_owned_inner_hooks (a script-basename match), not by
    comparing an old block's exact command/args to the new one and not by
    discarding a whole block just because ONE of its inner entries matches
    -- see that function's docstring for why both distinctions matter. A
    moved tools-repo checkout / changed venv / updated matcher still results
    in exactly one up-to-date entry per event instead of a stale one
    lingering alongside a fresh duplicate. Returns the full top-level
    document (ready to write as-is).
    """
    merged = copy.deepcopy(existing)
    target_hooks = merged.setdefault("hooks", {})
    for event, blocks in generated_hooks.items():
        existing_blocks = target_hooks.setdefault(event, [])
        cleaned_blocks = [
            residual
            for residual in (_strip_toolkit_owned_inner_hooks(b) for b in existing_blocks)
            if residual is not None
        ]
        cleaned_blocks.extend(copy.deepcopy(b) for b in blocks)
        existing_blocks[:] = cleaned_blocks
    return merged


def strip_toolkit_hooks(existing: dict) -> dict:
    """
    Removes any toolkit-owned INNER hook entry (see
    _strip_toolkit_owned_inner_hooks) from an existing `.claude/
    settings.json`'s "hooks" map, leaving any unrelated hook entries/blocks
    and every other top-level key untouched -- including a block that mixed
    this toolkit's own hook with a user's unrelated one, where only the
    toolkit's entry is removed and the block survives with the rest intact.

    Unlike merge_settings_hooks (which ADDS/refreshes this toolkit's own
    entries), this ACTIVELY REMOVES them with nothing to replace them with --
    used for `--qdrant-only`. Found in PR #113 review: `--qdrant-only`
    correctly omitted `local-compress` from the generated `.mcp.json` (see
    merge_mcp_json), but previously left settings.json completely untouched
    either way, so a repo that had a prior FULL setup (local-compress +
    hooks) still had compress_output.py/session_end_savings.py firing
    on every matching event, and the PreToolUse hook still denying WebFetch
    and redirecting to a `fetch_url` tool that's no longer configured --
    both silently broken rather than genuinely "qdrant only".
    """
    if "hooks" not in existing:
        return copy.deepcopy(existing)
    merged = copy.deepcopy(existing)
    target_hooks = merged["hooks"]
    for event in list(target_hooks.keys()):
        target_hooks[event] = [
            residual
            for residual in (_strip_toolkit_owned_inner_hooks(b) for b in target_hooks[event])
            if residual is not None
        ]
    return merged


def default_skills_dir(home_dir: Path) -> Path:
    """
    Default install destination for `plan_skill_installs` below (issue
    #205) -- `~/.claude/skills/`, the SAME user-level location
    `docs/session-continuity.md`/`docs/savings-tracker.md` already document
    installing skills to by hand ("install globally... only once"), and
    what `/my-resume`'s own per-project scoping (by current directory name,
    not by install location) already assumes. `--skills-dir` overrides this
    for e.g. a per-project `<repo>/.claude/skills` install instead.
    """
    return home_dir / ".claude" / "skills"


def split_skill_frontmatter(data: bytes) -> "tuple[bytes, bytes]":
    """
    Splits a SKILL.md's raw bytes into `(frontmatter_prefix, body)` (issue
    #449). `frontmatter_prefix` is everything from the opening `---` line
    through the closing `---` line PLUS the blank line(s) after it, so
    `frontmatter_prefix + body == data` always holds; it is `b""` when
    `data` has no frontmatter block.

    Why this exists: the shipped `skills/*/SKILL.md` files carry NO
    frontmatter (they start at `# Skill: <name>`), but a skill installed on
    a user's machine needs `name`/`description` frontmatter to be listed,
    so real installed copies start with a `---` ... `---` block the source
    lacks. A raw byte comparison would therefore call such a copy "update"
    on EVERY run. The decision (see `plan_skill_installs`): the installed
    copy's frontmatter is the USER'S (it may carry a hand-edited
    description), so it is preserved verbatim and only the body is compared
    and replaced. Never delete it, never invent one.

    Operates on bytes and tolerates CRLF line endings, so it is safe on a
    copy saved by a Windows editor. An opening fence with no closing fence
    is not frontmatter (returns `b""` prefix): there is no well-defined
    boundary to preserve, so the whole file is treated as body.
    """
    # A UTF-8 BOM before the opening fence (common from Windows editors) is
    # not part of the fence line: skip it for detection but keep its bytes in
    # the returned prefix, so the file is still recognised as having
    # frontmatter and a refresh keeps the BOM along with the block (Copilot
    # review on PR #451).
    bom = b"\xef\xbb\xbf" if data.startswith(b"\xef\xbb\xbf") else b""
    lines = data[len(bom):].splitlines(keepends=True)
    if not lines or lines[0].rstrip() != b"---":
        return b"", data
    close = next((i for i in range(1, len(lines)) if lines[i].rstrip() == b"---"), None)
    if close is None:
        return b"", data
    end = close + 1
    while end < len(lines) and lines[end].strip() == b"":
        end += 1
    prefix = bom + b"".join(lines[:end])
    return prefix, data[len(prefix):]


@dataclass
class SkillInstallPlan:
    """
    One entry per skill directory found under `tools_repo_dir/skills/`
    (issue #205) -- `name` is the skill's directory name (e.g.
    "my-compact"), `action` is one of "install" (dest doesn't exist yet),
    "up_to_date" (dest already holds exactly what this plan would write --
    nothing to do), or "update" (dest exists but differs -- overwrite).
    `content` is the FULL text this plan would write to `dest`: the shipped
    SKILL.md, preceded by the installed copy's own frontmatter block when it
    has one (issue #449, `frontmatter_preserved`), already computed here so a
    caller can write it without touching `source` again (and so
    --dry-run/tests never need to re-read the file to report what WOULD be
    written). Bytes that are not valid UTF-8 inside a preserved frontmatter
    block round-trip via `surrogateescape` (see `SKILL_TEXT_ERRORS`).
    """

    name: str
    action: str
    source: Path
    dest: Path
    content: str
    frontmatter_preserved: bool = False


# The one error handler used to decode/encode skill text, shared with
# tools/setup_project.py's _atomic_write so a preserved frontmatter block
# that is not valid UTF-8 survives the round trip instead of crashing the
# run or being silently dropped (issue #449). For valid UTF-8 it behaves
# exactly like plain "strict".
SKILL_TEXT_ERRORS = "surrogateescape"


def plan_skill_installs(tools_repo_dir: Path, skills_dir: Path) -> list:
    """
    Read-only planning step for `claude-runway-setup init --install-skills`
    (issue #205) and, since issue #449, `upgrade`'s skill refresh -- scans
    `tools_repo_dir/skills/*/SKILL.md` (the shipped
    source of truth: the SAME path for a clone and a pipx/`uv tool install`,
    since `skills/` is shipped as wheel data files (setup.py's `data_files()`)
    landing as a sibling of `libs/tools/templates/hooks` under `<env>/src/`,
    exactly the same "sibling directories" arithmetic every other
    TOOLS_REPO_DIR-relative path in this module already relies on) and
    reports what installing into `skills_dir` would do to each one -- never
    writes anything itself. Mirrors `run_setup`'s own "pure computation, the
    CLI does the actual writing" split, so this stays trivially testable
    with only tmp directories, no mocking of file I/O needed.

    Frontmatter rule (issue #449): when the shipped source has no
    frontmatter but the installed copy does, the installed copy's
    frontmatter block is preserved and only the body is compared/replaced
    (`split_skill_frontmatter`). So an installed copy that differs from the
    shipped one ONLY by its frontmatter is "up_to_date" (no perpetual false
    "update"), and an "update" rewrites the body while keeping the user's
    frontmatter. A fresh "install" writes the shipped source as-is (no
    frontmatter is invented). If the shipped source itself ever gains
    frontmatter, an installed copy's own block STILL wins (the shipped block
    is not written over it); only an installed copy with no frontmatter gets
    the shipped file whole.

    Only `SKILL.md` files are considered; a skill directory without one
    (shouldn't happen for anything shipped by this repo, but a defensive
    skip rather than a crash for e.g. a stray non-skill directory someone
    drops under `skills/`) is silently omitted from the plan. Returns `[]`
    if `tools_repo_dir/skills` doesn't exist at all, rather than raising --
    letting a caller decide how to report "nothing to install" is more
    useful than forcing every caller to handle a missing-directory
    exception for what may just be an unusual install layout.

    Sorted by skill name for stable, deterministic output across runs (a
    plain `Path.iterdir()` order isn't guaranteed) -- matters for both
    human-readable --dry-run output and assertion-friendly tests.
    """
    plans: list = []
    skills_root = Path(tools_repo_dir) / "skills"
    if not skills_root.is_dir():
        return plans
    skills_dir = Path(skills_dir)
    for skill_dir in sorted(p for p in skills_root.iterdir() if p.is_dir()):
        source = skill_dir / "SKILL.md"
        if not source.is_file():
            continue
        # Read as raw bytes then decode, NOT source.read_text() (Copilot
        # review on PR #224): read_text() opens in universal-newlines text
        # mode, which silently translates a CRLF source's line endings to
        # bare "\n" -- lossy on every platform, and specifically broken on
        # Windows once combined with _atomic_write's own text-mode write
        # (which re-translates "\n" back to the PLATFORM's line ending,
        # CRLF there): the written dest then has CRLF bytes while `content`
        # (and every future run's freshly re-read `content`) has bare "\n",
        # so the byte comparison below never matches and every install
        # perpetually reports (and rewrites) as "update", never "up_to_date".
        # Reading raw bytes here, and _atomic_write below writing raw bytes
        # right back with no text-mode translation on either end, keeps
        # this byte-for-byte round-trippable regardless of platform or
        # what line-ending style the shipped SKILL.md actually uses.
        source_bytes = source.read_bytes()
        content = source_bytes.decode("utf-8")
        dest = skills_dir / skill_dir.name / "SKILL.md"
        preserved = False
        if not dest.exists():
            action = "install"
        else:
            # Compare raw bytes, not decoded text (Copilot review on PR #224):
            # a pre-existing dest SKILL.md a user hand-edited and saved in a
            # non-UTF-8 encoding would otherwise raise UnicodeDecodeError
            # here, crashing the entire --install-skills run before the
            # documented "different -> update" path ever got a chance to
            # run. Reading bytes on both sides needs no decoding of dest at
            # all -- source is always our own shipped UTF-8 content
            # (already read above), so encoding it once and comparing bytes
            # is both simpler and more precise than a lossy decode (e.g.
            # errors="replace") would be.
            dest_bytes = dest.read_bytes()
            # Issue #449: the installed copy's own frontmatter always wins.
            # It is carried over in front of the shipped BODY -- if a future
            # shipped source gains frontmatter of its own, that block is
            # dropped from what is written (an installed copy already has
            # one, which is the user's to keep), so a refresh can never
            # replace or delete it. An installed copy with NO frontmatter
            # gets the shipped file whole, frontmatter included if any.
            prefix = split_skill_frontmatter(dest_bytes)[0]
            if prefix:
                source_body = split_skill_frontmatter(source_bytes)[1]
                content = prefix.decode("utf-8", errors=SKILL_TEXT_ERRORS) + source_body.decode("utf-8")
                preserved = True
            if dest_bytes == content.encode("utf-8", errors=SKILL_TEXT_ERRORS):
                action = "up_to_date"
            else:
                action = "update"
        plans.append(
            SkillInstallPlan(
                name=skill_dir.name, action=action, source=source, dest=dest, content=content,
                frontmatter_preserved=preserved,
            )
        )
    return plans


@dataclass
class SetupResult:
    """
    `mcp_json`/`settings_json` are the FULL `.mcp.json`/`.claude/
    settings.json` documents this run would produce, ready to write to disk
    as-is -- including any pre-existing, unrelated content (other MCP
    servers, other hooks) merged in from the target repo's current files.

    `generated_mcp_servers`/`generated_settings_hooks`, by contrast, hold
    ONLY the toolkit-owned portion this run itself contributed -- no
    pre-existing content at all. These exist specifically so a caller
    previewing changes (`tools/setup_project.py`'s `--dry-run`) never has
    to print the full merged document just to show what changed. Found in
    PR #113 review: printing `mcp_json`/`settings_json` directly for
    --dry-run meant any unrelated MCP server's secret (an API token sitting
    in that server's own `env` block, merged in unchanged from the target's
    EXISTING `.mcp.json`) got printed to the terminal/CI log right along
    with it -- a real credential leak, not a hypothetical one, since
    merging in unrelated existing servers/hooks untouched is this whole
    module's own explicit design goal (see merge_mcp_json/
    merge_settings_hooks). `generated_settings_hooks` is None only for
    --skip-hooks's "leave settings.json alone entirely" -- NOT for
    --qdrant-only (issue #198): that mode still generates and returns the
    CORE record_session_id.py hooks (they're written/kept either way, not
    local-compress-gated), it just omits the compress-dependent blocks
    from what it builds. So --qdrant-only's `changes` entry summarizes a
    real (non-empty) `generated_settings_hooks` value, same as a normal
    run -- callers/dry-run code should not assume it's None for this mode.
    """

    mcp_json: dict
    settings_json: Optional[dict]
    generated_mcp_servers: dict
    generated_settings_hooks: Optional[dict]
    changes: list = field(default_factory=list)


def run_setup(
    target_repo: Path,
    tools_repo_dir: Path,
    *,
    collection_name: Optional[str] = None,
    venv_python: Optional[Path] = None,
    windows: bool = False,
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: str = "",
    collection_description: str = "",
    include_extensions: str = "",
    exclude_dirs: str = "",
    lmstudio_url: str = "http://localhost:1234/v1",
    lmstudio_model: str = "",
    track_savings: bool = False,
    savings_db: str = "",
    metrics_db: str = "",
    compact_collection: str = "",
    memory_bank_collection: str = "",
    memory_bank_id: str = "",
    include_compress: bool = True,
    include_hooks: bool = True,
    clean_hooks_if_unused: bool = False,
    home_dir: Optional[Path] = None,
    templates_dir: Optional[Path] = None,
    embedding_model: str = "",
    memory_bank_embedding_model: str = "",
) -> SetupResult:
    """
    Pure computation (no disk writes to `target_repo`) of the final
    `.mcp.json`/`.claude/settings.json` contents for `target_repo`. The
    caller (the CLI in tools/setup_project.py) is responsible for actually
    writing the result or printing it for --dry-run. Reads the target repo's
    CURRENT files if they already exist, so re-running this (or running it
    against a repo with pre-existing unrelated config) merges rather than
    clobbers -- see merge_mcp_json/merge_settings_hooks.

    `embedding_model` (issue #279, CLI `--embedding-model`) is the one
    setting that IS sticky across re-runs: blank means "keep whatever the
    target's existing `.mcp.json` already has" (template default for a
    brand-new project), never "reset to the template default". A non-blank
    value that differs from the existing one adds a WARNING to `changes`.

    `memory_bank_embedding_model` (issue #428, CLI `--memory-bank-embedding-model`)
    is the same sticky setting for the `memory-bank` block ONLY, and is fully
    independent of `embedding_model`: blank keeps the block's existing value
    (template default for a block that does not exist). A non-blank value that
    differs from the existing one adds a WARNING, because the shared collection
    is locked to the model that created it.

    `include_hooks=False` corresponds to the CLI's `--skip-hooks`: don't
    touch `.claude/settings.json` at all, even if it already has this
    toolkit's hooks in it -- the caller explicitly asked to leave that file
    alone. `clean_hooks_if_unused=True` is this same escape hatch's OWN
    lower-level cleanup path (see `strip_toolkit_hooks`), still supported
    for a caller that wants "leave settings.json alone UNLESS it already has
    stale toolkit hooks, in which case remove them" -- but the CLI itself no
    longer needs it (issue #198): `--qdrant-only` now keeps
    `include_hooks=True` (record_session_id.py is CORE/base install and must
    still be written) and instead passes `include_compress=False` into the
    `include_hooks=True` branch below, whose `build_settings_hooks(
    include_compress=False)` + `merge_settings_hooks` already strip any
    STALE compress-gated block a prior full setup left behind, as a normal
    part of that same write -- no separate "clean, don't add" branch
    required for that case anymore.
    """
    target_repo = Path(target_repo)
    tools_repo_dir = Path(tools_repo_dir)
    templates_dir = Path(templates_dir) if templates_dir else tools_repo_dir / "templates"
    home_dir = Path(home_dir) if home_dir else Path.home()
    resolved_venv_python = Path(venv_python) if venv_python else venv_python_path(tools_repo_dir, windows=windows)
    resolved_collection_name = collection_name or default_collection_name(target_repo)
    # Same fallback build_mcp_servers applies internally -- computed here too
    # so the collision check below (and any other caller) can compare against
    # the SAME resolved value rather than re-deriving/guessing it.
    resolved_memory_bank_collection = memory_bank_collection or "memory-bank"
    resolved_memory_bank_id = memory_bank_id or resolved_collection_name

    mcp_template = load_json(templates_dir / "mcp.json.template")

    # Issue #279: read the target's existing .mcp.json BEFORE deciding the
    # embedding model (and before the warm-up below, which must warm the
    # model that will actually be written). See resolve_embedding_model for
    # the explicit > existing > template rule.
    mcp_json_path = target_repo / ".mcp.json"
    existing_mcp = load_json(mcp_json_path) if mcp_json_path.exists() else {}
    template_embedding_model = mcp_template["mcpServers"]["qdrant"]["env"]["EMBEDDING_MODEL"]
    previous_embedding_model = existing_embedding_model(existing_mcp)
    resolved_embedding_model = resolve_embedding_model(embedding_model, existing_mcp, template_embedding_model)
    # Issue #428: memory-bank's model is resolved from its own sources only
    # (explicit flag > its own existing block > template default).
    template_memory_bank_model = mcp_template["mcpServers"]["memory-bank"]["env"]["EMBEDDING_MODEL"]
    previous_memory_bank_model = existing_memory_bank_model(existing_mcp)
    resolved_memory_bank_model = resolve_memory_bank_embedding_model(
        memory_bank_embedding_model, existing_mcp, template_memory_bank_model
    )

    generated_servers = build_mcp_servers(
        mcp_template,
        collection_name=resolved_collection_name,
        venv_python=resolved_venv_python,
        tools_repo_dir=tools_repo_dir,
        home_dir=home_dir,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_description=collection_description,
        include_extensions=include_extensions,
        exclude_dirs=exclude_dirs,
        lmstudio_url=lmstudio_url,
        lmstudio_model=lmstudio_model,
        track_savings=track_savings,
        savings_db=savings_db,
        metrics_db=metrics_db,
        compact_collection=compact_collection,
        memory_bank_collection=memory_bank_collection,
        memory_bank_id=memory_bank_id,
        include_compress=include_compress,
        embedding_model=resolved_embedding_model,
        memory_bank_embedding_model=resolved_memory_bank_model,
        compact_embedding_model=resolve_compact_embedding_model(existing_mcp, resolved_embedding_model),
    )
    unresolved = find_unresolved_placeholders(generated_servers)
    if unresolved:
        raise RuntimeError(
            f"setup_project_lib bug: unresolved placeholder(s) survived patching in mcp.json: {unresolved}"
        )

    final_mcp = merge_mcp_json(existing_mcp, generated_servers)
    changes = [f"mcpServers ({', '.join(sorted(generated_servers))}) -> {mcp_json_path}"]
    if previous_embedding_model and resolved_embedding_model != previous_embedding_model:
        # Only reachable via an explicit --embedding-model that differs from
        # what's configured: the drop-and-reindex scenario the README documents.
        changes.append(
            f"WARNING: EMBEDDING_MODEL changed from '{previous_embedding_model}' to "
            f"'{resolved_embedding_model}'. Vectors already indexed under the old model are "
            f"incompatible: drop the project's collection (directly in Qdrant) and re-run "
            f"index_repo. The memory-bank server's model is separate and is not changed by "
            f"--embedding-model."
        )
    elif previous_embedding_model and previous_embedding_model != template_embedding_model:
        changes.append(
            f"Preserved the existing EMBEDDING_MODEL '{previous_embedding_model}' (pass "
            f"--embedding-model to change it)."
        )
    if previous_memory_bank_model and resolved_memory_bank_model != previous_memory_bank_model:
        # Only reachable via an explicit --memory-bank-embedding-model.
        changes.append(
            f"WARNING: the memory-bank EMBEDDING_MODEL changed from '{previous_memory_bank_model}' to "
            f"'{resolved_memory_bank_model}'. Its collection ('{resolved_memory_bank_collection}') is shared "
            f"by every project and locked to the model that created it, so remember/recall will fail "
            f"for EVERY project using it until that collection is rebuilt (or they all move to the new model)."
        )
    elif previous_memory_bank_model and previous_memory_bank_model != template_memory_bank_model:
        changes.append(
            f"Preserved the existing memory-bank EMBEDDING_MODEL '{previous_memory_bank_model}' (pass "
            f"--memory-bank-embedding-model to change it)."
        )
    # Issue #175: memory-bank reserves "general" as its cross-project
    # sentinel (metadata.repo) -- a project whose own MEMORY_BANK_ID
    # resolves to exactly that string would be indistinguishable from
    # genuinely general knowledge. Checked against resolved_memory_bank_id,
    # NOT resolved_collection_name (PR #178 review, fifth pass) -- since
    # memory_bank_id is now an independently-settable value, it's the one
    # that actually ends up as the repo tag; collection_name is only its
    # DEFAULT when memory_bank_id isn't passed. Warn rather than silently
    # letting this collide (memory_bank_mcp_server.py itself also refuses
    # at first use, this just surfaces it earlier, at setup time).
    if resolved_memory_bank_id == "general":
        changes.append(
            "WARNING: this project's MEMORY_BANK_ID resolved to 'general', which collides "
            "with memory-bank's reserved general-knowledge tag -- pass --memory-bank-id "
            "with a different value."
        )

    # Found in PR #178 review: a target repo directory named e.g. "memory-bank"
    # with both COLLECTION_NAME and MEMORY_BANK_COLLECTION left at their
    # defaults resolves them to the SAME string -- tools/ingest_mcp_server.py's
    # reserved-collection-name guard then refuses EVERY index_repo/sync_repo
    # call for this project, since its own code-index collection IS the
    # reserved memory-bank name. Warn at setup time rather than leaving this
    # to surface later as a confusing "Error: ... is configured as the shared
    # memory-bank collection" on the very first index attempt.
    if resolved_collection_name == resolved_memory_bank_collection:
        changes.append(
            f"WARNING: this project's COLLECTION_NAME ('{resolved_collection_name}') is the same "
            f"as MEMORY_BANK_COLLECTION -- index_repo/sync_repo will refuse to run at all for this "
            f"project (they never touch the shared memory-bank collection). Pass --collection-name "
            f"or --memory-bank-collection so the two differ."
        )

    final_settings: Optional[dict] = None
    generated_hooks: Optional[dict] = None
    if include_hooks:
        settings_template = load_json(templates_dir / "settings.json.template")
        generated_hooks = build_settings_hooks(
            settings_template,
            venv_python=resolved_venv_python,
            tools_repo_dir=tools_repo_dir,
            include_compress=include_compress,
        )
        unresolved = find_unresolved_placeholders(generated_hooks)
        if unresolved:
            raise RuntimeError(
                f"setup_project_lib bug: unresolved placeholder(s) survived patching in settings.json: {unresolved}"
            )

        settings_path = target_repo / ".claude" / "settings.json"
        existing_settings = load_json(settings_path) if settings_path.exists() else {}
        final_settings = merge_settings_hooks(existing_settings, generated_hooks)
        changes.append(f"hooks ({', '.join(sorted(generated_hooks))}) -> {settings_path}")
    elif clean_hooks_if_unused:
        settings_path = target_repo / ".claude" / "settings.json"
        if settings_path.exists():
            existing_settings = load_json(settings_path)
            cleaned_settings = strip_toolkit_hooks(existing_settings)
            if cleaned_settings != existing_settings:
                final_settings = cleaned_settings
                changes.append(f"hooks (removing this toolkit's own blocks, --qdrant-only) -> {settings_path}")

    return SetupResult(
        mcp_json=final_mcp,
        settings_json=final_settings,
        generated_mcp_servers=generated_servers,
        generated_settings_hooks=generated_hooks,
        changes=changes,
    )
