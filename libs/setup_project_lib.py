"""
Shared, dependency-free logic behind `tools/setup_project.py` -- computing
defaults and patching `templates/mcp.json.template` /
`templates/settings.json.template` into a real, ready-to-use `.mcp.json` /
`.claude/settings.json` for a target project repo, instead of the manual
copy-and-hand-edit flow `docs/installation.md`'s steps 4/4b/5 otherwise require
(issue #48). Kept separate from `tools/setup_project.py` (the thin CLI) the
same way `qdrant_ingest_lib.py`/`local_compress_lib.py` are kept separate
from their own CLI/MCP-server callers -- pure functions here are what make
this testable without touching disk beyond reading the template files
themselves.

Deliberately patches KNOWN fields structurally (dict/list traversal) rather
than doing a blind string find-and-replace of the placeholder tokens over
the raw template text. This matters for one field specifically:
`mcp.json.template`'s `qdrant` server has `"command":
"REPLACE-WITH-VENV-PYTHON/bin/mcp-server-qdrant"`, but `docs/installation.md`'s own
step 3 instructs substituting `REPLACE-WITH-VENV-PYTHON`
everywhere with the *full python interpreter path* (e.g.
`.../.venv/bin/python`). Doing that literal substitution on this one field
produces a broken, doubled path: `.../.venv/bin/python/bin/mcp-server-qdrant`
-- the sibling console-script is installed as a SIBLING of the interpreter
in the same `bin`/`Scripts` directory, not something reachable by
appending a path segment onto the interpreter's own path. A human
following the README instructions verbatim would produce this exact
broken config. See `mcp_server_qdrant_path()` below for the fix.

That console-script's name is also easy to get wrong independently of the
doubled-path bug above: `pip install mcp-server-qdrant` registers the
entry point as the HYPHENATED `mcp-server-qdrant` (which is also what the
template's own literal text above now correctly spells) -- the underscored
`mcp_server_qdrant` is only the Python *import* package name (`from
mcp_server_qdrant.main import main`), never a file that exists on disk in
the venv's `bin`/`Scripts` directory. Found in PR #113 review and confirmed
directly (`ls .venv/bin/` shows `mcp-server-qdrant`, not
`mcp_server_qdrant`) -- only the Python function/variable names in this
file use underscores, per Python naming convention, which is unrelated to
the actual script filename on disk.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Flat import, not a package-relative one: every libs/*.py module lives in
# the same libs/ directory, which callers (tools/setup_project.py, test
# harnesses) add to sys.path directly rather than treating libs/ as a real
# Python package -- see e.g. libs/memory_bank_lib.py's own
# `from qdrant_retry import ...`. qdrant_ingest_lib.py stays importable
# without `fastembed`/`qdrant-client` installed (warm_fastembed_cache
# imports fastembed lazily, inside the function, only when actually
# called), so this doesn't add a real dependency to this otherwise
# dependency-free module.
from qdrant_ingest_lib import warm_fastembed_cache


def default_collection_name(repo_path: Path) -> str:
    """
    Derive a Qdrant collection name suggestion from the target repo's own
    directory name, e.g. "My Cool App!" -> "my-cool-app".

    This is only a convenience DEFAULT, not a collision-proofed identifier
    the way tools/compress_mcp_server.py's `_sanitize_project` is (that one
    hashes the original string to guard against a *silent* same-name
    collision for the auto-managed conversation-compacts collection, since
    nothing else would ever surface that collision to a human). Here, by
    contrast, the collection name is a one-time, visible choice a human
    reviews in the generated `.mcp.json` before ever committing it -- so a
    plain, readable slug is more useful than an opaque hash suffix. Pass
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


def mcp_server_qdrant_path(venv_python: Path) -> Path:
    """
    Path to the `mcp-server-qdrant` console-script pip installs as a SIBLING
    of the venv's python interpreter (same `bin`/`Scripts` directory) -- see
    this module's docstring for why this can't just be a
    "REPLACE-WITH-VENV-PYTHON/bin/mcp-server-qdrant" text-substitution the
    way the raw template literally spells it: substituting the full
    interpreter path there produces a broken, doubled path, since the
    sibling console-script is never reachable by appending a segment onto
    the interpreter's own path.
    """
    suffix = ".exe" if venv_python.suffix.lower() == ".exe" else ""
    return venv_python.parent / f"mcp-server-qdrant{suffix}"


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


def resolved_fastembed_cache_path(home_dir: Path) -> Path:
    """
    Single source of truth for where a generated `.mcp.json`'s `qdrant`
    server's `FASTEMBED_CACHE_PATH` points -- factored out of
    `build_mcp_servers` (issue #221) so `run_setup`'s own fastembed-warm-up
    step (which needs this SAME path to warm/verify before deciding whether
    `HF_HUB_OFFLINE=1` is safe to write) can't drift from what actually gets
    written into the config.
    """
    return home_dir / ".claude" / "claude-runway" / "fastembed-cache"


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
    compact_collection: str = "",
    memory_bank_collection: str = "",
    memory_bank_id: str = "",
    include_compress: bool = True,
    hf_hub_offline: bool = False,
) -> dict:
    """
    Returns a fresh `mcpServers` dict (deep-copied from `template`, never
    mutated in place) with every placeholder in the qdrant/codebase-indexer/
    local-compress blocks resolved to a real value. Paths are emitted as
    forward-slash (`Path.as_posix()`) to match the templates' own style,
    which uses forward slashes even in its Windows-facing example paths.

    `compact_collection`, if blank, defaults to the template's own historical
    "conversation-compacts" -- a single bucket shared by every project on
    the machine, with per-project isolation coming entirely from
    compress_mcp_server.py's own `_sanitize_project`+hash suffix at runtime
    (see `_collections_for_project`'s sibling-scan there, which finds other
    collections for the same project by matching this exact prefix).
    Deliberately NOT derived from `collection_name` here the way
    `COLLECTION_NAME` on the qdrant/codebase-indexer servers is: found in
    PR #130 review that doing so silently changes this value on EVERY
    already-configured project the moment its `.mcp.json` gets regenerated
    (a plain re-run with no new flags, the exact "safe to re-run" workflow
    this script's own module docstring advertises) -- since every project's
    `local-compress.env.COMPACT_COLLECTION` has only ever held this one
    constant so far (this function never touched the key at all before
    PR #130), and `merge_mcp_json` replaces the whole `local-compress` block
    wholesale, not a per-key deep merge. Changing the *base prefix* silently
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
    fallback in compress_mcp_server.py isn't enough on its own, since that
    fallback only fires when the key is genuinely ABSENT, not when it's
    present-but-empty -- an explicit `.mcp.json` env entry set to `""`
    would otherwise build a collection name with a leading stray hyphen
    (`-<sanitized-project>-<hash8>`).

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

    `qdrant_api_key` is applied to all four servers' `QDRANT_API_KEY`
    (confirmed the correct, shared env var name across `mcp-server-qdrant`'s
    own `QdrantSettings` -- `validation_alias="QDRANT_API_KEY"` -- and this
    repo's own `ingest_mcp_server.py`/`compress_mcp_server.py`, which read
    the exact same variable so a single project's `.mcp.json` keeps every
    Qdrant-talking server aligned). Found in PR #113 review: since this
    script previously always replaced these servers' `env` blocks wholesale
    with values it computed, an authenticated remote-Qdrant setup's
    manually-added `QDRANT_API_KEY` (a field the qdrant/codebase-indexer
    blocks didn't even have here before this fix) was silently dropped on
    every rerun, and local-compress's own copy was reset back to the
    template's blank default. Passing `--qdrant-api-key` explicitly on every
    run (the same pattern already used for `--qdrant-url`/`--collection-
    name`/etc. -- none of this CLI's other options are "sticky" either) is
    what keeps it from reverting.

    `hf_hub_offline` (issue #221) is deliberately NOT something this
    function decides for itself -- it stays a pure computation over an
    already-decided bool, computed by `run_setup`'s own fastembed-warm-up
    step (see `qdrant_ingest_lib.warm_fastembed_cache`) BEFORE calling this
    function, since deciding it here would require this function to make a
    live network call / touch disk beyond the template, breaking the "pure,
    no network/disk side effects" property every existing test of this
    function (and every OTHER caller) already relies on. When True, writes
    `HF_HUB_OFFLINE=1` into the qdrant server's env block -- confirmed
    directly that this cuts `mcp-server-qdrant`'s own startup from ~59.6s to
    ~2.0s against an already-warm cache, by skipping the live
    huggingface.co revision-resolution call `fastembed`'s `TextEmbedding`
    otherwise always makes even on a cache hit. Never left unset either way
    (same "present-but-blank, not absent" convention as
    `CLAUDE_RUNWAY_TRACK_SAVINGS`/`compact_collection`/etc. above) --
    `False` writes an explicit empty string, not a missing key.
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
    qdrant["command"] = mcp_server_qdrant_path(venv_python).as_posix()
    qdrant["env"]["QDRANT_URL"] = qdrant_url
    qdrant["env"]["QDRANT_API_KEY"] = qdrant_api_key
    qdrant["env"]["COLLECTION_NAME"] = collection_name
    qdrant["env"]["FASTEMBED_CACHE_PATH"] = resolved_fastembed_cache_path(home_dir).as_posix()
    qdrant["env"]["HF_HUB_OFFLINE"] = "1" if hf_hub_offline else ""

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

    if include_compress:
        compress = servers["local-compress"]
        compress["command"] = venv_python_str
        compress["args"] = [f"{tools_repo_str}/tools/compress_mcp_server.py"]
        compress["env"]["CLAUDE_RUNWAY_LMSTUDIO_URL"] = lmstudio_url
        compress["env"]["CLAUDE_RUNWAY_LMSTUDIO_MODEL"] = lmstudio_model
        compress["env"]["QDRANT_URL"] = qdrant_url
        compress["env"]["QDRANT_API_KEY"] = qdrant_api_key
        compress["env"]["CLAUDE_RUNWAY_TRACK_SAVINGS"] = "1" if track_savings else ""
        compress["env"]["CLAUDE_RUNWAY_SAVINGS_DB"] = savings_db
        compress["env"]["COMPACT_COLLECTION"] = compact_collection or "conversation-compacts"
    else:
        del servers["local-compress"]

    return servers


# (event, script filename) for every hook this toolkit's templates define,
# split into two groups (issue #198):
#   - _CORE_HOOK_SCRIPTS: written for EVERY project regardless of
#     --qdrant-only/local-compress config -- today, just record_session_id.py
#     (registered under both PostToolUse and SessionEnd), which keeps
#     libs/session_id_lib.py's SHADOW_FILE strategy actually populated.
#   - _COMPRESS_HOOK_SCRIPTS: only written when local-compress is configured
#     (include_compress=True below) -- the pre-existing three hooks.
# _TOOLKIT_HOOK_SCRIPTS (both groups combined) is the single source of truth
# for _is_toolkit_owned_hook below (which recognizes a block as ours by
# script filename alone, independent of the mutable path/matcher/array
# position around it) -- keeping every group derived from the same tuples is
# what stops them from silently drifting apart if a script is ever renamed
# or a new hook added.
_CORE_HOOK_SCRIPTS = (
    ("PostToolUse", "record_session_id.py"),
    ("SessionEnd", "record_session_id.py"),
)
_COMPRESS_HOOK_SCRIPTS = (
    ("PostToolUse", "compress_bash_output.py"),
    ("PreToolUse", "redirect_webfetch_to_fetch_url.py"),
    ("SessionEnd", "session_end_savings.py"),
)
_TOOLKIT_HOOK_SCRIPTS = _CORE_HOOK_SCRIPTS + _COMPRESS_HOOK_SCRIPTS


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
    Returns a fresh `hooks` dict (PostToolUse/PreToolUse/SessionEnd) with
    every REPLACE-WITH-VENV-PYTHON / /absolute/path/to/tools-repo/...
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
_OWNED_MCP_SERVER_KEYS = ("qdrant", "codebase-indexer", "memory-bank", "local-compress")


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
_TOOLKIT_HOOK_SCRIPT_NAMES = frozenset(name for _, name in _TOOLKIT_HOOK_SCRIPTS)


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
    alongside this toolkit's `compress_bash_output.py` entry in the SAME
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
    hooks) still had compress_bash_output.py/session_end_savings.py firing
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
    compact_collection: str = "",
    memory_bank_collection: str = "",
    memory_bank_id: str = "",
    include_compress: bool = True,
    include_hooks: bool = True,
    clean_hooks_if_unused: bool = False,
    home_dir: Optional[Path] = None,
    templates_dir: Optional[Path] = None,
    attempt_fastembed_warmup: bool = False,
    fastembed_warmup_fn: Optional[Callable[..., bool]] = None,
) -> SetupResult:
    """
    Pure computation (no disk writes to `target_repo`, with one deliberate
    exception -- see `attempt_fastembed_warmup` below) of the final
    `.mcp.json`/`.claude/settings.json` contents for `target_repo`. The
    caller (the CLI in tools/setup_project.py) is responsible for actually
    writing the result or printing it for --dry-run. Reads the target repo's
    CURRENT files if they already exist, so re-running this (or running it
    against a repo with pre-existing unrelated config) merges rather than
    clobbers -- see merge_mcp_json/merge_settings_hooks.

    `attempt_fastembed_warmup` (issue #221, default False -- opt-in, so
    every EXISTING caller/test of this function keeps its current
    network/disk-free behavior unchanged) is that one deliberate exception:
    when True, this calls `qdrant_ingest_lib.warm_fastembed_cache` (or
    `fastembed_warmup_fn` if given -- tests inject a fake here so they never
    import real `fastembed` or touch the network) BEFORE building the
    qdrant server's env block, using the SAME `EMBEDDING_MODEL` string
    already baked into `mcp.json.template` and the SAME
    `FASTEMBED_CACHE_PATH` this run is about to write (via
    `resolved_fastembed_cache_path`) -- so the thing that gets
    warmed/verified is always the thing the generated config actually
    points at. The CLI passes `attempt_fastembed_warmup=not args.dry_run`: a
    preview shouldn't have network/disk side effects. Whether this succeeds
    or not, a `changes` entry records the outcome so both --dry-run and a
    real run surface it.

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

    # Issue #221: decide -- BEFORE building the qdrant server's env block --
    # whether it's safe to set HF_HUB_OFFLINE=1, so mcp-server-qdrant's own
    # startup never pays for a live huggingface.co round-trip against a
    # model that's already fully cached. Uses the exact same
    # EMBEDDING_MODEL/FASTEMBED_CACHE_PATH values this run is about to
    # write, so what gets warmed/verified always matches what the generated
    # config actually points at. Opt-in (default False) and injectable
    # (fastembed_warmup_fn) so every existing caller/test is unaffected.
    hf_hub_offline = False
    if attempt_fastembed_warmup:
        warmup_fn = fastembed_warmup_fn or warm_fastembed_cache
        embedding_model = mcp_template["mcpServers"]["qdrant"]["env"]["EMBEDDING_MODEL"]
        fastembed_cache_path = resolved_fastembed_cache_path(home_dir)
        hf_hub_offline = bool(warmup_fn(embedding_model, fastembed_cache_path))

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
        compact_collection=compact_collection,
        memory_bank_collection=memory_bank_collection,
        memory_bank_id=memory_bank_id,
        include_compress=include_compress,
        hf_hub_offline=hf_hub_offline,
    )
    unresolved = find_unresolved_placeholders(generated_servers)
    if unresolved:
        raise RuntimeError(
            f"setup_project_lib bug: unresolved placeholder(s) survived patching in mcp.json: {unresolved}"
        )

    mcp_json_path = target_repo / ".mcp.json"
    existing_mcp = load_json(mcp_json_path) if mcp_json_path.exists() else {}
    final_mcp = merge_mcp_json(existing_mcp, generated_servers)
    changes = [f"mcpServers ({', '.join(sorted(generated_servers))}) -> {mcp_json_path}"]
    if attempt_fastembed_warmup:
        changes.append(
            "Confirmed the fastembed cache is warm; HF_HUB_OFFLINE=1 set for the qdrant server (issue #221)."
            if hf_hub_offline
            else "Could not confirm the fastembed cache is fully warm -- HF_HUB_OFFLINE left blank, so the "
            "qdrant server's startup may still occasionally pay for a slow huggingface.co round-trip "
            "(issue #221). Safe to re-run this setup later to retry."
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
