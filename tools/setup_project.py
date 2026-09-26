#!/usr/bin/env python3
"""
One-time setup for a target project repo: writes that repo's `.mcp.json`
and `.claude/settings.json` with every `REPLACE-WITH-*` /
`/absolute/path/to/...` placeholder from `templates/mcp.json.template` and
`templates/settings.json.template` filled in automatically, instead of the
manual copy-and-hand-edit flow in `docs/installation.md`'s steps 4/4b/5
(issue #48). Also structurally prevents stale hand-typed example paths
(issue #34/BUG-14) recurring, since every path here is computed, not typed.

Usage:
    python tools/setup_project.py init /path/to/target-repo
    python tools/setup_project.py init /path/to/target-repo --dry-run
    python tools/setup_project.py init /path/to/target-repo --qdrant-only
    python tools/setup_project.py init /path/to/target-repo \
        --collection-name my-project --lmstudio-model google/gemma-3-4b --track-savings

The templates stay the single source of truth: this script loads and
patches them (see libs/setup_project_lib.py) rather than duplicating their
content, so future template changes flow through to new setups automatically.

Safe to re-run: existing `.mcp.json`/`.claude/settings.json` content this
toolkit doesn't own (other MCP servers, other hooks) is preserved, and
re-running with the same options doesn't duplicate hook entries -- see
libs/setup_project_lib.py's merge_mcp_json/merge_settings_hooks.

Project minimum: Python 3.12 (policy floor; the dependency chain supports >=3.10).
"""

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from setup_project_lib import run_setup  # noqa: E402

TOOLS_REPO_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pip_installed_venv_python() -> Optional[Path]:
    """
    Issue #50: `setup_project_lib.venv_python_path`'s DEFAULT assumes the
    "clone this repo, `uv venv` right next to it" layout
    `docs/installation.md`'s step 3 documents -- `TOOLS_REPO_DIR / ".venv" / "bin" /
    "python"`. That assumption breaks when this script runs as the
    `claude-runway-setup` console script from a `pipx`/`uv tool install`
    install (the two methods `docs/installation.md`'s "Alternative" callout actually
    documents): there IS no `.venv` sibling directory next to the installed
    `tools`/`libs` packages under site-packages -- pipx/uv tool manage their
    own venv elsewhere, and `sys.executable` (whichever interpreter is
    actually running this console script right now) is the only
    generally-correct answer in that case, since that's exactly the
    environment `pip install claude-runway`'s own dependencies (including
    `mcp-server-qdrant`, whose console-script `mcp_server_qdrant_path`
    expects as a SIBLING of this same interpreter) were installed into --
    confirmed live for `uv tool install` specifically: its own venv exposes
    ONLY the installed package's own 3 console scripts on the shared `PATH`
    shim directory, keeping `mcp-server-qdrant` (a dependency, not this
    package's own entry point) reachable only as that real sibling-of-
    `sys.executable` file inside the tool's private venv -- so this is not
    just the simplest option, it's the only one that would have found it at
    all without extra machinery (a `shutil.which()`-based lookup would NOT
    find it, since pipx/uv tool deliberately don't expose it globally).

    Deliberately NOT handling: a bare `pip install`/`pip install --user`
    outside a dedicated venv (PR #136 review, Copilot) -- there,
    `sys.executable` can be a shared system interpreter while the installed
    console scripts land in a completely different directory (e.g.
    `~/.local/bin` under `--user`), breaking the "sibling of the
    interpreter" assumption `mcp_server_qdrant_path` relies on. This isn't
    silently unhandled: it's out of scope, since neither the issue nor
    `docs/installation.md`'s own "Alternative" callout documents or recommends that install
    method -- pipx/`uv tool install` are the only ones actually promised to
    work, and both were verified to satisfy this function's assumption
    directly (see the previous paragraph).

    Returns None (meaning: let `run_setup` fall back to its own pure
    `venv_python_path` default) when `TOOLS_REPO_DIR/.venv` already exists --
    i.e. the documented clone-based workflow -- so this is a zero-behavior-
    change no-op for every existing user/test of that flow. Only returns an
    override (`sys.executable`) when that directory is absent, which is
    exactly the pipx/`uv tool install` case (or a clone where step 3 hasn't
    run yet, where pointing at the currently-running interpreter is arguably
    MORE correct than a `.venv` that doesn't exist yet either).
    """
    if (TOOLS_REPO_DIR / ".venv").exists():
        return None
    return Path(sys.executable)


def _write_json(path: Path, data: dict) -> None:
    """
    Writes JSON atomically: serialize to a temp file in the SAME directory
    as the destination, flush + fsync it, then os.replace() over the real
    path. Found in PR #113 review: the previous open(path, "w") truncated
    the destination immediately, so an interruption or disk-full error
    mid-write could destroy the target repo's existing .mcp.json/.claude/
    settings.json -- exactly the content this script's merge logic exists
    to preserve, not clobber. os.replace() is atomic on both POSIX and
    Windows as long as source and destination share a filesystem, which a
    temp file created in the destination's own directory guarantees.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


_REDACTED = "***REDACTED***"


def _redact_secrets(mcp_servers: dict) -> dict:
    """
    Returns a deep copy of `mcp_servers` with any non-empty `QDRANT_API_KEY`
    value replaced by a redaction marker -- used ONLY for --dry-run's
    printed preview, never for what actually gets written to disk (the real
    write always uses the unredacted `result.mcp_json`/`generated_mcp_
    servers`, so the API key still ends up correctly configured on disk).

    Found in PR #113 review: `generated_mcp_servers` is already scoped to
    just this run's own toolkit-owned content, not the full merged document
    (see SetupResult's docstring for why that matters for OTHER servers'
    secrets) -- but it still includes the REAL `--qdrant-api-key` value the
    user just typed on this exact command line, so --dry-run was printing
    that credential straight to the terminal/CI log right along with
    everything else this toolkit generated.
    """
    redacted = copy.deepcopy(mcp_servers)
    for server in redacted.values():
        if isinstance(server, dict) and server.get("env", {}).get("QDRANT_API_KEY"):
            server["env"]["QDRANT_API_KEY"] = _REDACTED
    return redacted


_QDRANT_MCP_JSON_COMMIT_CAUTION = (
    "  - WARNING: --qdrant-api-key was set, so .mcp.json now contains that credential in PLAINTEXT "
    "across four env blocks (qdrant/codebase-indexer/memory-bank/local-compress). Do NOT commit .mcp.json as-is "
    "if this repo is or will be shared -- either keep .mcp.json out of version control for this "
    "project (add it to .gitignore) or manage the key through your own untracked mechanism instead "
    "of committing it."
)


_DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"


def _hook_env_reminders(args: argparse.Namespace) -> list:
    """
    --track-savings/--lmstudio-model/--lmstudio-url/--savings-db only reach
    the local-compress MCP server's env block in the generated .mcp.json --
    they do NOT reach compress_bash_output.py/session_end_savings.py, the
    hook scripts written into .claude/settings.json, since Claude Code hook
    entries have no env field of their own and instead inherit the shell's
    environment unfiltered (see README's "Environment variables" section).
    Found in PR #113 review: this script previously reported these options
    as fully applied with no indication that the hook-side half of the same
    setting still needs a matching shell export, so e.g. --track-savings
    silently dropped every hook-side event while looking fully configured.
    """
    reminders = []
    if args.track_savings:
        reminders.append("CLAUDE_RUNWAY_TRACK_SAVINGS=1")
    if args.savings_db:
        reminders.append(f"CLAUDE_RUNWAY_SAVINGS_DB={args.savings_db}")
    if args.lmstudio_model:
        reminders.append(f"CLAUDE_RUNWAY_LMSTUDIO_MODEL={args.lmstudio_model}")
    if args.lmstudio_url != _DEFAULT_LMSTUDIO_URL:
        reminders.append(f"CLAUDE_RUNWAY_LMSTUDIO_URL={args.lmstudio_url}")
    return reminders


def _quote_command(argv: list, *, windows: bool) -> str:
    """
    Renders `argv` as a copy-pasteable shell command for the CURRENT
    platform. Takes `windows` as an explicit parameter (mirroring
    setup_project_lib.venv_python_path's same pattern) rather than reading
    `os.name` internally, so this stays testable without patching a global
    that `pathlib.Path` itself also depends on (patching `os.name` broke
    `Path()` instantiation entirely when tested on a real POSIX machine).

    shlex.quote produces POSIX single-quoted output, which cmd.exe treats
    as LITERAL characters rather than quoting -- found in PR #113 review
    that README documents Windows cmd.exe usage too, so a path containing a
    space would print an uncopyable command there. subprocess.list2cmdline
    renders the same argv using cmd.exe/CreateProcess's own quoting rules
    instead.
    """
    if windows:
        return subprocess.list2cmdline(argv)
    return " ".join(shlex.quote(a) for a in argv)


def cmd_init(args: argparse.Namespace) -> None:
    target_repo = Path(args.target_repo).resolve()
    if not target_repo.is_dir():
        print(f"error: target repo path does not exist or is not a directory: {target_repo}", file=sys.stderr)
        sys.exit(1)

    # Issue #198: --qdrant-only no longer suppresses settings.json entirely --
    # record_session_id.py is CORE/base install and must be written (and, on
    # a rerun after a prior full setup, keep the compress-gated blocks
    # stripped) regardless of --qdrant-only. Only --skip-hooks (the explicit
    # full opt-out) leaves settings.json completely untouched.
    include_hooks = not args.skip_hooks
    # None when TOOLS_REPO_DIR/.venv exists (the documented clone workflow --
    # see _pip_installed_venv_python's docstring): run_setup then falls back
    # to its own pure venv_python_path default exactly as before this issue,
    # so this is a no-op for every existing user/test of that flow.
    venv_python_override = _pip_installed_venv_python()
    result = run_setup(
        target_repo,
        TOOLS_REPO_DIR,
        collection_name=args.collection_name,
        venv_python=venv_python_override,
        windows=(os.name == "nt"),
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_description=args.collection_description,
        include_extensions=args.include_extensions,
        exclude_dirs=args.exclude_dirs,
        lmstudio_url=args.lmstudio_url,
        lmstudio_model=args.lmstudio_model,
        track_savings=args.track_savings,
        savings_db=args.savings_db,
        compact_collection=args.compact_collection,
        memory_bank_collection=args.memory_bank_collection,
        memory_bank_id=args.memory_bank_id,
        include_compress=not args.qdrant_only,
        include_hooks=include_hooks,
        # clean_hooks_if_unused's own special-case path is only relevant
        # when include_hooks=False (--skip-hooks now, exclusively) -- since
        # --qdrant-only keeps include_hooks=True (issue #198: core hooks are
        # always written), build_settings_hooks(include_compress=False)
        # already drops the compress-gated blocks, and merge_settings_hooks
        # strips any STALE compress-gated block a prior full setup left in
        # an existing settings.json as part of the normal include_hooks=True
        # path -- no separate cleanup branch needed for that case anymore.
        clean_hooks_if_unused=False,
    )

    for change in result.changes:
        print(("Would write " if args.dry_run else "Writing ") + change)

    # NOT the same condition as include_hooks: --skip-hooks leaves
    # settings.json (and therefore any hooks a PRIOR full setup already
    # wrote there) completely untouched -- it does not remove them -- so
    # the hook-side env mismatch this reminder warns about is still just as
    # real under --skip-hooks as under a normal run. Only --qdrant-only is
    # guaranteed to leave no COMPRESS-DEPENDENT hooks active (removal
    # happens through build_settings_hooks(include_compress=False) plus the
    # normal merge path above -- NOT clean_hooks_if_unused, which is
    # hardcoded False here; see its own comment above for why) -- issue
    # #198 means it no longer removes ALL toolkit hooks (the CORE
    # record_session_id.py hook is written/kept either way), but that core
    # hook doesn't read any of the env vars this reminder is about, so
    # suppressing it under --qdrant-only is still correct -- just not
    # because every toolkit hook is gone. Found in PR #113 review: gating
    # on include_hooks silenced the reminder for --skip-hooks +
    # --track-savings/--lmstudio-model/etc, even though any existing hooks
    # would keep running with stale/default values.
    reminders = _hook_env_reminders(args) if not args.qdrant_only else []
    if reminders:
        print(
            "\nNote: "
            + ", ".join(reminders)
            + " only configure(s) the local-compress MCP server's env block. "
            "The generated PostToolUse/SessionEnd hooks have no env block of their own -- "
            "they inherit your shell's environment unfiltered instead. Export the same "
            "value(s) at the shell level too (e.g. in ~/.zshrc/~/.bashrc, wherever `claude` "
            "is launched from), or the hooks will silently keep using their own defaults. "
            "See README.md's 'Environment variables' section."
        )

    # Use the ACTUAL resolved collection name/Qdrant URL/include-exclude
    # filters from this run, not a generic placeholder -- found in PR #113
    # review across two rounds: a hardcoded "<name>" with no --qdrant-url
    # either fails to copy-paste as-is or silently indexes the wrong Qdrant
    # instance, and separately, omitting --include-ext/--exclude-dirs when
    # they were actually set means the copied command indexes a different
    # corpus than the codebase-indexer config this run just generated.
    # Quoted per-platform via _quote_command (see its docstring) so the
    # printed command is directly copy-pasteable even if target_repo (or
    # any option value) contains a space.
    resolved_collection_name = result.mcp_json["mcpServers"]["qdrant"]["env"]["COLLECTION_NAME"]
    # A pip/pipx/uvx install (venv_python_override set, see above) puts
    # claude-runway-ingest on PATH -- no "python /path/to/script.py" needed,
    # and no local clone whose path this reminder could even print. The
    # clone-based workflow's reminder is unchanged (issue #50): still the
    # exact `python tools/ingest_to_qdrant.py` invocation it always was.
    if venv_python_override is not None:
        index_argv = ["claude-runway-ingest"]
    else:
        index_argv = ["python", str(TOOLS_REPO_DIR / "tools" / "ingest_to_qdrant.py")]
    index_argv += [
        "--repo-path",
        str(target_repo),
        "--collection",
        resolved_collection_name,
        "--qdrant-url",
        args.qdrant_url,
    ]
    if args.include_extensions:
        index_argv += ["--include-ext", args.include_extensions]
    if args.exclude_dirs:
        index_argv += ["--exclude-dirs", args.exclude_dirs]
    if args.qdrant_api_key:
        # ingest_to_qdrant.py supports --qdrant-api-key and needs it to
        # reach an authenticated instance (found in PR #113 review: the
        # reminder previously omitted it entirely, so the suggested command
        # failed against a configured remote Qdrant) -- but the FLAG is
        # included with a placeholder, never the real value, so this
        # reminder itself never echoes the plaintext credential to the
        # terminal/CI log (the same leak class as generated_mcp_servers,
        # see _redact_secrets).
        index_argv += ["--qdrant-api-key", "<your-qdrant-api-key>"]
    index_argv.append("--dry-run")
    index_command = _quote_command(index_argv, windows=(os.name == "nt"))
    next_step_note = (
        "  - Run the initial index"
        + ("" if venv_python_override is not None else " (from this tools repo, venv activated)")
        + f":\n      {index_command}   # then again without --dry-run"
    )

    if args.dry_run:
        # Preview ONLY this run's own generated content, never
        # result.mcp_json/settings_json (the FULL merged document) --
        # found in PR #113 review that printing the full merged document
        # here leaked any unrelated existing MCP server's secret (an API
        # token sitting in that server's own env block, merged in
        # unchanged from the target's existing .mcp.json) straight to the
        # terminal/CI log. See SetupResult's docstring for the full
        # reasoning.
        print("\n--dry-run: nothing written. This toolkit would add/refresh in .mcp.json's mcpServers:")
        # _redact_secrets only affects this printed preview -- the real
        # write below always uses the unredacted result.mcp_json, so
        # --qdrant-api-key still ends up correctly configured on disk.
        print(json.dumps(_redact_secrets(result.generated_mcp_servers), indent=2))
        if result.generated_settings_hooks is not None:
            print("\nThis toolkit would add/refresh in .claude/settings.json's hooks:")
            print(json.dumps(result.generated_settings_hooks, indent=2))
        print(
            "\n(Any unrelated existing .mcp.json servers or .claude/settings.json hooks are preserved "
            "untouched -- not shown here since this is a preview of only what this toolkit itself "
            "contributes, not the full merged file."
            + (f" QDRANT_API_KEY is shown as {_REDACTED} above, not its real value." if args.qdrant_api_key else "")
            + ")"
        )
        print("\nAfter writing for real, remember to:")
        print(next_step_note)
        if args.qdrant_api_key:
            print(_QDRANT_MCP_JSON_COMMIT_CAUTION)
        return

    # Write order matters specifically for --qdrant-only: it REMOVES
    # local-compress from .mcp.json and REMOVES this toolkit's
    # local-compress-dependent hooks from settings.json in the same run
    # (issue #198: the CORE record_session_id.py hook is written/kept
    # either way, not removed). Found in PR #113 review: writing
    # .mcp.json first meant that if the settings.json write then failed
    # (e.g. an unwritable .claude/ directory), local-compress was already
    # gone but the stale PreToolUse hook was still active -- denying
    # WebFetch and redirecting to a now-unconfigured fetch_url, exactly the
    # inconsistent state --qdrant-only exists to prevent. Removing the
    # hooks FIRST means a failure on the second (.mcp.json) write instead
    # leaves local-compress still fully configured and working -- "not yet
    # qdrant-only" rather than "broken". A normal (non-qdrant-only) run
    # only ever ADDS/refreshes both files, where this ordering doesn't
    # matter either way, so it's left as mcp.json-then-settings.json there.
    if args.qdrant_only and result.settings_json is not None:
        _write_json(target_repo / ".claude" / "settings.json", result.settings_json)
        _write_json(target_repo / ".mcp.json", result.mcp_json)
    else:
        _write_json(target_repo / ".mcp.json", result.mcp_json)
        if result.settings_json is not None:
            _write_json(target_repo / ".claude" / "settings.json", result.settings_json)

    print("\nDone. Remember to:")
    print(next_step_note)
    if args.qdrant_api_key:
        # Found in PR #113 review: unconditionally telling the user to
        # commit .mcp.json is actively harmful once a real credential has
        # just been written into it in plaintext -- swap the instruction
        # for a warning instead of adding the warning alongside it.
        print(_QDRANT_MCP_JSON_COMMIT_CAUTION)
        print("  - Do NOT commit .claude/settings.json — it contains absolute paths specific to this machine.")
    else:
        print("  - Commit .mcp.json to the repo. Do NOT commit .claude/settings.json — it contains absolute paths specific to this machine.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Write .mcp.json/.claude/settings.json into a target repo")
    init.add_argument("target_repo", help="Path to the project repo to configure")
    init.add_argument(
        "--collection-name",
        default=None,
        help="Qdrant collection name; defaults to a slug derived from the target repo's directory name",
    )
    init.add_argument("--qdrant-url", default="http://localhost:6333")
    init.add_argument(
        "--qdrant-api-key",
        default="",
        help="API key for an authenticated remote Qdrant instance; leave blank for unauthenticated local "
        "Qdrant. Applied to all four servers' QDRANT_API_KEY -- not 'sticky': pass it on every run, "
        "the same as --qdrant-url/--collection-name, or a rerun without it resets these servers back "
        "to unauthenticated. WARNING: writes the key in PLAINTEXT into .mcp.json -- do not commit that "
        "file as-is if this repo is shared; this script will print a reminder instead of the usual "
        "'commit .mcp.json' instruction when this is set",
    )
    init.add_argument("--collection-description", default="", help="Short one-line hint for list_collections")
    init.add_argument("--include-extensions", default="", help="e.g. '.py,.md' -- blank uses the built-in defaults")
    init.add_argument("--exclude-dirs", default="", help="e.g. 'fixtures,generated' -- added to the built-in excludes")
    init.add_argument("--lmstudio-url", default="http://localhost:1234/v1")
    init.add_argument(
        "--lmstudio-model",
        default="",
        help="Exact model id loaded in LM Studio; leave blank to auto-detect (only works if exactly one model is loaded)",
    )
    init.add_argument("--track-savings", action="store_true", help="Enable the opt-in savings tracker")
    init.add_argument(
        "--savings-db", default="", help="Override the savings tracker DB path (leave blank for the default)"
    )
    init.add_argument(
        "--memory-bank-collection",
        default="",
        help="Base Qdrant collection name the memory-bank server's remember/recall/forget tools share "
        "across EVERY project (tools/memory_bank_mcp_server.py's MEMORY_BANK_COLLECTION); leave blank to "
        "use the 'memory-bank' default. Unlike --collection-name, this is deliberately meant to be the SAME "
        "value across every project on a machine -- every project's memories need to agree on this to "
        "actually be found by each other. Only change it if you're deliberately isolating a separate "
        "memory-bank collection from this machine's default.",
    )
    init.add_argument(
        "--memory-bank-id",
        default="",
        help="Plain identifier tagging this project's own memories in the shared memory-bank "
        "collection (memory_bank_mcp_server.py's MEMORY_BANK_ID); leave blank to default to "
        "--collection-name. Unlike --memory-bank-collection, this IS meant to be per-project -- "
        "but it's independent of --collection-name, not derived from it every run: pass the "
        "existing value explicitly on a re-run of a project that has already accumulated real "
        "memories, or changing --collection-name later would silently retag future memories "
        "while stranding existing ones under the old identifier (issue #175, PR #178 review).",
    )
    init.add_argument(
        "--compact-collection",
        default="",
        help="Base Qdrant collection name the my-compact/my-resume skills store conversation compacts under "
        "(compress_mcp_server.py's COMPACT_COLLECTION); leave blank to use the shared 'conversation-compacts' "
        "default. NOT 'sticky' to a per-project value on its own (matches every other option here): omitting "
        "this flag ALWAYS resolves to 'conversation-compacts' regardless of what a project's .mcp.json currently "
        "has, so a re-run is only a no-op for a project that has never customized this (via /my-setup-clauderunway's "
        "own Q5, or a prior explicit --compact-collection) -- for a project that HAS, omitting this flag on a "
        "later raw-CLI re-run resets it back to 'conversation-compacts', orphaning that project's existing "
        "conversation-compact history from compact lookup. Repeat the same --compact-collection value on every "
        "re-run for such a project, or go through /my-setup-clauderunway instead, which extracts and re-passes "
        "the current value automatically. compress_mcp_server.py still appends its own "
        "-<sanitized-project>-<hash8> suffix on top of whichever base this resolves to, at runtime, for "
        "per-project isolation -- so the final on-disk collection name is "
        "'<compact-collection>-<sanitized-project>-<hash8>'.",
    )
    # Mutually exclusive: their settings.json semantics genuinely conflict.
    # --qdrant-only actively REMOVES this toolkit's local-compress-dependent
    # hooks if present (the CORE record_session_id.py hook is written/kept
    # either way, per issue #198 -- it isn't removed);
    # --skip-hooks promises to leave settings.json untouched either way.
    # Found in PR #113 review: passing both together silently let
    # --qdrant-only's cleanup win, deleting existing hooks despite
    # --skip-hooks's own promise not to write/merge settings.json at all --
    # rejecting the combination up front (before any config is read or
    # written) is clearer than picking an implicit winner.
    hooks_mode = init.add_mutually_exclusive_group()
    hooks_mode.add_argument(
        "--qdrant-only",
        action="store_true",
        help="Only configure the Qdrant codebase-memory piece: omit local-compress from .mcp.json, "
        "and remove this toolkit's local-compress-dependent hook blocks from .claude/settings.json if "
        "a prior run had added them. The CORE record_session_id.py hook (issue #198) is still written "
        "(or kept) either way, since it's base install now, not local-compress-gated -- so the file is "
        "no longer left untouched even on a project that never had any hooks before; use --skip-hooks "
        "instead if you want settings.json fully untouched",
    )
    hooks_mode.add_argument(
        "--skip-hooks",
        action="store_true",
        help="Configure local-compress in .mcp.json but don't write/merge .claude/settings.json's hooks "
        "(leaves any existing settings.json completely untouched, unlike --qdrant-only)",
    )
    init.add_argument("--dry-run", action="store_true", help="Print what would be written without writing")
    init.set_defaults(func=cmd_init)

    return p.parse_args()


def main() -> None:
    """
    Entry point for both `python tools/setup_project.py` (the
    `if __name__` guard below) and the `claude-runway-setup` console
    script pip/pipx/uvx installs (issue #50) -- pulled out into its own
    function purely so `[project.scripts]` in pyproject.toml has something
    to point `module:function` at; behavior is unchanged either way.
    """
    parsed_args = parse_args()
    parsed_args.func(parsed_args)


if __name__ == "__main__":
    main()
