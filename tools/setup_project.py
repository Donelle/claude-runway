#!/usr/bin/env python3
"""
One-time setup for a target project repo: writes that repo's `.mcp.json`
and `.claude/settings.json` with every `REPLACE-WITH-*` /
`/absolute/path/to/...` placeholder from `templates/mcp.json.template` and
`templates/settings.json.template` filled in automatically, instead of the
manual copy-and-hand-edit flow in `docs/installation.md`'s steps 4/4b/5.
Also structurally prevents stale hand-typed example paths
(BUG-14) recurring, since every path here is computed, not typed.

`init` also brings the target's `CLAUDE.md` in line with the servers it
writes (`--skip-claude-md` to leave it alone): it creates the file, or
appends the usage section from `templates/CLAUDE.md.template` for each
configured server that lacks one, and removes the section of a toolkit
server that is no longer configured (Local compression under
`--qdrant-only`, Analytics without the savings tracker). Nothing else in
the file changes, and a run with nothing to add or remove does not rewrite
it. The template is read from this install's own `templates/`, so this
works the same from a clone and from a pipx/`uv tool install`.

Usage:
    python tools/setup_project.py init /path/to/target-repo
    python tools/setup_project.py init /path/to/target-repo --dry-run
    python tools/setup_project.py init /path/to/target-repo --qdrant-only
    python tools/setup_project.py init /path/to/target-repo \
        --collection-name my-project --lmstudio-model google/gemma-3-4b --track-savings
    python tools/setup_project.py init /path/to/target-repo --install-skills
    python tools/setup_project.py init --install-skills   # skills only, no target_repo

    python tools/setup_project.py upgrade /path/to/target-repo             # apply pending config migrations + refresh installed skills
    python tools/setup_project.py upgrade /path/to/target-repo --dry-run
    python tools/setup_project.py upgrade /path/to/target-repo --auto-yes
    python tools/setup_project.py upgrade --skills-dir ~/.claude/skills    # no project path: skills only

`upgrade` is for an ALREADY-configured project: unlike `init`,
which resets every toolkit-owned setting to whatever this run's flags/
defaults are, `upgrade` compares the target's current `.mcp.json`/
`.claude/settings.json` against a fixed set of named, individually
detectable config gaps (see `libs/upgrade_lib.py`) and applies only the ones
the user approves, leaving every other config setting untouched. It
ALSO refreshes product skills that are already installed under
`~/.claude/skills/` (`--skills-dir` to override) and differ from the shipped
ones -- a second, separate step after the config migrations (same
`--dry-run`/`--auto-yes` semantics), which rewrites only the body of an
installed SKILL.md (its own frontmatter is kept) and never installs a skill
that is not already there. Run as the `claude-runway-setup` console script,
that refresh compares against the skills shipped with that
install -- a snapshot taken when it was installed, not a checkout's
`skills/` -- so it says so, and warns about each stale skill whose copy in
the `CLAUDE_RUNWAY_DIR` checkout differs; `/cr-setup upgrade`
reads the checkout itself. With no project path, `upgrade` runs skills-only:
the rename and the refresh on `--skills-dir` and nothing else, so someone who
installed skills without ever configuring a project can still pick up the
`my-*` to `cr-*` rename.

The savings tracker is opt-in (it is what writes the `analytics` MCP server behind
/cr-savings and /cr-metrics). `init` and `upgrade` ask whether to enable it, with the same
wording as /cr-setup's Q4, but only when stdin is a terminal: a pipe or CI run,
`--dry-run`, `--auto-yes` (upgrade) and `--qdrant-only` (init) never ask and keep it off, and an
explicit `--track-savings` skips the question. When it ends up off, `init` says what that
costs and how to turn it on.

The `codebase-reviewer` MCP server (the local review loop) is opt-in too: `init --codebase-reviewer`
(optionally `--reviewers-file PATH`) writes its block. `upgrade` does not offer it while the server
is unreleased, and the mcp.json template carries no block for it. Its REVIEWERS_FILE is required and
has no default, so `init` never writes a default path or a reviewers file; it warns when it is not set.

The templates stay the single source of truth: this script loads and
patches them (see libs/setup_project_lib.py) rather than duplicating their
content, so future template changes flow through to new setups automatically.

Safe to re-run: existing `.mcp.json`/`.claude/settings.json` content this
toolkit doesn't own (other MCP servers, other hooks) is preserved, and
re-running with the same options doesn't duplicate hook entries -- see
libs/setup_project_lib.py's merge_mcp_json/merge_settings_hooks.

Project minimum: Python 3.12.
"""

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from setup_project_lib import (  # noqa: E402
    RENAMED_PRODUCT_SKILLS,
    SKILL_TEXT_ERRORS,
    MalformedConfigError,
    ask_savings_tracker,
    default_skills_dir,
    has_analytics_block,
    load_existing_mcp_json,
    local_compress_tracks_savings,
    plan_skill_installs,
    plan_skill_renames,
    run_setup,
    venv_python_path,
)
from upgrade_lib import is_project_configured, run_upgrade  # noqa: E402
from version_lib import get_version, version_string  # noqa: E402

TOOLS_REPO_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pip_installed_venv_python() -> Optional[Path]:
    """
    `setup_project_lib.venv_python_path`'s DEFAULT assumes the
    "clone this repo, `uv venv` right next to it" layout
    `docs/installation.md`'s step 3 documents -- `TOOLS_REPO_DIR / ".venv" / "bin" /
    "python"`. That assumption breaks when this script runs as the
    `claude-runway-setup` console script from a `pipx`/`uv tool install`
    install (the two methods `docs/installation.md`'s "Alternative" callout actually
    documents): there IS no `.venv` sibling directory next to the installed
    `tools`/`libs` directories (wheel data files under `<env>/src`)
    -- pipx/uv tool manage their
    own venv elsewhere, and `sys.executable` (whichever interpreter is
    actually running this console script right now) is the only
    generally-correct answer in that case, since that's exactly the
    environment `pip install claude-runway`'s own dependencies were
    installed into, and every MCP server in the generated `.mcp.json` runs
    its script under that one interpreter.

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


def _atomic_write(path: Path, content: str) -> None:
    """
    Writes text atomically: serialize to a temp file in the SAME directory
    as the destination, flush + fsync it, then os.replace() over the real
    path. Found in review: the previous open(path, "w") truncated
    the destination immediately, so an interruption or disk-full error
    mid-write could destroy the target repo's existing .mcp.json/.claude/
    settings.json -- exactly the content this script's merge logic exists
    to preserve, not clobber. os.replace() is atomic on both POSIX and
    Windows as long as source and destination share a filesystem, which a
    temp file created in the destination's own directory guarantees.

    Factored out of _write_json so --install-skills's SKILL.md
    writes share the exact same atomicity guarantee instead of a second,
    possibly-drifted copy of this logic -- a skill file is no less worth
    protecting from a truncate-then-crash than .mcp.json/settings.json is.

    Writes via a BINARY-mode file handle (`content.encode("utf-8")`), not
    a text-mode one: opening in text mode
    without an explicit `newline=""` translates every "\n" in `content` to
    the PLATFORM's own line ending on write (CRLF on Windows) -- combined
    with `libs/setup_project_lib.py`'s `plan_skill_installs` reading a
    skill's source as raw bytes specifically to preserve its ORIGINAL line
    endings (see that function's own comment), a text-mode write here would
    silently reintroduce a mismatch: the freshly-written dest file would
    have CRLF while the next run's freshly re-read `content` has whatever
    the source actually contains, so the byte comparison there would never
    settle into "up_to_date" on Windows -- every run would report (and
    rewrite) the same install as "update", forever. Binary mode does no
    newline translation at all, so what's encoded is exactly what lands on
    disk, matching plan_skill_installs' own byte-for-byte comparison.
    """
    _atomic_write_bytes(path, content.encode("utf-8", errors=SKILL_TEXT_ERRORS))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """The byte-level half of `_atomic_write` (same temp file in the same
    directory, fsync, `os.replace`), split out so a skill's
    non-text-edited script is written exactly as shipped, with no encode step
    in between."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, data: dict) -> None:
    """Thin JSON-serializing wrapper over _atomic_write -- see that
    function's docstring for the actual atomicity mechanics."""
    _atomic_write(path, json.dumps(data, indent=2) + "\n")


def _write_claude_md(path: Path, content: str) -> None:
    """
    Writes the planned CLAUDE.md through `_atomic_write`. A CLAUDE.md that is
    a symlink (to a shared AGENTS.md, say) is written at the file it points
    to: `os.replace` onto the link itself would swap the link for a regular
    file and silently cut the project off from the shared one.
    """
    _atomic_write(path.resolve() if path.is_symlink() else path, content)


_REDACTED = "***REDACTED***"


def _redact_secrets(mcp_servers: dict) -> dict:
    """
    Returns a deep copy of `mcp_servers` with any non-empty `QDRANT_API_KEY`
    value replaced by a redaction marker -- used ONLY for --dry-run's
    printed preview, never for what actually gets written to disk (the real
    write always uses the unredacted `result.mcp_json`/`generated_mcp_
    servers`, so the API key still ends up correctly configured on disk).

    Found in review: `generated_mcp_servers` is already scoped to
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
    "across the env blocks of the Qdrant-talking servers (qdrant/codebase-indexer/memory-bank, and "
    "codebase-reviewer when present). Do NOT commit .mcp.json as-is "
    "if this repo is or will be shared -- either keep .mcp.json out of version control for this "
    "project (add it to .gitignore) or manage the key through your own untracked mechanism instead "
    "of committing it."
)


_DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"


def _hook_env_reminders(args: argparse.Namespace) -> list:
    """
    --track-savings/--lmstudio-model/--lmstudio-url/--savings-db only reach
    the MCP servers' env blocks in the generated .mcp.json (local-compress
    for all but --savings-db, for the two LM Studio values also qdrant, and
    for --track-savings/--savings-db the opt-in analytics block, the only
    server that opens the savings database) --
    they do NOT reach compress_output.py/session_end_savings.py, the
    hook scripts written into .claude/settings.json, since Claude Code hook
    entries have no env field of their own and instead inherit the shell's
    environment unfiltered (see README's "Environment variables" section).
    Found in review: this script previously reported these options
    as fully applied with no indication that the hook-side half of the same
    setting still needs a matching shell export, so e.g. --track-savings
    silently dropped every hook-side event while looking fully configured.
    """
    reminders = []
    if args.track_savings:
        reminders.append("CLAUDE_RUNWAY_TRACK_SAVINGS=1")
    if args.savings_db:
        reminders.append(f"CLAUDE_RUNWAY_SAVINGS_DB={args.savings_db}")
    # CLAUDE_RUNWAY_METRICS_DB is deliberately not here: no hook reads it, so it
    # is not a hook-side reminder (and this list is skipped under --qdrant-only,
    # where the override is still written). See _metrics_db_reminder.
    if args.lmstudio_model:
        reminders.append(f"CLAUDE_RUNWAY_LMSTUDIO_MODEL={args.lmstudio_model}")
    if args.lmstudio_url != _DEFAULT_LMSTUDIO_URL:
        reminders.append(f"CLAUDE_RUNWAY_LMSTUDIO_URL={args.lmstudio_url}")
    return reminders


def _metrics_db_reminder(args: argparse.Namespace) -> str:
    """
    The note printed for `--metrics-db`, or "" when it was not passed. Separate
    from `_hook_env_reminders` because the reason is different: no hook reads
    the metrics database, but the reporting CLIs run from a shell
    (`tools/autowork_cost.py` and friends) read `CLAUDE_RUNWAY_METRICS_DB` from
    the shell, and the override is written into the memory-bank block (and the
    codebase-reviewer block when it is written, and the analytics block with --track-savings) in EVERY init mode, `--qdrant-only`
    included, so the note must not depend on the hook-side suppression either.
    """
    if not args.metrics_db:
        return ""
    return (
        f"\nNote: --metrics-db wrote CLAUDE_RUNWAY_METRICS_DB={args.metrics_db} into the memory-bank "
        "server's env block (and the codebase-reviewer block when written, and the analytics block, with --track-savings). The reporting CLIs run from "
        "a shell (e.g. tools/autowork_cost.py) read it from the shell environment, so export the same "
        "value there too, or they will read the default database and miss these events."
    )


def _stdin_is_interactive() -> bool:
    """
    True when stdin is a terminal, i.e. a person can answer a question. A
    pipe, a CI run and a redirected file are not, and neither is a closed
    stdin; any of those keeps `init`/`upgrade` non-interactive. A function,
    not an inline `isatty()`, so tests can pin it.
    """
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _existing_mcp_json_or_empty(target_repo: Path) -> dict:
    """The target's current `.mcp.json`, or {} when it is absent or unusable.
    A malformed file is not reported here: `run_setup` reads it again right
    after and raises `MalformedConfigError` with the message the user needs."""
    try:
        return load_existing_mcp_json(target_repo / ".mcp.json")
    except MalformedConfigError:
        return {}


_SAVINGS_OFF_NOTE = (
    "\nNote: the savings tracker is off, so no `analytics` MCP server is written. /cr-savings and /cr-metrics "
    "will fail with an unknown-tool error until it is. To enable it, re-run init with --track-savings and "
    "`export CLAUDE_RUNWAY_TRACK_SAVINGS=1` in your shell (see docs/savings-tracker.md)."
)


def _savings_tracker_note(args: argparse.Namespace, *, had_analytics: bool) -> str:
    """
    What `init` says about the savings tracker when it ends up off,
    or "" when it is on or the setup is `--qdrant-only` (no
    local-compress, so no tracker to offer). A run that removes an `analytics`
    block an earlier run wrote says so rather than doing it silently: a bare
    re-run resets every toolkit-owned setting to this run's flags.
    """
    if args.track_savings:
        return ""
    verb = "would remove" if args.dry_run else "is removing"
    if args.qdrant_only:
        if not had_analytics:
            return ""
        return (
            f"\nNote: --qdrant-only {verb} the existing `analytics` MCP server along with local-compress, so "
            "/cr-savings and /cr-metrics stop working. Re-run without --qdrant-only (and with --track-savings) "
            "to keep it."
        )
    if had_analytics:
        return (
            f"\nNote: this run {verb} the existing `analytics` MCP server because the savings tracker is off "
            "(init resets it to this run's flags), so /cr-savings and /cr-metrics stop working. Pass "
            "--track-savings to keep it."
        )
    return _SAVINGS_OFF_NOTE


def _quote_command(argv: list, *, windows: bool) -> str:
    """
    Renders `argv` as a copy-pasteable shell command for the CURRENT
    platform. Takes `windows` as an explicit parameter (mirroring
    setup_project_lib.venv_python_path's same pattern) rather than reading
    `os.name` internally, so this stays testable without patching a global
    that `pathlib.Path` itself also depends on (patching `os.name` broke
    `Path()` instantiation entirely when tested on a real POSIX machine).

    shlex.quote produces POSIX single-quoted output, which cmd.exe treats
    as LITERAL characters rather than quoting -- found in review
    that README documents Windows cmd.exe usage too, so a path containing a
    space would print an uncopyable command there. subprocess.list2cmdline
    renders the same argv using cmd.exe/CreateProcess's own quoting rules
    instead.
    """
    if windows:
        return subprocess.list2cmdline(argv)
    return " ".join(shlex.quote(a) for a in argv)


_CR_SETUP_CAVEAT = (
    "Note: cr-setup is among the skills covered above, but it still requires "
    "CLAUDE_RUNWAY_DIR (exported in your shell) pointing at a full clone with tools/setup_project.py "
    "on disk -- it checks this itself at its own step 1, and can't bootstrap itself purely from a "
    "pipx/`uv tool install` install. Harmless to have installed if you don't use it from this environment."
)


def _resolve_skills_dir(args: argparse.Namespace) -> Path:
    """Shared --skills-dir resolution for both cmd_init branches below --
    defaults to setup_project_lib.default_skills_dir(~) when --skills-dir
    wasn't passed, otherwise uses the given path as-is (resolved, so a
    relative --skills-dir behaves predictably regardless of cwd)."""
    if args.skills_dir:
        return Path(args.skills_dir).resolve()
    return default_skills_dir(Path.home())


def _apply_skill_plan(plan) -> None:
    """Writes one skill's SKILL.md and each of its files that is not already
    up to date. The one place both `init --install-skills` and
    `upgrade`'s refresh write a skill, so the two cannot drift. Each file is
    written atomically; the script goes first so SKILL.md, the file Claude
    Code lists the skill from, never points at a script that is not there."""
    for extra in plan.extra_files:
        if extra.action != "up_to_date":
            _atomic_write_bytes(extra.dest, extra.data)
    _atomic_write(plan.dest, plan.content)


def _report_and_apply_skill_plan(plans: list, *, dry_run: bool) -> None:
    """
    Prints per-skill install/update/up-to-date status
    and, unless --dry-run, actually writes each non-up-to-date skill (its
    SKILL.md and the files beside it) via `_apply_skill_plan`,
    which uses the atomic writers. Kept as one function (not split into "plan" + "print" +
    "apply" separately) since every existing caller in this file always
    wants both together -- --dry-run's "print only" behavior is expressed
    via the dry_run flag rather than by the caller simply not calling the
    apply half, so there's exactly one place this pairing can drift.
    """
    if not plans:
        print("  (no skills found under skills/ to install -- check this installation)")
        return
    verb = "Would install" if dry_run else "Installing"
    verb_update = "Would update" if dry_run else "Updating"
    for plan in plans:
        if plan.action == "up_to_date":
            print(f"  - {plan.name}: up to date ({plan.dest})")
        elif plan.action == "install":
            print(f"  - {verb} {plan.name} -> {plan.dest}")
        else:
            print(f"  - {verb_update} {plan.name} -> {plan.dest}")
        # The files beside SKILL.md, named so a dry run says what
        # else would be written; an up-to-date one is not worth a line.
        for extra in plan.extra_files:
            if extra.action != "up_to_date":
                print(f"      + {extra.name} -> {extra.dest}")
        if not dry_run and plan.action != "up_to_date":
            _apply_skill_plan(plan)
    if any(p.name == "cr-setup" for p in plans):
        print(_CR_SETUP_CAVEAT)


def cmd_init(args: argparse.Namespace) -> None:
    # `target_repo` is now optional (nargs="?") specifically so
    # `--install-skills` can run alone, with no project to configure at all
    # -- parse_args() already refuses this combination (target_repo absent
    # AND --install-skills absent) before cmd_init is ever called, so
    # reaching here with target_repo is None means --install-skills is
    # necessarily set. Skills-only mode touches nothing under target_repo --
    # no .mcp.json, no .claude/settings.json, no run_setup call at all.
    if args.target_repo is None:
        skills_dir = _resolve_skills_dir(args)
        plans = plan_skill_installs(TOOLS_REPO_DIR, skills_dir)
        print(("Would install/update skills under " if args.dry_run else "Installing/updating skills under ") + f"{skills_dir}:")
        _report_and_apply_skill_plan(plans, dry_run=args.dry_run)
        warn_old_skill_names(skills_dir)
        return

    target_repo = Path(args.target_repo).resolve()
    if not target_repo.is_dir():
        print(f"error: target repo path does not exist or is not a directory: {target_repo}", file=sys.stderr)
        sys.exit(1)

    # --qdrant-only no longer suppresses settings.json entirely --
    # record_session_id.py is CORE/base install and must be written (and, on
    # a rerun after a prior full setup, keep the compress-gated blocks
    # stripped) regardless of --qdrant-only. Only --skip-hooks (the explicit
    # full opt-out) leaves settings.json completely untouched.
    include_hooks = not args.skip_hooks
    # None when TOOLS_REPO_DIR/.venv exists (the documented clone workflow --
    # see _pip_installed_venv_python's docstring): run_setup then falls back
    # to its own pure venv_python_path default exactly as before,
    # so this is a no-op for every existing user/test of that flow.
    venv_python_override = _pip_installed_venv_python()

    # The savings tracker (and with it the `analytics` server that
    # /cr-savings and /cr-metrics need) is opt-in, and a CLI user was never
    # told. Ask once, with the wording /cr-setup's Q4 uses, only
    # when nobody has answered already and a person can: an explicit
    # --track-savings skips the question, a non-terminal stdin, --dry-run (a
    # preview writes and asks nothing) and --qdrant-only (no local-compress,
    # so no tracker) keep the off default. Blank input keeps whatever the
    # project already has, so pressing Enter on a re-run cannot drop it.
    existing_mcp = _existing_mcp_json_or_empty(target_repo)
    had_analytics = has_analytics_block(existing_mcp)
    if not args.track_savings and not args.qdrant_only and not args.dry_run and _stdin_is_interactive():
        args.track_savings = ask_savings_tracker(
            default_enabled=had_analytics or local_compress_tracks_savings(existing_mcp)
        )

    try:
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
            metrics_db=args.metrics_db,
            compact_collection=args.compact_collection,
            memory_bank_collection=args.memory_bank_collection,
            memory_bank_id=args.memory_bank_id,
            embedding_model=args.embedding_model,
            memory_bank_embedding_model=args.memory_bank_embedding_model,
            memory_bank_follow=args.memory_bank_follow,
            codebase_reviewer=args.codebase_reviewer,
            reviewers_file=args.reviewers_file,
            claude_md=not args.skip_claude_md,
            include_compress=not args.qdrant_only,
            include_hooks=include_hooks,
            # clean_hooks_if_unused's own special-case path is only relevant
            # when include_hooks=False (--skip-hooks now, exclusively) -- since
            # --qdrant-only keeps include_hooks=True (core hooks are
            # always written), build_settings_hooks(include_compress=False)
            # already drops the compress-gated blocks, and merge_settings_hooks
            # strips any STALE compress-gated block a prior full setup left in
            # an existing settings.json as part of the normal include_hooks=True
            # path -- no separate cleanup branch needed for that case anymore.
            clean_hooks_if_unused=False,
        )
    except MalformedConfigError as exc:
        # An existing .mcp.json/.claude/settings.json that is not
        # valid JSON or has the wrong shape. run_setup reads both before it
        # computes anything to write, so nothing has been written yet.
        print(f"error: {exc} -- fix or remove it, then re-run init", file=sys.stderr)
        sys.exit(1)

    for change in result.changes:
        print(("Would write " if args.dry_run else "Writing ") + change)
    if result.claude_md is not None and result.claude_md.content is None:
        print(f"CLAUDE.md already up to date, no changes ({result.claude_md.path})")

    # NOT the same condition as include_hooks: --skip-hooks leaves
    # settings.json (and therefore any hooks a PRIOR full setup already
    # wrote there) completely untouched -- it does not remove them -- so
    # the hook-side env mismatch this reminder warns about is still just as
    # real under --skip-hooks as under a normal run. Only --qdrant-only is
    # guaranteed to leave no COMPRESS-DEPENDENT hooks active (removal
    # happens through build_settings_hooks(include_compress=False) plus the
    # normal merge path above -- NOT clean_hooks_if_unused, which is
    # hardcoded False here; see its own comment above for why) -- it
    # no longer removes ALL toolkit hooks (the CORE
    # record_session_id.py hook is written/kept either way), but that core
    # hook doesn't read any of the env vars this reminder is about, so
    # suppressing it under --qdrant-only is still correct -- just not
    # because every toolkit hook is gone. Found in review: gating
    # on include_hooks silenced the reminder for --skip-hooks +
    # --track-savings/--lmstudio-model/etc, even though any existing hooks
    # would keep running with stale/default values.
    reminders = _hook_env_reminders(args) if not args.qdrant_only else []
    if reminders:
        print(
            "\nNote: "
            + ", ".join(reminders)
            + " only configure(s) the MCP servers' env blocks (qdrant for the LM Studio values, local-compress for all of them). "
            "The generated PostToolUse/SessionEnd hooks have no env block of their own -- "
            "they inherit your shell's environment unfiltered instead. Export the same "
            "value(s) at the shell level too (e.g. in ~/.zshrc/~/.bashrc, wherever `claude` "
            "is launched from), or the hooks will silently keep using their own defaults. "
            "See README.md's 'Environment variables' section."
        )
    metrics_reminder = _metrics_db_reminder(args)
    if metrics_reminder:
        print(metrics_reminder)
    savings_note = _savings_tracker_note(args, had_analytics=had_analytics)
    if savings_note:
        print(savings_note)

    # Use the ACTUAL resolved collection name/Qdrant URL/include-exclude
    # filters from this run, not a generic placeholder -- found in review across two
    # rounds: a hardcoded "<name>" with no --qdrant-url
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
    # clone-based workflow's reminder is unchanged: still the
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
        # ingest_to_qdrant.py's own --embedding-model default is the template
        # default, so a custom/preserved model MUST be passed
        # explicitly or the suggested command would index under the wrong
        # model and mismatch the servers this run just configured. Always
        # passed (the resolved value from what was generated), not only when
        # custom, so the reminder never depends on the two defaults agreeing.
        "--embedding-model",
        result.generated_mcp_servers["qdrant"]["env"]["EMBEDDING_MODEL"],
    ]
    if args.include_extensions:
        index_argv += ["--include-ext", args.include_extensions]
    if args.exclude_dirs:
        index_argv += ["--exclude-dirs", args.exclude_dirs]
    if args.qdrant_api_key:
        # ingest_to_qdrant.py supports --qdrant-api-key and needs it to
        # reach an authenticated instance (found in review: the
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
        # found in review that printing the full merged document
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
        if args.install_skills:
            skills_dir = _resolve_skills_dir(args)
            print(f"\nWould also install/update skills under {skills_dir}:")
            _report_and_apply_skill_plan(plan_skill_installs(TOOLS_REPO_DIR, skills_dir), dry_run=True)
            warn_old_skill_names(skills_dir)
        return

    # Write order matters specifically for --qdrant-only: it REMOVES
    # local-compress from .mcp.json and REMOVES this toolkit's
    # local-compress-dependent hooks from settings.json in the same run
    # (the CORE record_session_id.py hook is written/kept
    # either way, not removed). Found in review: writing
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
    # CLAUDE.md last: it only describes the servers, so if a config write
    # above failed it is left as it was rather than describing a config that
    # never landed.
    if result.claude_md is not None and result.claude_md.content is not None:
        _write_claude_md(result.claude_md.path, result.claude_md.content)

    print("\nDone. Remember to:")
    print(next_step_note)
    if args.qdrant_api_key:
        # Found in review: unconditionally telling the user to
        # commit .mcp.json is actively harmful once a real credential has
        # just been written into it in plaintext -- swap the instruction
        # for a warning instead of adding the warning alongside it.
        print(_QDRANT_MCP_JSON_COMMIT_CAUTION)
        print("  - Do NOT commit .claude/settings.json — it contains absolute paths specific to this machine.")
    else:
        print("  - Commit .mcp.json to the repo. Do NOT commit .claude/settings.json — it contains absolute paths specific to this machine.")

    if args.install_skills:
        skills_dir = _resolve_skills_dir(args)
        print(f"\nInstalling/updating skills under {skills_dir}:")
        _report_and_apply_skill_plan(plan_skill_installs(TOOLS_REPO_DIR, skills_dir), dry_run=False)
        warn_old_skill_names(skills_dir)


def _confirm_skill_refresh(plan) -> bool:
    """Default interactive Y/n prompt for one stale skill, mirroring
    upgrade_lib._console_prompt (blank input defaults to Yes). A closed stdin
    (EOFError, e.g. a non-interactive run without --auto-yes) counts as a
    decline rather than a crash: nothing is written, and the skill is
    reported again on the next run, exactly like a declined migration."""
    try:
        answer = input(f"      Refresh {plan.name}? [Y/n] ").strip().lower()
    except EOFError:
        print("      (no input available -- skipped; pass --auto-yes to apply without prompting)")
        return False
    return answer not in ("n", "no")


def _is_installed_snapshot(tools_repo_dir: Path) -> bool:
    """
    True when `tools_repo_dir` is the copy a `pipx`/`uv tool install` shipped
    to `<env>/src` rather than a source checkout. Same rule as
    `version_lib._checkout_root`: a checkout has `.git` (a directory, or a
    file in a worktree) right above `tools/`, an installed copy does not.
    Not `_pip_installed_venv_python()`, which is also true for a clone that
    has no `.venv` yet.
    """
    return not (Path(tools_repo_dir) / ".git").exists()


def _differing_checkout_skill(plan, tools_repo_dir: Path) -> "Path | None":
    """
    The `$CLAUDE_RUNWAY_DIR` checkout's SKILL.md for `plan`'s skill when it
    exists, is a different checkout from `tools_repo_dir`, and differs from
    the copy `plan` would write; else None. Compares the two
    shipped files byte for byte, so the installed copy's own frontmatter
    plays no part. A skill's own files beside SKILL.md (its
    script) count too: a checkout with a different script is a different
    skill even when the SKILL.md text is identical, and the path returned is
    then that file. Says nothing about which one is newer: no skill carries a
    version stamp.
    """
    raw = os.environ.get("CLAUDE_RUNWAY_DIR", "").strip()
    if not raw:
        return None
    checkout = Path(raw).expanduser()
    if not checkout.is_dir():
        return None
    try:
        if checkout.resolve() == Path(tools_repo_dir).resolve():
            return None
        theirs = checkout / "skills" / plan.name / "SKILL.md"
        if theirs.is_file() and theirs.read_bytes() != plan.source.read_bytes():
            return theirs
        for extra in plan.extra_files:
            their_extra = checkout / "skills" / plan.name / extra.name
            if their_extra.is_file() and their_extra.read_bytes() != extra.data:
                return their_extra
    except OSError:
        return None
    return None


def refresh_installed_skills(
    skills_dir: Path,
    *,
    auto_yes: bool,
    dry_run: bool,
    prompt_fn=None,
    tools_repo_dir: "Path | None" = None,
) -> "list[str]":
    """
    `upgrade`'s skill refresh: brings INSTALLED product skills
    that differ from the shipped ones up to date, reusing
    `plan_skill_installs` (one comparison, shared with `init
    --install-skills`, so the two can never disagree about what counts as
    stale -- including the frontmatter rule documented there).

    Deliberately narrower than `init --install-skills`: only skills that are
    ALREADY installed (plan action "update") are touched. A skill that is
    not installed is never installed here -- a user who skipped the skills
    must not be forced to get them by running `upgrade`; they are only told
    how many exist. Approval works like config migrations: a Y/n prompt per
    stale skill, every one with `auto_yes`, none written with `dry_run`
    (which also never prompts). Returns the names refreshed (or, for
    `dry_run`, that would be).

    Run as the `claude-runway-setup` console script, the shipped
    skills are the install-time snapshot under `<env>/src/skills`, not a
    checkout's `skills/` (which `/cr-setup` reads), so the two
    routes can offer different refreshes. Overwriting with the snapshot's copy
    is intended; this only makes it visible. A snapshot source is named up
    front (path and version), and each stale skill whose copy in the
    `$CLAUDE_RUNWAY_DIR` checkout differs gets a warning. Nothing about what
    is written, prompted or skipped depends on either message.
    """
    source_dir = Path(tools_repo_dir or TOOLS_REPO_DIR)
    plans = plan_skill_installs(source_dir, skills_dir)
    stale = [p for p in plans if p.action == "update"]
    not_installed = [p for p in plans if p.action == "install"]
    print(f"\nChecking installed skills under {skills_dir}...")
    if _is_installed_snapshot(source_dir):
        print(
            f"  Comparing against the skills shipped with this install ({source_dir / 'skills'}, "
            f"{get_version()}), not a source checkout."
        )
    if not stale:
        installed = len(plans) - len(not_installed)
        print(
            "  No installed skills to refresh."
            if installed == 0
            else "  All installed skills are up to date."
        )
    else:
        print(f"  Found {len(stale)} installed skill(s) that differ from the shipped version:")
    ask = prompt_fn or _confirm_skill_refresh
    refreshed: list = []
    for plan in stale:
        note = " (keeping its existing frontmatter)" if plan.frontmatter_preserved else ""
        differing = _differing_checkout_skill(plan, source_dir)
        warning = (
            f"      Warning: the checkout at {os.environ.get('CLAUDE_RUNWAY_DIR', '').strip()} ships a different "
            f"{plan.name}; this writes this install's copy. To match the checkout, run "
            "`/cr-setup upgrade`, or reinstall (`uv tool install --reinstall ...` / "
            "`pipx install --force ...`)."
        )
        # Say which of the skill's own files differ, so a refresh that
        # changes only the script is not a bare "Would refresh <name>".
        script_lines = [
            f"      + {extra.name} ({'new' if extra.action == 'install' else 'differs'})"
            for extra in plan.extra_files if extra.action != "up_to_date"
        ]
        if dry_run:
            print(f"  - Would refresh {plan.name} -> {plan.dest}{note}")
            for line in script_lines:
                print(line)
            if differing:
                print(warning)
            refreshed.append(plan.name)
            continue
        print(f"  - {plan.name} -> {plan.dest}{note}")
        for line in script_lines:
            print(line)
        if differing:
            print(warning)
        if not (auto_yes or ask(plan)):
            print("      Skipped.")
            continue
        _apply_skill_plan(plan)
        print("      Refreshed.")
        refreshed.append(plan.name)
    if not_installed:
        print(
            f"  ({len(not_installed)} shipped skill(s) are not installed and were left alone; "
            "`init --install-skills` installs them.)"
        )
    if dry_run and stale:
        print("  Dry run: no skills written.")
    return refreshed


def _confirm_skill_rename(plan) -> bool:
    """Default interactive Y/n prompt for one skill rename (blank is Yes; a
    closed stdin is a decline, as in `_confirm_skill_refresh`)."""
    try:
        answer = input(f"      Rename {plan.old_name} to {plan.new_name}? [Y/n] ").strip().lower()
    except EOFError:
        print("      (no input available -- skipped; pass --auto-yes to apply without prompting)")
        return False
    return answer not in ("n", "no")


def migrate_renamed_skills(
    skills_dir: Path,
    *,
    auto_yes: bool,
    dry_run: bool,
    prompt_fn=None,
    tools_repo_dir: "Path | None" = None,
) -> "list[str]":
    """
    `upgrade`'s skill-rename migration: a product skill installed under an old
    `my-*` name (`RENAMED_PRODUCT_SKILLS`) is replaced by the shipped `cr-*`
    copy and the old directory is removed. Nothing here can rename an
    installed skill by refreshing it, since the installed frontmatter `name:`
    is kept; so the new directory is written (with `name:` set to the new
    name) and only then the old one removed, never the other way round.

    Approval is a Y/n per skill, every one with `auto_yes`, none written or
    prompted with `dry_run`. Only a directory the table names is ever removed,
    and its path is printed before it goes. Inert until the shipped `skills/`
    holds the `cr-*` directory (`plan_skill_renames` omits the skill), and safe
    to re-run: once migrated the old directory is gone. Returns the old names
    migrated (or, for `dry_run`, that would be).
    """
    plans = plan_skill_renames(Path(tools_repo_dir or TOOLS_REPO_DIR), skills_dir)
    if not plans:
        return []
    print(f"\nChecking for renamed skills under {skills_dir}...")
    print(f"  Found {len(plans)} installed skill(s) under an old name:")
    ask = prompt_fn or _confirm_skill_rename
    migrated: list = []
    for plan in plans:
        what = (
            f"remove {plan.old_dir} ({plan.new_name} is already installed"
            + (", and gain your old frontmatter" if plan.needs_write else "")
            + ")"
            if plan.new_exists
            else f"install {plan.new_name} -> {plan.new_dest}, then remove {plan.old_dir}"
        )
        if dry_run:
            print(f"  - Would {what}")
            migrated.append(plan.old_name)
            continue
        print(f"  - {plan.old_name}: {what}")
        if not (auto_yes or ask(plan)):
            print("      Skipped.")
            continue
        # A prompt can sit open while something else changes the directory, so
        # nothing planned earlier is trusted at this point: write when the
        # replacement is missing (or needs the old frontmatter), then check it
        # is a plain file again, and the old directory is still a plain
        # directory, before anything is removed.
        new_dir = plan.new_dest.parent
        if not plan.new_exists and (plan.new_dest.exists() or plan.new_dest.is_symlink()):
            # Absent when planned, present now: the plan's content was built
            # for an empty destination and would overwrite whatever arrived.
            print("      Skipped: the new copy appeared while waiting; re-run upgrade.")
            continue
        if new_dir.is_symlink() or plan.new_dest.is_symlink():
            print("      Not removed: the new copy's location is a symlink.")
            continue
        if plan.needs_write or not plan.new_dest.is_file():
            _atomic_write(plan.new_dest, plan.content)
        if not plan.new_dest.is_file() or plan.new_dest.is_symlink():
            print("      Not removed: the new copy is not a plain file.")
            continue
        if plan.old_dir.is_symlink() or not plan.old_dir.is_dir():
            print("      Not removed: it is no longer a plain directory.")
            continue
        if (plan.old_dir / "SKILL.md").is_symlink():
            print("      Not removed: the old SKILL.md is now a symlink.")
            continue
        shutil.rmtree(plan.old_dir)
        print("      Renamed.")
        migrated.append(plan.old_name)
    if dry_run:
        print("  Dry run: nothing written or removed.")
    return migrated


def warn_old_skill_names(skills_dir: Path, *, tools_repo_dir: "Path | None" = None) -> None:
    """
    `init --install-skills` installs the `cr-*` names as usual; this adds a
    warning for each old name from `RENAMED_PRODUCT_SKILLS` that is still
    installed next to them, pointing at `upgrade`, which migrates it. Silent
    until the shipped `skills/` carries the new name (same inertness as the
    migration, so the warning never points at an `upgrade` that offers
    nothing).
    """
    shipped = Path(tools_repo_dir or TOOLS_REPO_DIR) / "skills"
    stale = [
        old for old, new in sorted(RENAMED_PRODUCT_SKILLS.items())
        if (shipped / new / "SKILL.md").is_file() and (Path(skills_dir) / old / "SKILL.md").is_file()
    ]
    if stale:
        print(
            f"  Warning: still installed under an old name: {', '.join(stale)}. Run "
            "`upgrade` (with no project path, it updates just the installed skills) to move each to its "
            "new name; until then both copies are listed."
        )


def cmd_upgrade(args: argparse.Namespace) -> None:
    """
    Structured like `cmd_init`: validate `target_repo` exists, compute the
    same `venv_python_override = _pip_installed_venv_python()` value
    `cmd_init` already relies on, then delegate to `upgrade_lib.run_upgrade`
    -- which owns the actual migration-detection/prompting/writing and its
    own printed output (see that function's docstring for why). Unlike
    `cmd_init`'s `run_setup`, `run_upgrade` needs a concrete `venv_python`
    (not `None`-meaning-"use your own pure default"), so the `None` case
    from `_pip_installed_venv_python()` (the documented clone workflow) is
    resolved here via `venv_python_path` -- the exact same fallback
    `run_setup` performs internally for `init`.

    Also checks `upgrade_lib.is_project_configured` up front, exiting with a
    clear error/exit code 1 if `target_repo` has no existing toolkit
    `.mcp.json` at all -- `upgrade` is only for an ALREADY-configured
    project (a brand-new one needs `init` first). Found in Copilot review and confirmed
    live: without this check, `run_upgrade` treated
    a missing/empty `.mcp.json` the same as `{}`, so every migration's
    `detect` fired unconditionally and wrote a BROKEN partial config (a
    `qdrant` server block with no `command`/`type` at all) for a directory
    that was never `init`'d. `run_upgrade` itself carries the same guard
    (see its own docstring) for a caller that invokes it directly, bypassing
    this CLI -- this earlier check exists purely so `cmd_upgrade` gets a
    clean, distinguishable exit code instead of a silent empty result.
    """
    if args.target_repo is None:
        # Skills-only mode: no project path means there is no project
        # configuration to migrate, only the installed skills. The two skill
        # steps take just `skills_dir`, so they never needed a project; a user
        # who installed skills with `init --install-skills` and never
        # configured a project has no other way to pick up a rename. A path
        # that IS given but is wrong still fails below, so a typo never
        # silently becomes skills-only.
        skills_dir = _resolve_skills_dir(args)
        print(f"No project path given: updating installed skills under {skills_dir} only.")
        migrate_renamed_skills(skills_dir, auto_yes=args.auto_yes, dry_run=args.dry_run)
        refresh_installed_skills(skills_dir, auto_yes=args.auto_yes, dry_run=args.dry_run)
        print("No project configuration was touched.")
        return
    target_repo = Path(args.target_repo).resolve()
    if not target_repo.is_dir():
        print(f"error: target repo path does not exist or is not a directory: {target_repo}", file=sys.stderr)
        sys.exit(1)
    try:
        configured = is_project_configured(target_repo)
    except MalformedConfigError as exc:
        # An unusable .mcp.json is reported as such, not as "no
        # toolkit configuration" (which would send the user to `init`).
        print(f"error: {exc} -- fix or remove it, then re-run upgrade", file=sys.stderr)
        sys.exit(1)
    if not configured:
        print(
            f"error: {target_repo} has no existing toolkit configuration (.mcp.json with a 'qdrant' "
            "server) -- `upgrade` only applies migrations to an ALREADY-configured project. Run "
            "`init` first, then `upgrade` to pick up anything released since.",
            file=sys.stderr,
        )
        sys.exit(1)

    venv_python_override = _pip_installed_venv_python()
    resolved_venv_python = venv_python_override or venv_python_path(TOOLS_REPO_DIR, windows=(os.name == "nt"))

    try:
        run_upgrade(
            target_repo,
            TOOLS_REPO_DIR,
            resolved_venv_python,
            auto_yes=args.auto_yes,
            dry_run=args.dry_run,
            # Ask about the savings tracker only when a person is
            # there to answer; --auto-yes and --dry-run never ask (run_upgrade
            # also checks both), and a pipe/CI run has nobody to ask.
            offer_savings_fn=(
                ask_savings_tracker if not args.auto_yes and not args.dry_run and _stdin_is_interactive() else None
            ),
            # The codebase-reviewer server is deliberately NOT offered: it is not
            # live yet, so `upgrade` must not mention it. The
            # explicit `init --codebase-reviewer` flags still write the block.
            # To restore the offer once it ships, pass `offer_reviewer_fn`
            # here again (`ask_codebase_reviewer` when a person is at a terminal
            # and neither --auto-yes nor --dry-run is set, like the savings one).
        )
    except MalformedConfigError as exc:
        # .claude/settings.json is only read here (is_project_configured reads
        # .mcp.json alone); run_upgrade loads both before it prompts or writes.
        print(f"error: {exc} -- fix or remove it, then re-run upgrade", file=sys.stderr)
        sys.exit(1)

    # Config migrations never touched installed skills, so a user
    # who only ran `upgrade` kept stale skill text indefinitely. Runs after
    # the config migrations, whether or not any were pending.
    # Renames run first so the refresh then sees the new directories as
    # installed (and up to date) rather than leaving them for a later run.
    skills_dir = _resolve_skills_dir(args)
    migrate_renamed_skills(skills_dir, auto_yes=args.auto_yes, dry_run=args.dry_run)
    refresh_installed_skills(skills_dir, auto_yes=args.auto_yes, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=version_string("claude-runway-setup"))
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Write .mcp.json/.claude/settings.json into a target repo")
    # Optional (nargs="?") so `--install-skills` can run alone,
    # skills-only, with no project to configure at all. Still required for
    # every OTHER use of `init` -- enforced below, after parsing, rather
    # than left to argparse itself (argparse has no built-in way to say "X
    # is required unless Y is set").
    init.add_argument(
        "target_repo",
        nargs="?",
        default=None,
        help="Path to the project repo to configure; omit when passing --install-skills alone (skills-only mode)",
    )
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
        "Qdrant. Applied to the qdrant, codebase-indexer and memory-bank servers' QDRANT_API_KEY -- not 'sticky': pass it on every run, "
        "the same as --qdrant-url/--collection-name, or a rerun without it resets these servers back "
        "to unauthenticated. WARNING: writes the key in PLAINTEXT into .mcp.json -- do not commit that "
        "file as-is if this repo is shared; this script will print a reminder instead of the usual "
        "'commit .mcp.json' instruction when this is set",
    )
    init.add_argument(
        "--embedding-model",
        default="",
        help="EMBEDDING_MODEL written into the qdrant/codebase-indexer blocks (NOT memory-bank, which has "
        "--memory-bank-embedding-model). Blank (default) PRESERVES the model already in the target's "
        "existing .mcp.json, falling back to the template "
        "default for a new project. Passing a value that differs from the existing one prints a warning: "
        "vectors indexed under the old model are incompatible, so drop the collection and re-index",
    )
    init.add_argument(
        "--memory-bank-embedding-model",
        default="",
        help="EMBEDDING_MODEL written into the memory-bank block only, independent of --embedding-model. "
        "Blank (default) PRESERVES the memory-bank block's existing value, falling back to the template "
        "default. The memory-bank collection is shared by every project and locked to the model that "
        "created it, so passing a different value prints a warning: remember/recall fail for every "
        "project until that collection is rebuilt",
    )
    init.add_argument("--collection-description", default="", help="Short one-line hint for list_collections")
    init.add_argument("--include-extensions", default="", help="e.g. '.py,.md' -- blank uses the built-in defaults")
    init.add_argument("--exclude-dirs", default="", help="e.g. 'fixtures,generated' -- added to the built-in excludes")
    init.add_argument("--lmstudio-url", default="http://localhost:1234/v1")
    init.add_argument(
        "--lmstudio-model",
        default="",
        help="Exact model id loaded in LM Studio; leave blank to auto-detect (only works if exactly one model is loaded). "
        "Written to the local-compress block and to the qdrant block (compact_store compresses with it)",
    )
    init.add_argument(
        "--track-savings",
        action="store_true",
        help="Enable the opt-in savings tracker. Also writes the opt-in `analytics` server (savings_summary/"
        "savings_detail/savings_trend, get_metrics, record_metric); without this flag that block is not "
        "written, unless you answer yes when init asks on a terminal (passing the flag skips the question)",
    )
    init.add_argument(
        "--codebase-reviewer",
        action="store_true",
        help="Add the opt-in `codebase-reviewer` MCP server (the local review loop; `upgrade` does not offer it "
        "while the server is unreleased, so this flag is the way to add it). Its REVIEWERS_FILE setting is "
        "REQUIRED and has no default: init never writes a default path or a default reviewers file, so without "
        "--reviewers-file the block is written with REVIEWERS_FILE blank and init warns. An existing "
        "codebase-reviewer block is kept (with its REVIEWERS_FILE) on a re-run without this flag",
    )
    init.add_argument(
        "--reviewers-file",
        default="",
        help="Path to YOUR reviewers JSON file (the models, prompts and rubric), written as the codebase-reviewer "
        "block's REVIEWERS_FILE exactly as given. Implies --codebase-reviewer. The file is not created or checked "
        "here; `claude-runway-doctor` reports a missing or invalid one",
    )
    init.add_argument(
        "--savings-db",
        default="",
        help="Override the savings tracker DB path (leave blank for the default). Written into the `analytics` "
        "block with --track-savings; the hooks read it from the shell, so export it there too",
    )
    init.add_argument(
        "--metrics-db",
        default="",
        help="Override the shared metrics DB path (leave blank for the default). Written into the `memory-bank` "
        "block (it writes into that store) and, with --track-savings, the `analytics` block, so both use one file",
    )
    init.add_argument(
        "--memory-bank-collection",
        default="",
        help="Base Qdrant collection name the memory-bank server's remember/recall/forget tools share "
        "across EVERY project (mcpservers/memory_bank_mcp_server.py's MEMORY_BANK_COLLECTION); leave blank to "
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
        "while stranding existing ones under the old identifier.",
    )
    init.add_argument(
        "--memory-bank-follow",
        default=None,
        metavar="IDS",
        help="Comma-separated MEMORY_BANK_IDs of OTHER repos whose `shared` memories this project's default "
        "recall also returns (written as MEMORY_BANK_FOLLOW_REPOS in the memory-bank block). Not passing the "
        "flag PRESERVES the list already in the target's .mcp.json (sticky, like --memory-bank-embedding-model); "
        "passing it replaces the list, and passing an empty value (--memory-bank-follow '') clears it (the key stays, set to blank). Entries "
        "equal to this project's own id or to 'general' are ignored by the server (already in scope) and "
        "init warns about them. Following adds read access only: it does not let this project write under "
        "another repo's tag, and forget/wipe_all behave as before (wipe_all clears only this project's own "
        "memories; forgetting one memory by id that belongs to another repo still needs confirm=True). Takes effect when the memory-bank server next starts: restart Claude Code or "
        "reconnect it",
    )
    init.add_argument(
        "--compact-collection",
        default="",
        help="Base Qdrant collection name the cr-compact/cr-resume skills store conversation compacts under "
        "(qdrant_mcp_server.py's COMPACT_COLLECTION, in the qdrant block); leave blank to use the shared 'conversation-compacts' "
        "default. NOT 'sticky' to a per-project value on its own (matches every other option here): omitting "
        "this flag ALWAYS resolves to 'conversation-compacts' regardless of what a project's .mcp.json currently "
        "has, so a re-run is only a no-op for a project that has never customized this (via /cr-setup's "
        "own Q5, or a prior explicit --compact-collection) -- for a project that HAS, omitting this flag on a "
        "later raw-CLI re-run resets it back to 'conversation-compacts', orphaning that project's existing "
        "conversation-compact history from compact lookup. Repeat the same --compact-collection value on every "
        "re-run for such a project, or go through /cr-setup instead, which extracts and re-passes "
        "the current value automatically. qdrant_mcp_server.py still appends its own "
        "-<sanitized-project> suffix on top of whichever base this resolves to, at runtime "
        "(this base is a shared prefix; per-project isolation comes from payload filtering, not name uniqueness) "
        "-- so the final on-disk collection name is "
        "'<compact-collection>-<sanitized-project>'.",
    )
    # Mutually exclusive: their settings.json semantics genuinely conflict.
    # --qdrant-only actively REMOVES this toolkit's local-compress-dependent
    # hooks if present (the CORE record_session_id.py hook is written/kept
    # either way -- it isn't removed);
    # --skip-hooks promises to leave settings.json untouched either way.
    # Found in review: passing both together silently let
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
        "a prior run had added them. The CORE record_session_id.py hook is still written "
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
    init.add_argument(
        "--skip-claude-md",
        action="store_true",
        help="Leave the target's CLAUDE.md untouched. By default init creates it, or adds to it the "
        "usage section from templates/CLAUDE.md.template for each server it writes and removes the "
        "section of a toolkit server that is no longer configured (e.g. Local compression under "
        "--qdrant-only); nothing else in the file is changed, and a re-run that has nothing to add or "
        "remove does not rewrite it",
    )
    init.add_argument(
        "--install-skills",
        action="store_true",
        help="Install/update this toolkit's product skills (skills/*/SKILL.md) into --skills-dir (default "
        "~/.claude/skills) -- missing skills are installed, changed ones are updated, identical ones are left "
        "alone. Combine with target_repo to do both in one run, or pass alone (omitting target_repo) for "
        "skills-only mode that touches no project config at all.",
    )
    init.add_argument(
        "--skills-dir",
        default="",
        help="Override the skills install destination (default: ~/.claude/skills); e.g. '<repo>/.claude/skills' "
        "for a per-project install instead of the usual global one",
    )
    init.add_argument("--dry-run", action="store_true", help="Print what would be written without writing")
    init.set_defaults(func=cmd_init)

    # a config-migration tool for an ALREADY-configured project --
    # unlike `init`, which resets every toolkit-owned setting to this run's
    # flags/defaults, `upgrade` compares the target's current .mcp.json/
    # .claude/settings.json against a fixed set of named, individually
    # detectable gaps (libs/upgrade_lib.py's MIGRATIONS) and applies only the
    # ones the user approves. No feature-selection flags (--qdrant-only,
    # --lmstudio-model, etc.) -- migrations are self-contained and derive
    # everything they need from the existing config or the template.
    # After the migrations it also refreshes already-installed
    # skills (see refresh_installed_skills), so the flags below cover both.
    upgrade = sub.add_parser(
        "upgrade",
        help="Apply pending config migrations to an already-configured project and refresh installed skills",
    )
    upgrade.add_argument(
        "target_repo",
        nargs="?",
        default=None,
        help="Path to the already-configured project repo to migrate. Omit it to update only the installed "
        "skills (--skills-dir, default ~/.claude/skills) and touch no project configuration",
    )
    upgrade.add_argument(
        "--dry-run", action="store_true", help="Print pending migrations and stale skills without writing anything"
    )
    upgrade.add_argument(
        "--auto-yes",
        action="store_true",
        help="Apply every pending migration and refresh every stale installed skill without prompting (for "
        "scripting/CI); also skips the questions about enabling the savings tracker and adding the "
        "codebase-reviewer server, leaving both off",
    )
    upgrade.add_argument(
        "--skills-dir",
        default="",
        help="Where installed product skills live (default: ~/.claude/skills) -- upgrade also refreshes "
        "any skill already installed there that differs from the shipped version; skills "
        "that are not installed are never installed by upgrade",
    )
    upgrade.set_defaults(func=cmd_upgrade)

    parsed = p.parse_args()
    # target_repo is nargs="?" specifically to allow skills-only mode
    # (--install-skills with no project to configure) -- but for every
    # OTHER invocation of `init`, it's still required. argparse itself has
    # no way to express "positional required unless this flag is set", so
    # enforce it here instead, using the same subparser (init.error) so the
    # error message/exit code (2) matches argparse's own usage-error
    # convention rather than a bespoke sys.exit elsewhere.
    if parsed.command == "init" and parsed.target_repo is None and not parsed.install_skills:
        init.error("the following arguments are required: target_repo (unless --install-skills is given alone)")
    return parsed


def main() -> None:
    """
    Entry point for both `python tools/setup_project.py` (the
    `if __name__` guard below) and the `claude-runway-setup` console
    script pip/pipx/uvx installs -- pulled out into its own
    function so the `claude_runway:setup_main` shim, which is what
    `[project.scripts]` in pyproject.toml points at, has
    something to call; behavior is unchanged either way.
    """
    parsed_args = parse_args()
    parsed_args.func(parsed_args)


if __name__ == "__main__":
    main()
