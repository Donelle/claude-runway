#!/usr/bin/env bash
# Developer bootstrap for macOS and Linux. POSIX counterpart of
# scripts/bootstrap.ps1: creates .venv, installs runtime + dev requirements
# and this checkout's console scripts, opts the clone into .githooks, writes
# the shell-side CLAUDE_RUNWAY_* environment file, then runs
# claude-runway-doctor. Safe to re-run -- every step checks first and changes
# nothing that is already correct.
#
#   scripts/bootstrap.sh                   full bootstrap
#   scripts/bootstrap.sh --env-only        only refresh ~/.claude/claude-runway/env.sh
#                                          (needs the .venv a full run created)
#   scripts/bootstrap.sh --update-profile  add the line that loads env.sh to your
#                                          shell profile without asking first
#
# Written for the bash 3.2 that macOS ships: no associative arrays, mapfile,
# ${var,,}, `sed -i`, `readlink -f`, `date -d`, realpath or mktemp, and no
# heredoc inside $( ). The env step's logic is the Python below, so JSON is
# never parsed in bash and nothing here needs a newer bash.
set -euo pipefail

die() {
    echo "bootstrap: $1" >&2
    exit 1
}

step() {
    echo "> $1"
}

ENV_ONLY=0
PROFILE_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --env-only) ENV_ONLY=1 ;;
        --update-profile) PROFILE_FLAG="--update-profile" ;;
        -h|--help)
            echo "usage: scripts/bootstrap.sh [--env-only] [--update-profile]"
            exit 0
            ;;
        *) die "unknown argument '$arg' (usage: scripts/bootstrap.sh [--env-only] [--update-profile])" ;;
    esac
done

command -v git >/dev/null 2>&1 || die "git was not found. Install git and rerun this script."

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
git rev-parse --show-toplevel >/dev/null 2>&1 || die "run this script from a git clone of ClaudeRunway."

VENV_DIR="$REPO_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
DOCTOR="$VENV_DIR/bin/claude-runway-doctor"

# The env step's logic. Kept as a function so the script can be handed to
# `python -c "$(env_step_py)"`: the heredoc lives in the function body, not
# inside the $( ), which bash 3.2 mis-parses, and python's stdin stays free
# for the profile question.
env_step_py() {
    cat <<'PY'
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ENV_FILE_NAME = "env.sh"
LOCAL_FILE_NAME = "env.local.sh"
BLOCK_BEGIN = "# >>> claude-runway env (added by scripts/bootstrap.sh) >>>"
BLOCK_END = "# <<< claude-runway env <<<"
PROFILE_BLOCK = (
    BLOCK_BEGIN + "\n"
    '[ -f "$HOME/.claude/claude-runway/env.sh" ] && . "$HOME/.claude/claude-runway/env.sh"\n'
    + BLOCK_END + "\n"
)
# What marks a profile as already wired, whoever wrote the line.
WIRED_MARKER = "claude-runway/env.sh"
PLACEHOLDER_PREFIX = "REPLACE-WITH-"


def _doctor():
    import doctor_lib

    return doctor_lib


def env_file_path(home):
    return Path(home) / ".claude" / "claude-runway" / ENV_FILE_NAME


def build_table(repo_root, home):
    """
    One row per variable: (name, source, default, comment lines).
    source "mcp": the value is taken from .mcp.json when it declares one
    (these are the CLAUDE_RUNWAY_* vars doctor_lib compares), else the
    default; "computed": a fact about this checkout; "switch": a shell-only
    setting. Every default is written so that an existing export wins.
    """
    dl = _doctor()
    return [
        (
            "CLAUDE_RUNWAY_DIR", "computed", str(repo_root),
            ["Absolute path of the claude-runway checkout the tools run from."],
        ),
        (
            dl.DUAL_ENV_VARS[0], "mcp", dl.DEFAULT_LMSTUDIO_URL,
            ["LM Studio server the compression tools and hooks call.",
             "Must match the CLAUDE_RUNWAY_LMSTUDIO_URL in .mcp.json."],
        ),
        (
            dl.DUAL_ENV_VARS[1], "mcp", "",
            ["Model to compress with; blank means use whatever model LM Studio has loaded.",
             "Must match the CLAUDE_RUNWAY_LMSTUDIO_MODEL in .mcp.json."],
        ),
        (
            dl.DUAL_ENV_VARS[2], "mcp", "0",
            ["Savings tracker: 1, true or yes records token savings; anything else is off.",
             "Must match the CLAUDE_RUNWAY_TRACK_SAVINGS in .mcp.json."],
        ),
        (
            dl.DUAL_ENV_VARS[3], "mcp", str(Path(home) / ".claude" / "claude-runway" / "savings.db"),
            ["Savings tracker database. The hooks write it and the analytics server reads it,",
             "so both must name the same file (CLAUDE_RUNWAY_SAVINGS_DB in .mcp.json)."],
        ),
        (
            "CLAUDE_RUNWAY_COMPRESS_THRESHOLD_CHARS", "switch", "2000",
            ["Tool output longer than this many characters is compressed. Shell only."],
        ),
        (
            "CLAUDE_RUNWAY_PARSE_TRANSCRIPT_TOKENS", "switch", "0",
            ["1 lets the savings hooks read real token counts from the transcript. Shell only."],
        ),
        (
            "CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS", "switch", "1",
            ["1 records memory-bank events; 0 turns that off."],
        ),
    ]


def read_declared(mcp_path, notes):
    """
    The values .mcp.json declares for the dual-config vars: the first server
    block, in SERVER_DUAL_ENV_VARS order, that declares a var wins. A literal
    REPLACE-WITH-* value is an unfilled template, so it counts as undeclared.
    A missing or malformed file declares nothing (the doctor run that follows
    reports a malformed one properly).
    """
    dl = _doctor()
    if not mcp_path.is_file():
        notes.append(".mcp.json not found next to the checkout; every value below is a default.")
        return {}
    declared = {}
    try:
        for server, names in dl.SERVER_DUAL_ENV_VARS.items():
            env = dl.load_server_env(mcp_path, server)
            if not env:
                continue
            for name in names:
                if name in declared or name not in env:
                    continue
                value = env[name]
                if value.startswith(PLACEHOLDER_PREFIX):
                    notes.append(
                        f"{name} in .mcp.json's {server} block is still the unfilled placeholder "
                        f"'{value}'; ignoring it."
                    )
                elif "\n" in value or "\r" in value:
                    notes.append(f"{name} in .mcp.json's {server} block spans several lines; ignoring it.")
                else:
                    declared[name] = value
    except dl.MalformedMcpJsonError as exc:
        notes.append(f"{exc}. Falling back to defaults; fix the file and rerun with --env-only.")
        return {}
    return declared


def _quote(value):
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    return '"' + escaped + '"'


# A `${NAME}` or the start of `${NAME:-default}`: the variable references
# Claude Code expands in .mcp.json, left for the shell to expand the same way.
_MCP_REFERENCE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*(?:\}|:-)")


def _quote_mcp(value):
    """
    Like _quote, but a `${NAME}`/`${NAME:-...}` reference stays live so the
    shell resolves it as Claude Code does for the MCP server (otherwise
    `${HOME}/x.db` would reach the hooks literally and the two halves would
    use different databases). Every other `$`, and every backtick, is still
    escaped: .mcp.json is committed with the repo and env.sh runs in every
    new terminal, so a `$(...)` in it, even nested in a default, must not run.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    out = []
    for i, c in enumerate(escaped):
        if c == "$" and not _MCP_REFERENCE.match(escaped, i + 1):
            out.append("\\$")
        else:
            out.append(c)
    return '"' + "".join(out) + '"'


def default_line(name, value):
    """
    Sets NAME to `value` only when it is unset, so a value the user already
    exported (even a blank one, the same rule os.environ.get applies) wins
    and env.sh never has to know what they exported. The default sits in an
    ordinary double-quoted assignment, not inside `${NAME-...}`: there a `'`
    (a home like /Users/O'Connor) breaks bash's parse and a `}` ends the
    expansion early.
    """
    return f'if [ -z "${{{name}+set}}" ]; then {name}={_quote(value)}; fi; export {name}'


def render_env_file(repo_root, home, environ, declared, notes):
    """
    The whole env.sh text. The script owns this file and rewrites it every
    run; it never reads it back, so nothing the user wrote can be reordered
    or truncated. A value .mcp.json declares, and CLAUDE_RUNWAY_DIR, are
    written as-is. Everything else is a default that an existing export
    overrides. User changes belong in env.local.sh, loaded at the end.
    """
    dl = _doctor()
    lines = [
        "# ClaudeRunway shell environment. Written by scripts/bootstrap.sh, which rewrites the",
        "# whole file on every run: do not edit it. Rerun `scripts/bootstrap.sh --env-only` after",
        "# changing .mcp.json. Put your own overrides and exports in",
        f"# ~/.claude/claude-runway/{LOCAL_FILE_NAME}, which is loaded at the end and never touched.",
    ]
    for name, source, default, comment in build_table(repo_root, home):
        if source == "computed":
            assignment = _quote(str(repo_root))
        elif name in declared:
            assignment = _quote_mcp(declared[name])
            if name in environ:
                # normalize() applies each var's own default rule, so `~/x.db`
                # and its absolute form, or a blank and an absent model, agree.
                want = dl._normalize(name, declared[name], home_dir=Path(home))
                have = dl._normalize(name, environ[name], home_dir=Path(home))
                if want != have:
                    notes.append(
                        f"{name} is exported in this shell as '{environ[name]}', but .mcp.json says "
                        f"'{declared[name]}'. env.sh sets .mcp.json's value; remove your export if it "
                        "is loaded after env.sh, or the two will disagree."
                    )
        else:
            lines.append("")
            lines.extend("# " + c for c in comment)
            lines.append(default_line(name, default))
            continue
        lines.append("")
        lines.extend("# " + c for c in comment)
        lines.append(f"export {name}={assignment}")
    lines.append("")
    lines.append("# Your own settings, if you have any. An if, not `&&`, so a missing file is not an error")
    lines.append("# for a shell running with `set -e`.")
    local = f'"$HOME/.claude/claude-runway/{LOCAL_FILE_NAME}"'
    lines.append(f"if [ -f {local} ]; then . {local}; fi")
    return "\n".join(lines) + "\n"


def _shells_to_check():
    """bash (the system one first: macOS's 3.2 is the strictest) and zsh, whichever exist."""
    found = []
    for candidate in ("/bin/bash", shutil.which("bash"), shutil.which("zsh")):
        if candidate and os.path.exists(candidate) and candidate not in found:
            found.append(candidate)
    return found


def _parses(text, shells):
    """The shells that reject `text` (`-n`: parse only, nothing runs), with their message."""
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        candidate = Path(tmp) / "candidate.sh"
        candidate.write_text(text, encoding="utf-8")
        for shell in shells:
            result = subprocess.run([shell, "-n", str(candidate)], capture_output=True, text=True)
            if result.returncode != 0:
                failures.append((shell, (result.stderr.strip().splitlines() or ["(no message)"])[-1]))
    return failures


def check_parses(content, shells=None):
    """
    None when every available shell parses `content`; otherwise a message
    naming the variable whose line does not. A last guard, whatever the
    quoting rules above miss: a file that does not parse would break every
    new terminal that loads it, so it is never installed.
    """
    shells = _shells_to_check() if shells is None else shells
    failures = _parses(content, shells)
    if not failures:
        return None
    culprit = "the generated file"
    for line in content.splitlines():
        if line.startswith(("export ", "if [")) and _parses(line + "\n", shells):
            match = re.search(r"\b(CLAUDE_RUNWAY_[A-Z_]+)=", line)
            culprit = match.group(1) if match else line
            break
    shell, message = failures[0]
    return f"{culprit} would not parse in {os.path.basename(shell)} ({message})"


def write_if_changed(path, content):
    """
    True when the file was written. Temp file + os.replace, so a reader never
    sees half a file. The temp file is created with the final mode before any
    content goes in: the file keeps its existing mode (a user who tightened it
    to 0600 keeps that, since kept export lines may hold secrets), and a new
    one is 0600 rather than whatever the umask would allow.
    """
    path = Path(path)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.is_file() else 0o600
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        if os.path.lexists(tmp):
            os.unlink(tmp)
        raise
    return True


def profile_target(shell, home):
    """
    The startup file to wire on macOS, or None for a shell this does not
    know. zsh: ~/.zshrc (every interactive shell, login or not). bash: the
    file a bash login shell reads, which Terminal opens, and which is NOT
    ~/.bashrc: the first of ~/.bash_profile, ~/.bash_login, ~/.profile that
    exists, else a new ~/.bash_profile. Picking the existing one means
    creating ~/.bash_profile never hides a ~/.profile bash was reading.
    """
    home = Path(home)
    name = os.path.basename(shell or "")
    if name == "zsh":
        return home / ".zshrc"
    if name == "bash":
        for candidate in (".bash_profile", ".bash_login", ".profile"):
            if os.path.lexists(home / candidate):
                return home / candidate
        return home / ".bash_profile"
    return None


def append_profile_block(target):
    """Append PROFILE_BLOCK to target (through a symlink), creating it if absent. Returns the real path."""
    real = Path(os.path.realpath(target))
    # Bytes, not text: a profile is not necessarily UTF-8 (a Latin-1 comment).
    existing = real.read_bytes() if real.is_file() else b""
    sep = b""
    if existing:
        sep = (b"" if existing.endswith(b"\n") else b"\n") + b"\n"
    with open(real, "ab") as handle:
        handle.write(sep + PROFILE_BLOCK.encode("utf-8"))
    return real


def _display(path, home):
    path, home = str(path), str(home)
    return "~" + path[len(home):] if path.startswith(home + os.sep) else path


def wire_profile(home, shell, system, update_profile, ask, out=print):
    """
    Offer to add the line that loads env.sh to the user's profile, and write
    it only on a yes (or --update-profile). `ask` is None when there is no
    terminal to ask on; then nothing is written and the block is printed.
    Returns what happened: wired, already, declined, printed or failed.
    """
    home = Path(home)
    if system != "Darwin":
        out("Add this to your shell's startup file so new terminals load the environment")
        out("(profile selection for Linux is not automated yet, so this was not written for you):")
        out("")
        out(PROFILE_BLOCK.rstrip("\n"))
        return "printed"
    target = profile_target(shell, home)
    if target is None:
        out(f"Your shell ({shell or 'unknown'}) is not zsh or bash, so no profile was chosen. Add this to its startup file:")
        out("")
        out(PROFILE_BLOCK.rstrip("\n"))
        return "printed"
    shown = _display(target, home)
    try:
        current = Path(os.path.realpath(target)).read_bytes() if os.path.lexists(target) else b""
    except OSError as exc:
        out(f"Could not read {shown} ({exc}); not changing it. Add this by hand:")
        out("")
        out(PROFILE_BLOCK.rstrip("\n"))
        return "failed"
    if WIRED_MARKER.encode("utf-8") in current:
        out(f"{shown} already loads the claude-runway environment; leaving it alone.")
        return "already"
    if not update_profile:
        if ask is None:
            out(f"No terminal to ask on, so {shown} was not changed. Add this to it to load the environment in new terminals:")
            out("")
            out(PROFILE_BLOCK.rstrip("\n"))
            return "printed"
        answer = ask(f"Add the claude-runway environment to {shown}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            out(f"Left {shown} unchanged. To load the environment in new terminals, add this to it:")
            out("")
            out(PROFILE_BLOCK.rstrip("\n"))
            return "declined"
    try:
        real = append_profile_block(target)
    except OSError as exc:
        out(f"Could not write {shown} ({exc}). Add this by hand:")
        out("")
        out(PROFILE_BLOCK.rstrip("\n"))
        return "failed"
    through = f" (a symlink; the change went to {real})" if os.path.islink(target) else ""
    out(f"Added the claude-runway environment to {shown}{through}.")
    out(f"Open a new terminal, or run `. {shown}`, for it to take effect.")
    return "wired"


def ask_on_terminal(prompt):
    try:
        return input(prompt)
    except EOFError:
        return ""


def gui_note(out=print):
    out("Note: Claude Code started from the Dock or a desktop app does not read your shell profile,")
    out("so it will not see these variables; start it from a terminal. (`launchctl setenv NAME value`")
    out("sets one for GUI apps; this script does not do that.) claude-runway-doctor compares .mcp.json")
    out("with the shell it runs in, so it reports OK from a terminal even when a GUI launch lacks them.")


def main(argv):
    update_profile = "--update-profile" in argv
    positional = [a for a in argv if not a.startswith("--")]
    repo_root = Path(positional[0]).resolve()
    sys.path.insert(0, str(repo_root / "libs"))
    home = Path.home()
    notes = []
    declared = read_declared(repo_root / ".mcp.json", notes)
    path = env_file_path(home)
    content = render_env_file(repo_root, home, os.environ, declared, notes)
    shown = _display(path, home)
    problem = check_parses(content)
    if problem:
        for note in notes:
            print(f"  note: {note}")
        print(
            f"bootstrap: {shown} was left as it was: {problem}. Check that value in .mcp.json, "
            "then rerun with --env-only.",
            file=sys.stderr,
        )
        return 1
    print(f"{'Wrote' if write_if_changed(path, content) else 'Already up to date:'} {shown}")
    for note in notes:
        print(f"  note: {note}")
    system = platform.system()
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    wire_profile(home, os.environ.get("SHELL", ""), system, update_profile, ask_on_terminal if interactive else None)
    if system == "Darwin":
        gui_note()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
PY
}

env_step() {
    step "Write the ClaudeRunway shell environment file"
    # PROFILE_FLAG is empty or one flag-shaped word, so it is left unquoted.
    "$VENV_PYTHON" -c "$(env_step_py)" "$REPO_ROOT" $PROFILE_FLAG
}

# The doctor compares .mcp.json with the shell it runs in, and this process
# has not loaded env.sh (a new terminal will, once the profile line is in
# place). Load it for the check, in a subshell, so the doctor checks what a
# new terminal will see instead of failing on a file that was just written.
run_doctor() {
    step "Run the ClaudeRunway doctor check (with the new environment file loaded)"
    local env_file="$HOME/.claude/claude-runway/env.sh"
    if [ -f "$env_file" ]; then
        ( . "$env_file"; "$DOCTOR" "$REPO_ROOT" )
    else
        "$DOCTOR" "$REPO_ROOT"
    fi
}

if [ "$ENV_ONLY" -eq 1 ]; then
    # Refresh env.sh after editing .mcp.json without reinstalling anything.
    [ -x "$VENV_PYTHON" ] || die "no .venv at '$VENV_DIR'. Run scripts/bootstrap.sh without --env-only first."
    [ -x "$DOCTOR" ] || die "the doctor command is not installed at '$DOCTOR'. Run scripts/bootstrap.sh without --env-only first."
    env_step
    run_doctor
    echo "bootstrap: environment file updated."
    exit 0
fi

# Prefer uv's interpreter lookup (it also finds uv-managed Pythons), then a
# plain python3.12 on PATH.
HAVE_UV=0
command -v uv >/dev/null 2>&1 && HAVE_UV=1

PYTHON312=""
if [ "$HAVE_UV" -eq 1 ]; then
    PYTHON312="$(uv python find 3.12 2>/dev/null | head -n 1 || true)"
fi
if [ -z "$PYTHON312" ] && command -v python3.12 >/dev/null 2>&1; then
    PYTHON312="$(command -v python3.12)"
fi
[ -n "$PYTHON312" ] && [ -x "$PYTHON312" ] || die "Python 3.12 was not found. Install it and rerun (uv users: 'uv python install 3.12'; macOS: 'brew install python@3.12'; Debian/Ubuntu: 'apt install python3.12 python3.12-venv')."

if [ ! -x "$VENV_PYTHON" ]; then
    # An existing .venv with no bin/python is some other layout (e.g. a
    # Windows Scripts/ venv from a shared checkout) -- never delete it for
    # the user, same stance as bootstrap.ps1.
    [ ! -e "$VENV_DIR" ] || die "the existing .venv has no bin/python. Remove or rename '$VENV_DIR' manually, then rerun this script."
    step "Create the Python 3.12 virtual environment"
    if [ "$HAVE_UV" -eq 1 ]; then
        uv venv --python "$PYTHON312" "$VENV_DIR"
    else
        "$PYTHON312" -m venv "$VENV_DIR"
    fi
fi
[ -x "$VENV_PYTHON" ] || die "virtual environment creation did not produce '$VENV_PYTHON'."

VENV_VERSION="$("$VENV_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" || die "could not run the Python executable in .venv."
[ "$VENV_VERSION" = "3.12" ] || die "the existing .venv uses Python $VENV_VERSION instead of 3.12. Remove or rename '$VENV_DIR' manually, then rerun this script."

# uv venv doesn't seed pip, so uv-created venvs must be installed into via
# `uv pip --python`; a venv made by `python -m venv` (no uv) has pip.
pip_install() {
    if [ "$HAVE_UV" -eq 1 ]; then
        uv pip install --python "$VENV_PYTHON" "$@"
    else
        "$VENV_PYTHON" -m pip install "$@"
    fi
}

# A .venv created by a previous run WITH uv has no pip, so re-running on a
# machine/shell where uv is no longer on PATH would fail at the first
# `python -m pip`. Seed pip into it once instead.
if [ "$HAVE_UV" -eq 0 ] && ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    step "Seed pip into the existing .venv (it was created by uv, which is not on PATH now)"
    "$VENV_PYTHON" -m ensurepip --upgrade || die "the existing .venv has no pip and ensurepip failed. Install uv, or remove '$VENV_DIR' and rerun this script."
fi

step "Install runtime requirements"
pip_install -r "$REPO_ROOT/requirements.txt" --index-url https://pypi.org/simple
step "Install developer requirements"
pip_install -r "$REPO_ROOT/requirements-dev.txt" --index-url https://pypi.org/simple
# Installs the console scripts (claude-runway-doctor etc.) from this
# checkout; dependencies were installed above, so skip resolving them again.
step "Install/update ClaudeRunway command-line tools from this checkout"
pip_install --no-deps --no-build-isolation "$REPO_ROOT"

[ -x "$DOCTOR" ] || die "the doctor command was not installed at '$DOCTOR'."

# Git won't let a repo set its own core.hooksPath, so each clone opts in once.
CURRENT_HOOKS_PATH="$(git config --get core.hooksPath 2>/dev/null || true)"
if [ "$CURRENT_HOOKS_PATH" != ".githooks" ]; then
    step "Enable the repository pre-push hook"
    git config core.hooksPath .githooks
else
    step "Repository pre-push hook already enabled"
fi
# A hook without its executable bit is silently ignored by git.
[ -x "$REPO_ROOT/.githooks/pre-push" ] || chmod +x "$REPO_ROOT/.githooks/pre-push"

# After the install and before the doctor, so the doctor checks the result.
env_step

run_doctor

echo "bootstrap: setup complete."
