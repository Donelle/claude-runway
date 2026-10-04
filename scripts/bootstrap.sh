#!/usr/bin/env bash
# Developer bootstrap for macOS and Linux (issue #31). POSIX counterpart of
# scripts/bootstrap.ps1: creates .venv, installs runtime + dev requirements
# and this checkout's console scripts, opts the clone into .githooks, then
# runs claude-runway-doctor. Safe to re-run -- every step checks first and
# changes nothing that is already correct.
set -euo pipefail

die() {
    echo "bootstrap: $1" >&2
    exit 1
}

step() {
    echo "> $1"
}

command -v git >/dev/null 2>&1 || die "git was not found. Install git and rerun this script."

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
git rev-parse --show-toplevel >/dev/null 2>&1 || die "run this script from a git clone of ClaudeRunway."

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

VENV_DIR="$REPO_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

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
# `python -m pip` (PR #41 review). Seed pip into it once instead.
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

DOCTOR="$VENV_DIR/bin/claude-runway-doctor"
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

step "Run the ClaudeRunway doctor check"
"$DOCTOR" "$REPO_ROOT"

echo "bootstrap: setup complete."
