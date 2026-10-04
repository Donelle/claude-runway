"""
Resolve "which claude-runway is this?" for the three CLIs' `--version` flag
(issue #348). stdlib-only, like the other libs/ modules the console scripts
load, so `--version` works even in a half-broken environment -- exactly when
someone filing a bug needs it most.

Two sources, in order:
1. `git describe` against the checkout this file lives in, when it IS a
   checkout (the clone workflow, `python tools/...`). Checked first so a
   checkout never reports some other installed copy's version.
2. importlib.metadata -- the version of the INSTALLED distribution
   (pipx/`uv tool install`/`pip install`). setuptools-scm derived it from the
   repo history at build time, so it identifies the exact commit installed.
"""

import subprocess
from importlib import metadata
from pathlib import Path
from typing import Optional

DIST_NAME = "claude-runway"


def _describe_checkout(repo_dir: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "describe", "--tags", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = result.stdout.strip()
    return out if result.returncode == 0 and out else None


def _checkout_root() -> Optional[Path]:
    """The checkout this file lives in, or None when it is an installed copy.

    An installed wheel ships this file to <env>/src/libs/ (see setup.py), whose
    parent has no `.git`; a clone has `.git` (a directory, or a file in a
    worktree) right above libs/. Copilot review (PR #372): a checkout run with
    an interpreter that ALSO has claude-runway installed must report the
    checkout's version, not the unrelated installed distribution's.
    """
    root = Path(__file__).resolve().parent.parent
    return root if (root / ".git").exists() else None


def get_version() -> str:
    checkout = _checkout_root()
    if checkout is not None:
        described = _describe_checkout(checkout)
        if described:
            return f"{described} (source checkout)"
    try:
        return metadata.version(DIST_NAME)
    except metadata.PackageNotFoundError:
        return "unknown (not installed, not a source checkout)"


def version_string(prog: str) -> str:
    return f"{prog} {get_version()}"
