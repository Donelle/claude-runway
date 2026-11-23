"""Defaults more than one part of the toolkit needs, written once so they are not retyped in every file."""

import os
import tempfile
import time
from pathlib import Path
from typing import Callable

# LM Studio's OpenAI-compatible endpoint: CLAUDE_RUNWAY_LMSTUDIO_URL when set, else LM Studio's own default address.
LMSTUDIO_URL = os.environ.get("CLAUDE_RUNWAY_LMSTUDIO_URL", "http://localhost:1234/v1")

# Where the codebase-reviewer records its runs. libs/reviewer_db_lib.py reads it and owns the real default
# (~/.claude/claude-runway/codebase-reviewer.db); the name is here so the harness does not retype it.
REVIEWER_DB_ENV = "CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB"


def isolate_reviewer_db(announce: Callable[[str], object] = print) -> Path:
    """Point this process, and every child it starts, at a throwaway reviewer database unless the caller chose one.

    The harness records its runs on purpose (that is how models are compared), but a test run must not write into the
    reviewer's real database under `~/.claude` and mix test rows with real review history. So when
    `CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB` is unset (or empty, as `reviewer_db_lib.resolve_db_path` treats it) it is set to a
    new folder under the system temp folder; a caller who sets it, to the real database or any other path, keeps it.

    This is a function the harness's entry points call, not an assignment made when the module is imported: other code
    imports this module (the reviewer's own server may), and an import must never move that code off the real database.
    It writes `os.environ` itself, not a copy, so a script started afterwards inherits it. The folder is not created here
    (the database and its log folder create their parents) and never deleted, since its path, announced here, is how the
    logs are found after the run. Returns the database path."""
    chosen = os.environ.get(REVIEWER_DB_ENV)
    if chosen:
        path = Path(chosen).expanduser()
        announce(f"reviewer database: {path} (from {REVIEWER_DB_ENV})")
        return path
    path = Path(tempfile.gettempdir()) / f"claude-runway-harness-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}" / "codebase-reviewer.db"
    os.environ[REVIEWER_DB_ENV] = str(path)
    announce(f"reviewer database: {path} (throwaway; set {REVIEWER_DB_ENV} to record into one you keep). Logs are beside it.")
    return path
