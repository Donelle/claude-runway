"""
Shared, general-purpose local SQLite store for ClaudeRunway caches that are
safe to lose. Unlike libs/savings_ledger.py's savings.db (data the user
wants kept forever -- session history, cross-project totals), everything
that ends up in this file is disposable: always re-derivable from a live
source (e.g. re-fetched from Qdrant), so deleting this file is always safe
-- just occasionally slower afterward until caches warm back up again.

Kept as a SEPARATE file from savings.db specifically so "delete this to
reset a stuck cache" never risks the savings history sitting right next to
it in the same file. That's a deliberate, explicit split (2026-08-20): one
file the user wants to keep, one file that's fine to nuke -- don't merge
them back together even though both are small SQLite files in the same
directory.

Callers own their own tables (their own CREATE TABLE IF NOT EXISTS against
the connection this module hands back) -- this module itself defines no
tables, just where the shared file lives and how to open it. See
libs/qdrant_collection_hints.py for the first table living here; future
caches (this toolkit will likely grow more) should reuse connect() below
rather than resolving their own separate DB file.

DB location resolution: CLAUDE_RUNWAY_CACHE_DB env var (absolute path) if
set, else ~/.claude/claude-runway/cache.db. Same convention as
savings_ledger.resolve_db_path() (including the relative-path-anchored-to-
home safety net for a relative override), but a DIFFERENT env var pointing
at a DIFFERENT file -- do not conflate this with CLAUDE_RUNWAY_SAVINGS_DB.
Read by the codebase-indexer MCP server (list_collections/set_collection_description
hint cache) and by the redirect_webfetch_to_fetch_url.py hook (denied-URL TTL
cache, issue #64). Since a hook can only inherit its environment from the shell
(hook entries in .claude/settings.json have no `env` field), overriding this
path requires a shell export -- not just a .mcp.json entry. The default needs
no configuration at all: both callers resolve it the same way.
"""

import os
import sqlite3
from pathlib import Path


def resolve_cache_db_path() -> Path:
    override = os.environ.get("CLAUDE_RUNWAY_CACHE_DB")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            # Same reasoning as savings_ledger.resolve_db_path(): anchor a
            # relative override to the home directory rather than leaving it
            # ambiguous against whichever process's cwd happened to be
            # current when this ran.
            path = Path.home() / path
        return path
    return Path.home() / ".claude" / "claude-runway" / "cache.db"


def connect() -> sqlite3.Connection:
    """
    Opens (creating the parent directory if needed) a connection to the
    shared cache DB. Callers are responsible for their own CREATE TABLE IF
    NOT EXISTS against it and for closing the returned connection -- this
    function does no table setup of its own, since it doesn't know what
    any particular caller needs.
    """
    db_path = resolve_cache_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))
