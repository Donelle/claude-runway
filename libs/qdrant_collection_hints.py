"""
Persistent, cross-session cache for per-collection description hints (issue
#81). tools/ingest_mcp_server.py's list_collections tool surfaces these
hints so a model can pick the right collection for find_in_collection on its
own, instead of the user always having to name it explicitly.

Why persistent, not just in-memory: the codebase-indexer MCP server is a
stdio subprocess spawned fresh per Claude Code session (per project's own
.mcp.json), so an in-memory-only cache would never survive past the session
that warmed it -- worthless for "any session, any repo," which is the whole
point of this feature. A SQLite file surviving restarts is what actually
makes list_collections's first call in a brand new session cheap once a
collection's hint has ever been fetched once, anywhere.

Storage lives in the SHARED cache DB from libs/cache_db.py (a file
explicitly meant for disposable, always-rebuildable caches -- see that
module's docstring for why it's kept separate from libs/savings_ledger.py's
savings.db, which the user wants kept forever). This module owns its own
table (collection_hints) against that shared connection; it doesn't resolve
or manage its own separate DB file.

Cache key is (qdrant_url, collection), not collection name alone: two
different Qdrant instances could coincidentally share a collection name
with unrelated content, and a name-only cache would silently serve the
wrong hint across instances with no way to detect it.

No TTL by design (per issue #81's explicit ask) -- a cached entry is only
ever refreshed by an explicit write. Only REAL hints get cached, though:
set_collection_description always writes through (an explicit, intentional
call -- even an empty string is a deliberate "clear this hint"), but
list_collections's own cache-miss fetch deliberately does NOT cache an
empty "no description set" result. That keeps this table limited to
collections someone actually described, not a growing pile of "nothing
here" rows, and means a description added outside this tool entirely
(e.g. directly via the Qdrant API) is picked up on list_collections's very
next call instead of staying invisible behind a stale cached "no hint."

This is a disposable, best-effort CACHE, not the source of truth (Qdrant's
own collection metadata is) -- so every function here fails open: a read
failure is treated as a full cache miss (falls through to a live Qdrant
fetch), and a write failure is swallowed with a stderr warning rather than
raised, since by the time either write call runs the authoritative Qdrant
operation has already succeeded. Letting a corrupt/locked/unwritable cache
file turn that into a reported tool failure would be strictly worse than
just not caching this one time -- confirmed as a real gap via Copilot
review on PR #82 (issue #81's PR): an unwritable cache before this fix
could make list_collections/set_collection_description fail outright even
though Qdrant itself was completely healthy.
"""

import datetime
import sqlite3
import sys
from contextlib import closing
from typing import Optional

from cache_db import connect as _connect_shared_cache


# Bumped whenever collection_hints' shape changes; _MIGRATIONS must gain a
# matching entry for every bump (issue #329, same scaffold shape as
# libs/savings_ledger.py's SCHEMA_VERSION/_MIGRATIONS/_run_migrations).
#
# One deliberate difference from savings_ledger: the version is NOT kept in
# PRAGMA user_version. That pragma is one integer per DB FILE, and this file
# (libs/cache_db.py's shared cache.db) is shared with other modules' tables,
# so claiming it would collide with the first of them to version its own
# table. Each module instead records its own version under its own name in
# the small schema_versions table below.
SCHEMA_VERSION = 1
_TABLE_NAME = "collection_hints"
# schema_versions key; deliberately the same as the table it versions.
_SCHEMA_NAME = _TABLE_NAME

# Always the LATEST shape -- a brand-new DB gets this directly and never
# replays a migration. `description` is nullable (NULL = never fetched, ''
# = explicitly cleared); `search_limit` is NULL until first computed (#326).
_CREATE_HINTS_DDL = """
    CREATE TABLE IF NOT EXISTS collection_hints (
        qdrant_url TEXT NOT NULL,
        collection TEXT NOT NULL,
        description TEXT,
        search_limit INTEGER,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (qdrant_url, collection)
    )
"""


def _migrate_to_v1(conn) -> None:
    """
    Makes `description` nullable and adds `search_limit` (issue #329) by
    dropping and recreating the table. A straight drop is safe because this
    table is a disposable cache (Qdrant's own collection metadata is the
    source of truth), so no rows are copied forward -- they repopulate on
    the next list_collections / set_collection_description call.

    Idempotent: if the table already has the v1 shape (e.g. a version row
    was lost) it is left alone rather than needlessly wiped.
    """
    info = conn.execute(f"PRAGMA table_info({_TABLE_NAME})").fetchall()
    # table_info rows: (cid, name, type, notnull, dflt_value, pk)
    columns = {row[1]: row for row in info}
    already_v1 = (
        "search_limit" in columns
        and "description" in columns
        and columns["description"][3] == 0
    )
    if already_v1:
        return
    conn.execute(f"DROP TABLE IF EXISTS {_TABLE_NAME}")
    conn.execute(_CREATE_HINTS_DDL)


_MIGRATIONS = {1: _migrate_to_v1}


def _is_current(conn) -> bool:
    """Lock-free read: True if the table exists and its recorded version is
    already >= SCHEMA_VERSION. Any sqlite error (e.g. schema_versions not
    created yet) just means "not known to be current"."""
    try:
        row = conn.execute(
            "SELECT version FROM schema_versions WHERE name = ? AND EXISTS "
            "(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?)",
            (_SCHEMA_NAME, _TABLE_NAME),
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None and row[0] >= SCHEMA_VERSION


def _run_migrations(conn) -> None:
    """
    Brings collection_hints up to SCHEMA_VERSION under one BEGIN IMMEDIATE
    lock, so two processes opening the same pre-existing cache.db at once
    (several Claude Code sessions each spawn their own MCP server) can't both
    read "not migrated" and then both migrate -- the whole check/migrate/
    record sequence is serialized, and SQLite DDL is transactional so a
    failure partway rolls back instead of leaving a half-migrated table.
    The fresh-DB fast path (no collection_hints table yet) creates the
    latest schema directly and records SCHEMA_VERSION without replaying
    history. A version with no registered migration raises rather than
    silently advancing the recorded version.
    """
    # Fast path: already at (or past) SCHEMA_VERSION means nothing to do, and
    # skipping BEGIN IMMEDIATE here keeps the common case (every cache read/
    # write after the first) from taking SQLite's write lock and contending
    # with other sessions' MCP servers. The check is repeated under the lock
    # below, so a racing migrator still can't be double-applied.
    if _is_current(conn):
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_versions "
            "(name TEXT PRIMARY KEY, version INTEGER NOT NULL)"
        )
        table_existed = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_TABLE_NAME,)
        ).fetchone() is not None
        if not table_existed:
            conn.execute(_CREATE_HINTS_DDL)
        else:
            row = conn.execute(
                "SELECT version FROM schema_versions WHERE name = ?", (_SCHEMA_NAME,)
            ).fetchone()
            # An existing table with no recorded version predates this
            # scaffold (the pre-#329 shape) -- treat it as version 0.
            current_version = row[0] if row else 0
            for version in range(current_version + 1, SCHEMA_VERSION + 1):
                if version not in _MIGRATIONS:
                    raise RuntimeError(
                        f"qdrant_collection_hints: SCHEMA_VERSION is {SCHEMA_VERSION} but no "
                        f"migration is registered for version {version} in _MIGRATIONS -- add one."
                    )
                _MIGRATIONS[version](conn)
        # MAX, never a plain overwrite, when the table already existed: an
        # older build opening a cache already migrated by a newer one must not
        # lower the recorded version, or the next newer process would replay
        # migrations. But when WE just created the table, it is exactly this
        # build's schema, so a stale higher row (table dropped, version row
        # left behind) must be overwritten -- keeping it would make a later
        # newer build think the table is already current and skip migrations.
        version_expr = (
            "MAX(schema_versions.version, excluded.version)"
            if table_existed
            else "excluded.version"
        )
        conn.execute(
            "INSERT INTO schema_versions (name, version) VALUES (?, ?) "
            f"ON CONFLICT(name) DO UPDATE SET version = {version_expr}",
            (_SCHEMA_NAME, SCHEMA_VERSION),
        )
    except Exception:
        # Guarded so a failing ROLLBACK (e.g. the same I/O error that caused
        # this) can't replace the original exception, which is the one worth
        # seeing. SQLite also auto-rolls-back on many such errors.
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    else:
        conn.execute("COMMIT")


def _connect():
    """
    Opens the shared cache DB and brings collection_hints up to date. Raises
    RuntimeError (not sqlite3.Error) only for developer misconfiguration --
    SCHEMA_VERSION bumped without a registered migration -- which the public
    functions below deliberately do NOT swallow: that is a code bug a test
    must catch (see tests' lockstep check), not a runtime cache failure to
    fail open on.
    """
    conn = _connect_shared_cache()
    try:
        _run_migrations(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def set_cached_description(qdrant_url: str, collection: str, description: str) -> None:
    """
    Upserts the cached hint for one collection. This always writes through
    unconditionally, including an empty string -- callers are expected to
    only call this when they actually want the value persisted (the
    explicit set_collection_description tool call, or list_collections's
    own cache-miss path choosing NOT to call this at all for an empty
    result -- see this module's docstring for why). This function itself
    doesn't apply that judgment; it just stores whatever it's given.

    Never raises on runtime cache failures (sqlite3.Error/OSError are caught
    and logged); the one deliberate exception is _connect's RuntimeError for
    a SCHEMA_VERSION/_MIGRATIONS mismatch -- a code bug, not a cache failure.
    Qdrant collection metadata is arbitrary JSON, so a
    description read back from it (list_collections's cache-miss path) may
    legally be None, a dict, or a list rather than a string. Since #329 the
    `description` column is nullable, so None would no longer raise -- but
    NULL there is the "never fetched" sentinel (get_cached_descriptions
    treats it as a miss), and this function's write is an explicit "this
    collection was checked" statement. Dicts/lists would still raise
    ProgrammingError when bound. So non-strings are normalized to "" (a
    cached, explicitly-empty hit) before anything is written, and this
    function never writes NULL; only set_cached_search_limit's insert path
    leaves description NULL. Separately,
    by the time this runs the caller's Qdrant write (if any) has already
    succeeded -- an unwritable/locked/corrupt cache file failing THIS write
    must not turn an already-successful operation into a reported failure,
    so any DB error here is caught and logged to stderr instead of raised.
    """
    if not isinstance(description, str):
        description = ""
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT INTO collection_hints (qdrant_url, collection, description, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(qdrant_url, collection) DO UPDATE SET "
                "description = excluded.description, updated_at = excluded.updated_at",
                (qdrant_url, collection, description, _now()),
            )
    except (sqlite3.Error, OSError) as e:
        print(f"[claude-runway] could not cache description for '{collection}': {e}", file=sys.stderr)


def get_cached_descriptions(qdrant_url: str, collections: list) -> dict:
    """
    Batch lookup for the given collection names under one qdrant_url.
    Returns only names actually present in the cache -- a name absent from
    the returned dict means "never fetched, or found to have no
    description and therefore deliberately not cached" (see this module's
    docstring), and callers treat that the same way either way: check live.
    A name present with an empty string means something explicitly cleared
    its hint via set_collection_description(collection, "") -- a real,
    intentional cached value, just an empty one.

    Never raises: a cache READ failure (unwritable/locked/corrupt file) is
    treated as if every requested name were a miss -- callers fall through
    to a live Qdrant fetch, which is strictly better than list_collections
    failing outright over a disposable cache being unavailable while Qdrant
    itself is perfectly healthy.
    """
    if not collections:
        return {}
    try:
        # description IS NOT NULL (below): a row can exist purely because
        # set_cached_search_limit created it (description NULL = "never
        # fetched"), which must read back as a description MISS, not a hit.
        placeholders = ",".join("?" for _ in collections)
        with closing(_connect()) as conn:
            rows = conn.execute(
                f"SELECT collection, description FROM collection_hints "
                f"WHERE qdrant_url = ? AND description IS NOT NULL AND collection IN ({placeholders})",
                (qdrant_url, *collections),
            ).fetchall()
        return {name: description for name, description in rows}
    except (sqlite3.Error, OSError) as e:
        print(f"[claude-runway] could not read cached descriptions: {e}", file=sys.stderr)
        return {}


def set_cached_search_limit(qdrant_url: str, collection: str, search_limit: int) -> None:
    """
    Upserts the cached default search limit for one collection (issue #329,
    part of #326). Only touches search_limit/updated_at -- never clobbers a
    cached description, just as set_cached_description never clobbers this.
    Never raises, for the same fail-open reasons as set_cached_description;
    a non-int (or bool) value is skipped with a warning instead of being
    bound into the INTEGER column.
    """
    if isinstance(search_limit, bool) or not isinstance(search_limit, int):
        print(
            f"[claude-runway] not caching non-integer search_limit for '{collection}': {search_limit!r}",
            file=sys.stderr,
        )
        return
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT INTO collection_hints (qdrant_url, collection, search_limit, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(qdrant_url, collection) DO UPDATE SET "
                "search_limit = excluded.search_limit, updated_at = excluded.updated_at",
                (qdrant_url, collection, search_limit, _now()),
            )
    except (sqlite3.Error, OSError) as e:
        print(f"[claude-runway] could not cache search_limit for '{collection}': {e}", file=sys.stderr)


def get_cached_search_limits(qdrant_url: str, collections: list) -> dict:
    """
    Batch lookup of cached search limits. Returns only names with a computed
    (non-NULL) limit -- absent means "never computed yet." Never raises: a
    read failure is treated as every name being a miss.
    """
    if not collections:
        return {}
    try:
        placeholders = ",".join("?" for _ in collections)
        with closing(_connect()) as conn:
            rows = conn.execute(
                f"SELECT collection, search_limit FROM collection_hints "
                f"WHERE qdrant_url = ? AND search_limit IS NOT NULL AND collection IN ({placeholders})",
                (qdrant_url, *collections),
            ).fetchall()
        return {name: limit for name, limit in rows}
    except (sqlite3.Error, OSError) as e:
        print(f"[claude-runway] could not read cached search limits: {e}", file=sys.stderr)
        return {}


def get_cached_search_limit(qdrant_url: str, collection: str) -> Optional[int]:
    """Single-collection convenience wrapper; returns None on a miss."""
    return get_cached_search_limits(qdrant_url, [collection]).get(collection)
