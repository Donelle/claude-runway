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

from cache_db import connect as _connect_shared_cache


def _connect():
    conn = _connect_shared_cache()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS collection_hints (
            qdrant_url TEXT NOT NULL,
            collection TEXT NOT NULL,
            description TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (qdrant_url, collection)
        )
        """
    )
    return conn


def set_cached_description(qdrant_url: str, collection: str, description: str) -> None:
    """
    Upserts the cached hint for one collection. This always writes through
    unconditionally, including an empty string -- callers are expected to
    only call this when they actually want the value persisted (the
    explicit set_collection_description tool call, or list_collections's
    own cache-miss path choosing NOT to call this at all for an empty
    result -- see this module's docstring for why). This function itself
    doesn't apply that judgment; it just stores whatever it's given.

    Never raises. Qdrant collection metadata is arbitrary JSON, so a
    description read back from it (list_collections's cache-miss path) may
    legally be None, a dict, or a list rather than a string -- reproduced
    directly that binding any of those into this table's TEXT NOT NULL
    column raises IntegrityError (None) or ProgrammingError (dict/list), so
    non-strings are normalized to "" before anything is written. Separately,
    by the time this runs the caller's Qdrant write (if any) has already
    succeeded -- an unwritable/locked/corrupt cache file failing THIS write
    must not turn an already-successful operation into a reported failure,
    so any DB error here is caught and logged to stderr instead of raised.
    """
    if not isinstance(description, str):
        description = ""
    try:
        conn = _connect()
        with conn:
            conn.execute(
                "INSERT INTO collection_hints (qdrant_url, collection, description, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(qdrant_url, collection) DO UPDATE SET "
                "description = excluded.description, updated_at = excluded.updated_at",
                (qdrant_url, collection, description, datetime.datetime.now(datetime.timezone.utc).isoformat()),
            )
        conn.close()
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
        conn = _connect()
        placeholders = ",".join("?" for _ in collections)
        rows = conn.execute(
            f"SELECT collection, description FROM collection_hints "
            f"WHERE qdrant_url = ? AND collection IN ({placeholders})",
            (qdrant_url, *collections),
        ).fetchall()
        conn.close()
        return {name: description for name, description in rows}
    except (sqlite3.Error, OSError) as e:
        print(f"[claude-runway] could not read cached descriptions: {e}", file=sys.stderr)
        return {}
