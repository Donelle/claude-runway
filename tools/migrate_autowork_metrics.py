#!/usr/bin/env python3
"""
One-time migration (issue #209): reads every compact currently stored under
Qdrant project "claude-runway-autowork-metrics" -- my-gh-autowork's
per-ticket run metrics, written by `.claude/skills/my-gh-autowork/SKILL.md`'s
Step 2b via `compact_store` (see `tools/compress_mcp_server.py`) -- and
inserts an equivalent row into the shared `metrics.db` (`libs/metrics_lib.py`,
issue #208) under `metric_id="autowork"`.

This is a THROWAWAY script, not a permanently-maintained tool -- it only
ever needs to run once, before issue #210's cutover removes the
`compact_store` call from Step 2b. Purely additive: it only ever READS
Qdrant (never deletes/modifies a compact) and only ever APPENDS to
metrics.db. It does NOT touch `.claude/skills/my-gh-autowork/SKILL.md` --
that cutover is issue #210's job, not this one's.

Collection resolution reuses `_collections_for_project`/`COMPACT_QDRANT_URL`/
`COMPACT_QDRANT_API_KEY` directly from `tools/compress_mcp_server.py` (this
repo's own single source of truth for how `compact_store`/`compact_find`
name and resolve a project's collection, including the case-insensitive
casing-drift aggregation and `_sanitize_project` hash-suffix logic it uses
internally) rather than re-deriving any of that here -- re-deriving it would
risk drifting from whatever compact_store/compact_find themselves resolve,
silently migrating from (or missing) the wrong collection.

Each Qdrant point's `information` payload field (JSON string; falls back to
the legacy `document` field for any pre-fix point, mirroring compact_find's
own fallback -- see compact_store's payload-shape comment) is parsed and
mapped to one metrics.db row:
    metric_id       "autowork"
    event_type      f"ticket_{outcome.lower()}" (e.g. "ticket_merged"),
                     or "ticket_unknown" if outcome is missing/unrecognized
    value           1.0 (one occurrence)
    metadata        the full parsed `information` dict, plus
                     "qdrant_point_id"/"qdrant_collection" for traceability
                     and idempotency (see below)
    event_timestamp the point's own `date` payload field (`YYYY-MM-DD`) as
                     `<date>T00:00:00Z` -- NOT migration run-time, so
                     trend()'s day/week bucketing reflects when each ticket
                     was actually worked, not when this script happened to
                     run. Falls back to run-time (logged as a warning) only
                     if `date` is missing/malformed on a given point.
    session_id      None (no session id is captured in this historical data)

Idempotent: before writing, reads every existing metrics.db row under
`metric_id="autowork"` whose `metadata.qdrant_point_id`/`qdrant_collection`
are set, and skips any Qdrant point already migrated -- keyed by the pair,
not `qdrant_point_id` alone, since a point id is only unique WITHIN its own
collection, not guaranteed unique across every collection this script can
resolve for one project (PR #246 review). Safe to re-run (e.g. after a
partial failure, or to pick up points logged after an earlier run) without
duplicating rows.

Guards against a REAL run overlapping with another concurrent real run of
this same script (PR #246 review): an exclusive lock file next to the
target metrics.db (`<metrics.db path>.migrate-autowork.lock`) is created
atomically before any writes and removed when the run finishes; a second
concurrent invocation fails fast with a clear error instead of racing the
idempotency check and duplicating rows. `--dry-run` never takes this lock
(it never writes, so concurrent dry-runs can't duplicate anything). If a
prior run crashed without cleaning up, delete the stale lock file manually
before retrying (the error message names its exact path).

Verification: after migrating, compares the TOTAL number of Qdrant points
found (across every resolved collection) against the number of
`metric_id="autowork"` rows in metrics.db that carry a
`(qdrant_collection, qdrant_point_id)` pair (i.e. rows this migration --
across all its runs -- is responsible for).
Prints PASS/FAIL and exits non-zero on a mismatch -- including when a point
was skipped as unparseable, since a source record silently dropped from the
migration is exactly the kind of incomplete-migration-gone-unnoticed
outcome this check exists to catch (PR #246 review).

Usage:
    .venv/bin/python tools/migrate_autowork_metrics.py [--dry-run]
        [--project NAME] [--qdrant-url URL] [--qdrant-api-key KEY]
        [--metrics-db PATH]

--dry-run previews what would be migrated (counts + a sample of the
mapping) without writing anything to metrics.db.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient  # noqa: E402

from qdrant_retry import call_with_retry  # noqa: E402
from compress_mcp_server import (  # noqa: E402
    COMPACT_QDRANT_API_KEY,
    COMPACT_QDRANT_URL,
    _collections_for_project,
)
from metrics_lib import MetricsStore  # noqa: E402

DEFAULT_PROJECT = "claude-runway-autowork-metrics"
METRIC_ID = "autowork"

# Outcomes documented in .claude/skills/my-gh-autowork/SKILL.md Step 2b --
# any other/missing value maps to "ticket_unknown" rather than crashing the
# whole migration over one malformed point.
_KNOWN_OUTCOMES = {"merged", "blocked", "failed"}


def _fetch_points(client: QdrantClient, project: str) -> list:
    """
    Every Qdrant point across every collection that resolves to `project`,
    via the exact same case-insensitive multi-collection resolution
    compact_find itself uses (see _collections_for_project) -- so this
    script can never see a different, narrower (or broader) set of points
    than compact_find would report for the same project.

    PR #246 review (Copilot): _collections_for_project resolves COLLECTION
    names that hold this project's history under ANY historical casing --
    it does NOT guarantee every point inside a resolved collection actually
    belongs to this project. A collection can be SHARED (compact_store's own
    `collection` param is a caller-controlled override -- see its docstring),
    so a resolved collection can legitimately contain another project's
    points alongside this one's. compact_find itself defends against exactly
    this with a case-insensitive per-point filter on the `project` payload
    field AFTER scrolling (tools/compress_mcp_server.py's default/no-query
    scroll path); an earlier version of this function appended every point
    unconditionally, so one matching autowork point in a shared collection
    would have caused an unrelated project's whole payload to be migrated in
    as an "autowork" metric (confirmed by reproduction: a fake shared
    collection with one point per project fed a foreign point straight
    through before this filter was added). Applying the identical filter
    here keeps this script's isolation guarantee in lockstep with
    compact_find's own.
    """
    case_insensitive_target = project.casefold()
    col_list = _collections_for_project(client, project)
    all_points = []
    for col in col_list:
        offset = None
        while True:
            batch, offset = call_with_retry(
                client.scroll,
                collection_name=col,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for p in batch:
                if (p.payload or {}).get("project", "").casefold() != case_insensitive_target:
                    continue
                all_points.append((col, p))
            if offset is None:
                break
    return all_points


def _parse_information(payload: dict) -> Optional[dict]:
    """
    Parse a point's `information` payload field as JSON, falling back to the
    legacy `document` field (compact_store wrote both fields identically
    going forward, but some very old points may only have `document` --
    same fallback compact_find itself applies). Returns None (never raises)
    if neither field is present/parseable, so one malformed point can be
    skipped and reported rather than aborting the whole run.
    """
    raw = payload.get("information") or payload.get("document")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _event_timestamp_for(payload: dict, point_id) -> str:
    """
    Derive metrics.db's `event_timestamp` from the point's own `date`
    payload field (`YYYY-MM-DD`, the same format compact_store validates on
    write). Falls back to current time -- logged as a warning, never a hard
    failure -- if `date` is missing or not a REAL calendar date in that
    format, since one malformed date on one point shouldn't block migrating
    everything else.

    PR #246 review (Copilot, round 1): a regex checking only
    `\\d{4}-\\d{2}-\\d{2}` SHAPE (not confirmed by reproduction to occur in
    real compact_store data -- every real point inspected had a valid date
    -- but not something this script can assume forever) would accept a
    calendar-invalid value like `"2026-99-99"` and persist it as
    `"2026-99-99T00:00:00Z"`; confirmed by reproduction that
    `datetime.fromisoformat` (which `MetricsStore.trend()` calls for
    `bucket="week"`) raises `ValueError` on exactly that string, crashing
    every future `trend()` call for metric_id="autowork", not just against
    the one bad row.

    PR #246 review (Copilot, round 2): `datetime.strptime` alone isn't
    sufficient either -- it's lenient about zero-padding, so it happily
    parses `"2026-8-5"` (non-zero-padded month/day) and would have returned
    `"2026-8-5T00:00:00Z"` verbatim, bypassing the intended fallback; that
    non-canonical string then crashes `datetime.fromisoformat` downstream
    exactly like the shape-only regex did (confirmed by reproduction).
    Reformatting the parsed date back through `strftime` and requiring an
    EXACT round-trip match to the original `date` string is what enforces
    the canonical zero-padded `YYYY-MM-DD` shape together with real
    calendar validity.
    """
    import datetime as _datetime

    date = payload.get("date", "")
    if isinstance(date, str):
        try:
            parsed = _datetime.datetime.strptime(date, "%Y-%m-%d")
            if parsed.strftime("%Y-%m-%d") != date:
                raise ValueError("not canonical (fails strict round-trip)")
        except ValueError:
            pass
        else:
            return f"{date}T00:00:00Z"
    print(
        f"[migrate_autowork_metrics] point {point_id}: missing/malformed 'date' payload "
        f"({date!r}) -- falling back to current time for event_timestamp.",
        file=sys.stderr,
    )
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _already_migrated_keys(store: MetricsStore) -> set:
    """
    Every `(qdrant_collection, qdrant_point_id)` pair already migrated in a
    PRIOR run of this script, read directly from metrics.db (metrics_lib
    exposes no query-by-metadata API of its own -- this is the one place
    outside metrics_lib.py's own test suite that reaches into the raw
    table, for exactly the same "read back what was actually stored"
    reason those tests do). Used to make this script idempotent: safe to
    re-run without duplicating rows.

    PR #246 review (Copilot, round 3): a Qdrant point id is only unique
    WITHIN its own collection, not guaranteed unique ACROSS the (possibly
    more than one, via casing-drift aggregation -- see
    _collections_for_project) collections this script can resolve for one
    project. An earlier version of this function keyed solely by
    `qdrant_point_id`, discarding `qdrant_collection` -- if two resolved
    collections happened to contain the same point id, both got migrated
    as two separate metrics.db rows on the first run (this part was
    already correct), but the point-id-only set here collapsed them to a
    single count, so `_autowork_row_count()` under-counted relative to the
    real Qdrant point total and row-count verification could never pass;
    confirmed by reproduction (two fake collections sharing one point id:
    2 points found, 2 rows actually written, but the old code reported
    only 1 migrated row). Worse, a PARTIAL prior run (e.g. crashed after
    migrating collection A's copy of the id but before reaching
    collection B's) would have made the point-id-only skip check treat
    collection B's still-unmigrated copy as "already migrated" too,
    silently losing it forever. Keying by the composite
    `(qdrant_collection, qdrant_point_id)` pair everywhere -- here, the
    skip check in migrate(), and the row count -- fixes both.

    PR #246 review (Copilot, round 2): an existing SQLite file with no
    `metrics` table yet (e.g. an empty file created by `touch`, or by some
    other process opening the path before anything ever wrote to it --
    the `--metrics-db` override documents pointing at an arbitrary path,
    not only ones this script itself created) made the `db_path.exists()`
    guard below take the "real db, query it" branch, and the raw query
    then raised `sqlite3.OperationalError: no such table: metrics` before
    MetricsStore.record()'s own `_connect()` ever got a chance to create
    it -- confirmed by reproduction. `db_path.exists()` alone isn't the
    right test for "has anything been migrated into this db yet"; "does
    the metrics table itself exist" is. Caught here and treated the same
    as a from-scratch db (nothing migrated yet), rather than crashing.
    """
    import sqlite3

    db_path = store._resolve_path()
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT metadata FROM metrics WHERE metric_id = ? AND metadata IS NOT NULL",
            (METRIC_ID,),
        ).fetchall()
    except sqlite3.OperationalError:
        # No `metrics` table yet -- an empty/uninitialized db file, not a
        # real error condition. Same "nothing migrated yet" result as the
        # db_path.exists() guard above.
        return set()
    finally:
        conn.close()
    seen = set()
    for (metadata_json,) in rows:
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, ValueError):
            continue
        point_id = metadata.get("qdrant_point_id")
        collection = metadata.get("qdrant_collection")
        if point_id and collection:
            seen.add((str(collection), str(point_id)))
    return seen


def _autowork_row_count(store: MetricsStore) -> int:
    """Count of metric_id="autowork" rows carrying a
    `(qdrant_collection, qdrant_point_id)` pair -- i.e. rows this migration
    (across all its runs) is responsible for, used by the row-count
    verification at the end of migrate()."""
    return len(_already_migrated_keys(store))


def _lock_path_for(store: MetricsStore) -> Path:
    return Path(str(store._resolve_path()) + ".migrate-autowork.lock")


def _acquire_lock(lock_path: Path) -> int:
    """
    Best-effort, atomic cross-process exclusive lock for the duration of a
    real (non-dry-run) migrate() call, scoped to THIS one-time script only
    -- does not touch metrics_lib.py's shared, domain-agnostic schema.

    PR #246 review (Copilot, "previously missed" -- concurrent migrations):
    the idempotency check (_already_migrated_keys) is a read performed
    BEFORE any inserts, with no lock between the read and the writes. Two
    concurrent invocations of this script can both read the same
    "not yet migrated" set, then both proceed to insert the same points --
    SQLite serializes each individual INSERT, but nothing stops both
    processes from making one, so the same source point can land as two
    duplicate metrics.db rows; the set-based verifier then collapses those
    duplicates back down and can misreport PASS despite the duplication.

    `os.open(..., O_CREAT | O_EXCL)` is atomic at the OS level (unlike a
    separate exists-then-create check, which would itself be a TOCTOU
    race) and works the same way on POSIX and Windows, unlike
    platform-specific `fcntl`/`msvcrt` locking -- appropriate here since
    this is a best-effort guard for a one-time script's realistic usage
    (one operator running it once), not a general-purpose distributed
    lock. Raises FileExistsError if another instance already holds it;
    the caller decides how to report that.
    """
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
    finally:
        os.close(fd)
    return fd


def migrate(args) -> int:
    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key or None)
    points = _fetch_points(client, args.project)
    total_found = len(points)
    print(f"Found {total_found} point(s) in Qdrant for project '{args.project}'.")

    store = MetricsStore(db_path=Path(args.metrics_db) if args.metrics_db else None)

    lock_path = _lock_path_for(store)
    lock_acquired = False
    if not args.dry_run:
        # Dry-run never writes, so two concurrent dry-runs (or a dry-run
        # alongside a real run) can't duplicate anything -- no lock needed.
        try:
            _acquire_lock(lock_path)
            lock_acquired = True
        except FileExistsError:
            print(
                f"FAIL: another migration run appears to be in progress (lock file exists: {lock_path}). "
                "If no other run is actually active (e.g. a prior run crashed without cleaning up), "
                "delete that file and retry.",
                file=sys.stderr,
            )
            return 1

    try:
        return _migrate_locked(args, points, total_found, store)
    finally:
        if lock_acquired:
            try:
                lock_path.unlink()
            except OSError:
                pass


def _migrate_locked(args, points: list, total_found: int, store: MetricsStore) -> int:
    already_migrated = _already_migrated_keys(store)
    if already_migrated:
        print(f"{len(already_migrated)} point(s) already migrated in a prior run -- will be skipped.")

    migrated = 0
    skipped_already = 0
    skipped_unparseable = []
    for col, point in points:
        point_id = str(point.id)
        if (col, point_id) in already_migrated:
            skipped_already += 1
            continue
        payload = point.payload or {}
        information = _parse_information(payload)
        if information is None:
            skipped_unparseable.append(point_id)
            continue

        outcome = str(information.get("outcome") or "").strip().lower()
        event_type = f"ticket_{outcome}" if outcome in _KNOWN_OUTCOMES else "ticket_unknown"
        event_timestamp = _event_timestamp_for(payload, point_id)
        metadata = dict(information)
        metadata["qdrant_point_id"] = point_id
        metadata["qdrant_collection"] = col

        if args.dry_run:
            print(
                f"  [dry-run] would migrate point {point_id} (collection={col}): "
                f"event_type={event_type!r}, event_timestamp={event_timestamp!r}, "
                f"issue={information.get('issue')!r}, outcome={information.get('outcome')!r}"
            )
            migrated += 1
            continue

        store.record(
            METRIC_ID,
            event_type,
            value=1.0,
            metadata=metadata,
            event_timestamp=event_timestamp,
        )
        migrated += 1

    if skipped_unparseable:
        print(
            f"Skipped {len(skipped_unparseable)} point(s) with no parseable "
            f"information/document payload: {skipped_unparseable}",
            file=sys.stderr,
        )

    if args.dry_run:
        print(
            f"Dry run: {migrated} point(s) would be migrated, "
            f"{skipped_already} already migrated, "
            f"{len(skipped_unparseable)} unparseable. Nothing written."
        )
        return 0

    print(
        f"Migrated {migrated} new point(s); {skipped_already} already migrated "
        f"(skipped); {len(skipped_unparseable)} unparseable (skipped)."
    )

    # Verification: row count in metrics.db (for rows this migration wrote,
    # across all its runs) must equal the TOTAL number of Qdrant points
    # found -- full stop, not "total minus whatever this run couldn't
    # parse." PR #246 review (Copilot): an earlier version subtracted
    # skipped_unparseable from `expected`, which let the script print PASS
    # and exit 0 even though those source records were silently dropped --
    # exactly the "incomplete migration goes unnoticed" failure this
    # verification step exists to catch. Any unparseable point (should not
    # happen against real compact_store data -- confirmed zero occurrences
    # across the real 23-point production collection -- but not something
    # to silently paper over if it ever does) now makes this FAIL until a
    # human looks at it and either fixes the source data or extends the
    # parser, rather than quietly accepting a smaller-than-expected count.
    expected = total_found
    actual = _autowork_row_count(store)
    print(f"Verification: expected {expected} migrated row(s), found {actual} in metrics.db.")
    if actual != expected:
        print("FAIL: row count mismatch -- see output above.", file=sys.stderr)
        return 1
    print("PASS: metrics.db row count matches Qdrant point count.")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=DEFAULT_PROJECT, help=f"Qdrant compact_store project name (default: {DEFAULT_PROJECT!r})")
    p.add_argument("--qdrant-url", default=COMPACT_QDRANT_URL)
    p.add_argument("--qdrant-api-key", default=COMPACT_QDRANT_API_KEY)
    p.add_argument("--metrics-db", default=None, help="Override metrics.db path (default: CLAUDE_RUNWAY_METRICS_DB env var, or ~/.claude/claude-runway/metrics.db)")
    p.add_argument("--dry-run", action="store_true", help="Preview the migration without writing anything to metrics.db")
    return p.parse_args()


def main() -> None:
    sys.exit(migrate(parse_args()))


if __name__ == "__main__":
    main()
