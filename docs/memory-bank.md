# Memory bank

A durable, cross-session knowledge store — `remember`, `recall`, `forget` — provided by the `memory-bank` MCP server (`tools/memory_bank_mcp_server.py`, backed by `libs/memory_bank_lib.py`).

This is separate from the Qdrant codebase memory piece described in the main README: `qdrant-store`/`qdrant-find` are for THIS project's own conceptual notes about its code; memory-bank is for lessons, decisions, and corrections meant to survive a fresh session, and can span every project on the machine.

## Why a separate collection, not per-project

`codebase-indexer` gives every project its own Qdrant collection, switched automatically by that project's `.mcp.json`. Memory-bank works differently on purpose: every project's memory-bank server writes into the SAME shared Qdrant collection (`MEMORY_BANK_COLLECTION`, default `memory-bank`) — that's what lets `recall` default to "this project's own memories plus anything tagged general" without a separate cross-repo lookup step (see [Cross-repo lookups](cross-repo-lookups.md) for how that lookup works for the *codebase* index, which memory-bank deliberately avoids needing).

`forget` does NOT share that same default-scope behavior — it's a deletion tool, not a lookup, so it's deliberately narrower: deleting by an exact `point_id` has no scope concept at all among memory-bank points (any memory-bank point, from any project, though deleting one that isn't this project's own requires `confirm=True` — it refuses outright to touch anything that isn't tagged as a memory-bank point in the first place), and its bulk `wipe_all` mode only ever clears THIS project's own memories, explicitly excluding anything tagged `general` — there's no bulk way to wipe cross-project memories at all.

`MEMORY_BANK_ID` is a plain identifier tagging which project wrote a given memory (`metadata.repo`) — it doesn't have to match any real Qdrant collection name. `MEMORY_BANK_COLLECTION` is the actual shared collection name, common to every project on the Qdrant instance.

Because the collection is shared, `EMBEDDING_MODEL` is locked in by whichever project's memory-bank server created it first — every project sharing this collection needs to agree on this one value (see the README's [Known limitations](../README.md#known-limitations) for the mismatch-detection behavior this implies).

## Setup

**Recommended:** pass `--memory-bank-collection`/`--memory-bank-id` to `tools/setup_project.py init` ([Installation](installation.md)'s step 4) — both are optional:

- `--memory-bank-collection` defaults to the literal `"memory-bank"` if blank. Deliberately NOT derived from anything project-specific, since this is meant to be the SAME value across every project sharing the collection — only override it if you're deliberately running more than one separate memory-bank collection on the same Qdrant instance.
- `--memory-bank-id` defaults to `--collection-name` (this project's own codebase-index collection name) if blank. Unlike `--memory-bank-collection`, this IS meant to vary per project.

**Neither default is sticky across a rerun.** `setup_project.py init` regenerates its owned server blocks wholesale each time, so omitting either flag on a LATER run recomputes these defaults fresh — it doesn't detect or preserve whatever value is already sitting in the project's current `.mcp.json`. For a project that has already accumulated real memories, rerunning without explicitly passing the SAME `--memory-bank-collection`/`--memory-bank-id` values used originally can silently disconnect from existing ones (a collection override reset back to `"memory-bank"`) or split future memories under a new tag (an ID re-derived from a `--collection-name` that changed since the last run). The CLI's own `--memory-bank-id` help text already warns about this specific case — pass both values explicitly on every rerun once a project has real memories, not just the first time.

The script warns (rather than silently allowing) two collisions:

- **`--memory-bank-id` resolving to `"general"`** — `"general"` is memory-bank's reserved sentinel for cross-project knowledge (see "Using it" below); a project's own ID colliding with it would tag that project's memories as if they applied everywhere. Pass an explicit `--memory-bank-id` to fix.
- **This project's `--collection-name` equal to `--memory-bank-collection`** — `index_repo`/`sync_repo` refuse outright to run against the shared memory-bank collection (issue #175), so this collision would leave the project's own codebase index unable to run at all. Pass a different `--collection-name` or `--memory-bank-collection` to fix.

**Manual fallback**: copy `templates/mcp.json.template`'s `memory-bank` block into your TARGET PROJECT's own `.mcp.json` (per [Installation](installation.md)'s step 4) and edit these fields there — never edit the shared `templates/mcp.json.template` file itself, which doesn't configure any project and risks committing project-specific paths or credentials into this tools repo:

- Replace `REPLACE-WITH-VENV-PYTHON` with the absolute venv Python path.
- Replace `/absolute/path/to/tools/memory_bank_mcp_server.py` with the real path.
- Replace `MEMORY_BANK_ID`'s placeholder with a plain identifier for this project (or leave it equal to this project's `COLLECTION_NAME` if unsure).
- Leave `MEMORY_BANK_COLLECTION` at its default (`memory-bank`) unless you have a specific reason to run a separate shared collection. **If you DO override it, set the exact same value in the `codebase-indexer` block's own `MEMORY_BANK_COLLECTION` entry too** — `index_repo`/`sync_repo`'s guard against ever touching the shared memory-bank collection checks against that block's copy, not this one, so a mismatch here leaves the indexer only protecting the default name while your memories actually live under the custom one. `setup_project.py`'s `--memory-bank-collection` flag (recommended, above) writes both automatically; this is specifically a manual-editing gotcha.
- `QDRANT_URL`/`QDRANT_API_KEY`/`EMBEDDING_MODEL` follow the same values as the `qdrant`/`codebase-indexer` blocks in the same file — `EMBEDDING_MODEL` specifically must match whatever model created the shared collection, not just this project's own codebase-index collection.

## Using it

- `recall(query, ...)` — defaults to this project's own memories plus anything tagged `general`. Pass `repo=<other-project-id>` or `all_repos=True` only when genuinely needed, not as a default broadening. Results are ranked by `effective_score` (raw cosine similarity rescaled to a nonnegative `[0, 1]` scale, then multiplied by `weight`), not raw similarity alone (issue #177) — each hit reports both `score` (raw similarity, unchanged meaning) and `effective_score` (what ranking/truncation actually used) so the re-ranking is inspectable, not hidden. Raw cosine similarity is rescaled BEFORE the multiply (not multiplied directly) because it can be negative, which would otherwise invert the weight semantics below (a `weight=0` memory could rank above a normal one) — see `libs/memory_bank_lib.py`'s `_normalize_similarity` docstring.
- `remember(summary, description, kind, general=False, weight=1.0, ...)` — `summary` is the only field actually searched, so write it the way a future query would be phrased; put the full verbatim detail in `description`. Set `general=True` only when the user explicitly signals the memory applies across every project. `weight` (issue #177) is a static, FINITE, NONNEGATIVE quality multiplier stored as `metadata.weight` (a negative, infinite, or NaN value is rejected with an error): `1.0` (default) is a true no-op against plain-similarity ranking, `>1.0` boosts a known-good memory above equally-similar competitors, `0.0` de-emphasizes a stale/superseded one without deleting it. A point written before this field existed is treated as `weight=1.0` at read time. There's no separate "update weight" tool — `forget` the point and `remember` it again to change it.
- `forget(...)` — requires explicit user confirmation before calling with `confirm=True`; show what would be deleted first.

For the exact usage guidance Claude should follow when deciding whether to call these tools, see `templates/CLAUDE.md.template`'s "Durable memory (memory-bank MCP)" section — copy it into a project's `CLAUDE.md` alongside the Qdrant/local-compress sections it already documents.

## Usage event logging

`recall`/`remember` calls are passively logged, one row per event, to a dedicated
`~/.claude/claude-runway/memory-events.db` (issue #179) — a diagnostic signal for
validating the reuse thesis behind this feature (are stored memories actually
getting recalled later, by which projects, how soon after creation) without
relying entirely on manually-run scenarios. `forget` is deliberately NOT logged
here — tracking it would only be meaningful alongside an autonomous-removal
feature that doesn't exist yet, since `forget` today always requires explicit
human `confirm=True`.

This is a SEPARATE file from `libs/savings_ledger.py`'s `savings.db` on purpose:
that file tracks session-level token-savings aggregates, a different concern
from per-point access tracking, and mixing the two would couple unrelated
metric domains together. It only ever stores ids/timestamps/kind/repo/session
metadata — never a memory's `summary` or `description` text.

Unlike `CLAUDE_RUNWAY_TRACK_SAVINGS` (opt-in, off by default), this is **on by
default** — it costs one local SQLite `INSERT`, no network call, no model
call, and no token cost, unlike the savings tracker's transcript-parsing
sibling feature. Set `CLAUDE_RUNWAY_TRACK_MEMORY_EVENTS` to a falsy value to
disable it, or `CLAUDE_RUNWAY_MEMORY_EVENTS_DB` to relocate the file — see
[Environment variables](environment-variables.md). Neither needs the
"set in both `.mcp.json` and the shell" coordination those two savings-tracker
variables need, since no hook ever reads this file — only the `memory-bank`
MCP server does.

Each row records `event_type` (`recall`/`remember`), `point_id`, `repo`,
`kind`, `summary_created_at` (the memory's own creation time, denormalized at
log time), `event_timestamp`, `turn` (a per-session call-sequence counter —
"3rd memory-bank call this session," not a real Claude Code conversation
turn), `session_id`, and `project`. `session_id` comes from
`libs/session_id_lib.py`'s `SessionIdStrategy.PROXY` (issue #198/#214) — a
process-lifetime UUID cached once when the MCP server starts, not Claude
Code's own internal session id. See that module's docstring for the full
four-strategy classification (`HOOK_PAYLOAD`/`SHADOW_FILE`/
`TRANSCRIPT_SCAN`/`PROXY`) of every way this toolkit's MCP-server/hook code
can get or approximate a session_id; PROXY is the right fit here because an
MCP server has no hook payload to read a real session id from (only hooks
receive one, via their stdin payload), and a stable per-process grouping key
is good enough for this passive log's purposes — since a stdio server is
spawned fresh per Claude Code session, that proxy is a documented, practical
stand-in for "this session," not the real thing. Surfacing this data (a
query/report tool, a `/my-savings`-style summary) is intentionally out of
scope for now — this is passive collection only.

## Where the data lives

The same Qdrant instance as `codebase-indexer`'s per-project collections, but in one collection shared across every project, and never touched by `index_repo`/`sync_repo`'s reset/delete paths — those always preserve `metadata.source == "memory-bank"` points, even during a full reset (issue #175; see the README's [Known limitations](../README.md#known-limitations) for the mechanics of that filtered-delete behavior).
