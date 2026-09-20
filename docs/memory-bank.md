# Memory bank

A durable, cross-session knowledge store — `remember`, `recall`, `forget` — provided by the `memory-bank` MCP server (`tools/memory_bank_mcp_server.py`, backed by `libs/memory_bank_lib.py`).

This is separate from the Qdrant codebase memory piece described in the main README: `qdrant-store`/`qdrant-find` are for THIS project's own conceptual notes about its code; memory-bank is for lessons, decisions, and corrections meant to survive a fresh session, and can span every project on the machine.

## Why a separate collection, not per-project

`codebase-indexer` gives every project its own Qdrant collection, switched automatically by that project's `.mcp.json`. Memory-bank works differently on purpose: every project's memory-bank server writes into the SAME shared Qdrant collection (`MEMORY_BANK_COLLECTION`, default `memory-bank`) — that's what lets `recall` default to "this project's own memories plus anything tagged general" without a separate cross-repo lookup step (see [Cross-repo lookups](cross-repo-lookups.md) for how that lookup works for the *codebase* index, which memory-bank deliberately avoids needing).

`forget` does NOT share that same default-scope behavior — it's a deletion tool, not a lookup, so it's deliberately narrower: deleting by an exact `point_id` has no scope concept at all (any point, from any project, though deleting one that isn't this project's own requires `confirm=True`), and its bulk `wipe_all` mode only ever clears THIS project's own memories, explicitly excluding anything tagged `general` — there's no bulk way to wipe cross-project memories at all.

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

- `recall(query, ...)` — defaults to this project's own memories plus anything tagged `general`. Pass `repo=<other-project-id>` or `all_repos=True` only when genuinely needed, not as a default broadening.
- `remember(summary, description, kind, general=False, ...)` — `summary` is the only field actually searched, so write it the way a future query would be phrased; put the full verbatim detail in `description`. Set `general=True` only when the user explicitly signals the memory applies across every project.
- `forget(...)` — requires explicit user confirmation before calling with `confirm=True`; show what would be deleted first.

For the exact usage guidance Claude should follow when deciding whether to call these tools, see `templates/CLAUDE.md.template`'s "Durable memory (memory-bank MCP)" section — copy it into a project's `CLAUDE.md` alongside the Qdrant/local-compress sections it already documents.

## Where the data lives

The same Qdrant instance as `codebase-indexer`'s per-project collections, but in one collection shared across every project, and never touched by `index_repo`/`sync_repo`'s reset/delete paths — those always preserve `metadata.source == "memory-bank"` points, even during a full reset (issue #175; see the README's [Known limitations](../README.md#known-limitations) for the mechanics of that filtered-delete behavior).
