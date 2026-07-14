# Local-Compute Token Savings for Claude Code

Two independent pieces, both built on the same idea: push work onto local compute (a local vector DB, a local LLM) instead of Claude's own context, so Claude only pays tokens for a distilled result rather than raw data.

1. **Qdrant codebase memory** — per-project semantic memory over code/docs, so Claude can retrieve relevant chunks instead of grepping and reading whole files, and persist distilled findings across sessions.
2. **Local compression (LM Studio)** — a local model summarizes large, mechanical content (logs, build output, big diffs) server-side before anything reaches Claude's context.

Each piece works independently — you don't need LM Studio to use the Qdrant memory, or vice versa.

## Files (13)

| File                              | Purpose                                                                                                                                                                                                                   |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `requirements.txt`                | `pip install -r requirements.txt` for everything both pieces need. Grouped by piece with comments — see the file itself if you only want a subset installed.                                                             |
| `libs/qdrant_ingest_lib.py`       | Shared, dependency-free chunking logic (code by line windows, docs by heading). Used by both files below — keeps them from drifting apart.                                                                                |
| `ingest_to_qdrant.py`             | Standalone CLI script for one-off/manual full indexing. Prints progress to the terminal; use this for the first big index of a repo.                                                                                      |
| `ingest_mcp_server.py`            | MCP server exposing `index_repo`, `sync_repo`, `preview_index`, and `get_collection_info` as tools Claude can call directly.                                                                                              |
| `libs/local_compress_lib.py`      | Shared, MCP-independent map-reduce compression logic (chunking, model resolution, LM Studio calls). Used by both `compress_mcp_server.py` and `hooks/compress_bash_output.py` — keeps them from drifting apart.          |
| `compress_mcp_server.py`          | MCP server exposing `compress_file`, `compress_command_output`, `fetch_url`, `compress_text`, and `list_local_models`, backed by a local LM Studio model. Still a draft — design notes and open questions are in the file's docstring. |
| `templates/mcp.json.template`     | Per-project Claude Code config wiring up the `qdrant-find`/`qdrant-store` server, the `codebase-indexer` server, and (optionally) `local-compress` to the same project.                                                   |
| `templates/CLAUDE.md.template`    | Usage rules for Claude covering when to reach for each tool (and when NOT to) — copy the relevant section(s) into a project's `CLAUDE.md`. Note: CLAUDE.md guidance only applies to open-ended prompts, not skills — see "Skills and hooks" below. |
| `templates/settings.json.template` | `PostToolUse` hook config that compresses ANY Bash output over a size threshold, regardless of which command produced it — see "Skills and hooks" below.                                                                  |
| `hooks/compress_bash_output.py`   | The hook script referenced by `templates/settings.json.template` at its path in the tools repo (nothing copied into the target project). Filename is legacy -- fires on Bash, Grep, Glob, WebFetch, and WebSearch (matcher covers all five, deliberately excluding Read/Write/Edit), compresses via `libs/local_compress_lib.py` (found automatically in `libs/` under the repo root) if output exceeds a threshold, fails open (leaves output untouched) if LM Studio is unreachable or the import path is somehow wrong. |
| `hooks/redirect-to-compress.sh`   | Superseded by `compress_bash_output.py` — a narrower, hard-blocking `PreToolUse` alternative for a hardcoded command list (`dotnet test`/`build`). Kept for reference; see the tradeoffs noted in `templates/settings.json.template`. |
| `hooks/redirect_webfetch_to_fetch_url.py` | `PreToolUse` hook on `WebFetch` — hard-denies the call and tells Claude to use `fetch_url` instead, but only when LM Studio is reachable right now (checked live); fails open (lets WebFetch through) otherwise. |
| `EVALUATION.md`                   | A plan for measuring whether this actually reduces token usage, rather than assuming it does — covers both the Qdrant memory piece and the local-compress piece as separate tracks, plus hard-won lessons on measuring `/usage` cleanly. |

## Prerequisites

- A running Qdrant instance (local or remote), reachable at a URL — required for piece 1
- LM Studio (https://lmstudio.ai) running locally with a model loaded — required for piece 2, optional otherwise
- Python 3.10+
- Claude Code

## Installation

**1. Install dependencies**

```bash
pip install -r requirements.txt --break-system-packages
```

This installs everything for both pieces plus the optional `.gitignore` support. If you only want the Qdrant memory piece and not local-compress, install just the core group listed at the top of `requirements.txt` and skip the rest:

```bash
pip install "mcp[cli]" mcp-server-qdrant qdrant-client --break-system-packages
# only if you're also using the local-compress piece:
pip install openai requests trafilatura --break-system-packages
# optional, enables respecting a project's .gitignore during indexing:
pip install pathspec --break-system-packages
```

If `compress_mcp_server.py` fails to start in Claude Code with `MCP error -32000: Connection closed`, it's almost always a missing dependency (`requests`/`trafilatura` were added later for `fetch_url` and are easy to miss if you installed `openai` before that). Run `python /absolute/path/to/compress_mcp_server.py` directly in a terminal to see the real `ModuleNotFoundError` instead of just the opaque connection error.

**2. Place the files somewhere stable** (not inside a project repo — one copy serves all projects), e.g.:

```bash
git clone <this-repo> ~/tools/qdrant-claude-memory
```

**3. Per project you want memory for:**

- Copy `templates/mcp.json.template` to `.mcp.json` at that project's repo root.
- Replace `REPLACE-WITH-THIS-PROJECTS-NAME` (appears twice) with a unique collection name for that project.
- Replace `/absolute/path/to/ingest_mcp_server.py` with the real path from step 2.
- Commit `.mcp.json` to the repo.

This is what makes memory automatically project-scoped: opening Claude Code in a project loads that project's `.mcp.json`, which points at that project's collection. Opening a different project uses a different collection automatically — no manual switching.

**Controlling what gets ingested:** by default, `index_repo`/`sync_repo`/`ingest_to_qdrant.py` include a built-in list of common code/doc extensions, skip common noise folders (`.git`, `node_modules`, `venv`, `dist`, `build`, etc.), and additionally respect the project's own `.gitignore` if the optional `pathspec` package is installed (`pip install pathspec --break-system-packages`). To narrow this per project, set in `.mcp.json`'s `codebase-indexer` env block:

- `INDEX_INCLUDE_EXTENSIONS`: comma-separated extensions (e.g. `".py,.md"`) — overrides the built-in list entirely, so only these get ingested.
- `INDEX_EXCLUDE_DIRS`: comma-separated folder names (e.g. `"fixtures,generated"`) — adds to (not replaces) the built-in excludes.

Same options exist as `--include-ext`/`--exclude-dirs`/`--no-gitignore` flags on `ingest_to_qdrant.py`, and as `include_extensions`/`exclude_dirs`/`respect_gitignore` parameters on the `index_repo`/`sync_repo`/`preview_index` tools if you want to override the env defaults for a one-off run. Use `preview_index` first to confirm the filtering is doing what you expect before running a real index.

**3b. If also using local-compress**, add this server to the same `.mcp.json`:

```json
"local-compress": {
  "command": "python",
  "type": "stdio",
  "args": ["/absolute/path/to/compress_mcp_server.py"],
  "env": {
    "LMSTUDIO_BASE_URL": "http://localhost:1234/v1",
    "LMSTUDIO_MODEL": "<exact model id loaded in LM Studio, or omit to auto-detect if only one model is loaded>"
  }
}
```

**4. Add the relevant section(s) from `templates/CLAUDE.md.template` to each project's `CLAUDE.md`** — only include a section for a tool that's actually configured in that project's `.mcp.json`.

**5. Initial index** (first time only, per project):

```bash
python ingest_to_qdrant.py --repo-path /path/to/project --collection <project-collection-name> --dry-run
# check the preview, then run for real:
python ingest_to_qdrant.py --repo-path /path/to/project --collection <project-collection-name>
```

Or ask Claude to run it via the `index_repo` tool once `.mcp.json` is set up.

**6. Ongoing use:** ask Claude to run `sync_repo` at the start of a session, or add it as a standing instruction in `CLAUDE.md` (see `templates/CLAUDE.md.template`). It only re-embeds changed files, so it's cheap to run routinely.

## Verifying it's working

- Call `get_collection_info` to confirm the collection has a nonzero point count.
- Ask a conceptual question about the codebase and check the Claude Code transcript (`~/.claude/projects/<project-hash>/<session-id>.jsonl`) for whether it called `qdrant-find` before `Grep`.
- Compare `/cost` on a conceptual question before and after indexing.

## Skills and hooks

`CLAUDE.md` guidance only helps for open-ended prompts. If a project uses custom skills (slash commands like `/load-context` or `/rr-plan`) that have their own step-by-step instructions, those dominate — a skill's explicit steps are followed over CLAUDE.md's general guidance whenever they compete, because CLAUDE.md content stays in context throughout the session but isn't more directive than a skill's own prescribed steps. To get `qdrant-find`/`compress_command_output` used inside a specific skill's workflow, edit that skill's own `SKILL.md` to reference the tool explicitly — there's no shared instructions folder that skills automatically inherit beyond CLAUDE.md itself.

For deterministic enforcement instead of prose guidance, use a hook. `settings.json.template` + `hooks/compress_bash_output.py` register a `PostToolUse` hook on `Bash`, `Grep`, `Glob`, `WebFetch`, and `WebSearch` that compresses ANY of their output once it exceeds a size threshold (`HOOK_COMPRESS_THRESHOLD_CHARS`, default 2000 chars) — not a hardcoded list of commands. This works because `PostToolUse` fires after the tool has already run, when the real output size is known, and can rewrite what Claude actually sees via `updatedToolOutput` before it ever reaches Claude's context. Unlike CLAUDE.md, this applies regardless of what skill (if any) is active, since hooks fire on tool events rather than being tied to a particular skill. To install: nothing needs to be copied into the target repo — merge `templates/settings.json.template`'s hooks block into that repo's `.claude/settings.json`, pointing the `args` path at `hooks/compress_bash_output.py`'s real location in the tools repo you cloned in step 2 (same pattern as `.mcp.json` referencing `ingest_mcp_server.py` by absolute path). The script finds `local_compress_lib.py` automatically since it lives in `libs/` under the tools repo root; set `TOOLS_REPO_DIR` only if you've moved the files to a different location.

The dividing line for what's in the matcher isn't "runs locally" (WebFetch/WebSearch actually run through Anthropic's own infrastructure, not local compute) — it's whether Claude ever uses that tool's raw output as the exact, literal basis for a following `Edit`. Bash/Grep/Glob/WebFetch/WebSearch output is read for gist or lookup, never edited from directly, so lossy compression is safe there.

Bash's `tool_response` shape (`{stdout, stderr}`) is documented and tested directly. The others are not — Claude Code types `tool_response` as `Any`/`unknown` in its hooks docs, so the script doesn't guess field names. Instead it recursively finds every string value in the response, and if the large ones (≥200 chars, to skip metadata like a URL or file count) sum past the threshold, it compresses their concatenation and writes the result back into the single largest field, blanking the other large fields and leaving small ones untouched. This keeps the original JSON shape intact regardless of what the real schema turns out to be. Set `HOOK_DEBUG_LOG` to a file path to see the raw payload for any tool that passes through the hook, useful if you add another tool to the matcher and want to confirm what its `tool_response` actually looks like.

In practice, WebFetch and WebSearch are usually no-ops: both already run their own extraction server-side on Anthropic's infrastructure before Claude sees the result (confirmed empirically — a 2MB fetched Wikipedia page came back as a ~1300 char summary), so there's rarely anything left worth compressing further. Glob is also usually a no-op for a different reason: a large file listing is many *short* strings (individual paths), and the 200-char per-field floor means the aggregate size never triggers compression — deliberately, since file paths are exact identifiers a follow-up `Read`/`Edit`/`Grep` needs verbatim, so summarizing a file listing away would risk the same class of problem `Read`'s exclusion avoids.

`Read` (and `Write`/`Edit`/`NotebookEdit`) are deliberately excluded from the matcher: their output is often the exact basis for a following `Edit`, and a silent lossy rewrite there risks Claude editing from compressed content — a worse failure than a compressed log.

This hook fails OPEN, not closed: if LM Studio is unreachable or the request otherwise fails, the original output is left completely unchanged and a short note is attached via `additionalContext` so it doesn't silently degrade forever unnoticed. This is different from the harder-line "fail loud" stance used elsewhere in this project (e.g. refusing to guess an ambiguous model) — `PostToolUse` can't block the tool call anyway since the command already ran, so discarding real output on a compression failure would be strictly worse than leaving it alone.

An older, narrower `PreToolUse` approach (`hooks/redirect-to-compress.sh`) is kept for reference — it hard-blocks a hardcoded list of commands (`dotnet test`/`build`) before they run, forcing an explicit retry via `compress_command_output`. It's superseded by the `PostToolUse` approach above for the general case, but the tradeoff is real: a hard `deny` is more forceful (guarantees the raw output is never even generated) at the cost of breaking those specific commands entirely if LM Studio isn't running, versus the `PostToolUse` hook which always lets the command run and only fails to compress its output.

Hooks only work well here because "how large was this output" is measurable after the fact. "Should this have been a `qdrant-find` instead of a `Grep`" has no equivalent measurable signal to hook on — that gap still has to be closed at the skill or CLAUDE.md level, not via hooks. Not every "which tool should this have been" question is like that, though — see the WebFetch/`fetch_url` case below, which DOES have a usable signal (the tool name plus whether local infra is reachable), unlike qdrant-find/Grep which has no equivalent way to know in advance which one is "correct" for a given question.

### fetch_url vs. WebFetch vs. the hooks

There are now three overlapping ways a URL's content can end up compressed, worth being explicit about:

- **`fetch_url`** (in `compress_mcp_server.py`) fetches the page and runs the whole extraction/summarization step on your LOCAL LM Studio model, steerable via `focus`. This is the one that actually shifts the "review the page" work off Anthropic's infrastructure onto local compute — it's the closest to the local-compute cost-savings goal this whole project is built around.
- **WebFetch** (built-in) fetches the page and summarizes it using an Anthropic-hosted model, before Claude ever sees the result. Whether that summarization step itself counts against your token budget is NOT documented anywhere (checked `code.claude.com/docs/en/costs` directly), and real testing suggests it may be unanswerable from `/usage` alone regardless (see `EVALUATION.md`'s "Measuring cleanly" section) — not load-bearing for evaluating `fetch_url` itself, since `EVALUATION.md`'s Track B compares total session cost directly instead. Either way, the step runs on Anthropic's infrastructure, not local compute.
- **The `PostToolUse` hook** on WebFetch is a safety net on top of WebFetch, not a replacement for it — it only fires if WebFetch's own (Anthropic-side) summary somehow still comes back over the threshold, which empirically is rare.

Prefer `fetch_url` when you want the local model to do the reviewing (matches this project's actual goal). Prefer WebFetch directly for anything needing auth, JS rendering, or session/cookie handling — `fetch_url` is a plain unauthenticated GET and won't work for those.

Since CLAUDE.md's "prefer fetch_url" guidance is prose Claude can deprioritize (the same qdrant-find-vs-Grep problem noted above), `templates/settings.json.template` also registers a `PreToolUse` hook (`hooks/redirect_webfetch_to_fetch_url.py`) that enforces it deterministically: it hard-denies any `WebFetch` call and tells Claude to retry with `fetch_url`, but only when LM Studio is reachable right now (checked live, not cached) — if it's down, the hook fails open and lets WebFetch through normally, so a stopped local model never makes WebFetch unusable. This works as a hard `deny` (unlike the size-based `PostToolUse` hook above) because, unlike output size, "should this go to fetch_url instead" is knowable *before* the call — it only depends on the tool name and whether the local model is currently up, not on anything only known after the fact.

The deny reason's exact wording matters more than it might seem: an earlier version phrased it as a suggestion ("use fetch_url instead... if that captures what you're looking for"), and in real testing Claude treated that denial cautiously — it stopped and asked the user for permission to switch tools rather than just proceeding. That's the same "block enforced, retry not enforced" gap as prose guidance in general: the hook deterministically stopped WebFetch, but what Claude did *next* was still just its own judgment call. Rewording the reason to be explicitly imperative ("call fetch_url now — do NOT ask the user for confirmation first") fixed this in testing; Claude now proceeds automatically and only stops to ask if `fetch_url` itself then errors.

The known gap: this can't tell in advance whether a URL needs authentication, JavaScript rendering, or session/cookie handling — cases where `fetch_url`'s plain GET will fail and WebFetch is genuinely required. The deny reason tells Claude to fall back and inform you if that happens, but the hook will keep denying further WebFetch retries to that same URL rather than learning from the failure. If this matters for your usage, either drop `WebFetch` from that hook's matcher, or change its `permissionDecision` from `deny` to `ask` so you can approve WebFetch per-call instead of it being fully automatic.

## Known limitations

- Claude Code doesn't currently render MCP progress notifications visibly in the UI (open issue upstream), though `index_repo`/`sync_repo` still report progress internally to avoid call timeouts on large repos.
- `codebase-indexer` and the `qdrant-find`/`qdrant-store` server are independent MCP servers that both need to agree on `QDRANT_URL`/`COLLECTION_NAME`/`EMBEDDING_MODEL` — the `.mcp.json` template keeps them in sync via shared env vars; don't edit one without the other.
- Changing `EMBEDDING_MODEL` after a collection has data requires a full `index_repo --reset` rebuild — old and new vectors aren't compatible.
- `compress_file`/`compress_command_output`/`fetch_url` have been validated against real sources across multiple rounds of testing; `compress_text` and `list_local_models` are simpler and less exercised. `compress_text` only saves tokens for content you already legitimately hold (e.g. your own draft) — if you had to Read or run something into context first just to pass it to `compress_text`, that defeats the point, since the raw content already cost tokens to get there. Use `compress_file`/`compress_command_output` instead for files/commands, since those read the source server-side and never round-trip the raw content through Claude at all.
- Compression is lossy by nature — don't use either compress tool on source code you intend to edit, or on output you need verbatim (something to parse, a diff to apply, an exact string to grep for next).
- **Unverified assumption underlying `fetch_url`/the WebFetch redirect hook**: whether Claude Code's built-in `WebFetch` tool's own internal extraction/summarization step counts against your token budget is not documented anywhere (checked `code.claude.com/docs/en/costs` directly — no mention), and real testing suggests this specific question may be unanswerable from `/usage` alone regardless (only one model ever appeared in the "Usage by model" breakdown across testing, suggesting any internal step isn't billed against your visible account usage — see `EVALUATION.md`'s "Measuring cleanly" section). This doesn't block evaluating `fetch_url` itself, though: `EVALUATION.md`'s Track B compares total session cost for `fetch_url` vs. `WebFetch` directly on equivalent tasks, which answers "which costs less in practice" without needing to know why.
- `focus` on `compress_file`/`compress_command_output`/`fetch_url` handles positional asks ("the lead section," "the first N lines," "the abstract") automatically — recognized phrasings get truncated to the document's actual beginning (via structural section-heading detection where possible, a fixed-size fallback otherwise) before any classification runs, verified end-to-end against a real Wikipedia article and a real local model with no retries needed. This took several rounds of real-world testing to get right — see `libs/local_compress_lib.py`'s docstrings (`_find_first_heading_boundary`, `classify_relevant`, `compress`) for the specific failures found and fixed along the way, useful reading if you're extending this logic. Two residual limits worth knowing: the heading-detection heuristic only helps for content with clear heading-like structure (falls back to a fixed ~4000-char guess for plain text with no heading structure, e.g. a log file); and `_POSITIONAL_FOCUS_HINTS` is a fixed keyword list — a positional phrasing outside that list falls through to the position-aware classifier instead, which real testing showed isn't fully reliable on its own.
