# Skill: cr-load-context

Load this project's context at the start of a session, fast. The Qdrant index is synced in the background while the docs are read, then the skill reports what the project is, what was recalled about it, which pull requests are in play, and which of the toolkit's tools are live.

It replaces the prose in a project's `CLAUDE.md` that asks Claude to run `cb_indexer_sync_repo` first, check the index and recall memories, and it reads the project instead of guessing at it. It does not check configuration (that is `/cr-doctor`) and it creates no files.

## Prerequisites

- `git`.
- claude-runway's own MCP tools, configured in the project: `compact`, `codebase-indexer`, `local-compress` and `memory-bank`. Run `/cr-setup` if they are not.

Nothing else is required. `gh` (GitHub) and `az` (Azure DevOps, with the `azure-devops` extension) are optional extras: when one is missing, not signed in, slow or failing, only its own section of the report says `skipped: <reason>`, and every other section is unaffected. No step below may stop because of them.

## Usage

Run `/cr-load-context` from the root of a project. It loads that project (the current working directory). It takes no arguments.

## Steps

### 0. Locate the facts script

`facts.sh` sits next to this file. Claude Code prints a "Base directory for this skill: <path>" line when it loads a skill: the script is `<that path>/facts.sh`. If there is no such line, use `~/.claude/skills/cr-load-context/facts.sh`, then `skills/cr-load-context/facts.sh` in a claude-runway checkout. If none exists, skip the script and do the same by hand (step 1's facts bullet below).

### 1. Round 1: everything that needs no reading, in ONE message

Send all of these as parallel tool calls in a single message:

- **`cb_indexer_sync_repo` in the background, only when the project already has an index.** Start a subagent with the Agent tool (`subagent_type: general-purpose`, `model: haiku`) and this prompt: "First check, with Bash `test -f`, whether `<the project's absolute path>/.qdrant_index_manifest.json` exists. If it does not, do NOT call `cb_indexer_sync_repo` (with no manifest it is a full rebuild, a long first embed that is the user's to ask for): reply exactly `no manifest: sync not started` and stop. If it does exist, call the `cb_indexer_sync_repo` tool of the `codebase-indexer` MCP server with `repo_path` set to `<the project's absolute path>` and no other arguments, and reply with only its result, in two lines at most. Do nothing else." Nothing waits on it; its result arrives later as a notification. If the Agent tool is not available to you (you are yourself a subagent), skip this bullet in the parallel message, and call `cb_indexer_sync_repo` yourself once the facts script has printed `sync manifest: present`; do not call it when it printed `ABSENT`. The one side effect of the whole skill is here: when it runs, `cb_indexer_sync_repo` writes `.qdrant_index_manifest.json` and updates this project's Qdrant collection. Say so in the report; when it did not run, say that instead.
- **The facts script**, through Bash, with `--porcelain` as a bare, unquoted flag (the compress hook treats that flag as "exact output, do not summarize"; a quoted `--porcelain` loses it):

  ```bash
  bash "<skill folder>/facts.sh" --porcelain
  ```

  It prints git state, the entry docs (README, `CLAUDE.md`, a `docs/` listing with each page's title), whether the sync manifest exists, and the forge's open pull requests (GitHub through `gh`, Azure DevOps through `az`). It never reads `.mcp.json` and never prints a credential. If bash is not available, get the same facts by hand: `git branch --show-current`, `git status --porcelain`, the origin URL with any credential removed, and `gh pr list --state open --limit 30` or `az repos pr list --status active` when that CLI exists. Remove the credential in the same command that reads the URL: `git remote get-url origin | sed -E 's#^([A-Za-z][A-Za-z0-9+.-]*://)[^/]*@#\1#'`, or in PowerShell `(git remote get-url origin) -replace '^([A-Za-z][A-Za-z0-9+.-]*://)[^/]*@','$1'`. Never run a plain `git remote get-url origin` or `git config remote.origin.url`: an HTTPS origin can carry a password or token, and the output goes straight into your context. Read files with Read.
- **`cb_indexer_get_collection_info`** (`codebase-indexer`): does the index have data?
- **`list_local_models`** (`local-compress`): is LM Studio up, and is a model loaded?
- **`recall`** (`memory-bank`): `query` set to the project's name plus "decisions, corrections and gotchas". It defaults to this project's memories plus general ones; leave `repo` and `all_repos` alone. Each hit's `summary` is what to report; its `description` is long, so read it only for a memory that bears on this session.

A tool that is not connected, or errors, is a status line in the report, not a reason to stop: carry on with the rest.

Never read or print `.mcp.json` yourself: an inline `QDRANT_API_KEY` would reach the model context. What is configured is learned from which tools above exist and answer.

### 2. Read what the project says to read

Follow the project's own `CLAUDE.md`: read it with Read, then read the docs it points at (and the README if it names nothing), only as far as the task at hand needs. Use the facts' `docs/` listing, which gives each page's title and length, to choose. Use Read for every file, never `cat`, `sed` or `head` through Bash: the compress hook paraphrases large Bash output, and a paraphrased doc is a silent wrong answer.

Then run one or two `cb_indexer_find_repo` queries instead of opening many files: one for the architecture and entry points ("where does the program start, what are the main components"), and one for whatever the user said they are about to work on, if they said. Skip this when `cb_indexer_get_collection_info` showed the index empty or missing; if the sync manifest is ABSENT, this project has not been indexed here, so the first embed is long and `cb_indexer_sync_repo` was not started in round 1: say so and suggest `cb_indexer_index_repo`, and do not start it silently.

### 3. Learn the live tools

A project may not have claude-runway's own `CLAUDE.md` guidance, so this step puts it in your head for the rest of the session. Take the entry below for each tool group that is **live**: a group is live when one of its tools exists and answered. Round 1 called `codebase-indexer`, `local-compress` and `memory-bank`; the other groups are not called here, so they are live when one of their tools is among yours (a tool shows up as `mcp__<group>__<tool>`). Skip the entry of a group that is missing or errored, and say nothing about it here (the report's status line covers it). This guide is for you: do not recite it in the report, and do not teach the user tool names, since users ask in natural language.

- **`compact`** (this project's saved conversation compacts)
  - `compact_store`, `compact_find` and `compact_prune` keep conversation compacts in a separate collection per project (they back `/cr-compact` and `/cr-resume`). `cb_indexer_find_repo` does not search them: use `compact_find`, by project and an optional query. `compact_prune` deletes compacts (keep the last N, or drop those older than N days) and defaults to a dry run: run it only when the user asks, and show what it would remove before a real run. Never put notes about the code in compacts: that is what `cb_indexer_store_repo` is for, and a compact stored as a note is buried under code chunks.
- **`codebase-indexer`** (code and docs indexed into Qdrant)
  - `cb_indexer_find_repo` is semantic search. Use it before Grep or Read for conceptual questions ("where is X handled", "how does Y work", "which files deal with Z"); use Grep for exact identifiers, error strings and known symbols. Its results are leads, not facts: when a hit's metadata has a `file_path` and a line number or `line_range`, Read only that span, not the whole file. It returns 10 hits unless you pass `limit`. If the same question comes up again, reuse the earlier result instead of searching again.
  - `cb_indexer_store_repo` saves a short piece of text so `cb_indexer_find_repo` can return it by meaning: store short text, never a raw file dump; add `file_path` plus a line range as metadata when the text is about a file, `source`/`url` when it came from elsewhere.
  - `cb_indexer_sync_repo` only re-embeds files that changed, so it is cheap: run it at the start of a session and again after a batch of edits. `cb_indexer_index_repo` is for a first-time index, or after changing chunking or embedding settings (`reset=true` only then). `cb_indexer_preview_index` shows how files would be chunked and filtered without writing anything: use it before `cb_indexer_index_repo`. When `cb_indexer_find_repo` comes back empty, call `cb_indexer_get_collection_info` to see whether the collection has any data.
  - `cb_indexer_find_in_collection` (a `query` and the other repo's `collection`), `cb_indexer_list_collections` and `cb_indexer_set_collection_description` are for the one case where the answer lives in a **different** already-indexed repo (say, the backend a frontend calls). Use an exact collection name the user gave you; if you do not know it, call `cb_indexer_list_collections` first, never guess. Do not use them for questions this project's own collection answers. `cb_indexer_set_collection_description` stores a short hint on a collection that helps pick the right one, and only when the user asks.
- **`memory-bank`** (durable memory that outlives the session, shared across the projects on this machine)
  - `recall` before finalizing any non-trivial change: a past correction, root cause or design rationale may already be stored. It defaults to this project's memories plus general ones (and anything the repos this project follows marked shared); pass `repo` or `all_repos` only when you need a different or wider scope, and a `repo` only if the user named it.
  - `remember` after a real correction, a non-obvious root cause, or a design decision with rationale, not after every exchange. Ask first ("Is this something you'd like to remember for next time?"). Phrase `summary` the way a future query would (it is the only text searched) and put the full detail in `description`. Set `general=True` only when the user explicitly says it applies to every project, and `shared=True` only when they say it is relevant to other repos (it offers the memory to the repos that follow this one; a relevance filter, not privacy); never infer either from the content, and the two cannot be combined.
  - `forget` deletes one memory by id, or with `wipe_all` clears this project's own memories. It needs the user's explicit confirmation before `confirm=True`, and you show them first what would be deleted (a `recall` result, or the dry-run count a call without `confirm` returns).
  - `list_memory_banks` is a read-only count of which projects have memories (no memory text). `copy_memories` copies another project's memories into this one (it always writes under this project's tag, so run it from the destination project's session); `transfer_memories` imports memories from a legacy Qdrant collection. Use these two only when the user asks and names the exact source (the other project's id, or the collection), never a guessed one; run them with the default `dry_run=True` first, show the user the report, and pass `dry_run=False` only after they agree.
  - It takes priority over Claude Code's own local memory files for lessons, corrections and decisions with rationale, so there is one source of truth; keep the local memory system for what loads automatically with no retrieval step (the user's profile and collaboration preferences).
- **`local-compress`** (a local LM Studio model reads and summarizes, so the raw text never enters your context; that is the saving)
  - `compress_command_output` instead of Bash for a command whose output is large and mostly mechanical (test suites, builds, verbose linters) when you want the gist or whether it failed. `compress_file` instead of Read for a large mechanical file on disk (saved logs, big diffs). `fetch_url` instead of WebFetch for a public page you want summarized locally; steer it with `focus`, including a positional one such as "summarize the lead section". Do not use `fetch_url` for a page that needs login, JavaScript or a session: use WebFetch. `compress_text` is only for text you already hold yourself (your own long draft); using it on something you just read saves nothing.
  - Never compress source code you are about to edit, or output you need verbatim (something to parse, a diff to apply, an exact error string to grep for next): a summary only tells you where to look. When it shows you something that needs exact text, Read that file or re-run a narrower command.
  - `list_local_models` says whether LM Studio is up and a model is loaded. When it is not, read and run things directly instead of waiting on it.
- **`analytics`** (opt-in: only present when the project turned the savings tracker on)
  - `savings_summary`, `savings_detail` and `savings_trend` are the simple, detail and trend views of what local compression saved (characters before and after); `/cr-savings` calls them. They say tracking is off, which is an explanation and not an error, when it is not enabled. They are an observation, not a benchmark.
  - `get_metrics` reads and `record_metric` writes the shared metrics store (`/cr-metrics` calls `get_metrics`). The `metric_id`s that have a writer are `autowork` (per-ticket outcomes from `/cr-gh-autowork`), `memory-bank` (a tally of `remember`/`recall`/`forget` calls), `codebase-reviewer` (one event per review run) and `pr-review-rounds` (Claude's cost per pull-request review round, recorded at session end only when `CLAUDE_RUNWAY_CAPTURE_PR_ROUNDS` is on); any other id returns "no events recorded yet", and savings are a separate store read through the savings tools. Record a metric only when the user asks for one: the reviewer, memory-bank, the compression server and the SessionEnd hook record their own events.
- **`codebase-reviewer`** (no longer set up by default, so present only in a project that configured it earlier)
  - `list_reviewers` checks the reviewers file and the stage plan and runs no model. `review` runs a local-model review of a change on LM Studio and returns only the findings, so the diff stays out of your context; its `type` is `"reasoning"` (unverified leads, not confirmed defects), `"verifier"` (adds grounding and verification; prefer it when verifier reviewers are configured) or `"classifier"` (adds the user's rubric levels), and `params={}` returns a type's keys and an example. `review_report` shows past review times and costs from recorded runs. Run `review` only when the user asks for one. The reviewers file is the user's: when a tool says it is missing or invalid, report that, never guess a path or write one yourself.

### 4. Report

One short report, in this order, and nothing past it:

1. **The project**: what it is, how it is laid out, where it starts (from the docs and the `cb_indexer_find_repo` results).
2. **Recalled**: what `memory-bank` returned that matters for this project; "nothing stored" if empty.
3. **In play**: the current branch (ahead/behind, uncommitted count), then the open pull requests, the one for this branch marked. The forge facts are pull requests only: do not write "no issues", because the skill does not look at issues or tickets. A forge that was skipped says `skipped: <reason>` here, from the facts script's `forge facts skipped:` line (for an Azure DevOps repo that `az` reports disabled, or `TF401019`, say the repo is disabled, removed or inaccessible).
4. **What is live** (status only; do not recite step 3's guide): which tool groups answered, and what it changes ("LM Studio is down, so read files directly", "the index is empty, so run `cb_indexer_index_repo`", "memory-bank is not configured"). When anything looks wrong, suggest `/cr-doctor`; do not run it (it needs `CLAUDE_RUNWAY_DIR` and the tools-repo venv, and checks static configuration only).
5. **The sync**: when the subagent replied `no manifest: sync not started` (or the facts said ABSENT), say that the project has no index yet and suggest `cb_indexer_index_repo`. Otherwise say that `cb_indexer_sync_repo` is running in the background (it updates the index and writes `.qdrant_index_manifest.json`) and give its result if the notification has already arrived; do not wait for it. If the result is an httpcore disconnect error on macOS, that is the Docker Desktop idle-connection limit in `docs/known-limitations.md`: running `cb_indexer_sync_repo` once more normally succeeds.
