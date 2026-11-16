# ClaudeRunway

Local-compute token savings for Claude Code.

Built on one idea: push work onto local compute (a local vector DB, a local LLM) instead of Claude's own context, so Claude only pays tokens for a distilled result rather than raw data.

1. **Qdrant codebase memory** — per-project semantic memory over code/docs, so Claude can retrieve relevant chunks instead of grepping and reading whole files, and store any text in that same collection to return later.
2. **Local compression (LM Studio)** — a local model summarizes large, mechanical content (logs, build output, big diffs) server-side before anything reaches Claude's context.
3. **Memory bank** — durable cross-session lessons and decisions, stored in one Qdrant collection shared by every project.
4. **Session continuity** — save a conversation to Qdrant (compressed on LM Studio when it is running) and restore it later.
5. **Savings tracker** — opt-in and a shared metrics store report what compression avoided.

Pieces 1 and 2 are independent: you don't need LM Studio to use the Qdrant memory, or vice versa. Pieces 3 and 4 build on the Qdrant server, and piece 5 reports on piece 2's compression.

## Table of contents

- [Known limitations](#known-limitations)
- [All known limitations](docs/known-limitations.md)
- [Prerequisites](docs/prerequisites.md)
  - [Windows GPU setup for LM Studio](docs/windows-setup.md)
- [Local workstation and model guide](docs/workstation-guide.md)
- [Installation](docs/installation.md)
- [Environment variables](docs/environment-variables.md)
- [Verifying it's working](docs/verifying-its-working.md)
- [Cross-repo lookups](docs/cross-repo-lookups.md)
- [Memory bank](docs/memory-bank.md)
- [Codebase reviewer](docs/codebase-reviewer.md)
- [Skills and hooks](docs/skills-and-hooks.md)
  - [Exactness-critical commands (Bash only)](docs/skills-and-hooks.md#exactness-critical-commands-bash-only)
  - [fetch_url vs. WebFetch vs. the hooks](docs/skills-and-hooks.md#fetch_url-vs-webfetch-vs-the-hooks)
- [Session continuity skills](docs/session-continuity.md)
- [Savings tracker](docs/savings-tracker.md)
- [Shared metrics store](docs/metrics.md)
- [Development workflow for this repo](docs/development-workflow.md)
- [Thesis: Measurements and Tracks](EVALUATION.md)
- [Files](docs/files.md)

## Known limitations

The limits that most often decide whether this toolkit fits your setup; [all of them](docs/known-limitations.md) are on their own page.

- `codebase-indexer` alone decides the project collection (`COLLECTION_NAME`/`EMBEDDING_MODEL`) for indexing, `cb_indexer_find_repo` and `cb_indexer_store_repo`. The `compact` block does not carry them (its server reads neither); `upgrade` removes them from an older one. `memory-bank`'s `EMBEDDING_MODEL` is deliberately separate.
- Compression is lossy by nature: don't use the compress tools on source code you intend to edit, or on output you need verbatim (something to parse, a diff to apply, an exact string to grep for next).
- **The savings tracker (`/cr-savings`) is an online observation, not a benchmark.** It reports exact characters before and after compression, but it observes the counterfactual rather than measuring it via a controlled A/B run (that's what `EVALUATION.md`'s tracks are for). Treat what the content would have cost in tokens as unknown beyond the opt-in, derived per-model figure.
- **`CLAUDE_RUNWAY_TRACK_SAVINGS` needs setting in two places, not one** (`.mcp.json` for the MCP server, a shell export for the hooks, which have no `env` field). Any variable read by both halves of the toolkit has the same rule: see [Environment variables](docs/environment-variables.md) and [Savings tracker](docs/savings-tracker.md).
- **The shared `memory-bank` collection locks in whichever `EMBEDDING_MODEL` created it first**, so every project sharing it must agree on one value. It is the `memory-bank` block's own `EMBEDDING_MODEL`, separate from the code-index model.

## LICENSE

In summary, Have Fun & Share! For specifics see [link](LICENSE)

## Questions?

If you have any questions or comments please feel free to drop me a line :-).

Email: <donellesanders@gmail.com>
