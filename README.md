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

- [Known limitations](docs/known-limitations.md)
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

## LICENSE

In summary, Have Fun & Share! For specifics see [link](LICENSE)

## Questions?

If you have any questions or comments please feel free to drop me a line :-).

Email: <donellesanders@gmail.com>
