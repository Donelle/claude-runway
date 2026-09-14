# Skill: my-savings

Show the opt-in ClaudeRunway savings tracker: an estimate of context tokens avoided by local compression this session, plus this project's history. This includes both the `local-compress` MCP server's tools (`compress_file`, `compress_command_output`, `fetch_url`) and the `compress_bash_output.py` hook's own compressions of Bash/Grep/Glob/WebFetch/WebSearch output — both contribute to the same session total.

Optional argument: `detail` (e.g. `/my-savings detail`), `trend` (e.g. `/my-savings trend`, `/my-savings trend day`). Omit it for the simple view.

**Important — this is an estimate, not a benchmark.** This tracker computes a live, online estimate of a counterfactual ("what would this have cost without compression"), not a controlled A/B measurement. `EVALUATION.md`'s tracks measure that rigorously by running a task twice; this skill reports what a heuristic estimator observed in real time. Don't present the two as equivalent.

## When to use

Run this any time you want to check how much local compression has saved this session, or how this project trends over time. Requires `CLAUDE_RUNWAY_TRACK_SAVINGS` (accepted values: `1`, `true`, or `yes`, case-insensitive) to be set in **both** `.mcp.json`'s `local-compress` env block **and** exported in the shell environment that launches `claude` (hook entries in `.claude/settings.json` have no `env` field of their own, so they only ever see what the shell already has) — if it's off, the tools will say so.

## Steps

1. **Determine the view**: if the argument is `detail`, use the detail view; if the argument is `trend` (optionally followed by `day` or `week`), use the trend view; otherwise use the simple view.

2. **Simple view** (no argument): call the `savings_summary` MCP tool (from the `local-compress` server) with no arguments — it defaults `project` to the current working directory's folder name. Print its return value to the user verbatim (it's pre-formatted text, not something to summarize or reformat).

3. **Detail view** (`detail` argument): call the `savings_detail` MCP tool the same way. Print its return value verbatim.

4. **Trend view** (`trend` argument): call the `savings_trend` MCP tool. If the user also supplied `day` or `week` after `trend`, pass that as the `bucket` parameter; otherwise the tool defaults to weekly buckets. Print its return value verbatim. Example: `/my-savings trend day` calls `savings_trend(bucket="day")`.

5. **If the tool reports tracking is off**: relay that message plainly — don't try to work around it or guess a number. Tell the user both places `CLAUDE_RUNWAY_TRACK_SAVINGS=1` needs to be set: their project's `.mcp.json` `local-compress` env block, and exported in the shell environment that launches `claude` (not `.claude/settings.json` — hook entries there have no `env` field of their own).

6. **If the tool reports no session data yet**: this is normal at the very start of a session, or in a project where nothing has been compressed yet. Don't treat it as an error — just relay what it says.

## Notes

- This is read-only — it never writes, modifies, or deletes anything. (The perpetual SQLite rollup happens automatically at session end via a separate hook, not through this skill.)
- Only local-compression events are credited toward the headline number. Qdrant/`qdrant-find` activity is intentionally never counted here — its counterfactual (what Grep+Read would have cost) isn't observable, so crediting it would be a guess dressed up as a measurement. See the repo's `EVALUATION.md`.
- `fetch_url` calls are logged but never credited — its honest counterfactual is WebFetch's own already-compressed summary, which isn't observed. The tool's output will show fetch_url call counts separately from the credited savings figure.
