---
applyTo: "**/*.py"
---

# Code review focus for this repository

This repo is a token-savings toolkit for Claude Code (MCP servers + hooks). Reviews on
Python files should stay narrowly focused on correctness and safety — not style.

## Only flag disaster-level issues

Comment ONLY on things that would actually break something:

- A code path that will **crash, raise uncaught, or hang** in a case that can realistically
  occur (not a purely theoretical/impossible input).
- **Data loss or corruption** — silently dropping events, writing to the wrong location,
  overwriting data that should be preserved.
- **Security issues** — injection (shell/SQL), unsafe deserialization, credential/secret
  leakage, path traversal.
- **Race conditions** in code that touches shared state (files, the SQLite DB) from
  multiple processes.
- **Internally inconsistent output** — e.g. a displayed count/percentage/total computed
  from a different subset of data than a related figure shown right next to it, so the
  two numbers can't both be true at once.
- **Stale or actively misleading comments/docstrings** — a comment referencing a variable,
  function, or behavior that no longer exists or is wrong, since it will mislead the next
  person who reads it while debugging.

Do NOT comment on:

- Pure style, formatting, naming, or line-length preferences.
- Missing type hints, docstrings, or tests — this repo does not enforce those uniformly.
- "Best practice" suggestions with no concrete failure mode behind them (e.g. suggesting
  a refactor "for clarity" with no bug attached).
- Suggesting a broad `except Exception` be narrowed to a specific exception type **inside
  a hook script's top-level handler** — see below, this is intentional here.

If you're not sure whether something clears the bar above, don't comment on it.

## Basic Python standards actually worth enforcing

- Always specify `encoding="utf-8"` explicitly on `open()` calls — don't rely on the
  platform/locale default.
- Never use a bare `except:` — always catch `Exception` at minimum, with a comment
  explaining why the catch is broad if it's not obviously narrow.
- Never use a mutable default argument (`def f(x=[])`).
- A documented contract (e.g. "must be an absolute path", "returns non-negative") should
  either be enforced in code or the code should degrade safely if it isn't — flag it if
  neither is true.
- Don't leave an unclosed resource (file handle, DB connection) on a path that can return
  or raise before cleanup runs.

## Project-specific patterns that are intentional, not bugs

- **Hooks (`hooks/*.py`) deliberately fail open.** A broad `except Exception: pass` or
  `sys.exit(0)` in a hook's top-level error handling is intentional: a hook must never
  crash or block normal Bash/Grep/tool use just because an optional feature (compression,
  the savings tracker) hit an error. Don't flag this pattern as a bug or suggest narrowing
  it, unless the `except` is swallowing an error that should have surfaced via
  `additionalContext`/`systemMessage` instead of silently doing nothing.
- **The savings tracker estimates a counterfactual, not a measurement.** Token counts
  (`estimate_tokens`, chars ÷ 3.5) are an intentional approximation, not a bug — don't
  suggest replacing it with an exact tokenizer.
- **`libs/savings_ledger.py` is the sole storage interface** for the savings tracker by
  design (SQLite today, swappable later) — don't suggest inlining `sqlite3` calls
  elsewhere for "simplicity."
