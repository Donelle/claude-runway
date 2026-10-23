# Development workflow for this repo

Separately from the `skills/` directory (see the [Files](../README.md#files) table — it ships product features that get installed into *other* projects), this repo also carries its own contributor-facing skills in `.claude/skills/` — project-scoped, so they only work inside claude-runway itself, and hardcode this repo's own coordinates (`Donelle/claude-runway`) rather than trying to be generic. They're the recommended way to work an issue end to end, and are the ideal shape for this project's own development cycle.

| Skill | Invocation | What it does |
|---|---|---|
| `my-gh-triage` | `/my-gh-triage <paste the error, symptom, or question>` | For a reported symptom rather than a ready ticket. Runs four gates in order: investigate read-only (source, intent, whether it works, `file:line` evidence, parity across checkouts), state one verdict and one recommended fix, file a ticket via `/my-gh-create-issue` only when there is a concrete change to make (after a duplicate search), then work it now via `/my-gh-code-it` only if it is P0/P1 or small and low risk, unassigned, and needs no design call. |
| `my-gh-create-issue` | `/my-gh-create-issue type=bug theme=theme-savings-tracker priority=priority-p1 <description>` | Investigates a described bug/idea first (reproduces it or confirms it by reading the actual code — never files a claim that wasn't checked), then files a GitHub issue in the exact format the existing backlog uses: a **Location**/**Suggested fix** (bug) or **Sanity-checked**/**Proposal** (enhancement) body, labeled by type (`bug`/`enhancement`/`documentation`), `priority-p{0-3}`, and `theme-*`. Missing `type`/`theme`/`priority` get inferred and confirmed rather than guessed silently. |
| `my-gh-code-it` | `/my-gh-code-it <issue-number>` | Fetches the GitHub issue, sets its type (`Bug`/`Feature`/`Task`, from labels) and assignee, creates a **native GitHub linked branch** (`gh issue develop`) following this repo's `fix/`/`feature/` naming convention, then plans the fix (seeded from the issue's own **Location**/**Suggested fix** sections) and waits for approval before writing any code. |
| `my-gh-pr` | `/my-gh-pr <issue-number>` | Opens the PR (`Fixes #<issue-number>` in the body, so merging auto-closes the issue) and posts research/plan/validate artifacts as **comments on the issue** — the closest GitHub-native equivalent to attaching RPIV artifacts to a ticket, since GitHub has no generic file-attachment API for issues. |
| `my-gh-pr-feedback` | `/my-gh-pr-feedback <pr-number>` | Fetches and categorizes PR review feedback (blocking/suggestion/question/nitpick/praise), separates human from bot/Copilot comments, and enforces verifying each finding — reproduce it or explain specifically why not — before fixing or replying. Replies never tag `@copilot` — doing so reliably triggers a broken GitHub coding-agent invocation on this repo, confirmed repeatedly; Copilot's automated review re-runs on its own after every push regardless. |
| `my-gh-autowork` | `/my-gh-autowork [issue-number\|issue-number issue-number ...\|all]` | Fully autonomous version of the cycle below: delegates each ticket to a subagent that runs plan → code → PR → verified review-feedback loop → merge with no approval gates, stopping and reporting instead of guessing on anything genuinely ambiguous. For use once a human has explicitly authorized working the backlog without per-step check-ins — it packages the other four skills' procedures (plus the merge step none of them do) into one delegated pass per ticket. A specific list of ticket numbers works through exactly those, in the order given, one at a time — stopping immediately on the first `BLOCKED`/`FAILED` rather than skipping ahead to the next one in the list. A ticket stopped `BLOCKED` at the 5-round review cap can be finished afterwards by a follow-up subagent on the PR's own branch (3-round cap), but only once a human authorizes it — see the skill's "Continuing after a BLOCKED at the round cap" section. |
| `my-gh-load-context` | `/my-gh-load-context` | Loads full project context by reading `README.md`, `docs/`, `EVALUATION.md`, and — when present locally (all three are gitignored, so a fresh checkout won't have them) — `CLAUDE.md`, `.plans/` (the plan, research and validate files of tickets being worked) and `.ai-plans/` (the backlog work order, issue drafts and standalone notes). Checks what it read against the repo's actual current state (skill inventories, `EVALUATION.md` track status, `.plans/` vs. issue closure, a stale work order, drafts already filed) and reports mismatches as a TODO instead of trusting the docs, then proposes resuming any clear in-progress work or asks what's next. |

`tools/poll_copilot_review.py` is an internal helper of `/my-gh-autowork`, not a user-facing command. It waits (up to about three minutes per call) for Copilot's review of one exact commit on a PR and prints `review:<id>` lines or `NOT_YET`; the autowork subagent prompt calls it, so you never run it by hand and it needs no other documentation.

## Developer setup (running from a clone)

### macOS and Linux

From a shell in the repository, run the bootstrap script:

```bash
./scripts/bootstrap.sh
```

It checks for Python 3.12 (through `uv` or a `python3.12` on `PATH`), creates `.venv` if needed, installs `requirements.txt` and `requirements-dev.txt` plus this checkout's command-line tools, runs `git config core.hooksPath .githooks` (git doesn't let a repo set this for itself, so each clone opts in once), writes the shell-side `CLAUDE_RUNWAY_*` variables to `~/.claude/claude-runway/env.sh`, and runs the doctor check. Re-running it is safe: it won't touch a correct `.venv` or an already-set hooks path, and it stops with a message, rather than deleting anything, if an existing `.venv` is the wrong Python version or layout.

The environment file comes from `.mcp.json` where it declares a value, and from defaults otherwise; see [Generating the shell side](environment-variables.md#generating-the-shell-side) for the rules. The script rewrites that file on every run, so put your own settings in `~/.claude/claude-runway/env.local.sh`, which it loads last and never touches. The script then asks `Add the claude-runway environment to ~/.zshrc? [y/N]` and appends one marked block that loads the file only if you answer `y`. `--update-profile` skips the question, and a run with no terminal writes nothing and prints the block. `./scripts/bootstrap.sh --env-only` refreshes just that file and runs the doctor check, for after editing `.mcp.json` (it needs the `.venv` a full run created). On Linux the script prints the block rather than choosing a profile.

Manual fallback:

```bash
# With uv:
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --index-url https://pypi.org/simple
uv pip install --python .venv/bin/python -r requirements-dev.txt --index-url https://pypi.org/simple
uv pip install --python .venv/bin/python --no-deps --no-build-isolation .

# Without uv: python3.12 -m venv .venv, then the same installs through .venv/bin/python -m pip.

git config core.hooksPath .githooks
.venv/bin/claude-runway-doctor .
```

**Verification:** the install steps were recorded on Linux only. The environment-file step and the profile question were run on macOS (Apple Silicon, zsh, the system `/bin/bash` 3.2.57) against a scratch `HOME`: a full run, `--env-only`, a re-run that changes nothing, `.mcp.json` edits picked up, and the prompt on a real terminal. Intel macOS and Linux have not been checked. The unit tests exercise the embedded Python and run `--env-only` under every `bash` on the machine, with a fake `HOME`, so they never touch your real profile.

### Windows

From a Windows PowerShell prompt in the repository, run the bootstrap script:

```powershell
.\scripts\bootstrap.ps1
```

It checks for Python 3.12 (through `uv` or the `py -3.12` launcher), creates `.venv` if needed, installs `requirements.txt` and `requirements-dev.txt` plus this checkout's command-line tools, enables the repository's pre-push hook, and runs the doctor check. Re-running it is safe.

**Windows verification (2026-10-04):** On Windows 11 Pro 10.0.26200, a fresh disposable clone completed setup and a second run left its installed package inventory and `.git/config` unchanged; `git config --get core.hooksPath` returned `.githooks`. The fresh-clone and repeat runs used PowerShell 7.6.6. The configured clone also completed a further run under Windows PowerShell 5.1.

PowerShell enforces execution policy before loading a script, so a blocked bootstrap cannot print its own recovery hint. If PowerShell reports that script execution is disabled, run this command in the same PowerShell session, then retry the bootstrap. The bootstrap itself does not change execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\scripts\bootstrap.ps1
```

Manual fallback:

```powershell
# With uv:
uv venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt --index-url https://pypi.org/simple
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt --index-url https://pypi.org/simple
uv pip install --python .venv\Scripts\python.exe --no-deps --no-build-isolation .
```

Without `uv`, use `py -3.12 -m venv .venv`, then install the requirements and checkout with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation .
```

After either path, enable the hook and run the final check:

```powershell
git config core.hooksPath .githooks
.\.venv\Scripts\claude-runway-doctor.exe .
```

**Recommended cycle:** `/my-gh-triage` when starting from a symptom instead of a known change → `/my-gh-create-issue` to file new work in the same verified, labeled format as the existing backlog → pick an open issue → `/my-gh-code-it {N}` to implement it → `/my-gh-pr {N}` to open the PR and document it against the issue → `/my-gh-pr-feedback` to work through review comments as they come in. This is exactly how the code-review backlog filed against this repo (see `.plans/` locally, or the repo's GitHub issues) is meant to be worked one item at a time — and how new items should keep getting added to it. `/my-gh-autowork` runs this same cycle without stopping between steps, once told to.

**A note on issue and PR numbers:** no tracked file cites issue or PR numbers: not the docs, and not code comments, docstrings, templates or config either. `tests/test_docs_no_ticket_refs.py` fails if one is added. State the fact and the reason directly instead, so the text stands alone. Issue and PR numbers belong in commit messages and PR descriptions, where they resolve.

**Linting and type checking:** `pyproject.toml`'s `[tool.ruff.lint]` and `[tool.pyright]` sections configure ruff (`select = ["F", "B"]`, pyflakes + flake8-bugbear) and Pyright (`standard` mode) for `libs`/`tools`/`hooks`(/`tests` for ruff). Both are dev-only tools for CONTRIBUTING to claude-runway, not something an end user installing the toolkit into their own project needs — so they live in `requirements-dev.txt`, a separate file from the user-facing `requirements.txt`, installed the same way (`uv pip install -r requirements-dev.txt --index-url https://pypi.org/simple`) into the same venv. Resolve the venv's Python interpreter once — `uv venv` creates `.venv/Scripts/` on a Windows checkout and `.venv/bin/` on Linux/macOS, never both, so a hardcoded `.venv/bin/...` path fails with command-not-found on Windows — then run both checks through it: `PYTHON=$(if [ -f .venv/Scripts/python ]; then echo .venv/Scripts/python; else echo .venv/bin/python; fi)`, then `$PYTHON -m ruff check libs tools hooks tests` and `$PYTHON -m pyright --pythonpath "$PYTHON"`. `/my-gh-code-it`'s test-running step runs both alongside the unit test suite, so all three must be clean before a PR. Pyright replaced mypy specifically because it's also what VS Code's Pylance extension runs under the hood, so the editor and this repo's gates now report the same findings instead of two type checkers disagreeing. **The `--pythonpath "$PYTHON"` flag is required, not optional**: `pyproject.toml`'s `[tool.pyright]` deliberately has no `venvPath`/`venv` setting (a hardcoded `.venv` there broke CI, which never creates one — see that section's own comment), so without `--pythonpath` pointing Pyright at the interpreter that actually has the dependencies installed, it silently falls back to whatever Python is first on `PATH` and reports every third-party import as missing — confirmed live, this happens even via `$PYTHON -m pyright` run from inside the venv, not just the bare `pyright` case one might expect.

**Pre-push hook:** `.githooks/pre-push` runs the same ruff + Pyright checks (not the unit test suite — see the hook's own comment for why) before a push reaches the remote. It's opt-in per clone, not automatic:

```bash
git config core.hooksPath .githooks
```

This is a convenience, not the real enforcement — it only runs in a clone that has opted in with the command above, and any push can skip it with `git push --no-verify`. Linked worktrees of an opted-in clone, including `/my-gh-autowork`'s, inherit the setting, because `core.hooksPath` lives in the repo-local config that every worktree shares. CI (`.github/workflows/tests.yml`) runs the identical two checks and is what actually gates a PR.
