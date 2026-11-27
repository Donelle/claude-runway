# Development workflow for this repo

Separately from the `skills/` directory (see the [Files](files.md) table — it ships product features that get installed into *other* projects), this repo also carries its own contributor-facing skills in `.claude/skills/` — project-scoped, so they only work inside claude-runway itself, and hardcode this repo's own coordinates (`Donelle/claude-runway`) rather than trying to be generic. They're the recommended way to work an issue end to end, and are the ideal shape for this project's own development cycle.

| Skill | Invocation | What it does |
|---|---|---|
| `cr-gh-triage` | `/cr-gh-triage <paste the error, symptom, or question>` | For a reported symptom rather than a ready ticket. Runs four gates in order: investigate read-only (source, intent, whether it works, `file:line` evidence, parity across checkouts), state one verdict and one recommended fix, file a ticket via `/cr-gh-create-issue` only when there is a concrete change to make (after a duplicate search), then work it now via `/cr-gh-code-it` only if it is P0/P1 or small and low risk, unassigned, and needs no design call. |
| `cr-gh-spike` | `/cr-gh-spike <issue-number>` or `/cr-gh-spike <the question, and the decision it feeds>` | For finding something out before committing to a change ("is X feasible", "which of these two approaches"). Every spike has a ticket of issue type `Spike`: given a number it reads the whole ticket and takes the framing from it, given a question it looks for an existing spike and otherwise offers to file one through `/cr-gh-create-issue`. Runs four gates in order: frame one answerable question (the decision it informs, what evidence settles it, what is out of scope, and a check of memory, the index and `.plans/` for prior work), investigate read-only (experiments only in a throwaway scratch directory or worktree, each fact marked verified or inferred, and a stop rule that returns to you when the question changes, a third line of inquiry opens, or the evidence is in hand), state one verdict (`go`, `go with changes`, `no-go` or `inconclusive`) with one recommendation, then write the findings to `.plans/{N}-spike.md` and post them **as a comment on the spike ticket** (checked first for credentials, home-folder paths and private names, since the repo is public, and split in two if over GitHub's comment limit). It then offers, never does, the follow-ups: a ticket for the change via `/cr-gh-create-issue`, closing the spike ticket, or a `remember`. It never leaves code behind and never starts the implementation. |
| `cr-gh-create-issue` | `/cr-gh-create-issue type=bug theme=theme-savings-tracker priority=priority-p1 <description>` | Investigates a described bug/idea first (reproduces it or confirms it by reading the actual code — never files a claim that wasn't checked), then files a GitHub issue with a short `## Summary` for people (about six lines) above a long `## AI Instructions` for the implementing agent, which holds the **Location**/**Suggested fix** (bug) or **Sanity-checked**/**Proposal** (enhancement) fields the existing backlog uses, labeled by type (`bug`/`enhancement`/`documentation`), `priority-p{0-3}`, and `theme-*`. Missing `type`/`theme`/`priority` get inferred and confirmed rather than guessed silently. `type=spike` files a spike ticket instead: a `Spike:` title, a spike-shaped body (the questions, what done looks like, what is out of scope; no verdict or fix), the priority and theme labels, and GitHub's `Spike` issue type, which it sets itself right after filing because `/cr-gh-code-it` never works a spike. |
| `cr-gh-code-it` | `/cr-gh-code-it <issue-number>` | Fetches the GitHub issue, sets its type (`Bug`/`Feature`/`Task`, from labels) and assignee, creates a **native GitHub linked branch** (`gh issue develop`) following this repo's `fix/`/`feature/` naming convention, then plans the fix (seeded from the issue's own **Location**/**Suggested fix** sections) and waits for approval before writing any code. It stops, before changing anything, on a `Spike` ticket (a question, not a change) and points at `/cr-gh-spike`. |
| `cr-gh-code-review` | `/cr-gh-code-review [issue-number]` | Reviews the branch before a PR exists. First reads the GitHub issue (and its parent's decisions) and `.plans/<issue-number>-plan.md` and checks that they agree: every requirement covered, no contradictions, every deviation recorded; a contradiction it cannot settle goes to you, not into a guess. Then runs the three gates (tests, ruff, Pyright) and has the `code-reviewer` agent (`.claude/agents/code-reviewer.md`: Opus, read-only) review the diff against that ticket-and-plan brief and this repo's own review rules (`.github/instructions/`), reporting alignment first and then critical/major/minor findings, each verified before it is presented. It reports and offers fixes; it never changes code on its own. It reviews a branch of this repository and is unrelated to the `codebase-reviewer` MCP tool. |
| `cr-gh-pr` | `/cr-gh-pr <issue-number>` | Opens the PR (`Fixes #<issue-number>` in the body, so merging auto-closes the issue) and posts research/plan/validate artifacts as **comments on the issue** — the closest GitHub-native equivalent to attaching RPIV artifacts to a ticket, since GitHub has no generic file-attachment API for issues. |
| `cr-gh-pr-feedback` | `/cr-gh-pr-feedback <pr-number>` | Handles the review feedback on a PR in two halves. The `pr-feedback` agent (`.claude/agents/pr-feedback.md`: Opus 5.5, high effort, read-only) fetches and categorizes it (blocking/suggestion/question/nitpick/praise), separates human from bot/Copilot comments, and reproduces each finding in a throwaway copy, returning a verdict for each. The skill, in the main session, re-checks what reproduces, then fixes it (one commit per pass, pushed only when you say so), answers what doesn't with the evidence, and resolves threads. Replies never tag `@copilot` — doing so reliably triggers a broken GitHub coding-agent invocation on this repo, confirmed repeatedly; Copilot's automated review re-runs on its own after every push regardless. |
| `cr-gh-autowork` | `/cr-gh-autowork [issue-number\|issue-number issue-number ...\|all]` | Fully autonomous version of the cycle below: delegates each ticket to a subagent (`.claude/agents/autowork-ticket.md`: Sonnet 5.5, max effort) that runs plan → code → PR → verified review-feedback loop → merge with no approval gates, stopping and reporting instead of guessing on anything genuinely ambiguous. For use once a human has explicitly authorized working the backlog without per-step check-ins — it packages the other four skills' procedures (plus the merge step none of them do) into one delegated pass per ticket. A specific list of ticket numbers works through exactly those, in the order given, one at a time — stopping immediately on the first `BLOCKED`/`FAILED` rather than skipping ahead to the next one in the list. A ticket stopped `BLOCKED` at the 5-round review cap can be finished afterwards by a follow-up subagent on the PR's own branch (3-round cap), but only once a human authorizes it — see the skill's "Continuing after a BLOCKED at the round cap" section. It never auto-picks a `Spike` ticket (a question, not a change), and reports an explicitly named one BLOCKED, pointing at `/cr-gh-spike`. |
| `cr-gh-run-harness` | `/cr-gh-run-harness [config-name] [runs]` | Runs one of the codebase-reviewer harness live tests against real LM Studio models. A live test is a config file in `harness/live-tests/configs/`, so it lists them (script, models, baselines), asks which to run and how many runs unless you named them, hands the run to the `harness-runner` agent (`.claude/agents/harness-runner.md`: Haiku 5.5, low effort, no edit tools), which runs `run_live_tests.py` in the background after its no-model pre-flight, watches it with a monitor and says where it is every 30 seconds (the scripts print no progress of their own), and returns the pass/fail checks, the exit code (`3` means nothing was tested) and the log path. It never edits a config or a baseline to make a run pass. |
| `cr-gh-load-context` | `/cr-gh-load-context` | Loads project context in three rounds of tool calls: a read-only script (`facts.sh`, or `facts.ps1` on Windows, beside the skill) gathers the git, GitHub, plan-folder and inventory facts in one call while `cb_indexer_sync_repo` runs, then it reads `README.md`, the start of `EVALUATION.md` and, when present locally (all three are gitignored, so a fresh checkout won't have them), the latest `.ai-plans/` work-order snapshot and the open questions of `.plans/` entries still being worked. `docs/` is listed, not read, and file content is read with Read rather than through Bash, whose long output the compress hook would summarize. Checks what it read against the repo's actual current state (skill inventories, `EVALUATION.md` track status, `.plans/` vs. issue closure, a stale work order, drafts already filed) and reports mismatches as a TODO instead of trusting the docs, then proposes resuming any clear in-progress work or asks what's next. |

`tools/poll_copilot_review.py` is an internal helper of `/cr-gh-autowork`, not a user-facing command. It waits (up to about three minutes per call) for Copilot's review of one exact commit on a PR and prints `review:<id>` lines or `NOT_YET`; the autowork subagent prompt calls it, so you never run it by hand and it needs no other documentation.

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

**Recommended cycle:** `/cr-gh-triage` when starting from a symptom instead of a known change → `/cr-gh-create-issue` to file new work in the same verified, labeled format as the existing backlog → pick an open issue → `/cr-gh-code-it {N}` to implement it → `/cr-gh-code-review {N}` to check the finished branch against the ticket and the plan → `/cr-gh-pr {N}` to open the PR and document it against the issue → `/cr-gh-pr-feedback` to work through review comments as they come in. This is exactly how the code-review backlog filed against this repo (see `.plans/` locally, or the repo's GitHub issues) is meant to be worked one item at a time — and how new items should keep getting added to it. `/cr-gh-autowork` runs this same cycle without stopping between steps, once told to.

**A note on issue and PR numbers:** no tracked file cites issue or PR numbers: not the docs, and not code comments, docstrings, templates or config either. `tests/test_docs_no_ticket_refs.py` fails if one is added. State the fact and the reason directly instead, so the text stands alone. Issue and PR numbers belong in commit messages and PR descriptions, where they resolve.

**Linting and type checking:** `pyproject.toml`'s `[tool.ruff.lint]` and `[tool.pyright]` sections configure ruff (`select = ["F", "B"]`, pyflakes + flake8-bugbear) and Pyright (`standard` mode) for `libs`/`tools`/`hooks`(/`tests` for ruff). Both are dev-only tools for CONTRIBUTING to claude-runway, not something an end user installing the toolkit into their own project needs — so they live in `requirements-dev.txt`, a separate file from the user-facing `requirements.txt`, installed the same way (`uv pip install -r requirements-dev.txt --index-url https://pypi.org/simple`) into the same venv. Resolve the venv's Python interpreter once — `uv venv` creates `.venv/Scripts/` on a Windows checkout and `.venv/bin/` on Linux/macOS, never both, so a hardcoded `.venv/bin/...` path fails with command-not-found on Windows — then run both checks through it: `PYTHON=$(if [ -f .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe; else echo .venv/bin/python; fi)`, then `$PYTHON -m ruff check libs tools hooks tests` and `$PYTHON -m pyright --pythonpath "$PYTHON"`. `/cr-gh-code-it`'s test-running step runs both alongside the unit test suite, so all three must be clean before a PR. Pyright replaced mypy specifically because it's also what VS Code's Pylance extension runs under the hood, so the editor and this repo's gates now report the same findings instead of two type checkers disagreeing. **The `--pythonpath "$PYTHON"` flag is required, not optional**: `pyproject.toml`'s `[tool.pyright]` deliberately has no `venvPath`/`venv` setting (a hardcoded `.venv` there broke CI, which never creates one — see that section's own comment), so without `--pythonpath` pointing Pyright at the interpreter that actually has the dependencies installed, it silently falls back to whatever Python is first on `PATH` and reports every third-party import as missing — confirmed live, this happens even via `$PYTHON -m pyright` run from inside the venv, not just the bare `pyright` case one might expect.

**Pre-push hook:** `.githooks/pre-push` runs the same ruff + Pyright checks (not the unit test suite — see the hook's own comment for why) before a push reaches the remote. It's opt-in per clone, not automatic:

```bash
git config core.hooksPath .githooks
```

This is a convenience, not the real enforcement — it only runs in a clone that has opted in with the command above, and any push can skip it with `git push --no-verify`. Linked worktrees of an opted-in clone, including `/cr-gh-autowork`'s, inherit the setting, because `core.hooksPath` lives in the repo-local config that every worktree shares. CI (`.github/workflows/tests.yml`) runs the identical two checks and is what actually gates a PR.
