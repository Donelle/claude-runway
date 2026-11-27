# Codebase reviewer

The `codebase-reviewer` MCP server is an opt-in, standalone server that runs local models on LM Studio to review a change before Copilot does. It shares nothing with the other servers except `libs/`, so you can install it or leave it out on its own.

What exists today: configuration, the reviewers file, model loading, staged scheduling, the reasoning, verifier and classifier stages, and a database of what every review cost. It has three tools. `list_reviewers` validates your setup and shows what a review would run. `review(type, params)` runs a review; its types are `"reasoning"` (see [Running a review](#running-a-review)), `"verifier"` (see [Verifying findings](#verifying-findings)) and `"classifier"` (see [Classifying findings](#classifying-findings)). `review_report` reads the database (see [What every review records](#what-every-review-records)).

## Turning it on

```bash
python tools/setup_project.py init /path/to/project --codebase-reviewer --reviewers-file /path/you/choose/reviewers.json
```

`init --codebase-reviewer` writes the `codebase-reviewer` block into `.mcp.json`. `upgrade` does not offer it and `templates/mcp.json.template` carries no block for it while the server is unreleased (restoring the offer is tracked with the server's epic); the flag is the only way to add it. A re-run of `init` keeps an existing block as it is: `REVIEWERS_FILE`, `MAX_REVIEWERS` and `MAX_REVIEW_ROUNDS` are carried over key by key (a key the block does not have stays absent, so the shell's value still applies) and any env entries you added stay, because those are yours and no flag of a later run could restore them. Only the Qdrant settings follow the run's flags, and an explicit `--reviewers-file` replaces the path. A new block is written with `REVIEWERS_FILE` present but blank, which counts as "not set" and masks a `REVIEWERS_FILE` exported in your shell; fill it in, or delete the key to inherit the shell's.

**`REVIEWERS_FILE` is required and has no default.** The server cannot know which models you have, so it never assumes a path, a model or a prompt, and neither `init` nor `upgrade` ever writes a default path or a reviewers file. Without `--reviewers-file` the block is written with `REVIEWERS_FILE` blank and `init`/`upgrade` warn you. A blank, missing, unreadable or invalid file is an error when the server runs, naming the path and what is wrong, and no review runs. `claude-runway-doctor <project>` reports the same problem (and exits 1).

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `REVIEWERS_FILE` | none, required | Path to your reviewers JSON file. A leading `~` is expanded; a relative path resolves against the server's working directory. |
| `MAX_REVIEWERS` | `3` | How many reviewers run at once. `1` runs them one at a time; size it to your machine and to LM Studio's Max Concurrent Predictions. |
| `MAX_REVIEW_ROUNDS` | `3` | The most rounds of Copilot feedback an unattended run goes through before a human is brought in. |
| `QDRANT_URL`, `QDRANT_API_KEY` | `http://localhost:6333`, none | For the scoring database, a later piece of work. |
| `CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB` | `~/.claude/claude-runway/codebase-reviewer.db` | Where the server records every review run (see [What every review records](#what-every-review-records)). Inherited from the shell like `CLAUDE_RUNWAY_LMSTUDIO_URL` unless you put it in the block's `env`. |

`CLAUDE_RUNWAY_LMSTUDIO_URL` is inherited from the shell like the other servers' (see [Environment variables](environment-variables.md)). A `MAX_REVIEWERS` or `MAX_REVIEW_ROUNDS` that is not a whole number of 1 or more is an error, not a silent default.

## The reviewers file

```json
{
  "rubric": [
    { "name": "misleading-content", "level": "low", "definition": "a misleading message, comment or doc; nothing breaks.", "score": 2 },
    { "name": "data-loss", "level": "high", "definition": "...", "definitionFile": "data-loss.md", "score": 4 }
  ],
  "reviewers": [
    { "name": "quoting", "model": "<served model id>", "prompt": "...", "active": true,
      "type": ["reasoning", "verifier"], "loadOnDemand": false },
    { "name": "classifier", "model": "<served model id>", "promptFile": "classifier.md", "active": true,
      "type": ["classifier"], "loadOnDemand": true, "contextLength": 32768 },
    { "name": "odd-format", "model": "<served model id>", "promptFile": "odd-format.md", "active": true,
      "type": ["reasoning"], "loadOnDemand": false, "replyFormat": "lines", "translator": "odd-format-to-lines.py" }
  ]
}
```

`rubric[]` entries:

- `name`: unique identifier.
- `level`: one of the server's fixed levels, lowest to highest: `low`, `moderate`, `high`. Several entries can share a level.
- `definition`: what a finding is judged against.
- `definitionFile` (optional): a path (relative to the reviewers file) to a file holding a fuller definition in free text. It must be readable.
- `score`: a probability above 0 and at most 1, such as `0.7`: the minimum **average** probability the classifiers' "does this finding fall under this definition?" answers need to reach for a finding to match this entry. Classifiers are never shown it.

`reviewers[]` entries:

- `name`: unique identifier.
- `model`: the served model id.
- `prompt` and/or `promptFile`: the reviewer's lens. One of the two is required. `promptFile` is a path (relative to the reviewers file) to a markdown file; it must be readable and not blank, and it is read again for each review, so editing it needs no restart. When both are set they are merged into one prompt: the `prompt` text first, then the file's text.
- `replyFormat` (optional, reasoning reviewers): how the server reads the model's reply, `lines` (the default) or `json`. `lines` is plain `key: value` lines in `FINDING` ... `END` blocks (or the single word `NONE`), which small models write reliably: the `quote` field carries code, which is full of `"` and `\`, and models often fail to escape those inside a JSON string. `json` is the older format, one JSON object with a `findings` list. The reader for each is strict on purpose: it does not try to guess other formats (see `translator`). Verifiers always reply in JSON, and a classifier's answer is read from the model's token probabilities, not from a reply format, whatever this says.
- `translator` (optional, reasoning reviewers): the path (relative to the reviewers file) of a script that turns the model's reply into the `replyFormat` format, for a model whose own format is neither. It is run once per reply: a `.py` file under the server's Python, any other file directly (it must be executable). It receives the model's reply on its standard input (UTF-8) and must print the findings in the `replyFormat` format on its standard output (UTF-8). It runs with no shell, in its own directory, with only `PATH`, `HOME`, `LANG`, `LC_ALL`, `SYSTEMROOT`, `TEMP`, `TMP` and `TMPDIR` in its environment (so the server's own settings, such as `QDRANT_API_KEY`, are not passed to it), for at most 30 seconds and 1 MB of output. A translator that exits with an error, times out, writes too much, writes nothing or writes something that is not UTF-8 is reported as a problem for that request, and there is no fallback to the model's raw reply. **It runs with your privileges, so point it only at a script you trust.** It is read only from the reviewers file, never from the repository under review.
- `active`: turn a reviewer on or off.
- `type`: one or more of `reasoning` (produces findings), `verifier` (confirms or drops each finding) and `classifier` (checks whether a verified finding falls under each rubric entry's definition).
- `loadOnDemand`: `true` when the server should load the model itself.
- `contextLength` (optional): the context the model is loaded with; used only when `loadOnDemand` is `true`.
- `tools` (optional, default none): the tools this reviewer may ask for while it works, a list of `read_file` and `grep`. A reviewer with `tools` can reply with a request for lookups instead of its answer and get the result before it answers; without the key nothing about the reviewer changes. A request is read from the model's reply as written, before any `translator` runs, so a model that needs a translator (it does not write the reply format) will not use lookups. See [Letting a reviewer ask for more context](#letting-a-reviewer-ask-for-more-context).
- `contextShare` (optional, above 0 and at most 1, default 0.25): with `tools`, the share of the reviewer's room (its context after its prompt and its reply reserve) kept for what its lookups return, across all rounds. The diff chunks are cut smaller to leave it. Ignored without `tools`.

Validation reports every problem in the file at once, not just the first: unknown keys (a misspelt `loadOnDemand` would otherwise silently mean something else), an unknown `level`, duplicate `name`s, a `score` that is not a probability above 0 and at most 1 (the old 1 to 5 scores are refused), an unreadable `definitionFile`, and reviewers that share a model but disagree on `loadOnDemand` or `contextLength` (they are scheduled as one group, so they must agree). An active classifier with an empty rubric is also an error.

## Model loading

When `loadOnDemand` is `true`, the server loads the model through LM Studio's REST API (`POST /api/v1/models/load`, with `contextLength`) just before use and unloads it (`POST /api/v1/models/unload`) afterwards. It unloads only instances it loaded itself: a model you loaded (the compression model, say) is never unloaded, even when a reviewer names it with `loadOnDemand` set. When `loadOnDemand` is `false` and the model is not loaded, that is an error naming the model, and nothing is loaded for you.

## Scheduling

Stages run in order: every reasoning reviewer, then every verifier, then every classifier. A reviewer with several roles takes part in each of its stages, and a stage nobody is active in is skipped. Within a stage:

- reviewers run in parallel up to `MAX_REVIEWERS`;
- reviewers sharing a model are grouped, so a `loadOnDemand` model loads once per stage and stays loaded when the next stage uses it;
- a `loadOnDemand` model's group runs alone, so two on-demand models are never loaded at the same time just to fill slots (that is the machine `loadOnDemand` exists for), while the models that are already loaded share one pool first. A model kept for the next stage is unloaded first if another on-demand model has to load before then (and that group is skipped with an error if the unload fails), and groups are ordered so a model the next stage needs again goes last and can stay loaded.

`list_reviewers` prints this plan for your file.

## Running a review

`review` is one tool for every kind of review: `type` picks it and `params` is a dict of that type's arguments. Call it with an empty `params` to see a type's keys, defaults and an example. A bad type, an unknown or missing key and a wrong value come back as one `Error:` listing every problem.

`review(type="reasoning", params={"repo_path": "/path/to/repo", "base_ref": "origin/main"})` runs every active `reasoning` reviewer over the diff `base_ref...head_ref` and returns all of their findings. The diff and files are read on the server; only the findings come back to the caller.

| Key | Meaning |
|---|---|
| `repo_path` (required) | The git repository to review. |
| `base_ref` (required) | The base to diff against, such as `origin/main`. |
| `head_ref` | The end of the range; default `HEAD`. It is resolved once, at the start of a run, to the commit it names (for every review type), and the diff and every later read of a cited file use that commit, so a `HEAD` or branch that moves while a run takes minutes does not change what is reviewed. |
| `context_lines` | Lines of surrounding code shown around each change; default 20. The whole file is not sent, since it would not fit a small context. |
| `chunk_chars` | Characters of code per request. By default it comes from each reviewer's `contextLength` (8192 tokens is assumed when it is not set), at a low three characters per token. A value you pass is used as given, except for a reviewer with `tools`: there it is a ceiling and is cut to what is left after the room kept for the lookups' results. |
| `pr` | The pull request number being reviewed. It changes nothing about the review; it is recorded with the run's timing so a run can be joined to that pull request's other metrics. |

How the prompt is built, per reviewer: its own prompt, then the `.github/instructions/*.instructions.md` files whose `applyTo` matches a changed path (a file with no `applyTo` applies to every path; globs are comma-separated, with `*` inside one directory, a `**` segment across any number of directories, `?` for one character, and everything else literal, brackets and braces included; a glob over 512 characters is ignored, and matching takes bounded time however the glob is written), then the description of how a change is shown, then the output format for the reviewer's `replyFormat`. A reviewer with a `promptFile` is not given that last part: see "A `promptFile` replaces the reply instructions" below. Those instruction files are found automatically in the repository under review, so each must be a regular file that resolves to a place inside that repository, and at most 80,000 bytes of it are read. A symlink is followed only when it points at a file git tracks (a link to a tracked rules file is fine; a link to `.env`, to anything under `.git`, to an untracked file or out of the repository is ignored, and so is every link when the directory is not a git repository). A `promptFile` you name in the reviewers file is your own choice and is read wherever it points. Each changed file is shown to a reviewer as its new version: a `File:` line, then every line with its real line number on the left, a `+` marking lines the change added or modified, and one `~ N line(s) removed here` marker where lines were deleted (removed code itself is never shown, so a model cannot mistake it for current code). A file too big for a reviewer's context is cut per hunk and then per line, never inside a line, and every piece starts with its `File:` line and keeps its real line numbers. A line too long for one request is split, and each later fragment repeats its line number with a `>` in place of the `+`. When a change only deletes lines and nothing of it survives beside the removal (`context_lines` of 0), the deleted lines are listed with a `-` after `~ N line(s) removed after line L`, so the reviewer can judge whether removing them was safe. Binary, deleted, renamed-only and mode-only files have no new text and are not sent; the result lists them in a warning so a skipped file is never silent. One limit to know: when a deletion has surviving lines beside it, the reviewer sees only that lines were removed, not what they were, so it cannot tell that a removed guard mattered; keep that in mind for changes that mostly delete code. A diff needing more than 100 requests per reviewer is refused: review a smaller range.

### How a review runs: steps, a policy, and the loop

A review is a sequence of steps over one run's state (kept in memory for the one `review` call): `reason` (the reasoning reviewers read the diff), `ground` (findings whose quoted code is not in the file are dropped; no model), `verify` (the verifiers confirm or reject what is left), `classify` (the classifiers match what still stands against the rubric) and `finish` (format the result, close the run's record). A policy chooses the next step; today's only policy is the rules policy, which runs each type in the order it always ran: `reasoning` is reason; `verifier` is reason, ground, verify; `classifier` is reason, ground, verify, classify. So a review returns what it returned before this split.

The loop, not the policy, enforces what must always hold. Each step runs at most once, and in order: `ground` only after `reason`, `verify` only after `ground` (and only on grounded findings), `classify` only after `verify` (and only on findings the verifiers did not reject). An action that breaks a rule is recorded in the run's notes (stage `orchestrator`), replaced by `finish`, and shown in the result's warnings. A run that takes more than eight steps stops with a note, and the run always finishes: its record is closed as `finished`, `refused` or `errored` whichever way it ends. Reviewers' resident models are checked before the first step, the classifier's first, so a model that is not loaded is reported at once and not after minutes of reasoning. The code is in `libs/reviewer_orchestrator_lib.py`.

### A `promptFile` replaces the reply instructions

A reviewer that has a `promptFile` is not given the built-in instructions for how to reply (a verifier too, when `verifier` is its only type; see below). The file (merged after any inline `prompt` text, as above) says how the model replies, so it has to describe the format `replyFormat` selects, because the reader is strict. This is what lets the prompt change per model as you learn what suits it, with no change to the server and no restart: the file is read again on every review. The rules from the repository and the description of how a change is shown are still added, after the file.

- With no `promptFile` the server ends the prompt with its own reply instructions for `replyFormat`, so a reviewer that only has a lens works as it stands.
- An existing reasoning reviewer whose `promptFile` holds only a lens, and which relied on the server adding the reply instructions, needs the reply instructions added to the file (or its lens moved to the inline `prompt`); until then its replies cannot be read. `list_reviewers` and `doctor` say which reviewers this affects.
- A reviewer that has both the `reasoning` and `verifier` types shares the file, but the verifier still adds its own JSON verdict instructions after it, so keep reply instructions out of a shared file: give the model two reviewer entries, one per type (reviewers that share a model load it once).
- A reviewer whose **only** type is `verifier` and that has a `promptFile` gets the file (after any inline `prompt`) and nothing else: no built-in verifier steps, no built-in reply format. The file has to say how to verify and has to describe the JSON verdict the reader accepts (`verdict` `confirmed` or `rejected`, `failing_input`, `wrong_result`, `reasoning`; a `confirmed` without a failing input and the wrong result is not counted). A verifier without a `promptFile`, or one that shares its file with a reasoning role, is as it always was. `harness/live-tests/prompts/deepseek-loop-prompt.md` is a worked example, written for a verifier that can ask for lookups.
- A reviewer with `tools` and a `promptFile` (a reasoning reviewer, or a verifier-only one) is not given the built-in lookup section either: the file has to say how to ask, in the format the reviewer replies in (JSON for a verifier).
- `harness/prompts/` in this repository has ready-made files for the plain-lines format, a worked example, and a table of what has been measured for which model.
- A model whose natural format is neither `lines` nor `json` can be given a `translator` instead of a prompt that fights it.

Every finding from every reasoning reviewer is kept: there is no vote threshold, so a finding only one reviewer saw is not lost. Findings are grouped by class. Findings in the same file and class whose line ranges overlap are shown as one entry, but two different claims are never collapsed into one: the same claim (ignoring case, spacing and a final full stop) lists every reviewer that raised it, and a different claim, or the same claim with a different failing input, is kept under it as an "also" line with its own reviewers. Each entry carries its file, line range, class, claim and the concrete failing input it describes. They are not verified or classified by the `"reasoning"` type; the `"verifier"` and `"classifier"` types go on to do that. A reviewer whose requests all fail is reported in the result and does not stop the others.

## Choosing prompts for your reviewers

What a reviewer finds depends mostly on how much its prompt asks it to cover at once, and the prompt is yours to set. The server does not change a reviewer's scope on its own; `list_reviewers` only tells you what your setup implies, under "Advisories".

- **One reasoning reviewer.** It has to cover every kind of defect by itself, so give it a broad prompt. A narrow one leaves the rest of the code unchecked.
- **Several reasoning reviewers.** Each can take one narrow lens (injection, off-by-one and empty input, shared state, and so on). Small models usually do better on one narrow question than on "find any defect": in a test on a small file with three unrelated defects, one general prompt on a small model found one of them, and three narrow prompts (two of them on the same small model) found all three.
- **Reviewers on the same model share one load.** Splitting a small model into several narrow lenses costs time, not memory.
- **Same model, same prompt.** At temperature 0 they raise the same findings, so one of them is wasted. The same prompt on two *different* models is a second opinion, and is not reported.
- **The "whole codebase" is never in front of one reviewer.** A reviewer sees one file's changes at a time, in a context of a few thousand tokens. A broad prompt widens what it looks for, not how much code it can see.

## Verifying findings

`review(type="verifier", params={"repo_path": "/path/to/repo", "base_ref": "origin/main"})` takes the same params as `"reasoning"` and does more with the result. On the server it runs the reasoning stage, then **grounds** each finding, then has every active **verifier** reviewer check what is left. Only the outcome comes back to Claude.

**Grounding (no model).** The reasoning stage asks each reviewer to copy a `quote` with every finding: the exact line of code the claim is about. Before any model reads a finding, the quote is looked up in the cited file as it is at `head_ref`, resolved once to a commit before the diff is cut so a checkout or commit made while the models run does not change what is read (whitespace is ignored, so indentation and wrapping do not matter; a short span of lines is fine). Only files of the change can be read, so a finding cannot name its way to another file.

*   Found at the claimed lines: kept.
*   Found elsewhere in the file: kept, and its line numbers are corrected to the occurrence nearest the claim.
*   Not in the file, no quote, a quote too short to place (under 3 characters), or a file that is not part of the change: dropped and listed as **ungrounded** with the reason.

Small models report a lot of code that is not there (a bare `except` at the lines of a docstring, an "unused import" of a module that is used); a string search removes those for free and leaves the verifiers the claims about real code. Grounding runs even when no verifier is active.

**Verifying.** Each verifier is shown one finding and the numbered code around the lines it cites, and works from the reasoning alone (it runs nothing). It must **confirm** with a concrete failing input and the wrong result it produces, or **reject**. A confirmation without both is not counted. A verifier that answers nothing usable for a finding is left out of that finding's vote.

| Verifiers that answered | Result |
| --- | --- |
| all confirm | **confirmed** |
| all reject | **rejected**: dropped, and listed with the verifier's reason |
| they disagree | **disputed**: returned with each verifier's decision and reasoning |
| none answered, or no verifier is active | **unchecked**: kept, and the result says so |

With no verifier active the findings are *grounded but not verified*, and the result says that, so they are never mistaken for checked ones.

**Setting it up.** Give a reviewer `"type": ["verifier"]` in the reviewers file. A reasoning-distilled model (for example DeepSeek-R1-Distill-Qwen-7B) suits the role; see `harness/README.md` for the models and for measuring how well a verifier does. A reviewer may have several roles. The resident verifier models are checked before the reasoning stage starts, so a missing one is reported at once and not after minutes of reasoning.

**Limits.** At most 100 findings are verified per review (each costs one request per verifier); the rest come back unchecked with a warning. A model with both the reasoning and the verifier role, set `loadOnDemand`, is loaded again between the two stages.

## Letting a reviewer ask for more context

A reviewer or verifier sees only the change (a diff chunk, or one finding and the lines around it), so a claim such as "the caller can pass `None`" cannot be checked when the caller is in another file. The models that suit these roles (Qwen2.5-Coder, the DeepSeek-R1 distills) do not report `tool_use` in LM Studio, so they cannot be handed tools natively. They can still write, so the way to ask is a line in the reply, and **the server runs the tool, never the model**.

Give a reviewer `"tools": ["read_file", "grep"]` (either or both; that list is its menu). Its system prompt then carries a section that says when to look something up (a claim that depends on code the model cannot see), shows the tools it may ask for with an example, and says that this takes precedence over "you cannot confirm": a verifier is told to look the evidence up before it rejects a claim for lacking it.

**A model is never asked to switch formats to ask.** It asks in the format it already replies in:

| Reply format | A request looks like |
| --- | --- |
| JSON (every verifier; a reasoning reviewer with `replyFormat: json`) | `{"need": [{"tool": "grep", "pattern": "normalize_sku"}, {"tool": "read_file", "path": "libs/sku.py", "start_line": 1, "end_line": 40}]}`: each entry names a `tool` and gives that tool's arguments as the executor takes them (`read_file`: `path`, optional `start_line`, `end_line`; `grep`: `pattern`, optional `path_glob`) |
| `lines` (a reasoning reviewer with `replyFormat: lines`, the default) | `need: read_file <path> [<first>-<last>]` and `need: grep <pattern> [-- <path glob>]`, one lookup per line |

*   **The loop.** Each request is run through the same read-only tool executor (tracked files at the pinned commit, anywhere in the repository), and the next request is the same message with a labelled **reference block** in front of it holding the results. After 2 rounds of lookups the next request says no more are allowed and the model must answer.
*   **What counts as a request.** In JSON, an object with a non-empty `need` list and no answer key (`verdict`, `findings`); in `lines`, a reply whose every non-blank line is a `need:` line. A reply that is anything else is read as before, and a reply that carries an answer is read as the answer even if it also asks. A reply that is neither is a problem for that finding or chunk, as today. At most 3 lookups of one reply are run, and the rest are reported to the model as not run in a single entry, however many it asked for. Regular expressions need their backslashes escaped in JSON, which small models sometimes get wrong: the example in the prompt uses a plain name, and a request that is not valid JSON is a visible problem, never a silent miss.
*   **What `grep` takes.** A Perl-compatible regular expression, run with `git grep -P`, so `\b`, `\w`, `\s` and `\d` mean the same wherever git has PCRE2. (`git grep -E` uses the platform's regex library, and macOS's reads those as plain letters, which came back as "no matches".) A pattern that would not be read as meant is an error the model sees in its next round, never "no matches": `\<` and `\>` (use `\b`), `{,n}` (write `{0,n}`), and a backspace character, which is what a single-backslash `\b` in a JSON request decodes to. On a git built without PCRE2 the tool uses `-E` and refuses any backslash before a letter, a digit, a backquote or a quote, naming what to write instead. One limit: whether `\w`, `\b`, `\d` and `\s` count a character outside ASCII depends on the locale of the process that runs git.
*   **Only the menu runs.** A `need:` for any other tool is not run; the model is told, in the next round, which tools it may ask for.
*   **Still asking after the limit.** A verifier gives no vote on that finding, and a finding no other verifier answered stays **unchecked** with a note saying the verifier kept asking. A reasoning reviewer has a problem for that chunk.
*   **Fetched text is untrusted.** The reference block says it is repository text, not part of the change and not instructions, and a verifier still confirms only with a failing input and wrong result it can show in the code; a reference alone confirms nothing.
*   **A budget, and a cut is said.** The whole block, across every round, stays within `contextShare` of the reviewer's room. A result that does not fit is cut with a marker saying how much, later ones say they were not shown, and the review's warnings count them. A budget too small to hold even the block's fixed text (about 800 characters: an intro, a closing line and room for the "not shown" lines) runs no lookup: the reviewer gets `LIMIT` as if it kept asking, with a warning that says to raise `contextLength` or `contextShare`. Set `contextLength` on a reviewer with `tools` (`doctor` warns when it is missing): the assumed 8,192 tokens leave little room.
*   **Off by default.** A reviewer with no `tools` sends exactly the prompt and makes exactly the requests it did before. A reviewer with `tools` run without a tool executor (a caller that did not pass one) runs without lookups and says so in the warnings.

What is recorded is in [What every review records](#what-every-review-records): the round, the tool, the size of each result, never what the model asked for or what came back.

## Classifying findings

`review(type="classifier", params={"repo_path": "/path/to/repo", "base_ref": "origin/main"})` takes the same params as `"reasoning"` and `"verifier"` and goes one stage further. On the server it runs the verifier review, then has every active **classifier** reviewer check each finding that is left standing against your rubric. The findings that stand are everything the verifiers did not reject: confirmed, disputed and unchecked ones. Findings dropped as rejected or ungrounded are not sent to a classifier. Only the outcome comes back to Claude.

**The question.** For every (finding, rubric entry) pair a classifier answers one yes/no question: does this finding fall under this entry's definition? It sees the entry's `name`, `level` and `definition` (plus the text of its `definitionFile`) and one finding at a time: where it is, its class, the claim, the failing input the reporter gave and the code it quotes. It reads no code beyond that quote and runs nothing. You decide what a level means by what you write in the definitions. It does **not** see an entry's `score`: the score is the server's cut-off, and a model told the cut-off can learn to answer to it.

**The answer is a probability, not a sentence.** The request asks for one token and the model's token probabilities, at temperature 0. The answer is the model's probability of "yes" out of "yes" and "no" (`p(yes) / (p(yes) + p(no))`), a number from 0 to 1 that the model never writes, so there is no reply format to parse and asking again gives the same number. This needs a model and a runtime that return token probabilities. A model that does not (a reasoning model, whose reply starts with its thinking, is one) cannot be a classifier: the review reports an error naming the reviewer and the model. The requests go to LM Studio's OpenAI-compatible `/v1` URL, because the native `/api/v0` that the other stages use (for the timing statistics it adds) returns no token probabilities; the classifier's rows in `requests` therefore carry token counts but no time to first token or tokens per second.

**The score is arithmetic.** Each classifier is asked once per rubric entry for each finding. For each entry the server takes the plain **average** of the probabilities the classifiers gave it, and the entry matches when that average, rounded to six places, is at or above its `score`. Two classifiers give two probabilities per entry; one classifier gives one.

*   No entry reaches its score: the finding is **unclassified**. It is dropped from the result and recorded, with every entry's average.
*   One or more entries reach their score: the finding takes the entry with the highest `level`, in the fixed order `low` < `moderate` < `high`. Entries of the same level go to the higher average, then to the one that comes first in the rubric. When more than one reached its score that is an **overlap**, recorded, with the entries that lost.
*   Classifiers that disagree are not a separate case: their probabilities are averaged, so a split lowers each entry's average and, if none reaches its score, the finding is unclassified.
*   A request that returns no probability leaves that classifier without an answer for that finding (an answer needs a probability for every entry, since a missing one would read as "does not fit"), and it is reported as a problem. A finding no classifier answered, or with no classifier active, has no rubric level: it is kept, and the result says so.

The findings come back grouped by class, each with the entry's name, its level and its average probability, and with what the verifiers found for it: the failing input a verifier reproduced for a confirmed finding, and each verifier's verdict and reasoning for a disputed one, as in the verified review. A finding with no rubric level keeps the same lines.

**What is recorded.** In the reviewer's own database (the file `CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB` names), one row per finding in the `classifications` table: the run, the finding's number, file, lines and class, the verification status it arrived with, its outcome (`classified`, `unclassified` or `unchecked`), the entry, level and average probability (`confidence`), whether it was an overlap and which entries lost to the one it took, every classifier's answer (its probability for every entry, kept at full precision so any average can be recomputed and gives the stored outcome) and the average for each entry. The claim text is not stored there (this database holds no model-written text); the session log shows each finding with its classification. Each classifier request is also a row in `requests`, with the timing and tokens (a request whose reply could not be read as yes or no keeps its token counts), like the other stages. When a model's reply does not begin with Yes or No, the error the caller gets names its likeliest first tokens; the database and the session log hold the same sentence without them, since they are the model's own text. Together these show where a rubric needs tweaking: an entry that is never matched, one that is matched at its score every time, or a model whose probabilities never fall below the scores.

The run's `review` event in the shared `metrics.db` carries the counts only: findings per level (`classified_low`, `classified_moderate`, `classified_high`), `unclassified` and `overlaps`. A review without a classifier stage has none of these keys. Rubric entry names and definitions are your own text and stay in the local database.

**Setting it up.** Give a reviewer `"type": ["classifier"]` in the reviewers file, and a `rubric` (an active classifier with an empty rubric is refused). Use a model that returns token probabilities; `harness/README.md` lists the builds that have been measured, and how to measure another. To get an average from several models, activate several classifiers. A reviewer may have several roles. The resident classifier models are checked before the reasoning stage starts.

**Limits.** At most 100 findings are classified per review (each costs one request per rubric entry per classifier, each of one token); the rest come back without a rubric level and with a warning.

## What every review records

Choosing models by guesswork does not work once reviews take minutes, so every review run, from the `review` tool and from the test harness alike, is recorded in the server's own SQLite database, `~/.claude/claude-runway/codebase-reviewer.db` (override with `CLAUDE_RUNWAY_CODEBASE_REVIEWER_DB`). The harness records into a throwaway database under the system temp folder instead, unless you set that variable (see [Where a run is recorded](../harness/README.md#where-a-run-is-recorded)), so test runs stay out of this history. The server owns it, so it works whether or not the analytics server is installed. It is built to hold everything the reviewer records over time, and today it holds the performance data:

- **Per run:** start, end and wall time; the review type; the repository, the diff range, the branch of `head_ref` and the pull request when you pass `pr`; the number of chunks, findings and confirmed findings; the settings that change the cost (`MAX_REVIEWERS`, `chunk_chars`, `context_lines`); and how it ended (`finished`; `incomplete`, when the review returned but some stage got no answer from any reviewer because every request failed, its model never loaded, or every reviewer failed before sending a request (a diff too big for it); `refused`, for a configuration, a range or a model the run could not start with; `errored`). A run that fails or times out part-way is still recorded.
- **Per stage:** start and end. The stage is a value, not a column, so a stage added later records and reports with no change.
- **Per request:** start and end, reviewer, model, role, what it was about (a diff chunk for reasoning, a finding for the verifier and the classifier), prompt and reply size in characters, prompt and reply tokens, the latency, the time to first token and the average time between tokens (see [Timing and tokens](#timing-and-tokens)), how many problems the reply produced with a one-line summary of them, and whether it was `answered`, `timed_out`, `refused` (HTTP 400) or `failed`.
- **Per lookup** (a reviewer with `tools`): which round of the loop each request was (`requests.need_round`, 0 for the first ask) and whether it ended the loop by still asking after the round limit (`need_limit_hit`); and, in `tool_calls`, each lookup it asked for: the request, the round, the tool's name, the size of the result (`result_chars`), how much of it reached the model (`shown_chars`), whether the tool cut it (`truncated`) and whether it was an error. `truncated` is the tool's own cut; a result the reference budget cut or never showed is told apart by `shown_chars` being below `result_chars`. Never the arguments the model wrote, never what came back, and never a tool name the model made up (stored as `(not on the menu)`): the database holds no model-written text.
- **Per model load and unload:** start, end, the context length the model was loaded with, and the model's details from LM Studio (publisher, architecture, format, quantization, maximum context length). The context length is kept on each load, since the same model can be loaded with a different `contextLength` from one run to the next.
- **The machine:** chip, memory, OS and the LM Studio build, once per run, so results from different machines are never pooled. When `CLAUDE_RUNWAY_LMSTUDIO_URL` points at LM Studio on another computer, the models ran there, so the run is filed under that host instead (its chip, memory and OS unknown to this machine), and `this_machine` leaves it out. The build comes from `lms version`, which prints a commit hash and no version number, so it is recorded as `lms CLI commit <hash>` (empty when `lms` is not installed).
- **The reviewer setup:** the model, `replyFormat`, `translator` and a hash of the prompt text (with the reviewer's `tools` when it runs the `need:` loop, so the same model with and without lookups is two setups), so two setups are never mixed together. The run and setup ids are the identifiers later tables (scoring, rubric signals, Copilot feedback, replay results) refer to.

### Timing and tokens

Each request records these, so a model's speed can be compared with another's:

| Recorded | Meaning | Where it comes from |
|---|---|---|
| `latency_ms` | Total latency in milliseconds: from submitting the request to receiving the final token. A failed or timed-out request has one too. | Measured by the reviewer around the call, so it includes any wait in LM Studio's queue. |
| `ttft_ms` | Time to first token, in milliseconds (prompt processing included). | LM Studio's `stats.time_to_first_token`. |
| `inter_token_ms` | Average time between tokens, in milliseconds: `1000 / tokens_per_second`, the average interval while generating (the time to first token is not in it). | LM Studio's `stats.tokens_per_second`. |
| `tokens_per_second`, `generation_ms`, `stop_reason` | The rate behind the line above, the generation time the server reports, and why generation ended (`eosFound`, or a length cut-off). | LM Studio's `stats`. |
| `prompt_tokens`, `reply_tokens`, `reasoning_tokens` | Input and output token counts, and the part of the output spent thinking. | The response's `usage`. |

The reviewer's completion client is built on LM Studio's native REST API (`/api/v0`), not the OpenAI-compatible `/v1`: both answer chat completions in the same shape, but only the native one fills in `stats` (checked live: on `/v1` it is an empty object). The model loader already needs that API, so no new requirement is added. A request whose response carries no `stats` (another server, or an older LM Studio) still records the latency and the token counts, and leaves the other timing columns empty. `generation_ms` is not comparable between runtimes: LM Studio's llama.cpp (GGUF) runtime counts the wait for the first token in it and its MLX runtime does not (checked live: a GGUF request showed 1543 ms to first token inside 14613 ms of generation and a 14628 ms latency; an MLX one showed 508 ms to first token beside 3063 ms of generation and a 3579 ms latency), so total time is `latency_ms` and generation speed is `inter_token_ms`. `ttft_ms` and `inter_token_ms` are LM Studio's own measurements, so they leave out the time a request waited for a free slot, which is what the measured `latency_ms` includes; a latency well above the generation time means requests are queueing (see `MAX_REVIEWERS` and LM Studio's Max Concurrent Predictions).

**Comparing models: what is comparable.** The numbers depend on the runtime LM Studio uses for a model, not on the model itself: GGUF models run on its llama.cpp runtime and MLX models on its MLX runtime, and the `stats` are produced by whichever one served the request. Across the two runtimes seen so far:

| Figure | Comparable across runtimes? | Basis |
|---|---|---|
| `latency_ms` | Yes | Measured by the reviewer around the call; LM Studio is not involved. |
| `ttft_ms` | Appears to be | On both runtimes it comes out just under the measured latency for a short reply (an MLX request: 1295 ms to first token inside a 1376 ms latency). Not proven identical in every case, for example a very long prompt. |
| `inter_token_ms`, `tokens_per_second` | Yes | Both runtimes reproduce their own generation time from it: a GGUF request, 611 tokens × 21.43 ms, and an MLX request, 172 tokens × 17.81 ms. |
| `prompt_tokens`, `reply_tokens`, `reasoning_tokens` | Yes | The standard `usage` block. |
| `generation_ms` | **No** | The GGUF runtime counts the wait for the first token in it and the MLX runtime does not. Do not compare it across models; use `latency_ms` and `inter_token_ms`. |

This was checked by reconciling the figures of about a dozen real requests on two runtimes, not from LM Studio's source, so a third runtime (or a later LM Studio) may differ: when you start using a new kind of model, check that its `ttft_ms` plus its `inter_token_ms` times its reply tokens roughly add up to its `latency_ms` before trusting a comparison. A server that returns no `stats` at all (anything other than LM Studio's native API) leaves the timing columns empty.

**Each machine gives different results.** Every timing figure here is a property of the model, the runtime AND the machine it ran on: the chip, the memory, the LM Studio build, the context length the model was loaded with, whatever else the machine was doing, and how many requests LM Studio ran at once (`MAX_REVIEWERS`) all change it. A figure recorded on one machine says little about another, even for the same model, and the same model on the same machine will vary from run to run too, which is why the report shows medians with the number of requests behind them. Each run records its machine (chip, memory, OS, LM Studio build, or the host of a remote LM Studio) so results from different machines are never pooled by mistake, and `this_machine` (the `review_report` argument, `--this-machine` on the command line) limits a report to the machine you are on. Choose models from figures recorded on the machine that will run them.

**Tokens.** For a reasoning model the reply count INCLUDES its thinking: LM Studio returns the thinking separately from the reply text, yet `completion_tokens` counts both, and `completion_tokens_details.reasoning_tokens` (recorded as reasoning tokens) is the part of it spent thinking. Checked live on this setup with DeepSeek-R1-Distill models: a verifier request with 572 reply tokens, 463 of them reasoning tokens, returned a 505-character reply, about 109 tokens of visible text. So the reply text is `reply_tokens - reasoning_tokens` tokens, and a non-reasoning model records 0 reasoning tokens. A response without a `usage` leaves the counts empty. Read the counts as what the server reports, and compare them only between runs of the same setup.

**The session log.** What each stage evaluated goes to a readable log file, written like the harness's own logs, in a `codebase-reviewer-logs` folder next to the database: one file per session, named after the session id (`<session id>.log`, the same id as `runs.session_id` and as the summary event in `metrics.db`). The reasoning, verifier and classifier stages all write into that one file, not files of their own, and a session's reviews are sections of it, one after another. A review's section has:

- a header (run id, time, review type, repository, diff range, branch, pull request, settings);
- in time order, every request: `[  12.7s] verifier | v (model-v) | finding 3 | answered in 12.7s, first token 1062 ms, 21 ms/token, tokens 629 in / 554 out (473 thinking)  [request 1a2b3c4d]`, with the notes under it: the findings a reasoning reviewer raised (file, lines, class, claim, failing input and the quoted code), a verifier's verdict with its reasoning and the failing input and wrong result it gave, a classifier's entry, level and confidence (with the overlap, or the dispute), and the problems the reply produced (`problem: the reply was not JSON`);
- stage-level lines: what grounding kept and dropped, and why;
- the model loads and unloads, and how long each took;
- the result returned to the caller (cut at 20,000 characters), and a footer: outcome, wall time, model combination, requests, timeouts and failures.

A request in the file is identified by the first eight characters of its id in the `requests` table (`where id like '1a2b3c4d%'`). Long lines are cut at 600 characters and a note shows at most 12 detail lines, so a review cannot make a huge file. The file never contains the prompt sent to a model or the model's raw reply; it contains what the reviewer concluded from each. A problem is recorded as a fixed sentence: a translator that fails is recorded with its exit status only (what it printed, which can echo the reply, goes to the caller; the log's copy of the result returned shows `[translator output withheld]` in its place), a verdict that is neither `confirmed` nor `rejected` is recorded without the value the model wrote, and a classifier's entry name that is not in the rubric likewise. Every stage writes through one `note` call, so a stage added later appears in the same file with no change. A session's reviews append to its one file, so the folder grows with the number of sessions, not the number of reviews, and concurrent sessions write different files. Files beyond the newest 200, and any older than 30 days, are deleted when a run closes; a file another server wrote to after the folder was listed is kept. A run with no session id (none could be worked out) uses one `no-session-<pid>.log` per process.

**What the row keeps if the file is gone.** Because the files are pruned, each request's row also keeps `problems` (how many) and `problem_summary` (each distinct problem once with its count, without the reviewer name or chunk number, cut at 500 characters), and the run's row keeps its counts, outcome and timing, so what went wrong with a model's replies stays answerable without the file. The file adds the readable detail: which findings, which verdicts and why.

**Writes never get in the way of a review.** A run is held in memory and written once, in one transaction, when it ends; a database that cannot be written (locked, read-only, corrupt) is reported on the server's stderr and the review returns as usual. Several sessions share the file, so it runs in WAL mode with a short busy timeout (2 s); a run takes the write lock once, for its write (the schema is only touched when the file is from an older version), so a locked database costs a run at most that timeout. A run that is killed outright (the server process is stopped mid-review) is not recorded.

**Beside the database.** Each run also writes one summary event to the shared `metrics.db` (`metric_id` `codebase-reviewer`, see [Shared metrics store](metrics.md)), always and whether or not the analytics server is installed; a run with a classifier stage adds its classification counts to it (see [Classifying findings](#classifying-findings)). A review from the `review` tool also writes, when `CLAUDE_RUNWAY_TRACK_SAVINGS` is on, the characters of diff the local models read against the characters returned to Claude to the savings ledger; a harness run does not, since no Claude session was spared reading that diff. Both are derived from the run and fail open. The savings record needs `CLAUDE_RUNWAY_TRACK_SAVINGS` in the server's environment: leave it out of the `codebase-reviewer` block so the server inherits the shell's export. It goes to the live session of the project the server was started in (its working directory, the same way the analytics server finds it), not to the newest session of any project and not to the reviewed repository, and it is skipped when that project has no live session.

### Reading it

`review_report` (the tool) and `python tools/codebase_reviewer_report.py` (the same views from a shell, `--json` for machine-readable output) answer three questions without SQL. Add `this_machine` (`--this-machine`) to use only runs recorded on this machine.

| Question | View | Shows |
|---|---|---|
| Which model combination suits my machine? | `review_times` | Per combination (`reasoning:a+b\|verifier:c`), the median review wall time over its finished reviews and how its middle review splits into model loading, each stage and everything else, fastest first, with the number of reviews behind each. |
| How long does model X take? | `response_times` (`model` to pick one) | Per model and role, the median, p90 and maximum response time of answered requests, how many timed out or failed, and the medians (p90 for the first two) of latency, time to first token and time between tokens in milliseconds, with the input and output token counts. |
| I want a review under N minutes | `under_target` (`target_minutes`) | The combinations with a recorded review inside the target, and how many of their reviews that is out of how many it has. |

A stage's time in `review_times` is its wall time minus the model loads and unloads inside it, so load, reasoning and verifying add up to the review rather than counting a load twice. The split is that of one real review, the middle one (the lower median when the count is even), so its parts add up to its own wall time, which it shows; medians of each part taken on their own would not add up to anything. Only finished runs are used: an `incomplete` run returned quickly because it did not do the work, so it is left out of every review-time view. A median over two reviews says little: read the review count beside every figure.
