# Codebase reviewer

The `codebase-reviewer` MCP server is an opt-in, standalone server that runs local models on LM Studio to review a change before Copilot does. It shares nothing with the other servers except `libs/`, so you can install it or leave it out on its own.

What exists today: configuration, the reviewers file, model loading, staged scheduling and the reasoning stage. It has two tools. `list_reviewers` validates your setup and shows what a review would run. `review(type, params)` runs a review; its one type so far is `"reasoning"` (see [Running a review](#running-a-review)). The verifier and classifier stages plug into the same tool in later releases.

## Turning it on

```bash
python tools/setup_project.py init /path/to/project --codebase-reviewer --reviewers-file /path/you/choose/reviewers.json
```

`init --codebase-reviewer` writes the `codebase-reviewer` block into `.mcp.json`; an interactive `upgrade` offers it. A re-run of `init` keeps an existing block as it is: `REVIEWERS_FILE`, `MAX_REVIEWERS` and `MAX_REVIEW_ROUNDS` are carried over key by key (a key the block does not have stays absent, so the shell's value still applies) and any env entries you added stay, because those are yours and no flag of a later run could restore them. Only the Qdrant settings follow the run's flags, and an explicit `--reviewers-file` replaces the path. A new block is written with `REVIEWERS_FILE` present but blank, which counts as "not set" and masks a `REVIEWERS_FILE` exported in your shell; fill it in, or delete the key to inherit the shell's.

**`REVIEWERS_FILE` is required and has no default.** The server cannot know which models you have, so it never assumes a path, a model or a prompt, and neither `init` nor `upgrade` ever writes a default path or a reviewers file. Without `--reviewers-file` the block is written with `REVIEWERS_FILE` blank and `init`/`upgrade` warn you. A blank, missing, unreadable or invalid file is an error when the server runs, naming the path and what is wrong, and no review runs. `claude-runway-doctor <project>` reports the same problem (and exits 1).

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `REVIEWERS_FILE` | none, required | Path to your reviewers JSON file. A leading `~` is expanded; a relative path resolves against the server's working directory. |
| `MAX_REVIEWERS` | `3` | How many reviewers run at once. `1` runs them one at a time; size it to your machine and to LM Studio's Max Concurrent Predictions. |
| `MAX_REVIEW_ROUNDS` | `3` | The most rounds of Copilot feedback an unattended run goes through before a human is brought in. |
| `QDRANT_URL`, `QDRANT_API_KEY` | `http://localhost:6333`, none | For the scoring database, a later piece of work. |

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
      "type": ["classifier"], "loadOnDemand": true, "contextLength": 32768 }
  ]
}
```

`rubric[]` entries:

- `name`: unique identifier.
- `level`: one of the server's fixed levels, lowest to highest: `low`, `moderate`, `high`. Several entries can share a level.
- `definition`: what a finding is judged against.
- `definitionFile` (optional): a path (relative to the reviewers file) to a file holding a fuller definition in free text. It must be readable.
- `score`: a whole number from 1 to 5, the minimum confidence a classifier needs for a finding to match this entry.

`reviewers[]` entries:

- `name`: unique identifier.
- `model`: the served model id.
- `prompt` and/or `promptFile`: the reviewer's lens. One of the two is required. `promptFile` is a path (relative to the reviewers file) to a markdown file; it must be readable and not blank, and it is read again for each review, so editing it needs no restart. When both are set they are merged into one prompt: the `prompt` text first, then the file's text.
- `active`: turn a reviewer on or off.
- `type`: one or more of `reasoning` (produces findings), `verifier` (confirms or drops each finding) and `classifier` (matches a verified finding to a rubric entry).
- `loadOnDemand`: `true` when the server should load the model itself.
- `contextLength` (optional): the context the model is loaded with; used only when `loadOnDemand` is `true`.

Validation reports every problem in the file at once, not just the first: unknown keys (a misspelt `loadOnDemand` would otherwise silently mean something else), an unknown `level`, duplicate `name`s, a `score` outside 1 to 5, an unreadable `definitionFile`, and reviewers that share a model but disagree on `loadOnDemand` or `contextLength` (they are scheduled as one group, so they must agree). An active classifier with an empty rubric is also an error.

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
| `head_ref` | The end of the range; default `HEAD`. |
| `context_lines` | Lines of surrounding code shown around each change; default 20. The whole file is not sent, since it would not fit a small context. |
| `chunk_chars` | Characters of code per request. By default it comes from each reviewer's `contextLength` (8192 tokens is assumed when it is not set), at a low three characters per token. |

How the prompt is built, per reviewer: its own prompt, then the `.github/instructions/*.instructions.md` files whose `applyTo` matches a changed path (a file with no `applyTo` applies to every path; globs are comma-separated, with `*` inside one directory, a `**` segment across any number of directories, `?` for one character, and everything else literal, brackets and braces included; a glob over 512 characters is ignored, and matching takes bounded time however the glob is written), then any skill under `.claude/skills/` or `skills/` whose name contains "review", then the output format. Those instruction and skill files are found automatically in the repository under review, so each must be a regular file that resolves to a place inside that repository, and at most 80,000 bytes of it are read. A symlink is followed only when it points at a file git tracks (a link to a tracked rules file is fine; a link to `.env`, to anything under `.git`, to an untracked file or out of the repository is ignored, and so is every link when the directory is not a git repository). A `promptFile` you name in the reviewers file is your own choice and is read wherever it points. Each changed file is shown to a reviewer as its new version: a `File:` line, then every line with its real line number on the left, a `+` marking lines the change added or modified, and one `~ N line(s) removed here` marker where lines were deleted (removed code itself is never shown, so a model cannot mistake it for current code). A file too big for a reviewer's context is cut per hunk and then per line, never inside a line, and every piece starts with its `File:` line and keeps its real line numbers. A line too long for one request is split, and each later fragment repeats its line number with a `>` in place of the `+`. When a change only deletes lines and nothing of it survives beside the removal (`context_lines` of 0), the deleted lines are listed with a `-` after `~ N line(s) removed after line L`, so the reviewer can judge whether removing them was safe. Binary, deleted, renamed-only and mode-only files have no new text and are not sent; the result lists them in a warning so a skipped file is never silent. One limit to know: when a deletion has surviving lines beside it, the reviewer sees only that lines were removed, not what they were, so it cannot tell that a removed guard mattered; keep that in mind for changes that mostly delete code. A diff needing more than 100 requests per reviewer is refused: review a smaller range.

Every finding from every reasoning reviewer is kept: there is no vote threshold, so a finding only one reviewer saw is not lost. Findings are grouped by class. Findings in the same file and class whose line ranges overlap are shown as one entry, but two different claims are never collapsed into one: the same claim (ignoring case, spacing and a final full stop) lists every reviewer that raised it, and a different claim, or the same claim with a different failing input, is kept under it as an "also" line with its own reviewers. Each entry carries its file, line range, class, claim and the concrete failing input it describes. They are not verified or classified: those stages come later. A reviewer whose requests all fail is reported in the result and does not stop the others.

## Choosing prompts for your reviewers

What a reviewer finds depends mostly on how much its prompt asks it to cover at once, and the prompt is yours to set. The server does not change a reviewer's scope on its own; `list_reviewers` only tells you what your setup implies, under "Advisories".

- **One reasoning reviewer.** It has to cover every kind of defect by itself, so give it a broad prompt. A narrow one leaves the rest of the code unchecked.
- **Several reasoning reviewers.** Each can take one narrow lens (injection, off-by-one and empty input, shared state, and so on). Small models usually do better on one narrow question than on "find any defect": in a test on a small file with three unrelated defects, one general prompt on a small model found one of them, and three narrow prompts (two of them on the same small model) found all three.
- **Reviewers on the same model share one load.** Splitting a small model into several narrow lenses costs time, not memory.
- **Same model, same prompt.** At temperature 0 they raise the same findings, so one of them is wasted. The same prompt on two *different* models is a second opinion, and is not reported.
- **The "whole codebase" is never in front of one reviewer.** A reviewer sees one file's changes at a time, in a context of a few thousand tokens. A broad prompt widens what it looks for, not how much code it can see.
