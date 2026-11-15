# Live review harnesses

Scripts that run the codebase-reviewer against a **real LM Studio**. The unit
tests never do that, so they prove the plumbing and say nothing about review
quality. These answer the questions the unit tests cannot: does this model find
the bugs, how many false alarms does it raise, what did the loader and the
chunker actually do.

A development tool, not product code, and how the local review quality is measured (see Track F in `EVALUATION.md`). CI does not lint, type-check or test this folder, it is not
shipped in the wheel, and nothing here gates a PR. Run the scripts by hand:

```
.venv/bin/python harness/live_reasoning.py --model <lm-studio-id> --runs 3
```

| Harness                 | Stage                                                              | Status                                                           |
| :---------------------- | :----------------------------------------------------------------- | :--------------------------------------------------------------- |
| `live_reasoning.py`     | reasoning reviewers find defects in a diff                         | works                                                            |
| `live_classifier.py`    | classifier rates a verified finding against every rubric entry     | works                                                            |
| `live_verifier.py`      | verifier confirms or rejects a finding                             | works                                                            |
| `live_all_reviewers.py` | every configured reviewer over the four-file `orders` case, scored | runs reasoning, then the verifier, then the classifier |
| `live_routes.py`        | which reply route (`lines`, `json`, or a translator) a model returns reliably | works; reasoning stage only |

`live_classifier.py` and `live_verifier.py` take `--check` to validate and print
their fixtures without LM Studio; without it they run the stage.

## Checks to run: `live-tests/`

`live-tests/` runs the scripts for real against LM Studio models with explicit pass/fail checks (mechanics that must always hold, and quality baselines that catch a regression): `.venv/bin/python harness/live-tests/run_live_tests.py`. Each config in `live-tests/configs/` is a test, picked by name (`--list` shows them), and `/cr-gh-run-harness` does the choosing and running from Claude Code. See its README.

It starts with `live-tests/preflight/`: checks that need no LM Studio (exit codes and messages: reading a `--config`, choosing stages, refusing a bad combination). They take a few seconds, and if one fails no model is loaded. They can be run alone: `.venv/bin/python harness/live-tests/preflight/run_checks.py`.

## Answers live in `cases.py`

Each case is a base tree and a head tree. What is wrong in head is written down:

- a **bug** is a real defect; reporting it counts as finding it;
- a **decoy** is code that looks suspicious and is correct (a shell call with a
  constant command, a locked counter); a finding on one is a false alarm the
  verifier stage should reject;
- the `clean` cases (`clean`, and a `clean-` sibling of several seeded cases) have neither, so anything reported on them is a false positive.

A finding is matched to a bug by file and line (within three lines, because the
model reads line numbers off a diff), not by the class it chose. To add a case,
add a `Case` and a `marker` per bug: the marker is a string that occurs on
exactly one line, so the answer follows the code when it is edited.
Keep each marker unique: a marker that matches no line, or more than one, makes `cases.py` raise a `ValueError` when the case is scored.
Keep a decoy more than three lines from any other bug or decoy in its file, or a finding on one is credited to the other; `tests/test_harness_cases.py` checks the markers, the spacing, and that every case is stdlib-only Python.

### How big the case set is

**38 seeded bugs across 9 cases, 13 decoys, 5 clean cases, and 51 verifier fixtures** (38 real findings to confirm, 13 false ones to reject). It used to be 9 bugs in 2 cases, 3 decoys, 1 clean case and 9 fixtures, so any difference between two models or two reply formats rested on one seeded case. Read a comparison as resting on these sizes, and say them next to any number you report: on 38 bugs one bug is about 3 points of recall, on `seeded-three` it was 33.

| Case | Shape | What is wrong |
| :-- | :-- | :-- |
| `seeded-three` | one file | shell injection, off-by-one slice, division by zero |
| `orders` | four files | injection, path, off-by-one, formula, empty input, shared state, plus 3 decoys |
| `leaks` | one file | a file, a connection, a scratch directory and a child process not released on every path |
| `errors` | one file | bare `except`, a swallowed exception, the wrong exception type, `return` in `finally` |
| `concurrency` | two files | an unlocked update, check-then-act, a lock never released, iterating a dict others change; two thread-safe decoys |
| `inventory` | four files | a wrong comparison, a range past the end, float equality, `is` on a string, the wrong sort key |
| `handlers` | three files | shell, `pickle`, `eval`, SQL injection and a committed secret; decoys include a constant `shell=True` and a constant argument list, each with a hypothetical-environment attack |
| `guards-removed` | a deletion-only diff | three validation guards deleted; the diff has no added line |
| `paging` | one file | four off-by-one defects in pagination and chunking |
| `clean`, `clean-leaks`, `clean-errors`, `clean-concurrency`, `clean-guards-removed` | | the same code written correctly; every finding is a false alarm |

### How many runs a comparison needs

Runs and cases do different jobs, and the harness sends `temperature: 0`, so a repeat of the same case with the same model and prompt mostly returns the same reply. When the reply formats were first compared, the single seeded case and the 9 verifier fixtures gave identical repeats, and more runs added nothing. So **grow the cases before the runs**: a comparison on more cases can tell two models apart, a comparison on more runs of one case cannot.

Runs matter for one thing: the 95% readable gate of `live_routes.py`. The replies it counts are runs times cases, and one unreadable reply out of 10 is 90%, which fails, while one out of 20 is 95%, which passes. Send **at least 20 replies per route**, so a single bad reply cannot decide the verdict: runs = 20 divided by the number of cases, rounded up (2 cases: 10 runs, the default; 5 cases: 4 runs; all 14 cases: 2 runs). Use 3 runs when you want to see run-to-run spread for a model or setup that is not at `temperature: 0`.

A bar in a `*.baselines.json` (for example `min_real_confirmed: 4`) is a count over the case set it was measured on, and today that is `orders` (6 real fixtures), which is why `live_verifier.py` and the live tests default to it. Do not compare it with a run on another case set.

## A real repository with seeded bugs: the `harness-live-tests` branch

The cases in `cases.py` are tiny and self-contained: everything a reviewer needs to judge a finding is in the diff or the
lines quoted with it. That cannot measure the lookup loop (`tools` on a reviewer, see `docs/codebase-reviewer.md`), because
nothing ever needs another file. The branch `harness-live-tests` is the other half: a frozen copy of this repository, real code at
real size, with bugs seeded into real modules. Most of them can only be judged by reading code that the diff does not show.

**What it is.**

- `harness-live-tests` is an orphan branch with one commit, `cc54875`: a snapshot of `main` at `ea84b3e`, pruned. It keeps `libs/`,
  `mcpservers/`, `tools/`, `hooks/`, `tests/`, `claude_runway/`, `scripts/`, `templates/`, the packaging files and the repository's own review
  rules (`.github/instructions/`). It drops `harness/`, `docs/`, `skills/`, `.claude/`, `.githooks/`, the workflows, every other markdown
  file and the harness tests, because the tool executor reads every tracked file at the commit and a model must not be able to grep this
  answer key (or the older cases' answers) out of the tree. **The answer key is this section, on `main`'s side, never on the branch.**
- It is never merged or rebased. `main` moves on; this does not.
- Each change is ONE commit whose parent is the base, tagged `harness-live-tests/change-N`. Review `base...change-N`
  (`git diff harness-live-tests/base harness-live-tests/change-N`): the diff holds only that change's seeded edits, and the rest of the tree
  has no other seeded bug, so a finding can be scored against the rows below.
- Get it: `git fetch origin harness-live-tests 'refs/tags/harness-live-tests/*:refs/tags/harness-live-tests/*'`. A single-branch clone does not have it.
- Run a review against it with the repository path of any checkout that has the tags, for example
  `review(type="reasoning", params={"repo_path": "<path>", "base_ref": "harness-live-tests/base", "head_ref": "harness-live-tests/change-1"})`.
  Every read is of the commit, so the checked-out branch does not matter.

**Run types.** The two halves are the harness's two *run types*: `synthetic` (the seeded cases in `cases.py`) and `live` (this branch).
`live_reasoning.py` and `live_all_reviewers.py` take `--run-type synthetic|live`; the default is `synthetic`, so nothing changes without
it. The verifier and classifier scripts run fixtures built from `cases.py` and have no live mode: asked for `live` they say so and exit 2.
A live run takes `--config` and reads what it is scored against from the sidecar beside it, `<config>.baselines.json`, whose `runType` is
`live` and whose `metadata.tags` names each tag and, for each, the bugs and decoys (file, line, class, claim, evidence kind); the format is
in [live-tests/README.md](live-tests/README.md). The sidecar is on `main`; nothing is written on the branch. **The table below is a
separate copy for people, and no code reads it**: change an answer in both places. `--repo PATH` names a checkout that has the tags (default:
this one); before any model loads, a missing tag stops the run with the `git fetch` command above. The run reviews `base...<tag>` for
each tag in the sidecar and prints one block per tag, then one line per evidence kind (bugs found, decoy alarms) and one for the clean
control, which is how to tell whether a model whose proof lies in another file asks for it:

```
.venv/bin/python harness/live_reasoning.py --config my-config.json --run-type live --runs 2
.venv/bin/python harness/live_all_reviewers.py --config my-pipeline.json --run-type live   # reasoning, then the verifier and classifier, per tag
```

**Scoring rules.** A finding at a bug's line (or one that names the same defect) is a hit. A finding that matches a decoy's claim is a false
positive. A finding on `change-5` is a false positive (it is the clean control). **A finding that matches nothing below is unscored, not
wrong**: the code is real and may have real bugs nobody has listed. Do not count unscored findings against a model.

**Evidence kinds**, by where the proof lies, because that decides whether a reviewer must ask for more code:

| Kind | Meaning |
| --- | --- |
| `diff` | Everything needed is in the changed lines and the few lines around them. The control: a model that asks anyway is over-asking. |
| `same-file` | The proof is elsewhere in the same file, outside the changed lines. |
| `other-file` | The proof is in a different file (a definition, a caller, a setting, a guard). |

**The tags.** Hashes are recorded so a moved tag is noticed: `git rev-parse harness-live-tests/<tag>` must print the full hash.

| Tag | Commit | The change | Files |
| --- | --- | --- | --- |
| `base` | `cc54875cdcc8b4404fc2014e7caef7805f8e436d` | the pruned snapshot | |
| `change-1` | `6a7376613777a69a9191d855d6719cb54a7231f0` | name the index field once and tidy how a batch is built | `libs/qdrant_batch_store.py` |
| `change-2` | `ae363f14b651e2472c4c99f018fb3c80a943a3c1` | collapse whitespace and cap the length once; count spans as tuples | `libs/credential_redaction_lib.py`, `libs/local_compress_lib.py` |
| `change-3` | `3834f066c5afa94e7bcac4685dfcfd8c5c1d1a24` | build the manifest text first; record missed files that have no entry yet | `libs/sync_manifest_lib.py` |
| `change-4` | `ce60b6bce5692ce7665210904d1bfbaa36568787` | shorter cut marker for a clipped line; a flatter `_cut` | `libs/reviewer_tools_lib.py` |
| `change-5` | `a4a65fc06cd7b2f081353028c12f016a3128e953` | **clean control**: name the caught exception, reword comments | `libs/qdrant_retry.py`, `libs/cache_db.py` |
| `change-6` | `bcaca1c7ddceda8e7ba70617aa30c21bc6fa2019` | give `git describe` a little longer; fold the success test into one condition | `libs/version_lib.py` |

**Bugs** (line numbers are at the tag; the quoted line is what a finding should quote):

| ID | Tag | Where | The line | What is wrong, and an input that shows it | Evidence |
| --- | --- | --- | --- | --- | --- |
| B1 | `change-1` | `libs/qdrant_batch_store.py:99` | `make_point(embedding, vector_name, entry)` | The first two arguments are swapped: `make_point` takes `(vector_name, embedding, entry)`, so it builds `{embedding: vector_name}`, a dict keyed by a list. Any non-empty `entries` raises `TypeError: unhashable type`. | `other-file` (definition): `libs/qdrant_connector.py:48` is the signature, `:92` is a correct call. |
| B2 | `change-1` | `libs/qdrant_batch_store.py:48` | `FILE_PATH_INDEX_FIELD = "file_path"` | Points store the path nested, as `metadata.file_path`, and `sync_repo` deletes by that nested key. An index on the bare `file_path` is on a field no point has, so the index is created and used by nothing: per-file deletes still scan the whole collection. Run `ensure_file_path_index` on an existing collection, then a `sync_repo` delete. | `other-file` (config): `libs/qdrant_connector.py:53` nests the metadata under `METADATA_PATH`; `mcpservers/ingest_mcp_server.py:161` deletes by `metadata.file_path`. |
| B3 | `change-2` | `libs/credential_redaction_lib.py:177` | `text, _ = redact_credentials(text)` | Redaction now runs after the text is cut to `max_chars`, so a secret that straddles the cut is cut first and a fragment too short for the pattern survives. `"x" * 190 + " ghp_" + "A" * 40` leaves `ghp_AAAAA` in the output. | `same-file`: the function's docstring, at base lines 165-170 (`:167` in the base file), says redaction must run before whitespace collapsing and truncation, and why. |
| B4 | `change-2` | `libs/local_compress_lib.py:1023` | `return len({m.group(0) for m in credential_spans(text)})` | `credential_spans` returns `(start, end)` tuples, not match objects, so `.group` raises `AttributeError` as soon as the text holds one credential: `"token ghp_abcdef123456"`. | `other-file` (definition): `libs/credential_redaction_lib.py:123-125` (`return [m.span() ...]`); the caller at `libs/local_compress_lib.py:995` unpacks the tuples. |
| B5 | `change-3` | `libs/sync_manifest_lib.py:477` | `if rel not in baseline:` | `missed` comes from `unindexed_hashed_files`, which only returns files taken from `current_hashes`, so every missed file is already in `baseline`. With the test inverted none is ever marked `UNVERIFIED_HASH`, and a file that has text but was never chunked is recorded as up to date. | `same-file`: `libs/sync_manifest_lib.py:408-443` (`unindexed_hashed_files` iterates `current_hashes`) and the `index_baseline` docstring. |
| B6 | `change-4` | `libs/reviewer_tools_lib.py:100` | `return f"{line[:MAX_LINE_CHARS]} ...[cut {len(line) - MAX_LINE_CHARS} chars]"` | `has_line_cut` matches only ` ...[line cut: N more characters]`, so the new marker is never recognised: a clipped line no longer sets `truncated`, and a model is told a result is complete when a long line was cut. `read_file` on a file with a 500-character line. | `same-file` (a contract between two functions): `libs/reviewer_tools_lib.py:91` (`_LINE_CUT`) and `:103` (`has_line_cut`); `tests/test_reviewer_tools_lib.py:251` pins the old marker. |
| B7 | `change-6` | `libs/version_lib.py:35` | `return out if result.returncode != 0 and out else None` | The success test is inverted: `git describe` exits 0 on success, so a real result is thrown away and a failed run that printed something is returned. Inside any checkout `get_version()` falls back to the installed distribution's version. | `diff`: the line itself and `git describe`'s exit-code convention. |

**Decoys** (code that looks wrong in the diff and is not; the claim is what a careless reviewer would write):

| ID | Tag | Where | The line | The careless claim | Why it is fine | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | `change-1` | `libs/qdrant_batch_store.py:91` | `await connector.ensure_collection_exists(collection_name)` | Two processes can both find the collection missing and both create it, so one `store_batch` fails. | `ensure_collection_exists` already handles a concurrent creator: if the collection exists after a failed create it checks the schema matches and carries on. | `other-file` (guard): `libs/qdrant_connector.py:129-158`. |
| D2 | `change-3` | `libs/sync_manifest_lib.py:120` | `_atomic_write_text(repo / MANIFEST_FILENAME, text)` | The write is not atomic, so a crash leaves a truncated manifest that the next sync trusts. | The write is atomic: `_atomic_write_text` writes a temp file, fsyncs it and renames it over the manifest with `os.replace`, so a reader sees the old manifest or the new one, never a truncated file. And `load_manifest` returns `{}` on bad JSON, bad encoding or an `OSError`, so even a damaged file means a full rebuild. | `same-file` (guard): `libs/sync_manifest_lib.py:130` (`_atomic_write_text`) and `:87-96` (`load_manifest`). |
| D3 | `change-4` | `libs/reviewer_tools_lib.py:126` | `if newline > 0:` | When there is no newline before the limit the text is cut mid-line with no notice. | Every line is clipped to `MAX_LINE_CHARS` (300, line 55) first, so no line reaches the 8000-character limit; and `read_file` and `grep` add a `[cut: ...]` note whenever `_cut` reports a cut. | `same-file` (guard): `libs/reviewer_tools_lib.py:55` and the `[cut: ...]` notes in `_tool_read_file` and `_tool_grep`. |
| D4 | `change-6` | `libs/version_lib.py:30` | `timeout=10,` | A slow `git describe` raises `TimeoutExpired`, which escapes. | `TimeoutExpired` is a `subprocess.SubprocessError`, caught by the `except` a few lines below. | `diff` (guard): the `except` is in the same hunk. |

Also in the commits, not scored: `change-2` adds one blank line in `redact_credentials` (no behaviour change). `change-5` changes no
behaviour at all: `except Exception as e` becomes `as exc` in both retry functions, and two comments and one docstring are reworded.

**Totals.** 7 bugs: 1 `diff`, 3 `same-file`, 3 `other-file`. 4 decoys: 1 `diff`, 2 `same-file`, 1 `other-file`. One clean change.

**Adding to it.** Never move or re-create a tag: that changes what a recorded score was measured on. To add a change, commit it on the
base in a scratch worktree, tag it `change-N+1`, push the tag, and add its rows and hash here in the same commit on `main`.

**Not built yet.** A loader that scores a run against these rows, and a live config for it. Today the table is read by a person: run a
review of a tag, then compare. When the loader exists it should read the table's facts from a data file, not from this prose.

## Parity with Copilot: replay history, never ask for a review

The aim is to match or beat Copilot, so its past findings are the answer key. The harness
never asks Copilot for anything: it does not request, tag or re-request a review.
`collect_copilot_history.py` only reads (`gh pr view`, two `gh api` GETs and a `git fetch` of the PR head).

```
.venv/bin/python harness/collect_copilot_history.py 552          # read-only: GETs plus a git fetch of the PR head
.venv/bin/python harness/live_reasoning.py --model <id> --history harness/history/pr-552.json
```

`collect_copilot_history.py` records the PR's existing Copilot reviews and inline comments as
`harness/history/pr-<N>.json` (one round per Copilot review, each with the commit it reviewed).
`--history` then reviews the diff exactly as it stood at that commit and scores which of
Copilot's findings the local reviewers also raised, by file and line. `--round N` picks a later
round. Findings Copilot did not raise are reported as "further findings", unjudged: Copilot
misses things too, and sorting real from false is the verifier stage's job.

`harness/history/` is not committed (it is regenerated, and git ignores it), so a clone starts with
none. `collect_balanced_history.py` rebuilds a spread rather than a pile:

```
.venv/bin/python harness/collect_balanced_history.py --dry-run   # show the pick, record nothing
.venv/bin/python harness/collect_balanced_history.py             # pick 9 of the last 150 merged PRs and record them
```

It scans the recent merged PRs Copilot reviewed and picks them by a fixed rule: every kind of change
(docs, shell, Python, other), size (small, medium, large), number of Copilot rounds (1-2, 3-5, 6 or more)
and number of inline findings (1-2, 3-5, 6 or more) is covered before any is repeated, and the same PRs
always give the same set. `--count`, `--scan` and `--exclude <PR>` change the size of the set, the PRs it
looks at and the ones it skips; a PR that was retargeted cannot be replayed and is replaced. It only reads,
like the collector. To add one PR by hand, use `collect_copilot_history.py <N>`.

Snapshots only hold what the inline comments say. Findings that appear only in a review's body
(the "previously missed" items) are kept verbatim but not scored.

## Choosing reviewers and stages: `--config`

`live_reasoning.py`, `live_verifier.py`, `live_classifier.py` and `live_all_reviewers.py` take `--config FILE`, a JSON file that says which models and
prompts to test and which stages to test them in. A comparison becomes a file you can keep, diff and re-run
instead of a long command line.

```
.venv/bin/python harness/live_all_reviewers.py --config harness/config.example.json --runs 3
.venv/bin/python harness/live_reasoning.py --config my-config.json --show-config   # what it resolves to; needs no LM Studio
```

The file has three optional keys:

| Key         | Meaning                                                                                                                                                                                                                          |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reviewers` | The reviewers to test, exactly as in a reviewers file: `name`, `model`, `prompt` and/or `promptFile`, `active`, `type`, `loadOnDemand`, `contextLength`, `replyFormat`, `translator`. It is checked by the server's own validator, so a file that passes here is one `REVIEWERS_FILE` accepts. |
| `rubric`    | The rubric the classifier matches against, as in a reviewers file. Needs `reviewers`.                                                                                                                                              |
| `stages`    | Which stages to test: a non-empty combination of `reasoning`, `verifier` and `classifier`. Left out, it is every stage an active reviewer has a `type` for (all three when the file has no reviewers).                          |

- **A reviewers file is a valid config as it stands**: it just has no `stages`, so every stage it has reviewers for is tested.
- **Stages.** One left out of `stages` is not run, even when reviewers are configured for it. Selecting one that no active reviewer has the `type` for is an error. A config written for the whole pipeline runs every stage.
- **Which reviewers run.** `live_reasoning.py` runs every active reviewer with the `reasoning` type over each case, instead of the single reviewer `--model` and `--prompt` build, and `live_verifier.py` and `live_classifier.py` do the same for the `verifier` and `classifier` types. `live_all_reviewers.py` takes its reviewers, rubric and stages from the file instead of its three default lenses.
- **Flags.** The file names each reviewer's model, prompt and loading, so `--model`, `--prompt`, `--prompt-file`, `--context`, `--reply-format`, `--translator` and `--resident` are refused alongside it. Everything else (`--runs`, `--case`, `--history`, `--chunk-chars`, `--timeline`, ...) still applies. Without `--config` nothing changes.
- **Errors.** Every problem in a bad file is reported at once, naming the file: unreadable, not JSON, an unknown key, an unknown or repeated stage, an empty `reviewers` list (name a reviewer, or leave the key out to use the flags), plus whatever the reviewers-file validator finds.
- **A verifier that shares its entry with another type and has a `promptFile` is refused.** The harness runs one stage at a time, with each reviewer cut down to that stage's type, and such a verifier's prompt depends on its other types: the server appends the built-in verify steps after the file, and a verifier-only copy would get the file alone, so a result here would not predict the server's. The error says to give the verifier its own entry (the same model and a different name). Several types without a `promptFile`, or without the `verifier` type, are fine.
- **`--show-config`** prints the stages and reviewers a run resolves to (model, roles, how the model is loaded, where the prompt comes from) and exits without touching LM Studio.

`config.example.json` shows the shape: two narrow reasoning lenses on one model, a verifier on a reasoning-distilled model, and a classifier with a small rubric. Change the model ids to ones `lms ls` shows. `stages` selects all three.

## The verifier stage

The verifier takes a finding and the code it cites, and confirms it with a concrete failing input and the wrong result, or rejects it (see [Verifying findings](../docs/codebase-reviewer.md#verifying-findings)). Before any model sees a finding, a string search checks that the code it quotes is in the file; a finding about code that is not there is dropped as ungrounded. The reviewers it runs are the config's active reviewers with the `verifier` type, or one built from `--model` (and `--prompt`, default a one-line verifier lens) when no config is given.

```
.venv/bin/python harness/live_verifier.py --check                                  # the fixtures only; needs no LM Studio
.venv/bin/python harness/live_verifier.py --model deepseek-r1-distill-qwen-7b@4bit --runs 3
.venv/bin/python harness/live_verifier.py --config harness/config.example.json --show-config   # needs no LM Studio
.venv/bin/python harness/live_verifier.py --config my-config.json --runs 3
```

`live_verifier.py` gives the verifier each fixture of the cases chosen with `--case` (every seeded bug as a true finding, every decoy as a false one; `--case` can be repeated, and `--case all` takes every case that has answers, 51 fixtures; the default is `orders`, 9 fixtures, the set the verifier baselines were measured on) and reports the two numbers that matter: **real findings wrongly rejected** (a verifier that loses real bugs) and **false findings wrongly confirmed** (a verifier that lets decoys through). Anything disputed, unchecked or ungrounded is listed apart. Every fixture is shown with the same class, and a decoy carries a plausible failing input as a real finding does, so neither gives the answer away. A wrong verdict is printed with what each verifier said, so you can read why. It prints a `cases:` line naming the cases it ran, which `live-tests/run_live_tests.py` compares with the cases a baseline was counted on (a baselines file's `runType` and `metadata.cases`, see [live-tests/README.md](live-tests/README.md)), so a bar counted on `orders` is never applied to a run over every case.

A config whose `stages` leave out `verifier` makes it exit 0 without loading a model, and one that selects `verifier` with no active reviewer of that type is refused with a message naming the problem. Only the verifier models are checked up front. That check gives a verifier 2000 tokens to end its turn instead of 40, because a reasoning-distilled model thinks before it answers.

`live_all_reviewers.py` runs the verifier after reasoning whenever the config has an active verifier reviewer and `verifier` is a selected stage: each run's findings are grounded and verified, and it prints the score **before** and **after** verification, which is where the decoy false alarms should disappear. A reasoning-distilled model is the recommended verifier (see "Code Verifier Models" below).

## The classifier stage

The classifier takes a finding that stands after verification and the rubric (each entry's name, level and definition, never its `score`) and, one entry at a time, answers whether the finding falls under that definition (see [Classifying findings](../docs/codebase-reviewer.md#classifying-findings)). The answer is the model's probability of "yes", read from its token probabilities at temperature 0, so the model must return them. The probabilities of every classifier are averaged: an entry matches when its average is at or above its `score`, a probability from 0 to 1. A finding that matches nothing is dropped as unclassified, and one that fits several takes the highest `level`. The classifier reads the finding's own text and never the code, so the harness needs no repository for it. The reviewers it runs are the config's active reviewers with the `classifier` type, or one built from `--model` (and `--prompt`, default a one-line classifier lens) when no config is given. The rubric is the config's `rubric` when it has one, and `cases.RUBRIC` otherwise; a config with an active classifier and no rubric is refused by the reviewers-file validator.

```
.venv/bin/python harness/live_classifier.py --check                                  # the fixtures only; needs no LM Studio
.venv/bin/python harness/live_classifier.py --model qwen2.5-coder-7b-instruct-mlx --runs 3
.venv/bin/python harness/live_classifier.py --config harness/config.example.json --show-config   # needs no LM Studio
.venv/bin/python harness/live_classifier.py --config my-config.json --runs 3
```

`live_classifier.py` gives the classifier each of the 16 fixtures in `cases.classifier_cases()` (every seeded bug, a style-only remark, three claims that fit no entry, one that fits two, and borderline claims that only partly resemble an entry) and reports what matters for a rubric: **the share on the expected entry** (and each miss), whether the claims that fit nothing came back **unclassified**, whether the claim that fits two entries **took the higher level**, and **the average probability each entry got against that entry's `score`**. When no average falls below its `score` the report says **the scores dropped nothing**: the model answered yes to everything. Every fixture is shown with the same class so the class does not give the answer away. Findings nobody answered are listed apart, and are never credited: a claim that fits nothing counts as correctly dropped only when it came back unclassified, not when it simply got no answer. A fixture that expects an entry the config's rubric does not have is left out of the score, with a line saying so.

The probabilities are read at temperature 0, not sampled, so a fixture should give the same average for the same entry in every run. With `--runs` above 1 the report ends with **score repeat**: for each fixture and entry, whether its average came out the same over the runs (within `REPEAT_TOLERANCE`, 0.1); each one that did not is named.

A config whose `stages` leave out `classifier` makes it exit 0 without loading a model, and one that selects `classifier` with no active reviewer of that type is refused with a message naming the problem. Only the classifier models are checked up front.

`live_all_reviewers.py` runs the classifier after the verifier whenever the config has an active classifier reviewer and `classifier` is a selected stage: the findings the verifiers did not reject (every finding when there is no verifier) are checked against the rubric, and each is printed with its rubric level and average probability. See "Code Classifier Models" below for the builds that have been measured.

## Reply routes: which way a model's reply reaches the server

Every reasoning reply has to reach the server as `lines` or `json`. A model has three ways to get there: it replies in `lines`, it replies in `json`, or it replies in its own format and a `translator` converts that (a `promptFile` written for the model is part of any route). `live_routes.py` finds, per model, the first route that works reliably:

```
.venv/bin/python harness/live_routes.py --model <lm-studio-id>                      # lines and json, 10 runs of each case
.venv/bin/python harness/live_routes.py --model a --model b --runs 5                # each model in turn
.venv/bin/python harness/live_routes.py --model <id> \
    --route "native=lines,promptFile=my-prompt.md,translator=my-translator.py"       # also try a custom route
.venv/bin/python harness/live_routes.py --model <id> --show-config                  # what it would run; needs no LM Studio
```

The routes are `lines` and `json` (the server's own reply instructions), then each `--route` in the order given; `--no-builtin` leaves the first two out. A route is `NAME=FORMAT`, plus optional `promptFile=PATH` and `translator=PATH`. FORMAT is how the reply, or the translator's output, is read, and a prompt file replaces the built-in reply instructions, so it has to ask for FORMAT itself. The cases default to `seeded-three` and `clean`.

For each route it prints the requests sent, the requests LM Studio never answered, the replies that could be read, and why the rest could not (not parseable as the format, or a translator failure). **A route is reliable when 95% or more of the replies it got were readable**, and the verdict names the first reliable route. A request that got no reply at all is not counted against the format; it is listed apart. Bugs found and false alarms are printed for information and never decide the verdict.

Because unanswered requests leave the denominator, a model that stops answering can still show 100%: read the "no reply" note under the table too. the 8-bit MLX build of Llama-3.1-8B (`mlx-community/Meta-Llama-3.1-8B-Instruct-8bit`, since dropped) did exactly that on `lines` (10 of 44 requests unanswered, 34 of 34 readable). The Llama-3.1-8B build this repo keeps is the 4-bit one, which had 2 unanswered.

## Before it spends minutes

Each script asks the model one question first: reply with one word, capped at 40
tokens. A model that does not end its turn runs to the context limit on every
request and times out, which on a real run looks like nothing but 60 second
timeouts. The probe catches it in seconds and says which model. Typical cause: a
base model downloaded instead of its `-Instruct` build. `--no-probe` skips it.

A model with a classifier reviewer is also asked one yes/no question with an obvious
answer, the way the classifier asks them. Ending its turn is not enough: Phi-3.5-mini
does, and then begins every yes/no reply with something other than `Yes` or `No`, so
no probability can be read. The probe stops the run, naming the model and the first
tokens it began with, and says to check the model's chat template in LM Studio or use
another model. A model that returns no token probabilities is refused the same way.

## Useful options

| Option                                 | Meaning                                                                        |
| :------------------------------------- | :----------------------------------------------------------------------------- |
| `--config FILE`, `--show-config`       | reviewers, rubric and stages from a JSON file, and what it resolves to (see above) |
| `--model`, `--context`                 | model id as `lms ls` shows it, and the context to load it with                 |
| `--prompt`, `--prompt-file`            | the reviewer's lens; compare prompts on identical input. A prompt file replaces the built-in reply instructions: `prompts/` has ready-made ones |
| `--reply-format`, `--translator`       | how the reasoning reply is read (`lines`, the default, or `json`), and a script that converts the model's own format into it |
| `--runs N`                             | repeat; one run proves little                                                  |
| `--chunk-chars N`, `--context-lines N` | chunk size and lines of code around each change                                |
| `--resident`                           | the model is already loaded; do not load or unload it                          |
| `--timeline`                           | every REST call and completion request, in order                               |
| `--case own-diff`                      | this checkout against `origin/main`, unscored                                  |
| `--case own-planted`                   | the same, in a throwaway clone with one shell injection planted in a real file |

Models the harness loads are unloaded again; models you had loaded are left alone.

For local deployment within a **3 GB to 9 GB** memory or VRAM footprint, here are the recommended **Instruct-tuned** models. These models include built-in chat templates (like ChatML or Llama-3) and pre-configured stop tokens, ensuring they answer your prompts directly without endlessly hallucinating text.

### 1. Codebase-reviewer models

The models behind the `codebase-reviewer` need to hold large contexts (e.g., git diffs, pull request comments, project style guides) and communicate structured, actionable suggestions in idiomatic English.

When you report or compare a result, name the build (format and quant, for example "Llama-3.1-8B 4-bit MLX"), not just the family: LM Studio ids do not carry the quant reliably, and an id can change when a second build of the same model is installed. `prompts/README.md` has a table mapping each config to its LM Studio id, source repository and quant.

- **Qwen2.5-Coder-7B-Instruct**
  - **MLX Repo (the build measured):** `lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit` (\~4.3 GB); `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` is a different conversion and was not measured
  - **Higher-Precision MLX:** `mlx-community/Qwen2.5-Coder-7B-Instruct-8bit` (\~7.9 GB)
  - **GGUF Alternative:** `bartowski/Qwen2.5-Coder-7B-Instruct-GGUF` (`Q4_K_M` \~4.7 GB / `Q5_K_M` \~5.3 GB)
  - **Embedded Template:** Native ChatML (`<|im_start|>`, `<|im_end|>`)
  - **Reply format and prompt:** `lines`, built-in reply instructions, no `promptFile`, no `translator` (configs `reasoning-qwen-7b-4bit` and `reasoning-qwen-7b-8bit`). Measured on 38 seeded bugs over 14 cases, 2 runs: all replies readable. `json` also read (95% or so) and found more bugs with more false alarms; see `prompts/README.md`.
  - **Prompt files:** configs `reasoning-qwen-7b-4bit-promptfile` and `reasoning-qwen-7b-4bit-promptfile-example` run the 4-bit build with `prompts/reasoning-lines.md` and `prompts/reasoning-lines-example.md` as the whole prompt (a `promptFile` replaces the built-in reply instructions), so a real model is checked reading the lines format through a prompt file. Mechanics checks only, no baselines. No live test uses a `translator`: no measured model needs one.
  - **Why it works out of the box:** Unlike the Base variant, the `-Instruct` checkpoint contains pre-configured stop criteria in its tokenizer config. It handles multi-file diffs and refactoring without hallucinating follow-up text.

- **Llama-3.1-8B-Instruct**
  - **MLX Repo (the build measured):** `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (\~4.5 GB); `mlx-community/Llama-3.1-8B-Instruct-4bit` is a different 4-bit conversion that skipped findings and failed the live test, so it is not recommended
  - **GGUF Alternative:** `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (`Q4_K_M` \~4.9 GB)
  - **Embedded Template:** Llama-3 (`<|start_header_id|>`, `<|eot_id|>`)
  - **Reply format and prompt:** `lines`, built-in reply instructions, no `promptFile`, no `translator` (config `reasoning-llama-3.1-8b-4bit`). Measured on 38 seeded bugs over 14 cases, 2 runs: 42 of 44 requests answered and every reply that came back readable, and it passes the live test's mechanics checks. It raised 26 false alarms per run on `lines`, against 5 for either Qwen build (the fewest), 11 to 21 for the Llama-3.2-3B builds and 32 for Seed-Coder-8B (the most, with the most bugs found, 24 of 38), so prefer the Qwen builds when fewer false alarms matter. The 8-bit MLX build and the other 4-bit conversion did worse and were dropped; see `prompts/README.md` for each build's source and numbers.
  - **Why it works out of the box:** Highly aligned for stylistic critique and conversational feedback. It adheres strictly to custom review rules (e.g., "only highlight severity P0/P1 bugs and do not touch formatting").

### 2. Code Classifier Models

For each finding the classifier answers, one rubric entry at a time, whether the finding falls under that entry's definition, and its answer is the model's probability of "yes" read from its token probabilities. A classifier therefore has to return token probabilities (the reasoning-distilled DeepSeek builds, whose reply starts with their thinking, do not), and it has to rate the entries on a scale the rubric's `score` cut-offs fit. Only builds that have been run through the classifier fixtures (`harness/cases.py`: 16 findings against 4 rubric entries, `score` 0.5, 0.5, 0.5 and 0.75) are listed, each with its result; a model is added here after its result is, not before. Every figure below was measured with `live_classifier.py --runs 2` and was the same in both runs, with no entry average differing between the runs by more than the repeat tolerance.

- **Qwen2.5-Coder-7B-Instruct** (4-bit MLX): **13 of 16** on the expected entry.
  - **MLX Repo (the build measured):** `lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit` (\~4.3 GB), LM Studio id `qwen2.5-coder-7b-instruct-mlx`
  - **Embedded Template:** Native ChatML
  - **Measured:** the three claims that fit nothing came back unclassified, the finding that fits two entries took the higher level, and the three misses were `division-by-zero` and `empty-average` (`crash-on-input`, whose probability stayed under 0.5) and `weak/partial-wrong`. Config `classifier-qwen-7b-4bit` passes all of its live-test checks, and so does the pipeline config that classifies with it.

- **Qwen2.5-Coder-14B-Instruct** (GGUF Q4_K_M): **13 of 16**.
  - **GGUF Repo (the build measured):** `lmstudio-community/Qwen2.5-Coder-14B-Instruct-GGUF`, LM Studio id `qwen2.5-coder-14b-instruct`
  - **Measured:** the same two `crash-on-input` misses as the 7B and `weak/vague-name`; the overlap finding took the higher level.

- **Granite-3.3-8B-Instruct** (4-bit MLX): **10 of 16**.
  - **MLX Repo (the build measured):** `mlx-community/granite-3.3-8b-instruct-4bit`, LM Studio id `granite-3.3-8b-instruct`
  - **Measured:** three more misses than the Qwen builds, and the overlap finding went to `crash-on-input`, the lower level.

- **Qwen3-4B-Instruct-2507** (4-bit MLX, the instruct build, not the thinking one): **12 of 16**.
  - **MLX Repo (the build measured):** `lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit` (\~2.3 GB), LM Studio id `qwen3-4b-instruct-2507-mlx`
  - **Measured:** the same numbers in both runs. The three claims that fit nothing came back unclassified. It answered "no" to both injection claims (`orders/sql-injection`, `orders/path-traversal`), so those and the overlap finding came back unclassified, and `weak/vague-name` missed as well.

### 3. Code Verifier Models

Code verification requires explicit logic-tracing to catch off-by-one errors, race conditions, or unhandled null branches. These reasoning-distilled models generate hidden thinking chains before issuing an assessment.

- **DeepSeek-R1-Distill-Qwen-7B**
  - **MLX Repo:** `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` (\~4.4 GB)
  - **Higher-Precision MLX:** `mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit` (\~7.8 GB)
  - **GGUF Alternative:** `bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF` (`Q4_K_M` \~4.7 GB)
  - **Embedded Template:** Reasoning ChatML (outputs `<think>...</think>` before the answer)
  - **Why it works out of the box:** Distilled from DeepSeek-R1 onto Qwen2.5. Traces execution paths step-by-step and automatically stops generation once the final verdict is reached.

- **DeepSeek-R1-Distill-Llama-8B**
  - **MLX Repo:** `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit` (\~4.9 GB)
  - **GGUF Alternative:** `bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF` (`Q4_K_M` \~4.9 GB)
  - **Embedded Template:** Reasoning Llama-3 format
  - **Why it works out of the box:** Combines Llama-3 structure with reasoning tokens. Suitable for synthesizing edge-case unit tests (e.g., Hypothesis/PyTest) to falsify suspect code.

### Hugging Face Direct Download Reference

| Role           | Hugging Face Repository (MLX unless marked GGUF)  | Parameters | Precision / Quant | Memory Footprint | Embedded Template |
| -------------- | ------------------------------------------------- | ---------- | ----------------- | ---------------- | ----------------- |
| **Reviewer**   | `lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit` | 7B   | 4-bit             | \~4.3 GB         | ChatML            |
| **Reviewer**   | `mlx-community/Qwen2.5-Coder-7B-Instruct-8bit`    | 7B         | 8-bit             | \~7.9 GB         | ChatML            |
| **Reviewer**   | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`   | 8B         | 4-bit             | \~4.5 GB         | Llama-3           |
| **Classifier** | `lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit` | 7B   | 4-bit             | \~4.3 GB         | ChatML            |
| **Classifier** | `lmstudio-community/Qwen2.5-Coder-14B-Instruct-GGUF` (GGUF) | 14B | Q4_K_M       | \~8.4 GB         | ChatML            |
| **Classifier** | `mlx-community/granite-3.3-8b-instruct-4bit`      | 8B         | 4-bit             | \~4.3 GB         | Granite           |
| **Classifier** | `lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit` | 4B      | 4-bit             | \~2.3 GB         | ChatML            |
| **Verifier**   | `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`  | 7B         | 4-bit             | \~4.4 GB         | Reasoning ChatML  |
| **Verifier**   | `mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit`  | 7B         | 8-bit             | \~7.8 GB         | Reasoning ChatML  |
| **Verifier**   | `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit` | 8B         | 4-bit             | \~4.9 GB         | Reasoning Llama-3 |

### Quick Verification Test

To confirm that a downloaded Instruct model terminates correctly when tested via Python, load the model with its native chat template applied using this script:

```python
from mlx_lm import load, generate

# Ensure you use the '-Instruct' version!
repo_id = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
model, tokenizer = load(repo_id)

messages = [
    {"role": "system", "content": "You are a code reviewer. Reply with one concise paragraph."},
    {"role": "user", "content": "Review this function:\ndef add(a, b): return a + b"}
]

# Applies the embedded ChatML template and stop tokens
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512,
    verbose=False
)

print(response)
```
