# Reviewer prompt files

Markdown prompts for the reasoning stage of the `codebase-reviewer` server, used through a reviewer's `promptFile`
(in a reviewers file or a harness `--config`), or with a harness script's `--prompt-file`. They exist so that what a
model is told can change as we learn what suits it, with no code change: a `promptFile` is read again on every review.

## What a `promptFile` does

- It is **merged** with the reviewer's inline `prompt` text into the reviewer's lens: the inline text first, then the
  file. So a file can carry the lens, examples and extended instructions.
- It **replaces the built-in reply instructions.** Without a `promptFile` the server ends the prompt with the built-in
  instructions for the reviewer's `replyFormat`; with one it adds none. The file has to tell the model how to reply,
  in the format `replyFormat` selects (`lines` by default), because the reader is strict. A model whose own format is
  neither can be given a `translator`: a script that turns its reply into the `replyFormat` format. No model measured so
  far needs one, so no live test uses one yet.
- The repo's own rules and the description of how the diff is shown (`INPUT_FORMAT`) are still added, **after** the
  file. So the reply instructions in a file come before them, where the built-in text comes last; if that placement
  turns out to matter for a model, record it below.
- A reviewer that is both a reasoning and a verifier reviewer shares the file, but the verifier adds its own JSON verdict
  instructions after it. Give such a model two reviewer entries instead, one per role.

## The files

| File | What it is |
|---|---|
| `reasoning-lines.md` | The lens used in the first reply-format experiments, then the plain-lines reply instructions, verbatim. A reviewer using it behaves like one with no `promptFile` and `replyFormat: lines` (apart from where the reply text sits, above). Copy it to start a variant. |
| `reasoning-lines-example.md` | The same, plus one worked example reply whose quote contains double quotes. |

A test keeps `reasoning-lines.md` equal to the built-in text, so the copy cannot drift.

## What we have learned about models and prompts

Add a row for every measurement, with the case set and the number of runs, so nobody reads a small sample as stronger
than it is. These first rows are the first reply-format experiments: one seeded case plus a clean one, two runs each, so they show that
models differ and not which one is best.

| Model | Reply format and prompt | Result | Date |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct (4-bit, 8-bit) | json, built-in; lines, built-in | readable either way | 2026-10-29 |
| Llama-3.1-8B-Instruct | json, built-in | not readable (code pasted unescaped into a JSON string) | 2026-10-29 |
| Llama-3.1-8B-Instruct | json with escaping rules and two worked examples; lines, built-in | readable | 2026-10-29 |
| Llama-3.2-3B-Instruct (4-bit, 8-bit) | json, built-in; lines, built-in | readable either way | 2026-10-29 |
| Llama-3.2-3B-Instruct (4-bit, 8-bit) | json with examples | regressed on the clean case; XML-style tags dropped recall to 0% | 2026-10-29 |
| Llama-3.2-3B-Instruct GGUF | json, built-in; lines, built-in | json not readable; lines readable | 2026-10-29 |
| Seed-Coder-8B-Instruct (Q4_K_S) | json, built-in; json with examples; lines, built-in | json invalid on the clean case, json with examples invalid on the seeded case; lines readable | 2026-10-29 |
| Qwen2.5-Coder-7B-Instruct (4-bit MLX, `lmstudio-community`) | lines through a `promptFile` (`reasoning-lines.md`; `reasoning-lines-example.md`), against the same lens inline with built-in reply instructions | all replies readable by every variant (4 requests each, no reviewer error); `seeded-three` 1 of 3 bugs (the shell injection, 2 of 2 runs) every way; the clean case drew 0 false alarms with either prompt file and 2 with the inline prompt. 2 runs, `seeded-three` and `clean` only, `temperature: 0`, so a pointer and not a result; the file puts its reply instructions before the diff description and the built-in text puts them last, so this does not isolate which of the two caused the difference | 2026-11-01 |

### Reply route per model, measured 2026-10-31

Measured with `live_routes.py` on 14 cases (the 9 seeded cases holding 38 bugs and 13 decoys, plus the 5 clean ones), 2 runs, so 44 requests per route, at `temperature: 0`, with each model's built-in reply instructions and the same one-paragraph lens, loaded at a context of 8192. No `promptFile` and no `translator` was used or needed. "Unanswered" requests got no reply from LM Studio and are not counted against a format. "False alarms" are findings on a decoy plus findings that match no seeded bug, so they overstate real false alarms.

**Name the build, not just the model.** A model family comes in several quantizations and several conversions of the same weights, and they behave differently (the Llama-3.1-8B builds listed after the table are the clearest case). LM Studio's id for a build can also change: it was `meta-llama-3.1-8b-instruct` while one Llama-3.1 build was installed and became `meta-llama-3.1-8b-instruct@8bit` once a second was added, and back to `meta-llama-3.1-8b-instruct` once that second build was removed. Two 4-bit conversions of the same model also behaved differently. So re-check an id with `lms ls --json` before trusting it. So every row says the quant, and this table gives the exact source of each build as `lms ls --json` reports it (`path` and `quantization`). Use it, not the family name, to say which model a number belongs to.

The builds are split by where they run. **MLX runs on Apple Silicon only; GGUF runs on any platform LM Studio runs on, Apple Silicon included.** A machine without Apple Silicon can only reproduce the GGUF rows, and an MLX number does not describe the GGUF build of the same model (the two Llama-3.2-3B 4-bit rows below differ: 19 of 38 bugs and 21 false alarms per run as MLX, 15 and 11 as GGUF). An id is not a format: the id in a row is what `lms ls` showed for the build in that row, and the same id on another machine may load a different build.

**Apple Silicon only (MLX)**

| Config | LM Studio id | Source (`path`) | Format | Quant | Size |
|---|---|---|---|---|---|
| `reasoning-mlx-qwen-7b-4bit` | `qwen2.5-coder-7b-instruct-mlx` | `lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit` | MLX | 4-bit | 4.30 GB |
| `reasoning-mlx-qwen-7b-4bit-promptfile` | `qwen2.5-coder-7b-instruct-mlx` | `lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit` | MLX | 4-bit | 4.30 GB |
| `reasoning-mlx-qwen-7b-8bit` | `qwen2.5-coder-7b-instruct` | `mlx-community/Qwen2.5-Coder-7B-Instruct-8bit` | MLX | 8-bit | 8.10 GB |
| `reasoning-mlx-llama-3.1-8b-4bit` | `meta-llama-3.1-8b-instruct` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | MLX | 4-bit | 4.53 GB |
| `reasoning-mlx-llama-3.2-3b-4bit` | `llama-3.2-3b-instruct@4bit` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | MLX | 4-bit | 1.82 GB |
| `reasoning-mlx-llama-3.2-3b-8bit` | `llama-3.2-3b-instruct@8bit` | `mlx-community/Llama-3.2-3B-Instruct-8bit` | MLX | 8-bit | 3.42 GB |

**Any platform (GGUF)**

| Config | LM Studio id | Source (`path`) | Format | Quant | Size |
|---|---|---|---|---|---|
| `reasoning-gguf-seed-coder-8b` | `seed-coder-8b-instruct` | `unsloth/Seed-Coder-8B-Instruct-GGUF` (`Seed-Coder-8B-Instruct-Q4_K_S.gguf`) | GGUF | Q4_K_S | 4.84 GB |
| `reasoning-gguf-llama-3.2-3b` | `bartowski/llama-3.2-3b-instruct` | `bartowski/Llama-3.2-3B-Instruct-GGUF` (`Llama-3.2-3B-Instruct-Q4_K_S.gguf`) | GGUF | Q4_K_S | 1.93 GB |

**Verifier builds.** The two DeepSeek-Qwen verifier configs were measured on MLX builds (Apple Silicon only) and the DeepSeek-Llama one on a GGUF build (any platform), as confirmed on 2026-11-22. `lms ls --json` paths and the GGUF quant were not recorded for them, so the table gives only the id and the format.

| Config | LM Studio id | Format |
|---|---|---|
| `verifier-mlx-deepseek-7b` | `deepseek-r1-distill-qwen-7b@4bit` | MLX |
| `verifier-mlx-deepseek-7b-8bit` | `deepseek-r1-distill-qwen-7b@8bit` | MLX |
| `verifier-gguf-deepseek-llama-8b` | `deepseek-r1-distill-llama-8b` | GGUF |

**Llama-3.1-8B builds measured and not kept.** Three builds of Llama-3.1-8B-Instruct were measured on the same cases, and only the one in the table above is kept, because the other two did not serve the reasoning stage: `mlx-community/Llama-3.1-8B-Instruct-4bit` (id `llama-3.1-8b-instruct`, 4-bit MLX, a different conversion from the kept one) answered 42 of 44 `lines` requests but skipped 4 findings for having no `claim`, found 17 of 38 bugs, and its live test failed the mechanics check (0 of 3 seeded bugs on `seeded-three`); `json` was 90.9% readable, under the 95% gate. `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` (id `meta-llama-3.1-8b-instruct@8bit`, 8-bit MLX, 8.54 GB) did not answer 10 of 44 `lines` requests (the model did not end its turn), took 1775 seconds against 1119, and failed the mechanics check on the timeouts. Their configs were removed; the numbers stay here so nobody re-measures them to find out. The first experiment's rows (2026-10-29) are labeled only as far as the row says: the Llama-3.1-8B row names no quant, and the Llama-3.2-3B GGUF row names no GGUF quant, so those two were not recorded and should be read as unlabeled.

**Apple Silicon only (MLX)**

| Model and build | lines: readable | json: readable | Bugs found of 38, lines / json | False alarms per run, lines / json | Route set |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B 4-bit MLX | 44 of 44 | 40 of 42 (2 unanswered) | 13 / 21 | 5 / 11 | `lines` |
| Qwen2.5-Coder-7B 8-bit MLX | 44 of 44 | 42 of 44 | 13 / 23 | 5 / 8 | `lines` |
| Llama-3.1-8B 4-bit MLX (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`) | 42 of 42 (2 unanswered) | 40 of 42 (2 unanswered) | 21 / 18 | 26 / 37 | `lines` |
| Llama-3.2-3B 4-bit MLX | 44 of 44 | 42 of 44 | 19 / 8 | 21 / 19 | `lines` |
| Llama-3.2-3B 8-bit MLX | 44 of 44 | 26 of 40 (4 unanswered) | 19 / 12 | 15 / 11 | `lines` |

**Any platform (GGUF)**

| Model and build | lines: readable | json: readable | Bugs found of 38, lines / json | False alarms per run, lines / json | Route set |
|---|---|---|---|---|---|
| Seed-Coder-8B Q4_K_S GGUF | 44 of 44 | 32 of 44 | 24 / 16 | 32 / 13 | `lines` |
| Llama-3.2-3B Q4_K_S GGUF | 44 of 44 | 26 of 42 (2 unanswered) | 15 / 7 | 11 / 9 | `lines` |

What this does and does not show:

- Every model has a reliable `lines` route, so none needs `json` to be readable. `json` was not reliable for Seed-Coder-8B (73% readable), either of the 8-bit and GGUF Llama-3.2-3B builds (65% and 62%) and only just so for the Qwen builds, the kept Llama-3.1-8B build and Llama-3.2-3B 4-bit (95.2% to 95.5%, two unreadable replies each).
- **The Llama-3.1-8B result belongs to the build.** Of the three builds measured, only the Meta 4-bit conversion passes its live test: 42 of 44 `lines` requests answered (2 got no reply) and every reply that came back readable. The 100% is over the replies that came back, so by requests sent `lines` succeeded 95% of the time. On `lines` it raised 26 false alarms per run: fewer than Seed-Coder-8B (32) and more than the Llama-3.2-3B builds (11 to 21) and both Qwen builds (5, the fewest); on `json` it raised 37, the most of any model on that route. It is kept for the readable route it has, not as a recommendation over the Qwen builds.
- Under `json` the Qwen builds found more of the seeded bugs than under `lines` (21 and 23 against 13) with more false alarms. Readability decides the route here, not recall, so `json` stays an option for a reviewer whose model is known to read it.
- A larger set would be needed to say more: at `temperature: 0` a repeat of a case returns almost the same reply, so the 2 runs add little, and one bug is about 3 points of recall.

To measure a model's route, run `harness/live_routes.py` (see the harness README): a route counts as reliable when 95% or more of its replies are readable, over enough runs to mean something, and a row here should say how many.

In the first experiment (one seeded case, 3 bugs) recall did not change with the format: every readable model found the shell injection and almost
nothing else. On the larger case set (table above) it does change for some models, so read the first rows as a smoke test and the 2026-10-31 table as the measurement.
