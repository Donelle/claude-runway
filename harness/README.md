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
| `live_classifier.py`    | classifier matches a verified finding to a rubric entry            | fixtures only; waits for the classifier stage                    |
| `live_verifier.py`      | verifier confirms or rejects a finding                             | fixtures only; waits for the verifier stage                      |
| `live_all_reviewers.py` | every configured reviewer over the four-file `orders` case, scored | runs the reasoning stage today; picks up the others as they land |

`live_classifier.py` and `live_verifier.py` take `--check` to validate and print
their fixtures without LM Studio. Without it they exit with status 3 and say the
stage is not built (or that the harness needs a runner for it).

## Answers live in `cases.py`

Each case is a base tree and a head tree. What is wrong in head is written down:

- a **bug** is a real defect; reporting it counts as finding it;
- a **decoy** is code that looks suspicious and is correct (a shell call with a
  constant command, a locked counter); a finding on one is a false alarm the
  verifier stage should reject;
- the `clean` case has neither, so anything reported on it is a false positive.

A finding is matched to a bug by file and line (within three lines, because the
model reads line numbers off a diff), not by the class it chose. To add a case,
add a `Case` and a `marker` per bug: the marker is a string that occurs on
exactly one line, so the answer follows the code when it is edited.
Keep each marker unique: a marker that matches no line, or more than one, makes `cases.py` raise a `ValueError` when the case is scored.

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

Snapshots only hold what the inline comments say. Findings that appear only in a review's body
(the "previously missed" items) are kept verbatim but not scored.

## Choosing reviewers and stages: `--config`

`live_reasoning.py` and `live_all_reviewers.py` take `--config FILE`, a JSON file that says which models and
prompts to test and which stages to test them in. A comparison becomes a file you can keep, diff and re-run
instead of a long command line. (`live_verifier.py` and `live_classifier.py` do not take it yet: they have
nothing to run until their stages exist, and will get it with them.)

```
.venv/bin/python harness/live_all_reviewers.py --config harness/config.example.json --runs 3
.venv/bin/python harness/live_reasoning.py --config my-config.json --show-config   # what it resolves to; needs no LM Studio
```

The file has three optional keys:

| Key         | Meaning                                                                                                                                                                                                                          |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reviewers` | The reviewers to test, exactly as in a reviewers file: `name`, `model`, `prompt` and/or `promptFile`, `active`, `type`, `loadOnDemand`, `contextLength`. It is checked by the server's own validator, so a file that passes here is one `REVIEWERS_FILE` accepts. |
| `rubric`    | The rubric the classifier matches against, as in a reviewers file. Needs `reviewers`.                                                                                                                                              |
| `stages`    | Which stages to test: a non-empty combination of `reasoning`, `verifier` and `classifier`. Left out, it is every stage an active reviewer has a `type` for (all three when the file has no reviewers).                          |

- **A reviewers file is a valid config as it stands**: it just has no `stages`, so every stage it has reviewers for is tested.
- **Stages.** One left out of `stages` is not run, even when reviewers are configured for it. Selecting one that no active reviewer has the `type` for is an error. Selecting one this checkout cannot run yet (today, the verifier and the classifier) is reported as not run, so a config written for the whole pipeline keeps working as the stages arrive.
- **Which reviewers run.** `live_reasoning.py` runs every active reviewer with the `reasoning` type over each case, instead of the single reviewer `--model` and `--prompt` build. `live_all_reviewers.py` takes its reviewers, rubric and stages from the file instead of its three default lenses.
- **Flags.** The file names each reviewer's model, prompt and loading, so `--model`, `--prompt`, `--prompt-file`, `--context` and `--resident` are refused alongside it. Everything else (`--runs`, `--case`, `--history`, `--chunk-chars`, `--timeline`, ...) still applies. Without `--config` nothing changes.
- **Errors.** Every problem in a bad file is reported at once, naming the file: unreadable, not JSON, an unknown key, an unknown or repeated stage, an empty `reviewers` list (name a reviewer, or leave the key out to use the flags), plus whatever the reviewers-file validator finds.
- **`--show-config`** prints the stages and reviewers a run resolves to (model, roles, how the model is loaded, where the prompt comes from) and exits without touching LM Studio.

`config.example.json` shows the shape: two narrow reasoning lenses on one model, plus a verifier and a classifier with a small rubric. Change the model ids to ones `lms ls` shows. The verifier and classifier entries do nothing until those stages exist, because `stages` selects only `reasoning`.

## Before it spends minutes

Each script asks the model one question first: reply with one word, capped at 40
tokens. A model that does not end its turn runs to the context limit on every
request and times out, which on a real run looks like nothing but 60 second
timeouts. The probe catches it in seconds and says which model. Typical cause: a
base model downloaded instead of its `-Instruct` build. `--no-probe` skips it.

## Useful options

| Option                                 | Meaning                                                                        |
| :------------------------------------- | :----------------------------------------------------------------------------- |
| `--config FILE`, `--show-config`       | reviewers, rubric and stages from a JSON file, and what it resolves to (see above) |
| `--model`, `--context`                 | model id as `lms ls` shows it, and the context to load it with                 |
| `--prompt`, `--prompt-file`            | the reviewer's lens; compare prompts on identical input                        |
| `--runs N`                             | repeat; one run proves little                                                  |
| `--chunk-chars N`, `--context-lines N` | chunk size and lines of code around each change                                |
| `--resident`                           | the model is already loaded; do not load or unload it                          |
| `--timeline`                           | every REST call and completion request, in order                               |
| `--case own-diff`                      | this checkout against `origin/main`, unscored                                  |
| `--case own-planted`                   | the same, in a throwaway clone with one shell injection planted in a real file |

Models the harness loads are unloaded again; models you had loaded are left alone.

For local deployment within a **3 GB to 9 GB** memory or VRAM footprint, here are the recommended **Instruct-tuned** models. These models include built-in chat templates (like ChatML or Llama-3) and pre-configured stop tokens, ensuring they answer your prompts directly without endlessly hallucinating text.

### 1. Code Reviewer Models

Code reviewers need to hold large contexts (e.g., git diffs, pull request comments, project style guides) and communicate structured, actionable suggestions in idiomatic English.

- **Qwen2.5-Coder-7B-Instruct**
  - **MLX Repo:** `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (\~4.3 GB)
  - **Higher-Precision MLX:** `mlx-community/Qwen2.5-Coder-7B-Instruct-8bit` (\~7.9 GB)
  - **GGUF Alternative:** `bartowski/Qwen2.5-Coder-7B-Instruct-GGUF` (`Q4_K_M` \~4.7 GB / `Q5_K_M` \~5.3 GB)
  - **Embedded Template:** Native ChatML (`<|im_start|>`, `<|im_end|>`)
  - **Why it works out of the box:** Unlike the Base variant, the `-Instruct` checkpoint contains pre-configured stop criteria in its tokenizer config. It handles multi-file diffs and refactoring without hallucinating follow-up text.

- **Llama-3.1-8B-Instruct**
  - **MLX Repo:** `mlx-community/Llama-3.1-8B-Instruct-4bit` (\~4.9 GB)
  - **GGUF Alternative:** `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (`Q4_K_M` \~4.9 GB)
  - **Embedded Template:** Llama-3 (`<|start_header_id|>`, `<|eot_id|>`)
  - **Why it works out of the box:** Highly aligned for stylistic critique and conversational feedback. It adheres strictly to custom review rules (e.g., "only highlight severity P0/P1 bugs and do not touch formatting").

### 2. Code Classifier Models

Classifiers require deterministic outputs, low token generation latency, and strict adherence to structured schemas (JSON, labels, CWE vulnerability tags).

- **Qwen2.5-Coder-3B-Instruct**
  - **MLX Repo:** `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` (\~1.9 GB)
  - **Higher-Precision MLX:** `mlx-community/Qwen2.5-Coder-3B-Instruct-8bit` (\~3.5 GB)
  - **GGUF Alternative:** `bartowski/Qwen2.5-Coder-3B-Instruct-GGUF` (`Q8_0` \~3.4 GB)
  - **Embedded Template:** Native ChatML
  - **Why it works out of the box:** The 8-bit version consumes only \~3.5 GB of memory while delivering zero-shot JSON tag extractions (e.g., classifying snippets into `VULNERABILITY`, `DEAD_CODE`, or `PERFORMANCE_ISSUE`) without conversational filler.

- **Phi-3.5-mini-instruct (3.8B)**
  - **MLX Repo:** `mlx-community/Phi-3.5-mini-instruct-4bit` (\~2.2 GB)
  - **GGUF Alternative:** `bartowski/Phi-3.5-mini-instruct-GGUF` (`Q8_0` \~4.1 GB)
  - **Embedded Template:** Phi-3 format (`<|user|>...<|end|><|assistant|>`)
  - **Why it works out of the box:** Heavy synthetic-data pretraining makes it sharp at mapping snippets to technical taxonomy (such as CWE defect classifications and language detection).

- **Llama-3.2-3B-Instruct**
  - **MLX Repo:** `mlx-community/Llama-3.2-3B-Instruct-4bit` (\~1.8 GB)
  - **Higher-Precision MLX:** `mlx-community/Llama-3.2-3B-Instruct-8bit` (\~3.4 GB)
  - **GGUF Alternative:** `bartowski/Llama-3.2-3B-Instruct-GGUF` (`Q8_0` \~3.4 GB)
  - **Embedded Template:** Llama-3 format
  - **Why it works out of the box:** Lightweight classifier with minimal memory usage, leaving headroom for continuous batching or running alongside another process.

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

| Role           | Hugging Face Repository (MLX)                     | Parameters | Precision / Quant | Memory Footprint | Embedded Template |
| -------------- | ------------------------------------------------- | ---------- | ----------------- | ---------------- | ----------------- |
| **Reviewer**   | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`    | 7B         | 4-bit             | \~4.3 GB         | ChatML            |
| **Reviewer**   | `mlx-community/Qwen2.5-Coder-7B-Instruct-8bit`    | 7B         | 8-bit             | \~7.9 GB         | ChatML            |
| **Reviewer**   | `mlx-community/Llama-3.1-8B-Instruct-4bit`        | 8B         | 4-bit             | \~4.9 GB         | Llama-3           |
| **Classifier** | `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`    | 3B         | 4-bit             | \~1.9 GB         | ChatML            |
| **Classifier** | `mlx-community/Qwen2.5-Coder-3B-Instruct-8bit`    | 3B         | 8-bit             | \~3.5 GB         | ChatML            |
| **Classifier** | `mlx-community/Phi-3.5-mini-instruct-4bit`        | 3.8B       | 4-bit             | \~2.2 GB         | Phi-3             |
| **Classifier** | `mlx-community/Llama-3.2-3B-Instruct-8bit`        | 3B         | 8-bit             | \~3.4 GB         | Llama-3           |
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
