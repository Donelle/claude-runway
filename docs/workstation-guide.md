# Local workstation and model guide

Choose a workstation for the work ClaudeRunway sends to local compute: embeddings, retrieval, and text compression. Claude Code still uses its configured Claude service; installing ClaudeRunway does not make Claude itself run offline.

**Recommendation:** for compression alongside normal development, start with 32 GB system/unified memory, an SSD, and a small instruction-tuned model. For a workstation intended to host a roughly 30B development model as well, target 64 GB system RAM plus 24–32 GB discrete VRAM, or 64 GB Apple unified memory. These are planning recommendations, not measured minimums or a guarantee of performance.

Model/provider details were reviewed on **2026-10-04**. The configurations below are candidates to validate with your own workload; ClaudeRunway has not benchmarked this matrix.

## What runs locally today

| Component | Current behavior | Hardware implication |
|---|---|---|
| Codebase and durable memory | Qdrant stores vectors; FastEmbed produces embeddings. The shipped template uses `sentence-transformers/all-MiniLM-L6-v2`. | Budget CPU, RAM, and SSD space for indexing and collections. A large generative model is not required. |
| Compression | LM Studio serves a loaded model through its chat-completions API. ClaudeRunway summarizes logs, command output, documents, and other large text. | Favor fast, faithful summaries and low queue latency over model size. |
| Development model routing | Future use case; no development routing configuration is supplied by this guide. | Reserve capacity for longer prompts, generation, tool use, and possibly simultaneous compression. |

Qdrant memory and compression work independently. See [Prerequisites](prerequisites.md) and [Installation](installation.md) for the actual software requirements. CPU-only compression is an option where the runtime supports it, but benchmark hook latency before making it part of an interactive workflow.

## Workstation tiers

Assumptions: one developer, IDE/browser/build tools running, one development-model request at a time, and quantized weights. Discrete VRAM and system RAM are separate pools; Apple unified memory is shared by the OS, applications, and GPU. Installed memory is not all available for models.

The current-tool recommendations below use a deliberately small shortlist. **MiniLM is the shipped embedding default on every tier**; BGE and Jina are alternatives to evaluate for retrieval, not automatic upgrades when you buy a larger GPU. Embeddings use FastEmbed, independently of LM Studio. The table uses short names mapped to exact embedding IDs below; compression names link to provider cards in the model shortlist.

| Configuration | FastEmbed embedding shortlist | LM Studio compression shortlist | Selection guidance |
|---|---|---|---|
| Existing CPU-only machine, 16–32 GB RAM, SSD | MiniLM; BGE Small | Gemma 3 4B IT or Qwen3.5-4B, `Q4_K_M` GGUF | Prefer MiniLM for the existing setup; benchmark compression latency before enabling interactive hooks. |
| NVIDIA 8 GB VRAM, 32 GB RAM | MiniLM; BGE Small | Gemma 3 4B IT or Qwen3.5-4B, `Q4_K_M` GGUF | Keep embeddings on the standard CPU path and reserve VRAM for compression. |
| NVIDIA 12–16 GB VRAM, 32–64 GB RAM | MiniLM; BGE Small; BGE Base | Gemma 3 4B IT or Qwen3.5-4B, `Q4_K_M` / `Q8_0` GGUF; Qwen3.5-9B, `Q4_K_M` GGUF if needed | Start with 4B; evaluate 9B only if it preserves details better within the request budget. |
| NVIDIA 24 GB VRAM, 64 GB RAM; e.g. RTX 3090/4090 | MiniLM; BGE Base; Jina Code | Gemma 3 4B IT or Qwen3.5-4B, `Q4_K_M` / `Q8_0` GGUF; Qwen3.5-9B, `Q4_K_M` / `Q8_0` GGUF | Evaluate Jina for code retrieval. Keep the small compressor if it already meets fidelity requirements. |
| NVIDIA Blackwell 32 GB VRAM, 64 GB RAM; e.g. RTX 5090 | MiniLM; BGE Base; Jina Code | Gemma 3 4B IT or Qwen3.5-4B, `Q4_K_M` / `Q8_0` GGUF; Qwen3.5-9B, `Q4_K_M` / `Q8_0` GGUF | Extra VRAM provides concurrency/future routing headroom; it does not require a larger compressor. |
| Apple Silicon, 16–24 GB unified memory | MiniLM; BGE Small | Gemma 3 4B IT or Qwen3.5-4B, `Q4_K_M` GGUF | Small defaults leave room for macOS, builds, and multiple embedding processes. |
| Apple Silicon, 32–48 GB unified memory | MiniLM; BGE Small; BGE Base | Gemma 3 4B IT or Qwen3.5-4B, `Q4_K_M` / `Q8_0` GGUF; Qwen3.5-9B, `Q4_K_M` GGUF | Compare 9B only when needed; watch shared-memory pressure during indexing and builds. |
| Apple Silicon, 64–128 GB unified memory | MiniLM; BGE Base; Jina Code | Gemma 3 4B IT or Qwen3.5-4B, `Q4_K_M` / `Q8_0` GGUF; Qwen3.5-9B, `Q4_K_M` / `Q8_0` GGUF | Evaluate code-focused retrieval and keep a small dedicated compressor alongside future development models. |
| AMD discrete GPU or large-memory integrated workstation | MiniLM; BGE Small; BGE Base/Jina Code with sufficient host RAM | Gemma 3 4B IT or Qwen3.5-4B, `Q4_K_M` GGUF; Qwen3.5-9B, `Q4_K_M` GGUF after memory/latency validation | Validate the exact LM Studio backend and GPU. Host RAM and indexing latency govern embedding choice. |

### Quantization formats for LM Studio

The compression table names **GGUF `Q4_K_M`** as the starting build and **GGUF `Q8_0`** as a higher-precision comparison, where those builds are available and supported by the installed runtime. These are evaluation choices, not validated winners. On Apple Silicon, an LM Studio-supported MLX 4-bit build is another option; record its exact quantization configuration rather than assuming it is equivalent to `Q4_K_M`.

| Format | What the label means | Workstation selection |
|---|---|---|
| GGUF `Q4_K_M` | A roughly 4-bit K-quant recipe with some tensors at higher precision; GGUF is the container. | Starting compression candidate across supported CPU, NVIDIA, Apple Metal, and AMD runtime paths. |
| GGUF `Q8_0` | An 8-bit block quantization with scale metadata. | Compare when memory permits and it improves summary fidelity; benchmark latency too. |
| MXFP4 | 4-bit floating-point values with an 8-bit shared scale per block of 32 values. | Use only with an explicitly supported model build/runtime; hardware acceleration depends on the compute path. |
| NVFP4 | 4-bit floating-point values with an FP8 scale per block of 16 values and an additional tensor-wide scale. | Relevant to the future Nemotron Blackwell candidate; verify runtime support for native FP4 acceleration. |

MXFP4 and NVFP4 are both 4-bit formats, but their scaling differs from each other and from GGUF K-quants. Scale metadata, higher-precision tensors, caches, and runtime buffers mean model memory exceeds the raw four-bits-per-parameter estimate. A 4-bit model file also does not imply that all computation happens at 4-bit precision. See [llama.cpp quantization options](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) and [NVIDIA's MXFP4/NVFP4 explanation](https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/).

### Scoped embedding candidates

| Short name | Exact `EMBEDDING_MODEL` ID | Vector dimensions | Reason to evaluate |
|---|---|---|---|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Shipped baseline; retain it for compatibility with existing collections and low resource usage. |
| BGE Small | `BAAI/bge-small-en-v1.5` | 384 | Small English retrieval alternative for constrained workstations. Compare retrieval quality against MiniLM on representative queries. |
| BGE Base | `BAAI/bge-base-en-v1.5` | 768 | Broader English retrieval candidate when indexing throughput and collection size permit. Validate relevance on repo queries. |
| Jina Code | `jinaai/jina-embeddings-v2-base-code` | 768 | Code-focused candidate trained on code/question and code/docstring pairs; evaluate on code-heavy repositories. |

These four are listed in [FastEmbed's dense text model registry](https://qdrant.github.io/fastembed/examples/Supported_Models/). Confirm support with `TextEmbedding.list_supported_models()` in the installed environment before configuring an alternative.

Changing `EMBEDDING_MODEL` is a migration, not a config tweak. Each collection is bound to the embedding model and vector dimensions that created it, so an existing collection will reject or mismatch vectors from a different model. Plan a collection drop and full re-index for each affected project. The shared `memory-bank` collection must use one model across all projects, so do not rebuild it casually; change its model only as a deliberate, coordinated step. See the [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5) and [Jina Code model card](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) for intended usage. Model-supported sequence length does not establish the tokenizer limit in your installed FastEmbed path; inspect truncation against actual indexed chunks.

The standard installation uses FastEmbed's CPU path; this guide does not enable CUDA/Metal embedding acceleration or serve embeddings through LM Studio. Each MCP process may load its own embedding instance, so budget for process count as well as model size. At the same point count, 768-dimensional float32 vectors have twice the raw vector payload of 384-dimensional vectors; Qdrant's total footprint also includes indexes and metadata.

### Future development-model capacity

For future routing, start with Qwen3.5-9B in GGUF `Q4_K_M` on 8–16 GB NVIDIA cards or 16–24 GB Apple unified memory, initially loaded separately from compression. Evaluate supported GGUF `Q4_K_M` builds of Qwen3.8-27B or Qwen3-Coder-30B-A3B, or Meta's hardware-targeted Muse Glimmer K-quant builds on 24–32 GB NVIDIA cards or 32–48 GB Apple systems; 64 GB Apple unified memory gives more application headroom. Nemotron 3.5 Lightning NVFP4 is an additional Blackwell candidate. These remain planning estimates, with model details below; they are outside the current compression shortlist.

Use an SSD for model files, embedding caches, and Qdrant storage. Size free space from the actual download inventory, including multiple quantizations, optional vision/draft files, collections, and container images. More CPU cores help builds and ingestion but do not replace GPU/unified-memory bandwidth for inference. For sustained background operation, cooling, power consumption, and fan noise also matter.

## Model shortlist by role

These are evaluation candidates, not a universal quality ranking. Provider benchmark scores do not establish compression fidelity or throughput inside ClaudeRunway.

| Model | Role to evaluate | Selection notes |
|---|---|---|
| [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it) | Baseline compression | Small instruction-tuned model suited to summarization; existing environment-variable examples use a Gemma 3 4B identifier. Use an appropriate quantized build and review the Gemma license. |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Alternative small compressor | Compare fidelity and latency against Gemma on the same inputs. Configure non-thinking behavior where supported and verify the returned content. |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Higher-capacity compression; small development model | Consider when the 4B candidates omit important details. Keep it only if the quality improvement justifies latency and memory. |
| [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | Future coding route | Coding-focused MoE candidate. Few active parameters reduce computation; the full weight set still needs storage/residency planning. |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Future coding, reasoning, and multimodal route | Hybrid attention architecture; thinking is enabled by default in the provider configuration. Evaluate reasoning controls separately from compression. |
| [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | Future multimodal/tool-using route | Meta supplies quantizations targeting 24/32 GB envelopes. Vision and speculative draft components must be included in the memory budget. |
| [Nemotron 3.5 Lightning 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | Future NVIDIA agent route | NVIDIA documents native FP4 on Blackwell and other compute paths on older hardware. NVFP4 support is not interchangeable with a GGUF or Apple runtime path. |

ClaudeRunway's compressor sends text messages with `temperature=0`; it does not currently send images or request model-specific reasoning controls. A multimodal model's vision capability therefore adds no functionality to this compression path. Confirm any runtime-level reasoning settings with an actual compression call.

## Memory, context, and concurrency

Four-bit weights have a theoretical payload of roughly **0.5 GB per billion parameters**: 4B ≈ 2 GB, 9B ≈ 4.5 GB, and 30B ≈ 15 GB (decimal). Actual quantized files differ because of scales, mixed precision, and additional components. These figures exclude caches, runtime buffers, and application memory; use the runtime's loaded-memory estimate and monitor real usage.

Standard transformer KV-cache storage grows linearly with retained token count, layers, and concurrent sequences. Hybrid/recurrent or sliding-window architectures change that budget, depending on implementation. Advertised maximum context is not a recommendation to allocate it on a consumer workstation.

The current compressor defaults to **12,000 characters per chunk**, not 12,000 tokens. It may classify multiple chunks concurrently, compress them sequentially, and make a final combine request. Start by testing an 8K-token context for ordinary chunks, then increase it if your tokenizer, prompt, output, or combined summaries require more. There is no single context size that guarantees every compression input fits.

Each completion request defaults to a **60-second timeout**, with SDK retries disabled. Multiple calls can make total compression take longer than 60 seconds. Queued requests consume that budget too; timed-out hooks can pass original output through, losing the expected token savings. See [LM Studio concurrency and the timeout](environment-variables.md#lm-studio-concurrency-and-the-timeout).

For future routing, budget both models' weights plus caches/buffers if they stay loaded together. On limited memory, loading models separately avoids combined residency but introduces switching latency. Longer development requests can also delay compression when they share a server. Evaluate separate serving capacity or scheduling before relying on both concurrently.

## Configure and validate a compressor

1. Load a supported instruction-tuned quantization in LM Studio. Start with GGUF `Q4_K_M` where supported; compare `Q8_0` or another supported higher-precision build only when there is a measured fidelity problem or spare capacity.
2. Set `CLAUDE_RUNWAY_LMSTUDIO_MODEL` to the **actual served identifier**, in both the shell and the project's `local-compress` MCP environment. Provider repository names in this guide are not guaranteed to be LM Studio identifiers. Auto-detection only confirms load state when LM Studio's `/api/v0/models` endpoint is reachable, and it then requires exactly one loaded non-embedding model. If that endpoint is unavailable, the `/v1/models` fallback lists downloaded models whether or not they are loaded, so auto-detection can select an unloaded model. Explicitly pin compression when experimenting with another model, and when several models are installed.
3. Follow [Environment variables](environment-variables.md), run the doctor check, and complete [Verifying it's working](verifying-its-working.md).
4. Compare models on representative passing/failing build logs, long diffs, documents with multiple sections, and selective-focus inputs. Check preserved error messages, paths, identifiers, final status, and unsupported claims against the original text.
5. Measure end-to-end latency, timeout frequency, memory pressure, and savings while the IDE and build tools run. Repeat with realistic request bursts; single-request token speed does not predict queue behavior.

Use [EVALUATION.md](../EVALUATION.md) to measure token savings. Keep byte-exact outputs on the existing [exactness-critical paths](skills-and-hooks.md#exactness-critical-commands-bash-only); model selection cannot make lossy summarization lossless.

For future development routing, add task completion, tool-call validity, tests passing, and recovery from tool errors to the evaluation. Record the model/build, quantization, runtime/version, workstation, context, concurrency, and test inputs. Promote a recommendation to “validated” only when those results are available.

## Sources and maintenance

The [Educative article on local 30B agents](https://www.educative.io/newsletter/artificial-intelligence/local-ai-agents-30b-models) motivated the larger-model shortlist. Hardware assignments in this guide are planning inferences from model size, provider deployment notes, and ClaudeRunway's current implementation, rather than measurements from that article.

Check [LM Studio system requirements](https://lmstudio.ai/docs/app/system-requirements) and [parallel-request documentation](https://lmstudio.ai/docs/app/advanced/parallel-requests) for runtime requirements. [llama.cpp](https://github.com/ggml-org/llama.cpp) and [MLX-LM](https://github.com/ml-explore/mlx-lm) describe backend capabilities, but backend availability alone does not establish support for every listed model in the installed LM Studio version.

Refresh provider links and compatibility when updating this guide. Keep planning estimates separate from workstation results, and update the routing section when that feature ships.
