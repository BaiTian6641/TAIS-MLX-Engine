# Changelog

Versions are the engine's own; the pinned runtime versions it is built against are
in `requirements.lock`. Every number quoted here was measured on an M2 Ultra with
64 GiB and is reproducible with the `check_*.py` script named beside it.

## 1.3.0

An interactive selector, and the fixes a final review turned up.

### Choosing a model

- `k2mlx` with no arguments, or `k2mlx pick`, shows every profile with the context
  it can hold in this machine's free memory once its weights are resident, and its
  measured decode and prefill rates. Arrow keys or j/k move, enter serves, r
  re-reads memory, q quits; without a terminal it prints the table and reads a
  number, so it works in a script too.
- The context column uses the same `ContextPolicy` the server uses at request
  time, with the weights discounted from available memory - which is why Gemma 4
  31B reads 110k of its 262k window and K2-Horizon 298k of 524k.
- Speeds are measurements from this machine, marked as such; profiles not measured
  here fall back to a bandwidth model calibrated on the ones that were.

### Fixed by a final end-to-end review

- MTP replies corrupted any multi-byte character (CJK, emoji), emitted the
  end-of-turn marker into content, filed the prompt cache under tokens it did not
  cover, and lost the last characters of normalised replies. All four are fixed
  and verified against the same prompts that exposed them.
- Gemma 4 26B-A4B stopped loading: the text-tower adapter stripped
  `language_model.` from the weights but not from the per-module quantization map,
  so its 8-bit router was quantized at 4 bits.
- GPT-OSS, Muse Glimmer, Spark2.5 and GLM-4.7-Flash crashed with the default
  command, because sliding-window and MLA caches cannot take quantized KV. They
  are declared unquantized; every earlier check had passed `--kv-bits 0`, which is
  how a default-path bug hides.
- `k2mlx context --restore` crashed on every run, the first download through a
  mirror crashed, and `--dry-run` raised instead of reporting a blocked extension.

## 1.2.0

Spark2.5 runs, and context extension has a verified map.

### Spark2.5

- `spark2_5` is ported from the checkpoint's own modelling code into
  `vendor/spark2_5/`, because the architecture is in neither the pinned runtime
  nor any of its relatives: it is **dense** where the nearest candidate is
  mixture-of-experts, its attention carries a **per-head output gate**, its QKV is
  **fused into one projection**, and it uses **two rotary embeddings** - a quarter
  of the head dimensions at theta 5M for full layers, all of them at theta 10k for
  sliding ones. 55.3 tok/s at bf16, 1,048,576 native context, verified serving.
- One integration trap, found by serving it: transformers' generic RoPE validator
  rejects the checkpoint's nested per-layer-type `rope_parameters`, so loading the
  tokenizer raised before the model was ever built. The key is namespaced in the
  local copy and the port reads either name.
- The port is pinned by tests that do not need the checkpoint: incremental decoding
  through the caches must equal one forward over the whole sequence, which covers
  rotary offsets, the sliding/full mask split and cache handling at once.

### Context extension, verified per profile

Every profile was extended and then **served**, because a config change that
returns the right number can still hang the model:

| works | from → to |
|---|---|
| `minicpm5-2b` | 131,072 → 262,144 |
| `smollm3-3b` | 65,536 → 262,144 |
| `llama-3.2-3b` | 131,072 → 262,144 (replacing its llama3 scaling) |
| `gpt-oss-20b` | 4,096 → 262,144 |
| `glm-4.7-flash` | 202,752 → 405,504 |
| `nemotron-3.5-30b-a3b` | 262,144 → **524,288** |

| hangs, and was restored | why |
|---|---|
| `qwen3.5-9b`, `qwen3.6-35b-a3b`, `ornith-1.5-35b-a3b`, `qwen3.8-27b` | hybrid linear attention |
| `muse-glimmer-30b` | full layers carry no positional embedding |
| `gemma4-26b-a4b`, `gemma4-31b` | sliding-window layers with a rotating cache |

The Qwen3.5 series, Ornith and Gemma 4 are already at 262,144, and Spark at
1,048,576, so the refusals cost nothing that was not already there.

### Architecture audit

The four previously added models were audited against their reference
implementations and their checkpoints' tensor headers: Muse Glimmer and MiniCPM5
match, GPT-OSS matches apart from YaRN's `truncate` flag being ignored by the
pinned runtime, and GLM-4.7-Flash carries three small differences (MLA norm
epsilon 1e-5 against the reference's 1e-6, a bf16 rather than fp32 router matmul,
and the same `truncate` divergence).

## 1.1.0

Four more models, native context extension, and output channels.

### Models

- **MiniCPM5-2B** (dense, 1.4 GiB, 133.0 tok/s - the fastest profile in the table),
  **GPT-OSS-20B** (MXFP4 MoE, 11.3 GiB, 82.5 tok/s), **Muse-Glimmer-30B** (dense
  52-layer sliding-window/NoPE hybrid, 18.1 GiB, 26.7 tok/s) and
  **GLM-4.7-Flash** (64-expert MoE, 15.7 GiB, 48.2 tok/s), each verified through
  the API harness for chat, streaming and prefix reuse.
- Not added: **Spark-X2.5-4B** declares a `spark2_5` architecture that is neither
  in the pinned runtime nor shape-compatible with one that is - it is dense with a
  headwise attention output gate and mixed sliding-window layers, so it needs a
  vendored implementation rather than a profile entry. **G9v3-39A5B** was
  deprioritised at the requester's direction.

### Context extension

- `k2mlx context` writes YaRN rope scaling into a checkpoint and raises its
  declared window, keeping the original config for `--restore`. Applied:
  MiniCPM5-2B 131,072 -> 262,144, Muse-Glimmer's sibling profiles likewise,
  GLM-4.7-Flash 202,752 -> **405,504**, GPT-OSS 4,096 -> 262,144 on top of the
  vendor's own factor-32 scaling.
- Muse-Glimmer refuses the extension: its full-attention layers carry no positional
  embedding, and the extended configuration hangs the runtime. It stays at its
  native 131,072, which is the 128K it was trained for. `--restore` is the way back
  and was exercised on real checkpoints.

### Output channels

- GPT-OSS and Muse Glimmer answer inside channel envelopes. `output_channels.py`
  rewrites them into the ` thinking`/`<｜end▁of▁thinking｜>` convention the runtime understands, so
  the reply arrives in `content` and the deliberation in `reasoning`. Both models
  previously returned raw template markers, and Muse Glimmer's answer was buried in
  its private channel. Streaming output is tested to equal one-shot output at every
  chunk size.

### Fixed

- `check_profile_api.py` asserted arithmetic ("What is 2 + 2?" -> "4"), which
  GLM-4.7-Flash answers "5" to while answering "4" to "2+2" - a model behaviour
  that made a server smoke test fail. It now asks for a capital city, and reports
  prefix reuse instead of asserting on cache internals.

## 1.0.0

First release: a single-server MLX engine that serves dense, MoE,
block-diffusion and disk-streamed models, with concurrency, prompt reuse and
speculative decoding where each of them actually pays.

### Serving

- HTTP server on the pinned `mlx-lm` runtime, with per-profile context admission,
  telemetry (`metrics.json`) and a console monitor.
- Twelve profiles: K2-Horizon, Gemma 4 26B-A4B and 31B, DiffusionGemma,
  Qwen3.8-27B, Qwen3.6-35B-A3B, Ornith-1.5-35B-A3B, Nemotron-3.5-Lightning-30B-A3B,
  Qwen3.5-9B and 4B, Llama-3.2-3B, SmolLM3-3B, plus two streamed GGUF profiles.
- Continuous batching via `--decode-concurrency`, measured at 1.42x to 1.88x
  aggregate throughput depending on the model (`check_optimizations.py`).
- SSD-backed prompt cache, and a prefill snapshot per prompt so later requests
  that extend it reuse the work (`disk_cache.py`, `check_optimizations.py`).

### Speculative decoding

- MTP: the Qwen3.5-family head, extracted from the upstream checkpoint by HTTP
  range request, served through a vendored round loop. **1.59x** over HTTP
  (94.4 against 59.2 tok/s) with byte-identical output (`check_speculative.py`).
- DFlash and DSpark block drafters also run, and are **lossless but slower** on
  these targets (0.44x to 0.97x). The measurements and the reason - round cost
  scales with drafting layers times verify width, and verifying k tokens activates
  about 8(k+1) experts per layer on a 256-expert MoE - are in
  `docs/decoding-and-memory.md`.

### Quantized and streamed models

- GGUF-resident models with disk-streamed routed experts (`gguf_model.py`),
  IQ block decoders in MLX ops and Metal kernels (`iq_quants.py`, `iq_metal.py`),
  and bit-exact parity checks against the reference dequantizers.
- Expert residency telemetry, and an SSD hot-set cache with the saliency-based
  selection that made it pay.

### Operations

- `k2mlx` command line: `serve`, `stop`, `models`, `download`, `doctor`, `bench`,
  `version`, installable as a console script.
- Hugging Face mirror support: `--hf-endpoint` / `K2MLX_HF_ENDPOINT` and the
  conventional variables, for tokens, cache location and offline mode
  (`hf_env.py`).
- Startup verification of the serving flags against the loaded model's own caches:
  a cache that cannot be quantized, or cannot merge, is reported instead of
  failing mid-request (`runtime_support.cache_capabilities`).
- 49 regression tests, all skipping cleanly when the weights they need are absent.

### Documented failure modes

- macOS kills long GPU command buffers (`Impacting Interactivity`); a 198k-token
  prefill needs `--prefill-step-size 64` and then completes in 1,004.6 s.
- Prompt-prefix reuse needs unquantized KV, and a chat turn is not a token prefix
  of its own next turn: measured at 1215 of 1219 tokens shared, matching vLLM's
  production figure of a 1.7% hit rate across 610 agentic traces.
- Prefill costs about 14x more per token above roughly 2k prompt tokens. Measured
  identically on the stock upstream server, so it is the runtime's behaviour and
  not this engine's.
