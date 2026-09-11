# k2mlx

An MLX inference engine for Apple silicon that serves dense, mixture-of-experts,
block-diffusion and disk-streamed models behind one HTTP API, and measures what it
claims. Thirteen profiles run today, from a 3B model to a 35B mixture-of-experts
whose cold experts stream from SSD.

Built on the pinned `mlx-lm` runtime (`requirements.lock`) with vendored
architecture code in `vendor/`. **49 regression tests**, and every performance
number in the documentation comes from a script in this repository.

```sh
k2mlx doctor                                   # environment, dependencies, disk, Hub mirror
k2mlx models                                   # profiles, sizes, what is downloaded
k2mlx download qwen3.6-35b-a3b                 # mirror-aware weight download
k2mlx serve --model qwen3.6-35b-a3b --detach   # HTTP server on :8080
k2mlx bench api --model qwen3.6-35b-a3b        # correctness of chat, streaming, prefix reuse
k2mlx stop
```

Full operator documentation, including every flag and its measured effect, is in
[Operations](docs/operations.md). What follows on this page is the engine's
behaviour and the evidence for it.

## What it does

- **Serving** — OpenAI-compatible `/v1/chat/completions`, streaming and not, with
  memory-aware context admission, SSD prompt caching, and telemetry.
- **Batching** — continuous batching at 1.42x to 1.88x aggregate throughput,
  where the model's caches allow it, verified to return byte-identical answers.
- **Speculative decoding** — MTP, which is the variant that pays here: 1.59x over
  HTTP with byte-identical output. The block drafters (DFlash, DSpark) also run
  losslessly and are slower; both results are measured and explained.
- **Quantized weights** — 4/8-bit MLX checkpoints, and GGUF-resident models with
  IQ block decoders in MLX ops and Metal kernels, bit-exact against the reference.
- **Streamed experts** — cold routed experts read from SSD with a saliency-driven
  hot-set cache, so models larger than memory still serve.
- **Mirrors** — any Hub-compatible endpoint via `--hf-endpoint` or
  `K2MLX_HF_ENDPOINT`, with token, cache-location and offline controls.

## Profiles

| profile | resident | decode | notes |
|---|---:|---:|---|
| `k2-horizon` | 19.6 GiB | 46.6 tok/s | dual attention + routed MoVA experts, 524k context |
| `gemma4-26b-a4b` | 14.3 GiB | 191.7 tok/s | MoE with sliding-window KV, batches at 1.8x |
| `gemma4-31b` | 16.1 GiB | 22.6 tok/s | dense, 60 layers, bandwidth-bound |
| `qwen3.8-27b` | 15.0 GiB | 67.6 tok/s | dense |
| `qwen3.6-35b-a3b` | 19.0 GiB | 77.5 tok/s, 93.2 with MTP | hybrid linear attention, 256 experts |
| `ornith-1.5-35b-a3b` | 18.2 GiB | 78.2 tok/s, 94.4 with MTP | fine-tune of the same base |
| `nemotron-3.5-30b-a3b` | 16.6 GiB | 86.2 tok/s | 23 Mamba / 23 MoE / 6 attention layers |
| `qwen3.5-9b` / `qwen3.5-4b` | 4.8 / 2.3 GiB | 70.9 / 94.8 tok/s | dense |
| `llama-3.2-3b` / `smollm3-3b` | 1.7 / 1.6 GiB | 150.0 / 115.2 tok/s | dense |
| `qwen3.8-flash`, `deepseek-v4-flash` | streamed | 11.9 / 3.3 tok/s | IQ1 GGUF with expert streaming |

GGUF weights can also be **imported** for MLX execution; the importer supports
Llama, Qwen3.8-27B and Qwen3.6-35B-A3B text architectures, preserves compatible
Q4/Q8 grids, and offers explicit K/IQ conversion into 2/3/4/6/8-bit MLX weights.
See [GGUF import and quantization](docs/gguf-mlx.md). IQ block decoding runs as MLX
operations and in custom Metal kernels, both bit-exact against the reference
decoders; the architecture, measured quant inventory, profiling and remaining work
are in [IQ1 GGUF experts with MLX inference](docs/iq-streaming.md), and the sources
behind every borrowed technique are tracked in [References](docs/references.md).

```sh
# Pin and download the IQ1 quants (about 145 GiB, resumable):
.venv/bin/python setup_gguf.py qwen3.8-flash deepseek-v4-flash --quant UD-IQ1_S
```

The Flash text adapters and cache enhancements are also archived in
`checkpoints/flash-integration-before-gguf.tar.gz`. Both Flash adapters passed
small-model and HTTP fixture checks; full Flash checkpoint validation is pending.
See [Flash integration notes](docs/flash-server.md).

Also supports text generation with Qwen3.8-27B, Qwen3.6-35B-A3B, Ornith-1.5
35B-A3B, Gemma 4 31B, Gemma 4 26B-A4B and its block-diffusion sibling
DiffusionGemma (text paths only; see [Gemma 4 support](docs/gemma4.md)), plus
small models: Qwen3.5-9B, Qwen3.5-4B, Llama-3.2-3B and SmolLM3-3B.
Downloaded checkpoint revisions are recorded in `model-configs/*.lock.json`.

```sh
# Download the pinned profile checkpoint if needed:
.venv/bin/python setup_models.py qwen3.6-35b-a3b
# Keep up to 12 GiB of frequently selected expert projections in unified memory:
sh start.sh --model qwen3.6-35b-a3b --expert-cache-gib 12
# Dense Qwen profile:
sh start.sh --model qwen3.8-27b
# Small models resolve through the generic path:
.venv/bin/python setup_models.py smollm3-3b && sh start.sh --model smollm3-3b
```

Choose one server at a time. Use the selected profile name as the API model.

Serving knobs that change the tradeoff rather than the model:

- `--kv-bits 0` keeps the KV cache in bf16. It costs a few percent of decode
  speed and buys two things the pinned runtime cannot do with a quantized cache:
  continuous batching, and prompt-prefix reuse (a repeated 32k prompt returns its
  first token in 0.49 s instead of 48 s). `--kv-bits 4` (default) uses less
  memory; `--kv-bits 8` is a middle ground for long context.
- `--decode-concurrency` / `--prompt-concurrency` batch decoding and prefill.
  Measured 1.4-1.8x aggregate throughput at four concurrent requests, and legal
  only with unquantized KV.
- `--prompt-cache-size` sets how many prompt-prefix KV caches stay resident, so
  multi-turn callers skip re-prefilling the shared prefix.
- `--mtp-draft <dir>` serves a qwen3_5_moe profile with multi-token-prediction
  speculation: `models/qwen3.6-mtp` drafts one token per round and the target
  verifies two, which measured 94.4 tok/s against 59.2 unspeculated over HTTP on
  the same prompt with byte-identical output. It is single-request only, and it
  trades away prompt-prefix reuse for drafting.
- `--prefill-step-size` (default 512) is not a throughput lever once the model is
  warm; see [decoding and memory](docs/decoding-and-memory.md) for the measured
  prefill curve, including the 14x per-token cliff above ~2k tokens that is
  present in upstream too. It is the knob that keeps a very long prefill inside
  the window macOS gives a Metal command buffer: a 198k-token prompt is killed at
  the default step and completes at `--prefill-step-size 64`.

Without `--expert-cache-gib`, weights remain fully resident. With it, dense weights
stay resident and cold routed experts are read from SSD on demand. Every selected
expert still executes. The hot cache fills with use; context admission can shrink
its allowance for larger requests. This budget excludes temporary tensors and KV.
The monitor includes expert residency, hits, read volume and evictions.

This initial cache prioritizes bounded residency and output correctness; synchronous
reads and temporary expert banks can substantially reduce speed. See the
[implementation and decoding investigation](docs/decoding-and-memory.md) for the
mlx-flash/Colibri comparison, measurements, and batching/MTP/DFlash/DSpark findings.
Speculative decoding is served through `--mtp-draft`, single-request only; the
block drafters remain available to the measurement scripts.

Both Qwen profiles use their native 262,144-token limit, with memory admission
accounting for full-attention KV and recurrent state separately. Their SSD cache
can reuse a complete saved token prefix, but cannot rewind recurrent state to an
earlier branch. The K2-specific defaults and context figures below still apply to
the default `k2-horizon` profile.

Local MLX deployment of `abenzerps/K2-Horizon-MoVA-36B-A4B-MLX-4bit`,
model revision `0c57673`, on an M2 Ultra with 64 GB unified memory.
Custom model code is required. Dependencies are pinned in `requirements.lock`.

## Start, monitor, stop

From this directory:

```sh
sh start.sh
sh monitor.sh
```

The monitor refreshes twice per second from a server snapshot sampled approximately
every second. Ctrl-C closes the monitor and leaves the server running.
Use `sh monitor.sh --once` for one snapshot. No additional dependencies are needed.

The console shows:

- Running and queued requests, oldest queue wait, completed and failed counts.
- Per-request phase, queue wait, prefill progress, output tokens, decode speed,
  and time to first token (including queue and prefill).
- Aggregate generation tokens/second over the last five seconds, including reasoning.
- Whole-system CPU and whole-device GPU utilization; server CPU uses 100% per core.
- System unified memory estimates, MLX active/allocator/peak memory, context admission,
  and inactive SSD cache size/hit/miss counts.

GPU utilization comes from macOS IORegistry and includes other applications.
Unsupported counters display unavailable. RAM availability includes free, inactive,
and speculative pages; it is an estimate. MLX memory shares system RAM and should
not be added to it. A snapshot older than four seconds is marked offline/stale.
Local telemetry is in `metrics.json`; request text is not included.

The quantized KV path in this pinned MLX server runs **one request at a time**.
Additional clients queue. The dashboard reports actual concurrency; it does not
introduce parallel decoding or increase the existing concurrency setting.

Stop the background server:

```sh
kill "$(cat server.pid)"
```

Logs: `server.log`. The background process does not start automatically after reboot.

## Endpoint

- Base URL: `http://127.0.0.1:8080/v1`
- LAN: `http://<this-Mac-LAN-IP>:8080/v1` (binds to `0.0.0.0`)
- Model: `k2-horizon`
- No authentication; clients requiring an API key can use `local`.
- Chat completions: `/v1/chat/completions`, including streaming.
- Model discovery: `/v1/models`; `context_length` is the native maximum, not a
  guarantee of currently available memory.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"k2-horizon","messages":[{"role":"user","content":"Explain how a rainbow forms."}],"max_tokens":4096,"stream":true}'
```

High reasoning effort, temperature 1.0 and top-p 0.95 remain defaults.
Reasoning is returned separately in the `reasoning` field.

## Memory-aware context

The fixed 262,144-token cap has been removed. Before each request starts, admission
uses available system RAM, MLX active memory, and Metal's recommended working set.
It estimates 55,296 bytes/token for this model's 48-layer, 4-bit KV cache, including
scale and bias storage, then allows 10% growth overhead and allocation rounding.
The maximum is the model's native **524,288 tokens**, including prompt and output.

Defaults reserve 6 GiB for other system use and 4 GiB for temporary computation.
The output allowance (default 32,768) is reduced to fit the remaining context.
A prompt that leaves no room for output is rejected; prompt text is never silently
truncated. The dashboard records the limit and each request's output allowance in
`metrics.json`. Admission is an estimate made at request start, not a guarantee
against other applications consuming memory later.

Tune reserves at server launch if needed:

```sh
K2_RESERVE_GIB=6 K2_WORKSPACE_GIB=4 K2_SSD_CACHE_GIB=64 sh start.sh
```

Changes apply on restart. Lower reserves permit larger contexts but leave less
room for system activity and temporary tensors. KV storage grows with actual
processed tokens rather than preallocating the maximum. Prefill stays in 128-token
chunks; no sliding window drops older context.

## Inactive KV on SSD

Completed request caches are saved as safetensors under `kv-cache/<model-identity>/`
and released from MLX memory. Matching prompt prefixes are restored and trimmed
for the next request. The token index records the evaluated KV offset, excluding
any final sampled token that has not yet been evaluated. Model/code/tokenizer/runtime
identity prevents reuse across incompatible deployments.

The default SSD budget is 64 GiB per model identity, with least-recently-used
session eviction and 2 GiB free-disk headroom. A zero budget disables persistence.
Disk errors become cache misses or skipped saves rather than failed completions.
Restores exceeding the current estimated KV budget are skipped. Cache tensors
and indexes persist across restarts; they contain conversation-derived data.
Remove the `kv-cache` directory while the server is stopped to clear them.

Saving is synchronous on the generation thread: the completed response can finish
before its cache is saved, but the next request waits for the save. Large caches
can add significant SSD I/O latency. No inactive KV tensor is intentionally kept in
RAM; macOS may retain filesystem pages. Active attention still needs the current
request's KV in unified memory. This is inactive-session offload, not token-level
paging of an active request, and SSD capacity does not extend the native context.

## Validation

```sh
.venv/bin/python -m unittest test_runtime test_models test_flash test_gguf test_iq test_iq_metal test_gguf_reader test_gguf_model -v
# With the server running:
.venv/bin/python check_runtime.py
```

`test_gguf_reader` and `test_gguf_model` read the pinned IQ1 GGUFs when they are
present and skip otherwise. To run a Flash profile straight from GGUF:

```sh
.venv/bin/python check_gguf_model.py --model deepseek-v4-flash --expert-cache-gib 12
```

Unit tests cover memory/native context boundaries, output clipping, queue lifecycle,
quantized cache serialization, branching and exact-prefix restore, restart reuse,
eviction, restore-budget rejection, corrupt-file fallback, and bit-exact GGUF IQ
block decoding.

`runtime-check.json` records two concurrent real requests (normal and streaming),
queue observation, successful arithmetic answers, SSD reuse, throughput, hardware
samples, and return to model-only MLX memory. `check_runtime.py` expects enough free
memory for a context admission above 256K.

Earlier checks remain in `api-check.json`, `capacity-check.json`, and
`checksum-check.log`. The earlier 256K synthetic KV allocation test is not a
full-document prefill or retrieval test. Neither the earlier check nor these short
request tests validate 512K generation quality or long-document retrieval accuracy.
