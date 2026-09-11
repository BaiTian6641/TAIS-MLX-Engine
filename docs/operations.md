# Operations

Everything an operator needs: install, configure a Hub mirror, serve, measure, and
the failure modes this engine has actually hit.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock      # exact versions the engine is tested against
.venv/bin/pip install -e .                      # optional: the `k2mlx` command
```

`mlx` and `mlx-lm` in `requirements.lock` are pinned to revisions this engine has
been measured against; the versions in `pyproject.toml` are looser bounds for
people installing from a package index. `k2mlx doctor` reports both, and flags any
drift.

## Quick start

```sh
k2mlx doctor                                    # environment, dependencies, disk, mirror
k2mlx models                                    # which profiles have weights on disk
k2mlx download qwen3.6-35b-a3b                  # fetch one (mirror-aware)
k2mlx serve --model qwen3.6-35b-a3b --port 8081 --detach
k2mlx bench api --model qwen3.6-35b-a3b --port 8081
k2mlx stop
```

Without the package install, every command has a script equivalent:
`python cli.py doctor`, `python serve.py --model …`, `python setup_models.py …`.

## Command line

```
k2mlx serve [--detach] --model <profile> [engine options]
k2mlx stop [--timeout SECONDS]
k2mlx models [--json]
k2mlx download <profile>... [--metadata-only] [hub options]
k2mlx doctor
k2mlx bench <kind> --model <profile> [--port N] [...]
k2mlx version
```

`serve` passes anything after `--model` straight to the engine, so
`k2mlx serve --model ornith-1.5-35b-a3b --kv-bits 0 --decode-concurrency 4` is the
same as running `serve.py` with those flags. Without `--detach` the server runs in
the foreground and logs to stdout, which is what a container wants; with
`--detach` it writes `server.pid` and logs to `server.log`.

`bench` runs the measurement scripts against a running server:

| kind | script | what it establishes |
|---|---|---|
| `api` | `check_profile_api.py` | chat, streaming and exact-prefix continuation answer correctly |
| `concurrency` | `check_concurrency.py` | concurrent answers are byte-identical to sequential ones |
| `context` | `check_context_scaling.py` | decode and prefill against context length |
| `optimizations` | `check_optimizations.py` | prefill, prompt reuse and batch throughput under one labelled configuration |
| `speculative` | `check_speculative.py` | draft acceptance and speedup, and that output is unchanged |
| `diffusion` | `check_diffusion.py` | block-diffusion sampling |
| `runtime`, `capacity`, `gguf-parity` | the matching `check_*.py` | device behaviour, context admission, quantized-weight parity |

## Hugging Face mirrors

Weights are fetched through `huggingface_hub`, so any Hub-compatible mirror works.
Each setting is taken from a flag first, then a `K2MLX_` variable, then the
conventional variable:

| setting | flag | environment |
|---|---|---|
| endpoint or mirror | `--hf-endpoint https://hf-mirror.com` | `K2MLX_HF_ENDPOINT`, then `HF_ENDPOINT` |
| access token | `--hf-token hf_…` | `K2MLX_HF_TOKEN`, then `HF_TOKEN` |
| download cache | `--hf-home /data/hf` | `K2MLX_HF_HOME`, then `HF_HOME` |
| offline mode | `--hf-offline` | `HF_HUB_OFFLINE=1` |

```sh
k2mlx download qwen3.6-35b-a3b --hf-endpoint https://hf-mirror.com
K2MLX_HF_ENDPOINT=https://hf-mirror.com k2mlx download gemma4-31b
k2mlx download smollm3-3b --hf-offline        # serve from cache, never call out
```

The endpoint is normalised to scheme and host, because a path pasted from a model
page would otherwise be glued onto every request. `k2mlx doctor` prints the
effective endpoint, whether a token is set, and the cache directory, so a
deployment can record where its weights came from. Endpoints are also passed
explicitly to `snapshot_download` and `HfApi`, not only through the environment.

## Serving options

| option | default | effect |
|---|---|---|
| `--model` | `k2-horizon` | profile from `model_profiles.PROFILES`; the alias is the API model name |
| `--host`, `--port` | `0.0.0.0`, `8080` | listen address |
| `--model-path` | profile default | override the checkpoint directory |
| `--kv-bits {0,4,8}` | `4` | KV cache width. `0` keeps bf16, which is what enables batching and prompt-prefix reuse; `4` is the smallest; `8` is a middle ground for long context |
| `--quantized-kv-start N` | `0` | keep the first N tokens in bf16 and compress only the tail |
| `--prompt-cache-size N` | `1` | prompt-prefix caches held in memory |
| `--prefill-step-size N` | `512` | chunk size for prefill. Not a throughput knob; it is the knob that keeps a very long prefill inside the window macOS gives a Metal command buffer |
| `--decode-concurrency N` | `1` | batch size for decoding (needs unquantized KV) |
| `--prompt-concurrency N` | `1` | batch size for prefill |
| `--expert-cache-gib G` | off | stream cold routed experts from SSD, keeping G GiB resident |
| `--mtp-draft DIR` | off | serve with multi-token-prediction speculation (single request only) |

Flags that cannot work together are refused at startup with an explanation rather
than failing mid-request: batching with quantized KV, batching with a
non-mergeable cache, KV quantization on a cache type that raises, and MTP with
concurrency.

Which families quantize their KV cache is declared in `model_profiles.UNQUANTIZED_KV`
and re-checked against the loaded model's own caches, because getting it wrong is
not cosmetic: sliding-window layers build a rotating cache that raises on
quantization, GLM-4.7-Flash's MLA attention unpacks `update_and_fetch` into two
names where a quantized cache returns four, and the rest of that list hold only
recurrent state, where the flag would be a silent no-op.

## Models

`k2mlx models` prints this table live, including download state; the measurements
below are from `docs/decoding-and-memory.md`, on an M2 Ultra with 64 GiB.

| profile | resident | decode | notes |
|---|---:|---:|---|
| `k2-horizon` | 19.6 GiB | 46.6 tok/s | dual attention + routed MoVA experts, 524k context |
| `gemma4-26b-a4b` | 14.3 GiB | 191.7 tok/s | MoE, sliding-window KV, batches at 1.8x |
| `gemma4-31b` | 16.1 GiB | 22.6 tok/s | dense 60 layers, bandwidth-bound |
| `qwen3.8-27b` | 15.0 GiB | 67.6 tok/s | dense |
| `qwen3.6-35b-a3b` | 19.0 GiB | 77.5 tok/s, 93.2 with MTP | hybrid linear attention + 256 experts |
| `ornith-1.5-35b-a3b` | 18.2 GiB | 78.2 tok/s, 94.4 with MTP | fine-tune of the same base |
| `nemotron-3.5-30b-a3b` | 16.6 GiB | 86.2 tok/s | 23 Mamba / 23 MoE / 6 attention layers |
| `qwen3.5-9b` | 4.8 GiB | 70.9 tok/s | dense |
| `qwen3.5-4b` | 2.3 GiB | 94.8 tok/s | dense |
| `llama-3.2-3b` | 1.7 GiB | 150.0 tok/s | dense |
| `smollm3-3b` | 1.6 GiB | 115.2 tok/s | dense |
| `muse-glimmer-30b` | 18.1 GiB | 26.7 tok/s | dense, 52 layers alternating sliding-window and NoPE attention |
| `glm-4.7-flash` | 15.7 GiB | 48.2 tok/s | 64-expert MoE; keep thinking enabled, it answers worse without |
| `gpt-oss-20b` | 11.3 GiB | 82.5 tok/s | MXFP4 MoE, harmony channel output, YaRN from the vendor config |
| `minicpm5-2b` | 1.4 GiB | 133.0 tok/s | dense, fastest in the table |
| `spark-x2.5-4b` | 8.2 GiB | 55.3 tok/s | vendored runtime, bf16: fused QKV, per-head sigmoid gate, per-layer-type rope, 1,048,576 native |
| `qwen3.8-flash`, `deepseek-v4-flash` | — | 11.9 / 3.3 tok/s | streamed IQ1 GGUF; large models that do not fit resident |

Thinking models answer in the `reasoning` field and leave `content` empty until
they finish; raise `max_tokens` rather than assuming the model failed.

## Extending context

A checkpoint trained at 128K usually tolerates more once its rotary embeddings are
rescaled, which is what YaRN does. `k2mlx context` writes that scaling into the
config and raises the declared maximum:

```sh
k2mlx context --model minicpm5-2b --factor 2 --dry-run   # 131,072 -> 262,144
k2mlx context --model glm-4.7-flash --factor 2           # 202,752 -> 405,504
k2mlx context --model minicpm5-2b --restore              # back to the original
```

The original config is kept as `config.json.pre-yarn`, and the command refuses to
overwrite scaling a checkpoint already declares unless `--force` is given.
`gpt-oss-20b` ships YaRN from the vendor (factor 32 over a 4K window); doubling it
means `--original 4096 --factor 64`.

Two caveats worth knowing before extending anything:

- YaRN restores *usable* length, not the accuracy of the original window. Expect
  the trained range to be unaffected and quality to fall off with distance beyond
  it.
- It is not universal, and the pattern is architectural. Every profile was tried
  and verified by serving a request afterwards; the ones that work have plain
  (or MLA) attention, and the ones that fail have recurrent or hybrid state:

  | Extended | From | To | |
  |---|---:|---:|---|
  | `minicpm5-2b` | 131,072 | 262,144 | verified |
  | `smollm3-3b` | 65,536 | 262,144 | verified |
  | `llama-3.2-3b` | 131,072 | 262,144 | verified (overrides its llama3 scaling) |
  | `gpt-oss-20b` | 4,096 | 262,144 | verified (on the vendor's own factor-32) |
  | `glm-4.7-flash` | 202,752 | 405,504 | verified (MLA attention) |
  | `nemotron-3.5-30b-a3b` | 262,144 | 524,288 | verified (Mamba/MoE/attention hybrid) |
  | `muse-glimmer-30b` | — | — | hangs: full layers carry no positional embedding |
  | `qwen3.5-9b`, `qwen3.6-35b-a3b`, `ornith-1.5-35b-a3b`, `qwen3.8-27b` | — | — | hang: hybrid linear attention |
  | `gemma4-26b-a4b`, `gemma4-31b` | — | — | hang: sliding-window layers with a rotating cache |
  | `spark-x2.5-4b` | — | — | not needed: 1,048,576 native |

  "Hangs" means the request never returns while the server stays up, which is why
  the extension command keeps the original config and why every attempt above was
  followed by a served request rather than a config check. `--restore` is the way
  back and was exercised on real checkpoints.

  The models that refuse are already at 262,144 or more, so nothing is lost: the
  Qwen3.5 family and Nemotron are at 262,144 and Spark at 1,048,576 natively.

## Channel output

Two served models do not emit plain text. GPT-OSS wraps its reply in a harmony
envelope (`<|channel|>analysis<|message|>…<|channel|>final<|message|>…`) and Muse
Glimmer addresses each message (`to=self` for its own reasoning, `to=user` for the
reply). Both are answering correctly; a server that forwards the raw stream shows
template markers and private reasoning. `output_channels.py` rewrites the envelope
into the ` thinking`/`<｜end▁of▁thinking｜>` convention the runtime's state machine already
understands, so `content` carries the answer and `reasoning` the deliberation. The
transform is stateful because markers straddle token boundaries, and its streaming
output is tested to equal its one-shot output for every chunk size.

## Troubleshooting

**`[METAL] Command buffer execution failed: Impacting Interactivity`.** macOS killed
a GPU command that ran too long. It happens on very long prefills at the default
chunk size; a 198k-token prompt is killed outright. Lower it:

```sh
k2mlx serve --model nemotron-3.5-30b-a3b --prefill-step-size 64
```

Measured: 198,027 tokens in 1,004.6 s at step 64.

**Batching refuses to start.** `--decode-concurrency` needs `--kv-bits 0`; the
pinned runtime cannot merge quantized caches. The startup message says so.

**Prompt reuse does nothing.** It needs `--kv-bits 0` (trimming a quantized cache
is unsupported) *and* a request that extends a cached sequence. Exact repeats hit;
a chat turn usually does not, because a rendered conversation is not a token
prefix of its own next turn. This is measured in
`docs/decoding-and-memory.md`, and it matches what vLLM reports in production.

**A model refuses to load or behaves oddly after switching profiles.** Only one
server at a time: `k2mlx stop`. The engine refuses to start when `server.pid`
names a live process.

**Out of memory.** Admission control clips the context rather than the process.
`--expert-cache-gib` reduces resident experts for the streamed profiles, and
`--kv-bits 4` halves KV cost against bf16.

## Where the code lives

| module | responsibility |
|---|---|
| `serve.py` | HTTP server entry point: policy hooks, prompt cache, telemetry, MTP wiring |
| `model_profiles.py` | profile table, option parsing, cache fingerprint |
| `runtime_support.py` | context admission, host/GPU sampling, cache-capability probing |
| `disk_cache.py` | SSD-backed prompt cache |
| `expert_cache.py` | disk hot-set cache for routed experts |
| `flash_models.py` | adapters for the Flash/Qwen4Exp and Gemma 4 architectures |
| `gguf_model.py`, `gguf_reader.py`, `gguf_import.py` | GGUF-resident models, container reader, importer |
| `iq_quants.py`, `iq_metal.py` | IQ block decoders (MLX ops and Metal kernels) |
| `mtp_speculation.py` | MTP drafter and the speculative round loop |
| `diffusion_engine.py` | DiffusionGemma block sampler |
| `telemetry.py`, `monitor.py` | metrics and the console monitor |
| `hf_env.py` | Hub endpoint, token and cache configuration |
| `cli.py` | the `k2mlx` command surface |
| `check_*.py` | measurements; each writes the JSON its documentation cites |
| `test_*.py` | regression tests; they skip when a checkpoint is absent |
