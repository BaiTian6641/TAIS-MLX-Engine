# Gemma 4 support

Researched 2026-09-10. Gemma 4 is Google's released family (Apache 2.0): E2B,
E4B, 12B, 31B and the 26B-A4B mixture-of-experts, plus a 12B Unified variant and
a separate block-diffusion model, DiffusionGemma. This engine serves the
**text** path of the 26B-A4B checkpoint through the pinned `mlx_lm`, which
already ships a `gemma4_text` implementation.

## What the checkpoint looks like

`mlx-community/gemma-4-26b-a4b-it-4bit` is a multimodal conversion: its top-level
`model_type` is `gemma4` and it carries `text_config`, `vision_config` and
`audio_config` sub-configs. The language weights are prefixed
`language_model.`; the vision tower and the embedding projection are separate
tensor trees that a text-only server never touches.

Text configuration (from the checkpoint):

| | |
|---|---|
| `model_type` | `gemma4_text` |
| Layers / hidden | 30 / 2816 |
| Heads | 16 query, 8 KV, head_dim 256 |
| Attention pattern | 5 sliding-window layers per full-attention layer |
| Sliding window | 1024 |
| Quantisation | affine, group 64, 4-bit |

Two properties make it cheap to serve on a 64 GiB machine: most layers only keep
a 1024-token sliding window, and the full-attention layers share KV state with
the sliding ones rather than each holding their own. There is no recurrent state
and no per-layer embedding table. Its norms are plain RMSNorm weights — unlike
Qwen's zero-centred convention, so no `+1` adjustment is needed.

## How this engine supports it

`flash_models._gemma4_classes` returns a text-only view: `TextArgs` builds
`mlx_lm.models.gemma4_text.ModelArgs` from `text_config`, and `TextModel.sanitize`
keeps only the `language_model.` tensors, strips that prefix, and then defers to
the upstream `sanitize` (which fuses each layer's expert gate/up pair). The
loader registration in `flash_models.register` injects those classes for
`model_type == "gemma4"`, and `model_profiles` exposes the profile
`gemma4-26b-a4b`; `setup_models.py` downloads and pins the revision.

```sh
.venv/bin/python setup_models.py gemma4-26b-a4b
sh start.sh --model gemma4-26b-a4b
```

The text server loads only the language model; the vision and audio towers are
stripped. Images are served separately through the full VLM — see **Vision**
below.

KV stays **unquantized** for this family. The pinned runtime cannot quantize a
sliding-window cache (`RotatingKVCache.to_quantized` raises "Quantization NYI"),
so `model_profiles.uses_quantized_kv` lists the families that get no `--kv-bits`
flags, and `ContextPolicy` accounts for the hybrid cache directly: only the five
full-attention layers grow with the context (40,960 bytes/token), while the 25
sliding layers are bounded by their 1024-token window.

Measured on this machine:

| | |
|---|---:|
| Load | 1.7 s |
| MLX active | 14.2 GiB |
| Decode | **191.7 tok/s** |
| Context cost | 40,960 bytes/token |

That makes Gemma 4 the fastest profile in this engine, ahead of Qwen3.6-35B-A3B
(169.5 tok/s) — the MoE spine with mostly sliding-window attention and only five
growing KV layers is cheap on every axis.

## Vision

`gemma4-26b-a4b` and `gemma4-31b` also serve images. `serve.py` routes these
profiles through `install_vision()`: the checkpoint loads as a complete VLM via
`vision_engine.load_vision_model` (the vendored `flash_vlm` Gemma 4 — text
*and* vision tower), and a request carrying image parts runs through the VLM
instead of the text-only path. It stays single-request, because a prefill that
needs pixel values cannot go through the batch generator.

The pieces:

- **`vision_engine.py`** — loads the VLM and generates. The loader reproduces
  `mlx_lm`'s mixed-precision quantization: a per-path override in the
  checkpoint's `quantization` map wins (the MoE routers are 8-bit here), else a
  module is quantized iff the checkpoint carries `<path>.scales` for it — so
  the bf16 vision tower is left alone automatically.
- **`input_parts.extract_vision_messages`** — normalises an agentic content
  list into template-ready parts: text/tool parts flatten to text, each image
  part stays an `{"type": "image"}` marker, and the images are decoded (base64
  data URL, raw base64, or `http(s)` URL) in encounter order.
- **`VisionModel.build_inputs`** — applies the chat template (image markers
  become `<|image|>`), preprocesses the images to `pixel_values`, then expands
  every `<|image|>` into `{boi}{<|image|> x n}{eoi}` with `n` = that image's
  soft-token count, before tokenizing. The VLM scatters the vision features at
  those `<|image|>` positions in the embedding.

Verified end to end over HTTP (`check_vision.py`): colour identification, OCR,
and object counting all answer correctly on both profiles, and a text-only
request through the same server is unchanged. The thinking channel is split
into `reasoning`/`content` like every other profile. `gemma4-31b` is verified
in-process only (same code path; not re-measured over HTTP).

```sh
.venv/bin/python check_vision.py --model gemma4-26b-a4b
```

## DiffusionGemma

`google/diffusiongemma-26B-A4B-it` is a block-diffusion model on the same 26B-A4B
backbone: a causal encoder consumes the prompt and fills a hybrid cache, then a
decoder refines a 256-token canvas with bidirectional attention over that cache
and the canvas, up to 48 denoising steps, with the previous step's logits fed
back as self-conditioning. Positions the entropy rule rejects are re-randomised
and re-denoised; the final step's argmax canvas is committed and re-encoded to
extend the cache.

This engine runs it. `vendor/flash_vlm/models/diffusion_gemma/` (with the
`gemma4` files it imports) is the MLX-VLM implementation at the same pinned
revision as the rest of the vendored tree, and `diffusion_engine.py` implements
the sampler: the entropy-bound acceptance rule the model card requires, the
linear 0.8 → 0.4 temperature schedule, and the canvas commit loop.
`check_diffusion.py` drives it and `test_diffusion.py` covers the sampler rules.

Two things are worth knowing before using it:

- **The chat template is mandatory.** Feeding a bare prompt produces degenerate
  token salad — and so does the upstream reference on the same checkpoint, which
  is how this was isolated. With the template the model answers correctly
  ("What is the capital of France? Answer in one word." → `thought\nParis`).
- **Quantisation comes from the checkpoint, not from a rule.** The config's
  per-module map lists overrides only, so `check_diffusion.py` derives each
  module's bit width from its packed weight and scale shapes; guessing a blanket
  4-bit for everything leaves the layer-0-2 and embedding tensors mismatched.

Measured on this machine with the 4-bit MLX checkpoint:

| | |
|---|---:|
| Load | 0.7 s |
| MLX active | 16.5 GiB |
| One canvas pass (256 positions x 48 steps) | ~27 s |
| Canvas throughput | ~9 tok/s |

### Batching

`diffusion_engine.generate` takes a `batch_size`: that many independent canvases
share one decoder pass, each with its own canvas and stop state. This is ahead
of the upstream MLX-VLM sampler, which raises "only supports batch size 1" in
both of its entry points; the transformers implementation and the vLLM one are
batched (vLLM caps diffusion at `--max-num-seqs 4` for state memory).

Measured with a prompt that fills the canvas:

| batch | wall | tokens | aggregate | per sequence |
|---:|---:|---:|---:|---:|
| 1 | 32.3 s | 256 | **7.92 tok/s** | 7.92 tok/s |
| 4 | 102.3 s | 1024 | **10.01 tok/s** | 2.50 tok/s |

Batching buys 26% aggregate throughput at batch 4 while tripling per-sequence
latency. That matches the published picture: diffusion's parallelism is mostly
*within* a sequence (the 256-position canvas is refined in one pass), so adding
sequences competes with work the canvas already parallelises. An independent
A800 study found autoregressive decoding ahead of block diffusion at every batch
size, with diffusion ahead only up to roughly batch 2-4, and Google's own
throughput table shows the same shape on an H100.

The same caveat applies here as in Google's documentation: at 7.9-10 tok/s this
is roughly twenty times slower than the autoregressive Gemma 4 profile above
(191.7 tok/s) on the same machine, and unified-memory devices "may not see the
same acceleration over autoregressive models".
