# Flash model integration and mlx-flash review

Reviewed on 2026-09-09 for this M2 Ultra with 64 GiB unified memory.

## Model selection

| Server alias | Pinned checkpoint | Architecture | Weight tensor bytes |
|---|---|---|---:|
| `qwen3.8-flash` | `mlx-community/Qwen3.8-Flash-Next-4bit` at `07b5dc6c54600a359b87f1e53e7adf6351c72a2c` | `qwen4_exp` | 111,518,894,072 |
| `deepseek-v4-flash` | `mlx-community/DeepSeek-V4-Flash-4bit` at `38c0bd20a6fba70f22c5ee2940ec0092b36ab936` | `deepseek_v4` | 151,482,475,612 |

Sizes come from each downloaded safetensors index, excluding tokenizer/config files.
These are text-only server integrations. The DeepSeek profile selects the original
V4 Flash checkpoint, not Flash-0731 or a vision variant. Other conversions are not
interchangeable merely because they have the same model family name.

[Qwen's release](https://github.com/QwenLM/Qwen3.8-Flash-Next) describes a 125B main
model plus 51B n-gram embeddings, with Gated DeltaNet, sparse attention and gated
residual branches. Its 27B sibling is a different architecture. The selected
[MLX checkpoint](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit) includes
the corrected zero-centered normalization weights; the adapter must not add another
normalization offset during loading.

[DeepSeek's config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)
specifies 43 layers, 256 routed experts, 6 selected experts, compression ratios
0/4/128, hyper-connections, and a native 1,048,576-token context. This runtime uses
the pinned [MLX `_ds4` implementation](https://github.com/spicyneuron/mlx-lm/tree/df50b7a3f370909b3908a024e2f4e2e6fb729521),
including its companion cache, switch and hyper-connection modules. It is not
present in the installed MLX-LM mainline revision.

## Launch and memory behavior

```sh
.venv/bin/python setup_models.py qwen3.8-flash deepseek-v4-flash
sh start.sh --model qwen3.8-flash --expert-cache-gib 24
# Stop that server before selecting the other model:
kill "$(cat server.pid)"
sh start.sh --model deepseek-v4-flash --expert-cache-gib 24
sh monitor.sh
```

The Flash profiles default to a 24 GiB hot-expert ceiling. Dense parameters stay
resident. Cold routed projections remain in their original SSD checkpoint files;
each miss loads the router-selected rows before computation. No experts are skipped.
DeepSeek's separate gate/up tensors are joined only for the selected expert rows,
matching the runtime's fused projection. The cache supports native MXFP4 scales.

Qwen's n-gram table is also kept on SSD. A separate row cache is bounded by both
128 MiB of tensor data and 8,192 entries, limiting Python object overhead for its
small rows. The dashboard reports that cache separately from routed experts.
The normal token embedding and shared experts remain resident.

Context admission can reduce the hot-expert allowance before a request. Qwen retains
4-bit attention KV plus unquantized sparse-index state and recurrent/PLE state.
DeepSeek retains its native, unquantized compressed pools, bounded local attention
windows and compression buffers. Memory estimates include these distinct structures;
a generic full-attention 4-bit formula would be wrong for DeepSeek.
Native context limits are upper bounds, not a promise of available RAM or tested
long-context quality. The 24 GiB expert allowance is not a whole-process RAM limit.

The SSD prompt-cache codec preserves sparse index keys/positions, compression
remainders/overlap, and cache metadata. Flash prefix reuse requires a complete saved
token prefix. Recurrent/compressed state is not rewound to an earlier branch.
Independent-request batching is disabled for these adapters. MTP, DFlash and DSpark
are not enabled by selecting a Flash model.

## What the mlx-flash review changed

There are two projects with the name:

- [matt-k-wong/mlx-flash](https://github.com/matt-k-wong/mlx-flash) primarily streams
  layers. This is useful for dense weights, but routed experts and n-gram lookups
  offer finer read granularity for these models.
- [szibis/mlx-flash](https://github.com/szibis/mlx-flash) uses cached expert slot banks
  and routing-frequency/age tracking. Its inspected expert-streaming path maps
  expert IDs through a lookup and updates residency between tokens; its skip
  fallback changes routing scores. Adopting that path wholesale would not preserve
  our requirement that every selected expert execute with its original weight.

The server now uses a heap for frequency/recency eviction, replacing the linear
scan of all resident projections on each eviction. Stale heap records and old
nonresident routing history are bounded. Up to 64 checkpoint file handles are
reused per tensor index, avoiding an open/close for every row read.

The [upstream MLX-VLM PLE storage](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen4_exp/ple_storage.py)
independently supports reading selected n-gram rows. Its
[expert offload implementation](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/moe_offload.py)
uses a repacked expert store. This workspace reads the existing MLX safetensors
ranges directly, avoiding a second complete copy of hundreds of gigabytes.

Persistent expert slot banks remain a possible optimization. The current path still
constructs temporary selected-expert banks and synchronizes projection completion.
Async prediction/prefetch, active-KV paging and expert pruning are not implemented.
Upstream "near-full speed" claims are not measurements on this Mac.

## Validation

The small-model tests cover both architectures, expert-output parity, n-gram-row
parity, Qwen quantized sparse-cache restoration, DeepSeek hash routing and split
MXFP4 gate/up loading, and DeepSeek prefill across compression boundaries with
restored cache continuation. Existing K2 and Qwen regression tests are retained.

A 64-output-token Qwen3.6 count-sequence check produced identical token IDs with the
enhanced 4 GiB expert cache and fully resident weights. Offload measured 10.53 decode
tokens/sec, 8.97 seconds including prefill, and 5.43 GiB peak MLX allocation. Resident
measured 90.19 decode tokens/sec, 1.29 seconds including prefill, and 18.33 GiB peak.
The cache recorded 56,524 projection hits and 17,273 misses. Downloads were running
in the background, so these are single-run observations, not controlled benchmark
averages. They compare offload with residency, not the old cache with the new heap.
Artifacts: `qwen3.6-enhanced-offload.json` and `qwen3.6-enhanced-resident.json`.

Both Flash profiles also passed HTTP completions, chat, streaming and exact-prefix
SSD reuse using small local fixtures (`flash-http-fixtures.json`). Model downloads
were paused at the user's request. Full Flash checkpoint validation remains pending.
Do not infer full-checkpoint output quality or long-context reliability from the
small-model tests.

## Reproducibility

`vendor/sources.lock.json` records source revisions and file hashes. Upstream MIT
licenses are retained. The MLX-VLM dependency is reduced to its text import closure;
package initializers no longer eagerly register multimodal processors. The DeepSeek
module imports common MLX-LM helpers from the existing runtime. Neither change
upgrades MLX/MLX-LM. Pillow 12.3.0 is the only added installed package, required by
the upstream shared model-base module.
