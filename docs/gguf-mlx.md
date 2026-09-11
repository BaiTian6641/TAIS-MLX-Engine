# GGUF weights with MLX inference

This is a GGUF import path into the existing MLX server. No llama.cpp inference
backend or executable is installed or used. The `gguf` Python package supplies file
parsing and reference block decoders; MLX performs requantization and inference.

The first implementation writes a reusable MLX safetensors directory, processing
rows in bounded chunks. It does not execute K/IQ-packed GGUF tensors directly with
custom Metal kernels. The original GGUF files remain unchanged.

## Formats and architecture limits

| Input | Import behavior |
|---|---|
| Q4_0, Q4_1, Q8_0 | Repack the original quantization grid into MLX affine storage; no second quantization |
| F32, F16, BF16 | Read floating-point values; BF16 is represented as F32 in this importer |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | Explicit requantization into MLX 2/3/4/6/8-bit affine weights |
| IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL | Explicit requantization; covered by mixed-format fixture tests |
| Other formats decoded by pinned `gguf==0.19.0` | Compatibility path is available, but not individually validated here |
| Missing decoder, big-endian GGUF, unknown tensor layout | Rejected with an error |

Names such as `Q4_K_M` describe a mixture of tensor encodings; inspection reports
the actual tensor types. `--requantize-bits` is required for K/IQ imports. It also
requantizes other eligible matrices in the input. This can change model quality;
it does not preserve the original K/IQ calibration. Without that option, an
unsupported native format is rejected before conversion starts.

Validated architecture mappings are `llama`, `qwen35`, and `qwen35moe`, corresponding
to MLX `llama`, `qwen3_5`, and `qwen3_5_moe`. This covers the existing Qwen3.8-27B and
Qwen3.6-35B-A3B profiles. Supply the matching model's local config and tokenizer.
Do not substitute a config merely because its dimensions are similar.

**Qwen Flash-Next (`qwen4_exp`), DeepSeek V4, and K2 MoVA GGUF mappings are not yet
implemented.** Their MLX checkpoint support is separate. PLE tables, hyper-connections,
sparse-index tensors, and MTP variants need their own validated import transforms.
Extra/unmapped tensors, including explicit rotary-frequency overrides, are rejected.

## Usage

Inspect local files without importing MLX or allocating model tensors:

```sh
.venv/bin/python gguf_import.py --gguf /path/model.gguf --inspect
```

Preserve a compatible Q4_0/Q4_1/Q8_0 model's grids:

```sh
.venv/bin/python gguf_import.py \
  --gguf /path/model-Q4_0.gguf \
  --config-dir models/qwen3.6-35b-a3b \
  --output models/qwen3.6-from-gguf
```

Explicitly convert a K/IQ model into MLX quantization:

```sh
.venv/bin/python gguf_import.py \
  --gguf /path/model-Q4_K_M.gguf \
  --config-dir models/qwen3.6-35b-a3b \
  --output models/qwen3.6-from-gguf \
  --requantize-bits 4 --group-size 32 --chunk-mib 32
```

For a split GGUF, pass all shards to `--gguf`, for example
`--gguf /path/model-00001-of-00003.gguf /path/model-00002-of-00003.gguf /path/model-00003-of-00003.gguf`.
Missing and duplicate tensors are checked before writing output. Import never
fetches a model from the network.

Serve the resulting weights through MLX:

```sh
sh start.sh --model qwen3.6-35b-a3b \
  --model-path models/qwen3.6-from-gguf --expert-cache-gib 12
sh monitor.sh
```

For an imported dense Llama model use `--model llama --model-path /path/imported`.
Its tokenizer must include a suitable chat template for chat completions.

The importer refuses an existing output directory. It writes into a staging directory,
checks approximate output disk requirements with 2 GiB headroom, and renames the
completed directory into place. Failures remove that staging directory. Inputs are
memory-mapped; conversion does not expand the entire model into FP16. The chunk
setting controls row-batch working size, not total process RSS or OS file caching.
One source tensor produces one safetensors shard, with its packed weight/scales/biases
kept together. This favors bounded conversion memory over minimizing file count.

Exact native repacking uses F32 scale/bias metadata to retain the decoded grid, so
its storage overhead can differ from the source GGUF. Context admission accounts
conservatively for the imported model's F32 cache metadata and recurrent buffers.

## Verification and sources

Local tests cover exact Q4_0/Q4_1/Q8_0 decoded-weight equality, Llama rotary layout,
Qwen GDN head-layout reversal with unequal key/value head counts, full-model forward
parity on F32 fixtures, and SSD expert-offload parity after Q4_0 import. A Q4_K fixture
was imported at every supported target bit width and compared with independently
quantized MLX models. A mixed K/IQ fixture was also checked end to end. These are
small fixtures, not full published-model quality benchmarks.

[MLX's GGUF example](https://github.com/ml-explore/mlx-examples/tree/main/llms/gguf_llm)
documents native support for Q4_0/Q4_1/Q8_0 and expansion of other formats. That is why
this importer does not blindly call `mx.load` on a large K/IQ file. The
[GGUF Python decoders](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py)
provide the block-level compatibility decoding. Qwen layout reversal follows the
[exporter's GDN transformations](https://github.com/ggml-org/llama.cpp/blob/434ddbbc0e30522e897670681e503b797c12b7c1/conversion/qwen.py),
including tiled value heads, convolution layout and negative-exponential decay.
The config/tokenizer and strict tensor checks remain necessary; reading a container
format alone does not implement every architecture stored in it.
