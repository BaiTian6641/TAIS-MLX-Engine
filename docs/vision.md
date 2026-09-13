# Vision

Multimodal profiles serve images over the same OpenAI endpoint as text. A
request carrying `image_url` parts (base64 data URL, raw base64, or `http(s)`)
is decoded and run through the checkpoint's vision tower; a text-only request
takes the same path, since a VLM generates text identically.

## Profiles

| profile | family | verified |
|---|---|---|
| `gemma4-26b-a4b` | `gemma4` | over HTTP: colour, OCR, counting |
| `gemma4-31b` | `gemma4` | in-process (same code path) |
| `qwen3.8-27b` | `qwen3_5` | over HTTP: colour, OCR, counting |

`qwen3.6-35b-a3b` and `muse-glimmer-30b` also ship vision weights, but their
VLM implementations are a separate integration (the `qwen3_5_moe` and
muse-glimmer model classes). `ornith-1.5-35b-a3b` and `glm-4.7-flash` are
text-only conversions with no vision weights.

## How it works

`serve.py` routes a vision profile through `install_vision()`: the checkpoint
loads as a complete VLM (text **and** vision tower) via
`vision_engine.load_vision_model`, and every request — image or text — runs
through it. It stays single-request, because a prefill that needs pixel values
cannot go through the batch generator.

- **`vision_engine.py`** loads the VLM and generates. The loader reproduces
  `mlx_lm`'s mixed-precision quantization: a per-path override in the
  checkpoint's `quantization` map wins, else a module is quantized iff the
  checkpoint carries `<path>.scales` for it — so a bf16 vision tower is left
  alone automatically.
- **`input_parts.extract_vision_messages`** normalises an agentic content list
  into template-ready parts: text/tool parts flatten to text, each image part
  stays an `{"type": "image"}` marker, and the images are decoded in order.
- **`VisionModel.build_inputs`** applies the chat template, preprocesses the
  images, expands each image marker to that image's soft-token count, and
  tokenizes.

### Family differences

| | `gemma4` | `qwen3_5` |
|---|---|---|
| template marker | `<\|image\|>` | `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` |
| marker expands to | `{boi}{<\|image\|> x n}{eoi}` | `<\|image_pad\|> x n` |
| soft tokens `n` | from the image processor | `prod(grid_thw) / merge_size**2` |
| prefill inputs | `pixel_values` | `pixel_values` + `image_grid_thw` |

The VLM scatters the vision features at the marker positions in the embedding,
so the text and image streams are one sequence.

## Verifying

```sh
.venv/bin/python check_vision.py --model qwen3.8-27b
.venv/bin/python check_vision.py --model gemma4-26b-a4b
```

Each run writes `<profile>-vision-check.json` with the answers and timings.
`test_vision.py` covers the model-free logic (decoding, content normalisation,
marker expansion) and skips the expansion test when the checkpoint is absent.

## Notes

- An agentic request renders exactly as it does on the text path: the request's
  `tools` and the profile's `chat_template_args` are passed to the template, and
  `function.arguments` is normalised from the wire format's JSON string to a
  mapping (the Qwen template iterates `tool_call.arguments|items`, which raises
  on a string).
- Thinking models answer after a thought block; the output-channel handling
  splits the reply so `content` holds the answer and `reasoning` the trace.
- The image is resized by the family's own processor (aspect-ratio preserving),
  so a large image costs more soft tokens and prefill time, not accuracy.
- `gemma4-31b` is verified in-process only; it uses the identical code path as
  `gemma4-26b-a4b` but was not re-measured over HTTP.
