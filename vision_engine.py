"""Vision-capable generation for the engine's multimodal profiles.

The text-only serving path strips each multimodal checkpoint's vision tower
(``mlx_lm.gemma4_text`` for Gemma 4) so it cannot consume images. This module
loads the *full* VLM from the vendored ``flash_vlm`` implementation and runs
image + text generation:

1. apply the chat template (image content parts become markers),
2. preprocess images to ``pixel_values`` (+ ``image_grid_thw`` where the family
   needs it),
3. expand every image marker to that image's soft-token count,
4. tokenize, prefill with the pixel values, then decode autoregressively.

Two families are supported, because their image plumbing differs:

* ``gemma4`` - the template emits ``<|image|>``; the marker expands to
  ``{boi}{<|image|> x n}{eoi}`` with ``n`` from the image processor.
* ``qwen3_5`` - the template emits ``<|vision_start|><|image_pad|><|vision_end|>``;
  ``<|image_pad|>`` expands to ``prod(grid_thw) / merge_size**2`` copies, and
  the vision tower also needs ``image_grid_thw``.

The loader reproduces ``mlx_lm``'s mixed-precision quantization: a per-path
override in the checkpoint's ``quantization`` map wins, otherwise a module is
quantized iff the checkpoint carries ``<path>.scales`` for it - so a bf16
vision tower is left alone automatically.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

__all__ = ["VisionModel", "load_vision_model", "VISION_PROFILES"]


# Profiles whose checkpoints ship a vision tower reachable through a vendored
# flash_vlm implementation.
VISION_PROFILES = frozenset({"gemma4-26b-a4b", "gemma4-31b", "qwen3.8-27b"})

# family -> the config model_type(s) it serves
_FAMILY_BY_MODEL_TYPE = {"gemma4": "gemma4", "qwen3_5": "qwen3_5"}


class VisionModel:
    """A loaded VLM plus its tokenizer and image preprocessor."""

    def __init__(self, model, tokenizer, image_processor, config, family="gemma4"):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config
        self.family = family

    # -- input construction ------------------------------------------------
    def build_inputs(
        self,
        messages: List[Dict[str, Any]],
        images: Sequence[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        chat_template_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[mx.array, Dict[str, Any]]:
        """Return ``(input_ids, model_kwargs)`` for a chat conversation.

        ``model_kwargs`` carries the prefill-only inputs (``pixel_values`` and,
        for qwen, ``image_grid_thw``); it is empty for a text-only turn.

        ``tools`` and ``chat_template_args`` are the same values the text path
        hands the template, so an agentic request renders identically.
        """
        tok = self.tokenizer
        text = tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_args or {}),
        )
        extra: Dict[str, Any] = {}
        if images:
            counts, extra = self._preprocess(images)
            text = self._expand_markers(text, counts)
        input_ids = mx.array(tok.encode(text, add_special_tokens=False))
        return input_ids, extra

    def _preprocess(self, images: Sequence[Any]) -> Tuple[List[int], Dict[str, Any]]:
        """Run the image processor; return per-image soft-token counts + inputs."""
        if self.family == "qwen3_5":
            data = self.image_processor(images=list(images))
            grid = data["image_grid_thw"]
            merge = int(self.image_processor.merge_size) ** 2
            counts = [int(g[0]) * int(g[1]) * int(g[2]) // merge for g in grid]
            return counts, {
                "pixel_values": mx.array(data["pixel_values"]),
                "image_grid_thw": mx.array(grid),
            }
        data, counts = self.image_processor(list(images))
        return list(counts), {"pixel_values": mx.array(data["pixel_values"])}

    def _expand_markers(self, text: str, counts: Sequence[int]) -> str:
        """Replace each image marker with that image's soft-token sequence.

        ``re.sub`` never re-scans its own replacement, so a replacement that
        itself contains the marker is safe.
        """
        tok = self.tokenizer
        if self.family == "qwen3_5":
            replacements = [tok.image_token * n for n in counts]
        else:
            replacements = [
                f"{tok.boi_token}{tok.image_token * n}{tok.eoi_token}" for n in counts
            ]
        it = iter(replacements)
        return re.sub(re.escape(tok.image_token), lambda _: next(it), text)

    # -- generation ---------------------------------------------------------
    def generate_tokens(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        images: Sequence[Any] = (),
        max_tokens: int = 256,
        sampler=None,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        chat_template_args: Optional[Dict[str, Any]] = None,
        inputs: Optional[Tuple[mx.array, Dict[str, Any]]] = None,
    ):
        """Yield generated token ids one at a time (stops at an eos token).

        ``inputs`` short-circuits input construction when the caller has
        already built them (the serving path does, to render the context).
        """
        from vendor.flash_vlm.models.cache import make_prompt_cache

        if inputs is None:
            inputs = self.build_inputs(
                messages or [], images, tools=tools, chat_template_args=chat_template_args
            )
        input_ids, extra = inputs
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        cache = make_prompt_cache(self.model.language_model)
        eos = set(self._eos_ids())

        def pick(logits):
            if sampler is not None:
                t = sampler(logits)
                return t if getattr(t, "ndim", 0) > 0 else t[None]
            return _sample(logits, temperature)

        logits = self.model(input_ids, cache=cache, **extra)
        next_id = pick(_logits_of(logits)[:, -1, :])
        token = int(next_id.item())
        for _ in range(max_tokens):
            if token in eos:
                break
            yield token
            logits = self.model(next_id[None], cache=cache)
            next_id = pick(_logits_of(logits)[:, -1, :])
            token = int(next_id.item())

    def generate(
        self,
        messages: List[Dict[str, Any]],
        images: Sequence[Any] = (),
        max_tokens: int = 256,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        chat_template_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        out = list(self.generate_tokens(
            messages, images, max_tokens,
            temperature=temperature, tools=tools, chat_template_args=chat_template_args,
        ))
        return self.tokenizer.decode(out, skip_special_tokens=True)

    def _eos_ids(self) -> List[int]:
        tok = self.tokenizer
        ids = list(getattr(tok, "eos_token_ids", None) or [])
        if not ids and tok.eos_token_id is not None:
            ids.append(tok.eos_token_id)
        extra = getattr(self.config, "eos_token_id", None)
        if isinstance(extra, (list, tuple)):
            ids.extend(int(e) for e in extra)
        elif extra is not None:
            ids.append(int(extra))
        return sorted({i for i in ids if i is not None})


def _logits_of(output) -> mx.array:
    """The model returns a LanguageModelOutput; unwrap its logits."""
    return getattr(output, "logits", output)


def _sample(logits: mx.array, temperature: float) -> mx.array:
    if temperature and temperature > 0:
        return mx.random.categorical(logits / temperature)[None]
    return mx.argmax(logits, axis=-1)


def _quantize_in_place(model, weights, quant) -> None:
    """Reproduce mlx_lm's mixed-precision quantization for a loaded checkpoint."""
    if not quant or not quant.get("bits"):
        return

    def class_predicate(path, module):
        if path in quant:  # per-path override (e.g. 8-bit routers)
            return quant[path]
        if not hasattr(module, "to_quantized"):
            return False
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=quant.get("group_size", 64),
        bits=quant["bits"],
        mode=quant.get("mode", "affine"),
        class_predicate=class_predicate,
    )


def _read_weights(model_dir: str) -> Dict[str, mx.array]:
    weights: Dict[str, mx.array] = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    return weights


def _load_gemma4(model_dir: str, raw: Dict[str, Any]):
    from vendor.flash_vlm.models.gemma4.config import (
        AudioConfig,
        ModelConfig,
        TextConfig,
        VisionConfig,
    )
    from vendor.flash_vlm.models.gemma4.gemma4 import Model
    from vendor.flash_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor

    tc = TextConfig.from_dict(raw.get("text_config", raw))
    vc = VisionConfig.from_dict(raw.get("vision_config", {}))
    ac = AudioConfig.from_dict(raw["audio_config"]) if raw.get("audio_config") else None
    config = ModelConfig(
        text_config=tc,
        vision_config=vc,
        audio_config=ac,
        model_type=raw.get("model_type", "gemma4"),
        vocab_size=raw.get("vocab_size", 262144),
        eos_token_id=raw.get("eos_token_id"),
    )
    return Model(config), config, Gemma4ImageProcessor()


def _load_qwen3_5(model_dir: str, raw: Dict[str, Any]):
    from vendor.flash_vlm.models.qwen3_5.config import ModelConfig
    from vendor.flash_vlm.models.qwen3_5.qwen3_5 import Model
    from vendor.flash_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

    config = ModelConfig.from_dict(raw)
    pre = {}
    pre_path = Path(model_dir, "preprocessor_config.json")
    if pre_path.exists():
        pre = json.loads(pre_path.read_text())
    size = pre.get("size") or {}
    processor = Qwen3VLImageProcessor(
        patch_size=pre.get("patch_size", 16),
        temporal_patch_size=pre.get("temporal_patch_size", 2),
        merge_size=pre.get("merge_size", 2),
        min_pixels=size.get("shortest_edge", 56 * 56),
        max_pixels=size.get("longest_edge", 14 * 14 * 4 * 1280),
        image_mean=pre.get("image_mean"),
        image_std=pre.get("image_std"),
    )
    return Model(config), config, processor


_LOADERS = {"gemma4": _load_gemma4, "qwen3_5": _load_qwen3_5}


def load_vision_model(model_dir: str) -> VisionModel:
    """Load a multimodal checkpoint as a full VLM (text + vision tower)."""
    from mlx_lm.utils import load_tokenizer

    t0 = time.perf_counter()
    model_dir = str(model_dir)
    raw = json.loads(Path(model_dir, "config.json").read_text())
    model_type = raw.get("model_type")
    family = _FAMILY_BY_MODEL_TYPE.get(model_type)
    if family is None:
        raise ValueError(f"No vision loader for model_type {model_type!r}")

    model, config, image_processor = _LOADERS[family](model_dir, raw)
    weights = _read_weights(model_dir)
    _quantize_in_place(model, weights, raw.get("quantization"))
    model.load_weights(list(model.sanitize(weights).items()))
    mx.eval(model.parameters())

    tokenizer = load_tokenizer(model_dir, tokenizer_config_extra={"trust_remote_code": True})
    print(
        f"[vision] loaded {Path(model_dir).name} ({family}) in {time.perf_counter() - t0:.1f}s "
        f"(vision={model.vision_tower is not None})"
    )
    return VisionModel(model, tokenizer, image_processor, config, family)
