"""Vision-capable generation for the engine's multimodal profiles.

The text-only serving path strips each multimodal checkpoint's vision tower
(``mlx_lm.gemma4_text`` for Gemma 4) so it cannot consume images. This module
loads the *full* VLM from the vendored ``flash_vlm`` implementation and runs
image + text generation:

1. apply the chat template (image content parts become ``<|image|>`` markers),
2. preprocess images to ``pixel_values`` + a per-image soft-token count,
3. expand every ``<|image|>`` into ``{boi}{<|image|> x n}{eoi}``,
4. tokenize, prefill with ``pixel_values``, then decode autoregressively.

The loader reproduces ``mlx_lm``'s mixed-precision quantization: a per-path
override (the MoE routers are 8-bit here) wins, otherwise a module is quantized
iff the checkpoint carries ``<path>.scales`` for it - so the bf16 vision tower
is left alone automatically.
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


# Profiles whose checkpoints ship a vision tower we can serve through the
# vendored flash_vlm gemma4 implementation.
VISION_PROFILES = frozenset({"gemma4-26b-a4b", "gemma4-31b"})


class VisionModel:
    """A loaded VLM plus its tokenizer and image preprocessor."""

    def __init__(self, model, tokenizer, image_processor, config):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config

    # -- input construction ------------------------------------------------
    def build_inputs(
        self, messages: List[Dict[str, Any]], images: Sequence[Any]
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Return ``(input_ids, pixel_values)`` for a chat conversation."""
        tok = self.tokenizer
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pixel_values = None
        if images:
            data, num_soft_tokens = self.image_processor(list(images))
            pixel_values = mx.array(data["pixel_values"])
            # Expand each <|image|> placeholder to that image's soft-token count,
            # wrapped in begin/end-of-image markers.
            replacements = [
                f"{tok.boi_token}{tok.image_token * n}{tok.eoi_token}"
                for n in num_soft_tokens
            ]
            it = iter(replacements)
            pattern = re.escape(tok.image_token)
            text = re.sub(pattern, lambda _: next(it), text)
        input_ids = mx.array(tok.encode(text, add_special_tokens=False))
        return input_ids, pixel_values

    # -- generation ---------------------------------------------------------
    def generate_tokens(
        self,
        messages: List[Dict[str, Any]],
        images: Sequence[Any] = (),
        max_tokens: int = 256,
        sampler=None,
        temperature: float = 0.0,
    ):
        """Yield generated token ids one at a time (stops at an eos token)."""
        from vendor.flash_vlm.models.cache import make_prompt_cache

        input_ids, pixel_values = self.build_inputs(messages, images)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        cache = make_prompt_cache(self.model.language_model)
        eos = set(self._eos_ids())

        def pick(logits):
            if sampler is not None:
                t = sampler(logits)
                return t if getattr(t, "ndim", 0) > 0 else t[None]
            return _sample(logits, temperature)

        logits = self.model(input_ids, pixel_values=pixel_values, cache=cache)
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
    ) -> str:
        out = list(self.generate_tokens(messages, images, max_tokens, temperature=temperature))
        return self.tokenizer.decode(out, skip_special_tokens=True)

    def _eos_ids(self) -> List[int]:
        tok = self.tokenizer
        ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
        extra = getattr(self.config, "eos_token_id", None)
        if isinstance(extra, (list, tuple)):
            ids.extend(int(e) for e in extra)
        elif extra is not None:
            ids.append(int(extra))
        return [i for i in ids if i is not None]


def _logits_of(output) -> mx.array:
    """The model returns a LanguageModelOutput; unwrap its logits."""
    return getattr(output, "logits", output)


def _sample(logits: mx.array, temperature: float) -> mx.array:
    if temperature and temperature > 0:
        return mx.random.categorical(logits / temperature)[None]
    return mx.argmax(logits, axis=-1)


def load_vision_model(model_dir: str) -> VisionModel:
    """Load a multimodal checkpoint as a full VLM (text + vision tower)."""
    from mlx_lm.utils import load_tokenizer

    from vendor.flash_vlm.models.gemma4.config import (
        AudioConfig,
        ModelConfig,
        TextConfig,
        VisionConfig,
    )
    from vendor.flash_vlm.models.gemma4.gemma4 import Model
    from vendor.flash_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor

    t0 = time.perf_counter()
    model_dir = str(model_dir)
    raw = json.loads(Path(model_dir, "config.json").read_text())
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
    model = Model(config)

    weights: Dict[str, mx.array] = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        weights.update(mx.load(str(shard)))

    quant = raw.get("quantization")
    if quant:
        def class_predicate(path, module):
            if path in quant:  # per-path override (e.g. 8-bit routers)
                return quant[path]
            if not hasattr(module, "to_quantized"):
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    model.load_weights(list(model.sanitize(weights).items()))
    mx.eval(model.parameters())

    tokenizer = load_tokenizer(model_dir, tokenizer_config_extra={"trust_remote_code": True})
    image_processor = Gemma4ImageProcessor()
    print(
        f"[vision] loaded {Path(model_dir).name} in {time.perf_counter() - t0:.1f}s "
        f"(vision={model.vision_tower is not None})"
    )
    return VisionModel(model, tokenizer, image_processor, config)
