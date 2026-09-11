"""MTP speculative decoding for the HTTP server.

The MTP head drafts one token per round and the target verifies two positions,
which is the shape that actually pays on these targets (see
docs/decoding-and-memory.md). This module loads the head and drives the vendored
round loop; ``serve.py`` swaps it in for the ordinary single-request generator
when ``--mtp-draft`` names a drafter directory.
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class MtpTarget(nn.Module):
    """The vendored language model, shaped for both the server and speculation.

    The plain server path calls ``model(inputs, cache=...)`` and indexes the
    result, so ``__call__`` returns logits. The speculative prefill needs the
    hidden state and shared KV that only this implementation captures, so that
    path goes through ``capture_forward`` instead.
    """

    def __init__(self, config):
        super().__init__()
        from vendor.flash_vlm.models.qwen3_5_moe.language import LanguageModel

        self.config = config
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        return {
            key: value
            for key, value in weights.items()
            if not key.startswith(('vision_tower', 'model.visual'))
        }

    def __call__(self, inputs, cache=None, **kwargs):
        return self.language_model(inputs, cache=cache, **kwargs).logits

    def capture_forward(self, inputs, cache):
        return self.language_model(inputs, cache=cache, return_hidden=True,
                                   return_shared_kv=True)

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def quantization(self):
        return getattr(self.config, 'quantization', None)


def load_mtp_target(path):
    """Load a qwen3_5_moe checkpoint through the speculative-capable classes."""
    from mlx_lm.utils import load_model as serving_load

    class ModelArgs:
        @staticmethod
        def from_dict(params):
            import dataclasses
            from vendor.flash_vlm.models.qwen3_5_moe.config import ModelConfig, VisionConfig

            raw = dict(params)
            if 'vision_config' not in raw:
                # Text-only conversions drop the vision tower; the vendored
                # config class wants the section present, and its defaults
                # describe a tower nobody calls.
                raw['vision_config'] = dataclasses.asdict(VisionConfig())
            return ModelConfig.from_dict(raw)

    model, _ = serving_load(
        Path(path), get_model_classes=lambda config: (MtpTarget, ModelArgs)
    )
    return model, model.config


def load_drafter(path, target):
    """Load an extracted MTP head and bind it to the target."""
    from vendor.flash_vlm.speculative.drafters.qwen3_5_mtp import (
        ModelConfig as MtpConfig, Qwen3_5MTPDraftModel,
    )

    config = MtpConfig.from_dict(json.loads((Path(path) / 'config.json').read_text()))
    drafter = Qwen3_5MTPDraftModel(config)
    weights = {}
    for shard in sorted(Path(path).glob('*.safetensors')):
        weights.update(mx.load(str(shard)))
    drafter.load_weights(list(weights.items()))
    mx.eval(drafter.parameters())
    return drafter


class SpeculativeStream:
    """A ``stream_generate``-shaped generator backed by the MTP round loop.

    Yields ``(token, logprobs)``; the caller turns those into API responses. The
    bonus token's logprobs come from the prefill, the accepted drafts carry
    ``None`` because the round loop commits them internally.
    """

    def __init__(self, model, drafter, prompt_cache, prompt_ids, max_tokens, sampler,
                 greedy=True, block_size=None, eos_token_ids=None):
        self.model = model
        self.drafter = drafter
        self.prompt_cache = prompt_cache
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.sampler = sampler
        self.greedy = greedy
        self.block_size = block_size
        self.eos_token_ids = eos_token_ids or set()
        self.generated = []

    def __iter__(self):
        from vendor.flash_vlm.speculative import mtp as mtp_mod
        from vendor.flash_vlm.speculative.utils import run_speculative_rounds

        # One round is one target forward; acceptance is tokens per round.
        counter = {'rounds': 0}
        original_verify = mtp_mod._mtp_verify_target

        def counting_verify(*call_args, **kwargs):
            counter['rounds'] += 1
            return original_verify(*call_args, **kwargs)

        mtp_mod._mtp_verify_target = counting_verify
        out = self.model.capture_forward(self.prompt_ids, self.prompt_cache)
        logits = out.logits[:, -1, :]
        logprobs = nn.log_softmax(logits, -1)
        first = mx.argmax(logits, -1) if self.greedy else mx.array(
            [self.sampler(logits)[0]])[:, None].reshape(-1)
        mx.eval(first, logprobs)
        emitted = 0
        try:
            for token, _ in run_speculative_rounds(
                self.model, self.drafter, self.prompt_cache, self.prompt_ids,
                first, logprobs, out, draft_kind='mtp', max_tokens=self.max_tokens,
                sampler=self.sampler, draft_block_size=self.block_size,
                sampler_is_greedy=self.greedy,
            ):
                value = int(token) if not hasattr(token, 'item') else int(token.item())
                self.generated.append(value)
                emitted += 1
                yield value, None
                if value in self.eos_token_ids or emitted >= self.max_tokens:
                    return
        finally:
            mtp_mod._mtp_verify_target = original_verify
            self.rounds = counter['rounds']


def stream_speculative(model, drafter, prompt_cache, prompt_ids, max_tokens, sampler,
                       greedy=True, block_size=None, eos_token_ids=None):
    return SpeculativeStream(model, drafter, prompt_cache, prompt_ids, max_tokens,
                             sampler, greedy, block_size, eos_token_ids)
