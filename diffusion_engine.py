"""Block-diffusion generation for DiffusionGemma.

The checkpoint is not autoregressive: a causal encoder consumes the prompt and
fills a hybrid cache, then a decoder refines a 256-token canvas with
bidirectional attention over the cache and the canvas itself. Each denoising
step re-randomises the positions the entropy rule rejected and feeds the
previous step's logits back as self-conditioning; the final step's argmax canvas
is committed and re-encoded to extend the cache.

The algorithm mirrors the reference implementations (transformers'
 ``generation_diffusion_gemma.py`` and MLX-VLM's ``generate/diffusion.py``) for
the entropy-bound sampler, which the model card requires for quality; MLX-VLM's
``confidence-threshold`` sampler is a heuristic substitute.
"""
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class DiffusionConfig:
    canvas_length: int = 256
    max_denoising_steps: int = 48
    entropy_bound: float = 0.1
    t_max: float = 0.8
    t_min: float = 0.4
    eos_token_ids: tuple = (1, 106, 50)

    @classmethod
    def from_generation_config(cls, generation_config):
        config = dict(generation_config or {})
        sampler = config.get('sampler_config') or {}
        eos = config.get('eos_token_id', [1, 106, 50])
        return cls(
            max_denoising_steps=int(config.get('max_denoising_steps', 48)),
            entropy_bound=float(sampler.get('entropy_bound', 0.1)),
            t_max=float(config.get('t_max', 0.8)),
            t_min=float(config.get('t_min', 0.4)),
            eos_token_ids=tuple(eos) if isinstance(eos, (list, tuple)) else (eos,),
        )


def _canvas_entropy(logits):
    """Token entropy of the vocabulary distribution at each canvas position."""
    logits = logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.sum(mx.exp(log_probs) * log_probs, axis=-1)


def _entropy_transfer_mask(entropy, bound):
    """Accept the lowest-entropy positions whose cumulative spread fits the bound."""
    order = mx.argsort(entropy, axis=-1)
    ordered = mx.take_along_axis(entropy, order, axis=-1)
    spread = mx.cumsum(ordered, axis=-1) - mx.cummax(ordered, axis=-1)
    selected = spread <= bound
    mask = mx.zeros_like(selected)
    return mx.put_along_axis(mask, order, selected, axis=-1)


def _temperature(step, steps, config):
    return config.t_min + (config.t_max - config.t_min) * (step / steps)


def generate(model, tokenizer, prompt_ids, config, max_new_tokens=256, seed=None, batch_size=1):
    """Generate with the block-diffusion loop; returns (sequences, stats).

    ``batch_size`` runs that many independent canvases through the same
    decoder pass. Diffusion refines a canvas in parallel, so the denoising
    steps are shared across the batch and the per-token cost falls with it;
    every sequence keeps its own canvas and stop state.
    """
    if seed is not None:
        mx.random.seed(seed)
    config_obj = getattr(model, 'config', None) or getattr(model, 'args', None)
    text_config = getattr(config_obj, 'text_config', None) or config_obj
    vocab_size = int(text_config.vocab_size)
    prompt = mx.array([list(prompt_ids)] * batch_size, dtype=mx.int32)
    # The encoder returns a rebuilt cache; keeping the old object would leave
    # the decoder with no prompt context at all.
    cache = model.diffusion_prefill_cache(prompt, cache=model.make_cache())

    sequences = [[] for _ in range(batch_size)]
    done = [False] * batch_size
    stats = {'canvas_passes': 0, 'denoising_steps': 0, 'batch_size': batch_size}
    while not all(done):
        active = mx.array([[not flag] for flag in done], dtype=mx.bool_)
        canvas = mx.random.randint(0, vocab_size, (batch_size, config.canvas_length)).astype(mx.int32)
        condition_context = model.diffusion_prepare_self_conditioning()
        self_conditioning = None
        masks = model.diffusion_decoder_masks(canvas, cache)
        for step in reversed(range(1, config.max_denoising_steps + 1)):
            stats['denoising_steps'] += 1
            logits = model.diffusion_decoder_logits(
                canvas, cache=cache, self_conditioning=self_conditioning,
                decoder_attention_mask=masks)
            logits = logits / _temperature(step, config.max_denoising_steps, config)
            if step == 1:
                break
            denoised = mx.random.categorical(logits.astype(mx.float32)).astype(mx.int32)
            accept = _entropy_transfer_mask(_canvas_entropy(logits), config.entropy_bound)
            kept = mx.where(accept, denoised, canvas)
            fresh = mx.random.randint(0, vocab_size, (batch_size, config.canvas_length)).astype(mx.int32)
            updated = mx.where(accept, kept, fresh)
            canvas = mx.where(active, updated, canvas)      # finished rows stay put
            self_conditioning = model.diffusion_self_conditioning(logits, condition_context)
            mx.eval(canvas, self_conditioning)
        committed = mx.argmax(logits, axis=-1).astype(mx.int32)
        mx.eval(committed)
        stats['canvas_passes'] += 1
        for index in range(batch_size):
            if done[index]:
                continue
            block = committed[index].tolist()
            stop = len(block)
            for position, token in enumerate(block):
                if token in config.eos_token_ids or len(sequences[index]) + position >= max_new_tokens:
                    stop = position
                    break
            sequences[index].extend(block[:stop])
            if stop < len(block):
                done[index] = True
        cache = model.diffusion_update_cache(committed, cache=cache)
    return sequences, stats
