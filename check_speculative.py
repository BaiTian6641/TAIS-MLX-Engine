"""Run the vendored speculative-decoding stack against the local Qwen3.6 target.

Two modes:

``--mode parity``    load the target through the vendored (speculation-capable)
                    language model and check its logits against the serving
                    path, so a mismatch can never be blamed on the round loop.
``--mode speculate`` load a drafter (DFlash or DSpark) and run the dflash round
                    loop, reporting acceptance and decode rate against a plain
                    baseline.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn

from vendor.flash_vlm.models.qwen3_5_moe.config import ModelConfig as VendoredConfig
from vendor.flash_vlm.models.qwen3_5_moe.language import LanguageModel as VendoredTarget

ROOT = Path(__file__).resolve().parent


def read_config(path):
    return json.loads((Path(path) / 'config.json').read_text())


def load_weights(path):
    weights = {}
    for shard in sorted(Path(path).glob('*.safetensors')):
        weights.update(mx.load(str(shard)))
    return weights


class VendoredMoETarget(nn.Module):
    """Text tower wrapped so the checkpoint's naming stays intact.

    The 4-bit checkpoint names every tensor ``language_model.*`` and its
    per-module quantization map - the router gates are 8-bit while the rest is
    4-bit - is keyed the same way. Loading the bare language model would make
    every map entry miss, so the text tower lives under ``language_model`` here,
    exactly where the checkpoint and the speculative round loop both expect it.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_model = VendoredTarget(config.text_config, config)

    def sanitize(self, weights):
        return {
            key: value
            for key, value in weights.items()
            if not key.startswith(('vision_tower', 'model.visual'))
        }

    def __call__(self, inputs, cache=None, **kwargs):
        return self.language_model(inputs, cache=cache, **kwargs)

    def make_cache(self):
        return self.language_model.make_cache()


def load_vendored_target(path):
    from mlx_lm.utils import load_model as serving_load
    raw = read_config(path)
    config_path = Path(path) / 'config.json'
    original = config_path.read_text()
    if 'vision_config' not in raw:
        # Text-only conversions (Ornith's MLX build, for one) drop the vision
        # tower, but the vendored config class requires the section to exist.
        # Its defaults describe a tower we never call. The loader reads the file
        # itself, so the section goes in for the duration of the load and the
        # checkpoint is restored afterwards.
        import dataclasses
        from vendor.flash_vlm.models.qwen3_5_moe.config import VisionConfig
        raw['vision_config'] = dataclasses.asdict(VisionConfig())
        config_path.write_text(json.dumps(raw, indent=2) + '\n')
    try:
        model, _ = serving_load(
            Path(path), get_model_classes=lambda config: (VendoredMoETarget, VendoredConfig)
        )
    finally:
        config_path.write_text(original)
    return model, VendoredConfig.from_dict(raw)


def load_drafter(path, target_model=None):
    """Load a DFlash or DSpark drafter.

    Publishers ship these configs in two shapes. The vendored drafter wants a
    nested ``dflash_config``; checkpoints converted for other runtimes carry the
    same keys flat, sometimes with a 32,000-token draft vocabulary and
    ``d2t``/``t2d`` mapping tensors. The flat shape is normalised here.
    """
    config_dict = json.loads((Path(path) / 'config.json').read_text())
    is_mtp = config_dict.get('model_type') == 'qwen3_5_mtp'
    is_dspark = 'DSpark' in str(config_dict.get('architectures')) or \
        config_dict.get('speculators_model_type') == 'dspark'
    if is_mtp:
        from vendor.flash_vlm.speculative.drafters.qwen3_5_mtp import ModelConfig
        from vendor.flash_vlm.speculative.drafters.qwen3_5_mtp import Qwen3_5MTPDraftModel as Drafter
    elif is_dspark:
        from vendor.flash_vlm.speculative.drafters.dspark import DSparkDraftModel as Drafter
        from vendor.flash_vlm.speculative.drafters.dspark import ModelConfig
        if not isinstance(config_dict.get('dflash_config'), dict):
            nested_keys = ('mask_token_id', 'target_layer_ids', 'num_target_layers',
                           'runtime_block_size', 'draft_window_size', 'block_size_policy',
                           'dflash_initial_block_size', 'markov_rank', 'markov_head_type',
                           'enable_confidence_head', 'confidence_head_with_markov')
            config_dict['dflash_config'] = {
                **{k: config_dict[k] for k in nested_keys if k in config_dict},
                'projector_type': 'dspark',
            }
        config_dict.setdefault('markov_head_type', 'vanilla')
        config_dict.setdefault('enable_confidence_head', True)
    else:
        from vendor.flash_vlm.speculative.drafters.qwen3_dflash import DFlashDraftModel as Drafter
        from vendor.flash_vlm.speculative.drafters.qwen3_dflash import ModelConfig

    config = ModelConfig.from_dict(config_dict)
    model = Drafter(config)
    if target_model is not None and hasattr(model, 'bind'):
        model.bind(target_model)
    weights = {}
    for shard in sorted(Path(path).glob('*.safetensors')):
        weights.update(mx.load(str(shard)))
    # A draft-vocabulary checkpoint carries tensors the target already owns.
    shared = {'d2t', 't2d', 'embed_tokens.weight', 'lm_head.weight'}
    weights = {k: v for k, v in weights.items() if k not in shared}
    # ``bind`` points the drafter at the target's (quantized) embedding and head,
    # so those parameters are legitimately absent from a DSpark checkpoint.
    model.load_weights(list(weights.items()), strict=not is_dspark)
    mx.eval(model.parameters())
    kind = 'mtp' if is_mtp else 'dflash'
    return model, config, kind


def greedy_logits(model, inputs, cache, capture=None, prefill_kwargs=None):
    kwargs = dict(prefill_kwargs or {})
    if capture:
        kwargs['capture_layer_ids'] = capture
    out = model(inputs, cache=cache, **kwargs)
    logits = out.logits[:, -1, :]
    return out, logits


def baseline_decode(target, prompt_ids, tokens):
    """Plain greedy decode: one target forward per token."""
    cache = target.make_cache()
    out, logits = greedy_logits(target, prompt_ids, cache)
    mx.eval(logits)
    emitted = [int(mx.argmax(logits, -1)[0])]
    started = time.perf_counter()
    for _ in range(tokens - 1):
        out, logits = greedy_logits(target, mx.array([[emitted[-1]]]), cache)
        mx.eval(logits)
        emitted.append(int(mx.argmax(logits, -1)[0]))
    seconds = time.perf_counter() - started
    return emitted, seconds


def speculative_decode(target, draft_model, prompt_ids, tokens, capture_layer_ids, block_size=None, kind='dflash'):
    """Greedy speculative decode through the vendored round loop."""
    from vendor.flash_vlm.speculative import dflash as dflash_mod
    from vendor.flash_vlm.speculative.utils import run_speculative_rounds

    # Round count == number of target forward passes, which is the quantity
    # acceptance is measured against. Each loop has its own verify callable.
    rounds = {'count': 0}
    verify_name = '_mtp_verify_target' if kind == 'mtp' else '_dflash_verify'
    original_verify = getattr(dflash_mod, verify_name, None) or getattr(
        __import__('vendor.flash_vlm.speculative.mtp', fromlist=['x']), verify_name)

    def counting_verify(*call_args, **kwargs):
        rounds['count'] += 1
        return original_verify(*call_args, **kwargs)

    if kind == 'mtp':
        from vendor.flash_vlm.speculative import mtp as mtp_mod
        setattr(mtp_mod, verify_name, counting_verify)
        restore = lambda: setattr(mtp_mod, verify_name, original_verify)
    else:
        dflash_mod._dflash_verify = counting_verify
        restore = lambda: setattr(dflash_mod, verify_name, original_verify)
    cache = target.make_cache()
    # MTP reads the last hidden state and the shared KV captured during prefill;
    # the block drafters read hidden states of nominated target layers.
    prefill = ({'return_hidden': True, 'return_shared_kv': True} if kind == 'mtp'
               else {'capture_layer_ids': capture_layer_ids})
    out, logits = greedy_logits(target, prompt_ids, cache, capture=None, prefill_kwargs=prefill)
    logprobs = nn.log_softmax(logits, -1)
    first = mx.argmax(logits, -1)
    mx.eval(first, logprobs)
    emitted = []
    started = time.perf_counter()
    try:
        for token, _ in run_speculative_rounds(
            target, draft_model, cache, prompt_ids, first, logprobs, out,
            draft_kind=kind, max_tokens=tokens, draft_block_size=block_size,
            sampler=lambda x: mx.argmax(x, -1), sampler_is_greedy=True,
        ):
            emitted.append(token if isinstance(token, int) else int(token[0]))
    finally:
        restore()
    seconds = time.perf_counter() - started
    return emitted, seconds, rounds['count']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/qwen3.6-35b-a3b')
    parser.add_argument('--drafter')
    parser.add_argument('--mode', choices=('parity', 'speculate'), default='parity')
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--block-size', type=int, help='override the drafter block size')
    parser.add_argument('--prompt', default='int main(int argc, char **argv) {')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    tokenizer_path = Path(args.model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompt_ids = mx.array([tokenizer.encode(args.prompt)])
    report = {'model': args.model, 'mode': args.mode, 'prompt_tokens': prompt_ids.shape[1]}

    started = time.perf_counter()
    target, _ = load_vendored_target(args.model)
    mx.clear_cache()
    report['load_seconds'] = round(time.perf_counter() - started, 1)

    if args.mode == 'parity':
        cache = target.make_cache()
        out = target(prompt_ids, cache=cache)
        vendored_last = out.logits[:, -1, :]
        mx.eval(vendored_last)
        report['returns_hidden_states'] = out.hidden_states is not None
        del target, out, cache
        gc.collect()
        mx.clear_cache()

        from mlx_lm.utils import load_model as mlx_load
        serving, _ = mlx_load(tokenizer_path)
        serving_logits = serving(prompt_ids)
        serving_last = serving_logits[:, -1, :] if serving_logits.ndim == 3 else serving_logits
        mx.eval(serving_last)
        diff = mx.abs(vendored_last - serving_last)
        report['parity_max_abs_diff'] = float(diff.max())
        report['parity_top1_match'] = int(mx.argmax(vendored_last, -1)[0]) == int(mx.argmax(serving_last, -1)[0])
        top5 = lambda x: mx.argsort(-x, axis=-1)[0, :5].tolist()
        report['parity_top5'] = {'vendored': top5(vendored_last), 'serving': top5(serving_last)}
        report['parity_top5_match'] = report['parity_top5']['vendored'] == report['parity_top5']['serving']
        # The two implementations do not share a kernel path (rope/attention
        # differ), so bit-exactness is not the bar. What matters is that the
        # ranking - which is what verification consumes - agrees.
        report['parity_ok'] = bool(report['parity_top1_match'] and report['parity_top5_match'])
    else:
        draft_model, draft_config, draft_kind = load_drafter(args.drafter, target)
        if args.block_size:
            draft_config.block_size = args.block_size
            draft_config.runtime_block_size = args.block_size
        target_layer_ids = list(getattr(draft_config, 'target_layer_ids', []))
        report['drafter'] = args.drafter
        report['drafter_kind'] = draft_kind
        report['drafter_layers'] = getattr(draft_config, 'num_hidden_layers', None)
        report['draft_block_size'] = getattr(draft_config, 'block_size', None)
        report['target_layer_ids'] = target_layer_ids

        # Warm up both paths so the first (cold) forward is not charged to one.
        baseline_decode(target, prompt_ids, 8)
        speculative_decode(target, draft_model, prompt_ids, 8, target_layer_ids, kind=draft_kind)
        mx.clear_cache()

        baseline_tokens, baseline_seconds = baseline_decode(target, prompt_ids, args.tokens)
        spec_tokens, spec_seconds, target_forwards = speculative_decode(
            target, draft_model, prompt_ids, args.tokens, target_layer_ids, args.block_size, draft_kind
        )

        report['baseline_tokens_per_second'] = round(len(baseline_tokens) / baseline_seconds, 2)
        report['speculative_tokens_per_second'] = round(len(spec_tokens) / spec_seconds, 2)
        report['speedup'] = round((len(spec_tokens) / spec_seconds) / (len(baseline_tokens) / baseline_seconds), 3)
        report['target_forwards'] = target_forwards
        report['accepted_per_target_forward'] = round(len(spec_tokens) / max(target_forwards, 1), 2)
        report['baseline_text'] = tokenizer.decode(baseline_tokens)
        report['speculative_text'] = tokenizer.decode(spec_tokens)
        # Greedy speculation must be lossless: same tokens, same order.
        report['matches_baseline'] = baseline_tokens == spec_tokens
        report['ok'] = bool(report['matches_baseline'] and report['speedup'] > 1.0)

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    if args.mode == 'parity' and not report.get('parity_ok'):
        raise SystemExit('vendored target does not match the serving path')
    if args.mode == 'speculate' and not report.get('ok'):
        raise SystemExit('speculative decoding did not reproduce the baseline greedily')


if __name__ == '__main__':
    main()
