"""Prefill and decode speed against prompt length, for one resident profile.

Writes <model>-context-scaling.json. A long prompt is built by repeating filler
text, so the numbers describe the runtime's scaling rather than any particular
document.
"""
import argparse
import json
from pathlib import Path
import time

import mlx.core as mx

from model_profiles import parse_options, resolve_profile, uses_quantized_kv

ROOT = Path(__file__).resolve().parent
FILLER = ('The history of the coastal city is long and varied. Merchants traded '
          'spices, cloth and ideas along its docks for centuries. ')


def build_prompt(tokenizer, tokens):
    seeds = tokenizer.encode(FILLER)
    ids = []
    while len(ids) < tokens:
        ids.extend(seeds)
    return ids[:tokens]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--contexts', type=int, nargs='+', default=[2048, 8192, 32768])
    parser.add_argument('--decode-tokens', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    from mlx_lm import utils as mlx_utils
    from mlx_lm.generate import stream_generate
    from mlx_lm.models import cache as mlx_cache

    profile = resolve_profile(parse_options(['--model', args.model]))
    model, tokenizer = mlx_utils.load(str(profile['path']), lazy=False, trust_remote_code=True)
    quantized_kv = uses_quantized_kv(profile['config'])
    rows = []
    for context in args.contexts:
        ids = build_prompt(tokenizer, context)
        cache = mlx_cache.make_prompt_cache(model)
        started = time.perf_counter()
        first = None
        tokens = 0
        for _ in stream_generate(model, tokenizer, prompt=ids, max_tokens=args.decode_tokens,
                                 prompt_cache=cache, prefill_step_size=128,
                                 kv_bits=4 if quantized_kv else None, kv_group_size=64,
                                 quantized_kv_start=0):
            if first is None:
                first = time.perf_counter()
            tokens += 1
        decode_seconds = time.perf_counter() - first
        row = {'context': context, 'prefill_seconds': first - started,
               'prefill_tokens_per_second': context / (first - started),
               'decode_ms_per_token': 1000 * decode_seconds / max(tokens - 1, 1),
               'decode_tokens_per_second': (tokens - 1) / decode_seconds if decode_seconds else None,
               'mlx_active_bytes': mx.get_active_memory(),
               'mlx_peak_bytes': mx.get_peak_memory()}
        rows.append(row)
        print(f"ctx {context:7d}: prefill {row['prefill_tokens_per_second']:7.1f} tok/s | "
              f"decode {row['decode_tokens_per_second']:6.2f} tok/s | "
              f"active {row['mlx_active_bytes']/2**30:5.1f} GiB", flush=True)
        del cache
        mx.clear_cache()

    report = {'model': args.model, 'quantized_kv': quantized_kv, 'rows': rows}
    destination = args.output or ROOT / f'{args.model}-context-scaling.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f'wrote {destination.name}')


if __name__ == '__main__':
    main()
