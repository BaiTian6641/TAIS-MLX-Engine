"""Load a pinned GGUF profile through the MLX runtime and generate greedy tokens."""
import argparse
import json
from pathlib import Path
import time

import mlx.core as mx

ROOT = Path(__file__).resolve().parent
GGUF = {
    'deepseek-v4-flash': ('gguf/deepseek-v4-flash/UD-IQ1_S', 'models/deepseek-v4-flash'),
    'qwen3.8-flash': ('gguf/qwen3.8-flash/UD-IQ1_S', 'models/qwen3.8-flash'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=list(GGUF))
    parser.add_argument('--expert-cache-gib', type=float, default=0.0)
    parser.add_argument('--saliency-weight', type=float, default=1.0,
                        help='Router-mass weight for expert residency (0 disables)')
    parser.add_argument('--prompt', default='The capital of France is')
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    from gguf_reader import GGUFIndex
    from gguf_model import load_deepseek
    from tokenizers import Tokenizer

    gguf_dir, checkpoint = (ROOT / p for p in GGUF[args.model])
    index = GGUFIndex(sorted(gguf_dir.glob('*.gguf')))
    config = json.loads((checkpoint / 'config.json').read_text())
    tokenizer = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))

    started = time.time()
    model, store = load_deepseek(index, config, expert_cache_bytes=int(args.expert_cache_gib * 2**30),
                                 saliency_weight=args.saliency_weight)
    load_seconds = time.time() - started
    loaded_bytes = mx.get_active_memory()

    ids = tokenizer.encode(args.prompt).ids
    cache = model.make_cache()
    started = time.time()
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits)
    prefill_seconds = time.time() - started

    generated = []
    started = time.time()
    for _ in range(args.max_tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        generated.append(token)
        logits = model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
    decode_seconds = time.time() - started

    text = tokenizer.decode(generated, skip_special_tokens=False)
    report = {
        'model': args.model,
        'prompt': args.prompt,
        'prompt_tokens': len(ids),
        'generated_tokens': generated,
        'text': text,
        'expert_cache_gib': args.expert_cache_gib,
        'saliency_weight': args.saliency_weight,
        'load_seconds': load_seconds,
        'prefill_seconds': prefill_seconds,
        'decode_seconds': decode_seconds,
        'decode_tps': args.max_tokens / decode_seconds if decode_seconds else None,
        'mlx_active_bytes': mx.get_active_memory(),
        'mlx_peak_bytes': mx.get_peak_memory(),
        'expert_stats': store.stats(),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
