"""Run a DiffusionGemma completion from the local MLX checkpoint and time it."""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from diffusion_engine import DiffusionConfig, generate

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=ROOT / 'models/diffusiongemma-26b-a4b')
    parser.add_argument('--prompt', default='What is the capital of France? Answer in one word.')
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='independent canvases refined in one decoder pass')
    parser.add_argument('--raw', action='store_true',
                        help='send the prompt without the chat template (produces degenerate text)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from vendor.flash_vlm.models.diffusion_gemma.config import ModelConfig
    from vendor.flash_vlm.models.diffusion_gemma.diffusion_gemma import Model

    config = json.loads((args.model_path / 'config.json').read_text())
    generation = json.loads((args.model_path / 'generation_config.json').read_text())
    model_config = ModelConfig.from_dict(config)
    model = Model(model_config)

    quant = config.get('quantization') or {}
    started = time.time()
    weights = {}
    for shard in sorted(args.model_path.glob('*.safetensors')):
        weights.update(mx.load(str(shard)))
    if quant:
        # The per-module map in the config lists overrides only, so derive each
        # module's real format from the packed weight and scale shapes. The
        # group size is fixed by the config; the bit width follows from it.
        plan = {}
        for name, scales in weights.items():
            if not name.endswith('.scales'):
                continue
            base = name[:-len('.scales')]
            packed = weights.get(base + '.weight')
            if packed is None or packed.dtype != mx.uint32:
                continue
            group = int(quant.get(base, quant).get('group_size', 64)) \
                if isinstance(quant.get(base, quant), dict) else int(quant.get('group_size', 64))
            logical = scales.shape[-1] * group
            if logical == 0 or packed.shape[-1] * 32 % logical:
                continue
            bits = packed.shape[-1] * 32 // logical
            if bits in (2, 3, 4, 5, 6, 8):
                plan[base] = {'group_size': group, 'bits': int(bits), 'mode': 'affine'}

        def predicate(path, module):
            spec = plan.get(path)
            return spec if spec and hasattr(module, 'to_quantized') else False

        nn.quantize(model, **{k: quant[k] for k in ('group_size', 'bits', 'mode')},
                    class_predicate=predicate)
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    load_seconds = time.time() - started

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    if args.raw:
        prompt_ids = tokenizer.encode(args.prompt)
    else:
        # The released model is instruction tuned: the chat template is what the
        # reference implementation feeds it, and a bare prompt yields degenerate
        # text even through the upstream sampler.
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': args.prompt}],
            tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(text)
    diffusion = DiffusionConfig.from_generation_config(generation)
    started = time.time()
    sequences, stats = generate(model, tokenizer, prompt_ids, diffusion,
                                max_new_tokens=args.max_tokens, seed=args.seed,
                                batch_size=args.batch_size)
    seconds = time.time() - started
    tokens = sequences[0]
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    total_tokens = sum(len(sequence) for sequence in sequences)
    report = {'model': 'diffusiongemma-26b-a4b', 'prompt': args.prompt,
              'prompt_tokens': len(prompt_ids), 'tokens': len(tokens),
              'batch_size': args.batch_size, 'batch_tokens': total_tokens,
              'batch_tokens_per_second': total_tokens / seconds if seconds else None,
              'generated_tokens': tokens[:32], 'text': text[:400],
              'texts': [tokenizer.decode(s, skip_special_tokens=True)[:80] for s in sequences],
              'load_seconds': load_seconds, 'seconds': seconds,
              'tokens_per_second': len(tokens) / seconds if seconds else None,
              'canvas_passes': stats['canvas_passes'], 'denoising_steps': stats['denoising_steps'],
              'mlx_active_bytes': mx.get_active_memory(), 'mlx_peak_bytes': mx.get_peak_memory()}
    print(json.dumps({k: v for k, v in report.items() if k != 'generated_tokens'}, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
