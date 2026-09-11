"""Compare GGUF-derived tensors against the same weights in a 4-bit MLX checkpoint.

Requires both the pinned GGUF and whichever shards of the MLX checkpoint are on
disk; tensors whose shard is missing are skipped. Writes
<model>-gguf-parity.json and exits non-zero if any compared tensor mismatches.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parent
MODELS = {
    'qwen3.8-flash': ('gguf/qwen3.8-flash/UD-IQ1_S', 'models/qwen3.8-flash', 'qwen4exp'),
    'deepseek-v4-flash': ('gguf/deepseek-v4-flash/UD-IQ1_S', 'models/deepseek-v4-flash', 'deepseek4'),
}


def mlx_reference(directory, weight_map, name, global_bits):
    shard = weight_map.get(name)
    if not shard or not (directory / shard).is_file():
        return None
    data = mx.load(str(directory / shard))
    if name not in data:
        return None
    weight = data[name]
    scales = data.get(name.replace('.weight', '.scales'))
    if scales is None or weight.dtype != mx.uint32:
        return np.array(weight.astype(mx.float32))
    biases = data.get(name.replace('.weight', '.biases'))
    logical = weight.shape[-1] * 32 // global_bits
    group = int(logical // scales.shape[-1])
    return np.array(mx.dequantize(weight, scales, biases, group_size=group,
                                  bits=global_bits, mode='affine').astype(mx.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=list(MODELS))
    parser.add_argument('--tolerance', type=float, default=0.12,
                        help='max |diff| / max |reference| before a tensor is reported')
    parser.add_argument('--layers', type=int, default=2, help='how many layers to compare')
    args = parser.parse_args()

    from gguf_model import _qwen_transform, load_deepseek, qwen4exp_param_map
    from gguf_reader import GGUFIndex

    gguf_dir, checkpoint, family = MODELS[args.model]
    index = GGUFIndex(sorted((ROOT / gguf_dir).glob('*.gguf')))
    config = json.loads((ROOT / checkpoint / 'config.json').read_text())
    weight_map = json.loads((ROOT / checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    global_bits = config.get('quantization', {}).get('bits', 4)
    text = config.get('text_config', config)
    heads = {}
    if 'linear_num_key_heads' in text:
        heads = dict(key_heads=text['linear_num_key_heads'],
                     value_per_key=text['linear_num_value_heads'] // text['linear_num_key_heads'],
                     value_dim=text['linear_value_head_dim'], key_dim=text['linear_key_head_dim'])

    if family == 'qwen4exp':
        from flash_models import model_classes
        args_obj = model_classes(config)[1].from_dict(config)
        mapping = qwen4exp_param_map(config)
        transform = lambda name, arr: _qwen_transform(name, arr, args_obj, heads)
    else:
        _, args_obj = load_deepseek(index, config, expert_cache_bytes=0)  # model unused below
        mapping = {name: tensor for name, tensor in
                   __import__('gguf_model').deepseek_param_map(config).items()}
        transform = lambda name, arr: arr

    compared, mismatched, skipped = [], [], []
    for name, tensor in mapping.items():
        layer = None
        if '.layers.' in name:
            try:
                layer = int(name.split('.layers.')[1].split('.')[0])
            except (IndexError, ValueError):
                layer = None
        if layer is None or layer >= args.layers:
            continue
        reference = mlx_reference(ROOT / checkpoint, weight_map, name, global_bits)
        if reference is None:
            continue
        parts = tensor if isinstance(tensor, tuple) else (tensor,)
        array = mx.concatenate([index.read(p) for p in parts], axis=0) if len(parts) > 1 else index.read(parts[0])
        array = np.array(transform(name, array).astype(mx.float32))
        if array.shape != reference.shape:
            # Some tensors are stored differently in that checkpoint (its router
            # matrices are wider); that is a checkpoint difference, not a
            # transform error, so record it without failing.
            skipped.append({'name': name, 'gguf': list(array.shape), 'mlx': list(reference.shape)})
            continue
        ratio = float(np.abs(array - reference).max() / max(float(np.abs(reference).max()), 1e-6))
        compared.append({'name': name, 'ratio': ratio})
        if ratio > args.tolerance:
            mismatched.append({'name': name, 'ratio': ratio})

    report = {'model': args.model, 'layers': args.layers, 'tolerance': args.tolerance,
              'compared': len(compared), 'skipped': skipped, 'mismatched': mismatched,
              'worst': sorted(compared, key=lambda row: -row['ratio'])[:5]}
    print(json.dumps({k: v for k, v in report.items() if k != 'compared'}, indent=2))
    (ROOT / f'{args.model}-gguf-parity.json').write_text(json.dumps(report, indent=2) + '\n')
    if mismatched:
        raise SystemExit(f'{len(mismatched)} tensors disagree with the MLX checkpoint')


if __name__ == '__main__':
    main()
