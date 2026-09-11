"""Download pinned MLX checkpoints into models/. No model execution."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import hf_env  # noqa: E402  (must precede any huggingface_hub import)

hf_env.configure(root=ROOT)
REPOS = {
    'qwen3.8-flash': 'mlx-community/Qwen3.8-Flash-Next-4bit',
    'deepseek-v4-flash': 'mlx-community/DeepSeek-V4-Flash-4bit',
    'gemma4-26b-a4b': 'mlx-community/gemma-4-26b-a4b-it-4bit',
    'gemma4-31b': 'mlx-community/gemma-4-31b-it-4bit',
    'diffusiongemma-26b-a4b': 'mlx-community/diffusiongemma-26B-A4B-it-4bit',
    'qwen3.8-27b': 'mlx-community/Qwen3.8-27B-4bit',
    'qwen3.6-35b-a3b': 'mlx-community/Qwen3.6-35B-A3B-4bit',
    'ornith-1.5-35b-a3b': 'ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit',
    'nemotron-3.5-30b-a3b': 'mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit',
    'gpt-oss-20b': 'mlx-community/gpt-oss-20b-MXFP4-Q8',
    'muse-glimmer-30b': 'mlx-community/Muse-Glimmer-30B-4bit',
    'glm-4.7-flash': 'mlx-community/GLM-4.7-Flash-4bit',
    'minicpm5-2b': 'mlx-community/MiniCPM5-2B-mlx-4Bit',
    'qwen3.5-9b': 'mlx-community/Qwen3.5-9B-MLX-4bit',
    'qwen3.5-4b': 'mlx-community/Qwen3.5-4B-MLX-4bit',
    'llama-3.2-3b': 'mlx-community/Llama-3.2-3B-Instruct-4bit',
    'smollm3-3b': 'mlx-community/SmolLM3-3B-4bit',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='+', choices=list(REPOS))
    parser.add_argument('--metadata-only', action='store_true')
    hf_env.add_arguments(parser)
    args = parser.parse_args()
    hf_env.apply_from_args(args, root=ROOT)
    from huggingface_hub import HfApi, snapshot_download
    hub = hf_env.hub_kwargs()
    for alias in args.models:
        repo = REPOS[alias]
        lock_path = ROOT / 'model-configs' / (alias + '.lock.json')
        if lock_path.exists():
            lock = json.loads(lock_path.read_text())
        else:
            info = HfApi(**{k: v for k, v in hub.items() if k == 'token'}).model_info(
                repo, **{k: v for k, v in hub.items() if k == 'endpoint'})
            lock = {'repo_id': repo, 'revision': info.sha}
            lock_path.write_text(json.dumps(lock, indent=2) + '\n')
        patterns = ['*.json', '*.jinja', '*.model', '*.txt']
        if not args.metadata_only:
            patterns += ['*.safetensors']
        print(f'Downloading {alias}: {lock["revision"]}', flush=True)
        snapshot_download(repo_id=repo, revision=lock['revision'],
                          local_dir=ROOT / 'models' / alias,
                          allow_patterns=patterns, max_workers=2, **hub)
        print(f'Ready: models/{alias}', flush=True)


if __name__ == '__main__':
    main()
