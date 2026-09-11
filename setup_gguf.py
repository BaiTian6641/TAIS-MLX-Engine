"""Download pinned IQ/K-quant GGUF weights into gguf/. No conversion, no model execution."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import hf_env  # noqa: E402  (must precede any huggingface_hub import)

hf_env.configure(root=ROOT)
REPOS = {
    'qwen3.8-flash': 'unsloth/Qwen3.8-Flash-Next-GGUF',
    'deepseek-v4-flash': 'unsloth/DeepSeek-V4-Flash-GGUF',
}
QUANTS = ('UD-IQ1_S', 'UD-IQ1_M', 'UD-Q2_K_XL')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='+', choices=list(REPOS))
    parser.add_argument('--quant', default='UD-IQ1_S', choices=QUANTS)
    hf_env.add_arguments(parser)
    args = parser.parse_args()
    hf_env.apply_from_args(args, root=ROOT)
    from huggingface_hub import HfApi, snapshot_download
    hub = hf_env.hub_kwargs()
    for alias in args.models:
        repo = REPOS[alias]
        lock_path = ROOT / 'model-configs' / (alias + '.gguf.lock.json')
        if lock_path.exists():
            lock = json.loads(lock_path.read_text())
            if lock.get('quant') != args.quant:
                raise SystemExit(f'{lock_path.name} pins {lock.get("quant")}; pass --quant {lock.get("quant")} '
                                 'or delete the lock to repin')
        else:
            info = HfApi().model_info(repo)
            lock = {'repo_id': repo, 'revision': info.sha, 'quant': args.quant}
            lock_path.write_text(json.dumps(lock, indent=2) + '\n')
        print(f'Downloading {alias} {lock["quant"]}: {lock["revision"]}', flush=True)
        snapshot_download(**hub, repo_id=repo, revision=lock['revision'],
                          local_dir=ROOT / 'gguf' / alias,
                          allow_patterns=[lock['quant'] + '/*.gguf'], max_workers=2)
        print(f'Ready: gguf/{alias}/{lock["quant"]}', flush=True)


if __name__ == '__main__':
    main()
