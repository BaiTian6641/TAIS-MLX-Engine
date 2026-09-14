"""Local model selection and compatibility metadata without importing MLX."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES = {
    'qwen3.8-flash': {'path': 'models/qwen3.8-flash', 'model_type': 'qwen4_exp',
                      'chat_template_args': {'enable_thinking': True}, 'flash': True},
    'deepseek-v4-flash': {'path': 'models/deepseek-v4-flash', 'model_type': 'deepseek_v4',
                         'chat_template_args': {'thinking_mode': 'thinking'}, 'flash': True},
    'k2-horizon': {'path': 'model', 'model_type': 'k2_horizon_mova',
                   'chat_template_args': {'reasoning_effort': 'high'}},
    'gemma4-26b-a4b': {'path': 'models/gemma4-26b-a4b', 'model_type': 'gemma4',
                       'chat_template_args': {}},
    'gemma4-31b': {'path': 'models/gemma4-31b', 'model_type': 'gemma4',
                   'chat_template_args': {}},
    'qwen3.6-35b-a3b': {'path': 'models/qwen3.6-35b-a3b', 'model_type': 'qwen3_5_moe',
                        'chat_template_args': {'enable_thinking': True},
                        'drafter': 'models/qwen3.6-mtp'},
    'ornith-1.5-35b-a3b': {'path': 'models/ornith-1.5-35b-a3b', 'model_type': 'qwen3_5_moe',
                           'chat_template_args': {'enable_thinking': True}},
    'qwen3.8-27b': {'path': 'models/qwen3.8-27b', 'model_type': 'qwen3_5',
                    'chat_template_args': {'enable_thinking': True},
                    'sampling': {'temp': 0.6, 'top_p': 0.95, 'top_k': 20}},
    'nemotron-3.5-30b-a3b': {'path': 'models/nemotron-3.5-30b-a3b', 'model_type': 'nemotron_h',
                             'chat_template_args': {}},
    'gpt-oss-20b': {'path': 'models/gpt-oss-20b', 'model_type': 'gpt_oss',
                    'chat_template_args': {'reasoning_effort': 'low'}},
    'muse-glimmer-30b': {'path': 'models/muse-glimmer-30b', 'model_type': 'muse_glimmer',
                         # The template addresses messages and expects a reasoning
                         # strength; without it the model echoes raw template markers.
                         'chat_template_args': {'reasoning_strength': 'low'}},
    'glm-4.7-flash': {'path': 'models/glm-4.7-flash', 'model_type': 'glm4_moe_lite',
                      # Thinking stays on: with it disabled this model answers
                      # "5" to 2+2. The reply is still split into content and
                      # reasoning, so callers see the answer field cleanly.
                      'chat_template_args': {'enable_thinking': True}},
    'minicpm5-2b': {'path': 'models/minicpm5-2b', 'model_type': 'llama',
                    'chat_template_args': {}},
    'spark-x2.5-4b': {'path': 'models/spark-x2.5-4b', 'model_type': 'spark2_5',
                      'chat_template_args': {}},
    'qwen3.5-9b': {'path': 'models/qwen3.5-9b', 'model_type': 'qwen3_5',
                   'chat_template_args': {'enable_thinking': True}},
    'qwen3.5-4b': {'path': 'models/qwen3.5-4b', 'model_type': 'qwen3_5',
                   'chat_template_args': {'enable_thinking': True}},
    'llama-3.2-3b': {'path': 'models/llama-3.2-3b', 'model_type': 'llama', 'chat_template_args': {}},
    'smollm3-3b': {'path': 'models/smollm3-3b', 'model_type': 'smollm3', 'chat_template_args': {}},
}


# Model families whose caches the pinned runtime cannot quantize: DeepSeek's
# compressed pools, sliding-window caches (`RotatingKVCache Quantization NYI`,
# which every model mixing sliding and full attention builds for its sliding
# layers - Gemma 4, GPT-OSS, Muse Glimmer and Spark2.5 among them), hybrids whose
# only caches hold recurrent state, where the flag would be a silent no-op, and
# GLM-4.7-Flash, whose MLA attention unpacks `update_and_fetch` into two names
# where a quantized cache returns four, so the flag crashes it mid-request rather
# than degrading it. serve.py re-checks these claims against the loaded model's
# own caches, so a profile that lands here by mistake is reported rather than
# quietly served wrong.
UNQUANTIZED_KV = frozenset({'deepseek_v4', 'gemma4', 'nemotron_h', 'gpt_oss',
                            'muse_glimmer', 'spark2_5', 'glm4_moe_lite'})


def uses_quantized_kv(config):
    return config.get('model_type') not in UNQUANTIZED_KV


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description='Local K2/Qwen MLX server')
    parser.add_argument('--model', choices=PROFILES, default='k2-horizon')
    parser.add_argument('--model-path', type=Path, help='Override the selected profile checkpoint directory')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--expert-cache-gib', type=float, default=None,
                        help='Enable disk expert streaming with this hot-cache budget (0 = no hot retention)')
    parser.add_argument('--prefill-step-size', type=int, default=512)
    parser.add_argument('--prompt-cache-size', type=int, default=1,
                        help='prompt-prefix KV caches held in memory; raise it when callers share prefixes')
    parser.add_argument('--kv-bits', type=int, choices=(0, 4, 8), default=4,
                        help='KV cache width for profiles whose caches can be quantized; '
                             '0 keeps KV in bf16, which also allows batching and prompt-prefix reuse, '
                             '8 is a middle ground that survives long context better than 4')
    parser.add_argument('--quantized-kv-start', type=int, default=0,
                        help='token position where KV quantization starts; a nonzero value keeps the '
                             'prompt in bf16 and compresses only the growing tail')
    parser.add_argument('--thinking-budget', type=int, default=None, metavar='TOKENS',
                        help='trim reasoning in earlier assistant turns to at most this many '
                             'tokens each; 0 drops all earlier reasoning; omit to keep everything')
    parser.add_argument('--no-mtp', action='store_true',
                        help='disable multi-token prediction even when the profile ships a '
                             'drafter for it (it is lossless and faster, so it is on by default)')
    parser.add_argument('--mtp-draft', type=Path, default=None,
                        help='served MTP head (drafts one token per round); requires a qwen3_5_moe profile')
    parser.add_argument('--decode-concurrency', type=int, default=1,
                        help='batch size for decoding; requires unquantized KV '
                             '(the pinned runtime cannot merge quantized caches)')
    parser.add_argument('--prompt-concurrency', type=int, default=1,
                        help='batch size for prefill')
    parser.add_argument('--vision-max-images', type=int, default=4, metavar='N',
                        help='keep only the N most recent images of a request; older ones '
                             'become a note in the prompt. Long agentic sessions resend '
                             'archived screenshots, which otherwise dominate the prefill')
    parser.add_argument('--vision-empty-stop-retries', type=int, default=1, choices=(0, 1, 2),
                        help='retry a reply whose final answer is empty (the model sometimes '
                             'closes the turn right after </think>); the retry decodes greedily')
    args = parser.parse_args(argv)
    if args.expert_cache_gib is not None and not 0 <= args.expert_cache_gib < float('inf'):
        parser.error('--expert-cache-gib must be finite and nonnegative')
    if args.prefill_step_size < 1:
        parser.error('--prefill-step-size must be positive')
    if args.decode_concurrency < 1 or args.prompt_concurrency < 1:
        parser.error('concurrency must be at least 1')
    return args


def resolve_profile(options):
    profile = dict(PROFILES[options.model])
    path = (options.model_path or ROOT / profile['path']).expanduser().resolve()
    if not (path / 'config.json').is_file():
        raise ValueError(f'Checkpoint not found at {path}. Run .venv/bin/python setup_models.py {options.model}, '
                         'or supply --model-path.')
    config = json.loads((path / 'config.json').read_text())
    index_path = path / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        missing = [name for name in set(index['weight_map'].values()) if not (path / name).is_file()]
        if missing:
            raise ValueError(f'Checkpoint download incomplete: {len(missing)} missing weight shards at {path}. '
                             f'Run .venv/bin/python setup_models.py {options.model}.')
    if config['model_type'] != profile['model_type']:
        raise ValueError(f'{options.model} requires {profile["model_type"]}; found {config["model_type"]}')
    text = config.get('text_config', config)
    if options.expert_cache_gib is not None and text.get('num_experts', text.get('n_routed_experts', 0)) <= 0:
        raise ValueError(f'{options.model} is dense; it has no routed experts to offload')
    if profile.get('flash') and options.expert_cache_gib is None:
        options.expert_cache_gib = 24.0
    profile.update(path=path, config=config, alias=options.model)
    return profile


def fingerprint(path):
    identity = hashlib.sha256(b'local-runtime-cache-v3-flash')
    for name in ('flash_models.py', 'flash_cache.py', 'disk_cache.py'):
        identity.update((ROOT / name).read_bytes())
    for pattern in ('*.json', '*.jinja', '*.model', '*.py'):
        for p in sorted(path.glob(pattern)):
            identity.update(p.name.encode())
            identity.update(p.read_bytes())
    identity.update((ROOT / 'requirements.lock').read_bytes())
    for p in sorted(path.glob('*.safetensors')):
        identity.update(f'{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}'.encode())
    return identity.hexdigest()[:24]
