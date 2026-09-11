"""Real checkpoint smoke test, with optional expert streaming; leaves no server."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT/'hf-cache'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--expert-cache-gib', type=float)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--prompt', default='What is 2 + 2? Answer with only the number.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx.utils import tree_flatten
    from expert_cache import install_expert_offload
    from model_profiles import parse_options, resolve_profile, uses_quantized_kv
    from runtime_support import ContextPolicy
    selected = parse_options(['--model',args.model])
    profile = resolve_profile(selected)
    start = time.monotonic()
    if profile.get('flash'):
        from flash_models import register
        register()
        if args.expert_cache_gib is None:
            args.expert_cache_gib = selected.expert_cache_gib
    offload = args.expert_cache_gib is not None
    model, tokenizer = load(profile['path'], lazy=offload, trust_remote_code=True)
    cache = None
    embedding_cache = None
    if profile['config']['model_type'] == 'qwen4_exp':
        from flash_models import install_embedding_offload
        embedding_cache = install_embedding_offload(model, profile['path'])
    if offload:
        cache = install_expert_offload(model, profile['path'], int(args.expert_cache_gib*2**30))
        gc.collect()
        mx.clear_cache()
        mx.eval(model.parameters())
    load_seconds = time.monotonic()-start
    loaded_bytes = mx.get_active_memory()
    resident_parameters = sum(a.nbytes for _,a in tree_flatten(model.parameters()))
    prompt = tokenizer.apply_chat_template([{'role':'user','content':args.prompt}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                           thinking_mode='chat')
    responses = []
    started = time.monotonic()
    for response in stream_generate(model,tokenizer,prompt,max_tokens=args.tokens,
                                     sampler=make_sampler(temp=0),
                                     kv_bits=4 if uses_quantized_kv(profile['config']) else None, kv_group_size=64,
                                     quantized_kv_start=0, prefill_step_size=32):
        responses.append(response)
        print(response.text, end='', flush=True)
    result = {'model':args.model,'expert_cache_gib':args.expert_cache_gib,
              'load_seconds':load_seconds, 'generation_seconds':time.monotonic()-started,
              'loaded_mlx_bytes':loaded_bytes, 'resident_parameter_bytes':resident_parameters,
              'peak_mlx_bytes':mx.get_peak_memory(), 'tokens':[r.token for r in responses],
              'text':''.join(r.text for r in responses),
              'context_bytes_per_token':ContextPolicy(profile['config']).bytes_per_token}
    if responses:
        result.update(generation_tps=responses[-1].generation_tps,
                      prompt_tokens=responses[-1].prompt_tokens)
    if cache:
        result.update(cache.stats())
    suffix = 'offload' if offload else 'resident'
    out = args.output or ROOT/f'{args.model}-{suffix}-check.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('\n'+json.dumps({k:v for k,v in result.items() if k not in ('tokens','text')},indent=2))
    if args.model.startswith('qwen') and args.prompt.startswith('What is 2 + 2?'):
        assert '4' in result['text'], result['text']


if __name__ == '__main__':
    main()
