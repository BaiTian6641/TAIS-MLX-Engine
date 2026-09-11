"""Small GPU architecture probes, not full-model throughput benchmarks."""
import json
from pathlib import Path


def main():
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    from mlx_lm.models.cache import can_trim_prompt_cache
    from mlx_lm.generate import BatchGenerator, speculative_generate_step
    from test_models import tiny_config
    results = {}
    for moe in (False, True):
        model = Model(ModelArgs.from_dict(tiny_config(moe)))
        generator = BatchGenerator(model, max_tokens=4, completion_batch_size=2,
                                   prefill_batch_size=2, prefill_step_size=8)
        uids = generator.insert([[1,2,3],[4,5]])
        outputs = {uid:[] for uid in uids}
        max_parallel = 0
        try:
            for _ in range(32):
                _, tokens = generator.next()
                max_parallel = max(max_parallel,len(tokens))
                for token in tokens:
                    outputs[token.uid].append(token.token)
                if all(len(v)==4 for v in outputs.values()):
                    break
        finally:
            generator.close()
        assert max_parallel == 2 and all(len(v)==4 for v in outputs.values()), outputs
        cache = model.make_cache()
        assert not can_trim_prompt_cache(cache)
        try:
            next(speculative_generate_step(mx.array([1,2,3]), model, model, max_tokens=4))
        except ValueError as exc:
            rejection = str(exc)
        else:
            raise AssertionError('Expected recurrent-cache speculative rejection')
        results['moe' if moe else 'dense'] = {
            'batch_size_observed':max_parallel, 'generated_tokens':outputs,
            'cache_types':[type(c).__name__ for c in cache],
            'standard_speculative_rejection':rejection,
            'kv_quantization':'none in BatchGenerator; production wrapper uses 4-bit',
        }
    Path('decoding-probe.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
