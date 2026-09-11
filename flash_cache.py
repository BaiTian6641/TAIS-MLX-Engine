"""Explicit cache codec preserving auxiliary Flash state and legacy metadata."""
import json
import mlx.core as mx


def _classes():
    from vendor.flash_vlm.models.cache import ArraysCache, KVCache, QuantizedKVCache
    from vendor.flash_vlm.models.qwen4_exp.language import QSAKVCache, QSAQuantizedKVCache
    from vendor.deepseek_v4.cache import CacheList, RotatingKVCache
    return {c.__name__: c for c in (ArraysCache, KVCache, QuantizedKVCache,
                                   QSAKVCache, QSAQuantizedKVCache, CacheList, RotatingKVCache)}


def save(file_name, caches):
    arrays = {}
    def encode(v):
        if isinstance(v, mx.array):
            name = str(len(arrays))
            arrays[name] = v
            return {'array': name}
        if isinstance(v, (list, tuple)):
            return {'tuple' if isinstance(v, tuple) else 'list': [encode(x) for x in v]}
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        raise ValueError(f'Unsupported Flash cache state: {type(v)}')
    data = [{'class': type(c).__name__, 'state': encode(c.state),
             'meta': encode(c.meta_state)} for c in caches]
    mx.save_safetensors(file_name, arrays, {'flash_cache_v1': json.dumps(data)})


def load(file_name):
    arrays, metadata = mx.load(file_name, return_metadata=True)
    def decode(v):
        if not isinstance(v, dict):
            return v
        if 'array' in v:
            return arrays[v['array']]
        if 'tuple' in v:
            return tuple(decode(x) for x in v['tuple'])
        return [decode(x) for x in v['list']]
    classes = _classes()
    return [classes[c['class']].from_state(decode(c['state']), decode(c['meta']))
            for c in json.loads(metadata['flash_cache_v1'])]
