"""Bounded MLX expert residency with direct safetensors range reads.

Dense weights remain resident. Quantized routed projections load only selected
expert rows. This is an eager, correctness-first path: each projection completes
before temporary weights can be released. It trades sync/I/O for lower residency.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import time
import heapq
from collections import OrderedDict

import numpy as np


@dataclass(frozen=True)
class TensorRange:
    path: Path
    offset: int
    size: int
    shape: tuple
    dtype: str


@dataclass(frozen=True)
class JoinedRange:
    parts: tuple
    shape: tuple


class TensorIndex:
    def __init__(self, directory):
        self.tensors = {}
        self._files = OrderedDict()
        for path in sorted(Path(directory).glob('model*.safetensors')):
            with path.open('rb') as f:
                raw = f.read(8)
                if len(raw) != 8:
                    raise ValueError(f'Invalid safetensors header: {path}')
                size = struct.unpack('<Q', raw)[0]
                if size > 100 * 2**20:
                    raise ValueError(f'Oversized safetensors header: {path}')
                header = json.loads(f.read(size))
            file_size = path.stat().st_size
            for key, value in header.items():
                if key == '__metadata__':
                    continue
                start, end = value['data_offsets']
                if not (0 <= start <= end <= file_size - 8 - size):
                    raise ValueError(f'Invalid tensor offsets: {key}')
                self.tensors[key] = TensorRange(path, 8+size+start, end-start,
                                               tuple(value['shape']), value['dtype'])
        if not self.tensors:
            raise ValueError(f'No weight tensors in {directory}')

    def resolve(self, name):
        candidates = [name]
        if name.startswith('language_model.'):
            candidates += [name[len('language_model.'):],
                           name.replace('language_model.model.', 'model.language_model.', 1)]
        for key in candidates:
            if key in self.tensors:
                return self.tensors[key]
        raise ValueError(f'Expert tensor {name} missing; use an MLX-converted checkpoint with stacked switch projections')

    def read_expert(self, entry, expert):
        import mlx.core as mx
        if isinstance(entry, JoinedRange):
            return mx.concatenate([self.read_expert(p, expert) for p in entry.parts], axis=0)
        if not entry.shape or not 0 <= expert < entry.shape[0]:
            raise ValueError('Expert index out of bounds')
        dtype_map = {'U32': np.uint32, 'F16': np.float16, 'BF16': np.uint16,
                     'F32': np.float32, 'U8': np.uint8}
        if entry.dtype not in dtype_map:
            raise ValueError(f'Unsupported expert dtype {entry.dtype}')
        dtype = np.dtype(dtype_map[entry.dtype])
        size = math.prod(entry.shape[1:]) * dtype.itemsize
        if size * entry.shape[0] != entry.size:
            raise ValueError('Expert tensor size mismatch')
        # Reading only this expert avoids faulting/evaluating the full bank.
        if entry.path not in self._files:
            if len(self._files) >= 64:
                _, old = self._files.popitem(last=False)
                old.close()
            self._files[entry.path] = entry.path.open('rb', buffering=0)
        self._files.move_to_end(entry.path)
        raw = os.pread(self._files[entry.path].fileno(), size, entry.offset + expert * size)
        if len(raw) != size:
            raise OSError(f'Short expert read: {entry.path}')
        array = mx.array(np.frombuffer(raw, dtype=dtype).reshape(entry.shape[1:]).copy())
        if entry.dtype == 'BF16':
            array = array.view(mx.bfloat16)
        return array

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __del__(self):
        self.close()


class ExpertCache:
    def __init__(self, index, max_bytes, max_entries=None):
        if max_bytes < 0:
            raise ValueError('Expert cache budget must be nonnegative')
        self.index, self.max_bytes = index, int(max_bytes)
        self.max_entries = max_entries
        if max_entries is not None and max_entries < 1:
            raise ValueError('Expert entry limit must be positive')
        self.entries = {}
        self.frequency = {}
        self.last_used = {}
        self.bytes = self.hits = self.misses = self.read_bytes = self.evictions = 0
        self.clock = 0
        self.read_seconds = 0.0
        self._priority = []

    def _rebuild_priority(self):
        self._priority = [(self.frequency[k], self.last_used[k], k) for k in self.entries]
        heapq.heapify(self._priority)

    def _track(self, key):
        heapq.heappush(self._priority, (self.frequency[key], self.last_used[key], key))
        if len(self._priority) > 2 * len(self.entries) + 1024:
            self._rebuild_priority()

    def _victim(self):
        while self._priority:
            frequency, used, key = self._priority[0]
            if key in self.entries and used == self.last_used[key] and frequency == self.frequency[key]:
                return key
            heapq.heappop(self._priority)
        raise RuntimeError('Expert cache priority index is inconsistent')

    def _drop(self, key):
        arrays = self.entries.pop(key)
        self.bytes -= sum(a.nbytes for a in arrays.values())
        self.evictions += 1

    def resize(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        while self.entries and self.bytes > self.max_bytes:
            self._drop(self._victim())

    def fetch(self, projection, expert, ranges, count=1, evaluate=True):
        import mlx.core as mx
        key = projection, int(expert)
        self.clock += 1
        # Decay old routing preferences so a change of workload can replace them.
        if self.clock % 4096 == 0:
            # Bound routing-history metadata, especially for very large PLE
            # vocabularies whose cold rows may never be selected twice.
            self.frequency = {k: v/2 for k,v in self.frequency.items()
                              if k in self.entries or self.last_used.get(k, 0) >= self.clock - 65536}
            self.last_used = {k: self.last_used.get(k, 0) for k in self.frequency}
            self._rebuild_priority()
        self.frequency[key] = self.frequency.get(key, 0) + count
        self.last_used[key] = self.clock
        if key in self.entries:
            self.hits += 1
            self._track(key)
            return self.entries[key]
        self.misses += 1
        start = time.monotonic()
        arrays = {name: self.index.read_expert(entry, expert) for name,entry in ranges.items()}
        if evaluate:
            mx.eval(list(arrays.values()))
        self.read_seconds += time.monotonic() - start
        size = sum(a.nbytes for a in arrays.values())
        self.read_bytes += size
        if size <= self.max_bytes:
            admit = True
            while self.entries and (self.bytes + size > self.max_bytes or
                    self.max_entries is not None and len(self.entries) >= self.max_entries):
                victim = self._victim()
                if self.frequency[key] < self.frequency.get(victim,0):
                    admit = False
                    break
                self._drop(victim)
            if admit:
                self.entries[key] = arrays
                self.bytes += size
                self._track(key)
        return arrays

    def stats(self):
        return {'expert_cache_bytes': self.bytes, 'expert_budget_bytes': self.max_bytes,
                'expert_cached_projections': len(self.entries), 'expert_hits': self.hits,
                'expert_misses': self.misses, 'expert_read_bytes': self.read_bytes,
                'expert_read_seconds': self.read_seconds, 'expert_evictions': self.evictions}


def install_expert_offload(model, directory, max_bytes):
    """Replace lazy quantized switch modules BEFORE evaluating model parameters."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    from vendor.flash_vlm.models.switch_layers import QuantizedSwitchLinear as VLMQuantizedSwitch
    from vendor.flash_vlm.models.switch_layers import SwitchLinear as VLMSwitch
    from vendor.deepseek_v4.switch_layers import QuantizedSwitchLinear as DSQuantizedSwitch
    from vendor.deepseek_v4.switch_layers import SwitchLinear as DSSwitch

    class DiskSwitchLinear(nn.Module):
        def __init__(self, name, original, cache):
            super().__init__()
            self._name, self._cache = name, cache
            self.group_size, self.bits, self.mode = original.group_size, original.bits, original.mode
            self._ranges = {}
            for suffix in ('weight', 'scales', 'biases', 'bias'):
                if suffix in original and original[suffix] is not None:
                    entry = cache.index.resolve(name + '.' + suffix)
                    if entry.shape != tuple(original[suffix].shape):
                        # DeepSeek's runtime fuses gate/up; the MLX checkpoint
                        # stores them separately. Read both selected rows only.
                        if isinstance(original, DSQuantizedSwitch) and name.endswith('.gate_proj'):
                            up = cache.index.resolve(name[:-len('gate_proj')] + 'up_proj.' + suffix)
                            shape = (entry.shape[0], entry.shape[1] + up.shape[1], *entry.shape[2:])
                            if shape != tuple(original[suffix].shape):
                                raise ValueError(f'Fused expert shape mismatch: {name}.{suffix}')
                            entry = JoinedRange((entry, up), shape)
                        else:
                            raise ValueError(f'Expert shape mismatch: {name}.{suffix}')
                    self._ranges[suffix] = entry

        def __call__(self, x, indices, sorted_indices=False):
            mx.eval(indices)
            ids = np.array(indices)
            unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
            selected = [self._cache.fetch(self._name, int(i), self._ranges, int(count))
                        for i,count in zip(unique, counts)]
            banks = {name: mx.stack([row[name] for row in selected]) for name in self._ranges}
            mapped = mx.array(inverse.reshape(ids.shape).astype(np.uint32))
            out = mx.gather_qmm(x, banks['weight'], banks['scales'], banks.get('biases'),
                                rhs_indices=mapped, transpose=True, group_size=self.group_size,
                                bits=self.bits, mode=self.mode, sorted_indices=sorted_indices)
            if 'bias' in banks:
                out = out + mx.expand_dims(banks['bias'][mapped], -2)
            mx.eval(out)
            # Temporary stacks may be as large as one projection's active bank;
            # they do not remain in the model or the persistent hot cache.
            del banks, selected
            mx.clear_cache()
            return out

    index = TensorIndex(directory)
    cache = ExpertCache(index, max_bytes)
    replacements = []
    for name, module in tree_flatten(model.leaf_modules(), is_leaf=lambda x: isinstance(x, nn.Module)):
        if isinstance(module, (QuantizedSwitchLinear, VLMQuantizedSwitch, DSQuantizedSwitch)):
            replacements.append((name, DiskSwitchLinear(name, module, cache)))
        elif isinstance(module, (SwitchLinear, VLMSwitch, DSSwitch)):
            raise ValueError('Expert offload currently requires quantized MLX switch weights')
    if not replacements:
        raise ValueError('No supported routed expert projections found in this checkpoint')
    model.update_modules(tree_unflatten(replacements))
    return cache
