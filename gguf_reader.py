"""Direct GGUF tensor access: header parsing, expert/row slicing, MLX decode.

The pinned ``gguf`` reader loads whole tensors and needs the full file mapped;
this module only reads the header, then slices tensor data with ``os.pread`` so
a router-selected expert or an n-gram row costs one bounded read. Block decoding
is delegated to ``iq_quants`` (device) with its reference fallback.
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct

import numpy as np
import mlx.core as mx

from gguf import GGMLQuantizationType, GGML_QUANT_SIZES
from iq_quants import dequantize, geometry

MAX_OPEN_FILES = 64
_HEADER_LIMIT = 512 * 2**20


def _skip_value(buf, off, value_type):
    """Advance ``off`` past one GGUF metadata value; return (off, scalar_or_None)."""
    if value_type == 0:
        return off + 1, buf[off]
    if value_type == 1:
        return off + 1, struct.unpack_from('<b', buf, off)[0]
    if value_type == 2:
        return off + 2, struct.unpack_from('<H', buf, off)[0]
    if value_type == 3:
        return off + 2, struct.unpack_from('<h', buf, off)[0]
    if value_type == 4:
        return off + 4, struct.unpack_from('<I', buf, off)[0]
    if value_type == 5:
        return off + 4, struct.unpack_from('<i', buf, off)[0]
    if value_type == 6:
        return off + 4, struct.unpack_from('<f', buf, off)[0]
    if value_type == 7:
        return off + 1, bool(buf[off])
    if value_type == 8:
        size = struct.unpack_from('<Q', buf, off)[0]
        off += 8
        return off + size, buf[off:off + size].decode('utf-8', 'replace')
    if value_type == 9:
        element_type = struct.unpack_from('<I', buf, off)[0]
        count = struct.unpack_from('<Q', buf, off + 4)[0]
        off += 12
        if element_type == 8:
            for _ in range(count):
                size = struct.unpack_from('<Q', buf, off)[0]
                off += 8 + size
            return off, f'<{count} strings>'
        width = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[element_type]
        return off + width * count, f'<{count} values>'
    if value_type == 10:
        return off + 8, struct.unpack_from('<Q', buf, off)[0]
    if value_type == 11:
        return off + 8, struct.unpack_from('<q', buf, off)[0]
    if value_type == 12:
        return off + 8, struct.unpack_from('<d', buf, off)[0]
    raise ValueError(f'Unsupported GGUF metadata value type {value_type}')


def read_header(path):
    """Return (metadata, tensors, data_start) for one GGUF shard.

    ``tensors`` entries are (name, dims, type_name, relative_offset); ``dims``
    follows GGUF order, so ``dims[0]`` is the contiguous row dimension.
    ``data_start`` is the absolute file offset the relative offsets count from.
    """
    with Path(path).open('rb') as handle:
        raw = handle.read(24)
        magic, version, tensor_count, kv_count = struct.unpack_from('<4sIQQ', raw, 0)
        if magic != b'GGUF':
            raise ValueError(f'Not a GGUF file: {path}')
        if version != 3:
            raise ValueError(f'Unsupported GGUF version {version} in {path}')
        size = min(_HEADER_LIMIT, os.fstat(handle.fileno()).st_size)
        buf = handle.read(size - 24)
    off = 0
    metadata = {}
    for _ in range(kv_count):
        key_len = struct.unpack_from('<Q', buf, off)[0]
        off += 8
        key = buf[off:off + key_len].decode('utf-8', 'replace')
        off += key_len
        value_type = struct.unpack_from('<I', buf, off)[0]
        off += 4
        off, value = _skip_value(buf, off, value_type)
        metadata[key] = value
    tensors = []
    for _ in range(tensor_count):
        name_len = struct.unpack_from('<Q', buf, off)[0]
        off += 8
        name = buf[off:off + name_len].decode('utf-8', 'replace')
        off += name_len
        dims = struct.unpack_from('<I', buf, off)[0]
        off += 4
        shape = struct.unpack_from(f'<{dims}Q', buf, off)
        off += 8 * dims
        quant = struct.unpack_from('<I', buf, off)[0]
        off += 4
        offset = struct.unpack_from('<Q', buf, off)[0]
        off += 8
        qtype = GGMLQuantizationType(quant)
        block, block_bytes = GGML_QUANT_SIZES[qtype]
        elements = math.prod(shape)
        if elements % block:
            raise ValueError(f'{name}: {elements} elements is not a multiple of block {block}')
        tensors.append((name, tuple(shape), qtype.name, offset, elements // block * block_bytes))
    alignment = int(metadata.get('general.alignment', 32))
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError(f'Invalid GGUF alignment {alignment}')
    data_start = 24 + off
    data_start += -data_start % alignment
    return metadata, tensors, data_start


@dataclass(frozen=True)
class Tensor:
    name: str
    path: Path
    dims: tuple
    qtype: str
    offset: int
    nbytes: int

    @property
    def block_elements(self):
        return geometry(self.qtype)[0]

    @property
    def block_bytes(self):
        return geometry(self.qtype)[1]

    @property
    def row_length(self):
        return self.dims[0]

    @property
    def rows(self):
        return math.prod(self.dims[1:]) if len(self.dims) > 1 else 1

    @property
    def row_bytes(self):
        return self.row_length // self.block_elements * self.block_bytes

    @property
    def experts(self):
        return self.dims[2] if len(self.dims) > 2 else 1


_RAW_TYPES = {
    'F32': (np.float32, mx.float32),
    'F16': (np.float16, mx.float16),
    'BF16': (np.uint16, mx.bfloat16),
    'I32': (np.int32, mx.int32),
    'I16': (np.int16, mx.int16),
    'I64': (np.int64, mx.int64),
    'U32': (np.uint32, mx.uint32),
    'F64': (np.float64, mx.float64),
}


def _decode(raw, qtype, rows, row_length, block_bytes):
    """Decode one packed read into (rows, row_length) values."""
    if qtype in _RAW_TYPES:
        np_dtype, mx_dtype = _RAW_TYPES[qtype]
        values = mx.array(np.frombuffer(raw, dtype=np_dtype).reshape(rows, row_length))
        return values.view(mx_dtype) if qtype == 'BF16' else values
    from iq_quants import dequantize
    blocks = mx.array(raw.reshape(-1, block_bytes))
    return dequantize(blocks, qtype).reshape(rows, row_length).astype(mx.float32)


class GGUFIndex:
    """Tensor directory over one or more shards, with a bounded file-handle pool."""

    def __init__(self, files):
        self.files = [Path(f) for f in files]
        if not self.files:
            raise ValueError('No GGUF files given')
        self.tensors = {}
        self.metadata = {}
        self._handles = OrderedDict()
        self._shards = {}
        for path in self.files:
            metadata, entries, data_start = read_header(path)
            self._shards[path] = data_start
            for key, value in metadata.items():
                self.metadata.setdefault(key, value)
            for name, dims, qtype, offset, nbytes in entries:
                if name in self.tensors:
                    raise ValueError(f'Duplicate GGUF tensor {name}')
                self.tensors[name] = Tensor(name, path, dims, qtype, data_start + offset, nbytes)
        if not self.tensors:
            raise ValueError(f'No tensors in {[str(f) for f in self.files]}')

    def __len__(self):
        return len(self.tensors)

    def __contains__(self, name):
        return name in self.tensors

    def tensor(self, name):
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f'GGUF tensor {name} is not in {[f.name for f in self.files]}') from None

    def _handle(self, path):
        handle = self._handles.get(path)
        if handle is None:
            if len(self._handles) >= MAX_OPEN_FILES:
                _, oldest = self._handles.popitem(last=False)
                oldest.close()
            handle = path.open('rb', buffering=0)
            self._handles[path] = handle
        else:
            self._handles.move_to_end(path)
        return handle

    def _read_bytes(self, tensor, byte_offset, byte_count, keep=None):
        if byte_offset < 0 or byte_offset + byte_count > tensor.nbytes:
            raise ValueError(f'{tensor.name}: read of {byte_count} bytes at {byte_offset} exceeds {tensor.nbytes}')
        raw = os.pread(self._handle(tensor.path).fileno(), byte_count, tensor.offset + byte_offset)
        if len(raw) != byte_count:
            raise OSError(f'Short read for {tensor.name} from {tensor.path}')
        if keep is not None:
            # MLX may read from the buffer lazily, so the owner must outlive
            # evaluation; callers that defer evaluation pass a list here.
            keep.append(raw)
        return np.frombuffer(raw, dtype=np.uint8)

    def read_packed_rows(self, name, start=0, stop=None, keep=None):
        """Read rows [start, stop) without decoding, as (blocks, block_bytes) uint8."""
        tensor = self.tensor(name)
        stop = tensor.rows if stop is None else min(stop, tensor.rows)
        if not 0 <= start < stop:
            raise ValueError(f'{name}: invalid row range [{start}, {stop})')
        raw = self._read_bytes(tensor, start * tensor.row_bytes, (stop - start) * tensor.row_bytes, keep=keep)
        return mx.array(raw.reshape(-1, tensor.block_bytes))

    def read_packed_expert(self, name, expert, keep=None):
        """Packed bytes of one expert slice, for caching without decoded bloat."""
        tensor = self.tensor(name)
        if len(tensor.dims) < 3:
            raise ValueError(f'{name} is not a stacked expert tensor')
        if not 0 <= expert < tensor.experts:
            raise ValueError(f'{name}: expert {expert} out of range ({tensor.experts})')
        rows_per_expert = tensor.rows // tensor.experts
        start = expert * rows_per_expert
        return self.read_packed_rows(name, start, start + rows_per_expert, keep=keep)

    def read_rows(self, name, start=0, stop=None, dtype=mx.float16, chunk_rows=4096):
        """Decode rows [start, stop) of a tensor to (rows, row_length)."""
        tensor = self.tensor(name)
        stop = tensor.rows if stop is None else min(stop, tensor.rows)
        if not 0 <= start < stop:
            raise ValueError(f'{name}: invalid row range [{start}, {stop})')
        parts = []
        for first in range(start, stop, chunk_rows):
            last = min(first + chunk_rows, stop)
            raw = self._read_bytes(tensor, first * tensor.row_bytes, (last - first) * tensor.row_bytes)
            parts.append(_decode(raw, tensor.qtype, last - first, tensor.row_length, tensor.block_bytes))
        out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        return out.astype(dtype) if dtype is not None else out

    def read_expert(self, name, expert, dtype=mx.float16):
        """Decode one expert slice of a stacked (row, out, expert) tensor."""
        tensor = self.tensor(name)
        if len(tensor.dims) < 3:
            raise ValueError(f'{name} is not a stacked expert tensor')
        if not 0 <= expert < tensor.experts:
            raise ValueError(f'{name}: expert {expert} out of range ({tensor.experts})')
        rows_per_expert = tensor.rows // tensor.experts
        start = expert * rows_per_expert
        values = self.read_rows(name, start, start + rows_per_expert, dtype=None)
        return values.astype(dtype) if dtype is not None else values

    def read(self, name, dtype=mx.float16):
        """Decode a whole tensor; the result is in MLX order (GGUF dims reversed)."""
        tensor = self.tensor(name)
        values = self.read_rows(name, 0, tensor.rows, dtype=None).reshape(tuple(reversed(tensor.dims)))
        return values if dtype is None else values.astype(dtype)

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        self.close()
