"""Direct GGUF reader: header, expert/row slicing and decode order.

These read the pinned Unsloth IQ1 quants when they are present and skip
otherwise, so the suite stays runnable on a machine without the 145 GiB of
GGUFs.
"""
from pathlib import Path
import os
import unittest

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType as QT
from gguf.quants import dequantize as reference_dequantize

from gguf_reader import GGUFIndex

QWEN = Path('gguf/qwen3.8-flash/UD-IQ1_S')
DEEPSEEK = Path('gguf/deepseek-v4-flash/UD-IQ1_S')


def shards(directory):
    """All shards of a quant directory, or [] while its download is unfinished."""
    if not directory.is_dir() or list(directory.parent.rglob('*.incomplete')):
        return []
    return sorted(directory.glob('*.gguf'))


def reference_rows(tensor, start_row, count):
    """Decode rows straight from the file, without the reader's offset maths."""
    with open(tensor.path, 'rb') as handle:
        raw = os.pread(handle.fileno(), count * tensor.row_bytes, tensor.offset + start_row * tensor.row_bytes)
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, tensor.block_bytes)
    return reference_dequantize(blocks, QT[tensor.qtype]).reshape(count, tensor.row_length)


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = shards(QWEN)
        if not files:
            raise unittest.SkipTest('Qwen IQ1 GGUF not downloaded')
        cls.index = GGUFIndex(files)

    def test_header_inventory(self):
        self.assertEqual(len(self.index), 1224)
        self.assertEqual(self.index.metadata['general.architecture'], 'qwen4exp')
        self.assertEqual(self.index.metadata['qwen4exp.expert_count'], 512)

    def test_expert_slices_match_bytes_on_disk(self):
        for name, expert in (('blk.0.ffn_gate_exps.weight', 5), ('blk.7.ffn_down_exps.weight', 300)):
            tensor = self.index.tensor(name)
            rows_per_expert = tensor.rows // tensor.experts
            expected = reference_rows(tensor, expert * rows_per_expert, rows_per_expert)
            actual = np.array(self.index.read_expert(name, expert, dtype=None))
            self.assertTrue(np.array_equal(expected, actual), name)

    def test_ple_rows_match_bytes_on_disk(self):
        tensor = self.index.tensor('per_layer_token_embd.weight')
        for row in (0, 1000, 320001535):
            expected = reference_rows(tensor, row, 1)
            actual = np.array(self.index.read_rows(tensor.name, row, row + 1, dtype=None))
            self.assertTrue(np.array_equal(expected, actual), f'row {row}')

    def test_gguf_dims_are_reversed_for_mlx(self):
        self.assertEqual(self.index.read('blk.0.ffn_down_shexp.weight').shape, (2560, 640))
        self.assertEqual(self.index.read('blk.0.ssm_a').shape, (48,))
        tensor = self.index.tensor('blk.0.ffn_gate_exps.weight')
        self.assertEqual(tensor.experts, 512)
        self.assertEqual(self.index.read_expert('blk.0.ffn_gate_exps.weight', 0).shape, (tensor.dims[1], tensor.dims[0]))

    def test_range_guards(self):
        with self.assertRaises(ValueError):
            self.index.read_expert('blk.0.ffn_gate_exps.weight', 512)
        with self.assertRaises(ValueError):
            self.index.read_expert('blk.0.ssm_a', 0)
        with self.assertRaises(ValueError):
            self.index.read_rows('blk.0.ffn_gate_exps.weight', 5, 5)
        with self.assertRaises(KeyError):
            self.index.tensor('blk.0.missing')


class QwenTransformTests(unittest.TestCase):
    def test_v_head_reorder_inverts_converter(self):
        """The converter maps HF index k*N+n to GGUF index n*K+k; invert it."""
        from gguf_model import _untile_v_heads

        key_heads, value_per_key, head_dim, width = 16, 3, 4, 5
        hf = np.arange(key_heads * value_per_key * head_dim * width, dtype=np.float32).reshape(
            key_heads * value_per_key * head_dim, width)
        # forward transform the converter applies (grouped -> tiled)
        tiled = hf.reshape(key_heads, value_per_key, head_dim, width).swapaxes(0, 1).reshape(hf.shape)
        restored = np.array(_untile_v_heads(mx.array(tiled), 0, key_heads, value_per_key, head_dim))
        self.assertTrue(np.array_equal(hf, restored))

    def test_v_head_reorder_on_columns(self):
        from gguf_model import _untile_v_heads

        key_heads, value_per_key, head_dim, rows = 16, 3, 4, 2
        hf = np.arange(rows * key_heads * value_per_key * head_dim, dtype=np.float32).reshape(
            rows, key_heads * value_per_key * head_dim)
        tiled = hf.reshape(rows, key_heads, value_per_key, head_dim).swapaxes(1, 2).reshape(hf.shape)
        restored = np.array(_untile_v_heads(mx.array(tiled), 1, key_heads, value_per_key, head_dim))
        self.assertTrue(np.array_equal(hf, restored))


class DeepSeekReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = shards(DEEPSEEK)
        if not files:
            raise unittest.SkipTest('DeepSeek IQ1 GGUF not downloaded')
        cls.index = GGUFIndex(files)

    def test_header_inventory(self):
        self.assertEqual(len(self.index), 1328)
        self.assertEqual(self.index.metadata['general.architecture'], 'deepseek4')
        self.assertEqual(self.index.metadata['deepseek4.expert_count'], 256)

    def test_expert_slices_match_bytes_on_disk(self):
        for name, expert in (('blk.10.ffn_down_exps.weight', 7), ('blk.10.ffn_gate_exps.weight', 200)):
            tensor = self.index.tensor(name)
            rows_per_expert = tensor.rows // tensor.experts
            expected = reference_rows(tensor, expert * rows_per_expert, rows_per_expert)
            actual = np.array(self.index.read_expert(name, expert, dtype=None))
            self.assertTrue(np.array_equal(expected, actual), name)


if __name__ == '__main__':
    unittest.main()
