"""Streaming expert module: parity against independently decoded GGUF slices.

These read the pinned DeepSeek IQ1 GGUF when present and skip otherwise. The
reference decodes the expert bytes straight from the file with the gguf
reference decoders, so it is independent of gguf_reader and iq_quants.
"""
import json
from pathlib import Path
import unittest

import numpy as np
import mlx.core as mx

from gguf_model import DEEPSEEK_EXPERTS, GGUFExpertStore, StreamingFusedSwitchGLU


def make_module(store, layer, hidden, moe_intermediate, limit, **kwargs):
    from vendor.deepseek_v4.model import _limited_swiglu
    return StreamingFusedSwitchGLU.for_layer(
        store, layer, DEEPSEEK_EXPERTS, hidden, moe_intermediate,
        lambda gate, up: _limited_swiglu(gate, up, limit), **kwargs)
from gguf_reader import GGUFIndex
from test_gguf_reader import DEEPSEEK, shards, reference_rows


class StreamingMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = shards(DEEPSEEK)
        if not files:
            raise unittest.SkipTest('DeepSeek IQ1 GGUF not downloaded')
        cls.index = GGUFIndex(files)
        cls.config = json.loads(Path('models/deepseek-v4-flash/config.json').read_text())
        cls.hidden = cls.config['hidden_size']
        cls.moe_intermediate = cls.config['moe_intermediate_size']

    def expert_matrix(self, tensor_name, expert):
        """(out, in) matrix for one expert, decoded independently of the reader."""
        tensor = self.index.tensor(tensor_name)
        rows_per_expert = tensor.rows // tensor.experts
        return reference_rows(tensor, expert * rows_per_expert, rows_per_expert)

    def test_matches_per_selection_reference(self):
        layer = 0
        limit = self.config.get('swiglu_limit', 10.0)
        module = make_module(GGUFExpertStore(self.index, 0), layer, self.hidden,
                             self.moe_intermediate, limit)
        rng = np.random.default_rng(0)
        tokens = 3
        top_k = 6
        # Duplicates exercise the sort/unsort path used to batch by expert.
        picks = np.array([[3, 3, 17, 42, 255, 42], [0, 1, 1, 200, 7, 9], [255, 254, 1, 1, 1, 0]])
        x = (rng.normal(size=(1, tokens, self.hidden)) * 0.1).astype(np.float16)
        indices = mx.array(picks.reshape(1, tokens, top_k).astype(np.uint32))
        got = np.array(module(mx.array(x), indices), dtype=np.float32)

        expected = np.zeros((1, tokens, top_k, self.hidden), dtype=np.float32)
        for t in range(tokens):
            vector = x[0, t].astype(np.float32)
            for slot in range(top_k):
                expert = int(picks[t, slot])
                gate = self.expert_matrix(f'blk.{layer}.ffn_gate_exps.weight', expert)
                up = self.expert_matrix(f'blk.{layer}.ffn_up_exps.weight', expert)
                down = self.expert_matrix(f'blk.{layer}.ffn_down_exps.weight', expert)
                g = np.minimum(vector @ gate.T.astype(np.float32), limit)
                u = np.clip(vector @ up.T.astype(np.float32), -limit, limit)
                expected[0, t, slot] = (g / (1 + np.exp(-g)) * u) @ down.T.astype(np.float32)

        scale = np.abs(expected).max()
        self.assertLess(np.abs(expected - got).max() / scale, 5e-3)
        # A wrong expert or a mis-grouped token would shift values far beyond that.
        self.assertGreater(scale, 1e-3)

    def test_batched_banks_match_single_batch(self):
        limit = self.config.get('swiglu_limit', 10.0)
        store = GGUFExpertStore(self.index, 256 * 2**20)
        chunked = make_module(store, 0, self.hidden, self.moe_intermediate, limit, max_bank_experts=2)
        single = make_module(store, 0, self.hidden, self.moe_intermediate, limit, max_bank_experts=512)
        rng = np.random.default_rng(3)
        tokens, top_k = 6, 6
        picks = rng.integers(0, 24, size=(1, tokens, top_k))          # >2 unique experts per layer
        x = (rng.normal(size=(1, tokens, self.hidden)) * 0.1).astype(np.float16)
        indices = mx.array(picks.astype(np.uint32))
        a = chunked(mx.array(x), indices)
        b = single(mx.array(x), indices)
        mx.eval(a, b)
        self.assertLess(float(mx.max(mx.abs(a - b))), 1e-3)

    def test_budget_bounds_hot_set(self):
        store = GGUFExpertStore(self.index, 12 * 2**20)
        module = make_module(store, 0, self.hidden, self.moe_intermediate, 10.0)
        indices = mx.array(np.arange(0, 12, dtype=np.uint32).reshape(1, 2, 6))
        x = mx.zeros((1, 2, self.hidden), dtype=mx.float16)
        module(x, indices)
        self.assertLessEqual(store.cache.bytes, store.max_bytes)
        self.assertGreater(store.cache.misses, 0)


if __name__ == '__main__':
    unittest.main()
