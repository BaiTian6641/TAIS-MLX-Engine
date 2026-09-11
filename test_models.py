import json
from pathlib import Path
import tempfile
import unittest

from runtime_support import ContextPolicy, GIB
from model_profiles import parse_options, resolve_profile


def tiny_config(moe=False):
    return dict(model_type='qwen3_5_moe' if moe else 'qwen3_5', text_config=dict(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=1, num_key_value_heads=1, head_dim=64,
        vocab_size=128, max_position_embeddings=4096, full_attention_interval=2,
        linear_num_key_heads=1, linear_num_value_heads=1,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        num_experts=8 if moe else 0, num_experts_per_tok=2 if moe else 0,
        moe_intermediate_size=128 if moe else 0,
        shared_expert_intermediate_size=128 if moe else 0))


class HybridTests(unittest.TestCase):
    def test_context_counts_only_full_attention(self):
        policy = ContextPolicy(tiny_config())
        self.assertEqual(policy.bytes_per_token, 2*2*1*64*(.5+4/64))
        self.assertGreater(policy.fixed_state_bytes, 0)
        self.assertLessEqual(policy.capacity(40*GIB, 4*GIB, 52*GIB), 4096)

    def test_context_can_displace_hot_experts(self):
        config = json.loads(Path('model/config.json').read_text())
        policy = ContextPolicy(config)
        small = policy.expert_budget(30*GIB, 10*GIB, 52*GIB, 4*GIB, 4*GIB, 100, 1000)
        large = policy.expert_budget(30*GIB, 10*GIB, 52*GIB, 4*GIB, 4*GIB, 500000, 1000)
        self.assertEqual(small, 4*GIB)
        self.assertLess(large, small)

    def test_hybrid_cache_continuation_and_branch_miss(self):
        import mlx.core as mx
        from mlx_lm.models.qwen3_5 import Model, ModelArgs
        from disk_cache import DiskPromptCache
        model = Model(ModelArgs.from_dict(tiny_config()))
        tokens = [1,2,3,4]
        state = model.make_cache()
        mx.eval(model(mx.array([tokens]), cache=state))
        with tempfile.TemporaryDirectory() as tmp:
            disk = DiskPromptCache(tmp, 'hybrid')
            disk.insert_cache(('tiny',None,None), tokens+[5], state)
            self.assertEqual(len(disk), 1)
            disk = DiskPromptCache(tmp, 'hybrid')
            restored, rest = disk.fetch_nearest_cache(('tiny',None,None), tokens+[5,6])
            self.assertEqual(rest, [5,6])
            full = model(mx.array([tokens+[5,6]]), cache=model.make_cache())
            incremental = model(mx.array([rest]), cache=restored)
            mx.eval(full, incremental)
            self.assertTrue(mx.allclose(full[:,-2:], incremental, atol=2e-4, rtol=2e-4))
            self.assertEqual(disk.fetch_nearest_cache(('tiny',None,None), [1,2,9,4,5]),
                             (None,[1,2,9,4,5]))
            self.assertEqual(disk.fetch_nearest_cache(('tiny',None,None), tokens), (None,tokens))


class ExpertTests(unittest.TestCase):
    def test_quantized_projection_parity_and_budget(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm.models.switch_layers import SwitchGLU
        from expert_cache import install_expert_offload
        class Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = SwitchGLU(64, 128, 8)
            def __call__(self, x, indices):
                return self.experts(x, indices)
        model = Container()
        nn.quantize(model, group_size=64, bits=4)
        x = mx.random.normal((1,40,64))
        indices = mx.array([[[i%3,(i+1)%3] for i in range(40)]])
        expected = model(x, indices)
        mx.eval(expected)
        with tempfile.TemporaryDirectory() as tmp:
            mx.save_safetensors(str(Path(tmp)/'model.safetensors'), dict(tree_flatten(model.parameters())))
            cache = install_expert_offload(model, tmp, 60000)
            actual = model(x, indices)
            mx.eval(actual)
            self.assertTrue(mx.allclose(expected, actual, atol=1e-5, rtol=1e-5))
            self.assertLessEqual(cache.bytes, cache.max_bytes)
            self.assertGreater(cache.misses, 0)
            self.assertEqual(tree_flatten(model.parameters()), [])
            model(x, indices)
            self.assertGreater(cache.hits, 0)
            cache.resize(0)
            self.assertEqual(cache.bytes, 0)
            self.assertGreater(cache.evictions, 0)
            cold = model(x, indices)
            self.assertTrue(mx.allclose(expected, cold, atol=1e-5, rtol=1e-5))

    def test_tiny_qwen_moe_full_forward_parity(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
        from mlx_lm.models.switch_layers import SwitchLinear
        from expert_cache import install_expert_offload
        model = Model(ModelArgs.from_dict(tiny_config(moe=True)))
        nn.quantize(model, group_size=64, bits=4, class_predicate=lambda p,m: isinstance(m,SwitchLinear))
        tokens = mx.array([[1,2,3,4]])
        expected = model(tokens, cache=model.make_cache())
        mx.eval(expected)
        with tempfile.TemporaryDirectory() as tmp:
            mx.save_safetensors(str(Path(tmp)/'model.safetensors'), dict(tree_flatten(model.parameters())))
            cache = install_expert_offload(model, tmp, 100000)
            actual = model(tokens, cache=model.make_cache())
            mx.eval(actual)
            self.assertTrue(mx.allclose(expected, actual, atol=2e-4, rtol=2e-4))
            self.assertLessEqual(cache.bytes, 100000)


if __name__ == '__main__':
    unittest.main()
