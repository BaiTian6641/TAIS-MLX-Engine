import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from flash_models import model_classes, install_embedding_offload
from expert_cache import install_expert_offload
from disk_cache import DiskPromptCache
import flash_cache


def qwen_config():
    config = json.loads(Path('research/qwen-flash-config.json').read_text())
    config.pop('quantization', None)
    config.pop('quantization_config', None)
    t = config['text_config']
    t.update(hidden_size=128, num_hidden_layers=4, num_attention_heads=2,
             num_key_value_heads=1, head_dim=64, vocab_size=128,
             num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128,
             shared_expert_intermediate_size=128, linear_num_key_heads=1,
             linear_num_value_heads=1, linear_key_head_dim=128,
             linear_value_head_dim=128, hc_lowrank=32, max_position_embeddings=4096,
             ple_embed_dim=128, heads_per_ngram=2, ngram_vocab_size_base=128,
             split_ngram_parts=4, make_ngram_vocab_size_divisible_by=32,
             indexer_head_dim=64, indexer_n_heads=2, indexer_budget=8,
             layer_types=['linear_attention']*3+['full_attention'],
             rope_parameters={'rope_type':'default','rope_theta':10000,
                              'partial_rotary_factor':0.5,'mrope_section':[4,6,6]})
    return config


def ds_config():
    return dict(model_type='deepseek_v4', hidden_size=128, vocab_size=128,
                num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                head_dim=64, q_lora_rank=64, qk_rope_head_dim=32,
                o_groups=1, o_lora_rank=64, n_routed_experts=8,
                n_shared_experts=1, num_experts_per_tok=2, moe_intermediate_size=128,
                intermediate_size=128, num_hash_layers=0, index_head_dim=64,
                index_n_heads=2, index_topk=4, compress_ratios=[0,4,128],
                sliding_window=8, max_position_embeddings=4096)


class FlashTests(unittest.TestCase):
    def test_deepseek_mxfp4_split_checkpoint(self):
        from mlx_lm.utils import load_model
        config = ds_config()
        config['num_hash_layers'] = 1
        config['quantization'] = {'group_size':32,'bits':4,'mode':'mxfp4'}
        cls,args = model_classes(config)
        model = cls(args.from_dict(config))
        nn.quantize(model,group_size=32,bits=4,mode='mxfp4',
                    class_predicate=lambda p,m: 'switch_mlp' in p and hasattr(m,'to_quantized'))
        weights = dict(tree_flatten(model.parameters()))
        # Match the published checkpoint's separate gate/up projections.
        for name in list(weights):
            if '.switch_mlp.gate_proj.' in name:
                value = weights[name]
                half = value.shape[1]//2
                weights[name] = value[:,:half]
                weights[name.replace('.gate_proj.','.up_proj.')] = value[:,half:]
        tokens = mx.array([[1,2,3,4]])
        expected = model(tokens,cache=model.make_cache())
        mx.eval(expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'config.json').write_text(json.dumps(config))
            mx.save_safetensors(str(path/'model.safetensors'),weights)
            loaded,_ = load_model(path,lazy=True,get_model_classes=model_classes)
            cache = install_expert_offload(loaded,path,100000)
            actual = loaded(tokens,cache=loaded.make_cache())
            mx.eval(actual)
            self.assertTrue(mx.allclose(expected,actual,atol=2e-4,rtol=2e-4))
            self.assertGreater(cache.read_bytes,0)

    def test_qwen_quantized_sparse_cache(self):
        from mlx_lm.generate import maybe_quantize_kv_cache
        cls, args = model_classes(qwen_config())
        model = cls(args.from_dict(qwen_config()))
        state = model.make_cache()
        tokens = mx.array([[1,2,3,4,5,6]])
        mx.eval(model(tokens, cache=state))
        maybe_quantize_kv_cache(state, 0, 32, 4)
        mx.eval(model(mx.array([[7]]), cache=state))
        with tempfile.TemporaryDirectory() as tmp:
            file = str(Path(tmp)/'cache.safetensors')
            flash_cache.save(file, state)
            restored = flash_cache.load(file)
            a = model(mx.array([[8,9]]), cache=state)
            b = model(mx.array([[8,9]]), cache=restored)
            mx.eval(a,b)
            self.assertTrue(mx.allclose(a,b,atol=1e-5,rtol=1e-5))
            self.assertEqual(restored[-1].offset, 9)
            self.assertEqual(restored[-1].index_keys.shape[1], 9)

    def test_deepseek_compression_boundary(self):
        config = ds_config()
        cls,args = model_classes(config)
        model = cls(args.from_dict(config))
        tokens = mx.array([[i%100+1 for i in range(132)]])
        whole = model(tokens,cache=model.make_cache())
        state = model.make_cache()
        parts = []
        for start in range(0,132,7):
            parts.append(model(tokens[:,start:start+7],cache=state))
            mx.eval(parts[-1])
        chunked = mx.concatenate(parts,axis=1)
        mx.eval(whole,chunked)
        self.assertTrue(mx.allclose(whole,chunked,atol=3e-3,rtol=3e-3))
        from runtime_support import ContextPolicy
        policy = ContextPolicy(config)
        self.assertGreaterEqual(policy.fixed_state_bytes + 132*policy.bytes_per_token,
                                sum(c.nbytes for c in state))
        with tempfile.TemporaryDirectory() as tmp:
            file = str(Path(tmp)/'pooled.safetensors')
            flash_cache.save(file,state)
            restored = flash_cache.load(file)
            a = model(mx.array([[2,3,4]]),cache=state)
            b = model(mx.array([[2,3,4]]),cache=restored)
            mx.eval(a,b)
            self.assertTrue(mx.allclose(a,b,atol=1e-5,rtol=1e-5))

    def check_architecture(self, config, embedding=False):
        cls, args = model_classes(config)
        model = cls(args.from_dict(config))
        nn.quantize(model, group_size=32, bits=4,
                    class_predicate=lambda p,m: hasattr(m, 'to_quantized') and ('switch_mlp' in p or '.ngram_embedding.shards.' in p))
        tokens = list(range(1,18))
        expected = model(mx.array([tokens]), cache=model.make_cache())
        mx.eval(expected)
        with tempfile.TemporaryDirectory() as tmp:
            mx.save_safetensors(str(Path(tmp)/'model.safetensors'), dict(tree_flatten(model.parameters())))
            experts = install_expert_offload(model, tmp, 200000)
            if embedding:
                table = install_embedding_offload(model, tmp, 4096)
            actual = model(mx.array([tokens]), cache=model.make_cache())
            mx.eval(actual)
            self.assertTrue(mx.allclose(expected, actual, atol=2e-4, rtol=2e-4))
            self.assertLessEqual(experts.bytes, 200000)
            if embedding:
                self.assertLessEqual(table.bytes, 4096)
            state = model.make_cache()
            mx.eval(model(mx.array([tokens]), cache=state))
            disk = DiskPromptCache(tmp, 'flash', codec=flash_cache)
            disk.insert_cache(('tiny',None,None), tokens+[18], state)
            self.assertEqual(len(disk), 1, disk.last_error)
            disk = DiskPromptCache(tmp, 'flash', codec=flash_cache)
            restored, rest = disk.fetch_nearest_cache(('tiny',None,None), tokens+[18,19])
            self.assertIsNotNone(restored, disk.last_error)
            full = model(mx.array([tokens+[18,19]]), cache=model.make_cache())
            continued = model(mx.array([rest]), cache=restored)
            mx.eval(full, continued)
            self.assertTrue(mx.allclose(full[:,-2:], continued, atol=3e-3, rtol=3e-3))

    def test_qwen_flash(self):
        self.check_architecture(qwen_config(), embedding=True)

    def test_deepseek_flash(self):
        self.check_architecture(ds_config())

if __name__ == '__main__':
    unittest.main()


class QuantizationKeyTest(unittest.TestCase):
    """The text-tower adapter must keep the checkpoint's per-module quantization map."""

    def test_keys_are_stripped_for_the_load_and_restored_after(self):
        import json
        from pathlib import Path
        import tempfile

        from flash_models import quantization_keys_for_text_tower

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            original = {'model_type': 'gemma4',
                        'quantization': {'group_size': 64, 'bits': 4,
                                         'language_model.model.layers.0.route': {'bits': 8}},
                        'quantization_config': {'language_model.model.layers.0.route': {'bits': 8}}}
            config_path = path / 'config.json'
            config_path.write_text(json.dumps(original))
            with quantization_keys_for_text_tower(path):
                inside = json.loads(config_path.read_text())
                self.assertIn('model.layers.0.route', inside['quantization'])
                self.assertNotIn('language_model.model.layers.0.route', inside['quantization'])
                self.assertIn('model.layers.0.route', inside['quantization_config'])
            self.assertEqual(json.loads(config_path.read_text()), original)

    def test_a_config_without_the_prefix_is_left_alone(self):
        import json
        from pathlib import Path
        import tempfile

        from flash_models import quantization_keys_for_text_tower

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = {'quantization': {'bits': 4, 'model.layers.0.route': {'bits': 8}}}
            (path / 'config.json').write_text(json.dumps(config))
            with quantization_keys_for_text_tower(path):
                self.assertEqual(json.loads((path / 'config.json').read_text()), config)
