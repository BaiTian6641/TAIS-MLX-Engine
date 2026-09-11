"""Spark2.5 has no reference runtime here, so correctness is pinned by invariants.

The load-bearing one is that decoding token by token through the caches produces
the same logits as one forward over the whole sequence. That single comparison
covers rotary offsets, the sliding/full mask split, and cache handling at once -
the three places a port of this architecture goes wrong.
"""
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache, RotatingKVCache

from vendor.spark2_5 import Model, ModelArgs, sanitize


def tiny_args(**overrides):
    params = dict(hidden_size=64, num_hidden_layers=4, intermediate_size=128,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                  vocab_size=128, max_position_embeddings=4096, sliding_window=4,
                  layer_types=['sliding_attention', 'sliding_attention',
                               'sliding_attention', 'full_attention'],
                  rope_parameters={'full_attention': {'partial_rotary_factor': 0.25,
                                                      'rope_theta': 5000000},
                                   'sliding_attention': {'partial_rotary_factor': 1.0,
                                                         'rope_theta': 10000}})
    params.update(overrides)
    return ModelArgs.from_dict(params)


class Spark25Test(unittest.TestCase):
    def test_incremental_decoding_matches_a_single_forward(self):
        mx.random.seed(0)
        model = Model(tiny_args())
        tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])

        whole = model(tokens)

        cache = model.make_cache()
        steps = []
        for position in range(tokens.shape[1]):
            steps.append(model(tokens[:, position:position + 1], cache=cache))
        incremental = mx.concatenate(steps, axis=1)

        diff = mx.abs(whole - incremental).max()
        self.assertLess(float(diff), 1e-4, f'incremental decoding diverged by {float(diff)}')

    def test_sliding_layers_get_a_bounded_cache_and_full_layers_do_not(self):
        model = Model(tiny_args())
        caches = model.make_cache()
        self.assertIsInstance(caches[0], RotatingKVCache)
        self.assertEqual(caches[0].max_size, 4)
        self.assertIsInstance(caches[3], KVCache)
        self.assertFalse(isinstance(caches[3], RotatingKVCache))

    def test_the_two_layer_types_use_different_rotary_embeddings(self):
        model = Model(tiny_args())
        sliding = model.layers[0].self_attn.rope
        full = model.layers[3].self_attn.rope
        # A quarter of the head dimensions at theta 10k for sliding layers, all
        # of them at theta 5M for full ones.
        self.assertEqual(sliding.dims, 32)
        self.assertEqual(full.dims, 8)
        self.assertNotEqual(sliding.base, full.base)

    def test_logits_are_finite_and_vocabulary_shaped(self):
        model = Model(tiny_args())
        logits = model(mx.array([[1, 2, 3]]))
        self.assertEqual(logits.shape, (1, 3, 128))
        self.assertTrue(bool(mx.all(mx.isfinite(logits))))

    def test_sanitize_maps_the_published_names_onto_this_module(self):
        weights = {'model.embedding.weight': mx.zeros((8, 8)),
                   'model.layers.0.self_attn.q_k_v_proj.weight': mx.zeros((4, 8)),
                   'model.layers.0.self_attn.rotary_emb.inv_freq': mx.zeros((4,)),
                   'model.norm.weight': mx.zeros((8,))}
        mapped = sanitize(weights)
        self.assertIn('embed_tokens.weight', mapped)
        self.assertIn('layers.0.self_attn.q_k_v_proj.weight', mapped)
        self.assertIn('norm.weight', mapped)
        self.assertNotIn('model.embedding.weight', mapped)
        self.assertFalse(any('rotary_emb' in key for key in mapped))

    def test_every_parameter_is_reachable_from_the_checkpoint_names(self):
        """The port must declare exactly what a published checkpoint carries."""
        model = Model(tiny_args())
        expected = {key for key, _ in tree_flatten(model.parameters())}
        published = {'embed_tokens.weight', 'norm.weight'}
        for index in range(4):
            prefix = f'layers.{index}'
            published |= {f'{prefix}.input_layernorm.weight',
                          f'{prefix}.post_attention_layernorm.weight',
                          f'{prefix}.self_attn.q_k_v_proj.weight',
                          f'{prefix}.self_attn.g_proj.weight',
                          f'{prefix}.self_attn.out_proj.weight',
                          f'{prefix}.mlp.gate_proj.weight',
                          f'{prefix}.mlp.up_proj.weight',
                          f'{prefix}.mlp.down_proj.weight'}
        self.assertEqual(expected, published)
