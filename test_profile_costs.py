"""The selector's numbers must be arithmetic, not vibes: context from memory, speed
from measurement or a calibrated model."""
import json
from pathlib import Path
import tempfile
import unittest

from profile_costs import (EFFECTIVE_BANDWIDTH, MEASURED, checkpoint_bytes,
                           decode_estimate, profile_costs)


def write_checkpoint(root, shard_bytes=1 << 20, config=None):
    path = Path(root)
    (path / 'model-00001-of-00002.safetensors').write_bytes(b'0' * shard_bytes)
    (path / 'model-00002-of-00002.safetensors').write_bytes(b'0' * shard_bytes)
    (path / 'config.json').write_text(json.dumps(config or {
        'model_type': 'llama', 'hidden_size': 256, 'num_hidden_layers': 4,
        'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 64,
        'max_position_embeddings': 131072,
    }))
    return path


class ProfileCostsTest(unittest.TestCase):
    def test_checkpoint_bytes_sums_the_shards_not_the_index(self):
        """Some converters record the pre-quantization size in the index."""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_checkpoint(tmp, shard_bytes=1000)
            (path / 'model.safetensors.index.json').write_text(
                json.dumps({'metadata': {'total_size': 999_999_999}}))
            self.assertEqual(checkpoint_bytes(path), 2000)

    def test_context_is_limited_by_native_window_when_memory_is_plentiful(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_checkpoint(tmp)
            profile = {'alias': 'toy', 'path': str(path),
                       'config': json.loads((path / 'config.json').read_text())}
            costs = profile_costs(profile, available_bytes=64 << 30, working_set_bytes=48 << 30)
            self.assertEqual(costs['max_context'], costs['native_context'])

    def test_context_shrinks_when_there_is_less_memory(self):
        # A model whose cache is large enough to be the binding constraint, which
        # is the case this computation exists for.
        heavy = {'model_type': 'llama', 'hidden_size': 4096, 'num_hidden_layers': 80,
                 'num_attention_heads': 32, 'num_key_value_heads': 32, 'head_dim': 128,
                 'max_position_embeddings': 262144}
        with tempfile.TemporaryDirectory() as tmp:
            path = write_checkpoint(tmp, config=heavy)
            profile = {'alias': 'toy', 'path': str(path),
                       'config': json.loads((path / 'config.json').read_text())}
            roomy = profile_costs(profile, available_bytes=60 << 30, working_set_bytes=64 << 30)
            tight = profile_costs(profile, available_bytes=20 << 30, working_set_bytes=64 << 30)
            self.assertLess(roomy['max_context'], roomy['native_context'])
            self.assertLess(tight['max_context'], roomy['max_context'])

    def test_measured_profiles_report_measurements(self):
        alias = next(iter(MEASURED))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_checkpoint(tmp)
            profile = {'alias': alias, 'path': str(path),
                       'config': json.loads((path / 'config.json').read_text())}
            costs = profile_costs(profile, available_bytes=64 << 30, working_set_bytes=48 << 30)
            self.assertEqual(costs['source'], 'measured')
            self.assertEqual(costs['decode'], MEASURED[alias][0])

    def test_unknown_profiles_fall_back_to_the_bandwidth_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_checkpoint(tmp, shard_bytes=1 << 20)
            profile = {'alias': 'no-such-profile', 'path': str(path),
                       'config': json.loads((path / 'config.json').read_text())}
            costs = profile_costs(profile, available_bytes=64 << 30, working_set_bytes=48 << 30)
            self.assertEqual(costs['source'], 'estimated')
            self.assertIsNone(costs['prefill'])
            self.assertAlmostEqual(costs['decode'], decode_estimate(2 << 20), places=3)

    def test_the_estimate_is_monotonic_in_size(self):
        small = decode_estimate(1 << 30)
        large = decode_estimate(16 << 30)
        self.assertGreater(small, large)
        self.assertGreater(large, 0)
        self.assertLess(small, EFFECTIVE_BANDWIDTH)
