import json
from pathlib import Path
import tempfile
import unittest

from context_extension import extend, restore


class ContextExtensionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        (self.path / 'config.json').write_text(json.dumps(
            {'model_type': 'llama', 'max_position_embeddings': 131072}) + '\n')
        self.original = (self.path / 'config.json').read_text()

    def tearDown(self):
        self.tmp.cleanup()

    def config(self):
        return json.loads((self.path / 'config.json').read_text())

    def test_extension_sets_scaling_and_raises_the_declared_maximum(self):
        result = extend(self.path, 2)
        self.assertTrue(result['changed'])
        config = self.config()
        self.assertEqual(config['max_position_embeddings'], 262144)
        self.assertEqual(config['rope_scaling']['rope_type'], 'yarn')
        self.assertEqual(config['rope_scaling']['original_max_position_embeddings'], 131072)

    def test_restore_returns_the_original_config(self):
        extend(self.path, 2)
        restore(self.path)
        self.assertEqual((self.path / 'config.json').read_text(), self.original)

    def test_a_nested_text_config_is_extended_where_it_lives(self):
        (self.path / 'config.json').write_text(json.dumps({
            'model_type': 'nested', 'text_config': {'hidden_size': 8, 'max_position_embeddings': 4096}}) + '\n')
        extend(self.path, 4)
        text = self.config()['text_config']
        self.assertEqual(text['max_position_embeddings'], 16384)
        self.assertEqual(text['rope_scaling']['factor'], 4.0)

    def test_existing_scaling_is_not_overwritten_without_force(self):
        (self.path / 'config.json').write_text(json.dumps({
            'max_position_embeddings': 131072,
            'rope_scaling': {'rope_type': 'linear', 'factor': 2}}) + '\n')
        with self.assertRaises(ValueError):
            extend(self.path, 2)
        self.assertEqual(extend(self.path, 2, force=True)['changed'], True)

    def test_repeating_the_same_extension_is_a_no_op(self):
        extend(self.path, 2)
        self.assertEqual(extend(self.path, 2)['changed'], False)

    def test_a_factor_of_one_is_refused(self):
        with self.assertRaises(ValueError):
            extend(self.path, 1)

    def test_dry_run_writes_nothing(self):
        result = extend(self.path, 8, dry_run=True)
        self.assertEqual(result['new_max'], 131072 * 8)
        self.assertEqual((self.path / 'config.json').read_text(), self.original)
