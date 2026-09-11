"""Block-diffusion sampler helpers: the entropy rule and temperature schedule."""
import unittest

import mlx.core as mx

from diffusion_engine import DiffusionConfig, _canvas_entropy, _entropy_transfer_mask, _temperature


class SamplerTests(unittest.TestCase):
    def test_temperature_schedule_runs_from_t_max_to_t_min(self):
        config = DiffusionConfig(max_denoising_steps=48, t_max=0.8, t_min=0.4)
        self.assertAlmostEqual(_temperature(48, 48, config), 0.8, places=6)
        self.assertAlmostEqual(_temperature(1, 48, config), 0.4 + 0.4 / 48, places=6)
        self.assertLess(_temperature(1, 48, config), _temperature(48, 48, config))

    def test_entropy_mask_accepts_the_low_entropy_prefix(self):
        # The reference rule walks positions in ascending entropy order and
        # keeps accepting while the accumulated entropy stays inside the bound,
        # so a tight bound keeps the confident positions and drops the rest.
        peaked, flat = 20.0, 0.0
        logits = mx.stack([mx.array([peaked, flat, flat]), mx.array([flat, flat, flat]),
                           mx.array([flat, flat, flat]), mx.array([peaked, flat, flat])])
        logits = logits[None]                     # one canvas row, four positions
        entropy = _canvas_entropy(logits)
        self.assertLess(float(entropy[0, 0]), float(entropy[0, 1]))

        wide = _entropy_transfer_mask(entropy, 100.0)
        self.assertTrue(bool(mx.all(wide).item()), 'a wide bound accepts every position')

        tight = _entropy_transfer_mask(entropy, 0.5)
        accepted = [bool(value) for value in tight[0].tolist()]
        self.assertTrue(accepted[0] and accepted[3], 'confident positions must pass')
        self.assertEqual(sum(accepted), 3, f'the accumulated bound stops the last one: {accepted}')

    def test_entropy_mask_selection_is_a_prefix_in_entropy_order(self):
        # The rule accepts a prefix of the positions ordered by entropy: every
        # accepted position must be no less confident than every rejected one.
        flat = mx.array([0.0, 0.0, 0.0])
        peaked = mx.array([20.0, 0.0, 0.0])
        logits = mx.stack([flat, peaked, flat, peaked])[None]
        entropy = _canvas_entropy(logits)
        mask = _entropy_transfer_mask(entropy, 0.5)
        values = [float(v) for v in entropy[0].tolist()]
        accepted = [v for v, keep in zip(values, mask[0].tolist()) if keep]
        rejected = [v for v, keep in zip(values, mask[0].tolist()) if not keep]
        self.assertTrue(accepted and rejected, f'accepted={accepted} rejected={rejected}')
        self.assertLessEqual(max(accepted), min(rejected))

    def test_config_defaults_match_model_card(self):
        config = DiffusionConfig.from_generation_config({
            'max_denoising_steps': 48, 'eos_token_id': [1, 106, 50], 't_max': 0.8, 't_min': 0.4,
            'sampler_config': {'_cls_name': 'EntropyBoundSamplerConfig', 'entropy_bound': 0.1}})
        self.assertEqual(config.max_denoising_steps, 48)
        self.assertEqual(config.eos_token_ids, (1, 106, 50))
        self.assertAlmostEqual(config.entropy_bound, 0.1)
        self.assertAlmostEqual(config.t_max, 0.8)
        self.assertAlmostEqual(config.t_min, 0.4)
        self.assertEqual(config.canvas_length, 256)


if __name__ == '__main__':
    unittest.main()
