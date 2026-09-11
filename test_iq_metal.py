"""Metal IQ decoders must agree with the reference and the MLX-op decoders."""
import unittest

import numpy as np
import mlx.core as mx
from gguf import GGMLQuantizationType as QT

import iq_metal
import iq_quants
from test_iq import random_blocks


class MetalDecoderTests(unittest.TestCase):
    def test_matches_reference_decoders(self):
        from gguf.quants import IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_NL, MXFP4

        for qtype, reference in ((QT.IQ1_S, IQ1_S), (QT.IQ2_XXS, IQ2_XXS), (QT.IQ3_XXS, IQ3_XXS),
                                 (QT.IQ4_NL, IQ4_NL), (QT.MXFP4, MXFP4)):
            blocks = random_blocks(qtype, 512, seed=11)
            expected = reference.dequantize_blocks(blocks.copy()).astype(np.float16)
            actual = np.array(iq_metal.dequantize(mx.array(blocks), qtype.name))
            self.assertEqual(expected.shape, actual.shape, qtype.name)
            self.assertTrue(np.array_equal(expected, actual),
                            f'{qtype.name}: max diff {np.abs(expected.astype(np.float32) - actual.astype(np.float32)).max()}')

    def test_matches_mlx_op_decoders(self):
        for name in iq_metal.KERNELS:
            blocks = random_blocks(QT[name], 256, seed=5)
            expected = np.array(iq_quants.dequantize(mx.array(blocks), name).astype(mx.float16))
            actual = np.array(iq_metal.dequantize(mx.array(blocks), name))
            self.assertTrue(np.array_equal(expected, actual), name)

    def test_rejects_wrong_block_width(self):
        with self.assertRaises(ValueError):
            iq_metal.dequantize(mx.zeros((4, 7), dtype=mx.uint8), 'IQ1_S')


if __name__ == '__main__':
    unittest.main()
