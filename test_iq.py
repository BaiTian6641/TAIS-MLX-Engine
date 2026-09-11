"""Bit-exactness of the MLX GGUF block decoders against the pinned gguf reference."""
import unittest

import numpy as np
import mlx.core as mx
from gguf import GGMLQuantizationType as QT, GGML_QUANT_SIZES
from gguf.quants import IQ1_S, IQ1_M, IQ2_XXS, IQ3_XXS, IQ4_NL, MXFP4

import iq_quants


def random_blocks(qtype, count, seed):
    """Random blocks with a valid f16 scale so no NaN enters the comparison."""
    _, size = GGML_QUANT_SIZES[qtype]
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(count, size), dtype=np.uint8)
    if qtype is QT.MXFP4:
        blocks[:, 0] = rng.integers(120, 132, size=count, dtype=np.uint8)  # sane E8M0 exponents
    elif qtype is QT.IQ1_M:
        # IQ1_M builds its f16 scale from the top nibble of four words in the tail.
        words = blocks[:, 48:56].copy()
        packed = words.view(np.uint16)
        packed[:] = (packed & 0x0FFF) | (np.array([0x3, 0xC, 0x0, 0x0], dtype=np.uint16) << 12)
        blocks[:, 48:56] = words
    else:
        blocks[:, :2] = np.array([rng.uniform(0.02, 2.0)], dtype=np.float16).view(np.uint8)
    return blocks


class DecoderParityTests(unittest.TestCase):
    def check(self, qtype, reference):
        for seed in (0, 1, 7):
            blocks = random_blocks(qtype, 512, seed)
            expected = reference.dequantize_blocks(blocks.copy())
            actual = np.array(iq_quants.dequantize(mx.array(blocks), qtype.name))
            self.assertEqual(expected.shape, actual.shape, qtype.name)
            self.assertTrue(np.array_equal(expected, actual),
                            f'{qtype.name} seed {seed}: max diff {np.abs(expected - actual).max()}')

    def test_iq1_s(self):
        self.check(QT.IQ1_S, IQ1_S)

    def test_iq1_m(self):
        self.check(QT.IQ1_M, IQ1_M)

    def test_iq2_xxs(self):
        self.check(QT.IQ2_XXS, IQ2_XXS)

    def test_iq3_xxs(self):
        self.check(QT.IQ3_XXS, IQ3_XXS)

    def test_iq4_nl(self):
        self.check(QT.IQ4_NL, IQ4_NL)

    def test_mxfp4(self):
        self.check(QT.MXFP4, MXFP4)

    def test_geometry_matches_reference(self):
        for name in iq_quants.DEVICE_DECODERS:
            qtype = QT[name]
            elements, size = iq_quants.geometry(name)
            self.assertEqual((elements, size), GGML_QUANT_SIZES[qtype], name)
            decoded = iq_quants.dequantize(mx.array(random_blocks(qtype, 4, 3)), name)
            self.assertEqual(decoded.shape, (4, elements), name)


if __name__ == '__main__':
    unittest.main()
