"""MLX block decoders for the GGUF quant types MLX cannot load natively.

Covers the types used by the pinned Unsloth IQ1 quants of the two Flash text
profiles: IQ1_S, IQ1_M, IQ2_XXS, IQ3_XXS, IQ4_NL and MXFP4. Grid and sign
tables come from the pinned ``gguf`` package so there is one source of truth;
MLX evaluates the block arithmetic on device. Every decoder is bit-exact
against the reference implementation (see ``test_iq.py``).

Types without a device decoder fall back to the reference implementation.
"""
import numpy as np
import mlx.core as mx
from gguf import GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.quants import IQ1_S, IQ1_M, IQ2_XXS, IQ3_XXS, IQ4_NL, MXFP4, dequantize as _np_dequantize

QK_K = 256
DELTA = 0.125


def _grid(cls, shape):
    cls.init_grid() if cls.grid is None else None
    return mx.array(np.array(cls.grid, dtype=np.float32).reshape(shape))


GRID_IQ1 = _grid(IQ1_S, (2048, 8))          # IQ1_S and IQ1_M share this grid
IQ1_M.init_grid() if IQ1_M.grid is None else None   # reference builds its grid lazily per class
GRID_IQ2 = _grid(IQ2_XXS, (256, 8))
GRID_IQ3 = _grid(IQ3_XXS, (256, 4))
KSIGNS = mx.array(np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).reshape(128))
KVALUES_IQ4 = mx.array(np.array(IQ4_NL.kvalues, dtype=np.float32).reshape(16))
KVALUES_MXFP4 = mx.array(np.array(MXFP4.kvalues, dtype=np.float32).reshape(16))


def _half_view(x, dtype=mx.float16):
    """Reinterpret the trailing byte pairs of a uint8 array as f16."""
    return mx.view(x, dtype)


def _signs_from_fields(words, axis_reshape):
    """Decode IQ2/IQ3 sign words: 7-bit index into ksigns, then one bit per value."""
    fields = words >> mx.array([0, 7, 14, 21], dtype=mx.uint32).reshape(*[1] * (words.ndim - 1), 4)
    fields = (fields & 0x7F).reshape(*words.shape[:-1], 4, 1)
    bits = KSIGNS[fields]
    bits = (bits >> mx.arange(8, dtype=mx.uint8).reshape(1, 1, 1, 8)) & 1
    return mx.where(bits == 0, 1.0, -1.0).reshape(*axis_reshape)


def dequantize_iq1_s(blocks):
    """(n, 50) uint8 -> (n, 256) float32."""
    d = _half_view(blocks[:, :2]).astype(mx.float32).reshape(-1, 1)
    qs = blocks[:, 2:34].astype(mx.uint16).reshape(-1, 8, 4)
    qh = mx.view(blocks[:, 34:50], mx.uint16).reshape(-1, 8)
    dl = d * (2 * ((qh >> 12) & 7).astype(mx.float32) + 1)
    delta = mx.where((qh & 0x8000) == 0, DELTA, -DELTA).astype(mx.float32)
    idx = qs | (((qh[:, :, None] >> mx.array([0, 3, 6, 9], dtype=mx.uint16).reshape(1, 1, 4)) & 7) << 8)
    values = GRID_IQ1[idx]
    return (dl[:, :, None, None] * (values + delta[:, :, None, None])).reshape(-1, 256)


def dequantize_iq1_m(blocks):
    """(n, 56) uint8 -> (n, 256) float32. The f16 scale is split across four words."""
    qs, qh, scales = blocks[:, :32], blocks[:, 32:48], blocks[:, 48:56]
    words = mx.view(scales, mx.uint16)
    d_bits = (words & 0xF000) >> mx.array([12, 8, 4, 0], dtype=mx.uint16).reshape(1, 4)
    d = mx.view(d_bits[:, 0] | d_bits[:, 1] | d_bits[:, 2] | d_bits[:, 3], mx.float16).astype(mx.float32).reshape(-1, 1)
    sub = words.reshape(-1, 1) >> mx.array([0, 3, 6, 9], dtype=mx.uint16).reshape(1, 4)
    dl = (d * (2 * (sub & 7).reshape(-1, 16).astype(mx.float32) + 1)).reshape(-1, 8, 2, 1, 1)
    qh = qh.reshape(-1, 16, 1) >> mx.array([0, 4], dtype=mx.uint8).reshape(1, 1, 2)
    idx = qs.astype(mx.uint16) | (((qh & 7).astype(mx.uint16) << 8).reshape(-1, 32))
    delta = mx.where((qh & 8) == 0, DELTA, -DELTA).astype(mx.float32).reshape(-1, 8, 2, 2, 1)
    values = GRID_IQ1[idx].reshape(-1, 8, 2, 2, 8)
    return (dl * (values + delta)).reshape(-1, 256)


def dequantize_iq2_xxs(blocks):
    """(n, 66) uint8 -> (n, 256) float32."""
    d = _half_view(blocks[:, :2]).astype(mx.float32).reshape(-1, 1)
    qs = mx.view(blocks[:, 2:], mx.uint32).reshape(-1, 8, 2)
    db = (d * (0.5 + (qs[..., 1] >> 28).astype(mx.float32)) * 0.25).reshape(-1, 8, 1, 1)
    signs = _signs_from_fields(qs[..., 1].reshape(-1, 8, 1), (-1, 8, 4, 8))
    idx = mx.view(qs[..., 0], mx.uint8).reshape(-1, 8, 1, 1)
    values = GRID_IQ2[idx].reshape(-1, 8, 4, 8)
    return (db * values * signs).reshape(-1, 256)


def dequantize_iq3_xxs(blocks):
    """(n, 98) uint8 -> (n, 256) float32."""
    d = _half_view(blocks[:, :2]).astype(mx.float32).reshape(-1, 1)
    qs = blocks[:, 2:66]
    scales = mx.view(blocks[:, 66:98], mx.uint32).reshape(-1, 8)
    db = (d * (0.5 + (scales >> 28).astype(mx.float32)) * 0.5).reshape(-1, 8, 1, 1)
    signs = _signs_from_fields(scales.reshape(-1, 8, 1), (-1, 8, 4, 8))
    values = GRID_IQ3[qs.reshape(-1, 64, 1, 1)].reshape(-1, 8, 4, 8)
    return (db * values * signs).reshape(-1, 256)


def dequantize_iq4_nl(blocks):
    """(n, 18) uint8 -> (n, 32) float32."""
    d = _half_view(blocks[:, :2]).astype(mx.float32).reshape(-1, 1)
    qs = blocks[:, 2:18].reshape(-1, 1, 16) >> mx.array([0, 4], dtype=mx.uint8).reshape(1, 2, 1)
    return d * KVALUES_IQ4[(qs & 0x0F).reshape(-1, 32)]


def dequantize_mxfp4(blocks):
    """(n, 17) uint8 -> (n, 32) float32. Scale is an E8M0 exponent byte."""
    e = blocks[:, :1].astype(mx.uint32)
    bits = mx.where(e < 2, mx.array(0x00200000, dtype=mx.uint32) << e, (e - 1) << 23)
    d = mx.view(bits, mx.float32).reshape(-1, 1)
    qs = blocks[:, 1:17].reshape(-1, 1, 16) >> mx.array([0, 4], dtype=mx.uint8).reshape(1, 2, 1)
    return d * KVALUES_MXFP4[(qs & 0x0F).reshape(-1, 32)]


DEVICE_DECODERS = {
    'IQ1_S': dequantize_iq1_s,
    'IQ1_M': dequantize_iq1_m,
    'IQ2_XXS': dequantize_iq2_xxs,
    'IQ3_XXS': dequantize_iq3_xxs,
    'IQ4_NL': dequantize_iq4_nl,
    'MXFP4': dequantize_mxfp4,
}


def geometry(qtype):
    """Return (block elements, block bytes) for a GGML quantization type name."""
    quant = GGMLQuantizationType[qtype] if isinstance(qtype, str) else qtype
    return GGML_QUANT_SIZES[quant]


def dequantize(blocks, qtype):
    """Decode (n, block_bytes) uint8 blocks to (n, block_elements) float32."""
    name = qtype if isinstance(qtype, str) else qtype.name
    if name in DEVICE_DECODERS:
        return DEVICE_DECODERS[name](blocks)
    return mx.array(_np_dequantize(np.array(blocks), GGMLQuantizationType[name]))
