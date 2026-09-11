"""Single-dispatch Metal decoders for the packed GGUF IQ formats.

``iq_quants`` decodes with a chain of MLX operations; this module does the same
work in one Metal kernel per tensor, which matters because a decode step
decodes every selected expert matrix again (about 258 per token here). Grids,
sign tables and value tables are passed as buffers built from the pinned ``gguf``
package so both paths share one source of truth. Output is float16.
"""
import numpy as np
import mlx.core as mx
from gguf import GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.quants import IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_NL, MXFP4


def _grid(cls, shape):
    cls.init_grid() if cls.grid is None else None
    return mx.array(np.array(cls.grid, dtype=np.float32).reshape(shape).astype(np.int8))


GRID_IQ1 = _grid(IQ1_S, (2048, 8))          # values in {-1, 0, 1}
GRID_IQ2 = _grid(IQ2_XXS, (256, 8))
GRID_IQ3 = _grid(IQ3_XXS, (256, 4))
KSIGNS = mx.array(np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).copy().reshape(128))
KVALUES_IQ4 = mx.array(np.array(IQ4_NL.kvalues, dtype=np.int8))
KVALUES_MXFP4 = mx.array(np.array(MXFP4.kvalues, dtype=np.int8))

_HALF_FROM_BYTES = '''
    uint16_t dbits = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    float d = (float)as_type<half>(dbits);
'''

_IQ1_S = """
    uint tid = thread_position_in_grid.x;
    uint block = tid / 8u;
    uint group = tid % 8u;
    const device uint8_t* p = blocks + block * 50u;
    device half* out_row = out + block * 256u + group * 32u;
""" + _HALF_FROM_BYTES + """
    const device uint8_t* qh = p + 34u;
    uint16_t q = (uint16_t)qh[2u * group] | ((uint16_t)qh[2u * group + 1u] << 8);
    float dl = d * (2.0f * (float)((q >> 12) & 7u) + 1.0f);
    float delta = (q & 0x8000u) ? -0.125f : 0.125f;
    const device uint8_t* qs = p + 2u + 4u * group;
    for (uint l = 0; l < 4u; ++l) {
        uint idx = (uint)qs[l] | (((uint)((q >> (3u * l)) & 7u)) << 8);
        const device int8_t* grid_row = grid + idx * 8u;
        for (uint j = 0; j < 8u; ++j) {
            out_row[l * 8u + j] = (half)(dl * ((float)grid_row[j] + delta));
        }
    }
"""

_IQ2_XXS = """
    uint tid = thread_position_in_grid.x;
    uint block = tid / 8u;
    uint group = tid % 8u;
    const device uint8_t* p = blocks + block * 66u;
    device half* out_row = out + block * 256u + group * 32u;
""" + _HALF_FROM_BYTES + """
    const device uint8_t* q = p + 2u + 8u * group;
    uint32_t lo = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16) | ((uint32_t)q[3] << 24);
    uint32_t hi = (uint32_t)q[4] | ((uint32_t)q[5] << 8) | ((uint32_t)q[6] << 16) | ((uint32_t)q[7] << 24);
    float db = d * (0.5f + (float)(hi >> 28)) * 0.25f;
    for (uint l = 0; l < 4u; ++l) {
        const device int8_t* grid_row = grid + (uint)((lo >> (8u * l)) & 0xFFu) * 8u;
        uint8_t sign_byte = ksigns[(hi >> (7u * l)) & 0x7Fu];
        for (uint j = 0; j < 8u; ++j) {
            float sign = ((sign_byte >> j) & 1u) ? -1.0f : 1.0f;
            out_row[l * 8u + j] = (half)(db * (float)grid_row[j] * sign);
        }
    }
"""

_IQ3_XXS = """
    uint block = thread_position_in_grid.x;
    const device uint8_t* p = blocks + block * 98u;
    device half* out_row = out + block * 256u;
""" + _HALF_FROM_BYTES + """
    const device uint8_t* qs = p + 2u;
    const device uint8_t* sc = p + 66u;
    for (uint g = 0; g < 8u; ++g) {
        const device uint8_t* s = sc + 4u * g;
        uint32_t word = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16) | ((uint32_t)s[3] << 24);
        float db = d * (0.5f + (float)(word >> 28)) * 0.5f;
        for (uint i = 0; i < 8u; ++i) {
            const device int8_t* grid_row = grid + (uint)qs[8u * g + i] * 4u;
            for (uint j = 0; j < 4u; ++j) {
                uint slot = i * 4u + j;
                uint8_t sign_byte = ksigns[(word >> (7u * (slot / 8u))) & 0x7Fu];
                float sign = ((sign_byte >> (slot % 8u)) & 1u) ? -1.0f : 1.0f;
                out_row[g * 32u + slot] = (half)(db * (float)grid_row[j] * sign);
            }
        }
    }
"""

_IQ4_NL = """
    uint block = thread_position_in_grid.x;
    const device uint8_t* p = blocks + block * 18u;
    device half* out_row = out + block * 32u;
""" + _HALF_FROM_BYTES + """
    for (uint i = 0; i < 16u; ++i) {
        uint8_t packed = p[2u + i];
        out_row[i] = (half)(d * (float)kvalues[packed & 0x0Fu]);
        out_row[16u + i] = (half)(d * (float)kvalues[packed >> 4]);
    }
"""

_MXFP4 = """
    uint block = thread_position_in_grid.x;
    const device uint8_t* p = blocks + block * 17u;
    device half* out_row = out + block * 32u;
    uint exponent = p[0];
    uint32_t bits = (exponent < 2u) ? (0x00200000u << exponent) : ((uint32_t)(exponent - 1u) << 23);
    float d = as_type<float>(bits);
    for (uint i = 0; i < 16u; ++i) {
        uint8_t packed = p[1u + i];
        out_row[i] = (half)(d * (float)kvalues[packed & 0x0Fu]);
        out_row[16u + i] = (half)(d * (float)kvalues[packed >> 4]);
    }
"""


def _make(name, source, extra_inputs):
    return mx.fast.metal_kernel(
        name=f'iq_{name.lower()}_dequant',
        input_names=['blocks'] + extra_inputs,
        output_names=['out'],
        source=source,
    )


KERNELS = {
    'IQ1_S': (_make('IQ1_S', _IQ1_S, ['grid']), ['grid'], GRID_IQ1, 8),
    'IQ2_XXS': (_make('IQ2_XXS', _IQ2_XXS, ['grid', 'ksigns']), ['grid', 'ksigns'], (GRID_IQ2, KSIGNS), 8),
    'IQ3_XXS': (_make('IQ3_XXS', _IQ3_XXS, ['grid', 'ksigns']), ['grid', 'ksigns'], (GRID_IQ3, KSIGNS), 1),
    'IQ4_NL': (_make('IQ4_NL', _IQ4_NL, ['kvalues']), ['kvalues'], KVALUES_IQ4, 1),
    'MXFP4': (_make('MXFP4', _MXFP4, ['kvalues']), ['kvalues'], KVALUES_MXFP4, 1),
}


def geometry(qtype):
    quant = GGMLQuantizationType[qtype] if isinstance(qtype, str) else qtype
    return GGML_QUANT_SIZES[quant]


def dequantize(blocks, qtype):
    """Decode (n, block_bytes) uint8 blocks to (n, block_elements) float16."""
    name = qtype if isinstance(qtype, str) else qtype.name
    kernel, _, tables, threads_per_block = KERNELS[name]
    elements, size = geometry(name)
    if blocks.ndim != 2 or blocks.shape[1] != size:
        raise ValueError(f'{name}: expected blocks of shape (n, {size}), got {tuple(blocks.shape)}')
    count = blocks.shape[0]
    tables = tables if isinstance(tables, tuple) else (tables,)
    out = kernel(
        inputs=[blocks, *tables],
        grid=(count * threads_per_block, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(count, elements)],
        output_dtypes=[mx.float16],
    )
    return out[0]
