"""Spark2.5 architecture for MLX.

Ported from the checkpoint's own `modeling_spark.py` (Spark2_5ForCausalLM),
because the pinned runtime has no `spark2_5`. Shape and behaviour come from that
reference rather than from an assumption about which published architecture it
resembles: it is dense where the nearest candidate is mixture-of-experts, and its
attention is gated per head.

What makes it distinctive:

* **Per-layer-type rotary embeddings.** Sliding layers rotate all 256 head dims at
  theta 10,000; full layers rotate a quarter of them at theta 5,000,000. Two rope
  instances, one per type, chosen by the layer.
* **A fused QKV projection** (`q_k_v_proj`) rather than three projections.
* **A per-head output gate** (`g_proj`) with a sigmoid activation, applied to each
  head's attention output before the output projection.
* **Three sliding layers for every full layer**, window 512, in a repeating
  pattern of four.
"""
from dataclasses import dataclass, field
import json
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class ModelArgs:
    model_type: str = 'spark2_5'
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    intermediate_size: int = 10240
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 131072
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = True
    layer_types: List[str] = field(default_factory=list)
    sliding_window: Optional[int] = 512
    headwise_attn_output_gate: bool = True
    gate_attn_act_mode: str = 'sigmoid'
    rope_parameters: dict = field(default_factory=dict)
    spark_rope_parameters: dict = field(default_factory=dict)
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self):
        # The published checkpoint keeps per-layer-type rope settings under
        # `rope_parameters`, a nested shape that transformers' generic RoPE
        # validator rejects when the tokenizer is loaded. `spark_rope_parameters`
        # is the same content under a name nothing else reads, and either is
        # accepted here.
        self.rope_parameters = self.rope_parameters or self.spark_rope_parameters
        if not self.layer_types:
            self.layer_types = ['full_attention'] * self.num_hidden_layers
        self.rope_parameters = self.rope_parameters or {
            'full_attention': {'partial_rotary_factor': 0.25, 'rope_theta': 5000000},
            'sliding_attention': {'partial_rotary_factor': 1.0, 'rope_theta': 10000},
        }

    @classmethod
    def from_dict(cls, params: dict) -> 'ModelArgs':
        params = dict(params.get('text_config', params))
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in params.items() if k in fields})

    def rope_for(self, layer_type: str) -> dict:
        return self.rope_parameters.get(layer_type) or self.rope_parameters['full_attention']


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x):
        # The reference is gelu-gated; mlx's gelu is the tanh approximation, which
        # is what the reference's ACT2FN resolves to.
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.gate_mode = args.gate_attn_act_mode

        self.q_k_v_proj = nn.Linear(args.hidden_size, self.q_dim + 2 * self.kv_dim,
                                    bias=args.attention_bias)
        self.g_proj = nn.Linear(args.hidden_size, self.num_heads,
                                bias=args.attention_bias) if args.headwise_attn_output_gate else None
        self.out_proj = nn.Linear(self.q_dim, args.hidden_size, bias=args.attention_bias)

        rope = args.rope_for(self.layer_type)
        self.rope = initialize_rope(
            dims=int(self.head_dim * rope.get('partial_rotary_factor', 1.0)),
            base=rope.get('rope_theta', 10000.0),
            traditional=False,
            scaling_config=rope.get('rope_scaling'),
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask: Any, cache: Any) -> mx.array:
        batch, length, _ = x.shape
        qkv = self.q_k_v_proj(x)
        q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)

        q = q.reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        if self.g_proj is not None:
            gate = self.g_proj(x).astype(mx.float32)
            gate = mx.sigmoid(gate) if self.gate_mode == 'sigmoid' else nn.silu(gate)
            # One gate per (head, position), matching the reference's
            # (batch, heads, seq, 1) broadcast over the head dimension.
            out = out * gate.transpose(0, 2, 1)[..., None].astype(out.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, self.q_dim)
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Any, cache: Any) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, index) for index in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None,
                 capture_layer_ids: Optional[List[int]] = None, **kwargs):
        """Return logits, like the reference's ForCausalLM wrapper."""
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        # One mask per layer type: a rolling window for sliding layers, causal
        # for full ones. Both are derived from the caches the server supplied.
        masks = {}
        for layer_type in set(self.args.layer_types):
            example = next((c for layer, c in zip(self.layers, cache)
                            if layer.self_attn.layer_type == layer_type), None)
            window = self.args.sliding_window if layer_type == 'sliding_attention' else None
            masks[layer_type] = create_attention_mask(h, example, window_size=window)

        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, masks[layer.self_attn.layer_type], layer_cache)
        h = self.norm(h)
        if self.args.tie_word_embeddings:
            return self.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.self_attn.layer_type == 'sliding_attention' and self.args.sliding_window:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
            else:
                caches.append(KVCache())
        return caches


def sanitize(weights: dict) -> dict:
    """Map the published checkpoint onto this module's parameter names.

    The reference names the embedding ``model.embedding`` where the runtime's
    convention is ``embed_tokens``, and prefixes everything with ``model.``.
    There is no ``lm_head`` - the checkpoint ties it - and no rotary frequency
    table, because the two rope instances are built here instead.
    """
    out = {}
    for key, value in weights.items():
        if 'rotary_emb' in key:
            continue
        if key == 'model.embedding.weight':
            out['embed_tokens.weight'] = value
            continue
        if key.startswith('model.'):
            key = key[len('model.'):]
        out[key] = value
    return out


def load_config(path) -> ModelArgs:
    from pathlib import Path
    return ModelArgs.from_dict(json.loads((Path(path) / 'config.json').read_text()))
