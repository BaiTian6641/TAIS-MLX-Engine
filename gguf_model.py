"""Load the pinned IQ1 GGUFs into the vendored MLX runtimes.

Resident tensors are decoded once at load time. Routed experts stay in their
GGUF blocks: ``GGUFExpertStore`` keeps a byte-budgeted hot set of those
undecoded blocks in RAM (through ``expert_cache.ExpertCache``) and only the
experts a step selects are decoded, so the hot set holds roughly an order of
magnitude more experts than a decoded cache would.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from expert_cache import ExpertCache
from iq_metal import KERNELS as METAL_DECODERS
from iq_metal import dequantize as dequantize_metal
from iq_quants import dequantize as dequantize_ops


def decode_blocks(blocks, qtype):
    """Decode packed blocks to float16, on the GPU kernel when one exists."""
    if qtype in METAL_DECODERS:
        return dequantize_metal(blocks, qtype)
    return dequantize_ops(blocks, qtype)

DEEPSEEK_ROOT = {
    'model.embed_tokens.weight': 'token_embd.weight',
    'model.norm.weight': 'output_norm.weight',
    'lm_head.weight': 'output.weight',
    'model.hc_head.fn': 'output_hc_fn.weight',
    'model.hc_head.base': 'output_hc_base.weight',
    'model.hc_head.scale': 'output_hc_scale.weight',
}

# GGUF literals follow the llama.cpp deepseek4 converter; the pinned gguf
# package has no name map for this architecture.
DEEPSEEK_LAYER = {
    'attn.wq_a.weight': 'attn_q_a.weight',
    'attn.wq_b.weight': 'attn_q_b.weight',
    'attn.q_norm.weight': 'attn_q_a_norm.weight',
    'attn.wkv.weight': 'attn_kv.weight',
    'attn.kv_norm.weight': 'attn_kv_a_norm.weight',
    'attn.wo_a.weight': 'attn_output_a.weight',
    'attn.wo_b.weight': 'attn_output_b.weight',
    'attn.attn_sink': 'attn_sinks.weight',
    'attn.compressor.ape': 'attn_compressor_ape.weight',
    'attn.compressor.wkv.weight': 'attn_compressor_kv.weight',
    'attn.compressor.wgate.weight': 'attn_compressor_gate.weight',
    'attn.compressor.norm.weight': 'attn_compressor_norm.weight',
    'attn.indexer.wq_b.weight': 'indexer.attn_q_b.weight',
    'attn.indexer.weights_proj.weight': 'indexer.proj.weight',
    'attn.indexer.compressor.ape': 'indexer_compressor_ape.weight',
    'attn.indexer.compressor.wkv.weight': 'indexer_compressor_kv.weight',
    'attn.indexer.compressor.wgate.weight': 'indexer_compressor_gate.weight',
    'attn.indexer.compressor.norm.weight': 'indexer_compressor_norm.weight',
    'attn_hc.fn': 'hc_attn_fn.weight',
    'attn_hc.base': 'hc_attn_base.weight',
    'attn_hc.scale': 'hc_attn_scale.weight',
    'ffn_hc.fn': 'hc_ffn_fn.weight',
    'ffn_hc.base': 'hc_ffn_base.weight',
    'ffn_hc.scale': 'hc_ffn_scale.weight',
    'attn_norm.weight': 'attn_norm.weight',
    'ffn_norm.weight': 'ffn_norm.weight',
    'ffn.gate.weight': 'ffn_gate_inp.weight',
    'ffn.shared_experts.gate_proj.weight': 'ffn_gate_shexp.weight',
    'ffn.shared_experts.down_proj.weight': 'ffn_down_shexp.weight',
    'ffn.shared_experts.up_proj.weight': 'ffn_up_shexp.weight',
}

DEEPSEEK_EXPERTS = {
    'gate': 'ffn_gate_exps.weight',
    'up': 'ffn_up_exps.weight',
    'down': 'ffn_down_exps.weight',
}

QWEN_ROOT = {
    'model.embed_tokens.weight': 'token_embd.weight',
    'lm_head.weight': 'output.weight',
    'model.hyper_connection_mixer.hc_norm.weight': 'output_hc_norm.weight',
    'model.hyper_connection_mixer.input_mix_weight_down.weight': 'output_hc_down.weight',
    'model.hyper_connection_mixer.input_mix_weight_up.weight': 'output_hc_up.weight',
}

QWEN_LAYER = {
    'attn_hyper_connection.hc_norm.weight': 'hc_attn_norm.weight',
    'attn_hyper_connection.input_mix_weight_down.weight': 'hc_attn_down.weight',
    'attn_hyper_connection.input_mix_weight_up.weight': 'hc_attn_up.weight',
    'attn_hyper_connection.block_inject_weight.weight': 'hc_attn_inject.weight',
    'mlp_hyper_connection.hc_norm.weight': 'hc_ffn_norm.weight',
    'mlp_hyper_connection.input_mix_weight_down.weight': 'hc_ffn_down.weight',
    'mlp_hyper_connection.input_mix_weight_up.weight': 'hc_ffn_up.weight',
    'mlp_hyper_connection.block_inject_weight.weight': 'hc_ffn_inject.weight',
    'mlp.gate.weight': 'ffn_gate_inp.weight',
    'mlp.shared_expert.gate_proj.weight': 'ffn_gate_shexp.weight',
    'mlp.shared_expert.up_proj.weight': 'ffn_up_shexp.weight',
    'mlp.shared_expert.down_proj.weight': 'ffn_down_shexp.weight',
    'mlp.shared_expert_gate.weight': 'ffn_gate_inp_shexp.weight',
}

# Linear-attention layers, whose value heads are stored tiled and whose A_log
# is stored negated and exponentiated.
QWEN_LINEAR_LAYER = {
    'linear_attn.in_proj_qkv.weight': 'attn_qkv.weight',
    'linear_attn.in_proj_z.weight': 'attn_gate.weight',
    'linear_attn.in_proj_a.weight': 'ssm_alpha.weight',
    'linear_attn.in_proj_b.weight': 'ssm_beta.weight',
    'linear_attn.conv1d.weight': 'ssm_conv1d.weight',
    'linear_attn.A_log': 'ssm_a',
    'linear_attn.dt_bias': 'ssm_dt.bias',
    'linear_attn.norm.weight': 'ssm_norm.weight',
    'linear_attn.out_proj.weight': 'ssm_out.weight',
}

QWEN_SPARSE_LAYER = {
    'self_attn.q_proj.weight': 'attn_q.weight',
    'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight',
    'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight',
    'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'self_attn.indexer.q_layernorm.weight': 'indexer.q_norm.weight',
    'self_attn.indexer.k_layernorm.weight': 'indexer.k_norm.weight',
}

QWEN_PLE_LAYER = {
    'ple.key_proj.weight': 'ple_key.weight',
    'ple.value_proj.weight': 'ple_value.weight',
    'ple.norm_key.weight': 'ple_norm_key.weight',
    'ple.norm_query.weight': 'ple_norm_query.weight',
    'ple.norm_conv.weight': 'ple_norm_conv.weight',
    'ple.conv1d.weight': 'ple_conv1d.weight',
}

QWEN_EXPERTS = {
    'gate': 'ffn_gate_exps.weight',
    'up': 'ffn_up_exps.weight',
    'down': 'ffn_down_exps.weight',
}

# Parameters the runtime keeps in float32 (see Model.cast_predicate).
DEEPSEEK_FLOAT32 = ('attn_sink', 'e_score_correction_bias', '.attn_hc.', '.ffn_hc.', '.hc_head.')


def deepseek_param_map(config):
    """Map every resident MLX parameter name to its GGUF tensor name."""
    layers = config['num_hidden_layers']
    ratios = list(config['compress_ratios'])[:layers]
    if len(ratios) != layers:
        raise ValueError(f'compress_ratios covers {len(ratios)} of {layers} layers')
    hash_layers = config.get('num_hash_layers', 3)
    mapping = dict(DEEPSEEK_ROOT)
    for index, ratio in enumerate(ratios):
        prefix = f'model.layers.{index}.'
        for suffix, tensor in DEEPSEEK_LAYER.items():
            if suffix.startswith('attn.compressor.') and ratio == 0:
                continue
            if suffix.startswith('attn.indexer.') and ratio != 4:
                continue
            mapping[prefix + suffix] = f'blk.{index}.{tensor}'
        if index < hash_layers:
            mapping[prefix + 'ffn.gate.tid2eid'] = f'blk.{index}.ffn_gate_tid2eid.weight'
        else:
            mapping[prefix + 'ffn.gate.e_score_correction_bias'] = f'blk.{index}.exp_probs_b.bias'
    return mapping


class GGUFExpertStore:
    """Byte-budgeted hot set of undecoded GGUF expert blocks.

    Residency follows both traffic and router mass: every selection adds the
    router score to the cache's priority for that expert, so eviction drops the
    least-used, least-weighted experts first. Cold experts are never pruned;
    they are simply read from disk when a router does select them.
    """

    def __init__(self, index, max_bytes, saliency_weight=1.0):
        if max_bytes < 0:
            raise ValueError('Expert store budget must be nonnegative')
        self.index = index
        self.cache = ExpertCache(self, max_bytes)
        self.saliency_weight = float(saliency_weight)
        self.saliency = {}
        self._owners = []          # keeps read buffers alive until evaluation

    def read_expert(self, entry, expert):
        """Cache-facing read: packed blocks for one entry, no decode.

        An entry is either a stacked expert tensor name (the expert index picks
        a slice) or a ``('rows', name)`` pair (the index picks a single row of a
        2-D tensor, used for the n-gram table).
        """
        if isinstance(entry, tuple):
            _, name = entry
            return self.index.read_packed_rows(name, expert, expert + 1, keep=self._owners)
        return self.index.read_packed_expert(entry, expert, keep=self._owners)

    def fetch(self, projection, expert, entry):
        return self.cache.fetch(projection, expert, entry)

    def fetch_many(self, projection, experts, entry):
        """Fetch several experts, materializing all reads in one synchronization.

        Reads are handed back to the cache unevaluated; the buffers behind them
        are kept alive here until the single evaluation returns.
        """
        self._owners = []
        try:
            rows = [self.cache.fetch(projection, int(expert), entry, evaluate=False) for expert in experts]
            mx.eval([array for row in rows for array in row.values()])
        finally:
            self._owners = []
        return rows

    def record(self, projection, experts, scores):
        """Weight residency by router mass, not just selection count."""
        frequency = self.cache.frequency
        recency = self.cache.last_used
        mass = self.saliency.setdefault(projection, {})
        for expert, score in zip(experts.tolist(), scores.tolist()):
            key = (projection, int(expert))
            value = self.saliency_weight * float(score)
            mass[key[1]] = mass.get(key[1], 0.0) + value
            frequency[key] = frequency.get(key, 0) + value
            recency[key] = self.cache.clock      # these keys were just routed to

    def resize(self, max_bytes):
        self.cache.resize(max_bytes)

    @property
    def max_bytes(self):
        return self.cache.max_bytes

    def stats(self):
        stats = self.cache.stats()
        stats['saliency_tracked'] = sum(len(m) for m in self.saliency.values())
        return stats


class StreamingFusedSwitchGLU(nn.Module):
    """SwiGLU experts fetched from GGUF blocks instead of held as parameters.

    Holds no parameters. A layer's selected experts are fetched once, decoded
    into a bank, and applied with batched gathers, so a layer costs a handful of
    operations instead of one round trip per expert. The bank is capped so a
    long prompt cannot materialize every expert at once; the default cap is
    deliberately small because the transient bank competes with the hot set for
    unified memory (measured: capping a 24 GiB hot set's banks at 12 experts
    cut a warm 128-token prefill from 34.9 s to 16.5 s).

    ``activation`` receives the decoded gate and up projections and returns the
    hidden state; ``entry`` names the three GGUF tensors of the layer.
    """

    def __init__(self, store, entry, projection, shapes, qtypes, activation,
                 hidden_size, hidden_dims, max_bank_experts=12, fuse_gate_up=True):
        super().__init__()
        self.entry = dict(entry)
        self.projection = projection
        self.shape = dict(shapes)
        self.qtype = dict(qtypes)
        self.store = store
        self.activation = activation
        self.hidden_size = hidden_size
        self.hidden_dims = hidden_dims
        self.max_bank_experts = max(1, int(max_bank_experts))
        # Fusing needs identical row geometry *and* the same block format, since
        # the concatenated blocks are decoded by one kernel launch.
        self.fuse_gate_up = (fuse_gate_up and shapes.get('gate') == shapes.get('up')
                             and qtypes.get('gate') == qtypes.get('up'))

    @classmethod
    def for_layer(cls, store, layer, names, hidden_size, hidden_dims, activation,
                  max_bank_experts=12, fuse_gate_up=True):
        entry = {part: f'blk.{layer}.{name}' for part, name in names.items()}
        shapes = {part: tuple(reversed(store.index.tensor(name).dims))[1:]
                  for part, name in entry.items()}
        qtypes = {part: store.index.tensor(name).qtype for part, name in entry.items()}
        return cls(store, entry, f'blk.{layer}.experts', shapes, qtypes, activation,
                   hidden_size, hidden_dims, max_bank_experts, fuse_gate_up)

    def weights(self, expert):
        """Decoded (out, in) matrices for one expert, from the hot set or disk."""
        packed = self.store.fetch(self.projection, expert, self.entry)
        return {part: decode_blocks(packed[part], self.qtype[part]).reshape(self.shape[part])
                for part in self.entry}

    def _bank(self, experts):
        """(U, in, out) matrices for a group of experts, laid out for gather_mm.

        The packed blocks of every expert in the group are concatenated and
        decoded in a single kernel launch, which keeps the dispatch count per
        layer at two (gate+up fused, then down) regardless of how many experts
        the batch selects. Gate and up share an input dimension, so their rows
        concatenate into one bank and one gather.
        """
        packed = self.store.fetch_many(self.projection, experts, self.entry)
        bank = {}
        parts = list(self.entry)
        fuse = self.fuse_gate_up and 'gate' in self.entry and 'up' in self.entry
        if fuse:
            parts = ['gate_up' if part in ('gate', 'up') else part for part in parts]
            seen = []
            for part in parts:
                if part not in seen:
                    seen.append(part)
            parts = seen
        for part in parts:
            if part == 'gate_up':
                rows = []
                for row in packed:
                    rows.append(row['gate'])
                    rows.append(row['up'])
                decoded = decode_blocks(mx.concatenate(rows, axis=0), self.qtype['gate'])
                out_dim = self.shape['gate'][0]
                bank[part] = decoded.reshape(len(experts), 2 * out_dim, -1).swapaxes(-1, -2)
            else:
                rows = packed[0][part] if len(packed) == 1 else mx.concatenate([row[part] for row in packed], axis=0)
                decoded = decode_blocks(rows, self.qtype[part])
                bank[part] = decoded.reshape(len(experts), *self.shape[part]).swapaxes(-1, -2)
        return bank

    def _batches(self, ids):
        """Selection rows grouped into batches of at most max_bank_experts experts."""
        unique = np.unique(ids)
        if len(unique) <= self.max_bank_experts:
            return [(np.arange(len(ids)), unique)], None
        order = np.argsort(ids, kind='stable')
        ordered = ids[order]
        batches, start, seen = [], 0, set()
        for position, expert in enumerate(ordered):
            if len(seen) >= self.max_bank_experts and expert not in seen:
                batches.append((order[start:position], np.unique(ordered[start:position])))
                start, seen = position, set()
            seen.add(int(expert))
        batches.append((order[start:], np.unique(ordered[start:])))
        return batches, np.argsort(order)

    def __call__(self, x, indices, scores=None):
        hidden = self.hidden_size
        top_k = indices.shape[-1]
        # Each token appears once per selected expert, matching indices' order.
        flat = mx.broadcast_to(x[:, :, None, :], (*x.shape[:2], top_k, hidden)).reshape(-1, hidden)
        mx.eval(indices)
        ids = np.asarray(indices).reshape(-1)
        if scores is not None:
            self.store.record(self.projection, ids, np.asarray(scores).reshape(-1))

        batches, inverse = self._batches(ids)
        outputs = []
        for rows, experts in batches:
            bank = self._bank(experts)
            chosen = mx.take(flat, mx.array(rows.astype(np.uint32)), axis=0)
            slots = np.searchsorted(experts, ids[rows]).astype(np.uint32).reshape(1, -1, 1)
            if 'gate_up' in bank:
                gate_up = mx.gather_mm(chosen.reshape(1, -1, 1, 1, hidden), bank['gate_up'],
                                       rhs_indices=mx.array(slots)).reshape(-1, 2 * self.hidden_dims)
                gate, up = mx.split(gate_up, [self.hidden_dims], axis=-1)
            else:
                gate = mx.gather_mm(chosen.reshape(1, -1, 1, 1, hidden), bank['gate'],
                                    rhs_indices=mx.array(slots)).reshape(-1, self.hidden_dims)
                up = mx.gather_mm(chosen.reshape(1, -1, 1, 1, hidden), bank['up'],
                                  rhs_indices=mx.array(slots)).reshape(-1, self.hidden_dims)
            hidden_state = self.activation(gate, up)
            outputs.append(mx.gather_mm(hidden_state.reshape(1, -1, 1, 1, self.hidden_dims), bank['down'],
                                        rhs_indices=mx.array(slots)).reshape(-1, hidden))
        result = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
        if inverse is not None:
            result = mx.take(result, mx.array(inverse.astype(np.uint32)), axis=0)
        return result.reshape(*indices.shape, hidden)


def _install_scored_moe():
    """Let the vendor MoE forward router scores to the streaming expert module.

    The vendor ``__call__`` is mirrored without its distributed branches, which
    this single-process loader never enables; everything else is identical.
    """
    from vendor.deepseek_v4 import model as deepseek_model

    if getattr(deepseek_model.DeepseekV4MoE, '_gguf_scores', False):
        return
    vendor_moe = deepseek_model.DeepseekV4MoE

    class ScoredMoE(vendor_moe):
        _gguf_scores = True

        def __call__(self, x, input_ids):
            inds, scores = self.gate(x, input_ids)
            y = self.switch_mlp(x, inds, scores)
            y = (y * scores[..., None].astype(y.dtype)).sum(-2)
            return y + self.shared_experts(x)

    ScoredMoE.__name__ = vendor_moe.__name__
    deepseek_model.DeepseekV4MoE = ScoredMoE


def _check_shapes(model, weights):
    """MLX assigns broadcastable arrays silently, so verify every parameter."""
    expected = dict(tree_flatten(model.parameters()))
    for name, array in weights:
        if name not in expected:
            raise ValueError(f'{name} is not a parameter of this model')
        if tuple(expected[name].shape) != tuple(array.shape):
            raise ValueError(f'{name}: GGUF produced {tuple(array.shape)}, '
                             f'model expects {tuple(expected[name].shape)}')


def load_deepseek(index, config, expert_cache_bytes=0, dtype=mx.float16, saliency_weight=1.0):
    """Build the vendored DeepSeek-V4 model with GGUF-resident weights."""
    from vendor.deepseek_v4.model import Model, ModelArgs

    _install_scored_moe()
    args = ModelArgs.from_dict(config)
    model = Model(args)
    store = GGUFExpertStore(index, expert_cache_bytes, saliency_weight=saliency_weight)
    from vendor.deepseek_v4.model import _limited_swiglu

    def activation(gate, up, limit=args.swiglu_limit):
        return _limited_swiglu(gate, up, limit)

    for layer_index, layer in enumerate(model.model.layers):
        layer.ffn.switch_mlp = StreamingFusedSwitchGLU.for_layer(
            store, layer_index, DEEPSEEK_EXPERTS, args.hidden_size, args.moe_intermediate_size,
            activation)

    weights = []
    for name, tensor in deepseek_param_map(config).items():
        if tensor not in index:
            raise ValueError(f'GGUF tensor {tensor} is missing, needed by {name}')
        entry = index.tensor(tensor)
        if entry.qtype in ('I32', 'I16', 'I64', 'U32'):
            array = index.read(tensor, dtype=None)          # router tables stay integer
        else:
            keep_f32 = any(marker in name for marker in DEEPSEEK_FLOAT32)
            array = index.read(tensor, dtype=mx.float32 if keep_f32 else dtype)
        if name.endswith('.attn.wo_a.weight'):
            array = array.reshape(args.o_groups, args.o_lora_rank, -1)
        weights.append((name, array))
    _check_shapes(model, weights)
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model, store


# --- Qwen3.8-Flash-Next (qwen4exp) -------------------------------------------------

QWEN_NORM_PLUS_ONE = ('norm.weight', 'q_norm.weight', 'k_norm.weight', 'q_layernorm.weight',
                      'k_layernorm.weight', 'norm_key.weight', 'norm_query.weight',
                      'norm_conv.weight')
QWEN_NO_OFFSET_NORM = ('linear_attn.norm.weight',)


def _untile_v_heads(array, dim, key_heads, value_per_key, head_dim):
    """Invert the converter's grouped-to-tiled value-head reorder.

    The converter maps HF index ``k*N + n`` to GGUF index ``n*K + k``, so the
    inverse reshapes with the value-group axis first, transposes the two axes
    and flattens again. The operation is *not* an involution: the forward
    direction uses the ``(K, N)`` nesting, the inverse uses ``(N, K)``.
    """
    shape = array.shape
    if dim < 0:
        dim += len(shape)
    middle = [value_per_key, key_heads, head_dim]
    reshaped = array.reshape(*shape[:dim], *middle, *shape[dim + 1:])
    return reshaped.swapaxes(dim, dim + 1).reshape(shape)


class GGUFNGramTable(nn.Module):
    """Row-addressable n-gram table backed by the packed GGUF tensor.

    The vendor PLE module computes which rows it needs; only the lookup is
    replaced, so a 26.8 GiB table never has to be decoded into memory.
    """

    def __init__(self, index, name, row_width, max_bytes=512 * 2**20, max_entries=8192):
        super().__init__()
        self.name = name
        self.row_width = row_width
        self.qtype = index.tensor(name).qtype
        self.store = GGUFExpertStore(index, 0, saliency_weight=0.0)
        self.store.cache = ExpertCache(self.store, max_bytes, max_entries=max_entries)
        self.entry = {name: ('rows', name)}

    def __call__(self, ids):
        mx.eval(ids)
        flat = np.asarray(ids).reshape(-1)
        unique, inverse = np.unique(flat, return_inverse=True)
        rows = self.store.fetch_many(self.name, unique, self.entry)
        packed = mx.concatenate([row[self.name] for row in rows], axis=0)
        decoded = decode_blocks(packed, self.qtype).reshape(len(unique), self.row_width)
        values = mx.take(decoded, mx.array(inverse.reshape(-1).astype(np.uint32)), axis=0)
        return values.reshape(*ids.shape, self.row_width)

    def stats(self):
        return self.store.stats()


def qwen4exp_param_map(config):
    """Map every resident MLX parameter name to its GGUF tensor name(s)."""
    text = config.get('text_config', config)
    layers = text['num_hidden_layers']
    interval = text.get('full_attention_interval', 4)
    ple_indices = {int(i) - 1 for i in text.get('ple_layer_ids', [])}
    mapping = {}
    for key, tensor in QWEN_ROOT.items():
        mapping[f'language_model.{key}'] = tensor
    for index in range(layers):
        prefix = f'language_model.model.layers.{index}.'
        for suffix, tensor in QWEN_LAYER.items():
            mapping[prefix + suffix] = f'blk.{index}.{tensor}'
        if (index + 1) % interval == 0:
            for suffix, tensor in QWEN_SPARSE_LAYER.items():
                mapping[prefix + suffix] = f'blk.{index}.{tensor}'
            mapping[prefix + 'self_attn.indexer.index_qk_proj.weight'] = (
                f'blk.{index}.indexer.q_proj.weight', f'blk.{index}.indexer.k_proj.weight')
        else:
            for suffix, tensor in QWEN_LINEAR_LAYER.items():
                mapping[prefix + suffix] = f'blk.{index}.{tensor}'
        if index in ple_indices:
            for suffix, tensor in QWEN_PLE_LAYER.items():
                mapping[prefix + suffix] = f'blk.{index}.{tensor}'
    return mapping


def load_qwen4exp(index, config, expert_cache_bytes=0, dtype=mx.float16, saliency_weight=1.0):
    """Build the vendored Qwen4Exp model with GGUF-resident weights."""
    from mlx_lm.models.activations import swiglu

    from flash_models import model_classes

    model_cls, args_cls = model_classes(config)
    args = args_cls.from_dict(config)
    model = model_cls(args)
    inner = model.language_model.model
    text = config.get('text_config', config)
    store = GGUFExpertStore(index, expert_cache_bytes, saliency_weight=saliency_weight)

    for layer_index, layer in enumerate(inner.layers):
        layer.mlp.switch_mlp = StreamingFusedSwitchGLU.for_layer(
            store, layer_index, QWEN_EXPERTS, text['hidden_size'],
            text['moe_intermediate_size'], swiglu)
        if 'ple' in layer:
            ngram_heads = (text['ngram_size'] - 1) * text['heads_per_ngram']
            row_width = text['ple_embed_dim'] // ngram_heads
            layer.ple.ple_embedding.ngram_embedding = GGUFNGramTable(
                index, 'per_layer_token_embd.weight', row_width)

    heads = dict(
        key_heads=text['linear_num_key_heads'],
        value_per_key=text['linear_num_value_heads'] // text['linear_num_key_heads'],
        value_dim=text['linear_value_head_dim'],
        key_dim=text['linear_key_head_dim'],
    )
    weights = []
    for name, tensor in qwen4exp_param_map(config).items():
        parts = tensor if isinstance(tensor, tuple) else (tensor,)
        for part in parts:
            if part not in index:
                raise ValueError(f'GGUF tensor {part} is missing, needed by {name}')
        if len(parts) == 2:
            array = mx.concatenate([index.read(part) for part in parts], axis=0)
        else:
            array = index.read(parts[0])
        array = _qwen_transform(name, array, args, heads)
        weights.append((name, array))
    _check_shapes(model, weights)
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())
    # The vendor computes a few parameters from the config; anything else left
    # unloaded would silently stay at its random initialisation.
    unloaded = sorted(set(dict(tree_flatten(model.parameters()))) - {name for name, _ in weights})
    computed = ('ple_embedding.layer_multipliers', 'ple_embedding.ngram_heads_offsets',
                'ple_embedding.ngram_heads_vocab_sizes')
    unexpected = [name for name in unloaded if not name.endswith(computed)]
    if unexpected:
        raise ValueError(f'{len(unexpected)} parameters were not loaded, e.g. {unexpected[:4]}')
    return model, store


def _qwen_transform(name, array, args, heads):
    """Apply the GGUF-to-MLX transforms the converter applied in reverse."""
    text = args.text_config
    if name.endswith('mlp.shared_expert_gate.weight'):
        return array.reshape(1, -1)          # stored 1-D, consumed as a 1-row Linear
    if name.endswith('.linear_attn.A_log'):
        array = mx.log(-array.astype(mx.float32)).astype(array.dtype)
    if name.endswith('ple.conv1d.weight'):
        return array.reshape(*array.shape, 1)
    if 'linear_attn' in name:
        untile = lambda a, dim, dim_size: _untile_v_heads(a, dim, heads['key_heads'],
                                                          heads['value_per_key'], dim_size)
        if name.endswith('in_proj_qkv.weight'):
            split = heads['key_heads'] * heads['key_dim']
            q, k, v = array[:split], array[split:2 * split], array[2 * split:]
            return mx.concatenate([q, k, untile(v, 0, heads['value_dim'])], axis=0)
        if name.endswith(('in_proj_z.weight', 'attn_gate.weight')):
            return untile(array, 0, heads['value_dim'])
        if name.endswith(('in_proj_a.weight', 'in_proj_b.weight')):
            return untile(array, 0, 1)
        if name.endswith('out_proj.weight'):
            return untile(array, 1, heads['value_dim'])
        if name.endswith('linear_attn.conv1d.weight'):
            split = heads['key_heads'] * heads['key_dim'] * 2
            qk, v = array[:split], array[split:]
            return mx.concatenate([qk, untile(v, 0, heads['value_dim'])], axis=0).reshape(*array.shape, 1)
        if name.endswith('A_log') or name.endswith('dt_bias'):
            return untile(array, 0, 1).reshape(array.shape)
    if name.endswith(QWEN_NORM_PLUS_ONE) and not name.endswith(QWEN_NO_OFFSET_NORM):
        # The converter stores zero-centred norms as w + 1; the runtime wants w.
        return (array.astype(mx.float32) - 1.0).astype(array.dtype)
    return array
