"""What each profile costs and how fast it runs, for the interactive selector.

Two columns matter to someone choosing a model: how much context this machine can
hold with it loaded, and roughly how fast it will generate.

**Context** is not a config field, it is a memory question, so it is computed the
same way the server computes it at request time - the same `ContextPolicy`, the
same available memory, the same device limit - with the checkpoint's weight bytes
subtracted because they will be resident once it is serving. That is why the
number differs from the model's native window, and why it moves when other
processes do.

**Speed** is reported from measurements taken on this machine where they exist,
and estimated otherwise. The estimate is a bandwidth model - decode reads a token's
worth of weights per step, so throughput is roughly effective bandwidth divided by
the bytes a token touches, plus a fixed per-step cost - calibrated against the
measured dense models. It is honest to about a third either way, which is enough
to choose with and not enough to quote.
"""
import json
from pathlib import Path

# Measured on this machine (M2 Ultra, 64 GiB) with the engine's own check
# scripts, 4-bit weights unless noted. `decode` is single-stream tokens per
# second, `prefill` is a warm 2k-token prompt.
MEASURED = {
    'k2-horizon': (46.6, 1153.0),
    'gemma4-26b-a4b': (191.7, None),
    'gemma4-31b': (22.6, 4752.0),
    'qwen3.8-27b': (67.6, None),
    'qwen3.6-35b-a3b': (77.5, None),
    'ornith-1.5-35b-a3b': (78.2, None),
    'nemotron-3.5-30b-a3b': (86.2, 1136.0),
    'gpt-oss-20b': (82.5, 1081.0),
    'muse-glimmer-30b': (26.7, 213.0),
    'glm-4.7-flash': (48.2, 829.0),
    'minicpm5-2b': (133.0, 1755.0),
    'spark-x2.5-4b': (55.3, 1300.0),
    'qwen3.5-9b': (70.9, 10817.0),
    'qwen3.5-4b': (94.8, None),
    'llama-3.2-3b': (150.0, None),
    'smollm3-3b': (115.2, 16153.0),
    'qwen3.8-flash': (11.9, None),
    'deepseek-v4-flash': (3.3, None),
}

# Calibrated on the dense profiles above: an M2 Ultra moves about this many bytes
# per second through the decoder, and each step costs this long in dispatch.
EFFECTIVE_BANDWIDTH = 420e9
STEP_OVERHEAD_SECONDS = 0.0045


def checkpoint_bytes(path):
    """Weight bytes on disk, from the shard sizes.

    The index's ``total_size`` is not usable here: some converters record the
    pre-quantization size, which for K2-Horizon overstates the 4-bit checkpoint by
    more than three times and would push the context estimate to zero.
    """
    path = Path(path)
    shards = sorted(path.glob('*.safetensors'))
    if shards:
        return sum(shard.stat().st_size for shard in shards)
    return None


def active_bytes(config):
    """Bytes a token touches: every weight for a dense model, some experts for a MoE."""
    text = config.get('text_config', config)
    experts = text.get('num_experts') or text.get('n_routed_experts') or 0
    if not experts:
        return None  # dense: the whole checkpoint is read every step
    top_k = text.get('num_experts_per_tok') or 1
    layers = text.get('num_hidden_layers') or 1
    expert_bytes = (config.get('_quantized_expert_bytes')
                    or (text.get('moe_intermediate_size') or 0) * 3
                    * text.get('hidden_size', 0) * 0.5 * layers)
    return expert_bytes * top_k / experts if expert_bytes else None


def decode_estimate(weight_bytes):
    if not weight_bytes:
        return None
    return 1.0 / (weight_bytes / EFFECTIVE_BANDWIDTH + STEP_OVERHEAD_SECONDS)


def profile_costs(profile, available_bytes, working_set_bytes, resident_bytes=0):
    """Context ceiling and speed for one profile on this machine, right now."""
    from runtime_support import ContextPolicy

    config = profile['config']
    path = Path(profile['path'])
    weights = checkpoint_bytes(path)
    costs = {'weights_gib': (weights / 2**30) if weights else None}

    try:
        policy = ContextPolicy(config)
        # The weights are not resident yet, but they will be: discount them from
        # what is available before asking how much context fits beside them.
        free_for_cache = max(0, available_bytes - (weights or 0))
        costs['max_context'] = policy.capacity(
            free_for_cache, resident_bytes, working_set_bytes + resident_bytes)
        costs['native_context'] = policy.native_limit
        costs['bytes_per_token'] = policy.bytes_per_token
    except Exception as exc:  # a family the policy does not model
        costs['max_context'] = None
        costs['reason'] = str(exc)[:80]

    measured = MEASURED.get(profile['alias'])
    if measured:
        costs['decode'], costs['prefill'], costs['source'] = measured[0], measured[1], 'measured'
    else:
        per_token = active_bytes(config) or weights
        costs['decode'] = decode_estimate(per_token)
        costs['prefill'] = None
        costs['source'] = 'estimated'
    return costs
