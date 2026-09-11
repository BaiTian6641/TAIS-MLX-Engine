"""Host sampling and conservative, dynamically sized context admission."""
import ctypes
import os
from pathlib import Path
import re
import subprocess

GIB = 2**30


def command(args):
    return subprocess.check_output(args, text=True, timeout=3)


def memory_info():
    total = int(command(['sysctl', '-n', 'hw.memsize']))
    raw = command(['vm_stat'])
    page = int(re.search(r'page size of (\d+) bytes', raw)[1])
    fields = {k: int(v) for k, v in re.findall(r'([^\n:]+):\s+(\d+)\.', raw)}
    available = sum(fields.get(k, 0) for k in
                    ('Pages free', 'Pages inactive', 'Pages speculative')) * page
    return {'total': total, 'available': available, 'used': total - available}


def cpu_ticks():
    lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    data = (ctypes.c_uint32 * 4)()
    count = ctypes.c_uint32(4)
    if lib.host_statistics(lib.mach_host_self(), 3, data, ctypes.byref(count)):
        raise OSError('host_statistics failed')
    return list(data)


def gpu_usage():
    raw = command(['ioreg', '-r', '-c', 'AGXAccelerator', '-l'])
    match = re.search(r'"Device Utilization %"=(\d+)', raw)
    return int(match[1]) if match else None


class ContextPolicy:
    def __init__(self, config):
        family = config.get('model_type')
        config = config.get('text_config', config)
        self.native_limit = config['max_position_embeddings']
        count = config['num_hidden_layers']
        layer_types = config.get('layer_types')
        if layer_types is None and config.get('full_attention_interval'):
            every = config['full_attention_interval']
            layer_types = ['full_attention' if (i+1) % every == 0 else 'linear_attention'
                           for i in range(count)]
        if layer_types is not None and len(layer_types) != count:
            raise ValueError('layer_types does not match num_hidden_layers')
        full_layers = count if layer_types is None else layer_types.count('full_attention')
        linear_layers = count - full_layers
        # Two tensors, 4-bit values plus bf16 scale/bias per group of 64.
        kv_heads = config.get('num_key_value_heads') or config['num_attention_heads']
        head_dim = config.get('head_dim') or config['hidden_size']//config['num_attention_heads']
        element_bytes = config.get('_runtime_cache_element_bytes', 2)
        self.bytes_per_token = int(2 * full_layers * kv_heads * head_dim * (0.5 + 2*element_bytes / 64))
        self.fixed_state_bytes = 0
        if linear_layers and 'linear_num_value_heads' in config:
            value_heads = config['linear_num_value_heads']
            value_dim = config['linear_value_head_dim']
            key_heads = config['linear_num_key_heads']
            key_dim = config['linear_key_head_dim']
            conv_dim = 2*key_heads*key_dim + value_heads*value_dim
            self.fixed_state_bytes = linear_layers * (value_heads*key_dim*value_dim*4 +
                conv_dim*(config['linear_conv_kernel_dim']-1)*element_bytes)
        if family == 'qwen4_exp':
            # QSA keeps unquantized raw indexer keys, positions and compressed
            # index blocks in addition to quantized attention KV.
            index_dim = config['indexer_head_dim']
            self.bytes_per_token += full_layers * (index_dim * 4 + 8 +
                index_dim * 4 / config['indexer_compress_ratio'])
            self.fixed_state_bytes += len(config.get('ple_layer_ids', [])) * (
                config.get('ple_embed_dim', config['hidden_size']) *
                config.get('ple_conv_kernel_size', 4) * 4 + config.get('ngram_size', 3) * 8)
        elif family == 'gemma4':
            # KV stays unquantized because the runtime cannot quantize
            # sliding-window caches. Sliding layers are bounded by their
            # window; only the full-attention layers grow with the context.
            element = config.get('_runtime_cache_element_bytes', 2)
            per_layer = 2 * kv_heads * head_dim * element
            self.bytes_per_token = full_layers * per_layer
            self.fixed_state_bytes = (count - full_layers) * (config.get('sliding_window') or 0) * per_layer
        elif family == 'deepseek_v4':
            # Native compressed caches stay unquantized. Use float32 bounds
            # for growing pools and both remainder/overlap buffers.
            ratios = config['compress_ratios']
            if len(ratios) != count or any(r not in (0, 4, 128) for r in ratios):
                raise ValueError('Unsupported DeepSeek V4 compression layout')
            head = config['head_dim']
            index = config['index_head_dim']
            self.bytes_per_token = sum((head + (index if r == 4 else 0)) * 4 / r
                                      for r in ratios if r)
            self.fixed_state_bytes = count * (config['sliding_window'] + 256) * head * 8
            self.fixed_state_bytes += sum((head + (index if r == 4 else 0)) *
                (r * 32 + 256 * 4) for r in ratios if r)
        if self.bytes_per_token <= 0:
            raise ValueError('A full-attention layer is required for context admission')
        self.reserve = float(os.environ.get('K2_RESERVE_GIB', '6')) * GIB
        self.workspace = float(os.environ.get('K2_WORKSPACE_GIB', '4')) * GIB
        if self.reserve < 0 or self.workspace < 0:
            raise ValueError('Memory reserves must be nonnegative')

    def capacity(self, available, active, recommended, pending_expert_bytes=0):
        budget = max(0, min(available - self.reserve, recommended - active) - self.workspace
                     - self.fixed_state_bytes - max(0, pending_expert_bytes))
        # Account for allocation rounding and temporary cache growth.
        return min(self.native_limit, max(0, int(budget / (self.bytes_per_token * 1.1)) - 256))

    def admit(self, prompt, output, limit):
        if output < 1:
            raise ValueError('max_tokens must be positive')
        if prompt >= limit:
            raise ValueError(f'Prompt has {prompt} tokens; current memory-aware context limit '
                             f'is {limit}. Shorten the prompt or free memory.')
        return min(output, limit - prompt)

    def expert_budget(self, available, active, recommended, current, configured, prompt, output):
        """Let long contexts displace hot experts; cold experts remain on SSD."""
        desired = min(self.native_limit, prompt + output)
        context_bytes = (desired + 256) * self.bytes_per_token * 1.1 + self.fixed_state_bytes
        space = min(available - self.reserve, recommended - active) + current - self.workspace
        return max(0, min(int(configured), int(space - context_bytes)))


def cache_capabilities(caches):
    """What the runtime can actually do with this model's KV caches.

    Both answers mirror what the pinned runtime itself does. ``--kv-bits`` is
    applied only to caches that offer ``to_quantized``, so a hybrid model
    quantizes its attention caches and leaves its recurrent state alone - but a
    cache that offers the method and raises ``NotImplementedError`` (the
    rotating cache does) takes the server down, which is worth knowing before
    the flag is accepted rather than after. Batching is allowed only when every
    cache can merge, and only without quantization, which the caller enforces
    separately.

    A hand-kept list of architecture names cannot answer either question
    correctly once new profiles arrive; empty caches answer both exactly.
    """
    quantizes_any, quantize_safe, mergeable = False, True, bool(caches)
    for cache in caches:
        if not hasattr(cache, 'merge'):
            mergeable = False
        if not hasattr(cache, 'to_quantized'):
            continue
        try:
            cache.to_quantized(group_size=64, bits=8)
        except NotImplementedError:
            quantize_safe = False
        except Exception:
            # A shape or dtype complaint is not a statement about capability.
            pass
        else:
            quantizes_any = True
    return {'quantize_safe': quantize_safe, 'quantizes_any': quantizes_any,
            'mergeable': mergeable}
