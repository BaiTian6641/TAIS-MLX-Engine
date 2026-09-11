"""Text adapters for pinned Flash architectures; no remote model code execution."""
import json
from pathlib import Path


def _gemma4_classes(config):
    """Text-only view of a Gemma 4 checkpoint, which ships vision and audio towers.

    The checkpoint prefixes its language weights with ``language_model.``; the
    pinned ``mlx_lm`` text implementation expects them unprefixed, so the
    adapter strips the prefix and lets the upstream ``sanitize`` (which fuses
    the expert gate/up pairs) run on the remaining tensors.
    """
    from mlx_lm.models import gemma4_text

    class TextArgs:
        @staticmethod
        def from_dict(params):
            params = dict(params.get('text_config', params))
            params.pop('quantization', None)
            params.pop('quantization_config', None)
            return gemma4_text.ModelArgs.from_dict(params)

    class TextModel(gemma4_text.Model):
        def sanitize(self, weights):
            text = {key[len('language_model.'):]: value for key, value in weights.items()
                    if key.startswith('language_model.')}
            if not text:
                raise ValueError('Gemma 4 checkpoint has no language_model.* tensors')
            return super().sanitize(text)

    return TextModel, TextArgs


def model_classes(config):
    if config['model_type'] == 'deepseek_v4':
        from vendor.deepseek_v4.model import Model, ModelArgs
        return Model, ModelArgs
    if config['model_type'] == 'gemma4':
        return _gemma4_classes(config)
    if config['model_type'] != 'qwen4_exp':
        from mlx_lm.utils import _get_classes
        return _get_classes(config)
    import mlx.nn as nn
    from vendor.flash_vlm.models.qwen4_exp.config import ModelConfig
    from vendor.flash_vlm.models.qwen4_exp.language import LanguageModel

    class TextArgs:
        @staticmethod
        def from_dict(params):
            # MLX-LM applies checkpoint quantization after constructing the
            # model. Avoid VLM's multimodal namespace-conversion registration.
            params = dict(params)
            params.pop('quantization', None)
            params.pop('quantization_config', None)
            return ModelConfig.from_dict(params)

    class TextModel(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.language_model = LanguageModel(args.text_config, args)
            self.args = args

        def __call__(self, inputs, cache=None):
            return self.language_model(inputs, cache=cache).logits

        @property
        def layers(self):
            return self.language_model.layers

        def make_cache(self):
            return self.language_model.make_cache()

        def sanitize(self, weights):
            # Supported MLX export already has canonical names and zero-centered
            # norm weights. Do not fold another +1 into Qwen4ExpRMSNorm.
            return {k: v for k, v in weights.items() if k.startswith('language_model.')}

    return TextModel, TextArgs


def register():
    """Install explicit architecture selection into the existing MLX-LM loader."""
    from mlx_lm import utils
    if getattr(utils.load_model, '_flash_adapter', False):
        return
    original = utils.load_model

    def load_model(path, *args, **kwargs):
        config = json.loads((Path(path) / 'config.json').read_text())
        if config.get('model_type') in ('qwen4_exp', 'deepseek_v4', 'gemma4'):
            kwargs['get_model_classes'] = model_classes
        return original(path, *args, **kwargs)

    load_model._flash_adapter = True
    utils.load_model = load_model


def install_embedding_offload(model, directory, max_bytes=128 * 2**20):
    """Keep Qwen's large quantized n-gram table on SSD, cache selected rows."""
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten, tree_unflatten
    from expert_cache import ExpertCache, TensorIndex
    cache = ExpertCache(TensorIndex(directory), max_bytes, max_entries=8192)

    class DiskEmbedding(nn.Module):
        def __init__(self, name, original):
            super().__init__()
            self._name = name
            self._ranges = {s: cache.index.resolve(name + '.' + s)
                            for s in ('weight', 'scales', 'biases') if s in original}
            self._group, self._bits, self._mode = original.group_size, original.bits, original.mode

        def __call__(self, indices):
            mx.eval(indices)
            ids = np.array(indices)
            unique, inverse = np.unique(ids, return_inverse=True)
            rows = [cache.fetch(self._name, int(i), self._ranges) for i in unique]
            bank = {s: mx.stack([row[s] for row in rows]) for s in self._ranges}
            values = mx.dequantize(bank['weight'], bank['scales'], bank.get('biases'),
                                  group_size=self._group, bits=self._bits, mode=self._mode)
            result = values[mx.array(inverse.astype(np.int32))].reshape(*ids.shape, -1)
            mx.eval(result)
            return result

    replacements = []
    for name, module in tree_flatten(model.leaf_modules(), is_leaf=lambda x: isinstance(x, nn.Module)):
        if '.ple.ple_embedding.ngram_embedding.shards.' in name:
            if not isinstance(module, nn.QuantizedEmbedding):
                raise ValueError('Flash PLE streaming requires a quantized MLX embedding')
            replacements.append((name, DiskEmbedding(name, module)))
    if not replacements:
        raise ValueError('No supported Flash n-gram embedding shards found')
    model.update_modules(tree_unflatten(replacements))
    return cache
