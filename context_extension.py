"""Extend a checkpoint's usable context with YaRN rope scaling.

A model trained at 128K usually generalises further when its rotary embeddings
are rescaled, which is what YaRN does: it interpolates the low frequencies,
extrapolates the high ones, and applies an attention-temperature correction. The
pinned runtime already implements it (`mlx_lm.models.rope_utils.YarnRoPE`); what a
checkpoint needs is the ``rope_scaling`` block that turns it on, and a declared
maximum that matches.

This writes that block, keeping a copy of the original config so the change is
reversible, and refuses to overwrite a model that already declares different
scaling unless asked to.

Note on quality: YaRN restores *usable* length, not the accuracy of the original
window. Expect the untouched range to be unaffected and the extension to degrade
gently with distance from it; the factor is a trade, not a free multiplier.
"""
import json
from pathlib import Path
import shutil

# YaRN's own defaults for the correction range and temperature, as used by the
# Qwen and GPT-OSS releases.
DEFAULT_BETA_FAST = 32.0
DEFAULT_BETA_SLOW = 1.0


def yarn_block(factor, original, beta_fast=DEFAULT_BETA_FAST, beta_slow=DEFAULT_BETA_SLOW):
    return {
        'rope_type': 'yarn',
        'factor': float(factor),
        'original_max_position_embeddings': int(original),
        'beta_fast': float(beta_fast),
        'beta_slow': float(beta_slow),
        'truncate': False,
    }


def current_scaling(config):
    text = config.get('text_config', config)
    return text.get('rope_scaling') or config.get('rope_scaling')


def extend(path, factor, original=None, max_position=None, force=False, dry_run=False):
    """Write YaRN scaling into ``<path>/config.json``.

    ``original`` defaults to the checkpoint's declared maximum, which is the
    length its rotary embeddings were trained for. ``max_position`` defaults to
    ``original * factor``.
    """
    path = Path(path)
    config_path = path / 'config.json'
    config = json.loads(config_path.read_text())

    # A checkpoint may keep its text hyperparameters at the top level or under
    # `text_config`; both are written where they already exist, so whichever the
    # model class reads, it sees the same scaling.
    targets = [config]
    if isinstance(config.get('text_config'), dict):
        targets.append(config['text_config'])
    if not any('max_position_embeddings' in level for level in targets):
        raise ValueError(f'{path.name} declares no max_position_embeddings to extend')

    existing = current_scaling(config)
    if existing and not force:
        if existing.get('rope_type') == 'yarn' and float(existing.get('factor', 0)) == float(factor):
            return {'changed': False, 'reason': 'already extended by this factor', 'scaling': existing}
        if dry_run:
            # A dry run reports what would happen; refusing to describe it is not
            # useful, and the caller has changed nothing to undo.
            return {'changed': False, 'dry_run': True, 'blocked_by': existing,
                    'reason': 'the checkpoint already declares rope_scaling; a real run '
                              'would refuse without force=True'}
        raise ValueError(f'{path.name} already declares rope_scaling {existing}; '
                         'pass force=True to replace it')

    declared = next(int(level['max_position_embeddings']) for level in targets
                    if 'max_position_embeddings' in level)
    original = int(original or declared)
    if factor <= 1:
        raise ValueError(f'factor must be greater than 1, got {factor}')
    block = yarn_block(factor, original)
    new_max = int(max_position or original * factor)

    if dry_run:
        return {'changed': False, 'dry_run': True, 'scaling': block,
                'original_max': original, 'new_max': new_max}

    backup = config_path.with_suffix('.json.pre-yarn')
    if not backup.exists():
        shutil.copy2(config_path, backup)
    for level in targets:
        level['rope_scaling'] = block
        if 'max_position_embeddings' in level:
            level['max_position_embeddings'] = new_max
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    return {'changed': True, 'scaling': block, 'original_max': original,
            'new_max': new_max, 'backup': str(backup)}


def restore(path):
    """Put back the config saved by :func:`extend`."""
    config_path = Path(path) / 'config.json'
    backup = config_path.with_suffix('.json.pre-yarn')
    if not backup.exists():
        raise FileNotFoundError(f'no {backup.name} to restore in {config_path.parent}')
    shutil.copy2(backup, config_path)
    return {'restored': str(config_path), 'from': str(backup)}
