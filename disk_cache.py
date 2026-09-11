"""SSD-backed inactive prompt caches; accessed only by the generation thread."""
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
import uuid

import mlx.core as mx
from mlx_lm.models.cache import (save_prompt_cache, load_prompt_cache, trim_prompt_cache,
                                 can_trim_prompt_cache)


class DiskPromptCache:
    def __init__(self, directory, fingerprint, max_bytes=64 * 2**30, codec=None):
        self._save = codec.save if codec else save_prompt_cache
        self._load = codec.load if codec else load_prompt_cache
        self.directory = Path(directory) / fingerprint
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_bytes = max_bytes
        self.entries = {}
        self.hits = self.misses = 0
        self.last_error = None
        self.restore_budget = float('inf')
        for p in self.directory.glob('*.json'):
            try:
                entry = json.loads(p.read_text())
                tensor = p.with_suffix('.safetensors')
                if tensor.is_file():
                    entry['bytes'] = tensor.stat().st_size
                    self.entries[p.stem] = entry
            except (OSError, ValueError):
                logging.warning('Ignoring invalid cache index %s', p)
        self._evict(0)

    def __len__(self):
        return len(self.entries)

    @property
    def nbytes(self):
        return 0  # No inactive tensors retained in unified memory.

    @property
    def disk_bytes(self):
        return sum(e['bytes'] for e in self.entries.values())

    def stats_by_type(self):
        return {}

    def trim_to(self, **kwargs):
        pass  # RAM eviction is immediate on insert.

    def _remove(self, key):
        self.entries.pop(key, None)
        for suffix in ('.json', '.safetensors'):
            (self.directory / (key + suffix)).unlink(missing_ok=True)

    def _evict(self, needed):
        while self.entries and self.disk_bytes + needed > self.max_bytes:
            self._remove(min(self.entries, key=lambda k: self.entries[k]['used']))

    def insert_cache(self, model, tokens, prompt_cache, **kwargs):
        # The final sampled token has not necessarily been evaluated by the model.
        offsets = [c.offset for c in prompt_cache if hasattr(c, 'offset')]
        if not offsets or len(set(offsets)) != 1 or offsets[0] <= 0:
            logging.info('prompt cache: not saved, offsets %s',
                         sorted(set(offsets))[:4] if offsets else 'none')
            return
        # A cache whose offset reaches the end of the token list covers every
        # token in it - true of a prefill snapshot, which is taken before any
        # token is sampled. Only those may be matched on their final token.
        complete = len(tokens) <= offsets[0]
        tokens = tokens[:offsets[0]]
        if len(tokens) != offsets[0]:
            logging.info('prompt cache: not saved, cache offset %d beyond %d tracked tokens',
                         offsets[0], len(tokens))
            return
        needed = sum(c.nbytes for c in prompt_cache) + len(tokens) * 12 + 65536
        if needed > self.max_bytes:
            return
        key = hashlib.sha256(json.dumps([model, tokens]).encode()).hexdigest()
        temp = self.directory / (uuid.uuid4().hex + '.tmp.safetensors')
        try:
            self._remove(key)
            self._evict(needed)
            if shutil.disk_usage(self.directory).free < needed + 2 * 2**30:
                self.last_error = 'Insufficient free SSD space; skipped cache save'
                return
            self._save(str(temp), prompt_cache)
            os.chmod(temp, 0o600)
            dest = self.directory / (key + '.safetensors')
            temp.replace(dest)
            entry = {'model': list(model), 'tokens': tokens, 'used': time.time(),
                     'trimmable': can_trim_prompt_cache(prompt_cache),
                     'complete': complete, 'bytes': dest.stat().st_size}
            index = self.directory / (key + '.json')
            index_temp = index.with_suffix('.tmp')
            index_temp.write_text(json.dumps(entry))
            os.chmod(index_temp, 0o600)
            index_temp.replace(index)
            self.entries[key] = entry
            self.last_error = None
            self._evict(0)
        except Exception as exc:
            self.last_error = str(exc)
            logging.exception('SSD cache save failed; completion is still valid')
        finally:
            temp.unlink(missing_ok=True)

    def fetch_nearest_cache(self, model, tokens):
        best_key, best_prefix = None, 0
        for key, entry in self.entries.items():
            if entry['model'] != list(model):
                continue
            if entry['bytes'] > self.restore_budget:
                continue
            # A complete entry covers its own last token, a sampled-tail entry
            # does not: the model never evaluated the final sampled token.
            query = tokens if entry.get('complete') else tokens[:-1]
            common = 0
            for a, b in zip(entry['tokens'], query):
                if a != b:
                    break
                common += 1
            # Every caller needs at least one token left to process - the batch
            # generator raises on an empty prompt - so a match may never consume
            # the whole query. A longer query still reuses the whole entry, which
            # is the case that matters for a conversation.
            prefix = min(common, len(tokens) - 1)
            if not entry.get('trimmable', True) and prefix != len(entry['tokens']):
                continue
            if prefix > best_prefix:
                best_key, best_prefix = key, prefix
        if best_key is None:
            self.misses += 1
            return None, tokens
        try:
            cache = self._load(str(self.directory / (best_key + '.safetensors')))
            if any(c.offset != len(self.entries[best_key]['tokens']) for c in cache
                   if hasattr(c, 'offset')):
                raise ValueError('SSD cache offset does not match token index')
            offset = len(self.entries[best_key]['tokens'])
            if can_trim_prompt_cache(cache):
                trim_prompt_cache(cache, offset - best_prefix)
            elif best_prefix != offset:
                raise ValueError('Cannot rewind recurrent state')
            # Compact BEFORE evaluation: don't materialize a huge unused suffix.
            for c in cache:
                if self._load is load_prompt_cache and hasattr(c, 'keys_and_values'):
                    c.keys, c.values = c.keys_and_values()
            mx.eval([c.state for c in cache])
            self.entries[best_key]['used'] = time.time()
            self.hits += 1
            return cache, tokens[best_prefix:]
        except Exception as exc:
            self.last_error = str(exc)
            logging.exception('SSD cache restore failed; recomputing prompt')
            self._remove(best_key)
            self.misses += 1
            return None, tokens
