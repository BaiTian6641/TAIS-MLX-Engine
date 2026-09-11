"""K2 deployment with adaptive context, SSD prompt caching, and live telemetry."""
import copy
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import gc

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'hf-cache'))
import mlx.core as mx
from mlx_lm import server, tokenizer_utils
from disk_cache import DiskPromptCache
import output_channels
from runtime_support import ContextPolicy, memory_info
from telemetry import Telemetry
from model_profiles import parse_options, resolve_profile, fingerprint, uses_quantized_kv

OPTIONS = parse_options() if __name__ == '__main__' else parse_options([])
PROFILE = resolve_profile(OPTIONS)
MODEL_PATH = PROFILE['path']
MODEL_ALIAS = PROFILE['alias']
CONFIG = PROFILE['config']
POLICY = ContextPolicy(CONFIG)
METRICS = Telemetry()
METRICS.extra['native_limit'] = POLICY.native_limit
METRICS.extra['model'] = MODEL_ALIAS
FINGERPRINT = fingerprint(MODEL_PATH)
EXPERT_CACHE = None
EMBEDDING_CACHE = None
# id(prompt cache) -> (prompt tokens, cache as it was when prefill finished)
SNAPSHOTS = {}


def offload_stats():
    stats = EXPERT_CACHE.stats() if EXPERT_CACHE is not None else {}
    if EMBEDDING_CACHE is not None:
        stats.update({key.replace('expert_', 'embedding_'): value
                      for key, value in EMBEDDING_CACHE.stats().items()})
    return stats


METRICS.extra_source = offload_stats
if PROFILE.get('flash'):
    from flash_models import register
    register()
original_mlx_load = server.load


def load_checkpoint(*args, **kwargs):
    global EXPERT_CACHE, EMBEDDING_CACHE
    if OPTIONS.expert_cache_gib is None:
        return original_mlx_load(*args, **kwargs)
    from expert_cache import install_expert_offload
    kwargs['lazy'] = True
    model, tokenizer = original_mlx_load(*args, **kwargs)
    if CONFIG['model_type'] == 'qwen4_exp':
        from flash_models import install_embedding_offload
        EMBEDDING_CACHE = install_embedding_offload(model, MODEL_PATH)
    EXPERT_CACHE = install_expert_offload(model, MODEL_PATH,
                                         int(OPTIONS.expert_cache_gib * 2**30))
    gc.collect()
    mx.clear_cache()
    mx.eval(model.parameters())
    with METRICS.lock:
        METRICS.extra.update(EXPERT_CACHE.stats())
    return model, tokenizer


server.load = load_checkpoint

original_infer_thinking = tokenizer_utils._infer_thinking


def infer_thinking(tokenizer):
    start, end = '<ifm|think>', '</ifm|think>'
    vocab = tokenizer.get_vocab()
    if start in vocab and end in vocab:
        return start, end, (vocab[start],), (vocab[end],)
    return original_infer_thinking(tokenizer)


tokenizer_utils._infer_thinking = infer_thinking
original_tokenize = server.ResponseGenerator._tokenize


def bounded_tokenize(self, tokenizer, request, args):
    result = original_tokenize(self, tokenizer, request, args)
    mx.clear_cache()
    info = mx.device_info()
    if EXPERT_CACHE is not None:
        EXPERT_CACHE.resize(POLICY.expert_budget(
            memory_info()['available'], mx.get_active_memory(),
            info['max_recommended_working_set_size'], EXPERT_CACHE.bytes,
            OPTIONS.expert_cache_gib * 2**30, len(result[0]), args.max_tokens))
        mx.clear_cache()
    pending_experts = 0 if EXPERT_CACHE is None else max(0, EXPERT_CACHE.max_bytes - EXPERT_CACHE.bytes)
    if EMBEDDING_CACHE is not None:
        pending_experts += max(0, EMBEDDING_CACHE.max_bytes - EMBEDDING_CACHE.bytes)
    limit = POLICY.capacity(memory_info()['available'], mx.get_active_memory(),
                            info['max_recommended_working_set_size'], pending_experts)
    args.max_tokens = POLICY.admit(len(result[0]), args.max_tokens, limit)
    self.prompt_cache.restore_budget = limit * POLICY.bytes_per_token + POLICY.fixed_state_bytes
    with METRICS.lock:
        METRICS.extra['context_limit'] = limit
    METRICS.update(request._monitor_id, context_limit=limit,
                   output_allowance=args.max_tokens, prompt_total=len(result[0]))
    return result


server.ResponseGenerator._tokenize = bounded_tokenize
original_provider_init = server.ModelProvider.__init__


def provider_init(self, args):
    original_provider_init(self, args)
    self._model_map[MODEL_ALIAS] = args.model


server.ModelProvider.__init__ = provider_init
original_load = server.ModelProvider.load


def verify_cache_policy(model):
    """Check the serving flags against what this model's caches can do.

    The flags are chosen from the profile before the model exists, so this is
    where the guess meets the model. Both failure modes are real: a rotating
    cache takes the server down when `--kv-bits` is passed, and a cache without
    merge turns batching into a crash on the first multi-request batch.
    """
    from mlx_lm.models.cache import make_prompt_cache

    from runtime_support import cache_capabilities

    # Not every model class defines make_cache; make_prompt_cache falls back to a
    # plain KV cache for the ones that do not, which is what the server itself does.
    caps = cache_capabilities(make_prompt_cache(model))
    METRICS.extra.update(capabilities=caps)
    kv_requested = uses_quantized_kv(CONFIG) and OPTIONS.kv_bits > 0
    if kv_requested and not caps['quantize_safe']:
        raise SystemExit(f'{MODEL_ALIAS} has KV caches that cannot be quantized '
                         '(the runtime raises NotImplementedError for this cache type). '
                         'Run with --kv-bits 0.')
    if kv_requested and not caps['quantizes_any']:
        logging.warning('--kv-bits %d has no effect on %s: no cache in this model can be '
                        'quantized. Its context cost is the recurrent state, not a KV cache.',
                        OPTIONS.kv_bits, MODEL_ALIAS)
    if (OPTIONS.decode_concurrency > 1 or OPTIONS.prompt_concurrency > 1) and not caps['mergeable']:
        raise SystemExit(f'{MODEL_ALIAS} has KV caches that cannot merge, so the pinned runtime '
                         'cannot batch them. Run without the concurrency flags.')
    return caps


def load_local(self, model_path, adapter_path=None, draft_model_path=None):
    # Policy and SSD cache identity describe this deployed model only.
    if model_path not in ('default_model', MODEL_ALIAS, str(MODEL_PATH)) or adapter_path is not None or draft_model_path not in (None, 'default_model'):
        raise ValueError(f'This process serves only {MODEL_ALIAS}, without adapters or draft models')
    result = original_load(self, model_path, adapter_path, draft_model_path)
    verify_cache_policy(self.model)
    if PROFILE.get('flash') or OPTIONS.mtp_draft is not None:
        # Streaming adapters and the MTP path both need the single-request
        # generator: the batch generator builds its own cache types, which the
        # vendored linear-attention layers reject.
        self.is_batchable = False
    return result


server.ModelProvider.load = load_local
def install_prefill_snapshots():
    """Store each prompt cache as it was when its prefill finished.

    ``insert_cache`` is called once generation is done, so what it stores is the
    prompt *plus* the tokens the model went on to produce. The next chat turn
    re-renders the conversation and does not continue those tokens, so it is not
    an extension of that entry and - on a hybrid model, whose caches cannot be
    trimmed back to the common prefix - it finds nothing to reuse at all.

    It is an extension of the pre-generation cache. Snapshotting at the moment
    prefill completes (the progress callback's last call) gives an entry keyed by
    the prompt alone, which every later turn extends.
    """
    if getattr(server.stream_generate, '_snapshot_hook', False):
        return
    snapshots = SNAPSHOTS
    upstream = server.stream_generate

    def stream_generate(*args, **kwargs):
        cache = kwargs.get('prompt_cache')
        prompt = kwargs.get('prompt')
        progress = kwargs.get('prompt_progress_callback')

        def snapshot_progress(done, total):
            if cache is not None and prompt is not None and done >= total:
                # One prefill runs at a time in this thread, so a single slot is
                # enough - and unlike an id-keyed table it cannot be matched by
                # a later cache that happens to reuse a freed object's address.
                snapshots['pending'] = (list(prompt), copy.deepcopy(cache))
            progress and progress(done, total)

        kwargs['prompt_progress_callback'] = snapshot_progress
        yield from upstream(*args, **kwargs)

    stream_generate._snapshot_hook = True
    server.stream_generate = stream_generate


def snapshot_aware_insert(prompt_cache):
    """Wrap a prompt cache so the prefill snapshot is stored alongside the result."""
    original_insert = prompt_cache.insert_cache

    def insert_cache(model, tokens, prompt_cache_arg, **kwargs):
        original_insert(model, tokens, prompt_cache_arg, **kwargs)
        entry = SNAPSHOTS.pop('pending', None)
        if entry is None:
            return
        prompt_tokens, snapshot = entry
        # Only claim the snapshot if it really describes this prompt; the slot is
        # shared by every request this process serves.
        if len(prompt_tokens) <= len(tokens) and list(tokens[:len(prompt_tokens)]) == prompt_tokens:
            original_insert(model, prompt_tokens, snapshot, **kwargs)
            logging.info('prompt cache: stored a prefill snapshot for %d tokens',
                         len(prompt_tokens))

    prompt_cache.insert_cache = insert_cache
    return prompt_cache


original_init = server.ResponseGenerator.__init__


def generator_init(self, provider, prompt_cache):
    install_prefill_snapshots()
    if OPTIONS.mtp_draft is not None:
        # MTP keeps prompt caches in memory: the disk cache restores mlx_lm's
        # own cache classes, and the vendored linear-attention layers reject
        # them (`ArraysCache` has no `update_window`).
        original_init(self, provider, snapshot_aware_insert(prompt_cache))
        return
    codec = None
    if PROFILE.get('flash'):
        import flash_cache as codec
    disk = DiskPromptCache(ROOT / 'kv-cache', FINGERPRINT,
                          int(float(os.environ.get('K2_SSD_CACHE_GIB', '64')) * 2**30), codec=codec)
    with METRICS.lock:
        METRICS.extra.update(disk_entries=len(disk), disk_bytes=disk.disk_bytes)
    original_init(self, provider, snapshot_aware_insert(disk))


server.ResponseGenerator.__init__ = generator_init
original_single = server.ResponseGenerator._serve_single
SPECULATIVE = {'model': None, 'drafter': None}


def install_mtp(drafter_path):
    """Route model loading and single-request generation through the MTP loop.

    The vendored target is loaded once, in place of the pinned runtime's class,
    because only it captures the hidden state and shared KV the head drafts
    from. Its prompt cache is not batchable, so the server already sends every
    request down the single-request path this hook replaces.
    """
    import mtp_speculation
    from mlx_lm.utils import load_tokenizer
    upstream_load = server.load

    def load_speculative(model_path, *args, **kwargs):
        if str(model_path) != str(MODEL_PATH):
            return upstream_load(model_path, *args, **kwargs)
        model, _ = mtp_speculation.load_mtp_target(MODEL_PATH)
        tokenizer = load_tokenizer(
            str(MODEL_PATH),
            tokenizer_config_extra={'trust_remote_code': kwargs.get('trust_remote_code', False)},
        )
        SPECULATIVE['model'] = model
        SPECULATIVE['drafter'] = mtp_speculation.load_drafter(drafter_path, model)
        return model, tokenizer

    server.load = load_speculative

    def serve_single(self, request, stream):
        if SPECULATIVE['drafter'] is None:
            return original_single(self, request, stream)
        rqueue, request, args = request

        def progress(tokens_processed, tokens_total):
            rqueue.put((tokens_processed, tokens_total))

        try:
            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            prompt, _, _, initial_state = self._tokenize(tokenizer, request, args)
            stop_sequences, text_sm = self._make_state_machine(
                self.model_provider.model_key, tokenizer, args.stop_words)
            ctx = server.GenerationContext(
                has_thinking=tokenizer.has_thinking,
                has_tool_calling=tokenizer.has_tool_calling,
                tool_parser=tokenizer.tool_parser,
                text_sm=text_sm,
                initial_state=initial_state,
                prompt=prompt,
            )
            rqueue.put(ctx)
            if args.seed is not None:
                mx.random.seed(args.seed)
            sampler = server._make_sampler(args, tokenizer)
            self._log_cache_stats()
            cache, rest = self.prompt_cache.fetch_nearest_cache(
                self.model_provider.model_key, prompt)
            logging.info('MTP: prompt cache %s, %d of %d prompt tokens reusable',
                         'hit' if cache is not None else 'miss',
                         len(prompt) - len(rest), len(prompt))
            if cache is not None and len(rest):
                # Continue from the cached prefix: prefill only the new tokens
                # and let the round loop draft from there. Cache types survive
                # reuse because MTP mode holds them in memory rather than
                # restoring mlx_lm's own classes from disk.
                prompt_ids = mx.array([rest])
                ctx.prompt_cache_count = len(prompt) - len(rest)
            else:
                # Nothing to reuse (or the whole prompt is already cached, in
                # which case there is no prefill left to capture the hidden
                # state the head drafts from): start a fresh cache and prefill
                # the whole prompt.
                cache = model.make_cache()
                prompt_ids = mx.array([prompt])
                ctx.prompt_cache_count = 0
            cache_key = prompt[:]
            stop_matcher = stop_sequences.matcher()
            greedy = getattr(args, 'temp', 1.0) == 0
            eos_ids = set(getattr(tokenizer, 'eos_token_ids', None) or [])
            if not eos_ids and getattr(tokenizer, 'eos_token_id', None) is not None:
                eos_ids = {tokenizer.eos_token_id}
            emitted, text = 0, ''
            stream_state = mtp_speculation.stream_speculative(
                model, SPECULATIVE['drafter'], cache, prompt_ids,
                args.max_tokens, sampler, greedy=greedy, eos_token_ids=eos_ids,
            )
            for token, _ in stream_state:
                emitted += 1
                whole = tokenizer.decode([token], skip_special_tokens=True)
                text += whole
                rqueue.put(server.Response(whole, token, 0.0, None, None))
                cache_key.append(token)
                if stop_matcher.advance(token) or ctx._should_stop:
                    break
            rounds = getattr(stream_state, 'rounds', 0)
            logging.info('MTP: emitted %d tokens over %d rounds (%.2f accepted per round)',
                         emitted, rounds, emitted / max(rounds, 1))
            rqueue.put(None)
            self.prompt_cache.insert_cache(self.model_provider.model_key, cache_key, cache)
        except Exception as exc:
            rqueue.put(exc)

    server.ResponseGenerator._serve_single = serve_single


original_generate = server.ResponseGenerator.generate


def generate(self, request, generation_args, progress_callback=None):
    key = METRICS.submit(request)
    try:
        ctx, responses = original_generate(self, request, generation_args, progress_callback)
    except Exception:
        METRICS.finish(key, error=True)
        raise
    tokenizer = getattr(self.model_provider, 'tokenizer', None)
    if tokenizer is None or not output_channels.needs_normalising(tokenizer):
        return ctx, responses
    # GPT-OSS and Muse Glimmer answer inside a channel envelope; rewriting it
    # here means the runtime's own state machine routes the reply and the
    # reasoning to the right fields, without touching the generation loop.
    return ctx, _normalised(responses, output_channels.ChannelNormaliser(
        split_reasoning=bool(getattr(tokenizer, 'has_thinking', False))))


def _normalised(responses, normaliser):
    for response in responses:
        text = normaliser.feed(response.text)
        if text != response.text:
            response = dataclasses.replace(response, text=text)
        yield response


server.ResponseGenerator.generate = generate
original_single = server.ResponseGenerator._serve_single


class ObservedQueue:
    def __init__(self, queue, key):
        self.queue, self.key = queue, key
        self.error = False

    def put(self, value):
        if isinstance(value, Exception):
            self.error = True
        elif isinstance(value, server.Response):
            METRICS.token(self.key)
        elif isinstance(value, tuple):
            METRICS.update(self.key, phase='prefill', prompt_done=value[0], prompt_total=value[1])
        elif value is None:
            METRICS.update(self.key, phase='saving KV', decode_end=time.time())
        return self.queue.put(value)


def serve_single(self, item, stream):
    queue, request, args = item
    key = request._monitor_id
    observed = ObservedQueue(queue, key)
    METRICS.update(key, phase='preparing', started=time.time())
    try:
        return original_single(self, (observed, request, args), stream)
    finally:
        mx.clear_cache()
        disk = self.prompt_cache
        with METRICS.lock:
            METRICS.extra.update(disk_entries=len(disk), disk_bytes=disk.disk_bytes,
                                 cache_hits=disk.hits, cache_misses=disk.misses,
                                 cache_error=disk.last_error)
            if EXPERT_CACHE is not None:
                METRICS.extra.update(EXPERT_CACHE.stats())
        METRICS.finish(key, observed.error)


server.ResponseGenerator._serve_single = serve_single


def models_request(self):
    self._set_completion_headers(200)
    self.end_headers()
    self.wfile.write(json.dumps({'object': 'list', 'data': [{
        'id': MODEL_ALIAS, 'object': 'model', 'owned_by': 'local',
        'context_length': POLICY.native_limit,
        'context_policy': 'memory-aware; output allowance clipped to fit',
    }]}).encode())


server.APIHandler.handle_models_request = models_request

if __name__ == '__main__':
    METRICS.start_sampler(ROOT / 'metrics.json', mx)
    kv_quantized = uses_quantized_kv(CONFIG) and OPTIONS.kv_bits > 0
    if (OPTIONS.decode_concurrency > 1 or OPTIONS.prompt_concurrency > 1) and kv_quantized:
        raise SystemExit('Batching needs unquantized KV: the pinned runtime cannot merge quantized '
                         'caches. Retry with --kv-bits 0, or run a profile in UNQUANTIZED_KV.')
    sys.argv = [
        sys.argv[0], '--model', str(MODEL_PATH),
        '--host', OPTIONS.host, '--port', str(OPTIONS.port), '--trust-remote-code',
        *( ['--kv-bits', str(OPTIONS.kv_bits), '--kv-group-size', '64',
            '--quantized-kv-start', str(OPTIONS.quantized_kv_start)]
           if kv_quantized else []),
        '--prefill-step-size', str(OPTIONS.prefill_step_size),
        '--prompt-cache-size', str(OPTIONS.prompt_cache_size),
        '--decode-concurrency', str(OPTIONS.decode_concurrency),
        '--prompt-concurrency', str(OPTIONS.prompt_concurrency),
        '--max-tokens', '32768', '--temp', '1.0', '--top-p', '0.95',
        '--chat-template-args', json.dumps(PROFILE['chat_template_args']),
    ]
    if OPTIONS.mtp_draft is not None:
        if OPTIONS.decode_concurrency > 1 or OPTIONS.prompt_concurrency > 1:
            raise SystemExit('MTP drafting is single-request; drop the concurrency flags.')
        install_mtp(Path(OPTIONS.mtp_draft))
    server.main()
