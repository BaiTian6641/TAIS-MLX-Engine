"""K2 deployment with adaptive context, SSD prompt caching, and live telemetry."""
import copy
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
import time
import gc

ROOT = Path(__file__).resolve().parent
import hf_env  # noqa: E402  (before anything that imports huggingface_hub)

hf_env.configure(root=ROOT)
import mlx.core as mx
from mlx_lm import server, tokenizer_utils
from disk_cache import DiskPromptCache
import output_channels
from runtime_support import ContextPolicy, memory_info
from telemetry import Telemetry
from model_profiles import parse_options, resolve_profile, fingerprint, uses_quantized_kv
from vision_engine import VISION_PROFILES

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


def offload_stats():
    stats = EXPERT_CACHE.stats() if EXPERT_CACHE is not None else {}
    if EMBEDDING_CACHE is not None:
        stats.update({key.replace('expert_', 'embedding_'): value
                      for key, value in EMBEDDING_CACHE.stats().items()})
    return stats


METRICS.extra_source = offload_stats
# Registering the vendored architectures is unconditional and idempotent: the
# Flash profiles need it for their streaming adapters, and spark2_5 - which the
# pinned runtime does not ship at all - needs it regardless of how it is served.
from flash_models import register as register_architectures

register_architectures()
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
    global SERVED_PROFILE_NAME
    SERVED_PROFILE_NAME = args.model
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
    if PROFILE.get('flash') or OPTIONS.mtp_draft is not None or PROFILE['alias'] in VISION_PROFILES:
        # Streaming adapters and the MTP path both need the single-request
        # generator: the batch generator builds its own cache types, which the
        # vendored linear-attention layers reject.
        self.is_batchable = False
    return result


server.ModelProvider.load = load_local
original_init = server.ResponseGenerator.__init__


def generator_init(self, provider, prompt_cache):
    if OPTIONS.mtp_draft is not None:
        # MTP keeps prompt caches in memory: the disk cache restores mlx_lm's
        # own cache classes, and the vendored linear-attention layers reject
        # them (`ArraysCache` has no `update_window`).
        original_init(self, provider, prompt_cache)
        return
    codec = None
    if PROFILE.get('flash'):
        import flash_cache as codec
    disk = DiskPromptCache(ROOT / 'kv-cache', FINGERPRINT,
                          int(float(os.environ.get('K2_SSD_CACHE_GIB', '64')) * 2**30), codec=codec)
    with METRICS.lock:
        METRICS.extra.update(disk_entries=len(disk), disk_bytes=disk.disk_bytes)
    original_init(self, provider, disk)


server.ResponseGenerator.__init__ = generator_init
original_single = server.ResponseGenerator._serve_single
SPECULATIVE = {'model': None, 'drafter': None}


def cache_coverage(caches):
    """Tokens the caches hold, when every cache that tracks an offset agrees."""
    offsets = {c.offset for c in caches if hasattr(c, 'offset')}
    return offsets.pop() if len(offsets) == 1 else None


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
            # The speculative round loop's cache holds the prompt plus every
            # accepted draft, so its offset runs past its key - and a follow-up
            # turn that fetched it would continue from positions the key does
            # not record. That is silent corruption, so drafting never shares a
            # cache with anything: a fresh one per request, discarded at the end.
            cache = model.make_cache()
            prompt_ids = mx.array([prompt])
            ctx.prompt_cache_count = 0
            cache_key = prompt[:]
            stop_matcher = stop_sequences.matcher()
            greedy = getattr(args, 'temp', 1.0) == 0
            # End of turn for these templates is a special token that is not
            # always in the tokenizer's eos list, and the ordinary path strips it
            # through the text state machine, which this path does not run. Only
            # the eos set is honoured: the broader all_special_ids includes the
            # vision and audio markers, which are text-model vocabulary a reply
            # may legitimately sample.
            eos_ids = set(getattr(tokenizer, 'eos_token_ids', None) or [])
            if not eos_ids and getattr(tokenizer, 'eos_token_id', None) is not None:
                eos_ids = {tokenizer.eos_token_id}
            emitted, text = 0, ''
            # Tokens are decoded through the tokenizer's own detokenizer: a
            # character whose bytes span several tokens decodes to replacement
            # characters if each token is decoded on its own.
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            stream_state = mtp_speculation.stream_speculative(
                model, SPECULATIVE['drafter'], cache, prompt_ids,
                args.max_tokens, sampler, greedy=greedy, eos_token_ids=eos_ids,
            )
            finish = 'length'
            for token, _ in stream_state:
                # Checked before the detokenizer sees it: the end-of-turn token
                # is a special token, and adding it first would emit its marker
                # into the reply.
                if token in eos_ids:
                    finish = 'stop'
                    break
                if stop_matcher.advance(token) or ctx._should_stop:
                    finish = 'stop'
                    break
                emitted += 1
                detokenizer.add_token(token)
                whole = detokenizer.last_segment
                text += whole
                rqueue.put(server.Response(whole, token, 0.0, None, None))
            detokenizer.finalize()
            tail = detokenizer.last_segment
            # The end of the reply carries the finish reason, so a client can
            # tell a completed answer from one truncated at max_tokens.
            if tail or finish != 'length':
                rqueue.put(server.Response(tail, token, 0.0, finish, None))
            rounds = getattr(stream_state, 'rounds', 0)
            logging.info('MTP: emitted %d tokens over %d rounds (%.2f accepted per round)',
                         emitted, rounds, emitted / max(rounds, 1))
            rqueue.put(None)
            # The cache covers more tokens than its key, so storing it would
            # poison the next turn's reuse. It is discarded.
            del cache, cache_key
        except Exception as exc:
            rqueue.put(exc)

    server.ResponseGenerator._serve_single = serve_single


VISION = {'vm': None}


def install_vision():
    """Route model loading and single-request generation through the full VLM.

    The text-only path strips the vision tower, so it cannot consume images.
    Here the checkpoint is loaded as a complete VLM (text + vision) via
    ``vision_engine``; a request with image parts is decoded and run through
    the VLM, and a text-only request takes the same path (the VLM generates
    text identically). Generation is single-request: the VLM's prefill needs
    the pixel values, which the batch generator cannot supply.
    """
    import input_parts
    import vision_engine
    upstream_load = server.load

    def load_vision(model_path, *args, **kwargs):
        if str(model_path) != str(MODEL_PATH):
            return upstream_load(model_path, *args, **kwargs)
        vm = vision_engine.load_vision_model(str(MODEL_PATH))
        VISION['vm'] = vm
        global SERVED_SUPPORTS_VISION
        SERVED_SUPPORTS_VISION = True
        return vm.model, vm.tokenizer

    server.load = load_vision

    def serve_single(self, request, stream):
        if VISION['vm'] is None:
            return original_single(self, request, stream)
        rqueue, request, args = request
        try:
            vm = VISION['vm']
            tokenizer = vm.tokenizer
            messages = [dict(m) for m in request.messages]
            template_messages, images = input_parts.extract_vision_messages(messages)
            # Same template arguments the text path uses, so tools render.
            template_args = dict(self.model_provider.cli_args.chat_template_args or {})
            if getattr(args, 'chat_template_kwargs', None):
                template_args.update(args.chat_template_kwargs)
            tools = getattr(request, 'tools', None)
            input_ids, extra = vm.build_inputs(
                template_messages, images, tools=tools, chat_template_args=template_args)
            prompt = [int(t) for t in input_ids.tolist()]

            stop_sequences, text_sm = self._make_state_machine(
                self.model_provider.model_key, tokenizer, args.stop_words)
            initial_state = 'normal'
            if getattr(tokenizer, 'has_thinking', False):
                if tokenizer.rfind_think_start(prompt) > tokenizer.rfind_think_end(prompt):
                    initial_state = 'reasoning'
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
            stop_matcher = stop_sequences.matcher()
            eos_ids = set(vm._eos_ids())
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            emitted, finish, token = 0, 'length', None
            for token in vm.generate_tokens(
                    max_tokens=args.max_tokens, sampler=sampler, inputs=(input_ids, extra)):
                if token in eos_ids:
                    finish = 'stop'
                    break
                if stop_matcher.advance(token) or ctx._should_stop:
                    finish = 'stop'
                    break
                emitted += 1
                detokenizer.add_token(token)
                whole = detokenizer.last_segment
                rqueue.put(server.Response(whole, token, 0.0, None, None))
            detokenizer.finalize()
            tail = detokenizer.last_segment
            if tail or finish != 'length':
                rqueue.put(server.Response(tail, token, 0.0, finish, None))
            rqueue.put(None)
        except Exception as exc:
            rqueue.put(exc)

    server.ResponseGenerator._serve_single = serve_single


original_process_message_content = server.process_message_content

from input_parts import (
    ContentPartError,
    decode_image,
    normalize_message_content,
    normalize_tool_call_arguments,
)

# Set true by install_vision() once the vision model is loaded. The text path
# uses it to report an image part honestly on a profile that has no vision
# loader (a vision profile never reaches process_message_content: its requests
# are built by install_vision's serve_single).
SERVED_SUPPORTS_VISION = False
SERVED_PROFILE_NAME = "unknown"


def process_message_content(messages):
    """Normalise agentic content parts (text/tool_use/tool_result) into text.

    The pinned implementation raises ``ValueError("Only 'text' content type is
    supported")`` on any part whose type is not ``"text"``, which breaks every
    real agentic client. This handles the legitimate shapes uniformly (see
    ``input_parts``) and reports a genuinely unsupported part with its type.
    """
    normalized, images = normalize_message_content(messages)
    if images and not SERVED_SUPPORTS_VISION:
        raise ValueError(
            f"Image content was provided, but profile '{SERVED_PROFILE_NAME}' is "
            f"served text-only; vision inference is not wired for it."
        )
    messages[:] = normalized
    for message in messages:
        normalize_tool_call_arguments(message)


server.process_message_content = process_message_content


original_handle_completion = server.APIHandler.handle_completion


def handle_completion(self, request, stop_words):
    """Prune old reasoning from history before the prompt is even built.

    The attention cost of a token grows with everything before it, so a long
    conversation of reasoning turns is exactly the workload that slows decode.
    Dropping the deliberation the model already answered with is what bounds it.
    """
    budget = getattr(OPTIONS, 'thinking_budget', None)
    import logging
    logging.info('handle_completion entered: budget=%r messages=%r',
                 budget, type(getattr(request, 'messages', None)).__name__ if hasattr(request, 'messages') else 'absent')
    if budget is not None and getattr(request, 'messages', None):
        import thinking_history
        before = thinking_history.estimate_saved(request.messages, budget)
        if before:
            import logging
            logging.info('thinking history: pruning ~%d chars of older reasoning (budget %s)',
                         before, budget)
        request.messages = thinking_history.prune(request.messages, budget)
    return original_handle_completion(self, request, stop_words)


server.APIHandler.handle_completion = handle_completion
import logging as _lh_log
_lh_log.getLogger().info('THINKING-HOOK INSTALLED at module top')


original_generate = server.ResponseGenerator.generate


def generate(self, request, generation_args, progress_callback=None):
    key = METRICS.submit(request)
    try:
        ctx, responses = original_generate(self, request, generation_args, progress_callback)
    except Exception:
        METRICS.finish(key, error=True)
        raise
    tokenizer = getattr(self.model_provider, 'tokenizer', None)
    if tokenizer is None or not output_channels.addressed(tokenizer):
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
    # Markers straddle boundaries, so the transformer holds back a partial run.
    # Without this flush the final characters of every normalised reply are lost.
    tail = normaliser.feed('', final=True)
    if tail:
        yield dataclasses.replace(response, text=tail)


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
    # Metrics live beside the instance record so several servers can run at
    # once without overwriting each other's samples.
    run_dir = ROOT / 'run'
    run_dir.mkdir(exist_ok=True)
    METRICS.start_sampler(run_dir / f'{OPTIONS.port}.metrics.json', mx)
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
    # llama.cpp-shaped clients probe /health and read /props before they will
    # show a model at all; the routes are additive and never shadow the OpenAI
    # ones, so they are installed unconditionally.
    import llamacpp_compat

    llamacpp_compat.install(server, PROFILE, OPTIONS)
    # MTP is lossless and faster, so it is on by default where the profile ships
    # a drafter for it; --no-mtp or an explicit --mtp-draft overrides that.
    if OPTIONS.mtp_draft is None and not getattr(OPTIONS, 'no_mtp', False):
        shipped = PROFILE.get('drafter')
        if shipped and Path(shipped).is_dir():
            OPTIONS.mtp_draft = Path(shipped)
            import logging
            logging.info('MTP drafting enabled with %s', shipped)
    # Load eagerly: /health answers 503 until a model exists, and a client that
    # probes it before sending work would otherwise see "loading" until someone
    # else happened to make the first request.
    original_provider_load = server.ModelProvider.load_default

    def load_then_serve(provider):
        try:
            original_provider_load(provider)
        except Exception as exc:  # a failed load must still leave the server up
            import logging
            logging.error('model did not load: %s', exc)
            return
        import logging
        logging.info('model ready: %s', MODEL_ALIAS)

    server.ModelProvider.load_default = load_then_serve
    if PROFILE['alias'] in VISION_PROFILES:
        if OPTIONS.decode_concurrency > 1 or OPTIONS.prompt_concurrency > 1:
            raise SystemExit('Vision serving is single-request; drop the concurrency flags.')
        install_vision()
    elif OPTIONS.mtp_draft is not None:
        if OPTIONS.decode_concurrency > 1 or OPTIONS.prompt_concurrency > 1:
            raise SystemExit('MTP drafting is single-request; drop the concurrency flags.')
        install_mtp(Path(OPTIONS.mtp_draft))
    server.main()
