"""Answer the endpoints llama.cpp clients expect.

The engine speaks the OpenAI API, which is what mlx-lm's server implements. A
client written for `llama-server` - a chat UI, a manager on a small board, a
monitoring probe - checks `/health` first, reads `/props` for the model and its
context, and generates through the native `/completion`. Without those it looks
like nothing is running even though `/v1/chat/completions` works, which is
exactly the report this module exists to answer.

Shapes follow llama.cpp's own server documentation: `/health` answers 200 with
`{"status": "ok"}` once the model is loaded and 503 with an error object while it
is not; `/props` reports the model path, context and chat template;
`/v1/models` carries the same `meta` block; `/completion` streams Server-Sent
Events with `content`, `tokens` and `stop`.
"""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent


def _tokenizer(handler):
    provider = getattr(getattr(handler, 'response_generator', None), 'model_provider', None)
    return getattr(provider, 'tokenizer', None), provider


def _loaded(handler):
    _, provider = _tokenizer(handler)
    return bool(getattr(provider, 'model', None))


def install(server, profile, options):
    """Attach the compatibility routes to a running server module."""
    original_get = server.APIHandler.do_GET
    original_post = server.APIHandler.do_POST
    alias = profile['alias']
    model_path = str(Path(profile['path']))
    config = profile.get('config', {})
    text = config.get('text_config', config)

    def send_json(handler, payload, status=200):
        body = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header('Content-Type', 'application/json; charset=utf-8')
        handler.send_header('Content-Length', str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def send_text(handler, body, content_type='text/plain; charset=utf-8', status=200):
        body = body.encode() if isinstance(body, str) else body
        handler.send_response(status)
        handler.send_header('Content-Type', content_type)
        handler.send_header('Content-Length', str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def health(handler):
        """503 while loading, 200 with the llama.cpp body once ready.

        This is the endpoint a client probes before it will show a model, so it
        must not claim readiness it does not have - and must not keep claiming
        loading once the server can serve. serve.py loads the model at start-up
        for exactly this reason.
        """
        if not _loaded(handler):
            return send_json(handler, {'error': {'code': 503, 'message': 'Loading model',
                                                 'type': 'unavailable_error'}}, 503)
        return send_json(handler, {'status': 'ok'})

    def props(handler):
        tokenizer, _ = _tokenizer(handler)
        template = getattr(tokenizer, 'chat_template', None) or ''
        return send_json(handler, {
            'model_path': model_path,
            'alias': alias,
            'model_type': config.get('model_type'),
            'n_ctx': int(text.get('max_position_embeddings') or 0),
            'n_embd': text.get('hidden_size'),
            'n_params': None,
            'chat_template': template,
            'bos_token': getattr(tokenizer, 'bos_token', None),
            'eos_token': getattr(tokenizer, 'eos_token', None),
            'total_slots': 1,
            'default_generation_settings': {
                'id': 0,
                'id_task': -1,
                'n_ctx': int(text.get('max_position_embeddings') or 0),
                'speculative': bool(getattr(options, 'mtp_draft', None)),
                'is_processing': False,
                'params': {
                    'n_predict': -1,
                    'seed': 4294967295,
                    'temperature': 0.8,
                    'top_k': 40,
                    'top_p': 0.95,
                    'min_p': 0.05,
                    'stop': [],
                    'max_tokens': -1,
                    'n_keep': 0,
                    'stream': True,
                },
                'prompt': '',
            },
        })

    def models(handler):
        """OpenAI shape plus the `meta` block llama.cpp clients read."""
        import time
        return send_json(handler, {
            'object': 'list',
            'data': [{
                'id': alias,
                'object': 'model',
                'created': int(getattr(handler, '_started', 0) or time.time()),
                'owned_by': 'tais-mlx',
                'meta': {
                    'vocab_type': 1,
                    'n_vocab': text.get('vocab_size'),
                    'n_ctx_train': int(text.get('max_position_embeddings') or 0),
                    'n_embd': text.get('hidden_size'),
                    'n_params': None,
                    'size': None,
                },
            }],
        })

    def tokenize(handler, body):
        tokenizer, _ = _tokenizer(handler)
        content = body.get('content') or body.get('text') or ''
        if tokenizer is None:
            return send_json(handler, {'error': 'no tokenizer'}, 503)
        ids = tokenizer.encode(content)
        payload = {'tokens': list(ids)}
        if body.get('return_piece'):
            payload['pieces'] = [tokenizer.decode([i]) for i in ids]
        return send_json(handler, payload)

    def detokenize(handler, body):
        tokenizer, _ = _tokenizer(handler)
        if tokenizer is None:
            return send_json(handler, {'error': 'no tokenizer'}, 503)
        return send_json(handler, {'content': tokenizer.decode(body.get('tokens') or [])})

    def slots(handler):
        """One slot, idle; enough for a client that lists them before sending work."""
        return send_json(handler, [{
            'id': 0,
            'id_task': -1,
            'state': 0,
            'is_processing': False,
            'n_ctx': int(text.get('max_position_embeddings') or 0),
            'prompt_n': 0,
            'next_token': {'has_next_token': False, 'n_remain': -1},
            'model': alias,
        }])

    def metrics(handler):
        try:
            import services
            sample = services.metrics(options.port) or {}
        except Exception:
            sample = {}
        lines = [
            '# TYPE tais_up gauge',
            'tais_up 1',
            '# TYPE tais_requests_total counter',
            f'tais_requests_total {sample.get("completed", 0)}',
            '# TYPE tais_requests_failed_total counter',
            f'tais_requests_failed_total {sample.get("failed", 0)}',
            '# TYPE tais_tokens_per_second gauge',
            f'tais_tokens_per_second {float(sample.get("decode_tps") or 0)}',
            '# TYPE tais_cpu_percent gauge',
            f'tais_cpu_percent {float(sample.get("cpu_percent") or 0)}',
        ]
        return send_text(handler, '\n'.join(lines) + '\n')

    def completion(handler, body):
        """The native llama.cpp completion, streamed as Server-Sent Events."""
        import time

        from mlx_lm import server as mlx_server

        tokenizer, _ = _tokenizer(handler)
        prompt = body.get('prompt', '')
        if isinstance(prompt, list):
            prompt = ''.join(str(item) for item in prompt)
        n_predict = body.get('n_predict')
        max_tokens = int(n_predict) if n_predict and int(n_predict) > 0 else getattr(
            handler, 'max_tokens', 512)
        stop_words = body.get('stop') or []
        if isinstance(stop_words, str):
            stop_words = [stop_words]
        stream = bool(body.get('stream'))

        # `requested_model` is set by the server's own body parser, which this
        # route intercepts before it runs, so fall back to the profile's alias.
        requested = getattr(handler, 'requested_model', None) or alias
        args = mlx_server.GenerationArguments(
            model=mlx_server.ModelDescription(model=requested, draft=None, adapter=None),
            sampling=mlx_server.SamplingArguments(
                temperature=float(body.get('temperature', 0.8)),
                top_p=float(body.get('top_p', 0.95)),
                top_k=int(body.get('top_k', 40)),
                min_p=float(body.get('min_p', 0.05)),
                xtc_probability=0.0, xtc_threshold=0.1),
            logits=mlx_server.LogitsProcessorArguments(
                logit_bias=None, repetition_penalty=None, repetition_context_size=20,
                presence_penalty=None, presence_context_size=20,
                frequency_penalty=None, frequency_context_size=20),
            stop_words=list(stop_words), max_tokens=max_tokens, num_draft_tokens=3,
            logprobs=False, top_logprobs=0, seed=body.get('seed'),
            chat_template_kwargs={},
        )
        request = mlx_server.CompletionRequest(request_type='text', prompt=prompt,
                                               messages=[], tools=None, role_mapping=None)
        try:
            _, responses = handler.response_generator.generate(request, args)
        except Exception as exc:
            return send_json(handler, {'error': str(exc)}, 404)

        if stream:
            handler.send_response(200)
            handler.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            handler.send_header('Cache-Control', 'no-cache')
            handler.end_headers()
        started = time.perf_counter()
        produced, finish, content = 0, 'limit', []
        try:
            for response in responses:
                if response.text:
                    produced += 1
                    content.append(response.text)
                if response.finish_reason:
                    finish = 'eos' if response.finish_reason == 'stop' else 'limit'
                if not stream:
                    continue
                chunk = {'content': response.text or '',
                         'tokens': [response.token] if response.token is not None else [],
                         'stop': bool(response.finish_reason)}
                handler.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
                handler.wfile.flush()
        except Exception as exc:
            if stream:
                handler.wfile.write(f'data: {json.dumps({"error": str(exc)})}\n\n'.encode())
                handler.wfile.flush()
                return
            return send_json(handler, {'error': str(exc)}, 500)

        seconds = max(1e-6, time.perf_counter() - started)
        if stream:
            tail = {'content': '', 'stop': True, 'tokens_predicted': produced,
                    'stop_type': finish, 'timings': {'predicted_per_second': produced / seconds}}
            handler.wfile.write(f'data: {json.dumps(tail)}\n\n'.encode())
            handler.wfile.write(b'data: [DONE]\n\n')
            handler.wfile.flush()
            return
        return send_json(handler, {
            'content': ''.join(content),
            'stop': True,
            'tokens_predicted': produced,
            'stop_type': finish,
            'model': alias,
            'timings': {'predicted_per_second': produced / seconds},
        })

    def do_GET(handler):
        path = urlsplit(handler.path).path.rstrip('/') or '/'
        try:
            if path in ('/health', '/v1/health'):
                return health(handler)
            if path == '/props':
                return props(handler)
            if path == '/v1/models':
                return models(handler)
            if path == '/slots':
                return slots(handler)
            if path == '/metrics':
                return metrics(handler)
        except Exception as exc:
            return send_json(handler, {'error': str(exc)}, 500)
        return original_get(handler)

    def do_POST(handler):
        path = urlsplit(handler.path).path.rstrip('/') or '/'
        if path not in ('/completion', '/tokenize', '/detokenize'):
            # Anything else belongs to the server's own handler, which reads the
            # body itself - consuming it here would leave that read waiting for
            # bytes that never arrive.
            return original_post(handler)
        try:
            length = int(handler.headers.get('Content-Length') or 0)
            body = json.loads(handler.rfile.read(length) or b'{}') if length else {}
        except Exception:
            return send_json(handler, {'error': 'malformed JSON body'}, 400)
        try:
            if path == '/completion':
                return completion(handler, body)
            if path == '/tokenize':
                return tokenize(handler, body)
            return detokenize(handler, body)
        except Exception as exc:
            return send_json(handler, {'error': str(exc)}, 500)

    server.APIHandler.do_GET = do_GET
    server.APIHandler.do_POST = do_POST
