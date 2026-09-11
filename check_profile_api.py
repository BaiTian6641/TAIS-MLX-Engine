"""Normal/streaming chat and exact-prefix cache continuation on a live profile."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=8081)
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}/v1'
    def post(path, body):
        body = dict(body, model=args.model, temperature=0)
        req = urllib.request.Request(base+path, data=json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            if body.get('stream'):
                content, done = '', False
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    raw = line[6:].strip()
                    if raw == b'[DONE]':
                        done = True
                        break
                    chunk = json.loads(raw)
                    if chunk.get('choices'):
                        content += chunk['choices'][0]['delta'].get('content','') or ''
                assert done
                return {'content':content,'done':done}
            return json.load(response)
    results = {}
    # Generous enough that a model which reasons before answering still reaches
    # its answer inside the budget.
    # A question every served model answers reliably; arithmetic phrasing is
    # model-sensitive (GLM-4.7-Flash answers "5" to "2 + 2" but "4" to "2+2"),
    # which says nothing about the server.
    body = {'messages':[{'role':'user','content':'What is the capital of France? Answer with only the city name.'}],
            'max_tokens':512, 'chat_template_kwargs':{'enable_thinking':False, 'thinking_mode':'chat'}}
    results['chat'] = post('/chat/completions',body)
    assert 'Paris' in results['chat']['choices'][0]['message']['content']
    results['stream'] = post('/chat/completions',dict(body,stream=True))
    assert 'Paris' in results['stream']['content']
    # Exact-prefix continuation, asserted on behaviour rather than on the
    # cache's internals: the same prompt sent twice must reuse the first
    # request's prefill. Scanning the cache directory for a checkpoint file
    # tested an implementation detail and reported false failures for models
    # whose caches are persisted through a different path.
    prompt = ('Count slowly: ' + ', '.join(str(n) for n in range(64)) + ',')
    first = post('/completions', {'prompt': prompt, 'max_tokens': 4})
    second = post('/completions', {'prompt': prompt, 'max_tokens': 4})
    cached = (second['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    results['cached_tokens'] = cached
    results['prefix_prompt_tokens'] = second['usage']['prompt_tokens']
    # Reported, not asserted: whether a request reaches the prompt cache depends
    # on which generator path the server chose (the batch path does not report
    # cached tokens at all), so reuse is measured by check_optimizations.py
    # instead of being a pass/fail condition of an API smoke test.
    # The llama.cpp-shaped routes: what a client written for llama-server probes
    # before it will show the model at all.
    import urllib.request as _url
    base_root = base.rsplit('/v1', 1)[0]
    def get_root(path):
        req = _url.Request(base_root + path)
        with _url.urlopen(req, timeout=120) as response:
            return response.status, json.loads(response.read())
    status, payload = get_root('/health')
    assert status == 200 and payload.get('status') == 'ok', f'/health said {status} {payload}'
    results['health'] = payload['status']
    _, props = get_root('/props')
    assert props.get('model_path') and props.get('n_ctx'), f'/props missing fields: {list(props)[:6]}'
    results['props_n_ctx'] = props['n_ctx']
    _, models = get_root('/v1/models')
    assert models['data'][0].get('meta'), '/v1/models has no meta block'
    results['models_meta'] = models['data'][0]['meta'].get('n_ctx_train')

    def root_post(path, body):
        req = _url.Request(base_root + path, data=json.dumps(body).encode(),
                           headers={'Content-Type': 'application/json'})
        with _url.urlopen(req, timeout=280) as response:
            return json.loads(response.read())

    tokens = root_post('/tokenize', {'content': 'Hello world'})['tokens']
    assert tokens, '/tokenize returned no tokens'
    text = root_post('/detokenize', {'tokens': tokens})['content']
    assert text.startswith('Hello'), f'/detokenize returned {text[:20]!r}'
    completion = root_post('/completion', {'prompt': 'The capital of France is', 'n_predict': 8,
                                           'temperature': 0})
    assert completion.get('content'), f'/completion returned {completion}'
    assert completion.get('stop') is True and completion.get('timings')
    results['completion'] = completion['content'][:40].strip()

    results['prefix_first'] = first
    results['prefix_continuation'] = second
    assert second['usage']['prompt_tokens_details']['cached_tokens'] > 0, second
    import services
    metrics_file = services.metrics_path(args.port)
    for _ in range(30):
        if not metrics_file.exists():
            time.sleep(0.5)
            continue
        stats = json.loads(metrics_file.read_text())
        if stats.get('model') == args.model and stats['running']==0 and stats['completed']>=4:
            break
        time.sleep(.5)
    results['metrics'] = stats
    Path(args.model+'-api-check.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'model': args.model, 'chat': 'passed', 'stream': 'passed',
                      'cached_tokens': second['usage']['prompt_tokens_details']['cached_tokens'],
                      'llamacpp_health': results.get('health'),
                      'llamacpp_props_n_ctx': results.get('props_n_ctx'),
                      'llamacpp_models_meta_n_ctx': results.get('models_meta'),
                      'llamacpp_completion': results.get('completion'),
                      'mlx_active_gib': stats['mlx_active_bytes'] / 2**30}, indent=2))


if __name__ == '__main__':
    main()
