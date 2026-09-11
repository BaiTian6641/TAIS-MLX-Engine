"""Prefill, decode, prompt-reuse and batch timings for a live server.

Every measurement is preceded by a warm-up request: the first request after a
model load pays page-fault and weight-materialisation costs that have nothing to
do with the setting under test, and on this machine that alone can move a number
by 5x.

One call measures one server configuration; each run is labelled and appended to
``<model>-optimizations.json`` so a sweep can be compared afterwards.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

FILLER = ('def process(records):\n    out = []\n    for record in records:\n'
          '        value = record.normalise()\n        out.append(value)\n    return out\n\n')
QUESTION = '\n\nExplain in one short sentence what this function does.'


def make_prompt(target_tokens, tokenizer):
    blocks, tokens = [], 0
    while tokens < target_tokens:
        blocks.append(FILLER)
        tokens = len(tokenizer.encode(''.join(blocks)))
    return ''.join(blocks) + QUESTION


def url(port, path='chat/completions'):
    return f'http://127.0.0.1:{port}/v1/{path}'


def post(port, body, timeout=1800):
    request = urllib.request.Request(url(port), data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload, time.perf_counter() - started


def stream_request(port, body, timeout=1800):
    """Time to first token and total wall time for a streaming request."""
    request = urllib.request.Request(url(port), data=json.dumps(dict(body, stream=True)).encode(),
                                     headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    first, usage, tokens = None, {}, 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                break
            chunk = json.loads(raw)
            if chunk.get('choices'):
                delta = chunk['choices'][0]['delta']
                # Thinking models stream `reasoning` before any `content`; the
                # first delta of either kind is the first token.
                if delta.get('content') or delta.get('reasoning'):
                    tokens += 1
                    if first is None:
                        first = time.perf_counter() - started
            if chunk.get('usage'):
                usage = chunk['usage']
    return {'ttft': first, 'wall': time.perf_counter() - started, 'usage': usage, 'tokens': tokens}


def chat(model, prompt, max_tokens=8):
    return {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-path', type=Path, help='checkpoint directory for the tokenizer')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--label', required=True, help='what configuration this run measures')
    parser.add_argument('--prompt-tokens', type=int, default=2048)
    parser.add_argument('--decode-tokens', type=int, default=128)
    parser.add_argument('--concurrency', type=int, default=1,
                        help='simultaneous decode requests used to measure batch throughput')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    model_path = args.model_path or Path('models') / args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = make_prompt(args.prompt_tokens, tokenizer)
    report = {'model': args.model, 'label': args.label,
              'prompt_tokens': len(tokenizer.encode(prompt)), 'concurrency': args.concurrency}

    # Warm-up with a *different* prompt of the same length: it materialises the
    # weight pages and page cache without leaving the measured prompt sitting in
    # the prompt cache, so the timed request is warm but still a cache miss.
    other_prompt = make_prompt(args.prompt_tokens, tokenizer)[::-1]
    if len(tokenizer.encode(other_prompt)) < args.prompt_tokens // 2:
        other_prompt = QUESTION + other_prompt
    stream_request(args.port, chat(args.model, other_prompt, 4))
    stream_request(args.port, chat(args.model, 'Say ready.', 4))

    ttft = stream_request(args.port, chat(args.model, prompt, 8))
    if not ttft['tokens'] or ttft['ttft'] is None:
        raise SystemExit(f'prefill produced no tokens for a {report["prompt_tokens"]}-token prompt; '
                         'the server may have rejected the request or hit a context limit')
    report['ttft_seconds'] = round(ttft['ttft'], 3)
    report['prefill_tokens_per_second'] = round(report['prompt_tokens'] / ttft['ttft'], 1)

    # Prompt-prefix reuse: the identical request again should skip prefill.
    reused = stream_request(args.port, chat(args.model, prompt, 8))
    report['ttft_reused_seconds'] = round(reused['ttft'], 3)
    report['reused_prompt_tokens'] = (reused['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
    report['prefill_speedup_on_reuse'] = round(ttft['ttft'] / reused['ttft'], 2)

    # Decode, single stream.
    payload, seconds = post(args.port, chat(args.model, 'Count from 1 to 200.', args.decode_tokens))
    completion = payload['usage']['completion_tokens']
    report['decode_tokens_per_second'] = round(completion / seconds, 1)

    # Decode, N streams in flight: what batching is for.
    if args.concurrency > 1:
        prompts = ['Count from 1 to 200.', 'Name colours one per line.',
                   'List prime numbers.', 'Describe a bicycle briefly.']
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            started = time.perf_counter()
            results = list(pool.map(
                lambda text: post(args.port, chat(args.model, text, args.decode_tokens)),
                [prompts[i % len(prompts)] for i in range(args.concurrency)]))
            batch_seconds = time.perf_counter() - started
        tokens = sum(payload['usage']['completion_tokens'] for payload, _ in results)
        report['batch_tokens'] = tokens
        report['batch_seconds'] = round(batch_seconds, 3)
        report['batch_tokens_per_second'] = round(tokens / batch_seconds, 1)
        report['batch_speedup'] = round((tokens / batch_seconds) / report['decode_tokens_per_second'], 2)

    destination = args.output or Path(__file__).resolve().parent / f'{args.model}-optimizations.json'
    history = json.loads(destination.read_text()) if destination.exists() else []
    history = [entry for entry in history if entry.get('label') != args.label]
    history.append(report)
    destination.write_text(json.dumps(history, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
