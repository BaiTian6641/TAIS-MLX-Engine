"""Concurrency correctness check against a live server.

Batching must not change what a request returns: every answer produced with
several requests in flight has to match the answer the same request gets on its
own. Run against a server started with --decode-concurrency > 1.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
PROMPTS = [
    ('What is the capital of France? Answer with only the city name.', 'Paris'),
    ('What is the capital of Italy? Answer with only the city name.', 'Rome'),
    ('What is the capital of Japan? Answer with only the city name.', 'Tokyo'),
    ('What is the capital of Spain? Answer with only the city name.', 'Madrid'),
]


def ask(port, model, prompt, max_tokens=64, stream=False):
    payload = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                          'max_tokens': max_tokens, 'temperature': 0, 'stream': stream}).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=payload,
                                     headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        if stream:
            chunks, done = [], False
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:].strip()
                if raw == b'[DONE]':
                    done = True
                    break
                chunk = json.loads(raw)
                if chunk.get('choices'):
                    chunks.append(chunk['choices'][0]['delta'].get('content', '') or '')
            return {'seconds': time.perf_counter() - started, 'content': ''.join(chunks), 'done': done}
        body = json.load(response)
    message = body['choices'][0]['message']
    # Thinking models put their scratch work in `reasoning`; with a tight token
    # budget the answer itself may not have started yet.
    return {'seconds': time.perf_counter() - started,
            'content': ((message.get('reasoning') or '') + (message.get('content') or '')),
            'tokens': body['usage']['completion_tokens']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    failures = []

    # 1. sequential baseline
    sequential = {}
    for prompt, expected in PROMPTS:
        result = ask(args.port, args.model, prompt)
        sequential[prompt] = result['content'].strip()
        if expected.lower() not in result['content'].lower():
            failures.append(f'sequential answer for {expected}: {result["content"]!r}')

    # 2. the same prompts, all in flight: answers must not change
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        concurrent = list(pool.map(lambda item: ask(args.port, args.model, item[0]), PROMPTS))
    mismatched = [(prompt, sequential[prompt], result['content'].strip())
                  for (prompt, _), result in zip(PROMPTS, concurrent)
                  if result['content'].strip() != sequential[prompt]]
    if mismatched:
        failures.append(f'{len(mismatched)} concurrent answers differ from sequential: {mismatched[:2]}')

    # 3. one prompt many times at once: every copy must be identical
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        repeats = list(pool.map(lambda _: ask(args.port, args.model, PROMPTS[0][0]), range(args.concurrency)))
    if len({result['content'].strip() for result in repeats}) != 1:
        failures.append(f'repeated concurrent answers differ: {[r["content"].strip() for r in repeats]}')

    # 4. oversubscribe: more requests than slots, all must complete
    extra = PROMPTS * 2
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(extra)) as pool:
        queued = list(pool.map(lambda item: ask(args.port, args.model, item[0]), extra))
    queue_seconds = time.perf_counter() - started
    bad = [result for result, (_, expected) in zip(queued, extra)
           if expected.lower() not in result['content'].lower()]
    if bad:
        failures.append(f'{len(bad)} of {len(extra)} oversubscribed requests returned a wrong answer')

    # 5. streaming while others are in flight
    with ThreadPoolExecutor(max_workers=3) as pool:
        streams = list(pool.map(lambda item: ask(args.port, args.model, item[0], stream=True), PROMPTS[:2]))
        inline = pool.submit(ask, args.port, args.model, PROMPTS[2][0]).result()
    if not all(result['done'] and result['content'].strip() for result in streams):
        failures.append(f'streaming under load failed: {streams}')
    if 'tokyo' not in inline['content'].lower():
        failures.append(f'non-streaming answer under load wrong: {inline["content"]!r}')

    report = {'model': args.model, 'concurrency': args.concurrency,
              'sequential': sequential,
              'concurrent_matched': not mismatched,
              'repeat_answers_identical': len({r['content'].strip() for r in repeats}) == 1,
              'oversubscribed_seconds': queue_seconds,
              'oversubscribed_ok': not bad,
              'stream_under_load_ok': all(result['done'] for result in streams),
              'failures': failures}
    print(json.dumps(report, indent=2))
    destination = args.output or ROOT / f'{args.model}-concurrency-check.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    if failures:
        raise SystemExit(f'{len(failures)} concurrency checks failed')


if __name__ == '__main__':
    main()
