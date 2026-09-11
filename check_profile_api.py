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
    body = {'messages':[{'role':'user','content':'What is 2 + 2? Answer with only the number.'}],
            'max_tokens':32, 'chat_template_kwargs':{'enable_thinking':False, 'thinking_mode':'chat'}}
    results['chat'] = post('/chat/completions',body)
    assert '4' in results['chat']['choices'][0]['message']['content']
    results['stream'] = post('/chat/completions',dict(body,stream=True))
    assert '4' in results['stream']['content']
    prompt = 'Count: 1, 2, 3,'
    first = post('/completions',{'prompt':prompt,'max_tokens':4})
    # Response detokenizers may omit leading whitespace. Reconstruct the exact
    # saved token boundary for this integration probe, then add a special-token
    # separator so BPE cannot merge across the saved prefix.
    from tokenizers import Tokenizer
    from model_profiles import parse_options, resolve_profile, fingerprint
    profile = resolve_profile(parse_options(['--model', args.model]))
    tokenizer = Tokenizer.from_file(str(profile['path']/'tokenizer.json'))
    directory = Path('kv-cache')/fingerprint(profile['path'])
    prefix = None
    for _ in range(30):
        candidates = []
        for p in directory.glob('*.json'):
            entry = json.loads(p.read_text())
            decoded = tokenizer.decode(entry['tokens'], skip_special_tokens=False)
            if decoded.startswith(prompt):
                candidates.append((entry['used'],decoded))
        if candidates:
            prefix = max(candidates)[1]
            break
        time.sleep(.1)
    assert prefix is not None, 'No completed prefix checkpoint saved'
    separator = '<｜end▁of▁sentence｜>' if args.model.startswith('deepseek') else '<|im_end|>'
    continuation = prefix+separator+'Continue'
    second = post('/completions',{'prompt':continuation,'max_tokens':4})
    results['prefix_first'] = first
    results['prefix_continuation'] = second
    assert second['usage']['prompt_tokens_details']['cached_tokens'] > 0, second
    for _ in range(30):
        stats = json.loads(Path('metrics.json').read_text())
        if stats.get('model') == args.model and stats['running']==0 and stats['completed']>=4:
            break
        time.sleep(.5)
    results['metrics'] = stats
    Path(args.model+'-api-check.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'model':args.model,'chat':'passed','stream':'passed',
                      'cached_tokens':second['usage']['prompt_tokens_details']['cached_tokens'],
                      'mlx_active_gib':stats['mlx_active_bytes']/2**30},indent=2))


if __name__ == '__main__':
    main()
