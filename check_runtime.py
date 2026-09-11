"""Integration check against a running local server; writes runtime-check.json."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def request(stream=False):
    payload = {'model':'k2-horizon','messages':[{'role':'user','content':'What is 2 + 2? Answer briefly.'}],
               'max_tokens':512,'temperature':0,'stream':stream}
    req = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',
                                 data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as response:
        if stream:
            chunks, done = [], False
            for line in response:
                if line.startswith(b'data: '):
                    value = line[6:].strip()
                    if value == b'[DONE]':
                        done = True
                        break
                    chunks.append(json.loads(value))
            assert done
            content = ''.join(c['choices'][0]['delta'].get('content','') or '' for c in chunks if c.get('choices'))
            result = {'content':content,'chunks':len(chunks),'done':done}
        else:
            result = json.load(response)
            content = result['choices'][0]['message']['content']
        assert '4' in content, result
        return {'seconds':time.monotonic()-start,'response':result}


if __name__ == '__main__':
    snapshots = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(request, stream) for stream in (False, True)]
        while not all(j.done() for j in jobs):
            snapshots.append(json.loads((ROOT/'metrics.json').read_text()))
            time.sleep(.5)
        results = [j.result() for j in jobs]
    deadline = time.monotonic()+30
    while time.monotonic()<deadline:
        final = json.loads((ROOT/'metrics.json').read_text())
        if final['running'] == final['queued'] == 0 and final['completed'] >= 2:
            break
        time.sleep(.5)
    assert any(s['running']==1 and s['queued']>=1 for s in snapshots), 'Queue not observed'
    assert final['cache_hits'] >= 1, final
    assert final['context_limit'] > 262144, final
    assert final['mlx_active_bytes'] < 22*2**30, 'Inactive cache memory retained'
    result = {'requests':results, 'queue_observed':True, 'final_metrics':final,
              'max_generation_tps':max(s['tokens_per_second'] for s in snapshots),
              'max_gpu_percent':max(s.get('gpu_percent',0) or 0 for s in snapshots)}
    (ROOT/'runtime-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='requests'},indent=2))
