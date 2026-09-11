"""Exercise the real HTTP wrapper with small Flash architecture fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from flash_models import model_classes
from test_flash import qwen_config, ds_config


def run(alias, config, path):
    cls, args = model_classes(config)
    model = cls(args.from_dict(config))
    nn.quantize(model, group_size=32, bits=4,
                class_predicate=lambda p,m: hasattr(m,'to_quantized') and
                ('switch_mlp' in p or '.ngram_embedding.shards.' in p))
    config['quantization'] = {'group_size':32,'bits':4,'mode':'affine'}
    config['eos_token_id'] = 2
    (path/'config.json').write_text(json.dumps(config))
    mx.save_safetensors(str(path/'model.safetensors'),dict(tree_flatten(model.parameters())))
    del model
    mx.clear_cache()
    vocabulary = {'[UNK]':0,'[BOS]':1,'[EOS]':2}
    vocabulary.update({f'w{i}':i for i in range(3,128)})
    core = Tokenizer(models.WordLevel(vocabulary,unk_token='[UNK]'))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core,unk_token='[UNK]',
                                       bos_token='[BOS]',eos_token='[EOS]')
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tokenizer.save_pretrained(path)
    log_path = Path(f'{alias}-fixture-server.log')
    with log_path.open('w') as log:
        proc = subprocess.Popen([sys.executable,'serve.py','--model',alias,'--model-path',str(path),
                                 '--expert-cache-gib','0.02','--host','127.0.0.1','--port','8082'],
                                stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if proc.poll() is not None:
                    raise RuntimeError(log_path.read_text())
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8082/v1/models',timeout=1) as r:
                        assert json.load(r)['data'][0]['id']==alias
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise TimeoutError('Fixture server did not start')
            prompt = 'w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19'
            def post(body,endpoint='/completions'):
                request = urllib.request.Request('http://127.0.0.1:8082/v1'+endpoint,
                    data=json.dumps(dict(body,model=alias,temperature=0,seed=42)).encode(),
                    headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=60) as r:
                    if body.get('stream'):
                        raw = r.read().decode()
                        assert 'data: [DONE]' in raw,raw
                        return {'stream':'passed'}
                    return json.load(r)
            first = post({'prompt':prompt,'max_tokens':4})
            from model_profiles import fingerprint
            directory = Path('kv-cache')/fingerprint(path)
            for _ in range(100):
                entries = list(directory.glob('*.json'))
                if entries:
                    break
                time.sleep(.05)
            entry = json.loads(max(entries,key=lambda p:p.stat().st_mtime).read_text())
            prefix = tokenizer.decode(entry['tokens'],skip_special_tokens=False)
            reused = post({'prompt':prefix+' w20','max_tokens':4})
            assert reused['usage']['prompt_tokens_details']['cached_tokens']>0,reused
            chat = post({'messages':[{'role':'user','content':prompt}],'max_tokens':4},'/chat/completions')
            stream = post({'prompt':prompt,'max_tokens':4,'stream':True})
            for _ in range(100):
                stats = json.loads(Path('metrics.json').read_text())
                if stats.get('pid')==proc.pid and stats.get('completed',0)>=4:
                    break
                time.sleep(.05)
            assert stats.get('expert_misses',0)>0,stats
            assert stats.get('cache_hits',0)>0,stats
            return {'first':first,'reused':reused,'chat':chat,'stream':stream,'metrics':stats}
        finally:
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()


results = {}
for alias,config in [('qwen3.8-flash',qwen_config()),('deepseek-v4-flash',ds_config())]:
    with tempfile.TemporaryDirectory() as tmp:
        results[alias] = run(alias,config,Path(tmp))
    print(alias,'HTTP fixture passed',flush=True)
Path('flash-http-fixtures.json').write_text(json.dumps(results,indent=2)+'\n')
