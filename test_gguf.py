import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import gguf
from gguf.quants import quantize, dequantize
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import _get_classes,load_model
from gguf_import import repack_native,import_model,inspect_files,NATIVE
from test_models import tiny_config


def export_fixture(path,config,model,quant=None,override=None):
    family={'llama':'llama','qwen3_5_moe':'qwen35moe','qwen3_5':'qwen35'}[config['model_type']]
    text=config.get('text_config',config)
    arch=next(a for a,n in gguf.MODEL_ARCH_NAMES.items() if n==family)
    mapper=gguf.get_tensor_name_map(arch,text['num_hidden_layers'])
    writer=gguf.GGUFWriter(str(path),family)
    writer.add_block_count(text['num_hidden_layers'])
    def forward_heads(x,dim,head_dim):
        k,v=text['linear_num_key_heads'],text['linear_num_value_heads']
        shape=x.shape
        expanded=shape[:dim]+(k,v//k,head_dim)+shape[dim+1:]
        return x.reshape(expanded).swapaxes(dim,dim+1).reshape(shape)
    for name,weight in tree_flatten(model.parameters()):
        value=np.array(weight)
        if config['model_type']=='llama' and name.endswith(('q_proj.weight','k_proj.weight')):
            heads=text['num_attention_heads'] if 'q_proj' in name else text['num_key_value_heads']
            value=value.reshape(heads,2,-1,value.shape[-1]).swapaxes(1,2).reshape(value.shape)
        if '.linear_attn.' in name:
            if name.endswith(('in_proj_qkv.weight','conv1d.weight')):
                split=2*text['linear_num_key_heads']*text['linear_key_head_dim']
                value=np.concatenate((value[:split],forward_heads(value[split:],0,text['linear_value_head_dim'])))
            elif name.endswith('in_proj_z.weight'):value=forward_heads(value,0,text['linear_value_head_dim'])
            elif name.endswith(('in_proj_a.weight','in_proj_b.weight','A_log','dt_bias')):value=forward_heads(value,0,1)
            elif name.endswith('out_proj.weight'):value=forward_heads(value,1,text['linear_value_head_dim'])
        if name.endswith('.A_log'):value=-np.exp(value)
        if name.endswith('conv1d.weight'):value=value.squeeze(-1)
        mapped=name.removeprefix('language_model.').replace('.switch_mlp.','.experts.').replace('.dt_bias','.dt_proj.bias')
        target=mapper.get_name(mapped,try_suffixes=('.weight','.bias'))
        assert target is not None,name
        qtype=quant if quant and value.ndim>=2 and value.shape[-1]%32==0 else gguf.GGMLQuantizationType.F32
        data=quantize(value.astype(np.float32),qtype)
        if override and target in override:
            qtype,data=override[target]
        writer.add_tensor(target,np.ascontiguousarray(data),raw_dtype=qtype)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()


class GGUFTests(unittest.TestCase):
    def test_mixed_k_and_iq_formats(self):
        import mlx.nn as nn
        config=dict(model_type='llama',hidden_size=256,intermediate_size=256,
                    num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                    rms_norm_eps=1e-6,vocab_size=256,max_position_embeddings=4096)
        cls,args=_get_classes(config)
        model=cls(args.from_dict(config))
        mapper=gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA,1)
        kinds=['Q2_K','Q3_K','Q4_K','Q5_K','Q6_K','IQ2_XXS','IQ2_XS','IQ3_XXS','IQ4_NL']
        overrides={}
        references={}
        for name,weight in tree_flatten(model.parameters()):
            if weight.ndim!=2:continue
            qtype=getattr(gguf.GGMLQuantizationType,kinds.pop(0))
            block,size=gguf.GGML_QUANT_SIZES[qtype]
            rows,cols=weight.shape
            raw=np.zeros((rows*cols//block,size),dtype=np.uint8)
            offset={'Q2_K':80,'Q3_K':108,'Q6_K':208}.get(qtype.name,0)
            raw[:,offset:offset+2]=np.array([.125],dtype=np.float16).view(np.uint8)
            if qtype.name=='Q2_K':raw[:,:16]=0x11
            if qtype.name in ('Q4_K','Q5_K'):raw[:,4:16]=1
            if qtype.name=='Q6_K':raw[:,192:208]=1
            raw=raw.reshape(rows,-1)
            # These fixtures are constant within a row, so the Llama Q/K
            # output-row permutation does not alter the independent reference.
            values=dequantize(raw,qtype)
            references[name]=mx.array(values)
            overrides[mapper.get_name(name,try_suffixes=('.weight',))]=(qtype,raw)
        model.load_weights(list(references.items()),strict=False)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (source/'config.json').write_text(json.dumps(config));(source/'tokenizer.json').write_text('{}')
            file=root/'mixed.gguf';export_fixture(file,config,model,override=overrides)
            import_model([file],source,root/'mlx',bits=4,chunk_mib=1)
            loaded,_=load_model(root/'mlx')
            nn.quantize(model,group_size=32,bits=4)
            tokens=mx.array([[1,2]])
            a=model(tokens);b=loaded(tokens);mx.eval(a,b)
            self.assertTrue(mx.allclose(a,b,atol=3e-4,rtol=3e-4))

    def test_k_quant_requantization_is_explicit_and_matches_mlx(self):
        import mlx.nn as nn
        config=dict(model_type='llama',hidden_size=256,intermediate_size=256,
                    num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                    rms_norm_eps=1e-6,vocab_size=512,max_position_embeddings=4096)
        cls,args=_get_classes(config)
        mx.random.seed(4)
        model=cls(args.from_dict(config))
        qtype=gguf.GGMLQuantizationType.Q4_K
        raw=np.zeros((512,144),dtype=np.uint8)
        raw[:,:2]=np.array([.125],dtype=np.float16).view(np.uint8)
        raw[:,2:4]=np.array([.0625],dtype=np.float16).view(np.uint8)
        raw[:,4:16]=np.arange(12,dtype=np.uint8)+1
        raw[:,16:]=np.random.default_rng(4).integers(0,256,size=(512,128),dtype=np.uint8)
        model.model.embed_tokens.weight=mx.array(dequantize(raw,qtype))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (source/'config.json').write_text(json.dumps(config))
            (source/'tokenizer.json').write_text('{}')
            file=root/'model.gguf'
            export_fixture(file,config,model,override={'token_embd.weight':(qtype,raw)})
            with self.assertRaisesRegex(ValueError,'requantize-bits'):
                import_model([file],source,root/'rejected',chunk_mib=1)
            self.assertFalse((root/'rejected').exists())
            for bits in (2,3,4,6,8):
                destination=root/f'bits-{bits}'
                report=import_model([file],source,destination,bits=bits,chunk_mib=1)
                loaded,_=load_model(destination)
                # Rebuild the independently quantized reference each time.
                mx.random.seed(4)
                reference=cls(args.from_dict(config))
                reference.model.embed_tokens.weight=mx.array(dequantize(raw,qtype))
                nn.quantize(reference,group_size=32,bits=bits)
                tokens=mx.array([[1,2,3]])
                expected=reference(tokens);actual=loaded(tokens)
                mx.eval(expected,actual)
                self.assertTrue(mx.allclose(expected,actual,atol=3e-4,rtol=3e-4),bits)
                self.assertEqual(report['conversion'],'lossy MLX affine requantization')

    def test_exact_native_repacking(self):
        rng=np.random.default_rng(42)
        values=rng.normal(size=(4,128)).astype(np.float32)
        for qtype,bits in NATIVE.items():
            data=quantize(values,qtype)
            packed,scales,biases=repack_native(data,qtype,128)
            decoded=mx.dequantize(mx.array(packed),mx.array(scales),mx.array(biases),group_size=32,bits=bits)
            self.assertTrue(np.array_equal(np.array(decoded),dequantize(data,qtype)),qtype)

    def check_model(self,config,quant=None):
        cls,args=_get_classes(config)
        mx.random.seed(42)
        model=cls(args.from_dict(config))
        tokens=mx.array([[1,2,3,4]])
        expected=model(tokens,cache=model.make_cache()) if hasattr(model,'make_cache') else model(tokens)
        mx.eval(expected)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (source/'config.json').write_text(json.dumps(config))
            (source/'tokenizer.json').write_text('{}')
            file=root/'model.gguf'
            export_fixture(file,config,model,quant)
            output=root/'converted'
            import_model([file],source,output,chunk_mib=1)
            loaded,_=load_model(output)
            actual=loaded(tokens,cache=loaded.make_cache()) if hasattr(loaded,'make_cache') else loaded(tokens)
            mx.eval(actual)
            if quant is None:
                self.assertTrue(mx.allclose(expected,actual,atol=2e-4,rtol=2e-4))
            else:
                self.assertTrue(mx.all(mx.isfinite(actual)))
                # Compatibility export retains packed modules and supports the
                # existing SSD expert loader without whole-model dequantization.
                from expert_cache import install_expert_offload
                cache=install_expert_offload(loaded,output,200000)
                streamed=loaded(tokens,cache=loaded.make_cache())
                mx.eval(streamed)
                self.assertTrue(mx.allclose(actual,streamed,atol=2e-4,rtol=2e-4))
            self.assertEqual(json.loads((output/'gguf-import.json').read_text())['conversion'],
                             'exact native-grid repack')

    def test_qwen_gdn_layout(self):
        config=tiny_config(True)
        config['text_config'].update(linear_num_key_heads=2,linear_num_value_heads=4)
        self.check_model(config)

    def test_qwen_native_quantized_experts(self):
        config=tiny_config(True)
        config['text_config'].update(linear_num_key_heads=2,linear_num_value_heads=4)
        self.check_model(config,gguf.GGMLQuantizationType.Q4_0)

    def test_llama_rotary_layout(self):
        self.check_model(dict(model_type='llama',hidden_size=64,intermediate_size=128,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
            rms_norm_eps=1e-6,vocab_size=128,max_position_embeddings=4096))

if __name__=='__main__':unittest.main()
