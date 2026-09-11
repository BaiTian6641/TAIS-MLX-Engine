"""Bounded-memory GGUF -> MLX import. Inference remains entirely in MLX."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np
import gguf
from gguf.quants import dequantize
from gguf.quants import _type_traits

FAMILIES = {'llama':'llama', 'qwen3_5':'qwen35', 'qwen3_5_moe':'qwen35moe'}
NATIVE = {gguf.GGMLQuantizationType.Q4_0:4, gguf.GGMLQuantizationType.Q4_1:4,
          gguf.GGMLQuantizationType.Q8_0:8}
FLOATS = {gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16,
          gguf.GGMLQuantizationType.BF16}


def inspect_files(files):
    readers = [gguf.GGUFReader(str(p)) for p in files]
    tensors = {}
    architectures = set()
    for reader in readers:
        if reader.endianess != gguf.GGUFEndian.LITTLE:
            raise ValueError('Only little-endian GGUF input is currently supported')
        architectures.add(reader.fields['general.architecture'].contents())
        for tensor in reader.tensors:
            if tensor.name in tensors:
                raise ValueError(f'Duplicate GGUF tensor: {tensor.name}')
            tensors[tensor.name] = tensor
    if len(architectures)!=1:
        raise ValueError('GGUF shards disagree on architecture')
    counts = {}
    for tensor in tensors.values():
        name = tensor.tensor_type.name
        counts[name] = counts.get(name,0)+1
    return readers,tensors,{'architecture':architectures.pop(),'tensors':len(tensors),
                           'quantization_counts':counts,'files':[str(p) for p in files]}


def repack_native(raw, qtype, columns):
    """Repack compatible GGML grids exactly, with float32 scale/bias metadata."""
    bits = NATIVE[qtype]
    block_bytes = gguf.GGML_QUANT_SIZES[qtype][1]
    blocks = np.ascontiguousarray(raw).reshape(-1,block_bytes)
    scale = blocks[:,:2].copy().view('<f2').astype(np.float32).reshape(-1)
    if qtype == gguf.GGMLQuantizationType.Q8_0:
        codes = blocks[:,2:].view(np.int8).astype(np.int16)+128
        bias = -128*scale
    else:
        start = 4 if qtype == gguf.GGMLQuantizationType.Q4_1 else 2
        packed = blocks[:,start:]
        codes = np.concatenate((packed&15,packed>>4),axis=1)
        bias = blocks[:,2:4].copy().view('<f2').astype(np.float32).reshape(-1) if start==4 else -8*scale
    codes = codes.astype(np.uint32).reshape(-1,columns)
    group = 32//bits
    weight = np.bitwise_or.reduce(codes.reshape(len(codes),-1,group) <<
                                  (np.arange(group,dtype=np.uint32)*bits),axis=-1)
    return weight,scale.reshape(-1,columns//32),bias.reshape(-1,columns//32)


def permutations(name, shape, config):
    """Undo GGUF's tiled GDN value-head layout; leave expert tensors untouched."""
    if '.linear_attn.' not in name:
        return None,None
    k,v = config['linear_num_key_heads'],config['linear_num_value_heads']
    if k==v:
        return None,None
    if v%k:
        raise ValueError('GDN value heads must be divisible by key heads')
    def inverse(dim):
        return np.arange(v*dim).reshape(v//k,k,dim).transpose(1,0,2).reshape(-1)
    if name.endswith(('in_proj_qkv.weight','conv1d.weight')):
        qk = config['linear_key_head_dim']*k*2
        return np.concatenate((np.arange(qk),qk+inverse(config['linear_value_head_dim']))),None
    if name.endswith('in_proj_z.weight'):
        return inverse(config['linear_value_head_dim']),None
    if name.endswith(('in_proj_a.weight','in_proj_b.weight','A_log','dt_bias')):
        return inverse(1),None
    if name.endswith('out_proj.weight'):
        return None,inverse(config['linear_value_head_dim'])
    return None,None


class TensorWriter:
    def __init__(self,path,entries):
        offset=0
        header={}
        self.offsets={}
        for name,shape,dtype in entries:
            size=math.prod(shape)*{'U32':4,'F32':4,'F16':2,'BF16':2}[dtype]
            header[name]={'dtype':dtype,'shape':list(shape),'data_offsets':[offset,offset+size]}
            self.offsets[name]=offset
            offset+=size
        raw=json.dumps(header,separators=(',',':')).encode()
        raw+=b' '*((-len(raw))%8)
        self.start=8+len(raw)
        self.file=path.open('w+b')
        self.file.write(struct.pack('<Q',len(raw))+raw)
        self.file.truncate(self.start+offset)
        self.size=self.start+offset
        self.data_size=offset

    def write(self,name,byte_offset,array):
        self.file.seek(self.start+self.offsets[name]+byte_offset)
        self.file.write(np.ascontiguousarray(array).tobytes())

    def close(self):
        self.file.close()


def import_model(files, config_dir, output, bits=None, group_size=32, chunk_mib=32):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.utils import _get_classes
    readers,tensors,report=inspect_files(files)
    config=json.loads((config_dir/'config.json').read_text())
    family=config['model_type']
    if family not in FAMILIES or report['architecture']!=FAMILIES[family]:
        raise ValueError(f'Validated GGUF import families: {FAMILIES}. Config={family}, GGUF={report["architecture"]}')
    config=copy.deepcopy(config)
    config.pop('quantization',None);config.pop('quantization_config',None)
    text=config.get('text_config',config)
    text.pop('quantization',None);text.pop('quantization_config',None)
    # Exact repacking stores float32 scales/biases. Conservatively account for
    # float32 activations/cache metadata in the server's memory admission.
    text['_runtime_cache_element_bytes']=4
    if family=='llama':
        config['rope_traditional']=True
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    if not (config_dir/'tokenizer.json').exists() and not (config_dir/'tokenizer.model').exists():
        raise ValueError('Provide matching local tokenizer files with --config-dir')
    cls,args=_get_classes(config)
    model=cls(args.from_dict(config))
    parameters=dict(tree_flatten(model.parameters()))
    modules=dict(tree_flatten(model.leaf_modules(),is_leaf=lambda m:isinstance(m,nn.Module)))
    arch=next(a for a,n in gguf.MODEL_ARCH_NAMES.items() if n==report['architecture'])
    mapper=gguf.get_tensor_name_map(arch,text['num_hidden_layers'])
    plans=[]
    used=set()
    quantization={'bits':4,'group_size':group_size,'mode':'affine'}
    for name,parameter in parameters.items():
        mapped=name.removeprefix('language_model.').replace('.switch_mlp.','.experts.').replace('.dt_bias','.dt_proj.bias')
        source=mapper.get_name(mapped,try_suffixes=('.weight','.bias'))
        if source not in tensors:
            raise ValueError(f'Missing GGUF mapping/tensor for {name}: {source}')
        tensor=tensors[source]
        shape=tuple(int(x) for x in reversed(tensor.shape))
        expected=tuple(parameter.shape)
        if name.endswith('conv1d.weight') and shape==expected[:-1] and expected[-1]==1:
            pass
        elif shape!=expected:
            raise ValueError(f'Shape mismatch {source}: GGUF {shape}, MLX {expected}')
        module=modules.get(name.removesuffix('.weight'))
        can_quantize=name.endswith('.weight') and hasattr(module,'to_quantized') and len(shape)>=2
        qtype=tensor.tensor_type
        if qtype not in FLOATS and qtype not in _type_traits:
            raise ValueError(f'No validated Python decoder for {qtype.name}')
        if qtype not in FLOATS and qtype not in NATIVE and bits is None:
            raise ValueError(f'{source} uses {qtype.name}: pass --requantize-bits explicitly; no automatic FP16 expansion')
        if qtype not in FLOATS and not can_quantize:
            raise ValueError(f'Quantized non-matrix tensor is unsupported: {source}')
        target_bits=bits if bits is not None and can_quantize else NATIVE.get(qtype)
        target_group=group_size if bits is not None else 32
        if target_bits and shape[-1]%target_group:
            raise ValueError(f'{source}: width {shape[-1]} is not divisible by group {target_group}')
        if target_bits:
            quantization[name[:-len('.weight')]]={'bits':target_bits,'group_size':target_group,'mode':'affine'}
        plans.append((name,tensor,shape,expected,target_bits,target_group))
        used.add(source)
    # Reject unknown architecture-specific tensors rather than dropping
    # semantics (including explicit rotary-frequency override tensors).
    unknown=set(tensors)-used
    if unknown:
        raise ValueError(f'Unmapped GGUF tensors: {sorted(unknown)[:8]}')
    del parameters,modules,model
    mx.clear_cache()
    output.parent.mkdir(parents=True,exist_ok=True)
    estimated_bytes=sum((math.prod(expected)*target_bits//8 +
                         math.prod(expected)//target_group*8) if target_bits else
                        math.prod(expected)*(2 if tensor.tensor_type==gguf.GGMLQuantizationType.F16 else 4)
                        for _,tensor,_,expected,target_bits,target_group in plans)
    if shutil.disk_usage(output.parent).free < estimated_bytes + len(plans)*4096 + 2*2**30:
        raise ValueError(f'Insufficient SSD space for approximately {estimated_bytes/2**30:.2f} GiB of imported weights')
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.import-',dir=output.parent))
    weight_map={}
    total_bytes=0
    report.update(config_dir=str(config_dir),requantize_bits=bits,group_size=group_size,
                  conversion='exact native-grid repack' if bits is None else 'lossy MLX affine requantization',
                  tensor_count=len(plans))
    try:
        for ordinal,(name,tensor,shape,expected,target_bits,target_group) in enumerate(plans):
            quantized=target_bits is not None
            is_vector=len(shape)==1
            columns=1 if is_vector else shape[-1]
            rows=math.prod(shape)//columns
            row_perm,col_perm=permutations(name,shape,text) if family!='llama' else (None,None)
            if quantized:
                prefix=name[:-len('.weight')]
                entries=[(name,(*expected[:-1],columns*target_bits//32),'U32'),
                         (prefix+'.scales',(*expected[:-1],columns//target_group),'F32'),
                         (prefix+'.biases',(*expected[:-1],columns//target_group),'F32')]
            else:
                dtype='F16' if tensor.tensor_type==gguf.GGMLQuantizationType.F16 else 'F32'
                entries=[(name,expected,dtype)]
            shard=f'model-{ordinal+1:05d}-of-{len(plans):05d}.safetensors'
            writer=TensorWriter(staging/shard,entries)
            count=max(1,int(chunk_mib*2**20)//max(1,columns*4*8))
            if tensor.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16):
                data=tensor.data.reshape(rows,columns)
            else:
                block,block_bytes=gguf.GGML_QUANT_SIZES[tensor.tensor_type]
                data=tensor.data.reshape(rows,columns//block*block_bytes)
            offsets={key:0 for key,_,_ in entries}
            try:
                for start in range(0,rows,count):
                    end=min(rows,start+count)
                    selection=slice(start,end) if row_perm is None else row_perm[start:end]
                    raw=np.asarray(data[selection])
                    if quantized and bits is None:
                        packed,scales,biases=repack_native(raw,tensor.tensor_type,columns)
                        if col_perm is not None:
                            # Head permutations preserve whole 32-value groups.
                            groups=col_perm.reshape(-1,32)
                            if not np.all(groups==groups[:,:1]+np.arange(32)):
                                raise ValueError('Column permutation crosses quantization groups')
                            group_perm=groups[:,0]//32
                            packed=packed.reshape(len(packed),-1,target_bits)[:,group_perm].reshape(packed.shape)
                            scales=scales[:,group_perm];biases=biases[:,group_perm]
                        arrays=[packed,scales,biases]
                    else:
                        values=dequantize(raw,tensor.tensor_type).reshape(end-start,columns)
                        if col_perm is not None: values=values[:,col_perm]
                        if name.endswith('.A_log'):
                            if np.any(values>=0): raise ValueError('GGUF GDN decay must be negative')
                            values=np.log(-values)
                        if quantized:
                            packed,scales,biases=mx.quantize(mx.array(values.astype(np.float32)),
                                group_size=target_group,bits=target_bits)
                            mx.eval(packed,scales,biases)
                            arrays=[np.array(packed),np.array(scales),np.array(biases)]
                        else:
                            arrays=[values.astype(np.float16 if entries[0][2]=='F16' else np.float32)]
                    for (key,_,_),array in zip(entries,arrays):
                        writer.write(key,offsets[key],array)
                        offsets[key]+=array.nbytes
                    del arrays,raw
                    mx.clear_cache()
            finally:
                writer.close()
            total_bytes+=writer.data_size
            for key,_,_ in entries:weight_map[key]=shard
            print(f'[{ordinal+1}/{len(plans)}] {name}: {tensor.tensor_type.name}',flush=True)
        if len(quantization)>3:config['quantization']=quantization
        (staging/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (staging/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total_bytes},'weight_map':weight_map},indent=2)+'\n')
        for pattern in ('tokenizer*','special_tokens_map.json','added_tokens.json','chat_template*','generation_config.json'):
            for file in config_dir.glob(pattern):
                if file.is_file():shutil.copy2(file,staging/file.name)
        (staging/'gguf-import.json').write_text(json.dumps(report,indent=2)+'\n')
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',nargs='+',type=Path,required=True,help='All GGUF shards, in any order')
    p.add_argument('--inspect',action='store_true')
    p.add_argument('--config-dir',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--requantize-bits',type=int,choices=[2,3,4,6,8])
    p.add_argument('--group-size',type=int,choices=[32,64,128],default=32)
    p.add_argument('--chunk-mib',type=int,default=32)
    a=p.parse_args()
    if a.chunk_mib<1:p.error('--chunk-mib must be positive')
    if a.inspect:
        _,_,report=inspect_files(a.gguf)
    else:
        if not a.config_dir or not a.output:p.error('--config-dir and --output are required for import')
        report=import_model(a.gguf,a.config_dir,a.output,a.requantize_bits,a.group_size,a.chunk_mib)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
