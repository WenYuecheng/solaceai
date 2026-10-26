"""Pinned PERM reward inventories; only explicit download action uses network."""
import argparse
import json
import struct
import math
from pathlib import Path
from experiments.qwen3_bonus.common import ROOT,output_path,sha256,write_json
from experiments.qwen3_bonus.models import SOURCES
CANDIDATES={'1.7b':{'repo_id':'Skywork/Skywork-Reward-V2-Qwen3-1.7B','revision':'e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc'},'0.6b':dict(SOURCES['reward'])}


def verify_model(path):
    path=Path(path).resolve();source=json.loads((path/'bonus_source.json').read_text())
    if not any(all(source.get(k)==v for k,v in c.items()) for c in CANDIDATES.values()):raise ValueError('Unapproved PERM model identity')
    files=source.get('files',{})
    if not all(k in files for k in ('config.json','tokenizer_config.json','tokenizer.json')) or not any(k.endswith('.safetensors') for k in files):raise ValueError('Incomplete model inventory')
    sensitive=lambda p: p.suffix in ('.json','.jinja','.safetensors','.bin','.pt','.pth','.model','.txt','.py') and p.name!='bonus_source.json' and not p.name.endswith('.safetensors.ranges.json')
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and '.cache' not in p.parts and sensitive(p)}
    if not actual<=set(files):raise ValueError('Unlisted loader-sensitive files')
    if any(n.endswith(('.bin','.pt','.pth','.py')) for n in actual):raise ValueError('Unexpected executable or alternative weights')
    for name,digest in files.items():
        target=(path/name).resolve()
        if not target.is_relative_to(path) or not target.is_file() or sha256(target)!=digest:raise ValueError('Model inventory mismatch')
    tensors={}
    weight_names=[n for n in files if n.endswith('.safetensors')]
    index_names=[n for n in files if n.endswith('.safetensors.index.json')]
    if len(index_names)>1 or (len(weight_names)>1 and not index_names):raise ValueError('Ambiguous weight shards')
    for name in weight_names:
        target=path/name
        with target.open('rb') as stream:
            prefix=stream.read(8)
            if len(prefix)!=8:raise ValueError('Truncated safetensors header')
            size=struct.unpack('<Q',prefix)[0]
            if size<2 or size>100000000 or size>target.stat().st_size-8:raise ValueError('Invalid safetensors header')
            header=json.loads(stream.read(size))
        offsets=[]
        for tensor,entry in header.items():
            if tensor=='__metadata__':continue
            if tensor in tensors:raise ValueError('Duplicate tensor across shards')
            begin,end=entry['data_offsets'];shape=entry['shape'];dtype=entry['dtype']
            sizes={'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}
            if dtype not in sizes or any(type(x) is not int or x<0 for x in shape) or type(begin) is not int or type(end) is not int or begin<0 or end-begin!=math.prod(shape)*sizes[dtype]:raise ValueError('Invalid tensor extent')
            offsets.append((begin,end));tensors[tensor]=name
        cursor=0
        for begin,end in sorted(offsets):
            if begin!=cursor:raise ValueError('Safetensors gaps/overlap')
            cursor=end
        if not offsets or cursor!=target.stat().st_size-8-size:raise ValueError('Incomplete safetensors body')
    if index_names:
        mapping=json.loads((path/index_names[0]).read_text()).get('weight_map',{})
        if mapping!=tensors or set(mapping.values())!=set(weight_names):raise ValueError('Incomplete safetensors index')
    elif weight_names!=['model.safetensors']:raise ValueError('Expected unique model.safetensors')
    return {**source,'model_path':str(path),'model_inventory_sha256':sha256(path/'bonus_source.json')}


def download(kind,path):
    from huggingface_hub import snapshot_download
    path=output_path(path)
    if (path/'bonus_source.json').exists():return verify_model(path)
    snapshot_download(**CANDIDATES[kind],local_dir=path,token=False,allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer.model','README.md','LICENSE*'])
    files={str(p.relative_to(path)):sha256(p) for p in sorted(path.rglob('*')) if p.is_file() and '.cache' not in p.parts and p.name!='bonus_source.json' and not p.name.endswith('.safetensors.ranges.json')}
    write_json(path/'bonus_source.json',{**CANDIDATES[kind],'files':files,'downloaded':True})
    return verify_model(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['verify','download']);parser.add_argument('--kind',choices=CANDIDATES,default='1.7b');parser.add_argument('--path',type=Path,required=True);a=parser.parse_args()
    print(json.dumps(download(a.kind,a.path) if a.action=='download' else verify_model(a.path),indent=2))
