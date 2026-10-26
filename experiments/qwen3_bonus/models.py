"""Download public, pinned weights into this experiment only; no inference or paid APIs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from experiments.qwen3_bonus.common import ROOT,output_path,write_json,sha256

SOURCES={
    'base':{'repo_id':'Qwen/Qwen3-4B-Instruct-2507','revision':'cdbee75f17c01a7cc42f958dc650907174af0554'},
    'reward':{'repo_id':'Skywork/Skywork-Reward-V2-Qwen3-0.6B','revision':'8c14a4e9e6321deaf572544339b16b8d6bbe8886'},
}


def verify_model(path: Path,kind: str) -> dict:
    source=json.loads((path/'bonus_source.json').read_text())
    if any(source.get(k)!=v for k,v in SOURCES[kind].items()):
        raise ValueError(f'{kind} identity/revision mismatch')
    hashes=source.get('files',{})
    if 'config.json' not in hashes or not any(k.endswith('.safetensors') for k in hashes):
        raise ValueError('Incomplete model inventory')
    for name,expected in hashes.items():
        target=(path/name).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file() or sha256(target)!=expected:
            raise ValueError(f'Model integrity mismatch: {name}')
    return source


def download(kind: str) -> Path:
    from huggingface_hub import snapshot_download
    target=output_path(ROOT/'artifacts/models'/kind)
    if (target/'bonus_source.json').exists():
        verify_model(target,kind);return target
    target.mkdir(parents=True,exist_ok=True)
    source=SOURCES[kind]
    snapshot_download(**source,local_dir=target,cache_dir=ROOT/'artifacts/hf-cache',token=False,allow_patterns=[
        '*.safetensors','*.json','*.jinja','tokenizer.model','merges.txt','vocab.txt','README.md','LICENSE*'],
        max_workers=2)
    files={str(p.relative_to(target)):sha256(p) for p in sorted(target.rglob('*'))
        if p.is_file() and '.cache' not in p.parts and p.name!='bonus_source.json'}
    write_json(target/'bonus_source.json',{**source,'files':files,'downloaded':True})
    verify_model(target,kind)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['download','verify','sources'])
    parser.add_argument('--kind',choices=list(SOURCES),default='reward')
    args=parser.parse_args()
    if args.action=='sources':print(json.dumps(SOURCES,indent=2));return
    result=download(args.kind) if args.action=='download' else verify_model(ROOT/'artifacts/models'/args.kind,args.kind)
    print(result)


if __name__=='__main__':main()
