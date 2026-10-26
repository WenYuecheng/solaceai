"""Remove regenerable project downloads only after matching live acceptance."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiments.qwen3_bonus.common import ROOT,sha256,write_json
PROJECT=ROOT.parents[1]
ART='experiments/qwen3_bonus/artifacts'


def checked_target(root: Path,relative: str) -> Path:
    root=root.resolve();target=root/relative
    if target.is_symlink() or target.resolve()==root or not target.resolve().is_relative_to(root):
        raise ValueError('Cleanup target must stay strictly inside the project without symlinks')
    return target


def size(path):
    if path.is_file():return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def plan(root=PROJECT):
    candidates={f'{ART}/{name}':'Regenerable download/install cache' for name in (
        'remote/wheel-relay','wheelhouse','uv-cache','models/base-parts','models/ms-parts')}
    for directory in ('archive/releases',):
        for p in (root/directory).glob('*.tar.gz'):
            candidates[str(p.relative_to(root))]='Obsolete application release package'
    for p in (root/'archive').glob('*before*.tar.gz'):
        candidates[str(p.relative_to(root))]='Obsolete application source snapshot'
    remote=root/ART/'remote'
    for pattern in ('solaceai-app-*.tar.gz','*.whl','upload.tar.gz','update.tar.gz'):
        for p in remote.glob(pattern):candidates[str(p.relative_to(root))]='Training host installer or upload package'
    for name,archive in [('weste-final','training-artifacts.tar.gz'),('fast-final','fast-training-artifacts.tar.gz')]:
        folder=remote/'received'/name
        if (folder/'extracted/artifacts/runs').is_dir():
            candidates[str((folder/archive).relative_to(root))]='Training package with retained extracted records'
    rows=[]
    for relative,reason in sorted(candidates.items()):
        p=checked_target(root,relative)
        if p.exists():rows.append({'path':relative,'bytes':size(p),'reason':reason})
    return rows


def cache_referenced(cache: Path):
    # uv normally clones/hardlinks packages. Preserve cache if an installed
    # environment instead uses symlinks or .pth entries pointing into it.
    for environment in [PROJECT/'.venv',ROOT/'.venv-metal',ROOT/'.venv-tools']:
        for base,dirs,files in os.walk(environment,followlinks=False):
            for name in dirs+files:
                p=Path(base)/name
                if p.is_symlink() and p.resolve().is_relative_to(cache):return True
                if name.endswith('.pth') and p.is_file() and str(cache) in p.read_text(errors='replace'):return True
    return False


def duplicate_ollama_download(raw: Path,store: Path):
    manifest=store/'manifests/registry.ollama.ai/library/solaceai-qwen2.5-7b-ft/latest'
    if not raw.is_file() or not manifest.is_file():return False
    import re
    for layer in json.loads(manifest.read_text()).get('layers',[]):
        digest=layer.get('digest','')
        if layer.get('mediaType')!='application/vnd.ollama.image.model' or not re.fullmatch(r'sha256:[0-9a-f]{64}',digest):continue
        blob=store/'blobs'/digest.replace(':','-')
        if blob.is_file() and blob.stat().st_size==raw.stat().st_size and sha256(raw)==digest[7:] and sha256(blob)==digest[7:]:return True
    return False


def apply():
    from experiments.qwen3_bonus.local_app import validate_candidate
    from experiments.qwen3_bonus.serving import ServingConfig
    config=ServingConfig.from_file(ROOT/'configs/serving.local.json')
    validate_candidate(ROOT/'artifacts/local-deployment/acceptance.json',config)
    import httpx
    with httpx.Client(trust_env=False,timeout=10) as client:
        client.get('http://127.0.0.1:8082/actuator/health').raise_for_status()
    # Raw weights are no longer needed after validated conversion, but keep
    # the source inventory and all training/evaluation evidence.
    rows=plan()
    raw=ROOT/'artifacts/models/base'
    converted=json.loads((Path(config.model_path)/'bonus_source.json').read_text())
    if converted.get('source_inventory_sha256')==sha256(raw/'bonus_source.json'):
        rows.extend({'path':str(p.relative_to(PROJECT)),'bytes':p.stat().st_size,
                     'reason':'Raw base weights superseded by validated MLX4 conversion'}
                    for p in raw.glob('*.safetensors'))
    old=PROJECT/'models/solaceai-qwen2.5-7b-ft/solaceai-qwen2.5-7b-ft-q4_k_m.gguf'
    if duplicate_ollama_download(old,Path.home()/'.ollama/models'):
        rows.append({'path':str(old.relative_to(PROJECT)),'bytes':old.stat().st_size,
                     'reason':'Identical imported Ollama model retained; remove duplicate project download'})
    before=shutil.disk_usage(PROJECT).free
    output=ROOT/'artifacts/local-deployment/cleanup.json'
    report={'started_unix':time.time(),'deleted':[],'retained':[],
            'free_bytes_before':before,'complete':False}
    write_json(output,report)
    for row in rows:
        target=checked_target(PROJECT,row['path'])
        if target.name=='uv-cache' and cache_referenced(target):
            report['retained'].append({**row,'reason':'Installed environment references this cache'});continue
        if target.is_dir():shutil.rmtree(target)
        else:target.unlink()
        report['deleted'].append(row)
        write_json(output,report)
    report.update(complete=True,finished_unix=time.time(),free_bytes_after=shutil.disk_usage(PROJECT).free)
    report['observed_free_bytes_increase']=report['free_bytes_after']-before
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(json.dumps(apply() if args.apply else plan(),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
