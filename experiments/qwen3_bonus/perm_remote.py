"""Bounded, verified PERM deployment; never provisions GPUs or starts training."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
from experiments.qwen3_bonus.common import ROOT, sha256, write_json
from experiments.qwen3_bonus.gdpo_remote import ssh_prefix, REMOTE_PYTHON

REMOTE_ROOT='/root/solace-perm-20261024'
MAX_FILE=2*1024*1024
MAX_TOTAL=64*1024*1024
MANIFEST='DEPLOYMENT_MANIFEST.json'


def build_bundle(destination: Path, *, experiment_root: Path=ROOT) -> dict:
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Deployment bundles are immutable')
    root=Path(experiment_root)
    selected={}
    patterns=['*.py','tests/test_perm*.py','tests/test_gdpo*.py','requirements*.lock','*.md','docs/PERM_GRPO.md',
              'artifacts/perm-grpo-20261024/data/**/*','artifacts/gdpo-20261024/data/**/*',
              'artifacts/gdpo-20261024/calibration*.json','artifacts/perm-grpo-20261024/*preflight*/*.json',
              'artifacts/perm-grpo-20261024/*preflight*.json']
    for pattern in patterns:
        for p in root.glob(pattern):
            if p.is_dir():continue
            selected['experiments/qwen3_bonus/'+p.relative_to(root).as_posix()]=p
    for pattern in ['artifacts/perm-grpo-20261024/*calibration*/**/*.json','artifacts/perm-grpo-20261024/*calibration*.json',
                    'artifacts/perm-grpo-20261024/local-final/calibration-*.json']:
        for p in root.glob(pattern):
            if p.is_file():selected['experiments/qwen3_bonus/artifacts/perm-grpo-20261024/local-evidence/'+p.relative_to(root/'artifacts/perm-grpo-20261024').as_posix()]=p
    payloads={};total=0
    for name,p in sorted(selected.items()):
        if p.is_symlink() or any(parent.is_symlink() for parent in p.parents):raise ValueError('Symlink source forbidden')
        if p.stat().st_size>MAX_FILE:raise ValueError('Deployment file too large')
        data=p.read_bytes();total+=len(data)
        if total>MAX_TOTAL:raise ValueError('Deployment bundle too large')
        payloads[name]=data
    if any('local-evidence/' in n for n in payloads):
        payloads['experiments/qwen3_bonus/artifacts/perm-grpo-20261024/local-evidence/README.txt']=b'Local CPU evidence only. Host-bound paths; not remote launch-ready.\n'
    records={name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as t:
        for name,data in [*payloads.items(),(MANIFEST,json.dumps(records,sort_keys=True).encode())]:
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;t.addfile(info,io.BytesIO(data))
    return {'archive':str(destination),'sha256':sha256(destination),'files':records,'bytes':destination.stat().st_size}


def safe_extract(archive: Path, root: Path) -> dict:
    """Validate the complete archive before writing any member."""
    root=Path(root)
    if root.is_symlink():raise ValueError('Symlink extraction root')
    data={};total=0
    with tarfile.open(archive) as t:
        for m in t:
            p=PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or p.as_posix()!=m.name or m.name in data:
                raise ValueError('Unsafe or duplicate archive member')
            if m.size>MAX_FILE:raise ValueError('Oversized archive member')
            total+=m.size
            if total>MAX_TOTAL:raise ValueError('Oversized archive')
            target=root/m.name
            if any(q.is_symlink() for q in [target,*target.parents]):raise ValueError('Symlink ancestor')
            data[m.name]=t.extractfile(m).read()
    try:manifest=json.loads(data.pop(MANIFEST))
    except (KeyError,ValueError) as exc:raise ValueError('Missing or invalid manifest') from exc
    if not isinstance(manifest,dict) or set(manifest)!=set(data):raise ValueError('Manifest membership mismatch')
    for name,payload in data.items():
        if hashlib.sha256(payload).hexdigest()!=manifest[name]:raise ValueError('Hash mismatch')
    for name,payload in data.items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    (root/MANIFEST).write_text(json.dumps(manifest,sort_keys=True))
    return manifest


def deploy(bundle: Path, host: str, port: int, control_path=None) -> dict:
    prefix=ssh_prefix(host,port,control_path)
    bundle=Path(bundle)
    subprocess.run(prefix+[shlex.join(['/bin/mkdir','-p',REMOTE_ROOT])],check=True,timeout=30)
    scp=['scp','-o','BatchMode=yes','-o','ConnectTimeout=10','-P',str(port)]
    if control_path:scp+=['-o','ControlPath='+str(control_path)]
    subprocess.run(scp+[str(bundle.resolve()),'root@'+host+':'+REMOTE_ROOT+'/deployment.tar.gz'],check=True,timeout=120)
    import inspect
    script='import hashlib,json,tarfile\nfrom pathlib import Path,PurePosixPath\n'+f'MAX_FILE={MAX_FILE}\nMAX_TOTAL={MAX_TOTAL}\nMANIFEST={MANIFEST!r}\n'
    script+=inspect.getsource(safe_extract)
    script+=f'''\nroot=Path({REMOTE_ROOT!r})
archive=root/'deployment.tar.gz'
if hashlib.sha256(archive.read_bytes()).hexdigest()!={sha256(bundle)!r}:raise ValueError('Archive hash mismatch')
manifest=safe_extract(archive,root)
exp=root/'experiments/qwen3_bonus'
models=exp/'artifacts/models'
models.mkdir(parents=True,exist_ok=True)
for kind in ('base','reward'):
 target=Path('/root/autodl-tmp/solace-qwen3/experiments/qwen3_bonus/artifacts/models')/kind
 link=models/kind
 if link.is_symlink():
  if link.resolve()!=target.resolve():raise ValueError('Changed model link')
 elif link.exists():raise ValueError('Existing model destination')
 else:link.symlink_to(target,target_is_directory=True)
(exp/'artifacts/perm-grpo-20261024/models/reward-1.7b').mkdir(parents=True,exist_ok=True)
print(json.dumps({{'remote_root':str(root),'files_verified':len(manifest),'manifest':manifest,'gpu_training_started':False}}))
'''
    result=subprocess.run(prefix+[shlex.join([REMOTE_PYTHON,'-c',script])],check=True,capture_output=True,text=True,timeout=90)
    receipt=json.loads(result.stdout);receipt['archive_sha256']=sha256(bundle)
    return receipt


def collection_allowed(path: Path, root: Path) -> bool:
    rel=path.relative_to(root);parts=rel.parts
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():return False
    if any(any(word in part.lower() for word in ('secret','credential','token','password','checkpoint','optimizer','fsdp')) or part.lower() in ('models','base','reward','.cache') for part in parts):return False
    adapter=parts[0]=='adapter_snapshots' or parts[:2]==('run','adapter_snapshots')
    if path.stat().st_size>(100*1024*1024 if adapter else 2*1024*1024):return False
    if adapter:return path.name in ('adapter_model.safetensors','adapter_config.json','README.md','snapshot.json')
    return path.suffix.lower() in ('.json','.jsonl','.log','.txt','.csv','.md')


def collect(host,port,destination,control_path=None):
    prefix=ssh_prefix(host,port,control_path);destination=Path(destination)
    if destination.exists():raise FileExistsError('Use fresh collection archive')
    import inspect
    script='import sys,tarfile\nfrom pathlib import Path\n'+inspect.getsource(collection_allowed)
    script+=f'''\nroot=Path({REMOTE_ROOT+'/experiments/qwen3_bonus/artifacts/perm-grpo-20261024'!r})
selected=[];total=0
for p in sorted(root.rglob('*')):
 if collection_allowed(p,root):
  total+=p.stat().st_size
  if total>500*1024*1024:raise ValueError('Collection exceeds total limit')
  selected.append(p)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for p in selected:t.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
'''
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        with destination.open('xb') as stream:
            subprocess.run(prefix+[shlex.join([REMOTE_PYTHON,'-c',script])],stdout=stream,check=True,timeout=120)
    except BaseException:
        destination.unlink(missing_ok=True);raise
    return {'path':str(destination),'sha256':sha256(destination)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['bundle','deploy','collect'])
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--host')
    parser.add_argument('--port',type=int,default=41526);parser.add_argument('--control-path',type=Path)
    parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    if args.action=='bundle':result=build_bundle(args.archive)
    else:
        if not args.host:parser.error('--host required')
        result=deploy(args.archive,args.host,args.port,args.control_path) if args.action=='deploy' else collect(args.host,args.port,args.archive,args.control_path)
    if args.receipt:write_json(args.receipt,result)
    print(json.dumps(result))

if __name__=='__main__':main()
