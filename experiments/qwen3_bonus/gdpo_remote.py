"""Small source/data deployment and bounded artifact collection over existing SSH.

No password storage, GPU provisioning, model upload or automatic training launch.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time
from experiments.qwen3_bonus.common import ROOT,output_path,sha256,write_json

REMOTE_ROOT='/root/solace-gdpo-20261024'
REMOTE_PYTHON='/root/autodl-tmp/solace-qwen3/experiments/qwen3_bonus/.venv-gpu/bin/python'


def build_bundle(destination: Path,*,experiment_root: Path=ROOT) -> dict:
    destination=output_path(destination)
    if destination.exists():raise FileExistsError('Deployment bundles are immutable')
    destination.parent.mkdir(parents=True,exist_ok=True)
    files=list(experiment_root.glob('*.py'))+list((experiment_root/'tests').glob('test_gdpo*.py'))
    files+=list(experiment_root.glob('requirements-*.lock'))
    data=experiment_root/'artifacts/gdpo-20261024'
    files+=list((data/'data').glob('*'))
    for name in ('calibration.json','calibration_reviewed.json'):
        if (data/name).is_file():files.append(data/name)
    for folder in ('metal-preflight','preflight'):
        files+=list((data/folder).glob('*.json'))
    records={}
    with tarfile.open(destination,'w:gz') as archive:
        for path in sorted(set(files)):
            if not path.is_file() or path.is_symlink():continue
            rel=path.relative_to(experiment_root)
            name='experiments/qwen3_bonus/'+rel.as_posix()
            records[name]=sha256(path);archive.add(path,arcname=name,recursive=False)
        payload=json.dumps(records,sort_keys=True).encode()
        info=tarfile.TarInfo('DEPLOYMENT_MANIFEST.json');info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
    return {'archive':str(destination),'sha256':sha256(destination),'files':records,'bytes':destination.stat().st_size}


def ssh_prefix(host,port,control_path=None):
    if not re.fullmatch(r'[A-Za-z0-9.-]+',host) or not 1<=port<=65535:raise ValueError('Invalid SSH address')
    args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p',str(port)]
    if control_path:args+=['-S',str(control_path)]
    return args+['root@'+host]


def deploy(bundle: Path,host: str,port: int,control_path=None) -> dict:
    prefix=ssh_prefix(host,port,control_path)
    subprocess.run(prefix+[shlex.join(['/bin/mkdir','-p',REMOTE_ROOT])],check=True,timeout=30)
    scp=['scp','-o','BatchMode=yes','-o','ConnectTimeout=10','-P',str(port)]
    if control_path:scp+=['-o','ControlPath='+str(control_path)]
    remote_archive=REMOTE_ROOT+'/deployment.tar.gz'
    subprocess.run(scp+[str(bundle),'root@'+host+':'+remote_archive],check=True,timeout=120)
    script='''import hashlib,json,tarfile
from pathlib import Path
root=Path('/root/solace-gdpo-20261024');archive=root/'deployment.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED
with tarfile.open(archive) as t:
 for m in t.getmembers():
  dest=(root/m.name).resolve()
  if not dest.is_relative_to(root) or not m.isfile():raise ValueError('Unsafe deployment member')
 t.extractall(root)
manifest=json.loads((root/'DEPLOYMENT_MANIFEST.json').read_text())
for name,digest in manifest.items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Deployment hash mismatch')
exp=root/'experiments/qwen3_bonus'
models=exp/'artifacts/models';models.mkdir(parents=True,exist_ok=True)
for kind in ('base','reward'):
 target=Path('/root/autodl-tmp/solace-qwen3/experiments/qwen3_bonus/artifacts/models')/kind
 link=models/kind
 if not link.exists():link.symlink_to(target,target_is_directory=True)
 elif link.resolve()!=target.resolve():raise ValueError('Model link changed')
print(json.dumps({'remote_root':str(root),'files_verified':len(manifest),'gpu_training_started':False}))
'''.replace('EXPECTED',repr(sha256(bundle)))
    result=subprocess.run(prefix+[shlex.join([REMOTE_PYTHON,'-c',script])],check=True,capture_output=True,text=True,timeout=90)
    return json.loads(result.stdout)


def collect(host,port,destination,control_path=None):
    """Copy only small inference adapters/logs; never full FSDP or base weights."""
    destination=output_path(destination)
    if destination.exists():raise FileExistsError('Use a fresh collection file')
    destination.parent.mkdir(parents=True,exist_ok=True)
    script='''import sys,tarfile
from pathlib import Path
root=Path('/root/solace-gdpo-20261024/experiments/qwen3_bonus/artifacts/gdpo-20261024')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for p in sorted(root.rglob('*')):
  if p.is_file() and not p.is_symlink() and 'checkpoints' not in p.relative_to(root).parts and p.stat().st_size<200*1024*1024:
   t.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
'''
    with destination.open('xb') as stream:
        subprocess.run(ssh_prefix(host,port,control_path)+[shlex.join([REMOTE_PYTHON,'-c',script])],
                       stdout=stream,check=True,timeout=120)
    return {'path':str(destination),'sha256':sha256(destination)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['bundle','deploy','collect'])
    parser.add_argument('--host');parser.add_argument('--port',type=int,default=41526)
    parser.add_argument('--control-path',type=Path)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    if args.action=='bundle':result=build_bundle(args.archive)
    else:
        if not args.host:parser.error('--host required')
        result=deploy(args.archive,args.host,args.port,args.control_path) if args.action=='deploy' else collect(
            args.host,args.port,args.archive,args.control_path)
    if args.receipt:write_json(args.receipt,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
