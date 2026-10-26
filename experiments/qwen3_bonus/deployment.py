"""Offline MLX base conversion and separate adapter packaging; never merges adapters."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.qwen3_bonus.common import ROOT,output_path,write_json,sha256
from experiments.qwen3_bonus.models import SOURCES,verify_model


def validate_training_adapter(run: Path,task: str) -> tuple[Path,dict]:
    result=json.loads((run/'training_result.json').read_text())
    if result.get('trained') is not True or result.get('steps',0)<=0:
        raise ValueError('A genuinely trained adapter with recorded optimization steps is required')
    if result.get('stop_reason')=='nonfinite_metrics':
        raise ValueError('Non-finite training run cannot be exported for serving')
    if task=='emotion':adapter=run/'adapter'
    elif task=='response':
        candidates=list((run/'checkpoints').glob('global_step_*/actor/lora_adapter'))
        if not candidates:raise ValueError('No verl PEFT export found')
        adapter=max(candidates,key=lambda p:int(p.parents[1].name.removeprefix('global_step_')))
    else:raise ValueError('Unknown adapter task')
    config=json.loads((adapter/'adapter_config.json').read_text())
    if (config.get('r')!=8 or config.get('lora_alpha')!=16
        or set(config.get('target_modules',[]))!={'q_proj','v_proj'}
        or config.get('task_type')!='CAUSAL_LM' or config.get('modules_to_save')
        or config.get('bias','none')!='none'):
        raise ValueError('Adapter layout differs from the shared base recipe')
    if not (adapter/'adapter_model.safetensors').is_file():raise ValueError('Missing PEFT weights')
    return adapter,config


def export_adapters(base: Path,emotion_run: Path,response_run: Path,output: Path) -> None:
    source=verify_model(base,'base')
    output=output_path(output)
    if output.exists():raise FileExistsError('Adapter bundles are immutable; choose a fresh output')
    checked=[]
    for task,run in (('emotion',emotion_run),('response',response_run)):
        adapter,config=validate_training_adapter(run,task)
        result=json.loads((run/'training_result.json').read_text())
        if result.get('base_revision')!=source['revision'] or result.get('base_inventory_sha256')!=sha256(base/'bonus_source.json'):
            raise ValueError('Training base revision/inventory mismatch')
        # Actual safe tensor parsing is performed before any destination is created.
        from safetensors import safe_open
        with safe_open(str(adapter/'adapter_model.safetensors'),framework='numpy') as weights:
            keys=list(weights.keys())
            if not keys or any(not any(module in key for module in ('.q_proj.lora_','.v_proj.lora_')) for key in keys):
                raise ValueError('Unexpected or empty adapter tensor inventory')
        checked.append((task,run,adapter,config))
    for task,run,adapter,config in checked:
        target=output/f'solace-{task}'
        target.mkdir(parents=True)
        for name in ('adapter_config.json','adapter_model.safetensors'):
            shutil.copyfile(adapter/name,target/name)
        # PEFT records the GPU's local training path. Deploy by verified identity,
        # keeping the training artifact untouched and hashing the exported config.
        write_json(target/'adapter_config.json',{**config,
            'base_model_name_or_path':source['repo_id'],'revision':source['revision']})
        write_json(target/'bonus_adapter.json',{'base_model':source['repo_id'],'base_revision':source['revision'],
            'task':task,'rank':8,'alpha':16,'target_modules':['q_proj','v_proj'],
            'adapter_weights_sha256':sha256(target/'adapter_model.safetensors'),
            'adapter_config_sha256':sha256(target/'adapter_config.json'),
            'training_result_sha256':sha256(run/'training_result.json'),'trained':True})


def convert_base(base: Path,output: Path) -> None:
    source=verify_model(base,'base')
    output=output_path(output)
    if output.exists():raise FileExistsError('Choose a fresh MLX base output')
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    subprocess.run([sys.executable,'-m','mlx_lm.convert','--hf-path',str(base),
        '--mlx-path',str(output),'-q','--q-bits','4','--q-group-size','64'],check=True,env=env)
    files={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'bonus_source.json',{'repo_id':source['repo_id'],'revision':source['revision'],
        'quantization_bits':4,'source_inventory_sha256':sha256(base/'bonus_source.json'),
        'conversion':'mlx_lm.convert, q_bits=4, q_group_size=64; no adapter merged','files':files})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    command=sub.add_parser('convert')
    command.add_argument('--base',type=Path,default=ROOT/'artifacts/models/base')
    command.add_argument('--output',type=Path,default=ROOT/'artifacts/models/base-mlx4')
    command=sub.add_parser('export')
    command.add_argument('--base',type=Path,default=ROOT/'artifacts/models/base')
    command.add_argument('--emotion-run',type=Path,required=True)
    command.add_argument('--response-run',type=Path,required=True)
    command.add_argument('--output',type=Path,default=ROOT/'artifacts/adapters')
    args=parser.parse_args()
    if args.action=='convert':convert_base(args.base,args.output)
    else:export_adapters(args.base,args.emotion_run,args.response_run,args.output)


if __name__=='__main__':main()
