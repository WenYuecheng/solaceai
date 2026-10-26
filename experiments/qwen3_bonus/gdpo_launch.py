"""Fail-closed single GDPO rental launch. No CPU-only server is upgraded here."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time
from experiments.qwen3_bonus.common import ROOT,output_path,read_jsonl,sha256,write_json
from experiments.qwen3_bonus.models import SOURCES,verify_model
from experiments.qwen3_bonus.gdpo_runtime import GDPOBudget
from experiments.qwen3_bonus.gdpo_platform import verify_platform_script,AUTODL_SHUTDOWN_SHA256


def verify_frozen_data(path: Path) -> dict:
    manifest=json.loads((path/'dataset_manifest.json').read_text())
    if manifest.get('frozen') is not True or manifest.get('license',{}).get('training_permitted') is not True:
        raise ValueError('Frozen licensed GDPO data required')
    for name,entry in manifest['files'].items():
        file=(path/name).resolve()
        if not file.is_relative_to(path.resolve()) or sha256(file)!=entry['sha256']:
            raise ValueError('Frozen GDPO artifact fingerprint changed')
    for split,count in [('train',256),('dev',32),('test',80)]:
        if len(read_jsonl(path/f'response_{split}.jsonl'))!=count:raise ValueError('Frozen split count mismatch')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'artifacts/models/base')
    parser.add_argument('--reward-model',type=Path,default=ROOT/'artifacts/models/reward')
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/gdpo-20261024/data')
    parser.add_argument('--calibration',type=Path,default=ROOT/'artifacts/gdpo-20261024/calibration.json')
    parser.add_argument('--preflight',type=Path,default=ROOT/'artifacts/gdpo-20261024/metal-preflight/report.json')
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/gdpo-20261024/run')
    parser.add_argument('--hourly-rate',type=float,required=True)
    parser.add_argument('--billing-quantum-seconds',type=int,required=True)
    parser.add_argument('--rental-started-at',required=True)
    parser.add_argument('--max-steps',type=int,default=64)
    parser.add_argument('--gpu-memory',type=float,default=.45)
    parser.add_argument('--shutdown-script-sha256',required=True)
    parser.add_argument('--wait',action='store_true',help='Stay attached to detached supervisor until completion')
    args=parser.parse_args()
    from experiments.qwen3_bonus.budget import parse_start
    from experiments.qwen3_bonus.gdpo_watchdog import launch_watchdog
    from experiments.qwen3_bonus.gdpo_entry import gdpo_overrides
    # Validate numeric recipe and billing input before importing the GPU stack.
    gdpo_overrides(args.base,args.reward_model,args.data,args.run,args.calibration,args.max_steps,gpu_memory=args.gpu_memory)
    budget=GDPOBudget(args.hourly_rate,parse_start(args.rental_started_at),args.billing_quantum_seconds)
    budget.cost(time.time())
    if platform.system()!='Linux':raise RuntimeError('Actual training requires prepared Linux/NVIDIA instance')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('No single exposed NVIDIA GPU; CPU preparation does not start training')
    if args.shutdown_script_sha256!=AUTODL_SHUTDOWN_SHA256:
        raise ValueError('Platform shutdown file differs from inspected provider script')
    verify_platform_script(Path('/usr/bin/shutdown'),args.shutdown_script_sha256)
    verify_model(args.base,'base');verify_model(args.reward_model,'reward');verify_frozen_data(args.data)
    calibration=json.loads(args.calibration.read_text());preflight=json.loads(args.preflight.read_text())
    if (calibration.get('gate',{}).get('passed') is not True
        or calibration.get('reward_model_revision')!=SOURCES['reward']['revision']
        or calibration.get('model_inventory_sha256')!=sha256(args.reward_model/'bonus_source.json')):
        raise ValueError('Matching real reward calibration must pass before launch')
    if (preflight.get('compatibility_passed') is not True or preflight.get('genuine_inference') is not True
        or preflight.get('static',{}).get('manifests_sha256',{}).get('base')!=sha256(args.base/'bonus_source.json')):
        raise ValueError('Matching genuine local Metal preflight required')
    run=output_path(args.run)
    if run.exists() and any(run.iterdir()):raise ValueError('Use a fresh run; no implicit second training')
    run.mkdir(parents=True,exist_ok=True)
    identity={'base_revision':SOURCES['base']['revision'],'base_inventory_sha256':sha256(args.base/'bonus_source.json'),
        'dataset_manifest_sha256':sha256(args.data/'dataset_manifest.json'),
        'calibration_sha256':sha256(args.calibration),
        'source_sha256':{p.name:sha256(p) for p in sorted(ROOT.glob('gdpo*.py'))}}
    write_json(run/'launch.json',{'budget':asdict(budget),'identity':identity,'seed':3407,'resume':False,
        'actual_algorithm':'gdpo','created_at':time.time(),'max_steps':args.max_steps,'production_enabled':False})
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true',
                      TOKENIZERS_PARALLELISM='false',HYDRA_FULL_ERROR='1')
    command=[sys.executable,'-m','experiments.qwen3_bonus.gdpo_entry','worker',
        '--base',str(args.base.resolve()),'--reward-model',str(args.reward_model.resolve()),
        '--data',str(args.data.resolve()),'--calibration',str(args.calibration.resolve()),
        '--run',str(run),'--max-steps',str(args.max_steps),'--gpu-memory',str(args.gpu_memory)]
    shutdown=[sys.executable,'-m','experiments.qwen3_bonus.gdpo_platform','--execute',
              '--sha256',args.shutdown_script_sha256]
    supervisor=launch_watchdog(run,command,budget,shutdown)
    print(json.dumps({'watchdog_pid':supervisor.pid,'run':str(run),'budget_cap_yuan':20,'actual_algorithm':'gdpo'}))
    if args.wait:raise SystemExit(supervisor.wait())


if __name__=='__main__':main()
