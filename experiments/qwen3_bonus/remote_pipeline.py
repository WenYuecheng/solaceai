"""Bounded single-instance runner: calibrate, pilot, train, plot, archive, then power off.
No SSH credentials are used here. Explicit --shutdown invokes AutoDL's documented
/usr/bin/shutdown after artifacts and logs have been flushed.
"""
from __future__ import annotations
import argparse,datetime,json,os,re,signal,subprocess,sys,threading,time,traceback,tarfile
from pathlib import Path
from experiments.qwen3_bonus.common import ROOT,write_json,sha256
from experiments.qwen3_bonus.budget import RentalBudget,parse_start

def completed_emotion(run: Path) -> bool:
    """Only skip an explicitly finished local classifier with exported weights."""
    try:
        result=json.loads((run/'training_result.json').read_text())
        adapter=Path(result['adapter']).resolve()
        return (result.get('trained') is True and result.get('status')=='completed'
            and result.get('task')=='four_class_emotion' and result.get('steps',0)>0
            and adapter.is_relative_to(run.resolve())
            and (adapter/'adapter_config.json').is_file()
            and (adapter/'adapter_model.safetensors').stat().st_size>8)
    except (OSError,ValueError,KeyError,TypeError):
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rate',required=True,type=float)
    parser.add_argument('--started-at',required=True)
    parser.add_argument('--cap',type=float,default=10)
    parser.add_argument('--shutdown',action='store_true')
    parser.add_argument('--skip-emotion',action='store_true')
    parser.add_argument('--run-suffix',default='')
    args=parser.parse_args()
    if args.run_suffix and not re.fullmatch(r'[a-z0-9-]{1,32}',args.run_suffix):
        parser.error('Run suffix must be 1..32 lowercase letters, numbers or hyphens')
    if args.skip_emotion and not completed_emotion(ROOT/'artifacts/runs/emotion'):
        parser.error('Cannot skip classifier without its completed result and local adapter')
    budget=RentalBudget(args.rate,args.cap,parse_start(args.started_at),60)
    state=ROOT/'artifacts/remote';state.mkdir(parents=True,exist_ok=True)
    (state/'archive-downloaded.ok').unlink(missing_ok=True)
    log=(state/'pipeline-events.jsonl').open('a',buffering=1)
    python=ROOT/'.venv-gpu/bin/python'
    project=ROOT.parents[1]
    def record(stage,status,**extra):
        value={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':stage,'status':status,
            'estimated_total_yuan':round(budget.cost(time.time()),4),'cap_yuan':args.cap,**extra}
        log.write(json.dumps(value,ensure_ascii=False)+'\n');write_json(state/'pipeline-status.json',value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
    def execute(stage,command,timeout=None):
        record(stage,'running',command=command)
        with (state/f'{stage}.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=project,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True,env={**os.environ,'WANDB_DISABLED':'true','TOKENIZERS_PARALLELISM':'false'})
            limit=min(budget.deadline(),time.time()+timeout if timeout else budget.deadline())
            while process.poll() is None:
                if time.time()>=limit:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise TimeoutError(f'{stage} exceeded its recorded budget')
                time.sleep(2)
            if process.returncode:raise RuntimeError(f'{stage} exited {process.returncode}; inspect {stage}.log')
        record(stage,'completed')
    tensorboard=None;sampler=None;finished=threading.Event();success=False
    try:
        record('prepare','waiting_for_environment_and_weights')
        prep_deadline=budget.started_at+1.5*3600/args.rate # bounded setup; total rental cap is still authoritative
        while True:
            assets=all((ROOT/f'artifacts/models/{kind}/bonus_source.json').exists() for kind in ('base','reward'))
            installed=any((python.parent.parent/'lib').glob('python*/site-packages/verl-0.5.0.dist-info/METADATA'))
            if assets and installed:break
            if time.time()>prep_deadline:raise TimeoutError('Environment/model preparation exceeded first-yuan budget')
            time.sleep(10)
        uv=sys.executable
        tools=Path('/root/autodl-tmp/solace-qwen3/tools')
        env_python='/root/miniconda3/bin/python'
        os.environ['PYTHONPATH']=str(tools)+os.pathsep+str(project)
        os.environ['UV_CACHE_DIR']='/root/autodl-tmp/solace-qwen3/cache'
        os.environ['UV_INDEX_URL']='https://repo.huaweicloud.com/repository/pypi/simple'
        execute('observability-install',[env_python,'-m','uv','pip','install','--python',str(python),
            '--require-hashes','-r',str(ROOT/'requirements-observability.lock')],timeout=240)
        execute('environment-check',[str(python),'-c',
            "import torch,verl,vllm,peft,bitsandbytes,matplotlib,tensorboard; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(),torch.cuda.is_bf16_supported())"],timeout=120)
        tensorboard=subprocess.Popen([str(python.parent/'tensorboard'),'--logdir',str(ROOT/'artifacts/runs'),
            '--host','127.0.0.1','--port','6006','--reload_interval','15'],cwd=project,
            stdout=(state/'tensorboard.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
        def telemetry():
            target=state/'gpu-telemetry.csv'
            with target.open('w',buffering=1) as stream:
                stream.write('utc,elapsed_seconds,utilization_percent,memory_mib,power_w,estimated_yuan\n')
                while not finished.is_set():
                    try:
                        value=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                        values=[float(v.strip()) for v in value.split(',')]
                        stream.write(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()},{time.time()-budget.started_at:.2f},{values[0]},{values[1]},{values[2]},{budget.cost(time.time()):.5f}\n')
                    except Exception:pass
                    finished.wait(15)
        sampler=threading.Thread(target=telemetry,daemon=True);sampler.start()
        reviewed=ROOT/'artifacts/calibration/model-reviewed.json'
        scored=ROOT/'artifacts/calibration/model-scored.json'
        if scored.exists():
            previous=json.loads(scored.read_text())
            if (previous.get('sheet_sha256')!=sha256(reviewed) or
                previous.get('model_inventory_sha256')!=sha256(ROOT/'artifacts/models/reward/bonus_source.json')):
                raise ValueError('Existing reward calibration does not match frozen inputs')
            record('reward-calibration','reused_verified_inputs')
        else:
            execute('reward-calibration',[str(python),'-m','experiments.qwen3_bonus.calibration','score',
                '--sheet',str(reviewed),'--output',str(scored),'--allow-model-review'],timeout=360)
        execute('reward-gate',[str(python),'-m','experiments.qwen3_bonus.calibration','check',
            '--report',str(scored),'--allow-model-review'],timeout=30)
        shared=['--hourly-rate',str(args.rate),'--rental-started-at',args.started_at,'--cap-yuan',str(args.cap),
            '--calibration',str(scored),'--allow-model-calibration','--defer-local-compatibility']
        suffix=f'-{args.run_suffix}' if args.run_suffix else ''
        execute('pilot',[str(python),'-m','experiments.qwen3_bonus.training','grpo',*shared,
            '--run-dir',str(ROOT/f'artifacts/runs/pilot{suffix}'),'--max-steps','2','--stage-cap-yuan','1'])
        # LoRA/Qwen experimental software compatibility can fail. Never hide that
        # by silently swapping to a different trainer/algorithm.
        execute('response-grpo',[str(python),'-m','experiments.qwen3_bonus.training','grpo',*shared,
            '--run-dir',str(ROOT/f'artifacts/runs/response{suffix}'),'--max-steps','32','--stage-cap-yuan','4'])
        if args.skip_emotion:
            record('emotion-sft','reused_completed_training')
        else:
            execute('emotion-sft',[str(python),'-m','experiments.qwen3_bonus.training','sft',*shared,
                '--run-dir',str(ROOT/'artifacts/runs/emotion'),'--stage-cap-yuan','3'])
        success=True;record('training','completed',production_enabled=False)
    except BaseException as exc:
        record('pipeline','failed',error_type=type(exc).__name__,message=str(exc),production_enabled=False)
        traceback.print_exc()
    finally:
        finished.set()
        if sampler:sampler.join(timeout=7)
        # Retain adapters, metrics, event logs and figures; exclude multi-GB FSDP
        # state, environments, base weights and caches from the download bundle.
        archive=state/'training-artifacts.tar.gz'
        with tarfile.open(archive,'w:gz') as bundle:
            for folder in (ROOT/'artifacts/runs',ROOT/'artifacts/calibration',state):
                for path in sorted(folder.rglob('*')):
                    if not path.is_file() or path==archive or path.suffix=='.sock':continue
                    if 'checkpoints' in path.parts:continue
                    if path.stat().st_size>200*1024*1024:continue
                    bundle.add(path,arcname=str(path.relative_to(ROOT)))
        record('archive','saved',archive=str(archive),training_completed=success)
        log.flush();os.sync()
        if args.shutdown:
            # The local non-LLM watcher acknowledges a downloaded archive. Bound
            # the grace so a disconnected laptop cannot leave the GPU billing.
            grace=300 if success else 600
            until=time.monotonic()+min(grace,max(0,budget.started_at+args.cap*3600/args.rate-time.time()-30))
            while time.monotonic()<until and not (state/'archive-downloaded.ok').exists():time.sleep(3)
            record('shutdown','requested',mechanism='/usr/bin/shutdown',billing_stop_unconfirmed=True)
            log.close();os.sync()
            subprocess.run(['/bin/bash','/usr/bin/shutdown'],check=False,timeout=30)
        else:
            record('shutdown','manual_action_required')

if __name__=='__main__':main()
