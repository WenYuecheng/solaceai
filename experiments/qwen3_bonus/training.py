"""Pure recipe helpers and explicit, budgeted training entry points (no import-time ML)."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from experiments.qwen3_bonus.budget import RentalBudget,parse_start
from experiments.qwen3_bonus.common import ROOT,output_path,read_jsonl,write_json,sha256


@dataclass
class StepMonitor:
    deadline: float
    flat_steps: int=0
    longest_step: float=0
    mean_reward: float=0
    mean_group_std: float=0

    def record_reward_groups(self,scores: list[float],group_size: int=2):
        if group_size<2 or not scores or len(scores)%group_size:
            raise ValueError('Reward groups must contain at least two candidates each')
        if not all(math.isfinite(v) for v in scores):
            raise ValueError('Non-finite rewards')
        stds=[statistics.pstdev(scores[i:i+group_size]) for i in range(0,len(scores),group_size)]
        self.mean_reward=statistics.mean(scores)
        self.mean_group_std=statistics.mean(stds)
        self.flat_steps=self.flat_steps+1 if max(stds)<1e-8 else 0

    def stop_reason(self,metrics: dict,now: float | None=None) -> str | None:
        if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in metrics.values()):
            return 'nonfinite_metrics'
        self.longest_step=max(self.longest_step,float(metrics.get('timing_s/step',0)))
        if (time.time() if now is None else now)+self.longest_step>=self.deadline:
            return 'budget_deadline'
        if self.flat_steps>=8:
            return 'zero_group_variance'
        return None


def completion_only_example(tokenizer,row: dict,max_length: int=256) -> dict:
    prefix=tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(row['prompt']+row['completion'],tokenize=True,add_generation_prompt=False)
    if full[:len(prefix)]!=prefix or len(full)<=len(prefix):
        raise ValueError('Chat template prefix mismatch or empty completion; loss mask is unsafe')
    if len(full)-len(prefix)>=max_length:
        raise ValueError('Completion exceeds max_length')
    labels=[-100]*len(prefix)+full[len(prefix):]
    return {'input_ids':full[-max_length:],'attention_mask':[1]*min(len(full),max_length),'labels':labels[-max_length:]}


def parquet_records(rows: list[dict]) -> list[dict]:
    return [{'data_source':'psydt','prompt':r['prompt'],
             'reward_model':{'style':'rule','ground_truth':''},
             'extra_info':{'id':r['id'],'dialogue_id':r['dialogue_id'],'messages':r['prompt']}}
            for r in rows]


def prepare_parquet(data_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    # Explicit schema prevents all-empty fields and nested Arrow inference from drifting.
    message=pa.struct([('role',pa.string()),('content',pa.string())])
    schema=pa.schema([('data_source',pa.string()),('prompt',pa.list_(message)),
        ('reward_model',pa.struct([('style',pa.string()),('ground_truth',pa.string())])),
        ('extra_info',pa.struct([('id',pa.string()),('dialogue_id',pa.string()),('messages',pa.list_(message))]))])
    manifest=data_dir/'parquet_manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if all(sha256(data_dir/name)==value for name,value in previous['files'].items()):return
        raise ValueError('Parquet/source fingerprint changed; use a fresh preparation directory')
    hashes={}
    for split in ('train','eval'):
        records=parquet_records(read_jsonl(data_dir/f'response_{split}.jsonl'))
        pq.write_table(pa.Table.from_pylist(records,schema=schema),output_path(data_dir/f'response_{split}.parquet'))
        for suffix in ('jsonl','parquet'):
            name=f'response_{split}.{suffix}';hashes[name]=sha256(data_dir/name)
    write_json(manifest,{'files':hashes,'reference_in_reward_ground_truth':False})


def verl_overrides(base: Path,rm: Path,data: Path,run: Path,max_steps: int=32) -> list[str]:
    if not 1<=max_steps<=32:
        raise ValueError('This budget recipe supports 1..32 optimization steps')
    pairs={
        'algorithm.adv_estimator':'grpo','algorithm.norm_adv_by_std_in_grpo':'True',
        'algorithm.use_kl_in_reward':'False',
        'data.train_files':json.dumps(str(data/'response_train.parquet')),
        'data.val_files':json.dumps(str(data/'response_eval.parquet')),
        'data.train_batch_size':1,'data.val_batch_size':1,'data.max_prompt_length':768,
        'data.max_response_length':256,'data.filter_overlong_prompts':'True','data.truncation':'error',
        'data.dataloader_num_workers':0,'data.shuffle':'True',
        'actor_rollout_ref.model.path':json.dumps(str(base)),
        'actor_rollout_ref.model.lora_rank':8,'actor_rollout_ref.model.lora_alpha':16,
        'actor_rollout_ref.model.target_modules':'[q_proj,v_proj]',
        'actor_rollout_ref.model.use_remove_padding':'False',
        'actor_rollout_ref.model.enable_gradient_checkpointing':'True',
        'actor_rollout_ref.model.trust_remote_code':'False',
        '+actor_rollout_ref.model.override_config._attn_implementation':'sdpa',
        'actor_rollout_ref.actor.strategy':'fsdp',
        'actor_rollout_ref.actor.optim.lr':'5e-6','actor_rollout_ref.actor.ppo_mini_batch_size':1,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':1,
        'actor_rollout_ref.actor.ppo_epochs':1,'actor_rollout_ref.actor.use_kl_loss':'True',
        'actor_rollout_ref.actor.kl_loss_coef':.02,'actor_rollout_ref.actor.kl_loss_type':'low_var_kl',
        'actor_rollout_ref.actor.policy_loss.loss_mode':'vanilla',
        'actor_rollout_ref.actor.loss_agg_mode':'seq-mean-token-mean',
        'actor_rollout_ref.actor.clip_ratio':.2,'actor_rollout_ref.actor.clip_ratio_low':.2,
        'actor_rollout_ref.actor.clip_ratio_high':.2,'actor_rollout_ref.actor.entropy_coeff':0,
        'actor_rollout_ref.actor.use_torch_compile':'False',
        'actor_rollout_ref.actor.fsdp_config.param_offload':'True',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload':'True',
        '+actor_rollout_ref.actor.fsdp_config.model_dtype':'bfloat16',
        'actor_rollout_ref.rollout.name':'vllm','actor_rollout_ref.rollout.mode':'sync',
        'actor_rollout_ref.rollout.n':2,'actor_rollout_ref.rollout.temperature':.8,
        'actor_rollout_ref.rollout.tensor_model_parallel_size':1,
        'actor_rollout_ref.rollout.gpu_memory_utilization':.65,
        'actor_rollout_ref.rollout.max_model_len':1024,
        'actor_rollout_ref.rollout.max_num_batched_tokens':1024,
        'actor_rollout_ref.rollout.max_num_seqs':2,
        'actor_rollout_ref.rollout.enable_chunked_prefill':'False',
        'actor_rollout_ref.rollout.enforce_eager':'True',
        'actor_rollout_ref.rollout.free_cache_engine':'True',
        'actor_rollout_ref.rollout.load_format':'safetensors',
        'actor_rollout_ref.rollout.layered_summon':'True',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':1,
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu':1,
        'reward_model.enable':'False','reward_model.launch_reward_fn_async':'False',
        'custom_reward_function.path':json.dumps(str(ROOT/'rewards.py')),
        'custom_reward_function.name':'compute_score',
        '+custom_reward_function.reward_kwargs.reward_model_path':json.dumps(str(rm)),
        'trainer.n_gpus_per_node':1,'trainer.nnodes':1,'trainer.logger':'[console]',
        'trainer.project_name':'solaceai_bonus','trainer.experiment_name':'qwen3_grpo',
        'trainer.total_training_steps':max_steps,'trainer.total_epochs':1,
        'trainer.save_freq':4,'trainer.test_freq':-1,'trainer.val_before_train':'False',
        'trainer.max_actor_ckpt_to_keep':1,'trainer.resume_mode':'disable',
        'trainer.default_local_dir':json.dumps(str(run/'checkpoints')),
        'trainer.rollout_data_dir':json.dumps(str(run/'rollouts')),
        'ray_init.num_cpus':4,
    }
    return [f'{key}={value}' for key,value in pairs.items()]


def validate_assets(data: Path,base: Path,rm: Path,gate: Path, *, allow_model_review=False) -> dict:
    from experiments.qwen3_bonus.rewards import calibration_gate
    report=json.loads(gate.read_text())
    if not calibration_gate(report['pairs'],allow_model_review=allow_model_review)['passed']:
        raise ValueError('Reward calibration needs 20 reviewed pairs and >=15 correct; model review needs explicit authorization')
    if report.get('reward_model_revision')!='8c14a4e9e6321deaf572544339b16b8d6bbe8886':
        raise ValueError('Calibration reward revision mismatch')
    manifest=json.loads((data/'dataset_manifest.json').read_text())
    if manifest.get('response',{}).get('license',{}).get('training_permitted') is not True:
        raise ValueError('Dataset license must explicitly permit this training use')
    # Freeze exact data used by training, including provenance. Preparation writes this map.
    for name in ('classification_train.jsonl','response_train.jsonl','response_eval.jsonl'):
        if name not in manifest.get('output_hashes',{}) or sha256(data/name)!=manifest['output_hashes'][name]:
            raise ValueError(f'Missing or stale dataset hash: {name}')
    from experiments.qwen3_bonus.models import verify_model
    verify_model(base,'base')
    verify_model(rm,'reward')
    if report.get('model_inventory_sha256')!=sha256(rm/'bonus_source.json'):
        raise ValueError('Calibration model inventory mismatch')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['recipe','parquet','grpo','sft'])
    parser.add_argument('--data-dir',type=Path,default=ROOT/'artifacts/data')
    parser.add_argument('--base',type=Path,default=ROOT/'artifacts/models/base')
    parser.add_argument('--reward-model',type=Path,default=ROOT/'artifacts/models/reward')
    parser.add_argument('--run-dir',type=Path,default=ROOT/'artifacts/runs/pilot')
    parser.add_argument('--calibration',type=Path,default=ROOT/'artifacts/calibration/scored.json')
    parser.add_argument('--compatibility-report',type=Path,default=ROOT/'artifacts/compatibility/report.json')
    parser.add_argument('--allow-model-calibration',action='store_true',help='Explicit user-authorized model review, recorded as non-human')
    parser.add_argument('--defer-local-compatibility',action='store_true',help='Research training only; local deployment remains unvalidated/disabled')
    parser.add_argument('--max-steps',type=int,default=32)
    parser.add_argument('--hourly-rate',type=float)
    parser.add_argument('--rental-started-at',help='ISO time including timezone; actual instance billing start')
    parser.add_argument('--cap-yuan',type=float,default=10)
    parser.add_argument('--billing-quantum-seconds',type=int,default=60)
    parser.add_argument('--stage-cap-yuan',type=float,default=4)
    args=parser.parse_args()
    run=output_path(args.run_dir)
    if args.task=='recipe':
        print(json.dumps({'framework':'verl==0.5.0','precision':'BF16 LoRA (not QLoRA)',
            'overrides':verl_overrides(args.base.resolve(),args.reward_model.resolve(),args.data_dir.resolve(),run,args.max_steps)},ensure_ascii=False,indent=2))
        return
    if args.task=='parquet':
        prepare_parquet(output_path(args.data_dir));return
    if args.hourly_rate is None or args.rental_started_at is None:
        parser.error('Training requires --hourly-rate and --rental-started-at; dry-run with recipe first')
    import platform
    if platform.system()!='Linux':
        parser.error('GPU training is Linux/NVIDIA only. This Mac is for preparation, evaluation and serving.')
    validate_assets(args.data_dir,args.base,args.reward_model,args.calibration,allow_model_review=args.allow_model_calibration)
    if not args.defer_local_compatibility:
        compatibility=json.loads(args.compatibility_report.read_text())
        if (compatibility.get('compatibility_passed') is not True or compatibility.get('genuine_inference') is not True
            or compatibility.get('base_inventory_sha256')!=sha256(args.base/'bonus_source.json')):
            raise ValueError('Run the real Mac temporary dual-LoRA compatibility probe before GPU training')
    if args.task=='grpo':
        parquet_manifest=json.loads((args.data_dir/'parquet_manifest.json').read_text())
        if not all(sha256(args.data_dir/name)==value for name,value in parquet_manifest['files'].items()):
            raise ValueError('Stale parquet/source fingerprint')
    budget=RentalBudget(args.hourly_rate,args.cap_yuan,parse_start(args.rental_started_at),args.billing_quantum_seconds)
    now=time.time();budget.cost(now)
    if not 0<args.stage_cap_yuan<=args.cap_yuan:
        parser.error('Stage cap must be positive and no larger than total cap')
    if run.exists() and any(run.iterdir()):
        parser.error('Use a new run directory; artifacts are never overwritten')
    run.mkdir(parents=True,exist_ok=True)
    stage_end=now+args.stage_cap_yuan*3600/args.hourly_rate-120
    deadline=min(budget.deadline(),stage_end)
    if deadline<=now:
        parser.error('Budget already exhausted or stage too short to reserve checkpoint time')
    write_json(run/'launch.json',{'task':args.task,'started_at':now,'rental_started_at':budget.started_at,
        'rate_yuan_per_hour':budget.hourly_rate,'cap_yuan':budget.cap_yuan,
        'deadline':deadline,'status':'launched','instance_stopped':False,
        'model_review_authorized':args.allow_model_calibration,'local_compatibility_deferred':args.defer_local_compatibility,
        'production_enabled':False})
    if args.task=='grpo':
        command=[sys.executable,'-m','experiments.qwen3_bonus.verl_entry']+verl_overrides(
            args.base.resolve(),args.reward_model.resolve(),args.data_dir.resolve(),run,args.max_steps)
    else:
        command=[sys.executable,'-m','experiments.qwen3_bonus.sft_entry','--base',str(args.base.resolve()),
                 '--data',str(args.data_dir.resolve()/'classification_train.jsonl'),'--output',str(run)]
    env=os.environ.copy()
    env.update(BONUS_DEADLINE=str(deadline),BONUS_RUN_DIR=str(run),HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false',
        HYDRA_FULL_ERROR='1')
    # Never run a shell; library/trainer receives a fixed executable and structured argv.
    process=subprocess.Popen(command,env=env,start_new_session=True)
    status='running'
    try:
        while process.poll() is None:
            if time.time()>=deadline+90:
                os.killpg(process.pid,signal.SIGTERM);status='deadline_terminated'
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                break
            time.sleep(1)
        if status=='running':status='completed' if process.returncode==0 else 'failed'
    except BaseException:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        status='interrupted';raise
    finally:
        write_json(run/'cost.json',{'status':status,'exit_code':process.returncode,
            'estimated_total_rental_cost_yuan':budget.cost(time.time()),'instance_stopped':False,
            'action_required':'Stop the rental instance in the provider console; process exit does not stop billing.'})
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__=='__main__':
    main()
