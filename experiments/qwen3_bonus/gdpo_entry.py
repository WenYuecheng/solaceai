"""Single-run GDPO recipe and pinned verl worker. Paid launch is gdpo_launch."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

from experiments.qwen3_bonus.common import ROOT,write_json,sha256
from experiments.qwen3_bonus.training import verl_overrides
from experiments.qwen3_bonus.gdpo import KEYS,WEIGHTS,TOLERANCES,REFERENCE_COMMIT

REQUIRED_STATE=['actor/model_world_size_1_rank_0.pt','actor/optim_world_size_1_rank_0.pt',
                'actor/extra_state_world_size_1_rank_0.pt','data.pt','driver_rng.pt',
                'actor/lora_adapter/adapter_model.safetensors','actor/lora_adapter/adapter_config.json']


def gdpo_overrides(base,rm,data,run,calibration,max_steps=64,*,seed=3407,gpu_memory=.45,max_num_seqs=2):
    if type(max_steps) is not int or not 1<=max_steps<=64:raise ValueError('GDPO supports at most 64 steps')
    if not .3<=gpu_memory<=.65:raise ValueError('Unvalidated GPU cache fraction')
    if max_num_seqs not in (1,2):raise ValueError('Only the validated serial/two-sequence layouts are supported')
    values=dict(x.split('=',1) for x in verl_overrides(Path(base),Path(rm),Path(data),Path(run),min(32,max_steps)))
    values.update({'data.train_files':json.dumps(str(Path(data)/'response_train.parquet')),
        'data.val_files':json.dumps(str(Path(data)/'response_dev.parquet')),
        'data.train_batch_size':'4','actor_rollout_ref.actor.ppo_mini_batch_size':'4',
        '+algorithm.bonus_adv_estimator':'gdpo',
        'trainer.total_training_steps':str(max_steps),'trainer.save_freq':'-1',
        'trainer.max_actor_ckpt_to_keep':'null','trainer.experiment_name':'qwen3_gdpo',
        'custom_reward_function.path':json.dumps(str(ROOT/'gdpo_rewards.py')),
        '+custom_reward_function.reward_kwargs.calibration_path':json.dumps(str(calibration)),
        'actor_rollout_ref.actor.entropy_from_logits_with_chunking':'True',
        'actor_rollout_ref.actor.entropy_checkpointing':'True',
        'actor_rollout_ref.rollout.gpu_memory_utilization':str(gpu_memory),
        'actor_rollout_ref.rollout.max_num_seqs':str(max_num_seqs),
        '+actor_rollout_ref.rollout.seed':str(seed),
        '+actor_rollout_ref.model.seed':str(seed),'+data.seed':str(seed)})
    return [f'{k}={v}' for k,v in values.items()]


def train_remote(config):
    import random
    import numpy as np
    import torch
    import ray
    import verl.trainer.ppo.ray_trainer as trainer_module
    import verl.utils.tracking as tracking
    from verl.trainer.main_ppo import create_rl_dataset,create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer,ResourcePoolManager,Role
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from experiments.qwen3_bonus.gdpo_compat import install_gdpo
    from experiments.qwen3_bonus.gdpo_runtime import AtomicCheckpointStore,GDPOBudget,should_save
    from experiments.qwen3_bonus.snapshots import save_snapshot

    run=Path(os.environ['BONUS_RUN_DIR'])
    launch=json.loads((run/'launch.json').read_text())
    budget=GDPOBudget(**launch['budget']);identity=launch['identity']
    store=AtomicCheckpointStore(run/'checkpoints')
    seed=launch['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    tokenizer=hf_tokenizer(config.actor_rollout_ref.model.path,trust_remote_code=False)
    reward=load_reward_manager(config,tokenizer,num_examine=0)
    last_saved=0;completed=0;flat=0;longest=0.;first_two=None;started=time.time()
    def append(name,value):
        with (run/name).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    def on_advantage(data,result):
        responses=data.batch['responses'];mask=data.batch['response_mask']
        records=[]
        for i in range(len(data)):
            length=int(mask[i].sum().item())
            tokens=responses[i,:length].detach().cpu().tolist()
            records.append({'id':str(data.non_tensor_batch['extra_info'][i]['id']),
                'uid':str(data.non_tensor_batch['uid'][i]),'answer':tokenizer.decode(tokens,skip_special_tokens=True),
                'response_tokens':length,'possibly_truncated':length>=config.data.max_response_length,
                'rewards':{k:float(data.non_tensor_batch[k][i]) for k in KEYS},
                'invalid_reason':str(data.non_tensor_batch['invalid_reason'][i])})
        append('rollouts.jsonl',{'step':trainer.global_steps,'records':records,
            'groups':result.group_records,'metrics':result.metrics,'actual_algorithm':'gdpo'})

    class StopRun(Exception):pass

    class Trainer(RayPPOTrainer):
        stop_reason=None
        def _save_checkpoint(self):
            nonlocal last_saved
            step=completed
            if step<=0 or step==last_saved:return
            begin=time.time()
            def save(stage):
                self.actor_rollout_wg.save_checkpoint(str(stage/'actor'),None,step,max_ckpt_to_keep=None)
                torch.save(self.train_dataloader.state_dict(),stage/'data.pt')
                torch.save({'python':random.getstate(),'numpy':np.random.get_state(),
                            'torch':torch.get_rng_state()},stage/'driver_rng.pt')
            # Worker writes FSDP files synchronously; required paths are known for one rank.
            path=store.commit(step,save,{'identity':identity,'completed_step':step},REQUIRED_STATE)
            save_snapshot(path/'actor/lora_adapter',run/'adapter_snapshots',step)
            last_saved=step
            append('checkpoint_events.jsonl',{'step':step,'seconds':time.time()-begin,'path':str(path)})

        def _load_checkpoint(self):
            nonlocal completed,last_saved,first_two,longest
            if not launch.get('resume'):return super()._load_checkpoint()
            progress=json.loads((run/'progress.json').read_text())
            entry=store.resume(identity,REQUIRED_STATE)
            if entry is None or entry['step']!=progress['completed_step']:
                raise ValueError('Resume would repeat completed updates')
            self.config.trainer.resume_mode='resume_path'
            self.config.trainer.resume_from_path=str(entry['path'])
            result=super()._load_checkpoint()
            state=torch.load(entry['path']/'driver_rng.pt',map_location='cpu',weights_only=False)
            random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
            completed=last_saved=entry['step']
            first_two=progress['first_two_completed_at']
            longest=float(progress.get('budget',{}).get('longest_step_seconds',0.))
            return result

    class SeededWorker(ActorRolloutRefWorker):
        def _build_model_optimizer(self,*args,**kwargs):
            # Model/LoRA creation happens in another process, not the driver.
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
            return super()._build_model_optimizer(*args,**kwargs)

    workers={Role.ActorRollout:ray.remote(SeededWorker),Role.RefPolicy:ray.remote(SeededWorker)}
    pool=ResourcePoolManager(resource_pool_spec={'single':[1]},mapping={r:'single' for r in workers})
    train=create_rl_dataset(config.data.train_files,config.data,tokenizer,None,is_train=True)
    val=create_rl_dataset(config.data.val_files,config.data,tokenizer,None,is_train=False)
    if len(train)!=256 or len(val)!=32:raise ValueError('Frozen GDPO train/dev count changed after token filtering')
    trainer=Trainer(config=config,tokenizer=tokenizer,processor=None,role_worker_mapping=workers,
        resource_pool_manager=pool,ray_worker_group_cls=RayWorkerGroup,reward_fn=reward,val_reward_fn=reward,
        train_dataset=train,val_dataset=val,collate_fn=collate_fn,train_sampler=create_rl_sampler(config.data,train))
    if trainer.use_critic:raise RuntimeError('GDPO must remain critic-less')
    original_tracking=tracking.Tracking
    with install_gdpo(trainer_module,config.algorithm,on_advantage) as adapter:
        adapter.self_test(trainer_module)
        write_json(run/'algorithm.json',{'actual_algorithm':'gdpo','critic_enabled':False,
            'trainer_initialization_mode':'grpo','dispatch_verified':True,'reward_keys':KEYS,
            'weights':WEIGHTS,'tolerances':TOLERANCES,'reference_commit':REFERENCE_COMMIT,
            'kernel_sha256':sha256(ROOT/'gdpo.py'),'compatibility_sha256':sha256(ROOT/'gdpo_compat.py')})
        class LocalTracking(original_tracking):
            def log(self,data,step,backend=None):
                nonlocal completed,flat,longest,first_two
                super().log(data=data,step=step,backend=backend)
                if adapter.last_result is None:raise RuntimeError('No GDPO calculation before Actor update')
                completed=int(step)
                metrics={k:float(v) for k,v in data.items() if isinstance(v,(int,float))}
                metrics.update(adapter.last_result.metrics)
                if not all(np.isfinite(v) for v in metrics.values()):
                    trainer.stop_reason='nonfinite_metrics';raise StopRun()
                longest=max(longest,float(metrics.get('timing_s/step',0)))
                flat=flat+1 if metrics['advantage/nonzero_ratio']==0 else 0
                if completed>=2 and first_two is None:first_two=time.time()
                status=budget.status(time.time(),completed,longest)
                write_json(run/'progress.json',{'completed_step':completed,'first_two_completed_at':first_two,
                    'identity':identity,'budget':status})
                append('steps.jsonl',{'step':completed,'elapsed_seconds':time.time()-started,
                    'metrics':metrics,'budget':status,'actual_algorithm':'gdpo'})
                reason=('zero_advantages' if flat>=8 else 'budget_deadline' if status['should_stop'] else
                        'watchdog_stop' if (run/'STOP_REQUESTED.json').exists() else None)
                stopping=bool(reason) or completed>=config.trainer.total_training_steps
                if should_save(completed,last_saved,stopping):trainer._save_checkpoint()
                if reason:trainer.stop_reason=reason;raise StopRun()
        tracking.Tracking=LocalTracking
        try:
            trainer.init_workers();trainer.fit()
        except StopRun:pass
        finally:tracking.Tracking=original_tracking
    result={'actual_algorithm':'gdpo','trained':completed>0 and last_saved>0,'steps':completed,
        'last_saved_step':last_saved,'stop_reason':trainer.stop_reason,'seed':seed,
        'base_revision':identity['base_revision'],'base_inventory_sha256':identity['base_inventory_sha256'],
        'status':'stopped' if trainer.stop_reason else 'completed','clinical_validated':False}
    write_json(run/'training_result.json',result)
    if not result['trained']:raise RuntimeError('No genuine GDPO checkpoint exported')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['recipe','worker'])
    parser.add_argument('--base',type=Path,default=ROOT/'artifacts/models/base')
    parser.add_argument('--reward-model',type=Path,default=ROOT/'artifacts/models/reward')
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/gdpo-20261024/data')
    parser.add_argument('--calibration',type=Path,default=ROOT/'artifacts/gdpo-20261024/calibration.json')
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/gdpo-20261024/run')
    parser.add_argument('--max-steps',type=int,default=64)
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--gpu-memory',type=float,default=.45)
    parser.add_argument('--max-num-seqs',type=int,default=2)
    args=parser.parse_args()
    overrides=gdpo_overrides(args.base,args.reward_model,args.data,args.run,args.calibration,
                             args.max_steps,seed=args.seed,gpu_memory=args.gpu_memory,max_num_seqs=args.max_num_seqs)
    if args.task=='recipe':
        print(json.dumps({'actual_algorithm':'gdpo','framework':'verl==0.5.0','seed':args.seed,
                          'reference_commit':REFERENCE_COMMIT,'overrides':overrides},indent=2));return
    import torch,ray,verl
    from hydra import compose,initialize_config_dir
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    if any(importlib.metadata.version(k)!=v for k,v in [('verl','0.5.0'),('vllm','0.8.5'),('torch','2.6.0')]):
        raise RuntimeError('Use the pinned GDPO CUDA environment')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one NVIDIA GPU')
    if not (args.run/'launch.json').is_file():raise ValueError('Launch through the billing watchdog first')
    os.environ['BONUS_RUN_DIR']=str(args.run.resolve())
    with initialize_config_dir(config_dir=str(Path(verl.__file__).resolve().parent/'trainer/config'),version_base=None):
        config=compose(config_name='ppo_trainer',overrides=overrides)
    runtime=get_ppo_ray_runtime_env();runtime.setdefault('env_vars',{})['BONUS_RUN_DIR']=os.environ['BONUS_RUN_DIR']
    ray.init(num_cpus=4,runtime_env=runtime,include_dashboard=False)
    try:print(json.dumps(ray.get(ray.remote(num_cpus=1)(train_remote).remote(config)),indent=2))
    finally:ray.shutdown()


if __name__=='__main__':main()
