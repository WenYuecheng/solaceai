"""Single-node GRPO using public verl v0.5.0 APIs, with bounded experiment logging.

Worker/dataset construction follows verl/trainer/main_ppo.py (Apache-2.0,
Copyright 2024 Bytedance). No installed verl or existing application files are edited.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from experiments.qwen3_bonus.common import write_json
from experiments.qwen3_bonus.training import StepMonitor


def train_remote(config):
    import ray
    import torch
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.main_ppo import create_rl_dataset,create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer,ResourcePoolManager,Role
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    import verl.utils.tracking as tracking
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    output=Path(os.environ['BONUS_RUN_DIR'])
    monitor=StepMonitor(float(os.environ['BONUS_DEADLINE']))
    tokenizer=hf_tokenizer(config.actor_rollout_ref.model.path,trust_remote_code=False)
    base_reward=load_reward_manager(config,tokenizer,num_examine=0)

    class Reward:
        batches=0
        def __call__(self,data,return_dict=False):
            result=base_reward(data,return_dict=True)
            values=result['reward_tensor'].sum(dim=-1).detach().cpu().tolist()
            monitor.record_reward_groups(values,config.actor_rollout_ref.rollout.n)
            self.batches+=1
            responses=data.batch['responses']
            lengths=data.batch['attention_mask'][:,-responses.shape[1]:].sum(dim=-1).detach().cpu().tolist()
            with (output/'rollouts.jsonl').open('a',encoding='utf-8') as stream:
                for index,(value,length) in enumerate(zip(values,lengths)):
                    tokens=responses[index,:int(length)].detach().cpu().tolist()
                    stream.write(json.dumps({'reward_batch':self.batches,'sample':index,'reward':value,
                        'response_tokens':int(length),'possibly_truncated':int(length)>=config.data.max_response_length,
                        'answer':tokenizer.decode(tokens,skip_special_tokens=True)},ensure_ascii=False,allow_nan=False)+'\n')
            return result if return_dict else result['reward_tensor']

    class StopExperiment(Exception):
        pass

    class Trainer(RayPPOTrainer):
        def _save_checkpoint(self):
            super()._save_checkpoint()
            from experiments.qwen3_bonus.snapshots import save_snapshot
            adapter=output/'checkpoints'/f'global_step_{self.global_steps}'/'actor/lora_adapter'
            save_snapshot(adapter,output/'adapter_snapshots',int(self.global_steps))
            from experiments.qwen3_bonus.plotting import safe_render
            safe_render(output)

        def fit(self):
            original=tracking.Tracking
            trainer=self

            class LocalTracking(original):
                def log(self,data,step,backend=None):
                    super().log(data=data,step=step,backend=backend)
                    metrics={k:float(v) for k,v in data.items() if isinstance(v,(int,float))}
                    metrics.update(reward_group_std=monitor.mean_group_std,reward_mean=monitor.mean_reward)
                    # JSON never emits NaN/Infinity, but these still trigger a real stop below.
                    safe={k:(v if math.isfinite(v) else None) for k,v in metrics.items()}
                    with (output/'steps.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps({'step':int(step),'metrics':safe},allow_nan=False)+'\n')
                    from experiments.qwen3_bonus.plotting import record_scalars
                    record_scalars(output,int(step),safe)
                    reason=monitor.stop_reason(metrics)
                    if reason:
                        trainer.stop_reason=reason
                        raise StopExperiment(reason)

            tracking.Tracking=LocalTracking
            self.stop_reason=None
            try:
                super().fit()
            except StopExperiment:
                # Save at a completed optimizer boundary; a watchdog handles hung steps.
                adapter=output/'checkpoints'/f'global_step_{self.global_steps}'/'actor/lora_adapter/adapter_model.safetensors'
                if not adapter.exists():self._save_checkpoint()
            finally:
                tracking.Tracking=original

    workers={Role.ActorRollout:ray.remote(ActorRolloutRefWorker),Role.RefPolicy:ray.remote(ActorRolloutRefWorker)}
    pool=ResourcePoolManager(resource_pool_spec={'single':[1]},mapping={role:'single' for role in workers})
    train=create_rl_dataset(config.data.train_files,config.data,tokenizer,None,is_train=True)
    validation=create_rl_dataset(config.data.val_files,config.data,tokenizer,None,is_train=False)
    if len(train)==0:
        raise ValueError('No prompts fit the token budget; do not launch an empty training run')
    # Avoid exhausting the single epoch before the configured final-save boundary.
    config.trainer.total_training_steps=min(config.trainer.total_training_steps,len(train))
    trainer=Trainer(config=config,tokenizer=tokenizer,processor=None,
        role_worker_mapping=workers,resource_pool_manager=pool,ray_worker_group_cls=RayWorkerGroup,
        reward_fn=Reward(),val_reward_fn=base_reward,train_dataset=train,val_dataset=validation,
        collate_fn=collate_fn,train_sampler=create_rl_sampler(config.data,train))
    trainer.init_workers()
    trainer.fit()
    adapters=[]
    for path in (output/'checkpoints').glob('global_step_*/actor/lora_adapter/adapter_config.json'):
        weights=path.parent/'adapter_model.safetensors'
        if weights.is_file() and weights.stat().st_size>8:
            config_record=json.loads(path.read_text())
            if config_record.get('r')==8 and config_record.get('task_type')=='CAUSAL_LM':
                adapters.append(path)
    adapters.sort(key=lambda p:int(p.parents[2].name.removeprefix('global_step_')))
    result={'status':'stopped' if trainer.stop_reason else 'completed','stop_reason':trainer.stop_reason,
        'adapter':str(adapters[-1].parent) if adapters else None,'framework':'verl==0.5.0',
        'precision':'BF16 LoRA','train_prompts_after_token_filter':len(train),
        'reward_model':'Skywork-Reward-V2-Qwen3-0.6B (CPU, frozen)',
        'steps':max((json.loads(line)['step'] for line in (output/'steps.jsonl').read_text().splitlines()),default=0),
        'trained':bool(adapters),'clinical_validated':False,'timestamp':time.time()}
    from experiments.qwen3_bonus.common import sha256
    from experiments.qwen3_bonus.models import SOURCES
    result.update(base_revision=SOURCES['base']['revision'],
        base_inventory_sha256=sha256(Path(config.actor_rollout_ref.model.path)/'bonus_source.json'))
    write_json(output/'training_result.json',result)
    from experiments.qwen3_bonus.plotting import safe_render
    safe_render(output)
    if not adapters:
        raise RuntimeError('verl finished without an exported PEFT adapter')
    return result


def main():
    import sys
    import ray
    import torch
    import verl
    from hydra import compose,initialize_config_dir
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    if importlib.metadata.version('verl')!='0.5.0':
        raise RuntimeError('Use the locked verl==0.5.0 environment; recipe APIs are version-specific')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one NVIDIA GPU using CUDA_VISIBLE_DEVICES')
    if time.time()>=float(os.environ['BONUS_DEADLINE']):
        raise RuntimeError('Rental budget expired before initialization')
    config_dir=str(Path(verl.__file__).resolve().parent/'trainer/config')
    with initialize_config_dir(config_dir=config_dir,version_base=None):
        config=compose(config_name='ppo_trainer',overrides=sys.argv[1:])
    runtime=get_ppo_ray_runtime_env()
    runtime.setdefault('env_vars',{}).update({k:os.environ[k] for k in (
        'BONUS_RUN_DIR','BONUS_DEADLINE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')})
    ray.init(num_cpus=4,runtime_env=runtime,include_dashboard=False)
    try:
        result=ray.get(ray.remote(num_cpus=1)(train_remote).remote(config))
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        ray.shutdown()


if __name__=='__main__':
    main()
