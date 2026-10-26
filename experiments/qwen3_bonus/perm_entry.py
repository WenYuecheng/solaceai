"""One budgeted PERM-inspired response LoRA using unchanged native verl GRPO."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from experiments.qwen3_bonus.common import ROOT, sha256, write_json
from experiments.qwen3_bonus.training import verl_overrides

ALGORITHM = 'perm_grpo'
REQUIRED_STATE = ['actor/model_world_size_1_rank_0.pt', 'actor/optim_world_size_1_rank_0.pt',
    'actor/extra_state_world_size_1_rank_0.pt', 'data.pt', 'driver_rng.pt',
    'actor/lora_adapter/adapter_model.safetensors', 'actor/lora_adapter/adapter_config.json']


def perm_overrides(base, rm, data, run, calibration, max_steps=64, *, seed=3407,
                   gpu_memory=.45, max_num_seqs=2):
    if type(max_steps) is not int or not 1 <= max_steps <= 64:
        raise ValueError('The single run supports 1..64 steps')
    if type(seed) is not int or seed != 3407:
        raise ValueError('Frozen experiment seed is 3407')
    if not math.isfinite(gpu_memory) or not .3 <= gpu_memory <= .65 or max_num_seqs not in (1, 2):
        raise ValueError('Unvalidated GPU resource layout')
    values = dict(x.split('=', 1) for x in verl_overrides(
        Path(base), Path(rm), Path(data), Path(run), min(max_steps, 32)))
    values.update({'data.val_files': json.dumps(str(Path(data)/'response_dev.parquet')),
        'data.train_batch_size': '4', 'actor_rollout_ref.actor.ppo_mini_batch_size': '4',
        'trainer.total_training_steps': str(max_steps), 'trainer.save_freq': '-1',
        'trainer.max_actor_ckpt_to_keep': 'null', 'trainer.experiment_name': ALGORITHM,
        'custom_reward_function.path': json.dumps(str(ROOT/'perm_rewards.py')),
        '+custom_reward_function.reward_kwargs.calibration_path': json.dumps(str(calibration)),
        'actor_rollout_ref.actor.entropy_from_logits_with_chunking': 'True',
        'actor_rollout_ref.actor.entropy_checkpointing': 'True',
        'actor_rollout_ref.rollout.gpu_memory_utilization': str(gpu_memory),
        'actor_rollout_ref.rollout.max_num_seqs': str(max_num_seqs),
        '+actor_rollout_ref.rollout.seed': str(seed), '+actor_rollout_ref.model.seed': str(seed),
        '+data.seed': str(seed)})
    return [f'{key}={value}' for key, value in values.items()]


class PermReward:
    """Rewrite scalar rewards, never the trainer's advantage calculation."""
    def __init__(self, base_reward, on_batch=None):
        self.base_reward = base_reward
        self.on_batch = on_batch
        self.last_data = None
        self.last_records = []
        self.last_metrics = {}
        self.calls = 0

    def __call__(self, data, return_dict=False):
        import torch
        import numpy as np
        from experiments.qwen3_bonus.perm_rewards import apply_group_gate
        responses = data.batch['responses']
        mask = data.batch.get('response_mask')
        if mask is None:
            mask = data.batch['attention_mask'][:, -responses.shape[-1]:]
        lengths = mask.sum(-1).long()
        if len(lengths) == 0 or (lengths <= 0).any():
            raise ValueError('Every response needs at least one valid token')
        start = time.monotonic()
        result = self.base_reward(data, return_dict=True)
        extra = result.get('reward_extra_info', {})
        if 'score' not in extra or any(len(v) != len(lengths) for v in extra.values()):
            raise ValueError('Every response needs matching reward components')
        records = [{k: (v[i].item() if isinstance(v[i], np.generic) else v[i])
                    for k, v in extra.items()} for i in range(len(lengths))]
        gated, metrics = apply_group_gate(records, list(data.non_tensor_batch['uid']))
        scores = [float(row['score']) for row in gated]
        if any(not math.isfinite(value) for value in scores):
            raise ValueError('Nonfinite final training reward')
        tensor = torch.zeros_like(result['reward_tensor'], dtype=torch.float32)
        if tensor.shape != responses.shape or tensor.shape != mask.shape:
            raise ValueError('Response/reward/mask shape mismatch')
        tensor[torch.arange(len(lengths), device=tensor.device), lengths.to(tensor.device)-1] = torch.tensor(
            scores, device=tensor.device)
        result['reward_tensor'] = tensor
        result['reward_extra_info'] = {key: [r[key] for r in gated] for key in gated[0]}
        self.last_data, self.last_records = data, gated
        self.calls += 1
        groups = {}
        for uid, score in zip(data.non_tensor_batch['uid'], scores):
            groups.setdefault(str(uid), []).append(score)
        diffs = [abs(pair[0]-pair[1]) for pair in groups.values()]
        paired = {}
        for uid, row in zip(data.non_tensor_batch['uid'], gated):
            paired.setdefault(str(uid), []).append(row)
        eligible_conflicts = []
        for a, b in paired.values():
            dp, dc = a['preference']-b['preference'], a['constraint']-b['constraint']
            if not a['invalid'] and not b['invalid'] and a['constraint_applicable'] and b['constraint_applicable'] and dp != 0 and dc != 0:
                eligible_conflicts.append(dp*dc < 0)
        def average(key, applicable=False):
            values = [float(r[key]) for r in gated if not applicable or r['constraint_applicable']]
            return sum(values)/len(values) if values else 0.
        metrics.update({'reward/preference_mean': average('preference'),
            'reward/group_gate_ratio': metrics['gated_groups']/metrics['groups'],
            'reward/preference_constraint_conflict': sum(eligible_conflicts)/len(eligible_conflicts) if eligible_conflicts else 0.,
            'reward/conflict_denominator': len(eligible_conflicts),
            'reward/constraint_mean': average('constraint', True), 'reward/main_mean': average('main'),
            'reward/repeat_mean': average('repeat_ratio'), 'reward/invalid_ratio': average('invalid'),
            'reward/constraint_applicable_ratio': average('constraint_applicable'),
            'reward/group_abs_diff_mean': sum(diffs)/len(diffs),
            'reward/group_std': float(np.mean([np.std(pair, ddof=1) for pair in groups.values()])),
            'reward/zero_group_ratio': sum(d == 0 for d in diffs)/len(diffs),
            'timing_s/cpu_reward': time.monotonic()-start})
        self.last_metrics = metrics
        if self.on_batch:
            self.on_batch(data, gated, metrics)
        return result if return_dict else tensor


def verify_native_grpo(trainer_module):
    """Invoke actual installed verl on a synthetic masked batch before rollout."""
    import torch
    import numpy as np
    from tensordict import TensorDict
    from verl import DataProto
    from omegaconf import OmegaConf
    function = trainer_module.compute_advantage
    mask = torch.tensor([[1., 1., 0.], [1., 0., 0.], [1., 1., 1.], [1., 1., 0.]])
    rewards = torch.tensor([[0., .6, 0.], [.2, 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    data = DataProto(batch=TensorDict({'token_level_rewards': rewards, 'response_mask': mask}, batch_size=[4]),
                     non_tensor_batch={'uid': np.array(['a', 'a', 'b', 'b'])})
    result = function(data, adv_estimator='grpo', config=OmegaConf.create({}), num_repeat=2,
                      norm_adv_by_std_in_grpo=True)
    expected = torch.tensor([1/math.sqrt(2), -1/math.sqrt(2), 0., 0.])[:, None]*mask
    if (trainer_module.compute_advantage is not function
            or not torch.allclose(result.batch['advantages'], expected, atol=1e-5)
            or not torch.equal(result.batch['returns'], result.batch['advantages'])):
        raise RuntimeError('Native GRPO startup verification failed')
    return {'native_grpo_verified': True, 'advantage_source_sha256': sha256(Path(trainer_module.__file__))}


def initialize_prepare_fit(trainer, sample, prepare):
    baseline = sample()
    trainer.init_workers()
    prepare(baseline)
    trainer.fit()


def sample_host_memory():
    """Use total cgroup usage (includes Ray siblings); system usage fallback includes all workers."""
    import psutil
    from experiments.qwen3_bonus.perm_launch import cgroup_memory_limit
    vm = psutil.virtual_memory()
    root = Path('/sys/fs/cgroup'); entries=[]
    if Path('/proc/self/cgroup').is_file():
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _,controllers,relative=line.split(':',2)
            if controllers == '' or 'memory' in controllers.split(','):
                target=root/relative.lstrip('/')
                if target.resolve().is_relative_to(root.resolve()):
                    for directory in [target,*[p for p in target.parents if p.is_relative_to(root)]]:
                        for lim,cur in [('memory.max','memory.current'),('memory.limit_in_bytes','memory.usage_in_bytes')]:
                            if (directory/lim).is_file() and (directory/cur).is_file():
                                value=(directory/lim).read_text().strip()
                                if value!='max' and 0<int(value)<2**60:
                                    entries.append((int(value),int((directory/cur).read_text()),str(directory)))
    if entries:
        limit,used,path=min(entries,key=lambda x:x[0])
        metric='cgroup usage including all resident Ray worker processes and charged cache: '+path
    else:
        limit=cgroup_memory_limit();used=vm.total-vm.available
        metric='system total minus available; conservative whole-host accounting including all owned model workers'
    return {'physical_memory_bytes':vm.total,'cgroup_limit_bytes':limit,'used_bytes':used,
            'measured_at':time.time(),'metric':metric,'platform':__import__('platform').platform()}


def check_preparation_budget(run, budget):
    if (Path(run)/'STOP_REQUESTED.json').exists() or budget.status(time.time(),0,0)['should_stop']:
        raise RuntimeError('Preparation reached existing setup/training budget deadline')


def prepare_reward(run, launch, budget, baseline):
    """One supervised model pool, sequential isolated target CPU processes, then immutable choice."""
    import subprocess, sys
    from experiments.qwen3_bonus.perm_models import verify_model
    from experiments.qwen3_bonus.perm_calibration import make_candidate, select_model
    from experiments.qwen3_bonus.perm_launch import live_resources, validate_selection
    from experiments.qwen3_bonus.gdpo_data import immutable_write, _json
    run=Path(run); candidates=[]; rejections=[]
    post=sample_host_memory()
    for model_path in launch['preparation']['candidate_paths']:
        check_preparation_budget(run,budget)
        model_path=Path(model_path);identity=verify_model(model_path)
        sample=sample_host_memory();effective=min(sample['physical_memory_bytes'],sample['cgroup_limit_bytes'] or sample['physical_memory_bytes'])
        # Two FP32 copies cover resident parameters and loading/conversion overlap.
        # Derive bytes from tensor shapes instead of assuming serialized dtype.
        import struct
        fp32_bytes=0
        for weight in model_path.glob('*.safetensors'):
            with weight.open('rb') as stream:
                header=json.loads(stream.read(struct.unpack('<Q',stream.read(8))[0]))
            fp32_bytes+=sum(4*math.prod(t['shape']) for name,t in header.items() if name!='__metadata__')
        bound=2*fp32_bytes+512*1024**2
        if sample['used_bytes']+bound>.8*effective:
            rejections.append({'model':identity,'reason':'measured_headroom_insufficient','measurement':sample,
                               'candidate_load_bound_bytes':bound,'bound_definition':'2 times actual FP32 tensor bytes for resident parameters and conversion overlap plus 512 MiB tokenizer/activation transient; retains 20 percent host reserve'})
            continue
        report=run/('calibration-target-'+identity['revision']+'.json')
        command=[sys.executable,'-m','experiments.qwen3_bonus.perm_calibration','score','--data',launch['preparation']['data'],
                 '--model',str(model_path),'--output',str(report),'--execute']
        with (run/('calibration-'+identity['revision']+'.log')).open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                     'PERM_CPU_THREADS':'2', 'TOKENIZERS_PARALLELISM':'false'})
        candidates.append(make_candidate(report))
    write_json(run/'preparation_rejections.json',rejections)
    check_preparation_budget(run,budget)
    current=sample_host_memory()
    resources=live_resources({'physical_memory_bytes':current['physical_memory_bytes'],'cgroup_limit_bytes':current['cgroup_limit_bytes'],
        'runtime_bytes':baseline['used_bytes'],
        'actor_ref_offload_bytes':max(post['used_bytes'],current['used_bytes'],baseline['used_bytes'])-baseline['used_bytes'],
        'offload_verified':True,'phase':'actor_reference_initialized',
        'memory_metric_definitions':{'runtime_bytes':baseline['metric'],
            'actor_ref_offload_bytes':'nonnegative post-init/current minus baseline host usage; includes modelpool/offload and runtime growth'},
        'measurements':{'baseline':baseline,'initialized':post,'after_calibration':current},
        'preferred_candidate_rejections':rejections},budget,time.time())
    evidence=run/'initialized_resources.json';immutable_write(evidence,_json(resources))
    resources.update(evidence_path=str(evidence),evidence_sha256=sha256(evidence))
    selection=select_model(candidates,resources)
    path=run/'reward_selection.json';immutable_write(path,_json(selection))
    selected=validate_selection(selection)
    check_preparation_budget(run,budget)
    launch['identity'].update(reward_inventory_sha256=selected['model_inventory_sha256'],reward_revision=selected['revision'],
        calibration_sha256=selected['calibration_sha256'],reward_selection_sha256=sha256(path),resource_evidence_sha256=sha256(evidence))
    launch['preparation']['completed']=True
    launch['selected']=selected
    write_json(run/'launch.json',launch)
    return selected


def train_remote(config):
    import random
    import numpy as np
    import torch
    import ray
    import verl.trainer.ppo.ray_trainer as trainer_module
    import verl.utils.tracking as tracking
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from experiments.qwen3_bonus.gdpo_runtime import AtomicCheckpointStore, GDPOBudget, should_save
    from experiments.qwen3_bonus.snapshots import save_snapshot

    run = Path(os.environ['BONUS_RUN_DIR'])
    launch = json.loads((run/'launch.json').read_text())
    budget = GDPOBudget(**launch['budget'])
    identity, seed = launch['identity'], launch['seed']
    if launch['actual_algorithm'] != ALGORITHM or config.algorithm.adv_estimator != 'grpo':
        raise ValueError('Frozen native GRPO algorithm identity mismatch')
    native_function = trainer_module.compute_advantage
    native_receipt = verify_native_grpo(trainer_module)
    store = AtomicCheckpointStore(run/'checkpoints')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=False)
    preparing = bool(launch.get('preparation')) and not launch['preparation'].get('completed')
    if launch.get('resume') and preparing:
        raise ValueError('Resumed run must already have a frozen reward selection')
    base_reward = None if preparing else load_reward_manager(config, tokenizer, num_examine=0)
    last_saved = completed = flat = 0
    longest = 0.
    first_two = None
    started = time.time()
    elapsed_prior = 0.

    def append(name, value):
        with (run/name).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')

    def on_batch(data, records, metrics):
        responses = data.batch['responses']
        mask = data.batch.get('response_mask', data.batch['attention_mask'][:, -responses.shape[-1]:])
        output = []
        for i, record in enumerate(records):
            length = int(mask[i].sum().item())
            output.append({'id': str(data.non_tensor_batch['extra_info'][i]['id']),
                'uid': str(data.non_tensor_batch['uid'][i]),
                'answer': tokenizer.decode(responses[i, :length].detach().cpu().tolist(), skip_special_tokens=True),
                'response_tokens': length, 'possibly_truncated': length >= config.data.max_response_length,
                'rewards': record})
        append('rollouts.jsonl', {'step': trainer.global_steps, 'records': output,
            'metrics': metrics, 'actual_algorithm': ALGORITHM, 'update_completed': False})

    reward = PermReward(base_reward, on_batch)
    class StopRun(Exception):
        pass

    class Trainer(RayPPOTrainer):
        stop_reason = None
        def _save_checkpoint(self):
            nonlocal last_saved
            if completed <= 0 or last_saved == completed:
                return
            begin = time.time()
            def save(stage):
                self.actor_rollout_wg.save_checkpoint(str(stage/'actor'), None, completed, max_ckpt_to_keep=None)
                torch.save(self.train_dataloader.state_dict(), stage/'data.pt')
                torch.save({'python': random.getstate(), 'numpy': np.random.get_state(),
                            'torch': torch.get_rng_state()}, stage/'driver_rng.pt')
            path = store.commit(completed, save, {'identity': identity, 'completed_step': completed}, REQUIRED_STATE)
            save_snapshot(path/'actor/lora_adapter', run/'adapter_snapshots', completed)
            last_saved = completed
            append('checkpoint_events.jsonl', {'step': completed, 'seconds': time.time()-begin, 'path': str(path)})

        def _load_checkpoint(self):
            nonlocal completed, last_saved, first_two, longest, elapsed_prior
            if not launch.get('resume'):
                return super()._load_checkpoint()
            progress = json.loads((run/'progress.json').read_text())
            entry = store.resume(identity, REQUIRED_STATE)
            if entry is None or entry['step'] != progress['completed_step']:
                raise ValueError('Resume would repeat completed updates')
            self.config.trainer.resume_mode = 'resume_path'
            self.config.trainer.resume_from_path = str(entry['path'])
            result = super()._load_checkpoint()
            state = torch.load(entry['path']/'driver_rng.pt', map_location='cpu', weights_only=False)
            random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
            completed = last_saved = entry['step']
            first_two = progress['first_two_completed_at']
            longest = float(progress.get('budget', {}).get('longest_step_seconds', 0.))
            elapsed_prior = float(progress.get('elapsed_seconds', 0.))
            return result

    class SeededWorker(ActorRolloutRefWorker):
        def _build_model_optimizer(self, *args, **kwargs):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            return super()._build_model_optimizer(*args, **kwargs)

    workers = {Role.ActorRollout: ray.remote(SeededWorker), Role.RefPolicy: ray.remote(SeededWorker)}
    pool = ResourcePoolManager(resource_pool_spec={'single': [1]}, mapping={role: 'single' for role in workers})
    train = create_rl_dataset(config.data.train_files, config.data, tokenizer, None, is_train=True)
    val = create_rl_dataset(config.data.val_files, config.data, tokenizer, None, is_train=False)
    if len(train) != 256 or len(val) != 32:
        raise ValueError('Frozen data count changed after token filtering')
    trainer = Trainer(config=config, tokenizer=tokenizer, processor=None, role_worker_mapping=workers,
        resource_pool_manager=pool, ray_worker_group_cls=RayWorkerGroup, reward_fn=reward,
        val_reward_fn=base_reward, train_dataset=train, val_dataset=val, collate_fn=collate_fn,
        train_sampler=create_rl_sampler(config.data, train))
    if trainer.use_critic:
        raise RuntimeError('Native GRPO must remain critic-less')
    write_json(run/'algorithm.json', {'actual_algorithm': ALGORITHM, 'adv_estimator': 'grpo',
        'critic_enabled': False, 'advantage_function_replaced': False, **native_receipt,
        'reward_source_sha256': sha256(ROOT/'perm_rewards.py')})
    original_tracking = tracking.Tracking
    class LocalTracking(original_tracking):
        def log(self, data, step, backend=None):
            nonlocal completed, flat, longest, first_two
            super().log(data=data, step=step, backend=backend)
            completed = int(step)  # log occurs only after successful Actor update
            metrics = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
            metrics.update(reward.last_metrics)
            batch = reward.last_data
            if batch is None or 'advantages' not in batch.batch or trainer_module.compute_advantage is not native_function:
                raise RuntimeError('Actual native GRPO advantages unavailable after Actor update')
            valid = batch.batch.get('response_mask')
            values = batch.batch['advantages'][valid.bool()]
            metrics['advantage/nonzero_ratio'] = float((values != 0).float().mean().item())
            metrics['advantage/std'] = float(values.float().std(correction=0).item())
            if not all(math.isfinite(v) for v in metrics.values()):
                trainer.stop_reason = 'nonfinite_metrics'
                raise StopRun()
            longest = max(longest, float(metrics.get('timing_s/step', 0)))
            flat = flat+1 if metrics['advantage/nonzero_ratio'] == 0 else 0
            if completed >= 2 and first_two is None:
                first_two = time.time()
            status = budget.status(time.time(), completed, longest)
            elapsed = elapsed_prior+time.time()-started
            write_json(run/'progress.json', {'completed_step': completed, 'first_two_completed_at': first_two,
                'identity': identity, 'budget': status, 'elapsed_seconds': elapsed})
            append('steps.jsonl', {'step': completed, 'elapsed_seconds': elapsed, 'metrics': metrics,
                'budget': status, 'actual_algorithm': ALGORITHM,
                'total_estimated_cost_yuan': status['accumulated_cost']+launch['preparation_cost_yuan']})
            reason = ('zero_advantages' if flat >= 8 else 'budget_deadline' if status['should_stop'] else
                      'watchdog_stop' if (run/'STOP_REQUESTED.json').exists() else None)
            if should_save(completed, last_saved, bool(reason) or completed >= config.trainer.total_training_steps):
                trainer._save_checkpoint()
            if reason:
                trainer.stop_reason = reason
                raise StopRun()
    tracking.Tracking = LocalTracking
    try:
        def prepare(baseline):
            if preparing:
                selected = prepare_reward(run, launch, budget, baseline)
                identity.update(launch['identity'])
                config.custom_reward_function.reward_kwargs.reward_model_path = selected['model_path']
                config.custom_reward_function.reward_kwargs.calibration_path = selected['calibration_path']
                reward.base_reward = load_reward_manager(config, tokenizer, num_examine=0)
                trainer.val_reward_fn = reward.base_reward
            check_preparation_budget(run, budget) if preparing else None
        initialize_prepare_fit(trainer, sample_host_memory, prepare)
    except StopRun:
        if completed > 0 and trainer.stop_reason != 'nonfinite_metrics':
            trainer._save_checkpoint()
    finally:
        tracking.Tracking = original_tracking
    result = {'actual_algorithm': ALGORITHM, 'trained': completed > 0 and last_saved > 0,
        'steps': completed, 'last_saved_step': last_saved, 'stop_reason': trainer.stop_reason,
        'seed': seed, 'status': 'stopped' if trainer.stop_reason else 'completed',
        'base_revision': identity['base_revision'], 'base_inventory_sha256': identity['base_inventory_sha256'],
        'preparation_cost_yuan': launch['preparation_cost_yuan'], 'clinical_validated': False}
    write_json(run/'training_result.json', result)
    if not result['trained']:
        raise RuntimeError('No genuine PERM-GRPO checkpoint exported')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['recipe', 'worker'])
    root = ROOT/'artifacts/perm-grpo-20261024'
    for name, default in [('base', ROOT/'artifacts/models/base'), ('reward-model', ROOT/'artifacts/models/reward'),
                          ('data', root/'data'), ('calibration', root/'calibration.json'), ('run', root/'run')]:
        parser.add_argument('--'+name, type=Path, default=default)
    parser.add_argument('--max-steps', type=int, default=64)
    parser.add_argument('--gpu-memory', type=float, default=.45)
    parser.add_argument('--max-num-seqs', type=int, default=2)
    args = parser.parse_args()
    overrides = perm_overrides(args.base, args.reward_model, args.data, args.run, args.calibration,
        args.max_steps, gpu_memory=args.gpu_memory, max_num_seqs=args.max_num_seqs)
    if args.task == 'recipe':
        print(json.dumps({'actual_algorithm': ALGORITHM, 'overrides': overrides}, indent=2))
        return
    for package, version in [('verl', '0.5.0'), ('vllm', '0.8.5'), ('torch', '2.6.0')]:
        if importlib.metadata.version(package) != version:
            raise RuntimeError('Use the locked single-GPU CUDA environment')
    import torch, ray, verl
    from hydra import compose, initialize_config_dir
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one NVIDIA GPU')
    if not (args.run/'launch.json').is_file():
        raise ValueError('Launch through the budget watchdog')
    os.environ['BONUS_RUN_DIR'] = str(args.run.resolve())
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer', overrides=overrides)
    runtime = get_ppo_ray_runtime_env()
    runtime.setdefault('env_vars', {}).update({k: os.environ[k] for k in (
        'BONUS_RUN_DIR', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'WANDB_DISABLED', 'TOKENIZERS_PARALLELISM')
        if k in os.environ})
    ray.init(num_cpus=4, runtime_env=runtime, include_dashboard=False)
    try:
        print(json.dumps(ray.get(ray.remote(num_cpus=1)(train_remote).remote(config)), indent=2))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
