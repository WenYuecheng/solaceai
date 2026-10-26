import math
from pathlib import Path
import pytest


def test_verl_recipe_uses_supported_native_lora_and_explicit_grpo(tmp_path):
    from experiments.qwen3_bonus.training import verl_overrides
    args=verl_overrides(tmp_path/'base', tmp_path/'rm', tmp_path/'data', tmp_path/'run',32)
    assert 'algorithm.adv_estimator=grpo' in args
    assert 'actor_rollout_ref.rollout.name=vllm' in args
    assert 'actor_rollout_ref.model.lora_rank=8' in args
    assert 'actor_rollout_ref.actor.kl_loss_coef=0.02' in args
    assert 'actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean' in args
    assert 'algorithm.norm_adv_by_std_in_grpo=True' in args
    assert 'trainer.val_before_train=False' in args
    assert 'trainer.n_gpus_per_node=1' in args
    assert 'reward_model.enable=False' in args
    assert 'actor_rollout_ref.rollout.n=2' in args
    assert not any('4bit' in a or 'wandb' in a for a in args)


def test_monitor_stops_on_eight_flat_groups_and_nonfinite_loss():
    from experiments.qwen3_bonus.training import StepMonitor
    monitor=StepMonitor(deadline=1000)
    for _ in range(7):
        monitor.record_reward_groups([.5,.5],2)
        assert monitor.stop_reason({'actor/pg_loss':.1},now=10) is None
    monitor.record_reward_groups([.5,.5],2)
    assert monitor.stop_reason({'actor/pg_loss':.1},now=10)=='zero_group_variance'
    assert StepMonitor(1000).stop_reason({'actor/pg_loss':math.nan},now=10)=='nonfinite_metrics'


def test_monitor_uses_longest_step_as_budget_reserve():
    from experiments.qwen3_bonus.training import StepMonitor
    monitor=StepMonitor(deadline=100)
    assert monitor.stop_reason({'timing_s/step':20},now=85)=='budget_deadline'


def test_max_steps_cannot_escape_the_small_experiment(tmp_path):
    from experiments.qwen3_bonus.training import verl_overrides
    with pytest.raises(ValueError):
        verl_overrides(tmp_path,tmp_path,tmp_path,tmp_path,33)


def test_completion_only_mask_excludes_prompt_and_retains_label():
    from experiments.qwen3_bonus.training import completion_only_example
    class Tokenizer:
        def apply_chat_template(self,messages,tokenize,add_generation_prompt=False):
            return [10,11,12] if add_generation_prompt else [10,11,12,20,21]
    row={'prompt':[{'role':'user','content':'焦虑'}], 'completion':[{'role':'assistant','content':'焦虑'}]}
    item=completion_only_example(Tokenizer(),row,4)
    assert item=={'input_ids':[11,12,20,21],'attention_mask':[1]*4,'labels':[-100,-100,20,21]}


def test_mask_fails_if_chat_template_prefix_is_not_exact():
    from experiments.qwen3_bonus.training import completion_only_example
    class Tokenizer:
        def apply_chat_template(self,messages,tokenize,add_generation_prompt=False):
            return [1,2] if add_generation_prompt else [1,3,4]
    with pytest.raises(ValueError,match='prefix'):
        completion_only_example(Tokenizer(),{'prompt':[],'completion':[]},256)


def test_frozen_eval_reference_is_not_used_as_reward_ground_truth():
    from experiments.qwen3_bonus.training import parquet_records
    row={'id':'t1','dialogue_id':'d1','prompt':[{'role':'user','content':'压力好大'}], 'reference':'不应进入训练的参考答案'}
    item=parquet_records([row])[0]
    assert item['reward_model']['ground_truth']==''
    assert item['extra_info']['messages']==row['prompt']
    assert 'reference' not in item['extra_info']
