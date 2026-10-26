import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def api():
    return importlib.import_module('experiments.qwen3_bonus.perm_entry')


def test_native_recipe_has_correct_batching_and_no_gdpo_dispatch(tmp_path):
    args = api().perm_overrides(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
    recipe = dict(x.split('=', 1) for x in args)
    assert recipe['algorithm.adv_estimator'] == 'grpo'
    assert recipe['data.train_batch_size'] == recipe['actor_rollout_ref.actor.ppo_mini_batch_size'] == '4'
    assert recipe['actor_rollout_ref.rollout.n'] == '2'
    assert recipe['trainer.total_training_steps'] == '64'
    assert recipe['trainer.save_freq'] == '-1'
    assert recipe['actor_rollout_ref.actor.kl_loss_coef'] == '0.02'
    assert recipe['algorithm.use_kl_in_reward'] == 'False'
    assert recipe['trainer.experiment_name'] == 'perm_grpo'
    assert not any('gdpo' in x or 'bonus_adv_estimator' in x for x in args)
    with pytest.raises(ValueError):
        api().perm_overrides(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, 65)


def test_recipe_composes_with_actual_pinned_hydra_schema(tmp_path):
    from hydra import compose, initialize_config_dir
    from experiments.qwen3_bonus.common import ROOT
    with initialize_config_dir(config_dir=str(ROOT/'vendor/verl-config-v0.5.0'), version_base=None):
        cfg = compose(config_name='ppo_trainer', overrides=api().perm_overrides(
            tmp_path, tmp_path, tmp_path, tmp_path, tmp_path))
    assert cfg.algorithm.adv_estimator == 'grpo'
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n == 8
    assert cfg.data.seed == 3407


def test_group_gate_changes_real_reward_tensor_and_keeps_padding_zero():
    torch = pytest.importorskip('torch')
    def base(data, return_dict):
        return {'reward_tensor':torch.tensor([[0.,-.02,0.],[0.,0.,-.08]]),
                'reward_extra_info':{'score':[-.02,-.08], 'preference':[.4,.7], 'constraint':[0.,0.],
                  'constraint_applicable':[1,1], 'invalid':[0,0], 'invalid_reason':['',''],
                  'main':[0.,0.], 'repeat_ratio':[.2,.8], 'raw_rm':[2.,3.], 'length_checks':[[],[]]}}
    data = SimpleNamespace(batch={'responses':torch.ones(2,3,dtype=torch.long),
                    'response_mask':torch.tensor([[1.,1.,0.],[1.,1.,1.]])},
                non_tensor_batch={'uid':np.array(['same','same'])})
    wrapper = api().PermReward(base)
    result = wrapper(data, return_dict=True)
    assert not result['reward_tensor'].any()
    assert result['reward_extra_info']['score'] == [0.,0.]
    assert result['reward_extra_info']['score_before_gate'] == [-.02,-.08]
    assert wrapper.last_data is data
    assert wrapper.last_metrics['reward/group_gate_ratio'] == 1.


def test_zero_response_tokens_cannot_put_reward_on_padding():
    torch = pytest.importorskip('torch')
    data = SimpleNamespace(batch={'responses':torch.zeros(2,2), 'response_mask':torch.zeros(2,2)},
                           non_tensor_batch={'uid':['a','a']})
    with pytest.raises(ValueError,match='response'):
        api().PermReward(lambda *a,**k:pytest.fail('validate mask before scoring'))(data)


def test_cpu_prep_cost_reduces_gpu_cap_without_changing_setup_and_save_reserve():
    mod=importlib.import_module('experiments.qwen3_bonus.perm_launch')
    budget=mod.training_budget(rate=2., started_at=100., quantum_seconds=60, preparation_cost=1.25)
    assert budget.cap_yuan == 18.75 and budget.reserve_yuan == 2.
    assert budget.cost(1900.)+1.25 <= 20
    for bad in (-1.,float('nan'),17.):
        with pytest.raises(ValueError):
            mod.training_budget(rate=2.,started_at=100.,quantum_seconds=60,preparation_cost=bad)


def test_model_selection_receipt_cannot_be_tampered_or_cross_bound(tmp_path):
    mod=importlib.import_module('experiments.qwen3_bonus.perm_launch')
    from experiments.qwen3_bonus.perm_calibration import fingerprint
    selection={'passed':True,'selected':{'repo_id':'Skywork/Skywork-Reward-V2-Qwen3-1.7B',
        'revision':'e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc', 'model_path':str(tmp_path/'rm'),
        'model_inventory_sha256':'a'*64,'calibration_path':str(tmp_path/'cal.json'),'calibration_sha256':'b'*64}}
    selection['selection_sha256']=fingerprint(selection)
    assert mod.validate_selection(selection)['revision']==selection['selected']['revision']
    selection['selected']['revision']='wrong'
    with pytest.raises(ValueError):mod.validate_selection(selection)


def test_launch_rejects_selection_without_measured_evidence():
    mod=importlib.import_module('experiments.qwen3_bonus.perm_launch')
    with pytest.raises(ValueError,match='evidence'):
        mod.verify_resource_evidence({'passed':True})


def test_cgroup_effective_memory_uses_current_process_limit(tmp_path):
    mod=importlib.import_module('experiments.qwen3_bonus.perm_launch')
    (tmp_path/'memory.max').write_text('8192\n')
    assert mod.cgroup_memory_limit(tmp_path)==8192
    (tmp_path/'memory.max').write_text('max\n')
    assert mod.cgroup_memory_limit(tmp_path) is None


def test_reward_metrics_record_conflict_denominator():
    torch=pytest.importorskip('torch')
    rows={'score':[.1,.8], 'preference':[.8,.2],'constraint':[0.,1.],
          'constraint_applicable':[True,True],'invalid':[False,False],'invalid_reason':['',''],
          'main':[0.,.9],'repeat_ratio':[0.,0.], 'length_checks':[[],[]], 'raw_rm':[3.,1.]}
    reward=api().PermReward(lambda *a,**kw:{'reward_tensor':torch.zeros(2,2),'reward_extra_info':rows})
    data=SimpleNamespace(batch={'responses':torch.ones(2,2),'response_mask':torch.ones(2,2)},
                         non_tensor_batch={'uid':['x','x']})
    reward(data)
    assert reward.last_metrics['reward/preference_constraint_conflict']==1.
    assert reward.last_metrics['reward/conflict_denominator']==1
