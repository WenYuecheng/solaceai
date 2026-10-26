from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import pytest


def test_adapter_routes_full_dataproto_then_restores_even_on_exception():
    torch=pytest.importorskip('torch')
    from experiments.qwen3_bonus.gdpo_compat import install_gdpo
    original=lambda *a,**k: (_ for _ in ()).throw(AssertionError('GRPO called'))
    module=SimpleNamespace(compute_advantage=original)
    data=SimpleNamespace(batch={'response_mask':torch.ones((4,2))},non_tensor_batch={
        'uid':np.array(['a','a','b','b']), 'preference':np.array([.9,.1,.8,.2]),
        'constraint':np.array([1,0,0,1]),'non_repetition':np.ones(4), 'invalid_reason':np.array(['']*4)})
    config={'bonus_adv_estimator':'gdpo'}
    with pytest.raises(RuntimeError,match='body'):
        with install_gdpo(module,config) as adapter:
            assert module.compute_advantage is not original
            result=module.compute_advantage(data,adv_estimator='grpo',config=config)
            assert adapter.calls==1
            assert result is data and 'advantages' in data.batch
            assert not torch.equal(data.batch['advantages'][0],data.batch['advantages'][2])
            raise RuntimeError('body')
    assert module.compute_advantage is original


def test_gdpo_recipe_is_explicit_without_changing_legacy_limits(tmp_path):
    from experiments.qwen3_bonus.gdpo_entry import gdpo_overrides
    from experiments.qwen3_bonus.training import verl_overrides
    args=gdpo_overrides(tmp_path/'b',tmp_path/'r',tmp_path/'d',tmp_path/'o',tmp_path/'c',64)
    assert 'algorithm.adv_estimator=grpo' in args
    assert '+algorithm.bonus_adv_estimator=gdpo' in args
    assert 'data.train_batch_size=4' in args
    assert 'actor_rollout_ref.actor.ppo_mini_batch_size=4' in args
    assert 'trainer.total_training_steps=64' in args
    assert 'trainer.save_freq=-1' in args
    assert not any(x.startswith('actor_rollout_ref.model.seed=') for x in args)
    assert '+actor_rollout_ref.model.seed=3407' in args
    with pytest.raises(ValueError):verl_overrides(tmp_path,tmp_path,tmp_path,tmp_path,64)


def test_missing_component_never_falls_back_to_grpo():
    torch=pytest.importorskip('torch')
    from experiments.qwen3_bonus.gdpo_compat import install_gdpo
    module=SimpleNamespace(compute_advantage=lambda *a,**k: pytest.fail('fallback'))
    with install_gdpo(module,{'bonus_adv_estimator':'gdpo'}):
        data=SimpleNamespace(batch={'response_mask':torch.ones(2,2)},non_tensor_batch={'uid':['a','a']})
        with pytest.raises(ValueError,match='component'):
            module.compute_advantage(data,config={'bonus_adv_estimator':'gdpo'})


def test_real_hydra_composes_the_entire_pinned_gdpo_recipe(tmp_path):
    pytest.importorskip('hydra')
    from hydra import compose,initialize_config_dir
    from experiments.qwen3_bonus.common import ROOT
    from experiments.qwen3_bonus.gdpo_entry import gdpo_overrides
    with initialize_config_dir(config_dir=str(ROOT/'vendor/verl-config-v0.5.0'),version_base=None):
        cfg=compose(config_name='ppo_trainer',overrides=gdpo_overrides(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path))
    assert cfg.algorithm.bonus_adv_estimator=='gdpo'
    assert cfg.data.seed==3407
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size*cfg.actor_rollout_ref.rollout.n==8
