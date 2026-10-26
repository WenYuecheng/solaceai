import json
from pathlib import Path
import pytest


def test_adapter_export_rejects_untrained_and_wrong_base(tmp_path):
    from experiments.qwen3_bonus.deployment import validate_training_adapter
    run=tmp_path/'run';run.mkdir()
    (run/'training_result.json').write_text(json.dumps({'trained':False}))
    with pytest.raises(ValueError,match='trained'):
        validate_training_adapter(run,'response')


def test_adapter_export_locates_last_real_verl_step_not_lexicographic(tmp_path):
    from experiments.qwen3_bonus.deployment import validate_training_adapter
    run=tmp_path/'run';run.mkdir()
    (run/'training_result.json').write_text(json.dumps({'trained':True,'steps':10}))
    for step in (8,10):
        path=run/f'checkpoints/global_step_{step}/actor/lora_adapter';path.mkdir(parents=True)
        (path/'adapter_config.json').write_text(json.dumps({'r':8,'lora_alpha':16,
            'target_modules':['q_proj','v_proj'],'task_type':'CAUSAL_LM','bias':'none'}))
        (path/'adapter_model.safetensors').write_bytes(b'nonempty-test-fixture-not-a-real-model')
    adapter,info=validate_training_adapter(run,'response')
    assert 'global_step_10' in str(adapter)
    assert info['r']==8


def test_adapter_export_rejects_extra_trainable_modules(tmp_path):
    from experiments.qwen3_bonus.deployment import validate_training_adapter
    run=tmp_path/'run';path=run/'adapter';path.mkdir(parents=True)
    (run/'training_result.json').write_text(json.dumps({'trained':True,'steps':1}))
    (path/'adapter_config.json').write_text(json.dumps({'r':8,'lora_alpha':16,
        'target_modules':['q_proj','v_proj','lm_head'],'task_type':'CAUSAL_LM'}))
    (path/'adapter_model.safetensors').write_bytes(b'x')
    with pytest.raises(ValueError,match='layout'):validate_training_adapter(run,'emotion')
