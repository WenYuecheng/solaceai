import json
import pytest


def test_model_review_is_explicit_never_counted_as_human():
    from experiments.qwen3_bonus.rewards import calibration_gate
    rows=[{'id':str(i),'reviewed':True,'reviewer':'Codex','review_type':'model',
        'expected':'a','score_a':1.,'score_b':0.} for i in range(20)]
    assert not calibration_gate(rows)['passed']
    result=calibration_gate(rows,allow_model_review=True)
    assert result['passed'] and result['human_reviewed']==0 and result['model_reviewed']==20


def test_adapter_snapshots_survive_rotation_of_full_checkpoints(tmp_path,monkeypatch):
    from experiments.qwen3_bonus import common
    from experiments.qwen3_bonus.snapshots import save_snapshot
    monkeypatch.setattr(common,'ROOT',tmp_path)
    original=tmp_path/'checkpoint';original.mkdir()
    (original/'adapter_config.json').write_text('{"r":8,"task_type":"CAUSAL_LM"}')
    (original/'adapter_model.safetensors').write_bytes(b'fixture-not-real-weights')
    saved=save_snapshot(original,tmp_path/'snapshots',4)
    (original/'adapter_model.safetensors').unlink()
    assert (saved/'adapter_model.safetensors').read_bytes()==b'fixture-not-real-weights'
    manifest=json.loads((saved/'snapshot.json').read_text())
    assert manifest['step']==4
    with pytest.raises(FileExistsError):save_snapshot(saved,tmp_path/'snapshots',4)


def test_chart_series_uses_actual_finite_logged_steps_only():
    from experiments.qwen3_bonus.plotting import series
    rows=[{'step':4,'metrics':{'reward_mean':.4}},{'step':8,'metrics':{'reward_mean':None}},
          {'step':12,'metrics':{'reward_mean':float('nan')}},{'step':16,'metrics':{'reward_mean':.6}}]
    assert series(rows,'reward_mean')==([4,16],[.4,.6])
    assert series(rows,'nonexistent')==([],[])
