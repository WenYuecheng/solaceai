from experiments.qwen3_bonus import common
import pytest


def test_probe_success_requires_every_actual_response():
    from experiments.qwen3_bonus.perm_preflight import probe_completed
    good={'genuine_inference':True,'smoke_passed':True,'error_type':None,
          'requests':[{'content':'具体回应','valid_format':True}]*2}
    assert probe_completed(good)
    assert not probe_completed({**good,'requests':good['requests'][:1]})
    assert not probe_completed({**good,'error_type':'EngineFailed'})
    assert not probe_completed({**good,'genuine_inference':False})


def test_live_probe_requires_explicit_execution_before_importing_engine(tmp_path):
    from experiments.qwen3_bonus.perm_preflight import run_probe
    with pytest.raises(ValueError,match='execute'):
        run_probe(tmp_path/'config.json',tmp_path/'base',tmp_path/'out',execute=False)


def test_historical_grpo_requires_matching_actual_snapshot_and_completed_run(tmp_path,monkeypatch):
    import json
    from experiments.qwen3_bonus import perm_preflight as mod
    from experiments.qwen3_bonus.models import SOURCES
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    adapter=tmp_path/'adapter';adapter.mkdir();(adapter/'adapter_model.safetensors').write_bytes(b'verified actual fixture weights')
    run=tmp_path/'artifacts/remote/received/weste-final/extracted/artifacts/runs/response-weste3'
    snap=run/'adapter_snapshots/step-000032';snap.mkdir(parents=True)
    (snap/'adapter_model.safetensors').write_bytes((adapter/'adapter_model.safetensors').read_bytes())
    (run/'training_result.json').write_text(json.dumps({'trained':True,'steps':32,'base_revision':SOURCES['base']['revision']}))
    assert mod.historical_grpo_provenance(adapter)['steps']==32
    (snap/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError):mod.historical_grpo_provenance(adapter)
