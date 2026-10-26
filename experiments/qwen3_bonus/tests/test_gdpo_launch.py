import json
import sys
from pathlib import Path
import pytest


def test_platform_wrapper_verifies_script_before_any_execution(tmp_path):
    from experiments.qwen3_bonus.gdpo_platform import verify_platform_script
    script=tmp_path/'shutdown';script.write_text('echo stop\n')
    with pytest.raises(ValueError,match='hash'):verify_platform_script(script,'0'*64)


def test_dataset_verifier_fails_for_changed_frozen_artifact(tmp_path):
    from experiments.qwen3_bonus.gdpo_launch import verify_frozen_data
    from experiments.qwen3_bonus.common import sha256
    p=tmp_path/'response_train.jsonl';p.write_text('{}\n')
    manifest={'files':{'response_train.jsonl':sha256(p)}}
    (tmp_path/'dataset_manifest.json').write_text(json.dumps(manifest))
    p.write_text('changed')
    with pytest.raises(ValueError):verify_frozen_data(tmp_path)


def test_remote_bundle_never_contains_models_environments_or_local_serving_secrets(tmp_path,monkeypatch):
    from experiments.qwen3_bonus.gdpo_remote import build_bundle
    import tarfile
    from experiments.qwen3_bonus import common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    # ROOT argument is explicit so the test uses only its tiny fixture tree.
    (tmp_path/'gdpo.py').write_text('pass\n')
    (tmp_path/'artifacts/models').mkdir(parents=True)
    (tmp_path/'artifacts/models/secret').write_text('never copy')
    (tmp_path/'configs').mkdir();(tmp_path/'configs/serving.local.json').write_text('private')
    target=tmp_path/'bundle.tar.gz'
    result=build_bundle(target,experiment_root=tmp_path)
    with tarfile.open(target) as archive:names=archive.getnames()
    assert 'experiments/qwen3_bonus/gdpo.py' in names
    assert all('models' not in n and 'serving.local' not in n for n in names)
    assert len(result['sha256'])==64
