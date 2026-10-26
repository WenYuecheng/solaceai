import pytest
from experiments.qwen3_bonus.cpu_preflight import validate_frozen_data
from experiments.qwen3_bonus.common import write_json

def test_preflight_rejects_changed_frozen_data(tmp_path, monkeypatch):
    import experiments.qwen3_bonus.cpu_preflight as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    folder=tmp_path/'artifacts/data';folder.mkdir(parents=True)
    (folder/'classification_train.jsonl').write_text('{}\n')
    # Avoid output_path's package boundary for this disposable test fixture.
    import json
    (folder/'dataset_manifest.json').write_text(json.dumps({'output_hashes':{'classification_train.jsonl':'wrong'}}))
    with pytest.raises(ValueError,match='fingerprint'):
        validate_frozen_data(folder)
