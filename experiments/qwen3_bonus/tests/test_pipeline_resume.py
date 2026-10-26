import json
from experiments.qwen3_bonus.remote_pipeline import completed_emotion

def test_resume_rejects_missing_adapter_even_if_result_says_trained(tmp_path):
    (tmp_path/'training_result.json').write_text(json.dumps({'trained':True,'steps':5,'status':'completed','task':'four_class_emotion','adapter':str(tmp_path/'missing')}))
    assert not completed_emotion(tmp_path)

def test_resume_accepts_completed_classifier_with_local_weights(tmp_path):
    adapter=tmp_path/'adapter';adapter.mkdir()
    (adapter/'adapter_config.json').write_text('{"r":8}')
    (adapter/'adapter_model.safetensors').write_bytes(b'actual adapter placeholder for path validation')
    (tmp_path/'training_result.json').write_text(json.dumps({'trained':True,'steps':150,'status':'completed','task':'four_class_emotion','adapter':str(adapter)}))
    assert completed_emotion(tmp_path)
