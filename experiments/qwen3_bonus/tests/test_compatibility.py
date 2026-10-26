"""Real safetensor serialization of a tiny shape fixture, never model inference."""
import json
from types import SimpleNamespace
import pytest


def test_temporary_adapters_use_qwen_gqa_shapes_and_remain_untrained(tmp_path,monkeypatch):
    pytest.importorskip('safetensors')
    from safetensors import safe_open
    from experiments.qwen3_bonus import common
    from experiments.qwen3_bonus.compatibility import create_temporary_adapters,validate_temporary
    from experiments.qwen3_bonus.models import SOURCES
    monkeypatch.setattr(common,'ROOT',tmp_path)
    raw=tmp_path/'raw';raw.mkdir()
    config={'model_type':'qwen3','hidden_size':16,'head_dim':4,'num_attention_heads':4,
            'num_key_value_heads':2,'num_hidden_layers':1}
    common.write_json(raw/'config.json',config)
    (raw/'model.safetensors').write_bytes(b'fixture-not-real-base')
    common.write_json(raw/'bonus_source.json',{**SOURCES['base'],'files':{
        name:common.sha256(raw/name) for name in ('config.json','model.safetensors')}})
    adapters=create_temporary_adapters(raw,tmp_path/'adapters')
    for directory in adapters.values():
        from pathlib import Path
        path=Path(directory)
        assert json.loads((path/'bonus_adapter.json').read_text())['trained'] is False
        with safe_open(str(path/'adapter_model.safetensors'),framework='numpy') as tensors:
            prefix='base_model.model.model.layers.0.self_attn.'
            assert tensors.get_tensor(prefix+'q_proj.lora_B.weight').shape==(16,8)
            assert tensors.get_tensor(prefix+'v_proj.lora_B.weight').shape==(8,8)
            assert not tensors.get_tensor(prefix+'v_proj.lora_A.weight').any()
    mlx=tmp_path/'mlx';mlx.mkdir()
    common.write_json(mlx/'config.json',{**config,'quantization':{'bits':4}})
    (mlx/'model.safetensors').write_bytes(b'fixture-not-real-base')
    common.write_json(mlx/'bonus_source.json',{**SOURCES['base'],'quantization_bits':4,'files':{
        name:common.sha256(mlx/name) for name in ('config.json','model.safetensors')}})
    serving=SimpleNamespace(model_path=str(mlx),emotion_adapter=adapters['emotion'],response_adapter=adapters['response'])
    validate_temporary(serving)
    (mlx/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):validate_temporary(serving)
