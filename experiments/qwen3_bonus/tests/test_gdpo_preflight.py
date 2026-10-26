import importlib

import pytest


def api():return importlib.import_module('experiments.qwen3_bonus.gdpo_preflight')


def test_calibration_requires_explicit_model_review_before_any_scoring():
    pairs=[{'id':str(i),'prompt':[],'a':'a','b':'b','expected':'a','reviewed':True,
            'review_type':'model','reviewer':'actual-model','review_rationale':'fixture'} for i in range(20)]
    seen=[]
    report=api().calibrate_pairs(pairs,lambda prompt,answer:seen.append(answer) or (2. if answer=='a' else 0.))
    assert report['gate']['passed'] and report['raw_scores']['count']==40
    assert report['raw_scores']['median']==1 and report['raw_scores']['iqr']==2
    assert report['gate']['model_reviewed']==20 and report['gate']['human_reviewed']==0
    bad=[{**row,'review_type':'human'} for row in pairs]
    with pytest.raises(ValueError):api().calibrate_pairs(bad,lambda *args:pytest.fail('must reject before inference'))
    assert len(seen)==40


def test_model_inventory_cannot_claim_training_cuda_on_metal():
    inventory=api().runtime_inventory()
    assert 'python_executable' in inventory
    assert set(('mlx','mlx-lm','vllm','vllm-metal','torch','transformers')).issubset(inventory['packages'])
    assert inventory['training_runtime']['scope']=='separate Linux CUDA environment required'
    assert inventory['genuine_inference'] is False


def test_static_manifest_rejects_unpinned_or_wrong_actual_rank(tmp_path,monkeypatch):
    np=pytest.importorskip('numpy');pytest.importorskip('safetensors')
    from safetensors.numpy import save_file
    from experiments.qwen3_bonus.models import SOURCES
    from experiments.qwen3_bonus import common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    def model(name,kind,mlx=False):
        path=tmp_path/name;path.mkdir();common.write_json(path/'config.json',{'model_type':'qwen3','quantization':{'bits':4}})
        (path/'model.safetensors').write_bytes(b'not-loaded-model-fixture')
        common.write_json(path/'bonus_source.json',{**SOURCES[kind], 'quantization_bits':4 if mlx else None,
          'files':{n:common.sha256(path/n) for n in ('config.json','model.safetensors')}})
        return path
    base=model('base','base');reward=model('reward','reward');mlx=model('mlx','base',True)
    data=__import__('json').loads((mlx/'bonus_source.json').read_text());data['source_inventory_sha256']=common.sha256(base/'bonus_source.json');common.write_json(mlx/'bonus_source.json',data)
    path=tmp_path/'adapter';path.mkdir()
    common.write_json(path/'adapter_config.json',{'r':8,'lora_alpha':16,'target_modules':['q_proj','v_proj'],
      'base_model_name_or_path':SOURCES['base']['repo_id'],'revision':SOURCES['base']['revision']})
    tensors={f'base_model.model.model.layers.0.self_attn.{m}.lora_{side}.weight':np.zeros(shape,np.float32)
             for m in ('q_proj','v_proj') for side,shape in [('A',(8,16)),('B',(16,8))]}
    save_file(tensors,str(path/'adapter_model.safetensors'))
    assert api().static_preflight(base,reward,mlx,{'current':path})['artifacts_valid']
    (base/'model.safetensors').unlink()
    with pytest.raises(ValueError):api().static_preflight(base,reward,mlx,{'current':path})
    metadata_only=api().static_preflight(base,reward,mlx,{'current':path},allow_missing_raw_weights=True)
    assert metadata_only['raw_weights_verified'] is False and metadata_only['converted_base_verified'] is True
    tensors[next(iter(tensors))]=np.zeros((7,16),np.float32);save_file(tensors,str(path/'adapter_model.safetensors'))
    with pytest.raises(ValueError,match='rank'):api().static_preflight(base,reward,mlx,{'current':path},allow_missing_raw_weights=True)


def test_calibration_exports_training_normalization_contract():
    pairs=[{'id':str(i),'prompt':[],'a':'a','b':'b','expected':'a','reviewed':True,
            'review_type':'model','reviewer':'actual-model','review_rationale':'fixture'} for i in range(20)]
    report=api().calibrate_pairs(pairs,lambda prompt,answer:2. if answer=='a' else 0.)
    assert report['median']==1 and report['iqr']==2


def test_runtime_inventory_distinguishes_current_and_metal_environments():
    inventory=api().runtime_inventory()
    assert inventory['tools_runtime']['python_executable']==inventory['python_executable']
    assert 'python_executable' in inventory['metal_runtime']
    assert 'inventory_error' in inventory['metal_runtime']


def test_calibration_binds_model_source_hash_and_rejects_null_reviewer():
    from experiments.qwen3_bonus.models import SOURCES
    pairs=[{'id':str(i),'prompt':[],'a':'a','b':'b','expected':'a','reviewed':True,
            'review_type':'model','reviewer':'actual-model','review_rationale':'fixture'} for i in range(20)]
    report=api().calibrate_pairs(pairs,lambda prompt,answer:2. if answer=='a' else 0.,
                                 model_inventory_sha256='a'*64)
    assert report['model_inventory_sha256']=='a'*64
    assert report['reward_model_revision']==SOURCES['reward']['revision']
    pairs[0]['reviewer']=None
    with pytest.raises(ValueError):api().calibrate_pairs(pairs,lambda *a:pytest.fail('reject before score'))


def test_inventory_only_raw_checks_never_claim_missing_weights_verified(tmp_path,monkeypatch):
    from experiments.qwen3_bonus import common
    from experiments.qwen3_bonus.models import SOURCES
    monkeypatch.setattr(common,'ROOT',tmp_path)
    common.write_json(tmp_path/'config.json',{'model_type':'qwen3','hidden_size':16,'head_dim':4,
       'num_attention_heads':4,'num_key_value_heads':2,'num_hidden_layers':1})
    (tmp_path/'tokenizer.json').write_text('fixture tokenizer')
    common.write_json(tmp_path/'bonus_source.json',{**SOURCES['base'],'files':{
       'config.json':common.sha256(tmp_path/'config.json'),'tokenizer.json':common.sha256(tmp_path/'tokenizer.json'),
       'model-00001.safetensors':'b'*64}})
    with pytest.raises(ValueError):api().verify_raw_base(tmp_path)
    source,report=api().verify_raw_base(tmp_path,allow_missing_raw_weights=True)
    assert source['revision']==SOURCES['base']['revision']
    assert report['raw_weights_verified'] is False and report['raw_inventory_verified'] is True
    assert report['missing_raw_weights']==['model-00001.safetensors']
    (tmp_path/'tokenizer.json').write_text('changed')
    with pytest.raises(ValueError):api().verify_raw_base(tmp_path,allow_missing_raw_weights=True)


def test_inventory_only_probe_adapters_use_config_without_raw_weights(tmp_path,monkeypatch):
    np=pytest.importorskip('numpy');pytest.importorskip('safetensors')
    from safetensors import safe_open
    from experiments.qwen3_bonus import common
    from experiments.qwen3_bonus.models import SOURCES
    monkeypatch.setattr(common,'ROOT',tmp_path)
    raw=tmp_path/'raw';raw.mkdir()
    common.write_json(raw/'config.json',{'model_type':'qwen3','hidden_size':16,'head_dim':4,
       'num_attention_heads':4,'num_key_value_heads':2,'num_hidden_layers':1})
    common.write_json(raw/'bonus_source.json',{**SOURCES['base'],'files':{
       'config.json':common.sha256(raw/'config.json'),'model.safetensors':'b'*64}})
    paths=api().create_probe_adapters(raw,tmp_path/'dummy',allow_missing_raw_weights=True)
    import json
    from pathlib import Path
    for directory in paths.values():
        directory=Path(directory)
        assert json.loads((directory/'bonus_adapter.json').read_text())['trained'] is False
        with safe_open(str(directory/'adapter_model.safetensors'),framework='numpy') as weights:
            assert weights.get_tensor('base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight').shape==(8,8)


def test_probe_checks_current_grpo32_and_dummy_and_requires_all_completed(tmp_path,monkeypatch):
    pytest.importorskip('fastapi')
    from experiments.qwen3_bonus import common,probe,serving,compatibility
    monkeypatch.setattr(common,'ROOT',tmp_path)
    config=tmp_path/'config.json';config.write_text(__import__('json').dumps({
       'model_path':str(tmp_path/'mlx'),'emotion_adapter':str(tmp_path/'emotion'),
       'response_adapter':str(tmp_path/'current')}))
    seen=[]
    async def fake_probe(cfg,**kwargs):
        seen.append(cfg.response_adapter)
        return {'genuine_inference':True,'smoke_passed':'grpo' not in cfg.response_adapter,
                'error_type':None,'requests':[{}]*4}
    monkeypatch.setattr(probe,'live_probe',fake_probe)
    monkeypatch.setattr(serving,'validate_artifacts',lambda cfg:None)
    monkeypatch.setattr(compatibility,'validate_temporary',lambda cfg:None)
    monkeypatch.setattr(api(),'static_preflight',lambda *args,**kwargs:{'raw_weights_verified':False})
    monkeypatch.setattr(api(),'create_probe_adapters',lambda *args,**kwargs:{'emotion':str(tmp_path/'dummy-e'),'response':str(tmp_path/'dummy-r')})
    report=api().run_probe(config,tmp_path/'raw',tmp_path/'result',reward=tmp_path/'reward',
                          grpo_adapter=tmp_path/'grpo',allow_missing_raw_weights=True)
    assert seen==[str(tmp_path/'current'),str(tmp_path/'grpo'),str(tmp_path/'dummy-r')]
    assert report['genuine_inference'] is False and report['compatibility_passed'] is False
    assert 'grpo32_adapter' in report
