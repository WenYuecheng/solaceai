import pytest
from experiments.qwen3_bonus import perm_calibration as p

def test_normalization_valid_only_and_gate():
    assert p.normalization([None,0.,1.,2.,3.])=={'median':1.5,'iqr':1.5,'valid_raw_count':4}
    with pytest.raises(ValueError):p.normalization([float('nan')])
    rows=[dict(id=str(i),expected='a',score_a=1,score_b=0) for i in range(20)]
    assert p.final_gate(rows,[str(i) for i in range(8)])['passed']
    rows[0]['score_b']=1
    assert p.final_gate(rows,[str(i) for i in range(8)])['ties']==1

def test_selection_rejects_unknown_offload_and_budget():
    candidate={'repo_id':p.CANDIDATES['1.7b']['repo_id'],'revision':p.CANDIDATES['1.7b']['revision'],'model_path':'/m','model_inventory_sha256':'h','calibration_path':'/c','calibration_sha256':'s','gate_passed':True,'peak_rm_bytes':100,'seconds_per_response':1}
    resources=dict(physical_memory_bytes=1000,cgroup_limit_bytes=800,runtime_bytes=100,actor_ref_offload_bytes=100,offload_verified=True,remaining_training_yuan=100,hourly_yuan=1,prep_setup_save_reserve_yuan=1)
    assert p.estimate_untrusted([candidate],resources)['estimated_passed']
    resources['offload_verified']=False
    assert not p.estimate_untrusted([candidate],resources)['estimated_passed']

def test_selection_uses_cgroup_reserve_and_prefers_small_when_large_fails():
    base=dict(model_path='/m',model_inventory_sha256='h',calibration_path='/c',calibration_sha256='s',gate_passed=True,peak_rm_bytes=10,seconds_per_response=1)
    large={**base,**p.CANDIDATES['1.7b'],'peak_rm_bytes':600}
    small={**base,**p.CANDIDATES['0.6b']}
    resources=dict(physical_memory_bytes=10000,cgroup_limit_bytes=800,runtime_bytes=100,actor_ref_offload_bytes=100,offload_verified=True,remaining_training_yuan=100,hourly_yuan=1,prep_setup_save_reserve_yuan=1)
    result=p.estimate_untrusted([large,small],resources)
    assert result['estimated_selected']['repo_id']==small['repo_id']
    resources['hourly_yuan']=1e9
    assert p.estimate_untrusted([large,small],resources)['estimated_selected'] is None

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,'800'])
def test_resource_cgroup_rejects_nonfinite_bool_and_string(bad):
    resources=dict(physical_memory_bytes=1000,cgroup_limit_bytes=bad,runtime_bytes=100,actor_ref_offload_bytes=100,offload_verified=True,remaining_training_yuan=100,hourly_yuan=1,prep_setup_save_reserve_yuan=1)
    with pytest.raises(ValueError):p.select_model([],resources)


def test_selection_rejects_fabricated_candidate():
    candidate={**p.CANDIDATES['1.7b'],'model_path':'/missing','model_inventory_sha256':'invented','calibration_path':'/missing','calibration_sha256':'invented','gate_passed':True,'peak_rm_bytes':0,'seconds_per_response':0}
    with pytest.raises((ValueError,FileNotFoundError)):p.make_candidate(candidate['calibration_path'])

@pytest.fixture
def bound_candidate(tmp_path):
    import json,struct
    from pathlib import Path
    from experiments.qwen3_bonus.perm_rewards import CODE_DEPENDENCIES
    model=tmp_path/'model';model.mkdir()
    for n in ('config.json','tokenizer.json','tokenizer_config.json'):(model/n).write_text('{}')
    header=json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    (model/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)
    (model/'bonus_source.json').write_text(json.dumps({**p.CANDIDATES['1.7b'],'files':{x.name:p.sha256(x) for x in model.iterdir()}}))
    identity=p.verify_model(model)
    manifest_path=Path(p.__file__).parent/'artifacts/perm-grpo-20261024/data/dataset_manifest.json'
    manifest=json.loads(manifest_path.read_text());pairs=json.loads((manifest_path.parent/'calibration_pairs.json').read_text())['pairs']
    raw=[];final=[]
    for r in pairs:
        scores={ch:(10. if ch==r['expected'] else -10.) for ch in ('a','b')}
        raw.append({'id':r['id'],'expected':r['expected'],'raw_a':scores['a'],'raw_b':scores['b']})
    norm=p.normalization([x['raw_'+ch] for x in raw for ch in ('a','b')])
    for r,x in zip(pairs,raw):
        rec=[p.component_scores(x['raw_'+ch],r[ch],r['constraints'],norm) for ch in ('a','b')]
        rec,_=p.apply_group_gate(rec,[r['id']]*2)
        final.append({'id':r['id'],'expected':r['expected'],'score_a':rec[0]['score'],'score_b':rec[1]['score']})
    gate=p.final_gate(final,manifest['task_delivery_ids'])
    report={**norm,'reward_spec':p.REWARD_SPEC,'input_format':p.INPUT_FORMAT,'input_format_sha256':p.fingerprint(p.INPUT_FORMAT),'gate':gate,'methods':{'harmonic':{'gate':gate,'pairs':final}},'raw_ordering':raw,'model':identity,'data_manifest_path':str(manifest_path),'data_manifest_sha256':p.sha256(manifest_path),'labels_sha256':p.fingerprint([{k:r[k] for k in ('id','expected','reviewed','reviewer','review_type')} for r in pairs]),'code_sha256':{n:p.sha256(Path(p.__file__).with_name(n)) for n in CODE_DEPENDENCIES},'benchmark':{'peak_process_rss_bytes':100,'eight_longest':[{'seconds':1}]*8,'seconds_per_response_conservative':1}}
    path=tmp_path/'calibration.json';path.write_text(json.dumps(report))
    return p.make_candidate(path),path


def test_verified_selection_requires_bound_report_and_runtime_snapshot(bound_candidate,tmp_path):
    import json
    candidate,path=bound_candidate
    resources=dict(physical_memory_bytes=1000,cgroup_limit_bytes=800,runtime_bytes=100,actor_ref_offload_bytes=100,offload_verified=True,remaining_training_yuan=100,hourly_yuan=1,prep_setup_save_reserve_yuan=1)
    evidence=tmp_path/'resource.json';evidence.write_text(json.dumps({**resources,'phase':'full_step','memory_metric_definitions':{'runtime_bytes':'non-RM peak','actor_ref_offload_bytes':'actor/reference host resident'}}))
    resources.update(evidence_path=str(evidence),evidence_sha256=p.sha256(evidence))
    result=p.select_model([candidate],resources)
    assert result['passed'] and result['candidates']==[candidate] and result['resources']==resources
    candidate['seconds_per_response']=0
    with pytest.raises(ValueError):p.select_model([candidate],resources)
    report=json.loads(path.read_text());report['methods']['harmonic']['pairs'][0]['score_a']=999
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):p.make_candidate(path)

def test_dependency_omission_and_runtime_evidence_drift_fail_closed(bound_candidate,tmp_path):
    import json
    candidate,path=bound_candidate
    report=json.loads(path.read_text());report['code_sha256'].pop('rewards.py');path.write_text(json.dumps(report))
    with pytest.raises(ValueError):p.make_candidate(path)
    resources=dict(physical_memory_bytes=1000,cgroup_limit_bytes=800,runtime_bytes=True,actor_ref_offload_bytes=100,offload_verified=True,remaining_training_yuan=100,hourly_yuan=1,prep_setup_save_reserve_yuan=1)
    with pytest.raises(ValueError):p.select_model([],resources)


def test_legitimate_failed_gate_remains_candidate_evidence(bound_candidate):
    import json
    from pathlib import Path
    from experiments.qwen3_bonus.perm_rewards import load_calibration
    _,path=bound_candidate
    report=json.loads(path.read_text())
    pairs=json.loads((Path(report['data_manifest_path']).parent/'calibration_pairs.json').read_text())['pairs']
    for row in report['raw_ordering']:
        row['raw_a'],row['raw_b']=row['raw_b'],row['raw_a']
    final=[]
    for pair,raw in zip(pairs,report['raw_ordering']):
        records=[p.component_scores(raw['raw_'+ch],pair[ch],pair['constraints'],report) for ch in ('a','b')]
        records,_=p.apply_group_gate(records,[pair['id']]*2)
        final.append({'id':pair['id'],'expected':pair['expected'],'score_a':records[0]['score'],'score_b':records[1]['score']})
    manifest=json.loads(Path(report['data_manifest_path']).read_text())
    gate=p.final_gate(final,manifest['task_delivery_ids'])
    assert not gate['passed']
    report.update(gate=gate,methods={'harmonic':{'gate':gate,'pairs':final}})
    path.write_text(json.dumps(report))
    candidate=p.make_candidate(path)
    assert candidate['gate_passed'] is False
    load_calibration.cache_clear()
    with pytest.raises(ValueError):load_calibration(str(path))


def test_reward_download_sidecar_ignored_but_loader_injection_rejected(bound_candidate):
    from pathlib import Path
    candidate,_=bound_candidate
    model=Path(candidate['model_path'])
    (model/'model.safetensors.ranges.json').write_text('{}')
    p.verify_model(model)
    for name in ('chat_template.jinja','model.safetensors.index.json','extra.safetensors'):
        injected=model/name;injected.write_text('{}')
        with pytest.raises(ValueError):p.verify_model(model)
        injected.unlink()
