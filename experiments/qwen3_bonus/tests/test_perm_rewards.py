import math
import pytest
from experiments.qwen3_bonus import perm_rewards as p


def test_scalar_length_is_log_only_and_invalid_consistent():
    a=p.component_scores(0.,'有效回答', [{'kind':'max_chars','value':1}],{'median':0,'iqr':1})
    assert a['score']==.5 and not a['constraint_applicable'] and a['length_checks']==[False]
    b=p.component_scores(None,'',[],{'median':0,'iqr':1})
    assert b.keys()==a.keys() and b['score']==-1 and b['raw_rm'] is None
    assert p.component_scores(0.,'有效回答',[{'kind':'required_text','text':'缺失'}],{'median':0,'iqr':1})['main']==0
    with pytest.raises(ValueError):p.component_scores(math.nan,'有效回答',[],{'median':0,'iqr':1})
    with pytest.raises(ValueError):p.component_scores(0,'有效回答',[{'kind':'oops'}],{'median':0,'iqr':1})


def test_group_gate_shuffled_and_invalid():
    bad=p.component_scores(0,'有效回答',[{'kind':'required_text','text':'缺失'}],{'median':0,'iqr':1})
    invalid=p.component_scores(None,'',[],{'median':0,'iqr':1})
    rows,metrics=p.apply_group_gate([bad,invalid,bad,bad],['x','y','x','y'])
    assert [r['group_gate'] for r in rows]==[True,False,True,False]
    assert rows[1]['score']==-1 and metrics['gated_groups']==1
    with pytest.raises(ValueError):p.apply_group_gate([bad],['x'])


def test_rm_messages_retain_history_filter_system():
    assert p.reward_messages([{'role':'system','content':'s'},{'role':'user','content':'u'},{'role':'assistant','content':'h'},{'role':'user','content':'v'}],'a')==[{'role':'user','content':'u'},{'role':'assistant','content':'h'},{'role':'user','content':'v'},{'role':'assistant','content':'a'}]
    with pytest.raises(ValueError):p.reward_messages([{'role':'user','content':3}],'a')

def test_load_calibration_rejects_nonfinite_and_wrong_spec(tmp_path):
    import json
    path=tmp_path/'cal.json'
    path.write_text(json.dumps({'median':float('nan'),'iqr':1,'gate':{'passed':True}}))
    with pytest.raises(ValueError):p.load_calibration(str(path))

def test_calibration_requires_exact_dependency_set(tmp_path):
    import json
    path=tmp_path/'partial.json'
    path.write_text(json.dumps({'code_sha256':{'perm_rewards.py':'fake'}}))
    with pytest.raises(ValueError):p.load_calibration(str(path))

def test_inventory_rejects_unlisted_template_and_weights(tmp_path):
    import json,hashlib,struct
    from experiments.qwen3_bonus.perm_models import verify_model,CANDIDATES
    for n in ('config.json','tokenizer.json','tokenizer_config.json'):(tmp_path/n).write_text('{}')
    header=json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    (tmp_path/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)
    files={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in tmp_path.iterdir()}
    (tmp_path/'bonus_source.json').write_text(json.dumps({**CANDIDATES['1.7b'],'files':files}))
    verify_model(tmp_path)
    for name in ('chat_template.jinja','extra.safetensors','model.safetensors.index.json'):
        (tmp_path/name).write_text('unlisted')
        with pytest.raises(ValueError):verify_model(tmp_path)
        (tmp_path/name).unlink()

def test_inventory_rejects_truncated_safetensors_even_if_hash_updated(tmp_path):
    import json,struct
    from experiments.qwen3_bonus.perm_models import verify_model,CANDIDATES
    from experiments.qwen3_bonus.common import sha256
    for name in ('config.json','tokenizer.json','tokenizer_config.json'):(tmp_path/name).write_text('{}')
    header=json.dumps({'x':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}).encode()
    (tmp_path/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)
    (tmp_path/'bonus_source.json').write_text(json.dumps({**CANDIDATES['1.7b'],'files':{x.name:sha256(x) for x in tmp_path.iterdir()}}))
    with pytest.raises(ValueError):verify_model(tmp_path)
