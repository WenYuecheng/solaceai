"""Synthetic fixtures only: no inference or reported experimental results."""
import copy
import importlib
import json
import pytest
from experiments.qwen3_bonus.tests.test_gdpo_evaluation import fixture


def api(): return importlib.import_module('experiments.qwen3_bonus.perm_evaluation')

def bundle(n=10, split='dev'):
    rows,caps=fixture(n)
    for row in caps:
        if row['variant']=='gdpo': row['variant']='perm_grpo'
        row.update(chat_template_sha256='a'*64,quantization='4bit')
    return api().freeze_captures(rows,caps,split=split)

def reviews(sheet):
    return {'sheet_sha256':api().digest_json(sheet),'review_type':'model','reviewer':'fixture',
      'provenance':{'source':'unit fixture'},'rows':[{'id':r['id'],'choice':'tie','rationale':'fixture',
      **{s:{'semantic_completed':True,'severe':[],**dict.fromkeys(api().AXES,3)} for s in ('a','b')}} for r in sheet['rows']]}

def candidate(weights, step=1, quality=3):
    from experiments.qwen3_bonus.common import sha256
    mod=api();frozen=bundle(32,'dev')
    for row in frozen['captures']:
        if row['variant']=='perm_grpo':row['adapter_sha256']=sha256(weights)
    # Re-freeze after binding captures to the actual checkpoint.
    frozen=mod.freeze_captures(frozen['cases'],frozen['captures'],split='dev')
    sheet,key=mod.make_pair_sheets(frozen);review=reviews(sheet)
    for row in review['rows']:
        for side in ('a','b'):row[side].update(dict.fromkeys(mod.AXES,quality))
    manifest_path=weights.parent/'fixture-dev'/'dataset_manifest.json'
    manifest_path.parent.mkdir(exist_ok=True)
    dev=manifest_path.parent/'response_dev.jsonl'
    if not manifest_path.exists():
        dev.write_text(''.join(json.dumps(row)+'\n' for row in frozen['cases']))
        manifest_path.write_text(json.dumps({'frozen':True,'files':{'response_dev.jsonl':{'sha256':sha256(dev)}}}))
    return mod.build_dev_candidate(frozen,sheet,key,review,weights,step,dev_manifest_path=manifest_path)

def test_four_axes_required_and_mean_used_in_dev_selection(tmp_path):
    mod=api(); sheet,key=mod.make_pair_sheets(bundle()); review=reviews(sheet)
    mod.validate_reviews(sheet,review)
    for value in (True,0,6,float('nan'),None):
        bad=copy.deepcopy(review);bad['rows'][0]['a'][mod.AXES[0]]=value
        with pytest.raises(ValueError): mod.validate_reviews(sheet,bad)
    weights=tmp_path/'weights';weights.write_bytes(b'fixture')
    candidates=[candidate(weights,step,q) for step,q in [(10,2),(20,4),(30,4)]]
    assert mod.select_checkpoint(candidates)['step']==20
    candidates[0]['split']='test'
    with pytest.raises(ValueError):mod.select_checkpoint(candidates)

def test_blind_sheets_and_final_denominator_full_item_scores():
    mod=api(); frozen=bundle(80,'test');sheet,key=mod.make_pair_sheets(frozen)
    assert frozen['actual_algorithm']=='perm_grpo'
    assert len(sheet['rows'])==288
    assert all(not {'case_id','variant','comparison'} & r.keys() for r in sheet['rows'])
    result=mod.evaluate_reviews(frozen,sheet,key,reviews(sheet))
    assert result['main_contrast']=='perm_grpo vs current'
    assert result['contrasts']['current']['n']==80
    assert result['contrasts']['current']['preference_rate']==.5
    assert result['answer_summaries']['perm_grpo']['n']==80
    assert result['answer_summaries']['perm_grpo']['emotional_understanding']==3
    assert len(result['per_item_scoring'])==240
    assert result['rechecks']['count']==48
    bad=copy.deepcopy(frozen);bad['actual_algorithm']='gdpo'
    with pytest.raises(ValueError):mod.make_pair_sheets(bad)

def test_protocol_requires_fixed_seed_template_and_quantized_base():
    mod=api(); frozen=bundle()
    with pytest.raises(ValueError):mod.make_pair_sheets(frozen,seed=2)
    assert 'semantic_completed' in mod.protocol_document()['rubric']

def test_freeze_selection_and_final_adapter_verification(tmp_path,monkeypatch):
    from experiments.qwen3_bonus import common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    mod=api();weights=tmp_path/'weights';weights.write_bytes(b'fixture')
    c=candidate(weights)
    path=tmp_path/'selected.json';mod.freeze_selection([c],path)
    assert mod.verify_selection(path)['actual_algorithm']=='perm_grpo'
    frozen=bundle(80,'test')
    with pytest.raises(ValueError): mod.verify_test_selection(frozen,path)
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError):mod.verify_selection(path)

def test_capture_requires_constant_explicit_template_and_quantization():
    mod=api();rows,caps=fixture()
    for c in caps:
        if c['variant']=='gdpo':c['variant']='perm_grpo'
    with pytest.raises(ValueError):mod.freeze_captures(rows,caps,split='dev')
    for c in caps:c.update(chat_template_sha256='a'*64,quantization='4bit')
    mod.freeze_captures(rows,caps,split='dev')
    caps[-1]['chat_template_sha256']='f'*64
    with pytest.raises(ValueError):mod.freeze_captures(rows,caps,split='dev')

def test_actual_training_plots_and_dev_curve_missing_are_explicit(tmp_path,monkeypatch):
    pytest.importorskip('matplotlib')
    from experiments.qwen3_bonus import common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    mod=api()
    logs=[{'step':1,'elapsed_seconds':12,'metrics':{'reward/main_mean':.4,
      'reward/constraint_mean':1.,'advantage/nonzero_ratio':1.,'timing_s/cpu_reward':2.}}]
    report=mod.plot_training_logs(logs,tmp_path/'figures')
    assert 'reward/main_mean' not in report['missing']
    assert 'timing_s/update_actor' in report['missing']
    missing=mod.plot_dev_curve([],logs,tmp_path/'missing-curve')
    assert missing['missing']==['dev_snapshots']
    candidates=[{'split':'dev','step':1,**dict.fromkeys(mod.AXES,4)}]
    curve=mod.plot_dev_curve(candidates,logs,tmp_path/'curve')
    assert curve['points'][0]['elapsed_seconds']==12
    assert curve['points'][0]['four_axis_mean']==4
    with pytest.raises(ValueError):mod.plot_dev_curve([{**candidates[0],'split':'test'}],logs,tmp_path/'bad')

def test_cli_freeze_sheets_report_select_and_private_key(tmp_path,monkeypatch):
    import json
    import sys
    from experiments.qwen3_bonus import common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    mod=api();frozen=bundle(80,'test')
    weights=tmp_path/'weights';weights.write_bytes(b'fixture-adapter')
    for row in frozen['captures']:
        if row['variant']=='perm_grpo':row['adapter_sha256']=common.sha256(weights)
    cases=tmp_path/'cases.jsonl';cases.write_text(''.join(json.dumps(r)+'\n' for r in frozen['cases']))
    caps=tmp_path/'captures.json';caps.write_text(json.dumps(frozen['captures']))
    def run(*args):
        monkeypatch.setattr(sys,'argv',['perm_evaluation',*map(str,args)])
        mod.main()
    def source(name,path):return ['--'+name,path,'--'+name+'-sha256',common.sha256(path)]
    frozen_path=tmp_path/'bundle.json'
    run('freeze',*source('data',cases),*source('captures',caps),'--split','test','--output',frozen_path)
    sheets=tmp_path/'sheets'
    run('sheets',*source('bundle',frozen_path),'--output',sheets)
    assert (sheets/'key.private.json').stat().st_mode & 0o777==0o600
    assert (sheets/'protocol.json').exists()
    candidate_path=tmp_path/'candidates.json';candidate_path.write_text(json.dumps([candidate(weights)]))
    selection=tmp_path/'selection.json';run('select',*source('candidates',candidate_path),'--output',selection)
    sheet=json.loads((sheets/'sheet.json').read_text())
    review=tmp_path/'reviews.json';review.write_text(json.dumps(reviews(sheet)))
    result=tmp_path/'result.json'
    run('report',*source('bundle',frozen_path),*source('sheet',sheets/'sheet.json'),
      *source('key',sheets/'key.private.json'),*source('reviews',review),*source('selection',selection),'--output',result)
    document=json.loads(result.read_text())
    assert document['selected_checkpoint_sha256']==common.sha256(weights)
    assert document['contrasts']['current']['n']==80
    with pytest.raises(FileExistsError):run('select',*source('candidates',candidate_path),'--output',selection)


@pytest.mark.parametrize('replacement',['test_ids','custom_content'])
def test_dev_candidate_rejects_other_32_cases_despite_valid_reviews(tmp_path,replacement):
    mod=api();weights=tmp_path/'weights';weights.write_bytes(b'fixture')
    valid=candidate(weights);evidence=valid['evidence']
    changed=copy.deepcopy(evidence['bundle'])
    if replacement=='test_ids':
        mapping={r['id']:'test-'+r['id'] for r in changed['cases']}
        for row in changed['cases']:row['id']=mapping[row['id']]
        for row in changed['captures']:row['id']=mapping[row['id']]
    else:
        for row in changed['cases']:row['prompt'][0]['content']='custom prompt'
        for row in changed['captures']:row['prompt'][0]['content']='custom prompt'
    changed=mod.freeze_captures(changed['cases'],changed['captures'],split='dev')
    sheet,key=mod.make_pair_sheets(changed);review=reviews(sheet)
    mod.validate_reviews(sheet,review)
    with pytest.raises(ValueError,match='canonical frozen dev case content'):
        mod.build_dev_candidate(changed,sheet,key,review,weights,1,dev_manifest_path=evidence['dev_manifest_path'])


def test_dev_candidate_revalidates_manifest_and_split_after_selection(tmp_path):
    mod=api();weights=tmp_path/'weights';weights.write_bytes(b'fixture')
    valid=candidate(weights)
    from pathlib import Path
    manifest=Path(valid['dev_manifest_path'])
    original=manifest.read_text();manifest.write_text(original+' ')
    with pytest.raises(ValueError):mod.select_checkpoint([valid])
    manifest.write_text(original)
    dev=manifest.parent/'response_dev.jsonl';dev.write_text(dev.read_text()+'\n')
    with pytest.raises(ValueError,match='digest required'):mod.select_checkpoint([valid])


def test_dev_candidate_allows_only_order_normalization(tmp_path):
    mod=api();weights=tmp_path/'weights';weights.write_bytes(b'fixture')
    valid=candidate(weights);evidence=valid['evidence'];frozen=evidence['bundle']
    reordered=mod.freeze_captures(list(reversed(frozen['cases'])),frozen['captures'],split='dev')
    sheet,key=mod.make_pair_sheets(reordered)
    result=mod.build_dev_candidate(reordered,sheet,key,reviews(sheet),weights,1,dev_manifest_path=evidence['dev_manifest_path'])
    assert result['semantic_complete']==32
