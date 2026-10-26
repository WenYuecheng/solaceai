"""Data integrity and statistical tests; all answers are tiny synthetic fixtures."""
import copy
import importlib
import hashlib
import json

import pytest


def api():
    return importlib.import_module('experiments.qwen3_bonus.gdpo_evaluation')


def fixture(n=10, split='dev'):
    from experiments.qwen3_bonus.models import SOURCES
    rows=[{'id':f'case-{i}','dialogue_id':f'dialogue-{i//2}', 'category':'listening',
           'prompt':[{'role':'user','content':f'prompt {i}'}], 'constraints':[],
           'requirements':['listen'],'provenance':{'source':'synthetic unit fixture'}} for i in range(n)]
    captures=[{'id':r['id'],'variant':variant,'prompt':r['prompt'],'response':f'{variant} answer {r["id"]}',
               'base_revision':SOURCES['base']['revision'],'converted_base_sha256':'b'*64,
               'temperature':0,'max_tokens':256,'adapter_sha256':None if variant=='base' else hashlib.sha256(variant.encode()).hexdigest()}
              for r in rows for variant in ('base','grpo32','current','gdpo')]
    return rows,captures


def reviews_for(sheet):
    return {'sheet_sha256':api().digest_json(sheet),'review_type':'model','reviewer':'fixture-model-identity',
            'provenance':{'runtime':'test only'},'rows':[{'id':r['id'],'choice':'tie','rationale':'Equal in fixture',
               'a':{'semantic_completed':True,'quality':3,'severe':[]},
               'b':{'semantic_completed':True,'quality':3,'severe':[]}} for r in sheet['rows']]}


def test_freeze_rejects_missing_duplicate_prompt_and_generation_mismatch():
    mod=api();rows,captures=fixture()
    bundle=mod.freeze_captures(rows,captures,split='dev')
    assert bundle['dataset_sha256']==mod.digest_json(rows)
    for change in ('missing','duplicate','prompt','temp','base'):
        bad=copy.deepcopy(captures)
        if change=='missing':bad.pop()
        if change=='duplicate':bad.append(bad[0])
        if change=='prompt':bad[0]['prompt']=[]
        if change=='temp':bad[0]['temperature']=.1
        if change=='base':bad[0]['converted_base_sha256']='c'*64
        with pytest.raises(ValueError):mod.freeze_captures(rows,bad,split='dev')


def test_anonymous_sheets_randomize_and_swap_exactly_twenty_percent():
    mod=api();rows,captures=fixture()
    bundle=mod.freeze_captures(rows,captures,split='dev')
    sheet,key=mod.make_pair_sheets(bundle)
    assert len(sheet['rows'])==36
    assert {r['comparison'] for r in key['pairs'].values()}=={'current','base','grpo32'}
    assert all('variant' not in r and 'comparison' not in r and 'case_id' not in r for r in sheet['rows'])
    assert sum(r['recheck_of'] is not None for r in key['pairs'].values())==6
    by_id={r['id']:r for r in sheet['rows']}
    for identity,entry in key['pairs'].items():
        if entry['recheck_of']:
            old=by_id[entry['recheck_of']];new=by_id[identity]
            assert new['a']==old['b'] and new['b']==old['a']
    assert mod.make_pair_sheets(bundle)==(sheet,key)


def test_review_provenance_content_and_numeric_scores_are_mandatory():
    mod=api();rows,captures=fixture();sheet,_=mod.make_pair_sheets(mod.freeze_captures(rows,captures,split='dev'))
    reviewed=reviews_for(sheet)
    assert mod.validate_reviews(sheet,reviewed)['review_type']=='model'
    for change in ('identity','hash','quality','semantic','severe','human'):
        bad=copy.deepcopy(reviewed)
        if change=='identity':bad['reviewer']=''
        if change=='hash':bad['sheet_sha256']='f'*64
        if change=='quality':bad['rows'][0]['a']['quality']=True
        if change=='semantic':bad['rows'][0]['a']['semantic_completed']='yes'
        if change=='severe':bad['rows'][0]['a']['severe']=[{'category':'fabricated_evidence','reason':''}]
        if change=='human':bad['review_type']='human'
        with pytest.raises(ValueError):mod.validate_reviews(sheet,bad)


def test_dialogue_bootstrap_clusters_turns_and_does_not_count_repetitions():
    mod=api()
    result=mod.paired_dialogue_bootstrap([{'dialogue_id':'a','score':1.0},
       {'dialogue_id':'a','score':1.0},{'dialogue_id':'b','score':0.0}],draws=10000)
    assert result['n']==3 and result['dialogues']==2
    assert result['score']==pytest.approx(2/3)
    assert result['ci95']==[0.0,1.0]
    assert result==mod.paired_dialogue_bootstrap([{'dialogue_id':'a','score':1.0},
       {'dialogue_id':'a','score':1.0},{'dialogue_id':'b','score':0.0}],draws=10000)


def test_checkpoint_selection_is_dev_only_lexicographic_and_freezes_weights(tmp_path,monkeypatch):
    from experiments.qwen3_bonus import common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    mod=api();weights=tmp_path/'weights';weights.write_bytes(b'fixture')
    candidates=[{'split':'dev','step':s,'severe_count':bad,'semantic_complete':good,'quality':q,
                 'checkpoint':str(weights),'checkpoint_sha256':common.sha256(weights)}
                for s,bad,good,q in [(10,1,10,5),(20,0,8,5),(30,0,9,3),(40,0,9,3)]]
    selected=mod.select_checkpoint(candidates)
    assert selected['step']==30
    with pytest.raises(ValueError):mod.select_checkpoint([{**candidates[0],'split':'test'}])
    path=tmp_path/'selection.json';mod.freeze_selection(candidates,path)
    with pytest.raises(FileExistsError):mod.freeze_selection(candidates,path)
    assert mod.verify_selection(path)['selected']['step']==30
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError):mod.verify_selection(path)


def test_final_report_requires_80_unique_cases_and_excludes_rechecks():
    mod=api();rows,captures=fixture(80)
    bundle=mod.freeze_captures(rows,captures,split='test')
    sheet,key=mod.make_pair_sheets(bundle)
    report=mod.evaluate_reviews(bundle,sheet,key,reviews_for(sheet))
    assert report['contrasts']['current']['n']==80
    assert report['contrasts']['current']['score']==.5
    assert report['rechecks']['count']==48
    assert report['review_type']=='model' and report['human_reviewed'] is False
    small_rows,small_caps=fixture()
    with pytest.raises(ValueError):mod.freeze_captures(small_rows,small_caps,split='test')


def test_frozen_bundle_and_key_mutation_are_rejected():
    mod=api();rows,captures=fixture();bundle=mod.freeze_captures(rows,captures,split='dev')
    sheet,key=mod.make_pair_sheets(bundle)
    key['pairs'][next(iter(key['pairs']))]['a']='base'
    with pytest.raises(ValueError):mod.evaluate_reviews(bundle,sheet,key,reviews_for(sheet))
    bundle['captures'][0]['response']='tampered'
    with pytest.raises(ValueError):mod.make_pair_sheets(bundle)


def test_capture_adapters_are_hashed_and_constant_within_each_variant():
    mod=api();rows,captures=fixture()
    for change in ('missing','drift','bad-digest'):
        bad=copy.deepcopy(captures)
        if change=='missing':bad[1].pop('adapter_sha256')
        if change=='drift':bad[1]['adapter_sha256']='f'*64
        if change=='bad-digest':bad[1]['adapter_sha256']='not-a-hash'
        with pytest.raises(ValueError):mod.freeze_captures(rows,bad,split='dev')


def test_cli_input_hash_and_root_output_confinement(tmp_path,monkeypatch):
    from experiments.qwen3_bonus import common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    mod=api();path=tmp_path/'source.json';path.write_text('{}')
    assert mod.read_verified(path,common.sha256(path))=={}
    with pytest.raises(ValueError):mod.read_verified(path,'f'*64)
    with pytest.raises(ValueError):mod.write_immutable(tmp_path.parent/'escape.json',{})
    target=tmp_path/'key.json';mod.write_immutable(target,{'private':True},private=True)
    assert target.stat().st_mode & 0o777==0o600
    with pytest.raises(FileExistsError):mod.write_immutable(target,{})


def test_plot_logs_labels_missing_metrics_without_synthesizing(tmp_path,monkeypatch):
    pytest.importorskip('matplotlib')
    from experiments.qwen3_bonus import common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    manifest=api().plot_training_logs([],tmp_path/'figures')
    assert 'quality' in manifest['missing'] and 'actor/kl_loss' in manifest['missing']
    assert set(manifest['figures'])=={'quality.png','components.png'}
    assert all((tmp_path/'figures'/name).is_file() for name in manifest['figures'])


def test_plot_uses_actual_nested_three_component_training_logs(tmp_path,monkeypatch):
    pytest.importorskip('matplotlib')
    from experiments.qwen3_bonus import common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    logs=[{'step':1,'elapsed_seconds':12.5,'metrics':{'advantage/std':1.,'actor/kl_loss':.02}}]
    rollouts=[{'step':1,'records':[{'rewards':{'preference':.4,'constraint':1.,'non_repetition':.9}}]}]
    result=api().plot_training_logs(logs,tmp_path/'actual',rollouts=rollouts)
    assert 'preference' not in result['missing'] and 'constraint' not in result['missing']
    assert 'non_repetition' not in result['missing'] and 'actor/kl_loss' not in result['missing']
    assert result['rollouts_sha256']==api().digest_json(rollouts)


def test_review_rejects_null_identity_and_null_severe_reason():
    mod=api();rows,captures=fixture();sheet,_=mod.make_pair_sheets(mod.freeze_captures(rows,captures,split='dev'))
    reviewed=reviews_for(sheet)
    reviewed['reviewer']=None
    with pytest.raises(ValueError):mod.validate_reviews(sheet,reviewed)
    reviewed=reviews_for(sheet)
    reviewed['rows'][0]['a']['severe']=[{'category':'other','reason':None}]
    with pytest.raises(ValueError):mod.validate_reviews(sheet,reviewed)
