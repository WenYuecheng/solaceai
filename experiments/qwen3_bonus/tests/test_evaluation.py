import pytest


def test_macro_f1_counts_invalid_outputs_and_high_risk_misses():
    from experiments.qwen3_bonus.evaluation import classification_metrics
    result=classification_metrics([
        {'expected':'正常','predicted':'正常'},
        {'expected':'焦虑','predicted':'焦虑'},
        {'expected':'低落','predicted':'低落'},
        {'expected':'高风险','predicted':'正常'},
    ])
    assert result['accuracy']==.75
    assert result['high_risk_misses']==1
    assert result['macro_f1']==pytest.approx(2/3)


def test_release_requires_complete_frozen_ids_no_new_high_risk_misses():
    from experiments.qwen3_bonus.evaluation import release_gate,DIMENSIONS
    labels=['正常','焦虑','低落','高风险']
    baseline=[{'id':str(i),'expected':label,'predicted':label} for i,label in enumerate(labels)]
    pairs=[{'id':str(i),'choice':'adapter','reviewed':True,'reviewer':'human','severe_safety_regression':False,'dimensions':{d:{'a':3,'b':4} for d in DIMENSIONS}} for i in range(40)]
    assert release_gate(baseline,baseline,pairs,[str(i) for i in range(40)])['passed']
    pairs.pop()
    assert not release_gate(baseline,baseline,pairs,[str(i) for i in range(40)])['passed']
    pairs.append({'id':'39','choice':'adapter','reviewed':True,'reviewer':'human','severe_safety_regression':True})
    assert not release_gate(baseline,baseline,pairs,[str(i) for i in range(40)])['passed']


def test_release_rejects_mismatched_labels_even_if_score_improves():
    from experiments.qwen3_bonus.evaluation import release_gate
    base=[{'id':'1','expected':'高风险','predicted':'正常'}]
    candidate=[{'id':'1','expected':'正常','predicted':'正常'}]
    with pytest.raises(ValueError,match='labels'):
        release_gate(base,candidate,[],[])


def test_blind_pairs_do_not_leak_model_identity_and_reject_missing_cases():
    from experiments.qwen3_bonus.evaluation import blind_pairs
    base=[{'id':'x','answer':'基座回答','prompt':[{'role':'user','content':'难过'}]}]
    adapter=[{'id':'x','answer':'训练回答','prompt':base[0]['prompt']}]
    sheet,key=blind_pairs(base,adapter,3407)
    assert set(sheet[0])=={'id','prompt','a','b','choice','reviewed','reviewer','severe_safety_regression','safety_flags','dimensions'}
    assert 'base' in key['x'].values() and 'adapter' in key['x'].values()
    with pytest.raises(ValueError):blind_pairs(base,[],3407)


@pytest.mark.parametrize('url',['https://api.deepseek.com/v1','http://127.0.0.1.evil/v1','http://user:pass@127.0.0.1/v1'])
def test_evaluation_rejects_paid_or_ambiguous_endpoints(url):
    from experiments.qwen3_bonus.evaluation import local_url
    with pytest.raises(ValueError):local_url(url)


def test_capture_is_bound_to_actual_frozen_prompt_and_label():
    from experiments.qwen3_bonus.evaluation import verify_capture
    cases=[{'id':'x','prompt':[{'role':'user','content':'难过'}],'expected':'低落'}]
    record={'task':'classification','dataset_sha256':'abc','rows':[{
        **cases[0],'expected':'正常','predicted':'正常'}]}
    with pytest.raises(ValueError,match='label'):verify_capture(record,cases,'abc','classification')
    record['rows'][0]['expected']='低落'
    record['rows'][0]['prompt']=[{'role':'user','content':'changed'}]
    with pytest.raises(ValueError,match='prompt'):verify_capture(record,cases,'abc','classification')


def test_human_annotations_may_change_but_blind_answers_may_not():
    from experiments.qwen3_bonus.evaluation import review_content_digest
    row={'id':'x','prompt':[],'a':'A','b':'B','reviewed':False}
    before=review_content_digest([row])
    row.update(reviewed=True,choice='a',reviewer='human')
    assert review_content_digest([row])==before
    row['a']='different answer'
    assert review_content_digest([row])!=before


def test_release_cannot_stamp_old_captures_with_new_adapter_hashes():
    from experiments.qwen3_bonus.evaluation import verify_deployment_captures
    identity={'base_manifest':'base-hash','emotion':{'weights':'E'},'response':{'weights':'R'}}
    emotion={'model':'solace-emotion','deployment_artifacts':identity}
    response={'model':'solace-response','deployment_artifacts':identity}
    base={'base_manifest_sha256':'base-hash'}
    verify_deployment_captures(identity,emotion,response,[base,base])
    changed={**identity,'response':{'weights':'unreviewed-R2'}}
    with pytest.raises(ValueError,match='Candidate'):
        verify_deployment_captures(changed,emotion,response,[base,base])
    with pytest.raises(ValueError,match='Baseline'):
        verify_deployment_captures(identity,emotion,response,[{'base_manifest_sha256':'other-base'}])
