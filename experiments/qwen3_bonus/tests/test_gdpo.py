import math
import numpy as np
import pytest


def calculate(scores, groups=None, mask=None, invalid=None):
    from experiments.qwen3_bonus.gdpo import compute_gdpo
    n=len(scores['preference'])
    return compute_gdpo(scores, groups or [str(i//2) for i in range(n)],
                        np.ones((n,2),np.float32) if mask is None else mask,
                        invalid or ['']*n)


def test_weighted_components_use_sample_std_and_token_masked_whitening():
    scores={'preference':[.9,.1,.9,.1], 'constraint':[1,0,0,1],
            'non_repetition':[1,1,1,1]}
    mask=np.array([[1,1,0],[1,0,0],[1,1,1],[1,1,0]],np.float32)
    result=calculate(scores,mask=mask)
    # Independent scalar reference: each group has sample sd sqrt(.5)*difference.
    a=.8/2/(.8/math.sqrt(2)+1e-6)
    b=.5/(1/math.sqrt(2)+1e-6)
    seq=[.6*a+.3*b,-.6*a-.3*b,.6*a-.3*b,-.6*a+.3*b]
    tokens=[seq[i] for i in range(4) for j in range(3) if mask[i,j]]
    mean=sum(tokens)/len(tokens)
    var=sum((x-mean)**2 for x in tokens)/(len(tokens)-1)
    expected=np.array([[(seq[i]-mean)/math.sqrt(var+1e-8) if mask[i,j] else 0
                        for j in range(3)] for i in range(4)],np.float32)
    np.testing.assert_allclose(result.advantages,expected,rtol=2e-6,atol=2e-6)
    assert result.metrics['reward/conflict_ratio']==.5
    assert result.metrics['reward/non_repetition_active_group_ratio']==0
    assert result.advantages.dtype==np.float32


def test_grouping_survives_reordering_and_padding_is_always_zero():
    x={'preference':[.2,.8,.1,.9], 'constraint':[0,1,1,0], 'non_repetition':[1]*4}
    reference=calculate(x)
    perm=[2,0,3,1]
    result=calculate({k:[v[i] for i in perm] for k,v in x.items()},
                     groups=['b','a','b','a'])
    np.testing.assert_allclose(result.advantages,reference.advantages[perm])


@pytest.mark.parametrize('delta,active',[(.019,False),(.02,False),(.021,True)])
def test_repetition_dead_zone_is_applied_before_normalizing(delta,active):
    result=calculate({'preference':[.5,.5], 'constraint':[0,0],
                      'non_repetition':[0.,delta]})
    assert bool(result.metrics['reward/non_repetition_active_group_ratio']) is active
    assert bool(np.any(result.advantages)) is active


def test_validity_is_one_explicit_gate_not_three_modified_rewards():
    scores={'preference':[.9,.1,.8,.7], 'constraint':[1,0,1,0], 'non_repetition':[1]*4}
    result=calculate(scores,invalid=['bad','','bad','bad'])
    np.testing.assert_array_equal(result.advantages,[[-1,-1],[1,1],[0,0],[0,0]])
    assert result.group_records[0]['raw_rewards']['preference']==[.9,.1]
    assert result.metrics['validity/mixed_group_ratio']==.5
    assert result.metrics['validity/all_invalid_group_ratio']==.5


def test_constant_batch_and_empty_mask_return_finite_zero():
    for mask in [np.ones((2,2)),np.zeros((2,2)),np.array([[1,0],[0,0]])]:
        result=calculate({k:[.5,.5] for k in ['preference','constraint','non_repetition']},mask=mask)
        assert np.isfinite(result.advantages).all()
        assert not result.advantages.any()


@pytest.mark.parametrize('defect',['missing','nan','singleton','mask'])
def test_invalid_inputs_fail_closed(defect):
    scores={k:[.2,.8] for k in ['preference','constraint','non_repetition']}
    groups=['a','a'];mask=np.ones((2,2))
    if defect=='missing':scores.pop('constraint')
    if defect=='nan':scores['preference'][0]=float('nan')
    if defect=='singleton':groups=['a','b']
    if defect=='mask':mask[0,0]=.3
    with pytest.raises(ValueError):calculate(scores,groups,mask)


def test_constraints_check_only_declarative_allowed_kinds():
    from experiments.qwen3_bonus.gdpo_rewards import constraint_score, component_scores
    text='1. 整理今天的待办。\n2. 给老师发一条消息。\n依据[A]。'
    constraints=[{'kind':'numbered_items','count':2},{'kind':'allowed_citations','ids':['A']},
                 {'kind':'max_chars','value':80},{'kind':'required_text','text':'老师'}]
    assert constraint_score(text,constraints)==(1.,True)
    assert constraint_score(text,[])==(0.,False)
    with pytest.raises(ValueError):constraint_score(text,[{'kind':'python','code':'print(1)'}])
    r=component_scores(8.,text,constraints,{'median':8.,'iqr':2.})
    assert r['preference']==.5 and r['constraint']==1
    assert r['invalid_reason']==''
    empty=component_scores(8.,'',[],{'median':8.,'iqr':2.})
    assert empty['invalid_reason']=='empty'
    assert empty['preference']==.5  # Independent raw definitions are not overwritten.


def test_arrow_numpy_constraint_lists_preserve_meaning():
    from experiments.qwen3_bonus.gdpo_rewards import normalize_constraints,constraint_score
    rules=np.array([{'kind':'allowed_citations','ids':np.array(['A']), 'count':None},
                    {'kind':'numbered_items','count':np.int64(1),'ids':None}],dtype=object)
    normalized=normalize_constraints(rules)
    assert constraint_score('1. 可以先整理一项任务[A]。',normalized)==(1.,True)
