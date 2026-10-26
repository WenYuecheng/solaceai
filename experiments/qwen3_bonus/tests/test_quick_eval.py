import pytest

from experiments.qwen3_bonus.quick_eval import blind, summarize, validate_cases, validate_captures


def test_case_inventory_and_balance():
    from experiments.qwen3_bonus.quick_cases import cases
    classification, response = cases()
    validate_cases(classification, response)
    with pytest.raises(ValueError):
        validate_cases(classification[:-1], response)


def test_blind_hides_identity_and_preserves_answers():
    groups = {name: [{'id': 'x', 'prompt': [], 'answer': answer}]
              for name, answer in [('base', 'one'), ('adapter', 'two')]}
    sheet, key = blind(groups)
    assert 'base' not in str(sheet) and 'adapter' not in str(sheet)
    assert {v['answer'] for v in sheet[0]['answers'].values()} == {'one', 'two'}
    assert set(key['x'].values()) == {'base', 'adapter'}
    with pytest.raises(ValueError):
        blind({**groups, 'bad': []})


def test_summary_requires_every_answer_and_valid_scores():
    sheet, key = blind({'base': [{'id': 'x', 'prompt': [], 'answer': 'one'}],
                        'adapter': [{'id': 'x', 'prompt': [], 'answer': 'two'}]})
    with pytest.raises(ValueError):
        summarize(sheet, key, [])
    scores = [{'id': 'x', 'side': side, 'scores': [4, 4, 4, 4, 5], 'reason': 'clear',
               'flags': [], 'severe_safety': False} for side in ('A', 'B')]
    result = summarize(sheet, key, scores)
    assert result['pairwise']['adapter_vs_base'] == {'wins': 0, 'ties': 1, 'losses': 0}
    scores[0]['scores'][0] = 6
    with pytest.raises(ValueError):
        summarize(sheet, key, scores)


def test_severe_safety_failure_loses_even_with_higher_mean():
    sheet, key = blind({'base': [{'id': 'x', 'prompt': [], 'answer': 'one'}],
                        'adapter': [{'id': 'x', 'prompt': [], 'answer': 'two'}]})
    scores = [{'id': 'x', 'side': side, 'scores': [5]*5 if key['x'][side] == 'adapter' else [3]*5,
               'reason': 'evidence', 'flags': [], 'severe_safety': key['x'][side] == 'adapter'}
              for side in ('A', 'B')]
    assert summarize(sheet, key, scores)['pairwise']['adapter_vs_base']['losses'] == 1


def test_capture_validator_rejects_changed_prompt_label_and_identity():
    from copy import deepcopy
    case={'id':'x','prompt':[{'role':'user','content':'safe'}],'expected':'正常'}
    rows=[{**case,'task':'classification','variant':variant,'requested_model':model,'reported_model':model,
           'answer':'正常','predicted':'正常','finish_reason':'stop'}
          for variant,model in [('base','base-path'),('adapter','solace-emotion')]]
    validate_captures(rows,{'classification':[case]},'base-path')
    for field,value in [('prompt',[]),('expected','焦虑'),('reported_model','other')]:
        changed=deepcopy(rows);changed[0][field]=value
        with pytest.raises(ValueError):
            validate_captures(changed,{'classification':[case]},'base-path')
    with pytest.raises(ValueError):
        validate_captures(rows+rows,{'classification':[case]},'base-path')
