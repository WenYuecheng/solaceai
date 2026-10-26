"""Independent reward components; declarative constraints never execute data."""
from __future__ import annotations
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from experiments.qwen3_bonus.rewards import invalid_reason,repetition_ratio,_model


def normalize_constraints(value):
    """Arrow nested list columns may arrive as NumPy arrays and scalars."""
    def plain(x):
        if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [plain(v) for v in x]
        if hasattr(x,'tolist'):return plain(x.tolist())
        return x
    return plain(value)


def constraint_score(text: str, constraints: list[dict]) -> tuple[float,bool]:
    if not isinstance(constraints,list):raise ValueError('Constraints must be a list')
    results=[]
    for rule in constraints:
        if not isinstance(rule,dict):raise ValueError('Invalid constraint')
        kind=rule.get('kind')
        if kind=='numbered_items':
            count=rule.get('count')
            if type(count) is not int or not 1<=count<=10:raise ValueError('Invalid item count')
            items=re.findall(r'^\s*(\d+)[.、)）]\s*(\S.{3,})$',text,re.M)
            results.append(len(items)==count and {int(i) for i,_ in items}==set(range(1,count+1)))
        elif kind=='max_chars':
            value=rule.get('value')
            if type(value) is not int or not 1<=value<=10000:raise ValueError('Invalid character limit')
            results.append(len(text.strip())<=value)
        elif kind in ('required_text','forbidden_text'):
            value=rule.get('text')
            if not isinstance(value,str) or not value or len(value)>512:raise ValueError('Invalid required/forbidden text')
            results.append((value in text) if kind=='required_text' else (value not in text))
        elif kind=='allowed_citations':
            ids=rule.get('ids')
            if not isinstance(ids,list) or not ids or any(not isinstance(v,str) or not v for v in ids):
                raise ValueError('Citation IDs required')
            found=re.findall(r'\[([^\]\n]{1,64})\]',text)
            results.append(bool(found) and set(found).issubset(ids))
        else:raise ValueError(f'Unsupported declarative constraint: {kind}')
    return (sum(results)/len(results),True) if results else (0.,False)


def component_scores(raw_rm: float,text: str,constraints: list[dict],calibration: dict) -> dict:
    median=float(calibration['median']);iqr=float(calibration['iqr'])
    if not all(math.isfinite(v) for v in (raw_rm,median,iqr)) or iqr<0:
        raise ValueError('Non-finite or invalid reward calibration')
    preference=1/(1+math.exp(-max(-700.,min(700.,(raw_rm-median)/max(iqr,.1)))))
    constraint,applicable=constraint_score(text,constraints)
    non_repetition=1-repetition_ratio(text)
    reason=invalid_reason(text) or ''
    return {'score':.6*preference+.3*constraint+.1*non_repetition,
            'preference':preference,'constraint':constraint,'non_repetition':non_repetition,
            'constraint_applicable':int(applicable),'invalid_reason':reason,'raw_rm':float(raw_rm)}


@lru_cache(maxsize=1)
def load_calibration(path: str) -> dict:
    value=json.loads(Path(path).read_text())
    if value.get('gate',{}).get('passed') is not True:raise ValueError('GDPO calibration gate has not passed')
    return value


def compute_score(data_source,solution_str,ground_truth,extra_info=None,reward_model_path=None,calibration_path=None):
    if not reward_model_path or not calibration_path:raise ValueError('Explicit local RM and calibration required')
    info=extra_info or {};messages=info.get('messages')
    if messages is not None and not isinstance(messages,list):messages=list(messages)
    if not messages:raise ValueError('Original messages required')
    constraints=normalize_constraints(info.get('constraints',[]))
    raw=_model(reward_model_path)(messages,solution_str)
    return component_scores(raw,solution_str,constraints,load_calibration(str(calibration_path)))
