"""PERM scalar reward; declarative checks do not measure semantic completion."""
from __future__ import annotations
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from experiments.qwen3_bonus.gdpo_rewards import constraint_score, normalize_constraints
from experiments.qwen3_bonus.rewards import FrozenRewardModel, invalid_reason, repetition_ratio

REWARD_SPEC='perm-harmonic-v1'
CODE_DEPENDENCIES=('perm_calibration.py','perm_rewards.py','perm_models.py','perm_data.py','rewards.py','gdpo_rewards.py','gdpo_data.py','data.py','common.py','models.py')
INPUT_FORMAT='rm-own-chat-template-system-filtered-no-truncation-v1'


def reward_messages(messages,response):
    messages=normalize_constraints(messages)
    if not isinstance(messages,list) or not messages or not isinstance(response,str):
        raise ValueError('Original messages and string response required')
    result=[]
    for m in messages:
        if not isinstance(m,dict) or m.get('role') not in ('system','user','assistant') or not isinstance(m.get('content'),str):
            raise ValueError('Invalid original history')
        if m['role']!='system':result.append({'role':m['role'],'content':m['content']})
    if not result or result[-1]['role']!='user' or any(m['role']!=('user' if i%2==0 else 'assistant') for i,m in enumerate(result)):
        raise ValueError('History must alternate user/assistant and end user')
    return result+[{'role':'assistant','content':response}]


def component_scores(raw_rm,text,constraints,calibration):
    median=float(calibration['median']);iqr=float(calibration['iqr'])
    if not all(math.isfinite(x) for x in (median,iqr)) or iqr<0:raise ValueError('Invalid calibration')
    constraints=normalize_constraints(constraints)
    constraint_score('',constraints) # validate all rules, including length-only rules
    checks=[x for x in constraints if x['kind']!='max_chars']
    lengths=[len(text.strip())<=x['value'] for x in constraints if x['kind']=='max_chars']
    c,applicable=constraint_score(text,checks)
    reason=invalid_reason(text) or '';repeat=repetition_ratio(text)
    if raw_rm is not None and not math.isfinite(float(raw_rm)):raise ValueError('Non-finite raw reward')
    if not reason and raw_rm is None:raise ValueError('Valid sample requires RM score')
    pref=0. if raw_rm is None else 1/(1+math.exp(-max(-700,min(700,(float(raw_rm)-median)/max(iqr,.1)))))
    main=0. if raw_rm is None else (2*pref*c/(pref+c+1e-6) if applicable else pref)
    score=-1. if reason else main-.1*repeat
    return dict(score=score,raw_rm=raw_rm,raw_rm_applicable=raw_rm is not None,main=main,preference=pref,constraint=c,
                constraint_applicable=bool(applicable),invalid=bool(reason),invalid_reason=reason,
                repeat_ratio=repeat,length_checks=lengths,score_before_gate=score,group_gate=False)


def apply_group_gate(score_records,uids):
    if len(score_records)!=len(uids):raise ValueError('UID count mismatch')
    groups=defaultdict(list);records=[dict(r) for r in score_records]
    for i,(r,uid) in enumerate(zip(records,uids)):
        if not math.isfinite(float(r['score'])):raise ValueError('Non-finite reward')
        groups[uid].append(i)
    if any(len(v)!=2 for v in groups.values()):raise ValueError('Exactly two responses per UID required')
    gated=0
    for indices in groups.values():
        gate=all(not records[i]['invalid'] and records[i]['constraint_applicable'] and records[i]['constraint']==0 for i in indices)
        gated+=int(gate)
        for i in indices:
            records[i]['score_before_gate']=records[i]['score'];records[i]['group_gate']=gate
            if gate:records[i]['score']=0.
    return records,{'groups':len(groups),'gated_groups':gated,'gated_responses':2*gated}


class FrozenPermRewardModel(FrozenRewardModel):
    def __init__(self,model_path,device='cpu'):
        if device!='cpu':raise ValueError('PERM reward model requires CPU FP32')
        from experiments.qwen3_bonus.perm_models import verify_model
        self.identity=verify_model(Path(model_path))
        super().__init__(model_path,'cpu')

    def __call__(self,messages,response):
        import torch
        history=reward_messages(messages,response)
        text=self.tokenizer.apply_chat_template(history,tokenize=False,add_generation_prompt=False)
        batch=self.tokenizer(text,return_tensors='pt',add_special_tokens=False,truncation=False).to('cpu')
        limit=getattr(self.model.config,'max_position_embeddings',None)
        if limit and batch['input_ids'].shape[-1]>limit:raise ValueError('RM context exceeds model limit; refusing truncation')
        with torch.inference_mode():value=float(self.model(**batch).logits.float().reshape(-1)[0].item())
        if not math.isfinite(value):raise ValueError('Non-finite RM output')
        return value


@lru_cache(maxsize=1)
def _model(path):return FrozenPermRewardModel(path)


@lru_cache(maxsize=1)
def _identity(path):
    from experiments.qwen3_bonus.perm_models import verify_model
    return verify_model(Path(path))


@lru_cache(maxsize=8)
def load_calibration(path, require_passed=True):
    value=json.loads(Path(path).read_text())
    from experiments.qwen3_bonus.common import sha256
    manifest=Path(value.get('data_manifest_path',''))
    if not manifest.is_file() or sha256(manifest)!=value.get('data_manifest_sha256'):raise ValueError('Calibration data manifest drift')
    if set(value.get('code_sha256',{}))!=set(CODE_DEPENDENCIES):raise ValueError('Incomplete calibration dependency fingerprint')
    for name,digest in value.get('code_sha256',{}).items():
        source=Path(__file__).parent/name
        if source.parent!=Path(__file__).parent or not source.is_file() or sha256(source)!=digest:raise ValueError('Calibration code drift')
    if not value.get('code_sha256') or not value.get('labels_sha256'):raise ValueError('Calibration provenance missing')
    if require_passed and value.get('gate',{}).get('passed') is not True:raise ValueError('PERM calibration gate has not passed')
    if value.get('reward_spec')!=REWARD_SPEC or value.get('input_format')!=INPUT_FORMAT:raise ValueError('PERM calibration specification mismatch')
    if not all(math.isfinite(float(value[k])) for k in ('median','iqr')) or value['iqr']<0:raise ValueError('Invalid calibration statistics')
    gate=value['gate']
    if gate.get('count')!=20 or gate.get('task_delivery_count')!=8 or (require_passed and (gate.get('correct',0)<16 or gate.get('task_delivery_correct',0)<7)):raise ValueError('Insufficient calibration gate evidence')
    from experiments.qwen3_bonus.perm_calibration import validate_report
    validate_report(value)
    return value


def compute_score(data_source,solution_str,ground_truth,extra_info=None,reward_model_path=None,calibration_path=None):
    if not reward_model_path or not calibration_path:raise ValueError('Explicit RM and calibration paths required')
    info=extra_info or {};messages=info.get('messages')
    reward_messages(messages,solution_str)
    calibration=load_calibration(str(calibration_path))
    identity=_identity(str(reward_model_path))
    if identity['model_inventory_sha256']!=calibration.get('model',{}).get('model_inventory_sha256'):raise ValueError('Calibration model mismatch')
    raw=None if invalid_reason(solution_str) else _model(str(reward_model_path))(messages,solution_str)
    return component_scores(raw,solution_str,info.get('constraints',[]),calibration)
