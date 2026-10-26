"""Standard scalar rewards for GRPO; not a clinical evaluator or PERM reproduction."""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path


def repetition_ratio(text: str) -> float:
    compact = re.sub(r'\s+', '', text)
    grams = [compact[i:i+4] for i in range(max(0,len(compact)-3))]
    return 1-len(set(grams))/len(grams) if grams else 0.0


def invalid_reason(text: str) -> str | None:
    if not isinstance(text,str) or not text.strip():
        return 'empty'
    if any(token in text for token in ('<tool_call>','<think>','</think>')):
        return 'invalid_output'
    # Narrow positive assertions only. This is deliberately NOT a safety classifier.
    if re.search(r'你(?:肯定|一定|已经|就是)(?:患有|得了|是)?(?:抑郁症|焦虑症|精神分裂症|双相障碍)',text):
        return 'diagnosis'
    if re.search(r'(?:建议你|你应该|你可以)(?:服用|口服|注射)(?:氟西汀|舍曲林|帕罗西汀|安眠药)',text):
        return 'prescription'
    if re.search(r'(?:自杀|自残)(?:的方法|步骤)(?:是|如下)',text):
        return 'harm_instruction'
    return None


def score_response(raw_score: float, text: str) -> dict:
    if not math.isfinite(raw_score):
        raise ValueError('Non-finite reward-model output')
    reason = invalid_reason(text)
    if reason:
        return {'score':-1.0,'invalid':1.0,'repeat_ratio':repetition_ratio(text),'raw_rm':raw_score}
    preference = 1/(1+math.exp(-max(-700,min(700,raw_score))))
    repeat = repetition_ratio(text)
    return {'score':preference-.2*repeat,'invalid':0.0,'repeat_ratio':repeat,'raw_rm':raw_score}


def calibration_gate(rows: list[dict], *, allow_model_review: bool = False) -> dict:
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate calibration IDs')
    reviewed = [r for r in rows if r.get('reviewed') is True and str(r.get('reviewer','')).strip() and r.get('expected') in ('a','b')
        and (r.get('review_type','human')=='human' or (allow_model_review and r.get('review_type')=='model'))]
    valid = [r for r in reviewed if all(isinstance(r.get(k),(int,float)) and math.isfinite(r[k]) for k in ('score_a','score_b'))]
    correct = sum((r['score_a']>r['score_b'] if r['expected']=='a' else r['score_b']>r['score_a']) for r in valid)
    # Exactly twenty reviewed pairs: extra model-scored rows cannot improve the gate.
    return {'count':len(rows),'reviewed':len(valid),'correct':correct,
            'human_reviewed':sum(r.get('review_type','human')=='human' for r in valid),
            'model_reviewed':sum(r.get('review_type')=='model' for r in valid),
            'passed':len(rows)==20 and len(valid)==20 and correct>=15,
            'interpretation':('User-authorized model review; not human preference validation.' if allow_model_review
                else 'Human preference annotations required; general preference calibration only.')}


class FrozenRewardModel:
    """Local, revision-pinned Skywork sequence classifier, loaded only on explicit use."""
    def __init__(self, model_path: str, device: str = 'cpu'):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path=Path(model_path).resolve()
        if not path.is_dir():
            raise ValueError('Reward model must be a previously downloaded local directory')
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForSequenceClassification.from_pretrained(
            path,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float32 if device=='cpu' else torch.bfloat16)
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.device=device

    def __call__(self,messages: list[dict], response: str) -> float:
        import torch
        # RM scores the actual (history,response); no invented rubric injected into user text.
        text=self.tokenizer.apply_chat_template(messages+[{'role':'assistant','content':response}],tokenize=False)
        batch=self.tokenizer(text,return_tensors='pt',truncation=True,max_length=1280).to(self.device)
        with torch.inference_mode():
            value=float(self.model(**batch).logits.float().reshape(-1)[0].item())
        if not math.isfinite(value):
            raise ValueError('Non-finite RM score')
        return value


@lru_cache(maxsize=1)
def _model(path: str) -> FrozenRewardModel:
    return FrozenRewardModel(path,'cpu')


def compute_score(data_source, solution_str, ground_truth, extra_info=None, reward_model_path=None):
    """verl v0.5.0 custom reward hook. Reference answer is never used for scoring."""
    if not reward_model_path:
        raise ValueError('Explicit local reward_model_path required')
    info=extra_info or {}
    messages=info.get('messages')
    if messages is not None and not isinstance(messages,list):
        messages=list(messages)
    if not messages:
        raise ValueError('Original prompt messages missing from parquet extra_info')
    if invalid_reason(solution_str):
        return score_response(0.0,solution_str)
    return score_response(_model(reward_model_path)(messages,solution_str),solution_str)
