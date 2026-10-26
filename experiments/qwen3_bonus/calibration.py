"""Prepare unreviewed Chinese sanity pairs and score a local frozen reward model."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from experiments.qwen3_bonus.common import ROOT,output_path,read_jsonl,write_json,sha256
from experiments.qwen3_bonus.models import SOURCES,verify_model
from experiments.qwen3_bonus.rewards import FrozenRewardModel,calibration_gate

WEAK_CONTROLS=(
    '别想太多，开心一点就好了。',
    '每个人都这样，坚持努力就一定会成功。',
    '我理解你的感受。我理解你的感受。我理解你的感受。',
    '这些事情没有什么好难过的，你应该马上振作。',
    '先别说这些了，制定一个详细计划，每天严格执行。',
)


def prepare(source: Path,output: Path) -> None:
    output=output_path(output)
    if output.exists():raise FileExistsError('Calibration sheet is frozen; choose a new directory')
    rows=read_jsonl(source)
    if len(rows)<20:raise ValueError('Need at least 20 independent training contexts')
    rng=random.Random(3407);selected=rng.sample(rows,20);pairs=[]
    for i,row in enumerate(selected):
        choices=[row['reference'],WEAK_CONTROLS[i%len(WEAK_CONTROLS)]];rng.shuffle(choices)
        pairs.append({'id':row['id'],'dialogue_id':row['dialogue_id'],'prompt':row['prompt'],
            'a':choices[0],'b':choices[1],'expected':None,'reviewed':False,'reviewer':'',
            'provenance':'PsyDTCorpus reference vs manually authored weak control; neither is human gold.'})
    write_json(output/'pairs.json',{'source_sha256':sha256(source),'pairs':pairs,
        'instructions':'Read the context. Set expected to a/b only after human review, reviewed=true and reviewer. '
            'If neither reply is acceptable, replace the pair and review again. No preference is pre-filled.'})


def score(sheet: Path,model_path: Path,output: Path, *, allow_model_review=False) -> dict:
    if output.exists():raise FileExistsError('Use a new scored output')
    verify_model(model_path,'reward')
    document=json.loads(sheet.read_text())
    pairs=document['pairs']
    if len(pairs)!=20 or len({r['id'] for r in pairs})!=20:raise ValueError('Exactly 20 unique pairs required')
    model=FrozenRewardModel(str(model_path),'cpu')
    scored=[]
    for index,row in enumerate(pairs):
        scored.append({**row,'score_a':model(row['prompt'],row['a']),'score_b':model(row['prompt'],row['b'])})
        print(f'{index+1}/20 reward pairs scored locally',flush=True)
    report={'pairs':scored,'gate':calibration_gate(scored,allow_model_review=allow_model_review),'sheet_sha256':sha256(sheet),
        'model_review_authorized':bool(allow_model_review),
        'reward_model_revision':SOURCES['reward']['revision'],'paid_api_calls':0,
        'model_inventory_sha256':sha256(model_path/'bonus_source.json')}
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    command=sub.add_parser('prepare')
    command.add_argument('--source',type=Path,default=ROOT/'artifacts/data/response_train.jsonl')
    command.add_argument('--output',type=Path,default=ROOT/'artifacts/calibration')
    command=sub.add_parser('score')
    command.add_argument('--sheet',type=Path,default=ROOT/'artifacts/calibration/pairs.json')
    command.add_argument('--model',type=Path,default=ROOT/'artifacts/models/reward')
    command.add_argument('--output',type=Path,default=ROOT/'artifacts/calibration/scored.json')
    command.add_argument('--allow-model-review',action='store_true')
    command=sub.add_parser('check')
    command.add_argument('--report',type=Path,default=ROOT/'artifacts/calibration/scored.json')
    command.add_argument('--allow-model-review',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.source,args.output)
    elif args.action=='score':print(json.dumps(score(args.sheet,args.model,args.output,allow_model_review=args.allow_model_review)['gate'],ensure_ascii=False))
    else:
        gate=calibration_gate(json.loads(args.report.read_text())['pairs'],allow_model_review=args.allow_model_review)
        print(json.dumps(gate,ensure_ascii=False,indent=2))
        if not gate['passed']:raise SystemExit(2)


if __name__=='__main__':main()
