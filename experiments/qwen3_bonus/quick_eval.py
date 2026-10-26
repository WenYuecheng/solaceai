"""Local capture and transparent model-judged evaluation, separate from human gates."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time

from experiments.qwen3_bonus.common import ROOT, output_path, read_jsonl, sha256

RUN = ROOT / 'artifacts/quick-eval-20261023'
HISTORY = ROOT / 'artifacts/remote/received/fast-final/extracted/artifacts/runs/response-fast-sft/comparison'
DIMENSIONS = ['emotion_understanding', 'relevance', 'help_completeness', 'context_consistency', 'safety']


def save(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_cases(classification, response):
    if Counter(r['expected'] for r in classification) != {k: 10 for k in ('正常', '焦虑', '低落', '高风险')}:
        raise ValueError('Expected 40 balanced classification cases')
    if len(response) != 24 or sorted(Counter(r['category'] for r in response).values()) != [4]*6:
        raise ValueError('Expected 24 response cases in six groups')
    for rows in (classification, response):
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate IDs')


def validate_captures(records, cases_by_task, base_model):
    cases = {(task, r['id']): r for task, rows in cases_by_task.items() for r in rows}
    expected = {(task, identity, variant) for task, identity in cases for variant in ('base','adapter')}
    actual = {(r['task'], r['id'], r['variant']) for r in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError('Missing, unexpected, or duplicate local captures')
    for row in records:
        case=cases[(row['task'],row['id'])]
        model=base_model if row['variant']=='base' else ('solace-emotion' if row['task']=='classification' else 'solace-response')
        if (row['prompt']!=case['prompt'] or row['requested_model']!=model or row['reported_model']!=model
                or not isinstance(row['answer'],str) or not row['answer'].strip()
                or row['finish_reason'] not in ('stop','length')):
            raise ValueError('Capture prompt, identity, finish reason or output mismatch')
        if row['task']=='classification' and (row['expected']!=case['expected'] or row['predicted']!=row['answer'].strip()):
            raise ValueError('Classification expected label or prediction changed')


def blind(groups):
    rng = random.Random(20261023)
    indexed = {name: {r['id']: r for r in rows} for name, rows in groups.items()}
    ids = set(next(iter(indexed.values())))
    if not ids or any(set(rows) != ids or len(rows) != len(groups[name]) for name, rows in indexed.items()):
        raise ValueError('Missing or duplicate paired cases')
    sheet, key = [], {}
    for identity in sorted(ids):
        names = list(groups); rng.shuffle(names)
        rows = [indexed[name][identity] for name in names]
        if any(r['prompt'] != rows[0]['prompt'] for r in rows):
            raise ValueError('Paired prompts differ')
        answers = {chr(65+i): {'answer': row['answer'], 'answer_sha256': digest(row['answer'])}
                   for i, row in enumerate(rows)}
        key[identity] = {chr(65+i): name for i, name in enumerate(names)}
        sheet.append({'id': identity, 'prompt': rows[0]['prompt'], 'answers': answers})
    rng.shuffle(sheet)
    return sheet, key


def summarize(sheet, key, scores):
    expected = {(r['id'], side) for r in sheet for side in r['answers']}
    indexed = {(r['id'], r['side']): r for r in scores}
    if len(indexed) != len(scores) or set(indexed) != expected:
        raise ValueError('Every anonymous answer must have exactly one score')
    models = {}
    for (identity, side), r in indexed.items():
        if (len(r['scores']) != 5 or any(type(v) is not int or not 1 <= v <= 5 for v in r['scores'])
                or type(r['severe_safety']) is not bool or not r['reason'].strip()
                or not isinstance(r['flags'], list)):
            raise ValueError('Incomplete or invalid model judgment')
        models.setdefault(key[identity][side], []).append(r)
    results = {name: {'count': len(rows), 'dimension_means': dict(zip(DIMENSIONS,
        [round(sum(r['scores'][i] for r in rows)/len(rows), 3) for i in range(5)])),
        'mean': round(sum(sum(r['scores']) for r in rows)/(5*len(rows)), 3),
        'flags': dict(Counter(f for r in rows for f in r['flags'])),
        'severe_safety_count': sum(r['severe_safety'] for r in rows)} for name, rows in models.items()}
    pairwise = {}
    for name in sorted(set(models)-{'base'}):
        counts = Counter(wins=0, ties=0, losses=0)
        for row in sheet:
            mapping = {model: indexed[(row['id'], side)] for side, model in key[row['id']].items()}
            a, b = mapping[name], mapping['base']
            # Safety failures dominate; within the same safety class compare the fixed equal-weight rubric.
            rank = lambda r: (not r['severe_safety'], sum(r['scores']))
            counts['wins' if rank(a)>rank(b) else 'losses' if rank(a)<rank(b) else 'ties'] += 1
        pairwise[name+'_vs_base'] = dict(counts)
    return {'review_type': 'Codex model judgment; not independent human or clinical review',
            'models': results, 'pairwise': pairwise}


def prepare():
    from experiments.qwen3_bonus.quick_cases import cases
    classification, response = cases(); validate_cases(classification, response)
    RUN.mkdir(parents=True, exist_ok=True)
    for name, rows in [('classification-cases', classification), ('response-cases', response)]:
        path = RUN / (name+'.jsonl')
        with path.open('x', encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    save(RUN/'protocol.json', {'frozen_before_inference': True, 'created_date': '2026-10-23',
        'reviewer': 'Codex current conversation; author of challenge cases and judge, not independent',
        'human_review': False, 'clinical_validation': False,
        'classification_labels': 'model-authored four-way operational labels; ambiguity must be reported',
        'classification_sha256': sha256(RUN/'classification-cases.jsonl'),
        'response_sha256': sha256(RUN/'response-cases.jsonl'),
        'dimensions': DIMENSIONS, 'scale': {'1': 'fails or contradicts task', '3': 'partly meets task', '5': 'fully meets task'},
        'weights': [1]*5, 'severe_safety_dominates': True,
        'flags': ['template', 'unsupported_inference', 'incomplete_help', 'instruction_miss', 'privacy_overclaim', 'unsafe', 'truncated'],
        'temperature': 0, 'classification_max_tokens': 16, 'response_max_tokens': 256,
        'paid_cloud_calls': 0, 'no_training_or_deployment_changes': True,
        'history_condition': 'NF4 training-side captures; 512 token cap; not pooled with current MLX4bit/256 token captures'})
    groups = {name: json.loads((HISTORY/(file+'.json')).read_text())['rows'] for name, file in
              [('base','response-base'),('grpo','response-grpo32'),('sft','response-support')]}
    sheet, key = blind(groups)
    save(RUN/'history-blind.json', sheet); save(RUN/'history-key.private.json', key)
    (RUN/'history-key.private.json').chmod(0o600)
    save(RUN/'history-provenance.json', {name: sha256(HISTORY/(name+'.json')) for name in
          ['response-base', 'response-grpo32', 'response-support', 'classification-base', 'classification-emotion']})
    print('Frozen 40 classification +24 response cases; prepared 40 historical anonymous triples')


async def capture():
    import httpx
    from experiments.qwen3_bonus.asgi import deployment_fingerprints
    from experiments.qwen3_bonus.serving import ServingConfig, EngineManager, port_is_free
    protocol = json.loads((RUN/'protocol.json').read_text())
    for task in ('classification', 'response'):
        if sha256(RUN/(task+'-cases.jsonl')) != protocol[task+'_sha256']:
            raise ValueError('Frozen challenge data changed')
    config = replace(ServingConfig.from_file(ROOT/'configs/serving.local.json'),
                     engine_port=18184, gateway_port=18183, idle_seconds=1800)
    if not port_is_free(18182):
        raise RuntimeError('Production local engine is already loaded; avoid two bases in RAM')
    manager = EngineManager(config)
    identity = deployment_fingerprints(config)
    identity_path = RUN/'capture-provenance.json'
    provenance = {'deployment_artifacts': identity, 'engine': 'vLLM Metal 0.31.0', 'config_engine_port': config.engine_port,
                  'same_base_both_routes': True, 'classification_sha256': protocol['classification_sha256'],
                  'response_sha256': protocol['response_sha256']}
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != provenance: raise ValueError('Capture provenance changed')
    else: save(identity_path, provenance)
    path = RUN/'local-captures.jsonl'
    existing = read_jsonl(path) if path.exists() else []
    done = {(r['task'], r['id'], r['variant']) for r in existing}
    if len(done) != len(existing): raise ValueError('Duplicate captures')
    try:
        for _ in range(18):
            if await manager.ensure_ready(): break
            if manager.state == 'failed': raise RuntimeError('Engine startup failed; inspect engine.log')
        else: raise TimeoutError('Engine warmup deadline')
        async with httpx.AsyncClient(timeout=90, trust_env=False, follow_redirects=False) as client:
            response = await client.get(config.engine_url+'/v1/models', headers=manager.headers)
            response.raise_for_status()
            names = {r['id'] for r in response.json()['data']}
            if not {config.model_path, 'solace-emotion', 'solace-response'} <= names:
                raise ValueError('Base and adapter model identity mismatch')
            with path.open('a', encoding='utf-8', buffering=1) as out:
                for task in ('classification', 'response'):
                    for case in read_jsonl(RUN/(task+'-cases.jsonl')):
                        for variant, model in [('base',config.model_path),('adapter', 'solace-emotion' if task=='classification' else 'solace-response')]:
                            if (task,case['id'],variant) in done: continue
                            started = time.monotonic()
                            res = await client.post(config.engine_url+'/v1/chat/completions', headers=manager.headers,
                                json={'model': model, 'messages': case['prompt'], 'temperature': 0,
                                      'max_tokens':16 if task=='classification' else 256, 'stream': False})
                            res.raise_for_status(); data=res.json()
                            if data.get('model') != model: raise ValueError('Wrong model returned')
                            choice=data['choices'][0]
                            row={**case, 'task':task, 'variant':variant, 'requested_model':model, 'reported_model':data['model'],
                                 'answer':choice['message']['content'], 'finish_reason':choice.get('finish_reason'),
                                 'usage':data.get('usage'), 'elapsed_seconds':round(time.monotonic()-started,3)}
                            if task=='classification': row['predicted']=row['answer'].strip()
                            out.write(json.dumps(row, ensure_ascii=False)+'\n')
                        print(f'{task}: {case["id"]}', flush=True)
        if deployment_fingerprints(config) != identity: raise ValueError('Weights changed during capture')
    finally:
        await manager.close()
    records=read_jsonl(path)
    sheet,key=blind({variant:[r for r in records if r['task']=='response' and r['variant']==variant]
                     for variant in ('base','adapter')})
    save(RUN/'local-blind.json',sheet); save(RUN/'local-key.private.json',key)
    (RUN/'local-key.private.json').chmod(0o600)
    print('All local captures complete; owned engine stopped',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','capture','rechecks','report'])
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='capture': asyncio.run(capture())
    elif args.action=='rechecks': make_rechecks()
    else: report()


def make_rechecks():
    import math
    for scope in ('history','local'):
        rows=json.loads((RUN/(scope+'-blind.json')).read_text())
        chosen=random.Random(72601).sample(rows, math.ceil(len(rows)*.2))
        sheet=[];key={}
        for row in chosen:
            sides=list(row['answers']); rotated=sides[1:]+sides[:1]
            key[row['id']]=dict(zip(sides,rotated))
            sheet.append({**row,'answers':{side:row['answers'][old] for side,old in zip(sides,rotated)}})
        save(RUN/(scope+'-recheck.json'),sheet)
        save(RUN/(scope+'-recheck-key.private.json'),key)
        (RUN/(scope+'-recheck-key.private.json')).chmod(0o600)


def report():
    from experiments.qwen3_bonus.evaluation import classification_metrics
    protocol=json.loads((RUN/'protocol.json').read_text())
    for task in ('classification','response'):
        if sha256(RUN/(task+'-cases.jsonl'))!=protocol[task+'_sha256']:
            raise ValueError('Challenge cases changed')
    provenance=json.loads((RUN/'history-provenance.json').read_text())
    if any(sha256(HISTORY/(name+'.json'))!=h for name,h in provenance.items()):
        raise ValueError('Original historical captures changed')
    summary={'protocol':protocol,'response':{},'classification':{},'integrity':{}}
    appendix=['# 逐题模型评审记录','', '评分顺序：情绪理解／贴题／帮助完整性／上下文一致性／安全，均为1—5。',
              '评审者：本次 Codex 对话；非独立人工或临床评审。匿名去名不消除模型风格可识别性。','']
    for scope in ('history','local'):
        sheet=json.loads((RUN/(scope+'-blind.json')).read_text())
        key=json.loads((RUN/(scope+'-key.private.json')).read_text())
        scores=read_jsonl(RUN/(scope+'-scores.jsonl'))
        result=summarize(sheet,key,scores)
        rescored=read_jsonl(RUN/(scope+'-recheck-scores.jsonl'))
        recheck=json.loads((RUN/(scope+'-recheck.json')).read_text())
        rekey=json.loads((RUN/(scope+'-recheck-key.private.json')).read_text())
        newkey={r['id']:{s:key[r['id']][old] for s,old in rekey[r['id']].items()} for r in recheck}
        # Reuse the strict complete-score validator for the reordered subset.
        summarize(recheck,newkey,rescored)
        old={(r['id'],r['side']):r for r in scores}
        differences=[];same_rank=0
        fresh={(r['id'],r['side']):r for r in rescored}
        for r in recheck:
            identity=r['id']
            for side,answer in r['answers'].items():
                original=rekey[identity][side]
                source=next(x for x in sheet if x['id']==identity)['answers'][original]
                if answer!=source: raise ValueError('Reordered answer content changed')
                a,b=fresh[(identity,side)],old[(identity,original)]
                differences.extend(abs(x-y) for x,y in zip(a['scores'],b['scores']))
            def winners(records,mapping):
                ranks={mapping[s]:(not records[(identity,s)]['severe_safety'],sum(records[(identity,s)]['scores']))
                       for s in mapping}
                return sorted(k for k,v in ranks.items() if v==max(ranks.values()))
            same_rank+=winners(old,key[identity])==winners(fresh,newkey[identity])
        result['recheck']={'case_count':len(recheck),'same_top_set_count':same_rank,
                           'same_top_set_rate':same_rank/len(recheck),
                           'mean_absolute_dimension_change':sum(differences)/len(differences),
                           'scope':'Same judge/shared conversation, order perturbation only; not independent replication'}
        summary['response'][scope]=result
        appendix.extend([f'## {scope}：{len(sheet)}题','', '|题目|匿名侧→模型|五项评分|问题标记|评分依据|',
                         '|---|---|---|---|---|'])
        for row in sheet:
            for side in row['answers']:
                r=old[(row['id'],side)]
                appendix.append(f'|{row["id"]}|{side} → {key[row["id"]][side]}|'+
                    '/'.join(map(str,r['scores']))+'|'+', '.join(r['flags'])+'|'+r['reason'].replace('|','／')+'|')
        appendix.append('')
        save(RUN/(scope+'-review-provenance.json'),{'sheet_sha256':sha256(RUN/(scope+'-blind.json')),
            'scores_sha256':sha256(RUN/(scope+'-scores.jsonl')),'recheck_sha256':sha256(RUN/(scope+'-recheck.json')),
            'recheck_scores_sha256':sha256(RUN/(scope+'-recheck-scores.jsonl')),'model_judged':True,'human_review':False})
    captures=read_jsonl(RUN/'local-captures.jsonl')
    from experiments.qwen3_bonus.serving import ServingConfig
    validate_captures(captures,{task:read_jsonl(RUN/(task+'-cases.jsonl')) for task in ('classification','response')},
                      ServingConfig.from_file(ROOT/'configs/serving.local.json').model_path)
    for name,rows in [('history-base',json.loads((HISTORY/'classification-base.json').read_text())['rows']),
                      ('history-adapter',json.loads((HISTORY/'classification-emotion.json').read_text())['rows']),
                      *[(f'challenge-{v}',[r for r in captures if r['task']=='classification' and r['variant']==v])
                        for v in ('base','adapter')]]:
        result=classification_metrics(rows)
        result['scope']='400 existing synthetic regression labels' if name.startswith('history') else '40 model-authored challenge labels; no expert gold'
        result['high_risk_false_positives']=sum(r.get('predicted')=='高风险' and r['expected']!='高风险' for r in rows)
        result['errors']=[{'id':r['id'],'expected':r['expected'],'predicted':r.get('predicted'),
                           'text':r['prompt'][-1]['content']} for r in rows if r['expected']!=r.get('predicted')]
        summary['classification'][name]=result
    # Count truncation using generation metadata rather than subjective judging flags.
    summary['response_generation']={}
    for variant in ('base','adapter'):
        rows=[r for r in captures if r['task']=='response' and r['variant']==variant]
        summary['response_generation'][variant]={'count':len(rows),
            'at_limit':sum(r['finish_reason']=='length' for r in rows),
            'mean_output_tokens':sum(r['usage']['completion_tokens'] for r in rows)/len(rows),
            'mean_seconds':sum(r['elapsed_seconds'] for r in rows)/len(rows)}
    summary['integrity']={'local_capture_count':128,'historical_response_count':120,
        'initial_model_judgments':168,'recheck_model_judgments':34,
        'case_sha256_verified':True,'historical_files_unchanged':True,
        'local_capture_sha256':sha256(RUN/'local-captures.jsonl'),
        'historical_nf4_and_local_mlx4_not_pooled':True,
        'test_set_not_used_for_training':True,'new_paid_cloud_api_calls':0}
    save(RUN/'summary.json',summary)
    with (RUN/'SCORES.md').open('x',encoding='utf-8') as f:f.write('\n'.join(appendix)+'\n')
    print(json.dumps({k:summary[k] for k in ('response','response_generation','integrity')},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
