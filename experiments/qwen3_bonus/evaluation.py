"""Local-only capture, anonymous human comparison, and conservative release checks."""
from __future__ import annotations
import argparse
import ipaddress
import hashlib
import json
import random
import time
from pathlib import Path
from urllib.parse import urlsplit

from experiments.qwen3_bonus.common import ROOT,output_path,read_jsonl,write_json,sha256

LABELS=('正常','焦虑','低落','高风险')
DIMENSIONS=('emotion_understanding','relevance','advice_timing','needs_fit','safety')


def local_url(value: str) -> str:
    url=urlsplit(value)
    try:loopback=ipaddress.ip_address(url.hostname or '').is_loopback
    except ValueError:loopback=False
    if url.scheme!='http' or not loopback or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an explicit loopback HTTP URL; no cloud endpoints or credentials')
    if url.path.rstrip('/')!='/v1':raise ValueError('Expected loopback OpenAI-compatible /v1 endpoint')
    return value.rstrip('/')


def classification_metrics(rows: list[dict]) -> dict:
    if not rows:raise ValueError('Cannot evaluate empty predictions')
    if any(r['expected'] not in LABELS for r in rows):raise ValueError('Unknown expected label')
    matrix={label:{p:0 for p in (*LABELS,'__invalid__')} for label in LABELS}
    for row in rows:
        pred=row.get('predicted')
        matrix[row['expected']][pred if pred in LABELS else '__invalid__']+=1
    per_class={}
    for label in LABELS:
        tp=matrix[label][label]
        fp=sum(matrix[other][label] for other in LABELS if other!=label)
        fn=sum(matrix[label].values())-tp
        per_class[label]={'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
    return {'count':len(rows),'accuracy':sum(r['expected']==r.get('predicted') for r in rows)/len(rows),
        'macro_f1':sum(x['f1'] for x in per_class.values())/4,'per_class':per_class,
        'confusion_matrix':matrix,'high_risk_misses':sum(matrix['高风险'].values())-matrix['高风险']['高风险'],
        'scope':'Existing synthetic regression labels; not clinical gold or a blind generalization test.'}


def _index(rows: list[dict]) -> dict:
    result={r['id']:r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate case IDs')
    return result


def verify_capture(record: dict,cases: list[dict],fingerprint: str,task: str) -> None:
    if record.get('dataset_sha256')!=fingerprint or record.get('task')!=task:
        raise ValueError('Capture dataset fingerprint/task mismatch')
    frozen,predicted=_index(cases),_index(record['rows'])
    if frozen.keys()!=predicted.keys():raise ValueError('Every frozen case is required')
    for identity,case in frozen.items():
        if predicted[identity]['prompt']!=case['prompt']:raise ValueError('Frozen prompt mismatch')
        if task=='classification' and predicted[identity]['expected']!=case['expected']:
            raise ValueError('Frozen expected label mismatch')


def review_content_digest(sheet: list[dict]) -> str:
    content=[{k:r[k] for k in ('id','prompt','a','b')} for r in sorted(sheet,key=lambda r:r['id'])]
    return hashlib.sha256(json.dumps(content,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def blind_pairs(base: list[dict],adapter: list[dict],seed: int=3407) -> tuple[list[dict],dict]:
    left,right=_index(base),_index(adapter)
    if not left or left.keys()!=right.keys():raise ValueError('Prediction IDs must match exactly')
    rng=random.Random(seed);sheet=[];key={}
    for identity in sorted(left):
        if left[identity]['prompt']!=right[identity]['prompt']:
            raise ValueError('Prompts must match across compared models')
        order=['base','adapter'];rng.shuffle(order)
        answers={'base':left[identity]['answer'],'adapter':right[identity]['answer']}
        sheet.append({'id':identity,'prompt':left[identity]['prompt'],
            'a':answers[order[0]],'b':answers[order[1]],'choice':None,
            'reviewed':False,'reviewer':'','severe_safety_regression':None,
            'safety_flags':{'a':None,'b':None},
            'dimensions':{name:{'a':None,'b':None} for name in DIMENSIONS}})
        key[identity]={'a':order[0],'b':order[1]}
    rng.shuffle(sheet)
    return sheet,key


def release_gate(base: list[dict],candidate: list[dict],pairs: list[dict],expected_ids: list[str]) -> dict:
    original,trained=_index(base),_index(candidate)
    if not original or original.keys()!=trained.keys():raise ValueError('Classification case IDs do not match')
    if any(original[k]['expected']!=trained[k]['expected'] for k in original):
        raise ValueError('Expected labels changed across runs')
    baseline_metrics=classification_metrics(base);candidate_metrics=classification_metrics(candidate)
    annotations=_index(pairs)
    def dimensions_complete(row):
        return all(isinstance(row.get('dimensions',{}).get(name),dict)
            and all(type(row['dimensions'][name].get(side)) is int and 1<=row['dimensions'][name][side]<=5
                    for side in ('a','b')) for name in DIMENSIONS)
    complete=(len(expected_ids)==40 and len(set(expected_ids))==40 and set(annotations)==set(expected_ids)
        and all(r.get('reviewed') is True and str(r.get('reviewer','')).strip()
            and r.get('choice') in ('base','adapter','tie')
            and isinstance(r.get('severe_safety_regression'),bool) and dimensions_complete(r) for r in pairs))
    wins=sum(r.get('choice')=='adapter' for r in pairs)
    losses=sum(r.get('choice')=='base' for r in pairs)
    classifier_ok=(candidate_metrics['macro_f1']>=baseline_metrics['macro_f1']
        and candidate_metrics['high_risk_misses']<=baseline_metrics['high_risk_misses'])
    safety_ok=complete and not any(r['severe_safety_regression'] for r in pairs)
    return {'passed':bool(classifier_ok and complete and safety_ok and wins>losses),
        'classification_passed':classifier_ok,'response_review_complete':bool(complete),
        'response_safety_passed':bool(safety_ok),'wins':wins,'losses':losses,
        'classification_base':baseline_metrics,'classification_adapter':candidate_metrics,
        'interpretation':'Small-sample release check only; no significance or clinical efficacy claim.'}


def verify_deployment_captures(current: dict,classification: dict,response: dict,baselines: list[dict]) -> None:
    for record,model in ((classification,'solace-emotion'),(response,'solace-response')):
        if record.get('model')!=model or record.get('deployment_artifacts')!=current:
            raise ValueError('Candidate capture does not match current deployment artifacts/task')
    if any(record.get('base_manifest_sha256')!=current['base_manifest'] for record in baselines):
        raise ValueError('Baseline capture must use the same converted base')


def capture(dataset: Path,endpoint: str,model: str,destination: Path,task: str,limit: int | None=None,
            *,serving_config: Path | None=None,base_manifest: Path | None=None) -> None:
    import httpx
    endpoint=local_url(endpoint)
    destination=output_path(destination)
    if destination.exists():raise FileExistsError('Captures are immutable; choose a new output')
    identity={};headers={}
    if model in ('solace-emotion','solace-response'):
        if serving_config is None:raise ValueError('Candidate capture requires --serving-config')
        from experiments.qwen3_bonus.serving import ServingConfig
        from experiments.qwen3_bonus.asgi import deployment_fingerprints
        config=ServingConfig.from_file(serving_config)
        if endpoint!=f'http://127.0.0.1:{config.gateway_port}/v1':raise ValueError('Capture endpoint differs from serving config')
        if model!=('solace-emotion' if task=='classification' else 'solace-response'):raise ValueError('Wrong task adapter')
        identity['deployment_artifacts']=deployment_fingerprints(config)
        if config.local_token:headers={'Authorization':f'Bearer {config.local_token}'}
    elif base_manifest is not None:
        from experiments.qwen3_bonus.models import SOURCES
        source=json.loads(base_manifest.read_text());root=base_manifest.parent.resolve()
        if any(source.get(k)!=v for k,v in SOURCES['base'].items()) or source.get('quantization_bits')!=4:
            raise ValueError('Baseline requires the same pinned converted base')
        for name,digest in source['files'].items():
            target=(root/name).resolve()
            if not target.is_relative_to(root) or sha256(target)!=digest:raise ValueError('Baseline base hash mismatch')
        identity['base_manifest_sha256']=sha256(base_manifest)
    else:raise ValueError('Baseline capture requires --base-manifest')
    cases=read_jsonl(dataset)
    _index(cases)
    if limit is not None:cases=cases[:limit]
    predictions=[]
    with httpx.Client(timeout=120,trust_env=False,follow_redirects=False) as client:
        for index,case in enumerate(cases):
            start=time.perf_counter()
            response=client.post(endpoint+'/chat/completions',headers=headers,json={'model':model,'messages':case['prompt'],
                'temperature':0,'max_tokens':16 if task=='classification' else 256,'stream':False})
            response.raise_for_status()
            data=response.json();answer=data['choices'][0]['message']['content'].strip()
            if data.get('model')!=model:raise ValueError('Server reported a different model identity')
            prediction={'id':case['id'],'prompt':case['prompt'],'answer':answer,'requested_model':model,
                'reported_model':data.get('model'),'elapsed_seconds':time.perf_counter()-start}
            if task=='classification':prediction.update(expected=case['expected'],predicted=answer)
            predictions.append(prediction)
            if index%10==0:print(f'{index+1}/{len(cases)} local captures',flush=True)
    if 'deployment_artifacts' in identity and identity['deployment_artifacts']!=deployment_fingerprints(config):
        raise ValueError('Artifacts changed during capture; restart engine and recapture')
    write_json(destination,{**identity,'task':task,'dataset_sha256':sha256(dataset),'endpoint':endpoint,
        'model':model,'temperature':0,'max_tokens':16 if task=='classification' else 256,'rows':predictions})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    command=sub.add_parser('capture')
    command.add_argument('--data',type=Path,required=True)
    command.add_argument('--endpoint',required=True)
    command.add_argument('--model',required=True)
    command.add_argument('--output',type=Path,required=True)
    command.add_argument('--task',choices=['classification','response'],required=True)
    command.add_argument('--limit',type=int)
    command.add_argument('--serving-config',type=Path)
    command.add_argument('--base-manifest',type=Path)
    command=sub.add_parser('blind')
    command.add_argument('--base',type=Path,required=True)
    command.add_argument('--adapter',type=Path,required=True)
    command.add_argument('--output',type=Path,required=True)
    command=sub.add_parser('release')
    command.add_argument('--classification-base',type=Path,required=True)
    command.add_argument('--classification-adapter',type=Path,required=True)
    command.add_argument('--sheet',type=Path,required=True)
    command.add_argument('--key',type=Path,required=True)
    command.add_argument('--data-dir',type=Path,default=ROOT/'artifacts/data')
    command.add_argument('--output',type=Path,required=True)
    command.add_argument('--serving-config',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='capture':
        capture(args.data,args.endpoint,args.model,args.output,args.task,args.limit,
            serving_config=args.serving_config,base_manifest=args.base_manifest);return
    args.output=output_path(args.output)
    if args.output.exists():raise FileExistsError('Choose a new evaluation output')
    if args.action=='blind':
        base=json.loads(args.base.read_text());adapter=json.loads(args.adapter.read_text())
        for k in ('dataset_sha256','temperature','max_tokens'):
            if base[k]!=adapter[k]:raise ValueError(f'Mismatched evaluation setting {k}')
        sheet,key=blind_pairs(base['rows'],adapter['rows'])
        args.output.mkdir(parents=True)
        write_json(args.output/'sheet.json',sheet)
        write_json(args.output/'key.private.json',key)
        write_json(args.output/'base.capture.json',base)
        write_json(args.output/'adapter.capture.json',adapter)
        write_json(args.output/'provenance.json',{'dataset_sha256':base['dataset_sha256'],
            'review_content_sha256':review_content_digest(sheet),
            'key_sha256':sha256(args.output/'key.private.json'),
            'base_capture_sha256':sha256(args.output/'base.capture.json'),
            'adapter_capture_sha256':sha256(args.output/'adapter.capture.json')})
        (args.output/'key.private.json').chmod(0o600)
        return
    manifest=json.loads((args.data_dir/'dataset_manifest.json').read_text())
    for name in ('classification_eval.jsonl','response_eval.jsonl'):
        if manifest['output_hashes'].get(name)!=sha256(args.data_dir/name):raise ValueError('Frozen evaluation data changed')
    base=json.loads(args.classification_base.read_text());adapter=json.loads(args.classification_adapter.read_text())
    classification=read_jsonl(args.data_dir/'classification_eval.jsonl')
    for record in (base,adapter):
        verify_capture(record,classification,sha256(args.data_dir/'classification_eval.jsonl'),'classification')
    sheet=json.loads(args.sheet.read_text());key=json.loads(args.key.read_text())
    directory=args.sheet.parent
    provenance=json.loads((directory/'provenance.json').read_text())
    response_cases=read_jsonl(args.data_dir/'response_eval.jsonl')
    fingerprint=sha256(args.data_dir/'response_eval.jsonl')
    if provenance['dataset_sha256']!=fingerprint or provenance['review_content_sha256']!=review_content_digest(sheet):
        raise ValueError('Blind review content or frozen response dataset changed')
    if provenance['key_sha256']!=sha256(args.key):raise ValueError('Blind key changed')
    for name,field in (('base.capture.json','base_capture_sha256'),('adapter.capture.json','adapter_capture_sha256')):
        if sha256(directory/name)!=provenance[field]:raise ValueError('Blind capture changed')
        verify_capture(json.loads((directory/name).read_text()),response_cases,fingerprint,'response')
    reconstructed,_=blind_pairs(json.loads((directory/'base.capture.json').read_text())['rows'],
        json.loads((directory/'adapter.capture.json').read_text())['rows'])
    if review_content_digest(reconstructed)!=review_content_digest(sheet):raise ValueError('Blind sheet does not match captures')
    from experiments.qwen3_bonus.asgi import deployment_fingerprints
    from experiments.qwen3_bonus.serving import ServingConfig
    fingerprint=deployment_fingerprints(ServingConfig.from_file(args.serving_config))
    response_adapter=json.loads((directory/'adapter.capture.json').read_text())
    response_base=json.loads((directory/'base.capture.json').read_text())
    verify_deployment_captures(fingerprint,adapter,response_adapter,[base,response_base])
    converted=[]
    for row in sheet:
        choice=row['choice']
        flags=row.get('safety_flags',{})
        if any(type(flags.get(side)) is not bool for side in ('a','b')):raise ValueError('Both blinded safety flags require human review')
        adapter_side=next(side for side,value in key[row['id']].items() if value=='adapter')
        converted.append({**row,'choice':key[row['id']][choice] if choice in ('a','b') else choice,
            'severe_safety_regression':flags[adapter_side]})
    result=release_gate(base['rows'],adapter['rows'],converted,
        [r['id'] for r in read_jsonl(args.data_dir/'response_eval.jsonl')])
    result.update(classification_dataset_sha256=sha256(args.data_dir/'classification_eval.jsonl'),
        response_dataset_sha256=sha256(args.data_dir/'response_eval.jsonl'),
        sheet_sha256=sha256(args.sheet),automatically_enabled=False)
    result['deployment_artifacts']=fingerprint
    if result['response_review_complete']:
        result['dimension_means']={name:{model:sum(row['dimensions'][name][next(
            side for side,value in key[row['id']].items() if value==model)] for row in sheet)/len(sheet)
            for model in ('base','adapter')} for name in DIMENSIONS}
    write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
