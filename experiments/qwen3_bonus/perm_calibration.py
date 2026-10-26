"""CPU-only explicit calibration and conservative resource-bound model selection."""
import argparse
import hashlib
import json
import math
import resource
import statistics
import sys
import time
from pathlib import Path
from experiments.qwen3_bonus.common import output_path,sha256
from experiments.qwen3_bonus.gdpo_data import immutable_write,_json
from experiments.qwen3_bonus.perm_models import CANDIDATES,verify_model
from experiments.qwen3_bonus.perm_rewards import FrozenPermRewardModel,component_scores,apply_group_gate,INPUT_FORMAT,REWARD_SPEC,CODE_DEPENDENCIES
from experiments.qwen3_bonus.rewards import invalid_reason


def fingerprint(value):return hashlib.sha256(_json(value)).hexdigest()


def normalization(raw):
    raw=[float(v) for v in raw if v is not None]
    if not raw or any(not math.isfinite(v) for v in raw):raise ValueError('Finite valid raw scores required')
    ordered=sorted(raw)
    def percentile(q):
        x=(len(ordered)-1)*q;lo=int(x);hi=min(lo+1,len(ordered)-1)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(x-lo)
    return {'median':statistics.median(raw),'iqr':percentile(.75)-percentile(.25),'valid_raw_count':len(raw)}


def final_gate(rows,subset):
    if len(rows)!=20 or len(set(subset))!=8 or not set(subset)<=set(r['id'] for r in rows):raise ValueError('Twenty pairs and frozen eight IDs required')
    if len({r['id'] for r in rows})!=20:raise ValueError('Duplicate pair IDs')
    for r in rows:
        if r['expected'] not in ('a','b') or any(not math.isfinite(float(r[k])) for k in ('score_a','score_b')):raise ValueError('Invalid calibration score/label')
    correct=lambda r:r['score_'+r['expected']]>r['score_'+('b' if r['expected']=='a' else 'a')]
    n=sum(correct(r) for r in rows);n8=sum(correct(r) for r in rows if r['id'] in subset)
    return {'correct':n,'count':20,'task_delivery_correct':n8,'task_delivery_count':8,'ties':sum(r['score_a']==r['score_b'] for r in rows),'passed':n>=16 and n8>=7}


def _selection_policy(candidates,resources):
    required=('physical_memory_bytes','runtime_bytes','actor_ref_offload_bytes','remaining_training_yuan','hourly_yuan','prep_setup_save_reserve_yuan')
    if any(k not in resources or type(resources[k]) not in (int,float) or not math.isfinite(resources[k]) or resources[k]<0 for k in required):raise ValueError('Explicit finite resource and billing inputs required')
    limits=[resources['physical_memory_bytes']]
    if resources.get('cgroup_limit_bytes') is not None:
        if type(resources['cgroup_limit_bytes']) not in (int,float) or not math.isfinite(resources['cgroup_limit_bytes']) or resources['cgroup_limit_bytes']<=0:raise ValueError('Invalid cgroup limit')
        limits.append(resources['cgroup_limit_bytes'])
    effective=min(limits);audits=[];eligible=[]
    for c in candidates:
        identity=next((k for k,v in CANDIDATES.items() if all(c.get(a)==b for a,b in v.items())),None)
        if identity is None:raise ValueError('Unapproved candidate identity')
        if any(type(c[k]) not in (int,float) for k in ('peak_rm_bytes','seconds_per_response')):raise ValueError('Numeric candidate measurements required')
        numbers=[float(c[k]) for k in ('peak_rm_bytes','seconds_per_response')]
        if any(not math.isfinite(v) or v<0 for v in numbers):raise ValueError('Invalid measured resources')
        memory=numbers[0]+resources['runtime_bytes']+resources['actor_ref_offload_bytes']
        cost=512*numbers[1]/3600*resources['hourly_yuan']
        budget=max(0,resources['remaining_training_yuan']-resources['prep_setup_save_reserve_yuan'])
        passed=c.get('gate_passed') is True and resources.get('offload_verified') is True and memory<=.8*effective and cost<=.25*budget and budget>0
        audits.append({'identity':identity,'passed':passed,'memory_required_bytes':memory,'effective_limit_bytes':effective,'predicted_512_scoring_yuan':cost,'scoring_budget_yuan':.25*budget})
        if passed:eligible.append((identity,c))
    eligible.sort(key=lambda x:0 if x[0]=='1.7b' else 1)
    selected=None if not eligible else {k:eligible[0][1][k] for k in ('repo_id','revision','model_path','model_inventory_sha256','calibration_path','calibration_sha256')}
    result={'schema_version':1,'passed':bool(selected),'selected':selected,'candidate_audits':audits,'resource_sha256':fingerprint(resources),'candidate_evidence_sha256':fingerprint(candidates),'input_format':INPUT_FORMAT,'input_format_sha256':fingerprint(INPUT_FORMAT),'candidates':candidates,'resources':resources}
    result['selection_sha256']=fingerprint(result)
    return result



def estimate_untrusted(candidates,resources):
    result=_selection_policy(candidates,resources)
    result['estimated_passed']=result['passed'];result['estimated_selected']=result['selected']
    result['passed']=False;result['selected']=None;result['evidence_verified']=False
    result.pop('selection_sha256');result['selection_sha256']=fingerprint(result)
    return result


def validate_report(report):
    manifest_path=Path(report['data_manifest_path']);manifest=json.loads(manifest_path.read_text())
    for name,entry in manifest['files'].items():
        target=(manifest_path.parent/name).resolve()
        if not target.is_relative_to(manifest_path.parent.resolve()) or sha256(target)!=entry['sha256']:raise ValueError('Calibration input drift')
    pairs=json.loads((manifest_path.parent/'calibration_pairs.json').read_text())['pairs']
    labels=[{k:r[k] for k in ('id','expected','reviewed','reviewer','review_type')} for r in pairs]
    if fingerprint(labels)!=report['labels_sha256'] or report['input_format_sha256']!=fingerprint(INPUT_FORMAT):raise ValueError('Calibration labels/input mismatch')
    raw=report['raw_ordering']
    if len(raw)!=20 or [x['id'] for x in raw]!=[r['id'] for r in pairs]:raise ValueError('Calibration pair coverage mismatch')
    norm=normalization([x['raw_'+c] for x in raw for c in ('a','b')])
    if any(report[k]!=v for k,v in norm.items()):raise ValueError('Calibration normalization mismatch')
    derived=[]
    for r,x in zip(pairs,raw):
        records=[component_scores(x['raw_'+c],r[c],r['constraints'],norm) for c in ('a','b')]
        gated,_=apply_group_gate(records,[r['id']]*2)
        derived.append({'id':r['id'],'expected':r['expected'],'score_a':gated[0]['score'],'score_b':gated[1]['score']})
    stored=report['methods']['harmonic']['pairs']
    if len(stored)!=20 or any(any(a[k]!=b[k] for k in ('id','expected','score_a','score_b')) for a,b in zip(derived,stored)):raise ValueError('Final pair records mismatch')
    gate=final_gate(derived,manifest['task_delivery_ids'])
    if gate!=report['gate'] or gate!=report['methods']['harmonic']['gate']:raise ValueError('Recomputed calibration gate mismatch')
    return gate


def make_candidate(calibration_path):
    from experiments.qwen3_bonus.perm_rewards import load_calibration
    path=Path(calibration_path).resolve()
    load_calibration.cache_clear()
    report=load_calibration(str(path), require_passed=False);validate_report(report)
    identity=verify_model(Path(report['model']['model_path']))
    if identity!=report['model']:raise ValueError('Model evidence mismatch')
    bench=report['benchmark'];times=bench['eight_longest']
    if len(times)!=8 or type(bench['peak_process_rss_bytes']) not in (int,float) or bench['peak_process_rss_bytes']<=0:raise ValueError('Actual measured process RSS required')
    for item in times:
        if type(item['seconds']) not in (int,float) or not math.isfinite(item['seconds']) or item['seconds']<=0:raise ValueError('Measured response timing required')
    seconds=max(x['seconds'] for x in times)
    if seconds!=bench['seconds_per_response_conservative']:raise ValueError('Timing summary mismatch')
    return {k:identity[k] for k in ('repo_id','revision','model_path','model_inventory_sha256')}|{'calibration_path':str(path),'calibration_sha256':sha256(path),'gate_passed':report['gate']['passed'],'peak_rm_bytes':bench['peak_process_rss_bytes'],'seconds_per_response':seconds}


def select_model(candidates,resources):
    # Validate numeric inputs first, including unavailable-but-explicit cgroup null.
    _selection_policy([],resources)
    verified=[]
    for candidate in candidates:
        actual=make_candidate(candidate['calibration_path'])
        if actual!=candidate:raise ValueError('Candidate claims differ from measured calibration evidence')
        verified.append(actual)
    evidence_path=Path(resources.get('evidence_path',''))
    if not evidence_path.is_file() or sha256(evidence_path)!=resources.get('evidence_sha256'):raise ValueError('Frozen runtime/offload evidence required')
    evidence=json.loads(evidence_path.read_text())
    for key,value in resources.items():
        if key not in ('evidence_path','evidence_sha256') and evidence.get(key)!=value:raise ValueError('Resource fields differ from evidence snapshot')
    if evidence.get('phase') not in ('actor_reference_initialized','full_step') or evidence.get('offload_verified') is not True or not evidence.get('memory_metric_definitions'):raise ValueError('Initialized actor/reference offload evidence required')
    result=_selection_policy(verified,resources)
    result['evidence_verified']=True
    result.pop('selection_sha256');result['selection_sha256']=fingerprint(result)
    return result


def calibrate(data_dir,model_path,output,*,execute=False):
    if not execute:raise ValueError('Actual CPU scoring requires --execute')
    import torch
    data_dir=Path(data_dir);manifest_path=data_dir/'dataset_manifest.json';manifest=json.loads(manifest_path.read_text())
    for name,entry in manifest['files'].items():
        if sha256(data_dir/name)!=entry['sha256']:raise ValueError('Frozen calibration input hash mismatch')
    sheet=json.loads((data_dir/'calibration_pairs.json').read_text());pairs=sheet['pairs'];subset=manifest['task_delivery_ids']
    if any(not r.get('reviewed') or not r.get('reviewer') for r in pairs):raise ValueError('Unreviewed labels')
    import os, torch
    threads=int(os.environ.get('PERM_CPU_THREADS', '2'))
    if not 1 <= threads <= 8:raise ValueError('PERM CPU threads must be in 1..8')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(min(threads,2))
    identity=verify_model(model_path);model=FrozenPermRewardModel(str(model_path));scored=[];raws=[];timings=[]
    for r in pairs:
        values={}
        for choice in ('a','b'):
            start=time.perf_counter();raw=None if invalid_reason(r[choice]) else model(r['prompt'],r[choice]);elapsed=time.perf_counter()-start
            values[choice]=raw;raws.append(raw);timings.append({'id':r['id'],'choice':choice,'chars':len(r[choice]),'seconds':elapsed,'skipped_invalid':raw is None})
        scored.append({'pair':r,'raw':values})
    norm=normalization(raws);methods={k:[] for k in ('original','calibrated','arithmetic','harmonic')};raw_order=[]
    for entry in scored:
        r=entry['pair'];records=[component_scores(entry['raw'][ch],r[ch],r['constraints'],norm) for ch in ('a','b')]
        base=[dict(x) for x in records];gated,metrics=apply_group_gate(records,[r['id']]*2)
        for method in methods:
            results=[]
            for rec,g in zip(base,gated):
                if rec['invalid']:s=-1.
                elif method=='original':s=1/(1+math.exp(-max(-700,min(700,rec['raw_rm']))))-.2*rec['repeat_ratio']
                elif method=='calibrated':s=rec['preference']-.1*rec['repeat_ratio']
                elif method=='arithmetic':s=((rec['preference']+rec['constraint'])/2 if rec['constraint_applicable'] else rec['preference'])-.1*rec['repeat_ratio']
                else:s=rec['score']
                if method=='harmonic' and g['group_gate']:s=0.
                results.append(s)
            methods[method].append({'id':r['id'],'expected':r['expected'],'score_a':results[0],'score_b':results[1],'group_gate':method=='harmonic' and gated[0]['group_gate'],'tie_cause':('group_gate' if method=='harmonic' and gated[0]['group_gate'] else 'equal_final_score') if results[0]==results[1] else None})
        raw_order.append({'id':r['id'],'expected':r['expected'],'raw_a':entry['raw']['a'],'raw_b':entry['raw']['b'],'records_before_gate':base,'records_after_gate':gated})
    # Full serialized conversation length, excluding invalid calls entirely.
    from experiments.qwen3_bonus.perm_rewards import reward_messages
    def token_lengths(r, ch):
        text=model.tokenizer.apply_chat_template(reward_messages(r['prompt'],r[ch]),tokenize=False,add_generation_prompt=False)
        total=len(model.tokenizer(text,add_special_tokens=False)['input_ids'])
        response=len(model.tokenizer(r[ch],add_special_tokens=False)['input_ids'])
        return {'full_input_tokens':total,'response_tokens':response,'prompt_and_template_tokens':total-response}
    ranked=[(token_lengths(r,ch),r,ch) for r in pairs for ch in ('a','b') if not invalid_reason(r[ch])]
    longest=sorted(ranked,key=lambda x:x[0]['full_input_tokens'],reverse=True)[:8];long_times=[]
    for lengths,r,ch in longest:
        start=time.perf_counter();model(r['prompt'],r[ch])
        long_times.append({'id':r['id'],'choice':ch,'seconds':time.perf_counter()-start,**lengths})
    import platform, importlib.metadata
    environment={'platform':platform.platform(),'python':sys.version,'machine':platform.machine(),
                 'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
                 'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers')}}
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    report={'schema_version':1,'reward_spec':REWARD_SPEC,**norm,'gate':final_gate(methods['harmonic'],subset),'methods':{k:{'gate':final_gate(v,subset),'pairs':v} for k,v in methods.items()},'raw_ordering':raw_order,'model':identity,'input_format':INPUT_FORMAT,'input_format_sha256':fingerprint(INPUT_FORMAT),'data_manifest_path':str(manifest_path.resolve()),'data_manifest_sha256':sha256(manifest_path),'labels_sha256':fingerprint([{k:r[k] for k in ('id','expected','reviewed','reviewer','review_type')} for r in pairs]),'code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in CODE_DEPENDENCIES},'benchmark':{'peak_process_rss_bytes':rss,'torch_threads':torch.get_num_threads(),'timings':timings,'eight_longest':long_times,'seconds_per_response_conservative':max(t['seconds'] for t in long_times),'whole_system_eligible':False},'environment':environment,'paid_api_calls':0,'review_note':'Inherited user-authorized model review; not human preference validation.'}
    immutable_write(output_path(output),_json(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('score');c.add_argument('--data',type=Path,required=True);c.add_argument('--model',type=Path,required=True);c.add_argument('--output',type=Path,required=True);c.add_argument('--execute',action='store_true')
    s=sub.add_parser('select');s.add_argument('--candidates',type=Path,required=True);s.add_argument('--resources',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    m=sub.add_parser('make-candidate');m.add_argument('--calibration',type=Path,required=True);m.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='make-candidate':
        result=make_candidate(a.calibration);immutable_write(output_path(a.output),_json(result));print(json.dumps(result,indent=2));raise SystemExit(0)
    result=calibrate(a.data,a.model,a.output,execute=a.execute) if a.action=='score' else select_model(json.loads(a.candidates.read_text()),json.loads(a.resources.read_text()))
    if a.action=='select':immutable_write(output_path(a.output),_json(result))
    print(json.dumps(result,ensure_ascii=False,indent=2))
