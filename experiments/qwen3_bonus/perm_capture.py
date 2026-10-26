"""Local Metal captures for the approved frozen PERM dev/test protocol."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time

from experiments.qwen3_bonus.common import ROOT, output_path, sha256, write_json, read_jsonl
from experiments.qwen3_bonus.gdpo_evaluation import checkpoint_digest
from experiments.qwen3_bonus.models import SOURCES


def validate_existing(rows, cases, adapters, identity):
    indexed={r['id']:r for r in cases};seen=set()
    for row in rows:
        key=(row['id'],row['variant'])
        if (key in seen or row['id'] not in indexed or row['variant'] not in adapters
            or row.get('prompt') != indexed[row['id']]['prompt']
            or row.get('adapter_sha256') != adapters[row['variant']]
            or any(row.get(k)!=v for k,v in identity.items())
            or row.get('temperature')!=0 or row.get('max_tokens')!=256 or row.get('quantization')!='4bit'
            or not isinstance(row.get('response'),str)):
            raise ValueError('Existing capture differs from frozen prompt/model/settings or repeats an item')
        seen.add(key)
    return seen


def select_jobs(split, run, selection):
    if split=='test':
        if selection is None:raise ValueError('Frozen dev selection required before test inference')
        from experiments.qwen3_bonus.perm_evaluation import verify_selection
        document=verify_selection(selection)
        selected=Path(document['selected']['checkpoint'])
        return {'perm_grpo':selected.parent}
    return {f'perm_{int(p.name.split("-")[-1])}':p for p in sorted((Path(run)/'adapter_snapshots').iterdir()) if p.is_dir()}


def capture(split, destination, selection=None):
    import httpx
    from experiments.qwen3_bonus.serving import ServingConfig, validate_artifacts, child_environment, port_is_free
    exp=ROOT/'artifacts/perm-grpo-20261024'
    run=exp/'runtime-18303-final/run';data=exp/'data'
    destination=output_path(Path(destination));destination.mkdir(parents=True,exist_ok=True)
    config=ServingConfig.from_file(ROOT/'configs/serving.local.json')
    validate_artifacts(config)
    cases=read_jsonl(data/f'response_{split}.jsonl')
    manifest=json.loads((data/'dataset_manifest.json').read_text())
    if sha256(data/f'response_{split}.jsonl')!=manifest['files'][f'response_{split}.jsonl']['sha256']:
        raise ValueError('Frozen split changed')
    if len(cases)!=(32 if split=='dev' else 80):raise ValueError('Unexpected split count')
    models={'base':None,'grpo32':ROOT/'artifacts/adapters/solace-response-grpo32',
            'current':ROOT/'artifacts/adapters/solace-response',**select_jobs(split,run,selection)}
    adapter_hashes={name:sha256(path/'adapter_model.safetensors') if path else None for name,path in models.items()}
    # Bind exact template and converted base, alongside runtime/file provenance.
    base=Path(config.model_path)
    template=base/'chat_template.jinja'
    if template.is_file():template_hash=sha256(template)
    else:
        from experiments.qwen3_bonus.gdpo_evaluation import digest_json
        template_hash=digest_json(json.loads((base/'tokenizer_config.json').read_text())['chat_template'])
    identity={'base_revision':SOURCES['base']['revision'],'converted_base_sha256':checkpoint_digest(base),
              'chat_template_sha256':template_hash}
    provenance={'split':split,'data_sha256':sha256(data/f'response_{split}.jsonl'),
        **identity,'adapters':{name:None if path is None else {
            'path':str(path.resolve()),'weights_sha256':adapter_hashes[name],
            'config_sha256':sha256(path/'adapter_config.json')} for name,path in models.items()},
        'temperature':0,'max_tokens':256,'quantization':'4bit',
        'versions':{p:importlib.metadata.version(p) for p in ['vllm','vllm-metal']},
        'selection_sha256':sha256(selection) if selection else None,
        'paid_api_calls':0,'source_sha256':sha256(Path(__file__))}
    prov=destination/'provenance.json'
    if prov.exists():
        if json.loads(prov.read_text())!=provenance:raise ValueError('Capture provenance changed')
    else:write_json(prov,provenance)
    path=destination/'captures.jsonl'
    rows=read_jsonl(path) if path.exists() else []
    done=validate_existing(rows,cases,adapter_hashes,identity)
    total=len(cases)*len(models)
    if len(done)==total:
        print('Captures already complete',flush=True);return
    port=18184
    if not port_is_free(port):raise ValueError('Evaluation engine port is occupied')
    # One quantized base, resident named LoRAs; never merge or train weights.
    argv=[config.engine_executable,'serve',str(base),'--host','127.0.0.1','--port',str(port),
          '--enable-lora','--lora-modules',*[f'{name}={folder}' for name,folder in models.items() if folder],
          '--max-loras','16','--max-cpu-loras','16','--max-lora-rank','8',
          '--max-model-len','2048','--max-num-seqs','1','--max-num-batched-tokens','512',
          '--gpu-memory-utilization','0.45','--no-enable-log-requests']
    token=secrets.token_hex(32);env=child_environment();env['VLLM_API_KEY']=token
    headers={'Authorization':'Bearer '+token};origin=f'http://127.0.0.1:{port}'
    process=None;started=time.monotonic();hard_deadline=started+5400
    try:
        with (destination/'engine.log').open('ab') as log:
            process=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(destination/'process.json',{'pid':process.pid,'started_epoch':time.time(),'phase':'loading','expected_captures':total})
        with httpx.Client(timeout=90,trust_env=False,follow_redirects=False) as client:
            deadline=time.monotonic()+180
            while True:
                if process.poll() is not None:raise RuntimeError('Evaluation engine exited; inspect engine.log')
                try:
                    response=client.get(origin+'/v1/models',headers=headers,timeout=2)
                    names={r['id'] for r in response.json()['data']}
                    if {str(base),*list(models.keys())[1:]}<=names:break
                except (httpx.HTTPError,ValueError,KeyError):pass
                if time.monotonic()>deadline:raise TimeoutError('Engine startup deadline')
                time.sleep(.5)
            with path.open('a',encoding='utf-8',buffering=1) as out:
                for case in cases:
                    for name,folder in models.items():
                        if (case['id'],name) in done:continue
                        if time.monotonic()>hard_deadline:raise TimeoutError('Capture hard deadline')
                        requested=name if folder else str(base);begin=time.monotonic()
                        response=client.post(origin+'/v1/chat/completions',headers=headers,json={
                            'model':requested,'messages':case['prompt'],'temperature':0,'max_tokens':256,'stream':False})
                        response.raise_for_status();result=response.json()
                        if result.get('model')!=requested:raise ValueError('Actual model identity mismatch')
                        answer=result['choices'][0]
                        row={'id':case['id'],'variant':name,'prompt':case['prompt'],'response':answer['message']['content'],
                             **identity,'adapter_sha256':adapter_hashes[name],'temperature':0,'max_tokens':256,'quantization':'4bit',
                             'finish_reason':answer.get('finish_reason'),'usage':result.get('usage'),
                             'elapsed_seconds':time.monotonic()-begin,'requested_model':requested,'reported_model':result['model']}
                        out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
                        done.add((case['id'],name))
                        write_json(destination/'progress.json',{'captured':len(done),'expected':total,'last_case':case['id'],
                            'last_variant':name,'elapsed_seconds':time.monotonic()-started,'process_pid':process.pid})
                    print(f'{split}: {len(done)}/{total} captures',flush=True)
        # Recheck all LoRA bytes and templates after real generation.
        if any(sha256(p/'adapter_model.safetensors')!=adapter_hashes[n] for n,p in models.items() if p):
            raise ValueError('Adapter changed during capture')
        rows=read_jsonl(path);validate_existing(rows,cases,adapter_hashes,identity)
        if len(rows)!=total:raise ValueError('Incomplete capture coverage')
        write_json(destination/'complete.json',{'count':total,'elapsed_seconds':time.monotonic()-started,
            'captures_sha256':sha256(path),'provenance_sha256':sha256(prov),'genuine_inference':True})
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split',choices=['dev','test']);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    args=parser.parse_args();capture(args.split,args.output,args.selection)


if __name__=='__main__':main()
