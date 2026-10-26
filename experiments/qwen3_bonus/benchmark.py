"""Measure one local SSE request, first text, retokenized speed, RSS and system swap."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from experiments.qwen3_bonus.common import ROOT,output_path,write_json


def content_from_sse(line: str) -> str:
    if not line.startswith('data:'):return ''
    value=line[5:].strip()
    if value=='[DONE]':return ''
    data=json.loads(value)
    if data.get('error'):raise RuntimeError('Local SSE reported an error')
    choices=data.get('choices',[])
    if not choices:return ''
    return choices[0].get('delta',{}).get('content') or ''


def swap_usage() -> str | None:
    try:
        return subprocess.check_output(['sysctl','-n','vm.swapusage'],text=True,timeout=2).strip()
    except (OSError,subprocess.SubprocessError):return None


def process_group_rss_mib(port: int) -> float | None:
    try:
        listeners=subprocess.check_output(['lsof','-nP',f'-iTCP:{port}','-sTCP:LISTEN','-t'],text=True,timeout=2)
        groups={os.getpgid(int(pid)) for pid in listeners.split()}
        table=subprocess.check_output(['ps','-axo','pgid=,rss='],text=True,timeout=2)
        values=[int(rss) for line in table.splitlines() if len(parts:=line.split())==2
                for group,rss in [parts] if int(group) in groups]
        return sum(values)/1024 if values else None
    except (OSError,ValueError,subprocess.SubprocessError):return None


def measure(config,tokenizer,model: str,prompt: str) -> dict:
    import httpx
    started=time.perf_counter();first=None;parts=[];memory=[];finished=threading.Event()
    def sample():
        while not finished.is_set():
            value=process_group_rss_mib(config.engine_port)
            if value is not None:memory.append(value)
            finished.wait(.5)
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    report={'model':model,'swap_before':swap_usage(),'success':False,'genuine_inference':False,
            'memory_scope':'0.5s sampled sum of engine process-group RSS, shared pages may be counted twice; not GPU peak.'}
    try:
        headers={'Authorization':f'Bearer {config.local_token}'} if config.local_token else {}
        with httpx.Client(timeout=config.request_timeout_seconds,trust_env=False,follow_redirects=False) as client:
            with client.stream('POST',f'http://127.0.0.1:{config.gateway_port}/v1/chat/completions',
                headers=headers,json={'model':model,'messages':[{'role':'user','content':prompt}],
                    'max_tokens':16 if model=='solace-emotion' else 256,'temperature':0,'stream':True}) as response:
                report['http_status']=response.status_code
                response.raise_for_status()
                for line in response.iter_lines():
                    content=content_from_sse(line)
                    if content:
                        if first is None:first=time.perf_counter()-started
                        parts.append(content)
        elapsed=time.perf_counter()-started;answer=''.join(parts)
        tokens=len(tokenizer.encode(answer,add_special_tokens=False))
        report.update(success=bool(answer),genuine_inference=bool(answer),answer=answer,
            first_text_seconds=first,total_seconds=elapsed,retokenized_output_tokens=tokens,
            approximate_decode_tokens_per_second=((tokens-1)/(elapsed-first) if first is not None and elapsed>first else None))
    except Exception as exc:
        report.update(error_type=type(exc).__name__,total_seconds=time.perf_counter()-started)
    finally:
        finished.set();sampler.join(timeout=5)
        report.update(sampled_peak_process_group_rss_mib=max(memory) if memory else None,
                      swap_after=swap_usage())
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--tokenizer',type=Path,default=ROOT/'artifacts/models/base')
    parser.add_argument('--model',choices=['solace-emotion','solace-response'],default='solace-response')
    parser.add_argument('--prompt',default='明天考试，我很紧张，想先和你聊聊。')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();output=output_path(args.output)
    if output.exists():raise FileExistsError('Choose a fresh benchmark report')
    from transformers import AutoTokenizer
    from experiments.qwen3_bonus.serving import ServingConfig
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    result=measure(ServingConfig.from_file(args.config),tokenizer,args.model,args.prompt)
    write_json(output,result)
    if not result['success']:raise SystemExit(2)


if __name__=='__main__':main()
