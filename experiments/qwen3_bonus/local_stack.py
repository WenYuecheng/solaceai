"""Start/stop this local deployment without changing the original app entrypoint."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.qwen3_bonus.common import ROOT,write_json

PROJECT=ROOT.parents[1]
STATE=ROOT/'artifacts/local-deployment'
PIDFILE=STATE/'stack.json'


def owned(pid):
    try:
        args=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True,timeout=2)
        return 'experiments.qwen3_bonus.local_stack run' in args
    except (OSError,subprocess.SubprocessError):
        return False


def run():
    from experiments.qwen3_bonus.asgi import selected_config
    from experiments.qwen3_bonus.local_app import validate_candidate
    config=selected_config()
    validate_candidate(STATE/'acceptance.json',config)
    from experiments.qwen3_bonus.serving import port_is_free
    if not all(port_is_free(p) for p in (config.gateway_port,config.engine_port,8082)):
        raise RuntimeError('A local deployment port is already occupied; existing process left intact')
    children=[]
    def shutdown(signum,frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    try:
        for name,args in (
            ('gateway',['-m','experiments.qwen3_bonus.serving','--config',str(ROOT/'configs/serving.local.json')]),
            ('application',['-m','uvicorn','experiments.qwen3_bonus.local_app:create_app','--factory',
                            '--host','127.0.0.1','--port','8082','--workers','1']),
        ):
            fd=os.open(STATE/f'{name}.log',os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
            with os.fdopen(fd,'ab') as log:
                children.append(subprocess.Popen([sys.executable,*args],cwd=PROJECT,env=os.environ.copy(),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True))
        write_json(PIDFILE,{'pid':os.getpid(),'gateway_pid':children[0].pid,'application_pid':children[1].pid,
                            'started_unix':time.time(),'app_url':'http://127.0.0.1:8082',
                            'gateway_url':f'http://127.0.0.1:{config.gateway_port}'})
        while all(p.poll() is None for p in children):
            time.sleep(1)
    finally:
        for process in reversed(children):
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        if PIDFILE.exists() and json.loads(PIDFILE.read_text()).get('pid')==os.getpid():PIDFILE.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','stop','status','run'])
    args=parser.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    # Trusted local config only; keys remain in existing app configuration.
    os.environ['BONUS_SERVING_CONFIG']=str(ROOT/'configs/serving.local.json')
    data=json.loads(PIDFILE.read_text()) if PIDFILE.exists() else {}
    running=bool(data.get('pid') and owned(data['pid']))
    if args.action=='status':print(json.dumps({'running':running,**(data if running else {})}));return
    if args.action=='stop':
        if running:os.kill(data['pid'],signal.SIGTERM)
        print('Stop requested' if running else 'Already stopped');return
    if args.action=='run':run();return
    if running:print('Already running: http://127.0.0.1:8082');return
    from experiments.qwen3_bonus.local_app import validate_candidate
    from experiments.qwen3_bonus.asgi import selected_config
    validate_candidate(STATE/'acceptance.json',selected_config())
    fd=os.open(STATE/'supervisor.log',os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
    with os.fdopen(fd,'ab') as log:
        p=subprocess.Popen([sys.executable,'-m','experiments.qwen3_bonus.local_stack','run'],
                           cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    for _ in range(50):
        if p.poll() is not None:raise RuntimeError('Local startup failed; see supervisor.log')
        if PIDFILE.exists():print('Started: http://127.0.0.1:8082');return
        time.sleep(.1)
    raise RuntimeError('Startup did not confirm; inspect local-deployment logs')


if __name__=='__main__':main()
