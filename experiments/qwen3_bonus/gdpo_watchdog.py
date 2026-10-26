"""Independent GDPO supervisor that owns its training process and rental release.

A caller can disappear without disabling this detached supervisor. It reads no
metric log, never accepts an arbitrary PID, and never invokes a shell.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from experiments.qwen3_bonus.common import ROOT, output_path, write_json
from experiments.qwen3_bonus.gdpo_runtime import AtomicCheckpointStore, GDPOBudget

_SHELLS = {'sh','bash','zsh','dash','fish','ksh','csh','tcsh','cmd','cmd.exe','powershell','powershell.exe','pwsh','pwsh.exe'}


def _argv(value) -> list[str]:
    if not isinstance(value,list) or not value or any(not isinstance(arg,str) or not arg or '\x00' in arg for arg in value):
        raise ValueError('An explicit nonempty structured argv list is required')
    if not Path(value[0]).is_absolute():
        raise ValueError('Executable must have an explicit absolute path')
    if Path(value[0]).name.lower() in _SHELLS or Path(value[0]).resolve().name.lower() in _SHELLS:
        raise ValueError('Commands must execute directly without shell interpolation')
    return list(value)


def validate_shutdown_argv(value) -> list[str]:
    argv = _argv(value)
    executable = Path(argv[0])
    if executable.name.lower() in _SHELLS or executable.resolve().name.lower() in _SHELLS:
        raise ValueError('Platform shutdown must execute directly without a shell')
    if any(arg.lower() in ('-c','/c','-command','-encodedcommand') for arg in argv[1:]):
        raise ValueError('Inline code or shell command strings are forbidden for shutdown')
    if not executable.is_file() or not os.access(executable,os.X_OK):
        raise ValueError('Shutdown executable is missing or not executable')
    return argv


def setup_complete(run: Path, budget: GDPOBudget) -> bool:
    """Only a durable completed boundary within the initial two-yuan cap qualifies."""
    try:
        progress = json.loads((run/'progress.json').read_text())
        step, when = progress['completed_step'], progress['first_two_completed_at']
        return (isinstance(step,int) and step >= 2 and isinstance(when,(int,float))
                and budget.started_at <= when <= budget.setup_deadline() and budget.cost(when) <= 2)
    except (OSError,ValueError,KeyError,TypeError):
        return False


def launch_watchdog(run: Path, training_argv: list[str], budget: GDPOBudget,
                    shutdown_argv: list[str] | None, *, save_seconds: float = 120,
                    term_grace_seconds: float = 10, kill_grace_seconds: float = 5,
                    poll_seconds: float = 1,
                    allow_preserve_allocation: bool = False) -> subprocess.Popen:
    """Start a detached supervisor; it, rather than this caller, launches training."""
    run = output_path(run)
    if shutdown_argv is None and budget.rate > 0 and not allow_preserve_allocation:
        raise ValueError('A paid training run requires an explicit validated shutdown command')
    shutdown = None if shutdown_argv is None else validate_shutdown_argv(shutdown_argv)
    training = _argv(training_argv)
    now = time.time()
    status = budget.status(now,0,0,save_seconds)
    if status['should_stop'] or now >= budget.setup_deadline():
        raise ValueError('Rental budget/setup allowance is exhausted before launch')
    config = {'run':str(run),'training_argv':training,'shutdown_argv':shutdown,
              'budget':asdict(budget),'save_seconds':save_seconds,
              'term_grace_seconds':term_grace_seconds,'kill_grace_seconds':kill_grace_seconds,
              'poll_seconds':poll_seconds}
    _schedule(config,budget)
    run.mkdir(parents=True,exist_ok=True)
    if (run/'watchdog.json').exists():
        raise ValueError('Run already has a watchdog configuration; use a new run directory')
    write_json(run/'watchdog.json',config)
    with (run/'watchdog.log').open('ab') as log:
        return subprocess.Popen([sys.executable,'-m','experiments.qwen3_bonus.gdpo_watchdog',
                                 '--config',str(run/'watchdog.json')],
                                cwd=ROOT.parents[1],stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)


def _schedule(config: dict, budget: GDPOBudget) -> tuple[float,float,float,float,float,float]:
    save = config.get('save_seconds',120)
    training = config.get('training_deadline',budget.training_deadline(save))
    release = config.get('release_deadline',budget.release_deadline())
    term = config.get('term_grace_seconds',10)
    kill = config.get('kill_grace_seconds',5)
    poll = config.get('poll_seconds',1)
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in (save,training,release,term,kill,poll)):
        raise ValueError('Watchdog schedule must be finite')
    if save < 120 or term <= 0 or kill <= 0 or not 0 < poll <= 5:
        raise ValueError('Invalid watchdog save, termination or polling interval')
    if training > budget.training_deadline(save) or release > budget.release_deadline() or training >= release:
        raise ValueError('Watchdog deadlines may only shorten the budget')
    return save,training,release,term,kill,poll



# Matches pinned verl's one-rank full-state contract; keep independent of ML imports.
_RETRY_REQUIRED_STATE = [
    'actor/model_world_size_1_rank_0.pt', 'actor/optim_world_size_1_rank_0.pt',
    'actor/extra_state_world_size_1_rank_0.pt', 'data.pt', 'driver_rng.pt',
    'actor/lora_adapter/adapter_model.safetensors', 'actor/lora_adapter/adapter_config.json',
]


def _attempt_has_oom(path: Path, start: int) -> bool:
    needles = (b'CUDA out of memory', b'torch.OutOfMemoryError')
    try:
        with path.open('rb') as stream:
            stream.seek(start)
            tail = b''
            while chunk := stream.read(65536):
                combined = tail+chunk
                if any(needle in combined for needle in needles):
                    return True
                tail = combined[-64:]
    except (OSError, ValueError):
        pass
    return False


def _retry_resources(argv: list[str]) -> tuple[list[str], float, float]:
    remaining = []
    values = {}
    index = 0
    while index < len(argv):
        arg = argv[index]
        key = arg.split('=',1)[0]
        if key in ('--gpu-memory','--max-num-seqs'):
            if key in values:
                raise ValueError('Repeated resource flags are ambiguous')
            if '=' in arg:
                value = arg.split('=',1)[1]
            else:
                index += 1
                value = argv[index]
            values[key] = value
        else:
            remaining.append(arg)
        index += 1
    original = float(values.get('--gpu-memory','.45'))
    lowered = round(original-.05,10)
    if not math.isfinite(original) or not .3 <= lowered < original <= .65:
        raise ValueError('No validated lower GPU memory fraction remains')
    return remaining+['--gpu-memory',format(lowered,'.10g'),'--max-num-seqs','1'], original, lowered


def prepare_oom_retry(run: Path, budget: GDPOBudget, training_argv: list[str], *,
                      retry_count: int, now: float, log_start: int = 0) -> dict | None:
    """Fail closed unless an exited OOM attempt can restart without replaying updates.

    Only the owning supervisor calls this after observing its child exit. This
    helper records the single approved retry; it never starts or kills a process.
    """
    run = output_path(run)
    if retry_count != 0 or (run/'STOP_REQUESTED.json').exists():
        return None
    if not _attempt_has_oom(run/'training.log',log_start):
        return None
    try:
        argv, original, lowered = _retry_resources(_argv(training_argv))
        launch = json.loads((run/'launch.json').read_text())
        if launch.get('retry_count',0) != 0 or GDPOBudget(**launch['budget']) != budget:
            return None
        identity = launch['identity']
        if not isinstance(identity,dict) or not identity or type(launch['seed']) is not int:
            return None
        progress_path = run/'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'completed_step':0}
        step = progress['completed_step']
        if type(step) is not int or step < 0 or progress.get('identity',identity) != identity:
            return None
        if step >= 2 and not setup_complete(run,budget):
            return None
        longest = float(progress.get('budget',{}).get('longest_step_seconds',0))
        status = budget.status(now,step,longest)
        if status['should_stop']:
            return None
        pointer = run/'checkpoints/latest.json'
        if step == 0:
            # An existing completed checkpoint contradicts a claimed fresh run.
            if pointer.exists() or launch.get('resume'):
                return None
        else:
            if not pointer.is_file():
                return None
            entry = AtomicCheckpointStore(run/'checkpoints').resume(identity,_RETRY_REQUIRED_STATE)
            if entry is None or entry['step'] != step:
                return None
        retry = {'retry_count':1,'completed_step':step,'resume':step > 0,
                 'training_argv':argv,'original_gpu_memory':original,
                 'gpu_memory':lowered,'max_num_seqs':1,'requested_at':now,
                 'longest_step_seconds':longest}
        launch.update({'resume':step > 0,'retry_count':1,'oom_retry':retry})
        write_json(run/'launch.json',launch)
        write_json(run/'oom_retry.json',retry)
        return retry
    except (OSError,ValueError,KeyError,TypeError,IndexError,OverflowError):
        return None


def run_watchdog(config: dict) -> dict:
    run = output_path(config['run'])
    shutdown = None if config.get('shutdown_argv') is None else validate_shutdown_argv(config['shutdown_argv'])
    training = _argv(config.get('training_argv'))
    budget = GDPOBudget(**config['budget'])
    save, training_deadline, release_deadline, term_grace, kill_grace, poll = _schedule(config,budget)
    run.mkdir(parents=True,exist_ok=True)
    result = {'terminated':False,'killed':False,'launch_error':None,'training_returncode':None,
              'shutdown_returncode':None,'shutdown_error':None,'retry_count':0,'attempts':[]}
    process = None
    soft_at = None
    attempt_log_start = (run/'training.log').stat().st_size if (run/'training.log').exists() else 0
    try:
        with (run/'training.log').open('ab') as log:
            process = subprocess.Popen(training,cwd=ROOT.parents[1],stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        write_json(run/'watchdog_started.json',{'pid':os.getpid(),'owned_child_pid':process.pid,
                   'training_deadline':training_deadline,'release_deadline':release_deadline,
                   'billing_started_at':budget.started_at,'rate':budget.rate,'quantum_seconds':budget.quantum_seconds})
        while True:
            now = time.time()
            if process.poll() is not None:
                result['training_returncode'] = process.returncode
                result['attempts'].append({'pid':process.pid,'returncode':process.returncode})
                retry = None
                if process.returncode != 0 and soft_at is None and now < min(training_deadline,release_deadline):
                    retry = prepare_oom_retry(run,budget,training,retry_count=result['retry_count'],
                                              now=now,log_start=attempt_log_start)
                if retry is None:
                    break
                # Full checkpoint hashing can consume the remaining allowance.
                # Recheck the actual clock immediately before replacing a child.
                fresh = time.time()
                if fresh >= min(training_deadline,release_deadline) or budget.status(
                        fresh,retry['completed_step'],retry['longest_step_seconds'])['should_stop']:
                    result['retry_cancelled'] = 'budget_expired_during_verification'
                    break
                # This group was created by this supervisor, never supplied as a
                # PID from disk. Clear remaining workers before another attempt.
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                result['retry_count'] = 1
                training = retry['training_argv']
                with (run/'training.log').open('ab') as log:
                    log.write(b'\n=== GDPO OOM retry 1: gpu-memory -0.05; max-num-seqs 1 ===\n')
                    log.flush()
                    attempt_log_start = log.tell()
                    process = subprocess.Popen(training,cwd=ROOT.parents[1],stdin=subprocess.DEVNULL,
                                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                write_json(run/'watchdog_retry_started.json',{'owned_child_pid':process.pid,**retry})
                continue
            setup_ok = setup_complete(run,budget)
            stop_deadline = training_deadline if setup_ok else min(training_deadline,budget.setup_deadline())
            # Setup cap stops computation at two yuan; the reserved saving window
            # still permits a graceful checkpoint before instance release.
            hard_release = release_deadline if setup_ok else min(release_deadline,budget.setup_deadline()+save)
            if soft_at is None and now >= stop_deadline:
                soft_at = now
                write_json(run/'STOP_REQUESTED.json',{'requested_at':now,'reason':'budget_deadline' if setup_ok else 'setup_cap',
                                                      'release_deadline':hard_release})
            term_at = min((soft_at+save) if soft_at is not None else hard_release,hard_release-kill_grace-term_grace)
            kill_at = min(term_at+term_grace,hard_release-kill_grace)
            if soft_at is not None and not result['terminated'] and now >= term_at:
                # process is our own unreaped session leader: its PID cannot be
                # recycled, and its process group was established by Popen.
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
                result['terminated'] = True
            if result['terminated'] and not result['killed'] and now >= kill_at:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                result['killed'] = True
            if now >= hard_release:
                break
            time.sleep(min(poll,max(.001,hard_release-now)))
    except Exception as error:
        result['launch_error'] = f'{type(error).__name__}: {error}'
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                    result['killed'] = True
                except ProcessLookupError:
                    pass
            try:
                result['training_returncode'] = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        result['shutdown_started_at'] = time.time()
        result['instance_preserved'] = shutdown is None
        if shutdown is not None:
            try:
                with (run/'shutdown.log').open('ab') as log:
                    completed = subprocess.run(shutdown,stdin=subprocess.DEVNULL,stdout=log,
                                               stderr=subprocess.STDOUT,timeout=30,check=False,close_fds=True)
                    result['shutdown_returncode'] = completed.returncode
            except (OSError,subprocess.SubprocessError) as error:
                result['shutdown_error'] = f'{type(error).__name__}: {error}'
        result['finished_at'] = time.time()
        result['accumulated_cost'] = budget.cost(max(budget.started_at,result['finished_at']))
        write_json(run/'watchdog_result.json',result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    result = run_watchdog(json.loads(output_path(args.config).read_text()))
    return 0 if result['instance_preserved'] or result['shutdown_returncode'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
