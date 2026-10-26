"""Launch one native-GRPO run only after frozen evidence and rental gates pass."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from experiments.qwen3_bonus.common import ROOT, output_path, read_jsonl, sha256, write_json
from experiments.qwen3_bonus.gdpo_runtime import GDPOBudget
from experiments.qwen3_bonus.models import SOURCES, verify_model as verify_base


def training_budget(*, rate, started_at, quantum_seconds, preparation_cost):
    if (type(preparation_cost) not in (float, int) or not math.isfinite(preparation_cost)
            or not 0 <= preparation_cost < 16):
        raise ValueError('Preparation cost must leave setup, training and save budget')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (rate, started_at, quantum_seconds)):
        raise ValueError('Finite numeric billing values required')
    return GDPOBudget(rate, started_at, quantum_seconds, cap_yuan=20-preparation_cost, reserve_yuan=2)


def base_content_identity(base, *, allow_missing_raw_weights=False):
    """Portable identity of verified, inventoried loader inputs, excluding documentation."""
    from experiments.qwen3_bonus.perm_calibration import fingerprint
    base = Path(base)
    if allow_missing_raw_weights:
        from experiments.qwen3_bonus.gdpo_preflight import verify_raw_base
        source, _ = verify_raw_base(base, allow_missing_raw_weights=True)
    else:
        source = verify_base(base, 'base')
    def relevant(name):
        p = Path(name)
        return (not p.name.endswith('.safetensors.ranges.json') and
                (p.suffix in ('.safetensors', '.bin', '.json', '.jinja', '.model') or p.name in ('merges.txt','vocab.txt')))
    files = {n: h for n,h in source['files'].items() if relevant(n)}
    for p in base.rglob('*'):
        if p.is_file() and '.cache' not in p.parts and p.name != 'bonus_source.json':
            name = str(p.relative_to(base))
            if relevant(name) and name not in files:
                raise ValueError('Uninventoried loader input: '+name)
    return fingerprint({'repo_id':source['repo_id'], 'revision':source['revision'], 'files':files})


def live_resources(resources, budget, now):
    return {**resources, 'remaining_training_yuan':max(0., budget.cap_yuan-budget.cost(now)),
            'hourly_yuan':budget.rate, 'prep_setup_save_reserve_yuan':budget.reserve_yuan}


def validate_preflight(base, preflight):
    if (preflight.get('genuine_inference') is not True or preflight.get('compatibility_passed') is not True
            or preflight.get('base_content_sha256') != base_content_identity(base)):
        raise ValueError('Matching actual old/new-format inference preflight required')


def cgroup_memory_limit(root=Path('/sys/fs/cgroup')):
    """Read current and ancestor cgroup limits; unlimited is not a byte count."""
    root = Path(root)
    directories = [root]
    if root == Path('/sys/fs/cgroup') and Path('/proc/self/cgroup').is_file():
        for row in Path('/proc/self/cgroup').read_text().splitlines():
            _, controllers, relative = row.split(':', 2)
            if controllers in ('', 'memory') or 'memory' in controllers.split(','):
                target = root/relative.lstrip('/')
                if not target.resolve().is_relative_to(root.resolve()):
                    continue
                directories.extend([target, *[p for p in target.parents if p.is_relative_to(root)]])
    limits = []
    for directory in directories:
        for name in ('memory.max', 'memory.limit_in_bytes'):
            path = directory/name
            if path.is_file():
                text = path.read_text().strip()
                if text != 'max':
                    value = int(text)
                    if value <= 0:
                        raise ValueError('Invalid effective cgroup memory limit')
                    if value < 2**60:
                        limits.append(value)
    return min(limits) if limits else None


def verify_resource_evidence(selection):
    from experiments.qwen3_bonus.perm_calibration import select_model
    resources, candidates = selection.get('resources'), selection.get('candidates')
    if not isinstance(resources, dict) or not isinstance(candidates, list):
        raise ValueError('Selection must preserve measured resource/candidate evidence')
    recalculated = select_model(candidates, resources)
    if recalculated != selection:
        raise ValueError('Selection evidence differs from frozen measured inputs')
    return resources, candidates


def validate_selection(selection):
    from experiments.qwen3_bonus.perm_calibration import fingerprint
    from experiments.qwen3_bonus.perm_models import CANDIDATES
    body = {k: v for k, v in selection.items() if k != 'selection_sha256'}
    if selection.get('passed') is not True or selection.get('selection_sha256') != fingerprint(body):
        raise ValueError('Frozen reward selection is missing, failed or changed')
    selected = selection.get('selected')
    if not isinstance(selected, dict) or not any(all(selected.get(k) == v for k, v in row.items())
                                               for row in CANDIDATES.values()):
        raise ValueError('Unapproved reward model identity')
    for key in ('model_inventory_sha256', 'calibration_sha256'):
        digest = selected.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Selection requires model and calibration fingerprints')
    if not selected.get('model_path') or not selected.get('calibration_path'):
        raise ValueError('Selection requires explicit local artifact paths')
    return selected


def verify_data(path):
    manifest = json.loads((path/'dataset_manifest.json').read_text())
    if manifest.get('frozen') is not True or manifest.get('train_no_applicable_checks') != 192:
        raise ValueError('Frozen reused PERM data required')
    for name, entry in manifest['files'].items():
        target = (path/name).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file() or sha256(target) != entry['sha256']:
            raise ValueError('Frozen data fingerprint changed')
    for split, count in [('train', 256), ('dev', 32), ('test', 80)]:
        rows = read_jsonl(path/f'response_{split}.jsonl')
        if len(rows) != count or len({r['id'] for r in rows}) != count:
            raise ValueError('Frozen split count mismatch')
    return manifest


def validate_assets(base, data, selection, preflight):
    from experiments.qwen3_bonus.perm_models import verify_model
    from experiments.qwen3_bonus.perm_rewards import load_calibration
    verify_base(base, 'base')
    manifest = verify_data(data)
    selected = validate_selection(selection)
    reward = verify_model(Path(selected['model_path']))
    if any(reward.get(k) != selected[k] for k in ('repo_id', 'revision', 'model_inventory_sha256')):
        raise ValueError('Selected reward inventory differs from actual model')
    calibration_path = Path(selected['calibration_path'])
    if sha256(calibration_path) != selected['calibration_sha256']:
        raise ValueError('Selected calibration changed')
    calibration = load_calibration(str(calibration_path))
    if (calibration['model']['model_inventory_sha256'] != reward['model_inventory_sha256']
            or calibration['data_manifest_sha256'] != sha256(data/'dataset_manifest.json')):
        raise ValueError('Calibration binds a different reward model or data freeze')
    validate_preflight(base, preflight)
    return selected, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = ROOT/'artifacts/perm-grpo-20261024'
    parser.add_argument('--base', type=Path, default=ROOT/'artifacts/models/base')
    parser.add_argument('--data', type=Path, default=root/'data')
    parser.add_argument('--selection', type=Path, help='Optional strict preselected mode; otherwise prepare on target before fit')
    parser.add_argument('--reward-candidate', type=Path, action='append', help='Repeat pinned local candidate paths, preferred 1.7B first')
    parser.add_argument('--preflight', type=Path, default=root/'preflight/report.json')
    parser.add_argument('--run', type=Path, default=root/'run')
    parser.add_argument('--hourly-rate', type=float, required=True)
    parser.add_argument('--billing-quantum-seconds', type=int, required=True)
    parser.add_argument('--rental-started-at', required=True)
    parser.add_argument('--preparation-cost-yuan', type=float, required=True)
    parser.add_argument('--shutdown-script-sha256', required=True)
    parser.add_argument('--max-steps', type=int, default=64)
    parser.add_argument('--gpu-memory', type=float, default=.45)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    from experiments.qwen3_bonus.budget import parse_start
    from experiments.qwen3_bonus.perm_entry import perm_overrides, ALGORITHM
    from experiments.qwen3_bonus.gdpo_watchdog import launch_watchdog
    from experiments.qwen3_bonus.gdpo_platform import verify_platform_script, AUTODL_SHUTDOWN_SHA256
    budget = training_budget(rate=args.hourly_rate, started_at=parse_start(args.rental_started_at),
        quantum_seconds=args.billing_quantum_seconds, preparation_cost=args.preparation_cost_yuan)
    budget.cost(time.time())
    if platform.system() != 'Linux':
        raise RuntimeError('Actual training requires the prepared Linux CUDA instance')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one NVIDIA GPU; CPU preparation never starts training')
    if args.shutdown_script_sha256 != AUTODL_SHUTDOWN_SHA256:
        raise ValueError('Unknown platform stop mechanism')
    verify_platform_script(Path('/usr/bin/shutdown'), args.shutdown_script_sha256)
    preflight = json.loads(args.preflight.read_text())
    preparing = args.selection is None
    if preparing:
        from experiments.qwen3_bonus.perm_models import verify_model, CANDIDATES
        verify_base(args.base, 'base')
        verify_data(args.data)
        validate_preflight(args.base, preflight)
        paths = args.reward_candidate or [root/'models/reward-1.7b', ROOT/'artifacts/models/reward']
        inventories = [verify_model(p) for p in paths]
        if len(inventories) != 2 or any(any(row.get(k) != v for k,v in approved.items())
                for row,approved in zip(inventories, CANDIDATES.values())):
            raise ValueError('Preparation requires exactly pinned 1.7B then 0.6B local candidates')
        selected = {'model_path':str(paths[0].resolve()), 'calibration_path':str(args.run/'calibration-pending.json')}
    else:
        selection = json.loads(args.selection.read_text())
        selected, _ = validate_assets(args.base, args.data, selection, preflight)
        resources, candidates = verify_resource_evidence(selection)
        import psutil
        limit = cgroup_memory_limit()
        if (resources['physical_memory_bytes'] > psutil.virtual_memory().total
                or (limit is not None and (resources.get('cgroup_limit_bytes') is None or resources['cgroup_limit_bytes'] > limit))
                or resources['hourly_yuan'] != args.hourly_rate):
            raise ValueError('Create fresh resource evidence for actual hardware and price')
        from experiments.qwen3_bonus.perm_calibration import _selection_policy
        actual = _selection_policy(candidates, live_resources(resources, budget, time.time()))
        if not actual['passed'] or actual['selected'] != selected:
            raise ValueError('Frozen reward choice no longer eligible within actual remaining rental allowance')
    perm_overrides(args.base, Path(selected['model_path']), args.data, args.run,
                   Path(selected['calibration_path']), args.max_steps, gpu_memory=args.gpu_memory)
    run = output_path(args.run)
    if run.exists() and any(run.iterdir()):
        raise ValueError('Use a fresh run; no implicit second training')
    run.mkdir(parents=True, exist_ok=True)
    sources = {p.name: sha256(p) for p in sorted(ROOT.glob('*.py'))}
    identity = {'base_revision': SOURCES['base']['revision'], 'base_inventory_sha256': sha256(args.base/'bonus_source.json'),
        'base_content_sha256':base_content_identity(args.base),
        'dataset_manifest_sha256': sha256(args.data/'dataset_manifest.json'), 'source_sha256': sources}
    manifest = {'budget': asdict(budget), 'identity': identity, 'seed': 3407,
        'resume': False, 'actual_algorithm': ALGORITHM, 'created_at': time.time(), 'max_steps': args.max_steps,
        'preparation_cost_yuan': args.preparation_cost_yuan, 'production_enabled': False}
    if preparing:
        manifest['preparation'] = {'candidate_paths':[str(p.resolve()) for p in paths],
            'candidate_inventories':inventories, 'data':str(args.data.resolve()), 'completed':False}
    else:
        identity.update(reward_inventory_sha256=selected['model_inventory_sha256'],reward_revision=selected['revision'],
            calibration_sha256=selected['calibration_sha256'],reward_selection_sha256=sha256(args.selection))
        manifest['selected'] = selected
    if budget.status(time.time(),0,0)['should_stop']:
        raise ValueError('Insufficient live rental allowance to start supervised preparation/training')
    write_json(run/'launch.json', manifest)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_DISABLED='true',
                      TOKENIZERS_PARALLELISM='false', HYDRA_FULL_ERROR='1')
    command = [sys.executable, '-m', 'experiments.qwen3_bonus.perm_entry', 'worker', '--base', str(args.base.resolve()),
        '--reward-model', selected['model_path'], '--data', str(args.data.resolve()),
        '--calibration', selected['calibration_path'], '--run', str(run), '--max-steps', str(args.max_steps),
        '--gpu-memory', str(args.gpu_memory)]
    # The user retains the allocated GPU; budget enforcement stops training only.
    stop = None
    manifest['instance_shutdown_policy'] = 'preserve_user_allocation'
    write_json(run/'launch.json', manifest)
    supervisor = launch_watchdog(run, command, budget, stop, allow_preserve_allocation=True)
    print(json.dumps({'watchdog_pid': supervisor.pid, 'run': str(run), 'total_cap_yuan': 20,
                      'gpu_cap_yuan': budget.cap_yuan, 'actual_algorithm': ALGORITHM}))
    if args.wait:
        raise SystemExit(supervisor.wait())


if __name__ == '__main__':
    main()
