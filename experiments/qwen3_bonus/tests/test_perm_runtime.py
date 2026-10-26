"""Real owned child/atomic-files checks; no GPU, model or cloud stop is used."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import pytest


@pytest.mark.parametrize('code', ['pass', 'raise SystemExit(3)'])
def test_watchdog_preserves_instance_when_shutdown_disabled(tmp_path, monkeypatch, code):
    from experiments.qwen3_bonus import common
    from experiments.qwen3_bonus.gdpo_runtime import GDPOBudget
    from experiments.qwen3_bonus.gdpo_watchdog import run_watchdog
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    result = run_watchdog({'run': str(tmp_path/'run'),
        'budget': asdict(GDPOBudget(1., time.time(), 1)),
        'training_argv': [sys.executable, '-c', code],
        'shutdown_argv': None})
    assert result['training_returncode'] == (0 if code == 'pass' else 3)
    assert result['instance_preserved'] is True
    assert result['shutdown_returncode'] is None
    assert not (tmp_path/'run/shutdown.log').exists()


@pytest.mark.parametrize('shutdown', [None, ['/usr/bin/true']])
def test_watchdog_stops_owned_silent_process_without_training_metrics(tmp_path, monkeypatch, shutdown):
    from experiments.qwen3_bonus import common
    from experiments.qwen3_bonus.gdpo_runtime import GDPOBudget
    from experiments.qwen3_bonus.gdpo_watchdog import run_watchdog
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    now = time.time()
    budget = GDPOBudget(1., now, 60)
    run = tmp_path/'run'
    result = run_watchdog({'run': str(run), 'budget': asdict(budget),
        'training_argv': [sys.executable, '-c', 'import time; time.sleep(60)'],
        'shutdown_argv': shutdown, 'save_seconds': 120,
        'training_deadline': now+.04, 'release_deadline': now+.5,
        'term_grace_seconds': .04, 'kill_grace_seconds': .02, 'poll_seconds': .01})
    assert result['terminated'] or result['killed']
    assert result['training_returncode'] != 0
    assert result['shutdown_returncode'] == (None if shutdown is None else 0)
    assert result['instance_preserved'] is (shutdown is None)
    assert not (run/'steps.jsonl').exists()
    assert json.loads((run/'STOP_REQUESTED.json').read_text())['reason'] == 'setup_cap'


def test_interrupted_save_preserves_old_full_checkpoint_and_all_small_snapshots(tmp_path, monkeypatch):
    import numpy as np
    from safetensors.numpy import save_file
    from experiments.qwen3_bonus import common
    from experiments.qwen3_bonus.gdpo_runtime import AtomicCheckpointStore
    from experiments.qwen3_bonus.snapshots import save_snapshot
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    store = AtomicCheckpointStore(tmp_path/'checkpoints')
    identity = {'model': 'explicit tiny fixture'}
    required = ['state.bin', 'actor/lora_adapter/adapter_config.json', 'actor/lora_adapter/adapter_model.safetensors']
    def save(stage):
        folder = stage/'actor/lora_adapter'; folder.mkdir(parents=True)
        (stage/'state.bin').write_bytes(b'explicit tiny state fixture')
        (folder/'adapter_config.json').write_text(json.dumps({'r':8, 'lora_alpha':16}))
        save_file({'fixture.lora_A.weight': np.zeros((8, 1), dtype=np.float32)}, str(folder/'adapter_model.safetensors'))
    first = store.commit(2, save, {'identity':identity}, required)
    first_small = save_snapshot(first/'actor/lora_adapter', tmp_path/'adapter_snapshots', 2)
    def interrupted(stage):
        (stage/'state.bin').write_bytes(b'unfinished')
        raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError, match='interruption'):
        store.commit(8, interrupted, {'identity':identity}, required)
    assert store.resume(identity, required)['step'] == 2
    assert first.is_dir() and first_small.is_dir()
    second = store.commit(8, save, {'identity':identity}, required)
    second_small = save_snapshot(second/'actor/lora_adapter', tmp_path/'adapter_snapshots', 8)
    assert store.resume(identity, required)['step'] == 8
    assert not first.exists() and first_small.is_dir() and second_small.is_dir()
    (second/'state.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        store.resume(identity, required)
