"""Actual local adapter compatibility without loading a CPU reward model."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path

from experiments.qwen3_bonus.common import ROOT, output_path, sha256


def probe_completed(record):
    requests = record.get('requests', [])
    return (record.get('genuine_inference') is True and record.get('smoke_passed') is True
        and not record.get('error_type') and len(requests) == 2
        and all(r.get('valid_format') is True and str(r.get('content', '')).strip() for r in requests))


def historical_grpo_provenance(adapter):
    """Old comparison adapter uses real archived training, not production packaging."""
    from experiments.qwen3_bonus.models import SOURCES
    run = ROOT/'artifacts/remote/received/weste-final/extracted/artifacts/runs/response-weste3'
    result = json.loads((run/'training_result.json').read_text())
    snapshot = run/'adapter_snapshots/step-000032/adapter_model.safetensors'
    if (result.get('trained') is not True or result.get('steps') != 32
            or result.get('base_revision') != SOURCES['base']['revision']
            or sha256(Path(adapter)/'adapter_model.safetensors') != sha256(snapshot)):
        raise ValueError('Historical GRPO adapter differs from the genuine step32 snapshot')
    return {'steps': 32, 'training_result_sha256': sha256(run/'training_result.json'),
            'snapshot_sha256': sha256(snapshot)}


def run_probe(config_path, raw_base, output, *, execute=False, grpo_adapter=None):
    if not execute:
        raise ValueError('Actual inference requires execute=True')
    from experiments.qwen3_bonus.serving import ServingConfig, validate_artifacts
    from experiments.qwen3_bonus.probe import live_probe
    from experiments.qwen3_bonus.compatibility import validate_temporary
    from experiments.qwen3_bonus.gdpo_preflight import verify_raw_base, _adapter_inventory, create_probe_adapters
    from experiments.qwen3_bonus.gdpo_evaluation import write_immutable, checkpoint_digest
    config_path, raw_base, output = Path(config_path), Path(raw_base), output_path(output)
    if output.exists():
        raise FileExistsError('Use a new immutable preflight output')
    config = ServingConfig.from_file(config_path)
    validate_artifacts(config)
    grpo_adapter = grpo_adapter or ROOT/'artifacts/adapters/solace-response-grpo32'
    old_config = replace(config, response_adapter=str(grpo_adapter))
    historical = historical_grpo_provenance(grpo_adapter)
    base_source, base_check = verify_raw_base(raw_base, allow_missing_raw_weights=True)
    adapters = {label: _adapter_inventory(Path(path), base_source) for label, path in (
        ('current', config.response_adapter), ('grpo32', grpo_adapter), ('emotion', config.emotion_adapter))}
    models = ('solace-response',)*2
    current = asyncio.run(live_probe(config, models=models))
    grpo = asyncio.run(live_probe(old_config, validate=False, models=models))
    dummy = create_probe_adapters(raw_base, output/'untrained-rank8', allow_missing_raw_weights=True)
    dummy_config = replace(config, emotion_adapter=dummy['emotion'], response_adapter=dummy['response'])
    validate_temporary(dummy_config)
    new_format = asyncio.run(live_probe(dummy_config, validate=False, models=models))
    passed = all(probe_completed(record) for record in (current, grpo, new_format))
    from experiments.qwen3_bonus.perm_launch import base_content_identity
    report = {'actual_algorithm': 'perm_grpo', 'genuine_inference': passed, 'compatibility_passed': passed,
        'base_inventory_sha256': sha256(raw_base/'bonus_source.json'),
        'base_content_sha256': base_content_identity(raw_base, allow_missing_raw_weights=True),
        'converted_base_inventory_sha256': sha256(Path(config.model_path)/'bonus_source.json'),
        'config_sha256': sha256(config_path), 'raw_base': base_check, 'adapters': adapters,
        'current': current, 'grpo32': grpo, 'historical_training': historical, 'untrained_rank8': new_format,
        'dummy_sha256': {name: checkpoint_digest(Path(path)) for name, path in dummy.items()},
        'production_enabled': False, 'paid_api_calls': 0}
    write_immutable(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=ROOT/'artifacts/models/base')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = run_probe(args.config, args.base, args.output, execute=args.execute)
    print(json.dumps({'output': str(args.output), 'genuine_inference': result['genuine_inference'],
                      'compatibility_passed': result['compatibility_passed']}))
    if not result['compatibility_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
