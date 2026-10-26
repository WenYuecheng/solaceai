"""Local GDPO static checks, CPU reward calibration, and opt-in Metal probes.

`static` hashes pinned raw/reward/MLX4 inventories and inspects actual rank-8 q/v
adapter tensors without loading a model. `calibrate --execute` explicitly loads
the frozen reward model on CPU for exactly twenty model-reviewed pairs. `probe
--execute` runs actual old trained adapters and a clearly untrained rank-8 dummy
through the existing local probe. It never claims inference from prerequisites.
All outputs are fresh and confined to this experiment; no paid API is used.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import replace
from pathlib import Path

from experiments.qwen3_bonus.common import ROOT, output_path, sha256
from experiments.qwen3_bonus.gdpo_evaluation import (code_checksums, digest_json, read_verified,
                                                   write_immutable, _text, _hash)
from experiments.qwen3_bonus.models import SOURCES, verify_model
from experiments.qwen3_bonus.rewards import calibration_gate


def runtime_inventory(*, metal_python: Path | None = None) -> dict:
    packages = {}
    for name in ('numpy', 'matplotlib', 'torch', 'transformers', 'safetensors', 'peft',
                 'mlx', 'mlx-lm', 'vllm', 'vllm-metal', 'verl', 'ray'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import subprocess
    metal_python = metal_python or ROOT / '.venv-metal/bin/python'
    metal = {'python_executable': str(metal_python), 'inventory_error': None, 'packages': {}}
    if metal_python.is_file():
        script = ('import importlib.metadata as m,json,sys;'
                  'p={d.metadata["Name"].lower():d.version for d in m.distributions()};'
                  'print(json.dumps({"python_executable":sys.executable,"packages":'
                  '{n:p.get(n) for n in ("mlx","mlx-lm","vllm","vllm-metal","torch","transformers")}}))')
        try:
            result = subprocess.run([str(metal_python), '-c', script], capture_output=True, text=True,
                                    check=True, timeout=10)
            metal.update(json.loads(result.stdout))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            metal['inventory_error'] = type(exc).__name__
    else:
        metal['inventory_error'] = 'InterpreterMissing'
    metal['apple_silicon'] = platform.system() == 'Darwin' and platform.machine() == 'arm64'
    return {'python': platform.python_version(), 'python_executable': sys.executable,
            'platform': platform.system(), 'architecture': platform.machine(), 'packages': packages,
            'tools_runtime': {'python_executable': sys.executable, 'packages': packages},
            'metal_runtime': metal,
            'training_runtime': {'scope': 'separate Linux CUDA environment required',
                                 'validated': False, 'cuda_runtime': None},
            'genuine_inference': False}


def _adapter_inventory(path: Path, base_source: dict) -> dict:
    from safetensors import safe_open
    config = json.loads((path / 'adapter_config.json').read_text(encoding='utf-8'))
    if (config.get('r') != 8 or config.get('lora_alpha') != 16
            or set(config.get('target_modules', [])) != {'q_proj', 'v_proj'}
            or config.get('base_model_name_or_path') != base_source['repo_id']
            or config.get('revision') != base_source['revision']
            or config.get('modules_to_save') or config.get('bias', 'none') != 'none'):
        raise ValueError('Adapter requires pinned revision, rank 8, alpha 16, and q/v targets')
    pairs = {}
    with safe_open(str(path / 'adapter_model.safetensors'), framework='numpy') as weights:
        for key in weights.keys():
            import re
            match = re.fullmatch(r'.*\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_([AB])(?:\.default)?\.weight', key)
            if match is None:
                raise ValueError('Unexpected adapter tensor layout')
            layer, module, side = match.groups()
            shape = tuple(weights.get_slice(key).get_shape())
            if len(shape) != 2 or shape[0 if side == 'A' else 1] != 8:
                raise ValueError('Actual adapter tensor rank must equal 8')
            pairs.setdefault((layer, module), {})[side] = shape
    if not pairs or any(set(sides) != {'A', 'B'} for sides in pairs.values()):
        raise ValueError('Incomplete q/v adapter tensor pairs')
    layers = {layer for layer, _ in pairs}
    if any((layer, module) not in pairs for layer in layers for module in ('q_proj', 'v_proj')):
        raise ValueError('Every adapter layer requires both q and v tensors')
    return {'config_sha256': sha256(path / 'adapter_config.json'),
            'weights_sha256': sha256(path / 'adapter_model.safetensors'),
            'rank': 8, 'alpha': 16, 'targets': ['q_proj', 'v_proj'], 'layers': len(layers)}



def verify_raw_base(raw_base: Path, *, allow_missing_raw_weights: bool = False) -> tuple[dict, dict]:
    """Explicit metadata-only mode never attests to original raw weight contents."""
    if not allow_missing_raw_weights:
        source = verify_model(raw_base, 'base')
        return source, {'raw_weights_verified': True, 'raw_inventory_verified': True,
                        'missing_raw_weights': [], 'raw_verification_mode': 'full'}
    source = json.loads((raw_base / 'bonus_source.json').read_text(encoding='utf-8'))
    if any(source.get(k) != v for k, v in SOURCES['base'].items()):
        raise ValueError('Raw base inventory identity/revision mismatch')
    files = source.get('files', {})
    if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Incomplete pinned raw inventory')
    checked, missing = {}, []
    for name, expected in files.items():
        target = (raw_base / name).resolve()
        if Path(name).is_absolute() or not target.is_relative_to(raw_base.resolve()) or not _hash(expected):
            raise ValueError('Invalid raw inventory path or SHA256')
        if name.endswith('.safetensors'):
            if not target.is_file(): missing.append(name)
            continue
        if not target.is_file() or sha256(target) != expected:
            raise ValueError('Raw nonweight metadata hash mismatch')
        checked[name] = expected
    return source, {'raw_weights_verified': False, 'raw_inventory_verified': True,
                    'missing_raw_weights': sorted(missing), 'raw_nonweight_sha256': checked,
                    'raw_verification_mode': 'inventory_only',
                    'scope': 'Pinned raw inventory and nonweight files only; original raw weights unverified.'}


def create_probe_adapters(raw_base: Path, output: Path, *, allow_missing_raw_weights=False) -> dict:
    """Zero rank8 tensors from verified config only; never copies or reads raw weights."""
    import numpy as np
    from safetensors.numpy import save_file
    source, _ = verify_raw_base(raw_base, allow_missing_raw_weights=allow_missing_raw_weights)
    config = json.loads((raw_base / 'config.json').read_text()); output = output_path(output)
    if config.get('model_type') != 'qwen3': raise ValueError('Expected Qwen3 config')
    if output.exists(): raise FileExistsError('Dummy adapter output must be fresh')
    hidden = config['hidden_size']; head = config.get('head_dim', hidden // config['num_attention_heads'])
    sizes = {'q_proj': config['num_attention_heads'] * head, 'v_proj': config['num_key_value_heads'] * head}
    result = {}
    for task in ('emotion', 'response'):
        path = output / f'solace-{task}'; path.mkdir(parents=True)
        tensors = {f'base_model.model.model.layers.{layer}.self_attn.{module}.lora_{side}.weight': np.zeros(shape, np.float32)
                   for layer in range(config['num_hidden_layers']) for module, size in sizes.items()
                   for side, shape in (('A', (8, hidden)), ('B', (size, 8)))}
        save_file(tensors, str(path / 'adapter_model.safetensors'))
        write_immutable(path / 'adapter_config.json', {'peft_type':'LORA', 'task_type':'CAUSAL_LM', 'r':8,
           'lora_alpha':16, 'lora_dropout':0., 'bias':'none', 'target_modules':['q_proj','v_proj'],
           'base_model_name_or_path':source['repo_id'], 'revision':source['revision'], 'inference_mode':True})
        write_immutable(path / 'bonus_adapter.json', {'trained':False, 'purpose':'Untrained local rank8 compatibility dummy'})
        result[task] = str(path)
    return result


def static_preflight(raw_base: Path, reward: Path, converted_base: Path, adapters: dict[str, Path],
                     *, allow_missing_raw_weights: bool = False) -> dict:
    base_source, raw_report = verify_raw_base(raw_base, allow_missing_raw_weights=allow_missing_raw_weights)
    verify_model(reward, 'reward')
    converted = verify_model(converted_base, 'base')
    config = json.loads((converted_base / 'config.json').read_text(encoding='utf-8'))
    quantization = config.get('quantization', config.get('quantization_config', {}))
    if (converted.get('quantization_bits') != 4 or quantization.get('bits') != 4
            or config.get('model_type') != 'qwen3'
            or converted.get('source_inventory_sha256') != sha256(raw_base / 'bonus_source.json')):
        raise ValueError('Converted base must be the same verified raw revision in MLX4')
    if not adapters:
        raise ValueError('Provide at least one adapter to inspect')
    inspected = {name: _adapter_inventory(Path(path), base_source) for name, path in adapters.items()}
    return {**raw_report, 'artifacts_valid': True, 'converted_base_verified': True,
            'adapters_verified': True, 'reward_weights_verified': True,
            'verified_scope': 'Converted MLX4 base, adapters, reward model; raw scope recorded separately', 'genuine_inference': False, 'release_gate_passed': False,
            'base_revision': base_source['revision'], 'reward_revision': SOURCES['reward']['revision'],
            'manifests_sha256': {'base': sha256(raw_base / 'bonus_source.json'),
                'reward': sha256(reward / 'bonus_source.json'), 'mlx4': sha256(converted_base / 'bonus_source.json')},
            'adapters': inspected, 'runtime': runtime_inventory(), 'code_sha256': code_checksums(),
            'paid_api_calls': 0}


def calibrate_pairs(pairs: list[dict], scorer, *, model_inventory_sha256: str | None = None) -> dict:
    """Scorer injection supports tiny tests; CLI always uses the real frozen CPU RM."""
    import numpy as np
    if len(pairs) != 20 or len({r['id'] for r in pairs}) != 20:
        raise ValueError('Exactly 20 unique preexpected pairs required')
    for row in pairs:
        if (row.get('review_type') != 'model' or row.get('reviewed') is not True
                or not _text(row.get('reviewer')) or row.get('expected') not in ('a', 'b')
                or not _text(row.get('review_rationale'))):
            raise ValueError('Every expectation requires explicit model review, identity, and rationale')
    scored = []
    for row in pairs:
        result = dict(row)
        for side in ('a', 'b'):
            score = float(scorer(row['prompt'], row[side]))
            if not math.isfinite(score):
                raise ValueError('Nonfinite raw calibration score')
            result['score_' + side] = score
        scored.append(result)
    raw = np.asarray([r['score_' + side] for r in scored for side in ('a', 'b')], dtype=float)
    quartiles = np.quantile(raw, [.25, .5, .75])
    return {'reward_model_revision': SOURCES['reward']['revision'],
            'model_inventory_sha256': model_inventory_sha256, 'median': float(quartiles[1]), 'iqr': float(quartiles[2] - quartiles[0]),
            'pairs': scored, 'gate': calibration_gate(scored, allow_model_review=True),
            'raw_scores': {'count': 40, 'median': float(quartiles[1]), 'q1': float(quartiles[0]),
                           'q3': float(quartiles[2]), 'iqr': float(quartiles[2] - quartiles[0])},
            'pairs_sha256': digest_json(pairs), 'review_type': 'model', 'human_reviewed': False,
            'paid_api_calls': 0}


def run_probe(config_path: Path, raw_base: Path, output: Path, *, reward: Path | None = None,
              grpo_adapter: Path | None = None, allow_missing_raw_weights: bool = False) -> dict:
    """Explicit actual-engine use only. Caller serializes memory with CPU calibration."""
    from experiments.qwen3_bonus.compatibility import validate_temporary
    from experiments.qwen3_bonus.probe import live_probe
    from experiments.qwen3_bonus.serving import ServingConfig, validate_artifacts
    output = output_path(output)
    if output.exists():
        raise FileExistsError('Probe output must be fresh')
    config = ServingConfig.from_file(config_path)
    validate_artifacts(config)
    # Inspection proves the old adapters really match the same r8 recipe before inference.
    grpo_adapter = grpo_adapter or ROOT / 'artifacts/adapters/solace-response-grpo32'
    grpo_config = replace(config, response_adapter=str(grpo_adapter))
    validate_artifacts(grpo_config)
    static = static_preflight(raw_base, reward or ROOT / 'artifacts/models/reward', Path(config.model_path),
          {'emotion': Path(config.emotion_adapter), 'current': Path(config.response_adapter), 'grpo32': grpo_adapter},
          allow_missing_raw_weights=allow_missing_raw_weights)
    models = ('solace-emotion', 'solace-response') * 2
    old = asyncio.run(live_probe(config, models=models))
    grpo = asyncio.run(live_probe(grpo_config, models=models))
    paths = create_probe_adapters(raw_base, output / 'untrained-rank8',
                                  allow_missing_raw_weights=allow_missing_raw_weights)
    dummy_config = replace(config, emotion_adapter=paths['emotion'], response_adapter=paths['response'])
    validate_temporary(dummy_config)
    dummy = asyncio.run(live_probe(dummy_config, validate=False, models=models))
    def completed(record):
        return (record.get('genuine_inference') is True and record.get('smoke_passed') is True
                and not record.get('error_type') and len(record.get('requests', [])) == len(models))
    report = {'static': static, 'old_adapters': old, 'grpo32_adapter': grpo, 'untrained_rank8_dummy': dummy,
              'genuine_inference': completed(old) and completed(grpo) and completed(dummy),
              'compatibility_passed': completed(old) and completed(grpo) and completed(dummy), 'release_gate_passed': False,
              'config_sha256': sha256(config_path), 'runtime': runtime_inventory(), 'paid_api_calls': 0,
              'scope': 'Actual local transport/shape compatibility only; dummy is untrained.'}
    write_immutable(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('static', help='Read-only model manifests and actual safetensor rank inspection')
    command.add_argument('--base', type=Path, default=ROOT / 'artifacts/models/base')
    command.add_argument('--reward', type=Path, default=ROOT / 'artifacts/models/reward')
    command.add_argument('--raw-inventory-only', action='store_true', help='Verify raw metadata only; explicitly record raw_weights_verified=false')
    command.add_argument('--mlx4', type=Path, default=ROOT / 'artifacts/models/base-mlx4')
    command.add_argument('--adapter', action='append', required=True, metavar='NAME=PATH')
    command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('calibrate', help='Actually score 20 reviewed pairs on CPU, 40 raw scores')
    command.add_argument('--sheet', type=Path, required=True)
    command.add_argument('--sheet-sha256', required=True)
    command.add_argument('--reward', type=Path, default=ROOT / 'artifacts/models/reward')
    command.add_argument('--execute', action='store_true', help='Required: explicitly load the CPU reward model')
    command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('probe', help='Actually load old adapters and untrained rank8 dummy on Metal')
    command.add_argument('--config', type=Path, required=True)
    command.add_argument('--config-sha256', required=True)
    command.add_argument('--base', type=Path, default=ROOT / 'artifacts/models/base')
    command.add_argument('--reward', type=Path, default=ROOT / 'artifacts/models/reward')
    command.add_argument('--grpo-adapter', type=Path, default=ROOT / 'artifacts/adapters/solace-response-grpo32')
    command.add_argument('--raw-inventory-only', action='store_true', help='Use verified raw config without requiring original raw shards; raw_weights_verified=false')
    command.add_argument('--execute', action='store_true', help='Required: explicitly run actual local inference')
    command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = output_path(args.output)
    if output.exists(): raise FileExistsError('Choose a fresh immutable output')
    if args.action == 'static':
        adapters = {}
        for spec in args.adapter:
            if '=' not in spec: raise ValueError('Expected NAME=PATH adapter argument')
            name, path = spec.split('=', 1)
            if not name or name in adapters: raise ValueError('Unique nonempty adapter names required')
            adapters[name] = Path(path)
        report = static_preflight(args.base, args.reward, args.mlx4, adapters,
                                  allow_missing_raw_weights=args.raw_inventory_only)
        write_immutable(output, report)
    elif args.action == 'calibrate':
        if not args.execute: parser.error('calibrate requires --execute for real CPU model scoring')
        document = read_verified(args.sheet, args.sheet_sha256)
        verify_model(args.reward, 'reward')
        pairs = document['pairs']
        # Validate reviewed expectations before allocating model memory.
        if len(pairs) != 20 or len({r['id'] for r in pairs}) != 20:
            raise ValueError('Exactly 20 unique preexpected pairs required')
        if any(r.get('review_type') != 'model' or not r.get('reviewed') or not r.get('reviewer')
               or not r.get('review_rationale') or r.get('expected') not in ('a', 'b') for r in pairs):
            raise ValueError('Explicit model review required before scoring')
        from experiments.qwen3_bonus.rewards import FrozenRewardModel
        report = calibrate_pairs(pairs, FrozenRewardModel(str(args.reward), 'cpu'),
                                 model_inventory_sha256=sha256(args.reward / 'bonus_source.json'))
        report.update(sheet_sha256=args.sheet_sha256, model_inventory_sha256=sha256(args.reward / 'bonus_source.json'),
                      reward_model_revision=SOURCES['reward']['revision'], runtime=runtime_inventory(), code_sha256=code_checksums())
        write_immutable(output, report)
        if not report['gate']['passed']: raise SystemExit(2)
    else:
        if not args.execute: parser.error('probe requires --execute for actual local inference')
        read_verified(args.config, args.config_sha256)
        report = run_probe(args.config, args.base, output, reward=args.reward, grpo_adapter=args.grpo_adapter,
                           allow_missing_raw_weights=args.raw_inventory_only)
        if not report['compatibility_passed']: raise SystemExit(2)


if __name__ == '__main__': main()
