"""Report local prerequisites, or explicitly smoke-test owned dual-LoRA inference.

Default invocation is read-only except for the report. --live is required to load
weights; it never contacts a cloud API and stops its owned process on completion.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from experiments.qwen3_bonus.common import output_path, write_json
from experiments.qwen3_bonus.serving import BASE_MODEL, BASE_REVISION, MODELS, EngineManager, ServingConfig, validate_artifacts


def prerequisites(config: ServingConfig) -> dict:
    versions = {}
    for distribution in ('vllm', 'vllm-metal', 'mlx', 'mlx-lm'):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    artifact_error = None
    try:
        validate_artifacts(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        artifact_error = type(exc).__name__  # avoid serializing local paths or keys
    apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'
    return {'platform': platform.system(), 'architecture': platform.machine(),
            'apple_silicon': apple_silicon, 'packages': versions,
            'artifacts_valid': artifact_error is None, 'artifact_error_type': artifact_error,
            'ready_for_live_probe': apple_silicon and all(versions.values()) and artifact_error is None,
            'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
            'models': list(MODELS), 'genuine_inference': False,
            'release_gate_passed': False}


async def live_probe(config: ServingConfig, *, validate: bool = True, models=MODELS) -> dict:
    """Bounded actual-engine smoke; only successful model outputs count as inference."""
    import httpx
    manager = EngineManager(config, validate_artifacts=validate)
    report = {'genuine_inference': False, 'smoke_passed': False, 'release_gate_passed': False,
              'requests': [], 'error_type': None}
    started = time.monotonic()
    try:
        ready = await manager.ensure_ready()
        while not ready and manager.state == 'starting' and time.monotonic() - started < config.startup_timeout_seconds + 2:
            await asyncio.sleep(config.poll_seconds)
            ready = await manager.ensure_ready()
        report['startup_seconds'] = round(time.monotonic() - started, 3)
        if not ready:
            report['error_type'] = 'LocalEngineNotReady'
            return report
        async with httpx.AsyncClient(timeout=config.request_timeout_seconds, trust_env=False,
                                     follow_redirects=False) as client:
            for model in models:
                prompt = ('只输出一个标签：正常、焦虑、低落、高风险。明天考试，我有些紧张。'
                          if model == 'solace-emotion' else '明天考试，我有些紧张，可以陪我整理一下吗？')
                await manager.acquire()
                request_started = time.monotonic()
                try:
                    response = await client.post(config.engine_url + '/v1/chat/completions',
                        headers=manager.headers, json={'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                            'max_tokens': 16 if model == 'solace-emotion' else 256,
                            'temperature': 0, 'stream': False})
                    response.raise_for_status()
                    if response.json().get('model')!=model:raise ValueError('Engine returned another model identity')
                    message = response.json()['choices'][0]['message']
                    content = message.get('content', '').strip()
                    valid = (content in ('正常', '焦虑', '低落', '高风险') if model == 'solace-emotion'
                             else bool(content) and not message.get('tool_calls') and not content.startswith(('{', '[')))
                    report['requests'].append({'model': model, 'seconds': round(time.monotonic() - request_started, 3),
                                               'content': content, 'valid_format': valid,
                                               'usage': response.json().get('usage', {})})
                finally:
                    manager.release()
        report['genuine_inference'] = bool(report['requests'])
        report['smoke_passed'] = len(report['requests']) == len(models) and all(r['valid_format'] for r in report['requests'])
        # A two-prompt smoke is deliberately insufficient for the quality release gate.
        return report
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        report['genuine_inference'] = bool(report['requests'])
        return report
    finally:
        await manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    target = output_path(args.output)
    config = ServingConfig.from_file(args.config)
    report = prerequisites(config)
    if args.live:
        report['live'] = asyncio.run(live_probe(config))
        report['genuine_inference'] = report['live']['genuine_inference']
    write_json(target, report)
    print(json.dumps({'report': str(target), 'genuine_inference': report['genuine_inference'],
                      'ready_for_live_probe': report['ready_for_live_probe']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
