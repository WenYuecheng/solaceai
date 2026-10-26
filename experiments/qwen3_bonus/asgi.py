"""Explicit optional SolaceAI entrypoint; importing this module has no side effects."""
from __future__ import annotations

import json
import os
from pathlib import Path


def deployment_fingerprints(config):
    """Bind quality evidence to validated immutable model bytes, not their paths."""
    from experiments.qwen3_bonus.common import sha256
    from experiments.qwen3_bonus.serving import validate_artifacts
    validate_artifacts(config)
    result={'base_manifest':sha256(Path(config.model_path)/'bonus_source.json')}
    for name,directory in (('emotion',config.emotion_adapter),('response',config.response_adapter)):
        result[name]={file:sha256(Path(directory)/file) for file in (
            'adapter_config.json','adapter_model.safetensors','bonus_adapter.json')}
    return result


def selected_config():
    from experiments.qwen3_bonus.common import ROOT
    from experiments.qwen3_bonus.serving import ServingConfig
    return ServingConfig.from_file(os.environ.get('BONUS_SERVING_CONFIG',ROOT/'configs/serving.example.json'))


def require_release_report(path=None, *, config=None):
    """Local production fallback is opt-in AND requires an accepted release gate."""
    selected = path or os.environ.get('BONUS_RELEASE_REPORT')
    if not selected:
        raise ValueError('Enabled fallback requires BONUS_RELEASE_REPORT with passed=true')
    target = Path(selected).expanduser().resolve()
    if target.stat().st_size > 1024 * 1024:
        raise ValueError('Release report too large')
    report = json.loads(target.read_text(encoding='utf-8'))
    if not isinstance(report, dict) or report.get('passed') is not True:
        raise ValueError('Release gate did not pass; optional fallback stays disabled')
    current=deployment_fingerprints(config or selected_config())
    if report.get('deployment_artifacts')!=current:
        raise ValueError('Release report does not match current deployment artifacts')
    return report


def create_app(*, local_fallback_enabled=None, local_client=None, release_report=None,
               replace_emotion=False, base_app=None):
    from app.application import create_app as original_create_app
    app = base_app if base_app is not None else original_create_app()
    enabled = (os.environ.get('BONUS_LOCAL_FALLBACK_ENABLED', '').strip().lower() == 'true'
               if local_fallback_enabled is None else bool(local_fallback_enabled))
    app.state.local_fallback_enabled = enabled
    if not enabled:
        return app
    # An injected test/research client can exercise wiring without fabricated
    # artifact quality claims. The CLI/default real HTTP client requires the gate.
    if local_client is None:
        config=selected_config()
        app.state.bonus_release_report = require_release_report(release_report,config=config)
        from experiments.qwen3_bonus.integration import LocalGatewayClient
        local_client = LocalGatewayClient(url=f'http://127.0.0.1:{config.gateway_port}',
            token=config.local_token,timeout=config.request_timeout_seconds)
    from fastapi.routing import APIRoute
    from experiments.qwen3_bonus.integration import clone_function, make_chat_service
    service_class = make_chat_service(local_client, replace_emotion=replace_emotion)
    app.state.local_emotion_replacement = replace_emotion
    for index, route in enumerate(app.router.routes):
        if isinstance(route, APIRoute) and route.path == '/api/chat/stream':
            # Rebuild only this app's route, including dependency graph, from the
            # existing public endpoint. Original module globals never change.
            endpoint = clone_function(route.endpoint, ChatService=service_class)
            replacement = APIRoute(
                route.path, endpoint, methods=route.methods, name=route.name,
                response_model=route.response_model, status_code=route.status_code,
                tags=route.tags, dependencies=route.dependencies,
                summary=route.summary, description=route.description,
                response_description=route.response_description,
                response_class=route.response_class, operation_id=route.operation_id,
                include_in_schema=route.include_in_schema,
                dependency_overrides_provider=app)
            app.router.routes[index] = replacement
            app.state.bonus_chat_service = service_class
            break
    else:
        raise RuntimeError('Expected existing SolaceAI chat route not found')
    return app
