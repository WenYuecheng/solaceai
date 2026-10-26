"""Explicitly selected, user-authorized local candidate; original app stays intact."""
from __future__ import annotations
import json
import os
from pathlib import Path
from experiments.qwen3_bonus.common import ROOT


def validate_candidate(report: Path, config) -> dict:
    from experiments.qwen3_bonus.asgi import deployment_fingerprints
    if report.stat().st_size > 1024*1024:
        raise ValueError('Candidate report too large')
    data=json.loads(report.read_text())
    if (data.get('mode')!='user_authorized_local_candidate' or
        data.get('genuine_inference') is not True or data.get('smoke_passed') is not True or
        data.get('classification_cases',0)<4 or data.get('alternating_requests',0)<6 or
        data.get('deployment_artifacts')!=deployment_fingerprints(config)):
        raise ValueError('Local candidate needs matching successful real inference evidence')
    return data


def create_app():
    from experiments.qwen3_bonus.asgi import selected_config, create_app as wire
    from experiments.qwen3_bonus.integration import LocalGatewayClient
    config=selected_config()
    report=validate_candidate(Path(os.environ.get('BONUS_CANDIDATE_REPORT', ROOT/'artifacts/local-deployment/acceptance.json')),config)
    from apps.offline.main import create_app as original
    from apps.offline.settings import load_settings
    # Same offline database, RAG and cloud credentials; only the emotion client
    # and final-answer failure branch are replaced in this selected app instance.
    settings=load_settings().model_copy(update={'local_model_switch_enabled':True,
        'local_gateway_url':f'http://127.0.0.1:{config.gateway_port}', 'local_gateway_token':config.local_token})
    client=LocalGatewayClient(url=f'http://127.0.0.1:{config.gateway_port}',token=config.local_token,
                              timeout=config.request_timeout_seconds)
    app=wire(local_fallback_enabled=True,local_client=client,replace_emotion=True,base_app=original(settings))
    app.state.bonus_candidate_report=report
    return app
