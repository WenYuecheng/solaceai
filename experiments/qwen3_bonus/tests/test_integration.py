import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app.schemas.dtos import AiMessage
from app.core.config import Settings
from app.core.enums import EmotionLabel
from experiments.qwen3_bonus import integration


class Client:
    def __init__(self, result='云端正常'):
        self.result = result
        self.calls = []
        self.settings = SimpleNamespace(ai_max_tokens=256)
    def complete(self, messages, response_format=None):
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result
    def decide(self, messages, tools, event_sink=None, cancel_event=None):
        return {'answer': self.complete(messages)}
    async def stream(self, messages):
        yield self.complete(messages)


def messages():
    return [AiMessage(role='system', content='回答用户，资料: [chunk:17] 证据'), AiMessage(role='user', content='历史：我在准备考试。现在很紧张')]


def test_primary_success_zero_local_and_recovery_resets_flag():
    primary, local = Client(), Client('本地回复')
    wrapper = integration.LocalFallbackClient(primary, local)
    assert wrapper.complete(messages()) == '云端正常'
    assert local.calls == [] and not wrapper.used_fallback
    primary.result = httpx.ReadTimeout('timeout')
    assert wrapper.complete(messages()) == '本地回复' and wrapper.used_fallback
    primary.result = '恢复'
    wrapper.reset_turn()
    assert wrapper.complete(messages()) == '恢复' and not wrapper.used_fallback


def test_decide_preserves_actual_context_and_never_local_tools():
    primary, local = Client(httpx.ReadTimeout('timeout')), Client('可以先慢慢呼吸。')
    wrapper = integration.LocalFallbackClient(primary, local)
    prompt = messages()
    assert wrapper.decide(prompt, {'dangerous': {}}) == {'answer': '可以先慢慢呼吸。'}
    assert local.calls[0][-2:] == prompt
    local.result = '{"tool":"dangerous","arguments":{}}'
    with pytest.raises(ValueError):
        wrapper.decide(prompt, {'dangerous': {}})


def test_context_overflow_is_rethrown_and_local_503_propagates():
    req = httpx.Request('POST', 'https://cloud.test')
    response = httpx.Response(400, json={'error': {'code': 'context_length_exceeded'}}, request=req)
    error = httpx.HTTPStatusError('overflow', request=req, response=response)
    primary, local = Client(error), Client()
    wrapper = integration.LocalFallbackClient(primary, local)
    with pytest.raises(httpx.HTTPStatusError):
        wrapper.complete(messages())
    assert local.calls == []
    primary.result = httpx.ReadTimeout('timeout')
    local.result = httpx.ConnectError('cold 503')
    with pytest.raises(httpx.ConnectError):
        wrapper.complete(messages())


def test_stream_after_prefix_error_does_not_switch():
    class Prefix(Client):
        async def stream(self, messages):
            yield '已输出'
            raise httpx.ReadTimeout('late')
    async def run():
        local = Client('本地')
        wrapper = integration.LocalFallbackClient(Prefix(), local)
        seen = []
        with pytest.raises(httpx.ReadTimeout):
            async for token in wrapper.stream(messages()):
                seen.append(token)
        assert seen == ['已输出'] and local.calls == []
    asyncio.run(run())


def test_cancel_never_activates_local():
    class Cancel(Client):
        async def stream(self, messages):
            raise asyncio.CancelledError()
            yield ''
    async def run():
        local = Client()
        with pytest.raises(asyncio.CancelledError):
            async for _ in integration.LocalFallbackClient(Cancel(), local).stream(messages()):
                pass
        assert not local.calls
    asyncio.run(run())


def test_emotion_only_on_cloud_failure_invalid_label_heuristic_and_high_risk():
    from apps.online.assessment import PsychologicalAssessmentService
    from apps.offline.assessment import heuristic
    primary = Client('{"emotion":"NORMAL","risk":"LOW","emotionScore":0}')
    local = Client('焦虑')
    service = integration.FallbackAssessmentService(PsychologicalAssessmentService(primary), local)
    assert service.assess('考试').emotion == EmotionLabel.NORMAL and not local.calls
    primary.result = httpx.ReadTimeout('timeout')
    assert service.assess('考试').emotion == EmotionLabel.ANXIETY
    assert service.last_source == 'bonus_local_emotion'
    local.result = '无效标签'
    assert service.assess('考试').emotion == heuristic('考试').emotion
    count = len(local.calls)
    assert service.assess('我想自杀').emotion == EmotionLabel.HIGH_RISK
    assert len(local.calls) == count


def test_explicit_app_route_is_wired_and_default_app_untouched(monkeypatch):
    import app.api.routes as routes
    import app.agents.harness as harness
    import app.agents.graph_runtime as graph
    from experiments.qwen3_bonus.asgi import create_app
    original_chat = routes.ChatService
    original_factory = harness.create_agent_runtime
    original_assess = graph.assess_with_source
    monkeypatch.delenv('BONUS_LOCAL_FALLBACK_ENABLED', raising=False)
    disabled = create_app()
    assert disabled.state.local_fallback_enabled is False
    enabled = create_app(local_fallback_enabled=True, local_client=Client('本地'))
    route = next(r for r in enabled.routes if getattr(r, 'path', '') == '/api/chat/stream')
    selected = route.endpoint.__globals__['ChatService']
    assert selected is not original_chat
    service = selected(None, Settings(ai_provider='mock'))
    factory = service.agent_harness._run.__func__.__globals__['create_agent_runtime']
    assert factory is not original_factory
    assert routes.ChatService is original_chat
    assert harness.create_agent_runtime is original_factory
    assert graph.assess_with_source is original_assess


def test_fallback_sticky_across_recovered_calls_in_one_turn():
    primary, local = Client(httpx.ReadTimeout('timeout')), Client('本地')
    wrapper = integration.LocalFallbackClient(primary, local)
    wrapper.complete(messages())
    primary.result = '云端恢复'
    assert wrapper.complete(messages()) == '云端恢复'
    assert wrapper.used_fallback  # graph must remain read-only for this whole turn


def test_decision_after_emitted_prefix_and_cancel_never_falls_back():
    import threading
    class Prefix(Client):
        def decide(self, messages, tools, event_sink=None, cancel_event=None):
            event_sink({'type': 'token', 'content': 'prefix'})
            raise httpx.ReadTimeout('late')
    local = Client('本地')
    wrapper = integration.LocalFallbackClient(Prefix(), local)
    with pytest.raises(httpx.ReadTimeout):
        wrapper.decide(messages(), {}, event_sink=lambda item: None)
    assert not local.calls
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError):
        wrapper.decide(messages(), {}, cancel_event=cancelled)
    assert not local.calls


def test_gateway_client_uses_only_loopback_and_rejects_tool_envelope():
    calls = []
    def transport(request):
        calls.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'tool_calls': [{'function': {}}], 'content': 'bad'}}]})
    client = integration.LocalGatewayClient(transport=httpx.MockTransport(transport))
    with pytest.raises(ValueError):
        client.complete(messages())
    assert calls[0].url.host == '127.0.0.1'
    assert json.loads(calls[0].content)['model'] == 'solace-response'
    with pytest.raises(ValueError):
        integration.LocalGatewayClient('http://example.com:18181')


def test_release_gate_required_before_real_local_client(monkeypatch, tmp_path):
    from experiments.qwen3_bonus.asgi import create_app
    monkeypatch.setenv('BONUS_LOCAL_FALLBACK_ENABLED', 'true')
    monkeypatch.delenv('BONUS_RELEASE_REPORT', raising=False)
    with pytest.raises(ValueError, match='BONUS_RELEASE_REPORT'):
        create_app()
    report = tmp_path / 'release.json'
    report.write_text('{"passed":false}')
    with pytest.raises(ValueError, match='did not pass'):
        create_app(release_report=report)
    import experiments.qwen3_bonus.asgi as entrypoint
    monkeypatch.setattr(entrypoint, 'deployment_fingerprints', lambda config: {'verified': 'test-artifact-hashes'})
    report.write_text(json.dumps({'passed': True, 'deployment_artifacts': {'verified': 'test-artifact-hashes'}}))
    app = create_app(release_report=report)
    assert app.state.local_fallback_enabled is True


def test_runtime_constructor_keeps_safety_and_intent_original(monkeypatch):
    import app.agents.graph_runtime as graph
    settings = Settings(ai_provider='mock')
    seen = {}
    def init(self, db, settings, decision_client=None, safety_client=None, intent_client=None, **kwargs):
        self.client = decision_client
        self.safety = safety_client or Client('safety')
        self.intent_client = intent_client or Client('intent')
        self.settings = settings
        seen.update(safety=self.safety, intent=self.intent_client)
    monkeypatch.setattr(graph.GraphAgentRuntime, '__init__', init)
    runtime = integration.make_runtime_factory(Client('本地'))(None, settings)
    assert isinstance(runtime.client, integration.LocalFallbackClient)
    assert runtime.safety is seen['safety'] and not isinstance(runtime.safety, integration.LocalFallbackClient)
    assert runtime.intent_client is seen['intent'] and not isinstance(runtime.intent_client, integration.LocalFallbackClient)
    assert settings.ai_provider == 'mock'


def test_opted_in_route_constructs_selected_service_without_patching_original(monkeypatch):
    from experiments.qwen3_bonus.asgi import create_app
    from app.api import routes
    from types import SimpleNamespace
    app = create_app(local_fallback_enabled=True, local_client=Client('本地'))
    route = next(r for r in app.routes if getattr(r, 'path', '') == '/api/chat/stream')
    chosen = app.state.bonus_chat_service
    seen = []
    def init(self, db, settings):
        seen.append((db, settings))
    async def stream(self, user, request):
        yield 'data: {"selected":"bonus"}\n\n'
    monkeypatch.setattr(chosen, '__init__', init)
    monkeypatch.setattr(chosen, 'stream_chat', stream)
    async def run():
        response = await route.endpoint(request=SimpleNamespace(), user=SimpleNamespace(roles=[]), db='test-db')
        content = b''.join([part.encode() if isinstance(part, str) else part async for part in response.body_iterator])
        assert b'bonus' in content
    asyncio.run(run())
    assert seen and seen[0][0] == 'test-db'
    assert routes.ChatService is not chosen



def test_release_fingerprints_bind_current_artifacts_and_gateway_config(monkeypatch, tmp_path):
    import experiments.qwen3_bonus.asgi as entrypoint
    from experiments.qwen3_bonus.serving import ServingConfig
    from experiments.qwen3_bonus.common import sha256
    model = tmp_path / 'base'
    model.mkdir()
    (model / 'bonus_source.json').write_text('{"revision":"test-only"}')
    adapters = {}
    for task in ('emotion', 'response'):
        path = tmp_path / task
        path.mkdir()
        for name in ('adapter_model.safetensors', 'adapter_config.json', 'bonus_adapter.json'):
            (path / name).write_text(task + ':' + name)
        adapters[task] = path
    config_data = {'model_path': str(model), 'emotion_adapter': str(adapters['emotion']),
                   'response_adapter': str(adapters['response']), 'gateway_port': 19181,
                   'engine_port': 19182, 'local_token': 'test-private-token'}
    config_path = tmp_path / 'serving.json'
    config_path.write_text(json.dumps(config_data))
    validated = []
    monkeypatch.setattr('experiments.qwen3_bonus.serving.validate_artifacts', lambda config: validated.append(config))
    config = ServingConfig.from_file(config_path)
    fingerprint = entrypoint.deployment_fingerprints(config)
    assert validated[-1] == config
    assert fingerprint['base_manifest'] == sha256(model / 'bonus_source.json')
    assert fingerprint['emotion']['adapter_model.safetensors'] == sha256(adapters['emotion'] / 'adapter_model.safetensors')
    report = tmp_path / 'release.json'
    report.write_text(json.dumps({'passed': True, 'deployment_artifacts': fingerprint}))
    monkeypatch.setenv('BONUS_LOCAL_FALLBACK_ENABLED', 'true')
    monkeypatch.setenv('BONUS_SERVING_CONFIG', str(config_path))
    monkeypatch.setenv('BONUS_RELEASE_REPORT', str(report))
    captured = []
    original_make=integration.make_chat_service
    monkeypatch.setattr(integration, 'make_chat_service', lambda local, **options: captured.append(local) or original_make(local, **options))
    app = entrypoint.create_app()
    assert app.state.local_fallback_enabled
    assert captured[-1].url == 'http://127.0.0.1:19181'
    assert captured[-1].token == 'test-private-token'
    (adapters['response'] / 'adapter_model.safetensors').write_text('changed adapter')
    with pytest.raises(ValueError, match='artifacts'):
        entrypoint.create_app()


def test_bare_passed_report_cannot_enable_real_gateway(monkeypatch, tmp_path):
    import experiments.qwen3_bonus.asgi as entrypoint
    monkeypatch.setattr(entrypoint, 'deployment_fingerprints', lambda config: {'actual': 'artifact-hashes'})
    report = tmp_path / 'release.json'
    report.write_text('{"passed":true}')
    with pytest.raises(ValueError, match='artifacts'):
        entrypoint.create_app(local_fallback_enabled=True, release_report=report)


def test_expired_original_budget_never_starts_local_generation(monkeypatch):
    import experiments.qwen3_bonus.integration as implementation
    local=Client('本地')
    primary=Client(TimeoutError('cloud timeout'))
    client=implementation.LocalFallbackClient(primary,local)
    token=implementation._model_deadline.set(implementation.time.monotonic()-1)
    try:
        with pytest.raises(TimeoutError,match='budget exhausted'):
            client.complete(messages())
        assert local.calls==[]
    finally:
        implementation._model_deadline.reset(token)
