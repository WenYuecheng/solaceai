"""No GPU, model weights, or paid endpoints are used in these gateway tests."""
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from experiments.qwen3_bonus import serving


def config(**changes):
    values = dict(model_path='/tmp/test-mlx', emotion_adapter='/tmp/emotion', response_adapter='/tmp/response')
    values.update(changes)
    return serving.ServingConfig(**values)


def test_config_builds_static_dual_lora_and_rejects_external_urls():
    cfg = config()
    argv = serving.engine_argv(cfg)
    assert argv[0] == cfg.engine_executable
    assert '--enable-lora' in argv
    assert argv[argv.index('--max-loras') + 1] == '2'
    assert argv[argv.index('--max-lora-rank') + 1] == '8'
    assert argv[argv.index('--max-model-len') + 1] == '2048'
    assert 'solace-emotion=/tmp/emotion' in argv
    assert 'solace-response=/tmp/response' in argv
    for url in ('https://example.org', 'http://localhost:18182', 'http://127.0.0.1:18182/evil', 'http://user@127.0.0.1:18182'):
        with pytest.raises(ValueError):
            serving.loopback_url(url)


def test_child_environment_does_not_inherit_cloud_secrets(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'private')
    monkeypatch.setenv('VLLM_ALLOW_RUNTIME_LORA_UPDATING', 'True')
    env = serving.child_environment()
    assert 'OPENAI_API_KEY' not in env
    assert env['VLLM_ALLOW_RUNTIME_LORA_UPDATING'] == 'False'


class ReadyManager:
    def __init__(self):
        self.serial = asyncio.Lock()
        self.active = 0
        self.ready_calls = 0
    async def ensure_ready(self):
        self.ready_calls += 1
        return True
    async def close(self):
        pass
    def status(self):
        return {'ready': True}
    async def acquire(self):
        await self.serial.acquire()
        self.active += 1
    def release(self):
        self.active -= 1
        self.serial.release()


def test_gateway_bounds_and_auth():
    async def run():
        seen = []
        def upstream(req):
            seen.append(json.loads(req.content))
            return httpx.Response(200, json={'choices': [{'message': {'content': '焦虑'}}]})
        manager = ReadyManager()
        client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False)
        app = serving.create_gateway(config(local_token='secret'), manager=manager, client=client)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as c:
            body = {'model': 'solace-emotion', 'messages': [{'role': 'user', 'content': '紧张'}], 'max_tokens': 999}
            assert (await c.post('/v1/chat/completions', json=body)).status_code == 401
            c.headers['Authorization'] = 'Bearer secret'
            assert (await c.post('/v1/chat/completions', json=body)).status_code == 200
            assert seen[-1]['max_tokens'] == 16
            for extra in ({'model': 'other'}, {'adapter_path': '/tmp/evil'}, {'tools': []}, {'messages': []}, {'stream': 'yes'}):
                assert (await c.post('/v1/chat/completions', json={**body, **extra})).status_code == 400
            assert (await c.post('/v1/chat/completions', content=b'{')).status_code == 400
            assert (await c.post('/v1/chat/completions', content=b'x' * 70000)).status_code == 413
            assert manager.active == 0 and not manager.serial.locked()
        await client.aclose()
    asyncio.run(run())


def test_cold_gateway_503_and_upstream_failure_releases_lock():
    async def run():
        manager = ReadyManager()
        async def cold():
            return False
        manager.ensure_ready = cold
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(502)))
        app = serving.create_gateway(config(), manager=manager, client=client)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as c:
            body = {'model': 'solace-response', 'messages': [{'role': 'user', 'content': 'hello'}]}
            assert (await c.post('/v1/chat/completions', json=body)).status_code == 503
            async def ready():
                return True
            manager.ensure_ready = ready
            assert (await c.post('/v1/chat/completions', json=body)).status_code == 503
            assert manager.active == 0 and not manager.serial.locked()
        await client.aclose()
    asyncio.run(run())


def test_engine_start_coalesces_cold_wait_and_idle_shutdown(monkeypatch):
    async def run():
        starts = []
        class Process:
            pid = 987654
            returncode = None
            async def wait(self):
                self.returncode = 0
                return 0
        process = Process()
        async def spawn(*argv, **kwargs):
            starts.append((argv, kwargs))
            return process
        monkeypatch.setattr(serving.asyncio, 'create_subprocess_exec', spawn)
        monkeypatch.setattr(serving, 'port_is_free', lambda port: True)
        monkeypatch.setattr(serving, 'owned_listener', lambda pid, port: True)
        monkeypatch.setattr(serving.os, 'killpg', lambda pid, sig: setattr(process, 'returncode', 0))
        started = asyncio.get_running_loop().time()
        async def health(req):
            if asyncio.get_running_loop().time() - started < .08:
                return httpx.Response(503)
            return httpx.Response(200, json={'data': [{'id': n} for n in serving.MODELS]})
        client = httpx.AsyncClient(transport=httpx.MockTransport(health))
        manager = serving.EngineManager(config(cold_wait_seconds=.01, startup_timeout_seconds=.5, idle_seconds=.06, poll_seconds=.01), client=client, validate_artifacts=False)
        assert await asyncio.gather(manager.ensure_ready(), manager.ensure_ready()) == [False, False]
        await asyncio.sleep(.11)
        assert await manager.ensure_ready()
        assert len(starts) == 1
        await manager.acquire()
        await asyncio.sleep(.08)
        assert process.returncode is None
        manager.release()
        await asyncio.sleep(.10)
        assert process.returncode == 0
        await manager.close()
        await client.aclose()
    asyncio.run(run())


def test_busy_unrelated_listener_is_not_adopted(monkeypatch):
    async def run():
        monkeypatch.setattr(serving, 'port_is_free', lambda port: False)
        manager = serving.EngineManager(config(cold_wait_seconds=.01), validate_artifacts=False)
        assert not await manager.ensure_ready()
        assert manager.process is None
        assert manager.status()['state'] == 'failed'
        await manager.close()
    asyncio.run(run())


def test_stream_serializes_and_cancellation_closes_upstream():
    async def run():
        manager = ReadyManager()
        entered, closed = asyncio.Event(), asyncio.Event()
        class Endless(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                await asyncio.Event().wait()
            async def aclose(self):
                closed.set()
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Endless())))
        app = serving.create_gateway(config(), manager=manager, client=upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as c:
            task = asyncio.create_task(c.post('/v1/chat/completions', json={
                'model': 'solace-response', 'messages': [{'role': 'user', 'content': 'hello'}], 'stream': True}))
            await asyncio.wait_for(entered.wait(), 1)
            assert manager.active == 1 and manager.serial.locked()
            waiting = asyncio.create_task(manager.acquire())
            await asyncio.sleep(.01)
            assert not waiting.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(waiting, 1)
            assert closed.is_set()
            manager.release()
            assert manager.active == 0 and not manager.serial.locked()
        await upstream.aclose()
    asyncio.run(run())


def test_local_child_http_startup_and_idle_reap(monkeypatch):
    """A real lightweight child HTTP server verifies lifecycle, with no model import."""
    import os
    import socket
    import signal
    import textwrap
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    script = textwrap.dedent('''
        import http.server, json, os, sys
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                if self.headers.get('Authorization') != 'Bearer ' + os.environ['VLLM_API_KEY']:
                    self.send_response(401); self.end_headers(); return
                body = json.dumps({'data': [{'id': 'solace-emotion'}, {'id': 'solace-response'}]}).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
    ''')
    monkeypatch.setattr(serving, 'engine_argv', lambda config: [sys.executable, '-c', script, str(config.engine_port)])
    async def run():
        manager = serving.EngineManager(config(engine_port=port, cold_wait_seconds=2, startup_timeout_seconds=3,
                                                idle_seconds=.1, poll_seconds=.02), validate_artifacts=False)
        try:
            result = await asyncio.gather(manager.ensure_ready(), manager.ensure_ready())
            assert result == [True, True]
            process = manager.process
            assert process.returncode is None
            await asyncio.sleep(.3)
            assert process.returncode is not None and manager.process is None
        finally:
            await manager.close()
    asyncio.run(run())


def artifact_config(tmp_path):
    import hashlib
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({'model_type': 'qwen3', 'quantization': {'bits': 4}}))
    (model / 'model.safetensors').write_bytes(b'synthetic base for metadata test only')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (model / 'bonus_source.json').write_text(json.dumps({
        'repo_id': serving.BASE_MODEL, 'revision': serving.BASE_REVISION,
        'quantization_bits': 4, 'source_inventory_sha256': 'a' * 64,
        'files': {name: digest(model / name) for name in ('config.json', 'model.safetensors')}}))
    adapters = []
    for task in ('emotion', 'response'):
        adapter = tmp_path / task
        adapter.mkdir()
        (adapter / 'adapter_model.safetensors').write_bytes(b'synthetic adapter')
        (adapter / 'adapter_config.json').write_text(json.dumps({
            'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 8, 'lora_alpha': 16,
            'base_model_name_or_path': serving.BASE_MODEL, 'revision': serving.BASE_REVISION,
            'target_modules': ['q_proj', 'v_proj']}))
        (adapter / 'bonus_adapter.json').write_text(json.dumps({
            'base_model': serving.BASE_MODEL, 'base_revision': serving.BASE_REVISION,
            'task': task, 'rank': 8, 'alpha': 16, 'target_modules': ['q_proj', 'v_proj'],
            'adapter_weights_sha256': digest(adapter / 'adapter_model.safetensors'),
            'training_result_sha256': 'a' * 64, 'trained': True,
            'adapter_config_sha256': digest(adapter / 'adapter_config.json')}))
        adapters.append(str(adapter))
    return config(model_path=str(model), emotion_adapter=adapters[0], response_adapter=adapters[1])


def test_artifact_provenance_rejects_changed_weights_revision_rank_and_task(tmp_path):
    cfg = artifact_config(tmp_path)
    serving.validate_artifacts(cfg)
    metadata = Path(cfg.emotion_adapter) / 'bonus_adapter.json'
    original = metadata.read_text()
    for key, value in [('base_revision', 'wrong'), ('rank', 16), ('task', 'response'), ('trained', False), ('adapter_config_sha256', 'b' * 64)]:
        data = json.loads(original)
        data[key] = value
        metadata.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            serving.validate_artifacts(cfg)
        metadata.write_text(original)
    weights = Path(cfg.emotion_adapter) / 'adapter_model.safetensors'
    weights.write_bytes(b'tampered')
    with pytest.raises(ValueError):
        serving.validate_artifacts(cfg)


def test_base_provenance_checks_hashes_and_refuses_path_escape(tmp_path):
    cfg = artifact_config(tmp_path)
    metadata = Path(cfg.model_path) / 'bonus_source.json'
    data = json.loads(metadata.read_text())
    data['files']['../outside'] = 'a' * 64
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        serving.validate_artifacts(cfg)


def test_unconsumed_stream_response_background_releases_lock():
    async def run():
        manager = ReadyManager()
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(
            200, headers={'content-type': 'text/event-stream'}, content=b'data: [DONE]\n\n')))
        app = serving.create_gateway(config(), manager=manager, client=client)
        route = next(route for route in app.routes if getattr(route, 'path', None) == '/v1/chat/completions')
        from starlette.requests import Request
        body = json.dumps({'model': 'solace-response', 'messages': [{'role': 'user', 'content': 'hello'}], 'stream': True}).encode()
        async def receive():
            return {'type': 'http.request', 'body': body, 'more_body': False}
        request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions', 'headers': []}, receive)
        response = await route.endpoint(request)
        assert manager.serial.locked()
        assert response.background is not None
        await response.background()
        assert not manager.serial.locked() and manager.active == 0
        await client.aclose()
    asyncio.run(run())


def test_cancel_cold_caller_keeps_background_warmup(monkeypatch):
    async def run():
        manager = serving.EngineManager(config(cold_wait_seconds=.2), validate_artifacts=False)
        async def start():
            manager.state = 'starting'
            await asyncio.sleep(.05)
            manager.state = 'ready'
        monkeypatch.setattr(manager, '_start', start)
        caller = asyncio.create_task(manager.ensure_ready())
        await asyncio.sleep(.01)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await asyncio.sleep(.06)
        assert manager.state == 'ready' and not manager._startup.cancelled()
        await manager.close()
    asyncio.run(run())


def test_failed_engine_start_budget_prevents_respawn_loop(monkeypatch):
    async def run():
        starts = []
        async def spawn(*args, **kwargs):
            starts.append(args)
            raise FileNotFoundError('not installed')
        monkeypatch.setattr(serving.asyncio, 'create_subprocess_exec', spawn)
        monkeypatch.setattr(serving, 'port_is_free', lambda port: True)
        manager = serving.EngineManager(config(failure_cooldown_seconds=.01, cold_wait_seconds=.05), validate_artifacts=False)
        for _ in range(5):
            assert await manager.ensure_ready() is False
            await asyncio.sleep(.02)
        assert len(starts) == 3 and manager.status()['startup_failures'] == 3
        await manager.close()
    asyncio.run(run())


def test_queue_and_cold_start_share_one_request_deadline():
    async def run():
        manager=ReadyManager()
        original_acquire=manager.acquire
        async def queued():
            await asyncio.sleep(.06)
            await original_acquire()
        async def cold():
            await asyncio.sleep(.06)
            return False
        manager.acquire=queued
        manager.ensure_ready=cold
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(500)))
        app=serving.create_gateway(config(cold_wait_seconds=.08),manager=manager,client=client)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://test') as c:
            start=asyncio.get_running_loop().time()
            result=await c.post('/v1/chat/completions',json={'model':'solace-response','messages':[{'role':'user','content':'hi'}]})
            elapsed=asyncio.get_running_loop().time()-start
            assert result.status_code==503
            assert elapsed<.115, f'queue and startup consumed separate budgets: {elapsed}'
            assert manager.active==0 and not manager.serial.locked()
        await client.aclose()
    asyncio.run(run())


def test_local_context_overflow_keeps_original_compaction_signal():
    async def run():
        manager=ReadyManager()
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request:httpx.Response(400,
            json={'error':{'message':'maximum context length exceeded','code':'BadRequestError'}})))
        app=serving.create_gateway(config(),manager=manager,client=client)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://test') as c:
            response=await c.post('/v1/chat/completions',json={'model':'solace-response',
                'messages':[{'role':'user','content':'long context'}]})
            assert response.status_code==400
            assert response.json()['error']['code']=='context_length_exceeded'
            assert manager.active==0
        await client.aclose()
    asyncio.run(run())


def test_port_probe_allows_restart_after_connection_close_but_rejects_listener():
    import socket
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',0));listener.listen()
        port=listener.getsockname()[1]
        assert not serving.port_is_free(port)
        with socket.create_connection(('127.0.0.1',port)) as client:
            connection,_=listener.accept()
            connection.close()
            assert client.recv(1)==b''
    assert serving.port_is_free(port)
