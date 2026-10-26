"""Isolated loopback OpenAI gateway for one MLX base and two static PEFT LoRAs.

Importing this module never loads a model, starts a process, or contacts a server.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import math
import os
import secrets
import re
import signal
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from experiments.qwen3_bonus.common import ROOT, sha256

BASE_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
BASE_REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
MODELS = ('solace-emotion', 'solace-response')
MAX_BODY = 65536


def loopback_url(value: str) -> str:
    """Accept an unambiguous numeric IPv4 loopback origin, without redirects."""
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or
            parsed.username or parsed.password or parsed.path not in ('', '/') or
            parsed.query or parsed.fragment or parsed.port is None):
        raise ValueError('Expected http://127.0.0.1:<port> loopback origin')
    if not 1024 <= parsed.port <= 65535:
        raise ValueError('Port must be between 1024 and 65535')
    return f'http://127.0.0.1:{parsed.port}'


@dataclass(frozen=True)
class ServingConfig:
    model_path: str
    emotion_adapter: str
    response_adapter: str
    engine_executable: str = 'vllm'
    gateway_port: int = 18181
    engine_port: int = 18182
    base_model: str = BASE_MODEL
    base_revision: str = BASE_REVISION
    cold_wait_seconds: float = 10.0
    startup_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 90.0
    idle_seconds: float = 300.0
    poll_seconds: float = .25
    failure_cooldown_seconds: float = 30.0
    max_start_failures: int = 3
    local_token: str | None = None

    def __post_init__(self):
        if self.base_model != BASE_MODEL or self.base_revision != BASE_REVISION:
            raise ValueError('Only the pinned Qwen3 base is supported')
        for name in ('model_path', 'emotion_adapter', 'response_adapter'):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f'{name} must be a trusted absolute local path')
        for port in (self.gateway_port, self.engine_port):
            if type(port) is not int:
                raise ValueError('Ports must be integers')
            loopback_url(f'http://127.0.0.1:{port}')
        if self.gateway_port == self.engine_port:
            raise ValueError('Gateway and child need separate ports')
        if not 0 < self.cold_wait_seconds <= 10:
            raise ValueError('Cold request wait must be in (0, 10] seconds')
        for name in ('startup_timeout_seconds', 'request_timeout_seconds', 'idle_seconds', 'poll_seconds', 'failure_cooldown_seconds'):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not 0 < value < 86400:
                raise ValueError(f'Invalid {name}')
        if type(self.max_start_failures) is not int or not 1 <= self.max_start_failures <= 5:
            raise ValueError('Startup failure budget must be 1..5')
        if not self.engine_executable or self.engine_executable.startswith('-'):
            raise ValueError('Expected a trusted vllm executable')
        if self.local_token is not None and (not isinstance(self.local_token, str) or not self.local_token):
            raise ValueError('Token must be a nonempty string or null')

    @classmethod
    def from_file(cls, path: str | Path):
        """Configuration is selected by the operator, never by an HTTP request."""
        target = Path(path).expanduser().resolve()
        if target.stat().st_size > MAX_BODY:
            raise ValueError('Configuration too large')
        data = json.loads(target.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('Configuration must be an object')
        return cls(**data)

    @property
    def engine_url(self):
        return loopback_url(f'http://127.0.0.1:{self.engine_port}')


def validate_artifacts(config: ServingConfig) -> None:
    root = Path(config.model_path)
    metadata = json.loads((root / 'config.json').read_text())
    quantization = metadata.get('quantization', metadata.get('quantization_config', {}))
    if quantization.get('bits') != 4 or metadata.get('model_type') != 'qwen3':
        raise ValueError('Expected a Qwen3 MLX 4-bit safetensors base')
    if not any(root.glob('*.safetensors')) or any(root.glob('*.gguf')):
        raise ValueError('MLX safetensors required; GGUF plus LoRA is unsupported')
    provenance = json.loads((root / 'bonus_source.json').read_text())
    if (provenance.get('repo_id') != BASE_MODEL or provenance.get('revision') != BASE_REVISION or
            provenance.get('quantization_bits') != 4 or
            not _digest(provenance.get('source_inventory_sha256'))):
        raise ValueError('Converted base provenance does not match pinned revision')
    files = provenance.get('files')
    required = {'config.json', *(str(p.relative_to(root)) for p in root.glob('*.safetensors'))}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError('Base provenance must hash configuration and safetensors')
    for name, digest in files.items():
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError('Invalid base manifest filename')
        item = (root / name).resolve()
        if not item.is_relative_to(root.resolve()) or not _digest(digest) or sha256(item) != digest:
            raise ValueError('Converted base hash mismatch or escaping manifest path')
    for adapter, task in ((config.emotion_adapter, 'emotion'), (config.response_adapter, 'response')):
        path = Path(adapter)
        data = json.loads((path / 'adapter_config.json').read_text())
        if (data.get('peft_type') != 'LORA' or data.get('task_type') != 'CAUSAL_LM' or
                type(data.get('r')) is not int or not 1 <= data['r'] <= 8 or
                data.get('base_model_name_or_path') != BASE_MODEL or
                data.get('revision', BASE_REVISION) not in (None, BASE_REVISION) or
                not (path / 'adapter_model.safetensors').is_file()):
            raise ValueError('Expected pinned Qwen3 PEFT causal LM adapter, rank <= 8')
        manifest = json.loads((path / 'bonus_adapter.json').read_text())
        if (manifest.get('base_model') != BASE_MODEL or manifest.get('base_revision') != BASE_REVISION or
                manifest.get('task') != task or manifest.get('rank') != data['r'] or
                manifest.get('trained') is not True or
                manifest.get('adapter_config_sha256') != sha256(path / 'adapter_config.json') or
                manifest.get('alpha') != data.get('lora_alpha') or
                set(manifest.get('target_modules', [])) != set(data.get('target_modules', [])) or
                not _digest(manifest.get('training_result_sha256')) or
                manifest.get('adapter_weights_sha256') != sha256(path / 'adapter_model.safetensors')):
            raise ValueError('Adapter provenance, task, hyperparameters, or weight hash mismatch')


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def engine_argv(config: ServingConfig) -> list[str]:
    return [config.engine_executable, 'serve', config.model_path,
            '--host', '127.0.0.1', '--port', str(config.engine_port),
            '--enable-lora', '--lora-modules',
            f'solace-emotion={config.emotion_adapter}',
            f'solace-response={config.response_adapter}',
            '--max-loras', '2', '--max-lora-rank', '8',
            '--max-model-len', '2048', '--max-num-seqs', '1',
            '--max-num-batched-tokens', '512', '--gpu-memory-utilization', '0.45',
            '--no-enable-log-requests']


def child_environment() -> dict[str, str]:
    """Do not pass cloud keys, proxy variables, or arbitrary parent environment."""
    allowed = ('PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL', 'SYSTEMROOT')
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               VLLM_ALLOW_RUNTIME_LORA_UPDATING='False')
    return env


def port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        # Match the HTTP server's bind semantics after idle shutdown (TIME_WAIT).
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def owned_listener(pid: int, port: int) -> bool:
    """macOS lsof identifies the actual listener; only our process group is valid."""
    try:
        result = subprocess.run(['lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t'],
                                capture_output=True, text=True, timeout=2, check=False)
        listeners = {int(value) for value in result.stdout.split()}
        return bool(listeners) and all(os.getpgid(owner) == pid for owner in listeners)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


class EngineManager:
    def __init__(self, config: ServingConfig, *, client: httpx.AsyncClient | None = None,
                 validate_artifacts: bool = True):
        self.config = config
        self.client = client or httpx.AsyncClient(timeout=2, trust_env=False, follow_redirects=False)
        self._owns_client = client is None
        self._validate = validate_artifacts
        self.process = None
        self.state = 'stopped'
        self.serial = asyncio.Lock()
        self._lifecycle = asyncio.Lock()
        self._startup = None
        self._idle_task = None
        self._closed = False
        self.active = 0
        self.last_activity = time.monotonic()
        self._failures = 0
        self._retry_at = 0.0
        self._engine_token = secrets.token_urlsafe(32)

    @property
    def headers(self):
        return {'Authorization': f'Bearer {self._engine_token}'}

    def status(self):
        alive = self.process is not None and self.process.returncode is None
        return {'state': self.state, 'ready': self.state == 'ready' and alive,
                'active_requests': self.active, 'startup_failures': self._failures}

    async def _healthy(self) -> bool:
        if not self.process or self.process.returncode is not None:
            return False
        if not await asyncio.to_thread(owned_listener, self.process.pid, self.config.engine_port):
            return False
        try:
            response = await self.client.get(self.config.engine_url + '/v1/models', headers=self.headers)
            if response.status_code != 200:
                return False
            names = {item['id'] for item in response.json().get('data', [])}
            return set(MODELS).issubset(names)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            return False

    async def _stop_process(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.wait_for(process.wait(), timeout=5)

    async def _start(self):
        self.state = 'starting'
        try:
            await self._stop_process()
            if self._validate:
                await asyncio.to_thread(validate_artifacts, self.config)
            if not await asyncio.to_thread(port_is_free, self.config.engine_port):
                raise RuntimeError('Engine port belongs to another listener')
            env = child_environment()
            env['VLLM_API_KEY'] = self._engine_token
            log_path = ROOT / 'artifacts/local-deployment/engine.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'ab') as log:
                self.process = await asyncio.create_subprocess_exec(
                    *engine_argv(self.config), env=env, start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            deadline = time.monotonic() + self.config.startup_timeout_seconds
            while not self._closed and time.monotonic() < deadline:
                if await self._healthy():
                    self.state = 'ready'
                    self._failures = 0
                    self.last_activity = time.monotonic()
                    return
                if self.process.returncode is not None:
                    raise RuntimeError('Engine exited before readiness')
                await asyncio.sleep(self.config.poll_seconds)
            raise RuntimeError('Engine startup deadline exceeded')
        except asyncio.CancelledError:
            await self._stop_process()
            raise
        except Exception:
            self.state = 'failed'
            self._failures += 1
            self._retry_at = time.monotonic() + self.config.failure_cooldown_seconds
            await self._stop_process()

    async def ensure_ready(self) -> bool:
        if self._closed:
            return False
        async with self._lifecycle:
            if self.state == 'ready' and await self._healthy():
                return True
            if self._startup is None or self._startup.done():
                if self._failures >= self.config.max_start_failures or time.monotonic() < self._retry_at:
                    return False
                self._startup = asyncio.create_task(self._start())
            if self._idle_task is None:
                self._idle_task = asyncio.create_task(self._idle_watch())
            task = self._startup
        try:
            # A caller's timeout/cancellation never cancels the shared warmup.
            await asyncio.wait_for(asyncio.shield(task), timeout=self.config.cold_wait_seconds)
        except asyncio.TimeoutError:
            return False
        return self.status()['ready']

    async def acquire(self):
        await self.serial.acquire()
        self.active += 1
        self.last_activity = time.monotonic()

    def release(self):
        self.active -= 1
        self.last_activity = time.monotonic()
        self.serial.release()

    async def _idle_watch(self):
        while not self._closed:
            await asyncio.sleep(min(self.config.poll_seconds, self.config.idle_seconds / 2))
            async with self._lifecycle:
                if (self.state == 'ready' and self.active == 0 and
                        time.monotonic() - self.last_activity >= self.config.idle_seconds):
                    await self._stop_process()
                    self.state = 'stopped'

    async def close(self):
        self._closed = True
        tasks = [task for task in (self._startup, self._idle_task) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._stop_process()
        self.state = 'stopped'
        if self._owns_client:
            await self.client.aclose()


def validated_payload(body: Any) -> dict:
    allowed = {'model', 'messages', 'max_tokens', 'temperature', 'stream'}
    if not isinstance(body, dict) or set(body) - allowed:
        raise ValueError('Unsupported request fields')
    if body.get('model') not in MODELS:
        raise ValueError('Unknown model; use solace-emotion or solace-response')
    if type(body.get('stream', False)) is not bool:
        raise ValueError('stream must be boolean')
    messages = body.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        raise ValueError('Expected 1..64 text messages')
    for item in messages:
        if (not isinstance(item, dict) or set(item) != {'role', 'content'} or
                item['role'] not in ('system', 'user', 'assistant') or
                not isinstance(item['content'], str) or not item['content']):
            raise ValueError('Only text system/user/assistant messages supported')
    maximum = 16 if body['model'] == 'solace-emotion' else 256
    requested = body.get('max_tokens', maximum)
    temperature = body.get('temperature', 0 if maximum == 16 else .5)
    if type(requested) is not int or requested < 1:
        raise ValueError('max_tokens must be a positive integer')
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise ValueError('temperature must be in [0, 2]')
    return {**body, 'max_tokens': min(requested, maximum), 'temperature': temperature,
            'stream': body.get('stream', False)}


def create_gateway(config: ServingConfig, *, manager=None, client=None) -> FastAPI:
    manager = manager or EngineManager(config)
    upstream = client or httpx.AsyncClient(timeout=httpx.Timeout(config.request_timeout_seconds, connect=2),
                                           trust_env=False, follow_redirects=False)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await manager.close()
        if client is None:
            await upstream.aclose()

    app = FastAPI(title='Optional Qwen3 dual LoRA loopback gateway', lifespan=lifespan)
    app.state.engine_manager = manager

    @app.get('/health')
    async def health():
        return manager.status()

    @app.post('/v1/chat/completions')
    async def completions(request: Request):
        if config.local_token and not hmac.compare_digest(
                request.headers.get('authorization', '').encode('utf-8'), f'Bearer {config.local_token}'.encode('utf-8')):
            raise HTTPException(401, 'Unauthorized')
        raw = bytearray()
        try:
            async for part in request.stream():
                raw.extend(part)
                if len(raw) > MAX_BODY:
                    raise HTTPException(413, 'Request body too large')
            payload = validated_payload(json.loads(raw))
        except (ValueError, TypeError, UnicodeError, RecursionError):
            raise HTTPException(400, 'Malformed or unsupported chat request') from None
        acquired = False
        response = None
        try:
            requested_timeout=float(request.headers.get('x-solace-timeout',config.request_timeout_seconds))
            if not math.isfinite(requested_timeout) or requested_timeout<=0:
                raise HTTPException(400,'Invalid remaining request budget')
            request_deadline=time.monotonic()+min(requested_timeout,config.request_timeout_seconds)
            # Queueing and readiness share one cold-request budget. ensure_ready
            # shields its startup task, so expiry leaves background loading alive.
            async with asyncio.timeout(min(config.cold_wait_seconds,requested_timeout)):
                await manager.acquire()
                acquired = True
                if not await manager.ensure_ready():
                    raise HTTPException(503, 'Local model warming or unavailable')
            headers = getattr(manager, 'headers', {})
            upstream_request = upstream.build_request('POST', config.engine_url + '/v1/chat/completions',
                                                      json=payload, headers=headers)
            async with asyncio.timeout(max(.001,request_deadline-time.monotonic())):
                response = await upstream.send(upstream_request, stream=payload['stream'])
                if response.status_code in (400,413):
                    await response.aread()
                    error=response.json().get('error',{})
                    message=str(error.get('message','')).lower()
                    if (error.get('code') in ('context_length_exceeded','context_window_exceeded','max_context_length_exceeded')
                        or any(value in message for value in ('maximum context length','maximum model length','max_model_len'))):
                        return JSONResponse({'error':{'code':'context_length_exceeded',
                            'message':'Local model context window exceeded'}},status_code=400)
            response.raise_for_status()
            if not payload['stream']:
                data = response.json()
                if not isinstance(data, dict) or not isinstance(data.get('choices'), list) or not data['choices']:
                    raise ValueError('Invalid local completion')
                return JSONResponse(data)
            if not response.headers.get('content-type', '').startswith('text/event-stream'):
                raise ValueError('Expected SSE stream')
            released = False
            async def cleanup():
                nonlocal released
                if released:
                    return
                released = True
                try:
                    await response.aclose()
                finally:
                    manager.release()
            async def chunks():
                total = 0
                try:
                    async with asyncio.timeout(config.request_timeout_seconds):
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > 1024 * 1024:
                                raise ValueError('Local stream too large')
                            yield chunk
                except (httpx.HTTPError, TimeoutError, ValueError):
                    yield b'data: {"error":{"message":"Local generation interrupted","code":"local_unavailable"}}\n\ndata: [DONE]\n\n'
                finally:
                    await cleanup()
            # Generator owns the response and request lock until disconnected/done.
            acquired = False
            return StreamingResponse(chunks(), media_type='text/event-stream', background=BackgroundTask(cleanup))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, 'Local generation unavailable') from None
        finally:
            if acquired:
                try:
                    if response is not None:
                        await response.aclose()
                finally:
                    manager.release()

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    options = parser.parse_args()
    config = ServingConfig.from_file(options.config)
    import uvicorn
    uvicorn.run(create_gateway(config), host='127.0.0.1', port=config.gateway_port, workers=1)


if __name__ == '__main__':
    main()
