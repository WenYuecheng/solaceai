"""SolaceAI 模型上下文协议服务与客户端 (Model Context Protocol - MCP Integration)"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from jsonschema import Draft202012Validator
from mcp import ClientSession, types
from mcp.server import Server
from mcp.shared.memory import create_client_server_memory_streams


@dataclass(frozen=True)
class RegisteredTool:
    """已注册业务工具规格元数据实体 (Registered Tool Metadata)"""
    name: str
    description: str
    schema: dict
    handler: Callable


class ToolRegistry:
    """受控业务工具注册与入参校验治理中心 (Tool Governance & Registry)"""

    def __init__(self):
        """初始化空的工具注册表"""
        self.tools: dict[str, RegisteredTool] = {}

    def register(self, name: str, description: str, schema: dict, handler: Callable) -> None:
        """注册一个新的业务工具并对入参 Schema 进行合规语法检查 (Register Tool)"""
        if name in self.tools:
            raise ValueError('Duplicate tool registration')
        Draft202012Validator.check_schema(schema)
        self.tools[name] = RegisteredTool(name, description, schema, handler)

    def schemas(self) -> dict[str, dict]:
        """导出所有已注册工具的 OpenAPI / JSON Schema 参数定义映射表 (Export Tool Schemas)"""
        return {name: {'description': tool.description, 'parameters': tool.schema}
                for name, tool in self.tools.items()}

    def call(self, name: str, arguments: dict) -> object:
        """安全校验入参并同步调用目标工具执行体 (Validated Tool Execution)"""
        tool = self.tools.get(name)
        if tool is None:
            raise PermissionError('Tool not authorized')
        errors = list(Draft202012Validator(tool.schema).iter_errors(arguments))
        if errors:
            raise ValueError('Invalid tool arguments: ' + errors[0].message)
        return tool.handler(arguments)


def create_tool_server(registry: ToolRegistry, dispatch: Callable) -> Server:
    """构造兼容官方 MCP 规范的工具服务端实例 (MCP Tool Server Factory)"""
    async def list_tools(context, params):
        """枚举当前注册表中的所有可用工具定义并按 MCP 协议格式返回。"""
        return types.ListToolsResult(tools=[types.Tool(name=tool.name,description=tool.description,input_schema=tool.schema) for tool in registry.tools.values()])
    async def call_tool(context, params):
        """处理 MCP 工具调用请求，经由 dispatch 转发后安全捕获异常并封装返回结果。"""
        try:
            value=await dispatch(params.name,params.arguments or {})
            return types.CallToolResult(content=[types.TextContent(type='text',text=json.dumps(value,ensure_ascii=False))])
        except Exception as exc:
            return types.CallToolResult(is_error=True,content=[types.TextContent(type='text',text=str(exc)[:500])])
    return Server('solaceai-private-tools',on_list_tools=list_tools,on_call_tool=call_tool)


class McpClient:
    """进程内全双工 MCP 协议长连接客户端 (Thread-Safe In-Memory MCP Client)"""

    def __init__(self, registry: ToolRegistry, timeout: float = 30.0):
        """初始化并启动 MCP 后台通信线程，等待握手初始化完成"""
        self.registry = registry
        self.requests = queue.Queue()
        self.call_lock = threading.Lock()
        self.timeout = timeout
        self.ready = threading.Event()
        self.failure = None
        self.thread = threading.Thread(target=self._thread, daemon=True, name='solaceai-mcp')
        self.thread.start()
        if not self.ready.wait(timeout):
            raise TimeoutError('MCP initialization timed out')
        if self.failure:
            raise RuntimeError('MCP initialization failed') from self.failure

    def _thread(self) -> None:
        """后台独立线程入口函数：启动专用 asyncio 事件循环并处理生命周期"""
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self.failure = exc
            self.ready.set()

    async def _serve(self) -> None:
        """异步服务主循环：建立内存全双工管道、握手初始化并枚举能力"""
        self.loop = asyncio.get_running_loop()
        self.stop = asyncio.Event()
        async def dispatch(name, arguments):
            """跨线程边界将 MCP 工具执行请求排队至工作线程并等待 Future 结果。"""
            response=concurrent.futures.Future()
            self.requests.put((name,arguments,response))
            return await asyncio.wrap_future(response)
        server=create_tool_server(self.registry,dispatch)
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            worker = asyncio.create_task(server.run(*server_streams, server.create_initialization_options()))
            try:
                async with ClientSession(*client_streams) as session:
                    self.session = session
                    await session.initialize()
                    listed = await session.list_tools()
                    self.discovered = {t.name:{'description':t.description or '', 'parameters':t.input_schema} for t in listed.tools}
                    self.ready.set()
                    await self.stop.wait()
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    def schemas(self) -> dict[str, dict]:
        """获取通过 MCP 协议握手枚举发现的全部可用工具 Schema 定义表"""
        return self.discovered

    def call(self, name: str, arguments: dict) -> object:
        """通过 MCP 管道跨线程调度执行指定工具 (Synchronous Call via MCP Pipeline)"""
        async def execute():
            """在后台事件循环中调用 session.call_tool 并解析响应正文。"""
            result = await self.session.call_tool(name, arguments=arguments)
            text = '\n'.join(item.text for item in result.content if hasattr(item,'text'))
            if result.is_error:
                raise ValueError(text[:500])
            return json.loads(text)
        with self.call_lock:
            future = asyncio.run_coroutine_threadsafe(execute(), self.loop)
            deadline = time.monotonic() + self.timeout
            while not future.done():
                if time.monotonic() >= deadline:
                    future.cancel()
                    raise TimeoutError('MCP tool timed out')
                try:
                    tool, args, response = self.requests.get(timeout=.02)
                except queue.Empty:
                    continue
                if response.cancelled():
                    continue
                try:
                    value = self.registry.call(tool,args)
                    if not response.cancelled(): response.set_result(value)
                except Exception as exc:
                    if not response.cancelled(): response.set_exception(exc)
                # 同步处理函数在调用方主线程上执行以保持数据库会话亲和性；
                # 一旦执行完成即交付其结果，绝不因协议确认包超时而错误重放已成功的业务副作用操作
                try:
                    return future.result(timeout=max(1.0, deadline - time.monotonic()))
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError('MCP response delivery timed out')
            return future.result()

    def close(self) -> None:
        """安全关闭客户端连接与后台通信线程 (Graceful Shutdown)"""
        if self.thread.is_alive():
            self.loop.call_soon_threadsafe(self.stop.set)
            self.thread.join(timeout=self.timeout)

    def __enter__(self) -> McpClient:
        """上下文管理器入口"""
        return self

    def __exit__(self, *exc) -> None:
        """上下文管理器退出，自动释放资源"""
        self.close()

