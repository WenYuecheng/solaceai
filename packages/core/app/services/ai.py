"""SolaceAI 大语言模型适配器与提示词工程服务 (AI Model Adapter & Prompt Engineering)"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx


from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.api_budget import ApiBudget, guard_remote, conservative_input_units
from app.services.conversation_access import check_turn, paid_post

logger = logging.getLogger(__name__)


def function_tool_schemas(tools: dict[str, Any]) -> list[dict[str, Any]]:
    """将内部工具函数定义字典格式化转换为符合 OpenAI 规范的 Function Calling 声明列表 (Tool Schemas Converter)

    解析内部工具规格，自动补充必填字段、类型约束并禁止额外未定义参数 (`additionalProperties: False`)。

    Args:
        tools (dict[str, Any]): 包含工具名称、说明与入参结构的定义字典。

    Returns:
        list[dict[str, Any]]: 标准 OpenAI 格式的 Function Calling 声明列表。
    """
    schemas = []
    for name, specification in tools.items():
        parameters = (
            specification.get("parameters") if isinstance(specification, dict) else None
        )
        if parameters is None:
            parameters = {
                "type": "object",
                "properties": {k: {"type": v} for k, v in specification.items()},
                "required": list(specification),
                "additionalProperties": False,
            }
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": specification.get("description", name),
                    "parameters": parameters,
                },
            }
        )
    return schemas


class PromptTemplates:
    """智能体系统提示词工程与模版工厂 (Prompt Engineering & Template Factory)"""

    @staticmethod
    def intent_prompt(history: list[AiMessage], user_input: str) -> list[AiMessage]:
        """构建用户意图识别分类提示词"""
        return [
            AiMessage(
                role="system",
                content=(
                    "你是一个用户意图分类器，只做意图识别，不回答问题。"
                    "只输出 CHAT、CONSULT、RISK 之一。CHAT 包含普通闲聊、学习、编程、作业、校园事务；"
                    "CONSULT 包含压力、焦虑、低落、失眠、情绪倾诉；RISK 包含自杀、自残、伤人或即时危险信号。"
                ),
            ),
            AiMessage(
                role="user",
                content=f"最近上下文：\n{format_history(history)}\n\n当前输入：\n{user_input}",
            ),
        ]

    @staticmethod
    def psychology_prompt(history: list[AiMessage], user_input: str) -> list[AiMessage]:
        """构建心理状态结构化诊断评估提示词"""
        return [
            AiMessage(
                role="system",
                content=(
                    "你负责分析校园心理健康消息。只返回严格 JSON："
                    '{"emotion":"NORMAL|ANXIETY|DEPRESSED|HIGH_RISK","emotionScore":0.0,'
                    '"risk":"LOW|MEDIUM|HIGH","summary":"short reason"}'
                    "emotionScore范围0到4，仅表示情绪困扰强度，不表示即时危险。"
                    "保研失利、家人批评、自责、焦虑或觉得失败本身不是HIGH。"
                    "HIGH用于明确自伤、伤人意图、计划、行动或当前人身危险；"
                    "缺乏即时危险证据的强烈困扰可为LOW或MEDIUM，先共情与澄清。"
                    "判断依据用户表达，不能把助手建议或知识资料中的危机用语当成用户危险。"
                    "明确否定自伤意图、课堂讨论或新闻转述且本人安全时，不因出现危机词就判HIGH。"
                    "否定与引用只适用于对应表达；若另有当前意图、计划、行动或危险证据，仍判HIGH。"
                ),
            ),
            AiMessage(
                role="user",
                content=f"最近上下文：\n{format_history(history)}\n\n当前输入：\n{user_input}",
            ),
        ]

    @staticmethod
    def answer_system_prompt(
        intent: IntentType,
        risk: RiskLevel,
        context: str,
        display_name: str,
        skill_context: str = "",
    ) -> AiMessage:
        """构建生成最终回复所需的系统指令提示词（System Prompt）"""
        if intent == IntentType.CHAT:
            content = (
                "你是 SolaceAI，一个面向学生的日常陪伴与校园生活助手。"
                "普通学习、编程、校园事务和通用问题请自然、准确、直接地回答。"
                "不要主动做心理测评，不要输出风险等级、心理标签、诊断结论或报告口吻。"
                f"学生显示名：{display_name}"
            )
            return AiMessage(role="system", content=content)
        crisis_rule = ""
        if risk == RiskLevel.HIGH:
            crisis_rule = (
                "\n高风险处理规则：先回应情绪，再关注当前安全；鼓励用户立刻联系身边可信任的人、"
                "学校辅导员/心理中心或当地紧急救助；不提供任何危险操作细节。"
            )
        content = (
            "你是 SolaceAI，一个面向学生的校园心理关怀智能体。"
            "回答要共情、谨慎、非评判，不诊断疾病，不开药，不替代持证心理咨询师。"
            "不要向学生输出风险等级、报告分数或后台标签。"
            "优先基于检索知识回答；知识不足时明确说明并给出安全通用建议。"
            f"\n学生显示名：{display_name}\n检索知识：\n{context}\n\n可用 skill 指引：\n{skill_context or '无'}{crisis_rule}"
        )
        return AiMessage(role="system", content=content)


class AiClient:
    """统一大模型客户端适配器"""

    def __init__(self, settings: Settings):
        """初始化大模型客户端"""
        self.settings = settings
        self.last_usage = {}
        self.last_response_model = ""

    def _provider_options(self) -> dict:
        """根据供应商特性动态装配供应商特定参数（如 DeepSeek 深度思考开关）。"""
        from urllib.parse import urlsplit

        enabled = getattr(self.settings, "openai_thinking_enabled", None)
        effort = self.settings.openai_reasoning_effort
        if (
            urlsplit(self.settings.openai_base_url).hostname == "open.bigmodel.cn"
            and effort
        ):
            return {"reasoning_effort": effort}
        if (
            urlsplit(self.settings.openai_base_url).hostname == "api.deepseek.com"
            and enabled is not None
        ):
            return {"thinking": {"type": "enabled" if enabled else "disabled"}}
        return {}

    def complete(
        self, messages: list[AiMessage], response_format: str | None = None
    ) -> str:
        """同步阻塞生成文本回复"""
        provider = self.settings.ai_provider.lower()
        if provider == "ollama":
            return self._ollama(messages, stream=False, response_format=response_format)
        if provider == "openai":
            return self._openai(messages, stream=False, response_format=response_format)
        return self._mock(messages)

    def decide(
        self, messages: list[AiMessage], tools: dict, event_sink=None, cancel_event=None
    ) -> dict:
        """原生函数调用决策接口：调用模型选择工具或返回结构化回答（实际工具执行由编排层沙箱负责）。"""
        if self.settings.ai_provider == "mock":
            # 离线确定性演示策略：仅用于单元测试与烟测，非大模型实际生成效果。
            prompt = json.loads(messages[-1].content)
            task = prompt.get("task", "")
            observations = prompt.get("observations", [])
            actions = [item.get("tool") for item in observations]
            if prompt.get("high_risk"):
                from app.agents.graph_runtime import SAFE_REPLY

                return {"answer": SAFE_REPLY}
            if (
                "search_memory" in tools
                and ("记住" in task or "偏好" in task)
                and "search_memory" not in actions
            ):
                return {"tool": "search_memory", "arguments": {"query": task}}
            if (
                "search_knowledge" in tools
                and (has_consult_signal(task) or "资料" in task)
                and "search_knowledge" not in actions
            ):
                return {"tool": "search_knowledge", "arguments": {"query": task}}
            evidence = []
            for observation in observations:
                if observation.get("tool") == "search_knowledge" and isinstance(
                    observation.get("result"), list
                ):
                    evidence.extend(observation["result"])
            if evidence:
                item = evidence[0]
                return {
                    "answer": f"可以先参考这条资料：[chunk:{item['chunk_id']}]\n{item.get('excerpt', '')[:300]}\n你希望先从哪一小步开始？"
                }
            return {
                "answer": "我在听。现在的资料还不足以给出具体判断，你最想先讨论哪件事？"
            }
        if self.settings.ai_provider != "openai":
            return json.loads(self.complete(messages, response_format="json_object"))
        schemas = function_tool_schemas(tools)
        payload = {
            "model": self.settings.openai_model,
            "messages": [m.model_dump() for m in messages],
            "tools": schemas,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
        }
        payload.update(self._provider_options())
        reservation = guard_remote(
            self.settings,
            "chat",
            self.settings.openai_model,
            self.settings.openai_base_url,
            conservative_input_units(messages)
            + len(json.dumps(schemas).encode("utf-8")),
            self.settings.ai_max_tokens,
        )
        if event_sink is not None:
            data = asyncio.run(self._stream_decision(payload, event_sink, cancel_event))
        else:
            response = paid_post(
                f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        usage = data.get("usage") or {}
        self.last_usage = usage
        self.last_response_model = data.get("model", "")
        if reservation and "prompt_tokens" in usage and "completion_tokens" in usage:
            ApiBudget(self.settings).commit(
                reservation, usage["prompt_tokens"], usage["completion_tokens"]
            )
        message = data["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if calls:
            if len(calls) > 6:
                raise ValueError("Tool batch exceeds decision budget")
            actions = []
            for item in calls:
                call = item["function"]
                arguments = json.loads(call["arguments"])
                if call["name"] not in tools or not isinstance(arguments, dict):
                    raise ValueError("Unauthorized tool batch")
                actions.append({"tool": call["name"], "arguments": arguments})
            result = actions[0]
            if len(actions) > 1:
                result["additional_tool_calls"] = actions[1:]
            return result
        content = message.get("content") or ""
        try:
            result = json.loads(content)
            if isinstance(result, dict) and ("answer" in result or "tool" in result):
                return result
        except ValueError:
            # 格式畸变的协议封装信封绝不可作为直接面向学生的回答；
            # 工具决策与外层 JSON 包装器坚决不能作为原始 JSON 泄露给前端
            if re.match(
                r'^\s*(?:```(?:json)?\s*)?\{\s*"(?:answer|tool|arguments)"\s*:', content
            ):
                raise ValueError("Malformed agent decision envelope")
        return {"answer": content}

    async def _stream_decision(self, payload, event_sink, cancel_event=None):
        async with asyncio.timeout(check_turn()):
            return await self._stream_decision_body(payload, event_sink, cancel_event)

    async def _stream_decision_body(self, payload, event_sink, cancel_event=None):
        """流式调度大语言模型决策请求与 Token 实时广播 (Stream Decision & Broadcast Tokens)

        执行逻辑：
        1. 注入流式选项与 Usage 统计参数，通过异步 HTTP 客户端发起 Server-Sent Events 流式连接；
        2. 实时解析 SSE 数据块并提取增量 content、tool_calls 与模型用量指标；
        3. 若模型直接输出回答，实时将增量文本 Token 推送至前端；若模型选择调用工具，则聚合工具参数并返回执行决策；
        4. 支持客户端 cancel_event 取消信号的即时响应与中断。

        Args:
            payload (dict): 发送给 LLM 的请求体字典。
            event_sink (Callable): 接收事件的异步/同步广播通道。
            cancel_event (Any, optional): 取消事件标志，默认为 None。

        Returns:
            dict: 包含 'answer' 或 ('tool', 'arguments') 的结构化决策字典。
        """
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        calls = {}
        parts = []
        usage = {}
        model = ""
        stream_plain = None
        pending = ""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60, connect=15),
            transport=httpx.AsyncHTTPTransport(retries=1),
        ) as client:
            async with client.stream(
                "POST",
                f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    check_turn()
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Cancelled model stream")
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    item = json.loads(raw)
                    model = item.get("model", model)
                    usage = item.get("usage") or usage
                    for choice in item.get("choices", []):
                        delta = choice.get("delta") or {}
                        for call in delta.get("tool_calls") or []:
                            index = call["index"]
                            target = calls.setdefault(
                                index, {"function": {"name": "", "arguments": ""}}
                            )
                            function = call.get("function") or {}
                            target["function"]["name"] += function.get("name") or ""
                            target["function"]["arguments"] += (
                                function.get("arguments") or ""
                            )
                        content = delta.get("content")
                        if content:
                            parts.append(content)
                            if not calls:
                                pending += content
                                if stream_plain is None and pending.strip():
                                    first = pending.lstrip()
                                    stream_plain = not first.startswith(("{", "`"))
                                if stream_plain:
                                    event_sink({"type": "token", "content": pending})
                                    pending = ""
        return {
            "model": model,
            "usage": usage,
            "choices": [
                {
                    "message": {
                        "content": "".join(parts),
                        "tool_calls": [calls[key] for key in sorted(calls)],
                    }
                }
            ],
        }

    async def stream(self, messages: list[AiMessage]):
        """异步流式 Token 生成器"""
        provider = self.settings.ai_provider.lower()
        if provider == "ollama":
            async for token in self._ollama_stream(messages):
                yield token
            return
        if provider == "openai":
            emitted = False
            async for token in self._openai_stream(messages):
                emitted = True
                yield token
            if not emitted:
                fallback = await asyncio.to_thread(self._openai, messages, stream=False)
                if not fallback:
                    raise RuntimeError(
                        "OpenAI-compatible provider returned an empty response"
                    )
                yield fallback
            return
        text = self._mock(messages)
        for chunk in split_text(text, 12):
            yield chunk

    def _ollama(
        self,
        messages: list[AiMessage],
        stream: bool,
        response_format: str | None = None,
    ) -> str:
        """调用本地 Ollama HTTP API 执行同步阻塞式大模型推理"""
        payload = {
            "model": self.settings.ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "options": {
                "temperature": self.settings.ai_temperature,
                "num_predict": self.settings.ai_max_tokens,
            },
        }
        if response_format == "json_object":
            payload["format"] = "json"
        response = paid_post(
            f"{self.settings.ollama_base_url}/api/chat", json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def _ollama_stream(self, messages: list[AiMessage]):
        """调用本地 Ollama HTTP API 执行异步长连接流式推理 (Async Generator)"""
        payload = {
            "model": self.settings.ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            "options": {
                "temperature": self.settings.ai_temperature,
                "num_predict": self.settings.ai_max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST", f"{self.settings.ollama_base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    check_turn()
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token

    def _openai(
        self,
        messages: list[AiMessage],
        stream: bool,
        response_format: str | None = None,
    ) -> str:
        """调用兼容 OpenAI 规范的 HTTP API 执行非流式推理"""
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.openai_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
            "stream": stream,
        }
        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        payload.update(self._provider_options())
        reservation = guard_remote(
            self.settings,
            "chat",
            self.settings.openai_model,
            self.settings.openai_base_url,
            conservative_input_units(messages),
            self.settings.ai_max_tokens,
        )
        response = paid_post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        self.last_usage = usage
        self.last_response_model = data.get("model", "")
        if reservation and "prompt_tokens" in usage and "completion_tokens" in usage:
            ApiBudget(self.settings).commit(
                reservation, usage["prompt_tokens"], usage["completion_tokens"]
            )
        return data["choices"][0]["message"]["content"]

    async def _openai_stream(self, messages: list[AiMessage]):
        async with asyncio.timeout(check_turn()):
            async for token in self._openai_stream_body(messages):
                yield token

    async def _openai_stream_body(self, messages: list[AiMessage]):
        """调用兼容 OpenAI 规范的 HTTP API 执行异步流式推理"""
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.openai_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
            "stream": True,
        }
        reservation = guard_remote(
            self.settings,
            "chat",
            self.settings.openai_model,
            self.settings.openai_base_url,
            conservative_input_units(messages),
            self.settings.ai_max_tokens,
        )
        payload["stream_options"] = {"include_usage": True}
        payload.update(self._provider_options())
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    check_turn()
                    if not line.startswith("data: "):
                        continue
                    raw = line.removeprefix("data: ").strip()
                    if raw == "[DONE]":
                        break
                    data = json.loads(raw)
                    usage = data.get("usage") or {}
                    if (
                        reservation
                        and "prompt_tokens" in usage
                        and "completion_tokens" in usage
                    ):
                        ApiBudget(self.settings).commit(
                            reservation,
                            usage["prompt_tokens"],
                            usage["completion_tokens"],
                        )
                    choices = data.get("choices") or []
                    token = (
                        choices[0].get("delta", {}).get("content", "")
                        if choices
                        else ""
                    )
                    if token:
                        yield token

    def _mock(self, messages: list[AiMessage]) -> str:
        """确定性 Mock 响应生成桩函数（用于纯单测与离线断网沙箱环境）"""
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        system = " ".join(m.content for m in messages if m.role == "system")
        if "上下文选择器" in system:
            context = json.loads(last)
            task = context["task"]
            intent = (
                "RISK"
                if has_high_risk_signal(task)
                else "CONSULT"
                if has_consult_signal(task)
                else "CHAT"
            )
            names = {skill["name"] for skill in context["skills"]}
            selected = (
                ["supportive_response_baseline"]
                if intent == "CONSULT" and "supportive_response_baseline" in names
                else []
            )
            return json.dumps(
                {
                    "intent": intent,
                    "selected_skills": selected,
                    "selected_memory_ids": [],
                }
            )
        if "严格 JSON" in system:
            if has_high_risk_signal(last):
                return '{"emotion":"HIGH_RISK","emotionScore":4.0,"risk":"HIGH","summary":"检测到明确高风险表达"}'
            if has_consult_signal(last):
                return '{"emotion":"ANXIETY","emotionScore":2.5,"risk":"LOW","summary":"检测到压力或情绪求助表达"}'
            return '{"emotion":"NORMAL","emotionScore":0.0,"risk":"LOW","summary":"未检测到明显风险信号"}'
        if "意图分类器" in system:
            if has_high_risk_signal(last):
                return "RISK"
            if has_consult_signal(last):
                return "CONSULT"
            return "CHAT"
        if "high_risk_safety_plan" in system and has_high_risk_signal(last):
            return "我听到你现在已经痛苦到觉得撑不下去了。现在最重要的是先让你不要一个人扛：请马上联系身边可信任的人，或者直接联系辅导员、学校心理中心、校园保卫/当地紧急服务。接下来 10 分钟，请先把自己移到有人在的地方，并把可能伤害自己的东西放远一点。如果可以，回我一句：你现在身边有没有可以马上联系或走过去找的人？"
        if "当前由 ResponseAgent 以 support mode" in system:
            return "我听到你最近压力很大，还影响到了睡眠，这种状态确实会让人很消耗。你可以先做两件小事：今晚把最担心的事情写成清单，先只选一个最小步骤处理；睡前 30 分钟把手机和学习任务放远一点，用缓慢呼吸或热水澡帮身体降下来。如果这种失眠持续一周以上，建议联系学校心理中心或辅导员一起看一看。"
        if "当前由 ResponseAgent 以 normal_chat mode" in system:
            return "我在。这个问题可以直接拆开来看，我们先从你最想解决的那一部分开始。"
        if "ContextAgent" in system and "SUFFICIENT" in system:
            return "SUFFICIENT"
        if "ContextAgent" in system:
            return last[:40] or "校园心理支持"
        return "我在。先把你现在最具体的困扰说出来，我们可以一步一步拆开。如果情况已经影响安全，请马上联系身边可信任的人或学校心理中心。"


class FallbackAiClient:
    """高可用双机热备大模型客户端包装器 (High-Availability Fallback AI Client)"""

    def __init__(self, primary: AiClient, fallback: AiClient):
        """初始化高可用双机热备客户端"""
        self.primary = primary
        self.fallback = fallback
        self.used_fallback = False

    def complete(
        self, messages: list[AiMessage], response_format: str | None = None
    ) -> str:
        """优先调用主模型执行同步推理，捕获异常后透明切换至备用模型"""
        try:
            self.used_fallback = False
            return self.primary.complete(messages, response_format=response_format)
        except Exception as exc:
            from app.services.compaction import is_context_overflow

            if is_context_overflow(exc):
                raise
            self.used_fallback = True
            logger.warning(
                "Primary AI completion failed; using fallback model: %s", exc
            )
            return self.fallback.complete(messages, response_format=response_format)

    async def stream(self, messages: list[AiMessage]):
        """执行流式大模型推理；仅在尚未产出首个 Token 时允许安全切流至备用模型"""
        emitted = False
        try:
            self.used_fallback = False
            async for token in self.primary.stream(messages):
                emitted = True
                yield token
            return
        except Exception as exc:
            if emitted:
                raise
            self.used_fallback = True
            logger.warning(
                "Primary AI stream failed before output; using fallback model: %s", exc
            )
        async for token in self.fallback.stream(messages):
            yield token


def format_history(history: list[AiMessage]) -> str:
    """将多轮对话历史消息对象序列格式化为纯文本上下文提示词"""
    if not history:
        return "无"
    return "\n".join(f"{m.role}: {m.content}" for m in history[-20:])


# 静态敏感词特征库
HIGH_RISK_WORDS = [
    "自杀",
    "自残",
    "不想活",
    "结束生命",
    "伤害自己",
    "轻生",
    "suicide",
    "kill myself",
    "self harm",
]
CONSULT_WORDS = [
    "焦虑",
    "抑郁",
    "压力",
    "失眠",
    "难过",
    "崩溃",
    "痛苦",
    "无助",
    "心理",
    "咨询",
    "anxious",
    "depress",
    "stress",
]


# Match only explicit denials and educational references. Never downgrade a model's
# independent assessment, or let one safe clause cancel another dangerous clause.
_RISK_TERMS = r"(?:自杀|自残|不想活|结束生命|伤害自己|轻生|suicide|kill myself|self harm)"
_CRITICAL_ACTIONS = (
    "已经吞了药", "已经吃了大量药", "已经割腕", "正在割腕", "准备跳下去",
    "站在楼顶", "站在天台", "写好了告别信", "准备彻底消失", "准备好遗书",
    "已经服毒", "准备跳楼",
)
_NEGATED_RISK = re.compile(
    r"(?:没有|并无|从未|不会|不打算|不准备|不想|不愿意|不考虑|不曾|没想过)"
    r"\s*(?:任何|再也|再)?\s*(?:想过|想要|想|要|去|会)?\s*" + "(?:" + _RISK_TERMS + "|" + "|".join(_CRITICAL_ACTIONS) + ")"
    + r"(?:的)?(?:想法|念头|意图|计划|冲动|行为)?"
    r"|\b(?:do not|don't|never|will not|won't)\s+(?:want to\s+)?"
      r"(?:kill myself|self harm|commit suicide)\b"
)
_REFERENCE_CONTEXT = re.compile(r"课堂|作业|新闻|报道|论文|小说|电影|科普|资料|预防|教材|案例|课程")
_SAFE_CONTEXT = re.compile(r"我(?:现在|目前)?(?:很)?安全|不是我的经历|没有.{0,8}(?:想法|意图|念头)")
_PERSONAL_INTENT = re.compile(
    r"(?:我|自己).{0,8}(?:想|要|准备|打算|决定).{0,6}" + _RISK_TERMS
)


def has_high_risk_signal(text: str) -> bool:
    """Detect current danger while scoping explicit denials/references locally.

    This rule only bypasses assessment on positive signals. A negative result
    still goes through the configured model, which may independently return HIGH.
    """
    normalized = text.lower()
    # Double negatives are ambiguous: retain the conservative original signal.
    if re.search(r"(?:不是|并非|不能说|不代表)\s*(?:没有|不想|不会)", normalized):
        return any(word in normalized for word in HIGH_RISK_WORDS)
    remaining = _NEGATED_RISK.sub("", normalized)
    if any(word in remaining for word in _CRITICAL_ACTIONS):
        return True
    # A denial does not cancel a subsequent concrete intention referring back to it.
    if (remaining != normalized and
            re.search(r"(?:但|可是|不过).{0,8}(?:我)?(?:打算|准备|决定).{0,8}(?:试一次|动手)", remaining)):
        return True
    safe_reference = bool(_SAFE_CONTEXT.search(normalized))
    for clause in re.split(r"[，,。；;！!？?\n]", remaining):
        if (safe_reference and _REFERENCE_CONTEXT.search(clause)
                and not _PERSONAL_INTENT.search(clause)):
            continue
        if any(word in clause for word in HIGH_RISK_WORDS + list(_CRITICAL_ACTIONS)):
            return True
    return False


def has_consult_signal(text: str) -> bool:
    """基于关键词特征库检查输入文本中是否包含心理情绪困扰或咨询诉求信号"""
    normalized = text.lower()
    return any(word in normalized for word in CONSULT_WORDS)


def split_text(text: str, size: int) -> Iterable[str]:
    """将连续文本按照固定字符长度进行分块切片（用于 Mock 环境模拟打字机流式输出）"""
    for index in range(0, len(text), size):
        yield text[index : index + size]


@dataclass(frozen=True)
class AgentModelProfile:
    """智能体模型配置规格实体 (Agent Model Profile)"""

    provider: str
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class AgentModelRoute:
    """主备模型路由配置实体 (Agent Model Routing Strategy)"""

    primary: AgentModelProfile
    fallback: AgentModelProfile | None = None


def default_response_route(settings: Settings) -> AgentModelRoute:
    """基于系统全局配置构造默认的主备模型路由策略 (Default Response Routing Strategy)"""
    provider = "mock" if settings.ai_provider == "mock" else "openai"
    return AgentModelRoute(
        AgentModelProfile(
            provider,
            "mock" if provider == "mock" else settings.openai_model,
            settings.ai_temperature,
            settings.ai_max_tokens,
        )
    )


def client_for_route(
    settings: Settings, route: AgentModelRoute
) -> AiClient | FallbackAiClient:
    """根据模型路由策略实例化具备容灾能力的客户端对象 (Route-Aware AI Client Factory)"""

    def build(profile: AgentModelProfile) -> AiClient:
        """根据指定的模型描述配置构造单实例大模型客户端。"""
        options = settings.model_copy(
            update={
                "ai_provider": profile.provider,
                "ai_temperature": profile.temperature,
                "ai_max_tokens": profile.max_tokens,
                "openai_model"
                if profile.provider == "openai"
                else "ollama_model": profile.model,
            }
        )
        return AiClient(options)

    primary = build(route.primary)
    return (
        primary
        if route.fallback is None
        else FallbackAiClient(primary, build(route.fallback))
    )


def resolve_model_dir(settings: Settings) -> Path:
    """解析微调模型资产存储目录的物理绝对路径。

    Args:
        settings (Settings): 系统全局配置对象。

    Returns:
        Path: 解析后的绝对目录路径对象。
    """
    path = Path(settings.finetuned_model_dir)
    return path if path.is_absolute() else settings.project_root / path


def finetuned_model_status(settings: Settings) -> dict[str, Any]:
    """检查并获取当前微调模型资产（GGUF 权重与 Modelfile）的就绪状态与元数据。

    Args:
        settings (Settings): 系统全局配置对象。

    Returns:
        dict[str, Any]: 包含模型名称、存储路径、GGUF 文件大小与就绪标志的状态字典。
    """
    root = settings.project_root
    model_dir = resolve_model_dir(settings)
    gguf_path = model_dir / settings.finetuned_model_file
    modelfile_path = model_dir / "Modelfile"
    return {
        "name": settings.finetuned_model_name,
        "directory": str(model_dir.relative_to(root)) if model_dir.is_relative_to(root) else str(model_dir),
        "ggufFile": settings.finetuned_model_file,
        "ggufExists": gguf_path.exists(),
        "ggufSizeBytes": gguf_path.stat().st_size if gguf_path.exists() else 0,
        "modelfileExists": modelfile_path.exists(),
        "ollamaCreateCommand": "scripts/create-finetuned-model.sh",
    }

