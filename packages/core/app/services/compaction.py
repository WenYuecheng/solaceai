"""DSH compaction policies and model-free projection, adapted under MIT.

Copyright (c) 2026 DeepSeek. See THIRD_PARTY_NOTICES.md.
Upstream: compaction-basic/config.ts, tool-result-pruner/config.ts and index.ts.

该模块实现了动态选择性历史压缩 (Dynamic Selective History / DSH) 算法与无模型投影机制：
1. 上下文预算数学建模 (CompactionPolicy):
   - input_tokens = window - output
   - threshold_tokens = floor(min(window * 0.8, input_tokens - headroom))
   - retain_tokens = floor(input_tokens * 0.16)
2. 工具结果智能无损剪裁 (prune_tool_result):
   - 针对超长观测数据（如大段抓取文本、原始返回），在不破坏 JSON 树形结构和关键主键元数据的前提下，
     采用 Head (4096 字符) + Marker + Tail (1024 字符) 双端保留策略截断冗余正文。
3. 对话轮次边界对齐剪枝 (balanced_history_cuts & history_cut):
   - 保证剪枝切分点只发生在完整会话轮次边界（Assistant 回复后且无悬挂 Tool Calls），严禁中断工具调用时序。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import httpx

PRUNE_MARKER = "\n\n[... tool result middle pruned ...]\n\n"
TEXT_FIELDS = {"content", "excerpt", "text", "stdout", "stderr", "output", "body"}


@dataclass(frozen=True)
class CompactionPolicy:
    """DSH 上下文预算与动态压缩策略配置。

    Attributes:
        window: 模型完整上下文窗口大小 (tokens)，默认 8192。
        output: 为模型输出保留的最大生成预算 (tokens)，默认 4096。
        headroom: 安全冗余裕量 (tokens)，默认 512，防止分词偏差导致的微小溢出。
        threshold_ratio: 触发压缩的软上限比例，固定为 0.8。
        retain_ratio: 压缩后保留的最近历史上下文比例，固定为 0.16。
    """

    window: int = 8192
    output: int = 4096
    headroom: int = 512
    threshold_ratio: float = 0.8
    retain_ratio: float = 0.16

    def __post_init__(self) -> None:
        """校验压缩参数的数学一致性与比率约束。"""
        if (self.threshold_ratio, self.retain_ratio) != (0.8, 0.16):
            raise ValueError("DSH ratios must remain 0.8 and 0.16")
        if (
            any(type(n) is not int for n in (self.window, self.output, self.headroom))
            or self.output <= 0
            or self.headroom <= 0
        ):
            raise ValueError("Invalid compaction token budgets")
        if self.threshold_tokens <= self.retain_tokens:
            raise ValueError(
                "Window must exceed output and headroom, with room for retention"
            )

    @property
    def input_tokens(self) -> int:
        """模型可用的最大输入 Prompt Token 预算: `window - output`。"""
        return self.window - self.output

    @property
    def threshold_tokens(self) -> int:
        """触发上下文压缩的 Token 门限值: `floor(min(window * 0.8, input_tokens - headroom))`。"""
        return math.floor(min(self.window * 0.8, self.input_tokens - self.headroom))

    @property
    def retain_tokens(self) -> int:
        """压缩后原文保留的最近轮次最低 Token 目标: `floor(input_tokens * 0.16)`。"""
        return math.floor(self.input_tokens * 0.16)


def prune_tool_result(value: Any) -> Any:
    """深度遍历并安全剪枝工具返回的超长文本内容。

    在不篡改 JSON 节点类型、主键 ID、错误信息和结构骨架的前提下，
    对超过 8192 字符的文本叶子节点执行头尾双向保留剪枝（前 4096 字符 + 中间省略标记 + 后 1024 字符）。

    Args:
        value: 原始工具返回值（可以是 dict, list, str 或标量）。

    Returns:
        Any: 剪裁后的安全数据结构深拷贝。
    """
    result = copy.deepcopy(value)
    fields = []

    def walk(node, parent=None, key=None):
        if isinstance(node, str):
            if (
                key in TEXT_FIELDS
                or key == "result"
                or (isinstance(key, int) and isinstance(parent, list))
            ):
                fields.append((parent, key, node))
        elif isinstance(node, dict):
            for k, v in node.items():
                if k not in {
                    "arguments",
                    "metadata",
                    "error",
                    "errors",
                    "source",
                    "sources",
                    "chunk_id",
                    "chunkId",
                    "chunk_uid",
                    "document_id",
                    "tenant_id",
                    "knowledge_base_id",
                    "status",
                    "url",
                    "name",
                }:
                    walk(v, node, k)
        elif isinstance(node, list):
            for k, v in enumerate(node):
                walk(v, node, k)

    if isinstance(result, str):
        return (
            result
            if len(result) <= 8192
            else result[:4096] + PRUNE_MARKER + result[-1024:]
        )
    walk(result)
    size = sum(len(text) for _, _, text in fields)
    if size <= 8192:
        return result
    offset = 0
    marked = False
    for parent, key, text in fields:
        head = text[: max(0, min(len(text), 4096 - offset))]
        tail = text[max(0, size - 1024 - offset):]
        omitted = len(head) + len(tail) < len(text)
        marker = PRUNE_MARKER if omitted and not marked else ""
        if omitted:
            marked = True
        parent[key] = head + marker + tail
        offset += len(text)
    return result


def evidence_view(value: Any) -> Any:
    """提取证据切片的领域核心字段视图，剥离 UI 展示冗余元数据。

    Args:
        value: 待投影的切片字典或切片列表。

    Returns:
        Any: 仅包含 chunk_id、source、content、section_path 等核心业务字段的轻量视图。
    """
    if isinstance(value, list):
        return [evidence_view(v) for v in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    if "chunk_id" not in value and "chunkId" not in value:
        return {k: evidence_view(v) for k, v in value.items()}
    keys = {
        "chunk_id",
        "chunkId",
        "chunk_uid",
        "source",
        "content",
        "excerpt",
        "page",
        "section_path",
        "document_id",
        "knowledge_base_id",
        "tenant_id",
        "version_id",
        "document_version",
        "error",
        "status",
    }
    result = {k: copy.deepcopy(v) for k, v in value.items() if k in keys}
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    for key in keys:
        if key not in result and key in metadata:
            result[key] = copy.deepcopy(metadata[key])
    return result


def balanced_history_cuts(messages: list[Any]) -> list[int]:
    """计算所有处于完整对话轮次边界的安全切割索引点。

    切割点必须满足：
    1. 必须位于助手完整回复与用户下一轮输入之间；
    2. 当前助手的所有原生工具调用 (`tool_calls`) 均已收敛并收到对应的 `tool` 观测响应 (`pending_calls == 0`)，
       严禁在工具调用中途阶段截断对话。

    Args:
        messages: 时序消息列表。

    Returns:
        list[int]: 满足严格轮次边界的消息索引列表。
    """
    def data(m):
        return m if isinstance(m, dict) else m.model_dump()

    cuts = []
    pending_calls = 0
    for i, message in enumerate(messages):
        entry = data(message)
        role = entry["role"]
        if (
            i
            and role == "user"
            and data(messages[i - 1])["role"] == "assistant"
            and pending_calls == 0
        ):
            cuts.append(i)
        if role == "assistant":
            pending_calls += len(entry.get("tool_calls", []))
        elif role == "tool":
            if not pending_calls:
                return []
            pending_calls -= 1
    return cuts


def history_cut(messages: list[Any], retain_tokens: int) -> int:
    """计算自适应历史裁剪的安全轮次切割点索引 (Compute Safe Dynamic History Cut Index)

    执行逻辑：
    1. 确保最新对话轮次至少满足 DSH 最近 Token 目标下限（该下限为最小值，而非绝对上限）；
    2. 向前回溯定位至完整的合法对话轮次边界（禁止截断在 assistant-tool 工具调用对中间）；
    3. 裁剪点之前的较旧历史消息交由工作记忆服务生成增量摘要，绝不发生静默丢弃；
    4. 当前未闭环的用户轮次会同时完整保留前一轮完整的上下文。

    Args:
        messages (list[Any]): 完整的历史消息列表。
        retain_tokens (int): 目标必须保留的最近 Token 预算下限。

    Returns:
        int: 安全切割点消息列表索引位置。
    """
    from app.services.context import token_count

    cuts = balanced_history_cuts(messages)
    if not cuts:
        return 0
    data = [m if isinstance(m, dict) else m.model_dump() for m in messages]
    if data[-1]['role'] == 'user':
        cuts = cuts[:-1]
    accumulated = 0
    keep_from = len(data)
    for index in range(len(data) - 1, -1, -1):
        accumulated += token_count(data[index])
        keep_from = index
        if accumulated >= retain_tokens:
            break
    # 向前安全回溯以确保命中完整轮次边界，避免将一次完整的工具交互对截断，
    # 或误删归属于保留回复的用户原始请求
    return max((cut for cut in cuts if cut <= keep_from), default=0)


def is_context_overflow(exc: Exception) -> bool:
    """精准判断异常是否属于大模型厂商返回的上下文超限错误 (Context Window Exceeded)。

    Args:
        exc: 捕获的异常对象。

    Returns:
        bool: 若为 HTTP 400/413 且明确带有 context_length_exceeded 等错误码，返回 True。
    """
    if not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code not in {
        400,
        413,
    }:
        return False
    try:
        code = exc.response.json().get("error", {}).get("code")
    except (ValueError, AttributeError):
        return False
    return code in {
        "context_length_exceeded",
        "context_window_exceeded",
        "max_context_length_exceeded",
    }
