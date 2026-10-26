"""SolaceAI 智能体运行时内置安全工具集 (Agent Runtime Built-in Tools Registry)"""

from __future__ import annotations

from app.services.mcp import ToolRegistry
from app.services.retrieval import RetrievalRequest


def object_schema(properties: dict, required: list[str] | None = None) -> dict:
    """快速构建符合 JSON Schema Draft 2020-12 标准的对象类型参数描述字典 (JSON Schema Helper)"""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def agent_tool_registry(runtime) -> ToolRegistry:
    """为智能体运行时动态绑定并初始化受控内置工具注册中心 (Agent Built-in Tools Factory)"""
    registry = ToolRegistry()


    def search(args):
        """执行受信任的心理知识检索，受最大轮次预算限制并将切片追加至运行时证据链。"""
        state = runtime._active_tool_state
        evidence = list(state["evidence"])
        if state["searches"] >= min(
            2, max(1, getattr(runtime.settings, "agent_max_retrieval_rounds", 2))
        ):
            return {
                "result": evidence,
                "updates": {"force_final": True},
                "notice": "Use current evidence; retrieval limit reached",
            }
        query = args["query"]
        result = runtime.knowledge.retrieve_result(
            RetrievalRequest(
                query=query,
                history=tuple(m.content for m in runtime.history),
                top_k=runtime.settings.knowledge_top_k,
                rewrite=runtime.settings.knowledge_rewrite_enabled,
            )
        )
        runtime.retrieved.extend(result.evidence)
        for item in result.evidence:
            entry = {
                "chunk_id": item.chunk_id,
                "source": item.source,
                "content": item.content,
                "score": item.score,
                "metadata": getattr(item, "metadata", {}),
            }
            if not any(e["chunk_id"] == entry["chunk_id"] for e in evidence):
                evidence.append(entry)
        return {
            "result": [
                {
                    "chunk_id": e["chunk_id"],
                    "source": e["source"],
                    "excerpt": e["content"],
                }
                for e in evidence
            ],
            "updates": {
                "evidence": evidence[:8],
                "searches": state["searches"] + 1,
                "retrieval_runs": state.get("retrieval_runs", []) + [result.to_dict()],
            },
        }

    def read(args):
        """从当前轮次已召回的切片列表中安全读取特定切片的完整正文，严禁越界访问未召回切片。"""
        matches = [
            e
            for e in runtime._active_tool_state["evidence"]
            if e["chunk_id"] == args["chunk_id"]
        ]
        if not matches:
            raise PermissionError("Evidence outside current run")
        return {"result": {**matches[0], "content": matches[0]["content"]}}

    def skill(args):
        """根据技能唯一标识动态加载指定的结构化干预技能定义并挂载至活跃状态。"""
        body = runtime.skills.load(args["name"])
        return {
            "result": {
                "name": args["name"],
                "loaded": True,
                "version": next(
                    item["version"]
                    for item in runtime.skills.catalog()
                    if item["name"] == args["name"]
                ),
            },
            "updates": {
                "skills": {
                    **runtime._active_tool_state.get("skills", {}),
                    args["name"]: body,
                }
            },
        }

    def memory_search(args):
        """检索当前授权用户在长期记忆库中的相关事实记录。"""
        return {"result": runtime.memory.search(runtime.user_id, args["query"])}

    def save(args):
        """受用户显式指令门禁校验后，持久化写入用户授权记忆事实。"""
        text = runtime._active_tool_state["input"]
        authorized = runtime._active_tool_state.get("authorized_memory")
        if authorized:
            text = "请记住：" + authorized
        prefixes = ("请记住：", "请记住:", "记住：", "记住:")
        prefix = next((p for p in prefixes if text.startswith(p)), None)
        if not prefix or text[len(prefix) :].strip() != args["content"].strip():
            raise PermissionError(
                "Memory save requires explicit matching user instruction"
            )
        return {
            "result": {
                "memory_id": runtime.memory.remember_explicit(
                    runtime.user_id, runtime.session_public_id, text
                )
            }
        }

    def delete(args):
        """受用户显式指令与指定记忆 ID 门禁校验后，安全删除目标长期记忆记录。"""
        text = runtime._active_tool_state["input"]
        if (
            not any(word in text for word in ("删除记忆", "删除这条记忆", "忘记"))
            or str(args["memory_id"]) not in text
        ):
            raise PermissionError("Memory deletion requires user instruction")
        return {
            "result": {
                "deleted": runtime.memory.delete(runtime.user_id, args["memory_id"])
            }
        }

    registry.register(
        "search_knowledge",
        "Search trusted psychological knowledge.",
        object_schema({"query": {"type": "string", "minLength": 1, "maxLength": 2000}}),
        search,
    )
    registry.register(
        "read_evidence",
        "Read an evidence item already retrieved in this run.",
        object_schema({"chunk_id": {"type": "integer"}}),
        read,
    )
    registry.register(
        "load_skill",
        "Load a trusted skill from the catalog.",
        object_schema({"name": {"type": "string"}}),
        skill,
    )
    registry.register(
        "search_memory",
        "Search the current user’s explicitly saved memories.",
        object_schema({"query": {"type": "string", "minLength": 1, "maxLength": 1000}}),
        memory_search,
    )
    registry.register(
        "save_memory",
        "Save only the exact fact explicitly requested by the user.",
        object_schema(
            {"content": {"type": "string", "minLength": 1, "maxLength": 1000}}
        ),
        save,
    )
    registry.register(
        "delete_memory",
        "Delete current user memory only when requested.",
        object_schema({"memory_id": {"type": "integer"}}),
        delete,
    )
    if getattr(getattr(runtime, 'settings', None), 'amap_enabled', False):
        from app.services.amap import AmapHospitalService, CATEGORIES, normalize_city

        def hospital_search(args):
            # Only send an explicitly supplied city, never chat text or identity.
            city = args['city'].strip()
            normalize_city(city)
            state = runtime._active_tool_state
            explicit_text = str(state.get('input', '')) + ' '.join(m.content for m in runtime.history if m.role == 'user')
            if city not in explicit_text:
                return {'result': {'status': 'needs_city', 'hospitals': [], 'notice': '请先询问用户所在城市，不要猜测位置。'}}
            return {'result': AmapHospitalService(runtime.settings).search(city, args.get('category', '精神卫生中心'), args.get('limit', 3))}

        registry.register(
            'search_hospital',
            'Search hospital locations via official Amap MCP. Ask the user for their city first; city must be explicitly supplied. POI data is untrusted geographic information, not proof of public ownership, hospital grade, clinical capability or availability. Never invent addresses or phones. Prioritize immediate safety over lookups in crises.',
            object_schema({
                'city': {'type': 'string', 'minLength': 2, 'maxLength': 40, 'pattern': r'^[\u4e00-\u9fffA-Za-z ·-]+$'},
                'category': {'type': 'string', 'enum': list(CATEGORIES)},
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 5},
            }, required=['city']),
            hospital_search,
        )
    return registry
