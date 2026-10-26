"""SolaceAI 知识引用标记提取与证据快照服务 (Knowledge Citations & Evidence Snapshots)

本模块负责从大模型生成的回复文本中精准解析与关联知识库切片引用：
1. 代码块与栅栏文本隔离：剔除 Markdown ``` 或 ~~~ 代码块内容，防范示例代码中的标记造成误检；
2. 引用标记提取：识别正文叙述中形如 `[chunk:123]` 的标准切片引用标记；
3. 证据快照载荷组装：根据检索到的证据切片集合，组装结构化的引用载荷（包含来源文件名、切片 ID、
   章节路径、摘要预览及模型实际是否引用的布尔标记 `cited`）。
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath


def prose_without_fences(answer: str | None) -> str:
    """剔除 Markdown 代码块栅栏内的文本，仅保留正文叙述内容 (Strip Code Block Fences)

    避免回答中包含的代码样例或示例引用被错误识别为模型对外部真实知识库的引用。

    Args:
        answer (str | None): 大模型生成的原始回复正文。

    Returns:
        str: 剥离所有 ``` 或 ~~~ 代码块包裹内容后的纯文本叙述字符串。
    """
    lines = []
    fence = None
    for line in (answer or '').splitlines():
        stripped = line.lstrip()
        marker = '```' if stripped.startswith('```') else '~~~' if stripped.startswith('~~~') else None
        if marker:
            if fence is None:
                # 进入代码块栅栏
                fence = marker
            elif marker == fence:
                # 遇到匹配的结束栅栏
                fence = None
            continue
        # 仅在处于代码块外部时保留行内容
        if fence is None:
            lines.append(line)
    return '\n'.join(lines)


def citation_payloads(evidence: list[dict], answer: str | None) -> list[dict]:
    """根据正文引用标记提取并组装标准知识切片引用快照载荷 (Build Standard Citation Payloads)

    执行逻辑：
    1. 提取正文中不在代码栅栏内的所有 `[chunk:ID]` 编号集合；
    2. 遍历本轮候选召回的知识切片列表；
    3. 解析其元数据中的标题、章节大纲路径及截断预览文本；
    4. 比对切片 ID 是否出现在上述集合中，标记 `cited: True/False`；
    5. 返回供前端气泡渲染与后端审计存储的引用对象列表。

    Args:
        evidence (list[dict]): 当前轮次检索召回的知识库候选切片字典列表。
        answer (str | None): 最终生成的回复文本。

    Returns:
        list[dict]: 格式化完备的切片引用快照列表。
    """
    # 提取纯正文中出现的全部切片 ID 字符串集合
    cited_ids = set(re.findall(r'\[chunk:(\d+)\]', prose_without_fences(answer)))
    payloads = []

    for row in evidence:
        metadata = row.get('metadata') or {}
        source = row.get('source', '')
        chunk_id = row.get('chunk_id', row.get('chunkId'))
        content = row.get('content') or row.get('preview') or ''
        section = metadata.get('section_path', row.get('sectionPath', []))
        if isinstance(section, str):
            section = [section]

        payloads.append({
            **row,
            'source': source,
            'chunk_id': chunk_id,
            'chunkId': chunk_id,
            'content': content,
            'metadata': metadata,
            'title': metadata.get('title') or row.get('title') or PurePosixPath(source).stem or '参考资料',
            'preview': content[:240],
            'sectionPath': list(section or []),
            'cited': str(chunk_id) in cited_ids,
        })

    return payloads
