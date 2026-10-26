"""SolaceAI 确定性原生文档结构化切片服务 (Deterministic Native Document Chunking)

本模块实现了携带父子上下文 (Parent/Child Context) 的高质量文档分块策略：
1. 确定性结构化切片：Markdown 与通用解析版面元素共用底层 `structural_chunks` 分块算法；
2. 保持阅读时序与层级出处：精确识别标题面包屑层级 (`section_path`)、表格说明与 PDF 原始物理页码；
3. 轻量自研无需外部 Tokenizer 模型：纯算法原生切片，规避重型深度学习 Tokenizer 的冷启动与显存开销；
4. 父子块双重视野：子切片用于高精度密集向量与词法匹配，父切片原文 (`parent_content`) 用于为大模型提供宏观背景。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings


@dataclass(frozen=True)
class StructuralChunk:
    """结构化知识切片数据传输对象（不可变）(Immutable Structural Chunk DTO)

    Attributes:
        content (str): 当前子切片的文本正文。
        document_id (str): 来源文档的全局唯一标识 SHA-256。
        document_version (str): 文档内容版本的 SHA-256 哈希值。
        section_path (tuple[str, ...]): 切片所属的标题面包屑层级元组 (如 ("第一章", "1.1 背景"))。
        parent_id (str): 父级逻辑段落的确定性 SHA-256 标识符。
        parent_content (str): 父级逻辑段落完整原文（用于提供扩充上下文）。
        page (int | None, optional): 切片在原始分页文档 (PDF) 中的页码。
        captions (tuple[str, ...], optional): 关联的图表说明文字元组。
        element_types (tuple[str, ...], optional): 切片包含的版面元素类型元组。
    """
    content: str
    document_id: str
    document_version: str
    section_path: tuple[str, ...]
    parent_id: str
    parent_content: str
    page: int | None = None
    captions: tuple[str, ...] = ()
    element_types: tuple[str, ...] = ()

    def __post_init__(self):
        """保证集合属性类型统一转换为不可变 tuple。"""
        if isinstance(self.section_path, list):
            object.__setattr__(self, 'section_path', tuple(self.section_path))
        if isinstance(self.captions, list):
            object.__setattr__(self, 'captions', tuple(self.captions))
        if isinstance(self.element_types, list):
            object.__setattr__(self, 'element_types', tuple(self.element_types))


def structural_chunks(
    content: str,
    source: str,
    size: int = 512,
    overlap: int = 64,
    parent_size: int | None = None,
) -> list[StructuralChunk]:
    """基于 Markdown 标题层级与自然段落边界切分结构化知识切片 (Markdown Structural Chunking)

    算法步骤：
    1. 逐行扫描正文，维护 Markdown 标题栈（1~6 级），自动剥离更深级别的旧标题；
    2. 检测 ``` 围栏代码块，确保代码块内部的空行与 # 不被误认为段落切分点或标题；
    3. 将自然段落聚合并按 `parent_size`（若指定）在标点句末边界处进行父段落折叠；
    4. 对每个父段落采用滑动窗口 (`size - overlap`) 步长切割出子切片，并记录标题路径与父哈希。

    Args:
        content (str): 待切分的完整 Markdown 文本。
        source (str): 来源文件名或源文档路径。
        size (int, optional): 子切片最大字符长度，默认 512。
        overlap (int, optional): 相邻切片重叠字符数，默认 64。
        parent_size (int | None, optional): 父段落最大字符长度，默认 None。

    Returns:
        list[StructuralChunk]: 结构化切片列表。

    Raises:
        ValueError: 当参数不满足 0 <= overlap < size 或 parent_size < size 时抛出。
    """
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError('Require 0 <= overlap < chunk size')
    if parent_size is not None and parent_size < size:
        raise ValueError('Parent size must be at least child size')

    document_id = hashlib.sha256(source.encode()).hexdigest()
    version = hashlib.sha256(content.encode()).hexdigest()
    headings: list[tuple[int, str]] = []
    paragraphs, lines = [], []
    fenced = False

    def flush():
        """将当前行缓存池归并为一个完整的逻辑段落，并挂载当前有效的标题层级面包屑。"""
        text = '\n'.join(lines).strip()
        if text:
            paragraphs.append((tuple(title for _, title in headings), text))
        lines.clear()

    # 1. 逐行解析 Markdown 标题大纲与段落
    for line in content.splitlines():
        if line.lstrip().startswith('```'):
            fenced = not fenced
        match = re.match(r'^(#{1,6})\s+(.+?)\s*#*$', line) if not fenced else None
        if match:
            flush()
            level = len(match[1])
            headings[:] = [(n, t) for n, t in headings if n < level]
            headings.append((level, match[2]))
        elif not line.strip() and not fenced:
            flush()
        else:
            lines.append(line)
    flush()

    # 2. 若配置了 parent_size，在自然句子标点边界执行父段落截断与约束
    if parent_size:
        bounded = []
        for path, paragraph in paragraphs:
            start = 0
            while start < len(paragraph):
                end = min(start + parent_size, len(paragraph))
                if end < len(paragraph):
                    boundary = max(
                        paragraph.rfind(mark, start + parent_size // 2, end)
                        for mark in ('。', '！', '？', '\n')
                    )
                    if boundary >= 0:
                        end = boundary + 1
                bounded.append((path, paragraph[start:end]))
                start = end
        paragraphs = bounded

    # 3. 滑动窗口切分子分块并计算父子关联哈希
    result = []
    for index, (path, paragraph) in enumerate(paragraphs):
        parent = hashlib.sha256(f'{document_id}:{version}:{index}'.encode()).hexdigest()
        for start in range(0, len(paragraph), size - overlap):
            result.append(
                StructuralChunk(
                    paragraph[start:start + size],
                    document_id,
                    version,
                    path,
                    parent,
                    paragraph,
                )
            )
            if start + size >= len(paragraph):
                break
    return result


def chunk_text(content: str, size: int, overlap: int) -> list[str]:
    """固定字符长度滑动窗口切片基线算法 (Fixed-Size Sliding Window Chunking)

    去除多余空白字符后，按步长 (size - overlap) 线性切片，用于兼容模式或平铺文本快速入库。

    Args:
        content (str): 输入的原始纯文本字符串。
        size (int): 单个切片的最大字符数。
        overlap (int): 相邻切片之间的滑动重叠字符数。

    Returns:
        list[str]: 切分后的字符串列表。若输入全为空白则返回空列表。
    """
    text = re.sub(r"\s+", " ", content or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def docling_chunks(payload: dict | list | None, source: str, settings: Settings) -> list[StructuralChunk] | None:
    """将 Docling 解析输出的版面树结构适配转换为标准结构化切片 (Adapt Docling Layout to Structural Chunks)

    设计原则：
    Docling 仅负责底层多模态解析与 OCR 转录。本函数依据文档的阅读时序、层级标题树与表格说明，
    采用与 Markdown 摄入完全一致的字符窗口执行确定性切片，切片引用仅在当前文档内部闭环解析。

    Args:
        payload (dict | list | None): Docling 解析出的 JSON 字典或列表。
        source (str): 来源文档名称或路径标识。
        settings (Settings): 系统配置实例。

    Returns:
        list[StructuralChunk] | None: 结构化切片列表；若载荷格式不合法返回 None。
    """
    if not isinstance(payload, dict) or 'schema_name' not in payload:
        return None
    import json
    from dataclasses import replace

    document_id = hashlib.sha256(source.encode()).hexdigest()
    version = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    visited = set()
    headings = []
    result = []

    def resolve(value):
        if not isinstance(value, dict):
            return {}
        reference = value.get('$ref')
        if reference is None:
            return value
        if not isinstance(reference, str) or not reference.startswith('#/'):
            return {}
        current = payload
        for segment in reference[2:].split('/'):
            segment = segment.replace('~1', '/').replace('~0', '~')
            if isinstance(current, dict):
                current = current.get(segment)
            elif isinstance(current, list) and segment.isdigit() and int(segment) < len(current):
                current = current[int(segment)]
            else:
                return {}
        return current if isinstance(current, dict) else {}

    def text_of(item):
        text = item.get('text') or item.get('orig')
        if isinstance(text, str) and text.strip():
            return text.strip()
        data = item.get('data') or {}
        if not isinstance(data, dict):
            return ''
        grid = data.get('grid')
        rows = []
        if isinstance(grid, list):
            rows = [[str(cell.get('text', '')) if isinstance(cell, dict) else str(cell)
                     for cell in row] for row in grid if isinstance(row, list)]
        else:
            mapped = {}
            for cell in data.get('table_cells', []):
                if not isinstance(cell, dict):
                    continue
                row = cell.get('start_row_offset_idx', 0)
                col = cell.get('start_col_offset_idx', 0)
                if type(row) is int and type(col) is int and 0 <= row < 100000 and 0 <= col < 1000:
                    mapped.setdefault(row, {})[col] = str(cell.get('text', ''))
            width = max((max(cells) + 1 for cells in mapped.values()), default=0)
            rows = [[mapped[row].get(col, '') for col in range(width)] for row in sorted(mapped)]
        if not rows:
            return ''
        rendered = ['| ' + ' | '.join(cell.replace('|', '\\|').replace('\n', ' ') for cell in row) + ' |' for row in rows]
        rendered.insert(1, '| ' + ' | '.join('---' for _ in rows[0]) + ' |')
        return '\n'.join(rendered)

    def emit(item):
        label = str(item.get('label', 'text'))
        if label in {'page_header', 'page_footer', 'caption'}:
            return
        text = text_of(item)
        if label in {'title', 'section_header'}:
            if text:
                level = item.get('level', 1 if label == 'title' else 2)
                level = min(6, max(1, level)) if type(level) is int else 2
                headings[:] = [(depth, title) for depth, title in headings if depth < level]
                headings.append((level, text))
            return
        if not text:
            return
        pages = [prov.get('page_no') for prov in item.get('prov', []) if isinstance(prov, dict)
                 and type(prov.get('page_no')) is int and prov['page_no'] > 0]
        captions = tuple(filter(None, (text_of(resolve(ref)) for ref in item.get('captions', []))))
        section = tuple(title for _, title in headings)
        element_index = len(result)
        for chunk in structural_chunks(text, source, settings.knowledge_chunk_size,
                settings.knowledge_chunk_overlap, parent_size=settings.knowledge_parent_chunk_size or None):
            context = '\n'.join([*section, *captions, chunk.parent_content])
            parent_id = hashlib.sha256(f'{document_id}:{version}:{element_index}:{chunk.parent_id}'.encode()).hexdigest()
            result.append(replace(chunk, document_id=document_id, document_version=version,
                section_path=section + chunk.section_path, parent_id=parent_id, parent_content=context,
                page=min(pages) if pages else None, captions=captions, element_types=(label,)))

    # 采用迭代式树遍历算法，彻底防御深层递归爆栈与循环引用死循环
    children = (payload.get('body') or {}).get('children', [])
    if not children:
        children = [item for key in ('texts', 'tables', 'pictures') for item in payload.get(key, [])]
    stack = list(reversed(children))
    while stack:
        item = resolve(stack.pop())
        if not item or id(item) in visited:
            continue
        visited.add(id(item))
        emit(item)
        stack.extend(reversed(item.get('children', [])))
    return result
