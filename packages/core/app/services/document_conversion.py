"""SolaceAI 统一文档解析转录与版面分析服务 (Multi-Modal Document Conversion Service)

本模块作为系统知识库的多模态文档解析核心，承担非结构化文件的标准化转录：
1. 全格式支持与统一入口：覆盖 PDF、Word (.docx)、PowerPoint (.pptx)、Excel (.xlsx)、
   Markdown/TXT、HTML、CSV 以及扫描图片格式；
2. Docling 现代版面分析管道集成：
   - 智能图文感知 OCR (PDF_AWARE_LAYOUT_REGIONS / FULL_PAGE)；
   - 页面大纲标题层级结构深度推断 (Heading Hierarchy H1~H6)；
   - 精准表格结构识别 (TableFormer Accurate Mode)；
   - 页眉页脚噪音消除（自动识别并剥离 PAGE_HEADER 与 PAGE_FOOTER，保持切片正文纯净）；
3. 质量前检前处理拦截：接入 `validate_text_quality` 拦截乱码与空白损坏文档；
4. 离线元数据导入优化：支持导入预先在本机生成的 `.metadata.json` 转录工件，使轻量生产环境免安装庞大 OCR 依赖。
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import RLock

# 单个上传文档允许的最大字节数（10MB）
MAX_BYTES = 10 * 1024 * 1024

# 系统支持解析的有效文件扩展名集合
SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.md', '.markdown',
    '.html', '.csv', '.png', '.jpg', '.jpeg',
}

# 进程内可重入锁，确保底层 Docling / PyTorch 转换管道线程安全
_CONVERSION_LOCK = RLock()


@dataclass(frozen=True)
class ConvertedDocument:
    """文档多模态解析转录标准化结果实体 (Immutable Converted Document Result)

    Attributes:
        source (str): 来源文档文件名或外部标识符。
        markdown (str): 解析提取出的结构化 Markdown 正文。
        content_hash (str): 原始文档二进制内容的 SHA-256 完整性校验指纹。
        parser (str): 执行转录的解析引擎类型标识 (如 'docling', 'converted')。
        parser_version (str): 解析管道与规则版本号 (如 'unified-v1')。
        pages (list | dict): 各分页详细版面信息或原始 Docling 节点树。
        structured_chunks (list[dict]): 附带版面层级的预计算结构化切片列表。
    """
    source: str
    markdown: str
    content_hash: str
    parser: str
    parser_version: str
    pages: list | dict
    structured_chunks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """将转录结果序列化为可持久化的字典格式。"""
        return asdict(self)


def _validate_input(filename: str, data: bytes) -> str:
    """输入文件名与数据体积防御性安全校验 (Defensive Input Validation)

    防范路径穿越（Directory Traversal）攻击与空字节截断（Null Byte Injection）。

    Args:
        filename (str): 原始上传文件名。
        data (bytes): 二进制文档字节流。

    Returns:
        str: 规整并过滤路径后的纯文件名。

    Raises:
        ValueError: 当文件名非法、包含空字节或文件体积超过 10MB 时抛出。
    """
    source = Path(filename.replace('\\', '/')).name
    if not source or source in {'.', '..'} or len(source) > 256 or '\x00' in source:
        raise ValueError('无效文件名')
    if len(data) > MAX_BYTES:
        raise ValueError('文件超过10MB')
    return source


def convert_document(filename: str, data: bytes, *, allow_ocr: bool = False) -> ConvertedDocument:
    """文档统一解析主入口函数 (Unified Document Conversion Dispatcher)

    支持双重解析路径：
    1. 预转换元数据直接加载：若传入 `.metadata.json` 文件，直接反序列化预计算好的结构与 Markdown，
       执行内容完整性校验与质量预检后秒级返回；
    2. 原生 Docling 引擎转换：调用深度学习模型执行多模态版面解析与 OCR 文本转录。

    Args:
        filename (str): 输入文档文件名。
        data (bytes): 文档二进制内容。
        allow_ocr (bool, optional): 兼容旧版调用的参数，PDF 格式在 Docling 下始终开启智能图文感知。

    Returns:
        ConvertedDocument: 标准化转录文档对象。

    Raises:
        ValueError: 当文档损坏、元数据格式错误或解析结果质量未达标时抛出。
    """
    source = _validate_input(filename, data)

    # 路径一：导入离线预生成的 .metadata.json
    if source.lower().endswith('.metadata.json'):
        payload = json.loads(data)
        if not isinstance(payload, dict) or not isinstance(payload.get('markdown'), str) or not isinstance(payload.get('source'), str):
            raise ValueError('转换元数据必须包含source与markdown')
        original = _validate_input(payload['source'], b'')
        if not re.fullmatch(r'[a-f0-9]{64}', str(payload.get('content_hash', ''))):
            raise ValueError('无效文档哈希')
        if not payload['markdown'].strip():
            raise ValueError('转换内容为空')

        from app.core.config import get_settings
        from app.services.document_quality import validate_text_quality
        settings = get_settings()

        # 对导入的预转换文本执行质量与乱码拦截
        validate_text_quality(
            payload['markdown'],
            threshold=settings.knowledge_quality_threshold,
            enabled=settings.knowledge_quality_precheck_enabled,
        )
        pages = payload.get('pages', [])
        if not isinstance(pages, (list, dict)):
            raise ValueError('无效文档结构')
        return ConvertedDocument(
            original,
            payload['markdown'],
            payload['content_hash'],
            str(payload.get('parser', 'converted'))[:64],
            str(payload.get('parser_version', '1'))[:64],
            pages,
            payload.get('structured_chunks', []),
        )

    # Keep Docling's ML/Pydantic dependencies out of the API environment.
    from app.core.config import get_settings
    settings = get_settings()
    if settings.runtime_profile == 'offline' and settings.document_converter_python:
        with _CONVERSION_LOCK:
            return convert_in_isolated_environment(source, data, settings)
    # 路径二：调用 Docling 引擎进行实时多模态解析
    return convert_with_docling(source, data, ocr=True)


DOCUMENT_OPTION_FIELDS = (
    'knowledge_chunk_size', 'knowledge_chunk_overlap', 'knowledge_parent_chunk_size',
    'knowledge_quality_threshold', 'knowledge_quality_precheck_enabled',
    'docling_ocr_mode', 'docling_document_timeout', 'docling_heading_hierarchy_enabled',
)


def convert_in_isolated_environment(source, data, settings):
    import os
    import subprocess
    from tempfile import TemporaryDirectory
    from app.core.paths import PROJECT_ROOT

    python = Path(settings.document_converter_python)
    if not python.is_absolute():
        python = PROJECT_ROOT / python
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    # Only parsing options cross this boundary; no application credentials.
    env['SOLACEAI_DOCUMENT_OPTIONS'] = json.dumps({name: getattr(settings, name) for name in DOCUMENT_OPTION_FIELDS})
    env['TOKENIZERS_PARALLELISM'] = 'false'
    if settings.docling_models_offline:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with TemporaryDirectory(prefix='solaceai-docling-') as folder:
        input_path = Path(folder) / source
        output_path = Path(folder) / 'converted-output.md'
        input_path.write_bytes(data)
        try:
            result = subprocess.run([str(python), '-m', 'app.services.document_conversion', str(input_path),
                '--output', str(output_path), '--isolated'], env=env, capture_output=True,
                timeout=max(60, min(1800, settings.docling_document_timeout + 90)), check=False)
        except FileNotFoundError as exc:
            raise ValueError('独立文档解析环境未就绪，请在模型环境安装 deploy/requirements/requirements-documents.txt') from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError('文档解析超时，请减少页数后重试。原有知识版本未改动。') from exc
        if result.returncode:
            # Keep detailed diagnostics server-side, never expose paths or document text in API errors.
            import logging
            logging.getLogger(__name__).error('Isolated Docling failed: %s', result.stderr.decode('utf-8', errors='replace')[-6000:])
            raise ValueError('Docling 文档解析未完成，请检查本机解析服务日志。原有知识版本未改动。')
        metadata_path = output_path.with_suffix('.metadata.json')
        if not metadata_path.is_file() or metadata_path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('文档解析结果缺失或过大')
        payload = json.loads(metadata_path.read_text(encoding='utf-8'))
        if payload.get('source') != source or payload.get('content_hash') != hashlib.sha256(data).hexdigest():
            raise ValueError('文档解析结果校验失败')
        return ConvertedDocument(**payload)


@lru_cache(maxsize=1)
def _get_converter():
    """按需延迟加载并单例复用 Docling DocumentConverter 实例 (Lazy Converter Singleton)

    配置并装配各专业子管道：
    - OCR 识别模式：根据配置选择 PDF_AWARE_LAYOUT_REGIONS、LAYOUT_REGIONS 或 FULL_PAGE；
    - 硬件加速器：默认采用 CPU 设备推理；
    - 标题层级推断：启用 heading_hierarchy_options；
    - 表格高精度恢复：启用 TableFormer ACCURATE 模式。

    Returns:
        DocumentConverter: 完成参数调优的 Docling 转换器单例。

    Raises:
        ValueError: 当未安装 docling 依赖时抛出安装指引。
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice, OcrMode,
        )
        from docling.datamodel.object_detection_engine_options import TransformersObjectDetectionEngineOptions
    except ImportError as exc:
        raise ValueError('文档解析环境未安装 Docling；请安装 deploy/requirements/requirements-documents.txt，线上轻量版可导入本机转换的 metadata.json') from exc

    from app.core.config import get_settings
    settings = get_settings()

    # 1. 配置 OCR 解析模式
    ocr_mode_setting = getattr(settings, 'docling_ocr_mode', 'pdf_aware')
    if ocr_mode_setting == 'pdf_aware' and hasattr(OcrMode, 'PDF_AWARE_LAYOUT_REGIONS'):
        ocr_mode = OcrMode.PDF_AWARE_LAYOUT_REGIONS
    elif ocr_mode_setting == 'layout_regions' and hasattr(OcrMode, 'LAYOUT_REGIONS'):
        ocr_mode = OcrMode.LAYOUT_REGIONS
    else:
        ocr_mode = getattr(OcrMode, 'FULL_PAGE', 'full_page')

    # 2. 构造 PDF 专业处理管道参数
    options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        document_timeout=getattr(settings, 'docling_document_timeout', 120.0),
    )
    options.ocr_options.mode = ocr_mode
    options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
    options.layout_options.engine_options = TransformersObjectDetectionEngineOptions(compile_model=False)

    # 3. 开启标题层级大纲识别
    if hasattr(options, 'heading_hierarchy_options'):
        options.heading_hierarchy_options.enabled = getattr(settings, 'docling_heading_hierarchy_enabled', True)

    # 4. 开启精确表格重构
    if hasattr(options, 'table_structure_options'):
        try:
            from docling.datamodel.pipeline_options import TableFormerMode
            options.table_structure_options.mode = TableFormerMode.ACCURATE
        except Exception:
            pass

    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX,
            InputFormat.XLSX, InputFormat.MD, InputFormat.HTML,
            InputFormat.CSV, InputFormat.IMAGE,
        ],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


def convert_with_docling(source: str, data: bytes, *, ocr: bool = True) -> ConvertedDocument:
    """使用 Docling 核心引擎在内存中流式转录文档 (In-Memory Docling Conversion)

    执行流程：
    1. 校验文件扩展名合法性；
    2. 加载转换器单例并针对 TXT/MD 或复合文档创建 DocumentStream；
    3. 调用 `export_to_markdown` 导出文本，优先过滤页眉页脚（PAGE_HEADER / PAGE_FOOTER）噪声；
    4. 拦截空白内容与低质量有效字符率；
    5. 将解析布局交给项目自有结构切块器，保留标题、页码和图表说明。

    Args:
        source (str): 来源文件名。
        data (bytes): 二进制输入数据。
        ocr (bool, optional): 兼容调用参数。

    Returns:
        ConvertedDocument: 包含正文、结构树与切片集合的转录结果。

    Raises:
        ValueError: 不支持的格式、提取内容为空或文本质量校验未达标。
    """
    source = _validate_input(source, data)
    suffix = Path(source).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f'不支持的文件类型: {suffix}')

    with _CONVERSION_LOCK:
        converter = _get_converter()
        from docling.datamodel.base_models import DocumentStream, InputFormat

        # 纯文本与 Markdown 文件采用专有高速后端转换
        if suffix in {'.txt', '.md', '.markdown'}:
            result = converter.convert_string(data.decode('utf-8-sig'), format=InputFormat.MD, name=source)
        else:
            stream = DocumentStream(name=Path(source).stem + suffix, stream=io.BytesIO(data))
            result = converter.convert(stream, max_file_size=MAX_BYTES)

        # 导出 Markdown 时优先过滤页眉页脚（PAGE_HEADER / PAGE_FOOTER）以净化切片
        clean_labels = None
        try:
            from docling_core.types.doc.labels import DocItemLabel
            if hasattr(DocItemLabel, '__iter__') and not isinstance(DocItemLabel, type):
                all_labels = set(DocItemLabel)
            elif hasattr(DocItemLabel, '__members__'):
                all_labels = set(DocItemLabel)
            elif isinstance(DocItemLabel, type):
                all_labels = {getattr(DocItemLabel, k) for k in dir(DocItemLabel) if not k.startswith('_')}
            else:
                all_labels = set(vars(DocItemLabel).values())
            blocked = {
                getattr(DocItemLabel, 'PAGE_HEADER', None),
                getattr(DocItemLabel, 'PAGE_FOOTER', None),
            } - {None}
            clean_labels = {l for l in all_labels if l not in blocked}
        except Exception:
            clean_labels = None

        if clean_labels is not None:
            try:
                markdown = result.document.export_to_markdown(labels=clean_labels)
            except (TypeError, Exception):
                markdown = result.document.export_to_markdown()
        else:
            markdown = result.document.export_to_markdown()

        payload = result.document.export_to_dict()

    if not markdown.strip():
        raise ValueError('文档未提取到文字')

    from app.core.config import get_settings
    from app.services.document_quality import validate_text_quality
    settings = get_settings()

    # 执行有效字符占比与乱码前置检查
    validate_text_quality(
        markdown,
        threshold=settings.knowledge_quality_threshold,
        enabled=settings.knowledge_quality_precheck_enabled,
    )

    from app.services.chunking import docling_chunks
    chunks = docling_chunks(payload, source, settings)
    return ConvertedDocument(
        source,
        markdown,
        hashlib.sha256(data).hexdigest(),
        'docling',
        'unified-v1',
        payload,
        [asdict(chunk) for chunk in chunks] if chunks is not None else [],
    )


def main() -> None:
    """文档离线批量转换与元数据导出命令行工具入口 (Offline Converter CLI)"""
    import argparse
    parser = argparse.ArgumentParser(description='使用Docling转换文档，输出Markdown与来源元数据')
    parser.add_argument('input', help='待转换的源文档物理路径')
    parser.add_argument('--output', required=True, help='转录 Markdown 文本目标输出路径')
    parser.add_argument('--ocr', action='store_true', help='兼容旧参数；PDF始终整页OCR')
    parser.add_argument('--parser', choices=['docling', 'native'], default='docling', help='兼容旧参数；统一使用Docling')
    parser.add_argument('--isolated', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.isolated:
        import os
        from app.core.config import Settings, configure_settings
        values = json.loads(os.environ.get('SOLACEAI_DOCUMENT_OPTIONS', '{}'))
        configure_settings(Settings(_env_file=None, **{k: values[k] for k in DOCUMENT_OPTION_FIELDS if k in values}))

    path = Path(args.input)
    with path.open('rb') as stream:
        converter = convert_with_docling if args.isolated else convert_document
        converted = converter(path.name, stream.read(MAX_BYTES + 1))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(converted.markdown, encoding='utf-8')
    target.with_suffix('.metadata.json').write_text(
        json.dumps(converted.to_dict(), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
