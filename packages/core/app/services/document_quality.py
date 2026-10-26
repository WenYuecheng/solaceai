"""文档质量前处理检测模块 (Document Ingestion Quality Pre-check)。

依据 Garbage In Garbage Out 架构原则，在文档入库和切块前进行质量预检：
1. 评估正文或 OCR 抽取内容的有效字符占比与可识别密度；
2. 识别并拦截乱码、CID 映射缺失（(cid:xxx)）、Unicode 替换符（\\ufffd）及控制符噪声；
3. 有效字符率低于设定阈值（默认 80%）时阻止入库，保护知识库与向量检索质量。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 常用中日韩文字与常用扩展
_CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
# 中文常用标点与全角字符
_CJK_PUNCT_PATTERN = re.compile(r'[\u3000-\u303f\uff00-\uffef]')
# 标准英文字母与数字
_ALPHANUM_PATTERN = re.compile(r'[a-zA-Z0-9]')
# 标准 ASCII 标点与符号
_ASCII_PUNCT_SET = set(r""".,;:!?'"()[]{}-_/<=>+*#@&%$%^~`|\`""'""")

# 常见乱码特征：
# 1. Unicode 替换字符 (REPLACEMENT CHARACTER \ufffd)
# 2. 控制字符 (除 \t, \n, \r 外)
# 3. 私有字符区 (Private Use Area \ue000-\uf8ff，常由未嵌字体的 PDF 乱转生成)
# 4. CID 字体编码缺失标记 (如 (cid:123))
_CID_PATTERN = re.compile(r'\(cid:\d+\)')
_REPLACEMENT_CHAR = '\ufffd'
_CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_PUA_PATTERN = re.compile(r'[\ue000-\uf8ff]')


class DocumentQualityError(ValueError):
    """文档质量不达标专属业务异常，继承自 ValueError 便于与 FastAPI 422 异常处理器无缝衔接。"""

    def __init__(self, message: str = "该文档质量不达标，请检查后重新上传", report: QualityReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class QualityReport:
    """文档质量检测报告"""

    is_valid: bool
    valid_ratio: float
    total_chars: int
    non_space_chars: int
    valid_chars: int
    garbage_chars: int
    reason: str


def inspect_text_quality(text: str, threshold: float = 0.80) -> QualityReport:
    """评估提取后文本或 OCR 转换结果的质量指标。

    :param text: 抽样文本或全量 Markdown 文本
    :param threshold: 有效字符最低占比阈值（默认 0.80，即 80%）
    :return: QualityReport 结构化质量指标
    """
    total_chars = len(text)
    if total_chars == 0:
        return QualityReport(
            is_valid=False,
            valid_ratio=0.0,
            total_chars=0,
            non_space_chars=0,
            valid_chars=0,
            garbage_chars=0,
            reason="文档未提取到任何有效字符或内容为空",
        )

    # 统计 CID 乱码块
    cid_matches = _CID_PATTERN.findall(text)
    cid_garbage_len = sum(len(m) for m in cid_matches)
    text_without_cid = _CID_PATTERN.sub('', text)

    non_space_chars = 0
    valid_chars = 0
    garbage_chars = cid_garbage_len

    for char in text_without_cid:
        if char.isspace():
            continue
        non_space_chars += 1

        # 检查明确垃圾字符
        if char == _REPLACEMENT_CHAR or _CONTROL_CHAR_PATTERN.match(char) or _PUA_PATTERN.match(char):
            garbage_chars += 1
            continue

        # 检查有效字符：中文字符、字母数字、标准中英文标点
        if (
            _CJK_PATTERN.match(char)
            or _ALPHANUM_PATTERN.match(char)
            or _CJK_PUNCT_PATTERN.match(char)
            or char in _ASCII_PUNCT_SET
        ):
            valid_chars += 1

    # 如果有 CID 乱码，也计入非空白字符总数
    non_space_chars += cid_garbage_len

    if non_space_chars == 0:
        return QualityReport(
            is_valid=False,
            valid_ratio=0.0,
            total_chars=total_chars,
            non_space_chars=0,
            valid_chars=0,
            garbage_chars=garbage_chars,
            reason="文档未提取到非空白文字内容",
        )

    valid_ratio = round(valid_chars / non_space_chars, 4)
    is_valid = valid_ratio >= threshold

    reason = (
        f"文档质量达标（有效字符率 {valid_ratio:.1%} >= {threshold:.1%}）"
        if is_valid
        else f"该文档质量不达标，有效字符率仅为 {valid_ratio:.1%}（低于 {threshold:.1%} 阈值）"
    )

    return QualityReport(
        is_valid=is_valid,
        valid_ratio=valid_ratio,
        total_chars=total_chars,
        non_space_chars=non_space_chars,
        valid_chars=valid_chars,
        garbage_chars=garbage_chars,
        reason=reason,
    )


def validate_text_quality(
    text: str,
    threshold: float = 0.80,
    *,
    enabled: bool = True,
) -> QualityReport:
    """校验文档文本质量，未达标时抛出 DocumentQualityError 阻止入库。"""
    if not enabled:
        return QualityReport(
            is_valid=True,
            valid_ratio=1.0,
            total_chars=len(text),
            non_space_chars=len(text),
            valid_chars=len(text),
            garbage_chars=0,
            reason="文档质量预检已配置为跳过",
        )

    report = inspect_text_quality(text, threshold=threshold)
    if not report.is_valid:
        logger.warning(
            "文档质量预检未通过并触发拒绝入库: valid_ratio=%.2f, threshold=%.2f, reason=%s",
            report.valid_ratio,
            threshold,
            report.reason,
        )
        raise DocumentQualityError("该文档质量不达标，请检查后重新上传", report=report)

    return report
