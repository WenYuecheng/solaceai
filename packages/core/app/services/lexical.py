"""SolaceAI 确定性词法检索、轻量分词与降级精排服务 (Lexical Retrieval, Tokenization & Fallback Ranking)

本模块提供不依赖任何重量级外部 NLP 框架的纯 Python 高性能词法检索能力：
1. 双重分词器 (`tokenize`)：集成中文字符、二元文法 (2-gram) 与英文标点词元切分；
2. 经典 Okapi BM25 打分引擎 (`bm25_scores`)：
   - 工业级标准参数：k1=1.5（词频饱和度控制）、b=0.75（文档长度归一化惩罚）；
   - 平滑 IDF 计算公式，有效抵御极端词频偏置；
3. 本地多特征精排公式 (`rerank_score`)：融合向量基础分 (55%)、混合词法分 (25%)、
   查询词覆盖度 (15%) 与连续短语完全命中度 (5%)；
4. 线性归一化 (`normalize_scores`)：提供极差 Min-Max 缩放至 [0.0, 1.0] 闭区间，支撑多路召回融合。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Hashable, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.entities import KnowledgeChunk


def hybrid_score(query: str, content: str) -> float:
    """混合词法综合打分 (Hybrid Lexical Matching Score)

    计算公式：
        Hybrid_Score = Token_Cosine * 0.75 + Keyword_Recall * 0.25

    Args:
        query (str): 用户检索提问文本。
        content (str): 候选知识切片正文。

    Returns:
        float: 介于 0.0 至 1.0 之间的词法匹配得分。
    """
    return token_cosine(query, content) * 0.75 + keyword_score(query, content) * 0.25


def bm25_scores(query: str, chunks: list[KnowledgeChunk]) -> dict[int, float]:
    """基于 Okapi BM25 算法计算候选切片的词法相关度打分 (Okapi BM25 Ranking)

    数学公式：
        BM25(D, Q) = sum_{t in Q} [ IDF(t) * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (|D| / avgdl))) ]
        其中 IDF(t) = ln(1.0 + (N - n(t) + 0.5) / (n(t) + 0.5))

    Args:
        query (str): 检索查询词。
        chunks (list[KnowledgeChunk]): 候选切片实体列表。

    Returns:
        dict[int, float]: 包含非零得分切片的映射字典 `{chunk_id: score}`。
    """
    query_terms = Counter(tokenize(query))
    if not query_terms or not chunks:
        return {}

    documents = []
    doc_freqs: dict[str, int] = {}

    # 1. 统计文档词频与全局词项文档频率 (Document Frequency, DF)
    for chunk in chunks:
        if chunk.id is None:
            continue
        token_counts = Counter(tokenize(chunk.content))
        documents.append((chunk.id, token_counts, sum(token_counts.values())))
        for term in token_counts:
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    total_docs = len(documents)
    if total_docs == 0:
        return {}

    # 2. 计算平均文档长度 (avgdl)
    average_length = sum(length for _, _, length in documents) / total_docs or 1.0
    k1 = 1.5
    b = 0.75
    scores: dict[int, float] = {}

    # 3. 逐文档计算 Okapi BM25 综合得分
    for chunk_id, token_counts, doc_length in documents:
        score = 0.0
        length_norm = k1 * (1.0 - b + b * doc_length / average_length)
        for term, query_frequency in query_terms.items():
            term_frequency = token_counts.get(term, 0)
            if term_frequency == 0:
                continue
            doc_frequency = doc_freqs.get(term, 0)
            # 平滑逆文档频率计算
            idf = math.log(1.0 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            query_boost = 1.0 + math.log(query_frequency)
            score += (
                idf * query_boost * (term_frequency * (k1 + 1.0)) / (term_frequency + length_norm)
            )
        if score > 0:
            scores[chunk_id] = score

    return scores


def rerank_score(query: str, content: str, base_score: float) -> float:
    """本地多特征语义精排加权打分公式 (Multi-Feature Heuristic Reranker)

    加权分配方案：
    - base_score (55%): 粗排基线相关度得分（向量余弦相似度或 BM25 归一化得分）；
    - lexical (25%): 混合词法余弦与关键词召回打分；
    - coverage (15%): 用户查询 Token 在切片中的覆盖率；
    - phrase (5%): 连续自然语言短语的精确命中率。

    Args:
        query (str): 查询文本。
        content (str): 切片候选文本。
        base_score (float): 粗排基线分数。

    Returns:
        float: 最终精排综合得分。
    """
    lexical = hybrid_score(query, content)
    coverage = query_token_coverage(query, content)
    phrase = phrase_score(query, content)
    return base_score * 0.55 + lexical * 0.25 + coverage * 0.15 + phrase * 0.05


def query_token_coverage(query: str, content: str) -> float:
    """计算查询词中被切片包含的 Token 比例 (Query Token Coverage Ratio)

    Args:
        query (str): 检索查询。
        content (str): 文档切片。

    Returns:
        float: 包含 Token 数除以总查询 Token 数，范围 [0.0, 1.0]。
    """
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize(content))
    return len(query_tokens & content_tokens) / len(query_tokens)


def phrase_score(query: str, content: str) -> float:
    """计算连续自然语言短语的完全包含匹配度得分 (Exact Phrase Containment Score)

    若去除全部空格后查询串子串出现在文档中，直接获得满分 1.0；否则回退至关键词覆盖率。

    Args:
        query (str): 用户查询。
        content (str): 文档正文。

    Returns:
        float: 匹配得分 (0.0~1.0)。
    """
    normalized_query = compact_text(query)
    if not normalized_query:
        return 0.0
    normalized_content = compact_text(content)
    if normalized_query in normalized_content:
        return 1.0
    return keyword_score(query, content)


def compact_text(text: str) -> str:
    """去除输入文本中的全部空白字符并转换为小写 (Compact Normalized Text)

    Args:
        text (str): 原始文本。

    Returns:
        str: 紧凑化小写文本。
    """
    return re.sub(r"\s+", "", text.lower())


def normalize_scores(scores: dict[Hashable, float]) -> dict[Hashable, float]:
    """Min-Max 分数线性归一化到 [0.0, 1.0] 区间 (Min-Max Linear Normalization)

    Args:
        scores (dict[Hashable, float]): 原始分数映射字典。

    Returns:
        dict[Hashable, float]: 归一化后的同结构分数映射字典。
    """
    positives = [score for score in scores.values() if score > 0]
    if not positives:
        return {key: 0.0 for key in scores}
    lowest = min(positives)
    highest = max(positives)
    if math.isclose(lowest, highest):
        return {key: 1.0 if score > 0 else 0.0 for key, score in scores.items()}
    return {
        key: (score - lowest) / (highest - lowest) if score > 0 else 0.0
        for key, score in scores.items()
    }


def tokenize(text: str) -> list[str]:
    """中文单字/二元文法 (2-gram) 与英文单词轻量混合分词器 (Hybrid Character & 2-gram Tokenizer)

    不需要引入 jieba 等外部词库，即可高效捕捉中文多字复合词组语义。

    Args:
        text (str): 待分词的原始文本。

    Returns:
        list[str]: 词元与 2-gram 字符串列表。
    """
    words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
    grams = words[:]
    compact = "".join(ch for ch in text.lower() if "\u4e00" <= ch <= "\u9fff")
    grams.extend(compact[i:i + 2] for i in range(max(0, len(compact) - 1)))
    return [item for item in grams if item.strip()]


def token_cosine(left: str, right: str) -> float:
    """计算两段文本在轻量分词后的词频多维向量余弦相似度 (Token Frequency Cosine Similarity)

    Args:
        left (str): 第一段文本。
        right (str): 第二段文本。

    Returns:
        float: 余弦夹角余弦值 (0.0~1.0)。
    """
    left_counts = Counter(tokenize(left))
    right_counts = Counter(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0
    dot = sum(value * right_counts.get(key, 0) for key, value in left_counts.items())
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    return 0.0 if left_norm == 0 or right_norm == 0 else dot / (left_norm * right_norm)


def keyword_score(query: str, content: str) -> float:
    """计算查询短语切分词项在文档正文中的命中覆盖比例 (Term Recall Rate)

    按常见标点切分长度 >= 2 的有效词项，统计其在内容中出现的比例。

    Args:
        query (str): 查询文本。
        content (str): 文档内容。

    Returns:
        float: 召回比例，范围 [0.0, 1.0]。
    """
    terms = [
        term for term in re.split(r"[\s，。！？、；：,.!?;:]+", query.lower()) if len(term) >= 2
    ]
    if not terms:
        return 0.0
    lower = content.lower()
    matched = sum(1 for term in terms if term in lower)
    return min(1.0, matched / len(terms))
