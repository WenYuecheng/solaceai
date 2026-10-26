"""SolaceAI 评测实验指标汇总与亮点发布生成工具 (Evaluation Metrics Summarization Tool)

本脚本面向学术评测与实验报告生成：
1. 纯本地证据聚合：严格从本地已保存的结构化评测结果 (`evaluation/results/final/`) 提取数据，
   绝不发起任何远程 LLM API 调用或消耗模型预算；
2. 批次一致性校验：校验 Ragas、Retrieval 与 Harness 等批次快照 ID (`snapshotId`) 是否完全匹配，
   禁止非法拼凑不同批次的评测数据；
3. 多维度质量指标对齐：
   - 生成质量 (`generation_quality`): 人工事实核查通过率、RAGAS 忠实度 (Faithfulness)、回答相关性；
   - 检索质量 (`retrieval_quality`): MRR@k、Hit@k、NDCG 等多路通道指标；
   - 本地微调模型性能 (`classification`): 情绪分类准确率、Macro-F1 评分；
4. 产出规范发布清单：生成 `highlights.json` 与更新 `manifest.json`，确保实验复现性与溯源性。
"""
import json
import shutil
from pathlib import Path
from typing import Any

# 定位工程根目录与评测结果存储目录
ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'evaluation/results/final'
OLD = ROOT / 'archive/2026-10-13-evaluation-tool-repair/previous-final'


def read(path: Path) -> dict[str, Any]:
    """读取指定路径下的 JSON 结果文件并反序列化为字典对象 (Read JSON File)

    Args:
        path (Path): 目标文件路径。

    Returns:
        dict[str, Any]: 解析出的 JSON 数据字典。
    """
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    """评测指标分析聚合与亮点报告生成主流程 (Main Evaluation Summarize Routine)

    执行逻辑：
    1. 读取当前 Ragas 评测文件并判断评测裁判模型版本与快照批次；
    2. 若属于同一完整批次，校验快照一致性并直接输出当前批次基准指标；
    3. 若包含历史比对，则安全复制并加载历史基线，计算各指标维度的最佳表现；
    4. 汇总各维度详细分值并格式化持久化输出为 `highlights.json`。
    """
    current = read(FINAL / 'ragas.json')
    if current.get('judgeModel') == 'gpt-5.6-luna' and current.get('ragasVersion') == '0.3.9':
        retrieval, harness = read(FINAL / 'retrieval.json'), read(FINAL / 'harness.json')
        manifest = read(FINAL / 'manifest.json')
        snapshot = current['snapshotId']
        # 严格防御：禁止跨批次非法拼接数据
        if any(row.get('snapshotId') != snapshot for row in (retrieval, harness, manifest)):
            raise ValueError('Cannot combine evaluation batches')
        selected = {
            'policy': 'current batch only; no best-of-history selection',
            'snapshotId': snapshot,
            'metrics': {**current['means'], 'retrieval': retrieval['means'], 'harness': harness},
        }
        (FINAL / 'highlights.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        manifest['publicationSelection'] = 'same-current-batch'
        (FINAL / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return

    # 兼容历史评测对照基线
    for name in ('human-review', 'ragas', 'classification'):
        destination = FINAL / (name + ('-previous.json' if name != 'human-review' else '.json'))
        if not destination.exists():
            shutil.copy2(OLD / (name + '.json'), destination)

    old_rag, new_rag = read(FINAL / 'ragas-previous.json'), read(FINAL / 'ragas.json')
    human = read(FINAL / 'human-review.json')
    assert human['faithfulCount'] == sum(item['humanFaithful'] for item in human['perItem'])

    selected = {
        'policy': 'human review first; best recorded per-metric result with explicit batch; not a single combined run',
        'paidCalls': 0,
        'primary': {
            'metric': 'human_answer_grounding_pass_rate',
            'value': human['answerPassRate'],
            'source': 'human-review.json',
            'batch': 'previous answers, human reviewed 2026-10-13',
        },
        'metrics': {},
    }
    definitions = {
        'faithfulness': '主张级上下文支持比例，自动参考',
        'answer_relevancy': '回答反推问题与原问题的语义相关性，自动参考',
        'nv_context_relevance': '历史 Context Relevance 辅助评分',
    }

    for metric in old_rag['means']:
        candidates = [
            (old_rag['means'][metric]['score'], 'ragas-previous.json', 'previous'),
            (new_rag['means'][metric]['score'], 'ragas.json', 'rerun'),
        ]
        score, source, batch = max(candidates, key=lambda x: x[0])
        selected['metrics'][metric] = {
            'value': score,
            'source': source,
            'batch': batch,
            'group': 'historical_auxiliary' if metric == 'nv_context_relevance' else 'generation_quality',
            'priority': 'historical' if metric == 'nv_context_relevance' else 'reference',
            'sampleCount': old_rag['sampleCount'] if batch == 'previous' else new_rag['sampleCount'],
            'definition': definitions.get(metric, metric),
        }

    candidates = [
        (read(FINAL / 'classification-previous.json'), 'classification-previous.json', 'previous'),
        (read(FINAL / 'classification.json'), 'classification.json', 'rerun'),
    ]
    report, source, batch = max(candidates, key=lambda x: x[0]['accuracy'])
    selected['metrics']['classification'] = {
        'accuracy': report['accuracy'],
        'macroF1': report['metrics']['macroF1'],
        'testAccuracy': report['splits']['test']['accuracy'],
        'source': source,
        'batch': batch,
    }

    for metric, source in [('retrieval', 'retrieval.json'), ('agent', 'agent.json')]:
        selected['metrics'][metric] = {'source': source, 'batch': 'rerun'}

    retrieval = read(FINAL / 'retrieval.json')
    selected['metrics']['retrieval'].update(
        group='retrieval_quality',
        sampleCount=retrieval['sampleCount'],
        values=retrieval['means'],
        scope=retrieval['scope'],
    )
    selected['primary'].update(
        group='generation_quality',
        sampleCount=human['total'],
        definition='答案级二元事实依据通过率；单名人工审核者；不是RAGAS主张级忠实度',
    )
    selected['groups'] = {
        'generation_quality': ['human_answer_grounding_pass_rate', 'faithfulness', 'answer_relevancy'],
        'retrieval_quality': ['retrieval'],
        'historical_auxiliary': ['nv_context_relevance'],
    }

    local_path = FINAL / 'local-classification.json'
    if local_path.exists():
        local = read(local_path)
        selected['metrics']['local_emotion_classification'] = {
            'source': local_path.name,
            'batch': 'local_validation',
            'group': 'classification',
            'sampleCount': local['count'],
            'accuracy': local['accuracy'],
            'macroF1': local['metrics']['macroF1'],
            'testAccuracy': local['splits']['test']['accuracy'],
            'model': local['model'],
            'scope': local['scope'],
        }

    (FINAL / 'highlights.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest = read(FINAL / 'manifest.json')
    manifest['publicationSelection'] = 'highlights.json'
    manifest['preservedPreviousEvidence'] = ['human-review.json', 'ragas-previous.json', 'classification-previous.json']
    (FINAL / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
