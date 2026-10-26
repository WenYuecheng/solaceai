"""SolaceAI 知识库向量索引离线预热与准备工具 (Offline Knowledge Index Pre-computation Tool)

本脚本面向系统初始化与部署前置流程：
1. 预热加速：在服务正式对外提供咨询问答前，提前完成知识切片的向量化特征提取与索引构建；
2. 避免在线阻塞：彻底规避首次用户会话请求发起时现场计算大模型向量导致的 HTTP 请求超时或长时间等待；
3. 缓存复用：保留已缓存的向量结果，仅针对缺失向量的切片执行增量建库。
"""
import argparse
import importlib
from pathlib import Path
import sys

# 注入工程根路径以支持绝对导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    """执行知识库向量索引预热与补齐主流程 (Main Index Preparation Routine)

    执行步骤：
    1. 动态加载目标运行模式（online 或 offline）的特化配置；
    2. 执行数据表结构初始化 (`create_schema`)；
    3. 播种心理咨询专业预置知识切片 (`seed_data`)；
    4. 调用对应后端的 `prepare_index` 接口完成向量库批量建库。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=('online', 'offline'), help='目标运行环境 (online/offline)')
    args = parser.parse_args()

    settings = importlib.import_module(f'apps.{args.profile}.settings').load_settings()
    from app.core.config import configure_settings
    configure_settings(settings)
    from app.core.bootstrap import create_schema, seed_data
    from app.core.database import SessionLocal
    from app.services.knowledge import KnowledgeService

    create_schema()
    with SessionLocal() as db:
        seed_data(db)
        service = KnowledgeService(db, settings)
        count = service.backend.prepare_index(service)
        print(f'{args.profile}: index ready; prepared {count} chunks')


if __name__ == '__main__':
    main()
