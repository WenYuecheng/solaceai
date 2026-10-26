"""SolaceAI 学生端可见执行轨迹持久化模型 (Student-Visible Answer Trace Model)

本模块定义面向学生客户端安全暴露的推理与工具执行轨迹实体：
1. 权限与隐私隔离：仅持久化脱敏后的阶段摘要与过程步骤，严格与管理员后台
   包含系统提示词、原始模型思考链路与未脱敏隐私的深度审计载荷（`agent_run_traces`）隔离；
2. 跨数据库方言大文本适配：针对 MySQL 方言采用 `MEDIUMTEXT` 类型，规避常规 `TEXT`
   在超长多轮检索步骤堆叠下发生 64KB 截断溢出。
"""
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class PublicAnswerTrace(Base):
    """学生可见公开问答执行追踪实体 (Public Answer Trace Entity)

    记录单次模型生成过程中的标准化七阶段（准备、思考、行动、观察、回答、审核、交付）轻量摘要，
    支持前端以时序折叠卡片形式向学生可视化展示 AI 思考与检索证据的过程。

    Attributes:
        run_id (str): 单轮智能体运行的唯一追踪 UUID（主键，最长 64 字符）。
        user_id (int): 发起请求的学生用户 ID (`user_accounts.id`)，用于属主可见性校验。
        session_id (int): 所属对话会话主键 ID (`chat_sessions.id`)。
        status (str): 当前运行生命周期状态（例如 'running', 'completed', 'failed', 'degraded'）。
        steps_json (str): 序列化的步骤数组 JSON 字符串（MySQL 下采用 MEDIUMTEXT 避免 64KB 溢出）。
    """
    __tablename__ = 'public_answer_traces'
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    session_id: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(20), default='running')
    steps_json: Mapped[str] = mapped_column(Text().with_variant(MEDIUMTEXT(), 'mysql'), default='[]')
