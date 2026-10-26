"""SolaceAI 高风险危机个案流转服务 (Case Workflow Domain Service).

本模块为心理支持系统中的危机个案 (RiskCase) 与人工跟进记录 (CaseNote) 提供
高内聚的领域状态机流转控制：
1. 状态机流转契约：
   - 严格遵循 `OPEN` / `ALERT_SENT` -> `ACKNOWLEDGED` (已接手) -> `CLOSED` (已结案) 的单向生命周期；
   - 处于终态 `CLOSED` 的个案受严格写保护，禁止再次接手或追加常规跟进记录；
   - 具备幂等性保障：重复确认接手或重复结案返回当前实体快照，避免重复记录。
2. 并发与事务安全：
   - 采用条件乐观锁 (`UPDATE ... WHERE id = :id AND status = :expected_status`)，
     有效防范并发场景下持有陈旧内存对象覆盖已结案工单的脏写竞态；
   - 个案状态流转与审计日志追加在单一数据库事务内原子提交，失败自动全量回滚。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models.entities import CaseNote, RiskCase
from app.schemas.dtos import CaseNoteResponse, RiskCaseResponse
from app.services.privacy import PrivacySanitizer


class CaseWorkflowError(HTTPException):
    """个案流转领域异常基类，兼容 HTTP 表现层状态码。"""

    def __init__(self, status_code: int, detail: str) -> None:
        """初始化个案工作流异常。

        Args:
            status_code: 对应的 HTTP 响应状态码。
            detail: 面向客户端或调用方的错误详细描述。
        """
        super().__init__(status_code=status_code, detail=detail)


class CaseNotFoundError(CaseWorkflowError):
    """目标危机个案在数据库中不存在。"""

    def __init__(self, detail: str = "个案不存在") -> None:
        super().__init__(status_code=404, detail=detail)


class CaseConflictError(CaseWorkflowError):
    """个案状态机扭转冲突或违反终态保护规则。"""

    def __init__(self, detail: str = "个案状态冲突") -> None:
        super().__init__(status_code=409, detail=detail)


class CaseValidationError(ValueError):
    """输入说明或备注文本格式与长度校验不合格。"""
    pass


def case_response(case: RiskCase) -> RiskCaseResponse:
    """将 RiskCase ORM 实体转换为脱敏后的前端 DTO 响应对象。

    Args:
        case: 数据库个案实体对象。

    Returns:
        RiskCaseResponse: 字段经过 PII 隐私脱敏后的响应实体。
    """
    sanitizer = PrivacySanitizer()
    return RiskCaseResponse(
        id=case.id,
        reportId=case.report_id,
        riskLevel=case.risk_level,
        status=case.status,
        owner=case.owner,
        summary=sanitizer.sanitize(case.summary),
        handoffSummary=sanitizer.sanitize(case.handoff_summary),
        acknowledgedBy=case.acknowledged_by,
        acknowledgedAt=case.acknowledged_at,
        createdAt=case.created_at,
        updatedAt=case.updated_at,
    )


def note_response(note: CaseNote) -> CaseNoteResponse:
    """将 CaseNote ORM 实体转换为脱敏后的跟进记录 DTO 响应对象。

    Args:
        note: 数据库个案跟进记录实体。

    Returns:
        CaseNoteResponse: 包含操作人与脱敏跟进内容的响应 DTO。
    """
    return CaseNoteResponse(
        id=note.id,
        caseId=note.case_id,
        actor=note.actor,
        note=PrivacySanitizer().sanitize(note.note),
        createdAt=note.created_at,
    )


class CaseWorkflow:
    """高风险心理危机个案流转服务（原子事务与审计日志强绑定）。

    本类封装了个案受理确认、跟进日志追加与闭环结案的全生命周期操作。
    所有操作均在持有数据库行锁并验证乐观锁条件的单一原子事务内完成。

    Attributes:
        db (Session): 当前请求绑定的 SQLAlchemy ORM 会话。
    """

    def __init__(self, db: Session) -> None:
        """初始化个案流转领域服务。

        Args:
            db: 具备事务支持的数据库会话实例。
        """
        self.db = db

    def apply(
        self,
        case_id: int,
        actor: str,
        action: Literal["acknowledge", "notes", "close"],
        text: str = "",
    ) -> RiskCaseResponse | CaseNoteResponse:
        """在单一数据库事务内原子执行个案状态扭转并同步插入审计追踪记录。

        执行机制：
        1. 悲观锁读取当前个案记录（`with_for_update`），保证 SQLite 与高并发数据库的一致性；
        2. 终态与幂等性检查：
           - 若已结案且请求为 `close`，幂等返回当前实体；
           - 若个案已结案，拦截任何非查询操作并抛出 `CaseConflictError`；
           - 若已受理且请求为 `acknowledge`，幂等返回当前实体；
           - 结案前必须处于已受理状态 (`ACKNOWLEDGED`)。
        3. 乐观锁更新：执行 `UPDATE ... WHERE id = :id AND status = :old_status`，
           断言生效行数严格为 1，严防并发事务抢先变更状态；
        4. 追加 `CaseNote` 审计日志，并原子提交事务。

        Args:
            case_id: 目标个案工单的自增主键 ID。
            actor: 当前经过身份认证的操作人用户名（如心理咨询师或管理员）。
            action: 状态机动作类型，支持 'acknowledge' (受理), 'notes' (追加备注), 'close' (结案)。
            text: 附加的操作原因说明或面谈备注文本（长度限 1~2000 字符）。

        Returns:
            RiskCaseResponse | CaseNoteResponse:
                - 若 action 为 'notes'，返回最新创建的跟进记录 DTO；
                - 其余动作返回状态扭转后的个案工单 DTO。

        Raises:
            CaseValidationError: 当输入文本为空白或超出 2000 字符限制时抛出。
            CaseNotFoundError: 当指定 ID 的个案不存在时抛出 (HTTP 404)。
            CaseConflictError: 当状态流转违反前置条件或乐观锁碰撞时抛出 (HTTP 409)。
        """
        text = text.strip()
        if action in {"notes", "close"} and (not text or len(text) > 2000):
            raise CaseValidationError("记录长度应为1至2000字符")
        if action == "acknowledge" and len(text) > 2000:
            raise CaseValidationError("接手说明过长")

        try:
            case = self.db.scalar(
                select(RiskCase).where(RiskCase.id == case_id).with_for_update()
            )
            if case is None:
                raise CaseNotFoundError("个案不存在")

            # 幂等性处理：重复结案
            if action == "close" and case.status == "CLOSED":
                return case_response(case)
            # 终态保护：已结案不可修改
            if case.status == "CLOSED":
                raise CaseConflictError("已结案个案不能修改")

            # 幂等性处理：重复接手
            if action == "acknowledge" and case.status == "ACKNOWLEDGED":
                return case_response(case)
            # 前置条件：未接手不可直接结案
            if action == "close" and case.status != "ACKNOWLEDGED":
                raise CaseConflictError("请先确认接手再结案")

            timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
            values = {"updated_at": timestamp}

            if action == "acknowledge":
                if case.status not in ("OPEN", "ALERT_SENT"):
                    raise CaseConflictError("当前状态无法接手")
                values.update(
                    status="ACKNOWLEDGED",
                    acknowledged_by=actor,
                    acknowledged_at=timestamp,
                    owner=actor,
                )
                text = text or "已确认接手该个案"
            elif action == "close":
                values["status"] = "CLOSED"
                text = f"结案原因：{text}"
            elif action != "notes":
                raise CaseValidationError(f"Unknown case action: {action}")

            # 采用乐观锁条件更新防并发竞态，同时兼容 SQLite 与支持行级锁的企业级关系数据库
            result = self.db.execute(
                update(RiskCase)
                .where(RiskCase.id == case_id, RiskCase.status == case.status)
                .values(**values)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                raise CaseConflictError("个案状态已变化，请刷新后重试")

            note = CaseNote(
                case_id=case_id,
                actor=actor,
                note=text,
                created_at=timestamp,
            )
            self.db.add(note)
            self.db.commit()
            self.db.refresh(case)
            return note_response(note) if action == "notes" else case_response(case)
        except Exception:
            self.db.rollback()
            raise
