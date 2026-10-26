"""SolaceAI 管理员高风险危机个案流转管理路由模块 (Case Workflow Routes)

本模块作为心理咨询危机干预业务闭环的 HTTP 接入网关，面向辅导员和心理中心管理团队提供：
1. 危机个案接手确认 (`/acknowledge`)：辅导员认领高风险个案并锁定处置责任人；
2. 线下干预笔记追加 (`/notes`)：记录面对面访谈、家校联动、就医转介等动态跟进日志；
3. 个案闭环结案审核 (`/close`)：在危机解除或安全转介后提交结案理由并归档。

安全设计：
- 所有路由强制经由 `require_admin` 门禁鉴权，非管理员角色发起请求直接阻断 (HTTP 403)；
- 采用 Pydantic V2 强类型约束，禁止额外未定义字段 (`extra='forbid'`)，并通过非空校验器防范空白字符提交。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_admin
from app.models.entities import UserAccount
from app.schemas.dtos import CaseNoteResponse, RiskCaseResponse
from app.services.case_workflow import CaseWorkflow

router = APIRouter(prefix='/api/admin/cases', tags=['case-workflow'])
Database = Annotated[Session, Depends(get_db)]
Administrator = Annotated[UserAccount, Depends(require_admin)]
CaseId = Annotated[int, Path(gt=0, description="危机个案自增主键 ID（必须大于 0）")]


class NoteInput(BaseModel):
    """危机个案跟进记录追加请求体 (Case Note Input DTO)

    用于辅导员在跟进过程中录入线下接触、心理评估与干预动向。
    """
    model_config = ConfigDict(extra='forbid')

    note: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="跟进笔记正文，长度范围 1~2000 字符，不可为空白字符",
    )

    @field_validator('note')
    @classmethod
    def nonblank(cls, value: str) -> str:
        """验证并清理输入文本，防范纯空白字符提交

        Args:
            value (str): 待校验文本。

        Returns:
            str: 首尾去空白后的规范化字符串。

        Raises:
            ValueError: 当文本全为空白字符时抛出。
        """
        if not value.strip():
            raise ValueError('Text must not be blank')
        return value.strip()


class CloseInput(BaseModel):
    """危机个案闭环结案申请请求体 (Case Closure Input DTO)

    用于辅导员在危机平息或转介完成时录入结案依据。
    """
    model_config = ConfigDict(extra='forbid')

    reason: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="结案原因说明与干预效果总结，不可为空白",
    )

    @field_validator('reason')
    @classmethod
    def nonblank(cls, value: str) -> str:
        """验证并清理输入文本，防范纯空白字符提交

        Args:
            value (str): 待校验文本。

        Returns:
            str: 首尾去空白后的规范化字符串。

        Raises:
            ValueError: 当文本全为空白字符时抛出。
        """
        if not value.strip():
            raise ValueError('Text must not be blank')
        return value.strip()


@router.post('/{case_id}/acknowledge', response_model=RiskCaseResponse)
def acknowledge(case_id: CaseId, user: Administrator, db: Database):
    """管理员确认接手特定高风险个案 (Acknowledge Risk Case)

    业务逻辑：
    1. 校验目标个案是否存在；
    2. 将个案流转状态更新为 `ACKNOWLEDGED`；
    3. 记录确认接手辅导员的工号与时间戳，锁定处置责任人。

    Args:
        case_id (int): 目标个案自增主键 ID。
        user (UserAccount): 当前已鉴权的管理员/辅导员实体。
        db (Session): 数据库事务会话。

    Returns:
        RiskCaseResponse: 包含最新接手状态与操作人信息的个案详情 DTO。

    Raises:
        HTTPException(404): 个案 ID 不存在时由服务层抛出。
        HTTPException(409): 个案当前状态不允许接手（如已结案）时抛出。
    """
    return CaseWorkflow(db).apply(case_id, user.username, 'acknowledge')


@router.post('/{case_id}/notes', response_model=CaseNoteResponse)
def add_note(case_id: CaseId, body: NoteInput, user: Administrator, db: Database):
    """为特定危机个案追加人工跟进笔记 (Append Case Note)

    业务逻辑：
    1. 校验个案存在性；
    2. 插入一条类型为人工跟进的 `CaseNote` 记录，关联当前个案与记录人；
    3. 自动更新个案主表的 `updated_at` 时间戳。

    Args:
        case_id (int): 目标危机个案主键 ID。
        body (NoteInput): 包含笔记正文的经过校验的请求体。
        user (UserAccount): 当前提交笔记的辅导员实体。
        db (Session): 数据库会话。

    Returns:
        CaseNoteResponse: 新创建的跟进笔记结构体，包含主键 ID 与创建时间。

    Raises:
        HTTPException(404): 目标个案不存在时抛出。
    """
    return CaseWorkflow(db).apply(case_id, user.username, 'notes', body.note)


@router.post('/{case_id}/close', response_model=RiskCaseResponse)
def close(case_id: CaseId, body: CloseInput, user: Administrator, db: Database):
    """危机个案闭环结案处理 (Close Risk Case)

    业务逻辑：
    1. 校验个案状态是否处于已接手或跟进中；
    2. 追加一条包含结案理由的系统跟进笔记；
    3. 将个案流转状态更新为 `CLOSED` 并持久化归档。

    Args:
        case_id (int): 目标危机个案主键 ID。
        body (CloseInput): 结案原因与说明载荷。
        user (UserAccount): 执行结案操作的管理员实体。
        db (Session): 数据库会话。

    Returns:
        RiskCaseResponse: 结案后的个案完整终态信息。

    Raises:
        HTTPException(404): 目标个案不存在时抛出。
        HTTPException(409): 个案尚未接手或已处于关闭状态时抛出。
    """
    return CaseWorkflow(db).apply(case_id, user.username, 'close', body.reason)
