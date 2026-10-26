"""SolaceAI 用户对话历史与会话可见性路由模块 (Conversation Routes)

本模块为学生端前端提供会话列表查询、会话完整历史回溯以及软删除（隐藏）管理：
1. 历史会话列表分页检索 (`/api/sessions`)：基于当前登录用户 ID 过滤，仅展示对该用户可见的会话；
2. 会话多轮明细获取 (`/api/sessions/{public_id}`)：按时间序完整加载会话下的问答消息流；
3. 会话软删除隐藏 (`DELETE /api/sessions/{public_id}`)：通过 `SessionVisibility` 机制将指定会话对学生列表隐藏，
   保留数据库历史数据以供心理危机溯源与辅导员审查。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import current_user
from app.models.entities import UserAccount
from app.services.conversations import ConversationService

router = APIRouter()


@router.get('/api/sessions')
def list_sessions(
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=50, description="单页拉取最大记录数 (1~50)")] = 50,
    offset: Annotated[int, Query(ge=0, description="列表游标偏移量")] = 0,
):
    """分页获取当前登录用户的可见历史会话列表 (List Visible User Sessions)

    执行逻辑：
    1. 根据当前登录学生 `user.id` 过滤所属会话；
    2. 关联 `SessionVisibility` 表过滤被软删除隐藏的会话；
    3. 按会话最近更新时间 `updated_at` 降序排列并执行切片分页。

    Args:
        user (UserAccount): 当前已完成登录认证的用户实体。
        db (Session): 数据库会话。
        limit (int, optional): 每页展示的最大条数，默认 50。
        offset (int, optional): 分页偏移量，默认 0。

    Returns:
        list[dict]: 包含 public_id、title、created_at、updated_at 等属性的会话摘要列表。
    """
    return ConversationService(db).list(user.id, limit, offset)


@router.get('/api/sessions/{public_id}')
def session_detail(
    public_id: str,
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """获取当前用户特定会话的全部消息历史 (Get Session Message History)

    执行逻辑：
    1. 校验目标会话是否存在且归属于当前登录用户；
    2. 按时间升序拉取会话下所有的 `ChatMessage` 实体；
    3. 组装为包含用户消息与 AI 回复的时序消息数组返回。

    Args:
        public_id (str): 会话对外公开的 UUID 格式字符串标识。
        user (UserAccount): 当前登录用户。
        db (Session): 数据库会话。

    Returns:
        dict: 包含会话 public_id、title 及 messages 数组的完整详情。

    Raises:
        HTTPException(404): 会话不存在或无权访问他人会话时由领域服务抛出。
    """
    return ConversationService(db).detail(user.id, public_id)


@router.delete('/api/sessions/{public_id}')
def delete_session(
    public_id: str,
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """用户端软删除（隐藏）指定历史会话 (Hide Session For User)

    为保护危机干预证据链与合规追溯要求，系统不执行物理 DELETE 删除，
    而是在 `SessionVisibility` 映射表中将 `hidden` 置为 True，仅对学生端列表隐藏。

    Args:
        public_id (str): 待隐藏会话的 public_id。
        user (UserAccount): 当前登录用户。
        db (Session): 数据库会话。

    Returns:
        dict: 操作结果字典，例如 {"success": True}。

    Raises:
        HTTPException(404): 会话不存在或不归属当前用户时抛出。
    """
    return ConversationService(db).hide(user.id, public_id)
