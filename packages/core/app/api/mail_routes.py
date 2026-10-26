"""SolaceAI 管理员邮件预警诊断与危机告警重试路由模块 (Mail Diagnostics & Alert Recovery Routes)

本模块面向心理危机干预中心管理团队，提供危机邮件投递的审计诊断与容错恢复服务：
1. 邮件服务健康状态诊断 (`GET /api/admin/mail/status`)：校验 SMTP 配置就绪状态与安全连通性；
2. 固定目标测试邮件发送 (`POST /api/admin/mail/test`)：仅向服务端预配置的 `ALERT_EMAIL_TO` 发送安全探针邮件，
   收件人完全由服务端环境变量控制，杜绝任意收件人中继开放代理漏洞；
3. 危机预警租约重试与中断恢复 (`POST /api/admin/mail/alerts/{record_id}/retry`)：
   - 针对投递失败 (`FAILED`) 或进程崩溃中断 (`RUNNING`) 的预警记录进行幂等恢复；
   - 引入排他持久化租约 (`Durable Retry Lease`)，防范多管理员并发抢占导致邮件重复轰炸；
   - 针对非原子性中断场景，强制管理员确认可能重复投递的风险后方可恢复。
"""
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.core.security import require_admin
from app.models.entities import AlertRecord, PsychologicalReport, RiskCase, UserAccount
from app.services.business_tools import ToolOrchestrationService
from app.services.mailer import MailService
from app.services.mail_delivery_lease import lease_expired, new_lease

router = APIRouter(prefix='/api/admin/mail', tags=['mail'])
Admin = Annotated[UserAccount, Depends(require_admin)]
Database = Annotated[Session, Depends(get_db)]
Config = Annotated[Settings, Depends(get_settings)]


class TestMailRequest(BaseModel):
    """邮件连通性测试请求体载荷 (Test Mail Request DTO)

    禁止额外字段，防范恶意入参篡改。
    """
    model_config = ConfigDict(extra='forbid')


class RetryMailRequest(BaseModel):
    """危机邮件预警重试请求体载荷 (Retry Alert Request DTO)

    Attributes:
        recoverInterrupted (bool): 是否强制确认恢复已中断超时的投递任务（默认 False）。
    """
    model_config = ConfigDict(extra='forbid')
    recoverInterrupted: bool = False


@router.get('/status')
def mail_status(_: Admin, settings: Config) -> dict[str, Any]:
    """获取当前系统邮件发送基础设施就绪状态 (Get SMTP Service Status)

    Args:
        _: 当前已认证的管理员账号。
        settings: 系统全局配置实例。

    Returns:
        dict: 包含启用状态、发件人、收件人列表以及连通性指标的字典。
    """
    return MailService(settings).status()


@router.post('/test')
def test_mail(_: Admin, settings: Config, body: Annotated[TestMailRequest | None, Body()] = None) -> dict[str, Any]:
    """向系统预配置的预警邮箱发送连通性测试探针邮件 (Send Fixed-Destination Test Mail)

    安全设计：
    收件人必须严格取自服务端配置的 `ALERT_EMAIL_TO`，坚决不接受客户端传入的目标邮箱，
    彻底防御将内部邮件网关当做垃圾邮件发送跳板的开放中继（Open Relay）攻击。

    Args:
        _: 当前已认证的管理员账号。
        settings: 系统全局配置实例。
        body: 可选的严格请求载荷。

    Returns:
        dict: 包含投递状态（如 'SMTP_ACCEPTED'）与接收方信息的投递凭证。

    Raises:
        HTTPException(409): 当系统尚未配置预警收件人邮箱时抛出。
    """
    service = MailService(settings)
    # 收件人严格由服务端配置决定，不可由入参控制
    if not service.recipients():
        raise HTTPException(409, '请先配置 ALERT_EMAIL_TO')
    return service.send(
        service.recipients(),
        '[SolaceAI] 邮件服务测试',
        '这是一封由管理员发起的邮件服务测试，不代表发生风险事件。',
    )


@router.post('/alerts/{record_id}/retry')
def retry_alert(
    record_id: int,
    actor: Admin,
    db: Database,
    settings: Config,
    body: Annotated[RetryMailRequest | None, Body()] = None,
) -> dict[str, Any]:
    """重试或恢复失败的危机告警邮件投递 (Retry or Recover Failed Crisis Alert)

    执行逻辑：
    1. 实体存在性与渠道检查：仅允许重试渠道为 'email' 且状态为 'FAILED' 或 'RUNNING'（中断）的预警；
    2. 中断租约校验：若原记录为 'RUNNING'，必须满足租约已超时，且请求显式提供了 `recoverInterrupted=True` 确认；
    3. 分布式排他租约抢占：基于 `(id, status, message)` 的乐观锁更新，将状态置为 `RUNNING` 并写入带有效期的租约，
       同时将同一份心理报告下的历史失败/中断预警批量抢占，防止重复轰炸；
    4. 执行邮件重发与个案/报告联动投递；
    5. 成功后更新旧记录状态为 `RETRIED` 并持久化留痕本次操作责任管理员。

    Args:
        record_id (int): 目标预警记录主键 ID。
        actor (Admin): 执行重试操作的管理员用户实体。
        db (Session): 数据库会话。
        settings (Config): 系统配置。
        body (RetryMailRequest | None): 包含是否强制确认恢复中断的入参载荷。

    Returns:
        dict: 包含新生成的预警记录 ID、投递状态与时序元数据的字典。

    Raises:
        HTTPException(404): 预警记录或关联报告不存在。
        HTTPException(409): 租约尚未到期、缺少重复确认或发生并发抢占冲突。
        HTTPException(503): 外部 SMTP 通信故障或投递结果不确定。
    """
    record = db.get(AlertRecord, record_id)
    if record is None:
        raise HTTPException(404, '预警记录不存在')
    interrupted = record.status == 'RUNNING'
    if record.channel != 'email' or record.status not in {'FAILED', 'RUNNING'}:
        raise HTTPException(409, '仅允许重试失败或已中断的邮件预警')
    if interrupted and not lease_expired(record.message):
        raise HTTPException(409, '投递租约尚未到期，请勿重复发送')
    if interrupted and not (body and body.recoverInterrupted):
        raise HTTPException(409, '投递结果不确定，确认可能重复后方可恢复')
    report = db.get(PsychologicalReport, record.report_id)
    if report is None:
        raise HTTPException(404, '关联报告不存在')

    # 乐观并发控制：比对原有状态与消息文本，防止过期请求覆盖刚刚延期的有效租约
    old_message, old_status = record.message, record.status
    lease = new_lease(actor.username, settings.smtp_timeout_seconds, interrupted)
    claimed = db.query(AlertRecord).filter(
        AlertRecord.id == record_id,
        AlertRecord.status == old_status,
        AlertRecord.message == old_message,
    ).update({'status': 'RUNNING', 'message': lease}, synchronize_session=False)
    if not claimed:
        db.rollback()
        raise HTTPException(409, '该预警正在重试或已处理')

    active = db.query(AlertRecord).filter(
        AlertRecord.report_id == report.id,
        AlertRecord.status == 'RUNNING',
        AlertRecord.id != record_id,
    ).all()
    if any(not (interrupted and item.message == old_message) for item in active):
        db.rollback()
        raise HTTPException(409, '该报告的预警正在重试')

    db.query(AlertRecord).filter(
        AlertRecord.report_id == report.id,
        AlertRecord.channel == 'email',
        or_(
            AlertRecord.status == 'FAILED',
            (AlertRecord.status == 'RUNNING') & (AlertRecord.message == old_message),
        ),
    ).update({'status': 'RUNNING', 'message': lease}, synchronize_session=False)
    db.commit()

    case = db.query(RiskCase).filter(RiskCase.report_id == report.id).first()
    service = ToolOrchestrationService(db, settings)
    try:
        result = service.send_case_alert(case) if case else service.notify(report)
    except Exception:
        db.rollback()
        # 保留持久化租约：SMTP 可能已接收但数据库持久化失败。租约到期后由人工显式介入
        raise HTTPException(503, '投递结果暂不确定，请在租约到期后核查并恢复') from None

    suffix = '；中断恢复可能重复，已由管理员确认' if interrupted else ''
    db.query(AlertRecord).filter(
        AlertRecord.report_id == report.id,
        AlertRecord.status == 'RUNNING',
        AlertRecord.message == lease,
    ).update(
        {'status': 'RETRIED', 'message': f'管理员 {actor.username} 已重试，结果记录 #{result.id}{suffix}'},
        synchronize_session=False,
    )
    db.commit()
    return {
        'recoveredInterrupted': interrupted,
        'id': result.id,
        'reportId': result.report_id,
        'status': result.status,
        'channel': result.channel,
        'recipient': result.recipient,
        'message': result.message,
        'createdAt': result.created_at.isoformat(),
    }
