"""SolaceAI 业务工具注册、危机预警、幂等执行与管理员恢复服务 (Business Tools & Execution Subsystem)

本模块实现了 SolaceAI 中所有业务副作用操作的受控安全执行、幂等控制与审计恢复机制：
1. 确定性业务幂等键 (Deterministic Idempotency Key):
   - key = sha256(json.dumps([context.run_id, context.user_id, tool_name, arguments], sort_keys=True))
   - 在进入实际执行前，校验并记录至 `tool_executions` 状态机；
   - 处于 SUCCESS 状态直接返回已有快照，杜绝重复产生外部副作用（如重复发邮件、重复立案）；
   - 内存并发控制由 `_ACTIVE` 集合与 `_ACTIVE_LOCK` 线程锁排他保护。
2. JSON Schema Draft 2020-12 严格类型治理:
   - 工具入参必须通过 `Draft202012Validator` 结构校验方可进入调用。
3. 危机个案与紧急预警协同 (Crisis Intervention & Alerts):
   - 联动 `RiskCase` 与 `AlertRecord`，支持 SMTP 真实邮件发送与模拟投递。
4. 管理员手动断点恢复 (Disaster Recovery):
   - 针对外部网络闪断或超时置为 FAILED / 中断为 RUNNING 的工具，支持管理员手动安全重试 `recover_execution`。
"""

from __future__ import annotations

import hashlib
import json
import smtplib
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.enums import RiskCaseStatus, ToolStatus
from app.models.entities import (
    AlertRecord,
    CaseNote,
    PsychologicalReport,
    RiskCase,
    ToolExecution,
    UserAccount,
)
from app.services.agent_tools import object_schema
from app.services.mcp import ToolRegistry
from app.services.mailer import MailService
from app.services.skills import SolaceAISkillLibrary


class ToolOrchestrationService:
    """外部业务工具集成与副作用编排服务 (Tool Orchestration Service)"""

    def __init__(self, db: Session, settings: Settings):
        """初始化工具编排与执行服务实例"""
        self.db = db
        self.settings = settings

    def create_case(self, report: PsychologicalReport) -> RiskCase:
        """针对中高风险心理报告创建或复用危机个案工单"""
        existing = self.db.query(RiskCase).filter(RiskCase.report_id == report.id).first()
        if existing is not None:
            return existing
        existing = self.db.query(RiskCase).join(
            PsychologicalReport, PsychologicalReport.id == RiskCase.report_id
        ).filter(PsychologicalReport.session_id == report.session_id,
                 PsychologicalReport.user_id == report.user_id,
                 RiskCase.status != 'CLOSED').order_by(RiskCase.id.desc()).first()
        if existing is not None:
            if report.risk_level == 'HIGH' and existing.risk_level != 'HIGH':
                existing.risk_level = 'HIGH'
                existing.report_id = report.id
                existing.summary = report.summary
                existing.handoff_summary = SolaceAISkillLibrary.counselor_handoff_summary(report, self.db.get(UserAccount, report.user_id))
            existing.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.db.commit()
            return existing
        user = self.db.get(UserAccount, report.user_id)
        case = RiskCase(
            report_id=report.id,
            risk_level=report.risk_level,
            status=RiskCaseStatus.OPEN.value,
            owner=self._primary_owner(),
            summary=report.summary,
            handoff_summary=SolaceAISkillLibrary.counselor_handoff_summary(report, user),
        )
        self.db.add(case)
        self.db.commit()
        return case

    def send_case_alert(self, case: RiskCase) -> AlertRecord:
        """针对指定危机个案发送紧急预警通知并更新个案状态"""
        report = self.db.get(PsychologicalReport, case.report_id)
        if report is None:
            raise RuntimeError(f"report {case.report_id} not found")
        record = self.notify(report, case)
        if record.status == ToolStatus.SUCCESS.value:
            self.db.query(RiskCase).filter(
                RiskCase.id == case.id, RiskCase.status == RiskCaseStatus.OPEN.value
            ).update(
                {
                    "status": RiskCaseStatus.ALERT_SENT.value,
                    "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
                },
                synchronize_session=False,
            )
        self.db.commit()
        self.db.refresh(case)
        return record

    def acknowledge_case(self, case_id: int, actor: str, note: str = "") -> RiskCase:
        """辅导员/心理咨询师在线确认接手危机个案"""
        from app.services.case_workflow import CaseWorkflow

        CaseWorkflow(self.db).apply(case_id, actor.strip() or "unknown", "acknowledge", note)
        return self.db.get(RiskCase, case_id)

    def add_case_note(self, case_id: int, actor: str, note: str) -> CaseNote:
        """向指定危机干预个案中追加一条不可篡改的人工跟进面谈备注"""
        from app.services.case_workflow import CaseWorkflow

        result = CaseWorkflow(self.db).apply(case_id, actor.strip() or "unknown", "notes", note)
        return self.db.get(CaseNote, result.id)

    def notify(self, report: PsychologicalReport, case: RiskCase | None = None) -> AlertRecord:
        """高危预警投递核心方法"""
        existing = (
            self.db.query(AlertRecord)
            .filter(
                AlertRecord.report_id == report.id, AlertRecord.status == ToolStatus.SUCCESS.value,
                AlertRecord.channel == "email",
                ~AlertRecord.message.contains("deliveryMode=log"),
            )
            .first()
        )
        if existing is not None:
            return existing
        recipient = self.settings.alert_email_to.strip() or "unconfigured"
        mode = self.settings.alert_email_delivery_mode.strip().lower()
        if mode == "log":
            return self._save_alert(
                report,
                recipient if recipient != "unconfigured" else "log",
                "LOG_ONLY",
                f"高风险预警已记录：reportId={report.id}，caseId={(case.id if case else 'none')}，deliveryMode=log",
            )
        if mode != "smtp":
            return self._save_alert(
                report,
                recipient,
                ToolStatus.FAILED.value,
                f"高风险预警邮件未发送：未知投递模式 {self.settings.alert_email_delivery_mode}",
            )
        missing = self._missing_email_config()
        if missing:
            return self._save_alert(
                report,
                recipient,
                ToolStatus.FAILED.value,
                f"高风险预警邮件未发送：缺少配置 {', '.join(missing)}",
            )
        try:
            self._send_alert_email(report, case)
        except Exception as exc:
            return self._save_alert(
                report,
                recipient,
                ToolStatus.FAILED.value,
                "高风险预警邮件发送失败：请检查 SMTP 授权码、配置和网络",
            )
        return self._save_alert(
            report,
            recipient,
            ToolStatus.SUCCESS.value,
            f"SMTP 服务器已接受高风险预警邮件（不保证进入收件箱）：reportId={report.id}",
        )

    def _save_alert(
        self, report: PsychologicalReport, recipient: str, status: str, message: str
    ) -> AlertRecord:
        """内部辅助方法：构造并原子持久化单条预警通知审计记录"""
        record = AlertRecord(
            report_id=report.id,
            channel="log" if status == "LOG_ONLY" else "email",
            recipient=recipient,
            status=status,
            message=message,
        )
        self.db.add(record)
        self.db.commit()
        return record

    def _missing_email_config(self) -> list[str]:
        """排查并返回当前 SMTP 邮件服务缺失的关键必要配置项列表"""
        state = MailService(self.settings).status()
        return state["missing"] + state["issues"] + state["alertIssues"]

    def _send_alert_email(self, report: PsychologicalReport, case: RiskCase | None = None) -> None:
        """调用底层统一 SMTP 传输层向预警收件人投递危机通知邮件 (Dispatch Alert Email via SMTP)

        Args:
            report (PsychologicalReport): 关联的高危心理评估报告。
            case (RiskCase | None, optional): 关联的危机个案实体。

        Raises:
            RuntimeError: 当 SMTP 服务端拒绝投递或网络抛出异常时抛出。
        """
        result = MailService(self.settings).send(
            self._recipients(),
            f"{self.settings.alert_email_subject_prefix} reportId={report.id}",
            self._email_body(report, case),
        )
        if result["status"] != "SMTP_ACCEPTED":
            raise RuntimeError(result["message"])

    def _email_body(self, report: PsychologicalReport, case: RiskCase | None = None) -> str:
        """组装发送给辅导员与危机处置组的高清预警邮件富文本正文"""
        user = self.db.get(UserAccount, report.user_id)
        username = user.username if user else f"userId={report.user_id}"
        display_name = user.display_name if user else ""
        handoff = (
            case.handoff_summary
            if case
            else SolaceAISkillLibrary.counselor_handoff_summary(report, user)
        )
        return "\n".join(
            [
                "SolaceAI 检测到一条高风险心理预警，请尽快安排辅导员或管理员跟进。",
                "",
                f"个案ID：{case.id}" if case else "个案ID：未创建",
                f"报告ID：{report.id}",
                f"学生：{display_name} ({username})" if display_name else f"学生：{username}",
                f"风险等级：{report.risk_level}",
                f"情绪标签：{report.emotion}",
                f"摘要：{report.summary}",
                f"创建时间：{report.created_at.isoformat()}",
                "",
                "交接摘要：",
                handoff,
            ]
        )

    def _primary_owner(self) -> str:
        """获取危机预警邮件配置中的首选第一责任人"""
        recipients = self._recipients()
        if recipients:
            return recipients[0]
        return "unassigned"

    def _sender(self) -> str:
        """获取发件人地址字符串"""
        return self.settings.alert_email_from.strip() or self.settings.smtp_username.strip()

    def _recipients(self) -> list[str]:
        """解析并格式化返回所有预警收件人邮箱列表"""
        normalized = self.settings.alert_email_to.replace(";", ",")
        return [recipient.strip() for recipient in normalized.split(",") if recipient.strip()]


_ACTIVE = set()
_ACTIVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ToolContext:
    """业务工具调用执行上下文实体 (Business Tool Execution Context)"""

    user_id: int
    session_id: str
    run_id: str
    role: str = "system"


def execute_registered(
    db: Session, context: ToolContext, registry: ToolRegistry, name: str, arguments: dict
) -> object:
    """安全互斥调度执行已注册的业务工具 (Idempotent Mutex Execution)"""
    key = hashlib.sha256(
        json.dumps(
            [context.run_id, context.user_id, name, arguments], sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()
    with _ACTIVE_LOCK:
        if key in _ACTIVE:
            raise RuntimeError("Tool execution is already active")
        _ACTIVE.add(key)
    try:
        return _execute_registered(db, context, registry, name, arguments, key)
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.discard(key)


def _execute_registered(
    db: Session, context: ToolContext, registry: ToolRegistry, name: str, arguments: dict, key: str
) -> object:
    """根据幂等键查询收据并执行工具业务逻辑 (Internal Execution & Receipt Lifecycle)"""
    receipt = db.query(ToolExecution).filter(ToolExecution.execution_key == key).first()
    if receipt and receipt.status == "SUCCESS":
        return json.loads(receipt.result_json or "{}")
    if receipt and receipt.status == "RUNNING":
        raise RuntimeError("Tool execution is already running; manual recovery required")
    if receipt is None:
        receipt = ToolExecution(
            execution_key=key,
            run_id=context.run_id,
            user_id=context.user_id,
            session_id=context.session_id,
            tool_name=name,
            status="RUNNING",
            arguments_json=json.dumps(arguments, ensure_ascii=False),
            result_json="{}",
        )
        db.add(receipt)
    else:
        receipt.status = "RUNNING"
    db.commit()
    try:
        value = registry.call(name, arguments)
        receipt.status = "SUCCESS"
        receipt.result_json = json.dumps(value, ensure_ascii=False)
        receipt.error = None
        db.commit()
        return value
    except Exception as exc:
        db.rollback()
        receipt = db.query(ToolExecution).filter(ToolExecution.execution_key == key).one()
        receipt.status = "FAILED"
        receipt.error = str(exc)[:500]
        db.commit()
        raise


def business_registry(db: Session, settings: Settings, context: ToolContext) -> ToolRegistry:
    """按调用者权限动态装配受控业务工具注册表 (Context-Aware Business Tool Registry Factory)"""
    registry = ToolRegistry()
    service = ToolOrchestrationService(db, settings)

    def report(args):
        """根据报告 ID 鉴权并获取归属当前用户或管理员权限的心理报告实体。"""
        value = db.get(PsychologicalReport, args["report_id"])
        if value is None or (context.role != "ROLE_ADMIN" and value.user_id != context.user_id):
            raise PermissionError("Report not authorized")
        return value

    def create(args):
        """从授权的心理报告幂等创建或重用风险个案工单。"""
        case = service.create_case(report(args))
        return {"case_id": case.id, "report_id": case.report_id, "status": case.status}

    def alert(args):
        """为归属当前会话的有效风险工单派发告警通知并记录处置收据。"""
        case = db.get(RiskCase, args["case_id"])
        if case is None:
            raise ValueError("Case not found")
        report({"report_id": case.report_id})
        value = service.send_case_alert(case)
        if getattr(value, "status", "") not in {"SUCCESS", "LOG_ONLY"}:
            raise RuntimeError(getattr(value, "message", "Notification failed"))
        return {"case_id": case.id, "status": value.status}

    registry.register(
        "create_risk_case",
        "Create or reuse a risk case from an owned report.",
        object_schema({"report_id": {"type": "integer"}}),
        create,
    )
    registry.register(
        "send_risk_alert",
        "Record a notification for an owned risk case.",
        object_schema({"case_id": {"type": "integer"}}),
        alert,
    )
    if context.role == "ROLE_ADMIN":

        def case_action(args, action):
            """管理员专用：驱动危机干预个案工单状态流转（确认受理、添加跟进批注、闭环结案）。"""
            from app.services.case_workflow import CaseWorkflow

            actor = db.get(UserAccount, context.user_id)
            if actor is None or "ROLE_ADMIN" not in actor.roles:
                raise PermissionError("Administrator required")
            case = CaseWorkflow(db).apply(
                args["case_id"], actor.username, action, args.get("note", "")
            )
            return case.model_dump(mode="json")

        for name, action in [
            ("acknowledge_case", "acknowledge"),
            ("add_case_note", "notes"),
            ("close_case", "close"),
        ]:
            registry.register(
                name,
                "Administrator case workflow operation.",
                object_schema(
                    {"case_id": {"type": "integer"}, "note": {"type": "string", "maxLength": 4000}},
                    ["case_id"],
                ),
                lambda args, action=action: case_action(args, action),
            )

        def export(args):
            """校验筛选日期范围并生成受保护的心理报告台账 Excel 导出下载链接。"""
            from datetime import date
            from urllib.parse import urlencode

            for key in ("start", "end"):
                if args.get(key):
                    date.fromisoformat(args[key])
            return {
                "downloadUrl": "/api/admin/reports/export?" + urlencode(args),
                "filename": "solaceai-reports.xlsx",
            }

        registry.register(
            "export_reports",
            "Get an authenticated report ledger download link.",
            object_schema(
                {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                },
                [],
            ),
            export,
        )
    return registry


def recover_execution(
    db: Session, settings: Settings, execution_id: int, actor: UserAccount
) -> object:
    """管理员专用：重试或恢复失败/中断的业务工具执行 (Admin Recovery Operation)"""
    if "ROLE_ADMIN" not in getattr(actor, "roles", []):
        raise PermissionError("Administrator required")
    receipt = db.get(ToolExecution, execution_id)
    if receipt is None:
        raise ValueError("Tool execution not found")
    if receipt.status not in {"FAILED", "RUNNING"}:
        raise ValueError("Only failed or interrupted tools can be recovered")
    with _ACTIVE_LOCK:
        if receipt.execution_key in _ACTIVE:
            raise ValueError("Tool is still executing")
    if receipt.status == "RUNNING":
        receipt.status = "FAILED"
        receipt.error = "Interrupted execution selected for administrator recovery"
        db.commit()
    context = ToolContext(receipt.user_id, receipt.session_id, receipt.run_id)
    registry = business_registry(db, settings, context)
    return execute_registered(
        db, context, registry, receipt.tool_name, json.loads(receipt.arguments_json)
    )
