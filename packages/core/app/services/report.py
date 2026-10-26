"""SolaceAI 数据报表、审计与个案查询服务 (Report & Case Query Service)"""

from __future__ import annotations

from sqlalchemy import select, func
from datetime import datetime
from io import BytesIO
from openpyxl import Workbook
from sqlalchemy.orm import Session
from app.services.privacy import PrivacySanitizer

from app.models.entities import (
    AgentRunTrace,
    AlertRecord,
    CaseNote,
    ChatMessage,
    ChatSession,
    DeadLetterRecord,
    ExcelRecord,
    PsychologicalReport,
    RiskCase,
    ToolAuditRecord,
    ToolJob,
    UserAccount,
)
from app.schemas.dtos import (
    AgentRunTraceResponse,
    CaseNoteResponse,
    ConversationMessageResponse,
    ConversationResponse,
    DeadLetterResponse,
    ReportResponse,
    RiskCaseResponse,
    ToolAuditResponse,
    ToolJobResponse,
    ToolRecordResponse,
)


class ReportService:
    """心理健康评估报表与多维审计聚合服务 (Assessment Reporting & Audit Service)"""

    def __init__(self, db: Session):
        """初始化报表与审计聚合服务实例"""
        self.db = db

    def latest_reports(self, user_id: int | None = None) -> list[ReportResponse]:
        """分页获取最新的心理评估报告列表"""
        query = self.db.query(PsychologicalReport).order_by(PsychologicalReport.created_at.desc())
        if user_id is not None:
            query = query.filter(PsychologicalReport.user_id == user_id)
        latest = self.db.query(func.max(PsychologicalReport.id)).group_by(PsychologicalReport.session_id)
        return [self._report_response(item) for item in query.filter(PsychologicalReport.id.in_(latest)).limit(100).all()]

    def excel_records(self) -> list[ToolRecordResponse]:
        """获取最新的 Excel 台账写入审计记录"""
        rows = self.db.query(ExcelRecord).order_by(ExcelRecord.created_at.desc()).limit(100).all()
        return [
            ToolRecordResponse(id=row.id, reportId=row.report_id, status=row.status, message=row.message, createdAt=row.created_at, filePath=row.file_path)
            for row in rows
        ]

    def alert_records(self) -> list[ToolRecordResponse]:
        """获取最新的高危预警通知投递记录"""
        rows = self.db.query(AlertRecord).order_by(AlertRecord.created_at.desc()).limit(100).all()
        return [
            ToolRecordResponse(
                id=row.id,
                reportId=row.report_id,
                status=row.status,
                message=row.message,
                createdAt=row.created_at,
                channel=row.channel,
                recipient=row.recipient,
            )
            for row in rows
        ]

    def risk_cases(self) -> list[RiskCaseResponse]:
        """获取辅导员跟进的心理危机个案列表"""
        rows = self.db.query(RiskCase).order_by(RiskCase.updated_at.desc()).limit(100).all()
        # 在前端展示层面聚合去重历史遗留的活跃个案；数据库内完整保留所有审计行供留痕追溯
        grouped = {}
        for row in rows:
            report = self.db.get(PsychologicalReport, row.report_id)
            key = ('case', row.id) if row.status == 'CLOSED' or report is None else ('session', report.session_id)
            previous = grouped.get(key)
            priority = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
            if previous is None or priority.get(row.risk_level, 0) > priority.get(previous.risk_level, 0):
                grouped[key] = row
        rows = sorted(grouped.values(), key=lambda row: row.updated_at, reverse=True)
        return [
            RiskCaseResponse(
                id=row.id,
                reportId=row.report_id,
                riskLevel=row.risk_level,
                status=row.status,
                owner=row.owner,
                summary=PrivacySanitizer().sanitize(row.summary),
                handoffSummary=PrivacySanitizer().sanitize(row.handoff_summary),
                acknowledgedBy=row.acknowledged_by,
                acknowledgedAt=row.acknowledged_at,
                createdAt=row.created_at,
                updatedAt=row.updated_at,
            )
            for row in rows
        ]

    def case_notes(self, case_id: int) -> list[CaseNoteResponse]:
        """获取指定危机个案的历次跟进处理笔记流"""
        rows = self.db.query(CaseNote).filter(CaseNote.case_id == case_id).order_by(CaseNote.created_at.asc(),CaseNote.id.asc()).all()
        return [
            CaseNoteResponse(id=row.id, caseId=row.case_id, actor=row.actor, note=PrivacySanitizer().sanitize(row.note), createdAt=row.created_at)
            for row in rows
        ]

    def tool_jobs(self) -> list[ToolJobResponse]:
        """获取后台异步任务队列的作业流转状态"""
        rows = self.db.query(ToolJob).order_by(ToolJob.created_at.desc()).limit(100).all()
        return [
            ToolJobResponse(
                id=row.id,
                reportId=row.report_id,
                kind=row.kind,
                status=row.status,
                attempts=row.attempts,
                maxAttempts=row.max_attempts,
                dependsOnJobId=row.depends_on_job_id,
                runAfter=row.run_after,
                lastError=row.last_error,
                createdAt=row.created_at,
                updatedAt=row.updated_at,
            )
            for row in rows
        ]

    def dead_letters(self) -> list[DeadLetterResponse]:
        """获取执行彻底失败进入死信队列的任务归档记录"""
        rows = self.db.query(DeadLetterRecord).order_by(DeadLetterRecord.created_at.desc()).limit(100).all()
        return [
            DeadLetterResponse(
                id=row.id,
                jobId=row.job_id,
                reportId=row.report_id,
                kind=row.kind,
                reason=row.reason,
                payload=row.payload,
                createdAt=row.created_at,
            )
            for row in rows
        ]

    def agent_run_traces(self) -> list[AgentRunTraceResponse]:
        """获取多智能体全流程决策执行链路追踪数据"""
        rows = self.db.query(AgentRunTrace).order_by(AgentRunTrace.created_at.desc()).limit(100).all()
        responses = []
        for row in rows:
            user = self.db.get(UserAccount, row.user_id)
            session = self.db.get(ChatSession, row.session_id)
            responses.append(
                AgentRunTraceResponse(
                    id=row.id,
                    sessionId=session.public_id if session else "",
                    reportId=row.report_id,
                    username=user.username if user else "",
                    intent=row.intent,
                    riskLevel=row.risk_level,
                    originalInput=PrivacySanitizer().sanitize(row.original_input),
                    sanitizedInput=row.sanitized_input,
                    memoryBrief=row.memory_brief,
                    agentSteps=_loads(row.agent_steps_json, []),
                    retrievedKnowledge=_loads(row.retrieved_knowledge_json, []),
                    responseMessages=_loads(row.response_messages_json, []),
                    assessment=_loads(row.assessment_json, {}),
                    createdAt=row.created_at,
                )
            )
        return responses

    def tool_audits(self) -> list[ToolAuditResponse]:
        """获取工具安全治理策略拦截与放行审计记录"""
        rows = self.db.query(ToolAuditRecord).order_by(ToolAuditRecord.created_at.desc()).limit(100).all()
        return [
            ToolAuditResponse(
                id=row.id,
                jobId=row.job_id,
                reportId=row.report_id,
                toolName=row.tool_name,
                policy=row.policy,
                allowed=row.allowed,
                status=row.status,
                reason=row.reason,
                payload=_loads(row.payload, {}),
                createdAt=row.created_at,
                updatedAt=row.updated_at,
            )
            for row in rows
        ]

    def conversation(self, public_id: str) -> ConversationResponse:
        """查询并组装指定会话的完整历史对话记录"""
        session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id).first()
        if session is None:
            raise ValueError("Session not found")
        rows = self.db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at.asc()).all()
        return ConversationResponse(
            sessionId=session.public_id,
            title=PrivacySanitizer().sanitize(session.title),
            messages=[ConversationMessageResponse(role=row.role, content=PrivacySanitizer().sanitize(row.content), createdAt=row.created_at) for row in rows],
        )

    def _report_response(self, report: PsychologicalReport) -> ReportResponse:
        """内部辅助方法：将 ORM PsychologicalReport 转换为 ReportResponse DTO"""
        user = self.db.get(UserAccount, report.user_id)
        session = self.db.get(ChatSession, report.session_id)
        return ReportResponse(
            id=report.id,
            sessionId=session.public_id if session else "",
            username=user.username if user else "",
            displayName=user.display_name if user else "",
            content=PrivacySanitizer().sanitize(report.content),
            intent=report.intent,
            emotion=report.emotion,
            emotionScore=report.emotion_score,
            riskLevel=report.risk_level,
            summary=PrivacySanitizer().sanitize(report.summary),
            createdAt=report.created_at,
        )


def _loads(raw: str, default):
    """安全反序列化 JSON 字符串，解析损坏或异常时回退返回指定的默认值"""
    import json

    try:
        return json.loads(raw or "")
    except Exception:
        return default


def safe_cell(value: object) -> object:
    """Excel/CSV 单元格注入安全防御清洗器 (Formula Injection Sanitizer)"""
    if isinstance(value, str):
        value = ''.join(c for c in value if ord(c) >= 32 or c in '\n\t')
        if value.lstrip().startswith(('=', '+', '-', '@')):
            value = "'" + value
    return value


def export_reports(db: Session, start: datetime | None = None, end: datetime | None = None, risk: str | None = None) -> bytes:
    """动态筛选并导出心理支持评估台账 Excel 电子表格文件 (Export Assessment Ledger)"""
    statement = select(PsychologicalReport, UserAccount).join(UserAccount, PsychologicalReport.user_id == UserAccount.id)
    if start: statement = statement.where(PsychologicalReport.created_at >= start)
    if end: statement = statement.where(PsychologicalReport.created_at < end)
    if risk: statement = statement.where(PsychologicalReport.risk_level == risk)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = '心理支持报告台账'
    sheet.append(['报告编号', '学生账号', '学生姓名', '会话编号', '咨询意图', '情绪标签', '风险等级', '情绪分值', '报告摘要', '生成时间（UTC）'])
    for report, user in db.execute(statement.order_by(PsychologicalReport.created_at.desc())).all():
        sheet.append([safe_cell(value) for value in [report.id, user.username, user.display_name, report.session_id, report.intent, report.emotion, report.risk_level, report.emotion_score, report.summary, report.created_at]])
    sheet.freeze_panes = 'A2'
    sheet.auto_filter.ref = sheet.dimensions
    for column in sheet.columns:
        sheet.column_dimensions[column[0].column_letter].width = 22 if column[0].column != 9 else 65
    for row in sheet.iter_rows(min_row=2, min_col=10, max_col=10): row[0].number_format = 'yyyy-mm-dd hh:mm:ss'
    output = BytesIO(); workbook.save(output)
    return output.getvalue()

