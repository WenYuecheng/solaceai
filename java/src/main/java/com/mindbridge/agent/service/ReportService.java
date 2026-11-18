package com.mindbridge.agent.service;

import com.mindbridge.agent.domain.ChatMessage;
import com.mindbridge.agent.domain.ChatSession;
import com.mindbridge.agent.domain.PsychologicalReport;
import com.mindbridge.agent.domain.RiskLevel;
import com.mindbridge.agent.domain.ToolStatus;
import com.mindbridge.agent.domain.UserAccount;
import com.mindbridge.agent.dto.AlertRecordResponse;
import com.mindbridge.agent.dto.AgentRunTraceResponse;
import com.mindbridge.agent.dto.ConversationResponse;
import com.mindbridge.agent.dto.ExcelRecordResponse;
import com.mindbridge.agent.repository.AlertRecordRepository;
import com.mindbridge.agent.repository.ChatMessageRepository;
import com.mindbridge.agent.repository.ChatSessionRepository;
import com.mindbridge.agent.repository.PsychologicalReportRepository;
import java.util.List;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.time.DateTimeException;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.ArrayList;
import jakarta.persistence.criteria.Predicate;
import org.apache.poi.ss.util.CellRangeAddress;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

@Service
/**
 * 管理员后台数据查询服务。
 *
 * <p>封装报告列表、Excel 写入记录、预警记录和完整会话读取逻辑。</p>
 */
public class ReportService {

    private static final String ROLE_ADMIN = "ROLE_ADMIN";

    private final PsychologicalReportRepository psychologicalReportRepository;
    private final ChatSessionRepository chatSessionRepository;
    private final ChatMessageRepository chatMessageRepository;
    private final AlertRecordRepository alertRecordRepository;
    private final AgentRunTraceService agentRunTraceService;

    public ReportService(
            PsychologicalReportRepository psychologicalReportRepository,
            ChatSessionRepository chatSessionRepository,
            ChatMessageRepository chatMessageRepository,
            AlertRecordRepository alertRecordRepository,
            AgentRunTraceService agentRunTraceService
    ) {
        this.psychologicalReportRepository = psychologicalReportRepository;
        this.chatSessionRepository = chatSessionRepository;
        this.chatMessageRepository = chatMessageRepository;
        this.alertRecordRepository = alertRecordRepository;
        this.agentRunTraceService = agentRunTraceService;
    }

    @Transactional(readOnly = true)
    public List<PsychologicalReport> myReports(Long userId) {
        return psychologicalReportRepository.findTop50ByUser_IdOrderByCreatedAtDesc(userId);
    }

    @Transactional(readOnly = true)
    public List<PsychologicalReport> latestReports() {
        // 管理员后台只展示学生对话产生的报告，避免管理员测试消息混入统计大屏。
        return psychologicalReportRepository.findTop100ByOrderByCreatedAtDesc().stream()
                .filter(ReportService::isStudentReport)
                .toList();
    }

    @Transactional(readOnly = true)
    public byte[] exportReports(String start, String end, String risk) throws IOException {
        Instant from;
        Instant until;
        try {
            from = start == null || start.isEmpty() ? null : LocalDate.parse(start).atStartOfDay().toInstant(ZoneOffset.UTC);
            until = end == null || end.isEmpty() ? null : LocalDate.parse(end).plusDays(1).atStartOfDay().toInstant(ZoneOffset.UTC);
        } catch (DateTimeException error) {
            throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY, "日期格式应为 YYYY-MM-DD");
        }
        RiskLevel level;
        try {
            level = risk == null || risk.isEmpty() ? null : RiskLevel.valueOf(risk);
        } catch (IllegalArgumentException error) {
            throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY, "风险等级无效");
        }
        if (from != null && until != null && !from.isBefore(until)) {
            throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY, "结束日期不能早于开始日期");
        }

        var reports = psychologicalReportRepository.findAll((root, query, criteria) -> {
            var filters = new ArrayList<Predicate>();
            if (from != null) filters.add(criteria.greaterThanOrEqualTo(root.get("createdAt"), from));
            if (until != null) filters.add(criteria.lessThan(root.get("createdAt"), until));
            if (level != null) filters.add(criteria.equal(root.get("riskLevel"), level));
            return criteria.and(filters.toArray(Predicate[]::new));
        }, Sort.by(Sort.Direction.DESC, "createdAt", "id"));
        try (var workbook = new XSSFWorkbook(); var output = new ByteArrayOutputStream()) {
            var sheet = workbook.createSheet("心理支持报告台账");
            String[] headers = {"报告编号", "学生账号", "学生姓名", "会话编号", "咨询意图", "情绪标签",
                    "风险等级", "情绪分值", "报告摘要", "生成时间（UTC）"};
            var header = sheet.createRow(0);
            for (int column = 0; column < headers.length; column++) {
                header.createCell(column).setCellValue(headers[column]);
                sheet.setColumnWidth(column, (column == 8 ? 65 : 22) * 256);
            }
            var dateStyle = workbook.createCellStyle();
            dateStyle.setDataFormat(workbook.createDataFormat().getFormat("yyyy-mm-dd hh:mm:ss"));
            int index = 0;
            for (var report : reports) {
                if (!isStudentReport(report)) continue;
                var row = sheet.createRow(++index);
                row.createCell(0).setCellValue(report.getId());
                row.createCell(1).setCellValue(report.getUser().getUsername());
                row.createCell(2).setCellValue(report.getUser().getDisplayName());
                if (report.getSession() != null) row.createCell(3).setCellValue(report.getSession().getId());
                row.createCell(4).setCellValue(report.getIntent().name());
                row.createCell(5).setCellValue(report.getEmotion().name());
                row.createCell(6).setCellValue(report.getRiskLevel().name());
                row.createCell(7).setCellValue(report.getEmotionScore());
                row.createCell(8).setCellValue(report.getSummary());
                var date = row.createCell(9);
                date.setCellValue(report.getCreatedAt().atOffset(ZoneOffset.UTC).toLocalDateTime());
                date.setCellStyle(dateStyle);
            }
            sheet.createFreezePane(0, 1);
            sheet.setAutoFilter(new CellRangeAddress(0, index, 0, headers.length - 1));
            workbook.write(output);
            return output.toByteArray();
        }
    }

    @Transactional(readOnly = true)
    public List<ExcelRecordResponse> excelRecords() {
        return psychologicalReportRepository
                .findTop100ByExcelStatusOrderByCreatedAtDesc(ToolStatus.SUCCESS)
                .stream()
                .filter(ReportService::isStudentReport)
                .map(ExcelRecordResponse::from)
                .toList();
    }

    @Transactional(readOnly = true)
    public List<AlertRecordResponse> alertRecords() {
        return alertRecordRepository.findTop100ByOrderByCreatedAtDesc().stream()
                .filter(alertRecord -> isStudentReport(alertRecord.getReport()))
                .map(AlertRecordResponse::from)
                .toList();
    }

    @Transactional(readOnly = true)
    public ConversationResponse conversation(String sessionId) {
        ChatSession session = chatSessionRepository.findByPublicId(sessionId)
                .orElseThrow(() -> new IllegalArgumentException("Conversation not found"));
        // 管理员只能查看学生会话；非学生会话统一按不存在处理，减少后台数据误展示。
        if (!isStudentUser(session.getUser())) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Conversation not found");
        }
        List<ChatMessage> messages = chatMessageRepository.findBySession_PublicIdOrderByCreatedAtAsc(sessionId);
        List<AgentRunTraceResponse> runTraces = agentRunTraceService.tracesForSession(sessionId);
        return ConversationResponse.from(session, messages, runTraces);
    }

    private static boolean isStudentReport(PsychologicalReport report) {
        return report != null && isStudentUser(report.getUser());
    }

    private static boolean isStudentUser(UserAccount user) {
        return user != null && !user.getRoles().contains(ROLE_ADMIN);
    }
}
