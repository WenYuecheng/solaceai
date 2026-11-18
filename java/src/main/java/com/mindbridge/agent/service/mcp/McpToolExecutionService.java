package com.mindbridge.agent.service.mcp;

import com.mindbridge.agent.config.MindBridgeProperties;
import java.nio.file.Path;
import java.time.Instant;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class McpToolExecutionService {

    private static final Logger logger = LoggerFactory.getLogger(McpToolExecutionService.class);

    private final Path excelPath;
    private final JavaMailSender mailSender;
    private final MindBridgeProperties properties;

    public McpToolExecutionService(JavaMailSender mailSender, MindBridgeProperties properties) {
        this.mailSender = mailSender;
        this.properties = properties;
        this.excelPath = Path.of(properties.getMcp().getExcel().getLocalPath());
    }

    public String writeExcelReport(
            Long reportId,
            Long userId,
            String username,
            String sessionId,
            String intent,
            String emotion,
            double emotionScore,
            String riskLevel,
            double confidence,
            String summary,
            String content,
            String createdAt
    ) {
        AtomicReportWorkbook.write(excelPath, reportId, row -> {
            cell(row, 0).setCellValue(reportId.toString());
            cell(row, 1).setCellValue(nullSafe(userId));
            cell(row, 2).setCellValue(nullSafe(username));
            cell(row, 3).setCellValue(nullSafe(sessionId));
            cell(row, 4).setCellValue(nullSafe(intent));
            cell(row, 5).setCellValue(nullSafe(emotion));
            cell(row, 6).setCellValue(emotionScore);
            cell(row, 7).setCellValue(nullSafe(riskLevel));
            cell(row, 8).setCellValue(confidence);
            cell(row, 9).setCellValue(nullSafe(summary));
            cell(row, 10).setCellValue(nullSafe(content));
            cell(row, 11).setCellValue(nullSafe(createdAt));
        });
        return "Excel report written: " + reportId;
    }

    public String sendRiskAlert(
            String recipient,
            Long reportId,
            Long userId,
            String username,
            String displayName,
            String riskLevel,
            String emotion,
            double emotionScore,
            String summary,
            String content
    ) {
        if ("log".equalsIgnoreCase(properties.getMcp().getEmail().getMcpServerDeliveryMode())) {
            logger.warn(
                    "MCP risk alert recipient={} reportId={} userId={} username={} riskLevel={} summary={}",
                    recipient, reportId, userId, username, riskLevel, summary);
            return "Risk alert logged: " + reportId;
        }

        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom(properties.getMcp().getEmail().getFrom());
        message.setTo(recipient);
        message.setSubject("【高危心理预警】学生用户 %s 存在高风险信号".formatted(username));
        message.setText("""
                系统在对话中监测到 1 名学生出现高风险心理状态，请及时关注并干预。

                【预警信息如下】
                报告ID：%s
                用户ID：%s
                学生：%s
                对话内容：%s
                情绪判定：%s
                综合情绪得分：%.2f
                风险等级：%s
                判断摘要：%s
                发送时间：%s

                """.formatted(
                reportId,
                userId,
                nullSafe(displayName),
                nullSafe(content),
                nullSafe(emotion),
                emotionScore,
                nullSafe(riskLevel),
                nullSafe(summary),
                Instant.now()));
        mailSender.send(message);
        return "Risk alert sent: " + reportId;
    }

    private Cell cell(Row row, int index) {
        return row.createCell(index);
    }

    private String nullSafe(String value) {
        return value == null ? "" : value;
    }

    private long nullSafe(Long value) {
        return value == null ? 0L : value;
    }
}
