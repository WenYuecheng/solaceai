package com.mindbridge.agent.service.mcp;

import com.mindbridge.agent.config.MindBridgeProperties;
import com.mindbridge.agent.domain.PsychologicalReport;
import java.nio.file.Path;
import java.time.ZoneId;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;

/**
 * 本地 Excel 文件写入实现。
 *
 * <p>适合演示环境使用，会把报告追加到 data 目录下的工作簿中。</p>
 */
public class LocalExcelReportWriter implements ExcelReportWriter {

    private final Path path;

    public LocalExcelReportWriter(MindBridgeProperties properties) {
        this.path = Path.of(properties.getMcp().getExcel().getLocalPath());
    }

    @Override
    public void write(PsychologicalReport report) {
        AtomicReportWorkbook.write(path, report.getId(), row -> writeReport(row, report));
    }

    private void writeReport(Row row, PsychologicalReport report) {
        cell(row, 0).setCellValue(report.getId().toString());
        cell(row, 1).setCellValue(nullSafe(report.getUser().getId()));
        cell(row, 2).setCellValue(report.getUser().getUsername());
        cell(row, 3).setCellValue(report.getSession() == null ? "" : report.getSession().getPublicId());
        cell(row, 4).setCellValue(report.getIntent().name());
        cell(row, 5).setCellValue(report.getEmotion().name());
        cell(row, 6).setCellValue(report.getEmotionScore());
        cell(row, 7).setCellValue(report.getRiskLevel().name());
        cell(row, 8).setCellValue(report.getConfidence());
        cell(row, 9).setCellValue(report.getSummary());
        cell(row, 10).setCellValue(report.getContent());
        cell(row, 11).setCellValue(report.getCreatedAt().atZone(ZoneId.systemDefault()).toLocalDateTime().toString());
    }

    private Cell cell(Row row, int index) {
        return row.createCell(index);
    }

    private long nullSafe(Long value) {
        return value == null ? 0L : value;
    }
}
