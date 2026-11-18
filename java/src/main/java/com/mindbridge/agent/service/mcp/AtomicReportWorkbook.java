package com.mindbridge.agent.service.mcp;

import java.io.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

/** Report id is the ledger key. The sidecar lock remains stable across atomic replacement. */
final class AtomicReportWorkbook {
    private static final ConcurrentHashMap<Path, Object> LOCKS = new ConcurrentHashMap<>();
    private static final String[] HEADERS = {"报告ID","用户ID","账号","会话ID","意图","情绪标签","情绪总分","风险等级","置信度","判断摘要","对话内容","对话时间"};

    static void write(Path configuredPath, Long reportId, Consumer<Row> writer) {
        if (reportId == null) throw new IllegalArgumentException("A persisted report id is required");
        Path path = configuredPath.toAbsolutePath().normalize();
        synchronized (LOCKS.computeIfAbsent(path, ignored -> new Object())) {
            Path temporary = null;
            try {
                Files.createDirectories(path.getParent());
                try (var channel = FileChannel.open(path.resolveSibling(path.getFileName()+".lock"),
                        StandardOpenOption.CREATE, StandardOpenOption.WRITE); var fileLock = channel.lock();
                     Workbook workbook = open(path)) {
                    Sheet sheet = workbook.getNumberOfSheets() == 0 ? workbook.createSheet("reports") : workbook.getSheetAt(0);
                    if(sheet.getPhysicalNumberOfRows()==0) {
                        Row header=sheet.createRow(0);
                        for(int i=0;i<HEADERS.length;i++) header.createCell(i).setCellValue(HEADERS[i]);
                    }
                    DataFormatter formatter = new DataFormatter(java.util.Locale.ROOT);
                    for(Row row:sheet) {
                        if(row.getRowNum()>0 && reportId.toString().equals(formatter.formatCellValue(row.getCell(0)))) return;
                    }
                    writer.accept(sheet.createRow(sheet.getLastRowNum()+1));
                    temporary=Files.createTempFile(path.getParent(), ".reports-", ".xlsx");
                    try(OutputStream output=Files.newOutputStream(temporary)) {workbook.write(output);}
                    try(var output=FileChannel.open(temporary,StandardOpenOption.WRITE)){output.force(true);}
                    // Fail closed if the filesystem cannot atomically replace the workbook.
                    Files.move(temporary,path,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);
                }
            } catch(Exception error) {throw new IllegalStateException("Failed to atomically write report workbook",error);}
            finally {if(temporary!=null) try{Files.deleteIfExists(temporary);}catch(IOException ignored){}}
        }
    }
    private static Workbook open(Path path) throws Exception {
        if(!Files.exists(path))return new XSSFWorkbook();
        try(InputStream input=Files.newInputStream(path)){return WorkbookFactory.create(input);}
    }
}
