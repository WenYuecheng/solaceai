package com.mindbridge.agent.service.knowledge;

import java.nio.charset.StandardCharsets;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.mindbridge.agent.config.ParityKnowledgeProperties;
import com.mindbridge.agent.domain.KnowledgeVersion;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.Locale;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.stereotype.Service;

@Service
/**
 * 管理员文件上传知识库服务。
 *
 * <p>负责文件大小校验、类型识别和文本抽取，抽取后的文本交给 KnowledgeService 处理。</p>
 */
public class KnowledgeFileService {

    private static final int MAX_FILE_BYTES = 10 * 1024 * 1024;

    private final KnowledgeService knowledgeService;
    private final DocumentConverter converter=new DocumentConverter(new ObjectMapper());
    private ParityKnowledgeProperties parity = new ParityKnowledgeProperties();
    private KnowledgeLifecycleService lifecycle;
    @Autowired(required=false) public void setParity(ParityKnowledgeProperties parity) { this.parity=parity; }
    @Autowired(required=false) public void setLifecycle(KnowledgeLifecycleService lifecycle) { this.lifecycle=lifecycle; }
    public DocumentConverter.Converted preview(String filename,byte[] bytes) {
        DocumentConverter.validateFilename(filename);
        if(bytes==null || bytes.length==0 || bytes.length>MAX_FILE_BYTES) throw new IllegalArgumentException("Invalid document size");
        String lower=filename.toLowerCase(Locale.ROOT);
        if(lower.endsWith(".metadata.json")) return converter.convertMetadata(filename,bytes);
        if(lower.endsWith(".txt") || lower.endsWith(".md") || lower.endsWith(".markdown")) {
            try {
                String text=StandardCharsets.UTF_8.newDecoder().decode(java.nio.ByteBuffer.wrap(bytes)).toString();
                DocumentConverter.validateQuality(text);
                String digest=java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes));
                return new DocumentConverter.Converted(filename,text,digest,"text","1",new ObjectMapper().createArrayNode(),java.util.List.of());
            } catch(java.nio.charset.CharacterCodingException | java.security.NoSuchAlgorithmException e) { throw new IllegalArgumentException("Invalid UTF-8 document",e); }
        }
        if(!lower.matches(".*\\.(pdf|docx|pptx|xlsx|html|csv|png|jpg|jpeg)$")) throw new IllegalArgumentException("Unsupported document format");
        return converter.convertExternal(filename,bytes,parity.getDocumentPython(),parity.getDocumentModulePath());
    }
    public KnowledgeVersion ingestVersion(String filename,byte[] bytes,String documentId) {
        if(lifecycle==null) throw new IllegalStateException("Knowledge lifecycle unavailable");
        return lifecycle.importConverted(documentId,preview(filename,bytes));
    }

    public KnowledgeFileService(KnowledgeService knowledgeService) {
        this.knowledgeService = knowledgeService;
    }

    public int ingest(String filename, byte[] bytes) {
        // 文件上传入口只负责校验和抽取文本，真正切块、向量化、落库交给 KnowledgeService。
        if (bytes.length == 0) {
            throw new IllegalArgumentException("文件内容为空");
        }
        if (bytes.length > MAX_FILE_BYTES) {
            throw new IllegalArgumentException("文件不能超过 10MB");
        }
        String source = sanitizeSource(filename);
        String text = extractText(source, bytes);
        if (text.isBlank()) {
            throw new IllegalArgumentException("没有从文件中解析出可用文本");
        }
        return knowledgeService.ingest(source, text);
    }

    private String extractText(String filename, byte[] bytes) {
        String lower = filename.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".pdf")) {
            return extractPdf(bytes);
        }
        // Markdown 和 txt 都按 UTF-8 文本处理，适合管理员维护轻量知识库。
        if (lower.endsWith(".md") || lower.endsWith(".markdown") || lower.endsWith(".txt")) {
            return new String(bytes, StandardCharsets.UTF_8);
        }
        throw new IllegalArgumentException("仅支持 PDF、Markdown 和 txt 文件");
    }

    private String extractPdf(byte[] bytes) {
        try (PDDocument document = Loader.loadPDF(bytes)) {
            return new PDFTextStripper().getText(document);
        } catch (Exception exception) {
            throw new IllegalArgumentException("PDF 文本解析失败：" + exception.getMessage());
        }
    }

    private String sanitizeSource(String filename) {
        String source = filename == null || filename.isBlank() ? "uploaded-knowledge" : filename.trim();
        // source 会进入数据库和后台列表，去掉路径分隔符避免显示本地路径。
        source = source.replaceAll("[\\\\/]+", "-");
        return source.length() > 180 ? source.substring(source.length() - 180) : source;
    }
}
