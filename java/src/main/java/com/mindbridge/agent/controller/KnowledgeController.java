package com.mindbridge.agent.controller;

import com.mindbridge.agent.dto.KnowledgeIngestRequest;
import com.mindbridge.agent.dto.KnowledgeIngestResponse;
import com.mindbridge.agent.service.knowledge.KnowledgeFileService;
import com.mindbridge.agent.service.knowledge.KnowledgeService;
import com.mindbridge.agent.service.knowledge.KnowledgeLifecycleService;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;
import jakarta.validation.Valid;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.MediaType;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/api/admin/knowledge")
/**
 * 管理员知识库维护接口。
 *
 * <p>支持直接写入文本，也支持上传 PDF、Markdown、txt 文件作为 RAG 知识库来源。</p>
 */
public class KnowledgeController {

    private final KnowledgeService knowledgeService;
    private final KnowledgeFileService knowledgeFileService;
    private KnowledgeLifecycleService lifecycle;

    @Autowired(required = false)
    public void setLifecycle(KnowledgeLifecycleService lifecycle) { this.lifecycle = lifecycle; }

    public KnowledgeController(KnowledgeService knowledgeService, KnowledgeFileService knowledgeFileService) {
        this.knowledgeService = knowledgeService;
        this.knowledgeFileService = knowledgeFileService;
    }

    @PostMapping
    public KnowledgeIngestResponse ingest(@Valid @RequestBody KnowledgeIngestRequest request) {
        if (lifecycle == null) return new KnowledgeIngestResponse(request.source(), knowledgeService.ingest(request.source(), request.content()));
        var version=lifecycle.importText(request.documentId(),request.source(),request.content());
        if (!version.getStatus().equals("ACTIVE")) throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Indexing failed");
        return new KnowledgeIngestResponse(request.source(),version.getChunkCount(),version.getDocumentId());
    }

    public Mono<KnowledgeIngestResponse> ingestFile(FilePart file) { return ingestFileWithId(file,null); }

    @PostMapping(value = "/file", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<KnowledgeIngestResponse> ingestFileWithId(@RequestPart("file") FilePart file,
                                                          @RequestParam(value="document_id",required=false) String documentId) {
        return DataBufferUtils.join(file.content(), 10 * 1024 * 1024).flatMap(buffer -> {
            byte[] bytes=readBytes(buffer);
            return Mono.fromCallable(() -> {
                if(lifecycle==null) return new KnowledgeIngestResponse(file.filename(),knowledgeFileService.ingest(file.filename(),bytes));
                var version=knowledgeFileService.ingestVersion(file.filename(),bytes,documentId);
                if(!version.getStatus().equals("ACTIVE")) throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,"Indexing failed");
                return new KnowledgeIngestResponse(version.getSource(),version.getChunkCount(),version.getDocumentId());
            }).subscribeOn(reactor.core.scheduler.Schedulers.boundedElastic());
        }).onErrorMap(org.springframework.core.io.buffer.DataBufferLimitException.class, e -> new ResponseStatusException(HttpStatus.PAYLOAD_TOO_LARGE,"File exceeds 10 MB",e));
    }

    @PostMapping(value="/preview",consumes=MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<com.mindbridge.agent.service.knowledge.DocumentConverter.Converted> preview(@RequestPart("file") FilePart file) {
        return DataBufferUtils.join(file.content(),10*1024*1024).flatMap(buffer -> {
            byte[] bytes=readBytes(buffer);
            return Mono.fromCallable(()->knowledgeFileService.preview(file.filename(),bytes)).subscribeOn(reactor.core.scheduler.Schedulers.boundedElastic());
        }).onErrorMap(org.springframework.core.io.buffer.DataBufferLimitException.class, e -> new ResponseStatusException(HttpStatus.PAYLOAD_TOO_LARGE,"File exceeds 10 MB",e));
    }

    @GetMapping("/documents")
    public List<com.mindbridge.agent.domain.KnowledgeVersion> documents() {
        return lifecycle.allVersions().stream().filter(v -> !v.getStatus().equals("DELETED")).toList();
    }

    @GetMapping("/document-content")
    public Map<String, Object> documentContent(@RequestParam("document_id") String documentId, @RequestParam(defaultValue = "0") int offset) {
        if (offset < 0) throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY, "Invalid offset");
        var rows=lifecycle.documentChunks(documentId);
        if (rows.isEmpty()) throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Document not found");
        return Map.of("total",rows.size(),"chunks",rows.stream().skip(offset).limit(50).map(c->Map.of("id",c.getId(),"content",c.getContent())).toList());
    }

    @DeleteMapping("/documents")
    public Map<String, Integer> deleteDocument(@RequestParam("document_id") String documentId) {
        int count=lifecycle.documentChunks(documentId).size();
        lifecycle.deleteDocument(documentId);
        if (lifecycle.allVersions().stream().anyMatch(v -> v.getDocumentId().equals(documentId) && v.getStatus().equals("DELETE_PENDING")))
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Vector deletion is pending; retry deletion");
        return Map.of("deletedChunks",count);
    }

    @PostMapping("/versions/{id}/retry")
    public Map<String, Integer> retry(@PathVariable String id) {
        var v=lifecycle.retry(id);
        if (!v.getStatus().equals("ACTIVE")) throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Indexing failed");
        return Map.of("chunks",v.getChunkCount());
    }

    @PostMapping("/versions/{id}/cleanup")
    public Map<String, Integer> cleanup(@PathVariable String id) { return Map.of("chunks",lifecycle.cleanup(id)); }

    @PostMapping("/backup")
    public Map<String, Object> backup() { throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Vector backup is not supported"); }

    @GetMapping("/status")
    public Map<String,Object> status() { return lifecycle.status(); }

    @PostMapping("/rebuild-vector")
    public Map<String, Object> rebuild() {
        try { return Map.of("indexedChunks", lifecycle.rebuild()); }
        catch(RuntimeException e) { throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,"Vector rebuild failed",e); }
    }

    @org.springframework.web.bind.annotation.ExceptionHandler(java.util.NoSuchElementException.class)
    public org.springframework.http.ResponseEntity<Map<String,String>> notFound(java.util.NoSuchElementException e) {
        return org.springframework.http.ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of("detail","Knowledge item not found"));
    }
    @org.springframework.web.bind.annotation.ExceptionHandler(IllegalStateException.class)
    public org.springframework.http.ResponseEntity<Map<String,String>> conflict(IllegalStateException e) {
        return org.springframework.http.ResponseEntity.status(HttpStatus.CONFLICT).body(Map.of("detail","Knowledge operation conflicts with current state"));
    }
    @org.springframework.web.bind.annotation.ExceptionHandler(IllegalArgumentException.class)
    public org.springframework.http.ResponseEntity<Map<String,String>> invalid(IllegalArgumentException e) {
        return org.springframework.http.ResponseEntity.unprocessableEntity().body(Map.of("detail","Invalid knowledge input"));
    }

    private byte[] readBytes(DataBuffer dataBuffer) {
        try {
            ByteArrayOutputStream output = new ByteArrayOutputStream(dataBuffer.readableByteCount());
            dataBuffer.asInputStream().transferTo(output);
            return output.toByteArray();
        } catch (IOException exception) {
            throw new IllegalArgumentException("文件读取失败：" + exception.getMessage());
        } finally {
            DataBufferUtils.release(dataBuffer);
        }
    }
}
