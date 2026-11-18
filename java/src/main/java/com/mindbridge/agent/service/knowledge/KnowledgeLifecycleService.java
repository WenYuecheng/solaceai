package com.mindbridge.agent.service.knowledge;

import com.mindbridge.agent.domain.KnowledgeChunk;
import com.mindbridge.agent.domain.KnowledgeVersion;
import com.mindbridge.agent.repository.KnowledgeChunkRepository;
import com.mindbridge.agent.repository.KnowledgeVersionRepository;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

/** Serializes local imports, indexing and deletion around durable version transitions. */
public class KnowledgeLifecycleService {
    private final KnowledgeChunkRepository chunks;
    private final KnowledgeVersionRepository versions;
    private final TransactionTemplate tx;
    private final KnowledgeIndex index;
    private final String tenantId, knowledgeBaseId;
    private final KnowledgeChunker chunker=new KnowledgeChunker();
    private KnowledgeWriterLock lock=new KnowledgeWriterLock(java.nio.file.Path.of("data/knowledge/writer.lock"));
    public void setWriterLockPath(String path) { lock=new KnowledgeWriterLock(java.nio.file.Path.of(path)); }
    public KnowledgeLifecycleService(KnowledgeChunkRepository chunks, KnowledgeVersionRepository versions, PlatformTransactionManager manager, KnowledgeIndex index) {
        this(chunks,versions,manager,index,"default","default");
    }
    public KnowledgeLifecycleService(KnowledgeChunkRepository chunks, KnowledgeVersionRepository versions, PlatformTransactionManager manager, KnowledgeIndex index,String tenantId,String knowledgeBaseId) {
        if(tenantId==null || knowledgeBaseId==null || !tenantId.matches("[A-Za-z0-9_-]{1,64}") || !knowledgeBaseId.matches("[A-Za-z0-9_-]{1,64}")) throw new IllegalArgumentException("Invalid knowledge scope");
        this.chunks=chunks; this.versions=versions; this.tx=new TransactionTemplate(manager); this.index=index; this.tenantId=tenantId; this.knowledgeBaseId=knowledgeBaseId;
    }
    private List<KnowledgeVersion> documentVersions(String id) { return versions.findByTenantIdAndKnowledgeBaseIdAndDocumentIdOrderByOrdinalAsc(tenantId,knowledgeBaseId,id); }
    private KnowledgeVersion scopedVersion(String id) { var v=versions.findById(id).orElseThrow(); if(!tenantId.equals(v.getTenantId()) || !knowledgeBaseId.equals(v.getKnowledgeBaseId())) throw new NoSuchElementException("Version not found"); return v; }
    public boolean includesLegacy() { return "default".equals(tenantId) && "default".equals(knowledgeBaseId); }
    public boolean vectorEnabled() { return index instanceof MilvusGateway; }
    public List<SearchResult> vectorSearch(String query,List<KnowledgeChunk> active,int limit) {
        if (!(index instanceof MilvusGateway gateway)) return List.of();
        return gateway.search(query,active,limit);
    }
    public List<SearchResult> crossRerank(String query,List<SearchResult> candidates,int limit) {
        if (!(index instanceof MilvusGateway gateway)) return candidates.stream().limit(limit).toList();
        return gateway.rerank(query,candidates,limit);
    }
    public KnowledgeVersion importText(String documentId,String source,String content) {
        if(source==null || source.isBlank() || source.length()>180 || content==null || content.isBlank()) throw new IllegalArgumentException("Invalid document");
        List<String> parts=chunker.chunk(content,512,64);
        return importParts(documentId,source,hash(content),parts,Collections.nCopies(parts.size(),null));
    }
    public KnowledgeVersion importConverted(String documentId, DocumentConverter.Converted converted) {
        List<String> parts=new ArrayList<>(); List<String> metadata=new ArrayList<>();
        for(var node:converted.structuredChunks()) {
            if (!node.isObject() || !node.path("content").isTextual() || node.path("content").asText().isBlank()) throw new IllegalArgumentException("Invalid structured chunk");
            parts.add(node.path("content").asText()); metadata.add(node.toString());
        }
        if (parts.isEmpty()) { parts=chunker.chunk(converted.markdown(),512,64); metadata=Collections.nCopies(parts.size(),null); }
        return importParts(documentId,converted.source(),converted.contentHash(),parts,metadata);
    }
    private KnowledgeVersion importParts(String documentId,String source,String contentHash,List<String> parts,List<String> metadata) {
        try(var writer=lock.acquire()) {
            String doc=documentId==null ? UUID.randomUUID().toString() : documentId;
            if (!doc.matches("[A-Za-z0-9_-]{1,64}")) throw new IllegalArgumentException("Invalid document ID");
            KnowledgeVersion v=tx.execute(status->{
                List<KnowledgeVersion> previous=documentVersions(doc);
                if (!previous.isEmpty() && previous.stream().noneMatch(x->x.getStatus().equals("ACTIVE")) && previous.stream().anyMatch(x->x.getStatus().equals("DELETED") || x.getStatus().equals("DELETE_PENDING"))) throw new IllegalStateException("Document is deleted");
                int next=previous.stream().mapToInt(KnowledgeVersion::getOrdinal).max().orElse(0)+1;
                KnowledgeVersion created=new KnowledgeVersion(doc,source,next,contentHash,parts.size()); created.setScope(tenantId,knowledgeBaseId); created=versions.save(created);
                for(int i=0;i<parts.size();i++){ KnowledgeChunk c=new KnowledgeChunk(); c.setVersionId(created.getId()); c.setSource(source); c.setSourceIndex(i); c.setContent(parts.get(i)); c.setMetadataJson(metadata.get(i)); chunks.save(c); }
                return created;
            });
            return indexAndActivate(v.getId());
        }
    }
    public KnowledgeVersion retry(String id) {
        try(var writer=lock.acquire()) {
            KnowledgeVersion v=tx.execute(status->{
                KnowledgeVersion found=scopedVersion(id);
                if (!List.of("FAILED", "PENDING").contains(found.getStatus())) throw new IllegalStateException("Only failed or interrupted versions can be retried");
                boolean newer=documentVersions(found.getDocumentId()).stream().anyMatch(x->x.getStatus().equals("ACTIVE") && x.getOrdinal()>found.getOrdinal());
                if (newer) throw new IllegalStateException("A newer version is active");
                found.setStatus("PENDING"); return versions.save(found);
            });
            return indexAndActivate(v.getId());
        }
    }
    private KnowledgeVersion indexAndActivate(String id) {
        try {
            tx.executeWithoutResult(s->{KnowledgeVersion current=scopedVersion(id);current.startAttempt();versions.save(current);});
            List<KnowledgeChunk> rows=tx.execute(s->chunks.findByVersionIdOrderBySourceIndexAsc(id));
            index.upsert(rows);
            return tx.execute(s->{
                KnowledgeVersion target=scopedVersion(id); target.setStatus("INDEXED"); versions.save(target);
                for(KnowledgeVersion prior:documentVersions(target.getDocumentId())) if(prior.getStatus().equals("ACTIVE")){prior.setStatus("ARCHIVED");versions.save(prior);}
                target.setStatus("ACTIVE"); return versions.save(target);
            });
        } catch (Exception e) {
            return tx.execute(s->{KnowledgeVersion failed=scopedVersion(id); failed.setStatus("FAILED"); failed.setLastError(e.getClass().getSimpleName()); return versions.save(failed);});
        }
    }
    public void deleteDocument(String doc) {
        try(var writer=lock.acquire()) {
            List<KnowledgeVersion> pending=tx.execute(s->{
                List<KnowledgeVersion> found=documentVersions(doc);
                if(found.isEmpty()) throw new NoSuchElementException("Document not found");
                for(KnowledgeVersion v:found) if(!v.getStatus().equals("DELETED")){v.setStatus("DELETE_PENDING"); versions.save(v);}
                return found.stream().filter(v->v.getStatus().equals("DELETE_PENDING")).toList();
            });
            for(KnowledgeVersion v:pending) {
                try {
                    List<Long> ids=tx.execute(s->chunks.findByVersionIdOrderBySourceIndexAsc(v.getId()).stream().map(KnowledgeChunk::getId).toList());
                    index.delete(ids);
                    tx.executeWithoutResult(s->{KnowledgeVersion current=scopedVersion(v.getId()); current.setStatus("DELETED"); versions.save(current);});
                } catch(Exception ignored) { /* Persisted tombstone permits a later retry. */ }
            }
        }
    }
    public int cleanup(String id) {
        try(var writer=lock.acquire()) {
            List<Long> ids=tx.execute(s->{
                KnowledgeVersion v=scopedVersion(id);
                if (!List.of("ARCHIVED", "FAILED", "PENDING", "DELETE_PENDING").contains(v.getStatus())) throw new IllegalStateException("Only inactive versions may be cleaned");
                v.setStatus("DELETE_PENDING"); versions.save(v);
                return chunks.findByVersionIdOrderBySourceIndexAsc(id).stream().map(KnowledgeChunk::getId).toList();
            });
            try { index.delete(ids); } catch(RuntimeException e) { throw new org.springframework.web.server.ResponseStatusException(org.springframework.http.HttpStatus.SERVICE_UNAVAILABLE,"Vector deletion remains pending",e); }
            tx.executeWithoutResult(s->{chunks.deleteByVersionId(id); KnowledgeVersion v=scopedVersion(id); v.setStatus("DELETED"); versions.save(v);});
            return ids.size();
        }
    }
    public Map<String,Object> versionMetadata(String id) {
        return tx.execute(s -> versions.findById(id).filter(v -> tenantId.equals(v.getTenantId()) && knowledgeBaseId.equals(v.getKnowledgeBaseId())).<Map<String,Object>>map(v -> Map.of("document_id",v.getDocumentId(),"version_id",v.getId(),"document_version",v.getOrdinal(),"content_hash",v.getContentHash(),"tenant_id",tenantId,"knowledge_base_id",knowledgeBaseId)).orElseGet(Map::of));
    }
    public List<KnowledgeVersion> allVersions() { return tx.execute(s->versions.findByTenantIdAndKnowledgeBaseId(tenantId,knowledgeBaseId)); }
    public List<KnowledgeChunk> documentChunks(String documentId) {
        return tx.execute(s->documentVersions(documentId).stream().filter(v->v.getStatus().equals("ACTIVE"))
                .flatMap(v->chunks.findByVersionIdOrderBySourceIndexAsc(v.getId()).stream()).toList());
    }
    public Map<String,Object> status() {
        Map<String,Object> result=new LinkedHashMap<>();
        if(index instanceof MilvusGateway gateway) result.putAll(gateway.status());
        else { result.put("backend","bm25"); result.put("workerHealthy",false); result.put("vectorHealthy",false); result.put("rerankerLoaded",false); }
        result.put("activeChunks",activeChunks().size()); result.put("versions",allVersions().size()); return result;
    }
    public int rebuild() { try(var writer=lock.acquire()) { return index.rebuild(activeChunks()); } }
    public List<KnowledgeChunk> activeChunks() {
        return tx.execute(s->versions.findByTenantIdAndKnowledgeBaseIdAndStatus(tenantId,knowledgeBaseId,"ACTIVE").stream().flatMap(v->chunks.findByVersionIdOrderBySourceIndexAsc(v.getId()).stream()).toList());
    }
    private static String hash(String content) {
        try { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(content.getBytes(StandardCharsets.UTF_8))); }
        catch(Exception e){throw new IllegalStateException(e);}
    }
}
