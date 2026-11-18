package com.mindbridge.agent.domain;

import jakarta.persistence.*;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name="knowledge_versions", uniqueConstraints=@UniqueConstraint(name="uk_knowledge_scope_ordinal", columnNames={"tenantId","knowledgeBaseId","documentId","ordinal"}))
public class KnowledgeVersion {
    @Id @Column(length=36) private String id=UUID.randomUUID().toString();
    @Column(nullable=false,length=64) private String documentId;
    @Column(nullable=false,length=64) private String tenantId="default";
    @Column(nullable=false,length=64) private String knowledgeBaseId="default";
    @Column(nullable=false,length=180) private String source;
    @Column(nullable=false) private int ordinal;
    @Column(nullable=false,length=24) private String status="PENDING";
    @Column(nullable=false,length=64) private String contentHash;
    @Column(nullable=false) private int chunkCount;
    @Column(nullable=false) private int attempts;
    @Column(length=255) private String lastError;
    @Column(nullable=false,updatable=false) private Instant createdAt=Instant.now();
    protected KnowledgeVersion() {}
    public KnowledgeVersion(String documentId,String source,int ordinal,String contentHash,int chunkCount) {
        this.documentId=documentId; this.source=source; this.ordinal=ordinal; this.contentHash=contentHash; this.chunkCount=chunkCount;
    }
    public String getTenantId(){return tenantId;} public String getKnowledgeBaseId(){return knowledgeBaseId;}
    public void setScope(String tenantId,String baseId){this.tenantId=tenantId;this.knowledgeBaseId=baseId;}
    public String getId(){return id;} public String getDocumentId(){return documentId;} public String getSource(){return source;}
    public int getOrdinal(){return ordinal;} public String getStatus(){return status;} public String getContentHash(){return contentHash;}
    public int getChunkCount(){return chunkCount;} public Instant getCreatedAt(){return createdAt;}
    public int getAttempts(){return attempts;} public String getLastError(){return lastError;}
    public void startAttempt(){attempts++;lastError=null;}
    public void setLastError(String error){lastError=error;}
    public void setStatus(String status){this.status=status;}
}
