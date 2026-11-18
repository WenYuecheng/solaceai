package com.mindbridge.agent.repository;
import com.mindbridge.agent.domain.KnowledgeVersion;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
public interface KnowledgeVersionRepository extends JpaRepository<KnowledgeVersion,String> {
    List<KnowledgeVersion> findByDocumentIdOrderByOrdinalAsc(String documentId);
    List<KnowledgeVersion> findByStatus(String status);
    List<KnowledgeVersion> findByTenantIdAndKnowledgeBaseIdAndDocumentIdOrderByOrdinalAsc(String tenantId,String knowledgeBaseId,String documentId);
    List<KnowledgeVersion> findByTenantIdAndKnowledgeBaseIdAndStatus(String tenantId,String knowledgeBaseId,String status);
    List<KnowledgeVersion> findByTenantIdAndKnowledgeBaseId(String tenantId,String knowledgeBaseId);
}
