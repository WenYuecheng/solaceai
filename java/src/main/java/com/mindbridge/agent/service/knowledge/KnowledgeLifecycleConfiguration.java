package com.mindbridge.agent.service.knowledge;

import com.mindbridge.agent.config.ParityKnowledgeProperties;
import com.mindbridge.agent.domain.KnowledgeChunk;
import com.mindbridge.agent.repository.KnowledgeChunkRepository;
import com.mindbridge.agent.repository.KnowledgeVersionRepository;
import java.util.List;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.transaction.PlatformTransactionManager;

@Configuration
public class KnowledgeLifecycleConfiguration {
    @Bean
    public KnowledgeLifecycleService knowledgeLifecycleService(KnowledgeChunkRepository chunks, KnowledgeVersionRepository versions,
                                                                 PlatformTransactionManager tx, ParityKnowledgeProperties properties, com.fasterxml.jackson.databind.ObjectMapper mapper) {
        KnowledgeIndex index="milvus_bge".equalsIgnoreCase(properties.getBackend()) ? new MilvusGateway(properties,mapper) : new KnowledgeIndex(){
            public void upsert(List<KnowledgeChunk> rows) { requireBm25(); }
            public void delete(List<Long> ids) { requireBm25(); }
            private void requireBm25() { if(!"bm25".equalsIgnoreCase(properties.getBackend())) throw new IllegalStateException("Configured knowledge vector backend is unavailable"); }
        };
        var lifecycle=new KnowledgeLifecycleService(chunks,versions,tx,index,properties.getTenantId(),properties.getKnowledgeBaseId());
        lifecycle.setWriterLockPath(properties.getWriterLockPath());
        return lifecycle;
    }
}
