package com.mindbridge.agent.service.knowledge;
import com.mindbridge.agent.domain.KnowledgeChunk;
import java.util.List;
public interface KnowledgeIndex {
    void upsert(List<KnowledgeChunk> chunks);
    void delete(List<Long> ids);
    default int rebuild(List<KnowledgeChunk> chunks) { throw new UnsupportedOperationException("Vector rebuild is unavailable"); }
}
