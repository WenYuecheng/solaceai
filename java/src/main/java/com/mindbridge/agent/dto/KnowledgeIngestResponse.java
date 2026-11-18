package com.mindbridge.agent.dto;
import com.fasterxml.jackson.annotation.JsonProperty;
public record KnowledgeIngestResponse(String source,int chunks,@JsonProperty("document_id") String documentId) {
    public KnowledgeIngestResponse(String source,int chunks) { this(source,chunks,null); }
}
