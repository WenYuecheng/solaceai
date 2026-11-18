package com.mindbridge.agent.dto;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
public record KnowledgeIngestRequest(@NotBlank @Size(max=180) String source, @NotBlank String content,
                                     @JsonProperty("document_id") String documentId) {
    public KnowledgeIngestRequest(String source,String content) { this(source,content,null); }
}
