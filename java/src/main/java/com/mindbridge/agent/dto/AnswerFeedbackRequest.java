package com.mindbridge.agent.dto;
import jakarta.validation.constraints.*;
public record AnswerFeedbackRequest(@Positive long messageId,@NotBlank @Size(max=64) String runId,
    @NotNull Boolean helpful,@Pattern(regexp="general|irrelevant|evidence|unsafe|other") String category,@Size(max=1000) String note) {
    public AnswerFeedbackRequest {if(category==null)category="general";if(note==null)note="";}
}
