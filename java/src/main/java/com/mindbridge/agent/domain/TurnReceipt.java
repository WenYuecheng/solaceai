package com.mindbridge.agent.domain;

import jakarta.persistence.*;
import java.time.Instant;

/** Durable run receipt: database commits bind messages, reports and final reply exactly once. */
@Entity
@Table(name="agent_turn_receipts")
public class TurnReceipt {
    @Id @Column(length=32)
    private String runId;
    @Column(nullable=false)
    private long userId;
    @Column(nullable=false,length=64)
    private String sessionId;
    @Column(nullable=false,length=64)
    private String inputHash;
    @Column(nullable=false,length=16)
    private String modelBackend;
    @Column(nullable=false,length=24)
    private String status;

    private Long userMessageId;

    private Long assistantMessageId;

    private Long reportId;
    @Column(length=64)
    private String traceId;
    @Lob
    private String outcomeJson;
    @Lob
    private String stepsJson;

    private Double firstTokenSeconds;

    private Double elapsedSeconds;
    @Column(nullable=false)
    private Instant createdAt = Instant.now();
    @Column(nullable=false)
    private Instant updatedAt = Instant.now();
    @Version private long version;
    public String getRunId() { return runId; }
    public void setRunId(String value) { runId=value; }
    public long getUserId() { return userId; }
    public void setUserId(long value) { userId=value; }
    public String getSessionId() { return sessionId; }
    public void setSessionId(String value) { sessionId=value; }
    public String getInputHash() { return inputHash; }
    public void setInputHash(String value) { inputHash=value; }
    public String getModelBackend() { return modelBackend; }
    public void setModelBackend(String value) { modelBackend=value; }
    public String getStatus() { return status; }
    public void setStatus(String value) { status=value; }
    public Long getUserMessageId() { return userMessageId; }
    public void setUserMessageId(Long value) { userMessageId=value; }
    public Long getAssistantMessageId() { return assistantMessageId; }
    public void setAssistantMessageId(Long value) { assistantMessageId=value; }
    public Long getReportId() { return reportId; }
    public void setReportId(Long value) { reportId=value; }
    public String getTraceId() { return traceId; }
    public void setTraceId(String value) { traceId=value; }
    public String getOutcomeJson() { return outcomeJson; }
    public void setOutcomeJson(String value) { outcomeJson=value; }
    public String getStepsJson() { return stepsJson; }
    public void setStepsJson(String value) { stepsJson=value; }
    public Double getFirstTokenSeconds() { return firstTokenSeconds; }
    public void setFirstTokenSeconds(Double value) { firstTokenSeconds=value; }
    public Double getElapsedSeconds() { return elapsedSeconds; }
    public void setElapsedSeconds(Double value) { elapsedSeconds=value; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant value) { createdAt=value; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant value) { updatedAt=value; }
}
