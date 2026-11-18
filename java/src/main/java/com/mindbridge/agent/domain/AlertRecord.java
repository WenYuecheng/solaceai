package com.mindbridge.agent.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.time.Instant;

@Entity
@Table(name = "alert_records")
/**
 * 高风险预警发送记录。
 *
 * <p>一条心理报告可能对应多个收件人，每个收件人都有独立的发送状态、重试次数和错误信息。</p>
 */
public class AlertRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "report_id")
    private PsychologicalReport report;

    @Column(nullable = false, length = 240)
    private String recipient;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private ToolStatus status = ToolStatus.PENDING;

    @Column(length = 500)
    private String errorMessage;

    @Column(nullable = false)
    private int attempts;

    @Column(nullable = false, updatable = false)
    private Instant createdAt = Instant.now();

    @Column(nullable = false)
    private Instant updatedAt = Instant.now();

    private Instant leaseExpiresAt;
    @Column(length = 36) private String leaseOwner;
    @Column(length = 240) private String lastActor;
    private boolean recoveredInterrupted;

    public Instant getLeaseExpiresAt() { return leaseExpiresAt; }
    public String getLeaseOwner() { return leaseOwner; }
    public String getLastActor() { return lastActor; }
    public boolean isRecoveredInterrupted() { return recoveredInterrupted; }
    public void claim(String owner, String actor, boolean recovered) {
        leaseOwner = owner;
        lastActor = actor;
        recoveredInterrupted = recovered;
        leaseExpiresAt = Instant.now().plusSeconds(600);
        incrementAttempts();
        setStatus(ToolStatus.RUNNING);
        errorMessage = null;
    }

    public Long getId() {
        return id;
    }

    public PsychologicalReport getReport() {
        return report;
    }

    public void setReport(PsychologicalReport report) {
        this.report = report;
    }

    public String getRecipient() {
        return recipient;
    }

    public void setRecipient(String recipient) {
        this.recipient = recipient;
    }

    public ToolStatus getStatus() {
        return status;
    }

    public void setStatus(ToolStatus status) {
        this.status = status;
        this.updatedAt = Instant.now();
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public int getAttempts() {
        return attempts;
    }

    public void incrementAttempts() {
        this.attempts++;
        this.updatedAt = Instant.now();
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }
}
