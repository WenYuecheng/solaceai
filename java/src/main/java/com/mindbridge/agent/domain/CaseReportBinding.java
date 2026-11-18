package com.mindbridge.agent.domain;

import jakarta.persistence.*;
import java.time.Instant;

/** Immutable idempotency association; the case's current report pointer may advance. */
@Entity
@Table(name="case_report_bindings")
public class CaseReportBinding {
    @Id @Column(updatable=false) private Long reportId;
    @Column(nullable=false,updatable=false) private long caseId;
    @Column(nullable=false,updatable=false) private Instant createdAt=Instant.now();
    protected CaseReportBinding() {}
    public CaseReportBinding(long reportId,long caseId){this.reportId=reportId;this.caseId=caseId;}
    public Long getReportId(){return reportId;}
    public long getCaseId(){return caseId;}
    public Instant getCreatedAt(){return createdAt;}
}
