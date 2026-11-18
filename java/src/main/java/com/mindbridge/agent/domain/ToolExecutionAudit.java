package com.mindbridge.agent.domain;

import jakarta.persistence.*;
import java.time.Instant;

/** Append-only operational audit without message content, credentials or recipient input. */
@Entity
@Table(name="tool_execution_audits")
public class ToolExecutionAudit {
    @Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;
    private Long reportId;
    @Column(length=240) private String actor;
    @Column(length=60) private String action;
    @Column(length=40) private String outcome;
    private Instant createdAt=Instant.now();
    public ToolExecutionAudit(){}
    public ToolExecutionAudit(Long reportId,String actor,String action,String outcome){this.reportId=reportId;this.actor=actor;this.action=action;this.outcome=outcome;}
    public Long getId(){return id;}
    public Long getReportId(){return reportId;}
    public String getActor(){return actor;}
    public String getAction(){return action;}
    public String getOutcome(){return outcome;}
    public Instant getCreatedAt(){return createdAt;}
}
