package com.mindbridge.agent.domain;

import jakarta.persistence.*;
import java.time.Instant;

/** Durable report-tool receipt; report id is the deterministic execution key. */
@Entity
@Table(name="business_tool_executions")
public class BusinessToolExecution {
    @Id private Long reportId;
    @Enumerated(EnumType.STRING) @Column(nullable=false,length=20) private ToolStatus status=ToolStatus.PENDING;
    private Instant leaseExpiresAt;
    @Column(length=36) private String leaseOwner;
    @Column(length=240) private String actor;
    private boolean recoveredInterrupted;
    private int attempts;
    @Column(length=500) private String errorMessage;
    @Column(updatable=false) private Instant createdAt=Instant.now();
    private Instant updatedAt=Instant.now();
    public BusinessToolExecution(){}
    public BusinessToolExecution(Long reportId){this.reportId=reportId;}
    public Long getReportId(){return reportId;}
    public ToolStatus getStatus(){return status;}
    public void setStatus(ToolStatus status){this.status=status;updatedAt=Instant.now();}
    public Instant getLeaseExpiresAt(){return leaseExpiresAt;}
    public String getLeaseOwner(){return leaseOwner;}
    public String getActor(){return actor;}
    public boolean isRecoveredInterrupted(){return recoveredInterrupted;}
    public int getAttempts(){return attempts;}
    public String getErrorMessage(){return errorMessage;}
    public void setErrorMessage(String error){errorMessage=error;}
    public Instant getUpdatedAt(){return updatedAt;}
    public Instant getCreatedAt(){return createdAt;}
    public void claim(String owner,String actor,boolean recovered){leaseOwner=owner;this.actor=actor;recoveredInterrupted=recovered;leaseExpiresAt=Instant.now().plusSeconds(600);attempts++;setStatus(ToolStatus.RUNNING);}
}
