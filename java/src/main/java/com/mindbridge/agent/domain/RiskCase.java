package com.mindbridge.agent.domain;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name="risk_cases")
public class RiskCase {
    @Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;
    @Column(nullable=false,unique=true) private long reportId;
    @Column(nullable=false) private long userId;
    @Column(nullable=false,length=64) private String sessionId;
    @Enumerated(EnumType.STRING) @Column(nullable=false,length=20) private RiskLevel riskLevel;
    @Column(nullable=false,length=24) private String status;
    @Column(length=120) private String owner;
    @Column(length=2000) private String summary;
    @Column(length=4000) private String handoffSummary;
    @Column(length=120) private String acknowledgedBy;
     private Instant acknowledgedAt;
    @Column(nullable=false) private Instant createdAt=Instant.now();
    @Column(nullable=false) private Instant updatedAt=Instant.now();
    @Version private long version;
    public Long getId(){return id;} public void setId(Long value){id=value;}
    public long getReportId(){return reportId;} public void setReportId(long value){reportId=value;}
    public long getUserId(){return userId;} public void setUserId(long value){userId=value;}
    public String getSessionId(){return sessionId;} public void setSessionId(String value){sessionId=value;}
    public RiskLevel getRiskLevel(){return riskLevel;} public void setRiskLevel(RiskLevel value){riskLevel=value;}
    public String getStatus(){return status;} public void setStatus(String value){status=value;}
    public String getOwner(){return owner;} public void setOwner(String value){owner=value;}
    public String getSummary(){return summary;} public void setSummary(String value){summary=value;}
    public String getHandoffSummary(){return handoffSummary;} public void setHandoffSummary(String value){handoffSummary=value;}
    public String getAcknowledgedBy(){return acknowledgedBy;} public void setAcknowledgedBy(String value){acknowledgedBy=value;}
    public Instant getAcknowledgedAt(){return acknowledgedAt;} public void setAcknowledgedAt(Instant value){acknowledgedAt=value;}
    public Instant getCreatedAt(){return createdAt;} public void setCreatedAt(Instant value){createdAt=value;}
    public Instant getUpdatedAt(){return updatedAt;} public void setUpdatedAt(Instant value){updatedAt=value;}
}
