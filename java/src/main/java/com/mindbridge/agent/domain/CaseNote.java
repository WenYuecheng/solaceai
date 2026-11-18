package com.mindbridge.agent.domain;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name="case_notes")
public class CaseNote {
    @Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;
    @Column(nullable=false) private long caseId;
    @Column(nullable=false,length=120) private String actor;
    @Column(nullable=false,length=2048) private String note;
    @Column(nullable=false) private Instant createdAt=Instant.now();
    public Long getId(){return id;} public void setId(Long value){id=value;}
    public long getCaseId(){return caseId;} public void setCaseId(long value){caseId=value;}
    public String getActor(){return actor;} public void setActor(String value){actor=value;}
    public String getNote(){return note;} public void setNote(String value){note=value;}
    public Instant getCreatedAt(){return createdAt;} public void setCreatedAt(Instant value){createdAt=value;}
}
