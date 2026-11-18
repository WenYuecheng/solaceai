package com.mindbridge.agent.domain;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name="answer_feedback",uniqueConstraints=@UniqueConstraint(columnNames={"userId","messageId"}))
public class AnswerFeedback {
    @Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;
    @Column(nullable=false) private long userId;
    @Column(nullable=false) private long messageId;
    @Column(nullable=false,length=64) private String runId;
    @Column(nullable=false) private boolean helpful;
    @Column(nullable=false,length=32) private String category="general";
    @Column(nullable=false,length=1000) private String note="";
    @Column(nullable=false) private Instant updatedAt=Instant.now();
    public Long getId(){return id;}
    public long getUserId(){return userId;} public void setUserId(long value){userId=value;}
    public long getMessageId(){return messageId;} public void setMessageId(long value){messageId=value;}
    public String getRunId(){return runId;} public void setRunId(String value){runId=value;}
    public boolean isHelpful(){return helpful;} public void setHelpful(boolean value){helpful=value;}
    public String getCategory(){return category;} public void setCategory(String value){category=value;}
    public String getNote(){return note;} public void setNote(String value){note=value;}
    public Instant getUpdatedAt(){return updatedAt;} public void touch(){updatedAt=Instant.now();}
}
