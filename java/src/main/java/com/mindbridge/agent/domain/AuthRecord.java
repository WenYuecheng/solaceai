package com.mindbridge.agent.domain;
import jakarta.persistence.*;
import java.time.Instant;
/** Only hashes of opaque sessions and email challenges are stored. */
@Entity @Table(name="auth_records")
public class AuthRecord {
 @Id @Column(length=64) public String id;
 @Column(name="credential_hash",length=255) public String value;
 @Version public Long version;
 public Long userId;
 public Instant expiresAt;
 public Instant sentAt;
 public int attempts;
 public boolean consumed;
 public AuthRecord(){}
 public AuthRecord(String id){this.id=id;}
}
