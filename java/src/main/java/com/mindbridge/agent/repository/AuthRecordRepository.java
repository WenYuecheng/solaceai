package com.mindbridge.agent.repository;
import com.mindbridge.agent.domain.AuthRecord;
import java.util.Optional;
import org.springframework.data.jpa.repository.*;
import org.springframework.data.repository.query.Param;
import jakarta.persistence.LockModeType;
public interface AuthRecordRepository extends JpaRepository<AuthRecord,String> {
 @Lock(LockModeType.PESSIMISTIC_WRITE) @Query("select a from AuthRecord a where a.id=:id") Optional<AuthRecord> locked(@Param("id") String id);
 void deleteByExpiresAtBefore(java.time.Instant now);
 void deleteByUserId(Long userId);
}
