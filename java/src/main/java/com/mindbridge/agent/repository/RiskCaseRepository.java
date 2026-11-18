package com.mindbridge.agent.repository;
import com.mindbridge.agent.domain.RiskCase;
import java.util.*;
import org.springframework.data.jpa.repository.*;
import jakarta.persistence.LockModeType;
public interface RiskCaseRepository extends JpaRepository<RiskCase,Long> {
    Optional<RiskCase> findByReportId(long reportId);
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    Optional<RiskCase> findFirstByUserIdAndSessionIdAndStatusNotOrderByIdDesc(long userId,String sessionId,String status);
    @Query("select c.sessionId from RiskCase c where c.id=:id")
    Optional<String> sessionKey(@org.springframework.data.repository.query.Param("id") long id);
    @Query("select c.userId from RiskCase c where c.id=:id")
    Optional<Long> userKey(@org.springframework.data.repository.query.Param("id") long id);
    List<RiskCase> findTop100ByOrderByUpdatedAtDesc();
    @Lock(LockModeType.PESSIMISTIC_WRITE) @Query("select c from RiskCase c where c.id=:id") Optional<RiskCase> locked(@org.springframework.data.repository.query.Param("id") long id);
}
