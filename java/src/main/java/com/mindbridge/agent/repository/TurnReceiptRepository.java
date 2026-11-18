package com.mindbridge.agent.repository;
import com.mindbridge.agent.domain.TurnReceipt;
import java.util.*;
import org.springframework.data.jpa.repository.JpaRepository;
public interface TurnReceiptRepository extends JpaRepository<TurnReceipt,String> {
    @org.springframework.data.jpa.repository.Lock(jakarta.persistence.LockModeType.PESSIMISTIC_WRITE)
    @org.springframework.data.jpa.repository.Query("select r from TurnReceipt r where r.runId=:run")
    Optional<TurnReceipt> locked(@org.springframework.data.repository.query.Param("run") String run);
    Optional<TurnReceipt> findByAssistantMessageId(Long messageId);
    List<TurnReceipt> findByUserIdAndSessionIdOrderByCreatedAtAsc(long userId,String sessionId);
    List<TurnReceipt> findTop100ByOrderByCreatedAtDesc();
    List<TurnReceipt> findByTraceIdIn(Collection<String> traceIds);
}
