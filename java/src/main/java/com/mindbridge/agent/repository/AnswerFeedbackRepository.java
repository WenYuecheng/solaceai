package com.mindbridge.agent.repository;
import com.mindbridge.agent.domain.AnswerFeedback;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
public interface AnswerFeedbackRepository extends JpaRepository<AnswerFeedback,Long> {
    Optional<AnswerFeedback> findByUserIdAndMessageId(long userId,long messageId);
}
