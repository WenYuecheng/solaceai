package com.mindbridge.agent.repository;

import com.mindbridge.agent.domain.ChatMessage;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

/**
 * 聊天消息的数据访问接口。
 */
public interface ChatMessageRepository extends JpaRepository<ChatMessage, Long> {
    String HISTORY_FILTER = " from ChatMessage m where m.user.id=:user and m.session.publicId=:session and m.id<:before and m.role<>com.mindbridge.agent.domain.MessageRole.SYSTEM";

    @org.springframework.data.jpa.repository.Query("select count(m),coalesce(max(m.id),0)" + HISTORY_FILTER)
    List<Object[]> historyStamp(long user, String session, long before);

    @org.springframework.data.jpa.repository.Query("select m" + HISTORY_FILTER + " order by m.id asc")
    List<ChatMessage> historyBefore(long user, String session, long before);

    /** 取最近消息用于模型上下文，Service 层会再反转为时间正序。 */
    List<ChatMessage> findTop20BySession_IdOrderByCreatedAtDesc(Long sessionId);

    /** 管理员点开记录时读取完整会话。 */
    List<ChatMessage> findBySession_PublicIdOrderByCreatedAtAsc(String publicId);
}
