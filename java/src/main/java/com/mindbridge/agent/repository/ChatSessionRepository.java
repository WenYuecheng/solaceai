package com.mindbridge.agent.repository;

import com.mindbridge.agent.domain.ChatSession;
import java.util.Optional;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

/**
 * 会话的数据访问接口。
 */
public interface ChatSessionRepository extends JpaRepository<ChatSession, Long> {
    @org.springframework.data.jpa.repository.Lock(jakarta.persistence.LockModeType.PESSIMISTIC_WRITE)
    @org.springframework.data.jpa.repository.Query("select s from ChatSession s where s.id=:id")
    Optional<ChatSession> locked(@org.springframework.data.repository.query.Param("id") long id);

    @org.springframework.data.jpa.repository.Query(value="select * from chat_sessions where user_id=:user and hidden=false order by updated_at desc,id desc limit :limit offset :offset",nativeQuery=true)
    java.util.List<ChatSession> visiblePage(@org.springframework.data.repository.query.Param("user") long user,@org.springframework.data.repository.query.Param("limit") int limit,@org.springframework.data.repository.query.Param("offset") int offset);
    long countByUser_IdAndHiddenFalse(long user);

    /** 学生继续对话时，必须校验会话属于当前用户。 */
    Optional<ChatSession> findByPublicIdAndUser_Id(String publicId, Long userId);

    /** 管理员查看会话详情时需要连同用户信息一起加载。 */
    @EntityGraph(attributePaths = "user")
    Optional<ChatSession> findByPublicId(String publicId);
}
