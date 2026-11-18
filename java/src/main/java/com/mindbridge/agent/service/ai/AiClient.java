package com.mindbridge.agent.service.ai;

import java.util.List;
import reactor.core.publisher.Flux;

/**
 * 模型调用统一接口。
 *
 * <p>业务层通过 complete 做分类/评估，通过 stream 做学生端流式回答。</p>
 */
public interface AiClient {

    String complete(List<AiMessage> messages);

    Flux<String> stream(List<AiMessage> messages);

    /** Execute no tools here: only the governed graph tool node may do that. */
    default com.mindbridge.agent.runtime.ModelDecision decide(
            List<AiMessage> messages, java.util.Map<String, java.util.Map<String, Object>> tools,
            java.util.function.Consumer<String> tokens, java.util.function.BooleanSupplier cancelled) {
        if (cancelled.getAsBoolean()) throw new java.util.concurrent.CancellationException();
        return com.mindbridge.agent.runtime.ModelDecision.parse(complete(messages), new com.fasterxml.jackson.databind.ObjectMapper());
    }
}
