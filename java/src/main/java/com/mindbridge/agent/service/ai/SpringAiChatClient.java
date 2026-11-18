package com.mindbridge.agent.service.ai;

import java.util.List;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.StreamingChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

/**
 * 基于 Spring AI 的聊天模型适配器。
 *
 * <p>项目内部继续使用简洁的 {@link AiClient} 接口；真正的大模型调用由
 * Spring AI 的 ChatModel / StreamingChatModel 完成。</p>
 */
public class SpringAiChatClient implements AiClient {

    private final ChatModel chatModel;
    private final StreamingChatModel streamingChatModel;

    public SpringAiChatClient(ChatModel chatModel, StreamingChatModel streamingChatModel) {
        this.chatModel = chatModel;
        this.streamingChatModel = streamingChatModel;
    }

    @Override
    public String complete(List<AiMessage> messages) {
        ChatResponse response = chatModel.call(new Prompt(toSpringMessages(messages)));
        return extractText(response);
    }

    @Override
    public Flux<String> stream(List<AiMessage> messages) {
        return streamingChatModel.stream(new Prompt(toSpringMessages(messages)))
                .map(this::extractText)
                .filter(token -> !token.isEmpty());
    }

    @Override
    public com.mindbridge.agent.runtime.ModelDecision decide(List<AiMessage> messages,
            java.util.Map<String, java.util.Map<String, Object>> tools,
            java.util.function.Consumer<String> tokens, java.util.function.BooleanSupplier cancelled) {
        var callbacks = tools.entrySet().stream().map(entry -> new org.springframework.ai.tool.ToolCallback() {
            public org.springframework.ai.tool.definition.ToolDefinition getToolDefinition() {
                return org.springframework.ai.tool.definition.ToolDefinition.builder().name(entry.getKey())
                        .description(String.valueOf(entry.getValue().getOrDefault("description", entry.getKey())))
                        .inputSchema(com.mindbridge.agent.runtime.DshCompaction.json(entry.getValue().get("parameters"))).build();
            }
            public String call(String input) { throw new SecurityException("Tool execution belongs to the governed graph node"); }
        }).map(org.springframework.ai.tool.ToolCallback.class::cast).toList();
        var options = org.springframework.ai.model.tool.DefaultToolCallingChatOptions.builder()
                .toolCallbacks(callbacks).internalToolExecutionEnabled(false).build();
        StringBuilder text = new StringBuilder();
        java.util.Map<String, org.springframework.ai.chat.messages.AssistantMessage.ToolCall> calls = new java.util.LinkedHashMap<>();
        final boolean[] plain = {false};
        streamingChatModel.stream(new Prompt(toSpringMessages(messages), options)).doOnNext(response -> {
            if (cancelled.getAsBoolean()) throw new java.util.concurrent.CancellationException();
            if (response.getResult() == null) return;
            var output = response.getResult().getOutput();
            for (var call : output.getToolCalls()) {
                var prior = calls.get(call.id());
                calls.put(call.id(), prior == null ? call : new org.springframework.ai.chat.messages.AssistantMessage.ToolCall(
                        call.id(), call.type(), call.name() == null || call.name().isEmpty() ? prior.name() : call.name(),
                        (prior.arguments() == null ? "" : prior.arguments()) + (call.arguments() == null ? "" : call.arguments())));
            }
            String token = output.getText();
            if (token == null || token.isEmpty()) return;
            text.append(token);
            if (!plain[0] && !text.toString().isBlank()) {
                String prefix = text.toString().stripLeading();
                plain[0] = !prefix.startsWith("{") && !prefix.startsWith("[") && !prefix.startsWith("`");
                if (plain[0] && calls.isEmpty()) tokens.accept(text.toString());
            } else if (plain[0] && calls.isEmpty()) tokens.accept(token);
        }).blockLast();
        if (!calls.isEmpty()) {
            if (plain[0]) throw new IllegalArgumentException("Mixed answer/tool response");
            var json = new com.fasterxml.jackson.databind.ObjectMapper();
            java.util.List<com.mindbridge.agent.runtime.ModelDecision.Action> actions = new java.util.ArrayList<>();
            for (var call : calls.values()) {
                try {
                    java.util.Map<String,Object> args = json.readValue(call.arguments(), new com.fasterxml.jackson.core.type.TypeReference<>() {});
                    actions.add(new com.mindbridge.agent.runtime.ModelDecision.Action(call.name(), args));
                } catch (Exception e) { throw new IllegalArgumentException("Malformed agent decision envelope", e); }
            }
            if (actions.size() > 6) throw new IllegalArgumentException("Tool batch exceeds decision budget");
            return new com.mindbridge.agent.runtime.ModelDecision(null, actions);
        }
        return com.mindbridge.agent.runtime.ModelDecision.parse(text.toString(), new com.fasterxml.jackson.databind.ObjectMapper());
    }

    private List<Message> toSpringMessages(List<AiMessage> messages) {
        return messages.stream()
                .map(this::toSpringMessage)
                .toList();
    }

    private Message toSpringMessage(AiMessage message) {
        return switch (message.role()) {
            case "system" -> new SystemMessage(message.content());
            case "assistant" -> new AssistantMessage(message.content());
            default -> new UserMessage(message.content());
        };
    }

    private String extractText(ChatResponse response) {
        if (response == null || response.getResult() == null || response.getResult().getOutput() == null) {
            return "";
        }
        String text = response.getResult().getOutput().getText();
        return text == null ? "" : text;
    }
}
