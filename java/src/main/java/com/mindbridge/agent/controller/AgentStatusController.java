package com.mindbridge.agent.controller;

import com.mindbridge.agent.config.MindBridgeProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/agent")
/**
 * 智能体运行状态接口。
 *
 * <p>前端用它展示当前 provider、项目模型名称、RAG 参数和模型连接模式。</p>
 */
public class AgentStatusController {

    private final MindBridgeProperties properties;
    private com.mindbridge.agent.runtime.RuntimeProperties runtime;
    private com.mindbridge.agent.config.ParityKnowledgeProperties retrieval;
    @org.springframework.beans.factory.annotation.Autowired public void runtime(com.mindbridge.agent.runtime.RuntimeProperties runtime,com.mindbridge.agent.config.ParityKnowledgeProperties retrieval){this.runtime=runtime;this.retrieval=retrieval;}

    public AgentStatusController(MindBridgeProperties properties) {
        this.properties = properties;
    }

    @GetMapping("/status")
    public AgentStatusResponse status() {
        // realModelEnabled 只表示当前使用真实模型客户端，不代表业务评估一定会展示给学生。
        String provider = properties.getAi().getProvider().toLowerCase(Locale.ROOT);
        boolean realModelEnabled = "ollama".equals(provider) || "openai".equals(provider);
        return new AgentStatusResponse(
                provider,
                modelName(provider),
                realModelEnabled,
                false,
                properties.getKnowledge().getTopK(),
                retrieval!=null&&retrieval.isRerankerEnabled(),
                properties.getKnowledge().getRerankerCandidateLimit(),
                false,
                properties.getMemory().getTopK(),
                "LangGraph4j + Spring AI；模型连通性尚需独立探测。",
                "langgraph4j",runtime==null?"cloud":runtime.getDefaultBackend(),runtime!=null&&runtime.isLocalModelEnabled()?java.util.List.of("cloud","local"):java.util.List.of("cloud"),retrieval==null?"bm25":retrieval.getBackend(),runtime!=null&&runtime.isLocalFallbackEnabled()
        );
    }

    private String modelName(String provider) {
        if ("ollama".equals(provider)) {
            return properties.getAi().getOllama().getModel();
        }
        if ("openai".equals(provider)) {
            return properties.getAi().getOpenai().getModel();
        }
        return "unknown";
    }

    /**
     * 前端状态栏需要的最小状态信息。
     */
    public record AgentStatusResponse(
            String provider,
            String model,
            boolean realModelEnabled,
            boolean chromaEnabled,
            int ragTopK,
            boolean rerankerEnabled,
            int rerankerCandidateLimit,
            boolean memoryChromaEnabled,
            int memoryTopK,
            String note,
            String framework,String defaultBackend,java.util.List<String> availableModelBackends,String knowledgeBackend,boolean localFallbackEnabled
    ) {
        @JsonProperty("defaultChatBackend")
        public String defaultChatBackend() {
            return defaultBackend;
        }

        @JsonProperty("availableChatBackends")
        public List<String> availableChatBackends() {
            return availableModelBackends;
        }

        @JsonProperty("chatModels")
        public Map<String, ChatModel> chatModels() {
            var cloud = new ChatModel(provider, model);
            return availableModelBackends.contains("local")
                    ? Map.of("cloud", cloud, "local", new ChatModel("local-gateway", "solace-response"))
                    : Map.of("cloud", cloud);
        }
    }

    public record ChatModel(String provider, String model) {}
}
