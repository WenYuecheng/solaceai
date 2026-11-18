package com.mindbridge.agent.dto;

import com.mindbridge.agent.domain.IntentType;
import com.mindbridge.agent.domain.RiskLevel;
import java.time.Instant;
import java.util.List;
import java.util.Map;

public record AdminAgentTrace(Long id, String sessionId, String runId, Long reportId,
                              IntentType intent, RiskLevel riskLevel, String sanitizedInput,
                              List<Map<String, Object>> agentSteps, List<Map<String, Object>> retrievedKnowledge,
                              Instant createdAt, Double firstTokenSeconds, Double elapsedSeconds) {
}
