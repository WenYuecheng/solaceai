package com.mindbridge.agent.dto;

import com.mindbridge.agent.domain.ToolStatus;
import java.time.Instant;
import java.util.Map;

public record AdminToolExecution(Long id, String runId, String toolName, ToolStatus status, String error,
                                 Map<String, Object> result, Instant createdAt, Instant updatedAt,
                                 Long reportId, String errorMessage, int attempts, String actor,
                                 Instant leaseExpiresAt, String leaseOwner, boolean recoveredInterrupted) {
}
