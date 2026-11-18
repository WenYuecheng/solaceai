package com.mindbridge.agent.service;

import com.mindbridge.agent.domain.AgentRunTrace;
import com.mindbridge.agent.domain.AgentRunTraceStep;
import com.mindbridge.agent.domain.ChatMessage;
import com.mindbridge.agent.domain.ChatSession;
import com.mindbridge.agent.domain.RiskLevel;
import com.mindbridge.agent.domain.TurnReceipt;
import com.mindbridge.agent.domain.UserAccount;
import com.mindbridge.agent.dto.AdminAgentTrace;
import com.mindbridge.agent.dto.AgentRunTraceResponse;
import com.mindbridge.agent.dto.AgentRunTraceSummaryResponse;
import com.mindbridge.agent.repository.AgentRunTraceRepository;
import com.mindbridge.agent.repository.TurnReceiptRepository;
import com.mindbridge.agent.runtime.RunOutcome;
import com.mindbridge.agent.service.agent.AgentRunResult;
import com.mindbridge.agent.service.agent.AgentStep;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;
import java.util.stream.Collectors;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

@Service
/**
 * Agent run trace 的落库与管理员查询服务。
 */
public class AgentRunTraceService {

    private static final String ROLE_ADMIN = "ROLE_ADMIN";

    private final AgentRunTraceRepository agentRunTraceRepository;
    private final TurnReceiptRepository receipts;
    private final ObjectMapper json;
    private final PrivacySanitizer privacy;

    public AgentRunTraceService(AgentRunTraceRepository agentRunTraceRepository, TurnReceiptRepository receipts,
                                ObjectMapper json, PrivacySanitizer privacy) {
        this.agentRunTraceRepository = agentRunTraceRepository;
        this.receipts = receipts;
        this.json = json;
        this.privacy = privacy;
    }

    @Transactional
    public AgentRunTrace saveRun(
            UserAccount user,
            ChatSession session,
            ChatMessage triggerMessage,
            String input,
            AgentRunResult agentRun,
            Instant startedAt,
            Instant completedAt
    ) {
        AgentRunTrace trace = new AgentRunTrace();
        trace.setTraceId(UUID.randomUUID().toString().replace("-", ""));
        trace.setUser(user);
        trace.setSession(session);
        trace.setTriggerMessage(triggerMessage);
        trace.setInput(input);
        trace.setIntent(agentRun.intent());
        trace.setRiskLevel(agentRun.riskLevel() == null ? RiskLevel.LOW : agentRun.riskLevel());
        trace.setMemoryBrief(agentRun.memoryBrief());
        trace.setKnowledgeQuery(agentRun.knowledgeQuery());
        trace.setResponsePlan(agentRun.responsePlan());
        trace.setResponseAgent(agentRun.responseAgent());
        trace.setStepCount(agentRun.steps().size());
        trace.setStartedAt(startedAt);
        trace.setCompletedAt(completedAt);
        agentRun.steps().forEach(step -> trace.addStep(toEntity(step)));
        return agentRunTraceRepository.save(trace);
    }

    @Transactional(readOnly = true)
    public List<AgentRunTraceSummaryResponse> latestTraces() {
        return agentRunTraceRepository.findTop100ByOrderByStartedAtDesc().stream()
                .filter(trace -> isStudentUser(trace.getUser()))
                .map(AgentRunTraceSummaryResponse::from)
                .toList();
    }

    @Transactional(readOnly = true)
    public List<AdminAgentTrace> latestAuditTraces() {
        var traces = agentRunTraceRepository.findTop100ByOrderByStartedAtDesc().stream()
                .filter(trace -> isStudentUser(trace.getUser()))
                .toList();
        if (traces.isEmpty()) return List.of();
        var receiptsByTrace = receipts.findByTraceIdIn(traces.stream().map(AgentRunTrace::getTraceId).toList())
                .stream().collect(Collectors.toMap(TurnReceipt::getTraceId, Function.identity()));
        return traces.stream().map(trace -> auditTrace(trace, receiptsByTrace.get(trace.getTraceId()))).toList();
    }

    private AdminAgentTrace auditTrace(AgentRunTrace trace, TurnReceipt receipt) {
        RunOutcome outcome = null;
        if (receipt != null && receipt.getOutcomeJson() != null) {
            try {
                outcome = json.readValue(receipt.getOutcomeJson(), RunOutcome.class);
            } catch (com.fasterxml.jackson.core.JsonProcessingException error) {
                org.slf4j.LoggerFactory.getLogger(AgentRunTraceService.class)
                        .warn("Cannot read stored trace outcome for trace {}", trace.getTraceId());
            }
        }
        var steps = outcome != null && outcome.agentSteps() != null ? outcome.agentSteps()
                : trace.getSteps().stream().map(step -> Map.<String, Object>of("step", step.getStepNumber(),
                        "agent", step.getAgent().name(), "action", step.getAction().name(),
                        "observation", step.getObservation())).toList();
        return new AdminAgentTrace(trace.getId(), trace.getSession().getPublicId(),
                receipt == null ? null : receipt.getRunId(), receipt == null ? null : receipt.getReportId(),
                trace.getIntent(), trace.getRiskLevel(), privacy.sanitize(trace.getInput()), steps,
                outcome == null || outcome.modelVisibleEvidence() == null ? List.of() : outcome.modelVisibleEvidence(),
                trace.getStartedAt(), receipt == null ? null : receipt.getFirstTokenSeconds(),
                receipt == null ? null : receipt.getElapsedSeconds());
    }

    @Transactional(readOnly = true)
    public List<AgentRunTraceResponse> tracesForSession(String sessionId) {
        return agentRunTraceRepository.findBySession_PublicIdOrderByStartedAtAsc(sessionId).stream()
                .filter(trace -> isStudentUser(trace.getUser()))
                .map(AgentRunTraceResponse::from)
                .toList();
    }

    @Transactional(readOnly = true)
    public AgentRunTraceResponse trace(String traceId) {
        AgentRunTrace trace = agentRunTraceRepository.findByTraceId(traceId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Run trace not found"));
        if (!isStudentUser(trace.getUser())) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Run trace not found");
        }
        return AgentRunTraceResponse.from(trace);
    }

    private static AgentRunTraceStep toEntity(AgentStep step) {
        AgentRunTraceStep entity = new AgentRunTraceStep();
        entity.setStepNumber(step.step());
        entity.setAgent(step.agent());
        entity.setAction(step.action());
        entity.setObservation(step.observation() == null ? "" : step.observation());
        entity.setCreatedAt(step.createdAt());
        return entity;
    }

    private static boolean isStudentUser(UserAccount user) {
        return user != null && !user.getRoles().contains(ROLE_ADMIN);
    }
}
