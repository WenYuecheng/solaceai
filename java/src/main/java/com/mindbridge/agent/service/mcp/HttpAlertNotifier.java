package com.mindbridge.agent.service.mcp;

import com.mindbridge.agent.config.MindBridgeProperties;
import com.mindbridge.agent.domain.AlertRecord;
import com.mindbridge.agent.domain.PsychologicalReport;
import java.util.Map;
import org.springframework.web.reactive.function.client.WebClient;

/**
 * 通过 HTTP MCP 服务发送高风险预警。
 */
public class HttpAlertNotifier implements AlertNotifier {

    private final WebClient webClient;

    public HttpAlertNotifier(WebClient.Builder webClientBuilder, MindBridgeProperties properties) {
        this.webClient = webClientBuilder.baseUrl(properties.getMcp().getEmail().getUrl()).build();
    }

    @Override
    public void notify(AlertRecord alertRecord, PsychologicalReport report) {
        String response = webClient.post()
                .uri("/send")
                .bodyValue(Map.of(
                        "recipient", alertRecord.getRecipient(),
                        "reportId", report.getId(),
                        "userId", report.getUser().getId(),
                        "username", report.getUser().getUsername(),
                        "riskLevel", report.getRiskLevel().name(),
                        "summary", report.getSummary(),
                        "content", report.getContent()))
                .retrieve()
                .bodyToMono(String.class)
                .block(java.time.Duration.ofSeconds(30));
        if (response == null || response.isBlank()) throw new IllegalStateException("HTTP mail delivery acknowledgement missing");
        String status;
        try {status = new com.fasterxml.jackson.databind.ObjectMapper().readTree(response).path("status").asText();}
        catch (com.fasterxml.jackson.core.JsonProcessingException error) {throw new IllegalStateException("Invalid mail delivery acknowledgement",error);}
        if ("LOG_ONLY".equals(status)) throw new LogOnlyDeliveryException();
        if (!"SMTP_ACCEPTED".equals(status)) throw new IllegalStateException("Mail acceptance was not acknowledged");
    }
}
