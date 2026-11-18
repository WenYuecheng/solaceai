package com.mindbridge.agent.service;

import org.springframework.stereotype.Service;

@Service
/**
 * 输入隐私脱敏服务。
 *
 * <p>对发送给模型和评估链路的文本做轻量脱敏，降低敏感标识进入上下文的概率。</p>
 */
public class PrivacySanitizer {

    public String sanitize(String text) {
        if (text == null || text.isBlank()) {
            return "";
        }
        // 模型侧只需要语义，不需要手机号、学号、证件号等敏感标识。
        String sanitized = text.replaceAll("(?<![0-9])[0-9]{17}[0-9Xx](?![0-9])", "[已脱敏]");
        sanitized = sanitized.replaceAll("1[3-9]\\d{9}", "[已脱敏]");
        sanitized = sanitized.replaceAll("[\\w.+-]+@[\\w-]+(?:\\.[\\w-]+)+", "[已脱敏]");
        return sanitized;
    }
}
