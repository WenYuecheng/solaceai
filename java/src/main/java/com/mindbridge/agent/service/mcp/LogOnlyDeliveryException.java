package com.mindbridge.agent.service.mcp;

/** Explicit remote dry-run outcome; never a successful mail acceptance. */
public class LogOnlyDeliveryException extends RuntimeException {
    public LogOnlyDeliveryException(){super("Delivery was logged only");}
}
