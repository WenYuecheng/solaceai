package com.mindbridge.agent.runtime;

import com.fasterxml.jackson.databind.*;
import com.mindbridge.agent.service.ai.*;
import java.net.*;
import java.net.http.*;
import java.time.Duration;
import java.util.*;
import reactor.core.publisher.Flux;

/** OpenAI-compatible local dual-LoRA transport. Never permits tool execution or redirects. */
public final class LocalGatewayClient implements AiClient {
    private final RuntimeProperties properties; private final String model; private final ObjectMapper json;
    // The local Uvicorn gateway speaks HTTP/1.1; do not attempt a cleartext HTTP/2 upgrade.
    private final HttpClient http=HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(3)).followRedirects(HttpClient.Redirect.NEVER).build();
    public LocalGatewayClient(RuntimeProperties p,String model,ObjectMapper json){
        URI uri=URI.create(p.getLocalGatewayUrl());
        if(!"http".equals(uri.getScheme())||!"127.0.0.1".equals(uri.getHost())||uri.getPort()<1||uri.getUserInfo()!=null||uri.getQuery()!=null||uri.getFragment()!=null||!(uri.getPath().isEmpty()||uri.getPath().equals("/")))throw new IllegalArgumentException("Only a numeric loopback gateway is allowed");
        if(!Set.of("solace-response","solace-emotion").contains(model))throw new IllegalArgumentException("Unsupported local adapter");this.properties=p;this.model=model;this.json=json;
    }
    public String complete(List<AiMessage> messages){
        int timeout=model.equals("solace-emotion")?properties.getEmotionTimeoutSeconds():properties.getModelTimeoutSeconds();
        try{
            String payload=json.writeValueAsString(Map.of("model",model,"messages",messages,"temperature",0,"max_tokens",model.equals("solace-emotion")?16:256,"stream",false));
            var builder=HttpRequest.newBuilder(URI.create(properties.getLocalGatewayUrl().replaceAll("/$","")+"/v1/chat/completions"))
                .timeout(Duration.ofSeconds(Math.max(1,timeout))).header("Content-Type","application/json").header("X-Solace-Timeout",String.valueOf(timeout)).POST(HttpRequest.BodyPublishers.ofString(payload));
            if(!properties.getLocalGatewayToken().isBlank())builder.header("Authorization","Bearer "+properties.getLocalGatewayToken());
            HttpResponse<String> response=null;for(int i=0;i<5;i++){response=http.send(builder.build(),HttpResponse.BodyHandlers.ofString());if(response.statusCode()!=503)break;if(Thread.currentThread().isInterrupted())throw new InterruptedException();Thread.sleep(200);}
            if(response==null||response.statusCode()!=200)throw new IllegalStateException("Local model unavailable");
            JsonNode body=json.readTree(response.body());if(!model.equals(body.path("model").asText()))throw new IllegalArgumentException("Local adapter identity mismatch");
            JsonNode message=body.path("choices").path(0).path("message");var calls=message.path("tool_calls");var function=message.path("function_call");
            if((!calls.isMissingNode()&&!calls.isNull()&&!(calls.isArray()&&calls.isEmpty()))||(!function.isMissingNode()&&!function.isNull()))
                throw new IllegalArgumentException("Unexpected local tool call");
            if(!message.path("content").isTextual()||message.path("content").asText().isBlank())throw new IllegalArgumentException("Empty local response");return message.path("content").asText().trim();
        }catch(InterruptedException e){Thread.currentThread().interrupt();throw new java.util.concurrent.CancellationException();}catch(RuntimeException e){throw e;}catch(Exception e){throw new IllegalStateException("Local model unavailable",e);}
    }
    public Flux<String> stream(List<AiMessage> messages){return Flux.defer(()->Flux.just(complete(messages)));}
}
