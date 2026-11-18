package com.mindbridge.agent.service;

import com.mindbridge.agent.dto.ChatRequest;
import com.mindbridge.agent.runtime.GraphHarness;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

/** SSE adapter around the LangGraph4j harness. Cancellation never releases a still-running lease. */
@Service
public class ChatService {
    private final GraphHarness harness;
    public ChatService(GraphHarness harness){this.harness=harness;}
    public Flux<ServerSentEvent<Map<String,Object>>> streamChat(Long userId,ChatRequest request){
        return Flux.<ServerSentEvent<Map<String,Object>>>create(sink->{
            var cancelled=new AtomicBoolean();sink.onCancel(()->cancelled.set(true));
            Schedulers.boundedElastic().schedule(()->{
                var streamed=new StringBuilder();
                java.util.function.Consumer<Map<String,Object>> emit=event->{if(cancelled.get()||sink.isCancelled())return;if("token".equals(event.get("type")))streamed.append(event.getOrDefault("content",""));sink.next(frame(event));};
                try{
                    emit.accept(Map.of("type","status","message","正在理解问题并整理依据"));
                    var result=harness.execute(userId,request,emit,cancelled);if(cancelled.get())return;
                    String answer=result.outcome().answer();
                    if(streamed.isEmpty()){for(int i=0;i<answer.length();){int end=answer.offsetByCodePoints(i,Math.min(32,answer.codePointCount(i,answer.length())));emit.accept(Map.of("type","token","content",answer.substring(i,end),"sessionId",result.sessionId()));i=end;}}
                    else if(!streamed.toString().equals(answer))emit.accept(Map.of("type","replace","content",answer,"sessionId",result.sessionId()));
                    if(!result.outcome().evidence().isEmpty()){
                        var citations=com.mindbridge.agent.runtime.CitationPayloads.build(result.outcome().evidence(),answer);
                        emit.accept(Map.of("type","citation","citations",citations));
                    }
                    Map<String,Object> done=new LinkedHashMap<>();done.put("type","done");done.put("sessionId",result.sessionId());done.put("runId",result.runId());done.put("messageId",result.messageId());done.put("traceStatus",result.outcome().degraded()?"degraded":"completed");done.put("firstTokenSeconds",result.firstTokenSeconds());done.put("elapsedSeconds",result.elapsedSeconds());emit.accept(done);sink.complete();
                }catch(RuntimeException e){if(!cancelled.get()){String code=e.getMessage()!=null&&e.getMessage().contains("正在回复")?"SESSION_BUSY":"RUN_FAILED";emit.accept(Map.of("type","error","code",code,"message",code.equals("SESSION_BUSY")?"本次对话正在回复，请等待完成。":"这次回复未能完成，请稍后重试。"));sink.complete();}}
            });
        },reactor.core.publisher.FluxSink.OverflowStrategy.BUFFER);
    }
    private static ServerSentEvent<Map<String,Object>> frame(Map<String,Object> event){return ServerSentEvent.<Map<String,Object>>builder(event).event(String.valueOf(event.get("type"))).build();}
}
