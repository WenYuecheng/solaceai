package com.mindbridge.agent.runtime;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.mindbridge.agent.domain.*;
import com.mindbridge.agent.dto.ChatRequest;
import com.mindbridge.agent.repository.*;
import com.mindbridge.agent.service.*;
import com.mindbridge.agent.service.agent.*;
import com.mindbridge.agent.service.ai.AiMessage;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.*;
import java.util.function.Consumer;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

/** Owns turn leases, durable receipts and transactional completion around the graph. */
@Service
public class GraphHarness {
    private final GraphRuntime graph;private final RuntimeProperties properties;private final TurnLocks locks;
    private final UserAccountRepository users;private final ChatSessionRepository sessions;private final ChatMessageRepository messages;
    private final TurnReceiptRepository receipts;private final PsychologicalReportRepository reports;
    private final AgentRunTraceService traces;private final ToolOrchestrationService business;private final ObjectMapper json;private final TransactionTemplate tx;
    @jakarta.persistence.PersistenceContext private jakarta.persistence.EntityManager entityManager;
    public GraphHarness(GraphRuntime graph,RuntimeProperties properties,UserAccountRepository users,ChatSessionRepository sessions,ChatMessageRepository messages,TurnReceiptRepository receipts,PsychologicalReportRepository reports,AgentRunTraceService traces,ToolOrchestrationService business,ObjectMapper json,PlatformTransactionManager manager){this.graph=graph;this.properties=properties;this.locks=new TurnLocks(Path.of(properties.getLockPath()));this.users=users;this.sessions=sessions;this.messages=messages;this.receipts=receipts;this.reports=reports;this.traces=traces;this.business=business;this.json=json;this.tx=new TransactionTemplate(manager);}
    private HistorySnapshotCache historyCache;
    @org.springframework.beans.factory.annotation.Autowired public void setHistoryCache(HistorySnapshotCache historyCache){this.historyCache=historyCache;}
    private CaseWorkflow cases;
    @org.springframework.beans.factory.annotation.Autowired public void setCases(CaseWorkflow cases){this.cases=cases;}
    public record Result(String sessionId,String runId,Long messageId,RunOutcome outcome,Double firstTokenSeconds,double elapsedSeconds,boolean replay){}
    public Result execute(long userId,ChatRequest request,Consumer<Map<String,Object>> sink,AtomicBoolean cancelled){
        String run=request.runId()==null?UUID.randomUUID().toString().replace("-",""):request.runId().toLowerCase(Locale.ROOT);
        String backend=request.modelBackend()==null?properties.getDefaultBackend():request.modelBackend();
        String input=request.message().trim();String digest=TurnLocks.hash(input);long start=System.nanoTime();
        // On a replay with no session supplied, obtain its already-bound session before taking locks.
        String sessionKey=request.sessionId();var existing=receipts.findById(run);if(sessionKey==null&&existing.isPresent()&&existing.get().getUserId()==userId)sessionKey=existing.get().getSessionId();
        try(var lease=locks.acquire(userId,sessionKey,run)){
            TurnReceipt receipt=tx.execute(status->prepare(userId,request,run,backend,input,digest));
            try(var sessionLease=sessionKey==null?locks.acquire(userId,receipt.getSessionId(),"session-init:"+run):null){
            sink.accept(Map.of("type","meta","sessionId",receipt.getSessionId(),"runId",run));
            if(receipt.getOutcomeJson()!=null&&receipt.getStatus().equals("completed")){
                RunOutcome outcome=decode(receipt.getOutcomeJson());
                for(var step:outcome.steps())sink.accept(Map.of("type","step","sessionId",receipt.getSessionId(),"runId",run,"step",step));
                return new Result(receipt.getSessionId(),run,receipt.getAssistantMessageId(),outcome,receipt.getFirstTokenSeconds(),receipt.getElapsedSeconds()==null?0:receipt.getElapsedSeconds(),true);
            }
            var events=new CopyOnWriteArrayList<Map<String,Object>>();var first=new AtomicReference<Double>();
            Consumer<Map<String,Object>> emit=event->{if(cancelled.get())return;events.add(new LinkedHashMap<>(event));if("token".equals(event.get("type")))first.compareAndSet(null,(System.nanoTime()-start)/1_000_000_000.0);if("step".equals(event.get("type")))saveSteps(run,events);sink.accept(event);};
            try{
                List<AiMessage> history=loadHistory(userId,receipt);
                RunOutcome outcome=graph.run(userId,receipt.getSessionId(),input,history,run,backend,emit,cancelled::get,start+properties.getTimeoutSeconds()*1_000_000_000L);
                if(cancelled.get())throw new java.util.concurrent.CancellationException();
                double elapsed=(System.nanoTime()-start)/1_000_000_000.0;
                TurnReceipt completed=tx.execute(status->complete(run,outcome,first.get(),elapsed));
                if(historyCache!=null)historyCache.evict(userId,receipt.getSessionId());
                if(completed.getReportId()!=null){try{business.handleAsync(completed.getReportId());}catch(RuntimeException ignored){/* Recovery is driven by the durable report/tool records. */}}
                return new Result(completed.getSessionId(),run,completed.getAssistantMessageId(),outcome,first.get(),elapsed,false);
            }catch(RuntimeException e){interrupt(run,cancelled.get()||e instanceof java.util.concurrent.CancellationException,events);throw e;}
            }
        }
    }
    private List<AiMessage> loadHistory(long user,TurnReceipt receipt){
        long before=receipt.getUserMessageId();
        var stamp=tx.execute(s->messages.historyStamp(user,receipt.getSessionId(),before).get(0));
        long count=((Number)stamp[0]).longValue(), latest=((Number)stamp[1]).longValue();
        if(historyCache!=null&&count<=40&&latest<before){var cached=historyCache.load(user,receipt.getSessionId(),latest,count);if(cached!=null)return cached;}
        List<Map<String,Object>> rows=tx.execute(s->messages.historyBefore(user,receipt.getSessionId(),before).stream().map(m->Map.<String,Object>of("id",m.getId(),"role",m.getRole().name().toLowerCase(Locale.ROOT),"content",m.getContent())).toList());
        if(historyCache!=null)historyCache.store(user,receipt.getSessionId(),rows);
        return rows.stream().map(m->new AiMessage((String)m.get("role"),(String)m.get("content"))).toList();
    }
    private TurnReceipt prepare(long user,ChatRequest request,String run,String backend,String input,String hash){
        var prior=receipts.findById(run);
        if(prior.isPresent()){TurnReceipt row=prior.get();if(row.getUserId()!=user||!row.getInputHash().equals(hash))throw new IllegalArgumentException("运行记录不存在或输入已变化");if(!backend.equals(row.getModelBackend()))throw new IllegalArgumentException("本轮模型模式已变化");if(request.sessionId()!=null&&!request.sessionId().equals(row.getSessionId()))throw new IllegalArgumentException("会话不匹配");owned(user,row.getSessionId());return row;}
        UserAccount account=users.findById(user).orElseThrow(()->new IllegalArgumentException("User not found"));
        ChatSession session;
        if(request.sessionId()!=null)session=owned(user,request.sessionId());else{session=new ChatSession();session.setPublicId(UUID.randomUUID().toString().replace("-",""));session.setUser(account);session.setTitle(input.substring(0,Math.min(36,input.length())));sessions.saveAndFlush(session);}
        TurnReceipt row=new TurnReceipt();row.setRunId(run);row.setUserId(user);row.setSessionId(session.getPublicId());row.setInputHash(hash);row.setModelBackend(backend);row.setStatus("running");
        ChatMessage message=new ChatMessage();message.setSession(session);message.setUser(account);message.setRole(MessageRole.USER);message.setContent(input);message.setTurnId(run);messages.saveAndFlush(message);row.setUserMessageId(message.getId());session.touch();sessions.save(session);return receipts.saveAndFlush(row);
    }
    private ChatSession owned(long user,String publicId){ChatSession session=sessions.findByPublicIdAndUser_Id(publicId,user).orElseThrow(()->new IllegalArgumentException("会话不存在"));if(session.isHidden())throw new IllegalArgumentException("会话不存在");return session;}
    private TurnReceipt complete(String run,RunOutcome outcome,Double first,double elapsed){
        TurnReceipt row=receipts.findById(run).orElseThrow();if(row.getStatus().equals("completed"))return row;
        ChatSession session=owned(row.getUserId(),row.getSessionId());UserAccount account=users.findById(row.getUserId()).orElseThrow();
        ChatMessage answer=row.getAssistantMessageId()==null?new ChatMessage():messages.findById(row.getAssistantMessageId()).orElseThrow();answer.setUser(account);answer.setSession(session);answer.setRole(MessageRole.ASSISTANT);answer.setContent(outcome.answer());answer.setTurnId(run);messages.saveAndFlush(answer);row.setAssistantMessageId(answer.getId());
        if(outcome.requiresReport()&&row.getReportId()==null){var report=new PsychologicalReport();report.setUser(account);report.setSession(session);report.setContent(messages.findById(row.getUserMessageId()).orElseThrow().getContent());report.setIntent(outcome.intent());report.setEmotion(outcome.assessment().emotion());report.setEmotionScore(outcome.assessment().emotionScore());report.setRiskLevel(outcome.risk());report.setConfidence(outcome.assessment().confidence());report.setSummary(outcome.assessment().summary());reports.saveAndFlush(report);row.setReportId(report.getId());entityManager.persist(new BusinessToolExecution(report.getId()));if(cases!=null)cases.ensure(report);}
        if(row.getTraceId()==null){var steps=outcome.agentSteps().stream().map(step->new AgentStep(((Number)step.get("step")).intValue(),actor(step.get("agent").toString()),action(step.get("action").toString()),step.get("observation").toString(),Instant.now())).toList();
            var legacy=new AgentRunResult(outcome.intent(),outcome.risk(),outcome.assessment(),List.of(),List.of(),List.of(),outcome.memoryBrief(),null,null,outcome.intent()==IntentType.CHAT?AgentName.COMPANION_AGENT:AgentName.COUNSELOR_AGENT,steps);
            var trace=traces.saveRun(account,session,messages.findById(row.getUserMessageId()).orElseThrow(),new PrivacySanitizer().sanitize(messages.findById(row.getUserMessageId()).orElseThrow().getContent()),legacy,row.getCreatedAt(),Instant.now());row.setTraceId(trace.getTraceId());}
        row.setStatus("completed");row.setOutcomeJson(encode(outcome));row.setStepsJson(encode(outcome.steps()));row.setFirstTokenSeconds(first);row.setElapsedSeconds(elapsed);row.setUpdatedAt(Instant.now());session.touch();sessions.save(session);return receipts.saveAndFlush(row);
    }
    private void saveSteps(String run,List<Map<String,Object>> events){List<Object> steps=new ArrayList<>();Map<String,Object> byId=new LinkedHashMap<>();for(var event:events)if("step".equals(event.get("type"))&&event.get("step") instanceof Map<?,?> step)byId.put(String.valueOf(step.get("id")),step);steps.addAll(byId.values());tx.executeWithoutResult(s->{var row=receipts.findById(run).orElseThrow();row.setStepsJson(encode(steps));row.setUpdatedAt(Instant.now());receipts.saveAndFlush(row);});}
    private void interrupt(String run,boolean cancelled,List<Map<String,Object>> events){tx.executeWithoutResult(s->{var row=receipts.findById(run).orElseThrow();if(row.getStatus().equals("completed"))return;var account=users.findById(row.getUserId()).orElseThrow();var session=sessions.findByPublicIdAndUser_Id(row.getSessionId(),row.getUserId()).orElseThrow();var answer=row.getAssistantMessageId()==null?new ChatMessage():messages.findById(row.getAssistantMessageId()).orElseThrow();answer.setUser(account);answer.setSession(session);answer.setRole(MessageRole.ASSISTANT);answer.setTurnId(run);answer.setContent(cancelled?"本轮已停止。":"这次回复未能完成，请稍后重试。");messages.saveAndFlush(answer);row.setAssistantMessageId(answer.getId());row.setStatus(cancelled?"cancelled":"failed");row.setUpdatedAt(Instant.now());receipts.save(row);});}
    private static AgentName actor(String actor){return switch(actor){case "SafetyAgent"->AgentName.SAFETY_AGENT;case "ContextSelector"->AgentName.CONTEXT_SELECTOR;case "EmotionClassifier"->AgentName.EMOTION_CLASSIFIER;default->AgentName.MAIN_AGENT;};}
    private static AgentAction action(String action){return switch(action){case "review_reply"->AgentAction.REVIEW_REPLY;case "context_projection","context_compacted"->AgentAction.CONTEXT_PROJECTION;case "citation_audit"->AgentAction.CITATION_AUDIT;case "answer"->AgentAction.PLAN_RESPONSE;case "input_check","emotion_observation"->AgentAction.ASSESS_RISK;case "context_selected"->AgentAction.ROUTE_INTENT;default->AgentAction.CALL_TOOL;};}
    public String encode(Object value){try{return json.writeValueAsString(value);}catch(Exception e){throw new IllegalArgumentException("Invalid run state",e);}}
    public RunOutcome decode(String value){try{return json.readValue(value,RunOutcome.class);}catch(Exception e){throw new IllegalStateException("Stored run state is invalid",e);}}
}
