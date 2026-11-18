package com.mindbridge.agent.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.mindbridge.agent.domain.*;
import com.mindbridge.agent.dto.AnswerFeedbackRequest;
import com.mindbridge.agent.repository.*;
import com.mindbridge.agent.runtime.*;
import java.util.*;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

@Service
public class ConversationService {
    private final ChatSessionRepository sessions;private final ChatMessageRepository messages;private final TurnReceiptRepository receipts;
    private final KnowledgeChunkRepository chunks;private final AnswerFeedbackRepository feedback;private final ObjectMapper json;
    private HistorySnapshotCache cache;
    @org.springframework.beans.factory.annotation.Autowired public void setCache(HistorySnapshotCache cache){this.cache=cache;}
    private final PrivacySanitizer privacy=new PrivacySanitizer();
    public ConversationService(ChatSessionRepository sessions,ChatMessageRepository messages,TurnReceiptRepository receipts,KnowledgeChunkRepository chunks,AnswerFeedbackRepository feedback,ObjectMapper json){this.sessions=sessions;this.messages=messages;this.receipts=receipts;this.chunks=chunks;this.feedback=feedback;this.json=json;}
    private ChatSession owned(long user,String id){return sessions.findByPublicIdAndUser_Id(id,user).filter(s->!s.isHidden()).orElseThrow(()->missing("会话不存在"));}
    private static ResponseStatusException missing(String message){return new ResponseStatusException(HttpStatus.NOT_FOUND,message);}
    private Map<String,Object> summary(ChatSession s){return Map.of("sessionId",s.getPublicId(),"title",privacy.sanitize(s.getTitle()),"createdAt",s.getCreatedAt().toString(),"updatedAt",s.getUpdatedAt().toString());}
    @Transactional(readOnly=true) public Map<String,Object> list(long user,int limit,int offset){
        if(limit<1||limit>50||offset<0)throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY,"分页参数超出范围");
        return Map.of("items",sessions.visiblePage(user,limit,offset).stream().map(this::summary).toList(),"total",sessions.countByUser_IdAndHiddenFalse(user),"limit",limit,"offset",offset);
    }
    @Transactional(readOnly=true) public Map<String,Object> detail(long user,String publicId){
        var session=owned(user,publicId);Map<String,TurnReceipt> runs=new HashMap<>();receipts.findByUserIdAndSessionIdOrderByCreatedAtAsc(user,publicId).forEach(r->runs.put(r.getRunId(),r));
        var items=messages.findBySession_PublicIdOrderByCreatedAtAsc(publicId).stream().filter(m->m.getUser().getId()==user&&(m.getRole()==MessageRole.USER||m.getRole()==MessageRole.ASSISTANT)).map(m->{
            var r=runs.get(m.getTurnId());boolean bound=r!=null&&(Objects.equals(r.getAssistantMessageId(),m.getId())||Objects.equals(r.getUserMessageId(),m.getId()));
            Map<String,Object> item=new LinkedHashMap<>();item.put("id",m.getId().toString());item.put("messageId",bound?m.getId():null);item.put("runId",bound?r.getRunId():null);item.put("role",m.getRole().name().toLowerCase(Locale.ROOT));item.put("content",privacy.sanitize(m.getContent()));item.put("createdAt",m.getCreatedAt().toString());item.put("citations",bound?CitationPayloads.build(outcome(r).map(RunOutcome::evidence).orElse(List.of()),m.getContent()):List.of());
            item.put("answerTrace",bound&&m.getRole()==MessageRole.ASSISTANT?Map.of("runId",r.getRunId(),"messageId",m.getId(),"status",traceStatus(r),"steps",steps(r)):null);return item;
        }).toList();var result=new LinkedHashMap<>(summary(session));result.put("messages",items);return result;
    }
    @Transactional public Map<String,Boolean> hide(long user,String id){var session=owned(user,id);session.setHidden(true);sessions.save(session);if(cache!=null)cache.evict(user,id);return Map.of("deleted",true);}
    @Transactional public Map<String,Object> saveFeedback(long user,AnswerFeedbackRequest request){
        var receipt=receipts.locked(request.runId()).filter(r->r.getUserId()==user&&Objects.equals(r.getAssistantMessageId(),request.messageId())).orElseThrow(()->missing("回答不存在"));
        messages.findById(request.messageId()).filter(m->m.getRole()==MessageRole.ASSISTANT&&m.getUser().getId()==user&&m.getSession().getPublicId().equals(receipt.getSessionId())).orElseThrow(()->missing("回答不存在"));
        // The run receipt is the serialization point for concurrent feedback updates.
        var row=feedback.findByUserIdAndMessageId(user,request.messageId()).orElseGet(AnswerFeedback::new);row.setUserId(user);row.setMessageId(request.messageId());row.setRunId(request.runId());row.setHelpful(request.helpful());row.setCategory(request.category());row.setNote(privacy.sanitize(request.note().trim()));row.touch();feedback.saveAndFlush(row);return Map.of("id",row.getId(),"saved",true);
    }
    @Transactional(readOnly=true) public Map<String,Object> evidence(long user,String run,long chunkId){
        var receipt=receipts.findById(run).filter(r->r.getUserId()==user).orElseThrow(()->missing("证据不存在"));owned(user,receipt.getSessionId());
        var row=outcome(receipt).map(RunOutcome::evidence).orElse(List.of()).stream().filter(e->String.valueOf(e.getOrDefault("chunk_id",e.get("chunkId"))).equals(String.valueOf(chunkId))).findFirst().orElseThrow(()->missing("证据不存在"));
        Map<?,?> metadata=row.get("metadata") instanceof Map<?,?> m?m:Map.of();Map<String,Object> result=new LinkedHashMap<>();result.put("chunkId",chunkId);result.put("source",row.getOrDefault("source",""));result.put("content",row.getOrDefault("content",row.getOrDefault("preview","")));result.put("sectionPath",metadata.containsKey("section_path")?metadata.get("section_path"):List.of());result.put("documentVersion",metadata.get("document_version"));result.put("page",metadata.containsKey("page")?metadata.get("page"):metadata.get("page_number"));result.put("title",metadata.containsKey("title")?metadata.get("title"):row.getOrDefault("source",""));result.put("url",metadata.get("url"));result.put("sourceUrl",metadata.get("url"));result.put("license",metadata.get("license"));result.put("metadata",metadata);result.put("available",chunks.existsById(chunkId));return result;
    }
    private Optional<RunOutcome> outcome(TurnReceipt receipt){try{return Optional.ofNullable(json.readValue(receipt.getOutcomeJson(),RunOutcome.class));}catch(Exception e){return Optional.empty();}}
    private Object steps(TurnReceipt receipt){try{return json.readValue(receipt.getStepsJson(),List.class);}catch(Exception e){return List.of();}}
    private String traceStatus(TurnReceipt receipt){return outcome(receipt).filter(RunOutcome::degraded).isPresent()?"degraded":receipt.getStatus();}
}
