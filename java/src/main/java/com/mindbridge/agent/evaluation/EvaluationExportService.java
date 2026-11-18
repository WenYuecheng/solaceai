package com.mindbridge.agent.evaluation;

import com.fasterxml.jackson.databind.*;
import com.mindbridge.agent.repository.*;
import java.util.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

/** Allowlisted persisted-run export; never exports configuration, credentials or full prompt/history. */
@Service
public class EvaluationExportService {
    private final TurnReceiptRepository receipts;
    private final ChatMessageRepository messages;
    private final ObjectMapper json;
    public EvaluationExportService(TurnReceiptRepository receipts,ChatMessageRepository messages,ObjectMapper json){this.receipts=receipts;this.messages=messages;this.json=json;}
    @Transactional(readOnly=true)
    public Map<String,Object> export(String runId){
        var row=receipts.findById(runId).orElseThrow(()->new ResponseStatusException(HttpStatus.NOT_FOUND,"Run not found"));
        JsonNode outcome=json.createObjectNode();
        try{if(row.getOutcomeJson()!=null)outcome=json.readTree(row.getOutcomeJson());}
        catch(Exception e){throw new ResponseStatusException(HttpStatus.CONFLICT,"Stored outcome cannot be decoded");}
        var result=new LinkedHashMap<String,Object>();
        result.put("schema","mindbridge-evaluation-v1");result.put("targetBackend","java");result.put("id",runId);
        result.put("user_input",row.getUserMessageId()==null?"":messages.findById(row.getUserMessageId()).map(m->m.getContent()).orElse(""));
        result.put("response",outcome.path("answer").asText(""));
        // Only evidence actually projected to the model can ground RAGAS faithfulness.
        var evidence=new ArrayList<Map<String,Object>>();var contexts=new ArrayList<String>();
        for(var item:outcome.path("modelVisibleEvidence")){
            var clean=new LinkedHashMap<String,Object>();
            for(String key:List.of("source","content","chunk_uid","chunk_id","version","page"))if(item.has(key)&&item.get(key).isValueNode())clean.put(key,item.get(key).asText());
            var visibleText=item.has("content")?item.get("content"):item.path("excerpt");
            if(visibleText.isTextual()){clean.put("content",visibleText.asText());contexts.add(visibleText.asText());}evidence.add(clean);
        }
        result.put("evidence",evidence);result.put("retrieved_contexts",contexts);
        result.put("risk",outcome.path("risk").asText(""));result.put("intent",outcome.path("intent").asText(""));
        var actions=new ArrayList<String>();for(var step:outcome.path("agentSteps"))if(step.path("action").isTextual())actions.add(step.path("action").asText());result.put("actions",actions);
        result.put("receipt",Map.of("status",row.getStatus(),"hasAnswer",row.getAssistantMessageId()!=null,"hasReport",row.getReportId()!=null,"hasTrace",row.getTraceId()!=null));
        result.put("meta",Map.of("modelBackend",row.getModelBackend(),"degraded",outcome.path("degraded").asBoolean(),"evidenceScope","model_visible","redacted",false));
        return result;
    }
}
