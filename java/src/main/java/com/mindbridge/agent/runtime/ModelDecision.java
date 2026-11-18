package com.mindbridge.agent.runtime;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.core.type.TypeReference;
import java.util.*;

public record ModelDecision(String answer,List<Action> actions) {
    public record Action(String tool,Map<String,Object> arguments){}
    public static ModelDecision answer(String text){return new ModelDecision(text,List.of());}
    public static ModelDecision parse(String raw,ObjectMapper json){
        String text=raw==null?"":raw.trim();
        if(text.isEmpty())throw new IllegalArgumentException("Empty model response");
        if(!text.startsWith("{")&&!text.startsWith("[")&&!text.startsWith("```"))return answer(text);
        try{
            String clean=text.replaceFirst("^```(?:json)?\\s*","").replaceFirst("\\s*```$","");
            JsonNode value=json.readTree(clean);
            if(!value.isObject())throw new IllegalArgumentException("Malformed agent decision envelope");
            if(value.path("answer").isTextual()&&!value.path("answer").asText().isBlank())return answer(value.path("answer").asText());
            List<Action> actions=new ArrayList<>();actions.add(action(value,json));
            JsonNode extra=value.path("additional_tool_calls");if(!extra.isMissingNode()){if(!extra.isArray()||extra.size()>5)throw new IllegalArgumentException("Tool batch exceeds decision budget");for(JsonNode row:extra)actions.add(action(row,json));}
            return new ModelDecision(null,List.copyOf(actions));
        }catch(Exception e){throw new IllegalArgumentException("Malformed agent decision envelope",e);}
    }
    private static Action action(JsonNode row,ObjectMapper json){if(!row.path("tool").isTextual()||!row.path("arguments").isObject())throw new IllegalArgumentException("Malformed tool");return new Action(row.path("tool").asText(),json.convertValue(row.path("arguments"),new TypeReference<Map<String,Object>>(){}));}
}
