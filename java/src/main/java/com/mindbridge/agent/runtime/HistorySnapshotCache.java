package com.mindbridge.agent.runtime;

import com.fasterxml.jackson.databind.*;
import com.mindbridge.agent.service.PrivacySanitizer;
import com.mindbridge.agent.service.ai.AiMessage;
import java.time.Duration;
import java.util.*;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

/** SQL-watermarked, complete and user-scoped Redis snapshots; cache failures never lose history. */
@Component
public class HistorySnapshotCache {
    private final StringRedisTemplate redis;private final RuntimeProperties properties;private final ObjectMapper json;
    private volatile long cooldownUntil;
    public HistorySnapshotCache(StringRedisTemplate redis,RuntimeProperties properties,ObjectMapper json){this.redis=redis;this.properties=properties;this.json=json;}
    private String key(long user,String session){return properties.getRedisNamespace()+":history:v2:"+user+":"+session;}
    private boolean usable(){return redis!=null&&properties.isRedisHistoryEnabled()&&System.nanoTime()>cooldownUntil;}
    private void failed(){cooldownUntil=System.nanoTime()+5_000_000_000L;}
    public List<AiMessage> load(long user,String session,long latest,long count){if(!usable())return null;try{return parse(redis.opsForValue().get(key(user,session)),user,session,latest,count);}catch(Exception e){failed();return null;}}
    public void store(long user,String session,List<Map<String,Object>> messages){if(!usable()||messages.size()>40)return;try{String value=serialize(user,session,messages);if(value.getBytes(java.nio.charset.StandardCharsets.UTF_8).length<=256*1024)redis.opsForValue().set(key(user,session),value,Duration.ofHours(24));}catch(Exception e){failed();}}
    public void evict(long user,String session){if(!usable())return;try{redis.delete(key(user,session));}catch(Exception e){failed();}}
    String serialize(long user,String session,List<Map<String,Object>> messages) throws Exception {
        var clean=messages.stream().map(m->Map.of("id",m.get("id"),"role",m.get("role").toString().toLowerCase(Locale.ROOT),"content",new PrivacySanitizer().sanitize(m.get("content").toString()))).toList();
        return json.writeValueAsString(Map.of("schemaVersion",2,"userId",user,"sessionId",session,"lastMessageId",messages.isEmpty()?0:messages.get(messages.size()-1).get("id"),"complete",true,"messages",clean));
    }
    List<AiMessage> parse(String raw,long user,String session,long latest,long count){
        if(raw==null||raw.getBytes(java.nio.charset.StandardCharsets.UTF_8).length>256*1024)return null;
        try {JsonNode root=json.readTree(raw);if(!root.isObject()||root.path("schemaVersion").asInt()!=2||!root.path("userId").isIntegralNumber()||root.path("userId").asLong()!=user||!root.path("sessionId").asText().equals(session)||!root.path("complete").isBoolean()||!root.path("complete").asBoolean()||!root.path("lastMessageId").isIntegralNumber()||root.path("lastMessageId").asLong()!=latest)return null;
            var rows=root.path("messages");if(!rows.isArray()||rows.size()>40||rows.size()!=count)return null;List<AiMessage> out=new ArrayList<>();long previous=0;
            for(var row:rows){if(!row.path("id").isIntegralNumber()||row.path("id").asLong()<=previous||!row.path("content").isTextual()||!Set.of("user","assistant").contains(row.path("role").asText()))return null;previous=row.path("id").asLong();out.add(new AiMessage(row.path("role").asText(),new PrivacySanitizer().sanitize(row.path("content").asText())));}return previous==latest?out:null;
        }catch(Exception e){return null;}
    }
}
