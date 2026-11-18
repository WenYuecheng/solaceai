package com.mindbridge.agent.service.knowledge;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.mindbridge.agent.config.ParityKnowledgeProperties;
import com.mindbridge.agent.domain.KnowledgeChunk;
import java.net.URI;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.*;

/** Milvus REST v2 hybrid adapter. Collection is isolated by app prefix and model identity. */
public final class MilvusGateway implements KnowledgeIndex {
    private final ParityKnowledgeProperties properties;
    private final ObjectMapper mapper;
    private final BgeClient bge;
    private final HttpClient http=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).followRedirects(HttpClient.Redirect.NEVER).build();
    private final URI base;
    public MilvusGateway(ParityKnowledgeProperties properties,ObjectMapper mapper) {
        this.properties=properties;this.mapper=mapper;
        if (!properties.getCollection().matches("mindbridge_[A-Za-z0-9_]{1,90}")) throw new IllegalArgumentException("Invalid MindBridge collection prefix");
        base=URI.create(properties.getMilvusUrl());
        if (!List.of("http","https").contains(base.getScheme()) || base.getHost()==null || base.getUserInfo()!=null || base.getQuery()!=null || base.getFragment()!=null) throw new IllegalArgumentException("Invalid Milvus URL");
        bge=new BgeClient(properties.getBgeUrl(),Duration.ofSeconds(180),mapper);
    }
    public JsonNode workerHealth(){return bge.health();}
    public Map<String,Object> status() {
        Map<String,Object> status=new LinkedHashMap<>(); status.put("backend","milvus_bge");
        boolean worker=false, reranker=false, vector=false;
        try {
            JsonNode health=workerHealth(); worker="UP".equalsIgnoreCase(health.path("status").asText()) && "solaceai-local-bge".equals(health.path("service").asText());
            for(JsonNode model:health.path("loadedModels")) if("reranker".equals(model.asText())) reranker=true;
            if(worker) { String name=collectionName(); vector=hasCollection(name) && "LoadStateLoaded".equals(call("/collections/get_load_state",Map.of("collectionName",name)).path("data").path("loadState").asText()); }
        } catch(Exception ignored) { /* Return observed health instead of implying readiness. */ }
        status.put("workerHealthy",worker); status.put("vectorHealthy",vector); status.put("rerankerLoaded",reranker); status.put("rerankerEnabled",properties.isRerankerEnabled());
        return status;
    }
    public String collectionName() {
        JsonNode health=bge.health();
        String identity=health.path("embeddingIdentity").asText("");
        if (!"solaceai-local-bge".equals(health.path("service").asText()) || !"UP".equalsIgnoreCase(health.path("status").asText()) || identity.isBlank()) throw new IllegalStateException("BGE worker is not ready");
        try {String digest=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest((properties.getTenantId()+"\0"+properties.getKnowledgeBaseId()+"\0"+identity).getBytes(StandardCharsets.UTF_8)));return properties.getCollection()+"_v2_"+digest.substring(0,16);}
        catch(Exception e){throw new IllegalStateException(e);}
    }
    private boolean hasCollection(String name){
        JsonNode has=call("/collections/has",Map.of("collectionName",name)).path("data").path("has");
        if(!has.isBoolean()) throw new IllegalStateException("Invalid Milvus collection response");
        return has.booleanValue();
    }
    private void ensure(String name){
        if(hasCollection(name)) return;
        Map<String,Object> schema=Map.of("autoId",false,"enableDynamicField",false,"fields",List.of(
                Map.of("fieldName","id","dataType","Int64","isPrimary",true),
                Map.of("fieldName","dense","dataType","FloatVector","elementTypeParams",Map.of("dim","1024")),
                Map.of("fieldName","sparse","dataType","SparseFloatVector")));
        var indexes=List.of(Map.of("fieldName","dense","indexName","dense_idx","metricType","COSINE","params",Map.of("index_type","FLAT")),
                Map.of("fieldName","sparse","indexName","sparse_idx","metricType","IP","params",Map.of("index_type","SPARSE_INVERTED_INDEX")));
        call("/collections/create",Map.of("collectionName",name,"schema",schema,"indexParams",indexes,"params",Map.of("consistencyLevel","Strong")));
    }
    @Override public void upsert(List<KnowledgeChunk> chunks) {
        if(chunks.isEmpty()) return;
        String collection=collectionName();ensure(collection);
        for(int start=0;start<chunks.size();start+=8){
            var batch=chunks.subList(start,Math.min(chunks.size(),start+8));
            var embedding=bge.embed(batch.stream().map(this::embeddingText).toList());
            List<Map<String,Object>> data=new ArrayList<>();
            for(int i=0;i<batch.size();i++) data.add(Map.of("id",batch.get(i).getId(),"dense",embedding.dense().get(i),"sparse",embedding.sparse().get(i)));
            call("/entities/upsert",Map.of("collectionName",collection,"data",data));
        }
    }
    private String embeddingText(KnowledgeChunk chunk) {
        try {
            JsonNode metadata=chunk.getMetadataJson()==null ? mapper.createObjectNode() : mapper.readTree(chunk.getMetadataJson());
            List<String> titles=new ArrayList<>(); for(JsonNode title:metadata.path("section_path")) if(title.isTextual()) titles.add(title.asText());
            return String.join(" > ",titles)+"\n"+chunk.getContent();
        } catch(Exception e) {throw new IllegalArgumentException("Invalid chunk metadata",e);}
    }
    @Override public void delete(List<Long> ids) {
        if(ids.isEmpty()) return;
        String collection=collectionName(); if(!hasCollection(collection)) return;
        for(int start=0;start<ids.size();start+=1000) call("/entities/delete",Map.of("collectionName",collection,"filter","id in ["+String.join(",",ids.subList(start,Math.min(ids.size(),start+1000)).stream().map(String::valueOf).toList())+"]"));
    }
    @Override public int rebuild(List<KnowledgeChunk> active) {
        String collection=collectionName();
        if(!active.isEmpty()) upsert(active);
        if(!hasCollection(collection)) return 0;
        Set<Long> existing=new HashSet<>();
        // REST offset pagination is bounded by Milvus. Do not delete anything on incomplete enumeration.
        for(int offset=0;;offset+=1000) {
            if(offset>=16000) throw new IllegalStateException("Collection too large for safe REST rebuild enumeration");
            JsonNode rows=call("/entities/query",Map.of("collectionName",collection,"filter","id >= 0","outputFields",List.of("id"),"offset",offset,"limit",Math.min(1000,16384-offset))).path("data");
            if(!rows.isArray()) throw new IllegalStateException("Invalid Milvus query response");
            for(JsonNode row:rows) { if(!validId(row.path("id"))) throw new IllegalStateException("Invalid Milvus ID"); existing.add(parseId(row.path("id"))); }
            if(rows.size()<1000) break;
        }
        Set<Long> wanted=new HashSet<>(); active.forEach(c->wanted.add(c.getId())); existing.removeAll(wanted);
        delete(new ArrayList<>(existing)); return active.size();
    }
    public List<SearchResult> search(String query,List<KnowledgeChunk> active,int limit) {
        if(active.isEmpty() || limit<=0) return List.of();
        String collection=collectionName(); if(!hasCollection(collection)) return List.of();
        var embedding=bge.embed(List.of(query));
        Map<Long,KnowledgeChunk> byId=new LinkedHashMap<>();active.forEach(c->byId.put(c.getId(),c));
        String filter="id in ["+String.join(",",byId.keySet().stream().map(String::valueOf).toList())+"]";
        var dense=Map.of("annsField","dense","data",List.of(embedding.dense().get(0)),"filter",filter,"limit",limit,"metricType","COSINE");
        var sparse=Map.of("annsField","sparse","data",List.of(embedding.sparse().get(0)),"filter",filter,"limit",limit,"metricType","IP");
        JsonNode data=call("/entities/hybrid_search",Map.of("collectionName",collection,"search",List.of(dense,sparse),"rerank",Map.of("strategy","rrf","params",Map.of("k",60)),"limit",limit,"outputFields",List.of("id"))).path("data");
        if (!data.isArray()) throw new IllegalStateException("Invalid Milvus search response");
        List<SearchResult> result=new ArrayList<>();
        for(JsonNode row:data){ if(!validId(row.path("id"))) continue; KnowledgeChunk c=byId.get(parseId(row.path("id")));
            if(c!=null) result.add(new SearchResult(c.getId(),c.getSource(),c.getContent(),row.path("distance").asDouble(0))); }
        return result;
    }
    public List<SearchResult> rerank(String query,List<SearchResult> candidates,int limit) {
        List<SearchResult> selected=candidates.stream().limit(32).toList();
        if(selected.isEmpty()) return List.of();
        return bge.rerank(query,selected.stream().map(SearchResult::content).toList()).stream().limit(limit)
                .map(r->{SearchResult c=selected.get(r.index());return new SearchResult(c.chunkId(),c.source(),c.content(),r.score());}).toList();
    }
    private static boolean validId(JsonNode node) {
        try {parseId(node);return true;} catch(RuntimeException e){return false;}
    }
    private static long parseId(JsonNode node) {
        if(node.isIntegralNumber() && node.canConvertToLong()) return node.longValue();
        if(node.isTextual() && node.asText().matches("[0-9]{1,19}")) return Long.parseLong(node.asText());
        throw new IllegalArgumentException("Invalid Milvus ID");
    }
    private JsonNode call(String path,Object body) {
        try {
            var builder=HttpRequest.newBuilder(base.resolve("/v2/vectordb"+path)).timeout(Duration.ofSeconds(30)).header("Content-Type","application/json");
            if(!properties.getMilvusToken().isBlank()) builder.header("Authorization","Bearer "+properties.getMilvusToken());
            var req=builder.POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body))).build();
            var response=http.send(req,HttpResponse.BodyHandlers.ofString());
            if(response.statusCode()!=200) throw new IllegalStateException("Milvus HTTP "+response.statusCode());
            JsonNode result=mapper.readTree(response.body()); if(result.path("code").asInt(-1)!=0) throw new IllegalStateException("Milvus request failed"); return result;
        } catch(InterruptedException e){Thread.currentThread().interrupt();throw new IllegalStateException("Milvus interrupted",e);}
        catch(Exception e){throw new IllegalStateException("Milvus unavailable",e);}
    }
}
