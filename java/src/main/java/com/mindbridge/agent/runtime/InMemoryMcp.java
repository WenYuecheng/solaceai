package com.mindbridge.agent.runtime;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.modelcontextprotocol.client.*;
import io.modelcontextprotocol.server.*;
import io.modelcontextprotocol.spec.*;
import io.modelcontextprotocol.spec.McpSchema.*;
import java.time.Duration;
import java.util.*;
import java.util.function.Function;
import reactor.core.publisher.Mono;

/** Official MCP SDK client/server over serialized in-process JSON-RPC; no listening port. */
public final class InMemoryMcp implements AutoCloseable {
    private final McpSyncClient client;private final McpSyncServer server;private final ObjectMapper json;
    public InMemoryMcp(GovernedTools tools,ObjectMapper json){
        this.json=json;var pipe=new Pipe(json);
        var builder=McpServer.sync(pipe).serverInfo("solaceai-private-tools","1.0.0")
                .capabilities(ServerCapabilities.builder().tools(false).build()).objectMapper(json);
        for(var entry:tools.schemas().entrySet())builder.tool(new Tool(entry.getKey(),entry.getValue().get("description").toString(),DshCompaction.json(entry.getValue().get("parameters"))),
                (exchange,args)->{try{return new CallToolResult(DshCompaction.json(tools.call(entry.getKey(),args)),false);}catch(RuntimeException e){return new CallToolResult("Invalid or unauthorized tool request",true);}});
        server=builder.build();
        client=McpClient.sync(pipe.client).clientInfo(new Implementation("solaceai","1.0.0"))
                .requestTimeout(Duration.ofSeconds(180)).initializationTimeout(Duration.ofSeconds(5)).build();
        try{client.initialize();}catch(RuntimeException e){client.close();server.close();throw e;}
    }
    public Map<String,Map<String,Object>> schemas(){Map<String,Map<String,Object>> result=new LinkedHashMap<>();for(Tool tool:client.listTools().tools())result.put(tool.name(),Map.of("description",tool.description(),"parameters",json.convertValue(tool.inputSchema(),new TypeReference<Map<String,Object>>(){})));return result;}
    public Object call(String name,Map<String,Object> args){CallToolResult result=client.callTool(new CallToolRequest(name,args));if(Boolean.TRUE.equals(result.isError()))throw new IllegalArgumentException("Invalid or unauthorized tool request");String text=result.content().stream().filter(TextContent.class::isInstance).map(TextContent.class::cast).map(TextContent::text).reduce("",String::concat);try{return json.readValue(text,Object.class);}catch(Exception e){throw new IllegalArgumentException("Invalid MCP response",e);}}
    public void close(){try{client.close();}finally{server.close();}}
    private static final class Pipe implements McpServerTransportProvider {
        final ObjectMapper json;McpServerSession session;Function<Mono<JSONRPCMessage>,Mono<JSONRPCMessage>> receiver;
        final McpClientTransport client=new McpClientTransport(){
            public Mono<Void> connect(Function<Mono<JSONRPCMessage>,Mono<JSONRPCMessage>> handler){receiver=handler;return Mono.empty();}
            public Mono<Void> sendMessage(JSONRPCMessage message){return Mono.defer(()->session.handle(wire(message)));}
            public <T>T unmarshalFrom(Object data,TypeReference<T> type){return json.convertValue(data,type);}
            public Mono<Void> closeGracefully(){return Mono.empty();}
        };
        Pipe(ObjectMapper json){this.json=json;}
        JSONRPCMessage wire(JSONRPCMessage message){try{return McpSchema.deserializeJsonRpcMessage(json,json.writeValueAsString(message));}catch(Exception e){throw new IllegalArgumentException("Invalid MCP JSON-RPC frame",e);}}
        public void setSessionFactory(McpServerSession.Factory factory){session=factory.create(new McpServerTransport(){
            public Mono<Void> sendMessage(JSONRPCMessage message){return Mono.defer(()->receiver.apply(Mono.just(wire(message))).then());}
            public <T>T unmarshalFrom(Object data,TypeReference<T> type){return json.convertValue(data,type);}
            public Mono<Void> closeGracefully(){return Mono.empty();}
        });}
        public Mono<Void> notifyClients(String method,Object params){return session==null?Mono.empty():session.sendNotification(method,params);}
        public Mono<Void> closeGracefully(){return session==null?Mono.empty():session.closeGracefully();}
    }
}
