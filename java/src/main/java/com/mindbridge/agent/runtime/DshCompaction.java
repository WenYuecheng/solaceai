package com.mindbridge.agent.runtime;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.IntArrayList;
import com.mindbridge.agent.service.ai.AiMessage;
import java.util.*;

/** DSH policies adapted from DeepSeek's MIT implementation; see THIRD_PARTY_NOTICES.md. */
public final class DshCompaction {
    private static final Encoding ENCODING=Encodings.newLazyEncodingRegistry().getEncoding(EncodingType.CL100K_BASE);
    private static final ObjectMapper JSON=new ObjectMapper();
    private static final String MARKER="\n\n[... tool result middle pruned ...]\n\n";
    private static final Set<String> TEXT=Set.of("content","excerpt","text","stdout","stderr","output","body","result");
    private static final Set<String> PROTECTED=Set.of("arguments","metadata","error","errors","source","sources","chunk_id","chunkId","chunk_uid","document_id","tenant_id","knowledge_base_id","status","url","name");
    private DshCompaction(){}
    public record Policy(int window,int output,int headroom) {
        public Policy { if(output<=0||headroom<=0||Math.floor(Math.min(window*.8,window-output-headroom))<=Math.floor((window-output)*.16)) throw new IllegalArgumentException("Invalid compaction budgets"); }
        public int input(){return window-output;}
        public int threshold(){return (int)Math.floor(Math.min(window*.8,input()-headroom));}
        public int retain(){return (int)Math.floor(input()*.16);}
    }
    public static String json(Object v){try{return JSON.writeValueAsString(v);}catch(Exception e){throw new IllegalArgumentException("Cannot encode context",e);}}
    public static int tokens(Object v){return ENCODING.countTokensOrdinary(v instanceof String s?s:json(v));}
    public static String limit(String s,int n){var all=ENCODING.encodeOrdinary(s);var out=new IntArrayList();for(int i=0;i<Math.min(Math.max(0,n),all.size());i++)out.add(all.get(i));return ENCODING.decode(out);}
    public static int historyCut(List<AiMessage> messages,int retain){
        List<Integer> cuts=new ArrayList<>();
        for(int i=1;i<messages.size();i++)if(messages.get(i).role().equals("user")&&messages.get(i-1).role().equals("assistant"))cuts.add(i);
        if(!messages.isEmpty()&&messages.get(messages.size()-1).role().equals("user")&&!cuts.isEmpty())cuts.remove(cuts.size()-1);
        int sum=0,keep=messages.size();for(int i=messages.size()-1;i>=0;i--){sum+=tokens(messages.get(i));keep=i;if(sum>=retain)break;}
        int result=0;for(int c:cuts)if(c<=keep)result=c;return result;
    }
    public static Object prune(Object value){
        if(value instanceof String s)return clip(s);
        Object copy=copy(value); List<Leaf> leaves=new ArrayList<>();walk(copy,null,null,leaves);
        int total=leaves.stream().mapToInt(l->length(l.text)).sum();if(total<=8192)return copy;
        int offset=0;boolean marked=false;
        for(Leaf leaf:leaves){String s=leaf.text;int size=length(s);int head=Math.max(0,Math.min(size,4096-offset));int tail=Math.min(size,Math.max(0,total-1024-offset));boolean omitted=head<tail;String replacement=s.substring(0,s.offsetByCodePoints(0,head))+(omitted&&!marked?MARKER:"")+s.substring(s.offsetByCodePoints(0,Math.max(head,tail)));if(omitted)marked=true;leaf.set(replacement);offset+=size;}
        return copy;
    }
    private static int length(String s){return s.codePointCount(0,s.length());}
    private static String clip(String s){int size=length(s);return size<=8192?s:s.substring(0,s.offsetByCodePoints(0,4096))+MARKER+s.substring(s.offsetByCodePoints(0,size-1024));}
    @SuppressWarnings("unchecked") private static Object copy(Object value){if(value instanceof Map<?,?> map){Map<String,Object> result=new LinkedHashMap<>();map.forEach((k,v)->result.put(k.toString(),copy(v)));return result;}if(value instanceof List<?> list)return new ArrayList<>(list.stream().map(DshCompaction::copy).toList());return value;}
    private static void walk(Object value,Object parent,Object key,List<Leaf> leaves){if(value instanceof String s){if(parent instanceof List<?>||TEXT.contains(String.valueOf(key)))leaves.add(new Leaf(parent,key,s));}else if(value instanceof Map<?,?> map){map.forEach((k,v)->{if(!PROTECTED.contains(k.toString()))walk(v,map,k,leaves);});}else if(value instanceof List<?> list){for(int i=0;i<list.size();i++)walk(list.get(i),list,i,leaves);}}
    private record Leaf(Object parent,Object key,String text){@SuppressWarnings("unchecked") void set(String value){if(parent instanceof Map m)m.put(key,value);else ((List)parent).set((Integer)key,value);}}
}
