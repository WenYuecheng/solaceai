package com.mindbridge.agent.service.knowledge;

import java.util.*;

public final class ReciprocalRankFusion {
    private ReciprocalRankFusion() {}
    public static List<SearchResult> fuse(List<List<SearchResult>> lanes, int limit) {
        Map<String,SearchResult> originals=new LinkedHashMap<>(); Map<String,Double> scores=new LinkedHashMap<>();
        for (List<SearchResult> lane:lanes) {
            Set<String> seen=new HashSet<>();
            for (int i=0;i<lane.size();i++) {
                SearchResult r=lane.get(i); String key=r.chunkId()!=null ? "id:"+r.chunkId() : "content:"+r.source()+"\0"+r.content();
                if (!seen.add(key)) continue;
                originals.putIfAbsent(key,r); scores.merge(key,1.0/(60+i+1),Double::sum);
            }
        }
        return originals.entrySet().stream().sorted((a,b)->Double.compare(scores.get(b.getKey()),scores.get(a.getKey())))
            .limit(Math.max(0,limit)).map(e->{SearchResult r=e.getValue();return new SearchResult(r.chunkId(),r.source(),r.content(),scores.get(e.getKey()));}).toList();
    }
}
