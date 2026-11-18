package com.mindbridge.agent.service.knowledge;
import java.util.*;
public final class LexicalReranker {
    private LexicalReranker() {}
    public static List<SearchResult> rerank(String query,List<SearchResult> input,int limit) {
        return input.stream().map(r -> new SearchResult(r.chunkId(),r.source(),r.content(),score(query,r.content(),r.score())))
                .sorted(Comparator.comparingDouble(SearchResult::score).reversed()).limit(Math.max(0,limit)).toList();
    }
    static double score(String query,String content,double base) {
        List<String> q=Bm25Scorer.tokenize(query),d=Bm25Scorer.tokenize(content); Map<String,Integer> qf=count(q),df=count(d);
        double dot=0,qn=0,dn=0;for(var e:qf.entrySet()){dot+=e.getValue()*df.getOrDefault(e.getKey(),0);qn+=e.getValue()*e.getValue();}for(int n:df.values())dn+=n*n;
        double cosine=qn==0 || dn==0 ? 0 : dot/Math.sqrt(qn*dn);
        String lower=content.toLowerCase(Locale.ROOT), compact=lower.replaceAll("\\s+","");
        String queryLower=query.toLowerCase(Locale.ROOT), compactQuery=queryLower.replaceAll("\\s+","");
        String[] raw=queryLower.split("[\\s，。！？、；：,.!?;:]+"); int terms=0,hits=0;
        for(String term:raw) if(term.length()>=2){terms++;if(lower.contains(term))hits++;}
        double keyword=terms==0?0:hits/(double)terms;
        Set<String> distinct=new HashSet<>(q);Set<String> found=new HashSet<>(d);found.retainAll(distinct);
        double coverage=distinct.isEmpty()?0:found.size()/(double)distinct.size();
        double phrase=!compactQuery.isEmpty() && compact.contains(compactQuery)?1:keyword;
        return base*0.55+(cosine*0.75+keyword*0.25)*0.25+coverage*0.15+phrase*0.05;
    }
    private static Map<String,Integer> count(List<String> list){Map<String,Integer> out=new HashMap<>();list.forEach(x->out.merge(x,1,Integer::sum));return out;}
}
