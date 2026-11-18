package com.mindbridge.agent.runtime;

import java.util.*;
import java.util.regex.Pattern;

/** Citation snapshots use retrieved content and count only markers in prose. */
public final class CitationPayloads {
    private CitationPayloads() {}
    public static List<Map<String,Object>> build(List<Map<String,Object>> evidence,String answer) {
        Set<String> cited=new HashSet<>();String fence=null;StringBuilder prose=new StringBuilder();
        for(String line:Objects.requireNonNullElse(answer,"").split("\\R")) {
            String s=line.stripLeading();String marker=s.startsWith("```")?"```":s.startsWith("~~~")?"~~~":null;
            if(marker!=null){if(fence==null)fence=marker;else if(fence.equals(marker))fence=null;continue;}
            if(fence==null)prose.append(line).append('\n');
        }
        var matcher=Pattern.compile("\\[chunk:(\\d+)\\]").matcher(prose);while(matcher.find())cited.add(matcher.group(1));
        List<Map<String,Object>> out=new ArrayList<>();
        for(var original:evidence) {
            var row=new LinkedHashMap<>(original);Object id=row.getOrDefault("chunk_id",row.get("chunkId"));
            Map<?,?> metadata=row.get("metadata") instanceof Map<?,?> m?m:Map.of();
            String source=Objects.toString(row.get("source"),""),content=Objects.toString(row.getOrDefault("content",row.get("preview")),"");
            Object section=metadata.containsKey("section_path")?metadata.get("section_path"):row.getOrDefault("sectionPath",List.of());
            String filename=source.substring(Math.max(source.lastIndexOf('/'),source.lastIndexOf('\\'))+1);int dot=filename.lastIndexOf('.');
            String title=Objects.toString(metadata.get("title"),Objects.toString(row.get("title"),dot>0?filename.substring(0,dot):filename));
            row.put("source",source);row.put("chunk_id",id);row.put("chunkId",id);row.put("content",content);row.put("metadata",metadata);
            row.put("title",title.isBlank()?"参考资料":title);row.put("preview",content.substring(0,content.offsetByCodePoints(0,Math.min(240,content.codePointCount(0,content.length())))));
            row.put("sectionPath",section instanceof String s?List.of(s):section==null?List.of():section);row.put("cited",cited.contains(String.valueOf(id)));out.add(row);
        }
        return out;
    }
}
