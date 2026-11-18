package com.mindbridge.agent.runtime;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Component;

/** Immutable catalog of the exact shared SolaceAI skill assets. */
@Component
public class SkillLibrary {
    private final Map<String,Map<String,String>> skills=new TreeMap<>();
    private final Map<String,String> prompts;
    public SkillLibrary(RuntimeProperties properties,ObjectMapper json){try{
        var resources=new PathMatchingResourcePatternResolver();
        try(var in=resources.getResource("classpath:runtime/prompts.json").getInputStream()){prompts=json.readValue(in,new com.fasterxml.jackson.core.type.TypeReference<>(){});}
        if(properties.getSkillsPath().isBlank()){
            for(var resource:resources.getResources("classpath*:runtime/skills/*/SKILL.md"))try(var in=resource.getInputStream()){add(new String(in.readAllBytes(),StandardCharsets.UTF_8));}
        }else{Path root=Path.of(properties.getSkillsPath()).toAbsolutePath().normalize();try(var paths=Files.list(root)){for(Path p:paths.filter(Files::isDirectory).sorted().toList()){Path file=p.resolve("SKILL.md");if(Files.isRegularFile(file))add(Files.readString(file));}}}
    }catch(Exception e){throw new IllegalStateException("Shared prompt/skill assets unavailable",e);}}
    private void add(String text){if(!text.startsWith("---\n"))throw new IllegalArgumentException("Invalid skill frontmatter");int end=text.indexOf("\n---",4);if(end<0)throw new IllegalArgumentException("Invalid skill");Map<String,String> meta=new HashMap<>();for(String line:text.substring(4,end).split("\n")){int colon=line.indexOf(':');if(colon>0)meta.put(line.substring(0,colon).trim(),line.substring(colon+1).trim().replaceAll("^[\"']|[\"']$",""));}String name=meta.get("name");if(name==null||!name.matches("[A-Za-z0-9_-]{1,100}"))throw new IllegalArgumentException("Invalid skill name");String body=text.substring(end+4).trim();skills.put(name,Map.of("name",name,"description",meta.getOrDefault("description",""),"when_to_use",meta.getOrDefault("when_to_use",meta.getOrDefault("description","")),"body",body,"version",TurnLocks.hash(body).substring(0,12)));}
    public String prompt(String name){String value=prompts.get(name);if(value==null)throw new IllegalArgumentException("Unknown prompt");return value;}
    public List<Map<String,Object>> catalog(){return skills.values().stream().map(row->{Map<String,Object> view=new LinkedHashMap<>(row);view.remove("body");return view;}).toList();}
    public String load(String name){var row=skills.get(name);if(row==null)throw new IllegalArgumentException("Unknown skill");String body=row.get("body");return body.substring(0,Math.min(12000,body.length()));}
    public boolean contains(String name){return skills.containsKey(name);}
}
