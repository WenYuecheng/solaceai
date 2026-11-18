package com.mindbridge.agent.runtime;

import java.util.*;
import java.util.function.Function;

/** Strict JSON-schema tool registry; caller identity is closed over, never supplied by the model. */
public final class GovernedTools {
    private final Map<String,Map<String,Object>> schemas=new LinkedHashMap<>();
    private final Map<String,Function<Map<String,Object>,Object>> handlers=new HashMap<>();
    public void register(String name,String description,Map<String,Object> parameters,Function<Map<String,Object>,Object> handler){if(schemas.containsKey(name))throw new IllegalArgumentException("Duplicate tool registration");schemas.put(name,Map.of("description",description,"parameters",parameters));handlers.put(name,handler);}
    public Map<String,Map<String,Object>> schemas(){return Collections.unmodifiableMap(schemas);}
    @SuppressWarnings("unchecked") public Object call(String name,Map<String,Object> arguments){var spec=schemas.get(name);if(spec==null)throw new SecurityException("Tool not authorized");Map<String,Object> schema=(Map<String,Object>)spec.get("parameters");Map<String,Map<String,Object>> props=(Map<String,Map<String,Object>>)schema.get("properties");if(!props.keySet().containsAll(arguments.keySet()))throw new IllegalArgumentException("Invalid tool arguments: unknown field");for(String required:(List<String>)schema.get("required"))if(!arguments.containsKey(required))throw new IllegalArgumentException("Invalid tool arguments: missing field");for(var entry:arguments.entrySet()){var rule=props.get(entry.getKey());Object value=entry.getValue();String type=(String)rule.get("type");if(type.equals("string")){if(!(value instanceof String s)||s.length()<((Number)rule.getOrDefault("minLength",0)).intValue()||s.length()>((Number)rule.getOrDefault("maxLength",12000)).intValue())throw new IllegalArgumentException("Invalid tool string");}else if(type.equals("integer")){if(!(value instanceof Integer||value instanceof Long))throw new IllegalArgumentException("Invalid tool integer");}else throw new IllegalArgumentException("Unsupported schema");}return handlers.get(name).apply(arguments);}
    public static Map<String,Object> object(Map<String,Map<String,Object>> fields){return Map.of("type","object","properties",fields,"required",new ArrayList<>(fields.keySet()),"additionalProperties",false);}
    public static Map<String,Object> text(int max){return Map.of("type","string","minLength",1,"maxLength",max);}
    public static Map<String,Object> integer(){return Map.of("type","integer");}
}
