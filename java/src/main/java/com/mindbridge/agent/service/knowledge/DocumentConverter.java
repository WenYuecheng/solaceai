package com.mindbridge.agent.service.knowledge;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

public final class DocumentConverter {
    private static final Pattern CID=Pattern.compile("\\(cid:\\d+\\)");
    private final ObjectMapper mapper;
    public record Converted(String source,String markdown,@com.fasterxml.jackson.annotation.JsonProperty("content_hash") String contentHash,String parser,@com.fasterxml.jackson.annotation.JsonProperty("parser_version") String parserVersion,JsonNode pages,@com.fasterxml.jackson.annotation.JsonProperty("structured_chunks") List<JsonNode> structuredChunks) {}
    public DocumentConverter(ObjectMapper mapper){this.mapper=mapper;}
    public Converted convertMetadata(String filename,byte[] data) {
        validateFilename(filename);
        if(data.length==0 || data.length>10*1024*1024) throw new IllegalArgumentException("Invalid document size");
        try {
            JsonNode root=mapper.readTree(data); String source=root.path("source").asText(""); validateFilename(source);
            JsonNode markdown=root.path("markdown"); String hash=root.path("content_hash").asText("");
            if (!markdown.isTextual() || !hash.matches("[a-f0-9]{64}")) throw new IllegalArgumentException("Invalid metadata");
            validateQuality(markdown.asText());
            JsonNode pages=root.path("pages"); if (!pages.isMissingNode() && !pages.isArray() && !pages.isObject()) throw new IllegalArgumentException("Invalid pages");
            JsonNode chunks=root.path("structured_chunks"); if (!chunks.isMissingNode() && !chunks.isArray()) throw new IllegalArgumentException("Invalid chunks");
            List<JsonNode> structured=new ArrayList<>(); if (chunks.isArray()) chunks.forEach(node->{validateChunk(node,source,hash);structured.add(node);});
            return new Converted(source,markdown.asText(),hash,root.path("parser").asText("converted"),root.path("parser_version").asText("1"),pages,List.copyOf(structured));
        } catch(IOException e){throw new IllegalArgumentException("Invalid metadata",e);}
    }
    public Converted convertExternal(String filename, byte[] data, String python, String modulePath) {
        validateFilename(filename);
        if (data.length==0 || data.length>10*1024*1024) throw new IllegalArgumentException("Invalid document size");
        if (python==null || python.isBlank() || modulePath==null || modulePath.isBlank()) throw new IllegalStateException("Isolated document parser is not configured");
        java.nio.file.Path dir=null;
        try {
            dir=java.nio.file.Files.createTempDirectory("mindbridge-docling-");
            var inputDir=java.nio.file.Files.createDirectory(dir.resolve("input"));
            var input=inputDir.resolve(filename); var output=dir.resolve("converted.md");
            java.nio.file.Files.write(input,data);
            ProcessBuilder builder=new ProcessBuilder(python,"-m","app.services.document_conversion",input.toString(),"--output",output.toString(),"--isolated");
            builder.environment().clear(); builder.environment().put("PYTHONPATH",modulePath);
            builder.environment().put("HF_HUB_OFFLINE","1"); builder.environment().put("TRANSFORMERS_OFFLINE","1");
            builder.environment().put("TOKENIZERS_PARALLELISM","false"); builder.environment().put("SOLACEAI_DOCUMENT_OPTIONS","{\"knowledge_quality_threshold\":0.8}");
            builder.redirectOutput(ProcessBuilder.Redirect.DISCARD); builder.redirectError(ProcessBuilder.Redirect.DISCARD);
            Process process=builder.start();
            if (!process.waitFor(210,java.util.concurrent.TimeUnit.SECONDS)) { process.destroyForcibly(); throw new IllegalStateException("Document parser timed out"); }
            if(process.exitValue()!=0) throw new IllegalStateException("Document parser failed");
            var metadata=dir.resolve("converted.metadata.json");
            if (!java.nio.file.Files.isRegularFile(metadata) || java.nio.file.Files.size(metadata)>64L*1024*1024) throw new IllegalStateException("Invalid parser output");
            // Converter output can be larger than an uploaded metadata file, but is still bounded.
            byte[] outputBytes=java.nio.file.Files.readAllBytes(metadata);
            Converted result=convertMetadataBounded(filename,outputBytes);
            String digest=java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(data));
            if(!result.source().equals(filename) || !result.contentHash().equals(digest)) throw new IllegalStateException("Parser output integrity check failed");
            return result;
        } catch(InterruptedException e){Thread.currentThread().interrupt(); throw new IllegalStateException("Parser interrupted",e);}
        catch(IOException | java.security.NoSuchAlgorithmException e){throw new IllegalStateException("Document parser failed",e);}
        finally { if(dir!=null) try(var paths=java.nio.file.Files.walk(dir)){paths.sorted(java.util.Comparator.reverseOrder()).forEach(path->{try{java.nio.file.Files.deleteIfExists(path);}catch(IOException ignored){}});}catch(IOException ignored){} }
    }
    private Converted convertMetadataBounded(String filename,byte[] data) {
        // Isolated output has a separate maximum from the 10 MB upload limit.
        try {
            JsonNode root=mapper.readTree(data); String source=root.path("source").asText(""); validateFilename(source);
            String markdown=root.path("markdown").asText(""); String hash=root.path("content_hash").asText("");
            if(!hash.matches("[a-f0-9]{64}")) throw new IllegalArgumentException("Invalid metadata hash"); validateQuality(markdown);
            JsonNode pages=root.path("pages"), chunks=root.path("structured_chunks");
            if(!pages.isMissingNode() && !pages.isObject() && !pages.isArray() || !chunks.isMissingNode() && !chunks.isArray()) throw new IllegalArgumentException("Invalid metadata structure");
            List<JsonNode> rows=new ArrayList<>(); if(chunks.isArray()) chunks.forEach(node->{validateChunk(node,source,hash);rows.add(node);});
            return new Converted(source,markdown,hash,root.path("parser").asText("converted"),root.path("parser_version").asText("1"),pages,List.copyOf(rows));
        }catch(IOException e){throw new IllegalArgumentException("Invalid parser output",e);}
    }
    private static void validateChunk(JsonNode node,String source,String hash) {
        if(!node.isObject() || !node.path("content").isTextual()) throw new IllegalArgumentException("Invalid structured chunk");
        validateQuality(node.path("content").asText());
        if(node.has("source") && !source.equals(node.path("source").asText())) throw new IllegalArgumentException("Chunk source mismatch");
        if(node.has("content_hash") && !hash.equals(node.path("content_hash").asText())) throw new IllegalArgumentException("Chunk hash mismatch");
        for(String key:List.of("section_path","captions","element_types")) if(node.has(key)) {
            if(!node.path(key).isArray()) throw new IllegalArgumentException("Invalid chunk metadata");
            for(JsonNode item:node.path(key)) if(!item.isTextual()) throw new IllegalArgumentException("Invalid chunk metadata");
        }
        if(node.has("parent_content") && !node.path("parent_content").isTextual()) throw new IllegalArgumentException("Invalid parent content");
        if(node.has("page") && !node.path("page").isNull() && (!node.path("page").isIntegralNumber() || node.path("page").asInt()<0)) throw new IllegalArgumentException("Invalid page");
    }
    static void validateFilename(String name) {
        if(name==null || name.isBlank() || name.length()>180 || name.equals(".") || name.equals("..") || name.indexOf('\0')>=0 || name.contains("/") || name.contains("\\")) throw new IllegalArgumentException("Invalid filename");
    }
    public static void validateQuality(String text) {
        if(text==null || text.isBlank()) throw new IllegalArgumentException("Empty document");
        var matcher=CID.matcher(text); int bad=0; while(matcher.find()) bad+=matcher.end()-matcher.start();
        String cleaned=CID.matcher(text).replaceAll(""); int total=bad, valid=0;
        for (int cp:cleaned.codePoints().toArray()) {
            if(Character.isWhitespace(cp)) continue; total++;
            if ((cp>=0x4e00 && cp<=0x9fff)||(cp>=0x3400 && cp<=0x4dbf)||(cp>=0xf900 && cp<=0xfaff)
                || (cp>=0x3000 && cp<=0x303f)||(cp>=0xff00 && cp<=0xffef)
                || (cp>='a' && cp<='z')||(cp>='A' && cp<='Z')||(cp>='0' && cp<='9')
                || ".,;:!?'\"()[]{}-_/<=>+*#@&%$^~`|\\".indexOf(cp)>=0) valid++;
        }
        if(total==0 || Math.round((valid/(double)total)*10000.0)/10000.0 < 0.8) throw new IllegalArgumentException("Document quality is below 0.8");
    }
}
