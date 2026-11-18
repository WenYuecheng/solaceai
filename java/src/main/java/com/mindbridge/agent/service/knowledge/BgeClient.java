package com.mindbridge.agent.service.knowledge;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;

/** Client for the isolated loopback BGE worker; response validation is deliberately strict. */
public final class BgeClient {
    private final URI base;
    private final ObjectMapper mapper;
    private final HttpClient http;
    private final Duration timeout;
    public record Embeddings(List<List<Double>> dense, List<Map<Integer,Double>> sparse) {}
    public record Rank(int index, double score) {}
    public BgeClient(String url, Duration timeout, ObjectMapper mapper) {
        URI uri = URI.create(url);
        String host = uri.getHost();
        if (!"http".equals(uri.getScheme()) || host == null || uri.getUserInfo() != null || uri.getQuery() != null || uri.getFragment() != null
                || !(host.equalsIgnoreCase("localhost") || host.equals("127.0.0.1") || host.equals("[::1]") || host.equals("::1")))
            throw new IllegalArgumentException("BGE endpoint must be loopback HTTP");
        this.base = uri; this.timeout = timeout; this.mapper = mapper;
        // The worker serves plaintext HTTP/1.1. An h2c upgrade can discard the first POST body.
        this.http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(5)).followRedirects(HttpClient.Redirect.NEVER).build();
    }
    public JsonNode health() { return call("/health", null); }
    public Embeddings embed(List<String> texts) { return parseEmbeddings(call("/embed", Map.of("texts",texts)), texts.size()); }
    public List<Rank> rerank(String query, List<String> texts) { return parseRerank(call("/rerank",Map.of("query",query,"texts",texts)),texts.size()); }
    private JsonNode call(String path, Object body) {
        try {
            HttpRequest.Builder request = HttpRequest.newBuilder(base.resolve(path)).timeout(timeout);
            if (body == null) request.GET();
            else request.header("Content-Type","application/json").POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)));
            HttpResponse<String> response = http.send(request.build(),HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) throw new IllegalStateException("BGE worker HTTP " + response.statusCode());
            return mapper.readTree(response.body());
        } catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IllegalStateException("BGE request interrupted",e); }
        catch (Exception e) { throw new IllegalStateException("BGE worker request failed",e); }
    }
    public static Embeddings parseEmbeddings(JsonNode root, int count) {
        JsonNode dense = root.path("dense"), sparse = root.path("sparse");
        if (!dense.isArray() || !sparse.isArray() || dense.size()!=count || sparse.size()!=count) throw new IllegalArgumentException("Embedding batch size mismatch");
        List<List<Double>> vectors = new ArrayList<>(); List<Map<Integer,Double>> weights = new ArrayList<>();
        for (int i=0;i<count;i++) {
            JsonNode row=dense.get(i), terms=sparse.get(i);
            if (!row.isArray() || row.size()!=1024 || !terms.isObject() || terms.isEmpty()) throw new IllegalArgumentException("Invalid BGE embedding");
            List<Double> vector=new ArrayList<>(1024);
            row.forEach(n -> { if (!n.isNumber() || !Double.isFinite(n.doubleValue())) throw new IllegalArgumentException("Invalid dense weight"); vector.add(n.doubleValue()); });
            Map<Integer,Double> clean=new LinkedHashMap<>();
            terms.fields().forEachRemaining(entry -> {
                int key; try { key=Integer.parseInt(entry.getKey()); } catch (NumberFormatException e) { throw new IllegalArgumentException("Invalid sparse index",e); }
                JsonNode value=entry.getValue();
                if (key<0 || !value.isNumber() || !Double.isFinite(value.doubleValue()) || value.doubleValue()<0 || clean.putIfAbsent(key,value.doubleValue())!=null) throw new IllegalArgumentException("Invalid sparse weight");
            });
            vectors.add(List.copyOf(vector)); weights.add(Map.copyOf(clean));
        }
        return new Embeddings(List.copyOf(vectors),List.copyOf(weights));
    }
    public static List<Rank> parseRerank(JsonNode root, int count) {
        JsonNode rows=root.has("results") ? root.path("results") : root.path("data");
        if (!rows.isArray() || rows.size()!=count || count==0) throw new IllegalArgumentException("Incomplete rerank response");
        Set<Integer> seen=new HashSet<>(); List<Rank> result=new ArrayList<>();
        for (JsonNode row:rows) {
            JsonNode idx=row.path("index"), score=row.has("relevance_score") ? row.path("relevance_score") : row.path("score");
            if (!idx.isIntegralNumber() || !idx.canConvertToInt() || idx.intValue()<0 || idx.intValue()>=count || !seen.add(idx.intValue()) || !score.isNumber() || !Double.isFinite(score.doubleValue())) throw new IllegalArgumentException("Invalid rerank response");
            result.add(new Rank(idx.intValue(),score.doubleValue()));
        }
        result.sort(Comparator.comparingDouble(Rank::score).reversed()); return List.copyOf(result);
    }
}
