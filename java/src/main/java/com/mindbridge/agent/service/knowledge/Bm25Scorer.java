package com.mindbridge.agent.service.knowledge;

import com.mindbridge.agent.domain.KnowledgeChunk;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Lightweight BM25 scorer for keyword retrieval.
 */
public class Bm25Scorer {

    private static final double K1 = 1.5;
    private static final double B = 0.75;

    public List<SearchResult> rank(String query, List<KnowledgeChunk> chunks, int limit) {
        if (limit <= 0 || query == null || query.isBlank() || chunks.isEmpty()) {
            return List.of();
        }
        List<String> queryTerms = tokenize(query);
        if (queryTerms.isEmpty()) {
            return List.of();
        }
        CorpusStats stats = buildCorpusStats(chunks);
        return chunks.stream()
                .map(chunk -> new SearchResult(
                        chunk.getId(),
                        chunk.getSource(),
                        chunk.getContent(),
                        score(queryTerms, stats, chunk)))
                .filter(result -> result.score() > 0.0)
                .sorted(Comparator.comparingDouble(SearchResult::score).reversed())
                .limit(limit)
                .toList();
    }

    private CorpusStats buildCorpusStats(List<KnowledgeChunk> chunks) {
        Map<Long, Map<String, Integer>> termFrequencies = new HashMap<>();
        Map<Long, Integer> lengths = new HashMap<>();
        Map<String, Integer> documentFrequencies = new HashMap<>();
        int totalLength = 0;

        for (KnowledgeChunk chunk : chunks) {
            List<String> terms = tokenize(chunk.getContent());
            Map<String, Integer> frequencies = new HashMap<>();
            for (String term : terms) {
                frequencies.merge(term, 1, Integer::sum);
            }
            Long key = chunk.getId();
            termFrequencies.put(key, frequencies);
            lengths.put(key, terms.size());
            totalLength += terms.size();

            Set<String> uniqueTerms = new HashSet<>(terms);
            for (String term : uniqueTerms) {
                documentFrequencies.merge(term, 1, Integer::sum);
            }
        }

        double averageLength = chunks.isEmpty() ? 0.0 : totalLength / (double) chunks.size();
        return new CorpusStats(chunks.size(), averageLength, termFrequencies, lengths, documentFrequencies);
    }

    private double score(List<String> queryTerms, CorpusStats stats, KnowledgeChunk chunk) {
        Map<String, Integer> frequencies = stats.termFrequencies().getOrDefault(chunk.getId(), Map.of());
        int length = stats.lengths().getOrDefault(chunk.getId(), 0);
        if (frequencies.isEmpty() || length == 0 || stats.averageLength() == 0.0) {
            return 0.0;
        }

        double score = 0.0;
        Map<String,Integer> queryFrequencies = new HashMap<>();
        queryTerms.forEach(term -> queryFrequencies.merge(term,1,Integer::sum));
        for (String term : queryFrequencies.keySet()) {
            int tf = frequencies.getOrDefault(term, 0);
            if (tf == 0) {
                continue;
            }
            int df = stats.documentFrequencies().getOrDefault(term, 0);
            double idf = Math.log(1.0 + (stats.documentCount() - df + 0.5) / (df + 0.5));
            double denominator = tf + K1 * (1.0 - B + B * length / stats.averageLength());
            score += idf * (1.0 + Math.log(queryFrequencies.get(term))) * (tf * (K1 + 1.0)) / denominator;
        }
        return score;
    }

    static List<String> tokenize(String text) {
        if(text==null || text.isBlank()) return List.of();
        String normalized=text.toLowerCase(Locale.ROOT);
        List<String> terms=new ArrayList<>();
        var matcher=java.util.regex.Pattern.compile("[a-z0-9_]+|[\\u4e00-\\u9fff]").matcher(normalized);
        while(matcher.find()) terms.add(matcher.group());
        StringBuilder compact=new StringBuilder();
        normalized.chars().filter(c->c>=0x4e00 && c<=0x9fff).forEach(c->compact.append((char)c));
        for(int i=0;i+1<compact.length();i++) terms.add(compact.substring(i,i+2));
        return terms;
    }

    private record CorpusStats(
            int documentCount,
            double averageLength,
            Map<Long, Map<String, Integer>> termFrequencies,
            Map<Long, Integer> lengths,
            Map<String, Integer> documentFrequencies
    ) {
    }
}
