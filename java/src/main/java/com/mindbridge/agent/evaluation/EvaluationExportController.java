package com.mindbridge.agent.evaluation;

import java.util.Map;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

/** SecurityConfig protects /api/admin/** with ROLE_ADMIN. Exports contain evaluation text/PII. */
@RestController
@RequestMapping("/api/admin/evaluation")
public class EvaluationExportController {
    private final EvaluationExportService exports;
    public EvaluationExportController(EvaluationExportService exports){this.exports=exports;}
    @GetMapping("/runs/{runId}")
    public Mono<Map<String,Object>> export(@PathVariable String runId){return Mono.fromCallable(()->exports.export(runId)).subscribeOn(Schedulers.boundedElastic());}
}
