package com.mindbridge.agent.controller;

import com.mindbridge.agent.service.CaseWorkflow;
import com.mindbridge.agent.security.CurrentUser;
import java.util.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

@RestController
@RequestMapping("/api/admin/cases")
public class CaseController {
    private final CaseWorkflow cases;
    public CaseController(CaseWorkflow cases){this.cases=cases;}
    private static <T> Mono<T> blocking(java.util.concurrent.Callable<T> call){return Mono.fromCallable(call).subscribeOn(Schedulers.boundedElastic());}
    @GetMapping public Mono<List<Map<String,Object>>> list(){return blocking(cases::list);}
    @GetMapping("/{id}/notes") public Mono<List<Map<String,Object>>> notes(@PathVariable long id){return blocking(()->cases.notes(id));}
    @PostMapping("/{id}/acknowledge") public Mono<Map<String,Object>> acknowledge(@PathVariable long id,@AuthenticationPrincipal CurrentUser actor){return blocking(()->cases.apply(id,actor.getUsername(),"acknowledge",""));}
    @PostMapping("/{id}/notes") public Mono<Map<String,Object>> note(@PathVariable long id,@AuthenticationPrincipal CurrentUser actor,@RequestBody Map<String,Object> body){return blocking(()->cases.apply(id,actor.getUsername(),"notes",text(body,"note")));}
    @PostMapping("/{id}/close") public Mono<Map<String,Object>> close(@PathVariable long id,@AuthenticationPrincipal CurrentUser actor,@RequestBody Map<String,Object> body){return blocking(()->cases.apply(id,actor.getUsername(),"close",text(body,"reason")));}
    private static String text(Map<String,Object> body,String field){if(!body.keySet().equals(Set.of(field))||!(body.get(field) instanceof String value))throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY,"请求字段无效");return value;}
}
