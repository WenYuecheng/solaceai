package com.mindbridge.agent.controller;

import com.mindbridge.agent.dto.AnswerFeedbackRequest;
import com.mindbridge.agent.runtime.RuntimeStore;
import com.mindbridge.agent.security.CurrentUser;
import com.mindbridge.agent.service.ConversationService;
import jakarta.validation.Valid;
import java.util.*;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

@RestController
public class ConversationController {
    private final ConversationService conversations;private final RuntimeStore memory;
    public ConversationController(ConversationService conversations,RuntimeStore memory){this.conversations=conversations;this.memory=memory;}
    private static <T> Mono<T> blocking(java.util.concurrent.Callable<T> task){return Mono.fromCallable(task).subscribeOn(Schedulers.boundedElastic());}
    @GetMapping("/api/sessions") public Mono<Map<String,Object>> list(@AuthenticationPrincipal CurrentUser user,@RequestParam(defaultValue="50") int limit,@RequestParam(defaultValue="0") int offset){return blocking(()->conversations.list(user.getId(),limit,offset));}
    @GetMapping("/api/sessions/{id}") public Mono<Map<String,Object>> detail(@AuthenticationPrincipal CurrentUser user,@PathVariable String id){return blocking(()->conversations.detail(user.getId(),id));}
    @DeleteMapping("/api/sessions/{id}") public Mono<Map<String,Boolean>> hide(@AuthenticationPrincipal CurrentUser user,@PathVariable String id){return blocking(()->conversations.hide(user.getId(),id));}
    @PostMapping("/api/feedback") public Mono<Map<String,Object>> feedback(@AuthenticationPrincipal CurrentUser user,@Valid @RequestBody AnswerFeedbackRequest request){return blocking(()->conversations.saveFeedback(user.getId(),request));}
    @GetMapping("/api/evidence/{run}/{chunk}") public Mono<Map<String,Object>> evidence(@AuthenticationPrincipal CurrentUser user,@PathVariable String run,@PathVariable long chunk){return blocking(()->conversations.evidence(user.getId(),run,chunk));}
    @GetMapping("/api/memories") public Mono<List<Map<String,Object>>> memories(@AuthenticationPrincipal CurrentUser user){return blocking(()->memory.memories(user.getId()));}
    @DeleteMapping("/api/memories/{id}") public Mono<Map<String,Boolean>> forget(@AuthenticationPrincipal CurrentUser user,@PathVariable long id){return blocking(()->{if(!memory.deleteMemory(user.getId(),id))throw new ResponseStatusException(HttpStatus.NOT_FOUND,"记忆不存在");return Map.of("deleted",true);});}
}
