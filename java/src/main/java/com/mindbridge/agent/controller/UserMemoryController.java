package com.mindbridge.agent.controller;

import com.mindbridge.agent.dto.UserMemoryItemResponse;
import com.mindbridge.agent.domain.UserMemoryType;
import com.mindbridge.agent.runtime.RuntimeStore;
import com.mindbridge.agent.security.CurrentUser;
import java.time.Instant;
import java.util.List;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

/** Legacy response adapter over the same explicit memory store used by the graph. */
@RestController
@RequestMapping("/api/profile/memory")
public class UserMemoryController {
    private final RuntimeStore memory;
    public UserMemoryController(RuntimeStore memory){this.memory=memory;}
    @GetMapping public Mono<List<UserMemoryItemResponse>> memories(@AuthenticationPrincipal CurrentUser user){return Mono.fromCallable(()->memory.memories(user.getId()).stream().map(row->{Instant time=Instant.parse(row.get("created_at").toString());String content=row.get("content").toString();return new UserMemoryItemResponse(((Number)row.get("id")).longValue(),UserMemoryType.PREFERENCE,content,content,1,time,time,time);}).toList()).subscribeOn(Schedulers.boundedElastic());}
    @DeleteMapping("/{memoryId}") public Mono<Void> delete(@AuthenticationPrincipal CurrentUser user,@PathVariable long memoryId){return Mono.<Void>fromRunnable(()->{if(!memory.deleteMemory(user.getId(),memoryId))throw new ResponseStatusException(HttpStatus.NOT_FOUND,"记忆不存在");}).subscribeOn(Schedulers.boundedElastic());}
}
