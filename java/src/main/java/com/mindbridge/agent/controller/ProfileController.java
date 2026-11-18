package com.mindbridge.agent.controller;

import com.mindbridge.agent.security.CurrentUser;
import java.util.Map;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/profile")
/**
 * 当前登录账号信息接口。
 *
 * <p>前端通过角色判断展示学生聊天界面还是管理员后台。</p>
 */
public class ProfileController {

    private final com.mindbridge.agent.security.AccountAuthService auth;
    public ProfileController(com.mindbridge.agent.security.AccountAuthService auth) { this.auth = auth; }
    @GetMapping
    public reactor.core.publisher.Mono<Map<String, Object>> profile(@AuthenticationPrincipal CurrentUser currentUser) {
        return reactor.core.publisher.Mono.fromCallable(() -> auth.profile(currentUser.getId()))
                .subscribeOn(reactor.core.scheduler.Schedulers.boundedElastic());
    }
}
