package com.mindbridge.agent.controller;
import com.mindbridge.agent.security.*;
import java.util.*;
import java.util.function.Supplier;
import org.springframework.web.bind.annotation.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
@RestController @RequestMapping("/api/auth")
public class AuthController {
 private final AccountAuthService auth;
 public AuthController(AccountAuthService auth){this.auth=auth;}
 private <T> Mono<T> run(Supplier<T> action){return Mono.fromSupplier(action).subscribeOn(Schedulers.boundedElastic());}
 private void fields(Map<String,String> body,String... allowed){if(!Set.of(allowed).containsAll(body.keySet()))throw AccountAuthService.error(422,"请检查输入信息及密码确认");}
 @PostMapping("/login") public Mono<?> login(@RequestBody Map<String,String> b,ServerWebExchange x){fields(b,"username","password");return run(()->auth.login(b.get("username"),b.get("password"),x));}
 @PostMapping("/code") public Mono<?> code(@RequestBody Map<String,String> b,ServerWebExchange x){fields(b,"email","purpose");if(b.containsKey("purpose")&&!"login".equals(b.get("purpose")))throw AccountAuthService.error(422,"请检查输入信息及密码确认");return run(()->auth.sendCode(b.get("email"),x));}
 @PostMapping("/email-login") public Mono<?> emailLogin(@RequestBody Map<String,String> b,ServerWebExchange x){fields(b,"email","code");checkCode(b);return run(()->auth.emailLogin(b.get("email"),b.get("code"),x));}
 @PostMapping("/logout") public Mono<?> logout(ServerWebExchange x){return run(()->auth.logout(x));}
 @PostMapping("/setup") public Mono<?> setup(@AuthenticationPrincipal CurrentUser u,@RequestBody Map<String,String> b,ServerWebExchange x){fields(b,"username","password","confirmPassword");return run(()->auth.setup(u.getId(),b.get("username"),b.get("password"),b.get("confirmPassword"),x));}
 @PostMapping("/password") public Mono<?> password(@AuthenticationPrincipal CurrentUser u,@RequestBody Map<String,String> b,ServerWebExchange x){fields(b,"currentPassword","password","confirmPassword");return run(()->auth.changePassword(u.getId(),b.get("currentPassword"),b.get("password"),b.get("confirmPassword"),x));}
 @PostMapping("/email/code") public Mono<?> bindCode(@AuthenticationPrincipal CurrentUser u,@RequestBody Map<String,String> b,ServerWebExchange x){fields(b,"email","currentPassword");return run(()->auth.bindCode(u.getId(),b.get("email"),b.get("currentPassword"),x));}
 @PostMapping("/email") public Mono<?> bind(@AuthenticationPrincipal CurrentUser u,@RequestBody Map<String,String> b,ServerWebExchange x){fields(b,"email","currentPassword","code");checkCode(b);return run(()->auth.bind(u.getId(),b.get("email"),b.get("currentPassword"),b.get("code"),x));}
 private void checkCode(Map<String,String> b){if(b.get("code")==null||!b.get("code").matches("[0-9]{6}"))throw AccountAuthService.error(422,"请检查输入信息及密码确认");}
}
