package com.mindbridge.agent.security;
import org.springframework.web.server.*;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.ReactiveSecurityContextHolder;
import org.springframework.http.HttpMethod;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import java.util.Set;
public final class AuthWebFilters {
 private AuthWebFilters(){}
 public static WebFilter cookies(AccountAuthService auth){return (x,chain)->{
  var cookie=x.getRequest().getCookies().getFirst(AccountAuthService.COOKIE);
  if(Set.of("/api/auth/login","/api/auth/code","/api/auth/email-login","/api/auth/logout").contains(x.getRequest().getPath().value()))return chain.filter(x);
  if(x.getRequest().getHeaders().containsKey("Authorization")||cookie==null)return chain.filter(x);
  return Mono.fromCallable(()->auth.sessionUser(cookie.getValue())).subscribeOn(Schedulers.boundedElastic()).flatMap(u->chain.filter(x).contextWrite(ReactiveSecurityContextHolder.withAuthentication(new UsernamePasswordAuthenticationToken(u,null,u.getAuthorities()))));
 };}
 public static WebFilter guard(){return (x,chain)->{
  String path=x.getRequest().getPath().value();var method=x.getRequest().getMethod();boolean mutation=method!=HttpMethod.GET&&method!=HttpMethod.HEAD&&method!=HttpMethod.OPTIONS;
  boolean cookie=x.getRequest().getCookies().getFirst(AccountAuthService.COOKIE)!=null&&!x.getRequest().getHeaders().containsKey("Authorization");
  boolean browser=x.getRequest().getHeaders().containsKey("Origin")||x.getRequest().getHeaders().containsKey("Sec-Fetch-Site");
  try{if(mutation&&(path.startsWith("/api/auth/")||cookie||browser))AuthOrigin.check(x);}catch(ResponseStatusException e){return Mono.error(e);}
  return x.getPrincipal().flatMap(p->{if(p instanceof org.springframework.security.core.Authentication a&&a.getPrincipal() instanceof CurrentUser u&&u.isRequiresSetup()&&!Set.of("/api/profile","/api/auth/setup","/api/auth/logout").contains(path))return Mono.<Boolean>error(AccountAuthService.error(403,"请先设置用户名和密码"));return Mono.just(true);}).defaultIfEmpty(true).flatMap(ignored->chain.filter(x));
 };}
}
