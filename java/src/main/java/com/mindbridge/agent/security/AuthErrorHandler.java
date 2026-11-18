package com.mindbridge.agent.security;
import java.nio.charset.StandardCharsets;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.server.*;
import reactor.core.publisher.Mono;
/** Keep auth errors free of submitted passwords, codes and mail transport details. */
@Component @Order(-2)
public class AuthErrorHandler implements WebExceptionHandler {
 private final com.fasterxml.jackson.databind.ObjectMapper json;
 public AuthErrorHandler(com.fasterxml.jackson.databind.ObjectMapper json){this.json=json;}
 public Mono<Void> handle(ServerWebExchange x,Throwable e){
  boolean auth=x.getRequest().getPath().value().startsWith("/api/auth/");
  if(!(e instanceof ResponseStatusException)&&!auth)return Mono.error(e);
  if(x.getResponse().isCommitted())return Mono.error(e);
  int status=e instanceof ResponseStatusException r?r.getStatusCode().value():500;
  String detail=e instanceof ResponseStatusException r?r.getReason():"认证服务暂不可用";
  if(auth&&status==400){status=422;detail="请检查输入信息及密码确认";}
  if(e instanceof org.springframework.dao.DataIntegrityViolationException){status=409;detail="用户名或邮箱已被使用";}
  x.getResponse().setStatusCode(org.springframework.http.HttpStatusCode.valueOf(status));x.getResponse().getHeaders().setContentType(org.springframework.http.MediaType.APPLICATION_JSON);x.getResponse().getHeaders().setCacheControl("no-store");
  try{return x.getResponse().writeWith(Mono.just(x.getResponse().bufferFactory().wrap(json.writeValueAsString(java.util.Map.of("detail",detail==null?"请求失败":detail)).getBytes(StandardCharsets.UTF_8))));}catch(Exception ignored){return Mono.error(e);}
 }
}
