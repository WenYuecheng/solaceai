package com.mindbridge.agent.security;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;
public final class AuthOrigin {
 private AuthOrigin(){}
 public static void check(ServerWebExchange exchange){
  var request=exchange.getRequest(); var h=request.getHeaders(); var uri=request.getURI();
  String expected=uri.getScheme()+"://"+uri.getRawAuthority(); String origin=h.getOrigin();
  if ((origin!=null&&!origin.equals(expected)) || (origin==null&&!"SolaceAI".equals(h.getFirst("X-Requested-With"))) || "cross-site".equals(h.getFirst("Sec-Fetch-Site"))) throw new ResponseStatusException(HttpStatus.FORBIDDEN,"请求来源无效");
 }
}
