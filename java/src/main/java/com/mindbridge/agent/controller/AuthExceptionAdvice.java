package com.mindbridge.agent.controller;
import java.util.Map;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.server.ResponseStatusException;
/** Credential endpoints never echo submitted input or infrastructure errors. */
@RestControllerAdvice(assignableTypes=AuthController.class) @Order(Ordered.HIGHEST_PRECEDENCE)
public class AuthExceptionAdvice {
 @ExceptionHandler(Exception.class)
 public ResponseEntity<Map<String,String>> error(Exception e){int status=500;String detail="认证服务暂不可用";
  if(e instanceof ResponseStatusException r){status=r.getStatusCode().value();detail=r.getReason();}
  if(e instanceof org.springframework.web.server.ServerWebInputException||status==400){status=422;detail="请检查输入信息及密码确认";}
  if(e instanceof org.springframework.dao.DataIntegrityViolationException){status=409;detail="用户名或邮箱已被使用";}
  return ResponseEntity.status(status).header("Cache-Control","no-store").body(Map.of("detail",detail==null?"请求失败":detail));
 }
}
