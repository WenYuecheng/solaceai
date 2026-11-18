package com.mindbridge.agent.controller;

import java.net.URI;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
/**
 * 静态首页入口。
 *
 * <p>直接访问根路径时返回前端页面，方便本地演示不用单独启动前端服务。</p>
 */
public class HomeController {

    @GetMapping("/")
    public ResponseEntity<Void> index() {
        return ResponseEntity.status(HttpStatus.SEE_OTHER).location(URI.create("/index.html")).build();
    }
}
