package com.mindbridge.agent.config;

import com.mindbridge.agent.security.CurrentUserDetailsService;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.authentication.ReactiveAuthenticationManager;
import org.springframework.security.authentication.UserDetailsRepositoryReactiveAuthenticationManager;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.server.SecurityWebFilterChain;
import org.springframework.security.config.web.server.ServerHttpSecurity;

@Configuration
@EnableWebFluxSecurity
/**
 * WebFlux 安全配置。
 *
 * <p>支持不透明 Cookie 会话和 HTTP Basic；管理员接口要求 ADMIN 角色，
 * 普通 API 要求已登录用户。</p>
 */
public class SecurityConfig {

    @Bean
    public SecurityWebFilterChain securityFilterChain(
            ServerHttpSecurity http,
            ReactiveAuthenticationManager authenticationManager,
            com.mindbridge.agent.security.AccountAuthService authService,
            @org.springframework.beans.factory.annotation.Value("${spring.ai.mcp.server.sse-endpoint:/sse}") String mcpSse,
            @org.springframework.beans.factory.annotation.Value("${spring.ai.mcp.server.sse-message-endpoint:/mcp/messages}") String mcpMessages
    ) {
        return http.csrf(ServerHttpSecurity.CsrfSpec::disable)
                .authenticationManager(authenticationManager)
                .httpBasic(basic -> basic.authenticationEntryPoint((exchange, exception) -> reactor.core.publisher.Mono.fromRunnable(() -> authService.basicFailure(exchange)).subscribeOn(reactor.core.scheduler.Schedulers.boundedElastic()).then(authError(exchange, 401, "Bad credentials"))))
                .exceptionHandling(errors -> errors
                        .authenticationEntryPoint((exchange, exception) -> authError(exchange, 401, "Missing Basic authorization"))
                        .accessDeniedHandler((exchange, exception) -> authError(exchange, 403, "Admin role required")))
                .addFilterAt(com.mindbridge.agent.security.AuthWebFilters.cookies(authService), org.springframework.security.config.web.server.SecurityWebFiltersOrder.AUTHENTICATION)
                .addFilterBefore(com.mindbridge.agent.security.AuthWebFilters.guard(), org.springframework.security.config.web.server.SecurityWebFiltersOrder.AUTHORIZATION)
                .formLogin(ServerHttpSecurity.FormLoginSpec::disable)
                .authorizeExchange(auth -> auth
                        .pathMatchers("/actuator/health", "/h2-console/**").permitAll()
                        .pathMatchers("/api/auth/login", "/api/auth/code", "/api/auth/email-login", "/api/auth/logout").permitAll()
                        .pathMatchers("/api/reports/me").authenticated()
                        .pathMatchers(mcpSse, mcpMessages).hasRole("ADMIN")
                        .pathMatchers("/mcp", "/mcp/**", "/sse", "/sse/**").hasRole("ADMIN")
                        .pathMatchers("/api/admin/**").hasRole("ADMIN")
                        .pathMatchers("/api/reports/**").hasRole("ADMIN")
                        .pathMatchers("/api/**").authenticated()
                        .anyExchange().permitAll())
                .build();
    }

    private reactor.core.publisher.Mono<Void> authError(org.springframework.web.server.ServerWebExchange x, int status, String message) {
        x.getResponse().setStatusCode(org.springframework.http.HttpStatusCode.valueOf(status));
        x.getResponse().getHeaders().setContentType(org.springframework.http.MediaType.APPLICATION_JSON);
        return x.getResponse().writeWith(reactor.core.publisher.Mono.just(x.getResponse().bufferFactory().wrap(("{\"detail\":\"" + message + "\"}").getBytes(java.nio.charset.StandardCharsets.UTF_8))));
    }

    @Bean
    public ReactiveAuthenticationManager authenticationManager(
            CurrentUserDetailsService userDetailsService,
            PasswordEncoder passwordEncoder
    ) {
        UserDetailsRepositoryReactiveAuthenticationManager manager =
                new UserDetailsRepositoryReactiveAuthenticationManager(userDetailsService);
        manager.setPasswordEncoder(passwordEncoder);
        return manager;
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new com.mindbridge.agent.security.FullInputPasswordEncoder();
    }
}
