package com.mindbridge.agent.controller;

import com.mindbridge.agent.config.MindBridgeProperties;
import com.mindbridge.agent.domain.*;
import com.mindbridge.agent.dto.AdminToolExecution;
import com.mindbridge.agent.security.CurrentUser;
import com.mindbridge.agent.service.ToolOrchestrationService;
import java.util.*;
import org.springframework.http.HttpStatus;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

/** These routes inherit the existing /api/admin/** administrator authorization rule. */
@RestController
@RequestMapping("/api/admin")
public class NotificationAdminController {
    private final MindBridgeProperties properties;
    private final ToolOrchestrationService tools;
    private final JavaMailSender mailSender;
    public NotificationAdminController(MindBridgeProperties properties,ToolOrchestrationService tools,JavaMailSender mailSender){this.properties=properties;this.tools=tools;this.mailSender=mailSender;}
    @GetMapping("/mail/status") public Map<String,Object> status(){
        var config=properties.getMcp().getEmail();
        var missing=new ArrayList<String>();
        if(config.getFrom()==null||config.getFrom().isBlank())missing.add("from");
        if(recipients().isEmpty())missing.add("recipients");
        return Map.of("mode",config.getMode(),"from",Objects.requireNonNullElse(config.getFrom(),""),"recipients",recipients(),"missing",missing,
                "ready",missing.isEmpty() && "smtp".equalsIgnoreCase(config.getMode()),"deliveryGuarantee","SMTP acceptance only; inbox delivery is not verified");
    }
    @PostMapping("/mail/test") public Map<String,Object> test(@AuthenticationPrincipal CurrentUser actor,@RequestBody(required=false) Map<String,Object> body){
        strict(body,Set.of());
        if(recipients().isEmpty())throw new ResponseStatusException(HttpStatus.CONFLICT,"请先配置服务端预警收件人");
        String mode=properties.getMcp().getEmail().getMode();
        if("log".equalsIgnoreCase(mode)){tools.audit(actor.getUsername(),"MAIL_TEST","LOG_ONLY");return Map.of("status","LOG_ONLY","recipients",recipients());}
        if(!"smtp".equalsIgnoreCase(mode))throw new ResponseStatusException(HttpStatus.CONFLICT,"测试邮件仅支持直接 SMTP 或日志模式");
        tools.audit(actor.getUsername(),"MAIL_TEST","STARTED");
        var message=new SimpleMailMessage();message.setFrom(properties.getMcp().getEmail().getFrom());message.setTo(recipients().toArray(String[]::new));
        message.setSubject("[SolaceAI] 邮件服务测试");message.setText("这是一封管理员发起的邮件服务测试，不代表发生风险事件。");
        try {mailSender.send(message);tools.audit(actor.getUsername(),"MAIL_TEST","SMTP_ACCEPTED");}
        catch(Exception error){tools.audit(actor.getUsername(),"MAIL_TEST","OUTCOME_UNKNOWN");throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,"测试投递结果不确定，请核查邮件服务");}
        return Map.of("status","SMTP_ACCEPTED","recipients",recipients());
    }
    @PostMapping("/mail/alerts/{id}/retry") public Map<String,Object> retry(@PathVariable Long id,@AuthenticationPrincipal CurrentUser actor,@RequestBody(required=false) Map<String,Object> body){
        tools.retryAlert(id,recovery(body),actor.getUsername());return Map.of("id",id,"status","PROCESSED");
    }
    @GetMapping("/tool-jobs") public List<Object> historicalJobs(){return List.of();}
    @GetMapping("/dead-letters") public List<Object> historicalDeadLetters(){return List.of();}
    @GetMapping("/tool-executions") public List<AdminToolExecution> executions(){return tools.auditExecutions();}
    @GetMapping({"/tool-executions/audits","/tool-audits"}) public List<ToolExecutionAudit> audits(){return tools.audits();}
    @PostMapping("/tool-executions/{executionId}/recover") public Map<String,Object> recover(@PathVariable Long executionId,@AuthenticationPrincipal CurrentUser actor,@RequestBody(required=false) Map<String,Object> body){
        var status = tools.recoverExecution(executionId,recovery(body),actor.getUsername());
        return Map.of("id",executionId,"reportId",executionId,"status","PROCESSED","executionStatus",status);
    }
    private List<String> recipients(){return properties.getMcp().getEmail().getRecipients().stream().filter(Objects::nonNull).map(String::trim).filter(s->!s.isEmpty()).distinct().toList();}
    private boolean recovery(Map<String,Object> body){strict(body,Set.of("recoverInterrupted"));Object flag=body==null?null:body.get("recoverInterrupted");if(flag!=null && !(flag instanceof Boolean))throw new IllegalArgumentException("recoverInterrupted must be a boolean");return Boolean.TRUE.equals(flag);}
    private void strict(Map<String,Object> body,Set<String> allowed){if(body!=null&&!allowed.containsAll(body.keySet()))throw new IllegalArgumentException("Unknown request fields");}
}
