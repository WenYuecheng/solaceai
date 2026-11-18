package com.mindbridge.agent.service;

import com.mindbridge.agent.config.MindBridgeProperties;
import com.mindbridge.agent.domain.*;
import com.mindbridge.agent.dto.AdminToolExecution;
import com.mindbridge.agent.repository.*;
import com.mindbridge.agent.service.mcp.*;
import jakarta.persistence.*;
import java.time.Instant;
import java.util.*;
import java.util.function.Supplier;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.task.TaskExecutor;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

/** Durable side-effect orchestration. No external I/O runs inside the lease transaction. */
@Service
public class ToolOrchestrationService {
    private final ExcelReportWriter excelReportWriter;
    private final AlertNotifier alertNotifier;
    private final PsychologicalReportRepository reportRepository;
    private final AlertRecordRepository alertRecordRepository;
    private final MindBridgeProperties properties;
    private final TaskExecutor mcpTaskExecutor;
    private final TransactionTemplate transactionTemplate;
    @PersistenceContext private EntityManager entityManager;
    private CaseWorkflow caseWorkflow;

    public ToolOrchestrationService(ExcelReportWriter excelReportWriter, AlertNotifier alertNotifier,
            PsychologicalReportRepository reportRepository, AlertRecordRepository alertRecordRepository,
            MindBridgeProperties properties, @Qualifier("mcpTaskExecutor") TaskExecutor mcpTaskExecutor,
            TransactionTemplate transactionTemplate) {
        this.excelReportWriter=excelReportWriter;this.alertNotifier=alertNotifier;
        this.reportRepository=reportRepository;this.alertRecordRepository=alertRecordRepository;
        this.properties=properties;this.mcpTaskExecutor=mcpTaskExecutor;
        this.transactionTemplate=new TransactionTemplate(Objects.requireNonNull(transactionTemplate.getTransactionManager()));
        this.transactionTemplate.setPropagationBehavior(TransactionDefinition.PROPAGATION_REQUIRES_NEW);
    }
    @Autowired(required=false) public void setCaseWorkflow(CaseWorkflow workflow){this.caseWorkflow=workflow;}
    public void handleAsync(Long reportId){mcpTaskExecutor.execute(()->dispatchPending(reportId));}
    public void handle(Long reportId){execute(reportId,false,"system",false,false,false);}
    public void recover(Long reportId, boolean recoverInterrupted, String actor){execute(reportId,recoverInterrupted,actor,true,false,false);}

    public ToolStatus recoverExecution(Long executionId, boolean recoverInterrupted, String actor) {
        try {
            execute(executionId, recoverInterrupted, actor, true, false, true);
            var result = tx(() -> entityManager.find(BusinessToolExecution.class, executionId).getStatus());
            if (result == ToolStatus.SUCCESS || result == ToolStatus.LOG_ONLY) return result;
        } catch (ResponseStatusException error) {
            throw error;
        } catch (RuntimeException error) {
            throw recoveryFailed(error);
        }
        throw recoveryFailed(null);
    }

    @Scheduled(initialDelay = 10000, fixedDelay = 15000)
    public void reconcilePending() {
        List<Long> pending = tx(() -> entityManager.createQuery(
                "select e.reportId from BusinessToolExecution e where e.status=:status order by e.updatedAt", Long.class)
                .setParameter("status", ToolStatus.PENDING).setMaxResults(16).getResultList());
        pending.forEach(this::dispatchPending);
    }

    private void dispatchPending(Long reportId) {
        try {
            execute(reportId, false, "system", false, true, false);
        } catch (RuntimeException error) {
            org.slf4j.LoggerFactory.getLogger(ToolOrchestrationService.class)
                    .warn("Report tool dispatch did not finish for report {} ({})", reportId, error.getClass().getSimpleName());
        }
    }

    private void execute(Long reportId, boolean recoverInterrupted, String actor, boolean manual,
                         boolean pendingOnly, boolean existingOnly) {
        String owner=UUID.randomUUID().toString();
        PsychologicalReport report=tx(()->{
            PsychologicalReport value=entityManager.find(PsychologicalReport.class,reportId,LockModeType.PESSIMISTIC_WRITE);
            if(value==null && existingOnly)throw conflict("工具执行不存在");
            if(value==null)throw new ResponseStatusException(HttpStatus.NOT_FOUND,"报告不存在");
            BusinessToolExecution receipt=entityManager.find(BusinessToolExecution.class,reportId);
            if(pendingOnly && (receipt==null || receipt.getStatus()!=ToolStatus.PENDING))return null;
            if(existingOnly && (receipt==null || (receipt.getStatus()!=ToolStatus.FAILED && receipt.getStatus()!=ToolStatus.RUNNING)))
                throw conflict("只能恢复失败或中断的工具执行");
            if(receipt!=null && receipt.getStatus()==ToolStatus.SUCCESS)return null;
            if(receipt!=null && receipt.getStatus()==ToolStatus.RUNNING) {
                if(!manual || !recoverInterrupted || receipt.getLeaseExpiresAt()==null || receipt.getLeaseExpiresAt().isAfter(Instant.now()))
                    throw conflict("执行仍在运行，或需在租约到期后显式确认中断恢复");
            }
            boolean interrupted=receipt!=null && receipt.getStatus()==ToolStatus.RUNNING;
            if(receipt==null){receipt=new BusinessToolExecution(reportId);entityManager.persist(receipt);}
            receipt.claim(owner,actor,interrupted);
            entityManager.persist(new ToolExecutionAudit(reportId,actor,manual?"RECOVER_EXECUTION":"START_EXECUTION",interrupted?"INTERRUPTED_CONFIRMED":"CLAIMED"));
            // Resolve lazy relations while the report transaction is open.
            value.getUser().getUsername();if(value.getSession()!=null)value.getSession().getPublicId();
            return value;
        });
        if(report==null)return;
        try {
            // Deliver the committed alert before optional case follow-up or workbook work can fail.
            if(report.getRiskLevel()==RiskLevel.HIGH)sendAlerts(report,owner,actor,recoverInterrupted);
            RiskCase riskCase=caseWorkflow==null?null:caseWorkflow.ensure(report);
            if(report.getExcelStatus()!=ToolStatus.SUCCESS) {
                try {excelReportWriter.write(report);report.setExcelStatus(ToolStatus.SUCCESS);report.setToolError(null);}
                catch(Exception error){report.setExcelStatus(ToolStatus.FAILED);report.setToolError("Excel 写入失败，请检查存储与工具服务");}
            }
            if(riskCase!=null && report.getEmailStatus()==ToolStatus.SUCCESS)caseWorkflow.alertSent(riskCase.getId());
            tx(()->{
                BusinessToolExecution receipt=owned(reportId,owner);
                PsychologicalReport current=entityManager.find(PsychologicalReport.class,reportId);
                current.setExcelStatus(report.getExcelStatus());current.setEmailStatus(report.getEmailStatus());current.setToolError(report.getToolError());
                receipt.setStatus(report.getExcelStatus()==ToolStatus.SUCCESS && report.getEmailStatus()!=ToolStatus.FAILED && report.getEmailStatus()!=ToolStatus.RUNNING ? ToolStatus.SUCCESS:ToolStatus.FAILED);
                if(receipt.getStatus()==ToolStatus.SUCCESS && report.getEmailStatus()==ToolStatus.LOG_ONLY)receipt.setStatus(ToolStatus.LOG_ONLY);
                receipt.setErrorMessage(receipt.getStatus()==ToolStatus.FAILED
                        ? (report.getToolError()==null?"预警未完成，请核查投递记录与租约":report.getToolError()) : null);
                entityManager.persist(new ToolExecutionAudit(reportId,actor,"FINISH_EXECUTION",receipt.getStatus().name()));
                return null;
            });
        } catch(Exception error) {
            // Do not erase a committed RUNNING lease when the outcome is uncertain.
            tx(()->{owned(reportId,owner).setErrorMessage("执行中断；请核查外部结果后恢复");return null;});
            throw error;
        }
    }
    private void sendAlerts(PsychologicalReport report,String owner,String actor,boolean recoverInterrupted) {
        var recipients=configuredRecipients();
        ToolStatus aggregate=recipients.isEmpty()?ToolStatus.FAILED:ToolStatus.SUCCESS;
        for(String recipient:recipients) {
            AlertRecord alert=tx(()->{
                owned(report.getId(),owner);
                var records=alertRecordRepository.findByReport_Id(report.getId()).stream().filter(a->recipient.equals(a.getRecipient())).toList();
                if(records.stream().anyMatch(a->a.getStatus()==ToolStatus.SUCCESS))return null;
                AlertRecord value=records.stream().filter(a->a.getStatus()==ToolStatus.RUNNING).findFirst().orElse(records.isEmpty()?null:records.get(0));
                if(value!=null && value.getStatus()==ToolStatus.RUNNING && (!recoverInterrupted || value.getLeaseExpiresAt()==null || value.getLeaseExpiresAt().isAfter(Instant.now())))return value;
                if(value==null){value=new AlertRecord();value.setReport(entityManager.getReference(PsychologicalReport.class,report.getId()));value.setRecipient(recipient);}
                boolean interrupted=value.getStatus()==ToolStatus.RUNNING;
                value.claim(owner,actor,interrupted);
                entityManager.persist(new ToolExecutionAudit(report.getId(),actor,interrupted?"RECOVER_ALERT":"SEND_ALERT","CLAIMED"));
                return alertRecordRepository.saveAndFlush(value);
            });
            if(alert==null)continue;
            if(!owner.equals(alert.getLeaseOwner())){aggregate=ToolStatus.RUNNING;continue;}
            ToolStatus result;
            try {alertNotifier.notify(alert,report);result=alertNotifier.isLogOnly()?ToolStatus.LOG_ONLY:ToolStatus.SUCCESS;}
            catch(LogOnlyDeliveryException logOnly){result=ToolStatus.LOG_ONLY;}
            catch(Exception error){result=ToolStatus.RUNNING;}
            ToolStatus outcome=result;
            tx(()->{
                owned(report.getId(),owner);
                AlertRecord current=entityManager.find(AlertRecord.class,alert.getId(),LockModeType.PESSIMISTIC_WRITE);
                if(!owner.equals(current.getLeaseOwner()))throw conflict("预警租约已变更");
                current.setStatus(outcome);
                current.setErrorMessage(outcome==ToolStatus.RUNNING?"投递结果不确定；租约到期后须人工确认恢复":null);
                return null;
            });
            if(result==ToolStatus.RUNNING || aggregate!=ToolStatus.RUNNING && result!=ToolStatus.SUCCESS)aggregate=result;
        }
        report.setEmailStatus(aggregate);
    }
    private BusinessToolExecution owned(Long id,String owner){
        BusinessToolExecution receipt=entityManager.find(BusinessToolExecution.class,id,LockModeType.PESSIMISTIC_WRITE);
        if(receipt==null||!owner.equals(receipt.getLeaseOwner()))throw conflict("执行租约已变更");return receipt;
    }
    public List<BusinessToolExecution> executions(){return tx(()->entityManager.createQuery("select e from BusinessToolExecution e order by e.updatedAt desc",BusinessToolExecution.class).setMaxResults(100).getResultList());}
    public List<AdminToolExecution> auditExecutions() {
        return tx(() -> {
            var rows = entityManager.createQuery(
                    "select e from BusinessToolExecution e order by e.reportId desc", BusinessToolExecution.class)
                    .setMaxResults(200).getResultList();
            if (rows.isEmpty()) return List.of();
            var ids = rows.stream().map(BusinessToolExecution::getReportId).toList();
            var receipts = new HashMap<Long, TurnReceipt>();
            entityManager.createQuery("select r from TurnReceipt r where r.reportId in :ids", TurnReceipt.class)
                    .setParameter("ids", ids).getResultList()
                    .forEach(receipt -> receipts.put(receipt.getReportId(), receipt));
            var reports = new HashMap<Long, PsychologicalReport>();
            entityManager.createQuery("select r from PsychologicalReport r where r.id in :ids", PsychologicalReport.class)
                    .setParameter("ids", ids).getResultList()
                    .forEach(report -> reports.put(report.getId(), report));
            return rows.stream().map(row -> {
                Long id = row.getReportId();
                var receipt = receipts.get(id);
                var report = reports.get(id);
                Instant created = row.getCreatedAt();
                if (created == null && report != null) created = report.getCreatedAt();
                String error = row.getErrorMessage();
                if (error == null && row.getStatus()==ToolStatus.FAILED && report != null) error = report.getToolError();
                Map<String, Object> result = report == null ? Map.of()
                        : Map.of("reportId", id, "excelStatus", report.getExcelStatus(), "emailStatus", report.getEmailStatus());
                return new AdminToolExecution(id, receipt == null ? null : receipt.getRunId(), "process_report",
                        row.getStatus(), error, result, created, row.getUpdatedAt(), id, row.getErrorMessage(),
                        row.getAttempts(), row.getActor(), row.getLeaseExpiresAt(), row.getLeaseOwner(), row.isRecoveredInterrupted());
            }).toList();
        });
    }
    public List<ToolExecutionAudit> audits(){return tx(()->entityManager.createQuery("select a from ToolExecutionAudit a order by a.createdAt desc",ToolExecutionAudit.class).setMaxResults(100).getResultList());}
    public void audit(String actor,String action,String outcome){tx(()->{entityManager.persist(new ToolExecutionAudit(null,actor,action,outcome));return null;});}
    public void retryAlert(Long alertId,boolean recoverInterrupted,String actor){
        Long reportId=tx(()->{
            AlertRecord row=entityManager.find(AlertRecord.class,alertId);
            if(row==null)throw new ResponseStatusException(HttpStatus.NOT_FOUND,"预警不存在");
            if(row.getStatus()==ToolStatus.SUCCESS)return null;
            if(row.getStatus()!=ToolStatus.FAILED && row.getStatus()!=ToolStatus.RUNNING)throw conflict("仅允许重试失败或中断的邮件预警");
            if(!configuredRecipients().contains(row.getRecipient()))throw conflict("原收件人已不在服务端配置中");
            if(row.getStatus()==ToolStatus.RUNNING && (!recoverInterrupted || row.getLeaseExpiresAt()==null || row.getLeaseExpiresAt().isAfter(Instant.now())))throw conflict("投递结果不确定；到期并显式确认后方可恢复");
            return row.getReport().getId();
        });
        if(reportId!=null)recover(reportId,recoverInterrupted,actor);
    }
    private List<String> configuredRecipients(){return properties.getMcp().getEmail().getRecipients().stream().filter(Objects::nonNull).map(String::trim).filter(s->!s.isEmpty()).distinct().toList();}
    private <T>T tx(Supplier<T> action){return transactionTemplate.execute(status->action.get());}
    private static ResponseStatusException conflict(String message){return new ResponseStatusException(HttpStatus.CONFLICT,message);}
    private static ResponseStatusException recoveryFailed(Throwable cause) {
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "工具恢复失败，请查看执行记录", cause);
    }
}
