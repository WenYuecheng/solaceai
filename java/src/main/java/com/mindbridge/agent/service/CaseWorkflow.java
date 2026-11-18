package com.mindbridge.agent.service;

import com.mindbridge.agent.domain.*;
import com.mindbridge.agent.repository.*;
import java.time.Instant;
import java.util.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

@Service
public class CaseWorkflow {
    private final RiskCaseRepository cases;private final CaseNoteRepository notes;private final ChatSessionRepository sessions;
    private final PrivacySanitizer privacy;
    private final CaseReportBindingRepository bindings;
    @jakarta.persistence.PersistenceContext private jakarta.persistence.EntityManager entityManager;
    public CaseWorkflow(RiskCaseRepository cases,CaseNoteRepository notes,ChatSessionRepository sessions,PrivacySanitizer privacy,CaseReportBindingRepository bindings){this.cases=cases;this.notes=notes;this.sessions=sessions;this.privacy=privacy;this.bindings=bindings;}
    @Transactional public RiskCase ensure(PsychologicalReport report){
        if(report.getRiskLevel()==RiskLevel.LOW)return null;
        if(report.getSession()==null)throw new IllegalArgumentException("报告缺少会话");
        sessions.locked(report.getSession().getId()).orElseThrow();
        var binding=bindings.findById(report.getId());
        if(binding.isPresent())return cases.locked(binding.get().getCaseId()).orElseThrow();
        // Existing installations only know the latest report pointer; lazily preserve it.
        var existing=cases.findByReportId(report.getId());
        if(existing.isPresent()){var prior=cases.locked(existing.get().getId()).orElseThrow();bind(report.getId(),prior.getId());return prior;}
        var active=cases.findFirstByUserIdAndSessionIdAndStatusNotOrderByIdDesc(report.getUser().getId(),report.getSession().getPublicId(),"CLOSED");
        RiskCase row=active.orElseGet(RiskCase::new);
        if(active.isEmpty()){row.setUserId(report.getUser().getId());row.setSessionId(report.getSession().getPublicId());row.setReportId(report.getId());row.setRiskLevel(report.getRiskLevel());row.setStatus("OPEN");row.setOwner("待接手");}
        if(active.isEmpty()||report.getRiskLevel()==RiskLevel.HIGH&&row.getRiskLevel()!=RiskLevel.HIGH){row.setRiskLevel(report.getRiskLevel());row.setReportId(report.getId());row.setSummary(privacy.sanitize(report.getSummary()));row.setHandoffSummary("风险线索："+privacy.sanitize(report.getSummary())+"。请由专业人员联系并核实当前安全与支持需要。");}
        row.setUpdatedAt(Instant.now());row=cases.saveAndFlush(row);bind(report.getId(),row.getId());return row;
    }
    @Transactional public Map<String,Object> apply(long id,String actor,String action,String text){
        text=Objects.requireNonNullElse(text,"").trim();if(Set.of("notes","close").contains(action)&&(text.isEmpty()||text.length()>2000))throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY,"记录长度应为1至2000字符");
        var row=lockCase(id);
        if(action.equals("close")&&row.getStatus().equals("CLOSED")||action.equals("acknowledge")&&row.getStatus().equals("ACKNOWLEDGED"))return response(row);
        if(row.getStatus().equals("CLOSED"))throw new ResponseStatusException(HttpStatus.CONFLICT,"已结案个案不能修改");
        switch(action){case "acknowledge"->{if(!Set.of("OPEN","ALERT_SENT").contains(row.getStatus()))throw new ResponseStatusException(HttpStatus.CONFLICT,"当前状态无法接手");row.setStatus("ACKNOWLEDGED");row.setOwner(actor);row.setAcknowledgedBy(actor);row.setAcknowledgedAt(Instant.now());text="已确认接手该个案";}
            case "close"->{if(!row.getStatus().equals("ACKNOWLEDGED"))throw new ResponseStatusException(HttpStatus.CONFLICT,"请先确认接手再结案");row.setStatus("CLOSED");text="结案原因："+text;}
            case "notes"->{} default->throw new IllegalArgumentException("Unknown case action");}
        row.setUpdatedAt(Instant.now());cases.saveAndFlush(row);var note=new CaseNote();note.setCaseId(id);note.setActor(actor);note.setNote(privacy.sanitize(text));notes.saveAndFlush(note);return action.equals("notes")?noteResponse(note):response(row);
    }
    @Transactional public void alertSent(long id){var row=lockCase(id);if(row.getStatus().equals("OPEN")){row.setStatus("ALERT_SENT");row.setUpdatedAt(Instant.now());cases.save(row);}}
    private void bind(long reportId,long caseId){
        // Called only while holding the associated session lock, never reassign a report.
        var prior=bindings.findById(reportId);
        if(prior.isPresent()){if(prior.get().getCaseId()!=caseId)throw new IllegalStateException("Report already belongs to another case");return;}
        bindings.saveAndFlush(new CaseReportBinding(reportId,caseId));
    }
    private RiskCase lockCase(long id){
        String key=cases.sessionKey(id).orElseThrow(()->new ResponseStatusException(HttpStatus.NOT_FOUND,"个案不存在"));
        long user=cases.userKey(id).orElseThrow();
        Long session=entityManager.createQuery("select s.id from ChatSession s where s.publicId=:key and s.user.id=:user",Long.class).setParameter("key",key).setParameter("user",user).getSingleResult();
        sessions.locked(session).orElseThrow();
        return cases.locked(id).orElseThrow(()->new ResponseStatusException(HttpStatus.NOT_FOUND,"个案不存在"));
    }
    @Transactional(readOnly=true) public List<Map<String,Object>> list(){return cases.findTop100ByOrderByUpdatedAtDesc().stream().map(this::response).toList();}
    @Transactional(readOnly=true) public List<Map<String,Object>> notes(long id){if(!cases.existsById(id))throw new ResponseStatusException(HttpStatus.NOT_FOUND,"个案不存在");return notes.findByCaseIdOrderByCreatedAtAscIdAsc(id).stream().map(this::noteResponse).toList();}
    private Map<String,Object> response(RiskCase row){Map<String,Object> m=new LinkedHashMap<>();m.put("id",row.getId());m.put("reportId",row.getReportId());m.put("riskLevel",row.getRiskLevel());m.put("status",row.getStatus());m.put("owner",row.getOwner());m.put("summary",privacy.sanitize(row.getSummary()));m.put("handoffSummary",privacy.sanitize(row.getHandoffSummary()));m.put("acknowledgedBy",row.getAcknowledgedBy());m.put("acknowledgedAt",row.getAcknowledgedAt());m.put("createdAt",row.getCreatedAt());m.put("updatedAt",row.getUpdatedAt());return m;}
    private Map<String,Object> noteResponse(CaseNote n){return Map.of("id",n.getId(),"caseId",n.getCaseId(),"actor",n.getActor(),"note",privacy.sanitize(n.getNote()),"createdAt",n.getCreatedAt());}
}
