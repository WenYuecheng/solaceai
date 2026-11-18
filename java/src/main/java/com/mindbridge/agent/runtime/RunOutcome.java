package com.mindbridge.agent.runtime;

import com.mindbridge.agent.domain.*;
import com.mindbridge.agent.service.PsychologyAssessment;
import java.util.*;

public record RunOutcome(String runId,String answer,IntentType intent,RiskLevel risk,
                         PsychologyAssessment assessment,List<Map<String,Object>> evidence,
                         List<Map<String,Object>> steps,List<Map<String,Object>> agentSteps,
                         String memoryBrief,String modelBackend,boolean degraded,
                         List<Map<String,Object>> modelVisibleEvidence) {
    public RunOutcome(String runId,String answer,IntentType intent,RiskLevel risk,PsychologyAssessment assessment,List<Map<String,Object>> evidence,List<Map<String,Object>> steps,List<Map<String,Object>> agentSteps,String memoryBrief,String modelBackend,boolean degraded){this(runId,answer,intent,risk,assessment,evidence,steps,agentSteps,memoryBrief,modelBackend,degraded,List.of());}
    public boolean requiresReport(){return intent!=IntentType.CHAT;}
}
