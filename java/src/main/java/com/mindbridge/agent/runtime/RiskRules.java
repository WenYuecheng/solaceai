package com.mindbridge.agent.runtime;

import com.mindbridge.agent.domain.*;
import com.mindbridge.agent.service.PsychologyAssessment;
import java.util.*;
import java.util.regex.Pattern;

/** Same scoped high-risk rules as SolaceAI; a negative rule does not override model assessment. */
public final class RiskRules {
    private static final List<String> HIGH=List.of("自杀","自残","不想活","结束生命","伤害自己","轻生","suicide","kill myself","self harm");
    private static final List<String> CONSULT=List.of("焦虑","抑郁","压力","失眠","难过","崩溃","痛苦","无助","心理","咨询","anxious","depress","stress");
    private static final List<String> ACTIONS=List.of("已经吞了药","已经吃了大量药","已经割腕","正在割腕","准备跳下去","站在楼顶","站在天台","写好了告别信","准备彻底消失","准备好遗书","已经服毒","准备跳楼");
    private static final String TERMS="(?:自杀|自残|不想活|结束生命|伤害自己|轻生|suicide|kill myself|self harm)";
    private static final Pattern NEGATED=Pattern.compile("(?:没有|并无|从未|不会|不打算|不准备|不想|不愿意|不考虑|不曾|没想过)\\s*(?:任何|再也|再)?\\s*(?:想过|想要|想|要|去|会)?\\s*(?:"+TERMS+"|"+String.join("|",ACTIONS)+")(?:的)?(?:想法|念头|意图|计划|冲动|行为)?|\\b(?:do not|don't|never|will not|won't)\\s+(?:want to\\s+)?(?:kill myself|self harm|commit suicide)\\b");
    private static final Pattern REFERENCE=Pattern.compile("课堂|作业|新闻|报道|论文|小说|电影|科普|资料|预防|教材|案例|课程");
    private static final Pattern SAFE=Pattern.compile("我(?:现在|目前)?(?:很)?安全|不是我的经历|没有.{0,8}(?:想法|意图|念头)");
    private static final Pattern PERSONAL=Pattern.compile("(?:我|自己).{0,8}(?:想|要|准备|打算|决定).{0,6}"+TERMS);
    private RiskRules(){}
    public static boolean high(String input){String text=input.toLowerCase(Locale.ROOT);if(Pattern.compile("(?:不是|并非|不能说|不代表)\\s*(?:没有|不想|不会)").matcher(text).find())return contains(text,HIGH);String remaining=NEGATED.matcher(text).replaceAll("");if(contains(remaining,ACTIONS))return true;if(!remaining.equals(text)&&Pattern.compile("(?:但|可是|不过).{0,8}(?:我)?(?:打算|准备|决定).{0,8}(?:试一次|动手)").matcher(remaining).find())return true;for(String clause:remaining.split("[，,。；;！!？?\\n]")){if(SAFE.matcher(text).find()&&REFERENCE.matcher(clause).find()&&!PERSONAL.matcher(clause).find())continue;if(contains(clause,HIGH)||contains(clause,ACTIONS))return true;}return false;}
    public static boolean consult(String text){return contains(text.toLowerCase(Locale.ROOT),CONSULT);}
    private static boolean contains(String s,List<String> words){return words.stream().anyMatch(s::contains);}
    public static PsychologyAssessment heuristic(String text){if(high(text))return label("高风险");if(contains(text.toLowerCase(Locale.ROOT),List.of("抑郁","低落","崩溃","难过","depress","hopeless")))return new PsychologyAssessment(EmotionLabel.DEPRESSED,3.1,RiskLevel.MEDIUM,.75,"检测到低落或抑郁相关表达");if(consult(text))return new PsychologyAssessment(EmotionLabel.ANXIETY,2.2,RiskLevel.LOW,.72,"检测到焦虑或压力相关表达");return label("正常");}
    public static PsychologyAssessment label(String label){return switch(label){case "正常"->new PsychologyAssessment(EmotionLabel.NORMAL,0,RiskLevel.LOW,.75,"本地Qwen分类：正常；风险按固定标签映射");case "焦虑"->new PsychologyAssessment(EmotionLabel.ANXIETY,2,RiskLevel.LOW,.75,"本地Qwen分类：焦虑；风险按固定标签映射");case "低落"->new PsychologyAssessment(EmotionLabel.DEPRESSED,3,RiskLevel.MEDIUM,.75,"本地Qwen分类：低落；风险按固定标签映射");case "高风险"->new PsychologyAssessment(EmotionLabel.HIGH_RISK,4,RiskLevel.HIGH,.95,"检测到明确高风险表达");default->throw new IllegalArgumentException("Invalid local emotion label");};}
}
