package com.mindbridge.agent.runtime;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix="mindbridge.runtime")
public class RuntimeProperties {
    private String statePath="data/runtime/memory.sqlite";
    private String checkpointPath="data/runtime/checkpoints.sqlite";
    private String lockPath="data/runtime/turn-locks";
    private String localGatewayUrl="http://127.0.0.1:18181";
    private String localGatewayToken="";
    private boolean localModelEnabled=false;
    private boolean localFallbackEnabled=false;
    private String defaultBackend="cloud";
    private int contextTokens=8192;
    private int outputTokens=4096;
    private int headroomTokens=512;
    private int maxDecisions=6;
    private int maxRetrievalRounds=2;
    private int timeoutSeconds=180;
    private int modelTimeoutSeconds=30;
    private int emotionTimeoutSeconds=20;
    private String skillsPath="";
    private String redisNamespace="mindbridge:offline";
    private boolean redisHistoryEnabled=true;
    public boolean isRedisHistoryEnabled(){return redisHistoryEnabled;} public void setRedisHistoryEnabled(boolean value){redisHistoryEnabled=value;}
    public String getStatePath(){return statePath;} public void setStatePath(String v){statePath=v;}
    public String getCheckpointPath(){return checkpointPath;} public void setCheckpointPath(String v){checkpointPath=v;}
    public String getLockPath(){return lockPath;} public void setLockPath(String v){lockPath=v;}
    public String getLocalGatewayUrl(){return localGatewayUrl;} public void setLocalGatewayUrl(String v){localGatewayUrl=v;}
    public String getLocalGatewayToken(){return localGatewayToken;} public void setLocalGatewayToken(String v){localGatewayToken=v;}
    public boolean isLocalModelEnabled(){return localModelEnabled;} public void setLocalModelEnabled(boolean v){localModelEnabled=v;}
    public boolean isLocalFallbackEnabled(){return localFallbackEnabled;} public void setLocalFallbackEnabled(boolean v){localFallbackEnabled=v;}
    public String getDefaultBackend(){return defaultBackend;} public void setDefaultBackend(String v){defaultBackend=v;}
    public int getContextTokens(){return contextTokens;} public void setContextTokens(int v){contextTokens=v;}
    public int getOutputTokens(){return outputTokens;} public void setOutputTokens(int v){outputTokens=v;}
    public int getHeadroomTokens(){return headroomTokens;} public void setHeadroomTokens(int v){headroomTokens=v;}
    public int getMaxDecisions(){return maxDecisions;} public void setMaxDecisions(int v){maxDecisions=v;}
    public int getMaxRetrievalRounds(){return maxRetrievalRounds;} public void setMaxRetrievalRounds(int v){maxRetrievalRounds=v;}
    public int getTimeoutSeconds(){return timeoutSeconds;} public void setTimeoutSeconds(int v){timeoutSeconds=v;}
    public int getModelTimeoutSeconds(){return modelTimeoutSeconds;} public void setModelTimeoutSeconds(int v){modelTimeoutSeconds=v;}
    public int getEmotionTimeoutSeconds(){return emotionTimeoutSeconds;} public void setEmotionTimeoutSeconds(int v){emotionTimeoutSeconds=v;}
    public String getSkillsPath(){return skillsPath;} public void setSkillsPath(String v){skillsPath=v;}
    public String getRedisNamespace(){return redisNamespace;} public void setRedisNamespace(String v){redisNamespace=v;}
}
