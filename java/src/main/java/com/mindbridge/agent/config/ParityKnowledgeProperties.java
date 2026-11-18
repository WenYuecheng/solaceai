package com.mindbridge.agent.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix="mindbridge.parity.knowledge")
public class ParityKnowledgeProperties {
    private String writerLockPath="data/knowledge/writer.lock";
    public String getWriterLockPath(){return writerLockPath;} public void setWriterLockPath(String value){writerLockPath=value;}
    private String backend="bm25";
    private String tenantId="default";
    private String knowledgeBaseId="default";
    private String bgeUrl="http://127.0.0.1:8091";
    private String milvusUrl="http://127.0.0.1:19530";
    private String collection="mindbridge_knowledge";
    private String milvusToken="";
    private boolean rerankerEnabled;
    private String documentPython="";
    private String documentModulePath="";
    public String getTenantId(){return tenantId;} public void setTenantId(String value){this.tenantId=value;}
    public String getKnowledgeBaseId(){return knowledgeBaseId;} public void setKnowledgeBaseId(String value){this.knowledgeBaseId=value;}
    public String getBackend(){return backend;} public void setBackend(String backend){this.backend=backend;}
    public String getBgeUrl(){return bgeUrl;} public void setBgeUrl(String bgeUrl){this.bgeUrl=bgeUrl;}
    public String getMilvusUrl(){return milvusUrl;} public void setMilvusUrl(String milvusUrl){this.milvusUrl=milvusUrl;}
    public String getCollection(){return collection;} public void setCollection(String collection){this.collection=collection;}
    public String getMilvusToken(){return milvusToken;} public void setMilvusToken(String milvusToken){this.milvusToken=milvusToken;}
    public String getDocumentPython(){return documentPython;} public void setDocumentPython(String value){this.documentPython=value;}
    public String getDocumentModulePath(){return documentModulePath;} public void setDocumentModulePath(String value){this.documentModulePath=value;}
    public boolean isRerankerEnabled(){return rerankerEnabled;} public void setRerankerEnabled(boolean enabled){this.rerankerEnabled=enabled;}
}
