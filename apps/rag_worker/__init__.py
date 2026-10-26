"""SolaceAI 本地深度学习 RAG 推理工作进程包 (Local RAG Worker Package)

本包提供独立的计算密集型进程，负责加载 BGE-M3 向量嵌入与 Cross-Encoder 重排模型，
实现进程级物理隔离与并发控制。
"""
