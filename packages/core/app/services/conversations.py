"""SolaceAI 用户对话会话与审计留痕服务 (User Conversation & Audit Preservation Service)"""
import json
from app.services.citations import citation_payloads
from app.services.answer_trace import load_answer_trace
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models.entities import AnswerFeedback, KnowledgeChunk, ChatMessage, ChatMessageRun, AgentTurnReceipt, ChatSession, SessionVisibility
from app.services.privacy import PrivacySanitizer


class ConversationService:
    """用户对话会话业务服务（支持审计保留的可见性控制）。"""

    def __init__(self, db: Session):
        """初始化会话服务。"""
        self.db = db
        self.privacy = PrivacySanitizer()

    def is_hidden(self, user_id: int, session_id: int) -> bool:
        """检查指定会话是否已被该用户隐藏。"""
        marker = self.db.get(SessionVisibility, (user_id, session_id))
        return bool(marker and marker.hidden)

    def _owned(self, user_id: int, public_id: str) -> ChatSession:
        """根据公开 UUID 获取当前用户拥有的非隐藏会话。"""
        session = self.db.scalar(select(ChatSession).where(ChatSession.user_id == user_id, ChatSession.public_id == public_id))
        if session is None or self.is_hidden(user_id, session.id):
            raise HTTPException(404, '会话不存在')
        return session

    def _summary(self, session: ChatSession) -> dict:
        """将会话实体转换为脱敏后的前端摘要字典。"""
        return {'sessionId': session.public_id, 'title': self.privacy.sanitize(session.title),
                'createdAt': session.created_at.isoformat(), 'updatedAt': session.updated_at.isoformat()}

    def list(self, user_id: int, limit: int = 50, offset: int = 0) -> dict:
        """分页查询指定用户所有可见会话。"""
        hidden = select(SessionVisibility.session_id).where(SessionVisibility.user_id == user_id, SessionVisibility.hidden.is_(True))
        predicate = (ChatSession.user_id == user_id, ChatSession.id.not_in(hidden))
        total = self.db.scalar(select(func.count()).select_from(ChatSession).where(*predicate))
        sessions = self.db.scalars(select(ChatSession).where(*predicate).order_by(ChatSession.updated_at.desc(), ChatSession.id.desc()).limit(limit).offset(offset)).all()
        return {'items': [self._summary(item) for item in sessions], 'total': total, 'limit': limit, 'offset': offset}

    def detail(self, user_id: int, public_id: str) -> dict:
        """获取指定会话的完整脱敏历史消息详情。"""
        session = self._owned(user_id, public_id)
        messages = self.db.scalars(select(ChatMessage).where(ChatMessage.session_id == session.id, ChatMessage.user_id == user_id, func.lower(ChatMessage.role).in_(['user', 'assistant'])).order_by(ChatMessage.created_at, ChatMessage.id)).all()
        items = []
        for item in messages:
            binding = self.db.get(ChatMessageRun, item.id)
            if binding and binding.user_id != user_id:
                binding = None
            receipt = self.db.get(AgentTurnReceipt, binding.run_id) if binding else None
            if receipt and (receipt.user_id != user_id or receipt.session_id != session.id):
                receipt = None
                binding = None
            citations = json.loads(receipt.citations_json or '[]') if receipt else []
            items.append({'id': str(item.id), 'messageId': item.id if binding else None, 'runId': binding.run_id if binding else None,
                          'role': item.role.lower(), 'content': self.privacy.sanitize(item.content), 'createdAt': item.created_at.isoformat(),
                          'citations': citation_payloads(citations, item.content),
                          'answerTrace': load_answer_trace(self.db, user_id, session.id, binding.run_id, item.id)
                          if binding and item.role.lower() == 'assistant' else None})
        return {**self._summary(session), 'messages': items}

    def hide(self, user_id: int, public_id: str) -> dict:
        """软删除/隐藏指定会话。"""
        session = self._owned(user_id, public_id)
        marker = self.db.get(SessionVisibility, (user_id, session.id))
        if marker is None:
            self.db.add(SessionVisibility(user_id=user_id, session_id=session.id, hidden=True))
        else:
            marker.hidden = True
        self.db.commit()
        try:
            from app.core.config import get_settings
            from app.services.memory import RedisShortTermMemoryStore
            RedisShortTermMemoryStore(get_settings()).evict(user_id, public_id)
        except Exception:
            pass
        return {'deleted': True}


def save_feedback(db: Session, user_id: int, request) -> dict:
    """持久化记录用户对指定回答质量的反馈评价 (Save Answer Feedback)"""
    message = db.get(ChatMessage, request.messageId)
    binding = db.get(ChatMessageRun, request.messageId)
    if not message or message.user_id != user_id or message.role.lower() != 'assistant' or not binding or binding.user_id != user_id or binding.run_id != request.runId:
        raise HTTPException(404, '回答不存在')
    row = db.scalar(select(AnswerFeedback).where(AnswerFeedback.user_id == user_id, AnswerFeedback.message_id == message.id))
    if row is None:
        row = AnswerFeedback(user_id=user_id, message_id=message.id, run_id=request.runId)
        db.add(row)
    row.helpful, row.category, row.note = request.helpful, request.category, request.note.strip()
    db.commit()
    return {'id': row.id, 'saved': True}


def evidence_detail(db: Session, user_id: int, run_id: str, chunk_id: int) -> dict:
    """获取单轮回答中引用的知识切片快照详情 (Retrieve Evidence Snapshot Detail)"""
    receipt = db.get(AgentTurnReceipt, run_id)
    if not receipt or receipt.user_id != user_id:
        raise HTTPException(404, '证据不存在')
    session = db.get(ChatSession, receipt.session_id)
    if session is None or session.user_id != user_id or ConversationService(db).is_hidden(user_id, session.id):
        raise HTTPException(404, '证据不存在')
    citations = json.loads(receipt.citations_json or '[]')
    citation = next((item for item in citations if str(item.get('chunk_id', item.get('chunkId'))) == str(chunk_id)), None)
    if citation is None:
        raise HTTPException(404, '证据不存在')
    chunk = db.get(KnowledgeChunk, chunk_id)
    # 已删除的文档保留引用快照，不将当前其他版本当作旧证据。
    snapshot = citation.get('metadata') or {}
    return {'chunkId': chunk_id, 'source': citation.get('source', ''),
            'content': citation.get('content') or citation.get('preview') or '',
            'sectionPath': snapshot.get('section_path') or citation.get('sectionPath') or [],
            'documentVersion': snapshot.get('document_version'),
            'page': snapshot.get('page') or snapshot.get('page_number'),
            'title': citation.get('title') or snapshot.get('title') or citation.get('source', ''),
            'url': snapshot.get('url'), 'sourceUrl': snapshot.get('url'), 'license': snapshot.get('license'),
            'metadata': snapshot,
            'available': chunk is not None}

