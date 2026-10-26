"""SolaceAI 认证会话凭证与邮件验证码持久化模型 (Authentication & Challenge Models)

本模块定义与身份认证、会话状态保持以及安全防护相关的持久化实体：
1. 不透明浏览器会话凭据 (`AuthSession`)：基于安全哈希映射持久化客户端 Cookie 会话，
   杜绝数据库内直接明文存储 Token，支持级联注销与过期时间快速失效检索；
2. 绑定业务意图的邮箱验证挑战码 (`EmailAuthCode`)：针对学生邮箱免密登录或凭据恢复，
   持久化加盐散列摘要并跟踪重试尝试次数、有效期与单次消费状态；
3. 接口频控计数器 (`AuthRateLimit`)：防范暴力穷举攻击与邮件配额滥用，实现基于时间窗口的速率限制。
"""
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Boolean, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class AuthSession(Base):
    """用户安全会话持久化实体 (Persistent Opaque Auth Session)

    通过 SHA-256 哈希映射维护客户端 Session Cookie 与用户实体的绑定关系。
    不持久化明文 Token，即便数据库凭证泄露也无法伪造有效 Cookie。

    Attributes:
        token_hash (str): 随机 Session Token 经 SHA-256 计算后的 64 位十六进制散列主键。
        user_id (int): 关联的用户账号 ID (`user_accounts.id`)，级联删除。
        expires_at (datetime): 会话绝对失效时间戳，支持过期快速清理索引。
    """
    __tablename__ = 'auth_sessions'
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('user_accounts.id', ondelete='CASCADE'), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class EmailAuthCode(Base):
    """邮箱一次性验证码挑战实体 (Purpose-Bound Email Challenge Code)

    记录发送给用户邮箱的验证挑战，用于无密码验证码登录或敏感身份核验。
    通过唯一约束 `(email, purpose)` 确保同一邮箱在同一业务场景下仅存活最新的一组挑战。

    Attributes:
        id (int): 挑战记录自增主键。
        email (str): 目标邮箱地址（规范化小写，最长 254 字符）。
        purpose (str): 业务意图标识（例如 'login', 'register'，最长 16 字符）。
        code_hash (str): 随机生成的数字验证码经安全散列后的哈希摘要。
        sent_at (datetime): 验证码成功投递至 SMTP 服务的时间戳。
        expires_at (datetime): 验证码的生命周期截止时间戳（通常为 5~10 分钟）。
        attempts (int): 用户输入错误校验的累计失败次数（超限直接作废，防范穷举暴力破解）。
        consumed (bool): 是否已被成功消费标记（单次有效，不可重放）。
    """
    __tablename__ = 'email_auth_codes'
    __table_args__ = (UniqueConstraint('email', 'purpose'),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), index=True)
    purpose: Mapped[str] = mapped_column(String(16))
    code_hash: Mapped[str] = mapped_column(String(128))
    sent_at: Mapped[datetime] = mapped_column(DateTime)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    consumed: Mapped[bool] = mapped_column(Boolean, default=False)


class AuthRateLimit(Base):
    """安全认证接口速率限制持久化实体 (Authentication Rate Limiting Counter)

    提供跨进程和轻量持久化的接口频控滑动/固定窗口计数。

    Attributes:
        key (str): 频控维度键名（如 `ip:127.0.0.1:send_code` 或 `email:test@domain:login`，最长 64 字符）。
        count (int): 当前窗口内已累计的请求次数。
        expires_at (datetime): 当前频控统计窗口的过期重置时间戳。
    """
    __tablename__ = 'auth_rate_limits'
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
