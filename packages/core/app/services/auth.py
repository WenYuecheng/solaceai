"""SolaceAI 认证事务频控与一次性凭据管理服务 (Authentication Transactions & Credential Management)

本模块为系统身份安全层提供事务级频控与敏感凭据全生命周期管理：
1. 关系型数据库原子限流 (`rate_limit`, `request_budget`)：
   - 基于时间窗口分片与保存点（Savepoint / `begin_nested`），支持多 Worker 进程并发下的无锁竞争频控；
   - 自动级联限制客户端来源 IP 与操作目标实体（邮箱/用户名），防范暴力破解与短信邮件资源滥用；
2. 一次性安全验证码颁发与消费 (`issue_code`, `consume_code`)：
   - 采用密码学级安全随机数 (`secrets.randbelow`) 与 Argon2 散列持久化；
   - 强制 60 秒冷却发送防重发，限制 5 次最大错误尝试次数，消费与校验行锁绑定，彻底根除并发重放竞争条件；
3. 不透明会话生成与安全 Cookie 管理 (`set_session`)：
   - 颁发高熵 32 字节随机 URL-safe Token，仅在数据库持久化其 SHA-256 哈希散列；
   - 注入配置完备的客户端安全响应头（`HttpOnly`, `SameSite=Lax`, `Secure` 条件判断，`Cache-Control: no-store`）。
"""
import hashlib
import secrets
from datetime import timedelta
from typing import Any

from fastapi import HTTPException, Request, Response
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.auth import AuthRateLimit, AuthSession, EmailAuthCode
from app.models.entities import UserAccount, now
from app.core.security import hash_password, verify_password

# 客户端 Session 对应的 Cookie 键名
COOKIE_NAME: str = 'solace_session'
# 登录会话绝对生命周期时长（8 小时）
SESSION_SECONDS: int = 8 * 60 * 60


def digest(value: str) -> str:
    """计算给定字符串的标准 SHA-256 十六进制哈希散列 (Compute SHA-256 Hex Digest)

    用于将客户端原始 Session Token 转换为数据库可持久化的不可逆索引键，
    防止数据库明文泄露导致所有在线用户被会话劫持。

    Args:
        value (str): 输入的明文字符串。

    Returns:
        str: 64 位十六进制小写哈希字符串。
    """
    return hashlib.sha256(value.encode()).hexdigest()


def rate_limit(db: Session, namespace: str, subject: str, limit: int, seconds: int) -> None:
    """基于关系型数据库的原子固定窗口接口速率限制器 (Atomic Fixed-Window Rate Limiter)

    执行逻辑：
    1. 根据当前时间戳除以窗口秒数计算当前时间分片 (`window`)；
    2. 基于 `namespace:subject:window` 生成确定性 SHA-256 主键；
    3. 利用子事务（Savepoint）插入计数器初始行，若发生并发主键冲突则忽略以复用已有记录；
    4. 执行带条件约束的自增更新：`count < limit`；
    5. 若受影响行数不为 1，说明当前窗口配额已满，提交后抛出 HTTP 429 异常；
    6. 顺带物理清理已过期的历史频控记录，防止表体积无限膨胀。

    Args:
        db (Session): 数据库会话。
        namespace (str): 业务命名空间（例如 'send-code:ip', 'login:subject'）。
        subject (str): 频控目标对象（如客户端 IP 地址、邮箱地址或用户名）。
        limit (int): 当前时间窗口内允许的最大请求次数上限。
        seconds (int): 时间窗口持续时长（秒）。

    Raises:
        HTTPException(429): 当请求频率超限时抛出。
    """
    current = now()
    window = int(current.timestamp()) // seconds
    key = digest(f'{namespace}:{subject}:{window}')
    # 高并发请求下预期会发生创建主键冲突。保存点可以保留外层事务中已持有的所有配额
    try:
        with db.begin_nested():
            db.add(AuthRateLimit(key=key, count=0, expires_at=current + timedelta(seconds=seconds * 2)))
            db.flush()
    except IntegrityError:
        pass
    changed = db.execute(
        update(AuthRateLimit)
        .where(AuthRateLimit.key == key, AuthRateLimit.count < limit)
        .values(count=AuthRateLimit.count + 1)
    )
    db.commit()
    if changed.rowcount != 1:
        raise HTTPException(429, '请求过于频繁，请稍后再试')
    # 限制仅清理已过期的陈旧记录，绝不误删当前活动的频控配额
    db.execute(delete(AuthRateLimit).where(AuthRateLimit.expires_at < current))
    db.commit()


def request_budget(
    db: Session,
    request: Request,
    namespace: str,
    subject: str,
    limit: int = 10,
    seconds: int = 900,
    ip_limit: int = 60,
) -> None:
    """双维度联合接口频控防护 (Composite Request Budget Guard)

    同时对请求来源 IP 地址与业务主体（邮箱/用户名）施加双重时间窗口限流，
    有效防范单 IP 轮换用户名撞库或单用户名分布式代理穷举攻击。

    Args:
        db (Session): 数据库会话。
        request (Request): 当前 HTTP 请求对象（用于提取来源 client.host）。
        namespace (str): 业务操作命名空间。
        subject (str): 业务标识（邮箱/账号）。
        limit (int, optional): 业务主体限额，默认 10 次。
        seconds (int, optional): 窗口时长，默认 900 秒 (15 分钟)。
        ip_limit (int, optional): 来源 IP 限额，默认 60 次。
    """
    ip = request.client.host if request.client else 'unknown'
    rate_limit(db, namespace + ':ip', ip, ip_limit, seconds)
    rate_limit(db, namespace + ':subject', subject, limit, seconds)


def issue_code(db: Session, email: str, purpose: str) -> str:
    """生成并持久化一次性邮箱数字验证挑战码 (Issue Email Challenge Code)

    执行逻辑：
    1. 生成密码学安全的 6 位随机整数字符串（000000 ~ 999999）；
    2. 计算其 Argon2id 安全哈希散列；
    3. 检查重发冷却时间（60 秒）：尝试更新发送时间在 60 秒前的历史记录；
       若更新影响行数为 0 且表中已存在未满 60 秒的同意图记录，抛出 HTTP 429 限制重发；
    4. 提交事务并返回明文验证码用于后续邮件发送。

    Args:
        db (Session): 数据库会话。
        email (str): 目标接收邮箱。
        purpose (str): 验证用途意图（如 'login'）。

    Returns:
        str: 6 位明文数字验证码。

    Raises:
        HTTPException(429): 当距离上次发送未满 60 秒冷却期时抛出。
    """
    current = now()
    code = f'{secrets.randbelow(1000000):06d}'
    values = dict(
        code_hash=hash_password(code),
        sent_at=current,
        expires_at=current + timedelta(minutes=5),
        attempts=0,
        consumed=False,
    )
    changed = db.execute(
        update(EmailAuthCode)
        .where(
            EmailAuthCode.email == email,
            EmailAuthCode.purpose == purpose,
            EmailAuthCode.sent_at <= current - timedelta(seconds=60),
        )
        .values(**values)
    )
    if not changed.rowcount:
        if db.scalar(select(EmailAuthCode.id).where(EmailAuthCode.email == email, EmailAuthCode.purpose == purpose)):
            db.rollback()
            raise HTTPException(429, '请等待60秒后重新发送验证码')
        db.add(EmailAuthCode(email=email, purpose=purpose, **values))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(429, '请等待60秒后重新发送验证码') from None
    return code


def consume_code(db: Session, email: str, purpose: str, code: str) -> None:
    """原子核验并消费一次性邮箱挑战码 (Atomically Consume Email Code)

    安全机制：
    1. 条件更新与行锁：在单一写事务中更新尝试次数 `attempts = attempts + 1`，
       条件包含：未过期 (`expires_at > now()`)、尝试未超限 (`attempts < 5`)、尚未被消费 (`consumed == False`)；
    2. 比对哈希散列：更新成功后在同一事务连接下拉取散列执行 `verify_password` 校验；
    3. 校验失败时立即 `commit` 保存失败尝试计数，防止回滚重试穷举；
    4. 校验成功后立即更新 `consumed = True`，阻止并发重放。

    Args:
        db (Session): 数据库会话。
        email (str): 目标邮箱。
        purpose (str): 挑战用途。
        code (str): 用户输入的 6 位验证码。

    Raises:
        HTTPException(401): 验证码错误、已过期、已被消费或失败超限。
    """
    # 条件 UPDATE 在提交前持有排他写锁，防止并发竞争消费
    changed = db.execute(
        update(EmailAuthCode).where(
            EmailAuthCode.email == email,
            EmailAuthCode.purpose == purpose,
            EmailAuthCode.expires_at > now(),
            EmailAuthCode.attempts < 5,
            EmailAuthCode.consumed.is_(False),
        ).values(attempts=EmailAuthCode.attempts + 1)
        .execution_options(synchronize_session=False)
    )
    challenge = db.execute(
        select(EmailAuthCode.code_hash, EmailAuthCode.id).where(
            EmailAuthCode.email == email, EmailAuthCode.purpose == purpose,
        )
    ).first() if changed.rowcount == 1 else None

    valid = challenge is not None and verify_password(code, challenge.code_hash)
    if not valid:
        # 失败尝试必须持久化提交，防止恶意调用方通过回滚规避 5 次错误尝试上限
        db.commit()
        raise HTTPException(401, '验证码无效或已过期')

    db.execute(update(EmailAuthCode).where(EmailAuthCode.id == challenge.id).values(consumed=True))


def set_session(db: Session, request: Request, response: Response, user: UserAccount) -> None:
    """为认证通过的用户颁发并持久化安全浏览器会话 (Establish Authenticated Session)

    执行逻辑：
    1. 若请求中携带旧 Session Cookie，立即从数据库硬删除对应的会话映射；
    2. 生成新的 32 字节密码学高熵 URL-safe 随机字符串；
    3. 清理全库已超时的废弃会话；
    4. 将新 Token 的 SHA-256 散列写入 `auth_sessions` 表，绑定目标 `user_id` 并设定 8 小时有效期；
    5. 在 HTTP 响应头中注入 Set-Cookie，附加 `HttpOnly`, `SameSite=Lax`, 以及条件 `Secure` 属性；
    6. 附加 `Cache-Control: no-store` 防止会话响应被中间代理缓存。

    Args:
        db (Session): 数据库会话。
        request (Request): HTTP 请求对象。
        response (Response): HTTP 响应对象。
        user (UserAccount): 当前登录认证的用户实体。
    """
    old = request.cookies.get(COOKIE_NAME)
    if old:
        db.execute(delete(AuthSession).where(AuthSession.token_hash == digest(old)))
    token = secrets.token_urlsafe(32)
    db.execute(delete(AuthSession).where(AuthSession.expires_at < now()))
    db.add(AuthSession(token_hash=digest(token), user_id=user.id, expires_at=now() + timedelta(seconds=SESSION_SECONDS)))
    db.commit()
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=SESSION_SECONDS,
        httponly=True,
        secure=request.url.scheme == 'https',
        samesite='lax',
        path='/',
    )
    response.headers['Cache-Control'] = 'no-store'


def profile(user: UserAccount) -> dict[str, Any]:
    """提取面向前端展示的用户脱敏个人公开档案数据 (Extract Public Profile)

    Args:
        user (UserAccount): 用户 ORM 实体对象。

    Returns:
        dict[str, Any]: 包含 id, username, displayName, roles 等公开属性的字典。
    """
    from app.schemas.dtos import authority
    return dict(
        id=user.id,
        username=user.username,
        email=user.email,
        requiresSetup=user.requires_setup,
        displayName=user.display_name,
        roles=[authority(role) for role in user.roles],
    )
