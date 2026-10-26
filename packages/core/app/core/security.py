"""SolaceAI 安全鉴权与访问控制模块 (Security, Authentication & RBAC)

本模块构建全系统的安全防护体系与基于角色的访问控制（RBAC）：
1. 凭据加盐与哈希：采用通过密码学前沿竞赛认证的 Argon2id 内存硬化散列算法，防御 GPU/ASIC 离线彩虹表攻击；
2. 历史兼容演进：登录校验时提供对早期 SHA-256 哈希凭据的恒定时间（Constant-Time）对比兼容；
3. HTTP Basic 凭据解析：从标准 Authorization 请求头中安全提取 base64 编码的登录凭据；
4. 依赖注入门禁：提供当前用户鉴权依赖 `current_user` 以及管理员权限门禁 `require_admin`。
"""

import base64
import hashlib
import hmac
from typing import Annotated

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import UserAccount


def hash_password(password: str) -> str:
    """计算明文密码的高强度安全散列密文 (Argon2id Hash)

    使用行业金标准的 Argon2id 算法生成密码哈希，内置随机盐（Salt）生成，
    具备抗 GPU 暴力破解与内存硬化（Memory-Hard）防护特性。

    Args:
        password (str): 用户输入的原始明文密码。

    Returns:
        str: 编码了算法参数、随机盐与哈希结果的标准 Argon2id 字符串 (以 '$argon2' 开头)。
    """
    return PasswordHasher().hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """安全校验明文密码与已存储的密码哈希密文是否匹配 (Constant-Time Verification)

    双重验证策略：
    1. 优先校验 Argon2 格式密文：捕获 `VerificationError` 与 `InvalidHashError`，校验失败返回 False；
    2. 历史向下兼容校验：若非 Argon2 格式，则假定为旧版 SHA-256 哈希，
       通过 `hmac.compare_digest` 进行恒定时间比较以防御时序侧信道攻击（Timing Attack）。

    Args:
        password (str): 待校验的用户输入的明文密码。
        hashed (str): 数据库中持久化的密码哈希密文。

    Returns:
        bool: 匹配成功返回 True，密码错误或密文格式非法返回 False。
    """
    # 优先校验标准 Argon2 密码格式
    if hashed.startswith('$argon2'):
        try:
            return PasswordHasher().verify(hashed, password)
        except (VerificationError, InvalidHashError):
            return False

    # 兼容历史 SHA-256 密码哈希账号登录；采用 hmac.compare_digest 防御时序侧信道攻击
    return hmac.compare_digest(
        hashlib.sha256(password.encode('utf-8')).hexdigest(),
        hashed,
    )


def _credentials(request: Request) -> tuple[str, str]:
    """从 HTTP Authorization 请求头中解析提取 Basic 认证凭证 (Extract HTTP Basic Credentials)

    Args:
        request (Request): 当前 FastAPI HTTP 请求对象。

    Returns:
        tuple[str, str]: 解析出的 (username, password) 凭据二元组。

    Raises:
        HTTPException(401): 请求头未携带 Authorization、不符合 Basic 规范或 Base64 格式损坏时抛出。
    """
    # 接口鉴权失败保留在单页应用登录流程中处理；若在此处返回 WWW-Authenticate: Basic 响应头，
    # 会导致主流浏览器在会话 Cookie 恢复时弹出原生弹窗提示；显式传入的 Basic 凭据仍向后兼容已有 API 客户端
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Basic authorization",
        )
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        username, password = decoded.split(":", 1)
        return username, password
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Basic authorization",
        ) from exc


def current_user(request: Request, db: Annotated[Session, Depends(get_db)]) -> UserAccount:
    """FastAPI 鉴权依赖项：解析并校验当前登录用户的有效身份 (Current User Dependency)

    执行流程：
    1. 从请求头解析 Basic 凭据；
    2. 按 username 检索数据库用户实体；
    3. 调用 `verify_password` 执行安全密码对比；
    4. 鉴权通过后返回当前登录的用户实体。

    Args:
        request (Request): 当前 FastAPI 请求上下文。
        db (Session): 数据库会话依赖项。

    Returns:
        UserAccount: 已完成鉴权认真的当前用户 ORM 实体对象。

    Raises:
        HTTPException(401): 用户名不存在或密码错误时统一抛出 401 Bad credentials。
    """
    from app.models.auth import AuthSession
    from app.models.entities import now
    from app.services.auth import COOKIE_NAME, digest, request_budget
    if not request.headers.get("authorization") and request.cookies.get(COOKIE_NAME):
        token = request.cookies[COOKIE_NAME]
        session = db.get(AuthSession, digest(token))
        if session is None or session.expires_at <= now():
            raise HTTPException(401, "登录已过期，请重新登录")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            check_origin(request)
        user = db.get(UserAccount, session.user_id)
        if user is None:
            raise HTTPException(401, "登录已过期，请重新登录")
        _check_setup(user, request)
        return user
    username, password = _credentials(request)
    user = db.query(UserAccount).filter(UserAccount.username == username).first()
    if user is None or not verify_password(password, user.password_hash):
        request_budget(db, request, 'basic-login', username, limit=10, ip_limit=60)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bad credentials",
        )
    _check_setup(user, request)
    return user


def _check_setup(user: UserAccount, request: Request) -> None:
    if user.requires_setup and request.url.path not in {'/api/profile', '/api/auth/setup', '/api/auth/logout'}:
        raise HTTPException(403, '请先设置用户名和密码')


def require_admin(user: Annotated[UserAccount, Depends(current_user)]) -> UserAccount:
    """FastAPI 权限门禁依赖项：强制要求管理员/心理辅导员角色权限 (Require Admin RBAC Dependency)

    在 `current_user` 鉴权成功的基础上，进一步核查用户角色列表中是否包含 'ROLE_ADMIN'。

    Args:
        user (UserAccount): 当前已成功登录的用户实体。

    Returns:
        UserAccount: 拥有管理员权限的用户实体。

    Raises:
        HTTPException(403): 缺少 ROLE_ADMIN 权限时抛出 403 Forbidden。
    """
    if "ROLE_ADMIN" not in user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


def check_origin(request: Request) -> None:
    """同源浏览器写请求与防跨站请求伪造 (CSRF) 审查门禁 (Check Origin & Custom Header)

    安全机制：
    1. Origin 校验：若请求包含 Origin 请求头，必须严格与当前服务 scheme://netloc 一致；
    2. 自定义请求头兜底：若缺少 Origin 头（例如部分同源 POST），必须携带 `X-Requested-With: SolaceAI`；
    3. Sec-Fetch-Site 校验：严格阻断被浏览器标记为跨站来源 (`cross-site`) 的非幂等写入请求。

    Args:
        request (Request): 当前进入的 HTTP 请求对象。

    Raises:
        HTTPException(403): 当请求来源不合规或检测到潜在 CSRF 跨站伪造时抛出。
    """
    origin = request.headers.get('origin')
    expected = f'{request.url.scheme}://{request.url.netloc}'
    if origin is not None:
        if origin != expected:
            raise HTTPException(403, '请求来源无效')
    elif request.headers.get('X-Requested-With') != 'SolaceAI':
        raise HTTPException(403, '请求来源无效')
    if request.headers.get('sec-fetch-site') == 'cross-site':
        raise HTTPException(403, '请求来源无效')
