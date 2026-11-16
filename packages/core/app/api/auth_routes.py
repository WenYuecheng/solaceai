"""SolaceAI 学生认证与会话管理路由模块 (Authentication & Session Routes)

本模块为学生端及通用认证流程提供安全的 HTTP 接入网关，涵盖：
1. 邮箱免密验证码下发 (`POST /api/auth/code`)：内置 IP 与邮箱双重频控，防止短信/邮件炸弹与撞库嗅探；
2. 账号密码凭据认证 (`POST /api/auth/login`)：基于 Argon2id 散列，防时序侧信道推测账号存在性；
3. 邮箱验证码直登/自动建档 (`POST /api/auth/email-login`)：一次性验证码消费，支持新学生零门槛安全注册；
4. 会话安全注销 (`POST /api/auth/logout`)：物理删除服务端 Session 并清理客户端 HttpOnly Cookie；
5. 敏感信息防回显安全路由 (`SafeAuthRoute`)：拦截参数校验异常，杜绝在报错信息中反射用户提交的明文密码或验证码。
"""
import re
import secrets
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import check_origin, hash_password, verify_password
from app.models.auth import AuthSession, EmailAuthCode
from app.models.entities import UserAccount
from app.services.auth import COOKIE_NAME, consume_code, digest, issue_code, profile, request_budget, set_session
from app.services.mailer import MailService


class SafeAuthRoute(APIRoute):
    """防敏感数据回显的安全认证路由类 (Safe Auth Route Handler)

    重写 FastAPI 默认的路由执行句柄，当入参验证失败抛出 `RequestValidationError` 时，
    拦截默认包含请求体反射的错误细节，统一替换为安全提示，杜绝明文密码或验证码回显至响应。
    """

    def get_route_handler(self):
        """包装路由请求处理函数，捕获并擦除参数校验反射信息。"""
        handler = super().get_route_handler()

        async def safe_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError:
                raise HTTPException(422, '请检查输入信息及密码确认') from None

        return safe_handler


router = APIRouter(
    prefix='/api/auth',
    tags=['auth'],
    dependencies=[Depends(check_origin)],
    route_class=SafeAuthRoute,
)
DB = Annotated[Session, Depends(get_db)]


class StrictPayload(BaseModel):
    """禁止额外字段的严格请求载荷基类 (Strict Forbidden Extra Base DTO)"""
    model_config = ConfigDict(extra='forbid')


class EmailPayload(StrictPayload):
    """邮箱基础输入载荷 (Email Base Payload DTO)

    Attributes:
        email (str): 用户的电子邮箱地址，统一清洗为小写。
    """
    email: str = Field(min_length=3, max_length=254, description="电子邮箱地址")

    @field_validator('email')
    @classmethod
    def normalize_email(cls, value: str) -> str:
        """清洗并验证邮箱地址格式，防范非法字符与连续点注入。"""
        value = value.strip().lower()
        if not re.fullmatch(r'[A-Za-z0-9.!#$%&\x27*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,63}', value) or '..' in value:
            raise ValueError('请输入有效邮箱地址')
        return value


class CodePayload(EmailPayload):
    """发送验证码请求载荷 (Send Verification Code DTO)

    Attributes:
        purpose (str): 验证码业务意图，目前锁定为 'login'。
    """
    purpose: Literal['login'] = 'login'


class EmailLoginPayload(EmailPayload):
    """邮箱验证码登录请求载荷 (Email Login Request DTO)

    Attributes:
        code (str): 6 位数字验证码。
    """
    code: str = Field(pattern=r'^\d{6}$', description="6位数字验证码")


class PasswordPayload(StrictPayload):
    """账号密码登录请求载荷 (Password Login Request DTO)

    Attributes:
        username (str): 登录用户名（学号/工号）。
        password (str): 登录原始明文密码。
    """
    username: str = Field(min_length=1, max_length=254, description="用户名或邮箱")
    password: str = Field(min_length=1, max_length=128, description="用户密码")


def student(user: UserAccount | None) -> bool:
    """判定目标用户是否仅具备普通学生权限 (Is Pure Student Role)

    用于权限沙箱隔离：管理员账号禁止通过学生专属的免密验证码通道直接登录，防范越权风险。

    Args:
        user (UserAccount | None): 用户实体对象。

    Returns:
        bool: 若用户仅拥有 ROLE_USER 角色则返回 True，否则返回 False。
    """
    return user is not None and set(user.roles) == {'ROLE_USER'}


@router.post('/code')
def send_code(payload: CodePayload, request: Request, db: DB) -> dict[str, Any]:
    """申请向指定邮箱发送登录/核验数字验证码 (Send Email Auth Code)

    执行逻辑：
    1. 频控审查：针对目标邮箱（1小时限6次）与来源 IP（1小时限30次）执行滑动窗口限流；
    2. 目标账户核验：查询数据库，仅允许未注册邮箱或普通学生账号触发邮件发送；
       若为管理员邮箱或不存在账号，统一返回一致的成功响应形态，防御账号嗅探侧信道；
    3. 生成加盐散列并写入 `email_auth_codes` 表，调用 SMTP 客户端发送邮件；
    4. 邮件发送失败时立即废弃该验证码并抛出 503 异常。

    Args:
        payload (CodePayload): 目标邮箱与用途载荷。
        request (Request): 当前 HTTP 请求对象。
        db (Session): 数据库会话。

    Returns:
        dict: 统一的响应字典，提示用户重试间隔与有效期。
    """
    request_budget(db, request, 'send-code', payload.email, limit=6, seconds=3600, ip_limit=30)
    user = db.scalar(select(UserAccount).where(UserAccount.email == payload.email))
    eligible = student(user) or (user is None and get_settings().runtime_profile not in {'online', 'light'})
    # 对未注册用户与特权管理员保持相同的成功响应形态，防范账号探测
    if eligible:
        code = issue_code(db, payload.email, payload.purpose)
        try:
            result = MailService(get_settings()).send_code(payload.email, code, payload.purpose)
        except Exception:
            result = {'status': 'FAILED'}
        if result.get('status') != 'SMTP_ACCEPTED':
            db.execute(update(EmailAuthCode).where(EmailAuthCode.email == payload.email, EmailAuthCode.purpose == payload.purpose).values(consumed=True))
            db.commit()
            raise HTTPException(503, '邮件服务暂不可用，请稍后重试')
    return {'message': '如邮箱可用于学生登录，验证码将发送至该邮箱', 'resendAfter': 60, 'expiresIn': 300}


@router.post('/login')
def password_login(payload: PasswordPayload, request: Request, response: Response, db: DB) -> dict[str, Any]:
    """账号与密码凭据登录认证端点 (Password Authentication Login)

    执行逻辑：
    1. 频控限制：单用户名与单 IP 分钟级速率防爆破限制；
    2. 用户检索与恒定时间散列校验：若用户不存在，使用预置虚拟散列 `_DUMMY_HASH` 执行真实耗时的
       Argon2 校验，从算法耗时层面彻底抹平用户存在与否的时序差异；
    3. 凭据校验成功后，生成随机 64 位不透明 Session Token，建立服务端会话并在响应中注入 HttpOnly Cookie。

    Args:
        payload (PasswordPayload): 用户名与明文密码。
        request (Request): 当前 HTTP 请求对象。
        response (Response): 用于注入 Set-Cookie 的响应对象。
        db (Session): 数据库会话。

    Returns:
        dict: 用户公开个人档案数据 (Profile)。
    """
    identifier = payload.username.strip()
    request_budget(db, request, 'password-login', identifier.lower() if '@' in identifier else identifier, limit=10, ip_limit=60)
    criterion = UserAccount.email == identifier.lower() if '@' in identifier else UserAccount.username == identifier
    user = db.scalar(select(UserAccount).where(criterion))
    # 针对未知用户名也执行真实哈希校验，抹平时间侧信道特征
    candidate_hash = user.password_hash if user else _DUMMY_HASH
    if not verify_password(payload.password, candidate_hash) or user is None:
        raise HTTPException(401, '用户名或密码错误')
    set_session(db, request, response, user)
    return profile(user)


@router.post('/email-login')
def email_login(payload: EmailLoginPayload, request: Request, response: Response, db: DB) -> dict[str, Any]:
    """邮箱验证码免密登录与新学生即时建档端点 (Email Code Login & Auto-Provision)

    执行逻辑：
    1. 频控限制：单邮箱与单 IP 频次审查；
    2. 校验并原子消费验证码 (`consume_code`)，单次有效不可重放；
    3. 若对应邮箱尚未注册用户，安全创建随机初始学号与高强度随机密码的 `UserAccount` 记录；
    4. 角色权限校验：禁止管理员通过该通道免密登录；
    5. 生成服务端持久化会话并写入客户端 HttpOnly Cookie。

    Args:
        payload (EmailLoginPayload): 邮箱与 6 位数字验证码。
        request (Request): 当前 HTTP 请求对象。
        response (Response): 用于注入 Set-Cookie 的响应对象。
        db (Session): 数据库会话。

    Returns:
        dict: 用户公开个人档案数据 (Profile)。
    """
    request_budget(db, request, 'email-login', payload.email, limit=15, ip_limit=60)
    consume_code(db, payload.email, 'login', payload.code)
    user = db.scalar(select(UserAccount).where(UserAccount.email == payload.email))
    if user is None and get_settings().runtime_profile in {'online', 'light'}:
        db.commit()
        raise HTTPException(403, '根据有关要求暂不开放注册，实际账号体验请联系管理员。')
    if user is None:
        # 验证码核验通过证明其对邮箱的所有权，自动创建新学生身份
        # 丢弃随机生成的强密码，对外不暴露默认密码
        try:
            with db.begin_nested():
                user = UserAccount(
                    username='student_' + secrets.token_hex(12),
                    display_name='新同学',
                    requires_setup=True,
                    email=payload.email,
                    password_hash=hash_password(secrets.token_urlsafe(48)),
                    roles_csv='ROLE_USER',
                )
                db.add(user)
                db.flush()
        except IntegrityError:
            # 应对并发注册竞争同一邮箱的极端边缘情况
            user = db.scalar(select(UserAccount).where(UserAccount.email == payload.email).with_for_update())
    if not student(user):
        db.commit()
        raise HTTPException(401, '验证码无效或已过期')
    set_session(db, request, response, user)
    return profile(user)


@router.post('/logout')
def logout(request: Request, response: Response, db: DB) -> dict[str, str]:
    """用户会话主动安全注销端点 (User Session Logout)

    执行逻辑：
    1. 从请求 Cookie 中读取会话 Token；
    2. 计算其 SHA-256 散列并从 `auth_sessions` 数据表中物理硬删除；
    3. 清除客户端浏览器上的会话 Cookie，并显式附加 `Cache-Control: no-store`。

    Args:
        request (Request): 当前 HTTP 请求对象。
        response (Response): 响应对象。
        db (Session): 数据库会话。

    Returns:
        dict[str, str]: 操作完成确认消息。
    """
    token = request.cookies.get(COOKIE_NAME)
    if token:
        db.execute(delete(AuthSession).where(AuthSession.token_hash == digest(token)))
        db.commit()
    response.delete_cookie(COOKIE_NAME, httponly=True, secure=request.url.scheme == 'https', samesite='lax', path='/')
    response.headers['Cache-Control'] = 'no-store'
    return {'message': '已退出登录'}


# 预计算的恒定时间防侧信道虚拟哈希密文
_DUMMY_HASH = hash_password('opaque-dummy-password-not-a-user')


from app.api.account_routes import router as account_router
router.include_router(account_router)
