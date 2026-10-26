"""Authenticated credential setup, password changes and verified email binding."""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import Field, model_validator
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError

from app.api.auth_routes import DB, EmailPayload, SafeAuthRoute, StrictPayload
from app.core.config import get_settings
from app.core.security import current_user, hash_password, verify_password
from app.models.auth import AuthSession
from app.models.entities import UserAccount
from app.services.auth import consume_code, issue_code, profile, request_budget, set_session
from app.services.mailer import MailService

router = APIRouter(route_class=SafeAuthRoute)
User = Annotated[UserAccount, Depends(current_user)]


class NewPassword(StrictPayload):
    password: str = Field(min_length=8, max_length=128)
    confirmPassword: str = Field(min_length=8, max_length=128)

    @model_validator(mode='after')
    def confirm(self):
        if self.password != self.confirmPassword or not self.password.strip():
            raise ValueError('密码确认不一致')
        return self


class SetupPayload(NewPassword):
    username: str = Field(min_length=3, max_length=64, pattern=r'^[A-Za-z0-9_\-]+$')


class ChangePassword(NewPassword):
    currentPassword: str = Field(min_length=1, max_length=128)


class BindEmail(EmailPayload):
    currentPassword: str = Field(min_length=1, max_length=128)


class ConfirmEmail(BindEmail):
    code: str = Field(pattern=r'^\d{6}$')


def locked_user(db, user):
    return db.scalar(select(UserAccount).where(UserAccount.id == user.id).with_for_update().execution_options(populate_existing=True))


def require_password(user, password):
    if not verify_password(password, user.password_hash):
        raise HTTPException(401, '原密码不正确')


@router.post('/setup')
def setup(payload: SetupPayload, request: Request, response: Response, db: DB, user: User):
    request_budget(db, request, 'setup', str(user.id))
    user = locked_user(db, user)
    if not user.requires_setup:
        raise HTTPException(409, '账号已完成设置')
    values = dict(username=payload.username, password_hash=hash_password(payload.password), requires_setup=False)
    if user.display_name == '新同学':
        values['display_name'] = payload.username
    try:
        changed = db.execute(update(UserAccount).where(UserAccount.id == user.id, UserAccount.requires_setup.is_(True)).values(**values).execution_options(synchronize_session=False))
        if changed.rowcount != 1:
            db.rollback()
            raise HTTPException(409, '账号已完成设置')
        db.refresh(user)
        db.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
        set_session(db, request, response, user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, '用户名或邮箱已被使用') from None
    return profile(user)


@router.post('/password')
def change_password(payload: ChangePassword, request: Request, response: Response, db: DB, user: User):
    request_budget(db, request, 'change-password', str(user.id), limit=5)
    user = locked_user(db, user)
    require_password(user, payload.currentPassword)
    old_hash = user.password_hash
    next_hash = hash_password(payload.password)
    changed = db.execute(update(UserAccount).where(UserAccount.id == user.id, UserAccount.password_hash == old_hash).values(password_hash=next_hash).execution_options(synchronize_session=False))
    if changed.rowcount != 1:
        db.rollback()
        raise HTTPException(409, '账号凭据已更新，请重新登录')
    db.refresh(user)
    db.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
    set_session(db, request, response, user)
    return profile(user)


def check_binding(db, user, payload):
    require_password(user, payload.currentPassword)
    if user.email:
        raise HTTPException(409, '账号已绑定邮箱')
    if db.scalar(select(UserAccount.id).where(UserAccount.email == payload.email)):
        raise HTTPException(409, '用户名或邮箱已被使用')


@router.post('/email/code')
def binding_code(payload: BindEmail, request: Request, db: DB, user: User):
    request_budget(db, request, 'bind-code-user', str(user.id), limit=6, seconds=3600, ip_limit=30)
    request_budget(db, request, 'bind-code-email', payload.email, limit=6, seconds=3600, ip_limit=30)
    check_binding(db, user, payload)
    purpose = f'b{user.id}'
    code = issue_code(db, payload.email, purpose)
    try:
        result = MailService(get_settings()).send_code(payload.email, code, 'bind')
    except Exception:
        result = {'status': 'FAILED'}
    if result.get('status') != 'SMTP_ACCEPTED':
        # Invalidate failed deliveries so an undelivered challenge cannot be used.
        from sqlalchemy import update
        from app.models.auth import EmailAuthCode
        db.execute(update(EmailAuthCode).where(EmailAuthCode.email == payload.email, EmailAuthCode.purpose == purpose).values(consumed=True))
        db.commit()
        raise HTTPException(503, '邮件服务暂不可用，请稍后重试')
    return {'resendAfter': 60, 'expiresIn': 300}


@router.post('/email')
def bind_email(payload: ConfirmEmail, request: Request, response: Response, db: DB, user: User):
    request_budget(db, request, 'bind-email', str(user.id), limit=10)
    user = locked_user(db, user)
    check_binding(db, user, payload)
    consume_code(db, payload.email, f'b{user.id}', payload.code)
    try:
        changed = db.execute(update(UserAccount).where(UserAccount.id == user.id, UserAccount.email.is_(None), UserAccount.password_hash == user.password_hash).values(email=payload.email).execution_options(synchronize_session=False))
        if changed.rowcount != 1:
            db.rollback()
            raise HTTPException(409, '账号凭据已更新，请重新登录')
        db.refresh(user)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, '用户名或邮箱已被使用') from None
    response.headers['Cache-Control'] = 'no-store'
    return profile(user)
