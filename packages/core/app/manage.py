"""SolaceAI 运维管理与系统账号初始化命令行工具 (CLI System Management & User Provisioning)

本模块提供面向系统运维人员与辅导员管理团队的终端命令行工具：
1. 安全凭据初始化：在脱离前端界面或首次系统布署时，通过交互式命令行安全创建管理员与学生账号；
2. 输入安全性校验：对用户名、显示名称、角色权限以及密码长度进行多维度合规审查；
3. 密码静默输入与确认：通过 `getpass` 避免明文回显并在终端进行二次比对；
4. 事务一致性保护：用户创建失败时自动执行数据库回滚。
"""

from __future__ import annotations

import argparse
import getpass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.bootstrap import create_schema
from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models.entities import UserAccount


def create_user(
    db: Session,
    username: str,
    display_name: str,
    password: str,
    role: str = 'student',
) -> UserAccount:
    """在数据库中创建并持久化新的系统用户账号 (Provision New User Account)

    执行前置校验并对密码进行 Argon2id 安全散列，随后持久化至 `user_accounts` 表。

    Args:
        db (Session): 当前活动的 SQLAlchemy 会话。
        username (str): 登录用户名（学号/工号），长度范围 1~64 字符。
        display_name (str): 真实姓名或系统显示名称，长度范围 1~128 字符。
        password (str): 原始明文密码，长度范围 8~128 字符。
        role (str, optional): 分配的角色类型，可选 'student' 或 'admin'，默认为 'student'。

    Returns:
        UserAccount: 持久化并刷新后的用户 ORM 实体对象。

    Raises:
        ValueError: 当参数格式不合法或用户名已存在时抛出。
        Exception: 数据库提交发生异常时向外抛出。
    """
    username = username.strip()
    display_name = display_name.strip()

    # 1. 严格校验字段长度与角色合法性
    if (
        role not in {'student', 'admin'}
        or not 1 <= len(username) <= 64
        or not 1 <= len(display_name) <= 128
        or not 8 <= len(password) <= 128
    ):
        raise ValueError('账号、显示名、角色或密码长度不符合要求')

    # 2. 检查用户名唯一性，防止主键冲突
    if db.scalar(select(UserAccount).where(UserAccount.username == username)):
        raise ValueError('用户名已存在')

    # 3. 构造用户实体并基于 Argon2id 散列明文密码
    user = UserAccount(
        username=username,
        display_name=display_name,
        password_hash=hash_password(password),
    )
    # 按角色分配 Spring Security 兼容的权限集合
    user.roles = {'ROLE_USER', 'ROLE_ADMIN'} if role == 'admin' else {'ROLE_USER'}
    db.add(user)

    # 4. 显式事务控制与异常回滚保护
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(user)
    return user


def main() -> None:
    """运维 CLI 命令行调度入口 (Command-Line Interface Entrypoint)

    解析命令行传入的 action 与用户属性，引导管理员完成密码静默交互输入，
    确保表结构已建立后执行账号落库。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create-user'], help='执行的操作类型')
    parser.add_argument('--username', required=True, help='待创建的登录用户名')
    parser.add_argument('--display-name', required=True, help='用户显示名称')
    parser.add_argument('--role', choices=['student', 'admin'], default='student', help='用户角色权限 (student 或 admin)')
    args = parser.parse_args()

    # 交互式隐藏输入密码，防御旁路窥视
    password = getpass.getpass('密码（8至128字符）：')
    if password != getpass.getpass('再次输入密码：'):
        parser.error('两次密码不一致')

    # 确保数据库表结构与增量迁移已执行完毕
    create_schema()
    with SessionLocal() as db:
        create_user(db, args.username, args.display_name, password, args.role)
    print('账号已创建')


if __name__ == '__main__':
    main()
