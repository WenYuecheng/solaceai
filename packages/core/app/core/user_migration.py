"""SolaceAI 用户身份跨库增量数据迁移基础设施 (Inter-Database User Identity Migration)

本模块提供生产环境中在不同独立数据库（如全局历史库与特定隔离 Profile 数据库）间
安全同步与恢复用户身份实体的幂等迁移机制：
1. 增量导入：仅同步目标数据库中缺失的用户身份记录，已存在的完全一致记录自动安全跳过；
2. 强身份一致性冲突检测：严格校验用户名与邮箱的唯一映射约束。若发现同名账户绑定了不同邮箱，
   或相同邮箱归属于不同用户名，立即触发异常并中断操作；
3. 严格事务一致性保护：任何冲突或异常均触发目标数据库全局回滚 (`rollback`)，杜绝脏数据提交。
"""
from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.entities import UserAccount


def migrate_missing_users(source: Session, target: Session) -> dict[str, int]:
    """将源数据库中缺失的用户账号增量迁移复制至目标数据库 (Migrate Missing Users Safely)

    执行逻辑：
    1. 预先加载目标数据库全量现存用户的用户名映射表与邮箱映射表；
    2. 遍历源数据库中的所有 `UserAccount` 实体；
    3. 冲突校验：
       - 若用户名已存在且邮箱一致，标记为已存在 (`skipped`) 并跳过；
       - 若用户名已存在但邮箱不一致，或邮箱已存在但归属另一用户名，抛出 `ValueError` 标识身份冲突；
    4. 对通过校验的缺失用户构建实体副本并插入目标库；
    5. 执行目标数据库事务提交；发生异常时自动执行回滚。

    Args:
        source (Session): 源数据库活跃 SQLAlchemy 会话（只读拉取）。
        target (Session): 目标数据库活跃 SQLAlchemy 会话（写入迁移）。

    Returns:
        dict[str, int]: 包含迁移统计结果的字典，例如 `{'imported': 5, 'existing': 10}`。

    Raises:
        ValueError: 当检测到用户名与邮箱存在交叉冲突或身份不一致时抛出。
        Exception: 底层数据库提交发生异常时向外重新抛出。
    """
    # 建立目标库用户名与邮箱的快速内存查找映射
    existing = list(target.scalars(select(UserAccount)))
    by_name = {user.username: user for user in existing}
    by_email = {user.email.strip().lower(): user for user in existing if user.email}
    imported = skipped = 0

    try:
        for user in source.scalars(select(UserAccount)):
            email = user.email.strip().lower() if user.email else None
            same_name = by_name.get(user.username)
            same_email = by_email.get(email) if email else None

            # 用户名已存在时的冲突判定
            if same_name:
                current_email = same_name.email.strip().lower() if same_name.email else None
                if email and email != current_email:
                    raise ValueError('Account identity conflict; no changes committed')
                skipped += 1
                continue

            # 用户名不存在但邮箱已被其他用户占用的冲突判定
            if same_email:
                raise ValueError('Account identity conflict; no changes committed')

            # 构造隔离副本并追加至目标会话
            copy = UserAccount(
                username=user.username,
                email=email,
                requires_setup=user.requires_setup,
                display_name=user.display_name,
                password_hash=user.password_hash,
                roles_csv=user.roles_csv,
                created_at=user.created_at,
            )
            target.add(copy)
            by_name[copy.username] = copy
            if email:
                by_email[email] = copy
            imported += 1

        target.commit()
    except Exception:
        target.rollback()
        raise

    return {'imported': imported, 'existing': skipped}
