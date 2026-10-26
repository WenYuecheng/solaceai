"""SolaceAI 邮件预警重试持久化租约协议 (Durable Mail Delivery Lease Protocol)

本模块实现了在现有数据库审计字段中轻量编码排他性任务租约的协议机制：
1. 解决非原子性分布式投递风险：由于 SMTP 服务器接收邮件与本地数据库事务提交无法做到两阶段原子性，
   当网络闪断或服务进程崩溃时，可能会遗留处于 `RUNNING` 状态的未知投递记录；
2. 轻量嵌入现有审计字段：无需增加额外数据表结构，直接利用 `AlertRecord.message` 列以 JSON 结构
   持久化存储租约所有者、过期时间戳以及中断恢复标识；
3. 显式人工介入保障：租约到期后强制需要操作员在确认可能产生重复投递的前提下显式恢复，杜绝后台死循环盲目重发。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone


def new_lease(actor: str, timeout_seconds: float, recovered: bool) -> str:
    """生成包含过期时间与操作人标识的持久化排他租约载荷 (Generate New Delivery Lease)

    租约时长依据底层 SMTP 超时参数动态推算，设定为 `max(300, timeout_seconds * 6 + 60)` 秒，
    确保即便网络发生最大重传延迟，也不会发生租约提前失效导致的并发抢占。

    Args:
        actor (str): 执行重试操作的管理员用户名。
        timeout_seconds (float): 系统配置的单次 SMTP 请求超时时间（秒）。
        recovered (bool): 是否属于中断任务的人工确认恢复。

    Returns:
        str: 序列化为 JSON 字符串的租约数据。
    """
    expiry = datetime.now(timezone.utc) + timedelta(seconds=max(300, timeout_seconds * 6 + 60))
    return json.dumps({
        'mailRetryLease': expiry.isoformat(),
        'actor': actor,
        'recoveredInterrupted': recovered,
    }, ensure_ascii=False)


def lease_expired(message: str) -> bool:
    """判断给定的预警消息字符串中嵌入的租约是否已经超时失效 (Check Lease Expiration)

    执行逻辑：
    1. 尝试将文本解析为 JSON 字典；
    2. 提取 `mailRetryLease` 字段并转换为带有时区信息的 UTC datetime；
    3. 比对当前 UTC 时间，若当前时间已超过租约过期时间则返回 True；
    4. 若格式无效或为不含租约元数据的历史记录，保守返回 False（需人工介入）。

    Args:
        message (str): 预警记录中的消息描述文本。

    Returns:
        bool: 若租约已明确过期返回 True，否则返回 False。
    """
    try:
        expiry = datetime.fromisoformat(json.loads(message)['mailRetryLease'])
        if expiry.tzinfo is None:
            return False
        return expiry <= datetime.now(timezone.utc)
    except (ValueError, TypeError, KeyError):
        # 缺少租约元数据的旧版 RUNNING 记录需要人工排查，不自动判定为过期
        return False
