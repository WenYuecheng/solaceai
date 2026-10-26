"""SolaceAI 外部工具与服务弹性调用治理核心模块 (Resilience & Fault Tolerance Service)

本模块实现智能体系统对外部工具与下游服务的弹性容灾治理体系：
1. 结构化故障分类 (FaultClassifier)：区分确定性错误（参数错误、Schema不匹配、权限不足）
   与瞬时可重试故障（HTTP 429、502/503/504、网络超时、连接重置）；
2. 受控退避重试 (execute_with_resilience)：对瞬时故障严格遵循 Retry-After 标头，
   并采用指数退避（Exponential Backoff）与随机抖动（Full Jitter），严格受单任务总时间预算守卫；
3. 三态断路熔断器 (CircuitBreaker)：维护 CLOSED -> OPEN -> HALF_OPEN 状态机，
   连续失败达阈值后立即阻断调用，防止雪崩冲击下游；并在冷却期后单探针试探自愈。
"""

from __future__ import annotations

import enum
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FaultCategory(str, enum.Enum):
    """故障类型分类枚举"""
    DETERMINISTIC_FATAL = "deterministic_fatal"  # 确定性不可重试错误（参数/Schema/权限等）
    TRANSIENT_RETRYABLE = "transient_retryable"  # 瞬时可重试故障（502/503/504/网络抖动）
    UPSTREAM_OVERLOAD = "upstream_overload"      # 下游显式过载/限流（429 或带 Retry-After）


class CircuitBreakerOpenException(RuntimeError):
    """断路器打开异常：下游服务熔断保护中，快速失败不冲击下游"""
    pass


class FaultClassifier:
    """根据异常特征进行结构化分类诊断"""

    @staticmethod
    def classify(exc: Exception) -> tuple[FaultCategory, str, float | None]:
        """对捕获的异常进行精准诊断分类。

        Args:
            exc: 捕获的异常实例。

        Returns:
            tuple[FaultCategory, str, float | None]: 
                - FaultCategory: 故障分类枚举
                - str: 规范化原因摘要
                - float | None: 服务端指示的等待秒数 (Retry-After)
        """
        # 1. 明确的客户端与业务确定性错误（原样重试 100% 失败）
        if isinstance(exc, (ValueError, TypeError, PermissionError, KeyError)):
            return FaultCategory.DETERMINISTIC_FATAL, str(exc), None

        # 2. HTTP 协议异常诊断 (支持 httpx.HTTPStatusError 以及包含 status_code 的 API 异常)
        status_code = getattr(exc, "status_code", None)
        headers = {}
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            headers = exc.response.headers
        elif hasattr(exc, "response") and hasattr(getattr(exc, "response"), "headers"):
            headers = getattr(exc.response, "headers") or {}

        if status_code is not None and isinstance(status_code, int):
            retry_after_header = headers.get("Retry-After") if hasattr(headers, "get") else None
            wait_seconds = None
            if retry_after_header:
                try:
                    wait_seconds = float(retry_after_header)
                except ValueError:
                    wait_seconds = None

            if status_code == 429:
                return (
                    FaultCategory.UPSTREAM_OVERLOAD,
                    f"HTTP 429 Too Many Requests: {exc}",
                    wait_seconds,
                )
            if status_code in {502, 503, 504}:
                return (
                    FaultCategory.TRANSIENT_RETRYABLE,
                    f"HTTP {status_code} Upstream Transient Error: {exc}",
                    wait_seconds,
                )
            if 400 <= status_code < 500:
                return (
                    FaultCategory.DETERMINISTIC_FATAL,
                    f"HTTP {status_code} Client Error: {exc}",
                    None,
                )
            return (
                FaultCategory.TRANSIENT_RETRYABLE,
                f"HTTP {status_code} Server Error: {exc}",
                wait_seconds,
            )

        # 3. 网络超时与连接层抖动
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return FaultCategory.TRANSIENT_RETRYABLE, f"Timeout: {exc}", None
        if isinstance(exc, (httpx.ConnectError, ConnectionError)):
            return FaultCategory.TRANSIENT_RETRYABLE, f"Network Connect Error: {exc}", None

        # 4. 其他未知异常：默认保守处理为确定性错误，避免盲目重试加重负担
        return FaultCategory.DETERMINISTIC_FATAL, f"{type(exc).__name__}: {exc}", None


class CircuitBreaker:
    """标准三态断路熔断器 (CLOSED, OPEN, HALF_OPEN)
    
    保护下游组件免受雪崩冲击。当连续失败次数达到 threshold 时转入 OPEN 状态拒绝调用；
    在 recovery_time 冷却窗口到期后自动转入 HALF_OPEN 状态放行单个探针试探健康。
    """

    def __init__(self, failure_threshold: int = 3, recovery_time: float = 30.0):
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_time = max(0.01, recovery_time)
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_state_change = time.monotonic()

    def can_execute(self) -> bool:
        """检查当前断路器是否允许放行调用请求。"""
        now = time.monotonic()
        if self.state == "OPEN":
            if now - self.last_state_change >= self.recovery_time:
                logger.info("Circuit breaker cooldown expired; switching to HALF_OPEN probe state.")
                self.state = "HALF_OPEN"
                self.last_state_change = now
                return True
            return False
        return True

    def record_success(self) -> None:
        """记录一次成功调用：重置计数器并将半开状态恢复为 CLOSED。"""
        if self.state == "HALF_OPEN":
            logger.info("Probe succeeded; resetting circuit breaker from HALF_OPEN to CLOSED.")
        self.state = "CLOSED"
        self.failure_count = 0

    def record_failure(self) -> None:
        """记录一次瞬时故障调用：累加计数并在达到阈值时触发熔断。"""
        self.failure_count += 1
        if self.state == "HALF_OPEN" or self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_state_change = time.monotonic()
            logger.warning(
                "Circuit breaker tripped to OPEN! Consecutive failures: %d. Blocking calls for %.1fs.",
                self.failure_count,
                self.recovery_time,
            )

    def reset(self) -> None:
        """手动重置断路器至初始闭合状态。"""
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_state_change = time.monotonic()


@dataclass
class TaskBudget:
    """单次 Agent 任务或工具执行周期的总体时间预算守卫。"""
    deadline: float

    @classmethod
    def with_timeout(cls, timeout_seconds: float) -> TaskBudget:
        """基于给定的相对超时秒数构造时间预算守卫实例。"""
        return cls(deadline=time.monotonic() + max(0.1, timeout_seconds))

    @property
    def remaining(self) -> float:
        """获取当前距任务截止时间所剩的有效秒数。"""
        return max(0.0, self.deadline - time.monotonic())

    @property
    def is_expired(self) -> bool:
        """判断当前任务时间预算是否已经耗尽。"""
        return time.monotonic() >= self.deadline


def execute_with_resilience(
    func: Callable[[], T],
    circuit_breaker: CircuitBreaker | None = None,
    budget: TaskBudget | None = None,
    max_retries: int = 2,
    base_delay: float = 1.0,
    max_delay: float = 6.0,
    sleep_func: Callable[[float], None] = time.sleep,
    retry_allowed: Callable[[], bool] | None = None,
) -> T:
    """受控弹性重试与熔断包装调度执行器。

    执行策略：
    1. 前置熔断拦截：若断路器处于 OPEN 状态，立即抛出 CircuitBreakerOpenException；
    2. 执行目标函数并捕获异常：
       - 若为确定性错误 (DETERMINISTIC_FATAL)：严禁重试，不累加断路器失败计数，立即快速失败抛出；
       - 若为瞬时故障 (TRANSIENT_RETRYABLE / UPSTREAM_OVERLOAD)：
         - 若预算耗尽或达到 max_retries 上限：累加断路器失败并抛出异常；
         - 若仍在预算内：优先提取 Retry-After，否则按 Full Jitter 计算退避延迟并安全休眠后重试。

    Args:
        func: 待执行的无参包装函数。
        circuit_breaker: 可选的三态断路器实例。
        budget: 可选的任务总时间预算守卫。
        max_retries: 瞬时故障最大重试次数（默认为 2）。
        base_delay: 指数退避基准等待时间（秒）。
        max_delay: 指数退避等待时间上限（秒）。
        sleep_func: 休眠函数（便于单元测试 Mock 时间流逝）。
        retry_allowed: 可选的重试边界检查，例如已输出文本或已取消时禁止重试。

    Returns:
        T: 目标函数执行成功的返回值。

    Raises:
        CircuitBreakerOpenException: 当断路器处于熔断开启状态时。
        Exception: 当函数抛出确定性异常或耗尽重试次数时。
    """
    if circuit_breaker is not None and not circuit_breaker.can_execute():
        raise CircuitBreakerOpenException("Circuit breaker is OPEN; service temporarily disabled.")

    attempt = 0
    while True:
        try:
            result = func()
            if circuit_breaker is not None:
                circuit_breaker.record_success()
            return result
        except Exception as exc:
            category, reason, retry_after = FaultClassifier.classify(exc)

            # 1. 确定性错误（参数错误/Schema不符/权限不足）：严禁重试！直接快速失败
            if category == FaultCategory.DETERMINISTIC_FATAL:
                raise exc

            # 2. 检查是否耗尽重试次数或时间预算超期
            attempt += 1
            if (
                attempt > max_retries
                or (budget is not None and budget.is_expired)
                or (retry_allowed is not None and not retry_allowed())
            ):
                if circuit_breaker is not None:
                    circuit_breaker.record_failure()
                raise exc

            # 3. 计算退避时间：遵循服务端的 Retry-After，否则使用 Full Jitter 随机抖动
            if retry_after is not None and retry_after > 0:
                sleep_time = min(retry_after, max_delay)
            else:
                # Full Jitter: random.uniform(0, min(max_delay, base_delay * 2^(attempt-1)))
                backoff_cap = min(max_delay, base_delay * (2 ** (attempt - 1)))
                sleep_time = random.uniform(0, backoff_cap)

            # 4. 若休眠时间超过剩余任务预算，提前终止重试，防止无谓等待
            if budget is not None and sleep_time > budget.remaining:
                if circuit_breaker is not None:
                    circuit_breaker.record_failure()
                raise TimeoutError("Task budget exhausted during retry backoff") from exc

            logger.info(
                "Transient fault detected (%s). Retrying %d/%d after %.2fs...",
                reason,
                attempt,
                max_retries,
                sleep_time,
            )
            sleep_func(sleep_time)


class ModelEffectivenessMonitor:
    """在线模型业务有效性与质量退化监测器 (Model Business Effectiveness Monitor)

    不仅关注技术连通性（200 / 延迟），还持续监测协议完整性、参数合规率、工具越权幻觉与安全审查违规率。
    当连续业务有效性故障达到阈值时，标记模型能力退化 (is_degraded)，驱动系统执行非对称能力降权或切流。
    """

    def __init__(self, failure_threshold: int = 2):
        self.failure_threshold = max(1, failure_threshold)
        self.consecutive_business_failures = 0
        self.total_evaluations = 0
        self.protocol_errors = 0
        self.unauthorized_tool_attempts = 0
        self.safety_rejections = 0
        self.last_failure_reason: str = ""

    def record_success(self) -> None:
        """记录一次完全合规的业务交互，重置连续故障计数。"""
        self.consecutive_business_failures = 0
        self.total_evaluations += 1

    def record_business_failure(self, failure_type: str, detail: str = "") -> None:
        """记录一次业务有效性异常（如信封损坏、越权幻觉工具、高危被拒等）。"""
        self.consecutive_business_failures += 1
        self.total_evaluations += 1
        self.last_failure_reason = detail or failure_type
        if failure_type == "protocol":
            self.protocol_errors += 1
        elif failure_type == "unauthorized_tool":
            self.unauthorized_tool_attempts += 1
        elif failure_type == "safety":
            self.safety_rejections += 1
        logger.warning(
            "Model business effectiveness failure recorded (%s: %s). Consecutive failures: %d/%d",
            failure_type,
            detail,
            self.consecutive_business_failures,
            self.failure_threshold,
        )

    @property
    def is_degraded(self) -> bool:
        """当连续业务失败达到阈值时判定模型能力退化。"""
        return self.consecutive_business_failures >= self.failure_threshold

    def reset(self) -> None:
        """重置监测器状态。"""
        self.consecutive_business_failures = 0
        self.last_failure_reason = ""

    def summary(self) -> dict:
        """生成业务质量可观测指标摘要。"""
        return {
            "totalEvaluations": self.total_evaluations,
            "consecutiveFailures": self.consecutive_business_failures,
            "protocolErrors": self.protocol_errors,
            "unauthorizedToolAttempts": self.unauthorized_tool_attempts,
            "safetyRejections": self.safety_rejections,
            "isDegraded": self.is_degraded,
            "lastFailureReason": self.last_failure_reason,
        }

