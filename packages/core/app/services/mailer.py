"""SolaceAI 邮件发送与危机告警投递基础设施服务 (Mail & SMTP Delivery Service)

本模块为心理咨询系统提供关键的邮件通信支撑：
1. 双重投递模式支持：
   - 生产 `smtp` 模式：通过标准加密通道 (SSL/TLS) 连接公网或校园 SMTP 网关；
   - 本地调试 `log` 模式：仅在日志中留痕，不产生物理网络发送，防止测试阶段骚扰真实邮箱；
2. 严格的凭据与隐私保护：
   - 发送异常捕获时不外泄明文密码、授权码或原始 SMTP 错误响应文本，避免在日志与 API 返回中造成信息泄露；
3. 校园高频邮件服务（如 QQ 企业邮箱/网易 163）安全协议特征校验与端口合规诊断；
4. 证书信任链增强：借助 `certifi` CA 证书包增强本地 Python 环境对公共 SSL 根证书的信任校验。
"""
from __future__ import annotations

import re
import smtplib
import ssl
from typing import Any

import certifi
from email.message import EmailMessage

from app.core.config import Settings


class MailService:
    """SMTP 邮件协议通信与安全投递服务 (Mail Delivery Service)

    Attributes:
        settings (Settings): 系统运行时全局配置实例。
    """

    def __init__(self, settings: Settings):
        """初始化邮件投递服务。

        Args:
            settings (Settings): 系统配置实例。
        """
        self.settings = settings

    def recipients(self) -> list[str]:
        """解析并清洗服务端配置的预警收件人邮箱列表 (Parse Alert Recipients)

        支持逗号与分号混合分隔，自动过滤空白字符。

        Returns:
            list[str]: 格式化清洗后的收件人邮箱地址列表。
        """
        return [value.strip() for value in self.settings.alert_email_to.replace(';', ',').split(',') if value.strip()]

    def sender(self) -> str:
        """获取当前系统发件人规范邮箱地址 (Get Sender Email Address)

        优先读取 `alert_email_from`，为空时回退至 `smtp_username`。

        Returns:
            str: 格式化发件人地址字符串。
        """
        return self.settings.alert_email_from.strip() or self.settings.smtp_username.strip()

    @staticmethod
    def valid_address(value: str) -> bool:
        """校验给定字符串是否为合规的单个电子邮箱地址 (Validate Email Format)

        基于通用规范正则检查，严格限制总长度不超过 254 字符并排除空白与控制字符。

        Args:
            value (str): 待检查的邮箱文本。

        Returns:
            bool: 格式合规返回 True，否则返回 False。
        """
        return bool(re.fullmatch(r'[^\s@<>;,\r\n]+@[^\s@<>;,\r\n]+\.[^\s@<>;,\r\n]+', value)) and len(value) <= 254

    def status(self) -> dict[str, Any]:
        """诊断当前系统的 SMTP 配置完整度与合规安全性 (Diagnose Mail Settings Status)

        全面核对必要字段是否缺失（Host、用户名、密码、发件人）、投递模式（smtp/log）、
        加密协议单选互斥（SSL 与 STARTTLS 不可同时开启）、QQ 邮箱 465 专有端口匹配等规则。

        Returns:
            dict[str, Any]: 包含 ready, alertReady, missing, issues, alertIssues 等详细诊断指标的字典。
        """
        settings = self.settings
        mode = settings.alert_email_delivery_mode.strip().lower()
        missing = []
        for name, value in [('SMTP_HOST', settings.smtp_host.strip()), ('SMTP_USERNAME', settings.smtp_username.strip()),
                            ('SMTP_PASSWORD', settings.smtp_password), ('ALERT_EMAIL_FROM', self.sender())]:
            if not value:
                missing.append(name)
        issues = []
        if mode not in {'smtp', 'log'}:
            issues.append('投递模式必须为 smtp 或 log')
        if not (settings.smtp_use_ssl or settings.smtp_use_tls):
            issues.append('SMTP 必须启用 SSL 或 STARTTLS')
        if settings.smtp_use_ssl and settings.smtp_use_tls:
            issues.append('SSL 和 STARTTLS 只能启用一种')
        if settings.smtp_host.lower().endswith('qq.com') and not (settings.smtp_use_ssl and settings.smtp_port == 465):
            issues.append('QQ 邮箱请使用 SSL 465 端口')
        if self.sender() and not self.valid_address(self.sender()):
            issues.append('发件邮箱格式不正确')
        alert_issues = []
        if not self.recipients():
            alert_issues.append('请配置 ALERT_EMAIL_TO')
        elif any(not self.valid_address(value) for value in self.recipients()):
            alert_issues.append('预警收件邮箱格式不正确')
        ready = mode == 'smtp' and not missing and not issues
        return {
            'mode': mode,
            'ready': ready,
            'alertReady': ready and not alert_issues,
            'alertIssues': alert_issues,
            'missing': missing,
            'issues': issues,
            'sender': self.sender(),
            'alertRecipients': self.recipients(),
            'host': settings.smtp_host,
            'port': settings.smtp_port,
            'ssl': settings.smtp_use_ssl,
            'tls': settings.smtp_use_tls,
            'credentialsConfigured': bool(settings.smtp_username and settings.smtp_password),
            'message': '仅记录，未发送邮件' if mode == 'log' else 'SMTP 接收成功仅代表服务器接受，不保证进入收件箱',
        }

    def send_code(self, email: str, code: str, purpose: str) -> dict[str, str]:
        """向指定邮箱发送身份验证码短信格式邮件 (Send Authentication Code Email)

        Args:
            email (str): 目标学生邮箱地址。
            code (str): 6位纯数字验证码。
            purpose (str): 验证用途（'login' 登录 或 'register' 注册）。

        Returns:
            dict[str, str]: 包含 status ('SMTP_ACCEPTED' 等) 与用户友好提示的字典。
        """
        label = {'register': '注册', 'bind': '绑定邮箱'}.get(purpose, '登录')
        return self.send(
            [email],
            f'[SolaceAI] {label}验证码',
            f'您的{label}验证码为：{code}\n验证码有效期为 5 分钟。请勿向他人透露验证码。如非本人操作，请忽略。',
        )

    def send(self, recipients: list[str], subject: str, body: str) -> dict[str, str]:
        """执行底层邮件封装与 SMTP 协议投递 (Send Email via Configured Delivery Mode)

        执行逻辑：
        1. 若当前处于 log 仅记录模式，直接返回 LOG_ONLY；
        2. 校验配置完备度与收件人格式；
        3. 组装标准 MIME `EmailMessage` 并调用 `deliver` 执行物理投递；
        4. 统一捕获异常并隐藏敏感错误响应，防止在前端反射泄露内部网络与鉴权细节。

        Args:
            recipients (list[str]): 收件人邮箱列表。
            subject (str): 邮件标题主题。
            body (str): 邮件纯文本正文内容。

        Returns:
            dict[str, str]: 包含 status 与提示信息的投递回执。
        """
        if self.settings.alert_email_delivery_mode.strip().lower() == 'log':
            return {'status': 'LOG_ONLY', 'message': '仅记录，未发送邮件'}
        if not self.status()['ready']:
            return {'status': 'FAILED', 'message': '邮件配置不完整或不安全，请联系管理员'}
        if not recipients or any(not self.valid_address(value) for value in recipients):
            return {'status': 'FAILED', 'message': '收件邮箱格式不正确'}
        try:
            message = EmailMessage()
            message['Subject'] = subject
            message['From'] = self.sender()
            message['To'] = ', '.join(recipients)
            message.set_content(body)
            self.deliver(message)
        except Exception:
            # 捕获异常，坚决不向外反射 SMTP 原始应答（可能包含账号授权码或内网拓扑）
            return {'status': 'FAILED', 'message': 'SMTP 投递失败，请检查授权码、配置和网络后重试'}
        return {'status': 'SMTP_ACCEPTED', 'message': 'SMTP 服务器已接受邮件，请检查收件箱及垃圾邮件'}

    def deliver(self, message: EmailMessage) -> None:
        """建立安全加密通道并投递邮件实体 (Establish Encrypted Transport & Deliver)

        根据配置自动协商纯 SSL 隐式加密 (通常为 465 端口) 或显式 STARTTLS (通常为 587/25 端口)，
        通过 certifi 注入受信根证书包校验服务端身份。

        Args:
            message (EmailMessage): 已封装就绪的标准邮件消息对象。
        """
        context = ssl.create_default_context()
        # 兼容部分系统精简版 Python 无系统 CA 根证书包的问题，加载 certifi 证书链
        context.load_verify_locations(cafile=certifi.where())
        settings = self.settings
        if settings.smtp_use_ssl:
            with smtplib.SMTP_SSL(
                settings.smtp_host,
                settings.smtp_port,
                timeout=settings.smtp_timeout_seconds,
                context=context,
            ) as server:
                self._send_message(server, message)
        else:
            with smtplib.SMTP(
                settings.smtp_host,
                settings.smtp_port,
                timeout=settings.smtp_timeout_seconds,
            ) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                self._send_message(server, message)

    def _send_message(self, server: smtplib.SMTP, message: EmailMessage) -> None:
        """登录 SMTP 服务器并发出消息体，检测收件人拒收异常 (Authenticate & Dispatch Message)

        Args:
            server (smtplib.SMTP): 活跃的 SMTP 客户端连接。
            message (EmailMessage): 待发出的邮件实体。

        Raises:
            smtplib.SMTPRecipientsRefused: 当全部或部分收件人被服务器拒收时抛出。
        """
        server.login(self.settings.smtp_username, self.settings.smtp_password)
        if server.send_message(message):
            raise smtplib.SMTPRecipientsRefused({})
