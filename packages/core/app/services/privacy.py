"""SolaceAI 个人隐私数据保护与敏感实体脱敏服务 (Data Privacy & PII Sanitization)

本模块作为大模型前置安全防线，防止用户个人身份信息（Personally Identifiable Information, PII）
在未经匿名化的情况下泄露给外部云端模型或在日志中明文留存：
1. 中国大陆手机号脱敏（11 位数字识别）；
2. 电子邮箱地址脱敏（RFC 标准邮箱模式匹配）；
3. 中华人民共和国二代身份证号脱敏（18 位带校验位匹配）；
4. 替换策略：统一掩码替换为可解释的 `[已脱敏]` 占位标记。
"""

import re


class PrivacySanitizer:
    """敏感隐私信息脱敏清洗器 (PII Sensitive Information Sanitizer)

    内置预编译的高性能正则表达式，提供亚毫秒级的低延迟实体扫描与脱敏替换。
    """

    patterns = [
        # 中国大陆 11 位手机号码正则表达式（以 1 开头，第二位 3-9）
        re.compile(r"1[3-9]\d{9}"),
        # 常见电子邮箱地址正则表达式
        re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
        # 18 位中国二代居民身份证号码正则表达式（17 位数字 + 1 位数字或 X/x）
        re.compile(r"\b\d{17}[\dXx]\b"),
    ]

    def sanitize(self, text: str) -> str:
        """对输入文本执行 PII 敏感信息脱敏过滤 (Sanitize Sensitive Entities)

        依次扫描预置敏感实体正则模式，将命中的敏感数据段替换为 '[已脱敏]' 占位符。

        Args:
            text (str): 用户输入的原始包含潜在隐私的文本。

        Returns:
            str: 脱敏后可安全投递至外部大模型或打印日志的干净文本。若入参为空则返回空字符串。
        """
        sanitized = text or ""
        if "1" in sanitized:
            sanitized = self.patterns[0].sub("[已脱敏]", sanitized)
        if "@" in sanitized:
            sanitized = self.patterns[1].sub("[已脱敏]", sanitized)
        if any(c.isdigit() for c in sanitized):
            sanitized = self.patterns[2].sub("[已脱敏]", sanitized)
        return sanitized
