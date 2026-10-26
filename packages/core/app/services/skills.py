"""业务Skill注册、校验、交接摘要与渐进式按需加载。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.enums import RiskLevel
from app.models.entities import PsychologicalReport, UserAccount


class SkillLoadError(RuntimeError):
    """心理技能文件加载与解析异常 (Skill Parsing & Validation Error)"""

    pass


@dataclass(frozen=True)
class SkillValidationIssue:
    """技能静态校验问题项"""

    level: str
    message: str


@dataclass(frozen=True)
class SolaceAISkill:
    """心理技能领域模型"""

    name: str
    description: str
    body: str
    path: Path
    metadata: dict[str, str] = field(default_factory=dict)

    def validation_issues(self) -> list[SkillValidationIssue]:
        """对技能定义执行合规静态检查"""
        issues: list[SkillValidationIssue] = []
        if self.path.parent.name != self.name:
            issues.append(
                SkillValidationIssue(
                    "WARN",
                    f"目录名 {self.path.parent.name} 与 skill name {self.name} 不一致",
                )
            )
        if not any(heading in self.body for heading in ("## Workflow", "## 执行规则")):
            issues.append(
                SkillValidationIssue(
                    "WARN", "建议包含 ## 执行规则 或 ## Workflow 小节，便于人工审阅和模型稳定加载"
                )
            )
        if len(self.description) < 20:
            issues.append(
                SkillValidationIssue(
                    "WARN", "description 太短，可能无法准确表达触发场景"
                )
            )
        if self.name == "counselor_handoff_summary" and "```text" not in self.body:
            issues.append(
                SkillValidationIssue(
                    "ERROR", "counselor_handoff_summary 必须包含 text 模板"
                )
            )
        return issues


class SolaceAISkillRegistry:
    """领域专业心理技能注册中心 (Domain Skill Registry)"""

    def __init__(self, root: Path | None = None):
        """初始化技能注册中心"""
        if root is not None:
            self.root = root
        else:
            app_skills = Path(__file__).resolve().parents[1] / "skills"
            self.root = app_skills if app_skills.exists() else __import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT / "skills"

    def list_skills(self) -> list[SolaceAISkill]:
        """扫描并加载根目录下所有已注册的技能"""
        if not self.root.exists():
            return []
        skills = []
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            skills.append(self._load_skill_file(skill_file))
        return skills

    def status_items(self) -> list[dict]:
        """获取所有技能的运行与合规状态摘要（供管理端 API 暴露）"""
        if not self.root.exists():
            return []
        items = []
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            try:
                skill = self._load_skill_file(skill_file)
                issues = skill.validation_issues()
            except SkillLoadError as exc:
                items.append(
                    {
                        "name": skill_file.parent.name,
                        "status": "FAILED",
                        "description": str(exc),
                        "path": str(skill_file.relative_to(self.root.parent)),
                        "issues": [{"level": "ERROR", "message": str(exc)}],
                    }
                )
                continue
            has_error = any(issue.level == "ERROR" for issue in issues)
            items.append(
                {
                    "name": skill.name,
                    "status": "FAILED"
                    if has_error
                    else "READY"
                    if not issues
                    else "WARN",
                    "description": skill.description,
                    "path": str(skill.path.relative_to(self.root.parent)),
                    "issues": [
                        {"level": issue.level, "message": issue.message}
                        for issue in issues
                    ],
                    "metadata": skill.metadata,
                }
            )
        return items

    def get_required(self, name: str) -> SolaceAISkill:
        """获取指定名称的必选技能"""
        for skill in self.list_skills():
            if skill.name == name:
                return skill
        raise SkillLoadError(f"required standard skill not found: {name}")

    def template_for(self, name: str) -> str:
        """提取指定技能中的 ```text 文本模板内容"""
        skill = self.get_required(name)
        match = re.search(r"```text\s*\n(?P<template>.*?)\n```", skill.body, re.DOTALL)
        if match is None:
            raise SkillLoadError(
                f"standard skill {name} does not define a text template"
            )
        return match.group("template").strip()

    def _load_skill_file(self, path: Path) -> SolaceAISkill:
        """物理读取并深度解析指定路径下的 SKILL.md 技能定义文件"""
        text = path.read_text(encoding="utf-8")
        metadata, body = _split_frontmatter(text, path)
        name = metadata.get("name") or path.parent.name
        description = metadata.get("description", "")
        if not name.strip():
            raise SkillLoadError(f"{path} is missing frontmatter name")
        if not description.strip():
            raise SkillLoadError(f"{path} is missing frontmatter description")
        if not body.strip():
            raise SkillLoadError(f"{path} is missing skill body")
        return SolaceAISkill(
            name=name.strip(),
            description=description.strip(),
            body=body.strip(),
            path=path,
            metadata=metadata,
        )


class SolaceAISkillLibrary:
    """技能业务应用门面库 (Skill Library Facade)"""

    @staticmethod
    def registry() -> SolaceAISkillRegistry:
        """获取或构造底层全局技能注册表管理实例"""
        return SolaceAISkillRegistry()

    @staticmethod
    def list_skills() -> list[SolaceAISkill]:
        """获取当前系统中扫描并加载的所有合法心理技能列表"""
        return SolaceAISkillLibrary.registry().list_skills()

    @staticmethod
    def status_items() -> list[dict]:
        """获取全量技能的运行状态、合规校验级别及告警清单，供管理端看板审计"""
        return SolaceAISkillLibrary.registry().status_items()

    @staticmethod
    def counselor_handoff_summary(
        report: PsychologicalReport, user: UserAccount | None
    ) -> str:
        """渲染辅导员危机个案交接摘要"""
        template = SolaceAISkillLibrary.registry().template_for(
            "counselor_handoff_summary"
        )
        student = _student_label(user, report.user_id)
        urgency = (
            "立即跟进" if report.risk_level == RiskLevel.HIGH.value else "尽快跟进"
        )
        next_steps = [
            f"{urgency}，确认学生当前位置、身边是否有人陪伴，以及当前是否安全。",
            "联系学生本人或其可用的现实支持人，并记录已采取的联系方式。",
            "必要时联系校园保卫、心理中心值班老师或当地紧急救助。",
            "将后续安排、接手人和下一次复访时间写入个案备注。",
        ]
        return _render_template(
            template,
            {
                "report_id": str(report.id),
                "student": student,
                "risk_level": report.risk_level,
                "emotion": report.emotion,
                "summary": report.summary,
                "next_steps": "\n".join(f"- {step}" for step in next_steps),
                "content_excerpt": _truncate(report.content, 700),
            },
        )


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str]:
    """解析 Markdown 文件开头的 YAML Frontmatter 区域与正文"""
    if not text.startswith("---\n"):
        raise SkillLoadError(f"{path} is missing YAML frontmatter")
    end = text.find("\n---", 4)
    if end == -1:
        raise SkillLoadError(f"{path} has unterminated YAML frontmatter")
    metadata = {}
    for line in text[4:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SkillLoadError(f"{path} has invalid frontmatter line: {line}")
        key, value = stripped.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text[end + len("\n---") :].strip()


def _render_template(template: str, values: dict[str, str]) -> str:
    """基于双大括号插值（如 `{{student}}`）渲染文本模板"""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def _student_label(user: UserAccount | None, user_id: int) -> str:
    """生成合规脱敏且信息明确的学生身份摘要标签"""
    if user is None:
        return f"userId={user_id}"
    if user.display_name:
        return f"{user.display_name} ({user.username})"
    return user.username


def _truncate(text: str, limit: int) -> str:
    """将文本空白多余字符归一化并安全截断至指定长度，末尾附加省略号"""
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


class ProgressiveSkills:
    """渐进式专业技能加载器。"""

    def __init__(self, registry=None):
        """初始化渐进式技能加载管理器。"""
        self.registry = registry or SolaceAISkillRegistry()
        self.loaded = {}
        self.optional = set()

    def catalog(self) -> list[dict]:
        """获取当前可用技能的精简元数据目录列表。"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "when_to_use": s.metadata.get("when_to_use", s.description),
                "version": __import__("hashlib")
                .sha256(s.body.encode())
                .hexdigest()[:12],
            }
            for s in self.registry.list_skills()
        ]

    def load(self, name: str, mandatory: bool = False) -> str:
        """按技能名称加载技能正文文本。"""
        if name in self.loaded:
            return self.loaded[name]
        if not mandatory and len(self.optional) >= 3:
            raise ValueError("Optional skill budget exhausted")
        skill = self.registry.get_required(name)
        self.loaded[name] = skill.body[:12000]
        if not mandatory:
            self.optional.add(name)
        return self.loaded[name]
