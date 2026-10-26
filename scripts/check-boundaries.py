"""SolaceAI 架构分层与包依赖边界静态检查脚本 (Architecture Dependency Boundary Checker)

本脚本利用 Python AST 抽象语法树对工程源码执行确定性的静态依赖关系扫描：
1. 核心包解耦防腐 (`packages/core/app`)：
   - 严禁核心业务代码反向依赖任何具体 Profile (`apps`, `app.online`, `app.offline`) 或评测套件；
2. 应用模式隔离 (`apps/online` vs `apps/offline`)：
   - 在线模式严禁直接依赖离线模式组件或评测代码；
   - 离线模式严禁直接依赖在线模式组件或评测代码；
3. CI/CD 门禁保障：一旦检测到越界非法引用立即抛出 SystemExit 阻断构建，确保架构高内聚低耦合。
"""
import ast
from pathlib import Path

# 解析工程根目录基准路径
root = Path(__file__).resolve().parents[1]
violations = []

# 定义各目录必须遵守的非法依赖前缀黑名单
BOUNDARY_RULES = [
    ('packages/core/app', ('apps', 'app.online', 'app.offline', 'app.evaluation', 'app.harness', 'evaluation', 'qwen3_bonus')),
    ('apps/online', ('apps.offline', 'app.offline', 'evaluation')),
    ('apps/offline', ('apps.online', 'app.online', 'evaluation')),
]

for directory, forbidden in BOUNDARY_RULES:
    for path in (root / directory).rglob('*.py'):
        # 忽略构建产物目录
        if 'build' in path.parts:
            continue
        # 解析文件 AST 语法树并收集 import 与 import from 语句
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = (
                [node.module or ''] if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names] if isinstance(node, ast.Import)
                else []
            )
            for name in names:
                if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden):
                    violations.append(f'{path.relative_to(root)}:{node.lineno}: {name}')

if violations:
    raise SystemExit('\n'.join(violations))

print('Dependency boundaries passed')
