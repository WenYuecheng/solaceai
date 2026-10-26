"""SolaceAI 分发轮子包独立环境解耦冒烟验证工具 (Distribution Wheel Smoke Test)

本脚本面向发布与持续集成阶段，在完全脱离源码目录的临时隔离环境中验证构建生成的 Wheel 包：
1. 真实解压模拟安装：将 `solaceai_core` 与对应的 `solaceai_{online/offline}` 解压至干净临时目录；
2. 互斥性隔离断言：断言在 online 环境下绝对探测不到 offline 代码，反之亦然；
3. 核心接口与资源冒烟：启动最小化应用实例并测试健康检查 (`/actuator/health`)、静态单页路由与鉴权拦截；
4. 严格模块污染校验：断言未违规载入对端 Profile 模块或重型计算依赖（如未在 online 下载入 torch）。
"""
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import zipfile

# 提取存放构建好的 Wheel 文件的目录路径
wheel_dir = Path(sys.argv[1])
deps = sysconfig.get_paths()['purelib']

# 分别对 online 和 offline 两种发行包执行独立隔离冒烟测试
for profile in ('online', 'offline'):
    with tempfile.TemporaryDirectory(prefix=f'solaceai-{profile}-') as temp:
        root = Path(temp)
        # 提取并解包核心基础包与当前特化 Profile 包
        for name in ('core', profile):
            wheel = next(wheel_dir.glob(f'solaceai_{name}-*.whl'))
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(root / 'installed')

        # 构造纯净子进程执行代码，使用 -I 和 -S 选项屏蔽宿主系统 site-packages 路径干扰
        code = f'''
import sys
sys.path[:0] = [{str(root / 'installed')!r}, {deps!r}]
import importlib.util

# 断言互斥包未被污染载入
assert importlib.util.find_spec('apps.{"offline" if profile == "online" else "online"}') is None
from apps.{profile}.settings import load_settings
from apps.{profile}.main import create_app
from fastapi.testclient import TestClient

# 组装最小离线测试配置
s = load_settings(
    _env_file=None,
    ai_provider='mock',
    knowledge_vector_enabled=False,
    knowledge_rewrite_enabled=False,
    knowledge_contextual_enabled=False,
    reranker_enabled=False,
    local_rag_autostart=False,
    database_url='sqlite:///{root}/test.sqlite',
)
app = create_app(s)

# 测试健康检查、静态资源托管与安全认证门禁
with TestClient(app) as client:
    assert client.get('/actuator/health').json() == {{'status': 'UP'}}
    assert client.get('/').status_code == 200
    assert client.get('/api/profile').status_code in (401, 403)

# 确保无违规导入
assert not any(n.startswith(('evaluation', 'torch', 'apps.{"offline" if profile == "online" else "online"}')) for n in sys.modules)
print('{profile}: packaged startup, resources, auth and health passed')
'''
        subprocess.run([sys.executable, '-I', '-S', '-c', code], cwd=root, check=True)
