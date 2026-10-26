"""SolaceAI 路径解析与根目录探测基础设施 (Project Path Detection & Resource Roots)

本模块提供确定性的项目根目录解析能力：
1. 优先支持容器化或生产部署中显式指定的 `SOLACEAI_ROOT` 环境变量；
2. 针对源码检出环境（Source Checkout）向上遍历父级目录，自动识别
   包含 `packages/core` 与 `apps` 的 Monorepo 顶层基准目录；
3. 降级使用当前进程的工作目录（`Path.cwd()`），确保开发、测试及打包分发时路径引用一致。
"""
import os
from pathlib import Path


def project_root() -> Path:
    """探测并计算 SolaceAI 工程项目的根目录绝对路径 (Detect Project Root Directory)

    探测顺序：
    1. 检查操作系统环境变量 `SOLACEAI_ROOT`，若配置则直接解析为绝对路径返回；
    2. 从当前文件所在物理位置逐级向上回溯其父目录，寻找同时存在 `packages/core`
       与 `apps` 子目录的顶层根目录；
    3. 若未命中特征目录，则安全回退至当前进程启动时的工作目录 (`Path.cwd()`)。

    Returns:
        Path: 解析出的工程根目录绝对路径对象。
    """
    configured = os.environ.get('SOLACEAI_ROOT')
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / 'packages/core').is_dir() and (parent / 'apps').is_dir():
            return parent
    return Path.cwd()


# 系统全局项目根目录标准绝对路径常量
PROJECT_ROOT: Path = project_root()
