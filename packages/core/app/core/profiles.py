"""SolaceAI 应用运行模式配置隔离加载器 (Isolated Application Profile Settings Loader)

本模块实现了多运行模式（Profile，如 Online、Offline）环境配置的隔离解析机制：
1. 配置源物理分层：既能继承顶层项目级 `.env` 文件中的通用连接配置（如大模型 API Key），
   又能防止不同 Profile 之间专有的特化字段相互污染；
2. 动态字段屏蔽：通过探测 `settings_type` 类型注解中的专有字段，在解析全局 `.env`
   时动态剔除这些字段，保证 Profile 专属的子配置具有绝对的覆盖控制权；
3. 类型稳定性保证：构造返回目标 `settings_type` 的原生稳定类型实例，
   兼容 Pydantic V2 的动态校验与单例比较。
"""
from pathlib import Path
from typing import Any, Type, TypeVar
from pydantic_settings import BaseSettings, DotEnvSettingsSource

T = TypeVar("T", bound=BaseSettings)


def load_profile_settings(settings_type: Type[T], project_root: Path, **overrides: Any) -> T:
    """加载并组装指定 Profile 的系统运行配置 (Load Isolated Profile Settings)

    执行逻辑：
    1. 收集目标 Profile 配置类中显式声明的所有属性字段名称；
    2. 动态定制 Pydantic Settings 数据源优先级：针对顶层全局 `.env`，过滤并剔除
       所有已在当前 Profile 中定义的特有字段，防止全局默认值意外覆盖子 Profile 配置；
    3. 合并运行时传入的显式覆盖参数（如自定义 `_env_file`）；
    4. 实例化后通过 `model_construct` 重建稳定类型的配置对象返回。

    Args:
        settings_type (Type[T]): 继承自 `Settings` 的特化 Profile 配置类型（如 `ProfileSettings`）。
        project_root (Path): 工程根目录路径，用于定位项目级全局 `.env` 文件。
        **overrides (Any): 运行时显式传入的覆盖参数（例如环境变量文件路径、键值对）。

    Returns:
        T: 经过安全隔离与校验的 Profile 配置实例。
    """
    # 提取 Profile 特有字段名称并转为小写，用于在加载全局 .env 时精准过滤
    profile_fields = {name.lower() for name in settings_type.__annotations__}

    class ConfiguredSettings(settings_type):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: Any,
            env_settings: Any,
            dotenv_settings: Any,
            file_secret_settings: Any,
        ) -> tuple[Any, ...]:
            # 加载全局 .env 但屏蔽所有被当前 Profile 专属占用的字段
            shared = DotEnvSettingsSource(settings_cls, env_file=project_root / '.env')
            shared.env_vars = {
                key: value for key, value in shared.env_vars.items()
                if key.lower() not in profile_fields
            }
            return init_settings, env_settings, dotenv_settings, shared, file_secret_settings

    settings = ConfiguredSettings(**overrides)
    # 重构为原始公共类型，保证在多次加载与 configure_settings 相等性检查时类型一致
    return settings_type.model_construct(**settings.model_dump())
