"""配置：合并全局 ~/.coda/settings.json 与项目 .coda/settings.json。

合并规则：标量和字典按键覆盖（项目优先）；permissions 下的规则列表拼接。
API Key 只从环境变量或 ~/.coda/.env 读取，不读工作区里的 .env——那是被开发项目自己的密钥。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Provider = Literal["deepseek", "openai"]
Mode = Literal["default", "accept-edits", "plan", "yolo"]
MODES_HELP = "权限模式：default（编辑和命令要确认）、accept-edits（编辑自动）、plan（只读）、yolo（全部自动）"


def coda_home() -> Path:
    """全局目录，可用 CODA_HOME 覆盖（测试时指向临时目录）。"""
    return Path(os.environ.get("CODA_HOME", Path.home() / ".coda"))


class Price(BaseModel):
    """每百万 token 的价格（美元）。"""

    input: float = 0.0
    output: float = 0.0
    cache_hit: float = 0.0


class ModelProfile(BaseModel):
    provider: Provider = "openai"
    base_url: str
    api_key_env: str
    model: str
    thinking: bool = False  # 仅 deepseek 生效
    reasoning_effort: Literal["low", "high", "max"] | None = None
    context_budget: int = 128_000
    max_tokens: int | None = None
    price_per_m: Price | None = None


class ContextConfig(BaseModel):
    """上下文压缩。比例相对于模型档案的 context_budget。"""

    offload_chars: int = 8000  # 工具结果超过这个字符数就落盘，模型只看到头尾
    micro_ratio: float = 0.6  # 达到预算 60%：旧工具结果换成占位符
    summary_ratio: float = 0.85  # 达到预算 85%：较早的历史换成摘要
    keep_tool_results: int = 3  # 微压缩保留最近几次工具结果
    offload: bool = True
    micro: bool = True


class McpServerConfig(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    enabled: bool = True


class Permissions(BaseModel):
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


DEFAULT_MODELS: dict[str, dict[str, Any]] = {
    "deepseek-flash": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-flash",
        "thinking": True,
        "reasoning_effort": "high",
        "context_budget": 128_000,
        "price_per_m": {"input": 0.3, "output": 1.2, "cache_hit": 0.006},
    },
    "qwen3-coder": {
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "context_budget": 128_000,
    },
    "glm-4.5-air": {
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "model": "zai-org/GLM-4.5-Air",
        "context_budget": 96_000,
    },
}


class Settings(BaseModel):
    model: str = "deepseek-flash"
    models: dict[str, ModelProfile] = Field(default_factory=dict)
    mode: Mode = "default"
    max_steps: int = 60
    permissions: Permissions = Field(default_factory=Permissions)
    context: ContextConfig = Field(default_factory=ContextConfig)
    mcpServers: dict[str, McpServerConfig] = Field(
        default_factory=dict
    )  # 与 Claude Code 等工具相同的键名
    show_thinking: bool = True
    tools: list[str] | None = None  # 启用的工具名（如 ["bash"]）；None 表示全部

    def profile(self, name: str | None = None) -> ModelProfile:
        key = name or self.model
        if key not in self.models:
            known = "、".join(self.models) or "（无）"
            raise KeyError(f"未知的模型档案 {key!r}，可用：{known}")
        return self.models[key]


_LIST_SECTIONS = {"permissions"}


def _merge(base: dict[str, Any], override: dict[str, Any], *, concat_lists: bool) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        old = out.get(key)
        if isinstance(old, dict) and isinstance(value, dict):
            out[key] = _merge(old, value, concat_lists=concat_lists or key in _LIST_SECTIONS)
        elif concat_lists and isinstance(old, list) and isinstance(value, list):
            out[key] = old + [v for v in value if v not in old]
        else:
            out[key] = value
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} 不是合法的 JSON：{e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是对象")
    return data


def project_settings_path(workdir: Path) -> Path:
    return workdir / ".coda" / "settings.json"


def load_settings(workdir: Path) -> Settings:
    merged: dict[str, Any] = {"models": DEFAULT_MODELS}
    merged = _merge(merged, _read_json(coda_home() / "settings.json"), concat_lists=False)
    merged = _merge(merged, _read_json(project_settings_path(workdir)), concat_lists=False)
    return Settings.model_validate(merged)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            values[key.strip()] = value
    return values


def get_api_key(env_name: str) -> str | None:
    """环境变量优先，其次 ~/.coda/.env。"""
    return os.environ.get(env_name) or _parse_env_file(coda_home() / ".env").get(env_name)
