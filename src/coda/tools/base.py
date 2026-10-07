"""工具协议与注册表。

每个工具：
- 参数用 Pydantic 定义，同时生成发给模型的 JSON Schema；校验失败的字段路径直接回给模型。
- 返回 ToolResult(ok, content, error_type, hint)，发给模型时格式化成
  "Error[not_read]: ...\nHint: ..."，让模型能按错误类型决定下一步。
- kind 决定权限类别：read（自动）/ edit（按模式）/ bash（按命令）/ mcp（默认询问）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ValidationError

from coda.state.filestate import FileState

ToolKind = Literal["read", "edit", "bash", "mcp"]


class ErrorType(StrEnum):
    INVALID_ARGS = "invalid_args"
    NOT_FOUND = "not_found"
    NOT_READ = "not_read"
    STALE = "stale"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"
    INVALID_REGEX = "invalid_regex"
    DENIED = "denied"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"
    TOOL_ERROR = "tool_error"  # MCP 工具自己报告的错误


@dataclass
class ToolResult:
    ok: bool
    content: str
    error_type: ErrorType | None = None
    hint: str | None = None
    # 给界面看的附加信息，不发给模型：summary（一行摘要）、diff、output（折叠显示的输出）
    display: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def error(
        cls, error_type: ErrorType, message: str, hint: str | None = None, **display: Any
    ) -> ToolResult:
        return cls(False, message, error_type, hint, display)

    def to_text(self) -> str:
        if self.error_type is None:
            return self.content
        text = f"Error[{self.error_type}]: {self.content}"
        if self.hint:
            text += f"\nHint: {self.hint}"
        return text


@dataclass
class ToolContext:
    """工具运行时共享的状态。"""

    workdir: Path
    filestate: FileState
    cancel: threading.Event = field(default_factory=threading.Event)
    todos: list[dict[str, str]] = field(default_factory=list)  # todo_write 维护的任务清单

    def resolve(self, path: str) -> Path:
        """相对路径按工作区解析；先解析符号链接，再交给权限层判断是否在工作区内。"""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.workdir / p
        return p.resolve()

    def inside(self, p: Path) -> bool:
        return p == self.workdir or self.workdir in p.parents

    def rel(self, p: Path) -> str:
        """显示用路径：工作区内用相对路径。"""
        return str(p.relative_to(self.workdir)) if self.inside(p) else str(p)


class Tool:
    name: ClassVar[str]
    description: ClassVar[str]
    Params: ClassVar[type[BaseModel]]
    kind: ClassVar[ToolKind] = "read"

    @property
    def read_only(self) -> bool:
        return self.kind == "read"

    def schema(self) -> dict[str, Any]:
        params = self.Params.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    def validate(self, args: dict[str, Any]) -> BaseModel:
        return self.Params.model_validate(args)

    def describe(self, params: Any, ctx: ToolContext) -> str:
        """一行描述，显示在界面的工具行和权限弹层里。"""
        return ""

    def target_path(self, params: Any, ctx: ToolContext) -> Path | None:
        """文件类工具操作的路径，供权限层判断工作区边界。"""
        return None

    def preview(self, params: Any, ctx: ToolContext) -> str | None:
        """执行前的预览（编辑类返回 diff），显示在权限弹层里。"""
        return None

    def run(self, params: Any, ctx: ToolContext) -> ToolResult:
        raise NotImplementedError

    def run_call(self, params: Any, ctx: ToolContext, call_id: str) -> ToolResult:
        """执行器的入口。需要知道调用 id 的工具（task 要按 id 上报子智能体进度）覆盖这个方法。"""
        return self.run(params, ctx)


def format_validation_error(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "(参数)"
        parts.append(f"{loc}: {e['msg']}")
    return "；".join(parts)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def unregister_prefix(self, prefix: str) -> None:
        for name in [n for n in self._tools if n.startswith(prefix)]:
            del self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        # MCP 工具在后台线程里注册，先复制一份再遍历
        return [t.schema() for t in list(self._tools.values())]


def truncate_middle(text: str, limit: int) -> str:
    """超长文本保留头尾。工具结果的落盘在执行器里统一做（context/offload.py），这里只防内存失控。"""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n…（中间省略 {omitted} 个字符）…\n{text[-half:]}"
