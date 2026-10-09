"""主循环对外的两个接口：EventSink（发事件）和 Approver（问权限）。

主循环不知道界面的存在：TUI 把事件转成组件，无头模式把事件打印或收集成 JSON，
测试里直接断言事件序列。同一套主循环支撑三种用法。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from coda.llm.usage import Usage

# ---------------------------------------------------------------- 事件


@dataclass
class TurnStart:
    user_input: str


@dataclass
class StepStart:
    step: int  # 从 1 开始


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class TextDelta:
    text: str


@dataclass
class AssistantEnd:
    """一次模型回复结束（无论是否带工具调用）。"""

    content: str
    reasoning: str
    tool_calls: int
    finish_reason: str | None
    cancelled: bool = False


@dataclass
class ToolStart:
    call_id: str
    name: str
    desc: str  # 一行描述：路径、命令或搜索模式
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolEnd:
    call_id: str
    name: str
    desc: str
    ok: bool
    error_type: str | None
    text: str  # 发给模型的完整文本
    display: dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0


@dataclass
class TodoUpdate:
    todos: list[
        dict[str, str]
    ]  # [{"content", "status"}]，status 为 pending / in_progress / completed


@dataclass
class FilesChanged:
    """文件被修改或撤销后发出，界面据此刷新侧栏的改动列表。"""

    stats: dict[str, tuple[int, int]]  # 相对路径 → (增, 删)


@dataclass
class UsageUpdate:
    last: Usage
    total: Usage
    context_tokens: int  # 最近一次主循环请求的输入 token
    context_budget: int


@dataclass
class Compacted:
    """上下文压缩完成。micro：旧工具结果换成占位符；summary：较早的历史换成摘要。"""

    kind: Literal["micro", "summary"]
    before: int  # 压缩前估算的上下文 token
    after: int
    detail: str  # 如 "清理 12 个工具结果" / "摘要了 48 条消息"
    reason: str = "auto"  # auto / manual / overflow


@dataclass
class SubagentUpdate:
    """子智能体的进度：每一步开始、每个工具结束时发出，界面更新折叠块标题和内容。"""

    call_id: str  # 父级 task 调用的 id
    step: int
    tokens: int
    tool: str | None = None  # 刚结束的工具：名字
    desc: str = ""
    ok: bool = True


@dataclass
class PermissionLog:
    """权限决定（只记需要询问或被拒绝的），写进会话 trace，界面不显示。"""

    tool: str
    desc: str
    decision: Literal["allowed", "always", "denied", "blocked", "interrupted"]
    reason: str = ""


@dataclass
class Notice:
    level: Literal["info", "warning", "error"]
    text: str


TurnStatus = Literal["done", "interrupted", "max_steps", "error"]


@dataclass
class TurnEnd:
    status: TurnStatus
    steps: int
    final_text: str = ""
    error: str | None = None


Event = (
    TurnStart
    | StepStart
    | ThinkingDelta
    | TextDelta
    | AssistantEnd
    | ToolStart
    | ToolEnd
    | TodoUpdate
    | FilesChanged
    | UsageUpdate
    | Compacted
    | SubagentUpdate
    | PermissionLog
    | Notice
    | TurnEnd
)


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class ListSink:
    """把事件收集到列表里，测试和无头 JSON 输出用。"""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._lock = threading.Lock()

    def emit(self, event: Event) -> None:
        with self._lock:
            self.events.append(event)

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, kind)]


# ---------------------------------------------------------------- 权限询问


@dataclass
class PermissionRequest:
    tool: str
    kind: str  # read / edit / bash
    desc: str
    reason: str  # 为什么需要询问
    preview: str | None = None  # 编辑类的 diff
    danger: bool = False  # 高危操作：弹层标红
    always: list[str] = field(
        default_factory=list
    )  # 选"总是允许"时加入的会话规则；为空则不提供该选项


@dataclass
class Decision:
    allow: bool
    always: bool = False  # 本会话总是允许
    reason: str = ""  # 拒绝理由，回填给模型


class Approver(Protocol):
    def ask(self, request: PermissionRequest, cancel: threading.Event) -> Decision: ...


class DenyApprover:
    """无头模式：没有人确认，一律拒绝，并把原因告诉模型。"""

    def ask(self, request: PermissionRequest, cancel: threading.Event) -> Decision:
        return Decision(
            False,
            reason="当前是无头模式，没有人可以确认这个操作。换一种不需要确认的做法，"
            "或在最终回答里说明需要用户用 --mode accept-edits 等方式放宽权限后重试。",
        )
