"""权限确认弹层：y 允许一次 / a 本会话总是允许 / n 拒绝（可输入理由，回填给模型）/ Esc 中断本轮。

高危操作标红，且不提供"总是允许"。"总是允许"会显示具体加入的会话规则，例如 bash(uv run pytest*)。
"""

from __future__ import annotations

from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from coda.agent.events import Decision, PermissionRequest

KIND_LABEL = {"edit": "修改文件", "bash": "执行命令", "read": "读取"}


class PermissionScreen(ModalScreen[Decision | None]):
    BINDINGS = [
        Binding("y", "allow", "允许一次"),
        Binding("a", "always", "总是允许"),
        Binding("n", "deny", "拒绝"),
        Binding("escape", "interrupt", "中断"),
    ]

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__()
        self.request = request
        self._asking_reason = False

    def compose(self) -> ComposeResult:
        r = self.request
        title_color = "red" if r.danger else "yellow"
        keys = "[b]y[/] 允许一次   "
        if r.always:
            keys += f"[b]a[/] 本会话总是允许 [dim]{escape('、'.join(r.always))}[/]   "
        keys += "[b]n[/] 拒绝并说明理由   [b]Esc[/] 中断本轮"
        with Vertical(id="perm", classes="danger" if r.danger else ""):
            yield Static(
                Text.from_markup(
                    f"[b {title_color}]{'⚠ ' if r.danger else ''}需要确认：{escape(r.tool)}[/]  "
                    f"{KIND_LABEL.get(r.kind, '')} · {escape(r.reason)}"
                ),
                id="perm-title",
            )
            with VerticalScroll(id="perm-body"):
                if r.kind == "bash":
                    yield Static(Syntax(r.desc, "bash", background_color="default", word_wrap=True))
                else:
                    yield Static(Text(r.desc, style="bold"))
                if r.preview:
                    yield Static(Syntax(r.preview, "diff", background_color="default"))
            yield Static(Text.from_markup(keys), id="perm-keys")
            yield Input(placeholder="拒绝理由（可留空），Enter 确认", id="perm-reason")

    def on_mount(self) -> None:
        self.query_one("#perm-reason").display = False

    def action_allow(self) -> None:
        if not self._asking_reason:
            self.dismiss(Decision(True))

    def action_always(self) -> None:
        if not self._asking_reason and self.request.always:
            self.dismiss(Decision(True, always=True))

    def action_deny(self) -> None:
        if self._asking_reason:
            return
        self._asking_reason = True
        box = self.query_one("#perm-reason", Input)
        box.display = True
        box.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(Decision(False, reason=event.value.strip()))

    def action_interrupt(self) -> None:
        # 只设置取消标志；弹层由 TuiApprover 发现取消后关闭，避免重复 dismiss
        self.app.agent.interrupt()  # type: ignore[attr-defined]
