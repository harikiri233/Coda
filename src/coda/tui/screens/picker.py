"""通用列表选择弹层：会话选择（/resume、coda --resume）和模型选择（/model）共用。

↑↓ 选择 · Enter 确定 · Esc 取消；返回选中项的 id，取消返回 None。
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


class PickerScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(
        self, title: str, items: list[tuple[str, Text | str]], *, current: str | None = None
    ) -> None:
        """items：[(id, 显示内容)]。current：初始高亮的 id。"""
        super().__init__()
        self.title_text = title
        self.items = items
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Static(self.title_text, id="picker-title")
            yield OptionList(
                *[Option(label, id=key) for key, label in self.items], id="picker-list"
            )
            yield Static("↑↓ 选择 · Enter 确定 · Esc 取消", id="picker-keys")

    def on_mount(self) -> None:
        lst = self.query_one(OptionList)
        lst.focus()
        keys = [k for k, _ in self.items]
        if self.current in keys:
            lst.highlighted = keys.index(self.current)
        elif keys:
            lst.highlighted = 0

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)
