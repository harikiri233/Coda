"""多行输入框：Enter 发送，Ctrl+J（或 Shift+Enter）换行，输入框为空时 ↑ ↓ 翻历史输入。

补全列表（`/` 命令、`@` 文件）打开时：↑↓ 选择，Tab / Enter 填入，Esc 关闭。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import events
from textual.message import Message
from textual.widgets import TextArea

if TYPE_CHECKING:
    from coda.tui.widgets.completer import Completer


class PromptInput(TextArea):
    @dataclass
    class Submitted(Message):
        text: str

    def __init__(self, **kwargs) -> None:
        super().__init__(
            soft_wrap=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
            compact=True,
            placeholder="输入任务，Enter 发送；/ 查看命令，@ 引用文件",
            **kwargs,
        )
        self.input_history: list[str] = []
        self._hist_pos: int | None = None  # 正在浏览的历史下标
        self.completer: Completer | None = None

    def _browse(self, delta: int) -> None:
        if not self.input_history:
            return
        pos = len(self.input_history) if self._hist_pos is None else self._hist_pos
        pos = max(0, min(len(self.input_history), pos + delta))
        self._hist_pos = None if pos == len(self.input_history) else pos
        self.text = "" if self._hist_pos is None else self.input_history[pos]
        self.move_cursor(self.document.end)

    # ---- 补全 ----

    def _cursor_index(self) -> int:
        return self.document.get_index_from_location(self.cursor_location)

    def refresh_completion(self) -> None:
        if self.completer is not None:
            self.completer.update_for(self.text, self._cursor_index())

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self.refresh_completion()

    def _accept_completion(self) -> bool:
        """把选中的补全项填进输入框。返回是否填入了。"""
        c = self.completer
        if c is None or not c.active:
            return False
        choice = c.selected()
        kind = c.kind
        c.hide()
        if choice is None:
            return False
        if kind == "command":
            self.text = choice + " "
            self.move_cursor(self.document.end)
        else:
            cursor = self._cursor_index()
            text = self.text
            self.text = f"{text[: c.token_start]}@{choice} {text[cursor:]}"
            end = c.token_start + len(choice) + 2
            self.move_cursor(self.document.get_location_from_index(end))
        c.hide()
        return True

    async def _on_key(self, event: events.Key) -> None:
        key = event.key
        c = self.completer
        if c is not None and c.active:
            handled = True
            if key in ("up", "down"):
                c.move(-1 if key == "up" else 1)
            elif key == "tab":
                self._accept_completion()
            elif key == "enter":
                # 命令补全：选中后直接发送（不带参数的命令最常见）；文件补全：只填入
                if c.kind == "command":
                    choice = c.selected()
                    c.hide()
                    if choice:
                        self.text = choice
                    handled = False  # 继续走下面的发送逻辑
                else:
                    self._accept_completion()
            elif key == "escape":
                c.hide()
            else:
                handled = False
            if handled:
                event.stop()
                event.prevent_default()
                return
        if key == "enter":
            event.stop()
            event.prevent_default()
            text = self.text.strip()
            if text:
                if not self.input_history or self.input_history[-1] != text:
                    self.input_history.append(text)
                self._hist_pos = None
                self.text = ""
                if c is not None:
                    c.hide()
                self.post_message(self.Submitted(text))
            return
        if key in ("ctrl+j", "shift+enter"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        browsing = not self.text or self._hist_pos is not None
        if key in ("up", "down") and browsing and "\n" not in self.text:
            event.stop()
            event.prevent_default()
            self._browse(-1 if key == "up" else 1)
            return
        if event.is_printable:
            self._hist_pos = None
        await super()._on_key(event)
