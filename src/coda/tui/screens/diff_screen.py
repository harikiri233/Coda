"""/diff：全屏查看本会话所有文件的累计改动，←→ 或 Tab 切换文件，Esc / q 关闭。"""

from __future__ import annotations

from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from coda.tools.files import diff_stats


class DiffScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape,q", "close", "关闭"),
        Binding("right,tab,n", "move(1)", "下一个文件"),
        Binding("left,shift+tab,p", "move(-1)", "上一个文件"),
    ]

    def __init__(self, diffs: dict[str, str]) -> None:
        super().__init__()
        self.diffs = diffs
        self.names = list(diffs)
        self.index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="diffview"):
            yield Static(id="diff-tabs")
            with VerticalScroll(id="diff-body"):
                yield Static(id="diff-content")
            yield Static(
                Text("←→ / Tab 切换文件 · ↑↓ 滚动 · Esc 关闭 · /undo 撤销上一轮", style="dim"),
                id="diff-keys",
            )

    def on_mount(self) -> None:
        self._show()
        self.query_one("#diff-body").focus()

    def _show(self) -> None:
        tabs = []
        for i, name in enumerate(self.names):
            a, r = diff_stats(self.diffs[name])
            label = f"{escape(name)} [green]+{a}[/] [red]-{r}[/]"
            tabs.append(f"[reverse] {label} [/]" if i == self.index else f" {label} ")
        self.query_one("#diff-tabs", Static).update(Text.from_markup("  ".join(tabs)))
        diff = self.diffs[self.names[self.index]].rstrip("\n")
        self.query_one("#diff-content", Static).update(
            Syntax(diff, "diff", background_color="default", line_numbers=False)
        )
        self.query_one("#diff-body", VerticalScroll).scroll_home(animate=False)

    def action_move(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.names)
        self._show()

    def action_close(self) -> None:
        self.dismiss(None)
