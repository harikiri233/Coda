"""侧栏：任务清单、本会话改动的文件、上下文占用、用量与花费。"""

from __future__ import annotations

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import ProgressBar, Static

from coda.llm import Usage

TODO_STYLE = {
    "completed": ("[green]✔[/]", "dim"),
    "in_progress": ("[b yellow]◐[/]", "b"),
    "pending": ("[dim]☐[/]", ""),
}


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class Sidebar(VerticalScroll):
    def __init__(self, budget: int, micro: float = 0.6, summary: float = 0.85) -> None:
        super().__init__(id="side")
        self.budget = budget
        self.micro = micro
        self.summary = summary
        self.context_tokens = 0

    def compose(self) -> ComposeResult:
        yield Static("任务清单", classes="side-title", id="side-todo-title")
        yield Static(Text("（暂无）", style="dim"), id="side-todo")
        yield Static("本会话改动", classes="side-title")
        yield Static(Text("（暂无）", style="dim"), id="side-changes")
        yield Static("上下文", classes="side-title")
        yield ProgressBar(total=self.budget, show_eta=False, show_percentage=True, id="side-ctx")
        yield Static(self._ctx_text(), id="side-ctx-text")
        yield Static("用量", classes="side-title")
        yield Static(Text("尚未调用模型", style="dim"), id="side-usage")

    def update_todos(self, todos: list[dict[str, str]]) -> None:
        box = self.query_one("#side-todo", Static)
        title = self.query_one("#side-todo-title", Static)
        if not todos:
            title.update("任务清单")
            box.update(Text("（暂无）", style="dim"))
            return
        done = sum(t["status"] == "completed" for t in todos)
        title.update(f"任务清单 {done}/{len(todos)}")
        lines = []
        for t in todos:
            mark, style = TODO_STYLE[t["status"]]
            text = escape(t["content"])
            lines.append(f"{mark} [{style}]{text}[/]" if style else f"{mark} {text}")
        box.update(Text.from_markup("\n".join(lines)))

    def update_changes(self, stats: dict[str, tuple[int, int]]) -> None:
        box = self.query_one("#side-changes", Static)
        if not stats:
            box.update(Text("（暂无）", style="dim"))
            return
        lines = [f"{escape(p)}  [green]+{a}[/] [red]-{r}[/]" for p, (a, r) in stats.items()]
        lines.append("[dim]/undo 撤销上一轮 · /diff 查看[/]")
        box.update(Text.from_markup("\n".join(lines)))

    def _ctx_text(self) -> Text:
        used = self.context_tokens / self.budget if self.budget else 0
        color = "red" if used >= self.summary else ("yellow" if used >= self.micro else "dim")
        return Text.from_markup(
            f"[{color}]{_k(self.context_tokens)} / {_k(self.budget)}[/]\n"
            f"[dim]{self.micro:.0%} 微压缩 · {self.summary:.0%} 摘要 · /compact[/]"
        )

    def update_context(self, tokens: int, budget: int | None = None) -> None:
        if budget:
            self.budget = budget
        self.context_tokens = tokens
        bar = self.query_one("#side-ctx", ProgressBar)
        bar.update(total=self.budget, progress=min(tokens, self.budget))
        self.query_one("#side-ctx-text", Static).update(self._ctx_text())

    def update_usage(self, total: Usage, context_tokens: int, budget: int) -> None:
        self.update_context(context_tokens, budget)
        self.query_one("#side-usage", Static).update(
            Text.from_markup(
                f"输入 {_k(total.input_tokens)}  输出 {_k(total.output_tokens)}\n"
                f"缓存命中 [b green]{total.cache_hit_rate:.0%}[/]\n"
                f"花费 [b]${total.cost_usd:.4f}[/]"
            )
        )
